// Qwen 3.6 full-attention megakernel — NVFP4-KV path, WARP-COOPERATIVE.
//
// NVFP4-KV sibling of the F16-KV warp-cooperative QKV megakernel
// (commit fa48141). Folds the 5 unfused steps of the per-layer
// full-attn pipeline:
//   1. Q proj (FP8 GEMV)       : input → Q+gate [head, 2*head_dim] f16
//   2. K proj (FP8 GEMV)       : input → K       [head, head_dim]   f16
//   3. V proj (FP8 GEMV)       : input → V       [head, head_dim]   f16
//   4. Q+gate split             : extract Q to q_split, gate to gate
//   5. Q-norm + K-norm + RoPE   : per-head RMSNorm + partial-NeoX rotate
//      + Q FP8 quant
//      + K/V NVFP4 quant + pack into key/value caches with
//        per-16-element FP8-e4m3 microscales
// into ONE launch (same kernel symbol concept as the F16-KV variant).
//
// Numerical contract:
//   * Per output element of the GEMV phase, the FP8 dequant + scale
//     + MAC matches `fp8_gemv_blockwise_wpr_native_f16in_kernel`
//     EXACTLY (same 8-elem lane-strided pattern, fp8x2_to_f32 hw
//     cvt, blockwise scale per 128-K-block).
//   * Norm + RoPE + FP8-Q quant + K/V NVFP4 quant + pack matches
//     `fused_qnorm_knorm_rope_qwen_partial_nvfp4kv_kernel` (commit
//     6f6a25a) given the same normalised inputs.
//
// Block geometry (mirrors the F16-KV variant):
//   Grid:   (num_tokens, num_heads + 2*num_kv_heads, 1)
//   Block:  (head_dim * 2, 1, 1)   // 512 threads = 16 warps for head_dim=256
//   Shared: ~ 4 KB input + 2 KB outputs + 1 KB partials/scratch = ~7 KB
//
// block.y range:
//   [0, num_heads):                          → Q head
//   [num_heads, num_heads + num_kv_heads):   → K head
//   [num_heads + num_kv_heads, total):       → V head
//
// Phase 8 QKV megakernel NVFP4-KV sibling (2026-05-23).

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>
#include "../../kernels/nvfp4_utils.cuh"

using rvllm_nvfp4::fp4_encode;

#define QKVN_MAX_HEAD_DIM_2X 512   // head_dim=256 → Q+gate 512 wide
#define QKVN_MAX_HALF_HEAD   128   // half_head when head_dim=256
#define QKVN_MAX_HIDDEN_U64  512   // hidden=2048 / 4 f16 per u64

// Identical helpers to the F16 variant (fp8 hw cvt + scalar fallback).
__device__ __forceinline__ void qkvn_fp8x2_to_f32(unsigned short packed,
                                                   float& f0, float& f1) {
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(packed));
    unsigned short lo = (unsigned short)(f16x2);
    unsigned short hi = (unsigned short)(f16x2 >> 16);
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
}

__device__ __forceinline__ float qkvn_fp8e4m3_to_float(unsigned char val) {
    unsigned int s = (val >> 7) & 1u;
    unsigned int e = (val >> 3) & 0xFu;
    unsigned int m = val & 0x7u;
    unsigned int f32_bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
    unsigned int is_normal = (e != 0u) & ((e != 0xFu) | (m != 0x7u));
    f32_bits &= (unsigned int)(-(int)is_normal);
    return __uint_as_float(f32_bits);
}

// Same 16-element block peak reduction used by the NVFP4 RoPE kernel.
__device__ __forceinline__ float qkvn_block16_peak_abs(float v) {
    float a = fabsf(v);
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
    }
    return a;
}

extern "C"
__global__ void fused_qkv_proj_qnorm_knorm_rope_qwen_partial_nvfp4kv_kernel(
    const __half* __restrict__ input,
    const unsigned char* __restrict__ w_q,
    const unsigned char* __restrict__ w_k,
    const unsigned char* __restrict__ w_v,
    const float* __restrict__ scale_q,
    const float* __restrict__ scale_k,
    const float* __restrict__ scale_v,
    __nv_fp8_e4m3* __restrict__ q_fp8_out,             // [num_tokens, num_heads, head_dim]
    __half*        __restrict__ gate_out,              // [num_tokens, num_heads, head_dim] f16
    uint8_t*       __restrict__ key_cache_packed,      // [num_slots, num_kv_heads, head_dim/2]
    uint8_t*       __restrict__ value_cache_packed,    // [num_slots, num_kv_heads, head_dim/2]
    __nv_fp8_e4m3* __restrict__ key_cache_scale,       // [num_slots, num_kv_heads, head_dim/16]
    __nv_fp8_e4m3* __restrict__ value_cache_scale,     // [num_slots, num_kv_heads, head_dim/16]
    const __half*  __restrict__ cos_table,
    const __half*  __restrict__ sin_table,
    const __half*  __restrict__ q_norm_weight,
    const __half*  __restrict__ k_norm_weight,
    const int*     __restrict__ positions,
    const int*     __restrict__ slot_mapping,
    const float*   __restrict__ q_scale_ptr,           // static-Q-scale, used when q_scale_cache == nullptr
    float*         __restrict__ q_scale_cache,         // dynamic-Q-scale [num_tokens, num_heads]
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int hidden,
    int rotary_dim,
    int num_col_blocks_q,
    int num_col_blocks_kv,
    float eps
) {
    const int token_idx = blockIdx.x;
    const int head_global = blockIdx.y;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    const int half_rot  = rotary_dim / 2;
    const int half_head = head_dim / 2;
    const int head_dim_2x = head_dim * 2;
    const int lane_in_group = tid & 15;

    const int pos = positions[token_idx];

    // ── Projection-arm dispatch ─────────────────────────────────────
    const bool is_q = (head_global < num_heads);
    const bool is_k = (!is_q) && (head_global < num_heads + num_kv_heads);
    const bool is_v = (!is_q) && (!is_k);
    int local_head;
    const unsigned char* w_base;
    const float* scale_base;
    int n_dim;
    int num_col_blocks;
    if (is_q) {
        local_head = head_global;
        w_base = w_q + (long long)local_head * head_dim_2x * hidden;
        scale_base = scale_q + (long long)local_head * head_dim_2x
                              / 128 * num_col_blocks_q;
        n_dim = head_dim_2x;
        num_col_blocks = num_col_blocks_q;
    } else if (is_k) {
        local_head = head_global - num_heads;
        w_base = w_k + (long long)local_head * head_dim * hidden;
        scale_base = scale_k + (long long)local_head * head_dim
                              / 128 * num_col_blocks_kv;
        n_dim = head_dim;
        num_col_blocks = num_col_blocks_kv;
    } else {
        local_head = head_global - num_heads - num_kv_heads;
        w_base = w_v + (long long)local_head * head_dim * hidden;
        scale_base = scale_v + (long long)local_head * head_dim
                              / 128 * num_col_blocks_kv;
        n_dim = head_dim;
        num_col_blocks = num_col_blocks_kv;
    }

    // ── Shared mem ──────────────────────────────────────────────────
    __shared__ __align__(8) __half s_input[QKVN_MAX_HIDDEN_U64 * 4];
    __shared__ float s_outputs[QKVN_MAX_HEAD_DIM_2X];
    __shared__ float s_partial[QKVN_MAX_HALF_HEAD];
    __shared__ float s_inv_norm;
    __shared__ float s_scratch[QKVN_MAX_HALF_HEAD];  // Q FP8 dynamic-scale reduction

    // ── Stage input row → shared mem (cooperative) ──────────────────
    {
        const int hidden_u64 = hidden >> 2;
        const unsigned long long* x_row_u64 =
            reinterpret_cast<const unsigned long long*>(
                input + (long long)token_idx * hidden);
        unsigned long long* s_input_u64 =
            reinterpret_cast<unsigned long long*>(s_input);
        for (int i = tid; i < hidden_u64; i += blockDim.x) {
            s_input_u64[i] = __ldg(x_row_u64 + i);
        }
    }
    __syncthreads();

    // ── Phase 1: Warp-cooperative FP8 GEMV ──────────────────────────
    for (int n_local = warp; n_local < n_dim; n_local += num_warps) {
        const int scale_row = n_local >> 7;
        const unsigned char* w_row =
            w_base + (long long)n_local * hidden;

        float acc0 = 0.0f, acc1 = 0.0f;

        for (int k = lane * 8; k + 7 < hidden; k += 256) {
            unsigned long long w8 = __ldg(
                reinterpret_cast<const unsigned long long*>(w_row + k));
            unsigned long long x_lo = *reinterpret_cast<const unsigned long long*>(
                &s_input[k]);
            unsigned long long x_hi = *reinterpret_cast<const unsigned long long*>(
                &s_input[k + 4]);

            int sc0 = k >> 7;
            float s0 = __ldg(&scale_base[
                (long long)scale_row * num_col_blocks + sc0]);
            int sc4 = (k + 4) >> 7;
            float s4 = (sc4 != sc0)
                ? __ldg(&scale_base[(long long)scale_row * num_col_blocks + sc4])
                : s0;

            float w0, w1, w2, w3, w4, w5, w6, w7;
            qkvn_fp8x2_to_f32((unsigned short)(w8),       w0, w1);
            qkvn_fp8x2_to_f32((unsigned short)(w8 >> 16), w2, w3);
            qkvn_fp8x2_to_f32((unsigned short)(w8 >> 32), w4, w5);
            qkvn_fp8x2_to_f32((unsigned short)(w8 >> 48), w6, w7);

            float x0, x1, x2, x3, x4, x5, x6, x7;
            asm("cvt.f32.f16 %0, %1;" : "=f"(x0) : "h"((unsigned short)(x_lo)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x1) : "h"((unsigned short)(x_lo >> 16)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x2) : "h"((unsigned short)(x_lo >> 32)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x3) : "h"((unsigned short)(x_lo >> 48)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x4) : "h"((unsigned short)(x_hi)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x5) : "h"((unsigned short)(x_hi >> 16)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x6) : "h"((unsigned short)(x_hi >> 32)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x7) : "h"((unsigned short)(x_hi >> 48)));

            acc0 += w0 * s0 * x0;
            acc0 += w1 * s0 * x1;
            acc0 += w2 * s0 * x2;
            acc0 += w3 * s0 * x3;
            acc1 += w4 * s4 * x4;
            acc1 += w5 * s4 * x5;
            acc1 += w6 * s4 * x6;
            acc1 += w7 * s4 * x7;
        }

        float acc = acc0 + acc1;

        // Remainder (Qwen hidden=2048 → never executes; kept for parity).
        {
            int aligned_k = (hidden / 8) * 8;
            for (int kr = aligned_k + lane; kr < hidden; kr += 32) {
                int sc = kr >> 7;
                float s = __ldg(&scale_base[(long long)scale_row
                                            * num_col_blocks + sc]);
                acc += qkvn_fp8e4m3_to_float(__ldg(w_row + kr)) * s
                       * __half2float(s_input[kr]);
            }
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, off);
        }
        if (lane == 0) {
            s_outputs[n_local] = acc;
        }
    }
    __syncthreads();

    // ── V branch: NVFP4 quant + pack → value_cache ──────────────────
    if (is_v) {
        if (tid < head_dim) {
            const int slot = slot_mapping[token_idx];
            if (slot >= 0) {
                const int groups_per_head  = head_dim >> 4;
                const int cache_off_bytes  =
                    (slot * num_kv_heads + local_head) * (head_dim >> 1);
                const int cache_off_scales =
                    (slot * num_kv_heads + local_head) * groups_per_head;

                float v = s_outputs[tid];
                float peak = qkvn_block16_peak_abs(v);
                float scale_f32 = fmaxf(peak * (1.0f / 6.0f), 1e-12f);
                float scale = float(__nv_fp8_e4m3(scale_f32));
                if (scale <= 0.0f) scale = 1e-12f;
                uint32_t nib = fp4_encode(v / scale);
                uint32_t partner_nib = __shfl_xor_sync(0xFFFFFFFFu, nib, 1);
                if ((tid & 1) == 0) {
                    uint8_t byte = uint8_t((partner_nib & 0xFu) << 4
                                         | (nib & 0xFu));
                    value_cache_packed[cache_off_bytes + (tid >> 1)] = byte;
                }
                if (lane_in_group == 0) {
                    int group_id = tid >> 4;
                    value_cache_scale[cache_off_scales + group_id] =
                        __nv_fp8_e4m3(scale_f32);
                }
            }
        }
        return;
    }

    // ── Q / K: RMSNorm + RoPE ───────────────────────────────────────
    if (tid < half_head) {
        float lo = s_outputs[tid];
        float hi = s_outputs[tid + half_head];
        s_partial[tid] = lo * lo + hi * hi;
    }
    __syncthreads();
    if (tid < half_head) {
        for (int s = half_head / 2; s > 0; s >>= 1) {
            if (tid < s) s_partial[tid] += s_partial[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            float mean_sq = s_partial[0] / (float)head_dim;
            s_inv_norm = rsqrtf(mean_sq + eps);
        }
    } else {
        for (int s = half_head / 2; s > 0; s >>= 1) {
            __syncthreads();
        }
    }
    __syncthreads();
    float inv_norm = s_inv_norm;

    if (tid < head_dim) {
        const __half* gamma = is_q ? q_norm_weight : k_norm_weight;
        float g = __half2float(gamma[tid]);
        s_outputs[tid] = s_outputs[tid] * inv_norm * g;
    }
    __syncthreads();

    // Phase 2: partial-NeoX RoPE on the normalised Q (or K).
    float rotated;
    if (tid < head_dim) {
        if (tid >= rotary_dim) {
            rotated = s_outputs[tid];
        } else {
            const bool is_lo = (tid < half_rot);
            const int  pair  = is_lo ? (tid + half_rot) : (tid - half_rot);
            const int  freq  = is_lo ? tid : pair;
            float self_v = s_outputs[tid];
            float pair_v = s_outputs[pair];
            float c = __half2float(cos_table[pos * half_rot + freq]);
            float s = __half2float(sin_table[pos * half_rot + freq]);
            rotated = is_lo ? (self_v * c - pair_v * s)
                            : (pair_v * s + self_v * c);
        }
    }

    if (is_q) {
        // Q branch: FP8-quantise the rotated value + write gate
        // passthrough (no quant, no norm, no RoPE) to gate_out f16.
        float q_scale_inv;
        if (q_scale_cache != nullptr) {
            // Dynamic per-(token, head) Q-scale: block-reduce amax
            // across head_dim of the rotated Q values; only the first
            // head_dim threads participate.
            float my_abs = (tid < head_dim) ? fabsf(rotated) : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                my_abs = fmaxf(my_abs,
                    __shfl_xor_sync(0xFFFFFFFFu, my_abs, off));
            }
            const int my_warp = tid >> 5;
            const int my_lane = tid & 31;
            if (my_lane == 0 && tid < head_dim) {
                s_scratch[my_warp] = my_abs;
            }
            __syncthreads();
            const int num_warps_active = (head_dim + 31) >> 5;
            if (my_warp == 0) {
                float a = (my_lane < num_warps_active) ? s_scratch[my_lane] : 0.0f;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
                }
                if (my_lane == 0) {
                    float scale = fmaxf(a * (1.0f / 448.0f), 1e-12f);
                    s_scratch[0] = scale;
                    q_scale_cache[token_idx * num_heads + local_head] = scale;
                }
            }
            __syncthreads();
            q_scale_inv = 1.0f / s_scratch[0];
        } else {
            q_scale_inv = 1.0f / (*q_scale_ptr);
        }

        if (tid < head_dim) {
            const int q_dst = (token_idx * num_heads + local_head)
                              * head_dim + tid;
            q_fp8_out[q_dst] = __nv_fp8_e4m3(rotated * q_scale_inv);
        } else if (tid < head_dim_2x) {
            const int gate_dst = (token_idx * num_heads + local_head)
                                 * head_dim + (tid - head_dim);
            gate_out[gate_dst] = __float2half(s_outputs[tid]);
        }
    } else if (is_k) {
        // K branch: NVFP4 quant + pack rotated K into key_cache.
        if (tid < head_dim) {
            const int slot = slot_mapping[token_idx];
            if (slot >= 0) {
                const int groups_per_head  = head_dim >> 4;
                const int cache_off_bytes  =
                    (slot * num_kv_heads + local_head) * (head_dim >> 1);
                const int cache_off_scales =
                    (slot * num_kv_heads + local_head) * groups_per_head;

                float peak = qkvn_block16_peak_abs(rotated);
                float scale_f32 = fmaxf(peak * (1.0f / 6.0f), 1e-12f);
                float scale = float(__nv_fp8_e4m3(scale_f32));
                if (scale <= 0.0f) scale = 1e-12f;
                uint32_t nib = fp4_encode(rotated / scale);
                uint32_t partner_nib = __shfl_xor_sync(0xFFFFFFFFu, nib, 1);
                if ((tid & 1) == 0) {
                    uint8_t byte = uint8_t((partner_nib & 0xFu) << 4
                                         | (nib & 0xFu));
                    key_cache_packed[cache_off_bytes + (tid >> 1)] = byte;
                }
                if (lane_in_group == 0) {
                    int group_id = tid >> 4;
                    key_cache_scale[cache_off_scales + group_id] =
                        __nv_fp8_e4m3(scale_f32);
                }
            }
        }
    }
}
