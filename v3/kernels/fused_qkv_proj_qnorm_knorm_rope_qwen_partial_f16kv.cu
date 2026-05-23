// Qwen 3.6 full-attention megakernel — F16-KV path, WARP-COOPERATIVE.
//
// Replaces the Phase-2 naive 1-thread-per-output GEMV (commit 63ef718)
// with a warp-cooperative implementation: each WARP (32 threads)
// produces one output via 32-thread K-dim cooperation. This restores
// coalesced FP8 weight reads + halves the inner-loop instruction
// budget vs the naive variant (lane*8 stride 256 pattern from
// `fp8_gemv_blockwise_wpr_native_f16in_kernel`).
//
// On top of the warp-cooperative GEMV the kernel also stages the
// input row into shared memory once per block — the same hidden=2048
// row is reused across all per-head outputs (head_dim or head_dim*2
// of them), eliminating the n_dim-fold redundant load that the naive
// kernel paid through L1.
//
// Same fusions as Phase 2 (preserved):
//   * Q proj (FP8 GEMV)       : input → Q+gate [head, 2*head_dim] f16
//   * K proj (FP8 GEMV)       : input → K       [head, head_dim]   f16
//   * V proj (FP8 GEMV)       : input → V       [head, head_dim]   f16
//   * Q+gate split             : extract Q to q_out, gate to gate_out
//   * Q-norm + K-norm          : per-head RMSNorm with gamma
//   * Partial-NeoX RoPE        : rotate first rotary_dim of Q/K
//   * KV-cache write           : pack rotated K + passthrough V into
//                                key_cache / value_cache
//
// Numerical contract:
//   * Per output element, the FP8 dequant + scale + MAC matches
//     `fp8_gemv_blockwise_wpr_native_f16in_kernel` EXACTLY — same
//     8-elem lane-strided chunking, same fp8x2_to_f32 hw cvt
//     instruction, same per-block-128 scale apply, same warp
//     shuffle-down reduction order.
//   * Norm + RoPE phases byte-identical to
//     `fused_qnorm_knorm_rope_qwen_partial_f16kv_kernel` (Phase 1,
//     commit 943f8bb) given normalised inputs.
//
// Launch geometry (unchanged from the naive variant):
//   Grid:  (num_tokens, num_heads + 2*num_kv_heads, 1)
//   Block: (head_dim * 2, 1, 1)   // = 16 warps for head_dim=256
//   Shared: ≈ 4 KB input + 2 KB outputs + 0.5 KB partials = 6.5 KB
//
// block.y range:
//   [0, num_heads):                          → Q head
//   [num_heads, num_heads + num_kv_heads):   → K head
//   [num_heads + num_kv_heads, total):       → V head
//
// Phase 8 QKV megakernel Phase 2 (warp-cooperative, 2026-05-23).

#include <cuda_fp16.h>
#include <math.h>

#define QKV_MAX_HEAD_DIM_2X 512   // head_dim=256 → Q+gate 512 wide
#define QKV_MAX_HALF_HEAD   128   // half_head when head_dim=256
#define QKV_MAX_HIDDEN_U64  512   // hidden=2048 / 4 f16 per u64

// Identical to fp8_gemv.cu::fp8x2_to_f32 — hardware FP8x2 → f16x2
// then f16 → f32. 3 instructions per 2 FP8 elements on sm_121.
__device__ __forceinline__ void qkv_fp8x2_to_f32(unsigned short packed,
                                                  float& f0, float& f1) {
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(packed));
    unsigned short lo = (unsigned short)(f16x2);
    unsigned short hi = (unsigned short)(f16x2 >> 16);
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
}

__device__ __forceinline__ float qkv_fp8e4m3_to_float(unsigned char val) {
    unsigned int s = (val >> 7) & 1u;
    unsigned int e = (val >> 3) & 0xFu;
    unsigned int m = val & 0x7u;
    unsigned int f32_bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
    unsigned int is_normal = (e != 0u) & ((e != 0xFu) | (m != 0x7u));
    f32_bits &= (unsigned int)(-(int)is_normal);
    return __uint_as_float(f32_bits);
}

extern "C"
__global__ void fused_qkv_proj_qnorm_knorm_rope_qwen_partial_f16kv_kernel(
    const __half* __restrict__ input,                 // [num_tokens, hidden] f16
    const unsigned char* __restrict__ w_q,            // [num_heads*2*head_dim, hidden] fp8
    const unsigned char* __restrict__ w_k,            // [num_kv_heads*head_dim, hidden] fp8
    const unsigned char* __restrict__ w_v,            // [num_kv_heads*head_dim, hidden] fp8
    const float* __restrict__ scale_q,                // [num_heads*2*head_dim/128, hidden/128] f32
    const float* __restrict__ scale_k,                // [num_kv_heads*head_dim/128, hidden/128] f32
    const float* __restrict__ scale_v,                // [num_kv_heads*head_dim/128, hidden/128] f32
    __half* __restrict__ q_out,                       // [num_tokens, num_heads, head_dim] f16
    __half* __restrict__ gate_out,                    // [num_tokens, num_heads, head_dim] f16
    __half* __restrict__ key_cache,                   // [num_slots, num_kv_heads, head_dim] f16
    __half* __restrict__ value_cache,                 // [num_slots, num_kv_heads, head_dim] f16
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    const __half* __restrict__ q_norm_weight,         // [head_dim] f16
    const __half* __restrict__ k_norm_weight,         // [head_dim] f16
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int hidden,
    int rotary_dim,
    int num_col_blocks_q,                              // hidden / 128
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

    // ── Shared memory: input row, GEMV outputs, norm partial ────────
    // s_input is __half-typed but kept aligned for u64 strided loads
    // (the GEMV inner loop reads 4 f16 = 8 B via a single u64 ldg-
    // equivalent shared-mem load).
    __shared__ __align__(8) __half s_input[QKV_MAX_HIDDEN_U64 * 4];
    __shared__ float s_outputs[QKV_MAX_HEAD_DIM_2X];
    __shared__ float s_partial[QKV_MAX_HALF_HEAD];
    __shared__ float s_inv_norm;

    // ── Stage input row → shared mem (cooperative) ──────────────────
    // hidden is divisible by 4 for Qwen 3.6 (hidden=2048). Each
    // thread reads ONE u64 (4 f16) when within range.
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
    // Each warp produces n_dim/num_warps outputs sequentially. Lanes
    // 0..31 cooperate on the K-dim reduction for each output.
    for (int n_local = warp; n_local < n_dim; n_local += num_warps) {
        const int scale_row = n_local >> 7;
        const unsigned char* w_row =
            w_base + (long long)n_local * hidden;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        // Aligned main loop: each iter consumes 256 K elements
        // across the warp (32 lanes × 8 elements). Identical math
        // to fp8_gemv_blockwise_wpr_native_f16in_kernel.
        for (int k = lane * 8; k + 7 < hidden; k += 256) {
            unsigned long long w8 = __ldg(
                reinterpret_cast<const unsigned long long*>(w_row + k));

            // Shared-mem read (no __ldg — shared mem is not cached
            // by L1/L2; sm_121 shared-bank bandwidth handles 16
            // warps × 16 B reads/iter trivially).
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
            qkv_fp8x2_to_f32((unsigned short)(w8),       w0, w1);
            qkv_fp8x2_to_f32((unsigned short)(w8 >> 16), w2, w3);
            qkv_fp8x2_to_f32((unsigned short)(w8 >> 32), w4, w5);
            qkv_fp8x2_to_f32((unsigned short)(w8 >> 48), w6, w7);

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

        // Remainder loop covers K not divisible by 8 (Qwen hidden=
        // 2048 is divisible by 8 so this never executes for the
        // production model — kept for byte-identity with the
        // fp8_gemv_blockwise_wpr_native_f16in_kernel reference).
        {
            int aligned_k = (hidden / 8) * 8;
            for (int kr = aligned_k + lane; kr < hidden; kr += 32) {
                int sc = kr >> 7;
                float s = __ldg(&scale_base[(long long)scale_row
                                            * num_col_blocks + sc]);
                acc += qkv_fp8e4m3_to_float(__ldg(w_row + kr)) * s
                       * __half2float(s_input[kr]);
            }
        }

        // Warp shuffle-down reduction.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, off);
        }

        if (lane == 0) {
            s_outputs[n_local] = acc;
        }
    }
    __syncthreads();

    // ── V branch: write straight to value_cache and return ──────────
    if (is_v) {
        if (tid < head_dim) {
            const int slot = slot_mapping[token_idx];
            if (slot >= 0) {
                const int cache_off =
                    (slot * num_kv_heads + local_head) * head_dim;
                value_cache[cache_off + tid] = __float2half(s_outputs[tid]);
            }
        }
        return;
    }

    // ── Q / K: RMSNorm + RoPE ───────────────────────────────────────
    // Norm reduces over the head's first head_dim outputs only (for
    // Q heads the gate half — s_outputs[head_dim..head_dim_2x] —
    // is passthrough). Sum-of-squares is computed as a tree over
    // half_head paired elements (tid, tid + half_head).
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
        // Non-norm-participating threads still need to hit the same
        // sync count for warp-uniform progress through the reduce.
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

    // Phase 2: partial-NeoX RoPE for the rotary half of Q (or K).
    float final_val;
    if (tid < head_dim) {
        if (tid >= rotary_dim) {
            final_val = s_outputs[tid];
        } else {
            const bool is_lo = (tid < half_rot);
            const int  pair  = is_lo ? (tid + half_rot) : (tid - half_rot);
            const int  freq  = is_lo ? tid : pair;
            float self_v = s_outputs[tid];
            float pair_v = s_outputs[pair];
            float c = __half2float(cos_table[pos * half_rot + freq]);
            float s = __half2float(sin_table[pos * half_rot + freq]);
            final_val = is_lo ? (self_v * c - pair_v * s)
                              : (pair_v * s + self_v * c);
        }
    }

    if (is_q) {
        if (tid < head_dim) {
            const int q_dst = (token_idx * num_heads + local_head)
                              * head_dim + tid;
            q_out[q_dst] = __float2half(final_val);
        } else if (tid < head_dim_2x) {
            // Gate output: passthrough (no norm, no RoPE).
            const int gate_dst = (token_idx * num_heads + local_head)
                                 * head_dim + (tid - head_dim);
            gate_out[gate_dst] = __float2half(s_outputs[tid]);
        }
    } else if (is_k) {
        if (tid < head_dim) {
            const int slot = slot_mapping[token_idx];
            if (slot >= 0) {
                const int cache_off =
                    (slot * num_kv_heads + local_head) * head_dim;
                key_cache[cache_off + tid] = __float2half(final_val);
            }
        }
    }
}
