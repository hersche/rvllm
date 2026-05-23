// Qwen 3.6 partial-NeoX RoPE + NVFP4 paged-KV-cache write + FP8 Q,
// with Q-norm + K-norm FUSED inline. NVFP4 sibling of the F16-KV
// `fused_qnorm_knorm_rope_qwen_partial_f16kv_kernel` (commit
// 943f8bb).
//
// Folds the two standalone `rmsnorm_inplace_f16` launches (Q-norm
// + K-norm per full-attn layer) into the existing NVFP4 RoPE
// kernel via a 2-phase per-head body:
//
//   Phase 1: each thread (one per head_dim element) loads its
//   value, block-reduces sum-of-squares across head_dim, computes
//   inv_norm = rsqrtf(mean_sq + eps), applies gamma[tid] *
//   inv_norm and writes the normalised value to shared memory.
//   Phase 2 (after __syncthreads): the standard partial-NeoX
//   rotation reads two normalised values from shared memory
//   (self + pair via index tid ± half_rot), produces the rotated
//   f32, then the existing FP8-Q quantise / NVFP4-KV pack + scale
//   logic runs unchanged.
//
// Numerical contract:
//   * Phase 1 matches `rmsnorm_inplace_f16`'s algorithm (mean-of-
//     squares → rsqrtf → per-element gamma multiply).
//   * Phase 2 (rotation + FP8 Q quantise + NVFP4 K/V quantise &
//     pack) is byte-identical to `fused_rope_qwen_partial_nvfp4kv`
//     given the same (normalised) Q/K input.
//
// Same launch geometry as the unfused NVFP4 kernel:
//   Grid:  (num_tokens, max(num_heads, num_kv_heads), 1)
//   Block: (head_dim, 1, 1)  // one thread per element

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>
#include "../../kernels/nvfp4_utils.cuh"

using rvllm_nvfp4::fp4_encode;

__device__ __forceinline__ float qknr_block16_peak_abs(float v) {
    float a = fabsf(v);
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
    }
    return a;
}

extern "C"
__global__ void fused_qnorm_knorm_rope_qwen_partial_nvfp4kv_kernel(
    const __half* __restrict__ q_in,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    __nv_fp8_e4m3* __restrict__ q_fp8_out,
    uint8_t*       __restrict__ key_cache_packed,
    uint8_t*       __restrict__ value_cache_packed,
    __nv_fp8_e4m3* __restrict__ key_cache_scale,
    __nv_fp8_e4m3* __restrict__ value_cache_scale,
    const __half*  __restrict__ cos_table,
    const __half*  __restrict__ sin_table,
    const __half*  __restrict__ q_norm_weight,  // [head_dim] f16
    const __half*  __restrict__ k_norm_weight,  // [head_dim] f16
    const int*     __restrict__ positions,
    const int*     __restrict__ slot_mapping,
    const float*   __restrict__ q_scale_ptr,
    float*         __restrict__ q_scale_cache,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float eps
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    if (tid >= head_dim) return;

    // Shared mem partitioned:
    //   s_partial: head_dim f32 for sum-of-squares reduction
    //   s_normalized: head_dim f32 for the post-norm pre-rotation
    //                 values (so rotation can read self + pair).
    //   s_inv_norm: 1 f32 block-broadcast scalar.
    //   s_scratch: head_dim f32 — reused for the Q dynamic-scale
    //              amax reduction (unchanged behavior).
    __shared__ float s_partial[256];
    __shared__ float s_normalized[256];
    __shared__ float s_inv_norm;
    __shared__ float s_scratch[256];

    const int half_rot = rotary_dim >> 1;
    const int lane_in_group = tid & 15;
    const int pos = positions[token_idx];

    // ─── Q phase ──────────────────────────────────────────────────
    if (head_idx < num_heads) {
        const int q_base = (token_idx * num_heads + head_idx) * head_dim;

        // Phase 1: Q-norm.
        float my_q = __half2float(q_in[q_base + tid]);
        s_partial[tid] = my_q * my_q;
        __syncthreads();
        for (int s = head_dim / 2; s > 0; s >>= 1) {
            if (tid < s) s_partial[tid] += s_partial[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            float mean_sq = s_partial[0] / (float)head_dim;
            s_inv_norm = rsqrtf(mean_sq + eps);
        }
        __syncthreads();
        float inv = s_inv_norm;
        float g_q = __half2float(q_norm_weight[tid]);
        float qn = my_q * inv * g_q;
        s_normalized[tid] = qn;
        __syncthreads();

        // Phase 2: partial-NeoX RoPE on the normalised Q.
        float v;
        if (tid >= rotary_dim) {
            v = s_normalized[tid];
        } else {
            const bool is_lo = (tid < half_rot);
            const int  pair  = is_lo ? (tid + half_rot) : (tid - half_rot);
            const int  freq  = is_lo ? tid : pair;
            float self_v = s_normalized[tid];
            float pair_v = s_normalized[pair];
            float c = __half2float(cos_table[pos * half_rot + freq]);
            float s = __half2float(sin_table[pos * half_rot + freq]);
            v = is_lo ? (self_v * c - pair_v * s)
                      : (pair_v * s + self_v * c);
        }

        // FP8 Q quantise (dynamic or static scale) — unchanged from
        // the unfused kernel's Q branch.
        float q_scale_inv;
        if (q_scale_cache != nullptr) {
            float my_abs = fabsf(v);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                my_abs = fmaxf(my_abs,
                    __shfl_xor_sync(0xFFFFFFFFu, my_abs, off));
            }
            const int lane = tid & 31;
            const int warp = tid >> 5;
            if (lane == 0) s_scratch[warp] = my_abs;
            __syncthreads();
            const int num_warps = (head_dim + 31) >> 5;
            if (warp == 0) {
                float a = (lane < num_warps) ? s_scratch[lane] : 0.0f;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
                }
                if (lane == 0) {
                    float scale = fmaxf(a * (1.0f / 448.0f), 1e-12f);
                    s_scratch[0] = scale;
                    q_scale_cache[token_idx * num_heads + head_idx] = scale;
                }
            }
            __syncthreads();
            q_scale_inv = 1.0f / s_scratch[0];
        } else {
            q_scale_inv = 1.0f / (*q_scale_ptr);
        }
        q_fp8_out[q_base + tid] = __nv_fp8_e4m3(v * q_scale_inv);
    }
    __syncthreads();  // smem reuse between Q + K phases

    // ─── K, V phase ──────────────────────────────────────────────
    if (head_idx < num_kv_heads) {
        if (key_cache_packed == nullptr) return;
        const int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        const int slot   = slot_mapping[token_idx];
        if (slot < 0) return;

        // Phase 1: K-norm.
        float my_k = __half2float(k_in[k_base + tid]);
        s_partial[tid] = my_k * my_k;
        __syncthreads();
        for (int s = head_dim / 2; s > 0; s >>= 1) {
            if (tid < s) s_partial[tid] += s_partial[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            float mean_sq = s_partial[0] / (float)head_dim;
            s_inv_norm = rsqrtf(mean_sq + eps);
        }
        __syncthreads();
        float k_inv = s_inv_norm;
        float g_k = __half2float(k_norm_weight[tid]);
        float kn = my_k * k_inv * g_k;
        s_normalized[tid] = kn;
        __syncthreads();

        const int groups_per_head  = head_dim >> 4;
        const int cache_off_bytes  =
            (slot * num_kv_heads + head_idx) * (head_dim >> 1);
        const int cache_off_scales =
            (slot * num_kv_heads + head_idx) * groups_per_head;

        // Phase 2 + NVFP4 quantize+pack for K (RoPE'd) and V
        // (passthrough). The amax6 scale policy is preserved
        // verbatim from the unfused kernel.
        auto quant_and_write = [&](
            bool           apply_rope,
            const __half*  raw_in_for_v,   // only used when apply_rope==false (V passthrough)
            uint8_t*       out_packed,
            __nv_fp8_e4m3* out_scales
        ) {
            float v;
            if (apply_rope) {
                if (tid >= rotary_dim) {
                    v = s_normalized[tid];
                } else {
                    const bool is_lo = (tid < half_rot);
                    const int  pair  = is_lo ? (tid + half_rot) : (tid - half_rot);
                    const int  freq  = is_lo ? tid : pair;
                    float self_v = s_normalized[tid];
                    float pair_v = s_normalized[pair];
                    float c = __half2float(cos_table[pos * half_rot + freq]);
                    float s = __half2float(sin_table[pos * half_rot + freq]);
                    v = is_lo ? (self_v * c - pair_v * s)
                              : (pair_v * s + self_v * c);
                }
            } else {
                // V: no norm, no RoPE — read directly from v_in.
                v = __half2float(raw_in_for_v[k_base + tid]);
            }
            float peak = qknr_block16_peak_abs(v);
            float scale_f32 = fmaxf(peak * (1.0f / 6.0f), 1e-12f);
            float scale = float(__nv_fp8_e4m3(scale_f32));
            if (scale <= 0.0f) scale = 1e-12f;
            uint32_t nib = fp4_encode(v / scale);
            uint32_t partner_nib =
                __shfl_xor_sync(0xFFFFFFFFu, nib, 1);
            if ((tid & 1) == 0) {
                uint8_t byte = uint8_t((partner_nib & 0xFu) << 4
                                     | (nib & 0xFu));
                out_packed[cache_off_bytes + (tid >> 1)] = byte;
            }
            if (lane_in_group == 0) {
                int group_id = tid >> 4;
                out_scales[cache_off_scales + group_id] =
                    __nv_fp8_e4m3(scale_f32);
            }
        };

        quant_and_write(true,  k_in /*unused*/,           key_cache_packed,   key_cache_scale);
        quant_and_write(false, v_in,                       value_cache_packed, value_cache_scale);
    }
}
