// Qwen 3.6 27B partial-NeoX RoPE + NVFP4 paged-KV-cache write + FP8 Q.
//
// Qwen-specific sibling of `fused_rope_partial_nvfp4kv.cu` (the Gemma 4
// version). Three differences from the Gemma kernel:
//
//   1. RoPE pairing. Qwen uses NeoX partial pairing
//      `(i, i + rotary_dim/2)` INSIDE the first rotary_dim elements
//      of each head; indices [rotary_dim, head_dim) pass through
//      untouched. Gemma's kernel pairs `(i, i + head_dim/2)` which is
//      only correct for full-rotary heads.
//   2. No Hadamard rotation. Qwen 3.6 27B doesn't use it.
//   3. No MSE scale-policy or debug sidecars. amax6 (peak/6) only.
//
// Inputs:
//   q_in : [num_tokens, num_heads,    head_dim] f16
//   k_in : [num_tokens, num_kv_heads, head_dim] f16
//   v_in : [num_tokens, num_kv_heads, head_dim] f16
//   cos_table, sin_table : [max_pos, rotary_dim/2] f16
//   positions    : [num_tokens] i32
//   slot_mapping : [num_tokens] i32 (-1 = skip)
//   q_scale_ptr  : [1] f32 (per-tensor static scale; only used when
//                  q_scale_cache is null)
//   q_scale_cache: [num_tokens, num_heads] f32 (per-(tok,head)
//                  dynamic scale; null -> static scalar)
//
// Outputs:
//   q_fp8_out          : [num_tokens, num_heads, head_dim] fp8_e4m3
//                        (rotated + quantized with 1/q_scale)
//   key_cache_packed   : [slot, num_kv_heads, head_dim/2]   u8 (packed nibbles)
//   value_cache_packed : same shape
//   key_cache_scale    : [slot, num_kv_heads, head_dim/16]  fp8_e4m3
//   value_cache_scale  : same shape
//
// Launch:
//   Grid:  (num_tokens, max(num_heads, num_kv_heads), 1)
//   Block: (head_dim, 1, 1)  // one thread per element
//
// head_dim must be a multiple of 16 (NVFP4 block size). Qwen 3.6 27B
// has head_dim=256.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "../../kernels/nvfp4_utils.cuh"

using rvllm_nvfp4::fp4_encode;

__device__ __forceinline__ float qwen_block16_peak_abs(float v) {
    float a = fabsf(v);
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
    }
    return a;
}

extern "C"
__global__ void fused_rope_qwen_partial_nvfp4kv_kernel(
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
    const int*     __restrict__ positions,
    const int*     __restrict__ slot_mapping,
    const float*   __restrict__ q_scale_ptr,
    float*         __restrict__ q_scale_cache,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    if (tid >= head_dim) return;

    __shared__ float s_scratch[256];   // head_dim <= 256 on Qwen 3.6 27B
    const int half_rot = rotary_dim >> 1;
    const int lane_in_group = tid & 15;
    const int pos = positions[token_idx];

    // Qwen partial NeoX RoPE: only the first `rotary_dim` elements
    // of each head rotate, paired (i, i + half_rot). The remaining
    // `head_dim - rotary_dim` indices pass through unchanged.
    auto rope_qwen = [&] (const __half* in, int base) -> float {
        if (tid >= rotary_dim) return __half2float(in[base + tid]);
        const bool is_lo = (tid < half_rot);
        const int  pair  = is_lo ? (tid + half_rot) : (tid - half_rot);
        const int  freq  = is_lo ? tid : pair;
        float self_v = __half2float(in[base + tid]);
        float pair_v = __half2float(in[base + pair]);
        float c = __half2float(cos_table[pos * half_rot + freq]);
        float s = __half2float(sin_table[pos * half_rot + freq]);
        return is_lo ? (self_v * c - pair_v * s)
                     : (pair_v * s + self_v * c);
    };

    // ─── Q: rotate -> Hadamard-skip -> FP8 quantize ────────────────
    if (head_idx < num_heads) {
        const int q_base = (token_idx * num_heads + head_idx) * head_dim;
        float v = rope_qwen(q_in, q_base);

        float q_scale_inv;
        if (q_scale_cache != nullptr) {
            // Dynamic Q scale: block-reduce amax across head_dim, derive
            // scale = max(amax / 448, 1e-12), broadcast via smem slot 0.
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
    __syncthreads();   // smem reuse between Q + K phases

    // ─── K, V: NVFP4-packed cache write ────────────────────────────
    if (head_idx < num_kv_heads) {
        if (key_cache_packed == nullptr) return;
        const int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        const int slot   = slot_mapping[token_idx];
        if (slot < 0) return;

        const int groups_per_head  = head_dim >> 4;
        const int cache_off_bytes  =
            (slot * num_kv_heads + head_idx) * (head_dim >> 1);
        const int cache_off_scales =
            (slot * num_kv_heads + head_idx) * groups_per_head;

        // Helper: quantize+pack one (K or V) head. K is RoPE'd, V is
        // pass-through. amax6 scale policy (range-preserving).
        auto quant_and_write = [&](
            const __half*  in,
            uint8_t*       out_packed,
            __nv_fp8_e4m3* out_scales,
            bool           apply_rope
        ) {
            float v = apply_rope ? rope_qwen(in, k_base)
                                 : __half2float(in[k_base + tid]);
            float peak = qwen_block16_peak_abs(v);
            float scale_f32 = fmaxf(peak * (1.0f / 6.0f), 1e-12f);
            // Round scale to E4M3 (same as production path) and back to
            // f32 for the divide.
            float scale = float(__nv_fp8_e4m3(scale_f32));
            if (scale <= 0.0f) scale = 1e-12f;
            uint32_t nib = fp4_encode(v / scale);
            // Pack two nibbles per byte. Lane-pair = (2k, 2k+1) within
            // each 16-lane group => byte at offset tid/2 of the group.
            // Use shfl to fetch neighbor's nibble.
            uint32_t partner_nib =
                __shfl_xor_sync(0xFFFFFFFFu, nib, 1);
            if ((tid & 1) == 0) {
                uint8_t byte = uint8_t((partner_nib & 0xFu) << 4
                                     | (nib & 0xFu));
                out_packed[cache_off_bytes + (tid >> 1)] = byte;
            }
            // Write one scale per 16-element group. Lane 0 of each group.
            if (lane_in_group == 0) {
                int group_id = tid >> 4;
                out_scales[cache_off_scales + group_id] =
                    __nv_fp8_e4m3(scale_f32);
            }
        };

        quant_and_write(k_in, key_cache_packed,   key_cache_scale,   true);
        quant_and_write(v_in, value_cache_packed, value_cache_scale, false);
    }
}
