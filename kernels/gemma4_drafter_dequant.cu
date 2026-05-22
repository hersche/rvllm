// Gemma 4 E4B assistant-drafter shadow-KV dequant kernels.
//
// Two elementwise kernels that mirror the base's source-layer K/V
// into the drafter's F16 shadow regions (allocated in
// `Gemma4DrafterRuntime::shadow_kv`). They exist so the assistant
// cross-attention launcher can stay on a single uniform F16-IO
// kernel (`flash_attention_2_decode_f16io_kernel`) regardless of
// the base's KV policy (F16 / FP8-E4M3 / NVFP4).
//
// * `gemma4_drafter_dequant_fp8_to_f16_kernel`
//     src: __nv_fp8_e4m3 [N]
//     dst: __half        [N]
//     Trivial elementwise widen via the device cast operator.
//
// * `gemma4_drafter_dequant_nvfp4_to_f16_kernel`
//     packed: uint8_t [N / 2]           — 4-bit NVFP4, 2 elems/byte
//     scales: __nv_fp8_e4m3 [N / 16]    — one E4M3 per 16-elem block
//     dst:    __half [N]
//     Decoder uses `rvllm_nvfp4::fp4_decode(nibble) * scale_block`
//     matching the production NVFP4 paged-decode kernel set so the
//     dequant output is bit-equivalent to what the FA-2 NVFP4
//     decode kernel sees internally.
//
// Both kernels launch with a 1-D block of 256 threads; grid covers
// the element count. Element counts up to ~33 M per (K or V) buffer
// at the E4B sliding source layer with `RVLLM_NUM_BLOCKS=1024`,
// `block_size=32`, `num_kv_heads=2`, `head_dim=256`. Memory-bound;
// roughly 32 MiB read + 32 MiB write per F16-out buffer.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include "nvfp4_utils.cuh"
#include "hadamard.cuh"

// Per-(slot, kv_head) scaled FP8 dequant.
//
// The base's FP8 KV cache is NOT a plain E4M3 byte stream. Each
// (slot, kv_head) pair carries its own f32 scale in a companion
// `k_scale_cache` / `v_scale_cache` buffer of shape
// `[num_slots * num_kv_heads]` (computed in
// `fused_rope_partial_fp8kv.cu` as `amax_of_that_head / 448`).
// To reconstruct: `f16 = decode_e4m3(byte) * scale[slot * nkvh + kv_head]`.
//
// Element layout in the K (or V) half:
//   `[num_slots, num_kv_heads, head_dim]` row-major, total `n` elems.
// For element `i`:
//   `slot_idx     = i / (nkvh * head_dim)`
//   `kv_head_idx  = (i / head_dim) % nkvh`
//   `scale_offset = slot_idx * nkvh + kv_head_idx`
extern "C" __global__ void gemma4_drafter_dequant_fp8_to_f16_kernel(
    const __nv_fp8_e4m3* __restrict__ src,
    const float*         __restrict__ scales,
    __half*              __restrict__ dst,
    int nkvh,
    int head_dim,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        long long elem_per_slot = (long long)nkvh * (long long)head_dim;
        long long slot_idx = idx / elem_per_slot;
        long long within = idx % elem_per_slot;
        long long kv_head = within / (long long)head_dim;
        float s = (scales != nullptr)
            ? scales[slot_idx * (long long)nkvh + kv_head]
            : 1.0f;
        float v = (float)src[idx] * s;
        dst[idx] = __float2half(v);
    }
}

extern "C" __global__ void gemma4_drafter_dequant_nvfp4_to_f16_kernel(
    const uint8_t*       __restrict__ packed,
    const __nv_fp8_e4m3* __restrict__ scales,
    __half*              __restrict__ dst,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        long long byte_idx = idx >> 1;
        long long scale_idx = idx >> 4;
        uint8_t byte = packed[byte_idx];
        uint8_t nib  = ((idx & 1LL) != 0)
                       ? (uint8_t)((byte >> 4) & 0xFu)
                       : (uint8_t)(byte & 0xFu);
        float s = (float)scales[scale_idx];
        float v = rvllm_nvfp4::fp4_decode((uint32_t)nib) * s;
        dst[idx] = __float2half(v);
    }
}

// =====================================================================
// Task #34 deferred fix — fused dequant + Hadamard un-rotate.
//
// The post-pass `hadamard_unrotate_f16_kernel` reads/writes the shadow
// region a second time after populate dequant. Fuse the two passes so
// the dequanted f16 row leaves the kernel ALREADY in HF-native frame:
//
//   1. dequant nibble (NVFP4) or E4M3 (FP8) → f32 in smem
//   2. FWHT on the head_dim vector (cooperative across threads)
//   3. apply signs ⊙ D (per-channel ±1 from `signs`)
//   4. write back f16
//
// Launch geometry: grid = (num_tokens, num_kv_heads), block = head_dim.
// head_dim must be a power of 2 (Gemma 4 uses 256 / 512); smem sized
// at head_dim * 4 bytes. Mirrors `hadamard_unrotate_f16_kernel`'s
// per-(token, head) decomposition but with the dequant fold-in.
//
// The fused path is opt-in via the host dispatcher — non-Hadamard or
// HADAMARD-off configs continue to use the plain elementwise kernels
// above, which stay bit-identical to today's production output.
// =====================================================================

extern "C" __global__ void gemma4_drafter_dequant_nvfp4_to_f16_unrotate_kernel(
    const uint8_t*       __restrict__ packed,     // [num_tokens, nkvh, head_dim/2] nibble-packed
    const __nv_fp8_e4m3* __restrict__ scales,     // [num_tokens, nkvh, head_dim/16] microscale
    const signed char*   __restrict__ signs,      // [head_dim] per-channel ±1
    __half*              __restrict__ dst,        // [num_tokens, nkvh, head_dim]
    int                                num_tokens,
    int                                nkvh,
    int                                head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    if (token_idx >= num_tokens || head_idx >= nkvh) return;
    if (tid >= head_dim) return;

    extern __shared__ float s_buf[];

    const long long row_base_elems =
        ((long long)token_idx * (long long)nkvh + (long long)head_idx) * (long long)head_dim;
    const long long elem_idx = row_base_elems + (long long)tid;

    // Dequant the tid-th nibble in the (token, head) row.
    long long byte_idx  = elem_idx >> 1;
    long long scale_idx = elem_idx >> 4;
    uint8_t byte = packed[byte_idx];
    uint8_t nib  = ((elem_idx & 1LL) != 0)
                   ? (uint8_t)((byte >> 4) & 0xFu)
                   : (uint8_t)(byte & 0xFu);
    float sf = (float)scales[scale_idx];
    s_buf[tid] = rvllm_nvfp4::fp4_decode((uint32_t)nib) * sf;
    __syncthreads();

    // Apply R^T = diag(D) · H: FWHT first, then sign flip — same
    // sequence as `hadamard_unrotate_f16_kernel`.
    rvllm_hadamard::fwht_inplace_f32(s_buf, head_dim);
    rvllm_hadamard::apply_signs_f32(s_buf, signs, head_dim);

    dst[elem_idx] = __float2half(s_buf[tid]);
}

extern "C" __global__ void gemma4_drafter_dequant_fp8_to_f16_unrotate_kernel(
    const __nv_fp8_e4m3* __restrict__ src,        // [num_tokens, nkvh, head_dim]
    const float*         __restrict__ scales,     // [num_tokens, nkvh] per-(slot, kv_head)
    const signed char*   __restrict__ signs,      // [head_dim] per-channel ±1
    __half*              __restrict__ dst,        // [num_tokens, nkvh, head_dim]
    int                                num_tokens,
    int                                nkvh,
    int                                head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    if (token_idx >= num_tokens || head_idx >= nkvh) return;
    if (tid >= head_dim) return;

    extern __shared__ float s_buf[];

    const long long row_base_elems =
        ((long long)token_idx * (long long)nkvh + (long long)head_idx) * (long long)head_dim;
    const long long elem_idx  = row_base_elems + (long long)tid;
    const long long scale_off = (long long)token_idx * (long long)nkvh + (long long)head_idx;

    float s = (scales != nullptr) ? scales[scale_off] : 1.0f;
    s_buf[tid] = (float)src[elem_idx] * s;
    __syncthreads();

    rvllm_hadamard::fwht_inplace_f32(s_buf, head_dim);
    rvllm_hadamard::apply_signs_f32(s_buf, signs, head_dim);

    dst[elem_idx] = __float2half(s_buf[tid]);
}
