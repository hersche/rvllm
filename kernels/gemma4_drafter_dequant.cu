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

extern "C" __global__ void gemma4_drafter_dequant_fp8_to_f16_kernel(
    const __nv_fp8_e4m3* __restrict__ src,
    __half*              __restrict__ dst,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // E4M3 → f32 via the device cast operator, then narrow to
        // f16. Matches `fp8e4m3_bytes_to_f16_bytes` in the loader
        // path so the on-disk → device round-trip is consistent.
        float v = (float)src[idx];
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
