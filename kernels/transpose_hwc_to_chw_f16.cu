// Transpose [H, W, C] -> [C, H, W] for f16 tensors.
// Inverse of transpose_chw_to_hwc_f16. Needed because f16_gemm_f32
// produces row-major [n, m] output: for the audio subsample conv
// call `(weight[out_ch, k], im2col[k, spatial])` with m=out_ch,
// n=spatial, the result is [spatial, out_ch] = HWC, but the
// downstream layernorm + im2col chain expects CHW.

#include <cuda_fp16.h>

extern "C" __global__ void transpose_hwc_to_chw_f16_kernel(
    const __half* __restrict__ src,   // [H, W, C]
    __half*       __restrict__ dst,   // [C, H, W]
    const int                  C,
    const int                  H,
    const int                  W
) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)C * H * W;
    if (tid >= total) return;
    const long long hw_per_c = (long long)H * W;
    const int c = (int)(tid / hw_per_c);
    const long long rem_c = tid - (long long)c * hw_per_c;
    const int h = (int)(rem_c / W);
    const int w = (int)(rem_c - (long long)h * W);

    const long long src_idx = (long long)h * W * C + (long long)w * C + c;
    dst[tid] = src[src_idx];
}
