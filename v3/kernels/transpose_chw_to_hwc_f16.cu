// Transpose `[C, H, W] → [H, W, C]` for f16 tensors.
//
// Audio subsample exit path: after stage-1 Conv2d + LayerNorm + ReLU
// the tensor is `[C=32, H=T/4, W=32]` channel-first. HF then permutes
// to `[H, W, C]` and reshapes to `[H, W*C=1024]` for the
// input_proj_linear matmul. We do the permute explicitly so the next
// GEMM consumes a contiguous `[H, W*C]` layout.
//
// One thread per output element. Grid covers `H*W*C`.

#include <cuda_fp16.h>

extern "C"
__global__ void transpose_chw_to_hwc_f16_kernel(
    const __half* __restrict__ src,   // [C, H, W]
    __half*       __restrict__ dst,   // [H, W, C]
    const int                  C,
    const int                  H,
    const int                  W
) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)C * H * W;
    if (tid >= total) return;
    // Decode tid → (h, w, c) in dst.
    const long long hwc_per_h = (long long)W * C;
    const int h = (int)(tid / hwc_per_h);
    const long long rem_h = tid - (long long)h * hwc_per_h;
    const int w = (int)(rem_h / C);
    const int c = (int)(rem_h - (long long)w * C);

    const long long src_idx = (long long)c * H * W + (long long)h * W + w;
    dst[tid] = src[src_idx];
}
