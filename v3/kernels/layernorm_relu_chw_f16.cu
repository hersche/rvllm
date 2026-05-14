// Fused LayerNorm + ReLU over the channel axis of a CHW tensor.
//
// Normalises and rectifies the output of the Gemma 4 audio
// SubSampleConvProjection's Conv2d layer along the `C` axis without
// any explicit permute — matches HF's
// `act(norm(x.permute(0,2,3,1)).permute(0,3,1,2))` math while staying
// in channel-first layout.
//
// Per-pixel math (one CUDA block per (h, w) coordinate):
//   x_norm[c] = (x[c, h, w] - mean_c(x)) / sqrt(var_c(x) + eps) * gamma[c]
//   x[c, h, w] = max(0, x_norm[c])
//
// gamma is the HF Conv2d layer's `norm.weight` (shape [C]); there is
// no `beta` since HF's `nn.LayerNorm(..., bias=False)`. We honour that
// by inlining a zero bias.
//
// Launch:
//   Grid:  (H * W, 1, 1)
//   Block: (C, 1, 1)                  C ≤ 1024 (E4B uses 128, 32)
//
// Buffer layout:
//   x      [C, H, W]  in-place           f16
//   gamma  [C]                            f16

#include <cuda_fp16.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();
    const int num_warps = (blockDim.x + 31) / 32;
    val = (lane < num_warps) ? smem[lane] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
layernorm_relu_chw_f16_kernel(
    __half*       __restrict__ x,        // [C, H, W] in-place
    const __half* __restrict__ gamma,    // [C]
    const float                eps,
    const int                  C,
    const int                  H,
    const int                  W
) {
    const int pix = blockIdx.x;                 // 0 .. H*W
    if (pix >= H * W) return;
    const int h = pix / W;
    const int w = pix - h * W;
    const int c = threadIdx.x;
    if (c >= C) return;

    // Stride math: channel-first contiguous along H*W within each
    // channel; element (c, h, w) lives at c*(H*W) + h*W + w.
    const long long base = (long long)h * W + w;
    const long long stride_c = (long long)H * W;
    const long long my_idx = (long long)c * stride_c + base;

    const float xc = __half2float(x[my_idx]);

    __shared__ float smem_mean[WARPS_MAX];
    __shared__ float smem_var[WARPS_MAX];
    __shared__ float row_mean;
    __shared__ float row_rstd;

    // Pass 1: mean of x_c across C.
    float sum = block_reduce_sum(xc, smem_mean);
    if (threadIdx.x == 0) row_mean = sum / (float)C;
    __syncthreads();

    // Pass 2: variance (Σ(x - μ)²) / C.
    const float d = xc - row_mean;
    float sq = block_reduce_sum(d * d, smem_var);
    if (threadIdx.x == 0) row_rstd = rsqrtf(sq / (float)C + eps);
    __syncthreads();

    const float g    = __half2float(gamma[c]);
    const float xn   = d * row_rstd * g;
    const float xact = xn > 0.0f ? xn : 0.0f;
    x[my_idx] = __float2half(xact);
}
