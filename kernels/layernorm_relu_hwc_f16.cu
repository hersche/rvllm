// Fused LayerNorm + ReLU over the channel axis of an HWC f16 tensor.
// Matches the layout produced by f16_gemm_f32(W[out_ch, k], im2col[k, spatial])
// whose output is row-major [spatial, out_ch] = [H, W, C].
//
// Per-pixel math (one CUDA block per (h, w) coordinate):
//   x_norm[c] = (x[h, w, c] - mean_c(x)) / sqrt(var_c(x) + eps) * gamma[c]
//   x[h, w, c] = max(0, x_norm[c])
//
// gamma is the HF Conv2d layer's `norm.weight` (shape [C]); no bias
// (HF's nn.LayerNorm(..., bias=False)).
//
// Launch:
//   Grid:  (H * W, 1, 1)
//   Block: (C, 1, 1)                  C <= 1024

#include <cuda_fp16.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum_h(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float block_reduce_sum_h(float val, float* smem) {
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    val = warp_reduce_sum_h(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();
    const int num_warps = (blockDim.x + 31) / 32;
    val = (lane < num_warps) ? smem[lane] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum_h(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
layernorm_relu_hwc_f16_kernel(
    __half*       __restrict__ x,        // [H, W, C] in-place
    const __half* __restrict__ gamma,    // [C]
    const float                eps,
    const int                  C,
    const int                  H,
    const int                  W
) {
    const int pix = blockIdx.x;
    if (pix >= H * W) return;
    const int c = threadIdx.x;
    if (c >= C) return;

    const long long my_idx = (long long)pix * C + c;
    const float xc = __half2float(x[my_idx]);

    __shared__ float smem_a[WARPS_MAX];
    __shared__ float smem_b[WARPS_MAX];
    __shared__ float row_mean;
    __shared__ float row_rstd;

    float sum = block_reduce_sum_h(xc, smem_a);
    if (threadIdx.x == 0) row_mean = sum / (float)C;
    __syncthreads();

    const float d = xc - row_mean;
    float sq = block_reduce_sum_h(d * d, smem_b);
    if (threadIdx.x == 0) row_rstd = rsqrtf(sq / (float)C + eps);
    __syncthreads();

    const float g    = __half2float(gamma[c]);
    const float xn   = d * row_rstd * g;
    const float xact = xn > 0.0f ? xn : 0.0f;
    x[my_idx] = __float2half(xact);
}
