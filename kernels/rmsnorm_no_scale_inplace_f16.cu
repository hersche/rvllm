// Parameter-free RMSNorm (no gamma) in-place on f16 with f32 accumulator.
// Matches HF Gemma4RMSNorm(with_scale=False).
//
// Launch: Grid (num_rows, 1, 1), Block (min(dim, 1024), 1, 1).

#include <cuda_fp16.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_sum_n(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float block_sum_n(float v, float* smem) {
    const int wid = threadIdx.x / 32;
    const int lid = threadIdx.x & 31;
    v = warp_sum_n(v);
    if (lid == 0) smem[wid] = v;
    __syncthreads();
    const int num_w = (blockDim.x + 31) / 32;
    v = (lid < num_w) ? smem[lid] : 0.0f;
    if (wid == 0) v = warp_sum_n(v);
    return v;
}

extern "C" __global__ void __launch_bounds__(1024)
rmsnorm_no_scale_inplace_f16_kernel(
    __half* __restrict__ x,
    const float          eps,
    const int            hidden
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_off = row * hidden;

    __shared__ float smem[WARPS_MAX];
    __shared__ float row_rstd;

    float ss = 0.0f;
    for (int i = tid; i < hidden; i += stride) {
        float v = __half2float(x[row_off + i]);
        ss += v * v;
    }
    float total = block_sum_n(ss, smem);
    if (tid == 0) row_rstd = rsqrtf(total / (float)hidden + eps);
    __syncthreads();

    for (int i = tid; i < hidden; i += stride) {
        float v = __half2float(x[row_off + i]);
        x[row_off + i] = __float2half(v * row_rstd);
    }
}
