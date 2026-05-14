// Parameter-free RMSNorm in-place on an f32 [num_rows, hidden] tensor.
// Launch: Grid (num_rows, 1, 1), Block (min(hidden, 1024), 1, 1).

#define WARPS_MAX 32

__device__ __forceinline__ float warp_sum_o(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float block_sum_o(float v, float* smem) {
    const int wid = threadIdx.x / 32;
    const int lid = threadIdx.x & 31;
    v = warp_sum_o(v);
    if (lid == 0) smem[wid] = v;
    __syncthreads();
    const int num_w = (blockDim.x + 31) / 32;
    v = (lid < num_w) ? smem[lid] : 0.0f;
    if (wid == 0) v = warp_sum_o(v);
    return v;
}

extern "C" __global__ void __launch_bounds__(1024)
rmsnorm_no_scale_inplace_f32_kernel(
    float* __restrict__ x,
    const float          eps,
    const int            hidden
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const long long row_off = (long long)row * hidden;

    __shared__ float smem[WARPS_MAX];
    __shared__ float row_rstd;

    float ss = 0.0f;
    for (int i = tid; i < hidden; i += stride) {
        float v = x[row_off + i];
        ss += v * v;
    }
    float total = block_sum_o(ss, smem);
    if (tid == 0) row_rstd = rsqrtf(total / (float)hidden + eps);
    __syncthreads();

    for (int i = tid; i < hidden; i += stride) {
        x[row_off + i] = x[row_off + i] * row_rstd;
    }
}
