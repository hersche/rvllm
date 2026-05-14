extern "C" __global__ void add_inplace_f32_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    const int n
) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    a[tid] += b[tid];
}
