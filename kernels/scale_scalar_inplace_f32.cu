extern "C" __global__ void scale_scalar_inplace_f32_kernel(
    float* __restrict__ x,
    const float c,
    const int n
) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    x[tid] = x[tid] * c;
}
