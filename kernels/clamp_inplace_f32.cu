extern "C" __global__ void clamp_inplace_f32_kernel(
    float* __restrict__ x,
    const float lo,
    const float hi,
    const int n
) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float v = x[tid];
    if (v < lo) v = lo;
    else if (v > hi) v = hi;
    x[tid] = v;
}
