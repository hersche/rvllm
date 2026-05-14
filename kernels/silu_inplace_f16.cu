// Plain SiLU activation in-place on f16: x = x * sigmoid(x).
//
// Used by Gemma 4 audio FeedForward + LightConv1D (hidden_act="silu").
// One thread per element. Grid (ceil(N/BLOCK), 1, 1).

#include <cuda_fp16.h>

extern "C" __global__ void silu_inplace_f16_kernel(
    __half* __restrict__ x,
    const int            n
) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const float v = __half2float(x[tid]);
    const float sig = 1.0f / (1.0f + __expf(-v));
    x[tid] = __float2half(v * sig);
}
