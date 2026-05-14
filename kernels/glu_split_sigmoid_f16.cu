// GLU split (sigmoid gate) for f16, matches PyTorch nn.functional.glu(x, dim=-1):
//   out[n, h] = x[n, h] * sigmoid(x[n, H + h])
//
// Input  : [N, 2H] f16
// Output : [N,  H] f16
//
// Used by Gemma 4 audio LightConv1D where the linear_start
// expansion 1024 -> 2048 is consumed via GLU (PyTorch sigmoid GLU,
// NOT SiLU GLU).
//
// One thread per output element. Grid (ceil(N*H/BLOCK), 1, 1).

#include <cuda_fp16.h>

extern "C" __global__ void glu_split_sigmoid_f16_kernel(
    const __half* __restrict__ src,   // [N, 2H]
    __half*       __restrict__ dst,   // [N, H]
    const int                  n,
    const int                  h_out
) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)n * h_out;
    if (tid >= total) return;
    const int row = (int)(tid / h_out);
    const int col = (int)(tid - (long long)row * h_out);
    const long long src_row = (long long)row * (2 * h_out);
    const float a = __half2float(src[src_row + col]);
    const float b = __half2float(src[src_row + h_out + col]);
    const float sig = 1.0f / (1.0f + __expf(-b));
    dst[tid] = __float2half(a * sig);
}
