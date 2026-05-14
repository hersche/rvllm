// Add an f16 bias vector to each row of an f32 tensor in place.
// Grid: (num_tokens, 1, 1). Block: (min(dim, 1024), 1, 1).
#include <cuda_fp16.h>

extern "C" __global__ void add_bias_f16_to_f32_kernel(
    float* __restrict__ tensor,         // [num_tokens, dim], f32, in-place
    const __half* __restrict__ bias,    // [dim], f16
    int dim
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const long long row_off = (long long)t * dim;
    for (int i = tid; i < dim; i += stride) {
        const float b = __half2float(bias[i]);
        tensor[row_off + i] += b;
    }
}
