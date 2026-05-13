// GELU(tanh)(gate) * per_layer_input -> F16 output (E4B PLE).
//
// Same math as `fused_gelu_mul_f16` but reads gate and the second
// operand from TWO SEPARATE pointers instead of a single
// gate-||-up concatenated buffer. Needed by the E4B Per-Layer
// Embeddings (PLE) injection at the end of each layer's forward:
//
//   gate     = hidden_states @ per_layer_input_gate.T   # [T, ple_dim]
//   per_li   = per_layer_inputs[:, L, :]                # [T, ple_dim]
//   gated    = GELU(tanh)(gate) * per_li                # this kernel
//
// Layout:
//   Both input buffers are row-major `[num_tokens, ple_dim]` f16.
//   The output buffer is row-major `[num_tokens, ple_dim]` f16.
//   No interleaving / concatenation — distinct device pointers.
//
// Launch:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(ple_dim, 1024), 1, 1)

#include <cuda_fp16.h>
#include <math.h>

extern "C" __global__ void __launch_bounds__(1024)
gelu_tanh_mul_dual_f16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ gate,
    const __half* __restrict__ per_li,
    int ple_dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const long long off = (long long)row * ple_dim;

    for (int i = tid; i < ple_dim; i += stride) {
        float g = __half2float(gate[off + i]);
        float p = __half2float(per_li[off + i]);
        // GELU(tanh) approximation, same as fused_gelu_mul_f16.
        float g3 = g * g * g;
        float inner = 0.7978845608f * (g + 0.044715f * g3);
        float gelu = 0.5f * g * (1.0f + tanhf(inner));
        output[off + i] = __float2half(gelu * p);
    }
}
