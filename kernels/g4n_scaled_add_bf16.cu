// g4n_scaled_add_bf16: device-side `dst[i] += alpha * src[i]` for
// bf16 vectors with a bf16 scalar `alpha` on device.
//
// Background: Option B's post-MLP residual add is
//   residual_bf16 += layer_scalar_bf16 * mlp_normed_bf16
// where `layer_scalar` is a per-layer learned bf16 scalar
// (~0.09 on 31B layer 0; values stored on device by the loader
// at `layer.layer_scalar.offset_bytes`). The legacy host
// implementation cost one fence + DtoH(hidden*2 bytes) +
// host scale loop + DtoH(scalar, 2 bytes) + HtoD(hidden*2) per
// layer before invoking `vector_add_bf16`. Across 60 layers
// that's the last per-layer host-sync block on the Option B
// device-resident chain (codex Stream-5a step 2 left it as
// the only remaining fence).
//
// This kernel folds the scale + add into one pass. The
// caller passes the device pointer of the layer's
// `layer_scalar` (bf16 [1]); each thread broadcasts
// alpha to its lane and computes the fused multiply-add
// against `src[i]` into `dst[i]`.
//
// Math is done in f32 to avoid bf16 multiply round-off.
//
// Launch: grid (ceil(n/256), 1, 1), block (256, 1, 1).

#include <cuda_bf16.h>

extern "C" __launch_bounds__(256) __global__ void
g4n_scaled_add_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,    // [n] bf16 in-place
    const __nv_bfloat16* __restrict__ src,    // [n] bf16
    const __nv_bfloat16* __restrict__ alpha,  // [1] bf16 scalar
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // Single global read for alpha; every thread reads the same
    // address, which coalesces into one transaction at the warp
    // level. f32 math keeps the multiply from losing precision
    // before the add.
    float a = __bfloat162float(*alpha);
    float d = __bfloat162float(dst[idx]);
    float s = __bfloat162float(src[idx]);
    dst[idx] = __float2bfloat16(d + a * s);
}
