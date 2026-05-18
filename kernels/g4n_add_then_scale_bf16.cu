// g4n_add_then_scale_bf16: device-side `dst[i] = (dst[i] + src[i]) * alpha`
// for bf16 vectors with a bf16 scalar `alpha` on device.
//
// HF Gemma 4 layer epilogue (modeling_gemma4.py:1399-1410):
//   hidden = residual + post_ff_norm(mlp)       (post-MLP add)
//   hidden = hidden * self.layer_scalar         (end-of-layer scale)
//
// Production's fused_norm_add_residual_*_kernel folds the
// RMSNorm + add + scale into one kernel. Option B uses
// separate kernels (rmsnorm_inplace_bf16_gbf16 then this), so
// this kernel only handles the (add + scale) tail.
//
// Note: this REPLACES `g4n_scaled_add_bf16` for the post-MLP
// residual add. The previous semantic
//   residual += layer_scalar * mlp_contrib
// did NOT match HF — HF applies layer_scalar to the WHOLE
// residual (residual + mlp_contrib), not just the contribution.
// The pre-fix Option-B forward was the mode-collapse smoking
// gun (codex Round 4, 2026-05-18): at L0/L1/L59 with
// layer_scalar 0.04-0.09 HF shrinks the residual 11-27× per
// such layer, Option B left it ~unchanged, so the residual
// stream drifted into a fixed-direction attractor across 60
// layers.
//
// Math is done in f32 to avoid bf16 multiply round-off, same
// as the sibling g4n_scaled_add_bf16.
//
// Launch: grid (ceil(n/256), 1, 1), block (256, 1, 1).

#include <cuda_bf16.h>

extern "C" __launch_bounds__(256) __global__ void
g4n_add_then_scale_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,    // [n] bf16 in-place
    const __nv_bfloat16* __restrict__ src,    // [n] bf16
    const __nv_bfloat16* __restrict__ alpha,  // [1] bf16 scalar
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float a = __bfloat162float(*alpha);
    float d = __bfloat162float(dst[idx]);
    float s = __bfloat162float(src[idx]);
    dst[idx] = __float2bfloat16((d + s) * a);
}
