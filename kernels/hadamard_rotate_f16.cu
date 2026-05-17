// Apply R = H · diag(D) to a per-(token, head) f16 vector in place.
//
// Mirror of hadamard_unrotate_f16.cu but in the FORWARD direction
// (rotation, not unrotation). Used for the Gemma 4 31B speculative-
// decoding drafter Q-side: when the base writes K/V to its NVFP4 KV
// cache with `RVLLM_NVFP4_HADAMARD=1`, base K (and Q at attention
// time) get rotated by R per-layer so that softmax(Q·K^T) is unchanged
// while the K distribution is reshaped for better NVFP4 quantization.
//
// The DRAFTER cross-attends to base's shadow K (= dequanted, STILL
// ROTATED). Drafter Q is in the natural (unrotated) frame and Q·K^T
// would be ≈ 0 (Q·(R·K)^T = Q·K^T·R^T which is uncorrelated unless
// we also rotate Q). Applying R to drafter Q after Q-norm + RoPE
// puts Q in the same rotated frame as the source-layer's K, so the
// inner product is preserved.
//
// R = H · diag(D). Same sign sequence as base K, so the kernel
// applies `apply_signs_f32 then fwht_inplace_f32` (signs FIRST,
// matching `fused_rope_partial_nvfp4kv.cu` line ~373 which does
// signs→fwht on K). The `unrotate` kernel reverses to fwht→signs.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "hadamard.cuh"

extern "C"
__global__ void hadamard_rotate_f16_kernel(
    __half*       __restrict__ x,            // [num_tokens, num_heads, head_dim] f16
    const signed char* __restrict__ signs,   // [head_dim] ±1 per channel (per-layer fixed)
    int num_tokens,
    int num_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    if (tid >= head_dim) return;
    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    __shared__ float s_buf[512];

    const int base = (token_idx * num_heads + head_idx) * head_dim;

    // Stage to f32 smem
    s_buf[tid] = __half2float(x[base + tid]);
    __syncthreads();

    // Apply R = H · diag(D). Same order as rope kernel:
    //   signs first (D ⊙), then FWHT (H ·).
    rvllm_hadamard::apply_signs_f32(s_buf, signs, head_dim);
    rvllm_hadamard::fwht_inplace_f32(s_buf, head_dim);

    // Write back as f16
    x[base + tid] = __float2half(s_buf[tid]);
}
