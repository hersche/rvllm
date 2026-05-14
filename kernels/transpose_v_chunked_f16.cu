// V context transpose for Gemma 4 audio chunked attention.
//
// Input:  V_ctx [num_blocks, context, num_heads, head_dim]  f16
// Output: V_T   [num_blocks, num_heads, head_dim, context]  f16
//
// For each (blk, ctx, h, d): V_T[blk, h, d, ctx] = V_ctx[blk, ctx, h, d]
//
// Launch:
//   Grid:  (context * num_blocks, num_heads, 1)
//   Block: (head_dim, 1, 1)              — head_dim ≤ 1024

#include <cuda_fp16.h>

extern "C" __global__ void transpose_v_chunked_f16_kernel(
    __half* __restrict__ out,                // [B, H, D, C]
    const __half* __restrict__ in,           // [B, C, H, D]
    const int num_blocks,
    const int context,
    const int num_heads,
    const int head_dim
) {
    const int cb = blockIdx.x;                // 0..context*num_blocks
    const int h  = blockIdx.y;
    const int d  = threadIdx.x;
    if (d >= head_dim) return;
    const int blk = cb / context;
    const int ctx = cb - blk * context;
    if (blk >= num_blocks) return;

    const long long in_idx = ((long long)blk * context + ctx) * num_heads * head_dim
                           + (long long)h * head_dim + d;
    const long long out_idx = ((long long)blk * num_heads + h) * head_dim * context
                            + (long long)d * context + ctx;
    out[out_idx] = in[in_idx];
}
