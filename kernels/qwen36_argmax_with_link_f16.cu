// Qwen 3.6 fused argmax + step-link for the captured multi-step
// macro-graph. Replaces the (argmax_f16_kernel + qwen36_step_link
// _i32_kernel) pair that previously ran back-to-back at the tail of
// each non-final iteration in a macro-block.
//
// Behavior:
//   * Same f16 argmax reduction as argmax_f16_kernel: per-row
//     thread-local max followed by shared-memory tree reduction.
//   * Thread 0 of the (only) row's block ALSO performs the step-
//     link side-effects when `do_link != 0`:
//       - token_dst[0] = argmax_token_id
//       - pos_dst[0]   = pos_dst[0]  + 1
//       - ctx_dst[0]   = ctx_dst[0]  + 1
//   * Slot for the resulting argmax is `argmax_token_dst[0]`,
//     written by thread 0 unconditionally.
//
// The fused kernel cuts ONE launch per non-final iteration in the
// macro-graph (was: argmax + linker = 2 nodes per iter; now: 1).
// At N=8 macro-block: 7 launches saved → marginal latency win but
// real graph-node count reduction.
//
// `do_link == 0` mode is functionally identical to argmax_f16_kernel
// — used for the LAST iteration of a macro-block (no successor to
// link to). Same launch geometry as the standalone argmax:
//   Grid:  (1, 1, 1)        // single-row decoder argmax
//   Block: (min(vocab, 1024), 1, 1)

#include <float.h>
#include <cuda_fp16.h>

extern "C"
__global__ void qwen36_argmax_with_link_f16_kernel(
    const __half* __restrict__ logits,
    int* __restrict__ argmax_token_dst,   // i32 [1] — written unconditionally
    int* __restrict__ token_dst,          // i32 [1] — token_dev for next iter (NULL when do_link == 0)
    int* __restrict__ pos_dst,            // i32 [1] — pos_dev (NULL when do_link == 0)
    int* __restrict__ ctx_dst,            // i32 [1] — ctx_dev (NULL when do_link == 0)
    int vocab_size,
    int do_link                            // 0 = pure argmax (last iter); !=0 = also link
) {
    const int tid = threadIdx.x;
    const int n   = blockDim.x;
    const __half* x = logits;  // single-row decode → row 0 directly

    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];

    float local_max = -FLT_MAX;
    int   local_idx = 0;
    for (int i = tid; i < vocab_size; i += n) {
        float v = __half2float(x[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Same tree-reduction as argmax_f16_kernel. Power-of-two n
    // assumed for single-row decoder argmax (block = min(vocab,
    // 1024) where 1024 is power-of-two).
    for (int s = n / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        if (s * 2 < n && tid == 0) {
            if (s_val[s * 2] > s_val[0]) {
                s_val[0] = s_val[s * 2];
                s_idx[0] = s_idx[s * 2];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int tok = s_idx[0];
        argmax_token_dst[0] = tok;
        if (do_link != 0) {
            // Fused step-link: copy argmax → token_dev for the next
            // iteration's embed_gather + bump pos/ctx by 1 for the
            // next iteration's RoPE + paged-attn.
            token_dst[0] = tok;
            pos_dst[0]   = pos_dst[0] + 1;
            ctx_dst[0]   = ctx_dst[0] + 1;
        }
    }
}
