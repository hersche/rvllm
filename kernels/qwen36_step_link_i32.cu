// qwen36_step_link_i32: device-side macro-step linker for the
// captured-graph multi-step replay (Phase 8 deeper optimization,
// 2026-05-23).
//
// Sits BETWEEN consecutive decode-step forward kernel sequences
// inside a multi-step captured graph. After iteration i's forward
// has written its argmax token to `argmax_token_src`, this kernel:
//
//   1. Copies the i32 token from `argmax_token_src` into `token_dst`
//      (= the shared workspace.token_dev slot the next iteration's
//      embed_gather reads).
//   2. Increments the shared `pos_dst` (workspace.pos_dev) by 1, so
//      the next iteration's RoPE + KV-slot kernels see position+1.
//   3. Increments the shared `ctx_dst` (workspace.ctx_dev) by 1, so
//      the next iteration's paged-attention sees context_len+1.
//
// All four pointers are device addresses to i32 [1] scalars; the
// kernel runs as a single-thread launch (grid=1, block=1), which is
// captured into the macro-graph alongside the surrounding forward
// kernels.
//
// Why a kernel (vs. cuMemsetD32 + cuMemcpyDtoD): a single fused
// kernel keeps the macro-graph compact (1 node per linker vs. 3),
// and cuMemcpyDtoD inside a captured graph is graph-safe but adds
// scheduler overhead. The kernel is trivial (3 device loads + 3
// device stores, plus 2 adds), so launch overhead dominates anyway.
//
// Stream-capture-safe: pure device-side ops, no host sync, no
// implicit allocations.

#include <cstdint>

extern "C" __global__ void __launch_bounds__(32)
qwen36_step_link_i32_kernel(
    const int* __restrict__ argmax_token_src,
    int* __restrict__       token_dst,
    int* __restrict__       pos_dst,
    int* __restrict__       ctx_dst
) {
    // Single-thread kernel — the surrounding kernels do the bulk of
    // the work; this is just a tiny stitching step between forward
    // iterations. `threadIdx.x == 0` is the only thread that runs.
    if (threadIdx.x != 0) return;
    int tok  = argmax_token_src[0];
    int pos  = pos_dst[0];
    int ctx  = ctx_dst[0];
    token_dst[0] = tok;
    pos_dst[0]   = pos + 1;
    ctx_dst[0]   = ctx + 1;
}
