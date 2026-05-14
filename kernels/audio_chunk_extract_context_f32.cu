// Per-block context extraction for Gemma 4 audio chunked attention.
//
// Mirrors the HF math
//   K_ctx[b, blk, c, h, d] = K_padded[b, blk*chunk + c, h, d]
// where K_padded has `past_horizon` zero rows prepended and
// `future_horizon + chunk - 1` zero rows appended. Equivalent to
// PyTorch's `F.pad(...).unfold(1, context_size, chunk_size)` minus
// the explicit `movedim` permute (we lay the output out directly in
// the order [num_blocks, context_size, num_heads, head_dim] so the
// downstream batched-strided GEMM can consume it).
//
// Layout:
//   src [N_padded, num_heads, head_dim]  contiguous, channel-last
//                                         where N_padded =
//                                         past + N + future + slack
//   dst [num_blocks, context_size, num_heads, head_dim]  contiguous
//
// One thread per output element. Grid sized to
// `num_blocks * context_size * num_heads * head_dim`. Out-of-range
// reads short-circuit to zero (the caller must guarantee the
// `past_horizon` is already baked into `src_offset` via `src_base`).

extern "C" __global__ void audio_chunk_extract_context_f32_kernel(
    const float* __restrict__ src,         // [N_padded, H, D]
    float*       __restrict__ dst,         // [num_blocks, context, H, D]
    const int                  num_blocks,
    const int                  context,
    const int                  num_heads,
    const int                  head_dim,
    const int                  chunk_size,
    const int                  n_padded
) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)num_blocks * context * num_heads * head_dim;
    if (tid >= total) return;

    const long long per_head = (long long)head_dim;
    const long long per_pos  = per_head * num_heads;
    const long long per_blk  = per_pos * context;

    const long long blk = tid / per_blk;
    long long rem = tid - blk * per_blk;
    const long long ctx_c = rem / per_pos;
    rem -= ctx_c * per_pos;
    const long long h = rem / per_head;
    const long long d = rem - h * per_head;

    const long long src_pos = blk * chunk_size + ctx_c;
    float v = 0.0f;
    if (src_pos < n_padded) {
        v = src[src_pos * per_pos + h * per_head + d];
    }
    dst[tid] = v;
}
