// Gemma 4 audio chunked-attention mask: invalidate context positions
// whose source K row is outside [0, n_tokens).
//
// scores layout: [num_heads, num_blocks, chunk, context]  f32
// For each element at (h, blk, q, ctx):
//   src_pos = blk * chunk_size + ctx - past_horizon
//   valid   = src_pos in [0, n_tokens)
//   if invalid -> scores[...] = invalid_value
//
// One thread per element.

extern "C" __global__ void apply_audio_attn_mask_f32_kernel(
    float* __restrict__ scores,
    const int num_heads,
    const int num_blocks,
    const int chunk_size,
    const int context,
    const int past_horizon,
    const int n_tokens,
    const float invalid_value
) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long per_blk = (long long)chunk_size * context;
    const long long per_head = (long long)num_blocks * per_blk;
    const long long total = (long long)num_heads * per_head;
    if (tid >= total) return;
    const long long h = tid / per_head;
    long long rem = tid - h * per_head;
    const long long blk = rem / per_blk;
    rem -= blk * per_blk;
    const long long q = rem / context;       // unused but kept for clarity
    (void)q;
    const long long ctx = rem - q * context;
    const long long src_pos = blk * chunk_size + ctx - past_horizon;
    if (src_pos < 0 || src_pos >= n_tokens) {
        scores[tid] = invalid_value;
    }
}
