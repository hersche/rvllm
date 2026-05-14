// Transformer-XL-style relative-position shift for Gemma 4 audio
// chunked attention (matches Gemma4AudioAttention._rel_shift in HF).
//
// Input  matrix_bd [batch, heads, num_blocks, chunk_size, pos_len]   f32
// Output matrix_bd [batch, heads, num_blocks, chunk_size, context]   f32
//
// Per the HF reference:
//   pad x right to width `context + 1`
//   flatten the last two dims (chunk * (context+1))
//   keep the first chunk * context elements
//   view as (chunk, context)
//
// Equivalent direct mapping (no scratch buffer needed):
//   target_flat = r * context + c                     for output (r, c)
//   src_r = target_flat / (context + 1)
//   src_c = target_flat % (context + 1)
//   if src_c >= pos_len → output 0
//   else                → output input[src_r, src_c]
//
// One thread per output element. Grid covers
// batch * heads * num_blocks * chunk * context.

extern "C" __global__ void rel_shift_audio_f32_kernel(
    const float* __restrict__ src,         // [B, H, NB, chunk, pos_len]
    float*       __restrict__ dst,         // [B, H, NB, chunk, context]
    const int                  batch,
    const int                  heads,
    const int                  num_blocks,
    const int                  chunk,
    const int                  pos_len,
    const int                  context
) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long bhnb = (long long)batch * heads * num_blocks;
    const long long per_blk = (long long)chunk * context;
    const long long total = bhnb * per_blk;
    if (tid >= total) return;

    const long long blk_id = tid / per_blk;
    const long long blk_off = tid - blk_id * per_blk;
    const int r = (int)(blk_off / context);
    const int c = (int)(blk_off - (long long)r * context);

    const int target_flat = r * context + c;
    const int divisor = context + 1;
    const int src_r = target_flat / divisor;
    const int src_c = target_flat - src_r * divisor;

    float v = 0.0f;
    if (src_c < pos_len && src_r < chunk) {
        const long long src_blk_base = blk_id * (long long)chunk * pos_len;
        v = src[src_blk_base + (long long)src_r * pos_len + src_c];
    }
    dst[tid] = v;
}
