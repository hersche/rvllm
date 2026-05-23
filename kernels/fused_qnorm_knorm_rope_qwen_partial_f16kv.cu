// Qwen 3.6 partial-NeoX RoPE + F16 paged-KV-cache write
// — with Q-norm + K-norm FUSED inline (Phase 1 of the
// QKV megakernel goal). Saves 2 launches per full-attn layer
// per token.
//
// Two-phase per-head structure:
//   Phase 1 (Q-norm or K-norm): each thread loads its two
//   half_head-paired elements, block-reduces sum-of-squares,
//   computes inv_norm = rsqrtf(mean_sq + eps), applies
//   gamma[d] * inv_norm to both elements, writes normalised
//   values back to the buffer in-place.
//   Phase 2 (RoPE + KV-write): __syncthreads, then for tid <
//   half_rot the thread reads the normalised rotary pair
//   (tid, tid + half_rot), rotates by (cos, sin) at the
//   token's position, and writes the rotated pair to q_out
//   (or key_cache for K). Threads with tid >= half_rot leave
//   the already-normalised non-rotary tail in place.
//
// Phase 1 covers ALL head_dim elements (each of half_head=64
// or 128 threads writes 2 elements), so the non-rotary tail
// [rotary_dim, head_dim) gets normalised even with the in-
// place pattern. The previous standalone Q-norm/K-norm
// rmsnorm_inplace launches did the same; this kernel just
// folds them into the existing RoPE+KV-write block.
//
// Numerical contract: Phase 1's RMSNorm matches
// rmsnorm_inplace_f16's algorithm (mean-of-squares → rsqrtf
// → per-element gamma). Phase 2's rotation + KV-cache write
// is byte-identical to fused_rope_qwen_partial_f16kv given
// the same (normalised) Q/K input.
//
// Launch (unchanged from the unfused RoPE kernel):
//   Grid:  (num_tokens, max(num_heads, num_kv_heads), 1)
//   Block: (head_dim/2, 1, 1)

#include <cuda_fp16.h>
#include <math.h>

// Shared-mem partial-sum array sized for the maximum block
// width we expect on Qwen 3.6 (head_dim = 256 → half_head = 128).
#define QKR_MAX_HALF_HEAD 128

extern "C"
__global__ void fused_qnorm_knorm_rope_qwen_partial_f16kv_kernel(
    const __half* __restrict__ q_in,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    __half* __restrict__ q_out,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    const __half* __restrict__ q_norm_weight,   // [head_dim] f16
    const __half* __restrict__ k_norm_weight,   // [head_dim] f16
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float eps
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    const int half_rot  = rotary_dim / 2;
    const int half_head = head_dim / 2;
    if (tid >= half_head) return;

    const int pos = positions[token_idx];

    __shared__ float s_partial[QKR_MAX_HALF_HEAD];
    __shared__ float s_inv_norm;

    // ── Q ──────────────────────────────────────────────────────────
    if (head_idx < num_heads) {
        const int q_base = (token_idx * num_heads + head_idx) * head_dim;

        // Phase 1: RMSNorm on Q[head_idx].
        float q_lo = __half2float(q_in[q_base + tid]);
        float q_hi = __half2float(q_in[q_base + tid + half_head]);
        s_partial[tid] = q_lo * q_lo + q_hi * q_hi;
        __syncthreads();
        for (int s = half_head / 2; s > 0; s >>= 1) {
            if (tid < s) s_partial[tid] += s_partial[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            float mean_sq = s_partial[0] / (float)head_dim;
            s_inv_norm = rsqrtf(mean_sq + eps);
        }
        __syncthreads();
        float inv = s_inv_norm;
        float g_lo = __half2float(q_norm_weight[tid]);
        float g_hi = __half2float(q_norm_weight[tid + half_head]);
        float n_lo = q_lo * inv * g_lo;
        float n_hi = q_hi * inv * g_hi;
        q_out[q_base + tid]             = __float2half(n_lo);
        q_out[q_base + tid + half_head] = __float2half(n_hi);
        __syncthreads();

        // Phase 2: partial-NeoX RoPE on the rotary half.
        if (tid < half_rot) {
            float c = __half2float(cos_table[pos * half_rot + tid]);
            float s = __half2float(sin_table[pos * half_rot + tid]);
            float r_lo = __half2float(q_out[q_base + tid]);
            float r_hi = __half2float(q_out[q_base + tid + half_rot]);
            q_out[q_base + tid]            = __float2half(r_lo * c - r_hi * s);
            q_out[q_base + tid + half_rot] = __float2half(r_lo * s + r_hi * c);
        }
        // tid >= half_rot: normalised non-rotary tail already
        // written in Phase 1c — nothing more to do.
    }

    // Ensure Q phase is fully done before reusing s_partial /
    // s_inv_norm for the K reduction (only one of Q-only / K-only
    // / Q+K branches executes per block thanks to the head_idx
    // guards, but the syncs here are cheap insurance).
    __syncthreads();

    // ── K + V ─────────────────────────────────────────────────────
    if (head_idx < num_kv_heads) {
        const int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        const int slot   = slot_mapping[token_idx];
        if (slot >= 0) {
            const int cache_off = (slot * num_kv_heads + head_idx) * head_dim;

            // Phase 1: RMSNorm on K[head_idx].
            float k_lo = __half2float(k_in[k_base + tid]);
            float k_hi = __half2float(k_in[k_base + tid + half_head]);
            s_partial[tid] = k_lo * k_lo + k_hi * k_hi;
            __syncthreads();
            for (int s = half_head / 2; s > 0; s >>= 1) {
                if (tid < s) s_partial[tid] += s_partial[tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                float mean_sq = s_partial[0] / (float)head_dim;
                s_inv_norm = rsqrtf(mean_sq + eps);
            }
            __syncthreads();
            float k_inv = s_inv_norm;
            float kg_lo = __half2float(k_norm_weight[tid]);
            float kg_hi = __half2float(k_norm_weight[tid + half_head]);
            float kn_lo = k_lo * k_inv * kg_lo;
            float kn_hi = k_hi * k_inv * kg_hi;

            // Phase 2: write rotated (or passthrough) K to
            // key_cache, and passthrough V to value_cache.
            if (tid < half_rot) {
                float c = __half2float(cos_table[pos * half_rot + tid]);
                float s = __half2float(sin_table[pos * half_rot + tid]);
                float partner = __half2float(k_in[k_base + tid + half_rot])
                              * k_inv
                              * __half2float(k_norm_weight[tid + half_rot]);
                key_cache[cache_off + tid]            = __float2half(kn_lo * c - partner * s);
                key_cache[cache_off + tid + half_rot] = __float2half(kn_lo * s + partner * c);
                // Non-rotary high pair for THIS thread's (tid +
                // half_head) — passthrough normalised. tid in
                // [0, half_rot) writes index tid + half_head,
                // covering K[half_head, half_head + half_rot).
                key_cache[cache_off + tid + half_head] = __float2half(kn_hi);
            } else {
                // Non-rotary tail: write normalised lo + hi for
                // this thread's (tid, tid + half_head) pair.
                key_cache[cache_off + tid]             = __float2half(kn_lo);
                key_cache[cache_off + tid + half_head] = __float2half(kn_hi);
            }
            // V passthrough — no norm, no RoPE.
            value_cache[cache_off + tid]             = v_in[k_base + tid];
            value_cache[cache_off + tid + half_head] = v_in[k_base + tid + half_head];
        }
    }
}
