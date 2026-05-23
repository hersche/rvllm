// Qwen 3.6 full-attention megakernel — F16-KV path, naive
// 1-thread-per-output GEMV. Fuses:
//   * Q proj (FP8 GEMV)       : input → Q+gate [head, 2*head_dim] f16
//   * K proj (FP8 GEMV)       : input → K       [head, head_dim]   f16
//   * V proj (FP8 GEMV)       : input → V       [head, head_dim]   f16
//   * Q+gate split             : extract Q to q_out, gate to gate_out
//   * Q-norm + K-norm          : per-head RMSNorm with gamma
//   * Partial-NeoX RoPE        : rotate first rotary_dim of Q/K
//   * KV-cache write           : pack rotated K + passthrough V into
//                                key_cache / value_cache
// in ONE launch (vs the current 5 launches per full-attn layer).
//
// Block geometry: (head_dim_2x, 1, 1) where head_dim_2x = 2 *
// head_dim — covers Q+gate for Q heads; K/V heads use the first
// head_dim threads, the rest idle.
// Grid: (num_tokens, num_heads + num_kv_heads + num_kv_heads, 1).
//   block.y in [0, num_heads):                          → Q head
//   block.y in [num_heads, num_heads + num_kv_heads):   → K head
//   block.y in [num_heads + num_kv_heads, total):       → V head
//
// PERF CAVEAT: this naive variant has each thread compute a single
// output element via a SEQUENTIAL K-dim reduction across hidden
// (each thread reads its own row of W). Row reads at a fixed k
// across the warp are STRIDED (stride = K bytes), so the FP8
// weight reads are uncoalesced — slower than the warp-cooperative
// fp8_gemv pattern. A warp-cooperative re-implementation is the
// next perf optimization step. Operator-opt-in via env so default
// production paths stay unaffected.
//
// Numerical contract: per output element, the FP8 dequant + scale
// + MAC is byte-identical to the existing
// fp8_gemv_blockwise_wpr_native_f16in_kernel. The norm + RoPE
// phases are byte-identical to fused_qnorm_knorm_rope_qwen_partial
// _f16kv (commit 943f8bb).
//
// Phase 8 QKV megakernel Phase 2 (2026-05-23, F16-KV naive).

#include <cuda_fp16.h>
#include <math.h>

#define QKV_MAX_HEAD_DIM_2X 512   // covers head_dim=256 (Q has 2x for Q+gate)
#define QKV_MAX_HALF_HEAD   128   // half_head when head_dim=256

__device__ __forceinline__ float qkv_fp8e4m3_to_float(unsigned char val) {
    unsigned int s = (val >> 7) & 1u;
    unsigned int e = (val >> 3) & 0xFu;
    unsigned int m = val & 0x7u;
    unsigned int f32_bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
    unsigned int is_normal = (e != 0u) & ((e != 0xFu) | (m != 0x7u));
    f32_bits &= (unsigned int)(-(int)is_normal);
    return __uint_as_float(f32_bits);
}

extern "C"
__global__ void fused_qkv_proj_qnorm_knorm_rope_qwen_partial_f16kv_kernel(
    const __half* __restrict__ input,                 // [num_tokens, hidden] f16
    const unsigned char* __restrict__ w_q,            // [num_heads*2*head_dim, hidden] fp8
    const unsigned char* __restrict__ w_k,            // [num_kv_heads*head_dim, hidden] fp8
    const unsigned char* __restrict__ w_v,            // [num_kv_heads*head_dim, hidden] fp8
    const float* __restrict__ scale_q,                // [num_heads*2*head_dim/128, hidden/128] f32
    const float* __restrict__ scale_k,                // [num_kv_heads*head_dim/128, hidden/128] f32
    const float* __restrict__ scale_v,                // [num_kv_heads*head_dim/128, hidden/128] f32
    __half* __restrict__ q_out,                       // [num_tokens, num_heads, head_dim] f16
    __half* __restrict__ gate_out,                    // [num_tokens, num_heads, head_dim] f16
    __half* __restrict__ key_cache,                   // [num_slots, num_kv_heads, head_dim] f16
    __half* __restrict__ value_cache,                 // [num_slots, num_kv_heads, head_dim] f16
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    const __half* __restrict__ q_norm_weight,         // [head_dim] f16
    const __half* __restrict__ k_norm_weight,         // [head_dim] f16
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int hidden,
    int rotary_dim,
    int num_col_blocks_q,                              // hidden / 128
    int num_col_blocks_kv,
    float eps
) {
    const int token_idx = blockIdx.x;
    const int head_global = blockIdx.y;
    const int tid = threadIdx.x;

    const int half_rot  = rotary_dim / 2;
    const int half_head = head_dim / 2;
    const int head_dim_2x = head_dim * 2;
    if (tid >= head_dim_2x) return;

    const int pos = positions[token_idx];

    __shared__ float s_partial[QKV_MAX_HALF_HEAD];
    __shared__ float s_inv_norm;
    // s_outputs[0..head_dim) holds normalized Q (or K). Indices
    // [head_dim, 2*head_dim) hold gate (Q heads only).
    __shared__ float s_outputs[QKV_MAX_HEAD_DIM_2X];

    // ── Determine which projection this block handles ──────────────
    const bool is_q = (head_global < num_heads);
    const bool is_k = (!is_q) && (head_global < num_heads + num_kv_heads);
    const bool is_v = (!is_q) && (!is_k);
    int local_head;
    const unsigned char* w_base;
    const float* scale_base;
    int n_dim;                  // per-head output dim (head_dim_2x for Q, head_dim for K/V)
    int row_stride_bytes;       // weight row stride in bytes (= hidden)
    int num_col_blocks;
    if (is_q) {
        local_head = head_global;
        w_base = w_q + (long long)local_head * head_dim_2x * hidden;
        scale_base = scale_q + (long long)local_head * head_dim_2x
                              * num_col_blocks_q / 128;
        n_dim = head_dim_2x;
        row_stride_bytes = hidden;
        num_col_blocks = num_col_blocks_q;
    } else if (is_k) {
        local_head = head_global - num_heads;
        w_base = w_k + (long long)local_head * head_dim * hidden;
        scale_base = scale_k + (long long)local_head * head_dim
                              * num_col_blocks_kv / 128;
        n_dim = head_dim;
        row_stride_bytes = hidden;
        num_col_blocks = num_col_blocks_kv;
    } else {
        local_head = head_global - num_heads - num_kv_heads;
        w_base = w_v + (long long)local_head * head_dim * hidden;
        scale_base = scale_v + (long long)local_head * head_dim
                              * num_col_blocks_kv / 128;
        n_dim = head_dim;
        row_stride_bytes = hidden;
        num_col_blocks = num_col_blocks_kv;
    }

    // ── Phase 1: FP8 GEMV — each thread computes output[tid] ───────
    float my_out = 0.0f;
    if (tid < n_dim) {
        // Per blockwise-FP8 layout: scale_base is indexed as
        // [n_block, k_block] in the per-head scale matrix where
        // n_block = tid / 128 and k_block = k / 128. The
        // existing fp8_gemv_blockwise_wpr_native_f16in_kernel
        // applies scale[scale_row * num_col_blocks + sc] per
        // K-block. scale_row = n_block = tid / 128.
        const int scale_row = tid / 128;
        const unsigned char* w_row =
            w_base + (long long)tid * row_stride_bytes;
        const __half* x_row =
            input + (long long)token_idx * hidden;

        for (int k = 0; k < hidden; k++) {
            int sc_block = k / 128;
            float s = scale_base[(long long)scale_row * num_col_blocks + sc_block];
            float w = qkv_fp8e4m3_to_float(w_row[k]);
            float x = __half2float(x_row[k]);
            my_out += w * s * x;
        }
    }

    // Stash output into shared mem for the rest of the chain.
    if (tid < n_dim) {
        s_outputs[tid] = my_out;
    }
    __syncthreads();

    // ── V: no norm, no RoPE; write directly to value_cache ─────────
    if (is_v) {
        if (tid < head_dim) {
            const int slot = slot_mapping[token_idx];
            if (slot >= 0) {
                const int cache_off =
                    (slot * num_kv_heads + local_head) * head_dim;
                value_cache[cache_off + tid] = __float2half(my_out);
            }
        }
        return;
    }

    // ── Q+gate or K: norm/RoPE/write ───────────────────────────────
    // For Q: threads 0..head_dim-1 handle Q (norm+RoPE), threads
    // head_dim..2*head_dim-1 handle gate (passthrough).
    // For K: threads 0..head_dim-1 handle K. Threads head_dim..2*head_dim
    // idle.

    // RMSNorm over the head's first head_dim outputs (Q or K).
    // Each thread covers 2 elements via (tid, tid+half_head) within
    // the head's first head_dim. tid runs 0..half_head for this.
    if (tid < half_head) {
        float lo = s_outputs[tid];
        float hi = s_outputs[tid + half_head];
        s_partial[tid] = lo * lo + hi * hi;
    } else if (tid < head_dim) {
        // 0 partial — already covered by the tid<half_head pair.
    }
    __syncthreads();

    if (tid < half_head) {
        // Tree reduce over half_head elements.
        for (int s = half_head / 2; s > 0; s >>= 1) {
            if (tid < s) s_partial[tid] += s_partial[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            float mean_sq = s_partial[0] / (float)head_dim;
            s_inv_norm = rsqrtf(mean_sq + eps);
        }
    } else {
        // Non-norm-participating threads still hit the same sync
        // count.
        for (int s = half_head / 2; s > 0; s >>= 1) {
            __syncthreads();
        }
    }
    __syncthreads();
    float inv_norm = s_inv_norm;

    // Apply gamma * inv_norm to Q (or K) outputs in shared mem.
    if (tid < head_dim) {
        const __half* gamma = is_q ? q_norm_weight : k_norm_weight;
        float g = __half2float(gamma[tid]);
        s_outputs[tid] = s_outputs[tid] * inv_norm * g;
    }
    __syncthreads();

    // Phase 2: partial-NeoX RoPE for the rotary half of Q (or K).
    // Output destination:
    //   Q: q_out[token, local_head, head_dim]
    //   K: key_cache[slot, local_head, head_dim]
    float final_val;
    if (tid < head_dim) {
        if (tid >= rotary_dim) {
            final_val = s_outputs[tid];
        } else {
            const bool is_lo = (tid < half_rot);
            const int  pair  = is_lo ? (tid + half_rot) : (tid - half_rot);
            const int  freq  = is_lo ? tid : pair;
            float self_v = s_outputs[tid];
            float pair_v = s_outputs[pair];
            float c = __half2float(cos_table[pos * half_rot + freq]);
            float s = __half2float(sin_table[pos * half_rot + freq]);
            final_val = is_lo ? (self_v * c - pair_v * s)
                              : (pair_v * s + self_v * c);
        }
    }

    if (is_q) {
        if (tid < head_dim) {
            const int q_dst = (token_idx * num_heads + local_head) * head_dim + tid;
            q_out[q_dst] = __float2half(final_val);
        } else if (tid < head_dim_2x) {
            // Gate output: passthrough (no norm, no RoPE).
            const int gate_dst = (token_idx * num_heads + local_head) * head_dim
                               + (tid - head_dim);
            gate_out[gate_dst] = __float2half(s_outputs[tid]);
        }
    } else if (is_k) {
        if (tid < head_dim) {
            const int slot = slot_mapping[token_idx];
            if (slot >= 0) {
                const int cache_off =
                    (slot * num_kv_heads + local_head) * head_dim;
                key_cache[cache_off + tid] = __float2half(final_val);
            }
        }
    }
}
