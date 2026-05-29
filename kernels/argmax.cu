// GPU-side argmax kernel: finds the token ID with maximum logit per row.
// Eliminates full logits DtoH copy for greedy (temperature=0) decoding.
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(vocab_size, 1024), 1, 1)
//   Shared memory: none (uses static shared arrays)
//
// Each block finds the argmax of one token's logits row via shared memory reduction,
// then writes the winning token ID to output_token[row].

#include <float.h>
#include <cuda_fp16.h>

extern "C"
__global__ void argmax_f16_kernel(
    const __half* __restrict__ logits,
    int* __restrict__ output_token,
    int vocab_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int n = blockDim.x;

    const __half* x = logits + (long long)row * vocab_size;

    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];

    float local_max = -FLT_MAX;
    int   local_idx = 0;
    for (int i = tid; i < vocab_size; i += stride) {
        float v = __half2float(x[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

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
        output_token[row] = s_idx[0];
    }
}

// Top-k + top-p + temperature categorical sampler over f16 logits.
// Single block per row (grid.x = num_tokens). Replaces argmax_f16 for
// the temperature>0 path (Qwen3 is documented to repeat/degrade under
// greedy; its recommended sampling is temp=0.6 / top_k=20 / top_p=0.95).
//
// Pipeline (one CTA):
//   1. Iterative top-k extraction (k passes of a masked block-argmax)
//      → the k largest (logit, index) pairs, descending, in shared.
//   2. thread 0: temperature-softmax over those k, top-p nucleus filter
//      (keep the shortest descending prefix with cumulative prob >= p),
//      then inverse-CDF sample with a per-call uniform drawn from `seed`.
//
// k is capped at K_MAX=64 (shared arrays). temperature<=0, k<=1, or
// k>vocab degenerates to argmax (pick top-1). Deterministic given the
// same (seed, logits): the host advances `seed` per generated token.
#define SAMPLE_K_MAX 64

extern "C"
__global__ void sample_topk_topp_f16_kernel(
    const __half* __restrict__ logits,
    int* __restrict__ output_token,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    unsigned long long seed
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int n = blockDim.x;
    const __half* x = logits + (long long)row * vocab_size;

    int k = top_k;
    if (k <= 0 || k > SAMPLE_K_MAX) k = SAMPLE_K_MAX;
    if (k > vocab_size) k = vocab_size;

    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];
    __shared__ float top_val[SAMPLE_K_MAX];
    __shared__ int   top_idx[SAMPLE_K_MAX];

    // --- 1. Iterative top-k extraction via masked block-argmax. ---
    for (int j = 0; j < k; ++j) {
        float local_max = -FLT_MAX;
        int   local_idx = -1;
        for (int i = tid; i < vocab_size; i += stride) {
            // Skip indices already picked (the j picked so far live in
            // top_idx[0..j); j is small so this scan is cheap).
            bool picked = false;
            for (int p = 0; p < j; ++p) {
                if (top_idx[p] == i) { picked = true; break; }
            }
            if (picked) continue;
            float v = __half2float(x[i]);
            if (v > local_max) { local_max = v; local_idx = i; }
        }
        s_val[tid] = local_max;
        s_idx[tid] = local_idx;
        __syncthreads();
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
            top_val[j] = s_val[0];
            top_idx[j] = s_idx[0];
        }
        __syncthreads();
    }

    // --- 2. thread 0: temperature-softmax + top-p + CDF sample. ---
    if (tid == 0) {
        // Greedy fallback.
        if (temperature <= 0.0f || k <= 1) {
            output_token[row] = top_idx[0];
            return;
        }
        float inv_t = 1.0f / temperature;
        // Softmax over the k kept (descending). top_val[0] is the max.
        float maxv = top_val[0] * inv_t;
        float probs[SAMPLE_K_MAX];
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            float p = expf(top_val[j] * inv_t - maxv);
            probs[j] = p;
            sum += p;
        }
        // Normalise + top-p nucleus: keep the shortest descending
        // prefix whose cumulative prob >= top_p, renormalise over it.
        float tp = (top_p > 0.0f && top_p < 1.0f) ? top_p : 1.0f;
        int keep = k;
        float cum = 0.0f;
        for (int j = 0; j < k; ++j) {
            cum += probs[j] / sum;
            if (cum >= tp) { keep = j + 1; break; }
        }
        float kept_sum = 0.0f;
        for (int j = 0; j < keep; ++j) kept_sum += probs[j];
        // Uniform draw from `seed` (splitmix64 → [0,1)).
        unsigned long long z = seed + 0x9E3779B97F4A7C15ULL
                             + ((unsigned long long)row << 32);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        float u = (float)((z >> 11) * (1.0 / 9007199254740992.0)); // [0,1)
        float target = u * kept_sum;
        float acc = 0.0f;
        int chosen = top_idx[keep - 1];
        for (int j = 0; j < keep; ++j) {
            acc += probs[j];
            if (acc >= target) { chosen = top_idx[j]; break; }
        }
        output_token[row] = chosen;
    }
}

extern "C"
__global__ void argmax_kernel(
    const float* __restrict__ logits,
    int* __restrict__ output_token,
    int vocab_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int n = blockDim.x;

    const float* x = logits + (long long)row * vocab_size;

    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];

    // Pass 1: thread-local max across strided elements
    float local_max = -FLT_MAX;
    int   local_idx = 0;
    for (int i = tid; i < vocab_size; i += stride) {
        float v = x[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Tree reduction for argmax. Handles any `n` (= blockDim.x), not
    // just powers-of-two — the orphan check below covers the slot
    // that the main pair-fold misses on each step.
    //
    // Correctness argument for non-power-of-two n:
    //   Each step folds pairs (i, i+s) for i in [0, s), so slots
    //   [2*s, n) are NOT touched by the main fold. The orphan check
    //   `if (s*2 < n)` folds the slot at index `2*s` (the leftmost
    //   uncovered slot) into slot 0. The DEPTH of slot `2*s` carries
    //   the data of every higher slot, because at the previous step
    //   `2s_prev = s` and the main fold of THAT step pulled higher
    //   data down into `[0, s_prev)`. Inductively, slot `2*s` after
    //   each step is a complete max over the upper region the main
    //   fold could not reach. Walking n=6 / n=11 / n=14 confirms it.
    //
    //   The reduction therefore terminates with `s_val[0]` holding
    //   the global max. Power-of-two n simply degenerates to the
    //   familiar branch-free version because the orphan check is
    //   never triggered (s*2 == n at every step).
    for (int s = n / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        // Orphan fold: catch slot `2*s` when n was not perfectly
        // halved by the previous step. See the proof above.
        if (s * 2 < n && tid == 0) {
            if (s_val[s * 2] > s_val[0]) {
                s_val[0] = s_val[s * 2];
                s_idx[0] = s_idx[s * 2];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes the result
    if (tid == 0) {
        output_token[row] = s_idx[0];
    }
}
