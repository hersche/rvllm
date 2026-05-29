// Repetition penalty for the qwen35 dense decode path (design A:
// count histogram). A small device-resident i32[vocab] histogram
// records how many times each token has been emitted in the current
// request. Before each token selection the penalty kernel subtracts a
// frequency- and presence-scaled amount from the f16 logits in place;
// after each pick the increment kernel bumps the chosen token's count.
//
// Why a SEPARATE pass instead of folding into argmax/sample: the
// `argmax_f16_kernel` in argmax.cu is shared across every model family
// (gemma4, gemma4-nvfp4, gemma4-drafter, qwen35, qwen36) — its
// signature must not change. This pass runs only when a penalty is
// active (qwen35 worker skips both launches when freq==0 && presence==0),
// so the greedy/argmax and sampler paths stay byte-identical when off.
//
// Formula (OpenAI convention): for tokens with count c >= min_count,
//   logit -= frequency_penalty * c + presence_penalty
// The min_count gate (default raise to 2) protects common function
// words / subwords that legitimately recur — the gemma path learned the
// un-gated penalty is "too blunt for greedy decode".

#include <cuda_fp16.h>

extern "C"
__global__ void qwen_apply_freq_presence_penalty_f16_kernel(
    __half* __restrict__ logits,
    const int* __restrict__ count,
    int vocab,
    float frequency_penalty,
    float presence_penalty,
    int min_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (; i < vocab; i += stride) {
        int c = count[i];
        if (c > 0 && c >= min_count) {
            float v = __half2float(logits[i]);
            v -= frequency_penalty * (float)c + presence_penalty;
            logits[i] = __float2half(v);
        }
    }
}

// Single-thread bump of the picked token's count. `token_ptr` points at
// the i32 token id the selection kernel just wrote (token_out_ptr).
extern "C"
__global__ void qwen_rep_count_increment_kernel(
    int* __restrict__ count,
    const int* __restrict__ token_ptr
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        count[token_ptr[0]] += 1;
    }
}
