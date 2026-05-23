// Batched-prefill fused router GEMV + top-k + softmax.
//
// Combines `router_gemv_batched_f16_to_f32` and
// `topk_softmax_batched_f32` into a single kernel via the
// per-token "last-block-does-topk" atomic pattern. Eliminates one
// launch per layer in the batched-MoE prefill path
// (apply_layer_moe_batched / route_override == None branch).
//
// Per-token last-block detection: each token `t` has its own
// counter slot `counter[t]`. Per-(e, t) block does the GEMV,
// writes logits[t, e], threadfence + atomicAdd(counter[t]). The
// block whose prev value == num_experts - 1 is the last block for
// THAT token; it resets counter[t] to 0 via atomicExch and
// proceeds to stage 2 (topk-softmax over logits[t, *]).
//
// The launcher allocates a `[num_tokens]` u32 counter region
// (zeroed once at worker bring-up via 4*num_tokens HtoD); the
// kernel self-resets per-token slots after each call.
//
// Launch geometry mirrors the unfused pair:
//   Grid:  (num_experts, num_tokens, 1)
//   Block: (num_experts, 1, 1)
//
// Block size must be num_experts (so stage 2 has the right thread
// count). For num_experts > 1024 not supported (consistent with
// the single-token kernel).
//
// Numerical contract: GEMV reduction byte-identical to
// router_gemv_batched_f16_to_f32. Topk-softmax algorithm byte-
// identical to topk_softmax_batched_f32.
//
// Phase 8 batched router+topk fusion (2026-05-23).

#include <cuda_fp16.h>
#include <math.h>

#define ROUTER_BATCHED_MAX_BLOCK 1024
#define ROUTER_BATCHED_MAX_K     32

__device__ __forceinline__ float router_b_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C" __global__ void __launch_bounds__(ROUTER_BATCHED_MAX_BLOCK)
router_gemv_with_topk_batched_f16_to_f32_kernel(
    int*          __restrict__ top_idx,        // [num_tokens, k] i32
    float*        __restrict__ top_w,          // [num_tokens, k] f32
    float*        __restrict__ logits_scratch, // [num_tokens, num_experts] f32
    const __half* __restrict__ router_w,       // [num_experts, hidden] f16
    const __half* __restrict__ input,          // [num_tokens, hidden] f16
    unsigned int* __restrict__ counter,        // u32 [num_tokens] — zeroed
                                                //                    by host
                                                //                    once;
                                                //                    self-reset
    int num_experts,
    int hidden,
    int k,
    int num_tokens
) {
    const int e = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if (e >= num_experts || t >= num_tokens) return;

    // Stage 1: per-(e, t) GEMV.
    const __half* row = router_w + (long long)e * hidden;
    const __half* inp = input + (long long)t * hidden;
    __shared__ float gemv_smem[32];
    float local = 0.0f;
    for (int kk = tid; kk < hidden; kk += stride) {
        local += __half2float(row[kk]) * __half2float(inp[kk]);
    }
    float acc = router_b_warp_reduce_sum(local);
    int wid = tid / 32;
    int lid = tid % 32;
    if (lid == 0) gemv_smem[wid] = acc;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    if (wid == 0) {
        acc = (lid < nw) ? gemv_smem[lid] : 0.0f;
        acc = router_b_warp_reduce_sum(acc);
    }

    __shared__ int is_last_block;
    if (tid == 0) {
        logits_scratch[(long long)t * num_experts + e] = acc;
        __threadfence();
        unsigned int prev = atomicAdd(&counter[t], 1u);
        if (prev == (unsigned int)(num_experts - 1)) {
            atomicExch(&counter[t], 0u);
            is_last_block = 1;
        } else {
            is_last_block = 0;
        }
    }
    __syncthreads();
    if (is_last_block == 0) return;

    // Stage 2: topk-softmax over logits_scratch[t, *]. Block has
    // num_experts threads (per the launcher contract).
    const float* my_logits  = logits_scratch + (long long)t * num_experts;
    int*         my_top_idx = top_idx + (long long)t * k;
    float*       my_top_w   = top_w   + (long long)t * k;

    float my_val = (tid < num_experts) ? my_logits[tid] : -INFINITY;
    int   my_idx = tid;

    __shared__ float s_val[ROUTER_BATCHED_MAX_BLOCK];
    __shared__ int   s_idx[ROUTER_BATCHED_MAX_BLOCK];
    __shared__ int   sel_idx[ROUTER_BATCHED_MAX_K];
    __shared__ float sel_val[ROUTER_BATCHED_MAX_K];

    for (int kk = 0; kk < k; kk++) {
        s_val[tid] = my_val;
        s_idx[tid] = my_idx;
        __syncthreads();

        int n = num_experts;
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
            sel_idx[kk] = s_idx[0];
            sel_val[kk] = s_val[0];
        }
        __syncthreads();

        if (my_idx == sel_idx[kk]) {
            my_val = -INFINITY;
        }
        __syncthreads();
    }

    if (tid < 32) {
        float m = sel_val[0];
        float ee = (tid < k) ? expf(sel_val[tid] - m) : 0.0f;
        float sum = ee;
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, off);
        }
        if (tid < k) {
            my_top_idx[tid] = sel_idx[tid];
            my_top_w[tid]   = ee / sum;
        }
    }
}
