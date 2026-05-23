// Fused router GEMV + top-k + softmax for Qwen 3.6 MoE routing.
//
// Replaces the back-to-back pair `router_gemv_f16_to_f32_kernel`
// (one block per expert, computes logits[e]) +
// `topk_softmax_f32_kernel` (single block, picks top-K from logits)
// with ONE kernel using the "last block does the reduction"
// pattern via atomic counter:
//
//   Stage 1 (every block): compute logits[blockIdx.x] cooperatively
//   over `hidden`, write to global scratch.
//   __threadfence + atomicAdd(counter, 1u). The block whose prev
//   value equals num_experts - 1 is the LAST block to finish.
//   atomicExch resets the counter for the next call.
//
//   Stage 2 (last block only): topk-softmax reduction over
//   logits_scratch[0..num_experts], write top_idx[k] / top_w[k].
//   Block must have num_experts threads (the launcher guarantees
//   blockDim.x == num_experts when num_experts <= 1024).
//
// Eliminates 1 launch per MoE layer per token. At 40 MoE layers ×
// ~5 µs launch overhead = ~200 µs/token saved.
//
// Numerical contract: GEMV inner reduction byte-identical to the
// unfused `router_gemv_f16_to_f32_kernel`. Topk-softmax algorithm
// byte-identical to `topk_softmax_f32_kernel` (same k-round
// argmax tree reduction + softmax over selected logits).
//
// Caller contract:
//   * `counter` is a device i32[1] zeroed BEFORE the first call
//     after each worker startup. The kernel resets to 0 inside the
//     last block (via atomicExch), so subsequent calls find it at
//     zero again.
//   * `blockDim.x == num_experts` (so stage 2 has the right thread
//     count for topk reduction; per topk_softmax_f32_kernel's
//     contract).
//
// Launch:
//   Grid:  (num_experts, 1, 1)
//   Block: (num_experts, 1, 1)
//   Shared: ~12 KiB (topk reduction arrays)
//
// Phase 8 router+topk fusion (2026-05-23).

#include <cuda_fp16.h>
#include <math.h>

#define ROUTER_MAX_BLOCK 1024
#define ROUTER_MAX_K     32

__device__ __forceinline__ float router_warp_reduce_sum_rwt(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C" __global__ void __launch_bounds__(ROUTER_MAX_BLOCK)
router_gemv_with_topk_f16_to_f32_kernel(
    int*          __restrict__ top_idx,        // [k] i32
    float*        __restrict__ top_w,          // [k] f32
    float*        __restrict__ logits_scratch, // [num_experts] f32 — RMW
    const __half* __restrict__ router_w,       // [num_experts, hidden] f16
    const __half* __restrict__ input,          // [hidden] f16
    unsigned int* __restrict__ counter,        // u32 [1] — zeroed by host
                                                //          on first call;
                                                //          self-reset thereafter
    int num_experts,
    int hidden,
    int k
) {
    const int e = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Stage 1: per-expert GEMV — compute logits_scratch[e].
    const __half* row = router_w + (long long)e * hidden;
    __shared__ float gemv_smem[32];
    float local = 0.0f;
    for (int kk = tid; kk < hidden; kk += stride) {
        local += __half2float(row[kk]) * __half2float(input[kk]);
    }
    float acc = router_warp_reduce_sum_rwt(local);
    int wid = tid / 32;
    int lid = tid % 32;
    if (lid == 0) gemv_smem[wid] = acc;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    if (wid == 0) {
        acc = (lid < nw) ? gemv_smem[lid] : 0.0f;
        acc = router_warp_reduce_sum_rwt(acc);
    }

    __shared__ int is_last_block;
    if (tid == 0) {
        logits_scratch[e] = acc;
        __threadfence();
        unsigned int prev = atomicAdd(counter, 1u);
        if (prev == (unsigned int)(num_experts - 1)) {
            atomicExch(counter, 0u);
            is_last_block = 1;
        } else {
            is_last_block = 0;
        }
    }
    __syncthreads();
    if (is_last_block == 0) return;

    // Stage 2: topk-softmax over logits_scratch. Mirrors
    // topk_softmax_f32_kernel (block has num_experts threads;
    // each thread holds one expert's logit).
    float my_val = (tid < num_experts) ? logits_scratch[tid] : -INFINITY;
    int   my_idx = tid;

    __shared__ float s_val[ROUTER_MAX_BLOCK];
    __shared__ int   s_idx[ROUTER_MAX_BLOCK];

    __shared__ int   sel_idx[ROUTER_MAX_K];
    __shared__ float sel_val[ROUTER_MAX_K];

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
            top_idx[tid] = sel_idx[tid];
            top_w[tid]   = ee / sum;
        }
    }
}
