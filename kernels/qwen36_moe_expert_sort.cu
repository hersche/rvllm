// Task #94: per-expert grouping pre-pass for qwen36 MoE prefill MMA path.
//
// Reads the routing table `top_idx[M, top_k]` and produces, per
// expert id, a list of (token_idx, k_round) assignments laid out
// contiguously in a [num_experts × max_per_expert × 2] i32 buffer.
// Companion `expert_counts[num_experts]` carries the per-expert
// fill count. The downstream MMA dual_silu kernel consumes these
// to build [M=16, N=8] tiles that share ONE expert (recovering the
// M-direction MMA reuse the task #93 first-cut sacrificed).
//
// Caller contract:
//   * `expert_counts` MUST be zero-initialised before launch
//     (cuMemsetD32Async).
//   * `max_per_expert` MUST be ≥ the worst-case per-expert count;
//     `M * top_k` is the safe upper bound. Out-of-range slots are
//     silently dropped (the host caller MUST validate that
//     `max(expert_counts) ≤ max_per_expert` post-kernel before
//     consuming the sorted buffer — overflow is a hard error and
//     the caller falls back to the GEMV path).
//   * Grid: (ceil(M*top_k / block_size), 1, 1); block.x = 256 ok.
//
// Sort is stable-ish: assignment order WITHIN one expert's bucket
// depends on atomicAdd race winners. The downstream MMA kernel
// scatters output by (token_idx, k_round) so the per-expert sort
// order does not affect correctness — only the bookkeeping order.

#include <cstdint>

extern "C"
__global__ void qwen36_moe_expert_sort_kernel(
    const int* __restrict__ top_idx,        // [M, top_k] i32
    int*       __restrict__ sorted_per_expert,
                                            // [num_experts * max_per_expert * 2] i32
                                            // packed as (token_idx, k_round) pairs
    int*       __restrict__ expert_counts,  // [num_experts] i32 (zero-init outside)
    int M,
    int top_k,
    int num_experts,
    int max_per_expert
) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * (long long)top_k;
    if ((long long)tid >= total) return;

    int m       = tid / top_k;
    int k_round = tid % top_k;
    int e       = top_idx[(long long)m * (long long)top_k + (long long)k_round];
    if (e < 0 || e >= num_experts) return;

    int slot = atomicAdd(&expert_counts[e], 1);
    if (slot >= max_per_expert) {
        // Overflow — silently drop. Caller is expected to detect
        // via `max(expert_counts) > max_per_expert` and fall back.
        return;
    }
    long long base = ((long long)e * (long long)max_per_expert + (long long)slot) * 2;
    sorted_per_expert[base + 0] = m;
    sorted_per_expert[base + 1] = k_round;
}
