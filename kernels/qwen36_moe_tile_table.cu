// Task #94 helper: build per-expert M=16 tile descriptors from the
// post-sort `expert_counts[num_experts]` produced by
// `qwen36_moe_expert_sort_kernel`.
//
// Per expert e with C[e] sorted assignments, emits ceil(C[e]/16)
// tile descriptors (expert_id, m_offset_in_expert, tile_size_m)
// into `tile_descriptors[num_total_tiles * 3]`, where
// `tile_size_m = min(16, C[e] - m_offset)`. Final tile per expert
// may be a partial (1..15-row) tile.
//
// `tile_count[0]` is the global atomic counter — written by this
// kernel and read by the host to size the MMA kernel's grid.y.
//
// Single thread per expert. Launch: grid = (ceil(num_experts/64), 1, 1),
// block = 64.
//
// Out-of-band overflow is silently dropped if a per-expert count
// exceeds the buffer's `max_per_expert`; the caller MUST detect
// `max(expert_counts) > max_per_expert` before consuming and fall
// back to the legacy GEMV path. Same contract as the sort kernel.

#include <cstdint>

extern "C"
__global__ void qwen36_moe_tile_table_kernel(
    const int* __restrict__ expert_counts,    // [num_experts] i32
    int*       __restrict__ tile_descriptors, // [max_total_tiles * 3] i32
    int*       __restrict__ tile_count,       // [1] i32 atomic counter
    int num_experts,
    int max_total_tiles
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_experts) return;

    int c = expert_counts[e];
    if (c <= 0) return;

    int num_tiles_for_e = (c + 15) / 16;  // ceil(c / 16)
    int base = atomicAdd(tile_count, num_tiles_for_e);
    if (base + num_tiles_for_e > max_total_tiles) return; // overflow

    for (int t = 0; t < num_tiles_for_e; ++t) {
        int m_off    = t * 16;
        int tile_sz  = c - m_off;
        if (tile_sz > 16) tile_sz = 16;
        long long off = ((long long)(base + t)) * 3;
        tile_descriptors[off + 0] = e;
        tile_descriptors[off + 1] = m_off;
        tile_descriptors[off + 2] = tile_sz;
    }
}
