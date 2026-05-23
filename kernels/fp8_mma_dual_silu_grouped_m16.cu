// Task #94: TensorCore MMA dual_silu with TRUE M=16 grouping.
//
// Successor to `fp8_mma_dual_silu_indirect_kround_batched_kernel`
// (task #93 first cut, M=1 per block, 15/16 MMA throughput wasted).
// This kernel processes [M=16, N=8] tiles that all share ONE expert
// — recovering the M-direction MMA reuse the first cut sacrificed.
// Targets the qwen36 MoE prefill bottleneck (62% of prefill GPU
// time per the 2026-05-23 nsys profile).
//
// Caller prepares two device-side inputs:
//   sorted_per_expert[num_experts * max_per_expert * 2] i32
//     — (token_idx, k_round) pairs grouped per expert, produced by
//       `qwen36_moe_expert_sort_kernel`.
//   tile_descriptors[num_total_tiles * 3] i32
//     — (expert_id, m_offset_within_expert, tile_size_m) triples,
//       one per tile. `tile_size_m` ∈ [1, 16]; the kernel zero-pads
//       trailing rows when tile_size_m < 16.
//
// Grid: (ceil(N/8), num_total_tiles, 1).  Block: (32, 1, 1).
//
// Per block:
//   1. Look up the tile descriptor → expert id + assignment range.
//   2. Gather 16 input rows: x[sorted_token_idx[i]] for i in
//      [0, tile_size_m); zero rows for i in [tile_size_m, 16).
//   3. Per K=128 sub-block: amax-quant of A tile, blockscaled MMA
//      for both gate + up against expert's W_g + W_u.
//   4. Per K=128: multiply per-row a_scale[r] * weight scale, fold
//      into outer f32 accumulators.
//   5. After all K: silu(g)*u → scatter back to
//      out_silu[k_round[i], token_idx[i], n_base + col] f16, per
//      row i in the live range.
//
// Input layout: same as the GEMV variant
//   input [M_full, K] f16, top_idx-derived (no longer needed by this
//   kernel since the sort already mapped routing → expert).
//   Weights identical to the GEMV: [num_experts, N, K] FP8 + per-
//   (n>>7, k>>7) blockscale f32.
//   Output: out_silu[k_round, M_full, N] f16 — same layout as
//   GEMV / task #93 first cut so the down-projection consumer reads
//   identical bytes.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>
#include "fp8_mma_frag_pack.cuh"

namespace {

__device__ __forceinline__ unsigned char f32_to_e4m3(float v) {
    return (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

}  // anon

extern "C"
__global__ void fp8_mma_dual_silu_grouped_m16_kernel(
    __half*       __restrict__ out_silu,       // [top_k, M_full, N] f16
    const unsigned char* __restrict__ base_w_g,  // [E, N, K] fp8
    const unsigned char* __restrict__ base_w_u,
    const float*  __restrict__ base_s_g,       // [E, N/128, K/128] f32
    const float*  __restrict__ base_s_u,
    const __half* __restrict__ input,          // [M_full, K] f16
    const int*    __restrict__ sorted_per_expert,
                                               // [E * max_per_expert * 2] i32
    const int*    __restrict__ tile_descriptors,
                                               // [num_total_tiles * 3] i32
    long long w_stride,
    long long s_stride,
    int M_full,
    int N,
    int K,
    int num_col_blocks,
    int top_k,
    int max_per_expert
) {
    int lane     = threadIdx.x & 31;
    int n_block  = blockIdx.x;
    int tile_idx = blockIdx.y;
    if (n_block * 8 >= N) return;
    int n_base = n_block * 8;

    int e            = tile_descriptors[tile_idx * 3 + 0];
    int m_off_in_exp = tile_descriptors[tile_idx * 3 + 1];
    int tile_m       = tile_descriptors[tile_idx * 3 + 2];
    if (tile_m <= 0) return;

    const unsigned char* w_g_base = base_w_g + (long long)e * w_stride;
    const unsigned char* w_u_base = base_w_u + (long long)e * w_stride;
    const float*         s_g_base = base_s_g + (long long)e * s_stride;
    const float*         s_u_base = base_s_u + (long long)e * s_stride;

    int scale_row = n_base >> 7;

    // Per-row token_idx / k_round, gathered once from sorted list.
    // Only rows [0, tile_m) are live; rows [tile_m, 16) carry the
    // sentinel token_idx = -1 → input row zeroed.
    // To keep this in lane-distributed form: each lane handles a
    // subset of rows; we broadcast via warp shfl when needed.
    // Simpler: read all 16 row-meta via lane < 16.
    int sorted_base = (e * max_per_expert + m_off_in_exp) * 2;
    int my_token_idx = -1;
    int my_k_round   = -1;
    if (lane < tile_m) {
        my_token_idx = sorted_per_expert[sorted_base + lane * 2 + 0];
        my_k_round   = sorted_per_expert[sorted_base + lane * 2 + 1];
    }

    // === MMA accumulators (per-lane fragment, 4 f32) for gate + up.
    float g_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float u_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // === Shared smem layout:
    //   smem_a:   [16 rows × 32 K bytes] = 512 B (per K=32 sub-tile)
    //   smem_b_g: [ 8 rows × 32 K bytes] = 256 B
    //   smem_b_u: [ 8 rows × 32 K bytes] = 256 B
    //   smem_ascale: [16 f32] = 64 B (per-row a_scale broadcast across kt)
    // Total: 1088 B.
    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_a       = smem_raw;
    unsigned char* smem_b_g     = smem_a   + 16 * 32;
    unsigned char* smem_b_u     = smem_b_g +  8 * 32;
    float*         smem_ascale  = reinterpret_cast<float*>(smem_b_u +  8 * 32);

    // === Per-row token pointer cache (gathered).
    // For row r, x_row[r] = input + token_idx[r] * K (or null when
    // row is zero-padded). Build by reading via warp from
    // sorted_per_expert.
    // We re-read token_idx per kt rather than caching — that's only
    // 16 lookups but it's cheap; OR cache once in registers per lane.
    // Already cached: `my_token_idx` per lane (lane in [0,16) holds
    // its row's token_idx; lanes 16..31 hold -1).

    // ---- One-time: row 0..15 input-row pointer cached as offset
    //      Per-lane: row index = lane (when lane<16).

    int K_per_block  = 128;
    int num_k_blocks = K / K_per_block;

    for (int kblk = 0; kblk < num_k_blocks; ++kblk) {
        int k_base = kblk * K_per_block;

        // === Compute per-row amax over the K=128 block.
        // Each lane handles ONE row (lane<16); inactive lanes carry
        // amax=0. Loop over the K=128 in chunks of 32 reading 32
        // elements per chunk (4 chunks); but a single lane can only
        // read sequentially → 128 reads/lane. Expensive but
        // straightforward. (Optimisation: collaborative reduce across
        // 4 lanes per row using 4-lane sub-warps — parked.)
        float my_amax = 0.0f;
        if (lane < 16 && lane < tile_m && my_token_idx >= 0) {
            const __half* x_row = input + (long long)my_token_idx * K;
            #pragma unroll 4
            for (int k = 0; k < K_per_block; ++k) {
                int kg = k_base + k;
                if (kg < K) {
                    float xv = __half2float(x_row[kg]);
                    my_amax = fmaxf(my_amax, fabsf(xv));
                }
            }
        }
        float a_scale = my_amax / 448.0f;
        if (a_scale == 0.0f) a_scale = 1e-30f;
        if (lane < 16) smem_ascale[lane] = a_scale;
        __syncwarp();

        // === Weight blockscales (one per (n_block, kblk)).
        float sg = s_g_base[scale_row * num_col_blocks + kblk];
        float su = s_u_base[scale_row * num_col_blocks + kblk];

        // === Per-K=128 inner accumulators (per-lane f32 fragment).
        // Outer accumulators store sum across kblks (without per-row
        // a_scale folding — applied at output-write time so we don't
        // need per-(m, n) scale arrays in registers). The MMA
        // accumulates: g[m, n] = sum_k (x_e4m3[m,k] * sg[m_blk]) *
        // (w_e4m3[n,k] * sg_w[n_blk]) → factor sg_w out (uniform per
        // K=128) here; per-row sg (a_scale[m]) applied at write.
        float g_inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float u_inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            int k_off = kt * 32;
            int k_g   = k_base + k_off;

            // ---- Stage A tile: 16 rows × 32 bytes. Each row r is
            //      written by lane r (lanes 0..15 write 32 bytes
            //      each, lane 16..31 idle). Each lane reads 32 f16
            //      values from its row, quantizes by its row's
            //      a_scale, packs 32 FP8 bytes.
            // Stage A tile: lanes 0..15 own one row each. Each row's
            // token_idx + a_scale are already in the lane's regs
            // (my_token_idx) or smem_ascale[row]. No cross-lane comm
            // needed.
            if (lane < 16) {
                int row = lane;
                bool live = (row < tile_m && my_token_idx >= 0);
                float inv_s = live ? (1.0f / smem_ascale[row]) : 0.0f;
                const __half* x_row = live
                    ? (input + (long long)my_token_idx * K) : nullptr;
                #pragma unroll 32
                for (int kk = 0; kk < 32; ++kk) {
                    int kg = k_g + kk;
                    float xv = 0.0f;
                    if (x_row && kg < K) {
                        xv = __half2float(x_row[kg]);
                    }
                    smem_a[row * 32 + kk] = f32_to_e4m3(xv * inv_s);
                }
            }

            // ---- Stage B_g + B_u (8 rows × 32 bytes each).
            //      Each lane writes 8 bytes across rows (8 rows × 1
            //      byte per lane = 8 writes per lane to B_g + 8 to B_u).
            //      Wait — we have 32 lanes × 32 bytes total per buffer
            //      (8 rows × 32 bytes = 256 bytes). 256 / 32 = 8 bytes
            //      per lane per buffer.
            #pragma unroll
            for (int n_off = 0; n_off < 8; ++n_off) {
                int n_g = n_base + n_off;
                int k_idx = k_g + lane;
                unsigned char wg = 0, wu = 0;
                if (n_g < N && k_idx < K) {
                    wg = w_g_base[(long long)n_g * K + k_idx];
                    wu = w_u_base[(long long)n_g * K + k_idx];
                }
                smem_b_g[n_off * 32 + lane] = wg;
                smem_b_u[n_off * 32 + lane] = wu;
            }
            __syncwarp();

            // ---- Pack A + B fragments + issue dual MMAs.
            uint32_t a_frag[4];
            uint32_t bg_frag[2];
            uint32_t bu_frag[2];
            rvllm::pack_a_frag_row_major_m16k32(smem_a,   32, a_frag,  lane);
            rvllm::pack_b_frag_col_major_n8k32(smem_b_g,  32, bg_frag, lane);
            rvllm::pack_b_frag_col_major_n8k32(smem_b_u,  32, bu_frag, lane);

            rvllm::mma_m16n8k32_e4m3_e4m3_f32(g_inner, a_frag, bg_frag);
            rvllm::mma_m16n8k32_e4m3_e4m3_f32(u_inner, a_frag, bu_frag);
        }

        // Fold per-K=128 inner into outer with weight scale (not
        // a_scale — that's per-row, applied at write).
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            g_outer[i] += sg * g_inner[i];
            u_outer[i] += su * u_inner[i];
        }
    }

    // === Scatter output.
    // Per-lane D-frag (m16n8):
    //   d[0]: row (lane/4),    col (lane%4)*2 + 0
    //   d[1]: row (lane/4),    col (lane%4)*2 + 1
    //   d[2]: row (lane/4+8),  col (lane%4)*2 + 0
    //   d[3]: row (lane/4+8),  col (lane%4)*2 + 1
    //
    // Apply per-row a_scale at write time. For row r live (r <
    // tile_m), write to out_silu[k_round[r], token_idx[r], n_base+col].
    // Stale rows (r >= tile_m) skip the write.
    int r_lo = lane >> 2;
    int r_hi = r_lo + 8;
    int c0   = (lane & 3) * 2;
    int c1   = c0 + 1;
    int n0   = n_base + c0;
    int n1   = n_base + c1;

    auto write_row = [&] (int row, float g_v, float u_v, int n_col) {
        if (row >= tile_m || n_col >= N) return;
        // Look up token_idx + k_round for this row via the sorted list.
        // (Cheap re-read; better cached but acceptable here.)
        int tok = sorted_per_expert[sorted_base + row * 2 + 0];
        int kr  = sorted_per_expert[sorted_base + row * 2 + 1];
        float scale_r = smem_ascale[row];
        float g_scaled = scale_r * g_v;
        float u_scaled = scale_r * u_v;
        long long off = ((long long)kr * M_full + tok) * N + n_col;
        out_silu[off] = __float2half(silu_f(g_scaled) * u_scaled);
    };

    write_row(r_lo, g_outer[0], u_outer[0], n0);
    write_row(r_lo, g_outer[1], u_outer[1], n1);
    write_row(r_hi, g_outer[2], u_outer[2], n0);
    write_row(r_hi, g_outer[3], u_outer[3], n1);
}
