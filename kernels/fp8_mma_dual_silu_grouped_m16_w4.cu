// Task #95: W=4 multi-warp grouped MMA dual_silu.
//
// Sibling of `fp8_mma_dual_silu_grouped_m16_kernel` (task #94, 1 warp
// per block). This variant runs 4 warps per block; all 4 warps share
// ONE A tile (the 16 routed-token rows for one expert × K=32) and
// each warp covers a DIFFERENT N=8 column tile. Effective output
// area per block: [M=16, N=32] instead of [M=16, N=8]. Same MMA
// throughput per warp, but A tile loads + amax computations + B-tile
// staging coordination amortise across 4 warps → fewer total blocks,
// fewer total A loads.
//
//   Grid:  (ceil(N/32), num_total_tiles, 1)     ← 4× fewer X blocks
//   Block: (128, 1, 1)                           ← 4 warps
//
// Per block:
//   * warp 0 (lanes 0..15) stages the A tile + computes per-row
//     amax + a_scale (broadcast via smem to all warps).
//   * All 4 warps stage their own B_g[w] + B_u[w] (different cols).
//   * 4 MMAs per K=128 per warp; accumulators private per-warp.
//   * Output scatter: each warp writes to its own N=8 slice of the
//     output `[k_round, M, N]` tile.
//
// Numerical contract: identical to the 1-warp variant up to MMA
// reduction order (each warp's accumulator is independent and
// scaled by the same blockscale). Output bytes match within
// per-element rounding.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>
#include "fp8_mma_frag_pack.cuh"

#define WARPS_PER_BLOCK 4

namespace {

__device__ __forceinline__ unsigned char f32_to_e4m3(float v) {
    return (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

}  // anon

extern "C"
__global__ void fp8_mma_dual_silu_grouped_m16_w4_kernel(
    __half*       __restrict__ out_silu,         // [top_k, M_full, N] f16
    const unsigned char* __restrict__ base_w_g,  // [E, N, K] fp8
    const unsigned char* __restrict__ base_w_u,
    const float*  __restrict__ base_s_g,         // [E, N/128, K/128] f32
    const float*  __restrict__ base_s_u,
    const __half* __restrict__ input,            // [M_full, K] f16
    const int*    __restrict__ sorted_per_expert,
    const int*    __restrict__ tile_descriptors,
    long long w_stride,
    long long s_stride,
    int M_full,
    int N,
    int K,
    int num_col_blocks,
    int top_k,
    int max_per_expert
) {
    int tid       = threadIdx.x;
    int warp_id   = tid >> 5;
    int lane      = tid & 31;
    int n_supblk  = blockIdx.x;          // covers N cols [n_super, n_super + 32)
    int tile_idx  = blockIdx.y;
    int n_super   = n_supblk * (8 * WARPS_PER_BLOCK);
    int n_base    = n_super + warp_id * 8;
    if (n_base >= N) return;

    int e            = tile_descriptors[tile_idx * 3 + 0];
    int m_off_in_exp = tile_descriptors[tile_idx * 3 + 1];
    int tile_m       = tile_descriptors[tile_idx * 3 + 2];
    if (tile_m <= 0) return;

    const unsigned char* w_g_base = base_w_g + (long long)e * w_stride;
    const unsigned char* w_u_base = base_w_u + (long long)e * w_stride;
    const float*         s_g_base = base_s_g + (long long)e * s_stride;
    const float*         s_u_base = base_s_u + (long long)e * s_stride;

    int scale_row = n_base >> 7;

    // Per-warp metadata for rows. Warp 0 lanes 0..15 hold the row
    // token_idx (used during staging + scatter). Other warps don't
    // need it — they consume the SHARED smem_a + smem_ascale.
    int sorted_base = (e * max_per_expert + m_off_in_exp) * 2;
    int w0_token_idx = -1;
    int w0_k_round   = -1;
    if (warp_id == 0 && lane < tile_m) {
        w0_token_idx = sorted_per_expert[sorted_base + lane * 2 + 0];
        w0_k_round   = sorted_per_expert[sorted_base + lane * 2 + 1];
    }

    // Per-warp MMA accumulators.
    float g_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float u_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // SHARED smem layout:
    //   smem_a:     [16, 32] = 512 B (shared by all warps)
    //   smem_b_g[W]:[W*8, 32] = W*256 B
    //   smem_b_u[W]:[W*8, 32] = W*256 B
    //   smem_ascale:[16] f32  = 64 B
    // Total at W=4: 512 + 1024 + 1024 + 64 = 2624 B
    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_a       = smem_raw;
    unsigned char* smem_b_g_all = smem_a + 16 * 32;
    unsigned char* smem_b_u_all = smem_b_g_all + WARPS_PER_BLOCK * 8 * 32;
    float*         smem_ascale  = reinterpret_cast<float*>(
                                    smem_b_u_all + WARPS_PER_BLOCK * 8 * 32);
    // Per-warp B-tile pointers.
    unsigned char* smem_b_g = smem_b_g_all + warp_id * 8 * 32;
    unsigned char* smem_b_u = smem_b_u_all + warp_id * 8 * 32;

    int K_per_block  = 128;
    int num_k_blocks = K / K_per_block;

    for (int kblk = 0; kblk < num_k_blocks; ++kblk) {
        int k_base = kblk * K_per_block;

        // === Warp 0: amax over K=128 per row + write to smem_ascale.
        if (warp_id == 0) {
            float my_amax = 0.0f;
            if (lane < 16 && lane < tile_m && w0_token_idx >= 0) {
                const __half* x_row = input + (long long)w0_token_idx * K;
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
        }
        __syncthreads();

        // === Weight blockscales (per-warp scale_row).
        float sg = s_g_base[scale_row * num_col_blocks + kblk];
        float su = s_u_base[scale_row * num_col_blocks + kblk];

        // === Per-K=128 inner accumulators.
        float g_inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float u_inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            int k_off = kt * 32;
            int k_g   = k_base + k_off;

            // === Stage A tile (warp 0 only; lanes 0..15 own one row each). ===
            if (warp_id == 0 && lane < 16) {
                int row = lane;
                bool live = (row < tile_m && w0_token_idx >= 0);
                float inv_s = live ? (1.0f / smem_ascale[row]) : 0.0f;
                const __half* x_row = live
                    ? (input + (long long)w0_token_idx * K) : nullptr;
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

            // === Stage B_g + B_u (per-warp). ===
            // n_base..n_base+7 are this warp's N columns.
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
            __syncthreads();

            // === Pack frags + issue dual MMAs. ===
            uint32_t a_frag[4];
            uint32_t bg_frag[2];
            uint32_t bu_frag[2];
            rvllm::pack_a_frag_row_major_m16k32(smem_a,   32, a_frag,  lane);
            rvllm::pack_b_frag_col_major_n8k32(smem_b_g,  32, bg_frag, lane);
            rvllm::pack_b_frag_col_major_n8k32(smem_b_u,  32, bu_frag, lane);

            rvllm::mma_m16n8k32_e4m3_e4m3_f32(g_inner, a_frag, bg_frag);
            rvllm::mma_m16n8k32_e4m3_e4m3_f32(u_inner, a_frag, bu_frag);
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            g_outer[i] += sg * g_inner[i];
            u_outer[i] += su * u_inner[i];
        }
    }

    // === Scatter output (per-warp, per-row gathered).
    // Per-lane D-frag rows: r_lo = lane/4, r_hi = lane/4 + 8.
    // Per-warp: only rows < tile_m written. Each warp needs its
    // OWN token_idx + k_round per row. Re-read from sorted list.
    int r_lo = lane >> 2;
    int r_hi = r_lo + 8;
    int c0   = (lane & 3) * 2;
    int c1   = c0 + 1;
    int n0   = n_base + c0;
    int n1   = n_base + c1;

    auto write_row = [&] (int row, float g_v, float u_v, int n_col) {
        if (row >= tile_m || n_col >= N) return;
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
