// Task #103: grouped MMA dual_silu for qwen36 SHARED expert.
//
// Adapts task #96's W=4 cooperative-A MMA pattern to the shared-
// expert FFN. Unlike the routed-MoE dual_silu (where each tile
// uses one expert's weights and gathers tokens via expert-sort),
// the shared expert applies a SINGLE per-layer weight matrix to
// ALL tokens — no routing, no sort. Tile mapping is the direct
// (m_block, n_block) grid coverage.
//
// Hot kernel pre-task: `fp8_gemv_dual_silu_kernel` was 4.6% of
// qwen36 prefill GPU time per the post-#98+#99 nsys (800
// instances). Per-warp scalar-reduction GEMV with 1 output per
// warp; M-direction reuse is wasted.
//
//   Grid:  (ceil(N/32), ceil(M/16), 1)
//   Block: (128, 1, 1)      ← 4 warps
//
// Per block: [M=16, N=32] output tile. All 4 warps share ONE A
// tile; each warp covers a different N=8 slice and stages its own
// B_g + B_u. Per-K=128 amax-quant of A, dual MMA for gate + up,
// per-row a_scale folded per-kblk (task #99 fix), silu(g)*u →
// f16 write to silu_sh[m, n].
//
// Numerical contract: equivalent to v1 GEMV up to MMA reduction
// ordering + FP8 quant rounding noise. Output bytes differ but
// quality preserved.

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
}

extern "C"
__global__ void fp8_mma_shared_dual_silu_m16_w4c_kernel(
    __half*       __restrict__ out_silu,       // [M, N] f16 row-major
    const unsigned char* __restrict__ w_g,     // [N, K] fp8
    const unsigned char* __restrict__ w_u,
    const float*  __restrict__ s_g,            // [N/128, K/128] f32
    const float*  __restrict__ s_u,
    const __half* __restrict__ input,          // [M, K] f16
    int M,
    int N,
    int K,
    int num_col_blocks
) {
    int tid       = threadIdx.x;
    int warp_id   = tid >> 5;
    int lane      = tid & 31;
    int n_supblk  = blockIdx.x;
    int m_block   = blockIdx.y;
    int n_super   = n_supblk * (8 * WARPS_PER_BLOCK);
    int n_base    = n_super + warp_id * 8;
    int m_base    = m_block * 16;
    if (n_base >= N || m_base >= M) return;

    int tile_m = M - m_base;
    if (tile_m > 16) tile_m = 16;

    int scale_row = n_base >> 7;

    // Pre-compute per-lane MMA D-frag row indices (per task #99).
    int r_lo_c = lane >> 2;
    int r_hi_c = r_lo_c + 8;

    float g_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float u_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // SHARED smem: A 512 + 4*B_g 1024 + 4*B_u 1024 + ascale 64 = 2624 B
    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_a       = smem_raw;
    unsigned char* smem_b_g_all = smem_a + 16 * 32;
    unsigned char* smem_b_u_all = smem_b_g_all + WARPS_PER_BLOCK * 8 * 32;
    float*         smem_ascale  = reinterpret_cast<float*>(
                                    smem_b_u_all + WARPS_PER_BLOCK * 8 * 32);
    unsigned char* smem_b_g     = smem_b_g_all + warp_id * 8 * 32;
    unsigned char* smem_b_u     = smem_b_u_all + warp_id * 8 * 32;

    int K_per_block  = 128;
    int num_k_blocks = K / K_per_block;

    for (int kblk = 0; kblk < num_k_blocks; ++kblk) {
        int k_base = kblk * K_per_block;

        // === Warp 0: per-row amax over K=128.
        if (warp_id == 0) {
            float my_amax = 0.0f;
            if (lane < 16 && lane < tile_m) {
                int m = m_base + lane;
                const __half* x_row = input + (long long)m * K;
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

        float sg = s_g[scale_row * num_col_blocks + kblk];
        float su = s_u[scale_row * num_col_blocks + kblk];

        float g_inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float u_inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            int k_off = kt * 32;
            int k_g   = k_base + k_off;

            // === Cooperative A-staging: 128 threads × 4 bytes each.
            {
                int row     = tid >> 3;
                int byte0   = (tid & 7) << 2;
                bool live   = (row < tile_m);
                int m       = m_base + row;
                float inv_s = live ? (1.0f / smem_ascale[row]) : 0.0f;
                const __half* x_row = live ? (input + (long long)m * K) : nullptr;
                unsigned int packed = 0u;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int kg = k_g + byte0 + j;
                    float xv = 0.0f;
                    if (x_row && kg < K) {
                        xv = __half2float(x_row[kg]);
                    }
                    unsigned char q = f32_to_e4m3(xv * inv_s);
                    packed |= ((unsigned int)q) << (j * 8);
                }
                *reinterpret_cast<unsigned int*>(
                    smem_a + row * 32 + byte0) = packed;
            }

            // === Stage B_g + B_u (per-warp).
            #pragma unroll
            for (int n_off = 0; n_off < 8; ++n_off) {
                int n_g = n_base + n_off;
                int k_idx = k_g + lane;
                unsigned char wg_v = 0, wu_v = 0;
                if (n_g < N && k_idx < K) {
                    wg_v = w_g[(long long)n_g * K + k_idx];
                    wu_v = w_u[(long long)n_g * K + k_idx];
                }
                smem_b_g[n_off * 32 + lane] = wg_v;
                smem_b_u[n_off * 32 + lane] = wu_v;
            }
            __syncthreads();

            uint32_t a_frag[4];
            uint32_t bg_frag[2];
            uint32_t bu_frag[2];
            rvllm::pack_a_frag_row_major_m16k32(smem_a,   32, a_frag,  lane);
            rvllm::pack_b_frag_col_major_n8k32(smem_b_g,  32, bg_frag, lane);
            rvllm::pack_b_frag_col_major_n8k32(smem_b_u,  32, bu_frag, lane);

            rvllm::mma_m16n8k32_e4m3_e4m3_f32(g_inner, a_frag, bg_frag);
            rvllm::mma_m16n8k32_e4m3_e4m3_f32(u_inner, a_frag, bu_frag);
        }

        // Task #99: per-row a_scale folded per-kblk.
        float a_lo = smem_ascale[r_lo_c];
        float a_hi = smem_ascale[r_hi_c];
        float klo_g = a_lo * sg;
        float klo_u = a_lo * su;
        float khi_g = a_hi * sg;
        float khi_u = a_hi * su;
        g_outer[0] += klo_g * g_inner[0];
        g_outer[1] += klo_g * g_inner[1];
        u_outer[0] += klo_u * u_inner[0];
        u_outer[1] += klo_u * u_inner[1];
        g_outer[2] += khi_g * g_inner[2];
        g_outer[3] += khi_g * g_inner[3];
        u_outer[2] += khi_u * u_inner[2];
        u_outer[3] += khi_u * u_inner[3];
    }

    // === Scatter output (direct (m, n) mapping, no gather).
    int r_lo = lane >> 2;
    int r_hi = r_lo + 8;
    int c0   = (lane & 3) * 2;
    int c1   = c0 + 1;
    int n0   = n_base + c0;
    int n1   = n_base + c1;

    auto write_row = [&] (int row, float g_v, float u_v, int n_col) {
        if (row >= tile_m || n_col >= N) return;
        int m = m_base + row;
        if (m >= M) return;
        long long off = (long long)m * N + n_col;
        out_silu[off] = __float2half(silu_f(g_v) * u_v);
    };

    write_row(r_lo, g_outer[0], u_outer[0], n0);
    write_row(r_lo, g_outer[1], u_outer[1], n1);
    write_row(r_hi, g_outer[2], u_outer[2], n0);
    write_row(r_hi, g_outer[3], u_outer[3], n1);
}
