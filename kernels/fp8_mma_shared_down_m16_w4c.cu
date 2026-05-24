// Task #106: grouped MMA dense GEMM for qwen36 SHARED-expert down.
//
// Sibling of #103's shared dual_silu but for the down projection.
// Same W=4 cooperative-A pattern, single per-layer weight, no
// routing/sort, no silu/mul, no top_w weighting, no atomic — just
// a dense FP8 GEMM with bf16/f16 quantised activation.
//
// Targets the shared-expert down `Fp8GemvF16InLaunch` site (fires
// `fp8_gemv_blockwise_wpr_native_f16in_kernel`, part of the 6.2%
// "per-token GEMV" share in the post-#98+#99 nsys; the shared down
// fraction at prefill M=4412 is ~ a few percent of total).
//
//   Grid:  (ceil(N/32), ceil(M/16), 1)
//   Block: (128, 1, 1)
//
// Per block: [M=16, N=8 per warp × 4 = 32] tile, one MMA per K=32
// sub-block. Per-kblk a_scale fold (#99 fix). Output is direct
// f16 write to out[m, n] (no per-row weight at write).
//
// Numerical contract: equivalent to v1 GEMV up to MMA reduction
// ordering + FP8 quant rounding noise.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>
#include "fp8_mma_frag_pack.cuh"

#define WARPS_PER_BLOCK 4

namespace {
__device__ __forceinline__ unsigned char f32_to_e4m3(float v) {
    return (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
}

extern "C"
__global__ void fp8_mma_shared_down_m16_w4c_kernel(
    __half*       __restrict__ out,            // [M, N] f16 row-major
    const unsigned char* __restrict__ w_d,     // [N, K] fp8
    const float*  __restrict__ s_d,            // [N/128, K/128] f32
    const __half* __restrict__ input,          // [M, K] f16 (silu_sh)
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
    int r_lo_c = lane >> 2;
    int r_hi_c = r_lo_c + 8;

    float g_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // SHARED smem: A 512 + 4*B 1024 + ascale 64 = 1600 B
    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_a       = smem_raw;
    unsigned char* smem_b_all   = smem_a + 16 * 32;
    float*         smem_ascale  = reinterpret_cast<float*>(
                                    smem_b_all + WARPS_PER_BLOCK * 8 * 32);
    unsigned char* smem_b       = smem_b_all + warp_id * 8 * 32;

    int K_per_block  = 128;
    int num_k_blocks = K / K_per_block;

    for (int kblk = 0; kblk < num_k_blocks; ++kblk) {
        int k_base = kblk * K_per_block;

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

        float sw = s_d[scale_row * num_col_blocks + kblk];

        float g_inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            int k_off = kt * 32;
            int k_g   = k_base + k_off;

            // Cooperative A staging.
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

            // Per-warp B staging.
            #pragma unroll
            for (int n_off = 0; n_off < 8; ++n_off) {
                int n_g = n_base + n_off;
                int k_idx = k_g + lane;
                unsigned char wv = 0;
                if (n_g < N && k_idx < K) {
                    wv = w_d[(long long)n_g * K + k_idx];
                }
                smem_b[n_off * 32 + lane] = wv;
            }
            __syncthreads();

            uint32_t a_frag[4];
            uint32_t b_frag[2];
            rvllm::pack_a_frag_row_major_m16k32(smem_a, 32, a_frag, lane);
            rvllm::pack_b_frag_col_major_n8k32(smem_b, 32, b_frag, lane);
            rvllm::mma_m16n8k32_e4m3_e4m3_f32(g_inner, a_frag, b_frag);
        }

        // Task #99: per-row a_scale folded per-kblk.
        float a_lo = smem_ascale[r_lo_c];
        float a_hi = smem_ascale[r_hi_c];
        float klo = a_lo * sw;
        float khi = a_hi * sw;
        g_outer[0] += klo * g_inner[0];
        g_outer[1] += klo * g_inner[1];
        g_outer[2] += khi * g_inner[2];
        g_outer[3] += khi * g_inner[3];
    }

    // Direct (m, n) f16 write.
    int r_lo = lane >> 2;
    int r_hi = r_lo + 8;
    int c0   = (lane & 3) * 2;
    int c1   = c0 + 1;
    int n0   = n_base + c0;
    int n1   = n_base + c1;

    auto write_row = [&] (int row, float v, int n_col) {
        if (row >= tile_m || n_col >= N) return;
        int m = m_base + row;
        if (m >= M) return;
        out[(long long)m * N + n_col] = __float2half(v);
    };

    write_row(r_lo, g_outer[0], n0);
    write_row(r_lo, g_outer[1], n1);
    write_row(r_hi, g_outer[2], n0);
    write_row(r_hi, g_outer[3], n1);
}
