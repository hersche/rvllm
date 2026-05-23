// Task #96: W=4 grouped MMA dual_silu with COOPERATIVE A-staging.
//
// Sibling of `fp8_mma_dual_silu_grouped_m16_w4_kernel` (task #95).
// Same shape (4 warps per block, [M=16, N=32] output area, shared A
// tile across warps), but A-staging is distributed across all 128
// threads instead of warp 0 serial.
//
// Per K=32 sub-tile staging breakdown:
//   * Old (task #95): warp 0 lane 0..15 sequentially write 32 bytes
//     each (16 lanes × 32 inner-loop iter = 512 byte-writes serial).
//   * New (this task): 128 threads each quantize+write 4 contiguous
//     bytes (row = tid/8, byte_off = (tid%8)*4). Single pass.
//
// Per-row metadata (token_idx[16]) is broadcast via smem so all
// warps can gather their assigned row's input. amax compute also
// shifted to a per-row warp-parallel form using a 16-lane sub-group
// of warp 0 (each lane handles its row, walks K=128, warp-reduce is
// implicit since lanes already write smem_ascale[lane]).
//
// Output layout identical to the W=4 variant.

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
__global__ void fp8_mma_dual_silu_grouped_m16_w4c_kernel(
    __half*       __restrict__ out_silu,
    const unsigned char* __restrict__ base_w_g,
    const unsigned char* __restrict__ base_w_u,
    const float*  __restrict__ base_s_g,
    const float*  __restrict__ base_s_u,
    const __half* __restrict__ input,
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
    int n_supblk  = blockIdx.x;
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

    int sorted_base = (e * max_per_expert + m_off_in_exp) * 2;

    int r_lo_c = lane >> 2;
    int r_hi_c = r_lo_c + 8;
    float g_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float u_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // SHARED smem layout (with token_idx broadcast):
    //   smem_a:        [16, 32] = 512 B
    //   smem_b_g[W]:   [W*8, 32] = W*256 B
    //   smem_b_u[W]:   [W*8, 32] = W*256 B
    //   smem_ascale:   [16] f32  = 64 B
    //   smem_token_idx:[16] i32  = 64 B
    // Total at W=4: 512 + 1024 + 1024 + 64 + 64 = 2688 B
    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_a       = smem_raw;
    unsigned char* smem_b_g_all = smem_a + 16 * 32;
    unsigned char* smem_b_u_all = smem_b_g_all + WARPS_PER_BLOCK * 8 * 32;
    float*         smem_ascale  = reinterpret_cast<float*>(
                                    smem_b_u_all + WARPS_PER_BLOCK * 8 * 32);
    int*           smem_tok     = reinterpret_cast<int*>(
                                    reinterpret_cast<unsigned char*>(smem_ascale) + 16 * 4);
    unsigned char* smem_b_g     = smem_b_g_all + warp_id * 8 * 32;
    unsigned char* smem_b_u     = smem_b_u_all + warp_id * 8 * 32;

    // === Broadcast token_idx[0..15] via smem. Warp 0 lanes 0..15 do it.
    if (warp_id == 0 && lane < 16) {
        smem_tok[lane] = (lane < tile_m)
            ? sorted_per_expert[sorted_base + lane * 2 + 0] : -1;
    }
    __syncthreads();

    int K_per_block  = 128;
    int num_k_blocks = K / K_per_block;

    for (int kblk = 0; kblk < num_k_blocks; ++kblk) {
        int k_base = kblk * K_per_block;

        // === Per-row amax over K=128. Warp 0 lanes 0..15 do it,
        //     each lane handles its own row sequentially.
        if (warp_id == 0) {
            float my_amax = 0.0f;
            if (lane < 16) {
                int tok = smem_tok[lane];
                if (tok >= 0) {
                    const __half* x_row = input + (long long)tok * K;
                    #pragma unroll 4
                    for (int k = 0; k < K_per_block; ++k) {
                        int kg = k_base + k;
                        if (kg < K) {
                            float xv = __half2float(x_row[kg]);
                            my_amax = fmaxf(my_amax, fabsf(xv));
                        }
                    }
                }
            }
            float a_scale = my_amax / 448.0f;
            if (a_scale == 0.0f) a_scale = 1e-30f;
            if (lane < 16) smem_ascale[lane] = a_scale;
        }
        __syncthreads();

        float sg = s_g_base[scale_row * num_col_blocks + kblk];
        float su = s_u_base[scale_row * num_col_blocks + kblk];

        float g_inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float u_inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            int k_off = kt * 32;
            int k_g   = k_base + k_off;

            // === Cooperative A-staging: all 128 threads write 4 bytes.
            //     row = tid / 8, byte_off = (tid % 8) * 4.
            {
                int row     = tid >> 3;          // tid / 8, in [0, 16)
                int byte0   = (tid & 7) << 2;    // (tid % 8) * 4, in {0,4,...,28}
                int tok     = smem_tok[row];
                bool live   = (row < tile_m) && (tok >= 0);
                float inv_s = live ? (1.0f / smem_ascale[row]) : 0.0f;
                const __half* x_row = live ? (input + (long long)tok * K) : nullptr;
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
                // Single 4-byte aligned write per thread.
                *reinterpret_cast<unsigned int*>(
                    smem_a + row * 32 + byte0) = packed;
            }

            // === Stage B_g + B_u (per-warp).
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

            uint32_t a_frag[4];
            uint32_t bg_frag[2];
            uint32_t bu_frag[2];
            rvllm::pack_a_frag_row_major_m16k32(smem_a,   32, a_frag,  lane);
            rvllm::pack_b_frag_col_major_n8k32(smem_b_g,  32, bg_frag, lane);
            rvllm::pack_b_frag_col_major_n8k32(smem_b_u,  32, bu_frag, lane);

            rvllm::mma_m16n8k32_e4m3_e4m3_f32(g_inner, a_frag, bg_frag);
            rvllm::mma_m16n8k32_e4m3_e4m3_f32(u_inner, a_frag, bu_frag);
        }

        // Task #99 numerical fix: per-row a_scale folded per-kblk.
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

    // === Scatter output (per-warp, per-row gathered).
    int r_lo = lane >> 2;
    int r_hi = r_lo + 8;
    int c0   = (lane & 3) * 2;
    int c1   = c0 + 1;
    int n0   = n_base + c0;
    int n1   = n_base + c1;

    // a_scale already folded into g_outer per-kblk above.
    auto write_row = [&] (int row, float g_v, float u_v, int n_col) {
        if (row >= tile_m || n_col >= N) return;
        int tok = sorted_per_expert[sorted_base + row * 2 + 0];
        int kr  = sorted_per_expert[sorted_base + row * 2 + 1];
        long long off = ((long long)kr * M_full + tok) * N + n_col;
        out_silu[off] = __float2half(silu_f(g_v) * u_v);
    };

    write_row(r_lo, g_outer[0], u_outer[0], n0);
    write_row(r_lo, g_outer[1], u_outer[1], n1);
    write_row(r_hi, g_outer[2], u_outer[2], n0);
    write_row(r_hi, g_outer[3], u_outer[3], n1);
}
