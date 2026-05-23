// Task #98: grouped MMA `down` projection sibling for qwen36 MoE prefill.
//
// Adapts task #94+#95+#96's grouped MMA dual_silu pattern to the
// down projection (currently
// `fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add_kround_batched_kernel`,
// 27% of prefill GPU time per the 2026-05-23 nsys profile).
//
// Key differences vs dual_silu:
//   * Input: `input_kround[top_k, M_full, K_in]` f16 — k_round-major.
//     Each (token, k_round) pair gathers its OWN row at
//     `input_kround + k_round*M_full*K_in + token*K_in`.
//   * Output: `acc_f32[M_full, N_down]` f32 — needs atomicAdd because
//     each (m, n) receives contributions from top_k=8 different tiles
//     (one per (expert, k_round) pair serving that token). The
//     legacy GEMV path avoids atomics by serialising k_rounds inside
//     one warp; the grouped MMA path can't (each tile is one
//     expert, no expert-wise reuse to amortise).
//   * Per-row top_w lookup: `top_w[token, k_round]` f32 multiplied
//     into the per-row output before atomic-add.
//
// Tile shape identical to the dual_silu w4cb (W=4 multi-warp, M=16 ×
// N=32 effective output per block, 4 warps each owning [M=16, N=8]).
// Per K=128 amax+quant of input is per-(token,k_round) row.
//
// Grid: (ceil(N_down/32), num_total_tiles, 1)
// Block: (128, 1, 1)

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
__global__ void fp8_mma_down_grouped_m16_w4_kernel(
    float*        __restrict__ acc_f32,           // [M_full, N_down] f32 RMW
    const unsigned char* __restrict__ base_w,     // [E, N_down, K_in] fp8
    const float*  __restrict__ base_s,            // [E, N_down/128, K_in/128] f32
    const __half* __restrict__ input_kround,      // [top_k, M_full, K_in] f16
    const float*  __restrict__ top_w,             // [M_full, top_k] f32
    const int*    __restrict__ sorted_per_expert,
    const int*    __restrict__ tile_descriptors,
    long long w_stride,
    long long s_stride,
    int M_full,
    int N_down,
    int K_in,
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
    if (n_base >= N_down) return;

    int e            = tile_descriptors[tile_idx * 3 + 0];
    int m_off_in_exp = tile_descriptors[tile_idx * 3 + 1];
    int tile_m       = tile_descriptors[tile_idx * 3 + 2];
    if (tile_m <= 0) return;

    const unsigned char* w_base = base_w + (long long)e * w_stride;
    const float*         s_base = base_s + (long long)e * s_stride;
    int scale_row = n_base >> 7;

    int sorted_base = (e * max_per_expert + m_off_in_exp) * 2;

    // Pre-compute per-lane MMA output row indices.
    int r_lo_c = lane >> 2;          // row of d[0], d[1]
    int r_hi_c = r_lo_c + 8;         // row of d[2], d[3]

    float g_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // SHARED smem layout:
    //   smem_a:        [16, 32] = 512 B
    //   smem_b_w[W]:   [W*8, 32] = W*256 B = 1024 B
    //   smem_ascale:   [16] f32  = 64 B
    //   smem_tok:      [16] i32  = 64 B
    //   smem_kr:       [16] i32  = 64 B
    //   smem_topw:     [16] f32  = 64 B
    // Total at W=4: 512 + 1024 + 64 + 64 + 64 + 64 = 1792 B
    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_a       = smem_raw;
    unsigned char* smem_b_w_all = smem_a + 16 * 32;
    float*         smem_ascale  = reinterpret_cast<float*>(
                                    smem_b_w_all + WARPS_PER_BLOCK * 8 * 32);
    int*           smem_tok     = reinterpret_cast<int*>(
                                    reinterpret_cast<unsigned char*>(smem_ascale) + 16 * 4);
    int*           smem_kr      = smem_tok + 16;
    float*         smem_topw    = reinterpret_cast<float*>(smem_kr + 16);
    unsigned char* smem_b_w     = smem_b_w_all + warp_id * 8 * 32;

    // Broadcast per-row metadata via smem (warp 0).
    if (warp_id == 0 && lane < 16) {
        if (lane < tile_m) {
            int tok = sorted_per_expert[sorted_base + lane * 2 + 0];
            int kr  = sorted_per_expert[sorted_base + lane * 2 + 1];
            smem_tok[lane]   = tok;
            smem_kr[lane]    = kr;
            smem_topw[lane]  = top_w[(long long)tok * top_k + kr];
        } else {
            smem_tok[lane]   = -1;
            smem_kr[lane]    = -1;
            smem_topw[lane]  = 0.0f;
        }
    }
    __syncthreads();

    int K_per_block  = 128;
    int num_k_blocks = K_in / K_per_block;

    for (int kblk = 0; kblk < num_k_blocks; ++kblk) {
        int k_base = kblk * K_per_block;

        // Per-row amax over K=128 (warp 0).
        if (warp_id == 0) {
            float my_amax = 0.0f;
            if (lane < 16) {
                int tok = smem_tok[lane];
                int kr  = smem_kr[lane];
                if (tok >= 0 && kr >= 0) {
                    const __half* x_row = input_kround
                        + (long long)kr * M_full * K_in
                        + (long long)tok * K_in;
                    #pragma unroll 4
                    for (int k = 0; k < K_per_block; ++k) {
                        int kg = k_base + k;
                        if (kg < K_in) {
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

        float sw = s_base[scale_row * num_col_blocks + kblk];

        float g_inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            int k_off = kt * 32;
            int k_g   = k_base + k_off;

            // === Cooperative A-staging (128 threads, 4 bytes each).
            {
                int row     = tid >> 3;
                int byte0   = (tid & 7) << 2;
                int tok     = smem_tok[row];
                int kr      = smem_kr[row];
                bool live   = (row < tile_m) && (tok >= 0) && (kr >= 0);
                float inv_s = live ? (1.0f / smem_ascale[row]) : 0.0f;
                const __half* x_row = live
                    ? (input_kround + (long long)kr * M_full * K_in
                                    + (long long)tok * K_in)
                    : nullptr;
                unsigned int packed = 0u;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int kg = k_g + byte0 + j;
                    float xv = 0.0f;
                    if (x_row && kg < K_in) {
                        xv = __half2float(x_row[kg]);
                    }
                    unsigned char q = f32_to_e4m3(xv * inv_s);
                    packed |= ((unsigned int)q) << (j * 8);
                }
                *reinterpret_cast<unsigned int*>(
                    smem_a + row * 32 + byte0) = packed;
            }

            // === Per-warp B-staging (8 rows × 32 bytes). ===
            #pragma unroll
            for (int n_off = 0; n_off < 8; ++n_off) {
                int n_g = n_base + n_off;
                int k_idx = k_g + lane;
                unsigned char wb = 0;
                if (n_g < N_down && k_idx < K_in) {
                    wb = w_base[(long long)n_g * K_in + k_idx];
                }
                smem_b_w[n_off * 32 + lane] = wb;
            }
            __syncthreads();

            uint32_t a_frag[4];
            uint32_t b_frag[2];
            rvllm::pack_a_frag_row_major_m16k32(smem_a,   32, a_frag, lane);
            rvllm::pack_b_frag_col_major_n8k32(smem_b_w,  32, b_frag, lane);
            rvllm::mma_m16n8k32_e4m3_e4m3_f32(g_inner, a_frag, b_frag);
        }

        // Per-row a_scale for THIS kblk (varies per kblk; previously
        // only the last-kblk value was applied at write, which under-
        // counted contributions from kblks where a_scale differs).
        float a_lo = smem_ascale[r_lo_c];
        float a_hi = smem_ascale[r_hi_c];
        float klo = a_lo * sw;
        float khi = a_hi * sw;
        g_outer[0] += klo * g_inner[0];
        g_outer[1] += klo * g_inner[1];
        g_outer[2] += khi * g_inner[2];
        g_outer[3] += khi * g_inner[3];
    }

    // === Scatter via atomicAdd. ===
    // Per-lane D-frag layout (m16n8):
    //   d[0]: row (lane/4),    col (lane%4)*2 + 0
    //   d[1]: row (lane/4),    col (lane%4)*2 + 1
    //   d[2]: row (lane/4+8),  col (lane%4)*2 + 0
    //   d[3]: row (lane/4+8),  col (lane%4)*2 + 1
    //
    // Apply per-row a_scale + topw at scatter. Output is f32
    // routed_sum[token, n] += topw * a_scale * d_value.
    int r_lo = lane >> 2;
    int r_hi = r_lo + 8;
    int c0   = (lane & 3) * 2;
    int c1   = c0 + 1;
    int n0   = n_base + c0;
    int n1   = n_base + c1;

    // a_scale already folded into g_outer per-kblk above. Apply only
    // top_w at write time.
    auto write_row = [&] (int row, float d_v, int n_col) {
        if (row >= tile_m || n_col >= N_down) return;
        int tok = smem_tok[row];
        if (tok < 0) return;
        float t_w = smem_topw[row];
        float contrib = t_w * d_v;
        long long off = (long long)tok * N_down + n_col;
        atomicAdd(&acc_f32[off], contrib);
    };

    write_row(r_lo, g_outer[0], n0);
    write_row(r_lo, g_outer[1], n1);
    write_row(r_hi, g_outer[2], n0);
    write_row(r_hi, g_outer[3], n1);
}
