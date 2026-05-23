// Task #97: W=4 grouped MMA dual_silu with cooperative A AND B staging.
//
// Sibling of task #96's `fp8_mma_dual_silu_grouped_m16_w4c_kernel`
// (cooperative A only). Adds u64-vector B-staging:
//
//   Old (w4c): per-warp B_g/B_u stage runs 8 unrolled iterations,
//     each iter: 32 lanes × 1 byte = 32 bytes per row. 8 rows total.
//   New (w4cb): per-warp B_g/B_u stage runs 1 pass of 32 lanes ×
//     8 bytes (u64). Lane mapping (n_row, k_chunk) = (lane/4, lane%4),
//     each lane reads 8 contiguous gmem bytes via aligned u64 load
//     and writes 8 contiguous smem bytes via aligned u64 store.
//
// Same total bytes moved; fewer instructions; better LDG/STS
// throughput. Coalescing pattern preserved (4 lanes per row read
// contiguous 32-byte segments).
//
// Output bytes byte-equivalent to w4c modulo any GPU-driver kernel
// rearrangement (no math change).

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
__global__ void fp8_mma_dual_silu_grouped_m16_w4cb_kernel(
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

    float g_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float u_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};

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

    if (warp_id == 0 && lane < 16) {
        smem_tok[lane] = (lane < tile_m)
            ? sorted_per_expert[sorted_base + lane * 2 + 0] : -1;
    }
    __syncthreads();

    int K_per_block  = 128;
    int num_k_blocks = K / K_per_block;

    // Lane decomposition for cooperative B-staging (u64 per lane):
    //   n_row  = lane / 4  ∈ [0, 8)
    //   chunk  = lane % 4  ∈ [0, 4)  → k_byte_off = chunk * 8 ∈ {0,8,16,24}
    int b_n_row = lane >> 2;
    int b_kchk  = (lane & 3) << 3;   // 0, 8, 16, 24

    for (int kblk = 0; kblk < num_k_blocks; ++kblk) {
        int k_base = kblk * K_per_block;

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

            // === Cooperative A-staging (same as w4c). ===
            {
                int row     = tid >> 3;
                int byte0   = (tid & 7) << 2;
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
                *reinterpret_cast<unsigned int*>(
                    smem_a + row * 32 + byte0) = packed;
            }

            // === Cooperative B-staging via u64 per lane. ===
            // Each lane stages 8 bytes (one chunk of 8 K-positions)
            // of one B row, for both B_g and B_u. 32 lanes cover
            // 8 rows × 4 chunks = the entire 8×32 tile in 1 pass.
            {
                int n_g    = n_base + b_n_row;
                int k_idx  = k_g + b_kchk;
                unsigned long long wg8 = 0ULL;
                unsigned long long wu8 = 0ULL;
                if (n_g < N && k_idx + 7 < K) {
                    wg8 = *reinterpret_cast<const unsigned long long*>(
                        w_g_base + (long long)n_g * K + k_idx);
                    wu8 = *reinterpret_cast<const unsigned long long*>(
                        w_u_base + (long long)n_g * K + k_idx);
                } else if (n_g < N) {
                    // Partial K tail (rare on qwen36; K=2048 aligned).
                    unsigned char tg[8] = {0}, tu[8] = {0};
                    for (int j = 0; j < 8 && k_idx + j < K; ++j) {
                        tg[j] = w_g_base[(long long)n_g * K + k_idx + j];
                        tu[j] = w_u_base[(long long)n_g * K + k_idx + j];
                    }
                    for (int j = 0; j < 8; ++j) {
                        wg8 |= ((unsigned long long)tg[j]) << (j * 8);
                        wu8 |= ((unsigned long long)tu[j]) << (j * 8);
                    }
                }
                *reinterpret_cast<unsigned long long*>(
                    smem_b_g + b_n_row * 32 + b_kchk) = wg8;
                *reinterpret_cast<unsigned long long*>(
                    smem_b_u + b_n_row * 32 + b_kchk) = wu8;
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

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            g_outer[i] += sg * g_inner[i];
            u_outer[i] += su * u_inner[i];
        }
    }

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
