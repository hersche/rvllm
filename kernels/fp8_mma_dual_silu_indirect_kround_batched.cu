// Task #93: TensorCore-MMA-based FP8 dual_silu indirect (k_round batched).
//
// Drop-in alternative to
//   fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_kround_batched_kernel
// that uses `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`
// instead of the per-warp-per-output GEMV reduction. Targets the
// qwen36 MoE prefill hot path (88.9% of prefill GPU time per the
// 2026-05-23 nsys profile).
//
// First-cut semantics (CORRECTNESS first, perf optimization second):
//
//   Grid:  (ceil(N / 8), M, top_k)         ← 1 token per block
//   Block: (32, 1, 1)                       ← 1 warp
//
// Each block computes the [N=8] N-tile for ONE (token, k_round).
// Internally the MMA tile is [M=16, N=8] but only row 0 is the active
// token; rows 1..15 are zero-staged so the MMA produces zeros there
// which are discarded at write time.
//
// This wastes 15/16 of the MMA throughput per block. The TRUE win
// requires a pre-pass that sorts (token, k_round) pairs by routed
// expert id so a tile of 16 rows shares one expert; that is parked
// as the follow-up. The first-cut kernel exists to:
//   1. Validate the FP8 × FP8 MMA path compiles + executes on sm_121
//      against real qwen36 weights;
//   2. Establish A/B baseline (expected SLOWER than the GEMV — see
//      perf analysis in the commit message);
//   3. Stand as the foundation for the M-direction MMA reuse work.
//
// Input quantization: inline per-K=128 amax → 448-scale → e4m3 quant
// happens inside the K loop, per active row.
//
// Output layout: `[k_round, M, N]` (k_round-major) — identical to
// the GEMV variant so the down-projection consumer reads the same
// bytes.

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

}  // anon namespace

extern "C"
__global__ void fp8_mma_dual_silu_indirect_kround_batched_kernel(
    __half*       __restrict__ out_silu,         // [top_k, M, N] f16
    const unsigned char* __restrict__ base_w_g,  // [num_experts, N, K] fp8
    const unsigned char* __restrict__ base_w_u,
    const float*  __restrict__ base_s_g,         // [num_experts, N/128, K/128] f32
    const float*  __restrict__ base_s_u,
    const __half* __restrict__ input,            // [M, K] f16
    const int*    __restrict__ top_idx,          // [M, top_k] i32
    long long w_stride,
    long long s_stride,
    int M, int N, int K,
    int num_col_blocks,
    int top_k
) {
    int lane     = threadIdx.x & 31;
    int n_block  = blockIdx.x;     // N-tile id, covers [n_base, n_base+8)
    int m        = blockIdx.y;     // single token row
    int k_round  = blockIdx.z;
    if (n_block * 8 >= N || m >= M || k_round >= top_k) return;
    int n_base   = n_block * 8;

    int e = top_idx[(long long)m * top_k + k_round];

    const unsigned char* w_g_base = base_w_g + (long long)e * w_stride;
    const unsigned char* w_u_base = base_w_u + (long long)e * w_stride;
    const float*         s_g_base = base_s_g + (long long)e * s_stride;
    const float*         s_u_base = base_s_u + (long long)e * s_stride;

    // n_base..n_base+7 fall within one 128-N-block (callers must
    // ensure N is a multiple of 128 — qwen36 moe_int=512 satisfies).
    int scale_row = n_base >> 7;

    const __half* x_row = input + (long long)m * K;

    // F32 outer accumulators (per-lane MMA D fragment).
    float g_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float u_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Shared smem: 16 rows × 32 K bytes for A + 8 rows × 32 for B_g
    // + 8 rows × 32 for B_u = 512 + 256 + 256 = 1024 bytes.
    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_a   = smem_raw;
    unsigned char* smem_b_g = smem_a   + 16 * 32;
    unsigned char* smem_b_u = smem_b_g +  8 * 32;

    // ---- One-time: zero smem_a rows 1..15. Row 0 is rewritten per
    //      kt below. 32 lanes cooperatively zero 15 × 32 = 480 bytes.
    if (lane < 15) {
        uint64_t* row = reinterpret_cast<uint64_t*>(
            smem_a + (lane + 1) * 32);
        row[0] = 0ULL;
        row[1] = 0ULL;
        row[2] = 0ULL;
        row[3] = 0ULL;
    }
    __syncwarp();

    int K_per_block  = 128;
    int num_k_blocks = K / K_per_block;

    for (int kblk = 0; kblk < num_k_blocks; ++kblk) {
        int k_base = kblk * K_per_block;

        // ---- Compute per-row amax over the K=128 block (row 0 only). ----
        float amax = 0.0f;
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            int k = k_base + kt * 32 + lane;
            if (k < K) {
                float xv = __half2float(x_row[k]);
                amax = fmaxf(amax, fabsf(xv));
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        }
        float a_scale = amax / 448.0f;
        if (a_scale == 0.0f) a_scale = 1e-30f;
        float a_inv   = 1.0f / a_scale;

        // ---- Weight blockscales (one per (n_block, kblk)). ----
        float sg = s_g_base[scale_row * num_col_blocks + kblk];
        float su = s_u_base[scale_row * num_col_blocks + kblk];

        // ---- Per-K=128 inner accumulators ----
        float g_inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float u_inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            int k_off = kt * 32;
            int k_g   = k_base + k_off;

            // Stage A row 0 (32 bytes, lane writes 1 byte).
            float xv = (k_g + lane < K)
                ? __half2float(x_row[k_g + lane]) : 0.0f;
            smem_a[0 * 32 + lane] = f32_to_e4m3(xv * a_inv);

            // Stage B_g + B_u (8 rows × 32 bytes).
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

            uint32_t a_frag[4];
            uint32_t bg_frag[2];
            uint32_t bu_frag[2];
            rvllm::pack_a_frag_row_major_m16k32(smem_a,   32, a_frag,  lane);
            rvllm::pack_b_frag_col_major_n8k32(smem_b_g,  32, bg_frag, lane);
            rvllm::pack_b_frag_col_major_n8k32(smem_b_u,  32, bu_frag, lane);

            rvllm::mma_m16n8k32_e4m3_e4m3_f32(g_inner, a_frag, bg_frag);
            rvllm::mma_m16n8k32_e4m3_e4m3_f32(u_inner, a_frag, bu_frag);
        }

        // Apply K=128 blockscale to inner accumulators, fold into outer.
        float kg = a_scale * sg;
        float ku = a_scale * su;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            g_outer[i] += kg * g_inner[i];
            u_outer[i] += ku * u_inner[i];
        }
    }

    // Per-lane D-frag layout (m16n8):
    //   d[0]: row (lane/4),    col (lane%4)*2 + 0
    //   d[1]: row (lane/4),    col (lane%4)*2 + 1
    //   d[2]: row (lane/4+8),  col (lane%4)*2 + 0
    //   d[3]: row (lane/4+8),  col (lane%4)*2 + 1
    //
    // Only row 0 is real (rows 1..15 carry zero-stage A → MMA output
    // is zero there; discard). lane/4 == 0 → lanes 0..3 own d[0]+d[1]
    // for the 4 (m=0, col_pair) entries.
    if ((lane >> 2) == 0) {
        int c0 = (lane & 3) * 2;
        int c1 = c0 + 1;
        int n0 = n_base + c0;
        int n1 = n_base + c1;
        long long base = ((long long)k_round * M + m) * N;
        if (n0 < N) {
            out_silu[base + n0] = __float2half(silu_f(g_outer[0]) * u_outer[0]);
        }
        if (n1 < N) {
            out_silu[base + n1] = __float2half(silu_f(g_outer[1]) * u_outer[1]);
        }
    }
}
