// Task #144 — FP8 GEMV variant using TensorCore MMA tiles for M-batched
// decode + verify paths on gemma4-nvfp4 (and any other FP8-blockscale
// caller that fits the same ABI).
//
// Sibling of `fp8_gemv_blockwise_wpr_native_f16in_kernel` from
// `kernels/fp8_gemv.cu`. ABI is byte-compatible (same args, same
// output layout), but the inner GEMV reduction is replaced by
// `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`
// fragments. The kernel processes up to M=16 input rows per launch:
//
//   * M=1  decode hot path → 15 of 16 fragment rows zero-padded; the
//     win is launch-overhead amortization across 4 warps + better K-dim
//     pipelining via cooperative-A staging (same as the
//     `fp8_mma_dual_silu_grouped_m16_w4c` task #96 pattern).
//   * M=8  verify-batch path (gemma4-nvfp4 K=7 spec → K+1 = up to 8
//     input rows fed to the verify forward) — half-utilization of the
//     fragment, ~6-8× MMA throughput vs scalar GEMV at the same launch
//     count.
//   * M=16 future continuous-batch / multi-request fusion — full
//     fragment utilization.
//
// Per-block: 4 warps, [M=16 (active rows), N=32 (= 4 warps × 8 cols)]
// output area, K-dim swept in K_per_block=128 outer iterations × 4
// MMA tiles of K=32 each. Cooperative A-staging across all 128
// threads (task #96 pattern). Per-row amax → per-row a_scale folded
// into the per-kblk accumulator (task #99 numerical-correctness fix).
//
// Cross-model invariant note: this kernel is ADDITIVE — it does not
// replace any existing dispatch. The caller decides per-call whether
// to use it (typically gated behind an env knob during the bring-up
// + A/B period). The shared kernels in `fp8_gemv.cu` are unchanged.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>
#include "fp8_mma_frag_pack.cuh"

#define WARPS_PER_BLOCK 4

namespace {
__device__ __forceinline__ unsigned char f32_to_e4m3(float v) {
    return (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float e4m3_to_f32(unsigned char b) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    __half h = *reinterpret_cast<__half*>(&hr);
    return __half2float(h);
}
}

// extern "C" launch ABI:
//   output:        [M, N] f16
//   weight:        [N, K] fp8 e4m3 (packed bytes; matches fp8_gemv base)
//   scale:         [ceil(N/128), num_col_blocks] f32 (matches base)
//   input:         [M, K] f16
//   M:             active input rows (1 ≤ M ≤ 16)
//   N:             output cols (= weight rows)
//   K:             contraction dim (multiple of 128 expected on the
//                  Gemma 4 / Qwen 3.x decode paths; remainder beyond
//                  K%128 falls back to zero contribution because the
//                  outer loop is num_k_blocks = K / 128, matching the
//                  base scale layout. A short tail loop covers ragged
//                  K, mirroring the base kernel's behaviour.)
//   num_col_blocks: scale-row stride along K (= K / 128 typically).
//
// Grid:  (ceil(N / 32), 1)
// Block: 128 threads (= WARPS_PER_BLOCK × 32)
// Smem:  [16 × 32] (A) + 4 × [8 × 32] (B per warp) + [16] f32 (a_scale)
//        = 512 + 1024 + 64 = 1600 B
extern "C"
__global__ void fp8_gemv_mma_m8_w4c_kernel(
    __half*       __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float*  __restrict__ scale,
    const __half* __restrict__ input,
    int M,
    int N,
    int K,
    int num_col_blocks
) {
    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int n_supblk = blockIdx.x;
    int n_super  = n_supblk * (8 * WARPS_PER_BLOCK);
    int n_base   = n_super + warp_id * 8;
    if (n_base >= N) return;

    int scale_row = n_base >> 7;
    int M_eff = M < 16 ? M : 16;

    // Per-lane accumulators: [r_lo, r_hi] × [n_lo, n_hi] = 4 outputs.
    float acc_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // SHARED smem layout:
    //   smem_a:        [16, 32] = 512 B (cooperative; all warps read)
    //   smem_b[W]:     [W*8, 32] = 256*W B (per-warp B tile)
    //   smem_ascale:   [16] f32  = 64 B
    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_a       = smem_raw;
    unsigned char* smem_b_all   = smem_a + 16 * 32;
    float*         smem_ascale  = reinterpret_cast<float*>(
                                    smem_b_all + WARPS_PER_BLOCK * 8 * 32);
    unsigned char* smem_b       = smem_b_all + warp_id * 8 * 32;

    int K_per_block  = 128;
    int num_k_blocks = K / K_per_block;

    int r_lo_c = lane >> 2;
    int r_hi_c = r_lo_c + 8;

    for (int kblk = 0; kblk < num_k_blocks; ++kblk) {
        int k_base = kblk * K_per_block;

        // === Per-row amax over K=128. Warp 0 lanes 0..15 do it,
        //     each lane handles its own row sequentially.
        if (warp_id == 0) {
            float my_amax = 0.0f;
            if (lane < M_eff) {
                const __half* x_row = input + (long long)lane * K;
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
            if (lane < 16) smem_ascale[lane] = (lane < M_eff) ? a_scale : 1.0f;
        }
        __syncthreads();

        float sw = scale[scale_row * num_col_blocks + kblk];

        float acc_inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            int k_off = kt * 32;
            int k_g   = k_base + k_off;

            // === Cooperative A-staging: all 128 threads write 4 bytes.
            //     row = tid / 8, byte_off = (tid % 8) * 4.
            //     Rows ≥ M_eff stage zeros so the MMA's contribution
            //     from those lanes is identically 0.
            {
                int row     = tid >> 3;          // tid / 8, in [0, 16)
                int byte0   = (tid & 7) << 2;    // (tid % 8) * 4
                bool live   = (row < M_eff);
                float inv_s = live ? (1.0f / smem_ascale[row]) : 0.0f;
                const __half* x_row = live ? (input + (long long)row * K) : nullptr;
                unsigned int packed = 0u;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int kg = k_g + byte0 + j;
                    float xv = 0.0f;
                    if (x_row && kg < K) {
                        xv = __half2float(x_row[kg]);
                    }
                    unsigned char q = live ? f32_to_e4m3(xv * inv_s) : 0u;
                    packed |= ((unsigned int)q) << (j * 8);
                }
                // Single 4-byte aligned write per thread.
                *reinterpret_cast<unsigned int*>(
                    smem_a + row * 32 + byte0) = packed;
            }

            // === Stage B (per-warp, this warp's 8 cols).
            #pragma unroll
            for (int n_off = 0; n_off < 8; ++n_off) {
                int n_g  = n_base + n_off;
                int k_idx = k_g + lane;
                unsigned char wv = 0;
                if (n_g < N && k_idx < K) {
                    wv = weight[(long long)n_g * K + k_idx];
                }
                smem_b[n_off * 32 + lane] = wv;
            }
            __syncthreads();

            uint32_t a_frag[4];
            uint32_t b_frag[2];
            rvllm::pack_a_frag_row_major_m16k32(smem_a, 32, a_frag, lane);
            rvllm::pack_b_frag_col_major_n8k32(smem_b, 32, b_frag, lane);
            rvllm::mma_m16n8k32_e4m3_e4m3_f32(acc_inner, a_frag, b_frag);
        }

        // Task #99 numerical-correctness pattern: per-row a_scale
        // folded per-kblk into the outer accumulator. Each lane owns
        // the (r_lo, r_hi) row pair for its 2 N-columns.
        float a_lo = smem_ascale[r_lo_c];
        float a_hi = smem_ascale[r_hi_c];
        float klo = a_lo * sw;
        float khi = a_hi * sw;
        acc_outer[0] += klo * acc_inner[0];
        acc_outer[1] += klo * acc_inner[1];
        acc_outer[2] += khi * acc_inner[2];
        acc_outer[3] += khi * acc_inner[3];
    }

    // === Ragged-K tail: matches the base kernel's behaviour of
    //     summing into the same row outputs. A naive single-warp scan
    //     here would cost smem layout duplication; instead we do a
    //     simple per-(row, n) scalar loop for the leftover columns
    //     [num_k_blocks * 128, K). This adds at most K%128 scalar ops
    //     per (row, n), gated on K not being a multiple of 128.
    int k_tail_start = num_k_blocks * K_per_block;
    if (k_tail_start < K) {
        int r_lo = lane >> 2;
        int r_hi = r_lo + 8;
        int c0   = (lane & 3) * 2;
        int c1   = c0 + 1;
        int n0   = n_base + c0;
        int n1   = n_base + c1;
        int sc   = k_tail_start >> 7;
        float sw_tail = (sc < num_col_blocks)
            ? scale[scale_row * num_col_blocks + sc] : 0.0f;
        float tail_lo_n0 = 0.0f, tail_lo_n1 = 0.0f;
        float tail_hi_n0 = 0.0f, tail_hi_n1 = 0.0f;
        for (int k = k_tail_start; k < K; ++k) {
            float w_n0 = 0.0f, w_n1 = 0.0f;
            if (n0 < N) w_n0 = e4m3_to_f32(weight[(long long)n0 * K + k]);
            if (n1 < N) w_n1 = e4m3_to_f32(weight[(long long)n1 * K + k]);
            if (r_lo < M_eff) {
                float x = __half2float(input[(long long)r_lo * K + k]);
                tail_lo_n0 += x * w_n0;
                tail_lo_n1 += x * w_n1;
            }
            if (r_hi < M_eff) {
                float x = __half2float(input[(long long)r_hi * K + k]);
                tail_hi_n0 += x * w_n0;
                tail_hi_n1 += x * w_n1;
            }
        }
        acc_outer[0] += sw_tail * tail_lo_n0;
        acc_outer[1] += sw_tail * tail_lo_n1;
        acc_outer[2] += sw_tail * tail_hi_n0;
        acc_outer[3] += sw_tail * tail_hi_n1;
    }

    // === Scatter output. Each lane owns rows (r_lo, r_hi) × cols
    //     (c0, c1) — 4 outputs per lane. Inactive rows / out-of-range
    //     cols dropped.
    int r_lo = lane >> 2;
    int r_hi = r_lo + 8;
    int c0   = (lane & 3) * 2;
    int c1   = c0 + 1;
    int n0   = n_base + c0;
    int n1   = n_base + c1;
    auto write_out = [&] (int row, int n_col, float v) {
        if (row >= M_eff || n_col >= N) return;
        output[(long long)row * N + n_col] = __float2half(v);
    };
    write_out(r_lo, n0, acc_outer[0]);
    write_out(r_lo, n1, acc_outer[1]);
    write_out(r_hi, n0, acc_outer[2]);
    write_out(r_hi, n1, acc_outer[3]);
}
