// K-round batched fused indirect FP8 GEMV + scaled f32 accumulate.
//
// Drop-in replacement for the host-side loop of `top_k` separate
// `fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add_kernel`
// launches in the per-token MoE decode hot path
// (`apply_layer_moe_with_override`). Each warp handles one (m, n)
// output element and sequentially accumulates contributions from
// all `top_k` k_rounds within the SAME warp — no atomic needed
// because exactly ONE warp (the one handling output `n`) owns the
// `acc_f32[m * N + n]` slot.
//
// Per warp per k_round:
//   1. Read expert id from `top_idx[m * top_k + k_round]`.
//   2. Read this k_round's silu_mul slice from
//      `input + k_round * (M * K) * 2 + m * K * 2` (silu_region is
//      k_round-major after the dual_silu kround-batch fusion).
//   3. Compute the FP8 GEMV dot product over K (identical inner
//      loop to the unfused kernel).
//   4. Multiply by `top_w[m * top_k + k_round]` and accumulate to
//      a local f32 register.
//
// After processing all k_rounds, lane 0 writes the accumulated
// f32 sum to `acc_f32[m * N + n]` (adding the prior value already
// there, mirroring the unfused kernel's RMW epilogue).
//
// Launch geometry: same per-warp parallelism as the unfused kernel
// — grid=(ceil(N/8), M), block=256 (8 warps × 32 lanes). Inside
// the block, work scales by `top_k` (each warp does top_k
// sequential GEMV phases instead of 1), so per-block time is
// ~top_k * single_kernel_time. Net trade: 1 launch instead of
// top_k → saves (top_k - 1) × launch_overhead per layer.
//
// Numerical contract: each k_round's inner GEMV reduction is
// byte-identical to the unfused indirect_scaled_add kernel.
// The sequential accumulation in a local register is
// associative-equivalent to the prior 8 separate RMW launches
// (each launch did `prev + w*acc` independently, same order).
//
// Phase 8 down k_round-batch fusion (2026-05-23).

#include <cuda_fp16.h>

__device__ __forceinline__ float fp8e4m3_to_float_kb(unsigned char val) {
    unsigned int s = (val >> 7) & 1u;
    unsigned int e = (val >> 3) & 0xFu;
    unsigned int m = val & 0x7u;
    unsigned int f32_bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
    unsigned int is_normal = (e != 0u) & ((e != 0xFu) | (m != 0x7u));
    f32_bits &= (unsigned int)(-(int)is_normal);
    return __uint_as_float(f32_bits);
}
__device__ __forceinline__ void fp8x2_to_f32_kb(unsigned short packed_fp8x2,
                                                  float& f0, float& f1) {
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(packed_fp8x2));
    unsigned short lo = (unsigned short)(f16x2);
    unsigned short hi = (unsigned short)(f16x2 >> 16);
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
}

extern "C"
__global__ void fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add_kround_batched_kernel(
    float* __restrict__        acc_f32,           // [M, N] f32 — RMW
    const unsigned char* __restrict__ base_w,     // [num_experts, N, K] fp8
    const float* __restrict__  base_s,            // [num_experts, N/128, K/128] f32
    const __half* __restrict__ input_kround,      // [top_k, M, K] f16 — k_round-major
    const int* __restrict__    top_idx,           // [M, top_k] i32
    const float* __restrict__  top_w,             // [M, top_k] f32
    long long w_stride,
    long long s_stride,
    int M, int N, int K,
    int num_col_blocks,
    int top_k
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 8 + warp;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    // Sequential k_round accumulator, owned exclusively by this
    // warp's lane 0 (after the warp-shuffle reduction inside each
    // k_round phase).
    float w_acc = 0.0f;

    for (int kr = 0; kr < top_k; kr++) {
        int e = top_idx[(long long)m * top_k + kr];
        const unsigned char* weight = base_w + (long long)e * w_stride;
        const float*         scale  = base_s + (long long)e * s_stride;
        int scale_row = n >> 7;
        const unsigned char* w_row = weight + (long long)n * K;
        const __half*        x_row = input_kround
            + (long long)kr * M * K
            + (long long)m * K;

        float acc0 = 0.0f, acc1 = 0.0f;

        for (int k = lane * 8; k + 7 < K; k += 256) {
            unsigned long long w8 = __ldg(reinterpret_cast<const unsigned long long*>(w_row + k));
            unsigned long long x_lo = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k));
            unsigned long long x_hi = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 4));

            int sc0 = k >> 7;
            float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);
            int sc4 = (k + 4) >> 7;
            float s4 = (sc4 != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc4]) : s0;

            float w0, w1, w2, w3, w4, w5, w6, w7;
            fp8x2_to_f32_kb((unsigned short)(w8),       w0, w1);
            fp8x2_to_f32_kb((unsigned short)(w8 >> 16), w2, w3);
            fp8x2_to_f32_kb((unsigned short)(w8 >> 32), w4, w5);
            fp8x2_to_f32_kb((unsigned short)(w8 >> 48), w6, w7);

            float x0, x1, x2, x3, x4, x5, x6, x7;
            asm("cvt.f32.f16 %0, %1;" : "=f"(x0) : "h"((unsigned short)(x_lo)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x1) : "h"((unsigned short)(x_lo >> 16)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x2) : "h"((unsigned short)(x_lo >> 32)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x3) : "h"((unsigned short)(x_lo >> 48)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x4) : "h"((unsigned short)(x_hi)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x5) : "h"((unsigned short)(x_hi >> 16)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x6) : "h"((unsigned short)(x_hi >> 32)));
            asm("cvt.f32.f16 %0, %1;" : "=f"(x7) : "h"((unsigned short)(x_hi >> 48)));

            acc0 += w0 * s0 * x0;
            acc0 += w1 * s0 * x1;
            acc0 += w2 * s0 * x2;
            acc0 += w3 * s0 * x3;
            acc1 += w4 * s4 * x4;
            acc1 += w5 * s4 * x5;
            acc1 += w6 * s4 * x6;
            acc1 += w7 * s4 * x7;
        }

        float acc = acc0 + acc1;

        {
            int aligned_k = (K / 8) * 8;
            for (int kk = aligned_k + lane; kk < K; kk += 32) {
                int sc = kk >> 7;
                float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
                acc += fp8e4m3_to_float_kb(__ldg(w_row + kk)) * s
                       * __half2float(__ldg(x_row + kk));
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        // Lane 0 has the full warp-reduced f32 result for this
        // k_round. Multiply by top_w and accumulate into the
        // warp-local running sum.
        if (lane == 0) {
            float w = top_w[(long long)m * top_k + kr];
            w_acc += w * acc;
        }
    }

    // Final write: lane 0 RMW into the f32 accumulator.
    if (lane == 0) {
        long long off = (long long)m * N + n;
        float prev = acc_f32[off];
        acc_f32[off] = prev + w_acc;
    }
}
