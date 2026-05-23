// Batched-topk indirect FP8 GEMV + per-row scaled f32 accumulate
// — fused replacement for the back-to-back pair:
//
//   1. fp8_gemv_blockwise_wpr_native_f16in_indirect_batched_topk_kernel
//      (writes down-projection f16 to a [num_tokens, N] temp buffer)
//   2. scaled_add_f16_to_f32_devw_batched_topk_kernel
//      (reads the temp buffer, multiplies by top_w[m, k_round],
//       accumulates to routed_sum[m, N] f32)
//
// Same fusion shape as
// `fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add_kernel`
// (single-token variant, commit f0f79d5), extended for batched
// prefill: each block handles one (m, n) output element; per-row
// expert index comes from `top_idx[m * top_k + k_round]`; per-row
// weight comes from `top_w[m * top_k + k_round]`.
//
// Numerical contract: inner FP8 GEMV reduction is byte-identical
// to the indirect_batched_topk kernel (same fp8 decode, blockscale
// loads, lane stride, warp-shuffle). Epilogue replaces the f16
// store with:
//
//     routed_sum[m*N + n] = routed_sum[m*N + n] + top_w[m*top_k +
//                                                      k_round] * acc
//
// Per-(m, n) accumulator slot is written by exactly ONE thread
// (lane 0 of the warp handling that output). No atomic needed.
// f16 round-trip eliminated → preserves accumulation precision.
//
// Launch geometry (identical to the unfused indirect_batched_topk):
//   Grid:  (ceil(N/8), num_tokens, 1)
//   Block: (256, 1, 1)
//
// Phase 8 MoE-fusion follow-on (2026-05-23).

#include <cuda_fp16.h>

__device__ __forceinline__ float fp8e4m3_to_float_isabt(unsigned char val) {
    unsigned int s = (val >> 7) & 1u;
    unsigned int e = (val >> 3) & 0xFu;
    unsigned int m = val & 0x7u;
    unsigned int f32_bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
    unsigned int is_normal = (e != 0u) & ((e != 0xFu) | (m != 0x7u));
    f32_bits &= (unsigned int)(-(int)is_normal);
    return __uint_as_float(f32_bits);
}
__device__ __forceinline__ void fp8x2_to_f32_isabt(unsigned short packed_fp8x2,
                                                     float& f0, float& f1) {
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(packed_fp8x2));
    unsigned short lo = (unsigned short)(f16x2);
    unsigned short hi = (unsigned short)(f16x2 >> 16);
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
}

extern "C"
__global__ void fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add_batched_topk_kernel(
    float* __restrict__        routed_sum,        // [num_tokens, N] f32 — RMW
    const unsigned char* __restrict__ base_w,     // [num_experts, N, K] fp8
    const float* __restrict__  base_s,            // [num_experts, N/128, K/128] f32
    const __half* __restrict__ input,             // [num_tokens, K] f16
    const int* __restrict__    top_idx,           // [num_tokens, top_k] i32
    const float* __restrict__  top_w,             // [num_tokens, top_k] f32
    long long w_stride,
    long long s_stride,
    int N, int K,
    int num_col_blocks,
    int top_k,
    int k_round,
    int num_tokens
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 8 + warp;
    int m = blockIdx.y;
    if (n >= N || m >= num_tokens) return;

    int e = top_idx[(long long)m * top_k + k_round];

    const unsigned char* weight = base_w + (long long)e * w_stride;
    const float*         scale  = base_s + (long long)e * s_stride;

    int scale_row = n >> 7;
    const unsigned char* w_row = weight + (long long)n * K;
    const __half*        x_row = input + (long long)m * K;

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
        fp8x2_to_f32_isabt((unsigned short)(w8),       w0, w1);
        fp8x2_to_f32_isabt((unsigned short)(w8 >> 16), w2, w3);
        fp8x2_to_f32_isabt((unsigned short)(w8 >> 32), w4, w5);
        fp8x2_to_f32_isabt((unsigned short)(w8 >> 48), w6, w7);

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
        for (int kr = aligned_k + lane; kr < K; kr += 32) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            acc += fp8e4m3_to_float_isabt(__ldg(w_row + kr)) * s
                   * __half2float(__ldg(x_row + kr));
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        float w = top_w[(long long)m * top_k + k_round];
        long long off = (long long)m * N + n;
        float prev = routed_sum[off];
        routed_sum[off] = prev + w * acc;
    }
}
