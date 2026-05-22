// Fused FP8 GEMV (f16 input/output) + in-place residual add.
//
// Drop-in replacement for the back-to-back pair:
//   1. fp8_gemv_blockwise_wpr_native_f16in_kernel  (writes GEMV result to a
//      temp f16 buffer)
//   2. vector_add_f16_kernel                       (h_residual += temp)
//
// Used by Qwen 3.5/3.6 27B dense decode in qwen35_bring_up.rs for:
//   * Per-layer o_proj + post-attention residual add
//   * Per-layer ffn_down + post-FFN residual add
// (and any other site with the same shape pattern).
//
// Numerical contract: per-output element `n`, the inner reduction is
// byte-identical to `fp8_gemv_blockwise_wpr_native_f16in_kernel`
// (same fp8 decode, blockscale loads, lane stride, warp-shuffle).
// Epilogue: lane 0 reads `h_residual[n]` as f16, converts to f32,
// adds the GEMV's f32 acc, narrows back to f16, writes back. The
// `f16 → f32 → narrow` round-trip is identical to what the
// vector_add_f16 kernel does, just folded into the same kernel.
//
// Per-element accumulator is touched by exactly ONE thread (lane 0
// of the warp handling output `n`) — no atomic needed.
//
// Eliminates per decode token:
//   * 1 kernel launch per residual-add site (per layer × 2 sites).
//   * 1 f16-write of the GEMV's temp buffer (no consumer remains).
//
// Launch geometry (identical to `fp8_gemv_blockwise_wpr_native_f16in`):
//   Grid:  (ceil(N / 8), M, 1)
//   Block: (256, 1, 1)
//
// Phase 8 other-models fusion (2026-05-23).

#include <cuda_fp16.h>

__device__ __forceinline__ float fp8e4m3_to_float_fra(unsigned char val) {
    unsigned int s = (val >> 7) & 1u;
    unsigned int e = (val >> 3) & 0xFu;
    unsigned int m = val & 0x7u;
    unsigned int f32_bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
    unsigned int is_normal = (e != 0u) & ((e != 0xFu) | (m != 0x7u));
    f32_bits &= (unsigned int)(-(int)is_normal);
    return __uint_as_float(f32_bits);
}
__device__ __forceinline__ void fp8x2_to_f32_fra(unsigned short packed_fp8x2,
                                                   float& f0, float& f1) {
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(packed_fp8x2));
    unsigned short lo = (unsigned short)(f16x2);
    unsigned short hi = (unsigned short)(f16x2 >> 16);
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
}

extern "C"
__global__ void fp8_gemv_blockwise_wpr_native_f16in_residual_add_kernel(
    __half* __restrict__       h_residual,        // [M, N] f16 — read-modify-write
    const unsigned char* __restrict__ weight,     // [N, K] fp8
    const float* __restrict__  scale,             // [N/128, K/128] f32
    const __half* __restrict__ input,             // [M, K] f16
    int M, int N, int K,
    int num_col_blocks
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 8 + warp;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

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
        fp8x2_to_f32_fra((unsigned short)(w8),       w0, w1);
        fp8x2_to_f32_fra((unsigned short)(w8 >> 16), w2, w3);
        fp8x2_to_f32_fra((unsigned short)(w8 >> 32), w4, w5);
        fp8x2_to_f32_fra((unsigned short)(w8 >> 48), w6, w7);

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
            acc += fp8e4m3_to_float_fra(__ldg(w_row + kr)) * s
                   * __half2float(__ldg(x_row + kr));
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        // Fused epilogue: read existing residual, add GEMV result,
        // write back. f32 path mirrors what vector_add_f16 does
        // (load f16 → __half2float → add → __float2half → store).
        long long idx = (long long)m * N + n;
        float prev = __half2float(h_residual[idx]);
        h_residual[idx] = __float2half(prev + acc);
    }
}
