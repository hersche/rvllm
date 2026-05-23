// Fused FP8 GEMV (f16 input) + per-row scaled f32 accumulate via
// device-pointer scalar weight + IN-PLACE f16 residual add to a
// caller-provided hidden buffer.
//
// Drop-in replacement for the back-to-back PAIR currently used in
// the Qwen 3.6 MoE per-token closer (apply_layer_moe_with_override):
//
//   1. fp8_gemv_blockwise_wpr_native_f16in_scaled_add_devw_kernel
//      (shared-expert down → routed_sum_f32[n] += sigmoid * gemv_acc)
//   2. f16_plus_f32_inplace_f16_kernel
//      (hidden_f16[n] += f16(routed_sum_f32[n]))
//
// The fused variant folds both side-effects into the GEMV's epilogue
// on the same warp's lane 0 — no extra launch, no extra DRAM
// traffic vs the unfused pair (we still read routed_sum once, write
// routed_sum once, read hidden once, write hidden once).
//
// The routed_sum_f32 write is PRESERVED so the
// `RVLLM_QWEN36_DEBUG_MOE` post-residual L2 dump continues to see
// the combined routed+shared value in the f32 accumulator.
//
// Numerical contract:
//   * Inner GEMV reduction byte-identical to
//     `fp8_gemv_blockwise_wpr_native_f16in_kernel` (same fp8 decode,
//     blockscale loads, lane stride, warp-shuffle).
//   * f32 accumulate: `routed_sum_f32[n] += *devw * gemv_acc` —
//     identical to the existing `scaled_add_devw` kernel.
//   * Hidden RMW: `hidden_f16[n] += f16(routed_sum_f32[n] +
//     *devw * gemv_acc)` — identical to running
//     `f16_plus_f32_inplace_f16_kernel` on the post-write f32 value.
//
// Per output element `n`, both side-effect writes (acc_f32[n] and
// hidden_f16[n]) are touched by EXACTLY ONE thread (lane 0 of the
// warp owning n) — no atomic needed.
//
// Launch geometry: identical to
// `fp8_gemv_blockwise_wpr_native_f16in_scaled_add_devw_kernel`.
//   Grid:  (ceil(N / 8), M, 1)
//   Block: (256, 1, 1)
//
// Phase 8 closer fusion (2026-05-23).

#include <cuda_fp16.h>

__device__ __forceinline__ float fp8e4m3_to_float_clr(unsigned char val) {
    unsigned int s = (val >> 7) & 1u;
    unsigned int e = (val >> 3) & 0xFu;
    unsigned int m = val & 0x7u;
    unsigned int f32_bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
    unsigned int is_normal = (e != 0u) & ((e != 0xFu) | (m != 0x7u));
    f32_bits &= (unsigned int)(-(int)is_normal);
    return __uint_as_float(f32_bits);
}
__device__ __forceinline__ void fp8x2_to_f32_clr(unsigned short packed_fp8x2,
                                                   float& f0, float& f1) {
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(packed_fp8x2));
    unsigned short lo = (unsigned short)(f16x2);
    unsigned short hi = (unsigned short)(f16x2 >> 16);
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
}

extern "C"
__global__ void fp8_gemv_blockwise_wpr_native_f16in_scaled_add_devw_then_residual_kernel(
    float* __restrict__        acc_f32,           // [M, N] f32 — RMW (routed_sum)
    __half* __restrict__       hidden_f16,        // [M, N] f16 — RMW (residual stream)
    const unsigned char* __restrict__ weight,     // [N, K] fp8
    const float* __restrict__  scale,             // [N/128, K/128] f32
    const __half* __restrict__ input,             // [M, K] f16
    const float* __restrict__  devw,              // f32 [1] — scalar weight
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
        fp8x2_to_f32_clr((unsigned short)(w8),       w0, w1);
        fp8x2_to_f32_clr((unsigned short)(w8 >> 16), w2, w3);
        fp8x2_to_f32_clr((unsigned short)(w8 >> 32), w4, w5);
        fp8x2_to_f32_clr((unsigned short)(w8 >> 48), w6, w7);

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
            acc += fp8e4m3_to_float_clr(__ldg(w_row + kr)) * s
                   * __half2float(__ldg(x_row + kr));
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        float w = devw[0];
        long long off = (long long)m * N + n;
        float prev = acc_f32[off];
        float total = prev + w * acc;
        // (a) Preserve the routed_sum f32 write so the
        //     `RVLLM_QWEN36_DEBUG_MOE` post-residual L2 probe still
        //     sees the combined routed+shared sum.
        acc_f32[off] = total;
        // (b) Fold the f16_plus_f32_inplace_f16 RMW into the same
        //     warp's lane 0. f16 → f32 conversion + add + narrow
        //     identical to the standalone kernel's epilogue.
        float h_f32 = __half2float(hidden_f16[off]);
        hidden_f16[off] = __float2half(h_f32 + total);
    }
}
