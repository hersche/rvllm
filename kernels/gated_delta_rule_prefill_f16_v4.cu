// Task #108 v4: ILP via accumulator split.
//
// v3 dropped the inner f16 RTNE round (+2.3%); v4 attacks the
// underlying FMA dependency chain.
//
// Per-iter inner loop (Phase 1):
//   for (kd = 0..128) v_corr -= s_row[kd] * k_smem[kd]
// Serial dependency: v_corr accumulates with 4-cycle FMA latency
// → 512 cycles per token per thread = the dominant cost.
//
// v4 splits into N_LANES=8 parallel accumulators (vc0..vc7),
// reduced at end of the inner loop. Critical path: 128/8 = 16
// FMAs per accumulator = 64 cycles instead of 512. Same applies
// to o_acc in Phase 2. Total FMA count unchanged.
//
// Same numerical contract as v3 (no inner f16 RTNE; only boundary
// state rounding). Output bytes within fp32 reduction-ordering
// noise vs v3.

#include <cuda_fp16.h>

extern "C" __global__ void gated_delta_rule_prefill_f16_v4_kernel(
    __half*       __restrict__ state,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const float*  __restrict__ alpha,
    const float*  __restrict__ beta,
    __half*       __restrict__ out,
    float scale,
    int num_tokens,
    int num_v_heads,
    int head_v_dim,
    int head_k_dim
) {
    const int v_head = blockIdx.x;
    const int vd     = threadIdx.x;
    if (vd >= head_v_dim) return;

    extern __shared__ float smem[];
    float* k_smem    = smem;
    float* q_smem    = smem + head_k_dim;
    float* corr_smem = smem + 2 * head_k_dim;

    constexpr int MAX_HEAD_K_DIM = 128;
    float s_row[MAX_HEAD_K_DIM];
    long long row_off = ((long long)v_head * head_v_dim + vd) * head_k_dim;
    for (int kd = 0; kd < head_k_dim; ++kd) {
        s_row[kd] = __half2float(state[row_off + kd]);
    }

    for (int t = 0; t < num_tokens; ++t) {
        long long qk_t_off = ((long long)t * num_v_heads + v_head) * head_k_dim;
        for (int kd = vd; kd < head_k_dim; kd += blockDim.x) {
            k_smem[kd] = __half2float(k[qk_t_off + kd]);
            q_smem[kd] = __half2float(q[qk_t_off + kd]);
        }
        __syncthreads();

        const float a = alpha[(long long)t * num_v_heads + v_head];
        const float b = beta [(long long)t * num_v_heads + v_head];

        long long v_off = ((long long)t * num_v_heads + v_head) * head_v_dim + vd;

        // Phase 1: 8-wide accumulator split.
        float vc0=0, vc1=0, vc2=0, vc3=0, vc4=0, vc5=0, vc6=0, vc7=0;
        #pragma unroll 16
        for (int kd = 0; kd < head_k_dim; kd += 8) {
            float s0 = s_row[kd+0] * a; s_row[kd+0] = s0;
            float s1 = s_row[kd+1] * a; s_row[kd+1] = s1;
            float s2 = s_row[kd+2] * a; s_row[kd+2] = s2;
            float s3 = s_row[kd+3] * a; s_row[kd+3] = s3;
            float s4 = s_row[kd+4] * a; s_row[kd+4] = s4;
            float s5 = s_row[kd+5] * a; s_row[kd+5] = s5;
            float s6 = s_row[kd+6] * a; s_row[kd+6] = s6;
            float s7 = s_row[kd+7] * a; s_row[kd+7] = s7;
            vc0 -= s0 * k_smem[kd+0];
            vc1 -= s1 * k_smem[kd+1];
            vc2 -= s2 * k_smem[kd+2];
            vc3 -= s3 * k_smem[kd+3];
            vc4 -= s4 * k_smem[kd+4];
            vc5 -= s5 * k_smem[kd+5];
            vc6 -= s6 * k_smem[kd+6];
            vc7 -= s7 * k_smem[kd+7];
        }
        float v_corr = __half2float(v[v_off])
                       + ((vc0 + vc1) + (vc2 + vc3))
                       + ((vc4 + vc5) + (vc6 + vc7));
        v_corr *= b;
        corr_smem[vd] = v_corr;
        __syncthreads();

        // Phase 2: 8-wide accumulator split.
        float bv = corr_smem[vd];
        float oc0=0, oc1=0, oc2=0, oc3=0, oc4=0, oc5=0, oc6=0, oc7=0;
        #pragma unroll 16
        for (int kd = 0; kd < head_k_dim; kd += 8) {
            float s0 = s_row[kd+0] + bv * k_smem[kd+0]; s_row[kd+0] = s0;
            float s1 = s_row[kd+1] + bv * k_smem[kd+1]; s_row[kd+1] = s1;
            float s2 = s_row[kd+2] + bv * k_smem[kd+2]; s_row[kd+2] = s2;
            float s3 = s_row[kd+3] + bv * k_smem[kd+3]; s_row[kd+3] = s3;
            float s4 = s_row[kd+4] + bv * k_smem[kd+4]; s_row[kd+4] = s4;
            float s5 = s_row[kd+5] + bv * k_smem[kd+5]; s_row[kd+5] = s5;
            float s6 = s_row[kd+6] + bv * k_smem[kd+6]; s_row[kd+6] = s6;
            float s7 = s_row[kd+7] + bv * k_smem[kd+7]; s_row[kd+7] = s7;
            oc0 += s0 * q_smem[kd+0];
            oc1 += s1 * q_smem[kd+1];
            oc2 += s2 * q_smem[kd+2];
            oc3 += s3 * q_smem[kd+3];
            oc4 += s4 * q_smem[kd+4];
            oc5 += s5 * q_smem[kd+5];
            oc6 += s6 * q_smem[kd+6];
            oc7 += s7 * q_smem[kd+7];
        }
        float o_acc = ((oc0 + oc1) + (oc2 + oc3))
                    + ((oc4 + oc5) + (oc6 + oc7));
        long long o_off = ((long long)t * num_v_heads + v_head) * head_v_dim + vd;
        out[o_off] = __float2half(o_acc * scale);
        __syncthreads();
    }

    for (int kd = 0; kd < head_k_dim; ++kd) {
        state[row_off + kd] = __float2half(s_row[kd]);
    }
}
