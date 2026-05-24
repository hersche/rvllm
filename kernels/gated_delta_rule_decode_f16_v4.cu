// gated_delta_rule_decode_f16_v4: ILP via accumulator split for the
// per-token decode kernel of the Qwen 3.6 / qwen3-next Gated-DeltaNet
// linear-attention block. Direct sibling of
// kernels/gated_delta_rule_prefill_f16_v4.cu (task #108) — same idea,
// applied to the much simpler decode kernel.
//
// v1 (kernels/gated_delta_rule_decode_f16.cu) per-thread inner loops:
//
//   Phase 1: for kd in 0..head_k_dim:
//                v_corr -= s_new * k_smem[kd]      // 128 serial FMAs
//   Phase 2: for kd in 0..head_k_dim:
//                o_acc  += s     * q_smem[kd]      // 128 serial FMAs
//
// Each accumulator chains through one f32 register with 4-cycle FMA
// latency = 512 cycles per phase per thread = 1024 cycles per decode
// step per (v_head, vd) lane on the critical path. Decode kernel is
// 8.8% of total decode GPU time per the latest nsys profile in
// CLAUDE.md (4740 instances × 72 µs avg).
//
// v4 splits each accumulator into N_LANES=8 parallel partials,
// reduced at end-of-loop via pairwise tree. Critical path: 128/8 =
// 16 FMAs per accumulator = 64 cycles. Total FMA count unchanged;
// ILP exposed for the sm_121 scheduler.
//
// Numerical contract: pairwise-tree reduction order differs from v1's
// strict sequential order — tiny bit differences within f32 reduction
// noise. State writes inside each phase are unchanged (one __float2half
// per kd, position-stable). Quality verified via end-to-end smoke (see
// gemma4_nvfp4 ringbuf0 cycle for the validation pattern). Default-off;
// opt-in via RVLLM_QWEN36_LINEAR_ATTN_DECODE_V4=1 until A/B confirms
// the win.

#include <cuda_fp16.h>

extern "C" __global__ void gated_delta_rule_decode_f16_v4_kernel(
    __half*       __restrict__ state,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const float*  __restrict__ alpha,
    const float*  __restrict__ beta,
    __half*       __restrict__ out,
    float scale,
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

    // Stage K and Q for this v-head into shared memory.
    for (int kd = vd; kd < head_k_dim; kd += blockDim.x) {
        k_smem[kd] = __half2float(k[v_head * head_k_dim + kd]);
        q_smem[kd] = __half2float(q[v_head * head_k_dim + kd]);
    }
    __syncthreads();

    const float a = alpha[v_head];
    const float b = beta[v_head];

    long long row_off = ((long long)v_head * head_v_dim + vd) * head_k_dim;

    // ---- Phase 1: forget S by alpha + compute (v - S·K) for this row.
    //
    // Split the v_corr accumulator into 8 parallel partials. State
    // writes stay 1-per-iter; the partials only affect the dot-product
    // critical path.
    const float v_self = __half2float(v[v_head * head_v_dim + vd]);
    float vc0 = 0.0f, vc1 = 0.0f, vc2 = 0.0f, vc3 = 0.0f;
    float vc4 = 0.0f, vc5 = 0.0f, vc6 = 0.0f, vc7 = 0.0f;

    int kd = 0;
    for (; kd + 8 <= head_k_dim; kd += 8) {
        float s0 = __half2float(state[row_off + kd + 0]) * a;
        float s1 = __half2float(state[row_off + kd + 1]) * a;
        float s2 = __half2float(state[row_off + kd + 2]) * a;
        float s3 = __half2float(state[row_off + kd + 3]) * a;
        float s4 = __half2float(state[row_off + kd + 4]) * a;
        float s5 = __half2float(state[row_off + kd + 5]) * a;
        float s6 = __half2float(state[row_off + kd + 6]) * a;
        float s7 = __half2float(state[row_off + kd + 7]) * a;
        state[row_off + kd + 0] = __float2half(s0);
        state[row_off + kd + 1] = __float2half(s1);
        state[row_off + kd + 2] = __float2half(s2);
        state[row_off + kd + 3] = __float2half(s3);
        state[row_off + kd + 4] = __float2half(s4);
        state[row_off + kd + 5] = __float2half(s5);
        state[row_off + kd + 6] = __float2half(s6);
        state[row_off + kd + 7] = __float2half(s7);
        vc0 += s0 * k_smem[kd + 0];
        vc1 += s1 * k_smem[kd + 1];
        vc2 += s2 * k_smem[kd + 2];
        vc3 += s3 * k_smem[kd + 3];
        vc4 += s4 * k_smem[kd + 4];
        vc5 += s5 * k_smem[kd + 5];
        vc6 += s6 * k_smem[kd + 6];
        vc7 += s7 * k_smem[kd + 7];
    }
    // Tail (head_k_dim % 8 != 0 — defensive; production head_k_dim=128).
    for (; kd < head_k_dim; ++kd) {
        float s = __half2float(state[row_off + kd]) * a;
        state[row_off + kd] = __float2half(s);
        vc0 += s * k_smem[kd];
    }

    // Pairwise tree-reduce the 8 partials, then subtract from v_self.
    float vcA = vc0 + vc1, vcB = vc2 + vc3, vcC = vc4 + vc5, vcD = vc6 + vc7;
    float vcAB = vcA + vcB, vcCD = vcC + vcD;
    float v_corr = v_self - (vcAB + vcCD);
    v_corr *= b;
    corr_smem[vd] = v_corr;
    __syncthreads();

    // ---- Phase 2: S += v_corr · K, then O = S · (Q · scale).
    //
    // Same ILP split. Note: corr_smem[vd] is read once for this row's
    // bv (v_corr from Phase 1 for this row), and we accumulate o_acc
    // by reading 8 state slots, updating them in place, multiplying
    // by 8 q_smem slots.
    float bv = corr_smem[vd];
    float oc0 = 0.0f, oc1 = 0.0f, oc2 = 0.0f, oc3 = 0.0f;
    float oc4 = 0.0f, oc5 = 0.0f, oc6 = 0.0f, oc7 = 0.0f;

    kd = 0;
    for (; kd + 8 <= head_k_dim; kd += 8) {
        float s0 = __half2float(state[row_off + kd + 0]) + bv * k_smem[kd + 0];
        float s1 = __half2float(state[row_off + kd + 1]) + bv * k_smem[kd + 1];
        float s2 = __half2float(state[row_off + kd + 2]) + bv * k_smem[kd + 2];
        float s3 = __half2float(state[row_off + kd + 3]) + bv * k_smem[kd + 3];
        float s4 = __half2float(state[row_off + kd + 4]) + bv * k_smem[kd + 4];
        float s5 = __half2float(state[row_off + kd + 5]) + bv * k_smem[kd + 5];
        float s6 = __half2float(state[row_off + kd + 6]) + bv * k_smem[kd + 6];
        float s7 = __half2float(state[row_off + kd + 7]) + bv * k_smem[kd + 7];
        state[row_off + kd + 0] = __float2half(s0);
        state[row_off + kd + 1] = __float2half(s1);
        state[row_off + kd + 2] = __float2half(s2);
        state[row_off + kd + 3] = __float2half(s3);
        state[row_off + kd + 4] = __float2half(s4);
        state[row_off + kd + 5] = __float2half(s5);
        state[row_off + kd + 6] = __float2half(s6);
        state[row_off + kd + 7] = __float2half(s7);
        oc0 += s0 * q_smem[kd + 0];
        oc1 += s1 * q_smem[kd + 1];
        oc2 += s2 * q_smem[kd + 2];
        oc3 += s3 * q_smem[kd + 3];
        oc4 += s4 * q_smem[kd + 4];
        oc5 += s5 * q_smem[kd + 5];
        oc6 += s6 * q_smem[kd + 6];
        oc7 += s7 * q_smem[kd + 7];
    }
    for (; kd < head_k_dim; ++kd) {
        float s = __half2float(state[row_off + kd]) + bv * k_smem[kd];
        state[row_off + kd] = __float2half(s);
        oc0 += s * q_smem[kd];
    }

    float ocA = oc0 + oc1, ocB = oc2 + oc3, ocC = oc4 + oc5, ocD = oc6 + oc7;
    float ocAB = ocA + ocB, ocCD = ocC + ocD;
    float o_acc = ocAB + ocCD;
    out[v_head * head_v_dim + vd] = __float2half(o_acc * scale);
}
