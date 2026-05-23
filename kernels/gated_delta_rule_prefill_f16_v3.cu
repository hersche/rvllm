// Task #101 v3: skip the inner-loop f16 round-trip on s_row.
//
// The v1 kernel rounds s_row[kd] through f16 (`__half2float(__float2half(s))`)
// inside both Phase 1 and Phase 2 to stay byte-equivalent with the
// per-token decode kernel (which stores state in f16 between every
// token and re-loads through RTNE). For PREFILL we accumulate state
// in fp32 registers and only round at write-back time — this is the
// mathematically more correct path (less quantisation noise across
// the recurrence) and removes 2× `__float2half`/`__half2float` round-
// trips per inner loop iteration.
//
// Mathematical contract: NOT bit-equivalent to v1. The state evolves
// in pure fp32 within one kernel call. The boundary load (start) and
// boundary write (end) still round to f16, matching v1 at those
// points. For short prefill (≤ a few thousand tokens) the drift vs
// the per-token decode kernel is bounded; long prefill may show
// small accumulated differences.

#include <cuda_fp16.h>

extern "C" __global__ void gated_delta_rule_prefill_f16_v3_kernel(
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
        float v_corr = __half2float(v[v_off]);
        // Phase 1: pure fp32 — no f16 round-trip on s_row.
        for (int kd = 0; kd < head_k_dim; ++kd) {
            float s_new = s_row[kd] * a;
            s_row[kd] = s_new;
            v_corr -= s_new * k_smem[kd];
        }
        v_corr *= b;
        corr_smem[vd] = v_corr;
        __syncthreads();

        float bv = corr_smem[vd];
        float o_acc = 0.0f;
        // Phase 2: pure fp32 — no f16 round-trip on s_row.
        for (int kd = 0; kd < head_k_dim; ++kd) {
            float s = s_row[kd] + bv * k_smem[kd];
            s_row[kd] = s;
            o_acc += s * q_smem[kd];
        }
        long long o_off = ((long long)t * num_v_heads + v_head) * head_v_dim + vd;
        out[o_off] = __float2half(o_acc * scale);
        __syncthreads();
    }

    // Boundary: round to f16 once at write-back, matching v1.
    for (int kd = 0; kd < head_k_dim; ++kd) {
        state[row_off + kd] = __float2half(s_row[kd]);
    }
}
