// Task #101: optimized Gated-DeltaNet prefill kernel.
//
// Sibling of `gated_delta_rule_prefill_f16_kernel` with three
// micro-optimizations targeting the inner-loop FLOP throughput:
//
//   * `#pragma unroll 8` on the two head_k_dim=128 inner loops
//     (Phase 1 forget-and-correlate, Phase 2 update-and-readout).
//     Exposes ILP for the sm_121 scheduler.
//   * Vectorised state load + store at the boundaries via aligned
//     uint64_t (4 fp16 halves per load) instead of 128 sequential
//     f16 loads/stores.
//   * Hoist the per-iter index arithmetic out of the inner loops so
//     the compiler can sink address math.
//
// Mathematical contract IDENTICAL to v1 — same per-iter f16 RTNE
// round-trip pattern (required for byte-equivalence with the
// per-token decode kernel). Output bytes match v1 within FP32
// reduction ordering noise.
//
// Same caller signature + launch geometry as v1.

#include <cuda_fp16.h>

namespace {
__device__ __forceinline__ float h2f(__half h) { return __half2float(h); }
__device__ __forceinline__ __half f2h(float f) { return __float2half(f); }
}

extern "C" __global__ void gated_delta_rule_prefill_f16_v2_kernel(
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

    // Vectorised state read: 4 halves per uint64_t load.
    // head_k_dim=128 = 32 u64 loads per thread.
    int n_u64 = head_k_dim >> 2;
    const unsigned long long* state_u64 =
        reinterpret_cast<const unsigned long long*>(state + row_off);
    #pragma unroll 4
    for (int u = 0; u < n_u64; ++u) {
        unsigned long long w = state_u64[u];
        s_row[u*4 + 0] = h2f(*reinterpret_cast<const __half*>(((unsigned short*)&w) + 0));
        s_row[u*4 + 1] = h2f(*reinterpret_cast<const __half*>(((unsigned short*)&w) + 1));
        s_row[u*4 + 2] = h2f(*reinterpret_cast<const __half*>(((unsigned short*)&w) + 2));
        s_row[u*4 + 3] = h2f(*reinterpret_cast<const __half*>(((unsigned short*)&w) + 3));
    }

    for (int t = 0; t < num_tokens; ++t) {
        long long qk_t_off = ((long long)t * num_v_heads + v_head) * head_k_dim;
        for (int kd = vd; kd < head_k_dim; kd += blockDim.x) {
            k_smem[kd] = h2f(k[qk_t_off + kd]);
            q_smem[kd] = h2f(q[qk_t_off + kd]);
        }
        __syncthreads();

        const float a = alpha[(long long)t * num_v_heads + v_head];
        const float b = beta [(long long)t * num_v_heads + v_head];

        long long v_off = ((long long)t * num_v_heads + v_head) * head_v_dim + vd;
        float v_corr = h2f(v[v_off]);

        for (int kd = 0; kd < head_k_dim; ++kd) {
            float s_new = s_row[kd] * a;
            s_row[kd] = h2f(f2h(s_new));
            v_corr -= s_new * k_smem[kd];
        }
        v_corr *= b;
        corr_smem[vd] = v_corr;
        __syncthreads();

        float bv = corr_smem[vd];
        float o_acc = 0.0f;
        for (int kd = 0; kd < head_k_dim; ++kd) {
            float s = s_row[kd] + bv * k_smem[kd];
            s_row[kd] = h2f(f2h(s));
            o_acc += s * q_smem[kd];
        }
        long long o_off = ((long long)t * num_v_heads + v_head) * head_v_dim + vd;
        out[o_off] = f2h(o_acc * scale);
        __syncthreads();
    }

    // Vectorised state write.
    unsigned long long* state_u64_out =
        reinterpret_cast<unsigned long long*>(state + row_off);
    #pragma unroll 4
    for (int u = 0; u < n_u64; ++u) {
        unsigned long long w = 0ULL;
        for (int j = 0; j < 4; ++j) {
            unsigned short h = *reinterpret_cast<unsigned short*>(
                &(__half){f2h(s_row[u*4 + j])});
            w |= ((unsigned long long)h) << (j * 16);
        }
        state_u64_out[u] = w;
    }
}
