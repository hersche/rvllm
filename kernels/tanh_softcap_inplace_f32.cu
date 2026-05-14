// In-place tanh-based logit soft-cap on f32:
//   x = tanh(x / cap) * cap
//
// Used by the Gemma 4 audio attention (`attention_logit_cap=50.0`).
// One thread per element. Grid (ceil(N/BLOCK), 1, 1).

extern "C" __global__ void tanh_softcap_inplace_f32_kernel(
    float* __restrict__ x,
    const float         cap,
    const int           n
) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const float v = x[tid];
    const float inv = (cap != 0.0f) ? (1.0f / cap) : 0.0f;
    x[tid] = tanhf(v * inv) * cap;
}
