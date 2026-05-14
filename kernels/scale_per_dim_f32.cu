// Per-channel scale on f32 tensor, multiplying along the last
// dimension. Used by Gemma 4 audio attention to apply the precomputed
//   q_scale[d] = (head_dim^-0.5 / ln 2) * softplus(per_dim_scale[d])
// to every Q[n, h, d] before the matrix_ac matmul.
//
// Input layout is [outer, head_dim] f32 where `outer = batch * seq * heads`
// (or any flatten of the upper axes). Each thread handles one element.
//
//   out[i, d] = in[i, d] * scale[d]
//
// One thread per element. Grid: (ceil(total/BLOCK), 1, 1). The kernel
// is in-place (`out` may alias `in`).

extern "C" __global__ void scale_per_dim_f32_kernel(
    float*       __restrict__ x,             // [outer, head_dim]
    const float* __restrict__ scale,         // [head_dim]
    const int                  outer,
    const int                  head_dim
) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)outer * head_dim;
    if (tid >= total) return;
    const int d = (int)(tid % head_dim);
    x[tid] = x[tid] * scale[d];
}
