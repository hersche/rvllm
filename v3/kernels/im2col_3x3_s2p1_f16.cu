// Im2col for Conv2d with kernel=(3, 3), stride=(2, 2), padding=1 in f16.
//
// Used by the Gemma 4 E4B audio subsample stage: a Conv2d with these
// fixed hyperparams maps the mel-spectrogram input ([T, n_mels=128])
// down to [T/2, 64] in the first stage and again to [T/4, 32] in the
// second stage. The fixed (3, 3) kernel + stride 2 + padding 1 layout
// is universal across both stages; only in_channels / out_channels
// differ. We do not generalise to arbitrary k / s / p because the
// audio tower is the only on-disk caller and a fixed-shape kernel
// is faster + simpler to verify.
//
// Layout:
//   input   [in_ch, H_in, W_in]                       f16 channel-first
//   output  [in_ch * 9, H_out * W_out]                f16
//
// Where H_out = (H_in + 1) / 2 and W_out = (W_in + 1) / 2 — matches
// PyTorch's `floor((H_in + 2*padding - kernel) / stride) + 1` with
// stride 2, padding 1, kernel 3.
//
// The output is the standard im2col layout that lets a downstream
// GEMM with weight `[out_ch, in_ch * 9]` (the flattened Conv2d
// weight `[out_ch, in_ch, 3, 3]`) produce the convolved output
// in `[out_ch, H_out * W_out]` shape. The caller transposes to
// channel-last after the GEMM.

#include <cuda_fp16.h>

extern "C"
__global__ void im2col_3x3_s2p1_f16_kernel(
    const __half* __restrict__ input,   // [in_ch * H_in * W_in]
    __half* __restrict__       output,  // [in_ch * 9, H_out * W_out]
    const int in_ch,
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out
) {
    // grid.x covers spatial output (chunked by threadIdx.x);
    // grid.y is one block per (in_ch, kh*3+kw) patch row.
    const int patch_idx   = blockIdx.y;                                 // 0 .. in_ch*9
    const int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;       // 0 .. H_out*W_out

    if (patch_idx >= in_ch * 9) return;
    const int spatial_total = H_out * W_out;
    if (spatial_idx >= spatial_total) return;

    const int h_out = spatial_idx / W_out;
    const int w_out = spatial_idx - h_out * W_out;

    const int ch    = patch_idx / 9;
    const int kk    = patch_idx - ch * 9;
    const int kh    = kk / 3;
    const int kw    = kk - kh * 3;

    // Conv2d with stride=2 padding=1: in-coord = out_coord * 2 - 1 + k.
    const int h_in = 2 * h_out - 1 + kh;
    const int w_in = 2 * w_out - 1 + kw;

    __half v;
    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
        const long long idx = ((long long)ch * H_in + h_in) * (long long)W_in + w_in;
        v = input[idx];
    } else {
        v = __float2half(0.0f);
    }

    // Output layout: row-major [spatial, in_ch * 9] so that cuBLAS's
    // f16_gemm_f32 (which expects B as col-major [k, n] == row-major
    // [n, k] for the caller-side b_f16) reads im2col with k on the
    // INNER axis. Previously we wrote [in_ch * 9, spatial], causing
    // cuBLAS to interpret im2col with stride-mismatched access, which
    // silently produced near-orthogonal results vs HF.
    const long long out_idx =
        (long long)spatial_idx * (in_ch * 9) + patch_idx;
    output[out_idx] = v;
}
