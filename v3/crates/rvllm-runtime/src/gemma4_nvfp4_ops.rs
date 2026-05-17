//! Phase 3c+ commit #2: BF16 projection helper for the Gemma 4
//! 31B NVFP4 weight forward path.
//!
//! Per `nvidia/Gemma-4-31B-IT-NVFP4`'s quant config, the
//! self_attn q/k/v/o projections stay bf16 (unquantized). The
//! Mistral 3.5 NVFP4 forward path doesn't apply — Mistral
//! quantizes attention to NVFP4 too. The Gemma fp8-block
//! production forward at `gemma4_layer_exec.rs` uses
//! `Fp8GemvF16InLaunch` for attention projections — also wrong
//! dtype for this checkpoint.
//!
//! This module wraps `cublaslt::bf16_gemm_f32` (already cached
//! per shape, sm_121-validated on production Mistral paths) for
//! the four attention projections.
//!
//! Output stays f32 — the next forward step (residual + norm
//! or KV-write RoPE) can consume f32 directly without an
//! extra narrowing kernel. When the eventual MLP / closer
//! consumes bf16, a separate `f32_to_bf16` narrow kernel
//! (TODO commit #3a) bridges. For the smoke test below and for
//! the immediate next step (RoPE prep), f32 is the natural
//! type.

#![cfg(feature = "cuda")]

use rvllm_core::Result;
use rvllm_cutlass::cublaslt::CublasLt;
use rvllm_loader::weights::F16Weight;

/// One attention projection: `out_f32[M, N] = act_bf16[M, K] @ weight_bf16[N, K]^T`.
///
/// Routes through `cublaslt::bf16_gemm_f32` which:
///   * Uses `CUDA_R_16BF` layouts on both inputs (correct bf16
///     interpretation — NOT a reinterpret of f16).
///   * Caches the descriptor + heuristic per (m, n, k).
///   * Accumulates in f32 and writes f32 output.
///
/// `m` is the number of activation rows (1 for decode, K+1 for
/// spec-decode verify, prompt_len for prefill). `n` is the
/// projection output dim (8192 for q_proj/o_proj sliding-input;
/// 4096 for k/v_proj sliding; 16384 for q_proj/o_proj global;
/// 2048 for k_proj global, no v on k_eq_v layers — see
/// `gemma4_weights.rs` header for the full table).
/// `k` is the input dim (hidden_size = 5376 for 31B).
///
/// `weight_bf16` is the `F16Weight.offset_bytes` from
/// `Gemma4Nvfp4LayerLoaded`'s q/k/v/o_proj field. The 16-bit
/// payload IS bf16 — the F16Weight type just reuses the f16
/// container for any 16-bit-class weight (see `weights.rs`
/// docstring).
pub unsafe fn gemma4_nvfp4_attn_proj(
    cublaslt: &CublasLt,
    act_bf16: u64,
    weight_bf16: u64,
    out_f32: u64,
    m: i32,
    n: i32,
    k: i32,
    stream: u64,
) -> Result<()> {
    cublaslt.bf16_gemm_f32(act_bf16, weight_bf16, out_f32, m, n, k, stream)
}

/// Convenience wrapper that takes the `F16Weight` directly off
/// the loaded layer struct, hiding the `offset_bytes` field.
pub unsafe fn gemma4_nvfp4_attn_proj_from_weight(
    cublaslt: &CublasLt,
    act_bf16: u64,
    weight: &F16Weight,
    out_f32: u64,
    m: i32,
    stream: u64,
) -> Result<()> {
    if weight.shape.len() != 2 {
        return Err(rvllm_core::RvllmError::cuda(
            "gemma4_nvfp4_attn_proj_from_weight: weight not 2-D",
            rvllm_core::CudaErrorKind::Other,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    let n = weight.shape[0] as i32;
    let k = weight.shape[1] as i32;
    gemma4_nvfp4_attn_proj(cublaslt, act_bf16, weight.offset_bytes, out_f32, m, n, k, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemma4_nvfp4_load::{Gemma4Nvfp4ShardPool, upload_gemma4_nvfp4_layer};
    use std::path::PathBuf;

    /// Commit #2 smoke: layer-0 q_proj on synthetic bf16 input.
    /// Verifies the BF16 projection helper produces non-degenerate
    /// f32 output through the cuBLASLt bf16_gemm_f32 path. No
    /// reference comparison in this commit — that's the Python
    /// PyTorch oracle in v3/tools/ which lands when commit #3
    /// needs MLP cross-check.
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_ops::tests::ondisk_layer0_q_proj_bf16_smoke \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_layer0_q_proj_bf16_smoke() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping bf16 q_proj smoke");
                return;
            }
        };
        let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&dir)
            .expect("Gemma4Arch::from_dir");

        // CUDA init + 512 MiB arena. Layer 0 + an f32 [M=1, N=8192]
        // output + a bf16 [M=1, K=5376] input is ~10 MiB; arena is
        // sized for layer 0's ~330 MiB resident footprint with room
        // for the smoke scratch.
        let ctx = rvllm_mem::context::CudaContextHandle::init(0)
            .expect("CudaContextHandle::init");
        let arena = rvllm_mem::HbmArena::new(&ctx, 512 * 1024 * 1024)
            .expect("HbmArena::new");

        let pool = Gemma4Nvfp4ShardPool::open(&dir).expect("ShardPool::open");
        let layer0 = upload_gemma4_nvfp4_layer(&arena, &pool, &arch, 0)
            .expect("upload_gemma4_nvfp4_layer(0)");

        // q_proj shape: [8192, 5376] (sliding layer 0 on 31B).
        let q = &layer0.q_proj;
        assert_eq!(q.shape, vec![8192, 5376],
                   "layer 0 q_proj shape unexpected: {:?}", q.shape);
        let n = q.shape[0] as i32;
        let k = q.shape[1] as i32;
        let m: i32 = 1;

        // Build a CublasLt handle with a small workspace region.
        // bf16_gemm_f32 may stash plans/heuristics in this buffer;
        // 16 MiB is comfortable for one (m, n, k).
        let ws_region = arena
            .region("nvfp4_qproj_smoke_ws", 16 * 1024 * 1024, 256)
            .expect("ws region");
        let cublaslt = rvllm_cutlass::cublaslt::CublasLt::new(
            ws_region.device_ptr(),
            16 * 1024 * 1024,
        ).expect("CublasLt::new");

        // Synthetic activation: [M=1, K=5376] bf16, all 0.01 (small
        // enough that the f32 output magnitudes stay well within
        // expected range — full 1.0 would overflow on some heads).
        // bf16 0.01 ≈ 0x3C23 → 2 bytes per elem, K*2 = 10752 bytes.
        let bf16_one_hundredth: u16 = 0x3C23; // ~= 0.0099945068 in bf16
        let act_bytes: Vec<u8> = (0..k as usize)
            .flat_map(|_| bf16_one_hundredth.to_le_bytes())
            .collect();
        let act_region = arena
            .region("nvfp4_qproj_smoke_act", act_bytes.len(), 16)
            .expect("act region");
        unsafe { act_region.copy_from_host(&act_bytes).expect("act HtoD") };

        // f32 output [M=1, N=8192]
        let out_region = arena
            .region("nvfp4_qproj_smoke_out", (n as usize) * 4, 16)
            .expect("out region");

        // cuBLASLt on sm_121 refuses default stream (stream=0)
        // for bf16 matmul — observed LaunchFailed across both
        // bf16_gemm_f32 and fp8_gemm. Allocate a non-default
        // CUDA stream the way the production worker does.
        let cuda_stream = rvllm_mem::Stream::new(&ctx)
            .expect("Stream::new");
        let stream: u64 = cuda_stream.raw();
        unsafe {
            gemma4_nvfp4_attn_proj(
                &cublaslt,
                act_region.device_ptr(),
                q.offset_bytes,
                out_region.device_ptr(),
                m, n, k, stream,
            ).expect("gemma4_nvfp4_attn_proj");
        }
        // Fence the non-default stream before DtoH.
        cuda_stream.fence().expect("stream fence");

        let mut out_f32: Vec<f32> = vec![0.0; n as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                out_f32.as_mut_ptr() as *mut _,
                out_region.device_ptr(),
                (n as usize) * 4,
            );
            assert_eq!(rc, CUresult::CUDA_SUCCESS, "DtoH failed");
        }

        let nan_count = out_f32.iter().filter(|x| x.is_nan()).count();
        let inf_count = out_f32.iter().filter(|x| x.is_infinite()).count();
        let mean_abs: f32 = out_f32.iter().map(|x| x.abs()).sum::<f32>() / (n as f32);
        let max_abs: f32 = out_f32.iter().fold(0f32, |acc, &x| acc.max(x.abs()));

        eprintln!(
            "[qproj-smoke] N={n} K={k}  out: nan={nan_count} inf={inf_count} \
             mean_abs={mean_abs:.6} max_abs={max_abs:.6} first8={:?}",
            &out_f32[..8],
        );

        assert_eq!(nan_count, 0, "{nan_count} NaN(s) in q_proj output");
        assert_eq!(inf_count, 0, "{inf_count} Inf(s) in q_proj output");
        assert!(mean_abs > 0.0, "q_proj output is all zeros");
        assert!(mean_abs < 100.0,
                "q_proj mean_abs={mean_abs} is implausibly large for 0.01 input");
        assert!(max_abs < 1000.0,
                "q_proj max_abs={max_abs} is implausibly large for 0.01 input");
    }
}
