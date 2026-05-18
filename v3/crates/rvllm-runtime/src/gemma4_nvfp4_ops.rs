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
use rvllm_kernels::KernelFn;
use rvllm_loader::gemma4_nvfp4_weights::Gemma4Nvfp4LinearLoaded;
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

/// Kernel handles for the NVFP4 MLP forward path. Loaded once
/// at bring-up from the existing PTX manifest entries; same
/// kernels Mistral 3.5 uses (codex A1 confirmed input_scale is
/// W4A4-only, so the Mistral W4A16 kernels apply verbatim to
/// the Gemma checkpoint with no per-token activation pre-scale).
#[derive(Copy, Clone)]
pub struct Gemma4Nvfp4MlpKernels {
    /// `mistral35_w4a16_gemv_bf16_kernel` — single linear at M=1.
    /// Used for `down_proj` (and as a fallback when the fused
    /// gate+up path isn't available).
    pub fn_w4a16_gemv: KernelFn,
    /// `mistral35_w4a16_gemm_mn_bf16_kernel` — M>1 W4A16 GEMM.
    /// Used by K-step batched verify to avoid per-token MLP loops.
    pub fn_w4a16_gemm_mn: KernelFn,
    /// `mistral35_w4a16_gate_up_gemv_bf16_kernel` — fused gate+up
    /// (one launch produces both [i_size] outputs). Mistral
    /// confirms shape-generic at runtime (i_size + K are kernel
    /// args, no block-size constants tied to Mistral dims).
    pub fn_w4a16_gate_up_gemv: KernelFn,
    /// `gelu_tanh_mul_bf16_kernel` — Gemma uses gelu_pytorch_tanh
    /// (different from Mistral's silu_mul). Elementwise
    /// `out = gelu_tanh(gate) * up`.
    pub fn_gelu_tanh_mul: KernelFn,
}

/// One bf16 W4A16 MLP linear: `out_bf16[1, N] = act_bf16[1, K] @ dequant(W)^T`.
/// Used for down_proj. For gate+up, prefer
/// `gemma4_nvfp4_gate_up_fused` which does both linears in one
/// launch.
pub unsafe fn gemma4_nvfp4_w4a16_gemv(
    fn_w4a16_gemv: KernelFn,
    act_bf16: u64,
    weight: &Gemma4Nvfp4LinearLoaded,
    out_bf16: u64,
    stream: u64,
) -> Result<()> {
    let mut out = out_bf16;
    let mut wp = weight.packed_ptr;
    let mut ws = weight.sfb_natural_ptr;
    let mut gs = weight.global_scale_ptr;
    let mut act = act_bf16;
    let mut n = weight.shape.n as i32;
    let mut k = weight.shape.k as i32;
    let args: [*mut std::ffi::c_void; 7] = [
        (&mut out) as *mut u64 as *mut _,
        (&mut wp)  as *mut u64 as *mut _,
        (&mut ws)  as *mut u64 as *mut _,
        (&mut gs)  as *mut u64 as *mut _,
        (&mut act) as *mut u64 as *mut _,
        (&mut n)   as *mut i32 as *mut _,
        (&mut k)   as *mut i32 as *mut _,
    ];
    rvllm_fused::launch_raw(
        fn_w4a16_gemv,
        (weight.shape.n as u32, 1, 1),
        (256, 1, 1),
        0, stream, &args,
    )
}

/// One bf16 W4A16 M>1 GEMM:
/// `out_bf16[M, N] = act_bf16[M, K] @ dequant(W[N, K])^T`.
pub unsafe fn gemma4_nvfp4_w4a16_gemm_mn(
    fn_w4a16_gemm_mn: KernelFn,
    act_bf16: u64,
    weight: &Gemma4Nvfp4LinearLoaded,
    out_bf16: u64,
    m: u32,
    stream: u64,
) -> Result<()> {
    if m == 0 {
        return Err(rvllm_core::RvllmError::cuda(
            "gemma4_nvfp4_w4a16_gemm_mn: M must be > 0",
            rvllm_core::CudaErrorKind::Other,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    let mut out = out_bf16;
    let mut wp = weight.packed_ptr;
    let mut ws = weight.sfb_natural_ptr;
    let mut gs = weight.global_scale_ptr;
    let mut act = act_bf16;
    let mut m_arg = m as i32;
    let mut n = weight.shape.n as i32;
    let mut k = weight.shape.k as i32;
    let args: [*mut std::ffi::c_void; 8] = [
        (&mut out) as *mut u64 as *mut _,
        (&mut wp) as *mut u64 as *mut _,
        (&mut ws) as *mut u64 as *mut _,
        (&mut gs) as *mut u64 as *mut _,
        (&mut act) as *mut u64 as *mut _,
        (&mut m_arg) as *mut i32 as *mut _,
        (&mut n) as *mut i32 as *mut _,
        (&mut k) as *mut i32 as *mut _,
    ];
    const M_TILE: u32 = 8;
    let grid_y = (m + M_TILE - 1) / M_TILE;
    rvllm_fused::launch_raw(
        fn_w4a16_gemm_mn,
        (weight.shape.n as u32, grid_y, 1),
        (256, 1, 1),
        0,
        stream,
        &args,
    )
}

/// Fused gate+up GEMV: one launch produces `out_gate[1, i_size]`
/// and `out_up[1, i_size]` from a shared `act_bf16[1, K]` input.
/// Both weights must share the same (N, K) shape — checked.
pub unsafe fn gemma4_nvfp4_gate_up_fused(
    fn_w4a16_gate_up_gemv: KernelFn,
    act_bf16: u64,
    gate: &Gemma4Nvfp4LinearLoaded,
    up: &Gemma4Nvfp4LinearLoaded,
    out_gate_bf16: u64,
    out_up_bf16: u64,
    stream: u64,
) -> Result<()> {
    if gate.shape != up.shape {
        return Err(rvllm_core::RvllmError::cuda(
            "gemma4_nvfp4_gate_up_fused: gate/up shapes differ",
            rvllm_core::CudaErrorKind::Other,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    let mut o_g = out_gate_bf16;
    let mut o_u = out_up_bf16;
    let mut wp_g = gate.packed_ptr;
    let mut ws_g = gate.sfb_natural_ptr;
    let mut gs_g = gate.global_scale_ptr;
    let mut wp_u = up.packed_ptr;
    let mut ws_u = up.sfb_natural_ptr;
    let mut gs_u = up.global_scale_ptr;
    let mut act = act_bf16;
    let mut i_size = gate.shape.n as i32;
    let mut k = gate.shape.k as i32;
    let args: [*mut std::ffi::c_void; 11] = [
        (&mut o_g)    as *mut u64 as *mut _,
        (&mut o_u)    as *mut u64 as *mut _,
        (&mut wp_g)   as *mut u64 as *mut _,
        (&mut ws_g)   as *mut u64 as *mut _,
        (&mut gs_g)   as *mut u64 as *mut _,
        (&mut wp_u)   as *mut u64 as *mut _,
        (&mut ws_u)   as *mut u64 as *mut _,
        (&mut gs_u)   as *mut u64 as *mut _,
        (&mut act)    as *mut u64 as *mut _,
        (&mut i_size) as *mut i32 as *mut _,
        (&mut k)      as *mut i32 as *mut _,
    ];
    let total_rows = (2 * i_size) as u32;
    rvllm_fused::launch_raw(
        fn_w4a16_gate_up_gemv,
        (total_rows, 1, 1),
        (256, 1, 1),
        0, stream, &args,
    )
}

/// Elementwise `out[i] = gelu_pytorch_tanh(gate[i]) * up[i]` for
/// `i in 0..n`. Gemma 4 uses gelu_pytorch_tanh — distinct from
/// Mistral's silu_mul. Per gemma4_arch.rs:11.
pub unsafe fn gemma4_nvfp4_gelu_tanh_mul(
    fn_gelu_tanh_mul: KernelFn,
    out_bf16: u64,
    gate_bf16: u64,
    up_bf16: u64,
    n: u32,
    stream: u64,
) -> Result<()> {
    let mut out = out_bf16;
    let mut g = gate_bf16;
    let mut u = up_bf16;
    let mut n_arg = n as i32;
    let args: [*mut std::ffi::c_void; 4] = [
        (&mut out)   as *mut u64 as *mut _,
        (&mut g)     as *mut u64 as *mut _,
        (&mut u)     as *mut u64 as *mut _,
        (&mut n_arg) as *mut i32 as *mut _,
    ];
    const BLOCK: u32 = 256;
    let grid = ((n + BLOCK - 1) / BLOCK, 1, 1);
    rvllm_fused::launch_raw(
        fn_gelu_tanh_mul, grid, (BLOCK, 1, 1), 0, stream, &args,
    )
}

/// One full MLP block forward: `out = down(gelu_tanh(gate(act)) * up(act))`.
///
/// `scratch_bf16` must provide `2 * intermediate_size` bf16
/// slots (so 2 * I_size * 2 bytes): the fused gate+up writes
/// `out_gate` at offset 0 and `out_up` at offset `I_size * 2`
/// bytes; the GELU-mul reads both and writes to offset 0 (in
/// place over the gate buffer); down_proj reads from offset 0.
pub unsafe fn gemma4_nvfp4_mlp_forward(
    kernels: &Gemma4Nvfp4MlpKernels,
    act_bf16: u64,
    out_bf16: u64,
    gate: &Gemma4Nvfp4LinearLoaded,
    up: &Gemma4Nvfp4LinearLoaded,
    down: &Gemma4Nvfp4LinearLoaded,
    scratch_bf16: u64,
    stream: u64,
) -> Result<()> {
    let i_size = gate.shape.n;
    let i_bytes = (i_size * 2) as u64;
    let gate_out_ptr = scratch_bf16;
    let up_out_ptr = scratch_bf16 + i_bytes;

    gemma4_nvfp4_gate_up_fused(
        kernels.fn_w4a16_gate_up_gemv,
        act_bf16, gate, up,
        gate_out_ptr, up_out_ptr, stream,
    )?;
    // Reuse gate_out_ptr as the GELU-mul output (write-after-read
    // pattern at the same element index is well-defined for the
    // 1-thread-per-element kernel above).
    gemma4_nvfp4_gelu_tanh_mul(
        kernels.fn_gelu_tanh_mul,
        gate_out_ptr, gate_out_ptr, up_out_ptr, i_size as u32, stream,
    )?;
    gemma4_nvfp4_w4a16_gemv(
        kernels.fn_w4a16_gemv,
        gate_out_ptr, down, out_bf16, stream,
    )?;
    Ok(())
}

/// Batched MLP block forward:
/// `out[M, H] = down(gelu_tanh(gate(act[M, H])) * up(act[M, H]))`.
///
/// `scratch_bf16` must provide `2 * M * intermediate_size` bf16 slots.
pub unsafe fn gemma4_nvfp4_mlp_forward_batched(
    kernels: &Gemma4Nvfp4MlpKernels,
    act_bf16: u64,
    out_bf16: u64,
    gate: &Gemma4Nvfp4LinearLoaded,
    up: &Gemma4Nvfp4LinearLoaded,
    down: &Gemma4Nvfp4LinearLoaded,
    scratch_bf16: u64,
    m: u32,
    stream: u64,
) -> Result<()> {
    if gate.shape != up.shape {
        return Err(rvllm_core::RvllmError::cuda(
            "gemma4_nvfp4_mlp_forward_batched: gate/up shapes differ",
            rvllm_core::CudaErrorKind::Other,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    if m == 0 {
        return Err(rvllm_core::RvllmError::cuda(
            "gemma4_nvfp4_mlp_forward_batched: M must be > 0",
            rvllm_core::CudaErrorKind::Other,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    let i_size = gate.shape.n;
    let elem_count = (m as usize) * i_size;
    let gate_out_ptr = scratch_bf16;
    let up_out_ptr = scratch_bf16 + (elem_count * 2) as u64;

    gemma4_nvfp4_w4a16_gemm_mn(
        kernels.fn_w4a16_gemm_mn,
        act_bf16,
        gate,
        gate_out_ptr,
        m,
        stream,
    )?;
    gemma4_nvfp4_w4a16_gemm_mn(
        kernels.fn_w4a16_gemm_mn,
        act_bf16,
        up,
        up_out_ptr,
        m,
        stream,
    )?;
    gemma4_nvfp4_gelu_tanh_mul(
        kernels.fn_gelu_tanh_mul,
        gate_out_ptr,
        gate_out_ptr,
        up_out_ptr,
        elem_count as u32,
        stream,
    )?;
    gemma4_nvfp4_w4a16_gemm_mn(
        kernels.fn_w4a16_gemm_mn,
        gate_out_ptr,
        down,
        out_bf16,
        m,
        stream,
    )?;
    Ok(())
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

    /// Commit #3 smoke: layer-0 MLP forward (gate_up_fused +
    /// gelu_tanh_mul + down_proj) on synthetic bf16 input.
    /// Verifies the Mistral W4A16 kernels work for Gemma's
    /// shapes (i_size=21504, K=5376 vs Mistral's i_size=28672,
    /// K=12288). Output is bf16 [1, 5376]; assert no NaN/Inf
    /// and plausible magnitudes.
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_ops::tests::ondisk_layer0_mlp_smoke \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_layer0_mlp_smoke() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping MLP smoke");
                return;
            }
        };
        let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&dir)
            .expect("Gemma4Arch::from_dir");
        assert_eq!(arch.hidden_size, 5376);
        assert_eq!(arch.intermediate_size, 21504);

        let ctx = rvllm_mem::context::CudaContextHandle::init(0)
            .expect("CudaContextHandle::init");
        let arena = rvllm_mem::HbmArena::new(&ctx, 768 * 1024 * 1024)
            .expect("HbmArena::new");

        let pool = Gemma4Nvfp4ShardPool::open(&dir).expect("ShardPool::open");
        let layer0 = upload_gemma4_nvfp4_layer(&arena, &pool, &arch, 0)
            .expect("upload_gemma4_nvfp4_layer(0)");

        // Load the three kernels we need. They're already in the
        // sm_121 PTX manifest (Mistral path consumes them in prod).
        let manifest_path = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121/manifest.json",
        );
        let verified = rvllm_kernels::KernelManifest::load_and_verify(&manifest_path)
            .expect("KernelManifest::load_and_verify");
        let loader = rvllm_kernels::KernelLoader::new(verified);
        let mod_gemv = loader.load_ptx("mistral35_w4a16_gemv_bf16")
            .expect("load_ptx gemv");
        let fn_gemv = mod_gemv
            .get_function("mistral35_w4a16_gemv_bf16_kernel")
            .expect("get_function gemv");
        let mod_gateup = loader.load_ptx("mistral35_w4a16_gate_up_gemv_bf16")
            .expect("load_ptx gate_up");
        let fn_gateup = mod_gateup
            .get_function("mistral35_w4a16_gate_up_gemv_bf16_kernel")
            .expect("get_function gate_up");
        let mod_gelu = loader.load_ptx("gelu_tanh_mul_bf16")
            .expect("load_ptx gelu");
        let fn_gelu = mod_gelu
            .get_function("gelu_tanh_mul_bf16_kernel")
            .expect("get_function gelu");
        let mlp_kernels = Gemma4Nvfp4MlpKernels {
            fn_w4a16_gemv: fn_gemv,
            fn_w4a16_gate_up_gemv: fn_gateup,
            fn_gelu_tanh_mul: fn_gelu,
        };

        let cuda_stream = rvllm_mem::Stream::new(&ctx).expect("Stream::new");
        let stream: u64 = cuda_stream.raw();

        // Synthetic activation [1, 5376] bf16, all 0.01.
        let bf16_one_hundredth: u16 = 0x3C23;
        let act_bytes: Vec<u8> = (0..arch.hidden_size)
            .flat_map(|_| bf16_one_hundredth.to_le_bytes())
            .collect();
        let act_region = arena
            .region("nvfp4_mlp_smoke_act", act_bytes.len(), 256)
            .expect("act region");
        unsafe { act_region.copy_from_host(&act_bytes).expect("act HtoD") };

        // Scratch: [1, 2 * intermediate_size] bf16 = 2*21504*2 = 86016 B
        let scratch_region = arena
            .region("nvfp4_mlp_smoke_scratch",
                    2 * arch.intermediate_size * 2, 256)
            .expect("scratch region");
        // Output: [1, hidden_size] bf16
        let out_region = arena
            .region("nvfp4_mlp_smoke_out", arch.hidden_size * 2, 256)
            .expect("out region");

        unsafe {
            gemma4_nvfp4_mlp_forward(
                &mlp_kernels,
                act_region.device_ptr(),
                out_region.device_ptr(),
                &layer0.gate_proj,
                &layer0.up_proj,
                &layer0.down_proj,
                scratch_region.device_ptr(),
                stream,
            ).expect("gemma4_nvfp4_mlp_forward");
        }
        cuda_stream.fence().expect("fence");

        // DtoH the bf16 output and convert to f32 for inspection.
        let mut out_bytes: Vec<u8> = vec![0u8; arch.hidden_size * 2];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                out_bytes.as_mut_ptr() as *mut _,
                out_region.device_ptr(),
                out_bytes.len(),
            );
            assert_eq!(rc, CUresult::CUDA_SUCCESS, "DtoH failed");
        }
        // bf16 → f32 (manual, no helper crate needed)
        let out_f32: Vec<f32> = out_bytes.chunks_exact(2).map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            f32::from_bits((bits as u32) << 16)
        }).collect();

        let nan_count = out_f32.iter().filter(|x| x.is_nan()).count();
        let inf_count = out_f32.iter().filter(|x| x.is_infinite()).count();
        let mean_abs: f32 = out_f32.iter().map(|x| x.abs()).sum::<f32>()
                          / (arch.hidden_size as f32);
        let max_abs: f32 = out_f32.iter().fold(0f32, |acc, &x| acc.max(x.abs()));

        eprintln!(
            "[mlp-smoke] hidden={} intermediate={}  out: nan={nan_count} inf={inf_count} \
             mean_abs={mean_abs:.6} max_abs={max_abs:.6} first8={:?}",
            arch.hidden_size, arch.intermediate_size, &out_f32[..8],
        );

        assert_eq!(nan_count, 0, "{nan_count} NaN(s) in MLP output");
        assert_eq!(inf_count, 0, "{inf_count} Inf(s) in MLP output");
        assert!(mean_abs > 0.0, "MLP output is all zeros");
        assert!(mean_abs < 1000.0,
                "MLP mean_abs={mean_abs} implausibly large");
    }
}
