//! Phase 3c+ commit #4a: text-only `Gemma4Nvfp4Bringup` skeleton
//! with the load constructor and a pre-attention sub-block
//! smoke (embed → input_layernorm → q_proj on layer 0).
//!
//! This commit lays the structural foundation that future
//! commits build on:
//!   * #4b: per-layer attention math (Q/K-norm, RoPE, paged
//!     attention into the KV cache, O-projection, residual).
//!   * #4c: per-layer MLP composition (pre-FF norm, MLP via
//!     commit #3 helpers, post-FF norm, layer_scalar, residual).
//!     Wires the full 60-layer loop.
//!   * #4d: final RMSNorm + tied LM head + argmax + decoder
//!     loop. Single-token greedy generation smoke.
//!   * #5: NVFP4 KV cache integration (Hadamard rotation,
//!     bf16-in NVFP4 KV write kernel).
//!   * #6: spec-decode + vision splice.
//!
//! Scope of THIS commit:
//!   * Gemma4Nvfp4Bringup struct definition + Drop semantics.
//!   * `load(model_dir, arena_bytes)` constructor that opens
//!     a CUDA context + arena + stream + cuBLASLt handle +
//!     KernelLoader, runs `load_gemma4_nvfp4_text` (uploads
//!     embed + final_norm + all 60 layers), and loads the
//!     PTX kernel handles the forward path will need.
//!   * `pre_attn_one_token(token_id)` — a partial forward
//!     that exercises embed lookup + layer-0 input_layernorm +
//!     layer-0 q_proj. Returns the [N_q] f32 output as a host
//!     Vec for asserting plausible magnitudes.
//!   * `#[ignore]` integration smoke against the on-disk
//!     checkpoint.
//!
//! Out of scope: attention math, MLP composition, KV cache,
//! spec-decode, vision. Those are #4b..#6.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use rvllm_core::Result;
use rvllm_cutlass::cublaslt::CublasLt;
use rvllm_kernels::{KernelFn, KernelLoader, LoadedModule};
use rvllm_loader::gemma4_arch::Gemma4Arch;
use rvllm_loader::gemma4_nvfp4_weights::Gemma4Nvfp4LoadedModel;
use rvllm_mem::{HbmArena, Stream};
use rvllm_mem::context::CudaContextHandle;

use crate::gemma4_nvfp4_load::load_gemma4_nvfp4_text;
use crate::gemma4_nvfp4_ops::{
    gemma4_nvfp4_attn_proj, Gemma4Nvfp4MlpKernels,
};

/// Kernel handles the forward path needs beyond the MLP set.
/// Loaded once at bring-up time from the existing sm_121 PTX
/// manifest.
///
/// Loaded-but-held: the `LoadedModule` values must outlive the
/// `KernelFn` handles (the PTX module owns the function code).
/// They live on `Gemma4Nvfp4Bringup` to bound that lifetime.
struct ForwardKernels {
    _embed_mod: LoadedModule,
    fn_embedding_gather_bf16: KernelFn,
    _rmsnorm_mod: LoadedModule,
    fn_rmsnorm_inplace_bf16: KernelFn,
    /// Reserved for #4b residual add (vector_add_bf16).
    _vector_add_mod: LoadedModule,
    #[allow(dead_code)]
    fn_vector_add_bf16: KernelFn,
}

/// Phase 3c+ text-only bring-up for `nvidia/Gemma-4-31B-IT-NVFP4`.
///
/// Owns: a CUDA primary context (kept alive via the handle),
/// the HbmArena that all weight + scratch allocations live in,
/// a stream, a cuBLASLt handle with its own workspace region,
/// the loaded model (embed + 60 layers + final_norm), and the
/// kernel handles the forward path uses.
pub struct Gemma4Nvfp4Bringup {
    /// Drop order matters: ctx must outlive every CUDA resource
    /// allocated under it. Rust drops fields in declaration
    /// order, so this stays at the END.
    pub arch: Gemma4Arch,
    pub model: Gemma4Nvfp4LoadedModel,
    pub mlp_kernels: Gemma4Nvfp4MlpKernels,
    forward_kernels: ForwardKernels,
    /// PTX modules backing `mlp_kernels`. Kept alive here.
    _mlp_w4a16_gemv_mod: LoadedModule,
    _mlp_w4a16_gate_up_mod: LoadedModule,
    _mlp_gelu_tanh_mul_mod: LoadedModule,
    pub cublaslt: CublasLt,
    pub stream: Stream,
    pub arena: HbmArena<'static>,
    /// Held to keep the primary CUDA context alive.
    _ctx: CudaContextHandle,
}

impl Gemma4Nvfp4Bringup {
    /// Open the CUDA context + arena + stream + cuBLASLt, run
    /// `load_gemma4_nvfp4_text` against `model_dir`, and load
    /// the forward-path kernel handles.
    ///
    /// `arena_bytes` controls the persistent arena size. Need
    /// ~22 GiB for weights on 31B + scratch for forward. 32 GiB
    /// is a safe minimum for the text-only path; the production
    /// profile (`mobile-31b-nvfp4w-rvllm-spec.env`) targets
    /// 96 GiB to cover spec + vision scratch + KV in later
    /// commits.
    pub fn load(
        model_dir: &Path,
        arena_bytes: usize,
        kernels_dir: &Path,
    ) -> Result<Self> {
        let ctx = CudaContextHandle::init(0)?;
        // SAFETY: HbmArena borrows the context for its lifetime.
        // We extend the borrow to 'static by moving the context
        // into Self alongside the arena — the struct's drop
        // order keeps the context alive longer than the arena.
        let arena = unsafe {
            let arena_borrowed: HbmArena<'_> = HbmArena::new(&ctx, arena_bytes)?;
            std::mem::transmute::<HbmArena<'_>, HbmArena<'static>>(arena_borrowed)
        };
        let stream = Stream::new(&ctx)?;

        // cuBLASLt: 64 MiB workspace, 256-byte aligned (cuBLASLt
        // requirement — see commit #2 bisect note).
        let cublaslt_ws = arena.region("gemma4_nvfp4_cublaslt_ws",
                                       64 * 1024 * 1024, 256)?;
        let cublaslt = CublasLt::new(cublaslt_ws.device_ptr(),
                                     64 * 1024 * 1024)?;

        // Parse arch + run the text loader (Phase 3b).
        let arch = Gemma4Arch::from_dir(model_dir)?;
        let model = load_gemma4_nvfp4_text(&arena, model_dir, &arch)?;

        // Load forward-path kernels.
        let manifest_path = kernels_dir.join("manifest.json");
        let manifest = rvllm_kernels::KernelManifest::load_and_verify(&manifest_path)?;
        let loader = KernelLoader::new(manifest);

        let embed_mod = loader.load_ptx("embedding_gather_bf16")?;
        let fn_embedding_gather_bf16 =
            embed_mod.get_function("embedding_gather_bf16_kernel")?;

        let rmsnorm_mod = loader.load_ptx("rmsnorm_inplace_bf16_gbf16")?;
        let fn_rmsnorm_inplace_bf16 =
            rmsnorm_mod.get_function("rmsnorm_inplace_bf16_gbf16_kernel")?;

        let vector_add_mod = loader.load_ptx("vector_add_bf16")?;
        let fn_vector_add_bf16 =
            vector_add_mod.get_function("vector_add_bf16_kernel")?;

        // MLP kernels (commit #3).
        let mlp_gemv_mod = loader.load_ptx("mistral35_w4a16_gemv_bf16")?;
        let fn_w4a16_gemv =
            mlp_gemv_mod.get_function("mistral35_w4a16_gemv_bf16_kernel")?;
        let mlp_gate_up_mod = loader.load_ptx("mistral35_w4a16_gate_up_gemv_bf16")?;
        let fn_w4a16_gate_up_gemv =
            mlp_gate_up_mod.get_function("mistral35_w4a16_gate_up_gemv_bf16_kernel")?;
        let mlp_gelu_mod = loader.load_ptx("gelu_tanh_mul_bf16")?;
        let fn_gelu_tanh_mul =
            mlp_gelu_mod.get_function("gelu_tanh_mul_bf16_kernel")?;

        let mlp_kernels = Gemma4Nvfp4MlpKernels {
            fn_w4a16_gemv,
            fn_w4a16_gate_up_gemv,
            fn_gelu_tanh_mul,
        };

        let forward_kernels = ForwardKernels {
            _embed_mod: embed_mod,
            fn_embedding_gather_bf16,
            _rmsnorm_mod: rmsnorm_mod,
            fn_rmsnorm_inplace_bf16,
            _vector_add_mod: vector_add_mod,
            fn_vector_add_bf16,
        };

        Ok(Self {
            arch,
            model,
            mlp_kernels,
            forward_kernels,
            _mlp_w4a16_gemv_mod: mlp_gemv_mod,
            _mlp_w4a16_gate_up_mod: mlp_gate_up_mod,
            _mlp_gelu_tanh_mul_mod: mlp_gelu_mod,
            cublaslt,
            stream,
            arena,
            _ctx: ctx,
        })
    }

    /// Pre-attention sub-block on layer 0:
    ///   embed[token_id] → input_layernorm → q_proj.
    ///
    /// Returns the [N_q] f32 q-projection output as a host Vec
    /// (N_q = 8192 for sliding layer 0, 16384 for global). The
    /// smoke test uses this to assert non-degenerate magnitudes
    /// without yet wiring attention math, MLP, or the per-layer
    /// loop.
    pub fn pre_attn_one_token(&self, token_id: u32) -> Result<Vec<f32>> {
        let layer0 = &self.model.layers[0];
        let n_q = layer0.q_proj.shape[0] as i32;
        let hidden = self.arch.hidden_size as u32;

        // Scratch: token_id i32 + bf16 residual [1, hidden] + f32 q output.
        let tok_region = self.arena.region("gemma4_nvfp4_pre_attn_tok", 4, 16)?;
        unsafe {
            tok_region.copy_from_host(&(token_id as i32).to_le_bytes())?;
        }
        let residual_region = self.arena.region(
            "gemma4_nvfp4_pre_attn_residual",
            (hidden as usize) * 2, 256,
        )?;
        let q_region = self.arena.region(
            "gemma4_nvfp4_pre_attn_q",
            (n_q as usize) * 4, 256,
        )?;
        let stream_u64 = self.stream.raw();

        // (1) embed lookup → residual_region [1, hidden] bf16.
        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1,
                hidden,
                vocab: self.arch.vocab_size as u32,
            }
            .launch(
                self.forward_kernels.fn_embedding_gather_bf16,
                residual_region.device_ptr(),
                self.model.outside.embed_tokens.offset_bytes,
                tok_region.device_ptr(),
                stream_u64,
            )?;
        }

        // (2) input_layernorm in-place on residual_region.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1,
                hidden,
                eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                residual_region.device_ptr(),
                layer0.input_layernorm.offset_bytes,
                stream_u64,
            )?;
        }

        // (3) q_proj: bf16 [1, hidden] @ bf16 [N_q, hidden]^T → f32 [1, N_q].
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt,
                residual_region.device_ptr(),
                layer0.q_proj.offset_bytes,
                q_region.device_ptr(),
                1, n_q, hidden as i32, stream_u64,
            )?;
        }

        self.stream.fence()?;
        let mut out = vec![0f32; n_q as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut _,
                q_region.device_ptr(),
                (n_q as usize) * 4,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "pre_attn_one_token: q DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Commit #4a smoke: full bring-up load + pre-attention
    /// sub-block on layer 0 with the BOS token.
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_load_and_pre_attn \
    ///     -- --ignored --nocapture
    ///
    /// Requires rvllm-serve stopped — the full 60-layer upload
    /// uses ~22 GiB of unified memory and the smoke allocates
    /// a 32 GiB arena. Coexisting with prod would OOM.
    #[test]
    #[ignore]
    fn ondisk_bringup_load_and_pre_attn() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping bring-up smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );

        // 32 GiB arena — covers ~22 GiB of model weights + ~10
        // GiB of forward scratch headroom. Production profile
        // (mobile-31b-nvfp4w-rvllm-spec.env) uses 96 GiB to
        // include KV cache + spec scratch which we don't allocate here.
        let t0 = std::time::Instant::now();
        let bringup = Gemma4Nvfp4Bringup::load(
            &dir, 32 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");
        let load_s = t0.elapsed().as_secs_f64();
        eprintln!(
            "[bringup-smoke] load complete in {load_s:.1}s — layers={} hidden={} \
             intermediate={} vocab={}",
            bringup.arch.num_hidden_layers,
            bringup.arch.hidden_size,
            bringup.arch.intermediate_size,
            bringup.arch.vocab_size,
        );

        // BOS token (Gemma 4 tokenizer: <bos>=2).
        let token_id: u32 = 2;
        let q = bringup.pre_attn_one_token(token_id)
            .expect("pre_attn_one_token");

        let nan = q.iter().filter(|x| x.is_nan()).count();
        let inf = q.iter().filter(|x| x.is_infinite()).count();
        let mean_abs: f32 = q.iter().map(|x| x.abs()).sum::<f32>() / (q.len() as f32);
        let max_abs: f32 = q.iter().fold(0f32, |a, &x| a.max(x.abs()));
        eprintln!(
            "[bringup-smoke] layer-0 pre-attn q_proj on token {token_id}: \
             N_q={} nan={nan} inf={inf} mean_abs={mean_abs:.4} max_abs={max_abs:.4} \
             first8={:?}",
            q.len(), &q[..8],
        );

        assert_eq!(nan, 0, "{nan} NaN(s) in q_proj");
        assert_eq!(inf, 0, "{inf} Inf(s) in q_proj");
        assert!(mean_abs > 0.0, "q_proj all zero");
        // After RMSNorm the residual has unit-ish magnitude, then
        // q_proj weights are small Gaussian-ish → expect output
        // magnitudes in roughly the [0.01, 10.0] range. Reject
        // extreme outliers that would indicate a bug.
        assert!(mean_abs < 100.0,
                "q_proj mean_abs={mean_abs} implausibly large");
        assert!(max_abs < 1000.0,
                "q_proj max_abs={max_abs} implausibly large");
    }
}
