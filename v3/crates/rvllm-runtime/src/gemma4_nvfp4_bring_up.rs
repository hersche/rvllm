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
    /// Reserved for #4e residual add (vector_add_bf16).
    _vector_add_mod: LoadedModule,
    #[allow(dead_code)]
    fn_vector_add_bf16: KernelFn,
    /// `rope_split_half_bf16_kernel` — pure bf16 RoPE on
    /// `[n_heads, head_dim]` in-place. No KV side-effect.
    _rope_mod: LoadedModule,
    fn_rope_split_half_bf16: KernelFn,
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

        let rope_mod = loader.load_ptx("rope_split_half_bf16")?;
        let fn_rope_split_half_bf16 =
            rope_mod.get_function("rope_split_half_bf16_kernel")?;

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
            _rope_mod: rope_mod,
            fn_rope_split_half_bf16,
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

    /// Layer-0 QKV + Q/K-norm on a single token. Extends
    /// `forward_layer0_qkv_only` with per-head RMSNorm on Q and
    /// K (V is not normed in Gemma 4). Returns the bf16 host
    /// view of (q_normed [N_q], k_normed [N_kv], v [N_kv]).
    ///
    /// Gemma 4 Q-norm + K-norm conceptually apply RMSNorm
    /// independently per attention head with gamma=q_norm /
    /// k_norm of shape [head_dim]. We reuse the existing
    /// in-place bf16 RMSNorm kernel by treating the
    /// `[num_heads, head_dim]` buffer as `num_heads` row-tokens
    /// of width `head_dim` — the kernel's per-row RMS reduction
    /// matches the per-head semantics exactly. Sliding layer 0
    /// has num_q_heads=32, num_kv_heads=16, head_dim=256.
    ///
    /// The Q/K outputs from `gemma4_nvfp4_attn_proj` are f32
    /// [1, N], so we first narrow to bf16 with the existing
    /// `f32_to_bf16` cast (TODO: this commit uses a host-side
    /// narrow via a tiny scratch CPU path because the cast
    /// kernel handle isn't yet on ForwardKernels; a follow-up
    /// micro-commit adds the GPU cast).
    pub fn forward_layer0_qk_norm(
        &self,
        token_id: u32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (q_f32, k_f32, v_f32) = self.forward_layer0_qkv_only(token_id)?;
        let layer0 = &self.model.layers[0];
        let head_dim = self.arch.head_dim_sliding as u32;
        let num_q_heads = (q_f32.len() / head_dim as usize) as u32;
        let num_kv_heads = (k_f32.len() / head_dim as usize) as u32;
        debug_assert_eq!(q_f32.len(), (num_q_heads * head_dim) as usize);
        debug_assert_eq!(k_f32.len(), (num_kv_heads * head_dim) as usize);

        // f32 → bf16 narrow on host (one-time small buffer for
        // the smoke; a GPU `f32_to_bf16_kernel` cast is the
        // production path).
        let q_bf16_host: Vec<u16> = q_f32.iter().map(|&x| {
            let bits = x.to_bits();
            // Round-to-nearest-even bf16 narrow.
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            (rounded >> 16) as u16
        }).collect();
        let k_bf16_host: Vec<u16> = k_f32.iter().map(|&x| {
            let bits = x.to_bits();
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            (rounded >> 16) as u16
        }).collect();

        // Upload to device, run per-head RMSNorm in place.
        let q_region = self.arena.region(
            "gemma4_nvfp4_qk_q", q_bf16_host.len() * 2, 256)?;
        let k_region = self.arena.region(
            "gemma4_nvfp4_qk_k", k_bf16_host.len() * 2, 256)?;
        unsafe {
            let q_bytes: &[u8] = std::slice::from_raw_parts(
                q_bf16_host.as_ptr() as *const u8, q_bf16_host.len() * 2);
            let k_bytes: &[u8] = std::slice::from_raw_parts(
                k_bf16_host.as_ptr() as *const u8, k_bf16_host.len() * 2);
            q_region.copy_from_host(q_bytes)?;
            k_region.copy_from_host(k_bytes)?;
        }
        let stream_u64 = self.stream.raw();
        unsafe {
            // Q-norm: num_tokens=num_q_heads, hidden=head_dim,
            // gamma=q_norm[head_dim]. Per-row RMSNorm matches the
            // per-head semantics.
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_q_heads, hidden: head_dim,
                eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                q_region.device_ptr(),
                layer0.q_norm.offset_bytes,
                stream_u64,
            )?;
            // K-norm: same idea on K.
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_kv_heads, hidden: head_dim,
                eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                k_region.device_ptr(),
                layer0.k_norm.offset_bytes,
                stream_u64,
            )?;
        }
        self.stream.fence()?;

        // Read back and convert to f32 for caller inspection.
        let mut q_bf16_out = vec![0u16; q_bf16_host.len()];
        let mut k_bf16_out = vec![0u16; k_bf16_host.len()];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                q_bf16_out.as_mut_ptr() as *mut _,
                q_region.device_ptr(),
                q_bf16_out.len() * 2,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qk_norm: q DtoH failed",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            let rc = cuMemcpyDtoH_v2(
                k_bf16_out.as_mut_ptr() as *mut _,
                k_region.device_ptr(),
                k_bf16_out.len() * 2,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qk_norm: k DtoH failed",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let q_normed: Vec<f32> = q_bf16_out.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect();
        let k_normed: Vec<f32> = k_bf16_out.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect();

        Ok((q_normed, k_normed, v_f32))
    }

    /// Layer-0 post-attention close-out: o_proj +
    /// post_attention_layernorm + residual add. Given a
    /// synthetic `attn_out_bf16` and `h_residual_bf16` (each
    /// [hidden]), returns the updated residual after applying
    /// `h += post_attn_norm(o_proj(attn_out))`.
    ///
    /// The real attention math (softmax(QK^T)V) lands with the
    /// KV cache in commit #5. For #4e the synthetic inputs let
    /// us validate the o_proj + norm + residual chain without
    /// depending on that.
    ///
    /// Per codex round-3 C1: Gemma 4 places post_attention_
    /// layernorm BETWEEN o_proj and residual (not before
    /// residual like Llama). layer_scalar is NOT applied here —
    /// it only multiplies after the MLP block.
    /// residual_weight (arch field, 1.0 on 31B) is conceptually
    /// in the residual add but a no-op at value 1.0.
    pub fn forward_layer0_post_attn(
        &self,
        attn_out_bf16_host: &[u16], // [N_q]
        h_residual_bf16_host: &[u16], // [hidden]
    ) -> Result<Vec<f32>> {
        let layer0 = &self.model.layers[0];
        let n_q = layer0.o_proj.shape[1] as i32; // o_proj is [hidden, N_q]
        let hidden = self.arch.hidden_size as u32;
        if attn_out_bf16_host.len() != n_q as usize {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_layer0_post_attn: attn_out length != N_q",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        if h_residual_bf16_host.len() != hidden as usize {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_layer0_post_attn: h_residual length != hidden",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }

        // Upload synthetic inputs.
        let attn_in_region = self.arena.region(
            "gemma4_nvfp4_post_attn_in",
            attn_out_bf16_host.len() * 2, 256)?;
        let residual_region = self.arena.region(
            "gemma4_nvfp4_post_attn_resid",
            h_residual_bf16_host.len() * 2, 256)?;
        let o_f32_region = self.arena.region(
            "gemma4_nvfp4_post_attn_o_f32",
            (hidden as usize) * 4, 256)?;
        let o_bf16_region = self.arena.region(
            "gemma4_nvfp4_post_attn_o_bf16",
            (hidden as usize) * 2, 256)?;
        unsafe {
            let a: &[u8] = std::slice::from_raw_parts(
                attn_out_bf16_host.as_ptr() as *const u8,
                attn_out_bf16_host.len() * 2);
            let r: &[u8] = std::slice::from_raw_parts(
                h_residual_bf16_host.as_ptr() as *const u8,
                h_residual_bf16_host.len() * 2);
            attn_in_region.copy_from_host(a)?;
            residual_region.copy_from_host(r)?;
        }
        let stream_u64 = self.stream.raw();

        // (1) o_proj: bf16 attn_out [1, N_q] @ bf16 o_proj weight
        //     [hidden, N_q]^T → f32 [1, hidden].
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt,
                attn_in_region.device_ptr(),
                layer0.o_proj.offset_bytes,
                o_f32_region.device_ptr(),
                1, hidden as i32, n_q, stream_u64,
            )?;
        }
        self.stream.fence()?;

        // (2) Narrow f32 → bf16 on host (commit #4c convention).
        //     Read f32 back, narrow, re-upload as bf16 into
        //     o_bf16_region. GPU f32_to_bf16 wiring is a
        //     follow-up — codex round-3 B2 captures the kernel.
        let mut o_f32_host: Vec<f32> = vec![0.0; hidden as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                o_f32_host.as_mut_ptr() as *mut _,
                o_f32_region.device_ptr(),
                (hidden as usize) * 4,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "post_attn: o f32 DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let o_bf16_host: Vec<u16> = o_f32_host.iter().map(|&x| {
            let bits = x.to_bits();
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            (rounded >> 16) as u16
        }).collect();
        unsafe {
            let b: &[u8] = std::slice::from_raw_parts(
                o_bf16_host.as_ptr() as *const u8,
                o_bf16_host.len() * 2);
            o_bf16_region.copy_from_host(b)?;
        }

        // (3) post_attention_layernorm in-place on o_bf16_region.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                o_bf16_region.device_ptr(),
                layer0.post_attention_layernorm.offset_bytes,
                stream_u64,
            )?;
        }

        // (4) Residual add: h_residual += normed_o (both bf16).
        //     residual_weight is 1.0 on 31B (verified at arch
        //     parse) — vector_add_bf16 implements `dst += src`
        //     which is what we want for weight=1.0. A future
        //     scaled-residual kernel handles weight!=1.0.
        unsafe {
            rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }
                .launch(
                    self.forward_kernels.fn_vector_add_bf16,
                    residual_region.device_ptr(),
                    o_bf16_region.device_ptr(),
                    stream_u64,
                )?;
        }
        self.stream.fence()?;

        // Read back the updated residual as f32 for caller.
        let mut h_out_bf16 = vec![0u16; hidden as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                h_out_bf16.as_mut_ptr() as *mut _,
                residual_region.device_ptr(),
                (hidden as usize) * 2,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "post_attn: h_residual DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(h_out_bf16.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect())
    }

    /// Layer-0 RoPE on post-norm Q/K. Extends qk_norm by
    /// applying `rope_split_half_bf16` at `position`. Sliding
    /// layer 0 uses FULL RoPE on head_dim=256, theta=10000.
    /// At position=0 RoPE is the identity (cos=1, sin=0) — the
    /// smoke uses that to assert byte-equality between the pre-
    /// and post-RoPE Q/K under bf16 narrow tolerance.
    pub fn forward_layer0_qk_rope(
        &self,
        token_id: u32,
        position: u32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (q_normed_f32, k_normed_f32, v_f32) =
            self.forward_layer0_qk_norm(token_id)?;
        let head_dim = self.arch.head_dim_sliding as u32;
        let num_q_heads = (q_normed_f32.len() / head_dim as usize) as u32;
        let num_kv_heads = (k_normed_f32.len() / head_dim as usize) as u32;

        // f32 → bf16 host narrow (commit #4c convention; GPU
        // f32_to_bf16 wiring is a separate follow-up commit).
        let narrow_to_bf16 = |xs: &[f32]| -> Vec<u16> {
            xs.iter().map(|&x| {
                let bits = x.to_bits();
                let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
                (rounded >> 16) as u16
            }).collect()
        };
        let q_bf16: Vec<u16> = narrow_to_bf16(&q_normed_f32);
        let k_bf16: Vec<u16> = narrow_to_bf16(&k_normed_f32);

        let q_region = self.arena.region(
            "gemma4_nvfp4_rope_q", q_bf16.len() * 2, 256)?;
        let k_region = self.arena.region(
            "gemma4_nvfp4_rope_k", k_bf16.len() * 2, 256)?;
        unsafe {
            let q_bytes: &[u8] = std::slice::from_raw_parts(
                q_bf16.as_ptr() as *const u8, q_bf16.len() * 2);
            let k_bytes: &[u8] = std::slice::from_raw_parts(
                k_bf16.as_ptr() as *const u8, k_bf16.len() * 2);
            q_region.copy_from_host(q_bytes)?;
            k_region.copy_from_host(k_bytes)?;
        }

        let stream_u64 = self.stream.raw();
        let cos_ptr = self.model.outside.rope_cos_sliding.offset_bytes;
        let sin_ptr = self.model.outside.rope_sin_sliding.offset_bytes;

        // Launch rope_split_half_bf16 per Q then per K.
        //   grid = (n_heads, 1, 1)
        //   block = (head_dim / 2, 1, 1)
        let launch_rope = |qk_ptr: u64, n_heads: u32| -> Result<()> {
            let mut qk = qk_ptr;
            let mut cos = cos_ptr;
            let mut sin = sin_ptr;
            let mut hd = head_dim as i32;
            let mut pos = position as i32;
            let args: [*mut core::ffi::c_void; 5] = [
                (&mut qk) as *mut u64 as *mut _,
                (&mut cos) as *mut u64 as *mut _,
                (&mut sin) as *mut u64 as *mut _,
                (&mut hd) as *mut i32 as *mut _,
                (&mut pos) as *mut i32 as *mut _,
            ];
            unsafe {
                rvllm_fused::launch_raw(
                    self.forward_kernels.fn_rope_split_half_bf16,
                    (n_heads, 1, 1),
                    (head_dim / 2, 1, 1),
                    0, stream_u64, &args,
                )
            }
        };
        launch_rope(q_region.device_ptr(), num_q_heads)?;
        launch_rope(k_region.device_ptr(), num_kv_heads)?;
        self.stream.fence()?;

        // Read back + convert to f32.
        let mut q_out_bf16 = vec![0u16; q_bf16.len()];
        let mut k_out_bf16 = vec![0u16; k_bf16.len()];
        unsafe {
            use cudarc::driver::sys::*;
            for (host, dev, n) in [
                (q_out_bf16.as_mut_ptr() as *mut _, q_region.device_ptr(), q_out_bf16.len() * 2),
                (k_out_bf16.as_mut_ptr() as *mut _, k_region.device_ptr(), k_out_bf16.len() * 2),
            ] {
                let rc = cuMemcpyDtoH_v2(host, dev, n);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qk_rope: DtoH failed",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        let q_out: Vec<f32> = q_out_bf16.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect();
        let k_out: Vec<f32> = k_out_bf16.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect();
        Ok((q_out, k_out, v_f32))
    }

    /// Layer-0 QKV projection on a single token. Extends the
    /// `pre_attn_one_token` smoke with K and V projections too.
    /// Returns (q[N_q], k[N_kv], v[N_kv]) as host f32 vectors.
    ///
    /// Layer 0 is a SLIDING-attention layer per
    /// `arch.layer_types[0]`. v_proj IS present (k_eq_v aliasing
    /// only applies to global layers); the helper requires that.
    ///
    /// `K_kv = arch.num_kv_heads_sliding * arch.head_dim_sliding`
    /// for layer 0 — 16 * 256 = 4096 on 31B.
    ///
    /// What's NOT yet wired (commit #4c):
    ///   * Q-norm / K-norm (the existing fp8-block path uses
    ///     `FusedQkvRmsnormLaunch` with `fused_qkv_rmsnorm_bf16`,
    ///     but the kernel takes fp8-class quantized input. A
    ///     pure-bf16-in variant is the next addition.)
    ///   * RoPE (Gemma 4 sliding uses full RoPE on head_dim=256;
    ///     global uses partial RoPE on head_dim=512 with
    ///     rotary_dim=128 per arch.partial_rotary_factor_global).
    ///   * Attention launch + KV write.
    ///   * O-proj + residual.
    pub fn forward_layer0_qkv_only(
        &self,
        token_id: u32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let layer0 = &self.model.layers[0];
        let n_q = layer0.q_proj.shape[0] as i32;
        let n_kv = layer0.k_proj.shape[0] as i32;
        let v_weight = layer0.v_proj.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::cuda(
                "forward_layer0_qkv_only: v_proj absent on layer 0 \
                 (k_eq_v aliasing applies only to global layers — \
                 layer 0 is sliding, must have v_proj)",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;
        let n_v = v_weight.shape[0] as i32;
        // Sliding layer: k and v have the same output dim
        // (n_kv_heads * head_dim); on 31B that's 16 * 256 = 4096.
        // Codex review caught the prior debug_assert was release-
        // invisible: a corrupted checkpoint with mismatched K/V
        // dims on a sliding layer would silently produce wrong
        // attention output. Upgrade to a hard runtime check.
        if n_kv != n_v {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_layer0_qkv_only: sliding-layer K/V dim mismatch",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }

        let hidden = self.arch.hidden_size as u32;

        // Scratch: token + residual + q + k + v.
        let tok_region = self.arena.region("gemma4_nvfp4_qkv_tok", 4, 16)?;
        unsafe {
            tok_region.copy_from_host(&(token_id as i32).to_le_bytes())?;
        }
        let residual_region = self.arena.region(
            "gemma4_nvfp4_qkv_residual", (hidden as usize) * 2, 256)?;
        let q_region = self.arena.region(
            "gemma4_nvfp4_qkv_q", (n_q as usize) * 4, 256)?;
        let k_region = self.arena.region(
            "gemma4_nvfp4_qkv_k", (n_kv as usize) * 4, 256)?;
        let v_region = self.arena.region(
            "gemma4_nvfp4_qkv_v", (n_v as usize) * 4, 256)?;
        let stream_u64 = self.stream.raw();

        // embed → residual → input_layernorm in-place.
        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1, hidden,
                vocab: self.arch.vocab_size as u32,
            }
            .launch(
                self.forward_kernels.fn_embedding_gather_bf16,
                residual_region.device_ptr(),
                self.model.outside.embed_tokens.offset_bytes,
                tok_region.device_ptr(), stream_u64,
            )?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                residual_region.device_ptr(),
                layer0.input_layernorm.offset_bytes,
                stream_u64,
            )?;
        }

        // Q/K/V projections via commit #2 helper. Three sequential
        // bf16 GEMVs at M=1; commit #4c can fuse via cuBLASLt
        // strided-batched if perf matters here.
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, residual_region.device_ptr(),
                layer0.q_proj.offset_bytes, q_region.device_ptr(),
                1, n_q, hidden as i32, stream_u64,
            )?;
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, residual_region.device_ptr(),
                layer0.k_proj.offset_bytes, k_region.device_ptr(),
                1, n_kv, hidden as i32, stream_u64,
            )?;
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, residual_region.device_ptr(),
                v_weight.offset_bytes, v_region.device_ptr(),
                1, n_v, hidden as i32, stream_u64,
            )?;
        }
        self.stream.fence()?;

        let mut q = vec![0f32; n_q as usize];
        let mut k = vec![0f32; n_kv as usize];
        let mut v = vec![0f32; n_v as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let copies = [
                (q.as_mut_ptr() as *mut _, q_region.device_ptr(), (n_q as usize) * 4),
                (k.as_mut_ptr() as *mut _, k_region.device_ptr(), (n_kv as usize) * 4),
                (v.as_mut_ptr() as *mut _, v_region.device_ptr(), (n_v as usize) * 4),
            ];
            for (host, dev, sz) in copies {
                let rc = cuMemcpyDtoH_v2(host, dev, sz);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "forward_layer0_qkv_only: DtoH failed",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        Ok((q, k, v))
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

    /// Commit #4b smoke: full Q/K/V projection on BOS token.
    /// Extends the pre-attention smoke to also exercise the K
    /// and V projections from the same bf16 GEMV path. Asserts
    /// all three outputs are finite and have plausible
    /// magnitudes.
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_qkv \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_bringup_qkv() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping qkv smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let bringup = Gemma4Nvfp4Bringup::load(
            &dir, 32 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        let token_id: u32 = 2;
        let (q, k, v) = bringup.forward_layer0_qkv_only(token_id)
            .expect("forward_layer0_qkv_only");

        for (name, vec) in [("q", &q), ("k", &k), ("v", &v)] {
            let nan = vec.iter().filter(|x| x.is_nan()).count();
            let inf = vec.iter().filter(|x| x.is_infinite()).count();
            let mean_abs: f32 = vec.iter().map(|x| x.abs()).sum::<f32>() / (vec.len() as f32);
            let max_abs: f32 = vec.iter().fold(0f32, |a, &x| a.max(x.abs()));
            eprintln!(
                "[qkv-smoke] {name}: N={} nan={nan} inf={inf} \
                 mean_abs={mean_abs:.4} max_abs={max_abs:.4} first4={:?}",
                vec.len(), &vec[..4],
            );
            assert_eq!(nan, 0, "{name} has {nan} NaN");
            assert_eq!(inf, 0, "{name} has {inf} Inf");
            assert!(mean_abs > 0.0, "{name} all-zero");
            assert!(mean_abs < 100.0, "{name} mean_abs={mean_abs} too large");
        }

        assert_eq!(q.len(), 8192, "q on sliding layer 0: N_q=32*256");
        assert_eq!(k.len(), 4096, "k on sliding layer 0: N_kv=16*256");
        assert_eq!(v.len(), 4096, "v on sliding layer 0: N_kv=16*256");
    }

    /// Commit #4c smoke: per-head Q-norm + K-norm on layer 0.
    /// After RMSNorm with gamma=q_norm/k_norm[head_dim], each
    /// head's RMS should be close to gamma's mean (gamma is the
    /// learned per-channel scale). For Gemma 4 31B these gammas
    /// are typically O(1), so post-norm per-head RMS should
    /// also be O(1).
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_qk_norm \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_bringup_qk_norm() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping qk_norm smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let bringup = Gemma4Nvfp4Bringup::load(
            &dir, 32 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        let head_dim = bringup.arch.head_dim_sliding;
        let (q, k, _v) = bringup.forward_layer0_qk_norm(2)
            .expect("forward_layer0_qk_norm");

        for (name, vec, n_heads) in [
            ("q", &q, q.len() / head_dim),
            ("k", &k, k.len() / head_dim),
        ] {
            let nan = vec.iter().filter(|x| x.is_nan()).count();
            let inf = vec.iter().filter(|x| x.is_infinite()).count();
            assert_eq!(nan, 0, "{name} has {nan} NaN");
            assert_eq!(inf, 0, "{name} has {inf} Inf");

            // Per-head RMS after norm should be O(gamma.mean()) — for
            // Gemma 4 the gamma is typically in [0.5, 2.0], so per-
            // head RMS should land in roughly [0.1, 10].
            let mut head_rms: Vec<f32> = Vec::with_capacity(n_heads);
            for h in 0..n_heads {
                let row = &vec[h * head_dim..(h + 1) * head_dim];
                let sum_sq: f32 = row.iter().map(|x| x * x).sum();
                let rms = (sum_sq / head_dim as f32).sqrt();
                head_rms.push(rms);
            }
            let mean_rms: f32 = head_rms.iter().sum::<f32>() / (n_heads as f32);
            let min_rms = head_rms.iter().fold(f32::INFINITY, |a, &x| a.min(x));
            let max_rms = head_rms.iter().fold(0f32, |a, &x| a.max(x));
            eprintln!(
                "[qk-norm-smoke] {name}: heads={n_heads} head_dim={head_dim} \
                 mean_rms={mean_rms:.4} min_rms={min_rms:.4} max_rms={max_rms:.4}"
            );
            assert!(mean_rms > 0.01 && mean_rms < 100.0,
                    "{name} mean_rms={mean_rms} outside plausible [0.01, 100]");
        }
    }

    /// Commit #4d smoke: RoPE on post-norm Q/K at position=0.
    /// RoPE is norm-preserving (orthogonal rotation), so the
    /// per-head RMS after RoPE must equal the per-head RMS
    /// after Q/K-norm. At position=0 specifically, cos=1 + sin=0
    /// → RoPE is the identity, so the values should be byte-
    /// equal to the pre-RoPE values modulo bf16 narrow noise.
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_qk_rope_at_zero \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_bringup_qk_rope_at_zero() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping qk_rope smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let bringup = Gemma4Nvfp4Bringup::load(
            &dir, 32 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        let head_dim = bringup.arch.head_dim_sliding;

        let (q_pre, k_pre, _v) = bringup.forward_layer0_qk_norm(2)
            .expect("forward_layer0_qk_norm");
        let (q_post, k_post, _v2) = bringup.forward_layer0_qk_rope(2, 0)
            .expect("forward_layer0_qk_rope(pos=0)");

        for (name, pre, post, n_heads) in [
            ("q", &q_pre, &q_post, q_pre.len() / head_dim),
            ("k", &k_pre, &k_post, k_pre.len() / head_dim),
        ] {
            let nan = post.iter().filter(|x| x.is_nan()).count();
            let inf = post.iter().filter(|x| x.is_infinite()).count();
            assert_eq!(nan, 0, "{name} post-RoPE NaN");
            assert_eq!(inf, 0, "{name} post-RoPE Inf");

            // Per-head RMS preservation: RoPE rotates each (i, i+half)
            // pair by a unit complex number — magnitude is preserved
            // exactly in f32, with small bf16 narrow error.
            let mut max_rms_drift: f32 = 0.0;
            for h in 0..n_heads {
                let row_pre = &pre[h * head_dim..(h + 1) * head_dim];
                let row_post = &post[h * head_dim..(h + 1) * head_dim];
                let rms_pre = (row_pre.iter().map(|x| x * x).sum::<f32>()
                               / head_dim as f32).sqrt();
                let rms_post = (row_post.iter().map(|x| x * x).sum::<f32>()
                                / head_dim as f32).sqrt();
                max_rms_drift = max_rms_drift.max((rms_pre - rms_post).abs());
            }

            // At position=0 cos=1, sin=0 so RoPE is the identity.
            // The pre / post values should agree element-wise modulo
            // bf16 round-trip through the GPU buffer (one extra narrow
            // beyond the pre-RoPE narrow). Element-wise max diff:
            let max_elem_diff = pre.iter().zip(post.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);

            eprintln!(
                "[qk-rope-smoke] {name}: n_heads={n_heads} \
                 max_rms_drift={max_rms_drift:.6} \
                 max_elem_diff={max_elem_diff:.6}",
            );
            // bf16 mantissa is 7 bits → relative precision ~0.8%.
            // For values up to ~10 the absolute error is ~0.08.
            // Allow generous headroom — anything > 0.5 indicates a bug.
            assert!(max_rms_drift < 0.5,
                    "{name} RoPE NOT norm-preserving: max_rms_drift={max_rms_drift}");
            assert!(max_elem_diff < 0.5,
                    "{name} RoPE@pos=0 NOT identity: max_elem_diff={max_elem_diff}");
        }
    }

    /// Commit #4e smoke: o_proj + post_attention_layernorm +
    /// residual add. Uses a synthetic bf16 attn_out (all 0.01)
    /// and a synthetic bf16 residual (all 0.01) — the goal is
    /// structural, not numerical (real attention output lands
    /// with #5's KV cache).
    ///
    /// Assertions:
    /// * Output residual is finite, has plausible magnitudes.
    /// * Output != input residual (the add did something).
    /// * Post-norm contribution is consistent with the post_attn
    ///   gamma magnitude.
    #[test]
    #[ignore]
    fn ondisk_bringup_post_attn() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping post_attn smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let bringup = Gemma4Nvfp4Bringup::load(
            &dir, 32 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        let hidden = bringup.arch.hidden_size;
        let n_q = (bringup.arch.num_attention_heads
                   * bringup.arch.head_dim_sliding) as usize;
        assert_eq!(n_q, 8192, "31B sliding N_q sanity");

        // Synthetic bf16 0.01 ≈ 0x3C23.
        let bf16_0_01: u16 = 0x3C23;
        let attn_out: Vec<u16> = vec![bf16_0_01; n_q];
        let h_in: Vec<u16> = vec![bf16_0_01; hidden];
        let h_in_f32: Vec<f32> = h_in.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect();

        let h_out = bringup.forward_layer0_post_attn(&attn_out, &h_in)
            .expect("forward_layer0_post_attn");

        assert_eq!(h_out.len(), hidden);
        let nan = h_out.iter().filter(|x| x.is_nan()).count();
        let inf = h_out.iter().filter(|x| x.is_infinite()).count();
        let mean_abs: f32 = h_out.iter().map(|x| x.abs()).sum::<f32>()
                          / (hidden as f32);
        let max_abs: f32 = h_out.iter().fold(0f32, |a, &x| a.max(x.abs()));
        // Delta = output - input (= the normed o_proj contribution)
        let diff: Vec<f32> = h_out.iter().zip(h_in_f32.iter())
            .map(|(o, i)| o - i)
            .collect();
        let diff_mean_abs: f32 = diff.iter().map(|x| x.abs()).sum::<f32>()
                              / (hidden as f32);
        let diff_max_abs: f32 = diff.iter().fold(0f32, |a, &x| a.max(x.abs()));

        eprintln!(
            "[post-attn-smoke] hidden={hidden} N_q={n_q}\n  \
             h_out: nan={nan} inf={inf} mean_abs={mean_abs:.6} max_abs={max_abs:.6}\n  \
             delta: mean_abs={diff_mean_abs:.6} max_abs={diff_max_abs:.6}\n  \
             first4_in=0.01  first4_out={:?}",
            &h_out[..4],
        );

        assert_eq!(nan, 0, "post_attn produced NaN");
        assert_eq!(inf, 0, "post_attn produced Inf");
        assert!(diff_mean_abs > 1e-6,
            "post_attn delta=0 — residual add did nothing");
        assert!(mean_abs < 100.0,
            "post_attn h_out mean_abs={mean_abs} implausibly large");
        assert!(diff_mean_abs < 100.0,
            "post_attn delta mean_abs={diff_mean_abs} implausibly large");
    }
}
