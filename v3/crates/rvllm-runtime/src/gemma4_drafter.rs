//! Gemma 4 E4B-it assistant-drafter runtime side.
//!
//! Commit 4 of the speculative-decode landing. This module:
//!
//!   1. Mmaps the safetensors shard described by the
//!      `Gemma4DrafterWeightLayout` from `rvllm-loader::gemma4_drafter`.
//!   2. Converts every BF16 weight to F16 bytes and uploads the
//!      buffer to the shared HBM arena. The lone non-BF16 tensor —
//!      `masked_embedding.token_ordering` (I64 [262144]) — is
//!      uploaded as-is.
//!   3. Records per-tensor device pointers so the forward path
//!      (commits 5–7) can launch kernels without re-reading the file.
//!
//! What this commit DOES NOT do:
//!
//!   * Implement the actual forward step. `Gemma4DrafterRuntime::
//!     forward_step` is a stub that returns
//!     `AttentionError::FeatureNotAvailable` so any caller that
//!     attempts to invoke it without the supporting kernels (commits
//!     5–6) fails cleanly. The runtime upload still happens, so
//!     `ensure_drafter` can be exercised in isolation today.
//!   * Mutate `Gemma4Bringup` state outside its new `drafter` field.
//!     The base model's decode/prefill behaviour is byte-identical
//!     when `ServerConfig::spec_decode` is false (the default).
//!
//! Lazy lifecycle: `Gemma4Bringup::ensure_drafter` is called on the
//! first request that observes `spec_decode == true`. The upload is
//! one-shot — subsequent requests reuse the same device pointers.
//! When `spec_decode == false` the drafter is never constructed and
//! its HBM footprint is zero.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rvllm_core::{AttentionError, AttnCtx, Result, RvllmError};
use rvllm_loader::gemma4_drafter::{
    DrafterLayerType, Gemma4DrafterArch, Gemma4DrafterWeightLayout,
};
#[cfg(feature = "cuda")]
use rvllm_mem::HbmArena;

/// Base-model K/V cache view exposed to the assistant cross-attention
/// layer. The assistant has Q-only attention weights, so its K/V comes
/// from Gemma 4 E4B's shared-KV source layers.
#[derive(Clone, Copy, Debug)]
pub struct DrafterBaseKvView {
    /// F16 / FP8 / NVFP4 K cache for the chosen source layer.
    pub k_cache: u64,
    /// F16 / FP8 / NVFP4 V cache for the chosen source layer.
    pub v_cache: u64,
    /// FP8/NVFP4 K scale cache for the same layer; zero for F16 KV.
    pub k_scale_cache: u64,
    /// FP8/NVFP4 V scale cache for the same layer; zero for F16 KV.
    pub v_scale_cache: u64,
    /// Per-token Q scale cache used by FP8/NVFP4 attention launchers;
    /// zero for an eventual pure-F16 assistant cross-attn kernel.
    pub q_scale_cache: u64,
    /// Paged-cache block table for the active sequence.
    pub block_tables: u64,
    /// Device i32[1] containing committed base context length.
    pub context_lens: u64,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
    /// Matches `crate::gemma4_layer_exec::KvDtype` without making this
    /// module own cache-policy decisions.
    pub kv_dtype: crate::gemma4_layer_exec::KvDtype,
}

/// One assistant MTP step. All pointers are device addresses and are
/// single-sequence today, matching `Gemma4Bringup::run_generate`.
#[derive(Clone, Copy, Debug)]
pub struct DrafterForwardStep {
    /// Base final-normalized hidden for the last accepted token:
    /// f16[backbone_hidden_size].
    pub base_hidden_last_step: u64,
    /// Base embedding row for the last accepted token:
    /// f16[backbone_hidden_size].
    pub last_token_embed: u64,
    pub sliding_kv: DrafterBaseKvView,
    pub full_kv: DrafterBaseKvView,
    /// Absolute position of the token being drafted. Used for RoPE.
    pub position: u32,
    /// Output f32 logits or sparse-logit workspace, owned by caller.
    pub out_logits: u64,
    /// Output f16[backbone_hidden_size] post_projection hidden for
    /// chaining the next MTP step.
    pub out_hidden: u64,
    /// Output i32[1] assistant argmax token after MaskedEmbedder.
    pub out_token_id: u64,
}

/// Per-request scratch for one assistant draft step. The resident
/// weights live in [`Gemma4DrafterRuntime`]; these buffers are safe to
/// allocate below the server scratch checkpoint and restore after each
/// request.
#[derive(Clone, Debug)]
pub struct DrafterStepWorkspace {
    /// f16[2 * backbone_hidden_size] concat of
    /// `[last_token_embed; base_hidden_last_step]`.
    pub pre_projection_in: u64,
    /// f16[hidden_size] assistant hidden stream.
    pub hidden: u64,
    /// f16[hidden_size] residual_1 save buffer. Holds the pre-norm
    /// residual that `q_side` snapshots before doing the in-place
    /// `input_layernorm`. `attn_finisher` reads from this for the
    /// residual_1 add. Added in commit 22 alongside the layer_scalar
    /// fix — without this snapshot the residual_1 add was using the
    /// post-input-layernorm tensor instead of the original residual,
    /// silently destroying the residual stream every layer.
    pub residual1: u64,
    /// f32 scratch for GEMM outputs. Sized for the largest assistant
    /// projection row count used by q_proj/gate/up/down/pre/post.
    pub gemm_f32: u64,
    /// f16 scratch for projection outputs that feed elementwise kernels.
    pub proj_f16: u64,
    /// f16[num_attention_heads * global_head_dim] query scratch.
    pub q: u64,
    /// f16[num_attention_heads * global_head_dim] attention output.
    pub attn_out: u64,
    /// f16[intermediate_size] MLP gate scratch.
    pub mlp_gate: u64,
    /// f16[intermediate_size] MLP up scratch.
    pub mlp_up: u64,
    /// f16[backbone_hidden_size] post_projection output for chaining.
    pub out_hidden: u64,
    /// f32[num_centroids] centroid scores before top-k expansion.
    pub centroid_logits: u64,
    /// i32[centroid_intermediate_top_k] selected centroid ids.
    pub top_centroid_ids: u64,
    /// f32[centroid_intermediate_top_k] selected centroid scores.
    pub top_centroid_scores: u64,
    /// i32[1] selected token id.
    pub out_token_id: u64,
    pub bytes: usize,
}

fn validate_ptr(name: &'static str, ptr: u64, stream: u64, head_dim: u32) -> Result<()> {
    if ptr != 0 {
        return Ok(());
    }
    Err(RvllmError::Attention {
        err: AttentionError::FeatureNotAvailable {
            op: name,
            backend: "Gemma4Drafter",
        },
        ctx: AttnCtx {
            op: name,
            stream,
            num_seqs: 1,
            head_dim,
        },
        bt: std::backtrace::Backtrace::capture(),
    })
}

/// Per-layer device pointers (F16 weights, one element-pair per
/// 2 bytes). All offsets are absolute device pointers — no
/// per-layer base subtraction needed.
#[derive(Clone, Debug)]
pub struct Gemma4DrafterLayerPtrs {
    pub input_layernorm: u64,
    pub post_attention_layernorm: u64,
    pub pre_feedforward_layernorm: u64,
    pub post_feedforward_layernorm: u64,
    pub layer_scalar: u64,
    pub mlp_gate_proj: u64,
    pub mlp_up_proj: u64,
    pub mlp_down_proj: u64,
    pub self_attn_q_proj: u64,
    pub self_attn_q_norm: u64,
    pub self_attn_o_proj: u64,
    /// Mirrored from `Gemma4DrafterLayerOffsets` for convenience —
    /// the forward step needs the head dim to size Q/o GEMMs.
    pub effective_head_dim: usize,
    pub layer_type: DrafterLayerType,
}

/// Top-level (non-layer) device pointers.
#[derive(Clone, Debug)]
pub struct Gemma4DrafterTopPtrs {
    /// `masked_embedding.centroids.weight` [num_centroids, hidden] F16.
    pub centroids: u64,
    /// `masked_embedding.token_ordering` [vocab] **I64**, uploaded
    /// as-is (no F16 conversion). The MaskedEmbedder kernel (commit
    /// 6) reads it as `int64_t*`.
    pub token_ordering: u64,
    /// `model.embed_tokens.weight` [vocab, hidden] F16. Also serves
    /// as the `lm_head` weight per HF
    /// `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`.
    pub embed_tokens: u64,
    /// `pre_projection.weight` [hidden, 2*backbone_hidden] F16.
    pub pre_projection: u64,
    /// `post_projection.weight` [backbone_hidden, hidden] F16.
    pub post_projection: u64,
    /// `model.norm.weight` [hidden] F16.
    pub final_norm: u64,
}

/// Drafter runtime — uploaded weights + the parsed arch metadata.
///
/// Held inside `Gemma4Bringup::drafter` behind a `Mutex<Option<_>>`
/// so initialisation is lazy and at-most-once per engine instance.
#[derive(Debug)]
pub struct Gemma4DrafterRuntime {
    pub arch: Gemma4DrafterArch,
    pub top: Gemma4DrafterTopPtrs,
    pub layers: Vec<Gemma4DrafterLayerPtrs>,
    /// Total HBM bytes consumed by the drafter upload — convenience
    /// for logging.
    pub bytes_resident: usize,
    /// Path the upload came from. Surfaced in error messages.
    pub shard_path: PathBuf,
    /// MaskedEmbedder PTX module + entry handle. Loaded lazily by
    /// `Gemma4Bringup::ensure_drafter` from
    /// `kernels/sm_121/gemma4_masked_embedder.ptx`. The `LoadedModule`
    /// is RAII-dropped only when `Gemma4DrafterRuntime` itself is
    /// dropped, so the `KernelFn` stays valid for the engine lifetime.
    /// Wrapped in `Option` so a non-cuda mock build can leave it
    /// `None` and the cuda build asserts presence at call time.
    pub masked_embedder_mod: Option<rvllm_kernels::LoadedModule>,
    pub fn_masked_embedder_argmax_f16: Option<rvllm_kernels::KernelFn>,
    /// Spec-decode commit 9: F16 shadow KV regions used by the
    /// assistant cross-attention. The drafter has no K/V projections
    /// of its own — at attention time it consumes the base's
    /// last-non-shared-layer K/V (sliding source = layer 22 on E4B,
    /// full source = layer 23). To keep the cross-attention launcher
    /// uniformly F16 regardless of what dtype the base allocates its
    /// own KV in (F16 / FP8 / NVFP4), we maintain a small F16 mirror
    /// at those two source layers and populate it explicitly from
    /// the base side during spec-decode prefill (deferred to a
    /// follow-up commit).
    ///
    /// `Some` on a fully-armed `ensure_drafter` cuda path; `None`
    /// for the non-cuda mock build (or until `attach_shadow_kv` is
    /// called).
    pub shadow_kv: Option<DrafterShadowKv>,
    /// Spec-decode commit 11: paged f16-IO FA-2 decode kernel,
    /// loaded from `kernels/sm_121/flash_attention.ptx`. The
    /// assistant's cross-attention reads from `shadow_kv` (which is
    /// stored in the f16io paged-decode layout) using this same
    /// kernel as the base path. Lazy-attached by
    /// `Gemma4Bringup::ensure_drafter` so the PTX is only loaded
    /// when spec-decode is on.
    pub flash_attention_mod: Option<rvllm_kernels::LoadedModule>,
    pub fn_flash_attention_2_decode_f16io: Option<rvllm_kernels::KernelFn>,
    /// BC=16 build of the f16io decode kernel, loaded from
    /// `flash_attention_decode_f16io_bc16.ptx`. Used by
    /// `launch_cross_attn_global` so head_dim=512 fits the
    /// sm_121 dynamic-smem ceiling (~64 KiB instead of ~128 KiB).
    /// Same kernel symbol name as the BC=32 path; different module.
    pub flash_attention_bc16_mod: Option<rvllm_kernels::LoadedModule>,
    pub fn_flash_attention_2_decode_f16io_bc16: Option<rvllm_kernels::KernelFn>,
    /// Spec-decode commit 10b/10c: shadow-KV dequant kernels. Loaded
    /// from `kernels/sm_121/gemma4_drafter_dequant.ptx` by
    /// `Gemma4Bringup::ensure_drafter`. The two entries handle FP8
    /// E4M3 base KV and packed-4-bit NVFP4 base KV respectively;
    /// `populate_shadow_kv_from_base` dispatches on `base_kv_dtype`.
    pub drafter_dequant_mod: Option<rvllm_kernels::LoadedModule>,
    pub fn_drafter_dequant_fp8_to_f16: Option<rvllm_kernels::KernelFn>,
    pub fn_drafter_dequant_nvfp4_to_f16: Option<rvllm_kernels::KernelFn>,
}

/// F16 shadow KV regions used by the assistant cross-attention. One
/// pair per source layer (`sliding` + `full`). Layout matches the
/// base's paged-decode expectation:
///
///   `[num_blocks_total * block_size * num_kv_heads * head_dim]` f16
///
/// — same shape as `flash_attention_2_decode_f16io_kernel`'s
/// `key_cache` / `value_cache` arguments, so the existing launcher
/// can read these regions directly without a new kernel.
#[derive(Clone, Copy, Debug)]
pub struct DrafterShadowKv {
    pub sliding_k_ptr: u64,
    pub sliding_v_ptr: u64,
    pub full_k_ptr: u64,
    pub full_v_ptr: u64,
    /// Bytes per (K or V) buffer per layer type — exposed so the
    /// populator can compute slot offsets and the launcher can
    /// derive `max_blocks_per_seq`.
    pub sliding_layer_bytes: usize,
    pub full_layer_bytes: usize,
    pub block_size: u32,
    pub num_blocks_total: u32,
    pub max_blocks_per_seq: u32,
    pub sliding_num_kv_heads: u32,
    pub sliding_head_dim: u32,
    pub full_num_kv_heads: u32,
    pub full_head_dim: u32,
}

impl Gemma4DrafterRuntime {
    /// Allocate transient buffers for one assistant draft step.
    #[cfg(feature = "cuda")]
    pub fn alloc_step_workspace(
        &self,
        arena: &HbmArena<'_>,
    ) -> Result<DrafterStepWorkspace> {
        let a = &self.arch;
        let max_q_rows = a.num_attention_heads * a.head_dim_global;
        let max_projection = [
            a.pre_projection_in_dim,
            a.hidden_size,
            a.intermediate_size,
            max_q_rows,
            a.backbone_hidden_size,
            a.num_centroids,
        ]
        .into_iter()
        .max()
        .unwrap_or(a.backbone_hidden_size);

        let mut bytes = 0usize;
        let mut region = |name: &'static str, nbytes: usize, align: usize| -> Result<u64> {
            bytes += nbytes;
            Ok(arena.region(name, nbytes.max(16), align)?.device_ptr())
        };

        Ok(DrafterStepWorkspace {
            pre_projection_in: region(
                "drafter_preproj_in",
                a.pre_projection_in_dim * 2,
                16,
            )?,
            hidden: region("drafter_hidden", a.hidden_size * 2, 16)?,
            residual1: region("drafter_residual1", a.hidden_size * 2, 16)?,
            gemm_f32: region("drafter_gemm_f32", max_projection * 4, 16)?,
            proj_f16: region("drafter_proj_f16", max_projection * 2, 16)?,
            q: region("drafter_q", max_q_rows * 2, 16)?,
            attn_out: region("drafter_attn_out", max_q_rows * 2, 16)?,
            mlp_gate: region("drafter_mlp_gate", a.intermediate_size * 2, 16)?,
            mlp_up: region("drafter_mlp_up", a.intermediate_size * 2, 16)?,
            out_hidden: region("drafter_out_hidden", a.backbone_hidden_size * 2, 16)?,
            centroid_logits: region("drafter_centroid_logits", a.num_centroids * 4, 16)?,
            top_centroid_ids: region(
                "drafter_top_centroid_ids",
                a.centroid_intermediate_top_k * 4,
                16,
            )?,
            top_centroid_scores: region(
                "drafter_top_centroid_scores",
                a.centroid_intermediate_top_k * 4,
                16,
            )?,
            out_token_id: region("drafter_out_token_id", 4, 4)?,
            bytes,
        })
    }

    /// Read the safetensors shard described by `layout`, convert
    /// BF16 → F16 (and pass I64 through), upload each tensor to
    /// `arena`. Returns the populated runtime.
    ///
    /// Cuda-only path. The non-cuda build returns a clear error so
    /// callers compile but can't accidentally exercise the path.
    #[cfg(feature = "cuda")]
    pub fn load(
        layout: &Gemma4DrafterWeightLayout,
        arena: &HbmArena<'_>,
    ) -> Result<Self> {
        let arch = layout.arch.clone();
        let shard_path = layout.shard_path.clone();
        let mmap_bytes = std::fs::read(&shard_path).map_err(|source| {
            RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: shard_path.clone(),
                source,
            }
        })?;

        let mut bytes_resident: usize = 0;

        // Upload helper: BF16 → F16 in-host buffer, then arena.region
        // + HtoD. Region name carries the tensor for debug surfaces.
        let upload_bf16 = |name: &str,
                          offset: u64,
                          numel: usize|
         -> Result<u64> {
            let in_bytes_start = offset as usize;
            let in_bytes_end = in_bytes_start + numel * 2; // BF16 = 2 bytes/elem
            if in_bytes_end > mmap_bytes.len() {
                return Err(corrupt(&shard_path, format!(
                    "tensor {name}: range [{in_bytes_start}, {in_bytes_end}) past file len {}",
                    mmap_bytes.len())));
            }
            let src = &mmap_bytes[in_bytes_start..in_bytes_end];
            let host_f16 = bf16_to_f16_bytes(src);
            // arena.region needs &'static str — pass a fixed name
            // for the whole drafter upload. The per-tensor identity
            // is still useful in error messages below.
            let region = arena.region(
                "drafter_weight",
                host_f16.len(),
                16,
            )?;
            let _ = name;
            unsafe {
                region.copy_from_host(&host_f16)?;
            }
            Ok(region.device_ptr())
        };

        // Upload I64 token_ordering as-is (no conversion).
        let upload_i64 = |name: &str,
                         offset: u64,
                         numel: usize|
         -> Result<u64> {
            let in_bytes_start = offset as usize;
            let in_bytes_end = in_bytes_start + numel * 8; // I64 = 8 bytes/elem
            if in_bytes_end > mmap_bytes.len() {
                return Err(corrupt(&shard_path, format!(
                    "tensor {name}: range [{in_bytes_start}, {in_bytes_end}) past file len {}",
                    mmap_bytes.len())));
            }
            let src = &mmap_bytes[in_bytes_start..in_bytes_end];
            let region = arena.region(
                "drafter_weight",
                src.len(),
                16,
            )?;
            let _ = name;
            unsafe {
                region.copy_from_host(src)?;
            }
            Ok(region.device_ptr())
        };

        let hidden = arch.hidden_size;
        let inter = arch.intermediate_size;
        let vocab = arch.vocab_size;
        let n_cent = arch.num_centroids;
        let backbone = arch.backbone_hidden_size;
        let pre_in = arch.pre_projection_in_dim;

        let top = Gemma4DrafterTopPtrs {
            centroids: upload_bf16(
                "masked_embedding.centroids.weight",
                layout.top.centroids,
                n_cent * hidden,
            )?,
            token_ordering: upload_i64(
                "masked_embedding.token_ordering",
                layout.top.token_ordering,
                vocab,
            )?,
            embed_tokens: upload_bf16(
                "model.embed_tokens.weight",
                layout.top.embed_tokens,
                vocab * hidden,
            )?,
            pre_projection: upload_bf16(
                "pre_projection.weight",
                layout.top.pre_projection,
                hidden * pre_in,
            )?,
            post_projection: upload_bf16(
                "post_projection.weight",
                layout.top.post_projection,
                backbone * hidden,
            )?,
            final_norm: upload_bf16(
                "model.norm.weight",
                layout.top.final_norm,
                hidden,
            )?,
        };

        bytes_resident += 2 * (
            n_cent * hidden + vocab * hidden +
            hidden * pre_in + backbone * hidden + hidden
        ) + 8 * vocab;

        let mut layers = Vec::with_capacity(arch.num_hidden_layers);
        for (li, lo) in layout.layers.iter().enumerate() {
            let eff_hd = lo.effective_head_dim;
            let q_rows = arch.num_attention_heads * eff_hd;
            let q_norm_dim = eff_hd;
            let prefix = format!("layer_{li}");
            layers.push(Gemma4DrafterLayerPtrs {
                input_layernorm: upload_bf16(
                    &format!("{prefix}.input_layernorm.weight"),
                    lo.input_layernorm, hidden)?,
                post_attention_layernorm: upload_bf16(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    lo.post_attention_layernorm, hidden)?,
                pre_feedforward_layernorm: upload_bf16(
                    &format!("{prefix}.pre_feedforward_layernorm.weight"),
                    lo.pre_feedforward_layernorm, hidden)?,
                post_feedforward_layernorm: upload_bf16(
                    &format!("{prefix}.post_feedforward_layernorm.weight"),
                    lo.post_feedforward_layernorm, hidden)?,
                layer_scalar: upload_bf16(
                    &format!("{prefix}.layer_scalar"),
                    lo.layer_scalar, 1)?,
                mlp_gate_proj: upload_bf16(
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    lo.mlp_gate_proj, inter * hidden)?,
                mlp_up_proj: upload_bf16(
                    &format!("{prefix}.mlp.up_proj.weight"),
                    lo.mlp_up_proj, inter * hidden)?,
                mlp_down_proj: upload_bf16(
                    &format!("{prefix}.mlp.down_proj.weight"),
                    lo.mlp_down_proj, hidden * inter)?,
                self_attn_q_proj: upload_bf16(
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    lo.self_attn_q_proj, q_rows * hidden)?,
                self_attn_q_norm: upload_bf16(
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    lo.self_attn_q_norm, q_norm_dim)?,
                self_attn_o_proj: upload_bf16(
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    lo.self_attn_o_proj, hidden * q_rows)?,
                effective_head_dim: eff_hd,
                layer_type: lo.layer_type,
            });
            bytes_resident += 2 * (
                4 * hidden + 1 + 3 * (inter * hidden)
                + q_rows * hidden + q_norm_dim + hidden * q_rows
            );
        }

        eprintln!(
            "[gemma4-drafter] uploaded {} layers + 6 top-level tensors: \
             {:.1} MiB resident in arena (BF16→F16 + I64 token_ordering). \
             arch hidden={hidden} vocab={vocab} centroids={n_cent} \
             num_layers={}",
            arch.num_hidden_layers,
            bytes_resident as f64 / (1024.0 * 1024.0),
            arch.num_hidden_layers,
        );

        Ok(Self {
            arch, top, layers, bytes_resident, shard_path,
            masked_embedder_mod: None,
            fn_masked_embedder_argmax_f16: None,
            shadow_kv: None,
            flash_attention_mod: None,
            fn_flash_attention_2_decode_f16io: None,
            flash_attention_bc16_mod: None,
            fn_flash_attention_2_decode_f16io_bc16: None,
            drafter_dequant_mod: None,
            fn_drafter_dequant_fp8_to_f16: None,
            fn_drafter_dequant_nvfp4_to_f16: None,
        })
    }

    /// Caller (`Gemma4Bringup::ensure_drafter`) loads
    /// `gemma4_masked_embedder` PTX via the engine's `KernelLoader`
    /// and hands the resulting module + entry to the drafter. Doing
    /// this here keeps the PTX load gated on the same spec-decode
    /// env var as the weight upload — non-spec runs never pay for it.
    pub fn attach_masked_embedder_kernel(
        &mut self,
        module: rvllm_kernels::LoadedModule,
        entry: rvllm_kernels::KernelFn,
    ) {
        self.masked_embedder_mod = Some(module);
        self.fn_masked_embedder_argmax_f16 = Some(entry);
    }

    /// Spec-decode commit 9: install F16 shadow KV pointers. Caller
    /// (`Gemma4Bringup::ensure_drafter`) sizes the four regions from
    /// the BASE arch (not the drafter) — `num_kv_heads_for_layer` +
    /// `head_dim_for_layer` of the two source layers — and allocates
    /// them above the scratch checkpoint via the engine's arena.
    /// Once installed, the cross-attention launcher (follow-up
    /// commit) reads these device pointers in place of the base's
    /// own KV regions.
    pub fn attach_shadow_kv(&mut self, shadow: DrafterShadowKv) {
        self.shadow_kv = Some(shadow);
        self.bytes_resident += 2 * (
            shadow.sliding_layer_bytes + shadow.full_layer_bytes
        );
    }

    /// Spec-decode commit 11: install the paged-decode FA-2 f16io
    /// kernel handle that the cross-attention launcher uses to read
    /// from the shadow KV. Mirrors `attach_masked_embedder_kernel`.
    pub fn attach_flash_attention_kernel(
        &mut self,
        module: rvllm_kernels::LoadedModule,
        entry: rvllm_kernels::KernelFn,
    ) {
        self.flash_attention_mod = Some(module);
        self.fn_flash_attention_2_decode_f16io = Some(entry);
    }

    /// BC=16 sibling of `attach_flash_attention_kernel` used by the
    /// global-layer cross-attention path. Loaded from
    /// `flash_attention_decode_f16io_bc16.ptx`.
    pub fn attach_flash_attention_bc16_kernel(
        &mut self,
        module: rvllm_kernels::LoadedModule,
        entry: rvllm_kernels::KernelFn,
    ) {
        self.flash_attention_bc16_mod = Some(module);
        self.fn_flash_attention_2_decode_f16io_bc16 = Some(entry);
    }

    /// Spec-decode commit 10b/10c: install the two shadow-KV dequant
    /// kernel handles (FP8 → F16 and NVFP4 → F16). Caller passes a
    /// single `LoadedModule` plus both entry handles so the RAII
    /// lifetime is tied to one drop.
    pub fn attach_drafter_dequant_kernels(
        &mut self,
        module: rvllm_kernels::LoadedModule,
        fp8_entry: rvllm_kernels::KernelFn,
        nvfp4_entry: rvllm_kernels::KernelFn,
    ) {
        self.drafter_dequant_mod = Some(module);
        self.fn_drafter_dequant_fp8_to_f16 = Some(fp8_entry);
        self.fn_drafter_dequant_nvfp4_to_f16 = Some(nvfp4_entry);
    }

    /// Spec-decode commit 10 (+10b/10c): copy/dequant base
    /// source-layer K/V into the drafter's F16 shadow regions.
    ///
    /// * **F16 base** — plain `cuMemcpyDtoDAsync` ×4 (sliding K/V,
    ///   full K/V). Zero-kernel fast path.
    /// * **FP8 base** — `gemma4_drafter_dequant_fp8_to_f16_kernel`
    ///   ×4. Element count = `shadow_layer_bytes / 2`. Reads E4M3
    ///   bytes, writes f16; bit-equivalent to the loader's
    ///   `fp8e4m3_bytes_to_f16_bytes` host helper.
    /// * **NVFP4 base** — `gemma4_drafter_dequant_nvfp4_to_f16_kernel`
    ///   ×4. Reads packed 4-bit + per-16-elem E4M3 microscales.
    ///   Decoder matches the production `fp4_decode` table used by
    ///   the NVFP4 paged-decode kernel set so the round-trip is
    ///   bit-equivalent to what FA-2 NVFP4 sees internally.
    ///
    /// `sliding_bytes` / `full_bytes` are the per-buffer (K or V)
    /// sizes the caller has already computed for the active spec
    /// request — must match `shadow_kv.{sliding,full}_layer_bytes`.
    ///
    /// Commit 20: takes separate `sliding_kv_dtype` and `full_kv_dtype`
    /// instead of a single dtype for both. On hybrid configs (e.g.
    /// NVFP4 sliding + FP8 global) layer 22 and layer 23 use different
    /// KV cache dtypes — feeding both layers through the sliding-layer
    /// dtype path silently dequanted the full source layer with the
    /// wrong codec, producing junk K/V in the shadow.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn populate_shadow_kv_from_base(
        &self,
        base_sliding_k: u64,
        base_sliding_v: u64,
        base_full_k: u64,
        base_full_v: u64,
        base_sliding_k_scale: u64,
        base_sliding_v_scale: u64,
        base_full_k_scale: u64,
        base_full_v_scale: u64,
        sliding_kv_dtype: crate::gemma4_layer_exec::KvDtype,
        full_kv_dtype: crate::gemma4_layer_exec::KvDtype,
        sliding_bytes: usize,
        full_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let shadow = self.shadow_kv.as_ref().ok_or_else(|| RvllmError::Attention {
            err: AttentionError::FeatureNotAvailable {
                op: "populate_shadow_kv_from_base: shadow_kv not attached",
                backend: "Gemma4Drafter",
            },
            ctx: AttnCtx {
                op: "populate_shadow_kv_from_base",
                stream,
                num_seqs: 1,
                head_dim: self.arch.head_dim_global as u32,
            },
            bt: std::backtrace::Backtrace::capture(),
        })?;
        if sliding_bytes != shadow.sliding_layer_bytes
            || full_bytes != shadow.full_layer_bytes
        {
            return Err(RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    op: "populate_shadow_kv_from_base: \
                         caller layer-bytes mismatch shadow layout",
                    backend: "Gemma4Drafter",
                },
                ctx: AttnCtx {
                    op: "populate_shadow_kv_from_base",
                    stream,
                    num_seqs: 1,
                    head_dim: self.arch.head_dim_global as u32,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        // F16 element counts in the shadow (one buffer = K or V).
        let sliding_elems = (sliding_bytes / 2) as i64;
        let full_elems = (full_bytes / 2) as i64;

        // Sliding source layer (K + V).
        self.populate_one_source_layer(
            shadow.sliding_k_ptr,
            shadow.sliding_v_ptr,
            base_sliding_k,
            base_sliding_v,
            base_sliding_k_scale,
            base_sliding_v_scale,
            shadow.sliding_num_kv_heads as i32,
            shadow.sliding_head_dim as i32,
            sliding_bytes,
            sliding_elems,
            sliding_kv_dtype,
            "sliding",
            stream,
        )?;

        // Full / global source layer (K + V).
        self.populate_one_source_layer(
            shadow.full_k_ptr,
            shadow.full_v_ptr,
            base_full_k,
            base_full_v,
            base_full_k_scale,
            base_full_v_scale,
            shadow.full_num_kv_heads as i32,
            shadow.full_head_dim as i32,
            full_bytes,
            full_elems,
            full_kv_dtype,
            "full",
            stream,
        )?;
        Ok(())
    }

    /// Per-source-layer dispatch helper for `populate_shadow_kv_from_base`.
    /// One call writes both K and V for ONE source layer using the dtype
    /// the base actually used at that layer (the bug commit 20 fixed:
    /// hybrid configs have different sliding vs full dtypes).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn populate_one_source_layer(
        &self,
        shadow_k: u64,
        shadow_v: u64,
        base_k: u64,
        base_v: u64,
        base_k_scale: u64,
        base_v_scale: u64,
        nkvh: i32,
        head_dim: i32,
        layer_bytes: usize,
        n_elems: i64,
        kv_dtype: crate::gemma4_layer_exec::KvDtype,
        which: &'static str,
        stream: u64,
    ) -> Result<()> {
        match kv_dtype {
            crate::gemma4_layer_exec::KvDtype::F16 => {
                let _ = (base_k_scale, base_v_scale, nkvh, head_dim);
                use cudarc::driver::sys::*;
                let do_copy = |dst: u64, src: u64, n: usize, op: &'static str|
                    -> Result<()> {
                    let rc = cuMemcpyDtoDAsync_v2(dst, src, n, stream as CUstream);
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(RvllmError::Cuda {
                            kind: rvllm_core::CudaErrorKind::MemcpyFailed,
                            op: "drafter_shadow_kv_dtoD",
                            ctx: rvllm_core::CudaCtx {
                                stream,
                                kernel: op,
                                launch: None,
                                device: 0,
                            },
                            bt: std::backtrace::Backtrace::capture(),
                        });
                    }
                    Ok(())
                };
                let k_op = if which == "sliding" { "sliding_k" } else { "full_k" };
                let v_op = if which == "sliding" { "sliding_v" } else { "full_v" };
                do_copy(shadow_k, base_k, layer_bytes, k_op)?;
                do_copy(shadow_v, base_v, layer_bytes, v_op)?;
                Ok(())
            }
            crate::gemma4_layer_exec::KvDtype::Fp8 => {
                let fn_fp8 = self.fn_drafter_dequant_fp8_to_f16.ok_or_else(|| {
                    RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            op: "populate_shadow_kv_from_base: \
                                 FP8 dequant kernel not attached",
                            backend: "Gemma4Drafter",
                        },
                        ctx: AttnCtx {
                            op: "populate_shadow_kv_from_base(fp8)",
                            stream,
                            num_seqs: 1,
                            head_dim: self.arch.head_dim_global as u32,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    }
                })?;
                self.launch_fp8_dequant_to_shadow(
                    fn_fp8, base_k, base_k_scale,
                    shadow_k, nkvh, head_dim, n_elems, stream)?;
                self.launch_fp8_dequant_to_shadow(
                    fn_fp8, base_v, base_v_scale,
                    shadow_v, nkvh, head_dim, n_elems, stream)?;
                let _ = which;
                Ok(())
            }
            crate::gemma4_layer_exec::KvDtype::Nvfp4 => {
                let fn_nvfp4 = self.fn_drafter_dequant_nvfp4_to_f16.ok_or_else(|| {
                    RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            op: "populate_shadow_kv_from_base: \
                                 NVFP4 dequant kernel not attached",
                            backend: "Gemma4Drafter",
                        },
                        ctx: AttnCtx {
                            op: "populate_shadow_kv_from_base(nvfp4)",
                            stream,
                            num_seqs: 1,
                            head_dim: self.arch.head_dim_global as u32,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    }
                })?;
                self.launch_nvfp4_dequant_to_shadow(
                    fn_nvfp4, base_k, base_k_scale,
                    shadow_k, n_elems, stream)?;
                self.launch_nvfp4_dequant_to_shadow(
                    fn_nvfp4, base_v, base_v_scale,
                    shadow_v, n_elems, stream)?;
                let _ = (nkvh, head_dim, which);
                Ok(())
            }
        }
    }

    /// Shared launch glue for `gemma4_drafter_dequant_fp8_to_f16_kernel`.
    /// 256 threads/block, grid covers `n` elements.
    ///
    /// Commit 19: now passes the per-(slot, kv_head) f32 scale buffer
    /// that the base wrote in `fused_rope_partial_fp8kv`. Without it
    /// the dequant produced raw E4M3 mantissas with all per-slot
    /// scales = 1.0 — every drafter cross-attn K/V read was off by
    /// the missing scale factor, dragging accept_rate to 0.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_fp8_dequant_to_shadow(
        &self,
        kernel: rvllm_kernels::KernelFn,
        src: u64,
        scales: u64,
        dst: u64,
        nkvh: i32,
        head_dim: i32,
        n_elems: i64,
        stream: u64,
    ) -> Result<()> {
        if n_elems <= 0 {
            return Ok(());
        }
        use cudarc::driver::sys::*;
        let block: u32 = 256;
        let grid: u32 = ((n_elems + block as i64 - 1) / block as i64) as u32;
        let mut a_src = src;
        let mut a_scales = scales;
        let mut a_dst = dst;
        let mut a_nkvh = nkvh;
        let mut a_hd = head_dim;
        let mut a_n = n_elems;
        let args: [*mut core::ffi::c_void; 6] = [
            &mut a_src    as *mut _ as *mut _,
            &mut a_scales as *mut _ as *mut _,
            &mut a_dst    as *mut _ as *mut _,
            &mut a_nkvh   as *mut _ as *mut _,
            &mut a_hd     as *mut _ as *mut _,
            &mut a_n      as *mut _ as *mut _,
        ];
        let rc = cuLaunchKernel(
            kernel.raw() as CUfunction,
            grid, 1, 1,
            block, 1, 1,
            0,
            stream as CUstream,
            args.as_ptr() as *mut *mut core::ffi::c_void,
            core::ptr::null_mut(),
        );
        if rc != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::Cuda {
                kind: rvllm_core::CudaErrorKind::LaunchFailed,
                op: "gemma4_drafter_dequant_fp8_to_f16",
                ctx: rvllm_core::CudaCtx {
                    stream,
                    kernel: "gemma4_drafter_dequant_fp8_to_f16_kernel",
                    launch: None,
                    device: 0,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        Ok(())
    }

    /// Shared launch glue for `gemma4_drafter_dequant_nvfp4_to_f16_kernel`.
    /// Same 256-thread block; passes packed + scales + dst + n.
    #[cfg(feature = "cuda")]
    unsafe fn launch_nvfp4_dequant_to_shadow(
        &self,
        kernel: rvllm_kernels::KernelFn,
        packed: u64,
        scales: u64,
        dst: u64,
        n_elems: i64,
        stream: u64,
    ) -> Result<()> {
        if n_elems <= 0 {
            return Ok(());
        }
        use cudarc::driver::sys::*;
        let block: u32 = 256;
        let grid: u32 = ((n_elems + block as i64 - 1) / block as i64) as u32;
        let mut a_packed = packed;
        let mut a_scales = scales;
        let mut a_dst = dst;
        let mut a_n = n_elems;
        let args: [*mut core::ffi::c_void; 4] = [
            &mut a_packed as *mut _ as *mut _,
            &mut a_scales as *mut _ as *mut _,
            &mut a_dst    as *mut _ as *mut _,
            &mut a_n      as *mut _ as *mut _,
        ];
        let rc = cuLaunchKernel(
            kernel.raw() as CUfunction,
            grid, 1, 1,
            block, 1, 1,
            0,
            stream as CUstream,
            args.as_ptr() as *mut *mut core::ffi::c_void,
            core::ptr::null_mut(),
        );
        if rc != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::Cuda {
                kind: rvllm_core::CudaErrorKind::LaunchFailed,
                op: "gemma4_drafter_dequant_nvfp4_to_f16",
                ctx: rvllm_core::CudaCtx {
                    stream,
                    kernel: "gemma4_drafter_dequant_nvfp4_to_f16_kernel",
                    launch: None,
                    device: 0,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        Ok(())
    }

    /// Spec-decode commit 14: cross-attention for a global-source
    /// drafter layer (layer 3 on E4B). Mirrors
    /// `launch_cross_attn_sliding` but reads from
    /// `shadow_kv.full_{k,v}_ptr`, uses `full_num_kv_heads /
    /// full_head_dim`, and passes `window_size_left = -1` so the
    /// FA-2 decode kernel attends to the full context.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_cross_attn_global(
        &self,
        output: u64,
        query: u64,
        block_tables: u64,
        context_lens: u64,
        scale: f32,
        stream: u64,
    ) -> Result<()> {
        let shadow = self.shadow_kv.as_ref().ok_or_else(|| RvllmError::Attention {
            err: AttentionError::FeatureNotAvailable {
                op: "launch_cross_attn_global: shadow_kv not attached",
                backend: "Gemma4Drafter",
            },
            ctx: AttnCtx {
                op: "launch_cross_attn_global",
                stream,
                num_seqs: 1,
                head_dim: self.arch.head_dim_global as u32,
            },
            bt: std::backtrace::Backtrace::capture(),
        })?;
        // Commit 15: prefer the BC=16 build so head_dim=512 fits the
        // sm_121 per-CTA dynamic-smem ceiling (~64 KiB at BC=16 vs.
        // ~128 KiB at BC=32). Falls back to BC=32 if the BC=16 module
        // wasn't loaded (older engine that missed `ensure_drafter`'s
        // BC=16 attach — kept defensive even though the load is
        // unconditional in the current build).
        let (fn_decode, fa2_bc) = match self.fn_flash_attention_2_decode_f16io_bc16 {
            Some(k) => (k, 16i32),
            None => {
                let fb = self.fn_flash_attention_2_decode_f16io.ok_or_else(|| {
                    RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            op: "launch_cross_attn_global: neither BC=16 nor BC=32 \
                                 f16io kernel attached",
                            backend: "Gemma4Drafter",
                        },
                        ctx: AttnCtx {
                            op: "launch_cross_attn_global",
                            stream,
                            num_seqs: 1,
                            head_dim: self.arch.head_dim_global as u32,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    }
                })?;
                (fb, 32i32)
            }
        };
        use cudarc::driver::sys::*;
        const FA2_THREADS: i32 = 128;
        let num_heads = self.arch.num_attention_heads as i32;
        let num_kv_heads = shadow.full_num_kv_heads as i32;
        let head_dim = shadow.full_head_dim as i32;
        let smem_bytes =
            2 * fa2_bc * head_dim * 4 + fa2_bc * 4 + (FA2_THREADS / 32) * 4;
        const SM121_MAX_DYN_SMEM_BYTES: i32 = 96 * 1024;
        if smem_bytes > SM121_MAX_DYN_SMEM_BYTES {
            return Err(RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    op: "launch_cross_attn_global: smem requirement \
                         exceeds the sm_121 ~100 KiB per-CTA cap even \
                         at BC=16; head_dim probably too large for \
                         the current kernel build",
                    backend: "Gemma4Drafter",
                },
                ctx: AttnCtx {
                    op: "launch_cross_attn_global",
                    stream,
                    num_seqs: 1,
                    head_dim: head_dim as u32,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if smem_bytes as u32 >= 48 * 1024 {
            let rc = cuFuncSetAttribute(
                fn_decode.raw() as CUfunction,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem_bytes,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "launch_cross_attn_global: cuFuncSetAttribute",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let mut a_out = output;
        let mut a_q = query;
        let mut a_k = shadow.full_k_ptr;
        let mut a_v = shadow.full_v_ptr;
        let mut a_bt = block_tables;
        let mut a_cl = context_lens;
        let mut a_scale = scale;
        let mut a_nh = num_heads;
        let mut a_nkvh = num_kv_heads;
        let mut a_hd = head_dim;
        let mut a_bs = shadow.block_size as i32;
        let mut a_mbps = shadow.max_blocks_per_seq as i32;
        let mut a_win: i32 = -1; // global attention — no window
        let args: [*mut core::ffi::c_void; 13] = [
            &mut a_out  as *mut _ as *mut _,
            &mut a_q    as *mut _ as *mut _,
            &mut a_k    as *mut _ as *mut _,
            &mut a_v    as *mut _ as *mut _,
            &mut a_bt   as *mut _ as *mut _,
            &mut a_cl   as *mut _ as *mut _,
            &mut a_scale as *mut _ as *mut _,
            &mut a_nh   as *mut _ as *mut _,
            &mut a_nkvh as *mut _ as *mut _,
            &mut a_hd   as *mut _ as *mut _,
            &mut a_bs   as *mut _ as *mut _,
            &mut a_mbps as *mut _ as *mut _,
            &mut a_win  as *mut _ as *mut _,
        ];
        let rc = cuLaunchKernel(
            fn_decode.raw() as CUfunction,
            1, num_heads as u32, 1,
            FA2_THREADS as u32, 1, 1,
            smem_bytes as u32,
            stream as CUstream,
            args.as_ptr() as *mut *mut core::ffi::c_void,
            core::ptr::null_mut(),
        );
        if rc != CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::cuda(
                "launch_cross_attn_global: flash_attention_2_decode_f16io",
                rvllm_core::CudaErrorKind::LaunchFailed,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        Ok(())
    }

    /// Spec-decode commit 11: launch the assistant's cross-attention
    /// for a sliding-source layer. Wraps the shared
    /// `flash_attention_2_decode_f16io_kernel` against the drafter's
    /// F16 shadow K/V — no new attention kernel.
    ///
    /// Arguments:
    /// * `query` — drafter post-RoPE Q, f16
    ///   `[num_heads * head_dim]` = workspace.q.
    /// * `output` — f16 `[num_heads * head_dim]` for the attention
    ///   output (= workspace.attn_out).
    /// * `block_tables` / `context_lens` — same shapes the base
    ///   path expects (identity table + i32[1] ctx_len).
    /// * `sliding_window` — sliding-window size on the source
    ///   layer (positive), or `-1` for the future global-layer
    ///   variant.
    ///
    /// # Safety
    /// `query` / `output` / `block_tables` / `context_lens` must be
    /// valid device pointers of the stated shape. The shadow region
    /// pointers come from `self.shadow_kv` and were initialised by
    /// `ensure_drafter`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_cross_attn_sliding(
        &self,
        output: u64,
        query: u64,
        block_tables: u64,
        context_lens: u64,
        scale: f32,
        sliding_window: i32,
        stream: u64,
    ) -> Result<()> {
        let shadow = self.shadow_kv.as_ref().ok_or_else(|| RvllmError::Attention {
            err: AttentionError::FeatureNotAvailable {
                op: "launch_cross_attn_sliding: shadow_kv not attached",
                backend: "Gemma4Drafter",
            },
            ctx: AttnCtx {
                op: "launch_cross_attn_sliding",
                stream,
                num_seqs: 1,
                head_dim: self.arch.head_dim_sliding as u32,
            },
            bt: std::backtrace::Backtrace::capture(),
        })?;
        let fn_decode = self.fn_flash_attention_2_decode_f16io.ok_or_else(|| {
            RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    op: "launch_cross_attn_sliding: flash_attention_2_decode_f16io \
                         kernel not attached",
                    backend: "Gemma4Drafter",
                },
                ctx: AttnCtx {
                    op: "launch_cross_attn_sliding",
                    stream,
                    num_seqs: 1,
                    head_dim: self.arch.head_dim_sliding as u32,
                },
                bt: std::backtrace::Backtrace::capture(),
            }
        })?;
        use cudarc::driver::sys::*;
        const FA2_THREADS: i32 = 128;
        const FA2_BC: i32 = 32;
        let num_heads = self.arch.num_attention_heads as i32;
        let num_kv_heads = shadow.sliding_num_kv_heads as i32;
        let head_dim = shadow.sliding_head_dim as i32;
        let smem_bytes =
            2 * FA2_BC * head_dim * 4 + FA2_BC * 4 + (FA2_THREADS / 32) * 4;
        if smem_bytes as u32 >= 48 * 1024 {
            let rc = cuFuncSetAttribute(
                fn_decode.raw() as CUfunction,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem_bytes,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "launch_cross_attn_sliding: cuFuncSetAttribute",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let mut a_out = output;
        let mut a_q = query;
        let mut a_k = shadow.sliding_k_ptr;
        let mut a_v = shadow.sliding_v_ptr;
        let mut a_bt = block_tables;
        let mut a_cl = context_lens;
        let mut a_scale = scale;
        let mut a_nh = num_heads;
        let mut a_nkvh = num_kv_heads;
        let mut a_hd = head_dim;
        let mut a_bs = shadow.block_size as i32;
        let mut a_mbps = shadow.max_blocks_per_seq as i32;
        let mut a_win: i32 = sliding_window;
        let args: [*mut core::ffi::c_void; 13] = [
            &mut a_out  as *mut _ as *mut _,
            &mut a_q    as *mut _ as *mut _,
            &mut a_k    as *mut _ as *mut _,
            &mut a_v    as *mut _ as *mut _,
            &mut a_bt   as *mut _ as *mut _,
            &mut a_cl   as *mut _ as *mut _,
            &mut a_scale as *mut _ as *mut _,
            &mut a_nh   as *mut _ as *mut _,
            &mut a_nkvh as *mut _ as *mut _,
            &mut a_hd   as *mut _ as *mut _,
            &mut a_bs   as *mut _ as *mut _,
            &mut a_mbps as *mut _ as *mut _,
            &mut a_win  as *mut _ as *mut _,
        ];
        let rc = cuLaunchKernel(
            fn_decode.raw() as CUfunction,
            1, num_heads as u32, 1,
            FA2_THREADS as u32, 1, 1,
            smem_bytes as u32,
            stream as CUstream,
            args.as_ptr() as *mut *mut core::ffi::c_void,
            core::ptr::null_mut(),
        );
        if rc != CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::cuda(
                "launch_cross_attn_sliding: flash_attention_2_decode_f16io",
                rvllm_core::CudaErrorKind::LaunchFailed,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        Ok(())
    }

    /// Non-cuda build stub. Linkable but the path is unreachable
    /// without the `cuda` feature. The runtime is gated by
    /// `ServerConfig::spec_decode`, which is itself unreachable from
    /// the non-cuda mock-worker code path, so this never fires in
    /// practice — it's here so the crate keeps compiling under
    /// `cargo build` without features.
    #[cfg(not(feature = "cuda"))]
    pub fn load_mock(layout: &Gemma4DrafterWeightLayout) -> Result<Self> {
        Ok(Self {
            arch: layout.arch.clone(),
            top: Gemma4DrafterTopPtrs {
                centroids: 0, token_ordering: 0, embed_tokens: 0,
                pre_projection: 0, post_projection: 0, final_norm: 0,
            },
            layers: layout.layers.iter().map(|lo| Gemma4DrafterLayerPtrs {
                input_layernorm: 0, post_attention_layernorm: 0,
                pre_feedforward_layernorm: 0, post_feedforward_layernorm: 0,
                layer_scalar: 0,
                mlp_gate_proj: 0, mlp_up_proj: 0, mlp_down_proj: 0,
                self_attn_q_proj: 0, self_attn_q_norm: 0, self_attn_o_proj: 0,
                effective_head_dim: lo.effective_head_dim,
                layer_type: lo.layer_type,
            }).collect(),
            bytes_resident: 0,
            shard_path: layout.shard_path.clone(),
            masked_embedder_mod: None,
            fn_masked_embedder_argmax_f16: None,
            shadow_kv: None,
            flash_attention_mod: None,
            fn_flash_attention_2_decode_f16io: None,
            flash_attention_bc16_mod: None,
            fn_flash_attention_2_decode_f16io_bc16: None,
            drafter_dequant_mod: None,
            fn_drafter_dequant_fp8_to_f16: None,
            fn_drafter_dequant_nvfp4_to_f16: None,
        })
    }

    /// Forward stub — one MTP step.
    ///
    /// **Commit 4 status**: returns `FeatureNotAvailable`. The
    /// cross-attention kernel (commit 5) and MaskedEmbedder kernel
    /// (commit 6) aren't wired yet, so a clean error is the right
    /// behaviour. Once those land, this method drives:
    ///
    ///   pre_projection([emb_last_token; base_hidden_last_step])
    ///     → 4 layers of (norm → q_proj → q_norm → RoPE
    ///                    → cross-attn(base sliding/full K-V)
    ///                    → o_proj → MLP → norms + residuals)
    ///     → model.norm
    ///     → MaskedEmbedder(hidden, embed_tokens.weight)
    ///   → (vocab logits, post_projection(hidden))
    ///
    /// Inputs are grouped in [`DrafterForwardStep`] so the real
    /// implementation has the full paged-cache view needed by
    /// cross-attention. The previous stub only carried raw K/V base
    /// pointers, which was insufficient for block-table lookup and
    /// FP8/NVFP4 scale handling.
    /// - `stream`: CUDA stream id.
    pub unsafe fn forward_step(
        &self,
        step: DrafterForwardStep,
        workspace: &DrafterStepWorkspace,
        stream: u64,
    ) -> Result<()> {
        self.prepare_pre_projection_input(&step, workspace, stream)?;
        Err(RvllmError::Attention {
            err: AttentionError::FeatureNotAvailable {
                op: "Gemma4DrafterRuntime::forward_step \
                     (pre_projection + cross-attn + MaskedEmbedder pending)",
                backend: "Gemma4Drafter",
            },
            ctx: AttnCtx {
                op: "Gemma4DrafterRuntime::forward_step",
                stream,
                num_seqs: 1,
                head_dim: self.arch.head_dim_global as u32,
            },
            bt: std::backtrace::Backtrace::capture(),
        })
    }

    /// Build the assistant pre-projection input:
    ///
    /// ```text
    /// pre_projection_in = [last_token_embed ; base_hidden_last_step]
    /// ```
    ///
    /// Both halves are base-hidden-width f16 device vectors. This is
    /// intentionally implemented with DtoD copies instead of a custom
    /// CUDA kernel, so the first real drafter operation is simple and
    /// easy to validate before we add the cross-attention and
    /// MaskedEmbedder kernels.
    pub unsafe fn prepare_pre_projection_input(
        &self,
        step: &DrafterForwardStep,
        workspace: &DrafterStepWorkspace,
        stream: u64,
    ) -> Result<()> {
        let hd = self.arch.head_dim_global as u32;
        validate_ptr("drafter.last_token_embed", step.last_token_embed, stream, hd)?;
        validate_ptr("drafter.base_hidden_last_step", step.base_hidden_last_step, stream, hd)?;
        validate_ptr("drafter.pre_projection_in", workspace.pre_projection_in, stream, hd)?;

        #[cfg(feature = "cuda")]
        {
            let half_bytes = self.arch.backbone_hidden_size * 2;
            let r0 = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                workspace.pre_projection_in,
                step.last_token_embed,
                half_bytes,
                stream as cudarc::driver::sys::CUstream,
            );
            if r0 != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(RvllmError::Cuda {
                    kind: rvllm_core::CudaErrorKind::MemcpyFailed,
                    op: "drafter_pre_projection_last_token_dtoD",
                    ctx: rvllm_core::CudaCtx {
                        stream,
                        kernel: "cuMemcpyDtoDAsync_v2",
                        launch: None,
                        device: 0,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
            let r1 = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                workspace.pre_projection_in + half_bytes as u64,
                step.base_hidden_last_step,
                half_bytes,
                stream as cudarc::driver::sys::CUstream,
            );
            if r1 != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(RvllmError::Cuda {
                    kind: rvllm_core::CudaErrorKind::MemcpyFailed,
                    op: "drafter_pre_projection_base_hidden_dtoD",
                    ctx: rvllm_core::CudaCtx {
                        stream,
                        kernel: "cuMemcpyDtoDAsync_v2",
                        launch: None,
                        device: 0,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
        }
        Ok(())
    }
}

/// Lazy slot held inside `Gemma4Bringup`. Commit 7 will populate it
/// on the first request that observes `ServerConfig::spec_decode`.
/// Default state is an empty mutex — when `spec_decode` is false the
/// drafter is never constructed and consumes zero HBM.
pub type DrafterSlot = Mutex<Option<Gemma4DrafterRuntime>>;

/// Launch the MaskedEmbedder fused kernel
/// (`kernels/gemma4_masked_embedder.cu`). One CTA, 256 threads,
/// single-token argmax over a sparse `top_k * per_centroid` candidate
/// set (4096 tokens on E4B default sizing). The kernel is loaded
/// through the standard `KernelLoader::load_ptx` path; pass the
/// resolved `KernelFn` in.
///
/// # Pointers
///   * `hidden` — f16 [hidden_size], the assistant's pre-lm-head
///     hidden state for the position being scored.
///   * `centroids` — f16 [num_centroids, hidden_size].
///   * `token_ordering` — i64 [vocab].
///   * `lm_head` — f16 [vocab, hidden_size]. Tied to
///     `model.embed_tokens.weight` per HF.
///   * `out_token_id` — i32 [1], filled with the argmax token id.
///   * `out_logit` — f32 [1] or 0 to skip writing the top logit.
///
/// # Safety
/// All non-null pointers must be valid device addresses of the stated
/// dtype + length. Kernel performs no bounds checks beyond the
/// `tok < 0 || tok >= vocab` guard on `token_ordering` reads.
///
/// # Smem
/// Sized as
///   `hidden_size*4 + num_centroids*4 + top_k*4 + 8*4 + 8*4` bytes.
/// At E4B defaults (hidden=256, n_cent=2048, top_k=32) that's 9408
/// bytes — fits in the default 48 KiB allowance without
/// `cuFuncSetAttribute`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_embedder_argmax_f16(
    fn_masked: rvllm_kernels::KernelFn,
    hidden: u64,
    centroids: u64,
    token_ordering: u64,
    lm_head: u64,
    hidden_size: i32,
    n_centroids: i32,
    top_k: i32,
    per_centroid: i32,
    vocab: i32,
    out_token_id: u64,
    out_logit: u64,
    stream: u64,
) -> Result<()> {
    if hidden_size <= 0 || n_centroids <= 0 || top_k <= 0
        || per_centroid <= 0 || vocab <= 0
    {
        return Err(RvllmError::Attention {
            err: AttentionError::FeatureNotAvailable {
                op: "launch_masked_embedder_argmax_f16: \
                     non-positive dimension argument",
                backend: "Gemma4Drafter",
            },
            ctx: AttnCtx {
                op: "launch_masked_embedder_argmax_f16",
                stream,
                num_seqs: 1,
                head_dim: hidden_size.max(0) as u32,
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    // The kernel hard-codes 8 warp-best slots in smem (see
    // `warp_best_logits + 8` and `warp_best_ids + 8` in the .cu).
    // Block sizes that produce more than 8 warps would overrun the
    // smem reservation; refuse to launch.
    const BLOCK: u32 = 256;
    let num_warps = BLOCK / 32;
    debug_assert!(num_warps <= 8);

    let smem_bytes: u32 = (hidden_size as u32 * 4)
        + (n_centroids as u32 * 4)
        + (top_k as u32 * 4)
        + (num_warps * 4)   // warp_best_logits
        + (num_warps * 4);  // warp_best_ids

    use cudarc::driver::sys::*;
    let mut a_hidden = hidden;
    let mut a_cent = centroids;
    let mut a_tok_ord = token_ordering;
    let mut a_lm = lm_head;
    let mut a_hs = hidden_size;
    let mut a_nc = n_centroids;
    let mut a_tk = top_k;
    let mut a_pc = per_centroid;
    let mut a_vc = vocab;
    let mut a_out_id = out_token_id;
    let mut a_out_lg = out_logit;
    let args: [*mut core::ffi::c_void; 11] = [
        &mut a_hidden  as *mut _ as *mut _,
        &mut a_cent    as *mut _ as *mut _,
        &mut a_tok_ord as *mut _ as *mut _,
        &mut a_lm      as *mut _ as *mut _,
        &mut a_hs      as *mut _ as *mut _,
        &mut a_nc      as *mut _ as *mut _,
        &mut a_tk      as *mut _ as *mut _,
        &mut a_pc      as *mut _ as *mut _,
        &mut a_vc      as *mut _ as *mut _,
        &mut a_out_id  as *mut _ as *mut _,
        &mut a_out_lg  as *mut _ as *mut _,
    ];
    let rc = cuLaunchKernel(
        fn_masked.raw() as CUfunction,
        1, 1, 1,
        BLOCK, 1, 1,
        smem_bytes,
        stream as CUstream,
        args.as_ptr() as *mut *mut core::ffi::c_void,
        core::ptr::null_mut(),
    );
    if rc != CUresult::CUDA_SUCCESS {
        return Err(RvllmError::Cuda {
            kind: rvllm_core::CudaErrorKind::LaunchFailed,
            op: "gemma4_masked_embedder_argmax_f16",
            ctx: rvllm_core::CudaCtx {
                stream,
                kernel: "gemma4_masked_embedder_argmax_f16_kernel",
                launch: None,
                device: 0,
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    Ok(())
}

/// Eager BF16 → F16 element-wise conversion. Each input pair of
/// bytes is a single BF16 (1 sign / 8 exp / 7 mantissa); each output
/// pair is the matching F16 (1 sign / 5 exp / 10 mantissa). Uses the
/// `half::bf16 -> f32 -> half::f16` round-trip — denormals,
/// infinities and NaN propagate consistently with the loader's
/// existing `bf16_bytes_to_f16_bytes` helper.
fn bf16_to_f16_bytes(raw: &[u8]) -> Vec<u8> {
    let n = raw.len() / 2;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let lo = raw[2 * i];
        let hi = raw[2 * i + 1];
        // BF16 packs into the high 16 bits of an IEEE-754 f32. Read
        // the two-byte BF16 word as the top half of an f32, zero the
        // bottom half, then narrow to f16.
        let as_f32 = f32::from_bits(u32::from_le_bytes([0, 0, lo, hi]));
        out.extend_from_slice(&half::f16::from_f32(as_f32).to_le_bytes());
    }
    out
}

fn corrupt(path: &Path, detail: String) -> RvllmError {
    RvllmError::Loader {
        err: rvllm_core::LoaderError::Corrupt { detail },
        ctx: rvllm_core::LoaderCtx {
            path: path.to_path_buf(),
            tensor: None,
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_to_f16_roundtrip_known_values() {
        // BF16 of 1.0 = 0x3F80 (LE: 0x80, 0x3F).
        // BF16 of -2.0 = 0xC000 (LE: 0x00, 0xC0).
        let raw = vec![0x80, 0x3F, 0x00, 0xC0];
        let out = bf16_to_f16_bytes(&raw);
        assert_eq!(out.len(), 4);
        let v0 = half::f16::from_le_bytes([out[0], out[1]]).to_f32();
        let v1 = half::f16::from_le_bytes([out[2], out[3]]).to_f32();
        assert_eq!(v0, 1.0_f32);
        assert_eq!(v1, -2.0_f32);
    }
}
