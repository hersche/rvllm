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
}

impl Gemma4DrafterRuntime {
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

        Ok(Self { arch, top, layers, bytes_resident, shard_path })
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
    /// Inputs (signature pinned to the eventual real call):
    /// - `base_hidden_last_step`: [backbone_hidden] f16 — the base's
    ///   final-norm output at the last accepted token.
    /// - `last_token_embed`: [backbone_hidden] f16 — the base's
    ///   `embed_tokens` row for the last accepted token id.
    /// - `base_sliding_kv_ptr` / `base_full_kv_ptr`: device pointers
    ///   to the source-layer K/V in the base's paged cache (from
    ///   `Gemma4Bringup::assistant_kv_sources` + the per-request
    ///   layer-offset table).
    /// - `base_ctx_len_dev_ptr`: i32 [1] context length (number of
    ///   committed base tokens) for the cross-attn mask.
    /// - `stream`: CUDA stream id.
    /// - `_out_logits_ptr` / `_out_hidden_ptr`: device-side outputs
    ///   for the next MTP step's `post_projection` chain.
    pub unsafe fn forward_step(
        &self,
        _base_hidden_last_step: u64,
        _last_token_embed: u64,
        _base_sliding_kv_ptr: u64,
        _base_full_kv_ptr: u64,
        _base_ctx_len_dev_ptr: u64,
        stream: u64,
        _out_logits_ptr: u64,
        _out_hidden_ptr: u64,
    ) -> Result<()> {
        Err(RvllmError::Attention {
            err: AttentionError::FeatureNotAvailable {
                op: "Gemma4DrafterRuntime::forward_step \
                     (commit 4 stub — cross-attn + MaskedEmbedder kernels \
                     land in commits 5-6)",
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
}

/// Lazy slot held inside `Gemma4Bringup`. Commit 7 will populate it
/// on the first request that observes `ServerConfig::spec_decode`.
/// Default state is an empty mutex — when `spec_decode` is false the
/// drafter is never constructed and consumes zero HBM.
pub type DrafterSlot = Mutex<Option<Gemma4DrafterRuntime>>;

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
