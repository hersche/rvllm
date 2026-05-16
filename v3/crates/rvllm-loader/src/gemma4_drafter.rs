//! Gemma 4 E4B-it assistant-drafter — config parser + safetensors
//! header validator.
//!
//! Commit 2 of the speculative-decode landing. This module reads the
//! `Gemma4AssistantForCausalLM` checkpoint at
//! `/home/r00t/gemma4-e4b-assistant/` (or whatever path the operator
//! sets via `RVLLM_GEMMA4_DRAFTER_DIR`) and:
//!
//! 1. Parses `config.json` into `Gemma4DrafterArch` (4 layers,
//!    hidden=256, vocab=262144, 3 sliding-attention + 1 full-attention,
//!    num_centroids=2048).
//! 2. Loads `model.safetensors` header + asserts the expected tensor
//!    name/shape/dtype set. The asserted layout is the one observed
//!    on disk on 2026-05-15:
//!
//!      masked_embedding.centroids.weight       BF16 [2048, 256]
//!      masked_embedding.token_ordering         I64  [262144]
//!      model.embed_tokens.weight               BF16 [262144, 256]
//!      pre_projection.weight                   BF16 [256, 5120]
//!      post_projection.weight                  BF16 [2560, 256]
//!      model.norm.weight                       BF16 [256]
//!      per-layer (4×):
//!        input_layernorm.weight                BF16 [256]
//!        post_attention_layernorm.weight       BF16 [256]
//!        pre_feedforward_layernorm.weight      BF16 [256]
//!        post_feedforward_layernorm.weight     BF16 [256]
//!        layer_scalar                          BF16 [1]
//!        mlp.gate_proj.weight                  BF16 [2048, 256]
//!        mlp.up_proj.weight                    BF16 [2048, 256]
//!        mlp.down_proj.weight                  BF16 [256, 2048]
//!        self_attn.q_proj.weight               BF16 [1024 | 2048, 256]
//!        self_attn.q_norm.weight               BF16 [256 | 512]
//!        self_attn.o_proj.weight               BF16 [256, 1024 | 2048]
//!
//! Three notes that diverge from the original spec-decode plan:
//!
//!   * Centroids and ordered embedding are namespaced under
//!     `masked_embedding.*`, NOT `centroid_decoder` / `ordered_embed`.
//!   * The drafter only carries `q_proj`/`q_norm`/`o_proj` per layer
//!     — no `k_proj`, no `v_proj`, no `k_norm`. The 4-layer mini
//!     decoder reads K/V from the base model's KV cache via the
//!     `post_projection` → drafter-hidden bridge. That is, the
//!     drafter is KV-SHARED with the base by design, not unshared
//!     (the original plan's default).
//!   * Layer 3 is the global-attention layer (`layer_types` ends in
//!     `"full_attention"`) and uses `global_head_dim=512` while
//!     layers 0..=2 use `head_dim=256` (sliding). q_proj / q_norm /
//!     o_proj shapes follow accordingly.
//!
//! Commit 2 does not upload weights — it produces the offset table
//! and validates the shard. Commit 3 (forward skeleton) will use
//! the same offsets to drive BF16→F16 upload from the mmap into the
//! shared HBM arena.

use std::path::{Path, PathBuf};

use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};

use crate::safetensors::ShardHeader;

/// Layer attention class (drafter has only sliding + full).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrafterLayerType {
    Sliding,
    Full,
}

/// Parsed `config.json` for `Gemma4AssistantForCausalLM`. Stores the
/// fields the runtime needs at forward time; ignores the rest.
#[derive(Clone, Debug)]
pub struct Gemma4DrafterArch {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim_sliding: usize,
    pub head_dim_global: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub rms_norm_eps: f32,
    pub layer_types: Vec<DrafterLayerType>,

    /// Centroid codebook size. 2048 on E4B drafter.
    pub num_centroids: usize,
    /// Top-K centroids selected by the drafter's logit head before
    /// expansion into the full vocab via `masked_embedding.token_ordering`.
    pub centroid_intermediate_top_k: usize,
    /// Base model's hidden size — `post_projection` projects drafter
    /// output (256) BACK into this space (2560 on E4B-it). Must
    /// match the base bring-up's `hidden_size` at wire time.
    pub backbone_hidden_size: usize,
    /// Width of the `pre_projection` input — `[hidden_size, this]`.
    /// 5120 on E4B drafter (= 2 × `backbone_hidden_size`).
    pub pre_projection_in_dim: usize,
    /// Top-level `use_ordered_embeddings` flag from the drafter's
    /// `config.json`. When `true` (e.g. E4B-it assistant) the drafter
    /// ships `masked_embedding.centroids.weight` +
    /// `masked_embedding.token_ordering` and the decoder runs the
    /// centroid → top-K-vocab masked-embedding head. When `false`
    /// (e.g. 31B-it assistant) those tensors are absent and the
    /// decoder runs a full-vocab tied LM head against
    /// `model.embed_tokens.weight` directly. Mirrors HF
    /// `Gemma4AssistantForCausalLM.forward()` which only constructs
    /// `masked_embedding` when this flag is set.
    pub use_ordered_embeddings: bool,
}

/// One drafter layer's tensor offsets in the safetensors shard.
#[derive(Clone, Debug)]
pub struct Gemma4DrafterLayerOffsets {
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
    /// Per-layer effective head dim: `head_dim_sliding` for Sliding,
    /// `head_dim_global` for Full.
    pub effective_head_dim: usize,
    /// Same source as `arch.layer_types[i]` — kept inline so the
    /// runtime side doesn't need to re-index.
    pub layer_type: DrafterLayerType,
}

/// Top-level tensor offsets (everything not in a per-layer block).
#[derive(Clone, Debug)]
pub struct Gemma4DrafterTopOffsets {
    /// `masked_embedding.centroids.weight` — present iff
    /// `arch.use_ordered_embeddings == true`. `None` on 31B drafter.
    pub centroids: Option<u64>,
    /// `masked_embedding.token_ordering` (i64!) — present iff
    /// `arch.use_ordered_embeddings == true`. `None` on 31B drafter.
    pub token_ordering: Option<u64>,
    pub embed_tokens: u64,      // model.embed_tokens.weight
    pub pre_projection: u64,    // pre_projection.weight
    pub post_projection: u64,   // post_projection.weight
    pub final_norm: u64,        // model.norm.weight
}

/// Drafter weight layout — arch metadata + safetensors offsets. Does
/// NOT include device pointers; uploading happens in commit 3.
#[derive(Clone, Debug)]
pub struct Gemma4DrafterWeightLayout {
    pub arch: Gemma4DrafterArch,
    /// Absolute path of the safetensors shard the offsets refer to.
    pub shard_path: PathBuf,
    pub top: Gemma4DrafterTopOffsets,
    pub layers: Vec<Gemma4DrafterLayerOffsets>,
}

impl Gemma4DrafterArch {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let bytes = std::fs::read(&p).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: p.clone(),
            source,
        })?;
        let v: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| corrupt(&p, format!("config.json: {e}")))?;

        let arch_name = v.get("architectures")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(|s| s.as_str())
            .unwrap_or("");
        if arch_name != "Gemma4AssistantForCausalLM" {
            return Err(corrupt(&p, format!(
                "expected architectures=[Gemma4AssistantForCausalLM], got {arch_name:?}")));
        }
        let tc = v.get("text_config")
            .ok_or_else(|| corrupt(&p, "missing text_config".into()))?;
        let geti = |k: &str| -> Result<usize> {
            tc.get(k).and_then(|x| x.as_u64()).map(|n| n as usize)
                .ok_or_else(|| corrupt(&p, format!("text_config.{k} missing/not-uint")))
        };
        let getf = |k: &str| -> Result<f32> {
            tc.get(k).and_then(|x| x.as_f64()).map(|n| n as f32)
                .ok_or_else(|| corrupt(&p, format!("text_config.{k} missing/not-float")))
        };

        let hidden_size = geti("hidden_size")?;
        let intermediate_size = geti("intermediate_size")?;
        let num_hidden_layers = geti("num_hidden_layers")?;
        let num_attention_heads = geti("num_attention_heads")?;
        let num_key_value_heads = geti("num_key_value_heads")?;
        let head_dim_sliding = geti("head_dim")?;
        let head_dim_global = geti("global_head_dim")?;
        let vocab_size = geti("vocab_size")?;
        let max_position_embeddings = geti("max_position_embeddings")?;
        let sliding_window = geti("sliding_window")?;
        let rms_norm_eps = getf("rms_norm_eps")?;
        let layer_type_strs: Vec<String> = tc.get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or_else(|| corrupt(&p, "text_config.layer_types missing".into()))?
            .iter().filter_map(|s| s.as_str().map(String::from)).collect();
        if layer_type_strs.len() != num_hidden_layers {
            return Err(corrupt(&p, format!(
                "layer_types len {} != num_hidden_layers {}",
                layer_type_strs.len(), num_hidden_layers)));
        }
        let layer_types: Vec<DrafterLayerType> = layer_type_strs.iter().map(|s| match s.as_str() {
            "sliding_attention" => DrafterLayerType::Sliding,
            "full_attention" => DrafterLayerType::Full,
            other => panic!("unsupported drafter layer_type {other:?}"),
        }).collect();

        let num_centroids = v.get("num_centroids")
            .and_then(|x| x.as_u64()).map(|n| n as usize)
            .ok_or_else(|| corrupt(&p, "num_centroids missing".into()))?;
        let centroid_intermediate_top_k = v.get("centroid_intermediate_top_k")
            .and_then(|x| x.as_u64()).map(|n| n as usize)
            .ok_or_else(|| corrupt(&p, "centroid_intermediate_top_k missing".into()))?;
        let backbone_hidden_size = v.get("backbone_hidden_size")
            .and_then(|x| x.as_u64()).map(|n| n as usize)
            .ok_or_else(|| corrupt(&p, "backbone_hidden_size missing".into()))?;
        // pre_projection_in_dim isn't a config field — it's derived
        // from the safetensors shape (2 × backbone_hidden_size on
        // E4B-it). Validated against the actual shard in
        // `Gemma4DrafterWeightLayout::from_dir`.
        let pre_projection_in_dim = 2 * backbone_hidden_size;
        // Top-level `use_ordered_embeddings` selects the decoder
        // head (centroid masked-embedding vs full-vocab tied LM
        // head). E4B drafter ships `true`; 31B drafter ships
        // `false`. Default conservatively to `true` for legacy
        // checkpoints that omit the field — those would already be
        // E4B-style by virtue of shipping the masked-embedding
        // tensors.
        let use_ordered_embeddings = v.get("use_ordered_embeddings")
            .and_then(|x| x.as_bool())
            .unwrap_or(true);

        Ok(Self {
            hidden_size, intermediate_size, num_hidden_layers,
            num_attention_heads, num_key_value_heads,
            head_dim_sliding, head_dim_global, vocab_size,
            max_position_embeddings, sliding_window, rms_norm_eps,
            layer_types,
            num_centroids, centroid_intermediate_top_k,
            backbone_hidden_size, pre_projection_in_dim,
            use_ordered_embeddings,
        })
    }
}

impl Gemma4DrafterWeightLayout {
    /// Parse arch + safetensors header. Validates that every tensor
    /// listed above is present with the expected dtype + shape.
    /// Returns the offset table; no GPU work, no weight copy.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let arch = Gemma4DrafterArch::from_dir(dir)?;
        let shard_path = dir.join("model.safetensors");
        let bytes = std::fs::read(&shard_path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: shard_path.clone(),
            source,
        })?;
        let hdr = ShardHeader::parse(&shard_path, &bytes)?;

        let require = |name: &str, want_dtype: DType, want_shape: &[usize]| -> Result<u64> {
            let entry = hdr.tensors.get(name).ok_or_else(|| corrupt(
                &shard_path, format!("missing tensor {name}")))?;
            if entry.dtype != want_dtype {
                return Err(corrupt(&shard_path, format!(
                    "tensor {name}: dtype {:?} != expected {:?}",
                    entry.dtype, want_dtype)));
            }
            if entry.shape.as_slice() != want_shape {
                return Err(corrupt(&shard_path, format!(
                    "tensor {name}: shape {:?} != expected {:?}",
                    entry.shape, want_shape)));
            }
            Ok(entry.file_offset)
        };

        let hidden = arch.hidden_size;
        let inter = arch.intermediate_size;
        let vocab = arch.vocab_size;
        let n_cent = arch.num_centroids;
        let backbone = arch.backbone_hidden_size;

        // 31B drafter (`use_ordered_embeddings=false`) does not ship
        // the centroid masked-embedding tensors; only require them
        // when the config flag asks for the masked-embedding head.
        let (centroids, token_ordering) = if arch.use_ordered_embeddings {
            (
                Some(require(
                    "masked_embedding.centroids.weight",
                    DType::Bf16, &[n_cent, hidden])?),
                Some(require(
                    "masked_embedding.token_ordering",
                    DType::I64, &[vocab])?),
            )
        } else {
            (None, None)
        };

        let top = Gemma4DrafterTopOffsets {
            centroids,
            token_ordering,
            embed_tokens: require(
                "model.embed_tokens.weight",
                DType::Bf16, &[vocab, hidden])?,
            pre_projection: require(
                "pre_projection.weight",
                DType::Bf16, &[hidden, arch.pre_projection_in_dim])?,
            post_projection: require(
                "post_projection.weight",
                DType::Bf16, &[backbone, hidden])?,
            final_norm: require(
                "model.norm.weight",
                DType::Bf16, &[hidden])?,
        };

        let mut layers = Vec::with_capacity(arch.num_hidden_layers);
        for (li, lt) in arch.layer_types.iter().enumerate() {
            // Per-layer effective head dim: sliding uses
            // head_dim_sliding (256); global/full uses head_dim_global
            // (512). q_proj output and o_proj input pick up the
            // matching width.
            let eff_hd = match lt {
                DrafterLayerType::Sliding => arch.head_dim_sliding,
                DrafterLayerType::Full => arch.head_dim_global,
            };
            // q_proj output rows = num_attention_heads × eff_hd. With
            // the on-disk file (num_attention_heads=4): sliding rows
            // = 1024, full rows = 2048.
            let q_rows = arch.num_attention_heads * eff_hd;
            // q_norm width is eff_hd (NOT q_rows). On E4B that's 256
            // / 512.
            let q_norm_dim = eff_hd;

            let lp = |t: &str| format!("model.layers.{li}.{t}");
            layers.push(Gemma4DrafterLayerOffsets {
                input_layernorm: require(
                    &lp("input_layernorm.weight"), DType::Bf16, &[hidden])?,
                post_attention_layernorm: require(
                    &lp("post_attention_layernorm.weight"), DType::Bf16, &[hidden])?,
                pre_feedforward_layernorm: require(
                    &lp("pre_feedforward_layernorm.weight"), DType::Bf16, &[hidden])?,
                post_feedforward_layernorm: require(
                    &lp("post_feedforward_layernorm.weight"), DType::Bf16, &[hidden])?,
                layer_scalar: require(
                    &lp("layer_scalar"), DType::Bf16, &[1])?,
                mlp_gate_proj: require(
                    &lp("mlp.gate_proj.weight"), DType::Bf16, &[inter, hidden])?,
                mlp_up_proj: require(
                    &lp("mlp.up_proj.weight"), DType::Bf16, &[inter, hidden])?,
                mlp_down_proj: require(
                    &lp("mlp.down_proj.weight"), DType::Bf16, &[hidden, inter])?,
                self_attn_q_proj: require(
                    &lp("self_attn.q_proj.weight"), DType::Bf16, &[q_rows, hidden])?,
                self_attn_q_norm: require(
                    &lp("self_attn.q_norm.weight"), DType::Bf16, &[q_norm_dim])?,
                self_attn_o_proj: require(
                    &lp("self_attn.o_proj.weight"), DType::Bf16, &[hidden, q_rows])?,
                effective_head_dim: eff_hd,
                layer_type: *lt,
            });
        }

        Ok(Self { arch, shard_path, top, layers })
    }
}

fn corrupt(path: &Path, detail: String) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt { detail },
        ctx: LoaderCtx { path: path.to_path_buf(), tensor: None },
        bt: std::backtrace::Backtrace::capture(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drafter_dir() -> Option<PathBuf> {
        let p = PathBuf::from("/home/r00t/gemma4-e4b-assistant");
        if p.is_dir() { Some(p) } else { None }
    }

    #[test]
    fn parses_arch() {
        let Some(d) = drafter_dir() else {
            eprintln!("skipping: drafter dir absent");
            return;
        };
        let a = Gemma4DrafterArch::from_dir(&d).expect("parse arch");
        assert_eq!(a.hidden_size, 256);
        assert_eq!(a.intermediate_size, 2048);
        assert_eq!(a.num_hidden_layers, 4);
        assert_eq!(a.num_attention_heads, 4);
        assert_eq!(a.num_key_value_heads, 2);
        assert_eq!(a.head_dim_sliding, 256);
        assert_eq!(a.head_dim_global, 512);
        assert_eq!(a.vocab_size, 262144);
        assert_eq!(a.num_centroids, 2048);
        assert_eq!(a.centroid_intermediate_top_k, 32);
        assert_eq!(a.backbone_hidden_size, 2560);
        assert_eq!(a.pre_projection_in_dim, 5120);
        assert_eq!(a.layer_types, vec![
            DrafterLayerType::Sliding, DrafterLayerType::Sliding,
            DrafterLayerType::Sliding, DrafterLayerType::Full,
        ]);
    }

    #[test]
    fn validates_safetensors_layout() {
        let Some(d) = drafter_dir() else {
            eprintln!("skipping: drafter dir absent");
            return;
        };
        let layout = Gemma4DrafterWeightLayout::from_dir(&d).expect("layout");
        assert_eq!(layout.layers.len(), 4);
        for (i, l) in layout.layers.iter().enumerate() {
            let expected = if i == 3 { 512 } else { 256 };
            assert_eq!(l.effective_head_dim, expected,
                       "layer {i} effective_head_dim");
        }
        // Layer 3 is the global one.
        assert_eq!(layout.layers[3].layer_type, DrafterLayerType::Full);
    }

    #[test]
    fn rejects_missing_dir() {
        let bogus = PathBuf::from("/this/dir/does/not/exist/rvllm-drafter");
        assert!(Gemma4DrafterArch::from_dir(&bogus).is_err());
    }
}
