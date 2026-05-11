//! Qwen 3.5 27B dense architecture summary (Phase 0 scaffolding).
//!
//! Sibling of `qwen36_arch` for the **dense** Qwen 3.5 family
//! (`model_type=qwen3_5`). The model shares the Qwen 3.6 35B-A3B's
//! linear/full hybrid attention pattern + `attn_output_gate=true`
//! finisher but replaces the 256-expert MoE FFN with a plain
//! `gate_proj → silu_mul → down_proj` dense MLP (`intermediate_size`
//! shared by all layers).
//!
//! Markers used to disambiguate from Qwen 3.6:
//!   * `model_type == "qwen3_5"` AND
//!   * `text_config.num_experts` absent or 0 AND
//!   * `text_config.intermediate_size` present (dense MLP).
//!
//! Phase 0 only loads + validates these markers and logs a summary.
//! No tensor loading, no forward pass — `Qwen35Bringup` returns a
//! "Phase 0 only" error for any generate path. See
//! `QWEN35_BRINGUP_PLAN.md` for the multi-week phase plan.

use std::path::Path;

use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_loader::{LayerAttnType, ModelArch};

/// Qwen 3.5 specific markers read from `config.json`. The base
/// transformer fields (layers, hidden_size, head_dim, etc.) live on
/// the embedded `ModelArch`.
#[derive(Debug, Clone)]
pub struct Qwen35Arch {
    pub base: ModelArch,
    pub attn_output_gate: bool,
    pub mtp_present: bool,
    pub n_linear: usize,
    pub n_full: usize,
    /// Image token id (`image_token_id` at the top level of
    /// config.json). Used by the chat-template renderer to splice
    /// per-image soft-token blocks.
    pub image_token_id: Option<u32>,
    /// Vision tower hidden size (Qwen3-VL ViT block hidden).
    pub vision_hidden_size: Option<usize>,
    /// Vision tower depth (number of ViT blocks).
    pub vision_depth: Option<usize>,
    /// Vision merger output size (= language hidden_size).
    pub vision_out_hidden_size: Option<usize>,
    /// Phase 2 prep: RoPE specifics. Qwen 3.5 27B dense ships
    /// `partial_rotary_factor = 0.25` (only 64 of 256 head_dim is
    /// rotated) and `rope_theta = 10_000_000`. The mRoPE section
    /// split [11, 11, 10] is for multimodal positional encoding
    /// (temporal/height/width) and is currently unused on the
    /// text-only decode path.
    pub rope_partial_rotary_factor: f32,
    pub rope_theta: f32,
    pub mrope_section: Option<[u32; 3]>,
}

impl Qwen35Arch {
    /// Probe a model directory and decide whether it's a Qwen 3.5
    /// dense checkpoint. Returns `Ok(None)` if the markers don't
    /// match (caller should fall through to the next family).
    pub fn from_dir(model_dir: &Path) -> Result<Option<Self>> {
        let cfg_path = model_dir.join("config.json");
        let bytes = match std::fs::read(&cfg_path) {
            Ok(b) => b,
            Err(_) => return Ok(None),
        };
        let v: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };

        // Marker 1: model_type must be qwen3_5 (top level OR text_config).
        let model_type = v["model_type"].as_str()
            .or_else(|| v["text_config"]["model_type"].as_str())
            .unwrap_or("");
        if model_type != "qwen3_5" && model_type != "qwen3_5_text" {
            return Ok(None);
        }

        let tc = if v["text_config"]["hidden_size"].is_u64() {
            &v["text_config"]
        } else {
            &v
        };

        // Marker 2: attn_output_gate must be true (shared with Qwen 3.6).
        let attn_output_gate = tc["attn_output_gate"].as_bool().unwrap_or(false);
        if !attn_output_gate {
            return Ok(None);
        }

        // Marker 3: dense MLP — no num_experts and intermediate_size present.
        let num_experts = tc["num_experts"].as_u64().unwrap_or(0);
        if num_experts != 0 {
            return Ok(None);
        }
        let intermediate_size = tc["intermediate_size"].as_u64().unwrap_or(0);
        if intermediate_size == 0 {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "qwen35 config.json: model_type=qwen3_5 + \
                         attn_output_gate=true but intermediate_size=0; \
                         this is neither a dense Qwen 3.5 nor a recognised \
                         variant. Path: {}",
                        cfg_path.display(),
                    ),
                },
                ctx: LoaderCtx { path: cfg_path, tensor: None },
                bt: std::backtrace::Backtrace::capture(),
            });
        }

        let mut base = ModelArch::from_dir(model_dir)?;
        // Qwen 3.5 nests rope under text_config.rope_parameters.rope_theta
        // (no sliding-attention sub-key like Gemma 4). Mirror the Qwen 3.6
        // probe's correction so the loader's generic from_dir doesn't fall
        // back to 10_000.
        if let Some(t) = tc["rope_parameters"]["rope_theta"].as_f64() {
            base.rope_theta = t as f32;
        } else if let Some(t) = tc["rope_theta"].as_f64() {
            base.rope_theta = t as f32;
        }
        let n_linear = base.layer_types.iter()
            .filter(|t| **t == LayerAttnType::Linear).count();
        let n_full = base.layer_types.iter()
            .filter(|t| **t == LayerAttnType::Full).count();
        // A dense Qwen 3.5 without any linear-attention layers is
        // possible in principle but we haven't seen that in the
        // checkpoints we care about; reject so we don't silently
        // assume the hybrid pattern is present.
        if n_linear == 0 {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: "qwen35: layer_types has no linear_attention entries; \
                             expected hybrid linear/full layout".into(),
                },
                ctx: LoaderCtx { path: cfg_path, tensor: None },
                bt: std::backtrace::Backtrace::capture(),
            });
        }

        let mtp_present = model_dir.join("mtp.safetensors").exists();
        let image_token_id = v["image_token_id"].as_u64().map(|n| n as u32);
        let vision_hidden_size = v["vision_config"]["hidden_size"]
            .as_u64().map(|n| n as usize);
        let vision_depth = v["vision_config"]["depth"]
            .as_u64().map(|n| n as usize);
        let vision_out_hidden_size = v["vision_config"]["out_hidden_size"]
            .as_u64().map(|n| n as usize);

        // Phase 2 prep: RoPE specifics. The config nests under
        // text_config.rope_parameters; partial_rotary_factor lives
        // both there and (legacy) at text_config top-level.
        let rope_partial_rotary_factor = tc["rope_parameters"]["partial_rotary_factor"]
            .as_f64()
            .or_else(|| tc["partial_rotary_factor"].as_f64())
            .unwrap_or(1.0) as f32;
        let rope_theta = tc["rope_parameters"]["rope_theta"]
            .as_f64()
            .or_else(|| tc["rope_theta"].as_f64())
            .unwrap_or(10_000.0) as f32;
        let mrope_section = tc["rope_parameters"]["mrope_section"]
            .as_array()
            .and_then(|arr| {
                if arr.len() == 3 {
                    let a = arr[0].as_u64()? as u32;
                    let b = arr[1].as_u64()? as u32;
                    let c = arr[2].as_u64()? as u32;
                    Some([a, b, c])
                } else { None }
            });

        Ok(Some(Self {
            base,
            attn_output_gate,
            mtp_present,
            n_linear,
            n_full,
            image_token_id,
            vision_hidden_size,
            vision_depth,
            vision_out_hidden_size,
            rope_partial_rotary_factor,
            rope_theta,
            mrope_section,
        }))
    }

    pub fn log_summary(&self) {
        let rotary_dim = (self.base.head_dim as f32
                          * self.rope_partial_rotary_factor) as usize;
        eprintln!(
            "[loader] Qwen 3.5 dense: {} layers ({} linear + {} full), \
             hidden={}, heads={}/{} kvh, hd={}, vocab={}, \
             intermediate={}, attn_output_gate={}, tied_emb={}, \
             mtp={}, image_token_id={:?}, vision={:?}d×{:?}h→{:?}, \
             rope_theta={:.1e}, partial_rotary={} (rotary_dim={}), \
             mrope_section={:?}",
            self.base.num_hidden_layers,
            self.n_linear,
            self.n_full,
            self.base.hidden_size,
            self.base.num_attention_heads,
            self.base.num_key_value_heads,
            self.base.head_dim,
            self.base.vocab_size,
            self.base.intermediate_size,
            self.attn_output_gate,
            self.base.tie_word_embeddings,
            self.mtp_present,
            self.image_token_id,
            self.vision_depth,
            self.vision_hidden_size,
            self.vision_out_hidden_size,
            self.rope_theta,
            self.rope_partial_rotary_factor,
            rotary_dim,
            self.mrope_section,
        );
    }
}
