//! Gemma 4 NVFP4 weight inventory (Phase 1 of the nvidia/Gemma-4-31B-IT-NVFP4
//! bring-up). Metadata-only — no device upload here.
//!
//! Quantization scope per `hf_quant_config.json` (modelopt 0.37.0):
//! * MLP linears (`gate_proj`, `up_proj`, `down_proj`) → NVFP4
//!   packed 4-bit weights + per-group-of-16 E4M3 microscales +
//!   f32 global weight scale + static (PTQ) f32 input scale.
//! * Every layer's `self_attn.*` is in `exclude_modules` →
//!   q/k/v/o_proj stay `bf16`.
//! * `lm_head`, `model.embed_vision*`, `model.vision_tower*`
//!   stay `f16/bf16`.
//! * Checkpoint KV scheme is FP8; rvllm ignores those scales and
//!   uses its own NVFP4 KV recipe (orthogonal — attention weights
//!   are bf16 either way).
//!
//! Tensor naming (verified against the downloaded checkpoint):
//!   `model.language_model.layers.{L}.mlp.{gate_proj|up_proj|down_proj}.weight`
//!   `…weight_scale`, `…weight_scale_2`, `…input_scale`.
//!
//! Shapes (layer 0, 31B):
//!   gate_proj.weight  = [21504, 2688]  u8     (out=21504, in=5376, packed K/2)
//!   gate_proj.weight_scale = [21504, 336]   fp8_e4m3 (336 = 5376/16)
//!   gate_proj.weight_scale_2 = [] f32  (global)
//!   gate_proj.input_scale    = [] f32  (static PTQ)
//!   down_proj.weight  = [5376, 10752]  u8     (out=5376, in=21504, packed K/2)
//!   down_proj.weight_scale = [5376, 1344]   fp8_e4m3 (1344 = 21504/16)
//!
//! This module mirrors the Mistral 3.5 NVFP4 inventory pass
//! (`mistral35_weights::validate_mistral35_inventory`) but:
//! * Gemma uses `weight` / `weight_scale_2` (not `weight_packed` /
//!   `weight_global_scale`) and adds a per-tensor `input_scale`
//!   (Mistral does not).
//! * Only the 3 MLP linears per layer are NVFP4; the 4 attention
//!   linears are bf16 sentinels (verified, not loaded as NVFP4).

use std::collections::BTreeMap;

use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};

use crate::gemma4_arch::Gemma4Arch;
use crate::mistral35_weights::Nvfp4LinearShape;
use crate::safetensors::TensorEntry;

/// One NVFP4 MLP linear in a single Gemma 4 layer. Metadata only —
/// the device pointers land in a separate device-resident struct
/// after upload (Phase 2).
#[derive(Clone, Debug)]
pub struct Gemma4Nvfp4LinearWeight {
    pub shape: Nvfp4LinearShape,
    /// `…weight`           — `U8`,        `[N, K/2]`.
    pub weight: TensorEntry,
    /// `…weight_scale`     — `Fp8E4M3`,   `[N, K/16]`.
    pub weight_scale: TensorEntry,
    /// `…weight_scale_2`   — `F32`,       scalar.
    pub weight_scale_2: TensorEntry,
    /// `…input_scale`      — `F32`,       scalar. Static PTQ scale
    /// (`input_activations.dynamic = false` in the quant config),
    /// applied at the GEMM call. Different from the Mistral path
    /// which derives input scales dynamically per token.
    pub input_scale: TensorEntry,
}

/// The 3 MLP projections per Gemma 4 layer that ship as NVFP4.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Gemma4Nvfp4MlpKind {
    GateProj,
    UpProj,
    DownProj,
}

impl Gemma4Nvfp4MlpKind {
    pub const ALL: [Self; 3] = [Self::GateProj, Self::UpProj, Self::DownProj];

    pub fn name(self) -> &'static str {
        match self {
            Self::GateProj => "mlp.gate_proj",
            Self::UpProj => "mlp.up_proj",
            Self::DownProj => "mlp.down_proj",
        }
    }

    /// `(N, K)` from the architecture. Gate/up are `[intermediate,
    /// hidden]`; down is `[hidden, intermediate]`. K must be
    /// divisible by 16 (group size) — Gemma 4 31B's intermediate
    /// 21504 and hidden 5376 both satisfy this.
    pub fn shape_for(self, arch: &Gemma4Arch) -> Nvfp4LinearShape {
        let h = arch.hidden_size;
        let i = arch.intermediate_size;
        let (n, k) = match self {
            Self::GateProj | Self::UpProj => (i, h),
            Self::DownProj => (h, i),
        };
        Nvfp4LinearShape { n, k }
    }
}

/// Per-tensor counts kept as a cheap loader smoke check. Mirrors
/// the Mistral 3.5 inventory's count struct.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Gemma4Nvfp4TensorCounts {
    /// Expected: `num_hidden_layers * 3` (one per MLP linear).
    pub weight: usize,
    pub weight_scale: usize,
    pub weight_scale_2: usize,
    pub input_scale: usize,
    /// Attention linears verified to remain `bf16` (per layer × 4
    /// projections for sliding; global skips v_proj). Counted, not
    /// validated against an exact total — the bring-up's
    /// architecture-aware shape check is the canonical gate.
    pub attn_bf16: usize,
}

#[derive(Clone, Debug)]
pub struct Gemma4Nvfp4Inventory {
    pub num_layers: usize,
    /// `[layer_idx][mlp_kind_idx]` in `Gemma4Nvfp4MlpKind::ALL`
    /// order (gate, up, down).
    pub layers: Vec<[Gemma4Nvfp4LinearWeight; 3]>,
    pub counts: Gemma4Nvfp4TensorCounts,
    /// `model.language_model` for the standard nvidia checkpoint
    /// layout. Surfaced so callers (esp. the Phase 2 upload pass)
    /// don't re-derive it.
    pub weight_prefix: String,
}

/// Walk the safetensors index and validate the NVFP4 MLP weights
/// plus the bf16 attention sentinels. Pure metadata pass — does
/// not read any bytes.
///
/// `weight_prefix` should be `model.language_model` for the
/// nvidia/Gemma-4-31B-IT-NVFP4 checkpoint. The fp8-block Gemma 4
/// checkpoint uses `model`, so the caller picks the right one
/// based on which checkpoint is being loaded.
pub fn validate_gemma4_nvfp4_inventory(
    arch: &Gemma4Arch,
    weight_prefix: &str,
    tensors: &BTreeMap<String, TensorEntry>,
) -> Result<Gemma4Nvfp4Inventory> {
    let mut layers: Vec<[Option<Gemma4Nvfp4LinearWeight>; 3]> =
        (0..arch.num_hidden_layers).map(|_| Default::default()).collect();
    let mut counts = Gemma4Nvfp4TensorCounts::default();

    for layer_idx in 0..arch.num_hidden_layers {
        for (kind_idx, kind) in Gemma4Nvfp4MlpKind::ALL.iter().enumerate() {
            let shape = kind.shape_for(arch);
            if shape.k % 16 != 0 {
                return Err(corrupt(format!(
                    "Gemma 4 NVFP4 layer {layer_idx} {kind:?}: K={} \
                     not divisible by group_size=16",
                    shape.k
                )));
            }
            let base = format!("{weight_prefix}.layers.{layer_idx}.{}", kind.name());
            let weight = require_tensor(
                tensors,
                &format!("{base}.weight"),
                DType::U8,
                &[shape.n, shape.packed_cols()],
            )?;
            let weight_scale = require_tensor(
                tensors,
                &format!("{base}.weight_scale"),
                DType::Fp8E4M3,
                &[shape.n, shape.scale_cols()],
            )?;
            let weight_scale_2 = require_scalar(
                tensors,
                &format!("{base}.weight_scale_2"),
                DType::F32,
            )?;
            let input_scale = require_scalar(
                tensors,
                &format!("{base}.input_scale"),
                DType::F32,
            )?;
            counts.weight += 1;
            counts.weight_scale += 1;
            counts.weight_scale_2 += 1;
            counts.input_scale += 1;
            layers[layer_idx][kind_idx] = Some(Gemma4Nvfp4LinearWeight {
                shape,
                weight,
                weight_scale,
                weight_scale_2,
                input_scale,
            });
        }
    }

    // Sentinel sweep: attention projections must remain in bf16 (or
    // f16). Don't enforce exact shapes here — the Gemma 4 bring-up's
    // shape check is the canonical gate. We just refuse a checkpoint
    // that accidentally NVFP4-quantized an attention tensor (which
    // would silently change the forward path's expected dtype).
    for (name, entry) in tensors.iter() {
        if !name.starts_with(weight_prefix) {
            continue;
        }
        if !name.contains(".self_attn.") {
            continue;
        }
        if !name.ends_with(".weight") {
            continue;
        }
        if entry.dtype != DType::Bf16 && entry.dtype != DType::F16 {
            return Err(corrupt(format!(
                "Gemma 4 NVFP4 attention tensor {name} is NOT bf16/f16 \
                 (got {:?}); checkpoint quant_config says self_attn must \
                 stay unquantized — refusing to load.",
                entry.dtype
            )));
        }
        counts.attn_bf16 += 1;
    }

    let resolved: Vec<[Gemma4Nvfp4LinearWeight; 3]> = layers
        .into_iter()
        .enumerate()
        .map(|(layer_idx, slots)| -> Result<[Gemma4Nvfp4LinearWeight; 3]> {
            let arr: [Gemma4Nvfp4LinearWeight; 3] = std::array::from_fn(|i| {
                slots[i].clone().unwrap_or_else(|| {
                    panic!("validate_gemma4_nvfp4_inventory: \
                            layer {layer_idx} slot {i} not populated \
                            but require_tensor() would have errored")
                })
            });
            Ok(arr)
        })
        .collect::<Result<_>>()?;

    Ok(Gemma4Nvfp4Inventory {
        num_layers: arch.num_hidden_layers,
        layers: resolved,
        counts,
        weight_prefix: weight_prefix.to_string(),
    })
}

fn require_tensor(
    tensors: &BTreeMap<String, TensorEntry>,
    name: &str,
    expected_dtype: DType,
    expected_shape: &[usize],
) -> Result<TensorEntry> {
    let e = tensors.get(name).cloned().ok_or_else(|| missing(name))?;
    if e.dtype != expected_dtype {
        return Err(corrupt(format!(
            "Gemma 4 NVFP4 tensor {name}: dtype mismatch (got {:?}, expected {:?})",
            e.dtype, expected_dtype
        )));
    }
    if e.shape != expected_shape {
        return Err(corrupt(format!(
            "Gemma 4 NVFP4 tensor {name}: shape mismatch (got {:?}, expected {:?})",
            e.shape, expected_shape
        )));
    }
    Ok(e)
}

fn require_scalar(
    tensors: &BTreeMap<String, TensorEntry>,
    name: &str,
    expected_dtype: DType,
) -> Result<TensorEntry> {
    let e = tensors.get(name).cloned().ok_or_else(|| missing(name))?;
    if e.dtype != expected_dtype {
        return Err(corrupt(format!(
            "Gemma 4 NVFP4 tensor {name}: dtype mismatch (got {:?}, expected {:?})",
            e.dtype, expected_dtype
        )));
    }
    // modelopt emits scalars as `[]` (rank 0); historical safetensors
    // dumpers sometimes promote to `[1]`. Accept either.
    if !(e.shape.is_empty() || e.shape == [1]) {
        return Err(corrupt(format!(
            "Gemma 4 NVFP4 tensor {name}: expected scalar shape ([] or [1]), got {:?}",
            e.shape
        )));
    }
    Ok(e)
}

// ─── Device-resident loaded model (post-upload) ────────────────────
//
// Populated by the rvllm-runtime side `gemma4_nvfp4_load`. Structs
// live here so the loader can construct the natural-layout
// intermediates without crossing the DAG line into rvllm-cutlass.

/// One per-layer NVFP4 MLP linear after weight upload. All pointers
/// are absolute device addresses; the bring-up's arena owns the
/// backing storage.
///
/// Differences from `mistral35_weights::Nvfp4LinearLoaded`:
/// * Adds `input_scale_ptr` — Gemma 4 NVFP4 ships a static (PTQ)
///   input scale per linear (`input_activations.dynamic = false`
///   in the quant config). Mistral derives input scales dynamically
///   per-token and has no equivalent tensor.
/// * Uses `weight_scale_2` instead of `weight_global_scale` on disk
///   (semantically identical — a single f32 scalar — stored here as
///   the post-decode-scale device scalar `global_scale_ptr`).
#[derive(Debug, Clone, Copy)]
pub struct Gemma4Nvfp4LinearLoaded {
    pub shape: Nvfp4LinearShape,
    /// `[N, K/2]` U8 NVFP4-packed weight bytes.
    pub packed_ptr: u64,
    /// `[N, K/16]` E4M3 weight scale, row-major natural layout.
    /// Used by the W4A16 dequant-then-bf16-GEMM path.
    pub sfb_natural_ptr: u64,
    /// CUTLASS-interleaved E4M3 SFB scratch — `0` unless the legacy
    /// W4A4 tensor-core GEMM path is wired in. Mirrors the Mistral
    /// `RVLLM_W4A16_GEMV=0` opt-in: the default fused W4A16 GEMV
    /// reads `sfb_natural_ptr` and never touches this pointer.
    pub sfb_cutlass_ptr: u64,
    /// `[1]` F32 device scalar — `1 / weight_scale_2` (decode form),
    /// suitable for CUTLASS's `alpha_ptr` epilogue without a host
    /// stall. Matches Mistral's `global_scale_ptr` semantics; the
    /// rename in source carries the Gemma checkpoint's
    /// `weight_scale_2` tensor name.
    pub global_scale_ptr: u64,
    /// `[1]` F32 device scalar — static (PTQ) input scale from
    /// `input_scale` tensor. Forwarded to the GEMM's input-side
    /// quantization step.
    pub input_scale_ptr: u64,
    pub packed_bytes: usize,
    pub sfb_bytes: usize,
    /// `[N, K]` BF16 dequantized weight. Zero until a runtime-side
    /// one-shot dequant pass populates it. Mirrors Mistral's
    /// `bf16_ptr` lazy-fill semantics.
    pub bf16_ptr: u64,
}

/// One Gemma 4 layer's NVFP4 MLP linears + retained-bf16 attention
/// weights. Attention projection pointers stay as `F16Weight`
/// because `hf_quant_config.json` excludes every layer's
/// `self_attn.*` from quantization.
///
/// Populated by the runtime-side upload; consumed by the per-layer
/// forward in the bring-up.
#[derive(Debug)]
pub struct Gemma4Nvfp4LayerLoaded {
    pub input_layernorm: crate::weights::F16Weight,
    pub post_attention_layernorm: crate::weights::F16Weight,
    pub pre_feedforward_layernorm: crate::weights::F16Weight,
    pub post_feedforward_layernorm: crate::weights::F16Weight,
    /// Per-layer residual multiplier `[1]` f32 (Gemma 4 `layer_scalar`).
    pub layer_scalar: crate::weights::F16Weight,
    /// Q/K head-dim RMSNorm gammas (`q_norm`, `k_norm`), per Gemma 4
    /// architecture. Stay bf16.
    pub q_norm: crate::weights::F16Weight,
    pub k_norm: crate::weights::F16Weight,
    /// Attention projections — bf16 per `exclude_modules`.
    pub q_proj: crate::weights::F16Weight,
    pub k_proj: crate::weights::F16Weight,
    /// `None` on layers where `attention_k_eq_v=true` (V is aliased
    /// to K and never stored on disk).
    pub v_proj: Option<crate::weights::F16Weight>,
    pub o_proj: crate::weights::F16Weight,
    /// MLP — NVFP4.
    pub gate_proj: Gemma4Nvfp4LinearLoaded,
    pub up_proj: Gemma4Nvfp4LinearLoaded,
    pub down_proj: Gemma4Nvfp4LinearLoaded,
}

impl Gemma4Nvfp4LayerLoaded {
    /// Iterate the per-layer NVFP4 linears in
    /// `Gemma4Nvfp4MlpKind::ALL` order: gate, up, down.
    pub fn nvfp4_linears(&self) -> [&Gemma4Nvfp4LinearLoaded; 3] {
        [&self.gate_proj, &self.up_proj, &self.down_proj]
    }
}

/// Outside-the-stack text weights for the Gemma 4 NVFP4
/// checkpoint. The model has `tie_word_embeddings=true`, so
/// `lm_head` aliases `embed_tokens.T` at forward time — no
/// separate tensor on disk.
#[derive(Debug)]
pub struct Gemma4Nvfp4OutsideText {
    /// `[vocab=262144, hidden=5376]` bf16. Doubles as the
    /// LM-head weight via the tied-embedding convention.
    pub embed_tokens: crate::weights::F16Weight,
    /// `[hidden=5376]` bf16. Final RMSNorm before lm_head.
    pub final_norm: crate::weights::F16Weight,
}

/// Top-level Gemma 4 NVFP4 model after upload. Vision tower
/// not loaded by Phase 3b — that's Phase 3c (vision splice
/// path is orthogonal and can be wired in after the text
/// forward is green).
#[derive(Debug)]
pub struct Gemma4Nvfp4LoadedModel {
    pub outside: Gemma4Nvfp4OutsideText,
    pub layers: Vec<Gemma4Nvfp4LayerLoaded>,
}

fn missing(name: &str) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::MissingTensor { name: name.to_string() },
        ctx: LoaderCtx {
            path: std::path::PathBuf::from("(gemma4 nvfp4 inventory)"),
            tensor: Some(name.to_string()),
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

fn corrupt(detail: String) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt { detail },
        ctx: LoaderCtx {
            path: std::path::PathBuf::from("(gemma4 nvfp4 inventory)"),
            tensor: None,
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemma4_arch::Gemma4LayerType;

    fn arch_fixture(num_layers: usize) -> Gemma4Arch {
        // Minimal 31B-shaped arch for shape/inventory tests. Only the
        // fields the inventory reads are populated; the rest carry
        // defaults that are irrelevant to validate_*.
        let mut layer_types = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layer_types.push(if (i + 1) % 6 == 0 {
                Gemma4LayerType::GlobalAttention
            } else {
                Gemma4LayerType::SlidingAttention
            });
        }
        Gemma4Arch {
            num_hidden_layers: num_layers,
            hidden_size: 5376,
            num_attention_heads: 32,
            head_dim_sliding: 256,
            head_dim_global: 512,
            num_kv_heads_sliding: 16,
            num_kv_heads_global: 4,
            intermediate_size: 21504,
            vocab_size: 262144,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 8192,
            sliding_window_size: 512,
            rope_theta_sliding: 10_000.0,
            rope_theta_global: 1_000_000.0,
            partial_rotary_factor_global: 1.0,
            logit_softcap: 0.0,
            layer_types,
            weight_prefix: "model.language_model".into(),
            tie_word_embeddings: true,
            num_kv_shared_layers: None,
            vision_config: None,
            audio_config: None,
            hidden_size_per_layer_input: None,
            per_layer_model_projection_scale: 0.0,
            per_layer_input_scale: 0.0,
        }
    }

    fn fake_entry(name: &str, dtype: DType, shape: &[usize]) -> TensorEntry {
        TensorEntry {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            file_offset: 0,
            nbytes: 0,
        }
    }

    fn populate_mlp_layers(tensors: &mut BTreeMap<String, TensorEntry>, arch: &Gemma4Arch) {
        let prefix = &arch.weight_prefix;
        for layer_idx in 0..arch.num_hidden_layers {
            for kind in Gemma4Nvfp4MlpKind::ALL {
                let shape = kind.shape_for(arch);
                let base = format!("{prefix}.layers.{layer_idx}.{}", kind.name());
                tensors.insert(
                    format!("{base}.weight"),
                    fake_entry(&format!("{base}.weight"), DType::U8,
                               &[shape.n, shape.packed_cols()]),
                );
                tensors.insert(
                    format!("{base}.weight_scale"),
                    fake_entry(&format!("{base}.weight_scale"), DType::Fp8E4M3,
                               &[shape.n, shape.scale_cols()]),
                );
                tensors.insert(
                    format!("{base}.weight_scale_2"),
                    fake_entry(&format!("{base}.weight_scale_2"), DType::F32, &[]),
                );
                tensors.insert(
                    format!("{base}.input_scale"),
                    fake_entry(&format!("{base}.input_scale"), DType::F32, &[]),
                );
            }
        }
    }

    fn populate_attn_bf16(tensors: &mut BTreeMap<String, TensorEntry>, arch: &Gemma4Arch) {
        let prefix = &arch.weight_prefix;
        for layer_idx in 0..arch.num_hidden_layers {
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                let key = format!("{prefix}.layers.{layer_idx}.self_attn.{name}.weight");
                tensors.insert(key.clone(), fake_entry(&key, DType::Bf16, &[1, 1]));
            }
        }
    }

    #[test]
    fn shape_helpers_match_31b_spec() {
        let arch = arch_fixture(60);
        let g = Gemma4Nvfp4MlpKind::GateProj.shape_for(&arch);
        assert_eq!(g, Nvfp4LinearShape { n: 21504, k: 5376 });
        assert_eq!(g.packed_cols(), 2688);  // matches downloaded checkpoint
        assert_eq!(g.scale_cols(), 336);    // matches downloaded checkpoint
        let d = Gemma4Nvfp4MlpKind::DownProj.shape_for(&arch);
        assert_eq!(d, Nvfp4LinearShape { n: 5376, k: 21504 });
        assert_eq!(d.packed_cols(), 10752); // matches downloaded checkpoint
        assert_eq!(d.scale_cols(), 1344);   // matches downloaded checkpoint
    }

    #[test]
    fn full_60_layer_inventory_validates() {
        let arch = arch_fixture(60);
        let mut tensors = BTreeMap::new();
        populate_mlp_layers(&mut tensors, &arch);
        populate_attn_bf16(&mut tensors, &arch);
        let inv = validate_gemma4_nvfp4_inventory(
            &arch, &arch.weight_prefix.clone(), &tensors,
        ).expect("valid");
        assert_eq!(inv.num_layers, 60);
        assert_eq!(inv.layers.len(), 60);
        assert_eq!(inv.counts.weight, 60 * 3);
        assert_eq!(inv.counts.weight_scale, 60 * 3);
        assert_eq!(inv.counts.weight_scale_2, 60 * 3);
        assert_eq!(inv.counts.input_scale, 60 * 3);
        assert_eq!(inv.counts.attn_bf16, 60 * 4);
    }

    #[test]
    fn missing_weight_tensor_reports_named_error() {
        let arch = arch_fixture(2);
        let mut tensors = BTreeMap::new();
        populate_mlp_layers(&mut tensors, &arch);
        let key = format!(
            "{}.layers.1.{}.weight",
            arch.weight_prefix,
            Gemma4Nvfp4MlpKind::GateProj.name()
        );
        tensors.remove(&key);
        let err = validate_gemma4_nvfp4_inventory(
            &arch, &arch.weight_prefix.clone(), &tensors,
        ).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("weight"));
        assert!(msg.contains("layers.1"));
    }

    #[test]
    fn quantized_attention_is_rejected() {
        let arch = arch_fixture(1);
        let mut tensors = BTreeMap::new();
        populate_mlp_layers(&mut tensors, &arch);
        // Plant a U8 attention tensor — would be silent silently-wrong.
        let key = format!("{}.layers.0.self_attn.q_proj.weight", arch.weight_prefix);
        tensors.insert(key, fake_entry("q_proj.weight", DType::U8, &[1, 1]));
        let err = validate_gemma4_nvfp4_inventory(
            &arch, &arch.weight_prefix.clone(), &tensors,
        ).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("self_attn"));
        assert!(msg.contains("must stay unquantized"));
    }

    /// Integration test — only runs when GEMMA4_NVFP4_DIR is set
    /// to a directory containing the nvidia/Gemma-4-31B-IT-NVFP4
    /// checkpoint (config.json + model.safetensors.index.json +
    /// shards). Parses the index headers (no payload reads) and
    /// runs the full inventory validator.
    ///
    /// Run:  GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///       cargo test -p rvllm-loader gemma4_nvfp4_weights::tests::ondisk -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_nvfp4_checkpoint_inventory_passes() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => std::path::PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping on-disk inventory test");
                return;
            }
        };
        let arch = crate::gemma4_arch::Gemma4Arch::from_dir(&dir)
            .expect("Gemma4Arch::from_dir");
        eprintln!(
            "[ondisk] arch: layers={} hidden={} intermediate={} prefix={}",
            arch.num_hidden_layers, arch.hidden_size, arch.intermediate_size,
            arch.weight_prefix,
        );

        // Mirror mistral35_bring_up::scan_safetensors_index — mmap
        // each shard header to collect a tensor map.
        use crate::safetensors::{ShardHeader, ShardIndex};
        let idx = ShardIndex::resolve(&dir).expect("ShardIndex::resolve");
        let mut tensors: BTreeMap<String, TensorEntry> = BTreeMap::new();
        for shard in &idx.shards {
            let f = std::fs::File::open(shard).expect("open shard");
            let mmap = unsafe { memmap2::Mmap::map(&f) }.expect("mmap shard");
            let header = ShardHeader::parse(shard, &mmap).expect("ShardHeader::parse");
            for (name, entry) in header.tensors.into_iter() {
                tensors.insert(name, entry);
            }
        }
        eprintln!("[ondisk] tensors total: {}", tensors.len());

        let prefix = arch.weight_prefix.clone();
        let inv = validate_gemma4_nvfp4_inventory(&arch, &prefix, &tensors)
            .expect("inventory");
        eprintln!(
            "[ondisk] counts: weight={} weight_scale={} weight_scale_2={} \
             input_scale={} attn_bf16={}",
            inv.counts.weight, inv.counts.weight_scale,
            inv.counts.weight_scale_2, inv.counts.input_scale,
            inv.counts.attn_bf16,
        );
        assert_eq!(inv.counts.weight,         arch.num_hidden_layers * 3);
        assert_eq!(inv.counts.weight_scale,   arch.num_hidden_layers * 3);
        assert_eq!(inv.counts.weight_scale_2, arch.num_hidden_layers * 3);
        assert_eq!(inv.counts.input_scale,    arch.num_hidden_layers * 3);
        // 60 layers × 4 attn projs = 240 on 31B; the global v_proj
        // entries are absent for k_eq_v layers, so the actual count
        // is slightly lower. Verify at least one bf16 attention
        // tensor was seen, not the exact total.
        assert!(inv.counts.attn_bf16 > 0,
                "no bf16 attention tensors found — quant_config violation suspected");
    }

    #[test]
    fn scalar_shape_accepts_empty_or_one() {
        let arch = arch_fixture(1);
        let mut tensors = BTreeMap::new();
        populate_mlp_layers(&mut tensors, &arch);
        // Swap one weight_scale_2 to shape [1] — must still validate.
        let key = format!(
            "{}.layers.0.{}.weight_scale_2",
            arch.weight_prefix,
            Gemma4Nvfp4MlpKind::GateProj.name()
        );
        tensors.insert(key.clone(), fake_entry(&key, DType::F32, &[1]));
        populate_attn_bf16(&mut tensors, &arch);
        validate_gemma4_nvfp4_inventory(
            &arch, &arch.weight_prefix.clone(), &tensors,
        ).expect("scalar [1] accepted same as []");
    }
}
