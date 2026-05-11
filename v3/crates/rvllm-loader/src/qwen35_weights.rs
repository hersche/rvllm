//! Qwen 3.5 27B dense weight structures.
//!
//! Shares almost everything with Qwen 3.6 35B-A3B's `qwen36_weights`
//! module — same hybrid linear/full attention, same `attn_output_gate`
//! finisher, same Qwen3-VL vision tower geometry — but the MoE block
//! (`Qwen36MoeBlock` with 256 experts + shared expert + router) is
//! replaced by a flat dense MLP (`Qwen35DenseMlpBlock` with gate/up/
//! down, FP8 e4m3 with BF16 block-128 `weight_scale_inv` scales).
//!
//! See `v3/QWEN35_BRINGUP_PLAN.md` for the multi-week phased plan.
//! The structs below mirror Phase 2's `qwen36_weights` shape so the
//! loader + forward path can share helpers.
//!
//! Storage convention (matches Qwen 3.6):
//!   * BF16 → uploaded as F16Weight (existing arena infra is f16
//!     native; `tensor_to_f16_bytes` in `load.rs` performs the cast).
//!   * FP8 e4m3 + per-block (128×128) BF16 `weight_scale_inv` table
//!     → uploaded as Fp8Weight, scale_inv kept verbatim.

use crate::qwen36_weights::Qwen36Vision;
use crate::weights::{F16Weight, Fp8Weight};

#[derive(Debug)]
pub struct Qwen35LoadedOutside {
    pub embed_tokens: F16Weight,
    pub final_norm: F16Weight,
    pub lm_head: F16Weight,
    /// CPU-quantized FP8 mirror of lm_head for the `fp8_gemv` path
    /// (matches the Qwen 3.6 Phase 3d pattern).
    pub lm_head_fp8: Fp8Weight,
    pub embed_tokens_bytes: u64,
    pub final_norm_bytes: u64,
    pub lm_head_bytes: u64,
    pub lm_head_fp8_bytes: u64,
}

/// Full-attention layer weights (identical shape to
/// `Qwen36FullAttnLayer`). 16 of the 64 Qwen 3.5 27B layers (every
/// 4th index starting at 3) carry standard self-attention with
/// per-head Q/K RMSNorm and the attn_output_gate finisher: `q_proj`
/// output width is 2 × num_q_heads × head_dim = 12288, the second
/// half is the output gate applied as `sigmoid(gate) * attn` before
/// `o_proj`.
#[derive(Debug)]
pub struct Qwen35FullAttnLayer {
    pub input_layernorm: F16Weight,
    pub post_attention_layernorm: F16Weight,
    pub q_norm: F16Weight,
    pub k_norm: F16Weight,
    pub q_proj: Fp8Weight,
    pub k_proj: Fp8Weight,
    pub v_proj: Fp8Weight,
    pub o_proj: Fp8Weight,
}

/// Linear-attention (Gated DeltaNet) layer weights — identical
/// fields to `Qwen36LinearAttnLayer`. 48 of the 64 Qwen 3.5 27B
/// layers are linear-attention.
#[derive(Debug)]
pub struct Qwen35LinearAttnLayer {
    pub input_layernorm: F16Weight,
    pub post_attention_layernorm: F16Weight,
    pub a_log: F16Weight,
    pub dt_bias: F16Weight,
    pub conv1d: F16Weight,
    pub in_proj_a: F16Weight,
    pub in_proj_b: F16Weight,
    pub in_proj_qkv: Fp8Weight,
    pub in_proj_z: Fp8Weight,
    pub norm: F16Weight,
    pub out_proj: Fp8Weight,
}

/// Dense MLP block — the key Qwen 3.5 vs 3.6 difference. All 64
/// Qwen 3.5 layers carry this (both linear and full attention
/// variants). Same shape as Mistral 3.5 / Gemma 4 dense MLP:
///   gate_proj : Fp8 [intermediate, hidden] = [17408, 5120]
///   up_proj   : Fp8 [intermediate, hidden] = [17408, 5120]
///   down_proj : Fp8 [hidden, intermediate] = [5120, 17408]
/// SiLU·gate·up → down_proj.
#[derive(Debug)]
pub struct Qwen35DenseMlpBlock {
    pub gate_proj: Fp8Weight,
    pub up_proj: Fp8Weight,
    pub down_proj: Fp8Weight,
}

#[derive(Debug)]
pub enum Qwen35LayerAttn {
    Linear(Qwen35LinearAttnLayer),
    Full(Qwen35FullAttnLayer),
}

#[derive(Debug)]
pub struct Qwen35Layer {
    pub attn: Qwen35LayerAttn,
    pub mlp: Qwen35DenseMlpBlock,
}

/// Qwen 3.5 loaded model. Phase 1 lands `outside` only; Phase 2+
/// fills `layers` and `vision`.
#[derive(Debug)]
pub struct Qwen35LoadedModel {
    pub outside: Qwen35LoadedOutside,
    /// 64 layer slots, populated in Phase 2. The attn variant
    /// matches `layer_types[idx]`. Empty in Phase 1.
    pub layers: Vec<Qwen35Layer>,
    /// Vision tower (Qwen3-VL ViT, shape-identical to Qwen 3.6's).
    /// Reused via `Qwen36Vision` — same patch_embed, 27 blocks, and
    /// patch merger. `None` for text-only loads.
    pub vision: Option<Qwen36Vision>,
}
