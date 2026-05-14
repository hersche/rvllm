//! Gemma 4 weight structures.
//!
//! Sliding and global layers do NOT share identical attention weight shapes.
//! Sliding layers use `(q, k, v, o) = (8192, 4096, 4096, 8192)` over the
//! head axis, while global layers use `(16384, 2048, no v_proj, 16384)`.
//! Global attention has `attention_k_eq_v=true`, so the K projection is
//! reused for V when building the fused QKV weight.
//!
//! Per-layer extras vs Llama/Qwen:
//!   - 4 norms (input, post_attn, pre_ff, post_ff)
//!   - QK-norm gammas (q_norm [256], k_norm [256])
//!   - layer_scalar [1] (per-layer residual multiplier)
//!
//! Sliding layer shapes:
//!   q_proj:        [8192, 5376]
//!   k_proj:        [4096, 5376]
//!   v_proj:        [4096, 5376]
//!   o_proj:        [5376, 8192]
//!
//! Global layer shapes:
//!   q_proj:        [16384, 5376]
//!   k_proj:        [2048, 5376]
//!   v_proj:        absent, reuse `k_proj`
//!   o_proj:        [5376, 16384]
//!
//! Shared MLP / norm shapes:
//!   gate_proj:     [21504, 5376]
//!   up_proj:       [21504, 5376]
//!   down_proj:     [5376, 21504]
//!   q_norm:        [256]
//!   k_norm:        [256]
//!   layer_scalar:  [1]
//!   *_layernorm:   [5376]

use crate::weights::{AwqLayerWeights, F16Weight, Fp8Weight};

#[derive(Debug)]
pub struct Gemma4LayerWeights {
    /// FP8 attention QKV weights. `None` when this layer is AWQ-quantized
    /// (the `awq` field below carries Q/K/V instead). At least one of
    /// `qkv` / `qkv_f16` / `awq.q_proj` must be `Some` for the layer
    /// to execute; bring-up reflects that as a non-zero pointer in
    /// the corresponding `Gemma4LayerWeightPtrs` slot.
    ///
    /// Cycle 48 step 7a: made Optional to support AWQ-only layers
    /// without forcing dummy FP8 weights to occupy ~half the GPU
    /// arena.
    pub qkv: Option<Fp8Weight>,
    pub o_proj: Option<Fp8Weight>,
    pub gate_up: Option<Fp8Weight>,
    pub down_proj: Option<Fp8Weight>,
    pub qkv_f16: Option<F16Weight>,
    pub o_proj_f16: Option<F16Weight>,
    pub gate_up_f16: Option<F16Weight>,
    pub down_proj_f16: Option<F16Weight>,
    pub input_layernorm: F16Weight,
    pub post_attention_layernorm: F16Weight,
    pub pre_feedforward_layernorm: F16Weight,
    pub post_feedforward_layernorm: F16Weight,
    pub q_norm: F16Weight,
    pub k_norm: F16Weight,
    pub layer_scalar: F16Weight,
    /// E4B Per-Layer Embeddings (PLE). `None` on 31B; `Some` on E4B.
    /// Used by the runtime to compute the additional residual
    /// contribution at the end of each layer's forward (HF
    /// `Gemma4TextDecoderLayer.forward`).
    pub per_layer_input_gate: Option<F16Weight>, // [ple_dim, hidden]
    pub per_layer_projection: Option<F16Weight>, // [hidden, ple_dim]
    pub post_per_layer_input_norm: Option<F16Weight>, // [hidden]
    /// Cycle 46 step 5c: optional AWQ INT4 W4A16 weights for this layer.
    /// `None` = FP8 path stays in charge for every linear in the layer
    /// (no behavior change for non-AWQ checkpoints). `Some` = the seven
    /// projections (q/k/v/o + gate/up/down) have AWQ tensors uploaded
    /// via `compressed_tensors::upload_gemma4_awq_layer`; bring-up
    /// reads this and populates `Gemma4AwqLayerPtrs` so exec_layer
    /// dispatches through `awq_int4_gemv_f16_kernel`.
    ///
    /// Populated by an AWQ-aware load path that is wired in cycle 47;
    /// the field is added here so the runtime crate can already
    /// thread it through bring-up without a follow-up data-shape
    /// change.
    pub awq: Option<AwqLayerWeights>,
}

#[derive(Debug)]
pub struct Gemma4LoadedModel {
    pub embedding: F16Weight,
    pub lm_head_fp8: Fp8Weight,
    pub lm_head_f16: F16Weight,
    pub final_norm: F16Weight,
    /// Sliding layers: theta=10000, full rotation (rotary_dim=256)
    pub rope_cos_sliding: F16Weight,
    pub rope_sin_sliding: F16Weight,
    /// Global layers: theta=1M, partial rotation (rotary_dim=128 of head_dim=512)
    pub rope_cos_global: F16Weight,
    pub rope_sin_global: F16Weight,
    pub layers: Vec<Gemma4LayerWeights>,
    /// Vision tower (SigLIP-style ViT) for Gemma 4 multimodal.
    /// `None` for text-only checkpoints.
    pub vision: Option<Gemma4Vision>,
    /// E4B audio tower (12-layer encoder + 2-stage Conv2d subsampling
    /// + output projection). `None` on 31B and any text-only checkpoint
    /// that lacks the `model.audio_tower.*` tensors.
    pub audio: Option<Gemma4Audio>,
    /// E4B Per-Layer Embeddings (PLE) global side. `None` on 31B;
    /// `Some` on E4B-it. Combined with the per-layer triple on each
    /// `Gemma4LayerWeights` to inject an additive residual at the
    /// end of every layer's forward.
    pub ple: Option<Gemma4Ple>,
    /// Echoed from `Gemma4Arch::num_kv_shared_layers`. `None` on 31B;
    /// `Some(18)` on E4B-it. Bring-up reads this to alias the trailing
    /// sliding layers' K/V onto the most recent full-attention layer's
    /// KV slot (A4 lands the physical aliasing — A2 only plumbs).
    pub num_kv_shared_layers: Option<u32>,
}

// ─── Per-Layer Embeddings (E4B-it) ────────────────────────────────────
//
// Gemma 4 E4B introduces a secondary signal pathway that
// rvllm needs to honour for coherent output. The math is documented
// upstream in HF `Gemma4TextModel.get_per_layer_inputs` +
// `project_per_layer_inputs` + the tail of
// `Gemma4TextDecoderLayer.forward`. Geometry on E4B:
//
//   ple_dim = 256
//   num_hidden_layers = 42
//   hidden_size = 2560
//
//   embed_tokens_per_layer       [vocab, num_layers * ple_dim]
//                                = [262144, 10752]
//   per_layer_model_projection   [num_layers * ple_dim, hidden_size]
//                                = [10752, 2560]
//   per_layer_projection_norm    [num_layers * ple_dim] = [10752]
//
//   (per layer) per_layer_input_gate.weight   [ple_dim, hidden]
//   (per layer) per_layer_projection.weight   [hidden,  ple_dim]
//   (per layer) post_per_layer_input_norm     [hidden]
//
// Scales:
//   embed_tokens_per_layer is a ScaledWordEmbedding → output is
//   `lookup × sqrt(ple_dim)`. Bake at upload time (same pattern as
//   the main embed_tokens.weight pre-scale).
//   per_layer_model_projection_scale defaults to `1/sqrt(hidden_size)`.
//   per_layer_input_scale defaults to `1/sqrt(2)`.
//
#[derive(Debug)]
pub struct Gemma4Ple {
    /// `[vocab, num_layers * ple_dim]`; sqrt(ple_dim) pre-scaled at
    /// upload time so the runtime lookup is a plain gather.
    pub embed_tokens_per_layer: F16Weight,
    /// `[num_layers * ple_dim, hidden_size]`. The runtime multiplies
    /// the linear output by `arch.per_layer_model_projection_scale`.
    pub per_layer_model_projection: F16Weight,
    /// `[ple_dim]` — SHARED across all `num_hidden_layers` layers.
    /// HF instantiates as `Gemma4RMSNorm(hidden_size_per_layer_input)`
    /// then applies after reshape to `[T, num_layers, ple_dim]`,
    /// broadcasting over the layer axis. One rmsnorm launch on a
    /// `[T * num_layers, ple_dim]` view with this γ vector is the
    /// correct precompute primitive — not 42 separate sliced
    /// launches.
    pub per_layer_projection_norm: F16Weight,
}

// ─── Vision (Gemma 4 SigLIP-style ViT) ────────────────────────────────

/// Geometry (verified via state-dict):
/// - patch_embedder.input_proj.weight: [1152, 768] (16² * 3 → hidden=1152)
/// - patch_embedder.position_embedding_table: [2, 10240, 1152] f16
///   (axis 0 indexes row vs col, axis 1 picks position 0..10239)
/// - 27 encoder layers, each carries:
///     input_layernorm.weight (RMSNorm, Gemma-style +1 shift)
///     post_attention_layernorm.weight
///     pre_feedforward_layernorm.weight
///     post_feedforward_layernorm.weight
///     self_attn.q_proj.linear.weight [1152, 1152] (NO bias; attention_bias=False)
///     self_attn.k_proj.linear.weight [1152, 1152]
///     self_attn.v_proj.linear.weight [1152, 1152]
///     self_attn.o_proj.linear.weight [1152, 1152]
///     self_attn.q_norm.weight [72] (per-head RMSNorm, Gemma-style +1)
///     self_attn.k_norm.weight [72]
///     mlp.gate_proj.linear.weight [4304, 1152]
///     mlp.up_proj.linear.weight   [4304, 1152]
///     mlp.down_proj.linear.weight [1152, 4304]
/// - std_bias, std_scale: [1152] applied after the encoder output,
///   before the embed_vision projection (Gemma vision_config
///   `standardize=True`).
/// - embed_vision.embedding_projection.weight: [5376, 1152]
///   post-pool linear from vision-hidden to text-hidden.
#[derive(Debug)]
pub struct Gemma4Vision {
    pub patch_embedder_input_proj: F16Weight, // [1152, 768]
    pub patch_embedder_pos_table: F16Weight,  // [2, 10240, 1152]
    pub blocks: Vec<Gemma4VisionBlock>,
    /// `[hidden]`. Present iff `vision_config.standardize=true`
    /// (31B); `None` on E4B-it which sets standardize=false and
    /// omits these tensors entirely.
    pub std_bias: Option<F16Weight>,
    pub std_scale: Option<F16Weight>,
    pub embed_vision_projection: F16Weight,    // [out_hidden, hidden]
}

// ─── E4B audio tower (B5) ────────────────────────────────────────────
//
// Geometry per `/home/r00t/gemma4-e4b/config.json::audio_config`:
//   num_hidden_layers           = 12
//   hidden_size                 = 1024  (audio-tower internal hidden;
//                                        NOT text hidden=2560)
//   num_attention_heads         = 8  → head_dim = 128
//   conv_kernel_size            = 5  (depthwise causal Conv1d inside
//                                     each layer's lconv1d block)
//   subsampling_conv_channels   = [128, 32] (two-stage Conv2d
//                                            kernel=(3,3) stride=(2,2))
//   output_proj_dims            = 1536 (audio-tower output width;
//                                       projected to text hidden by
//                                       embed_audio.embedding_projection)
//   use_clipped_linears         = true  (every Linear ships
//                                        input_max/input_min/output_max
//                                        /output_min scalar clip stats)
//
// Tensor layout on disk (`model.audio_tower.*`):
//   subsample_conv_projection.input_proj_linear.weight    [1024, 1024]
//   subsample_conv_projection.layer{0,1}.conv.weight      [128 or 32,
//                                                          1 or 128, 3, 3]
//   subsample_conv_projection.layer{0,1}.norm.weight      [128 or 32]
//   output_proj.weight                                    [1536, 1024]
//   output_proj.bias                                      [1536]
//
//   layers.{L}.norm_pre_attn.weight                       [1024]
//   layers.{L}.norm_post_attn.weight                      [1024]
//   layers.{L}.norm_out.weight                            [1024]
//
//   layers.{L}.self_attn.{q,k,v}_proj.{linear.weight,
//                                      input_max, input_min,
//                                      output_max, output_min}
//   layers.{L}.self_attn.post.* (same clipped-linear shape)
//   layers.{L}.self_attn.relative_k_proj.weight           [1024, 1024]
//   layers.{L}.self_attn.per_dim_scale                    [128]
//
//   layers.{L}.feed_forward{1,2}.pre_layer_norm.weight    [1024]
//   layers.{L}.feed_forward{1,2}.post_layer_norm.weight   [1024]
//   layers.{L}.feed_forward{1,2}.ffw_layer_{1,2}.*
//                                                          (clipped linear)
//
//   layers.{L}.lconv1d.pre_layer_norm.weight              [1024]
//   layers.{L}.lconv1d.conv_norm.weight                   [1024]
//   layers.{L}.lconv1d.depthwise_conv1d.weight            [1024, 1, 5]
//   layers.{L}.lconv1d.linear_{start,end}.*               (clipped linear)
//
//   embed_audio.embedding_projection.weight               [2560, 1536]
//
// All on-disk dtypes are bf16 → f16 at upload (same F16_ONLY path
// the text decoder uses).

/// A Linear weight bundled with its 4 scalar clip statistics
/// (`use_clipped_linears=true` on E4B). The runtime applies the
/// clip on input (saturate to [`input_min`, `input_max`]) before
/// the matmul and on output (saturate to [`output_min`,
/// `output_max`]) after; see HF Gemma4 `ClippedLinear.forward`.
/// The scalar clips are stored as f32 (loaded from bf16 at upload
/// time) so the runtime kernel can apply them with one multiply +
/// `fmaxf`/`fminf` per element with no per-launch dtype dance.
#[derive(Debug)]
pub struct ClippedLinearWeight {
    pub weight: F16Weight,
    pub input_max: f32,
    pub input_min: f32,
    pub output_max: f32,
    pub output_min: f32,
}

#[derive(Debug)]
pub struct Gemma4AudioFfn {
    pub pre_norm: F16Weight,
    pub post_norm: F16Weight,
    pub layer_1: ClippedLinearWeight,
    pub layer_2: ClippedLinearWeight,
}

#[derive(Debug)]
pub struct Gemma4AudioLConv1d {
    pub pre_norm: F16Weight,
    pub conv_norm: F16Weight,
    pub depthwise: F16Weight,
    pub linear_start: ClippedLinearWeight,
    pub linear_end: ClippedLinearWeight,
}

#[derive(Debug)]
pub struct Gemma4AudioAttention {
    pub q: ClippedLinearWeight,
    pub k: ClippedLinearWeight,
    pub v: ClippedLinearWeight,
    pub post: ClippedLinearWeight,
    pub relative_k: F16Weight,
    pub per_dim_scale: F16Weight,
}

#[derive(Debug)]
pub struct Gemma4AudioBlock {
    pub norm_pre_attn: F16Weight,
    pub norm_post_attn: F16Weight,
    pub norm_out: F16Weight,
    pub feed_forward1: Gemma4AudioFfn,
    pub feed_forward2: Gemma4AudioFfn,
    pub lconv1d: Gemma4AudioLConv1d,
    pub self_attn: Gemma4AudioAttention,
}

#[derive(Debug)]
pub struct Gemma4AudioSubsample {
    pub input_proj: F16Weight,   // `input_proj_linear.weight`
    pub layer0_conv: F16Weight,  // 2D Conv kernel=(3,3) stride=(2,2)
    pub layer0_norm: F16Weight,
    pub layer1_conv: F16Weight,
    pub layer1_norm: F16Weight,
}

#[derive(Debug)]
pub struct Gemma4Audio {
    pub subsample: Gemma4AudioSubsample,
    pub blocks: Vec<Gemma4AudioBlock>,
    pub output_proj_w: F16Weight, // [output_proj_dims, hidden]
    pub output_proj_b: F16Weight, // [output_proj_dims]
    /// `model.embed_audio.embedding_projection.weight` —
    /// [text_hidden, output_proj_dims] linear from audio-tower
    /// output to the text decoder hidden width. Applied after the
    /// audio tower's output_proj when splicing into the prefill
    /// residual.
    pub embed_audio_projection: F16Weight,
}

#[derive(Debug)]
pub struct Gemma4VisionBlock {
    pub input_layernorm_w: F16Weight,
    pub post_attention_layernorm_w: F16Weight,
    pub pre_feedforward_layernorm_w: F16Weight,
    pub post_feedforward_layernorm_w: F16Weight,
    pub q_proj_w: F16Weight,
    pub k_proj_w: F16Weight,
    pub v_proj_w: F16Weight,
    pub o_proj_w: F16Weight,
    pub q_norm_w: F16Weight, // [72]
    pub k_norm_w: F16Weight, // [72]
    pub gate_proj_w: F16Weight,
    pub up_proj_w: F16Weight,
    pub down_proj_w: F16Weight,
}
