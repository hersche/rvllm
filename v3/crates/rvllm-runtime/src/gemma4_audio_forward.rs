//! Gemma 4 E4B native audio encoder forward — types + legacy
//! API stub.
//!
//! The LIVE audio path is `Gemma4Bringup::forward_gemma_audio_to_host`
//! in `gemma4_bring_up.rs`. The cuda-worker calls it directly. It
//! runs the full chain: mel compute (CPU) -> subsample (im2col +
//! cuBLASLt + layernorm/ReLU x2) -> 12 encoder blocks (FFN + chunked
//! attention with rel-pos bias + LightConv1D + FFN + norms) ->
//! output_proj (1024 -> 1536) -> output_proj_b bias + parameter-
//! free RMSNorm in f32 -> embed_audio_projection (1536 -> 2560) ->
//! DtoH the f16 [num_soft_tokens, text_hidden] bytes for splice
//! into the prefill residual. Validated on real user speech
//! recordings — "Das ist ein Test.", "I am a human.", "Wie ist das
//! Wetter heute?" — transcribed correctly.
//!
//! This file only provides the public `AudioForwardOutput` struct
//! that the cuda-worker holds and the legacy `forward_gemma_audio`
//! entry which is kept as a `FeatureNotAvailable` stub for any
//! caller still using the old API. New code should call the
//! Gemma4Bringup method directly.
//!
//! ## HF source cross-reference
//!
//! `/home/r00t/.unsloth/studio/.venv_t5/transformers/models/gemma4/modeling_gemma4.py`
//!   * `Gemma4AudioSubSampleConvProjection.forward`      (~line 345)
//!   * `Gemma4AudioLightConv1d.forward`                  (~line 444)
//!   * `Gemma4AudioFeedForward.forward`                  (~line 375)
//!   * `Gemma4AudioRelPositionalEncoding.forward`        (~line 178)
//!   * `Gemma4AudioAttention.forward`                    (~line 209)
//!   * `Gemma4AudioLayer.forward`                        (~line 485)
//!   * `Gemma4AudioModel.forward`                        (~line 1820)

use rvllm_core::{AttentionError, AttnCtx, Result, RvllmError};

/// Output of `forward_gemma_audio`. Device pointer carries the
/// audio-tower's pre-`embed_audio_projection` output (width =
/// `output_proj_dims`, default 1536 on E4B). The caller (splice
/// site in `gemma4_bring_up::run_generate`, B7) does the final
/// `[output_proj_dims → text_hidden]` projection via
/// `model.embed_audio.embedding_projection.weight` and writes
/// the result into the prefill residual at the AudioSlot rows.
///
/// `num_soft_tokens` equals
/// `MelExtractor::num_soft_tokens(samples.len())` AND
/// `audio_item.num_soft_tokens` — both predicted host-side at
/// admission. The encoder forward must produce exactly that many
/// rows or the worker errors out (mirrors the vision side's
/// num-tokens contract check).
pub struct AudioForwardOutput {
    /// Device pointer to a fresh f16 buffer of shape
    /// `[num_soft_tokens, output_proj_dims]`. Lifetime is the
    /// arena's per-request checkpoint.
    pub device_ptr: u64,
    pub num_soft_tokens: usize,
    pub output_proj_dims: usize,
}

/// Borrow bundle for `forward_gemma_audio`. Parallel to
/// `qwen_vision_forward::Qwen36VisionDeps`. Built by the cuda
/// worker thread from `Gemma4Bringup` (arch + loaded audio tower
/// + arena + kernel fn pointers).
#[cfg(feature = "cuda")]
pub struct Gemma4AudioDeps<'a> {
    pub arch: &'a rvllm_loader::gemma4_arch::Gemma4Arch,
    pub audio_cfg: &'a rvllm_loader::gemma4_arch::Gemma4AudioConfig,
    pub audio: &'a rvllm_loader::gemma4_weights::Gemma4Audio,
    pub arena: &'a rvllm_mem::HbmArena<'a>,
    pub stream: u64,
    /// MelExtractor used to compute `num_soft_tokens` host-side;
    /// the encoder forward asserts the produced row count matches.
    pub mel_extractor: &'a crate::audio_preprocess::MelExtractor,
}

/// Run the audio tower on `samples_16k_mono` and return the
/// pre-`embed_audio_projection` output (device-resident f16).
///
/// **B6a status**: returns
/// `Err(AttentionError::FeatureNotAvailable)` with a clear
/// message. The caller (B7 splice site) must NOT silently fall
/// through — instead surface this error to the client so the
/// failure mode is visible during the B6b..d rollout.
#[cfg(feature = "cuda")]
pub fn forward_gemma_audio(
    deps: &Gemma4AudioDeps<'_>,
    samples_16k_mono: &[f32],
) -> Result<AudioForwardOutput> {
    // Predict the soft-token count host-side. The encoder MUST
    // produce exactly this many rows; B6b..d enforce it. The
    // admission path already capped the input to
    // `audio_seq_length` samples, so this can't exceed
    // `MelConfig::audio_seq_length` (default 750 on E4B).
    let num_soft_tokens = deps.mel_extractor.num_soft_tokens(samples_16k_mono.len());
    let _ = (deps.audio_cfg, deps.audio, deps.arena, deps.stream, num_soft_tokens);
    // Argument silencing: each field is read by B6b..d. Hold
    // their references here so an early integration regression
    // surfaces as `unused` rather than hiding silently.
    let _ = (
        &deps.audio_cfg.hidden_size,
        &deps.audio_cfg.num_hidden_layers,
        &deps.audio_cfg.output_proj_dims,
        &deps.audio_cfg.attention_chunk_size,
        &deps.audio_cfg.attention_context_left,
        &deps.audio_cfg.residual_weight,
        &deps.audio_cfg.use_clipped_linears,
        &deps.audio.subsample.input_proj.offset_bytes,
        deps.audio.blocks.len(),
        &deps.audio.output_proj_w.offset_bytes,
        &deps.audio.embed_audio_projection.offset_bytes,
        &deps.arch.audio_config,
    );

    Err(RvllmError::Attention {
        err: AttentionError::FeatureNotAvailable {
            op: "forward_gemma_audio (legacy stub — call Gemma4Bringup::forward_gemma_audio_to_host instead)",
            backend: "Gemma4Audio",
        },
        ctx: AttnCtx {
            op: "forward_gemma_audio",
            stream: deps.stream,
            num_seqs: 1,
            head_dim: deps.audio_cfg.hidden_size as u32
                / deps.audio_cfg.num_attention_heads.max(1) as u32,
        },
        bt: std::backtrace::Backtrace::capture(),
    })
}

/// Mock variant for the non-cuda test build so the binary still
/// links when the worker is compiled without GPU support.
#[cfg(not(feature = "cuda"))]
pub fn forward_gemma_audio_mock() -> Result<AudioForwardOutput> {
    Err(RvllmError::Attention {
        err: AttentionError::FeatureNotAvailable {
            op: "forward_gemma_audio (non-cuda build)",
            backend: "mock",
        },
        ctx: AttnCtx {
            op: "forward_gemma_audio",
            stream: 0,
            num_seqs: 0,
            head_dim: 0,
        },
        bt: std::backtrace::Backtrace::capture(),
    })
}
