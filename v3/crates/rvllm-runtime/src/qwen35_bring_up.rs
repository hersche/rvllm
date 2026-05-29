//! Qwen 3.5 27B dense — Phase 0 bring-up scaffolding.
//!
//! See `v3/QWEN35_BRINGUP_PLAN.md` for the phase plan. Today the
//! engine only:
//!
//!  1. Parses + validates the config.json via `Qwen35Arch::from_dir`.
//!  2. Logs the arch summary.
//!  3. Returns a `NotImplemented` error from every generate-path
//!     entry point so the operator gets a clean startup signal
//!     pointing at the unwired phases.
//!
//! Phases (to be implemented in subsequent sessions):
//!   * Phase 1 — FP8 weight inventory + safetensors loader
//!     (mirrors qwen36_load with no MoE expert tensors, no router).
//!   * Phase 2 — decoder forward (linear + full attn reusing
//!     qwen36's gated_delta_rule + FA2 kernels; dense MLP via
//!     gate/up/down/silu_mul reusing Mistral 3.5 / Gemma 4 kernels).
//!   * Phase 3 — Qwen3-VL ViT (27 blocks, hidden=1152, intermediate
//!     =4304, patch_embed [1152,3,2,16,16], patch merger
//!     [4608→4608→5120]).
//!   * Phase 4 — slot-aware vision splice (port from
//!     `Mistral35Bringup::generate_with_vision_slots`).
//!   * Phase 5 — perf optimisations (FA-decode, batched prefill,
//!     codex round on the new arch).

use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};
#[cfg(feature = "cuda")]
use rvllm_cutlass::{cublaslt::CublasLt, CutlassBackend};
#[cfg(feature = "cuda")]
use rvllm_kernels::{KernelFn, KernelLoader, LoadedModule};
#[cfg(feature = "cuda")]
use rvllm_loader::qwen35_weights::Qwen35LoadedModel;
#[cfg(feature = "cuda")]
use rvllm_mem::{context::CudaContextHandle, stream::Stream, HbmArena};

use crate::qwen35_arch::Qwen35Arch;

/// Engine paths — identical shape to the other family bring-ups so
/// `cuda_worker` can spawn either through the same interface.
#[derive(Clone, Debug)]
pub struct Qwen35EnginePaths {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub cutlass_so: PathBuf,
    pub fa3_so: PathBuf,
    pub policy_json: PathBuf,
}

/// KV dtype carried by the per-layer cache. F16 is the legacy /
/// production path; Nvfp4 is the packed-4-bit + E4M3 microscale
/// path matching the Gemma 4 NVFP4 KV plumbing (4 elements / 2
/// bytes for packed K/V, plus `[blocks, head_dim/16]` E4M3
/// scales).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35KvDtype {
    F16,
    Nvfp4,
}

/// Per-full-attn-layer KV cache pair. Linear-attn layers don't
/// carry a positional KV cache (state is SSM-style in
/// `Qwen35LinearState`).
///
/// For F16 the packed/scale pointers are zero. For NVFP4 the
/// `k_ptr` / `v_ptr` are the packed 4-bit buffers (1 byte per 2
/// elements) and the `k_scale_ptr` / `v_scale_ptr` are the
/// `[blocks, head_dim/16]` E4M3 microscale buffers; `dtype` is
/// `Nvfp4`. Decoders dispatch via the dtype field.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct Qwen35LayerKv {
    pub k_ptr: u64,
    pub v_ptr: u64,
    pub k_scale_ptr: u64,
    pub v_scale_ptr: u64,
    pub dtype: Qwen35KvDtype,
}

/// Sparse KV cache — only the 16 full-attn layers carry a
/// `Qwen35LayerKv`. `layer_idx_to_full_idx[layer_idx]` returns
/// `Some(full_rank)` for full-attn layers and `None` for linear.
///
/// `dtype` is uniform across all full-attn layers (per request and
/// per profile). NVFP4 enabled via `RVLLM_NVFP4_KV=1` follows the
/// same env-knob convention as the Gemma 4 path.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct Qwen35KvCache {
    pub layers: Vec<Qwen35LayerKv>,
    pub layer_idx_to_full_idx: Vec<Option<usize>>,
    pub max_pos: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub per_layer_bytes: usize,
    /// Per-layer scale-buffer bytes for NVFP4 (zero for F16). Each
    /// full-attn layer holds a K-scale and a V-scale buffer of this
    /// size; the per-layer entry stores their device pointers.
    pub per_layer_scale_bytes: usize,
    pub dtype: Qwen35KvDtype,
    /// Persistent identity-permutation block_tables, shape
    /// `[max_pos]` i32, value `i` at index `i`. Allocated +
    /// filled once at bring-up. With `block_size=1` this is the
    /// straight identity mapping the FA-2 paged kernel expects;
    /// the only thing that changes per decode/prefill step is
    /// `context_lens` below. Was rebuilt on the host every step
    /// per full-attn layer (Codex review #2, 2026-05-12) —
    /// O(T²) host work + HtoD bandwidth for nothing.
    pub block_tables_ptr: u64,
    /// Persistent device i32 [1] holding the current
    /// `context_len`. Written per step from the host; the FA-2
    /// kernel reads it. Small enough that an HtoD here costs
    /// less than the kernel launch following it.
    pub context_lens_ptr: u64,
}

/// Per-linear-attn-layer SSM state. Layout: one contiguous region
/// holding all 48 layers' states. Per-layer offset is
/// `linear_attn_rank × per_layer_bytes`. Zero-initialised at
/// bring-up (session reset). Phase 2c forward will update + read
/// this region via the existing `gated_delta_rule_decode_f16`
/// kernel.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct Qwen35LinearState {
    pub base_ptr: u64,
    pub per_layer_bytes: usize,
    pub n_linear_layers: usize,
    pub num_ssm_heads: usize,
    pub d_state: usize,
    pub layer_idx_to_linear_idx: Vec<Option<usize>>,
    /// Per-linear-layer causal-conv1d state ring (ks-1 timesteps
    /// of conv input × conv_dim per layer). Zero-initialised at
    /// bring-up; consumed by `conv_state_advance_f16` then
    /// updated in-place every linear-attn forward step. Indexed
    /// the same way as the delta-rule state (via
    /// `layer_idx_to_linear_idx`).
    pub conv_state_base_ptr: u64,
    pub conv_state_per_layer_bytes: usize,
}

/// Linear-attention head/dim config for Qwen 3.5. Read at bring-up
/// from `config.json` (`text_config.linear_num_{key,value}_heads`,
/// `linear_{key,value}_head_dim`, `linear_conv_kernel_dim`). Derived
/// fields (`key_dim`, `value_dim`, `conv_dim`, `v_per_k`) are
/// pre-computed so the per-layer forward path doesn't recompute them
/// on every layer.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct Qwen35LaDims {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel_dim: usize,
    pub key_dim: usize,         // num_k_heads * head_k_dim
    pub value_dim: usize,       // num_v_heads * head_v_dim
    pub conv_dim: usize,        // 2*key_dim + value_dim  (= in_proj_qkv n)
    pub v_per_k: usize,         // num_v_heads / num_k_heads
}

/// Host-built partial-rotary cos/sin tables. Shape
/// `[max_pos, rotary_dim/2]` F16. Indexed by `(pos, i)` where
/// `i < rotary_dim/2`.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct Qwen35RopeTables {
    pub cos_ptr: u64,
    pub sin_ptr: u64,
    pub max_pos: usize,
    pub rotary_dim: usize,
}

/// Per-decode-token forward scratch. Reusable across requests
/// via `arena.checkpoint() / restore()`. Phase 2c will allocate
/// a separate T-batched scratch for prefill.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct Qwen35Scratch {
    pub token_in_ptr: u64,     // i32 [1]
    pub h_residual_ptr: u64,   // F16 [hidden]
    pub h_work_ptr: u64,       // F16 [hidden]
    pub q_out_ptr: u64,        // F16 [n_q_heads * head_dim] (full-attn)
    pub q_gate_ptr: u64,       // F16 [n_q_heads * head_dim]  output gate slice
    pub k_out_ptr: u64,        // F16 [n_kv_heads * head_dim]
    pub v_out_ptr: u64,        // F16 [n_kv_heads * head_dim]
    pub attn_out_ptr: u64,     // F16 [n_q_heads * head_dim]
    pub o_out_ptr: u64,        // F16 [hidden]
    pub gate_out_ptr: u64,     // F16 [intermediate]
    pub up_out_ptr: u64,       // F16 [intermediate]
    pub silu_mid_ptr: u64,     // F16 [intermediate]
    pub down_out_ptr: u64,     // F16 [hidden]
    pub logits_ptr: u64,       // F32 [vocab]
    pub token_out_ptr: u64,    // i32 [1]
    // ── Phase 2c-B scratch: FP8-quantized intermediate for the
    // dense MLP's down_proj cuBLASLt fp8_gemm_blockwise call.
    pub silu_mid_fp8_ptr: u64,    // FP8 [intermediate]
    pub silu_mid_scales_ptr: u64, // F32 [intermediate / 128]
    // ── NVFP4 KV scratch (zero unless RVLLM_NVFP4_KV=1) ────
    // q_fp8 holds the rotated FP8-E4M3 Q output produced by the
    // NVFP4 RoPE kernel — shape [n_q_heads * head_dim] u8.
    // q_scale_cache holds the per-(token, head) dynamic Q scale
    // — shape [num_tokens, n_q_heads] f32. The "decode" scratch
    // is sized for num_tokens=1; prefill carries its own batched
    // version through the arena per call.
    pub q_fp8_ptr: u64,
    pub q_scale_cache_ptr: u64,
    pub scratch_bytes: usize,
}

/// Outside-path + per-layer kernel handles. Phase 2c-A loads the
/// outside set (embed / final-norm-with-FP8-quant / argmax /
/// cuBLASLt fp8_gemm). Phase 2c-B adds the layer kernels for
/// full-attn / linear-attn / dense MLP dispatch. Dispatch itself
/// (the per-layer forward loop) lands in Phase 2c-B-core.
#[cfg(feature = "cuda")]
pub struct Qwen35OutsideKernels {
    // ── Outside path (Phase 2c-A) ───────────────────────────
    pub embedding_gather_f16_mod: LoadedModule,
    pub fn_embedding_gather_f16: KernelFn,
    pub fused_rmsnorm_fp8_quant_mod: LoadedModule,
    pub fn_fused_rmsnorm_fp8_quant: KernelFn,
    pub argmax_mod: LoadedModule,
    pub fn_argmax_f16: KernelFn,
    /// temperature + top_k + top_p categorical sampler over f16 logits
    /// (lives in the same argmax.ptx module). Used when the request asks
    /// for stochastic decoding; degenerates to argmax at temp<=0.
    pub fn_sample_topk_topp_f16: KernelFn,
    // ── Per-layer (Phase 2c-B) — RMSNorm + dense MLP ────────
    pub rmsnorm_inplace_f16_mod: LoadedModule,
    pub fn_rmsnorm_inplace_f16: KernelFn,
    pub fp8_gemv_dual_silu_mod: LoadedModule,
    pub fn_fp8_gemv_blockwise_wpr_native_f16in_dual_silu: KernelFn,
    pub fp8_quantize_per_token_f16_mod: LoadedModule,
    pub fn_fp8_quantize_per_token_f16: KernelFn,
    /// Per-token amax quantizer used by CUTLASS SM120's blockscale
    /// FP8 GEMM path. Default-off for Qwen35 MLP because it changes
    /// activation numerics versus the f16-input GEMV reference.
    pub fp8_quantize_per_token_amax_f16_mod: LoadedModule,
    pub fn_fp8_quantize_per_token_amax_f16: KernelFn,
    /// Pointwise SwiGLU epilogue for the experimental CUTLASS MLP path
    /// (separate gate/up GEMMs followed by SiLU(gate) * up).
    pub silu_mul_f16_mod: LoadedModule,
    pub fn_silu_mul_f16: KernelFn,
    /// Single-output FP8 GEMV with F16 input + blockwise scale.
    /// Qwen 3.6's M=1 fallback when cuBLASLt has no blockwise FP8
    /// algo (the sm_121 case). Reused here for down_proj. SM100+
    /// gated — `None` on older devices, but GB10 (sm_121) has it.
    pub fp8_gemv_mod: LoadedModule,
    pub fn_fp8_gemv_wpr_native_f16in: Option<KernelFn>,
    // ── Per-layer (Phase 2c-B) — full-attn ─────────────────
    pub split_q_gate_f16_mod: LoadedModule,
    pub fn_split_q_gate_f16: KernelFn,
    pub fused_rope_qwen_partial_f16kv_mod: LoadedModule,
    pub fn_fused_rope_qwen_partial_f16kv: KernelFn,
    /// Qwen-specific NVFP4 RoPE + KV-write + FP8-Q kernel. Loaded
    /// only when `RVLLM_NVFP4_KV=1`. Sibling of the Gemma 4 kernel
    /// `fused_rope_partial_nvfp4kv`. Step 3 plumbing-only commit:
    /// the kernel is resident but no dispatch site invokes it yet
    /// (step 4 wires the decode path; step 5 wires prefill).
    pub fused_rope_qwen_partial_nvfp4kv_mod: Option<LoadedModule>,
    pub fn_fused_rope_qwen_partial_nvfp4kv: Option<KernelFn>,
    /// `flash_attention_2_decode_nvfp4kv_kernel` — paged FA-2 decode
    /// kernel that reads packed-4-bit K/V + per-(slot, kv_head)
    /// E4M3 microscale and writes f16 output. Loaded only when
    /// `RVLLM_NVFP4_KV=1`. One CTA per (seq, query_head); GQA cap
    /// (`MAX_GQA_DECODE=4`) does not apply to this per-head path,
    /// so Qwen 3.6 27B's GQA=6 is fine.
    pub flash_attention_nvfp4kv_mod: Option<LoadedModule>,
    pub fn_flash_attention_2_decode_nvfp4kv: Option<KernelFn>,
    pub flash_attention_mod: LoadedModule,
    pub fn_flash_attention_2_decode_f16io: KernelFn,
    pub sigmoid_mul_f16_mod: LoadedModule,
    pub fn_sigmoid_mul_f16: KernelFn,
    // ── Per-layer (Phase 2c-B) — linear-attn ───────────────
    pub conv_state_advance_f16_mod: LoadedModule,
    pub fn_conv_state_advance_f16: KernelFn,
    pub causal_conv1d_f16_mod: LoadedModule,
    pub fn_causal_conv1d_f16: KernelFn,
    pub qwen_linear_alpha_beta_f16_mod: LoadedModule,
    pub fn_qwen_linear_alpha_beta_f16: KernelFn,
    pub qwen_linear_silu_l2_gqa_f16_mod: LoadedModule,
    pub fn_qwen_linear_silu_l2_gqa_f16: KernelFn,
    pub gated_delta_rule_decode_f16_mod: LoadedModule,
    pub fn_gated_delta_rule_decode_f16: KernelFn,
    pub qwen_linear_rmsnorm_gated_f16_mod: LoadedModule,
    pub fn_qwen_linear_rmsnorm_gated_f16: KernelFn,
    // ── Batched-prefill linear-attn (Phase #1-b) ───────────
    // Batched conv1d state advance: assembles [N+ks-1, conv_dim]
    // history block from the persistent (ks-1)-row state cache
    // and the current N-token QKV stripe, then rotates the state
    // to its tail (ks-1) rows in one launch.
    pub conv_state_advance_batched_f16_mod: LoadedModule,
    pub fn_conv_state_advance_batched_f16: KernelFn,
    // Batched gated-delta-rule kernel — one launch advances the
    // SSM state across all N tokens internally and writes the
    // [N, num_v_heads, head_v_dim] readout. Replaces N×
    // `gated_delta_rule_decode_f16` launches at prefill time.
    pub gated_delta_rule_prefill_f16_mod: LoadedModule,
    pub fn_gated_delta_rule_prefill_f16: KernelFn,
    // Phase #1-c: batched-prefill full-attn support.
    // `flash_attention_2_f16kv_kernel` is the prefill-time
    // FA-2 kernel (f32-Q / f16-KV / f32-O, causal-masked over
    // num_query_tokens=N). Lives in the same `flash_attention`
    // PTX module that already supplied the decode-time kernel.
    pub fn_flash_attention_2_f16kv: KernelFn,
    // `cast_f16_to_f32_kernel` lives in the same `cast_fp`
    // module as the already-loaded `fn_cast_f32_to_f16`; this
    // is the inverse direction needed to sandwich the
    // f32-IO FA-2 prefill kernel between our f16 GEMVs.
    pub fn_cast_f16_to_f32: KernelFn,
    // ── Vector residual add (used after attn + MLP) ────────
    pub f16_plus_f32_inplace_f16_mod: LoadedModule,
    pub fn_f16_plus_f32_inplace_f16: KernelFn,
    pub vector_add_f16_mod: LoadedModule,
    pub fn_vector_add_f16: KernelFn,
    /// Phase 8 other-models fusion (2026-05-23): fused FP8 GEMV
    /// (f16 input/output) + in-place residual add. Single-launch
    /// replacement for the back-to-back pair
    /// (`fp8_gemv_blockwise_wpr_native_f16in_kernel` →
    /// `vector_add_f16_kernel`) used per layer × 2 sites
    /// (post-attn residual after o_proj, post-FFN residual after
    /// ffn_down). 80 launches/token saved on 40-layer Qwen 3.5/3.6
    /// 27B dense.
    pub fp8_gemv_f16in_residual_add_mod: LoadedModule,
    pub fn_fp8_gemv_f16in_residual_add: KernelFn,
    // ── Vision (Qwen3-VL ViT, Phase 3-a-v) ─────────────────
    // 11 modules + 12 kernel handles. cast_fp_mod carries both
    // cast_f32_to_f16 (used here) and cast_f16_to_f32 (unused,
    // kept resident so a future vision-debug dump-to-f32 doesn't
    // need a second load). The vector_add_f16 above is shared
    // with the text path.
    pub layernorm_inplace_f16_mod: LoadedModule,
    pub fn_layernorm_inplace_f16: KernelFn,
    pub gelu_tanh_f16_mod: LoadedModule,
    pub fn_gelu_tanh_f16: KernelFn,
    pub softmax_row_f16_mod: LoadedModule,
    pub fn_softmax_row_f16: KernelFn,
    pub vit_rotary_2d_f16_mod: LoadedModule,
    pub fn_vit_rotary_2d_f16: KernelFn,
    pub vit_pos_embed_interp_f16_mod: LoadedModule,
    pub fn_vit_pos_embed_interp_f16: KernelFn,
    pub scale_inplace_f16_mod: LoadedModule,
    pub fn_scale_inplace_f16: KernelFn,
    pub transpose_2d_f16_mod: LoadedModule,
    pub fn_transpose_2d_f16: KernelFn,
    pub add_bias_f16_mod: LoadedModule,
    pub fn_add_bias_f16: KernelFn,
    pub cast_fp_mod: LoadedModule,
    pub fn_cast_f32_to_f16: KernelFn,
    pub extract_head_f16_mod: LoadedModule,
    pub fn_extract_head_f16: KernelFn,
    pub fn_scatter_head_f16: KernelFn,
    /// Phase-perf 2: batched ViT attention support (mirrors Qwen 3.6).
    pub softmax_row_f32_to_f16_mod: LoadedModule,
    pub fn_softmax_row_f32_to_f16: KernelFn,
    pub transpose_heads_v_f16_mod: LoadedModule,
    pub fn_transpose_heads_v_f16: KernelFn,
    pub scatter_heads_f16_mod: LoadedModule,
    pub fn_scatter_heads_f16: KernelFn,
    pub scale_inplace_f32_mod: LoadedModule,
    pub fn_scale_inplace_f32: KernelFn,
}

/// Phase 2c-A engine handle. Adds outside kernels + cuBLASLt on
/// top of Phase 2b (substrate). Forward path now executes the
/// embed → final-RMSNorm → lm_head → argmax pipe end-to-end,
/// SKIPPING all 64 transformer layers. Output is structurally
/// valid (a token id) but semantically meaningless until Phase
/// 2c-B wires the per-layer forward.
pub struct Qwen35Bringup {
    pub paths: Qwen35EnginePaths,
    pub arch: Qwen35Arch,
    pub arena_bytes: usize,
    #[cfg(feature = "cuda")]
    pub ctx: Option<Arc<CudaContextHandle>>,
    #[cfg(feature = "cuda")]
    pub arena: Option<HbmArena<'static>>,
    #[cfg(feature = "cuda")]
    pub stream: Option<Stream>,
    /// Loaded weights (outside tensors + 64 per-layer slots). Access
    /// `.outside.embed_tokens` etc. through `model.as_ref()`.
    #[cfg(feature = "cuda")]
    pub model: Option<Qwen35LoadedModel>,
    #[cfg(feature = "cuda")]
    pub kv_cache: Option<Qwen35KvCache>,
    #[cfg(feature = "cuda")]
    pub linear_state: Option<Qwen35LinearState>,
    #[cfg(feature = "cuda")]
    pub rope_tables: Option<Qwen35RopeTables>,
    #[cfg(feature = "cuda")]
    pub scratch: Option<Qwen35Scratch>,
    #[cfg(feature = "cuda")]
    pub outside_kernels: Option<Qwen35OutsideKernels>,
    /// SM121 FA2 backend for the NVFP4 batched-prefill full-attn path.
    #[cfg(feature = "cuda")]
    pub attn_backend_full: Option<rvllm_attention::AttentionBackend>,
    #[cfg(feature = "cuda")]
    pub cublaslt: Option<CublasLt>,
    #[cfg(feature = "cuda")]
    pub cutlass: CutlassBackend,
    /// Linear-attn head/dim config. Always present once `load()`
    /// completes; defaulted to Qwen 3.5 27B (`16/48/128/128, ks=4`)
    /// when the config is missing the keys.
    #[cfg(feature = "cuda")]
    pub la_dims: Option<Qwen35LaDims>,
    /// Per-request token-sampling state. Set by the worker before each
    /// generate via [`Qwen35Bringup::set_sampling`]; read at the
    /// token-selection launch site. When `temperature <= 0` the path
    /// degenerates to argmax (greedy) — byte-identical to the legacy
    /// behaviour, so non-sampling callers (and other model families)
    /// are unaffected. Qwen 3 is documented to repeat/degrade under
    /// greedy; its recommended sampling is temp=0.6/top_k=20/top_p=0.95.
    pub sampling: Qwen35SamplingState,
}

/// Atomically-published token-sampling parameters for the qwen35 decode
/// loop. f32 values are stored as their bit patterns so the whole struct
/// stays lock-free (the worker writes once per request, the decode loop
/// reads once per token and advances `seed`).
#[derive(Default)]
pub struct Qwen35SamplingState {
    pub temp_bits: std::sync::atomic::AtomicU32,
    pub top_k: std::sync::atomic::AtomicU32,
    pub top_p_bits: std::sync::atomic::AtomicU32,
    pub seed: std::sync::atomic::AtomicU64,
}

impl Qwen35Bringup {
    /// Publish per-request sampling params before a generate call. The
    /// decode loop reads these once per token. `temperature <= 0` (or
    /// `top_k <= 1`) keeps the greedy/argmax path. `base_seed` seeds the
    /// per-token RNG; the launch site advances it per emitted token so a
    /// retry with the same seed is reproducible.
    pub fn set_sampling(
        &self,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        base_seed: u64,
    ) {
        use std::sync::atomic::Ordering::Relaxed;
        self.sampling.temp_bits.store(temperature.to_bits(), Relaxed);
        self.sampling.top_k.store(top_k, Relaxed);
        self.sampling.top_p_bits.store(top_p.to_bits(), Relaxed);
        self.sampling.seed.store(base_seed, Relaxed);
    }

    /// Snapshot the published sampling params. Returns `None` when the
    /// path should stay greedy (temperature <= 0 or top_k <= 1).
    #[cfg(feature = "cuda")]
    fn sampling_snapshot(&self) -> Option<(f32, i32, f32, u64)> {
        use std::sync::atomic::Ordering::Relaxed;
        let temp = f32::from_bits(self.sampling.temp_bits.load(Relaxed));
        let top_k = self.sampling.top_k.load(Relaxed) as i32;
        if temp <= 0.0 || top_k <= 1 {
            return None;
        }
        let top_p = f32::from_bits(self.sampling.top_p_bits.load(Relaxed));
        // Advance the per-token seed so successive tokens draw independent
        // uniforms; the kernel also mixes in `row`.
        let seed = self
            .sampling
            .seed
            .fetch_add(0x9E3779B97F4A7C15, Relaxed);
        Some((temp, top_k, top_p, seed))
    }

    /// Launch the per-token LM-head token selection over the f16 logits
    /// at `logits_ptr` (vocab elems, written by fp8_gemm) into the i32
    /// slot at `out_ptr`. Greedy by default (`argmax_f16_kernel`);
    /// dispatches `sample_topk_topp_f16_kernel` when this request
    /// published `temperature > 0` via [`set_sampling`]. Single block,
    /// `min(vocab, 1024)` threads — same launch geometry as argmax.
    #[cfg(feature = "cuda")]
    unsafe fn launch_token_select(
        &self,
        ker: &Qwen35OutsideKernels,
        logits_ptr: u64,
        out_ptr: u64,
        vocab: u32,
        stream_raw: u64,
        op_label: &'static str,
    ) -> Result<()> {
        use cudarc::driver::sys::*;
        let block_dim: u32 = vocab.min(1024);
        let mut row_ptr = logits_ptr;
        let mut out = out_ptr;
        let mut vsz: i32 = vocab as i32;
        let rc = if let Some((temp, top_k, top_p, seed)) = self.sampling_snapshot() {
            let mut t = temp;
            let mut k = top_k;
            let mut p = top_p;
            let mut sd = seed;
            let args = [
                (&mut row_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut vsz) as *mut i32 as *mut core::ffi::c_void,
                (&mut t) as *mut f32 as *mut core::ffi::c_void,
                (&mut k) as *mut i32 as *mut core::ffi::c_void,
                (&mut p) as *mut f32 as *mut core::ffi::c_void,
                (&mut sd) as *mut u64 as *mut core::ffi::c_void,
            ];
            cuLaunchKernel(
                ker.fn_sample_topk_topp_f16.raw() as CUfunction,
                1, 1, 1, block_dim, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            )
        } else {
            let args = [
                (&mut row_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut vsz) as *mut i32 as *mut core::ffi::c_void,
            ];
            cuLaunchKernel(
                ker.fn_argmax_f16.raw() as CUfunction,
                1, 1, 1, block_dim, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            )
        };
        if rc != CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::cuda(
                op_label,
                rvllm_core::CudaErrorKind::LaunchFailed,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        Ok(())
    }
}

impl std::fmt::Debug for Qwen35Bringup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35Bringup")
            .field("paths", &self.paths)
            .field("arch", &self.arch)
            .field("arena_bytes", &self.arena_bytes)
            .finish_non_exhaustive()
    }
}

impl Qwen35Bringup {
    /// Phase 1a: parse + validate `config.json`, allocate the arena,
    /// upload outside-the-stack tensors. NO per-layer upload, NO
    /// forward path. See QWEN35_BRINGUP_PLAN.md.
    pub fn load(paths: Qwen35EnginePaths, arena_bytes: usize) -> Result<Self> {
        let arch = Qwen35Arch::from_dir(&paths.model_dir)?
            .ok_or_else(|| corrupt(
                paths.model_dir.clone(),
                "Qwen35Bringup::load: config.json does not match Qwen 3.5 \
                 dense markers (model_type==qwen3_5, attn_output_gate==true, \
                 num_experts absent/0, intermediate_size present)".into(),
            ))?;
        arch.log_summary();

        #[cfg(feature = "cuda")]
        {
            let ctx = Arc::new(CudaContextHandle::init(0)?);
            #[cfg(feature = "gb10")]
            let arena = {
                let (major, minor) = ctx.compute_capability();
                let target = rvllm_core::CompileTarget::from_compute_capability(
                    major, minor);
                if matches!(target, Some(rvllm_core::CompileTarget::Sm121)) {
                    rvllm_mem::UnifiedArena::new(&ctx, arena_bytes)?.into_inner()
                } else {
                    HbmArena::new(&ctx, arena_bytes)?
                }
            };
            #[cfg(not(feature = "gb10"))]
            let arena = HbmArena::new(&ctx, arena_bytes)?;
            let arena: HbmArena<'static> = unsafe { std::mem::transmute(arena) };
            let stream = Stream::new(&ctx)?;
            let compile_target: Option<rvllm_core::CompileTarget> = {
                let (major, minor) = ctx.compute_capability();
                rvllm_core::CompileTarget::from_compute_capability(major, minor)
            };

            // Phase 1b: full per-layer upload. Reads `layer_types`
            // from the arch (3:1 linear:full pattern in the 27B
            // dense checkpoint) and routes each slot to the matching
            // builder. Dense MLP lands on every layer.
            let model = rvllm_loader::qwen35_load::load_qwen35_model(
                &paths.model_dir, &arena, &arch.base.layer_types,
            )?;

            // ── Phase 2b: scratch + KV + RoPE + linear-attn state ─
            let kv_max_pos: usize = std::env::var("RVLLM_QWEN35_KV_MAX_POS")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or(8192);

            // RoPE tables: partial rotary (rotary_dim = 64 of head_dim
            // = 256 for the 27B dense), [max_pos, rotary_dim/2] f16.
            let head_dim = arch.base.head_dim;
            let rotary_dim = (head_dim as f32 * arch.rope_partial_rotary_factor) as usize;
            let rope_tables = build_rope_tables(
                &arena, kv_max_pos, head_dim, rotary_dim,
                arch.rope_theta,
            )?;

            // KV cache: F16 [max_pos, n_kv_heads, head_dim] per FULL-
            // attn layer by default. NVFP4 path (opt-in via
            // RVLLM_NVFP4_KV=1) allocates packed-4-bit K/V (2 elements
            // per byte) plus a separate [max_pos, n_kv_heads,
            // head_dim/16] E4M3 scale buffer per layer, matching the
            // Gemma 4 NVFP4 KV layout. Read path dispatches on
            // `Qwen35LayerKv::dtype` (downstream wiring lands in
            // step 3+; this commit is the plumbing-only no-op for f16).
            let n_kv_heads = arch.base.num_key_value_heads;
            let nvfp4_kv = std::env::var("RVLLM_NVFP4_KV")
                .ok().as_deref().map(|s| s != "0" && !s.is_empty())
                .unwrap_or(false);
            let kv_dtype = if nvfp4_kv {
                Qwen35KvDtype::Nvfp4
            } else {
                Qwen35KvDtype::F16
            };
            let total_kv_elems = kv_max_pos * n_kv_heads * head_dim;
            // Packed NVFP4 = 1 byte per 2 elements. F16 = 2 bytes per
            // element. Scale buffer (NVFP4 only) = head_dim/16 E4M3
            // bytes per (pos, head). For F16 the scale bytes are 0.
            let (per_layer_kv_bytes, per_layer_scale_bytes) = match kv_dtype {
                Qwen35KvDtype::F16 => (total_kv_elems * 2, 0usize),
                Qwen35KvDtype::Nvfp4 => (
                    total_kv_elems / 2,
                    kv_max_pos * n_kv_heads * (head_dim / 16),
                ),
            };
            let mut kv_layers: Vec<Qwen35LayerKv> = Vec::new();
            let mut layer_idx_to_full_idx: Vec<Option<usize>> =
                vec![None; arch.base.num_hidden_layers];
            let mut next_full: usize = 0;
            for (li, ty) in arch.base.layer_types.iter().enumerate() {
                if matches!(ty, rvllm_loader::LayerAttnType::Full) {
                    let k_region = arena.region(
                        "qwen35_k_cache", per_layer_kv_bytes, 16)?;
                    let v_region = arena.region(
                        "qwen35_v_cache", per_layer_kv_bytes, 16)?;
                    let (k_scale_ptr, v_scale_ptr) = if per_layer_scale_bytes > 0 {
                        let ks = arena.region(
                            "qwen35_k_scale", per_layer_scale_bytes, 16)?;
                        let vs = arena.region(
                            "qwen35_v_scale", per_layer_scale_bytes, 16)?;
                        (ks.device_ptr(), vs.device_ptr())
                    } else {
                        (0u64, 0u64)
                    };
                    kv_layers.push(Qwen35LayerKv {
                        k_ptr: k_region.device_ptr(),
                        v_ptr: v_region.device_ptr(),
                        k_scale_ptr,
                        v_scale_ptr,
                        dtype: kv_dtype,
                    });
                    layer_idx_to_full_idx[li] = Some(next_full);
                    next_full += 1;
                }
            }
            // Persistent identity block_tables [max_pos] i32:
            // table[i] = i. With block_size=1 this is the straight
            // identity mapping FA-2 paged-attn expects; per-step
            // only `context_lens` (a single i32) needs to change.
            let bt_bytes = kv_max_pos * 4;
            let bt_region = arena.region(
                "qwen35_fattn_block_tables_persistent", bt_bytes, 16)?;
            let mut bt_host: Vec<u8> = Vec::with_capacity(bt_bytes);
            for i in 0..(kv_max_pos as i32) {
                bt_host.extend_from_slice(&i.to_le_bytes());
            }
            unsafe { bt_region.copy_from_host(&bt_host)?; }
            let cl_region = arena.region(
                "qwen35_fattn_context_lens_persistent", 4, 4)?;
            // Zero-init; first step overwrites with the real value.
            zero_region(cl_region.device_ptr(), 4)?;

            let kv_cache = Qwen35KvCache {
                layers: kv_layers,
                layer_idx_to_full_idx,
                max_pos: kv_max_pos,
                n_kv_heads,
                head_dim,
                per_layer_bytes: per_layer_kv_bytes,
                per_layer_scale_bytes,
                dtype: kv_dtype,
                block_tables_ptr: bt_region.device_ptr(),
                context_lens_ptr: cl_region.device_ptr(),
            };
            eprintln!(
                "[qwen35] KV cache: {} full-attn layers × \
                 (k+v) {} MiB/layer = {:.2} GiB total \
                 (max_pos={}, n_kv_heads={}, head_dim={})",
                kv_cache.layers.len(),
                per_layer_kv_bytes / (1024 * 1024),
                (kv_cache.layers.len() * per_layer_kv_bytes * 2) as f64
                    / (1024.0 * 1024.0 * 1024.0),
                kv_max_pos, n_kv_heads, head_dim,
            );

            // Linear-attn config (read once). Determines:
            //   - delta-rule SSM state shape  [num_v_heads, hkd, hvd]
            //   - causal-conv1d state shape   [ks-1, conv_dim]
            //   - in_proj_{qkv,z} / out_proj split  inside per-layer fwd
            let la_dims = qwen35_la_dims(&paths.model_dir);
            eprintln!(
                "[qwen35] linear-attn dims: kh={}/{} vh={}/{} \
                 conv_dim={} ks={} v_per_k={} value_dim={}",
                la_dims.num_k_heads, la_dims.head_k_dim,
                la_dims.num_v_heads, la_dims.head_v_dim,
                la_dims.conv_dim, la_dims.conv_kernel_dim,
                la_dims.v_per_k, la_dims.value_dim,
            );

            // Delta-rule SSM state: 48 layers × num_v_heads × hkd × hvd × 2.
            // Stored contiguously; per-layer offset =
            // linear_attn_rank × per_layer_bytes.
            let per_layer_state_bytes =
                la_dims.num_v_heads * la_dims.head_k_dim * la_dims.head_v_dim * 2;
            let mut layer_idx_to_linear_idx: Vec<Option<usize>> =
                vec![None; arch.base.num_hidden_layers];
            let mut n_linear_layers = 0usize;
            for (li, ty) in arch.base.layer_types.iter().enumerate() {
                if matches!(ty, rvllm_loader::LayerAttnType::Linear) {
                    layer_idx_to_linear_idx[li] = Some(n_linear_layers);
                    n_linear_layers += 1;
                }
            }
            let total_linear_state_bytes = n_linear_layers * per_layer_state_bytes;
            let linear_state_region = arena.region(
                "qwen35_linear_state", total_linear_state_bytes, 16)?;
            zero_region(linear_state_region.device_ptr(), total_linear_state_bytes)?;

            // Causal-conv1d state: (ks-1) timesteps of conv input
            // per linear-attn layer.
            //   per-layer = (ks-1) × conv_dim × 2 bytes
            //   total     = n_linear_layers × per-layer
            // Zero-initialised: the first call to `conv_state_advance`
            // walks zeros for the leading 3 timesteps, matching a
            // session-fresh state. Updated in-place per forward step.
            let conv_state_per_layer_bytes =
                (la_dims.conv_kernel_dim.saturating_sub(1))
                    * la_dims.conv_dim * 2;
            let conv_state_total_bytes =
                n_linear_layers * conv_state_per_layer_bytes;
            let conv_state_region = arena.region(
                "qwen35_conv_state", conv_state_total_bytes.max(1), 16)?;
            if conv_state_total_bytes > 0 {
                zero_region(conv_state_region.device_ptr(), conv_state_total_bytes)?;
            }
            let linear_state = Qwen35LinearState {
                base_ptr: linear_state_region.device_ptr(),
                per_layer_bytes: per_layer_state_bytes,
                n_linear_layers,
                num_ssm_heads: la_dims.num_v_heads,
                d_state: la_dims.head_v_dim,
                layer_idx_to_linear_idx,
                conv_state_base_ptr: conv_state_region.device_ptr(),
                conv_state_per_layer_bytes,
            };
            eprintln!(
                "[qwen35] linear-attn state: {} layers × \
                 {} MiB delta + {} KiB conv per layer = \
                 {:.2} GiB delta + {:.2} MiB conv (zero-init)",
                n_linear_layers,
                per_layer_state_bytes / (1024 * 1024),
                conv_state_per_layer_bytes / 1024,
                total_linear_state_bytes as f64
                    / (1024.0 * 1024.0 * 1024.0),
                conv_state_total_bytes as f64 / (1024.0 * 1024.0),
            );

            // Per-decode-token scratch. All buffers F16 except logits
            // (F32) and token_in/out (i32). Sized for M=1 decode.
            let hidden = arch.base.hidden_size;
            let intermediate = arch.base.intermediate_size;
            let n_q_heads = arch.base.num_attention_heads;
            let vocab = arch.base.vocab_size;
            let f16_bytes = |n: usize| n * 2;
            let scratch = Qwen35Scratch {
                token_in_ptr: arena.region("qwen35_token_in", 4, 4)?.device_ptr(),
                h_residual_ptr: arena.region(
                    "qwen35_h_residual", f16_bytes(hidden), 16)?.device_ptr(),
                h_work_ptr: arena.region(
                    "qwen35_h_work", f16_bytes(hidden), 16)?.device_ptr(),
                q_out_ptr: arena.region(
                    "qwen35_q_out", f16_bytes(n_q_heads * head_dim), 16)?.device_ptr(),
                q_gate_ptr: arena.region(
                    "qwen35_q_gate", f16_bytes(n_q_heads * head_dim), 16)?.device_ptr(),
                k_out_ptr: arena.region(
                    "qwen35_k_out", f16_bytes(n_kv_heads * head_dim), 16)?.device_ptr(),
                v_out_ptr: arena.region(
                    "qwen35_v_out", f16_bytes(n_kv_heads * head_dim), 16)?.device_ptr(),
                attn_out_ptr: arena.region(
                    "qwen35_attn_out", f16_bytes(n_q_heads * head_dim), 16)?.device_ptr(),
                o_out_ptr: arena.region(
                    "qwen35_o_out", f16_bytes(hidden), 16)?.device_ptr(),
                gate_out_ptr: arena.region(
                    "qwen35_gate_out", f16_bytes(intermediate), 16)?.device_ptr(),
                up_out_ptr: arena.region(
                    "qwen35_up_out", f16_bytes(intermediate), 16)?.device_ptr(),
                silu_mid_ptr: arena.region(
                    "qwen35_silu_mid", f16_bytes(intermediate), 16)?.device_ptr(),
                down_out_ptr: arena.region(
                    "qwen35_down_out", f16_bytes(hidden), 16)?.device_ptr(),
                logits_ptr: arena.region(
                    "qwen35_logits", vocab * 4, 16)?.device_ptr(),
                token_out_ptr: arena.region("qwen35_token_out", 4, 4)?.device_ptr(),
                silu_mid_fp8_ptr: arena.region(
                    "qwen35_silu_mid_fp8", intermediate, 16)?.device_ptr(),
                silu_mid_scales_ptr: arena.region(
                    "qwen35_silu_mid_scales", (intermediate / 128) * 4, 16)?.device_ptr(),
                // NVFP4 Q-side scratch: only allocated when the KV
                // layout is NVFP4. The NVFP4 RoPE kernel writes
                // FP8-E4M3 Q + per-(token, head) f32 scale; decode
                // is num_tokens=1, so q_scale_cache is [n_q_heads]
                // f32 and q_fp8 is [n_q_heads * head_dim] u8.
                q_fp8_ptr: match kv_dtype {
                    Qwen35KvDtype::Nvfp4 => arena.region(
                        "qwen35_q_fp8",
                        n_q_heads * head_dim, 16)?.device_ptr(),
                    Qwen35KvDtype::F16 => 0,
                },
                q_scale_cache_ptr: match kv_dtype {
                    Qwen35KvDtype::Nvfp4 => arena.region(
                        "qwen35_q_scale_cache",
                        n_q_heads * 4, 16)?.device_ptr(),
                    Qwen35KvDtype::F16 => 0,
                },
                scratch_bytes:
                    f16_bytes(hidden) * 4 +  // h_residual, h_work, o_out, down_out
                    f16_bytes(n_q_heads * head_dim) * 3 +  // q_out, q_gate, attn_out
                    f16_bytes(n_kv_heads * head_dim) * 2 +  // k_out, v_out
                    f16_bytes(intermediate) * 3 +  // gate, up, silu
                    vocab * 4 + 8,  // logits + tokens
            };
            eprintln!(
                "[qwen35] decode scratch allocated: {:.2} MiB \
                 (M=1, hidden={}, intermediate={}, vocab={})",
                scratch.scratch_bytes as f64 / (1024.0 * 1024.0),
                hidden, intermediate, vocab,
            );

            // ── Phase 2c-A: outside-path kernels + cuBLASLt ──
            // Loads the four kernels needed to drive the
            // embed→final-norm→lm_head→argmax pipe. Layer
            // kernels (RoPE, FA-2, FP8 GEMV, linear-attn, etc.)
            // land in Phase 2c-B.
            let manifest_path = paths.kernels_dir.join("sm_121/manifest.json");
            let manifest = rvllm_kernels::manifest::KernelManifest::load_and_verify(
                &manifest_path)?;
            let kernels = Arc::new(KernelLoader::new(manifest));
            // Outside path (Phase 2c-A).
            let embedding_gather_f16_mod = kernels.load_ptx("embedding_gather_f16")?;
            let fn_embedding_gather_f16 =
                embedding_gather_f16_mod.get_function("embedding_gather_f16_kernel")?;
            let fused_rmsnorm_fp8_quant_mod =
                kernels.load_ptx("fused_rmsnorm_fp8_quant")?;
            let fn_fused_rmsnorm_fp8_quant = fused_rmsnorm_fp8_quant_mod
                .get_function("fused_rmsnorm_fp8_quant_kernel")?;
            let argmax_mod = kernels.load_ptx("argmax")?;
            let fn_argmax_f16 = argmax_mod.get_function("argmax_f16_kernel")?;
            let fn_sample_topk_topp_f16 =
                argmax_mod.get_function("sample_topk_topp_f16_kernel")?;

            // Per-layer (Phase 2c-B) — RMSNorm + dense MLP.
            let rmsnorm_inplace_f16_mod = kernels.load_ptx("rmsnorm_inplace_f16")?;
            let fn_rmsnorm_inplace_f16 = rmsnorm_inplace_f16_mod
                .get_function("rmsnorm_inplace_f16_kernel")?;
            let fp8_gemv_dual_silu_mod = kernels.load_ptx(
                "fp8_gemv_blockwise_wpr_native_f16in_dual_silu")?;
            let fn_fp8_gemv_blockwise_wpr_native_f16in_dual_silu =
                fp8_gemv_dual_silu_mod.get_function(
                    "fp8_gemv_blockwise_wpr_native_f16in_dual_silu_kernel")?;
            let fp8_quantize_per_token_f16_mod =
                kernels.load_ptx("fp8_quantize_per_token_f16")?;
            let fn_fp8_quantize_per_token_f16 = fp8_quantize_per_token_f16_mod
                .get_function("fp8_quantize_per_token_f16_kernel")?;
            let fp8_quantize_per_token_amax_f16_mod =
                kernels.load_ptx("fp8_quantize_per_token_amax_f16")?;
            let fn_fp8_quantize_per_token_amax_f16 = fp8_quantize_per_token_amax_f16_mod
                .get_function("fp8_quantize_per_token_amax_f16_kernel")?;
            let silu_mul_f16_mod = kernels.load_ptx("silu_mul_f16")?;
            let fn_silu_mul_f16 = silu_mul_f16_mod.get_function("silu_mul_f16_kernel")?;
            // Single-output FP8 GEMV (gemma4-launcher Fp8GemvF16InLaunch
            // entry point). Used as the M=1 down_proj fallback when
            // cuBLASLt blockwise FP8 has no sm_121 algo.
            let fp8_gemv_mod = kernels.load_ptx(rvllm_kernels::FP8_GEMV_PTX_STEM)?;
            let fn_fp8_gemv_wpr_native_f16in = {
                match compile_target {
                    Some(t) if rvllm_kernels::Fp8GemvVariant::WprNativeF16In
                        .available_for(t) =>
                    {
                        Some(fp8_gemv_mod.get_function(
                            rvllm_kernels::Fp8GemvVariant::WprNativeF16In
                                .entry_point())?)
                    }
                    _ => None,
                }
            };

            // Per-layer (Phase 2c-B) — full-attn.
            let split_q_gate_f16_mod = kernels.load_ptx("split_q_gate_f16")?;
            let fn_split_q_gate_f16 = split_q_gate_f16_mod
                .get_function("split_q_gate_f16_kernel")?;
            let fused_rope_qwen_partial_f16kv_mod =
                kernels.load_ptx("fused_rope_qwen_partial_f16kv")?;
            let fn_fused_rope_qwen_partial_f16kv =
                fused_rope_qwen_partial_f16kv_mod
                    .get_function("fused_rope_qwen_partial_f16kv_kernel")?;
            // Step 3 (Qwen 3.6 27B NVFP4 plumbing). Resident-only.
            // Loaded under the same env gate that drives the KV
            // allocator above so the resident set matches the KV
            // layout. Dispatch lands in steps 4-5.
            let (fused_rope_qwen_partial_nvfp4kv_mod,
                 fn_fused_rope_qwen_partial_nvfp4kv) = match kv_dtype {
                Qwen35KvDtype::Nvfp4 => {
                    let m = kernels.load_ptx(
                        "fused_rope_qwen_partial_nvfp4kv")?;
                    let f = m.get_function(
                        "fused_rope_qwen_partial_nvfp4kv_kernel")?;
                    (Some(m), Some(f))
                }
                Qwen35KvDtype::F16 => (None, None),
            };
            let (flash_attention_nvfp4kv_mod,
                 fn_flash_attention_2_decode_nvfp4kv) = match kv_dtype {
                Qwen35KvDtype::Nvfp4 => {
                    let m = kernels.load_ptx("flash_attention_nvfp4kv")?;
                    let f = m.get_function(
                        "flash_attention_2_decode_nvfp4kv_kernel")?;
                    (Some(m), Some(f))
                }
                Qwen35KvDtype::F16 => (None, None),
            };
            let flash_attention_mod = kernels.load_ptx("flash_attention")?;
            let fn_flash_attention_2_decode_f16io = flash_attention_mod
                .get_function("flash_attention_2_decode_f16io_kernel")?;
            let fn_flash_attention_2_f16kv = flash_attention_mod
                .get_function("flash_attention_2_f16kv_kernel")?;
            let sigmoid_mul_f16_mod = kernels.load_ptx("sigmoid_mul_f16")?;
            let fn_sigmoid_mul_f16 = sigmoid_mul_f16_mod
                .get_function("sigmoid_mul_f16_kernel")?;

            // Per-layer (Phase 2c-B) — linear-attn.
            let conv_state_advance_f16_mod =
                kernels.load_ptx("conv_state_advance_f16")?;
            let fn_conv_state_advance_f16 = conv_state_advance_f16_mod
                .get_function("conv_state_advance_f16_kernel")?;
            let causal_conv1d_f16_mod = kernels.load_ptx("causal_conv1d_f16")?;
            let fn_causal_conv1d_f16 = causal_conv1d_f16_mod
                .get_function("causal_conv1d_f16_kernel")?;
            let qwen_linear_alpha_beta_f16_mod =
                kernels.load_ptx("qwen_linear_alpha_beta_f16")?;
            let fn_qwen_linear_alpha_beta_f16 = qwen_linear_alpha_beta_f16_mod
                .get_function("qwen_linear_alpha_beta_f16_kernel")?;
            let qwen_linear_silu_l2_gqa_f16_mod =
                kernels.load_ptx("qwen_linear_silu_l2_gqa_f16")?;
            let fn_qwen_linear_silu_l2_gqa_f16 = qwen_linear_silu_l2_gqa_f16_mod
                .get_function("qwen_linear_silu_l2_gqa_f16_kernel")?;
            let gated_delta_rule_decode_f16_mod =
                kernels.load_ptx("gated_delta_rule_decode_f16")?;
            let fn_gated_delta_rule_decode_f16 = gated_delta_rule_decode_f16_mod
                .get_function("gated_delta_rule_decode_f16_kernel")?;
            let qwen_linear_rmsnorm_gated_f16_mod =
                kernels.load_ptx("qwen_linear_rmsnorm_gated_f16")?;
            let fn_qwen_linear_rmsnorm_gated_f16 = qwen_linear_rmsnorm_gated_f16_mod
                .get_function("qwen_linear_rmsnorm_gated_f16_kernel")?;
            // Phase #1-b: batched-prefill linear-attn support
            // (ported from qwen36 — same PTX names).
            let conv_state_advance_batched_f16_mod =
                kernels.load_ptx("conv_state_advance_batched_f16")?;
            let fn_conv_state_advance_batched_f16 = conv_state_advance_batched_f16_mod
                .get_function("conv_state_advance_batched_f16_kernel")?;
            let gated_delta_rule_prefill_f16_mod =
                kernels.load_ptx("gated_delta_rule_prefill_f16")?;
            let fn_gated_delta_rule_prefill_f16 = gated_delta_rule_prefill_f16_mod
                .get_function("gated_delta_rule_prefill_f16_kernel")?;

            // Vector residual.
            let f16_plus_f32_inplace_f16_mod =
                kernels.load_ptx("f16_plus_f32_inplace_f16")?;
            let fn_f16_plus_f32_inplace_f16 = f16_plus_f32_inplace_f16_mod
                .get_function("f16_plus_f32_inplace_f16_kernel")?;
            let vector_add_f16_mod = kernels.load_ptx("vector_add_f16")?;
            let fn_vector_add_f16 = vector_add_f16_mod
                .get_function("vector_add_f16_kernel")?;
            let fp8_gemv_f16in_residual_add_mod =
                kernels.load_ptx("fp8_gemv_f16in_residual_add")?;
            let fn_fp8_gemv_f16in_residual_add =
                fp8_gemv_f16in_residual_add_mod.get_function(
                    "fp8_gemv_blockwise_wpr_native_f16in_residual_add_kernel")?;

            // ── Vision kernels (Phase 3-a-v) ────────────────────
            let layernorm_inplace_f16_mod =
                kernels.load_ptx("layernorm_inplace_f16")?;
            let fn_layernorm_inplace_f16 = layernorm_inplace_f16_mod
                .get_function("layernorm_inplace_f16_kernel")?;
            let gelu_tanh_f16_mod = kernels.load_ptx("gelu_tanh_f16")?;
            let fn_gelu_tanh_f16 = gelu_tanh_f16_mod
                .get_function("gelu_tanh_f16_kernel")?;
            let softmax_row_f16_mod = kernels.load_ptx("softmax_row_f16")?;
            let fn_softmax_row_f16 = softmax_row_f16_mod
                .get_function("softmax_row_f16_kernel")?;
            let vit_rotary_2d_f16_mod = kernels.load_ptx("vit_rotary_2d_f16")?;
            let fn_vit_rotary_2d_f16 = vit_rotary_2d_f16_mod
                .get_function("vit_rotary_2d_f16_kernel")?;
            let vit_pos_embed_interp_f16_mod =
                kernels.load_ptx("vit_pos_embed_interp_f16")?;
            let fn_vit_pos_embed_interp_f16 = vit_pos_embed_interp_f16_mod
                .get_function("vit_pos_embed_interp_f16_kernel")?;
            let scale_inplace_f16_mod = kernels.load_ptx("scale_inplace_f16")?;
            let fn_scale_inplace_f16 = scale_inplace_f16_mod
                .get_function("scale_inplace_f16_kernel")?;
            let transpose_2d_f16_mod = kernels.load_ptx("transpose_2d_f16")?;
            let fn_transpose_2d_f16 = transpose_2d_f16_mod
                .get_function("transpose_2d_f16_kernel")?;
            let add_bias_f16_mod = kernels.load_ptx("add_bias_f16")?;
            let fn_add_bias_f16 = add_bias_f16_mod
                .get_function("add_bias_f16_kernel")?;
            let cast_fp_mod = kernels.load_ptx("cast_fp")?;
            let fn_cast_f32_to_f16 = cast_fp_mod
                .get_function("cast_f32_to_f16_kernel")?;
            let fn_cast_f16_to_f32 = cast_fp_mod
                .get_function("cast_f16_to_f32_kernel")?;
            let extract_head_f16_mod = kernels.load_ptx("extract_head_f16")?;
            let fn_extract_head_f16 = extract_head_f16_mod
                .get_function("extract_head_f16_kernel")?;
            let fn_scatter_head_f16 = extract_head_f16_mod
                .get_function("scatter_head_f16_kernel")?;
            let softmax_row_f32_to_f16_mod =
                kernels.load_ptx("softmax_row_f32_to_f16")?;
            let fn_softmax_row_f32_to_f16 = softmax_row_f32_to_f16_mod
                .get_function("softmax_row_f32_to_f16_kernel")?;
            let transpose_heads_v_f16_mod =
                kernels.load_ptx("transpose_heads_v_f16")?;
            let fn_transpose_heads_v_f16 = transpose_heads_v_f16_mod
                .get_function("transpose_heads_v_f16_kernel")?;
            let scatter_heads_f16_mod =
                kernels.load_ptx("scatter_heads_f16")?;
            let fn_scatter_heads_f16 = scatter_heads_f16_mod
                .get_function("scatter_heads_f16_kernel")?;
            let scale_inplace_f32_mod = kernels.load_ptx("scale_inplace_f32")?;
            let fn_scale_inplace_f32 = scale_inplace_f32_mod
                .get_function("scale_inplace_f32_kernel")?;

            let outside_kernels = Qwen35OutsideKernels {
                embedding_gather_f16_mod,
                fn_embedding_gather_f16,
                fused_rmsnorm_fp8_quant_mod,
                fn_fused_rmsnorm_fp8_quant,
                argmax_mod,
                fn_argmax_f16,
                fn_sample_topk_topp_f16,
                rmsnorm_inplace_f16_mod,
                fn_rmsnorm_inplace_f16,
                fp8_gemv_dual_silu_mod,
                fn_fp8_gemv_blockwise_wpr_native_f16in_dual_silu,
                fp8_quantize_per_token_f16_mod,
                fn_fp8_quantize_per_token_f16,
                fp8_quantize_per_token_amax_f16_mod,
                fn_fp8_quantize_per_token_amax_f16,
                silu_mul_f16_mod,
                fn_silu_mul_f16,
                fp8_gemv_mod,
                fn_fp8_gemv_wpr_native_f16in,
                split_q_gate_f16_mod,
                fn_split_q_gate_f16,
                fused_rope_qwen_partial_f16kv_mod,
                fn_fused_rope_qwen_partial_f16kv,
                fused_rope_qwen_partial_nvfp4kv_mod,
                fn_fused_rope_qwen_partial_nvfp4kv,
                flash_attention_nvfp4kv_mod,
                fn_flash_attention_2_decode_nvfp4kv,
                flash_attention_mod,
                fn_flash_attention_2_decode_f16io,
                sigmoid_mul_f16_mod,
                fn_sigmoid_mul_f16,
                conv_state_advance_f16_mod,
                fn_conv_state_advance_f16,
                causal_conv1d_f16_mod,
                fn_causal_conv1d_f16,
                qwen_linear_alpha_beta_f16_mod,
                fn_qwen_linear_alpha_beta_f16,
                qwen_linear_silu_l2_gqa_f16_mod,
                fn_qwen_linear_silu_l2_gqa_f16,
                gated_delta_rule_decode_f16_mod,
                fn_gated_delta_rule_decode_f16,
                conv_state_advance_batched_f16_mod,
                fn_conv_state_advance_batched_f16,
                gated_delta_rule_prefill_f16_mod,
                fn_gated_delta_rule_prefill_f16,
                fn_flash_attention_2_f16kv,
                fn_cast_f16_to_f32,
                qwen_linear_rmsnorm_gated_f16_mod,
                fn_qwen_linear_rmsnorm_gated_f16,
                f16_plus_f32_inplace_f16_mod,
                fn_f16_plus_f32_inplace_f16,
                vector_add_f16_mod,
                fn_vector_add_f16,
                fp8_gemv_f16in_residual_add_mod,
                fn_fp8_gemv_f16in_residual_add,
                layernorm_inplace_f16_mod,
                fn_layernorm_inplace_f16,
                gelu_tanh_f16_mod,
                fn_gelu_tanh_f16,
                softmax_row_f16_mod,
                fn_softmax_row_f16,
                vit_rotary_2d_f16_mod,
                fn_vit_rotary_2d_f16,
                vit_pos_embed_interp_f16_mod,
                fn_vit_pos_embed_interp_f16,
                scale_inplace_f16_mod,
                fn_scale_inplace_f16,
                transpose_2d_f16_mod,
                fn_transpose_2d_f16,
                add_bias_f16_mod,
                fn_add_bias_f16,
                cast_fp_mod,
                fn_cast_f32_to_f16,
                extract_head_f16_mod,
                fn_extract_head_f16,
                fn_scatter_head_f16,
                softmax_row_f32_to_f16_mod,
                fn_softmax_row_f32_to_f16,
                transpose_heads_v_f16_mod,
                fn_transpose_heads_v_f16,
                scatter_heads_f16_mod,
                fn_scatter_heads_f16,
                scale_inplace_f32_mod,
                fn_scale_inplace_f32,
            };
            let attn_backend_full = rvllm_attention::AttentionBackend::Fa2Ptx(
                rvllm_attention::Fa2PtxKernels::load(
                    &*kernels,
                    arch.base.head_dim as u32,
                )?,
            );

            // cuBLASLt for the FP8 lm_head matmul. 32 MiB workspace
            // (matches Qwen 3.6 / Gemma 4 sizing).
            let cublaslt_ws_bytes: usize = 32 * 1024 * 1024;
            let cublaslt_ws_region = arena.region(
                "qwen35_cublaslt_ws", cublaslt_ws_bytes, 256)?;
            let cublaslt = CublasLt::new(
                cublaslt_ws_region.device_ptr(), cublaslt_ws_bytes)?;
            let cutlass = CutlassBackend::load_for(
                compile_target,
                paths.cutlass_so.clone(),
                &[],
            )?;
            eprintln!(
                "[qwen35] cuBLASLt initialised with {} MiB workspace; \
                 outside + per-layer kernel set loaded (embed, rmsnorm \
                 ×3 variants, fp8_gemv_dual_silu, fp8_quantize_per_token, \
                 split_q_gate, fused_rope_qwen_partial, flash_attention, \
                 sigmoid_mul, conv_state_advance, causal_conv1d, \
                 qwen_linear_alpha_beta / silu_l2_gqa / rmsnorm_gated, \
                 gated_delta_rule_decode, residual_add, argmax).",
                cublaslt_ws_bytes / (1024 * 1024),
            );

            eprintln!(
                "[qwen35] Phase 2c-A complete: arch + outside + {} layers \
                 + KV cache + linear-attn state + RoPE tables + decode \
                 scratch + outside kernels + cuBLASLt. arena.used={:.2} \
                 GiB. Forward path = embed→final-norm→lm_head→argmax \
                 (transformer layers SKIPPED — Phase 2c-B is the layer \
                 dispatch). See v3/QWEN35_BRINGUP_PLAN.md.",
                arch.base.num_hidden_layers,
                arena.used() as f64 / (1024.0 * 1024.0 * 1024.0),
            );

            return Ok(Self {
                paths, arch, arena_bytes,
                ctx: Some(ctx),
                arena: Some(arena),
                stream: Some(stream),
                model: Some(model),
                kv_cache: Some(kv_cache),
                linear_state: Some(linear_state),
                rope_tables: Some(rope_tables),
                scratch: Some(scratch),
                outside_kernels: Some(outside_kernels),
                attn_backend_full: Some(attn_backend_full),
                cublaslt: Some(cublaslt),
                cutlass,
                la_dims: Some(la_dims),
                sampling: Qwen35SamplingState::default(),
            });
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!(
                "[qwen35] Phase 0 ONLY (no-cuda build): arch validated, \
                 NO weight upload. See v3/QWEN35_BRINGUP_PLAN.md."
            );
            Ok(Self { paths, arch, arena_bytes, sampling: Qwen35SamplingState::default() })
        }
    }
}

#[cfg(feature = "cuda")]
impl Qwen35Bringup {
    #[allow(clippy::too_many_arguments)]
    unsafe fn qwen35_fp8_cutlass_blockscale_sm120(
        &self,
        ker: &Qwen35OutsideKernels,
        lib: &rvllm_cutlass::lib_so::CutlassSm120Lib,
        out_f16: u64,
        weight_fp8: u64,
        weight_blockscale: u64,
        input_f16: u64,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
        scratch_prefix: &'static str,
    ) -> Result<()> {
        if weight_blockscale == 0 {
            return Err(rvllm_core::RvllmError::cuda(
                "qwen35 CUTLASS FP8 path requires blockscale weights",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        // CUTLASS SM120 blockscale FP8 requires M >= 128 (MmaTileShape_M
        // hard-coded to 128). For M < 128 we PAD: allocate a 128-row
        // intermediate, zero-fill the padding rows, run the kernel at
        // M_padded = 128, then DtoD-copy the first `m` rows of the
        // output back to the caller's buffer. The wasted compute is
        // `(128 - m) / 128` of the GEMM (max 99% at m=1, 30% at m=89,
        // 0% at m=128) — but the per-row cost on the CUTLASS path is
        // dramatically lower than the per-token GEMV fallback, so the
        // crossover is well below m=128 in practice.
        //
        // M=128 minimum is the only constraint relaxed here. K=128
        // tile + scale-block alignment + cooperative-kernel schedule
        // limits stay untouched.
        const M_TILE: u32 = 128;
        let m_pad = m.div_ceil(M_TILE).max(1) * M_TILE;
        let need_pad = m_pad != m;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "qwen35_fp8_cutlass_blockscale_sm120: arena absent".into()))?;

        // Padded input buffer (zero rows for [m..m_pad)). When m==m_pad
        // we use the caller's input_f16 directly — no extra copy.
        let input_used_ptr: u64 = if need_pad {
            let pad_bytes = (m_pad as usize) * (k as usize) * 2;
            let real_bytes = (m as usize) * (k as usize) * 2;
            let pad_in = arena.region(
                "qwen35_cutlass_pad_in_f16", pad_bytes, 16)?;
            use cudarc::driver::sys::*;
            // Zero entire padded region first (including padding tail).
            let rc = cuMemsetD8Async(
                pad_in.device_ptr(), 0, pad_bytes, stream as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 CUTLASS FP8 path: pad input zero",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
            // Copy real rows on top of the zeros.
            let rc = cuMemcpyDtoDAsync_v2(
                pad_in.device_ptr(),
                input_f16,
                real_bytes,
                stream as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 CUTLASS FP8 path: pad input copy",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
            pad_in.device_ptr()
        } else {
            input_f16
        };

        // Padded output buffer — same pattern. CUTLASS writes m_pad rows;
        // we copy the first m back to the caller's out_f16.
        let out_used_ptr: u64 = if need_pad {
            let pad_out_bytes = (m_pad as usize) * (n as usize) * 2;
            let pad_out = arena.region(
                "qwen35_cutlass_pad_out_f16", pad_out_bytes, 16)?;
            pad_out.device_ptr()
        } else {
            out_f16
        };

        let fp8_bytes = (m_pad as usize) * (k as usize);
        let amax_bytes = (m_pad as usize) * 4;
        let in_fp8 = arena.region(scratch_prefix, fp8_bytes, 16)?;
        let in_amax = arena.region("qwen35_cutlass_in_amax", amax_bytes, 16)?;
        {
            use cudarc::driver::sys::*;
            let block_dim: u32 = k.min(1024);
            let mut o_fp8 = in_fp8.device_ptr();
            let mut o_amax = in_amax.device_ptr();
            let mut i_ptr = input_used_ptr;
            let mut k_i: i32 = k as i32;
            let args = [
                (&mut o_fp8) as *mut u64 as *mut core::ffi::c_void,
                (&mut o_amax) as *mut u64 as *mut core::ffi::c_void,
                (&mut i_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_fp8_quantize_per_token_amax_f16.raw() as CUfunction,
                m_pad, 1, 1, block_dim, 1, 1, 0,
                stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 CUTLASS FP8 path: amax quantize launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let sfa_n = lib.sfa_bytes(m_pad as i32, k as i32);
        let sfb_n = lib.sfb_bytes(n as i32, k as i32);
        let ws_n = lib.workspace_size(m_pad as i32, n as i32, k as i32);
        if sfa_n == 0 || sfb_n == 0 {
            return Err(rvllm_core::RvllmError::cuda(
                "qwen35 CUTLASS FP8 path: SM120 prep helpers unavailable",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        let sfa = arena.region("qwen35_cutlass_sfa", sfa_n.max(4), 16)?;
        let sfb = arena.region("qwen35_cutlass_sfb", sfb_n.max(4), 16)?;
        let ws = arena.region("qwen35_cutlass_ws", ws_n.max(16), 256)?;
        lib.launch_prep_sfa(
            in_amax.device_ptr(),
            sfa.device_ptr(),
            m_pad as i32,
            k as i32,
            stream,
        )?;
        lib.launch_prep_sfb(
            weight_blockscale,
            sfb.device_ptr(),
            n as i32,
            k as i32,
            stream,
        )?;
        lib.launch_fp8_gemm_blockscale(
            out_used_ptr,
            in_fp8.device_ptr(),
            weight_fp8,
            sfa.device_ptr(),
            sfb.device_ptr(),
            m_pad as i32,
            n as i32,
            k as i32,
            ws.device_ptr(),
            ws_n,
            stream,
        )?;

        // Copy first `m` rows of padded output back to caller's buffer.
        if need_pad {
            let real_out_bytes = (m as usize) * (n as usize) * 2;
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                out_f16,
                out_used_ptr,
                real_out_bytes,
                stream as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 CUTLASS FP8 path: pad output copy back",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        Ok(())
    }

    /// Phase 2c-A: outside-only smoke forward. Drives the
    /// embed → final-RMSNorm-with-FP8-quant → cuBLASLt fp8_gemm
    /// lm_head → argmax_f16 pipe end-to-end. ALL 64 TRANSFORMER
    /// LAYERS ARE SKIPPED — the output token id is structurally
    /// valid (HtoD → forward → DtoH all work) but semantically
    /// meaningless until Phase 2c-B wires the per-layer forward.
    ///
    /// Useful right now as a kernel-load + scratch + cuBLASLt
    /// integration smoke. Mirrors the qwen36
    /// `forward_outside_only` pattern.
    pub unsafe fn forward_outside_only_smoke(&self, token_id: u32) -> Result<u32> {
        let arch = &self.arch;
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "Qwen35Bringup::forward_outside_only_smoke: model not loaded".into(),
        ))?;
        let scr = self.scratch.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "Qwen35Bringup::forward_outside_only_smoke: scratch absent".into(),
        ))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "Qwen35Bringup::forward_outside_only_smoke: outside_kernels absent".into(),
        ))?;
        let cublaslt = self.cublaslt.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "Qwen35Bringup::forward_outside_only_smoke: cublaslt absent".into(),
        ))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "Qwen35Bringup::forward_outside_only_smoke: stream absent".into(),
        ))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "Qwen35Bringup::forward_outside_only_smoke: arena absent".into(),
        ))?;
        let hidden = arch.base.hidden_size as u32;
        let vocab = arch.base.vocab_size as u32;
        let eps = arch.base.rms_norm_eps;
        let stream_raw = stream.raw() as u64;

        // (1) HtoD token id.
        let token_bytes = (token_id as i32).to_le_bytes();
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyHtoDAsync_v2(
                scr.token_in_ptr as CUdeviceptr,
                token_bytes.as_ptr() as *const _,
                4,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 HtoD token",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // (2) Embedding gather → h_residual.
        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1, hidden, vocab,
            }.launch(
                ker.fn_embedding_gather_f16,
                scr.h_residual_ptr,
                model.outside.embed_tokens.offset_bytes,
                scr.token_in_ptr,
                stream_raw,
            )?;
        }

        // (3) Final RMSNorm + per-token FP8 quantize on the (unmodified)
        // residual. Output: fp8 hidden + per-token f32 scale ready for
        // cuBLASLt fp8_gemm.
        let hidden_fp8_region = arena.region(
            "qwen35_smoke_hidden_fp8", hidden as usize, 16)?;
        let hidden_scale_region = arena.region(
            "qwen35_smoke_hidden_scale", 4, 4)?;
        unsafe {
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: 1, hidden, eps,
            }.launch(
                ker.fn_fused_rmsnorm_fp8_quant,
                hidden_fp8_region.device_ptr(),
                hidden_scale_region.device_ptr(),
                scr.h_residual_ptr,
                model.outside.final_norm.offset_bytes,
                stream_raw,
            )?;
        }

        // (4) cuBLASLt FP8 GEMM: [1, hidden] · [vocab, hidden]^T → [1, vocab].
        unsafe {
            cublaslt.fp8_gemm(
                hidden_fp8_region.device_ptr(),
                model.outside.lm_head_fp8.offset_bytes,
                scr.logits_ptr,
                1, vocab as i32, hidden as i32,
                hidden_scale_region.device_ptr(),
                model.outside.lm_head_fp8.scale_ptr,
                stream_raw,
            )?;
        }

        // (5) argmax_f16 — but wait, fp8_gemm writes f16 logits NOT
        // f32. Re-use the f32 logits_ptr slot for the f16 output
        // (it's pre-allocated as `vocab * 4` bytes which is larger
        // than `vocab * 2` so the f16 fits).
        // argmax_f16_kernel signature: (logits_ptr, out_token_ptr, vsz).
        unsafe {
            use cudarc::driver::sys::*;
            let block_dim: u32 = vocab.min(1024);
            let mut row_ptr = scr.logits_ptr;
            let mut out_ptr = scr.token_out_ptr;
            let mut vsz: i32 = vocab as i32;
            let args = [
                (&mut row_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut vsz) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_argmax_f16.raw() as CUfunction,
                1, 1, 1, block_dim, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 argmax_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        stream.fence()?;

        // (6) DtoH predicted token.
        let mut predicted: i32 = 0;
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                &mut predicted as *mut i32 as *mut _,
                scr.token_out_ptr as CUdeviceptr,
                4,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 DtoH predicted",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(predicted as u32)
    }
}

#[cfg(feature = "cuda")]
impl Qwen35Bringup {
    /// Phase 2c-B-A smoke forward: embed → dense_mlp(layer 0) →
    /// final RMSNorm + FP8 quant → cuBLASLt fp8_gemm → argmax.
    /// Applies ONE layer's dense MLP path on actual model weights.
    /// Attention block (linear or full) is still SKIPPED. Output
    /// token is closer to "real" than Phase 2c-A's pure outside
    /// pipe but still not a valid model forward.
    pub unsafe fn forward_one_dense_mlp_smoke(&self, token_id: u32) -> Result<u32> {
        // First-stage embed → h_residual is identical to the Phase
        // 2c-A path; reuse its inner steps by manually walking them.
        let scr = self.scratch.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_one_dense_mlp_smoke: scratch absent".into(),
        ))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_one_dense_mlp_smoke: outside_kernels absent".into(),
        ))?;
        let cublaslt = self.cublaslt.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_one_dense_mlp_smoke: cublaslt absent".into(),
        ))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_one_dense_mlp_smoke: stream absent".into(),
        ))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_one_dense_mlp_smoke: arena absent".into(),
        ))?;
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_one_dense_mlp_smoke: model absent".into(),
        ))?;
        let arch = &self.arch;
        let hidden = arch.base.hidden_size as u32;
        let vocab = arch.base.vocab_size as u32;
        let eps = arch.base.rms_norm_eps;
        let stream_raw = stream.raw() as u64;

        // (1) HtoD token id + embed lookup.
        let token_bytes = (token_id as i32).to_le_bytes();
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyHtoDAsync_v2(
                scr.token_in_ptr as CUdeviceptr,
                token_bytes.as_ptr() as *const _,
                4, stream_raw as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 mlp-smoke HtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1, hidden, vocab,
            }.launch(
                ker.fn_embedding_gather_f16,
                scr.h_residual_ptr,
                model.outside.embed_tokens.offset_bytes,
                scr.token_in_ptr,
                stream_raw,
            )?;
        }

        // (2) Apply layer 0's dense MLP block in place on h_residual.
        self.apply_dense_mlp_layer(0)?;

        // (3) Final RMSNorm + FP8 quantize → h_fp8 + per-token scale.
        let hidden_fp8_region = arena.region(
            "qwen35_mlp_hidden_fp8", hidden as usize, 16)?;
        let hidden_scale_region = arena.region(
            "qwen35_mlp_hidden_scale", 4, 4)?;
        unsafe {
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: 1, hidden, eps,
            }.launch(
                ker.fn_fused_rmsnorm_fp8_quant,
                hidden_fp8_region.device_ptr(),
                hidden_scale_region.device_ptr(),
                scr.h_residual_ptr,
                model.outside.final_norm.offset_bytes,
                stream_raw,
            )?;
        }

        // (4) cuBLASLt fp8_gemm: logits = h_fp8 · lm_head_fp8^T.
        unsafe {
            cublaslt.fp8_gemm(
                hidden_fp8_region.device_ptr(),
                model.outside.lm_head_fp8.offset_bytes,
                scr.logits_ptr,
                1, vocab as i32, hidden as i32,
                hidden_scale_region.device_ptr(),
                model.outside.lm_head_fp8.scale_ptr,
                stream_raw,
            )?;
        }

        // (5) argmax_f16 over logits row.
        unsafe {
            use cudarc::driver::sys::*;
            let block_dim: u32 = vocab.min(1024);
            let mut row_ptr = scr.logits_ptr;
            let mut out_ptr = scr.token_out_ptr;
            let mut vsz: i32 = vocab as i32;
            let args = [
                (&mut row_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut vsz) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_argmax_f16.raw() as CUfunction,
                1, 1, 1, block_dim, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 mlp-smoke argmax",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        stream.fence()?;
        let mut predicted: i32 = 0;
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                &mut predicted as *mut i32 as *mut _,
                scr.token_out_ptr as CUdeviceptr, 4);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 mlp-smoke DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        Ok(predicted as u32)
    }

    /// Phase 2c-C-b smoke: embed → apply_linear_attn_layer(0) →
    /// final-norm → lm_head → argmax. Layer 0 is the first
    /// linear-attn slot in the 3:1 pattern. Drives the full
    /// 10-op Gated DeltaNet block against layer 0's real
    /// weights; full-attn + all other 63 layers are still
    /// SKIPPED.
    pub unsafe fn forward_layer0_linear_smoke(&self, token_id: u32) -> Result<u32> {
        let scr = self.scratch.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_layer0_linear_smoke: scratch absent".into()))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_layer0_linear_smoke: outside_kernels absent".into()))?;
        let cublaslt = self.cublaslt.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_layer0_linear_smoke: cublaslt absent".into()))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_layer0_linear_smoke: stream absent".into()))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_layer0_linear_smoke: arena absent".into()))?;
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_layer0_linear_smoke: model absent".into()))?;
        let arch = &self.arch;
        let hidden = arch.base.hidden_size as u32;
        let vocab = arch.base.vocab_size as u32;
        let eps = arch.base.rms_norm_eps;
        let stream_raw = stream.raw() as u64;

        // (1) HtoD token id + embed.
        let token_bytes = (token_id as i32).to_le_bytes();
        {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyHtoDAsync_v2(
                scr.token_in_ptr as CUdeviceptr,
                token_bytes.as_ptr() as *const _,
                4, stream_raw as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 la-smoke HtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        rvllm_fused::EmbeddingGatherLaunch {
            num_tokens: 1, hidden, vocab,
        }.launch(
            ker.fn_embedding_gather_f16,
            scr.h_residual_ptr,
            model.outside.embed_tokens.offset_bytes,
            scr.token_in_ptr,
            stream_raw,
        )?;

        // (2) Linear-attn forward on layer 0.
        self.apply_linear_attn_layer(0)?;

        // (3) Final RMSNorm + FP8 quant.
        let hidden_fp8_region = arena.region(
            "qwen35_la_hidden_fp8", hidden as usize, 16)?;
        let hidden_scale_region = arena.region(
            "qwen35_la_hidden_scale", 4, 4)?;
        rvllm_fused::FusedRmsnormFp8QuantLaunch {
            num_tokens: 1, hidden, eps,
        }.launch(
            ker.fn_fused_rmsnorm_fp8_quant,
            hidden_fp8_region.device_ptr(),
            hidden_scale_region.device_ptr(),
            scr.h_residual_ptr,
            model.outside.final_norm.offset_bytes,
            stream_raw,
        )?;

        // (4) cuBLASLt fp8_gemm: logits.
        cublaslt.fp8_gemm(
            hidden_fp8_region.device_ptr(),
            model.outside.lm_head_fp8.offset_bytes,
            scr.logits_ptr,
            1, vocab as i32, hidden as i32,
            hidden_scale_region.device_ptr(),
            model.outside.lm_head_fp8.scale_ptr,
            stream_raw,
        )?;

        // (5) argmax + DtoH.
        {
            use cudarc::driver::sys::*;
            let block_dim: u32 = vocab.min(1024);
            let mut row_ptr = scr.logits_ptr;
            let mut out_ptr = scr.token_out_ptr;
            let mut vsz: i32 = vocab as i32;
            let args = [
                (&mut row_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut vsz) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_argmax_f16.raw() as CUfunction,
                1, 1, 1, block_dim, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 la-smoke argmax",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        stream.fence()?;
        let mut predicted: i32 = 0;
        {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                &mut predicted as *mut i32 as *mut _,
                scr.token_out_ptr as CUdeviceptr, 4);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 la-smoke DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        Ok(predicted as u32)
    }

    /// Phase 2c-B-B-i: apply ONE full-attn layer's Q/K/V
    /// projections + `split_q_gate` in place. Does NOT run RoPE,
    /// KV-write, FA-2 decode, sigmoid_mul, or o_proj — those
    /// land in Phase 2c-B-B-ii/iii. The h_residual is RMSNormed
    /// into h_work and three FP8 GEMVs fire against the layer's
    /// q/k/v weights. q is split into q (kept in q_out_ptr) +
    /// gate (kept in q_gate_ptr) by `split_q_gate_f16`.
    ///
    /// Used by `forward_layer3_qkv_plus_mlp_smoke` to validate
    /// the full-attn projection path at the actual Qwen 3.5
    /// shape (Q=12288=2·24·256 with attn_output_gate, K/V=1024).
    pub unsafe fn apply_full_attn_qkv_only(&self, layer_idx: usize, position: u32) -> Result<()> {
        let arch = &self.arch;
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_qkv_only: model absent".into(),
        ))?;
        let scr = self.scratch.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_qkv_only: scratch absent".into(),
        ))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_qkv_only: outside_kernels absent".into(),
        ))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_qkv_only: stream absent".into(),
        ))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_qkv_only: arena absent".into(),
        ))?;
        let layer = model.layers.get(layer_idx).ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            format!("apply_full_attn_qkv_only: layer {layer_idx} out of range"),
        ))?;
        // Reject linear-attn slots — caller's bug.
        let full = match &layer.attn {
            rvllm_loader::qwen35_weights::Qwen35LayerAttn::Full(f) => f,
            rvllm_loader::qwen35_weights::Qwen35LayerAttn::Linear(_) => {
                return Err(corrupt(
                    self.paths.model_dir.clone(),
                    format!("apply_full_attn_qkv_only: layer {layer_idx} is \
                             linear-attention; this helper only handles \
                             full-attn slots"),
                ));
            }
        };

        let hidden = arch.base.hidden_size as i32;
        let head_dim = arch.base.head_dim as i32;
        let n_q_heads = arch.base.num_attention_heads as i32;
        let n_kv_heads = arch.base.num_key_value_heads as i32;
        let eps = arch.base.rms_norm_eps;
        let stream_raw = stream.raw() as u64;

        // (1) h_work ← rmsnorm(h_residual, input_layernorm). Same
        // kernel + sig as the dense-MLP path; uses input_layernorm
        // instead of post_attention_layernorm.
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                scr.h_work_ptr as CUdeviceptr,
                scr.h_residual_ptr as CUdeviceptr,
                (hidden as usize) * 2,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 fattn dtod h_work",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        unsafe {
            use cudarc::driver::sys::*;
            let mut hw_ptr = scr.h_work_ptr;
            let mut gamma_ptr = full.input_layernorm.offset_bytes;
            let mut eps_arg = eps;
            let mut hd = hidden;
            let args = [
                (&mut hw_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut gamma_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_arg) as *mut f32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_rmsnorm_inplace_f16.raw() as CUfunction,
                1, 1, 1, (hidden as u32).min(1024), 1, 1, 32 * 4,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 fattn input_ln launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (2) Q projection: [12288, 5120] FP8 GEMV → qg_interleaved
        // F16 [n_q_heads, 2*head_dim] = [24, 512] = 12288 elems
        // (24 KiB). Allocated inline; cheap.
        let fp8_gemv_fn = ker.fn_fp8_gemv_wpr_native_f16in.ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_qkv_only: fp8_gemv_wpr_native_f16in unavailable".into(),
        ))?;
        let qg_bytes = (2 * (n_q_heads as usize) * (head_dim as usize)) * 2;
        let qg_region = arena.region("qwen35_fattn_qg_interleaved", qg_bytes, 16)?;
        let qg_ptr = qg_region.device_ptr();
        let q_n = (2 * n_q_heads * head_dim) as u32; // 12288
        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m: 1, n: q_n, k: hidden as u32,
            }.launch(
                fp8_gemv_fn,
                qg_ptr,
                full.q_proj.offset_bytes,
                full.q_proj.blockscale_ptr.unwrap_or(0),
                scr.h_work_ptr,
                stream_raw,
            )?;
        }

        // (3) split_q_gate_f16: qg [n_heads, 2*hd] → q (q_out_ptr),
        // gate (q_gate_ptr). Grid (n_heads, num_tokens=1), block hd.
        unsafe {
            use cudarc::driver::sys::*;
            let mut q_out = scr.q_out_ptr;
            let mut g_out = scr.q_gate_ptr;
            let mut qg_in = qg_ptr;
            let mut nh = n_q_heads;
            let mut hd = head_dim;
            let args = [
                (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut g_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut qg_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_split_q_gate_f16.raw() as CUfunction,
                n_q_heads as u32, 1, 1, head_dim as u32, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 split_q_gate_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (3.5) q_norm: per-head RMSNorm on q. Grid (n_q_heads, 1, 1),
        // block (head_dim.min(1024), 1, 1). Kernel signature is the
        // same `(x_inout, gamma, eps, hidden)` 4-arg form as the
        // pre-attn input_layernorm above — `gamma` is `q_norm`
        // [head_dim], hidden=head_dim, the n_q_heads grid dim
        // broadcasts the same gamma across heads.
        unsafe {
            use cudarc::driver::sys::*;
            let mut q_ptr = scr.q_out_ptr;
            let mut gamma_ptr = full.q_norm.offset_bytes;
            let mut eps_arg = eps;
            let mut hd = head_dim;
            let args = [
                (&mut q_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut gamma_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_arg) as *mut f32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_rmsnorm_inplace_f16.raw() as CUfunction,
                n_q_heads as u32, 1, 1,
                (head_dim as u32).min(1024), 1, 1, 32 * 4,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 q_norm launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (4) K projection: [1024, 5120] FP8 GEMV → k_out_ptr.
        let kv_n = (n_kv_heads * head_dim) as u32; // 1024
        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m: 1, n: kv_n, k: hidden as u32,
            }.launch(
                fp8_gemv_fn,
                scr.k_out_ptr,
                full.k_proj.offset_bytes,
                full.k_proj.blockscale_ptr.unwrap_or(0),
                scr.h_work_ptr,
                stream_raw,
            )?;
        }

        // (5) V projection: [1024, 5120] FP8 GEMV → v_out_ptr.
        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m: 1, n: kv_n, k: hidden as u32,
            }.launch(
                fp8_gemv_fn,
                scr.v_out_ptr,
                full.v_proj.offset_bytes,
                full.v_proj.blockscale_ptr.unwrap_or(0),
                scr.h_work_ptr,
                stream_raw,
            )?;
        }

        // (6) k_norm: per-kv-head RMSNorm. Mirrors q_norm. Grid
        // (n_kv_heads, 1, 1), block (head_dim, 1, 1). gamma is
        // k_norm [head_dim].
        unsafe {
            use cudarc::driver::sys::*;
            let mut k_ptr = scr.k_out_ptr;
            let mut gamma_ptr = full.k_norm.offset_bytes;
            let mut eps_arg = eps;
            let mut hd = head_dim;
            let args = [
                (&mut k_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut gamma_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_arg) as *mut f32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_rmsnorm_inplace_f16.raw() as CUfunction,
                n_kv_heads as u32, 1, 1,
                (head_dim as u32).min(1024), 1, 1, 32 * 4,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 k_norm launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (7) fused_rope_qwen_partial_f16kv: apply Qwen partial RoPE
        // to q (in-place, rotary_dim of head_dim — only first 64 of
        // 256), and write rotated K + raw V into the per-layer KV
        // cache at slot=`position`. Q is rotated in place
        // (q_in == q_out aliasing is safe — each thread owns its
        // own (i, i+rotary_dim/2) pair).
        let kv_cache = self.kv_cache.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_qkv_only: kv_cache absent".into(),
        ))?;
        let rope = self.rope_tables.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_qkv_only: rope_tables absent".into(),
        ))?;
        let full_idx = kv_cache.layer_idx_to_full_idx[layer_idx].ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            format!("apply_full_attn_qkv_only: layer {layer_idx} not full-attn"),
        ))?;
        let layer_kv = &kv_cache.layers[full_idx];

        // Inline allocate positions + slot_mapping i32 device buffers.
        // For this smoke we always position=0 (single token at start).
        // Phase 2c-B-C will thread the real `position` through from
        // the request.
        let pos_region = arena.region(
            "qwen35_fattn_positions", 4, 4)?;
        let slot_region = arena.region(
            "qwen35_fattn_slot_mapping", 4, 4)?;
        let pos_bytes: [u8; 4] = (position as i32).to_le_bytes();
        unsafe {
            use cudarc::driver::sys::*;
            for dst in [pos_region.device_ptr(), slot_region.device_ptr()] {
                let rc = cuMemcpyHtoDAsync_v2(
                    dst as CUdeviceptr,
                    pos_bytes.as_ptr() as *const _, 4,
                    stream_raw as CUstream,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 fattn positions/slot HtoD",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        // Step 4 (Qwen 3.6 27B NVFP4) dispatch.
        //   * F16:   existing fused_rope_qwen_partial_f16kv launch
        //   * Nvfp4: fused_rope_qwen_partial_nvfp4kv launch — writes
        //     packed-4-bit K/V + per-(slot, kv_head) E4M3 scales +
        //     FP8-E4M3 Q with per-(token, head) dynamic scale.
        match layer_kv.dtype {
        Qwen35KvDtype::F16 =>
        unsafe {
            use cudarc::driver::sys::*;
            let mut q_in = scr.q_out_ptr;
            let mut k_in = scr.k_out_ptr;
            let mut v_in = scr.v_out_ptr;
            let mut q_out = scr.q_out_ptr; // in-place
            let mut key_cache = layer_kv.k_ptr;
            let mut value_cache = layer_kv.v_ptr;
            let mut cos = rope.cos_ptr;
            let mut sin = rope.sin_ptr;
            let mut positions_ptr = pos_region.device_ptr();
            let mut slot_ptr = slot_region.device_ptr();
            let mut num_tokens: i32 = 1;
            let mut nh = n_q_heads;
            let mut nkh = n_kv_heads;
            let mut hd = head_dim;
            let mut rd: i32 = rope.rotary_dim as i32;
            let args = [
                (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut key_cache) as *mut u64 as *mut core::ffi::c_void,
                (&mut value_cache) as *mut u64 as *mut core::ffi::c_void,
                (&mut cos) as *mut u64 as *mut core::ffi::c_void,
                (&mut sin) as *mut u64 as *mut core::ffi::c_void,
                (&mut positions_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut slot_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut num_tokens) as *mut i32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut rd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_fused_rope_qwen_partial_f16kv.raw() as CUfunction,
                1, n_q_heads as u32, 1,
                (head_dim as u32 / 2), 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 fused_rope_qwen_partial_f16kv launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        },
        Qwen35KvDtype::Nvfp4 => {
            let fn_rope = ker.fn_fused_rope_qwen_partial_nvfp4kv
                .ok_or_else(|| corrupt(
                    self.paths.model_dir.clone(),
                    "apply_full_attn_qkv_only: NVFP4 RoPE kernel \
                     not loaded".into()))?;
            unsafe {
                use cudarc::driver::sys::*;
                let mut q_in = scr.q_out_ptr;
                let mut k_in = scr.k_out_ptr;
                let mut v_in = scr.v_out_ptr;
                let mut q_fp8_out = scr.q_fp8_ptr;
                let mut key_packed = layer_kv.k_ptr;
                let mut value_packed = layer_kv.v_ptr;
                let mut key_scale = layer_kv.k_scale_ptr;
                let mut value_scale = layer_kv.v_scale_ptr;
                let mut cos = rope.cos_ptr;
                let mut sin = rope.sin_ptr;
                let mut positions_ptr = pos_region.device_ptr();
                let mut slot_ptr = slot_region.device_ptr();
                // No static-scalar Q descale here — we always feed
                // the dynamic q_scale_cache. The kernel writes to
                // q_scale_cache; the same buffer is read back by
                // the NVFP4 decode kernel. The static fallback is
                // unused (q_scale_cache != nullptr) but the arg
                // must be a valid device pointer — reuse the
                // cache pointer itself; the divide never sees this
                // address because the dynamic path takes priority.
                let mut q_scale_static = scr.q_scale_cache_ptr;
                let mut q_scale_cache = scr.q_scale_cache_ptr;
                let mut num_tokens: i32 = 1;
                let mut nh = n_q_heads;
                let mut nkh = n_kv_heads;
                let mut hd = head_dim;
                let mut rd: i32 = rope.rotary_dim as i32;
                let args = [
                    (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
                    (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
                    (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
                    (&mut q_fp8_out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut key_packed) as *mut u64 as *mut core::ffi::c_void,
                    (&mut value_packed) as *mut u64 as *mut core::ffi::c_void,
                    (&mut key_scale) as *mut u64 as *mut core::ffi::c_void,
                    (&mut value_scale) as *mut u64 as *mut core::ffi::c_void,
                    (&mut cos) as *mut u64 as *mut core::ffi::c_void,
                    (&mut sin) as *mut u64 as *mut core::ffi::c_void,
                    (&mut positions_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut slot_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut q_scale_static) as *mut u64 as *mut core::ffi::c_void,
                    (&mut q_scale_cache) as *mut u64 as *mut core::ffi::c_void,
                    (&mut num_tokens) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nkh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                    (&mut rd) as *mut i32 as *mut core::ffi::c_void,
                ];
                // Grid: (num_tokens, max(num_heads, num_kv_heads)).
                // Block: (head_dim) — one thread per element.
                let grid_y = n_q_heads.max(n_kv_heads) as u32;
                let rc = cuLaunchKernel(
                    fn_rope.raw() as CUfunction,
                    1, grid_y, 1,
                    head_dim as u32, 1, 1, 0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 fused_rope_qwen_partial_nvfp4kv launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            }
        }
        }

        // (8) flash_attention_2_decode_f16io: single-token decode
        // attention against the KV cache. Cache layout is
        //   [max_pos, num_kv_heads, head_dim] f16
        // which matches FA-2's paged-attn assumption when we treat
        //   block_size       = 1
        //   num_blocks       = max_pos
        //   block_tables     = identity [0..max_pos]   (PERSISTENT)
        //   context_lens     = [position+1]            (HtoD here)
        //
        // The block_tables buffer is allocated + identity-filled
        // exactly once at bring-up (see Qwen35KvCache); per step we
        // only need to write the current `context_len` to
        // `kv_cache.context_lens_ptr`. This kills the O(T²) host
        // arithmetic + per-FA-layer arena alloc that was happening
        // before (Codex review #2, 2026-05-12).
        let context_len = (position + 1) as i32;
        let bt_ptr = kv_cache.block_tables_ptr;
        let cl_ptr = kv_cache.context_lens_ptr;
        unsafe {
            use cudarc::driver::sys::*;
            let bytes = context_len.to_le_bytes();
            let rc = cuMemcpyHtoDAsync_v2(
                cl_ptr as CUdeviceptr,
                bytes.as_ptr() as *const _, 4,
                stream_raw as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 fattn context_lens HtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        const FA2_THREADS: i32 = 128;
        const FA2_BC: i32 = 32;
        // Step 4 (Qwen 3.6 27B NVFP4) decode dispatch.
        //   * F16:   existing flash_attention_2_decode_f16io
        //            (2 * BC * hd f32 = 2 * 32 * 256 * 4 = 64 KiB)
        //   * Nvfp4: flash_attention_2_decode_nvfp4kv. K/V dequant
        //            target is f16 smem (2 bytes/elem; halves the
        //            K/V tile footprint vs. f32 → 32 KiB).
        let smem_bytes = match layer_kv.dtype {
            Qwen35KvDtype::F16 =>
                2 * FA2_BC * head_dim * 4 + FA2_BC * 4 + (FA2_THREADS / 32) * 4,
            Qwen35KvDtype::Nvfp4 =>
                2 * FA2_BC * head_dim * 2 + FA2_BC * 4 + (FA2_THREADS / 32) * 4,
        };
        match layer_kv.dtype {
        Qwen35KvDtype::F16 =>
        unsafe {
            use cudarc::driver::sys::*;
            if smem_bytes as u32 >= 48 * 1024 {
                let rc = cuFuncSetAttribute(
                    ker.fn_flash_attention_2_decode_f16io.raw() as CUfunction,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem_bytes,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 fa2-decode cuFuncSetAttribute",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            let mut output = scr.attn_out_ptr;
            let mut query = scr.q_out_ptr;
            let mut key_cache = layer_kv.k_ptr;
            let mut value_cache = layer_kv.v_ptr;
            let mut block_tables = bt_ptr;
            let mut context_lens = cl_ptr;
            let mut scale_arg = scale;
            let mut nh = n_q_heads;
            let mut nkvh = n_kv_heads;
            let mut hd = head_dim;
            let mut bs: i32 = 1;
            // mbps = the number of identity slots the kernel may
            // walk per seq. The block_tables buffer has max_pos
            // entries; safe upper bound regardless of position.
            let mut mbps: i32 = kv_cache.max_pos as i32;
            let mut window: i32 = -1;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut query) as *mut u64 as *mut core::ffi::c_void,
                (&mut key_cache) as *mut u64 as *mut core::ffi::c_void,
                (&mut value_cache) as *mut u64 as *mut core::ffi::c_void,
                (&mut block_tables) as *mut u64 as *mut core::ffi::c_void,
                (&mut context_lens) as *mut u64 as *mut core::ffi::c_void,
                (&mut scale_arg) as *mut f32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut bs) as *mut i32 as *mut core::ffi::c_void,
                (&mut mbps) as *mut i32 as *mut core::ffi::c_void,
                (&mut window) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_flash_attention_2_decode_f16io.raw() as CUfunction,
                1, n_q_heads as u32, 1,
                FA2_THREADS as u32, 1, 1,
                smem_bytes as u32,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 flash_attention_2_decode_f16io launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        },
        Qwen35KvDtype::Nvfp4 => {
            let fn_dec = ker.fn_flash_attention_2_decode_nvfp4kv
                .ok_or_else(|| corrupt(
                    self.paths.model_dir.clone(),
                    "apply_full_attn_qkv_only: NVFP4 decode kernel \
                     not loaded".into()))?;
            unsafe {
                use cudarc::driver::sys::*;
                if smem_bytes as u32 >= 48 * 1024 {
                    let rc = cuFuncSetAttribute(
                        fn_dec.raw() as CUfunction,
                        CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        smem_bytes,
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen35 nvfp4-decode cuFuncSetAttribute",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                let mut output = scr.attn_out_ptr;
                let mut query = scr.q_fp8_ptr;
                let mut key_packed = layer_kv.k_ptr;
                let mut value_packed = layer_kv.v_ptr;
                let mut key_scale = layer_kv.k_scale_ptr;
                let mut value_scale = layer_kv.v_scale_ptr;
                let mut q_scale_cache = scr.q_scale_cache_ptr;
                let mut block_tables = bt_ptr;
                let mut context_lens = cl_ptr;
                // Static-scalar Q descale fallback. Never read on
                // the dynamic path (q_scale_cache != nullptr) but
                // must be a valid device pointer; reuse the cache.
                let mut q_descale = scr.q_scale_cache_ptr;
                let mut scale_arg = scale;
                let mut nh = n_q_heads;
                let mut nkvh = n_kv_heads;
                let mut hd = head_dim;
                let mut bs: i32 = 1;
                let mut mbps: i32 = kv_cache.max_pos as i32;
                let mut window: i32 = -1;
                let args = [
                    (&mut output) as *mut u64 as *mut core::ffi::c_void,
                    (&mut query) as *mut u64 as *mut core::ffi::c_void,
                    (&mut key_packed) as *mut u64 as *mut core::ffi::c_void,
                    (&mut value_packed) as *mut u64 as *mut core::ffi::c_void,
                    (&mut key_scale) as *mut u64 as *mut core::ffi::c_void,
                    (&mut value_scale) as *mut u64 as *mut core::ffi::c_void,
                    (&mut q_scale_cache) as *mut u64 as *mut core::ffi::c_void,
                    (&mut block_tables) as *mut u64 as *mut core::ffi::c_void,
                    (&mut context_lens) as *mut u64 as *mut core::ffi::c_void,
                    (&mut q_descale) as *mut u64 as *mut core::ffi::c_void,
                    (&mut scale_arg) as *mut f32 as *mut core::ffi::c_void,
                    (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                    (&mut bs) as *mut i32 as *mut core::ffi::c_void,
                    (&mut mbps) as *mut i32 as *mut core::ffi::c_void,
                    (&mut window) as *mut i32 as *mut core::ffi::c_void,
                ];
                let rc = cuLaunchKernel(
                    fn_dec.raw() as CUfunction,
                    1, n_q_heads as u32, 1,
                    FA2_THREADS as u32, 1, 1,
                    smem_bytes as u32,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 flash_attention_2_decode_nvfp4kv launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            }
        }
        }

        // (9) sigmoid_mul_f16: apply the attn_output_gate finisher.
        //   attn_out = sigmoid(q_gate) * attn_out   element-wise
        // Layout: 1-D over n = n_q_heads * head_dim.
        let gated_n = (n_q_heads * head_dim) as i32;
        unsafe {
            use cudarc::driver::sys::*;
            let mut out = scr.attn_out_ptr;
            let mut vals = scr.attn_out_ptr;
            let mut gate = scr.q_gate_ptr;
            let mut n_arg = gated_n;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut vals) as *mut u64 as *mut core::ffi::c_void,
                (&mut gate) as *mut u64 as *mut core::ffi::c_void,
                (&mut n_arg) as *mut i32 as *mut core::ffi::c_void,
            ];
            let grid_x: u32 = ((gated_n as u32) + 255) / 256;
            let rc = cuLaunchKernel(
                ker.fn_sigmoid_mul_f16.raw() as CUfunction,
                grid_x, 1, 1, 256, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 sigmoid_mul_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (10-11) Fused o_proj + post-attn residual add (Phase 8
        // other-models fusion, 2026-05-23). Single-kernel
        // replacement for the back-to-back Fp8GemvF16In →
        // vector_add_f16 pair. Default-on. Opt-out via
        // `RVLLM_QWEN35_FP8_GEMV_RESIDUAL_FUSED=0` runs the
        // pre-39f7c1a unfused chain (fp8_gemv into scratch +
        // vector_add_f16).
        let o_n = hidden as u32;
        let o_k = (n_q_heads * head_dim) as u32;
        let fused = std::env::var("RVLLM_QWEN35_FP8_GEMV_RESIDUAL_FUSED")
            .map(|s| s != "0")
            .unwrap_or(true);
        unsafe {
            use cudarc::driver::sys::*;
            if fused {
                let mut h_resid = scr.h_residual_ptr;
                let mut w_ptr = full.o_proj.offset_bytes;
                let mut s_ptr = full.o_proj.blockscale_ptr.unwrap_or(0);
                let mut x_ptr = scr.attn_out_ptr;
                let mut m_i: i32 = 1;
                let mut n_i: i32 = o_n as i32;
                let mut k_i: i32 = o_k as i32;
                let mut ncb: i32 = ((o_k as i32) + 127) / 128;
                let args = [
                    (&mut h_resid) as *mut u64 as *mut core::ffi::c_void,
                    (&mut w_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut s_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut x_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut ncb) as *mut i32 as *mut core::ffi::c_void,
                ];
                let grid = ((o_n + 7) / 8, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    ker.fn_fp8_gemv_f16in_residual_add.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    256, 1, 1, 0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 fp8_gemv_f16in_residual_add (o_proj+resid) launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            } else {
                let arena = self.arena.as_ref().ok_or_else(|| corrupt(
                    self.paths.model_dir.clone(),
                    "qwen35 unfused o_proj+resid: arena unset".into()))?;
                let temp = arena.region("qwen35_oproj_unfused_temp",
                                        (o_n as usize) * 2, 16)?;
                let gemv_fn = ker.fn_fp8_gemv_wpr_native_f16in.ok_or_else(|| corrupt(
                    self.paths.model_dir.clone(),
                    "qwen35 unfused o_proj+resid: fn_fp8_gemv_wpr_native_f16in unloaded".into()))?;
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m: 1, n: o_n, k: o_k,
                }.launch(
                    gemv_fn,
                    temp.device_ptr(),
                    full.o_proj.offset_bytes,
                    full.o_proj.blockscale_ptr.unwrap_or(0),
                    scr.attn_out_ptr,
                    stream_raw,
                )?;
                let mut dst = scr.h_residual_ptr;
                let mut src = temp.device_ptr();
                let mut n_elem: i32 = o_n as i32;
                let v_args = [
                    (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                    (&mut src) as *mut u64 as *mut core::ffi::c_void,
                    (&mut n_elem) as *mut i32 as *mut core::ffi::c_void,
                ];
                let v_block: u32 = 256;
                let v_grid: u32 = ((o_n + v_block - 1) / v_block).max(1);
                let rc = cuLaunchKernel(
                    ker.fn_vector_add_f16.raw() as CUfunction,
                    v_grid, 1, 1,
                    v_block, 1, 1,
                    0,
                    stream_raw as CUstream,
                    v_args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 unfused o_proj+resid vector_add launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            }
        }
        Ok(())
    }

    /// Phase #1-c body: batched full-attention block over N tokens.
    ///
    /// Mirrors `Qwen36Bringup::apply_layer_full_attn_batched` with
    /// Qwen 3.5's dim numbers (num_heads=24, num_kv_heads=4,
    /// head_dim=256, rotary_dim=64, attn_output_gate=true). Reuses
    /// the existing `flash_attention_2_f16kv_kernel` prefill kernel
    /// (f32-Q / f16-KV / f32-O) sandwiched between two cast
    /// launches.
    ///
    /// 9-op chain (analog of per-token `apply_full_attn_qkv_only`):
    ///   1) RMSNorm input on [N, hidden] copy → normed
    ///   2) q_proj + split_q_gate + q_norm batched (Q has
    ///      attn_output_gate — q_proj output is [N, 2*num_q*hd]
    ///      and split into q_split + gate, then q_norm applied
    ///      per head)
    ///   3) k_proj + k_norm batched
    ///   4) v_proj batched
    ///   5) fused_rope_qwen_partial_f16kv batched
    ///      (positions_dev_ptr is [N] i32, slot=positions)
    ///   6) cast q→f32, flash_attention_2_f16kv (causal-masked,
    ///      num_query_tokens=N), cast attn_out→f16
    ///   7) sigmoid_mul (attn_output_gate) elementwise [N, q_size]
    ///   8) o_proj batched [N, q_size] → [N, hidden]
    ///   9) residual: h_residual_buf += out elementwise [N*hidden]
    ///
    /// Caller must provide `positions_dev_ptr` ([N] i32, value
    /// `[start_pos, start_pos+1, …, start_pos+N-1]`) and
    /// `prefill_ctx_len_dev_ptr` ([1] i32, value N). The layer-major
    /// driver (#1-d) allocates + fills both once per request.
    /// `num_tokens == 1` defers to the per-token path so the decode
    /// canary stays byte-stable.
    pub unsafe fn apply_full_attn_layer_batched(
        &self,
        layer_idx: usize,
        num_tokens: u32,
        start_position: u32,
        h_residual_buf: u64,
        positions_dev_ptr: u64,
        prefill_ctx_len_dev_ptr: u64,
    ) -> Result<()> {
        if num_tokens == 0 {
            return Ok(());
        }
        if num_tokens == 1 {
            // Decode degenerate: defer to per-token path so the
            // greedy canary stays byte-identical with the
            // production decode loop.
            return self.apply_full_attn_qkv_only(layer_idx, start_position);
        }
        let arch = &self.arch;
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_layer_batched: model absent".into()))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_layer_batched: outside_kernels absent".into()))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_layer_batched: stream absent".into()))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_layer_batched: arena absent".into()))?;
        let kv_cache = self.kv_cache.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_layer_batched: kv_cache absent".into()))?;
        let rope = self.rope_tables.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_layer_batched: rope_tables absent".into()))?;
        let layer = model.layers.get(layer_idx).ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            format!("apply_full_attn_layer_batched: layer {layer_idx} out of range")))?;
        let full = match &layer.attn {
            rvllm_loader::qwen35_weights::Qwen35LayerAttn::Full(f) => f,
            rvllm_loader::qwen35_weights::Qwen35LayerAttn::Linear(_) => {
                return Err(corrupt(self.paths.model_dir.clone(),
                    format!("apply_full_attn_layer_batched: layer {layer_idx} is linear-attn")));
            }
        };
        let full_idx = kv_cache.layer_idx_to_full_idx[layer_idx].ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            format!("apply_full_attn_layer_batched: layer {layer_idx} not full-attn")))?;
        let layer_kv = &kv_cache.layers[full_idx];

        let stream_raw = stream.raw() as u64;
        let hidden = arch.base.hidden_size as i32;
        let hidden_u = hidden as u32;
        let head_dim = arch.base.head_dim as i32;
        let n_q_heads = arch.base.num_attention_heads as i32;
        let n_kv_heads = arch.base.num_key_value_heads as i32;
        let q_size = (n_q_heads * head_dim) as u32; // 6144 for Qwen 3.5
        let qsize_us = q_size as usize;
        let eps = arch.base.rms_norm_eps;
        let n = num_tokens as usize;
        let h = hidden as usize;
        let fp8_gemv_fn = ker.fn_fp8_gemv_wpr_native_f16in.ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_full_attn_layer_batched: fp8_gemv_wpr_native_f16in unavailable".into()))?;
        let cutlass_full_min_tokens =
            std::env::var("RVLLM_QWEN35_FULL_CUTLASS_MIN_TOKENS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(128);
        let cutlass_full_requested =
            crate::gemma4_bring_up::parse_truthy_env("RVLLM_QWEN35_FULL_CUTLASS_SM120")
                .unwrap_or(false)
                && num_tokens >= cutlass_full_min_tokens;
        let cutlass_full_lib = if cutlass_full_requested {
            match &self.cutlass {
                CutlassBackend::SoSm120(lib) => Some(lib),
                _ => None,
            }
        } else {
            None
        };

        // (1) RMSNorm batched on [N, hidden] copy of h_residual.
        let normed = arena.region("qwen35_bfattn_normed", n * h * 2, 16)?.device_ptr();
        {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                normed, h_residual_buf, n * h * 2,
                stream_raw as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bfattn DtoD normed",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens, hidden: hidden_u, eps,
        }.launch(
            ker.fn_rmsnorm_inplace_f16,
            normed,
            full.input_layernorm.offset_bytes,
            stream_raw,
        )?;

        // (2) Q proj at m=N: [N, hidden] → [N, 2*q_size] (Q has
        // attn_output_gate; output is interleaved Q + gate).
        let qg_n = (2 * n_q_heads * head_dim) as u32; // 12288
        let qg_region = arena.region(
            "qwen35_bfattn_qg", n * (qg_n as usize) * 2, 16)?.device_ptr();
        let q_bs = full.q_proj.blockscale_ptr.unwrap_or(0);
        if let Some(lib) = cutlass_full_lib {
            self.qwen35_fp8_cutlass_blockscale_sm120(
                ker,
                lib,
                qg_region,
                full.q_proj.offset_bytes,
                q_bs,
                normed,
                num_tokens,
                qg_n,
                hidden_u,
                stream_raw,
                "qwen35_bfattn_qg_in_fp8",
            )?;
        } else {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m: num_tokens, n: qg_n, k: hidden_u,
            }.launch(
                fp8_gemv_fn, qg_region,
                full.q_proj.offset_bytes, q_bs, normed, stream_raw,
            )?;
        }

        // (3) split_q_gate batched: kernel grid (n_q_heads, N, 1).
        let q_region    = arena.region("qwen35_bfattn_q",    n * qsize_us * 2, 16)?.device_ptr();
        let gate_region = arena.region("qwen35_bfattn_gate", n * qsize_us * 2, 16)?.device_ptr();
        {
            use cudarc::driver::sys::*;
            let mut qo = q_region;
            let mut go = gate_region;
            let mut qi = qg_region;
            let mut nh: i32 = n_q_heads;
            let mut hd: i32 = head_dim;
            let args = [
                (&mut qo) as *mut u64 as *mut core::ffi::c_void,
                (&mut go) as *mut u64 as *mut core::ffi::c_void,
                (&mut qi) as *mut u64 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_split_q_gate_f16.raw() as CUfunction,
                n_q_heads as u32, num_tokens, 1,
                head_dim as u32, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bfattn split_q_gate",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (3b) q_norm batched: rmsnorm over (n_q_heads*num_tokens)
        // rows of length head_dim each. Same trick qwen36 uses.
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: (n_q_heads as u32) * num_tokens,
            hidden: head_dim as u32, eps,
        }.launch(
            ker.fn_rmsnorm_inplace_f16,
            q_region, full.q_norm.offset_bytes, stream_raw,
        )?;

        // (4) K proj at m=N, then k_norm (per kv-head across tokens).
        let k_n = (n_kv_heads * head_dim) as u32; // 1024
        let k_region = arena.region(
            "qwen35_bfattn_k", n * (k_n as usize) * 2, 16)?.device_ptr();
        let k_bs = full.k_proj.blockscale_ptr.unwrap_or(0);
        if let Some(lib) = cutlass_full_lib {
            self.qwen35_fp8_cutlass_blockscale_sm120(
                ker,
                lib,
                k_region,
                full.k_proj.offset_bytes,
                k_bs,
                normed,
                num_tokens,
                k_n,
                hidden_u,
                stream_raw,
                "qwen35_bfattn_k_in_fp8",
            )?;
        } else {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m: num_tokens, n: k_n, k: hidden_u,
            }.launch(
                fp8_gemv_fn, k_region,
                full.k_proj.offset_bytes, k_bs, normed, stream_raw,
            )?;
        }
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: (n_kv_heads as u32) * num_tokens,
            hidden: head_dim as u32, eps,
        }.launch(
            ker.fn_rmsnorm_inplace_f16,
            k_region, full.k_norm.offset_bytes, stream_raw,
        )?;

        // (5) V proj at m=N.
        let v_n = k_n;
        let v_region = arena.region(
            "qwen35_bfattn_v", n * (v_n as usize) * 2, 16)?.device_ptr();
        let v_bs = full.v_proj.blockscale_ptr.unwrap_or(0);
        if let Some(lib) = cutlass_full_lib {
            self.qwen35_fp8_cutlass_blockscale_sm120(
                ker,
                lib,
                v_region,
                full.v_proj.offset_bytes,
                v_bs,
                normed,
                num_tokens,
                v_n,
                hidden_u,
                stream_raw,
                "qwen35_bfattn_v_in_fp8",
            )?;
        } else {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m: num_tokens, n: v_n, k: hidden_u,
            }.launch(
                fp8_gemv_fn, v_region,
                full.v_proj.offset_bytes, v_bs, normed, stream_raw,
            )?;
        }

        // (6) RoPE + KV-cache write batched. positions and slot_mapping
        // are both `positions_dev_ptr` ([N] i32) — qwen3-next slot==position.
        let (q_fp8, q_scale_cache) = match layer_kv.dtype {
            Qwen35KvDtype::F16 => (0u64, 0u64),
            Qwen35KvDtype::Nvfp4 => {
                let q_fp8 = arena.region("qwen35_bfattn_q_fp8", n * qsize_us, 16)?;
                let q_scale = arena.region(
                    "qwen35_bfattn_q_scale_cache",
                    n * (n_q_heads as usize) * 4,
                    16,
                )?;
                (q_fp8.device_ptr(), q_scale.device_ptr())
            }
        };
        match layer_kv.dtype {
        Qwen35KvDtype::F16 => {
            use cudarc::driver::sys::*;
            let mut q_in  = q_region;
            let mut k_in  = k_region;
            let mut v_in  = v_region;
            let mut q_out = q_region;       // in-place
            let mut kc    = layer_kv.k_ptr;
            let mut vc    = layer_kv.v_ptr;
            let mut cos   = rope.cos_ptr;
            let mut sin   = rope.sin_ptr;
            let mut pos   = positions_dev_ptr;
            let mut slot  = positions_dev_ptr;
            let mut nt: i32 = num_tokens as i32;
            let mut nh: i32 = n_q_heads;
            let mut nkh: i32 = n_kv_heads;
            let mut hd: i32 = head_dim;
            let mut rd: i32 = rope.rotary_dim as i32;
            let args = [
                (&mut q_in)  as *mut u64 as *mut core::ffi::c_void,
                (&mut k_in)  as *mut u64 as *mut core::ffi::c_void,
                (&mut v_in)  as *mut u64 as *mut core::ffi::c_void,
                (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut kc)    as *mut u64 as *mut core::ffi::c_void,
                (&mut vc)    as *mut u64 as *mut core::ffi::c_void,
                (&mut cos)   as *mut u64 as *mut core::ffi::c_void,
                (&mut sin)   as *mut u64 as *mut core::ffi::c_void,
                (&mut pos)   as *mut u64 as *mut core::ffi::c_void,
                (&mut slot)  as *mut u64 as *mut core::ffi::c_void,
                (&mut nt)    as *mut i32 as *mut core::ffi::c_void,
                (&mut nh)    as *mut i32 as *mut core::ffi::c_void,
                (&mut nkh)   as *mut i32 as *mut core::ffi::c_void,
                (&mut hd)    as *mut i32 as *mut core::ffi::c_void,
                (&mut rd)    as *mut i32 as *mut core::ffi::c_void,
            ];
            let max_h = n_q_heads.max(n_kv_heads) as u32;
            let block_x: u32 = (head_dim / 2) as u32;
            let rc = cuLaunchKernel(
                ker.fn_fused_rope_qwen_partial_f16kv.raw() as CUfunction,
                num_tokens, max_h, 1,
                block_x, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bfattn fused_rope_qwen_partial_f16kv",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        },
        Qwen35KvDtype::Nvfp4 => {
            let fn_rope = ker.fn_fused_rope_qwen_partial_nvfp4kv
                .ok_or_else(|| corrupt(
                    self.paths.model_dir.clone(),
                    "qwen35_bfattn: NVFP4 RoPE kernel not loaded".into()))?;
            use cudarc::driver::sys::*;
            let mut q_in = q_region;
            let mut k_in = k_region;
            let mut v_in = v_region;
            let mut q_fp8_out = q_fp8;
            let mut key_packed = layer_kv.k_ptr;
            let mut value_packed = layer_kv.v_ptr;
            let mut key_scale = layer_kv.k_scale_ptr;
            let mut value_scale = layer_kv.v_scale_ptr;
            let mut cos = rope.cos_ptr;
            let mut sin = rope.sin_ptr;
            let mut pos = positions_dev_ptr;
            let mut slot = positions_dev_ptr;
            let mut q_scale_static = q_scale_cache;
            let mut q_scale_dyn = q_scale_cache;
            let mut nt: i32 = num_tokens as i32;
            let mut nh: i32 = n_q_heads;
            let mut nkh: i32 = n_kv_heads;
            let mut hd: i32 = head_dim;
            let mut rd: i32 = rope.rotary_dim as i32;
            let args = [
                (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_fp8_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut key_packed) as *mut u64 as *mut core::ffi::c_void,
                (&mut value_packed) as *mut u64 as *mut core::ffi::c_void,
                (&mut key_scale) as *mut u64 as *mut core::ffi::c_void,
                (&mut value_scale) as *mut u64 as *mut core::ffi::c_void,
                (&mut cos) as *mut u64 as *mut core::ffi::c_void,
                (&mut sin) as *mut u64 as *mut core::ffi::c_void,
                (&mut pos) as *mut u64 as *mut core::ffi::c_void,
                (&mut slot) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_scale_static) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_scale_dyn) as *mut u64 as *mut core::ffi::c_void,
                (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut rd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let max_h = n_q_heads.max(n_kv_heads) as u32;
            let rc = cuLaunchKernel(
                fn_rope.raw() as CUfunction,
                num_tokens, max_h, 1,
                head_dim as u32, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bfattn fused_rope_qwen_partial_nvfp4kv",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        }

        // (7-9) Attention. F16 uses the existing f32-ABI prefill kernel;
        // NVFP4 uses the unified packed-KV prefill and returns f16 directly.
        let attn_out = match layer_kv.dtype {
        Qwen35KvDtype::F16 => {
        // Cast q [N, q_size] f16 -> f32 for the prefill kernel.
        let q_f32_bytes = n * qsize_us * 4;
        let q_f32 = arena.region("qwen35_bfattn_qf32", q_f32_bytes, 16)?.device_ptr();
        {
            use cudarc::driver::sys::*;
            let n_elem = (n * qsize_us) as i32;
            let mut out = q_f32;
            let mut input = q_region;
            let mut nn = n_elem;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n_elem as u32) + block - 1) / block;
            let rc = cuLaunchKernel(
                ker.fn_cast_f16_to_f32.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bfattn cast f16→f32",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // Allocate seq_start_pos = [0, N] (2 i32) for the prefill
        // kernel. Stream-ordered cuMemsetD32Async sets both slots.
        let seq_start = arena.region(
            "qwen35_bfattn_seqstart", 2 * 4, 16)?.device_ptr();
        {
            use cudarc::driver::sys::*;
            let r0 = cuMemsetD32Async(seq_start, 0, 1, stream_raw as CUstream);
            let r1 = cuMemsetD32Async(seq_start + 4, num_tokens, 1, stream_raw as CUstream);
            if r0 != CUresult::CUDA_SUCCESS || r1 != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bfattn seq_start_pos memset",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (8) FA-2 prefill kernel. Same smem requirement as decode
        // (head_dim=256 → ~64 KiB → cuFuncSetAttribute needed).
        let attn_out_f32 = arena.region(
            "qwen35_bfattn_attn_f32", q_f32_bytes, 16)?.device_ptr();
        const FA2_THREADS: i32 = 128;
        const FA2_BC: i32 = 32;
        let smem_bytes = 2 * FA2_BC * head_dim * 4 + FA2_BC * 4 + (FA2_THREADS / 32) * 4;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        {
            use cudarc::driver::sys::*;
            if smem_bytes as u32 >= 48 * 1024 {
                let rc = cuFuncSetAttribute(
                    ker.fn_flash_attention_2_f16kv.raw() as CUfunction,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem_bytes,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35_bfattn fa2-prefill cuFuncSetAttribute",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            }
            let mut output       = attn_out_f32;
            let mut query        = q_f32;
            let mut key_cache    = layer_kv.k_ptr;
            let mut value_cache  = layer_kv.v_ptr;
            let mut block_tables = kv_cache.block_tables_ptr;
            let mut context_lens = prefill_ctx_len_dev_ptr;
            let mut seq_start_p  = seq_start;
            let mut scale_arg = scale;
            let mut nh = n_q_heads;
            let mut nkvh = n_kv_heads;
            let mut hd = head_dim;
            let mut bs: i32 = 1;
            let mut max_ctx: i32 = (start_position + num_tokens) as i32;
            let mut mbps: i32 = kv_cache.max_pos as i32;
            let mut nqt: i32 = num_tokens as i32;
            let mut causal: i32 = 1;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut query) as *mut u64 as *mut core::ffi::c_void,
                (&mut key_cache) as *mut u64 as *mut core::ffi::c_void,
                (&mut value_cache) as *mut u64 as *mut core::ffi::c_void,
                (&mut block_tables) as *mut u64 as *mut core::ffi::c_void,
                (&mut context_lens) as *mut u64 as *mut core::ffi::c_void,
                (&mut seq_start_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut scale_arg) as *mut f32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut bs) as *mut i32 as *mut core::ffi::c_void,
                (&mut max_ctx) as *mut i32 as *mut core::ffi::c_void,
                (&mut mbps) as *mut i32 as *mut core::ffi::c_void,
                (&mut nqt) as *mut i32 as *mut core::ffi::c_void,
                (&mut causal) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_flash_attention_2_f16kv.raw() as CUfunction,
                1u32, n_q_heads as u32, 1,
                FA2_THREADS as u32, 1, 1,
                smem_bytes as u32,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bfattn flash_attention_2_f16kv",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // Cast attn_out [N, q_size] f32 -> f16.
        let attn_out = arena.region(
            "qwen35_bfattn_attn_f16", n * qsize_us * 2, 16)?.device_ptr();
        {
            use cudarc::driver::sys::*;
            let n_elem = (n * qsize_us) as i32;
            let mut out = attn_out;
            let mut input = attn_out_f32;
            let mut nn = n_elem;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n_elem as u32) + block - 1) / block;
            let rc = cuLaunchKernel(
                ker.fn_cast_f32_to_f16.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bfattn cast f32→f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        attn_out
        },
        Qwen35KvDtype::Nvfp4 => {
            let attn_out = arena.region(
                "qwen35_bfattn_attn_f16", n * qsize_us * 2, 16)?.device_ptr();
            // cu_seqlens = [0, num_tokens] (i32 × 2). Previously this was
            // populated via `copy_from_host` which is a SYNCHRONOUS HtoD
            // on the legacy default stream — 16 full-attn layers ×
            // 1 prefill = 16 host blocks per request. Use stream-ordered
            // `cuMemsetD32Async` to match the F16 path's setup
            // (qwen35_bfattn_seqstart uses the same pattern at line ~2734).
            let cu_seqlens_region = arena.region(
                "qwen35_bfattn_cu_seqlens", 2 * 4, 16)?;
            let cu_seqlens_ptr = cu_seqlens_region.device_ptr();
            unsafe {
                use cudarc::driver::sys::*;
                let r0 = cuMemsetD32Async(
                    cu_seqlens_ptr, 0, 1, stream_raw as CUstream);
                let r1 = cuMemsetD32Async(
                    cu_seqlens_ptr + 4, num_tokens, 1,
                    stream_raw as CUstream);
                if r0 != CUresult::CUDA_SUCCESS || r1 != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35_bfattn nvfp4 cu_seqlens memset",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            }
            unsafe {
                let params = rvllm_attention::PagedPrefillParams {
                    num_seqs: 1,
                    num_tokens,
                    num_heads: n_q_heads as u32,
                    num_kv_heads: n_kv_heads as u32,
                    head_dim: head_dim as u32,
                    block_size: 1,
                    max_blocks_per_seq: kv_cache.max_pos as u32,
                    num_blocks_total: kv_cache.max_pos as u32,
                    scale: 1.0_f32 / (head_dim as f32).sqrt(),
                    window_size_left: -1,
                };
                let num_queries_per_kv = (n_q_heads / n_kv_heads) as u32;
                let unified = rvllm_attention::UnifiedPrefillParams {
                    num_queries_per_kv,
                    tile_size: std::env::var("RVLLM_QWEN35_UNIFIED_TILE_SIZE")
                        .ok().and_then(|s| s.parse().ok())
                        .unwrap_or(if head_dim <= 256 { 32 } else { 16 }),
                    block_q: (rvllm_attention::UNIFIED_PREFILL_BLOCK_M
                        / num_queries_per_kv.max(1)).max(1),
                    use_mma: true,
                };
                let backend = self.attn_backend_full.as_ref().ok_or_else(|| corrupt(
                    self.paths.model_dir.clone(),
                    "qwen35_bfattn: attention backend absent".into()))?;
                let prefill = rvllm_attention::PagedPrefillNvfp4Launcher::new(backend);
                prefill.launch_nvfp4kv_unified_sm121(
                    params,
                    unified,
                    attn_out,
                    q_fp8,
                    layer_kv.k_ptr,
                    layer_kv.v_ptr,
                    layer_kv.k_scale_ptr,
                    layer_kv.v_scale_ptr,
                    q_scale_cache,
                    kv_cache.block_tables_ptr,
                    cu_seqlens_ptr,
                    prefill_ctx_len_dev_ptr,
                    q_scale_cache,
                    false,
                    stream_raw,
                )?;
            }
            attn_out
        }
        };

        // (10) attn_output_gate (sigmoid_mul) over N * q_size.
        let gated = arena.region(
            "qwen35_bfattn_gated", n * qsize_us * 2, 16)?.device_ptr();
        {
            use cudarc::driver::sys::*;
            let n_elem = (n * qsize_us) as i32;
            let mut out = gated;
            let mut vals = attn_out;
            let mut gate = gate_region;
            let mut nn = n_elem;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut vals) as *mut u64 as *mut core::ffi::c_void,
                (&mut gate) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n_elem as u32) + block - 1) / block;
            let rc = cuLaunchKernel(
                ker.fn_sigmoid_mul_f16.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bfattn sigmoid_mul",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (11) o_proj batched: [N, q_size] → [N, hidden].
        let o_n = hidden_u;
        let o_k = q_size;
        let o_bs = full.o_proj.blockscale_ptr.unwrap_or(0);
        let out_region = arena.region(
            "qwen35_bfattn_out", n * (o_n as usize) * 2, 16)?.device_ptr();
        if let Some(lib) = cutlass_full_lib {
            self.qwen35_fp8_cutlass_blockscale_sm120(
                ker,
                lib,
                out_region,
                full.o_proj.offset_bytes,
                o_bs,
                gated,
                num_tokens,
                o_n,
                o_k,
                stream_raw,
                "qwen35_bfattn_out_in_fp8",
            )?;
        } else {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m: num_tokens, n: o_n, k: o_k,
            }.launch(
                fp8_gemv_fn, out_region,
                full.o_proj.offset_bytes, o_bs, gated, stream_raw,
            )?;
        }

        // (12) Residual: h_residual_buf += out elementwise [N*hidden].
        {
            use cudarc::driver::sys::*;
            let n_elem = (n as i32) * hidden;
            let mut dst = h_residual_buf;
            let mut src = out_region;
            let mut nn: i32 = n_elem;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n_elem as u32) + block - 1) / block;
            let rc = cuLaunchKernel(
                ker.fn_vector_add_f16.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bfattn residual vector_add",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        Ok(())
    }

    /// Phase #1-d: layer-major batched prefill driver.
    ///
    /// One call processes all N prompt tokens through the 64-layer
    /// transformer stack — layer-major (each layer in turn over the
    /// full [N, hidden] buffer), not token-major. Returns the
    /// predicted next token (the lm_head argmax of the LAST prompt
    /// position's hidden state).
    ///
    /// Steps:
    ///   1. Allocate [N, hidden] h_residual + per-request small
    ///      device buffers (positions[N] i32, ctx_len[1] i32,
    ///      token_ids[N] i32).
    ///   2. HtoD positions = [0, 1, …, N-1], ctx_len = N,
    ///      token_ids = prompt_ids.
    ///   3. Batched embed: EmbeddingGatherLaunch at num_tokens=N
    ///      → fills [N, hidden].
    ///   4. Vision splices via DtoD over the right rows.
    ///   5. For li in 0..num_hidden_layers:
    ///        match arch.layer_types[li]:
    ///          Linear → apply_linear_attn_layer_batched
    ///          Full   → apply_full_attn_layer_batched
    ///        apply_dense_mlp_layer_batched
    ///   6. DtoD last row of h_residual into scr.h_residual_ptr
    ///      so the existing forward_finalize_argmax pipeline can
    ///      finish lm_head + argmax untouched.
    ///   7. Return forward_finalize_argmax()'s predicted token.
    ///
    /// Caller is responsible for restoring the arena to the
    /// checkpoint after this returns — same contract as the
    /// per-token forward_all_layers_with_splice path.
    pub unsafe fn forward_qwen35_decode_argmax_all(
        &self,
        token_ids: &[u32],
        start_position: u32,
    ) -> Result<Vec<i32>> {
        if token_ids.len() == 1 {
            let tok = self.forward_all_layers_smoke(token_ids[0], start_position)?;
            return Ok(vec![tok as i32]);
        }
        let h_residual_buf =
            self.forward_qwen35_tokens_batched(token_ids, start_position)?;
        self.forward_finalize_argmax_many_streamed(
            h_residual_buf,
            token_ids.len() as u32,
        )
    }

    /// Exact-math multi-row closer for speculative verification.
    ///
    /// This deliberately keeps the production M=1 lm_head path for
    /// every row so per-row FP8 activation scales stay identical to
    /// `forward_finalize_argmax()`. The win is host-side: enqueue all
    /// row closers and device-token copies on the CUDA stream, then
    /// fence and DtoH once instead of once per row.
    unsafe fn forward_finalize_argmax_many_streamed(
        &self,
        h_residual_buf: u64,
        rows: u32,
    ) -> Result<Vec<i32>> {
        if rows == 0 {
            return Ok(Vec::new());
        }
        let scr = self.scratch.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_finalize_argmax_many_streamed: scratch absent".into()))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_finalize_argmax_many_streamed: outside_kernels absent".into()))?;
        let cublaslt = self.cublaslt.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_finalize_argmax_many_streamed: cublaslt absent".into()))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_finalize_argmax_many_streamed: stream absent".into()))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_finalize_argmax_many_streamed: arena absent".into()))?;
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_finalize_argmax_many_streamed: model absent".into()))?;
        let arch = &self.arch;
        let hidden = arch.base.hidden_size as u32;
        let vocab = arch.base.vocab_size as u32;
        let eps = arch.base.rms_norm_eps;
        let stream_raw = stream.raw() as u64;
        let row_bytes = hidden as usize * 2;

        let hidden_fp8_region = arena.region(
            "qwen35_many_hidden_fp8", hidden as usize, 16)?;
        let hidden_scale_region = arena.region(
            "qwen35_many_hidden_scale", 4, 4)?;
        let tokens_region = arena.region(
            "qwen35_many_tokens", rows as usize * 4, 4)?;

        for row in 0..rows {
            use cudarc::driver::sys::*;
            let src = h_residual_buf + (row as u64) * (row_bytes as u64);
            let rc = cuMemcpyDtoDAsync_v2(
                scr.h_residual_ptr,
                src,
                row_bytes,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 many-finalize row DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: 1, hidden, eps,
            }.launch(
                ker.fn_fused_rmsnorm_fp8_quant,
                hidden_fp8_region.device_ptr(),
                hidden_scale_region.device_ptr(),
                scr.h_residual_ptr,
                model.outside.final_norm.offset_bytes,
                stream_raw,
            )?;
            cublaslt.fp8_gemm(
                hidden_fp8_region.device_ptr(),
                model.outside.lm_head_fp8.offset_bytes,
                scr.logits_ptr,
                1, vocab as i32, hidden as i32,
                hidden_scale_region.device_ptr(),
                model.outside.lm_head_fp8.scale_ptr,
                stream_raw,
            )?;
            let block_dim: u32 = vocab.min(1024);
            let mut row_ptr = scr.logits_ptr;
            let mut out_ptr = scr.token_out_ptr;
            let mut vsz: i32 = vocab as i32;
            let args = [
                (&mut row_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut vsz) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_argmax_f16.raw() as CUfunction,
                1, 1, 1, block_dim, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 many-finalize argmax",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
            let dst = tokens_region.device_ptr() + (row as u64) * 4;
            let rc = cuMemcpyDtoDAsync_v2(
                dst,
                scr.token_out_ptr,
                4,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 many-finalize token DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        stream.fence()?;
        let mut out = vec![0i32; rows as usize];
        {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut _,
                tokens_region.device_ptr() as CUdeviceptr,
                rows as usize * 4);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 many-finalize DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        Ok(out)
    }

    pub unsafe fn forward_qwen35_decode_commit_only(
        &self,
        token_ids: &[u32],
        start_position: u32,
    ) -> Result<()> {
        if token_ids.len() == 1 {
            self.forward_layers_only(token_ids[0], start_position, None)?;
            return Ok(());
        }
        // SPEC-DECODE CORRECTNESS FIX (2026-05-21):
        // Use sequential single-token forwards instead of the batched
        // multi-token kernel chain. Rationale:
        //
        //   `apply_linear_attn_layer_batched` / `apply_full_attn_layer_batched`
        //   advance the recurrent state (linear-attn delta + conv1d)
        //   over N tokens in one kernel launch. The BF16 accumulation
        //   order inside the batched kernel differs from N sequential
        //   single-token launches, producing slightly different state
        //   bytes. Over ~3 verify iterations in a single spec session
        //   the drift compounds enough to flip an argmax decision and
        //   the spec output diverges from the eager byte-identical
        //   sequence (md5 686dbb vs eager 531d59a on the canonical
        //   zeroclaw-shape prompt, 80-tok decode).
        //
        //   `commit_only` is invoked AFTER `restore_recurrent_state`
        //   to replay the accepted prefix from the snapshot baseline.
        //   For the spec session to produce byte-identical output to
        //   eager, the replay must advance state EXACTLY as N
        //   sequential eager `forward_all_layers_smoke` calls would.
        //   Sequential `forward_layers_only` here gives that
        //   guarantee. Cost: slower than batched for large T, but
        //   commit_only's T is always `accept_len + 1` (bounded by
        //   spec_K which is small) so the overhead is bounded.
        //
        //   This does NOT affect `forward_qwen35_decode_argmax_all`'s
        //   verify path — that one still uses the batched kernel and
        //   the divergence there only matters if a draft happens to
        //   match a stale argmax, which is exceedingly rare in
        //   prompt-lookup workloads.
        for (i, &tok) in token_ids.iter().enumerate() {
            let pos = start_position
                .checked_add(i as u32)
                .ok_or_else(|| corrupt(
                    self.paths.model_dir.clone(),
                    "forward_qwen35_decode_commit_only: position overflow"
                        .into()))?;
            self.forward_layers_only(tok, pos, None)?;
        }
        Ok(())
    }

    unsafe fn forward_qwen35_tokens_batched(
        &self,
        token_ids: &[u32],
        start_position: u32,
    ) -> Result<u64> {
        let num_tokens_usize = token_ids.len();
        if num_tokens_usize == 0 {
            return Err(corrupt(
                self.paths.model_dir.clone(),
                "forward_qwen35_tokens_batched: empty token_ids".into()));
        }
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_qwen35_tokens_batched: outside_kernels absent".into()))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_qwen35_tokens_batched: stream absent".into()))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_qwen35_tokens_batched: arena absent".into()))?;
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_qwen35_tokens_batched: model absent".into()))?;
        let arch = &self.arch;
        let hidden = arch.base.hidden_size as u32;
        let vocab = arch.base.vocab_size as u32;
        let row_bytes = (hidden as usize) * 2;
        let stream_raw = stream.raw() as u64;
        let num_tokens = num_tokens_usize as u32;

        let h_residual_buf = arena.region(
            "qwen35_spec_h_residual",
            num_tokens_usize * row_bytes,
            16,
        )?.device_ptr();
        let positions_dev = arena.region(
            "qwen35_spec_positions",
            num_tokens_usize * 4,
            16,
        )?.device_ptr();
        let ctx_len_dev = arena.region(
            "qwen35_spec_ctx_len",
            4,
            4,
        )?.device_ptr();
        let token_ids_dev = arena.region(
            "qwen35_spec_token_ids",
            num_tokens_usize * 4,
            16,
        )?.device_ptr();

        let mut positions_host: Vec<u8> = Vec::with_capacity(num_tokens_usize * 4);
        for i in 0..num_tokens_usize {
            let pos = start_position
                .checked_add(i as u32)
                .ok_or_else(|| corrupt(
                    self.paths.model_dir.clone(),
                    "forward_qwen35_tokens_batched: position overflow".into()))?;
            positions_host.extend_from_slice(&(pos as i32).to_le_bytes());
        }
        let mut token_ids_host: Vec<u8> = Vec::with_capacity(num_tokens_usize * 4);
        for &t in token_ids {
            token_ids_host.extend_from_slice(&(t as i32).to_le_bytes());
        }
        let ctx_len = start_position
            .checked_add(num_tokens)
            .ok_or_else(|| corrupt(
                self.paths.model_dir.clone(),
                "forward_qwen35_tokens_batched: context length overflow".into()))?;
        let ctx_len_bytes = (ctx_len as i32).to_le_bytes();
        {
            use cudarc::driver::sys::*;
            for (dst, src, n) in [
                (positions_dev, positions_host.as_ptr(), positions_host.len()),
                (token_ids_dev, token_ids_host.as_ptr(), token_ids_host.len()),
                (ctx_len_dev, ctx_len_bytes.as_ptr(), 4usize),
            ] {
                let rc = cuMemcpyHtoDAsync_v2(
                    dst as CUdeviceptr,
                    src as *const _,
                    n,
                    stream_raw as CUstream,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 spec HtoD per-request scalars",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            }
        }

        rvllm_fused::EmbeddingGatherLaunch {
            num_tokens,
            hidden,
            vocab,
        }.launch(
            ker.fn_embedding_gather_f16,
            h_residual_buf,
            model.outside.embed_tokens.offset_bytes,
            token_ids_dev,
            stream_raw,
        )?;

        for (li, ty) in arch.base.layer_types.iter().enumerate() {
            match ty {
                rvllm_loader::LayerAttnType::Linear => {
                    self.apply_linear_attn_layer_batched(li, num_tokens, h_residual_buf)?;
                }
                rvllm_loader::LayerAttnType::Full => {
                    self.apply_full_attn_layer_batched(
                        li,
                        num_tokens,
                        start_position,
                        h_residual_buf,
                        positions_dev,
                        ctx_len_dev,
                    )?;
                }
                other => {
                    return Err(corrupt(
                        self.paths.model_dir.clone(),
                        format!("forward_qwen35_tokens_batched: layer {li} has \
                                 unsupported attn type {other:?}")));
                }
            }
            self.apply_dense_mlp_layer_batched(li, num_tokens, h_residual_buf)?;
        }
        Ok(h_residual_buf)
    }

    pub unsafe fn forward_qwen35_prefill_batched<'a>(
        &self,
        prompt_ids: &[u32],
        splices: &'a [(usize, &'a [u8])],
    ) -> Result<u32> {
        let n_prompt = prompt_ids.len();
        if n_prompt == 0 {
            return Err(corrupt(
                self.paths.model_dir.clone(),
                "forward_qwen35_prefill_batched: empty prompt".into()));
        }
        let scr = self.scratch.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_qwen35_prefill_batched: scratch absent".into()))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_qwen35_prefill_batched: outside_kernels absent".into()))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_qwen35_prefill_batched: stream absent".into()))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_qwen35_prefill_batched: arena absent".into()))?;
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_qwen35_prefill_batched: model absent".into()))?;
        let arch = &self.arch;
        let hidden = arch.base.hidden_size as u32;
        let vocab = arch.base.vocab_size as u32;
        let row_bytes = (hidden as usize) * 2;
        let stream_raw = stream.raw() as u64;
        let num_tokens = n_prompt as u32;
        let prefill_trace =
            std::env::var("RVLLM_QWEN35_PREFILL_PERF_TRACE").as_deref() == Ok("1");
        let trace_begin = if prefill_trace {
            qwen35_sync_stream(stream_raw, "qwen35 prefill trace begin")?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut trace_last = trace_begin;
        let mut trace_embed_ms: f64 = 0.0;
        let mut trace_linear_ms: f64 = 0.0;
        let mut trace_full_ms: f64 = 0.0;
        let mut trace_mlp_ms: f64 = 0.0;
        let mut trace_finalize_ms: f64 = 0.0;

        // (1) Allocate batched buffers.
        let h_residual_buf = arena.region(
            "qwen35_lmp_h_residual", n_prompt * row_bytes, 16)?.device_ptr();
        let positions_dev = arena.region(
            "qwen35_lmp_positions", n_prompt * 4, 16)?.device_ptr();
        let ctx_len_dev = arena.region(
            "qwen35_lmp_ctx_len", 4, 4)?.device_ptr();
        let token_ids_dev = arena.region(
            "qwen35_lmp_token_ids", n_prompt * 4, 16)?.device_ptr();

        // (2) Build + HtoD the small per-request scalars.
        let mut positions_host: Vec<u8> = Vec::with_capacity(n_prompt * 4);
        for i in 0..(n_prompt as i32) {
            positions_host.extend_from_slice(&i.to_le_bytes());
        }
        let mut token_ids_host: Vec<u8> = Vec::with_capacity(n_prompt * 4);
        for &t in prompt_ids {
            token_ids_host.extend_from_slice(&(t as i32).to_le_bytes());
        }
        let ctx_len_bytes: [u8; 4] = (n_prompt as i32).to_le_bytes();
        {
            use cudarc::driver::sys::*;
            for (dst, src, n) in [
                (positions_dev, positions_host.as_ptr(), positions_host.len()),
                (token_ids_dev, token_ids_host.as_ptr(), token_ids_host.len()),
                (ctx_len_dev,   ctx_len_bytes.as_ptr(),  4usize),
            ] {
                let rc = cuMemcpyHtoDAsync_v2(
                    dst as CUdeviceptr, src as *const _, n,
                    stream_raw as CUstream);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 prefill HtoD per-request scalars",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            }
        }

        // (3) Batched embed: gather num_tokens rows.
        rvllm_fused::EmbeddingGatherLaunch {
            num_tokens, hidden, vocab,
        }.launch(
            ker.fn_embedding_gather_f16,
            h_residual_buf,
            model.outside.embed_tokens.offset_bytes,
            token_ids_dev,
            stream_raw,
        )?;
        if prefill_trace {
            qwen35_sync_stream(stream_raw, "qwen35 prefill trace embed")?;
            if let Some(t0) = trace_last {
                trace_embed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "[qwen35-prefill-perf] tokens={n_prompt} stage=embed ms={trace_embed_ms:.3}"
                );
            }
            trace_last = Some(std::time::Instant::now());
        }

        // (4) Vision splices: DtoD per-slot over the [N, hidden]
        //     buffer. Each splice is (token_start, &[u8] of
        //     num_rows * row_bytes).
        for (start, data) in splices {
            if data.len() % row_bytes != 0 {
                return Err(corrupt(
                    self.paths.model_dir.clone(),
                    format!("forward_qwen35_prefill_batched: splice has \
                             {} bytes, not a multiple of row_bytes={row_bytes}",
                            data.len())));
            }
            let n_rows = data.len() / row_bytes;
            if start + n_rows > n_prompt {
                return Err(corrupt(
                    self.paths.model_dir.clone(),
                    format!("forward_qwen35_prefill_batched: splice {start}+{n_rows} \
                             past prompt_len={n_prompt}")));
            }
            use cudarc::driver::sys::*;
            let dst = h_residual_buf + ((*start) as u64) * (row_bytes as u64);
            let rc = cuMemcpyHtoDAsync_v2(
                dst as CUdeviceptr, data.as_ptr() as *const _, data.len(),
                stream_raw as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 prefill vision splice HtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (5) Layer-major dispatch.
        //
        // Per-layer arena checkpoint+restore: each layer's
        // apply_*_attn_layer_batched + apply_dense_mlp_layer_batched
        // bump-allocate ~hundreds of MiB of f32 Q/K/V/MLP scratch via
        // arena.region. Without rewinding between layers the 64 layers
        // accumulate ~tens of GiB of dead scratch and OOM on long
        // prompts. The buffers allocated BEFORE the loop
        // (h_residual_buf / positions_dev / ctx_len_dev / token_ids_dev)
        // sit BELOW the checkpoint and survive the per-layer restore.
        let layer_ck = arena.checkpoint();
        for (li, ty) in arch.base.layer_types.iter().enumerate() {
            match ty {
                rvllm_loader::LayerAttnType::Linear => {
                    self.apply_linear_attn_layer_batched(li, num_tokens, h_residual_buf)?;
                    if prefill_trace {
                        qwen35_sync_stream(stream_raw, "qwen35 prefill trace linear")?;
                        if let Some(t0) = trace_last {
                            let ms = t0.elapsed().as_secs_f64() * 1000.0;
                            trace_linear_ms += ms;
                            eprintln!(
                                "[qwen35-prefill-perf] tokens={n_prompt} layer={li} kind=linear ms={ms:.3}"
                            );
                        }
                        trace_last = Some(std::time::Instant::now());
                    }
                }
                rvllm_loader::LayerAttnType::Full => {
                    self.apply_full_attn_layer_batched(
                        li, num_tokens,
                        /* start_position */ 0,
                        h_residual_buf,
                        positions_dev,
                        ctx_len_dev,
                    )?;
                    if prefill_trace {
                        qwen35_sync_stream(stream_raw, "qwen35 prefill trace full")?;
                        if let Some(t0) = trace_last {
                            let ms = t0.elapsed().as_secs_f64() * 1000.0;
                            trace_full_ms += ms;
                            eprintln!(
                                "[qwen35-prefill-perf] tokens={n_prompt} layer={li} kind=full ms={ms:.3}"
                            );
                        }
                        trace_last = Some(std::time::Instant::now());
                    }
                }
                other => {
                    return Err(corrupt(
                        self.paths.model_dir.clone(),
                        format!("forward_qwen35_prefill_batched: layer {li} has \
                                 unsupported attn type {other:?}")));
                }
            }
            self.apply_dense_mlp_layer_batched(li, num_tokens, h_residual_buf)?;
            if prefill_trace {
                qwen35_sync_stream(stream_raw, "qwen35 prefill trace mlp")?;
                if let Some(t0) = trace_last {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    trace_mlp_ms += ms;
                    eprintln!(
                        "[qwen35-prefill-perf] tokens={n_prompt} layer={li} kind=mlp ms={ms:.3}"
                    );
                }
                trace_last = Some(std::time::Instant::now());
            }
            // Rewind this layer's bump-allocated scratch before the
            // next layer claims the same arena region.
            unsafe { arena.restore(layer_ck); }
        }

        // (6) Move last row of h_residual into the single-row
        //     scratch so forward_finalize_argmax can run lm_head
        //     unchanged. Last row offset = (n_prompt-1) * row_bytes.
        {
            use cudarc::driver::sys::*;
            let last_src = h_residual_buf + ((n_prompt - 1) as u64) * (row_bytes as u64);
            let rc = cuMemcpyDtoDAsync_v2(
                scr.h_residual_ptr, last_src, row_bytes,
                stream_raw as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 prefill DtoD last-row → scratch",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (7) Final norm + lm_head + argmax — reuses the existing
        //     single-row finisher.
        let predicted = self.forward_finalize_argmax()?;
        if prefill_trace {
            qwen35_sync_stream(stream_raw, "qwen35 prefill trace finalize")?;
            if let Some(t0) = trace_last {
                trace_finalize_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            let total_ms = trace_begin
                .map(|t0| t0.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            eprintln!(
                "[qwen35-prefill-perf] tokens={n_prompt} summary total_ms={total_ms:.3} \
                 embed_ms={trace_embed_ms:.3} linear_ms={trace_linear_ms:.3} \
                 full_ms={trace_full_ms:.3} mlp_ms={trace_mlp_ms:.3} \
                 finalize_ms={trace_finalize_ms:.3}"
            );
        }
        Ok(predicted)
    }

    /// Phase 3-a-v: borrow bundle for the shared Qwen-VL ViT
    /// forward path. Mirrors `Qwen36Bringup::vision_deps`; the
    /// free fn that consumes it lives in
    /// `crate::qwen_vision_forward` and is shared between the two
    /// bringups. Returns a clean error if the vision tower wasn't
    /// loaded (no `model.visual.*` keys in the checkpoint).
    pub fn vision_deps(
        &self,
    ) -> Result<crate::qwen_vision_forward::QwenVisionDeps<'_>> {
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "vision_deps: model absent".into()))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "vision_deps: arena absent".into()))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "vision_deps: stream absent".into()))?;
        let cublaslt = self.cublaslt.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "vision_deps: cublaslt absent".into()))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "vision_deps: outside_kernels absent".into()))?;
        let vision = model.vision.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "vision_deps: model.vision not loaded — checkpoint lacks \
             model.visual.* tensors".into()))?;
        Ok(crate::qwen_vision_forward::QwenVisionDeps {
            vision,
            arena,
            stream,
            cublaslt,
            fn_layernorm_inplace_f16: ker.fn_layernorm_inplace_f16,
            fn_gelu_tanh_f16: ker.fn_gelu_tanh_f16,
            fn_softmax_row_f16: ker.fn_softmax_row_f16,
            fn_vit_rotary_2d_f16: ker.fn_vit_rotary_2d_f16,
            fn_vit_pos_embed_interp_f16: ker.fn_vit_pos_embed_interp_f16,
            fn_scale_inplace_f16: ker.fn_scale_inplace_f16,
            fn_transpose_2d_f16: ker.fn_transpose_2d_f16,
            fn_add_bias_f16: ker.fn_add_bias_f16,
            fn_cast_f32_to_f16: ker.fn_cast_f32_to_f16,
            fn_extract_head_f16: ker.fn_extract_head_f16,
            fn_scatter_head_f16: ker.fn_scatter_head_f16,
            fn_softmax_row_f32_to_f16: ker.fn_softmax_row_f32_to_f16,
            fn_transpose_heads_v_f16: ker.fn_transpose_heads_v_f16,
            fn_scatter_heads_f16: ker.fn_scatter_heads_f16,
            fn_scale_inplace_f32: ker.fn_scale_inplace_f32,
            fn_vector_add_f16: ker.fn_vector_add_f16,
        })
    }

    /// Phase 3-a-iii: Qwen-VL ViT forward for Qwen 3.5.
    ///
    /// Thin wrapper around `crate::qwen_vision_forward::forward_qwen_vision`
    /// — the same shared free fn that powers Qwen 3.6 vision.
    /// Drives the 27-block ViT + PatchMerger over a single image
    /// (PNG/JPEG/WebP) and returns the f16 embeddings ready for
    /// splice into the text-side hidden buffer after the embed
    /// gather (Phase 4 / Phase 3-a-vi).
    pub fn forward_qwen_vision(
        &self,
        image_bytes: &[u8],
    ) -> Result<crate::qwen36_bring_up::VisionForwardOutput> {
        let deps = self.vision_deps()?;
        crate::qwen_vision_forward::forward_qwen_vision(&deps, image_bytes)
    }

    /// Phase 2c-C-b: linear-attn (Gated DeltaNet) forward block.
    ///
    /// Port of qwen36::apply_layer_linear_attn for the Qwen 3.5
    /// dense family. 13-op kernel chain:
    ///
    ///   1. input_layernorm(h_residual → h_work)
    ///   2. in_proj_qkv FP8 GEMV       → qkv [conv_dim]
    ///   3. conv_state_advance + causal_conv1d → conv_out
    ///   4. silu_l2_gqa                → q, k (GQA-expanded), v
    ///   5. alpha_beta                 → alpha, beta (per v-head, f32)
    ///   6. gated_delta_rule_decode    → readout, state update
    ///   7. in_proj_z FP8 GEMV         → z [value_dim]
    ///   8. rmsnorm_gated              → gated [value_dim]
    ///   9. out_proj FP8 GEMV          → out [hidden]
    ///  10. h_residual += out
    ///
    /// All intermediates allocate via the arena; the persistent
    /// state lives on `linear_state` (delta-rule SSM state + conv
    /// state).
    pub unsafe fn apply_linear_attn_layer(&self, layer_idx: usize) -> Result<()> {
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer: model absent".into()))?;
        let scr = self.scratch.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer: scratch absent".into()))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer: outside_kernels absent".into()))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer: stream absent".into()))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer: arena absent".into()))?;
        let la_dims = self.la_dims.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer: la_dims absent".into()))?;
        let linear_state = self.linear_state.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer: linear_state absent".into()))?;
        let layer = model.layers.get(layer_idx).ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            format!("apply_linear_attn_layer: layer {layer_idx} out of range"),
        ))?;
        let la = match &layer.attn {
            rvllm_loader::qwen35_weights::Qwen35LayerAttn::Linear(l) => l,
            rvllm_loader::qwen35_weights::Qwen35LayerAttn::Full(_) => {
                return Err(corrupt(
                    self.paths.model_dir.clone(),
                    format!("apply_linear_attn_layer: layer {layer_idx} is \
                             full-attention; use apply_full_attn_qkv_only"),
                ));
            }
        };
        let linear_idx = linear_state.layer_idx_to_linear_idx
            .get(layer_idx).copied().flatten()
            .ok_or_else(|| corrupt(
                self.paths.model_dir.clone(),
                format!("apply_linear_attn_layer: layer {layer_idx} not \
                         linear-attn"),
            ))?;

        let stream_raw = stream.raw() as u64;
        let arch = &self.arch;
        let hidden = arch.base.hidden_size as i32;
        let hidden_u = hidden as u32;
        let hidden_bytes = (hidden as usize) * 2;
        let eps = arch.base.rms_norm_eps;
        let num_k_heads = la_dims.num_k_heads;
        let num_v_heads = la_dims.num_v_heads;
        let head_k_dim = la_dims.head_k_dim;
        let head_v_dim = la_dims.head_v_dim;
        let key_dim = la_dims.key_dim;
        let v_per_k = la_dims.v_per_k;
        let conv_dim = la_dims.conv_dim;
        let ks = la_dims.conv_kernel_dim;
        let fp8_gemv_fn = ker.fn_fp8_gemv_wpr_native_f16in.ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer: fp8_gemv_wpr_native_f16in unavailable".into(),
        ))?;

        // (1) input_layernorm: copy h_residual → h_work, then
        // rmsnorm in-place. We need a clean copy so the residual at
        // step (10) still sees the un-normed pre-layer value.
        {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                scr.h_work_ptr,
                scr.h_residual_ptr,
                hidden_bytes,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 la copy h_residual->h_work",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        {
            use cudarc::driver::sys::*;
            let mut x = scr.h_work_ptr;
            let mut gamma = la.input_layernorm.offset_bytes;
            let mut eps_a = eps;
            let mut hd = hidden;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut gamma) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_a) as *mut f32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_rmsnorm_inplace_f16.raw() as CUfunction,
                1, 1, 1,
                (hidden_u).min(1024), 1, 1, 32 * 4,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 la input_layernorm",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (2) in_proj_qkv FP8 GEMV → qkv [conv_dim].
        let qkv_n = conv_dim as u32;
        let qkv_bytes = conv_dim * 2;
        let qkv_region = arena.region("qwen35_la_qkv", qkv_bytes, 16)?;
        let qkv_bs = la.in_proj_qkv.blockscale_ptr.ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer: in_proj_qkv blockscale missing".into()))?;
        rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
            m: 1, n: qkv_n, k: hidden_u,
        }.launch(
            fp8_gemv_fn,
            qkv_region.device_ptr(),
            la.in_proj_qkv.offset_bytes,
            qkv_bs,
            scr.h_work_ptr,
            stream_raw,
        )?;

        // (3a) conv_state_advance: assemble conv_in = [ks-1 from
        // state | cur qkv], rotate state to drop oldest + append cur.
        let conv_in_bytes = ks * conv_dim * 2;
        let conv_in_region = arena.region("qwen35_la_cin", conv_in_bytes, 16)?;
        let conv_state_p = linear_state.conv_state_ptr(linear_idx);
        {
            use cudarc::driver::sys::*;
            let mut conv_in = conv_in_region.device_ptr();
            let mut state = conv_state_p;
            let mut cur = qkv_region.device_ptr();
            let mut ts_i: i32 = conv_dim as i32;
            let args = [
                (&mut conv_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut state) as *mut u64 as *mut core::ffi::c_void,
                (&mut cur) as *mut u64 as *mut core::ffi::c_void,
                (&mut ts_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((conv_dim as u32 + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                ker.fn_conv_state_advance_f16.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 la conv_state_advance",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (3b) causal_conv1d: conv_in [ks, conv_dim] × conv1d w → conv_out.
        let conv_out_region = arena.region("qwen35_la_cout", qkv_bytes, 16)?;
        {
            use cudarc::driver::sys::*;
            let mut output = conv_out_region.device_ptr();
            let mut input = conv_in_region.device_ptr();
            let mut weight = la.conv1d.offset_bytes;
            let mut sl: i32 = 1;
            let mut ch: i32 = conv_dim as i32;
            let mut k_arg: i32 = ks as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut weight) as *mut u64 as *mut core::ffi::c_void,
                (&mut sl) as *mut i32 as *mut core::ffi::c_void,
                (&mut ch) as *mut i32 as *mut core::ffi::c_void,
                (&mut k_arg) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid_x: u32 = (conv_dim as u32 + block - 1) / block;
            let rc = cuLaunchKernel(
                ker.fn_causal_conv1d_f16.raw() as CUfunction,
                grid_x, 1, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 la causal_conv1d_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (4) silu_l2_gqa: conv_out → q [num_v_heads, head_k_dim],
        // k [num_v_heads, head_k_dim] (GQA-expanded), v [num_v_heads,
        // head_v_dim] (silu+pack).
        let qk_bytes_pre = num_v_heads * head_k_dim * 2;
        let v_bytes_pre  = num_v_heads * head_v_dim * 2;
        let q_region = arena.region("qwen35_la_q", qk_bytes_pre, 16)?;
        let k_region = arena.region("qwen35_la_k", qk_bytes_pre, 16)?;
        let v_region = arena.region("qwen35_la_v", v_bytes_pre, 16)?;
        {
            use cudarc::driver::sys::*;
            let mut q_out = q_region.device_ptr();
            let mut k_out = k_region.device_ptr();
            let mut v_out = v_region.device_ptr();
            let mut conv_p = conv_out_region.device_ptr();
            let mut vus_i: i32 = num_v_heads as i32;
            let mut hkd_i: i32 = head_k_dim as i32;
            let mut hvd_i: i32 = head_v_dim as i32;
            let mut kd_i:  i32 = key_dim as i32;
            let mut nvh:   i32 = num_v_heads as i32;
            let mut vpk:   i32 = v_per_k as i32;
            let args = [
                (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut conv_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut vus_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut hkd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut hvd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut kd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut nvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut vpk) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = head_k_dim.max(head_v_dim) as u32;
            let rc = cuLaunchKernel(
                ker.fn_qwen_linear_silu_l2_gqa_f16.raw() as CUfunction,
                num_v_heads as u32, 1, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 la silu_l2_gqa",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (5) alpha_beta: read in_proj_a + in_proj_b weights against
        // the post-norm input → alpha[vus] + beta[vus] f32.
        let alpha_region = arena.region("qwen35_la_alpha", num_v_heads * 4, 16)?;
        let beta_region  = arena.region("qwen35_la_beta",  num_v_heads * 4, 16)?;
        {
            use cudarc::driver::sys::*;
            let mut a_out = alpha_region.device_ptr();
            let mut b_out = beta_region.device_ptr();
            let mut a_w_p = la.in_proj_a.offset_bytes;
            let mut b_w_p = la.in_proj_b.offset_bytes;
            let mut a_log_p = la.a_log.offset_bytes;
            let mut dt_bias_p = la.dt_bias.offset_bytes;
            let mut in_p = scr.h_work_ptr;
            let mut vus_i: i32 = num_v_heads as i32;
            let mut h_i: i32 = hidden;
            let args = [
                (&mut a_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut b_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut a_w_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut b_w_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut a_log_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut dt_bias_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut in_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut vus_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut h_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256u32.min(hidden_u).max(1);
            let rc = cuLaunchKernel(
                ker.fn_qwen_linear_alpha_beta_f16.raw() as CUfunction,
                num_v_heads as u32, 1, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 la alpha_beta",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (6) gated_delta_rule_decode: forget + delta correction +
        // state update + readout. State is updated in-place.
        let layer_state_ptr = linear_state.delta_state_ptr(linear_idx);
        let scale = 1.0_f32 / (head_k_dim as f32).sqrt();
        let v_bytes = num_v_heads * head_v_dim * 2;
        let readout_region = arena.region("qwen35_la_readout", v_bytes, 16)?;
        {
            use cudarc::driver::sys::*;
            let mut state = layer_state_ptr;
            let mut q_ptr = q_region.device_ptr();
            let mut k_ptr = k_region.device_ptr();
            let mut v_ptr = v_region.device_ptr();
            let mut a_ptr = alpha_region.device_ptr();
            let mut b_ptr = beta_region.device_ptr();
            let mut o_ptr = readout_region.device_ptr();
            let mut scale_arg = scale;
            let mut hvd_i = head_v_dim as i32;
            let mut hkd_i = head_k_dim as i32;
            let args = [
                (&mut state) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut a_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut b_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut o_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut scale_arg) as *mut f32 as *mut core::ffi::c_void,
                (&mut hvd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut hkd_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let smem: u32 = (2 * head_k_dim as u32 + head_v_dim as u32) * 4;
            let rc = cuLaunchKernel(
                ker.fn_gated_delta_rule_decode_f16.raw() as CUfunction,
                num_v_heads as u32, 1, 1,
                head_v_dim as u32, 1, 1,
                smem,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 la gated_delta_rule_decode_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (7) in_proj_z FP8 GEMV → z [value_dim].
        let z_n = la.in_proj_z.shape[0] as u32;
        let z_bs = la.in_proj_z.blockscale_ptr.ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer: in_proj_z blockscale missing".into()))?;
        let z_region = arena.region("qwen35_la_z", (z_n as usize) * 2, 16)?;
        rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
            m: 1, n: z_n, k: hidden_u,
        }.launch(
            fp8_gemv_fn,
            z_region.device_ptr(),
            la.in_proj_z.offset_bytes,
            z_bs,
            scr.h_work_ptr,
            stream_raw,
        )?;

        // (8) rmsnorm_gated: per-v-head RMSNorm + silu(z) gate fused
        // → gated [value_dim] f16.
        let gated_region = arena.region("qwen35_la_gated", v_bytes, 16)?;
        {
            use cudarc::driver::sys::*;
            let mut g_out = gated_region.device_ptr();
            let mut r_in = readout_region.device_ptr();
            let mut z_in = z_region.device_ptr();
            let mut gamma_p = la.norm.offset_bytes;
            let mut vus_i: i32 = num_v_heads as i32;
            let mut hvd_i: i32 = head_v_dim as i32;
            let mut eps_f: f32 = 1e-6;
            let args = [
                (&mut g_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut r_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut z_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut gamma_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut vus_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut hvd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut eps_f) as *mut f32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_qwen_linear_rmsnorm_gated_f16.raw() as CUfunction,
                num_v_heads as u32, 1, 1,
                head_v_dim as u32, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 la rmsnorm_gated",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // (9-10) Fused out_proj + residual add (Phase 8 other-models
        // fusion, 2026-05-23). Default-on; opt-out via
        // `RVLLM_QWEN35_FP8_GEMV_RESIDUAL_FUSED=0`.
        let out_n = la.out_proj.shape[0] as u32;
        let out_k = la.out_proj.shape[1] as u32;
        let out_bs = la.out_proj.blockscale_ptr.ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer: out_proj blockscale missing".into()))?;
        let fused = std::env::var("RVLLM_QWEN35_FP8_GEMV_RESIDUAL_FUSED")
            .map(|s| s != "0")
            .unwrap_or(true);
        unsafe {
            use cudarc::driver::sys::*;
            if fused {
                let mut h_resid = scr.h_residual_ptr;
                let mut w_ptr = la.out_proj.offset_bytes;
                let mut s_ptr = out_bs;
                let mut x_ptr = gated_region.device_ptr();
                let mut m_i: i32 = 1;
                let mut n_i: i32 = out_n as i32;
                let mut k_i: i32 = out_k as i32;
                let mut ncb: i32 = ((out_k as i32) + 127) / 128;
                let args = [
                    (&mut h_resid) as *mut u64 as *mut core::ffi::c_void,
                    (&mut w_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut s_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut x_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut ncb) as *mut i32 as *mut core::ffi::c_void,
                ];
                let grid = ((out_n + 7) / 8, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    ker.fn_fp8_gemv_f16in_residual_add.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    256, 1, 1, 0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 fp8_gemv_f16in_residual_add (la out_proj+resid) launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            } else {
                let arena = self.arena.as_ref().ok_or_else(|| corrupt(
                    self.paths.model_dir.clone(),
                    "qwen35 unfused la out_proj+resid: arena unset".into()))?;
                let temp = arena.region("qwen35_la_outproj_unfused_temp",
                                        (out_n as usize) * 2, 16)?;
                let gemv_fn = ker.fn_fp8_gemv_wpr_native_f16in.ok_or_else(|| corrupt(
                    self.paths.model_dir.clone(),
                    "qwen35 unfused la out_proj+resid: fn_fp8_gemv_wpr_native_f16in unloaded".into()))?;
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m: 1, n: out_n, k: out_k,
                }.launch(
                    gemv_fn,
                    temp.device_ptr(),
                    la.out_proj.offset_bytes,
                    out_bs,
                    gated_region.device_ptr(),
                    stream_raw,
                )?;
                let mut dst = scr.h_residual_ptr;
                let mut src = temp.device_ptr();
                let mut n_elem: i32 = out_n as i32;
                let v_args = [
                    (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                    (&mut src) as *mut u64 as *mut core::ffi::c_void,
                    (&mut n_elem) as *mut i32 as *mut core::ffi::c_void,
                ];
                let v_block: u32 = 256;
                let v_grid: u32 = ((out_n + v_block - 1) / v_block).max(1);
                let rc = cuLaunchKernel(
                    ker.fn_vector_add_f16.raw() as CUfunction,
                    v_grid, 1, 1,
                    v_block, 1, 1,
                    0,
                    stream_raw as CUstream,
                    v_args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 unfused la out_proj+resid vector_add launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            }
        }
        Ok(())
    }

    /// Phase #1-b: batched linear-attn (Gated DeltaNet) over N
    /// tokens. Ported from `Qwen36Bringup::apply_layer_linear_attn_batched`
    /// with Qwen 3.5 dim numbers (kh=16, vh=48, hkd=hvd=128 →
    /// key_dim=2048, value_dim=6144, conv_dim=10240). 10-op chain
    /// identical to per-token; just runs at M=N instead of M=1:
    ///
    ///   1) rmsnorm(input_layernorm) on [N, hidden]
    ///   2) in_proj_qkv batched FP8 GEMV (m=N)
    ///   3) conv_state_advance_batched_f16 — assembles
    ///      [N+ks-1, conv_dim] history block + rotates the
    ///      persistent (ks-1) state to the tail of the chunk
    ///   3b) causal_conv1d_f16 at seq_len=N
    ///   4) silu_l2_gqa over (vus, N)
    ///   5) alpha_beta over (vus, N)
    ///   6) gated_delta_rule_prefill_f16 — one launch advances
    ///      the SSM state across all N tokens and writes the
    ///      [N, vus, hvd] readout
    ///   7) in_proj_z batched FP8 GEMV (m=N)
    ///   8) rmsnorm_gated over (vus, N)
    ///   9) out_proj batched FP8 GEMV (m=N)
    ///  10) h_residual += out elementwise over N*hidden
    ///
    /// Caller passes the [N, hidden] h_residual buffer directly.
    /// All scratch is per-call via the arena. The persistent
    /// linear-attn SSM state + conv1d state both advance
    /// correctly because the batched kernels handle the
    /// recurrence internally — same correctness contract as
    /// qwen36's apply_layer_linear_attn_batched.
    pub unsafe fn apply_linear_attn_layer_batched(
        &self,
        layer_idx: usize,
        num_tokens: u32,
        h_residual_buf: u64,
    ) -> Result<()> {
        if num_tokens == 0 {
            return Ok(());
        }
        if num_tokens == 1 {
            // Degenerate case: delegate to the per-token path so
            // we don't accidentally fork numerics between the
            // batched and per-token kernels for the decode case.
            // The caller must arrange `scr.h_residual_ptr` to
            // alias the single-row `h_residual_buf` before
            // calling — this is enforced by the layer-major
            // driver above. For now, returning Ok forces the
            // caller to use the per-token path explicitly.
            return self.apply_linear_attn_layer(layer_idx);
        }

        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer_batched: model absent".into()))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer_batched: outside_kernels absent".into()))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer_batched: stream absent".into()))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer_batched: arena absent".into()))?;
        let la_dims = self.la_dims.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer_batched: la_dims absent".into()))?;
        let linear_state = self.linear_state.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer_batched: linear_state absent".into()))?;
        let layer = model.layers.get(layer_idx).ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            format!("apply_linear_attn_layer_batched: layer {layer_idx} out of range")))?;
        let la = match &layer.attn {
            rvllm_loader::qwen35_weights::Qwen35LayerAttn::Linear(l) => l,
            rvllm_loader::qwen35_weights::Qwen35LayerAttn::Full(_) => {
                return Err(corrupt(
                    self.paths.model_dir.clone(),
                    format!("apply_linear_attn_layer_batched: layer {layer_idx} is full-attn"),
                ));
            }
        };
        let linear_idx = linear_state.layer_idx_to_linear_idx
            .get(layer_idx).copied().flatten()
            .ok_or_else(|| corrupt(
                self.paths.model_dir.clone(),
                format!("apply_linear_attn_layer_batched: layer {layer_idx} not linear-attn")))?;

        let stream_raw = stream.raw() as u64;
        let arch = &self.arch;
        let hidden = arch.base.hidden_size as i32;
        let hidden_u = hidden as u32;
        let eps = arch.base.rms_norm_eps;
        let num_k_heads = la_dims.num_k_heads;
        let num_v_heads = la_dims.num_v_heads;
        let head_k_dim = la_dims.head_k_dim;
        let head_v_dim = la_dims.head_v_dim;
        let key_dim = la_dims.key_dim;
        let v_per_k = la_dims.v_per_k;
        let conv_dim = la_dims.conv_dim;
        let ks = la_dims.conv_kernel_dim;
        let n = num_tokens as usize;
        let h = hidden as usize;
        let qk_bytes_per_token = num_v_heads * head_k_dim * 2;
        let v_bytes_per_token  = num_v_heads * head_v_dim * 2;
        let fp8_gemv_fn = ker.fn_fp8_gemv_wpr_native_f16in.ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer_batched: fp8_gemv_wpr_native_f16in unavailable".into()))?;
        let _ = num_k_heads;
        let cutlass_linear_min_tokens =
            std::env::var("RVLLM_QWEN35_LINEAR_CUTLASS_MIN_TOKENS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(128);
        let cutlass_linear_requested =
            crate::gemma4_bring_up::parse_truthy_env("RVLLM_QWEN35_LINEAR_CUTLASS_SM120")
                .unwrap_or(false)
                && num_tokens >= cutlass_linear_min_tokens;
        let cutlass_linear_lib = if cutlass_linear_requested {
            match &self.cutlass {
                CutlassBackend::SoSm120(lib) => Some(lib),
                _ => None,
            }
        } else {
            None
        };
        let linear_trace =
            std::env::var("RVLLM_QWEN35_LINEAR_PERF_TRACE").as_deref() == Ok("1");
        let trace_begin = if linear_trace {
            qwen35_sync_stream(stream_raw, "qwen35 linear trace begin")?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut trace_last = trace_begin;
        let mut trace_norm_ms = 0.0_f64;
        let mut trace_qkv_ms = 0.0_f64;
        let mut trace_conv_state_ms = 0.0_f64;
        let mut trace_conv1d_ms = 0.0_f64;
        let mut trace_silu_l2_ms = 0.0_f64;
        let mut trace_alpha_beta_ms = 0.0_f64;
        let mut trace_delta_ms = 0.0_f64;
        let mut trace_z_ms = 0.0_f64;
        let mut trace_rms_gated_ms = 0.0_f64;
        let mut trace_out_ms = 0.0_f64;
        let mut trace_residual_ms = 0.0_f64;

        // (1) RMSNorm a [N, hidden] copy of the chunk.
        let normed_bytes = n * h * 2;
        let normed = arena.region("qwen35_blattn_normed", normed_bytes, 16)?.device_ptr();
        {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                normed, h_residual_buf, normed_bytes, stream_raw as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_blattn DtoD normed",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens, hidden: hidden_u, eps,
        }.launch(
            ker.fn_rmsnorm_inplace_f16,
            normed,
            la.input_layernorm.offset_bytes,
            stream_raw,
        )?;
        if linear_trace {
            qwen35_sync_stream(stream_raw, "qwen35 linear trace norm")?;
            if let Some(t0) = trace_last {
                trace_norm_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }

        // (2) in_proj_qkv [N, hidden] → [N, conv_dim].
        let qkv_n = conv_dim as u32;
        let qkv_bs = la.in_proj_qkv.blockscale_ptr.ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer_batched: in_proj_qkv blockscale missing".into()))?;
        let qkv_region = arena.region("qwen35_blattn_qkv", n * conv_dim * 2, 16)?.device_ptr();
        if let Some(lib) = cutlass_linear_lib {
            self.qwen35_fp8_cutlass_blockscale_sm120(
                ker,
                lib,
                qkv_region,
                la.in_proj_qkv.offset_bytes,
                qkv_bs,
                normed,
                num_tokens,
                qkv_n,
                hidden_u,
                stream_raw,
                "qwen35_blattn_qkv_in_fp8",
            )?;
        } else {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m: num_tokens, n: qkv_n, k: hidden_u,
            }.launch(
                fp8_gemv_fn, qkv_region,
                la.in_proj_qkv.offset_bytes, qkv_bs, normed, stream_raw,
            )?;
        }
        if linear_trace {
            qwen35_sync_stream(stream_raw, "qwen35 linear trace qkv")?;
            if let Some(t0) = trace_last {
                trace_qkv_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }

        // (3) conv1d batched state advance + causal_conv1d.
        let conv_in_bytes  = (n + (ks - 1)) * conv_dim * 2;
        let conv_out_bytes = n * conv_dim * 2;
        let conv_in  = arena.region("qwen35_blattn_cin",  conv_in_bytes, 16)?.device_ptr();
        let conv_out = arena.region("qwen35_blattn_cout", conv_out_bytes, 16)?.device_ptr();
        let conv_state_p = linear_state.conv_state_ptr(linear_idx);
        {
            use cudarc::driver::sys::*;
            let mut conv_in_a = conv_in;
            let mut state = conv_state_p;
            let mut cur = qkv_region;
            let mut ts_i: i32 = conv_dim as i32;
            let mut nt_i: i32 = num_tokens as i32;
            let args = [
                (&mut conv_in_a) as *mut u64 as *mut core::ffi::c_void,
                (&mut state) as *mut u64 as *mut core::ffi::c_void,
                (&mut cur) as *mut u64 as *mut core::ffi::c_void,
                (&mut ts_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut nt_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((conv_dim as u32 + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                ker.fn_conv_state_advance_batched_f16.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_blattn conv_state_advance_batched",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        if linear_trace {
            qwen35_sync_stream(stream_raw, "qwen35 linear trace conv_state")?;
            if let Some(t0) = trace_last {
                trace_conv_state_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }
        {
            use cudarc::driver::sys::*;
            let mut output = conv_out;
            let mut input = conv_in;
            let mut weight = la.conv1d.offset_bytes;
            let mut sl: i32 = num_tokens as i32;
            let mut ch: i32 = conv_dim as i32;
            let mut k_arg: i32 = ks as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut weight) as *mut u64 as *mut core::ffi::c_void,
                (&mut sl) as *mut i32 as *mut core::ffi::c_void,
                (&mut ch) as *mut i32 as *mut core::ffi::c_void,
                (&mut k_arg) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid_x: u32 = (conv_dim as u32 + block - 1) / block;
            let rc = cuLaunchKernel(
                ker.fn_causal_conv1d_f16.raw() as CUfunction,
                grid_x, num_tokens, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_blattn causal_conv1d_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        if linear_trace {
            qwen35_sync_stream(stream_raw, "qwen35 linear trace conv1d")?;
            if let Some(t0) = trace_last {
                trace_conv1d_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }

        // (4) silu_l2_gqa over (vus, N).
        let q_region = arena.region("qwen35_blattn_q", n * qk_bytes_per_token, 16)?.device_ptr();
        let k_region = arena.region("qwen35_blattn_k", n * qk_bytes_per_token, 16)?.device_ptr();
        let v_region = arena.region("qwen35_blattn_v", n * v_bytes_per_token,  16)?.device_ptr();
        {
            use cudarc::driver::sys::*;
            let mut q_out = q_region;
            let mut k_out = k_region;
            let mut v_out = v_region;
            let mut conv_p = conv_out;
            let mut vus_i: i32 = num_v_heads as i32;
            let mut hkd_i: i32 = head_k_dim as i32;
            let mut hvd_i: i32 = head_v_dim as i32;
            let mut kd_i:  i32 = key_dim as i32;
            let mut nvh:   i32 = num_v_heads as i32;
            let mut vpk:   i32 = v_per_k as i32;
            let args = [
                (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut conv_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut vus_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut hkd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut hvd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut kd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut nvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut vpk) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = head_k_dim.max(head_v_dim) as u32;
            let rc = cuLaunchKernel(
                ker.fn_qwen_linear_silu_l2_gqa_f16.raw() as CUfunction,
                num_v_heads as u32, num_tokens, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_blattn silu_l2_gqa",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        if linear_trace {
            qwen35_sync_stream(stream_raw, "qwen35 linear trace silu_l2")?;
            if let Some(t0) = trace_last {
                trace_silu_l2_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }

        // (5) alpha_beta over (vus, N).
        let alpha_region = arena.region("qwen35_blattn_alpha", n * num_v_heads * 4, 16)?.device_ptr();
        let beta_region  = arena.region("qwen35_blattn_beta",  n * num_v_heads * 4, 16)?.device_ptr();
        {
            use cudarc::driver::sys::*;
            let mut a_out = alpha_region;
            let mut b_out = beta_region;
            let mut a_w_p = la.in_proj_a.offset_bytes;
            let mut b_w_p = la.in_proj_b.offset_bytes;
            let mut a_log_p = la.a_log.offset_bytes;
            let mut dt_bias_p = la.dt_bias.offset_bytes;
            let mut in_p = normed;
            let mut vus_i: i32 = num_v_heads as i32;
            let mut h_i: i32 = hidden;
            let args = [
                (&mut a_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut b_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut a_w_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut b_w_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut a_log_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut dt_bias_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut in_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut vus_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut h_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256u32.min(hidden_u).max(1);
            let rc = cuLaunchKernel(
                ker.fn_qwen_linear_alpha_beta_f16.raw() as CUfunction,
                num_v_heads as u32, num_tokens, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_blattn alpha_beta",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        if linear_trace {
            qwen35_sync_stream(stream_raw, "qwen35 linear trace alpha_beta")?;
            if let Some(t0) = trace_last {
                trace_alpha_beta_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }

        // (6) gated_delta_rule_prefill — one launch over all N
        //     tokens; advances SSM state internally.
        let layer_state_ptr = linear_state.delta_state_ptr(linear_idx);
        let scale = 1.0_f32 / (head_k_dim as f32).sqrt();
        let readout_region = arena.region("qwen35_blattn_readout", n * v_bytes_per_token, 16)?.device_ptr();
        {
            use cudarc::driver::sys::*;
            let mut state = layer_state_ptr;
            let mut q_ptr = q_region;
            let mut k_ptr = k_region;
            let mut v_ptr = v_region;
            let mut a_ptr = alpha_region;
            let mut b_ptr = beta_region;
            let mut o_ptr = readout_region;
            let mut scale_arg = scale;
            let mut nt_i = num_tokens as i32;
            let mut nvh_i = num_v_heads as i32;
            let mut hvd_i = head_v_dim as i32;
            let mut hkd_i = head_k_dim as i32;
            let args = [
                (&mut state) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut a_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut b_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut o_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut scale_arg) as *mut f32 as *mut core::ffi::c_void,
                (&mut nt_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut nvh_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut hvd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut hkd_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let smem: u32 = (2 * head_k_dim as u32 + head_v_dim as u32) * 4;
            let rc = cuLaunchKernel(
                ker.fn_gated_delta_rule_prefill_f16.raw() as CUfunction,
                num_v_heads as u32, 1, 1,
                head_v_dim as u32, 1, 1, smem,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_blattn gated_delta_rule_prefill",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        if linear_trace {
            qwen35_sync_stream(stream_raw, "qwen35 linear trace delta")?;
            if let Some(t0) = trace_last {
                trace_delta_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }

        // (7) in_proj_z batched FP8 GEMV [N, hidden] → [N, value_dim].
        let z_n = la.in_proj_z.shape[0] as u32;
        let z_bs = la.in_proj_z.blockscale_ptr.ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer_batched: in_proj_z blockscale missing".into()))?;
        let z_region = arena.region("qwen35_blattn_z", n * (z_n as usize) * 2, 16)?.device_ptr();
        if let Some(lib) = cutlass_linear_lib {
            self.qwen35_fp8_cutlass_blockscale_sm120(
                ker,
                lib,
                z_region,
                la.in_proj_z.offset_bytes,
                z_bs,
                normed,
                num_tokens,
                z_n,
                hidden_u,
                stream_raw,
                "qwen35_blattn_z_in_fp8",
            )?;
        } else {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m: num_tokens, n: z_n, k: hidden_u,
            }.launch(
                fp8_gemv_fn, z_region,
                la.in_proj_z.offset_bytes, z_bs, normed, stream_raw,
            )?;
        }
        if linear_trace {
            qwen35_sync_stream(stream_raw, "qwen35 linear trace z")?;
            if let Some(t0) = trace_last {
                trace_z_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }

        // (8) rmsnorm_gated over (vus, N).
        let gated_region = arena.region("qwen35_blattn_gated", n * num_v_heads * head_v_dim * 2, 16)?.device_ptr();
        {
            use cudarc::driver::sys::*;
            let mut g_out = gated_region;
            let mut r_in = readout_region;
            let mut z_in = z_region;
            let mut gamma_p = la.norm.offset_bytes;
            let mut vus_i: i32 = num_v_heads as i32;
            let mut hvd_i: i32 = head_v_dim as i32;
            let mut eps_f: f32 = 1e-6;
            let args = [
                (&mut g_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut r_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut z_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut gamma_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut vus_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut hvd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut eps_f) as *mut f32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_qwen_linear_rmsnorm_gated_f16.raw() as CUfunction,
                num_v_heads as u32, num_tokens, 1,
                head_v_dim as u32, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_blattn rmsnorm_gated",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        if linear_trace {
            qwen35_sync_stream(stream_raw, "qwen35 linear trace rms_gated")?;
            if let Some(t0) = trace_last {
                trace_rms_gated_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }

        // (9) out_proj batched FP8 GEMV [N, value_dim] → [N, hidden].
        let out_n = la.out_proj.shape[0] as u32;
        let out_k = la.out_proj.shape[1] as u32;
        let out_bs = la.out_proj.blockscale_ptr.ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer_batched: out_proj blockscale missing".into()))?;
        let out_region = arena.region("qwen35_blattn_out", n * (out_n as usize) * 2, 16)?.device_ptr();
        if let Some(lib) = cutlass_linear_lib {
            self.qwen35_fp8_cutlass_blockscale_sm120(
                ker,
                lib,
                out_region,
                la.out_proj.offset_bytes,
                out_bs,
                gated_region,
                num_tokens,
                out_n,
                out_k,
                stream_raw,
                "qwen35_blattn_out_in_fp8",
            )?;
        } else {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m: num_tokens, n: out_n, k: out_k,
            }.launch(
                fp8_gemv_fn, out_region,
                la.out_proj.offset_bytes, out_bs, gated_region, stream_raw,
            )?;
        }
        if linear_trace {
            qwen35_sync_stream(stream_raw, "qwen35 linear trace out")?;
            if let Some(t0) = trace_last {
                trace_out_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }

        // (10) h_residual += out elementwise over N*hidden.
        {
            use cudarc::driver::sys::*;
            let n_elem = (n as i32) * hidden;
            let mut dst = h_residual_buf;
            let mut src = out_region;
            let mut nn: i32 = n_elem;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n_elem as u32) + block - 1) / block;
            let rc = cuLaunchKernel(
                ker.fn_vector_add_f16.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_blattn residual vector_add",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        if linear_trace {
            qwen35_sync_stream(stream_raw, "qwen35 linear trace residual")?;
            if let Some(t0) = trace_last {
                trace_residual_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            let total_ms = trace_begin
                .map(|t0| t0.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            eprintln!(
                "[qwen35-linear-perf] tokens={} layer={} total_ms={:.3} \
                 norm_ms={:.3} qkv_ms={:.3} conv_state_ms={:.3} \
                 conv1d_ms={:.3} silu_l2_ms={:.3} alpha_beta_ms={:.3} \
                 delta_ms={:.3} z_ms={:.3} rms_gated_ms={:.3} \
                 out_ms={:.3} residual_ms={:.3}",
                num_tokens,
                layer_idx,
                total_ms,
                trace_norm_ms,
                trace_qkv_ms,
                trace_conv_state_ms,
                trace_conv1d_ms,
                trace_silu_l2_ms,
                trace_alpha_beta_ms,
                trace_delta_ms,
                trace_z_ms,
                trace_rms_gated_ms,
                trace_out_ms,
                trace_residual_ms,
            );
        }
        Ok(())
    }

    /// Phase 2c-B-B-i smoke: embed → apply_full_attn_qkv_only(3)
    /// → dense_mlp(3) → final-norm → lm_head → argmax. Layer 3
    /// is the first full-attn slot in the 3:1 linear:full
    /// pattern. Q/K/V projections execute on layer 3's actual
    /// weights but the attention block itself is still SKIPPED
    /// (no RoPE, KV-write, FA-2, sigmoid_mul, o_proj).
    pub unsafe fn forward_layer3_qkv_plus_mlp_smoke(&self, token_id: u32) -> Result<u32> {
        let scr = self.scratch.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_layer3_qkv_plus_mlp_smoke: scratch absent".into()))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_layer3_qkv_plus_mlp_smoke: outside_kernels absent".into()))?;
        let cublaslt = self.cublaslt.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_layer3_qkv_plus_mlp_smoke: cublaslt absent".into()))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_layer3_qkv_plus_mlp_smoke: stream absent".into()))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_layer3_qkv_plus_mlp_smoke: arena absent".into()))?;
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_layer3_qkv_plus_mlp_smoke: model absent".into()))?;
        let arch = &self.arch;
        let hidden = arch.base.hidden_size as u32;
        let vocab = arch.base.vocab_size as u32;
        let eps = arch.base.rms_norm_eps;
        let stream_raw = stream.raw() as u64;
        let token_bytes = (token_id as i32).to_le_bytes();
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyHtoDAsync_v2(
                scr.token_in_ptr as CUdeviceptr,
                token_bytes.as_ptr() as *const _, 4,
                stream_raw as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 qkv-mlp HtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1, hidden, vocab,
            }.launch(
                ker.fn_embedding_gather_f16,
                scr.h_residual_ptr,
                model.outside.embed_tokens.offset_bytes,
                scr.token_in_ptr,
                stream_raw,
            )?;
        }
        // QKV projections on layer 3 (first full-attn layer).
        self.apply_full_attn_qkv_only(3, 0)?;
        // Dense MLP on layer 3.
        self.apply_dense_mlp_layer(3)?;
        // Final-norm + lm_head + argmax (same as Phase 2c-A tail).
        let hidden_fp8_region = arena.region(
            "qwen35_qkvmlp_hidden_fp8", hidden as usize, 16)?;
        let hidden_scale_region = arena.region(
            "qwen35_qkvmlp_hidden_scale", 4, 4)?;
        unsafe {
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: 1, hidden, eps,
            }.launch(
                ker.fn_fused_rmsnorm_fp8_quant,
                hidden_fp8_region.device_ptr(),
                hidden_scale_region.device_ptr(),
                scr.h_residual_ptr,
                model.outside.final_norm.offset_bytes,
                stream_raw,
            )?;
        }
        unsafe {
            cublaslt.fp8_gemm(
                hidden_fp8_region.device_ptr(),
                model.outside.lm_head_fp8.offset_bytes,
                scr.logits_ptr,
                1, vocab as i32, hidden as i32,
                hidden_scale_region.device_ptr(),
                model.outside.lm_head_fp8.scale_ptr,
                stream_raw,
            )?;
        }
        unsafe {
            use cudarc::driver::sys::*;
            let block_dim: u32 = vocab.min(1024);
            let mut row_ptr = scr.logits_ptr;
            let mut out_ptr = scr.token_out_ptr;
            let mut vsz: i32 = vocab as i32;
            let args = [
                (&mut row_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut vsz) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                ker.fn_argmax_f16.raw() as CUfunction,
                1, 1, 1, block_dim, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 qkv-mlp argmax",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        stream.fence()?;
        let mut predicted: i32 = 0;
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                &mut predicted as *mut i32 as *mut _,
                scr.token_out_ptr as CUdeviceptr, 4);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 qkv-mlp DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        Ok(predicted as u32)
    }

    /// Phase 2c-D smoke: embed → 64-layer forward → final-norm →
    /// lm_head → argmax. Each layer dispatches by
    /// `arch.layer_types[li]`:
    ///   Linear → apply_linear_attn_layer(li)
    ///   Full   → apply_full_attn_qkv_only(li, position)
    /// followed by apply_dense_mlp_layer(li) for both. `position`
    /// is the same KV slot index for every full-attn layer in
    /// the single-token decode case; pass it through from the
    /// caller (smoke wrapper supplies 0).
    pub unsafe fn forward_all_layers_smoke(
        &self, token_id: u32, position: u32,
    ) -> Result<u32> {
        self.forward_all_layers_with_splice(token_id, position, None)
    }

    /// Phase 3-b: same as `forward_all_layers_smoke` but when
    /// `splice_h_residual = Some(bytes)`, the post-embed
    /// h_residual is overwritten with the supplied `[hidden] f16`
    /// row instead of looking up `embed_tokens[token_id]`. Used
    /// for vision splice — `bytes.len()` must equal
    /// `hidden_size * 2` (one row of the vision-tower output).
    /// `token_id` is still honoured (HtoD'd) so the linear-attn
    /// state evolves the same way it would for a non-vision
    /// step; it just isn't read by the embedding gather.
    pub unsafe fn forward_all_layers_with_splice(
        &self, token_id: u32, position: u32,
        splice_h_residual: Option<&[u8]>,
    ) -> Result<u32> {
        self.forward_layers_only(token_id, position, splice_h_residual)?;
        self.forward_finalize_argmax()
    }

    /// Phase 4-a: prefill-step variant — runs embed (or splice)
    /// + the 64-layer loop only. Skips final-norm + lm_head GEMM
    /// + argmax + the host-side DtoH/fence. KV cache + linear-attn
    /// state advance the same way they do in the full smoke path,
    /// so the next call sees the correct context.
    ///
    /// Used by `generate_session_with_vision` for every prefill
    /// step except the last (which needs the argmax to seed
    /// decode). Saves ~lm_head_gemm + 1 fence + 1 DtoH per
    /// prefill token at the cost of one extra method boundary.
    pub unsafe fn forward_layers_only(
        &self, token_id: u32, position: u32,
        splice_h_residual: Option<&[u8]>,
    ) -> Result<()> {
        let scr = self.scratch.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_all_layers_smoke: scratch absent".into()))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_all_layers_smoke: outside_kernels absent".into()))?;
        let cublaslt = self.cublaslt.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_all_layers_smoke: cublaslt absent".into()))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_all_layers_smoke: stream absent".into()))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_all_layers_smoke: arena absent".into()))?;
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_all_layers_smoke: model absent".into()))?;
        let arch = &self.arch;
        let hidden = arch.base.hidden_size as u32;
        let vocab = arch.base.vocab_size as u32;
        let eps = arch.base.rms_norm_eps;
        let stream_raw = stream.raw() as u64;

        // (1) Source h_residual. Two paths:
        //   * Vision splice (Some): HtoD the vision-tower row
        //     directly. Skips token HtoD + embed_gather — neither
        //     is needed because nothing downstream reads
        //     `token_in_ptr` after embed.
        //   * Text (None): HtoD token id, then embed_gather.
        if let Some(bytes) = splice_h_residual {
            let want = (hidden as usize) * 2;
            if bytes.len() != want {
                return Err(corrupt(
                    self.paths.model_dir.clone(),
                    format!("forward_all_layers_with_splice: splice has \
                             {} bytes but expected {} (hidden*2)",
                            bytes.len(), want),
                ));
            }
            use cudarc::driver::sys::*;
            let rc = cuMemcpyHtoDAsync_v2(
                scr.h_residual_ptr as CUdeviceptr,
                bytes.as_ptr() as *const _, want,
                stream_raw as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 splice HtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        } else {
            let token_bytes = (token_id as i32).to_le_bytes();
            {
                use cudarc::driver::sys::*;
                let rc = cuMemcpyHtoDAsync_v2(
                    scr.token_in_ptr as CUdeviceptr,
                    token_bytes.as_ptr() as *const _, 4,
                    stream_raw as CUstream);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 all-layers HtoD",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            }
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1, hidden, vocab,
            }.launch(
                ker.fn_embedding_gather_f16,
                scr.h_residual_ptr,
                model.outside.embed_tokens.offset_bytes,
                scr.token_in_ptr,
                stream_raw,
            )?;
        }

        // (2) Per-layer dispatch: attention block then dense MLP.
        for (li, ty) in arch.base.layer_types.iter().enumerate() {
            match ty {
                rvllm_loader::LayerAttnType::Linear => {
                    self.apply_linear_attn_layer(li)?;
                }
                rvllm_loader::LayerAttnType::Full => {
                    self.apply_full_attn_qkv_only(li, position)?;
                }
                other => {
                    return Err(corrupt(
                        self.paths.model_dir.clone(),
                        format!("forward_all_layers_smoke: layer {li} has \
                                 unsupported attn type {other:?}; Qwen 3.5 \
                                 layer_types must be {{Linear, Full}}"),
                    ));
                }
            }
            self.apply_dense_mlp_layer(li)?;
        }
        Ok(())
    }

    /// Phase 4-a: finalize the current `h_residual` into a single
    /// predicted token. Runs final RMSNorm + FP8 quantize +
    /// cuBLASLt FP8 GEMM (lm_head) + argmax + DtoH the scalar
    /// token id back to the host. Used both by the
    /// final prefill step (to seed decode) and every decode
    /// step.
    pub unsafe fn forward_finalize_argmax(&self) -> Result<u32> {
        let scr = self.scratch.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_finalize_argmax: scratch absent".into()))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_finalize_argmax: outside_kernels absent".into()))?;
        let cublaslt = self.cublaslt.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_finalize_argmax: cublaslt absent".into()))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_finalize_argmax: stream absent".into()))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_finalize_argmax: arena absent".into()))?;
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_finalize_argmax: model absent".into()))?;
        let arch = &self.arch;
        let hidden = arch.base.hidden_size as u32;
        let vocab = arch.base.vocab_size as u32;
        let eps = arch.base.rms_norm_eps;
        let stream_raw = stream.raw() as u64;

        let hidden_fp8_region = arena.region(
            "qwen35_all_hidden_fp8", hidden as usize, 16)?;
        let hidden_scale_region = arena.region(
            "qwen35_all_hidden_scale", 4, 4)?;
        rvllm_fused::FusedRmsnormFp8QuantLaunch {
            num_tokens: 1, hidden, eps,
        }.launch(
            ker.fn_fused_rmsnorm_fp8_quant,
            hidden_fp8_region.device_ptr(),
            hidden_scale_region.device_ptr(),
            scr.h_residual_ptr,
            model.outside.final_norm.offset_bytes,
            stream_raw,
        )?;
        cublaslt.fp8_gemm(
            hidden_fp8_region.device_ptr(),
            model.outside.lm_head_fp8.offset_bytes,
            scr.logits_ptr,
            1, vocab as i32, hidden as i32,
            hidden_scale_region.device_ptr(),
            model.outside.lm_head_fp8.scale_ptr,
            stream_raw,
        )?;
        self.launch_token_select(ker, scr.logits_ptr, scr.token_out_ptr,
            vocab, stream_raw, "qwen35 finalize")?;
        stream.fence()?;
        let mut predicted: i32 = 0;
        {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                &mut predicted as *mut i32 as *mut _,
                scr.token_out_ptr as CUdeviceptr, 4);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 finalize DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        Ok(predicted as u32)
    }

    /// Phase 2c-E: run prefill over `prompt_ids` (accumulates KV
    /// cache + linear-attn state, discarding all per-step
    /// argmaxes), then decode up to `max_new_tokens` steps from
    /// the last prompt position. `on_token` is called for each
    /// generated token and may short-circuit by returning `false`.
    /// Returns total completion tokens emitted.
    ///
    /// Arena is checkpointed before each forward step and
    /// restored after the predicted token has been read back,
    /// so a long prompt + long decode reuses the same scratch
    /// slab. Persistent state (KV cache, linear-attn SSM,
    /// conv1d ring) lives below the checkpoint and is untouched.
    pub unsafe fn generate_session(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: u32,
        on_token: impl FnMut(u32, u32) -> bool,
    ) -> Result<u32> {
        self.generate_session_with_vision(prompt_ids, max_new_tokens, &[], on_token)
    }

    /// Zero the Gated-DeltaNet linear-attn state. Mirror of
    /// `Qwen36Bringup::reset_linear_state`. Codex review flagged
    /// that the worker currently does NOT reset Qwen 3.5's
    /// persistent recurrent state per request, so cross-request
    /// state contamination is possible. No-op when the model has
    /// no linear-attn layers.
    pub fn reset_linear_state(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let ls = match self.linear_state.as_ref() {
                Some(s) => s,
                None => return Ok(()),
            };
            let total = ls.n_linear_layers.saturating_mul(ls.per_layer_bytes);
            if total == 0 || ls.base_ptr == 0 { return Ok(()); }
            let stream = match self.stream.as_ref() {
                Some(s) => s.raw() as CUstream,
                None => return Ok(()),
            };
            let rc = cuMemsetD8Async(
                ls.base_ptr, 0, total, stream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 reset_linear_state",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Recurrent-state snapshot for future Qwen 3.5 prompt-lookup
    /// speculative decode. Like Qwen 3.6, Qwen 3.5 carries
    /// Gated-DeltaNet delta state plus causal-conv1d state across
    /// decode steps, so a verifier that advances through rejected
    /// drafts must be able to roll those recurrent buffers back.
    ///
    /// The caller owns `dst_*_ptr` scratch allocation sized from
    /// `recurrent_state_bytes()`.
    pub fn snapshot_recurrent_state(&self, dst_linear_ptr: u64, dst_conv_ptr: u64) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let ls = match self.linear_state.as_ref() {
                Some(s) => s,
                None => return Ok(()),
            };
            let stream = match self.stream.as_ref() {
                Some(s) => s.raw() as CUstream,
                None => return Ok(()),
            };
            let linear_bytes = ls.n_linear_layers.saturating_mul(ls.per_layer_bytes);
            if linear_bytes > 0 && ls.base_ptr != 0 {
                let rc = cuMemcpyDtoDAsync_v2(
                    dst_linear_ptr,
                    ls.base_ptr,
                    linear_bytes,
                    stream,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 snapshot_recurrent_state(linear)",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            let conv_bytes = ls
                .n_linear_layers
                .saturating_mul(ls.conv_state_per_layer_bytes);
            if conv_bytes > 0 && ls.conv_state_base_ptr != 0 {
                let rc = cuMemcpyDtoDAsync_v2(
                    dst_conv_ptr,
                    ls.conv_state_base_ptr,
                    conv_bytes,
                    stream,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 snapshot_recurrent_state(conv)",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        let _ = (dst_linear_ptr, dst_conv_ptr);
        Ok(())
    }

    /// Restore a recurrent-state snapshot captured by
    /// `snapshot_recurrent_state`.
    pub fn restore_recurrent_state(&self, src_linear_ptr: u64, src_conv_ptr: u64) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let ls = match self.linear_state.as_ref() {
                Some(s) => s,
                None => return Ok(()),
            };
            let stream = match self.stream.as_ref() {
                Some(s) => s.raw() as CUstream,
                None => return Ok(()),
            };
            let linear_bytes = ls.n_linear_layers.saturating_mul(ls.per_layer_bytes);
            if linear_bytes > 0 && ls.base_ptr != 0 {
                let rc = cuMemcpyDtoDAsync_v2(
                    ls.base_ptr,
                    src_linear_ptr,
                    linear_bytes,
                    stream,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 restore_recurrent_state(linear)",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            let conv_bytes = ls
                .n_linear_layers
                .saturating_mul(ls.conv_state_per_layer_bytes);
            if conv_bytes > 0 && ls.conv_state_base_ptr != 0 {
                let rc = cuMemcpyDtoDAsync_v2(
                    ls.conv_state_base_ptr,
                    src_conv_ptr,
                    conv_bytes,
                    stream,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 restore_recurrent_state(conv)",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        let _ = (src_linear_ptr, src_conv_ptr);
        Ok(())
    }

    pub fn recurrent_state_bytes(&self) -> (usize, usize) {
        #[cfg(feature = "cuda")]
        {
            if let Some(ls) = self.linear_state.as_ref() {
                return (
                    ls.n_linear_layers.saturating_mul(ls.per_layer_bytes),
                    ls.n_linear_layers
                        .saturating_mul(ls.conv_state_per_layer_bytes),
                );
            }
        }
        (0, 0)
    }

    /// Debug/startup probe for the snapshot helpers. This does not
    /// mutate logical model state: it snapshots current recurrent
    /// buffers to arena scratch, then restores the same bytes.
    pub fn spec_state_snapshot_selftest(&self) -> Result<()> {
        let (linear_bytes, conv_bytes) = self.recurrent_state_bytes();
        #[cfg(feature = "cuda")]
        {
            let arena = self.arena.as_ref().ok_or_else(|| corrupt(
                self.paths.model_dir.clone(),
                "qwen35 spec_state_snapshot_selftest: arena absent".into()))?;
            let snap_linear = arena.region(
                "qwen35_spec_selftest_linear",
                linear_bytes.max(1),
                16,
            )?;
            let snap_conv = arena.region(
                "qwen35_spec_selftest_conv",
                conv_bytes.max(1),
                16,
            )?;
            self.snapshot_recurrent_state(
                snap_linear.device_ptr(),
                snap_conv.device_ptr(),
            )?;
            self.restore_recurrent_state(
                snap_linear.device_ptr(),
                snap_conv.device_ptr(),
            )?;
        }
        let _ = (linear_bytes, conv_bytes);
        Ok(())
    }

    /// Debug/startup probe for the Qwen35 spec-decode building
    /// blocks. Runs a tiny batched verifier chunk and a commit-only
    /// chunk, then resets persistent caches so normal serving starts
    /// from a clean state.
    pub unsafe fn spec_decode_primitives_selftest(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let arena = self.arena.as_ref().ok_or_else(|| corrupt(
                self.paths.model_dir.clone(),
                "qwen35 spec_decode_primitives_selftest: arena absent".into()))?;
            let ck = arena.checkpoint();
            self.reset_linear_state()?;
            self.reset_conv_state()?;
            self.reset_kv_cache()?;
            let argmaxes = self.forward_qwen35_decode_argmax_all(&[1, 100], 0)?;
            if argmaxes.len() != 2 {
                return Err(corrupt(
                    self.paths.model_dir.clone(),
                    format!("qwen35 spec_decode_primitives_selftest: expected 2 \
                             argmaxes, got {}", argmaxes.len())));
            }
            self.reset_linear_state()?;
            self.reset_conv_state()?;
            self.reset_kv_cache()?;
            self.forward_qwen35_decode_commit_only(&[1], 0)?;
            self.reset_linear_state()?;
            self.reset_conv_state()?;
            self.reset_kv_cache()?;
            arena.restore(ck);
        }
        Ok(())
    }

    /// Zero the causal-conv1d state. Mirror of
    /// `Qwen36Bringup::reset_conv_state`.
    pub fn reset_conv_state(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let ls = match self.linear_state.as_ref() {
                Some(s) => s,
                None => return Ok(()),
            };
            let total = ls.n_linear_layers.saturating_mul(ls.conv_state_per_layer_bytes);
            if total == 0 || ls.conv_state_base_ptr == 0 { return Ok(()); }
            let stream = match self.stream.as_ref() {
                Some(s) => s.raw() as CUstream,
                None => return Ok(()),
            };
            let rc = cuMemsetD8Async(
                ls.conv_state_base_ptr, 0, total, stream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 reset_conv_state",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Zero the full-attn KV cache (and per-layer scale buffers
    /// for NVFP4 KV). Unlike Qwen 3.6 (single contiguous KV
    /// region), Qwen 3.5 allocates per-layer regions; this
    /// iterates the layer table and memsets each.
    pub fn reset_kv_cache(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let kvc = match self.kv_cache.as_ref() {
                Some(c) => c,
                None => return Ok(()),
            };
            let stream = match self.stream.as_ref() {
                Some(s) => s.raw() as CUstream,
                None => return Ok(()),
            };
            for layer in kvc.layers.iter() {
                if layer.k_ptr != 0 {
                    let rc = cuMemsetD8Async(
                        layer.k_ptr, 0, kvc.per_layer_bytes, stream,
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen35 reset_kv_cache (k)",
                            rvllm_core::CudaErrorKind::MemcpyFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                if layer.v_ptr != 0 {
                    let rc = cuMemsetD8Async(
                        layer.v_ptr, 0, kvc.per_layer_bytes, stream,
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen35 reset_kv_cache (v)",
                            rvllm_core::CudaErrorKind::MemcpyFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                if kvc.per_layer_scale_bytes > 0 {
                    if layer.k_scale_ptr != 0 {
                        let rc = cuMemsetD8Async(
                            layer.k_scale_ptr, 0,
                            kvc.per_layer_scale_bytes, stream,
                        );
                        if rc != CUresult::CUDA_SUCCESS {
                            return Err(rvllm_core::RvllmError::cuda(
                                "qwen35 reset_kv_cache (k_scale)",
                                rvllm_core::CudaErrorKind::MemcpyFailed,
                                rvllm_core::CudaCtx::setup(),
                            ));
                        }
                    }
                    if layer.v_scale_ptr != 0 {
                        let rc = cuMemsetD8Async(
                            layer.v_scale_ptr, 0,
                            kvc.per_layer_scale_bytes, stream,
                        );
                        if rc != CUresult::CUDA_SUCCESS {
                            return Err(rvllm_core::RvllmError::cuda(
                                "qwen35 reset_kv_cache (v_scale)",
                                rvllm_core::CudaErrorKind::MemcpyFailed,
                                rvllm_core::CudaCtx::setup(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Phase 3-b: same as `generate_session` but accepts vision
    /// splice tuples `[(token_start, &[u8])]`. Each tuple's bytes
    /// are `[num_rows, hidden] f16` — i.e. one row per spliced
    /// prompt position. The function builds a per-prompt-token
    /// splice map up-front and routes each prefill step through
    /// `forward_all_layers_with_splice`. The splice does NOT
    /// affect decode steps; only prefill positions in
    /// `[token_start, token_start + num_rows)` see vision rows.
    ///
    /// Hidden-row size = `arch.base.hidden_size * 2` bytes.
    /// `data.len() % row_bytes == 0` is asserted at every tuple.
    pub unsafe fn generate_session_with_vision<'a>(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: u32,
        vision_splices: &'a [(usize, &'a [u8])],
        mut on_token: impl FnMut(u32, u32) -> bool,
    ) -> Result<u32> {
        if prompt_ids.is_empty() {
            return Ok(0);
        }
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "generate_session: arena absent".into()))?;
        let row_bytes = (self.arch.base.hidden_size) * 2;

        // Build per-prompt-token splice map.
        let mut splice_map: Vec<Option<&[u8]>> = vec![None; prompt_ids.len()];
        for (start, data) in vision_splices {
            if data.len() % row_bytes != 0 {
                return Err(corrupt(
                    self.paths.model_dir.clone(),
                    format!("generate_session: vision splice at token_start={start} \
                             has {} bytes, not a multiple of row_bytes={row_bytes}",
                            data.len()),
                ));
            }
            let n_rows = data.len() / row_bytes;
            for k in 0..n_rows {
                let idx = start + k;
                if idx >= prompt_ids.len() {
                    return Err(corrupt(
                        self.paths.model_dir.clone(),
                        format!("generate_session: vision splice row {idx} \
                                 (start={start}+{k}) past prompt_len={}",
                                prompt_ids.len()),
                    ));
                }
                splice_map[idx] = Some(&data[k * row_bytes .. (k + 1) * row_bytes]);
            }
        }

        let ck = arena.checkpoint();
        let _ = splice_map; // map is only consumed by the per-token path
        let prompt_len = prompt_ids.len() as u32;

        // Phase #1-e: env-gated layer-major batched prefill.
        // RVLLM_QWEN35_BATCHED_PREFILL=1 → one call into the
        // batched driver replaces the N forward_layers_only loops.
        // Default OFF (per-token, byte-stable canary path) until
        // this gets a runtime correctness sign-off.
        let batched_prefill = std::env::var("RVLLM_QWEN35_BATCHED_PREFILL")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        let mut last_predicted: u32 = 0;
        if batched_prefill {
            last_predicted = self.forward_qwen35_prefill_batched(
                prompt_ids, vision_splices)?;
            arena.restore(ck);
        } else {
            // Per-token path (Phase 4-a). Skip lm_head on every
            // prefill token except the last to save the GEMM +
            // fence + DtoH cost.
            let last_idx = prompt_ids.len() - 1;
            for (p, &tok) in prompt_ids.iter().enumerate() {
                if p < last_idx {
                    self.forward_layers_only(tok, p as u32, splice_map[p])?;
                } else {
                    last_predicted = self.forward_all_layers_with_splice(
                        tok, p as u32, splice_map[p])?;
                }
                arena.restore(ck);
            }
        }

        // Decode. `last_predicted` is the prefill's lm_head argmax
        // at position N-1 — i.e. the first generated token at
        // absolute position prompt_len. Emit it FIRST, then feed it
        // back at position prompt_len to produce the SECOND token,
        // and so on.
        //
        // The previous version was off-by-one: it ran one full
        // forward step before any emission, so the first answer
        // token (which the prefill had already computed for free)
        // was thrown away and the user saw the second answer token
        // first. That bug was visible as "Bild zeigt …" instead of
        // "Das Bild zeigt …" on Qwen 3.5 27B dense (2026-05-12).
        let mut current = last_predicted;
        let mut emitted: u32 = 0;
        for step in 0..max_new_tokens {
            let pos = prompt_len + step;
            // Emit `current` (the token at position `pos`). The
            // callback returns `false` to request a short-circuit
            // (EOS, cancellation, etc.).
            if !on_token(current, pos) {
                emitted += 1;
                break;
            }
            emitted += 1;
            if emitted >= max_new_tokens {
                break;
            }
            // Compute next: feed `current` at `pos`, lm_head's
            // argmax of position `pos` predicts the token at
            // position `pos + 1`.
            current = self.forward_all_layers_smoke(current, pos)?;
            arena.restore(ck);
        }
        Ok(emitted)
    }

    /// Phase 2c-B-A: apply one layer's dense MLP block to
    /// `h_residual` in place. Reads `h_residual`, writes
    /// `h_residual = h_residual + down_proj(SiLU(gate(h_norm)) *
    /// up(h_norm))` where `h_norm = rmsnorm(h_residual,
    /// post_attention_layernorm[layer_idx])`.
    ///
    /// Used by Phase 2c-B-A's `forward_dense_mlp_smoke` to
    /// validate the MLP path on real model weights without yet
    /// running the attention block. Phase 2c-B-core will fold
    /// this into the per-layer forward loop.
    pub unsafe fn apply_dense_mlp_layer(&self, layer_idx: usize) -> Result<()> {
        let arch = &self.arch;
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_dense_mlp_layer: model absent".into(),
        ))?;
        let scr = self.scratch.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_dense_mlp_layer: scratch absent".into(),
        ))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_dense_mlp_layer: outside_kernels absent".into(),
        ))?;
        let cublaslt = self.cublaslt.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_dense_mlp_layer: cublaslt absent".into(),
        ))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_dense_mlp_layer: stream absent".into(),
        ))?;
        let layer = model.layers.get(layer_idx).ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            format!("apply_dense_mlp_layer: layer {layer_idx} out of range"),
        ))?;
        let hidden = arch.base.hidden_size as i32;
        let intermediate = arch.base.intermediate_size as i32;
        let eps = arch.base.rms_norm_eps;
        let stream_raw = stream.raw() as u64;

        // (1) h_work ← rmsnorm(h_residual, post_attn_layernorm).
        // Kernel is in-place: we DtoD copy first then norm.
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                scr.h_work_ptr as CUdeviceptr,
                scr.h_residual_ptr as CUdeviceptr,
                (hidden as usize) * 2,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 mlp dtod h_work",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // Kernel signature: (x_inout, gamma, eps, hidden). Grid
        // (num_tokens, 1, 1), block (min(hidden, 1024), 1, 1),
        // smem = 32 * 4 (warp-reduction scratch). Mirrors
        // gemma4_launcher::RmsnormInplaceLaunch.
        unsafe {
            use cudarc::driver::sys::*;
            let mut hw_ptr = scr.h_work_ptr;
            let mut gamma_ptr = post_attn_ln_ptr(&layer.attn);
            let mut eps_arg = eps;
            let mut hd = hidden;
            let args = [
                (&mut hw_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut gamma_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_arg) as *mut f32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block_dim = (hidden as u32).min(1024);
            let rc = cuLaunchKernel(
                ker.fn_rmsnorm_inplace_f16.raw() as CUfunction,
                1, 1, 1, block_dim, 1, 1, 32 * 4,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 rmsnorm_inplace_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // (2) silu_mid ← SiLU(gate(h_work)) * up(h_work)
        // grid (ceil(N/8), M), block 256. M=1, N=intermediate.
        unsafe {
            use cudarc::driver::sys::*;
            let mut out_ptr = scr.silu_mid_ptr;
            let mut wg_ptr = layer.mlp.gate_proj.offset_bytes;
            let mut wu_ptr = layer.mlp.up_proj.offset_bytes;
            let mut sg_ptr = layer.mlp.gate_proj.blockscale_ptr.unwrap_or(0);
            let mut su_ptr = layer.mlp.up_proj.blockscale_ptr.unwrap_or(0);
            let mut inp_ptr = scr.h_work_ptr;
            let mut m_arg: i32 = 1;
            let mut n_arg: i32 = intermediate;
            let mut k_arg: i32 = hidden;
            let mut ncb: i32 = hidden / 128;
            let args = [
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut wg_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut wu_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut sg_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut su_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut inp_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut m_arg) as *mut i32 as *mut core::ffi::c_void,
                (&mut n_arg) as *mut i32 as *mut core::ffi::c_void,
                (&mut k_arg) as *mut i32 as *mut core::ffi::c_void,
                (&mut ncb) as *mut i32 as *mut core::ffi::c_void,
            ];
            let grid_x: u32 = ((intermediate as u32) + 7) / 8;
            let rc = cuLaunchKernel(
                ker.fn_fp8_gemv_blockwise_wpr_native_f16in_dual_silu.raw() as CUfunction,
                grid_x, 1, 1, 256, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 fp8_gemv_dual_silu launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // (3+4) down_out ← FP8 GEMV with F16 input + blockwise
        // weight scale. Single launch via the gemma4-style
        // launcher; takes F16 input directly (no separate FP8
        // quantize step) and consumes the blockwise scale at
        // `b_chscale` (the parameter slot is shape-agnostic —
        // qwen36 reuses it with blockwise scales for the same
        // reason). M=1 single-row path; the looped variant
        // applies if M>1.
        let fp8_gemv_fn = ker.fn_fp8_gemv_wpr_native_f16in.ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_dense_mlp_layer: fp8_gemv_wpr_native_f16in unavailable \
             (need sm_100+; GB10/sm_121 should have it)".into(),
        ))?;
        // (4-5) Fused dense FFN-down + post-FFN residual add
        // (Phase 8 other-models fusion, 2026-05-23). Single-launch
        // replacement for the back-to-back FP8 GEMV →
        // vector_add_f16 pair. Writes directly into
        // `scr.h_residual_ptr`; the per-call `scr.down_out_ptr`
        // temp buffer is no longer consumed by anyone after this
        // fusion (kept allocated for arena address stability).
        let _ = cublaslt;
        let fused = std::env::var("RVLLM_QWEN35_FP8_GEMV_RESIDUAL_FUSED")
            .map(|s| s != "0")
            .unwrap_or(true);
        unsafe {
            use cudarc::driver::sys::*;
            if fused {
                let mut h_resid = scr.h_residual_ptr;
                let mut w_ptr = layer.mlp.down_proj.offset_bytes;
                let mut s_ptr = layer.mlp.down_proj.blockscale_ptr.unwrap_or(0);
                let mut x_ptr = scr.silu_mid_ptr;
                let mut m_i: i32 = 1;
                let mut n_i: i32 = hidden;
                let mut k_i: i32 = intermediate as i32;
                let mut ncb: i32 = (intermediate as i32 + 127) / 128;
                let args = [
                    (&mut h_resid) as *mut u64 as *mut core::ffi::c_void,
                    (&mut w_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut s_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut x_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut ncb) as *mut i32 as *mut core::ffi::c_void,
                ];
                let grid = (((hidden as u32) + 7) / 8, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    ker.fn_fp8_gemv_f16in_residual_add.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    256, 1, 1, 0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 fp8_gemv_f16in_residual_add (dense FFN down+resid) launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            } else {
                let arena = self.arena.as_ref().ok_or_else(|| corrupt(
                    self.paths.model_dir.clone(),
                    "qwen35 unfused dense ffn down+resid: arena unset".into()))?;
                let temp = arena.region("qwen35_ffn_down_unfused_temp",
                                        (hidden as usize) * 2, 16)?;
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m: 1, n: hidden as u32, k: intermediate as u32,
                }.launch(
                    fp8_gemv_fn,
                    temp.device_ptr(),
                    layer.mlp.down_proj.offset_bytes,
                    layer.mlp.down_proj.blockscale_ptr.unwrap_or(0),
                    scr.silu_mid_ptr,
                    stream_raw,
                )?;
                let mut dst = scr.h_residual_ptr;
                let mut src = temp.device_ptr();
                let mut n_elem: i32 = hidden;
                let v_args = [
                    (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                    (&mut src) as *mut u64 as *mut core::ffi::c_void,
                    (&mut n_elem) as *mut i32 as *mut core::ffi::c_void,
                ];
                let v_block: u32 = 256;
                let v_grid: u32 = ((hidden as u32 + v_block - 1) / v_block).max(1);
                let rc = cuLaunchKernel(
                    ker.fn_vector_add_f16.raw() as CUfunction,
                    v_grid, 1, 1,
                    v_block, 1, 1,
                    0,
                    stream_raw as CUstream,
                    v_args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen35 unfused dense ffn down+resid vector_add launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            }
        }
        Ok(())
    }

    /// Phase-#1 building block: batched dense MLP over N tokens.
    ///
    /// Runs the same 5-op chain as `apply_dense_mlp_layer` but
    /// over a `[N, hidden]` residual buffer in one launch each
    /// (grid.y = N). All three sub-kernels already support
    /// M-batching natively:
    ///   * `rmsnorm_inplace_f16_kernel` — grid (num_tokens, 1, 1)
    ///   * `fp8_gemv_blockwise_wpr_native_f16in_dual_silu_kernel`
    ///     — grid (ceil(N_out/8), M, 1)
    ///   * `fp8_gemv_blockwise_wpr_native_f16in_kernel` — same
    ///   * `vector_add_f16` — element-wise over N*hidden
    ///
    /// Scratch (h_work, silu_mid, down_out) is allocated
    /// per-call via the arena and sized at `num_tokens`. The
    /// existing single-token `apply_dense_mlp_layer` is
    /// preserved for the decode path (num_tokens = 1).
    ///
    /// Caller passes the `[N, hidden]` h_residual buffer
    /// directly so this composes with the layer-major prefill
    /// driver that will live above it.
    pub unsafe fn apply_dense_mlp_layer_batched(
        &self,
        layer_idx: usize,
        num_tokens: u32,
        h_residual_buf: u64,
    ) -> Result<()> {
        let arch = &self.arch;
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_dense_mlp_layer_batched: model absent".into()))?;
        let ker = self.outside_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_dense_mlp_layer_batched: outside_kernels absent".into()))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_dense_mlp_layer_batched: stream absent".into()))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_dense_mlp_layer_batched: arena absent".into()))?;
        let layer = model.layers.get(layer_idx).ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            format!("apply_dense_mlp_layer_batched: layer {layer_idx} out of range")))?;
        let hidden = arch.base.hidden_size as i32;
        let intermediate = arch.base.intermediate_size as i32;
        let eps = arch.base.rms_norm_eps;
        let stream_raw = stream.raw() as u64;
        let mlp_trace =
            std::env::var("RVLLM_QWEN35_MLP_PERF_TRACE").as_deref() == Ok("1");
        let trace_begin = if mlp_trace {
            qwen35_sync_stream(stream_raw, "qwen35 mlp trace begin")?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut trace_last = trace_begin;
        let mut trace_copy_ms = 0.0_f64;
        let mut trace_norm_ms = 0.0_f64;
        let mut trace_gate_up_ms = 0.0_f64;
        let mut trace_down_ms = 0.0_f64;
        let mut trace_residual_ms = 0.0_f64;
        let n = num_tokens as usize;
        let nh = (n * hidden as usize) * 2;
        let ni = (n * intermediate as usize) * 2;
        let mlp_ck = arena.checkpoint();
        let h_work_buf  = arena.region("qwen35_bmlp_h_work",  nh, 16)?.device_ptr();
        let silu_mid_buf = arena.region("qwen35_bmlp_silu",   ni, 16)?.device_ptr();
        let down_out_buf = arena.region("qwen35_bmlp_down",   nh, 16)?.device_ptr();

        // (1) h_work ← rmsnorm(h_residual, post_attn_layernorm).
        //     Copy [N, hidden] → h_work then norm with
        //     grid.x = num_tokens.
        {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                h_work_buf, h_residual_buf, nh, stream_raw as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bmlp dtod h_work",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        if mlp_trace {
            qwen35_sync_stream(stream_raw, "qwen35 mlp trace copy")?;
            if let Some(t0) = trace_last {
                trace_copy_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }
        {
            use cudarc::driver::sys::*;
            let mut hw_ptr = h_work_buf;
            let mut gamma_ptr = post_attn_ln_ptr(&layer.attn);
            let mut eps_arg = eps;
            let mut hd = hidden;
            let args = [
                (&mut hw_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut gamma_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_arg) as *mut f32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block_dim = (hidden as u32).min(1024);
            let rc = cuLaunchKernel(
                ker.fn_rmsnorm_inplace_f16.raw() as CUfunction,
                num_tokens, 1, 1, block_dim, 1, 1, 32 * 4,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bmlp rmsnorm_inplace_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        if mlp_trace {
            qwen35_sync_stream(stream_raw, "qwen35 mlp trace norm")?;
            if let Some(t0) = trace_last {
                trace_norm_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }

        // (2) silu_mid ← SiLU(gate(h_work)) * up(h_work).
        //
        // Default path: f16-activation row-batched FP8 GEMV, which is
        // the quality reference. Experimental path:
        // RVLLM_QWEN35_MLP_CUTLASS_SM120=1 uses CUTLASS blockscale FP8
        // GEMM for M>=RVLLM_QWEN35_MLP_CUTLASS_MIN_TOKENS (default 128).
        // It is opt-in because it quantizes activations to FP8.
        let cutlass_min_tokens = std::env::var("RVLLM_QWEN35_MLP_CUTLASS_MIN_TOKENS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(128);
        let cutlass_requested =
            crate::gemma4_bring_up::parse_truthy_env("RVLLM_QWEN35_MLP_CUTLASS_SM120")
                .unwrap_or(false)
                && num_tokens >= cutlass_min_tokens;
        let mut used_cutlass_mlp = false;
        if cutlass_requested {
            if let CutlassBackend::SoSm120(ref lib) = self.cutlass {
                let gate_buf = arena.region("qwen35_bmlp_gate_cutlass", ni, 16)?.device_ptr();
                let up_buf = arena.region("qwen35_bmlp_up_cutlass", ni, 16)?.device_ptr();
                self.qwen35_fp8_cutlass_blockscale_sm120(
                    ker,
                    lib,
                    gate_buf,
                    layer.mlp.gate_proj.offset_bytes,
                    layer.mlp.gate_proj.blockscale_ptr.unwrap_or(0),
                    h_work_buf,
                    num_tokens,
                    intermediate as u32,
                    hidden as u32,
                    stream_raw,
                    "qwen35_bmlp_gate_in_fp8",
                )?;
                self.qwen35_fp8_cutlass_blockscale_sm120(
                    ker,
                    lib,
                    up_buf,
                    layer.mlp.up_proj.offset_bytes,
                    layer.mlp.up_proj.blockscale_ptr.unwrap_or(0),
                    h_work_buf,
                    num_tokens,
                    intermediate as u32,
                    hidden as u32,
                    stream_raw,
                    "qwen35_bmlp_up_in_fp8",
                )?;
                {
                    use cudarc::driver::sys::*;
                    let mut out_ptr = silu_mid_buf;
                    let mut gate_ptr = gate_buf;
                    let mut up_ptr = up_buf;
                    let mut elem_n: i32 = (num_tokens as i32) * intermediate;
                    let args = [
                        (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                        (&mut gate_ptr) as *mut u64 as *mut core::ffi::c_void,
                        (&mut up_ptr) as *mut u64 as *mut core::ffi::c_void,
                        (&mut elem_n) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let grid_x = ((elem_n as u32) + 255) / 256;
                    let rc = cuLaunchKernel(
                        ker.fn_silu_mul_f16.raw() as CUfunction,
                        grid_x, 1, 1, 256, 1, 1, 0,
                        stream_raw as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen35_bmlp CUTLASS silu_mul_f16 launch",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup()));
                    }
                }
                used_cutlass_mlp = true;
            }
        }
        if !used_cutlass_mlp {
            use cudarc::driver::sys::*;
            let mut out_ptr = silu_mid_buf;
            let mut wg_ptr = layer.mlp.gate_proj.offset_bytes;
            let mut wu_ptr = layer.mlp.up_proj.offset_bytes;
            let mut sg_ptr = layer.mlp.gate_proj.blockscale_ptr.unwrap_or(0);
            let mut su_ptr = layer.mlp.up_proj.blockscale_ptr.unwrap_or(0);
            let mut inp_ptr = h_work_buf;
            let mut m_arg: i32 = num_tokens as i32;
            let mut n_arg: i32 = intermediate;
            let mut k_arg: i32 = hidden;
            let mut ncb: i32 = hidden / 128;
            let args = [
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut wg_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut wu_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut sg_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut su_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut inp_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut m_arg) as *mut i32 as *mut core::ffi::c_void,
                (&mut n_arg) as *mut i32 as *mut core::ffi::c_void,
                (&mut k_arg) as *mut i32 as *mut core::ffi::c_void,
                (&mut ncb) as *mut i32 as *mut core::ffi::c_void,
            ];
            let grid_x: u32 = ((intermediate as u32) + 7) / 8;
            let rc = cuLaunchKernel(
                ker.fn_fp8_gemv_blockwise_wpr_native_f16in_dual_silu.raw() as CUfunction,
                grid_x, num_tokens, 1, 256, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bmlp fp8_gemv_dual_silu launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        if mlp_trace {
            qwen35_sync_stream(stream_raw, "qwen35 mlp trace gate_up")?;
            if let Some(t0) = trace_last {
                trace_gate_up_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }

        if used_cutlass_mlp {
            if let CutlassBackend::SoSm120(ref lib) = self.cutlass {
                self.qwen35_fp8_cutlass_blockscale_sm120(
                    ker,
                    lib,
                    down_out_buf,
                    layer.mlp.down_proj.offset_bytes,
                    layer.mlp.down_proj.blockscale_ptr.unwrap_or(0),
                    silu_mid_buf,
                    num_tokens,
                    hidden as u32,
                    intermediate as u32,
                    stream_raw,
                    "qwen35_bmlp_down_in_fp8",
                )?;
            }
        } else {
            // (3+4) down_proj at M=num_tokens — one launch via the
            //       row-batched FP8 GEMV (kernel already supports
            //       grid.y=M; see qwen36 fp8_proj_dispatch fix).
            let fp8_gemv_fn = ker.fn_fp8_gemv_wpr_native_f16in.ok_or_else(|| corrupt(
                self.paths.model_dir.clone(),
                "apply_dense_mlp_layer_batched: fp8_gemv_wpr_native_f16in unavailable".into()))?;
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m: num_tokens, n: hidden as u32, k: intermediate as u32,
            }.launch(
                fp8_gemv_fn,
                down_out_buf,
                layer.mlp.down_proj.offset_bytes,
                layer.mlp.down_proj.blockscale_ptr.unwrap_or(0),
                silu_mid_buf,
                stream_raw,
            )?;
        }
        if mlp_trace {
            qwen35_sync_stream(stream_raw, "qwen35 mlp trace down")?;
            if let Some(t0) = trace_last {
                trace_down_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            trace_last = Some(std::time::Instant::now());
        }

        // (5) h_residual += down_out elementwise over N*hidden.
        {
            use cudarc::driver::sys::*;
            let n_elem = (num_tokens as i32) * hidden;
            let mut dst = h_residual_buf;
            let mut src = down_out_buf;
            let mut n_arg: i32 = n_elem;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut n_arg) as *mut i32 as *mut core::ffi::c_void,
            ];
            let grid_x: u32 = ((n_elem as u32) + 255) / 256;
            let rc = cuLaunchKernel(
                ker.fn_vector_add_f16.raw() as CUfunction,
                grid_x, 1, 1, 256, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35_bmlp vector_add_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        if mlp_trace {
            qwen35_sync_stream(stream_raw, "qwen35 mlp trace residual")?;
            if let Some(t0) = trace_last {
                trace_residual_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
            let total_ms = trace_begin
                .map(|t0| t0.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            eprintln!(
                "[qwen35-mlp-perf] tokens={} layer={} path={} total_ms={:.3} \
                 copy_ms={:.3} norm_ms={:.3} gate_up_ms={:.3} \
                 down_ms={:.3} residual_ms={:.3}",
                num_tokens,
                layer_idx,
                if used_cutlass_mlp { "cutlass-sm120" } else { "gemv-f16in" },
                total_ms,
                trace_copy_ms,
                trace_norm_ms,
                trace_gate_up_ms,
                trace_down_ms,
                trace_residual_ms,
            );
        }
        arena.restore(mlp_ck);
        Ok(())
    }
}

/// Free function — extract the per-layer
/// post_attention_layernorm pointer. The layer enum lives in
/// rvllm-loader so we can't impl methods on it from here
/// (orphan rule); pattern-match inline instead.
#[cfg(feature = "cuda")]
fn post_attn_ln_ptr(attn: &rvllm_loader::qwen35_weights::Qwen35LayerAttn) -> u64 {
    use rvllm_loader::qwen35_weights::Qwen35LayerAttn;
    match attn {
        Qwen35LayerAttn::Linear(l) => l.post_attention_layernorm.offset_bytes,
        Qwen35LayerAttn::Full(f) => f.post_attention_layernorm.offset_bytes,
    }
}

#[cfg(feature = "cuda")]
fn build_rope_tables(
    arena: &HbmArena,
    max_pos: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
) -> Result<Qwen35RopeTables> {
    let half = rotary_dim / 2;
    // Proportional-RoPE frequency convention: divisor is head_dim
    // (NOT rotary_dim) — matches Qwen 3.6 and Gemma 4. So even
    // though only the first `rotary_dim` slots of each head are
    // rotated, the frequency spread is over the full head dim.
    let inv_theta: Vec<f32> = (0..half)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let table_elems = max_pos * half;
    let table_bytes = table_elems * 2; // f16
    let mut cos_bytes = Vec::with_capacity(table_bytes);
    let mut sin_bytes = Vec::with_capacity(table_bytes);
    for pos in 0..max_pos {
        for &freq in &inv_theta {
            let angle = pos as f32 * freq;
            let c = half::f16::from_f32(angle.cos()).to_bits();
            let s = half::f16::from_f32(angle.sin()).to_bits();
            cos_bytes.extend_from_slice(&c.to_le_bytes());
            sin_bytes.extend_from_slice(&s.to_le_bytes());
        }
    }
    let cos_region = arena.region("qwen35_rope_cos", table_bytes, 16)?;
    let sin_region = arena.region("qwen35_rope_sin", table_bytes, 16)?;
    unsafe {
        cos_region.copy_from_host(&cos_bytes)?;
        sin_region.copy_from_host(&sin_bytes)?;
    }
    eprintln!(
        "[qwen35] RoPE tables uploaded: max_pos={max_pos} head_dim={head_dim} \
         rotary_dim={rotary_dim} theta={rope_theta:.1e} ({} KiB cos + {} KiB sin)",
        table_bytes / 1024,
        table_bytes / 1024,
    );
    Ok(Qwen35RopeTables {
        cos_ptr: cos_region.device_ptr(),
        sin_ptr: sin_region.device_ptr(),
        max_pos,
        rotary_dim,
    })
}

/// Read the linear-attn SSM head config from `text_config`.
/// Qwen 3.5 27B dense ships `linear_num_value_heads=48,
/// linear_value_head_dim=128`. Falls back to Qwen 3.6's defaults
/// (32, 128) if the keys are missing — this is the right move for
/// non-canonical checkpoints; production-style configs always have
/// them.
#[cfg(feature = "cuda")]
impl Qwen35LinearState {
    /// Device pointer to layer `linear_idx`'s delta-rule state
    /// slice (shape `[num_v_heads, hkd, hvd]` f16). Indexed by
    /// `layer_idx_to_linear_idx[absolute_layer_idx]`.
    pub fn delta_state_ptr(&self, linear_idx: usize) -> u64 {
        let off = linear_idx.saturating_mul(self.per_layer_bytes);
        let total = self.n_linear_layers * self.per_layer_bytes;
        if off + self.per_layer_bytes > total { return 0; }
        self.base_ptr + off as u64
    }

    /// Device pointer to layer `linear_idx`'s causal-conv1d state
    /// slice (shape `[ks-1, conv_dim]` f16). Returns 0 if the
    /// per-layer bytes are 0 (i.e. the model has no linear-attn
    /// layers — defensive, shouldn't happen for Qwen 3.5).
    pub fn conv_state_ptr(&self, linear_idx: usize) -> u64 {
        if self.conv_state_per_layer_bytes == 0 { return 0; }
        let off = linear_idx.saturating_mul(self.conv_state_per_layer_bytes);
        let total = self.n_linear_layers * self.conv_state_per_layer_bytes;
        if off + self.conv_state_per_layer_bytes > total { return 0; }
        self.conv_state_base_ptr + off as u64
    }
}

#[cfg(feature = "cuda")]
fn qwen35_linear_dims(model_dir: &Path) -> (usize, usize) {
    let d = qwen35_la_dims(model_dir);
    (d.num_v_heads, d.head_v_dim)
}

/// Full linear-attn head/dim probe — read from `text_config`
/// (linear_num_{key,value}_heads, linear_{key,value}_head_dim,
/// linear_conv_kernel_dim). Defaults match Qwen 3.5 27B
/// (16/48/128/128, ks=4) — production checkpoints always carry the
/// keys; the defaults exist only so test fixtures don't have to
/// duplicate them.
#[cfg(feature = "cuda")]
fn qwen35_la_dims(model_dir: &Path) -> Qwen35LaDims {
    let p = model_dir.join("config.json");
    let parsed = std::fs::read(&p).ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
    let v = parsed.as_ref();
    let tc = match v {
        Some(v) if v["text_config"]["hidden_size"].is_u64() => &v["text_config"],
        Some(v) => v,
        None => &serde_json::Value::Null,
    };
    let num_k_heads = tc["linear_num_key_heads"].as_u64().unwrap_or(16) as usize;
    let num_v_heads = tc["linear_num_value_heads"].as_u64().unwrap_or(48) as usize;
    let head_k_dim = tc["linear_key_head_dim"].as_u64().unwrap_or(128) as usize;
    let head_v_dim = tc["linear_value_head_dim"].as_u64().unwrap_or(128) as usize;
    let conv_kernel_dim = tc["linear_conv_kernel_dim"].as_u64().unwrap_or(4) as usize;
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let v_per_k = if num_k_heads == 0 { 0 } else { num_v_heads / num_k_heads };
    Qwen35LaDims {
        num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_dim, key_dim, value_dim, conv_dim, v_per_k,
    }
}

#[cfg(feature = "cuda")]
fn qwen35_sync_stream(stream_raw: u64, op: &'static str) -> Result<()> {
    unsafe {
        use cudarc::driver::sys::*;
        let rc = cuStreamSynchronize(stream_raw as CUstream);
        if rc != CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::cuda(
                op,
                rvllm_core::CudaErrorKind::LaunchFailed,
                rvllm_core::CudaCtx::setup(),
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn zero_region(dev_ptr: u64, nbytes: usize) -> Result<()> {
    use cudarc::driver::sys::*;
    let rc = unsafe { cuMemsetD8_v2(dev_ptr, 0, nbytes) };
    if rc != CUresult::CUDA_SUCCESS {
        return Err(rvllm_core::RvllmError::cuda(
            "qwen35_zero_region",
            rvllm_core::CudaErrorKind::MemcpyFailed,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    Ok(())
}


#[derive(Debug)]
pub enum Qwen35Error {
    /// Generation called before the GPU forward path is wired.
    ForwardNotImplemented,
    /// Weight loader not yet implemented.
    LoaderNotImplemented,
    /// Vision tower not yet implemented.
    VisionNotImplemented,
}

impl std::fmt::Display for Qwen35Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForwardNotImplemented => write!(
                f, "qwen35: forward path not wired yet (Phase 0 only — \
                    see QWEN35_BRINGUP_PLAN.md)"),
            Self::LoaderNotImplemented => write!(
                f, "qwen35: weight loader not wired yet (Phase 1 pending)"),
            Self::VisionNotImplemented => write!(
                f, "qwen35: vision tower not wired yet (Phase 3 pending)"),
        }
    }
}

impl std::error::Error for Qwen35Error {}

fn corrupt(path: PathBuf, detail: String) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt { detail },
        ctx: LoaderCtx { path, tensor: None },
        bt: std::backtrace::Backtrace::capture(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tempdir() -> PathBuf {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "rvllm-qwen35-{}-{}",
            std::process::id(),
            N.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_config(dir: &Path, body: &str) {
        let mut f = std::fs::File::create(dir.join("config.json")).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    /// Minimal qwen3_5 config sufficient for the arch parser. Layer
    /// pattern is the 3:1 linear:full mix the 27B-dense uses.
    fn qwen35_full() -> &'static str {
        r#"{
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "model_type": "qwen3_5",
          "image_token_id": 248056,
          "vision_config": {
            "hidden_size": 1152, "depth": 27, "out_hidden_size": 5120
          },
          "text_config": {
            "num_hidden_layers": 8, "hidden_size": 5120,
            "intermediate_size": 17408, "num_attention_heads": 24,
            "num_key_value_heads": 4, "head_dim": 256,
            "vocab_size": 248320, "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-5, "hidden_act": "silu",
            "tie_word_embeddings": false, "attn_output_gate": true,
            "rope_parameters": {"rope_theta": 5000000.0,
                                  "partial_rotary_factor": 0.25,
                                  "mrope_section": [11, 11, 10]},
            "layer_types": [
              "linear_attention","linear_attention","linear_attention","full_attention",
              "linear_attention","linear_attention","linear_attention","full_attention"
            ]
          }
        }"#
    }
    // Note: arch_parses_rope_fields expects rope_theta=5e6 because
    // the qwen35_full() fixture uses that value (vs the real
    // checkpoint's 1e7) for compactness.

    #[test]
    fn arch_parses_dense_qwen35() {
        let tmp = tempdir();
        write_config(&tmp, qwen35_full());
        let arch = Qwen35Arch::from_dir(&tmp).unwrap().expect("dense qwen35");
        assert_eq!(arch.base.num_hidden_layers, 8);
        assert_eq!(arch.n_linear, 6);
        assert_eq!(arch.n_full, 2);
        assert!(arch.attn_output_gate);
        assert_eq!(arch.image_token_id, Some(248056));
        assert_eq!(arch.vision_depth, Some(27));
    }

    #[test]
    fn arch_rejects_qwen36_moe_config() {
        let tmp = tempdir();
        write_config(&tmp, &qwen35_full().replace(
            "\"num_hidden_layers\": 8",
            "\"num_hidden_layers\": 40, \"num_experts\": 256",
        ));
        // num_experts != 0 → Qwen35 probe returns None, leaving room
        // for the Qwen36 probe in the family resolver.
        let arch = Qwen35Arch::from_dir(&tmp).unwrap();
        assert!(arch.is_none());
    }

    #[test]
    fn arch_rejects_mistral_config() {
        let tmp = tempdir();
        write_config(&tmp, r#"{
          "architectures": ["Mistral3ForConditionalGeneration"],
          "model_type": "mistral3",
          "text_config": {"hidden_size": 12288}
        }"#);
        let arch = Qwen35Arch::from_dir(&tmp).unwrap();
        assert!(arch.is_none());
    }

    #[test]
    fn arch_parses_rope_fields() {
        let tmp = tempdir();
        write_config(&tmp, qwen35_full());
        let arch = Qwen35Arch::from_dir(&tmp).unwrap().expect("dense qwen35");
        assert_eq!(arch.rope_partial_rotary_factor, 0.25);
        assert!((arch.rope_theta - 5_000_000.0).abs() < 1e-3);
    }

    /// Phase 2c-C-a regression: the linear-attn dim probe reads the
    /// canonical Qwen 3.5 27B keys and computes the derived widths.
    #[cfg(feature = "cuda")]
    #[test]
    fn la_dims_reads_qwen35_27b_canonical() {
        let tmp = tempdir();
        write_config(&tmp, &qwen35_full().replace(
            "\"attn_output_gate\": true,",
            "\"attn_output_gate\": true, \
             \"linear_num_key_heads\": 16, \"linear_num_value_heads\": 48, \
             \"linear_key_head_dim\": 128, \"linear_value_head_dim\": 128, \
             \"linear_conv_kernel_dim\": 4,",
        ));
        let d = qwen35_la_dims(&tmp);
        assert_eq!(d.num_k_heads, 16);
        assert_eq!(d.num_v_heads, 48);
        assert_eq!(d.head_k_dim, 128);
        assert_eq!(d.head_v_dim, 128);
        assert_eq!(d.conv_kernel_dim, 4);
        assert_eq!(d.key_dim, 16 * 128);
        assert_eq!(d.value_dim, 48 * 128);
        assert_eq!(d.conv_dim, 2 * 16 * 128 + 48 * 128); // 10240
        assert_eq!(d.v_per_k, 3);
    }

    /// Phase 2c-C-a defaults: when the linear-attn keys are missing
    /// the probe falls back to the Qwen 3.5 27B canonical values so
    /// test fixtures don't need to duplicate them.
    #[cfg(feature = "cuda")]
    #[test]
    fn la_dims_defaults_to_qwen35_27b() {
        let tmp = tempdir();
        write_config(&tmp, qwen35_full());
        let d = qwen35_la_dims(&tmp);
        assert_eq!(d.num_k_heads, 16);
        assert_eq!(d.num_v_heads, 48);
        assert_eq!(d.head_v_dim, 128);
        assert_eq!(d.v_per_k, 3);
        // legacy 2-tuple alias must agree.
        let (nvh, hvd) = qwen35_linear_dims(&tmp);
        assert_eq!(nvh, d.num_v_heads);
        assert_eq!(hvd, d.head_v_dim);
    }

    /// Phase 3-a defaults: vision config keys propagate onto the
    /// arch struct (they drive the chat-template image-pad
    /// expansion + admission predictor).
    #[test]
    fn arch_carries_vision_config() {
        let tmp = tempdir();
        write_config(&tmp, qwen35_full());
        let arch = Qwen35Arch::from_dir(&tmp).unwrap().expect("dense qwen35");
        assert_eq!(arch.vision_hidden_size, Some(1152));
        assert_eq!(arch.vision_depth, Some(27));
        assert_eq!(arch.vision_out_hidden_size, Some(5120));
        assert_eq!(arch.image_token_id, Some(248056));
    }

    #[test]
    fn typed_errors_render_helpful_messages() {
        // Verify the typed error messages still point at the plan
        // doc so operators get clear breadcrumbs.
        let fwd = Qwen35Error::ForwardNotImplemented.to_string();
        assert!(fwd.contains("Phase 0"));
        let ldr = Qwen35Error::LoaderNotImplemented.to_string();
        assert!(ldr.contains("Phase 1"));
        let vis = Qwen35Error::VisionNotImplemented.to_string();
        assert!(vis.contains("Phase 3"));
    }
}
