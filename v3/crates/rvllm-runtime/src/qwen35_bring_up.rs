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
use rvllm_cutlass::cublaslt::CublasLt;
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

/// Per-full-attn-layer KV cache pair. Linear-attn layers don't
/// carry a positional KV cache (state is SSM-style in
/// `Qwen35LinearState`).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct Qwen35LayerKv {
    pub k_ptr: u64,
    pub v_ptr: u64,
}

/// Sparse BF16 KV cache — only the 16 full-attn layers carry a
/// `Qwen35LayerKv`. `layer_idx_to_full_idx[layer_idx]` returns
/// `Some(full_rank)` for full-attn layers and `None` for linear.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct Qwen35KvCache {
    pub layers: Vec<Qwen35LayerKv>,
    pub layer_idx_to_full_idx: Vec<Option<usize>>,
    pub max_pos: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub per_layer_bytes: usize,
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
    // ── Per-layer (Phase 2c-B) — RMSNorm + dense MLP ────────
    pub rmsnorm_inplace_f16_mod: LoadedModule,
    pub fn_rmsnorm_inplace_f16: KernelFn,
    pub fp8_gemv_dual_silu_mod: LoadedModule,
    pub fn_fp8_gemv_blockwise_wpr_native_f16in_dual_silu: KernelFn,
    pub fp8_quantize_per_token_f16_mod: LoadedModule,
    pub fn_fp8_quantize_per_token_f16: KernelFn,
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
    // ── Vector residual add (used after attn + MLP) ────────
    pub f16_plus_f32_inplace_f16_mod: LoadedModule,
    pub fn_f16_plus_f32_inplace_f16: KernelFn,
    pub vector_add_f16_mod: LoadedModule,
    pub fn_vector_add_f16: KernelFn,
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
    #[cfg(feature = "cuda")]
    pub cublaslt: Option<CublasLt>,
    /// Linear-attn head/dim config. Always present once `load()`
    /// completes; defaulted to Qwen 3.5 27B (`16/48/128/128, ks=4`)
    /// when the config is missing the keys.
    #[cfg(feature = "cuda")]
    pub la_dims: Option<Qwen35LaDims>,
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

            // KV cache: BF16 [max_pos, n_kv_heads, head_dim] per
            // FULL-attn layer.
            let n_kv_heads = arch.base.num_key_value_heads;
            let per_layer_kv_bytes = kv_max_pos * n_kv_heads * head_dim * 2;
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
                    kv_layers.push(Qwen35LayerKv {
                        k_ptr: k_region.device_ptr(),
                        v_ptr: v_region.device_ptr(),
                    });
                    layer_idx_to_full_idx[li] = Some(next_full);
                    next_full += 1;
                }
            }
            let kv_cache = Qwen35KvCache {
                layers: kv_layers,
                layer_idx_to_full_idx,
                max_pos: kv_max_pos,
                n_kv_heads,
                head_dim,
                per_layer_bytes: per_layer_kv_bytes,
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
            // Single-output FP8 GEMV (gemma4-launcher Fp8GemvF16InLaunch
            // entry point). Used as the M=1 down_proj fallback when
            // cuBLASLt blockwise FP8 has no sm_121 algo.
            let fp8_gemv_mod = kernels.load_ptx(rvllm_kernels::FP8_GEMV_PTX_STEM)?;
            let fn_fp8_gemv_wpr_native_f16in = {
                let (major, minor) = ctx.compute_capability();
                let target = rvllm_core::CompileTarget::from_compute_capability(
                    major, minor);
                match target {
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
            let flash_attention_mod = kernels.load_ptx("flash_attention")?;
            let fn_flash_attention_2_decode_f16io = flash_attention_mod
                .get_function("flash_attention_2_decode_f16io_kernel")?;
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

            // Vector residual.
            let f16_plus_f32_inplace_f16_mod =
                kernels.load_ptx("f16_plus_f32_inplace_f16")?;
            let fn_f16_plus_f32_inplace_f16 = f16_plus_f32_inplace_f16_mod
                .get_function("f16_plus_f32_inplace_f16_kernel")?;
            let vector_add_f16_mod = kernels.load_ptx("vector_add_f16")?;
            let fn_vector_add_f16 = vector_add_f16_mod
                .get_function("vector_add_f16_kernel")?;

            let outside_kernels = Qwen35OutsideKernels {
                embedding_gather_f16_mod,
                fn_embedding_gather_f16,
                fused_rmsnorm_fp8_quant_mod,
                fn_fused_rmsnorm_fp8_quant,
                argmax_mod,
                fn_argmax_f16,
                rmsnorm_inplace_f16_mod,
                fn_rmsnorm_inplace_f16,
                fp8_gemv_dual_silu_mod,
                fn_fp8_gemv_blockwise_wpr_native_f16in_dual_silu,
                fp8_quantize_per_token_f16_mod,
                fn_fp8_quantize_per_token_f16,
                fp8_gemv_mod,
                fn_fp8_gemv_wpr_native_f16in,
                split_q_gate_f16_mod,
                fn_split_q_gate_f16,
                fused_rope_qwen_partial_f16kv_mod,
                fn_fused_rope_qwen_partial_f16kv,
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
                qwen_linear_rmsnorm_gated_f16_mod,
                fn_qwen_linear_rmsnorm_gated_f16,
                f16_plus_f32_inplace_f16_mod,
                fn_f16_plus_f32_inplace_f16,
                vector_add_f16_mod,
                fn_vector_add_f16,
            };

            // cuBLASLt for the FP8 lm_head matmul. 32 MiB workspace
            // (matches Qwen 3.6 / Gemma 4 sizing).
            let cublaslt_ws_bytes: usize = 32 * 1024 * 1024;
            let cublaslt_ws_region = arena.region(
                "qwen35_cublaslt_ws", cublaslt_ws_bytes, 256)?;
            let cublaslt = CublasLt::new(
                cublaslt_ws_region.device_ptr(), cublaslt_ws_bytes)?;
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
                cublaslt: Some(cublaslt),
                la_dims: Some(la_dims),
            });
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!(
                "[qwen35] Phase 0 ONLY (no-cuda build): arch validated, \
                 NO weight upload. See v3/QWEN35_BRINGUP_PLAN.md."
            );
            Ok(Self { paths, arch, arena_bytes })
        }
    }
}

#[cfg(feature = "cuda")]
impl Qwen35Bringup {
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
        }

        // (8) flash_attention_2_decode_f16io: single-token decode
        // attention against the KV cache. The cache layout is
        //   [max_pos, num_kv_heads, head_dim] f16
        // which matches the FA-2 paged-attn assumption when we treat
        //   block_size = 1, num_blocks = max_pos,
        //   block_tables = identity [0, 1, ..., position],
        //   context_lens = [position+1]   (decode step at `position`)
        let context_len = (position + 1) as i32;
        let bt_bytes = (context_len as usize) * 4;
        let bt_region = arena.region(
            "qwen35_fattn_block_tables", bt_bytes.max(4), 16)?;
        let cl_region = arena.region(
            "qwen35_fattn_context_lens", 4, 16)?;
        let mut bt_host: Vec<u8> = Vec::with_capacity(bt_bytes);
        for i in 0..(context_len as i32) {
            bt_host.extend_from_slice(&i.to_le_bytes());
        }
        let cl_bytes: [u8; 4] = context_len.to_le_bytes();
        unsafe {
            bt_region.copy_from_host(&bt_host)?;
            cl_region.copy_from_host(&cl_bytes)?;
        }
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        const FA2_THREADS: i32 = 128;
        const FA2_BC: i32 = 32;
        let smem_bytes =
            2 * FA2_BC * head_dim * 4 + FA2_BC * 4 + (FA2_THREADS / 32) * 4;
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
            let mut block_tables = bt_region.device_ptr();
            let mut context_lens = cl_region.device_ptr();
            let mut scale_arg = scale;
            let mut nh = n_q_heads;
            let mut nkvh = n_kv_heads;
            let mut hd = head_dim;
            let mut bs: i32 = 1;
            let mut mbps: i32 = context_len;
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

        // (10) o_proj: FP8 GEMV [hidden, n_q_heads*head_dim] —
        // reuses the Fp8GemvF16InLaunch path. Output goes to
        // scr.o_out_ptr.
        let o_n = hidden as u32;
        let o_k = (n_q_heads * head_dim) as u32;
        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m: 1, n: o_n, k: o_k,
            }.launch(
                fp8_gemv_fn,
                scr.o_out_ptr,
                full.o_proj.offset_bytes,
                full.o_proj.blockscale_ptr.unwrap_or(0),
                scr.attn_out_ptr,
                stream_raw,
            )?;
        }

        // (11) h_residual += o_out (F16 + F16 → F16).
        unsafe {
            use cudarc::driver::sys::*;
            let mut dst = scr.h_residual_ptr;
            let mut src = scr.o_out_ptr;
            let mut n_arg: i32 = hidden;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut n_arg) as *mut i32 as *mut core::ffi::c_void,
            ];
            let grid_x: u32 = ((hidden as u32) + 255) / 256;
            let rc = cuLaunchKernel(
                ker.fn_vector_add_f16.raw() as CUfunction,
                grid_x, 1, 1, 256, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 attn vector_add_f16 (residual) launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        Ok(())
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

        // (9) out_proj FP8 GEMV → o_out [hidden].
        let out_n = la.out_proj.shape[0] as u32;
        let out_k = la.out_proj.shape[1] as u32;
        let out_bs = la.out_proj.blockscale_ptr.ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "apply_linear_attn_layer: out_proj blockscale missing".into()))?;
        let out_region = arena.region("qwen35_la_out", (out_n as usize) * 2, 16)?;
        rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
            m: 1, n: out_n, k: out_k,
        }.launch(
            fp8_gemv_fn,
            out_region.device_ptr(),
            la.out_proj.offset_bytes,
            out_bs,
            gated_region.device_ptr(),
            stream_raw,
        )?;

        // (10) h_residual += out (residual sum).
        {
            use cudarc::driver::sys::*;
            let mut dst = scr.h_residual_ptr;
            let mut src = out_region.device_ptr();
            let mut nn: i32 = hidden;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((hidden_u + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                ker.fn_vector_add_f16.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 la residual vector_add_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup()));
            }
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

        // (1) HtoD token + embed.
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

        // (3) Final RMSNorm + FP8 quantize.
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

        // (4) cuBLASLt fp8_gemm logits.
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
                    "qwen35 all-layers argmax",
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
                    "qwen35 all-layers DtoH",
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
        mut on_token: impl FnMut(u32, u32) -> bool,
    ) -> Result<u32> {
        if prompt_ids.is_empty() {
            return Ok(0);
        }
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "generate_session: arena absent".into()))?;
        let ck = arena.checkpoint();

        // Prefill: walk every prompt token; KV slot = its position.
        let prompt_len = prompt_ids.len() as u32;
        let mut last_predicted: u32 = 0;
        for (p, &tok) in prompt_ids.iter().enumerate() {
            last_predicted = self.forward_all_layers_smoke(tok, p as u32)?;
            // Free this step's scratch; persistent state above ck
            // is not touched.
            arena.restore(ck);
        }

        // Decode: feed `last_predicted` at position = prompt_len,
        // then its successor at prompt_len+1, etc.
        let mut emitted: u32 = 0;
        for step in 0..max_new_tokens {
            let pos = prompt_len + step;
            let tok = last_predicted;
            let predicted = self.forward_all_layers_smoke(tok, pos)?;
            arena.restore(ck);
            let cont = on_token(predicted, pos + 1);
            emitted += 1;
            last_predicted = predicted;
            if !cont {
                break;
            }
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
        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m: 1,
                n: hidden as u32,
                k: intermediate as u32,
            }.launch(
                fp8_gemv_fn,
                scr.down_out_ptr,
                layer.mlp.down_proj.offset_bytes,
                layer.mlp.down_proj.blockscale_ptr.unwrap_or(0),
                scr.silu_mid_ptr,
                stream_raw,
            )?;
        }
        // (cublaslt only used for outside lm_head now; the
        // blockwise path is the looped m=1 GEMV above.)
        let _ = cublaslt;

        // (5) h_residual += down_out (F16 + F16 → F16).
        unsafe {
            use cudarc::driver::sys::*;
            let mut dst = scr.h_residual_ptr;
            let mut src = scr.down_out_ptr;
            let mut n_arg: i32 = hidden;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut n_arg) as *mut i32 as *mut core::ffi::c_void,
            ];
            let grid_x: u32 = ((hidden as u32) + 255) / 256;
            let rc = cuLaunchKernel(
                ker.fn_vector_add_f16.raw() as CUfunction,
                grid_x, 1, 1, 256, 1, 1, 0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen35 vector_add_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
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
