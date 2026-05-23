//! Qwen 3.6 35B-A3B bring-up (Phase 1: outside-tensor upload).
//!
//! Phase 1 contract:
//!   - Initialize CUDA context, arena, stream.
//!   - Upload the three "outside-the-stack" tensors (embedding, final
//!     RMSNorm, lm_head) via [`rvllm_loader::qwen36_load::load_qwen36_outside`].
//!   - All forward methods (`run_generate`, `run_bench`, `run_ppl`,
//!     `init_prefix_cache`) `unimplemented!()` with phase-pointer
//!     messages — per-layer tensors + forward kernels are Phase 2/3.
//!
//! See `~/.claude/plans/abundant-meandering-sifakis.md` for the
//! phase-list.

use std::path::PathBuf;
use std::sync::Arc;

use rvllm_core::Result;
use rvllm_cutlass::{CublasLt, CutlassBackend};
use rvllm_kernels::{KernelFn, KernelLoader, LoadedModule};
use rvllm_loader::qwen36_weights::Qwen36LoadedModel;
use rvllm_mem::{context::CudaContextHandle, stream::Stream, HbmArena};

use crate::gemma4_bring_up::Gemma4EnginePaths;
use crate::qwen36_arch::Qwen36Arch;

/// Paged KV-cache dtype for Qwen 3.6 full-attention layers. F16 is
/// the production default; Nvfp4 is opt-in via `RVLLM_NVFP4_KV=1` and
/// matches the layout used by the Qwen 3.5 27B + Gemma 4 NVFP4 paths
/// (packed 4-bit K/V + per-(slot, kv_head, head_dim/16) E4M3
/// microscale). Decoders dispatch on this field; F16 stays
/// byte-identical when the gate is off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36KvDtype {
    F16,
    Nvfp4,
}

/// Kernel function pointers + their LoadedModule anchors needed for the
/// Qwen 3.6 outside-only forward path (embedding lookup, final RMSNorm,
/// lm_head matmul, argmax).
///
/// `LoadedModule` is RAII — its `Drop` calls `cuModuleUnload`, after
/// which the matching `KernelFn` handles become invalid. Holding the
/// modules alongside the function pointers in this struct keeps the
/// pair alive for the lifetime of the bring-up.
pub struct Qwen36OutsideKernels {
    pub embedding_gather_f16_mod: LoadedModule,
    pub fn_embedding_gather_f16: KernelFn,
    pub rmsnorm_inplace_f16_mod: LoadedModule,
    pub fn_rmsnorm_inplace_f16: KernelFn,
    /// FP8 GEMV for the lm_head matmul. Phase 3d will CPU-quantize the
    /// bf16 lm_head to FP8 at load time so this kernel can be used.
    /// `None` on platforms where the f16-input native-CVT variant
    /// isn't available (gated on `__CUDA_ARCH__ >= 1000`).
    pub fp8_gemv_mod: LoadedModule,
    pub fn_fp8_gemv_wpr_native_f16in: Option<KernelFn>,
    pub argmax_mod: LoadedModule,
    pub fn_argmax: KernelFn,
    /// f16-input sibling of `fn_argmax` — used by the Qwen36 closer
    /// to argmax over the lm_head's f16 logits row directly on the
    /// GPU. Replaces the previous DtoH-of-full-vocab + host-side
    /// scan; the GPU kernel returns a single i32 token id.
    pub fn_argmax_f16: KernelFn,
    /// Per-token f16→fp8 amax-quantise. Used by
    /// `fp8_proj_dispatch`'s m≥2 branch to feed cuBLASLt
    /// `fp8_gemm` (which expects fp8 input + per-token f32 scale).
    /// At m=1 the GEMV path consumes f16 directly so this kernel
    /// is unused.
    pub fp8_quantize_per_token_f16_mod: LoadedModule,
    pub fn_fp8_quantize_per_token_f16: KernelFn,
    /// Per-token-amax sibling of `fp8_quantize_per_token_f16` (one
    /// f32 per row, vs per-K-block scales). Feeds CUTLASS SM120's
    /// `prep_sfa` entry point on the m≥128 fast path; cuBLASLt
    /// blockwise uses the per-K-block kernel above instead.
    pub fp8_quantize_per_token_amax_f16_mod: LoadedModule,
    pub fn_fp8_quantize_per_token_amax_f16: KernelFn,
    /// Phase 3g: fused (final RMSNorm + FP8-quantize) for the lm_head
    /// pre-matmul step. Outputs FP8 hidden + per-token f32 scale that
    /// cuBLASLt's fp8_gemm consumes.
    pub fused_rmsnorm_fp8_quant_mod: LoadedModule,
    pub fn_fused_rmsnorm_fp8_quant: KernelFn,
    /// Phase 4g: partial-rotary RoPE kernel (f16 Q/K/V → f16 q_out
    /// + KV cache write). Qwen 3.6 uses partial_rotary_factor=0.25,
    /// so only `head_dim * 0.25 = 64` of the 256 head_dim is rotated.
    pub fused_rope_partial_f16kv_mod: LoadedModule,
    pub fn_fused_rope_partial_f16kv: KernelFn,
    /// Qwen-specific partial-NeoX RoPE + KV-cache write. Differs
    /// from the Gemma sibling in pair convention: pairs `(i, i +
    /// rotary_dim/2)` instead of `(i, i + head_dim/2)`. Replaces a
    /// host DtoH→CPU-RoPE→HtoD path that was the dominant per-token
    /// cost in `apply_layer_full_attn` (Phase 4b prep).
    pub fused_rope_qwen_partial_f16kv_mod: LoadedModule,
    pub fn_fused_rope_qwen_partial_f16kv: KernelFn,
    /// Phase 8 QKV-megakernel Phase 2 (2026-05-23, naive F16-KV):
    /// fuses Q-proj + K-proj + V-proj + Q-norm + K-norm + RoPE +
    /// KV-cache write into ONE launch. Each thread computes one
    /// output element via sequential K-dim FP8 GEMV reduction
    /// (slower than the warp-cooperative `fp8_gemv` pattern due to
    /// uncoalesced row reads, but correct and bounded). Opt-in via
    /// `RVLLM_QWEN36_QKV_MEGAKERNEL=1`; default off to avoid
    /// regression on production paths.
    pub fused_qkv_proj_qnorm_knorm_rope_qwen_partial_f16kv_mod: LoadedModule,
    pub fn_fused_qkv_proj_qnorm_knorm_rope_qwen_partial_f16kv: KernelFn,
    /// Phase 8 QKV-megakernel Phase 1 (2026-05-23): fuses Q-norm +
    /// K-norm into the RoPE + KV-write kernel. Each block already
    /// handles one (token, head), so adding a block-reduced
    /// RMSNorm before the rotation costs just one extra
    /// `__syncthreads` + a per-element gamma scale. Saves 2
    /// launches per full-attn layer per token.
    pub fused_qnorm_knorm_rope_qwen_partial_f16kv_mod: LoadedModule,
    pub fn_fused_qnorm_knorm_rope_qwen_partial_f16kv: KernelFn,
    /// NVFP4 commit 2: Qwen-specific NVFP4 RoPE + packed-4-bit KV
    /// write + FP8-E4M3 Q kernel. Same PTX the Qwen 3.5 27B path
    /// uses (parameterised on `num_heads`, `num_kv_heads`,
    /// `head_dim`, `rotary_dim`; partial NeoX pairing inside the
    /// first `rotary_dim` elements). Loaded only when
    /// `RVLLM_NVFP4_KV=1`. Dispatch lands in commit 3.
    pub fused_rope_qwen_partial_nvfp4kv_mod: Option<LoadedModule>,
    pub fn_fused_rope_qwen_partial_nvfp4kv: Option<KernelFn>,
    /// Phase 8 QKV-megakernel Phase 1 NVFP4 sibling (2026-05-23):
    /// fuses Q-norm + K-norm into the NVFP4 RoPE + FP8-Q + NVFP4-KV
    /// kernel. Same 2-phase per-head structure as the F16 sibling
    /// (943f8bb) but uses shared-mem `s_normalized` to pass the
    /// normalised Q/K to the rotation stage so the FP8/NVFP4
    /// quantise epilogue stays byte-identical. Gated on `Option`
    /// because the NVFP4 KV path is itself env-gated via
    /// `RVLLM_NVFP4_KV=1`.
    pub fused_qnorm_knorm_rope_qwen_partial_nvfp4kv_mod: Option<LoadedModule>,
    pub fn_fused_qnorm_knorm_rope_qwen_partial_nvfp4kv: Option<KernelFn>,
    /// NVFP4 commit 2: paged FA-2 decode kernel that reads packed
    /// 4-bit K/V + per-(slot, kv_head) E4M3 microscale and writes
    /// f16 output. Same PTX the Qwen 3.5 27B path uses; one CTA
    /// per (seq, query_head) — the per-head dispatch has no GQA
    /// cap, so Qwen 3.6's GQA=8 (16 q-heads / 2 kv-heads) is fine.
    pub flash_attention_nvfp4kv_mod: Option<LoadedModule>,
    pub fn_flash_attention_2_decode_nvfp4kv: Option<KernelFn>,
    /// Splits q_proj's interleaved `[num_heads, 2*head_dim]` output
    /// into separate q `[num_heads, head_dim]` + gate
    /// `[num_heads, head_dim]` regions. Replaces a host DtoH +
    /// per-head copy_from_slice + HtoD round-trip per token.
    pub split_q_gate_f16_mod: LoadedModule,
    pub fn_split_q_gate_f16: KernelFn,
    /// Conv1d state advance + conv_in assembly for the linear-attn
    /// block. Replaces a 2× DtoH + 2× HtoD pure-shuffle round-trip
    /// with one launch.
    pub conv_state_advance_f16_mod: LoadedModule,
    pub fn_conv_state_advance_f16: KernelFn,
    /// Fused alpha/beta computation for Qwen Gated-DeltaNet
    /// linear-attn. Replaces a host DtoH + f16→f32 + nested CPU
    /// GEMV (130k FLOPs/layer/token) + HtoD round-trip with one
    /// launch. Outputs f32 vectors (matches the existing
    /// alpha_region / beta_region dtype).
    pub qwen_linear_alpha_beta_f16_mod: LoadedModule,
    pub fn_qwen_linear_alpha_beta_f16: KernelFn,
    /// Fused silu + Q/K L2-norm + GQA-expand + V silu-pack for
    /// Qwen Gated-DeltaNet linear-attn. Replaces the host pipeline
    /// (DtoH conv_out + CPU silu/L2/GQA + HtoD q_exp/k_exp/v_pack)
    /// with one launch.
    pub qwen_linear_silu_l2_gqa_f16_mod: LoadedModule,
    pub fn_qwen_linear_silu_l2_gqa_f16: KernelFn,
    /// Per-v-head RMSNormGated with silu(z) gate. Replaces the
    /// host pipeline (DtoH readout/z/gamma + CPU rms + sigmoid·z
    /// + HtoD gated) at the end of `apply_layer_linear_attn`.
    pub qwen_linear_rmsnorm_gated_f16_mod: LoadedModule,
    pub fn_qwen_linear_rmsnorm_gated_f16: KernelFn,
    /// Pointwise SwiGLU activation `out = silu(gate) * up`. Used in
    /// `apply_layer_moe` to replace the per-expert host pipeline
    /// (DtoH gate + DtoH up + CPU silu·mul + HtoD silu) — saves
    /// two DtoH + one HtoD per expert × top_k experts × 30 MoE
    /// layers per token (Phase 4b-prep iter11).
    pub silu_mul_f16_mod: LoadedModule,
    pub fn_silu_mul_f16: KernelFn,
    /// Router-GEMV for the per-layer MoE gate. Reads
    /// `router_weight[num_experts, hidden]` f16 + the rmsnormed
    /// hidden state f16 from device memory and writes f32 logits.
    /// Replaces the host-cached f32 matvec (Phase 4b-prep iter17).
    pub router_gemv_f16_to_f32_mod: LoadedModule,
    pub fn_router_gemv_f16_to_f32: KernelFn,
    /// Phase 8 router+topk fusion (2026-05-23): fused router GEMV
    /// + topk-softmax via last-block-does-topk pattern. Eliminates
    /// the standalone topk_softmax launch per MoE layer per token.
    pub router_gemv_with_topk_f16_to_f32_mod: LoadedModule,
    pub fn_router_gemv_with_topk_f16_to_f32: KernelFn,
    /// Phase 8 batched router+topk fusion (2026-05-23): same
    /// last-block-does-topk pattern as the single-token variant
    /// but with per-token counter slots so multiple tokens'
    /// reductions can run independently in the batched grid.
    pub router_gemv_with_topk_batched_f16_to_f32_mod: LoadedModule,
    pub fn_router_gemv_with_topk_batched_f16_to_f32: KernelFn,
    /// Persistent per-token counter u32[max_pos] for the batched
    /// fused kernel. Zeroed once at worker bring-up; the kernel
    /// resets each token's slot via atomicExch after the last-
    /// block-does-topk path fires.
    pub router_gemv_with_topk_batched_counter_dev: u64,
    /// Persistent device counter u32[1] for the fused router+topk
    /// kernel's atomic last-block detection. Zeroed once at worker
    /// bring-up (lives above scratch_ck); the fused kernel resets
    /// to 0 via atomicExch in the last block, so subsequent calls
    /// find it at zero.
    pub router_topk_counter_dev: u64,
    /// Per-expert weighted accumulator into f32 routed_sum:
    /// `acc[i] += weight * f16_to_f32(in[i])`. Replaces the host
    /// pipeline (fence + DtoH down + CPU scaled-add) per expert
    /// (Phase 4b-prep iter18, after the fence-before-DtoH bug
    /// from iter13/14 was diagnosed in iter17).
    pub scaled_add_f16_to_f32_mod: LoadedModule,
    pub fn_scaled_add_f16_to_f32: KernelFn,
    /// In-place residual add `inout_f16 += add_f32`. Replaces the
    /// final per-MoE-layer host residual round-trip (DtoH
    /// last_hidden + CPU f16+f32 add + HtoD residual) so the whole
    /// MoE forward stays device-side (Phase 4b-prep iter19).
    pub f16_plus_f32_inplace_f16_mod: LoadedModule,
    pub fn_f16_plus_f32_inplace_f16: KernelFn,
    /// Fused shared-expert gate dot product + sigmoid. Output is a
    /// single f32 scalar on the device, consumed directly by
    /// `scaled_add_f16_to_f32_devw` (Phase 4b-prep iter21).
    pub shared_gate_dot_sigmoid_f16_mod: LoadedModule,
    pub fn_shared_gate_dot_sigmoid_f16: KernelFn,
    /// scaled_add variant that reads the scalar weight from a device
    /// pointer (Phase 4b-prep iter21).
    pub scaled_add_f16_to_f32_devw_mod: LoadedModule,
    pub fn_scaled_add_f16_to_f32_devw: KernelFn,
    /// Dual-output FP8 GEMV: same f16 input, two FP8 weights → two
    /// f16 outputs in one launch. Replaces the per-expert
    /// `gate FP8 GEMV` + `up FP8 GEMV` pair with a single fused
    /// kernel — saves 8 launches/layer × 30 layers/token plus halves
    /// the input-tile bandwidth (Phase 4b-prep iter31).
    pub fp8_gemv_dual_mod: LoadedModule,
    pub fn_fp8_gemv_dual: KernelFn,
    /// Triple-fuse: gate FP8 GEMV + up FP8 GEMV + silu_mul → single
    /// f16 output. Replaces (gate FP8 + up FP8 + silu_mul) for the
    /// per-expert FFN (Phase 4b-prep iter32).
    pub fp8_gemv_dual_silu_mod: LoadedModule,
    pub fn_fp8_gemv_dual_silu: KernelFn,
    /// GPU top-k + softmax over the f32 router logits. Output is
    /// (top_idx[k] i32, top_w[k] f32) on the device. Foundation for
    /// the indirect-MoE / CUDA-Graph project (Phase 4b-prep iter33).
    pub topk_softmax_f32_mod: LoadedModule,
    pub fn_topk_softmax_f32: KernelFn,
    /// Indirect-expert FP8 GEMV variants. Read the expert index
    /// from a device buffer and compute per-expert weight/scale
    /// pointer offsets internally — same launch params on every
    /// call, which makes the per-expert MoE chain CUDA-Graph
    /// captureable (Phase 4b-prep iter34).
    pub fp8_gemv_dual_silu_indirect_mod: LoadedModule,
    pub fn_fp8_gemv_dual_silu_indirect: KernelFn,
    /// Phase 8 dual_silu k_round-batch fusion (2026-05-23):
    /// `grid.z=top_k` batches the per-token MoE host-side loop
    /// over k_rounds into ONE launch. Output `silu` is now
    /// k_round-major [top_k, M, N]; subsequent down launches read
    /// their slice via offset addition. Same numerical contract
    /// as the unfused dual_silu_indirect kernel (byte-identical
    /// inner reduction).
    pub fp8_gemv_dual_silu_indirect_kround_batched_mod: LoadedModule,
    pub fn_fp8_gemv_dual_silu_indirect_kround_batched: KernelFn,
    pub fp8_gemv_indirect_mod: LoadedModule,
    pub fn_fp8_gemv_indirect: KernelFn,
    /// Phase 8 MoE-fusion (2026-05-23): fused FP8 GEMV (indirect-
    /// expert) + scaled f32 accumulation. Single-kernel replacement
    /// for the back-to-back pair (fp8_gemv_indirect + scaled_add_
    /// f16_to_f32_devw) used per k-round in the per-token MoE
    /// decode path. Eliminates 1 launch + 1 f16-roundtrip per
    /// k-round × 8 k-rounds × 40 MoE layers = 320 launches saved
    /// per decode token.
    pub fp8_gemv_indirect_scaled_add_mod: LoadedModule,
    pub fn_fp8_gemv_indirect_scaled_add: KernelFn,
    /// Phase 8 down k_round-batch fusion (2026-05-23): fuses the
    /// 8-iter host loop of `fp8_gemv_indirect_scaled_add` calls
    /// (one per k_round) into ONE launch. Each warp owns one
    /// (m, n) output slot and sequentially processes top_k
    /// k_rounds with a warp-local f32 accumulator — no atomic,
    /// no global RMW per k_round (just one at the end).
    pub fp8_gemv_indirect_scaled_add_kround_batched_mod: LoadedModule,
    pub fn_fp8_gemv_indirect_scaled_add_kround_batched: KernelFn,
    /// Phase 8 shared-expert fusion (2026-05-23): fused FP8 GEMV
    /// (single-expert, f16 in) + scaled f32 accumulate with a
    /// device-pointer scalar weight. Drop-in for the
    /// (shared_expert_down + scaled_add_f16_to_f32_devw) pair in
    /// the per-token MoE shared-expert chain. Eliminates 1 launch
    /// + 1 f16 round-trip per layer per token.
    pub fp8_gemv_f16in_scaled_add_devw_mod: LoadedModule,
    pub fn_fp8_gemv_f16in_scaled_add_devw: KernelFn,
    /// Phase 4h: paged f16 attention decode kernel
    /// (`flash_attention_2_decode_f16io_kernel`). f16 Q/K/V with a
    /// paged f16 KV cache; sliding-window param `< 0` means no window
    /// (Qwen 3.6 full-attn layers don't use sliding).
    pub flash_attention_mod: LoadedModule,
    pub fn_flash_attention_2_decode_f16io: KernelFn,
    /// Phase Full / Round-26: F16-KV prefill kernel from
    /// flash_attention.cu. Takes [num_tokens, num_heads, head_dim]
    /// f32 query, produces f32 output, with causal mask + GQA support.
    /// Used by `apply_layer_full_attn_batched` as the
    /// one-launch-per-layer attention call (replacing N decode-kernel
    /// launches in the per-token loop). The f16->f32 / f32->f16
    /// converts around it use cast_fp kernels.
    pub fn_flash_attention_2_f16kv: KernelFn,
    /// Phase Full / Round-26: f16 -> f32 elementwise cast (used to
    /// convert query rows for the f16kv prefill kernel which expects
    /// f32 query).
    pub fn_cast_f16_to_f32: KernelFn,
    /// Phase 4j: Qwen-specific element-wise `sigmoid(gate) * values`
    /// for the `attn_output_gate=true` path. New CUDA kernel added
    /// in this phase — kernels/sigmoid_mul_f16.cu.
    pub sigmoid_mul_f16_mod: LoadedModule,
    pub fn_sigmoid_mul_f16: KernelFn,
    /// Phase 4q: depthwise causal 1D convolution for the
    /// Gated-DeltaNet linear-attn block. New CUDA kernel —
    /// kernels/causal_conv1d_f16.cu.
    pub causal_conv1d_f16_mod: LoadedModule,
    pub fn_causal_conv1d_f16: KernelFn,
    /// Phase 4r: Gated-DeltaNet per-head delta-rule state update
    /// kernel — the recurrent core of the linear-attn block. New
    /// CUDA kernel — kernels/gated_delta_state_update_f16.cu.
    pub gated_delta_state_update_f16_mod: LoadedModule,
    pub fn_gated_delta_state_update_f16: KernelFn,
    /// Phase 5i: full Gated-DeltaNet decode-step kernel doing forget +
    /// delta correction + state update + readout in one launch.
    /// Replaces the host-side delta-rule loop in apply_layer_linear_attn.
    pub gated_delta_rule_decode_f16_mod: LoadedModule,
    pub fn_gated_delta_rule_decode_f16: KernelFn,
    /// Batched-prefill counterpart of `gated_delta_rule_decode_f16`.
    /// Processes `num_tokens` Q/K/V/alpha/beta inputs sequentially
    /// with the recurrent state carried inside the kernel — one
    /// launch per linear-attn layer per prefill chunk instead of
    /// per token. Kernel: kernels/gated_delta_rule_prefill_f16.cu.
    pub gated_delta_rule_prefill_f16_mod: LoadedModule,
    pub fn_gated_delta_rule_prefill_f16: KernelFn,
    /// Round-26: device-side fill of per-token `positions` and
    /// `context_lens` arrays. Replaces the legacy per-token
    /// `pos_cl_region.copy_from_host(...)` HtoD that raced with
    /// non-blocking-stream kernels; this kernel runs ON
    /// `self.stream`, so it's stream-ordered with subsequent reads.
    /// Kernel: kernels/qwen_fill_pos_slots_i32.cu.
    pub qwen_fill_pos_slots_i32_mod: LoadedModule,
    pub fn_qwen_fill_pos_slots_i32: KernelFn,
    /// Phase 8 deeper (multi-step capture): step-linker kernel that
    /// runs BETWEEN consecutive decode-step forwards inside a
    /// macro-captured graph. Copies argmax → next-token and
    /// increments pos/ctx by 1, fully device-side.
    pub qwen36_step_link_i32_mod: LoadedModule,
    pub fn_qwen36_step_link_i32: KernelFn,
    /// Phase 8 kernel fusion: argmax + step-link merged into one
    /// kernel. When `do_link != 0`, the kernel writes argmax to
    /// `argmax_token_dst[0]` AND `token_dst[0]`, plus increments
    /// pos/ctx by 1 — same effect as the separate
    /// `argmax_f16_kernel` + `qwen36_step_link_i32_kernel` pair but
    /// one launch instead of two. Used inside the macro-captured
    /// graph for iterations 0..N-2; iteration N-1 uses `do_link == 0`
    /// (pure argmax, no successor to link to).
    pub qwen36_argmax_with_link_f16_mod: LoadedModule,
    pub fn_qwen36_argmax_with_link_f16: KernelFn,
    /// Phase 6a / Round-27: batched router GEMV. Per-token grid.y
    /// dimension over the existing single-token kernel; one launch
    /// per layer instead of N. Kernel:
    /// kernels/router_gemv_batched_f16_to_f32.cu.
    pub router_gemv_batched_f16_to_f32_mod: LoadedModule,
    pub fn_router_gemv_batched_f16_to_f32: KernelFn,
    /// Phase 6a / Round-27: batched top-k+softmax. Per-token grid.x
    /// over the existing kernel; one launch per layer. Kernel:
    /// kernels/topk_softmax_batched_f32.cu.
    pub topk_softmax_batched_f32_mod: LoadedModule,
    pub fn_topk_softmax_batched_f32: KernelFn,
    /// Phase 6b / Round-27: row-batched indirect FP8 GEMV for the
    /// MoE down-proj. Per token row m, reads expert index from
    /// top_idx[m * top_k + k_round]. One launch per layer per
    /// k-round (3 ker. * 8 = 24 launches per layer instead of N*8*3).
    pub fp8_gemv_blockwise_wpr_native_f16in_indirect_batched_topk_mod: LoadedModule,
    pub fn_fp8_gemv_blockwise_wpr_native_f16in_indirect_batched_topk: KernelFn,
    /// Phase 8 MoE-fusion follow-on (2026-05-23): fused indirect
    /// FP8 GEMV + scaled f32 accumulate, batched over num_tokens.
    /// Drop-in for the (indirect_batched_topk + scaled_add_batched
    /// _topk) pair in the qwen36 prefill / batched-decode path.
    pub fp8_gemv_indirect_scaled_add_batched_topk_mod: LoadedModule,
    pub fn_fp8_gemv_indirect_scaled_add_batched_topk: KernelFn,
    /// Phase 6b / Round-27: row-batched indirect FP8 dual-silu GEMV
    /// for the MoE gate+up-proj fused path.
    pub fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_batched_topk_mod: LoadedModule,
    pub fn_fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_batched_topk: KernelFn,
    /// Phase 6b / Round-27: row-batched scaled_add. For each token
    /// m and hidden n: acc[m, n] += top_w[m, k_round] * f16_to_f32(in[m, n]).
    pub scaled_add_f16_to_f32_devw_batched_topk_mod: LoadedModule,
    pub fn_scaled_add_f16_to_f32_devw_batched_topk: KernelFn,
    /// Phase 6c / Round-27: batched shared_gate dot+sigmoid.
    /// One block per token, output [num_tokens] f32. Replaces
    /// N per-token launches of `shared_gate_dot_sigmoid_f16`.
    pub shared_gate_dot_sigmoid_f16_batched_mod: LoadedModule,
    pub fn_shared_gate_dot_sigmoid_f16_batched: KernelFn,
    /// Phase 6c / Round-27: row-batched scaled_add with per-token
    /// weight (devw[m]). Different from the topk variant because
    /// the weight indexing here is a flat [N] array rather than
    /// [N, top_k] indexed by k_round.
    pub scaled_add_f16_to_f32_devw_batched_mod: LoadedModule,
    pub fn_scaled_add_f16_to_f32_devw_batched: KernelFn,
    /// Batched-prefill conv1d state-advance + flat history assembly.
    /// Builds the `[num_tokens + ks - 1, channels]` history buffer
    /// the existing `causal_conv1d_f16` kernel expects, with state
    /// rotation applied at the end so the conv1d kernel sees the
    /// pre-rotation history. Replaces N per-token launches of the
    /// scalar `conv_state_advance_f16` kernel. Kernel:
    /// kernels/conv_state_advance_batched_f16.cu.
    pub conv_state_advance_batched_f16_mod: LoadedModule,
    pub fn_conv_state_advance_batched_f16: KernelFn,
    /// Vision Phase 1: LayerNorm with bias for Qwen ViT (norm1, norm2,
    /// merger.norm). Different from the GemmaRMSNorm-style RMSNorm
    /// used on text-side (additive +1 shift). Kernel:
    /// kernels/layernorm_inplace_f16.cu.
    pub layernorm_inplace_f16_mod: LoadedModule,
    pub fn_layernorm_inplace_f16: KernelFn,
    /// Vision Phase 1: pointwise gelu_pytorch_tanh (Qwen ViT MLP +
    /// Gemma ViT MLP). Kernel: kernels/gelu_tanh_f16.cu.
    pub gelu_tanh_f16_mod: LoadedModule,
    pub fn_gelu_tanh_f16: KernelFn,
    /// Vision Phase 1: row-wise softmax for ViT attention scores.
    /// Kernel: kernels/softmax_row_f16.cu.
    pub softmax_row_f16_mod: LoadedModule,
    pub fn_softmax_row_f16: KernelFn,
    /// Vision Phase 1: 2D-rotary applying to Q/K with per-token
    /// cos/sin tables encoding row+col axes. Kernel:
    /// kernels/vit_rotary_2d_f16.cu.
    pub vit_rotary_2d_f16_mod: LoadedModule,
    pub fn_vit_rotary_2d_f16: KernelFn,
    /// Vision Phase 1: 2D average pooling for Gemma vision tower.
    /// Kernel: kernels/vit_avgpool_f16.cu.
    pub vit_avgpool_f16_mod: LoadedModule,
    pub fn_vit_avgpool_f16: KernelFn,
    /// Vision Phase A2: bilinear interpolation of the learned absolute
    /// pos_embed table (Qwen3-VL specific). Kernel:
    /// kernels/vit_pos_embed_interp_f16.cu.
    pub vit_pos_embed_interp_f16_mod: LoadedModule,
    pub fn_vit_pos_embed_interp_f16: KernelFn,
    /// Vision: scalar in-place scale on f16 (used for the
    /// 1/sqrt(head_dim) attention-score scale).
    pub scale_inplace_f16_mod: LoadedModule,
    pub fn_scale_inplace_f16: KernelFn,
    /// Vision attention: transpose V from [N, head_dim] → [head_dim, N]
    /// so the second GEMM (scores @ V) uses our `input @ weight^T`
    /// helper without computing scores @ V^T by accident.
    pub transpose_2d_f16_mod: LoadedModule,
    pub fn_transpose_2d_f16: KernelFn,
    /// Vision helpers (also used elsewhere in the codebase): in-place
    /// per-row bias add (tensor[t,d] += bias[d]) and f32→f16 cast.
    pub add_bias_f16_mod: LoadedModule,
    pub fn_add_bias_f16: KernelFn,
    pub cast_fp_mod: LoadedModule,
    pub fn_cast_f32_to_f16: KernelFn,
    /// Vision: GPU residual add `dst[i] += src[i]` for the
    /// pre/post-attn and pre/post-MLP residual paths. Replaces the
    /// earlier DtoH-add-HtoD round-trip that synced 27× per image.
    pub vector_add_f16_mod: LoadedModule,
    pub fn_vector_add_f16: KernelFn,
    /// Vision: per-head gather/scatter from `[N, num_heads*head_dim]`
    /// in a single launch. Replaces the per-token DtoD loop that ran
    /// `n_tokens × num_heads × {Q,K,V}` async memcpys per block.
    pub extract_head_f16_mod: LoadedModule,
    pub fn_extract_head_f16: KernelFn,
    pub fn_scatter_head_f16: KernelFn,
    /// Phase-perf 2: batched ViT attention. `softmax_row_f32_to_f16`
    /// fuses scale+softmax over H×N rows; `transpose_heads_v_f16`
    /// rearranges V from [N, H*D] interleaved to [H, D, N] head-major
    /// for the second batched GEMM.
    pub softmax_row_f32_to_f16_mod: LoadedModule,
    pub fn_softmax_row_f32_to_f16: KernelFn,
    pub transpose_heads_v_f16_mod: LoadedModule,
    pub fn_transpose_heads_v_f16: KernelFn,
    pub scatter_heads_f16_mod: LoadedModule,
    pub fn_scatter_heads_f16: KernelFn,
    pub scale_inplace_f32_mod: LoadedModule,
    pub fn_scale_inplace_f32: KernelFn,
}

/// Pre-converted f32 weight caches for one linear-attention layer.
/// Built once at bring-up; consumed by `apply_layer_linear_attn`'s
/// alpha/beta computation in place of per-token DtoH+f16→f32.
#[derive(Debug)]
pub struct Qwen36LinearAttnHostCache {
    pub a_w_f32: Vec<f32>,     // [vus, h_us]  row-major
    pub b_w_f32: Vec<f32>,     // [vus, h_us]  row-major
    pub a_log_f32: Vec<f32>,   // [vus]
    pub dt_bias_f32: Vec<f32>, // [vus]
    pub vus: usize,
    pub h_us: usize,
}

pub struct Qwen36Bringup {
    pub paths: Gemma4EnginePaths,
    pub arena_bytes: usize,
    pub arch: Qwen36Arch,
    pub ctx: Arc<CudaContextHandle>,
    pub arena: HbmArena<'static>,
    pub stream: Stream,
    pub model: Qwen36LoadedModel,
    pub kernels: Arc<KernelLoader>,
    pub outside_kernels: Qwen36OutsideKernels,
    /// SM121 FA2 backend for Qwen full-attention layers. Used by the
    /// NVFP4 batched-prefill path; single-token decode still uses the
    /// already-loaded direct kernel handles in `outside_kernels`.
    attn_backend_full: rvllm_attention::AttentionBackend,
    pub cublaslt: CublasLt,
    /// CUTLASS SM120 backend for blockwise FP8 GEMM at m≥128.
    /// Loaded at bring-up; on sm_121 this resolves to
    /// `CutlassBackend::SoSm120` when `libcutlass_sm120.so` is found
    /// (the same .so Gemma uses for its lm_head fast path) and
    /// `CutlassBackend::Absent` otherwise — in which case
    /// `fp8_proj_dispatch` falls back to its looped-GEMV path.
    pub cutlass: CutlassBackend,
    /// Phase 4f: precomputed RoPE cos/sin tables for Qwen 3.6
    /// (rope_theta=10M, head_dim=256). Single-axis tables uploaded
    /// at bring-up; MRoPE's section-aware position encoding
    /// (sections [11, 11, 10]) is applied at launch time on top of
    /// these base tables in Phase 4g.
    pub rope_cos: u64,
    pub rope_sin: u64,
    pub rope_max_pos: u32,
    /// Phase 4t: per-sequence linear-attn state buffer. Sized
    /// `[num_linear_layers, num_heads, d_v, d_k]` f16. Lives ABOVE
    /// the scratch checkpoint so `arena.restore()` between requests
    /// doesn't reclaim it — state must persist across decode steps
    /// for the recurrent Gated-DeltaNet path. Single-sequence pool
    /// for now; multi-sequence batching is Phase 4v+.
    pub linear_state_ptr: u64,
    pub linear_state_bytes: usize,
    pub linear_state_layer_bytes: usize,
    /// Per-linear-attn-layer host-side f32 caches of constant
    /// weights consumed by the Gated-DeltaNet alpha/beta loop. The
    /// pre-Phase-5 implementation re-DtoH-copied these every token
    /// (in_proj_a + in_proj_b ≈ 256 KB per token, plus a_log /
    /// dt_bias) and re-converted f16→f32 in the same loop. Now we
    /// dequantise once at bring-up and cache the f32 vectors here;
    /// the per-token loop reads directly from RAM.
    /// Layout per entry: `a_w[v, k]` row-major `[vus, h_us]`,
    /// same for `b_w`; `a_log[v]` and `dt_bias[v]` are length-vus.
    pub linear_attn_host_cache: Vec<Qwen36LinearAttnHostCache>,
    /// Per-layer host-side f32 cache of the router weight matrix
    /// (`[num_experts, hidden]`). Pre-Phase-4b-prep iter15 the
    /// router GEMV did a fresh DtoH of ~1 MiB f16 weights every
    /// MoE layer × every token, then converted f16→f32 and ran
    /// the host matvec. The weight is constant — caching it as
    /// f32 once at bring-up replaces 30 MiB of per-token DtoH +
    /// 16M f16→f32 conversions with a direct RAM read.
    pub router_host_cache: Vec<Vec<f32>>,
    /// Per-layer host-side f32 cache of the shared-expert gate
    /// weight (`[hidden]`). The pre-iter16 path DtoH'd this every
    /// MoE layer × every token (4 KiB) and converted f16→f32; same
    /// caching pattern as `router_host_cache` (Phase 4b-prep iter16).
    pub shared_gate_host_cache: Vec<Vec<f32>>,
    /// Phase 4u: paged KV cache for the 10 full-attention layers.
    /// Layout when `kv_dtype == F16`:
    ///   `[num_full_layers, 2 (K+V), num_blocks, block_size,
    ///    num_kv_heads, head_dim]` f16 (2 bytes/elem).
    /// Layout when `kv_dtype == Nvfp4` (opt-in via `RVLLM_NVFP4_KV=1`):
    ///   same shape but packed 4-bit (1 byte per 2 elems), so the
    ///   buffer is half the F16 size; the companion microscale
    ///   buffer (`kv_cache_scale_ptr`) holds the per-(slot, kv_head,
    ///   block16) E4M3 scales required to reconstruct the values.
    /// Pre-allocated above the scratch checkpoint so it survives
    /// `arena.restore()` between requests. Reset alongside
    /// `reset_linear_state` on fresh sessions.
    pub kv_cache_ptr: u64,
    pub kv_cache_bytes: usize,
    pub kv_cache_layer_bytes: usize,
    pub kv_cache_num_blocks: u32,
    pub kv_cache_block_size: u32,
    /// NVFP4 commit 1: KV layout discriminator. F16 is the
    /// production default; Nvfp4 is opt-in via `RVLLM_NVFP4_KV=1`
    /// and matches the Qwen 3.5 27B NVFP4 KV plumbing — packed
    /// 4-bit K/V + per-(slot, kv_head, head_dim/16) E4M3 microscale.
    pub kv_dtype: Qwen36KvDtype,
    /// NVFP4 commit 1: companion microscale buffer pointer. Zero
    /// when `kv_dtype == F16`. When NVFP4 is on, layout is
    ///   `[num_full_layers, 2 (K+V), num_blocks, block_size,
    ///    num_kv_heads, head_dim/16]` __nv_fp8_e4m3 (1 byte/elem).
    /// One scale per 16-element NVFP4 block.
    pub kv_cache_scale_ptr: u64,
    pub kv_cache_scale_bytes: usize,
    pub kv_cache_scale_layer_bytes: usize,
    /// Persistent device pointer to the identity block table
    /// `[0, 1, …, kv_cache_num_blocks-1]` i32. The paged-attention
    /// path used to rebuild + re-upload this constant table every
    /// full-attn layer × every token; now we upload once at bring-up
    /// (Phase 4b-prep iter25).
    pub bt_persistent_ptr: u64,
    /// Phase 5f: persistent conv1d state cache for linear-attn layers.
    /// `[num_linear_layers, conv_kernel-1=3, conv_dim=8192]` f16.
    /// Holds the previous (kernel-1) conv-input timesteps so per-token
    /// causal_conv1d can attend to actual prior tokens (not zeros).
    /// Reset alongside reset_linear_state on session boundaries.
    pub conv_state_ptr: u64,
    pub conv_state_bytes: usize,
    pub conv_state_layer_bytes: usize,
    /// Phase 8 commit 3: lazily-populated single-token-decode CUDA
    /// graph. Populated by `try_capture_decode_step` on the first
    /// captured step; replayed by `replay_decode_step` thereafter.
    /// `None` when capture has not run yet OR when capture was
    /// attempted and rejected (eager path stays in use). Reset by
    /// the operator via dropping the bringup (per-request reset is
    /// handled at the call site).
    pub decode_capture:
        std::sync::Mutex<Option<rvllm_graph::pool::CapturedGraph>>,
    /// Phase 8 deeper-optimization slot: separate captured graph
    /// for the N-step macro-replay path. Distinct from
    /// `decode_capture` (single-step) because the macro-graph
    /// records N iterations of forward + N-1 step_link kernels.
    /// Bucket=N tags the graph with its macro-step count;
    /// re-capture is needed if the operator changes
    /// `RVLLM_QWEN36_DECODE_MULTI_STEP` mid-process (just restart
    /// the worker for that).
    pub decode_capture_multi_step:
        std::sync::Mutex<Option<rvllm_graph::pool::CapturedGraph>>,
}

/// Output of `Qwen36Bringup::forward_qwen_vision`.
pub struct VisionForwardOutput {
    /// Raw little-endian f16 bytes, layout `[num_tokens, hidden_dim]`.
    pub data: Vec<u8>,
    pub num_tokens: usize,
    pub hidden_dim: usize,
    pub grid_thw: [u32; 3],
}

/// Phase 8 follow-on: RAII guard that restores the arena bump
/// pointer to a captured checkpoint when dropped. Used at the entry
/// of `forward_qwen36_decode_inner_with_workspace_overrides_v2` to
/// bound per-call arena growth + make per-call allocation addresses
/// deterministic across decode steps + across requests.
///
/// Drop ordering vs CUDA work in flight: `arena.restore` is a pure
/// bump-pointer mutation (no `cuFree`), so kernels still running on
/// the stream that hold pointers into the restored region keep
/// reading valid GPU memory. The next caller allocates into the
/// same address range AFTER the previous kernels have completed
/// (subsequent `arena.region` calls themselves go through the same
/// stream's prior submissions). The captured-graph replay reads
/// the restored region's address but the kernel sequence
/// immediately re-fills it, so by the time the result is consumed,
/// the data is fresh.
struct Qwen36DecodeArenaGuard<'a> {
    arena: &'a rvllm_mem::HbmArena<'static>,
    checkpoint: usize,
}

impl<'a> Qwen36DecodeArenaGuard<'a> {
    fn new(arena: &'a rvllm_mem::HbmArena<'static>, checkpoint: usize) -> Self {
        Self { arena, checkpoint }
    }
}

impl<'a> Drop for Qwen36DecodeArenaGuard<'a> {
    fn drop(&mut self) {
        unsafe { self.arena.restore(self.checkpoint); }
    }
}

impl Qwen36Bringup {
    /// Phase 8 commit 3: try to capture a single decode step into a
    /// CUDA graph. The eager body runs once during capture (so this
    /// call still returns the real argmax token via the workspace);
    /// the captured `CapturedGraph` exec handle is stored on
    /// `self.decode_capture` for subsequent
    /// [`Self::replay_decode_step`] calls.
    ///
    /// On capture failure (residual host-sync inside the body, PTX
    /// quirk, etc.) the body still ran eagerly, the workspace's
    /// `argmax_token_dev` holds the result, and the
    /// `decode_capture` slot stays None — caller falls back to
    /// eager mode for the remainder of the request.
    ///
    /// Caller contract: write `workspace.token_dev` BEFORE calling
    /// (via cuMemcpyHtoDAsync or cuMemsetD32Async); read the
    /// result via `argmax_dev_to_host_token(workspace.argmax_token_dev)`
    /// AFTER.
    #[cfg(feature = "cuda")]
    pub fn try_capture_decode_step(
        &self,
        workspace:
            &crate::qwen36_decode_workspace::Qwen36DecodeWorkspace,
        position: u32,
    ) -> Result<()> {
        let stream_u64 = self.stream.raw() as u64;
        let layout = rvllm_metadata::MetadataLayout::compute(1, 1);
        let layout_hash = layout.hash();
        let fingerprint = rvllm_graph::pool::GraphFingerprint([0u8; 32]);
        // Capture closure: just the workspace-driven step. The
        // closure RUNS the body eagerly during `cuStreamEndCapture`
        // — `argmax_token_dev` holds the result regardless of
        // whether the graph instantiation succeeds.
        let capture_result = unsafe {
            rvllm_graph::pool::CapturedGraph::capture(
                /* bucket */ 0,
                /* max_blocks */ 0,
                layout_hash,
                fingerprint,
                stream_u64,
                || -> Result<()> {
                    self.forward_qwen36_decode_step_to_workspace(
                        workspace, position)
                },
            )
        };
        match capture_result {
            Ok(g) => {
                // Phase 8 debug 2026-05-22: CUDA stream capture in
                // THREAD_LOCAL mode RECORDS kernel launches but does
                // NOT execute them eagerly (despite earlier
                // assumptions in our comments). The `body()` closure
                // submits kernels which are routed to the graph,
                // leaving the KV cache + argmax slot unmodified.
                // Without an explicit replay here, step 0 returns a
                // stale `workspace.argmax_token_dev`, the KV slot at
                // step 0's position stays empty, and step 1+ runs
                // forward from a broken KV state — manifesting as
                // truncated output like "Die.".
                //
                // Fix: replay the freshly-captured graph
                // immediately. The graph contains exactly the body's
                // kernel sequence; replaying it executes those
                // kernels on `stream` for real, producing the
                // correct argmax + advancing KV state. The graph is
                // then stored on `self.decode_capture` for any
                // subsequent replay opt-in.
                unsafe { g.replay(self.stream.raw() as u64)?; }
                let mut guard = self.decode_capture.lock().unwrap();
                *guard = Some(g);
                Ok(())
            }
            Err(e) => {
                // Capture FAILED. Per the CUDA semantics above the
                // body's launches did NOT execute either. The
                // workspace's argmax_token_dev is stale and KV state
                // wasn't advanced. Re-run the body eagerly to
                // produce a correct result + advance KV state.
                tracing::warn!(
                    "qwen36: decode-step graph capture rejected: {e:?}. \
                     Falling back to eager (workspace path) for this iter."
                );
                self.forward_qwen36_decode_step_to_workspace(workspace, position)
            }
        }
    }

    /// Phase 8 commit 3: replay a previously-captured decode step.
    /// Caller must have already populated `workspace.token_dev`
    /// with the current step's token id (via cuMemcpyHtoDAsync or
    /// cuMemsetD32Async) BEFORE calling. Result lands in
    /// `workspace.argmax_token_dev`; extract via
    /// `argmax_dev_to_host_token` after.
    ///
    /// **Position-indirect limitation**: the captured graph holds
    /// RoPE / KV-slot kernel scalar args from capture time. Until
    /// position-indirect kernel variants land (Qwen's equivalent of
    /// Gemma 4 NVFP4's `G4N_DECODE_GRAPH_INDIRECT=1`), replay is
    /// only correct at the SAME `position` value as capture.
    /// `Err(NotCaptured)` when nothing was captured yet.
    #[cfg(feature = "cuda")]
    pub fn replay_decode_step(
        &self,
        _workspace:
            &crate::qwen36_decode_workspace::Qwen36DecodeWorkspace,
    ) -> Result<()> {
        let guard = self.decode_capture.lock().unwrap();
        let g = match guard.as_ref() {
            Some(g) => g,
            None => return Err(rvllm_core::RvllmError::cuda(
                "qwen36 replay_decode_step: no captured graph; \
                 call try_capture_decode_step first",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )),
        };
        unsafe { g.replay(self.stream.raw() as u64)?; }
        Ok(())
    }

    /// Phase 8 helper: write `token_id` to `workspace.token_dev`
    /// via cuMemcpyHtoDAsync. Stand-alone entry for the
    /// workspace-eager isolation path that the worker uses to
    /// validate the workspace forward without going through
    /// `decode_step_via_graph_or_eager` (which always touches the
    /// capture machinery). 4-byte async HtoD on
    /// `self.stream`; same-stream ordering covers the subsequent
    /// embed_gather read.
    #[cfg(feature = "cuda")]
    pub fn write_token_to_workspace(
        &self,
        workspace:
            &crate::qwen36_decode_workspace::Qwen36DecodeWorkspace,
        token_id: i32,
    ) -> Result<()> {
        use cudarc::driver::sys::*;
        let tok_host = token_id;
        unsafe {
            let rc = cuMemcpyHtoDAsync_v2(
                workspace.token_dev as CUdeviceptr,
                (&tok_host) as *const i32 as *const _,
                4,
                self.stream.raw() as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 write_token_to_workspace HtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Phase 8 kernel fusion: graph-capture-friendly closer that
    /// fuses lm_head argmax with the step-link side-effects. Mirror
    /// of `forward_qwen36_outside_closer_device_argmax` but the
    /// trailing argmax kernel is `qwen36_argmax_with_link_f16_kernel`
    /// instead of plain `argmax_f16_kernel`. When `do_link != 0`,
    /// the kernel ALSO writes argmax → `token_dst` and increments
    /// `pos_dst` / `ctx_dst` — same effect as a separate
    /// `qwen36_step_link_i32_kernel` launch, but ONE launch instead
    /// of TWO inside the macro-captured graph.
    ///
    /// The first three stages (rmsnorm + fp8_quant + fp8_gemm) are
    /// byte-identical to `forward_qwen36_outside_closer_device_argmax`
    /// — diverges only at the tail (fused vs plain argmax kernel).
    #[cfg(feature = "cuda")]
    fn forward_qwen36_outside_closer_device_argmax_with_link(
        &self,
        hidden_dev_ptr: u64,
        num_tokens: u32,
        hidden: u32,
        vocab: u32,
        last_idx: usize,
        argmax_token_dst: u64,
        token_dst: u64,
        pos_dst: u64,
        ctx_dst: u64,
        do_link: i32,
    ) -> Result<()> {
        let _ = num_tokens;
        let eps = self.arch.base.rms_norm_eps;
        let hidden_fp8_bytes = hidden as usize;
        let hidden_scale_bytes = 4usize;
        let logits_bytes = (vocab as usize) * 2;
        let hidden_fp8_region =
            self.arena.region("qwen36_pl_h_fp8_argl", hidden_fp8_bytes, 16)?;
        let hidden_scale_region =
            self.arena.region("qwen36_pl_h_scale_argl", hidden_scale_bytes, 16)?;
        let logits_region =
            self.arena.region("qwen36_pl_logits_argl", logits_bytes, 16)?;
        let stream_raw = self.stream.raw() as u64;
        let last_hidden_row_ptr =
            hidden_dev_ptr + (last_idx as u64) * (hidden as u64) * 2;
        unsafe {
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: 1,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_fused_rmsnorm_fp8_quant,
                hidden_fp8_region.device_ptr(),
                hidden_scale_region.device_ptr(),
                last_hidden_row_ptr,
                self.model.outside.final_norm.offset_bytes,
                stream_raw,
            )?;
            self.cublaslt.fp8_gemm(
                hidden_fp8_region.device_ptr(),
                self.model.outside.lm_head_fp8.offset_bytes,
                logits_region.device_ptr(),
                1,
                vocab as i32,
                hidden as i32,
                hidden_scale_region.device_ptr(),
                self.model.outside.lm_head_fp8.scale_ptr,
                stream_raw,
            )?;
            use cudarc::driver::sys::*;
            let mut a_logits = logits_region.device_ptr();
            let mut a_argmax_dst = argmax_token_dst;
            let mut a_token_dst = token_dst;
            let mut a_pos_dst = pos_dst;
            let mut a_ctx_dst = ctx_dst;
            let mut a_vocab = vocab as i32;
            let mut a_link = do_link;
            let args: [*mut core::ffi::c_void; 7] = [
                (&mut a_logits)     as *mut _ as *mut _,
                (&mut a_argmax_dst) as *mut _ as *mut _,
                (&mut a_token_dst)  as *mut _ as *mut _,
                (&mut a_pos_dst)    as *mut _ as *mut _,
                (&mut a_ctx_dst)    as *mut _ as *mut _,
                (&mut a_vocab)      as *mut _ as *mut _,
                (&mut a_link)       as *mut _ as *mut _,
            ];
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_qwen36_argmax_with_link_f16.raw() as CUfunction,
                1, 1, 1,
                512, 1, 1,
                0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 argmax_with_link_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Phase 8 deeper: launch the step-linker kernel device-side
    /// between two consecutive decode-step iterations inside the
    /// macro-captured graph. Reads `argmax_token_src`, writes to
    /// `workspace.token_dev`, and increments `workspace.pos_dev` +
    /// `workspace.ctx_dev` by 1 — all in one tiny kernel.
    #[cfg(feature = "cuda")]
    fn launch_step_link_i32(
        &self,
        argmax_token_src: u64,
        workspace:
            &crate::qwen36_decode_workspace::Qwen36DecodeWorkspace,
    ) -> Result<()> {
        use cudarc::driver::sys::*;
        unsafe {
            let mut a_src = argmax_token_src;
            let mut a_tok = workspace.token_dev;
            let mut a_pos = workspace.pos_dev;
            let mut a_ctx = workspace.ctx_dev;
            let args: [*mut core::ffi::c_void; 4] = [
                (&mut a_src) as *mut _ as *mut _,
                (&mut a_tok) as *mut _ as *mut _,
                (&mut a_pos) as *mut _ as *mut _,
                (&mut a_ctx) as *mut _ as *mut _,
            ];
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_qwen36_step_link_i32.raw() as CUfunction,
                1, 1, 1,
                32, 1, 1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 step_link_i32 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Phase 8 deeper: capture N consecutive decode steps as ONE
    /// macro-graph. Caller has already populated `workspace.token_dev`
    /// (first token), `workspace.pos_dev` (start position), and
    /// `workspace.ctx_dev` (start_position+1). The captured body:
    ///
    ///   * iter 0: forward step → writes argmax to
    ///     `argmax_tokens_dev_base[0]`. (Single-step path's
    ///     `workspace.argmax_token_dev` aliases this slot.)
    ///   * step_link kernel: `workspace.token_dev = argmax_tokens
    ///     [0]; workspace.pos_dev += 1; workspace.ctx_dev += 1`.
    ///   * iter 1: forward step → writes argmax to
    ///     `argmax_tokens_dev_base[1]`. (For this iteration the
    ///     forward kernel sequence is captured with a SHIFTED sub-
    ///     workspace whose `argmax_token_dev = base + 1*4`.)
    ///   * ... and so on for iter 2..N-1.
    ///
    /// On capture failure (residual sync somewhere), the body still
    /// ran eagerly under capture rules — replay would have failed
    /// to instantiate, so the caller falls back to per-step replay.
    /// Stores the macro-graph on `self.decode_capture_multi_step`
    /// (separate slot from the single-step `decode_capture`).
    #[cfg(feature = "cuda")]
    pub fn try_capture_decode_steps_n(
        &self,
        workspace:
            &crate::qwen36_decode_workspace::Qwen36DecodeWorkspace,
        position: u32,
        n_steps: u32,
    ) -> Result<()> {
        if n_steps < 2 {
            // n=1 reduces to single-step capture; reuse the existing
            // entry. Sanity: callers should gate on multi_step > 1
            // before calling this.
            return self.try_capture_decode_step(workspace, position);
        }
        if n_steps > workspace.max_steps {
            return Err(rvllm_core::RvllmError::cuda(
                "try_capture_decode_steps_n: n_steps exceeds workspace.max_steps",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        let stream_u64 = self.stream.raw() as u64;
        let layout = rvllm_metadata::MetadataLayout::compute(1, 1);
        let layout_hash = layout.hash();
        let fingerprint = rvllm_graph::pool::GraphFingerprint([0u8; 32]);
        let base_argmax = workspace.argmax_tokens_dev_base;
        let capture_result = unsafe {
            rvllm_graph::pool::CapturedGraph::capture(
                /* bucket */ n_steps,
                /* max_blocks */ 0,
                layout_hash,
                fingerprint,
                stream_u64,
                || -> Result<()> {
                    // Phase 8 kernel-fusion: each iter runs the
                    // layer stack via the skip-closer wrapper, then
                    // the FUSED argmax+link closer emits per-iter
                    // argmax AND (when not last) writes the next-
                    // iter token + bumps pos/ctx in ONE launch.
                    // Replaces the (plain argmax + step_link_i32)
                    // PAIR with ONE kernel — saves N kernel-graph
                    // nodes per macro-block (1 closer + 0 linker
                    // per iter, vs 1 closer + 1 linker before).
                    //
                    // Phase 8 hidden-state→workspace (e13e2eb, this
                    // commit's follow-up): the fused closer now
                    // reads from the PERSISTENT `workspace.hidden_dev`
                    // slot directly, NOT from a per-iter
                    // arena.region("qwen36_pl_hidden", ...) re-
                    // allocation. The decode_step's
                    // forward_qwen36_decode_step_to_workspace_no_closer
                    // ALREADY routes its hidden writes to
                    // workspace.hidden_dev (via hidden_dev_override).
                    // The closer reads from the same address →
                    // body-writes-then-closer-reads ordering holds
                    // per iteration on the stream. The per-iter
                    // arena.region allocation is now retired; the
                    // captured graph references a buffer that's
                    // address-stable across the worker's lifetime.
                    let hidden = self.arch.base.hidden_size as u32;
                    let vocab = self.arch.base.vocab_size as u32;
                    for i in 0..n_steps {
                        self.forward_qwen36_decode_step_to_workspace_no_closer(
                            workspace, position + i)?;
                        let argmax_dst = base_argmax + (i as u64) * 4;
                        let do_link = if i + 1 < n_steps { 1 } else { 0 };
                        self.forward_qwen36_outside_closer_device_argmax_with_link(
                            workspace.hidden_dev,
                            /* num_tokens */ 1,
                            hidden,
                            vocab,
                            /* last_idx */ 0,
                            argmax_dst,
                            workspace.token_dev,
                            workspace.pos_dev,
                            workspace.ctx_dev,
                            do_link,
                        )?;
                    }
                    Ok(())
                },
            )
        };
        match capture_result {
            Ok(g) => {
                // Per the explicit-replay-after-capture invariant
                // (commit bb9a3cf): CUDA stream capture in
                // THREAD_LOCAL mode records but doesn't execute
                // kernels. Replay once to actually run the macro-
                // graph for THIS first call's result.
                unsafe { g.replay(stream_u64)?; }
                let mut guard = self.decode_capture_multi_step.lock().unwrap();
                *guard = Some(g);
                Ok(())
            }
            Err(e) => {
                tracing::warn!(
                    "qwen36 multi-step capture rejected ({n_steps} steps): {e:?}. \
                     Falling back to per-step replay path."
                );
                // Body kernels weren't executed under capture; re-
                // run eagerly via the fused closer path (same as
                // the capture body above, also using workspace
                // .hidden_dev) so the host has correct argmax
                // tokens in `argmax_tokens_dev_base[0..n_steps]`.
                let hidden = self.arch.base.hidden_size as u32;
                let vocab = self.arch.base.vocab_size as u32;
                for i in 0..n_steps {
                    self.forward_qwen36_decode_step_to_workspace_no_closer(
                        workspace, position + i)?;
                    let argmax_dst = base_argmax + (i as u64) * 4;
                    let do_link = if i + 1 < n_steps { 1 } else { 0 };
                    self.forward_qwen36_outside_closer_device_argmax_with_link(
                        workspace.hidden_dev,
                        1, hidden, vocab, 0,
                        argmax_dst,
                        workspace.token_dev,
                        workspace.pos_dev,
                        workspace.ctx_dev,
                        do_link,
                    )?;
                }
                Ok(())
            }
        }
    }

    /// Phase 8 deeper: replay a previously-captured N-step
    /// macro-graph. Worker has already populated the first
    /// iteration's token/pos/ctx into the workspace's shared
    /// slots; the captured graph re-runs the whole N-step chain.
    /// Returns immediately — caller does the DtoH of
    /// `argmax_tokens_dev_base[0..n_steps]` to harvest the N tokens.
    #[cfg(feature = "cuda")]
    pub fn replay_decode_steps_n(
        &self,
        _workspace:
            &crate::qwen36_decode_workspace::Qwen36DecodeWorkspace,
    ) -> Result<()> {
        let guard = self.decode_capture_multi_step.lock().unwrap();
        let g = match guard.as_ref() {
            Some(g) => g,
            None => return Err(rvllm_core::RvllmError::cuda(
                "qwen36 replay_decode_steps_n: no captured macro-graph; \
                 call try_capture_decode_steps_n first",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )),
        };
        unsafe { g.replay(self.stream.raw() as u64)?; }
        Ok(())
    }

    /// Phase 8 deeper: DtoH harvest of N argmax tokens from the
    /// macro-replay's per-iteration argmax slots. Fences the stream
    /// first so all captured kernels are observed.
    #[cfg(feature = "cuda")]
    pub fn argmax_tokens_dev_to_host(
        &self,
        workspace:
            &crate::qwen36_decode_workspace::Qwen36DecodeWorkspace,
        n_steps: u32,
    ) -> Result<Vec<i32>> {
        if n_steps == 0 { return Ok(Vec::new()); }
        if n_steps > workspace.max_steps {
            return Err(rvllm_core::RvllmError::cuda(
                "argmax_tokens_dev_to_host: n_steps > max_steps",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        self.stream.fence()?;
        let mut buf = vec![0i32; n_steps as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut _,
                workspace.argmax_tokens_dev_base,
                (n_steps as usize) * 4,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 argmax_tokens_dev_to_host DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(buf)
    }

    /// Phase 8 deeper: high-level macro-decode entry. Worker calls
    /// once per `n_steps`-token chunk. Mirrors
    /// `decode_step_via_graph_or_eager` but for the N-step macro
    /// path. Capture-on-first-use, replay subsequent calls — the
    /// macro-graph is keyed by `n_steps` (one cached graph per
    /// distinct macro-step count).
    ///
    /// Caller contract:
    ///   1. Write the first token to `workspace.token_dev` (the
    ///      worker already does this via
    ///      `write_token_to_workspace`).
    ///   2. Write the starting position to `workspace.pos_dev` +
    ///      pos+1 to `workspace.ctx_dev` (via
    ///      `write_position_to_workspace`).
    ///   3. Call this method.
    ///   4. Receive `Vec<i32>` of N argmax tokens.
    #[cfg(feature = "cuda")]
    pub fn decode_steps_n_via_graph_or_eager(
        &self,
        workspace:
            &crate::qwen36_decode_workspace::Qwen36DecodeWorkspace,
        first_token: i32,
        position: u32,
        n_steps: u32,
    ) -> Result<Vec<i32>> {
        self.write_token_to_workspace(workspace, first_token)?;
        self.write_position_to_workspace(workspace, position)?;
        let graph_enabled =
            crate::qwen36_decode_workspace::qwen36_decode_graph_enabled();
        let replay_enabled = std::env::var("RVLLM_QWEN36_DECODE_GRAPH_REPLAY")
            .ok()
            .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            .unwrap_or(false);
        let have_capture = {
            let guard = self.decode_capture_multi_step.lock().unwrap();
            guard.is_some()
        };
        if graph_enabled && replay_enabled && have_capture {
            self.replay_decode_steps_n(workspace)?;
            return self.argmax_tokens_dev_to_host(workspace, n_steps);
        }
        if graph_enabled && !have_capture {
            self.try_capture_decode_steps_n(workspace, position, n_steps)?;
        } else {
            // Pure eager path (no capture machinery). Same fused-
            // closer body as the captured macro-block — both read
            // hidden state from the persistent workspace.hidden_dev
            // slot (Phase 8 hidden-state→workspace refactor).
            let base_argmax = workspace.argmax_tokens_dev_base;
            let hidden = self.arch.base.hidden_size as u32;
            let vocab = self.arch.base.vocab_size as u32;
            for i in 0..n_steps {
                self.forward_qwen36_decode_step_to_workspace_no_closer(
                    workspace, position + i)?;
                let argmax_dst = base_argmax + (i as u64) * 4;
                let do_link = if i + 1 < n_steps { 1 } else { 0 };
                self.forward_qwen36_outside_closer_device_argmax_with_link(
                    workspace.hidden_dev,
                    1, hidden, vocab, 0,
                    argmax_dst,
                    workspace.token_dev,
                    workspace.pos_dev,
                    workspace.ctx_dev,
                    do_link,
                )?;
            }
        }
        self.argmax_tokens_dev_to_host(workspace, n_steps)
    }

    /// Phase 8 deeper: clear the cached multi-step macro-graph.
    /// Called by the worker before re-capture when the per-request
    /// arena layout changes (mirrors the single-step
    /// `clear_decode_capture`).
    pub fn clear_decode_capture_multi_step(&self) {
        let mut guard = self.decode_capture_multi_step.lock().unwrap();
        *guard = None;
    }

    /// Phase 8: clear any cached captured graph. The captured graph
    /// holds device-pointer references to per-call arena
    /// allocations (positions_region, hidden_region, q_split_region,
    /// etc.) which are released by `arena.restore(scratch_ck)` at
    /// the END of each request. The worker MUST call this BEFORE
    /// the next request's `alloc_decode_workspace` so a stale
    /// captured graph from request N doesn't replay against
    /// reclaimed memory in request N+1 (which manifests as a hang
    /// or wrong output).
    pub fn clear_decode_capture(&self) {
        let mut guard = self.decode_capture.lock().unwrap();
        *guard = None;
    }

    /// Phase 8 commit 3: high-level decode-step entry that picks
    /// capture vs eager vs replay based on the operator gate
    /// (`RVLLM_QWEN36_DECODE_GRAPH=1`) and the current capture
    /// state. Production cuda_worker can call this once per
    /// decode iteration:
    ///
    ///   1. Eager (capture disabled or NOT yet captured): runs
    ///      `forward_qwen36_decode_step_to_workspace` eagerly,
    ///      optionally attempts capture for next time.
    ///   2. Replay (capture present): writes the new token to
    ///      `workspace.token_dev` via HtoD-async, replays the
    ///      captured graph.
    ///   3. Either way: returns the resulting token after
    ///      `argmax_dev_to_host_token`.
    ///
    /// `position` is the absolute position for this decode step
    /// (start_position + step_index). With current kernels, replay
    /// requires position == capture-position; multi-step replay
    /// across positions is the remaining position-indirect work.
    #[cfg(feature = "cuda")]
    pub fn decode_step_via_graph_or_eager(
        &self,
        workspace:
            &crate::qwen36_decode_workspace::Qwen36DecodeWorkspace,
        token_id: i32,
        position: u32,
    ) -> Result<i32> {
        let stream_u64 = self.stream.raw() as u64;
        // Write the current token to workspace.token_dev — both
        // capture and replay paths read from this stable pointer.
        self.write_token_to_workspace(workspace, token_id)?;
        // Phase 8 position-indirect: write the current position to
        // workspace.pos_dev / ctx_dev so captured-replay reads the
        // updated values (not step-0's frozen scalar).
        self.write_position_to_workspace(workspace, position)?;
        let _ = stream_u64;

        let graph_enabled =
            crate::qwen36_decode_workspace::qwen36_decode_graph_enabled();
        // **Two-gate design**: `RVLLM_QWEN36_DECODE_GRAPH=1` enables
        // ATTEMPTING capture (validates that the workspace-driven body
        // is capture-clean: no residual sync HtoD/DtoH). Actual
        // REPLAY of the captured graph requires the SECOND env knob
        // `RVLLM_QWEN36_DECODE_GRAPH_REPLAY=1`. The split exists
        // because replay is incorrect at moving positions until
        // Qwen's RoPE + KV-slot kernels grow indirect-args variants
        // (their position scalar is captured into the graph; replay
        // re-applies it as a frozen value, producing garbage tokens
        // from step 2 onward). Operators validating the capture
        // infrastructure can set the first gate alone (always eager
        // + first-step capture attempt); operators experimenting
        // with the parked indirect-kernel path enable both.
        let replay_enabled = std::env::var("RVLLM_QWEN36_DECODE_GRAPH_REPLAY")
            .ok()
            .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            .unwrap_or(false);
        let have_capture = {
            let guard = self.decode_capture.lock().unwrap();
            guard.is_some()
        };

        if graph_enabled && replay_enabled && have_capture {
            // Replay path (currently incorrect for position > capture
            // position — operator-opt-in for indirect-kernel A/B).
            self.replay_decode_step(workspace)?;
            return self.argmax_dev_to_host_token(workspace.argmax_token_dev);
        }

        // Eager path. When graph_enabled but no capture yet, the
        // first call eagerly runs AND attempts to capture for the
        // next call. With replay_enabled OFF (the default), every
        // step continues eager — capture is only validated, not
        // consumed. When graph_disabled, just runs eager and never
        // attempts capture.
        if graph_enabled && !have_capture {
            self.try_capture_decode_step(workspace, position)?;
        } else {
            self.forward_qwen36_decode_step_to_workspace(
                workspace, position)?;
        }
        self.argmax_dev_to_host_token(workspace.argmax_token_dev)
    }

    /// Phase 8 commit 2b: workspace-driven single-step decode.
    /// Combines the two overrides on
    /// `forward_qwen36_decode_inner_with_workspace_overrides` so a
    /// single call from the cuda-worker drives one decode step
    /// using the workspace's stable device pointers throughout
    /// (token in, argmax out). NO sync HtoD and NO sync DtoH inside.
    ///
    /// Caller contract:
    ///   1. Write the current token id (i32) to `workspace.token_dev`
    ///      via `cuMemcpyHtoDAsync_v2` (4 bytes) or
    ///      `cuMemsetD32Async` BEFORE calling this method.
    ///   2. Call this method.
    ///   3. Call `argmax_dev_to_host_token(workspace.argmax_token_dev)`
    ///      to fence + extract the resulting token.
    ///
    /// The captured-graph wrapping in cuda_worker (commit 3) records
    /// step 2 into a CUDA graph; replay re-runs the same kernel
    /// sequence with whatever token_id was just written to
    /// `workspace.token_dev` in step 1. Token IDs change per
    /// replay; the rest of the kernel arg vector is captured-stable.
    ///
    /// Position is currently still passed as a host-side scalar
    /// kernel arg (RoPE, KV slot indexing). For the captured-graph
    /// path to support a moving position, the RoPE + KV-slot kernel
    /// variants need "indirect" siblings that read position from a
    /// device i32 pointer (same pattern as Gemma 4 NVFP4's
    /// `G4N_DECODE_GRAPH_INDIRECT=1`). Until those land, the
    /// captured path is single-position only — useful for
    /// validating the capture/replay infrastructure, but production
    /// decode loops need the indirect kernels for multi-step
    /// replay.
    #[cfg(feature = "cuda")]
    pub fn forward_qwen36_decode_step_to_workspace(
        &self,
        workspace:
            &crate::qwen36_decode_workspace::Qwen36DecodeWorkspace,
        position: u32,
    ) -> Result<()> {
        // The single-token slice is unused on the override path —
        // the embed_gather reads token_dev directly. Passing an
        // arbitrary i32 so the inner function's debug paths still
        // see a non-empty slice when something downstream needs it.
        let dummy_tok = [0i32; 1];
        self.forward_qwen36_decode_inner_with_workspace_overrides_v2(
            &dummy_tok,
            position,
            /* vision_splice */ &[],
            /* cancel */ None,
            /* all_argmaxes */ None,
            /* skip_closer */ false,
            /* mtp_shadow_out */ None,
            Some(workspace.token_dev),
            Some(workspace.argmax_token_dev),
            // Phase 8 position-indirect: stable workspace pos/ctx
            // slots. Worker writes both BEFORE calling this method
            // (see Qwen36Bringup::write_position_to_workspace).
            // Captured graph reads from these stable pointers, so
            // replay picks up per-step values without freezing
            // step-0's scalar.
            Some(workspace.pos_dev),
            Some(workspace.ctx_dev),
            // Phase 8 hidden-state→workspace: route the residual
            // stream through the persistent `workspace.hidden_dev`
            // slot so the captured-graph references survive the
            // inner-checkpoint restore + stay address-stable
            // across requests.
            Some(workspace.hidden_dev),
        )?;
        Ok(())
    }

    /// Phase 8 kernel fusion: skip-closer variant of
    /// `forward_qwen36_decode_step_to_workspace`. Runs the full
    /// layer stack + KV writes + linear-state advancement but
    /// returns BEFORE the closer kernel sequence — caller invokes
    /// the closer separately (typically via
    /// `forward_qwen36_outside_closer_device_argmax_with_link` so
    /// the per-iteration argmax+link fuse into one kernel inside
    /// the macro-captured graph).
    #[cfg(feature = "cuda")]
    pub fn forward_qwen36_decode_step_to_workspace_no_closer(
        &self,
        workspace:
            &crate::qwen36_decode_workspace::Qwen36DecodeWorkspace,
        position: u32,
    ) -> Result<()> {
        let dummy_tok = [0i32; 1];
        self.forward_qwen36_decode_inner_with_workspace_overrides_v2(
            &dummy_tok,
            position,
            /* vision_splice */ &[],
            /* cancel */ None,
            /* all_argmaxes */ None,
            /* skip_closer */ true,
            /* mtp_shadow_out */ None,
            Some(workspace.token_dev),
            // closer_argmax_dev is meaningless when skip_closer=true,
            // but pass `argmax_token_dev` anyway so the override
            // shape stays uniform.
            Some(workspace.argmax_token_dev),
            Some(workspace.pos_dev),
            Some(workspace.ctx_dev),
            // Phase 8 hidden-state→workspace: route the residual
            // stream through the persistent workspace slot.
            Some(workspace.hidden_dev),
        )?;
        Ok(())
    }

    /// Phase 8 helper: write the per-step `position` to
    /// `workspace.pos_dev` and `position + 1` to `workspace.ctx_dev`
    /// (the qwen36 context-length convention is `position + 1` for
    /// a causal decode step). Both via cuMemsetD32Async on the
    /// stream. Companion to `write_token_to_workspace` — the worker
    /// calls both before each decode step.
    #[cfg(feature = "cuda")]
    pub fn write_position_to_workspace(
        &self,
        workspace:
            &crate::qwen36_decode_workspace::Qwen36DecodeWorkspace,
        position: u32,
    ) -> Result<()> {
        use cudarc::driver::sys::*;
        unsafe {
            let stream = self.stream.raw() as CUstream;
            let rc1 = cuMemsetD32Async(
                workspace.pos_dev as CUdeviceptr,
                position,
                1,
                stream,
            );
            let rc2 = cuMemsetD32Async(
                workspace.ctx_dev as CUdeviceptr,
                position + 1,
                1,
                stream,
            );
            if rc1 != CUresult::CUDA_SUCCESS || rc2 != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 write_position_to_workspace cuMemsetD32Async",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Phase 8 scaffold: allocate the stable per-step decode workspace
    /// for one request. Must be called BEFORE the per-request scratch
    /// checkpoint so subsequent `arena.restore(ck)` doesn't reclaim
    /// the workspace regions. Consumer wiring (the captured
    /// `decode_step_launch_only` body) lands in the follow-up commit.
    ///
    /// `qwen36_decode_graph_enabled()` is the operator gate for the
    /// captured path; when off, this method can be skipped entirely
    /// and the eager `forward_qwen36_decode_inner` runs untouched.
    #[cfg(feature = "cuda")]
    pub fn alloc_decode_workspace(
        &self,
    ) -> Result<crate::qwen36_decode_workspace::Qwen36DecodeWorkspace> {
        crate::qwen36_decode_workspace::Qwen36DecodeWorkspace::alloc(
            &self.arena, &self.arch)
    }

    /// Phase 1: CUDA init + arena + outside-tensor upload.
    /// Returns `Err` if `config.json` is missing required Qwen-3.6
    /// markers, or if any of the three outside tensors is missing.
    pub fn load(paths: Gemma4EnginePaths, arena_bytes: usize) -> Result<Self> {
        let arch = match Qwen36Arch::from_dir(&paths.model_dir)? {
            Some(a) => a,
            None => {
                panic!(
                    "Qwen36Bringup::load called for model_dir={:?} but \
                     Qwen36Arch::from_dir returned None — caller dispatched \
                     incorrectly",
                    paths.model_dir
                );
            }
        };
        arch.log_summary();

        let ctx = Arc::new(CudaContextHandle::init(0)?);

        #[cfg(feature = "cuda")]
        let compile_target: Option<rvllm_core::CompileTarget> = {
            let (major, minor) = ctx.compute_capability();
            rvllm_core::CompileTarget::from_compute_capability(major, minor)
        };
        #[cfg(not(feature = "cuda"))]
        let compile_target: Option<rvllm_core::CompileTarget> = None;

        // GB10 (sm_121) has no dedicated HBM — `cuMemAllocManaged` is the
        // right backing. Mirrors the Gemma 4 selection at
        // `gemma4_bring_up.rs::Gemma4Bringup::load`.
        let arena = {
            #[cfg(feature = "gb10")]
            {
                if matches!(compile_target, Some(rvllm_core::CompileTarget::Sm121)) {
                    rvllm_mem::UnifiedArena::new(&ctx, arena_bytes)?.into_inner()
                } else {
                    HbmArena::new(&ctx, arena_bytes)?
                }
            }
            #[cfg(not(feature = "gb10"))]
            {
                HbmArena::new(&ctx, arena_bytes)?
            }
        };
        let arena: HbmArena<'static> = unsafe { std::mem::transmute(arena) };
        let stream = Stream::new(&ctx)?;

        let mut model = rvllm_loader::qwen36_load::load_qwen36_model(
            &paths.model_dir,
            &arena,
            &arch.base.layer_types,
            arch.num_experts,
        )?;
        if std::env::var("RVLLM_QWEN36_LOAD_MTP")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            let mtp = rvllm_loader::qwen36_load::load_qwen36_mtp(
                &paths.model_dir,
                &arena,
                arch.num_experts,
            )?;
            println!("[qwen36-loader] MTP block uploaded");
            model.mtp = Some(mtp);
        }

        // Phase 3a: load + verify the PTX kernel manifest the same way
        // Gemma 4 does. The Qwen forward path will need the same
        // model-agnostic kernel set (embedding_gather, fused_rmsnorm,
        // CUTLASS FP8-GEMM, FA2 paged-attn) plus future Qwen-specific
        // ones (qwen_qk_norm, attn_output_gate, linear-attn recurrent,
        // MoE grouped-FP8-GEMM). Initializing the loader here proves
        // the manifest path and arch-pinning still hold for the Qwen
        // bring-up before Phase 3b starts wiring kernel calls.
        let kernels_dir = crate::bring_up::resolve_kernels_dir(&ctx, &paths.kernels_dir)?;
        let manifest_path = kernels_dir.join("manifest.json");
        let manifest = rvllm_kernels::manifest::KernelManifest::load_and_verify(&manifest_path)?;
        if let Some(t) = compile_target {
            manifest.assert_arch(t.as_sm_str())?;
        }
        manifest.warn_if_revision_drift(rvllm_kernels::manifest::VerifiedManifest::BUILD_REVISION);
        let kernels = Arc::new(KernelLoader::new(manifest));

        // Phase 3b: resolve the outside-path kernel function pointers
        // (embedding_gather + final RMSNorm). The matching modules
        // stay in the bring-up struct so the function handles outlive
        // any per-request scope. lm_head matmul resolution is deferred
        // to Phase 3c — bf16 vs FP8 lm_head dispatch needs an explicit
        // decision (CPU-quantize at load like Gemma's tied path, or
        // route through a bf16 cuBLASLt GEMM).
        let embedding_gather_f16_mod = kernels.load_ptx("embedding_gather_f16")?;
        let fn_embedding_gather_f16 =
            embedding_gather_f16_mod.get_function("embedding_gather_f16_kernel")?;
        let rmsnorm_inplace_f16_mod = kernels.load_ptx("rmsnorm_inplace_f16")?;
        let fn_rmsnorm_inplace_f16 =
            rmsnorm_inplace_f16_mod.get_function("rmsnorm_inplace_f16_kernel")?;
        let fp8_gemv_mod = kernels.load_ptx(rvllm_kernels::FP8_GEMV_PTX_STEM)?;
        let fn_fp8_gemv_wpr_native_f16in = match compile_target {
            Some(t) if rvllm_kernels::Fp8GemvVariant::WprNativeF16In.available_for(t) => Some(
                fp8_gemv_mod
                    .get_function(rvllm_kernels::Fp8GemvVariant::WprNativeF16In.entry_point())?,
            ),
            _ => None,
        };
        let argmax_mod = kernels.load_ptx("argmax")?;
        let fn_argmax = argmax_mod.get_function("argmax_kernel")?;
        let fn_argmax_f16 = argmax_mod.get_function("argmax_f16_kernel")?;
        let fp8_quantize_per_token_f16_mod = kernels.load_ptx("fp8_quantize_per_token_f16")?;
        let fn_fp8_quantize_per_token_f16 =
            fp8_quantize_per_token_f16_mod.get_function("fp8_quantize_per_token_f16_kernel")?;
        let fp8_quantize_per_token_amax_f16_mod =
            kernels.load_ptx("fp8_quantize_per_token_amax_f16")?;
        let fn_fp8_quantize_per_token_amax_f16 = fp8_quantize_per_token_amax_f16_mod
            .get_function("fp8_quantize_per_token_amax_f16_kernel")?;
        let fused_rmsnorm_fp8_quant_mod = kernels.load_ptx("fused_rmsnorm_fp8_quant")?;
        let fn_fused_rmsnorm_fp8_quant =
            fused_rmsnorm_fp8_quant_mod.get_function("fused_rmsnorm_fp8_quant_kernel")?;
        let fused_rope_partial_f16kv_mod = kernels.load_ptx("fused_rope_partial_f16kv")?;
        let fn_fused_rope_partial_f16kv =
            fused_rope_partial_f16kv_mod.get_function("fused_rope_partial_f16kv_kernel")?;
        let fused_rope_qwen_partial_f16kv_mod =
            kernels.load_ptx("fused_rope_qwen_partial_f16kv")?;
        let fn_fused_rope_qwen_partial_f16kv = fused_rope_qwen_partial_f16kv_mod
            .get_function("fused_rope_qwen_partial_f16kv_kernel")?;
        let fused_qnorm_knorm_rope_qwen_partial_f16kv_mod =
            kernels.load_ptx("fused_qnorm_knorm_rope_qwen_partial_f16kv")?;
        let fn_fused_qnorm_knorm_rope_qwen_partial_f16kv =
            fused_qnorm_knorm_rope_qwen_partial_f16kv_mod
                .get_function("fused_qnorm_knorm_rope_qwen_partial_f16kv_kernel")?;
        let fused_qkv_proj_qnorm_knorm_rope_qwen_partial_f16kv_mod =
            kernels.load_ptx(
                "fused_qkv_proj_qnorm_knorm_rope_qwen_partial_f16kv")?;
        let fn_fused_qkv_proj_qnorm_knorm_rope_qwen_partial_f16kv =
            fused_qkv_proj_qnorm_knorm_rope_qwen_partial_f16kv_mod
                .get_function(
                    "fused_qkv_proj_qnorm_knorm_rope_qwen_partial_f16kv_kernel")?;
        // NVFP4 commit 2: load the Qwen NVFP4 RoPE + paged decode
        // kernels only when the same `RVLLM_NVFP4_KV=1` gate that
        // drives the packed-K/V allocator is on, so the resident set
        // matches the cache layout. The KV-allocator branch lives
        // later in `load()`; reading the env here keeps the source
        // of truth single (the env var itself), no plumbing needed.
        let nvfp4_kv_on = std::env::var("RVLLM_NVFP4_KV")
            .ok()
            .as_deref()
            .map(|s| s != "0" && !s.is_empty())
            .unwrap_or(false);
        let (fused_rope_qwen_partial_nvfp4kv_mod, fn_fused_rope_qwen_partial_nvfp4kv) =
            if nvfp4_kv_on {
                let m = kernels.load_ptx("fused_rope_qwen_partial_nvfp4kv")?;
                let f = m.get_function("fused_rope_qwen_partial_nvfp4kv_kernel")?;
                (Some(m), Some(f))
            } else {
                (None, None)
            };
        let (
            fused_qnorm_knorm_rope_qwen_partial_nvfp4kv_mod,
            fn_fused_qnorm_knorm_rope_qwen_partial_nvfp4kv,
        ) = if nvfp4_kv_on {
            let m = kernels.load_ptx(
                "fused_qnorm_knorm_rope_qwen_partial_nvfp4kv")?;
            let f = m.get_function(
                "fused_qnorm_knorm_rope_qwen_partial_nvfp4kv_kernel")?;
            (Some(m), Some(f))
        } else {
            (None, None)
        };
        let (flash_attention_nvfp4kv_mod, fn_flash_attention_2_decode_nvfp4kv) = if nvfp4_kv_on {
            let m = kernels.load_ptx("flash_attention_nvfp4kv")?;
            let f = m.get_function("flash_attention_2_decode_nvfp4kv_kernel")?;
            (Some(m), Some(f))
        } else {
            (None, None)
        };
        let split_q_gate_f16_mod = kernels.load_ptx("split_q_gate_f16")?;
        let fn_split_q_gate_f16 = split_q_gate_f16_mod.get_function("split_q_gate_f16_kernel")?;
        let conv_state_advance_f16_mod = kernels.load_ptx("conv_state_advance_f16")?;
        let fn_conv_state_advance_f16 =
            conv_state_advance_f16_mod.get_function("conv_state_advance_f16_kernel")?;
        let qwen_linear_alpha_beta_f16_mod = kernels.load_ptx("qwen_linear_alpha_beta_f16")?;
        let fn_qwen_linear_alpha_beta_f16 =
            qwen_linear_alpha_beta_f16_mod.get_function("qwen_linear_alpha_beta_f16_kernel")?;
        let qwen_linear_silu_l2_gqa_f16_mod = kernels.load_ptx("qwen_linear_silu_l2_gqa_f16")?;
        let fn_qwen_linear_silu_l2_gqa_f16 =
            qwen_linear_silu_l2_gqa_f16_mod.get_function("qwen_linear_silu_l2_gqa_f16_kernel")?;
        let qwen_linear_rmsnorm_gated_f16_mod =
            kernels.load_ptx("qwen_linear_rmsnorm_gated_f16")?;
        let fn_qwen_linear_rmsnorm_gated_f16 = qwen_linear_rmsnorm_gated_f16_mod
            .get_function("qwen_linear_rmsnorm_gated_f16_kernel")?;
        let silu_mul_f16_mod = kernels.load_ptx("silu_mul_f16")?;
        let fn_silu_mul_f16 = silu_mul_f16_mod.get_function("silu_mul_f16_kernel")?;
        let router_gemv_f16_to_f32_mod = kernels.load_ptx("router_gemv_f16_to_f32")?;
        let router_gemv_with_topk_f16_to_f32_mod =
            kernels.load_ptx("router_gemv_with_topk_f16_to_f32")?;
        let fn_router_gemv_with_topk_f16_to_f32 =
            router_gemv_with_topk_f16_to_f32_mod
                .get_function("router_gemv_with_topk_f16_to_f32_kernel")?;
        let router_gemv_with_topk_batched_f16_to_f32_mod =
            kernels.load_ptx("router_gemv_with_topk_batched_f16_to_f32")?;
        let fn_router_gemv_with_topk_batched_f16_to_f32 =
            router_gemv_with_topk_batched_f16_to_f32_mod
                .get_function("router_gemv_with_topk_batched_f16_to_f32_kernel")?;
        let fn_router_gemv_f16_to_f32 =
            router_gemv_f16_to_f32_mod.get_function("router_gemv_f16_to_f32_kernel")?;
        let scaled_add_f16_to_f32_mod = kernels.load_ptx("scaled_add_f16_to_f32")?;
        let fn_scaled_add_f16_to_f32 =
            scaled_add_f16_to_f32_mod.get_function("scaled_add_f16_to_f32_kernel")?;
        let f16_plus_f32_inplace_f16_mod = kernels.load_ptx("f16_plus_f32_inplace_f16")?;
        let fn_f16_plus_f32_inplace_f16 =
            f16_plus_f32_inplace_f16_mod.get_function("f16_plus_f32_inplace_f16_kernel")?;
        let shared_gate_dot_sigmoid_f16_mod = kernels.load_ptx("shared_gate_dot_sigmoid_f16")?;
        let fn_shared_gate_dot_sigmoid_f16 =
            shared_gate_dot_sigmoid_f16_mod.get_function("shared_gate_dot_sigmoid_f16_kernel")?;
        let scaled_add_f16_to_f32_devw_mod = kernels.load_ptx("scaled_add_f16_to_f32_devw")?;
        let fn_scaled_add_f16_to_f32_devw =
            scaled_add_f16_to_f32_devw_mod.get_function("scaled_add_f16_to_f32_devw_kernel")?;
        let fp8_gemv_dual_mod = kernels.load_ptx("fp8_gemv_blockwise_wpr_native_f16in_dual")?;
        let fn_fp8_gemv_dual =
            fp8_gemv_dual_mod.get_function("fp8_gemv_blockwise_wpr_native_f16in_dual_kernel")?;
        let fp8_gemv_dual_silu_mod =
            kernels.load_ptx("fp8_gemv_blockwise_wpr_native_f16in_dual_silu")?;
        let fn_fp8_gemv_dual_silu = fp8_gemv_dual_silu_mod
            .get_function("fp8_gemv_blockwise_wpr_native_f16in_dual_silu_kernel")?;
        let topk_softmax_f32_mod = kernels.load_ptx("topk_softmax_f32")?;
        let fn_topk_softmax_f32 = topk_softmax_f32_mod.get_function("topk_softmax_f32_kernel")?;
        let fp8_gemv_dual_silu_indirect_mod =
            kernels.load_ptx("fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect")?;
        let fp8_gemv_dual_silu_indirect_kround_batched_mod = kernels
            .load_ptx("fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_kround_batched")?;
        let fn_fp8_gemv_dual_silu_indirect_kround_batched =
            fp8_gemv_dual_silu_indirect_kround_batched_mod.get_function(
                "fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_kround_batched_kernel",
            )?;
        let fn_fp8_gemv_dual_silu_indirect = fp8_gemv_dual_silu_indirect_mod
            .get_function("fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_kernel")?;
        let fp8_gemv_indirect_mod =
            kernels.load_ptx("fp8_gemv_blockwise_wpr_native_f16in_indirect")?;
        let fn_fp8_gemv_indirect = fp8_gemv_indirect_mod
            .get_function("fp8_gemv_blockwise_wpr_native_f16in_indirect_kernel")?;
        let fp8_gemv_indirect_scaled_add_mod = kernels
            .load_ptx("fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add")?;
        let fn_fp8_gemv_indirect_scaled_add = fp8_gemv_indirect_scaled_add_mod
            .get_function("fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add_kernel")?;
        let fp8_gemv_indirect_scaled_add_kround_batched_mod = kernels.load_ptx(
            "fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add_kround_batched")?;
        let fn_fp8_gemv_indirect_scaled_add_kround_batched =
            fp8_gemv_indirect_scaled_add_kround_batched_mod.get_function(
                "fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add_kround_batched_kernel",
            )?;
        let fp8_gemv_f16in_scaled_add_devw_mod =
            kernels.load_ptx("fp8_gemv_f16in_scaled_add_devw")?;
        let fn_fp8_gemv_f16in_scaled_add_devw = fp8_gemv_f16in_scaled_add_devw_mod
            .get_function("fp8_gemv_blockwise_wpr_native_f16in_scaled_add_devw_kernel")?;
        let flash_attention_mod = kernels.load_ptx("flash_attention")?;
        let fn_flash_attention_2_f16kv =
            flash_attention_mod.get_function("flash_attention_2_f16kv_kernel")?;
        let fn_flash_attention_2_decode_f16io =
            flash_attention_mod.get_function("flash_attention_2_decode_f16io_kernel")?;
        let sigmoid_mul_f16_mod = kernels.load_ptx("sigmoid_mul_f16")?;
        let fn_sigmoid_mul_f16 = sigmoid_mul_f16_mod.get_function("sigmoid_mul_f16_kernel")?;
        let causal_conv1d_f16_mod = kernels.load_ptx("causal_conv1d_f16")?;
        let fn_causal_conv1d_f16 =
            causal_conv1d_f16_mod.get_function("causal_conv1d_f16_kernel")?;
        let gated_delta_state_update_f16_mod = kernels.load_ptx("gated_delta_state_update_f16")?;
        let fn_gated_delta_state_update_f16 =
            gated_delta_state_update_f16_mod.get_function("gated_delta_state_update_f16_kernel")?;
        let gated_delta_rule_decode_f16_mod = kernels.load_ptx("gated_delta_rule_decode_f16")?;
        let fn_gated_delta_rule_decode_f16 =
            gated_delta_rule_decode_f16_mod.get_function("gated_delta_rule_decode_f16_kernel")?;
        // Phase 4b/Linear: prefill-batched delta-rule + conv-state-advance.
        let gated_delta_rule_prefill_f16_mod = kernels.load_ptx("gated_delta_rule_prefill_f16")?;
        let fn_gated_delta_rule_prefill_f16 =
            gated_delta_rule_prefill_f16_mod.get_function("gated_delta_rule_prefill_f16_kernel")?;
        let conv_state_advance_batched_f16_mod =
            kernels.load_ptx("conv_state_advance_batched_f16")?;
        let fn_conv_state_advance_batched_f16 = conv_state_advance_batched_f16_mod
            .get_function("conv_state_advance_batched_f16_kernel")?;
        let qwen36_step_link_i32_mod = kernels.load_ptx("qwen36_step_link_i32")?;
        let fn_qwen36_step_link_i32 =
            qwen36_step_link_i32_mod.get_function("qwen36_step_link_i32_kernel")?;
        let qwen36_argmax_with_link_f16_mod =
            kernels.load_ptx("qwen36_argmax_with_link_f16")?;
        let fn_qwen36_argmax_with_link_f16 =
            qwen36_argmax_with_link_f16_mod
                .get_function("qwen36_argmax_with_link_f16_kernel")?;
        let qwen_fill_pos_slots_i32_mod = kernels.load_ptx("qwen_fill_pos_slots_i32")?;
        let fn_qwen_fill_pos_slots_i32 =
            qwen_fill_pos_slots_i32_mod.get_function("qwen_fill_pos_slots_i32_kernel")?;
        let router_gemv_batched_f16_to_f32_mod =
            kernels.load_ptx("router_gemv_batched_f16_to_f32")?;
        let fn_router_gemv_batched_f16_to_f32 = router_gemv_batched_f16_to_f32_mod
            .get_function("router_gemv_batched_f16_to_f32_kernel")?;
        let topk_softmax_batched_f32_mod = kernels.load_ptx("topk_softmax_batched_f32")?;
        let fn_topk_softmax_batched_f32 =
            topk_softmax_batched_f32_mod.get_function("topk_softmax_batched_f32_kernel")?;
        let fp8_gemv_blockwise_wpr_native_f16in_indirect_batched_topk_mod =
            kernels.load_ptx("fp8_gemv_blockwise_wpr_native_f16in_indirect_batched_topk")?;
        let fn_fp8_gemv_blockwise_wpr_native_f16in_indirect_batched_topk =
            fp8_gemv_blockwise_wpr_native_f16in_indirect_batched_topk_mod
                .get_function("fp8_gemv_blockwise_wpr_native_f16in_indirect_batched_topk_kernel")?;
        let fp8_gemv_indirect_scaled_add_batched_topk_mod = kernels.load_ptx(
            "fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add_batched_topk")?;
        let fn_fp8_gemv_indirect_scaled_add_batched_topk = fp8_gemv_indirect_scaled_add_batched_topk_mod
            .get_function(
                "fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add_batched_topk_kernel",
            )?;
        let fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_batched_topk_mod = kernels
            .load_ptx("fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_batched_topk")?;
        let fn_fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_batched_topk =
            fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_batched_topk_mod.get_function(
                "fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_batched_topk_kernel",
            )?;
        let scaled_add_f16_to_f32_devw_batched_topk_mod =
            kernels.load_ptx("scaled_add_f16_to_f32_devw_batched_topk")?;
        let fn_scaled_add_f16_to_f32_devw_batched_topk =
            scaled_add_f16_to_f32_devw_batched_topk_mod
                .get_function("scaled_add_f16_to_f32_devw_batched_topk_kernel")?;
        let shared_gate_dot_sigmoid_f16_batched_mod =
            kernels.load_ptx("shared_gate_dot_sigmoid_f16_batched")?;
        let fn_shared_gate_dot_sigmoid_f16_batched = shared_gate_dot_sigmoid_f16_batched_mod
            .get_function("shared_gate_dot_sigmoid_f16_batched_kernel")?;
        let scaled_add_f16_to_f32_devw_batched_mod =
            kernels.load_ptx("scaled_add_f16_to_f32_devw_batched")?;
        let fn_scaled_add_f16_to_f32_devw_batched = scaled_add_f16_to_f32_devw_batched_mod
            .get_function("scaled_add_f16_to_f32_devw_batched_kernel")?;
        // Vision-tower kernels (Phase 1 .cu added on rusty_sm121_vision).
        let layernorm_inplace_f16_mod = kernels.load_ptx("layernorm_inplace_f16")?;
        let fn_layernorm_inplace_f16 =
            layernorm_inplace_f16_mod.get_function("layernorm_inplace_f16_kernel")?;
        let gelu_tanh_f16_mod = kernels.load_ptx("gelu_tanh_f16")?;
        let fn_gelu_tanh_f16 = gelu_tanh_f16_mod.get_function("gelu_tanh_f16_kernel")?;
        let softmax_row_f16_mod = kernels.load_ptx("softmax_row_f16")?;
        let fn_softmax_row_f16 = softmax_row_f16_mod.get_function("softmax_row_f16_kernel")?;
        let vit_rotary_2d_f16_mod = kernels.load_ptx("vit_rotary_2d_f16")?;
        let fn_vit_rotary_2d_f16 =
            vit_rotary_2d_f16_mod.get_function("vit_rotary_2d_f16_kernel")?;
        let vit_avgpool_f16_mod = kernels.load_ptx("vit_avgpool_f16")?;
        let fn_vit_avgpool_f16 = vit_avgpool_f16_mod.get_function("vit_avgpool_f16_kernel")?;
        let vit_pos_embed_interp_f16_mod = kernels.load_ptx("vit_pos_embed_interp_f16")?;
        let fn_vit_pos_embed_interp_f16 =
            vit_pos_embed_interp_f16_mod.get_function("vit_pos_embed_interp_f16_kernel")?;
        let scale_inplace_f16_mod = kernels.load_ptx("scale_inplace_f16")?;
        let fn_scale_inplace_f16 =
            scale_inplace_f16_mod.get_function("scale_inplace_f16_kernel")?;
        let transpose_2d_f16_mod = kernels.load_ptx("transpose_2d_f16")?;
        let fn_transpose_2d_f16 = transpose_2d_f16_mod.get_function("transpose_2d_f16_kernel")?;
        let add_bias_f16_mod = kernels.load_ptx("add_bias_f16")?;
        let fn_add_bias_f16 = add_bias_f16_mod.get_function("add_bias_f16_kernel")?;
        let cast_fp_mod = kernels.load_ptx("cast_fp")?;
        let fn_cast_f32_to_f16 = cast_fp_mod.get_function("cast_f32_to_f16_kernel")?;
        let fn_cast_f16_to_f32 = cast_fp_mod.get_function("cast_f16_to_f32_kernel")?;
        let vector_add_f16_mod = kernels.load_ptx("vector_add_f16")?;
        let fn_vector_add_f16 = vector_add_f16_mod.get_function("vector_add_f16_kernel")?;
        let extract_head_f16_mod = kernels.load_ptx("extract_head_f16")?;
        let fn_extract_head_f16 = extract_head_f16_mod.get_function("extract_head_f16_kernel")?;
        let fn_scatter_head_f16 = extract_head_f16_mod.get_function("scatter_head_f16_kernel")?;
        let softmax_row_f32_to_f16_mod = kernels.load_ptx("softmax_row_f32_to_f16")?;
        let fn_softmax_row_f32_to_f16 =
            softmax_row_f32_to_f16_mod.get_function("softmax_row_f32_to_f16_kernel")?;
        let transpose_heads_v_f16_mod = kernels.load_ptx("transpose_heads_v_f16")?;
        let fn_transpose_heads_v_f16 =
            transpose_heads_v_f16_mod.get_function("transpose_heads_v_f16_kernel")?;
        let scatter_heads_f16_mod = kernels.load_ptx("scatter_heads_f16")?;
        let fn_scatter_heads_f16 =
            scatter_heads_f16_mod.get_function("scatter_heads_f16_kernel")?;
        let scale_inplace_f32_mod = kernels.load_ptx("scale_inplace_f32")?;
        let fn_scale_inplace_f32 =
            scale_inplace_f32_mod.get_function("scale_inplace_f32_kernel")?;
        let outside_kernels = Qwen36OutsideKernels {
            embedding_gather_f16_mod,
            fn_embedding_gather_f16,
            rmsnorm_inplace_f16_mod,
            fn_rmsnorm_inplace_f16,
            fp8_gemv_mod,
            fn_fp8_gemv_wpr_native_f16in,
            argmax_mod,
            fn_argmax,
            fn_argmax_f16,
            fp8_quantize_per_token_f16_mod,
            fn_fp8_quantize_per_token_f16,
            fp8_quantize_per_token_amax_f16_mod,
            fn_fp8_quantize_per_token_amax_f16,
            fused_rmsnorm_fp8_quant_mod,
            fn_fused_rmsnorm_fp8_quant,
            fused_rope_partial_f16kv_mod,
            fn_fused_rope_partial_f16kv,
            fused_rope_qwen_partial_f16kv_mod,
            fn_fused_rope_qwen_partial_f16kv,
            fused_qnorm_knorm_rope_qwen_partial_f16kv_mod,
            fn_fused_qnorm_knorm_rope_qwen_partial_f16kv,
            fused_qkv_proj_qnorm_knorm_rope_qwen_partial_f16kv_mod,
            fn_fused_qkv_proj_qnorm_knorm_rope_qwen_partial_f16kv,
            fused_rope_qwen_partial_nvfp4kv_mod,
            fused_qnorm_knorm_rope_qwen_partial_nvfp4kv_mod,
            fn_fused_qnorm_knorm_rope_qwen_partial_nvfp4kv,
            fn_fused_rope_qwen_partial_nvfp4kv,
            flash_attention_nvfp4kv_mod,
            fn_flash_attention_2_decode_nvfp4kv,
            split_q_gate_f16_mod,
            fn_split_q_gate_f16,
            conv_state_advance_f16_mod,
            fn_conv_state_advance_f16,
            qwen_linear_alpha_beta_f16_mod,
            fn_qwen_linear_alpha_beta_f16,
            qwen_linear_silu_l2_gqa_f16_mod,
            fn_qwen_linear_silu_l2_gqa_f16,
            qwen_linear_rmsnorm_gated_f16_mod,
            fn_qwen_linear_rmsnorm_gated_f16,
            silu_mul_f16_mod,
            fn_silu_mul_f16,
            router_gemv_f16_to_f32_mod,
            fn_router_gemv_f16_to_f32,
            router_gemv_with_topk_f16_to_f32_mod,
            fn_router_gemv_with_topk_f16_to_f32,
            router_gemv_with_topk_batched_f16_to_f32_mod,
            fn_router_gemv_with_topk_batched_f16_to_f32,
            router_gemv_with_topk_batched_counter_dev: 0u64,
            router_topk_counter_dev: 0u64,  // populated post-construct
                                              // via dedicated arena
                                              // alloc before scratch_ck
            scaled_add_f16_to_f32_mod,
            fn_scaled_add_f16_to_f32,
            f16_plus_f32_inplace_f16_mod,
            fn_f16_plus_f32_inplace_f16,
            shared_gate_dot_sigmoid_f16_mod,
            fn_shared_gate_dot_sigmoid_f16,
            scaled_add_f16_to_f32_devw_mod,
            fn_scaled_add_f16_to_f32_devw,
            fp8_gemv_dual_mod,
            fn_fp8_gemv_dual,
            fp8_gemv_dual_silu_mod,
            fn_fp8_gemv_dual_silu,
            topk_softmax_f32_mod,
            fn_topk_softmax_f32,
            fp8_gemv_dual_silu_indirect_mod,
            fn_fp8_gemv_dual_silu_indirect,
            fp8_gemv_dual_silu_indirect_kround_batched_mod,
            fn_fp8_gemv_dual_silu_indirect_kround_batched,
            fp8_gemv_indirect_mod,
            fn_fp8_gemv_indirect,
            fp8_gemv_indirect_scaled_add_mod,
            fn_fp8_gemv_indirect_scaled_add,
            fp8_gemv_indirect_scaled_add_kround_batched_mod,
            fn_fp8_gemv_indirect_scaled_add_kround_batched,
            fp8_gemv_f16in_scaled_add_devw_mod,
            fn_fp8_gemv_f16in_scaled_add_devw,
            flash_attention_mod,
            fn_flash_attention_2_decode_f16io,
            fn_flash_attention_2_f16kv,
            fn_cast_f16_to_f32,
            sigmoid_mul_f16_mod,
            fn_sigmoid_mul_f16,
            causal_conv1d_f16_mod,
            fn_causal_conv1d_f16,
            gated_delta_state_update_f16_mod,
            fn_gated_delta_state_update_f16,
            gated_delta_rule_decode_f16_mod,
            fn_gated_delta_rule_decode_f16,
            gated_delta_rule_prefill_f16_mod,
            fn_gated_delta_rule_prefill_f16,
            conv_state_advance_batched_f16_mod,
            fn_conv_state_advance_batched_f16,
            qwen_fill_pos_slots_i32_mod,
            fn_qwen_fill_pos_slots_i32,
            qwen36_step_link_i32_mod,
            fn_qwen36_step_link_i32,
            qwen36_argmax_with_link_f16_mod,
            fn_qwen36_argmax_with_link_f16,
            router_gemv_batched_f16_to_f32_mod,
            fn_router_gemv_batched_f16_to_f32,
            topk_softmax_batched_f32_mod,
            fn_topk_softmax_batched_f32,
            fp8_gemv_blockwise_wpr_native_f16in_indirect_batched_topk_mod,
            fn_fp8_gemv_blockwise_wpr_native_f16in_indirect_batched_topk,
            fp8_gemv_indirect_scaled_add_batched_topk_mod,
            fn_fp8_gemv_indirect_scaled_add_batched_topk,
            fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_batched_topk_mod,
            fn_fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_batched_topk,
            scaled_add_f16_to_f32_devw_batched_topk_mod,
            fn_scaled_add_f16_to_f32_devw_batched_topk,
            shared_gate_dot_sigmoid_f16_batched_mod,
            fn_shared_gate_dot_sigmoid_f16_batched,
            scaled_add_f16_to_f32_devw_batched_mod,
            fn_scaled_add_f16_to_f32_devw_batched,
            layernorm_inplace_f16_mod,
            fn_layernorm_inplace_f16,
            gelu_tanh_f16_mod,
            fn_gelu_tanh_f16,
            softmax_row_f16_mod,
            fn_softmax_row_f16,
            vit_rotary_2d_f16_mod,
            fn_vit_rotary_2d_f16,
            vit_avgpool_f16_mod,
            fn_vit_avgpool_f16,
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
            vector_add_f16_mod,
            fn_vector_add_f16,
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
        eprintln!(
            "[qwen36] outside kernels resolved: embedding_gather_f16, \
             rmsnorm_inplace_f16, fp8_gemv (wpr_native_f16in: {}), \
             argmax, fused_rmsnorm_fp8_quant, fused_rope_partial_f16kv, \
             flash_attention_2_decode_f16io, sigmoid_mul_f16, \
             causal_conv1d_f16, gated_delta_state_update_f16.",
            outside_kernels.fn_fp8_gemv_wpr_native_f16in.is_some(),
        );
        let attn_backend_full = rvllm_attention::AttentionBackend::Fa2Ptx(
            rvllm_attention::Fa2PtxKernels::load(&*kernels, arch.base.head_dim as u32)?,
        );

        // Phase 3g: cuBLASLt for the lm_head FP8 GEMM. Same 32 MiB
        // workspace size Gemma 4 uses (gemma4_bring_up.rs:1415).
        let cublaslt_ws_bytes: usize = 32 * 1024 * 1024;
        let cublaslt_ws_region = arena.region("qwen36_cublaslt_ws", cublaslt_ws_bytes, 256)?;
        let cublaslt = CublasLt::new(cublaslt_ws_region.device_ptr(), cublaslt_ws_bytes)?;
        eprintln!(
            "[qwen36] cuBLASLt initialized with {} MiB workspace.",
            cublaslt_ws_bytes / (1024 * 1024)
        );

        // CUTLASS SM120 backend for the m≥128 batched-prefill fast
        // path on sm_121. Same .so Gemma loads. We don't need the
        // policy variant table (sm_121 ships only the blockscale
        // entry point), so pass an empty variants slice.
        let cutlass = CutlassBackend::load_for(compile_target, paths.cutlass_so.clone(), &[])?;
        eprintln!(
            "[qwen36] cutlass backend = {}",
            match &cutlass {
                CutlassBackend::SoSm120(_) => "SoSm120 (sm_121 fast path)",
                CutlassBackend::So(_) => "So",
                CutlassBackend::Absent => "Absent (m≥2 path → looped GEMV fallback)",
                _ => "Other",
            }
        );

        // Phase 4f: precompute single-axis RoPE cos/sin tables.
        // Qwen 3.6's `rope_theta` (10_000_000) + `head_dim` (256)
        // give a base table of shape `[max_pos, head_dim/2]` for
        // each of cos and sin. Cap `max_pos` to RVLLM_MAX_TOKENS_CAP
        // (typically 4096) — the model's `max_position_embeddings`
        // is 262144 which would mean 128 MiB of f16 cos/sin tables;
        // capping keeps the bring-up cost reasonable until Phase 4g
        // wires the real RoPE launch and we know the actual decode
        // window. MRoPE section math (sections [11, 11, 10]) is
        // applied on top of these tables at launch time, not baked
        // into the tables themselves (text-only mode, sections
        // collapse to standard RoPE per-position).
        let rope_max_pos = std::env::var("RVLLM_MAX_TOKENS_CAP")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(4096)
            .min(262_144);
        let rope_theta = arch.base.rope_theta;
        let rope_head_dim = arch.base.head_dim as u32;
        // Partial RoPE: rotary_dim = head_dim * 0.25 = 64 for Qwen 3.6.
        // Tables must be stride [rotary_dim/2] per position (matches the
        // fused_rope kernel's `pos * half_rotary + tid` indexing).
        // Frequencies still use head_dim as divisor — proportional-RoPE
        // convention shared with Gemma 4 (gemma4_load.rs:599-604).
        let rotary_dim = (rope_head_dim as f32 * 0.25) as u32;
        let half = (rotary_dim / 2) as usize;
        let inv_theta: Vec<f32> = (0..half)
            .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / rope_head_dim as f32))
            .collect();
        let rope_table_elems = (rope_max_pos as usize) * half;
        let rope_table_bytes = rope_table_elems * 2; // f16
        let mut cos_bytes = Vec::with_capacity(rope_table_bytes);
        let mut sin_bytes = Vec::with_capacity(rope_table_bytes);
        for pos in 0..rope_max_pos as usize {
            for &freq in &inv_theta {
                let angle = pos as f32 * freq;
                let c = angle.cos();
                let s = angle.sin();
                // Inline f32 → f16 (avoid pulling `half` as a runtime dep).
                cos_bytes.extend_from_slice(&f32_to_f16_bits(c).to_le_bytes());
                sin_bytes.extend_from_slice(&f32_to_f16_bits(s).to_le_bytes());
            }
        }
        let rope_cos_region = arena.region("qwen36_rope_cos", rope_table_bytes, 16)?;
        let rope_sin_region = arena.region("qwen36_rope_sin", rope_table_bytes, 16)?;
        unsafe {
            rope_cos_region.copy_from_host(&cos_bytes)?;
            rope_sin_region.copy_from_host(&sin_bytes)?;
        }
        let rope_cos = rope_cos_region.device_ptr();
        let rope_sin = rope_sin_region.device_ptr();
        eprintln!(
            "[qwen36] RoPE tables uploaded: max_pos={rope_max_pos} head_dim={rope_head_dim} \
             theta={rope_theta:.1e} ({:.1} KiB cos + {:.1} KiB sin)",
            rope_table_bytes as f64 / 1024.0,
            rope_table_bytes as f64 / 1024.0,
        );

        // Phase 4t: per-sequence linear-attn state cache. Persists
        // across decode steps in a single arena region above the
        // scratch checkpoint. Single-sequence layout for now.
        let n_linear_layers = arch
            .base
            .layer_types
            .iter()
            .filter(|t| matches!(t, rvllm_loader::LayerAttnType::Linear))
            .count();
        let num_ssm_heads: usize = 32;
        let d_state: usize = 128;
        let linear_state_layer_bytes = num_ssm_heads * d_state * d_state * 2; // f16
        let linear_state_bytes = n_linear_layers * linear_state_layer_bytes;
        let linear_state_region = arena.region("qwen36_linear_state", linear_state_bytes, 16)?;
        // Zero the state at bring-up (analogous to a session-start
        // reset). cuMemsetD8 is the fastest path; falls back to a
        // host-side zero buffer if needed.
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8_v2(linear_state_region.device_ptr(), 0, linear_state_bytes);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_linear_state cuMemsetD8",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let linear_state_ptr = linear_state_region.device_ptr();
        eprintln!(
            "[qwen36] linear-attn state cache allocated: {} layers × \
             {:.1} MiB = {:.1} MiB total (zero-initialised, persists \
             across decode steps).",
            n_linear_layers,
            linear_state_layer_bytes as f64 / (1024.0 * 1024.0),
            linear_state_bytes as f64 / (1024.0 * 1024.0),
        );

        // Phase 4u: paged f16 KV cache for full-attention layers.
        // Sized for max_tokens_cap context (default 4096) at the
        // current num_kv_heads / head_dim layout. block_size 16 is
        // the standard rvllm paged-attn tile.
        let n_full_layers = arch
            .base
            .layer_types
            .iter()
            .filter(|t| matches!(t, rvllm_loader::LayerAttnType::Full))
            .count();
        let kv_max_tokens = std::env::var("RVLLM_MAX_TOKENS_CAP")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(4096);
        let kv_cache_block_size: u32 = 16;
        let kv_cache_num_blocks = kv_max_tokens.div_ceil(kv_cache_block_size);
        let nkvh = arch.base.num_key_value_heads;
        let hd = arch.base.head_dim;
        // NVFP4 commit 1: pick dtype before sizing. F16 stays 2
        // bytes/elem; NVFP4 is packed 4-bit (1 byte per 2 elems)
        // and carries a separate E4M3 microscale buffer with one
        // scale per 16-element block. Decoders dispatch on
        // `Qwen36Bringup::kv_dtype`. Env gate is shared with the
        // Qwen 3.5 and Gemma 4 NVFP4 paths.
        let nvfp4_kv = std::env::var("RVLLM_NVFP4_KV")
            .ok()
            .as_deref()
            .map(|s| s != "0" && !s.is_empty())
            .unwrap_or(false);
        let kv_dtype = if nvfp4_kv {
            Qwen36KvDtype::Nvfp4
        } else {
            Qwen36KvDtype::F16
        };
        // 2 = K + V. Bytes-per-elem differs per dtype.
        let kv_slots = (kv_cache_num_blocks as usize) * (kv_cache_block_size as usize) * nkvh * hd;
        let kv_cache_layer_bytes = match kv_dtype {
            Qwen36KvDtype::F16 => 2usize * kv_slots * 2, // 2-byte f16
            Qwen36KvDtype::Nvfp4 => 2usize * (kv_slots / 2), // 4-bit packed
        };
        let mtp_kv_layers = usize::from(model.mtp.is_some());
        let kv_cache_layers = n_full_layers + mtp_kv_layers;
        let kv_cache_bytes = kv_cache_layers * kv_cache_layer_bytes;
        let kv_cache_region = arena.region("qwen36_kv_cache", kv_cache_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8_v2(kv_cache_region.device_ptr(), 0, kv_cache_bytes);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_kv_cache cuMemsetD8",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let kv_cache_ptr = kv_cache_region.device_ptr();
        // NVFP4 commit 1: companion microscale buffer. Only present
        // when `kv_dtype == Nvfp4`. One E4M3 scale (1 byte) per
        // 16-element packed block.
        let (kv_cache_scale_ptr, kv_cache_scale_layer_bytes, kv_cache_scale_bytes) = match kv_dtype
        {
            Qwen36KvDtype::F16 => (0u64, 0usize, 0usize),
            Qwen36KvDtype::Nvfp4 => {
                let layer_bytes = 2usize * (kv_slots / 16); // K+V × slots/16
                let total = kv_cache_layers * layer_bytes;
                let region = arena.region("qwen36_kv_cache_scale", total, 16)?;
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let rc = cuMemsetD8_v2(region.device_ptr(), 0, total);
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36_kv_cache_scale cuMemsetD8",
                            rvllm_core::CudaErrorKind::MemcpyFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                (region.device_ptr(), layer_bytes, total)
            }
        };
        eprintln!(
            "[qwen36] paged KV cache allocated: {n_full_layers} full-attn \
             layers + {mtp_kv_layers} MTP shadow layer(s) = {kv_cache_layers} total \
             layers × {:.1} MiB ({} blocks × {} tokens × 2 (K+V) × \
             {nkvh} kv_heads × {hd} hd × {dt_label}) = {:.1} MiB total \
             (zero-initialised). NVFP4 scale buffer: {:.1} MiB total.",
            kv_cache_layer_bytes as f64 / (1024.0 * 1024.0),
            kv_cache_num_blocks,
            kv_cache_block_size,
            kv_cache_bytes as f64 / (1024.0 * 1024.0),
            kv_cache_scale_bytes as f64 / (1024.0 * 1024.0),
            dt_label = match kv_dtype {
                Qwen36KvDtype::F16 => "f16",
                Qwen36KvDtype::Nvfp4 => "nvfp4-packed",
            },
        );

        // Phase 5f: conv1d state cache. Each linear-attn layer keeps the
        // previous (kernel-1=3) conv-input timesteps so per-token decode
        // sees the real prior context. Layout per layer: [3, 8192] f16.
        let conv_kernel_minus_1: usize = 3;
        let conv_dim: usize = 8192;
        let conv_state_layer_bytes = conv_kernel_minus_1 * conv_dim * 2;
        let conv_state_bytes = n_linear_layers * conv_state_layer_bytes;
        let conv_state_region = arena.region("qwen36_conv_state", conv_state_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8_v2(conv_state_region.device_ptr(), 0, conv_state_bytes);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_conv_state cuMemsetD8",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let conv_state_ptr = conv_state_region.device_ptr();
        eprintln!(
            "[qwen36] conv1d state cache allocated: {n_linear_layers} \
             linear-attn layers × {:.1} KiB ({conv_kernel_minus_1} \
             timesteps × {conv_dim} channels × f16) = {:.1} MiB total.",
            conv_state_layer_bytes as f64 / 1024.0,
            conv_state_bytes as f64 / (1024.0 * 1024.0),
        );

        let n_full = model
            .layers
            .iter()
            .filter(|l| {
                matches!(
                    l.attn,
                    rvllm_loader::qwen36_weights::Qwen36LayerAttn::Full(_)
                )
            })
            .count();
        let n_linear = model
            .layers
            .iter()
            .filter(|l| {
                matches!(
                    l.attn,
                    rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(_)
                )
            })
            .count();
        eprintln!(
            "[qwen36] Phase 5f bring-up complete: outside (incl. \
             FP8-quantized lm_head) + {n_full} full-attention + \
             {n_linear} linear-attention + per-layer MoE blocks \
             ({} experts/layer) + KernelLoader + outside kernel \
             pointers + cuBLASLt + outside-only forward smoke \
             (embed → rmsnorm+fp8quant → fp8_gemm → cpu_argmax) \
             validated end-to-end. arena.used()={used:.2} GiB / \
             {total:.2} GiB. Per-layer forward (full-attn + \
             linear-attn + MoE) still TODO. \
             See ~/.claude/plans/abundant-meandering-sifakis.md.",
            arch.num_experts,
            used = arena.used() as f64 / (1024.0 * 1024.0 * 1024.0),
            total = arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        let mut bringup = Self {
            paths,
            arena_bytes,
            arch,
            ctx,
            arena,
            stream,
            model,
            kernels,
            outside_kernels,
            attn_backend_full,
            cublaslt,
            cutlass,
            rope_cos,
            rope_sin,
            rope_max_pos,
            linear_state_ptr,
            linear_state_bytes,
            linear_state_layer_bytes,
            linear_attn_host_cache: Vec::new(), // populated below
            router_host_cache: Vec::new(),      // populated below
            shared_gate_host_cache: Vec::new(), // populated below

            kv_cache_ptr,
            kv_cache_bytes,
            kv_cache_layer_bytes,
            kv_cache_num_blocks,
            kv_cache_block_size,
            kv_dtype,
            kv_cache_scale_ptr,
            kv_cache_scale_bytes,
            kv_cache_scale_layer_bytes,
            bt_persistent_ptr: 0, // populated below
            conv_state_ptr,
            conv_state_bytes,
            conv_state_layer_bytes,
            decode_capture: std::sync::Mutex::new(None),
            decode_capture_multi_step: std::sync::Mutex::new(None),
        };
        // Phase 4b-prep iter25: upload the constant identity block
        // table once. The paged-attention layer used to rebuild it
        // every full-attn layer × every token.
        {
            let max_blocks = bringup.kv_cache_num_blocks as usize;
            let bt_bytes = max_blocks * 4;
            let bt_region = bringup.arena.region("qwen36_bt_persistent", bt_bytes, 16)?;
            let mut bt_host = Vec::with_capacity(bt_bytes);
            for b in 0..max_blocks {
                bt_host.extend_from_slice(&(b as i32).to_le_bytes());
            }
            unsafe {
                bt_region.copy_from_host(&bt_host)?;
            }
            bringup.bt_persistent_ptr = bt_region.device_ptr();
        }
        // Phase 8 router+topk fusion: allocate the persistent u32[1]
        // counter the fused kernel uses for atomic last-block
        // detection. Zeroed via a 4-byte HtoD; the kernel self-
        // resets to 0 in the last block, so this single zero
        // suffices for the entire worker lifetime.
        {
            let counter_region = bringup.arena.region(
                "qwen36_router_topk_counter", 4, 4)?;
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let zero: u32 = 0;
                let _ = cuMemcpyHtoD_v2(
                    counter_region.device_ptr(),
                    &zero as *const u32 as *const _,
                    4,
                );
            }
            bringup.outside_kernels.router_topk_counter_dev =
                counter_region.device_ptr();
        }
        // Phase 8 batched router+topk fusion: per-token counter
        // slots sized by `kv_cache_num_blocks` (the per-worker
        // upper bound on prefill token count). Zeroed once via
        // a single cuMemsetD8Async; the kernel resets each
        // token's slot after use, so the region stays at all-
        // zeros across requests.
        {
            let max_tokens = bringup.kv_cache_num_blocks as usize;
            let counter_b_bytes = max_tokens * 4;
            let counter_b_region = bringup.arena.region(
                "qwen36_router_topk_batched_counter",
                counter_b_bytes, 4)?;
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let r = cuMemsetD8Async(
                    counter_b_region.device_ptr(),
                    0,
                    counter_b_bytes,
                    bringup.stream.raw() as CUstream,
                );
                if r != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 router_gemv_with_topk_batched counter zero",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
                // Fence so subsequent worker init kernels see the
                // zeros. (The stream is shared across worker init
                // anyway; explicit fence to be safe.)
                bringup.stream.fence()?;
            }
            bringup.outside_kernels.router_gemv_with_topk_batched_counter_dev =
                counter_b_region.device_ptr();
        }
        // Build per-linear-attn-layer host f32 weight caches BEFORE
        // any probe / smoke runs (some of those reach into
        // `apply_layer_linear_attn` and would panic on an empty
        // cache). Pre-Phase-5 code DtoH-copied these weights every
        // token (~256 KB of constants per token); now we dequantise
        // once to f32 and the per-token alpha/beta loop reads
        // straight from RAM.
        {
            let mut linear_caches: Vec<Qwen36LinearAttnHostCache> = Vec::new();
            for layer in bringup.model.layers.iter() {
                if let rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(la) = &layer.attn {
                    let vus = la.in_proj_a.shape[0];
                    let h_us = la.in_proj_a.shape[1];
                    let proj_bytes = vus * h_us * 2;
                    let mut a_w = vec![0u8; proj_bytes];
                    let mut b_w = vec![0u8; proj_bytes];
                    let mut a_log_h = vec![0u8; vus * 2];
                    let mut dt_bias_h = vec![0u8; vus * 2];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        cuMemcpyDtoH_v2(
                            a_w.as_mut_ptr() as *mut _,
                            la.in_proj_a.offset_bytes,
                            proj_bytes,
                        );
                        cuMemcpyDtoH_v2(
                            b_w.as_mut_ptr() as *mut _,
                            la.in_proj_b.offset_bytes,
                            proj_bytes,
                        );
                        cuMemcpyDtoH_v2(
                            a_log_h.as_mut_ptr() as *mut _,
                            la.a_log.offset_bytes,
                            a_log_h.len(),
                        );
                        cuMemcpyDtoH_v2(
                            dt_bias_h.as_mut_ptr() as *mut _,
                            la.dt_bias.offset_bytes,
                            dt_bias_h.len(),
                        );
                    }
                    let mut a_w_f32 = vec![0.0f32; vus * h_us];
                    let mut b_w_f32 = vec![0.0f32; vus * h_us];
                    for i in 0..(vus * h_us) {
                        a_w_f32[i] =
                            f16_bits_to_f32(u16::from_le_bytes([a_w[i * 2], a_w[i * 2 + 1]]));
                        b_w_f32[i] =
                            f16_bits_to_f32(u16::from_le_bytes([b_w[i * 2], b_w[i * 2 + 1]]));
                    }
                    let mut a_log_f32 = vec![0.0f32; vus];
                    let mut dt_bias_f32 = vec![0.0f32; vus];
                    for v in 0..vus {
                        a_log_f32[v] = f16_bits_to_f32(u16::from_le_bytes([
                            a_log_h[v * 2],
                            a_log_h[v * 2 + 1],
                        ]));
                        dt_bias_f32[v] = f16_bits_to_f32(u16::from_le_bytes([
                            dt_bias_h[v * 2],
                            dt_bias_h[v * 2 + 1],
                        ]));
                    }
                    linear_caches.push(Qwen36LinearAttnHostCache {
                        a_w_f32,
                        b_w_f32,
                        a_log_f32,
                        dt_bias_f32,
                        vus,
                        h_us,
                    });
                }
            }
            let cache_bytes: usize = linear_caches
                .iter()
                .map(|c| {
                    (c.a_w_f32.len() + c.b_w_f32.len()) * 4
                        + (c.a_log_f32.len() + c.dt_bias_f32.len()) * 4
                })
                .sum();
            eprintln!(
                "[qwen36] linear_attn host cache: {} layers, {:.1} MiB",
                linear_caches.len(),
                cache_bytes as f64 / (1024.0 * 1024.0)
            );
            bringup.linear_attn_host_cache = linear_caches;
        }

        // Phase 4b-prep iter15: cache the router weight matrices as
        // f32 host vectors so the per-layer router GEMV can skip the
        // 1 MiB DtoH + f16→f32 unpack every token.
        {
            let num_experts = bringup.arch.num_experts;
            let hidden_us = bringup.arch.base.hidden_size as usize;
            let mut caches: Vec<Vec<f32>> = Vec::new();
            for layer in bringup.model.layers.iter() {
                let router_bytes = num_experts * hidden_us * 2;
                let mut router_host = vec![0u8; router_bytes];
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    cuMemcpyDtoH_v2(
                        router_host.as_mut_ptr() as *mut _,
                        layer.moe.router.offset_bytes,
                        router_bytes,
                    );
                }
                let mut router_f32 = vec![0.0f32; num_experts * hidden_us];
                for i in 0..(num_experts * hidden_us) {
                    router_f32[i] = f16_bits_to_f32(u16::from_le_bytes([
                        router_host[i * 2],
                        router_host[i * 2 + 1],
                    ]));
                }
                caches.push(router_f32);
            }
            let total_bytes: usize = caches.iter().map(|c| c.len() * 4).sum();
            eprintln!(
                "[qwen36] router host cache: {} layers, {:.1} MiB",
                caches.len(),
                total_bytes as f64 / (1024.0 * 1024.0)
            );
            bringup.router_host_cache = caches;
        }

        // Phase 4b-prep iter16: cache shared-expert gate weights as
        // f32 host vectors. Saves 4 KiB DtoH × num_layers × per token.
        {
            let hidden_us = bringup.arch.base.hidden_size as usize;
            let mut caches: Vec<Vec<f32>> = Vec::new();
            for layer in bringup.model.layers.iter() {
                let sg_bytes = hidden_us * 2;
                let mut sg_host = vec![0u8; sg_bytes];
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    cuMemcpyDtoH_v2(
                        sg_host.as_mut_ptr() as *mut _,
                        layer.moe.shared_expert_gate_logit.offset_bytes,
                        sg_bytes,
                    );
                }
                let mut sg_f32 = vec![0.0f32; hidden_us];
                for k in 0..hidden_us {
                    sg_f32[k] =
                        f16_bits_to_f32(u16::from_le_bytes([sg_host[k * 2], sg_host[k * 2 + 1]]));
                }
                caches.push(sg_f32);
            }
            bringup.shared_gate_host_cache = caches;
        }

        // Phase 3e smoke: actually launch embedding_gather against the
        // loaded weights. If this throws, the rest of the forward
        // pipeline can't be built on top — fail fast.
        bringup.forward_outside_smoke()?;

        // Phase 4a: verify the reusable `forward_outside_only` API by
        // calling it on a slightly different sequence and logging the
        // argmax token id. Confirms the same kernel chain works
        // through the public method, ready for cuda_worker dispatch
        // in Phase 4b.
        let probe = bringup.forward_outside_only(&[1, 200, 2000, 20_000, 50_000])?;
        eprintln!(
            "[qwen36] forward_outside_only smoke: 5-token input → \
             argmax_token_id={probe} (garbage by design — per-layer \
             forward still TODO)"
        );

        // Phase 4c/4d: per-layer kernel chain. Launches Q+K+V
        // projections against the first full-attention layer's
        // blockwise FP8 weights. Synthetic input — proves all three
        // projection roles work end-to-end through the same kernel.
        bringup.forward_layer3_qkv_probe()?;
        // Phase 4g: partial RoPE launch against the precomputed
        // cos/sin tables. Identity at position 0, sanity-checks
        // ABI + table indexing.
        bringup.forward_layer3_rope_probe()?;
        // Phase 4h: paged f16 attention decode launch with a
        // single-token KV cache stand-in. Validates the FA2
        // decode kernel ABI works for Qwen's heterogeneous
        // (num_heads=16, num_kv_heads=2) shape.
        bringup.forward_layer3_paged_attn_probe()?;
        // Phase 4i: o_proj (closes the attention block).
        bringup.forward_layer3_o_proj_probe()?;
        // Phase 4j: attn_output_gate (sigmoid · attn_out via new
        // sigmoid_mul_f16 kernel).
        bringup.forward_layer3_attn_gate_probe()?;
        // Phase 4k: shared-expert gate_proj (first MoE-block kernel
        // launch). Routed-expert dispatch + bf16 router land in 4l/4m.
        bringup.forward_layer3_moe_shared_probe()?;
        // Phase 4l: router GEMV + top-8 selection (CPU-side smoke).
        bringup.forward_layer3_router_probe()?;
        // Phase 4m: full SwiGLU FFN for routed expert 0
        // (gate + up → silu·mul (host) → down).
        bringup.forward_layer3_routed_expert_probe()?;
        // Phase 4n: complete MoE block — top-8 routed experts (with
        // per-expert offsets) + shared expert (sigmoid-gated) →
        // weighted sum.
        bringup.forward_layer3_full_moe_probe()?;
        // Phase 4o (skeleton): linear-attn weight presence + shape
        // log. Recurrent kernel (Gated-DeltaNet ssm-scan) is real
        // CUDA work TBD.
        bringup.forward_layer0_linear_attn_probe()?;
        // Phase 4p: linear-attn input projections (in_proj_qkv +
        // in_proj_z). Confirms the FP8 GEMV kernel works on
        // linear-attn weight shapes.
        bringup.forward_layer0_linear_in_proj_probe()?;
        // Phase 4q: causal_conv1d_f16 — first piece of the
        // Gated-DeltaNet ssm-scan, runs against real layer-0
        // conv1d weights.
        bringup.forward_layer0_conv1d_probe()?;
        // Phase 4r: gated_delta_state_update_f16 — recurrent
        // state-space update, the heart of the linear-attn block.
        bringup.forward_layer0_ssm_state_probe()?;
        // Phase 4s: chain everything for layer 0 linear-attn.
        bringup.forward_layer0_linear_chain_probe()?;
        // Phase 4t: per-sequence state cache (zero-init + reset).
        bringup.linear_state_cache_probe()?;
        // Phase 4u: paged KV cache for full-attn layers.
        bringup.kv_cache_probe()?;

        // Phase 4v: end-to-end smoke that threads real hidden state
        // through layer 0's linear-attn in/out projections + final
        // norm + lm_head + argmax. Token output is garbage by design
        // (degenerate composition between in_proj and out_proj), but
        // proves per-layer FP8 kernels run on the production decode
        // path's actual hidden buffer instead of synthetic inputs.
        // Reset state before this synthetic call so it starts clean
        // (cuda_worker also resets on every request).
        bringup.reset_linear_state()?;
        bringup.reset_kv_cache()?;
        bringup.reset_conv_state()?;
        let probe_5d = bringup.forward_qwen36_decode(&[1, 200, 2000, 20_000, 50_000], 0, &[])?;
        eprintln!(
            "[qwen36] Phase 5d forward_qwen36_decode smoke: 5-token \
             input → argmax_token_id={probe_5d} (linear-attn rewritten \
             to vLLM-correct layout: Q[16,128]+K[16,128]+V[32,128] \
             split, per-head L2-norm on Q/K, in_proj_a/b for α/β, \
             GQA-expanded K/Q for state update, per-v-head readout)"
        );
        if std::env::var("RVLLM_QWEN36_MTP_FORWARD_SMOKE").as_deref() == Ok("1") {
            bringup.reset_linear_state()?;
            bringup.reset_kv_cache()?;
            bringup.reset_conv_state()?;
            let mtp_tok = bringup.forward_qwen36_mtp_one_token_probe(200, 2000, 0)?;
            bringup.reset_linear_state()?;
            bringup.reset_kv_cache()?;
            bringup.reset_conv_state()?;
            eprintln!(
                "[qwen36-mtp] one-token forward smoke: hidden_token=200 \
                 draft_input=2000 -> argmax_token_id={mtp_tok}"
            );
        }

        // Phase 2b-γ: vision smoke (gated by env). Reads a fixture
        // image from RVLLM_QWEN36_VISION_PROBE_PATH and runs the full
        // ViT forward, dumping output stats. Confirms the 27-block
        // forward + PatchMerger compose without crash and produce
        // sane f16 magnitudes. Quality vs HF reference is Phase 5.
        if let Ok(path) = std::env::var("RVLLM_QWEN36_VISION_PROBE_PATH") {
            match std::fs::read(&path) {
                Ok(bytes) => match bringup.forward_qwen_vision(&bytes) {
                    Ok(out) => {
                        let f32s: Vec<f32> = out
                            .data
                            .chunks_exact(2)
                            .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                            .collect();
                        let l2 = f32s.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let max = f32s.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                        eprintln!(
                            "[qwen36] vision probe: image={} → tokens={} hidden={} grid_thw={:?} L2={:.3} max={:.3}",
                            path, out.num_tokens, out.hidden_dim, out.grid_thw, l2, max,
                        );
                    }
                    Err(e) => eprintln!("[qwen36] vision probe failed: {e:?}"),
                },
                Err(e) => eprintln!("[qwen36] vision probe: cannot read {path}: {e}"),
            }
        }

        Ok(bringup)
    }

    pub fn kernels_dir(&self) -> &PathBuf {
        &self.paths.kernels_dir
    }

    /// Phase 4u: device pointer for full-attn layer's slice of the
    /// paged KV cache. `layer_seq_idx` is the sequential index of
    /// the full-attn layer (0..num_full_layers), NOT the absolute
    /// model layer index — caller maps via the layer_types array.
    pub fn kv_cache_layer_ptr(&self, layer_seq_idx: u32) -> u64 {
        let off = (layer_seq_idx as usize).saturating_mul(self.kv_cache_layer_bytes);
        if off + self.kv_cache_layer_bytes > self.kv_cache_bytes {
            return 0;
        }
        self.kv_cache_ptr + off as u64
    }

    /// NVFP4 commit 1: device pointer for full-attn layer's slice of
    /// the companion microscale buffer. Returns 0 when
    /// `kv_dtype == F16` (no scale buffer allocated) or when the
    /// computed offset overflows the buffer. Same `layer_seq_idx`
    /// semantics as `kv_cache_layer_ptr`.
    pub fn kv_cache_scale_layer_ptr(&self, layer_seq_idx: u32) -> u64 {
        if self.kv_cache_scale_ptr == 0 || self.kv_cache_scale_layer_bytes == 0 {
            return 0;
        }
        let off = (layer_seq_idx as usize).saturating_mul(self.kv_cache_scale_layer_bytes);
        if off + self.kv_cache_scale_layer_bytes > self.kv_cache_scale_bytes {
            return 0;
        }
        self.kv_cache_scale_ptr + off as u64
    }

    fn mtp_kv_layer_seq_idx(&self) -> u32 {
        self.arch
            .base
            .layer_types
            .iter()
            .filter(|t| matches!(t, rvllm_loader::LayerAttnType::Full))
            .count() as u32
    }

    /// Phase 4u: zero out the paged KV cache (all full-attn layers).
    /// Called on session boundaries alongside `reset_linear_state`.
    /// NVFP4 commit 1: also zeros the companion microscale buffer
    /// when present.
    pub fn reset_kv_cache(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8Async(
                self.kv_cache_ptr,
                0,
                self.kv_cache_bytes,
                self.stream.raw() as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 reset_kv_cache",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            if self.kv_cache_scale_ptr != 0 && self.kv_cache_scale_bytes > 0 {
                let rc = cuMemsetD8Async(
                    self.kv_cache_scale_ptr,
                    0,
                    self.kv_cache_scale_bytes,
                    self.stream.raw() as CUstream,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 reset_kv_cache (scale)",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Phase 4t: zero out the per-sequence linear-attn state cache.
    /// Called on session boundaries (`/new`, fresh request without
    /// session continuity) so a stale state from a prior sequence
    /// doesn't leak into the new one.
    pub fn reset_linear_state(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8Async(
                self.linear_state_ptr,
                0,
                self.linear_state_bytes,
                self.stream.raw() as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 reset_linear_state",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Recurrent-state snapshot for prompt-lookup spec-decode. The
    /// Gated-DeltaNet linear state + Conv1d state are RECURRENT: a
    /// verify forward over [current, d_0, …, d_{K-1}] advances them
    /// through all K drafts in place. If we accept fewer than K
    /// drafts, the state has been polluted by rejected tokens with
    /// no built-in rollback. This pair of helpers lets the spec
    /// session snapshot the state before verify and restore it on
    /// any partial-accept iter, after which a separate commit-only
    /// forward replays only the accepted prefix.
    ///
    /// `dst_*_ptr` must point at device buffers sized to
    /// `linear_state_bytes` / `conv_state_bytes` respectively. The
    /// spec session owns the scratch allocation.
    pub fn snapshot_recurrent_state(&self, dst_linear_ptr: u64, dst_conv_ptr: u64) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                dst_linear_ptr,
                self.linear_state_ptr,
                self.linear_state_bytes,
                self.stream.raw() as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 snapshot_recurrent_state(linear)",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            let rc = cuMemcpyDtoDAsync_v2(
                dst_conv_ptr,
                self.conv_state_ptr,
                self.conv_state_bytes,
                self.stream.raw() as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 snapshot_recurrent_state(conv)",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let _ = (dst_linear_ptr, dst_conv_ptr);
        Ok(())
    }

    pub fn restore_recurrent_state(&self, src_linear_ptr: u64, src_conv_ptr: u64) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                self.linear_state_ptr,
                src_linear_ptr,
                self.linear_state_bytes,
                self.stream.raw() as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 restore_recurrent_state(linear)",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            let rc = cuMemcpyDtoDAsync_v2(
                self.conv_state_ptr,
                src_conv_ptr,
                self.conv_state_bytes,
                self.stream.raw() as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 restore_recurrent_state(conv)",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let _ = (src_linear_ptr, src_conv_ptr);
        Ok(())
    }

    pub fn recurrent_state_bytes(&self) -> (usize, usize) {
        (self.linear_state_bytes, self.conv_state_bytes)
    }

    /// Phase 5f: zero out the conv1d state cache.
    pub fn reset_conv_state(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8Async(
                self.conv_state_ptr,
                0,
                self.conv_state_bytes,
                self.stream.raw() as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 reset_conv_state",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Phase 5f: device pointer for layer N's conv-state slice.
    pub fn conv_state_layer_ptr(&self, layer_seq_idx: u32) -> u64 {
        let off = (layer_seq_idx as usize).saturating_mul(self.conv_state_layer_bytes);
        if off + self.conv_state_layer_bytes > self.conv_state_bytes {
            return 0;
        }
        self.conv_state_ptr + off as u64
    }

    /// Phase 4t: device pointer for layer N's slice of the state cache.
    /// Returns 0 if the layer index is out of bounds.
    pub fn linear_state_layer_ptr(&self, layer_seq_idx: u32) -> u64 {
        let off = (layer_seq_idx as usize).saturating_mul(self.linear_state_layer_bytes);
        if off + self.linear_state_layer_bytes > self.linear_state_bytes {
            return 0;
        }
        self.linear_state_ptr + off as u64
    }

    /// Phase 3e smoke test: launch the embedding-gather kernel against
    /// the loaded `embed_tokens` table for a hardcoded 4-token input,
    /// fence the stream, and DtoH the first 4 hidden-state floats so
    /// we can log them. Validates that:
    ///   - the bring-up's CUDA context + arena + stream + KernelLoader
    ///     are wired correctly
    ///   - the f16 `embed_tokens` upload reaches the device with
    ///     the right layout
    ///   - the EmbeddingGatherLaunch ABI matches Qwen 3.6 dims
    /// without exercising any per-layer math (still Phase 3f+).
    /// Phase 4a: outside-only forward over arbitrary input tokens.
    /// Skips all 40 transformer layers — runs only:
    ///   embed_tokens → final_norm + fp8quant → lm_head fp8_gemm →
    ///   CPU argmax over the LAST token's logits.
    /// Returns the argmax token id of the last input position. Output
    /// will be garbage (no per-layer math), but the kernel pipeline
    /// is exercised end-to-end.
    pub fn forward_outside_only(&self, token_ids: &[i32]) -> Result<i32> {
        if token_ids.is_empty() {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_outside_only: empty token_ids",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        let last_idx = token_ids.len() - 1;
        let hidden = self.arch.base.hidden_size as u32;
        let vocab = self.arch.base.vocab_size as u32;
        let num_tokens = token_ids.len() as u32;

        // Allocate device regions: token IDs (i32) + hidden state (f16).
        let mut token_bytes_owned: Vec<u8> = Vec::with_capacity(token_ids.len() * 4);
        for t in token_ids {
            token_bytes_owned.extend_from_slice(&t.to_le_bytes());
        }
        let tokens_region =
            self.arena
                .region("qwen36_outside_tokens", token_bytes_owned.len(), 16)?;
        unsafe { tokens_region.copy_from_host(&token_bytes_owned)? };

        let hidden_bytes = (num_tokens as usize) * (hidden as usize) * 2; // f16
        let hidden_region = self
            .arena
            .region("qwen36_outside_hidden", hidden_bytes, 16)?;

        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens,
                hidden,
                vocab,
            }
            .launch(
                self.outside_kernels.fn_embedding_gather_f16,
                hidden_region.device_ptr(),
                self.model.outside.embed_tokens.offset_bytes,
                tokens_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }

        let eps = self.arch.base.rms_norm_eps;
        let hidden_fp8_bytes = (num_tokens as usize) * (hidden as usize);
        let hidden_scale_bytes = (num_tokens as usize) * 4;
        let logits_bytes = (num_tokens as usize) * (vocab as usize) * 2;
        let hidden_fp8_region =
            self.arena
                .region("qwen36_outside_hidden_fp8", hidden_fp8_bytes, 16)?;
        let hidden_scale_region =
            self.arena
                .region("qwen36_outside_hidden_scale", hidden_scale_bytes, 16)?;
        let logits_region = self
            .arena
            .region("qwen36_outside_logits", logits_bytes, 16)?;
        let stream_raw = self.stream.raw() as u64;

        unsafe {
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_fused_rmsnorm_fp8_quant,
                hidden_fp8_region.device_ptr(),
                hidden_scale_region.device_ptr(),
                hidden_region.device_ptr(),
                self.model.outside.final_norm.offset_bytes,
                stream_raw,
            )?;
        }

        #[cfg(feature = "cuda")]
        unsafe {
            self.cublaslt.fp8_gemm(
                hidden_fp8_region.device_ptr(),
                self.model.outside.lm_head_fp8.offset_bytes,
                logits_region.device_ptr(),
                num_tokens as i32,
                vocab as i32,
                hidden as i32,
                hidden_scale_region.device_ptr(),
                self.model.outside.lm_head_fp8.scale_ptr,
                stream_raw,
            )?;
        }
        self.stream.fence()?;

        // GPU-side argmax over the LAST token's logits row. The
        // argmax_f16_kernel does one block-reduction per row and
        // writes a single i32 — we DtoH 4 bytes instead of the full
        // ~526 KB vocab f16 buffer the previous host-side scan
        // pulled across PCIe every step. (Codex round 16 #3.)
        let logits_row_bytes = (vocab as usize) * 2;
        let last_offset = (last_idx as u64) * (logits_row_bytes as u64);
        let token_region = self.arena.region("qwen36_argmax_tok", 4, 4)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            // Launch with one block per row (here always 1) and up to
            // 1024 threads collaborating on the reduction. Matches
            // the kernel's documented launch config.
            let block_dim: u32 = (vocab as u32).min(1024);
            let mut row_ptr = logits_region.device_ptr() + last_offset;
            let mut out_ptr = token_region.device_ptr();
            let mut vsz: i32 = vocab as i32;
            let args = [
                (&mut row_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut vsz) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_argmax_f16.raw() as CUfunction,
                /*grid*/ 1,
                1,
                1,
                /*block*/ block_dim,
                1,
                1,
                /*shared*/ 0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 argmax_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;
        let mut tok_buf = [0u8; 4];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(tok_buf.as_mut_ptr() as *mut _, token_region.device_ptr(), 4);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 argmax_f16 DtoH(token)",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(i32::from_le_bytes(tok_buf))
    }

    /// Phase 4c/4d probe: chain Q + K + V projection launches against
    /// the first full-attention layer's blockwise FP8 weights. Uses a
    /// synthetic all-ones f16 input row to keep the smoke self-
    /// contained — the kernel chain is real, the values aren't.
    /// Validates:
    ///   - blockwise FP8 GEMV ABI consumes per-layer weight + [N/128,
    ///     K/128] f32 blockscale for all three projection roles.
    ///   - q_proj / k_proj / v_proj device pointers + blockscales are
    ///     populated correctly by the qwen36 loader.
    ///   - shape relations: q_proj outputs `2 * num_heads * head_dim`
    ///     (Q + per-head gate concat for `attn_output_gate=true`);
    ///     k_proj / v_proj output `num_kv_heads * head_dim`.
    pub fn forward_layer3_qkv_probe(&self) -> Result<()> {
        let layer_idx = match self
            .arch
            .base
            .layer_types
            .iter()
            .position(|t| matches!(t, rvllm_loader::LayerAttnType::Full))
        {
            Some(i) => i,
            None => {
                eprintln!(
                    "[qwen36] forward_layer3_qkv_probe: no full-attention layer found, skipping"
                );
                return Ok(());
            }
        };
        let layer = match &self.model.layers[layer_idx].attn {
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Full(l) => l,
            _ => {
                eprintln!("[qwen36] forward_layer3_qkv_probe: layer {layer_idx} is not Full");
                return Ok(());
            }
        };
        let kernel = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => {
                eprintln!("[qwen36] forward_layer3_qkv_probe: f16in GEMV kernel unavailable on this arch, skipping");
                return Ok(());
            }
        };
        let hidden = self.arch.base.hidden_size as u32;
        let m: u32 = 1;

        // Synthetic f16 all-ones input — h(i) = 1.0 (f16 1.0 = 0x3c00).
        let one_bits = 0x3c00u16.to_le_bytes();
        let mut input_bytes = Vec::with_capacity((hidden as usize) * 2);
        for _ in 0..hidden {
            input_bytes.extend_from_slice(&one_bits);
        }
        let in_region = self
            .arena
            .region("qwen36_l3qkv_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };

        // Closure: run one projection role + return the first 4 f16
        // output values as f32 for logging.
        let project = |name: &'static str,
                       region_name: &'static str,
                       weight: &rvllm_loader::weights::Fp8Weight|
         -> Result<[f32; 4]> {
            let blockscale_ptr = match weight.blockscale_ptr {
                Some(p) => p,
                None => {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36_qkv_probe missing blockscale_ptr",
                        rvllm_core::CudaErrorKind::Other,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            };
            let n = weight.shape[0] as u32;
            let k = weight.shape[1] as u32;
            let out_bytes = (m as usize) * (n as usize) * 2;
            let out_region = self.arena.region(region_name, out_bytes, 16)?;
            unsafe {
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m, n, k }.launch(
                    kernel,
                    out_region.device_ptr(),
                    weight.offset_bytes,
                    blockscale_ptr,
                    in_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
            }
            self.stream.fence()?;
            let mut probe = [0u8; 8];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let rc = cuMemcpyDtoH_v2(
                    probe.as_mut_ptr() as *mut _,
                    out_region.device_ptr(),
                    probe.len(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36_qkv_probe DtoH",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            let _ = name;
            Ok([
                f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]])),
                f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]])),
                f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]])),
                f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]])),
            ])
        };

        let q = project("q", "qwen36_l3q_out", &layer.q_proj)?;
        let k = project("k", "qwen36_l3k_out", &layer.k_proj)?;
        let v = project("v", "qwen36_l3v_out", &layer.v_proj)?;
        eprintln!(
            "[qwen36] forward_layer3_qkv_probe: layer={layer_idx} \
             q={:?} ({:?}) k={:?} ({:?}) v={:?} ({:?})",
            q, layer.q_proj.shape, k, layer.k_proj.shape, v, layer.v_proj.shape,
        );

        // Phase 4e: Q-Norm + K-Norm. Qwen 3.6 ships per-head RMSNorm
        // weights separately (`q_norm [head_dim]`, `k_norm [head_dim]`)
        // — different from Gemma 4's fused QK-norm. We re-use the
        // generic `rmsnorm_inplace_f16` kernel with `num_tokens =
        // <heads>` rows of `hidden = head_dim`, which is exactly the
        // per-head pattern (each head's [head_dim] vector RMSNorm'd
        // against the same gamma).
        //
        // Smoke uses a fresh synthetic per-head all-twos input rather
        // than chaining off the previous Q/K out regions — keeps the
        // probe self-contained and proves the kernel + per-layer
        // q_norm / k_norm gamma weights work without depending on the
        // qkv-projection layout (which varies with `attn_output_gate`).
        let head_dim = self.arch.base.head_dim as u32;
        let num_heads = self.arch.base.num_attention_heads as u32;
        let num_kv_heads = self.arch.base.num_key_value_heads as u32;
        let eps = self.arch.base.rms_norm_eps;
        let two_bits = 0x4000u16.to_le_bytes(); // f16 2.0
        let mut q_in_bytes = Vec::with_capacity((num_heads as usize) * (head_dim as usize) * 2);
        for _ in 0..(num_heads as usize) * (head_dim as usize) {
            q_in_bytes.extend_from_slice(&two_bits);
        }
        let mut k_in_bytes = Vec::with_capacity((num_kv_heads as usize) * (head_dim as usize) * 2);
        for _ in 0..(num_kv_heads as usize) * (head_dim as usize) {
            k_in_bytes.extend_from_slice(&two_bits);
        }
        let q_norm_in_region = self
            .arena
            .region("qwen36_l3qnorm_in", q_in_bytes.len(), 16)?;
        let k_norm_in_region = self
            .arena
            .region("qwen36_l3knorm_in", k_in_bytes.len(), 16)?;
        unsafe {
            q_norm_in_region.copy_from_host(&q_in_bytes)?;
            k_norm_in_region.copy_from_host(&k_in_bytes)?;
        }

        // Per-head RMSNorm: num_tokens = num_heads, hidden = head_dim.
        let launch_norm =
            |x_ptr: u64, gamma_ptr: u64, heads: u32, label: &'static str| -> Result<()> {
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let mut x = x_ptr;
                    let mut g = gamma_ptr;
                    let mut e = eps;
                    let mut h = head_dim as i32;
                    let args = [
                        (&mut x) as *mut u64 as *mut core::ffi::c_void,
                        (&mut g) as *mut u64 as *mut core::ffi::c_void,
                        (&mut e) as *mut f32 as *mut core::ffi::c_void,
                        (&mut h) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let block = head_dim.min(1024);
                    let rc = cuLaunchKernel(
                        self.outside_kernels.fn_rmsnorm_inplace_f16.raw() as CUfunction,
                        heads,
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            label,
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                Ok(())
            };

        launch_norm(
            q_norm_in_region.device_ptr(),
            layer.q_norm.offset_bytes,
            num_heads,
            "qwen36_q_norm",
        )?;
        launch_norm(
            k_norm_in_region.device_ptr(),
            layer.k_norm.offset_bytes,
            num_kv_heads,
            "qwen36_k_norm",
        )?;
        self.stream.fence()?;

        let mut q_probe = [0u8; 8];
        let mut k_probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                q_probe.as_mut_ptr() as *mut _,
                q_norm_in_region.device_ptr(),
                q_probe.len(),
            );
            let _ = cuMemcpyDtoH_v2(
                k_probe.as_mut_ptr() as *mut _,
                k_norm_in_region.device_ptr(),
                k_probe.len(),
            );
        }
        let qn = [
            f16_bits_to_f32(u16::from_le_bytes([q_probe[0], q_probe[1]])),
            f16_bits_to_f32(u16::from_le_bytes([q_probe[2], q_probe[3]])),
            f16_bits_to_f32(u16::from_le_bytes([q_probe[4], q_probe[5]])),
            f16_bits_to_f32(u16::from_le_bytes([q_probe[6], q_probe[7]])),
        ];
        let kn = [
            f16_bits_to_f32(u16::from_le_bytes([k_probe[0], k_probe[1]])),
            f16_bits_to_f32(u16::from_le_bytes([k_probe[2], k_probe[3]])),
            f16_bits_to_f32(u16::from_le_bytes([k_probe[4], k_probe[5]])),
            f16_bits_to_f32(u16::from_le_bytes([k_probe[6], k_probe[7]])),
        ];
        eprintln!(
            "[qwen36] forward_layer3_qknorm: per-head rmsnorm (head_dim={head_dim}, eps={eps:.0e}) \
             q_norm[head0,0..4]={qn:?} k_norm[head0,0..4]={kn:?}"
        );
        Ok(())
    }

    /// Phase 4g probe: launch `fused_rope_partial_f16kv` against
    /// synthetic Q/K/V buffers and a small KV-cache stand-in. Validates:
    ///   - the partial RoPE kernel ABI (15 args: q_in/k_in/v_in/
    ///     q_out/k_cache/v_cache/cos/sin/positions/slot_mapping +
    ///     5 ints) matches what the qwen36 RoPE wiring will need.
    ///   - the RoPE cos/sin tables uploaded in Phase 4f are usable
    ///     by the kernel (kernel reads cos[pos*hd/2 + i] / sin
    ///     analogues, so a position-0 lookup hits cos=1.0/sin=0.0,
    ///     which means RoPE is identity for position 0 — the
    ///     post-rope Q values should equal the pre-rope Q values).
    /// MRoPE section dispatch (sections [11, 11, 10]) is a no-op
    /// for text-only mode at position 0; Phase 4h+ wires real
    /// positions from the input sequence.
    pub fn forward_layer3_rope_probe(&self) -> Result<()> {
        let head_dim = self.arch.base.head_dim as u32;
        let num_heads = self.arch.base.num_attention_heads as u32;
        let num_kv_heads = self.arch.base.num_key_value_heads as u32;
        // Qwen 3.6 partial_rotary_factor = 0.25 → rotary_dim = 64
        // (only the first 64 of the 256 head_dim values are rotated).
        let rotary_dim = (head_dim as f32 * 0.25) as u32;
        let num_tokens: u32 = 1;

        // Synthetic f16 all-twos for Q/K/V (per-head, all positions).
        let two_bits = 0x4000u16.to_le_bytes();
        let q_elems = (num_tokens as usize) * (num_heads as usize) * (head_dim as usize);
        let kv_elems = (num_tokens as usize) * (num_kv_heads as usize) * (head_dim as usize);
        let mut q_in_bytes = Vec::with_capacity(q_elems * 2);
        for _ in 0..q_elems {
            q_in_bytes.extend_from_slice(&two_bits);
        }
        let mut kv_in_bytes = Vec::with_capacity(kv_elems * 2);
        for _ in 0..kv_elems {
            kv_in_bytes.extend_from_slice(&two_bits);
        }

        let q_region = self.arena.region("qwen36_l3rope_q", q_in_bytes.len(), 16)?;
        let k_region = self
            .arena
            .region("qwen36_l3rope_k", kv_in_bytes.len(), 16)?;
        let v_region = self
            .arena
            .region("qwen36_l3rope_v", kv_in_bytes.len(), 16)?;
        unsafe {
            q_region.copy_from_host(&q_in_bytes)?;
            k_region.copy_from_host(&kv_in_bytes)?;
            v_region.copy_from_host(&kv_in_bytes)?;
        }
        // KV cache stand-in: 1 slot's worth, same byte size as one
        // [num_kv_heads, head_dim] vector. Real KV cache lives in
        // Phase 4h.
        let k_cache_region = self
            .arena
            .region("qwen36_l3rope_kc", kv_in_bytes.len(), 16)?;
        let v_cache_region = self
            .arena
            .region("qwen36_l3rope_vc", kv_in_bytes.len(), 16)?;

        // Position [0] + slot_mapping [0] (single-token, slot index 0).
        let zero_i32 = 0i32.to_le_bytes();
        let pos_region = self.arena.region("qwen36_l3rope_pos", 4, 16)?;
        let slot_region = self.arena.region("qwen36_l3rope_slot", 4, 16)?;
        unsafe {
            pos_region.copy_from_host(&zero_i32)?;
            slot_region.copy_from_host(&zero_i32)?;
        }

        // Launch fused_rope_partial_f16kv. Kernel sig (16 args from
        // gemma4_layer_exec.rs::rope_f16kv): q_in, k_in, v_in, q_out
        // (alias=q_in for in-place), k_cache, v_cache, cos, sin,
        // positions, slot_mapping, num_tokens, num_heads, num_kv_heads,
        // head_dim, rotary_dim.
        #[cfg(feature = "cuda")]
        unsafe {
            let mut q_in = q_region.device_ptr();
            let mut k_in = k_region.device_ptr();
            let mut v_in = v_region.device_ptr();
            let mut q_out = q_region.device_ptr(); // in-place
            let mut k_cache = k_cache_region.device_ptr();
            let mut v_cache = v_cache_region.device_ptr();
            let mut cos = self.rope_cos;
            let mut sin = self.rope_sin;
            let mut positions = pos_region.device_ptr();
            let mut slot_mapping = slot_region.device_ptr();
            let mut nt = num_tokens as i32;
            let mut nh = num_heads as i32;
            let mut nkvh = num_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut rd = rotary_dim as i32;
            let args = [
                (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_cache) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_cache) as *mut u64 as *mut core::ffi::c_void,
                (&mut cos) as *mut u64 as *mut core::ffi::c_void,
                (&mut sin) as *mut u64 as *mut core::ffi::c_void,
                (&mut positions) as *mut u64 as *mut core::ffi::c_void,
                (&mut slot_mapping) as *mut u64 as *mut core::ffi::c_void,
                (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut rd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let max_heads = num_heads.max(num_kv_heads);
            let grid = (num_tokens, max_heads, 1);
            let block = ((head_dim / 2).max(32), 1, 1);
            rvllm_fused::launch_raw(
                self.outside_kernels.fn_fused_rope_partial_f16kv,
                grid,
                block,
                0,
                self.stream.raw() as u64,
                &args,
            )?;
        }
        self.stream.fence()?;

        // DtoH first 4 f16 of post-rope Q. At position=0 RoPE should
        // be identity (cos=1, sin=0), so q_out[0..4] == 2.0 if the
        // kernel honours the position-0 contract.
        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                q_region.device_ptr(),
                probe.len(),
            );
        }
        let q0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let q1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let q2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let q3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_layer3_rope_probe: head_dim={head_dim} \
             rotary_dim={rotary_dim} (partial 0.25) pos=0 → \
             q_post_rope[head0,0..4]=[{q0:.3}, {q1:.3}, {q2:.3}, {q3:.3}] \
             (expected ≈ 2.0 at pos=0 — RoPE is identity there)"
        );
        Ok(())
    }

    /// Phase 4h probe: launch `flash_attention_2_decode_f16io_kernel`
    /// against a tiny paged f16 KV cache. Validates the 14-arg ABI of
    /// the f16-IO decode kernel works for Qwen's dims (num_heads=16,
    /// num_kv_heads=2, head_dim=256). Synthetic single-token input
    /// with `context_len = 1` (one slot of KV in the cache); attention
    /// reduces to `softmax(QK^T/sqrt(d))V` over a single key — the
    /// softmax weight is exactly 1.0, so output should equal V.
    pub fn forward_layer3_paged_attn_probe(&self) -> Result<()> {
        let head_dim = self.arch.base.head_dim as u32;
        let num_heads = self.arch.base.num_attention_heads as u32;
        let num_kv_heads = self.arch.base.num_key_value_heads as u32;
        let block_size: u32 = 16;
        let num_blocks: u32 = 1;
        let max_blocks_per_seq: u32 = 1;
        let num_seqs: u32 = 1;

        // Sized buffers (all f16 except block_tables/context_lens int).
        let q_bytes = (num_seqs as usize) * (num_heads as usize) * (head_dim as usize) * 2;
        let kv_cache_bytes = (num_blocks as usize)
            * (block_size as usize)
            * (num_kv_heads as usize)
            * (head_dim as usize)
            * 2;
        let out_bytes = q_bytes;

        // Q = all-twos, V = all-threes, K = all-ones (so softmax weight
        // simplifies but the output isn't trivially equal to V — lets
        // us see that the kernel actually composes Q·K^T·V correctly).
        let q_region = self.arena.region("qwen36_l3pa_q", q_bytes, 16)?;
        let k_cache_region = self.arena.region("qwen36_l3pa_kc", kv_cache_bytes, 16)?;
        let v_cache_region = self.arena.region("qwen36_l3pa_vc", kv_cache_bytes, 16)?;
        let out_region = self.arena.region("qwen36_l3pa_out", out_bytes, 16)?;

        let two_bits = 0x4000u16.to_le_bytes(); // f16 2.0
        let one_bits = 0x3c00u16.to_le_bytes(); // f16 1.0
        let three_bits = 0x4200u16.to_le_bytes(); // f16 3.0
        let mut q_init = Vec::with_capacity(q_bytes);
        for _ in 0..q_bytes / 2 {
            q_init.extend_from_slice(&two_bits);
        }
        let mut k_init = Vec::with_capacity(kv_cache_bytes);
        for _ in 0..kv_cache_bytes / 2 {
            k_init.extend_from_slice(&one_bits);
        }
        let mut v_init = Vec::with_capacity(kv_cache_bytes);
        for _ in 0..kv_cache_bytes / 2 {
            v_init.extend_from_slice(&three_bits);
        }
        unsafe {
            q_region.copy_from_host(&q_init)?;
            k_cache_region.copy_from_host(&k_init)?;
            v_cache_region.copy_from_host(&v_init)?;
        }

        // block_tables = [[0]] (seq 0 → block 0)
        // context_lens = [1] (one valid token in the cache)
        let zero_i32 = 0i32.to_le_bytes();
        let one_i32 = 1i32.to_le_bytes();
        let bt_region = self.arena.region("qwen36_l3pa_bt", 4, 16)?;
        let cl_region = self.arena.region("qwen36_l3pa_cl", 4, 16)?;
        unsafe {
            bt_region.copy_from_host(&zero_i32)?;
            cl_region.copy_from_host(&one_i32)?;
        }

        let scale = 1.0 / (head_dim as f32).sqrt();

        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            // FA2 decode kernel needs dynamic shared memory:
            //   2 * FA2_BC * hd * 4 (K/V tiles) + FA2_BC * 4 (max_logits) + warps * 4
            // Mirrors gemma4 decode.rs:290. For Qwen hd=256 this is
            // ~64 KiB which exceeds the 48 KiB static cap, so the
            // kernel needs `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE)`
            // before the launch.
            const FA2_THREADS: i32 = 128;
            const FA2_BC: i32 = 32;
            let hd_i = head_dim as i32;
            let smem_bytes = 2 * FA2_BC * hd_i * 4 + FA2_BC * 4 + (FA2_THREADS / 32) * 4;
            if smem_bytes as u32 >= 48 * 1024 {
                let rc = cuFuncSetAttribute(
                    self.outside_kernels.fn_flash_attention_2_decode_f16io.raw() as CUfunction,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem_bytes,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36_l3pa cuFuncSetAttribute",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }

            let mut output = out_region.device_ptr();
            let mut query = q_region.device_ptr();
            let mut key_cache = k_cache_region.device_ptr();
            let mut value_cache = v_cache_region.device_ptr();
            let mut block_tables = bt_region.device_ptr();
            let mut context_lens = cl_region.device_ptr();
            let mut scale_arg = scale;
            let mut nh = num_heads as i32;
            let mut nkvh = num_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut bs = block_size as i32;
            let mut mbps = max_blocks_per_seq as i32;
            let mut window: i32 = -1; // no sliding window
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
                self.outside_kernels.fn_flash_attention_2_decode_f16io.raw() as CUfunction,
                num_seqs,
                num_heads,
                1,
                FA2_THREADS as u32,
                1,
                1,
                smem_bytes as u32,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_l3pa flash_attention_2_decode_f16io",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;

        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                out_region.device_ptr(),
                probe.len(),
            );
        }
        let o0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let o1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let o2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let o3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_layer3_paged_attn_probe: heads={num_heads} \
             kv_heads={num_kv_heads} hd={head_dim} block_size={block_size} \
             ctx_len=1 → out[head0,0..4]=[{o0:.3}, {o1:.3}, {o2:.3}, {o3:.3}] \
             (expected ≈ 3.0 — single-key softmax weight is 1.0, output = V = 3.0)"
        );
        Ok(())
    }

    /// Phase 4i probe: launch o_proj (last step of the attention
    /// block). Same blockwise FP8 GEMV kernel as Q/K/V — only the
    /// shape changes: `[hidden=2048, n=num_heads*head_dim=4096]`. The
    /// input is the per-token attention output (concatenated across
    /// heads); the output is the residual contribution that feeds
    /// the post-attention residual + MoE block.
    pub fn forward_layer3_o_proj_probe(&self) -> Result<()> {
        let layer_idx = match self
            .arch
            .base
            .layer_types
            .iter()
            .position(|t| matches!(t, rvllm_loader::LayerAttnType::Full))
        {
            Some(i) => i,
            None => return Ok(()),
        };
        let layer = match &self.model.layers[layer_idx].attn {
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Full(l) => l,
            _ => return Ok(()),
        };
        let kernel = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => return Ok(()),
        };
        let n = layer.o_proj.shape[0] as u32;
        let k = layer.o_proj.shape[1] as u32;
        let m: u32 = 1;
        let blockscale_ptr = match layer.o_proj.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };

        // Synthetic f16 all-twos input matching o_proj's K dim.
        let two_bits = 0x4000u16.to_le_bytes();
        let mut input_bytes = Vec::with_capacity((k as usize) * 2);
        for _ in 0..k {
            input_bytes.extend_from_slice(&two_bits);
        }
        let in_region = self.arena.region("qwen36_l3op_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };
        let out_bytes = (m as usize) * (n as usize) * 2;
        let out_region = self.arena.region("qwen36_l3op_out", out_bytes, 16)?;

        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m, n, k }.launch(
                kernel,
                out_region.device_ptr(),
                layer.o_proj.offset_bytes,
                blockscale_ptr,
                in_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }
        self.stream.fence()?;

        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                out_region.device_ptr(),
                probe.len(),
            );
        }
        let o0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let o1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let o2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let o3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_layer3_o_proj_probe: o_proj=[{n}, {k}] → \
             out[0..4]=[{o0:.3}, {o1:.3}, {o2:.3}, {o3:.3}] \
             (blockwise FP8 GEMV o_proj against synthetic f16 attn-out)"
        );
        Ok(())
    }

    /// Phase 4j probe: launch the new `sigmoid_mul_f16_kernel` with
    /// gate_logits = 0.0 (sigmoid(0) = 0.5) and values = 4.0, so the
    /// expected output is 2.0 across the buffer. Validates the new
    /// CUDA kernel built + the launch ABI.
    pub fn forward_layer3_attn_gate_probe(&self) -> Result<()> {
        let head_dim = self.arch.base.head_dim as u32;
        let num_heads = self.arch.base.num_attention_heads as u32;
        let n = num_heads * head_dim;

        // values = 4.0 (f16 0x4400), gate_logits = 0.0 (f16 0x0000).
        let four_bits = 0x4400u16.to_le_bytes();
        let zero_bits = 0x0000u16.to_le_bytes();
        let mut v_bytes = Vec::with_capacity((n as usize) * 2);
        let mut g_bytes = Vec::with_capacity((n as usize) * 2);
        for _ in 0..n {
            v_bytes.extend_from_slice(&four_bits);
            g_bytes.extend_from_slice(&zero_bits);
        }
        let v_region = self.arena.region("qwen36_l3sg_v", v_bytes.len(), 16)?;
        let g_region = self.arena.region("qwen36_l3sg_g", g_bytes.len(), 16)?;
        let o_region = self.arena.region("qwen36_l3sg_o", v_bytes.len(), 16)?;
        unsafe {
            v_region.copy_from_host(&v_bytes)?;
            g_region.copy_from_host(&g_bytes)?;
        }

        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut output = o_region.device_ptr();
            let mut values = v_region.device_ptr();
            let mut gate = g_region.device_ptr();
            let mut nn = n as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut values) as *mut u64 as *mut core::ffi::c_void,
                (&mut gate) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = (n + block - 1) / block;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_sigmoid_mul_f16.raw() as CUfunction,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_l3sg sigmoid_mul_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;

        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                o_region.device_ptr(),
                probe.len(),
            );
        }
        let o0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let o1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let o2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let o3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_layer3_attn_gate_probe: n={n} \
             values=4.0 gate_logits=0.0 → out[0..4]=[{o0:.3}, {o1:.3}, \
             {o2:.3}, {o3:.3}] (expected ≈ 2.0 — sigmoid(0)=0.5, \
             4.0×0.5=2.0)"
        );
        Ok(())
    }

    /// Phase 4k probe: launch the SHARED-EXPERT gate_proj of the
    /// per-layer MoE block. Reuses the same blockwise FP8 GEMV kernel
    /// as Q/K/V/o_proj — proves the MoE block's `shared_expert.*`
    /// weight pointers + blockscales are populated correctly by the
    /// Phase-2b loader. The 256 routed experts (top-8 dispatch) and
    /// the bf16 router (`mlp.gate`) are deferred to Phase 4l/4m where
    /// the per-token expert selection + grouped-GEMM dispatch land.
    pub fn forward_layer3_moe_shared_probe(&self) -> Result<()> {
        let layer_idx = match self
            .arch
            .base
            .layer_types
            .iter()
            .position(|t| matches!(t, rvllm_loader::LayerAttnType::Full))
        {
            Some(i) => i,
            None => return Ok(()),
        };
        let moe = &self.model.layers[layer_idx].moe;
        let kernel = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => return Ok(()),
        };
        let w = &moe.shared_expert_gate_proj;
        let blockscale_ptr = match w.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let n = w.shape[0] as u32;
        let k = w.shape[1] as u32;
        let m: u32 = 1;

        // Synthetic f16 all-twos input matching gate_proj's K dim
        // (= hidden_size = 2048).
        let two_bits = 0x4000u16.to_le_bytes();
        let mut input_bytes = Vec::with_capacity((k as usize) * 2);
        for _ in 0..k {
            input_bytes.extend_from_slice(&two_bits);
        }
        let in_region = self
            .arena
            .region("qwen36_l3moe_sh_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };
        let out_bytes = (m as usize) * (n as usize) * 2;
        let out_region = self.arena.region("qwen36_l3moe_sh_out", out_bytes, 16)?;

        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m, n, k }.launch(
                kernel,
                out_region.device_ptr(),
                w.offset_bytes,
                blockscale_ptr,
                in_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }
        self.stream.fence()?;

        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                out_region.device_ptr(),
                probe.len(),
            );
        }
        let o0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let o1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let o2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let o3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_layer3_moe_shared_probe: layer={layer_idx} \
             shared_expert.gate_proj=[{n}, {k}] → \
             out[0..4]=[{o0:.3}, {o1:.3}, {o2:.3}, {o3:.3}] \
             (blockwise FP8 GEMV against per-layer shared-expert weights)"
        );
        Ok(())
    }

    /// Phase 4l probe: router GEMV + top-8 selection on the host
    /// (no GPU launch). The router weight is small — `[256, 2048]`
    /// bf16 = 1 MiB — so a one-shot DtoH + CPU matmul on a synthetic
    /// hidden state is cheap enough for a smoke probe and avoids
    /// committing to a GPU bf16-GEMV kernel before the dispatch
    /// shape is finalized. Phase 4m wires this on-device once the
    /// per-token routing decision drives a real grouped-expert
    /// dispatch.
    ///
    /// The MoE block also stores the router weight as f16 (after the
    /// loader's bf16→f16 upload), so we DtoH the f16 buffer directly.
    pub fn forward_layer3_router_probe(&self) -> Result<()> {
        let layer_idx = match self
            .arch
            .base
            .layer_types
            .iter()
            .position(|t| matches!(t, rvllm_loader::LayerAttnType::Full))
        {
            Some(i) => i,
            None => return Ok(()),
        };
        let moe = &self.model.layers[layer_idx].moe;
        let hidden = self.arch.base.hidden_size as usize;
        let num_experts = self.arch.num_experts;
        let top_k = self.arch.num_experts_per_tok;

        // The router weight is stored as f16 [num_experts, hidden]
        // (loader converts bf16 → f16 in `LoadCtx::upload_f16`).
        let weight_bytes = num_experts * hidden * 2;
        let mut router_f16 = vec![0u8; weight_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                router_f16.as_mut_ptr() as *mut _,
                moe.router.offset_bytes,
                weight_bytes,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_l3router DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Synthetic hidden state h[i] = (i % 8) * 0.125 — small, varied,
        // not constant so the per-expert dot products differentiate.
        let hidden_state: Vec<f32> = (0..hidden).map(|i| ((i % 8) as f32) * 0.125).collect();

        // Host matmul: logits[e] = Σ_k router[e, k] * hidden[k].
        let mut logits = vec![0.0f32; num_experts];
        for e in 0..num_experts {
            let row_off = e * hidden * 2;
            let mut acc = 0.0f32;
            for k in 0..hidden {
                let bits = u16::from_le_bytes([
                    router_f16[row_off + k * 2],
                    router_f16[row_off + k * 2 + 1],
                ]);
                acc += f16_bits_to_f32(bits) * hidden_state[k];
            }
            logits[e] = acc;
        }

        // Top-k selection: partial sort over (logit, expert_idx).
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(usize, f32)> = indexed.iter().take(top_k).copied().collect();

        // Softmax-normalize the top-k (Qwen's router head_mode).
        let max = top
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top.iter().map(|(_, v)| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let weights: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let pretty: Vec<String> = top
            .iter()
            .zip(weights.iter())
            .map(|((e, l), w)| format!("e{e}({l:.3}/{w:.3})"))
            .collect();
        eprintln!(
            "[qwen36] forward_layer3_router_probe: layer={layer_idx} \
             num_experts={num_experts} top_k={top_k} → \
             selected: {}",
            pretty.join(", ")
        );
        Ok(())
    }

    /// Phase 4m probe: full SwiGLU FFN for ONE routed expert.
    /// Chain: input f16 → expert0.gate_proj → expert0.up_proj →
    ///        host(silu(gate)*up) → expert0.down_proj → out f16.
    ///
    /// The host silu*mul step is a stand-in until a `silu_mul_f16`
    /// device kernel lands (would mirror Phase 4j's `sigmoid_mul_f16`
    /// — both fall in the same "tiny new f16 element-wise" CUDA cost).
    /// Validates the fused-experts arena layout: expert e's weight
    /// slice begins at `experts_*_proj_fused.offset_bytes +
    /// e * per_expert_fp8_bytes` and the blockscale slice at
    /// `blockscale_ptr + e * per_expert_blockscale_f32_bytes`. For
    /// e=0 both offsets are zero, which is the simplest-cut probe.
    pub fn forward_layer3_routed_expert_probe(&self) -> Result<()> {
        let layer_idx = match self
            .arch
            .base
            .layer_types
            .iter()
            .position(|t| matches!(t, rvllm_loader::LayerAttnType::Full))
        {
            Some(i) => i,
            None => return Ok(()),
        };
        let moe = &self.model.layers[layer_idx].moe;
        let kernel = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => return Ok(()),
        };

        // experts_gate_proj_fused.shape = [num_experts, N, K]
        // For Qwen 3.6: N = moe_intermediate_size = 512, K = hidden = 2048.
        let gate_w = &moe.experts_gate_proj_fused;
        let up_w = &moe.experts_up_proj_fused;
        let down_w = &moe.experts_down_proj_fused;
        let n_int = gate_w.shape[1] as u32; // 512
        let k_in = gate_w.shape[2] as u32; // 2048
        let n_down = down_w.shape[1] as u32; // 2048
        let k_down = down_w.shape[2] as u32; // 512
        let m: u32 = 1;

        // Synthetic f16 all-twos input over the hidden dim.
        let two_bits = 0x4000u16.to_le_bytes();
        let mut input_bytes = Vec::with_capacity((k_in as usize) * 2);
        for _ in 0..k_in {
            input_bytes.extend_from_slice(&two_bits);
        }
        let in_region = self
            .arena
            .region("qwen36_l3rex_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };

        // Outputs of gate / up are size N_int = 512 each (f16).
        let mid_bytes = (n_int as usize) * 2;
        let gate_out_region = self.arena.region("qwen36_l3rex_g", mid_bytes, 16)?;
        let up_out_region = self.arena.region("qwen36_l3rex_u", mid_bytes, 16)?;
        let silu_mul_region = self.arena.region("qwen36_l3rex_silu", mid_bytes, 16)?;
        let down_bytes = (n_down as usize) * 2;
        let down_out_region = self.arena.region("qwen36_l3rex_o", down_bytes, 16)?;

        // Expert 0's weight slice begins at the fused region start.
        // For e>0: weight_ptr += e * (N * K), blockscale_ptr +=
        //          e * (N/128 * K/128 * sizeof(f32)).
        let gate_blockscale = match gate_w.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let up_blockscale = match up_w.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let down_blockscale = match down_w.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };

        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m,
                n: n_int,
                k: k_in,
            }
            .launch(
                kernel,
                gate_out_region.device_ptr(),
                gate_w.offset_bytes,
                gate_blockscale,
                in_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m,
                n: n_int,
                k: k_in,
            }
            .launch(
                kernel,
                up_out_region.device_ptr(),
                up_w.offset_bytes,
                up_blockscale,
                in_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }
        self.stream.fence()?;

        // Host-side silu(gate) * up: DtoH both, compute, HtoD result.
        let mut gate_host = vec![0u8; mid_bytes];
        let mut up_host = vec![0u8; mid_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                gate_host.as_mut_ptr() as *mut _,
                gate_out_region.device_ptr(),
                mid_bytes,
            );
            let _ = cuMemcpyDtoH_v2(
                up_host.as_mut_ptr() as *mut _,
                up_out_region.device_ptr(),
                mid_bytes,
            );
        }
        let mut silu_mul_host = Vec::with_capacity(mid_bytes);
        for i in 0..(n_int as usize) {
            let g = f16_bits_to_f32(u16::from_le_bytes([gate_host[i * 2], gate_host[i * 2 + 1]]));
            let u = f16_bits_to_f32(u16::from_le_bytes([up_host[i * 2], up_host[i * 2 + 1]]));
            // SiLU: x · sigmoid(x) = x / (1 + exp(-x))
            let silu_g = g / (1.0f32 + (-g).exp());
            let v = silu_g * u;
            silu_mul_host.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
        }
        unsafe { silu_mul_region.copy_from_host(&silu_mul_host)? };

        // down_proj: mid [N=512] → out [hidden=2048]
        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m,
                n: n_down,
                k: k_down,
            }
            .launch(
                kernel,
                down_out_region.device_ptr(),
                down_w.offset_bytes,
                down_blockscale,
                silu_mul_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }
        self.stream.fence()?;

        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                down_out_region.device_ptr(),
                probe.len(),
            );
        }
        let o0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let o1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let o2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let o3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));

        // Log a sample of the silu*mul intermediate too, so we see
        // both stages of the SwiGLU FFN.
        let m0 = f16_bits_to_f32(u16::from_le_bytes([silu_mul_host[0], silu_mul_host[1]]));
        let m1 = f16_bits_to_f32(u16::from_le_bytes([silu_mul_host[2], silu_mul_host[3]]));
        let m2 = f16_bits_to_f32(u16::from_le_bytes([silu_mul_host[4], silu_mul_host[5]]));
        let m3 = f16_bits_to_f32(u16::from_le_bytes([silu_mul_host[6], silu_mul_host[7]]));

        eprintln!(
            "[qwen36] forward_layer3_routed_expert_probe: layer={layer_idx} \
             expert=0 gate/up=[{n_int}, {k_in}] down=[{n_down}, {k_down}] \
             silu_mul[0..4]=[{m0:.3}, {m1:.3}, {m2:.3}, {m3:.3}] \
             out[0..4]=[{o0:.3}, {o1:.3}, {o2:.3}, {o3:.3}] \
             (full SwiGLU FFN: gate → up → silu·mul (host) → down)"
        );
        Ok(())
    }

    /// Phase 4n: full MoE block for one token. Combines:
    ///   1. router → top-8 expert indices + softmax weights (CPU,
    ///      same as Phase 4l)
    ///   2. for each chosen expert: full SwiGLU FFN at the expert's
    ///      slice of the fused arena regions (extending Phase 4m to
    ///      non-zero expert offsets)
    ///   3. weighted sum of the 8 routed expert outputs (host f32)
    ///   4. shared expert: full SwiGLU FFN
    ///   5. shared_expert_gate sigmoid (single-element scalar) ·
    ///      shared output
    ///   6. final = routed_sum + gated_shared
    pub fn forward_layer3_full_moe_probe(&self) -> Result<()> {
        let layer_idx = match self
            .arch
            .base
            .layer_types
            .iter()
            .position(|t| matches!(t, rvllm_loader::LayerAttnType::Full))
        {
            Some(i) => i,
            None => return Ok(()),
        };
        let moe = &self.model.layers[layer_idx].moe;
        let kernel = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => return Ok(()),
        };
        let hidden = self.arch.base.hidden_size as usize;
        let n_int = moe.experts_gate_proj_fused.shape[1] as u32; // 512
        let k_in = moe.experts_gate_proj_fused.shape[2] as u32; // 2048
        let n_down = moe.experts_down_proj_fused.shape[1] as u32; // 2048
        let k_down = moe.experts_down_proj_fused.shape[2] as u32; // 512
        let num_experts = self.arch.num_experts;
        let top_k = self.arch.num_experts_per_tok;
        let m: u32 = 1;

        // ---- 1. router → top-8 ---------------------------------------
        let weight_bytes = num_experts * hidden * 2;
        let mut router_f16 = vec![0u8; weight_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                router_f16.as_mut_ptr() as *mut _,
                moe.router.offset_bytes,
                weight_bytes,
            );
        }
        let hidden_state: Vec<f32> = (0..hidden).map(|i| ((i % 8) as f32) * 0.125).collect();
        let mut logits = vec![0.0f32; num_experts];
        for e in 0..num_experts {
            let row = e * hidden * 2;
            let mut acc = 0.0f32;
            for k in 0..hidden {
                let bits =
                    u16::from_le_bytes([router_f16[row + k * 2], router_f16[row + k * 2 + 1]]);
                acc += f16_bits_to_f32(bits) * hidden_state[k];
            }
            logits[e] = acc;
        }
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(usize, f32)> = indexed.iter().take(top_k).copied().collect();
        let max = top
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top.iter().map(|(_, v)| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let weights: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // ---- 2-3. routed experts: FFN at each top-k offset, weighted sum
        // hidden_state f16 input region.
        let mut input_bytes = Vec::with_capacity(hidden * 2);
        for h in &hidden_state {
            input_bytes.extend_from_slice(&f32_to_f16_bits(*h).to_le_bytes());
        }
        let in_region = self
            .arena
            .region("qwen36_l3moe_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };

        let mid_bytes = (n_int as usize) * 2;
        let down_bytes = (n_down as usize) * 2;
        let gate_region = self.arena.region("qwen36_l3moe_g", mid_bytes, 16)?;
        let up_region = self.arena.region("qwen36_l3moe_u", mid_bytes, 16)?;
        let silu_region = self.arena.region("qwen36_l3moe_s", mid_bytes, 16)?;
        let down_region = self.arena.region("qwen36_l3moe_d", down_bytes, 16)?;

        // Per-expert byte strides into the fused regions.
        let int_per_expert_w = (n_int as u64) * (k_in as u64); // FP8: 1 byte/elem
        let int_per_expert_bs = ((n_int as u64) / 128) * ((k_in as u64) / 128) * 4; // f32 blockscale
        let down_per_expert_w = (n_down as u64) * (k_down as u64);
        let down_per_expert_bs = ((n_down as u64) / 128) * ((k_down as u64) / 128) * 4;

        let gate_bs = moe.experts_gate_proj_fused.blockscale_ptr.unwrap_or(0);
        let up_bs = moe.experts_up_proj_fused.blockscale_ptr.unwrap_or(0);
        let down_bs = moe.experts_down_proj_fused.blockscale_ptr.unwrap_or(0);
        if gate_bs == 0 || up_bs == 0 || down_bs == 0 {
            return Ok(());
        }

        let mut routed_sum = vec![0.0f32; n_down as usize];

        for ((e_idx, _logit), w) in top.iter().zip(weights.iter()) {
            let e = *e_idx as u64;
            // gate
            unsafe {
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m,
                    n: n_int,
                    k: k_in,
                }
                .launch(
                    kernel,
                    gate_region.device_ptr(),
                    moe.experts_gate_proj_fused.offset_bytes + e * int_per_expert_w,
                    gate_bs + e * int_per_expert_bs,
                    in_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m,
                    n: n_int,
                    k: k_in,
                }
                .launch(
                    kernel,
                    up_region.device_ptr(),
                    moe.experts_up_proj_fused.offset_bytes + e * int_per_expert_w,
                    up_bs + e * int_per_expert_bs,
                    in_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
            }
            self.stream.fence()?;
            // host silu·mul
            let mut g_host = vec![0u8; mid_bytes];
            let mut u_host = vec![0u8; mid_bytes];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    g_host.as_mut_ptr() as *mut _,
                    gate_region.device_ptr(),
                    mid_bytes,
                );
                let _ = cuMemcpyDtoH_v2(
                    u_host.as_mut_ptr() as *mut _,
                    up_region.device_ptr(),
                    mid_bytes,
                );
            }
            let mut silu_host = Vec::with_capacity(mid_bytes);
            for i in 0..(n_int as usize) {
                let g = f16_bits_to_f32(u16::from_le_bytes([g_host[i * 2], g_host[i * 2 + 1]]));
                let u = f16_bits_to_f32(u16::from_le_bytes([u_host[i * 2], u_host[i * 2 + 1]]));
                let s = g / (1.0f32 + (-g).exp());
                silu_host.extend_from_slice(&f32_to_f16_bits(s * u).to_le_bytes());
            }
            unsafe { silu_region.copy_from_host(&silu_host)? };

            unsafe {
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m,
                    n: n_down,
                    k: k_down,
                }
                .launch(
                    kernel,
                    down_region.device_ptr(),
                    moe.experts_down_proj_fused.offset_bytes + e * down_per_expert_w,
                    down_bs + e * down_per_expert_bs,
                    silu_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
            }
            self.stream.fence()?;

            let mut d_host = vec![0u8; down_bytes];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    d_host.as_mut_ptr() as *mut _,
                    down_region.device_ptr(),
                    down_bytes,
                );
            }
            for i in 0..(n_down as usize) {
                let v = f16_bits_to_f32(u16::from_le_bytes([d_host[i * 2], d_host[i * 2 + 1]]));
                routed_sum[i] += v * w;
            }
        }

        // ---- 4. shared expert: full FFN ----
        let sh_gate_bs = moe.shared_expert_gate_proj.blockscale_ptr.unwrap_or(0);
        let sh_up_bs = moe.shared_expert_up_proj.blockscale_ptr.unwrap_or(0);
        let sh_down_bs = moe.shared_expert_down_proj.blockscale_ptr.unwrap_or(0);
        if sh_gate_bs != 0 && sh_up_bs != 0 && sh_down_bs != 0 {
            unsafe {
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m,
                    n: n_int,
                    k: k_in,
                }
                .launch(
                    kernel,
                    gate_region.device_ptr(),
                    moe.shared_expert_gate_proj.offset_bytes,
                    sh_gate_bs,
                    in_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m,
                    n: n_int,
                    k: k_in,
                }
                .launch(
                    kernel,
                    up_region.device_ptr(),
                    moe.shared_expert_up_proj.offset_bytes,
                    sh_up_bs,
                    in_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
            }
            self.stream.fence()?;
            let mut g_host = vec![0u8; mid_bytes];
            let mut u_host = vec![0u8; mid_bytes];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    g_host.as_mut_ptr() as *mut _,
                    gate_region.device_ptr(),
                    mid_bytes,
                );
                let _ = cuMemcpyDtoH_v2(
                    u_host.as_mut_ptr() as *mut _,
                    up_region.device_ptr(),
                    mid_bytes,
                );
            }
            let mut silu_host = Vec::with_capacity(mid_bytes);
            for i in 0..(n_int as usize) {
                let g = f16_bits_to_f32(u16::from_le_bytes([g_host[i * 2], g_host[i * 2 + 1]]));
                let u = f16_bits_to_f32(u16::from_le_bytes([u_host[i * 2], u_host[i * 2 + 1]]));
                let s = g / (1.0f32 + (-g).exp());
                silu_host.extend_from_slice(&f32_to_f16_bits(s * u).to_le_bytes());
            }
            unsafe { silu_region.copy_from_host(&silu_host)? };
            unsafe {
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m,
                    n: n_down,
                    k: k_down,
                }
                .launch(
                    kernel,
                    down_region.device_ptr(),
                    moe.shared_expert_down_proj.offset_bytes,
                    sh_down_bs,
                    silu_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
            }
            self.stream.fence()?;
            let mut sh_host = vec![0u8; down_bytes];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    sh_host.as_mut_ptr() as *mut _,
                    down_region.device_ptr(),
                    down_bytes,
                );
            }
            // shared_expert_gate: single-element bf16-as-f16 → sigmoid
            let mut sh_gate_host = [0u8; 2];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    sh_gate_host.as_mut_ptr() as *mut _,
                    moe.shared_expert_gate_logit.offset_bytes,
                    2,
                );
            }
            let g_logit = f16_bits_to_f32(u16::from_le_bytes(sh_gate_host));
            let g_sigmoid = 1.0f32 / (1.0f32 + (-g_logit).exp());
            for i in 0..(n_down as usize) {
                let v = f16_bits_to_f32(u16::from_le_bytes([sh_host[i * 2], sh_host[i * 2 + 1]]));
                routed_sum[i] += v * g_sigmoid;
            }
        }

        eprintln!(
            "[qwen36] forward_layer3_full_moe_probe: layer={layer_idx} \
             top_k={top_k}/{num_experts} → final[0..4]=[{:.3}, {:.3}, \
             {:.3}, {:.3}] (routed-weighted-sum + sigmoid·shared)",
            routed_sum[0], routed_sum[1], routed_sum[2], routed_sum[3],
        );
        Ok(())
    }

    /// Phase 4s probe: chain all linear-attn kernels end-to-end for
    /// layer 0, single timestep. Order:
    ///   1. in_proj_qkv f16 GEMV → conv_input [8192 = 4×2048]
    ///   2. causal_conv1d_f16 (state-cache padded with zeros — no
    ///      history yet, single-step)
    ///   3. host SiLU + split into Q/K/V/extra streams [4 × 2048]
    ///      (qwen3-next exact split is q=k=v=2048 + dt_input=2048
    ///      per the reference; this probe uses that convention)
    ///   4. host: dt = silu(conv_dt) + dt_bias, alpha = exp(-exp(A_log)·dt),
    ///      beta = dt (Mamba-2-style decay+write derivation)
    ///   5. gated_delta_state_update_f16 (state init = 0)
    ///   6. host: read-out per head (Q · state → 32×128 = 4096 dim)
    ///   7. in_proj_z FP8 GEMV → z_logits [4096], host sigmoid · readout
    ///   8. out_proj FP8 GEMV → final delta [hidden=2048]
    ///
    /// Synthetic input. Math approximates qwen3-next without claiming
    /// numerical match against the reference — that's Phase 4u where
    /// a single-token comparison against vLLM nails down any
    /// off-by-one in the dt/alpha/beta computation, the QKV split
    /// order, or the post-SSM normalisation. Goal here: prove every
    /// kernel + host glue step composes without ABI errors.
    pub fn forward_layer0_linear_chain_probe(&self) -> Result<()> {
        let layer_idx = match self
            .arch
            .base
            .layer_types
            .iter()
            .position(|t| matches!(t, rvllm_loader::LayerAttnType::Linear))
        {
            Some(i) => i,
            None => return Ok(()),
        };
        let la = match &self.model.layers[layer_idx].attn {
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(l) => l,
            _ => return Ok(()),
        };
        let kernel_gemv = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => return Ok(()),
        };
        let hidden = self.arch.base.hidden_size as u32;
        let qkv_n = la.in_proj_qkv.shape[0] as u32; // 8192
        let z_n = la.in_proj_z.shape[0] as u32; // 4096
        let out_n = la.out_proj.shape[0] as u32; // 2048
        let out_k = la.out_proj.shape[1] as u32; // 4096
        let m: u32 = 1;
        let num_heads: u32 = 32;
        let d_state: u32 = 128;
        let head_split: u32 = qkv_n / 4; // 2048 per stream

        // Synthetic hidden input: small varied values.
        let mut input_bytes = Vec::with_capacity((hidden as usize) * 2);
        for i in 0..hidden as usize {
            let v = ((i % 16) as f32) * 0.0625 - 0.5;
            input_bytes.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
        }
        let in_region = self.arena.region("qwen36_l0lc_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };

        // 1. in_proj_qkv FP8 GEMV → qkv_concat [8192]
        let qkv_bytes = (qkv_n as usize) * 2;
        let qkv_region = self.arena.region("qwen36_l0lc_qkv", qkv_bytes, 16)?;
        let qkv_bs = la.in_proj_qkv.blockscale_ptr.unwrap_or(0);
        if qkv_bs == 0 {
            return Ok(());
        }
        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m,
                n: qkv_n,
                k: hidden,
            }
            .launch(
                kernel_gemv,
                qkv_region.device_ptr(),
                la.in_proj_qkv.offset_bytes,
                qkv_bs,
                in_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }
        self.stream.fence()?;

        // 2. causal_conv1d (state padded with zeros for single-step).
        // conv1d expects [seq+ks-1, channels]. We pad with 3 zero
        // timesteps + 1 real timestep = ks=4 input window.
        let ks: u32 = 4;
        let conv_in_elems = ((1 + ks - 1) as usize) * (qkv_n as usize);
        let conv_in_bytes = conv_in_elems * 2;
        let conv_in_region = self.arena.region("qwen36_l0lc_cin", conv_in_bytes, 16)?;
        let conv_out_region = self.arena.region("qwen36_l0lc_cout", qkv_bytes, 16)?;
        // Build conv input on host: 3 zero timesteps + qkv_concat.
        let mut conv_in_host = vec![0u8; conv_in_bytes];
        let mut qkv_host = vec![0u8; qkv_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                qkv_host.as_mut_ptr() as *mut _,
                qkv_region.device_ptr(),
                qkv_bytes,
            );
        }
        // Place qkv_host at last (ks-1=3) timestep position. Earlier
        // positions stay zero.
        let last_off = ((ks - 1) as usize) * (qkv_n as usize) * 2;
        conv_in_host[last_off..last_off + qkv_bytes].copy_from_slice(&qkv_host);
        unsafe { conv_in_region.copy_from_host(&conv_in_host)? };

        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut output = conv_out_region.device_ptr();
            let mut input = conv_in_region.device_ptr();
            let mut weight = la.conv1d.offset_bytes;
            let mut sl: i32 = 1;
            let mut ch = qkv_n as i32;
            let mut k_arg = ks as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut weight) as *mut u64 as *mut core::ffi::c_void,
                (&mut sl) as *mut i32 as *mut core::ffi::c_void,
                (&mut ch) as *mut i32 as *mut core::ffi::c_void,
                (&mut k_arg) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid_x = (qkv_n + block - 1) / block;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_causal_conv1d_f16.raw() as CUfunction,
                grid_x,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 causal_conv1d_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;

        // 3. Host: read conv_out, SiLU, split into Q/K/V/dt streams.
        let mut conv_out_host = vec![0u8; qkv_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                conv_out_host.as_mut_ptr() as *mut _,
                conv_out_region.device_ptr(),
                qkv_bytes,
            );
        }
        let mut conv_silu_f32 = Vec::with_capacity(qkv_n as usize);
        for i in 0..qkv_n as usize {
            let bits = u16::from_le_bytes([conv_out_host[i * 2], conv_out_host[i * 2 + 1]]);
            let v = f16_bits_to_f32(bits);
            conv_silu_f32.push(v / (1.0f32 + (-v).exp()));
        }
        let q_off = 0usize;
        let k_off = head_split as usize;
        let v_off = (2 * head_split) as usize;
        let dt_off = (3 * head_split) as usize;
        let split_per = head_split as usize;

        // 4. Host: alpha/beta from A_log + dt_bias + dt_input.
        // dt_input is the 4th split = `conv_silu[dt_off..dt_off+split_per]`.
        // For a smoke probe we average dt_input per head into a scalar.
        let mut a_log_host = vec![0u8; (num_heads as usize) * 2];
        let mut dt_bias_host = vec![0u8; (num_heads as usize) * 2];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                a_log_host.as_mut_ptr() as *mut _,
                la.a_log.offset_bytes,
                a_log_host.len(),
            );
            let _ = cuMemcpyDtoH_v2(
                dt_bias_host.as_mut_ptr() as *mut _,
                la.dt_bias.offset_bytes,
                dt_bias_host.len(),
            );
        }
        let per_head = (split_per / num_heads as usize) as usize;
        let mut alpha_f32 = Vec::with_capacity(num_heads as usize);
        let mut beta_f32 = Vec::with_capacity(num_heads as usize);
        for h in 0..num_heads as usize {
            // dt = softplus(dt_input_avg + dt_bias[h])
            let mut dt_avg = 0.0f32;
            for j in 0..per_head {
                dt_avg += conv_silu_f32[dt_off + h * per_head + j];
            }
            dt_avg /= per_head as f32;
            let bias = f16_bits_to_f32(u16::from_le_bytes([
                dt_bias_host[h * 2],
                dt_bias_host[h * 2 + 1],
            ]));
            let dt = (1.0f32 + (dt_avg + bias).exp()).ln(); // softplus
            let a_log = f16_bits_to_f32(u16::from_le_bytes([
                a_log_host[h * 2],
                a_log_host[h * 2 + 1],
            ]));
            let a = (-(a_log.exp()) * dt).exp();
            alpha_f32.push(a);
            beta_f32.push(dt);
        }

        // 5. Pack Q/K/V f16 [num_heads, d_state=128] from the splits.
        // (per_head should equal d_state=128 for qwen3-next 35B-A3B.)
        let qkv_per_head_bytes = (num_heads as usize) * (d_state as usize) * 2;
        let mut q_host = Vec::with_capacity(qkv_per_head_bytes);
        let mut k_host = Vec::with_capacity(qkv_per_head_bytes);
        let mut v_host = Vec::with_capacity(qkv_per_head_bytes);
        for h in 0..num_heads as usize {
            for d in 0..d_state as usize {
                let q = conv_silu_f32[q_off + h * per_head + d];
                let k = conv_silu_f32[k_off + h * per_head + d];
                let v = conv_silu_f32[v_off + h * per_head + d];
                q_host.extend_from_slice(&f32_to_f16_bits(q).to_le_bytes());
                k_host.extend_from_slice(&f32_to_f16_bits(k).to_le_bytes());
                v_host.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
        }

        // 6. ssm state update.
        let state_bytes = (num_heads as usize) * (d_state as usize) * (d_state as usize) * 2;
        let state_region = self.arena.region("qwen36_l0lc_state", state_bytes, 16)?;
        let q_region = self.arena.region("qwen36_l0lc_q", q_host.len(), 16)?;
        let k_region = self.arena.region("qwen36_l0lc_k", k_host.len(), 16)?;
        let v_region = self.arena.region("qwen36_l0lc_v", v_host.len(), 16)?;
        let alpha_bytes: Vec<u8> = alpha_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        let beta_bytes: Vec<u8> = beta_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        let alpha_region = self
            .arena
            .region("qwen36_l0lc_alpha", alpha_bytes.len(), 16)?;
        let beta_region = self
            .arena
            .region("qwen36_l0lc_beta", beta_bytes.len(), 16)?;
        let zero_state = vec![0u8; state_bytes];
        unsafe {
            state_region.copy_from_host(&zero_state)?;
            q_region.copy_from_host(&q_host)?;
            k_region.copy_from_host(&k_host)?;
            v_region.copy_from_host(&v_host)?;
            alpha_region.copy_from_host(&alpha_bytes)?;
            beta_region.copy_from_host(&beta_bytes)?;
        }
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut state = state_region.device_ptr();
            let mut k = k_region.device_ptr();
            let mut v = v_region.device_ptr();
            let mut alpha = alpha_region.device_ptr();
            let mut beta = beta_region.device_ptr();
            let mut nh = num_heads as i32;
            let mut dk = d_state as i32;
            let mut dv = d_state as i32;
            let args = [
                (&mut state) as *mut u64 as *mut core::ffi::c_void,
                (&mut k) as *mut u64 as *mut core::ffi::c_void,
                (&mut v) as *mut u64 as *mut core::ffi::c_void,
                (&mut alpha) as *mut u64 as *mut core::ffi::c_void,
                (&mut beta) as *mut u64 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut dk) as *mut i32 as *mut core::ffi::c_void,
                (&mut dv) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_gated_delta_state_update_f16.raw() as CUfunction,
                num_heads,
                1,
                1,
                16,
                16,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 gated_delta_state_update_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;

        // 7. Host: read state, compute Q · S → readout [num_heads, d_state].
        let mut state_host = vec![0u8; state_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                state_host.as_mut_ptr() as *mut _,
                state_region.device_ptr(),
                state_bytes,
            );
        }
        let mut readout_f32 = vec![0.0f32; (num_heads as usize) * (d_state as usize)];
        for h in 0..num_heads as usize {
            for v in 0..d_state as usize {
                let mut acc = 0.0f32;
                for k in 0..d_state as usize {
                    let s_idx =
                        h * (d_state as usize) * (d_state as usize) + v * (d_state as usize) + k;
                    let s = f16_bits_to_f32(u16::from_le_bytes([
                        state_host[s_idx * 2],
                        state_host[s_idx * 2 + 1],
                    ]));
                    let q_idx = h * (d_state as usize) + k;
                    let q = f16_bits_to_f32(u16::from_le_bytes([
                        q_host[q_idx * 2],
                        q_host[q_idx * 2 + 1],
                    ]));
                    acc += s * q;
                }
                readout_f32[h * (d_state as usize) + v] = acc;
            }
        }

        // 8. in_proj_z FP8 GEMV → z_logits, host sigmoid · readout,
        //    then host: per-head norm with linear_attn.norm.weight,
        //    then out_proj FP8 GEMV.
        let z_bytes_dev = (z_n as usize) * 2;
        let z_region = self.arena.region("qwen36_l0lc_z", z_bytes_dev, 16)?;
        let z_bs = la.in_proj_z.blockscale_ptr.unwrap_or(0);
        if z_bs == 0 {
            return Ok(());
        }
        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m,
                n: z_n,
                k: hidden,
            }
            .launch(
                kernel_gemv,
                z_region.device_ptr(),
                la.in_proj_z.offset_bytes,
                z_bs,
                in_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }
        self.stream.fence()?;
        let mut z_host_bytes = vec![0u8; z_bytes_dev];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                z_host_bytes.as_mut_ptr() as *mut _,
                z_region.device_ptr(),
                z_bytes_dev,
            );
        }
        // Per-head RMSNorm on the readout (gamma = norm.weight [128]).
        let mut norm_gamma = vec![0u8; (d_state as usize) * 2];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                norm_gamma.as_mut_ptr() as *mut _,
                la.norm.offset_bytes,
                norm_gamma.len(),
            );
        }
        let mut gated_readout = vec![0u8; (num_heads as usize) * (d_state as usize) * 2];
        for h in 0..num_heads as usize {
            let mut sumsq = 0.0f32;
            for d in 0..d_state as usize {
                let v = readout_f32[h * (d_state as usize) + d];
                sumsq += v * v;
            }
            let rms = (sumsq / d_state as f32 + 1e-6).sqrt();
            for d in 0..d_state as usize {
                let v = readout_f32[h * (d_state as usize) + d] / rms;
                let g = f16_bits_to_f32(u16::from_le_bytes([
                    norm_gamma[d * 2],
                    norm_gamma[d * 2 + 1],
                ]));
                let z_logit = f16_bits_to_f32(u16::from_le_bytes([
                    z_host_bytes[(h * (d_state as usize) + d) * 2],
                    z_host_bytes[(h * (d_state as usize) + d) * 2 + 1],
                ]));
                let sigmoid_z = 1.0f32 / (1.0f32 + (-z_logit).exp());
                let out = v * g * sigmoid_z;
                let bytes = f32_to_f16_bits(out).to_le_bytes();
                gated_readout[(h * (d_state as usize) + d) * 2] = bytes[0];
                gated_readout[(h * (d_state as usize) + d) * 2 + 1] = bytes[1];
            }
        }
        let gated_region = self
            .arena
            .region("qwen36_l0lc_gated", gated_readout.len(), 16)?;
        unsafe { gated_region.copy_from_host(&gated_readout)? };

        // out_proj: [hidden, num_heads*d_state=4096] FP8 GEMV.
        let out_bytes = (out_n as usize) * 2;
        let out_region = self.arena.region("qwen36_l0lc_out", out_bytes, 16)?;
        let out_bs = la.out_proj.blockscale_ptr.unwrap_or(0);
        if out_bs == 0 {
            return Ok(());
        }
        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m,
                n: out_n,
                k: out_k,
            }
            .launch(
                kernel_gemv,
                out_region.device_ptr(),
                la.out_proj.offset_bytes,
                out_bs,
                gated_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }
        self.stream.fence()?;
        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                out_region.device_ptr(),
                probe.len(),
            );
        }
        let o0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let o1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let o2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let o3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_layer0_linear_chain_probe: layer={layer_idx} \
             chain in_proj_qkv → conv1d → silu+split → \
             {{α,β from A_log+dt_bias+dt}} → ssm_state_update → \
             Q·S readout → norm + sigmoid·z → out_proj → \
             out[0..4]=[{o0:.4}, {o1:.4}, {o2:.4}, {o3:.4}] \
             (full linear-attn block, single-step, state init=0)"
        );
        Ok(())
    }

    /// Phase 4u probe: verify the paged KV cache is zero-initialised
    /// + per-layer addressable + reset round-trips correctly.
    pub fn kv_cache_probe(&self) -> Result<()> {
        let n_full = self.kv_cache_bytes / self.kv_cache_layer_bytes;
        let l0_ptr = self.kv_cache_layer_ptr(0);
        let last_ptr = self.kv_cache_layer_ptr((n_full - 1) as u32);
        if l0_ptr == 0 || last_ptr == 0 {
            return Ok(());
        }
        let mut probe = [0xFFu8; 16];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(probe.as_mut_ptr() as *mut _, l0_ptr, 8);
            let _ = cuMemcpyDtoH_v2(probe.as_mut_ptr().add(8) as *mut _, last_ptr, 8);
        }
        let l0_zero = probe[0..8].iter().all(|b| *b == 0);
        let l_n_zero = probe[8..16].iter().all(|b| *b == 0);
        eprintln!(
            "[qwen36] kv_cache_probe: {n_full} full-attn slots, \
             layer0 head8b zeroed={l0_zero}, last_layer head8b zeroed={l_n_zero}, \
             total={:.1} MiB persistent",
            self.kv_cache_bytes as f64 / (1024.0 * 1024.0),
        );
        Ok(())
    }

    /// Phase 4t probe: verify the persistent linear-attn state cache
    /// is zero-initialised + accessible. After bring-up the cache
    /// must hold all zeros at every layer's slice.
    pub fn linear_state_cache_probe(&self) -> Result<()> {
        let mut probe = [0xFFu8; 16];
        // Sample layer 0 + last linear layer's offsets.
        let n_linear_layers = self.linear_state_bytes / self.linear_state_layer_bytes;
        let last_layer = (n_linear_layers - 1) as u32;
        let l0_ptr = self.linear_state_layer_ptr(0);
        let ln_ptr = self.linear_state_layer_ptr(last_layer);
        if l0_ptr == 0 || ln_ptr == 0 {
            eprintln!("[qwen36] linear_state_cache_probe: pointers invalid, skipping");
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(probe.as_mut_ptr() as *mut _, l0_ptr, 8);
            let _ = cuMemcpyDtoH_v2(probe.as_mut_ptr().add(8) as *mut _, ln_ptr, 8);
        }
        let l0_zero = probe[0..8].iter().all(|b| *b == 0);
        let ln_zero = probe[8..16].iter().all(|b| *b == 0);
        eprintln!(
            "[qwen36] linear_state_cache_probe: {n_linear_layers} layer slots, \
             layer0 head8b zeroed={l0_zero}, layer{last_layer} head8b zeroed={ln_zero}, \
             total={:.1} MiB persistent (above scratch checkpoint)",
            self.linear_state_bytes as f64 / (1024.0 * 1024.0),
        );
        // Reset round-trip: dirty layer 0, reset, re-read.
        let dirty: u8 = 0xAA;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemsetD8_v2(l0_ptr, dirty, 8);
        }
        self.reset_linear_state()?;
        self.stream.fence()?;
        let mut after = [0xFFu8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(after.as_mut_ptr() as *mut _, l0_ptr, 8);
        }
        let reset_ok = after.iter().all(|b| *b == 0);
        eprintln!(
            "[qwen36] linear_state_cache_probe: reset round-trip — \
             dirty(0xAA) → reset_linear_state() → zeroed: {reset_ok}"
        );
        Ok(())
    }

    /// Phase 4r probe: launch the Gated-DeltaNet state-update kernel
    /// against an all-zero starting state, K=V=1.0, alpha=0.5,
    /// beta=2.0. After one step the state should equal `beta * 1.0 *
    /// 1.0 = 2.0` everywhere (decay term contributes 0 because the
    /// initial state is zero). Validates ABI + per-head broadcast +
    /// f16/f32 accumulator math.
    pub fn forward_layer0_ssm_state_probe(&self) -> Result<()> {
        // qwen3-next per-head dims: head_count=32 (A_log shape),
        // d_state = 128 (linear_attn.norm.weight shape).
        let num_heads: u32 = 32;
        let d_k: u32 = 128;
        let d_v: u32 = 128;
        let state_elems = (num_heads as usize) * (d_v as usize) * (d_k as usize);
        let state_bytes = state_elems * 2; // f16

        // K = V = 1.0 (f16 0x3c00); state starts at 0.
        let one_bits = 0x3c00u16.to_le_bytes();
        let zero_bytes = vec![0u8; state_bytes];
        let mut k_bytes = Vec::with_capacity((num_heads as usize) * (d_k as usize) * 2);
        let mut v_bytes = Vec::with_capacity((num_heads as usize) * (d_v as usize) * 2);
        for _ in 0..(num_heads as usize) * (d_k as usize) {
            k_bytes.extend_from_slice(&one_bits);
        }
        for _ in 0..(num_heads as usize) * (d_v as usize) {
            v_bytes.extend_from_slice(&one_bits);
        }
        let alpha_host: Vec<u8> = (0..num_heads).flat_map(|_| 0.5f32.to_le_bytes()).collect();
        let beta_host: Vec<u8> = (0..num_heads).flat_map(|_| 2.0f32.to_le_bytes()).collect();

        let state_region = self.arena.region("qwen36_l0ssm_s", state_bytes, 16)?;
        let k_region = self.arena.region("qwen36_l0ssm_k", k_bytes.len(), 16)?;
        let v_region = self.arena.region("qwen36_l0ssm_v", v_bytes.len(), 16)?;
        let alpha_region = self.arena.region("qwen36_l0ssm_a", alpha_host.len(), 16)?;
        let beta_region = self.arena.region("qwen36_l0ssm_b", beta_host.len(), 16)?;
        unsafe {
            state_region.copy_from_host(&zero_bytes)?;
            k_region.copy_from_host(&k_bytes)?;
            v_region.copy_from_host(&v_bytes)?;
            alpha_region.copy_from_host(&alpha_host)?;
            beta_region.copy_from_host(&beta_host)?;
        }

        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut state = state_region.device_ptr();
            let mut k = k_region.device_ptr();
            let mut v = v_region.device_ptr();
            let mut alpha = alpha_region.device_ptr();
            let mut beta = beta_region.device_ptr();
            let mut nh = num_heads as i32;
            let mut dk = d_k as i32;
            let mut dv = d_v as i32;
            let args = [
                (&mut state) as *mut u64 as *mut core::ffi::c_void,
                (&mut k) as *mut u64 as *mut core::ffi::c_void,
                (&mut v) as *mut u64 as *mut core::ffi::c_void,
                (&mut alpha) as *mut u64 as *mut core::ffi::c_void,
                (&mut beta) as *mut u64 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut dk) as *mut i32 as *mut core::ffi::c_void,
                (&mut dv) as *mut i32 as *mut core::ffi::c_void,
            ];
            // 16x16 = 256 threads per head; each thread strides over
            // its share of the 128*128=16384 state elements.
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_gated_delta_state_update_f16.raw() as CUfunction,
                num_heads,
                1,
                1,
                16,
                16,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_l0ssm gated_delta_state_update_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;

        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                state_region.device_ptr(),
                probe.len(),
            );
        }
        let s0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let s1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let s2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let s3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_layer0_ssm_state_probe: heads={num_heads} \
             d_k={d_k} d_v={d_v} alpha=0.5 beta=2.0 K=V=1.0 init=0 → \
             state[head0,0..4]=[{s0:.3}, {s1:.3}, {s2:.3}, {s3:.3}] \
             (expected ≈ 2.0 — beta·K·V = 2·1·1 = 2; decay·init = 0)"
        );
        Ok(())
    }

    /// Phase 4q probe: launch the new `causal_conv1d_f16_kernel`
    /// against linear-attn layer 0's `conv1d.weight [8192, 1, 4]`.
    /// Synthetic input chosen so the expected output is closed-form:
    /// input is all-ones in every channel, so out[t, c] = Σ_k w[c, 0, k]
    /// — i.e. each output equals the sum of that channel's 4 conv1d
    /// weight values. Probes the per-channel shape + ABI + first
    /// channel's actual weight sum.
    pub fn forward_layer0_conv1d_probe(&self) -> Result<()> {
        let layer_idx = match self
            .arch
            .base
            .layer_types
            .iter()
            .position(|t| matches!(t, rvllm_loader::LayerAttnType::Linear))
        {
            Some(i) => i,
            None => return Ok(()),
        };
        let la = match &self.model.layers[layer_idx].attn {
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(l) => l,
            _ => return Ok(()),
        };
        // conv1d.shape = [channels, 1, ks] = [8192, 1, 4]
        let channels = la.conv1d.shape[0] as u32;
        let ks = la.conv1d.shape[2] as u32;
        let seq_len: u32 = 1;

        // Input: (seq_len + ks - 1, channels) all-ones → each output
        // channel = sum of its `ks` weight values.
        let one_bits = 0x3c00u16.to_le_bytes();
        let in_elems = ((seq_len + ks - 1) as usize) * (channels as usize);
        let mut input_bytes = Vec::with_capacity(in_elems * 2);
        for _ in 0..in_elems {
            input_bytes.extend_from_slice(&one_bits);
        }
        let in_region = self.arena.region("qwen36_l0c1_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };

        let out_bytes = (seq_len as usize) * (channels as usize) * 2;
        let out_region = self.arena.region("qwen36_l0c1_out", out_bytes, 16)?;

        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut output = out_region.device_ptr();
            let mut input = in_region.device_ptr();
            let mut weight = la.conv1d.offset_bytes;
            let mut sl = seq_len as i32;
            let mut ch = channels as i32;
            let mut k = ks as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut weight) as *mut u64 as *mut core::ffi::c_void,
                (&mut sl) as *mut i32 as *mut core::ffi::c_void,
                (&mut ch) as *mut i32 as *mut core::ffi::c_void,
                (&mut k) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid_x = (channels + block - 1) / block;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_causal_conv1d_f16.raw() as CUfunction,
                grid_x,
                seq_len,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_l0c1 causal_conv1d_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;

        // DtoH first 4 channels' output + the corresponding 4 weight
        // values for cross-check.
        let mut out_probe = [0u8; 8];
        let mut w_probe = [0u8; (4 * 4) * 2]; // 4 channels × 4 ks × 2 bytes
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                out_probe.as_mut_ptr() as *mut _,
                out_region.device_ptr(),
                out_probe.len(),
            );
            let _ = cuMemcpyDtoH_v2(
                w_probe.as_mut_ptr() as *mut _,
                la.conv1d.offset_bytes,
                w_probe.len(),
            );
        }
        let o: [f32; 4] = [
            f16_bits_to_f32(u16::from_le_bytes([out_probe[0], out_probe[1]])),
            f16_bits_to_f32(u16::from_le_bytes([out_probe[2], out_probe[3]])),
            f16_bits_to_f32(u16::from_le_bytes([out_probe[4], out_probe[5]])),
            f16_bits_to_f32(u16::from_le_bytes([out_probe[6], out_probe[7]])),
        ];
        // Expected: out[c] = Σ_k w[c, 0, k] for c=0..3.
        let mut expected = [0.0f32; 4];
        for c in 0..4 {
            let mut s = 0.0f32;
            for k in 0..4 {
                let off = (c * 4 + k) * 2;
                s += f16_bits_to_f32(u16::from_le_bytes([w_probe[off], w_probe[off + 1]]));
            }
            expected[c] = s;
        }
        eprintln!(
            "[qwen36] forward_layer0_conv1d_probe: layer={layer_idx} \
             channels={channels} ks={ks} seq_len={seq_len} \
             out[0..4]={o:?} expected={expected:?}"
        );
        Ok(())
    }

    /// Phase 4p probe: launch the two FP8 input projections of the
    /// linear-attn block (in_proj_qkv + in_proj_z) against the first
    /// linear-attention layer's weights. The blockwise FP8 GEMV
    /// kernel (Phase 4c onwards) is model-agnostic — this probe
    /// confirms it works for the linear-attn projection shapes
    /// `[8192, hidden]` (qkv: 4 × 2048 = q/k/v/extra concat) and
    /// `[4096, hidden]` (z gating stream) which differ from the
    /// full-attention layer's projections.
    ///
    /// The recurrent ssm-scan + conv1d + per-sequence state cache
    /// still need a custom kernel — this only exercises the FP8
    /// matmul entry points to the linear-attn block.
    pub fn forward_layer0_linear_in_proj_probe(&self) -> Result<()> {
        let layer_idx = match self
            .arch
            .base
            .layer_types
            .iter()
            .position(|t| matches!(t, rvllm_loader::LayerAttnType::Linear))
        {
            Some(i) => i,
            None => return Ok(()),
        };
        let la = match &self.model.layers[layer_idx].attn {
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(l) => l,
            _ => return Ok(()),
        };
        let kernel = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => return Ok(()),
        };
        let hidden = self.arch.base.hidden_size as u32;
        let m: u32 = 1;

        // Synthetic f16 all-twos input, hidden-sized.
        let two_bits = 0x4000u16.to_le_bytes();
        let mut input_bytes = Vec::with_capacity((hidden as usize) * 2);
        for _ in 0..hidden {
            input_bytes.extend_from_slice(&two_bits);
        }
        let in_region = self
            .arena
            .region("qwen36_l0lin_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };

        // Closure to launch one FP8 projection role.
        let project =
            |w: &rvllm_loader::weights::Fp8Weight, region_name: &'static str| -> Result<[f32; 4]> {
                let n = w.shape[0] as u32;
                let k = w.shape[1] as u32;
                let bs = match w.blockscale_ptr {
                    Some(p) => p,
                    None => {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36_l0lin missing blockscale",
                            rvllm_core::CudaErrorKind::Other,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                };
                let out_bytes = (m as usize) * (n as usize) * 2;
                let out_region = self.arena.region(region_name, out_bytes, 16)?;
                unsafe {
                    rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m, n, k }.launch(
                        kernel,
                        out_region.device_ptr(),
                        w.offset_bytes,
                        bs,
                        in_region.device_ptr(),
                        self.stream.raw() as u64,
                    )?;
                }
                self.stream.fence()?;
                let mut probe = [0u8; 8];
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(
                        probe.as_mut_ptr() as *mut _,
                        out_region.device_ptr(),
                        probe.len(),
                    );
                }
                Ok([
                    f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]])),
                    f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]])),
                    f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]])),
                    f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]])),
                ])
            };

        let qkv = project(&la.in_proj_qkv, "qwen36_l0lin_qkv")?;
        let z = project(&la.in_proj_z, "qwen36_l0lin_z")?;
        eprintln!(
            "[qwen36] forward_layer0_linear_in_proj_probe: layer={layer_idx} \
             in_proj_qkv={:?} ({:?}) in_proj_z={:?} ({:?})",
            qkv, la.in_proj_qkv.shape, z, la.in_proj_z.shape,
        );
        Ok(())
    }

    /// Phase 4o probe (skeleton): linear-attn weight presence + shape
    /// check. Doesn't launch the recurrent kernel — Gated-DeltaNet
    /// state-space math (A_log + conv1d + dt_bias + ssm-scan) needs
    /// custom CUDA work that's ~500+ LOC of new kernel by itself.
    /// This phase verifies the loader's per-linear-layer pointers are
    /// reachable and shapes match the qwen3-next architecture so the
    /// future kernel implementation can pick up the right weights.
    pub fn forward_layer0_linear_attn_probe(&self) -> Result<()> {
        let layer_idx = match self
            .arch
            .base
            .layer_types
            .iter()
            .position(|t| matches!(t, rvllm_loader::LayerAttnType::Linear))
        {
            Some(i) => i,
            None => return Ok(()),
        };
        let la = match &self.model.layers[layer_idx].attn {
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(l) => l,
            _ => return Ok(()),
        };
        eprintln!(
            "[qwen36] forward_layer0_linear_attn_probe: layer={layer_idx} \
             A_log={:?} dt_bias={:?} conv1d={:?} in_proj_a={:?} \
             in_proj_b={:?} in_proj_qkv={:?} in_proj_z={:?} norm={:?} \
             out_proj={:?} (Gated-DeltaNet recurrent kernel TODO — \
             needs new CUDA work for ssm-scan + per-sequence state cache)",
            la.a_log.shape,
            la.dt_bias.shape,
            la.conv1d.shape,
            la.in_proj_a.shape,
            la.in_proj_b.shape,
            la.in_proj_qkv.shape,
            la.in_proj_z.shape,
            la.norm.shape,
            la.out_proj.shape,
        );
        Ok(())
    }

    /// Phase 4v: experimental forward path that threads the embedded
    /// hidden state through ONE real linear-attn layer (layer 0)
    /// before the lm_head closer. First time per-layer kernels run
    /// against the production decode path's actual hidden state
    /// (not synthetic probe input).
    ///
    /// Output is still expected to be wrong: the math approximation
    /// in the linear-attn chain is unverified, only one layer fires
    /// (39 layers skipped), positions aren't threaded, the
    /// state-cache uses the persistent layer-0 slot but never
    /// updates the per-token position. Phase 4w+ tightens against
    /// vLLM. Goal here: prove the per-layer kernels can consume the
    /// real hidden buffer that comes out of `embedding_gather` and
    /// feed `fused_rmsnorm_fp8_quant + cublaslt.fp8_gemm + argmax`.
    /// Phase 4x/5b/5c: full 40-layer decode loop. Linear-attn runs
    /// Gated-DeltaNet against the persistent state cache; full-attn
    /// Phase 2b-γ: full Qwen3-VL vision tower forward.
    ///
    /// Decodes an image (PNG/JPEG/WebP), runs the 27-layer ViT +
    /// PatchMerger natively in Rust+CUDA, returns f16 embeddings ready
    /// for splice into the post-embed text-side hidden buffer.
    ///
    /// Output layout: `[num_merged_tokens, 2048]` f16 (out_hidden_size
    /// == text hidden_size for Qwen 3.6).
    ///
    /// Phase 3-a-ii (planned) will move the body into a free fn in
    /// `crate::qwen_vision_forward` that operates on the borrow
    /// bundle returned by `vision_deps()`, so the Qwen 3.5 bringup
    /// can drive the same forward chain without duplicating ~1100
    /// LOC. The bundle exists already; nothing currently consumes
    /// it.
    pub fn vision_deps(&self) -> Result<crate::qwen_vision_forward::QwenVisionDeps<'_>> {
        let vision = self.model.vision.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::cuda(
                "vision_deps: model.vision not loaded",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;
        Ok(crate::qwen_vision_forward::QwenVisionDeps {
            vision,
            arena: &self.arena,
            stream: &self.stream,
            cublaslt: &self.cublaslt,
            fn_layernorm_inplace_f16: self.outside_kernels.fn_layernorm_inplace_f16,
            fn_gelu_tanh_f16: self.outside_kernels.fn_gelu_tanh_f16,
            fn_softmax_row_f16: self.outside_kernels.fn_softmax_row_f16,
            fn_vit_rotary_2d_f16: self.outside_kernels.fn_vit_rotary_2d_f16,
            fn_vit_pos_embed_interp_f16: self.outside_kernels.fn_vit_pos_embed_interp_f16,
            fn_scale_inplace_f16: self.outside_kernels.fn_scale_inplace_f16,
            fn_transpose_2d_f16: self.outside_kernels.fn_transpose_2d_f16,
            fn_add_bias_f16: self.outside_kernels.fn_add_bias_f16,
            fn_cast_f32_to_f16: self.outside_kernels.fn_cast_f32_to_f16,
            fn_extract_head_f16: self.outside_kernels.fn_extract_head_f16,
            fn_scatter_head_f16: self.outside_kernels.fn_scatter_head_f16,
            fn_softmax_row_f32_to_f16: self.outside_kernels.fn_softmax_row_f32_to_f16,
            fn_transpose_heads_v_f16: self.outside_kernels.fn_transpose_heads_v_f16,
            fn_scatter_heads_f16: self.outside_kernels.fn_scatter_heads_f16,
            fn_scale_inplace_f32: self.outside_kernels.fn_scale_inplace_f32,
            fn_vector_add_f16: self.outside_kernels.fn_vector_add_f16,
        })
    }

    pub fn forward_qwen_vision(&self, image_bytes: &[u8]) -> Result<VisionForwardOutput> {
        // Phase 3-a-ii: body extracted to a shared free fn in
        // `crate::qwen_vision_forward`. Both Qwen 3.5 + Qwen 3.6
        // drive the same forward chain via `vision_deps()`.
        let deps = self.vision_deps()?;
        crate::qwen_vision_forward::forward_qwen_vision(&deps, image_bytes)
    }

    /// Phase 4x/5b/5c: full 40-layer decode loop. Linear-attn runs
    /// Gated-DeltaNet against the persistent state cache; full-attn
    /// runs Q/K/V → q/k_norm → RoPE → FA2 paged decode →
    /// attn_output_gate → o_proj; MoE applies after each attn block.
    ///
    /// `start_position` is the absolute position of token_ids[0] in
    /// the sequence. For prefill: start_position=0, token_ids = full
    /// prompt. For decode-step: start_position=prefill_len, token_ids
    /// = single just-sampled token. State + KV cache are NEVER reset
    /// inside this method — caller (cuda_worker) resets per request.
    pub fn forward_qwen36_decode(
        &self,
        token_ids: &[i32],
        start_position: u32,
        vision_splice: &[(usize, &[u8])],
    ) -> Result<i32> {
        self.forward_qwen36_decode_cancellable(token_ids, start_position, vision_splice, None)
    }

    /// Spec-decode variant: runs the same layer stack as
    /// `forward_qwen36_decode_cancellable` ONCE over the K input
    /// tokens, then calls the closer K times (one per row) to
    /// produce argmaxes at every position. Required for prompt-
    /// lookup speculative decoding — Qwen 3.6's linear-attn
    /// (Gated DeltaNet) state is RECURRENT and NOT idempotent
    /// across repeated forward calls, so the "call forward N
    /// times with growing prefixes" trick that works for full-
    /// attn-only models corrupts the linear state. Single-call
    /// is the only correct path.
    ///
    /// Returns argmaxes in input order: result[i] is the base's
    /// prediction for position `start_position + i + 1` given
    /// inputs `token_ids[0..=i]`.
    pub fn forward_qwen36_decode_argmax_all(
        &self,
        token_ids: &[i32],
        start_position: u32,
        vision_splice: &[(usize, &[u8])],
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Vec<i32>> {
        let mut out = Vec::with_capacity(token_ids.len());
        self.forward_qwen36_decode_inner(
            token_ids,
            start_position,
            vision_splice,
            cancel,
            /* all_argmaxes */ Some(&mut out),
            /* skip_closer */ false,
            /* mtp_shadow_out */ None,
        )?;
        Ok(out)
    }

    /// Commit-only forward: runs the full layer stack (KV writes +
    /// recurrent state advance) but SKIPS the lm_head closer. Used
    /// by prompt-lookup spec-decode after a partial accept to
    /// replay the accepted prefix onto a freshly-restored recurrent
    /// state.
    pub fn forward_qwen36_decode_commit_only(
        &self,
        token_ids: &[i32],
        start_position: u32,
        vision_splice: &[(usize, &[u8])],
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<()> {
        self.forward_qwen36_decode_inner(
            token_ids,
            start_position,
            vision_splice,
            cancel,
            /* all_argmaxes */ None,
            /* skip_closer */ true,
            /* mtp_shadow_out */ None,
        )?;
        Ok(())
    }

    /// Same as `forward_qwen36_decode` but with caller-supplied
    /// cancellation. The flag is checked between each prompt-token
    /// iteration so a long Qwen 3.6 prefill on a client-disconnected
    /// request stops blocking the GPU. Mirrors Gemma 4's `cancel`
    /// arg in `run_generate`.
    pub fn forward_qwen36_decode_cancellable(
        &self,
        token_ids: &[i32],
        start_position: u32,
        vision_splice: &[(usize, &[u8])],
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<i32> {
        self.forward_qwen36_decode_inner(
            token_ids,
            start_position,
            vision_splice,
            cancel,
            None,
            /* skip_closer */ false,
            /* mtp_shadow_out */ None,
        )
    }

    pub fn forward_qwen36_decode_cancellable_with_mtp_shadow(
        &self,
        token_ids: &[i32],
        start_position: u32,
        vision_splice: &[(usize, &[u8])],
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<(i32, Option<i32>)> {
        let mut mtp_shadow = None;
        let base = self.forward_qwen36_decode_inner(
            token_ids,
            start_position,
            vision_splice,
            cancel,
            None,
            /* skip_closer */ false,
            Some(&mut mtp_shadow),
        )?;
        Ok((base, mtp_shadow))
    }

    fn forward_qwen36_decode_inner(
        &self,
        token_ids: &[i32],
        start_position: u32,
        vision_splice: &[(usize, &[u8])],
        cancel: Option<&std::sync::atomic::AtomicBool>,
        all_argmaxes: Option<&mut Vec<i32>>,
        skip_closer: bool,
        mtp_shadow_out: Option<&mut Option<i32>>,
    ) -> Result<i32> {
        self.forward_qwen36_decode_inner_with_workspace_overrides(
            token_ids, start_position, vision_splice, cancel,
            all_argmaxes, skip_closer, mtp_shadow_out,
            /* tok_device_override */ None,
            /* closer_argmax_dev    */ None,
        )
    }

    /// Phase 8 commit 2b — decode_inner with workspace overrides for
    /// graph-capture-friendly dispatch.
    ///
    /// * `tok_device_override`: when `Some(ptr)`, the per-call
    ///   host-side `tok_bytes` + `tok_region.copy_from_host` step is
    ///   skipped and the embed_gather reads token indices directly
    ///   from `ptr` (caller-stable device i32 pointer, e.g.
    ///   `workspace.token_dev`). The legacy default-stream sync
    ///   HtoD vanishes — required for capture.
    /// * `closer_argmax_dev`: when `Some(ptr)`, the closer writes
    ///   the argmax token id to `ptr` via
    ///   `forward_qwen36_outside_closer_device_argmax` and returns
    ///   `0` as a sentinel (caller must read the result via
    ///   `argmax_dev_to_host_token(ptr)` AFTER fencing). The
    ///   capture-breaking `cuMemcpyDtoH_v2` at the tail of the
    ///   eager closer vanishes.
    ///
    /// Both overrides default-None preserve byte-identical behavior
    /// to the pre-Phase-8 eager path. Passing one without the other
    /// is supported (operator can validate the embed-side rewrite
    /// without the closer-side rewrite, and vice versa).
    fn forward_qwen36_decode_inner_with_workspace_overrides(
        &self,
        token_ids: &[i32],
        start_position: u32,
        vision_splice: &[(usize, &[u8])],
        cancel: Option<&std::sync::atomic::AtomicBool>,
        all_argmaxes: Option<&mut Vec<i32>>,
        skip_closer: bool,
        mtp_shadow_out: Option<&mut Option<i32>>,
        tok_device_override: Option<u64>,
        closer_argmax_dev: Option<u64>,
    ) -> Result<i32> {
        self.forward_qwen36_decode_inner_with_workspace_overrides_v2(
            token_ids, start_position, vision_splice, cancel,
            all_argmaxes, skip_closer, mtp_shadow_out,
            tok_device_override, closer_argmax_dev,
            /* pos_dev_override */ None,
            /* ctx_dev_override */ None,
            /* hidden_dev_override */ None,
        )
    }

    /// Phase 8 position-indirect overrides: `pos_dev_override` and
    /// `ctx_dev_override` point at caller-managed STABLE i32 device
    /// slots holding the current decode step's absolute position
    /// and context-length. When both `Some` (and num_tokens=1), the
    /// per-call `positions_region` + `context_lens_region` arena
    /// allocations + the `qwen_fill_pos_slots_i32` fill kernel are
    /// SKIPPED — `tok_pos_dev_ptr` / `tok_cl_dev_ptr` bind directly
    /// to the overrides. This is the missing piece that makes
    /// captured-graph REPLAY produce correct output across moving
    /// positions: the captured graph reads from the stable
    /// `workspace.pos_dev` / `workspace.ctx_dev` slots, and the
    /// worker writes those slots per step BEFORE replay.
    fn forward_qwen36_decode_inner_with_workspace_overrides_v2(
        &self,
        token_ids: &[i32],
        start_position: u32,
        vision_splice: &[(usize, &[u8])],
        cancel: Option<&std::sync::atomic::AtomicBool>,
        mut all_argmaxes: Option<&mut Vec<i32>>,
        skip_closer: bool,
        mtp_shadow_out: Option<&mut Option<i32>>,
        tok_device_override: Option<u64>,
        closer_argmax_dev: Option<u64>,
        pos_dev_override: Option<u64>,
        ctx_dev_override: Option<u64>,
        // Phase 8 hidden-state→workspace refactor (2026-05-23):
        // when `Some`, the per-call `hidden_region` arena
        // allocation is BYPASSED and ALL hidden-state reads /
        // writes inside the function (embed_gather, residual
        // stream, vision splice, layer loop, closer) use this
        // pointer instead. Caller must guarantee the buffer is
        // at least `num_tokens * hidden * 2` bytes. Used by
        // the workspace forward path so the captured graph
        // references the persistent `workspace.hidden_dev` slot
        // (stable across requests) instead of an arena-bumped
        // address (stable only within one request).
        hidden_dev_override: Option<u64>,
    ) -> Result<i32> {
        if token_ids.is_empty() {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_qwen36_decode: empty token_ids",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        // Phase 8 follow-on (graph-cache reuse): take an internal
        // arena checkpoint at fn entry; restore on every exit path
        // via a Drop guard. This makes per-call arena allocations
        // (positions_region, hidden_region, q_split_region, etc.)
        // land at the same device addresses on every call —
        // deterministic across decode steps AND across requests
        // (since the worker keeps the workspace persistent above
        // `scratch_ck`, every decode_inner call enters with
        // `bump == scratch_ck`). The captured graph from request N's
        // step 0 is therefore valid for request N+1's decode steps:
        // workspace addresses are persistent, and per-call addresses
        // are re-created at the same locations every call.
        let _ck_guard = Qwen36DecodeArenaGuard::new(
            &self.arena, self.arena.checkpoint());
        let hidden = self.arch.base.hidden_size as u32;
        let vocab = self.arch.base.vocab_size as u32;
        let num_tokens = token_ids.len() as u32;
        let last_idx = (num_tokens - 1) as usize;
        let stream_raw = self.stream.raw() as u64;
        let last_hidden_bytes = (hidden as usize) * 2;

        // 1. Token IDs + embed_gather → hidden_region [num_tokens, hidden] f16.
        // Phase 8 commit 2b: when `tok_device_override` is Some, the
        // caller already populated the token index buffer on-device
        // (workspace.token_dev). Skip the per-call host-side
        // `copy_from_host` (legacy default-stream sync HtoD) so the
        // captured graph body contains zero sync HtoD.
        let token_dev_ptr: u64 = if let Some(p) = tok_device_override {
            // Override mode: caller-provided device i32 pointer.
            // Sanity: only single-token decode is supported via the
            // override today; multi-token prefill keeps using the
            // legacy host-side path.
            if num_tokens != 1 {
                return Err(rvllm_core::RvllmError::cuda(
                    "forward_qwen36_decode: tok_device_override only \
                     supports num_tokens=1 (Phase 8 single-step decode)",
                    rvllm_core::CudaErrorKind::Other,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            p
        } else {
            let mut tok_bytes = Vec::with_capacity(token_ids.len() * 4);
            for t in token_ids {
                tok_bytes.extend_from_slice(&t.to_le_bytes());
            }
            let tok_region = self.arena.region("qwen36_pl_tok", tok_bytes.len(), 16)?;
            unsafe { tok_region.copy_from_host(&tok_bytes)? };
            tok_region.device_ptr()
        };
        let hidden_bytes = (num_tokens as usize) * (hidden as usize) * 2;
        let hidden_region = self.arena.region("qwen36_pl_hidden", hidden_bytes, 16)?;
        // Phase 8 hidden-state→workspace refactor (2026-05-23):
        // every read/write of the hidden residual stream below
        // goes through `hidden_dev_ptr` rather than
        // `hidden_region.device_ptr()` directly. When the caller
        // passes `Some(workspace.hidden_dev)`, the persistent
        // workspace slot replaces the per-call arena address —
        // captured graphs end up referencing a buffer that
        // SURVIVES the inner-checkpoint restore + stays at the
        // same address across requests.
        //
        // The arena `hidden_region` allocation above is kept
        // unconditionally (cheap; bumps arena by hidden_bytes)
        // so the rest of the function's arena layout stays
        // identical between override-on and override-off paths.
        // The unused fallback allocation costs ~4 KB at
        // hidden=2048 — trivial.
        let hidden_dev_ptr: u64 = hidden_dev_override
            .unwrap_or(hidden_region.device_ptr());
        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens,
                hidden,
                vocab,
            }
            .launch(
                self.outside_kernels.fn_embedding_gather_f16,
                hidden_dev_ptr,
                self.model.outside.embed_tokens.offset_bytes,
                token_dev_ptr,
                stream_raw,
            )?;
        }
        // No fence: embed_gather wrote hidden_region on stream_raw and
        // every subsequent reader (vision splice HtoDAsync, decode loop
        // kernels) is also on stream_raw — same-stream ordering covers
        // it (Phase 4b-prep iter28).

        // Phase 4d vision splice: overwrite the placeholder-token
        // slots in hidden_region with the per-image vision-tower
        // embeddings. Each entry: (token_start_in_prompt, raw f16
        // bytes for [num_tokens, hidden_dim] = num_tokens * hidden * 2
        // bytes). The token_start is in PROMPT coordinates; we splice
        // only when start_position == 0 (i.e. prefill, when the full
        // prompt is in this hidden_region).
        if !vision_splice.is_empty() && start_position == 0 {
            let row_bytes = (hidden as usize) * 2;
            for (token_start, emb_bytes) in vision_splice {
                let dst_off = (*token_start as u64) * (row_bytes as u64);
                let len = emb_bytes.len();
                if *token_start + len / row_bytes > num_tokens as usize {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision splice would overrun hidden_region",
                        rvllm_core::CudaErrorKind::Other,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let r = cuMemcpyHtoDAsync_v2(
                        hidden_dev_ptr + dst_off,
                        emb_bytes.as_ptr() as *const _,
                        len,
                        self.stream.raw() as _,
                    );
                    if r != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 vision splice HtoDAsync",
                            rvllm_core::CudaErrorKind::MemcpyFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
            }
            // No fence: vision splice uses cuMemcpyHtoDAsync_v2 on
            // stream_raw and the decode-loop kernels also run on
            // stream_raw, so same-stream ordering is automatic
            // (Phase 4b-prep iter29).
        }

        // 2. Resolve the FP8 GEMV kernel. Hard-fail when missing —
        // the previous silent fallback to `forward_qwen36_outside_
        // closer` skipped ALL 40 transformer layers (only embed →
        // final-norm → lm_head) and produced syntactically plausible
        // but semantically garbage output, indistinguishable from a
        // successful response on the wire. Hard erroring keeps the
        // server honest: an operator with a broken kernel build
        // sees the failure at the first request, not days later
        // when "the model got dumber."
        let kernel_gemv = self
            .outside_kernels
            .fn_fp8_gemv_wpr_native_f16in
            .ok_or_else(|| {
                rvllm_core::RvllmError::cuda(
                    "qwen36 forward: fn_fp8_gemv_wpr_native_f16in not loaded — \
                 transformer layers cannot run; refusing to fall back to \
                 embed/final-norm-only path which would produce garbage tokens",
                    rvllm_core::CudaErrorKind::Other,
                    rvllm_core::CudaCtx::setup(),
                )
            })?;
        // Phase 1 of the batched-prefill plan: the per-token slot
        // is now read/written through `tok_ptr = hidden_region +
        // tok_local × hidden_bytes` directly. The previous
        // `last_hidden_region` scratch buffer + DtoD shuttle in/out
        // are gone; layer functions take `last_hidden_ptr: u64`.

        // Phase 5b: outer per-token loop. For each prompt token in
        // order: pass its slot pointer through all 40 layers
        // (linear-attn updates
        // its persistent state, full-attn writes K/V at slot=position
        // and attends causally over [0..position]), apply MoE, write
        // updated hidden back to hidden_region's slot for the closer.
        // The recurrent linear-attn state + monotonically-growing KV
        // cache mean each token sees full prompt context by the time
        // we hit the last position.
        //
        // KNOWN-PERF: prefill is O(prompt_tokens × layers). The two
        // per-token DtoD-fences (extract + writeback) were dropped
        // — same-stream ordering already guarantees what they were
        // synchronising — but the structural batched-prefill /
        // CUDA-graph path (matching Gemma's `unified_prefill`) is
        // still missing. That requires per-layer kernels that
        // accept a [N, D] hidden region rather than a single
        // [1, D] slot, plus chunked-recurrent linear-attention
        // and batched-causal full-attention variants. Substantial
        // refactor; tracked as Codex round 16 #2 follow-up.
        // Round-26: device-side fill of per-token positions and
        // context_lens. Replaces the legacy 8-byte pos_cl_region +
        // per-token `cuMemcpyHtoD_v2` (default-stream sync) with two
        // [num_tokens] i32 arrays populated by a single kernel
        // launch on `self.stream`. The race that diagnosed in
        // Round-25/26 (sync HtoD vs CU_STREAM_NON_BLOCKING) is now
        // structurally gone: every reader of `positions_region` /
        // `context_lens_region` runs on the same stream, so stream
        // ordering covers what host-side fences used to cover. Cleans
        // up both the token-major prefill loop and my layer-major
        // full-attn sub-loop. (Caller-side: per-token full-attn now
        // passes `positions + t*4` / `context_lens + t*4` as the
        // scalar-looking pointers; the existing fused_rope kernel
        // reads positions[token_idx=0] regardless of grid.x=1, so the
        // scalar view is a single-element array view of one token's
        // slot in the shared array.)
        let n_tokens_usize = num_tokens as usize;
        let positions_region = self
            .arena
            .region("qwen36_pf_positions", n_tokens_usize * 4, 16)?;
        let context_lens_region = self
            .arena
            .region("qwen36_pf_clens", n_tokens_usize * 4, 16)?;
        // Phase Full: a single-element [1] i32 with value N for the
        // f16kv prefill kernel's `context_lens[seq_idx=0]` slot. Same
        // self.stream for the memset → no race vs subsequent kernel
        // reads.
        let prefill_ctx_len_region = self.arena.region("qwen36_pf_prefill_clen", 4, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let r = cuMemsetD32Async(
                prefill_ctx_len_region.device_ptr(),
                num_tokens,
                1,
                self.stream.raw() as _,
            );
            if r != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 prefill: prefill_ctx_len memset",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // Phase 8 position-indirect: skip the fill kernel when the
        // caller provided stable pos/ctx device slots. Worker is
        // responsible for writing the per-step values to those
        // slots BEFORE invoking the decode-step entry. Without the
        // skip, the captured graph would record the scalar `start`
        // arg at capture time and re-apply step-0's position on
        // every replay — the position-frozen failure mode.
        let pos_ctx_overridden =
            pos_dev_override.is_some() && ctx_dev_override.is_some();
        if !pos_ctx_overridden {
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut pos_ptr = positions_region.device_ptr();
                let mut cl_ptr = context_lens_region.device_ptr();
                let mut start = start_position as i32;
                let mut nt = num_tokens as i32;
                let args = [
                    (&mut pos_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut cl_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut start) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid: u32 = ((num_tokens + block - 1) / block).max(1);
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_qwen_fill_pos_slots_i32.raw() as CUfunction,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 prefill: qwen_fill_pos_slots_i32 launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        // Per-token scratch checkpoint — bounds arena growth to ONE
        // token's worth of layer scratch instead of `tokens × layers`.
        // The persistent state (KV cache, linear-attn state, conv
        // state, host caches' device mirrors, the bt_persistent_ptr
        // and pos_cl_region above) all live ABOVE this checkpoint
        // (allocated at bring-up), so the per-token restore is safe.
        let per_token_ck = self.arena.checkpoint();

        // Round-24 / Phase Linear: optional layer-major prefill.
        // When `RVLLM_QWEN36_BATCH_LINEAR_PREFILL=1` and we're in
        // prefill (num_tokens > 1, no caller-supplied per-token
        // start_position quirk) the linear-attn layers run once for
        // the whole [N, D] chunk via `apply_layer_linear_attn_batched`
        // — the recurrent-state kernel carries state across all N
        // tokens internally instead of N host-driven launches. Full-
        // attn layers and MoE stay per-token in this v0 because they
        // need per-position KV-slot writes (full-attn) or per-token
        // expert routing (MoE) — those are the next phases.
        //
        // Default OFF until canary green vs the per-token reference.
        // Round-27d: batched-prefill phases are default-ON in production
        // post-audit (Phases Linear / Full / MoE 6a–6c all byte-identical
        // to the per-token reference across 1782 layer/phase/token dump
        // rows). Set the env-var to "0" / "false" to opt back into the
        // legacy token-major path.
        let batch_linear = num_tokens > 1
            && std::env::var("RVLLM_QWEN36_BATCH_LINEAR_PREFILL")
                .map(|s| !matches!(s.as_str(), "0" | "false" | "FALSE" | "no"))
                .unwrap_or(true);
        if batch_linear {
            // Round-24 audit: mirror the per-token path's last-token
            // dump from `RVLLM_QWEN36_DUMP_DIR`. Drops the same files
            // (embed.f16, layer_NN_attn.f16, layer_NN_moe.f16) for
            // the LAST prompt token's hidden row so the per-token
            // and batched paths can be diffed layer-by-layer with a
            // simple `cmp` / cosine harness.
            let dump_dir = std::env::var("RVLLM_QWEN36_DUMP_DIR").ok();
            let dump_token_idx = num_tokens.saturating_sub(1) as usize;
            let dump_active = dump_dir.is_some() && start_position == 0;
            // Round-26 audit: dump per-(layer, tok) hidden rows. At
            // any "post layer-L attn for tok t" moment, hidden[t] is
            // the right snapshot. In layer-major we dump all rows at
            // once at the end of each layer's block (all rows post-
            // layer-L). In token-major we dump hidden[t] AT THE
            // MOMENT tok t reaches layer L — not later, because
            // subsequent layers have already overwritten that row by
            // batch-dump time. File naming: `layer_LL_attn_tok{TT}.f16`
            // (4096 bytes each = 1 hidden row). cmp_qwen36 iterates
            // (L, phase, t) and reports per-row divergence.
            if let Some(dir) = dump_dir.as_ref() {
                if dump_active {
                    let _ = std::fs::create_dir_all(dir);
                    self.stream.fence()?;
                    for t in 0..num_tokens {
                        let mut buf = vec![0u8; last_hidden_bytes];
                        let row_ptr =
                            hidden_dev_ptr + (t as u64) * (last_hidden_bytes as u64);
                        #[cfg(feature = "cuda")]
                        unsafe {
                            use cudarc::driver::sys::*;
                            let _ = cuMemcpyDtoH_v2(
                                buf.as_mut_ptr() as *mut _,
                                row_ptr,
                                last_hidden_bytes,
                            );
                        }
                        let _ = std::fs::write(format!("{dir}/embed_tok{t:02}.f16"), &buf);
                    }
                }
            }
            let _ = dump_token_idx;
            // Pre-stage pos+context_len for each token. Full-attn
            // still runs per-token below and reuses the existing
            // pos_cl_region, so we update it inside the per-token
            // sub-loop (one HtoD per full-attn layer hit).
            //
            // Layer-major orchestration: each layer processes ALL N
            // tokens before the next layer starts. Within a linear
            // layer that's one batched call; within a full-attn or
            // MoE layer it's a per-token sub-loop (their kernels are
            // not yet batched). This is correctness-equivalent to the
            // token-major loop because:
            //   • linear-attn's recurrent state advances across N
            //     tokens identically whether the host loops or the
            //     kernel loops.
            //   • full-attn writes per-token K/V at slot=tok_pos and
            //     reads slots [0..tok_pos+1]; per-token sub-loop in
            //     ascending order preserves causality.
            //   • MoE is independent per-token.
            let mut linear_seq: u32 = 0;
            let mut full_seq: u32 = 0;
            for layer_idx in 0..self.model.layers.len() {
                let layer_ck = self.arena.checkpoint();
                let post_attn_norm_ptr = match &self.model.layers[layer_idx].attn {
                    rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(la) => {
                        // Round-24 audit: env-gate
                        // RVLLM_QWEN36_BATCH_LINEAR_HOST_LOOP=1 bypasses the
                        // batched kernel and replays the per-token function
                        // N times in this layer's slot. If that produces
                        // byte-identical hidden[N-1] dumps to the
                        // token-major path, the layer-major orchestration
                        // is correct and any divergence is in the batched
                        // kernel itself.
                        let host_loop_only = std::env::var("RVLLM_QWEN36_BATCH_LINEAR_HOST_LOOP")
                            .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                            .unwrap_or(false);
                        if host_loop_only {
                            for t in 0..num_tokens {
                                let tok_ptr = hidden_dev_ptr
                                    + (t as u64) * (last_hidden_bytes as u64);
                                let inner_ck = self.arena.checkpoint();
                                self.apply_layer_linear_attn(
                                    la,
                                    linear_seq,
                                    tok_ptr,
                                    kernel_gemv,
                                    hidden,
                                    last_hidden_bytes,
                                )?;
                                unsafe {
                                    self.arena.restore(inner_ck);
                                }
                            }
                        } else {
                            self.apply_layer_linear_attn_batched(
                                la,
                                linear_seq,
                                hidden_dev_ptr,
                                num_tokens,
                                kernel_gemv,
                                hidden,
                            )?;
                        }
                        linear_seq += 1;
                        la.post_attention_layernorm.offset_bytes
                    }
                    rvllm_loader::qwen36_weights::Qwen36LayerAttn::Full(fl) => {
                        // Round-26 / Phase Full: env-gate
                        // RVLLM_QWEN36_BATCH_FULL_PREFILL=1 enables the
                        // batched-prefill path (one launch per layer
                        // for the entire chunk). When off, fall back to
                        // the per-token sub-loop. start_position must
                        // be 0 (= prefill chunk anchored at sequence
                        // start) for batched mode; later chunks use the
                        // per-token path.
                        // Round-27d: default-ON post-audit. "0"/"false"
                        // selects the legacy per-token sub-loop.
                        // NVFP4 batched full-attn prefill remains opt-in
                        // while it is validated against the per-token
                        // reference. The opt-in path uses the conservative
                        // non-unified NVFP4 prefill kernel by default; the
                        // faster unified kernel has its own separate flag
                        // because it is not yet output-equivalent on the
                        // repeat-pattern probe.
                        let nvfp4_batch_full = !matches!(self.kv_dtype, Qwen36KvDtype::Nvfp4)
                            || std::env::var("RVLLM_QWEN36_NVFP4_BATCH_FULL_PREFILL")
                                .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                                .unwrap_or(false);
                        let batch_full = start_position == 0
                            && nvfp4_batch_full
                            && std::env::var("RVLLM_QWEN36_BATCH_FULL_PREFILL")
                                .map(|s| !matches!(s.as_str(), "0" | "false" | "FALSE" | "no"))
                                .unwrap_or(true);
                        if batch_full {
                            let inner_ck = self.arena.checkpoint();
                            self.apply_layer_full_attn_batched(
                                fl,
                                full_seq,
                                hidden_dev_ptr,
                                num_tokens,
                                kernel_gemv,
                                hidden,
                                last_hidden_bytes,
                                positions_region.device_ptr(),
                                prefill_ctx_len_region.device_ptr(),
                                full_seq,
                            )?;
                            unsafe {
                                self.arena.restore(inner_ck);
                            }
                        } else {
                            for t in 0..num_tokens {
                                let tok_pos = start_position + t;
                                let tok_ptr = hidden_dev_ptr
                                    + (t as u64) * (last_hidden_bytes as u64);
                                let pos_p = positions_region.device_ptr() + (t as u64) * 4;
                                let cl_p = context_lens_region.device_ptr() + (t as u64) * 4;
                                let inner_ck = self.arena.checkpoint();
                                self.apply_layer_full_attn(
                                    fl,
                                    full_seq,
                                    tok_pos,
                                    tok_ptr,
                                    kernel_gemv,
                                    hidden,
                                    last_hidden_bytes,
                                    pos_p,
                                    cl_p,
                                )?;
                                unsafe {
                                    self.arena.restore(inner_ck);
                                }
                            }
                        }
                        full_seq += 1;
                        fl.post_attention_layernorm.offset_bytes
                    }
                };
                if dump_active {
                    let dir = dump_dir.as_ref().unwrap();
                    self.stream.fence()?;
                    for t in 0..num_tokens {
                        let mut buf = vec![0u8; last_hidden_bytes];
                        let row_ptr =
                            hidden_dev_ptr + (t as u64) * (last_hidden_bytes as u64);
                        #[cfg(feature = "cuda")]
                        unsafe {
                            use cudarc::driver::sys::*;
                            let _ = cuMemcpyDtoH_v2(
                                buf.as_mut_ptr() as *mut _,
                                row_ptr,
                                last_hidden_bytes,
                            );
                        }
                        let _ = std::fs::write(
                            format!("{dir}/layer_{layer_idx:02}_attn_tok{t:02}.f16"),
                            &buf,
                        );
                    }
                }
                // MoE: env-gated batched routing (Phase 6a /
                // Round-27). When `RVLLM_QWEN36_BATCH_MOE_PREFILL=1`
                // and start_position == 0, route batched (one
                // launch each for router GEMV + topk-softmax) and
                // delegate per-token expert FFN with override. Else
                // fall back to per-token routing+FFN.
                // Round-27d: default-ON post-audit.
                let batch_moe = start_position == 0
                    && std::env::var("RVLLM_QWEN36_BATCH_MOE_PREFILL")
                        .map(|s| !matches!(s.as_str(), "0" | "false" | "FALSE" | "no"))
                        .unwrap_or(true);
                if batch_moe {
                    let inner_ck = self.arena.checkpoint();
                    self.apply_layer_moe_batched(
                        &self.model.layers[layer_idx].moe,
                        post_attn_norm_ptr,
                        hidden_dev_ptr,
                        num_tokens,
                        kernel_gemv,
                        hidden,
                        last_hidden_bytes,
                        layer_idx,
                    )?;
                    unsafe {
                        self.arena.restore(inner_ck);
                    }
                } else {
                    for t in 0..num_tokens {
                        let tok_ptr =
                            hidden_dev_ptr + (t as u64) * (last_hidden_bytes as u64);
                        let inner_ck = self.arena.checkpoint();
                        self.apply_layer_moe(
                            &self.model.layers[layer_idx].moe,
                            post_attn_norm_ptr,
                            tok_ptr,
                            kernel_gemv,
                            hidden,
                            last_hidden_bytes,
                            layer_idx,
                        )?;
                        unsafe {
                            self.arena.restore(inner_ck);
                        }
                    }
                }
                if dump_active {
                    let dir = dump_dir.as_ref().unwrap();
                    self.stream.fence()?;
                    for t in 0..num_tokens {
                        let mut buf = vec![0u8; last_hidden_bytes];
                        let row_ptr =
                            hidden_dev_ptr + (t as u64) * (last_hidden_bytes as u64);
                        #[cfg(feature = "cuda")]
                        unsafe {
                            use cudarc::driver::sys::*;
                            let _ = cuMemcpyDtoH_v2(
                                buf.as_mut_ptr() as *mut _,
                                row_ptr,
                                last_hidden_bytes,
                            );
                        }
                        let _ = std::fs::write(
                            format!("{dir}/layer_{layer_idx:02}_moe_tok{t:02}.f16"),
                            &buf,
                        );
                    }
                }
                if let Some(c) = cancel {
                    if c.load(std::sync::atomic::Ordering::Relaxed) {
                        return Err(rvllm_core::RvllmError::cuda(
                            "forward_qwen36_decode: cancelled by caller",
                            rvllm_core::CudaErrorKind::Other,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                unsafe {
                    self.arena.restore(layer_ck);
                }
            }

            // Skip the legacy token-major loop below by jumping to
            // the closer's reads. Use a sentinel `num_tokens` of 0
            // (well, we set the flag and break out).
            // Actually we need to fall through to the closer that
            // reads hidden_region's last token; the token-major
            // loop already wrote into the same region, so just skip
            // that loop. Implement this by gating the for loop on
            // !batch_linear, which means we need to wrap it.
            // Instead: short-circuit here by setting a flag the
            // for loop below can read.
        }
        let _batch_linear_done = batch_linear;
        for tok_local in 0..(if batch_linear { 0 } else { num_tokens }) {
            // Cancellation check — long prefills can be cancelled by
            // a client disconnect; without this the GPU keeps running
            // through every prompt token even though the HTTP handler
            // has already returned.
            if let Some(c) = cancel {
                if c.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(rvllm_core::RvllmError::cuda(
                        "forward_qwen36_decode: cancelled by caller",
                        rvllm_core::CudaErrorKind::Other,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            // Reset scratch from the previous token's pass. First
            // iteration's restore is a no-op (used == per_token_ck
            // already).
            unsafe {
                self.arena.restore(per_token_ck);
            }
            let tok_pos = start_position + tok_local;
            // Update pos+context_len for THIS token. Sync HtoD; runs
            // outside any future graph-captured region (it's a
            // host-side mutation of the device buffer that the
            // captured kernels read from on the next replay).
            // Round-26: positions/context_lens now live in
            // device-filled arrays from the single launch above; no
            // per-token HtoD anymore. Per-token full-attn calls below
            // pass `positions + t*4` / `context_lens + t*4`.
            // Phase 8 position-indirect: when both pos+ctx overrides
            // are present, bind directly to the stable workspace
            // slots. For tok_local > 0 this would be wrong (single-
            // token override only), but pos_ctx_overridden is only
            // set when num_tokens=1 → tok_local always 0.
            let tok_pos_dev_ptr = if pos_ctx_overridden {
                pos_dev_override.unwrap()
            } else {
                positions_region.device_ptr() + (tok_local as u64) * 4
            };
            let tok_cl_dev_ptr = if pos_ctx_overridden {
                ctx_dev_override.unwrap()
            } else {
                context_lens_region.device_ptr() + (tok_local as u64) * 4
            };
            // Phase 1 of the Qwen batched-prefill plan: the layer
            // functions now accept a raw device pointer to the
            // per-token slot in `hidden_region`. The previous
            // DtoD-extract (`hidden_region[off]` → `last_hidden_region`)
            // and DtoD-writeback are gone — `apply_layer_*` reads and
            // writes through `tok_ptr` directly, eliminating two
            // launches per token.
            let tok_ptr =
                hidden_dev_ptr + (tok_local as u64) * (last_hidden_bytes as u64);

            // Phase 5e: optional per-layer activation dump for
            // numerical-correctness audit vs vLLM. Set
            // RVLLM_QWEN36_DUMP_DIR to enable. Dumps last_hidden_region
            // contents (f16, hidden=2048 elems → 4096 bytes) after each
            // layer's attn block and after each layer's MoE block, but
            // ONLY for the last prompt token (tok_local == num_tokens-1
            // && start_position == 0) to avoid runaway disk usage on
            // multi-step decode.
            // Round-26 audit: dump per-(layer, phase, tok) hidden rows.
            // Now fires for ALL tokens (was last-token-only) so the
            // cmp harness can compare per-token rows between token-
            // major and layer-major paths and localise where any
            // intermediate token rows diverge.
            let dump_dir = std::env::var("RVLLM_QWEN36_DUMP_DIR").ok();
            let dump_this_token = dump_dir.is_some() && start_position == 0;
            if let Some(dir) = dump_dir.as_ref() {
                if dump_this_token {
                    let _ = std::fs::create_dir_all(dir);
                    // Dump THIS token's embedding (input to layer 0).
                    let mut buf = vec![0u8; last_hidden_bytes];
                    self.stream.fence()?;
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        let _ =
                            cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut _, tok_ptr, last_hidden_bytes);
                    }
                    let _ = std::fs::write(format!("{dir}/embed_tok{tok_local:02}.f16"), &buf);
                }
            }

            let mut linear_seq: u32 = 0;
            let mut full_seq: u32 = 0;
            for layer_idx in 0..self.model.layers.len() {
                let post_attn_norm_ptr = match &self.model.layers[layer_idx].attn {
                    rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(la) => {
                        self.apply_layer_linear_attn(
                            la,
                            linear_seq,
                            tok_ptr,
                            kernel_gemv,
                            hidden,
                            last_hidden_bytes,
                        )?;
                        linear_seq += 1;
                        la.post_attention_layernorm.offset_bytes
                    }
                    rvllm_loader::qwen36_weights::Qwen36LayerAttn::Full(fl) => {
                        self.apply_layer_full_attn(
                            fl,
                            full_seq,
                            tok_pos,
                            tok_ptr,
                            kernel_gemv,
                            hidden,
                            last_hidden_bytes,
                            tok_pos_dev_ptr,
                            tok_cl_dev_ptr,
                        )?;
                        full_seq += 1;
                        fl.post_attention_layernorm.offset_bytes
                    }
                };
                if dump_this_token {
                    let dir = dump_dir.as_ref().unwrap();
                    let mut buf = vec![0u8; last_hidden_bytes];
                    self.stream.fence()?;
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        let _ =
                            cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut _, tok_ptr, last_hidden_bytes);
                    }
                    let _ = std::fs::write(
                        format!("{dir}/layer_{layer_idx:02}_attn_tok{tok_local:02}.f16"),
                        &buf,
                    );
                }
                self.apply_layer_moe(
                    &self.model.layers[layer_idx].moe,
                    post_attn_norm_ptr,
                    tok_ptr,
                    kernel_gemv,
                    hidden,
                    last_hidden_bytes,
                    layer_idx,
                )?;
                if dump_this_token {
                    let dir = dump_dir.as_ref().unwrap();
                    let mut buf = vec![0u8; last_hidden_bytes];
                    self.stream.fence()?;
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        let _ =
                            cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut _, tok_ptr, last_hidden_bytes);
                    }
                    let _ = std::fs::write(
                        format!("{dir}/layer_{layer_idx:02}_moe_tok{tok_local:02}.f16"),
                        &buf,
                    );
                }
            }
            // No DtoD writeback needed: the layer functions wrote
            // their final residual directly into `tok_ptr`, which
            // already points at the destination slot in hidden_region.
        }
        // No fence: the per-token chain's last residual-GPU kernel
        // runs on stream_raw, and the closer's first op
        // (`fused_rmsnorm_fp8_quant`) is also on stream_raw. cublasLt
        // attaches to stream_raw via the explicit `stream_raw` arg
        // and the iter27 argmax+DtoH inside the closer already has
        // its own fence-before-DtoH. (Phase 4b-prep iter30.)

        // 5. Final norm + lm_head + argmax (outside-only closer).
        //    For spec-decode (`all_argmaxes = Some(_)`), the closer
        //    runs K times — once per row — so the caller gets the
        //    argmax at every input position. The layer stack above
        //    only runs once (with the K-token input). Linear-attn
        //    state stays consistent because we don't re-run any
        //    prefix; we just read more positions from the same
        //    hidden_region. Cost: K closer calls vs 1, dominated
        //    by K lm_head GEMMs at M=1. For K=4-8 this is
        //    structurally negligible vs the K-position layer stack
        //    which is the real spend.
        if skip_closer {
            // Commit-only path: layer stack + KV writes + recurrent
            // state update have already run; the closer would
            // produce argmaxes the caller doesn't need. Return 0
            // — the value is never consumed by callers passing
            // skip_closer=true.
            return Ok(0);
        }
        if let Some(out) = all_argmaxes.as_mut() {
            // Closer-all path (single fused rmsnorm + M=K fp8_gemm +
            // grid=K argmax + one fence + one rows*4 DtoH). Faster
            // per call than the K-times closer loop BUT
            // hardware-bisected 2026-05-17 with corrupted output
            // on long generations:
            //   * Short outputs (≤60 tok): byte-identical to closer-
            //     loop output.
            //   * Long outputs (~120 tok factorial): closer-all
            //     produces a corrupted "duplicated header" preamble
            //     before the real function body.
            //
            // Root cause (codex review 2026-05-17, with citations):
            // cublaslt.fp8_gemm at v3/crates/rvllm-cutlass/src/
            // cublaslt.rs:972-987 wires `a_scale_ptr` and
            // `b_scale_ptr` as SCALAR pointers (cuBLASLt's
            // MATMUL_DESC_A_SCALE_POINTER / _B_SCALE_POINTER take
            // a single fp32 per matmul). Only `b_channelscale`
            // sets MATMUL_DESC_B_SCALE_MODE to outer-vector at
            // line 989-998 — and that is the WEIGHT side, not the
            // activation side. So when closer-all calls fp8_gemm
            // with M=K=rows>1 and a [rows] f32 hidden_scale_region
            // at line 8387-8398 here, cuBLASLt reads the FIRST
            // f32 (row 0's activation scale) and applies it to
            // every row. The per-row closer at qwen36_bring_up.rs
            // :8233-8259 avoids this because each call passes
            // num_tokens: 1 and a single scale, so the scalar
            // mode is correct.
            //
            // Fix is NOT "fp8_gemm fallback layout" as the prior
            // comment claimed. The right fix is either:
            //   (a) loop the LM-head GEMM at M=1 per row inside
            //       closer-all (defeats the perf goal but unblocks
            //       correctness), or
            //   (b) add a true per-row activation-scale mode to
            //       rvllm-cutlass::fp8_gemm (likely via
            //       MATMUL_DESC_A_SCALE_MODE = ROW or VECTOR),
            //       then re-bisect.
            // Until (b) lands: gate is OFF by default.
            let closer_all = std::env::var("RVLLM_QWEN36_SPEC_CLOSER_ALL").as_deref() == Ok("1");
            if closer_all {
                #[cfg(feature = "cuda")]
                {
                    self.forward_qwen36_outside_closer_all(
                        hidden_dev_ptr,
                        num_tokens,
                        hidden,
                        vocab,
                        out,
                    )?;
                }
                #[cfg(not(feature = "cuda"))]
                {
                    out.clear();
                    out.resize(num_tokens as usize, 0i32);
                }
            } else {
                out.clear();
                out.reserve(num_tokens as usize);
                for i in 0..(num_tokens as usize) {
                    let t = self.forward_qwen36_outside_closer(
                        hidden_dev_ptr,
                        num_tokens,
                        hidden,
                        vocab,
                        i,
                    )?;
                    out.push(t);
                }
            }
            Ok(*out.last().unwrap_or(&0))
        } else {
            // Phase 8 commit 2b: when `closer_argmax_dev` is Some,
            // run the device-argmax closer instead of the eager
            // DtoH-tailed closer. Returns 0 as a sentinel — the
            // caller is expected to call `argmax_dev_to_host_token`
            // OUTSIDE any captured-graph body to extract the real
            // token id after fencing the stream.
            //
            // MTP shadow is skipped in this override path (Phase 8
            // captured decode is single-token, no spec).
            let base = if let Some(argmax_dev) = closer_argmax_dev {
                if mtp_shadow_out.is_some() {
                    return Err(rvllm_core::RvllmError::cuda(
                        "forward_qwen36_decode: closer_argmax_dev + \
                         mtp_shadow_out is unsupported (Phase 8 + \
                         MTP-spec are mutually exclusive)",
                        rvllm_core::CudaErrorKind::Other,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
                self.forward_qwen36_outside_closer_device_argmax(
                    hidden_dev_ptr, num_tokens, hidden, vocab,
                    last_idx, argmax_dev)?;
                0i32
            } else {
                let base = self.forward_qwen36_outside_closer(
                    hidden_dev_ptr, num_tokens, hidden, vocab, last_idx)?;
                if let Some(out) = mtp_shadow_out {
                    let last_hidden_row_ptr =
                        hidden_dev_ptr + (last_idx as u64) * (hidden as u64) * 2;
                    *out = Some(self.forward_qwen36_mtp_from_hidden_ptr(
                        last_hidden_row_ptr,
                        base,
                        start_position + last_idx as u32,
                    )?);
                }
                base
            };
            Ok(base)
        }
    }

    /// Phase 5d helper: apply one linear-attn layer's Gated-DeltaNet
    /// chain on `last_hidden_region`. Reference: vLLM
    /// `gdn_linear_attn.py::fused_post_conv_prep` and surrounding
    /// per-step decode path.
    ///
    /// Shapes (Qwen 3.6 35B-A3B):
    ///   - num_k_heads = 16, head_k_dim = 128 → key_dim = 2048
    ///   - num_v_heads = 32, head_v_dim = 128 → value_dim = 4096
    ///   - in_proj_qkv [8192, hidden] → conv_dim = 2*key_dim + value_dim = 8192
    ///     contiguous as [Q (key_dim) | K (key_dim) | V (value_dim)]
    ///   - in_proj_z [value_dim, hidden]
    ///   - in_proj_a [num_v_heads, hidden] (bf16)
    ///   - in_proj_b [num_v_heads, hidden] (bf16)
    ///   - A_log [num_v_heads], dt_bias [num_v_heads]
    ///   - norm.weight [head_v_dim] (per-v-head RMSNorm gamma)
    ///
    /// Pipeline:
    ///   1. input_layernorm on copy of last_hidden
    ///   2. in_proj_qkv FP8 GEMV → conv-input [conv_dim]
    ///   3. causal_conv1d → conv_out [conv_dim]
    ///   4. Host: SiLU + split into Q[16,128], K[16,128], V[32,128]
    ///   5. Host: L2-norm Q and K per-head
    ///   6. Host: in_proj_a · normed → a [num_v_heads]
    ///      Host: in_proj_b · normed → b [num_v_heads]
    ///      α[v] = exp(-exp(A_log[v]) * softplus(a[v] + dt_bias[v]))
    ///      β[v] = sigmoid(b[v])
    ///   7. GQA expand: K_exp[v]=K[v/2], Q_exp[v]=Q[v/2] (each k-head
    ///      shared by 2 v-heads since num_v=2*num_k)
    ///   8. ssm state update kernel (32 v-heads, head_v=128, head_k=128)
    ///      against persistent state slice
    ///   9. Q_exp · S → readout[32, 128]
    ///  10. in_proj_z FP8 GEMV → z [value_dim]
    ///  11. Per-v-head RMSNorm with norm.weight + sigmoid(z) gate
    ///  12. out_proj FP8 GEMV → o_buf [hidden]
    ///  13. Host residual sum → write back to last_hidden_region
    #[allow(clippy::too_many_arguments)]
    /// `last_hidden_ptr`: device pointer to the (single-token, for now)
    /// hidden slot this layer reads from and writes back into. Phase 1
    /// (Qwen batched-prefill plan) replaced an `&Region` parameter
    /// here so the prefill loop can pass an offset into the larger
    /// `hidden_region` directly — no per-token DtoD shuttle. See
    /// `v3/QWEN_BATCHED_PREFILL_PLAN.md`.
    fn apply_layer_linear_attn(
        &self,
        la: &rvllm_loader::qwen36_weights::Qwen36LinearAttnLayer,
        linear_seq_idx: u32,
        last_hidden_ptr: u64,
        kernel_gemv: rvllm_kernels::KernelFn,
        hidden: u32,
        last_hidden_bytes: usize,
    ) -> Result<()> {
        let stream_raw = self.stream.raw() as u64;
        let qkv_n = la.in_proj_qkv.shape[0] as u32; // 8192 = conv_dim
        let z_n = la.in_proj_z.shape[0] as u32; // 4096 = value_dim
        let out_n = la.out_proj.shape[0] as u32; // 2048 = hidden
        let out_k = la.out_proj.shape[1] as u32; // 4096 = value_dim
        let m: u32 = 1;
        let qkv_bs = match la.in_proj_qkv.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let z_bs = match la.in_proj_z.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let out_bs = match la.out_proj.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };

        // Hardcoded for Qwen 3.6 35B-A3B; could be plumbed from arch.
        let num_k_heads: u32 = 16;
        let num_v_heads: u32 = 32;
        let head_k_dim: u32 = 128;
        let head_v_dim: u32 = 128;
        let key_dim = num_k_heads * head_k_dim; // 2048
        let _value_dim = num_v_heads * head_v_dim; // 4096
        let v_per_k = num_v_heads / num_k_heads; // 2
        let _kus = num_k_heads as usize;
        let vus = num_v_heads as usize;
        let hkd = head_k_dim as usize;
        let hvd = head_v_dim as usize;

        // 1. input_layernorm on copy of last_hidden.
        let normed_region = self
            .arena
            .region("qwen36_pl_normed", last_hidden_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoDAsync_v2(
                normed_region.device_ptr(),
                last_hidden_ptr,
                last_hidden_bytes,
                self.stream.raw() as _,
            );
        }
        let eps = self.arch.base.rms_norm_eps;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                normed_region.device_ptr(),
                la.input_layernorm.offset_bytes,
                stream_raw,
            )?;
        }
        // No fence: in_proj_qkv runs on the same stream and reads
        // normed_region after rmsnorm has written to it; stream
        // ordering guarantees that.

        // 2. in_proj_qkv FP8 GEMV → conv-input. (Phase 4a routing.)
        let qkv_bytes_dev = (qkv_n as usize) * 2;
        let qkv_region = self.arena.region("qwen36_pl_qkv", qkv_bytes_dev, 16)?;
        unsafe {
            self.fp8_proj_dispatch(
                kernel_gemv,
                qkv_region.device_ptr(),
                la.in_proj_qkv.offset_bytes,
                qkv_bs,
                normed_region.device_ptr(),
                m,
                qkv_n,
                hidden,
                stream_raw,
            )?;
        }
        // No fence: conv_state_advance + conv1d run on the same
        // stream and read qkv after the GEMV has written to it.

        // 3. causal_conv1d. Single-step: prepend the ks-1=3 previous
        //    timesteps from the persistent conv-state cache, append
        //    current qkv = 4 timesteps. After conv1d, update the cache
        //    by shifting (drop oldest, append current).
        let ks: u32 = 4;
        let conv_in_bytes = ((ks as usize) * (qkv_n as usize)) * 2;
        let conv_in_region = self.arena.region("qwen36_pl_cin", conv_in_bytes, 16)?;
        let conv_out_region = self.arena.region("qwen36_pl_cout", qkv_bytes_dev, 16)?;
        // GPU-side conv_in assembly + state advance. One launch
        // replaces 2× DtoH + 2× HtoD + two CPU vec slicings.
        let conv_state_ptr_layer = self.conv_state_layer_ptr(linear_seq_idx);
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut conv_in = conv_in_region.device_ptr();
            let mut state = conv_state_ptr_layer;
            let mut cur = qkv_region.device_ptr();
            let mut ts_i: i32 = qkv_n as i32;
            let args = [
                (&mut conv_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut state) as *mut u64 as *mut core::ffi::c_void,
                (&mut cur) as *mut u64 as *mut core::ffi::c_void,
                (&mut ts_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((qkv_n + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_conv_state_advance_f16.raw() as CUfunction,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn conv_state_advance launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut output = conv_out_region.device_ptr();
            let mut input = conv_in_region.device_ptr();
            let mut weight = la.conv1d.offset_bytes;
            let mut sl: i32 = 1;
            let mut ch = qkv_n as i32;
            let mut k_arg = ks as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut weight) as *mut u64 as *mut core::ffi::c_void,
                (&mut sl) as *mut i32 as *mut core::ffi::c_void,
                (&mut ch) as *mut i32 as *mut core::ffi::c_void,
                (&mut k_arg) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid_x = (qkv_n + block - 1) / block;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_causal_conv1d_f16.raw() as CUfunction,
                grid_x,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 causal_conv1d_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // No fence: silu_l2_gqa runs on the same stream after conv1d.

        // 4+5. GPU-side fused silu + Q/K L2-norm + GQA-expand + V silu-pack.
        // Allocates the q_exp / k_exp / v_pack device regions and
        // writes them directly. Replaces the host pipeline (DtoH
        // conv_out + CPU silu + per-k-head L2 + GQA-expand into
        // host bytes + HtoD q/k/v).
        let qk_bytes_pre = vus * hkd * 2;
        let v_bytes_pre = vus * hvd * 2;
        let q_region = self.arena.region("qwen36_pl_q", qk_bytes_pre, 16)?;
        let k_region = self.arena.region("qwen36_pl_k", qk_bytes_pre, 16)?;
        let v_region = self.arena.region("qwen36_pl_v", v_bytes_pre, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut q_out = q_region.device_ptr();
            let mut k_out = k_region.device_ptr();
            let mut v_out = v_region.device_ptr();
            let mut conv_p = conv_out_region.device_ptr();
            let mut vus_i: i32 = vus as i32;
            let mut hkd_i: i32 = hkd as i32;
            let mut hvd_i: i32 = hvd as i32;
            let mut kd_i: i32 = key_dim as i32;
            let mut nvh: i32 = num_v_heads as i32;
            let mut vpk: i32 = v_per_k as i32;
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
            let block: u32 = (hkd.max(hvd)) as u32;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_qwen_linear_silu_l2_gqa_f16.raw() as CUfunction,
                vus as u32,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn silu_l2_gqa launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // 6. GPU alpha/beta: fused dot-product + softplus / sigmoid
        // launched against the live normed input. Pre-allocate
        // alpha_region and beta_region as f32 device buffers; the
        // kernel writes them directly. Replaces the host pipeline
        // (DtoH input → f16→f32 → nested CPU GEMV → HtoD alpha/beta)
        // with one launch. The bring-up-time host cache from iter3
        // becomes unused on the production path; kept as a fallback
        // reference for diagnostics.
        let h_us = hidden as usize;
        let alpha_region = self.arena.region("qwen36_pl_alpha", vus * 4, 16)?;
        let beta_region = self.arena.region("qwen36_pl_beta", vus * 4, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut a_out = alpha_region.device_ptr();
            let mut b_out = beta_region.device_ptr();
            let mut a_w_p = la.in_proj_a.offset_bytes;
            let mut b_w_p = la.in_proj_b.offset_bytes;
            let mut a_log_p = la.a_log.offset_bytes;
            let mut dt_bias_p = la.dt_bias.offset_bytes;
            let mut in_p = normed_region.device_ptr();
            let mut vus_i: i32 = vus as i32;
            let mut h_i: i32 = h_us as i32;
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
            let block: u32 = 256u32.min(h_us as u32).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_qwen_linear_alpha_beta_f16.raw() as CUfunction,
                vus as u32,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn alpha_beta launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Phase 5i: GPU delta-rule kernel.
        // Build GQA-expanded Q/K (one row per v-head) on host, push to
        // device, alongside V (already per-v-head), alpha, beta. Kernel
        // does forget + delta correction + state update + readout in
        // one launch, writing readout to readout_region. State is
        // updated in-place in the persistent linear-state slice.
        let layer_state_ptr = self.linear_state_layer_ptr(linear_seq_idx);
        let scale = 1.0f32 / (head_k_dim as f32).sqrt();
        let v_bytes = vus * hvd * 2;
        // q_region, k_region, v_region were allocated + filled by
        // the silu_l2_gqa GPU kernel in step 4+5; alpha_region and
        // beta_region by the alpha_beta kernel in step 6. All four
        // are device-resident already — no further HtoD needed.
        let readout_region = self.arena.region("qwen36_pl_readout", v_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
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
            // Block = head_v_dim threads, one per output row.
            // Shared mem = (2*head_k_dim + head_v_dim) * 4 bytes.
            let smem = (2 * head_k_dim + head_v_dim) * 4;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_gated_delta_rule_decode_f16.raw() as CUfunction,
                num_v_heads,
                1,
                1,
                head_v_dim,
                1,
                1,
                smem,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 gated_delta_rule_decode_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // No fence: in_proj_z + rmsnorm_gated run on the same
        // stream after the delta-rule kernel.

        // 10. in_proj_z FP8 GEMV → z [value_dim] (stays on device).
        let z_bytes_dev = (z_n as usize) * 2;
        let z_region = self.arena.region("qwen36_pl_z", z_bytes_dev, 16)?;
        unsafe {
            self.fp8_proj_dispatch(
                kernel_gemv,
                z_region.device_ptr(),
                la.in_proj_z.offset_bytes,
                z_bs,
                normed_region.device_ptr(),
                m,
                z_n,
                hidden,
                stream_raw,
            )?;
        }

        // 11. GPU per-v-head RMSNormGated + silu(z) gate fused into
        // one launch. Replaces 3× DtoH (readout, z, norm.gamma) +
        // host CPU loop over (vus × hvd) elements + 1× HtoD gated.
        let gated_region = self.arena.region("qwen36_pl_gated", vus * hvd * 2, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut g_out = gated_region.device_ptr();
            let mut r_in = readout_region.device_ptr();
            let mut z_in = z_region.device_ptr();
            let mut gamma_p = la.norm.offset_bytes;
            let mut vus_i: i32 = vus as i32;
            let mut hvd_i: i32 = hvd as i32;
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
            let block: u32 = hvd as u32;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_qwen_linear_rmsnorm_gated_f16.raw() as CUfunction,
                vus as u32,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn rmsnorm_gated launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // 12. out_proj FP8 GEMV → o_buf [hidden].
        let out_region = self
            .arena
            .region("qwen36_pl_out", (out_n as usize) * 2, 16)?;
        unsafe {
            self.fp8_proj_dispatch(
                kernel_gemv,
                out_region.device_ptr(),
                la.out_proj.offset_bytes,
                out_bs,
                gated_region.device_ptr(),
                m,
                out_n,
                out_k,
                stream_raw,
            )?;
        }
        // No fence: residual vector_add runs on the same stream
        // after out_proj.

        // 13. Residual sum: last_hidden_new = last_hidden + o_buf.
        // GPU residual: last_hidden += out_buf via vector_add_f16.
        // Replaces 2× DtoH + CPU loop + 1× HtoD with one launch.
        // Same numerics (`__hadd` is f16 RTNE, matching the previous
        // f16→f32→add→f16-RTNE pipeline byte-for-byte).
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let n_elem = (hidden as usize) * (m as usize);
            let mut dst = last_hidden_ptr;
            let mut src = out_region.device_ptr();
            let mut nn: i32 = n_elem as i32;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 1024.min(n_elem as u32).max(1);
            let grid = ((n_elem as u32 + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_vector_add_f16.raw() as CUfunction,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn residual vector_add_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // No function-exit fence: the next layer call (or the
        // outer-loop's end-of-prefill fence before lm_head) runs
        // on the same stream and same-stream ordering already
        // guarantees the residual write is visible.
        Ok(())
    }

    /// Phase Linear / Round-24: batched-prefill counterpart of
    /// `apply_layer_linear_attn`. Operates on `[num_tokens, hidden]`
    /// in the contiguous prompt-hidden region (`hidden_ptr` points at
    /// row 0). Equivalent to running the per-token function for each
    /// token sequentially — the recurrent linear-attn state evolves
    /// the same way — but compresses the launch chain:
    ///
    /// * Projections (in_proj_qkv / in_proj_z / out_proj) go through
    ///   `fp8_proj_dispatch` with `m = num_tokens`. At `m≥128` on
    ///   sm_121 that lands on the CUTLASS SM120 blockwise-FP8 GEMM,
    ///   replacing N×GEMV chatter with one GEMM.
    /// * `causal_conv1d_f16` already takes `seq_len`; we feed it
    ///   `num_tokens` directly with a flat `[num_tokens + ks-1, ts]`
    ///   history buffer built by `conv_state_advance_batched_f16`.
    /// * `gated_delta_rule_prefill_f16` carries the recurrent state
    ///   inside the kernel across all N tokens — one launch per
    ///   layer per chunk instead of N.
    /// * `rmsnorm_inplace_f16` natively accepts `num_tokens`; we
    ///   pass `num_tokens` directly.
    /// * `vector_add_f16` is element-wise; `n_elem = num_tokens *
    ///   hidden`.
    ///
    /// `silu_l2_gqa`, `alpha_beta`, and `rmsnorm_gated` are kept on
    /// the host-loop bridge for the v0 of this path (each takes one
    /// token; we loop them N times). They're independent across
    /// tokens, so a true batched kernel is mechanical follow-up
    /// work but produces the same numerics — the launch-overhead
    /// they consume is bounded and the wins from the four real
    /// batched calls above already give us most of the prefill
    /// speed-up. The bridge keeps this commit reviewable.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_linear_attn_batched(
        &self,
        la: &rvllm_loader::qwen36_weights::Qwen36LinearAttnLayer,
        linear_seq_idx: u32,
        hidden_ptr: u64,
        num_tokens: u32,
        kernel_gemv: rvllm_kernels::KernelFn,
        hidden: u32,
    ) -> Result<()> {
        if num_tokens == 0 {
            return Ok(());
        }
        if num_tokens == 1 {
            // Degenerate case — delegate to the per-token path so we
            // don't accidentally introduce a numerics divergence vs
            // the established greedy canary.
            return self.apply_layer_linear_attn(
                la,
                linear_seq_idx,
                hidden_ptr,
                kernel_gemv,
                hidden,
                (hidden as usize) * 2,
            );
        }
        let stream_raw = self.stream.raw() as u64;
        let n = num_tokens as usize;
        let h = hidden as usize;
        let qkv_n = la.in_proj_qkv.shape[0] as u32; // 8192
        let z_n = la.in_proj_z.shape[0] as u32; // 4096
        let out_n = la.out_proj.shape[0] as u32; // 2048
        let out_k = la.out_proj.shape[1] as u32; // 4096
        let qkv_bs = match la.in_proj_qkv.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let z_bs = match la.in_proj_z.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let out_bs = match la.out_proj.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };

        let num_k_heads: u32 = 16;
        let num_v_heads: u32 = 32;
        let head_k_dim: u32 = 128;
        let head_v_dim: u32 = 128;
        let key_dim = num_k_heads * head_k_dim;
        let v_per_k = num_v_heads / num_k_heads;
        let vus = num_v_heads as usize;
        let hkd = head_k_dim as usize;
        let hvd = head_v_dim as usize;
        let qk_bytes_per_token = vus * hkd * 2;
        let v_bytes_per_token = vus * hvd * 2;

        // 1. RMSNorm on a [N, hidden] copy of the chunk.
        let normed_bytes = n * h * 2;
        let normed_region = self.arena.region("qwen36_plb_normed", normed_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let r = cuMemcpyDtoDAsync_v2(
                normed_region.device_ptr(),
                hidden_ptr,
                normed_bytes,
                self.stream.raw() as _,
            );
            if r != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn_batched DtoD copy",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let eps = self.arch.base.rms_norm_eps;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                normed_region.device_ptr(),
                la.input_layernorm.offset_bytes,
                stream_raw,
            )?;
        }

        // 2. in_proj_qkv: [N, hidden] → [N, qkv_n].
        let qkv_region = self
            .arena
            .region("qwen36_plb_qkv", n * (qkv_n as usize) * 2, 16)?;
        unsafe {
            self.fp8_proj_dispatch(
                kernel_gemv,
                qkv_region.device_ptr(),
                la.in_proj_qkv.offset_bytes,
                qkv_bs,
                normed_region.device_ptr(),
                num_tokens,
                qkv_n,
                hidden,
                stream_raw,
            )?;
        }

        // 3. conv1d state advance + flat history assembly + conv1d.
        // Output of state-advance: [num_tokens + ks-1, qkv_n] with the
        // first ks-1 rows = current state, rest = current_qkv. State
        // is rotated to (s_{N-2}, s_{N-1}, s_N) inside the same kernel.
        let ks: u32 = 4;
        let conv_in_bytes = (n + (ks as usize - 1)) * (qkv_n as usize) * 2;
        let conv_in_region = self.arena.region("qwen36_plb_cin", conv_in_bytes, 16)?;
        let conv_out_bytes = n * (qkv_n as usize) * 2;
        let conv_out_region = self.arena.region("qwen36_plb_cout", conv_out_bytes, 16)?;
        let conv_state_ptr_layer = self.conv_state_layer_ptr(linear_seq_idx);
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut conv_in = conv_in_region.device_ptr();
            let mut state = conv_state_ptr_layer;
            let mut cur = qkv_region.device_ptr();
            let mut ts_i: i32 = qkv_n as i32;
            let mut nt_i: i32 = num_tokens as i32;
            let args = [
                (&mut conv_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut state) as *mut u64 as *mut core::ffi::c_void,
                (&mut cur) as *mut u64 as *mut core::ffi::c_void,
                (&mut ts_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut nt_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((qkv_n + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_conv_state_advance_batched_f16.raw() as CUfunction,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn_batched conv_state_advance",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // causal_conv1d with seq_len = num_tokens.
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut output = conv_out_region.device_ptr();
            let mut input = conv_in_region.device_ptr();
            let mut weight = la.conv1d.offset_bytes;
            let mut sl: i32 = num_tokens as i32;
            let mut ch: i32 = qkv_n as i32;
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
            let grid_x = (qkv_n + block - 1) / block;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_causal_conv1d_f16.raw() as CUfunction,
                grid_x,
                num_tokens,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn_batched causal_conv1d",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // 4+5. silu_l2_gqa per token (host-loop bridge). Each call
        // produces `[vus, head_k_dim] q + k` and `[vus, head_v_dim] v`
        // for a single token; we stride into per-token offsets within
        // the [N, vus, *] output regions.
        let q_region = self
            .arena
            .region("qwen36_plb_q", n * qk_bytes_per_token, 16)?;
        let k_region = self
            .arena
            .region("qwen36_plb_k", n * qk_bytes_per_token, 16)?;
        let v_region = self
            .arena
            .region("qwen36_plb_v", n * v_bytes_per_token, 16)?;
        // Single batched launch over (vus, num_tokens) — replaces
        // the per-token host loop. Kernel folds `blockIdx.y *
        // per_token_stride` into its pointer math; base pointers
        // are the start of the [N, …] buffers.
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut q_out = q_region.device_ptr();
            let mut k_out = k_region.device_ptr();
            let mut v_out = v_region.device_ptr();
            let mut conv_p = conv_out_region.device_ptr();
            let mut vus_i: i32 = vus as i32;
            let mut hkd_i: i32 = hkd as i32;
            let mut hvd_i: i32 = hvd as i32;
            let mut kd_i: i32 = key_dim as i32;
            let mut nvh: i32 = num_v_heads as i32;
            let mut vpk: i32 = v_per_k as i32;
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
            let block: u32 = (hkd.max(hvd)) as u32;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_qwen_linear_silu_l2_gqa_f16.raw() as CUfunction,
                vus as u32,
                num_tokens as u32,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn_batched silu_l2_gqa",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // 6. alpha/beta per token (host-loop bridge). Outputs
        // [num_tokens, vus] f32 each.
        let alpha_region = self.arena.region("qwen36_plb_alpha", n * vus * 4, 16)?;
        let beta_region = self.arena.region("qwen36_plb_beta", n * vus * 4, 16)?;
        // Single batched launch (vus, num_tokens). Kernel folds
        // per-token offsets into input / alpha_out / beta_out.
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut a_out = alpha_region.device_ptr();
            let mut b_out = beta_region.device_ptr();
            let mut a_w_p = la.in_proj_a.offset_bytes;
            let mut b_w_p = la.in_proj_b.offset_bytes;
            let mut a_log_p = la.a_log.offset_bytes;
            let mut dt_bias_p = la.dt_bias.offset_bytes;
            let mut in_p = normed_region.device_ptr();
            let mut vus_i: i32 = vus as i32;
            let mut h_i: i32 = h as i32;
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
            let block: u32 = 256u32.min(h as u32).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_qwen_linear_alpha_beta_f16.raw() as CUfunction,
                vus as u32,
                num_tokens as u32,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn_batched alpha_beta",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // 7. Batched delta-rule. One launch carries the recurrent
        // state across all N tokens internally.
        let layer_state_ptr = self.linear_state_layer_ptr(linear_seq_idx);
        let scale = 1.0f32 / (head_k_dim as f32).sqrt();
        let readout_region = self
            .arena
            .region("qwen36_plb_readout", n * v_bytes_per_token, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut state = layer_state_ptr;
            let mut q_ptr = q_region.device_ptr();
            let mut k_ptr = k_region.device_ptr();
            let mut v_ptr = v_region.device_ptr();
            let mut a_ptr = alpha_region.device_ptr();
            let mut b_ptr = beta_region.device_ptr();
            let mut o_ptr = readout_region.device_ptr();
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
            let smem = (2 * head_k_dim + head_v_dim) * 4;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_gated_delta_rule_prefill_f16.raw() as CUfunction,
                num_v_heads,
                1,
                1,
                head_v_dim,
                1,
                1,
                smem,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn_batched gated_delta_rule_prefill",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // 8. in_proj_z: [N, hidden] → [N, z_n].
        let z_region = self
            .arena
            .region("qwen36_plb_z", n * (z_n as usize) * 2, 16)?;
        unsafe {
            self.fp8_proj_dispatch(
                kernel_gemv,
                z_region.device_ptr(),
                la.in_proj_z.offset_bytes,
                z_bs,
                normed_region.device_ptr(),
                num_tokens,
                z_n,
                hidden,
                stream_raw,
            )?;
        }

        // 9. rmsnorm_gated per token (host-loop bridge). Each call
        // produces [vus, head_v_dim] from one token's readout + z
        // slices.
        let gated_region = self
            .arena
            .region("qwen36_plb_gated", n * vus * hvd * 2, 16)?;
        // Single batched launch (vus, num_tokens).
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut g_out = gated_region.device_ptr();
            let mut r_in = readout_region.device_ptr();
            let mut z_in = z_region.device_ptr();
            let mut gamma_p = la.norm.offset_bytes;
            let mut vus_i: i32 = vus as i32;
            let mut hvd_i: i32 = hvd as i32;
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
            let block: u32 = hvd as u32;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_qwen_linear_rmsnorm_gated_f16.raw() as CUfunction,
                vus as u32,
                num_tokens as u32,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn_batched rmsnorm_gated",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // 10. out_proj: [N, out_k] → [N, out_n].
        let out_region = self
            .arena
            .region("qwen36_plb_out", n * (out_n as usize) * 2, 16)?;
        unsafe {
            self.fp8_proj_dispatch(
                kernel_gemv,
                out_region.device_ptr(),
                la.out_proj.offset_bytes,
                out_bs,
                gated_region.device_ptr(),
                num_tokens,
                out_n,
                out_k,
                stream_raw,
            )?;
        }

        // 11. Residual: hidden_ptr += out_region (element-wise, [N*h]).
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let n_elem = n * h;
            let mut dst = hidden_ptr;
            let mut src = out_region.device_ptr();
            let mut nn: i32 = n_elem as i32;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 1024.min(n_elem as u32).max(1);
            let grid = ((n_elem as u32 + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_vector_add_f16.raw() as CUfunction,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn_batched residual vector_add",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        Ok(())
    }

    /// Phase Full / Round-26: batched-prefill counterpart of
    /// `apply_layer_full_attn`. Operates on `[num_tokens, hidden]`
    /// in `hidden_ptr`. One full-attn launch per layer instead of
    /// N. Reuses `flash_attention_2_f16kv_kernel` (the existing F32-Q
    /// / F16-KV / F32-O FA2 prefill kernel from
    /// kernels/flash_attention.cu) sandwiched between two cheap
    /// f16↔f32 cast launches; codex round-26 explicitly picked this
    /// over writing a brand-new f16-IO prefill kernel.
    ///
    /// Invariant: `positions[t] = start_pos + t` and the KV slot
    /// for token t equals `positions[t]` — both buffers were filled
    /// by the round-26 device-side fill kernel, so no host-driven
    /// HtoD races vs the non-blocking stream.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_full_attn_batched(
        &self,
        fl: &rvllm_loader::qwen36_weights::Qwen36FullAttnLayer,
        full_seq_idx: u32,
        hidden_ptr: u64,
        num_tokens: u32,
        kernel_gemv: rvllm_kernels::KernelFn,
        hidden: u32,
        last_hidden_bytes: usize,
        positions_dev_ptr: u64,
        prefill_ctx_len_dev_ptr: u64,
        full_layer_ordinal: u32,
    ) -> Result<()> {
        if num_tokens == 0 {
            return Ok(());
        }
        if num_tokens == 1 {
            // Degenerate: delegate to the per-token path so the
            // single-token decode case stays byte-identical.
            return self.apply_layer_full_attn(
                fl,
                full_seq_idx,
                0,
                hidden_ptr,
                kernel_gemv,
                hidden,
                last_hidden_bytes,
                positions_dev_ptr,
                prefill_ctx_len_dev_ptr,
            );
        }
        let stream_raw = self.stream.raw() as u64;
        let n = num_tokens as usize;
        let head_dim = self.arch.base.head_dim as u32;
        let num_heads = self.arch.base.num_attention_heads as u32;
        let num_kv_heads = self.arch.base.num_key_value_heads as u32;
        let q_n = fl.q_proj.shape[0] as u32;
        let k_n = fl.k_proj.shape[0] as u32;
        let v_n = fl.v_proj.shape[0] as u32;
        let o_n = fl.o_proj.shape[0] as u32;
        let o_k = fl.o_proj.shape[1] as u32;
        let q_bs = match fl.q_proj.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let k_bs = match fl.k_proj.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let v_bs = match fl.v_proj.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let o_bs = match fl.o_proj.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };

        let q_size = num_heads * head_dim;
        let _ = last_hidden_bytes;
        let h = hidden as usize;

        // 1. RMSNorm batched on a [N, hidden] copy of the chunk.
        let normed_bytes = n * h * 2;
        let normed_region = self.arena.region("qwen36_pfb_normed", normed_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let r = cuMemcpyDtoDAsync_v2(
                normed_region.device_ptr(),
                hidden_ptr,
                normed_bytes,
                self.stream.raw() as _,
            );
            if r != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 full_attn_batched DtoD copy",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let eps = self.arch.base.rms_norm_eps;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                normed_region.device_ptr(),
                fl.input_layernorm.offset_bytes,
                stream_raw,
            )?;
        }

        // 2. Q/K/V projections at m=num_tokens via dispatcher.
        let q_region = self
            .arena
            .region("qwen36_pfb_qg", n * (q_n as usize) * 2, 16)?;
        let k_region = self
            .arena
            .region("qwen36_pfb_k", n * (k_n as usize) * 2, 16)?;
        let v_region = self
            .arena
            .region("qwen36_pfb_v", n * (v_n as usize) * 2, 16)?;
        unsafe {
            self.fp8_proj_dispatch(
                kernel_gemv,
                q_region.device_ptr(),
                fl.q_proj.offset_bytes,
                q_bs,
                normed_region.device_ptr(),
                num_tokens,
                q_n,
                hidden,
                stream_raw,
            )?;
            self.fp8_proj_dispatch(
                kernel_gemv,
                k_region.device_ptr(),
                fl.k_proj.offset_bytes,
                k_bs,
                normed_region.device_ptr(),
                num_tokens,
                k_n,
                hidden,
                stream_raw,
            )?;
            self.fp8_proj_dispatch(
                kernel_gemv,
                v_region.device_ptr(),
                fl.v_proj.offset_bytes,
                v_bs,
                normed_region.device_ptr(),
                num_tokens,
                v_n,
                hidden,
                stream_raw,
            )?;
        }

        // 3. split_q_gate batched: kernel grid is (num_heads, m, 1).
        // m=num_tokens works out of the box.
        let qsize_us = q_size as usize;
        let q_split_region = self.arena.region("qwen36_pfb_qs", n * qsize_us * 2, 16)?;
        let gate_region = self.arena.region("qwen36_pfb_gt", n * qsize_us * 2, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut qo = q_split_region.device_ptr();
            let mut go = gate_region.device_ptr();
            let mut qi = q_region.device_ptr();
            let mut nh: i32 = num_heads as i32;
            let mut hd_i: i32 = head_dim as i32;
            let args = [
                (&mut qo) as *mut u64 as *mut core::ffi::c_void,
                (&mut go) as *mut u64 as *mut core::ffi::c_void,
                (&mut qi) as *mut u64 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_split_q_gate_f16.raw() as CUfunction,
                num_heads,
                num_tokens,
                1,
                head_dim,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 full_attn_batched split_q_gate",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // 4. q_norm + k_norm batched. The rmsnorm kernel processes
        // grid.x rows of length hidden each; treating each (token,
        // head) as one row gives us the needed per-head normalisation
        // across all tokens in one launch each.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_heads * num_tokens,
                hidden: head_dim,
                eps,
            }
            .launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                q_split_region.device_ptr(),
                fl.q_norm.offset_bytes,
                stream_raw,
            )?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_kv_heads * num_tokens,
                hidden: head_dim,
                eps,
            }
            .launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                k_region.device_ptr(),
                fl.k_norm.offset_bytes,
                stream_raw,
            )?;
        }

        // 5. RoPE + KV-cache write batched. Both Qwen RoPE kernels
        // have array semantics for positions/slot_mapping, so
        // grid.x=num_tokens drives the per-row slot. The NVFP4 branch
        // also quantizes Q to FP8 and writes per-(token, head) Q
        // descales for unified prefill.
        let rotary_dim = (head_dim as f32 * 0.25) as u32; // 64
        let kv_layer_ptr = self.kv_cache_layer_ptr(full_seq_idx);
        let half = (self.kv_cache_layer_bytes / 2) as u64;
        let k_cache_layer_ptr = kv_layer_ptr;
        let v_cache_layer_ptr = kv_layer_ptr + half;
        let scale_layer_ptr = self.kv_cache_scale_layer_ptr(full_seq_idx);
        let scale_half = (self.kv_cache_scale_layer_bytes / 2) as u64;
        let k_scale_layer_ptr = scale_layer_ptr;
        let v_scale_layer_ptr = if scale_layer_ptr == 0 {
            0
        } else {
            scale_layer_ptr + scale_half
        };
        let (q_fp8_ptr, q_scale_cache_ptr) = match self.kv_dtype {
            Qwen36KvDtype::F16 => (0u64, 0u64),
            Qwen36KvDtype::Nvfp4 => {
                let q_fp8_region = self.arena.region("qwen36_pfb_q_fp8", n * qsize_us, 16)?;
                let q_scale_region = self.arena.region(
                    "qwen36_pfb_q_scale_cache",
                    n * (num_heads as usize) * 4,
                    16,
                )?;
                (q_fp8_region.device_ptr(), q_scale_region.device_ptr())
            }
        };
        #[cfg(feature = "cuda")]
        match self.kv_dtype {
            Qwen36KvDtype::F16 => unsafe {
                use cudarc::driver::sys::*;
                let mut q_in_p = q_split_region.device_ptr();
                let mut k_in_p = k_region.device_ptr();
                let mut v_in_p = v_region.device_ptr();
                let mut q_out_p = q_split_region.device_ptr(); // in-place
                let mut kc = k_cache_layer_ptr;
                let mut vc = v_cache_layer_ptr;
                let mut cos_p = self.rope_cos;
                let mut sin_p = self.rope_sin;
                let mut pos_p = positions_dev_ptr;
                let mut slot_p = positions_dev_ptr; // slot==pos in qwen3-next
                let mut nt: i32 = num_tokens as i32;
                let mut nh: i32 = num_heads as i32;
                let mut nkh: i32 = num_kv_heads as i32;
                let mut hd_i: i32 = head_dim as i32;
                let mut rd: i32 = rotary_dim as i32;
                let args = [
                    (&mut q_in_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut k_in_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut v_in_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut q_out_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut kc) as *mut u64 as *mut core::ffi::c_void,
                    (&mut vc) as *mut u64 as *mut core::ffi::c_void,
                    (&mut cos_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut sin_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut pos_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut slot_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nkh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut rd) as *mut i32 as *mut core::ffi::c_void,
                ];
                let max_h = num_heads.max(num_kv_heads);
                let block_x: u32 = (head_dim / 2) as u32;
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_fused_rope_qwen_partial_f16kv.raw() as CUfunction,
                    num_tokens,
                    max_h,
                    1,
                    block_x,
                    1,
                    1,
                    0,
                    self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 full_attn_batched fused_rope",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            },
            Qwen36KvDtype::Nvfp4 => {
                let fn_rope = self
                    .outside_kernels
                    .fn_fused_rope_qwen_partial_nvfp4kv
                    .expect(
                        "qwen36 NVFP4 RoPE kernel not loaded — \
                             RVLLM_NVFP4_KV env gate inconsistency",
                    );
                unsafe {
                    use cudarc::driver::sys::*;
                    let mut q_in_p = q_split_region.device_ptr();
                    let mut k_in_p = k_region.device_ptr();
                    let mut v_in_p = v_region.device_ptr();
                    let mut q_fp8_out = q_fp8_ptr;
                    let mut key_packed = k_cache_layer_ptr;
                    let mut value_packed = v_cache_layer_ptr;
                    let mut key_scale = k_scale_layer_ptr;
                    let mut value_scale = v_scale_layer_ptr;
                    let mut cos_p = self.rope_cos;
                    let mut sin_p = self.rope_sin;
                    let mut pos_p = positions_dev_ptr;
                    let mut slot_p = positions_dev_ptr;
                    let mut q_scale_static = q_scale_cache_ptr;
                    let mut q_scale_dyn = q_scale_cache_ptr;
                    let mut nt: i32 = num_tokens as i32;
                    let mut nh: i32 = num_heads as i32;
                    let mut nkh: i32 = num_kv_heads as i32;
                    let mut hd_i: i32 = head_dim as i32;
                    let mut rd: i32 = rotary_dim as i32;
                    let args = [
                        (&mut q_in_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut k_in_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut v_in_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut q_fp8_out) as *mut u64 as *mut core::ffi::c_void,
                        (&mut key_packed) as *mut u64 as *mut core::ffi::c_void,
                        (&mut value_packed) as *mut u64 as *mut core::ffi::c_void,
                        (&mut key_scale) as *mut u64 as *mut core::ffi::c_void,
                        (&mut value_scale) as *mut u64 as *mut core::ffi::c_void,
                        (&mut cos_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut sin_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut pos_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut slot_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut q_scale_static) as *mut u64 as *mut core::ffi::c_void,
                        (&mut q_scale_dyn) as *mut u64 as *mut core::ffi::c_void,
                        (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                        (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                        (&mut nkh) as *mut i32 as *mut core::ffi::c_void,
                        (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
                        (&mut rd) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let grid_y = num_heads.max(num_kv_heads);
                    let rc = cuLaunchKernel(
                        fn_rope.raw() as CUfunction,
                        num_tokens,
                        grid_y,
                        1,
                        head_dim,
                        1,
                        1,
                        0,
                        self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 full_attn_batched fused_rope_qwen_nvfp4",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
            }
        }

        // 6. Attention. F16 KV keeps the existing prefill kernel
        // and casts around its f32 ABI. NVFP4 uses the unified prefill
        // kernel directly over all prompt rows and returns f16 output.
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let attn_out_region = match self.kv_dtype {
            Qwen36KvDtype::F16 => {
                let q_f32_bytes = n * qsize_us * 4;
                let q_f32_region = self.arena.region("qwen36_pfb_qf32", q_f32_bytes, 16)?;
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let n_elem = n * qsize_us;
                    let mut output = q_f32_region.device_ptr();
                    let mut input = q_split_region.device_ptr();
                    let mut nn: i32 = n_elem as i32;
                    let args = [
                        (&mut output) as *mut u64 as *mut core::ffi::c_void,
                        (&mut input) as *mut u64 as *mut core::ffi::c_void,
                        (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let block: u32 = 256;
                    let grid = ((n_elem as u32 + block - 1) / block).max(1);
                    let rc = cuLaunchKernel(
                        self.outside_kernels.fn_cast_f16_to_f32.raw() as CUfunction,
                        grid,
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 full_attn_batched q cast f16->f32",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                let attn_out_f32_region =
                    self.arena.region("qwen36_pfb_attn_f32", q_f32_bytes, 16)?;
                let seq_start_region = self.arena.region("qwen36_pfb_seqstart", 2 * 4, 16)?;
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let r0 = cuMemsetD32Async(
                        seq_start_region.device_ptr(),
                        0,
                        1,
                        self.stream.raw() as _,
                    );
                    let r1 = cuMemsetD32Async(
                        seq_start_region.device_ptr() + 4,
                        num_tokens,
                        1,
                        self.stream.raw() as _,
                    );
                    if r0 != CUresult::CUDA_SUCCESS || r1 != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 full_attn_batched seq_start_pos memset",
                            rvllm_core::CudaErrorKind::MemcpyFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                const FA2_THREADS: i32 = 128;
                const FA2_BC: i32 = 32;
                let smem_bytes =
                    2 * FA2_BC * head_dim as i32 * 4 + FA2_BC * 4 + (FA2_THREADS / 32) * 4;
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    if smem_bytes as u32 >= 48 * 1024 {
                        let _ = cuFuncSetAttribute(
                            self.outside_kernels.fn_flash_attention_2_f16kv.raw() as CUfunction,
                            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            smem_bytes,
                        );
                    }
                    let mut output = attn_out_f32_region.device_ptr();
                    let mut query = q_f32_region.device_ptr();
                    let mut key_cache = k_cache_layer_ptr;
                    let mut value_cache = v_cache_layer_ptr;
                    let mut block_tables = self.bt_persistent_ptr;
                    let mut context_lens = prefill_ctx_len_dev_ptr;
                    let mut seq_start_pos = seq_start_region.device_ptr();
                    let mut scale_arg = scale;
                    let mut nh = num_heads as i32;
                    let mut nkvh = num_kv_heads as i32;
                    let mut hd = head_dim as i32;
                    let mut bs = self.kv_cache_block_size as i32;
                    let mut max_ctx = num_tokens as i32;
                    let mut mbps = self.kv_cache_num_blocks as i32;
                    let mut nqt = num_tokens as i32;
                    let mut causal: i32 = 1;
                    let args = [
                        (&mut output) as *mut u64 as *mut core::ffi::c_void,
                        (&mut query) as *mut u64 as *mut core::ffi::c_void,
                        (&mut key_cache) as *mut u64 as *mut core::ffi::c_void,
                        (&mut value_cache) as *mut u64 as *mut core::ffi::c_void,
                        (&mut block_tables) as *mut u64 as *mut core::ffi::c_void,
                        (&mut context_lens) as *mut u64 as *mut core::ffi::c_void,
                        (&mut seq_start_pos) as *mut u64 as *mut core::ffi::c_void,
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
                        self.outside_kernels.fn_flash_attention_2_f16kv.raw() as CUfunction,
                        1u32,
                        num_heads,
                        1,
                        FA2_THREADS as u32,
                        1,
                        1,
                        smem_bytes as u32,
                        self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 full_attn_batched flash_attention_2_f16kv",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                let attn_out_region =
                    self.arena
                        .region("qwen36_pfb_attn_f16", n * qsize_us * 2, 16)?;
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let n_elem = n * qsize_us;
                    let mut output = attn_out_region.device_ptr();
                    let mut input = attn_out_f32_region.device_ptr();
                    let mut nn: i32 = n_elem as i32;
                    let args = [
                        (&mut output) as *mut u64 as *mut core::ffi::c_void,
                        (&mut input) as *mut u64 as *mut core::ffi::c_void,
                        (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let block: u32 = 256;
                    let grid = ((n_elem as u32 + block - 1) / block).max(1);
                    let rc = cuLaunchKernel(
                        self.outside_kernels.fn_cast_f32_to_f16.raw() as CUfunction,
                        grid,
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 full_attn_batched out cast f32->f16",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                attn_out_region
            }
            Qwen36KvDtype::Nvfp4 => {
                let attn_out_region =
                    self.arena
                        .region("qwen36_pfb_attn_f16", n * qsize_us * 2, 16)?;
                let cu_seqlens_region = self.arena.region("qwen36_pfb_cu_seqlens", 2 * 4, 16)?;
                unsafe {
                    let cu: [i32; 2] = [0, num_tokens as i32];
                    let bytes = std::slice::from_raw_parts(
                        cu.as_ptr() as *const u8,
                        std::mem::size_of_val(&cu),
                    );
                    cu_seqlens_region.copy_from_host(bytes)?;
                    let params = rvllm_attention::PagedPrefillParams {
                        num_seqs: 1,
                        num_tokens,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        block_size: self.kv_cache_block_size,
                        max_blocks_per_seq: self.kv_cache_num_blocks,
                        num_blocks_total: self.kv_cache_num_blocks,
                        scale,
                        window_size_left: -1,
                    };
                    let num_queries_per_kv = num_heads / num_kv_heads;
                    let unified = rvllm_attention::UnifiedPrefillParams {
                        num_queries_per_kv,
                        tile_size: if head_dim <= 256 { 32 } else { 16 },
                        block_q: (rvllm_attention::UNIFIED_PREFILL_BLOCK_M
                            / num_queries_per_kv.max(1))
                        .max(1),
                        use_mma: true,
                    };
                    let prefill =
                        rvllm_attention::PagedPrefillNvfp4Launcher::new(&self.attn_backend_full);
                    let unified_from = std::env::var("RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_FROM")
                        .ok()
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(0);
                    // Default-ON since 2026-05-22. Mirrors the Gemma 4
                    // NVFP4 (Option B) and qwen35 dense paths which call
                    // `launch_nvfp4kv_unified_sm121` unconditionally.
                    // Operator opts OUT via `=0` for A/B-vs-per-token
                    // diagnostics. Prior FROM=5 default was the only
                    // reason the 35B-A3B profile carried this env
                    // explicitly (per CLAUDE.md "resolved 2026-05-20"
                    // note); with FULL_PREFILL default-on the FROM
                    // selector still applies (default 0 = every layer).
                    let use_unified = full_layer_ordinal >= unified_from
                        && std::env::var("RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_PREFILL")
                            .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                            .unwrap_or(true);
                    if use_unified {
                        prefill.launch_nvfp4kv_unified_sm121(
                            params,
                            unified,
                            attn_out_region.device_ptr(),
                            q_fp8_ptr,
                            k_cache_layer_ptr,
                            v_cache_layer_ptr,
                            k_scale_layer_ptr,
                            v_scale_layer_ptr,
                            q_scale_cache_ptr,
                            self.bt_persistent_ptr,
                            cu_seqlens_region.device_ptr(),
                            prefill_ctx_len_dev_ptr,
                            q_scale_cache_ptr,
                            false,
                            stream_raw,
                        )?;
                    } else {
                        prefill.launch(
                            params,
                            attn_out_region.device_ptr(),
                            q_fp8_ptr,
                            k_cache_layer_ptr,
                            v_cache_layer_ptr,
                            k_scale_layer_ptr,
                            v_scale_layer_ptr,
                            q_scale_cache_ptr,
                            self.bt_persistent_ptr,
                            prefill_ctx_len_dev_ptr,
                            cu_seqlens_region.device_ptr(),
                            q_scale_cache_ptr,
                            num_tokens,
                            stream_raw,
                        )?;
                    }
                }
                attn_out_region
            }
        };

        // 7. attn_output_gate (sigmoid_mul) batched: n_elem = N * q_size
        let gated_region = self
            .arena
            .region("qwen36_pfb_gated", n * qsize_us * 2, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let n_elem = n * qsize_us;
            let mut output = gated_region.device_ptr();
            let mut values = attn_out_region.device_ptr();
            let mut gate = gate_region.device_ptr();
            let mut nn: i32 = n_elem as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut values) as *mut u64 as *mut core::ffi::c_void,
                (&mut gate) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = ((n_elem as u32 + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_sigmoid_mul_f16.raw() as CUfunction,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 full_attn_batched sigmoid_mul",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // 8. o_proj: [N, o_k] → [N, o_n] via dispatcher.
        let out_region = self
            .arena
            .region("qwen36_pfb_out", n * (o_n as usize) * 2, 16)?;
        unsafe {
            self.fp8_proj_dispatch(
                kernel_gemv,
                out_region.device_ptr(),
                fl.o_proj.offset_bytes,
                o_bs,
                gated_region.device_ptr(),
                num_tokens,
                o_n,
                o_k,
                stream_raw,
            )?;
        }

        // 9. Residual: hidden_ptr += out_region elementwise [N*hidden].
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let n_elem = n * h;
            let mut dst = hidden_ptr;
            let mut src = out_region.device_ptr();
            let mut nn: i32 = n_elem as i32;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 1024.min(n_elem as u32).max(1);
            let grid = ((n_elem as u32 + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_vector_add_f16.raw() as CUfunction,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 full_attn_batched residual",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Phase 4z/5b helper: full-attention layer forward on a single
    /// token at `position` (0-indexed). Composes input_layernorm →
    /// q/k/v_proj → host-deinterleave Q+gate → q/k_norm → RoPE +
    /// KV-cache write at slot=position → paged FA2 decode with
    /// context_len=position+1 (causal) → sigmoid_mul gate → o_proj →
    /// residual sum into last_hidden.
    #[allow(clippy::too_many_arguments)]
    /// `last_hidden_ptr`: see `apply_layer_linear_attn` doc — same
    /// Phase-1 contract.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_full_attn(
        &self,
        fl: &rvllm_loader::qwen36_weights::Qwen36FullAttnLayer,
        full_seq_idx: u32,
        position: u32,
        last_hidden_ptr: u64,
        kernel_gemv: rvllm_kernels::KernelFn,
        hidden: u32,
        last_hidden_bytes: usize,
        // Phase 4b-prep iter35: caller hoists the pos+context_len
        // HtoD out of the per-layer loop. Pass the already-uploaded
        // device pointers (pos_dev_ptr is also re-used as slot_ptr,
        // see iter24).
        pos_dev_ptr: u64,
        cl_dev_ptr: u64,
    ) -> Result<()> {
        let stream_raw = self.stream.raw() as u64;
        let head_dim = self.arch.base.head_dim as u32;
        let num_heads = self.arch.base.num_attention_heads as u32; // 16
        let num_kv_heads = self.arch.base.num_key_value_heads as u32; // 2
        let q_n = fl.q_proj.shape[0] as u32; // 8192 = num_heads*head_dim*2
        let k_n = fl.k_proj.shape[0] as u32; // 512
        let v_n = fl.v_proj.shape[0] as u32; // 512
        let o_n = fl.o_proj.shape[0] as u32; // 2048
        let o_k = fl.o_proj.shape[1] as u32; // 4096
        let m: u32 = 1;
        let q_bs = match fl.q_proj.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let k_bs = match fl.k_proj.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let v_bs = match fl.v_proj.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let o_bs = match fl.o_proj.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };

        // 1. input_layernorm on copy of last_hidden.
        let normed_region = self
            .arena
            .region("qwen36_pf_normed", last_hidden_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoDAsync_v2(
                normed_region.device_ptr(),
                last_hidden_ptr,
                last_hidden_bytes,
                self.stream.raw() as _,
            );
        }
        let eps = self.arch.base.rms_norm_eps;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                normed_region.device_ptr(),
                fl.input_layernorm.offset_bytes,
                stream_raw,
            )?;
        }
        // No fence: q/k/v projections run on the same stream.

        // Phase 8 QKV megakernel Phase 2: env-gated opt-in. When
        // set and the KV-cache dtype is F16, ALL of the proj +
        // split + norm + RoPE + KV-write chain (steps 2-5 below)
        // is replaced by a single megakernel launch. Default off
        // so production paths stay on the existing chain.
        let qkv_megakernel_on =
            std::env::var("RVLLM_QWEN36_QKV_MEGAKERNEL").as_deref() == Ok("1")
                && matches!(self.kv_dtype, Qwen36KvDtype::F16);

        // 2. q_proj, k_proj, v_proj GEMVs.
        let q_region = self.arena.region("qwen36_pf_qg", (q_n as usize) * 2, 16)?;
        let k_region = self.arena.region("qwen36_pf_k", (k_n as usize) * 2, 16)?;
        let v_region = self.arena.region("qwen36_pf_v", (v_n as usize) * 2, 16)?;
        let _ = (&q_region, &k_region, &v_region);
        // Phase 4a: route projections through `fp8_proj_dispatch`.
        // At m=1 (today's caller) this dispatches to the same
        // Fp8GemvF16InLaunch byte-identically. Phase 4b/5/7 will
        // flip m=1 → m=num_tokens and the dispatcher will pick up
        // CUTLASS SM120 (m≥128) automatically — no further edits
        // needed in this function.
        if !qkv_megakernel_on {
            unsafe {
                self.fp8_proj_dispatch(
                    kernel_gemv,
                    q_region.device_ptr(),
                    fl.q_proj.offset_bytes,
                    q_bs,
                    normed_region.device_ptr(),
                    m,
                    q_n,
                    hidden,
                    stream_raw,
                )?;
                self.fp8_proj_dispatch(
                    kernel_gemv,
                    k_region.device_ptr(),
                    fl.k_proj.offset_bytes,
                    k_bs,
                    normed_region.device_ptr(),
                    m,
                    k_n,
                    hidden,
                    stream_raw,
                )?;
                self.fp8_proj_dispatch(
                    kernel_gemv,
                    v_region.device_ptr(),
                    fl.v_proj.offset_bytes,
                    v_bs,
                    normed_region.device_ptr(),
                    m,
                    v_n,
                    hidden,
                    stream_raw,
                )?;
            }
        }
        // No fence: split_q_gate runs on the same stream.

        // 3. GPU split of q_proj output [num_tokens, num_heads, 2*head_dim]
        //    into Q + gate [num_tokens, num_heads, head_dim] each.
        //    Replaces a DtoH + CPU per-head copy_from_slice + HtoD
        //    round-trip per token with one launch.
        let q_size = (num_heads * head_dim) as usize; // 4096
        let _hd = head_dim as usize;
        let q_split_region = self
            .arena
            .region("qwen36_pf_qs", q_size * (m as usize) * 2, 16)?;
        let gate_region = self
            .arena
            .region("qwen36_pf_gt", q_size * (m as usize) * 2, 16)?;
        if !qkv_megakernel_on {
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut qo = q_split_region.device_ptr();
                let mut go = gate_region.device_ptr();
                let mut qi = q_region.device_ptr();
                let mut nh: i32 = num_heads as i32;
                let mut hd_i: i32 = head_dim as i32;
                let args = [
                    (&mut qo) as *mut u64 as *mut core::ffi::c_void,
                    (&mut go) as *mut u64 as *mut core::ffi::c_void,
                    (&mut qi) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
                ];
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_split_q_gate_f16.raw() as CUfunction,
                    num_heads,
                    m,
                    1,
                    head_dim,
                    1,
                    1,
                    0,
                    self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 full_attn split_q_gate launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }

        // 4. q_norm + k_norm — Phase 8 QKV-megakernel Phase 1:
        // FUSED into the subsequent RoPE+KV-write kernel for BOTH
        // F16-KV (`fn_fused_qnorm_knorm_rope_qwen_partial_f16kv`,
        // commit 943f8bb) AND NVFP4-KV
        // (`fn_fused_qnorm_knorm_rope_qwen_partial_nvfp4kv`,
        // this commit's sibling). The standalone
        // rmsnorm_inplace launches that previously ran here are
        // now retired on both KV-dtype paths.
        // No fence: fused_rope runs on the same stream.

        // 5. NeoX-style partial RoPE on GPU + KV-cache write.
        //
        // Replaces the previous host pipeline (DtoH cos/sin, DtoH q/k/v,
        // CPU NeoX rotation, HtoD rotated Q, HtoD K+V to cache slots —
        // 5+ round-trips per token + a 16-head × 32-element CPU loop)
        // with one kernel launch. `fused_rope_qwen_partial_f16kv`
        // pairs `(i, i + rotary_dim/2)` within the first `rotary_dim`
        // elements of each head — Qwen's partial-NeoX convention,
        // distinct from the Gemma kernel's `(i, i + head_dim/2)`
        // pairing.
        let rotary_dim = (head_dim as f32 * 0.25) as u32; // 64
        let kv_layer_ptr = self.kv_cache_layer_ptr(full_seq_idx);
        let half = (self.kv_cache_layer_bytes / 2) as u64;
        let k_cache_layer_ptr = kv_layer_ptr;
        let v_cache_layer_ptr = kv_layer_ptr + half;
        // NVFP4 commit 3: companion scale buffer layer pointer.
        // Zero on F16; on Nvfp4 split into K/V halves the same way
        // as the packed cache (K then V).
        let scale_layer_ptr = self.kv_cache_scale_layer_ptr(full_seq_idx);
        let scale_half = (self.kv_cache_scale_layer_bytes / 2) as u64;
        let k_scale_layer_ptr = scale_layer_ptr;
        let v_scale_layer_ptr = if scale_layer_ptr == 0 {
            0
        } else {
            scale_layer_ptr + scale_half
        };
        // NVFP4 commit 3: per-call FP8 Q + dynamic Q scale scratch.
        // Only allocated on the Nvfp4 path. Decode is num_tokens=1,
        // so q_fp8 is [num_heads * head_dim] u8 and q_scale_cache
        // is [num_heads] f32.
        let (q_fp8_ptr, q_scale_cache_ptr) = match self.kv_dtype {
            Qwen36KvDtype::F16 => (0u64, 0u64),
            Qwen36KvDtype::Nvfp4 => {
                let r_fp8 =
                    self.arena
                        .region("qwen36_pf_q_fp8", (num_heads * head_dim) as usize, 16)?;
                let r_sc =
                    self.arena
                        .region("qwen36_pf_q_scale_cache", (num_heads as usize) * 4, 16)?;
                (r_fp8.device_ptr(), r_sc.device_ptr())
            }
        };
        // Phase 4b-prep iter35: caller-hoisted pos+cl HtoD; we just
        // use the device pointers passed in.
        let _ = position; // RoPE reads from pos_dev_ptr instead.
        let slot_dev_ptr = pos_dev_ptr;
        // NVFP4 commit 3: dispatch the RoPE+KV-write kernel by dtype.
        // F16 path unchanged; Nvfp4 path uses the Qwen-specific NVFP4
        // RoPE kernel from `fused_rope_qwen_partial_nvfp4kv.cu`,
        // shared with the Qwen 3.5 27B NVFP4 wiring.
        #[cfg(feature = "cuda")]
        match self.kv_dtype {
            Qwen36KvDtype::F16 if qkv_megakernel_on => unsafe {
                // Phase 8 QKV-megakernel Phase 2 (F16-KV): single-
                // launch fusion of Q+K+V projections + norm + RoPE
                // + KV-write. Skipped steps 2-5 of the unfused
                // chain (gated on `RVLLM_QWEN36_QKV_MEGAKERNEL=1`).
                use cudarc::driver::sys::*;
                let mut input_p = normed_region.device_ptr();
                let mut w_q = fl.q_proj.offset_bytes;
                let mut w_k = fl.k_proj.offset_bytes;
                let mut w_v = fl.v_proj.offset_bytes;
                let mut s_q = q_bs;
                let mut s_k = k_bs;
                let mut s_v = v_bs;
                let mut q_out_p = q_split_region.device_ptr();
                let mut gate_out_p = gate_region.device_ptr();
                let mut kc = k_cache_layer_ptr;
                let mut vc = v_cache_layer_ptr;
                let mut cos_p = self.rope_cos;
                let mut sin_p = self.rope_sin;
                let mut qn_p = fl.q_norm.offset_bytes;
                let mut kn_p = fl.k_norm.offset_bytes;
                let mut pos_p = pos_dev_ptr;
                let mut slot_p = slot_dev_ptr;
                let mut nt: i32 = m as i32;
                let mut nh: i32 = num_heads as i32;
                let mut nkh: i32 = num_kv_heads as i32;
                let mut hd_i: i32 = head_dim as i32;
                let mut hi: i32 = hidden as i32;
                let mut rd: i32 = rotary_dim as i32;
                let mut ncb_q: i32 = (hidden as i32) / 128;
                let mut ncb_kv: i32 = (hidden as i32) / 128;
                let mut eps_f: f32 = eps;
                let args = [
                    (&mut input_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut w_q) as *mut u64 as *mut core::ffi::c_void,
                    (&mut w_k) as *mut u64 as *mut core::ffi::c_void,
                    (&mut w_v) as *mut u64 as *mut core::ffi::c_void,
                    (&mut s_q) as *mut u64 as *mut core::ffi::c_void,
                    (&mut s_k) as *mut u64 as *mut core::ffi::c_void,
                    (&mut s_v) as *mut u64 as *mut core::ffi::c_void,
                    (&mut q_out_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut gate_out_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut kc) as *mut u64 as *mut core::ffi::c_void,
                    (&mut vc) as *mut u64 as *mut core::ffi::c_void,
                    (&mut cos_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut sin_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut qn_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut kn_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut pos_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut slot_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nkh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut hi) as *mut i32 as *mut core::ffi::c_void,
                    (&mut rd) as *mut i32 as *mut core::ffi::c_void,
                    (&mut ncb_q) as *mut i32 as *mut core::ffi::c_void,
                    (&mut ncb_kv) as *mut i32 as *mut core::ffi::c_void,
                    (&mut eps_f) as *mut f32 as *mut core::ffi::c_void,
                ];
                let grid_y = num_heads + 2 * num_kv_heads;
                let block_x: u32 = (head_dim * 2) as u32;
                let rc = cuLaunchKernel(
                    self.outside_kernels
                        .fn_fused_qkv_proj_qnorm_knorm_rope_qwen_partial_f16kv
                        .raw() as CUfunction,
                    m as u32, grid_y, 1,
                    block_x, 1, 1,
                    0,
                    self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 full_attn qkv_megakernel launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            },
            Qwen36KvDtype::F16 => unsafe {
                // Phase 8 QKV-megakernel Phase 1: Q-norm + K-norm
                // FUSED into the RoPE+KV-write kernel via two
                // extra arg pointers (q_norm_weight, k_norm_weight)
                // + eps. Block geometry unchanged from the unfused
                // RoPE kernel. Per-head block-reduces sum-of-
                // squares across head_dim, computes inverse rms,
                // applies gamma-scaled normalisation, then the
                // existing partial-NeoX rotation.
                use cudarc::driver::sys::*;
                let mut q_in_p = q_split_region.device_ptr();
                let mut k_in_p = k_region.device_ptr();
                let mut v_in_p = v_region.device_ptr();
                let mut q_out_p = q_split_region.device_ptr(); // in-place ok
                let mut kc = k_cache_layer_ptr;
                let mut vc = v_cache_layer_ptr;
                let mut cos_p = self.rope_cos;
                let mut sin_p = self.rope_sin;
                let mut qn_p = fl.q_norm.offset_bytes;
                let mut kn_p = fl.k_norm.offset_bytes;
                let mut pos_p = pos_dev_ptr;
                let mut slot_p = slot_dev_ptr;
                let mut nt: i32 = m as i32;
                let mut nh: i32 = num_heads as i32;
                let mut nkh: i32 = num_kv_heads as i32;
                let mut hd_i: i32 = head_dim as i32;
                let mut rd: i32 = rotary_dim as i32;
                let mut eps_f: f32 = eps;
                let args = [
                    (&mut q_in_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut k_in_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut v_in_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut q_out_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut kc) as *mut u64 as *mut core::ffi::c_void,
                    (&mut vc) as *mut u64 as *mut core::ffi::c_void,
                    (&mut cos_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut sin_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut qn_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut kn_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut pos_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut slot_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nkh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut rd) as *mut i32 as *mut core::ffi::c_void,
                    (&mut eps_f) as *mut f32 as *mut core::ffi::c_void,
                ];
                let max_h = num_heads.max(num_kv_heads);
                let block_x: u32 = (head_dim / 2) as u32;
                let rc = cuLaunchKernel(
                    self.outside_kernels
                        .fn_fused_qnorm_knorm_rope_qwen_partial_f16kv
                        .raw() as CUfunction,
                    m as u32,
                    max_h,
                    1,
                    block_x,
                    1,
                    1,
                    0,
                    self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 full_attn fused_rope_qwen launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            },
            Qwen36KvDtype::Nvfp4 => {
                // Phase 8 QKV-megakernel Phase 1 NVFP4 sibling: Q-norm
                // + K-norm fused into the NVFP4 RoPE + FP8-Q + NVFP4-KV
                // kernel. Same 2-phase per-head body as the F16 sibling
                // (commit 943f8bb) — block-reduces sum-of-squares,
                // applies gamma*inv_norm into a shared-mem buffer,
                // then standard rotation reads normalised values from
                // shared mem and feeds the existing FP8/NVFP4 quantise
                // epilogue.
                let fn_rope = self
                    .outside_kernels
                    .fn_fused_qnorm_knorm_rope_qwen_partial_nvfp4kv
                    .expect(
                        "qwen36 NVFP4 fused qnorm+knorm+RoPE kernel \
                         not loaded — RVLLM_NVFP4_KV env gate \
                         inconsistency",
                    );
                unsafe {
                    use cudarc::driver::sys::*;
                    let mut q_in_p = q_split_region.device_ptr();
                    let mut k_in_p = k_region.device_ptr();
                    let mut v_in_p = v_region.device_ptr();
                    let mut q_fp8_out = q_fp8_ptr;
                    let mut key_packed = k_cache_layer_ptr;
                    let mut value_packed = v_cache_layer_ptr;
                    let mut key_scale = k_scale_layer_ptr;
                    let mut value_scale = v_scale_layer_ptr;
                    let mut cos_p = self.rope_cos;
                    let mut sin_p = self.rope_sin;
                    let mut qn_p = fl.q_norm.offset_bytes;
                    let mut kn_p = fl.k_norm.offset_bytes;
                    let mut pos_p = pos_dev_ptr;
                    let mut slot_p = slot_dev_ptr;
                    let mut q_scale_static = q_scale_cache_ptr;
                    let mut q_scale_dyn = q_scale_cache_ptr;
                    let mut nt: i32 = m as i32;
                    let mut nh: i32 = num_heads as i32;
                    let mut nkh: i32 = num_kv_heads as i32;
                    let mut hd_i: i32 = head_dim as i32;
                    let mut rd: i32 = rotary_dim as i32;
                    let mut eps_f: f32 = eps;
                    let args = [
                        (&mut q_in_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut k_in_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut v_in_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut q_fp8_out) as *mut u64 as *mut core::ffi::c_void,
                        (&mut key_packed) as *mut u64 as *mut core::ffi::c_void,
                        (&mut value_packed) as *mut u64 as *mut core::ffi::c_void,
                        (&mut key_scale) as *mut u64 as *mut core::ffi::c_void,
                        (&mut value_scale) as *mut u64 as *mut core::ffi::c_void,
                        (&mut cos_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut sin_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut qn_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut kn_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut pos_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut slot_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut q_scale_static) as *mut u64 as *mut core::ffi::c_void,
                        (&mut q_scale_dyn) as *mut u64 as *mut core::ffi::c_void,
                        (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                        (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                        (&mut nkh) as *mut i32 as *mut core::ffi::c_void,
                        (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
                        (&mut rd) as *mut i32 as *mut core::ffi::c_void,
                        (&mut eps_f) as *mut f32 as *mut core::ffi::c_void,
                    ];
                    let grid_y = num_heads.max(num_kv_heads);
                    let rc = cuLaunchKernel(
                        fn_rope.raw() as CUfunction,
                        m as u32,
                        grid_y,
                        1,
                        head_dim,
                        1,
                        1,
                        0,
                        self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 full_attn fused_qnorm_knorm_rope_nvfp4 launch",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
            }
        }
        // No fence: paged FA2 decode runs on the same stream after
        // the fused_rope kernel writes Q (in-place into q_split_region)
        // and K/V into the cache slots.
        // (Phase-4b prep) Q-rotation, K-rotation, KV-cache write all
        // happened in the GPU launch above; the previous host-side
        // pipeline (DtoH cos/sin + DtoH q/k/v + CPU NeoX rotation +
        // HtoD rotated Q + HtoD K, V to cache slots) is gone.

        // 6. Paged FA2 decode. block_tables=identity-mapping
        //    [0, 1, 2, ..., max_blocks_per_seq-1] since each logical
        //    block in our single-sequence cache maps to itself in
        //    physical layout. context_lens=[position+1] (causal —
        //    attend to all prior tokens).
        // Phase 4b-prep iter25/26: identity block table is uploaded
        // once at bring-up; `context_len` was packed with `position`
        // into pos_region above (one combined HtoD per layer).
        let attn_out_region = self.arena.region("qwen36_pf_attn_out", q_size * 2, 16)?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        // NVFP4 commit 3: decode-attention dispatch.
        //   * F16:   existing flash_attention_2_decode_f16io (smem
        //            uses f32 K/V tiles → 2*BC*hd*4 bytes).
        //   * NVFP4: flash_attention_2_decode_nvfp4kv. K/V dequant
        //            target is f16 smem (2 bytes/elem; halves the
        //            K/V tile footprint vs F16 → 32 KiB at hd=256).
        #[cfg(feature = "cuda")]
        match self.kv_dtype {
            Qwen36KvDtype::F16 => unsafe {
                use cudarc::driver::sys::*;
                const FA2_THREADS: i32 = 128;
                const FA2_BC: i32 = 32;
                let hd_i = head_dim as i32;
                let smem_bytes = 2 * FA2_BC * hd_i * 4 + FA2_BC * 4 + (FA2_THREADS / 32) * 4;
                if smem_bytes as u32 >= 48 * 1024 {
                    let _ = cuFuncSetAttribute(
                        self.outside_kernels.fn_flash_attention_2_decode_f16io.raw() as CUfunction,
                        CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        smem_bytes,
                    );
                }
                let mut output = attn_out_region.device_ptr();
                let mut query = q_split_region.device_ptr();
                let mut key_cache = k_cache_layer_ptr;
                let mut value_cache = v_cache_layer_ptr;
                let mut block_tables = self.bt_persistent_ptr;
                let mut context_lens = cl_dev_ptr;
                let mut scale_arg = scale;
                let mut nh = num_heads as i32;
                let mut nkvh = num_kv_heads as i32;
                let mut hd = head_dim as i32;
                let mut bs = self.kv_cache_block_size as i32;
                let mut mbps = self.kv_cache_num_blocks as i32;
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
                    self.outside_kernels.fn_flash_attention_2_decode_f16io.raw() as CUfunction,
                    1,
                    num_heads,
                    1,
                    FA2_THREADS as u32,
                    1,
                    1,
                    smem_bytes as u32,
                    self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 flash_attention_2_decode_f16io launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            },
            Qwen36KvDtype::Nvfp4 => {
                let fn_dec = self
                    .outside_kernels
                    .fn_flash_attention_2_decode_nvfp4kv
                    .expect(
                        "qwen36 NVFP4 decode kernel not loaded — \
                         RVLLM_NVFP4_KV env gate inconsistency",
                    );
                unsafe {
                    use cudarc::driver::sys::*;
                    const FA2_THREADS: i32 = 128;
                    const FA2_BC: i32 = 32;
                    let hd_i = head_dim as i32;
                    // f16 smem dequant (2 bytes/elem) vs F16's f32 (4).
                    let smem_bytes = 2 * FA2_BC * hd_i * 2 + FA2_BC * 4 + (FA2_THREADS / 32) * 4;
                    if smem_bytes as u32 >= 48 * 1024 {
                        let _ = cuFuncSetAttribute(
                            fn_dec.raw() as CUfunction,
                            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            smem_bytes,
                        );
                    }
                    let mut output = attn_out_region.device_ptr();
                    let mut query = q_fp8_ptr;
                    let mut key_packed = k_cache_layer_ptr;
                    let mut value_packed = v_cache_layer_ptr;
                    let mut key_scale = k_scale_layer_ptr;
                    let mut value_scale = v_scale_layer_ptr;
                    let mut q_scale_dyn = q_scale_cache_ptr;
                    let mut block_tables = self.bt_persistent_ptr;
                    let mut context_lens = cl_dev_ptr;
                    // Static-scalar Q descale fallback ptr; same
                    // never-dereferenced-on-dynamic-path reasoning as
                    // the RoPE kernel above.
                    let mut q_descale = q_scale_cache_ptr;
                    let mut scale_arg = scale;
                    let mut nh = num_heads as i32;
                    let mut nkvh = num_kv_heads as i32;
                    let mut hd = head_dim as i32;
                    let mut bs = self.kv_cache_block_size as i32;
                    let mut mbps = self.kv_cache_num_blocks as i32;
                    let mut window: i32 = -1;
                    let args = [
                        (&mut output) as *mut u64 as *mut core::ffi::c_void,
                        (&mut query) as *mut u64 as *mut core::ffi::c_void,
                        (&mut key_packed) as *mut u64 as *mut core::ffi::c_void,
                        (&mut value_packed) as *mut u64 as *mut core::ffi::c_void,
                        (&mut key_scale) as *mut u64 as *mut core::ffi::c_void,
                        (&mut value_scale) as *mut u64 as *mut core::ffi::c_void,
                        (&mut q_scale_dyn) as *mut u64 as *mut core::ffi::c_void,
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
                        1,
                        num_heads,
                        1,
                        FA2_THREADS as u32,
                        1,
                        1,
                        smem_bytes as u32,
                        self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 flash_attention_2_decode_nvfp4kv launch",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
            }
        }
        // No fence: attn_output_gate kernel runs on the same stream.

        // 7. attn_output_gate: attn_out * sigmoid(gate).
        let gated_region = self.arena.region("qwen36_pf_gated", q_size * 2, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut output = gated_region.device_ptr();
            let mut values = attn_out_region.device_ptr();
            let mut gate = gate_region.device_ptr();
            let mut nn = q_size as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut values) as *mut u64 as *mut core::ffi::c_void,
                (&mut gate) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = (q_size as u32 + block - 1) / block;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_sigmoid_mul_f16.raw() as CUfunction,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 sigmoid_mul_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // No fence: o_proj runs on the same stream after sigmoid_mul.

        // 8. o_proj FP8 GEMV → out_buf [hidden]. (Phase 4a routing.)
        let out_region = self.arena.region("qwen36_pf_out", (o_n as usize) * 2, 16)?;
        unsafe {
            self.fp8_proj_dispatch(
                kernel_gemv,
                out_region.device_ptr(),
                fl.o_proj.offset_bytes,
                o_bs,
                gated_region.device_ptr(),
                m,
                o_n,
                o_k,
                stream_raw,
            )?;
        }
        // No fence: residual vector_add runs on the same stream
        // after o_proj.

        // 9. Residual sum: last_hidden += out_buf, GPU-side.
        // Replaces a 2× DtoH + CPU loop over `hidden` halves + HtoD
        // round-trip with one `vector_add_f16` launch — same numeric
        // result (the CPU loop did f16 → f32 → add → f16 RTNE; the
        // kernel uses `__hadd`, which IS f16 RTNE, so output bytes
        // are identical).
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let n_elem = (hidden as usize) * (m as usize);
            let mut dst = last_hidden_ptr;
            let mut src = out_region.device_ptr();
            let mut nn: i32 = n_elem as i32;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 1024.min(n_elem as u32).max(1);
            let grid = ((n_elem as u32 + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_vector_add_f16.raw() as CUfunction,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 full_attn residual vector_add_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // No function-exit fence: same-stream ordering covers the
        // next layer's first read of last_hidden_ptr.
        Ok(())
    }

    /// Phase 5a helper: per-layer MoE block forward.
    /// Composes post_attention_layernorm → router top-k softmax →
    /// 8 routed FFNs (gate/up FP8 → host silu·mul → down FP8) +
    /// shared FFN scaled by sigmoid(shared_expert_gate_logit) →
    /// host weighted sum → residual into last_hidden.
    /// post_attn_norm_ptr is the layer's post_attention_layernorm
    /// weight offset (lives on the attn struct, not the moe block).
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_moe(
        &self,
        moe: &rvllm_loader::qwen36_weights::Qwen36MoeBlock,
        post_attn_norm_ptr: u64,
        last_hidden_ptr: u64,
        kernel_gemv: rvllm_kernels::KernelFn,
        hidden: u32,
        last_hidden_bytes: usize,
        _layer_idx: usize,
    ) -> Result<()> {
        self.apply_layer_moe_with_override(
            moe,
            post_attn_norm_ptr,
            last_hidden_ptr,
            kernel_gemv,
            hidden,
            last_hidden_bytes,
            _layer_idx,
            None,
            None,
        )
    }

    /// Phase 6a / Round-27: batched routing for the MoE layer.
    /// Pre-computes top-k indices + weights for ALL N tokens via one
    /// `router_gemv_batched_f16_to_f32` launch + one
    /// `topk_softmax_batched_f32` launch, then loops the per-token
    /// expert FFN compute (existing `apply_layer_moe_with_override`)
    /// with a route override pointing at each token's slot in the
    /// batched arrays. The expert FFN itself remains per-token in
    /// this first cut — codex round-27's recommended "first 200 LOC
    /// cut: router+topk batched, routed FFN unchanged" — to isolate
    /// routing-semantics correctness before tackling the
    /// gather/scatter rewrite for batched expert FFN. Save: 2*N
    /// launches per layer drop to 2 per layer (N=22 → 88 → 4 saved
    /// kernels per layer per request); the batched routing also lets
    /// the FFN reuse a single batched-RMSNorm post_attention_layernorm
    /// (one launch instead of N).
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_moe_batched(
        &self,
        moe: &rvllm_loader::qwen36_weights::Qwen36MoeBlock,
        post_attn_norm_ptr: u64,
        hidden_ptr: u64,
        num_tokens: u32,
        kernel_gemv: rvllm_kernels::KernelFn,
        hidden: u32,
        last_hidden_bytes: usize,
        layer_idx: usize,
    ) -> Result<()> {
        if num_tokens == 0 {
            return Ok(());
        }
        if num_tokens == 1 {
            return self.apply_layer_moe(
                moe,
                post_attn_norm_ptr,
                hidden_ptr,
                kernel_gemv,
                hidden,
                last_hidden_bytes,
                layer_idx,
            );
        }
        let n = num_tokens as usize;
        let h = hidden as usize;
        let stream_raw = self.stream.raw() as u64;
        let num_experts = self.arch.num_experts;
        let top_k = self.arch.num_experts_per_tok;

        // 1. Batched RMSNorm on a [N, hidden] copy.
        let normed_bytes = n * h * 2;
        let normed_region = self.arena.region("qwen36_pmb_normed", normed_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let r = cuMemcpyDtoDAsync_v2(
                normed_region.device_ptr(),
                hidden_ptr,
                normed_bytes,
                self.stream.raw() as _,
            );
            if r != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 moe_batched DtoD copy",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let eps = self.arch.base.rms_norm_eps;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                normed_region.device_ptr(),
                post_attn_norm_ptr,
                stream_raw,
            )?;
        }

        // Phase 8 batched router+topk fusion (2026-05-23): one
        // launch via last-block-does-topk (per-token counter
        // slots). Replaces the back-to-back
        // (router_gemv_batched + topk_softmax_batched) pair.
        let logits_bytes = n * num_experts * 4;
        let logits_region = self.arena.region("qwen36_pmb_logits", logits_bytes, 16)?;
        let topk_idx_region = self
            .arena
            .region("qwen36_pmb_topk_idx", n * top_k * 4, 16)?;
        let topk_w_region = self.arena.region("qwen36_pmb_topk_w", n * top_k * 4, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut idx_ptr = topk_idx_region.device_ptr();
            let mut w_ptr = topk_w_region.device_ptr();
            let mut logits_ptr = logits_region.device_ptr();
            let mut router_ptr = moe.router.offset_bytes;
            let mut input_ptr = normed_region.device_ptr();
            let mut counter_ptr =
                self.outside_kernels.router_gemv_with_topk_batched_counter_dev;
            let mut nx = num_experts as i32;
            let mut hh = hidden as i32;
            let mut kk = top_k as i32;
            let mut nt = num_tokens as i32;
            let args = [
                (&mut idx_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut w_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut logits_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut router_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut input_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut counter_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut nx) as *mut i32 as *mut core::ffi::c_void,
                (&mut hh) as *mut i32 as *mut core::ffi::c_void,
                (&mut kk) as *mut i32 as *mut core::ffi::c_void,
                (&mut nt) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = num_experts as u32;
            let grid_x: u32 = num_experts as u32;
            let grid_y: u32 = num_tokens;
            let rc = cuLaunchKernel(
                self.outside_kernels
                    .fn_router_gemv_with_topk_batched_f16_to_f32
                    .raw() as CUfunction,
                grid_x, grid_y, 1,
                block, 1, 1,
                0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 moe_batched router_gemv_with_topk_batched",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // 4. Routed FFN: env-gate Phase 6b (batched k-round expert
        // FFN via row-batched indirect kernels) vs Phase 6a (per-
        // token expert FFN with override). Codex round-27 routed-FFN
        // batched plan: 8 k-rounds × 3 batched kernels = 24 launches
        // per layer, retains the per-token k=0..7 add order.
        // Round-27d: default-ON post-audit.
        let routed_ffn_batched = std::env::var("RVLLM_QWEN36_BATCH_MOE_ROUTED_FFN")
            .map(|s| !matches!(s.as_str(), "0" | "false" | "FALSE" | "no"))
            .unwrap_or(true);
        let topk_idx_base = topk_idx_region.device_ptr();
        let topk_w_base = topk_w_region.device_ptr();
        let topk_stride_bytes = (top_k as u64) * 4;
        // Pre-compute n_int / n_down dimensions for routed FFN.
        let n_int = moe.experts_gate_proj_fused.shape[1] as u32;
        let k_in = moe.experts_gate_proj_fused.shape[2] as u32;
        let n_down = moe.experts_down_proj_fused.shape[1] as u32;
        let k_down = moe.experts_down_proj_fused.shape[2] as u32;
        let int_per_expert_w = (n_int as u64) * (k_in as u64);
        let int_per_expert_bs = ((n_int as u64) / 128) * ((k_in as u64) / 128) * 4;
        let down_per_expert_w = (n_down as u64) * (k_down as u64);
        let down_per_expert_bs = ((n_down as u64) / 128) * ((k_down as u64) / 128) * 4;
        let n_int_us = n_int as usize;
        let n_down_us = n_down as usize;
        let h_us = hidden as usize;
        let mut rs_batched_ptr_opt: Option<u64> = None;
        if routed_ffn_batched {
            let gate_bs = match moe.experts_gate_proj_fused.blockscale_ptr {
                Some(p) => p,
                None => return Ok(()),
            };
            let up_bs = match moe.experts_up_proj_fused.blockscale_ptr {
                Some(p) => p,
                None => return Ok(()),
            };
            let down_bs = match moe.experts_down_proj_fused.blockscale_ptr {
                Some(p) => p,
                None => return Ok(()),
            };
            // Phase 8 dual_silu k_round-batch fusion (2026-05-23):
            // silu_b grows to [top_k, num_tokens, N_int] f16. The
            // single k_round-batched dual_silu launch below fills
            // all k_rounds in one go; the per-k_round down loop
            // reads slices via `k_round * (num_tokens * N_int) * 2`
            // offset.
            let silu_b_bytes = top_k * n * n_int_us * 2;
            let down_b_bytes = n * n_down_us * 2;
            let rs_b_bytes = n * h_us * 4;
            let silu_b = self.arena.region("qwen36_pmb_silu", silu_b_bytes, 16)?;
            let down_b = self.arena.region("qwen36_pmb_down", down_b_bytes, 16)?;
            let rs_b = self.arena.region("qwen36_pmb_rs", rs_b_bytes, 16)?;
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let rc = cuMemsetD8Async(
                    rs_b.device_ptr(),
                    0,
                    rs_b_bytes,
                    self.stream.raw() as CUstream,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 moe_routed_batched rs_batched memset",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            let num_col_blocks_int = (k_in as i32) / 128;
            let num_col_blocks_down = (k_down as i32) / 128;
            // Phase 8 dual_silu k_round-batch (2026-05-23): one
            // launch computes silu_b[k_round, m, n] for ALL
            // k_rounds. Replaces top_k separate batched_topk
            // launches.
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut out_p = silu_b.device_ptr();
                let mut bwg = moe.experts_gate_proj_fused.offset_bytes;
                let mut bwu = moe.experts_up_proj_fused.offset_bytes;
                let mut bsg = gate_bs;
                let mut bsu = up_bs;
                let mut inp = normed_region.device_ptr();
                let mut idx_b = topk_idx_base;
                let mut wstride: i64 = int_per_expert_w as i64;
                let mut sstride: i64 = (int_per_expert_bs / 4) as i64;
                let mut m_i = num_tokens as i32;
                let mut n_i = n_int as i32;
                let mut k_i = k_in as i32;
                let mut ncb = num_col_blocks_int;
                let mut tk_i = top_k as i32;
                let args = [
                    (&mut out_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bwg) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bwu) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bsg) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bsu) as *mut u64 as *mut core::ffi::c_void,
                    (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                    (&mut idx_b) as *mut u64 as *mut core::ffi::c_void,
                    (&mut wstride) as *mut i64 as *mut core::ffi::c_void,
                    (&mut sstride) as *mut i64 as *mut core::ffi::c_void,
                    (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut ncb) as *mut i32 as *mut core::ffi::c_void,
                    (&mut tk_i) as *mut i32 as *mut core::ffi::c_void,
                ];
                let grid = (((n_int + 7) / 8).max(1), num_tokens, top_k as u32);
                let block: u32 = 256;
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_fp8_gemv_dual_silu_indirect_kround_batched
                        .raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block, 1, 1,
                    0,
                    self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 moe_routed_batched dual_silu_kround_batched",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            // Phase 8 batched-prefill down k_round-batch fusion
            // (2026-05-23, follow-on to decode-side commit b1f221e):
            // single launch handles all top_k k_rounds for every
            // (m, n_down) via per-warp sequential f32-register
            // accumulation. Reuses the decode kround_batched kernel
            // verbatim — it is M-agnostic (M=num_tokens here vs
            // M=1 in the decode hot path) and bit-equivalent to the
            // prior host-side loop of top_k separate
            // `fp8_gemv_indirect_scaled_add_batched_topk` launches
            // (same sum order: per-warp sequential over k_rounds).
            //
            // Eliminates (top_k - 1) launches per MoE layer per
            // prefill — top_k=8 × 40 MoE layers = 280 launches
            // saved per prefill batch.
            let _ = (n, silu_b_bytes); // silu_b_bytes still used elsewhere
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut acc = rs_b.device_ptr();
                let mut bw = moe.experts_down_proj_fused.offset_bytes;
                let mut bs = down_bs;
                // silu_b layout: [top_k, num_tokens, n_int] f16
                // (k_round-major, populated by the dual_silu
                // kround-batched launch above). The kernel reads
                // input_kround[k_round * M * K + m * K + ...].
                let mut inp = silu_b.device_ptr();
                let mut idx_p = topk_idx_base;
                let mut wp = topk_w_base;
                let mut w_stride: i64 = down_per_expert_w as i64;
                let mut s_stride: i64 = (down_per_expert_bs / 4) as i64;
                let mut m_i = num_tokens as i32;
                let mut n_i = n_down as i32;
                let mut k_i = k_down as i32;
                let mut ncb = num_col_blocks_down;
                let mut tk_i = top_k as i32;
                let args = [
                    (&mut acc) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bw) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bs) as *mut u64 as *mut core::ffi::c_void,
                    (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                    (&mut idx_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut wp) as *mut u64 as *mut core::ffi::c_void,
                    (&mut w_stride) as *mut i64 as *mut core::ffi::c_void,
                    (&mut s_stride) as *mut i64 as *mut core::ffi::c_void,
                    (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut ncb) as *mut i32 as *mut core::ffi::c_void,
                    (&mut tk_i) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid_x: u32 = ((n_down + 7) / 8).max(1);
                let rc = cuLaunchKernel(
                    self.outside_kernels
                        .fn_fp8_gemv_indirect_scaled_add_kround_batched
                        .raw() as CUfunction,
                    grid_x,
                    num_tokens,
                    1,
                    block,
                    1,
                    1,
                    0,
                    self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 moe_routed_batched down_kround_batched",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
                let _ = down_b.device_ptr();  // suppress unused warning
            }
            rs_batched_ptr_opt = Some(rs_b.device_ptr());
            // Keep regions alive for the per-token loop / shared-
            // batched stage below; the arena bump-alloc ensures
            // pointers remain valid until the outer caller restores.
            let _ = (silu_b, down_b);
        }

        // Phase 6c / Round-27: shared-expert batched. Runs only if
        // routed FFN was already batched (we need rs_b on device).
        // Codex round-27: 2 new kernels (shared_gate_dot_sigmoid_
        // batched, scaled_add_devw_batched), reuse existing
        // dual_silu and Fp8GemvF16InLaunch with m=N for the GEMVs,
        // existing f16_plus_f32_inplace_f16 with n=N*hidden for the
        // residual. byte-Identität pro Token erhalten weil dual_silu
        // / down GEMV / shared_gate / scaled_add jeweils row-
        // unabhängig sind und der inner-reduction Pfad m=1 vs m=N
        // identisch ist.
        // Round-27d: default-ON post-audit.
        let shared_batched = routed_ffn_batched
            && std::env::var("RVLLM_QWEN36_BATCH_MOE_SHARED")
                .map(|s| !matches!(s.as_str(), "0" | "false" | "FALSE" | "no"))
                .unwrap_or(true);
        if shared_batched {
            let sh_gate_bs = match moe.shared_expert_gate_proj.blockscale_ptr {
                Some(p) => p,
                None => return Ok(()),
            };
            let sh_up_bs = match moe.shared_expert_up_proj.blockscale_ptr {
                Some(p) => p,
                None => return Ok(()),
            };
            let sh_down_bs = match moe.shared_expert_down_proj.blockscale_ptr {
                Some(p) => p,
                None => return Ok(()),
            };
            let kernel_gemv_unwrap = self
                .outside_kernels
                .fn_fp8_gemv_wpr_native_f16in
                .ok_or_else(|| {
                    rvllm_core::RvllmError::cuda(
                        "qwen36 moe shared batched: fn_fp8_gemv_wpr_native_f16in missing",
                        rvllm_core::CudaErrorKind::Other,
                        rvllm_core::CudaCtx::setup(),
                    )
                })?;
            let _ = kernel_gemv;

            let silu_sh_bytes = n * n_int_us * 2;
            let down_sh_bytes = n * n_down_us * 2;
            let silu_sh = self.arena.region("qwen36_pmb_sh_silu", silu_sh_bytes, 16)?;
            let down_sh = self.arena.region("qwen36_pmb_sh_down", down_sh_bytes, 16)?;
            let sg_sigmoid_b = self.arena.region("qwen36_pmb_sg_sig", n * 4, 16)?;

            // shared dual_silu: m=N. Existing kernel grid (ceil(N/8), m, 1).
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut osi = silu_sh.device_ptr();
                let mut wg = moe.shared_expert_gate_proj.offset_bytes;
                let mut wu = moe.shared_expert_up_proj.offset_bytes;
                let mut sg = sh_gate_bs;
                let mut su = sh_up_bs;
                let mut inp = normed_region.device_ptr();
                let mut m_i = num_tokens as i32;
                let mut n_i = n_int as i32;
                let mut k_i = k_in as i32;
                let mut ncb = ((k_in + 127) / 128) as i32;
                let args = [
                    (&mut osi) as *mut u64 as *mut core::ffi::c_void,
                    (&mut wg) as *mut u64 as *mut core::ffi::c_void,
                    (&mut wu) as *mut u64 as *mut core::ffi::c_void,
                    (&mut sg) as *mut u64 as *mut core::ffi::c_void,
                    (&mut su) as *mut u64 as *mut core::ffi::c_void,
                    (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                    (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut ncb) as *mut i32 as *mut core::ffi::c_void,
                ];
                let grid_x = (n_int + 7) / 8;
                let block: u32 = 256;
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_fp8_gemv_dual_silu.raw() as CUfunction,
                    grid_x,
                    num_tokens,
                    1,
                    block,
                    1,
                    1,
                    0,
                    self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 moe shared dual_silu batched",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }

            // shared down: m=N via Fp8GemvF16InLaunch directly
            // (NOT fp8_proj_dispatch which would route to a different
            // m≥2 GEMM path; codex round-27 explicit).
            unsafe {
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m: num_tokens,
                    n: n_down,
                    k: k_down,
                }
                .launch(
                    kernel_gemv_unwrap,
                    down_sh.device_ptr(),
                    moe.shared_expert_down_proj.offset_bytes,
                    sh_down_bs,
                    silu_sh.device_ptr(),
                    self.stream.raw() as u64,
                )?;
            }

            // shared_gate_dot_sigmoid batched → sigmoid[N] f32.
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut out = sg_sigmoid_b.device_ptr();
                let mut weight = moe.shared_expert_gate_logit.offset_bytes;
                let mut input_ptr = normed_region.device_ptr();
                let mut hh = hidden as i32;
                let mut nt = num_tokens as i32;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut weight) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut hh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid: u32 = num_tokens;
                let rc = cuLaunchKernel(
                    self.outside_kernels
                        .fn_shared_gate_dot_sigmoid_f16_batched
                        .raw() as CUfunction,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 moe shared_gate_dot_sigmoid_batched",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }

            // scaled_add_devw_batched: rs_b[m, n] += sigmoid[m] * down_sh[m, n]
            let rs_b_ptr = rs_batched_ptr_opt.unwrap();
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut acc = rs_b_ptr;
                let mut inp = down_sh.device_ptr();
                let mut wp = sg_sigmoid_b.device_ptr();
                let mut hd = hidden as i32;
                let mut nt = num_tokens as i32;
                let args = [
                    (&mut acc) as *mut u64 as *mut core::ffi::c_void,
                    (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                    (&mut wp) as *mut u64 as *mut core::ffi::c_void,
                    (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid_x: u32 = ((hidden + block - 1) / block).max(1);
                let rc = cuLaunchKernel(
                    self.outside_kernels
                        .fn_scaled_add_f16_to_f32_devw_batched
                        .raw() as CUfunction,
                    grid_x,
                    num_tokens,
                    1,
                    block,
                    1,
                    1,
                    0,
                    self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 moe scaled_add_devw_batched (shared)",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }

            // Final residual batched: hidden_ptr += f16(rs_b)
            // elementwise [N * hidden].
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let n_elem = (n * h_us) as i32;
                let mut dst = hidden_ptr;
                let mut src = rs_b_ptr;
                let mut nn = n_elem;
                let args = [
                    (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                    (&mut src) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 1024.min(n_elem as u32).max(1);
                let grid: u32 = ((n_elem as u32 + block - 1) / block).max(1);
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_f16_plus_f32_inplace_f16.raw() as CUfunction,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 moe shared batched residual",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            // Skip the per-token loop below — shared+residual already
            // applied for all N tokens.
            return Ok(());
        }

        for t in 0..num_tokens {
            let tok_ptr = hidden_ptr + (t as u64) * (last_hidden_bytes as u64);
            let inner_ck = self.arena.checkpoint();
            let idx_t = topk_idx_base + (t as u64) * topk_stride_bytes;
            let w_t = topk_w_base + (t as u64) * topk_stride_bytes;
            let rs_seed = rs_batched_ptr_opt.map(|p| p + (t as u64) * (h_us as u64) * 4);
            self.apply_layer_moe_with_override(
                moe,
                post_attn_norm_ptr,
                tok_ptr,
                kernel_gemv,
                hidden,
                last_hidden_bytes,
                layer_idx,
                Some((idx_t, w_t)),
                rs_seed,
            )?;
            unsafe {
                self.arena.restore(inner_ck);
            }
        }
        Ok(())
    }

    /// Phase 6a / Round-27: per-token MoE forward with optional
    /// pre-computed routing. When `route_override = Some((idx_ptr,
    /// w_ptr))`, the in-function router-GEMV + top-k+softmax launches
    /// are skipped and the downstream expert FFN reads top-k from the
    /// caller-supplied device buffers (one (token's row-of-top_k))
    /// instead. Used by `apply_layer_moe_batched` to share batched
    /// routing across N tokens.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_moe_with_override(
        &self,
        moe: &rvllm_loader::qwen36_weights::Qwen36MoeBlock,
        post_attn_norm_ptr: u64,
        last_hidden_ptr: u64,
        kernel_gemv: rvllm_kernels::KernelFn,
        hidden: u32,
        last_hidden_bytes: usize,
        _layer_idx: usize,
        route_override: Option<(u64, u64)>,
        rs_seed: Option<u64>,
    ) -> Result<()> {
        let stream_raw = self.stream.raw() as u64;
        let n_int = moe.experts_gate_proj_fused.shape[1] as u32; // 512
        let k_in = moe.experts_gate_proj_fused.shape[2] as u32; // 2048
        let n_down = moe.experts_down_proj_fused.shape[1] as u32; // 2048
        let k_down = moe.experts_down_proj_fused.shape[2] as u32; // 512
        let num_experts = self.arch.num_experts;
        let top_k = self.arch.num_experts_per_tok;
        let m: u32 = 1;

        // 1. post_attention_layernorm on copy of last_hidden.
        let normed_region = self
            .arena
            .region("qwen36_pm_normed", last_hidden_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoDAsync_v2(
                normed_region.device_ptr(),
                last_hidden_ptr,
                last_hidden_bytes,
                self.stream.raw() as _,
            );
        }
        let eps = self.arch.base.rms_norm_eps;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                normed_region.device_ptr(),
                post_attn_norm_ptr,
                stream_raw,
            )?;
        }
        // No fence after rmsnorm: the next op is the GPU router GEMV
        // which runs on the same stream_raw, so ordering is automatic.
        // Removing the per-layer fence saves ~30 fence syscalls / token
        // (Phase 4b-prep iter22).

        // 2. Router top-k. When `route_override` is Some, skip both
        // the router GEMV and topk+softmax — the caller has already
        // pre-computed (top_idx, top_w) for this token in shared
        // batched arrays. Else compute fresh per-token routing.
        let hidden_us = hidden as usize;
        let _ = hidden_us;
        let (route_idx_ptr, route_w_ptr) = if let Some((idx, w)) = route_override {
            (idx, w)
        } else {
            // Phase 8 router+topk fusion (2026-05-23): ONE kernel
            // does the router GEMV + topk-softmax via the last-
            // block-does-topk atomic pattern. Replaces the back-
            // to-back (router_gemv + topk_softmax) launches with
            // a single launch.
            let logits_bytes = num_experts * 4;
            let logits_region = self.arena.region("qwen36_pm_logits", logits_bytes, 16)?;
            let topk_idx_region = self.arena.region("qwen36_pm_topk_idx", top_k * 4, 16)?;
            let topk_w_region = self.arena.region("qwen36_pm_topk_w", top_k * 4, 16)?;
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut idx_ptr = topk_idx_region.device_ptr();
                let mut w_ptr = topk_w_region.device_ptr();
                let mut logits_ptr = logits_region.device_ptr();
                let mut router_ptr = moe.router.offset_bytes;
                let mut input_ptr = normed_region.device_ptr();
                let mut counter_ptr = self.outside_kernels.router_topk_counter_dev;
                let mut nx = num_experts as i32;
                let mut hh = hidden as i32;
                let mut kk = top_k as i32;
                let args = [
                    (&mut idx_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut w_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut logits_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut router_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut counter_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nx) as *mut i32 as *mut core::ffi::c_void,
                    (&mut hh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut kk) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = num_experts as u32;
                let grid: u32 = num_experts as u32;
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_router_gemv_with_topk_f16_to_f32.raw() as CUfunction,
                    grid, 1, 1,
                    block, 1, 1,
                    0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 router_gemv_with_topk_f16_to_f32 launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            (topk_idx_region.device_ptr(), topk_w_region.device_ptr())
        };

        // 3. Routed experts: each runs gate+up FP8 → silu·mul → down FP8.
        let mid_bytes = (n_int as usize) * 2;
        let down_bytes = (n_down as usize) * 2;
        let _gate_region = self.arena.region("qwen36_pm_g", mid_bytes, 16)?;
        let _up_region = self.arena.region("qwen36_pm_u", mid_bytes, 16)?;
        // Phase 8 dual_silu k_round-batch (2026-05-23): silu_region
        // grows to [top_k, M, N_int] f16 so all k_rounds' silu
        // outputs live concurrently. Subsequent down launches read
        // their slice at offset `k_round * (M * N_int) * 2`. For
        // per-token decode (M=1), that's `k_round * mid_bytes`.
        let silu_region = self.arena.region(
            "qwen36_pm_s", (top_k as usize) * mid_bytes, 16)?;
        let down_region = self.arena.region("qwen36_pm_d", down_bytes, 16)?;
        let int_per_expert_w = (n_int as u64) * (k_in as u64);
        let int_per_expert_bs = ((n_int as u64) / 128) * ((k_in as u64) / 128) * 4;
        let down_per_expert_w = (n_down as u64) * (k_down as u64);
        let down_per_expert_bs = ((n_down as u64) / 128) * ((k_down as u64) / 128) * 4;
        let gate_bs = match moe.experts_gate_proj_fused.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let up_bs = match moe.experts_up_proj_fused.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let down_bs = match moe.experts_down_proj_fused.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        // Phase 4b-prep iter18: keep routed_sum on the GPU for the
        // entire expert loop. f32 accumulator, zeroed once on
        // stream_raw, then DtoH'd ONCE after the shared expert
        // finishes (with a self.stream.fence() before the DtoH —
        // that fence is the iter17-discovered invariant).
        let routed_sum_bytes = (n_down as usize) * 4;
        let routed_sum_region = self.arena.region("qwen36_pm_rs", routed_sum_bytes, 16)?;
        // Phase 6b: when `rs_seed = Some(seed)`, the routed FFN was
        // already computed batched by the caller into `seed`. Copy
        // that into our local routed_sum_region (so downstream
        // shared-expert + residual code path stays unchanged) and
        // skip the per-expert FFN k-loop entirely.
        if let Some(seed) = rs_seed {
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let rc = cuMemcpyDtoDAsync_v2(
                    routed_sum_region.device_ptr(),
                    seed,
                    routed_sum_bytes,
                    self.stream.raw() as _,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 moe rs_seed copy",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        } else {
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let rc = cuMemsetD8Async(
                    routed_sum_region.device_ptr(),
                    0,
                    routed_sum_bytes,
                    stream_raw as CUstream,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 moe routed_sum cuMemsetD8Async",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        let topk_idx_base = route_idx_ptr;
        let topk_w_base = route_w_ptr;
        let skip_routed_ffn = rs_seed.is_some();
        // Phase 8 dual_silu k_round-batch (2026-05-23): one launch
        // computes silu_region[k_round, m, n] for ALL k_rounds in
        // grid.z. Replaces the 8 host-loop dual_silu_indirect
        // launches with ONE. The subsequent down loop (below) still
        // runs 8 launches each reading silu_region at its
        // `k_round * mid_bytes` slice, because the accumulation to
        // routed_sum needs to be SEQUENTIAL across k_rounds (no
        // atomic, no per-k_round accumulator).
        if !skip_routed_ffn && top_k > 0 {
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut osi = silu_region.device_ptr();
                let mut bwg = moe.experts_gate_proj_fused.offset_bytes;
                let mut bwu = moe.experts_up_proj_fused.offset_bytes;
                let mut bsg = gate_bs;
                let mut bsu = up_bs;
                let mut inp = normed_region.device_ptr();
                let mut idx_b = topk_idx_base;
                let mut w_stride = int_per_expert_w as i64;
                let mut s_stride_elems = (int_per_expert_bs / 4) as i64;
                let mut m_i = m as i32;
                let mut n_i = n_int as i32;
                let mut k_i = k_in as i32;
                let mut ncb = ((k_in + 127) / 128) as i32;
                let mut tk_i = top_k as i32;
                let args = [
                    (&mut osi) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bwg) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bwu) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bsg) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bsu) as *mut u64 as *mut core::ffi::c_void,
                    (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                    (&mut idx_b) as *mut u64 as *mut core::ffi::c_void,
                    (&mut w_stride) as *mut i64 as *mut core::ffi::c_void,
                    (&mut s_stride_elems) as *mut i64 as *mut core::ffi::c_void,
                    (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut ncb) as *mut i32 as *mut core::ffi::c_void,
                    (&mut tk_i) as *mut i32 as *mut core::ffi::c_void,
                ];
                let grid = ((n_int + 7) / 8, m, top_k as u32);
                let block = (256u32, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_fp8_gemv_dual_silu_indirect_kround_batched
                        .raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block.0, block.1, block.2,
                    0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 fp8_gemv_dual_silu_indirect_kround_batched launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        if !skip_routed_ffn && top_k > 0 {
            // Phase 8 down k_round-batch fusion (2026-05-23):
            // ONE launch handles all top_k k_rounds for every
            // (m, n_down) via per-warp sequential accumulation.
            // The literal dual_silu+down megakernel is infeasible
            // (recomputing silu_mul per output element would
            // explode work by ~2048x); this is the closest
            // tractable analog — fuse the 8 down launches into 1
            // by exploiting per-warp f32-register accumulation,
            // no atomic, no global RMW per k_round.
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut acc_f32 = routed_sum_region.device_ptr();
                let mut bw = moe.experts_down_proj_fused.offset_bytes;
                let mut bs = down_bs;
                let mut inp = silu_region.device_ptr();  // [top_k, m, n_int]
                let mut idx_b = topk_idx_base;
                let mut wp = topk_w_base;
                let mut w_stride = down_per_expert_w as i64;
                let mut s_stride_elems = (down_per_expert_bs / 4) as i64;
                let mut m_i = m as i32;
                let mut n_i = n_down as i32;
                let mut k_i = k_down as i32;
                let mut ncb = ((k_down + 127) / 128) as i32;
                let mut tk_i = top_k as i32;
                let args = [
                    (&mut acc_f32) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bw) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bs) as *mut u64 as *mut core::ffi::c_void,
                    (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                    (&mut idx_b) as *mut u64 as *mut core::ffi::c_void,
                    (&mut wp) as *mut u64 as *mut core::ffi::c_void,
                    (&mut w_stride) as *mut i64 as *mut core::ffi::c_void,
                    (&mut s_stride_elems) as *mut i64 as *mut core::ffi::c_void,
                    (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut ncb) as *mut i32 as *mut core::ffi::c_void,
                    (&mut tk_i) as *mut i32 as *mut core::ffi::c_void,
                ];
                let grid = ((n_down + 7) / 8, m, 1u32);
                let block = (256u32, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_fp8_gemv_indirect_scaled_add_kround_batched
                        .raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block.0, block.1, block.2,
                    0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 fp8_gemv_indirect_scaled_add_kround_batched launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        // (Legacy per-k_round loop kept dead-code-gated so the
        // arena layout / closure captures stay identical to the
        // pre-fusion build; the fused launch above handles all
        // k_rounds in one go.)
        for i in 0..0usize {
            let idx_ptr_i = topk_idx_base + (i as u64) * 4;
            let w_ptr_i = topk_w_base + (i as u64) * 4;
            let silu_kround_ptr = silu_region.device_ptr()
                + (i as u64) * (mid_bytes as u64);
            #[cfg(feature = "cuda")]
            unsafe {
                let _ = (idx_ptr_i, w_ptr_i, silu_kround_ptr,
                         routed_sum_region.device_ptr(),
                         moe.experts_down_proj_fused.offset_bytes,
                         down_bs, down_per_expert_w, down_per_expert_bs,
                         m, n_down, k_down, stream_raw);
            }
            // (Old: separate scaled_add_f16_to_f32_devw call has
            // been folded into the fused kernel above. Kept the
            // arena layout (down_region, w_ptr_i, etc.) the same
            // so other call sites that might still use the
            // unfused path stay unaffected.)
            #[cfg(all(feature = "cuda", any()))]
            unsafe {
                use cudarc::driver::sys::*;
                let mut acc = routed_sum_region.device_ptr();
                let mut input = down_region.device_ptr();
                let mut devw = w_ptr_i;
                let mut nn = n_down as i32;
                let args = [
                    (&mut acc) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input) as *mut u64 as *mut core::ffi::c_void,
                    (&mut devw) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = (n_down as u32 + block - 1) / block;
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_scaled_add_f16_to_f32_devw.raw() as CUfunction,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 scaled_add_f16_to_f32_devw launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        // Optional debug L2 of routed-only sum.
        let routed_only_l2: f32 = if std::env::var("RVLLM_QWEN36_DEBUG_MOE").is_ok() {
            self.stream.fence()?;
            let mut rs_host = vec![0.0f32; n_down as usize];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    rs_host.as_mut_ptr() as *mut _,
                    routed_sum_region.device_ptr(),
                    routed_sum_bytes,
                );
            }
            rs_host.iter().map(|x| x * x).sum::<f32>().sqrt()
        } else {
            0.0
        };

        // 4. Shared expert FFN scaled by sigmoid(shared_expert_gate_logit).
        let sh_gate_bs = moe.shared_expert_gate_proj.blockscale_ptr.unwrap_or(0);
        let sh_up_bs = moe.shared_expert_up_proj.blockscale_ptr.unwrap_or(0);
        let sh_down_bs = moe.shared_expert_down_proj.blockscale_ptr.unwrap_or(0);
        if sh_gate_bs != 0 && sh_up_bs != 0 && sh_down_bs != 0 {
            // Phase 4b-prep iter32: same triple-fuse kernel for the
            // shared expert.
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut osi = silu_region.device_ptr();
                let mut wg = moe.shared_expert_gate_proj.offset_bytes;
                let mut wu = moe.shared_expert_up_proj.offset_bytes;
                let mut sg = sh_gate_bs;
                let mut su = sh_up_bs;
                let mut inp = normed_region.device_ptr();
                let mut m_i = m as i32;
                let mut n_i = n_int as i32;
                let mut k_i = k_in as i32;
                let mut ncb = ((k_in + 127) / 128) as i32;
                let args = [
                    (&mut osi) as *mut u64 as *mut core::ffi::c_void,
                    (&mut wg) as *mut u64 as *mut core::ffi::c_void,
                    (&mut wu) as *mut u64 as *mut core::ffi::c_void,
                    (&mut sg) as *mut u64 as *mut core::ffi::c_void,
                    (&mut su) as *mut u64 as *mut core::ffi::c_void,
                    (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                    (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut ncb) as *mut i32 as *mut core::ffi::c_void,
                ];
                let grid = ((n_int + 7) / 8, m, 1u32);
                let block = (256u32, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_fp8_gemv_dual_silu.raw() as CUfunction,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 fp8_gemv_dual_silu launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            // Phase 8 shared-expert fusion (2026-05-23): the
            // shared_expert_down GEMV and the subsequent
            // scaled_add_f16_to_f32_devw collapse into a single
            // kernel below — `fp8_gemv_f16in_scaled_add_devw`.
            // The standalone down GEMV here is skipped; the fused
            // closer below reads `silu_region` + the sigmoid
            // scalar + accumulates to `routed_sum_region` directly.
            let _ = kernel_gemv;
            let _ = down_region.device_ptr();
            let _ = sh_down_bs;
            // shared_expert_gate is Linear(hidden→1) per vLLM
            // qwen3_next.py:127-133: gate_logit = weight · normed_hidden
            // (scalar per token), then sigmoid(gate_logit) scales the
            // shared expert output. Weights cached as f32 host vec at
            // bring-up (Phase 4b-prep iter16); per-token path is just
            // a dot-product over RAM.
            // Phase 4b-prep iter21: fused GPU dot+sigmoid + a
            // device-pointer scaled_add kills the per-layer fence +
            // DtoH that an iter20 attempt regressed on. Both kernels
            // run on stream_raw, chained via the device scalar
            // `sg_sigmoid_region`; host never sees the value.
            let sg_sigmoid_region = self.arena.region("qwen36_pm_sg_sigmoid", 4, 16)?;
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut out = sg_sigmoid_region.device_ptr();
                let mut weight = moe.shared_expert_gate_logit.offset_bytes;
                let mut input_ptr = normed_region.device_ptr();
                let mut hh = hidden_us as i32;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut weight) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut hh) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid: u32 = 1;
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_shared_gate_dot_sigmoid_f16.raw() as CUfunction,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 shared_gate_dot_sigmoid_f16 launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            // Fused shared-expert (down + scaled-add) kernel:
            // computes the FP8 GEMV exactly as the standalone
            // `fp8_gemv_blockwise_wpr_native_f16in_kernel`, then on
            // lane 0 reads routed_sum[m*N+n], adds
            // *sg_sigmoid_region * acc, writes back. Replaces the
            // (down → scaled_add_f16_to_f32_devw) pair with ONE
            // launch.
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut acc = routed_sum_region.device_ptr();
                let mut weight = moe.shared_expert_down_proj.offset_bytes;
                let mut scl = sh_down_bs;
                let mut inp = silu_region.device_ptr();
                let mut devw = sg_sigmoid_region.device_ptr();
                let mut m_i = m as i32;
                let mut n_i = n_down as i32;
                let mut k_i = k_down as i32;
                let mut ncb = ((k_down + 127) / 128) as i32;
                let args = [
                    (&mut acc) as *mut u64 as *mut core::ffi::c_void,
                    (&mut weight) as *mut u64 as *mut core::ffi::c_void,
                    (&mut scl) as *mut u64 as *mut core::ffi::c_void,
                    (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                    (&mut devw) as *mut u64 as *mut core::ffi::c_void,
                    (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut ncb) as *mut i32 as *mut core::ffi::c_void,
                ];
                let grid = ((n_down + 7) / 8, m, 1u32);
                let block = (256u32, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_fp8_gemv_f16in_scaled_add_devw.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block.0, block.1, block.2,
                    0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 fp8_gemv_f16in_scaled_add_devw (shared-expert fused) launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        // 5. Residual sum, GPU-side (Phase 4b-prep iter19):
        //    last_hidden[i] = f16(f16_to_f32(last_hidden[i]) + routed_sum[i])
        // Same stream as the per-expert scaled_add and the prior
        // attn writeback into last_hidden_ptr — automatic ordering.
        // Also: with the residual now device-side, the optional debug
        // `RVLLM_QWEN36_DEBUG_MOE` path no longer has free routed_sum
        // on the host. We DtoH it lazily inside the env-gated branch.
        if std::env::var("RVLLM_QWEN36_DEBUG_MOE").is_ok() {
            self.stream.fence()?;
            let mut rs_host = vec![0.0f32; n_down as usize];
            let mut lh_host = vec![0u8; last_hidden_bytes];
            let mut normed_host = vec![0u8; last_hidden_bytes];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    rs_host.as_mut_ptr() as *mut _,
                    routed_sum_region.device_ptr(),
                    routed_sum_bytes,
                );
                let _ = cuMemcpyDtoH_v2(
                    lh_host.as_mut_ptr() as *mut _,
                    last_hidden_ptr,
                    last_hidden_bytes,
                );
                let _ = cuMemcpyDtoH_v2(
                    normed_host.as_mut_ptr() as *mut _,
                    normed_region.device_ptr(),
                    last_hidden_bytes,
                );
            }
            let total_l2: f32 = rs_host.iter().map(|x| x * x).sum::<f32>().sqrt();
            let shared_l2 = (total_l2 * total_l2 - routed_only_l2 * routed_only_l2)
                .max(0.0)
                .sqrt();
            let mut normed_l2_sq = 0.0f32;
            for i in 0..hidden_us {
                let v = f16_bits_to_f32(u16::from_le_bytes([
                    normed_host[i * 2],
                    normed_host[i * 2 + 1],
                ]));
                normed_l2_sq += v * v;
            }
            let normed_l2 = normed_l2_sq.sqrt();
            eprintln!("[moe] normed_L2={normed_l2:.2} routed_L2={routed_only_l2:.3} shared_L2~{shared_l2:.3} total_L2={total_l2:.3}");
            let _ = lh_host; // reserved for future per-element debugging
        }
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut inout = last_hidden_ptr;
            let mut add = routed_sum_region.device_ptr();
            let mut nn = hidden as i32;
            let args = [
                (&mut inout) as *mut u64 as *mut core::ffi::c_void,
                (&mut add) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = (hidden + block - 1) / block;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_f16_plus_f32_inplace_f16.raw() as CUfunction,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 moe residual launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // No fence: the residual kernel runs on stream_raw, the next
        // layer's first read of last_hidden_ptr is also on stream_raw,
        // so ordering is automatic.
        Ok(())
    }

    /// Debug/probe path for the native Qwen MTP block. This is not
    /// production speculative decoding yet: it uses an embedding row
    /// as the "current hidden" stand-in so we can validate the real
    /// MTP tensor chain and kernel ABI before threading the method
    /// into the live generation loop.
    fn forward_qwen36_mtp_one_token_probe(
        &self,
        hidden_token_id: i32,
        draft_input_token_id: i32,
        position: u32,
    ) -> Result<i32> {
        let mtp = self.model.mtp.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::cuda(
                "qwen36 MTP probe requested but model.mtp is not loaded; set RVLLM_QWEN36_LOAD_MTP=1",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;
        let kernel_gemv = self
            .outside_kernels
            .fn_fp8_gemv_wpr_native_f16in
            .ok_or_else(|| {
                rvllm_core::RvllmError::cuda(
                    "qwen36 MTP probe: fn_fp8_gemv_wpr_native_f16in not loaded",
                    rvllm_core::CudaErrorKind::Other,
                    rvllm_core::CudaCtx::setup(),
                )
            })?;
        let hidden = self.arch.base.hidden_size as u32;
        let vocab = self.arch.base.vocab_size as u32;
        let hidden_bytes = hidden as usize * 2;
        let stream_raw = self.stream.raw() as u64;

        let token_region = self.arena.region("qwen36_mtp_probe_tokens", 8, 16)?;
        let mut token_bytes = Vec::with_capacity(8);
        token_bytes.extend_from_slice(&hidden_token_id.to_le_bytes());
        token_bytes.extend_from_slice(&draft_input_token_id.to_le_bytes());
        unsafe { token_region.copy_from_host(&token_bytes)? };
        let embed_region = self
            .arena
            .region("qwen36_mtp_probe_embed", hidden_bytes * 2, 16)?;
        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 2,
                hidden,
                vocab,
            }
            .launch(
                self.outside_kernels.fn_embedding_gather_f16,
                embed_region.device_ptr(),
                self.model.outside.embed_tokens.offset_bytes,
                token_region.device_ptr(),
                stream_raw,
            )?;
        }

        let norm_hidden = self
            .arena
            .region("qwen36_mtp_probe_norm_h", hidden_bytes, 16)?;
        let norm_embedding = self
            .arena
            .region("qwen36_mtp_probe_norm_e", hidden_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                norm_hidden.device_ptr(),
                embed_region.device_ptr(),
                hidden_bytes,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP probe hidden DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            let rc = cuMemcpyDtoDAsync_v2(
                norm_embedding.device_ptr(),
                embed_region.device_ptr() + hidden_bytes as u64,
                hidden_bytes,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP probe embedding DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let eps = self.arch.base.rms_norm_eps;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                norm_hidden.device_ptr(),
                mtp.pre_fc_norm_hidden.offset_bytes,
                stream_raw,
            )?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                norm_embedding.device_ptr(),
                mtp.pre_fc_norm_embedding.offset_bytes,
                stream_raw,
            )?;
        }

        let fc_input = self
            .arena
            .region("qwen36_mtp_probe_fc_input", hidden_bytes * 2, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                fc_input.device_ptr(),
                norm_embedding.device_ptr(),
                hidden_bytes,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP probe fc input embedding DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            let rc = cuMemcpyDtoDAsync_v2(
                fc_input.device_ptr() + hidden_bytes as u64,
                norm_hidden.device_ptr(),
                hidden_bytes,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP probe fc input hidden DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        let mtp_hidden_f32 = self
            .arena
            .region("qwen36_mtp_probe_fc_out_f32", hidden as usize * 4, 16)?;
        let mtp_hidden = self
            .arena
            .region("qwen36_mtp_probe_hidden", hidden_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            self.cublaslt.f16_gemm_f32(
                fc_input.device_ptr(),
                mtp.fc.offset_bytes,
                mtp_hidden_f32.device_ptr(),
                1,
                hidden as i32,
                (hidden * 2) as i32,
                stream_raw,
            )?;
            use cudarc::driver::sys::*;
            let mut output = mtp_hidden.device_ptr();
            let mut input = mtp_hidden_f32.device_ptr();
            let mut n = hidden as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut n) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = ((hidden + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_cast_f32_to_f16.raw() as CUfunction,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP probe cast fc output",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        let pos_region = self.arena.region("qwen36_mtp_probe_pos", 4, 16)?;
        let clen_region = self.arena.region("qwen36_mtp_probe_clen", 4, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD32Async(
                pos_region.device_ptr(),
                position,
                1,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP probe pos memset",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            let rc = cuMemsetD32Async(
                clen_region.device_ptr(),
                position + 1,
                1,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP probe context-len memset",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        let post_attn_norm_ptr = match &mtp.layer.attn {
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Full(fl) => {
                self.apply_layer_full_attn(
                    fl,
                    self.mtp_kv_layer_seq_idx(),
                    position,
                    mtp_hidden.device_ptr(),
                    kernel_gemv,
                    hidden,
                    hidden_bytes,
                    pos_region.device_ptr(),
                    clen_region.device_ptr(),
                )?;
                fl.post_attention_layernorm.offset_bytes
            }
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(_) => {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP probe expected a full-attention MTP layer",
                    rvllm_core::CudaErrorKind::Other,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        };
        self.apply_layer_moe(
            &mtp.layer.moe,
            post_attn_norm_ptr,
            mtp_hidden.device_ptr(),
            kernel_gemv,
            hidden,
            hidden_bytes,
            0,
        )?;
        self.forward_qwen36_mtp_closer(mtp_hidden.device_ptr(), mtp.norm.offset_bytes, hidden, vocab)
    }

    fn forward_qwen36_mtp_from_hidden_ptr(
        &self,
        source_hidden_ptr: u64,
        draft_input_token_id: i32,
        position: u32,
    ) -> Result<i32> {
        let mtp = self.model.mtp.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::cuda(
                "qwen36 MTP shadow requested but model.mtp is not loaded; set RVLLM_QWEN36_LOAD_MTP=1",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;
        let kernel_gemv = self
            .outside_kernels
            .fn_fp8_gemv_wpr_native_f16in
            .ok_or_else(|| {
                rvllm_core::RvllmError::cuda(
                    "qwen36 MTP shadow: fn_fp8_gemv_wpr_native_f16in not loaded",
                    rvllm_core::CudaErrorKind::Other,
                    rvllm_core::CudaCtx::setup(),
                )
            })?;
        let hidden = self.arch.base.hidden_size as u32;
        let vocab = self.arch.base.vocab_size as u32;
        let hidden_bytes = hidden as usize * 2;
        let stream_raw = self.stream.raw() as u64;

        let token_region = self.arena.region("qwen36_mtp_shadow_token", 4, 16)?;
        unsafe { token_region.copy_from_host(&draft_input_token_id.to_le_bytes())? };
        let embed_region = self
            .arena
            .region("qwen36_mtp_shadow_embed", hidden_bytes, 16)?;
        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1,
                hidden,
                vocab,
            }
            .launch(
                self.outside_kernels.fn_embedding_gather_f16,
                embed_region.device_ptr(),
                self.model.outside.embed_tokens.offset_bytes,
                token_region.device_ptr(),
                stream_raw,
            )?;
        }

        let norm_hidden = self
            .arena
            .region("qwen36_mtp_shadow_norm_h", hidden_bytes, 16)?;
        let norm_embedding = self
            .arena
            .region("qwen36_mtp_shadow_norm_e", hidden_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                norm_hidden.device_ptr(),
                source_hidden_ptr,
                hidden_bytes,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP shadow hidden DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            let rc = cuMemcpyDtoDAsync_v2(
                norm_embedding.device_ptr(),
                embed_region.device_ptr(),
                hidden_bytes,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP shadow embedding DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let eps = self.arch.base.rms_norm_eps;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                norm_hidden.device_ptr(),
                mtp.pre_fc_norm_hidden.offset_bytes,
                stream_raw,
            )?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                norm_embedding.device_ptr(),
                mtp.pre_fc_norm_embedding.offset_bytes,
                stream_raw,
            )?;
        }

        let fc_input = self
            .arena
            .region("qwen36_mtp_shadow_fc_input", hidden_bytes * 2, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                fc_input.device_ptr(),
                norm_embedding.device_ptr(),
                hidden_bytes,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP shadow fc input embedding DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            let rc = cuMemcpyDtoDAsync_v2(
                fc_input.device_ptr() + hidden_bytes as u64,
                norm_hidden.device_ptr(),
                hidden_bytes,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP shadow fc input hidden DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        let mtp_hidden_f32 = self
            .arena
            .region("qwen36_mtp_shadow_fc_out_f32", hidden as usize * 4, 16)?;
        let mtp_hidden = self
            .arena
            .region("qwen36_mtp_shadow_hidden", hidden_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            self.cublaslt.f16_gemm_f32(
                fc_input.device_ptr(),
                mtp.fc.offset_bytes,
                mtp_hidden_f32.device_ptr(),
                1,
                hidden as i32,
                (hidden * 2) as i32,
                stream_raw,
            )?;
            use cudarc::driver::sys::*;
            let mut output = mtp_hidden.device_ptr();
            let mut input = mtp_hidden_f32.device_ptr();
            let mut n = hidden as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut n) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = ((hidden + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_cast_f32_to_f16.raw() as CUfunction,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP shadow cast fc output",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        let pos_region = self.arena.region("qwen36_mtp_shadow_pos", 4, 16)?;
        let clen_region = self.arena.region("qwen36_mtp_shadow_clen", 4, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD32Async(
                pos_region.device_ptr(),
                position,
                1,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP shadow pos memset",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            let rc = cuMemsetD32Async(
                clen_region.device_ptr(),
                position + 1,
                1,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP shadow context-len memset",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        let post_attn_norm_ptr = match &mtp.layer.attn {
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Full(fl) => {
                self.apply_layer_full_attn(
                    fl,
                    self.mtp_kv_layer_seq_idx(),
                    position,
                    mtp_hidden.device_ptr(),
                    kernel_gemv,
                    hidden,
                    hidden_bytes,
                    pos_region.device_ptr(),
                    clen_region.device_ptr(),
                )?;
                fl.post_attention_layernorm.offset_bytes
            }
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(_) => {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP shadow expected a full-attention MTP layer",
                    rvllm_core::CudaErrorKind::Other,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        };
        self.apply_layer_moe(
            &mtp.layer.moe,
            post_attn_norm_ptr,
            mtp_hidden.device_ptr(),
            kernel_gemv,
            hidden,
            hidden_bytes,
            0,
        )?;
        self.forward_qwen36_mtp_closer(mtp_hidden.device_ptr(), mtp.norm.offset_bytes, hidden, vocab)
    }

    fn forward_qwen36_mtp_closer(
        &self,
        hidden_ptr: u64,
        norm_ptr: u64,
        hidden: u32,
        vocab: u32,
    ) -> Result<i32> {
        let eps = self.arch.base.rms_norm_eps;
        let stream_raw = self.stream.raw() as u64;
        let hidden_fp8_region =
            self.arena
                .region("qwen36_mtp_probe_closer_h_fp8", hidden as usize, 16)?;
        let hidden_scale_region =
            self.arena
                .region("qwen36_mtp_probe_closer_h_scale", 4, 16)?;
        let logits_region =
            self.arena
                .region("qwen36_mtp_probe_closer_logits", vocab as usize * 2, 16)?;
        unsafe {
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: 1,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_fused_rmsnorm_fp8_quant,
                hidden_fp8_region.device_ptr(),
                hidden_scale_region.device_ptr(),
                hidden_ptr,
                norm_ptr,
                stream_raw,
            )?;
        }
        #[cfg(feature = "cuda")]
        unsafe {
            self.cublaslt.fp8_gemm(
                hidden_fp8_region.device_ptr(),
                self.model.outside.lm_head_fp8.offset_bytes,
                logits_region.device_ptr(),
                1,
                vocab as i32,
                hidden as i32,
                hidden_scale_region.device_ptr(),
                self.model.outside.lm_head_fp8.scale_ptr,
                stream_raw,
            )?;
        }
        let token_region = self.arena.region("qwen36_mtp_probe_token", 4, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut logits_ptr = logits_region.device_ptr();
            let mut out_ptr = token_region.device_ptr();
            let mut vs = vocab as i32;
            let args = [
                (&mut logits_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut vs) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_argmax_f16.raw() as CUfunction,
                1,
                1,
                1,
                512,
                1,
                1,
                0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP probe argmax",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;
        let mut tok_buf = [0i32; 1];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(tok_buf.as_mut_ptr() as *mut _, token_region.device_ptr(), 4);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 MTP probe token DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(tok_buf[0])
    }

    /// Helper: shared closer for `forward_outside_only` and the
    /// Phase-4v experimental path. Takes a `hidden_region` already
    /// populated with f16 hidden state and returns the argmax token
    /// id over the last token's lm_head logits.
    /// Phase 8 commit 2: graph-capture-friendly closer variant.
    /// Mirrors `forward_qwen36_outside_closer` but writes the argmax
    /// token id to a CALLER-PROVIDED device pointer instead of doing
    /// a sync `cuMemcpyDtoH_v2` at the end. Outside the captured
    /// region, the operator calls
    /// [`Self::argmax_dev_to_host_token`] to extract the 4-byte
    /// result.
    ///
    /// Only the fp8 lm_head path is supported (the production
    /// default); the f16 lm_head debug knob
    /// `RVLLM_QWEN36_LM_HEAD_F16=1` continues to route through the
    /// eager closer because graph capture is opt-in via
    /// `RVLLM_QWEN36_DECODE_GRAPH=1` and the two are mutually
    /// exclusive operator gates.
    ///
    /// This path is the closer half of the Phase 8 captured forward.
    /// The corresponding workspace-based decode-step launch body
    /// (the per-layer chain that produces `hidden_region`) is the
    /// follow-up sub-commit; until it lands, this closer is
    /// callable for partial validation (eager prefill + eager
    /// per-layer chain + device-argmax closer + post-fence DtoH).
    #[cfg(feature = "cuda")]
    fn forward_qwen36_outside_closer_device_argmax(
        &self,
        hidden_dev_ptr: u64,
        num_tokens: u32,
        hidden: u32,
        vocab: u32,
        last_idx: usize,
        argmax_token_dev: u64,
    ) -> Result<()> {
        let _ = num_tokens; // kept for parity with the eager closer
        let eps = self.arch.base.rms_norm_eps;
        let hidden_fp8_bytes = hidden as usize;
        let hidden_scale_bytes = 4usize;
        let logits_bytes = (vocab as usize) * 2;
        let hidden_fp8_region =
            self.arena.region("qwen36_pl_h_fp8_devarg", hidden_fp8_bytes, 16)?;
        let hidden_scale_region =
            self.arena.region("qwen36_pl_h_scale_devarg", hidden_scale_bytes, 16)?;
        let logits_region =
            self.arena.region("qwen36_pl_logits_devarg", logits_bytes, 16)?;
        let stream_raw = self.stream.raw() as u64;
        let last_hidden_row_ptr =
            hidden_dev_ptr + (last_idx as u64) * (hidden as u64) * 2;
        unsafe {
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: 1,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_fused_rmsnorm_fp8_quant,
                hidden_fp8_region.device_ptr(),
                hidden_scale_region.device_ptr(),
                last_hidden_row_ptr,
                self.model.outside.final_norm.offset_bytes,
                stream_raw,
            )?;
            self.cublaslt.fp8_gemm(
                hidden_fp8_region.device_ptr(),
                self.model.outside.lm_head_fp8.offset_bytes,
                logits_region.device_ptr(),
                1,
                vocab as i32,
                hidden as i32,
                hidden_scale_region.device_ptr(),
                self.model.outside.lm_head_fp8.scale_ptr,
                stream_raw,
            )?;
            use cudarc::driver::sys::*;
            let mut logits_ptr = logits_region.device_ptr();
            let mut out_ptr = argmax_token_dev;
            let mut vs = vocab as i32;
            let args = [
                (&mut logits_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut vs) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_argmax_f16.raw() as CUfunction,
                1, 1, 1,
                512, 1, 1,
                0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 device-argmax_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // NO stream.fence() here, NO DtoH — the captured graph must
        // contain only kernels. Caller does the fence + DtoH OUTSIDE
        // the captured body via `argmax_dev_to_host_token`.
        Ok(())
    }

    /// Phase 8 commit 2: outside-the-captured-body argmax extractor.
    /// Fences the stream then DtoHs a single 4-byte token id from
    /// `argmax_token_dev`. Called AFTER `graph.replay()` (or eager
    /// `forward_qwen36_outside_closer_device_argmax`) returns.
    #[cfg(feature = "cuda")]
    pub fn argmax_dev_to_host_token(&self, argmax_token_dev: u64) -> Result<i32> {
        self.stream.fence()?;
        let mut tok_buf = [0i32; 1];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                tok_buf.as_mut_ptr() as *mut _, argmax_token_dev, 4);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 argmax_dev_to_host_token DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(tok_buf[0])
    }

    fn forward_qwen36_outside_closer(
        &self,
        hidden_dev_ptr: u64,
        num_tokens: u32,
        hidden: u32,
        vocab: u32,
        last_idx: usize,
    ) -> Result<i32> {
        // Only the LAST token's argmax is consumed downstream. The
        // earlier code ran rmsnorm + fp8_gemm over ALL `num_tokens`
        // rows and then argmax'd the last; on a 4 k-token prompt
        // with 262 k vocab that materialised ≈ 2 GiB of logits +
        // 4096× the lm_head FLOPs we actually needed. Slice down to
        // the single relevant row before doing any work.
        let _ = num_tokens; // kept in the signature for back-compat
        let eps = self.arch.base.rms_norm_eps;
        let hidden_fp8_bytes = hidden as usize;
        let hidden_scale_bytes = 4usize;
        let logits_bytes = (vocab as usize) * 2;
        let hidden_fp8_region = self.arena.region("qwen36_pl_h_fp8", hidden_fp8_bytes, 16)?;
        let hidden_scale_region = self
            .arena
            .region("qwen36_pl_h_scale", hidden_scale_bytes, 16)?;
        let logits_region = self.arena.region("qwen36_pl_logits", logits_bytes, 16)?;
        let stream_raw = self.stream.raw() as u64;
        // Pointer to the last token's hidden row inside the full
        // [num_tokens, hidden] f16 buffer (caller passes the buffer
        // base — either an arena `hidden_region.device_ptr()` or the
        // persistent workspace `hidden_dev` slot).
        let last_hidden_row_ptr =
            hidden_dev_ptr + (last_idx as u64) * (hidden as u64) * 2;
        unsafe {
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: 1,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_fused_rmsnorm_fp8_quant,
                hidden_fp8_region.device_ptr(),
                hidden_scale_region.device_ptr(),
                last_hidden_row_ptr,
                self.model.outside.final_norm.offset_bytes,
                stream_raw,
            )?;
        }
        if std::env::var("RVLLM_QWEN36_LM_HEAD_F16").as_deref() == Ok("1") {
            let normed_f16_region =
                self.arena
                    .region("qwen36_pl_h_normed_f16", hidden as usize * 2, 16)?;
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let rc = cuMemcpyDtoDAsync_v2(
                    normed_f16_region.device_ptr(),
                    last_hidden_row_ptr,
                    hidden as usize * 2,
                    stream_raw as CUstream,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 closer f16 row DtoD",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            unsafe {
                rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                    num_tokens: 1,
                    hidden,
                    eps,
                }
                .launch(
                    self.outside_kernels.fn_rmsnorm_inplace_f16,
                    normed_f16_region.device_ptr(),
                    self.model.outside.final_norm.offset_bytes,
                    stream_raw,
                )?;
            }
            let logits_f32_region =
                self.arena
                    .region("qwen36_pl_logits_f32", vocab as usize * 4, 16)?;
            #[cfg(feature = "cuda")]
            unsafe {
                self.cublaslt.f16_gemm_f32(
                    normed_f16_region.device_ptr(),
                    self.model.outside.lm_head.offset_bytes,
                    logits_f32_region.device_ptr(),
                    1,
                    vocab as i32,
                    hidden as i32,
                    stream_raw,
                )?;
            }
            let token_region = self.arena.region("qwen36_pl_token_f32", 4, 16)?;
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut logits_ptr = logits_f32_region.device_ptr();
                let mut out_ptr = token_region.device_ptr();
                let mut vs = vocab as i32;
                let args = [
                    (&mut logits_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut vs) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 512;
                let grid: u32 = 1;
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_argmax.raw() as CUfunction,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 argmax_f32 launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            self.stream.fence()?;
            let mut tok_buf = [0i32; 1];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let rc =
                    cuMemcpyDtoH_v2(tok_buf.as_mut_ptr() as *mut _, token_region.device_ptr(), 4);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 closer f32 token DtoH",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            return Ok(tok_buf[0]);
        }
        #[cfg(feature = "cuda")]
        unsafe {
            self.cublaslt.fp8_gemm(
                hidden_fp8_region.device_ptr(),
                self.model.outside.lm_head_fp8.offset_bytes,
                logits_region.device_ptr(),
                1,
                vocab as i32,
                hidden as i32,
                hidden_scale_region.device_ptr(),
                self.model.outside.lm_head_fp8.scale_ptr,
                stream_raw,
            )?;
        }
        // Phase 4b-prep iter27: GPU argmax over the f16 logits row,
        // replacing a `vocab * 2`-byte DtoH (~524 KiB at vocab=262K)
        // + host max-loop with a single 4-byte device→host copy of
        // the winning token id. With the slice-to-last-row above,
        // the argmax now reads from offset 0 (single-row buffer).
        let token_region = self.arena.region("qwen36_pl_token", 4, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut logits_ptr = logits_region.device_ptr();
            let mut out_ptr = token_region.device_ptr();
            let mut vs = vocab as i32;
            let args = [
                (&mut logits_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut vs) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 512;
            let grid: u32 = 1;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_argmax_f16.raw() as CUfunction,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 argmax_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;
        let mut tok_buf = [0i32; 1];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            // Consistency with the gemma4 closer: a swallowed DtoH
            // failure used to leave `tok_buf[0] == 0`, which the
            // server then emitted as a valid output token.
            let rc = cuMemcpyDtoH_v2(tok_buf.as_mut_ptr() as *mut _, token_region.device_ptr(), 4);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 closer token DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(tok_buf[0])
    }

    /// Spec-decode closer-all: runs RMSNorm + fp8_quant + lm_head +
    /// argmax for ALL `rows` positions in one pass. Replaces the
    /// codex-flagged K-times-closer loop in
    /// `forward_qwen36_decode_argmax_all` which paid K fences +
    /// K DtoHs (each 4 bytes) + K small lm_head GEMMs at M=1.
    ///
    /// Costs per call: ONE fused_rmsnorm_fp8_quant (row-batched
    /// via num_tokens), ONE fp8_gemm at M=rows (much better GEMM
    /// shape than M=1 on small/medium K), ONE argmax_f16_kernel
    /// launch with grid=rows (already block-per-row in
    /// kernels/argmax.cu:21), ONE stream fence, ONE DtoH of
    /// rows*4 bytes.
    ///
    /// `out_tokens`: caller-owned Vec; cleared and refilled with
    /// `rows` argmax ids in row order.
    #[cfg(feature = "cuda")]
    fn forward_qwen36_outside_closer_all(
        &self,
        hidden_dev_ptr: u64,
        rows: u32,
        hidden: u32,
        vocab: u32,
        out_tokens: &mut Vec<i32>,
    ) -> Result<()> {
        if rows == 0 {
            out_tokens.clear();
            return Ok(());
        }
        let eps = self.arch.base.rms_norm_eps;
        let rows_us = rows as usize;
        let hidden_us = hidden as usize;
        let vocab_us = vocab as usize;
        // [rows, hidden] fp8 + [rows] f32 scales + [rows, vocab] f16 logits + [rows] i32 tokens.
        let hidden_fp8_region =
            self.arena
                .region("qwen36_spec_closer_h_fp8", rows_us * hidden_us, 16)?;
        let hidden_scale_region =
            self.arena
                .region("qwen36_spec_closer_h_scale", rows_us * 4, 16)?;
        let logits_region =
            self.arena
                .region("qwen36_spec_closer_logits", rows_us * vocab_us * 2, 16)?;
        let tokens_region = self
            .arena
            .region("qwen36_spec_closer_tokens", rows_us * 4, 16)?;
        let stream_raw = self.stream.raw() as u64;

        unsafe {
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: rows,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_fused_rmsnorm_fp8_quant,
                hidden_fp8_region.device_ptr(),
                hidden_scale_region.device_ptr(),
                hidden_dev_ptr,
                self.model.outside.final_norm.offset_bytes,
                stream_raw,
            )?;
        }
        unsafe {
            // NOTE: codex review proposed routing rows>1 through
            // fp8_gemm_f16_per_row_act_scale (B_SCALE_MODE =
            // OUTER_VEC_32F) to fix the closer-all per-row scale
            // bug. The descriptor sets cleanly but
            // cublasLtMatmul returns LaunchFailed on sm_121 / GB10
            // for the FP8 + B-side outer-vector combo. Hypothesis:
            // cuBLASLt's FP8 path on sm_121 only accepts
            // A_SCALE_MODE=OUTER_VEC (weight side) and rejects the
            // analog on the B (activation) side. Verifying that
            // and finding the right cuBLASLt enum + hint flag is
            // follow-up work; the entrypoint stays available so
            // future probes can use it.
            //
            // Until then: closer-all keeps using plain fp8_gemm
            // with the scalar-scale bug, which is why the gate is
            // OFF by default and the closer-loop path is the
            // production route.
            self.cublaslt.fp8_gemm(
                hidden_fp8_region.device_ptr(),
                self.model.outside.lm_head_fp8.offset_bytes,
                logits_region.device_ptr(),
                rows as i32,
                vocab as i32,
                hidden as i32,
                hidden_scale_region.device_ptr(),
                self.model.outside.lm_head_fp8.scale_ptr,
                stream_raw,
            )?;
        }
        unsafe {
            use cudarc::driver::sys::*;
            let mut logits_ptr = logits_region.device_ptr();
            let mut out_ptr = tokens_region.device_ptr();
            let mut vs = vocab as i32;
            let args = [
                (&mut logits_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut vs) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 512;
            // grid=rows: argmax_f16_kernel uses blockIdx.x to pick its row.
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_argmax_f16.raw() as CUfunction,
                rows,
                1,
                1,
                block,
                1,
                1,
                0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 spec closer-all argmax_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;
        out_tokens.clear();
        out_tokens.resize(rows_us, 0i32);
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                out_tokens.as_mut_ptr() as *mut _,
                tokens_region.device_ptr(),
                rows_us * 4,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 spec closer-all tokens DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    pub fn forward_outside_smoke(&self) -> Result<()> {
        let hidden = self.arch.base.hidden_size as u32;
        let vocab = self.arch.base.vocab_size as u32;
        // Hardcoded canary tokens: BOS-likely + a few mid-vocab IDs.
        // Real generation needs the tokenizer; we just want non-zero
        // embeddings for the kernel output.
        let token_ids: [i32; 4] = [1, 100, 1000, 10_000];
        let num_tokens = token_ids.len() as u32;

        // Allocate device regions: token IDs (i32) + hidden state (f16).
        let tokens_region =
            self.arena
                .region("qwen36_smoke_tokens", std::mem::size_of_val(&token_ids), 16)?;
        let mut token_bytes = Vec::with_capacity(token_ids.len() * 4);
        for t in &token_ids {
            token_bytes.extend_from_slice(&t.to_le_bytes());
        }
        unsafe { tokens_region.copy_from_host(&token_bytes)? };

        let hidden_bytes = (num_tokens as usize) * (hidden as usize) * 2; // f16
        let hidden_region = self.arena.region("qwen36_smoke_hidden", hidden_bytes, 16)?;

        // Launch embedding_gather_f16 via the standard rvllm-fused
        // ABI used by Gemma 4.
        let launch = rvllm_fused::EmbeddingGatherLaunch {
            num_tokens,
            hidden,
            vocab,
        };
        unsafe {
            launch.launch(
                self.outside_kernels.fn_embedding_gather_f16,
                hidden_region.device_ptr(),
                self.model.outside.embed_tokens.offset_bytes,
                tokens_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }

        // Phase 3g: fused (RMSNorm + FP8-quantize) → cuBLASLt
        // FP8 matmul against lm_head → DtoH logits → CPU argmax.
        // Mirrors Gemma 4's outside-only path (gemma4_bring_up.rs:2058).
        let eps = self.arch.base.rms_norm_eps;
        let hidden_fp8_bytes = (num_tokens as usize) * (hidden as usize); // 1 byte/elem
        let hidden_scale_bytes = (num_tokens as usize) * 4; // f32/token
        let logits_bytes = (num_tokens as usize) * (vocab as usize) * 2; // f16
        let hidden_fp8_region =
            self.arena
                .region("qwen36_smoke_hidden_fp8", hidden_fp8_bytes, 16)?;
        let hidden_scale_region =
            self.arena
                .region("qwen36_smoke_hidden_scale", hidden_scale_bytes, 16)?;
        let logits_region = self.arena.region("qwen36_smoke_logits", logits_bytes, 16)?;

        let stream_raw = self.stream.raw() as u64;

        unsafe {
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_fused_rmsnorm_fp8_quant,
                hidden_fp8_region.device_ptr(),
                hidden_scale_region.device_ptr(),
                hidden_region.device_ptr(),
                self.model.outside.final_norm.offset_bytes,
                stream_raw,
            )?;
        }

        // cuBLASLt fp8_gemm: D = A * B^T.
        //   A = hidden_fp8 [num_tokens, hidden]
        //   B = lm_head_fp8 [vocab, hidden]
        //   D = logits f16 [num_tokens, vocab]
        #[cfg(feature = "cuda")]
        unsafe {
            self.cublaslt.fp8_gemm(
                hidden_fp8_region.device_ptr(),
                self.model.outside.lm_head_fp8.offset_bytes,
                logits_region.device_ptr(),
                num_tokens as i32,
                vocab as i32,
                hidden as i32,
                hidden_scale_region.device_ptr(),
                self.model.outside.lm_head_fp8.scale_ptr,
                stream_raw,
            )?;
        }
        self.stream.fence()?;

        // DtoH the first 4 normalized hidden values for sanity, plus
        // token-0's full logits row for CPU-side argmax (the f16→f32
        // upcast + argmax fits in <10 ms on host for vocab=248k).
        let mut hidden_probe = [0u8; 8];
        let logits_row_bytes = (vocab as usize) * 2;
        let mut logits_row_f16 = vec![0u8; logits_row_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc1 = cuMemcpyDtoH_v2(
                hidden_probe.as_mut_ptr() as *mut _,
                hidden_region.device_ptr(),
                hidden_probe.len(),
            );
            let rc2 = cuMemcpyDtoH_v2(
                logits_row_f16.as_mut_ptr() as *mut _,
                logits_region.device_ptr(),
                logits_row_bytes,
            );
            if rc1 != CUresult::CUDA_SUCCESS || rc2 != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_smoke DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let f0 = f16_bits_to_f32(u16::from_le_bytes([hidden_probe[0], hidden_probe[1]]));
        let f1 = f16_bits_to_f32(u16::from_le_bytes([hidden_probe[2], hidden_probe[3]]));
        let f2 = f16_bits_to_f32(u16::from_le_bytes([hidden_probe[4], hidden_probe[5]]));
        let f3 = f16_bits_to_f32(u16::from_le_bytes([hidden_probe[6], hidden_probe[7]]));

        let mut best_logit = f32::NEG_INFINITY;
        let mut best_token: i32 = -1;
        for v in 0..vocab as usize {
            let bits = u16::from_le_bytes([logits_row_f16[v * 2], logits_row_f16[v * 2 + 1]]);
            let l = f16_bits_to_f32(bits);
            if l > best_logit {
                best_logit = l;
                best_token = v as i32;
            }
        }

        eprintln!(
            "[qwen36] forward_outside_smoke: embed → rmsnorm+fp8 → \
             cublaslt.fp8_gemm → cpu_argmax \
             token_ids={token_ids:?} \
             token0_embed[0..4]=[{f0:.4}, {f1:.4}, {f2:.4}, {f3:.4}] \
             token0_argmax_id={best_token} logit={best_logit:.3} \
             eps={eps:.0e}"
        );
        Ok(())
    }

    pub fn run_generate(&self) -> ! {
        unimplemented!(
            "qwen36 phase 3f+ — outside-only forward (rmsnorm + lm_head + argmax) \
             then per-layer math still TODO"
        );
    }

    pub fn run_bench(&self) -> ! {
        unimplemented!("qwen36 phase 2 — bench harness not yet ported");
    }

    pub fn run_ppl(&self) -> ! {
        unimplemented!("qwen36 phase 2 — ppl harness not yet ported");
    }

    pub fn init_prefix_cache(&self) -> ! {
        unimplemented!("qwen36 phase 2 — prefix cache not yet ported");
    }

    /// Phase 3a (Qwen batched-prefill plan): single dispatch point for
    /// every per-layer projection in `apply_layer_*`. Today every
    /// projection call site instantiates `Fp8GemvF16InLaunch { m, n, k }`
    /// directly; routing them through this method is the prerequisite
    /// for Phase 4, which will batch the per-token loop and pass
    /// `m = num_tokens` instead of `m = 1`.
    ///
    /// Routing:
    /// * **m = 1**: delegates byte-identically to
    ///   `Fp8GemvF16InLaunch { m: 1, n, k }`. Existing tests +
    ///   determinism canaries stay green.
    /// * **m ≥ 2**: returns a typed error pointing at Phase 3b. The
    ///   plan there is to:
    ///     1. quantize the f16 input to fp8 + per-token f32 amax via
    ///        a small `fp8_quantize_per_token_f16` kernel (new),
    ///     2. either pass the existing `[N/128, K/128]` row-major
    ///        weight blockscale straight into `cublaslt.fp8_gemm` (if
    ///        the layout is acceptable to cuBLASLt) or transpose it
    ///        into MN-major like Gemma does for the CUTLASS path,
    ///     3. dispatch to `self.cublaslt.fp8_gemm` for tensor-core
    ///        throughput at large m, then validate per-shape cosine
    ///        ≥ 0.9999 against a reference implementation that loops
    ///        the m=1 GEMV N times.
    ///   The error message names Phase 3b explicitly so a future
    ///   Phase 4 patch that flips the caller to m=N immediately
    ///   surfaces "Phase 3b not done" instead of producing silently
    ///   wrong tokens.
    ///
    /// `kernel_gemv` is the resolved
    /// `fn_fp8_gemv_wpr_native_f16in` handle (the caller is expected
    /// to have already failed-fast if it's not loaded — see
    /// `forward_qwen36_decode`'s ok_or_else).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_proj_dispatch(
        &self,
        kernel_gemv: rvllm_kernels::KernelFn,
        out_f16: u64,
        weight_fp8: u64,
        b_blockscale: u64,
        input_f16: u64,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(rvllm_core::RvllmError::cuda(
                "qwen36 fp8_proj_dispatch: zero-size dim",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        if m == 1 {
            // SAFETY: caller-supplied pointers are already-validated
            // device addresses; Fp8GemvF16InLaunch internally re-checks
            // K%8 alignment.
            return rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m, n, k }.launch(
                kernel_gemv,
                out_f16,
                weight_fp8,
                b_blockscale,
                input_f16,
                stream,
            );
        }
        // Phase 3c: when m ≥ 128 AND CutlassBackend::SoSm120 is loaded,
        // dispatch through CUTLASS SM120's blockwise FP8 GEMM (the
        // same .so Gemma uses at lm_head). This is the production
        // fast path on sm_121 (GB10) — cuBLASLt has no blockwise
        // FP8 kernel for that arch; CUTLASS SM120 hard-asserts
        // M≥128 so smaller m falls through to Phase 3b's cuBLASLt
        // try-then-looped-GEMV path.
        #[cfg(feature = "cuda")]
        if m >= 128 {
            if let CutlassBackend::SoSm120(ref lib) = self.cutlass {
                // Per-token amax quantise (CUTLASS prep_sfa expects [M] f32).
                let fp8_bytes = (m as usize) * (k as usize);
                let amax_bytes = (m as usize) * 4;
                let in_fp8 = self
                    .arena
                    .region("qwen36_proj_in_fp8_cutlass", fp8_bytes, 16)?;
                let in_amax = self.arena.region("qwen36_proj_in_amax", amax_bytes, 16)?;
                unsafe {
                    use cudarc::driver::sys::*;
                    let block_dim: u32 = (k as u32).min(1024);
                    let mut o_fp8 = in_fp8.device_ptr();
                    let mut o_amax = in_amax.device_ptr();
                    let mut i_ptr = input_f16;
                    let mut k_i: i32 = k as i32;
                    let args = [
                        (&mut o_fp8) as *mut u64 as *mut core::ffi::c_void,
                        (&mut o_amax) as *mut u64 as *mut core::ffi::c_void,
                        (&mut i_ptr) as *mut u64 as *mut core::ffi::c_void,
                        (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let rc = cuLaunchKernel(
                        self.outside_kernels
                            .fn_fp8_quantize_per_token_amax_f16
                            .raw() as CUfunction,
                        m,
                        1,
                        1,
                        block_dim,
                        1,
                        1,
                        0,
                        stream as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 fp8_proj_dispatch: amax-quantise launch (CUTLASS path)",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                // Allocate SFA / SFB / workspace per CUTLASS sizing.
                let sfa_n = lib.sfa_bytes(m as i32, k as i32);
                let sfb_n = lib.sfb_bytes(n as i32, k as i32);
                let ws_n = lib.workspace_size(m as i32, n as i32, k as i32);
                if sfa_n == 0 || sfb_n == 0 {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 fp8_proj_dispatch: CUTLASS SM120 reported \
                         sfa_bytes/sfb_bytes==0 — legacy .so without these \
                         helpers; rebuild kernels/build_cutlass_sm120_so.sh",
                        rvllm_core::CudaErrorKind::Other,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
                let sfa = self.arena.region("qwen36_proj_sfa", sfa_n.max(4), 16)?;
                let sfb = self.arena.region("qwen36_proj_sfb", sfb_n.max(4), 16)?;
                let ws = self
                    .arena
                    .region("qwen36_proj_cutlass_ws", ws_n.max(16), 256)?;
                unsafe {
                    lib.launch_prep_sfa(
                        in_amax.device_ptr(),
                        sfa.device_ptr(),
                        m as i32,
                        k as i32,
                        stream,
                    )?;
                    lib.launch_prep_sfb(
                        b_blockscale,
                        sfb.device_ptr(),
                        n as i32,
                        k as i32,
                        stream,
                    )?;
                    lib.launch_fp8_gemm_blockscale(
                        out_f16,
                        in_fp8.device_ptr(),
                        weight_fp8,
                        sfa.device_ptr(),
                        sfb.device_ptr(),
                        m as i32,
                        n as i32,
                        k as i32,
                        ws.device_ptr(),
                        ws_n,
                        stream,
                    )?;
                }
                return Ok(());
            }
        }

        // Phase 3c-pad (Codex review #4-B): CUTLASS-with-M-pad
        // for 2 ≤ m < 128. **ABANDONED** — kept gated off
        // for archival reference; the path is structurally
        // unsuitable for sm_121 + this checkpoint, not a fixable
        // bug. Toggle `RVLLM_QWEN36_FP8_PAD_DEBUG=1` to enable
        // and `RVLLM_QWEN36_FP8_PAD_DIFF=1` to log per-row diff
        // stats vs the looped-GEMV reference (used during the
        // 2026-05-13 root-cause pass).
        //
        // Root cause (confirmed by side-by-side diff):
        //
        // `Fp8GemvF16InLaunch` (production sm_121 path) reads
        // **f16 activation × fp8 weight** — no activation
        // quantization. The CUTLASS SM120 blockwise FP8 GEMM
        // does **fp8 act × fp8 weight** after a per-token amax
        // quantize — additional per-row quantization noise.
        //
        // Measured diff at the first M<128 dispatch (m=5,
        // n=8192, k=2048) on a vision request:
        //   overall_max  = 0.25
        //   mean_per_row = 0.155
        // On a layer output of magnitude ~2 that's 10–15%
        // relative error per element. The model is robust to
        // this noise on plain text (greedy decode survives
        // small perturbations) but vision-spliced inputs
        // drift semantically — ball.png deterministically
        // hallucinated "Königslutter am Elm".
        //
        // The m≥128 native path on sm_121 has the SAME
        // numerics — but `BLOCKWISE_STATE` caches `NoAlgo`
        // after the first call (cuBLASLt has no kernel for
        // sm_121) so m≥128 also falls through to looped
        // GEMV in production. The CUTLASS SM120 .so is built
        // and present, just never invoked on sm_121 — the f16
        // activation path is the production reference for
        // every M on this arch.
        //
        // Right fix: the row-batched FP8 GEMV (Codex #4-B
        // alternative — already shipped at commit 31131b1).
        // Preserves f16 activation precision, batches the
        // launch overhead across the row dimension, and is
        // numerically identical to the m=1 reference on
        // EVERY input.
        //
        // The CUTLASS-pad scaffold + DIFF instrumentation
        // stay in tree so a future GB10 driver release that
        // adds a true cuBLASLt sm_121 blockwise kernel (or a
        // CUTLASS path that preserves f16 activations) can
        // re-evaluate quickly.
        let pad_debug = std::env::var("RVLLM_QWEN36_FP8_PAD_DEBUG")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        #[cfg(feature = "cuda")]
        if pad_debug && m >= 2 && m < 128 {
            if let CutlassBackend::SoSm120(ref lib) = self.cutlass {
                const PAD_M: u32 = 128;
                let pad_in_bytes = (PAD_M as usize) * (k as usize) * 2;
                let pad_out_bytes = (PAD_M as usize) * (n as usize) * 2;
                let pad_in = self
                    .arena
                    .region("qwen36_proj_pad_in_f16", pad_in_bytes, 16)?;
                let pad_out = self
                    .arena
                    .region("qwen36_proj_pad_out_f16", pad_out_bytes, 16)?;
                // Build pad_in as: real m rows from input_f16, then
                // replicate row 0 into rows m..PAD_M-1. Replicating
                // (rather than zero-padding) gives all 128 rows a
                // similar per-token amax, so the CUTLASS prep_sfa
                // chunk-level reduction picks a scale that
                // represents the real data — not artificially
                // inflated by amax=0 fallback rows (the cause of the
                // wrong output in the prior commit).
                unsafe {
                    use cudarc::driver::sys::*;
                    let row_bytes = (k as usize) * 2;
                    // (a) DtoD M real rows starting at row 0.
                    let m_rows_bytes = (m as usize) * row_bytes;
                    let rc = cuMemcpyDtoDAsync_v2(
                        pad_in.device_ptr(),
                        input_f16,
                        m_rows_bytes,
                        stream as CUstream,
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 fp8_proj_dispatch: pad-in DtoD (real rows)",
                            rvllm_core::CudaErrorKind::MemcpyFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                    // (b) Replicate row 0 into rows m..PAD_M-1
                    //     (PAD_M-m small DtoD copies).
                    for r in (m as u64)..(PAD_M as u64) {
                        let dst = pad_in.device_ptr() + r * (row_bytes as u64);
                        let rc =
                            cuMemcpyDtoDAsync_v2(dst, input_f16, row_bytes, stream as CUstream);
                        if rc != CUresult::CUDA_SUCCESS {
                            return Err(rvllm_core::RvllmError::cuda(
                                "qwen36 fp8_proj_dispatch: pad-in row-0 replication",
                                rvllm_core::CudaErrorKind::MemcpyFailed,
                                rvllm_core::CudaCtx::setup(),
                            ));
                        }
                    }
                }
                // Per-token amax-quantise the padded input.
                let fp8_bytes = (PAD_M as usize) * (k as usize);
                let amax_bytes = (PAD_M as usize) * 4;
                let in_fp8 = self.arena.region("qwen36_proj_pad_in_fp8", fp8_bytes, 16)?;
                let in_amax = self
                    .arena
                    .region("qwen36_proj_pad_in_amax", amax_bytes, 16)?;
                unsafe {
                    use cudarc::driver::sys::*;
                    let block_dim: u32 = (k as u32).min(1024);
                    let mut o_fp8 = in_fp8.device_ptr();
                    let mut o_amax = in_amax.device_ptr();
                    let mut i_ptr = pad_in.device_ptr();
                    let mut k_i: i32 = k as i32;
                    let args = [
                        (&mut o_fp8) as *mut u64 as *mut core::ffi::c_void,
                        (&mut o_amax) as *mut u64 as *mut core::ffi::c_void,
                        (&mut i_ptr) as *mut u64 as *mut core::ffi::c_void,
                        (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let rc = cuLaunchKernel(
                        self.outside_kernels
                            .fn_fp8_quantize_per_token_amax_f16
                            .raw() as CUfunction,
                        PAD_M,
                        1,
                        1,
                        block_dim,
                        1,
                        1,
                        0,
                        stream as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 fp8_proj_dispatch: pad amax-quantise launch",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                // CUTLASS SM120 launch at M=PAD_M.
                let sfa_n = lib.sfa_bytes(PAD_M as i32, k as i32);
                let sfb_n = lib.sfb_bytes(n as i32, k as i32);
                let ws_n = lib.workspace_size(PAD_M as i32, n as i32, k as i32);
                if sfa_n == 0 || sfb_n == 0 {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 fp8_proj_dispatch: CUTLASS SM120 pad path \
                         reported sfa_bytes/sfb_bytes==0",
                        rvllm_core::CudaErrorKind::Other,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
                let sfa = self.arena.region("qwen36_proj_pad_sfa", sfa_n.max(4), 16)?;
                let sfb = self.arena.region("qwen36_proj_pad_sfb", sfb_n.max(4), 16)?;
                let ws = self.arena.region("qwen36_proj_pad_ws", ws_n.max(16), 256)?;
                unsafe {
                    lib.launch_prep_sfa(
                        in_amax.device_ptr(),
                        sfa.device_ptr(),
                        PAD_M as i32,
                        k as i32,
                        stream,
                    )?;
                    lib.launch_prep_sfb(
                        b_blockscale,
                        sfb.device_ptr(),
                        n as i32,
                        k as i32,
                        stream,
                    )?;
                    lib.launch_fp8_gemm_blockscale(
                        pad_out.device_ptr(),
                        in_fp8.device_ptr(),
                        weight_fp8,
                        sfa.device_ptr(),
                        sfb.device_ptr(),
                        PAD_M as i32,
                        n as i32,
                        k as i32,
                        ws.device_ptr(),
                        ws_n,
                        stream,
                    )?;
                }
                // Optional diff instrumentation: RVLLM_QWEN36_FP8_PAD_DIFF=1
                // runs the looped-GEMV reference on the same input
                // and compares pad_out[0..M] against ref_out[0..M],
                // dumping per-row max-abs-diff + a few row samples
                // to the journal. Fires only on the first M < 128
                // dispatch per process so it doesn't flood logs.
                use std::sync::atomic::{AtomicBool, Ordering};
                static DIFF_DONE: AtomicBool = AtomicBool::new(false);
                let diff_on = std::env::var("RVLLM_QWEN36_FP8_PAD_DIFF")
                    .map(|v| v != "0" && !v.is_empty())
                    .unwrap_or(false);
                if diff_on && !DIFF_DONE.swap(true, Ordering::Relaxed) {
                    let ref_bytes = (m as usize) * (n as usize) * 2;
                    let ref_region = self.arena.region("qwen36_proj_pad_ref", ref_bytes, 16)?;
                    rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m, n, k }.launch(
                        kernel_gemv,
                        ref_region.device_ptr(),
                        weight_fp8,
                        b_blockscale,
                        input_f16,
                        stream,
                    )?;
                    self.stream.fence()?;
                    let mut pad_host = vec![0u16; (m as usize) * (n as usize)];
                    let mut ref_host = vec![0u16; (m as usize) * (n as usize)];
                    use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(
                        pad_host.as_mut_ptr() as *mut _,
                        pad_out.device_ptr(),
                        ref_bytes,
                    );
                    let _ = cuMemcpyDtoH_v2(
                        ref_host.as_mut_ptr() as *mut _,
                        ref_region.device_ptr(),
                        ref_bytes,
                    );
                    let f16_to_f32 = |x: u16| -> f32 {
                        let sign = ((x >> 15) & 1) as u32;
                        let exp = ((x >> 10) & 0x1f) as i32;
                        let mant = (x & 0x3ff) as u32;
                        if exp == 0 {
                            if mant == 0 {
                                return if sign == 1 { -0.0 } else { 0.0 };
                            }
                            // subnormal
                            let f = mant as f32 / 1024.0;
                            return (if sign == 1 { -1.0 } else { 1.0 }) * f * (2.0f32).powi(-14);
                        }
                        if exp == 0x1f {
                            return if mant == 0 {
                                if sign == 1 {
                                    f32::NEG_INFINITY
                                } else {
                                    f32::INFINITY
                                }
                            } else {
                                f32::NAN
                            };
                        }
                        let f = 1.0 + (mant as f32 / 1024.0);
                        (if sign == 1 { -1.0 } else { 1.0 }) * f * (2.0f32).powi(exp - 15)
                    };
                    let mut max_per_row: Vec<f32> = Vec::with_capacity(m as usize);
                    for r in 0..(m as usize) {
                        let mut max_abs = 0.0f32;
                        for c in 0..(n as usize) {
                            let p = f16_to_f32(pad_host[r * (n as usize) + c]);
                            let q = f16_to_f32(ref_host[r * (n as usize) + c]);
                            let d = (p - q).abs();
                            if d > max_abs {
                                max_abs = d;
                            }
                        }
                        max_per_row.push(max_abs);
                    }
                    let overall_max = max_per_row.iter().fold(0.0f32, |a, &b| a.max(b));
                    let mean_per_row: f32 = max_per_row.iter().sum::<f32>() / (m as f32);
                    tracing::warn!(
                        m,
                        n,
                        k,
                        overall_max,
                        mean_per_row,
                        "fp8_proj_pad_diff: first M<128 dispatch, per-row max-abs-diff stats"
                    );
                    // Sample dump: rows 0, m-1, plus rows with the
                    // largest diffs so we can see whether the bad
                    // rows cluster.
                    let mut ranked: Vec<(usize, f32)> =
                        max_per_row.iter().copied().enumerate().collect();
                    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let worst_rows: Vec<usize> = ranked.iter().take(4).map(|&(r, _)| r).collect();
                    let dump_rows: Vec<usize> = {
                        let mut v: Vec<usize> = vec![0, (m as usize) - 1];
                        for r in worst_rows {
                            if !v.contains(&r) {
                                v.push(r);
                            }
                        }
                        v
                    };
                    for r in dump_rows {
                        let p0 = f16_to_f32(pad_host[r * (n as usize) + 0]);
                        let p1 = f16_to_f32(pad_host[r * (n as usize) + 1]);
                        let p2 = f16_to_f32(pad_host[r * (n as usize) + 2]);
                        let p3 = f16_to_f32(pad_host[r * (n as usize) + 3]);
                        let q0 = f16_to_f32(ref_host[r * (n as usize) + 0]);
                        let q1 = f16_to_f32(ref_host[r * (n as usize) + 1]);
                        let q2 = f16_to_f32(ref_host[r * (n as usize) + 2]);
                        let q3 = f16_to_f32(ref_host[r * (n as usize) + 3]);
                        let max_abs = max_per_row[r];
                        tracing::warn!(
                            row = r,
                            max_abs,
                            pad_first4 = format!("[{:.4} {:.4} {:.4} {:.4}]", p0, p1, p2, p3),
                            ref_first4 = format!("[{:.4} {:.4} {:.4} {:.4}]", q0, q1, q2, q3),
                            "fp8_proj_pad_diff: row sample"
                        );
                    }
                }
                // DtoD the first M rows of pad_out into out_f16.
                unsafe {
                    use cudarc::driver::sys::*;
                    let m_rows_bytes = (m as usize) * (n as usize) * 2;
                    let rc = cuMemcpyDtoDAsync_v2(
                        out_f16,
                        pad_out.device_ptr(),
                        m_rows_bytes,
                        stream as CUstream,
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 fp8_proj_dispatch: pad-out DtoD",
                            rvllm_core::CudaErrorKind::MemcpyFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                return Ok(());
            }
        }

        // Phase 3b: m≥2 path. Two sub-paths, gated by a one-time
        // capability probe (Codex review #4):
        //
        //   * If cuBLASLt's blockwise FP8 algo is available
        //     (sm_100 / sm_120 Blackwell-server / RTX 5090),
        //     quantise input → fp8 + per-K-block f32 scale and
        //     dispatch to cuBLASLt.
        //   * If it isn't (sm_121 = GB10 — cuBLASLt has no
        //     blockwise FP8 kernel for that arch today), skip the
        //     quantise scratch entirely and fall straight through
        //     to M× Fp8GemvF16InLaunch on the original `input_f16`.
        //
        // The previous code unconditionally quantised + tried
        // cuBLASLt + caught the "no algo" error every call. On
        // sm_121 that was ~10–30 µs/call of waste in the hot path
        // (and a Vec<f32>-sized scratch allocation churn) for a
        // result that always landed in the looped-GEMV branch.
        //
        // `BLOCKWISE_STATE` is a process-global atomic with
        // states {Unknown=0, NoAlgo=1, HasAlgo=2}. The first
        // dispatch that reaches this branch tries cuBLASLt and
        // sets the state from the result; every subsequent
        // dispatch reads the cached state and skips the wasted
        // work.
        use std::sync::atomic::{AtomicI8, Ordering};
        static BLOCKWISE_STATE: AtomicI8 = AtomicI8::new(0);
        let state = BLOCKWISE_STATE.load(Ordering::Relaxed);
        // Codex review #4-B (alternative): row-batched FP8 GEMV.
        // The fp8_gemv kernel itself already supports M-batching
        // via grid.y; the per-row launch loop was just not using
        // it. One launch at m=M does the same work as M launches
        // at m=1, with ~M× lower CPU/driver overhead. Math is
        // byte-identical — each warp owns the same (m, n) pair
        // it would in the per-row case.
        let run_batched_gemv = || -> Result<()> {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m, n, k }.launch(
                kernel_gemv,
                out_f16,
                weight_fp8,
                b_blockscale,
                input_f16,
                stream,
            )
        };
        if state == 1 {
            // Cached NoAlgo (sm_121 today). Skip quantise + cuBLASLt
            // try, go straight to looped GEMV.
            return run_batched_gemv();
        }
        // state == 0 (Unknown, first call) OR state == 2 (HasAlgo).
        // Both still need the quantised input + a cuBLASLt try.
        let fp8_bytes = (m as usize) * (k as usize);
        let k_blocks = (k as usize + 127) / 128;
        let scale_bytes = (m as usize) * k_blocks * 4;
        let in_fp8 = self.arena.region("qwen36_proj_in_fp8", fp8_bytes, 16)?;
        let in_scale = self.arena.region("qwen36_proj_in_scale", scale_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut out_fp8_ptr = in_fp8.device_ptr();
            let mut out_scale_ptr = in_scale.device_ptr();
            let mut in_ptr = input_f16;
            let mut k_i: i32 = k as i32;
            let args = [
                (&mut out_fp8_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_scale_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut in_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_fp8_quantize_per_token_f16.raw() as CUfunction,
                k_blocks as u32,
                m,
                1,
                128,
                1,
                1,
                0,
                stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 fp8_proj_dispatch: fp8_quantize_per_token_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        #[cfg(feature = "cuda")]
        let blockwise_result = self.cublaslt.fp8_gemm_blockwise(
            in_fp8.device_ptr(),
            weight_fp8,
            out_f16,
            m as i32,
            n as i32,
            k as i32,
            in_scale.device_ptr(),
            b_blockscale,
            stream,
        );
        #[cfg(feature = "cuda")]
        match blockwise_result {
            Ok(()) => {
                if state == 0 {
                    BLOCKWISE_STATE.store(2, Ordering::Relaxed);
                    tracing::info!(
                        "qwen36 fp8_proj_dispatch: cuBLASLt blockwise FP8 \
                         available on this arch; caching HasAlgo state."
                    );
                }
                Ok(())
            }
            Err(_) => {
                if state == 0 {
                    BLOCKWISE_STATE.store(1, Ordering::Relaxed);
                    tracing::warn!(
                        "qwen36 fp8_proj_dispatch: cuBLASLt has no blockwise \
                         FP8 algo on this arch (sm_121/GB10 expected). Caching \
                         NoAlgo — future dispatches skip quantise + heuristic \
                         and go straight to looped GEMV. CUTLASS-pad fast \
                         path for 2 ≤ M < 128 is the next slice."
                    );
                }
                run_batched_gemv()
            }
        }
        // Non-cuda build: the entire m≥2 path was cfg-gated out; we
        // can't actually run anything, so signal a loud error rather
        // than silently returning Ok().
        #[cfg(not(feature = "cuda"))]
        Err(rvllm_core::RvllmError::cuda(
            "qwen36 fp8_proj_dispatch: m≥2 path requires `cuda` feature",
            rvllm_core::CudaErrorKind::Other,
            rvllm_core::CudaCtx::setup(),
        ))
    }
}

/// IEEE 754 f32 → f16 round-to-nearest-even encode without pulling
/// `half` in as a runtime dep here. Saturates to ±MAX_F16 outside
/// the f16 range; NaN preserved.
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp32 = ((bits >> 23) & 0xff) as i32;
    let mant32 = bits & 0x7f_ffff;
    if exp32 == 0xff {
        // NaN / inf
        let mant16 = if mant32 != 0 { 0x200 } else { 0 };
        return (sign << 15) | (0x1f << 10) | mant16;
    }
    let exp_unbiased = exp32 - 127;
    if exp_unbiased >= 16 {
        // Overflow → ±inf.
        return (sign << 15) | (0x1f << 10);
    }
    if exp_unbiased < -24 {
        // Underflow → ±0.
        return sign << 15;
    }
    if exp_unbiased < -14 {
        // Subnormal in f16.
        let shift = (-14 - exp_unbiased) as u32;
        let mant_full = mant32 | (1 << 23);
        let rshift = 13 + shift;
        let m = mant_full >> rshift;
        let round_bit = (mant_full >> (rshift - 1)) & 1;
        let sticky = mant_full & ((1 << (rshift - 1)) - 1);
        let m = m + (round_bit & ((sticky != 0) as u32 | (m & 1)));
        return (sign << 15) | (m as u16 & 0x3ff);
    }
    let exp16 = (exp_unbiased + 15) as u16;
    let mant16 = (mant32 >> 13) as u16;
    let round_bit = (mant32 >> 12) & 1;
    let sticky = mant32 & 0xfff;
    let m = mant16 + (round_bit as u16 & ((sticky != 0) as u16 | (mant16 & 1)));
    if m & 0x400 != 0 {
        let exp16 = exp16 + 1;
        if exp16 >= 0x1f {
            return (sign << 15) | (0x1f << 10);
        }
        return (sign << 15) | (exp16 << 10);
    }
    (sign << 15) | (exp16 << 10) | (m & 0x3ff)
}

/// Decode a stored f16 bit pattern into f32 without pulling in `half`
/// as an explicit dep of `rvllm-runtime`. Mirrors `half::f16::to_f32`
/// for finite values; specials (NaN/inf) are preserved by IEEE
/// composition.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal — renormalise.
            let mut e: i32 = -14;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let m = (m & 0x3ff) << 13;
            (sign << 31) | (((e + 127) as u32) << 23) | m
        }
    } else if exp == 0x1f {
        // NaN or inf.
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp + 112) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}
