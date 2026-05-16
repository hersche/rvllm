//! Gemma 4 engine bring-up.
//!
//! Parallel to `bring_up.rs` for Llama/Qwen. Assembles every subsystem
//! needed for Gemma 4 inference: variable-head attention, dual RoPE
//! tables, per-layer KV head variation, extra kernel modules.
//!
//! Usage: when `config.json` declares `"Gemma3ForCausalLM"` or similar,
//! the top-level dispatcher constructs `Gemma4Bringup` instead of
//! the regular `Bringup`.

use std::path::PathBuf;
use std::sync::Arc;

use rvllm_attention::{AttentionBackend, Fa3Kernels};
use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_cutlass::{CublasLt, CutlassBackend, Policy};
use rvllm_kernels::{KernelFn, KernelLoader, LoadedModule};
use rvllm_mem::{context::CudaContextHandle, stream::Stream, HbmArena};

use crate::gemma4_layer_exec::Gemma4LayerKernels;

// Cycle 56 step 7: cuda_check! macro hoisted to crate root in lib.rs.

/// Per-request sampling configuration handed to [`Gemma4Bringup::run_generate`].
///
/// `Greedy` is unconditional argmax — the runtime does NOT consult
/// `RVLLM_SAMPLING_TEMPERATURE` / `_TOP_P` here. A previous iteration
/// kept that env-var fallback as a "dev knob" for bench/probe, but it
/// also fired for HTTP requests that explicitly sent `temperature: 0`
/// and silently broke their determinism the moment the env var was
/// exported on the box. Stochastic semantics live exclusively on
/// `Stochastic`. Bench/probe binaries call `SamplingConfig::greedy()`
/// and run argmax; if they ever want sampling, they should plumb
/// their own CLI flags into `Stochastic`.
#[derive(Debug, Clone, Copy)]
pub enum SamplingConfig {
    Greedy,
    Stochastic {
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        seed: u64,
    },
}

impl SamplingConfig {
    /// Default for callers that do not opt into per-request sampling
    /// (bench, probe). Always argmax — there is no env-var fallback.
    pub fn greedy() -> Self {
        SamplingConfig::Greedy
    }
}

/// Resolve the effective NVFP4 attention partition size from the
/// environment. Single source of truth — the attention-dispatch site
/// in `gemma4_layer_exec.rs` and the decode-graph eligibility check
/// in `run_generate` BOTH consult this so the eligibility guard
/// can never disagree with the kernel that actually runs.
///
/// Default `1024` matches the dispatch-side default. Values that are
/// not powers of two or smaller than 64 fall back to the default,
/// mirroring the dispatch validator.
pub fn effective_partition_size() -> u32 {
    const DEFAULT: u32 = 1024;
    let raw: u32 = std::env::var("RVLLM_NVFP4_PARTITION_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT);
    if raw >= 64 && raw.is_power_of_two() {
        raw
    } else {
        DEFAULT
    }
}

/// Decide whether decode-graph capture is safe for a single generation.
///
/// The captured graph freezes the split-KV partition decision (single-
/// CTA vs multi-partition attention dispatch). If `capture_ctx`
/// (= `prompt_len + capture_decode_step + 1`) and
/// `prompt_len + max_new` (the end of generation) fall on different
/// sides of the `partition_size` threshold, replay would run the
/// frozen kernel for an entire generation that grows past the
/// capture-time decision. The kernel is still correct in that case
/// but pays the wrong-path overhead, which negates the launch-overhead
/// win that motivates capture in the first place.
///
/// `capture_decode_step` mirrors the `RVLLM_DECODE_GRAPH_CAPTURE_AT`
/// runtime knob — operators who tune the capture point also shift the
/// boundary at which this check evaluates.
///
/// Returns `true` when capture is safe. Codex36-1 tightened the
/// rule: capture is safe only when the partition count is identical
/// at capture and at end-of-generation. The earlier "both > 1"
/// branch tolerated count drift across partition-size boundaries,
/// but Codex35-2 made `gridDim.z = current_num_partitions` at
/// launch time. Captured graphs freeze that gridDim, so once the
/// live context crosses the next partition boundary the reducer
/// (which still reads the live `context_lens`) reads from
/// scratch slots no CTA wrote — silent stale-attention.
/// Drift case → eager path, which re-launches with the correct
/// per-step gridDim.z.
pub fn decode_graph_eligible_for_generation(
    prompt_len: u32,
    max_new: u32,
    partition_size: u32,
    capture_decode_step: u32,
    split_kv_active: bool,
    allow_recapture: bool,
) -> bool {
    if partition_size == 0 {
        return false;
    }
    // Codex37-3: the parts-count drift check only matters when split-KV
    // is actually used. With split-KV disabled (env=0, FP8/F16 KV, no
    // split kernels loaded, or unsuitable layer mix), gridDim.z stays
    // at 1 and the live `context_lens` read by the reducer can't read
    // stale slots — capture is safe regardless of partition crossings.
    // Earlier this function refused capture across partition boundaries
    // unconditionally, costing decode-graph perf with no correctness
    // gain.
    // Codex51-2: the decode loop runs `0..max_new - 1`, so the
    // largest decode_step ever reached is `max_new - 2`. If
    // `capture_decode_step` lands at or beyond `max_new - 1`, the
    // capture site is never executed and `RVLLM_DECODE_GRAPH=1`
    // silently does nothing. Reject up front so the eager-fallback
    // log line surfaces the misconfiguration. (max_new = 1 has zero
    // decode-loop iterations; nothing to capture there either.)
    if max_new < 2 || capture_decode_step >= max_new.saturating_sub(1) {
        return false;
    }
    if !split_kv_active {
        return true;
    }
    // Round-18 finding #4: when split-KV is active but the parts count
    // would change mid-generation, the caller can opt into re-capture
    // (one captured graph per stable parts-range). The decode loop
    // then drops the captured graph at each partition-boundary
    // crossing and recaptures against the new gridDim.z. Replay is
    // still O(launches/step) within each range — the only cost is a
    // single eager warmup step + capture per crossing, which is
    // sparse for realistic partition_sizes (≥ 256) on long contexts.
    // Callers who want the strict pre-Codex36 "all-or-nothing"
    // behaviour pass `allow_recapture=false`. The runtime caller
    // resolves this from `RVLLM_DECODE_GRAPH_RECAPTURE` (default 1).
    if allow_recapture {
        return true;
    }
    let capture_ctx = prompt_len
        .saturating_add(capture_decode_step)
        .saturating_add(1);
    // Codex38-3: end_ctx = prompt_len + max_new - 1. The decode loop
    // emits the first new token directly after prefill and runs the
    // remaining `max_new - 1` forward passes; the last one sees
    // context length `prompt_len + max_new - 1`. Using the old
    // `prompt_len + max_new` upper-bound rejected requests that
    // happened to land on a partition boundary just one token past
    // their actual final ctx.
    let end_ctx = prompt_len.saturating_add(max_new.saturating_sub(1)).max(1);
    let parts_at_capture = capture_ctx.div_ceil(partition_size).max(1);
    let parts_at_end = end_ctx.div_ceil(partition_size).max(1);
    parts_at_capture == parts_at_end
}

#[cfg(test)]
mod spec_decode_session_tests {
    use super::SpecDecodeSession;

    #[test]
    fn open_seeds_committed_zero_with_prompt_tokens() {
        let prompt = vec![1u32, 2, 3, 4];
        let s = SpecDecodeSession::open(&prompt, 0xDEAD_BEEF);
        assert_eq!(s.committed_len, 0);
        assert_eq!(s.tokens, prompt);
        assert_eq!(s.last_base_hidden_ptr, 0xDEAD_BEEF);
        assert_eq!(s.next_base_argmax, u32::MAX);
        assert_eq!(s.iter_count, 0);
        assert_eq!(s.last_committed_token(), None);
    }

    #[test]
    fn warmup_seed_marks_full_prompt_committed() {
        let mut s = SpecDecodeSession::open(&[10u32, 20, 30], 0x1000);
        s.seed_from_warmup(3, 99);
        assert_eq!(s.committed_len, 3);
        assert_eq!(s.next_base_argmax, 99);
        assert_eq!(s.last_committed_token(), Some(30));
        assert_eq!(s.drafter_input_position(), 2); // = committed_len - 1
    }

    #[test]
    fn commit_drafts_advances_correctly_under_partial_accept() {
        let mut s = SpecDecodeSession::open(&[1u32, 2, 3], 0);
        s.seed_from_warmup(3, 100);
        let drafts = vec![500u32, 600, 700, 800];
        // accept_len = 2 of 4. Bonus token = 999 (the divergence).
        s.commit_drafts(&drafts, 2, 999);
        // tokens = original + accepted[..2]; committed_len bumps by 2.
        assert_eq!(s.tokens, vec![1, 2, 3, 500, 600]);
        assert_eq!(s.committed_len, 5);
        assert_eq!(s.next_base_argmax, 999);
        assert_eq!(s.iter_count, 1);
        assert_eq!(s.last_committed_token(), Some(600));
        assert_eq!(s.drafter_input_position(), 4);
    }

    #[test]
    fn commit_drafts_accept_zero_is_a_noop_on_tokens() {
        let mut s = SpecDecodeSession::open(&[1u32, 2], 0);
        s.seed_from_warmup(2, 42);
        s.commit_drafts(&[500u32, 600], 0, 7);
        // accept_len=0: no drafts committed; only next_base_argmax + iter bumped.
        assert_eq!(s.tokens, vec![1, 2]);
        assert_eq!(s.committed_len, 2);
        assert_eq!(s.next_base_argmax, 7);
        assert_eq!(s.iter_count, 1);
    }

    #[test]
    fn rewind_truncates_metadata_only() {
        let mut s = SpecDecodeSession::open(&[1u32, 2, 3], 0);
        s.seed_from_warmup(3, 100);
        s.commit_drafts(&[500, 600, 700], 3, 42);
        assert_eq!(s.committed_len, 6);
        assert_eq!(s.tokens.len(), 6);
        s.rewind_to(4);
        assert_eq!(s.committed_len, 4);
        assert_eq!(s.tokens, vec![1, 2, 3, 500]);
        // rewind past current is a no-op
        s.rewind_to(10);
        assert_eq!(s.committed_len, 4);
        assert_eq!(s.tokens.len(), 4);
    }
}

#[cfg(test)]
mod decode_graph_eligibility_tests {
    use super::{decode_graph_eligible_for_generation, effective_partition_size};

    // Strict mode = recapture disabled (round-18 finding #4 added the
    // `allow_recapture` parameter). The pre-existing assertions all
    // documented strict semantics, so we keep them under strict mode
    // and add lenient-mode coverage below.
    const STRICT: bool = false;
    const LENIENT: bool = true;

    #[test]
    fn short_prompt_short_generation_eligible() {
        // Capture at ctx=2, end at ctx=33, partition=256 → both <= 256.
        assert!(decode_graph_eligible_for_generation(0, 33, 256, 1, true, STRICT));
    }

    #[test]
    fn long_prompt_inside_one_partition_eligible() {
        // prompt_len=200, max_new=40, partition=256 → end_ctx=240 ≤ 256.
        assert!(decode_graph_eligible_for_generation(200, 40, 256, 1, true, STRICT));
    }

    #[test]
    fn generation_crosses_partition_boundary_blocked_in_strict_mode() {
        // prompt_len=200, max_new=200, end_ctx=400 → 400/256=2 parts,
        // capture_ctx=202 → 1 part. Capture would freeze single-CTA
        // path; replay the second half wrongly. Block capture.
        assert!(!decode_graph_eligible_for_generation(200, 200, 256, 1, true, STRICT));
        // Lenient mode allows this — the runtime re-captures at
        // partition boundaries.
        assert!(decode_graph_eligible_for_generation(200, 200, 256, 1, true, LENIENT));
    }

    #[test]
    fn long_prompt_already_past_boundary_eligible() {
        // prompt_len=2000, max_new=40 — capture_ctx and end_ctx both
        // produce 8 partitions, so the split-path decision is stable.
        assert!(decode_graph_eligible_for_generation(2000, 40, 256, 1, true, STRICT));
    }

    #[test]
    fn long_prompt_crossing_higher_boundary_blocked_in_strict_mode() {
        // Codex36-1 regression: parts_at_capture=8, parts_at_end=9
        // (generation crosses one more partition boundary). Strict
        // mode rejects to avoid replaying a frozen gridDim.z;
        // lenient mode accepts because the decode loop will recapture
        // at the boundary.
        assert!(!decode_graph_eligible_for_generation(2000, 300, 256, 1, true, STRICT));
        assert!(decode_graph_eligible_for_generation(2000, 300, 256, 1, true, LENIENT));
    }

    #[test]
    fn crossing_boundary_eligible_when_split_inactive() {
        // Codex37-3 regression: when split-KV is OFF, gridDim.z stays
        // at 1 regardless of parts-count drift, so eligibility is
        // independent of the strict/lenient flag.
        assert!(decode_graph_eligible_for_generation(2000, 300, 256, 1, false, STRICT));
        assert!(decode_graph_eligible_for_generation(2000, 300, 256, 1, false, LENIENT));
    }

    #[test]
    fn zero_partition_size_blocks() {
        // Defensive: a misconfigured partition_size of 0 must not
        // panic via div_ceil; the eligibility just becomes false in
        // both modes (the strict-mode parts-stable check would also
        // div by zero, so we reject before the recapture branch).
        assert!(!decode_graph_eligible_for_generation(100, 100, 0, 1, true, STRICT));
        assert!(!decode_graph_eligible_for_generation(100, 100, 0, 1, true, LENIENT));
    }

    /// Regression: the eligibility check used to default to 256 while
    /// the attention dispatch defaulted to 1024. With 1024 the split-
    /// path threshold lives further out, so a default-config request
    /// of `prompt_len=900, max_new=200` (end_ctx=1100) crosses the
    /// boundary at 1024 and is rejected by STRICT mode.
    #[test]
    fn shared_default_blocks_900_plus_200_request_in_strict_mode() {
        let p = effective_partition_size();
        assert_eq!(p, 1024, "shared default drifted away from dispatch site");
        assert!(
            !decode_graph_eligible_for_generation(900, 200, p, 1, true, STRICT),
            "default-config 900+200 should cross the 1024 partition boundary",
        );
        // Lenient mode accepts because the runtime re-captures.
        assert!(decode_graph_eligible_for_generation(900, 200, p, 1, true, LENIENT));
    }

    /// `RVLLM_DECODE_GRAPH_CAPTURE_AT` shifts which step the strict
    /// parts-stability check evaluates. Lenient mode is unaffected
    /// — re-capture handles whatever crossings occur.
    #[test]
    fn capture_at_shifts_eligibility_boundary() {
        // capture_at=1, capture_ctx=202 (parts=1), end_ctx=500 (parts=2)
        assert!(!decode_graph_eligible_for_generation(200, 300, 256, 1, true, STRICT));
        // capture_at=200, capture_ctx=401 (parts=2), end_ctx=500 (parts=2)
        assert!(decode_graph_eligible_for_generation(200, 300, 256, 200, true, STRICT));
    }

    /// Codex51-2: capture_decode_step >= max_new - 1 is unreachable
    /// because the decode loop runs `0..max_new - 1`. Holds in both
    /// modes (recapture doesn't help if capture is never reached).
    #[test]
    fn unreachable_capture_at_blocks() {
        assert!(!decode_graph_eligible_for_generation(200, 300, 256, 299, true, LENIENT));
        assert!(!decode_graph_eligible_for_generation(200, 300, 256, 300, true, LENIENT));
        assert!(!decode_graph_eligible_for_generation(200, 1, 256, 0, true, LENIENT));
        // max_new = 2 → loop runs step 0 only; capture_at = 0 should
        // still be eligible if the parts check holds.
        assert!(decode_graph_eligible_for_generation(0, 2, 256, 0, true, LENIENT));
    }

    /// Lock down the env-var resolver: invalid (non-power-of-two,
    /// too small) values fall back to the dispatch-site default
    /// rather than letting the eligibility check use a bogus value
    /// the dispatch site rejects.
    #[test]
    fn effective_partition_size_rejects_invalid_env() {
        // Save + restore prior value so we don't pollute other tests.
        let prior = std::env::var_os("RVLLM_NVFP4_PARTITION_SIZE");
        std::env::set_var("RVLLM_NVFP4_PARTITION_SIZE", "777"); // not pow2
        assert_eq!(effective_partition_size(), 1024);
        std::env::set_var("RVLLM_NVFP4_PARTITION_SIZE", "32"); // < 64
        assert_eq!(effective_partition_size(), 1024);
        std::env::set_var("RVLLM_NVFP4_PARTITION_SIZE", "256"); // valid
        assert_eq!(effective_partition_size(), 256);
        match prior {
            Some(v) => std::env::set_var("RVLLM_NVFP4_PARTITION_SIZE", v),
            None => std::env::remove_var("RVLLM_NVFP4_PARTITION_SIZE"),
        }
    }
}

/// Cycle 46 step 5c: convert the loader's optional per-layer AWQ weight
/// table into the runtime's flat-pointer struct. Returns the all-zero
/// default when no AWQ tensors were uploaded for this layer (every
/// non-AWQ checkpoint), so the existing FP8 dispatch cascade runs.
fn awq_layer_ptrs(
    awq: Option<&rvllm_loader::AwqLayerWeights>,
) -> crate::gemma4_layer_exec::Gemma4AwqLayerPtrs {
    let Some(a) = awq else {
        return Default::default();
    };
    // group_size is the same across all 7 linears (AwqConfig contract:
    // one Linear config group). Pick from q_proj — any of them is
    // identical by construction.
    let group_size = a.q_proj.group_size;
    debug_assert!(
        a.k_proj.group_size    == group_size
            && a.v_proj.group_size    == group_size
            && a.o_proj.group_size    == group_size
            && a.gate_proj.group_size == group_size
            && a.up_proj.group_size   == group_size
            && a.down_proj.group_size == group_size,
        "AwqLayerWeights group_size must be uniform across linears"
    );
    crate::gemma4_layer_exec::Gemma4AwqLayerPtrs {
        q_packed:    a.q_proj.packed_offset_bytes,
        q_scale:     a.q_proj.scale_offset_bytes,
        q_zero:      a.q_proj.zero_point_offset_bytes,
        k_packed:    a.k_proj.packed_offset_bytes,
        k_scale:     a.k_proj.scale_offset_bytes,
        k_zero:      a.k_proj.zero_point_offset_bytes,
        v_packed:    a.v_proj.packed_offset_bytes,
        v_scale:     a.v_proj.scale_offset_bytes,
        v_zero:      a.v_proj.zero_point_offset_bytes,
        o_packed:    a.o_proj.packed_offset_bytes,
        o_scale:     a.o_proj.scale_offset_bytes,
        o_zero:      a.o_proj.zero_point_offset_bytes,
        gate_packed: a.gate_proj.packed_offset_bytes,
        gate_scale:  a.gate_proj.scale_offset_bytes,
        gate_zero:   a.gate_proj.zero_point_offset_bytes,
        up_packed:   a.up_proj.packed_offset_bytes,
        up_scale:    a.up_proj.scale_offset_bytes,
        up_zero:     a.up_proj.zero_point_offset_bytes,
        down_packed: a.down_proj.packed_offset_bytes,
        down_scale:  a.down_proj.scale_offset_bytes,
        down_zero:   a.down_proj.zero_point_offset_bytes,
        group_size,
    }
}

pub use crate::bring_up::HbmArenaCheckpoint;

/// Default per-tensor Q / KV scale for the FP8 E4M3 attention cache.
/// The older 418/448 ≈ 0.933 defaults assume `amax ≈ 418` for the
/// post-QK-norm / post-V-norm activations — an order of magnitude too
/// large for Gemma 4. Empirical calibration sweep (chunk_len=128,
/// English text) found `q=0.1, kv=0.08` minimizes PPL by 4× over the
/// old defaults (PPL 10.2 → 2.3). Both overridable per-run via
/// `RVLLM_Q_SCALE` / `RVLLM_KV_SCALE` env vars for further tuning or
/// per-model calibration.
const DEFAULT_Q_SCALE: f32 = 0.1;
const DEFAULT_KV_SCALE: f32 = 0.08;

/// Cycle 56 step 2: parse a `f32` env var with explicit logging on
/// malformed values. The earlier `.parse().ok().unwrap_or(default)`
/// idiom silently swallowed parse failures — operator setting
/// `RVLLM_Q_SCALE=invalid` got the default with no signal. With this
/// helper, malformed values emit a stderr warning and fall back; the
/// operator at least sees the misconfig in journalctl.
fn parse_f32_env_or_default(name: &'static str, default: f32) -> f32 {
    match std::env::var(name) {
        Err(_) => default, // env unset is the silent-OK case
        Ok(v) => match v.parse::<f32>() {
            Ok(parsed) => parsed,
            Err(e) => {
                eprintln!(
                    "[rvllm] WARN: {name}={v:?} failed to parse as f32 ({e}); \
                     falling back to default {default}",
                );
                default
            }
        },
    }
}

pub struct Gemma4EnginePaths {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub cutlass_so: PathBuf,
    pub fa3_so: PathBuf,
    pub policy_json: PathBuf,
}

pub struct Gemma4FusedModules {
    pub rmsnorm_mod: LoadedModule,
    pub rmsnorm_inplace_mod: LoadedModule,
    pub rope_mod: LoadedModule,
    /// Cycle 55 step 7 (Phase B): bf16-input sibling of rope_mod
    /// (`fused_rope_partial_fp8kv_bf16in`).
    pub rope_partial_fp8kv_bf16in_mod: LoadedModule,
    pub gelu_mod: LoadedModule,
    pub argmax_mod: LoadedModule,
    pub qk_norm_mod: LoadedModule,
    /// Cycle 55 step 5 (Phase B): bf16-typed sibling of qk_norm_mod.
    pub qk_norm_bf16_mod: LoadedModule,
    pub softcap_mod: LoadedModule,
    /// Codex41-3: device-side repetition penalty PTX module.
    pub repetition_penalty_mod: LoadedModule,
    pub residual_scale_mod: LoadedModule,
    pub vnorm_mod: LoadedModule,
    pub vector_add_mod: LoadedModule,
    pub bf16_to_f16_sat_mod: LoadedModule,
    pub rmsnorm_inplace_bf16_mod: LoadedModule,
    pub vector_add_bf16_to_f16_mod: LoadedModule,
    pub f32_to_bf16_mod: LoadedModule,
    pub f32_to_f16_sat_mod: LoadedModule,
    pub scale_cols_f32_mod: LoadedModule,
    pub scale_rows_f32_ratio_mod: LoadedModule,
    pub fused_gelu_mul_f16_mod: LoadedModule,
    /// Cycle 55 step 6 (Phase B): bf16 sibling of fused_gelu_mul_f16.
    pub fused_gelu_mul_bf16_mod: LoadedModule,
    /// E4B Per-Layer Embeddings (PLE) GELU(gate)·per_layer_input
    /// reading from two separate pointers (not gate||up concat).
    pub gelu_tanh_mul_dual_f16_mod: LoadedModule,
    pub fused_rope_partial_f16kv_mod: LoadedModule,
    pub fused_norm_add_residual_mod: LoadedModule,
    // Cycle 53+ Stage 1: BF16 residual chain modules.
    pub f16_to_bf16_mod: LoadedModule,
    pub fused_norm_add_residual_bf16_mod: LoadedModule,
    pub fused_rmsnorm_fp8_quant_bf16in_mod: LoadedModule,
    pub fn_rmsnorm: KernelFn,
    pub fn_rmsnorm_fp8_quant: KernelFn,
    pub fn_quantize: KernelFn,
    pub fn_rope_partial_fp8kv: KernelFn,
    /// Cycle 55 step 7: bf16-input sibling of fn_rope_partial_fp8kv.
    /// Same launch ABI; Q/K/V activation inputs flip f16 → bf16 while
    /// FP8 KV cache write side stays unchanged (FP8 by design).
    pub fn_rope_partial_fp8kv_bf16in: KernelFn,
    pub fn_gelu_mul: KernelFn,
    pub fn_argmax: KernelFn,
    pub fn_qk_rmsnorm: KernelFn,
    /// Cycle 55 step 5: bf16 sibling of fn_qk_rmsnorm. Same launch
    /// ABI; only the dtype interpretation of inputs/outputs/gamma
    /// flips f16 → bf16.
    pub fn_qk_rmsnorm_bf16: KernelFn,
    pub fn_softcap: KernelFn,
    /// Codex40-2: f32 variant for the generate-path logit softcap.
    /// generate samples directly from f32 logits (no f16 conversion);
    /// PPL/bench convert through f16 and use `fn_softcap`.
    pub fn_softcap_f32: KernelFn,
    /// Codex41-3: GPU-side repetition penalty (replaces the host
    /// DtoH-edit-HtoD path that cost ~7ms/decode-step).
    pub fn_apply_repetition_penalty_f32: KernelFn,
    pub fn_residual_scale: KernelFn,
    pub fn_vnorm: KernelFn,
    pub fn_vector_add: KernelFn,
    pub fn_bf16_to_f16_sat: KernelFn,
    pub fn_rmsnorm_inplace_bf16: KernelFn,
    pub fn_vector_add_bf16_to_f16: KernelFn,
    // Cycle 53+ Stage 1: BF16 residual chain function handles.
    pub fn_f16_to_bf16: KernelFn,
    pub fn_fused_norm_add_residual_bf16: KernelFn,
    pub fn_fused_norm_add_residual_bf16_f16in: KernelFn,
    /// Cycle 55 step 19 (Phase B): bf16-input + bf16-residual variant
    /// for the FULL_CHAIN dispatch where Fp8GemvBf16In produces bf16
    /// GEMV output directly. Eliminates the f16 narrow at the
    /// epilogue boundary that the `_bf16_f16in` variant required.
    pub fn_fused_norm_add_residual_bf16_bf16in: KernelFn,
    pub fn_fused_rmsnorm_fp8_quant_bf16in: KernelFn,
    pub fn_f32_to_bf16: KernelFn,
    pub fn_f32_to_f16_sat: KernelFn,
    pub fn_scale_cols_f32: KernelFn,
    pub fn_scale_rows_f32_ratio: KernelFn,
    pub fn_fused_gelu_mul_f16: KernelFn,
    /// Cycle 55 step 6: bf16 sibling of fn_fused_gelu_mul_f16.
    pub fn_fused_gelu_mul_bf16: KernelFn,
    /// E4B PLE: GELU(tanh)(gate) * per_layer_input on TWO separate
    /// pointers (not gate||up concat). See kernels/gelu_tanh_mul_dual_f16.cu.
    pub fn_gelu_tanh_mul_dual_f16: KernelFn,
    pub fn_fused_rope_partial_f16kv: KernelFn,
    pub fn_fused_norm_add_residual: KernelFn,
    pub fn_fused_norm_add_residual_f16: KernelFn,
    /// Variant that reads f16 input and skips channelscale; used by the
    /// Sm121 decode fast path after `fp8_gemv_wpr_native_f16in` has
    /// already applied the per-channel scale in the GEMV epilogue.
    pub fn_fused_norm_add_residual_f16in: KernelFn,
    pub fused_norm_add_residual_f16_mod: LoadedModule,
    pub fn_fused_qkv_rmsnorm: KernelFn,
    /// Cycle 55 step 11 (Phase B): bf16 sibling of fn_fused_qkv_rmsnorm.
    /// Same launch ABI; Q/K/V/gamma all flip f16 → bf16.
    pub fn_fused_qkv_rmsnorm_bf16: KernelFn,
    pub fused_qkv_rmsnorm_mod: LoadedModule,
    /// Cycle 55 step 11 (Phase B): bf16-typed sibling of fused_qkv_rmsnorm_mod.
    pub fused_qkv_rmsnorm_bf16_mod: LoadedModule,
    pub fn_scale_cols_f16: KernelFn,
    pub scale_cols_f16_mod: LoadedModule,

    // ── Vision (Phase 3b) — kernels for forward_gemma_vision. ─────────
    pub layernorm_inplace_f16_mod: LoadedModule,
    pub fn_layernorm_inplace_f16: KernelFn,
    pub softmax_row_f16_mod: LoadedModule,
    pub fn_softmax_row_f16: KernelFn,
    pub gelu_tanh_f16_mod: LoadedModule,
    pub fn_gelu_tanh_f16: KernelFn,
    pub gelu_tanh_mul_f16_mod: LoadedModule,
    pub fn_gelu_tanh_mul_f16: KernelFn,
    pub vit_avgpool_f16_mod: LoadedModule,
    pub fn_vit_avgpool_f16: KernelFn,
    pub vit_pos_emb_lookup_2d_f16_mod: LoadedModule,
    pub fn_vit_pos_emb_lookup_2d_f16: KernelFn,
    pub transpose_2d_f16_mod: LoadedModule,
    pub fn_transpose_2d_f16: KernelFn,
    // B6b: audio subsample kernels (Gemma 4 E4B). All three are
    // f16-only and follow the standard PTX-load convention. Optional
    // at bring-up — failing to load these only disables the audio
    // path, not text/vision.
    pub im2col_3x3_s2p1_f16_mod: LoadedModule,
    pub fn_im2col_3x3_s2p1_f16: KernelFn,
    pub layernorm_relu_chw_f16_mod: LoadedModule,
    pub fn_layernorm_relu_chw_f16: KernelFn,
    pub transpose_chw_to_hwc_f16_mod: LoadedModule,
    pub fn_transpose_chw_to_hwc_f16: KernelFn,
    // B6c: audio encoder block building blocks. glu_split (sigmoid
    // gate) for LightConv1d, plain silu inplace for FFN + LConv post.
    pub glu_split_sigmoid_f16_mod: LoadedModule,
    pub fn_glu_split_sigmoid_f16: KernelFn,
    pub silu_inplace_f16_mod: LoadedModule,
    pub fn_silu_inplace_f16: KernelFn,
    pub causal_conv1d_f16_mod: LoadedModule,
    pub fn_causal_conv1d_f16: KernelFn,
    pub tanh_softcap_inplace_f32_mod: LoadedModule,
    pub fn_tanh_softcap_inplace_f32: KernelFn,
    pub rel_shift_audio_f32_mod: LoadedModule,
    pub fn_rel_shift_audio_f32: KernelFn,
    pub scale_per_dim_f32_mod: LoadedModule,
    pub fn_scale_per_dim_f32: KernelFn,
    pub audio_chunk_extract_context_f32_mod: LoadedModule,
    pub fn_audio_chunk_extract_context_f32: KernelFn,
    pub add_inplace_f32_mod: LoadedModule,
    pub fn_add_inplace_f32: KernelFn,
    pub transpose_v_chunked_f16_mod: LoadedModule,
    pub fn_transpose_v_chunked_f16: KernelFn,
    pub scale_scalar_inplace_f32_mod: LoadedModule,
    pub fn_scale_scalar_inplace_f32: KernelFn,
    pub apply_audio_attn_mask_f32_mod: LoadedModule,
    pub fn_apply_audio_attn_mask_f32: KernelFn,
    pub transpose_hwc_to_chw_f16_mod: LoadedModule,
    pub fn_transpose_hwc_to_chw_f16: KernelFn,
    pub clamp_inplace_f16_mod: LoadedModule,
    pub fn_clamp_inplace_f16: KernelFn,
    pub clamp_inplace_f32_mod: LoadedModule,
    pub fn_clamp_inplace_f32: KernelFn,
    pub rmsnorm_no_scale_inplace_f16_mod: LoadedModule,
    pub fn_rmsnorm_no_scale_inplace_f16: KernelFn,
    pub rmsnorm_no_scale_inplace_f32_mod: LoadedModule,
    pub fn_rmsnorm_no_scale_inplace_f32: KernelFn,
    pub add_bias_f16_to_f32_mod: LoadedModule,
    pub fn_add_bias_f16_to_f32: KernelFn,
    pub scale_inplace_f16_mod: LoadedModule,
    pub fn_scale_inplace_f16: KernelFn,
    pub add_bias_f16_mod: LoadedModule,
    pub fn_add_bias_f16: KernelFn,
    pub cast_fp_mod: LoadedModule,
    pub fn_cast_f32_to_f16: KernelFn,
    pub vit_rotary_2d_f16_mod: LoadedModule,
    pub fn_vit_rotary_2d_f16: KernelFn,
    pub vit_rotary_gemma4_2d_f16_mod: LoadedModule,
    pub fn_vit_rotary_gemma4_2d_f16: KernelFn,
    pub softmax_row_f32_to_f16_mod: LoadedModule,
    pub fn_softmax_row_f32_to_f16: KernelFn,
    pub vit_standardize_f16_mod: LoadedModule,
    pub fn_vit_standardize_f16: KernelFn,
    /// Vision Phase 5 audit follow-up (option 1 in
    /// v3/GEMMA_VISION_AUDIT.md): keep the pooler→standardize span
    /// in f32 to recover the 31/256 rows that overflow f16 after
    /// `*= sqrt(hidden=1152) ≈ 33.94`. Narrows back to f16 only
    /// after std_scale has divided magnitudes back into f16-safe
    /// range.
    pub vit_avgpool_f16_to_f32_mod: LoadedModule,
    pub fn_vit_avgpool_f16_to_f32: KernelFn,
    pub scale_inplace_f32_mod: LoadedModule,
    pub fn_scale_inplace_f32: KernelFn,
    pub vit_standardize_f32_to_f16_mod: LoadedModule,
    pub fn_vit_standardize_f32_to_f16: KernelFn,
    pub extract_head_f16_mod: LoadedModule,
    pub fn_extract_head_f16: KernelFn,
    pub fn_scatter_head_f16: KernelFn,
    /// Vision attention batched-strided pipeline: per-block fuses 16
    /// heads' QK^T and (scores @ V) into 2 cuBLASLt launches each
    /// instead of 32 (Codex review #B round 3 / #4 round 4).
    pub transpose_heads_v_f16_mod: LoadedModule,
    pub fn_transpose_heads_v_f16: KernelFn,
    pub scatter_heads_f16_mod: LoadedModule,
    pub fn_scatter_heads_f16: KernelFn,

    // `fp8_gemv.ptx` — GB10 warp-per-row FP8 GEMV kernels. Loaded at
    // bringup so the Sm121 decode fast path (`launch_fp8_gemv_f16in`
    // in `gemma4_layer_exec.rs`) can call it without a per-step
    // module load. Only the f16-input variant is resolved — the
    // other enum variants in `Fp8GemvVariant` document what ships in
    // the PTX but nothing in the runtime path calls them.
    pub fp8_gemv_mod: LoadedModule,
    /// `None` when the live device is not Blackwell (sm_100+) — the
    /// native-CVT entry is gated on `__CUDA_ARCH__ >= 1000` in
    /// `kernels/fp8_gemv.cu`, so the symbol is absent from
    /// pre-Blackwell PTX. `Fp8GemvVariant::available_for(target)` is
    /// the source of truth for this gate. Used by the Sm121 decode
    /// path to run projection GEMMs (QKV / O / gate_up / down)
    /// directly off f16 activations, skipping the FP8 activation-
    /// quant step that cuBLASLt requires.
    pub fn_fp8_gemv_wpr_native_f16in: Option<KernelFn>,
    /// Cycle 55 step 3 (Phase B): bf16-input sibling of the f16in fast
    /// path above. Same kernel ABI (`Fp8GemvF16InLaunch` reuses), only
    /// the dtype interpretation of input/output buffers differs. Used
    /// when `dims.bf16_residual = true` (the default since cycle 55
    /// step 1) so the M=1 decode QKV + gate_up fast paths don't narrow
    /// bf16→f16-sat at projection entry.
    pub fn_fp8_gemv_wpr_native_bf16in: Option<KernelFn>,
    /// Companion to the V-rotation arm of the NVFP4 RoPE kernel: when
    /// V is stored rotated (V_cache = V·R), attn_out = P·V·R, and we
    /// need to right-multiply attn_out by R^T per (token, head) before
    /// the O-projection. `None` on branches without the PTX or when
    /// loading fails — the dispatch site falls back to "no V rotation"
    /// in that case.
    pub hadamard_unrotate_f16_mod: Option<LoadedModule>,
    pub fn_hadamard_unrotate_f16: Option<KernelFn>,
    /// AWQ INT4 W4A16 GEMV kernel (cycle 45 step 4.5c). PTX may be
    /// absent on older kernel trees / non-Blackwell branches; fall
    /// through to `None` and the dispatch site treats AWQ as
    /// unavailable for any layer (load_gemma4_model rejects an
    /// AwqConfig-bearing checkpoint when this is `None`).
    pub awq_int4_gemv_f16_mod: Option<LoadedModule>,
    pub fn_awq_int4_gemv_f16: Option<KernelFn>,
    /// Cycle 51 step 10d.4: AWQ INT4 W4A16 GEMM kernel (M>1 prefill).
    /// PTX may be absent on older kernel trees; the AWQ prefill
    /// dispatch falls through to the per-token GEMV loop when this is
    /// `None`.
    pub awq_int4_gemm_sm120_wmma_mod: Option<LoadedModule>,
    pub fn_awq_int4_gemm_sm120_wmma: Option<KernelFn>,
}

/// Session-level prefix cache state. Populated on first `run_generate`
/// call; each subsequent call inspects `last_tokens` for a common
/// prefix with the incoming prompt and, on hit, skips prefill for
/// the matched prefix (the KV entries from the previous request
/// remain valid in the persistent KV region because `kv_cache_ptr`
/// points above the worker's scratch checkpoint).
///
/// MVP of vLLM's block-level prefix caching — no hashing, no
/// reference counting, just a single "last request's prompt" slot.
/// Covers the common zeroclaw pattern (identical 15k-token persona
/// on every request) at a cost of ~100 LOC of plumbing. Full
/// multi-sequence prefix caching is future work.
/// Spec-decode env knobs consolidated into one struct, read ONCE
/// per request at the top of `run_generate_speculative`. Codex
/// review priority 6: hot-path env reads (especially the per-row
/// `RVLLM_GEMMA4_SPEC_ACCEPT_BIAS` read inside the K-prefix verify
/// loop) were both pure overhead and a hazard for runtime knob
/// reordering. Now derived once, copied by value into the verify
/// loop's closure / branches.
///
/// Future cleanup: when the legacy `run_generate_speculative` path
/// is collapsed into a single validated `SpecDecodeMode` (Phase
/// D-2 / D-3), this struct becomes a passed-in parameter from the
/// startup-validated worker config, eliminating env reads from
/// per-request code entirely.
#[derive(Copy, Clone, Debug)]
pub struct SpecDecodeRequestConfig {
    /// `RVLLM_GEMMA4_SPEC_BATCHED=1` — activates the batched-verify
    /// hot path inside `run_generate_speculative`.
    pub batched_verify_mode: bool,
    /// `RVLLM_GEMMA4_SPEC_TYPICAL=1` — enables typical-acceptance
    /// math on greedy-mismatch (vs strict greedy verify).
    pub typical_mode: bool,
    /// `RVLLM_GEMMA4_SPEC_ACCEPT_BIAS=<f64>` — bias on the log-
    /// acceptance ratio. Positive = loosen (more accept). 0 = strict
    /// Leviathan/Kalman. Quality tradeoff knob.
    pub accept_bias: f64,
    /// `RVLLM_GEMMA4_SPEC_LOSSY_THRESHOLD=<f32>` — when > 0, enables
    /// lossy greedy ratio test (separate path from typical).
    pub lossy_threshold: f32,
    /// `RVLLM_GEMMA4_SPEC_EMIT_ACCEPTED=1` — legacy: emit only the
    /// accepted prefix + 1 bonus token. Set by the batched wrapper
    /// before calling into the spec function.
    pub emit_accepted: bool,
}

impl SpecDecodeRequestConfig {
    pub fn from_env() -> Self {
        Self {
            batched_verify_mode: std::env::var("RVLLM_GEMMA4_SPEC_BATCHED")
                .as_deref() == Ok("1"),
            typical_mode: std::env::var("RVLLM_GEMMA4_SPEC_TYPICAL")
                .as_deref() == Ok("1"),
            accept_bias: std::env::var("RVLLM_GEMMA4_SPEC_ACCEPT_BIAS")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0),
            lossy_threshold: std::env::var("RVLLM_GEMMA4_SPEC_LOSSY_THRESHOLD")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(0.0),
            emit_accepted: std::env::var("RVLLM_GEMMA4_SPEC_EMIT_ACCEPTED")
                .as_deref() == Ok("1"),
        }
    }
}

/// Per-request speculative-decoding state, owned by the spec
/// inner loop and distinct from the cross-request `PrefixCacheState`.
///
/// Codex review (priority 1) flagged that the previous batched-verify
/// path leaked spec-internal token-by-token state into
/// `prefix_cache.last_tokens` / `committed_prefix_len`. With
/// `RVLLM_PREFILL_CHUNK_SIZE=2048` the cross-request cache's
/// committed-len is floored to chunk boundaries, so any spec iter
/// on a short prompt re-prefilled the entire prompt every loop —
/// killing wall-clock even when verify itself was correct.
///
/// `SpecDecodeSession` carries the spec-only state explicitly:
///   * `committed_len`: how many TOKENS of base K/V are valid at the
///     start of this iter (not chunk-aligned — strictly the actually-
///     written boundary).
///   * `last_base_hidden_ptr`: device ptr to the POST-final-norm
///     hidden at position `committed_len - 1`, owned upstream
///     (`ensure_drafter` pre-allocates `base_last_hidden_ptr`).
///   * `next_base_argmax`: base's prediction at position
///     `committed_len` from the most recent verify pass (so the next
///     iter's i=0 verify check has its target without re-running
///     warmup).
///   * `tokens`: full committed token sequence (prompt + accepted
///     drafts), for slot_mapping invariants.
///
/// Rollback (`rewind_to(target_len)`) is the explicit operation
/// the new path uses after each verify pass — replacing the
/// metadata-truncate-of-prefix_cache hack from commit 42/46.
///
/// Lifecycle (Phase C wiring):
///   1. `open(prompt_ids)` at request start. `committed_len` = 0;
///      session is empty.
///   2. After the FIRST warmup base prefill of the full prompt:
///      `advance_committed(prompt_len, last_hidden, first_argmax)`.
///   3. Per spec iter:
///        - drafter produces K drafts conditioned on
///          `last_base_hidden_ptr`
///        - `verify_batched_from_state(drafts, committed_len)` writes
///          K K/V slots at positions committed_len..committed_len+K
///          and returns K argmaxes + K hiddens
///        - accept_len computed from drafts vs argmaxes
///        - `commit_drafts(accept_len, K-buffer)` advances
///          `committed_len += accept_len`, sets last_hidden to
///          K-buffer[accept_len-1] (or to warmup hidden for
///          accept_len = 0 — fallback case)
///        - implicit rollback: positions
///          [committed_len+accept_len..committed_len+K) are still
///          PHYSICALLY in the KV cache but no longer logically
///          committed; next iter overwrites them.
///   4. `close()` at request end: optionally publish accumulated
///      tokens into `prefix_cache.last_tokens` for cross-request
///      reuse (chunk-aligned committed_len rule from the existing
///      run_generate end-of-request path).
#[derive(Debug)]
pub struct SpecDecodeSession {
    /// Tokens whose base K/V slots are guaranteed populated in the
    /// persistent KV cache. Strictly per-token; not chunk-aligned.
    pub committed_len: u32,
    /// Per-iter snapshot of committed tokens (prompt prefix +
    /// accepted drafts so far). Used for slot_mapping and emit.
    pub tokens: Vec<u32>,
    /// Device ptr to POST-final-norm hidden at position
    /// `committed_len - 1`. Owned by `Gemma4Bringup::base_last_hidden_ptr`
    /// (pre-allocated above scratch by `ensure_drafter`); this field
    /// is a logical re-binding for clarity, not a separate allocation.
    pub last_base_hidden_ptr: u64,
    /// Base's argmax at position `committed_len` from the most
    /// recent verify pass (= the token base wants emitted next).
    /// Equal to `_base_first_tok[0]` from the legacy warmup, but
    /// carried forward from the previous iter's verify so the
    /// upcoming iter's i=0 verify check has its target without
    /// re-running base prefill. `u32::MAX` is the sentinel "no
    /// value yet" (= use warmup result on first iter).
    pub next_base_argmax: u32,
    /// Bumped on every commit. Lets debug / metrics distinguish
    /// iterations across one request.
    pub iter_count: u32,
}

impl SpecDecodeSession {
    /// Open an empty session. Caller must populate
    /// `last_base_hidden_ptr` from the per-bringup pre-allocated
    /// buffer before the first verify call.
    pub fn open(prompt_ids: &[u32], last_base_hidden_ptr: u64) -> Self {
        Self {
            committed_len: 0,
            tokens: prompt_ids.to_vec(),
            last_base_hidden_ptr,
            next_base_argmax: u32::MAX,
            iter_count: 0,
        }
    }

    /// Called once after the warmup base prefill of the full
    /// prompt. Marks committed_len = prompt_len (all P tokens have
    /// base K/V) and seeds next_base_argmax = base's first decode
    /// argmax (= what was previously `_base_first_tok[0]`).
    pub fn seed_from_warmup(
        &mut self,
        prompt_len: u32,
        first_base_argmax: u32,
    ) {
        self.committed_len = prompt_len;
        self.next_base_argmax = first_base_argmax;
    }

    /// Commit `accept_len` accepted drafts onto the session.
    /// committed_len advances by accept_len (NOT accept_len + 1 —
    /// the divergence/bonus token has no base K/V yet; the next
    /// iter's verify pass writes its slot from scratch).
    ///
    /// `new_next_argmax` is the base's argmax to use as the next
    /// iter's i=0 verify target. For accept_len == K: base's
    /// prediction at position committed_len+K from the just-finished
    /// verify pass. For accept_len < K: base's prediction at
    /// position committed_len+accept_len (the divergence) which
    /// IS emitted as the bonus token.
    pub fn commit_drafts(
        &mut self,
        drafts: &[u32],
        accept_len: usize,
        new_next_argmax: u32,
    ) {
        debug_assert!(accept_len <= drafts.len());
        self.tokens.extend_from_slice(&drafts[..accept_len]);
        self.committed_len = self
            .committed_len
            .saturating_add(accept_len as u32);
        self.next_base_argmax = new_next_argmax;
        self.iter_count = self.iter_count.saturating_add(1);
    }

    /// Explicit rollback. Truncates `tokens` to `target_len` and
    /// sets `committed_len = target_len`. PHYSICAL K/V slots beyond
    /// the new boundary are NOT erased — the contract is that the
    /// next verify pass overwrites them.
    pub fn rewind_to(&mut self, target_len: u32) {
        let t = target_len as usize;
        if t <= self.tokens.len() {
            self.tokens.truncate(t);
        }
        if target_len < self.committed_len {
            self.committed_len = target_len;
        }
    }

    /// Snapshot for the drafter's next K-step forward.
    /// Position is `committed_len - 1` (the most recent committed
    /// token's position; drafter predicts position `committed_len`).
    pub fn drafter_input_position(&self) -> u32 {
        self.committed_len.saturating_sub(1)
    }

    /// Last committed token id, for the drafter's input embed.
    /// Returns None if the session has no committed tokens (first
    /// drafter step before warmup completes).
    pub fn last_committed_token(&self) -> Option<u32> {
        if self.committed_len == 0 {
            return None;
        }
        self.tokens.get((self.committed_len as usize) - 1).copied()
    }
}

/// Output of `run_drafter_k_from_state`: K candidate drafter tokens
/// plus optional per-step drafter log-probabilities (populated only
/// in typical-acceptance mode).
#[derive(Debug, Default)]
pub struct DraftBatch {
    pub tokens: Vec<u32>,
    pub log_q: Vec<f32>,
}

/// RAII guard disarming the spec-decode hook atomics on drop.
///
/// `verify_batched_from_state` / `prefill_one_from_state` set up to
/// four atomics (`force_common_prefix_override`,
/// `skip_prefix_cache_publish`, `force_prefill_only`,
/// `base_last_k_snapshot_pending`) and optionally swap
/// `base_last_k_hidden_ptr` to redirect a capture. If the inner
/// `run_generate` returns early (e.g. `force_prefill_only` short-
/// circuit at the skip_decode site), the late publish-skip reset
/// never runs and the flag leaks into the next request.
///
/// This guard makes the disarm structurally unmissable: on drop, all
/// four atomics are stored back to their disarmed sentinels and (if
/// armed) the K-hidden destination ptr is restored to its prior
/// value. `Release` order so the next request sees the disarmed
/// state.
struct SpecHookGuard<'a> {
    bringup: &'a Gemma4Bringup,
    saved_k_dst: Option<u64>,
}

impl<'a> SpecHookGuard<'a> {
    fn new(b: &'a Gemma4Bringup) -> Self {
        Self { bringup: b, saved_k_dst: None }
    }
    /// Swap `base_last_k_hidden_ptr` to `new_ptr`; the old value is
    /// remembered and restored on drop. Multiple calls overwrite
    /// the saved value — the guard only restores the FIRST swap.
    fn swap_k_dst(&mut self, new_ptr: u64) {
        let old = self.bringup
            .base_last_k_hidden_ptr
            .swap(new_ptr, std::sync::atomic::Ordering::AcqRel);
        if self.saved_k_dst.is_none() {
            self.saved_k_dst = Some(old);
        }
    }
}

impl Drop for SpecHookGuard<'_> {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Release;
        self.bringup.force_common_prefix_override.store(u32::MAX, Release);
        self.bringup.skip_prefix_cache_publish.store(false, Release);
        self.bringup.force_prefill_only.store(false, Release);
        self.bringup.base_last_k_snapshot_pending.store(false, Release);
        // Codex Round 4 #5: also disarm the single-token snapshot
        // pending flag. prefill_one_from_state arms it before its
        // run_generate call; if run_generate errors before the
        // final_norm hook consumes the flag, it leaks and the next
        // request silently overwrites its base_last_hidden_ptr with a
        // wrong-position hidden.
        self.bringup.base_last_hidden_snapshot_pending.store(false, Release);
        if let Some(old) = self.saved_k_dst.take() {
            self.bringup.base_last_k_hidden_ptr.store(old, Release);
        }
    }
}

/// Maximum spec-K for the batched-verify K-hidden capture buffer.
/// Pre-allocated once at ensure_drafter time (above the scratch
/// checkpoint), so the buffer must fit the largest `spec_k` the
/// server will ever see in a request. Config validation rejects
/// `spec_k > MAX_SPEC_K` at startup so the capture hook can rely
/// on `k_requested <= MAX_SPEC_K`.
pub const MAX_SPEC_K: usize = 16;

pub struct PrefixCacheState {
    pub last_tokens: Vec<u32>,
    pub kv_cache_ptr: u64,
    pub kv_cache_bytes: u64,
    pub kv_scale_ptr: u64,
    pub kv_scale_bytes: u64,
    pub kv_dtype: crate::gemma4_layer_exec::KvDtype,
    pub kv_layer_offsets: Vec<u64>,
    pub kv_scale_layer_offsets: Vec<u64>,
    pub num_blocks_total: u32,
    pub block_size: u32,
    /// Length (in tokens) of the prefix from `last_tokens` that
    /// is SAFE to reuse across a subsequent request. Capped at
    /// the last full prefill-chunk boundary so subsequent prompts
    /// that match this prefix are guaranteed to find KV entries
    /// written under the SAME chunk shape as the new request would
    /// use for those positions.
    ///
    /// Without this cap, a short request (e.g. classifier, 3057
    /// tokens prefilled in chunks 2048+1009) leaves slots
    /// [2048..3057) populated under chunk_q=1009. A subsequent
    /// long request (e.g. 15k tokens) would have written those
    /// same slots inside its own first chunk_q=2048. Optimized
    /// NVFP4 kernels are batch-variant; reusing classifier-shape
    /// KV at slots [2048..2922) inside the 15k request produces
    /// catastrophic garbage ("la la la × 1024" repetition collapse,
    /// observed in production via zeroclaw classifier-then-persona
    /// chains).
    ///
    /// Set to `floor(prompt_len / chunk_size) * chunk_size` after
    /// each request completes (or `prompt_len` when chunk_size = 0,
    /// i.e. no chunking).
    pub committed_prefix_len: u32,
    /// Provenance tuple. Cache is INVALIDATED on mismatch — KV
    /// entries written under different policy configuration are
    /// not generally reusable. Cheap to check; protects against
    /// silent miscompare across env-var flips between requests.
    pub provenance: PrefixProvenance,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PrefixProvenance {
    pub chunk_size: u32,
    pub kv_dtype: crate::gemma4_layer_exec::KvDtype,
    pub hybrid_global_fp8: bool,
    pub scale_policy: u32,        // 0 = amax6, 1 = mse, etc.
    // K/V scale-policy split (each falls back to scale_policy when unset).
    // Tracked here so the prefix cache invalidates if either side flips.
    pub k_scale_policy: u32,
    pub v_scale_policy: u32,
    // NVFP4 quality knobs that change how K/Q-side state is interpreted.
    pub hadamard: bool,
    /// Whether V was Hadamard-rotated before NVFP4 packing. Cached V
    /// bytes differ between rotated/unrotated, so flipping this gate
    /// across requests on the same prefix-cache slot reuses
    /// incompatible V data — must invalidate the cache.
    pub hadamard_v: bool,
    pub per_token_q_scale: bool,
    pub batch_prefill: bool,
    pub unified_prefill: bool,
    /// Split-KV decode (paged_attention_v2 style). The split kernel
    /// has documented quality issues on long-context tool-call
    /// prompts even with Hadamard rotation; tracked here because
    /// flipping it changes which kernel reads the V cache and a
    /// silent flip mid-session could mask the failure mode.
    pub split_kv: bool,
    /// Inverse hybrid (cycle 25): all sliding layers FP8 KV when set.
    /// Changes per-layer dtype dispatch so different bytes land in K/V
    /// cache for sliding layers — must invalidate prefix cache on flip.
    pub hybrid_sliding_fp8: bool,
    /// Comma-separated layer-index list (cycle 24) forced to FP8 KV when
    /// default is NVFP4. Stored as the raw env string so flipping any
    /// element triggers invalidation; `String::new()` when unset.
    pub fp8_kv_layers: String,
    /// Cycle 31: stochastic-rounding gate for V. Changes packed V bytes
    /// (different fp4_encode path) so flipping mid-session would silently
    /// reuse incompatible V data on prefix-cache hits.
    pub stoch_round_v: bool,
    /// Codex17-2: NVFP4 split-decode partition size. Doesn't change KV
    /// bytes, but changes which split-decode kernel + reducer shape runs
    /// at decode time. Tracked here so the prefix cache invalidates on
    /// partition-size flips between requests; otherwise reproducibility
    /// across runs that share a cache slot is silently lost.
    pub partition_size: u32,
}

impl PrefixProvenance {
    /// Read current env into a provenance tuple.
    pub fn from_env() -> Self {
        let chunk_size: u32 = std::env::var("RVLLM_PREFILL_CHUNK_SIZE")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let kv_dtype = crate::gemma4_layer_exec::KvDtype::from_env(false);
        let hybrid_global_fp8 =
            parse_truthy_env("RVLLM_NVFP4_HYBRID_GLOBAL_FP8").unwrap_or(false);
        fn parse_policy(v: &str) -> Option<u32> {
            match v {
                "amax6" | "0" => Some(0),
                "mse" | "1" => Some(1),
                _ => None,
            }
        }
        let scale_policy = std::env::var("RVLLM_NVFP4_SCALE_POLICY")
            .ok().and_then(|s| parse_policy(&s)).unwrap_or(0);
        let k_scale_policy = std::env::var("RVLLM_NVFP4_K_SCALE_POLICY")
            .ok().and_then(|s| parse_policy(&s)).unwrap_or(scale_policy);
        let v_scale_policy = std::env::var("RVLLM_NVFP4_V_SCALE_POLICY")
            .ok().and_then(|s| parse_policy(&s)).unwrap_or(scale_policy);
        // Codex56-default: A_prod sweep config (cycle 55) is the
        // sweep-validated production target — flipping the unset
        // defaults to ON means a fresh deployment with no profile
        // env vars set still lands on production-quality config.
        // Operators opt OUT explicitly via `=0` for diagnostics.
        let hadamard = parse_truthy_env("RVLLM_NVFP4_HADAMARD").unwrap_or(true);
        // Provenance tracks the NVFP4 path's effective gate. Per Codex10-2
        // HADAMARD=1 auto-implies PER_TOKEN_Q_SCALE; the explicit default
        // here matches that contract so unset-env behaviour stays clean.
        let per_token_q_scale = parse_truthy_env("RVLLM_PER_TOKEN_Q_SCALE").unwrap_or(true);
        let hadamard_v = parse_truthy_env("RVLLM_NVFP4_HADAMARD_V").unwrap_or(true);
        let batch_prefill = parse_truthy_env("RVLLM_BATCH_PREFILL").unwrap_or(false)
            && kv_dtype != crate::gemma4_layer_exec::KvDtype::F16;
        // Match the NVFP4/FP8 prefill dispatch gate: unified prefill is
        // default-on and only explicit false-ish values disable it. Using
        // mere env presence here made `RVLLM_UNIFIED_PREFILL=0` and `=1`
        // look identical to the prefix cache even though they route through
        // different attention kernels.
        let unified_prefill = parse_truthy_env("RVLLM_UNIFIED_PREFILL").unwrap_or(true);
        // Default ON in dispatch (gemma4_layer_exec.rs line ~870), opt-out via "0"/etc.
        let split_kv = parse_truthy_env("RVLLM_NVFP4_SPLIT_KV").unwrap_or(true);
        let hybrid_sliding_fp8 =
            parse_truthy_env("RVLLM_NVFP4_HYBRID_SLIDING_FP8").unwrap_or(false);
        let fp8_kv_layers = std::env::var("RVLLM_FP8_KV_LAYERS")
            .unwrap_or_default();
        let stoch_round_v = parse_truthy_env("RVLLM_NVFP4_STOCH_ROUND_V").unwrap_or(false);
        let partition_size = effective_partition_size();
        Self { chunk_size, kv_dtype, hybrid_global_fp8, scale_policy,
               k_scale_policy, v_scale_policy, hadamard, hadamard_v,
               per_token_q_scale, batch_prefill, unified_prefill, split_kv,
               hybrid_sliding_fp8, fp8_kv_layers, stoch_round_v,
               partition_size }
    }
}

/// Spec-decode commit 3: per-engine metadata describing which base
/// layers' K/V the `Gemma4AssistantForCausalLM` drafter cross-attends
/// to at draft time. Populated at bring-up from
/// `Gemma4Arch::assistant_shared_kv_sources()`; `None` on archs without
/// a `num_kv_shared_layers` tail (31B). Live K/V device pointers are
/// derived per-request inside `run_generate` from
/// `kv_cache_ptr + kv_layer_offsets[sliding_source]` etc. — they're
/// not held here because the cache pointer is per-session state.
#[derive(Debug, Clone, Copy)]
pub struct Gemma4AssistantKvSources {
    /// Latest sliding-attention layer index within the non-shared
    /// prefix. On E4B-it: 22.
    pub sliding_source_layer: u32,
    /// Latest full-attention layer index within the non-shared
    /// prefix. On E4B-it: 23.
    pub full_source_layer: u32,
}

/// Per-request K-prefix accept stats from `run_generate_speculative`.
/// Populated at the end of each spec-decode request and consumed by
/// the worker via `take_last_spec_stats()` for `SpeculativeStep`
/// event emission.
#[derive(Debug, Clone, Copy)]
pub struct LastSpecStats {
    pub drafted: u32,
    pub accepted: u32,
    pub cumulative_decoded: u32,
}

pub struct Gemma4Bringup {
    pub fused: Gemma4FusedModules,
    pub sliding_attention: AttentionBackend,
    pub global_attention: AttentionBackend,
    pub cutlass: CutlassBackend,
    pub cublaslt: CublasLt,
    pub cublaslt_ws: HbmArenaCheckpoint,
    pub policy: Policy,
    pub arch: rvllm_loader::gemma4_arch::Gemma4Arch,
    pub model: rvllm_loader::gemma4_weights::Gemma4LoadedModel,
    pub kernels: Arc<KernelLoader>,
    pub stream: Stream,
    pub arena: HbmArena<'static>,
    pub ctx: Arc<CudaContextHandle>,
    /// Spec-decode commit 3: source-layer indices for the
    /// `Gemma4AssistantForCausalLM` drafter's cross-attention K/V.
    /// `None` when the model has no shared-KV tail (e.g. 31B).
    /// Read by commit 4's drafter forward; no path currently
    /// consumes it.
    pub assistant_kv_sources: Option<Gemma4AssistantKvSources>,
    /// Spec-decode commit 4: lazy-uploaded Gemma 4 E4B assistant
    /// drafter weights. Populated by `ensure_drafter` on the first
    /// request observing `ServerConfig::spec_decode == true`. When
    /// spec-decode is off the slot stays `None` and consumes zero
    /// HBM. Holding behind a `Mutex<Option<_>>` keeps construction
    /// at-most-once + thread-safe under the cuda-worker's single
    /// thread + shared by future request paths.
    pub drafter: crate::gemma4_drafter::DrafterSlot,
    /// Spec-decode commit 16: persistent f16[hidden_size] buffer for
    /// the base's pre-lm-head last-prompt-token hidden state.
    /// Allocated above the scratch checkpoint by `ensure_drafter`
    /// when spec_decode is on; left at zero otherwise.
    ///
    /// `run_generate` writes into this buffer after final RMSNorm +
    /// bf16→f16 widen, gated on `self.base_last_hidden_ptr != 0` —
    /// so non-spec engines pay zero cost. `run_generate_speculative`
    /// reads from it as the drafter's `base_hidden_last_step` for
    /// the `pre_projection` input concat.
    pub base_last_hidden_ptr: std::sync::atomic::AtomicU64,
    /// Spec-decode commit 18: per-request one-shot gate for the
    /// `run_generate` hook above. `run_generate_speculative` sets
    /// this to `true` BEFORE calling `run_generate`; the hook
    /// captures the first hidden it sees (= post-prefill, last
    /// prompt token) and clears the flag so subsequent decode-step
    /// fires don't overwrite. Without this, max_new > 1 left the
    /// buffer holding the post-last-decode hidden — wrong position
    /// for drafter step 1 → accept_rate stuck near 0.
    pub base_last_hidden_snapshot_pending: std::sync::atomic::AtomicBool,
    /// Commit 38 (batched verify): device buffer for the LAST K
    /// hidden states (post-final-norm) captured during the verify
    /// run_generate call. Sized K * hidden_size * 2 bytes. Allocated
    /// lazily in run_generate_speculative_batched when first needed.
    pub base_last_k_hidden_ptr: std::sync::atomic::AtomicU64,
    /// Commit 38: number of rows the K-capture hook should grab
    /// (= spec_k for current request).
    pub base_last_k_count: std::sync::atomic::AtomicU32,
    /// Commit 38: one-shot gate, twin of base_last_hidden_snapshot_pending.
    pub base_last_k_snapshot_pending: std::sync::atomic::AtomicBool,
    /// Commit 43: skip-warmup flag — when true at run_generate_speculative
    /// entry, the warmup base prefill is bypassed (base_last_hidden was
    /// populated by the previous outer iter's batched-verify branch
    /// from K-buffer[accept_len - 1]).
    pub skip_next_warmup: std::sync::atomic::AtomicBool,
    /// Commit 43: cached base argmax at the next iter's warmup
    /// position (= base_argmax_K[accept_len - 1] from prev iter).
    /// `u32::MAX` is the sentinel "no cached value".
    pub saved_warmup_b_p: std::sync::atomic::AtomicU32,
    /// Commit 53 (Phase D-2): per-request override flag for
    /// `SpecDecodeRequestConfig::emit_accepted`. Set by the outer
    /// wrappers (`run_generate_speculative_batched`,
    /// `run_generate_speculative_iterative`) to force the inner
    /// spec function to emit accepted-prefix-only, replacing the
    /// `env::set_var("RVLLM_GEMMA4_SPEC_EMIT_ACCEPTED", "1")`
    /// + save/restore pattern (codex review priority 2 "scribbles
    /// globals"). One-shot: consumed by `SpecDecodeRequestConfig::
    /// from_env_with_overrides` at fn entry.
    pub force_emit_accepted: std::sync::atomic::AtomicBool,
    /// Commit 55 (codex review priority 0.1): per-request override
    /// flag forcing `SpecDecodeRequestConfig::batched_verify_mode`
    /// = true regardless of env. Set by the worker when
    /// `spec_cfg.enabled` is on and no legacy-debug knob asks
    /// otherwise. Eliminates the silent fall-through to
    /// sequential-decode verify that was the actual default when
    /// users set only RVLLM_GEMMA4_SPEC_DECODE=1.
    pub force_batched_verify: std::sync::atomic::AtomicBool,
    /// Codex review priority 1 (commit 49 — Phase B-2): when
    /// non-`u32::MAX`, overrides the `prefix_cache` token-match +
    /// chunk_size-cap computation inside `run_generate`. The next
    /// `run_generate` call treats this many tokens as already
    /// committed and only prefills the suffix. One-shot semantics:
    /// the value is `swap(u32::MAX)`'d on consumption.
    ///
    /// Used by `verify_batched_from_state` to bypass the cross-
    /// request `committed_prefix_len = floor(P/chunk_size)*chunk_size`
    /// cap that was forcing whole-prompt re-prefill on every spec
    /// iteration with short prompts. Sentinel default keeps the
    /// non-spec hot path untouched (cheap atomic load + branch).
    pub force_common_prefix_override: std::sync::atomic::AtomicU32,
    /// Codex review priority 1 (commit 49 — Phase B-2): when true,
    /// the next `run_generate` call SKIPS the end-of-request
    /// `prefix_cache.last_tokens` / `committed_prefix_len` write.
    /// `verify_batched_from_state` sets this on so spec-internal
    /// state doesn't pollute the cross-request prefix cache (which
    /// is meant for the NEXT request's prompt, not the SAME
    /// request's spec iterations).
    pub skip_prefix_cache_publish: std::sync::atomic::AtomicBool,
    /// Commit 56 (codex review priority 0.2): per-request override
    /// forcing `run_generate` to return Ok(Vec::new()) immediately
    /// after the K-row hidden-state capture, skipping the wasted
    /// row-extract + final_norm + lm_head + softcap + argmax + DtoH
    /// that produces the "bonus" decode token. Both batched-verify
    /// callers (`verify_batched_from_state`, the inline batched-verify
    /// branch in `run_generate_speculative`) already discard that
    /// bonus — it's equivalent to `base_argmax_K[K-1]` which they
    /// compute themselves from the K-buffer. One-shot: consumed
    /// (`swap(false)`) inside `run_generate`.
    pub force_prefill_only: std::sync::atomic::AtomicBool,
    /// Spec-decode commit 25: last-request K-prefix accept-rate
    /// stats from `run_generate_speculative`. Populated at the end
    /// of the spec-decode path; cleared (taken) by the worker right
    /// after `run_generate_speculative` returns, then re-emitted as
    /// a `GenerateEvent::SpeculativeStep` so the HTTP handler can
    /// set the `X-RVLLM-Accept-Rate` response header. Idle when
    /// spec-decode is off.
    pub last_spec_stats: std::sync::Mutex<Option<LastSpecStats>>,
    /// Commit 26: per-decode-step base logits captured during
    /// `run_generate` when the spec-decode path needs them for the
    /// lossy-greedy acceptance check. Layout: row-major
    /// `[K+1, vocab]` f32. Allocated lazily when capture is first
    /// requested; reused across requests. Stays empty when
    /// `spec_logits_capture_active == false`.
    pub spec_decode_step_logits: std::sync::Mutex<Vec<f32>>,
    /// Commit 26: capture gate for the decode-loop logits export.
    /// `run_generate_speculative` flips this on before its
    /// `run_generate` call (when lossy greedy mode is enabled) and
    /// off after, so non-spec engines pay zero cost. Reads in the
    /// decode loop are a single relaxed atomic load.
    pub spec_logits_capture_active: std::sync::atomic::AtomicBool,
    /// Session-level prefix cache. Populated lazily on first
    /// `run_generate` call; kept across subsequent calls so the
    /// KV cache survives the worker's scratch-checkpoint restore.
    pub prefix_cache: std::sync::Mutex<Option<PrefixCacheState>>,
    // === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
    /// Ground-truth F16 shadow KV region for the instrumented layer set.
    /// Populated on first run_generate when RVLLM_NVFP4_SHADOW_F16=1.
    pub nvfp4_shadow: std::sync::Mutex<Option<NvFp4ShadowAlloc>>,
    /// One-shot latch for first-token dump.
    pub nvfp4_shadow_dumped: std::sync::atomic::AtomicBool,
    // === END NVFP4 SHADOW DIAGNOSTIC ===
    // === HADAMARD ROTATION ===
    /// Per-layer ±1 sign vectors for signed Walsh-Hadamard rotation
    /// of NVFP4 KV-cache K (and matching Q rotation pre-FP8). `None`
    /// when `RVLLM_NVFP4_HADAMARD` is unset OR kv_dtype != Nvfp4.
    /// Lazy-init on first `run_generate` (same pattern as
    /// `nvfp4_shadow`); deterministic seed so the same run
    /// reproduces identical R matrices across calls.
    pub nvfp4_hadamard: std::sync::Mutex<Option<NvFp4HadamardAlloc>>,
    // === END HADAMARD ROTATION ===
}

// === HADAMARD ROTATION ===
/// Per-layer ±1 sign vectors. Total size = `num_layers * head_dim`
/// bytes (i8 storage). Layer `l`'s slice begins at
/// `base + l * head_dim` (head_dim is uniform across Gemma 4 layers
/// at the rope-input level — both sliding and global use the same
/// per-head dimension; `arch.max_head_dim()` covers both).
pub struct NvFp4HadamardAlloc {
    pub base_ptr: u64,
    pub bytes: u64,
    pub head_dim: u32,
    pub num_layers: u32,
}

impl NvFp4HadamardAlloc {
    /// Device pointer for layer `layer_idx`. Returns 0 when out of
    /// range (caller should treat as "rotation disabled" — kernel's
    /// nullptr check then bypasses).
    pub fn layer_ptr(&self, layer_idx: u32) -> u64 {
        if layer_idx >= self.num_layers {
            return 0;
        }
        self.base_ptr + (layer_idx as u64) * (self.head_dim as u64)
    }
}

/// Tri-state env truthiness parser used by every NVFP4/FP8 quality
/// gate. Returns `Some(true)` for `"1"|"true"|"TRUE"|"yes"|"on"`,
/// `Some(false)` for `"0"|"false"|"FALSE"|"no"|"off"|""`, and `None`
/// for anything else so the caller falls back to a documented default.
/// Centralised so a profile typo (e.g. `RVLLM_PER_TOKEN_Q_SCALE=yess`)
/// behaves consistently across allocation, rope, decode, prefill, and
/// the prefix-cache provenance check.
pub(crate) fn parse_truthy_env(name: &str) -> Option<bool> {
    let v = std::env::var(name).ok()?;
    match v.as_str() {
        "1" | "true" | "TRUE" | "yes" | "on" => Some(true),
        "0" | "false" | "FALSE" | "no" | "off" | "" => Some(false),
        _ => None,
    }
}

/// Master env gate. Codex56-default: ON — sweep-validated A_prod
/// config (cycle 55) is the production target; unset-env now lands
/// on production-quality. Operator opts OUT via `=0` for diagnostics.
pub fn nvfp4_hadamard_enabled() -> bool {
    parse_truthy_env("RVLLM_NVFP4_HADAMARD").unwrap_or(true)
}

/// BF16 residual chain. Cycle 54 Stage 1 introduced this as an
/// opt-in gate (RVLLM_RESIDUAL_BF16=1). Cycle 55 step 1 flips the
/// **default to ON** because Gemma 4 was trained in bf16 and every
/// modern foundation model trains in bf16; f16 storage is a
/// distribution shift relative to training. The env still overrides
/// to false for diagnostics / regression bisects.
///
/// This is Phase A of the cycle-55 "fully native bf16" effort. Phases
/// B-G will progressively eliminate the f16↔bf16 conversions still
/// happening at projection entry (the F16-in narrowing) and at
/// embedding-gather / LM-head boundaries by building bf16-input
/// kernel siblings, ultimately deleting the dead f16 kernels.
pub fn bf16_residual_enabled() -> bool {
    parse_truthy_env("RVLLM_RESIDUAL_BF16").unwrap_or(true)
}

/// Cycle 55 step 14 master gate: enable end-to-end bf16-native chain
/// on the M=1 decode path. Implies `bf16_native_qkv_fast_path_enabled`
/// + bf16 dispatch at fused_qkv_rmsnorm + fused_rope_partial + GeLU
/// + gate_up/down F16-in fast paths + post-attn/FF epilogues. The
/// attention kernel's output stays f16 (its bf16-out siblings exist
/// from cycle 55 step 9 but are not yet dispatched in this gate; the
/// O-projection consumes f16 attn_out and writes bf16 via
/// `fused_norm_add_residual_bf16_f16in`'s built-in narrow).
///
/// Default OFF — empirical regression on long context (iteration 12
/// WHO@17k) is unresolved. Per user directive cycle 55 step 14: wire
/// the chain end-to-end accepting it may break short-term so we have
/// the substrate for further investigation; production override
/// keeps the f16 chain via `RVLLM_BF16_NATIVE_FULL_CHAIN=0`.
pub fn bf16_native_full_chain_enabled() -> bool {
    parse_truthy_env("RVLLM_BF16_NATIVE_FULL_CHAIN").unwrap_or(false)
}

/// Cycle 55 step 15 bisect overrides. When `RVLLM_BF16_NATIVE_FULL_CHAIN=1`
/// is on, individual sub-dispatches default to bf16. Setting any of
/// these env vars to 1 disables that one site (forces f16) while
/// keeping the rest bf16. Use to localize which bf16 kernel introduces
/// the empirical regression.
pub fn bf16_disable_qkv_rmsnorm() -> bool {
    parse_truthy_env("RVLLM_BF16_DISABLE_QKV_RMSNORM").unwrap_or(false)
}
pub fn bf16_disable_rope() -> bool {
    parse_truthy_env("RVLLM_BF16_DISABLE_ROPE").unwrap_or(false)
}

/// Cycle 55 step 13/18: bf16-native dispatch on the M=1 decode QKV
/// F16-in fast path (the production decode hot path). When ON, the
/// residual is consumed as bf16 directly by `rmsnorm_inplace_bf16` +
/// `Fp8GemvBf16In`, with a bf16→f16-sat narrow at the GEMV output to
/// keep the downstream RoPE+attention chain on f16.
///
/// **Default OFF (reverted from step-19's brief flip-to-ON).**
/// Iteration-16's "3/3 WHO@17k coherent" reading turned out to be
/// non-reproducible under iteration-17's clean-restart + zeroclaw
/// restart (3/3 garbage). Variance on this single prompt at
/// long-context is wider than a 3-sample read can reliably detect;
/// the cycle-54 stage-2.1 narrow path is more stable empirically.
/// Step-13 stays in code as opt-in research substrate; production
/// stays on cycle-54.
pub fn bf16_native_qkv_fast_path_enabled() -> bool {
    parse_truthy_env("RVLLM_BF16_NATIVE_QKV_FAST_PATH").unwrap_or(false)
}

// (Cycle 55 step 19 NOTE) The wholesale bf16 chain extension via
// FULL_CHAIN is empirically null/regressive — bf16 mantissa loss
// compounds through pre-FF rmsnorm + GEMV + bf16 gelu + ... producing
// NVFP4-incompatible Q/K/V values downstream. Cycle-54 stage-2.1's
// bf16→f16 narrow at projection input is the precision-bounding
// shape that keeps the chain stable. The FULL_CHAIN gate stays in
// code but the wholesale gate_up/down/gelu/epilogue extensions were
// MANUALLY REVERTED back to f16 in `gemma4_layer_exec.rs` after the
// iteration-17 wholesale flip empirically broke even short context.
// What stays under FULL_CHAIN: QKV F16-in fast path bf16 (also
// reachable via `RVLLM_BF16_NATIVE_QKV_FAST_PATH`) + fused_qkv_rmsnorm
// _bf16 + RoPE bf16in dispatch — these stay wired as a research
// substrate even though FULL_CHAIN itself remains empirically
// regressive (use it only for diagnostic experiments).

/// Per-token Q scale gate.
/// * For the FP8-KV path the scratch allocation defaults ON because
///   per-token Q materially helps PPL on prose (memory aa010018 in the
///   rvllm-coder scenario). Operator opts out via
///   `RVLLM_PER_TOKEN_Q_SCALE=0`.
/// * For the NVFP4-KV path the rope launcher gates separately and
///   defaults OFF; operator opts in via `RVLLM_PER_TOKEN_Q_SCALE=1`.
///
/// **Hadamard auto-implies per-token Q-scale.** When
/// `RVLLM_NVFP4_HADAMARD=1` is set, the rotated Q saturates the static
/// scalar Q-scale (rotation amax shifts so the fixed scalar can no
/// longer cover the post-rotation range). The rope kernel comment
/// pins this requirement explicitly. The two env vars used to be
/// independent — operators who set `HADAMARD=1` without
/// `PER_TOKEN_Q_SCALE=1` would silently get saturated Q values and
/// numerically wrong attention. We now force per-token Q-scale ON
/// whenever Hadamard is requested, with a one-shot warn so the
/// operator sees the auto-enable in journalctl. An explicit
/// `RVLLM_PER_TOKEN_Q_SCALE=0` together with `HADAMARD=1` is treated
/// as a contradiction the operator clearly meant by accident — the
/// auto-enable wins, but the warn line names it.
pub(crate) fn per_token_q_scale_enabled(default_on: bool) -> bool {
    static HADAMARD_OVERRIDE_WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    let raw = parse_truthy_env("RVLLM_PER_TOKEN_Q_SCALE").unwrap_or(default_on);
    let hadamard = parse_truthy_env("RVLLM_NVFP4_HADAMARD").unwrap_or(true);
    if hadamard && !raw {
        HADAMARD_OVERRIDE_WARNED.get_or_init(|| {
            tracing::warn!(
                "RVLLM_NVFP4_HADAMARD=1 requires per-token Q-scale to keep \
                 rotated Q below the static-scalar saturation threshold; \
                 auto-enabling RVLLM_PER_TOKEN_Q_SCALE for this process \
                 (set both to 1 explicitly to silence this warning)"
            );
        });
        return true;
    }
    raw
}

/// Generate a deterministic ±1 sign byte from
/// (layer_idx, channel_idx). Uses SplitMix32 (Java SplittableRandom
/// finalizer) for full 32-bit avalanche — an earlier FNV-1a + `(h&1)`
/// extractor here collapsed to a degenerate stride-2 pattern
/// `[1,-1,1,-1,...]` for adjacent channels because the prime
/// multiplier (0x01000193) preserves LSB parity, so the extracted
/// bit was effectively `channel_idx mod 2`. That broke the
/// rotation: R = H · diag(stride-2-±1) is structurally equivalent
/// to a Walsh-Hadamard variant of half the rank, not a random
/// orthogonal rotation.
///
/// SplitMix32: any single extracted bit is uncorrelated with
/// channel_idx LSB. Same seed → same chain → reproducible across
/// runs.
fn sign_byte_for(layer_idx: u32, channel_idx: u32) -> i8 {
    let seed = 0x9E3779B1u32
        .wrapping_mul(layer_idx.wrapping_add(0xC2B2AE35))
        .wrapping_add(channel_idx);
    let mut h = seed;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EBCA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2AE35);
    h ^= h >> 16;
    if (h & 1) == 0 { 1 } else { -1 }
}

/// Build the i8 sign-vector buffer host-side and upload to device.
/// Returns `None` when `RVLLM_NVFP4_HADAMARD` is off.
///
/// `Region` is not `Drop`, so the bytes returned by `arena.region(...)`
/// stay reserved as soon as the bump pointer has advanced — letting the
/// `Region` handle fall out of scope here is enough; no `mem::forget`
/// dance is required. This buffer must be allocated BEFORE the cuda
/// worker takes its scratch checkpoint, so subsequent
/// `arena.restore(scratch_ck)` calls don't rewind past it.
#[cfg(feature = "cuda")]
pub fn build_nvfp4_hadamard_signs(
    num_layers: u32,
    head_dim: u32,
    arena: &HbmArena<'_>,
) -> Result<Option<NvFp4HadamardAlloc>> {
    if !nvfp4_hadamard_enabled() {
        return Ok(None);
    }
    let bytes = (num_layers as u64) * (head_dim as u64);
    let mut host: Vec<i8> = Vec::with_capacity(bytes as usize);
    for l in 0..num_layers {
        for c in 0..head_dim {
            host.push(sign_byte_for(l, c));
        }
    }
    let region = arena.region("nvfp4_hadamard_signs", bytes as usize, 16)?;
    let base_ptr = region.device_ptr();
    unsafe {
        let r = cudarc::driver::sys::cuMemcpyHtoD_v2(
            base_ptr,
            host.as_ptr() as *const _,
            bytes as usize,
        );
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::Cuda {
                kind: rvllm_core::CudaErrorKind::MemcpyFailed,
                op: "cuMemcpyHtoD nvfp4_hadamard_signs",
                ctx: rvllm_core::CudaCtx {
                    stream: 0,
                    kernel: "nvfp4_hadamard_signs_upload",
                    launch: None,
                    device: 0,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
    }
    eprintln!(
        "[hadamard] uploaded {} layers × {} signs ({} bytes) to {:#x}",
        num_layers, head_dim, bytes, base_ptr
    );
    Ok(Some(NvFp4HadamardAlloc {
        base_ptr,
        bytes,
        head_dim,
        num_layers,
    }))
}
// === END HADAMARD ROTATION ===

/// Allocate the NVFP4 shadow KV / Q / Q-throwaway regions for the
/// layers named in `RVLLM_NVFP4_SHADOW_LAYERS`. Returns `Ok(None)`
/// when the env is unset (no shadow path active).
///
/// Same lifetime-correctness contract as `build_nvfp4_hadamard_signs`:
/// the bytes returned by `arena.region(...)` stay reserved as soon as
/// the bump pointer has advanced, but the regions MUST be allocated
/// BEFORE the cuda worker takes its scratch checkpoint — otherwise
/// `arena.restore(scratch_ck)` between requests rewinds the bump
/// pointer past them and the persistent pointer in
/// `Gemma4Bringup::nvfp4_shadow` aliases freshly-overwritten scratch.
/// The previous lazy-allocation site inside `run_generate` had this
/// exact bug (Codex14 / Codex16 #2): silent corruption from request
/// 2 onward. The cuda worker now calls this at startup, before the
/// checkpoint; `run_generate` re-resolves the same allocation through
/// `self.nvfp4_shadow.lock()` and never re-allocates.
#[cfg(feature = "cuda")]
pub fn build_nvfp4_shadow_alloc(
    arch: &rvllm_loader::gemma4_arch::Gemma4Arch,
    num_blocks_total: u32,
    sliding_blocks: u32,
    block_size: u32,
    arena: &HbmArena<'_>,
) -> Result<Option<NvFp4ShadowAlloc>> {
    let shadow_set = match crate::gemma4_layer_exec::parse_shadow_layers() {
        Some(s) => s,
        None => return Ok(None),
    };
    if shadow_set.is_empty() {
        return Ok(None);
    }
    let mut layer_offsets: Vec<u64> = vec![u64::MAX; arch.num_hidden_layers];
    let mut shadow_total_bytes: u64 = 0;
    {
        let mut cursor: u64 = 0;
        for l in 0..arch.num_hidden_layers {
            if !shadow_set.contains(&(l as u32)) {
                continue;
            }
            layer_offsets[l] = cursor;
            let is_global = arch.layer_types[l]
                == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
            let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
            let nkvh = arch.num_kv_heads_for_layer(l) as u32;
            let hd = arch.head_dim_for_layer(l) as u32;
            let layer_bytes = 2u64 * (layer_blocks as u64) * (block_size as u64)
                * (nkvh as u64) * (hd as u64) * 2;
            cursor += layer_bytes;
            shadow_total_bytes += layer_bytes;
        }
    }
    let shadow_q_per_layer_bytes: u64 =
        2u64 * (arch.num_attention_heads as u64) * (arch.max_head_dim() as u64);
    let shadow_q_throwaway_bytes: u64 =
        (num_blocks_total as u64) * (block_size as u64) * shadow_q_per_layer_bytes;
    let shadow_q_total_bytes: u64 = shadow_q_per_layer_bytes * (shadow_set.len() as u64);

    let shadow_kv_bytes_alloc = shadow_total_bytes.max(16) as usize;
    let shadow_q_bytes_alloc = shadow_q_total_bytes.max(16) as usize;
    let shadow_throwaway_alloc = shadow_q_throwaway_bytes.max(16) as usize;

    let region = arena.region("nvfp4_shadow_kv", shadow_kv_bytes_alloc, 256)?;
    let q_region = arena.region("nvfp4_shadow_q", shadow_q_bytes_alloc, 256)?;
    let throwaway_region = arena.region(
        "nvfp4_shadow_q_throwaway",
        shadow_throwaway_alloc,
        256,
    )?;
    // Codex27-4: shadow init memsets used to drop their CUresult.
    // Diagnostic output is the whole point of this path; partially
    // uninitialised shadow regions would silently corrupt the dump.
    unsafe {
        cuda_check!(cudarc::driver::sys::cuMemsetD8_v2(
            region.device_ptr(), 0, shadow_kv_bytes_alloc),
            "nvfp4_shadow_alloc_kv_zero", 0u64);
        cuda_check!(cudarc::driver::sys::cuMemsetD8_v2(
            q_region.device_ptr(), 0, shadow_q_bytes_alloc),
            "nvfp4_shadow_alloc_q_zero", 0u64);
        cuda_check!(cudarc::driver::sys::cuMemsetD8_v2(
            throwaway_region.device_ptr(), 0, shadow_throwaway_alloc),
            "nvfp4_shadow_alloc_throwaway_zero", 0u64);
    }
    eprintln!(
        "[nvfp4-shadow] allocated {} MiB f16 shadow KV + {} KiB per-layer Q for {} layers \
         (above scratch checkpoint, persistent across requests): {:?}",
        shadow_total_bytes / (1024 * 1024),
        shadow_q_total_bytes / 1024,
        shadow_set.len(),
        shadow_set,
    );
    Ok(Some(NvFp4ShadowAlloc {
        shadow_ptr: region.device_ptr(),
        shadow_bytes: shadow_total_bytes,
        layer_offsets,
        layer_indices: shadow_set,
        shadow_q_ptr: q_region.device_ptr(),
        shadow_q_total_bytes,
        shadow_q_per_layer_bytes,
        shadow_q_throwaway_ptr: throwaway_region.device_ptr(),
    }))
}

// === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
/// Parallel to the main KV region but: (a) only the instrumented
/// layers have a slot; (b) every instrumented layer is stored as F16
/// regardless of the primary KV dtype. No scale region needed.
pub struct NvFp4ShadowAlloc {
    pub shadow_ptr: u64,
    pub shadow_bytes: u64,
    /// Per-layer byte offset into `shadow_ptr`. `u64::MAX` sentinel
    /// for layers NOT in the instrumented set.
    pub layer_offsets: Vec<u64>,
    pub layer_indices: Vec<u32>,
    /// Per-instrumented-layer Q snapshot region. Sized for
    /// `num_shadow_layers * num_attention_heads * max_head_dim * 2`
    /// bytes (f16). Populated on decode step 0 only, AFTER the shadow
    /// f16 RoPE (which writes post-RoPE Q into `scratch.q_normed`)
    /// and BEFORE the primary NVFP4 RoPE clobbers it. Per-layer slot
    /// size is uniform (`q_per_layer_bytes`) even when the layer's
    /// head_dim is smaller than max_head_dim — the tail of the slot
    /// is then zero and the Python analyzer truncates using
    /// `head_dim` from meta.json.
    pub shadow_q_ptr: u64,
    pub shadow_q_total_bytes: u64,
    pub shadow_q_per_layer_bytes: u64,
    /// Q throwaway scratch — a single-slot f16 buffer (same size as
    /// one per-layer Q slot) that `rope_f16kv_shadow` targets when we
    /// are NOT capturing (prefill steps, decode step > 0). Keeps
    /// `scratch.q_normed` untouched so the subsequent primary
    /// `rope_nvfp4kv` rotates Q exactly once. Without this, shadow
    /// rope's q_out=q_normed caused double-RoPE on q_fp8 and
    /// corrupted live inference whenever shadow was on.
    pub shadow_q_throwaway_ptr: u64,
}
// === END NVFP4 SHADOW DIAGNOSTIC ===

/// Defensive presence check used by `Gemma4Bringup::layer_kernels`.
///
/// `Gemma4LayerKernels` is a per-bringup struct shared across both
/// sliding and global layers, so it can hold only one symbol per
/// kernel slot. The current loader instantiates two `Fa2PtxKernels`
/// (one per attention backend) from the SAME NVFP4 RoPE PTX source,
/// so the symbols are functionally identical even though the raw
/// `KernelFn` pointers differ (two `cuModuleLoad` calls produce two
/// distinct handles into the same code). What matters structurally
/// is presence symmetry: if sliding has the kernel loaded but
/// global doesn't (or vice versa), pulling from `sliding_attention`
/// here would silently feed sliding's RoPE into global layers, or
/// global's `None` would mask a partial-build problem. We panic
/// loudly on the asymmetric case so a partial PTX rebuild surfaces
/// here instead of as wrong attention 60 layers later. Pointer
/// equality is NOT asserted because it's not the property the
/// downstream code relies on.
#[cfg(feature = "cuda")]
fn assert_rope_kernels_match(
    label: &'static str,
    sliding: Option<rvllm_kernels::KernelFn>,
    global: Option<rvllm_kernels::KernelFn>,
) -> Result<()> {
    // Codex26-2: a partial PTX/manifest state used to panic the
    // server here. Return a typed config error instead so the bring-
    // up failure surfaces as a request-level 500 (or shutdown path)
    // rather than aborting the whole process. Operators rebuild
    // kernels/ and retry.
    match (sliding.is_some(), global.is_some()) {
        (true, true) | (false, false) => Ok(()),
        _ => Err(rvllm_core::RvllmError::Config {
            err: rvllm_core::ConfigError::Inconsistent {
                reasons: vec![format!(
                    "{label}: sliding/global Fa2PtxKernels disagree on whether \
                     this NVFP4 RoPE kernel is loaded (sliding={}, global={}). \
                     Symptomatic of a partial PTX build; rebuild kernels/.",
                    sliding.is_some(),
                    global.is_some(),
                )],
            },
            field: "Fa2PtxKernels.rope_nvfp4kv",
        }),
    }
}

impl Gemma4Bringup {
    pub fn load(paths: Gemma4EnginePaths, arena_bytes: usize) -> Result<Self> {
        // RVLLM_NVFP4_SPLIT_GQA defaults to true (operator-validated
        // as the most stable production path on GB10/Gemma 4); explicit
        // `=0` opts out for diagnostic comparisons. Log only the opt-out
        // case so production startups stay quiet.
        if std::env::var_os("RVLLM_NVFP4_SPLIT_GQA")
            .map(|v| v == "0" || v == "false" || v == "FALSE")
            .unwrap_or(false)
        {
            tracing::warn!(
                "RVLLM_NVFP4_SPLIT_GQA=0: opted out of GQA-shared NVFP4 \
                 split-decode (production default). Per-Q split kernel \
                 will be used instead — expect slightly more CTAs per \
                 decode step. Set unset or =1 to restore the default."
            );
        }
        let ctx = Arc::new(CudaContextHandle::init(0)?);
        // Resolve the compile target once per bring-up and thread it
        // through — every call to `ctx.compute_capability()` + the
        // lookup costs nothing individually but spreading it across 5
        // sites means "which CC are we on?" reads inconsistent if a
        // future refactor accidentally shadows `ctx`.
        #[cfg(feature = "cuda")]
        let compile_target: Option<rvllm_core::CompileTarget> = {
            let (major, minor) = ctx.compute_capability();
            rvllm_core::CompileTarget::from_compute_capability(major, minor)
        };
        #[cfg(not(feature = "cuda"))]
        let compile_target: Option<rvllm_core::CompileTarget> = None;

        // Arena backing picked per compute capability — see `Bringup::load`
        // in bring_up.rs for the full rationale (GB10 has no dedicated HBM,
        // cuMemAllocManaged is the right allocator there).
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

        let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&paths.model_dir)?;
        let model = rvllm_loader::gemma4_load::load_gemma4_model(&paths.model_dir, &arena, &arch)?;

        // On sm_121 the arena is `cuMemAllocManaged` pages that fault
        // to the GPU on first touch. After the weight upload the
        // populated region (~30 GiB for Gemma 4 31B) hasn't faulted
        // yet — prefetching it here removes the page-fault storm
        // from the first decode iteration, so first-token latency
        // stops carrying 30 GiB of H→D page migration cost. CUDA 13
        // dropped the single-arg `cuMemPrefetchAsync` in favour of
        // `_v2` with a `CUmemLocation`; cudarc 0.19 only wraps the
        // v2 form for cuda-13. Best-effort: a non-zero RC is logged
        // but doesn't fail bring-up.
        #[cfg(all(feature = "gb10", feature = "cuda"))]
        unsafe {
            if matches!(compile_target, Some(rvllm_core::CompileTarget::Sm121)) {
                let prefetch_bytes = arena.used();
                if prefetch_bytes > 0 {
                    let loc = cudarc::driver::sys::CUmemLocation {
                        type_: cudarc::driver::sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                        id: 0,
                    };
                    let rc = cudarc::driver::sys::cuMemPrefetchAsync_v2(
                        arena.base_ptr(),
                        prefetch_bytes,
                        loc,
                        0,
                        stream.raw() as _,
                    );
                    if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        tracing::warn!(
                            "cuMemPrefetchAsync_v2({prefetch_bytes} bytes) rc={rc:?} — first-token latency may spike"
                        );
                    } else {
                        let _ = cudarc::driver::sys::cuStreamSynchronize(
                            stream.raw() as _,
                        );
                    }
                }
            }
        }

        // Per-arch kernel subdirectory resolution — see `resolve_kernels_dir`.
        let kernels_dir = crate::bring_up::resolve_kernels_dir(&ctx, &paths.kernels_dir)?;
        let manifest_path = kernels_dir.join("manifest.json");
        let manifest = rvllm_kernels::manifest::KernelManifest::load_and_verify(&manifest_path)?;
        // Codex29-1: hard-fail if manifest.arch != runtime arch. The
        // kernels_dir was picked by arch already, but a stale / copied
        // manifest from another arch can be size+sha-consistent and
        // would otherwise slip through.
        if let Some(t) = compile_target {
            manifest.assert_arch(t.as_sm_str())?;
        }
        // Codex23-2: warn (not fail) on manifest-vs-binary revision drift.
        // The kernels crate's build.rs bakes `RVLLM_BUILD_REVISION` from
        // git short HEAD; the manifest carries the same field per
        // kernels/build.sh. A self-consistent stale manifest passing
        // load_and_verify (size + sha match each other) but pairing
        // with a binary that has a newer launch ABI shows up here.
        manifest.warn_if_revision_drift(rvllm_kernels::manifest::VerifiedManifest::BUILD_REVISION);
        let kernels = Arc::new(KernelLoader::new(manifest));

        // Attention backend selection. On SM80/SM89/SM90 we stick with
        // the FA3 `.so` (WGMMA + TMA). On sm_121 (GB10) FA3 cannot
        // load — WGMMA doesn't exist on Blackwell consumer silicon —
        // so we route through the PTX-launched FA2 kernels we already
        // compile for every arch. The FA2 launch body is still a
        // follow-up (the decode/prefill launchers return
        // `FeatureNotAvailable` for the `Fa2Ptx` variant), but
        // bring-up now completes on GB10 without a hard fail on
        // `Fa3SoMissing`. See `rvllm_attention::Fa2PtxKernels` docs.
        let (sliding_attention, global_attention) = {
            #[cfg(feature = "gb10")]
            let is_gb10 = matches!(compile_target, Some(rvllm_core::CompileTarget::Sm121));
            #[cfg(not(feature = "gb10"))]
            let is_gb10 = false;
            if is_gb10 {
                let sliding = AttentionBackend::Fa2Ptx(rvllm_attention::Fa2PtxKernels::load(
                    &kernels,
                    arch.head_dim_sliding as u32,
                )?);
                let global = AttentionBackend::Fa2Ptx(rvllm_attention::Fa2PtxKernels::load(
                    &kernels,
                    arch.head_dim_global as u32,
                )?);
                (sliding, global)
            } else {
                // Sliding layers use the FA3 SM90 backend at head_dim=256.
                let sliding = AttentionBackend::Fa3(Fa3Kernels::load(
                    paths.fa3_so.clone(),
                    arch.head_dim_sliding as u32,
                )?);
                // Global layers use the generic fallback paged attention path.
                // Default location is next to the FA3 .so; an explicit override
                // keeps bench/deploy flows flexible while avoiding a new required flag.
                let global_attention_so = std::env::var_os("RVLLM_FA_FALLBACK_SO")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| paths.fa3_so.with_file_name("libfa_sm89_kernels.so"));
                let global = AttentionBackend::Fa3(Fa3Kernels::load(
                    global_attention_so,
                    arch.head_dim_global as u32,
                )?);
                (sliding, global)
            }
        };

        // Sm121 uses CutlassBackend::SoSm120 or Absent; neither consumes
        // the SM90 variant table, so the policy.json can stay unread on
        // that target. Saves a mandatory env var + avoids rejecting a
        // missing-or-placeholder file on GB10 runs.
        let skip_policy =
            matches!(compile_target, Some(rvllm_core::CompileTarget::Sm121));
        let (policy, variants): (Policy, Vec<_>) = if skip_policy {
            let empty = Policy {
                revision: String::new(),
                arch: "sm_121".into(),
                variants: Vec::new(),
                entries: Default::default(),
            };
            (empty, (0..16u32).map(rvllm_cutlass::VariantId).collect())
        } else {
            let policy_bytes = std::fs::read(&paths.policy_json)
                .map_err(|source| rvllm_core::RvllmError::Io {
                    err: rvllm_core::IoError::from(&source),
                    path: paths.policy_json.clone(),
                    source,
                })?;
            let policy: Policy = serde_json::from_slice(&policy_bytes).map_err(|e| {
                rvllm_core::RvllmError::config(
                    rvllm_core::ConfigError::Inconsistent {
                        reasons: vec![format!("policy.json parse: {e}")],
                    },
                    "policy.json",
                )
            })?;
            let mut variants: std::collections::BTreeSet<_> =
                policy.entries.values().map(|e| e.variant).collect();
            for v in 0..16u32 {
                variants.insert(rvllm_cutlass::VariantId(v));
            }
            (policy, variants.into_iter().collect())
        };
        // CUTLASS backend selection — see `bring_up::Bringup::load`
        // for the full rationale (sm_121 has no compatible `.so`).
        let cutlass =
            CutlassBackend::load_for(compile_target, paths.cutlass_so.clone(), &variants)?;

        // Codex51-3: surface the SM121 fast-path status so operators
        // can see at a glance whether `RVLLM_FP8_GEMM_CUTLASS_SM120`
        // is wired correctly. Single-token decode (M=1) uses the
        // native GEMV fastpath regardless; the CUTLASS SoSm120 path
        // matters for batched decode and prefill (M>=128). Without
        // explicit logging, a missing .so or unset env silently
        // routes M>=128 traffic onto cuBLASLt scalar mode, costing
        // significant prefill TTFT on sm_121.
        {
            // Codex53-3: dispatch gate is `parse_truthy_env(...)
            // .unwrap_or(true)` at gemma4_layer_exec.rs:2902 — env
            // unset DEFAULTS TO ENABLED. The Codex51-3 startup log
            // had this inverted (claimed unset → fallback) which
            // would have led operators to set the env unnecessarily
            // and possibly miss the real "opt-out" semantics
            // (`RVLLM_FP8_GEMM_CUTLASS_SM120=0`). Three states to
            // surface:
            //   .so loaded + env != "0"  → fast path active (default
            //                              when env is unset)
            //   .so loaded + env  = "0"  → operator opt-out (regression
            //                              diagnosis per the
            //                              gemma4_layer_exec.rs comment)
            //   .so missing              → fallback (warn if env != "0",
            //                              info otherwise)
            let env_raw = std::env::var("RVLLM_FP8_GEMM_CUTLASS_SM120").ok();
            let env_off = matches!(env_raw.as_deref(), Some("0") | Some("false") | Some("FALSE"));
            let backend_name = match &cutlass {
                CutlassBackend::SoSm120(_) => "SoSm120",
                CutlassBackend::So(_) => "So",
                CutlassBackend::Absent => "Absent",
                _ => "Other",
            };
            if matches!(compile_target, Some(rvllm_core::CompileTarget::Sm121)) {
                if matches!(cutlass, CutlassBackend::SoSm120(_)) {
                    if env_off {
                        tracing::warn!(
                            backend = backend_name,
                            env = env_raw.as_deref().unwrap_or(""),
                            "[cutlass-sm120] SoSm120 .so loaded but \
                             RVLLM_FP8_GEMM_CUTLASS_SM120=0 — M>=128 GEMM \
                             routed to cuBLASLt scalar (operator opt-out, \
                             regression-diagnosis mode). Unset the env or \
                             set =1 to re-enable the fast path."
                        );
                    } else {
                        tracing::info!(
                            backend = backend_name,
                            env = env_raw.as_deref().unwrap_or("(unset, default-on)"),
                            "[cutlass-sm120] SoSm120 .so loaded — M>=128 GEMM \
                             takes the CUTLASS blockwise fast path."
                        );
                    }
                } else if !env_off {
                    tracing::warn!(
                        backend = backend_name,
                        "[cutlass-sm120] CUTLASS .so not loaded \
                         (backend = {backend_name}); M>=128 paths run on \
                         cuBLASLt scalar. Build via \
                         kernels/build_cutlass_sm120_so.sh and verify \
                         kernels/sm_121/libcutlass_sm120.so is present \
                         to recover the prefill TTFT.",
                    );
                } else {
                    tracing::info!(
                        backend = backend_name,
                        "[cutlass-sm120] CUTLASS .so not loaded and \
                         RVLLM_FP8_GEMM_CUTLASS_SM120=0 — operator-disabled, \
                         M>=128 paths use cuBLASLt scalar."
                    );
                }
            }
        }

        let cublaslt_ws_bytes: usize = 32 * 1024 * 1024;
        let cublaslt_ws_region = arena.region("cublaslt_ws", cublaslt_ws_bytes, 256)?;
        let cublaslt = CublasLt::new(cublaslt_ws_region.device_ptr(), cublaslt_ws_bytes)?;
        let cublaslt_ws = HbmArenaCheckpoint {
            offset_bytes: 0,
            bytes: cublaslt_ws_bytes,
        };

        // `compile_target` is also what the fused loader uses to gate
        // `Fp8GemvVariant::WprNative` (sm_100+ only). `Some(None)` vs
        // `None` is distinct: it means "probe succeeded but CC isn't
        // in our target matrix", which falls back to `WprLut`.
        let fused = load_gemma4_fused(&kernels, compile_target)?;

        // Spec-decode commit 3: pre-compute the source-layer indices
        // the Gemma 4 assistant-drafter will consume at draft time.
        // `None` on archs without `num_kv_shared_layers` (e.g. 31B);
        // populated on E4B-it as `(22, 23)`.
        let assistant_kv_sources = arch
            .assistant_shared_kv_sources()
            .map(|(s, f)| {
                eprintln!(
                    "[gemma4] assistant-drafter shared-KV sources: \
                     sliding=layer {s}, full=layer {f} \
                     (num_hidden_layers={}, num_kv_shared_layers={:?})",
                    arch.num_hidden_layers, arch.num_kv_shared_layers
                );
                Gemma4AssistantKvSources {
                    sliding_source_layer: s as u32,
                    full_source_layer: f as u32,
                }
            });

        Ok(Self {
            ctx,
            arena,
            stream,
            arch,
            model,
            kernels,
            cutlass,
            cublaslt,
            cublaslt_ws,
            sliding_attention,
            global_attention,
            policy,
            fused,
            assistant_kv_sources,
            drafter: std::sync::Mutex::new(None),
            base_last_hidden_ptr: std::sync::atomic::AtomicU64::new(0),
            base_last_hidden_snapshot_pending:
                std::sync::atomic::AtomicBool::new(false),
            base_last_k_hidden_ptr: std::sync::atomic::AtomicU64::new(0),
            base_last_k_count: std::sync::atomic::AtomicU32::new(0),
            skip_next_warmup: std::sync::atomic::AtomicBool::new(false),
            saved_warmup_b_p: std::sync::atomic::AtomicU32::new(u32::MAX),
            force_emit_accepted: std::sync::atomic::AtomicBool::new(false),
            force_batched_verify: std::sync::atomic::AtomicBool::new(false),
            force_common_prefix_override: std::sync::atomic::AtomicU32::new(u32::MAX),
            skip_prefix_cache_publish: std::sync::atomic::AtomicBool::new(false),
            force_prefill_only: std::sync::atomic::AtomicBool::new(false),
            base_last_k_snapshot_pending:
                std::sync::atomic::AtomicBool::new(false),
            last_spec_stats: std::sync::Mutex::new(None),
            spec_decode_step_logits: std::sync::Mutex::new(Vec::new()),
            spec_logits_capture_active:
                std::sync::atomic::AtomicBool::new(false),
            prefix_cache: std::sync::Mutex::new(None),
            // (assistant_kv_sources is set above; drafter slot stays
            // empty until commit 7 calls `ensure_drafter` from the
            // spec-decode loop. Backward-compat: with spec_decode=false
            // the slot is never populated and HBM stays untouched.)
            // NVFP4 shadow diagnostic state (lazy-init in run_generate).
            nvfp4_shadow: std::sync::Mutex::new(None),
            nvfp4_shadow_dumped: std::sync::atomic::AtomicBool::new(false),
            // === HADAMARD ROTATION ===
            // Lazy-init in `run_generate` once we know the head_dim
            // and num_layers (mirroring nvfp4_shadow's lazy alloc).
            nvfp4_hadamard: std::sync::Mutex::new(None),
            // === END HADAMARD ROTATION ===
        })
    }

    /// Spec-decode commit 4: lazy-initialise the Gemma 4 assistant
    /// drafter. Reads `RVLLM_GEMMA4_DRAFTER_DIR` (validated at startup
    /// by `ServerConfig`), parses the safetensors layout via
    /// `rvllm_loader::gemma4_drafter::Gemma4DrafterWeightLayout`, then
    /// uploads every BF16 weight as F16 + the I64 token_ordering
    /// as-is to the engine's HBM arena.
    ///
    /// Backward-compat invariants:
    ///   * Only called from paths gated by
    ///     `ServerConfig::spec_decode == true`. With the gate off this
    ///     function is unreachable; the drafter slot stays `None` and
    ///     consumes zero HBM.
    ///   * At-most-once per engine instance — the mutex guards against
    ///     concurrent uploads. Subsequent calls observe `Some(_)` and
    ///     return immediately.
    ///   * Refuses to upload when the base arch has no
    ///     `assistant_kv_sources` (e.g. 31B) — surfaces a clear error
    ///     instead of silently producing a drafter that has no base
    ///     K/V to cross-attend to.
    ///
    /// On success, `self.drafter.lock()` returns `Some(runtime)` for
    /// future `run_generate` calls (wired in commit 7).
    pub fn ensure_drafter(&self, drafter_dir: &std::path::Path) -> Result<()> {
        // Cheap pre-check outside the lock.
        if self.drafter.lock().unwrap().is_some() {
            return Ok(());
        }
        if self.assistant_kv_sources.is_none() {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail:
                        "ensure_drafter: base model has no \
                         assistant_kv_sources (num_kv_shared_layers \
                         unset). The Gemma 4 assistant drafter requires \
                         an E4B-style model with a shared-KV tail."
                            .into(),
                },
                ctx: LoaderCtx { path: drafter_dir.to_path_buf(), tensor: None },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let layout = rvllm_loader::gemma4_drafter::Gemma4DrafterWeightLayout::from_dir(
            drafter_dir,
        )?;
        if layout.arch.backbone_hidden_size != self.arch.hidden_size {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "drafter backbone_hidden_size={} != base hidden_size={}",
                        layout.arch.backbone_hidden_size, self.arch.hidden_size
                    ),
                },
                ctx: LoaderCtx { path: drafter_dir.to_path_buf(), tensor: None },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if layout.arch.vocab_size != self.arch.vocab_size {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "drafter vocab_size={} != base vocab_size={}",
                        layout.arch.vocab_size, self.arch.vocab_size
                    ),
                },
                ctx: LoaderCtx { path: drafter_dir.to_path_buf(), tensor: None },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if layout.arch.pre_projection_in_dim != 2 * self.arch.hidden_size {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "drafter pre_projection_in_dim={} != 2 * base hidden_size={}",
                        layout.arch.pre_projection_in_dim, self.arch.hidden_size
                    ),
                },
                ctx: LoaderCtx { path: drafter_dir.to_path_buf(), tensor: None },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        #[cfg(feature = "cuda")]
        let mut rt = crate::gemma4_drafter::Gemma4DrafterRuntime::load(&layout, &self.arena)?;
        #[cfg(not(feature = "cuda"))]
        let mut rt = crate::gemma4_drafter::Gemma4DrafterRuntime::load_mock(&layout)?;
        // Commit 6: load the MaskedEmbedder PTX kernel and attach it to
        // the drafter runtime. The kernel is only built when this code
        // path runs, so a non-spec engine never pays the PTX load.
        #[cfg(feature = "cuda")]
        {
            let module = self.kernels.load_ptx("gemma4_masked_embedder")?;
            let entry = module
                .get_function("gemma4_masked_embedder_argmax_f16_kernel")?;
            rt.attach_masked_embedder_kernel(module, entry);
        }
        // Commit 11: load the paged-decode FA-2 f16io kernel so the
        // assistant cross-attention launcher can fire against the
        // F16 shadow KV without going through the base attention
        // backend (which holds Fp8/NVFP4-specific policy state).
        #[cfg(feature = "cuda")]
        {
            let module = self.kernels.load_ptx("flash_attention")?;
            let entry = module
                .get_function("flash_attention_2_decode_f16io_kernel")?;
            rt.attach_flash_attention_kernel(module, entry);
        }
        // Commit 15: BC=16 sibling of the f16io kernel, needed for
        // head_dim=512 (drafter global layer) so dynamic smem fits
        // the sm_121 ~100 KiB per-CTA cap.
        #[cfg(feature = "cuda")]
        {
            let module = self.kernels.load_ptx("flash_attention_decode_f16io_bc16")?;
            let entry = module
                .get_function("flash_attention_2_decode_f16io_kernel")?;
            rt.attach_flash_attention_bc16_kernel(module, entry);
        }
        // Commits 10b/10c: load the shadow-KV dequant kernels (FP8
        // → F16 and NVFP4-packed → F16). Same lazy gate as the rest
        // of the drafter PTX bundle.
        #[cfg(feature = "cuda")]
        {
            let module = self.kernels.load_ptx("gemma4_drafter_dequant")?;
            let fp8_entry = module
                .get_function("gemma4_drafter_dequant_fp8_to_f16_kernel")?;
            let nvfp4_entry = module
                .get_function("gemma4_drafter_dequant_nvfp4_to_f16_kernel")?;
            rt.attach_drafter_dequant_kernels(module, fp8_entry, nvfp4_entry);
        }
        // Commit 16: allocate the base_last_hidden_ptr buffer above
        // the scratch checkpoint so the next run_generate writes the
        // normalized pre-lm-head hidden of the last prompt token here
        // and arena.restore() doesn't reclaim it between requests.
        #[cfg(feature = "cuda")]
        {
            if self
                .base_last_hidden_ptr
                .load(std::sync::atomic::Ordering::Acquire)
                == 0
            {
                let h = self.arch.hidden_size;
                let region = self.arena.region(
                    "gemma4_base_last_hidden", h * 2, 16,
                )?;
                unsafe {
                    use cudarc::driver::sys::*;
                    let rc = cuMemsetD8_v2(region.device_ptr(), 0, h * 2);
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "gemma4_base_last_hidden zero-init",
                            rvllm_core::CudaErrorKind::MemcpyFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                self.base_last_hidden_ptr.store(
                    region.device_ptr(),
                    std::sync::atomic::Ordering::Release,
                );
                eprintln!(
                    "[gemma4-drafter] base_last_hidden buffer allocated \
                     above scratch ({} bytes f16)",
                    h * 2
                );
            }
            // Commit 38: allocate K-row hidden buffer for batched
            // verify. Sized for spec_k up to 16 (way more than the
            // typical 4-8 used in practice). Once-per-engine alloc.
            if self
                .base_last_k_hidden_ptr
                .load(std::sync::atomic::Ordering::Acquire)
                == 0
            {
                let h = self.arch.hidden_size;
                let bytes = MAX_SPEC_K * h * 2;
                let region = self.arena.region(
                    "gemma4_base_last_k_hidden", bytes, 16,
                )?;
                unsafe {
                    use cudarc::driver::sys::*;
                    let rc = cuMemsetD8_v2(region.device_ptr(), 0, bytes);
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "gemma4_base_last_k_hidden zero-init",
                            rvllm_core::CudaErrorKind::MemcpyFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                self.base_last_k_hidden_ptr.store(
                    region.device_ptr(),
                    std::sync::atomic::Ordering::Release,
                );
                eprintln!(
                    "[gemma4-drafter] base_last_k_hidden buffer allocated \
                     ({} bytes f16, MAX_SPEC_K={})",
                    bytes, MAX_SPEC_K
                );
            }
        }
        // Commit 9: allocate F16 shadow KV at the two base source
        // layers (sliding source = layer 22, full source = layer 23
        // on E4B). Sizing mirrors the base's paged-decode expectation
        // — `[num_blocks_total * block_size * num_kv_heads *
        // head_dim] f16` per K and per V buffer — so the existing
        // `flash_attention_2_decode_f16io_kernel` can read directly
        // from these regions (no new kernel needed). Allocation
        // happens ABOVE the scratch checkpoint so per-request
        // `arena.restore` never reclaims it.
        //
        // Population (copy/dequant from base's actual KV) is wired
        // in a follow-up commit; the launcher wiring follows that.
        // For now the regions are zero-initialised and resident.
        #[cfg(feature = "cuda")]
        {
            let sources = self
                .assistant_kv_sources
                .expect("checked above (asistant_kv_sources.is_none())");
            let sliding_li = sources.sliding_source_layer as usize;
            let full_li = sources.full_source_layer as usize;
            let block_size: u32 = 32;
            let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(1024);

            let sliding_nkvh = self.arch.num_kv_heads_for_layer(sliding_li) as u32;
            let sliding_hd = self.arch.head_dim_for_layer(sliding_li) as u32;
            let full_nkvh = self.arch.num_kv_heads_for_layer(full_li) as u32;
            let full_hd = self.arch.head_dim_for_layer(full_li) as u32;

            // bytes = num_blocks_total * block_size * nkvh * hd * 2
            //         (sizeof f16 = 2)
            let sliding_layer_bytes = (num_blocks_total as usize)
                * (block_size as usize)
                * (sliding_nkvh as usize)
                * (sliding_hd as usize)
                * 2;
            let full_layer_bytes = (num_blocks_total as usize)
                * (block_size as usize)
                * (full_nkvh as usize)
                * (full_hd as usize)
                * 2;

            let alloc_zeroed = |name: &'static str, bytes: usize| -> Result<u64> {
                let region = self.arena.region(name, bytes.max(16), 256)?;
                unsafe {
                    use cudarc::driver::sys::*;
                    let rc = cuMemsetD8_v2(region.device_ptr(), 0, bytes);
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "drafter shadow KV zero-init",
                            rvllm_core::CudaErrorKind::MemcpyFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                Ok(region.device_ptr())
            };

            let sliding_k_ptr = alloc_zeroed("drafter_shadow_k_sliding", sliding_layer_bytes)?;
            let sliding_v_ptr = alloc_zeroed("drafter_shadow_v_sliding", sliding_layer_bytes)?;
            let full_k_ptr    = alloc_zeroed("drafter_shadow_k_full",    full_layer_bytes)?;
            let full_v_ptr    = alloc_zeroed("drafter_shadow_v_full",    full_layer_bytes)?;

            let shadow = crate::gemma4_drafter::DrafterShadowKv {
                sliding_k_ptr,
                sliding_v_ptr,
                full_k_ptr,
                full_v_ptr,
                sliding_layer_bytes,
                full_layer_bytes,
                block_size,
                num_blocks_total,
                max_blocks_per_seq: num_blocks_total,
                sliding_num_kv_heads: sliding_nkvh,
                sliding_head_dim: sliding_hd,
                full_num_kv_heads: full_nkvh,
                full_head_dim: full_hd,
            };
            let total_mib = (2 * (sliding_layer_bytes + full_layer_bytes)) as f64
                / (1024.0 * 1024.0);
            eprintln!(
                "[gemma4-drafter] shadow KV allocated above scratch: \
                 sliding=layer {sliding_li} ({sliding_nkvh}×{sliding_hd} \
                 hd, {:.1} MiB K+V), full=layer {full_li} ({full_nkvh}×\
                 {full_hd} hd, {:.1} MiB K+V), total {:.1} MiB f16",
                2.0 * sliding_layer_bytes as f64 / (1024.0 * 1024.0),
                2.0 * full_layer_bytes as f64 / (1024.0 * 1024.0),
                total_mib,
            );
            rt.attach_shadow_kv(shadow);
        }
        let mut slot = self.drafter.lock().unwrap();
        // Re-check inside the lock — another thread may have raced.
        if slot.is_none() {
            *slot = Some(rt);
        }
        Ok(())
    }

    /// Allocate the session-level prefix cache's KV cache region.
    /// Must be called BEFORE the cuda worker takes its scratch
    /// checkpoint — the region's device pointer is captured as a
    /// raw u64 and the arena's bump pointer is advanced past it,
    /// so subsequent `arena.restore(scratch_ck)` calls won't clobber
    /// the persistent KV data.
    ///
    /// Safe to call multiple times; becomes a no-op after the first
    /// successful init.
    pub fn init_prefix_cache(&self) -> Result<()> {
        let mut guard = self.prefix_cache.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
        let arch = &self.arch;
        let block_size: u32 = 32;
        let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let sliding_blocks = num_blocks_total;
        let kv_dtype = crate::gemma4_layer_exec::KvDtype::from_env(false);

        let mut kv_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_total_bytes: u64 = 0;
        let mut kv_scale_total_bytes: u64 = 0;
        let mut kv_dtype_per_layer: Vec<crate::gemma4_layer_exec::KvDtype> =
            Vec::with_capacity(arch.num_hidden_layers);
        for l in 0..arch.num_hidden_layers {
            kv_layer_offsets.push(kv_total_bytes);
            kv_scale_layer_offsets.push(kv_scale_total_bytes);
            let is_global = arch.layer_types[l]
                == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
            let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
            let nkvh = arch.num_kv_heads_for_layer(l) as u32;
            let hd = arch.head_dim_for_layer(l) as u32;
            let layer_elems =
                2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
            let kv_dtype_l = crate::gemma4_layer_exec::KvDtype::for_layer_index_or_env(
                arch.layer_types[l], l, false);
            kv_dtype_per_layer.push(kv_dtype_l);
            kv_total_bytes += match kv_dtype_l {
                crate::gemma4_layer_exec::KvDtype::F16 => layer_elems * 2,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_elems,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 2,
            };
            let layer_scale_slots =
                2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64;
            kv_scale_total_bytes += match kv_dtype_l {
                crate::gemma4_layer_exec::KvDtype::F16 => 0,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_scale_slots * 4,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 16,
            };
        }

        let kv_region = self.arena.region("persistent_kv", kv_total_bytes as usize, 256)?;
        let kv_cache_ptr = kv_region.device_ptr();
        let kv_scale_bytes_alloc = kv_scale_total_bytes.max(16) as usize;
        let kv_scale_region =
            self.arena.region("persistent_kv_scale", kv_scale_bytes_alloc, 16)?;
        let kv_scale_ptr = kv_scale_region.device_ptr();

        // `Region` isn't `Drop`, so dropping the wrappers here just
        // discards the handles — the arena's bump pointer has already
        // advanced past both regions, so their bytes stay reserved
        // for the lifetime of `self.arena`. The cuda worker takes its
        // `scratch_ck` checkpoint AFTER this call, so its later
        // `arena.restore(scratch_ck)` cannot rewind past these bytes.

        // Cycle 56 step 4 (bug-audit finding #2 — HIGH): check
        // CUresult on prefix-cache init memset. Failure here corrupts
        // ALL subsequent decode requests for this service lifetime;
        // surfacing it as a typed RvllmError lets the operator see
        // the OOM / ECC at startup instead of silent garbage decode
        // hours later.
        #[cfg(feature = "cuda")]
        unsafe {
            cuda_check!(
                cudarc::driver::sys::cuMemsetD8_v2(kv_cache_ptr, 0, kv_total_bytes as usize),
                "init_prefix_cache_kv_zero", 0u64);
            cuda_check!(
                cudarc::driver::sys::cuMemsetD8_v2(kv_scale_ptr, 0, kv_scale_bytes_alloc),
                "init_prefix_cache_kv_scale_zero", 0u64);
        }

        *guard = Some(PrefixCacheState {
            last_tokens: Vec::new(),
            kv_cache_ptr,
            kv_cache_bytes: kv_total_bytes,
            kv_scale_ptr,
            kv_scale_bytes: kv_scale_total_bytes,
            kv_dtype,
            kv_layer_offsets,
            kv_scale_layer_offsets,
            num_blocks_total,
            block_size,
            committed_prefix_len: 0,
            provenance: PrefixProvenance::from_env(),
        });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub unsafe fn run_bench(
        &self,
        num_seqs: u32,
        iters: u32,
        warmup: u32,
    ) -> crate::bring_up::BenchResult {
        use crate::gemma4_layer_exec::*;
        use rvllm_loader::gemma4_arch::Gemma4LayerType;

        let _f16_only = false; // bench path always FP8 (documentation only; unused)
        let arch = &self.arch;
        let hidden = arch.hidden_size as u32;
        let max_hd = arch.max_head_dim() as u32;
        let max_nkvh = arch.max_kv_heads() as u32;
        let max_q_dim = (arch.num_attention_heads * arch.max_head_dim()) as u32;
        let max_kv_dim = (max_nkvh * max_hd) as u32;
        let max_qkv_rows = max_q_dim + 2 * max_kv_dim;
        let inter = arch.intermediate_size as u32;
        let vocab = arch.vocab_size as u32;
        let stream = self.stream.raw();

        let block_size: u32 = std::env::var("RVLLM_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        let num_blocks_total: u32 = 1024;
        let max_blocks_per_seq = (num_blocks_total / num_seqs).max(1);

        let arena = &self.arena;
        let hidden_fp8 = arena
            .region("hidden_fp8", (num_seqs * hidden) as usize, 16)
            .unwrap();
        let hidden_scale = arena
            .region("hidden_scale", (num_seqs * 4) as usize, 16)
            .unwrap();
        let qkv_out = arena
            .region("qkv_out", (num_seqs * max_qkv_rows * 2) as usize, 16)
            .unwrap();
        let q_base = qkv_out.device_ptr();
        let q_normed = arena
            .region("q_normed", (num_seqs * max_q_dim * 2) as usize, 16)
            .unwrap();
        let k_normed = arena
            .region("k_normed", (num_seqs * max_kv_dim * 2) as usize, 16)
            .unwrap();
        let v_normed = arena
            .region("v_normed", (num_seqs * max_kv_dim * 2) as usize, 16)
            .unwrap();
        let q_fp8 = arena
            .region("q_fp8", (num_seqs * max_q_dim) as usize, 16)
            .unwrap();
        let attn_out = arena
            .region("attn_out", (num_seqs * max_q_dim * 2) as usize, 16)
            .unwrap();
        let attn_out_fp8 = arena
            .region("attn_out_fp8", (num_seqs * max_q_dim) as usize, 16)
            .unwrap();
        let attn_out_scale = arena
            .region("attn_out_scale", (num_seqs * 4) as usize, 16)
            .unwrap();
        let gate_up_out = arena
            .region("gate_up_out", (num_seqs * 2 * inter * 2) as usize, 16)
            .unwrap();
        let gate_up_fp8 = arena
            .region("gate_up_fp8", (num_seqs * 2 * inter) as usize, 16)
            .unwrap();
        let gate_up_scale = arena
            .region("gate_up_scale", (num_seqs * 4) as usize, 16)
            .unwrap();
        let mlp_out_fp8 = arena
            .region("mlp_out_fp8", (num_seqs * inter) as usize, 16)
            .unwrap();
        let mlp_out_scale = arena
            .region("mlp_out_scale", (num_seqs * 4) as usize, 16)
            .unwrap();
        let delta_f16 = arena
            .region("delta_f16", (num_seqs * hidden * 2) as usize, 16)
            .unwrap();
        let gemm_f32_max_n = std::cmp::max(max_qkv_rows, 2 * inter);
        let gemm_f32_tmp = arena
            .region("gemm_f32_tmp", (num_seqs * gemm_f32_max_n * 4) as usize, 16)
            .unwrap();

        // Bench path KV dtype — Codex44-3: matches run_generate's
        // per-layer dispatch via `KvDtype::for_layer_index_or_env(l)`
        // so hybrid configs (`RVLLM_NVFP4_HYBRID_GLOBAL_FP8`,
        // `RVLLM_NVFP4_HYBRID_SLIDING_FP8`, `RVLLM_FP8_KV_LAYERS`)
        // measure the same dispatch shape the live server runs.
        // The earlier uniform `from_env(false)` made bench numbers
        // diverge silently from prod whenever a per-layer override
        // was active — wrong target for tuning. Per-layer sizing +
        // dispatch reads `kv_dtype_per_layer[layer_idx]` everywhere.
        let mut kv_dtype_per_layer: Vec<crate::gemma4_layer_exec::KvDtype> =
            Vec::with_capacity(arch.num_hidden_layers);
        // aa01001pftrope0 cliff-fix (F-series): sliding layers need
        // `slot_mapping[t] < sliding_blocks * block_size` at every rope
        // write. The old cap `sliding_window/block_size` broke at
        // prompt_len > sliding_window because slot_mapping is linear
        // 0..prompt_len-1 and ran off the end. Give sliding layers the
        // full pool — ~10 GiB extra at num_blocks_total=1024, fits in
        // the 50+ GiB arena.
        let sliding_blocks = num_blocks_total;

        let mut kv_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_total_bytes: u64 = 0;
        let mut kv_scale_total_bytes: u64 = 0;
        for l in 0..arch.num_hidden_layers {
            kv_layer_offsets.push(kv_total_bytes);
            kv_scale_layer_offsets.push(kv_scale_total_bytes);
            let layer_blocks = if arch.layer_types[l] == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention { num_blocks_total } else { sliding_blocks };
            let nkvh_l = arch.num_kv_heads_for_layer(l) as u32;
            let hd_l = arch.head_dim_for_layer(l) as u32;
            let layer_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh_l as u64 * hd_l as u64;
            // Codex44-3: per-layer dtype matches run_generate (line 1373).
            let kv_dtype_l = crate::gemma4_layer_exec::KvDtype::for_layer_index_or_env(
                arch.layer_types[l], l, false);
            kv_dtype_per_layer.push(kv_dtype_l);
            kv_total_bytes += match kv_dtype_l {
                crate::gemma4_layer_exec::KvDtype::F16 => layer_elems * 2,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_elems,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 2, // 2 elems/byte
            };
            // FP8 path: per-slot f32 K/V scales (F-series — one scale per
            // (block, tok, head) for both K and V). NVFP4 path: one E4M3
            // scale per 16 elems. F16 self-scaled so 0 bytes.
            let layer_scale_slots =
                2u64 * layer_blocks as u64 * block_size as u64 * nkvh_l as u64;
            kv_scale_total_bytes += match kv_dtype_l {
                crate::gemma4_layer_exec::KvDtype::F16 => 0,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_scale_slots * 4,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 16,
            };
        }
        let kv_cache = arena.region("kv_cache", kv_total_bytes as usize, 256).unwrap();
        let kv_scale_cache = arena.region(
            "kv_scale_cache", kv_scale_total_bytes as usize, 16).unwrap();
        // Per-(token, head) Q scale scratch, written fresh by rope each
        // forward and consumed by the same step's attention.
        // Codex21: rope writes `[token_idx, head]` for token_idx in
        // 0..num_tokens. Bench dispatches every layer with
        // `num_tokens: num_seqs` (single-token decode), so the alloc
        // upper-bound is num_seqs. If bench is ever extended to a
        // multi-token-per-seq prefill path, `max_tokens_per_step` must
        // grow accordingly or rope OOB-writes q_scale_cache.
        let max_tokens_per_step: u32 = num_seqs;
        let q_scale_scratch_bytes =
            (max_tokens_per_step as u64) * (arch.num_attention_heads as u64) * 4;
        let q_scale_scratch = arena.region(
            "q_scale_scratch", q_scale_scratch_bytes as usize, 16).unwrap();
        // Opt-out for A/B testing: RVLLM_PER_TOKEN_Q_SCALE=0 falls back to
        // the scalar q_scale_ptr (pre-c69f641 behaviour) so PPL can be
        // compared across the two calibration strategies without a rebuild.
        let q_scale_cache_ptr: u64 =
            if per_token_q_scale_enabled(/*default_on=*/true) {
                q_scale_scratch.device_ptr()
            } else {
                0
            };
        // Codex26-4: previously the bench-init memsets dropped their
        // CUresult. On OOM / ECC / invalid-pointer the path kept
        // running with uninitialised KV / scale state and produced
        // misleading bench numbers. run_bench returns BenchResult
        // (no Result<>), so the cuda_check! macro can't be used
        // directly; panic with context so bench failures surface
        // immediately instead of contaminating the published numbers.
        #[cfg(feature = "cuda")]
        {
            let memset_check = |r: cudarc::driver::sys::CUresult, what: &str| {
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    panic!("bench: cuMemsetD8_v2 {what} failed: {:?}", r);
                }
            };
            memset_check(
                cudarc::driver::sys::cuMemsetD8_v2(
                    kv_cache.device_ptr(), 0, kv_total_bytes as usize),
                "kv_cache");
            memset_check(
                cudarc::driver::sys::cuMemsetD8_v2(
                    kv_scale_cache.device_ptr(), 0, kv_scale_total_bytes as usize),
                "kv_scale_cache");
            memset_check(
                cudarc::driver::sys::cuMemsetD8_v2(
                    q_scale_scratch.device_ptr(), 0, q_scale_scratch_bytes as usize),
                "q_scale_scratch");
        }
        // Codex20-3: previously bench allocated a second NVFP4 scale
        // region ("kv_cache_scale") of the same size as kv_scale_cache
        // and held it RAII for the bench duration even though no
        // launcher pointed at it. That was redundant — kv_scale_cache
        // above already covers all NVFP4 scale slots; the duplicate
        // just ate VRAM and pushed long bench runs closer to OOM.
        // Removed. F16 / FP8 paths leave kv_scale_cache zero-sized
        // (kv_scale_total_bytes==0 on F16); kernels that don't read
        // scales pass 0 already.

        let q_scale_region = arena.region("q_scale", 4, 4).unwrap();
        let kv_scale_region = arena.region("kv_scale", 4, 4).unwrap();
        {
            let q_s = parse_f32_env_or_default("RVLLM_Q_SCALE", DEFAULT_Q_SCALE);
            let kv_s = parse_f32_env_or_default("RVLLM_KV_SCALE", DEFAULT_KV_SCALE);
            q_scale_region.copy_from_host(&q_s.to_le_bytes()).unwrap();
            kv_scale_region.copy_from_host(&kv_s.to_le_bytes()).unwrap();
        }

        const FA3_WS_BYTES: usize = 16 * 1024 * 1024;
        let fa3_ws = arena.region("fa3_ws", FA3_WS_BYTES, 256).unwrap();
        let residual = arena
            .region("residual", (num_seqs * hidden * 2) as usize, 16)
            .unwrap();
        cudarc::driver::sys::cuMemsetD8_v2(
            residual.device_ptr(),
            0,
            (num_seqs * hidden * 2) as usize,
        );

        let positions = arena
            .region("positions", (num_seqs * 4) as usize, 16)
            .unwrap();
        let slot_mapping = arena
            .region("slot_mapping", (num_seqs * 4) as usize, 16)
            .unwrap();
        let context_lens = arena
            .region("context_lens", (num_seqs * 4) as usize, 16)
            .unwrap();
        let block_tables = arena
            .region(
                "block_tables",
                (num_seqs * max_blocks_per_seq * 4) as usize,
                16,
            )
            .unwrap();
        {
            let n = num_seqs as usize;
            let pos: Vec<i32> = (0..n as i32).collect();
            let slot: Vec<i32> = (0..n as i32).collect();
            let ctx: Vec<i32> = vec![1; n];
            let mut bt: Vec<i32> = Vec::with_capacity(n * max_blocks_per_seq as usize);
            for i in 0..n as i32 {
                for b in 0..max_blocks_per_seq as i32 {
                    bt.push(i * max_blocks_per_seq as i32 + b);
                }
            }
            positions.copy_from_host(bytemuck_cast_i32(&pos)).unwrap();
            slot_mapping
                .copy_from_host(bytemuck_cast_i32(&slot))
                .unwrap();
            context_lens
                .copy_from_host(bytemuck_cast_i32(&ctx))
                .unwrap();
            block_tables.copy_from_host(bytemuck_cast_i32(&bt)).unwrap();
        }

        let logits = arena
            .region("logits", (num_seqs * vocab * 2) as usize, 16)
            .unwrap();
        let sampled_tokens = arena
            .region("sampled_tokens", (num_seqs * 4) as usize, 16)
            .unwrap();
        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena
            .region("cutlass_ws_gemma4", cutlass_ws_bytes, 256)
            .unwrap();
        let residual_ptr = residual.device_ptr();
        // Codex26-2: bench is dev-only and has no Result return path,
        // so unwrap with a clear panic message.
        let kernels = self.layer_kernels()
            .expect("bench: layer_kernels — partial PTX/manifest state, rebuild kernels/");

        // GEMM plans — uniform shapes across all layers (the sliding/global
        // distinction is a runtime head reshape, weight dims are identical).
        // Use the sliding-layer dims for the plan since those are the common case.
        let q_dim_s = (arch.num_attention_heads * arch.head_dim_sliding) as u32;
        let kv_dim_s = (arch.num_kv_heads_sliding * arch.head_dim_sliding) as u32;
        let qkv_rows_s = q_dim_s + 2 * kv_dim_s;
        use rvllm_cutlass::Fp8GemmPlan;
        let _gemm_plans = Gemma4GemmPlans {
            qkv: Fp8GemmPlan::from_policy(
                &self.policy,
                num_seqs,
                qkv_rows_s,
                hidden,
                rvllm_core::DType::Fp8E4M3,
            )
            .unwrap(),
            o: Fp8GemmPlan::from_policy_residual(
                &self.policy,
                num_seqs,
                hidden,
                q_dim_s,
                rvllm_core::DType::Fp8E4M3,
            )
            .unwrap(),
            gate_up: Fp8GemmPlan::from_policy(
                &self.policy,
                num_seqs,
                2 * inter,
                hidden,
                rvllm_core::DType::Fp8E4M3,
            )
            .unwrap(),
            down: Fp8GemmPlan::from_policy_residual(
                &self.policy,
                num_seqs,
                hidden,
                inter,
                rvllm_core::DType::Fp8E4M3,
            )
            .unwrap(),
        };

        // E4B PLE state for the bench dispatch site. Currently
        // unwired (run_bench has no fn_embed parameter nor embedding
        // gather — synthetic input residual). Left at (0, 0) so the
        // dims branch below behaves like 31B (PLE inactive). When
        // bench grows real embed integration this becomes the same
        // per-iteration Cell pattern run_ppl uses.
        let ple_state: std::cell::Cell<(u64, u32)> = std::cell::Cell::new((0u64, 0u32));
        let one_step = || -> rvllm_core::Result<()> {
            let (ple_base, ple_stride_elems) = ple_state.get();
            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                let lt = arch.layer_types[layer_idx];
                let hd = arch.head_dim_for_layer(layer_idx) as u32;
                let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                let q_dim = (arch.num_attention_heads as u32) * hd;
                let kv_dim = nkvh * hd;
                let _qkv_rows = q_dim + 2 * kv_dim;
                let layer_blocks = if lt == Gemma4LayerType::GlobalAttention { num_blocks_total } else { sliding_blocks };

                let dims = Gemma4LayerDims {
                    num_tokens: num_seqs,
                    hidden,
                    num_heads: arch.num_attention_heads as u32,
                    num_kv_heads: nkvh,
                    head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    intermediate: inter,
                    ple_dim: arch.hidden_size_per_layer_input.unwrap_or(0) as u32,
                    block_size,
                    max_blocks_per_seq: layer_blocks,
                    num_blocks_total: layer_blocks,
                    attn_scale: 1.0,
                    rms_eps: arch.rms_norm_eps,
                    layer_type: lt,
                    sliding_window: arch.sliding_window_size as u32,
                    // Codex44-3: per-layer dtype, matches run_generate.
                    f16_kv: kv_dtype_per_layer[layer_idx].is_f16(),
                    kv_dtype: kv_dtype_per_layer[layer_idx],
                    bf16_residual: bf16_residual_enabled(),
                    kv_share_source_layer: arch.kv_share_source_layer(layer_idx).map(|s| s as u32),
                    current_max_context_len: None,
                };

                // Row-major [num_tokens, q_dim+2*kv_dim]: k_out / v_out
                // point at row 0's K / V sub-slice. The rmsnorm kernel
                // applies `src_row_stride` to reach later tokens.
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let is_global = lt == Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                let layer_kv_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                // Codex44-3: per-layer dtype determines per-layer KV bytes.
                let layer_kv_dtype = kv_dtype_per_layer[layer_idx];
                let kv_layer_bytes = match layer_kv_dtype {
                    crate::gemma4_layer_exec::KvDtype::F16 => layer_kv_elems * 2,
                    crate::gemma4_layer_exec::KvDtype::Fp8 => layer_kv_elems,
                    crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_kv_elems / 2,
                };
                // E4B kv-share: when this layer aliases an earlier source
                // (Gemma 4 num_kv_shared_layers tail), the attention
                // launchers must read K/V from the SOURCE layer's region.
                // Pointing layer_kv_base at the source while passing
                // dims.kv_share_source_layer=Some(_) suppresses rope's
                // K/V writes (see fused_rope_partial_*kv.cu nullptr guard)
                // so the source layer's K/V cache is never clobbered.
                let kv_idx = arch.kv_share_source_layer(layer_idx).unwrap_or(layer_idx);
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[kv_idx];
                // FP8 path (F-series): per-slot f32 K/V scales, always
                // allocated in `kv_scale_cache` (kv_scale_total_bytes=0 on F16).
                // kv-share-aware (mirrors layer_kv_base above): for shared
                // layers, the scale arena base also maps to the source layer.
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[kv_idx];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);
                // NVFP4 path: K gets the first half of the layer's scale
                // slab, V the second half. layer_kv_elems covers K+V so
                // each half is `layer_kv_elems / 32` bytes (`/2/16`).
                let (k_cache_scale, v_cache_scale) = if layer_kv_dtype
                    == crate::gemma4_layer_exec::KvDtype::Nvfp4
                {
                    let k_scale_bytes = layer_kv_elems / 32;
                    (layer_kv_scale_base, layer_kv_scale_base + k_scale_bytes)
                } else {
                    (0u64, 0u64)
                };

                let (cos, sin) = match lt {
                    Gemma4LayerType::SlidingAttention => (
                        self.model.rope_cos_sliding.offset_bytes,
                        self.model.rope_sin_sliding.offset_bytes,
                    ),
                    Gemma4LayerType::GlobalAttention => (
                        self.model.rope_cos_global.offset_bytes,
                        self.model.rope_sin_global.offset_bytes,
                    ),
                };

                let w = Gemma4LayerWeightPtrs {
                    attn_norm_gamma: layer.input_layernorm.offset_bytes,
                    post_attn_norm_gamma: layer.post_attention_layernorm.offset_bytes,
                    pre_ff_norm_gamma: layer.pre_feedforward_layernorm.offset_bytes,
                    post_ff_norm_gamma: layer.post_feedforward_layernorm.offset_bytes,
                    q_norm_gamma: layer.q_norm.offset_bytes,
                    k_norm_gamma: layer.k_norm.offset_bytes,
                    qkv_fp8: layer.qkv.as_ref().map_or(0, |w| w.offset_bytes),
                    qkv_scale: layer.qkv.as_ref().map_or(0, |w| w.scale_ptr),
                    o_fp8: layer.o_proj.as_ref().map_or(0, |w| w.offset_bytes),
                    o_scale: layer.o_proj.as_ref().map_or(0, |w| w.scale_ptr),
                    gate_up_fp8: layer.gate_up.as_ref().map_or(0, |w| w.offset_bytes),
                    gate_up_scale: layer.gate_up.as_ref().map_or(0, |w| w.scale_ptr),
                    down_fp8: layer.down_proj.as_ref().map_or(0, |w| w.offset_bytes),
                    down_scale: layer.down_proj.as_ref().map_or(0, |w| w.scale_ptr),
                    layer_scalar_ptr: layer.layer_scalar.offset_bytes,
                    qkv_f16: layer.qkv_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    o_f16: layer.o_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    gate_up_f16: layer.gate_up_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    down_f16: layer.down_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    qkv_chscale: layer.qkv.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    o_chscale: layer.o_proj.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    gate_up_chscale: layer.gate_up.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    down_chscale: layer.down_proj.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    qkv_blockscale: layer.qkv.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    o_blockscale: layer.o_proj.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    gate_up_blockscale: layer.gate_up.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    down_blockscale: layer.down_proj.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    awq: awq_layer_ptrs(layer.awq.as_ref()),
                    // E4B PLE plumbing. Pointers from the loaded per-layer
                    // tensors (`None` → 0 on 31B/AWQ). `ple_per_layer_input`
                    // is populated by the per-request PLE precompute (Stage
                    // 3b); 0 here means PLE inactive for this forward
                    // dispatch — layer_exec then skips the PLE injection.
                    ple_input_gate: layer
                        .per_layer_input_gate
                        .as_ref()
                        .map_or(0, |w| w.offset_bytes),
                    ple_projection: layer
                        .per_layer_projection
                        .as_ref()
                        .map_or(0, |w| w.offset_bytes),
                    ple_post_input_norm_gamma: layer
                        .post_per_layer_input_norm
                        .as_ref()
                        .map_or(0, |w| w.offset_bytes),
                    // E4B PLE: per-layer slice into the per-token
                    // precompute buffer (set by enclosing per-token
                    // driver before one_step). Mirrors the chunked-
                    // prefill and decode-step sites. 0/0 when the
                    // driver hasn't wired PLE (e.g. run_bench's
                    // synthetic-input path) — layer_exec then
                    // short-circuits the PLE injection.
                    ple_per_layer_input: if ple_base != 0 {
                        ple_base + (layer_idx as u64)
                            * (arch.hidden_size_per_layer_input.unwrap_or(0) as u64)
                            * 2
                    } else {
                        0
                    },
                    ple_per_layer_stride_elems: ple_stride_elems,
                };

                let scratch = Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(),
                    hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base,
                    k_out,
                    v_out,
                    q_normed: q_normed.device_ptr(),
                    k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + kv_layer_bytes / 2,
                    k_scale_cache: layer_kv_scale_base,
                    v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                    q_scale_cache: q_scale_cache_ptr,
                    k_cache_scale,
                    v_cache_scale,
                    q_scale_ptr: q_scale_region.device_ptr(),
                    kv_scale_ptr: kv_scale_region.device_ptr(),
                    attn_out: attn_out.device_ptr(),
                    attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(),
                    delta_f16: delta_f16.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(),
                    gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(),
                    mlp_out_scale: mlp_out_scale.device_ptr(),
                    gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                    gemm_f32_tmp_bytes: (num_seqs * gemm_f32_max_n * 4) as usize,
                    cutlass_workspace: cutlass_ws.device_ptr(),
                    cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                    fa3_workspace_bytes: FA3_WS_BYTES as u64,
                    // NVFP4 shadow diagnostic: default 0 (no shadow).
                    // Overridden in the run_generate decode path when
                    // RVLLM_NVFP4_SHADOW_F16 is on.
                    shadow_k_cache: 0,
                    shadow_v_cache: 0,
                    shadow_q_cache: 0,
                    // === HADAMARD ROTATION ===
                    // Probe / profile paths don't use rotation; the
                    // rope kernel treats nullptr (=0) as "rotation
                    // off" and runs byte-identical to the pre-Hadamard
                    // path. Live decode/prefill paths below compute
                    // the per-layer pointer from
                    // `self.nvfp4_hadamard`.
                    hadamard_signs_q: 0,
                    hadamard_signs_k: 0,
                    // === END HADAMARD ROTATION ===
                };

                let meta = Gemma4MetadataPtrs {
                    positions: positions.device_ptr(),
                    slot_mapping: slot_mapping.device_ptr(),
                    cos,
                    sin,
                    block_tables: block_tables.device_ptr(),
                    context_lens: context_lens.device_ptr(),
                };

                gemma4_forward(
                    dims,
                    &kernels,
                    &w,
                    &scratch,
                    &meta,
                    &self.cublaslt,
                    &self.cutlass,
                    &self.sliding_attention,
                    &self.global_attention,
                    residual_ptr,
                    stream,
                )?;
            }

            // LM head: final norm + FP8 quant + GEMM + softcap + argmax
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: num_seqs,
                hidden,
                eps: arch.rms_norm_eps,
            }
            .launch(
                kernels.fused_rmsnorm_fp8_quant,
                hidden_fp8.device_ptr(),
                hidden_scale.device_ptr(),
                residual_ptr,
                self.model.final_norm.offset_bytes,
                stream,
            )?;
            self.cublaslt.fp8_gemm(
                hidden_fp8.device_ptr(),
                self.model.lm_head_fp8.offset_bytes,
                logits.device_ptr(),
                num_seqs as i32,
                vocab as i32,
                hidden as i32,
                hidden_scale.device_ptr(),
                self.model.lm_head_fp8.scale_ptr,
                stream,
            )?;
            logit_softcap(
                self.fused.fn_softcap,
                logits.device_ptr(),
                num_seqs,
                vocab,
                arch.logit_softcap,
                stream,
            )?;
            rvllm_fused::ArgmaxLaunch {
                num_tokens: num_seqs,
                vocab,
            }
            .launch(
                self.fused.fn_argmax,
                logits.device_ptr(),
                sampled_tokens.device_ptr(),
                stream,
            )?;
            Ok(())
        };

        // Warmup
        for _ in 0..warmup {
            one_step().unwrap();
        }
        self.stream.fence().unwrap();

        // Timed
        let no_graph = std::env::var("RVLLM_NO_GRAPH").ok().as_deref() == Some("1");
        let elapsed = if no_graph {
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                one_step().unwrap();
            }
            self.stream.fence().unwrap();
            t0.elapsed()
        } else {
            let graph = rvllm_graph::CapturedGraph::capture(
                num_seqs,
                max_blocks_per_seq,
                rvllm_metadata::MetadataLayout::compute(num_seqs, max_blocks_per_seq).hash(),
                rvllm_graph::GraphFingerprint([0u8; 32]),
                stream,
                || one_step(),
            ).unwrap();
            self.stream.fence().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                graph.replay(stream).unwrap();
            }
            self.stream.fence().unwrap();
            t0.elapsed()
        };

        crate::bring_up::BenchResult {
            ns_per_step: elapsed.as_nanos() / iters.max(1) as u128,
            total_ns: elapsed.as_nanos(),
            iters,
            num_seqs,
            ttft_ns: None,
            ttft_hot_ns: None,
        }
    }

    #[cfg(feature = "cuda")]
    pub unsafe fn run_ppl(
        &self,
        fn_embed: rvllm_kernels::KernelFn,
        token_ids: &[u32],
    ) -> Result<crate::bring_up::PplResult> {
        use crate::bring_up::{dtoh_async_sync, f16_to_f32};
        use crate::gemma4_layer_exec::*;
        use rvllm_loader::gemma4_arch::Gemma4LayerType;

        let arch = &self.arch;
        let hidden = arch.hidden_size as u32;
        let max_hd = arch.max_head_dim() as u32;
        let max_nkvh = arch.max_kv_heads() as u32;
        let max_q_dim = (arch.num_attention_heads * arch.max_head_dim()) as u32;
        let max_kv_dim = (max_nkvh * max_hd) as u32;
        let max_qkv_rows = max_q_dim + 2 * max_kv_dim;
        let inter = arch.intermediate_size as u32;
        let vocab = arch.vocab_size as u32;
        let stream = self.stream.raw();
        let num_seqs: u32 = 1;

        let max_layers: usize = std::env::var("RVLLM_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(arch.num_hidden_layers);
        let skip_softcap = std::env::var("RVLLM_NO_SOFTCAP").map_or(false, |v| v == "1");
        if max_layers < arch.num_hidden_layers {
            eprintln!(
                "[ppl] RVLLM_MAX_LAYERS={max_layers} (of {})",
                arch.num_hidden_layers
            );
        }
        if skip_softcap {
            eprintln!("[ppl] RVLLM_NO_SOFTCAP=1: softcap disabled");
        }
        eprintln!("[ppl] attn_scale=1.0 (Gemma4 QK-norm, no query_pre_attn_scalar)");

        let block_size: u32 = std::env::var("RVLLM_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let max_blocks_per_seq = (num_blocks_total / num_seqs).max(1);

        let arena = &self.arena;
        let hidden_fp8 = arena.region("hidden_fp8", (num_seqs * hidden) as usize, 16)?;
        let hidden_scale = arena.region("hidden_scale", (num_seqs * 4) as usize, 16)?;
        let qkv_out = arena.region("qkv_out", (num_seqs * max_qkv_rows * 2) as usize, 16)?;
        let q_base = qkv_out.device_ptr();
        let q_normed = arena.region("q_normed", (num_seqs * max_q_dim * 2) as usize, 16)?;
        let k_normed = arena.region("k_normed", (num_seqs * max_kv_dim * 2) as usize, 16)?;
        let v_normed = arena.region("v_normed", (num_seqs * max_kv_dim * 2) as usize, 16)?;
        let q_fp8 = arena.region("q_fp8", (num_seqs * max_q_dim) as usize, 16)?;
        let attn_out = arena.region("attn_out", (num_seqs * max_q_dim * 2) as usize, 16)?;
        let attn_out_fp8 = arena.region("attn_out_fp8", (num_seqs * max_q_dim) as usize, 16)?;
        let attn_out_scale = arena.region("attn_out_scale", (num_seqs * 4) as usize, 16)?;
        let gate_up_out = arena.region("gate_up_out", (num_seqs * 2 * inter * 2) as usize, 16)?;
        let gate_up_fp8 = arena.region("gate_up_fp8", (num_seqs * 2 * inter) as usize, 16)?;
        let gate_up_scale = arena.region("gate_up_scale", (num_seqs * 4) as usize, 16)?;
        let mlp_out_fp8 = arena.region("mlp_out_fp8", (num_seqs * inter) as usize, 16)?;
        let mlp_out_scale = arena.region("mlp_out_scale", (num_seqs * 4) as usize, 16)?;
        let delta_f16 = arena.region("delta_f16_ppl", (num_seqs * hidden * 2) as usize, 16)?;
        let gemm_f32_max_n = std::cmp::max(max_qkv_rows, 2 * inter);
        let gemm_f32_tmp = arena.region(
            "gemm_f32_tmp_ppl",
            (num_seqs * gemm_f32_max_n * 4) as usize,
            16,
        )?;

        let f16_only = std::env::var("RVLLM_F16_ONLY").map_or(false, |v| v == "1");
        // Codex47-1: mirror Codex44-3's per-layer KV-dtype plumbing so
        // run_ppl validates the same hybrid configs (HYBRID_GLOBAL_FP8,
        // HYBRID_SLIDING_FP8, RVLLM_FP8_KV_LAYERS) that run_generate
        // honours. RVLLM_F16_ONLY still forces every layer to F16 via
        // the f16_only flag threaded into for_layer_index_or_env. Used
        // for the per-layer scratch sizing below + the dispatch site
        // (~line 2299) — both consume kv_dtype_per_layer[layer_idx].
        // The single `kv_dtype` snapshot stays for the bytes-per-elem
        // log line (PPL header) — uses from_env baseline as the
        // reporting value.
        let kv_dtype = crate::gemma4_layer_exec::KvDtype::from_env(f16_only);
        let kv_bytes_per_elem_log: u32 = match kv_dtype {
            crate::gemma4_layer_exec::KvDtype::F16 => 2,
            crate::gemma4_layer_exec::KvDtype::Fp8 => 1,
            crate::gemma4_layer_exec::KvDtype::Nvfp4 => 0, // logging placeholder
        };
        let mut kv_dtype_per_layer: Vec<crate::gemma4_layer_exec::KvDtype> =
            Vec::with_capacity(arch.num_hidden_layers);

        // Per-layer KV budget: sliding layers cap at sliding_window/block_size blocks,
        // global layers use full num_blocks_total. Saves ~5x KV memory for long context.
        // aa01001pftrope0 cliff-fix: sliding layers need `slot_mapping[t] < sliding_blocks*block_size`
        // at every t the rope writes; the old cap sliding_blocks = sliding_window/block_size = 32
        // (= 1024 slots for Gemma 4) broke at prompt_len > sliding_window because slot_mapping
        // is linear 0..prompt_len-1 and index 1024+ ran off the end of the sliding KV region.
        // Proper fix is a per-sliding-layer ring buffer (slot = t mod sliding_window) but that
        // needs rope + attention kernel cooperation. For now give sliding layers the full pool —
        // ~10 GiB extra at num_blocks_total=1024, fits in the 50 GiB arena with Gemma 4 31B fp8.
        let sliding_blocks = num_blocks_total;

        let mut kv_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_total_bytes: u64 = 0;
        let mut kv_scale_total_bytes: u64 = 0;
        for l in 0..arch.num_hidden_layers {
            kv_layer_offsets.push(kv_total_bytes);
            kv_scale_layer_offsets.push(kv_scale_total_bytes);
            let is_global = arch.layer_types[l] == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
            let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
            let nkvh = arch.num_kv_heads_for_layer(l) as u32;
            let hd = arch.head_dim_for_layer(l) as u32;
            let layer_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
            // Codex47-1: per-layer dtype matches run_generate / run_bench.
            let kv_dtype_l = crate::gemma4_layer_exec::KvDtype::for_layer_index_or_env(
                arch.layer_types[l], l, f16_only);
            kv_dtype_per_layer.push(kv_dtype_l);
            kv_total_bytes += match kv_dtype_l {
                crate::gemma4_layer_exec::KvDtype::F16 => layer_elems * 2,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_elems,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 2,
            };
            // FP8 path: per-slot f32 K/V scales. NVFP4 path: one E4M3
            // scale per 16 elems. F16: self-scaled (0 bytes).
            let layer_scale_slots =
                2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64;
            kv_scale_total_bytes += match kv_dtype_l {
                crate::gemma4_layer_exec::KvDtype::F16 => 0,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_scale_slots * 4,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 16,
            };
        }
        eprintln!("[ppl] KV cache: {:.1} MB ({:?}, sliding={} blocks, global={} blocks, {} bytes/elem main)",
            kv_total_bytes as f64 / 1e6, kv_dtype, sliding_blocks, num_blocks_total, kv_bytes_per_elem_log);

        let kv_cache = arena.region("kv_cache", kv_total_bytes as usize, 256)?;
        // Codex26-4: wrap PPL memsets with cuda_check (run_ppl returns Result<>).
        cuda_check!(cudarc::driver::sys::cuMemsetD8_v2(
            kv_cache.device_ptr(), 0, kv_total_bytes as usize),
            "ppl_kv_cache_zero", stream);
        // Scale cache shared across FP8 (F-series per-slot f32 scales)
        // and NVFP4 (per-16-elem E4M3 scales). F16 path has 0 scale bytes
        // but we still allocate a placeholder region so region indexing
        // stays uniform.
        let kv_scale_cache =
            arena.region("kv_scale_cache", kv_scale_total_bytes.max(16) as usize, 16)?;
        cuda_check!(cudarc::driver::sys::cuMemsetD8_v2(
            kv_scale_cache.device_ptr(), 0, kv_scale_total_bytes as usize),
            "ppl_kv_scale_cache_zero", stream);
        // Codex21: PPL also dispatches with num_tokens==num_seqs today.
        // If the path grows to multi-token-per-seq prefill,
        // max_tokens_per_step must track the new upper bound; otherwise
        // rope OOB-writes q_scale_cache at `[tok * heads + head]`.
        let max_tokens_per_step: u32 = num_seqs;
        let q_scale_scratch_bytes =
            (max_tokens_per_step as u64) * (arch.num_attention_heads as u64) * 4;
        let q_scale_scratch = arena.region(
            "q_scale_scratch", q_scale_scratch_bytes as usize, 16)?;
        cuda_check!(cudarc::driver::sys::cuMemsetD8_v2(
            q_scale_scratch.device_ptr(), 0, q_scale_scratch_bytes as usize),
            "ppl_q_scale_scratch_zero", stream);
        // See run_bench: RVLLM_PER_TOKEN_Q_SCALE=0 opts out.
        let q_scale_cache_ptr: u64 =
            if per_token_q_scale_enabled(/*default_on=*/true) {
                q_scale_scratch.device_ptr()
            } else {
                0
            };

        let q_scale_region = arena.region("q_scale", 4, 4)?;
        let kv_scale_region = arena.region("kv_scale", 4, 4)?;
        {
            let q_s = parse_f32_env_or_default("RVLLM_Q_SCALE", DEFAULT_Q_SCALE);
            let kv_s = parse_f32_env_or_default("RVLLM_KV_SCALE", DEFAULT_KV_SCALE);
            q_scale_region.copy_from_host(&q_s.to_le_bytes())?;
            kv_scale_region.copy_from_host(&kv_s.to_le_bytes())?;
        }

        const FA3_WS_BYTES: usize = 16 * 1024 * 1024;
        let fa3_ws = arena.region("fa3_ws", FA3_WS_BYTES, 256)?;
        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena.region("cutlass_ws_ppl", cutlass_ws_bytes, 256)?;

        let positions = arena.region("positions", (num_seqs * 4) as usize, 16)?;
        let slot_mapping = arena.region("slot_mapping", (num_seqs * 4) as usize, 16)?;
        let context_lens = arena.region("context_lens", (num_seqs * 4) as usize, 16)?;
        let block_tables = arena.region(
            "block_tables",
            (num_seqs * max_blocks_per_seq * 4) as usize,
            16,
        )?;
        {
            // Codex22-1: NVFP4 attention kernels read
            // `block_tables[seq_idx * max_blocks_per_seq + page_idx]`,
            // so the table must hold `num_seqs * max_blocks_per_seq`
            // entries. The previous init wrote only the first row
            // (max_blocks_per_seq entries); for num_seqs > 1 every
            // sequence past row 0 read uninitialised page IDs →
            // either OOB-access into KV or cross-seq KV aliasing.
            // Tile the page-id sequence per row so each sequence
            // sees its own contiguous block range, mirroring how
            // run_bench / run_generate lay out their tables.
            let total_entries = (num_seqs as usize) * (max_blocks_per_seq as usize);
            let mut bt: Vec<i32> = Vec::with_capacity(total_entries);
            for s in 0..num_seqs as i32 {
                for b in 0..max_blocks_per_seq as i32 {
                    bt.push(s * max_blocks_per_seq as i32 + b);
                }
            }
            block_tables.copy_from_host(bytemuck_cast_i32(&bt))?;
        }

        let residual = arena.region("residual", (num_seqs * hidden * 2) as usize, 16)?;
        let logits = arena.region("logits_ppl", (num_seqs * vocab * 2) as usize, 16)?;
        let logits_f32 = arena.region("logits_f32_ppl", (num_seqs * vocab * 4) as usize, 16)?;
        let token_ids_region = arena.region("token_ids_ppl", (num_seqs * 4) as usize, 16)?;
        let residual_ptr = residual.device_ptr();
        let kernels = self.layer_kernels()?;

        let q_dim_s = (arch.num_attention_heads * arch.head_dim_sliding) as u32;
        let kv_dim_s = (arch.num_kv_heads_sliding * arch.head_dim_sliding) as u32;
        let qkv_rows_s = q_dim_s + 2 * kv_dim_s;
        use rvllm_cutlass::Fp8GemmPlan;
        let _gemm_plans = Gemma4GemmPlans {
            qkv: Fp8GemmPlan::from_policy(
                &self.policy,
                num_seqs,
                qkv_rows_s,
                hidden,
                rvllm_core::DType::Fp8E4M3,
            )?,
            o: Fp8GemmPlan::from_policy_residual(
                &self.policy,
                num_seqs,
                hidden,
                q_dim_s,
                rvllm_core::DType::Fp8E4M3,
            )?,
            gate_up: Fp8GemmPlan::from_policy(
                &self.policy,
                num_seqs,
                2 * inter,
                hidden,
                rvllm_core::DType::Fp8E4M3,
            )?,
            down: Fp8GemmPlan::from_policy_residual(
                &self.policy,
                num_seqs,
                hidden,
                inter,
                rvllm_core::DType::Fp8E4M3,
            )?,
        };

        let step_counter = std::cell::Cell::new(0u32);
        // E4B PLE state: (ple_base, ple_stride_elems) refreshed per token
        // by `ppl_forward` below right after the embedding gather. one_step
        // reads it lazily so every layer's dims pick up the per-token
        // precompute. (0, 0) when PLE inactive (31B path).
        let ple_state: std::cell::Cell<(u64, u32)> = std::cell::Cell::new((0u64, 0u32));
        let one_step = || -> Result<()> {
            let (ple_base, ple_stride_elems) = ple_state.get();
            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                if layer_idx >= max_layers {
                    break;
                }
                let lt = arch.layer_types[layer_idx];
                let hd = arch.head_dim_for_layer(layer_idx) as u32;
                let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                let q_dim = (arch.num_attention_heads as u32) * hd;
                let kv_dim = nkvh * hd;
                let layer_blocks = if lt == Gemma4LayerType::GlobalAttention { num_blocks_total } else { sliding_blocks };

                let dims = Gemma4LayerDims {
                    num_tokens: num_seqs,
                    hidden,
                    num_heads: arch.num_attention_heads as u32,
                    num_kv_heads: nkvh,
                    head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    intermediate: inter,
                    ple_dim: arch.hidden_size_per_layer_input.unwrap_or(0) as u32,
                    block_size,
                    max_blocks_per_seq: layer_blocks,
                    num_blocks_total: layer_blocks,
                    attn_scale: 1.0,
                    rms_eps: arch.rms_norm_eps,
                    layer_type: lt,
                    sliding_window: arch.sliding_window_size as u32,
                    // Codex47-1: per-layer dtype matches run_generate.
                    f16_kv: kv_dtype_per_layer[layer_idx].is_f16(),
                    kv_dtype: kv_dtype_per_layer[layer_idx],
                    bf16_residual: bf16_residual_enabled(),
                    kv_share_source_layer: arch.kv_share_source_layer(layer_idx).map(|s| s as u32),
                    current_max_context_len: None,
                };

                // Row-major [num_tokens, q_dim+2*kv_dim]: k_out / v_out
                // point at row 0's K / V sub-slice. The rmsnorm kernel
                // applies `src_row_stride` to reach later tokens.
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let is_global = lt == Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                let layer_kv_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                // Codex47-1: per-layer dtype determines per-layer KV bytes.
                let layer_kv_dtype = kv_dtype_per_layer[layer_idx];
                let kv_layer_bytes = match layer_kv_dtype {
                    crate::gemma4_layer_exec::KvDtype::F16 => layer_kv_elems * 2,
                    crate::gemma4_layer_exec::KvDtype::Fp8 => layer_kv_elems,
                    crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_kv_elems / 2,
                };
                // E4B kv-share: when this layer aliases an earlier source
                // (Gemma 4 num_kv_shared_layers tail), the attention
                // launchers must read K/V from the SOURCE layer's region.
                // Pointing layer_kv_base at the source while passing
                // dims.kv_share_source_layer=Some(_) suppresses rope's
                // K/V writes (see fused_rope_partial_*kv.cu nullptr guard)
                // so the source layer's K/V cache is never clobbered.
                let kv_idx = arch.kv_share_source_layer(layer_idx).unwrap_or(layer_idx);
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[kv_idx];
                // kv-share-aware (mirrors layer_kv_base above): for shared
                // layers, the scale arena base also maps to the source layer.
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[kv_idx];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);
                let (k_cache_scale, v_cache_scale) = if layer_kv_dtype
                    == crate::gemma4_layer_exec::KvDtype::Nvfp4
                {
                    (layer_kv_scale_base, layer_kv_scale_base + layer_kv_elems / 32)
                } else {
                    (0u64, 0u64)
                };
                let (cos, sin) = match lt {
                    Gemma4LayerType::SlidingAttention => (
                        self.model.rope_cos_sliding.offset_bytes,
                        self.model.rope_sin_sliding.offset_bytes,
                    ),
                    Gemma4LayerType::GlobalAttention => (
                        self.model.rope_cos_global.offset_bytes,
                        self.model.rope_sin_global.offset_bytes,
                    ),
                };

                let w = Gemma4LayerWeightPtrs {
                    attn_norm_gamma: layer.input_layernorm.offset_bytes,
                    post_attn_norm_gamma: layer.post_attention_layernorm.offset_bytes,
                    pre_ff_norm_gamma: layer.pre_feedforward_layernorm.offset_bytes,
                    post_ff_norm_gamma: layer.post_feedforward_layernorm.offset_bytes,
                    q_norm_gamma: layer.q_norm.offset_bytes,
                    k_norm_gamma: layer.k_norm.offset_bytes,
                    qkv_fp8: layer.qkv.as_ref().map_or(0, |w| w.offset_bytes),
                    qkv_scale: layer.qkv.as_ref().map_or(0, |w| w.scale_ptr),
                    o_fp8: layer.o_proj.as_ref().map_or(0, |w| w.offset_bytes),
                    o_scale: layer.o_proj.as_ref().map_or(0, |w| w.scale_ptr),
                    gate_up_fp8: layer.gate_up.as_ref().map_or(0, |w| w.offset_bytes),
                    gate_up_scale: layer.gate_up.as_ref().map_or(0, |w| w.scale_ptr),
                    down_fp8: layer.down_proj.as_ref().map_or(0, |w| w.offset_bytes),
                    down_scale: layer.down_proj.as_ref().map_or(0, |w| w.scale_ptr),
                    layer_scalar_ptr: layer.layer_scalar.offset_bytes,
                    qkv_f16: layer.qkv_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    o_f16: layer.o_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    gate_up_f16: layer.gate_up_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    down_f16: layer.down_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    qkv_chscale: layer.qkv.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    o_chscale: layer.o_proj.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    gate_up_chscale: layer.gate_up.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    down_chscale: layer.down_proj.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    qkv_blockscale: layer.qkv.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    o_blockscale: layer.o_proj.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    gate_up_blockscale: layer.gate_up.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    down_blockscale: layer.down_proj.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    awq: awq_layer_ptrs(layer.awq.as_ref()),
                    // E4B PLE plumbing. Pointers from the loaded per-layer
                    // tensors (`None` → 0 on 31B/AWQ). `ple_per_layer_input`
                    // is populated by the per-request PLE precompute (Stage
                    // 3b); 0 here means PLE inactive for this forward
                    // dispatch — layer_exec then skips the PLE injection.
                    ple_input_gate: layer
                        .per_layer_input_gate
                        .as_ref()
                        .map_or(0, |w| w.offset_bytes),
                    ple_projection: layer
                        .per_layer_projection
                        .as_ref()
                        .map_or(0, |w| w.offset_bytes),
                    ple_post_input_norm_gamma: layer
                        .post_per_layer_input_norm
                        .as_ref()
                        .map_or(0, |w| w.offset_bytes),
                    // E4B PLE: per-layer slice into the per-token
                    // precompute buffer (set by enclosing per-token
                    // driver before one_step). Mirrors the chunked-
                    // prefill and decode-step sites. 0/0 when the
                    // driver hasn't wired PLE (e.g. run_bench's
                    // synthetic-input path) — layer_exec then
                    // short-circuits the PLE injection.
                    ple_per_layer_input: if ple_base != 0 {
                        ple_base + (layer_idx as u64)
                            * (arch.hidden_size_per_layer_input.unwrap_or(0) as u64)
                            * 2
                    } else {
                        0
                    },
                    ple_per_layer_stride_elems: ple_stride_elems,
                };

                let scratch = Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(),
                    hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base,
                    k_out,
                    v_out,
                    q_normed: q_normed.device_ptr(),
                    k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + kv_layer_bytes / 2,
                    k_scale_cache: layer_kv_scale_base,
                    v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                    q_scale_cache: q_scale_cache_ptr,
                    k_cache_scale,
                    v_cache_scale,
                    q_scale_ptr: q_scale_region.device_ptr(),
                    kv_scale_ptr: kv_scale_region.device_ptr(),
                    attn_out: attn_out.device_ptr(),
                    attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(),
                    delta_f16: delta_f16.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(),
                    gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(),
                    mlp_out_scale: mlp_out_scale.device_ptr(),
                    gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                    gemm_f32_tmp_bytes: (num_seqs * gemm_f32_max_n * 4) as usize,
                    cutlass_workspace: cutlass_ws.device_ptr(),
                    cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                    fa3_workspace_bytes: FA3_WS_BYTES as u64,
                    // NVFP4 shadow diagnostic: default 0 (no shadow).
                    // Overridden in the run_generate decode path when
                    // RVLLM_NVFP4_SHADOW_F16 is on.
                    shadow_k_cache: 0,
                    shadow_v_cache: 0,
                    shadow_q_cache: 0,
                    // === HADAMARD ROTATION ===
                    // Probe / profile paths don't use rotation; the
                    // rope kernel treats nullptr (=0) as "rotation
                    // off" and runs byte-identical to the pre-Hadamard
                    // path. Live decode/prefill paths below compute
                    // the per-layer pointer from
                    // `self.nvfp4_hadamard`.
                    hadamard_signs_q: 0,
                    hadamard_signs_k: 0,
                    // === END HADAMARD ROTATION ===
                };

                let meta = Gemma4MetadataPtrs {
                    positions: positions.device_ptr(),
                    slot_mapping: slot_mapping.device_ptr(),
                    cos,
                    sin,
                    block_tables: block_tables.device_ptr(),
                    context_lens: context_lens.device_ptr(),
                };

                gemma4_forward(
                    dims,
                    &kernels,
                    &w,
                    &scratch,
                    &meta,
                    &self.cublaslt,
                    &self.cutlass,
                    &self.sliding_attention,
                    &self.global_attention,
                    residual_ptr,
                    stream,
                )?;

                if step_counter.get() == 0 && layer_idx == 0 {
                    cudarc::driver::sys::cuStreamSynchronize(stream as _);
                    let mut s = [0u16; 4];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, residual_ptr, 8);
                    let v: Vec<f32> = s.iter().map(|&x| f16_to_f32(x)).collect();
                    let mut amax = 0f32;
                    let n = hidden as usize;
                    let mut all = vec![0u16; n];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        all.as_mut_ptr() as *mut _,
                        residual_ptr,
                        (n * 2) as _,
                    );
                    for &b in &all {
                        let f = f16_to_f32(b).abs();
                        if f > amax && !f.is_nan() {
                            amax = f;
                        }
                    }
                    eprintln!("  [ppl L0] residual first4={:.6?} amax={:.6}", v, amax);
                    // Check layer_scalar value
                    let mut sc = [0u16; 1];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        sc.as_mut_ptr() as *mut _,
                        layer.layer_scalar.offset_bytes,
                        2,
                    );
                    eprintln!("  [ppl L0] layer_scalar={:.6}", f16_to_f32(sc[0]));
                    // Check norm gamma amax
                    let mut ng = vec![0u16; n];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        ng.as_mut_ptr() as *mut _,
                        layer.input_layernorm.offset_bytes,
                        (n * 2) as _,
                    );
                    let gamma_amax = ng.iter().map(|&b| f16_to_f32(b).abs()).fold(0f32, f32::max);
                    eprintln!("  [ppl L0] input_norm_gamma amax={:.6}", gamma_amax);
                }
                if step_counter.get() == 0 && layer_idx < 3 && layer_idx > 0 {
                    cudarc::driver::sys::cuStreamSynchronize(stream as _);
                    let mut s = [0u16; 4];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, residual_ptr, 8);
                    let v: Vec<f32> = s.iter().map(|&x| f16_to_f32(x)).collect();
                    eprintln!("  [ppl L{}] residual={:.4?}", layer_idx, v);
                }
            }

            // LM head: final norm (f16 in-place) + f16 GEMM -> f32 logits
            let dbg_lmhead = step_counter.get() == 0 && std::env::var("RVLLM_DBG_LAYER").is_ok();

            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_seqs,
                hidden,
                eps: arch.rms_norm_eps,
            }
            .launch(
                kernels.fused_rmsnorm,
                residual_ptr,
                self.model.final_norm.offset_bytes,
                stream,
            )?;
            if dbg_lmhead {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut s = [0u16; 4];
                cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, residual_ptr, 8);
                let v: Vec<f32> = s.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
                eprintln!("  [lm_head] after rmsnorm_f16: first4={:.4?}", v);
            }
            self.cublaslt.f16_gemm_f32(
                residual_ptr,
                self.model.lm_head_f16.offset_bytes,
                logits_f32.device_ptr(),
                num_seqs as i32,
                vocab as i32,
                hidden as i32,
                stream,
            )?;
            if dbg_lmhead {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let total = (vocab as usize) * (num_seqs as usize);
                let mut buf = vec![0.0f32; total];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut _,
                    logits_f32.device_ptr(),
                    (total * 4) as _,
                );
                let amax = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!(
                    "  [lm_head] raw_f32_logits first8={:.4?} amax={:.6e} (n={})",
                    &buf[..8.min(total)],
                    amax,
                    total
                );
            }
            rvllm_fused::gemma4_launcher::Bf16ToF16SatLaunch {
                n: num_seqs * vocab,
            }
            .launch(
                kernels.f32_to_f16_sat,
                logits.device_ptr(),
                logits_f32.device_ptr(),
                stream,
            )?;
            if dbg_lmhead {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut s = [0u16; 4];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    s.as_mut_ptr() as *mut _,
                    logits.device_ptr(),
                    8,
                );
                let v: Vec<f32> = s.iter().map(|&x| f16_to_f32(x)).collect();
                eprintln!(
                    "  [lm_head] after f32_to_f16_sat: logits_f16 first4={:.4?}",
                    v
                );
            }
            if !skip_softcap {
                logit_softcap(
                    self.fused.fn_softcap,
                    logits.device_ptr(),
                    num_seqs,
                    vocab,
                    arch.logit_softcap,
                    stream,
                )?;
            }
            if dbg_lmhead {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut s = [0u16; 4];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    s.as_mut_ptr() as *mut _,
                    logits.device_ptr(),
                    8,
                );
                let v: Vec<f32> = s.iter().map(|&x| f16_to_f32(x)).collect();
                eprintln!("  [lm_head] after softcap: logits_f16 first4={:.4?}", v);
            }
            step_counter.set(step_counter.get() + 1);
            Ok(())
        };

        let set_step_meta = |step: i32| -> Result<()> {
            let pos = [step];
            let slot = [step];
            let ctx = [step + 1];
            positions.copy_from_host(bytemuck_cast_i32(&pos))?;
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot))?;
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx))?;
            Ok(())
        };

        let logits_row_elems = vocab as usize;
        let logits_row_bytes_f32 = logits_row_elems * 4;
        let mut logits_host_f32: Vec<f32> = vec![0.0f32; logits_row_elems];
        let mut total_nll: f64 = 0.0;
        let mut n_evaluated: usize = 0;

        // E4B PLE precompute is per-token: it depends on inputs_embeds
        // (the post-gather residual) and feeds each decoder layer's
        // per-layer-input slot. Wire it into the same closure that
        // runs embed + layer chain. The graph-capture path below can't
        // host arena.region() allocations safely, so force eager mode
        // whenever PLE is active.
        let e4b_ple_enabled = std::env::var("RVLLM_E4B_PLE")
            .map_or(false, |v| v == "1")
            && self.model.ple.is_some();
        // Build a graph-capturable forward: embed + all layers + lm_head.
        // No debug probes (they break capture).
        let ppl_forward = || -> Result<()> {
            rvllm_fused::EmbeddingGatherLaunch { num_tokens: 1, hidden, vocab }
                .launch(fn_embed, residual_ptr, self.model.embedding.offset_bytes, token_ids_region.device_ptr(), stream)?;
            if e4b_ple_enabled {
                let (base, stride) = unsafe {
                    self.precompute_ple(
                        residual_ptr,
                        token_ids_region.device_ptr(),
                        1u32,
                        fn_embed,
                        &kernels,
                        stream as u64,
                    )?
                };
                ple_state.set((base, stride));
            }
            one_step()
        };

        let use_graph = !e4b_ple_enabled
            && std::env::var("RVLLM_NO_GRAPH").ok().as_deref() != Some("1");
        let ppl_graph = if use_graph {
            // Dry run to populate KV cache slot 0
            let tok_i32 = [token_ids[0] as i32];
            token_ids_region.copy_from_host(bytemuck_cast_i32(&tok_i32))?;
            set_step_meta(0)?;
            ppl_forward()?;
            self.stream.fence()?;

            let g = rvllm_graph::CapturedGraph::capture(
                num_seqs,
                max_blocks_per_seq,
                rvllm_metadata::MetadataLayout::compute(num_seqs, max_blocks_per_seq).hash(),
                rvllm_graph::GraphFingerprint([0u8; 32]),
                stream,
                || ppl_forward(),
            )?;
            self.stream.fence()?;
            Some(g)
        } else {
            None
        };

        for (t, &tok_id) in token_ids.iter().enumerate() {
            let tok_i32 = [tok_id as i32];
            token_ids_region.copy_from_host(bytemuck_cast_i32(&tok_i32))?;
            set_step_meta(t as i32)?;

            if let Some(ref graph) = ppl_graph {
                graph.replay(stream)?;
            } else {
                ppl_forward()?;
            }

            if t + 1 < token_ids.len() {
                dtoh_async_sync(
                    logits_f32.device_ptr(),
                    logits_host_f32.as_mut_ptr() as *mut i32,
                    logits_row_bytes_f32,
                    stream,
                )?;
                self.stream.fence()?;

                let cap = arch.logit_softcap;
                if !skip_softcap && cap > 0.0 {
                    for x in logits_host_f32.iter_mut() {
                        *x = cap * (*x / cap).tanh();
                    }
                }

                let target = token_ids[t + 1] as usize;
                if t == 0 {
                    let first5: Vec<f32> = logits_host_f32[..5].to_vec();
                    let max_val = logits_host_f32
                        .iter()
                        .copied()
                        .filter(|v| !v.is_nan())
                        .fold(f32::MIN, f32::max);
                    let min_val = logits_host_f32
                        .iter()
                        .copied()
                        .filter(|v| !v.is_nan())
                        .fold(f32::MAX, f32::min);
                    eprintln!(
                        "  [ppl] logits(f32+softcap): first5={:?} min={:.1} max={:.1}",
                        first5, min_val, max_val
                    );
                }
                let nll = crate::bring_up::compute_nll_f32(&logits_host_f32, target);
                total_nll += nll;
                n_evaluated += 1;

                if (t + 1) % 32 == 0 || t + 1 == token_ids.len() - 1 {
                    let running_ppl = (total_nll / n_evaluated as f64).exp();
                    eprintln!(
                        "  step {}/{}: running_ppl={:.4}",
                        t + 1,
                        token_ids.len(),
                        running_ppl
                    );
                }
            } else {
                self.stream.fence()?;
            }
        }

        let ppl = if n_evaluated > 0 {
            (total_nll / n_evaluated as f64).exp()
        } else {
            0.0
        };
        Ok(crate::bring_up::PplResult {
            ppl,
            total_nll,
            n_evaluated,
        })
    }

    /// Greedy Gemma 4 E4B assistant-drafter speculative decode entry
    /// point. This is intentionally separate from [`Self::run_generate`]
    #[doc(hidden)] fn __c28_anchor() {}
}

/// Commit 28: sparse-table softmax + categorical sample for the
/// Gemma 4 MTP drafter's typical-acceptance mode.
///
/// Inputs:
///   - `ids[N]`     — i32 candidate token IDs (-1 = sentinel, skip)
///   - `logits[N]`  — f32 candidate logits (-INFINITY = sentinel, skip)
///   - `temperature` — 0 == argmax; >0 == sample with stable softmax
///   - `rng_state`  — LCG state advanced in place
///
/// Returns `(sampled_id, log_q)` where `log_q` is `log(p_drafter(sampled_id))`
/// computed over the sparse-table normalization. On empty/degenerate
/// input returns `(-1, f32::NEG_INFINITY)`.
fn sparse_sample_with_temp(
    ids: &[i32],
    logits: &[f32],
    temperature: f32,
    rng_state: &mut u64,
) -> (i32, f32) {
    assert_eq!(ids.len(), logits.len());
    if ids.is_empty() {
        return (-1, f32::NEG_INFINITY);
    }
    // Strict greedy: argmax over the table; log_q = 0 (degenerate;
    // typical-acceptance is gated on temp > 0 in callers).
    if temperature <= 0.0 {
        let mut best = f32::NEG_INFINITY;
        let mut best_id: i32 = -1;
        for i in 0..ids.len() {
            if ids[i] < 0 || !logits[i].is_finite() { continue; }
            if logits[i] > best {
                best = logits[i];
                best_id = ids[i];
            }
        }
        return (best_id, 0.0);
    }
    // Stable softmax: subtract max, scale by 1/T, exp, normalize.
    let inv_t = 1.0_f32 / temperature.max(1e-6);
    let mut max_l = f32::NEG_INFINITY;
    for i in 0..ids.len() {
        if ids[i] >= 0 && logits[i].is_finite() && logits[i] > max_l {
            max_l = logits[i];
        }
    }
    if !max_l.is_finite() {
        return (-1, f32::NEG_INFINITY);
    }
    let mut sum: f64 = 0.0;
    let mut probs: Vec<f64> = Vec::with_capacity(ids.len());
    for i in 0..ids.len() {
        if ids[i] < 0 || !logits[i].is_finite() {
            probs.push(0.0);
            continue;
        }
        let e = ((logits[i] - max_l) * inv_t) as f64;
        let p = e.exp();
        probs.push(p);
        sum += p;
    }
    if !(sum > 0.0 && sum.is_finite()) {
        return (-1, f32::NEG_INFINITY);
    }
    // LCG advance: same constants as glibc's drand48 in the simpler form.
    *rng_state = rng_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let r01 = ((*rng_state >> 33) as f64) / ((1u64 << 31) as f64);
    let u = r01.clamp(0.0, 1.0_f64 - 1e-9) * sum;
    let mut acc: f64 = 0.0;
    for i in 0..ids.len() {
        acc += probs[i];
        if u <= acc {
            let p = probs[i] / sum;
            return (ids[i], (p.max(1e-30).ln()) as f32);
        }
    }
    // Numerical fallback: return last positive-prob entry.
    for i in (0..ids.len()).rev() {
        if probs[i] > 0.0 {
            let p = probs[i] / sum;
            return (ids[i], (p.max(1e-30).ln()) as f32);
        }
    }
    (-1, f32::NEG_INFINITY)
}

impl Gemma4Bringup {
    #[doc(hidden)] fn __c28_anchor_end() {}
    /// (Placeholder kept so the methods after `__c28_anchor` remain in
    /// the same impl block. This is intentional — the file has many
    /// methods and we did not want to refactor module-level layout
    /// just to insert a free function.)
    ///
    /// Greedy/typical Gemma 4 E4B assistant-drafter speculative decode
    /// entry point. This is intentionally separate from
    /// [`Self::run_generate`]
    /// so the default path stays source-stable while the spec path is
    /// filled in.
    ///
    /// Current status: request/lifecycle plumbing and resident drafter
    /// upload are wired. The actual assistant one-step forward still
    /// returns `FeatureNotAvailable` until the cross-attention and
    /// MaskedEmbedder kernels land. Keeping this as a runtime method
    /// gives the server a real gated call site and prevents
    /// `RVLLM_GEMMA4_SPEC_DECODE=1` from silently falling back to
    /// baseline generation.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    /// Commit 38 — batched-verify spec-decode loop.
    ///
    /// Replaces the per-iter K+1 sequential decodes (current iterative
    /// wrapper) with: K drafter forwards → ONE base run_generate that
    /// batched-prefills the K drafter tokens → lm_head on the K
    /// captured hidden states → K logit rows for verify.
    ///
    /// Flow per iteration:
    ///   1. Drafter K times using last captured base_hidden → drafts[K]
    ///   2. Set base_last_k_pending = true, base_last_hidden_pending = true
    ///   3. Call run_generate(prompt + drafts, max_new = 1)
    ///      - prefix-cache hits on unchanged prompt
    ///      - K-token batched prefill of drafts (THE speedup vs K decodes)
    ///      - 1 decode produces base argmax at position P + K (bonus)
    ///      - Capture site copies K residual rows (post-final-norm) into
    ///        base_last_k_hidden_ptr + 1 row into base_last_hidden_ptr
    ///   4. lm_head GEMM (M=K) on captured K rows → K * vocab logits f32
    ///   5. Compare drafts vs base argmax of each row + apply typical-
    ///      acceptance with bias → accept_len
    ///   6. Emit drafts[..accept_len] + bonus (= decode argmax)
    ///   7. Update prompt by accept_len+1 tokens
    ///   8. For next iter's drafter input: base_hidden_last is captured
    ///      at the FINAL prefilled position (P + K - 1). We want
    ///      hidden at the position of the bonus base argmax = P + K.
    ///      In max_new=1 run_generate, that's the decode-step hidden
    ///      which we don't currently capture per-step. For now we
    ///      accept slight position mismatch and use the captured K-row
    ///      at index accept_len (== the position the drafter at iter+1
    ///      conditions on). For full correctness this can be sharpened
    ///      later.
    ///
    /// Codex review priority 2 — direct verify-batched-from-state API.
    /// Phase B-1 (commit 48): API surface + contract only.
    /// Phase B-2 (next commit): real layer-loop body.
    ///
    /// THE PRODUCTION SPEC-VERIFY HOTPATH. Replaces the current
    /// "run_generate(prompt + drafts, max_new=1)" hack. Direct flow:
    ///
    ///   1. EmbeddingGather over `drafts[K]` → row 0..K of an
    ///      arena residual buffer (K rows × hidden f16).
    ///   2. Build positions[K] = [start_pos .. start_pos + K].
    ///      Build slot_mapping[K] = same range (= the boundary
    ///      tokens whose K/V the verify pass will write).
    ///      context_lens = [start_pos + K] (last query token
    ///      attends to start_pos+K keys = prompt + previous drafts
    ///      + this iter's K drafts).
    ///   3. cu_seqlens_q = [0, K], max_seqlen_q = K, num_seqs = 1
    ///      → `Gemma4Phase::Prefill { ... }` for the layer loop.
    ///   4. Run `gemma4_forward_phase` (or equivalent inline) for
    ///      all 42 base layers, writing K/V slots at positions
    ///      [start_pos .. start_pos + K) into the existing
    ///      persistent KV cache. Hadamard / NVFP4 / FP8 dispatch
    ///      identical to `run_generate`'s prefill path.
    ///   5. Final RMSNorm on K rows of the residual buffer in place.
    ///   6. cuBLASLt f16_gemm_f32 (M=K, N=vocab, K=hidden) →
    ///      f32 logits buffer.
    ///   7. Apply softcap (f32) on K × vocab.
    ///   8. ArgmaxLaunch { num_tokens=K, vocab } → device u32 buffer.
    ///   9. DtoH the K argmaxes. K hiddens stay on device (next iter's
    ///      drafter input is K-buffer[accept_len - 1]).
    ///   10. Return the K argmax tokens + the device ptr to the K-row
    ///       post-final-norm hidden buffer.
    ///
    /// What this method does NOT do:
    ///   - Does NOT call `run_generate`.
    ///   - Does NOT touch `prefix_cache.last_tokens` /
    ///     `committed_prefix_len` (`SpecDecodeSession` owns spec-
    ///     internal committed state).
    ///   - Does NOT compute a "bonus decode token" — caller decides
    ///     what to emit at accept_len divergence from the K logits.
    ///   - Does NOT run drafter forward — caller does that separately
    ///     using `last_base_hidden_ptr` from the session.
    ///
    /// Phase B-1 STATUS: returns `FeatureNotAvailable` until the
    /// layer-loop body is inlined in B-2. Callers must check + fall
    /// back to the legacy batched-verify branch until then. This
    /// commit locks in the API surface so consumers (run_generate
    /// _speculative_batched in Phase C) can be wired against it
    /// at compile time independently of B-2's implementation.
    ///
    /// Output buffer ownership: caller pre-allocates
    /// `k_hidden_out_ptr` (K rows × hidden f16, post-final-norm)
    /// and `k_argmax_host_out` (K u32 slots). This method writes
    /// to them and returns Ok(()).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn verify_batched_from_state(
        &self,
        fn_embed: rvllm_kernels::KernelFn,
        drafts: &[u32],
        start_pos: u32,
        session: &SpecDecodeSession,
        k_hidden_out_ptr: u64,
        k_argmax_host_out: &mut [u32],
    ) -> Result<()> {
        // Phase B-2: real body. Rather than duplicating the ~650 LOC
        // chunk-prefill body of `run_generate`, we drive that path
        // with two new override hooks added in this commit:
        //
        //   * `force_common_prefix_override = start_pos`: bypasses
        //     the token-id match + `committed_prefix_len` chunk-cap
        //     in the prefix-cache lookup. `run_generate` then
        //     prefills exactly K = drafts.len() new tokens at
        //     positions [start_pos .. start_pos+K).
        //
        //   * `skip_prefix_cache_publish = true`: skips the end-of-
        //     request `last_tokens` + `committed_prefix_len` write,
        //     so spec-internal iteration doesn't pollute the
        //     cross-request cache. The session itself owns spec
        //     state.
        //
        // The existing K-row capture hook (commit 40, fixed in
        // commit 40b) writes the K POST-layer-loop hiddens into
        // `base_last_k_hidden_ptr`. We re-aim it at the caller's
        // output buffer for this pass.
        //
        // This is NOT the "no run_generate" purity the Phase B-1
        // contract documented, but it IS the substantive codex
        // priority-1 fix: short-prompt spec iters no longer
        // re-prefill the entire prompt under chunk_size cap.
        if drafts.is_empty() {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "drafts",
                    reason: "empty drafts vec".into(),
                },
                field: "drafts",
            });
        }
        if k_argmax_host_out.len() < drafts.len() {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "k_argmax_host_out",
                    reason: format!(
                        "buffer len {} < drafts.len {}",
                        k_argmax_host_out.len(),
                        drafts.len(),
                    )
                    .into(),
                },
                field: "k_argmax_host_out",
            });
        }
        if (drafts.len() as usize) > MAX_SPEC_K {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "drafts.len",
                    reason: format!(
                        "{} exceeds MAX_SPEC_K={}",
                        drafts.len(),
                        MAX_SPEC_K,
                    )
                    .into(),
                },
                field: "drafts.len",
            });
        }

        let k = drafts.len();

        // Build the input prompt: session.tokens (= prompt + accepted
        // drafts) + this iter's K drafts.
        let mut input_ids: Vec<u32> =
            Vec::with_capacity(session.tokens.len() + k);
        input_ids.extend_from_slice(&session.tokens);
        input_ids.extend_from_slice(drafts);
        debug_assert_eq!(
            session.tokens.len() as u32,
            start_pos,
            "verify_batched_from_state: session.tokens.len() must equal start_pos",
        );

        // Codex Round 2 #1: arena checkpoint covering the inner
        // run_generate. Without this, run_generate's prompt-
        // proportional scratch (input_ids.len() = session.tokens.len()
        // + K) leaks into the spec session's arena per iteration —
        // OOM / catastrophic arena pressure on long E4B contexts.
        // `k_hidden_out_ptr` is allocated above scratch (by
        // ensure_drafter), so restoring this checkpoint is safe.
        let arena = &self.arena;
        let outer_ck = arena.checkpoint();
        // Run the verify pass inside a labelled scope so the guard
        // disarms hook atomics + the K-dst restore fires before we
        // restore the arena.
        let verify_result: Result<()> = (|| {
            // Codex Round 2 #2: RAII guard. force_prefill_only causes
            // run_generate to skip_decode early-return, which bypasses
            // the late skip_prefix_cache_publish.swap site. Without
            // the guard, the publish-skip flag leaked into the next
            // request and silently suppressed one cross-request cache
            // publish.
            let mut hook_guard = SpecHookGuard::new(self);
            hook_guard.swap_k_dst(k_hidden_out_ptr);
            self.base_last_k_count
                .store(k as u32, std::sync::atomic::Ordering::Release);
            self.base_last_k_snapshot_pending
                .store(true, std::sync::atomic::Ordering::Release);
            self.force_common_prefix_override
                .store(start_pos, std::sync::atomic::Ordering::Release);
            self.skip_prefix_cache_publish
                .store(true, std::sync::atomic::Ordering::Release);
            self.force_prefill_only
                .store(true, std::sync::atomic::Ordering::Release);

            let _bonus = self.run_generate(
                fn_embed,
                self.fused.fn_argmax,
                &input_ids,
                /* max_new */ 1,
                /* eos_ids */ &[],
                /* shadow_requested */ false,
                SamplingConfig::Greedy,
                /* cancel */ None,
                /* on_token */ None,
                /* vision_splice */ &[],
                /* audio_splice */ &[],
            )?;
            Ok(())
            // hook_guard drops here — atomics disarmed, K-dst restored.
        })();
        if let Err(e) = verify_result {
            arena.restore(outer_ck);
            return Err(e);
        }

        // K-row final_norm + lm_head + softcap + argmax over the
        // captured POST-layer-loop hiddens. Allocates only K-sized
        // scratch (hidden / logits / argmax), bounded independent of
        // session length.
        //
        // Codex Round 7 perf #2: when RVLLM_RESIDUAL_BF16=1, the
        // chunked-prefill layer loop runs in bf16 and the K-row hidden
        // capture writes bf16 bytes into k_hidden_out_ptr. Using the
        // f16 rmsnorm here mis-interprets those bytes — bf16's wider
        // exponent + narrower mantissa makes the reinterpret produce
        // garbled values, which flow into lm_head and pull the K
        // argmaxes toward attractor tokens regardless of actual base
        // distribution. Result: drafter agreement on those (correlated-
        // wrong) argmaxes was artificially low. Selecting the bf16
        // sibling rmsnorm + narrowing back to f16 via bf16_to_f16_sat
        // restores correct base argmaxes (and therefore real accept
        // rates).
        let stream = self.stream.raw();
        let hidden_u = self.arch.hidden_size as u32;
        let vocab_u = self.arch.vocab_size as u32;
        let bf16_res = bf16_residual_enabled();
        let final_norm_kernel = if bf16_res {
            self.fused.fn_rmsnorm_inplace_bf16
        } else {
            self.fused.fn_rmsnorm
        };
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: k as u32,
            hidden: hidden_u,
            eps: self.arch.rms_norm_eps,
        }
        .launch(
            final_norm_kernel,
            k_hidden_out_ptr,
            self.model.final_norm.offset_bytes,
            stream,
        )?;
        if bf16_res {
            rvllm_fused::gemma4_launcher::Bf16ToF16SatLaunch { n: hidden_u * (k as u32) }
                .launch(self.fused.fn_bf16_to_f16_sat, k_hidden_out_ptr, k_hidden_out_ptr, stream)?;
        }

        let logits_region = arena.region(
            "spec_verify_logits_f32",
            k * (vocab_u as usize) * 4,
            16,
        )?;
        self.cublaslt.f16_gemm_f32(
            k_hidden_out_ptr,
            self.model.lm_head_f16.offset_bytes,
            logits_region.device_ptr(),
            k as i32,
            vocab_u as i32,
            hidden_u as i32,
            stream,
        )?;
        if self.arch.logit_softcap > 0.0 {
            rvllm_fused::gemma4_launcher::LogitSoftcapLaunch {
                num_tokens: k as u32,
                vocab: vocab_u,
                cap: self.arch.logit_softcap,
            }
            .launch(
                self.fused.fn_softcap_f32,
                logits_region.device_ptr(),
                stream,
            )?;
        }
        let argmax_region = arena.region("spec_verify_argmax", k * 4, 16)?;
        rvllm_fused::ArgmaxLaunch {
            num_tokens: k as u32,
            vocab: vocab_u,
        }
        .launch(
            self.fused.fn_argmax,
            logits_region.device_ptr(),
            argmax_region.device_ptr(),
            stream,
        )?;
        self.stream.fence()?;
        let rc = cudarc::driver::sys::cuMemcpyDtoH_v2(
            k_argmax_host_out.as_mut_ptr() as *mut _,
            argmax_region.device_ptr(),
            k * 4,
        );
        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            arena.restore(outer_ck);
            return Err(rvllm_core::RvllmError::cuda(
                "verify_batched_from_state: argmax DtoH",
                rvllm_core::CudaErrorKind::MemcpyFailed,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        arena.restore(outer_ck);
        let _ = session; // future: cross-check committed_len consistency
        Ok(())
    }

    /// Drafter K-step from session state — the substantive primitive
    /// the batched-spec session loop uses each iteration. Does NOT
    /// call `run_generate` / `run_generate_speculative`; reads
    /// `session.last_base_hidden_ptr`, `session.last_committed_token()`
    /// and `session.drafter_input_position()` directly and threads
    /// them through the existing drafter helpers
    /// (`apply_pre_projection_embed_scale`, `run_drafter_pre_projection`,
    /// `run_drafter_layer_q_side`, `run_drafter_layer_attn_finisher`,
    /// `run_drafter_layer_mlp_finisher`).
    ///
    /// Arena: checkpoint at entry, restore at exit — per-iter scratch
    /// (block tables, ctx lens, last-token embed, drafter workspace)
    /// is released. Caller does not have to checkpoint.
    ///
    /// Caller invariants:
    ///   * `session.last_base_hidden_ptr` points at the POST-final-
    ///     norm hidden of token `session.committed_len - 1` (warmup or
    ///     prior verify pass copied K-row into it).
    ///   * Shadow KV is already populated for the active context
    ///     (slots `[0, session.committed_len)` of the prompt + accepted
    ///     drafts). Incremental shadow updates are the caller's job.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_drafter_k_from_state(
        &self,
        fn_embed: rvllm_kernels::KernelFn,
        session: &SpecDecodeSession,
        spec_k: u32,
        sampling: SamplingConfig,
        typical_mode: bool,
    ) -> Result<DraftBatch> {
        let arch = &self.arch;
        let hidden_u = arch.hidden_size as u32;
        let vocab_u = arch.vocab_size as u32;
        let stream = self.stream.raw();
        let arena = &self.arena;
        let arena_ck = arena.checkpoint();

        // Pull KV layout from prefix cache.
        let (
            kv_base_ptr,
            kv_scale_base_ptr,
            kv_layer_offsets,
            kv_scale_layer_offsets,
            num_blocks_total,
            block_size,
            max_blocks_per_seq,
        ) = {
            let pc_guard = self.prefix_cache.lock().unwrap();
            let pc = pc_guard.as_ref().ok_or_else(|| {
                rvllm_core::RvllmError::Attention {
                    err: rvllm_core::AttentionError::FeatureNotAvailable {
                        op: "run_drafter_k_from_state: prefix cache empty",
                        backend: "Gemma4SpecDecode",
                    },
                    ctx: rvllm_core::AttnCtx {
                        op: "run_drafter_k_from_state",
                        stream,
                        num_seqs: 1,
                        head_dim: arch.max_head_dim() as u32,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                }
            })?;
            (
                pc.kv_cache_ptr,
                pc.kv_scale_ptr,
                pc.kv_layer_offsets.clone(),
                pc.kv_scale_layer_offsets.clone(),
                pc.num_blocks_total,
                pc.block_size,
                pc.num_blocks_total,
            )
        };
        let sliding_blocks = num_blocks_total;

        let mut kv_dtype_per_layer: Vec<crate::gemma4_layer_exec::KvDtype> =
            Vec::with_capacity(arch.num_hidden_layers);
        for l in 0..arch.num_hidden_layers {
            kv_dtype_per_layer.push(
                crate::gemma4_layer_exec::KvDtype::for_layer_index_or_env(
                    arch.layer_types[l], l, false));
        }

        // Persistent identity block tables [0..num_blocks_total).
        let block_tables_region = arena.region(
            "spec_session_block_tables",
            (num_blocks_total as usize) * 4,
            16,
        )?;
        {
            let mut bt_host: Vec<u8> =
                Vec::with_capacity((num_blocks_total as usize) * 4);
            for b in 0..num_blocks_total {
                bt_host.extend_from_slice(&(b as i32).to_le_bytes());
            }
            block_tables_region.copy_from_host(&bt_host)?;
        }

        // context_lens = session.committed_len (drafter sees the
        // already-committed K/V; new draft positions are predicted).
        let ctx_len_val: i32 = session.committed_len as i32;
        let context_lens_region =
            arena.region("spec_session_ctx_lens", 4, 16)?;
        context_lens_region.copy_from_host(&ctx_len_val.to_le_bytes())?;

        // last_token_embed = embed(session.last_committed_token()).
        let last_committed_tok = session.last_committed_token().ok_or_else(|| {
            rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "session",
                    reason: "no committed tokens; warmup must run first".into(),
                },
                field: "session",
            }
        })?;
        let token_ids_region =
            arena.region("spec_session_tok_ids", 4, 16)?;
        token_ids_region.copy_from_host(&last_committed_tok.to_le_bytes())?;
        let last_token_embed = arena.region(
            "spec_session_last_tok_embed",
            (hidden_u as usize) * 2,
            16,
        )?;
        rvllm_fused::EmbeddingGatherLaunch {
            num_tokens: 1,
            hidden: hidden_u,
            vocab: vocab_u,
        }
        .launch(
            fn_embed,
            last_token_embed.device_ptr(),
            self.model.embedding.offset_bytes,
            token_ids_region.device_ptr(),
            stream,
        )?;

        let sources = self.assistant_kv_sources
            .expect("assistant_kv_sources guarded by caller");

        let source_view = |layer_idx: u32| -> crate::gemma4_drafter::DrafterBaseKvView {
            let li = layer_idx as usize;
            let off = kv_layer_offsets[li];
            let scale_off = kv_scale_layer_offsets[li];
            let is_global = arch.layer_types[li]
                == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
            let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
            let nkvh = arch.num_kv_heads_for_layer(li) as u32;
            let hd = arch.head_dim_for_layer(li) as u32;
            let layer_elems =
                2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
            let dtype = kv_dtype_per_layer[li];
            let k_v_half_bytes = match dtype {
                crate::gemma4_layer_exec::KvDtype::F16 => layer_elems,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_elems / 2,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 4,
            };
            let scale_half_slots =
                layer_blocks as u64 * block_size as u64 * nkvh as u64;
            let scale_half_bytes = match dtype {
                crate::gemma4_layer_exec::KvDtype::F16 => 0,
                crate::gemma4_layer_exec::KvDtype::Fp8 => scale_half_slots * 4,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 32,
            };
            let k_cache = kv_base_ptr + off;
            let v_cache = k_cache + k_v_half_bytes;
            let (k_scale_cache, v_scale_cache) = if dtype
                == crate::gemma4_layer_exec::KvDtype::F16
            {
                (0u64, 0u64)
            } else {
                let k_s = kv_scale_base_ptr + scale_off;
                (k_s, k_s + scale_half_bytes)
            };
            crate::gemma4_drafter::DrafterBaseKvView {
                k_cache,
                v_cache,
                k_scale_cache,
                v_scale_cache,
                q_scale_cache: 0,
                block_tables: block_tables_region.device_ptr(),
                context_lens: context_lens_region.device_ptr(),
                block_size,
                max_blocks_per_seq,
                num_blocks_total,
                kv_dtype: dtype,
            }
        };
        let sliding_kv = source_view(sources.sliding_source_layer);
        let full_kv = source_view(sources.full_source_layer);

        let workspace = {
            let guard = self.drafter.lock().unwrap();
            let d = guard.as_ref().expect("drafter resident");
            d.alloc_step_workspace(arena)?
        };

        let (spec_top_k_h, spec_per_centroid_h): (usize, usize) = {
            let guard = self.drafter.lock().unwrap();
            let d = guard.as_ref().expect("drafter resident");
            let tk = d.arch.centroid_intermediate_top_k;
            let nc = d.arch.num_centroids;
            let vc = d.arch.vocab_size;
            (tk, if nc > 0 { vc / nc } else { 0 })
        };

        let drafter_position: u32 = session.drafter_input_position();
        let mut current_base_hidden: u64 = session.last_base_hidden_ptr;
        let mut drafter_tokens: Vec<u32> = Vec::with_capacity(spec_k as usize);
        let mut drafter_log_q: Vec<f32> = Vec::with_capacity(spec_k as usize);

        // Codex Round 2 #4: device-side draft-id ring. Removes K-1
        // fences + K-1 host round-trips per spec iteration. Each
        // drafter step DtoD-copies its `workspace.out_token_id`
        // (i32[1]) into `draft_ids_dev[k_step]`; the next step's
        // EmbeddingGather reads that slot directly. After the loop a
        // single fence + K*4-byte DtoH yields the host vector.
        //
        // Typical-mode falls back to the per-step DtoH/HtoD path
        // because host sampling may override the token id between
        // the kernel write and the next step's read.
        let draft_ids_dev = arena.region(
            "spec_session_draft_ids",
            (spec_k as usize) * 4,
            16,
        )?;
        let mut next_rand_f32_spec: u64 = match sampling {
            SamplingConfig::Stochastic { seed, .. } =>
                seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
            SamplingConfig::Greedy => 0xDEAD_BEEF_CAFE_BABE,
        };

        let sliding_window = self.arch.sliding_window_size as i32;
        // Codex Round 6 — revert the Round-4 "mtp default" change.
        // The local HF-parity reference (v3/tools/manual_drafter_reference.py
        // lines 176, 264, 290) computes drafter cross-attention as
        //     scores = (q · k) * (1.0 / sqrt(head_dim))
        // for BOTH sliding and global source layers — i.e. the
        // standard attention scale. The Round-4 default of "mtp"
        // (scale = 1.0) was based on a misread of a legacy code
        // comment claiming vLLM uses 1.0 for Gemma4MTPAttention;
        // hardware smoke + the reference script both disagree.
        // Reverting to "stable" (= 1/sqrt(head_dim)) keeps the
        // drafter distribution aligned with HF until a proper
        // HF-parity dump definitively settles the question.
        let scale_mode = std::env::var("RVLLM_SPEC_FA_SCALE")
            .unwrap_or_else(|_| "stable".into());

        for k_step in 0..(spec_k as usize) {
            let step = crate::gemma4_drafter::DrafterForwardStep {
                base_hidden_last_step: current_base_hidden,
                last_token_embed: last_token_embed.device_ptr(),
                sliding_kv,
                full_kv,
                position: drafter_position + k_step as u32,
                out_logits: workspace.centroid_logits,
                out_hidden: workspace.out_hidden,
                out_token_id: workspace.out_token_id,
            };

            {
                let guard = self.drafter.lock().unwrap();
                let drafter = guard.as_ref().expect("drafter resident");
                drafter.prepare_pre_projection_input(&step, &workspace, stream)?;
                // Codex Round 6 (final): HF
                // `MultiTokenPredictionCandidateGenerator.get_candidates`
                // at candidate_generator.py:1383 builds
                //   `inputs_embeds = cat([last_token_embedding,
                //                         last_hidden_state], dim=-1)`
                // with NO sqrt(hidden_size) scaling on the embed half.
                // The earlier `apply_pre_projection_embed_scale` call
                // applied sqrt(2560), which inflated the embed half by
                // ~50x and dragged the drafter into a degenerate
                // prediction mode (always token 140 = "    " spaces).
                // After removing it: drafter at the same input predicts
                // "**" with the correct "Die" token at #2 — i.e., the
                // drafter now actually contributes useful candidates.
                // Verified against HF transformers source for
                // gemma4_assistant + the MTP candidate generator.
                let _ = drafter.arch.backbone_hidden_size; // intentionally unused
                self.run_drafter_pre_projection(drafter, &workspace)?;

                let num_layers = drafter.layers.len();
                for li in 0..num_layers {
                    let layer = &drafter.layers[li];
                    let is_global = matches!(
                        layer.layer_type,
                        rvllm_loader::gemma4_drafter::DrafterLayerType::Full
                    );
                    let eff_hd = layer.effective_head_dim as f32;
                    let scale = if scale_mode == "mtp" {
                        1.0_f32
                    } else {
                        1.0_f32 / eff_hd.sqrt()
                    };
                    self.run_drafter_layer_q_side(
                        drafter, &workspace, li, step.position)?;
                    if is_global {
                        drafter.launch_cross_attn_global(
                            workspace.attn_out,
                            workspace.q,
                            block_tables_region.device_ptr(),
                            context_lens_region.device_ptr(),
                            scale,
                            stream,
                        )?;
                    } else {
                        drafter.launch_cross_attn_sliding(
                            workspace.attn_out,
                            workspace.q,
                            block_tables_region.device_ptr(),
                            context_lens_region.device_ptr(),
                            scale,
                            sliding_window,
                            stream,
                        )?;
                    }
                    self.run_drafter_layer_attn_finisher(drafter, &workspace, li)?;
                    self.run_drafter_layer_mlp_finisher(drafter, &workspace, li)?;
                }

                rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                    num_tokens: 1,
                    hidden: drafter.arch.hidden_size as u32,
                    eps: drafter.arch.rms_norm_eps,
                }
                .launch(
                    self.fused.fn_rmsnorm,
                    workspace.hidden,
                    drafter.top.final_norm,
                    stream,
                )?;

                let n_centroids = drafter.arch.num_centroids as i32;
                let top_k = drafter.arch.centroid_intermediate_top_k as i32;
                let vocab = drafter.arch.vocab_size as i32;
                let per_centroid: i32 = if n_centroids > 0 {
                    vocab / n_centroids
                } else { 0 };
                let fn_masked = drafter
                    .fn_masked_embedder_argmax_f16
                    .expect("MaskedEmbedder kernel attached in ensure_drafter");
                let (sp_ids_ptr, sp_lg_ptr) = if typical_mode {
                    (workspace.sparse_ids, workspace.sparse_logits)
                } else {
                    (0u64, 0u64)
                };
                crate::gemma4_drafter::launch_masked_embedder_argmax_f16(
                    fn_masked,
                    workspace.hidden,
                    drafter.top.centroids,
                    drafter.top.token_ordering,
                    drafter.top.embed_tokens,
                    drafter.arch.hidden_size as i32,
                    n_centroids,
                    top_k,
                    per_centroid,
                    vocab,
                    workspace.out_token_id,
                    0,
                    sp_ids_ptr,
                    sp_lg_ptr,
                    stream,
                )?;

                self.cublaslt.f16_gemm_f32(
                    workspace.hidden,
                    drafter.top.post_projection,
                    workspace.gemm_f32,
                    1,
                    drafter.arch.backbone_hidden_size as i32,
                    drafter.arch.hidden_size as i32,
                    stream,
                )?;
                launch_cast_f32_to_f16(
                    &self.stream,
                    self.fused.fn_cast_f32_to_f16,
                    workspace.gemm_f32,
                    workspace.out_hidden,
                    drafter.arch.backbone_hidden_size as i32,
                )?;
            } // drop drafter lock

            if typical_mode {
                // Per-step DtoH/HtoD path: host sampling may override
                // the token id before the next step's embed read.
                self.stream.fence()?;
                let mut tok_host: [u8; 4] = [0; 4];
                let rc_t = cudarc::driver::sys::cuMemcpyDtoH_v2(
                    tok_host.as_mut_ptr() as *mut _,
                    workspace.out_token_id,
                    4,
                );
                if rc_t != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    self.arena.restore(arena_ck);
                    return Err(rvllm_core::RvllmError::cuda(
                        "run_drafter_k_from_state: DtoH drafter token",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
                let mut tok_u: u32 = i32::from_le_bytes(tok_host).max(0) as u32;

                let sparse_len = spec_top_k_h * spec_per_centroid_h;
                let mut h_ids = vec![-1i32; sparse_len];
                let mut h_lg = vec![f32::NEG_INFINITY; sparse_len];
                let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                    h_ids.as_mut_ptr() as *mut _,
                    workspace.sparse_ids,
                    sparse_len * 4,
                );
                let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                    h_lg.as_mut_ptr() as *mut _,
                    workspace.sparse_logits,
                    sparse_len * 4,
                );
                let temp: f32 = match sampling {
                    SamplingConfig::Greedy => 0.0,
                    SamplingConfig::Stochastic { temperature, .. } => temperature,
                };
                let (sampled_id, log_q_sampled) = sparse_sample_with_temp(
                    &h_ids, &h_lg, temp, &mut next_rand_f32_spec);
                if sampled_id >= 0 {
                    tok_u = sampled_id as u32;
                }
                drafter_log_q.push(log_q_sampled);
                drafter_tokens.push(tok_u);

                if k_step + 1 < (spec_k as usize) {
                    token_ids_region.copy_from_host(&tok_u.to_le_bytes())?;
                    rvllm_fused::EmbeddingGatherLaunch {
                        num_tokens: 1,
                        hidden: hidden_u,
                        vocab: vocab_u,
                    }
                    .launch(
                        fn_embed,
                        last_token_embed.device_ptr(),
                        self.model.embedding.offset_bytes,
                        token_ids_region.device_ptr(),
                        stream,
                    )?;
                    current_base_hidden = workspace.out_hidden;
                }
            } else {
                // Greedy path: device-chain. DtoD this step's token id
                // into draft_ids_dev[k_step] and (if not last) point
                // the next EmbeddingGather at that slot. No fence, no
                // DtoH/HtoD round-trip.
                use cudarc::driver::sys::*;
                let dst_slot = draft_ids_dev.device_ptr() + (k_step as u64) * 4;
                let rc_d = cuMemcpyDtoDAsync_v2(
                    dst_slot,
                    workspace.out_token_id,
                    4,
                    stream as CUstream,
                );
                if rc_d != CUresult::CUDA_SUCCESS {
                    self.arena.restore(arena_ck);
                    return Err(rvllm_core::RvllmError::cuda(
                        "run_drafter_k_from_state: DtoD draft-id chain",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
                if k_step + 1 < (spec_k as usize) {
                    rvllm_fused::EmbeddingGatherLaunch {
                        num_tokens: 1,
                        hidden: hidden_u,
                        vocab: vocab_u,
                    }
                    .launch(
                        fn_embed,
                        last_token_embed.device_ptr(),
                        self.model.embedding.offset_bytes,
                        dst_slot,
                        stream,
                    )?;
                    current_base_hidden = workspace.out_hidden;
                }
            }
        }

        // Greedy path: single DtoH of all K draft ids after the K loop.
        if !typical_mode {
            self.stream.fence()?;
            let mut h_buf = vec![0u8; (spec_k as usize) * 4];
            let rc = cudarc::driver::sys::cuMemcpyDtoH_v2(
                h_buf.as_mut_ptr() as *mut _,
                draft_ids_dev.device_ptr(),
                (spec_k as usize) * 4,
            );
            if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                self.arena.restore(arena_ck);
                return Err(rvllm_core::RvllmError::cuda(
                    "run_drafter_k_from_state: batched DtoH draft ids",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            drafter_tokens.clear();
            drafter_tokens.reserve(spec_k as usize);
            for k_step in 0..(spec_k as usize) {
                let off = k_step * 4;
                let arr = [h_buf[off], h_buf[off + 1], h_buf[off + 2], h_buf[off + 3]];
                // MaskedEmbedder writes i32; saturate negative
                // sentinel to 0 (same defensive behaviour as legacy).
                let id = i32::from_le_bytes(arr).max(0) as u32;
                drafter_tokens.push(id);
            }
        }

        self.arena.restore(arena_ck);
        Ok(DraftBatch { tokens: drafter_tokens, log_q: drafter_log_q })
    }

    /// Single-token base prefill from session state. Used by the
    /// session loop's `accept_len == 0` branch: the divergence/bonus
    /// token has no base K/V yet, so we run base over (tokens + bonus)
    /// to write its K/V slot, snapshot the new POST-final-norm hidden,
    /// and return base's argmax at the next position (which becomes
    /// `session.next_base_argmax` for the next iter).
    ///
    /// Updates `session.tokens`, `session.committed_len`, and
    /// `session.last_base_hidden_ptr` on success.
    #[cfg(feature = "cuda")]
    unsafe fn prefill_one_from_state(
        &self,
        fn_embed: rvllm_kernels::KernelFn,
        session: &mut SpecDecodeSession,
        bonus: u32,
    ) -> Result<u32> {
        let mut input = session.tokens.clone();
        input.push(bonus);
        let start_pos = session.committed_len;

        // Codex Round 2 #1+#2: arena checkpoint covering run_generate's
        // prompt-proportional scratch + RAII guard disarming hook
        // atomics on every exit path. base_last_hidden_ptr is allocated
        // above scratch by ensure_drafter, so restoring the checkpoint
        // does not invalidate the snapshot the hook just captured into
        // that buffer.
        // Codex Round 7 perf: skip the bonus's transformer-stack decode
        // step. Capture the bonus's POST-layer-loop PRE-final-norm
        // hidden via the K-row capture hook (count=1) — same primitive
        // verify_batched_from_state uses — then run final_norm +
        // lm_head + argmax over that single row ourselves. The decode
        // step's full 42-layer pass is ~30x more expensive than these
        // small kernels combined, so this saves roughly one
        // decode-equivalent per accept_len ≤ K iteration.
        let outer_ck = self.arena.checkpoint();
        let stream = self.stream.raw();
        let hidden_u = self.arch.hidden_size as u32;
        let vocab_u = self.arch.vocab_size as u32;
        let k_hidden_buf = self
            .base_last_k_hidden_ptr
            .load(std::sync::atomic::Ordering::Acquire);
        if k_hidden_buf == 0 {
            return Err(rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    op: "prefill_one_from_state: base_last_k_hidden_ptr is 0",
                    backend: "Gemma4SpecDecode",
                },
                ctx: rvllm_core::AttnCtx {
                    op: "prefill_one_from_state",
                    stream,
                    num_seqs: 1,
                    head_dim: self.arch.max_head_dim() as u32,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let call_result: Result<()> = (|| {
            let mut hook_guard = SpecHookGuard::new(self);
            hook_guard.swap_k_dst(k_hidden_buf);
            self.base_last_k_count
                .store(1, std::sync::atomic::Ordering::Release);
            self.base_last_k_snapshot_pending
                .store(true, std::sync::atomic::Ordering::Release);
            // Skip the decode step (next argmax recomputed below).
            self.force_prefill_only
                .store(true, std::sync::atomic::Ordering::Release);
            // No force_common_prefix_override: rely on natural prefix-
            // cache token-id match. Only the bonus is new, so new_q=1
            // via the standard path. skip_prefix_cache_publish keeps
            // spec-internal state out of the cross-request cache.
            self.skip_prefix_cache_publish
                .store(true, std::sync::atomic::Ordering::Release);

            let _ = self.run_generate(
                fn_embed,
                self.fused.fn_argmax,
                &input,
                /* max_new */ 1,
                /* eos_ids */ &[],
                /* shadow_requested */ false,
                SamplingConfig::Greedy,
                /* cancel */ None,
                /* on_token */ None,
                /* vision_splice */ &[],
                /* audio_splice */ &[],
            )?;
            Ok(())
        })();
        self.arena.restore(outer_ck);
        call_result?;

        // K-row capture wrote 1 row of POST-layer-loop PRE-final-norm
        // hidden at k_hidden_buf row 0. Run final_norm with the bf16-
        // aware kernel + narrow if bf16 residual, then lm_head + softcap
        // + argmax over the same row → next_base_argmax. Also DtoD-copy
        // the post-norm hidden into base_last_hidden_ptr for the next
        // drafter step.
        let bf16_res = bf16_residual_enabled();
        let final_norm_kernel = if bf16_res {
            self.fused.fn_rmsnorm_inplace_bf16
        } else {
            self.fused.fn_rmsnorm
        };
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1,
            hidden: hidden_u,
            eps: self.arch.rms_norm_eps,
        }
        .launch(
            final_norm_kernel,
            k_hidden_buf,
            self.model.final_norm.offset_bytes,
            stream,
        )?;
        if bf16_res {
            rvllm_fused::gemma4_launcher::Bf16ToF16SatLaunch { n: hidden_u }
                .launch(self.fused.fn_bf16_to_f16_sat, k_hidden_buf, k_hidden_buf, stream)?;
        }
        // Copy POST-final-norm row 0 into base_last_hidden_ptr for the
        // next iter's drafter.
        {
            use cudarc::driver::sys::*;
            let dst = self
                .base_last_hidden_ptr
                .load(std::sync::atomic::Ordering::Acquire);
            if dst == 0 {
                return Err(rvllm_core::RvllmError::Attention {
                    err: rvllm_core::AttentionError::FeatureNotAvailable {
                        op: "prefill_one_from_state: base_last_hidden_ptr is 0",
                        backend: "Gemma4SpecDecode",
                    },
                    ctx: rvllm_core::AttnCtx {
                        op: "prefill_one_from_state",
                        stream,
                        num_seqs: 1,
                        head_dim: self.arch.max_head_dim() as u32,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
            let rc = cuMemcpyDtoDAsync_v2(
                dst,
                k_hidden_buf,
                (hidden_u as usize) * 2,
                stream as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "prefill_one_from_state: bonus_hidden DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // lm_head_M=1 + softcap + argmax → next_base_argmax.
        let inner_ck = self.arena.checkpoint();
        let inner: Result<u32> = (|| {
            let logits = self.arena.region(
                "spec_prefill_one_logits",
                (vocab_u as usize) * 4,
                16,
            )?;
            self.cublaslt.f16_gemm_f32(
                k_hidden_buf,
                self.model.lm_head_f16.offset_bytes,
                logits.device_ptr(),
                1,
                vocab_u as i32,
                hidden_u as i32,
                stream,
            )?;
            if self.arch.logit_softcap > 0.0 {
                rvllm_fused::gemma4_launcher::LogitSoftcapLaunch {
                    num_tokens: 1,
                    vocab: vocab_u,
                    cap: self.arch.logit_softcap,
                }
                .launch(self.fused.fn_softcap_f32, logits.device_ptr(), stream)?;
            }
            let argmax_region = self.arena.region(
                "spec_prefill_one_argmax", 4, 16,
            )?;
            rvllm_fused::ArgmaxLaunch {
                num_tokens: 1,
                vocab: vocab_u,
            }
            .launch(
                self.fused.fn_argmax,
                logits.device_ptr(),
                argmax_region.device_ptr(),
                stream,
            )?;
            self.stream.fence()?;
            let mut host = [0u8; 4];
            let rc = cudarc::driver::sys::cuMemcpyDtoH_v2(
                host.as_mut_ptr() as *mut _,
                argmax_region.device_ptr(),
                4,
            );
            if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "prefill_one_from_state: argmax DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            Ok(u32::from_le_bytes(host))
        })();
        self.arena.restore(inner_ck);
        let new_argmax = inner?;

        session.tokens.push(bonus);
        session.committed_len = session.committed_len.saturating_add(1);
        session.last_base_hidden_ptr = self
            .base_last_hidden_ptr
            .load(std::sync::atomic::Ordering::Acquire);
        Ok(new_argmax)
    }

    /// SESSION-DRIVEN batched speculative decode. The active default
    /// when `RVLLM_GEMMA4_SPEC_DECODE=1`. Replaces the previous
    /// wrapper-around-`run_generate_speculative` implementation:
    ///
    ///   1. one prompt warmup (`run_generate(max_new=1)`)
    ///   2. open `SpecDecodeSession`, seed from warmup
    ///   3. initial shadow-KV populate over the prompt
    ///   4. loop {
    ///        drafts = run_drafter_k_from_state(session)
    ///        verify_batched_from_state -> K base argmaxes (+ K hiddens)
    ///        compute greedy accept_len
    ///        emit accepted drafts; on accept==0 emit deferred bonus
    ///        commit; incremental shadow-KV update
    ///        on accept==0: prefill_one_from_state(bonus)
    ///      }
    ///
    /// No recursion into `run_generate_speculative`. No
    /// `skip_next_warmup` / `saved_warmup_b_p` / `force_emit_accepted`
    /// orchestration. `verify_batched_from_state` is used as the
    /// transitional batched-prefill primitive (it still internally
    /// drives `run_generate` with `force_*_override` hooks, but the
    /// outer flow is now session-shape).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_generate_speculative_batched(
        &self,
        fn_embed: rvllm_kernels::KernelFn,
        fn_argmax: rvllm_kernels::KernelFn,
        prompt_ids: &[u32],
        max_new: usize,
        eos_ids: &[u32],
        spec_k: u32,
        sampling: SamplingConfig,
        cancel: Option<&std::sync::atomic::AtomicBool>,
        mut on_token: Option<&mut dyn FnMut(u32) -> bool>,
        vision_splice: &[(usize, &[u8])],
        audio_splice: &[(usize, &[u8])],
    ) -> Result<Vec<u32>> {
        // ---- Validation ----
        if max_new == 0 || spec_k == 0 {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "max_new/spec_k",
                    reason: "must be >= 1".into(),
                },
                field: "max_new",
            });
        }
        if (spec_k as usize) > MAX_SPEC_K {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "spec_k",
                    reason: format!(
                        "{} exceeds MAX_SPEC_K={}",
                        spec_k, MAX_SPEC_K
                    ).into(),
                },
                field: "spec_k",
            });
        }
        if prompt_ids.is_empty() {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "prompt_ids",
                    reason: "speculative decode requires a non-empty prompt".into(),
                },
                field: "prompt_ids",
            });
        }
        if self.assistant_kv_sources.is_none() {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "model",
                    reason: "assistant drafter requires E4B-style Gemma 4 with shared-KV source layers".into(),
                },
                field: "model",
            });
        }
        if self.drafter.lock().unwrap().is_none() {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "drafter",
                    reason: "drafter not resident; call ensure_drafter first".into(),
                },
                field: "drafter",
            });
        }
        // Codex Round 4 #4: match the legacy spec path's text-only
        // policy. Warmup would splice vision/audio but subsequent
        // verify / prefill_one_from_state calls run with empty
        // splices, and multimodal spec-decode parity (warmup KV,
        // shadow KV, PLE, assistant shared-KV vs HF on a vision
        // prompt) has never been validated. Reject up front until
        // that audit lands.
        if !vision_splice.is_empty() || !audio_splice.is_empty() {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "modalities",
                    reason: "Gemma 4 speculative decode is text-only \
                             until multimodal parity is validated".into(),
                },
                field: "modalities",
            });
        }
        // Codex Round 5 #1 (critical): F16-KV / `force_prefill_only`
        // collision. `verify_batched_from_state` (and our
        // `prefill_one_from_state`) set `force_prefill_only=true`,
        // which inside `run_generate` forces `skip_decode=true`. That
        // routes the call through the batch-prefill block — which
        // unconditionally casts non-NVFP4 layers to FP8 KV (no F16
        // prefill kernel exists). On a per-layer F16-KV configuration
        // that writes FP8 bytes into F16-shaped slots; subsequent
        // shadow-KV reads then dequant FP8 as F16, silently
        // corrupting the drafter's cross-attention.
        //
        // Hard-reject up front until either a real F16 suffix-prefill
        // kernel lands or the verify path stops piggybacking on
        // `run_generate`. Single F16 layer is enough to taint the
        // session: the drafter cross-attends to specific source
        // layers (sliding + full), and either source landing on F16
        // gets corrupted by the FP8 cast.
        {
            let mut has_f16 = false;
            for l in 0..self.arch.num_hidden_layers {
                let d = crate::gemma4_layer_exec::KvDtype::for_layer_index_or_env(
                    self.arch.layer_types[l], l, false);
                if matches!(d, crate::gemma4_layer_exec::KvDtype::F16) {
                    has_f16 = true;
                    break;
                }
            }
            if has_f16 {
                return Err(rvllm_core::RvllmError::Config {
                    err: rvllm_core::ConfigError::InvalidField {
                        name: "kv_dtype",
                        reason: "Gemma 4 speculative decode requires \
                                 FP8 or NVFP4 KV at every layer — the \
                                 batched-verify path routes through \
                                 batch-prefill, which has no F16 KV \
                                 kernel and would silently corrupt \
                                 F16 slots. Set RVLLM_F16_KV=0 or use \
                                 an NVFP4/FP8 profile, or disable \
                                 RVLLM_GEMMA4_SPEC_DECODE.".into(),
                    },
                    field: "kv_dtype",
                });
            }
        }

        let spec_cfg = SpecDecodeRequestConfig::from_env();
        // Codex Round 2 #6: hard-reject anything that isn't strict
        // greedy in the new session path. RVLLM_GEMMA4_SPEC_TYPICAL=1
        // is parked until the typical-acceptance rejection math is
        // wired against the K-row base logits — exposing it under
        // half-implemented semantics produces misleading accept-rate
        // numbers. Until then any non-greedy request errors out
        // instead of silently degrading to greedy match.
        if !matches!(sampling, SamplingConfig::Greedy) {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "sampling",
                    reason: "session-loop spec decode is greedy-only \
                             (typical-acceptance parked pending re-impl \
                             against the K-row base logits)".into(),
                },
                field: "sampling",
            });
        }
        if spec_cfg.typical_mode {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "RVLLM_GEMMA4_SPEC_TYPICAL",
                    reason: "typical-acceptance is parked in the session \
                             loop; unset RVLLM_GEMMA4_SPEC_TYPICAL or use \
                             RVLLM_GEMMA4_SPEC_LEGACY_DEBUG=1 to opt into \
                             the legacy single-step path".into(),
                },
                field: "RVLLM_GEMMA4_SPEC_TYPICAL",
            });
        }
        let typical_mode = false;

        let k = spec_k as usize;
        let arch = &self.arch;
        let hidden_u = arch.hidden_size as u32;
        let stream = self.stream.raw();

        // ---- Warmup: prefill the prompt + capture base_last_hidden ----
        // Codex Round 5 #3: hook RAII for the warmup. Without this,
        // a run_generate error before final_norm leaves
        // `base_last_hidden_snapshot_pending` armed for the next
        // request, which would then overwrite its hidden buffer with
        // a wrong-position snapshot. Defensive disarm at entry covers
        // any stale flag from an earlier crashed request too.
        self.init_prefix_cache()?;
        {
            use std::sync::atomic::Ordering::Release;
            self.base_last_hidden_snapshot_pending.store(false, Release);
            self.force_common_prefix_override.store(u32::MAX, Release);
            self.skip_prefix_cache_publish.store(false, Release);
            self.force_prefill_only.store(false, Release);
            self.base_last_k_snapshot_pending.store(false, Release);
        }
        let warmup_base = {
            let warmup_guard = SpecHookGuard::new(self);
            let _ = &warmup_guard; // guard disarms snapshot-pending on drop
            self.base_last_hidden_snapshot_pending
                .store(true, std::sync::atomic::Ordering::Release);
            self.run_generate(
                fn_embed,
                fn_argmax,
                prompt_ids,
                /* max_new */ 1,
                eos_ids,
                /* shadow_requested */ false,
                SamplingConfig::Greedy,
                cancel,
                None,
                vision_splice,
                audio_splice,
            )?
            // warmup_guard drops -> all spec hooks disarmed regardless
            // of error path inside run_generate.
        };

        let mut emitted: Vec<u32> = Vec::with_capacity(max_new);
        let first_base = match warmup_base.first().copied() {
            Some(t) => t,
            None => return Ok(emitted),
        };
        if eos_ids.contains(&first_base) {
            // Codex Round 4 #3: parity with the non-spec decode path
            // (`if eos_ids.contains(&next_id) { break; }` BEFORE push)
            // — EOS is consumed silently; never returned in the
            // emitted vec or bumped against completion_tokens.
            return Ok(emitted);
        }
        // Note: the warmup's first decode token is NOT emitted yet;
        // it lives in session.next_base_argmax and either gets accepted
        // (= emitted as drafts[0] in iter 0) or becomes the bonus on
        // accept_len == 0.

        // ---- Open session ----
        let base_last_hidden = self
            .base_last_hidden_ptr
            .load(std::sync::atomic::Ordering::Acquire);
        if base_last_hidden == 0 {
            return Err(rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    op: "spec_batched_session: base_last_hidden_ptr zero \
                         after warmup — ensure_drafter must pre-allocate it",
                    backend: "Gemma4SpecDecode",
                },
                ctx: rvllm_core::AttnCtx {
                    op: "run_generate_speculative_batched",
                    stream,
                    num_seqs: 1,
                    head_dim: arch.max_head_dim() as u32,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let mut session = SpecDecodeSession::open(prompt_ids, base_last_hidden);
        session.seed_from_warmup(prompt_ids.len() as u32, first_base);

        // ---- Loop-invariant KV layout (mirror of run_drafter_k_from_state) ----
        let (kv_base_ptr, kv_scale_base_ptr,
             kv_layer_offsets, kv_scale_layer_offsets,
             num_blocks_total, block_size) = {
            let pc_guard = self.prefix_cache.lock().unwrap();
            let pc = pc_guard.as_ref().ok_or_else(|| {
                rvllm_core::RvllmError::Attention {
                    err: rvllm_core::AttentionError::FeatureNotAvailable {
                        op: "spec_batched_session: prefix cache empty \
                             after warmup",
                        backend: "Gemma4SpecDecode",
                    },
                    ctx: rvllm_core::AttnCtx {
                        op: "run_generate_speculative_batched",
                        stream,
                        num_seqs: 1,
                        head_dim: arch.max_head_dim() as u32,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                }
            })?;
            (
                pc.kv_cache_ptr,
                pc.kv_scale_ptr,
                pc.kv_layer_offsets.clone(),
                pc.kv_scale_layer_offsets.clone(),
                pc.num_blocks_total,
                pc.block_size,
            )
        };
        let sliding_blocks = num_blocks_total;
        let mut kv_dtype_per_layer: Vec<crate::gemma4_layer_exec::KvDtype> =
            Vec::with_capacity(arch.num_hidden_layers);
        for l in 0..arch.num_hidden_layers {
            kv_dtype_per_layer.push(
                crate::gemma4_layer_exec::KvDtype::for_layer_index_or_env(
                    arch.layer_types[l], l, false));
        }
        let sources = self.assistant_kv_sources.expect("checked above");
        let sliding_li = sources.sliding_source_layer as usize;
        let full_li = sources.full_source_layer as usize;

        // Compute per-layer K/V pointers + scale pointers for the
        // sliding + full source layers (used by populate_shadow_kv_range).
        let compute_view = |layer_idx: u32| -> (u64, u64, u64, u64) {
            let li = layer_idx as usize;
            let off = kv_layer_offsets[li];
            let scale_off = kv_scale_layer_offsets[li];
            let is_global = arch.layer_types[li]
                == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
            let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
            let nkvh = arch.num_kv_heads_for_layer(li) as u32;
            let hd = arch.head_dim_for_layer(li) as u32;
            let layer_elems = 2u64 * layer_blocks as u64
                * block_size as u64 * nkvh as u64 * hd as u64;
            let dtype = kv_dtype_per_layer[li];
            let k_v_half_bytes = match dtype {
                crate::gemma4_layer_exec::KvDtype::F16 => layer_elems,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_elems / 2,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 4,
            };
            let scale_half_slots =
                layer_blocks as u64 * block_size as u64 * nkvh as u64;
            let scale_half_bytes = match dtype {
                crate::gemma4_layer_exec::KvDtype::F16 => 0,
                crate::gemma4_layer_exec::KvDtype::Fp8 => scale_half_slots * 4,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 32,
            };
            let k_cache = kv_base_ptr + off;
            let v_cache = k_cache + k_v_half_bytes;
            let (k_scale, v_scale) = if dtype
                == crate::gemma4_layer_exec::KvDtype::F16
            {
                (0u64, 0u64)
            } else {
                let k_s = kv_scale_base_ptr + scale_off;
                (k_s, k_s + scale_half_bytes)
            };
            (k_cache, v_cache, k_scale, v_scale)
        };
        let (sliding_k, sliding_v, sliding_ks, sliding_vs) =
            compute_view(sources.sliding_source_layer);
        let (full_k, full_v, full_ks, full_vs) =
            compute_view(sources.full_source_layer);

        let (shadow_sliding_bytes, shadow_full_bytes) = {
            let guard = self.drafter.lock().unwrap();
            let d = guard.as_ref().expect("checked above");
            let s = d.shadow_kv.expect("shadow_kv attached");
            (s.sliding_layer_bytes, s.full_layer_bytes)
        };

        // ---- Initial shadow KV populate (prompt slots) ----
        {
            let guard = self.drafter.lock().unwrap();
            let d = guard.as_ref().expect("checked above");
            d.populate_shadow_kv_range_from_base(
                sliding_k, sliding_v,
                full_k, full_v,
                sliding_ks, sliding_vs,
                full_ks, full_vs,
                kv_dtype_per_layer[sliding_li],
                kv_dtype_per_layer[full_li],
                shadow_sliding_bytes,
                shadow_full_bytes,
                /* slot_start */ 0,
                /* slot_count */ prompt_ids.len() as u32,
                stream,
            )?;
        }

        // ---- K-hidden capture buffer (pre-allocated by ensure_drafter) ----
        let k_hidden_buf = self
            .base_last_k_hidden_ptr
            .load(std::sync::atomic::Ordering::Acquire);
        if k_hidden_buf == 0 {
            return Err(rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    op: "spec_batched_session: base_last_k_hidden_ptr zero \
                         — ensure_drafter must pre-allocate K-capture buffer",
                    backend: "Gemma4SpecDecode",
                },
                ctx: rvllm_core::AttnCtx {
                    op: "run_generate_speculative_batched",
                    stream,
                    num_seqs: 1,
                    head_dim: arch.max_head_dim() as u32,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }

        let mut total_drafted: u32 = 0;
        let mut total_accepted: u32 = 0;
        let mut iter_count: u32 = 0;
        let mut base_argmax_k: Vec<u32> = Vec::with_capacity(k);

        // ---- Main session loop ----
        'outer: while emitted.len() < max_new {
            if let Some(c) = cancel {
                if c.load(std::sync::atomic::Ordering::Relaxed) { break; }
            }

            // Step 1: K drafter forwards.
            let drafts = self.run_drafter_k_from_state(
                fn_embed, &session, spec_k, sampling, typical_mode,
            )?;
            if drafts.tokens.is_empty() { break; }
            let k_actual = drafts.tokens.len();
            total_drafted = total_drafted.saturating_add(k_actual as u32);

            // Step 2: batched verify -> K base argmaxes + K hiddens.
            base_argmax_k.clear();
            base_argmax_k.resize(k_actual, 0u32);
            self.verify_batched_from_state(
                fn_embed,
                &drafts.tokens,
                session.committed_len,
                &session,
                k_hidden_buf,
                &mut base_argmax_k,
            )?;

            // Step 3: greedy accept_len.
            // (Typical-acceptance: parked — falls back to greedy here.
            // Mirror the legacy path before re-enabling.)
            let mut accept_len: usize = 0;
            for i in 0..k_actual {
                let base_pred = if i == 0 {
                    session.next_base_argmax
                } else {
                    base_argmax_k[i - 1]
                };
                if drafts.tokens[i] == base_pred {
                    accept_len += 1;
                } else {
                    break;
                }
            }
            // Codex Round 7 #5: count accepted only after we've actually
            // emitted them. Previously total_accepted was incremented up
            // front, so a max_new / on_token mid-emit abort over-reported
            // X-RVLLM-Accept-Rate.
            let verify_accepted = accept_len;

            let old_committed = session.committed_len;

            if accept_len > 0 {
                // Emit accepted drafts.
                let mut emitted_accepted: usize = 0;
                for i in 0..accept_len {
                    if emitted.len() >= max_new {
                        total_accepted = total_accepted.saturating_add(emitted_accepted as u32);
                        break 'outer;
                    }
                    let tok = drafts.tokens[i];
                    // Codex Round 4 #3: EOS-before-push parity with
                    // the non-spec decode path. The special token is
                    // consumed silently — never pushed into `emitted`
                    // and never delivered via on_token.
                    if eos_ids.contains(&tok) {
                        total_accepted = total_accepted.saturating_add(emitted_accepted as u32);
                        break 'outer;
                    }
                    emitted.push(tok);
                    emitted_accepted += 1;
                    if let Some(cb) = on_token.as_mut() {
                        if !cb(tok) {
                            total_accepted = total_accepted.saturating_add(emitted_accepted as u32);
                            break 'outer;
                        }
                    }
                }
                total_accepted = total_accepted.saturating_add(emitted_accepted as u32);
                let _ = verify_accepted; // verify_accepted == emitted_accepted on normal path

                // Codex Round 7 #2: switch from deferred-bonus to
                // CLASSICAL spec-decode emission. We now ALSO emit
                // base_argmax_k[accept_len - 1] (the bonus / divergence
                // token) this iter and commit its base K/V via
                // prefill_one_from_state. Net per-iter:
                //   cost  = K (verify) + 1 (bonus commit) = K + 1
                //   emit  = accept_len + 1
                //   cost/tok = (K + 1) / (accept_len + 1)
                // vs deferred-bonus per-iter:
                //   cost  = K (verify),   emit = accept_len  →
                //   cost/tok = K / accept_len
                // Classical is strictly faster at accept_len ∈ (0, K)
                // and equal at the boundaries. The K-hidden[accept_len -
                // 1] DtoD into base_last_hidden_ptr is replaced by the
                // bonus prefill's own POST-final-norm hidden snapshot,
                // which gives the next iter's drafter the hidden at the
                // ACTUALLY-emitted last token (i.e. the bonus position),
                // not the last-accepted-draft position.
                let bonus = base_argmax_k[accept_len - 1];

                // Commit the accepted prefix to the session now (without
                // changing next_base_argmax — we set that after the
                // bonus prefill below). We need to commit BEFORE
                // prefill_one_from_state so its input = session.tokens +
                // [bonus] has the right length for the override.
                session.commit_drafts(&drafts.tokens, accept_len, session.next_base_argmax);

                // Incremental shadow KV for the newly-committed slots
                // [old_committed, old_committed + accept_len).
                {
                    let guard = self.drafter.lock().unwrap();
                    let d = guard.as_ref().expect("drafter resident");
                    d.populate_shadow_kv_range_from_base(
                        sliding_k, sliding_v,
                        full_k, full_v,
                        sliding_ks, sliding_vs,
                        full_ks, full_vs,
                        kv_dtype_per_layer[sliding_li],
                        kv_dtype_per_layer[full_li],
                        shadow_sliding_bytes,
                        shadow_full_bytes,
                        /* slot_start */ old_committed,
                        /* slot_count */ accept_len as u32,
                        stream,
                    )?;
                }

                // Emit the bonus token (= base's prediction at position
                // old_committed + accept_len, computed by verify pass).
                if emitted.len() >= max_new { break 'outer; }
                if eos_ids.contains(&bonus) { break 'outer; }
                emitted.push(bonus);
                if let Some(cb) = on_token.as_mut() {
                    if !cb(bonus) { break 'outer; }
                }
                if emitted.len() >= max_new { break 'outer; }

                // Commit the bonus's base K/V + capture its
                // POST-final-norm hidden for the next iter's drafter.
                let new_next = self.prefill_one_from_state(
                    fn_embed, &mut session, bonus)?;
                session.next_base_argmax = new_next;

                // Shadow KV for the bonus's slot.
                let bonus_slot = old_committed + accept_len as u32;
                let guard = self.drafter.lock().unwrap();
                let d = guard.as_ref().expect("drafter resident");
                d.populate_shadow_kv_range_from_base(
                    sliding_k, sliding_v,
                    full_k, full_v,
                    sliding_ks, sliding_vs,
                    full_ks, full_vs,
                    kv_dtype_per_layer[sliding_li],
                    kv_dtype_per_layer[full_li],
                    shadow_sliding_bytes,
                    shadow_full_bytes,
                    /* slot_start */ bonus_slot,
                    /* slot_count */ 1,
                    stream,
                )?;
            } else {
                // accept_len == 0: emit the deferred bonus
                // (= session.next_base_argmax), then prefill it so its
                // base KV exists for the next iter.
                let bonus = session.next_base_argmax;
                if bonus == u32::MAX {
                    // Defensive: warmup didn't seed; cannot make
                    // progress.
                    break;
                }
                // Codex Round 4 #3: EOS-before-push parity. The
                // deferred bonus is consumed silently if it's a stop
                // token — no push, no on_token.
                if eos_ids.contains(&bonus) { break; }
                emitted.push(bonus);
                if let Some(cb) = on_token.as_mut() {
                    if !cb(bonus) { break; }
                }
                if emitted.len() >= max_new { break; }

                let new_next = self.prefill_one_from_state(
                    fn_embed, &mut session, bonus)?;
                session.next_base_argmax = new_next;

                // Shadow KV update for the single new committed slot.
                let guard = self.drafter.lock().unwrap();
                let d = guard.as_ref().expect("drafter resident");
                d.populate_shadow_kv_range_from_base(
                    sliding_k, sliding_v,
                    full_k, full_v,
                    sliding_ks, sliding_vs,
                    full_ks, full_vs,
                    kv_dtype_per_layer[sliding_li],
                    kv_dtype_per_layer[full_li],
                    shadow_sliding_bytes,
                    shadow_full_bytes,
                    /* slot_start */ old_committed,
                    /* slot_count */ 1,
                    stream,
                )?;
            }

            iter_count = iter_count.saturating_add(1);
        }

        *self.last_spec_stats.lock().unwrap() = Some(LastSpecStats {
            drafted: total_drafted,
            accepted: total_accepted,
            cumulative_decoded: emitted.len() as u32,
        });
        // Codex Round 2 #5: explicit deferred-bonus policy.
        //
        // For accept_len > 0 this loop emits ONLY drafts[..accept_len]
        // per iteration; the verify pass's `base_argmax_k[accept_len -
        // 1]` token (the classical spec-decode "bonus") is parked in
        // `session.next_base_argmax` and emitted the next iter (either
        // accepted as that iter's drafts[0] or surfaced via the
        // accept_len == 0 branch). This keeps the algorithm correct
        // without writing a base-K/V slot for a token base never saw
        // through the layer stack, but it costs the textbook
        // "accepted + 1" per-iter speedup. Throughput logged as
        // `accepted_per_verify`, NOT `accepted_plus_bonus`, so
        // measurements are honest.
        let accepted_per_verify = if iter_count > 0 {
            total_accepted as f32 / iter_count as f32
        } else { 0.0 };
        tracing::info!(
            iter_count,
            total_drafted,
            total_accepted,
            accepted_per_verify,
            emitted = emitted.len(),
            max_new,
            prompt_len = prompt_ids.len(),
            spec_k,
            "Gemma 4 batched speculative session loop complete \
             (deferred-bonus policy: bonus accounted to next iter)",
        );
        let _ = (k, max_new, sampling);
        Ok(emitted)
    }

    /// Commit 37 — task #2 final landing.
    ///
    /// Iterative outer wrapper around `run_generate_speculative`.
    /// Calls the existing single-step speculative function in a loop,
    /// extending the prompt by `accept_len + 1` tokens per iteration.
    /// The internal `run_generate(max_new = spec_k + 1)` call in each
    /// iteration's body uses the prefix-cache to skip re-prefilling
    /// the unchanged prompt prefix; only the few newly committed
    /// tokens trigger fresh prefill on each iteration.
    ///
    /// This delivers the CORRECT iterative spec-decode structure:
    /// per-iter K drafter forwards + K+1 base decodes + emit
    /// accept_len+1 tokens. Per-iter wall time:
    ///   - 1 prefill of ~accept_len+1 tokens (cheap via prefix cache)
    ///   - K+1 sequential base decodes (~30-40 ms × (K+1) on E4B)
    ///   - K drafter forwards (~150 μs × K)
    ///
    /// No wall-clock SPEEDUP until the inner `run_generate(K+1)` is
    /// replaced with a single batched-verify call producing K logit
    /// rows in one base prefill — but that swap is now a localized
    /// surgery inside `run_generate_speculative`, not a full
    /// architectural rewrite. The outer loop is sound.
    ///
    /// Gated by `RVLLM_GEMMA4_SPEC_ITERATIVE=1`. Default off — the
    /// non-iterative single-step path remains the primary entry to
    /// avoid regressing any in-flight tests.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_generate_speculative_iterative(
        &self,
        fn_embed: rvllm_kernels::KernelFn,
        fn_argmax: rvllm_kernels::KernelFn,
        prompt_ids: &[u32],
        max_new: usize,
        eos_ids: &[u32],
        spec_k: u32,
        sampling: SamplingConfig,
        cancel: Option<&std::sync::atomic::AtomicBool>,
        mut on_token: Option<&mut dyn FnMut(u32) -> bool>,
        vision_splice: &[(usize, &[u8])],
        audio_splice: &[(usize, &[u8])],
    ) -> Result<Vec<u32>> {
        if max_new == 0 || spec_k == 0 {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "max_new/spec_k",
                    reason: "must be >= 1".into(),
                },
                field: "max_new",
            });
        }

        // Each iter emits at most `spec_k + 1` tokens (accept_len up
        // to spec_k, plus one bonus base token at divergence).
        let max_emit_per_iter = (spec_k as usize) + 1;

        let mut emitted: Vec<u32> = Vec::with_capacity(max_new);
        let mut current_prompt: Vec<u32> = prompt_ids.to_vec();
        let mut iter_count: u32 = 0;
        let mut total_drafted: u32 = 0;
        let mut total_accepted: u32 = 0;

        // Phase D-2: force emit-accepted on each inner call via the
        // per-bringup atomic flag (replaces env::set_var scribble).
        // Re-armed at the top of each loop body since the inner
        // function consumes the flag with swap(false).
        let mut result = Ok::<Vec<u32>, rvllm_core::RvllmError>(Vec::new());
        'outer: while emitted.len() < max_new {
            // Cancel check between iterations — cheap atomic load.
            if let Some(c) = cancel {
                if c.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
            }
            let want = (max_new - emitted.len()).min(max_emit_per_iter);
            let inner_max_new = want.min(max_emit_per_iter);

            // Per-iteration spec call: returns accept_len + 1 tokens
            // (or 1 token if no acceptance). Vision/audio splice only
            // legal on the FIRST iteration (the inner function still
            // rejects them; we pass empty after iter 0).
            let (vs, as_) = if iter_count == 0 {
                (vision_splice, audio_splice)
            } else {
                (&[] as &[(usize, &[u8])], &[] as &[(usize, &[u8])])
            };

            // The on_token callback only fires from base's decode loop;
            // pass `None` to avoid double-emission. We'll deliver
            // accepted tokens to the worker via the return value.
            // Phase D-2: arm force_emit_accepted before each inner call.
            self.force_emit_accepted
                .store(true, std::sync::atomic::Ordering::Release);
            let chunk = match self.run_generate_speculative(
                fn_embed,
                fn_argmax,
                &current_prompt,
                inner_max_new,
                eos_ids,
                spec_k,
                sampling,
                cancel,
                None,
                vs,
                as_,
            ) {
                Ok(v) => v,
                Err(e) => {
                    result = Err(e);
                    break 'outer;
                }
            };
            if chunk.is_empty() {
                // Defensive: nothing emitted — break to avoid infinite loop.
                break;
            }
            if let Some(stats) = self.take_last_spec_stats() {
                total_drafted = total_drafted.saturating_add(stats.drafted);
                total_accepted = total_accepted.saturating_add(stats.accepted);
            }

            // Deliver each emitted token to the streaming callback
            // and accumulate; honor EOS + cancel.
            for &tok in &chunk {
                if emitted.len() >= max_new { break 'outer; }
                if eos_ids.contains(&tok) {
                    emitted.push(tok);
                    break 'outer;
                }
                emitted.push(tok);
                if let Some(cb) = on_token.as_mut() {
                    if !cb(tok) { break 'outer; }
                }
            }
            // Extend prompt with what we just emitted so the next
            // iter's prefix-cache lookup hits the right state.
            current_prompt.extend_from_slice(&chunk);
            iter_count = iter_count.saturating_add(1);
        }
        // Phase D-2: defensive — clear the atomic in case we
        // bailed out of the loop without consuming an armed flag.
        self.force_emit_accepted
            .store(false, std::sync::atomic::Ordering::Release);
        // Refresh aggregated stats for header / event emission.
        *self.last_spec_stats.lock().unwrap() = Some(LastSpecStats {
            drafted: total_drafted,
            accepted: total_accepted,
            cumulative_decoded: emitted.len() as u32,
        });
        tracing::info!(
            iter_count,
            total_drafted,
            total_accepted,
            emitted = emitted.len(),
            max_new,
            "Gemma 4 speculative iterative wrapper complete"
        );
        result?;
        Ok(emitted)
    }

    pub unsafe fn run_generate_speculative(
        &self,
        fn_embed: rvllm_kernels::KernelFn,
        _fn_argmax: rvllm_kernels::KernelFn,
        prompt_ids: &[u32],
        max_new: usize,
        _eos_ids: &[u32],
        spec_k: u32,
        sampling: SamplingConfig,
        _cancel: Option<&std::sync::atomic::AtomicBool>,
        mut _on_token: Option<&mut dyn FnMut(u32) -> bool>,
        vision_splice: &[(usize, &[u8])],
        audio_splice: &[(usize, &[u8])],
    ) -> Result<Vec<u32>> {
        if max_new == 0 {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "max_new",
                    reason: "must be >= 1; max_new=0 underflows the decode loop".into(),
                },
                field: "max_new",
            });
        }
        if spec_k == 0 {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "RVLLM_GEMMA4_SPEC_K",
                    reason: "must be >= 1".into(),
                },
                field: "RVLLM_GEMMA4_SPEC_K",
            });
        }
        // Commit 40: batched-verify hot path. When on, the warmup
        // base prefill is dropped to max_new=1 (we don't need K
        // sequential base decodes — verify happens via lm_head on
        // K captured hiddens from a SECOND base call that batched-
        // prefills the K drafts). Without this gate the legacy
        // sequential-decode verify runs (= K+1 base decodes).
        // Commit 52 (Phase D): consolidated single-read of all spec
        // env knobs at function entry. Codex review priority 6 — no
        // more env reads in the K-prefix verify loop (per-row
        // RVLLM_GEMMA4_SPEC_ACCEPT_BIAS was redundant overhead).
        //
        // Commit 53 (Phase D-2): also consume the
        // `force_emit_accepted` atomic flag set by the outer wrappers
        // (replaces the env::set_var scribble pattern codex priority
        // 2 called out).
        let mut spec_cfg = SpecDecodeRequestConfig::from_env();
        if self
            .force_emit_accepted
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            spec_cfg.emit_accepted = true;
        }
        // Commit 55 (codex priority 0.1): force-batched override
        // from the worker. Note: NOT swap-consume because the same
        // request may call run_generate_speculative multiple times
        // via the outer wrapper's loop, and we want every iter to
        // see batched_verify_mode = true. The flag is cleared at
        // request end by the worker (or implicitly reset on next
        // request since the worker always re-arms).
        if self
            .force_batched_verify
            .load(std::sync::atomic::Ordering::Acquire)
        {
            spec_cfg.batched_verify_mode = true;
        }
        let spec_cfg = spec_cfg;
        let batched_verify_mode = spec_cfg.batched_verify_mode;
        let typical_mode = spec_cfg.typical_mode;
        if !matches!(sampling, SamplingConfig::Greedy) && !typical_mode {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "sampling",
                    reason: "Gemma 4 speculative decode is greedy-only unless RVLLM_GEMMA4_SPEC_TYPICAL=1".into(),
                },
                field: "sampling",
            });
        }
        if !vision_splice.is_empty() || !audio_splice.is_empty() {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "modalities",
                    reason: "Gemma 4 speculative decode is text-only until assistant one-step parity is validated".into(),
                },
                field: "modalities",
            });
        }
        if self.assistant_kv_sources.is_none() {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "model",
                    reason: "assistant drafter requires an E4B-style Gemma 4 model with shared-KV source layers".into(),
                },
                field: "model",
            });
        }
        if self.drafter.lock().unwrap().is_none() {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "drafter",
                    reason: "drafter is not resident; call ensure_drafter before run_generate_speculative".into(),
                },
                field: "drafter",
            });
        }

        if prompt_ids.is_empty() {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "prompt_ids",
                    reason: "speculative decode requires a non-empty prompt".into(),
                },
                field: "prompt_ids",
            });
        }

        // Commit 16: REAL base prefill. Drive the existing
        // `run_generate` machinery for `max_new = 1` (prefix-cache
        // initialised, drafter dir / shadow attached). This:
        //
        //   1. fills `prefix_cache.kv_cache_ptr` with REAL K/V
        //      across all 42 base layers (including the source
        //      layers 22/23 the drafter cross-attends to).
        //   2. via the new commit-16 DtoD hook in `run_generate`,
        //      captures the normalized pre-lm-head hidden of the
        //      last prompt token into `self.base_last_hidden_ptr`.
        //
        // We then read both back here, build the drafter inputs on
        // REAL data, and run one MTP step. K-step draft loop +
        // verify + acceptance still pending below.
        let arch = &self.arch;
        let hidden_u = arch.hidden_size as u32;
        let vocab_u = arch.vocab_size as u32;
        let stream = self.stream.raw();
        let arena = &self.arena;

        // Step (a): init prefix cache (idempotent — no-op after
        // first call).
        self.init_prefix_cache()?;
        // Commit 40: snapshot arena BEFORE the warmup so the
        // batched-verify branch (much later in this function)
        // can reclaim warmup + drafter K-loop scratch before
        // calling run_generate a SECOND time. The bonus_tok_vec
        // and emitted Vec are host-side; the K-buffer is above
        // scratch. So restoring to this checkpoint is safe.
        let arena_ck_at_entry = self.arena.checkpoint();

        // Commit 18: arm the one-shot base_last_hidden snapshot
        // BEFORE calling run_generate. The hook inside run_generate
        // consumes this flag on the FIRST final_norm fire (= after
        // prefill, last prompt token), and ignores subsequent
        // decode-step fires. Without this gate, max_new>1 left the
        // captured buffer holding the wrong position's hidden and
        // dragged accept_rate to ~0.
        self.base_last_hidden_snapshot_pending
            .store(true, std::sync::atomic::Ordering::Release);

        // Commit 26 + 29: activate per-step logits capture if either
        // lossy-greedy OR typical-acceptance (commit 29 quick-check)
        // is enabled. Allocates max_new * vocab f32; capture happens
        // inside run_generate at the prefill + decode-loop argmax
        // sites.
        // Phase D: lossy_threshold sourced from SpecDecodeRequestConfig.
        let lossy_threshold: f32 = spec_cfg.lossy_threshold;
        let want_capture = lossy_threshold > 0.0 || typical_mode;
        if want_capture {
            let vocab_sz = arch.vocab_size as usize;
            let needed = max_new.saturating_mul(vocab_sz);
            let mut buf = self.spec_decode_step_logits.lock().unwrap();
            if buf.len() < needed {
                buf.resize(needed, 0.0);
            } else {
                for v in buf.iter_mut().take(needed) { *v = 0.0; }
            }
            drop(buf);
            self.spec_logits_capture_active
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }

        // Step (b): run base prefill. `max_new=1` is the smallest
        // legal value (run_generate rejects 0). The one decode
        // step costs ~30 ms; the resulting token is discarded.
        // No vision / audio splice. Greedy. No on_token / cancel.
        //
        // Commit 40: in batched-verify mode, only need the warmup
        // to capture base_last_hidden (drafter step 1 input). The
        // K verify tokens come from lm_head on captured K hiddens
        // after a SECOND run_generate call below. So drop the
        // warmup decodes to 1 — saves K sequential decodes per
        // outer iter (= the speedup gate).
        let warmup_max_new = if batched_verify_mode { 1 } else { max_new };
        // Commit 43: skip-warmup path. When the previous outer iter's
        // batched-verify branch armed `skip_next_warmup` (only fires
        // when accept_len >= 1 AND RVLLM_GEMMA4_SPEC_SKIP_WARMUP=1):
        //   - bypass the warmup run_generate entirely (saves one base
        //     prefill per iter, the headline speedup)
        //   - base_last_hidden_ptr was DtoD-populated by the prev iter
        //     from K-buffer[accept_len - 1] (POST-final-norm hidden of
        //     the last accepted draft)
        //   - _base_first_tok[0] is filled from saved_warmup_b_p (=
        //     prev iter's base_argmax_K[accept_len - 1] = base's
        //     prediction at the would-be warmup position).
        // Commit 51 (Phase C-2): drop the RVLLM_GEMMA4_SPEC_SKIP_WARMUP
        // env gate. When batched_verify_mode is on, honor the
        // `skip_next_warmup` atomic unconditionally. The atomic is
        // only set by code we control:
        //   (a) the batched-verify branch at end of prev iter, when
        //       accept_len >= 1 (saved_warmup_b_p = base_argmax_K[
        //       accept_len - 1], last_base_hidden DtoD-copied from
        //       K-buffer[accept_len - 1]).
        //   (b) the OUTER wrapper run_generate_speculative_batched
        //       before each inner call (this commit — arms from
        //       outer's own warmup so the first inner call also
        //       skips its redundant warmup).
        //
        // accept_len = 0 iters from (a) leave the atomic at false, so
        // the inner warmup runs normally — that's the forward-progress
        // fallback codex priority 4 documented. No regression at
        // accept_len = 0; saving comes from accept_len >= 1 iters +
        // the first iter (always saved).
        let skip_warmup_active = batched_verify_mode
            && self
                .skip_next_warmup
                .swap(false, std::sync::atomic::Ordering::AcqRel);
        // Commit 50 (Phase C-1): also skip prefix-cache publish on
        // the warmup. The warmup's prompt-length K/V write is per-
        // request work; publishing committed_prefix_len = floor(P /
        // 2048) * 2048 for short prompts (= 0 for ~30-token chats)
        // is what triggers the verify-call full re-prefill loop the
        // codex review priority-1 flagged. Skipping here means the
        // cross-request cache only updates on NON-spec requests.
        if batched_verify_mode {
            self.skip_prefix_cache_publish
                .store(true, std::sync::atomic::Ordering::Release);
        }
        let _base_first_tok: Vec<u32> = if skip_warmup_active {
            let cached = self
                .saved_warmup_b_p
                .swap(u32::MAX, std::sync::atomic::Ordering::AcqRel);
            if cached == u32::MAX {
                // Sentinel reached without a real value — fall through
                // to the normal warmup path. Defensive; should not fire
                // because the flag-and-value are armed together.
                self.run_generate(
                    fn_embed,
                    self.fused.fn_argmax,
                    prompt_ids,
                    warmup_max_new,
                    _eos_ids,
                    /* shadow_requested */ false,
                    SamplingConfig::Greedy,
                    _cancel,
                    _on_token.take(),
                    /* vision_splice */ &[],
                    /* audio_splice */ &[],
                )?
            } else {
                vec![cached]
            }
        } else {
            self.run_generate(
                fn_embed,
                self.fused.fn_argmax,
                prompt_ids,
                warmup_max_new,
                _eos_ids,
                /* shadow_requested */ false,
                SamplingConfig::Greedy,
                _cancel,
                _on_token.take(),
                /* vision_splice */ &[],
                /* audio_splice */ &[],
            )?
        };

        // Step (c): pull real KV layout + pointers out of the
        // session prefix cache and the per-layer dtype rule (same
        // rule run_generate uses).
        let base_hidden_last_step_ptr = self
            .base_last_hidden_ptr
            .load(std::sync::atomic::Ordering::Acquire);
        if base_hidden_last_step_ptr == 0 {
            return Err(rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    op: "run_generate_speculative: base_last_hidden_ptr \
                         is zero after run_generate — ensure_drafter \
                         must allocate it (spec_decode gate enabled?)",
                    backend: "Gemma4SpecDecode",
                },
                ctx: rvllm_core::AttnCtx {
                    op: "run_generate_speculative",
                    stream,
                    num_seqs: 1,
                    head_dim: self.arch.max_head_dim() as u32,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let (kv_base_ptr, kv_scale_base_ptr,
             kv_layer_offsets, kv_scale_layer_offsets,
             num_blocks_total, block_size, max_blocks_per_seq) = {
            let pc_guard = self.prefix_cache.lock().unwrap();
            let pc = pc_guard.as_ref().ok_or_else(|| rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    op: "run_generate_speculative: prefix cache not \
                         populated after run_generate",
                    backend: "Gemma4SpecDecode",
                },
                ctx: rvllm_core::AttnCtx {
                    op: "run_generate_speculative",
                    stream,
                    num_seqs: 1,
                    head_dim: self.arch.max_head_dim() as u32,
                },
                bt: std::backtrace::Backtrace::capture(),
            })?;
            (
                pc.kv_cache_ptr,
                pc.kv_scale_ptr,
                pc.kv_layer_offsets.clone(),
                pc.kv_scale_layer_offsets.clone(),
                pc.num_blocks_total,
                pc.block_size,
                pc.num_blocks_total,
            )
        };
        let sliding_blocks = num_blocks_total;

        // Per-layer dtype rule — same one the base allocator uses
        // (and the shadow KV expects).
        let mut kv_dtype_per_layer: Vec<crate::gemma4_layer_exec::KvDtype> =
            Vec::with_capacity(arch.num_hidden_layers);
        for l in 0..arch.num_hidden_layers {
            kv_dtype_per_layer.push(
                crate::gemma4_layer_exec::KvDtype::for_layer_index_or_env(
                    arch.layer_types[l], l, false));
        }

        // Persistent identity block-tables [0..num_blocks_total).
        let block_tables_region = arena.region(
            "spec_block_tables", (num_blocks_total as usize) * 4, 16)?;
        {
            let mut bt_host: Vec<u8> = Vec::with_capacity((num_blocks_total as usize) * 4);
            for b in 0..num_blocks_total {
                bt_host.extend_from_slice(&(b as i32).to_le_bytes());
            }
            block_tables_region.copy_from_host(&bt_host)?;
        }

        // Commit 18: position-align all three drafter inputs at
        // the LAST PROMPT TOKEN (position prompt_len - 1), so the
        // drafter predicts position prompt_len — directly
        // comparable to base's first argmax `_base_first_tok[0]`.
        //
        //   base_hidden_last_step ← snapshot (post-prefill,
        //                          pre-any-decode) = hidden at
        //                          position prompt_len - 1.
        //   last_token_embed      ← embed(prompt_ids.last()).
        //   step.position         ← prompt_len (where the drafter
        //                          predicts).
        //   verify target         ← _base_first_tok[0].
        //
        // Commit 18 (refined): bound context_lens to the PROMPT
        // only, not prompt + base's decoded tokens. The drafter is
        // predicting "what comes after the prompt", so its
        // cross-attention should see prompt K/V slots and nothing
        // else — matching the HF candidate_generator convention of
        // running the drafter on the last-prompt-token position.
        let ctx_len_val: i32 = prompt_ids.len() as i32;
        let context_lens_region = arena.region("spec_ctx_lens", 4, 16)?;
        context_lens_region.copy_from_host(&ctx_len_val.to_le_bytes())?;

        // last_token_embed = embed of the LAST PROMPT TOKEN (NOT
        // base's max_new'th decode). This is the input the drafter
        // pairs with `base_hidden_last_step` at position prompt_len
        // - 1 → predict position prompt_len.
        let last_committed_tok: u32 = *prompt_ids
            .last()
            .expect("prompt_ids non-empty checked above");
        let token_ids_region = arena.region("spec_tok_ids", 4, 16)?;
        token_ids_region.copy_from_host(&last_committed_tok.to_le_bytes())?;
        let last_token_embed = arena.region(
            "spec_last_tok_embed", (hidden_u as usize) * 2, 16)?;
        rvllm_fused::EmbeddingGatherLaunch { num_tokens: 1, hidden: hidden_u, vocab: vocab_u }
            .launch(
                fn_embed,
                last_token_embed.device_ptr(),
                self.model.embedding.offset_bytes,
                token_ids_region.device_ptr(),
                stream,
            )?;

        // base_hidden_last_step is no longer a stub — the run_generate
        // hook wrote it.
        let base_hidden_last_step_real = base_hidden_last_step_ptr;

        let sources = self.assistant_kv_sources
            .expect("guarded above by assistant_kv_sources.is_none()");
        let kv_dtype_root = crate::gemma4_layer_exec::KvDtype::from_env(false);

        let source_view = |layer_idx: u32| -> crate::gemma4_drafter::DrafterBaseKvView {
            let li = layer_idx as usize;
            let off = kv_layer_offsets[li];
            let scale_off = kv_scale_layer_offsets[li];
            let is_global = arch.layer_types[li]
                == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
            let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
            let nkvh = arch.num_kv_heads_for_layer(li) as u32;
            let hd = arch.head_dim_for_layer(li) as u32;
            let layer_elems =
                2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
            let dtype = kv_dtype_per_layer[li];
            let k_v_half_bytes = match dtype {
                crate::gemma4_layer_exec::KvDtype::F16 => layer_elems,       // *2 / 2
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_elems / 2,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 4,
            };
            let scale_half_slots =
                layer_blocks as u64 * block_size as u64 * nkvh as u64;
            let scale_half_bytes = match dtype {
                crate::gemma4_layer_exec::KvDtype::F16 => 0,
                crate::gemma4_layer_exec::KvDtype::Fp8 => scale_half_slots * 4,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 32,
            };
            let k_cache = kv_base_ptr + off;
            let v_cache = k_cache + k_v_half_bytes;
            let (k_scale_cache, v_scale_cache) = if dtype
                == crate::gemma4_layer_exec::KvDtype::F16
            {
                (0u64, 0u64)
            } else {
                let k_s = kv_scale_base_ptr + scale_off;
                (k_s, k_s + scale_half_bytes)
            };
            crate::gemma4_drafter::DrafterBaseKvView {
                k_cache,
                v_cache,
                k_scale_cache,
                v_scale_cache,
                q_scale_cache: 0,
                block_tables: block_tables_region.device_ptr(),
                context_lens: context_lens_region.device_ptr(),
                block_size,
                max_blocks_per_seq,
                num_blocks_total,
                kv_dtype: dtype,
            }
        };

        let sliding_kv = source_view(sources.sliding_source_layer);
        let full_kv = source_view(sources.full_source_layer);

        let _ = kv_dtype_root; // env snapshot for future logging.

        // Allocate workspace + build DrafterForwardStep. Position 0
        // is the placeholder position — commit 7 will thread the
        // real "last accepted position" through.
        let workspace = {
            let guard = self.drafter.lock().unwrap();
            let drafter = guard.as_ref().expect("checked above");
            drafter.alloc_step_workspace(arena)?
        };
        tracing::debug!(
            spec_k,
            workspace_bytes = workspace.bytes,
            "allocated Gemma 4 speculative drafter one-step workspace",
        );

        // Commit 18 (refined): HF candidate_generator.py:1370 uses
        // `position_ids = [[input_ids.shape[1] - 1]]` — the
        // drafter's RoPE position is the position OF the prompt's
        // last token (= prompt_len - 1), not the next slot. The
        // drafter's MTP head then maps that latent to "what comes
        // next" via `masked_embedding`, which is compared against
        // base's argmax `_base_first_tok[0]`.
        let drafter_position: u32 = (prompt_ids.len() as u32).saturating_sub(1);

        // Commit 23: K>1 chained drafter. Populate shadow KV ONCE
        // outside the iteration loop (it's a function of base K/V at
        // the source layers, which doesn't change across drafter
        // steps). Then loop spec_k times, feeding each step's
        // post_projection output back as the next step's
        // `base_hidden_last_step` and embed(D_{i-1}) as the next
        // `last_token_embed`. Per HF/vLLM Gemma4Assistant chaining
        // contract.
        let sliding_window = self.arch.sliding_window_size as i32;
        {
            let guard = self.drafter.lock().unwrap();
            let drafter = guard.as_ref().expect("checked above");
            let shadow = drafter.shadow_kv.expect(
                "shadow_kv attached by ensure_drafter when spec_decode is on"
            );
            let sliding_li = sources.sliding_source_layer as usize;
            let full_li = sources.full_source_layer as usize;
            let sliding_view = source_view(sources.sliding_source_layer);
            let full_view = source_view(sources.full_source_layer);
            // Commit 57 (codex #1): truncate populate work to the
            // current active-context slot count. Before: each spec
            // iteration dequanted the entire max-cache shadow
            // (RVLLM_NUM_BLOCKS * block_size slots = 32768 on
            // default config), most of which held zero base KV. Now:
            // only `prompt_ids.len()` slots are touched per iter.
            // For a 100-token chat prompt that's ~320× fewer
            // dequant work-elements per iter.
            let valid_len_slots = prompt_ids.len() as u32;
            drafter.populate_shadow_kv_from_base(
                sliding_view.k_cache,
                sliding_view.v_cache,
                full_view.k_cache,
                full_view.v_cache,
                sliding_view.k_scale_cache,
                sliding_view.v_scale_cache,
                full_view.k_scale_cache,
                full_view.v_scale_cache,
                kv_dtype_per_layer[sliding_li],
                kv_dtype_per_layer[full_li],
                shadow.sliding_layer_bytes,
                shadow.full_layer_bytes,
                valid_len_slots,
                stream,
            )?;
            let _ = (sliding_view, full_view);
        }

        // Commit 31c: per-slot K cache probe. Dumps max_abs per slot
        // for the first N prompt positions in the shadow's sliding K
        // layer. If one slot dominates magnitude → that explains the
        // saturated wrong-K argmax that's invariant to Q-RoPE.
        if std::env::var("RVLLM_SPEC_FA_DEBUG").as_deref() == Ok("1") {
            let (shadow_k, shadow_v, nkvh, hd): (u64, u64, usize, usize) = {
                let guard = self.drafter.lock().unwrap();
                let d = guard.as_ref().expect("drafter resident");
                let s = d.shadow_kv.expect("shadow_kv populated");
                (s.sliding_k_ptr,
                 s.sliding_v_ptr,
                 s.sliding_num_kv_heads as usize,
                 s.sliding_head_dim as usize)
            };
            let n_probe = 16usize.min(prompt_ids.len());
            let head_bytes = hd * 2;
            let slot_bytes = nkvh * head_bytes;
            let total_bytes = n_probe * slot_bytes;
            let mut buf = vec![0u16; total_bytes / 2];
            let _ = self.stream.fence();
            let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut _,
                shadow_k,
                total_bytes,
            );
            for s in 0..n_probe {
                let off = s * (nkvh * hd);
                let slice = &buf[off..off + (nkvh * hd)];
                let mut max_abs = 0f32;
                let mut sum_abs = 0f64;
                for &b in slice {
                    let v = half::f16::from_bits(b).to_f32();
                    if v.is_finite() {
                        sum_abs += v.abs() as f64;
                        if v.abs() > max_abs { max_abs = v.abs(); }
                    }
                }
                let mean_abs = (sum_abs / (slice.len() as f64)) as f32;
                eprintln!(
                    "[spec-kprobe] slot={} max_abs={:.3} mean_abs={:.4}",
                    s, max_abs, mean_abs,
                );
            }
            // Commit 31e: V cache magnitudes per slot. If V values
            // are uniform across slots → mean(V) ≈ V_attended →
            // Q-blind FA output is "explained by V uniformity, not
            // FA kernel bug." If V varies per slot → FA kernel
            // really IS ignoring Q.
            let mut vbuf = vec![0u16; total_bytes / 2];
            let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                vbuf.as_mut_ptr() as *mut _,
                shadow_v,
                total_bytes,
            );
            // Compute pairwise cosine sim of head0's V across slots
            // to test uniformity directly.
            let extract = |slot: usize| -> Vec<f32> {
                let off = slot * (nkvh * hd);
                vbuf[off..off + hd].iter()
                    .map(|&b| half::f16::from_bits(b).to_f32())
                    .collect()
            };
            for s in 0..n_probe {
                let off = s * (nkvh * hd);
                let slice = &vbuf[off..off + (nkvh * hd)];
                let mut max_abs = 0f32;
                let mut sum_abs = 0f64;
                let mut sum_sq = 0f64;
                for &b in slice {
                    let v = half::f16::from_bits(b).to_f32();
                    if v.is_finite() {
                        sum_abs += v.abs() as f64;
                        sum_sq += (v * v) as f64;
                        if v.abs() > max_abs { max_abs = v.abs(); }
                    }
                }
                let mean_abs = (sum_abs / slice.len() as f64) as f32;
                let rms = (sum_sq / slice.len() as f64).sqrt() as f32;
                eprintln!(
                    "[spec-vprobe] slot={} max_abs={:.3} mean_abs={:.4} rms={:.4}",
                    s, max_abs, mean_abs, rms,
                );
            }
            // Cosine similarity between V[slot=0] and V[slot=N-1] —
            // if highly uniform (>0.95) V cache hypothesis confirmed.
            if n_probe >= 2 {
                let v0 = extract(0);
                let vn = extract(n_probe - 1);
                let mut dot = 0f64;
                let mut n0 = 0f64;
                let mut nn = 0f64;
                for i in 0..hd {
                    dot += (v0[i] * vn[i]) as f64;
                    n0 += (v0[i] * v0[i]) as f64;
                    nn += (vn[i] * vn[i]) as f64;
                }
                let cos = dot / (n0.sqrt() * nn.sqrt() + 1e-30);
                eprintln!(
                    "[spec-vprobe] cosine(V[slot=0], V[slot={}]) = {:.4}",
                    n_probe - 1, cos,
                );
            }
            // Probe drafter layer 0 layernorm gammas — if these are
            // anomalously large, they're scaling the residual past
            // the cross-attn signal magnitude.
            let (l0_input_g, l0_post_attn_g, l0_pre_ff_g, l0_post_ff_g, hidden_sz) = {
                let guard = self.drafter.lock().unwrap();
                let d = guard.as_ref().expect("drafter resident");
                let l = &d.layers[0];
                (l.input_layernorm, l.post_attention_layernorm,
                 l.pre_feedforward_layernorm, l.post_feedforward_layernorm,
                 d.arch.hidden_size)
            };
            let mut probe_norm = |ptr: u64, label: &str| {
                let mut buf = vec![0u16; hidden_sz];
                let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut _, ptr, hidden_sz * 2);
                let mut max_abs = 0f32;
                let mut sum_abs = 0f64;
                for &b in &buf {
                    let v = half::f16::from_bits(b).to_f32();
                    if v.is_finite() {
                        sum_abs += v.abs() as f64;
                        if v.abs() > max_abs { max_abs = v.abs(); }
                    }
                }
                let head4: Vec<f32> = buf.iter().take(4)
                    .map(|&b| half::f16::from_bits(b).to_f32()).collect();
                eprintln!(
                    "[spec-gamma] L0 {:>20} max={:.3} mean={:.4} head4={:?}",
                    label, max_abs, sum_abs / hidden_sz as f64, head4,
                );
            };
            probe_norm(l0_input_g, "input_layernorm");
            probe_norm(l0_post_attn_g, "post_attn_layernorm");
            probe_norm(l0_pre_ff_g, "pre_ff_layernorm");
            probe_norm(l0_post_ff_g, "post_ff_layernorm");
        }

        // K-iteration drafter loop. Collects up to spec_k candidate
        // tokens. At commit 23 the return value still uses the BASE's
        // first decode token — verify + accept land in commits 24/25.
        let mut drafter_tokens: Vec<u32> = Vec::with_capacity(spec_k.max(1) as usize);
        // Commit 28: cache drafter masked-embedder dimensions outside
        // the locked drafter scope so the typical-mode host sampler
        // can size the DtoH at the per-iter checkpoint.
        let (spec_top_k_h, spec_per_centroid_h): (usize, usize) = {
            let guard = self.drafter.lock().unwrap();
            let d = guard.as_ref().expect("drafter resident");
            let tk = d.arch.centroid_intermediate_top_k;
            let nc = d.arch.num_centroids;
            let vc = d.arch.vocab_size;
            (tk, if nc > 0 { vc / nc } else { 0 })
        };
        // log_q[i] = drafter sampling log-probability of
        // drafter_tokens[i]. Populated only in typical mode.
        let mut drafter_log_q: Vec<f32> = Vec::with_capacity(spec_k.max(1) as usize);
        // Simple LCG state for typical-mode host sampling. Seed from
        // sampling config when available, else fall back to a fixed
        // pseudo-deterministic seed so accept-rate measurements are
        // reproducible across runs.
        let mut next_rand_f32_spec: u64 = match sampling {
            SamplingConfig::Stochastic { seed, .. } => seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
            SamplingConfig::Greedy => 0xDEAD_BEEF_CAFE_BABE,
        };
        let mut current_base_hidden: u64 = base_hidden_last_step_real;
        for k_step in 0..(spec_k.max(1) as usize) {
        let step = crate::gemma4_drafter::DrafterForwardStep {
            base_hidden_last_step: current_base_hidden,
            last_token_embed: last_token_embed.device_ptr(),
            sliding_kv,
            full_kv,
            position: drafter_position + k_step as u32,
            out_logits: workspace.centroid_logits,
            out_hidden: workspace.out_hidden,
            out_token_id: workspace.out_token_id,
        };

        // Inner block: borrow the drafter to run one forward step.
        {
            let guard = self.drafter.lock().unwrap();
            let drafter = guard.as_ref().expect("checked above");
            drafter.prepare_pre_projection_input(&step, &workspace, stream)?;
            // Commit 54: sqrt(backbone) scale on embed half is now in
            // `apply_pre_projection_embed_scale`; logic + comment
            // moved there. Stage 1a of the run_drafter_step_only
            // extraction (codex review priority 1).
            self.apply_pre_projection_embed_scale(
                &workspace,
                drafter.arch.backbone_hidden_size as i32,
                stream,
            )?;
            self.run_drafter_pre_projection(drafter, &workspace)?;
            // Commit 31h: probe pre_projection output (= workspace.hidden
            // after pre_projection GEMM + cast). This is layer 0's input
            // residual; if it's bad, the whole drafter forward inherits
            // garbage.
            if k_step == 0
                && std::env::var("RVLLM_SPEC_DEBUG_Q_BISECT").as_deref() == Ok("1")
            {
                let _ = self.stream.fence();
                let hidden_sz = drafter.arch.hidden_size;
                let mut buf = vec![0u16; hidden_sz];
                let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut _, workspace.hidden, hidden_sz * 2);
                let mut max_abs = 0f32;
                let mut sum_sq = 0f64;
                for &b in &buf {
                    let v = half::f16::from_bits(b).to_f32();
                    if v.is_finite() {
                        if v.abs() > max_abs { max_abs = v.abs(); }
                        sum_sq += (v * v) as f64;
                    }
                }
                let rms = (sum_sq / hidden_sz as f64).sqrt() as f32;
                let head8: Vec<f32> = buf.iter().take(8)
                    .map(|&b| half::f16::from_bits(b).to_f32()).collect();
                eprintln!(
                    "[spec-qbisect] post_pre_projection(hidden) max={:.3} rms={:.4} head8={:?}",
                    max_abs, rms, head8);
            }

            // Commits 7-14: drive all 4 drafter layers. Each layer
            // shares the same Q-side helper + an attn-finisher +
            // an mlp-finisher; only the cross-attn launcher differs
            // (sliding vs global). The driver loop here is the
            // narrow integration site.
            let num_layers = drafter.layers.len();
            for li in 0..num_layers {
                let layer = &drafter.layers[li];
                let is_global = matches!(
                    layer.layer_type,
                    rvllm_loader::gemma4_drafter::DrafterLayerType::Full
                );
                // Commit 31: scale=1.0 is the vLLM-documented value
                // (Gemma4MTPAttention.scaling=1.0). RVLLM_SPEC_FA_SCALE
                // overrides for debugging: "stable" uses
                // 1/sqrt(head_dim); "mtp" uses 1.0. Default is "mtp"
                // when typical mode is on, else "stable".
                let eff_hd = layer.effective_head_dim as f32;
                let scale_mode = std::env::var("RVLLM_SPEC_FA_SCALE")
                    .unwrap_or_else(|_| if typical_mode { "mtp".into() } else { "stable".into() });
                let scale = if scale_mode == "mtp" {
                    1.0_f32
                } else {
                    1.0_f32 / eff_hd.sqrt()
                };
                self.run_drafter_layer_q_side(
                    drafter, &workspace, li, step.position)?;
                // Commit 31d (codex): zero-Q test. Zero workspace.q
                // before cross-attn. If drafter output is UNCHANGED,
                // the FA kernel is not actually consuming our Q,
                // pinning the bug to a Q-pointer / kernel-wiring
                // issue rather than RoPE / softmax / KV.
                if std::env::var("RVLLM_SPEC_ZERO_Q").as_deref() == Ok("1") {
                    let q_rows = drafter.arch.num_attention_heads * layer.effective_head_dim;
                    let rc = cudarc::driver::sys::cuMemsetD8Async(
                        workspace.q, 0, q_rows * 2, stream as cudarc::driver::sys::CUstream);
                    if li == 0 {
                        let _ = self.stream.fence();
                        let mut buf = vec![0u16; q_rows];
                        let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                            buf.as_mut_ptr() as *mut _,
                            workspace.q,
                            q_rows * 2,
                        );
                        let max_abs = buf.iter()
                            .map(|&b| half::f16::from_bits(b).to_f32().abs())
                            .fold(0f32, f32::max);
                        eprintln!(
                            "[spec-zeroq] li=0 memset_rc={:?} q_max_abs_after_memset={:.6}",
                            rc, max_abs,
                        );
                    }
                }
                if is_global {
                    // Commit 15: BC=16 path now fits the sm_121
                    // smem cap; the earlier zero-attn fallback is
                    // gone. Errors bubble.
                    drafter.launch_cross_attn_global(
                        workspace.attn_out,
                        workspace.q,
                        block_tables_region.device_ptr(),
                        context_lens_region.device_ptr(),
                        scale,
                        stream,
                    )?;
                } else {
                    drafter.launch_cross_attn_sliding(
                        workspace.attn_out,
                        workspace.q,
                        block_tables_region.device_ptr(),
                        context_lens_region.device_ptr(),
                        scale,
                        sliding_window,
                        stream,
                    )?;
                }
                let probe_fn = |label: &str, ptr: u64, n_f16: usize| {
                    if std::env::var("RVLLM_SPEC_FA_DEBUG").as_deref() != Ok("1") { return; }
                    let fence_ok = self.stream.fence().is_ok();
                    let mut h = vec![0u16; n_f16];
                    let rc = unsafe {
                        cudarc::driver::sys::cuMemcpyDtoH_v2(
                            h.as_mut_ptr() as *mut _, ptr, n_f16 * 2)
                    };
                    let dtoh_ok = rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS;
                    let mut max_abs = 0f32;
                    let mut nan_count = 0usize;
                    let mut inf_count = 0usize;
                    for &b in &h {
                        let v = half::f16::from_bits(b).to_f32();
                        if v.is_nan() { nan_count += 1; }
                        else if v.is_infinite() { inf_count += 1; }
                        else if v.abs() > max_abs { max_abs = v.abs(); }
                    }
                    let head8: Vec<f32> = h.iter().take(8)
                        .map(|&b| half::f16::from_bits(b).to_f32()).collect();
                    eprintln!(
                        "[spec-fa] li={} {} fence_ok={} dtoh_ok={} nan={} inf={} max_abs={:.3} head8={:?}",
                        li, label, fence_ok, dtoh_ok, nan_count, inf_count, max_abs, head8,
                    );
                };
                let q_rows = drafter.arch.num_attention_heads * layer.effective_head_dim;
                if li == 0 {
                    probe_fn("layer0_entry(hidden)", workspace.hidden, 32);
                    probe_fn("layer0_q_post_rope(q)", workspace.q, q_rows.min(32));
                }
                probe_fn("after_xattn(attn_out)", workspace.attn_out, q_rows.min(32));

                // Commit 31g (codex round 8): CPU-reference cross-attn
                // at layer 0 only. Computes softmax(scale * Q·K^T) V
                // on host for the SAME inputs the FA kernel saw.
                // If CPU and GPU differ → FA kernel has a bug.
                // If they match → FA innocent, bug is upstream.
                if li == 0
                    && std::env::var("RVLLM_SPEC_DEBUG_CPU_ATTN").as_deref() == Ok("1")
                {
                    let _ = self.stream.fence();
                    let nheads = drafter.arch.num_attention_heads;
                    let hd = layer.effective_head_dim;
                    let ctx_len = prompt_ids.len();
                    // The drafter lock is already held by the outer
                    // block — re-locking would self-deadlock since
                    // std::Mutex is not re-entrant. Use the existing
                    // `drafter` reference from the outer scope.
                    let shadow = drafter.shadow_kv.expect("shadow_kv populated");
                    let shadow_k = shadow.sliding_k_ptr;
                    let shadow_v = shadow.sliding_v_ptr;
                    let nkvh = shadow.sliding_num_kv_heads as usize;
                    let mut h_q = vec![0u16; nheads * hd];
                    let mut h_k = vec![0u16; ctx_len * nkvh * hd];
                    let mut h_v = vec![0u16; ctx_len * nkvh * hd];
                    let mut h_attn = vec![0u16; nheads * hd];
                    let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                        h_q.as_mut_ptr() as *mut _, workspace.q,
                        nheads * hd * 2);
                    let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                        h_k.as_mut_ptr() as *mut _, shadow_k,
                        ctx_len * nkvh * hd * 2);
                    let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                        h_v.as_mut_ptr() as *mut _, shadow_v,
                        ctx_len * nkvh * hd * 2);
                    let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                        h_attn.as_mut_ptr() as *mut _, workspace.attn_out,
                        nheads * hd * 2);
                    let to_f32 = |b: u16| half::f16::from_bits(b).to_f32();
                    // CPU compute for head 0 only (representative).
                    let h = 0usize;
                    let kv_head = h * nkvh / nheads;
                    let q_h: Vec<f32> = (0..hd).map(|d| to_f32(h_q[h*hd + d])).collect();
                    let mut scores = vec![0f32; ctx_len];
                    for t in 0..ctx_len {
                        let mut dot = 0f32;
                        for d in 0..hd {
                            let k_td = to_f32(h_k[(t * nkvh + kv_head) * hd + d]);
                            dot += q_h[d] * k_td;
                        }
                        scores[t] = dot * scale;
                    }
                    // softmax
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_e = 0f64;
                    let mut exps = vec![0f64; ctx_len];
                    for t in 0..ctx_len {
                        let e = ((scores[t] - max_s) as f64).exp();
                        exps[t] = e;
                        sum_e += e;
                    }
                    let probs: Vec<f64> = exps.iter().map(|e| e / sum_e).collect();
                    // CPU attended V
                    let mut cpu_out = vec![0f32; hd];
                    for d in 0..hd {
                        let mut acc = 0f64;
                        for t in 0..ctx_len {
                            let v_td = to_f32(h_v[(t * nkvh + kv_head) * hd + d]) as f64;
                            acc += probs[t] * v_td;
                        }
                        cpu_out[d] = acc as f32;
                    }
                    // GPU head-0 attn_out
                    let gpu_out: Vec<f32> = (0..hd).map(|d| to_f32(h_attn[h*hd + d])).collect();
                    // Compare
                    let mut max_abs_diff = 0f32;
                    let mut dot = 0f64;
                    let mut nc = 0f64; let mut ng = 0f64;
                    for d in 0..hd {
                        let diff = (cpu_out[d] - gpu_out[d]).abs();
                        if diff > max_abs_diff { max_abs_diff = diff; }
                        dot += (cpu_out[d] * gpu_out[d]) as f64;
                        nc += (cpu_out[d] * cpu_out[d]) as f64;
                        ng += (gpu_out[d] * gpu_out[d]) as f64;
                    }
                    let cos = dot / (nc.sqrt() * ng.sqrt() + 1e-30);
                    // top-5 attention positions
                    let mut idx_sorted: Vec<usize> = (0..ctx_len).collect();
                    idx_sorted.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap_or(std::cmp::Ordering::Equal));
                    let top5: Vec<(usize, f64)> = idx_sorted.iter().take(5)
                        .map(|&i| (i, probs[i])).collect();
                    eprintln!(
                        "[spec-cpu-attn] li=0 h=0 ctx_len={} scale={:.4} top5={:?}",
                        ctx_len, scale, top5,
                    );
                    eprintln!(
                        "[spec-cpu-attn] CPU_head0[..8]={:?}",
                        &cpu_out[..8],
                    );
                    eprintln!(
                        "[spec-cpu-attn] GPU_head0[..8]={:?}",
                        &gpu_out[..8],
                    );
                    eprintln!(
                        "[spec-cpu-attn] max_abs_diff={:.4} cosine={:.6}",
                        max_abs_diff, cos,
                    );
                }
                self.run_drafter_layer_attn_finisher(drafter, &workspace, li)?;
                probe_fn("after_attn_finisher(hidden)", workspace.hidden, 32);
                self.run_drafter_layer_mlp_finisher(drafter, &workspace, li)?;
                probe_fn("after_mlp_finisher(hidden)", workspace.hidden, 32);
                let _ = is_global;
            }

            // Commit 14: `model.norm` final RMSNorm — single launch
            // over `workspace.hidden` with drafter's tied final norm
            // gamma. Mirrors HF's `Gemma4Model.norm` applied to
            // `last_hidden_state` before lm_head.
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1,
                hidden: drafter.arch.hidden_size as u32,
                eps: drafter.arch.rms_norm_eps,
            }
            .launch(
                self.fused.fn_rmsnorm,
                workspace.hidden,
                drafter.top.final_norm,
                stream,
            )?;

            // Commit 14: MaskedEmbedder argmax → workspace.out_token_id.
            // Uses the drafter's centroids + token_ordering +
            // embed_tokens (tied lm_head) and the post-final-norm
            // workspace.hidden. Output is the next draft token id
            // (i32 [1]); the run-out of writing it to centroid_logits
            // is unused at this stage.
            let n_centroids = drafter.arch.num_centroids as i32;
            let top_k = drafter.arch.centroid_intermediate_top_k as i32;
            let vocab = drafter.arch.vocab_size as i32;
            let per_centroid: i32 = if n_centroids > 0 {
                vocab / n_centroids
            } else { 0 };
            let fn_masked = drafter
                .fn_masked_embedder_argmax_f16
                .expect("MaskedEmbedder kernel attached in ensure_drafter");
            // Commit 28: feed sparse candidate buffers when typical
            // mode is on so the host sampler has the full top_k *
            // per_centroid table. In greedy mode pass 0 to keep the
            // kernel on its existing fast path (no sparse writes).
            let (sp_ids_ptr, sp_lg_ptr) = if typical_mode {
                (workspace.sparse_ids, workspace.sparse_logits)
            } else {
                (0u64, 0u64)
            };
            crate::gemma4_drafter::launch_masked_embedder_argmax_f16(
                fn_masked,
                workspace.hidden,
                drafter.top.centroids,
                drafter.top.token_ordering,
                drafter.top.embed_tokens,
                drafter.arch.hidden_size as i32,
                n_centroids,
                top_k,
                per_centroid,
                vocab,
                workspace.out_token_id,
                /* out_logit */ 0,
                sp_ids_ptr,
                sp_lg_ptr,
                stream,
            )?;

            // Commit 21: RVLLM_SPEC_DEBUG=1 — run a CPU reference of
            // the MaskedEmbedder against the same inputs and compare
            // to the GPU result. If CPU and GPU agree on the argmax
            // token, the MaskedEmbedder kernel is correct and the
            // accept_rate=0 bug is upstream (pre_projection, layer
            // stack, or hidden snapshot). If they disagree, the
            // kernel is wrong. Either way we get a concrete answer.
            if std::env::var("RVLLM_SPEC_DEBUG").as_deref() == Ok("1") {
                self.stream.fence()?;
                let hidden_sz = drafter.arch.hidden_size as usize;
                let n_cent = drafter.arch.num_centroids as usize;
                let vocab_sz = drafter.arch.vocab_size as usize;
                let pc = per_centroid as usize;
                let tk = top_k as usize;
                // DtoH copies.
                let mut h_hidden = vec![0u16; hidden_sz];
                let mut h_centroids = vec![0u16; n_cent * hidden_sz];
                let mut h_token_ordering = vec![0i64; vocab_sz];
                let mut h_embed = vec![0u16; vocab_sz * hidden_sz];
                let mut h_last_emb = vec![0u16; drafter.arch.backbone_hidden_size];
                let mut h_base_hid = vec![0u16; drafter.arch.backbone_hidden_size];
                let copy = |dst: *mut u8, src: u64, n: usize, what: &str| {
                    let rc = cudarc::driver::sys::cuMemcpyDtoH_v2(
                        dst as *mut _, src, n);
                    if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        eprintln!("[spec-debug] DtoH {} FAILED rc={:?}",
                                  what, rc);
                    }
                };
                copy(h_hidden.as_mut_ptr() as *mut u8,
                     workspace.hidden, hidden_sz * 2, "hidden");
                copy(h_centroids.as_mut_ptr() as *mut u8,
                     drafter.top.centroids, n_cent * hidden_sz * 2, "centroids");
                copy(h_token_ordering.as_mut_ptr() as *mut u8,
                     drafter.top.token_ordering, vocab_sz * 8, "token_ordering");
                copy(h_embed.as_mut_ptr() as *mut u8,
                     drafter.top.embed_tokens, vocab_sz * hidden_sz * 2, "embed_tokens");
                copy(h_last_emb.as_mut_ptr() as *mut u8,
                     step.last_token_embed,
                     drafter.arch.backbone_hidden_size * 2, "last_token_embed");
                copy(h_base_hid.as_mut_ptr() as *mut u8,
                     step.base_hidden_last_step,
                     drafter.arch.backbone_hidden_size * 2, "base_hidden");

                // Helper: f16 bits -> f32.
                let f16_to_f32 = |bits: u16| -> f32 {
                    half::f16::from_bits(bits).to_f32()
                };
                // RMS helper for sanity check.
                let rms = |v: &[u16]| -> f32 {
                    let mut s = 0f64;
                    for &b in v { let x = f16_to_f32(b) as f64; s += x*x; }
                    ((s / v.len() as f64).sqrt()) as f32
                };
                let head8 = |v: &[u16]| -> Vec<f32> {
                    v.iter().take(8).map(|&b| f16_to_f32(b)).collect()
                };
                eprintln!("[spec-debug] last_token_embed head8={:?} rms={:.4}",
                          head8(&h_last_emb), rms(&h_last_emb));
                eprintln!("[spec-debug] base_hidden       head8={:?} rms={:.4}",
                          head8(&h_base_hid), rms(&h_base_hid));
                eprintln!("[spec-debug] drafter_hidden    head8={:?} rms={:.4}",
                          head8(&h_hidden), rms(&h_hidden));

                // CPU MaskedEmbedder reference.
                let hidden_f32: Vec<f32> = h_hidden.iter().map(|&b| f16_to_f32(b)).collect();
                // Phase 2: centroid logits.
                let mut cent_logits = vec![0f32; n_cent];
                for c in 0..n_cent {
                    let row_off = c * hidden_sz;
                    let mut acc = 0f32;
                    for k in 0..hidden_sz {
                        acc += hidden_f32[k] * f16_to_f32(h_centroids[row_off + k]);
                    }
                    cent_logits[c] = acc;
                }
                // Phase 3: top-k.
                let mut idx_sorted: Vec<usize> = (0..n_cent).collect();
                idx_sorted.sort_by(|a, b| cent_logits[*b]
                    .partial_cmp(&cent_logits[*a]).unwrap_or(std::cmp::Ordering::Equal));
                let cpu_top: Vec<usize> = idx_sorted.iter().take(tk).copied().collect();
                eprintln!("[spec-debug] CPU top-8 centroids={:?} logits={:?}",
                          &cpu_top[..8.min(cpu_top.len())],
                          cpu_top.iter().take(8)
                              .map(|&i| cent_logits[i]).collect::<Vec<_>>());
                // Phase 4: 4096 candidate token dot products.
                let mut best = f32::NEG_INFINITY;
                let mut best_id: i32 = -1;
                for &c in &cpu_top {
                    for sub in 0..pc {
                        let t = h_token_ordering[c * pc + sub];
                        if t < 0 || (t as usize) >= vocab_sz { continue; }
                        let tu = t as usize;
                        let row_off = tu * hidden_sz;
                        let mut acc = 0f32;
                        for k in 0..hidden_sz {
                            acc += hidden_f32[k] * f16_to_f32(h_embed[row_off + k]);
                        }
                        if acc > best { best = acc; best_id = tu as i32; }
                    }
                }
                // Read back GPU result.
                let mut gpu_tok: [u8; 4] = [0; 4];
                let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                    gpu_tok.as_mut_ptr() as *mut _,
                    workspace.out_token_id, 4);
                let gpu_tok_i = i32::from_le_bytes(gpu_tok);
                eprintln!(
                    "[spec-debug] CPU drafter_tok={} logit={:.4} | GPU drafter_tok={} | agree={}",
                    best_id, best, gpu_tok_i, best_id == gpu_tok_i,
                );
            }

            // Commit 14: post_projection → workspace.out_hidden. Feeds
            // the NEXT MTP step's `pre_projection` input chain.
            // Output is captured but unused at this commit since the
            // K-draft loop hasn't landed; arena restore reclaims it.
            self.cublaslt.f16_gemm_f32(
                workspace.hidden,
                drafter.top.post_projection,
                workspace.gemm_f32,
                1,
                drafter.arch.backbone_hidden_size as i32,
                drafter.arch.hidden_size as i32,
                stream,
            )?;
            launch_cast_f32_to_f16(
                &self.stream,
                self.fused.fn_cast_f32_to_f16,
                workspace.gemm_f32,
                workspace.out_hidden,
                drafter.arch.backbone_hidden_size as i32,
            )?;
        } // end inner drafter-forward block (lock guard scope)

        // Read the drafter's predicted token id (i32[1]) per
        // iteration so we can both (a) feed its embedding back as
        // next-step input and (b) accumulate the K-draft Vec for
        // the commit-24 batched verify pass.
        self.stream.fence()?;
        let mut tok_host_iter: [u8; 4] = [0; 4];
        let rc_t = cudarc::driver::sys::cuMemcpyDtoH_v2(
            tok_host_iter.as_mut_ptr() as *mut _,
            workspace.out_token_id,
            4,
        );
        if rc_t != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::cuda(
                "run_generate_speculative: per-iter DtoH drafter token id",
                rvllm_core::CudaErrorKind::MemcpyFailed,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        let mut tok_iter_u: u32 = i32::from_le_bytes(tok_host_iter).max(0) as u32;

        // Commit 28: typical-acceptance — host samples from the
        // 4096-candidate sparse distribution at user temperature.
        // Replaces the kernel argmax for this step. Records log_q
        // (drafter probability of the sampled token) for commit 29's
        // rejection criterion. Greedy mode bypasses this entirely.
        if typical_mode {
            let sparse_len = spec_top_k_h * spec_per_centroid_h;
            let mut h_ids = vec![-1i32; sparse_len];
            let mut h_lg = vec![f32::NEG_INFINITY; sparse_len];
            let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                h_ids.as_mut_ptr() as *mut _,
                workspace.sparse_ids,
                sparse_len * 4,
            );
            let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                h_lg.as_mut_ptr() as *mut _,
                workspace.sparse_logits,
                sparse_len * 4,
            );
            let temp: f32 = match sampling {
                SamplingConfig::Greedy => 0.0,
                SamplingConfig::Stochastic { temperature, .. } => temperature,
            };
            let (sampled_id, log_q_sampled) =
                sparse_sample_with_temp(&h_ids, &h_lg, temp, &mut next_rand_f32_spec);
            if sampled_id >= 0 {
                tok_iter_u = sampled_id as u32;
            }
            drafter_log_q.push(log_q_sampled);
            tracing::debug!(
                k_step,
                sampled_token = tok_iter_u,
                log_q = log_q_sampled,
                temperature = temp,
                "Gemma 4 drafter typical-mode sparse sample",
            );
        }

        drafter_tokens.push(tok_iter_u);

        // Prepare inputs for the next K-step if there is one:
        //   last_token_embed ← embed(D_k_step) (base embedding table,
        //                       pre-scaled by sqrt(backbone_hidden))
        //   current_base_hidden ← workspace.out_hidden (the
        //                          post_projection just written above)
        //   position increments naturally via k_step+1 next iter.
        if k_step + 1 < (spec_k.max(1) as usize) {
            token_ids_region.copy_from_host(&tok_iter_u.to_le_bytes())?;
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1,
                hidden: hidden_u,
                vocab: vocab_u,
            }
            .launch(
                fn_embed,
                last_token_embed.device_ptr(),
                self.model.embedding.offset_bytes,
                token_ids_region.device_ptr(),
                stream,
            )?;
            current_base_hidden = workspace.out_hidden;
        }
        } // end for k_step

        // Commit 40 — batched-verify hot path.
        // After K drafter forwards completed, instead of comparing
        // against `_base_first_tok` (= K sequential base decodes from
        // the warmup, which is exactly the wall-clock cost spec-decode
        // is supposed to eliminate), do:
        //
        //   1. self.run_generate(prompt + drafts, max_new=1)
        //      → prefix-cache hits on the original P prompt tokens
        //      → batched-prefill of K drafts (ONE forward pass on
        //        FP8/NVFP4 KV; F16 KV silently falls back to per-token)
        //      → 1 decode step produces base argmax at position P+K
        //        (= the "bonus" base token when all drafts accepted)
        //      → the K-row capture hook (also commit 40, fixed) writes
        //        K post-layer-loop, pre-final-norm hiddens into
        //        `base_last_k_hidden_ptr`
        //   2. final_norm + lm_head_M=K + softcap + argmax_M=K on the
        //      K-buffer → K base argmaxes ('base_argmax_K') corresponding
        //      to verify at draft positions [P, P+1, ..., P+K-1].
        //   3. accept_len = longest prefix where drafts[i] == base_argmax_K[i].
        //   4. Emit drafts[..accept_len] + (bonus if all accepted, else
        //      divergence base argmax at position accept_len).
        //
        // Per-iter cost: 1 prefix-cached prompt prefill (cheap) + K
        // drafter forwards + 1 batched K-draft prefill + 1 base decode.
        // Vs legacy: 1 prompt prefill + K drafter forwards + K+1
        // sequential base decodes. Speedup factor = (K+1) base decodes
        // → (1 batched prefill of K + 1 decode) ≈ 1.7–2.6× per repo's
        // Qwen 3.6 batched-prefill numbers, at expected accept_rate.
        if batched_verify_mode && !drafter_tokens.is_empty() {
            let k_actual = drafter_tokens.len() as u32;
            // Reclaim ALL scratch from the warmup base prefill +
            // drafter K-loop. Drafter tokens are on host; shadow KV
            // and base_last_hidden_ptr live above the worker's
            // scratch checkpoint (allocated in ensure_drafter
            // pre-checkpoint); persistent_kv is also above
            // scratch (allocated in init_prefix_cache from the
            // worker before scratch_ck). So restore is safe and
            // frees enough scratch for the second batched-prefill
            // run_generate to fit on the same request.
            unsafe { self.arena.restore(arena_ck_at_entry); }
            // Arm K-row capture before the second base call.
            self.base_last_k_count
                .store(k_actual, std::sync::atomic::Ordering::Release);
            self.base_last_k_snapshot_pending
                .store(true, std::sync::atomic::Ordering::Release);

            // Build prompt + drafts.
            let mut prompt_and_drafts: Vec<u32> =
                Vec::with_capacity(prompt_ids.len() + drafter_tokens.len());
            prompt_and_drafts.extend_from_slice(prompt_ids);
            prompt_and_drafts.extend_from_slice(&drafter_tokens);

            // Commit 50 (Phase C-1) — codex review priority 1 fix.
            //
            // Arm the prefix-cache override hooks (added in commit 49)
            // so this verify call doesn't re-prefill the full prompt
            // under RVLLM_PREFILL_CHUNK_SIZE=2048's chunk_size cap.
            //
            //   * force_common_prefix_override = prompt_ids.len():
            //     bypass the token-match + chunk_size cap. The
            //     warmup base prefill above already wrote
            //     prompt-length K/V slots, so the spec contract
            //     says those P slots are valid. With the override
            //     active, new_q = (prompt_len + K) - prompt_len = K
            //     (NOT new_q = prompt_len + K with prefix=0 under
            //     the chunk cap, which was the bug).
            //
            //   * skip_prefix_cache_publish = true: spec-internal
            //     state doesn't pollute the cross-request cache.
            //
            // This is the substantive codex-priority-1 fix: short-
            // prompt spec iterations now prefill ONLY the K drafts
            // per verify call, not the whole prompt + K.
            self.force_common_prefix_override
                .store(prompt_ids.len() as u32, std::sync::atomic::Ordering::Release);
            self.skip_prefix_cache_publish
                .store(true, std::sync::atomic::Ordering::Release);
            // Commit 56 (codex priority 0.2): drop the wasted bonus
            // base decode. The verify call's K-row hidden capture
            // is the only output this branch consumes — we run our
            // own final_norm + lm_head_M=K + softcap + argmax_M=K
            // on the K-buffer below to produce `base_argmax_K`.
            // `base_argmax_K[K-1]` equals what the bonus token
            // would have been (= argmax over hidden_{P+K-1}), so
            // skipping the bonus saves one row-extract + 1-row
            // final_norm + M=1 GEMM (vocab columns) + softcap +
            // M=1 argmax + 4-byte DtoH per spec iteration, with no
            // semantic change. `bonus_tok_vec` was already marked
            // unused (`let _ = bonus_tok_vec;` ~30 lines below).
            self.force_prefill_only
                .store(true, std::sync::atomic::Ordering::Release);
            let bonus_tok_vec = self.run_generate(
                fn_embed,
                self.fused.fn_argmax,
                &prompt_and_drafts,
                1,
                _eos_ids,
                /* shadow_requested */ false,
                SamplingConfig::Greedy,
                _cancel,
                None,
                &[],
                &[],
            )?;

            // Apply final_norm + lm_head + softcap + argmax on the K
            // captured hiddens.
            let k_ptr = self
                .base_last_k_hidden_ptr
                .load(std::sync::atomic::Ordering::Acquire);
            if k_ptr == 0 {
                return Err(rvllm_core::RvllmError::Attention {
                    err: rvllm_core::AttentionError::FeatureNotAvailable {
                        op: "run_generate_speculative_batched: base_last_k_hidden_ptr is zero",
                        backend: "Gemma4SpecDecode",
                    },
                    ctx: rvllm_core::AttnCtx {
                        op: "run_generate_speculative_batched",
                        stream,
                        num_seqs: 1,
                        head_dim: self.arch.max_head_dim() as u32,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
            // Final norm on K rows in-place.
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: k_actual,
                hidden: hidden_u,
                eps: arch.rms_norm_eps,
            }
            .launch(
                self.fused.fn_rmsnorm,
                k_ptr,
                self.model.final_norm.offset_bytes,
                stream,
            )?;
            // LM head: GEMM M=K, N=vocab, K=hidden, out f32 logits.
            let logits_region = arena.region(
                "spec_batched_verify_logits",
                (k_actual as usize) * (vocab_u as usize) * 4,
                16,
            )?;
            self.cublaslt.f16_gemm_f32(
                k_ptr,
                self.model.lm_head_f16.offset_bytes,
                logits_region.device_ptr(),
                k_actual as i32,
                vocab_u as i32,
                hidden_u as i32,
                stream,
            )?;
            // Softcap on the K-row f32 logits (same as run_generate path).
            if arch.logit_softcap > 0.0 {
                rvllm_fused::gemma4_launcher::LogitSoftcapLaunch {
                    num_tokens: k_actual,
                    vocab: vocab_u,
                    cap: arch.logit_softcap,
                }
                .launch(
                    self.fused.fn_softcap_f32,
                    logits_region.device_ptr(),
                    stream,
                )?;
            }
            // Argmax K rows → device u32 buffer → host.
            let argmax_region = arena.region(
                "spec_batched_verify_argmax",
                (k_actual as usize) * 4,
                16,
            )?;
            rvllm_fused::ArgmaxLaunch {
                num_tokens: k_actual,
                vocab: vocab_u,
            }
            .launch(
                self.fused.fn_argmax,
                logits_region.device_ptr(),
                argmax_region.device_ptr(),
                stream,
            )?;
            self.stream.fence()?;
            let mut base_argmax_k = vec![0u32; k_actual as usize];
            let rc = cudarc::driver::sys::cuMemcpyDtoH_v2(
                base_argmax_k.as_mut_ptr() as *mut _,
                argmax_region.device_ptr(),
                (k_actual as usize) * 4,
            );
            if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "run_generate_speculative_batched: argmax DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }

            // Commit 41 — typical-acceptance in batched mode.
            //
            // When RVLLM_GEMMA4_SPEC_TYPICAL=1, the verify uses
            // modified-rejection-sampling (Leviathan/Kalman) on log-
            // probabilities derived from base's softcap'd vocab
            // logits + drafter's log_q (already collected during the
            // drafter K-loop). The K-row logits buffer covers positions
            // P+1..P+K (rows 0..K-1). For i=0 (position P) we need an
            // extra row: lm_head over base_last_hidden (= POST-final-
            // norm hidden at P-1 from the warmup). Costs one M=1 GEMM
            // + one M=1 softcap + one MB DtoH per outer iter — small
            // vs the per-iter base decode.
            //
            // Acceptance bias `RVLLM_GEMMA4_SPEC_ACCEPT_BIAS` is the
            // same env knob as commit 34. Use bias>0 to compensate for
            // the drafter vs base softmax temperature mismatch (the
            // E4B drafter's 2048-centroid sparse softmax is much
            // sharper than the 262K-vocab base softmax, so naive
            // ratio over-rejects).
            let typical_in_batched = typical_mode
                && !drafter_log_q.is_empty();
            let typical_buffers: Option<(Vec<f32>, Vec<f32>)> = if typical_in_batched {
                let vocab_sz = vocab_u as usize;
                // 1. lm_head over base_last_hidden → warmup logits row.
                let warmup_logits_region = arena.region(
                    "spec_batched_warmup_logits",
                    vocab_sz * 4,
                    16,
                )?;
                self.cublaslt.f16_gemm_f32(
                    base_hidden_last_step_real,
                    self.model.lm_head_f16.offset_bytes,
                    warmup_logits_region.device_ptr(),
                    1,
                    vocab_u as i32,
                    hidden_u as i32,
                    stream,
                )?;
                if arch.logit_softcap > 0.0 {
                    rvllm_fused::gemma4_launcher::LogitSoftcapLaunch {
                        num_tokens: 1,
                        vocab: vocab_u,
                        cap: arch.logit_softcap,
                    }
                    .launch(
                        self.fused.fn_softcap_f32,
                        warmup_logits_region.device_ptr(),
                        stream,
                    )?;
                }
                self.stream.fence()?;
                // 2. DtoH warmup logits + K logits.
                let mut warmup_logits = vec![0.0f32; vocab_sz];
                let rc = cudarc::driver::sys::cuMemcpyDtoH_v2(
                    warmup_logits.as_mut_ptr() as *mut _,
                    warmup_logits_region.device_ptr(),
                    vocab_sz * 4,
                );
                if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "spec batched typical: warmup logits DtoH",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
                let mut k_logits = vec![0.0f32; (k_actual as usize) * vocab_sz];
                let rc = cudarc::driver::sys::cuMemcpyDtoH_v2(
                    k_logits.as_mut_ptr() as *mut _,
                    logits_region.device_ptr(),
                    (k_actual as usize) * vocab_sz * 4,
                );
                if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "spec batched typical: K logits DtoH",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
                Some((warmup_logits, k_logits))
            } else {
                None
            };

            // Greedy accept_len. Verify mapping (position-aligned):
            //
            //   drafts[0]   (position P)     vs  warmup _base_first_tok[0]
            //                                    (= argmax over hidden_{P-1})
            //   drafts[i>=1] (position P+i)  vs  base_argmax_K[i-1]
            //                                    (= argmax over hidden_{P+i-1})
            //
            // The K-buffer was captured at K positions [P..P+K-1];
            // lm_head on those produces predictions for positions
            // [P+1..P+K] = base_argmax_K[0..K-1]. The base's prediction
            // for position P itself comes from the WARMUP base call
            // (hidden_{P-1}) and lives in _base_first_tok[0].
            let mut accept_len: usize = 0;
            let warmup_b_p = _base_first_tok.first().copied();
            let kmax = k_actual as usize;
            // Commit 41 — typical-acceptance bias + temperature, same
            // semantics as the legacy verify path so existing
            // calibration knobs carry over.
            let typical_temp: f32 = if typical_in_batched {
                match sampling {
                    SamplingConfig::Stochastic { temperature, .. } => temperature.max(1e-6),
                    SamplingConfig::Greedy => 1.0,
                }
            } else { 0.0 };
            // Phase D: sourced from SpecDecodeRequestConfig.
            let bias_env: f64 = spec_cfg.accept_bias;
            // Per-iter rng for the acceptance Us. Independent of the
            // drafter sampling rng so the two don't perturb each other.
            let mut accept_rng_state: u64 = {
                let mut h: u64 = 0x517C_C1B7_2722_0A95;
                for &t in prompt_ids.iter().take(16) {
                    h = h.wrapping_mul(0x100000001B3).wrapping_add(t as u64);
                }
                h.wrapping_add(drafter_tokens.iter().copied().sum::<u32>() as u64)
            };
            let mut next_u01 = || -> f64 {
                accept_rng_state ^= accept_rng_state << 13;
                accept_rng_state ^= accept_rng_state >> 7;
                accept_rng_state ^= accept_rng_state << 17;
                ((accept_rng_state >> 33) as f64) / ((1u64 << 31) as f64)
            };
            let vocab_sz = vocab_u as usize;
            for i in 0..kmax {
                let base_at_pos_i = if i == 0 {
                    warmup_b_p
                } else {
                    base_argmax_k.get(i - 1).copied()
                };
                let d_tok = drafter_tokens[i];
                if let Some(b) = base_at_pos_i {
                    if d_tok == b {
                        accept_len += 1;
                        continue;
                    }
                }
                // Greedy mismatch → fall through to typical-acceptance
                // if the env mode is on AND we have logits for this row.
                if let Some((ref warmup_logits, ref k_logits)) = typical_buffers {
                    let row: &[f32] = if i == 0 {
                        &warmup_logits[..]
                    } else {
                        let off = (i - 1) * vocab_sz;
                        &k_logits[off..off + vocab_sz]
                    };
                    let d_idx = d_tok as usize;
                    if d_idx < vocab_sz {
                        let inv_t = 1.0f32 / typical_temp.max(1e-6);
                        let mut max_l = f32::NEG_INFINITY;
                        for &l in row {
                            if l.is_finite() && l > max_l { max_l = l; }
                        }
                        let mut lse: f64 = 0.0;
                        for &l in row {
                            if !l.is_finite() { continue; }
                            lse += (((l - max_l) * inv_t) as f64).exp();
                        }
                        let log_p_b = ((row[d_idx] - max_l) * inv_t) as f64
                            - lse.ln();
                        let log_q = drafter_log_q
                            .get(i)
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY)
                            as f64;
                        let log_accept = (log_p_b - log_q + bias_env).min(0.0);
                        let u = next_u01().max(1e-300);
                        let log_u = u.ln();
                        if log_u <= log_accept {
                            accept_len += 1;
                            continue;
                        }
                    }
                }
                break;
            }

            // Stats.
            *self.last_spec_stats.lock().unwrap() = Some(LastSpecStats {
                drafted: k_actual,
                accepted: accept_len as u32,
                cumulative_decoded: (accept_len + 1) as u32,
            });

            // Emit accepted prefix + 1 divergence/bonus token.
            //
            //   If accept_len == K: all drafts accepted; bonus = base's
            //     prediction at P+K = base_argmax_K[K-1]
            //   If accept_len < K: divergence at position P+accept_len;
            //     the correct base token there is:
            //       accept_len == 0 → warmup _base_first_tok[0]
            //       accept_len >= 1 → base_argmax_K[accept_len - 1]
            let mut emitted: Vec<u32> = drafter_tokens
                .iter()
                .take(accept_len)
                .copied()
                .collect();
            if accept_len == kmax {
                if let Some(b) = base_argmax_k.last().copied() {
                    emitted.push(b);
                }
            } else if accept_len == 0 {
                if let Some(b) = warmup_b_p {
                    emitted.push(b);
                }
            } else {
                if let Some(b) = base_argmax_k.get(accept_len - 1).copied() {
                    emitted.push(b);
                }
            }
            let _ = bonus_tok_vec; // not used; equivalent to base_argmax_k[K-1]

            tracing::info!(
                spec_k,
                k_actual,
                accept_len,
                accept_rate = accept_len as f32 / k_actual as f32,
                emitted = emitted.len(),
                drafted = ?drafter_tokens,
                warmup_b_p = ?warmup_b_p,
                base_argmax = ?base_argmax_k,
                "Gemma 4 BATCHED-verify spec-decode (single batched-prefill of K drafts + lm_head M=K)"
            );

            // Commit 46 (Codex review item #3+#4) — rollback is now
            // STRUCTURALLY MANDATORY in the batched verify path.
            //
            // The verify call wrote (P + drafter_tokens.len()) tokens
            // worth of K/V slots into the persistent KV cache and
            // updated prefix_cache.last_tokens / committed_prefix_len
            // as if all K drafts were committed. In reality only
            // accept_len of them are; the divergence/bonus token at
            // position P+accept_len has NO K/V written yet (it was a
            // pure argmax of a stored hidden, never fed through the
            // base layer stack).
            //
            // Was env-gated (RVLLM_GEMMA4_SPEC_BATCHED_ROLLBACK=1)
            // until commit 46. Codex review item #3 flagged this as
            // a correctness hazard: without rollback the next iter's
            // prefix-cache match can hit a stale rejected-draft slot
            // and silently corrupt drafter input. Item #4 made the
            // same argument: rollback is not optional tuning —
            // structurally required by the algorithm. So it's now
            // unconditional inside this branch.
            //
            // Note: this is metadata-only. The KV cache PHYSICAL
            // slots beyond the new boundary still contain stale
            // bytes; the next iter's prefill at those positions
            // overwrites them.
            {
                if let Ok(mut guard) = self.prefix_cache.lock() {
                    if let Some(pc) = guard.as_mut() {
                        let target_len = prompt_ids.len() + accept_len;
                        if pc.last_tokens.len() > target_len {
                            pc.last_tokens.truncate(target_len);
                        }
                        let chunk_size: u32 = std::env::var("RVLLM_PREFILL_CHUNK_SIZE")
                            .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
                        let batch_prefill =
                            parse_truthy_env("RVLLM_BATCH_PREFILL").unwrap_or(false);
                        let target_u32 = target_len as u32;
                        pc.committed_prefix_len = if batch_prefill && chunk_size == 0 {
                            0
                        } else if batch_prefill && chunk_size > 0 {
                            (target_u32 / chunk_size) * chunk_size
                        } else {
                            target_u32
                        };
                        tracing::debug!(
                            target_len,
                            committed = pc.committed_prefix_len,
                            "spec batched rollback: prefix-cache rewound"
                        );
                    }
                }
            }

            // Commit 43 — skip-warmup arming.
            //
            // When RVLLM_GEMMA4_SPEC_SKIP_WARMUP=1 and accept_len >= 1:
            //   * Drop the bonus/divergence emit so the next iter
            //     conditions on K-buffer[accept_len - 1] without an
            //     uncommitted-hidden gap (the bonus token's hidden was
            //     never computed by base).
            //   * DtoD copy K-buffer[accept_len - 1] into
            //     base_last_hidden_ptr (post-final-norm hidden at
            //     position P+accept_len-1 — the last accepted draft).
            //   * Cache base_argmax_K[accept_len - 1] for next iter's
            //     warmup_b_p (= base's prediction at the position the
            //     next iter's drafter step 0 will emit at).
            //   * Arm `skip_next_warmup`. Next call to
            //     run_generate_speculative sees the flag, bypasses the
            //     run_generate(prompt, max_new=1) warmup call, and
            //     reads _base_first_tok[0] from saved_warmup_b_p.
            //
            // This is the headline speedup: -1 base prefill per iter.
            // Net per-iter cost in skip mode + accept_len > 0:
            //   - K drafter forwards (cheap)
            //   - 1 batched-prefill base call (K rows)  [verify]
            //   - lm_head_K + final_norm_K + argmax_K   [verify]
            // Vs baseline (no spec): 1 base decode per emitted token.
            // Each iter emits accept_len tokens. Breakeven shifts to
            // ~0 — any positive accept_rate yields speedup.
            // Commit 51 (Phase C-2): the env gate is dropped — when
            // we're in the batched-verify branch (= batched_verify_mode
            // was true at function entry), arming skip-warmup for the
            // next iter is structurally part of the protocol, not an
            // optional knob. Subsequent iter's skip_warmup_active check
            // honors this atomic unconditionally.
            let skip_warmup_env = true;
            let mut emit_for_skip = emitted.clone();
            if skip_warmup_env && accept_len >= 1 && accept_len <= kmax {
                // Drop the bonus/divergence token — next iter will
                // re-derive the prediction at position P+accept_len
                // via the cached saved_warmup_b_p.
                emit_for_skip.truncate(accept_len);
                // DtoD copy K-buffer[accept_len - 1] → base_last_hidden_ptr.
                let dst_hidden = self
                    .base_last_hidden_ptr
                    .load(std::sync::atomic::Ordering::Acquire);
                let k_src = self
                    .base_last_k_hidden_ptr
                    .load(std::sync::atomic::Ordering::Acquire);
                if dst_hidden != 0 && k_src != 0 {
                    let row_bytes = (hidden_u as usize) * 2;
                    let src_off = ((accept_len - 1) as u64) * (row_bytes as u64);
                    let rc = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_hidden,
                        k_src + src_off,
                        row_bytes,
                        stream as cudarc::driver::sys::CUstream,
                    );
                    if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "spec batched skip-warmup: K-row → base_last_hidden DtoD",
                            rvllm_core::CudaErrorKind::MemcpyFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                    // Commit 57 (codex #4): no fence needed. The
                    // DtoD copy is enqueued on `self.stream` and the
                    // next consumer (drafter step 0 of the next spec
                    // iter) reads `base_last_hidden_ptr` from the
                    // same stream — CUDA stream ordering guarantees
                    // the read sees the copied bytes. The host never
                    // reads the copy. A defensive fence here
                    // serialised CPU↔GPU per successful spec iter
                    // for no functional reason.
                    // Cache the next iter's warmup base argmax.
                    let cached_b = base_argmax_k[accept_len - 1];
                    self.saved_warmup_b_p
                        .store(cached_b, std::sync::atomic::Ordering::Release);
                    self.skip_next_warmup
                        .store(true, std::sync::atomic::Ordering::Release);
                }
            }
            let emitted_final = if skip_warmup_env && accept_len >= 1 {
                emit_for_skip
            } else {
                emitted
            };

            // Reclaim scratch (above-scratch K-buffer + base_last_hidden
            // survive). Host-side `emitted` already populated.
            unsafe { self.arena.restore(arena_ck_at_entry); }
            return Ok(emitted_final);
        }

        // Commit 24 + 26: K-prefix verify. accept_len is the longest
        // prefix where drafter agrees with base, either via:
        //   - strict greedy match (drafter[i] == base[i]) — always on
        //   - LOSSY GREEDY ratio test (when env
        //     `RVLLM_GEMMA4_SPEC_LOSSY_THRESHOLD` is set to a value
        //     in (0,1]): accept if p_base(D_i) / p_base(B_i) >= T.
        //     Implemented via captured per-step base logits in
        //     `spec_decode_step_logits` (commit 26).
        //
        // Important: lossy greedy changes the model output relative
        // to plain greedy decode. It trades some quality for higher
        // accept rate. Default is strict (threshold = 0).
        if want_capture {
            self.spec_logits_capture_active
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
        let kmax = drafter_tokens.len().min(_base_first_tok.len());
        let mut accept_len: usize = 0;
        let captured = if want_capture {
            Some(self.spec_decode_step_logits.lock().unwrap())
        } else {
            None
        };
        let vocab_sz = arch.vocab_size as usize;
        // Commit 29 (quick-check): typical-acceptance temperature.
        // Read from the user's sampling config when typical mode is
        // on; ignored otherwise.
        let typical_temp: f32 = if typical_mode {
            match sampling {
                SamplingConfig::Stochastic { temperature, .. } => temperature.max(1e-6),
                SamplingConfig::Greedy => 1.0,
            }
        } else { 0.0 };
        // Simple xorshift for the per-position acceptance Us. Keep
        // it independent of the drafter sampling LCG so reseeding
        // one doesn't perturb the other.
        let mut accept_rng_state: u64 = next_rand_f32_spec
            .wrapping_add(0x517C_C1B7_2722_0A95);
        let mut next_u01 = || -> f64 {
            accept_rng_state ^= accept_rng_state << 13;
            accept_rng_state ^= accept_rng_state >> 7;
            accept_rng_state ^= accept_rng_state << 17;
            ((accept_rng_state >> 33) as f64) / ((1u64 << 31) as f64)
        };
        for i in 0..kmax {
            let d = drafter_tokens[i] as usize;
            let b = _base_first_tok[i] as usize;
            if d == b {
                accept_len += 1;
                continue;
            }
            // Commit 29 (quick-check): typical-acceptance via
            // modified-rejection-sampling in log space.
            //
            // Important caveat: this uses base logits that are
            // BASE-SELF-CONDITIONED (row i was computed assuming base
            // sampled its own argmax at position i-1, NOT D_{i-1}).
            // That's not the strict Leviathan/Kalman procedure — for
            // mathematically correct typical-acceptance the base
            // logits must be DRAFT-CONDITIONED, which needs a new
            // step-and-get-logits base API (deferred). This
            // quick-check gives an upper bound on accept rate so we
            // can decide whether the full plumbing is worth the
            // additional session.
            if typical_mode {
                if let Some(buf) = captured.as_ref() {
                    let row_off = i * vocab_sz;
                    if buf.len() >= row_off + vocab_sz && d < vocab_sz {
                        let row = &buf[row_off..row_off + vocab_sz];
                        // log p_base(D_i) at this temperature.
                        let inv_t = 1.0f32 / typical_temp;
                        let mut max_l = f32::NEG_INFINITY;
                        for &l in row { if l.is_finite() && l > max_l { max_l = l; } }
                        let mut lse: f64 = 0.0;
                        for &l in row {
                            if !l.is_finite() { continue; }
                            lse += (((l - max_l) * inv_t) as f64).exp();
                        }
                        let log_p_b = ((row[d] - max_l) * inv_t) as f64
                            - lse.ln();
                        let log_q = drafter_log_q
                            .get(i).copied().unwrap_or(f32::NEG_INFINITY) as f64;
                        // log_accept = min(0, log p_b - log q)
                        // Phase D: bias sourced from
                        // SpecDecodeRequestConfig (read once at fn entry).
                        // Was a per-row env::var call — wasted work.
                        let log_accept = (log_p_b - log_q + spec_cfg.accept_bias).min(0.0);
                        let u = next_u01().max(1e-300);
                        let log_u = u.ln();
                        if log_u <= log_accept {
                            accept_len += 1;
                            continue;
                        }
                    }
                }
            }
            // Lossy greedy ratio test (env-gated).
            if lossy_threshold > 0.0 {
                if let Some(buf) = captured.as_ref() {
                    let row_off = i * vocab_sz;
                    if buf.len() >= row_off + vocab_sz && d < vocab_sz && b < vocab_sz {
                        let row = &buf[row_off..row_off + vocab_sz];
                        let l_d = row[d];
                        let l_b = row[b];
                        let ratio = (l_d - l_b).exp();
                        if std::env::var("RVLLM_SPEC_DEBUG").as_deref() == Ok("1") {
                            eprintln!(
                                "[spec-lossy] i={} D={} B={} l_D={:.3} l_B={:.3} ratio={:.6e} threshold={:.3}",
                                i, d, b, l_d, l_b, ratio, lossy_threshold,
                            );
                        }
                        if ratio.is_finite() && ratio >= lossy_threshold {
                            accept_len += 1;
                            continue;
                        }
                    }
                }
            }
            break;
        }
        drop(captured);
        let base_prefix: Vec<u32> = _base_first_tok.iter().take(kmax).copied().collect();
        let accept_rate: f32 = if drafter_tokens.is_empty() {
            0.0
        } else {
            accept_len as f32 / drafter_tokens.len() as f32
        };
        tracing::info!(
            spec_k,
            drafter_tokens = ?drafter_tokens,
            base_tokens = ?base_prefix,
            accept_len,
            accept_rate,
            base_total_tokens = _base_first_tok.len(),
            "Gemma 4 speculative K-prefix verify (sequential; base ran K \
             sequential decodes — batched-verify speedup deferred)"
        );
        // Commit 25: stash for the worker to drain and emit as a
        // SpeculativeStep event → X-RVLLM-Accept-Rate response header.
        *self.last_spec_stats.lock().unwrap() = Some(LastSpecStats {
            drafted: drafter_tokens.len() as u32,
            accepted: accept_len as u32,
            cumulative_decoded: _base_first_tok.len() as u32,
        });
        // Commit 35: when RVLLM_GEMMA4_SPEC_EMIT_ACCEPTED=1, emit the
        // accepted drafter prefix + 1 base bonus token instead of the
        // full base sequence. This makes the response text actually
        // reflect accepted drafts — proves the wire works end-to-end.
        // Quality tradeoff: with bias>0 the drafter tokens may differ
        // from base argmax (we accept them via typical-acceptance),
        // so output text DIFFERS from plain greedy. The ~speedup
        // however only materializes once batched-verify replaces the
        // current sequential base decode path (= task #2 proper).
        // Phase D: emit_accepted comes from SpecDecodeRequestConfig
        // (read once at fn entry). Used to be a per-request
        // env::var lookup here even though the upstream wrappers
        // had ALREADY set the env to "1" via the scribble-globals
        // pattern flagged by codex priority 2.
        if spec_cfg.emit_accepted
            && accept_len > 0
        {
            let mut emitted: Vec<u32> = drafter_tokens.iter()
                .take(accept_len).copied().collect();
            // Append 1 bonus base token at position accept_len (base's
            // own argmax at the divergence point).
            if let Some(&b) = _base_first_tok.get(accept_len) {
                emitted.push(b);
            }
            return Ok(emitted);
        }
        Ok(_base_first_tok)
    }

    /// Drain and return the last spec-decode request's K-prefix
    /// accept stats. Called by the worker right after
    /// `run_generate_speculative` returns; the value is then
    /// re-emitted as a `GenerateEvent::SpeculativeStep` event.
    /// Returns `None` if no spec-decode request has happened yet
    /// or if the previous value has already been taken.
    pub fn take_last_spec_stats(&self) -> Option<LastSpecStats> {
        self.last_spec_stats.lock().unwrap().take()
    }

    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    /// Commit 54 (codex priority 1, stage 1a): the sqrt(backbone)
    /// scale on the embed half of `workspace.pre_projection_in`,
    /// previously inlined into the K-step body inside
    /// `run_generate_speculative`. Per HF/vLLM Gemma4MTP reference,
    /// the drafter was trained with `inputs_embeds` pre-multiplied by
    /// sqrt(backbone_hidden_size) on top of the base embedding's own
    /// pre-scale. Without it, embed-half RMS is wildly off and
    /// pre_projection sees an imbalanced concat → Q direction wrong →
    /// softmax uniform → accept_rate=0. Verified via the manual
    /// PyTorch reference in `v3/tools/manual_drafter_reference.py`.
    ///
    /// First building block of the larger drafter-step extraction
    /// codex called out. Used here and in the upcoming
    /// `forward_one_drafter_step` method.
    #[cfg(feature = "cuda")]
    unsafe fn apply_pre_projection_embed_scale(
        &self,
        workspace: &crate::gemma4_drafter::DrafterStepWorkspace,
        backbone_hidden_size: i32,
        stream: u64,
    ) -> Result<()> {
        use cudarc::driver::sys::*;
        let sqrt_bb: f32 = (backbone_hidden_size as f32).sqrt();
        let mut x = workspace.pre_projection_in; // first-half base ptr
        let mut s = sqrt_bb;
        let mut n: i32 = backbone_hidden_size;
        let args = [
            (&mut x) as *mut u64 as *mut core::ffi::c_void,
            (&mut s) as *mut f32 as *mut core::ffi::c_void,
            (&mut n) as *mut i32 as *mut core::ffi::c_void,
        ];
        let block: u32 = 256;
        let grid: u32 = ((n as u32 + block - 1) / block).max(1);
        let rc = cuLaunchKernel(
            self.fused.fn_scale_inplace_f16.raw() as CUfunction,
            grid, 1, 1, block, 1, 1, 0,
            stream as CUstream,
            args.as_ptr() as *mut *mut core::ffi::c_void,
            core::ptr::null_mut(),
        );
        if rc != CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::cuda(
                "drafter pre_projection embed half sqrt(backbone) scale",
                rvllm_core::CudaErrorKind::LaunchFailed,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        Ok(())
    }

    unsafe fn run_drafter_pre_projection(
        &self,
        drafter: &crate::gemma4_drafter::Gemma4DrafterRuntime,
        workspace: &crate::gemma4_drafter::DrafterStepWorkspace,
    ) -> Result<()> {
        let hidden = drafter.arch.hidden_size;
        let pre_in = drafter.arch.pre_projection_in_dim;
        let stream = self.stream.raw();

        let require_ptr = |name: &'static str, ptr: u64| -> Result<()> {
            if ptr != 0 {
                return Ok(());
            }
            Err(rvllm_core::RvllmError::Attention {
                err: rvllm_core::AttentionError::FeatureNotAvailable {
                    op: name,
                    backend: "Gemma4SpecDecode",
                },
                ctx: rvllm_core::AttnCtx {
                    op: name,
                    stream,
                    num_seqs: 1,
                    head_dim: drafter.arch.head_dim_global as u32,
                },
                bt: std::backtrace::Backtrace::capture(),
            })
        };

        require_ptr("drafter.pre_projection_in", workspace.pre_projection_in)?;
        require_ptr("drafter.pre_projection.weight", drafter.top.pre_projection)?;
        require_ptr("drafter.gemm_f32", workspace.gemm_f32)?;
        require_ptr("drafter.hidden", workspace.hidden)?;

        self.cublaslt.f16_gemm_f32(
            workspace.pre_projection_in,
            drafter.top.pre_projection,
            workspace.gemm_f32,
            1,
            hidden as i32,
            pre_in as i32,
            stream,
        )?;
        launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            workspace.gemm_f32,
            workspace.hidden,
            hidden as i32,
        )?;
        tracing::trace!(
            hidden,
            pre_in,
            "completed Gemma 4 speculative drafter pre_projection",
        );
        Ok(())
    }

    /// Spec-decode commit 7+8 (+12 attn finisher): Q-side and
    /// attention-side close-out of one drafter sliding layer.
    /// Runs `input_layernorm → q_proj → q_norm → RoPE`, then
    /// expects the CALLER to fire `populate_shadow_kv_from_base` +
    /// `launch_cross_attn_sliding` (which write into
    /// `workspace.attn_out`). After this helper, the caller may
    /// invoke `run_drafter_layer_attn_finisher` to fold attn_out
    /// back into `workspace.hidden` via `o_proj +
    /// post_attention_layernorm + residual` — kept as a separate
    /// method to keep the cross-attn dispatch site visible.
    ///
    /// No new CUDA kernels — reuses `RmsnormInplaceLaunch`,
    /// `cublaslt.f16_gemm_f32`, `launch_cast_f32_to_f16`, and
    /// `fused_rope_partial_f16kv` with `num_kv_heads=0` so the K/V
    /// branch is dead-code and null KV pointers are safe.
    ///
    /// Output: `workspace.q` holds f16 `[num_heads * effective_head_dim]`
    /// Q after per-head RMSNorm + partial NeoX RoPE. The cross-
    /// attention against the base's source-layer K/V remains pending
    /// — the caller (`run_generate_speculative`) returns
    /// `FeatureNotAvailable` after invoking this helper.
    ///
    /// **Layer-type restriction (commit 8 scope)**: only sliding
    /// drafter layers are supported here. The global layer (layer 3
    /// on E4B) uses partial-factor RoPE; that path lands with the
    /// cross-attn integration to keep this commit narrow.
    ///
    /// `position` is the absolute base-side position the drafter is
    /// scoring (i.e. the position at which the next token would be
    /// emitted). In commit 5a the spec path still seeds this as 0
    /// because real base prefill hasn't run — that's an intentional
    /// no-op at the RoPE level (`cos=1`, `sin=0`).
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    unsafe fn run_drafter_layer_q_side(
        &self,
        drafter: &crate::gemma4_drafter::Gemma4DrafterRuntime,
        workspace: &crate::gemma4_drafter::DrafterStepWorkspace,
        layer_idx: usize,
        position: u32,
    ) -> Result<()> {
        let hidden = drafter.arch.hidden_size;
        let num_heads = drafter.arch.num_attention_heads;
        let layer = drafter.layers.get(layer_idx).ok_or_else(|| {
            rvllm_core::RvllmError::Loader {
                err: rvllm_core::LoaderError::Corrupt {
                    detail: format!(
                        "run_drafter_layer_q_side: layer_idx {layer_idx} \
                         out of range (drafter has {} layers)",
                        drafter.layers.len()
                    ),
                },
                ctx: rvllm_core::LoaderCtx {
                    path: drafter.shard_path.clone(),
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            }
        })?;
        let eff_hd = layer.effective_head_dim;
        let q_rows = num_heads * eff_hd;
        let stream = self.stream.raw();
        let eps = drafter.arch.rms_norm_eps;

        // Step 0 (commit 22): snapshot the pre-norm residual into
        // workspace.residual1 before the in-place input_layernorm
        // destroys it. attn_finisher reads back from residual1 for
        // the residual_1 add (Gemma's `residual + post_attn_norm(...)`
        // semantics).
        {
            use cudarc::driver::sys::*;
            let r = cuMemcpyDtoDAsync_v2(
                workspace.residual1,
                workspace.hidden,
                hidden * 2,
                stream as CUstream,
            );
            if r != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "drafter_q_side residual1 snapshot DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Commit 31h: Q-bisect probe.
        let q_bisect = layer_idx == 0
            && std::env::var("RVLLM_SPEC_DEBUG_Q_BISECT").as_deref() == Ok("1");
        let q_probe = |label: &str, ptr: u64, n: usize| {
            if !q_bisect { return; }
            let _ = self.stream.fence();
            let mut buf = vec![0u16; n];
            unsafe {
                let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut _, ptr, n * 2);
            }
            let mut max_abs = 0f32;
            let mut sum_sq = 0f64;
            for &b in &buf {
                let v = half::f16::from_bits(b).to_f32();
                if v.is_finite() {
                    if v.abs() > max_abs { max_abs = v.abs(); }
                    sum_sq += (v * v) as f64;
                }
            }
            let rms = (sum_sq / n as f64).sqrt() as f32;
            let head8: Vec<f32> = buf.iter().take(8)
                .map(|&b| half::f16::from_bits(b).to_f32()).collect();
            eprintln!("[spec-qbisect] {} max={:.3} rms={:.4} head8={:?}",
                      label, max_abs, rms, head8);
        };
        q_probe("pre_inputln(hidden)", workspace.hidden, hidden);

        // Step 1: input_layernorm on workspace.hidden, in place.
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1,
            hidden: hidden as u32,
            eps,
        }
        .launch(
            self.fused.fn_rmsnorm,
            workspace.hidden,
            layer.input_layernorm,
            stream,
        )?;
        q_probe("post_inputln(hidden)", workspace.hidden, hidden);

        // Step 2: q_proj = hidden @ Wq^T  → f32 GEMM scratch.
        //   Wq shape on disk: [q_rows, hidden] BF16->F16.
        //   m=1, n=q_rows, k=hidden.
        self.cublaslt.f16_gemm_f32(
            workspace.hidden,
            layer.self_attn_q_proj,
            workspace.gemm_f32,
            1,
            q_rows as i32,
            hidden as i32,
            stream,
        )?;

        // Step 3: cast f32 → f16 → workspace.q.
        launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            workspace.gemm_f32,
            workspace.q,
            q_rows as i32,
        )?;
        q_probe("post_q_proj(q)", workspace.q, q_rows);

        // Step 4: per-head q_norm. Treats each of `num_heads` heads
        // of length `effective_head_dim` as a separate "token" so
        // the existing inplace RMSNorm launcher fans out cleanly
        // (gamma is shared across heads, shape [effective_head_dim]).
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: num_heads as u32,
            hidden: eff_hd as u32,
            eps,
        }
        .launch(
            self.fused.fn_rmsnorm,
            workspace.q,
            layer.self_attn_q_norm,
            stream,
        )?;
        q_probe("post_q_norm(q)", workspace.q, q_rows);


        // Step 5: partial NeoX RoPE on Q. Layer 0..2 are sliding
        // with rotary_dim == effective_head_dim (full rotation per
        // the drafter config). The shared `fused_rope_partial_f16kv`
        // kernel rotates Q in place and skips the K/V branch entirely
        // when `num_kv_heads == 0` is passed — its inner guard is
        // `if (head_idx < num_kv_heads)`, dead-code with the count
        // zeroed, so the null K/V pointers below are never
        // dereferenced.
        // Commit 14: route both sliding and global layers. Sliding
        // layers use full rotation (rotary_dim == head_dim_sliding)
        // and the sliding cos/sin tables; global layers use partial
        // rotation (`partial_rotary_factor=0.25` on E4B → rotary_dim
        // = head_dim_global*0.25 = 128) with the global tables.
        let is_global = matches!(
            layer.layer_type,
            rvllm_loader::gemma4_drafter::DrafterLayerType::Full
        );
        // Drafter config explicitly carries partial_rotary_factor=0.25;
        // hardcoded here (same convention as base Gemma 4 E4B) until
        // `Gemma4DrafterArch` exposes the field.
        const PARTIAL_ROTARY_FACTOR_GLOBAL: f32 = 0.25;
        let rotary_dim: i32 = if is_global {
            ((eff_hd as f32) * PARTIAL_ROTARY_FACTOR_GLOBAL) as i32
        } else {
            eff_hd as i32
        };
        let (cos_table_off, sin_table_off) = if is_global {
            (self.model.rope_cos_global.offset_bytes,
             self.model.rope_sin_global.offset_bytes)
        } else {
            (self.model.rope_cos_sliding.offset_bytes,
             self.model.rope_sin_sliding.offset_bytes)
        };
        // Commit 31 experiment: RVLLM_SPEC_Q_ROPE_MODE selects
        // Q-side RoPE behavior for the drafter:
        //   current  — pass `position` to base RoPE table (default)
        //   pos0     — pass 0 (no rotation: cos=1, sin=0 at pos 0)
        //   no_rope  — skip Q-side RoPE launch entirely
        // Codex hypothesis: vLLM uses the draft model's own RoPE
        // object, not the base's table. If conv/period differs, our
        // Q is rotated inconsistently with base K and the cross-attn
        // softmax saturates on the wrong position.
        let q_rope_mode = std::env::var("RVLLM_SPEC_Q_ROPE_MODE")
            .unwrap_or_else(|_| "current".to_string());
        let effective_pos: i32 = match q_rope_mode.as_str() {
            "pos0" => 0,
            _ => position as i32,
        };
        let skip_rope = q_rope_mode == "no_rope";
        let pos_region = self.arena.region("spec_drafter_q_rope_pos", 4, 16)?;
        pos_region.copy_from_host(&effective_pos.to_le_bytes())?;
        if !skip_rope {
            use cudarc::driver::sys::*;
            let mut q_in: u64 = workspace.q;
            let mut k_in: u64 = 0;
            let mut v_in: u64 = 0;
            let mut q_out: u64 = workspace.q;
            let mut key_cache: u64 = 0;
            let mut value_cache: u64 = 0;
            let mut cos_table: u64 = cos_table_off;
            let mut sin_table: u64 = sin_table_off;
            let mut positions_ptr: u64 = pos_region.device_ptr();
            let mut slot_mapping_ptr: u64 = 0;
            let mut num_tokens_arg: i32 = 1;
            let mut num_heads_arg: i32 = num_heads as i32;
            let mut num_kv_heads_arg: i32 = 0;
            let mut head_dim_arg: i32 = eff_hd as i32;
            let mut rotary_dim_arg: i32 = rotary_dim;
            let args: [*mut core::ffi::c_void; 15] = [
                &mut q_in           as *mut _ as *mut _,
                &mut k_in           as *mut _ as *mut _,
                &mut v_in           as *mut _ as *mut _,
                &mut q_out          as *mut _ as *mut _,
                &mut key_cache      as *mut _ as *mut _,
                &mut value_cache    as *mut _ as *mut _,
                &mut cos_table      as *mut _ as *mut _,
                &mut sin_table      as *mut _ as *mut _,
                &mut positions_ptr  as *mut _ as *mut _,
                &mut slot_mapping_ptr as *mut _ as *mut _,
                &mut num_tokens_arg as *mut _ as *mut _,
                &mut num_heads_arg  as *mut _ as *mut _,
                &mut num_kv_heads_arg as *mut _ as *mut _,
                &mut head_dim_arg   as *mut _ as *mut _,
                &mut rotary_dim_arg as *mut _ as *mut _,
            ];
            // Grid: (num_tokens, max(num_heads, num_kv_heads), 1).
            // With num_kv_heads=0 grid_y collapses to num_heads.
            // Block: (head_dim / 2). Each thread covers a (low, high)
            // pair within its head.
            let rc = cuLaunchKernel(
                self.fused.fn_fused_rope_partial_f16kv.raw() as CUfunction,
                1, num_heads as u32, 1,
                (eff_hd as u32) / 2, 1, 1,
                0,
                stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "fused_rope_partial_f16kv launch (drafter Q-only)",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        tracing::trace!(
            layer_idx,
            num_heads,
            effective_head_dim = eff_hd,
            position,
            "completed Gemma 4 speculative drafter layer Q-side \
             (input_layernorm + q_proj + q_norm + RoPE)",
        );
        Ok(())
    }

    /// Spec-decode commit 12: attention-side close-out for one
    /// drafter layer. Runs `o_proj → post_attention_layernorm →
    /// residual_1` (`workspace.hidden += post_norm(o_proj(attn_out))`).
    ///
    /// Inputs:
    ///   * `workspace.attn_out` — f16 `[num_heads * effective_head_dim]`,
    ///     freshly written by `launch_cross_attn_sliding`.
    ///   * `workspace.hidden` — the rolling residual stream
    ///     (= drafter's "x" before this layer).
    ///   * `workspace.proj_f16` — hidden-sized scratch for the
    ///     o_proj output; mutated in place by the norm before the
    ///     residual add.
    ///
    /// Output:
    ///   * `workspace.hidden` ← hidden + post_attention_layernorm(o_proj(attn_out))
    ///
    /// No new CUDA kernels — three launches that reuse
    /// `cublaslt.f16_gemm_f32`, `launch_cast_f32_to_f16`,
    /// `RmsnormInplaceLaunch`, and `fn_vector_add`.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    unsafe fn run_drafter_layer_attn_finisher(
        &self,
        drafter: &crate::gemma4_drafter::Gemma4DrafterRuntime,
        workspace: &crate::gemma4_drafter::DrafterStepWorkspace,
        layer_idx: usize,
    ) -> Result<()> {
        let hidden = drafter.arch.hidden_size;
        let num_heads = drafter.arch.num_attention_heads;
        let layer = drafter.layers.get(layer_idx).ok_or_else(|| {
            rvllm_core::RvllmError::Loader {
                err: rvllm_core::LoaderError::Corrupt {
                    detail: format!(
                        "run_drafter_layer_attn_finisher: layer_idx \
                         {layer_idx} out of range (drafter has {} layers)",
                        drafter.layers.len()
                    ),
                },
                ctx: rvllm_core::LoaderCtx {
                    path: drafter.shard_path.clone(),
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            }
        })?;
        let eff_hd = layer.effective_head_dim;
        let q_rows = num_heads * eff_hd;
        let stream = self.stream.raw();
        let eps = drafter.arch.rms_norm_eps;

        // Step 1: o_proj = attn_out @ Wo^T  → f32 GEMM scratch.
        //   Wo shape on disk: [hidden, q_rows] BF16->F16.
        //   m=1, n=hidden, k=q_rows.
        self.cublaslt.f16_gemm_f32(
            workspace.attn_out,
            layer.self_attn_o_proj,
            workspace.gemm_f32,
            1,
            hidden as i32,
            q_rows as i32,
            stream,
        )?;

        // Step 2: cast f32 → f16 → workspace.proj_f16.
        launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            workspace.gemm_f32,
            workspace.proj_f16,
            hidden as i32,
        )?;

        // Step 3: post_attention_layernorm on workspace.proj_f16 in
        //   place. HF semantics:
        //     residual + post_attention_layernorm(self_attn_out)
        //   — so the norm is applied to the o_proj output BEFORE
        //   the residual add, not after the sum.
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1,
            hidden: hidden as u32,
            eps,
        }
        .launch(
            self.fused.fn_rmsnorm,
            workspace.proj_f16,
            layer.post_attention_layernorm,
            stream,
        )?;

        // Step 4: residual_1: workspace.hidden = workspace.residual1 +
        //   workspace.proj_f16. Reload the saved pre-norm residual
        //   from workspace.residual1 (snapshotted in q_side step 0),
        //   then vector_add the projected attn output.
        //
        //   Commit 22 fix: the previous code did
        //     workspace.hidden += workspace.proj_f16
        //   but workspace.hidden was the POST-input_layernorm tensor
        //   from q_side step 1 — the residual stream had been
        //   destroyed by the in-place norm. Result: residual stream
        //   silently broken, drafter produced a deterministic but
        //   wrong token every prompt, accept_rate stuck at 0.0.
        {
            use cudarc::driver::sys::*;
            let r = cuMemcpyDtoDAsync_v2(
                workspace.hidden,
                workspace.residual1,
                hidden * 2,
                stream as CUstream,
            );
            if r != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "drafter_layer_attn_finisher residual1 reload",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            let mut dst = workspace.hidden;
            let mut src = workspace.proj_f16;
            let mut n: i32 = hidden as i32;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut n)   as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n as u32 + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.fused.fn_vector_add.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0,
                stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "drafter_layer_attn_finisher residual_1 vector_add",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        tracing::trace!(
            layer_idx,
            hidden,
            q_rows,
            "completed Gemma 4 speculative drafter layer attn-finisher \
             (o_proj + post_attention_layernorm + residual_1)",
        );
        Ok(())
    }

    /// Spec-decode commit 13: MLP-side close-out for one drafter
    /// layer. Runs:
    ///
    ///   `pre_feedforward_layernorm → gate_proj + up_proj →
    ///    gelu_tanh(gate)*up → down_proj → post_feedforward_layernorm
    ///    → residual_2`.
    ///
    /// Layer_scalar is NOT applied here. Per HF
    /// `Gemma4TextDecoderLayer.__init__` it is initialised to 1.0
    /// (`init.ones_(module.layer_scalar)`); E4B's trained drafter
    /// uses the default. A follow-up commit will add a real
    /// device-side scalar multiply if profile dumps show the
    /// scalar drifted away from 1.0.
    ///
    /// Buffer usage:
    ///   * `workspace.hidden` — residual stream in/out.
    ///   * Per-call arena scratch (`spec_drafter_residual2`) holds
    ///     the saved residual across the MLP path. Sized for
    ///     `hidden_size * 2` bytes (f16).
    ///   * Per-call arena scratch (`spec_drafter_gate_up`) holds the
    ///     concatenated `[gate || up]` rows that
    ///     `fused_gelu_mul_f16_kernel` expects. Sized for
    ///     `2 * intermediate_size * 2` bytes (f16).
    ///   * `workspace.gemm_f32` is shared by both GEMMs (cast→f16
    ///     immediately, so no overlap).
    ///
    /// No new CUDA kernels — five existing launchers
    /// (`cublaslt.f16_gemm_f32` ×3, `launch_cast_f32_to_f16` ×3,
    /// `RmsnormInplaceLaunch` ×2, `fn_gelu_mul`, `fn_vector_add`).
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    unsafe fn run_drafter_layer_mlp_finisher(
        &self,
        drafter: &crate::gemma4_drafter::Gemma4DrafterRuntime,
        workspace: &crate::gemma4_drafter::DrafterStepWorkspace,
        layer_idx: usize,
    ) -> Result<()> {
        let hidden = drafter.arch.hidden_size;
        let intermediate = drafter.arch.intermediate_size;
        let layer = drafter.layers.get(layer_idx).ok_or_else(|| {
            rvllm_core::RvllmError::Loader {
                err: rvllm_core::LoaderError::Corrupt {
                    detail: format!(
                        "run_drafter_layer_mlp_finisher: layer_idx \
                         {layer_idx} out of range (drafter has {} layers)",
                        drafter.layers.len()
                    ),
                },
                ctx: rvllm_core::LoaderCtx {
                    path: drafter.shard_path.clone(),
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            }
        })?;
        let stream = self.stream.raw();
        let eps = drafter.arch.rms_norm_eps;

        // Scratches.
        let residual2_region = self.arena.region(
            "spec_drafter_residual2", hidden * 2, 16,
        )?;
        let gate_up_region = self.arena.region(
            "spec_drafter_gate_up", 2 * intermediate * 2, 16,
        )?;
        let gate_offset_bytes: u64 = 0;
        let up_offset_bytes: u64 = (intermediate as u64) * 2;
        let gate_ptr = gate_up_region.device_ptr() + gate_offset_bytes;
        let up_ptr = gate_up_region.device_ptr() + up_offset_bytes;

        // Step 1: residual_2 = workspace.hidden  (DtoD copy)
        {
            use cudarc::driver::sys::*;
            let r = cuMemcpyDtoDAsync_v2(
                residual2_region.device_ptr(),
                workspace.hidden,
                hidden * 2,
                stream as CUstream,
            );
            if r != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "drafter_mlp_finisher residual_2 DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Step 2: pre_feedforward_layernorm on workspace.hidden in
        // place.
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1,
            hidden: hidden as u32,
            eps,
        }
        .launch(
            self.fused.fn_rmsnorm,
            workspace.hidden,
            layer.pre_feedforward_layernorm,
            stream,
        )?;

        // Step 3: gate_proj = hidden @ Wgate^T  → f32 → f16 into
        // gate half of gate_up_combined.
        self.cublaslt.f16_gemm_f32(
            workspace.hidden,
            layer.mlp_gate_proj,
            workspace.gemm_f32,
            1,
            intermediate as i32,
            hidden as i32,
            stream,
        )?;
        launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            workspace.gemm_f32,
            gate_ptr,
            intermediate as i32,
        )?;

        // Step 4: up_proj = hidden @ Wup^T  → f32 → f16 into up half.
        self.cublaslt.f16_gemm_f32(
            workspace.hidden,
            layer.mlp_up_proj,
            workspace.gemm_f32,
            1,
            intermediate as i32,
            hidden as i32,
            stream,
        )?;
        launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            workspace.gemm_f32,
            up_ptr,
            intermediate as i32,
        )?;

        // Step 5: fused_gelu_mul_f16: output = gelu_tanh(gate) * up.
        // The kernel reads gate AND up at index i, then writes
        // output[i] — safe in place against the gate half because
        // each thread writes after reading from the same index.
        // Grid: (num_tokens=1, 1, 1), block: 1024-thread striped over
        // intermediate.
        {
            use cudarc::driver::sys::*;
            let mut out_p = gate_ptr;       // write back into gate half
            let mut gate_up = gate_up_region.device_ptr();
            let mut inter_i = intermediate as i32;
            let args = [
                (&mut out_p)   as *mut u64 as *mut core::ffi::c_void,
                (&mut gate_up) as *mut u64 as *mut core::ffi::c_void,
                (&mut inter_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            // 1024 threads is the __launch_bounds__ ceiling; cap at
            // intermediate so we don't over-spawn for tiny dims.
            let block: u32 = 1024u32.min(intermediate as u32).max(1);
            // Commit 33: was `fn_gelu_mul` — that handle resolves to
            // `fused_gelu_mul_fp8_quant_kernel` (4-arg ABI:
            // output_fp8, output_scales, gate_up, intermediate). The
            // drafter MLP passes 3 args (output_f16, gate_up,
            // intermediate) so it needs the f16 variant.
            let rc = cuLaunchKernel(
                self.fused.fn_fused_gelu_mul_f16.raw() as CUfunction,
                1, 1, 1,
                block, 1, 1,
                0,
                stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "drafter_mlp_finisher fused_gelu_mul_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Step 6: down_proj = mlp_out @ Wdown^T → f32 → f16 into
        // workspace.hidden (overwrites the residual'd stream).
        self.cublaslt.f16_gemm_f32(
            gate_ptr,
            layer.mlp_down_proj,
            workspace.gemm_f32,
            1,
            hidden as i32,
            intermediate as i32,
            stream,
        )?;
        launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            workspace.gemm_f32,
            workspace.hidden,
            hidden as i32,
        )?;

        // Step 7: post_feedforward_layernorm on workspace.hidden in
        // place.
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1,
            hidden: hidden as u32,
            eps,
        }
        .launch(
            self.fused.fn_rmsnorm,
            workspace.hidden,
            layer.post_feedforward_layernorm,
            stream,
        )?;

        // Step 8: residual_2: workspace.hidden = residual2 +
        // workspace.hidden. fn_vector_add ABI is (dst, src, n) →
        // dst += src.
        {
            use cudarc::driver::sys::*;
            let mut dst = workspace.hidden;
            let mut src = residual2_region.device_ptr();
            let mut n: i32 = hidden as i32;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut n)   as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n as u32 + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.fused.fn_vector_add.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0,
                stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "drafter_mlp_finisher residual_2 vector_add",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Step 9 (commit 22): hidden *= layer_scalar[1].
        //
        // Codex round-3 inspection of the assistant checkpoint:
        //   L0 = 0.0320  L1 = 0.2051  L2 = 0.3594  L3 = 0.1641
        // The trained scalars are NOT 1.0 — they damp each layer's
        // contribution. Skipping them inflated the drafter final
        // hidden RMS by ~9x (observed 9.13 vs expected ~1) and made
        // the MaskedEmbedder argmax point at a wrong-but-deterministic
        // token (255970), driving accept_rate to 0.0 even though the
        // MaskedEmbedder kernel itself is correct (verified commit 21).
        //
        // Apply at the END of each drafter layer, after
        // residual_2 = residual + post_ff_norm(down_proj(...)) —
        // matches the assistant's layer epilogue semantics.
        {
            use cudarc::driver::sys::*;
            // Codex review item #5: layer_scalar is cached host-side
            // at drafter-load time (`Gemma4DrafterLayerPtrs::layer_scalar_f32`).
            // No per-step `stream.fence()` + `cuMemcpyDtoH_v2(2 bytes)`.
            // Saves 4×K sync points per outer spec iter (= 24 syncs
            // at K=6).
            let scalar_f32 = layer.layer_scalar_f32;

            if std::env::var("RVLLM_SPEC_DEBUG").as_deref() == Ok("1") {
                let mut h = vec![0u16; hidden as usize];
                let _ = cuMemcpyDtoH_v2(
                    h.as_mut_ptr() as *mut _,
                    workspace.hidden,
                    (hidden as usize) * 2,
                );
                let mut s = 0f64;
                for &b in &h {
                    let x = half::f16::from_bits(b).to_f32() as f64;
                    s += x * x;
                }
                let pre_rms = (s / h.len() as f64).sqrt() as f32;
                eprintln!(
                    "[spec-debug] layer {} scalar={:.4} pre_rms={:.4} \
                     expected_post_rms={:.4}",
                    layer_idx, scalar_f32, pre_rms, pre_rms * scalar_f32,
                );
            }

            let mut x = workspace.hidden;
            let mut s = scalar_f32;
            let mut n = hidden as i32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut s) as *mut f32 as *mut core::ffi::c_void,
                (&mut n) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n as u32 + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.fused.fn_scale_inplace_f16.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0,
                stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "drafter_mlp_finisher layer_scalar scale_inplace",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        tracing::trace!(
            layer_idx,
            hidden,
            intermediate,
            "completed Gemma 4 speculative drafter layer MLP-finisher \
             (pre_ff_norm + gate/up + gelu_mul + down + post_ff_norm + \
              residual_2 + layer_scalar)",
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub unsafe fn run_generate(
        &self,
        fn_embed: rvllm_kernels::KernelFn,
        fn_argmax: rvllm_kernels::KernelFn,
        prompt_ids: &[u32],
        max_new: usize,
        eos_ids: &[u32],
        // === NVFP4 SHADOW DIAGNOSTIC === per-request opt-in. When
        // false, the shadow allocator and dump hook do nothing
        // regardless of env settings. Gates the feature at the
        // request boundary so upstream-client scaffold calls
        // (zeroclaw classifier, internal probes) cannot accidentally
        // burn the one-shot latch.
        shadow_requested: bool,
        // Per-request sampling decision. `Greedy` keeps the legacy
        // argmax path; `Stochastic` activates host-side temperature +
        // top-p sampling with the supplied seed.
        sampling: SamplingConfig,
        // Optional cancellation signal. The decode loop checks this
        // flag at the top of every step and returns the tokens
        // produced so far if it transitions to `true`. Without this,
        // an HTTP-layer timeout / client disconnect would 504 the
        // request but the worker thread would stay blocked in
        // `run_generate` until natural completion, monopolising the
        // GPU and starving subsequent requests. Pass `None` from
        // bench / probe binaries that have no cancellation source.
        cancel: Option<&std::sync::atomic::AtomicBool>,
        // Optional per-token emission callback. Called from the
        // worker thread immediately after each token (prefill's
        // first token + every decode-loop token, in order) is
        // produced. Returning `false` is interpreted as "stop the
        // stream" — the decode loop breaks. Default `None` keeps
        // the legacy "all tokens returned at end" semantics so
        // bench / probe / ppl callers are unaffected. The HTTP
        // worker plumbs in a closure that does
        // `events_tx.blocking_send(GenerateEvent::Token { ... })`
        // for true tokenwise SSE delivery and to feed the handler-
        // side stop-string detector incrementally.
        mut on_token: Option<&mut dyn FnMut(u32) -> bool>,
        // Phase 3b vision: per-image (token_start_in_prompt, raw f16
        // bytes for [num_tokens, hidden] embeddings) tuples. Empty for
        // text-only requests. The splice fires inside the chunked
        // prefill embedding-gather path: any chunk that overlaps a
        // splice slot has its residual rows for that overlap overwritten
        // with the corresponding embedding bytes BEFORE the optional
        // bf16 widen, so all downstream layers see the vision rows
        // through the same dtype path as text.
        vision_splice: &[(usize, &[u8])],
        audio_splice: &[(usize, &[u8])],
    ) -> Result<Vec<u32>> {
        // Cycle 37 P2 (codex audit): max_new=0 used to underflow at
        // `0..max_new - 1`. Caller (handlers.rs::resolve_max_new) already
        // rejects 0 at the HTTP layer, but defend the runtime entry too
        // so internal callers (probes, future schedulers) cannot trip it.
        if max_new == 0 {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "max_new",
                    reason: "must be >= 1; max_new=0 underflows the decode loop"
                        .into(),
                },
                field: "max_new",
            });
        }
        let arch = &self.arch;
        let hidden = arch.hidden_size as u32;
        let vocab = arch.vocab_size as u32;
        let stream = self.stream.raw();

        // === HOST-SIDE TEMPERATURE SAMPLING ===
        // Greedy argmax at razor-thin margins (~0.1-0.5 logit units)
        // makes long-context tool-call generation produce confident
        // garbage. Sampling at temp>0 lets the model explore
        // alternative continuations when the top-1 winner is
        // questionable.
        //
        // `SamplingConfig::Greedy` is UNCONDITIONAL argmax. We do NOT
        // read `RVLLM_SAMPLING_TEMPERATURE` / `_TOP_P` here. The
        // previous fallback meant a stray env var on the production
        // box silently broke deterministic `temperature: 0` API
        // requests. Stochastic decode requires an explicit
        // `SamplingConfig::Stochastic` from the caller — there is no
        // "ambient" sampling state.
        let (sampling_temp, sampling_top_p, sampling_top_k, mut rng_state): (f32, f32, Option<u32>, u64) = match sampling {
            SamplingConfig::Stochastic { temperature, top_p, top_k, seed } => {
                (temperature, top_p, top_k, seed)
            }
            // Hard zero. The host-side sampler below is gated on
            // `sampling_temp > 0.0`, so the inert tuple here just
            // routes through the existing `ArgmaxLaunch` path
            // without ever entering `host_sample_token`.
            SamplingConfig::Greedy => (0.0, 1.0, None, 0),
        };
        let mut next_rand_f32 = || -> f32 {
            // 64-bit LCG (Knuth) — bit-for-bit compatible with the
            // pre-SamplingConfig path. Replaceable by Philox-4×32 if
            // cross-engine reproducibility against vLLM ever matters;
            // for now LCG + caller-supplied seed is enough for
            // request-level determinism (same seed → same output).
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 40) as f32) / (1u32 << 24) as f32
        };
        // Per-token sampling buffers, hoisted out of `host_sample_token`
        // so we allocate once per request instead of `vec![…; vocab]` +
        // `Vec<(u32,f32)>` collect on every decode step. For Gemma 4
        // vocab=263 168 that's ~1 MiB f32 + ~2 MiB tuple-vec saved
        // per token. The DtoH copy itself remains — replacing it
        // needs a CUDA sampling kernel, which is out of scope here;
        // this just stops the heap traffic.
        let mut host_logits: Vec<f32> = Vec::new();
        let mut scaled: Vec<(u32, f32)> = Vec::new();
        // Both sampling paths reuse this single softmax-output buffer
        // (clear+extend instead of `.collect()`). The fast top_p>=1
        // path treats it as exp-of-shifted-logits; the slow path
        // first writes exp values, then in-place re-normalises them
        // to a probability vector — same shape either way.
        let mut probs: Vec<f32> = Vec::new();
        // Round-23 finding #3: read once per request, not once per
        // decode token. The env var doesn't change mid-generation; the
        // syscall + alloc on every sampled token is pure overhead.
        let top_p_candidate_cap: usize = std::env::var("RVLLM_TOP_P_CANDIDATE_CAP")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
        let mut host_sample_token = |logits_dev_ptr: u64,
                                 vocab: u32,
                                 temp: f32,
                                 top_p: f32,
                                 top_k: Option<u32>,
                                 rng_f32: &mut dyn FnMut() -> f32|
            -> Result<u32> {
            #[cfg(feature = "cuda")]
            unsafe {
                host_logits.clear();
                host_logits.resize(vocab as usize, 0.0);
                // Check the cuMemcpyDtoH return — a silently-failing
                // copy left `host_logits` all-zeros, every entry tied,
                // and lex-tiebreak picked token-id 0 deterministically.
                // The caller saw "successful" generation of <pad>/<bos>
                // sequences while a real CUDA fault sat masked under
                // it. Now we propagate the error and the request ends
                // with a clear `RvllmError::Cuda { kind: MemcpyFailed,
                // op: "host_sample_token_dtoh", … }`.
                let r = cudarc::driver::sys::cuMemcpyDtoH_v2(
                    host_logits.as_mut_ptr() as *mut _,
                    logits_dev_ptr,
                    (vocab as usize) * 4,
                );
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::Cuda {
                        kind: rvllm_core::CudaErrorKind::MemcpyFailed,
                        op: "host_sample_token_dtoh",
                        ctx: rvllm_core::CudaCtx {
                            stream: stream as u64,
                            kernel: "",
                            launch: None,
                            device: 0,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    });
                }
                // Apply temperature: divide by temp (clamped to >=1e-3 to
                // avoid div-by-zero on sloppy callers).
                let inv_temp = 1.0f32 / temp.max(1e-3);
                scaled.clear();
                scaled.extend(host_logits.iter().enumerate()
                    .map(|(i, &v)| (i as u32, v * inv_temp)));
                let cmp_desc = |a: &(u32, f32), b: &(u32, f32)| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        // Lex-tiebreak on token id so a caller-supplied
                        // `seed` fully reproduces a request.
                        .then(a.0.cmp(&b.0))
                };
                // Fast path: no top-k AND top_p covers the whole
                // distribution → no truncation, the only reason to
                // sort would be ordered iteration during the cumulative-
                // cutoff loop, which is a no-op here. Skip the
                // O(V log V) sort entirely and do a direct multinomial
                // over the un-sorted vocab. The bulk of "default"
                // OpenAI requests (temperature=1.0, top_p=1.0,
                // top_k=None) lands here, which previously paid the
                // full sort per token over a 262 k-element vocab on
                // every decode step.
                if top_k.is_none() && top_p >= 1.0 {
                    let max_l = scaled.iter().map(|&(_, l)| l)
                        .fold(f32::NEG_INFINITY, f32::max);
                    probs.clear();
                    probs.extend(scaled.iter().map(|&(_, l)| (l - max_l).exp()));
                    let z: f32 = probs.iter().sum();
                    if z > 0.0 {
                        let r = rng_f32() * z;
                        let mut acc = 0.0f32;
                        for (i, &e) in probs.iter().enumerate() {
                            acc += e;
                            if r < acc { return Ok(scaled[i].0); }
                        }
                    }
                    // Either z == 0 (all-NaN logits, treat as the
                    // last token) or floating drift past the cumsum
                    // — return the highest-id token deterministically.
                    return Ok(scaled.last().map(|x| x.0).unwrap_or(0));
                }
                // top_k: if set, drop everything past rank K before the
                // softmax. With K typically 40-64, a full O(V log V)
                // sort over a 262 k-vocab is wasteful — partial-select
                // the top-K with `select_nth_unstable_by` (O(V) average)
                // and only sort the K-sized prefix. Falls back to a
                // full sort when top_k isn't set.
                // Round-22 finding #3 / round-23 hoist: when top_p<1
                // and no top_k is set, we used to do a full
                // O(V log V) sort over the entire 262 k-vocab on every
                // decode token. Realistic top_p values (≤0.99) almost
                // never keep more than a few hundred tokens, so cap
                // candidates internally and use partial-select. The
                // cap (`RVLLM_TOP_P_CANDIDATE_CAP`, default 2048) is
                // resolved once per request above the closure to keep
                // the env-var read off the per-token hot path.
                let effective_len = match top_k {
                    Some(k) => {
                        let k = (k as usize).min(scaled.len()).max(1);
                        if k < scaled.len() {
                            // Partition: first k are "the K largest" but
                            // unsorted; sort that small prefix only.
                            scaled.select_nth_unstable_by(k - 1, cmp_desc);
                            scaled[..k].sort_by(cmp_desc);
                        } else {
                            scaled.sort_by(cmp_desc);
                        }
                        k
                    }
                    None => {
                        let cap = top_p_candidate_cap.min(scaled.len()).max(1);
                        if cap < scaled.len() {
                            scaled.select_nth_unstable_by(cap - 1, cmp_desc);
                            scaled[..cap].sort_by(cmp_desc);
                            cap
                        } else {
                            scaled.sort_by(cmp_desc);
                            scaled.len()
                        }
                    }
                };
                // Softmax-stabilise on the kept slice (avoids overflow
                // on huge logits — Gemma 4 sees 60+).
                let max_l = scaled[0].1;
                probs.clear();
                probs.extend(scaled[..effective_len].iter()
                    .map(|&(_, l)| (l - max_l).exp()));
                let z: f32 = probs.iter().sum();
                if z > 0.0 { for p in &mut probs { *p /= z; } }
                // top_p: keep tokens until cumulative >= top_p, drop rest.
                let mut cum = 0.0f32;
                let mut cutoff = probs.len();
                for (i, &p) in probs.iter().enumerate() {
                    cum += p;
                    if cum >= top_p { cutoff = i + 1; break; }
                }
                // Re-normalize over the top_p window
                let z2: f32 = probs[..cutoff].iter().sum();
                if z2 > 0.0 { for p in &mut probs[..cutoff] { *p /= z2; } }
                // Sample
                let r = rng_f32();
                let mut acc = 0.0f32;
                for (i, &p) in probs[..cutoff].iter().enumerate() {
                    acc += p;
                    if r < acc { return Ok(scaled[i].0); }
                }
                Ok(scaled[cutoff - 1].0)
            }
            #[cfg(not(feature = "cuda"))]
            { let _ = (logits_dev_ptr, vocab, temp, top_p, top_k, rng_f32); Ok(0u32) }
        };
        // === END HOST-SIDE TEMPERATURE SAMPLING ===

        // === REPETITION PENALTY ===
        // Greedy-compatible logits processor. When set, divides the
        // logit of any token in `recent_ids` by `penalty` (positive
        // logits get smaller, negative logits get pushed further
        // negative — standard HF transformers convention).
        //
        // RVLLM_REPETITION_PENALTY        f32, default 1.0 (= disabled)
        // RVLLM_REPETITION_PENALTY_WINDOW usize, default 64 (look-back
        //                                  window size in decoded tokens)
        //
        // Applied between `f16_gemm_f32` and `ArgmaxLaunch` so argmax
        // picks from penalized logits. Defends against single-token
        // attractor locks (e.g. "la la la" on pure NVFP4 tool-call
        // collapse) that the basic repetition guard only catches
        // post-hoc after 20 identical tokens.
        //
        // Cost: one DtoH + one HtoD of the full vocab f32 logits per
        // decode step (~2.1 MiB total round-trip for Gemma 4 vocab=
        // 263168). ~7 ms/step on GB10 unified memory; for a 1024-token
        // decode that's ~7 s extra wall time. Acceptable diagnostic
        // overhead; if needed in production, replace with a tiny CUDA
        // kernel.
        let rep_penalty: f32 = std::env::var("RVLLM_REPETITION_PENALTY")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let rep_window: usize = std::env::var("RVLLM_REPETITION_PENALTY_WINDOW")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        // Frequency gate. Default 1 = penalize on first occurrence
        // (HuggingFace transformers behavior). Raise to 2 or 3 to
        // ONLY penalize tokens that appeared multiple times — keeps
        // common German function words / subwords intact while still
        // breaking lock attractors. Per GPT-5.5 review of pure-NVFP4
        // multilingual leakage at moderate margins (e.g. step 86 of
        // R2 emitting Italian " facendo"), the un-gated penalty is
        // too blunt for greedy decode; min_count=2 + penalty=1.05
        // is recommended starting point.
        let rep_min_count: u32 = std::env::var("RVLLM_REPETITION_PENALTY_MIN_COUNT")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        let rep_active = rep_penalty > 1.0 + f32::EPSILON;
        // === END REPETITION PENALTY ===

        let block_size: u32 = 32;
        let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(1024);

        let arena = &self.arena;
        let max_hd = arch.max_head_dim() as u32;
        let max_nkvh = arch.max_kv_heads() as u32;
        let max_q_dim = (arch.num_attention_heads * arch.max_head_dim()) as u32;
        let max_kv_dim = (max_nkvh * max_hd) as u32;
        let max_qkv_rows = max_q_dim + 2 * max_kv_dim;
        let inter = arch.intermediate_size as u32;
        let max_blocks_per_seq = num_blocks_total;

        let prompt_len = prompt_ids.len() as u32;
        let max_tokens = prompt_len.max(1);

        let hidden_fp8 = arena.region("gen_hidden_fp8", (max_tokens * hidden) as usize, 16)?;
        let hidden_scale = arena.region("gen_hidden_scale", (max_tokens * 4) as usize, 16)?;
        let qkv_out = arena.region("gen_qkv", (max_tokens * max_qkv_rows * 2) as usize, 16)?;
        let q_base = qkv_out.device_ptr();
        let q_normed = arena.region("gen_q_normed", (max_tokens * max_q_dim * 2) as usize, 16)?;
        let k_normed = arena.region("gen_k_normed", (max_tokens * max_kv_dim * 2) as usize, 16)?;
        let v_normed = arena.region("gen_v_normed", (max_tokens * max_kv_dim * 2) as usize, 16)?;
        let q_fp8 = arena.region("gen_q_fp8", (max_tokens * max_q_dim) as usize, 16)?;
        let attn_out = arena.region("gen_attn_out", (max_tokens * max_q_dim * 2) as usize, 16)?;
        let attn_out_fp8 = arena.region("gen_attn_out_fp8", (max_tokens * max_q_dim) as usize, 16)?;
        let attn_out_scale = arena.region("gen_attn_out_scale", (max_tokens * 4) as usize, 16)?;
        let gate_up_out = arena.region("gen_gate_up", (max_tokens * 2 * inter * 2) as usize, 16)?;
        let gate_up_fp8 = arena.region("gen_gate_up_fp8", (max_tokens * 2 * inter) as usize, 16)?;
        let gate_up_scale = arena.region("gen_gate_up_scale", (max_tokens * 4) as usize, 16)?;
        let mlp_out_fp8 = arena.region("gen_mlp_fp8", (max_tokens * inter) as usize, 16)?;
        let mlp_out_scale = arena.region("gen_mlp_scale", (max_tokens * 4) as usize, 16)?;
        let delta_f16 = arena.region("gen_delta", (max_tokens * hidden * 2) as usize, 16)?;
        let gemm_f32_max_n = std::cmp::max(max_qkv_rows, 2 * inter);
        let gemm_f32_tmp = arena.region("gen_gemm_f32", (max_tokens * gemm_f32_max_n * 4) as usize, 16)?;

        // Prefix cache: use the persistent KV region allocated by
        // `init_prefix_cache`. If the cache wasn't pre-initialised
        // (old callers, rvllm-bench / probe), fall back to per-call
        // arena allocation — no cache hit available in that case.
        let sliding_blocks = num_blocks_total;
        let kv_dtype = crate::gemma4_layer_exec::KvDtype::from_env(false);
        // Codex review priority 1 (commit 49 — Phase B-2):
        // verify_batched_from_state arms this AtomicU32 with the
        // session's committed_len. When non-sentinel, the prefix-
        // cache match (token-id loop + chunk_size cap) is bypassed:
        // we use the forced value directly as common_prefix_len.
        // This eliminates the "RVLLM_PREFILL_CHUNK_SIZE=2048 forces
        // full re-prefill on short prompts" failure mode that was
        // killing wall-clock for spec-decode.
        let forced_prefix_override = self
            .force_common_prefix_override
            .swap(u32::MAX, std::sync::atomic::Ordering::AcqRel);
        let (kv_cache_ptr, kv_scale_ptr, kv_layer_offsets, kv_scale_layer_offsets,
             kv_total_bytes, kv_scale_total_bytes, common_prefix_len_raw) = {
            let guard = self.prefix_cache.lock().unwrap();
            match &*guard {
                Some(pc) => {
                    let mut prefix = 0usize;
                    if forced_prefix_override != u32::MAX {
                        // Spec-decode override path: caller has its
                        // own committed-len bookkeeping
                        // (SpecDecodeSession); honor it directly,
                        // skip the token-match loop AND skip the
                        // chunk_size cap below. The slots up to
                        // `forced_prefix_override` are guaranteed
                        // valid by the spec session contract.
                        prefix = (forced_prefix_override as usize)
                            .min(prompt_ids.len());
                    } else {
                    // Cache hit path: compute the longest common
                    // prefix in raw token ids.
                    while prefix < pc.last_tokens.len()
                        && prefix < prompt_ids.len()
                        && pc.last_tokens[prefix] == prompt_ids[prefix]
                    {
                        prefix += 1;
                    }
                    // Provenance check: invalidate cache entirely
                    // when batch shape / dtype / hybrid / scale
                    // policy / prefill mode differs from when the
                    // KV was written. Optimized NVFP4 kernels are
                    // batch-variant; reusing KV across mismatched
                    // policies produces silent miscompare.
                    let cur_prov = PrefixProvenance::from_env();
                    if cur_prov != pc.provenance {
                        eprintln!(
                            "[prefix-cache] provenance mismatch — invalidating \
                             (was {:?}, now {:?})",
                            pc.provenance, cur_prov
                        );
                        prefix = 0;
                        // Codex20-2: layout-relevant flags that affect
                        // per-layer KV byte-stride (HYBRID_GLOBAL_FP8,
                        // HYBRID_SLIDING_FP8, FP8_KV_LAYERS) should never
                        // change at runtime — env is read once at startup
                        // and the persistent KV allocation +
                        // kv_layer_offsets are sized for that snapshot.
                        // We continue to reuse pc.kv_layer_offsets here
                        // (computed at first run_generate); a runtime flip
                        // of those flags would write the new layout into
                        // offsets sized for the old layout. Out of scope:
                        // rvllm-serve has no env-reload mechanism, the
                        // service is restarted between configurations
                        // (see kv_policy_matrix.sh), so this path is not
                        // reachable in production. If env-reload ever
                        // lands, this branch must also re-allocate
                        // pc.kv_cache_ptr / pc.kv_scale_ptr or refuse
                        // the request.
                    } else {
                        // Chunk-shape cap: only reuse up to the last
                        // FULLY-WRITTEN chunk boundary of the
                        // previous request. Slots written by a
                        // shorter trailing chunk are unsafe to reuse
                        // because they were quantized under a
                        // different batch shape. Fixes "la la la"
                        // garbage on classifier-then-persona chains.
                        let cap = pc.committed_prefix_len as usize;
                        if cap < prefix {
                            eprintln!(
                                "[prefix-cache] capping reuse {}→{} \
                                 (last committed chunk boundary)",
                                prefix, cap
                            );
                            prefix = cap;
                        }
                    }
                    } // end else (non-forced-override branch)
                    // Leave at least one token for the prefill to
                    // process (otherwise there's nothing to decode
                    // the last hidden state from).
                    if prefix >= prompt_ids.len() {
                        prefix = prompt_ids.len().saturating_sub(1);
                    }
                    (
                        pc.kv_cache_ptr,
                        pc.kv_scale_ptr,
                        pc.kv_layer_offsets.clone(),
                        pc.kv_scale_layer_offsets.clone(),
                        pc.kv_cache_bytes,
                        pc.kv_scale_bytes,
                        prefix as u32,
                    )
                }
                None => (0, 0, Vec::new(), Vec::new(), 0, 0, 0),
            }
        };

        let use_prefix_cache = kv_cache_ptr != 0;
        // Per-call fallback when cache wasn't initialised.
        let _kv_cache_region;
        let _kv_scale_region;
        // Cycle 56 step 1: `_kv_scale_total_bytes` prefix silences
        // the unused-variable warning — the value is computed in the
        // else branch (and used there for the local memset), but the
        // outer destructure binding is never read further.
        let (kv_cache_ptr, kv_scale_ptr,
             kv_layer_offsets, kv_scale_layer_offsets,
             kv_total_bytes, _kv_scale_total_bytes) = if use_prefix_cache {
            (kv_cache_ptr, kv_scale_ptr,
             kv_layer_offsets, kv_scale_layer_offsets,
             kv_total_bytes, kv_scale_total_bytes)
        } else {
            let mut kv_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
            let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
            let mut kv_total_bytes: u64 = 0;
            let mut kv_scale_total_bytes: u64 = 0;
            for l in 0..arch.num_hidden_layers {
                kv_layer_offsets.push(kv_total_bytes);
                kv_scale_layer_offsets.push(kv_scale_total_bytes);
                let is_global = arch.layer_types[l]
                    == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                let nkvh = arch.num_kv_heads_for_layer(l) as u32;
                let hd = arch.head_dim_for_layer(l) as u32;
                let layer_elems =
                    2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                let kv_dtype_l = crate::gemma4_layer_exec::KvDtype::for_layer_index_or_env(
                    arch.layer_types[l], l, false);
                kv_total_bytes += match kv_dtype_l {
                    crate::gemma4_layer_exec::KvDtype::F16 => layer_elems * 2,
                    crate::gemma4_layer_exec::KvDtype::Fp8 => layer_elems,
                    crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 2,
                };
                let layer_scale_slots =
                    2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64;
                kv_scale_total_bytes += match kv_dtype_l {
                    crate::gemma4_layer_exec::KvDtype::F16 => 0,
                    crate::gemma4_layer_exec::KvDtype::Fp8 => layer_scale_slots * 4,
                    crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 16,
                };
            }
            let kvr = arena.region("gen_kv", kv_total_bytes as usize, 256)?;
            // Cycle 56 step 4 (bug-audit #6): check CUresult on
            // per-request scratch zeroing — failure here would leave
            // KV cache + scale region with arbitrary bytes from the
            // previous request and corrupt this request's decode.
            cuda_check!(
                cudarc::driver::sys::cuMemsetD8_v2(kvr.device_ptr(), 0, kv_total_bytes as usize),
                "run_generate_kv_zero", 0u64);
            let kvs = arena.region(
                "gen_kv_scale_cache", kv_scale_total_bytes.max(16) as usize, 16)?;
            cuda_check!(
                cudarc::driver::sys::cuMemsetD8_v2(
                    kvs.device_ptr(), 0, kv_scale_total_bytes as usize),
                "run_generate_kv_scale_zero", 0u64);
            let kc_ptr = kvr.device_ptr();
            let ks_ptr = kvs.device_ptr();
            _kv_cache_region = kvr;
            _kv_scale_region = kvs;
            (kc_ptr, ks_ptr, kv_layer_offsets, kv_scale_layer_offsets,
             kv_total_bytes, kv_scale_total_bytes)
        };

        // Clamp the prefix-match to the actual KV region size.
        let mut common_prefix_len: u32 = if use_prefix_cache {
            common_prefix_len_raw
        } else {
            0
        };
        // Vision-splice cache-correctness gate (Codex review #2 round 5).
        // The prefix cache only hashes prompt_ids, NOT the vision items
        // / image bytes / spliced embeddings. Two requests with the
        // same chat shape but different images produce identical
        // image-pad token sequences and would silently reuse the OLD
        // image's KV. Force a full prefill whenever the request brings
        // vision data so the new embeddings actually get spliced.
        if (!vision_splice.is_empty() || !audio_splice.is_empty())
            && common_prefix_len > 0
        {
            eprintln!(
                "[prefix-cache] bypassed: request has {} vision + {} audio splice slot(s); \
                 forcing common_prefix_len=0 to avoid stale-modality KV reuse",
                vision_splice.len(), audio_splice.len()
            );
            common_prefix_len = 0;
        }
        if common_prefix_len > 0 {
            eprintln!(
                "[prefix-cache] hit: reusing {} of {} prompt tokens",
                common_prefix_len, prompt_len
            );
        }
        // Wrap the raw pointers so downstream `.device_ptr()` calls
        // stay source-identical across the cache-hit and fallback
        // paths. This is a 16-byte local, zero runtime cost.
        struct KvHandle(u64);
        impl KvHandle {
            fn device_ptr(&self) -> u64 { self.0 }
        }
        let kv_cache = KvHandle(kv_cache_ptr);
        let kv_scale_cache = KvHandle(kv_scale_ptr);
        let q_scale_scratch_bytes =
            (max_tokens as u64) * (arch.num_attention_heads as u64) * 4;
        let q_scale_scratch = arena.region(
            "gen_q_scale_scratch", q_scale_scratch_bytes as usize, 16)?;
        cuda_check!(
            cudarc::driver::sys::cuMemsetD8_v2(
                q_scale_scratch.device_ptr(), 0, q_scale_scratch_bytes as usize),
            "run_generate_q_scale_scratch_zero", 0u64);
        // See run_bench: RVLLM_PER_TOKEN_Q_SCALE=0 opts out.
        let q_scale_cache_ptr: u64 =
            if per_token_q_scale_enabled(/*default_on=*/true) {
                q_scale_scratch.device_ptr()
            } else {
                0
            };

        let q_scale_region = arena.region("gen_q_scale", 4, 4)?;
        let kv_scale_region = arena.region("gen_kv_scale", 4, 4)?;
        {
            let q_s = parse_f32_env_or_default("RVLLM_Q_SCALE", DEFAULT_Q_SCALE);
            let kv_s = parse_f32_env_or_default("RVLLM_KV_SCALE", DEFAULT_KV_SCALE);
            q_scale_region.copy_from_host(&q_s.to_le_bytes())?;
            kv_scale_region.copy_from_host(&kv_s.to_le_bytes())?;
        }

        const FA3_WS_BYTES: usize = 128 * 1024 * 1024;
        let fa3_ws = arena.region("gen_fa3_ws", FA3_WS_BYTES, 256)?;
        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena.region("gen_cutlass_ws", cutlass_ws_bytes, 256)?;

        let positions = arena.region("gen_pos", (max_tokens * 4) as usize, 16)?;
        let slot_mapping = arena.region("gen_slot", (max_tokens * 4) as usize, 16)?;
        let context_lens = arena.region("gen_ctx", 4, 16)?;
        // Sized for max_tokens i32 entries (not just the 2-entry prefix
        // sum): the unified decode-per-qi attention loop reuses this
        // region to stage a per-qi context-lens array `[1, 2, ..., N]`
        // and indexes it by `qi * 4`. With only 8 bytes (old FA2
        // prefill layout) writing beyond entry 1 corrupted adjacent
        // arena regions and degenerated generation quality.
        // Codex26-3: prefix-sum holds at least `[0, chunk_q]` = 2 i32
        // entries even when max_tokens=1 (RVLLM_DIAG_SKIP_DECODE +
        // single-token prompt). The previous `max_tokens*4` produced
        // a 4-byte region, so the 8-byte copy_from_host below failed.
        let cu_seqlens_q = arena.region(
            "gen_cu_seqlens",
            ((max_tokens.max(1) + 1) * 4) as usize,
            16,
        )?;
        let block_tables = arena.region("gen_bt", (max_blocks_per_seq * 4) as usize, 16)?;
        {
            let bt: Vec<i32> = (0..max_blocks_per_seq as i32).collect();
            block_tables.copy_from_host(bytemuck_cast_i32(&bt))?;
        }

        let residual = arena.region("gen_residual", (max_tokens * hidden * 2) as usize, 16)?;
        let logits_f32 = arena.region("gen_logits_f32", (vocab * 4) as usize, 16)?;
        // Codex41-3: device-side scratch for the recent-IDs list the
        // GPU repetition-penalty kernel reads. 1024 i32 = 4 KB —
        // covers any reasonable rep-window without bouncing the full
        // vocab through host memory.
        let rep_ids_dev = arena.region("gen_rep_ids", 1024 * 4, 16)?;
        let token_ids_region = arena.region("gen_tok_ids", (max_tokens * 4) as usize, 16)?;
        let sampled = arena.region("gen_sampled", 4, 16)?;
        let residual_ptr = residual.device_ptr();
        let kernels = self.layer_kernels()?;

        use rvllm_loader::gemma4_arch::Gemma4LayerType;
        let max_layers = std::env::var("RVLLM_MAX_LAYERS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(arch.num_hidden_layers);

        // === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
        // Build (or re-use) the f16 shadow KV region for the instrumented
        // layers. Pure diagnostic — the allocation mirrors the primary
        // allocator but forces F16 for every instrumented layer so the
        // cache is ground-truth. Same layer-blocks / nkvh / hd sizing as
        // the primary path; no scale region (f16 self-scaled).
        // Per-request gate: even with RVLLM_NVFP4_SHADOW_F16=1 in env,
        // skip shadow path unless the operator explicitly opted THIS
        // request in via `X-Rvllm-Shadow: 1`. Closes the
        // upstream-client-scaffold-burns-latch hole.
        let shadow_set: Option<Vec<u32>> = if shadow_requested {
            crate::gemma4_layer_exec::parse_shadow_layers()
        } else {
            None
        };
        // Compute per-layer shadow offsets (populated only for layers
        // in the shadow set; sentinel u64::MAX otherwise). Needed both
        // when we have to allocate the region now AND on subsequent
        // calls when it already exists — cheap to recompute every call.
        let shadow_layer_offsets: Vec<u64> = if let Some(ref lset) = shadow_set {
            let mut offs = vec![u64::MAX; arch.num_hidden_layers];
            let mut cursor: u64 = 0;
            for l in 0..arch.num_hidden_layers {
                if !lset.contains(&(l as u32)) { continue; }
                offs[l] = cursor;
                let is_global = arch.layer_types[l]
                    == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                let nkvh = arch.num_kv_heads_for_layer(l) as u32;
                let hd = arch.head_dim_for_layer(l) as u32;
                // f16 = 2 bytes/elem; 2× for K and V.
                let layer_bytes =
                    2u64 * (layer_blocks as u64) * (block_size as u64)
                        * (nkvh as u64) * (hd as u64) * 2;
                cursor += layer_bytes;
            }
            offs
        } else {
            Vec::new()
        };
        let shadow_total_bytes: u64 = if let Some(ref lset) = shadow_set {
            let mut sum: u64 = 0;
            for l in 0..arch.num_hidden_layers {
                if !lset.contains(&(l as u32)) { continue; }
                let is_global = arch.layer_types[l]
                    == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                let nkvh = arch.num_kv_heads_for_layer(l) as u32;
                let hd = arch.head_dim_for_layer(l) as u32;
                sum += 2u64 * (layer_blocks as u64) * (block_size as u64)
                    * (nkvh as u64) * (hd as u64) * 2;
            }
            sum
        } else { 0 };
        // Per-layer Q slot size: num_attention_heads * max_head_dim * 2 (f16).
        // Uniform across layers so indexing is a simple multiply. We dump
        // exactly ONE Q row per layer (decode step 0's Q). Slot is fixed
        // at single-token size regardless of how many tokens prefill
        // wrote — the per-layer slot is only ever populated on decode
        // step 0, when num_tokens == 1.
        let shadow_q_per_layer_bytes: u64 = if shadow_set.is_some() {
            2u64 * (arch.num_attention_heads as u64) * (arch.max_head_dim() as u64)
        } else {
            0
        };
        // Throwaway scratch — must hold the largest single
        // rope_f16kv_shadow Q-output. Decode = 1 token; batch prefill
        // = up to chunk_q tokens (bounded by num_blocks_total *
        // block_size). Size for the prefill upper bound so batch
        // prefill scratch construction can route shadow Q here
        // safely. ~1 GiB worst case on Gemma 4 31B (32768 * 32 *
        // 512 * 2). Trivial vs 128 GiB unified.
        let shadow_q_throwaway_bytes: u64 = if shadow_set.is_some() {
            (num_blocks_total as u64) * (block_size as u64) * shadow_q_per_layer_bytes
        } else {
            0
        };
        let shadow_q_total_bytes: u64 = if let Some(ref lset) = shadow_set {
            shadow_q_per_layer_bytes * (lset.len() as u64)
        } else {
            0
        };
        let (shadow_ptr, shadow_q_ptr, shadow_q_throwaway_ptr): (u64, u64, u64) =
            if let Some(ref lset) = shadow_set {
            let mut guard = self.nvfp4_shadow.lock().unwrap();
            if let Some(ref existing) = *guard {
                (existing.shadow_ptr, existing.shadow_q_ptr, existing.shadow_q_throwaway_ptr)
            } else {
                // `Region` isn't `Drop`; the bump-pointer state held
                // by `arena` is what keeps these allocations alive
                // past the wrapper falling out of scope below.
                let bytes = shadow_total_bytes.max(16) as usize;
                let region = arena.region("nvfp4_shadow_kv", bytes, 256)?;
                // Codex26-4: NVFP4 shadow diagnostic memsets — wrap with
                // cuda_check so OOM / ECC during shadow init surfaces
                // as a typed error instead of producing misleading
                // shadow snapshots.
                cuda_check!(cudarc::driver::sys::cuMemsetD8_v2(
                    region.device_ptr(), 0, bytes),
                    "nvfp4_shadow_kv_zero", stream);
                let ptr = region.device_ptr();
                // Per-layer Q snapshot region.
                let q_bytes = shadow_q_total_bytes.max(16) as usize;
                let q_region = arena.region("nvfp4_shadow_q", q_bytes, 256)?;
                cuda_check!(cudarc::driver::sys::cuMemsetD8_v2(
                    q_region.device_ptr(), 0, q_bytes),
                    "nvfp4_shadow_q_zero", stream);
                let q_ptr = q_region.device_ptr();
                // Q throwaway scratch: one slot. Shadow rope targets
                // this when we're not capturing, so q_normed stays
                // untouched and the subsequent primary nvfp4 rope
                // gets pristine pre-RoPE Q as input (exactly-one-RoPE
                // invariant restored).
                let throwaway_bytes = shadow_q_throwaway_bytes.max(16) as usize;
                let throwaway_region = arena.region(
                    "nvfp4_shadow_q_throwaway", throwaway_bytes, 256)?;
                cuda_check!(cudarc::driver::sys::cuMemsetD8_v2(
                    throwaway_region.device_ptr(), 0, throwaway_bytes),
                    "nvfp4_shadow_q_throwaway_zero", stream);
                let throwaway_ptr = throwaway_region.device_ptr();
                *guard = Some(NvFp4ShadowAlloc {
                    shadow_ptr: ptr,
                    shadow_bytes: shadow_total_bytes,
                    layer_offsets: shadow_layer_offsets.clone(),
                    layer_indices: lset.clone(),
                    shadow_q_ptr: q_ptr,
                    shadow_q_total_bytes,
                    shadow_q_per_layer_bytes,
                    shadow_q_throwaway_ptr: throwaway_ptr,
                });
                eprintln!(
                    "[nvfp4-shadow] allocated {} MiB f16 shadow KV + {} KiB per-layer Q for {} layers: {:?}",
                    shadow_total_bytes / (1024 * 1024),
                    shadow_q_total_bytes / 1024,
                    lset.len(),
                    lset,
                );
                (ptr, q_ptr, throwaway_ptr)
            }
        } else { (0, 0, 0) };
        // === END NVFP4 SHADOW DIAGNOSTIC ===

        // === HADAMARD ROTATION ===
        // Lazy-init the per-layer Hadamard sign vectors on first run
        // (gated by env). All layers share the same head_dim at the
        // rope-input level (Gemma 4: sliding heads use head_dim=256,
        // global heads use head_dim=512 — but rotation buffer is
        // sized to `arch.max_head_dim()` so both fit; sliding layers
        // use the leading head_dim bytes of their slot).
        let hadamard_base_ptr: u64 = if nvfp4_hadamard_enabled() {
            let mut guard = self.nvfp4_hadamard.lock().unwrap();
            if guard.is_none() {
                let max_hd = arch.max_head_dim() as u32;
                let nl = arch.num_hidden_layers as u32;
                *guard = build_nvfp4_hadamard_signs(nl, max_hd, arena)?;
            }
            guard.as_ref().map(|a| a.base_ptr).unwrap_or(0)
        } else {
            0
        };
        let hadamard_head_dim_stride: u32 =
            if hadamard_base_ptr != 0 { arch.max_head_dim() as u32 } else { 0 };
        // === END HADAMARD ROTATION ===

        // Helper: run one token through all layers (decode path)
        // === CUDA-graph-capturable factoring (cycle 60+) ===
        //
        // Original `run_one_token` interleaved host→device input copies
        // (token id, positions, slot_mapping, context_lens) with the
        // pure-device kernel chain. CUDA Graph capture cannot replay
        // HtoD copies whose source pointers are stack-local — they go
        // out of scope after the capture closure returns. The PPL path
        // (line ~2099) already factors the same way; we mirror it here
        // for the decode hot path.
        //
        //   * `prepare_decode_inputs(tok_id, step)` — host-side HtoDs,
        //     ALWAYS runs eagerly between graph replays.
        //   * `decode_forward()` — pure device kernel chain (embedding
        //     gather + optional f16→bf16 widen + 60-layer body).
        //     Capturable into a CUDA graph; replayable any number of
        //     times as long as the device-side input buffers are
        //     refreshed via `prepare_decode_inputs` between replays.
        //
        // `run_one_token` keeps its existing signature for the prefill
        // (line ~3012) and diagnostic (line ~3044) call sites; it just
        // calls the two helpers in sequence.
        let prepare_decode_inputs = |tok_id: u32, step: usize| -> Result<()> {
            let tok_i32 = [tok_id as i32];
            token_ids_region.copy_from_host(bytemuck_cast_i32(&tok_i32))?;
            let pos = [step as i32];
            let slot = [step as i32];
            let ctx = [step as i32 + 1];
            positions.copy_from_host(bytemuck_cast_i32(&pos))?;
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot))?;
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx))?;
            Ok(())
        };
        let decode_forward = |step: usize| -> Result<()> {
            rvllm_fused::EmbeddingGatherLaunch { num_tokens: 1, hidden, vocab }
                .launch(fn_embed, residual_ptr, self.model.embedding.offset_bytes,
                    token_ids_region.device_ptr(), stream)?;

            // E4B PLE precompute for the decoded token. Mirrors the
            // chunked-prefill PLE precompute at line ~4338: every E4B
            // decoder layer reads a per-(token, layer) input slice and
            // injects it into the residual after the attention block.
            // Without this, the decode path diverges from prefill at
            // every layer (per-layer cosine drift starting at L0). PLE
            // depends on inputs_embeds, so it must run AFTER the
            // embedding gather above and BEFORE the optional bf16 widen
            // (precompute_ple expects f16 inputs_embeds).
            let ple_enabled = std::env::var("RVLLM_E4B_PLE")
                .map_or(false, |v| v == "1");
            let (ple_base, ple_stride_elems) = if ple_enabled {
                unsafe {
                    self.precompute_ple(
                        residual_ptr,
                        token_ids_region.device_ptr(),
                        1u32,
                        fn_embed,
                        &kernels,
                        stream as u64,
                    )?
                }
            } else {
                (0u64, 0u32)
            };

            // Cycle 54 Stage 1: BF16 residual chain entry. Embedding
            // gather writes f16; widen to bf16 in-place so subsequent
            // layers operate on bf16 storage.
            if bf16_residual_enabled() {
                rvllm_fused::gemma4_launcher::F16ToBf16Launch { n: hidden }
                    .launch(kernels.f16_to_bf16, residual_ptr, residual_ptr, stream)?;
            }

            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                if layer_idx >= max_layers { break; }
                let lt = arch.layer_types[layer_idx];
                let hd = arch.head_dim_for_layer(layer_idx) as u32;
                let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                let q_dim = (arch.num_attention_heads as u32) * hd;
                let kv_dim = nkvh * hd;
                let layer_blocks = if lt == Gemma4LayerType::GlobalAttention { num_blocks_total } else { sliding_blocks };
                let layer_kv_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                // E4B kv-share: when this layer aliases an earlier source
                // (Gemma 4 num_kv_shared_layers tail), the attention
                // launchers must read K/V from the SOURCE layer's region.
                // Pointing layer_kv_base at the source while passing
                // dims.kv_share_source_layer=Some(_) suppresses rope's
                // K/V writes (see fused_rope_partial_*kv.cu nullptr guard)
                // so the source layer's K/V cache is never clobbered.
                let kv_idx = arch.kv_share_source_layer(layer_idx).unwrap_or(layer_idx);
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[kv_idx];
                // kv-share-aware (mirrors layer_kv_base above): for shared
                // layers, the scale arena base also maps to the source layer.
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[kv_idx];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);
                // Per-layer dtype: hybrid mode swaps global layers to FP8,
                // sliding layers stay on env default. Cycle 24: pass
                // layer_idx so RVLLM_FP8_KV_LAYERS list env is honored
                // (must match the allocation-side decision in load_gemma4_fused
                // — layer 709 — or the cache layout disagrees with dispatch).
                let kv_dtype = crate::gemma4_layer_exec::KvDtype::for_layer_index_or_env(lt, layer_idx, false);
                let (k_cache_scale, v_cache_scale) = if kv_dtype
                    == crate::gemma4_layer_exec::KvDtype::Nvfp4
                {
                    (layer_kv_scale_base, layer_kv_scale_base + layer_kv_elems / 32)
                } else {
                    (0u64, 0u64)
                };

                let dims = crate::gemma4_layer_exec::Gemma4LayerDims {
                    num_tokens: 1, hidden,
                    num_heads: arch.num_attention_heads as u32, num_kv_heads: nkvh, head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    intermediate: inter,
                    ple_dim: arch.hidden_size_per_layer_input.unwrap_or(0) as u32,
                    block_size,
                    max_blocks_per_seq: layer_blocks, num_blocks_total: layer_blocks,
                    attn_scale: 1.0, rms_eps: arch.rms_norm_eps,
                    layer_type: lt, sliding_window: arch.sliding_window_size as u32,
                    f16_kv: kv_dtype.is_f16(),
                    kv_dtype,
                    bf16_residual: bf16_residual_enabled(),
                    kv_share_source_layer: arch.kv_share_source_layer(layer_idx).map(|s| s as u32),
                    // Decode step knows its own ctx CPU-side — `ctx = [step + 1]`
                    // was computed at line ~1843. Pass it so the split-KV
                    // dispatch gates on the current ctx length instead of
                    // the bucket max (avoids dispatching split on short
                    // early-generation turns where one-CTA decode wins).
                    current_max_context_len: Some((step as u32) + 1),
                };
                let w = crate::gemma4_layer_exec::Gemma4LayerWeightPtrs {
                    attn_norm_gamma: layer.input_layernorm.offset_bytes,
                    post_attn_norm_gamma: layer.post_attention_layernorm.offset_bytes,
                    pre_ff_norm_gamma: layer.pre_feedforward_layernorm.offset_bytes,
                    post_ff_norm_gamma: layer.post_feedforward_layernorm.offset_bytes,
                    q_norm_gamma: layer.q_norm.offset_bytes,
                    k_norm_gamma: layer.k_norm.offset_bytes,
                    qkv_fp8: layer.qkv.as_ref().map_or(0, |w| w.offset_bytes), qkv_scale: layer.qkv.as_ref().map_or(0, |w| w.scale_ptr),
                    o_fp8: layer.o_proj.as_ref().map_or(0, |w| w.offset_bytes), o_scale: layer.o_proj.as_ref().map_or(0, |w| w.scale_ptr),
                    gate_up_fp8: layer.gate_up.as_ref().map_or(0, |w| w.offset_bytes), gate_up_scale: layer.gate_up.as_ref().map_or(0, |w| w.scale_ptr),
                    down_fp8: layer.down_proj.as_ref().map_or(0, |w| w.offset_bytes), down_scale: layer.down_proj.as_ref().map_or(0, |w| w.scale_ptr),
                    layer_scalar_ptr: layer.layer_scalar.offset_bytes,
                    qkv_f16: layer.qkv_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    o_f16: layer.o_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    gate_up_f16: layer.gate_up_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    down_f16: layer.down_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    qkv_chscale: layer.qkv.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    o_chscale: layer.o_proj.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    gate_up_chscale: layer.gate_up.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    down_chscale: layer.down_proj.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    qkv_blockscale: layer.qkv.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    o_blockscale: layer.o_proj.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    gate_up_blockscale: layer.gate_up.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    down_blockscale: layer.down_proj.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    awq: awq_layer_ptrs(layer.awq.as_ref()),
                    // E4B PLE plumbing. Pointers from the loaded per-layer
                    // tensors (`None` → 0 on 31B/AWQ). `ple_per_layer_input`
                    // is populated by the per-request PLE precompute (Stage
                    // 3b); 0 here means PLE inactive for this forward
                    // dispatch — layer_exec then skips the PLE injection.
                    ple_input_gate: layer
                        .per_layer_input_gate
                        .as_ref()
                        .map_or(0, |w| w.offset_bytes),
                    ple_projection: layer
                        .per_layer_projection
                        .as_ref()
                        .map_or(0, |w| w.offset_bytes),
                    ple_post_input_norm_gamma: layer
                        .post_per_layer_input_norm
                        .as_ref()
                        .map_or(0, |w| w.offset_bytes),
                    // E4B PLE: per-layer slice into the precompute buffer
                    // computed once per decode step in decode_forward above.
                    // Layout matches the chunked-prefill site at ~line 4502:
                    // base + L * ple_dim * 2.
                    ple_per_layer_input: if ple_base != 0 {
                        ple_base + (layer_idx as u64)
                            * (arch.hidden_size_per_layer_input.unwrap_or(0) as u64)
                            * 2
                    } else {
                        0
                    },
                    ple_per_layer_stride_elems: ple_stride_elems,
                };
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let (cos, sin) = match lt {
                    Gemma4LayerType::SlidingAttention => (self.model.rope_cos_sliding.offset_bytes, self.model.rope_sin_sliding.offset_bytes),
                    Gemma4LayerType::GlobalAttention => (self.model.rope_cos_global.offset_bytes, self.model.rope_sin_global.offset_bytes),
                };
                let bytes_per_half_kv = match kv_dtype {
                    crate::gemma4_layer_exec::KvDtype::F16 => layer_kv_elems,
                    crate::gemma4_layer_exec::KvDtype::Fp8 => layer_kv_elems / 2,
                    crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_kv_elems / 4,
                };
                // === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
                // Populate shadow pointers only for instrumented NVFP4 layers.
                let is_shadow_layer = shadow_ptr != 0
                    && kv_dtype == crate::gemma4_layer_exec::KvDtype::Nvfp4
                    && layer_idx < shadow_layer_offsets.len()
                    && shadow_layer_offsets[layer_idx] != u64::MAX;
                let (shadow_k, shadow_v) = if is_shadow_layer {
                    let base = shadow_ptr + shadow_layer_offsets[layer_idx];
                    // f16 K/V: each is layer_kv_elems/2 elements × 2 bytes = layer_kv_elems bytes.
                    (base, base + layer_kv_elems)
                } else {
                    (0u64, 0u64)
                };
                // Shadow Q target. Two roles:
                //   (a) When the layer is instrumented AND this is the
                //       first decode step (step 0 after prompt), point
                //       at the per-layer Q slot so the Python analyzer
                //       gets post-RoPE f16 Q for logit_err / topk /
                //       out_err.
                //   (b) When the layer is instrumented at any OTHER
                //       step (all prefill steps + decode step >0),
                //       point at the shared throwaway so shadow rope
                //       has a valid q_out target WITHOUT clobbering
                //       `scratch.q_normed`. This is load-bearing: if
                //       q_normed is clobbered, the subsequent primary
                //       rope_nvfp4kv rotates it a second time and
                //       every forward pass is wrong.
                //   (c) When the layer is NOT instrumented, 0 — shadow
                //       rope doesn't run at all.
                let shadow_q = if is_shadow_layer {
                    if step == prompt_ids.len() && shadow_q_ptr != 0 {
                        let pos_in_set = shadow_set
                            .as_ref()
                            .and_then(|s| s.iter().position(|&l| l as usize == layer_idx));
                        match pos_in_set {
                            Some(i) => shadow_q_ptr + (i as u64) * shadow_q_per_layer_bytes,
                            None => shadow_q_throwaway_ptr,
                        }
                    } else {
                        shadow_q_throwaway_ptr
                    }
                } else {
                    0
                };
                // === END NVFP4 SHADOW DIAGNOSTIC ===
                // === HADAMARD ROTATION ===
                // Per-layer pointer into the sign-vector base buffer.
                // Only enabled on NVFP4 layers (other dtypes' rope
                // launchers don't read these fields). Both Q and K
                // share the same per-layer vector — see comment on
                // the field declaration.
                let hadamard_layer_ptr: u64 = if hadamard_base_ptr != 0
                    && kv_dtype == crate::gemma4_layer_exec::KvDtype::Nvfp4
                {
                    hadamard_base_ptr
                        + (layer_idx as u64) * (hadamard_head_dim_stride as u64)
                } else {
                    0
                };
                // === END HADAMARD ROTATION ===
                let scratch = crate::gemma4_layer_exec::Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(), hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base, k_out, v_out,
                    q_normed: q_normed.device_ptr(), k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + bytes_per_half_kv,
                    k_cache_scale,
                    v_cache_scale,
                    q_scale_ptr: q_scale_region.device_ptr(), kv_scale_ptr: kv_scale_region.device_ptr(),
                    k_scale_cache: layer_kv_scale_base,
                    v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                    q_scale_cache: q_scale_cache_ptr,
                    attn_out: attn_out.device_ptr(), attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(), delta_f16: delta_f16.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(), gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(), mlp_out_scale: mlp_out_scale.device_ptr(),
                    gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                    // Codex40-3: run_generate alloc uses `max_tokens`,
                    // not num_seqs.
                    gemm_f32_tmp_bytes: (max_tokens * gemm_f32_max_n * 4) as usize,
                    cutlass_workspace: cutlass_ws.device_ptr(), cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                    fa3_workspace_bytes: FA3_WS_BYTES as u64,
                    shadow_k_cache: shadow_k,
                    shadow_v_cache: shadow_v,
                    shadow_q_cache: shadow_q,
                    // === HADAMARD ROTATION ===
                    hadamard_signs_q: hadamard_layer_ptr,
                    hadamard_signs_k: hadamard_layer_ptr,
                    // === END HADAMARD ROTATION ===
                };
                let meta = crate::gemma4_layer_exec::Gemma4MetadataPtrs {
                    positions: positions.device_ptr(), slot_mapping: slot_mapping.device_ptr(),
                    cos, sin,
                    block_tables: block_tables.device_ptr(), context_lens: context_lens.device_ptr(),
                };
                crate::gemma4_layer_exec::gemma4_forward(
                    dims, &kernels, &w, &scratch, &meta,
                    &self.cublaslt, &self.cutlass, &self.sliding_attention, &self.global_attention,
                    residual_ptr, stream,
                )?;
                // E4B decode-step drill-down: when RVLLM_E4B_DECODE_DUMP_DIR
                // is set, dump the post-layer residual at this decode step
                // for every layer. Filenames carry the actual layer_idx +
                // step so prefill / decode / multiple steps don't collide.
                // Cost: D2H of `hidden * 2` bytes per layer, gated entirely
                // by env presence — zero overhead when unset.
                #[cfg(feature = "cuda")]
                if let Ok(dump_dir) = std::env::var("RVLLM_E4B_DECODE_DUMP_DIR") {
                    use cudarc::driver::sys::*;
                    cuStreamSynchronize(stream as CUstream);
                    let nbytes = (hidden as usize) * 2;
                    let mut buf = vec![0u8; nbytes];
                    cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut _, residual_ptr, nbytes);
                    let path = std::path::Path::new(&dump_dir)
                        .join(format!("e4b_decode_step{}_layer{}_output.bin", step, layer_idx));
                    std::fs::create_dir_all(&dump_dir).ok();
                    let _ = std::fs::write(&path, &buf);
                }
            }
            Ok(())
        };
        let run_one_token = |tok_id: u32, step: usize| -> Result<()> {
            prepare_decode_inputs(tok_id, step)?;
            decode_forward(step)
        };

        let t0 = std::time::Instant::now();

        // Phase 1: prompt through per-token decode (default, correct-by-design).
        //
        // On sm_121 with FP8 block-scale weights (Gemma 4 fp8-block), the
        // per-token path uses `fp8_gemv_blockwise_wpr_native_f16in_kernel`
        // which preserves the per-channel weight block-scale. The batch
        // (num_tokens>1) GEMM path goes through
        // `fp8_gemm_channelscale_or_fallback`, which on Blackwell consumer
        // collapses to a scalar weight scale because cuBLASLt's FP8
        // channelscale heuristic `LaunchFailed`s at this arch. That is a
        // genuine numerical difference, not a hidden bug — the two paths
        // are not bit-identical by design at num_tokens<CUTLASS_M_MIN(=128).
        //
        // Path forward for genuine batch-prefill speedup:
        //   * num_tokens >=128 : CUTLASS SM120 blockwise FP8 GEMM (landed;
        //     opt-in via RVLLM_FP8_GEMM_CUTLASS_SM120 + M>=128 gate in
        //     gemma4_layer_exec).  This preserves the per-channel scale
        //     via SFA/SFB prep.
        //   * num_tokens < 128 : per-token loop is optimal (fp8_gemv is
        //     M=1-only; running it T times reads weights T times but each
        //     call is already bandwidth-bound; cost parity with any batched
        //     solution at small M).
        //
        // So we keep the per-token loop as the default for ALL prompt
        // lengths today. RVLLM_BATCH_PREFILL=1 flips to the unified
        // batch path (diagnostic: verifies CUTLASS >=128 correctness,
        // or measures the collapsed-scalar quality floor at <128).
        // Commit 56: one-shot override consumed here. When set,
        // run_generate returns Ok(Vec::new()) at the post-K-capture
        // early-out site (~line 9361) — the K-row hidden-state buffer
        // is the only output the caller cares about. Skips
        // row-extract + final_norm + lm_head_M=1 + softcap + argmax
        // + DtoH (= the "bonus" decode token the batched-verify
        // callers were discarding anyway).
        let force_prefill_only = self
            .force_prefill_only
            .swap(false, std::sync::atomic::Ordering::AcqRel);
        let skip_decode = force_prefill_only
            || std::env::var_os("RVLLM_DIAG_SKIP_DECODE").is_some();
        let requested_batch_prefill = parse_truthy_env("RVLLM_BATCH_PREFILL").unwrap_or(false);
        // Vision-splice availability gate (Codex review #1 round 5).
        // The vision-embedding splice into residual_ptr lives ONLY in
        // the batch-prefill code path (the post-EmbeddingGather hook
        // ~line 4017). The per-token prefill path computes embeddings
        // via the embedding-gather kernel directly and never sees
        // the vision items. If this request brings vision data, force
        // the batch path on regardless of the env flag — and refuse
        // up front when the KV dtype rules out the batch path so the
        // user gets a clean error instead of silently dropped images.
        let needs_batch_for_vision = !vision_splice.is_empty() || !audio_splice.is_empty();
        if needs_batch_for_vision && kv_dtype == crate::gemma4_layer_exec::KvDtype::F16 {
            return Err(rvllm_core::RvllmError::cuda(
                "vision: F16 KV cache cannot service vision splice — \
                 the per-token prefill path the F16-KV setup uses doesn't \
                 carry the splice. Set RVLLM_F16_KV=0 (or use NVFP4 KV) for \
                 multimodal requests.",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        let use_batch_prefill =
            (requested_batch_prefill || needs_batch_for_vision)
                && kv_dtype != crate::gemma4_layer_exec::KvDtype::F16;
        if requested_batch_prefill
            && !use_batch_prefill
            && !needs_batch_for_vision
        {
            eprintln!(
                "[prefill] RVLLM_BATCH_PREFILL=1 ignored for F16 KV; \
                 using per-token path to keep KV dtype consistent"
            );
        }
        // Edge cases the batch-prefill block at line ~3420 cannot
        // service correctly:
        //   * prompt_len <= 1 — the gate further down uses
        //     `prompt_len > 1` so the batch block never runs, and
        //     without the per-token fallback below `residual_ptr`
        //     would stay stale and the LM head would read garbage.
        //   * total_new_q == 0 (full prefix-cache hit) — the chunk
        //     loop never iterates, `new_q` stays 0, and the diag
        //     capture below uses `new_q - 1` which underflows
        //     usize and reads from invalid memory.
        // Both collapse to "no real work for batch-prefill" → fall
        // through to the per-token path which handles them.
        let total_new_q_for_gate = prompt_len.saturating_sub(common_prefix_len);
        let use_batch_path =
            use_batch_prefill && prompt_len > 1 && total_new_q_for_gate > 0;
        if !skip_decode && !use_batch_path {
            // Prefix-cache fast path: if common_prefix_len > 0, the
            // persistent KV region already holds valid entries for
            // slots [0..common_prefix_len). Skip those tokens; the
            // per-token loop picks up at the first new token, attention
            // reads the cached KV for context. Batch-prefill path
            // below doesn't use this shortcut yet (unified kernel
            // would need partial-query support wired through).
            let start = common_prefix_len as usize;
            if start >= prompt_len as usize && prompt_len > 0 {
                // Full prefix-cache hit: every prompt token's KV is
                // already cached, but `residual_ptr` is empty for THIS
                // request. Recompute the last token's residual through
                // the per-layer loop so the LM head has something to
                // consume. The KV write at slot `prompt_len-1`
                // overwrites cached value with the same bits
                // (idempotent given identical prompt), so cache state
                // stays consistent for subsequent requests.
                let last_idx = (prompt_len - 1) as usize;
                run_one_token(prompt_ids[last_idx], last_idx)?;
            } else {
                for (i, &tok) in prompt_ids.iter().enumerate().skip(start) {
                    run_one_token(tok, i)?;
                }
            }
        }

        // Optional prefill-vs-decode residual compare (RVLLM_DIAG_COMPARE=1).
        // Captures the last-token residual produced by per-token decode
        // (correct reference), resets KV, re-runs the prompt via batch
        // prefill, captures the same row, and prints the diff. Combine
        // with `RVLLM_MAX_LAYERS=N` to bisect where the two paths
        // diverge. Only fires when prompt_len > 1 (decode==prefill
        // trivially at prompt_len=1).
        let diag_compare =
            std::env::var_os("RVLLM_DIAG_COMPARE").is_some() && !skip_decode;
        let mut decode_ref_last: Vec<u16> = Vec::new();
        let mut decode_ref_first: Vec<u16> = Vec::new();
        if diag_compare && prompt_len > 1 {
            // Already captured: residual_ptr holds LAST token's residual
            // after all prompt tokens were processed sequentially.
            self.stream.fence()?;
            decode_ref_last = vec![0u16; hidden as usize];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                decode_ref_last.as_mut_ptr() as *mut _,
                residual_ptr,
                (hidden * 2) as _,
            );

            // For FIRST token reference, re-run just token 0 through
            // a fresh KV cache — the residual after that step is what
            // prefill's row 0 should match (no prior context at
            // position 0 in either path).
            cudarc::driver::sys::cuMemsetD8_v2(kv_cache.device_ptr(), 0, kv_total_bytes as usize);
            self.stream.fence()?;
            run_one_token(prompt_ids[0], 0)?;
            self.stream.fence()?;
            decode_ref_first = vec![0u16; hidden as usize];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                decode_ref_first.as_mut_ptr() as *mut _,
                residual_ptr,
                (hidden * 2) as _,
            );

            // Reset KV again before prefill re-runs the whole prompt.
            cudarc::driver::sys::cuMemsetD8_v2(kv_cache.device_ptr(), 0, kv_total_bytes as usize);
            self.stream.fence()?;
        }

        // Batch-prefill path. Originally retained as instrumentation
        // (see commit history for the diag-only days), now also the
        // production path on sm_121/NVFP4 when RVLLM_BATCH_PREFILL=1
        // — the parameters_for_nvfp4_sm121.md profile recommends it
        // and Cortex production runs with this flag set. The earlier
        // "correctness is still broken" comment was stale.
        //
        // Two edge cases route AWAY from this block via the gate
        // above: prompt_len ≤ 1 (single-token prompts have no
        // batched work) and total_new_q == 0 (full prefix-cache
        // hit, no NEW tokens to prefill). Per-token fallback handles
        // both. With those guarded, the chunk loop below always
        // sees total_new_q ≥ 1 and `new_q ≥ 1` after the first
        // iteration, so `new_q - 1` arithmetic is safe.
        if (diag_compare && prompt_len > 1) || skip_decode || use_batch_path {
            // Prefix-cache aware batch prefill, OPTIONALLY chunked.
            //
            // When `use_prefix_cache` reports a common prefix of length
            // L, we skip prefill for slots [0..L). The remaining
            // `prompt_len - L` new tokens get processed in chunks of
            // `RVLLM_PREFILL_CHUNK_SIZE` (0 = single chunk / all new
            // tokens at once, matching the pre-chunked path).
            //
            // Each chunk is a "partial query with full-prefix KV
            // history" — the unified kernel handles this natively via
            // `context_lens = chunk_end, cu_seqlens_q = [0, chunk_q],
            // positions = [chunk_start..chunk_end)`. After the chunk
            // runs, its KV is in the persistent cache for the next
            // chunk's attention reads.
            //
            // Diag mode forces L=0 + single chunk so the row-0 /
            // row-(N-1) rel_err comparison stays meaningful.
            let prefix_skip = if diag_compare { 0 } else { common_prefix_len };
            let total_new_q = prompt_len - prefix_skip;
            let chunk_env: u32 = std::env::var("RVLLM_PREFILL_CHUNK_SIZE")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let chunk_size_max: u32 = if diag_compare || chunk_env == 0 {
                total_new_q
            } else {
                chunk_env
            };

            if chunk_size_max < total_new_q {
                eprintln!(
                    "[prefill-chunk] total_new_q={} chunk_size_max={} num_chunks={}",
                    total_new_q, chunk_size_max,
                    (total_new_q + chunk_size_max - 1) / chunk_size_max
                );
            }

            // Outer chunk loop. `new_q` at end-of-block holds the LAST
            // chunk's Q length so the downstream last-token-residual
            // extract picks the right row.
            let mut chunk_start_abs: u32 = prefix_skip;
            let mut new_q: u32 = 0;
            let mut chunk_idx: u32 = 0;
            while chunk_start_abs < prompt_len {
                let chunk_end_abs = std::cmp::min(
                    chunk_start_abs + chunk_size_max,
                    prompt_len,
                );
                let chunk_q = chunk_end_abs - chunk_start_abs;
                new_q = chunk_q;
                // HF Gemma4 replaces multimodal placeholder IDs with
                // pad_token_id (=0 on E4B) before the embedding lookup
                // and PLE gather; the actual multimodal embeddings get
                // spliced in afterwards. Without this swap the PLE
                // path indexes embed_tokens_per_layer at out-of-range
                // ids (vision 258880, audio 258881) producing garbage
                // per-layer embeddings that hijack attention and
                // collapse the response to prompt echo / token storm.
                // Codex round 9 fix.
                let chunk_a = chunk_start_abs as usize;
                let chunk_b = chunk_end_abs as usize;
                let mut tok_ids: Vec<i32> = prompt_ids[chunk_a..chunk_b]
                    .iter().map(|&t| t as i32).collect();
                for (slot_start, emb_bytes) in vision_splice {
                    let slot_n = emb_bytes.len() / ((hidden as usize) * 2);
                    let lo = (*slot_start).max(chunk_a);
                    let hi = (slot_start + slot_n).min(chunk_b);
                    for i in lo..hi { tok_ids[i - chunk_a] = 0; }
                }
                for (slot_start, emb_bytes) in audio_splice {
                    let slot_n = emb_bytes.len() / ((hidden as usize) * 2);
                    let lo = (*slot_start).max(chunk_a);
                    let hi = (slot_start + slot_n).min(chunk_b);
                    for i in lo..hi { tok_ids[i - chunk_a] = 0; }
                }
                token_ids_region.copy_from_host(bytemuck_cast_i32(&tok_ids))?;
                // Round-19 P1: this readback was a hand-rolled diagnostic
                // for verifying that `prefix_skip` slid the chunk window
                // correctly during the batch-prefill bring-up. It used
                // to fire unconditionally on every batch-prefill request
                // — `self.stream.fence()` + `cuMemcpyDtoH_v2` over the
                // whole chunk + synchronous `eprintln!` to stderr —
                // serialising the GPU pipeline and disk-I/O-bottlenecking
                // the worker per request. Now it's gated on the same
                // `RVLLM_DIAG_COMPARE` flag the rest of the prefill
                // diff harness uses.
                if chunk_idx == 0 && diag_compare {
                    self.stream.fence()?;
                    let mut readback = vec![0i32; chunk_q as usize];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        readback.as_mut_ptr() as *mut _,
                        token_ids_region.device_ptr(),
                        (chunk_q * 4) as _,
                    );
                    eprintln!(
                        "[DIAG] batch-prefill prefix_skip={} chunk0_q={} readback[..min(8)]={:?}",
                        prefix_skip, chunk_q, &readback[..readback.len().min(8)]
                    );
                }
                rvllm_fused::EmbeddingGatherLaunch { num_tokens: chunk_q, hidden, vocab }
                    .launch(fn_embed, residual_ptr, self.model.embedding.offset_bytes,
                        token_ids_region.device_ptr(), stream)?;

                // Phase 3b vision splice: overwrite residual rows for any
                // image-soft-token positions that fall inside this chunk,
                // BEFORE the optional bf16 widen so the rest of the chain
                // is dtype-uniform.
                //
                // No surrounding `self.stream.fence()`: every op here —
                // the embedding_gather above, our HtoDAsync, and the
                // F16ToBf16Launch below — runs on the same stream. CUDA
                // serialises stream-ordered work, so the fences only
                // host-stalled the worker without changing correctness.
                //
                // We DO check `cuMemcpyHtoDAsync_v2`'s return code now:
                // a copy error here used to silently leave the residual
                // populated with text-token embeddings while the rest
                // of the prefill ran to completion, producing
                // syntactically-clean but semantically wrong output for
                // the image-bearing chunk.
                if !vision_splice.is_empty() {
                    let row_bytes = (hidden as usize) * 2;
                    for (slot_start, emb_bytes) in vision_splice {
                        let slot_n = emb_bytes.len() / row_bytes;
                        let slot_end = slot_start + slot_n;
                        let chunk_a = chunk_start_abs as usize;
                        let chunk_b = chunk_end_abs as usize;
                        let lo = std::cmp::max(*slot_start, chunk_a);
                        let hi = std::cmp::min(slot_end, chunk_b);
                        if hi <= lo { continue; }
                        let n_rows = hi - lo;
                        let dst_off = ((lo - chunk_a) * row_bytes) as u64;
                        let src_off = ((lo - slot_start) * row_bytes) as usize;
                        let rc = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                            residual_ptr + dst_off,
                            emb_bytes[src_off .. src_off + n_rows * row_bytes].as_ptr() as *const _,
                            n_rows * row_bytes,
                            self.stream.raw() as _,
                        );
                        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                            return Err(rvllm_core::RvllmError::cuda(
                                "gemma4 vision splice cuMemcpyHtoDAsync",
                                rvllm_core::CudaErrorKind::MemcpyFailed,
                                rvllm_core::CudaCtx::setup(),
                            ));
                        }
                    }
                }
                if !audio_splice.is_empty() {
                    let row_bytes = (hidden as usize) * 2;
                    for (slot_start, emb_bytes) in audio_splice {
                        let slot_n = emb_bytes.len() / row_bytes;
                        let slot_end = slot_start + slot_n;
                        let chunk_a = chunk_start_abs as usize;
                        let chunk_b = chunk_end_abs as usize;
                        let lo = std::cmp::max(*slot_start, chunk_a);
                        let hi = std::cmp::min(slot_end, chunk_b);
                        if hi <= lo { continue; }
                        let n_rows = hi - lo;
                        let dst_off = ((lo - chunk_a) * row_bytes) as u64;
                        let src_off = ((lo - slot_start) * row_bytes) as usize;
                        let rc = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                            residual_ptr + dst_off,
                            emb_bytes[src_off .. src_off + n_rows * row_bytes].as_ptr() as *const _,
                            n_rows * row_bytes,
                            self.stream.raw() as _,
                        );
                        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                            return Err(rvllm_core::RvllmError::cuda(
                                "gemma4 audio splice cuMemcpyHtoDAsync",
                                rvllm_core::CudaErrorKind::MemcpyFailed,
                                rvllm_core::CudaCtx::setup(),
                            ));
                        }
                    }
                }

                // E4B PLE precompute (stage 3b2b). Must run BEFORE the
                // bf16 widen below since the GEMM in the helper expects
                // f16 inputs_embeds. Gated by RVLLM_E4B_PLE=1 until the
                // wiring is verified across all dispatch sites; default
                // off keeps the existing token-salad behavior so the
                // smoke baseline doesn't regress unexpectedly while
                // 3b2b lands.
                let ple_enabled = std::env::var("RVLLM_E4B_PLE")
                    .map_or(false, |v| v == "1");
                let (ple_base, ple_stride_elems) = if ple_enabled {
                    unsafe {
                        self.precompute_ple(
                            residual_ptr,
                            token_ids_region.device_ptr(),
                            chunk_q as u32,
                            fn_embed,
                            &kernels,
                            stream as u64,
                        )?
                    }
                } else {
                    (0u64, 0u32)
                };
                // Cycle 54 Stage 1: widen embedding-gather output (f16)
                // to bf16 in-place so the chunked-prefill residual chain
                // operates on bf16 between layers.
                if bf16_residual_enabled() {
                    rvllm_fused::gemma4_launcher::F16ToBf16Launch {
                        n: chunk_q * hidden,
                    }.launch(kernels.f16_to_bf16, residual_ptr, residual_ptr, stream)?;
                }
                // Codex Round 3 #1: post-gather row dump is a
                // diagnostic — gate behind RVLLM_DIAG_COMPARE so the
                // batch-prefill hot path (also the spec-verify path
                // via force_prefill_only) doesn't pay an unconditional
                // stream.fence + 2 DtoH + eprintln on chunk 0 of
                // every request.
                if chunk_idx == 0 && diag_compare {
                    self.stream.fence()?;
                    let mut r0 = vec![0u16; 4];
                    let mut r_n_minus_1 = vec![0u16; 4];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(r0.as_mut_ptr() as *mut _, residual_ptr, 8);
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        r_n_minus_1.as_mut_ptr() as *mut _,
                        residual_ptr + ((chunk_q - 1) as u64 * hidden as u64 * 2),
                        8,
                    );
                    eprintln!(
                        "[DIAG] post-gather row0[..4]={:?} rowN-1[..4]={:?}",
                        r0.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect::<Vec<_>>(),
                        r_n_minus_1.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect::<Vec<_>>(),
                    );
                }

                // positions / slot_mapping for THIS chunk; cu_seq + ctx
                // frame the partial query within the full sequence.
                let pos: Vec<i32> = (chunk_start_abs as i32 .. chunk_end_abs as i32).collect();
                let slot: Vec<i32> = (chunk_start_abs as i32 .. chunk_end_abs as i32).collect();
                let ctx = [chunk_end_abs as i32];
                let cu_seq = [0i32, chunk_q as i32];
                positions.copy_from_host(bytemuck_cast_i32(&pos))?;
                slot_mapping.copy_from_host(bytemuck_cast_i32(&slot))?;
                context_lens.copy_from_host(bytemuck_cast_i32(&ctx))?;
                cu_seqlens_q.copy_from_host(bytemuck_cast_i32(&cu_seq))?;

                let phase = crate::gemma4_layer_exec::Gemma4Phase::Prefill {
                    cu_seqlens_q: cu_seqlens_q.device_ptr(),
                    max_seqlen_q: chunk_q,
                    num_seqs: 1,
                };

                for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                if layer_idx >= max_layers { break; }
                let lt = arch.layer_types[layer_idx];
                let hd = arch.head_dim_for_layer(layer_idx) as u32;
                let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                let q_dim = (arch.num_attention_heads as u32) * hd;
                let kv_dim = nkvh * hd;
                let layer_blocks = if lt == Gemma4LayerType::GlobalAttention { num_blocks_total } else { sliding_blocks };
                let layer_kv_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                // E4B kv-share: when this layer aliases an earlier source
                // (Gemma 4 num_kv_shared_layers tail), the attention
                // launchers must read K/V from the SOURCE layer's region.
                // Pointing layer_kv_base at the source while passing
                // dims.kv_share_source_layer=Some(_) suppresses rope's
                // K/V writes (see fused_rope_partial_*kv.cu nullptr guard)
                // so the source layer's K/V cache is never clobbered.
                let kv_idx = arch.kv_share_source_layer(layer_idx).unwrap_or(layer_idx);
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[kv_idx];
                // kv-share-aware (mirrors layer_kv_base above): for shared
                // layers, the scale arena base also maps to the source layer.
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[kv_idx];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);
                // Per-layer dtype: hybrid swaps global to FP8 (sliding stays env default).
                // Cycle 24: pass layer_idx for RVLLM_FP8_KV_LAYERS list env.
                let kv_dtype = crate::gemma4_layer_exec::KvDtype::for_layer_index_or_env(lt, layer_idx, false);
                // Prefill uses FP8 KV when the ambient dtype is F16
                // (no F16 prefill kernel exists); NVFP4 prefill stays on NVFP4.
                let prefill_kv_dtype = if kv_dtype == crate::gemma4_layer_exec::KvDtype::Nvfp4 {
                    crate::gemma4_layer_exec::KvDtype::Nvfp4
                } else {
                    crate::gemma4_layer_exec::KvDtype::Fp8
                };
                let (k_cache_scale, v_cache_scale) = if prefill_kv_dtype
                    == crate::gemma4_layer_exec::KvDtype::Nvfp4
                {
                    (layer_kv_scale_base, layer_kv_scale_base + layer_kv_elems / 32)
                } else {
                    (0u64, 0u64)
                };

                let dims = crate::gemma4_layer_exec::Gemma4LayerDims {
                    num_tokens: new_q, hidden,
                    num_heads: arch.num_attention_heads as u32, num_kv_heads: nkvh, head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    intermediate: inter,
                    ple_dim: arch.hidden_size_per_layer_input.unwrap_or(0) as u32,
                    block_size,
                    max_blocks_per_seq: layer_blocks, num_blocks_total: layer_blocks,
                    attn_scale: 1.0, rms_eps: arch.rms_norm_eps,
                    layer_type: lt, sliding_window: arch.sliding_window_size as u32,
                    f16_kv: false, // prefill uses FP8 KV (no F16 prefill kernel)
                    kv_dtype: prefill_kv_dtype,
                    bf16_residual: bf16_residual_enabled(),
                    kv_share_source_layer: arch.kv_share_source_layer(layer_idx).map(|s| s as u32),
                    // Codex17-1: batch-prefill writes context_lens=chunk_end_abs and
                    // the unified-prefill kernel uses it to index block_tables. Pass
                    // it to the validator so OOB reads on long prompts/chunks past
                    // max_blocks_per_seq*block_size are caught at validate() time
                    // instead of becoming a silent garbage-block-ID kernel read.
                    current_max_context_len: Some(chunk_end_abs as u32),
                };
                let w = crate::gemma4_layer_exec::Gemma4LayerWeightPtrs {
                    attn_norm_gamma: layer.input_layernorm.offset_bytes,
                    post_attn_norm_gamma: layer.post_attention_layernorm.offset_bytes,
                    pre_ff_norm_gamma: layer.pre_feedforward_layernorm.offset_bytes,
                    post_ff_norm_gamma: layer.post_feedforward_layernorm.offset_bytes,
                    q_norm_gamma: layer.q_norm.offset_bytes,
                    k_norm_gamma: layer.k_norm.offset_bytes,
                    qkv_fp8: layer.qkv.as_ref().map_or(0, |w| w.offset_bytes), qkv_scale: layer.qkv.as_ref().map_or(0, |w| w.scale_ptr),
                    o_fp8: layer.o_proj.as_ref().map_or(0, |w| w.offset_bytes), o_scale: layer.o_proj.as_ref().map_or(0, |w| w.scale_ptr),
                    gate_up_fp8: layer.gate_up.as_ref().map_or(0, |w| w.offset_bytes), gate_up_scale: layer.gate_up.as_ref().map_or(0, |w| w.scale_ptr),
                    down_fp8: layer.down_proj.as_ref().map_or(0, |w| w.offset_bytes), down_scale: layer.down_proj.as_ref().map_or(0, |w| w.scale_ptr),
                    layer_scalar_ptr: layer.layer_scalar.offset_bytes,
                    qkv_f16: layer.qkv_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    o_f16: layer.o_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    gate_up_f16: layer.gate_up_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    down_f16: layer.down_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    qkv_chscale: layer.qkv.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    o_chscale: layer.o_proj.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    gate_up_chscale: layer.gate_up.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    down_chscale: layer.down_proj.as_ref().and_then(|w| w.channelscale_ptr).unwrap_or(0),
                    qkv_blockscale: layer.qkv.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    o_blockscale: layer.o_proj.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    gate_up_blockscale: layer.gate_up.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    down_blockscale: layer.down_proj.as_ref().and_then(|w| w.blockscale_ptr).unwrap_or(0),
                    awq: awq_layer_ptrs(layer.awq.as_ref()),
                    // E4B PLE plumbing. Pointers from the loaded per-layer
                    // tensors (`None` → 0 on 31B/AWQ). `ple_per_layer_input`
                    // is populated by the per-request PLE precompute (Stage
                    // 3b); 0 here means PLE inactive for this forward
                    // dispatch — layer_exec then skips the PLE injection.
                    ple_input_gate: layer
                        .per_layer_input_gate
                        .as_ref()
                        .map_or(0, |w| w.offset_bytes),
                    ple_projection: layer
                        .per_layer_projection
                        .as_ref()
                        .map_or(0, |w| w.offset_bytes),
                    ple_post_input_norm_gamma: layer
                        .post_per_layer_input_norm
                        .as_ref()
                        .map_or(0, |w| w.offset_bytes),
                    // Per-layer slice into the [T, num_layers, ple_dim]
                    // precompute buffer. Base + L*ple_dim*2 walks the
                    // contiguous-in-D, strided-in-T layout that
                    // gelu_tanh_mul_dual_f16 expects via
                    // per_li_row_stride_elems = ple_stride_elems.
                    ple_per_layer_input: if ple_base != 0 {
                        ple_base + (layer_idx as u64)
                            * (arch.hidden_size_per_layer_input.unwrap_or(0) as u64)
                            * 2
                    } else {
                        0
                    },
                    ple_per_layer_stride_elems: ple_stride_elems,
                };
                // Row-major [num_tokens, q_dim+2*kv_dim]: k_out / v_out
                // point at row 0's K / V sub-slice. The rmsnorm kernel
                // applies `src_row_stride` to reach later tokens — the
                // old `num_tokens * q_dim * 2` formula assumed a
                // columnar "all Q then all K then all V" layout that
                // the cuBLASLt QKV GEMM does NOT produce.
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let (cos, sin) = match lt {
                    Gemma4LayerType::SlidingAttention => (self.model.rope_cos_sliding.offset_bytes, self.model.rope_sin_sliding.offset_bytes),
                    Gemma4LayerType::GlobalAttention => (self.model.rope_cos_global.offset_bytes, self.model.rope_sin_global.offset_bytes),
                };
                let bytes_per_half_kv = match prefill_kv_dtype {
                    crate::gemma4_layer_exec::KvDtype::F16 => layer_kv_elems,
                    crate::gemma4_layer_exec::KvDtype::Fp8 => layer_kv_elems / 2,
                    crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_kv_elems / 4,
                };
                // === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
                // Mirror the decode-path scratch population so batch
                // prefill ALSO writes f16 shadow K/V for the prompt
                // tokens. Without this, prefill silently bypasses the
                // shadow hook and the dump on decode step 0 only sees
                // the single new-token write — useless for analysis.
                // Q always routes to the shared throwaway during
                // prefill (we only capture per-layer Q on decode
                // step 0, which goes through run_one_token, not here).
                let prefill_is_shadow_layer = shadow_ptr != 0
                    && prefill_kv_dtype == crate::gemma4_layer_exec::KvDtype::Nvfp4
                    && layer_idx < shadow_layer_offsets.len()
                    && shadow_layer_offsets[layer_idx] != u64::MAX;
                let (prefill_shadow_k, prefill_shadow_v) = if prefill_is_shadow_layer {
                    let base = shadow_ptr + shadow_layer_offsets[layer_idx];
                    (base, base + layer_kv_elems)
                } else {
                    (0u64, 0u64)
                };
                let prefill_shadow_q = if prefill_is_shadow_layer {
                    shadow_q_throwaway_ptr
                } else {
                    0
                };
                // === END NVFP4 SHADOW DIAGNOSTIC ===
                // === HADAMARD ROTATION ===
                let prefill_hadamard_layer_ptr: u64 = if hadamard_base_ptr != 0
                    && prefill_kv_dtype == crate::gemma4_layer_exec::KvDtype::Nvfp4
                {
                    hadamard_base_ptr
                        + (layer_idx as u64) * (hadamard_head_dim_stride as u64)
                } else {
                    0
                };
                // === END HADAMARD ROTATION ===
                let scratch = crate::gemma4_layer_exec::Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(), hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base, k_out, v_out,
                    q_normed: q_normed.device_ptr(), k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + bytes_per_half_kv,
                    k_cache_scale,
                    v_cache_scale,
                    q_scale_ptr: q_scale_region.device_ptr(), kv_scale_ptr: kv_scale_region.device_ptr(),
                    k_scale_cache: layer_kv_scale_base,
                    v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                    q_scale_cache: q_scale_cache_ptr,
                    attn_out: attn_out.device_ptr(), attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(), delta_f16: delta_f16.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(), gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(), mlp_out_scale: mlp_out_scale.device_ptr(),
                    gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                    // Codex40-3: run_generate alloc uses `max_tokens`,
                    // not num_seqs.
                    gemm_f32_tmp_bytes: (max_tokens * gemm_f32_max_n * 4) as usize,
                    cutlass_workspace: cutlass_ws.device_ptr(), cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                    fa3_workspace_bytes: FA3_WS_BYTES as u64,
                    shadow_k_cache: prefill_shadow_k,
                    shadow_v_cache: prefill_shadow_v,
                    shadow_q_cache: prefill_shadow_q,
                    // === HADAMARD ROTATION ===
                    hadamard_signs_q: prefill_hadamard_layer_ptr,
                    hadamard_signs_k: prefill_hadamard_layer_ptr,
                    // === END HADAMARD ROTATION ===
                };
                let meta = crate::gemma4_layer_exec::Gemma4MetadataPtrs {
                    positions: positions.device_ptr(), slot_mapping: slot_mapping.device_ptr(),
                    cos, sin,
                    block_tables: block_tables.device_ptr(), context_lens: context_lens.device_ptr(),
                };
                crate::gemma4_layer_exec::gemma4_forward_phase(
                    dims, &kernels, &w, &scratch, &meta,
                    &self.cublaslt, &self.cutlass, &self.sliding_attention, &self.global_attention,
                    residual_ptr, stream, phase,
                )?;
                // Cycle 53 step 4: per-layer residual L2-norm dump for the
                // last chunk's last few rows. Gated by
                // RVLLM_DUMP_RESIDUAL_NORMS=1. Only dumps for the final
                // chunk (where the last token is what the LM head will
                // consume). Goal: localize the layer where residual
                // magnitude collapses on long-ctx WEATHER vs stays sane
                // on long-ctx WHO.
                if std::env::var("RVLLM_DUMP_RESIDUAL_NORMS").is_ok()
                    && chunk_end_abs == prompt_len
                    && chunk_q > 0
                {
                    self.stream.fence()?;
                    let n_rows: u32 = chunk_q.min(5);
                    let row0 = (chunk_q - n_rows) as u64;
                    let bytes_per_row = (hidden as u64) * 2;
                    let mut buf = vec![0u16; (n_rows as usize) * (hidden as usize)];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        buf.as_mut_ptr() as *mut _,
                        residual_ptr + row0 * bytes_per_row,
                        ((n_rows as u64) * bytes_per_row) as usize,
                    );
                    // Cycle 54 step 4: bf16-aware decode. When the residual
                    // chain is bf16 (RVLLM_RESIDUAL_BF16=1), reinterpret
                    // the same u16 buffer through the bf16 unpacker so the
                    // dump shows real magnitudes. Single env probe per dump
                    // batch — cheap.
                    let to_f32: fn(u16) -> f32 = if bf16_residual_enabled() {
                        crate::bring_up::bf16_to_f32
                    } else {
                        crate::bring_up::f16_to_f32
                    };
                    let dt_tag = if bf16_residual_enabled() { "bf16" } else { "f16" };
                    let mut norms = String::new();
                    let mut nan_or_inf = false;
                    for r in 0..n_rows as usize {
                        let mut sum_sq = 0.0f64;
                        let mut amax = 0.0f32;
                        for c in 0..hidden as usize {
                            let f = to_f32(buf[r * hidden as usize + c]);
                            if !f.is_finite() { nan_or_inf = true; }
                            sum_sq += (f as f64) * (f as f64);
                            if f.abs() > amax { amax = f.abs(); }
                        }
                        let l2 = (sum_sq / hidden as f64).sqrt();
                        norms.push_str(&format!(" r{}={:.3e}/amax={:.3e}", r, l2, amax));
                    }
                    let last_row_off = (n_rows as usize - 1) * hidden as usize;
                    let last4: Vec<f32> = (hidden as usize - 4..hidden as usize)
                        .map(|i| to_f32(buf[last_row_off + i])).collect();
                    eprintln!(
                        "[res-norm/{}] L{} chunk{}{}{} last4={:?}",
                        dt_tag, layer_idx, chunk_idx,
                        if nan_or_inf { " NAN/INF" } else { "" },
                        norms, last4,
                    );
                }
            }
                chunk_start_abs = chunk_end_abs;
                chunk_idx += 1;
            } // end chunk loop

            // Codex Round 3 #1: diag capture is only consumed by the
            // diag_compare `else` branch below (lines ~10379-10384
            // `stats()` calls). Gating the fence + 2 hidden-sized
            // DtoH behind the same flag removes an unconditional
            // host-readback from the batch-prefill hot path. The
            // K-hidden capture above doesn't need this fence — its
            // DtoD is stream-ordered with the downstream argmax
            // DtoH the spec-verify path already issues.
            let mut prefill_first: Vec<u16> = Vec::new();
            let mut prefill_last: Vec<u16> = Vec::new();
            if diag_compare {
                self.stream.fence()?;
                prefill_first = vec![0u16; hidden as usize];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    prefill_first.as_mut_ptr() as *mut _,
                    residual_ptr,
                    (hidden * 2) as _,
                );
                // The residual buffer holds `new_q` rows after the
                // layer loop (prefix-cached slots are not re-prefilled).
                // Row `new_q - 1` is the last prompt token regardless
                // of how many tokens were cached.
                prefill_last = vec![0u16; hidden as usize];
                let last_off_diag = (new_q - 1) as u64 * hidden as u64 * 2;
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    prefill_last.as_mut_ptr() as *mut _,
                    residual_ptr + last_off_diag,
                    (hidden * 2) as _,
                );
            }

            // Commit 40 — spec-decode batched verify, FIRST half.
            // Capture LAST K residual rows BEFORE the single-row
            // extraction below collapses everything to row 0. These
            // are POST-layer-loop, PRE-final-norm hiddens.
            //
            // The consumer (`run_generate_speculative_batched`)
            // applies final-norm + lm_head + softcap + argmax on the
            // K-buffer itself, decoupled from this call's lm_head
            // (which still produces 1 logit row for the bonus decode).
            //
            // Default path: ptr == 0 → cheap atomic load + branch.
            {
                let k_dst = self
                    .base_last_k_hidden_ptr
                    .load(std::sync::atomic::Ordering::Acquire);
                if k_dst != 0
                    && self
                        .base_last_k_snapshot_pending
                        .swap(false, std::sync::atomic::Ordering::AcqRel)
                {
                    let k_requested = self
                        .base_last_k_count
                        .load(std::sync::atomic::Ordering::Acquire);
                    if (k_requested as usize) > MAX_SPEC_K {
                        return Err(rvllm_core::RvllmError::Config {
                            err: rvllm_core::ConfigError::InvalidField {
                                name: "RVLLM_GEMMA4_SPEC_K",
                                reason: format!(
                                    "{} exceeds MAX_SPEC_K={} (compile-time buffer cap)",
                                    k_requested, MAX_SPEC_K,
                                ).into(),
                            },
                            field: "RVLLM_GEMMA4_SPEC_K",
                        });
                    }
                    let k: u32 = k_requested.min(new_q);
                    if k > 0 {
                        let row_bytes = (hidden as usize) * 2;
                        let src_off = (new_q - k) as u64 * hidden as u64 * 2;
                        let rc = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                            k_dst,
                            residual_ptr + src_off,
                            (k as usize) * row_bytes,
                            stream as cudarc::driver::sys::CUstream,
                        );
                        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                            return Err(rvllm_core::RvllmError::cuda(
                                "run_generate: base_last_k_hidden DtoD capture",
                                rvllm_core::CudaErrorKind::MemcpyFailed,
                                rvllm_core::CudaCtx::setup(),
                            ));
                        }
                    }
                }
            }

            // Extract last token's residual for decode
            if new_q > 1 {
                let last_offset = (new_q - 1) as u64 * hidden as u64 * 2;
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    residual_ptr, residual_ptr + last_offset, (hidden * 2) as usize, stream as _,
                );
            }

            if !diag_compare {
                if skip_decode { return Ok(Vec::new()); }
                // use_batch_prefill: fall through to LM head.
            } else {
                let stats = |label: &str, reference: &[u16], probe: &[u16]| {
                    let mut max_abs = 0f32;
                    let mut sum_sq_diff = 0f64;
                    let mut sum_sq_ref = 0f64;
                    let mut first_diffs: Vec<(f32, f32)> = Vec::new();
                    for i in 0..hidden as usize {
                        let d = crate::bring_up::f16_to_f32(reference[i]);
                        let p = crate::bring_up::f16_to_f32(probe[i]);
                        let diff = (d - p).abs();
                        if diff > max_abs { max_abs = diff; }
                        sum_sq_diff += (diff as f64) * (diff as f64);
                        sum_sq_ref += (d as f64) * (d as f64);
                        if first_diffs.len() < 4 { first_diffs.push((d, p)); }
                    }
                    let rel_err = (sum_sq_diff / sum_sq_ref.max(1e-18)).sqrt();
                    eprintln!(
                        "[DIAG {label}] max_abs={max_abs:.4} rel_err={rel_err:.4e} \
                         first4_ref_probe={first_diffs:?}",
                    );
                };
                eprintln!(
                    "[DIAG] max_layers={} prompt_len={} hidden={}",
                    max_layers, prompt_len, hidden,
                );
                stats("row=0 (first token)", &decode_ref_first, &prefill_first);
                stats("row=N-1 (last token)", &decode_ref_last, &prefill_last);
            }
        }

        // LM head on last prompt token.
        // Cycle 54 Stage 1: when bf16 residual is active, run the bf16
        // rmsnorm sibling (math is identical, both use f32 accumulator
        // internally) and then narrow bf16→f16 in-place before the
        // existing f16 LM head GEMM consumes it.
        let lm_head_bf16 = bf16_residual_enabled();
        let lm_head_norm_kernel = if lm_head_bf16 {
            kernels.rmsnorm_inplace_bf16
        } else {
            kernels.fused_rmsnorm
        };
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1, hidden, eps: arch.rms_norm_eps,
        }.launch(lm_head_norm_kernel, residual_ptr, self.model.final_norm.offset_bytes, stream)?;
        if lm_head_bf16 {
            rvllm_fused::gemma4_launcher::Bf16ToF16SatLaunch { n: hidden }
                .launch(kernels.bf16_to_f16_sat, residual_ptr, residual_ptr, stream)?;
        }
        // Spec-decode commits 16 + 18: one-shot snapshot of the
        // normalized pre-lm-head hidden.
        //
        // The caller (`run_generate_speculative`) sets
        // `base_last_hidden_snapshot_pending = true` BEFORE invoking
        // run_generate. The first hook fire (= post-PREFILL
        // final_norm, last prompt token's hidden) consumes the flag
        // and copies; subsequent decode-step fires see the flag
        // cleared and skip.
        //
        // Without this gating, max_new>1 left the buffer holding the
        // hidden at position prompt_len + max_new - 1, which is
        // off-position vs. the drafter's step 1 input (which wants
        // position prompt_len, i.e. the LAST PROMPT TOKEN's hidden).
        // The off-position state systematically suppressed accept
        // rate.
        //
        // Three-way gate: ptr != 0 (buffer allocated) && pending
        // (caller asked) && load_then_swap (one-shot semantics).
        // Default non-spec runs: ptr==0 so an atomic load + branch.
        {
            let dst = self
                .base_last_hidden_ptr
                .load(std::sync::atomic::Ordering::Acquire);
            if dst != 0
                && self
                    .base_last_hidden_snapshot_pending
                    .swap(false, std::sync::atomic::Ordering::AcqRel)
            {
                let rc = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dst,
                    residual_ptr,
                    (hidden as usize) * 2,
                    stream as cudarc::driver::sys::CUstream,
                );
                if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "run_generate: base_last_hidden DtoD capture",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            // (commit 40) base_last_k_hidden capture moved to the
            // earlier site, BEFORE row-extraction collapses
            // residual_ptr to a single row. See same-commit edit
            // ~30 lines above.
        }
        self.cublaslt.f16_gemm_f32(residual_ptr, self.model.lm_head_f16.offset_bytes,
            logits_f32.device_ptr(), 1, vocab as i32, hidden as i32, stream)?;
        // Codex40-2: apply Gemma final logit softcap before bias /
        // sampling. Bench + PPL paths apply this; generate previously
        // skipped it so greedy/top-p decisions could flip on
        // ungated large logits, and tool-call bias landed on
        // pre-softcap values that don't match training-time
        // distribution. f32 variant matches the f32 logits dtype.
        if arch.logit_softcap > 0.0 {
            rvllm_fused::gemma4_launcher::LogitSoftcapLaunch {
                num_tokens: 1,
                vocab,
                cap: arch.logit_softcap,
            }
            .launch(self.fused.fn_softcap_f32, logits_f32.device_ptr(), stream)?;
        }
        // === TOOL-CALL OPEN-TAG BIAS (cycle 14 pragmatic fix) ===
        // Cumulative quantization noise over 60 layers gives ~1.5 logit
        // units of consistent bias toward token 49 ("<tool_call|>",
        // close-tag) over token 48 ("<|tool_call>", open-tag) at
        // decode-step 0 of long-context tool-eligible prompts. The
        // model gets razor-thin margins (0.27 vs 17+ in clean WHO)
        // and greedy lands on the wrong special token, which has no
        // valid training continuation → garbage spiral.
        // Apply a small positive bias to token 48 to nudge the choice
        // toward the training-shape canonical open-tag. Env-gated so
        // it can be A/B'd. Default OFF for safety.
        let tool_call_bias = std::env::var("RVLLM_TOOL_CALL_OPEN_BIAS")
            .ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
        if tool_call_bias != 0.0 {
            // Token 48 = `<|tool_call>` (training-shape open). Add the
            // bias to its logit position.
            let mut logit_48: f32 = 0.0;
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                &mut logit_48 as *mut _ as *mut _,
                logits_f32.device_ptr() + 48 * 4,
                4,
            );
            let new_logit = logit_48 + tool_call_bias;
            cudarc::driver::sys::cuMemcpyHtoD_v2(
                logits_f32.device_ptr() + 48 * 4,
                &new_logit as *const _ as *const _,
                4,
            );
        }
        // === END TOOL-CALL OPEN-TAG BIAS ===
        let mut host_tok = [0i32; 1];
        if sampling_temp > 0.0 {
            // Host-side temperature sampling
            self.stream.fence()?;
            host_tok[0] = host_sample_token(
                logits_f32.device_ptr(), vocab, sampling_temp, sampling_top_p, sampling_top_k,
                &mut next_rand_f32,
            )? as i32;
        } else {
            rvllm_fused::ArgmaxLaunch { num_tokens: 1, vocab }
                .launch(fn_argmax, logits_f32.device_ptr(), sampled.device_ptr(), stream)?;
            self.stream.fence()?;
            // Same anti-pattern as the host_sample_token DtoH already
            // fixed in Codex3: a silently-failing greedy-argmax DtoH
            // would leave host_tok[0] == 0 (or a stale value from a
            // prior step) and the server would emit token-0 as if the
            // greedy decode succeeded. CUDA faults must surface as
            // 500s, not as wrong tokens that look like model
            // hallucinations.
            cuda_check!(
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    host_tok.as_mut_ptr() as *mut _,
                    sampled.device_ptr(),
                    4,
                ),
                "argmax_sample_token_dtoh_decode",
                stream
            );
        }
        // Commit 26: spec-decode logits capture, prefill argmax row (row 0).
        // Mirrors RVLLM_DUMP_TOPK_LOGITS below but writes into the
        // engine-wide buffer instead of stdout. Cost: ~1 MiB DtoH per
        // captured row when the gate is on; zero when off.
        if self.spec_logits_capture_active.load(std::sync::atomic::Ordering::Relaxed) {
            self.stream.fence()?;
            let mut buf = self.spec_decode_step_logits.lock().unwrap();
            let v = vocab as usize;
            if buf.len() >= v {
                let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut _,
                    logits_f32.device_ptr(),
                    v * 4,
                );
            }
        }
        // Cycle 53 step 5: top-K logit dump at first decode step. Gated by
        // RVLLM_DUMP_TOPK_LOGITS=1. Cost: 1 MiB DtoH + partial-sort vocab
        // entries (host-side). Compares logit distribution shape across
        // prompts: a sharp top-1 (margin >> 1.0 over top-2) means the
        // model is decisive; a flat top-K (margin < 0.5) means
        // cumulative quantization noise has overwhelmed signal — the
        // canonical long-ctx failure mode documented above
        // (RVLLM_TOOL_CALL_OPEN_BIAS rationale).
        if std::env::var("RVLLM_DUMP_TOPK_LOGITS").is_ok() {
            self.stream.fence()?;
            let mut logits_host = vec![0.0f32; vocab as usize];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                logits_host.as_mut_ptr() as *mut _,
                logits_f32.device_ptr(),
                (vocab as usize * 4) as _,
            );
            let mut idx: Vec<usize> = (0..vocab as usize).collect();
            // Partial sort: select top-10 by logit
            idx.sort_by(|&a, &b| logits_host[b].partial_cmp(&logits_host[a]).unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<(usize, f32)> = idx.iter().take(10)
                .map(|&i| (i, logits_host[i])).collect();
            let mean = logits_host.iter().sum::<f32>() / vocab as f32;
            let mut sum_sq = 0.0f64;
            let mut amax = f32::NEG_INFINITY;
            let mut amin = f32::INFINITY;
            for &v in &logits_host {
                let d = (v - mean) as f64;
                sum_sq += d * d;
                if v > amax { amax = v; }
                if v < amin { amin = v; }
            }
            let std = (sum_sq / vocab as f64).sqrt();
            let margin_1_2 = top[0].1 - top[1].1;
            eprintln!(
                "[topk-logits] mean={:.3} std={:.3} amax={:.3} amin={:.3} margin(1-2)={:.3} top10={:?}",
                mean, std, amax, amin, margin_1_2, top,
            );
        }
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        // tracing::info! instead of unconditional eprintln! — operators
        // can gate via RUST_LOG, no synchronous stderr write on every
        // request.
        tracing::info!(
            tokens = prompt_ids.len(),
            ttft_ms = prefill_ms,
            "prefill complete"
        );

        let mut output_ids: Vec<u32> = Vec::with_capacity(max_new);
        let first_tok = host_tok[0] as u32;
        // Round-20 finding #2: check EOS BEFORE invoking the streaming
        // callback and BEFORE pushing to `output_ids`. The previous
        // order (push → cb → eos-check) leaked the EOS token into the
        // SSE stream and inflated `completion_tokens` by one. The cb
        // is what reaches the client; output_ids what reaches the
        // non-streaming response. Both must omit EOS.
        if eos_ids.contains(&first_tok) {
            return Ok(output_ids);
        }
        output_ids.push(first_tok);
        if let Some(cb) = on_token.as_mut() {
            if !cb(first_tok) {
                return Ok(output_ids);
            }
        }

        // Phase 2: Decode new tokens
        //
        // === CUDA Graph capture (cycle 60 step Y, RVLLM_DECODE_GRAPH=1) ===
        // The 60-layer-per-token kernel chain is ~660 cuLaunchKernel
        // calls per decode step. Eliminating per-step launch overhead
        // via CUDA Graph replay is the largest perf lever after the
        // partition-size win. We capture once on the second decode
        // step (decode_step == 1) — after eager warmup at step=0
        // populates KV[prompt_len] AND past the once-per-generate
        // shadow_q diagnostic gate. From then on, replay the graph
        // for every subsequent decode step. Per-step host work
        // (prepare_decode_inputs HtoDs, post-graph LM head, sampling,
        // rep-penalty) stays eager outside the graph.
        //
        // ## Known limitation: split-KV partition decision is frozen
        //
        // `decode_forward(step)` sets
        // `current_max_context_len = Some(step+1)`, and the attention
        // dispatch in `gemma4_layer_exec.rs` uses that to pick between
        // `decode.launch_split` (multi-partition) and `decode.launch`
        // (single CTA per seq/head). The captured graph records ONE
        // of those kernel calls. Replay then runs that kernel for
        // every later step regardless of how the partition count
        // would have evolved.
        //
        // This is a PERFORMANCE bug, not a correctness bug: the
        // non-split kernel handles long contexts correctly (it just
        // loops serially within one CTA), and the split kernel sizes
        // its workspace from `bucket_ctx = max_blocks_per_seq *
        // block_size` so it can run at any step's actual ctx.
        //
        // To minimise the perf hit, we delay capture until a
        // representative decode step — `RVLLM_DECODE_GRAPH_CAPTURE_AT`
        // (default 1) — and we ABORT capture eligibility when the
        // partition decision at the capture step would differ from
        // the decision at the GENERATION END (`prompt_len + max_new`).
        // In that case we silently fall through to eager
        // `run_one_token` for the whole generation and emit a one-
        // shot tracing::warn so the operator sees why the graph
        // didn't fire. A future cycle can capture multiple graphs
        // (one per partition-decision range) and replay the
        // appropriate one — out of scope here.
        // Hoisted from inside the decode loop. Reading `std::env::var`
        // is a syscall + alloc on every probe; doing it 4× per token
        // adds up over long completions and is unobservable to the
        // caller (env doesn't change mid-request).
        let cfg_guard_n: usize = std::env::var("RVLLM_REPETITION_GUARD_N")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(20);
        let cfg_cycle_k: usize = std::env::var("RVLLM_REPETITION_CYCLE_K")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
        let cfg_cycle_frac: f32 = std::env::var("RVLLM_REPETITION_CYCLE_MAX_FRAC")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0.5);
        let cfg_cycle_max_unique: usize = std::env::var("RVLLM_REPETITION_CYCLE_MAX_UNIQUE")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(5);

        let use_decode_graph = std::env::var("RVLLM_DECODE_GRAPH")
            .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
            .unwrap_or(false);
        // `RVLLM_DECODE_GRAPH_CAPTURE_AT` controls which decode_step
        // is used as the capture body. Default 1 — first step beyond
        // the eager warmup (decode_step==0 populates KV[prompt_len]
        // and runs the once-per-generate shadow_q diagnostic). The
        // floor of 1 is a hard requirement; values < 1 collapse to 1.
        let decode_graph_capture_at: usize = std::env::var("RVLLM_DECODE_GRAPH_CAPTURE_AT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1usize)
            .max(1);
        // Eligibility check: would the partition decision at capture
        // time still hold at the end of generation? If not, fall back
        // to eager; the captured-but-wrong-path replay would just be
        // a slow no-op compared to eager. Crucially, the partition
        // size queried here MUST match the value the attention
        // dispatch in `gemma4_layer_exec.rs` will use — both sites
        // call `effective_partition_size()` so a future change to
        // the default cannot silently desync the guard. (A previous
        // iteration hardcoded `256` here while the dispatch defaulted
        // to `1024`; the guard let plain-default requests through
        // and the captured graph froze a single-CTA branch that
        // flipped to split-KV mid-generation.)
        let partition_size_for_graph = effective_partition_size();
        // Codex37-3: only the split-KV path freezes a context-dependent
        // gridDim.z into the captured graph. With split-KV disabled
        // (env off, FP8/F16 KV) the parts-count drift can't poison
        // the reducer, so we can capture across partition boundaries.
        // Codex38-2: also gate on actual split-kernel availability.
        // Earlier the predicate trusted env+kv_dtype only; a kernels
        // tree without the split symbols (or running a head_dim
        // variant the build doesn't ship) used to reject graph
        // capture at partition boundaries even though decode never
        // actually launched split kernels.
        let split_decode_for_graph =
            rvllm_attention::PagedDecodeNvfp4Launcher::new(&self.sliding_attention);
        let global_split_decode_for_graph =
            rvllm_attention::PagedDecodeNvfp4Launcher::new(&self.global_attention);
        // Codex39-5: per-layer overrides (RVLLM_FP8_KV_LAYERS,
        // HYBRID_GLOBAL_FP8, HYBRID_SLIDING_FP8) can pull every layer
        // off NVFP4 even when the env-default is Nvfp4. If no layer
        // actually runs NVFP4 split, the parts-boundary check costs
        // graph capture for nothing. Walk all layers once at startup
        // and require at least one to land on Nvfp4 KV before
        // treating split-KV as active for the graph predicate.
        let any_layer_nvfp4 = (0..arch.num_hidden_layers).any(|li| {
            let lt = arch.layer_types[li];
            crate::gemma4_layer_exec::KvDtype::for_layer_index_or_env(lt, li, false)
                == crate::gemma4_layer_exec::KvDtype::Nvfp4
        });
        // Codex45-1: also mirror the runtime workspace gate at
        // gemma4_layer_exec.rs:1578 (`ws_need <= fa3_workspace_bytes`).
        // If the worst-case split scratch can't fit in the live FA3
        // workspace, the runtime path will never dispatch the split
        // kernel — the parts-boundary check would then refuse graph
        // capture for nothing. Compute the same `ws_need` formula the
        // dispatch site uses, against the upper-bound bucket context
        // (max_blocks_per_seq * block_size) and the worst-case
        // head_dim across layer types.
        let max_num_parts_for_graph: u64 = ((max_blocks_per_seq as u64) * (block_size as u64))
            .div_ceil(partition_size_for_graph.max(1) as u64)
            .max(1);
        let max_hd_for_graph: u64 = arch
            .head_dim_sliding
            .max(arch.head_dim_global) as u64;
        let split_slots: u64 = 1u64 // num_seqs in run_generate
            * (arch.num_attention_heads as u64)
            * max_num_parts_for_graph;
        // tmp_out f32 (codex cycle21 widening) + max_logits f32 + exp_sums f32.
        let split_ws_need: u64 = split_slots * max_hd_for_graph * 4
            + split_slots * 4 * 2;
        const FA3_WS_BYTES_FOR_GRAPH: u64 = 128 * 1024 * 1024;
        let workspace_can_fit_split = split_ws_need <= FA3_WS_BYTES_FOR_GRAPH;
        let split_kv_active_for_graph = parse_truthy_env("RVLLM_NVFP4_SPLIT_KV")
            .unwrap_or(true)
            && any_layer_nvfp4
            && workspace_can_fit_split
            && (split_decode_for_graph.has_split_kernels(arch.head_dim_sliding as u32, false)
                || global_split_decode_for_graph
                    .has_split_kernels(arch.head_dim_global as u32, false));
        let recapture_enabled = std::env::var("RVLLM_DECODE_GRAPH_RECAPTURE")
            .map(|s| !matches!(s.as_str(), "0" | "false" | "FALSE" | "no"))
            .unwrap_or(true);
        let decode_graph_eligible = use_decode_graph
            && decode_graph_eligible_for_generation(
                prompt_ids.len() as u32,
                max_new as u32,
                partition_size_for_graph,
                decode_graph_capture_at as u32,
                split_kv_active_for_graph,
                recapture_enabled,
            );
        if use_decode_graph && !decode_graph_eligible {
            tracing::warn!(
                prompt_len = prompt_ids.len(),
                max_new,
                partition_size = partition_size_for_graph,
                "RVLLM_DECODE_GRAPH=1 but partition decision would change \
                 mid-generation; falling back to eager decode for this \
                 request to avoid replaying a frozen split-KV branch"
            );
        }
        let use_decode_graph = decode_graph_eligible;
        let mut decode_graph: Option<rvllm_graph::CapturedGraph> = None;
        // Round-18 finding #4: track the parts-count the captured graph
        // was recorded against so we can re-capture across split-KV
        // partition boundaries instead of falling back to eager for the
        // entire generation. `None` here means "no graph captured yet";
        // `Some(parts)` means the live graph's frozen gridDim.z assumes
        // exactly `parts` partitions. When `parts_now != Some(parts)`
        // we drop the graph and recapture.
        let mut current_graph_parts: Option<u32> = None;
        for decode_step in 0..max_new - 1 {
            // Early-out on caller-side cancellation. Checked once per
            // step (cheap atomic load) so the worker thread releases
            // its monopoly on the GPU within ~one decode latency
            // (~270 ms on Gemma 4 31B / GB10) of a client timeout
            // rather than running to completion in the background.
            if let Some(c) = cancel {
                if c.load(std::sync::atomic::Ordering::Relaxed) {
                    tracing::info!(
                        decode_step,
                        max_new,
                        "run_generate cancelled by caller — returning partial output",
                    );
                    break;
                }
            }
            let tok_id = *output_ids.last().unwrap();
            let step = prompt_ids.len() + decode_step;
            if use_decode_graph && decode_step >= decode_graph_capture_at {
                // Post-warmup path: prepare inputs eagerly, then either
                // capture (first time / parts-count flipped) or replay
                // the captured graph.
                prepare_decode_inputs(tok_id, step)?;
                let ctx_now = (prompt_ids.len() + decode_step + 1) as u32;
                let parts_now = if split_kv_active_for_graph {
                    ctx_now
                        .div_ceil(partition_size_for_graph.max(1))
                        .max(1)
                } else {
                    1
                };
                let needs_recapture = decode_graph.is_none()
                    || (recapture_enabled
                        && current_graph_parts != Some(parts_now));
                if needs_recapture {
                    // Capture body: kernels are RECORDED into the graph,
                    // not executed. So we still need an eager run BEFORE
                    // capture to populate KV[step]; the capture body is
                    // a second invocation that records but does not run.
                    // Drop the previous (now-stale) graph before
                    // recording the new one so its CUgraphExec
                    // resources are released. `Option::take` returns
                    // the inner Some which drops at the end of this
                    // statement; explicit `_old` makes the intent
                    // (release before recapture) obvious.
                    let _old = decode_graph.take();
                    decode_forward(step)?;
                    self.stream.fence()?;
                    let g = rvllm_graph::CapturedGraph::capture(
                        1u32,
                        max_blocks_per_seq as u32,
                        rvllm_metadata::MetadataLayout::compute(
                            1u32, max_blocks_per_seq as u32).hash(),
                        rvllm_graph::GraphFingerprint([0u8; 32]),
                        stream,
                        || decode_forward(step),
                    )?;
                    self.stream.fence()?;
                    eprintln!("[decode-graph] captured at decode_step={} step={} parts={}",
                        decode_step, step, parts_now);
                    decode_graph = Some(g);
                    current_graph_parts = Some(parts_now);
                } else {
                    decode_graph.as_ref().unwrap().replay(stream)?;
                }
            } else {
                run_one_token(tok_id, step)?;
            }

            // === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
            // First-token dump: runs exactly once on decode_step == 0
            // when the shadow region is live. After this the latch is
            // set and every subsequent decode step is a no-op.
            if decode_step == 0
                && shadow_ptr != 0
                && !self.nvfp4_shadow_dumped.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                self.stream.fence()?;
                let dump_dir = std::env::var("RVLLM_NVFP4_SHADOW_DUMP_DIR")
                    .unwrap_or_else(|_| "/tmp/nvfp4_shadow".to_string());
                let _ = std::fs::create_dir_all(&dump_dir);
                let _ctx_now = (prompt_ids.len() + 1) as u32;
                let lset = shadow_set.as_ref().unwrap();
                let first_tok = output_ids[0];
                let bt_entries = max_blocks_per_seq as usize;
                // Host staging for block_tables / context_lens / slot_mapping.
                let mut bt_host = vec![0i32; bt_entries];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    bt_host.as_mut_ptr() as *mut _,
                    block_tables.device_ptr(),
                    (bt_entries * 4) as usize,
                );
                let mut ctx_host = [0i32; 1];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    ctx_host.as_mut_ptr() as *mut _,
                    context_lens.device_ptr(),
                    4,
                );
                let mut slot_host = [0i32; 1];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    slot_host.as_mut_ptr() as *mut _,
                    slot_mapping.device_ptr(),
                    4,
                );
                // Build per-layer metadata + dump bin files.
                let mut layer_meta_json = String::new();
                for &l in lset.iter() {
                    let l = l as usize;
                    if l >= arch.num_hidden_layers { continue; }
                    if l >= shadow_layer_offsets.len() { continue; }
                    if shadow_layer_offsets[l] == u64::MAX { continue; }
                    let lt = arch.layer_types[l];
                    let is_global = lt == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
                    let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                    let nkvh = arch.num_kv_heads_for_layer(l) as u32;
                    let hd = arch.head_dim_for_layer(l) as u32;
                    let layer_elems = 2u64 * (layer_blocks as u64) * (block_size as u64)
                        * (nkvh as u64) * (hd as u64);
                    // Shadow region: f16, layer_elems bytes for K then layer_elems bytes for V.
                    let shadow_base = shadow_ptr + shadow_layer_offsets[l];
                    let shadow_half_bytes = layer_elems; // f16 half-size per K or V
                    let mut k_shadow_host = vec![0u8; shadow_half_bytes as usize];
                    let mut v_shadow_host = vec![0u8; shadow_half_bytes as usize];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        k_shadow_host.as_mut_ptr() as *mut _,
                        shadow_base,
                        shadow_half_bytes as usize,
                    );
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        v_shadow_host.as_mut_ptr() as *mut _,
                        shadow_base + shadow_half_bytes,
                        shadow_half_bytes as usize,
                    );
                    let _ = std::fs::write(
                        format!("{}/layer_{}_k_shadow.bin", dump_dir, l),
                        &k_shadow_host,
                    );
                    let _ = std::fs::write(
                        format!("{}/layer_{}_v_shadow.bin", dump_dir, l),
                        &v_shadow_host,
                    );
                    // Primary NVFP4 K/V (packed bytes). The total NVFP4
                    // allocation per layer is `layer_elems / 2` bytes
                    // (because `layer_elems = 2 * X` already counts K+V
                    // and NVFP4 packs 2 elems/byte). Each of K and V is
                    // therefore `layer_elems / 4` bytes within the
                    // layer, matching `bytes_per_half_kv = layer_kv_elems / 4`
                    // used by the rope launcher's `v_cache` offset.
                    // An earlier revision of this dump used
                    // `layer_elems / 2` for the per-side size, which
                    // (a) read past the layer's allocation for V
                    // and (b) made the dumped V file actually contain
                    // the NEXT layer's K data — producing apparent
                    // 100%+ V rel_err in the analyzer when in fact V
                    // was never read from the right offset.
                    let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[l];
                    let primary_half_bytes = layer_elems / 4; // K-or-V bytes
                    let mut k_host = vec![0u8; primary_half_bytes as usize];
                    let mut v_host = vec![0u8; primary_half_bytes as usize];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        k_host.as_mut_ptr() as *mut _,
                        layer_kv_base,
                        primary_half_bytes as usize,
                    );
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        v_host.as_mut_ptr() as *mut _,
                        layer_kv_base + primary_half_bytes,
                        primary_half_bytes as usize,
                    );
                    let _ = std::fs::write(format!("{}/layer_{}_k.bin", dump_dir, l), &k_host);
                    let _ = std::fs::write(format!("{}/layer_{}_v.bin", dump_dir, l), &v_host);
                    // NVFP4 scale region: E4M3, layer_elems/16 bytes total; first
                    // half for K, second half for V.
                    let layer_kv_scale_base =
                        kv_scale_cache.device_ptr() + kv_scale_layer_offsets[l];
                    let scale_half_bytes = layer_elems / 32; // each of K,V = /32
                    let mut k_scale_host = vec![0u8; scale_half_bytes as usize];
                    let mut v_scale_host = vec![0u8; scale_half_bytes as usize];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        k_scale_host.as_mut_ptr() as *mut _,
                        layer_kv_scale_base,
                        scale_half_bytes as usize,
                    );
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        v_scale_host.as_mut_ptr() as *mut _,
                        layer_kv_scale_base + scale_half_bytes,
                        scale_half_bytes as usize,
                    );
                    let _ = std::fs::write(
                        format!("{}/layer_{}_k_scale.bin", dump_dir, l),
                        &k_scale_host,
                    );
                    let _ = std::fs::write(
                        format!("{}/layer_{}_v_scale.bin", dump_dir, l),
                        &v_scale_host,
                    );
                    // Per-layer Q dump (f16, post-RoPE). Snapshot written by
                    // `rope_f16kv_shadow` → memcpy hook in layer_exec.rs on
                    // decode step 0 into a dedicated per-layer slot of size
                    // `shadow_q_per_layer_bytes = num_attention_heads *
                    // max_head_dim * 2`. Tail may be zero when this layer's
                    // head_dim < max_head_dim; Python analyzer truncates
                    // using `head_dim` from meta.json.
                    let pos_in_set = lset.iter().position(|&li| li as usize == l);
                    if let Some(pi) = pos_in_set {
                        if shadow_q_ptr != 0 {
                            let q_slot_base =
                                shadow_q_ptr + (pi as u64) * shadow_q_per_layer_bytes;
                            let mut q_host = vec![0u8; shadow_q_per_layer_bytes as usize];
                            cudarc::driver::sys::cuMemcpyDtoH_v2(
                                q_host.as_mut_ptr() as *mut _,
                                q_slot_base,
                                shadow_q_per_layer_bytes as usize,
                            );
                            let _ = std::fs::write(
                                format!("{}/layer_{}_q.bin", dump_dir, l),
                                &q_host,
                            );
                        }
                    }
                    if !layer_meta_json.is_empty() {
                        layer_meta_json.push_str(",\n");
                    }
                    layer_meta_json.push_str(&format!(
                        "    {{\"layer\": {}, \"layer_type\": \"{:?}\", \"head_dim\": {}, \"num_kv_heads\": {}, \"num_blocks\": {}}}",
                        l, lt, hd, nkvh, layer_blocks,
                    ));
                }
                // Keep legacy single-layer Q dump (last executed layer, FP8)
                // for backward compat with older analyzer runs; per-layer
                // f16 Q files (layer_{L}_q.bin) are the canonical source.
                let q_bytes = (arch.num_attention_heads as u64) * (arch.max_head_dim() as u64);
                let mut q_host = vec![0u8; q_bytes as usize];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    q_host.as_mut_ptr() as *mut _,
                    q_fp8.device_ptr(),
                    q_bytes as usize,
                );
                let _ = std::fs::write(format!("{}/q_last_layer.bin", dump_dir), &q_host);
                // meta.json
                let max_head_dim = arch.max_head_dim();
                let meta_json = format!(
                    "{{\n  \"prompt_len\": {},\n  \"num_layers\": {},\n  \"block_size\": {},\n  \"num_heads\": {},\n  \"max_head_dim\": {},\n  \"q_dtype\": \"f16\",\n  \"q_per_layer_bytes\": {},\n  \"context_len\": {},\n  \"slot_mapping\": {},\n  \"first_token_id\": {},\n  \"shadow_layer_indices\": {:?},\n  \"block_table\": {:?},\n  \"layers\": [\n{}\n  ]\n}}\n",
                    prompt_ids.len(),
                    arch.num_hidden_layers,
                    block_size,
                    arch.num_attention_heads,
                    max_head_dim,
                    shadow_q_per_layer_bytes,
                    ctx_host[0],
                    slot_host[0],
                    first_tok,
                    lset,
                    bt_host,
                    layer_meta_json,
                );
                let _ = std::fs::write(format!("{}/meta.json", dump_dir), &meta_json);
                eprintln!(
                    "[nvfp4-shadow] dumped {} instrumented layers to {} (ctx={}, first_tok={})",
                    lset.len(), dump_dir, ctx_host[0], first_tok,
                );
            }
            // === END NVFP4 SHADOW DIAGNOSTIC ===

            // Cycle 54 Stage 1: same bf16 LM-head dispatch as the post-prefill site.
            let lm_head_bf16 = bf16_residual_enabled();
            let lm_head_norm_kernel = if lm_head_bf16 {
                kernels.rmsnorm_inplace_bf16
            } else {
                kernels.fused_rmsnorm
            };
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: arch.rms_norm_eps,
            }.launch(lm_head_norm_kernel, residual_ptr, self.model.final_norm.offset_bytes, stream)?;
            if lm_head_bf16 {
                rvllm_fused::gemma4_launcher::Bf16ToF16SatLaunch { n: hidden }
                    .launch(kernels.bf16_to_f16_sat, residual_ptr, residual_ptr, stream)?;
            }
            self.cublaslt.f16_gemm_f32(residual_ptr, self.model.lm_head_f16.offset_bytes,
                logits_f32.device_ptr(), 1, vocab as i32, hidden as i32, stream)?;
            // Codex41-1: apply Gemma final logit_softcap on EVERY
            // decode step. Codex40-2 only patched the prefill-site
            // softcap; the decode loop runs its own LM-head GEMM and
            // was sampling from un-capped logits for tokens 2..N,
            // diverging from PPL/bench and from the trained-output
            // distribution.
            if arch.logit_softcap > 0.0 {
                rvllm_fused::gemma4_launcher::LogitSoftcapLaunch {
                    num_tokens: 1,
                    vocab,
                    cap: arch.logit_softcap,
                }
                .launch(self.fused.fn_softcap_f32, logits_f32.device_ptr(), stream)?;
            }
            // Codex41-4: tool_call_open_bias also belongs on every
            // decode step. The first-token site (line ~4196) had it;
            // later tokens were missing the bias, so the sampler's
            // open-tag preference was inconsistent across the
            // generation. Mirror the bias write here.
            if tool_call_bias != 0.0 {
                let mut logit_48: f32 = 0.0;
                cuda_check!(cudarc::driver::sys::cuMemcpyDtoH_v2(
                    &mut logit_48 as *mut _ as *mut _,
                    logits_f32.device_ptr() + 48 * 4,
                    4,
                ), "tool_bias_logit48_dtoh_decode", stream);
                let new_logit = logit_48 + tool_call_bias;
                cuda_check!(cudarc::driver::sys::cuMemcpyHtoD_v2(
                    logits_f32.device_ptr() + 48 * 4,
                    &new_logit as *const _ as *const _,
                    4,
                ), "tool_bias_logit48_htod_decode", stream);
            }
            // === REPETITION PENALTY (decode-loop site) — Codex41-3 ===
            // Earlier this synchronised the stream and bounced the
            // entire 262k-vocab logits through host memory every step
            // (~7ms/token). Now we collect the recent-id set on the
            // host, HtoD-copy just that small list (≤ rep_window i32),
            // and invoke `apply_repetition_penalty_f32_kernel` to
            // mutate the f32 logits in place — no DtoH, no fence.
            if rep_active && !output_ids.is_empty() {
                let start_idx = output_ids.len().saturating_sub(rep_window);
                let mut counts: std::collections::HashMap<u32, u32> =
                    std::collections::HashMap::new();
                for &id in &output_ids[start_idx..] {
                    *counts.entry(id).or_insert(0) += 1;
                }
                let mut recent: std::collections::HashSet<u32> = counts
                    .iter()
                    .filter(|(_, &c)| c >= rep_min_count)
                    .map(|(&id, _)| id)
                    .collect();
                // Don't penalize EOS / stop tokens — let them fire when
                // the model wants to terminate.
                for sid in eos_ids { recent.remove(sid); }
                if !recent.is_empty() {
                    let cap = 1024usize; // device buffer capacity
                    let ids_vec: Vec<i32> = recent.iter()
                        .take(cap)
                        .map(|&id| id as i32)
                        .collect();
                    cuda_check!(cudarc::driver::sys::cuMemcpyHtoD_v2(
                        rep_ids_dev.device_ptr(),
                        ids_vec.as_ptr() as *const _,
                        ids_vec.len() * 4,
                    ), "rep_penalty_ids_htod", stream);
                    rvllm_fused::gemma4_launcher::ApplyRepetitionPenaltyLaunch {
                        num_ids: ids_vec.len() as u32,
                        vocab,
                        penalty: rep_penalty,
                    }.launch(
                        self.fused.fn_apply_repetition_penalty_f32,
                        logits_f32.device_ptr(),
                        rep_ids_dev.device_ptr(),
                        stream,
                    )?;
                }
            }
            // === END REPETITION PENALTY ===
            if sampling_temp > 0.0 {
                self.stream.fence()?;
                host_tok[0] = host_sample_token(
                    logits_f32.device_ptr(), vocab, sampling_temp, sampling_top_p, sampling_top_k,
                    &mut next_rand_f32,
                )? as i32;
            } else {
                rvllm_fused::ArgmaxLaunch { num_tokens: 1, vocab }
                    .launch(fn_argmax, logits_f32.device_ptr(), sampled.device_ptr(), stream)?;
                self.stream.fence()?;
                // See twin site at the prefill argmax above: a silent
                // DtoH failure left host_tok[0] at its prior value
                // and the loop emitted that token as the next one,
                // masking CUDA faults as model hallucinations.
                cuda_check!(
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        host_tok.as_mut_ptr() as *mut _,
                        sampled.device_ptr(),
                        4,
                    ),
                    "argmax_sample_token_dtoh_decode_loop",
                    stream
                );
            }
            // Commit 26: spec-decode decode-loop logits capture
            // (rows 1..). Row index = decode_step + 1 (row 0 was the
            // prefill argmax site above). Gated identically.
            if self.spec_logits_capture_active.load(std::sync::atomic::Ordering::Relaxed) {
                self.stream.fence()?;
                let row_idx = (decode_step + 1) as usize;
                let v = vocab as usize;
                let row_off = row_idx * v;
                let mut buf = self.spec_decode_step_logits.lock().unwrap();
                if buf.len() >= row_off + v {
                    let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                        buf[row_off..].as_mut_ptr() as *mut _,
                        logits_f32.device_ptr(),
                        v * 4,
                    );
                }
            }
            let next_id = host_tok[0] as u32;
            // Round-20 finding #2: EOS check BEFORE push + callback so
            // the special token never leaks to the SSE consumer or
            // bumps `completion_tokens`.
            if eos_ids.contains(&next_id) { break; }
            output_ids.push(next_id);
            // True-streaming hook: emit each token to the worker's
            // event channel. A `false` return means the consumer is
            // gone (closed channel) — treat as cancel and stop
            // decoding.
            if let Some(cb) = on_token.as_mut() {
                if !cb(next_id) {
                    break;
                }
            }
            // Cycle 33 fix (codex bug #5): tool-call close `<tool_call|>`
            // (token 49) was not a generation stop. After a valid tool
            // call closed, the model kept emitting hallucinated prose
            // (`<|tool_response>` text, "The weather in Zurich is 12°C..."
            // narration, multiple redundant tool calls). With this guard,
            // once token 48 (`<|tool_call>`) has been seen in the output
            // and we then emit token 49, treat it as a structural stop.
            // Standalone token 49 without prior 48 is not a stop (the
            // tag could appear in fragmented form during partial
            // streaming reuse — preserve previous behavior there).
            if next_id == 49u32 && output_ids.iter().rev().any(|&t| t == 48u32) {
                break;
            }

            // Repetition guard. When a low-precision KV path (e.g.
            // pure NVFP4) lands in a near-tied logit state — typically
            // inside tool-call markup or an unfamiliar prompt
            // continuation — the model can lock into a single-token
            // attractor and emit the same token thousands of times,
            // wasting GPU and producing empty visible content (when
            // the locked token sits inside markup that
            // `strip_tool_markup` removes).
            //
            // Bound cost via `RVLLM_REPETITION_GUARD_N` (default 20,
            // set 0 to disable). If the last N decoded tokens are
            // all the same id, abort cleanly: callers see a normal
            // stream end with whatever was produced so far. The
            // guard only fires after at least N decode steps; short
            // legitimate completions (e.g. classifier "REPLY")
            // never trigger it.
            let guard_n = cfg_guard_n;
            if guard_n >= 2 && output_ids.len() >= guard_n {
                let tail = &output_ids[output_ids.len() - guard_n..];
                if tail.iter().all(|&id| id == tail[0]) {
                    eprintln!(
                        "[repetition-guard] same token {} repeated {} times — \
                         aborting decode at step {}",
                        tail[0], guard_n, decode_step + 1
                    );
                    break;
                }
            }
            // Cycle-aware guard: catches multi-token attractors where
            // a small set of tokens cycles (e.g. Korean lock observed
            // 2026-04-25 with token 237372='서' alternating with
            // 7246='으로', 237490='도', etc. — no token reaches
            // `guard_n` consecutive but a 32-token window contains
            // only 5-8 distinct ids, with one dominating ~50%+).
            //
            // Triggers when EITHER:
            //   (a) some token covers >= MAX_FRAC of last K decoded
            //       tokens (default K=32, MAX_FRAC=0.5 → 16/32);
            //   (b) the last K tokens contain <= MAX_UNIQUE distinct
            //       ids (default 5).
            //
            // RVLLM_REPETITION_CYCLE_K          window (default 32, 0=disabled)
            // RVLLM_REPETITION_CYCLE_MAX_FRAC   ratio (default 0.5)
            // RVLLM_REPETITION_CYCLE_MAX_UNIQUE distinct count (default 5)
            let cycle_k = cfg_cycle_k;
            if cycle_k >= 8 && output_ids.len() >= cycle_k {
                let cycle_frac = cfg_cycle_frac;
                let cycle_max_unique = cfg_cycle_max_unique;
                let win = &output_ids[output_ids.len() - cycle_k..];
                let mut counts: std::collections::HashMap<u32, u32> =
                    std::collections::HashMap::new();
                for &id in win { *counts.entry(id).or_insert(0) += 1; }
                let unique_count = counts.len();
                let max_count = counts.values().copied().max().unwrap_or(0);
                let max_frac = (max_count as f32) / (cycle_k as f32);
                if unique_count <= cycle_max_unique || max_frac >= cycle_frac {
                    let dom_id = counts.iter().max_by_key(|(_, &c)| c)
                        .map(|(id, _)| *id).unwrap_or(0);
                    eprintln!(
                        "[repetition-guard] cycle detected — last {} tokens \
                         have {} unique ids (max id {} = {:.0}%) — aborting \
                         decode at step {}",
                        cycle_k, unique_count, dom_id,
                        max_frac * 100.0, decode_step + 1
                    );
                    break;
                }
            }
        }

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let decode_ms = total_ms - prefill_ms;

        tracing::info!(
            tokens = output_ids.len(),
            decode_ms = decode_ms,
            tok_per_s = output_ids.len() as f64 / (decode_ms / 1000.0),
            "generate complete"
        );

        // Update the prefix cache with this request's prompt so the
        // next request can benefit from a cache hit. We cache ONLY
        // the prompt tokens (not the generated output) — the
        // generated tokens' KV entries are in slots [prompt_len..]
        // and are indeed valid, but zeroclaw typically includes
        // prior assistant responses in the NEXT prompt's history
        // anyway, so persisting generated-token KV here adds
        // complexity without extra benefit.
        // Codex review priority 1 (commit 49 — Phase B-2):
        // verify_batched_from_state arms this flag so spec-internal
        // state doesn't pollute the cross-request prefix cache.
        // One-shot semantics: swap(false) on consume.
        let skip_publish = self
            .skip_prefix_cache_publish
            .swap(false, std::sync::atomic::Ordering::AcqRel);
        if use_prefix_cache && !skip_publish {
            if let Ok(mut guard) = self.prefix_cache.lock() {
                if let Some(pc) = guard.as_mut() {
                    pc.last_tokens.clear();
                    pc.last_tokens.extend_from_slice(prompt_ids);
                    // Cap committed prefix at the last full chunk
                    // boundary. Slots written by a trailing partial
                    // chunk are unsafe to reuse — see
                    // PrefixCacheState::committed_prefix_len doc.
                    let chunk_size: u32 = std::env::var("RVLLM_PREFILL_CHUNK_SIZE")
                        .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
                    let batch_prefill =
                        parse_truthy_env("RVLLM_BATCH_PREFILL").unwrap_or(false);
                    let prompt_len_u32 = prompt_ids.len() as u32;
                    pc.committed_prefix_len = if batch_prefill && chunk_size == 0 {
                        // One-shot batch prefill is batch-shape dependent:
                        // slots written for a 3k-token request are not
                        // guaranteed equivalent to the same positions inside
                        // a later 15k-token request. Fixed chunks give stable
                        // chunk shapes; without them, keep the cache metadata
                        // for diagnostics but do not reuse it.
                        0
                    } else if batch_prefill && chunk_size > 0 {
                        // Codex26-1: chunked batch-prefill commits ONLY full
                        // chunks. A short prompt (prompt_len < chunk_size)
                        // ran as a single sub-chunk under batch shape
                        // (1, prompt_len), which is NOT the same shape a
                        // later longer request's first chunk sees — sm121
                        // small-batch GEMM paths aren't bit-identical, so
                        // reusing those KV bytes can poison the next
                        // request. Strict floor commits 0 in that case.
                        (prompt_len_u32 / chunk_size) * chunk_size
                    } else {
                        // Per-token decode prefill is shape-stable, so the
                        // whole prompt is safe to reuse.
                        prompt_len_u32
                    };
                    // Refresh provenance so subsequent provenance
                    // checks compare against the env that ACTUALLY
                    // wrote this KV state.
                    pc.provenance = PrefixProvenance::from_env();
                }
            }
        }
        Ok(output_ids)
    }

    /// E4B Per-Layer Embeddings precompute (stage 3b2).
    ///
    /// Computes `per_layer_inputs[T, num_layers * ple_dim]` from
    /// `token_ids` and the post-gather embeddings, once per
    /// request/chunk. Returns `(base_ptr, row_stride_elems)`:
    ///   - `base_ptr` points at the precomputed `[T, num_layers, ple_dim]`
    ///     row-major f16 buffer.
    ///   - `row_stride_elems` = `num_layers * ple_dim`. Per-layer slice
    ///     for layer L lives at `base_ptr + L * ple_dim * 2`.
    ///
    /// Returns `(0, 0)` when PLE is inactive (31B/AWQ → caller skips
    /// the per-layer injection block).
    ///
    /// Math (HF Gemma4TextModel.{get,project}_per_layer_inputs):
    ///   lookup  = embed_tokens_per_layer[token_ids]   # × sqrt(D) × scale_in baked
    ///   context = inputs_embeds @ plmp.T              # raw GEMM (scale eaten by norm)
    ///   context = f32→f16
    ///   context = rmsnorm(context, γ)                 # γ × scale_in baked
    ///   per_layer_inputs = lookup + context
    ///
    /// scale_in = `arch.per_layer_input_scale` (= 1/√2 default).
    /// Both load-time scale bakes happen in gemma4_load.rs PLE block.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn precompute_ple(
        &self,
        inputs_embeds_ptr: u64,
        token_ids_ptr: u64,
        num_tokens: u32,
        fn_embed: rvllm_kernels::KernelFn,
        kernels: &Gemma4LayerKernels,
        stream: u64,
    ) -> Result<(u64, u32)> {
        let Some(ple) = self.model.ple.as_ref() else {
            return Ok((0, 0));
        };
        let Some(ple_dim) = self.arch.hidden_size_per_layer_input else {
            return Ok((0, 0));
        };
        let num_layers = self.arch.num_hidden_layers;
        let total_dim = (num_layers * ple_dim) as u32; // 10752 on E4B
        let hidden = self.arch.hidden_size as u32;
        let vocab = self.arch.vocab_size as u32;

        // Diff-against-HF dump hook. When RVLLM_E4B_PLE_DUMP_DIR is set,
        // each precompute step writes its f16 output to the same
        // filename layout that v3/tools/gemma4_e4b_ple_hf_dump.py
        // produces, for row-cosine comparison via
        // v3/tools/cmp_e4b_ple.py.
        let dump_dir = std::env::var("RVLLM_E4B_PLE_DUMP_DIR").ok();
        let dump = |name: &str, ptr: u64, n_bytes: usize| -> Result<()> {
            if let Some(ref dir) = dump_dir {
                cudarc::driver::sys::cuStreamSynchronize(stream as cudarc::driver::sys::CUstream);
                let mut buf = vec![0u8; n_bytes];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut _,
                    ptr,
                    n_bytes,
                );
                let path = std::path::Path::new(dir).join(name);
                std::fs::create_dir_all(dir).ok();
                std::fs::write(&path, &buf).map_err(|e| rvllm_core::RvllmError::cuda(
                    "ple-dump: write failed",
                    rvllm_core::CudaErrorKind::Other,
                    rvllm_core::CudaCtx::setup(),
                ))?;
                eprintln!("[ple-dump] {} ({} bytes)", path.display(), n_bytes);
            }
            Ok(())
        };

        // Dump inputs_embeds for HF parity.
        dump(
            "e4b_inputs_embeds.bin",
            inputs_embeds_ptr,
            (num_tokens as usize) * (hidden as usize) * 2,
        )?;
        // One-shot dump of cos/sin RoPE tables (used by HF parity
        // for layer-by-layer RoPE replay). Tables are model-global
        // — write once when PLE precompute first fires.
        let max_pos = self.arch.max_position_embeddings;
        // Buffer layout: [max_pos, rotary_dim/2] f16 (no cat-doubling).
        let half_sl = self.arch.head_dim_sliding / 2; // sliding = full rotation
        let half_gl = self.arch.rotary_dim_global() / 2; // partial 0.25 → 64
        dump(
            "e4b_rope_cos_sliding.bin",
            self.model.rope_cos_sliding.offset_bytes,
            max_pos * half_sl * 2,
        )?;
        dump(
            "e4b_rope_sin_sliding.bin",
            self.model.rope_sin_sliding.offset_bytes,
            max_pos * half_sl * 2,
        )?;
        dump(
            "e4b_rope_cos_global.bin",
            self.model.rope_cos_global.offset_bytes,
            max_pos * half_gl * 2,
        )?;
        dump(
            "e4b_rope_sin_global.bin",
            self.model.rope_sin_global.offset_bytes,
            max_pos * half_gl * 2,
        )?;

        // Allocate scratch. Both regions are auto-restored at request
        // end via the existing arena checkpoint mechanism.
        let bytes_f16 = (num_tokens as usize) * (total_dim as usize) * 2;
        let bytes_f32 = (num_tokens as usize) * (total_dim as usize) * 4;
        let per_layer_inputs = self
            .arena
            .region("ple_per_layer_inputs", bytes_f16, 16)?;
        let ctx_f32 = self.arena.region("ple_ctx_f32", bytes_f32, 16)?;

        // Step 1: lookup the per-layer token-identity embedding.
        // Scale (× √D × scale_in) is already baked in at load time;
        // EmbeddingGather just copies the row.
        rvllm_fused::EmbeddingGatherLaunch {
            num_tokens,
            hidden: total_dim,
            vocab,
        }
        .launch(
            fn_embed,
            per_layer_inputs.device_ptr(),
            ple.embed_tokens_per_layer.offset_bytes,
            token_ids_ptr,
            stream,
        )?;
        dump(
            "e4b_ple_lookup.bin",
            per_layer_inputs.device_ptr(),
            (num_tokens as usize) * (total_dim as usize) * 2,
        )?;

        // Step 2: context-aware projection. inputs_embeds [T, hidden]
        // × per_layer_model_projection.T [hidden, total_dim]
        // → ctx_f32 [T, total_dim] f32. RMSNorm later scrubs the
        // `per_layer_model_projection_scale` so we skip it.
        self.cublaslt.f16_gemm_f32(
            inputs_embeds_ptr,
            ple.per_layer_model_projection.offset_bytes,
            ctx_f32.device_ptr(),
            num_tokens as i32,
            total_dim as i32,
            hidden as i32,
            stream,
        )?;

        // Step 3: narrow f32 → f16 in-place onto the front half of
        // ctx_f32. After this, `ctx_f32.device_ptr()` is the f16
        // buffer of `num_tokens * total_dim * 2` bytes.
        let n_total = num_tokens * total_dim;
        rvllm_fused::gemma4_launcher::Bf16ToF16SatLaunch { n: n_total }.launch(
            kernels.f32_to_f16_sat,
            ctx_f32.device_ptr(),
            ctx_f32.device_ptr(),
            stream,
        )?;
        dump(
            "e4b_ple_context_pre_norm.bin",
            ctx_f32.device_ptr(),
            (n_total as usize) * 2,
        )?;

        // Step 4: RMSNorm over the ple_dim axis with the SHARED γ
        // (shape [ple_dim] broadcasts across all `T * num_layers`
        // slices). γ has `× per_layer_input_scale` baked in.
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: num_tokens * (num_layers as u32),
            hidden: ple_dim as u32,
            eps: self.arch.rms_norm_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            ctx_f32.device_ptr(),
            ple.per_layer_projection_norm.offset_bytes,
            stream,
        )?;
        dump(
            "e4b_ple_context_post_norm.bin",
            ctx_f32.device_ptr(),
            (n_total as usize) * 2,
        )?;

        // Step 5: per_layer_inputs += ctx_f16 (elementwise on the
        // packed [T * num_layers * ple_dim] flat buffer).
        // Use the vector_add_f16 kernel directly.
        let n = n_total as i32;
        let mut dst = per_layer_inputs.device_ptr();
        let mut src = ctx_f32.device_ptr();
        let mut nn = n;
        let args = [
            (&mut dst) as *mut u64 as *mut core::ffi::c_void,
            (&mut src) as *mut u64 as *mut core::ffi::c_void,
            (&mut nn) as *mut i32 as *mut core::ffi::c_void,
        ];
        let block: u32 = 256;
        let grid = ((n as u32 + block - 1) / block, 1u32, 1u32);
        let rc = cudarc::driver::sys::cuLaunchKernel(
            self.fused.fn_vector_add.raw() as cudarc::driver::sys::CUfunction,
            grid.0,
            grid.1,
            grid.2,
            block,
            1,
            1,
            0,
            stream as cudarc::driver::sys::CUstream,
            args.as_ptr() as *mut *mut core::ffi::c_void,
            core::ptr::null_mut(),
        );
        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::cuda(
                "ple: vector_add launch failed",
                rvllm_core::CudaErrorKind::LaunchFailed,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        dump(
            "e4b_per_layer_inputs.bin",
            per_layer_inputs.device_ptr(),
            (n_total as usize) * 2,
        )?;

        Ok((per_layer_inputs.device_ptr(), total_dim))
    }

    pub fn layer_kernels(&self) -> Result<Gemma4LayerKernels> {
        // (helper defined just above) — see assert_rope_kernels_match
        // NVFP4 RoPE kernel handle — `None` on branches without the
        // NVFP4 PTX built into $KERNELS_DIR. Lives on Fa2PtxKernels
        // so the module lifetime outlives the fn handle; extracting
        // via `match` here instead of a helper method to avoid
        // enlarging the AttentionBackend API for a single field.
        //
        // Sliding and global attention each own their own
        // Fa2PtxKernels; both currently load the SAME PTX file for
        // RoPE (`fused_rope_partial_nvfp4kv.ptx`) so the function
        // pointers are identical. We pull from `sliding_attention`
        // to feed `Gemma4LayerKernels` (struct is per-bringup, not
        // per-layer) and assert symmetry at runtime so any future
        // refactor that loads different RoPE modules for global vs
        // sliding fails LOUDLY here instead of silently feeding
        // sliding's RoPE into global layers (or vice versa).
        #[cfg(feature = "cuda")]
        let fused_rope_partial_nvfp4kv = {
            let sliding = match &self.sliding_attention {
                rvllm_attention::AttentionBackend::Fa2Ptx(fa2) => fa2.fn_rope_nvfp4kv,
                _ => None,
            };
            let global = match &self.global_attention {
                rvllm_attention::AttentionBackend::Fa2Ptx(fa2) => fa2.fn_rope_nvfp4kv,
                _ => None,
            };
            assert_rope_kernels_match(
                "fn_rope_nvfp4kv",
                sliding,
                global,
            )?;
            sliding
        };
        #[cfg(feature = "cuda")]
        let fused_rope_partial_nvfp4kv_bf16in = {
            let sliding = match &self.sliding_attention {
                rvllm_attention::AttentionBackend::Fa2Ptx(fa2) => fa2.fn_rope_nvfp4kv_bf16in,
                _ => None,
            };
            let global = match &self.global_attention {
                rvllm_attention::AttentionBackend::Fa2Ptx(fa2) => fa2.fn_rope_nvfp4kv_bf16in,
                _ => None,
            };
            assert_rope_kernels_match(
                "fn_rope_nvfp4kv_bf16in",
                sliding,
                global,
            )?;
            sliding
        };
        #[cfg(not(feature = "cuda"))]
        let fused_rope_partial_nvfp4kv = None;
        #[cfg(not(feature = "cuda"))]
        let fused_rope_partial_nvfp4kv_bf16in = None;

        Ok(Gemma4LayerKernels {
            fused_rmsnorm: self.fused.fn_rmsnorm,
            fused_rmsnorm_fp8_quant: self.fused.fn_rmsnorm_fp8_quant,
            fused_qk_rmsnorm: self.fused.fn_qk_rmsnorm,
            fused_qk_rmsnorm_bf16: self.fused.fn_qk_rmsnorm_bf16,
            fused_rope_partial_fp8kv: self.fused.fn_rope_partial_fp8kv,
            fused_rope_partial_fp8kv_bf16in: self.fused.fn_rope_partial_fp8kv_bf16in,
            fused_rope_partial_nvfp4kv,
            fused_rope_partial_nvfp4kv_bf16in,
            fused_gelu_mul: self.fused.fn_gelu_mul,
            quantize_fp8_per_token: self.fused.fn_quantize,
            residual_scale_f16: self.fused.fn_residual_scale,
            vnorm_f16: self.fused.fn_vnorm,
            vector_add_f16: self.fused.fn_vector_add,
            bf16_to_f16_sat: self.fused.fn_bf16_to_f16_sat,
            rmsnorm_inplace_bf16: self.fused.fn_rmsnorm_inplace_bf16,
            vector_add_bf16_to_f16: self.fused.fn_vector_add_bf16_to_f16,
            f32_to_bf16: self.fused.fn_f32_to_bf16,
            f32_to_f16_sat: self.fused.fn_f32_to_f16_sat,
            scale_cols_f32: self.fused.fn_scale_cols_f32,
            scale_rows_f32_ratio: self.fused.fn_scale_rows_f32_ratio,
            fused_gelu_mul_f16: self.fused.fn_fused_gelu_mul_f16,
            fused_gelu_mul_bf16: self.fused.fn_fused_gelu_mul_bf16,
            gelu_tanh_mul_dual_f16: self.fused.fn_gelu_tanh_mul_dual_f16,
            fused_rope_partial_f16kv: self.fused.fn_fused_rope_partial_f16kv,
            fused_norm_add_residual: self.fused.fn_fused_norm_add_residual,
            fused_norm_add_residual_f16: self.fused.fn_fused_norm_add_residual_f16,
            fused_norm_add_residual_f16in: self.fused.fn_fused_norm_add_residual_f16in,
            // Cycle 54 Stage 1: BF16 residual chain handles.
            f16_to_bf16: self.fused.fn_f16_to_bf16,
            fused_norm_add_residual_bf16: self.fused.fn_fused_norm_add_residual_bf16,
            fused_norm_add_residual_bf16_f16in: self.fused.fn_fused_norm_add_residual_bf16_f16in,
            fused_norm_add_residual_bf16_bf16in: self.fused.fn_fused_norm_add_residual_bf16_bf16in,
            fused_rmsnorm_fp8_quant_bf16in: self.fused.fn_fused_rmsnorm_fp8_quant_bf16in,
            fused_qkv_rmsnorm: self.fused.fn_fused_qkv_rmsnorm,
            fused_qkv_rmsnorm_bf16: self.fused.fn_fused_qkv_rmsnorm_bf16,
            scale_cols_f16: self.fused.fn_scale_cols_f16,
            fp8_gemv_wpr_native_f16in: self.fused.fn_fp8_gemv_wpr_native_f16in,
            fp8_gemv_wpr_native_bf16in: self.fused.fn_fp8_gemv_wpr_native_bf16in,
            hadamard_unrotate_f16: self.fused.fn_hadamard_unrotate_f16,
            awq_int4_gemv_f16: self.fused.fn_awq_int4_gemv_f16,
            awq_int4_gemm_sm120_wmma: self.fused.fn_awq_int4_gemm_sm120_wmma,
        })
    }

    /// B6b: Gemma 4 E4B audio subsample stage forward.
    ///
    /// Runs the host-side mel-spectrogram extractor on the 16 kHz
    /// mono f32 samples, uploads the result as f16, then walks the
    /// two-stage Conv2d subsampler (im2col + cuBLASLt GEMM + cast +
    /// fused LayerNorm+ReLU per stage), the channel-last transpose,
    /// and the input_proj_linear matmul. Output is a device-resident
    /// f16 buffer of shape `[num_soft_tokens_subsampled, audio_hidden=1024]`.
    ///
    /// **This is the subsample stage only.** B6c..d add the 12
    /// encoder blocks (FFN + chunked attention + LConv1D + FFN +
    /// norms) and the output_proj. Callers that need the full
    /// pre-embed_audio output should not consume this method's
    /// result directly — it isn't shape-compatible with
    /// `model.embed_audio.embedding_projection`.
    ///
    /// Returns the device pointer + dims + intended soft-token
    /// count. The arena buffers it allocates are auto-restored at
    /// request end via the existing checkpoint mechanism.
    #[cfg(feature = "cuda")]
    pub fn forward_gemma_audio_subsample(
        &self,
        samples_16k_mono: &[f32],
    ) -> Result<crate::gemma4_audio_forward::AudioForwardOutput> {
        use cudarc::driver::sys::*;

        let audio = self.model.audio.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::cuda(
                "audio: model.audio_tower not loaded",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;
        let acfg = self.arch.audio_config.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::cuda(
                "audio: arch.audio_config not set",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;

        // ── Host: compute mel spectrogram ────────────────────────────
        // Uses the same MelExtractor config the handler used to
        // predict num_soft_tokens, so the encoder cadence stays
        // consistent across admission + runtime.
        let mel_cfg = crate::audio_preprocess::MelConfig::gemma4_e4b();
        let mel_extr = crate::audio_preprocess::MelExtractor::new(mel_cfg);
        let mel = mel_extr.compute_mel(samples_16k_mono);
        let t_in = mel_extr.num_frames(samples_16k_mono.len());
        let n_mels = mel_cfg.n_mels;
        if t_in == 0 {
            return Err(rvllm_core::RvllmError::cuda(
                "audio: mel produced 0 frames (input shorter than frame_length)",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }

        // Conv2d output dims for k=3 s=2 p=1: floor((in - 1) / 2) + 1.
        let h0 = (t_in + 1) / 2;
        let w0 = (n_mels + 1) / 2; // 64
        let h1 = (h0 + 1) / 2;
        let w1 = (w0 + 1) / 2;     // 32
        let c0 = acfg.subsampling_conv_channels[0]; // 128
        let c1 = acfg.subsampling_conv_channels[1]; // 32
        let hidden = acfg.hidden_size;              // 1024
        let proj_in_dim = (c0 / 4) * c1;            // (128/4)*32 = 1024

        // ── Upload mel as f16 to device ──────────────────────────────
        // Allocate a scratch region for the f16-converted mel and
        // copy from host. Lives until the request's arena checkpoint
        // is restored.
        let mel_bytes_f16 = t_in * n_mels * 2;
        let mel_region = self.arena.region("g4a_mel_f16", mel_bytes_f16, 16)?;
        let mel_f16_host: Vec<u16> = mel
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let mel_f16_bytes: &[u8] = bytemuck_cast_u16(&mel_f16_host);
        unsafe { mel_region.copy_from_host(mel_f16_bytes)? };
        let mel_dev = mel_region.device_ptr();
        self.audio_dump_f16(mel_dev, t_in * n_mels, "audio_input_mel.bin");

        // ── Stage 0 Conv2d ───────────────────────────────────────────
        // im2col: input [1, T, 128]  →  [1*9, h0 * w0]
        let s0_spatial = h0 * w0;
        let im2col0 = self
            .arena
            .region("g4a_s0_im2col", 9 * s0_spatial * 2, 16)?;
        unsafe { launch_im2col_3x3_s2p1_f16(
            &self.stream,
            self.fused.fn_im2col_3x3_s2p1_f16,
            mel_dev,
            im2col0.device_ptr(),
            1,
            t_in as i32,
            n_mels as i32,
            h0 as i32,
            w0 as i32,
        ) }?;

        // GEMM: weight[c0, 9] × im2col[9, s0_spatial] → conv0[c0, s0_spatial] f32
        let conv0_f32 = self
            .arena
            .region("g4a_s0_conv_f32", c0 * s0_spatial * 4, 16)?;
        unsafe {
            self.cublaslt.f16_gemm_f32(
                audio.subsample.layer0_conv.offset_bytes,
                im2col0.device_ptr(),
                conv0_f32.device_ptr(),
                c0 as i32,
                s0_spatial as i32,
                9,
                self.stream.raw(),
            )?;
        }
        // f16_gemm_f32 output is row-major [m, n] = [out_ch, spatial]
        // = CHW already. Earlier codex round-2 claim that it was HWC was
        // wrong (verified by per-stage HF parity dump: rvllm dump at
        // [s=0, c=0..6] was zero matching the corner-padding pattern
        // for c=0 across w=0..6, which is the CHW interpretation).
        // No transpose needed before layernorm_relu_chw.
        let conv0_f16 = self
            .arena
            .region("g4a_s0_conv_f16", c0 * s0_spatial * 2, 16)?;
        unsafe { launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            conv0_f32.device_ptr(),
            conv0_f16.device_ptr(),
            (c0 * s0_spatial) as i32,
        ) }?;
        self.audio_dump_f32_as_f16(conv0_f32.device_ptr(), c0 * s0_spatial,
            "audio_diag_conv0_f32.bin");
        self.audio_dump_f16(conv0_f16.device_ptr(), c0 * s0_spatial,
            "audio_diag_conv0_chw_pre_ln.bin");
        // Fused LayerNorm-over-C + ReLU (CHW input).
        unsafe { launch_layernorm_relu_chw_f16(
            &self.stream,
            self.fused.fn_layernorm_relu_chw_f16,
            conv0_f16.device_ptr(),
            audio.subsample.layer0_norm.offset_bytes,
            acfg.rms_norm_eps,
            c0 as i32,
            h0 as i32,
            w0 as i32,
        ) }?;

        self.audio_dump_f16(conv0_f16.device_ptr(), c0 * s0_spatial,
            "audio_diag_after_s0_ln.bin");
        // ── Stage 1 Conv2d ───────────────────────────────────────────
        // im2col: input [c0=128, h0, w0]  →  [c0*9=1152, h1 * w1]
        let s1_spatial = h1 * w1;
        let im2col1 = self
            .arena
            .region("g4a_s1_im2col", c0 * 9 * s1_spatial * 2, 16)?;
        unsafe { launch_im2col_3x3_s2p1_f16(
            &self.stream,
            self.fused.fn_im2col_3x3_s2p1_f16,
            conv0_f16.device_ptr(),
            im2col1.device_ptr(),
            c0 as i32,
            h0 as i32,
            w0 as i32,
            h1 as i32,
            w1 as i32,
        ) }?;
        let conv1_f32 = self
            .arena
            .region("g4a_s1_conv_f32", c1 * s1_spatial * 4, 16)?;
        unsafe {
            self.cublaslt.f16_gemm_f32(
                audio.subsample.layer1_conv.offset_bytes,
                im2col1.device_ptr(),
                conv1_f32.device_ptr(),
                c1 as i32,
                s1_spatial as i32,
                (c0 * 9) as i32,
                self.stream.raw(),
            )?;
        }
        // GEMM output is CHW directly (see stage 0 comment).
        let conv1_f16 = self
            .arena
            .region("g4a_s1_conv_f16", c1 * s1_spatial * 2, 16)?;
        unsafe { launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            conv1_f32.device_ptr(),
            conv1_f16.device_ptr(),
            (c1 * s1_spatial) as i32,
        ) }?;
        unsafe { launch_layernorm_relu_chw_f16(
            &self.stream,
            self.fused.fn_layernorm_relu_chw_f16,
            conv1_f16.device_ptr(),
            audio.subsample.layer1_norm.offset_bytes,
            acfg.rms_norm_eps,
            c1 as i32,
            h1 as i32,
            w1 as i32,
        ) }?;

        // ── Transpose [C=32, h1, w1] → [h1, w1, 32] → flatten [h1, w1*32 = 1024]
        let permuted = self
            .arena
            .region("g4a_s1_hwc", c1 * s1_spatial * 2, 16)?;
        unsafe { launch_transpose_chw_to_hwc_f16(
            &self.stream,
            self.fused.fn_transpose_chw_to_hwc_f16,
            conv1_f16.device_ptr(),
            permuted.device_ptr(),
            c1 as i32,
            h1 as i32,
            w1 as i32,
        ) }?;
        // After this point the layout is [h1, w1 * c1] contiguous,
        // which matches HF's permute(0,2,3,1).reshape(B, h1, w1*c1)
        // for B=1.

        // ── input_proj_linear: [h1, 1024] @ weight[1024, 1024].T  ──
        // Weight on disk is [hidden=1024, proj_in_dim=1024]. cuBLASLt
        // f16_gemm_f32 computes D[m, n] = A^T @ B with A column-major
        // [k, m], B column-major [k, n]. Set:
        //   A = weight (col-major [k=1024, m=hidden=1024])
        //   B = permuted (col-major [k=1024, n=h1])
        //   D = output  (col-major [m=hidden, n=h1])  → row-major [h1, hidden]
        let out_f32 = self
            .arena
            .region("g4a_proj_f32", h1 * hidden * 4, 16)?;
        // f16_gemm_f32 computes D = A * B^T = [m, k] * [k, n] = [m, n] row-major.
        // For Linear out = x @ weight^T we want D[N, out] = x[N, in] * weight[out, in]^T.
        // Pass a=x (m=N, k=in), b=weight (n=out, k=in). NOT the codex-2 (weight, x)
        // order which gives D[out, N] (transposed of what downstream expects).
        unsafe {
            self.cublaslt.f16_gemm_f32(
                permuted.device_ptr(),
                audio.subsample.input_proj.offset_bytes,
                out_f32.device_ptr(),
                h1 as i32,
                hidden as i32,
                proj_in_dim as i32,
                self.stream.raw(),
            )?;
        }
        let out_f16 = self
            .arena
            .region("g4a_proj_f16", h1 * hidden * 2, 16)?;
        unsafe { launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            out_f32.device_ptr(),
            out_f16.device_ptr(),
            (h1 * hidden) as i32,
        ) }?;

        // Sync so any error in this chain surfaces before the caller
        // tries to consume the buffer; cost is one D2D fence per
        // audio item which is negligible vs. the 12 encoder blocks
        // that will follow in B6c..d.
        unsafe { cuStreamSynchronize(self.stream.raw() as CUstream) };

        self.audio_dump_f16(out_f16.device_ptr(), h1 * hidden, "audio_after_subsample.bin");

        Ok(crate::gemma4_audio_forward::AudioForwardOutput {
            device_ptr: out_f16.device_ptr(),
            num_soft_tokens: h1,
            output_proj_dims: hidden,
        })
    }

    /// B6c FFN sub-block of one Gemma 4 audio encoder layer
    /// (`Gemma4AudioFeedForward.forward`).
    ///
    /// HF math (PyTorch reference,
    /// `models/gemma4/modeling_gemma4.py::Gemma4AudioFeedForward.forward`):
    /// ```text
    /// residual = h
    /// h = clamp(h, -clip, clip)        # clip = 1e10, noop in f16
    /// h = pre_layer_norm(h)            # RMSNorm with gamma
    /// h = ffw_layer_1(h)               # [4H, H] linear
    /// h = silu(h)
    /// h = ffw_layer_2(h)               # [H, 4H] linear
    /// h = clamp(h, -clip, clip)        # noop in f16
    /// h = post_layer_norm(h)
    /// h *= residual_weight             # 0.5
    /// h += residual
    /// ```
    ///
    /// Clamps are skipped because `gradient_clipping=1e10` is larger
    /// than f16's representable range — the operation would only
    /// touch values that have already overflowed to ±inf, which f16
    /// can't carry into the next op anyway. If a future config picks
    /// a finite clip this needs revisiting.
    ///
    /// Buffers:
    /// - `hidden`         in/out, [n_tokens, H] f16
    /// - `residual`       in,     [n_tokens, H] f16 (caller-owned)
    /// - `inter_4h_f32`   scratch, [n_tokens, 4H] f32
    /// - `inter_4h_f16`   scratch, [n_tokens, 4H] f16
    /// - `out_h_f32`      scratch, [n_tokens, H]  f32
    ///
    /// All arena regions live for the duration of the request.
    #[cfg(feature = "cuda")]
    fn forward_audio_ffn(
        &self,
        ffn: &rvllm_loader::gemma4_weights::Gemma4AudioFfn,
        residual: u64,
        hidden: u64,
        n_tokens: usize,
        hidden_dim: usize,
        residual_weight: f32,
        eps: f32,
        scratch_inter_f32: &'static str,
        scratch_inter_f16: &'static str,
        scratch_out_f32: &'static str,
    ) -> Result<()> {
        use cudarc::driver::sys::*;
        if n_tokens == 0 {
            return Ok(());
        }
        let h = hidden_dim;
        let h4 = 4 * h;

        // pre-norm into `hidden` in-place.
        unsafe {
            let mut x = hidden;
            let mut g = ffn.pre_norm.offset_bytes;
            let mut eps_ = eps;
            let mut d = h as i32;
            let args = [
                (&mut x)    as *mut u64 as *mut core::ffi::c_void,
                (&mut g)    as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_) as *mut f32 as *mut core::ffi::c_void,
                (&mut d)    as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.fused.fn_rmsnorm.raw() as CUfunction,
                n_tokens as u32, 1, 1,
                (h as u32).min(1024), 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio ffn: pre-rmsnorm launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // ffw_layer_1: clipped-linear input clamp (on rmsnormed hidden) -> GEMM -> output clamp.
        self.audio_clamp_inplace_f16(hidden, (n_tokens * h) as i32,
            ffn.layer_1.input_min, ffn.layer_1.input_max)?;
        let inter_f32 = self.arena.region(scratch_inter_f32, n_tokens * h4 * 4, 16)?;
        unsafe {
            self.cublaslt.f16_gemm_f32(
                hidden,
                ffn.layer_1.weight.offset_bytes,
                inter_f32.device_ptr(),
                n_tokens as i32,
                h4 as i32,
                h as i32,
                self.stream.raw(),
            )?;
        }
        self.audio_clamp_inplace_f32(inter_f32.device_ptr(), (n_tokens * h4) as i32,
            ffn.layer_1.output_min, ffn.layer_1.output_max)?;
        let inter_f16 = self.arena.region(scratch_inter_f16, n_tokens * h4 * 2, 16)?;
        unsafe { launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            inter_f32.device_ptr(),
            inter_f16.device_ptr(),
            (n_tokens * h4) as i32,
        ) }?;

        // SiLU in-place on [n_tokens, 4H].
        unsafe { launch_silu_inplace_f16(
            &self.stream,
            self.fused.fn_silu_inplace_f16,
            inter_f16.device_ptr(),
            (n_tokens * h4) as i32,
        ) }?;

        // ffw_layer_2: clipped-linear input clamp -> GEMM -> output clamp.
        self.audio_clamp_inplace_f16(inter_f16.device_ptr(), (n_tokens * h4) as i32,
            ffn.layer_2.input_min, ffn.layer_2.input_max)?;
        let out_f32 = self.arena.region(scratch_out_f32, n_tokens * h * 4, 16)?;
        unsafe {
            self.cublaslt.f16_gemm_f32(
                inter_f16.device_ptr(),
                ffn.layer_2.weight.offset_bytes,
                out_f32.device_ptr(),
                n_tokens as i32,
                h as i32,
                h4 as i32,
                self.stream.raw(),
            )?;
        }
        self.audio_clamp_inplace_f32(out_f32.device_ptr(), (n_tokens * h) as i32,
            ffn.layer_2.output_min, ffn.layer_2.output_max)?;
        // cast back into `hidden` (over-writing the rmsnormed input).
        unsafe { launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            out_f32.device_ptr(),
            hidden,
            (n_tokens * h) as i32,
        ) }?;

        // post-norm in-place on `hidden`.
        unsafe {
            let mut x = hidden;
            let mut g = ffn.post_norm.offset_bytes;
            let mut eps_ = eps;
            let mut d = h as i32;
            let args = [
                (&mut x)    as *mut u64 as *mut core::ffi::c_void,
                (&mut g)    as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_) as *mut f32 as *mut core::ffi::c_void,
                (&mut d)    as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.fused.fn_rmsnorm.raw() as CUfunction,
                n_tokens as u32, 1, 1,
                (h as u32).min(1024), 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio ffn: post-rmsnorm launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // hidden *= residual_weight  (scale_inplace_f16).
        unsafe {
            let mut x = hidden;
            let mut s = residual_weight;
            let mut n = (n_tokens * h) as i32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut s) as *mut f32 as *mut core::ffi::c_void,
                (&mut n) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n_tokens * h + 255) / 256) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_scale_inplace_f16.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio ffn: scale_inplace launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // hidden += residual  (vector_add_f16, dst = a + b).
        unsafe {
            // vector_add_f16 ABI is (dst, src, n) — 3 args, dst += src.
            let mut dst = hidden;
            let mut src = residual;
            let mut n = (n_tokens * h) as i32;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut n)   as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n_tokens * h + 255) / 256) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_vector_add.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio ffn: vector_add launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        Ok(())
    }

    /// B6c LightConv1D sub-block of one Gemma 4 audio encoder layer
    /// (`Gemma4AudioLightConv1d.forward`).
    ///
    /// HF math:
    /// ```text
    /// residual = h
    /// h = pre_layer_norm(h)                  # RMSNorm
    /// h = linear_start(h)                    # [H -> 2H]
    /// h = glu(h, dim=-1)                     # [N, 2H] -> [N, H], sigmoid GLU
    /// h = depthwise_conv1d(h.T).T            # causal, ks=5, groups=H
    /// h = clamp(h)                           # noop in f16
    /// h = conv_norm(h)                       # RMSNorm
    /// h = silu(h)
    /// h = linear_end(h)                      # [H -> H]
    /// h += residual                          # full residual, no scale
    /// ```
    ///
    /// Tensor layout note: HF transposes (B, N, H) to (B, H, N) for
    /// the conv1d. The existing `causal_conv1d_f16` kernel consumes
    /// row-major `[seq_len+ks-1, channels]` (channel-last), which
    /// matches our (B=1, N, H) layout directly — no host-side
    /// transpose is needed. We just left-pad with `ks-stride = 4`
    /// zero rows.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn forward_audio_lconv1d(
        &self,
        lconv: &rvllm_loader::gemma4_weights::Gemma4AudioLConv1d,
        residual: u64,
        hidden: u64,
        n_tokens: usize,
        hidden_dim: usize,
        conv_kernel_size: usize,
        eps: f32,
        scratch_pre_gemm_f32: &'static str,
        scratch_glu_f16: &'static str,
        scratch_padded_f16: &'static str,
        scratch_post_gemm_f32: &'static str,
    ) -> Result<()> {
        use cudarc::driver::sys::*;
        if n_tokens == 0 {
            return Ok(());
        }
        let h = hidden_dim;
        let ks = conv_kernel_size;
        let left_pad = ks.saturating_sub(1); // stride is 1, dilation 1

        // pre_norm in-place on hidden.
        unsafe {
            let mut x = hidden;
            let mut g = lconv.pre_norm.offset_bytes;
            let mut eps_ = eps;
            let mut d = h as i32;
            let args = [
                (&mut x)    as *mut u64 as *mut core::ffi::c_void,
                (&mut g)    as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_) as *mut f32 as *mut core::ffi::c_void,
                (&mut d)    as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.fused.fn_rmsnorm.raw() as CUfunction,
                n_tokens as u32, 1, 1,
                (h as u32).min(1024), 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio lconv1d: pre-rmsnorm launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // linear_start: clipped input clamp -> GEMM -> output clamp.
        self.audio_clamp_inplace_f16(hidden, (n_tokens * h) as i32,
            lconv.linear_start.input_min, lconv.linear_start.input_max)?;
        let two_h = 2 * h;
        let pre_f32 = self.arena.region(scratch_pre_gemm_f32, n_tokens * two_h * 4, 16)?;
        unsafe {
            self.cublaslt.f16_gemm_f32(
                hidden,
                lconv.linear_start.weight.offset_bytes,
                pre_f32.device_ptr(),
                n_tokens as i32,
                two_h as i32,
                h as i32,
                self.stream.raw(),
            )?;
        }
        // linear_start output clamp on the f32 buffer.
        self.audio_clamp_inplace_f32(pre_f32.device_ptr(),
            (n_tokens * two_h) as i32,
            lconv.linear_start.output_min, lconv.linear_start.output_max)?;
        // Reuse `hidden` as the linear_start [N, 2H] target — too small.
        // We need a [N, 2H] f16 buffer; size = 2 * N*H * 2 bytes.
        // Use the padded buffer slot's first half by allocating a
        // dedicated f16 region.
        // glu produces [N, H] f16; write that into a separate scratch
        // so the padded conv buffer can layout zeros + glu output.
        let glu_buf = self.arena.region(scratch_glu_f16, n_tokens * h * 2, 16)?;
        // Cast linear_start output to f16. We need [N, 2H] f16 to feed
        // GLU. Reuse `pre_f32`'s slot is not possible (f32 vs f16
        // strides differ); allocate fresh f16 staging using the same
        // scratch name + offset trick. Simpler: do GEMM-then-cast
        // into `glu_buf` of size [N, 2H] f16, then call glu in-place
        // splitting; but glu writes [N, H] so input/output overlap is
        // a problem.
        //
        // Cleanest: allocate a separate `lstart_f16` [N, 2H] arena
        // region. We'll borrow `scratch_padded_f16` for that — its
        // size is [N+ks-1, H] = [N+4, H], i.e. (N+4)*H*2 bytes which
        // is ≥ 2*N*H*2 only when N <= 4 (not the common case). So we
        // still need a distinct slot.
        //
        // Cast linear_start output to a DISTINCT [N, 2H] f16 buffer.
        // The previous version cast f32->f16 in place on `pre_f32`, but
        // the cast kernel is per-element parallel: thread k writes 2
        // bytes at offset 2k while thread k/2 still needs to read 4
        // bytes at offset 4k = 2*(2k), so writes from later threads
        // can clobber f32 source bytes that earlier threads have yet
        // to read. Race condition (codex round 6 fix #1).
        let lstart_f16 = self.arena.region(
            "g4a_blk_lconv_lstart_f16", n_tokens * two_h * 2, 16)?;
        unsafe { launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            pre_f32.device_ptr(),
            lstart_f16.device_ptr(),
            (n_tokens * two_h) as i32,
        ) }?;

        // GLU split: read [N, 2H] f16 from lstart_f16, write [N, H]
        // f16 into glu_buf.
        unsafe { launch_glu_split_sigmoid_f16(
            &self.stream,
            self.fused.fn_glu_split_sigmoid_f16,
            lstart_f16.device_ptr(),
            glu_buf.device_ptr(),
            n_tokens as i32,
            h as i32,
        ) }?;

        // Build the padded conv input [N + ks-1, H] f16:
        //   rows [0..left_pad)            = 0
        //   rows [left_pad..left_pad+N)   = glu_buf
        let padded_rows = n_tokens + left_pad;
        let padded_bytes = padded_rows * h * 2;
        let padded = self.arena.region(scratch_padded_f16, padded_bytes, 16)?;
        unsafe {
            // Zero the leading left_pad rows.
            let zero_bytes = left_pad * h * 2;
            cuMemsetD8Async(
                padded.device_ptr(),
                0,
                zero_bytes,
                self.stream.raw() as CUstream,
            );
            // Copy GLU output into rows [left_pad..].
            cuMemcpyDtoDAsync_v2(
                padded.device_ptr() + zero_bytes as u64,
                glu_buf.device_ptr(),
                (n_tokens * h * 2) as usize,
                self.stream.raw() as CUstream,
            );
        }

        // causal_conv1d_f16: output [N, H] f16 into `hidden`
        // (overwrites the rmsnormed input we already consumed).
        unsafe {
            let mut out = hidden;
            let mut inp = padded.device_ptr();
            let mut w = lconv.depthwise.offset_bytes;
            let mut seq = n_tokens as i32;
            let mut ch = h as i32;
            let mut ks_ = ks as i32;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                (&mut w)   as *mut u64 as *mut core::ffi::c_void,
                (&mut seq) as *mut i32 as *mut core::ffi::c_void,
                (&mut ch)  as *mut i32 as *mut core::ffi::c_void,
                (&mut ks_) as *mut i32 as *mut core::ffi::c_void,
            ];
            // Grid (ceil(channels/BLOCK), seq_len, 1), block (BLOCK,1,1).
            let block: u32 = 128;
            let grid_x: u32 = ((h as u32 + block - 1) / block) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_causal_conv1d_f16.raw() as CUfunction,
                grid_x, n_tokens as u32, 1,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio lconv1d: causal_conv1d_f16 launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // conv_norm in-place on hidden.
        unsafe {
            let mut x = hidden;
            let mut g = lconv.conv_norm.offset_bytes;
            let mut eps_ = eps;
            let mut d = h as i32;
            let args = [
                (&mut x)    as *mut u64 as *mut core::ffi::c_void,
                (&mut g)    as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_) as *mut f32 as *mut core::ffi::c_void,
                (&mut d)    as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.fused.fn_rmsnorm.raw() as CUfunction,
                n_tokens as u32, 1, 1,
                (h as u32).min(1024), 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio lconv1d: conv_norm launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // SiLU in-place [N, H].
        unsafe { launch_silu_inplace_f16(
            &self.stream,
            self.fused.fn_silu_inplace_f16,
            hidden,
            (n_tokens * h) as i32,
        ) }?;

        // linear_end: clipped input clamp -> GEMM -> output clamp.
        self.audio_clamp_inplace_f16(hidden, (n_tokens * h) as i32,
            lconv.linear_end.input_min, lconv.linear_end.input_max)?;
        let post_f32 = self.arena.region(scratch_post_gemm_f32, n_tokens * h * 4, 16)?;
        unsafe {
            self.cublaslt.f16_gemm_f32(
                hidden,
                lconv.linear_end.weight.offset_bytes,
                post_f32.device_ptr(),
                n_tokens as i32,
                h as i32,
                h as i32,
                self.stream.raw(),
            )?;
        }
        self.audio_clamp_inplace_f32(post_f32.device_ptr(), (n_tokens * h) as i32,
            lconv.linear_end.output_min, lconv.linear_end.output_max)?;
        unsafe { launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            post_f32.device_ptr(),
            hidden,
            (n_tokens * h) as i32,
        ) }?;

        // hidden += residual.
        unsafe {
            // vector_add_f16 ABI is (dst, src, n) — 3 args, dst += src.
            let mut dst = hidden;
            let mut src = residual;
            let mut n = (n_tokens * h) as i32;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut n)   as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n_tokens * h + 255) / 256) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_vector_add.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio lconv1d: vector_add launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        Ok(())
    }

    /// Optional debug dump (gated by RVLLM_E4B_AUDIO_DUMP_DIR env var)
    /// of a device-side f16 buffer to the named .bin file. Caller is
    /// responsible for stream-fence semantics.
    #[cfg(feature = "cuda")]
    fn audio_dump_f16(&self, dev_ptr: u64, n_elems: usize, name: &str) {
        let Ok(dir) = std::env::var("RVLLM_E4B_AUDIO_DUMP_DIR") else { return; };
        let nbytes = n_elems * 2;
        let mut buf = vec![0u8; nbytes];
        unsafe {
            let rc = cudarc::driver::sys::cuStreamSynchronize(
                self.stream.raw() as cudarc::driver::sys::CUstream,
            );
            if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS { return; }
            let rc = cudarc::driver::sys::cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut _, dev_ptr, nbytes,
            );
            if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS { return; }
        }
        let _ = std::fs::create_dir_all(&dir);
        let path = std::path::Path::new(&dir).join(name);
        let _ = std::fs::write(&path, &buf);
    }

    /// Same as `audio_dump_f16` but for an f32 buffer cast to f16 on the
    /// host before writing (so the diff harness can read all dumps as f16).
    #[cfg(feature = "cuda")]
    fn audio_dump_f32_as_f16(&self, dev_ptr: u64, n_elems: usize, name: &str) {
        let Ok(dir) = std::env::var("RVLLM_E4B_AUDIO_DUMP_DIR") else { return; };
        let nbytes_f32 = n_elems * 4;
        let mut buf_f32 = vec![0u8; nbytes_f32];
        unsafe {
            let rc = cudarc::driver::sys::cuStreamSynchronize(
                self.stream.raw() as cudarc::driver::sys::CUstream,
            );
            if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS { return; }
            let rc = cudarc::driver::sys::cuMemcpyDtoH_v2(
                buf_f32.as_mut_ptr() as *mut _, dev_ptr, nbytes_f32,
            );
            if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS { return; }
        }
        let mut buf_f16 = Vec::with_capacity(n_elems * 2);
        for chunk in buf_f32.chunks_exact(4) {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let h = half::f16::from_f32(v);
            buf_f16.extend_from_slice(&h.to_bits().to_le_bytes());
        }
        let _ = std::fs::create_dir_all(&dir);
        let path = std::path::Path::new(&dir).join(name);
        let _ = std::fs::write(&path, &buf_f16);
    }

    /// In-place clamp of an f16 buffer to [lo, hi] on the current stream.
    #[cfg(feature = "cuda")]
    fn audio_clamp_inplace_f16(&self, x: u64, n: i32, lo: f32, hi: f32) -> Result<()> {
        use cudarc::driver::sys::*;
        if n <= 0 { return Ok(()); }
        unsafe {
            let mut x = x;
            let mut lo = lo;
            let mut hi = hi;
            let mut n_ = n;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut lo) as *mut f32 as *mut core::ffi::c_void,
                (&mut hi) as *mut f32 as *mut core::ffi::c_void,
                (&mut n_) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n as i64 + block as i64 - 1) / block as i64) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_clamp_inplace_f16.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio: clamp_inplace_f16 launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// In-place clamp of an f32 buffer to [lo, hi] on the current stream.
    #[cfg(feature = "cuda")]
    fn audio_clamp_inplace_f32(&self, x: u64, n: i32, lo: f32, hi: f32) -> Result<()> {
        use cudarc::driver::sys::*;
        if n <= 0 { return Ok(()); }
        unsafe {
            let mut x = x;
            let mut lo = lo;
            let mut hi = hi;
            let mut n_ = n;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut lo) as *mut f32 as *mut core::ffi::c_void,
                (&mut hi) as *mut f32 as *mut core::ffi::c_void,
                (&mut n_) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n as i64 + block as i64 - 1) / block as i64) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_clamp_inplace_f32.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio: clamp_inplace_f32 launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Compute the per-head Q scale vector used by Gemma 4 audio
    /// attention on the host and upload it as an f32 buffer of
    /// shape `[head_dim]`. The values are:
    ///   scale[d] = q_scale_scalar * softplus(per_dim_scale[d]) * k_scale
    /// where:
    ///   q_scale_scalar = (head_dim ** -0.5) / ln(2)
    ///   k_scale        = ln(1 + e) / ln(2)
    ///
    /// HF applies `q_scale_scalar * softplus(per_dim_scale)` to Q
    /// and `k_scale` to K. Mathematically the attention scores
    /// depend only on the product, so we fold both into the Q-side
    /// scale and skip the separate K scaling launch.
    ///
    /// Because `per_dim_scale` is a learned f16 weight of shape
    /// `[head_dim]` (128 values on E4B), the entire computation is
    /// trivially fast on the CPU and avoids needing a softplus
    /// kernel. The result is uploaded to a small arena region so
    /// the attention forward can multiply Q[n, h, d] by scale[d]
    /// per-channel without re-evaluating softplus.
    #[cfg(feature = "cuda")]
    fn audio_q_scale_vector_upload(
        &self,
        per_dim_scale: &rvllm_loader::weights::F16Weight,
        head_dim: usize,
        scratch_name: &'static str,
    ) -> Result<rvllm_mem::Region<'_>> {
        use cudarc::driver::sys::*;
        let head_dim_f = head_dim as f32;
        let q_scale_scalar = head_dim_f.powf(-0.5) / std::f32::consts::LN_2;
        // k_scale is applied to K (not relative_K) in HF, so do NOT fold
        // it here — apply it separately to matrix_ac after the GEMM.
        let combined_scalar = q_scale_scalar;
        // Read per_dim_scale f16 weight from device into host: it's
        // tiny (head_dim=128 -> 256 bytes), so a one-shot DtoH copy
        // is fine.
        let mut hb = vec![0u16; head_dim];
        unsafe {
            let rc = cuMemcpyDtoH_v2(
                hb.as_mut_ptr() as *mut core::ffi::c_void,
                per_dim_scale.offset_bytes,
                head_dim * 2,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio: DtoH per_dim_scale failed",
                    rvllm_core::CudaErrorKind::Other,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // softplus(x) = ln(1 + exp(x)).
        // Numerically stable: for x > 16, softplus(x) ≈ x.
        let mut scale_f32 = Vec::with_capacity(head_dim);
        for &h in &hb {
            let v = half::f16::from_bits(h).to_f32();
            let sp = if v > 16.0 {
                v
            } else {
                (1.0_f32 + v.exp()).ln()
            };
            scale_f32.push(combined_scalar * sp);
        }
        let region = self.arena.region(scratch_name, head_dim * 4, 16)?;
        unsafe {
            let rc = cuMemcpyHtoDAsync_v2(
                region.device_ptr(),
                scale_f32.as_ptr() as *const core::ffi::c_void,
                head_dim * 4,
                self.stream.raw() as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio: HtoD q_scale vector failed",
                    rvllm_core::CudaErrorKind::Other,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(region)
    }

    /// Compute the Gemma 4 audio relative-position encoding table
    /// on the host and upload it as an f32 buffer of shape
    /// `[pos_len, hidden_size]`.
    ///
    /// Per HF `Gemma4AudioRelPositionalEncoding.forward`:
    ///
    ///   num_timescales        = hidden_size / 2
    ///   log_increment         = ln(10000) / max(num_timescales - 1, 1)
    ///   inv_timescales[k]     = exp(k * -log_increment)
    ///   position_ids          = [pos_len-1, pos_len-2, ..., 0]    (length pos_len)
    ///   scaled_time[i, k]     = position_ids[i] * inv_timescales[k]
    ///   pos_embed[i, :half]   = sin(scaled_time[i])
    ///   pos_embed[i, half:]   = cos(scaled_time[i])
    ///
    /// `pos_len` equals `attention_context_left` (13 on E4B). The
    /// table is identical for every request and every layer, so we
    /// keep it as an f32 arena region (small: 13 × 1024 × 4 ≈ 52 KB)
    /// and feed it to each layer's `relative_k_proj` GEMM.
    ///
    /// Returns the uploaded f32 region.
    #[cfg(feature = "cuda")]
    fn audio_pos_embed_upload(
        &self,
        hidden_size: usize,
        pos_len: usize,
        scratch_name: &'static str,
    ) -> Result<rvllm_mem::Region<'_>> {
        use cudarc::driver::sys::*;
        let num_timescales = hidden_size / 2;
        let log_increment = (10000.0_f32).ln() / ((num_timescales.max(2) - 1) as f32);
        let mut inv_timescales = Vec::with_capacity(num_timescales);
        for k in 0..num_timescales {
            inv_timescales.push((-(k as f32) * log_increment).exp());
        }
        let mut table = vec![0.0_f32; pos_len * hidden_size];
        for i in 0..pos_len {
            // position_ids = pos_len-1 .. 0 (reversed) per HF.
            let pid = (pos_len - 1 - i) as f32;
            for k in 0..num_timescales {
                let t = pid * inv_timescales[k];
                table[i * hidden_size + k] = t.sin();
                table[i * hidden_size + num_timescales + k] = t.cos();
            }
        }
        let bytes = pos_len * hidden_size * 4;
        let region = self.arena.region(scratch_name, bytes, 16)?;
        unsafe {
            let rc = cuMemcpyHtoDAsync_v2(
                region.device_ptr(),
                table.as_ptr() as *const core::ffi::c_void,
                bytes,
                self.stream.raw() as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio: HtoD pos_embed failed",
                    rvllm_core::CudaErrorKind::Other,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(region)
    }

    /// Q/K/V projection + Q-scale apply for the Gemma 4 audio
    /// chunked attention. Three GEMMs (hidden_dim -> num_heads*head_dim
    /// each) produce f32 [n_tokens, hidden_dim] buffers; the Q output
    /// is scaled per-channel via the precomputed q_scale_vec.
    ///
    /// HF carries Q/K/V in f32 from projection through softmax. We
    /// match that — the cuBLASLt wrapper produces f32 output natively
    /// (f16 in, f32 accumulator + cast).
    ///
    /// Caller owns `q_scale_vec` (a head_dim f32 region from
    /// `audio_q_scale_vector_upload`, where k_scale is already folded
    /// in). After this call:
    ///   * Q has the combined `q_scale * softplus(per_dim) * k_scale`
    ///     applied per head_dim channel.
    ///   * K and V are unscaled (k_scale lives on the Q side now).
    ///
    /// Returns the three arena regions in (Q, K, V) order; lifetime
    /// is the request's checkpoint.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn forward_audio_attention_qkv(
        &self,
        attn: &rvllm_loader::gemma4_weights::Gemma4AudioAttention,
        hidden_f16: u64,
        n_tokens: usize,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
        q_scale_vec: u64,
        scratch_q_f32: &'static str,
        scratch_k_f32: &'static str,
        scratch_v_f32: &'static str,
    ) -> Result<(rvllm_mem::Region<'_>, rvllm_mem::Region<'_>, rvllm_mem::Region<'_>)> {
        use cudarc::driver::sys::*;
        let q_proj_out = num_heads * head_dim; // == hidden_dim on E4B
        let f32_bytes = n_tokens * q_proj_out * 4;

        let nq = (n_tokens * q_proj_out) as i32;
        // Q proj + output clamp (Gemma4ClippableLinear semantics).
        let q = self.arena.region(scratch_q_f32, f32_bytes, 16)?;
        unsafe {
            self.cublaslt.f16_gemm_f32(
                hidden_f16,
                attn.q.weight.offset_bytes,
                q.device_ptr(),
                n_tokens as i32,
                q_proj_out as i32,
                hidden_dim as i32,
                self.stream.raw(),
            )?;
        }
        self.audio_clamp_inplace_f32(q.device_ptr(), nq,
            attn.q.output_min, attn.q.output_max)?;
        // K proj + output clamp.
        let k = self.arena.region(scratch_k_f32, f32_bytes, 16)?;
        unsafe {
            self.cublaslt.f16_gemm_f32(
                hidden_f16,
                attn.k.weight.offset_bytes,
                k.device_ptr(),
                n_tokens as i32,
                q_proj_out as i32,
                hidden_dim as i32,
                self.stream.raw(),
            )?;
        }
        self.audio_clamp_inplace_f32(k.device_ptr(), nq,
            attn.k.output_min, attn.k.output_max)?;
        // V proj + output clamp.
        let v = self.arena.region(scratch_v_f32, f32_bytes, 16)?;
        unsafe {
            self.cublaslt.f16_gemm_f32(
                hidden_f16,
                attn.v.weight.offset_bytes,
                v.device_ptr(),
                n_tokens as i32,
                q_proj_out as i32,
                hidden_dim as i32,
                self.stream.raw(),
            )?;
        }
        self.audio_clamp_inplace_f32(v.device_ptr(), nq,
            attn.v.output_min, attn.v.output_max)?;

        // Apply per-head_dim scale to Q. The output layout from
        // cuBLASLt is row-major [n_tokens, q_proj_out] = [n_tokens,
        // num_heads * head_dim] so [outer = n_tokens * num_heads,
        // head_dim] is a valid view.
        unsafe {
            let mut x = q.device_ptr();
            let mut s = q_scale_vec;
            let mut outer = (n_tokens * num_heads) as i32;
            let mut hd = head_dim as i32;
            let args = [
                (&mut x)     as *mut u64 as *mut core::ffi::c_void,
                (&mut s)     as *mut u64 as *mut core::ffi::c_void,
                (&mut outer) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd)    as *mut i32 as *mut core::ffi::c_void,
            ];
            let total = (n_tokens * q_proj_out) as i64;
            let block: u32 = 256;
            let grid: u32 = ((total + block as i64 - 1) / block as i64) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_scale_per_dim_f32.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio attn: scale_per_dim_f32 launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        Ok((q, k, v))
    }

    /// Pad Q on the right with zeros to a multiple of `chunk_size`.
    /// Input Q is f32 `[n_tokens, hidden_dim]`; output is f32
    /// `[num_blocks * chunk_size, hidden_dim]` where
    /// `num_blocks = ceil(n_tokens / chunk_size)`.
    #[cfg(feature = "cuda")]
    fn audio_attention_pad_q_f32(
        &self,
        q: u64,
        n_tokens: usize,
        hidden_dim: usize,
        chunk_size: usize,
        scratch: &'static str,
    ) -> Result<(rvllm_mem::Region<'_>, usize)> {
        use cudarc::driver::sys::*;
        let num_blocks = (n_tokens + chunk_size - 1) / chunk_size;
        let padded_rows = num_blocks * chunk_size;
        let bytes = padded_rows * hidden_dim * 4;
        let region = self.arena.region(scratch, bytes, 16)?;
        unsafe {
            // Copy the live rows first.
            cuMemcpyDtoDAsync_v2(
                region.device_ptr(),
                q,
                (n_tokens * hidden_dim * 4) as usize,
                self.stream.raw() as CUstream,
            );
            // Zero-pad the trailing rows.
            let tail_rows = padded_rows - n_tokens;
            if tail_rows > 0 {
                cuMemsetD8Async(
                    region.device_ptr() + (n_tokens * hidden_dim * 4) as u64,
                    0,
                    tail_rows * hidden_dim * 4,
                    self.stream.raw() as CUstream,
                );
            }
        }
        Ok((region, num_blocks))
    }

    /// Pad K (or V) with `past_horizon` zero rows on the left + tail
    /// zeros so the total row count covers every block's context
    /// window. Then run `audio_chunk_extract_context_f32` to produce
    /// the `[num_blocks, context_size, num_heads, head_dim]` f32
    /// context tensor consumed by the matrix_ac batched-strided GEMM.
    ///
    /// The right-pad must be at least `chunk_size - 1 + future_horizon`
    /// rows so the last block's context window is fully in-range
    /// without an OOB guard in the kernel. We over-allocate slightly
    /// — cheap relative to the encoder block costs and lets us share
    /// one pad buffer across all 12 layers via static naming.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn audio_attention_extract_context_f32(
        &self,
        kv: u64,
        n_tokens: usize,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
        num_blocks: usize,
        chunk_size: usize,
        past_horizon: usize,
        context_size: usize,
        scratch_pad: &'static str,
        scratch_ctx: &'static str,
    ) -> Result<rvllm_mem::Region<'_>> {
        use cudarc::driver::sys::*;
        // Total padded row count: `past_horizon` zeros on the left,
        // `n_tokens` real rows, then enough zeros so block
        // (num_blocks-1)'s `context_size`-wide window fits.
        let last_block_start = (num_blocks - 1) * chunk_size;
        let max_src = last_block_start + context_size;
        let n_padded = max_src.max(past_horizon + n_tokens);
        let pad_bytes = n_padded * hidden_dim * 4;
        let pad = self.arena.region(scratch_pad, pad_bytes, 16)?;
        unsafe {
            // Zero the entire padded buffer first (covers both
            // left and right pad).
            cuMemsetD8Async(
                pad.device_ptr(),
                0,
                pad_bytes,
                self.stream.raw() as CUstream,
            );
            // Copy live rows into the [past_horizon..past_horizon+n_tokens) slice.
            cuMemcpyDtoDAsync_v2(
                pad.device_ptr() + (past_horizon * hidden_dim * 4) as u64,
                kv,
                (n_tokens * hidden_dim * 4) as usize,
                self.stream.raw() as CUstream,
            );
        }
        // Allocate context output: [num_blocks, context, H, D] f32.
        let ctx_bytes = num_blocks * context_size * hidden_dim * 4;
        let ctx = self.arena.region(scratch_ctx, ctx_bytes, 16)?;
        unsafe {
            let mut src = pad.device_ptr();
            let mut dst = ctx.device_ptr();
            let mut nb = num_blocks as i32;
            let mut cs = context_size as i32;
            let mut nh = num_heads as i32;
            let mut hd = head_dim as i32;
            let mut ck = chunk_size as i32;
            let mut np = n_padded as i32;
            let args = [
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut nb)  as *mut i32 as *mut core::ffi::c_void,
                (&mut cs)  as *mut i32 as *mut core::ffi::c_void,
                (&mut nh)  as *mut i32 as *mut core::ffi::c_void,
                (&mut hd)  as *mut i32 as *mut core::ffi::c_void,
                (&mut ck)  as *mut i32 as *mut core::ffi::c_void,
                (&mut np)  as *mut i32 as *mut core::ffi::c_void,
            ];
            let total = (num_blocks * context_size * num_heads * head_dim) as i64;
            let block: u32 = 256;
            let grid: u32 = ((total + block as i64 - 1) / block as i64) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_audio_chunk_extract_context_f32.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio attn: chunk_extract_context_f32 launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(ctx)
    }

    /// Per-layer projection of the shared sinusoidal position table
    /// through the audio-attention's `relative_k_proj` linear:
    ///   rel_K[pos_len, num_heads * head_dim] f32
    ///     = pos_embed_f16 @ Wrk^T
    /// where `pos_embed_f16` is the f16 cast of `pos_embed_f32`
    /// (precomputed once via `audio_pos_embed_upload`).
    ///
    /// The cast goes into a per-layer scratch slot so the f16 stage
    /// can be reused on each layer without re-touching the shared
    /// f32 pos_embed region.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn audio_attention_rel_k_proj(
        &self,
        attn: &rvllm_loader::gemma4_weights::Gemma4AudioAttention,
        pos_embed_f32: u64,
        pos_len: usize,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
        scratch_pos_f16: &'static str,
        scratch_rel_k_f32: &'static str,
    ) -> Result<rvllm_mem::Region<'_>> {
        let proj_out = num_heads * head_dim;
        // Cast pos_embed f32 -> f16 into a layer-scoped scratch
        // (size pos_len * hidden * 2 bytes; ~26 KB for pos_len=13,
        // hidden=1024).
        let pos_f16 = self.arena.region(scratch_pos_f16, pos_len * hidden_dim * 2, 16)?;
        // The existing cast_f32_to_f16 kernel handles this — same
        // path as the audio subsample stage.
        unsafe { launch_cast_f32_to_f16(
            &self.stream,
            self.fused.fn_cast_f32_to_f16,
            pos_embed_f32,
            pos_f16.device_ptr(),
            (pos_len * hidden_dim) as i32,
        ) }?;
        // GEMM:  D[m=proj_out, n=pos_len] = Wrk^T @ pos_embed_f16
        // Row-major view: D[pos_len, proj_out].
        let rel_k = self.arena.region(scratch_rel_k_f32, pos_len * proj_out * 4, 16)?;
        unsafe {
            self.cublaslt.f16_gemm_f32(
                pos_f16.device_ptr(),
                attn.relative_k.offset_bytes,
                rel_k.device_ptr(),
                pos_len as i32,
                proj_out as i32,
                hidden_dim as i32,
                self.stream.raw(),
            )?;
        }
        Ok(rel_k)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn forward_audio_attention(
        &self,
        attn: &rvllm_loader::gemma4_weights::Gemma4AudioAttention,
        hidden: u64,
        residual: u64,
        pos_embed_f32: u64,
        q_scale_vec: u64,
        n_tokens: usize,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
        chunk_size: usize,
        past_horizon: usize,
        context_size: usize,
        pos_len: usize,
        softcap: f32,
    ) -> Result<()> {
        use cudarc::driver::sys::*;
        if n_tokens == 0 {
            return Ok(());
        }
        let stream_raw = self.stream.raw();

        let (q_f32, k_f32, v_f32) = self.forward_audio_attention_qkv(
            attn, hidden, n_tokens, hidden_dim, num_heads, head_dim, q_scale_vec,
            "g4a_attn_q_f32", "g4a_attn_k_f32", "g4a_attn_v_f32",
        )?;

        let (q_padded_f32, num_blocks) = self.audio_attention_pad_q_f32(
            q_f32.device_ptr(), n_tokens, hidden_dim, chunk_size,
            "g4a_attn_q_padded_f32",
        )?;
        let q_padded_rows = num_blocks * chunk_size;

        let k_ctx_f32 = self.audio_attention_extract_context_f32(
            k_f32.device_ptr(), n_tokens, hidden_dim, num_heads, head_dim,
            num_blocks, chunk_size, past_horizon, context_size,
            "g4a_attn_k_pad_f32", "g4a_attn_k_ctx_f32",
        )?;
        let v_ctx_f32 = self.audio_attention_extract_context_f32(
            v_f32.device_ptr(), n_tokens, hidden_dim, num_heads, head_dim,
            num_blocks, chunk_size, past_horizon, context_size,
            "g4a_attn_v_pad_f32", "g4a_attn_v_ctx_f32",
        )?;

        let rel_k_f32 = self.audio_attention_rel_k_proj(
            attn, pos_embed_f32, pos_len, hidden_dim, num_heads, head_dim,
            "g4a_attn_pos_f16", "g4a_attn_rel_k_f32",
        )?;

        // Q, K, rel_K stay in f32 through matrix_ac and matrix_bd
        // (codex round 5 #1 / round 6 #2). V is cast to f16 because the
        // V-transpose kernel + (scores @ V) GEMM still run in mixed
        // precision via f16_gemm_f32_batched_strided. Lifting V to f32
        // is a follow-up.
        let v_ctx_f16 = self.arena.region(
            "g4a_attn_v_ctx_f16", num_blocks * context_size * hidden_dim * 2, 16)?;
        unsafe {
            launch_cast_f32_to_f16(&self.stream, self.fused.fn_cast_f32_to_f16,
                v_ctx_f32.device_ptr(), v_ctx_f16.device_ptr(),
                (num_blocks * context_size * hidden_dim) as i32)?;
        }

        // scores layout per head h: [num_blocks, chunk, context] f32 contiguous
        // matrix_bd buf per head: [num_blocks * chunk, pos_len] f32 contiguous
        // matrix_bd shifted: [num_blocks, chunk, context] f32 (we reuse scores
        // buffer with += semantics via add_inplace_f32 after rel_shift produces
        // matrix_bd_shifted in scratch)
        let scores_f32 = self.arena.region(
            "g4a_attn_scores_f32",
            num_heads * num_blocks * chunk_size * context_size * 4, 16)?;
        let matrix_bd_f32 = self.arena.region(
            "g4a_attn_bd_f32",
            num_heads * num_blocks * chunk_size * pos_len * 4, 16)?;
        let matrix_bd_shifted_f32 = self.arena.region(
            "g4a_attn_bd_shifted_f32",
            num_heads * num_blocks * chunk_size * context_size * 4, 16)?;
        let scores_f16 = self.arena.region(
            "g4a_attn_scores_f16",
            num_heads * num_blocks * chunk_size * context_size * 2, 16)?;

        // Per-head batched-strided GEMMs for matrix_ac and (scores @ V_ctx).
        // Q_padded layout f16: [num_blocks, chunk, num_heads, head_dim].
        // K_ctx     layout f16: [num_blocks, context, num_heads, head_dim].
        // V_ctx     layout f16: same as K_ctx.
        // For head h, per-block (blk):
        //   Q_blk_h: [chunk, head_dim]  starts at (blk*chunk*H*D + h*D)*2 bytes
        //            lda = H*D,   stride_a = chunk*H*D
        //   K_blk_h: [context, head_dim] starts at (blk*context*H*D + h*D)*2
        //            ldb = H*D,   stride_b = context*H*D
        //   out:     [chunk, context] contiguous per (h, blk), ldd = context,
        //            stride_d = chunk*context
        // matrix_ac in f32: Q_f32 [num_blocks, chunk, H, D] (head offset h*D, 4 bytes/elem)
        for h in 0..num_heads {
            let q_head_off = (h * head_dim * 4) as u64;
            let k_head_off = (h * head_dim * 4) as u64;
            let scores_head_off =
                (h * num_blocks * chunk_size * context_size * 4) as u64;
            unsafe {
                self.cublaslt.f32_gemm_f32_batched_strided(
                    q_padded_f32.device_ptr() + q_head_off,
                    k_ctx_f32.device_ptr() + k_head_off,
                    scores_f32.device_ptr() + scores_head_off,
                    chunk_size as i32,
                    context_size as i32,
                    head_dim as i32,
                    num_blocks as i32,
                    (num_heads * head_dim) as i32,
                    (num_heads * head_dim) as i32,
                    context_size as i32,
                    (chunk_size * num_heads * head_dim) as i64,
                    (context_size * num_heads * head_dim) as i64,
                    (chunk_size * context_size) as i64,
                    stream_raw,
                )?;
            }
        }

        // Apply k_scale to scores_f32 (matrix_ac contribution only). HF
        // scales K (real K context) by k_scale = ln(1+e)/ln2 but does NOT
        // scale relative_K. Since matrix_ac = Q @ K^T and scaling is
        // linear, equivalent to scaling scores_f32 by k_scale here.
        unsafe {
            let mut x = scores_f32.device_ptr();
            let mut c = (1.0_f32 + std::f32::consts::E).ln() / std::f32::consts::LN_2;
            let mut n = (num_heads * num_blocks * chunk_size * context_size) as i32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut c) as *mut f32 as *mut core::ffi::c_void,
                (&mut n) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n as i64 + block as i64 - 1) / block as i64) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_scale_scalar_inplace_f32.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0, stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio attn: k_scale apply failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // matrix_bd per head: Q_flat[h, num_blocks*chunk, head_dim] @ rel_K^T[h, head_dim, pos_len]
        //   Q is [num_blocks*chunk, H, D]: per head stride_a = D (between rows it's H*D)
        //   rel_K_f16 is [pos_len, H, D]:  per head stride_b = D, ldb = H*D
        //   out per head: [num_blocks*chunk, pos_len] contiguous
        // matrix_bd in f32: Q_f32 × rel_K_f32^T per head.
        let q_flat_rows = num_blocks * chunk_size;
        for h in 0..num_heads {
            let q_head_off = (h * head_dim * 4) as u64;
            let rel_k_head_off = (h * head_dim * 4) as u64;
            let bd_head_off = (h * q_flat_rows * pos_len * 4) as u64;
            unsafe {
                self.cublaslt.f32_gemm_f32_batched_strided(
                    q_padded_f32.device_ptr() + q_head_off,
                    rel_k_f32.device_ptr() + rel_k_head_off,
                    matrix_bd_f32.device_ptr() + bd_head_off,
                    q_flat_rows as i32,
                    pos_len as i32,
                    head_dim as i32,
                    1,
                    (num_heads * head_dim) as i32,
                    (num_heads * head_dim) as i32,
                    pos_len as i32,
                    0, 0, 0,
                    stream_raw,
                )?;
            }
        }

        // rel_shift each per-head [num_blocks, chunk, pos_len] -> [num_blocks, chunk, context]
        // The audio_chunk_extract_context_f32 kernel uses a 4D batch view
        // (num_blocks, context, heads, head_dim); rel_shift kernel takes
        // (batch, heads, num_blocks, chunk, pos_len/context). We call with
        // batch=1, heads=num_heads and have data laid out [H, NB, chunk, pos_len].
        unsafe {
            let mut src = matrix_bd_f32.device_ptr();
            let mut dst = matrix_bd_shifted_f32.device_ptr();
            let mut batch = 1i32;
            let mut heads = num_heads as i32;
            let mut nb = num_blocks as i32;
            let mut ck = chunk_size as i32;
            let mut pl = pos_len as i32;
            let mut ctx = context_size as i32;
            let args = [
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut batch) as *mut i32 as *mut core::ffi::c_void,
                (&mut heads) as *mut i32 as *mut core::ffi::c_void,
                (&mut nb) as *mut i32 as *mut core::ffi::c_void,
                (&mut ck) as *mut i32 as *mut core::ffi::c_void,
                (&mut pl) as *mut i32 as *mut core::ffi::c_void,
                (&mut ctx) as *mut i32 as *mut core::ffi::c_void,
            ];
            let total = (num_heads * num_blocks * chunk_size * context_size) as i64;
            let block: u32 = 256;
            let grid: u32 = ((total + block as i64 - 1) / block as i64) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_rel_shift_audio_f32.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0, stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio attn: rel_shift launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // scores[h] layout from matrix_ac per-head loop is [num_blocks, chunk, context].
        // matrix_bd_shifted layout from rel_shift is [H, num_blocks, chunk, context].
        // Need to add: per head h, scores[h, blk, c, ctx] += bd_shifted[h, blk, c, ctx]
        // scores layout: [num_heads, num_blocks, chunk, context] (per-head loop writes
        // into scores_f32 with stride num_blocks*chunk*context per head). So both have
        // the same flat layout [H * num_blocks * chunk * context] contiguous. Single
        // add_inplace_f32 over the whole region.
        unsafe {
            let mut a = scores_f32.device_ptr();
            let mut b = matrix_bd_shifted_f32.device_ptr();
            let mut n = (num_heads * num_blocks * chunk_size * context_size) as i32;
            let args = [
                (&mut a) as *mut u64 as *mut core::ffi::c_void,
                (&mut b) as *mut u64 as *mut core::ffi::c_void,
                (&mut n) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n as i64 + block as i64 - 1) / block as i64) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_add_inplace_f32.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0, stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio attn: add_inplace launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Softcap inplace on scores_f32.
        unsafe {
            let mut x = scores_f32.device_ptr();
            let mut cap = softcap;
            let mut n = (num_heads * num_blocks * chunk_size * context_size) as i32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut cap) as *mut f32 as *mut core::ffi::c_void,
                (&mut n) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n as i64 + block as i64 - 1) / block as i64) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_tanh_softcap_inplace_f32.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0, stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio attn: softcap launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Mask invalid context positions (source row out of [0, n_tokens)).
        unsafe {
            let mut x = scores_f32.device_ptr();
            let mut nh = num_heads as i32;
            let mut nb = num_blocks as i32;
            let mut cs = chunk_size as i32;
            let mut ctx = context_size as i32;
            let mut ph = past_horizon as i32;
            let mut nt = n_tokens as i32;
            let mut inv = -1.0e9_f32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nb) as *mut i32 as *mut core::ffi::c_void,
                (&mut cs) as *mut i32 as *mut core::ffi::c_void,
                (&mut ctx) as *mut i32 as *mut core::ffi::c_void,
                (&mut ph) as *mut i32 as *mut core::ffi::c_void,
                (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                (&mut inv) as *mut f32 as *mut core::ffi::c_void,
            ];
            let total = (num_heads * num_blocks * chunk_size * context_size) as i64;
            let block: u32 = 256;
            let grid: u32 = ((total + block as i64 - 1) / block as i64) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_apply_audio_attn_mask_f32.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0, stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio attn: mask launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Softmax: per-row softmax over context dim. num_rows = H*NB*chunk.
        unsafe {
            let mut out = scores_f16.device_ptr();
            let mut input = scores_f32.device_ptr();
            let mut sl = context_size as i32;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut sl) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rows = (num_heads * num_blocks * chunk_size) as u32;
            let block: u32 = (context_size as u32).min(1024);
            let rc = cuLaunchKernel(
                self.fused.fn_softmax_row_f32_to_f16.raw() as CUfunction,
                rows, 1, 1, block, 1, 1, 0, stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio attn: softmax launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Transpose V_ctx from [num_blocks, context, H, D] -> V_T [num_blocks, H, D, context]
        // so the GEMM helper (which computes A * B^T effectively) gets a
        // B that, when "transposed by the helper", yields V in the
        // natural [context, D] orientation.
        let v_t_f16 = self.arena.region(
            "g4a_attn_v_t_f16",
            num_blocks * num_heads * head_dim * context_size * 2, 16)?;
        unsafe {
            let mut out = v_t_f16.device_ptr();
            let mut inp = v_ctx_f16.device_ptr();
            let mut nb = num_blocks as i32;
            let mut ctx = context_size as i32;
            let mut nh = num_heads as i32;
            let mut hd = head_dim as i32;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                (&mut nb) as *mut i32 as *mut core::ffi::c_void,
                (&mut ctx) as *mut i32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.fused.fn_transpose_v_chunked_f16.raw() as CUfunction,
                (context_size * num_blocks) as u32, num_heads as u32, 1,
                head_dim as u32, 1, 1,
                0, stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio attn: transpose_v_chunked launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // scores @ V — per-head batched-strided GEMM.
        // wrapper computes D[m,n] = A * B^T (the documented Q@K^T pattern).
        // We want C[chunk, head_dim] = scores[chunk, context] @ V[context, head_dim].
        // Setting B = V_T[head_dim, context] (already transposed) gives
        // wrapper(scores, V_T) -> scores @ V_T^T = scores @ V. ✓
        //
        // scores_f16 per head per block: contiguous [chunk, context], lda = context.
        // V_T per (blk, h): contiguous [head_dim, context], ldb = context.
        // Output written into [num_blocks, chunk, H, D] perm layout at per-head
        // offset h*head_dim, stride between rows H*head_dim, stride between
        // batches chunk*H*head_dim.
        let attn_out_perm_f32 = self.arena.region(
            "g4a_attn_out_perm_f32",
            num_blocks * chunk_size * num_heads * head_dim * 4, 16)?;
        for h in 0..num_heads {
            let scores_head_off =
                (h * num_blocks * chunk_size * context_size * 2) as u64;
            let v_t_head_off = (h * head_dim * context_size * 2) as u64;
            let out_head_off = (h * head_dim * 4) as u64;
            unsafe {
                self.cublaslt.f16_gemm_f32_batched_strided(
                    scores_f16.device_ptr() + scores_head_off,
                    v_t_f16.device_ptr() + v_t_head_off,
                    attn_out_perm_f32.device_ptr() + out_head_off,
                    chunk_size as i32,
                    head_dim as i32,
                    context_size as i32,
                    num_blocks as i32,
                    context_size as i32,
                    context_size as i32,
                    (num_heads * head_dim) as i32,
                    (chunk_size * context_size) as i64,
                    (num_heads * head_dim * context_size) as i64,
                    (chunk_size * num_heads * head_dim) as i64,
                    stream_raw,
                )?;
            }
        }

        // Cast to f16 and trim to n_tokens rows.
        let attn_out_perm_f16 = self.arena.region(
            "g4a_attn_out_perm_f16",
            num_blocks * chunk_size * hidden_dim * 2, 16)?;
        unsafe {
            launch_cast_f32_to_f16(&self.stream, self.fused.fn_cast_f32_to_f16,
                attn_out_perm_f32.device_ptr(),
                attn_out_perm_f16.device_ptr(),
                (num_blocks * chunk_size * hidden_dim) as i32)?;
        }

        // Post projection (clipped linear): input clamp on attn output, GEMM, output clamp.
        self.audio_clamp_inplace_f16(
            attn_out_perm_f16.device_ptr(),
            (n_tokens * hidden_dim) as i32,
            attn.post.input_min, attn.post.input_max,
        )?;
        let post_out_f32 = self.arena.region(
            "g4a_attn_post_f32", n_tokens * hidden_dim * 4, 16)?;
        unsafe {
            self.cublaslt.f16_gemm_f32(
                attn_out_perm_f16.device_ptr(),
                attn.post.weight.offset_bytes,
                post_out_f32.device_ptr(),
                n_tokens as i32,
                hidden_dim as i32,
                hidden_dim as i32,
                stream_raw,
            )?;
        }
        self.audio_clamp_inplace_f32(
            post_out_f32.device_ptr(),
            (n_tokens * hidden_dim) as i32,
            attn.post.output_min, attn.post.output_max,
        )?;

        // Cast post output to f16 into `hidden` (overwriting the input).
        // No residual add here — HF order is norm_post_attn AFTER attention,
        // residual add AFTER norm. The caller (forward_audio_block) handles
        // both. `residual` parameter retained for backward-compat callers
        // but is intentionally ignored here.
        let _ = residual;
        unsafe {
            launch_cast_f32_to_f16(&self.stream, self.fused.fn_cast_f32_to_f16,
                post_out_f32.device_ptr(), hidden,
                (n_tokens * hidden_dim) as i32)?;
        }

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn forward_audio_block(
        &self,
        block_w: &rvllm_loader::gemma4_weights::Gemma4AudioBlock,
        hidden: u64,
        scratch_residual: u64,
        pos_embed_f32: u64,
        q_scale_vec: u64,
        n_tokens: usize,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
        chunk_size: usize,
        past_horizon: usize,
        context_size: usize,
        pos_len: usize,
        softcap: f32,
        residual_weight: f32,
        eps: f32,
        conv_kernel_size: usize,
    ) -> Result<()> {
        use cudarc::driver::sys::*;
        // residual1 (FFN1 input) captured via memcpy before mutating hidden.
        unsafe {
            cuMemcpyDtoDAsync_v2(
                scratch_residual, hidden,
                (n_tokens * hidden_dim * 2) as usize,
                self.stream.raw() as CUstream);
        }
        self.forward_audio_ffn(
            &block_w.feed_forward1, scratch_residual, hidden,
            n_tokens, hidden_dim, residual_weight, eps,
            "g4a_blk_ffn1_inter_f32", "g4a_blk_ffn1_inter_f16", "g4a_blk_ffn1_out_f32",
        )?;
        // Residual for attention sub-block (pre-norm + attn + post-norm + residual,
        // no scale) — capture current hidden as the residual.
        unsafe {
            cuMemcpyDtoDAsync_v2(
                scratch_residual, hidden,
                (n_tokens * hidden_dim * 2) as usize,
                self.stream.raw() as CUstream);
        }
        // norm_pre_attn in-place.
        unsafe {
            let mut x = hidden;
            let mut g = block_w.norm_pre_attn.offset_bytes;
            let mut eps_ = eps;
            let mut d = hidden_dim as i32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut g) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_) as *mut f32 as *mut core::ffi::c_void,
                (&mut d) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.fused.fn_rmsnorm.raw() as CUfunction,
                n_tokens as u32, 1, 1,
                (hidden_dim as u32).min(1024), 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio block: norm_pre_attn launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.forward_audio_attention(
            &block_w.self_attn,
            hidden, scratch_residual, pos_embed_f32, q_scale_vec,
            n_tokens, hidden_dim, num_heads, head_dim,
            chunk_size, past_horizon, context_size, pos_len, softcap,
        )?;
        // norm_post_attn(hidden) in-place.
        unsafe {
            let mut x = hidden;
            let mut g = block_w.norm_post_attn.offset_bytes;
            let mut eps_ = eps;
            let mut d = hidden_dim as i32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut g) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_) as *mut f32 as *mut core::ffi::c_void,
                (&mut d) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.fused.fn_rmsnorm.raw() as CUfunction,
                n_tokens as u32, 1, 1,
                (hidden_dim as u32).min(1024), 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio block: norm_post_attn launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // hidden += scratch_residual (the post-FFN1 residual).
        unsafe {
            // vector_add_f16 ABI is (dst, src, n) — 3 args, dst += src.
            let mut dst = hidden;
            let mut src = scratch_residual;
            let mut n = (n_tokens * hidden_dim) as i32;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut n) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n_tokens * hidden_dim + 255) / 256) as u32;
            let rc = cuLaunchKernel(
                self.fused.fn_vector_add.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio block: attn residual add failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // LConv1D.
        unsafe {
            cuMemcpyDtoDAsync_v2(scratch_residual, hidden,
                (n_tokens * hidden_dim * 2) as usize,
                self.stream.raw() as CUstream);
        }
        self.forward_audio_lconv1d(
            &block_w.lconv1d, scratch_residual, hidden,
            n_tokens, hidden_dim, conv_kernel_size, eps,
            "g4a_blk_lconv_pre_f32", "g4a_blk_lconv_glu_f16",
            "g4a_blk_lconv_padded_f16", "g4a_blk_lconv_post_f32",
        )?;
        // FFN2 + norm_out.
        unsafe {
            cuMemcpyDtoDAsync_v2(scratch_residual, hidden,
                (n_tokens * hidden_dim * 2) as usize,
                self.stream.raw() as CUstream);
        }
        self.forward_audio_ffn(
            &block_w.feed_forward2, scratch_residual, hidden,
            n_tokens, hidden_dim, residual_weight, eps,
            "g4a_blk_ffn2_inter_f32", "g4a_blk_ffn2_inter_f16", "g4a_blk_ffn2_out_f32",
        )?;
        // norm_out in-place.
        unsafe {
            let mut x = hidden;
            let mut g = block_w.norm_out.offset_bytes;
            let mut eps_ = eps;
            let mut d = hidden_dim as i32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut g) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_) as *mut f32 as *mut core::ffi::c_void,
                (&mut d) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.fused.fn_rmsnorm.raw() as CUfunction,
                n_tokens as u32, 1, 1,
                (hidden_dim as u32).min(1024), 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio block: norm_out launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Run the entire Gemma 4 audio path: subsample -> 12 encoder blocks ->
    /// output_proj -> embed_audio.embedding_projection. Returns an arena
    /// region holding the final f16 [num_soft_tokens, text_hidden_dim] tensor
    /// + the soft-token count + the text hidden dim.
    #[cfg(feature = "cuda")]
    pub fn forward_gemma_audio_full(
        &self,
        samples_16k_mono: &[f32],
        embed_audio_projection_offset: u64,
        text_hidden_dim: usize,
    ) -> Result<crate::gemma4_audio_forward::AudioForwardOutput> {
        let audio = self.model.audio.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::cuda("audio: model.audio not loaded",
                rvllm_core::CudaErrorKind::Other, rvllm_core::CudaCtx::setup())
        })?;
        let acfg = self.arch.audio_config.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::cuda("audio: arch.audio_config missing",
                rvllm_core::CudaErrorKind::Other, rvllm_core::CudaCtx::setup())
        })?;
        let hidden_dim = acfg.hidden_size;
        let num_heads = acfg.num_attention_heads;
        let head_dim = hidden_dim / num_heads.max(1);
        let chunk_size = acfg.attention_chunk_size;
        let context_left = acfg.attention_context_left;
        let context_right = acfg.attention_context_right;
        let past_horizon = context_left.saturating_sub(1);
        let context_size = chunk_size + past_horizon + context_right;
        let pos_len = context_left;
        let softcap = acfg.attention_logit_cap;
        let residual_weight = acfg.residual_weight;
        let eps = acfg.rms_norm_eps;
        let conv_kernel_size = acfg.conv_kernel_size;

        // Subsample.
        let sub = self.forward_gemma_audio_subsample(samples_16k_mono)?;
        let n_tokens = sub.num_soft_tokens;
        // sub.device_ptr is [n_tokens, hidden_dim] f16.
        // Precompute pos_embed once (shared across layers).
        let pos_embed = self.audio_pos_embed_upload(hidden_dim, pos_len,
            "g4a_pos_embed_f32")?;
        // Per-block scratch residual buffer.
        let scratch_residual = self.arena.region(
            "g4a_blk_residual_f16", n_tokens * hidden_dim * 2, 16)?;

        let mut hidden = sub.device_ptr;
        for (li, block_w) in audio.blocks.iter().enumerate() {
            // Per-layer Q-scale: HF stores per_dim_scale per layer.
            // Use a unique scratch name per layer to avoid arena slot
            // collision corrupting the in-flight value while the
            // previous layer's attention still reads it (codex
            // round-3 fix #2). The reuse-same-slot version regressed
            // audio perception to "I cannot hear" — separate slots
            // restored progress and let layer-specific scales apply.
            let scale_name: &'static str = match li {
                0 => "g4a_q_scale_L0", 1 => "g4a_q_scale_L1",
                2 => "g4a_q_scale_L2", 3 => "g4a_q_scale_L3",
                4 => "g4a_q_scale_L4", 5 => "g4a_q_scale_L5",
                6 => "g4a_q_scale_L6", 7 => "g4a_q_scale_L7",
                8 => "g4a_q_scale_L8", 9 => "g4a_q_scale_L9",
                10 => "g4a_q_scale_L10", 11 => "g4a_q_scale_L11",
                _ => "g4a_q_scale_other",
            };
            let q_scale_vec = self.audio_q_scale_vector_upload(
                &block_w.self_attn.per_dim_scale, head_dim, scale_name,
            )?;
            if li == 0 || li == 1 || li == 11 {
                self.audio_dump_f16(hidden, n_tokens * hidden_dim,
                    &format!("audio_layer{li}_input.bin"));
            }
            self.forward_audio_block(
                block_w, hidden, scratch_residual.device_ptr(),
                pos_embed.device_ptr(), q_scale_vec.device_ptr(),
                n_tokens, hidden_dim, num_heads, head_dim,
                chunk_size, past_horizon, context_size, pos_len,
                softcap, residual_weight, eps, conv_kernel_size,
            )?;
            if li == 0 || li == 1 || li == 11 {
                self.audio_dump_f16(hidden, n_tokens * hidden_dim,
                    &format!("audio_layer{li}_output.bin"));
            }
        }
        // pos_embed table itself (cast f32->f16 on host).
        self.audio_dump_f32_as_f16(pos_embed.device_ptr(),
            pos_len * hidden_dim, "audio_pos_embed.bin");

        // output_proj: [n, hidden] -> [n, output_proj_dims=1536]
        let out_proj_dim = acfg.output_proj_dims;
        let post_f32 = self.arena.region(
            "g4a_out_proj_f32", n_tokens * out_proj_dim * 4, 16)?;
        unsafe {
            self.cublaslt.f16_gemm_f32(
                hidden,
                audio.output_proj_w.offset_bytes,
                post_f32.device_ptr(),
                n_tokens as i32,
                out_proj_dim as i32,
                hidden_dim as i32,
                self.stream.raw(),
            )?;
        }
        // Stay in f32 through output_proj_b + embedding_pre_projection_norm
        // to match HF's float arithmetic; cast to f16 only just before the
        // embedding_projection GEMM.
        unsafe {
            use cudarc::driver::sys::*;
            let mut tensor = post_f32.device_ptr();
            let mut bias = audio.output_proj_b.offset_bytes;
            let mut dim = out_proj_dim as i32;
            let args = [
                (&mut tensor) as *mut u64 as *mut core::ffi::c_void,
                (&mut bias)   as *mut u64 as *mut core::ffi::c_void,
                (&mut dim)    as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = (out_proj_dim as u32).min(1024);
            let rc = cuLaunchKernel(
                self.fused.fn_add_bias_f16_to_f32.raw() as CUfunction,
                n_tokens as u32, 1, 1, block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio: output_proj bias add (f32) failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // embedding_pre_projection_norm (parameter-free RMSNorm in f32).
        unsafe {
            use cudarc::driver::sys::*;
            let mut x = post_f32.device_ptr();
            let mut eps_ = 1.0e-6_f32;
            let mut dim = out_proj_dim as i32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps_) as *mut f32 as *mut core::ffi::c_void,
                (&mut dim) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = (out_proj_dim as u32).min(1024);
            let rc = cuLaunchKernel(
                self.fused.fn_rmsnorm_no_scale_inplace_f32.raw() as CUfunction,
                n_tokens as u32, 1, 1, block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio: embedding_pre_projection_norm (f32) failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let post_f16 = self.arena.region(
            "g4a_out_proj_f16", n_tokens * out_proj_dim * 2, 16)?;
        unsafe {
            launch_cast_f32_to_f16(&self.stream, self.fused.fn_cast_f32_to_f16,
                post_f32.device_ptr(), post_f16.device_ptr(),
                (n_tokens * out_proj_dim) as i32)?;
        }
        // embed_audio_projection: [n, 1536] -> [n, text_hidden]
        let emb_f32 = self.arena.region(
            "g4a_embed_f32", n_tokens * text_hidden_dim * 4, 16)?;
        unsafe {
            self.cublaslt.f16_gemm_f32(
                post_f16.device_ptr(),
                embed_audio_projection_offset,
                emb_f32.device_ptr(),
                n_tokens as i32,
                text_hidden_dim as i32,
                out_proj_dim as i32,
                self.stream.raw(),
            )?;
        }
        // Dump the post_f16 stage (after output_proj + bias + pre-projection rmsnorm).
        self.audio_dump_f16(post_f16.device_ptr(), n_tokens * out_proj_dim,
            "audio_after_output_proj.bin");
        let emb_f16 = self.arena.region(
            "g4a_embed_f16", n_tokens * text_hidden_dim * 2, 16)?;
        unsafe {
            launch_cast_f32_to_f16(&self.stream, self.fused.fn_cast_f32_to_f16,
                emb_f32.device_ptr(), emb_f16.device_ptr(),
                (n_tokens * text_hidden_dim) as i32)?;
        }
        self.audio_dump_f16(emb_f16.device_ptr(), n_tokens * text_hidden_dim,
            "audio_after_embed_audio.bin");
        self.stream.fence()?;
        Ok(crate::gemma4_audio_forward::AudioForwardOutput {
            device_ptr: emb_f16.device_ptr(),
            num_soft_tokens: n_tokens,
            output_proj_dims: text_hidden_dim,
        })
    }

    /// Run the audio forward and copy the spliced embedding rows
    /// to host as raw f16 bytes (shape [num_soft_tokens * text_hidden] * 2).
    /// Returns (host_bytes, num_soft_tokens, text_hidden_dim).
    #[cfg(feature = "cuda")]
    pub fn forward_gemma_audio_to_host(
        &self,
        samples_16k_mono: &[f32],
    ) -> Result<(Vec<u8>, usize, usize)> {
        use cudarc::driver::sys::*;
        let text_hidden = self.arch.hidden_size;
        let audio = self.model.audio.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::cuda("audio: model.audio not loaded",
                rvllm_core::CudaErrorKind::Other, rvllm_core::CudaCtx::setup())
        })?;
        let embed_audio_offset = audio.embed_audio_projection.offset_bytes;
        let out = self.forward_gemma_audio_full(
            samples_16k_mono, embed_audio_offset, text_hidden,
        )?;
        let row_bytes = out.output_proj_dims * 2;
        let total_bytes = out.num_soft_tokens * row_bytes;
        let mut buf = vec![0u8; total_bytes];
        unsafe {
            let rc = cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut core::ffi::c_void,
                out.device_ptr,
                total_bytes,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "audio: DtoH final output",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok((buf, out.num_soft_tokens, out.output_proj_dims))
    }

    /// Phase 3b: Gemma 4 vision tower forward.
    ///
    /// Decodes an image, runs the 27-layer SigLIP-style ViT encoder
    /// (sandwich-norm blocks), pooler (avg-pool to default_output_length
    /// + sqrt(hidden) scaling), standardize (per-channel bias+scale),
    /// and embed_vision projector (RMSNorm parameter-free + Linear
    /// 1152→5376). Returns f16 embeddings of shape
    /// `[num_pooled_tokens, 5376]` ready for splice into the post-embed
    /// text-side hidden buffer.
    #[cfg(feature = "cuda")]
    pub fn forward_gemma_vision(
        &self,
        image_bytes: &[u8],
    ) -> Result<crate::qwen36_bring_up::VisionForwardOutput> {
        use crate::qwen36_bring_up::VisionForwardOutput;
        use crate::vision_preprocess::{decode_image, preprocess_gemma, GemmaPreprocessConfig};
        use cudarc::driver::sys::*;

        let vision = self.model.vision.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::cuda(
                "vision: model.vision_tower not loaded",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;

        let img = decode_image(image_bytes).map_err(|_e| {
            rvllm_core::RvllmError::cuda(
                "vision: image decode failed",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;
        let cfg = GemmaPreprocessConfig::default();
        let pp = preprocess_gemma(&img, &cfg).map_err(|_e| {
            rvllm_core::RvllmError::cuda(
                "vision: preprocess failed",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;

        // A4b: vision-tower dims now read from arch.vision_config so
        // E4B-it (h=768, layers=16, heads=12, head_dim=64,
        // intermediate=3072) can use this same forward path. Every
        // fallback equals the original 31B `const` value, so the
        // production 31B path stays bit-identical when vision_config
        // is absent or matches 31B. ROPE_THETA isn't in the published
        // vision_config schema today (both 31B and E4B); keep the
        // hardcoded 100.0 default until rope_parameters parsing lands.
        #[allow(non_snake_case)]
        let vc = self.arch.vision_config.as_ref();
        #[allow(non_snake_case)]
        let HIDDEN: usize = vc.map(|c| c.hidden_size).unwrap_or(1152);
        #[allow(non_snake_case)]
        let INTERMEDIATE: usize = vc.map(|c| c.intermediate_size).unwrap_or(4304);
        #[allow(non_snake_case)]
        let NUM_HEADS: usize = vc.map(|c| c.num_attention_heads).unwrap_or(16);
        #[allow(non_snake_case)]
        let HEAD_DIM: usize = vc.map(|c| c.head_dim).unwrap_or(72);
        #[allow(non_snake_case)]
        let NUM_POS: usize = vc.map(|c| c.position_embedding_size).unwrap_or(10240);
        #[allow(non_snake_case)]
        let POOL_K: usize = vc.map(|c| c.pooling_kernel_size).unwrap_or(3);
        #[allow(non_snake_case)]
        let OUT_HIDDEN: usize = self.arch.hidden_size; // text-side hidden
        #[allow(non_snake_case)]
        let ROPE_THETA: f32 = 100.0;
        #[allow(non_snake_case)]
        let RMS_EPS: f32 = vc.map(|c| c.rms_norm_eps).unwrap_or(1e-6);

        // Determine n_tokens from preprocess output (count non-padding rows).
        // preprocess_gemma writes per-patch position as (col, row) =
        // (pw, ph) — see vision_preprocess.rs:370 — so position_ids[2*t]
        // is the COLUMN index and [2*t+1] is the ROW index. Earlier
        // reads had row/col swapped which made non-square images
        // (NYT 42×57) pass the wrong axes to pos_emb_lookup and rotary.
        let total_rows = pp.pixel_values.len() / (3 * 16 * 16);
        let mut active_rows = 0usize;
        for t in 0..total_rows {
            let c = pp.position_ids[2 * t];
            let r = pp.position_ids[2 * t + 1];
            if r >= 0 && c >= 0 {
                active_rows += 1;
            }
        }
        let n_tokens = active_rows;
        if n_tokens == 0 {
            return Err(rvllm_core::RvllmError::cuda(
                "vision: 0 active patches",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        // Trim padded rows from pixel_values so we work on n_tokens only.
        let patch_dim = 3 * 16 * 16; // = 768
        let mut pixel_active = Vec::with_capacity(n_tokens * patch_dim);
        let mut row_pos_i32 = Vec::with_capacity(n_tokens);
        let mut col_pos_i32 = Vec::with_capacity(n_tokens);
        for t in 0..total_rows {
            // (col, row) layout per preprocess_gemma — see comment above.
            let c = pp.position_ids[2 * t];
            let r = pp.position_ids[2 * t + 1];
            if r >= 0 && c >= 0 {
                let off = t * patch_dim;
                pixel_active.extend_from_slice(&pp.pixel_values[off..off + patch_dim]);
                row_pos_i32.push(r as i32);
                col_pos_i32.push(c as i32);
            }
        }

        // Output length: HF code uses pixel_values.shape[-2] // pool_k².
        // For our active-only path that's n_tokens // (k²).
        let n_pooled_max = n_tokens / (POOL_K * POOL_K);
        if n_pooled_max == 0 {
            return Err(rvllm_core::RvllmError::cuda(
                "vision: pool kernel too large for input (need k² ≤ n_tokens)",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }

        // Upload patches as f16. preprocess_gemma produces values in
        // [0, 1] (just /255, no centring). HF Gemma4VisionPatchEmbedder
        // .forward then does `pixel_values = 2 * (pixel_values - 0.5)`
        // (modeling_gemma4.py:566–568) BEFORE input_proj. We bake that
        // 2*x - 1 transform here so we don't need a separate kernel.
        // Codex review #1 caught this — the earlier comment claiming
        // preprocess produced [-1, 1] was wrong, contrast / text-edge
        // information was being attenuated which matches the observed
        // 'Layout erkannt, Buchstaben nicht gelesen' behaviour.
        let mut patches_f16 = Vec::with_capacity(pixel_active.len() * 2);
        for &x in &pixel_active {
            let y = 2.0 * x - 1.0;
            patches_f16.extend_from_slice(&half::f16::from_f32(y).to_le_bytes());
        }
        let patches_region = self.arena.region("g4v_patches", patches_f16.len(), 16)?;
        unsafe { patches_region.copy_from_host(&patches_f16)? };

        let row_pos_bytes = unsafe {
            std::slice::from_raw_parts(row_pos_i32.as_ptr() as *const u8, row_pos_i32.len() * 4)
        };
        let col_pos_bytes = unsafe {
            std::slice::from_raw_parts(col_pos_i32.as_ptr() as *const u8, col_pos_i32.len() * 4)
        };
        let row_pos_region = self.arena.region("g4v_rowpos", n_tokens * 4, 16)?;
        let col_pos_region = self.arena.region("g4v_colpos", n_tokens * 4, 16)?;
        unsafe { row_pos_region.copy_from_host(row_pos_bytes)? };
        unsafe { col_pos_region.copy_from_host(col_pos_bytes)? };

        let stream_raw = self.stream.raw() as u64;

        // ── Helpers (closures that fence after each launch) ──────────
        let f32_scratch = self.arena.region(
            "g4v_f32_scratch",
            n_tokens * INTERMEDIATE.max(OUT_HIDDEN) * 4,
            16,
        )?;

        let linear_no_bias = |in_dev: u64,
                              w_dev: u64,
                              out_dev: u64,
                              m: usize,
                              n: usize,
                              k: usize|
         -> Result<()> {
            #[cfg(feature = "cuda")]
            unsafe {
                self.cublaslt.f16_gemm_f32(
                    in_dev, w_dev, f32_scratch.device_ptr(),
                    m as i32, n as i32, k as i32,
                    stream_raw,
                )?;
                let n_elem = (m * n) as i32;
                let mut out = out_dev;
                let mut input = f32_scratch.device_ptr();
                let mut nn = n_elem;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    self.fused.fn_cast_f32_to_f16.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision: cast_f32_to_f16 launch failed (linear_no_bias)",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            self.stream.fence()?;
            Ok(())
        };

        let rmsnorm = |x: u64, gamma: u64, rows: usize, dim: usize| -> Result<()> {
            #[cfg(feature = "cuda")]
            unsafe {
                let mut x = x;
                let mut g = gamma;
                let mut eps = RMS_EPS;
                let mut d = dim as i32;
                let args = [
                    (&mut x) as *mut u64 as *mut core::ffi::c_void,
                    (&mut g) as *mut u64 as *mut core::ffi::c_void,
                    (&mut eps) as *mut f32 as *mut core::ffi::c_void,
                    (&mut d) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = (dim as u32).min(1024);
                let rc = cuLaunchKernel(
                    self.fused.fn_rmsnorm.raw() as CUfunction,
                    rows as u32, 1, 1,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision: rmsnorm launch failed",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            self.stream.fence()?;
            Ok(())
        };

        // vnorm_f16_kernel signature is (v, eps, head_dim) — three args,
        // not two. Passing (v, dim) before silently put `eps` into the
        // dim slot and read `head_dim` from stack garbage, producing an
        // OOB sweep in the kernel's `for (i; i<head_dim; ...)` loop. The
        // OOB faulted on Gemma vision under back-to-back launches and
        // was the root cause of the "Out Of Range Address" Xid 13 we'd
        // been masking with host-side eprintln tracepoints.
        let vnorm = |x: u64, rows: usize, dim: usize| -> Result<()> {
            #[cfg(feature = "cuda")]
            unsafe {
                let mut x = x;
                let mut eps = RMS_EPS;
                let mut d = dim as i32;
                let args = [
                    (&mut x) as *mut u64 as *mut core::ffi::c_void,
                    (&mut eps) as *mut f32 as *mut core::ffi::c_void,
                    (&mut d) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = (dim as u32).min(1024);
                let rc = cuLaunchKernel(
                    self.fused.fn_vnorm.raw() as CUfunction,
                    rows as u32, 1, 1,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision: vnorm launch failed",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            self.stream.fence()?;
            Ok(())
        };

        // GPU-side residual add: dst[i] += src[i] in f16. Replaces the
        // earlier DtoH-add-HtoD pattern which round-tripped 2400×1152
        // f16 (~5.5 MB) over UMA twice per residual (54× per image)
        // and serialised the entire vision forward on the host. Codex
        // review #3 round 4 follow-up.
        let device_residual_add = |dst: u64, src: u64| -> Result<()> {
            let n_elem = (n_tokens * HIDDEN) as i32;
            #[cfg(feature = "cuda")]
            unsafe {
                let mut d = dst;
                let mut s = src;
                let mut nn = n_elem;
                let args = [
                    (&mut d) as *mut u64 as *mut core::ffi::c_void,
                    (&mut s) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    self.fused.fn_vector_add.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision: vector_add (residual) launch failed",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            self.stream.fence()?;
            Ok(())
        };

        // fence inside one of the per-head/pooler closures), keep them.

        // Phase 5 audit infrastructure: when RVLLM_GEMMA4_VIT_DUMP_DIR is
        // set, write per-stage f16 buffers there for layer-by-layer
        // cosine comparison against an HF Gemma4VisionModel reference.
        // Mirrors the stage-dump pattern that landed Qwen vision at
        // cos=0.9999/layer. Stages chosen from Codex review round 3 #A.
        let dump_dir: Option<std::path::PathBuf> =
            std::env::var("RVLLM_GEMMA4_VIT_DUMP_DIR").ok().map(Into::into);
        let dump_stage =
            |name: &str, dev_ptr: u64, n_rows: usize, ncols: usize| -> Result<()> {
                if let Some(dir) = dump_dir.as_ref() {
                    let bytes = n_rows * ncols * 2;
                    let mut host = vec![0u8; bytes];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        // Sync first — most callers fence right after the
                        // op of interest, but doing it here too is cheap
                        // and prevents reading still-in-flight rows.
                        let _ = self.stream.fence();
                        let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                            host.as_mut_ptr() as *mut _,
                            dev_ptr,
                            bytes,
                        );
                    }
                    let path = dir.join(format!("g4v_{name}.bin"));
                    if let Err(e) = std::fs::write(&path, &host) {
                        eprintln!("[g4v-audit] write {path:?}: {e}");
                    } else {
                        eprintln!(
                            "[g4v-audit] {name} → {path:?} ({n_rows}, {ncols}) f16",
                        );
                    }
                }
                Ok(())
            };

        // Per-sub-step dump for one targeted block. Set
        // RVLLM_GEMMA4_VIT_SUBSTEP_BLK=<idx> (and RVLLM_GEMMA4_VIT_DUMP_DIR)
        // to capture every intermediate buffer inside that block — used to
        // localise a divergent kernel in a future bf16-wiring debug run by
        // diffing per-step against an HF reference dump generated by
        // `v3/tools/gemma_vision_substep_hf_dump.py`. -1 / unset = off.
        let substep_blk: i32 = std::env::var("RVLLM_GEMMA4_VIT_SUBSTEP_BLK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(-1);
        let dump_substep =
            |blk_idx: usize, name: &str, dev_ptr: u64, n_rows: usize, ncols: usize| -> Result<()> {
                if substep_blk < 0 || blk_idx as i32 != substep_blk {
                    return Ok(());
                }
                if let Some(dir) = dump_dir.as_ref() {
                    let bytes = n_rows * ncols * 2;
                    let mut host = vec![0u8; bytes];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        let _ = self.stream.fence();
                        let _ = cudarc::driver::sys::cuMemcpyDtoH_v2(
                            host.as_mut_ptr() as *mut _, dev_ptr, bytes,
                        );
                    }
                    let path = dir.join(format!("g4v_blk{blk_idx}_{name}.bin"));
                    if let Err(e) = std::fs::write(&path, &host) {
                        eprintln!("[g4v-substep] write {path:?}: {e}");
                    } else {
                        eprintln!("[g4v-substep] blk{blk_idx}/{name} → {path:?} ({n_rows}, {ncols}) f16");
                    }
                }
                Ok(())
            };

        // ── Step 1: patch_embed (linear, no bias) [N, 768] → [N, 1152] ─
        let hidden_bytes = n_tokens * HIDDEN * 2;
        let hidden_region = self.arena.region("g4v_hidden", hidden_bytes, 16)?;
        linear_no_bias(
            patches_region.device_ptr(),
            vision.patch_embedder_input_proj.offset_bytes,
            hidden_region.device_ptr(),
            n_tokens, HIDDEN, 768,
        )?;
        dump_stage("patch_embed_linear", hidden_region.device_ptr(), n_tokens, HIDDEN)?;

        // ── Step 2: 2D position embedding lookup + add. ─────────────
        #[cfg(feature = "cuda")]
        unsafe {
            // Per HF (modeling_gemma4.py:550), pixel_position_ids is
            // (x, y) = (col, row), so position_table[0] is indexed by
            // col and position_table[1] by row. Pass col_pos as axis_0
            // and row_pos as axis_1. Codex review #2 caught the swap.
            let mut h = hidden_region.device_ptr();
            let mut tab = vision.patch_embedder_pos_table.offset_bytes;
            let mut a0 = col_pos_region.device_ptr();
            let mut a1 = row_pos_region.device_ptr();
            let mut np = NUM_POS as i32;
            let mut hd = HIDDEN as i32;
            let args = [
                (&mut h) as *mut u64 as *mut core::ffi::c_void,
                (&mut tab) as *mut u64 as *mut core::ffi::c_void,
                (&mut a0) as *mut u64 as *mut core::ffi::c_void,
                (&mut a1) as *mut u64 as *mut core::ffi::c_void,
                (&mut np) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.fused.fn_vit_pos_emb_lookup_2d_f16.raw() as CUfunction,
                n_tokens as u32, 1, 1,
                256, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "vision: pos_emb_lookup launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;
        dump_stage("posemb", hidden_region.device_ptr(), n_tokens, HIDDEN)?;


        // ── Step 3: build cos/sin tables for 2D rotary. ──────────────
        // Gemma 4 vision rotary (modeling_gemma4.py:705):
        //   for axis i ∈ {0, 1}:
        //       freqs_i = inv_freq * position_ids[..., i]
        //       cos_i   = cat([freqs_i, freqs_i], dim=-1).cos()    [N, 36]
        //   cos = cat([cos_0, cos_1], dim=-1)                      [N, 72]
        // pixel_position_ids is (col, row), so axis 0 = COL, axis 1 = ROW.
        // Final layout per token (Codex review #3):
        //   [0..18]   cos(col * inv[k])
        //   [18..36]  cos(col * inv[k])   (cat-of-itself mirror)
        //   [36..54]  cos(row * inv[k])
        //   [54..72]  cos(row * inv[k])   (cat-of-itself mirror)
        // Earlier code had row,row,col,col — wrong axis assignment.
        let inv_freq_dim = HEAD_DIM / 4; // 18
        let inv_theta: Vec<f32> = (0..inv_freq_dim)
            .map(|k| 1.0 / ROPE_THETA.powf(2.0 * k as f32 / (HEAD_DIM as f32 / 2.0)))
            .collect();
        let mut cos_host = vec![0u8; n_tokens * HEAD_DIM * 2];
        let mut sin_host = vec![0u8; n_tokens * HEAD_DIM * 2];
        for t in 0..n_tokens {
            let row = row_pos_i32[t] as f32;
            let col = col_pos_i32[t] as f32;
            for k in 0..inv_freq_dim {
                let a_col = col * inv_theta[k];
                let a_row = row * inv_theta[k];
                let cos_col = half::f16::from_f32(a_col.cos()).to_le_bytes();
                let sin_col = half::f16::from_f32(a_col.sin()).to_le_bytes();
                let cos_row = half::f16::from_f32(a_row.cos()).to_le_bytes();
                let sin_row = half::f16::from_f32(a_row.sin()).to_le_bytes();
                let rb = t * HEAD_DIM;
                // Chunk 0 (channels [0, 36)): col-axis, mirrored at offset 18.
                for &mirror in &[0usize, inv_freq_dim] {
                    let off = (rb + mirror + k) * 2;
                    cos_host[off] = cos_col[0]; cos_host[off + 1] = cos_col[1];
                    sin_host[off] = sin_col[0]; sin_host[off + 1] = sin_col[1];
                }
                // Chunk 1 (channels [36, 72)): row-axis, mirrored at offset 54.
                for &mirror in &[2 * inv_freq_dim, 3 * inv_freq_dim] {
                    let off = (rb + mirror + k) * 2;
                    cos_host[off] = cos_row[0]; cos_host[off + 1] = cos_row[1];
                    sin_host[off] = sin_row[0]; sin_host[off + 1] = sin_row[1];
                }
            }
        }
        let cos_region = self.arena.region("g4v_cos", cos_host.len(), 16)?;
        let sin_region = self.arena.region("g4v_sin", sin_host.len(), 16)?;
        unsafe {
            cos_region.copy_from_host(&cos_host)?;
            sin_region.copy_from_host(&sin_host)?;
        }


        // ── Step 4: 27-block sandwich-norm encoder loop. ─────────────
        let normed = self.arena.region("g4v_normed", n_tokens * HIDDEN * 2, 16)?;
        let q_buf = self.arena.region("g4v_q", n_tokens * HIDDEN * 2, 16)?;
        let k_buf = self.arena.region("g4v_k", n_tokens * HIDDEN * 2, 16)?;
        let v_buf = self.arena.region("g4v_v", n_tokens * HIDDEN * 2, 16)?;
        let attn_out = self.arena.region("g4v_attn_out", n_tokens * HIDDEN * 2, 16)?;
        let proj_out = self.arena.region("g4v_proj_out", n_tokens * HIDDEN * 2, 16)?;
        // Batched-strided attention scratch (Codex review #B round 3 /
        // #4 round 4 follow-up). One [H, N, N] / [H, N, D] / [H, D, N]
        // slab each — heads packed contiguously so cuBLASLt can run the
        // 16-head QK^T and (scores @ V) GEMMs as a single strided-batch
        // launch each, instead of 32 launches per block.
        // Memory per request at NYT-sized N≈2400:
        //   scores_f32_all: 16 × N² × 4 ≈ 369 MiB
        //   scores_buf_all: 16 × N² × 2 ≈ 184 MiB
        //   v_t_all       : 16 × D × N × 2 ≈ 5.5 MiB
        //   out_f32_all   : 16 × N × D × 4 ≈ 11 MiB
        //   out_hmajor    : 16 × N × D × 2 ≈ 5.5 MiB
        // Arena is 60 GiB so this fits comfortably; gets restored at
        // the per-request scratch checkpoint.
        let scores_buf_all = self
            .arena
            .region("g4v_scores_all", NUM_HEADS * n_tokens * n_tokens * 2, 16)?;
        let scores_f32_all = self
            .arena
            .region("g4v_scores_f32_all", NUM_HEADS * n_tokens * n_tokens * 4, 16)?;
        let v_t_all = self
            .arena
            .region("g4v_vt_all", NUM_HEADS * HEAD_DIM * n_tokens * 2, 16)?;
        let out_f32_all = self
            .arena
            .region("g4v_out_f32_all", NUM_HEADS * n_tokens * HEAD_DIM * 4, 16)?;
        let out_hmajor = self
            .arena
            .region("g4v_out_hmajor", NUM_HEADS * n_tokens * HEAD_DIM * 2, 16)?;
        let mlp_g = self.arena.region("g4v_mlp_g", n_tokens * INTERMEDIATE * 2, 16)?;
        let mlp_u = self.arena.region("g4v_mlp_u", n_tokens * INTERMEDIATE * 2, 16)?;
        let mlp_d = self.arena.region("g4v_mlp_d", n_tokens * HIDDEN * 2, 16)?;

        for (blk_idx, blk) in vision.blocks.iter().enumerate() {
            // === ATTENTION sub-block (sandwich norm) =====================
            // residual_attn = hidden;
            // x = input_layernorm(hidden)
            // x = self_attn(x)
            // x = post_attention_layernorm(x)
            // hidden = residual_attn + x
            #[cfg(feature = "cuda")]
            unsafe {
                let _ = cuMemcpyDtoDAsync_v2(
                    normed.device_ptr(),
                    hidden_region.device_ptr(),
                    n_tokens * HIDDEN * 2,
                    self.stream.raw() as _,
                );
            }
            self.stream.fence()?;
            rmsnorm(normed.device_ptr(), blk.input_layernorm_w.offset_bytes, n_tokens, HIDDEN)?;
            dump_substep(blk_idx, "input_ln", normed.device_ptr(), n_tokens, HIDDEN)?;

            // q/k/v projections (no bias).
            linear_no_bias(normed.device_ptr(), blk.q_proj_w.offset_bytes,
                q_buf.device_ptr(), n_tokens, HIDDEN, HIDDEN)?;
            linear_no_bias(normed.device_ptr(), blk.k_proj_w.offset_bytes,
                k_buf.device_ptr(), n_tokens, HIDDEN, HIDDEN)?;
            linear_no_bias(normed.device_ptr(), blk.v_proj_w.offset_bytes,
                v_buf.device_ptr(), n_tokens, HIDDEN, HIDDEN)?;
            dump_substep(blk_idx, "q_proj", q_buf.device_ptr(), n_tokens, HIDDEN)?;
            dump_substep(blk_idx, "k_proj", k_buf.device_ptr(), n_tokens, HIDDEN)?;
            dump_substep(blk_idx, "v_proj", v_buf.device_ptr(), n_tokens, HIDDEN)?;

            // q_norm / k_norm: per-head RMSNorm head_dim=72 (gammas
            // pre-shifted in checkpoint). View Q/K as [N*num_heads, 72].
            rmsnorm(q_buf.device_ptr(), blk.q_norm_w.offset_bytes,
                n_tokens * NUM_HEADS, HEAD_DIM)?;
            rmsnorm(k_buf.device_ptr(), blk.k_norm_w.offset_bytes,
                n_tokens * NUM_HEADS, HEAD_DIM)?;

            // v_norm: parameter-free RMSNorm via fn_vnorm.
            vnorm(v_buf.device_ptr(), n_tokens * NUM_HEADS, HEAD_DIM)?;
            dump_substep(blk_idx, "q_norm", q_buf.device_ptr(), n_tokens, HIDDEN)?;
            dump_substep(blk_idx, "k_norm", k_buf.device_ptr(), n_tokens, HIDDEN)?;
            dump_substep(blk_idx, "v_norm", v_buf.device_ptr(), n_tokens, HIDDEN)?;

            // Apply Gemma 4 multidimensional rotary to Q and K.
            // The Qwen-style vit_rotary_2d_f16 kernel (used in
            // forward_qwen_vision) does ROTATE_HALF over the full
            // head_dim (pairing 0..36 with 36..72 globally), which
            // mixes the col-axis chunk with the row-axis chunk —
            // wrong for Gemma. The dedicated vit_rotary_gemma4_2d_f16
            // kernel rotates within each 36-channel chunk
            // independently, matching HF apply_multidimensional_rope
            // (Codex review #3).
            for &qk_ptr in &[q_buf.device_ptr(), k_buf.device_ptr()] {
                #[cfg(feature = "cuda")]
                unsafe {
                    let mut x = qk_ptr;
                    let mut cos = cos_region.device_ptr();
                    let mut sin = sin_region.device_ptr();
                    let mut nh = NUM_HEADS as i32;
                    let mut hd_i = HEAD_DIM as i32;
                    let args = [
                        (&mut x) as *mut u64 as *mut core::ffi::c_void,
                        (&mut cos) as *mut u64 as *mut core::ffi::c_void,
                        (&mut sin) as *mut u64 as *mut core::ffi::c_void,
                        (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                        (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let rc = cuLaunchKernel(
                        self.fused.fn_vit_rotary_gemma4_2d_f16.raw() as CUfunction,
                        n_tokens as u32, NUM_HEADS as u32, 1,
                        (HEAD_DIM / 4) as u32, 1, 1,   // 18 threads = chunk_size/2
                        0, self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "vision: gemma rotary launch failed",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
            }
            self.stream.fence()?;
            dump_substep(blk_idx, "q_rot", q_buf.device_ptr(), n_tokens, HIDDEN)?;
            dump_substep(blk_idx, "k_rot", k_buf.device_ptr(), n_tokens, HIDDEN)?;

            // Per-head attention: Q*K^T → softmax → scores @ V.
            // Gemma vision: scaling = 1.0 (no 1/sqrt(d)) — config says so.
            // Use extract_head_f16 kernel for the gather (one launch per
            // head) instead of N per-token DtoD calls. With 27 blocks ×
            // 16 heads × N=2400 tokens, the per-token loop produced
            // millions of tiny CUDA driver calls and froze for minutes.
            let _extract_head = |dst: u64, src: u64, head_idx: usize| -> Result<()> {
                #[cfg(feature = "cuda")]
                unsafe {
                    let mut o = dst;
                    let mut i = src;
                    let mut hi = head_idx as i32;
                    let mut nh = NUM_HEADS as i32;
                    let mut hd = HEAD_DIM as i32;
                    let args = [
                        (&mut o) as *mut u64 as *mut core::ffi::c_void,
                        (&mut i) as *mut u64 as *mut core::ffi::c_void,
                        (&mut hi) as *mut i32 as *mut core::ffi::c_void,
                        (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                        (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let rc = cuLaunchKernel(
                        self.fused.fn_extract_head_f16.raw() as CUfunction,
                        n_tokens as u32, 1, 1,
                        HEAD_DIM as u32, 1, 1,
                        0, self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "vision: extract_head launch failed",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                Ok(())
            };
            let _scatter_head = |dst: u64, src: u64, head_idx: usize| -> Result<()> {
                #[cfg(feature = "cuda")]
                unsafe {
                    let mut o = dst;
                    let mut i = src;
                    let mut hi = head_idx as i32;
                    let mut nh = NUM_HEADS as i32;
                    let mut hd = HEAD_DIM as i32;
                    let args = [
                        (&mut o) as *mut u64 as *mut core::ffi::c_void,
                        (&mut i) as *mut u64 as *mut core::ffi::c_void,
                        (&mut hi) as *mut i32 as *mut core::ffi::c_void,
                        (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                        (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let rc = cuLaunchKernel(
                        self.fused.fn_scatter_head_f16.raw() as CUfunction,
                        n_tokens as u32, 1, 1,
                        HEAD_DIM as u32, 1, 1,
                        0, self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "vision: scatter_head launch failed",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                Ok(())
            };
            // === Batched-strided attention pipeline ===================
            // Replaces a 16-head Python-style loop (with 6 launches per
            // head per block: extract×3, QK^T GEMM, softmax, V-transpose,
            // (scores@V) GEMM, cast, scatter) with a constant number of
            // launches per block:
            //   1) batched QK^T            (1 cuBLASLt strided-batched call)
            //   2) batched softmax-f32→f16 (1 kernel,  grid = N×H rows)
            //   3) batched transpose V     (1 kernel,  grid = (N, H))
            //   4) batched scores @ V^T    (1 cuBLASLt strided-batched call)
            //   5) batched cast f32→f16    (1 kernel)
            //   6) scatter all heads       (1 kernel,  grid = (N, H))
            //
            // Q/K are read directly from the [N, H*D] interleaved
            // q_buf/k_buf via cuBLASLt's strided-batch view (lda = H*D,
            // stride_a = D). V needs an actual transpose to [H, D, N]
            // because the scores @ V step needs V_T per head contiguous.

            // (1) batched QK^T → scores_f32_all [H, N, N]
            #[cfg(feature = "cuda")]
            unsafe {
                self.cublaslt.f16_gemm_f32_batched_strided(
                    q_buf.device_ptr(),
                    k_buf.device_ptr(),
                    scores_f32_all.device_ptr(),
                    n_tokens as i32,                // m = N (Q rows)
                    n_tokens as i32,                // n = N (K rows = scores cols)
                    HEAD_DIM as i32,                // k = D
                    NUM_HEADS as i32,               // batch = H
                    (NUM_HEADS * HEAD_DIM) as i32,  // lda = H*D (interleaved)
                    (NUM_HEADS * HEAD_DIM) as i32,  // ldb = H*D
                    n_tokens as i32,                // ldd = N (head-major dense)
                    HEAD_DIM as i64,                // stride_a = D between heads
                    HEAD_DIM as i64,                // stride_b = D
                    (n_tokens * n_tokens) as i64,   // stride_d = N*N
                    stream_raw,
                )?;
            }
            self.stream.fence()?;

            // (2) batched softmax f32→f16: scores_f32_all [H, N, N]
            //     → scores_buf_all [H, N, N]. Launch with grid = N*H rows
            //     so the existing per-row kernel handles all heads at once.
            #[cfg(feature = "cuda")]
            unsafe {
                let mut out = scores_buf_all.device_ptr();
                let mut input = scores_f32_all.device_ptr();
                let mut sl = n_tokens as i32;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input) as *mut u64 as *mut core::ffi::c_void,
                    (&mut sl) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = (n_tokens as u32).min(1024);
                let rc = cuLaunchKernel(
                    self.fused.fn_softmax_row_f32_to_f16.raw() as CUfunction,
                    (n_tokens * NUM_HEADS) as u32, 1, 1,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision: batched softmax launch failed",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            self.stream.fence()?;

            // (3) transpose V from interleaved [N, H*D] to head-major
            //     [H, D, N] via the dedicated transpose_heads_v kernel.
            #[cfg(feature = "cuda")]
            unsafe {
                let mut out = v_t_all.device_ptr();
                let mut in_p = v_buf.device_ptr();
                let mut nh = NUM_HEADS as i32;
                let mut hd = HEAD_DIM as i32;
                let mut nt = n_tokens as i32;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut in_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                ];
                let rc = cuLaunchKernel(
                    self.fused.fn_transpose_heads_v_f16.raw() as CUfunction,
                    n_tokens as u32, NUM_HEADS as u32, 1,
                    HEAD_DIM as u32, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision: transpose_heads_v launch failed",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            self.stream.fence()?;

            // (4) batched scores @ V^T → out_f32_all [H, N, D].
            //     scores_buf_all per head [N, N] contiguous: lda = N,
            //     stride_a = N². V_T per head [D, N] contiguous:
            //     ldb = N, stride_b = D*N. Output per head [N, D]:
            //     ldd = D, stride_d = N*D.
            #[cfg(feature = "cuda")]
            unsafe {
                self.cublaslt.f16_gemm_f32_batched_strided(
                    scores_buf_all.device_ptr(),
                    v_t_all.device_ptr(),
                    out_f32_all.device_ptr(),
                    n_tokens as i32,                // m = N
                    HEAD_DIM as i32,                // n = D
                    n_tokens as i32,                // k = N
                    NUM_HEADS as i32,
                    n_tokens as i32,                // lda = N
                    n_tokens as i32,                // ldb = N
                    HEAD_DIM as i32,                // ldd = D
                    (n_tokens * n_tokens) as i64,   // stride_a = N²
                    (HEAD_DIM * n_tokens) as i64,   // stride_b = D*N
                    (n_tokens * HEAD_DIM) as i64,   // stride_d = N*D
                    stream_raw,
                )?;
            }
            self.stream.fence()?;

            // (5) cast out_f32_all → out_hmajor (H × N × D elements).
            #[cfg(feature = "cuda")]
            unsafe {
                let n_elem = (NUM_HEADS * n_tokens * HEAD_DIM) as i32;
                let mut out = out_hmajor.device_ptr();
                let mut input = out_f32_all.device_ptr();
                let mut nn = n_elem;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    self.fused.fn_cast_f32_to_f16.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision: batched cast (scores@V) launch failed",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            self.stream.fence()?;

            // (6) scatter out_hmajor [H, N, D] → attn_out [N, H*D].
            #[cfg(feature = "cuda")]
            unsafe {
                let mut out = attn_out.device_ptr();
                let mut in_p = out_hmajor.device_ptr();
                let mut nh = NUM_HEADS as i32;
                let mut hd = HEAD_DIM as i32;
                let mut nt = n_tokens as i32;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut in_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                ];
                let rc = cuLaunchKernel(
                    self.fused.fn_scatter_heads_f16.raw() as CUfunction,
                    n_tokens as u32, NUM_HEADS as u32, 1,
                    HEAD_DIM as u32, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision: scatter_heads launch failed",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            self.stream.fence()?;

            dump_substep(blk_idx, "attn_out", attn_out.device_ptr(), n_tokens, HIDDEN)?;

            // o_proj (no bias).
            linear_no_bias(attn_out.device_ptr(), blk.o_proj_w.offset_bytes,
                proj_out.device_ptr(), n_tokens, HIDDEN, HIDDEN)?;
            dump_substep(blk_idx, "o_proj", proj_out.device_ptr(), n_tokens, HIDDEN)?;

            // post_attention_layernorm on proj_out.
            rmsnorm(proj_out.device_ptr(), blk.post_attention_layernorm_w.offset_bytes,
                n_tokens, HIDDEN)?;
            dump_substep(blk_idx, "post_attn_ln", proj_out.device_ptr(), n_tokens, HIDDEN)?;

            // hidden = hidden + proj_out (residual_attn).
            device_residual_add(hidden_region.device_ptr(), proj_out.device_ptr())?;
            dump_substep(blk_idx, "post_attn_resid", hidden_region.device_ptr(), n_tokens, HIDDEN)?;

            // === FFN sub-block (sandwich norm) ============================
            // residual_ffn = hidden;
            // x = pre_feedforward_layernorm(hidden)
            // x = mlp(x) = down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))
            // x = post_feedforward_layernorm(x)
            // hidden = residual_ffn + x
            #[cfg(feature = "cuda")]
            unsafe {
                let _ = cuMemcpyDtoDAsync_v2(
                    normed.device_ptr(),
                    hidden_region.device_ptr(),
                    n_tokens * HIDDEN * 2,
                    self.stream.raw() as _,
                );
            }
            self.stream.fence()?;
            rmsnorm(normed.device_ptr(), blk.pre_feedforward_layernorm_w.offset_bytes,
                n_tokens, HIDDEN)?;
            dump_substep(blk_idx, "pre_ff_ln", normed.device_ptr(), n_tokens, HIDDEN)?;

            linear_no_bias(normed.device_ptr(), blk.gate_proj_w.offset_bytes,
                mlp_g.device_ptr(), n_tokens, INTERMEDIATE, HIDDEN)?;
            linear_no_bias(normed.device_ptr(), blk.up_proj_w.offset_bytes,
                mlp_u.device_ptr(), n_tokens, INTERMEDIATE, HIDDEN)?;
            dump_substep(blk_idx, "gate_proj", mlp_g.device_ptr(), n_tokens, INTERMEDIATE)?;
            dump_substep(blk_idx, "up_proj", mlp_u.device_ptr(), n_tokens, INTERMEDIATE)?;

            // gelu_tanh(gate) * up → mlp_g (in-place).
            #[cfg(feature = "cuda")]
            unsafe {
                let n_elem = (n_tokens * INTERMEDIATE) as i32;
                let mut out = mlp_g.device_ptr();
                let mut gate = mlp_g.device_ptr();
                let mut up = mlp_u.device_ptr();
                let mut nn = n_elem;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut gate) as *mut u64 as *mut core::ffi::c_void,
                    (&mut up) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    self.fused.fn_gelu_tanh_mul_f16.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision: gelu_tanh_mul launch failed",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            self.stream.fence()?;

            dump_substep(blk_idx, "gelu_mul", mlp_g.device_ptr(), n_tokens, INTERMEDIATE)?;

            linear_no_bias(mlp_g.device_ptr(), blk.down_proj_w.offset_bytes,
                mlp_d.device_ptr(), n_tokens, HIDDEN, INTERMEDIATE)?;
            dump_substep(blk_idx, "down_proj", mlp_d.device_ptr(), n_tokens, HIDDEN)?;

            // post_feedforward_layernorm on mlp_d.
            rmsnorm(mlp_d.device_ptr(), blk.post_feedforward_layernorm_w.offset_bytes,
                n_tokens, HIDDEN)?;
            dump_substep(blk_idx, "post_ff_ln", mlp_d.device_ptr(), n_tokens, HIDDEN)?;

            // hidden = hidden + mlp_d (residual_ffn).
            device_residual_add(hidden_region.device_ptr(), mlp_d.device_ptr())?;

            // Dump block output at strategic indices (Codex's first-pass
            // audit set: 0, 13, 26 — front, middle, back).
            if blk_idx == 0 || blk_idx == 13 || blk_idx == 26 {
                dump_stage(
                    &format!("blk{blk_idx}_out"),
                    hidden_region.device_ptr(),
                    n_tokens,
                    HIDDEN,
                )?;
            }
        }
        dump_stage("encoder_out_pre_pool", hidden_region.device_ptr(), n_tokens, HIDDEN)?;


        // ── Step 5: Pooler — avg-pool kernel=3 + sqrt(hidden) scale. ─
        // n_tokens must be divisible by k². Output rows = n_tokens / k².
        // For the simple single-image case the preprocess returns
        // patches arranged in row-major (row_pos=0..H, col_pos=0..W) so
        // we can do a straightforward 2D avg-pool on a (max_x, max_y)
        // grid. Compute max_x/y from the position arrays.
        let mut max_r: i32 = 0;
        let mut max_c: i32 = 0;
        for t in 0..n_tokens {
            if row_pos_i32[t] > max_r { max_r = row_pos_i32[t]; }
            if col_pos_i32[t] > max_c { max_c = col_pos_i32[t]; }
        }
        let grid_rows = (max_r as usize) + 1;
        let grid_cols = (max_c as usize) + 1;
        // HF Gemma4VisionPooler asserts `k² * output_length == input_seq_len`
        // (modeling_gemma4.py:592) and the resize math (gemma_aspect_resize_dims
        // uses side_mult = patch_size * pool_k = 48) is supposed to guarantee
        // both grid dims are multiples of POOL_K. Fail loudly if that ever
        // breaks; the avg-pool kernel silently drops the trailing rows/cols
        // and that drift is far more painful to diagnose later than a hard
        // error here. (Codex review #3 PR-blocker.)
        if grid_rows % POOL_K != 0 || grid_cols % POOL_K != 0 {
            return Err(rvllm_core::RvllmError::cuda(
                "vision: pooled grid not divisible by POOL_K — preprocess invariant broken",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        let pooled_rows = grid_rows / POOL_K;
        let pooled_cols = grid_cols / POOL_K;
        let n_pooled = pooled_rows * pooled_cols;
        if pooled_rows == 0 || pooled_cols == 0 {
            return Err(rvllm_core::RvllmError::cuda(
                "vision: pooled grid is empty (POOL_K too large)",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        // f32 pooler buffer: stays in f32 across avg-pool → sqrt-scale →
        // standardize. Narrowed back to f16 only at the very end of
        // standardize (where std_scale has divided peak magnitudes back
        // into f16-safe range). Audit option 1 in
        // v3/GEMMA_VISION_AUDIT.md — recovers the 31/256 rows that
        // overflowed f16 (became inf) when the post-encoder peak
        // (~2752) was multiplied by sqrt(1152) ≈ 33.94.
        let pooled_region_f32 =
            self.arena.region("g4v_pooled_f32", n_pooled * HIDDEN * 4, 16)?;
        let pooled_region = self.arena.region("g4v_pooled", n_pooled * HIDDEN * 2, 16)?;

        // avg-pool (f16 in → f32 out)
        #[cfg(feature = "cuda")]
        unsafe {
            let mut out = pooled_region_f32.device_ptr();
            let mut input = hidden_region.device_ptr();
            let mut gh = grid_rows as i32;
            let mut gw = grid_cols as i32;
            let mut hd = HIDDEN as i32;
            let mut k = POOL_K as i32;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut gh) as *mut i32 as *mut core::ffi::c_void,
                (&mut gw) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut k) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.fused.fn_vit_avgpool_f16_to_f32.raw() as CUfunction,
                n_pooled as u32, 1, 1,
                256, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "vision: vit_avgpool_f16_to_f32 launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;


        // sqrt(hidden_size) scale on the f32 buffer.
        let sqrt_h = (HIDDEN as f32).sqrt();
        #[cfg(feature = "cuda")]
        unsafe {
            let mut x = pooled_region_f32.device_ptr();
            let mut s = sqrt_h;
            let mut nn = (n_pooled * HIDDEN) as i32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut s) as *mut f32 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = ((nn as u32 + block - 1) / block, 1u32, 1u32);
            let rc = cuLaunchKernel(
                self.fused.fn_scale_inplace_f32.raw() as CUfunction,
                grid.0, grid.1, grid.2,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "vision: scale_inplace_f32 (sqrt(hidden)) launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;


        // ── Step 6: Standardize OR plain narrow ──────────────────────
        // 31B: (f32 - std_bias_f16) * std_scale_f16 → f16 via the
        //      vit_standardize_f32_to_f16 kernel. std_scale weights
        //      are <1.0 here so the multiply tames the post-pool
        //      magnitudes back into f16 range.
        // E4B (vision_config.standardize=false): no std_bias /
        //      std_scale tensors on disk. The pooler-bridge f32 path
        //      already produces values close to f16 range (no
        //      sqrt(hidden) inflation that 31B needs to undo), so a
        //      plain f32→f16 cast is the correct fallback.
        #[cfg(feature = "cuda")]
        if let (Some(std_bias), Some(std_scale)) =
            (vision.std_bias.as_ref(), vision.std_scale.as_ref())
        {
            unsafe {
                let mut out = pooled_region.device_ptr();
                let mut x = pooled_region_f32.device_ptr();
                let mut bias = std_bias.offset_bytes;
                let mut scale = std_scale.offset_bytes;
                let mut hd = HIDDEN as i32;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut x) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bias) as *mut u64 as *mut core::ffi::c_void,
                    (&mut scale) as *mut u64 as *mut core::ffi::c_void,
                    (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                ];
                let rc = cuLaunchKernel(
                    self.fused.fn_vit_standardize_f32_to_f16.raw() as CUfunction,
                    n_pooled as u32, 1, 1,
                    256, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision: vit_standardize_f32_to_f16 launch failed",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        } else {
            // E4B path: plain f32 → f16 narrow over n_pooled * HIDDEN.
            #[cfg(feature = "cuda")]
            unsafe {
                let n = (n_pooled * HIDDEN) as i32;
                let mut out = pooled_region.device_ptr();
                let mut input = pooled_region_f32.device_ptr();
                let mut nn = n;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = ((n as u32 + block - 1) / block, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    self.fused.fn_cast_f32_to_f16.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision: cast_f32_to_f16 launch failed (standardize-skip)",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        self.stream.fence()?;
        dump_stage("standardized", pooled_region.device_ptr(), n_pooled, HIDDEN)?;


        // ── Step 7: embed_vision = projection(parameter-free RMSNorm(x)) ─
        // First parameter-free RMSNorm in place on pooled_region.
        vnorm(pooled_region.device_ptr(), n_pooled, HIDDEN)?;

        // Linear [HIDDEN → OUT_HIDDEN] no bias.
        let final_region = self.arena.region("g4v_final", n_pooled * OUT_HIDDEN * 2, 16)?;
        linear_no_bias(
            pooled_region.device_ptr(),
            vision.embed_vision_projection.offset_bytes,
            final_region.device_ptr(),
            n_pooled, OUT_HIDDEN, HIDDEN,
        )?;
        dump_stage("post_projection", final_region.device_ptr(), n_pooled, OUT_HIDDEN)?;


        // ── Step 8: DtoH the final embeddings. ──────────────────────
        // Explicit stream fence before DtoH: cuMemcpyDtoH_v2 blocks on
        // the host but does not implicitly synchronise a non-default
        // stream, so without this the linear projection / RMSNorm above
        // could still be in flight when the DtoH kicks off, producing
        // garbage / out-of-bounds reads.
        self.stream.fence()?;
        let out_bytes_count = n_pooled * OUT_HIDDEN * 2;
        let mut out_bytes = vec![0u8; out_bytes_count];
        #[cfg(feature = "cuda")]
        unsafe {
            let _ = cuMemcpyDtoH_v2(
                out_bytes.as_mut_ptr() as *mut _,
                final_region.device_ptr(),
                out_bytes_count,
            );
        }

        Ok(VisionForwardOutput {
            data: out_bytes,
            num_tokens: n_pooled,
            hidden_dim: OUT_HIDDEN,
            grid_thw: [1, pooled_rows as u32, pooled_cols as u32],
        })
    }
}

fn bytemuck_cast_i32(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytemuck_cast_u16(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}

// ─── B6b audio kernel launchers ─────────────────────────────────────
//
// One thin wrapper per PTX entry-point. Each takes the kernel function
// handle resolved at bring-up + the device pointers + the dim ints the
// kernel reads. Launch params are chosen to match the kernel's grid /
// block contract documented in the .cu source.

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn launch_im2col_3x3_s2p1_f16(
    stream: &Stream,
    kernel: rvllm_kernels::KernelFn,
    input: u64,
    output: u64,
    in_ch: i32,
    h_in: i32,
    w_in: i32,
    h_out: i32,
    w_out: i32,
) -> Result<()> {
    use cudarc::driver::sys::*;
    let spatial = (h_out as i64) * (w_out as i64);
    if spatial <= 0 {
        return Ok(());
    }
    let block_x: u32 = 256;
    let grid_x: u32 = ((spatial as u64 + block_x as u64 - 1) / block_x as u64) as u32;
    let grid_y: u32 = (in_ch as u32) * 9;
    let mut input = input;
    let mut output = output;
    let mut in_ch = in_ch;
    let mut h_in = h_in;
    let mut w_in = w_in;
    let mut h_out = h_out;
    let mut w_out = w_out;
    let args = [
        (&mut input)  as *mut u64 as *mut core::ffi::c_void,
        (&mut output) as *mut u64 as *mut core::ffi::c_void,
        (&mut in_ch)  as *mut i32 as *mut core::ffi::c_void,
        (&mut h_in)   as *mut i32 as *mut core::ffi::c_void,
        (&mut w_in)   as *mut i32 as *mut core::ffi::c_void,
        (&mut h_out)  as *mut i32 as *mut core::ffi::c_void,
        (&mut w_out)  as *mut i32 as *mut core::ffi::c_void,
    ];
    let rc = cuLaunchKernel(
        kernel.raw() as CUfunction,
        grid_x, grid_y, 1,
        block_x, 1, 1,
        0,
        stream.raw() as CUstream,
        args.as_ptr() as *mut *mut core::ffi::c_void,
        core::ptr::null_mut(),
    );
    if rc != CUresult::CUDA_SUCCESS {
        return Err(rvllm_core::RvllmError::cuda(
            "im2col_3x3_s2p1_f16 launch failed",
            rvllm_core::CudaErrorKind::LaunchFailed,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
unsafe fn launch_layernorm_relu_chw_f16(
    stream: &Stream,
    kernel: rvllm_kernels::KernelFn,
    x: u64,
    gamma: u64,
    eps: f32,
    c: i32,
    h: i32,
    w: i32,
) -> Result<()> {
    use cudarc::driver::sys::*;
    let pix = (h as i64) * (w as i64);
    if pix <= 0 || c <= 0 {
        return Ok(());
    }
    let mut x = x;
    let mut gamma = gamma;
    let mut eps = eps;
    let mut c_ = c;
    let mut h_ = h;
    let mut w_ = w;
    let args = [
        (&mut x)     as *mut u64 as *mut core::ffi::c_void,
        (&mut gamma) as *mut u64 as *mut core::ffi::c_void,
        (&mut eps)   as *mut f32 as *mut core::ffi::c_void,
        (&mut c_)    as *mut i32 as *mut core::ffi::c_void,
        (&mut h_)    as *mut i32 as *mut core::ffi::c_void,
        (&mut w_)    as *mut i32 as *mut core::ffi::c_void,
    ];
    let rc = cuLaunchKernel(
        kernel.raw() as CUfunction,
        pix as u32, 1, 1,
        c as u32, 1, 1,
        0,
        stream.raw() as CUstream,
        args.as_ptr() as *mut *mut core::ffi::c_void,
        core::ptr::null_mut(),
    );
    if rc != CUresult::CUDA_SUCCESS {
        return Err(rvllm_core::RvllmError::cuda(
            "layernorm_relu_chw_f16 launch failed",
            rvllm_core::CudaErrorKind::LaunchFailed,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
unsafe fn launch_transpose_chw_to_hwc_f16(
    stream: &Stream,
    kernel: rvllm_kernels::KernelFn,
    src: u64,
    dst: u64,
    c: i32,
    h: i32,
    w: i32,
) -> Result<()> {
    use cudarc::driver::sys::*;
    let total = (c as i64) * (h as i64) * (w as i64);
    if total <= 0 {
        return Ok(());
    }
    let block: u32 = 256;
    let grid: u32 = ((total as u64 + block as u64 - 1) / block as u64) as u32;
    let mut src = src;
    let mut dst = dst;
    let mut c_ = c;
    let mut h_ = h;
    let mut w_ = w;
    let args = [
        (&mut src) as *mut u64 as *mut core::ffi::c_void,
        (&mut dst) as *mut u64 as *mut core::ffi::c_void,
        (&mut c_)  as *mut i32 as *mut core::ffi::c_void,
        (&mut h_)  as *mut i32 as *mut core::ffi::c_void,
        (&mut w_)  as *mut i32 as *mut core::ffi::c_void,
    ];
    let rc = cuLaunchKernel(
        kernel.raw() as CUfunction,
        grid, 1, 1,
        block, 1, 1,
        0,
        stream.raw() as CUstream,
        args.as_ptr() as *mut *mut core::ffi::c_void,
        core::ptr::null_mut(),
    );
    if rc != CUresult::CUDA_SUCCESS {
        return Err(rvllm_core::RvllmError::cuda(
            "transpose_chw_to_hwc_f16 launch failed",
            rvllm_core::CudaErrorKind::LaunchFailed,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    Ok(())
}

/// Launcher for the existing `cast_f32_to_f16_kernel` (kernels/cast_fp.cu).
/// One thread per element.
#[cfg(feature = "cuda")]
unsafe fn launch_cast_f32_to_f16(
    stream: &Stream,
    kernel: rvllm_kernels::KernelFn,
    src_f32: u64,
    dst_f16: u64,
    n: i32,
) -> Result<()> {
    use cudarc::driver::sys::*;
    if n <= 0 {
        return Ok(());
    }
    let block: u32 = 256;
    let grid: u32 = ((n as i64 + block as i64 - 1) / block as i64) as u32;
    // Kernel signature is cast_f32_to_f16_kernel(output_f16, input_f32, n).
    // The arg order MUST be (dst, src, n) — earlier launcher passed
    // (src, dst, n) so the kernel was effectively reading from the
    // uninitialized destination buffer and writing zeros into the
    // source. Subsample / output_proj / V cast were all silently
    // zeroing their outputs as a result.
    let mut dst = dst_f16;
    let mut src = src_f32;
    let mut n_ = n;
    let args = [
        (&mut dst) as *mut u64 as *mut core::ffi::c_void,
        (&mut src) as *mut u64 as *mut core::ffi::c_void,
        (&mut n_)  as *mut i32 as *mut core::ffi::c_void,
    ];
    let rc = cuLaunchKernel(
        kernel.raw() as CUfunction,
        grid, 1, 1,
        block, 1, 1,
        0,
        stream.raw() as CUstream,
        args.as_ptr() as *mut *mut core::ffi::c_void,
        core::ptr::null_mut(),
    );
    if rc != CUresult::CUDA_SUCCESS {
        return Err(rvllm_core::RvllmError::cuda(
            "cast_f32_to_f16 launch failed",
            rvllm_core::CudaErrorKind::LaunchFailed,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    Ok(())
}

/// Launcher for `silu_inplace_f16_kernel` (kernels/silu_inplace_f16.cu).
/// One thread per element; computes x = x * sigmoid(x) in-place.
#[cfg(feature = "cuda")]
unsafe fn launch_silu_inplace_f16(
    stream: &Stream,
    kernel: rvllm_kernels::KernelFn,
    x: u64,
    n: i32,
) -> Result<()> {
    use cudarc::driver::sys::*;
    if n <= 0 {
        return Ok(());
    }
    let block: u32 = 256;
    let grid: u32 = ((n as i64 + block as i64 - 1) / block as i64) as u32;
    let mut x = x;
    let mut n_ = n;
    let args = [
        (&mut x)  as *mut u64 as *mut core::ffi::c_void,
        (&mut n_) as *mut i32 as *mut core::ffi::c_void,
    ];
    let rc = cuLaunchKernel(
        kernel.raw() as CUfunction,
        grid, 1, 1,
        block, 1, 1,
        0,
        stream.raw() as CUstream,
        args.as_ptr() as *mut *mut core::ffi::c_void,
        core::ptr::null_mut(),
    );
    if rc != CUresult::CUDA_SUCCESS {
        return Err(rvllm_core::RvllmError::cuda(
            "silu_inplace_f16 launch failed",
            rvllm_core::CudaErrorKind::LaunchFailed,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    Ok(())
}

/// Launcher for `glu_split_sigmoid_f16_kernel`
/// (kernels/glu_split_sigmoid_f16.cu). Consumes [N, 2*H_out] f16
/// and writes [N, H_out] f16: `dst[n, h] = src[n, h] * sigmoid(src[n, H_out + h])`.
#[cfg(feature = "cuda")]
unsafe fn launch_glu_split_sigmoid_f16(
    stream: &Stream,
    kernel: rvllm_kernels::KernelFn,
    src: u64,
    dst: u64,
    n: i32,
    h_out: i32,
) -> Result<()> {
    use cudarc::driver::sys::*;
    let total = (n as i64) * (h_out as i64);
    if total <= 0 {
        return Ok(());
    }
    let block: u32 = 256;
    let grid: u32 = ((total + block as i64 - 1) / block as i64) as u32;
    let mut src = src;
    let mut dst = dst;
    let mut n_ = n;
    let mut h_ = h_out;
    let args = [
        (&mut src) as *mut u64 as *mut core::ffi::c_void,
        (&mut dst) as *mut u64 as *mut core::ffi::c_void,
        (&mut n_)  as *mut i32 as *mut core::ffi::c_void,
        (&mut h_)  as *mut i32 as *mut core::ffi::c_void,
    ];
    let rc = cuLaunchKernel(
        kernel.raw() as CUfunction,
        grid, 1, 1,
        block, 1, 1,
        0,
        stream.raw() as CUstream,
        args.as_ptr() as *mut *mut core::ffi::c_void,
        core::ptr::null_mut(),
    );
    if rc != CUresult::CUDA_SUCCESS {
        return Err(rvllm_core::RvllmError::cuda(
            "glu_split_sigmoid_f16 launch failed",
            rvllm_core::CudaErrorKind::LaunchFailed,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    Ok(())
}

fn load_gemma4_fused(
    loader: &KernelLoader,
    target: Option<rvllm_core::CompileTarget>,
) -> Result<Gemma4FusedModules> {
    let rmsnorm_mod = loader.load_ptx("fused_rmsnorm_fp8_quant")?;
    let rope_mod = loader.load_ptx("fused_rope_partial_fp8kv")?;
    let rope_partial_fp8kv_bf16in_mod = loader.load_ptx("fused_rope_partial_fp8kv_bf16in")?;
    let gelu_mod = loader.load_ptx("fused_gelu_mul_fp8_quant")?;
    let argmax_mod = loader.load_ptx("argmax")?;
    let qk_norm_mod = loader.load_ptx("fused_qk_rmsnorm")?;
    let qk_norm_bf16_mod = loader.load_ptx("fused_qk_rmsnorm_bf16")?;
    let softcap_mod = loader.load_ptx("logit_softcap")?;
    let residual_scale_mod = loader.load_ptx("residual_scale_f16")?;
    let vnorm_mod = loader.load_ptx("vnorm_f16")?;
    let vector_add_mod = loader.load_ptx("vector_add_f16")?;
    let bf16_to_f16_sat_mod = loader.load_ptx("bf16_to_f16_sat")?;
    let rmsnorm_inplace_bf16_mod = loader.load_ptx("rmsnorm_inplace_bf16")?;
    let vector_add_bf16_to_f16_mod = loader.load_ptx("vector_add_bf16_to_f16")?;
    let f32_to_bf16_mod = loader.load_ptx("f32_to_bf16")?;
    let f32_to_f16_sat_mod = loader.load_ptx("f32_to_f16_sat")?;
    // Cycle 53+ Stage 1: BF16 residual chain.
    let f16_to_bf16_mod = loader.load_ptx("f16_to_bf16")?;
    let fused_norm_add_residual_bf16_mod =
        loader.load_ptx("fused_norm_add_residual_bf16")?;
    let fused_rmsnorm_fp8_quant_bf16in_mod =
        loader.load_ptx("fused_rmsnorm_fp8_quant_bf16in")?;

    let rmsnorm_inplace_mod = loader.load_ptx("rmsnorm_inplace_f16")?;
    let fn_rmsnorm = rmsnorm_inplace_mod.get_function("rmsnorm_inplace_f16_kernel")?;
    let fn_rmsnorm_fp8_quant = rmsnorm_mod.get_function("fused_rmsnorm_fp8_quant_kernel")?;
    let fn_quantize = rmsnorm_mod.get_function("quantize_fp8_per_token_kernel")?;
    let fn_rope_partial_fp8kv = rope_mod.get_function("fused_rope_partial_fp8kv_kernel")?;
    let fn_rope_partial_fp8kv_bf16in = rope_partial_fp8kv_bf16in_mod
        .get_function("fused_rope_partial_fp8kv_bf16in_kernel")?;
    let fn_gelu_mul = gelu_mod.get_function("fused_gelu_mul_fp8_quant_kernel")?;
    let fn_argmax = argmax_mod.get_function("argmax_kernel")?;
    let fn_qk_rmsnorm = qk_norm_mod.get_function("fused_qk_rmsnorm_kernel")?;
    let fn_qk_rmsnorm_bf16 = qk_norm_bf16_mod.get_function("fused_qk_rmsnorm_bf16_kernel")?;
    let fn_softcap = softcap_mod.get_function("logit_softcap_kernel")?;
    let fn_softcap_f32 = softcap_mod.get_function("logit_softcap_f32_kernel")?;
    // Codex41-3: GPU repetition penalty.
    let repetition_penalty_mod = loader.load_ptx("repetition_penalty")?;
    let fn_apply_repetition_penalty_f32 =
        repetition_penalty_mod.get_function("apply_repetition_penalty_f32_kernel")?;
    let fn_residual_scale = residual_scale_mod.get_function("residual_scale_f16_kernel")?;
    let fn_vnorm = vnorm_mod.get_function("vnorm_f16_kernel")?;
    let fn_vector_add = vector_add_mod.get_function("vector_add_f16_kernel")?;
    let fn_bf16_to_f16_sat = bf16_to_f16_sat_mod.get_function("bf16_to_f16_sat_kernel")?;
    let fn_rmsnorm_inplace_bf16 =
        rmsnorm_inplace_bf16_mod.get_function("rmsnorm_inplace_bf16_kernel")?;
    let fn_vector_add_bf16_to_f16 =
        vector_add_bf16_to_f16_mod.get_function("vector_add_bf16_to_f16_kernel")?;
    let fn_f32_to_bf16 = f32_to_bf16_mod.get_function("f32_to_bf16_kernel")?;
    let fn_f32_to_f16_sat = f32_to_f16_sat_mod.get_function("f32_to_f16_sat_kernel")?;
    // Cycle 53+ Stage 1: BF16 residual chain function handles.
    let fn_f16_to_bf16 = f16_to_bf16_mod.get_function("f16_to_bf16_kernel")?;
    let fn_fused_norm_add_residual_bf16 =
        fused_norm_add_residual_bf16_mod.get_function("fused_norm_add_residual_bf16_kernel")?;
    let fn_fused_norm_add_residual_bf16_f16in =
        fused_norm_add_residual_bf16_mod.get_function("fused_norm_add_residual_bf16_f16in_kernel")?;
    let fn_fused_norm_add_residual_bf16_bf16in =
        fused_norm_add_residual_bf16_mod.get_function("fused_norm_add_residual_bf16_bf16in_kernel")?;
    let fn_fused_rmsnorm_fp8_quant_bf16in =
        fused_rmsnorm_fp8_quant_bf16in_mod.get_function("fused_rmsnorm_fp8_quant_bf16in_kernel")?;

    let scale_cols_f32_mod = loader.load_ptx("scale_cols_f32")?;
    let fn_scale_cols_f32 = scale_cols_f32_mod.get_function("scale_cols_f32_kernel")?;
    let scale_rows_f32_ratio_mod = loader.load_ptx("scale_rows_f32_ratio")?;
    let fn_scale_rows_f32_ratio =
        scale_rows_f32_ratio_mod.get_function("scale_rows_f32_ratio_kernel")?;

    let fused_gelu_mul_f16_mod = loader.load_ptx("fused_gelu_mul_f16")?;
    let fn_fused_gelu_mul_f16 = fused_gelu_mul_f16_mod.get_function("fused_gelu_mul_f16_kernel")?;
    let fused_gelu_mul_bf16_mod = loader.load_ptx("fused_gelu_mul_bf16")?;
    let fn_fused_gelu_mul_bf16 = fused_gelu_mul_bf16_mod.get_function("fused_gelu_mul_bf16_kernel")?;
    let gelu_tanh_mul_dual_f16_mod = loader.load_ptx("gelu_tanh_mul_dual_f16")?;
    let fn_gelu_tanh_mul_dual_f16 =
        gelu_tanh_mul_dual_f16_mod.get_function("gelu_tanh_mul_dual_f16_kernel")?;

    let fused_rope_partial_f16kv_mod = loader.load_ptx("fused_rope_partial_f16kv")?;
    let fn_fused_rope_partial_f16kv =
        fused_rope_partial_f16kv_mod.get_function("fused_rope_partial_f16kv_kernel")?;

    // `fp8_gemv.ptx` — see struct docs. The f16-input native-CVT
    // entry is gated on `__CUDA_ARCH__ >= 1000` in
    // `kernels/fp8_gemv.cu`, so we only resolve it when
    // `Fp8GemvVariant::available_for(target)` says yes.
    let fp8_gemv_mod = loader.load_ptx(rvllm_kernels::FP8_GEMV_PTX_STEM)?;
    let fn_fp8_gemv_wpr_native_f16in = match target {
        Some(t) if rvllm_kernels::Fp8GemvVariant::WprNativeF16In.available_for(t) => Some(
            fp8_gemv_mod
                .get_function(rvllm_kernels::Fp8GemvVariant::WprNativeF16In.entry_point())?,
        ),
        _ => None,
    };
    let fn_fp8_gemv_wpr_native_bf16in = match target {
        Some(t) if rvllm_kernels::Fp8GemvVariant::WprNativeBf16In.available_for(t) => Some(
            fp8_gemv_mod
                .get_function(rvllm_kernels::Fp8GemvVariant::WprNativeBf16In.entry_point())?,
        ),
        _ => None,
    };

    let fused_norm_add_residual_mod = loader.load_ptx("fused_norm_add_residual")?;
    let fn_fused_norm_add_residual =
        fused_norm_add_residual_mod.get_function("fused_norm_add_residual_kernel")?;

    let fused_norm_add_residual_f16_mod = loader.load_ptx("fused_norm_add_residual_f16")?;
    let fn_fused_norm_add_residual_f16 =
        fused_norm_add_residual_f16_mod.get_function("fused_norm_add_residual_f16_kernel")?;
    let fn_fused_norm_add_residual_f16in =
        fused_norm_add_residual_f16_mod.get_function("fused_norm_add_residual_f16in_kernel")?;

    let fused_qkv_rmsnorm_mod = loader.load_ptx("fused_qkv_rmsnorm")?;
    let fn_fused_qkv_rmsnorm =
        fused_qkv_rmsnorm_mod.get_function("fused_qkv_rmsnorm_kernel")?;
    let fused_qkv_rmsnorm_bf16_mod = loader.load_ptx("fused_qkv_rmsnorm_bf16")?;
    let fn_fused_qkv_rmsnorm_bf16 =
        fused_qkv_rmsnorm_bf16_mod.get_function("fused_qkv_rmsnorm_bf16_kernel")?;

    let scale_cols_f16_mod = loader.load_ptx("scale_cols_f16")?;
    let fn_scale_cols_f16 = scale_cols_f16_mod.get_function("scale_cols_f16_kernel")?;

    // Vision Phase 3b kernel modules.
    let layernorm_inplace_f16_mod = loader.load_ptx("layernorm_inplace_f16")?;
    let fn_layernorm_inplace_f16 =
        layernorm_inplace_f16_mod.get_function("layernorm_inplace_f16_kernel")?;
    let softmax_row_f16_mod = loader.load_ptx("softmax_row_f16")?;
    let fn_softmax_row_f16 = softmax_row_f16_mod.get_function("softmax_row_f16_kernel")?;
    let gelu_tanh_f16_mod = loader.load_ptx("gelu_tanh_f16")?;
    let fn_gelu_tanh_f16 = gelu_tanh_f16_mod.get_function("gelu_tanh_f16_kernel")?;
    let gelu_tanh_mul_f16_mod = loader.load_ptx("gelu_tanh_mul_f16")?;
    let fn_gelu_tanh_mul_f16 =
        gelu_tanh_mul_f16_mod.get_function("gelu_tanh_mul_f16_kernel")?;
    let vit_avgpool_f16_mod = loader.load_ptx("vit_avgpool_f16")?;
    let fn_vit_avgpool_f16 = vit_avgpool_f16_mod.get_function("vit_avgpool_f16_kernel")?;
    let vit_pos_emb_lookup_2d_f16_mod = loader.load_ptx("vit_pos_emb_lookup_2d_f16")?;
    let fn_vit_pos_emb_lookup_2d_f16 = vit_pos_emb_lookup_2d_f16_mod
        .get_function("vit_pos_emb_lookup_2d_f16_kernel")?;
    let transpose_2d_f16_mod = loader.load_ptx("transpose_2d_f16")?;
    let fn_transpose_2d_f16 =
        transpose_2d_f16_mod.get_function("transpose_2d_f16_kernel")?;
    // B6b: audio subsample kernels — required for the audio
    // encoder forward. Loaded here so the function pointers live
    // on `self.fused` alongside the vision kernels.
    let im2col_3x3_s2p1_f16_mod = loader.load_ptx("im2col_3x3_s2p1_f16")?;
    let fn_im2col_3x3_s2p1_f16 =
        im2col_3x3_s2p1_f16_mod.get_function("im2col_3x3_s2p1_f16_kernel")?;
    let layernorm_relu_chw_f16_mod = loader.load_ptx("layernorm_relu_chw_f16")?;
    let fn_layernorm_relu_chw_f16 =
        layernorm_relu_chw_f16_mod.get_function("layernorm_relu_chw_f16_kernel")?;
    let transpose_chw_to_hwc_f16_mod = loader.load_ptx("transpose_chw_to_hwc_f16")?;
    let fn_transpose_chw_to_hwc_f16 =
        transpose_chw_to_hwc_f16_mod.get_function("transpose_chw_to_hwc_f16_kernel")?;
    // B6c: audio encoder block helpers.
    let glu_split_sigmoid_f16_mod = loader.load_ptx("glu_split_sigmoid_f16")?;
    let fn_glu_split_sigmoid_f16 =
        glu_split_sigmoid_f16_mod.get_function("glu_split_sigmoid_f16_kernel")?;
    let silu_inplace_f16_mod = loader.load_ptx("silu_inplace_f16")?;
    let fn_silu_inplace_f16 =
        silu_inplace_f16_mod.get_function("silu_inplace_f16_kernel")?;
    let causal_conv1d_f16_mod = loader.load_ptx("causal_conv1d_f16")?;
    let fn_causal_conv1d_f16 =
        causal_conv1d_f16_mod.get_function("causal_conv1d_f16_kernel")?;
    let tanh_softcap_inplace_f32_mod = loader.load_ptx("tanh_softcap_inplace_f32")?;
    let fn_tanh_softcap_inplace_f32 =
        tanh_softcap_inplace_f32_mod.get_function("tanh_softcap_inplace_f32_kernel")?;
    let rel_shift_audio_f32_mod = loader.load_ptx("rel_shift_audio_f32")?;
    let fn_rel_shift_audio_f32 =
        rel_shift_audio_f32_mod.get_function("rel_shift_audio_f32_kernel")?;
    let scale_per_dim_f32_mod = loader.load_ptx("scale_per_dim_f32")?;
    let fn_scale_per_dim_f32 =
        scale_per_dim_f32_mod.get_function("scale_per_dim_f32_kernel")?;
    let audio_chunk_extract_context_f32_mod =
        loader.load_ptx("audio_chunk_extract_context_f32")?;
    let fn_audio_chunk_extract_context_f32 =
        audio_chunk_extract_context_f32_mod
            .get_function("audio_chunk_extract_context_f32_kernel")?;
    let add_inplace_f32_mod = loader.load_ptx("add_inplace_f32")?;
    let fn_add_inplace_f32 =
        add_inplace_f32_mod.get_function("add_inplace_f32_kernel")?;
    let transpose_v_chunked_f16_mod = loader.load_ptx("transpose_v_chunked_f16")?;
    let fn_transpose_v_chunked_f16 = transpose_v_chunked_f16_mod
        .get_function("transpose_v_chunked_f16_kernel")?;
    let scale_scalar_inplace_f32_mod = loader.load_ptx("scale_scalar_inplace_f32")?;
    let fn_scale_scalar_inplace_f32 = scale_scalar_inplace_f32_mod
        .get_function("scale_scalar_inplace_f32_kernel")?;
    let apply_audio_attn_mask_f32_mod = loader.load_ptx("apply_audio_attn_mask_f32")?;
    let fn_apply_audio_attn_mask_f32 = apply_audio_attn_mask_f32_mod
        .get_function("apply_audio_attn_mask_f32_kernel")?;
    let transpose_hwc_to_chw_f16_mod = loader.load_ptx("transpose_hwc_to_chw_f16")?;
    let fn_transpose_hwc_to_chw_f16 = transpose_hwc_to_chw_f16_mod
        .get_function("transpose_hwc_to_chw_f16_kernel")?;
    let clamp_inplace_f16_mod = loader.load_ptx("clamp_inplace_f16")?;
    let fn_clamp_inplace_f16 = clamp_inplace_f16_mod
        .get_function("clamp_inplace_f16_kernel")?;
    let clamp_inplace_f32_mod = loader.load_ptx("clamp_inplace_f32")?;
    let fn_clamp_inplace_f32 = clamp_inplace_f32_mod
        .get_function("clamp_inplace_f32_kernel")?;
    let rmsnorm_no_scale_inplace_f16_mod = loader.load_ptx("rmsnorm_no_scale_inplace_f16")?;
    let fn_rmsnorm_no_scale_inplace_f16 = rmsnorm_no_scale_inplace_f16_mod
        .get_function("rmsnorm_no_scale_inplace_f16_kernel")?;
    let rmsnorm_no_scale_inplace_f32_mod = loader.load_ptx("rmsnorm_no_scale_inplace_f32")?;
    let fn_rmsnorm_no_scale_inplace_f32 = rmsnorm_no_scale_inplace_f32_mod
        .get_function("rmsnorm_no_scale_inplace_f32_kernel")?;
    let add_bias_f16_to_f32_mod = loader.load_ptx("add_bias_f16_to_f32")?;
    let fn_add_bias_f16_to_f32 = add_bias_f16_to_f32_mod
        .get_function("add_bias_f16_to_f32_kernel")?;
    let scale_inplace_f16_mod = loader.load_ptx("scale_inplace_f16")?;
    let fn_scale_inplace_f16 =
        scale_inplace_f16_mod.get_function("scale_inplace_f16_kernel")?;
    let add_bias_f16_mod = loader.load_ptx("add_bias_f16")?;
    let fn_add_bias_f16 = add_bias_f16_mod.get_function("add_bias_f16_kernel")?;
    let cast_fp_mod = loader.load_ptx("cast_fp")?;
    let fn_cast_f32_to_f16 = cast_fp_mod.get_function("cast_f32_to_f16_kernel")?;
    let vit_rotary_2d_f16_mod = loader.load_ptx("vit_rotary_2d_f16")?;
    let fn_vit_rotary_2d_f16 =
        vit_rotary_2d_f16_mod.get_function("vit_rotary_2d_f16_kernel")?;
    let vit_rotary_gemma4_2d_f16_mod = loader.load_ptx("vit_rotary_gemma4_2d_f16")?;
    let fn_vit_rotary_gemma4_2d_f16 = vit_rotary_gemma4_2d_f16_mod
        .get_function("vit_rotary_gemma4_2d_f16_kernel")?;
    let softmax_row_f32_to_f16_mod = loader.load_ptx("softmax_row_f32_to_f16")?;
    let fn_softmax_row_f32_to_f16 = softmax_row_f32_to_f16_mod
        .get_function("softmax_row_f32_to_f16_kernel")?;
    let vit_standardize_f16_mod = loader.load_ptx("vit_standardize_f16")?;
    let fn_vit_standardize_f16 =
        vit_standardize_f16_mod.get_function("vit_standardize_f16_kernel")?;
    let vit_avgpool_f16_to_f32_mod = loader.load_ptx("vit_avgpool_f16_to_f32")?;
    let fn_vit_avgpool_f16_to_f32 = vit_avgpool_f16_to_f32_mod
        .get_function("vit_avgpool_f16_to_f32_kernel")?;
    let scale_inplace_f32_mod = loader.load_ptx("scale_inplace_f32")?;
    let fn_scale_inplace_f32 = scale_inplace_f32_mod
        .get_function("scale_inplace_f32_kernel")?;
    let vit_standardize_f32_to_f16_mod = loader.load_ptx("vit_standardize_f32_to_f16")?;
    let fn_vit_standardize_f32_to_f16 = vit_standardize_f32_to_f16_mod
        .get_function("vit_standardize_f32_to_f16_kernel")?;
    let extract_head_f16_mod = loader.load_ptx("extract_head_f16")?;
    let fn_extract_head_f16 =
        extract_head_f16_mod.get_function("extract_head_f16_kernel")?;
    let fn_scatter_head_f16 =
        extract_head_f16_mod.get_function("scatter_head_f16_kernel")?;
    let transpose_heads_v_f16_mod = loader.load_ptx("transpose_heads_v_f16")?;
    let fn_transpose_heads_v_f16 = transpose_heads_v_f16_mod
        .get_function("transpose_heads_v_f16_kernel")?;
    let scatter_heads_f16_mod = loader.load_ptx("scatter_heads_f16")?;
    let fn_scatter_heads_f16 = scatter_heads_f16_mod
        .get_function("scatter_heads_f16_kernel")?;

    // NVFP4 V-rotation companion (fwht then signs). PTX may be absent
    // on older kernel trees; fall through to None and the dispatch
    // site behaves as if RVLLM_NVFP4_HADAMARD_V is off.
    let hadamard_unrotate_f16_mod = loader.load_ptx("hadamard_unrotate_f16").ok();
    let fn_hadamard_unrotate_f16 = hadamard_unrotate_f16_mod
        .as_ref()
        .and_then(|m| m.get_function("hadamard_unrotate_f16_kernel").ok());

    // Cycle 45 step 4.5c: AWQ INT4 W4A16 GEMV. Optional — `None` is fine
    // and silently disables the AWQ load path. load_gemma4_model rejects
    // an AwqConfig-bearing checkpoint when this is `None`.
    let awq_int4_gemv_f16_mod = loader.load_ptx("awq_int4_gemv_f16").ok();
    let fn_awq_int4_gemv_f16 = awq_int4_gemv_f16_mod
        .as_ref()
        .and_then(|m| m.get_function("awq_int4_gemv_f16_kernel").ok());

    // Cycle 51 step 10d.4: AWQ INT4 W4A16 GEMM (WMMA, M>1 prefill).
    // Optional — None falls through to per-token GEMV loop fallback.
    let awq_int4_gemm_sm120_wmma_mod = loader.load_ptx("awq_int4_gemm_sm120_wmma").ok();
    let fn_awq_int4_gemm_sm120_wmma = awq_int4_gemm_sm120_wmma_mod
        .as_ref()
        .and_then(|m| m.get_function("awq_int4_gemm_sm120_wmma_kernel").ok());

    Ok(Gemma4FusedModules {
        rmsnorm_mod,
        rmsnorm_inplace_mod,
        rope_mod,
        rope_partial_fp8kv_bf16in_mod,
        gelu_mod,
        argmax_mod,
        qk_norm_mod,
        qk_norm_bf16_mod,
        softcap_mod,
        repetition_penalty_mod,
        residual_scale_mod,
        vnorm_mod,
        vector_add_mod,
        bf16_to_f16_sat_mod,
        rmsnorm_inplace_bf16_mod,
        vector_add_bf16_to_f16_mod,
        f32_to_bf16_mod,
        f32_to_f16_sat_mod,
        scale_cols_f32_mod,
        scale_rows_f32_ratio_mod,
        fused_gelu_mul_f16_mod,
        fused_gelu_mul_bf16_mod,
        gelu_tanh_mul_dual_f16_mod,
        fused_rope_partial_f16kv_mod,
        fused_norm_add_residual_mod,
        // Cycle 53+ Stage 1: BF16 residual chain.
        f16_to_bf16_mod,
        fused_norm_add_residual_bf16_mod,
        fused_rmsnorm_fp8_quant_bf16in_mod,
        fn_rmsnorm,
        fn_rmsnorm_fp8_quant,
        fn_quantize,
        fn_rope_partial_fp8kv,
        fn_rope_partial_fp8kv_bf16in,
        fn_gelu_mul,
        fn_argmax,
        fn_qk_rmsnorm,
        fn_qk_rmsnorm_bf16,
        fn_softcap,
        fn_softcap_f32,
        fn_apply_repetition_penalty_f32,
        fn_residual_scale,
        fn_vnorm,
        fn_vector_add,
        fn_bf16_to_f16_sat,
        fn_rmsnorm_inplace_bf16,
        fn_vector_add_bf16_to_f16,
        // Cycle 53+ Stage 1: BF16 residual chain function handles.
        fn_f16_to_bf16,
        fn_fused_norm_add_residual_bf16,
        fn_fused_norm_add_residual_bf16_f16in,
        fn_fused_norm_add_residual_bf16_bf16in,
        fn_fused_rmsnorm_fp8_quant_bf16in,
        fn_f32_to_bf16,
        fn_f32_to_f16_sat,
        fn_scale_cols_f32,
        fn_scale_rows_f32_ratio,
        fn_fused_gelu_mul_f16,
        fn_fused_gelu_mul_bf16,
        fn_gelu_tanh_mul_dual_f16,
        fn_fused_rope_partial_f16kv,
        fn_fused_norm_add_residual,
        fn_fused_norm_add_residual_f16,
        fn_fused_norm_add_residual_f16in,
        fused_norm_add_residual_f16_mod,
        fn_fused_qkv_rmsnorm,
        fn_fused_qkv_rmsnorm_bf16,
        fused_qkv_rmsnorm_mod,
        fused_qkv_rmsnorm_bf16_mod,
        fn_scale_cols_f16,
        scale_cols_f16_mod,
        layernorm_inplace_f16_mod,
        fn_layernorm_inplace_f16,
        softmax_row_f16_mod,
        fn_softmax_row_f16,
        gelu_tanh_f16_mod,
        fn_gelu_tanh_f16,
        gelu_tanh_mul_f16_mod,
        fn_gelu_tanh_mul_f16,
        vit_avgpool_f16_mod,
        fn_vit_avgpool_f16,
        vit_pos_emb_lookup_2d_f16_mod,
        fn_vit_pos_emb_lookup_2d_f16,
        transpose_2d_f16_mod,
        fn_transpose_2d_f16,
        im2col_3x3_s2p1_f16_mod,
        fn_im2col_3x3_s2p1_f16,
        layernorm_relu_chw_f16_mod,
        fn_layernorm_relu_chw_f16,
        transpose_chw_to_hwc_f16_mod,
        fn_transpose_chw_to_hwc_f16,
        glu_split_sigmoid_f16_mod,
        fn_glu_split_sigmoid_f16,
        silu_inplace_f16_mod,
        fn_silu_inplace_f16,
        causal_conv1d_f16_mod,
        fn_causal_conv1d_f16,
        tanh_softcap_inplace_f32_mod,
        fn_tanh_softcap_inplace_f32,
        rel_shift_audio_f32_mod,
        fn_rel_shift_audio_f32,
        scale_per_dim_f32_mod,
        fn_scale_per_dim_f32,
        audio_chunk_extract_context_f32_mod,
        fn_audio_chunk_extract_context_f32,
        add_inplace_f32_mod,
        fn_add_inplace_f32,
        transpose_v_chunked_f16_mod,
        fn_transpose_v_chunked_f16,
        scale_scalar_inplace_f32_mod,
        fn_scale_scalar_inplace_f32,
        apply_audio_attn_mask_f32_mod,
        fn_apply_audio_attn_mask_f32,
        transpose_hwc_to_chw_f16_mod,
        fn_transpose_hwc_to_chw_f16,
        clamp_inplace_f16_mod,
        fn_clamp_inplace_f16,
        clamp_inplace_f32_mod,
        fn_clamp_inplace_f32,
        rmsnorm_no_scale_inplace_f16_mod,
        fn_rmsnorm_no_scale_inplace_f16,
        rmsnorm_no_scale_inplace_f32_mod,
        fn_rmsnorm_no_scale_inplace_f32,
        add_bias_f16_to_f32_mod,
        fn_add_bias_f16_to_f32,
        scale_inplace_f16_mod,
        fn_scale_inplace_f16,
        add_bias_f16_mod,
        fn_add_bias_f16,
        cast_fp_mod,
        fn_cast_f32_to_f16,
        vit_rotary_2d_f16_mod,
        fn_vit_rotary_2d_f16,
        vit_rotary_gemma4_2d_f16_mod,
        fn_vit_rotary_gemma4_2d_f16,
        softmax_row_f32_to_f16_mod,
        fn_softmax_row_f32_to_f16,
        vit_standardize_f16_mod,
        fn_vit_standardize_f16,
        vit_avgpool_f16_to_f32_mod,
        fn_vit_avgpool_f16_to_f32,
        scale_inplace_f32_mod,
        fn_scale_inplace_f32,
        vit_standardize_f32_to_f16_mod,
        fn_vit_standardize_f32_to_f16,
        extract_head_f16_mod,
        fn_extract_head_f16,
        fn_scatter_head_f16,
        transpose_heads_v_f16_mod,
        fn_transpose_heads_v_f16,
        scatter_heads_f16_mod,
        fn_scatter_heads_f16,
        fp8_gemv_mod,
        fn_fp8_gemv_wpr_native_f16in,
        fn_fp8_gemv_wpr_native_bf16in,
        hadamard_unrotate_f16_mod,
        fn_hadamard_unrotate_f16,
        awq_int4_gemv_f16_mod,
        fn_awq_int4_gemv_f16,
        awq_int4_gemm_sm120_wmma_mod,
        fn_awq_int4_gemm_sm120_wmma,
    })
}
