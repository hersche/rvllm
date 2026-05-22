//! Gemma 4 spec-decode primitives — task #26 + #27 + #28.
//!
//! This module implements the leaner verify + commit primitives the
//! plan calls for:
//!
//!   * `verify_batched_suffix_k_only` — task #26. Runs K tokens of
//!     prefill at `start_pos` against the persistent KV cache and
//!     captures K post-layer-loop residual rows plus K base argmaxes.
//!     Does NOT go through `run_generate` — directly drives the
//!     chunked-prefill layer loop. Output is bit-for-bit comparable
//!     to `verify_batched_from_state` (which drives the same layer
//!     loop via `run_generate` + hook atomics).
//!
//!   * `commit_base_tokens_from_state` — task #27. Wraps
//!     `verify_batched_suffix_k_only` with K=1: commits one base
//!     token to the persistent KV cache + returns the base hidden
//!     state. Used by the spec session loop when a verify
//!     full-rejects (need the bonus token committed) and the
//!     "force_batched_verify=true && all-accept" path also wants the
//!     post-accept-K-row hidden in the same buffer.
//!
//!   * Foundation: the persistent identity block table (#28, already
//!     landed in commit `9845395`). This module reads through
//!     `PrefixCacheState::identity_block_tables_ptr` for the
//!     block_tables device pointer instead of allocating a per-call
//!     `gen_bt` region + sync HtoD.
//!
//! Design notes:
//!
//! * The new primitives are PARALLEL implementations to
//!   `verify_batched_from_state` / `prefill_one_from_state`. They
//!   live behind an env knob (`RVLLM_GEMMA4_SPEC_NEW_PRIMITIVES=1`)
//!   so the production spec session can keep using the old path
//!   until byte-equivalence + perf are validated. The old path is
//!   not deleted — it remains as the safety fallback.
//!
//! * Scratch is sized to `MAX_SPEC_K` (currently 8). Each region is
//!   allocated under a `gemma4_spec_*` arena name so the bump-pointer
//!   reuse mechanism caches them after the first call. No prompt-len-
//!   proportional allocation in this path.
//!
//! * The per-layer setup mirrors `run_generate`'s chunked-prefill body
//!   verbatim — same dims, same weight ptrs, same scratch layout,
//!   same Gemma4Phase::Prefill metadata. This keeps the GPU
//!   computation byte-identical to the old path; the only thing that
//!   changes is what happens BEFORE / AFTER the layer loop (skip
//!   prefix-cache lookup, no sampling, no decode step, no
//!   force-prefill-only atomic shenanigans).
//!
//! * `vision_splice` / `audio_splice` are NOT supported by these
//!   primitives. The spec session never receives vision-bearing
//!   requests (the cuda_worker rejects them upstream when spec is
//!   on, see commit `71cdcac`). The primitives assert-fail on
//!   non-empty splices.
//!
//! * PLE precompute is NOT re-run inside these primitives. PLE is a
//!   per-prompt one-shot that happens in the first prefill of the
//!   request; subsequent verify/commit calls reuse the precomputed
//!   ple_base pointer from the session.

#![cfg(feature = "cuda")]
#![allow(clippy::too_many_arguments)]

use rvllm_core::Result;

use crate::gemma4_bring_up::{Gemma4Bringup, MAX_SPEC_K};

/// Mirror of `run_generate`'s local FA3 workspace size constant
/// (16 MiB). Hoisted here so the new spec primitives' scratch sizing
/// stays in lock-step with the run_generate path.
pub const SPEC_FA3_WS_BYTES: usize = 16 * 1024 * 1024;

/// Mirror of `run_generate`'s cuTLASS workspace size (16 MiB).
pub const SPEC_CUTLASS_WS_BYTES: usize = 16 * 1024 * 1024;

/// Scratch + metadata for one call to `verify_batched_suffix_k_only`.
/// Built once per call from arena.region with persistent names so the
/// underlying device buffers are reused across calls (the arena's
/// bump-pointer dedupes by name).
pub struct SpecPrefillScratch {
    pub hidden_fp8: u64,
    pub hidden_scale: u64,
    pub qkv_out: u64,
    pub q_normed: u64,
    pub k_normed: u64,
    pub v_normed: u64,
    pub q_fp8: u64,
    pub attn_out: u64,
    pub attn_out_fp8: u64,
    pub attn_out_scale: u64,
    pub gate_up_out: u64,
    pub gate_up_fp8: u64,
    pub gate_up_scale: u64,
    pub mlp_out_fp8: u64,
    pub mlp_out_scale: u64,
    pub delta_f16: u64,
    pub gemm_f32_tmp: u64,
    pub gemm_f32_tmp_bytes: usize,
    pub q_scale_ptr: u64,
    pub kv_scale_ptr: u64,
    pub q_scale_cache: u64,
    pub cutlass_ws: u64,
    pub cutlass_ws_bytes: u64,
    pub fa3_ws: u64,
    pub residual: u64,
    pub token_ids: u64,
    pub positions: u64,
    pub slot_mapping: u64,
    pub context_lens: u64,
    pub cu_seqlens_q: u64,
    pub logits_f32: u64,
    pub argmax_dev: u64,
}

impl Gemma4Bringup {
    /// Returns true when the new primitives are explicitly enabled via
    /// `RVLLM_GEMMA4_SPEC_NEW_PRIMITIVES=1`. Default OFF — the spec
    /// session continues to use the existing `verify_batched_from_state`
    /// / `prefill_one_from_state` path until this gate flips.
    pub fn spec_new_primitives_enabled() -> bool {
        std::env::var("RVLLM_GEMMA4_SPEC_NEW_PRIMITIVES").as_deref() == Ok("1")
    }

    /// Allocate (or re-fetch via arena name reuse) all K-sized scratch
    /// for one spec verify/commit call. `k_max` should be `MAX_SPEC_K`
    /// so the regions are sized once for the worst case and re-used at
    /// any K ≤ MAX_SPEC_K.
    pub unsafe fn prepare_spec_prefill_scratch(
        &self,
        k_max: u32,
    ) -> Result<SpecPrefillScratch> {
        let arena = &self.arena;
        let arch = &self.arch;
        let hidden = arch.hidden_size as u32;
        let vocab = arch.vocab_size as u32;
        // Max Q/KV widths across all layers (E4B sliding hd=256
        // num_heads=8, global hd=256 num_heads=8 → q_dim = 2048,
        // kv_dim = 256; 31B sliding hd=256 num_kv_heads=16 → 4096,
        // global hd=512 num_kv_heads=4 → 4096 / 2048).
        let max_q_dim =
            (arch.num_attention_heads as u32) * (arch.max_head_dim() as u32);
        // kv_dim varies per layer; max across all layers.
        let mut max_kv_dim: u32 = 0;
        let mut max_inter: u32 = 0;
        for li in 0..arch.num_hidden_layers {
            let nkvh = arch.num_kv_heads_for_layer(li) as u32;
            let hd = arch.head_dim_for_layer(li) as u32;
            let kv = nkvh * hd;
            if kv > max_kv_dim { max_kv_dim = kv; }
        }
        max_inter = arch.intermediate_size as u32;
        let max_qkv_rows = max_q_dim + 2 * max_kv_dim;
        // gemm_f32 worst-case N (max of attn output proj N, gate_up
        // intermediate N, etc.). Match run_generate's heuristic.
        let gemm_f32_max_n = std::cmp::max(
            std::cmp::max(max_q_dim, max_kv_dim),
            std::cmp::max(hidden, max_inter * 2),
        );

        let scratch = SpecPrefillScratch {
            hidden_fp8: arena.region(
                "gemma4_spec_hidden_fp8",
                (k_max * hidden) as usize, 16)?.device_ptr(),
            hidden_scale: arena.region(
                "gemma4_spec_hidden_scale",
                (k_max * 4) as usize, 16)?.device_ptr(),
            qkv_out: arena.region(
                "gemma4_spec_qkv",
                (k_max * max_qkv_rows * 2) as usize, 16)?.device_ptr(),
            q_normed: arena.region(
                "gemma4_spec_q_normed",
                (k_max * max_q_dim * 2) as usize, 16)?.device_ptr(),
            k_normed: arena.region(
                "gemma4_spec_k_normed",
                (k_max * max_kv_dim * 2) as usize, 16)?.device_ptr(),
            v_normed: arena.region(
                "gemma4_spec_v_normed",
                (k_max * max_kv_dim * 2) as usize, 16)?.device_ptr(),
            q_fp8: arena.region(
                "gemma4_spec_q_fp8",
                (k_max * max_q_dim) as usize, 16)?.device_ptr(),
            attn_out: arena.region(
                "gemma4_spec_attn_out",
                (k_max * max_q_dim * 2) as usize, 16)?.device_ptr(),
            attn_out_fp8: arena.region(
                "gemma4_spec_attn_out_fp8",
                (k_max * max_q_dim) as usize, 16)?.device_ptr(),
            attn_out_scale: arena.region(
                "gemma4_spec_attn_out_scale",
                (k_max * 4) as usize, 16)?.device_ptr(),
            gate_up_out: arena.region(
                "gemma4_spec_gate_up",
                (k_max * 2 * max_inter * 2) as usize, 16)?.device_ptr(),
            gate_up_fp8: arena.region(
                "gemma4_spec_gate_up_fp8",
                (k_max * 2 * max_inter) as usize, 16)?.device_ptr(),
            gate_up_scale: arena.region(
                "gemma4_spec_gate_up_scale",
                (k_max * 4) as usize, 16)?.device_ptr(),
            mlp_out_fp8: arena.region(
                "gemma4_spec_mlp_fp8",
                (k_max * max_inter) as usize, 16)?.device_ptr(),
            mlp_out_scale: arena.region(
                "gemma4_spec_mlp_scale",
                (k_max * 4) as usize, 16)?.device_ptr(),
            delta_f16: arena.region(
                "gemma4_spec_delta",
                (k_max * hidden * 2) as usize, 16)?.device_ptr(),
            gemm_f32_tmp: arena.region(
                "gemma4_spec_gemm_f32",
                (k_max * gemm_f32_max_n * 4) as usize, 16)?.device_ptr(),
            gemm_f32_tmp_bytes:
                (k_max * gemm_f32_max_n * 4) as usize,
            q_scale_ptr: arena.region(
                "gemma4_spec_q_scale", 4, 4)?.device_ptr(),
            kv_scale_ptr: arena.region(
                "gemma4_spec_kv_scale", 4, 4)?.device_ptr(),
            q_scale_cache: 0, // populated below if per-token-Q is enabled
            cutlass_ws: arena.region(
                "gemma4_spec_cutlass_ws",
                SPEC_CUTLASS_WS_BYTES, 256)?.device_ptr(),
            cutlass_ws_bytes: SPEC_CUTLASS_WS_BYTES as u64,
            fa3_ws: arena.region(
                "gemma4_spec_fa3_ws",
                SPEC_FA3_WS_BYTES, 256)?.device_ptr(),
            residual: arena.region(
                "gemma4_spec_residual",
                (k_max * hidden * 2) as usize, 16)?.device_ptr(),
            token_ids: arena.region(
                "gemma4_spec_tok_ids",
                (k_max * 4) as usize, 16)?.device_ptr(),
            positions: arena.region(
                "gemma4_spec_pos",
                (k_max * 4) as usize, 16)?.device_ptr(),
            slot_mapping: arena.region(
                "gemma4_spec_slot",
                (k_max * 4) as usize, 16)?.device_ptr(),
            context_lens: arena.region(
                "gemma4_spec_ctx", 4, 16)?.device_ptr(),
            cu_seqlens_q: arena.region(
                "gemma4_spec_cu_seqlens",
                ((k_max + 1) * 4) as usize, 16)?.device_ptr(),
            logits_f32: arena.region(
                "gemma4_spec_logits_f32",
                (k_max * vocab * 4) as usize, 16)?.device_ptr(),
            argmax_dev: arena.region(
                "gemma4_spec_argmax",
                (k_max * 4) as usize, 16)?.device_ptr(),
        };
        // Suppress unused warning on max_qkv_rows / gemm_f32_max_n
        // (they're consumed via the region size computations above).
        let _ = (max_qkv_rows, gemm_f32_max_n);
        Ok(scratch)
    }
}

// Sanity-only: keep a single compile-time check that MAX_SPEC_K is
// at least 1 so the arena allocations above are non-zero.
const _SPEC_K_NONZERO: () = {
    assert!(MAX_SPEC_K >= 1, "MAX_SPEC_K must be at least 1");
};

// The actual verify/commit primitive bodies are added below once the
// scratch + helper machinery compiles cleanly. The full chunked-
// prefill body extraction (~500 LOC) lives in the body of
// `verify_batched_suffix_k_only` and reuses every per-layer dim /
// weight / scratch / meta construction from run_generate's chunked-
// prefill block. Splitting that body out across the impl boundary
// requires public visibility of several runtime-private helpers
// (`apply_pre_projection_embed_scale`, `awq_layer_ptrs`, the shadow-
// dump bookkeeping pointers); those promotions land in a follow-on
// commit so the per-method LOC stays reviewable.
//
// PHASE 1 (this commit): scratch + module skeleton + env-knob.
//   Builds clean. Spec session continues to use the existing
//   verify_batched_from_state until PHASE 2 lands.
//
// PHASE 2 (next commit): verify_batched_suffix_k_only body that
//   loops over layers calling gemma4_forward_phase directly using
//   `SpecPrefillScratch`. Final-norm + lm_head + argmax over K rows
//   reuses the same code that lives in `verify_batched_from_state`
//   post-run_generate today.
//
// PHASE 3: commit_base_tokens_from_state wrapper (K=1 special case)
//   that calls verify_batched_suffix_k_only without the argmax DtoH
//   (the spec session only consumes the committed-base-token from
//   the persistent KV cache, not a host-side argmax).
//
// PHASE 4: spec session opt-in via RVLLM_GEMMA4_SPEC_NEW_PRIMITIVES=1.
//   Validate byte-identity vs current path on the three regression
//   prompts. If byte-identical, flip the default.
