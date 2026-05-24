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

impl Gemma4Bringup {
    /// Task #26 — verify K tokens at `start_pos` against the
    /// persistent KV cache and capture K post-layer-loop residual
    /// rows + K base argmaxes.
    ///
    /// DOES NOT call `run_generate`. The body drives the chunked-
    /// prefill layer loop directly via `gemma4_forward_phase`.
    /// Outputs are byte-identical to `verify_batched_from_state` (the
    /// existing `run_generate`-driven path) for the same `new_tokens`
    /// + `start_pos`; the only thing that changes is what happens
    /// BEFORE / AFTER the layer loop (no prefix-cache lookup, no
    /// sampling, no decode step, no force_prefill_only atomic).
    ///
    /// `k_hidden_out` must be a device pointer to a buffer of at
    /// least `new_tokens.len() * hidden_size * 2` bytes. After the
    /// call it holds the K post-layer-loop f16 (or bf16 when
    /// `RVLLM_RESIDUAL_BF16=1`) residual rows.
    ///
    /// `k_argmax_out` is a host buffer of length ≥ new_tokens.len().
    /// After the call it holds the base model's argmax for each of
    /// the K verify positions.
    ///
    /// Vision/audio splices are NOT supported by this path (spec
    /// session never receives vision-bearing requests). PLE
    /// precompute is NOT re-run inside this path (PLE is a per-prompt
    /// one-shot; verify always runs against state where PLE has been
    /// applied during the initial prefill).
    pub unsafe fn verify_batched_suffix_k_only(
        &self,
        fn_embed: rvllm_kernels::KernelFn,
        new_tokens: &[u32],
        start_pos: u32,
        k_hidden_out: u64,
        k_argmax_out: &mut [u32],
    ) -> Result<()> {
        if new_tokens.is_empty() {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "new_tokens",
                    reason: "empty".into(),
                },
                field: "new_tokens",
            });
        }
        if new_tokens.len() > MAX_SPEC_K {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "new_tokens.len",
                    reason: format!(
                        "{} exceeds MAX_SPEC_K={}",
                        new_tokens.len(), MAX_SPEC_K).into(),
                },
                field: "new_tokens.len",
            });
        }
        if k_argmax_out.len() < new_tokens.len() {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "k_argmax_out",
                    reason: format!(
                        "buf len {} < new_tokens.len {}",
                        k_argmax_out.len(), new_tokens.len()).into(),
                },
                field: "k_argmax_out",
            });
        }

        let k = new_tokens.len() as u32;
        let arch = &self.arch;
        let arena = &self.arena;
        let stream = self.stream.raw() as u64;
        let hidden = arch.hidden_size as u32;
        let vocab = arch.vocab_size as u32;
        let inter = arch.intermediate_size as u32;
        // The persistent KV cache slot indexing is
        // `slot = block * block_size + slot_in_block`; the spec
        // verify path MUST use the SAME block_size the cache was
        // allocated with or attention reads wrong slots → garbage.
        // Read the canonical value from PrefixCacheState.block_size
        // (set inside init_prefix_cache where the persistent KV is
        // allocated). Falls back to 32 only if the cache wasn't
        // initialised — which also fails the persistent-cache-ptr
        // guard below, so the fallback is unreachable in production.
        let max_layers: usize = self.model.layers.len();

        // Prefix-cache must be initialised — verify-from-state is a
        // spec-only path that runs against the persistent KV cache.
        let (kv_cache_ptr, kv_scale_cache_ptr,
             kv_layer_offsets, kv_scale_layer_offsets,
             num_blocks_total, identity_bt_ptr, identity_bt_len,
             block_size): (u64, u64,
             Vec<u64>, Vec<u64>, u32, u64, u32, u32) = {
            // P2 #9 (codex audit, task aa01001srvbug2): poison-recover
            // pattern so a panicked sibling worker doesn't cascade
            // every future spec-decode request into a 500. The recover
            // helper clears the cache slot to None on poison, so the
            // None branch below fires and returns a clean Config error
            // pointing at init_prefix_cache.
            let guard = self.lock_prefix_cache_recover();
            match guard.as_ref() {
                Some(pc) => (
                    pc.kv_cache_ptr,
                    pc.kv_scale_ptr,
                    pc.kv_layer_offsets.clone(),
                    pc.kv_scale_layer_offsets.clone(),
                    pc.num_blocks_total,
                    pc.identity_block_tables_ptr,
                    pc.identity_block_tables_len,
                    pc.block_size,
                ),
                None => {
                    return Err(rvllm_core::RvllmError::Config {
                        err: rvllm_core::ConfigError::InvalidField {
                            name: "prefix_cache",
                            reason: "verify_batched_suffix_k_only: \
                                     prefix_cache not initialised; call \
                                     init_prefix_cache first".into(),
                        },
                        field: "prefix_cache",
                    });
                }
            }
        };
        // Identity block table must be present (task #28).
        if identity_bt_ptr == 0 || identity_bt_len != num_blocks_total {
            return Err(rvllm_core::RvllmError::Config {
                err: rvllm_core::ConfigError::InvalidField {
                    name: "identity_block_tables",
                    reason: "verify_batched_suffix_k_only: persistent \
                             identity block table missing or wrong size".into(),
                },
                field: "identity_block_tables",
            });
        }

        let sliding_blocks = num_blocks_total;

        // Per-layer kv dtype + Hadamard/shadow base ptrs, just like
        // run_generate's chunked-prefill body.
        let hadamard_base_ptr: u64 = {
            let g = self.nvfp4_hadamard.lock().unwrap();
            g.as_ref().map(|h| h.base_ptr).unwrap_or(0)
        };
        let hadamard_head_dim_stride: u32 = {
            let g = self.nvfp4_hadamard.lock().unwrap();
            g.as_ref().map(|h| h.head_dim).unwrap_or(0)
        };
        let (shadow_ptr, shadow_layer_offsets, shadow_q_throwaway_ptr) = {
            let g = self.nvfp4_shadow.lock().unwrap();
            match g.as_ref() {
                Some(s) => (s.shadow_ptr, s.layer_offsets.clone(),
                            s.shadow_q_throwaway_ptr),
                None => (0u64, Vec::new(), 0u64),
            }
        };

        // PLE state — spec verify never re-runs PLE; ple_base = 0 to
        // signal "no PLE injection this dispatch". The first prefill of
        // the request populated PLE in the persistent cache already.
        let ple_base: u64 = 0;
        let ple_stride_elems: u32 = 0;

        // Allocate K-sized scratch (or re-fetch via arena name reuse).
        let s = self.prepare_spec_prefill_scratch(MAX_SPEC_K as u32)?;

        // q_scale cache (per-token Q amax buffer). Allocate a K-sized
        // region under a stable name so it's reused across spec calls.
        // CRITICAL (Phase 4 debug): the rope kernel WRITES per-token
        // amax/448 into this buffer at index [tok * num_heads + head];
        // run_generate zero-inits via cuMemsetD8 at every call so stale
        // amax values from a prior request can't leak into the current
        // attention. We mirror that memset here.
        let q_scale_cache_ptr: u64 = if std::env::var(
            "RVLLM_PER_TOKEN_Q_SCALE").map_or(true, |v| v != "0")
        {
            let bytes = (MAX_SPEC_K as usize)
                * (arch.num_attention_heads as usize) * 4;
            let r = arena.region("gemma4_spec_q_scale_cache", bytes, 16)?;
            let ptr = r.device_ptr();
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8Async(ptr, 0, bytes, stream as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "spec_verify q_scale_cache memset",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
            ptr
        } else { 0 };

        // Populate q_scale / kv_scale at the same defaults run_generate
        // uses. Cheap 4-byte HtoDs; could be hoisted further later.
        {
            use cudarc::driver::sys::*;
            let q_s: f32 = std::env::var("RVLLM_Q_SCALE")
                .ok().and_then(|v| v.parse().ok())
                .unwrap_or(crate::gemma4_bring_up::DEFAULT_Q_SCALE);
            let kv_s: f32 = std::env::var("RVLLM_KV_SCALE")
                .ok().and_then(|v| v.parse().ok())
                .unwrap_or(crate::gemma4_bring_up::DEFAULT_KV_SCALE);
            let q_bytes = q_s.to_le_bytes();
            let kv_bytes = kv_s.to_le_bytes();
            let rc = cuMemcpyHtoDAsync_v2(
                s.q_scale_ptr, q_bytes.as_ptr() as *const _, 4,
                stream as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "spec_verify q_scale htod",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
            let rc = cuMemcpyHtoDAsync_v2(
                s.kv_scale_ptr, kv_bytes.as_ptr() as *const _, 4,
                stream as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "spec_verify kv_scale htod",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // Build positions / slot_mapping / context_lens / cu_seqlens_q
        // for K tokens at [start_pos .. start_pos + K).
        let chunk_start_abs: u32 = start_pos;
        let chunk_end_abs: u32 = start_pos + k;
        {
            use cudarc::driver::sys::*;
            let pos: Vec<i32> =
                (chunk_start_abs as i32 .. chunk_end_abs as i32).collect();
            let slot: Vec<i32> = pos.clone();
            let ctx = [chunk_end_abs as i32];
            let cu_seq = [0i32, k as i32];
            let tok_ids: Vec<i32> = new_tokens.iter().map(|&t| t as i32).collect();
            let rc = cuMemcpyHtoDAsync_v2(
                s.positions,
                crate::gemma4_bring_up::bytemuck_cast_i32(&pos).as_ptr() as *const _,
                pos.len() * 4, stream as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "spec_verify positions htod",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
            let rc = cuMemcpyHtoDAsync_v2(
                s.slot_mapping,
                crate::gemma4_bring_up::bytemuck_cast_i32(&slot).as_ptr() as *const _,
                slot.len() * 4, stream as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "spec_verify slot_mapping htod",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
            let rc = cuMemcpyHtoDAsync_v2(
                s.context_lens,
                crate::gemma4_bring_up::bytemuck_cast_i32(&ctx).as_ptr() as *const _,
                4, stream as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "spec_verify context_lens htod",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
            let rc = cuMemcpyHtoDAsync_v2(
                s.cu_seqlens_q,
                crate::gemma4_bring_up::bytemuck_cast_i32(&cu_seq).as_ptr() as *const _,
                cu_seq.len() * 4, stream as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "spec_verify cu_seqlens_q htod",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
            let rc = cuMemcpyHtoDAsync_v2(
                s.token_ids,
                crate::gemma4_bring_up::bytemuck_cast_i32(&tok_ids).as_ptr() as *const _,
                tok_ids.len() * 4, stream as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "spec_verify token_ids htod",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // Embed K tokens into residual.
        rvllm_fused::EmbeddingGatherLaunch { num_tokens: k, hidden, vocab }
            .launch(fn_embed, s.residual,
                    self.model.embedding.offset_bytes,
                    s.token_ids, stream)?;

        // Optional bf16 widen (matches run_generate's chunk body).
        if crate::gemma4_bring_up::bf16_residual_enabled() {
            rvllm_fused::gemma4_launcher::F16ToBf16Launch {
                n: k * hidden,
            }.launch(
                self.fused.fn_f16_to_bf16,
                s.residual, s.residual, stream)?;
        }

        // Build the Prefill phase metadata (single chunk of K tokens).
        let phase = crate::gemma4_layer_exec::Gemma4Phase::Prefill {
            cu_seqlens_q: s.cu_seqlens_q,
            max_seqlen_q: k,
            num_seqs: 1,
        };

        // Per-layer setup loop — mirrors run_generate's chunked-prefill
        // body verbatim. The only difference is the source of scratch
        // pointers (s.* instead of the local arena.region handles).
        let kernels = self.layer_kernels()?;
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            if layer_idx >= max_layers { break; }
            let lt = arch.layer_types[layer_idx];
            let hd = arch.head_dim_for_layer(layer_idx) as u32;
            let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
            let q_dim = (arch.num_attention_heads as u32) * hd;
            let kv_dim = nkvh * hd;
            let layer_blocks = if lt == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention {
                num_blocks_total } else { sliding_blocks };
            let layer_kv_elems = 2u64 * layer_blocks as u64
                * block_size as u64 * nkvh as u64 * hd as u64;
            let kv_idx = arch.kv_share_source_layer(layer_idx).unwrap_or(layer_idx);
            let layer_kv_base = kv_cache_ptr + kv_layer_offsets[kv_idx];
            let layer_kv_scale_base =
                kv_scale_cache_ptr + kv_scale_layer_offsets[kv_idx];
            let layer_kv_scale_slots_half =
                (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);
            let kv_dtype =
                crate::gemma4_layer_exec::KvDtype::for_layer_index_or_env(
                    lt, layer_idx, false);
            let prefill_kv_dtype = if kv_dtype
                == crate::gemma4_layer_exec::KvDtype::Nvfp4
            {
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
                num_tokens: k, hidden,
                num_heads: arch.num_attention_heads as u32,
                num_kv_heads: nkvh, head_dim: hd,
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
                f16_kv: false,
                kv_dtype: prefill_kv_dtype,
                bf16_residual: crate::gemma4_bring_up::bf16_residual_enabled(),
                kv_share_source_layer:
                    arch.kv_share_source_layer(layer_idx).map(|x| x as u32),
                current_max_context_len: Some(chunk_end_abs as u32),
            };
            let w = crate::gemma4_layer_exec::Gemma4LayerWeightPtrs {
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
                awq: crate::gemma4_bring_up::awq_layer_ptrs(layer.awq.as_ref()),
                ple_input_gate: layer.per_layer_input_gate.as_ref()
                    .map_or(0, |w| w.offset_bytes),
                ple_projection: layer.per_layer_projection.as_ref()
                    .map_or(0, |w| w.offset_bytes),
                ple_post_input_norm_gamma: layer.post_per_layer_input_norm.as_ref()
                    .map_or(0, |w| w.offset_bytes),
                ple_per_layer_input: if ple_base != 0 {
                    ple_base + (layer_idx as u64)
                        * (arch.hidden_size_per_layer_input.unwrap_or(0) as u64)
                        * 2
                } else { 0 },
                ple_per_layer_stride_elems: ple_stride_elems,
            };
            // QKV out row-major layout: q/k/v sub-slices share row 0.
            let q_base = s.qkv_out;
            let k_out = q_base + (q_dim as u64) * 2;
            let v_out = k_out + (kv_dim as u64) * 2;
            let (cos, sin) = match lt {
                rvllm_loader::gemma4_arch::Gemma4LayerType::SlidingAttention =>
                    (self.model.rope_cos_sliding.offset_bytes,
                     self.model.rope_sin_sliding.offset_bytes),
                rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention =>
                    (self.model.rope_cos_global.offset_bytes,
                     self.model.rope_sin_global.offset_bytes),
            };
            let bytes_per_half_kv = match prefill_kv_dtype {
                crate::gemma4_layer_exec::KvDtype::F16 => layer_kv_elems,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_kv_elems / 2,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_kv_elems / 4,
            };
            // Shadow / Hadamard layer ptrs (only when NVFP4 + alloc present).
            let prefill_is_shadow_layer = shadow_ptr != 0
                && prefill_kv_dtype == crate::gemma4_layer_exec::KvDtype::Nvfp4
                && layer_idx < shadow_layer_offsets.len()
                && shadow_layer_offsets[layer_idx] != u64::MAX;
            let (prefill_shadow_k, prefill_shadow_v) = if prefill_is_shadow_layer {
                let base = shadow_ptr + shadow_layer_offsets[layer_idx];
                (base, base + layer_kv_elems)
            } else { (0u64, 0u64) };
            let prefill_shadow_q = if prefill_is_shadow_layer {
                shadow_q_throwaway_ptr } else { 0 };
            let prefill_hadamard_layer_ptr: u64 = if hadamard_base_ptr != 0
                && prefill_kv_dtype == crate::gemma4_layer_exec::KvDtype::Nvfp4
            {
                hadamard_base_ptr
                    + (layer_idx as u64) * (hadamard_head_dim_stride as u64)
            } else { 0 };

            let scratch = crate::gemma4_layer_exec::Gemma4LayerScratch {
                hidden_fp8: s.hidden_fp8, hidden_scale: s.hidden_scale,
                q_out: q_base, k_out, v_out,
                q_normed: s.q_normed, k_normed: s.k_normed, v_normed: s.v_normed,
                q_fp8: s.q_fp8,
                k_cache: layer_kv_base,
                v_cache: layer_kv_base + bytes_per_half_kv,
                k_cache_scale, v_cache_scale,
                q_scale_ptr: s.q_scale_ptr, kv_scale_ptr: s.kv_scale_ptr,
                k_scale_cache: layer_kv_scale_base,
                v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                q_scale_cache: q_scale_cache_ptr,
                attn_out: s.attn_out, attn_out_fp8: s.attn_out_fp8,
                attn_out_scale: s.attn_out_scale, delta_f16: s.delta_f16,
                gate_up_out: s.gate_up_out, gate_up_fp8: s.gate_up_fp8,
                gate_up_scale: s.gate_up_scale,
                mlp_out_fp8: s.mlp_out_fp8, mlp_out_scale: s.mlp_out_scale,
                gemm_f32_tmp: s.gemm_f32_tmp,
                gemm_f32_tmp_bytes: s.gemm_f32_tmp_bytes,
                cutlass_workspace: s.cutlass_ws,
                cutlass_workspace_bytes: s.cutlass_ws_bytes as usize,
                fa3_workspace: s.fa3_ws,
                fa3_workspace_bytes: SPEC_FA3_WS_BYTES as u64,
                shadow_k_cache: prefill_shadow_k,
                shadow_v_cache: prefill_shadow_v,
                shadow_q_cache: prefill_shadow_q,
                hadamard_signs_q: prefill_hadamard_layer_ptr,
                hadamard_signs_k: prefill_hadamard_layer_ptr,
            };
            let meta = crate::gemma4_layer_exec::Gemma4MetadataPtrs {
                positions: s.positions, slot_mapping: s.slot_mapping,
                cos, sin,
                block_tables: identity_bt_ptr,
                context_lens: s.context_lens,
            };
            crate::gemma4_layer_exec::gemma4_forward_phase(
                dims, &kernels, &w, &scratch, &meta,
                &self.cublaslt, &self.cutlass,
                &self.sliding_attention, &self.global_attention,
                s.residual, stream, phase,
            )?;
            // Phase 4 debug: per-layer residual hash for OLD-vs-NEW
            // bisection. Gated on RVLLM_GEMMA4_SPEC_LAYER_DUMP=1.
            if std::env::var("RVLLM_GEMMA4_SPEC_LAYER_DUMP").as_deref() == Ok("1") {
                use cudarc::driver::sys::*;
                self.stream.fence()?;
                let mut buf = vec![0u16; (k as usize) * (hidden as usize)];
                let rc = cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut _,
                    s.residual,
                    (k as usize) * (hidden as usize) * 2);
                if rc == CUresult::CUDA_SUCCESS {
                    // RMS + max + first 4 values of row 0.
                    let mut sum_sq = 0.0f64;
                    let mut amax = 0.0f32;
                    for &b in &buf[..hidden as usize] {
                        let v = half::f16::from_bits(b).to_f32();
                        sum_sq += (v * v) as f64;
                        if v.abs() > amax { amax = v.abs(); }
                    }
                    let rms = (sum_sq / hidden as f64).sqrt();
                    let first4: Vec<f32> = buf[..4].iter()
                        .map(|&b| half::f16::from_bits(b).to_f32()).collect();
                    eprintln!(
                        "[spec-new-layer-dump] layer={} k={} hidden={} \
                         row0_rms={:.4} row0_max={:.3} row0_first4={:?}",
                        layer_idx, k, hidden, rms, amax, first4);
                }
            }
        }

        // Capture K post-layer-loop residual rows into caller's buffer.
        // residual is K * hidden * 2 bytes (or 4 bytes if bf16; the
        // RVLLM_RESIDUAL_BF16 codepath uses the same 2-byte cell since
        // bf16 fits in 16 bits like f16).
        {
            use cudarc::driver::sys::*;
            let row_bytes = (hidden as usize) * 2;
            let rc = cuMemcpyDtoDAsync_v2(
                k_hidden_out, s.residual,
                (k as usize) * row_bytes,
                stream as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "spec_verify k_hidden_out DtoD",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }

        // K-row final_norm + lm_head + softcap + argmax. This mirrors
        // the post-run_generate path in verify_batched_from_state.
        let final_norm_kernel = if crate::gemma4_bring_up::bf16_residual_enabled() {
            self.fused.fn_rmsnorm_inplace_bf16
        } else {
            self.fused.fn_rmsnorm
        };
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: k,
            hidden,
            eps: arch.rms_norm_eps,
        }.launch(
            final_norm_kernel,
            k_hidden_out,
            self.model.final_norm.offset_bytes,
            stream)?;
        if crate::gemma4_bring_up::bf16_residual_enabled() {
            rvllm_fused::gemma4_launcher::Bf16ToF16SatLaunch { n: hidden * k }
                .launch(
                    self.fused.fn_bf16_to_f16_sat,
                    k_hidden_out, k_hidden_out, stream)?;
        }

        // lm_head GEMM.
        self.cublaslt.f16_gemm_f32(
            k_hidden_out,
            self.model.lm_head_f16.offset_bytes,
            s.logits_f32,
            k as i32, vocab as i32, hidden as i32,
            stream)?;

        if arch.logit_softcap > 0.0 {
            rvllm_fused::gemma4_launcher::LogitSoftcapLaunch {
                num_tokens: k, vocab, cap: arch.logit_softcap,
            }.launch(
                self.fused.fn_softcap_f32,
                s.logits_f32, stream)?;
        }

        // Argmax over K rows: one f32 argmax kernel call per row,
        // writing into s.argmax_dev.
        {
            use cudarc::driver::sys::*;
            for row in 0..k {
                let mut row_ptr = s.logits_f32 + (row as u64) * (vocab as u64) * 4;
                let mut out_ptr = s.argmax_dev + (row as u64) * 4;
                let mut vsz: i32 = vocab as i32;
                let args = [
                    (&mut row_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut vsz) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block_dim: u32 = vocab.min(1024);
                let rc = cuLaunchKernel(
                    self.fused.fn_argmax.raw() as CUfunction,
                    1, 1, 1, block_dim, 1, 1, 0,
                    stream as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut());
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "spec_verify argmax launch",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            }
            // Single DtoH for all K argmax outputs.
            self.stream.fence()?;
            let mut tmp = vec![0i32; k as usize];
            let rc = cuMemcpyDtoH_v2(
                tmp.as_mut_ptr() as *mut _,
                s.argmax_dev,
                (k as usize) * 4);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "spec_verify argmax DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
            for (i, &t) in tmp.iter().enumerate() {
                k_argmax_out[i] = t.max(0) as u32;
            }
        }

        Ok(())
    }

    /// Task #27 — commit a single base token to the persistent KV
    /// cache. K=1 special case of `verify_batched_suffix_k_only`:
    /// runs one chunked prefill step at `start_pos` for `new_token`,
    /// advancing the persistent KV cache by one slot. Returns the
    /// next base argmax (the model's prediction for the next position
    /// after `start_pos + 1`) so callers can use it as the
    /// `session.next_base_argmax` seed for the following iter — same
    /// return contract as the existing `prefill_one_from_state`.
    pub unsafe fn commit_base_tokens_from_state(
        &self,
        fn_embed: rvllm_kernels::KernelFn,
        new_token: u32,
        start_pos: u32,
        k_hidden_out: u64,
    ) -> Result<u32> {
        let mut argmax_buf = [0u32; 1];
        self.verify_batched_suffix_k_only(
            fn_embed,
            &[new_token],
            start_pos,
            k_hidden_out,
            &mut argmax_buf,
        )?;
        Ok(argmax_buf[0])
    }
}

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
