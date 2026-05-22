//! Phase 8 — decode-step CUDA Graph capture infrastructure (scaffold).
//!
//! This module defines [`Qwen36DecodeWorkspace`], a struct of stable
//! device-pointer scratch buffers used by the per-step decode forward.
//! The existing `Qwen36Bringup::forward_qwen36_decode_inner` allocates
//! each scratch region via `arena.region(name, bytes, align)` on every
//! call — which is graph-friendly only as long as the same arena bump
//! pointer lands at the same address every call. In practice the
//! per-call `Region::copy_from_host` does a sync HtoD on the legacy
//! default stream, AND a sync `cuMemcpyDtoH` runs inside the closer to
//! extract the argmax token — both block `cuStreamBeginCapture`.
//!
//! The Phase 8 plan (see `v3/QWEN_BATCHED_PREFILL_PLAN.md`):
//!   1. Preallocate every per-step scratch ONCE at decode-loop entry.
//!   2. Factor a `decode_step_launch_only(workspace, token_dev_ptr,
//!      pos_dev_ptr, ctx_dev_ptr)` body that does pure kernel launches
//!      against stable device pointers — no `arena.region`, no
//!      `copy_from_host`, no DtoH.
//!   3. Capture the launch body once, replay every subsequent step
//!      with updated token / pos / ctx scalars via
//!      `cuMemsetD32Async`.
//!
//! Commit 1 (this file): defines the workspace struct + allocator.
//! Forward-side wiring (commit 2) and worker capture/replay (commit 3)
//! land in follow-up commits. The struct is `pub` so test harnesses
//! can construct a workspace and verify pointer stability across
//! repeated `alloc`s on a fresh arena.

#![cfg(feature = "cuda")]

use rvllm_core::Result;
use rvllm_mem::HbmArena;

use crate::qwen36_arch::Qwen36Arch;

/// Stable per-step scratch buffer set for the Qwen 3.6 decode forward.
///
/// Every field is a raw device address (`u64`) so the captured graph
/// records launches against stable pointers, not against per-call
/// allocator bump values. Sizes that are dynamic across requests
/// (KV-cache geometry, vocab size, hidden size) are captured at
/// `alloc` time and frozen on the struct for assertion + debugging.
///
/// **Layering**: every region in this workspace lives in
/// `Qwen36Bringup::arena`, just like the existing per-step
/// `arena.region(...)` allocations. The workspace MUST be allocated
/// BEFORE the per-request scratch checkpoint, otherwise
/// `arena.restore(checkpoint)` between requests would reclaim the
/// workspace bytes and the captured-graph replay would write to
/// freshly-overwritten scratch.
///
/// **Single-sequence**: this is the single-token decode path. Each
/// scratch slot is sized for `num_tokens = 1`. Multi-token prefill
/// continues to use the per-call `arena.region` path because the
/// query length varies per request.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36DecodeWorkspace {
    // ──────────────────────────── Per-step inputs ──────────────────────────
    /// Device i32[1] holding the current token id. Updated by the
    /// outer decode loop via `cuMemsetD32Async` before each replay.
    pub token_dev: u64,
    /// Device i32[1] holding the absolute position for the current
    /// decode step. Drives RoPE + KV slot indexing.
    pub pos_dev: u64,
    /// Device i32[1] holding `context_len` (committed prompt + emitted
    /// so far). Drives FA-2 paged attention's iteration bound.
    pub ctx_dev: u64,

    // ──────────────────────────── Per-step hidden flow ─────────────────────
    /// Residual hidden buffer for this token: f16 `[1, hidden]`.
    /// Embed-gather output lands here; every block reads/writes it.
    pub hidden_dev: u64,
    /// Pre-RMSNorm residual save: f16 `[1, hidden]`. Captures the
    /// residual stream snapshot before the in-place RMSNorm so the
    /// post-attn / post-MLP residual adds use the original value.
    pub residual_save_dev: u64,
    /// Post-RMSNorm scratch: f16 `[1, hidden]`. Holds the in-place
    /// RMSNorm output for q/k/v projections (and again for MLP).
    pub normed_dev: u64,

    // ──────────────────────────── Attention scratch ────────────────────────
    /// QKV projection output, fused: f16 `[1, (q + k + v)_dim]`.
    pub qkv_dev: u64,
    /// Q after Q-norm + RoPE: f16 `[1, num_heads * head_dim]`.
    pub q_dev: u64,
    /// K after K-norm + RoPE: f16 `[1, num_kv_heads * head_dim]`.
    pub k_dev: u64,
    /// V (no norm, no RoPE): f16 `[1, num_kv_heads * head_dim]`.
    pub v_dev: u64,
    /// Attention output: f16 `[1, num_heads * head_dim]`. FA-2
    /// paged-decode kernel writes here; consumed by o_proj.
    pub attn_out_dev: u64,

    // ──────────────────────────── Linear-attn (Gated DeltaNet) scratch ────
    /// Gated DeltaNet input scratch: f16 `[1, hidden]`. Per-layer
    /// projection output before the recurrent state update.
    pub linear_in_dev: u64,
    /// Gated DeltaNet output scratch: f16 `[1, hidden]`.
    pub linear_out_dev: u64,

    // ──────────────────────────── MLP / MoE scratch ────────────────────────
    /// Router GEMV output: f32 `[1, num_experts]`. Top-k pick reads
    /// from here.
    pub router_logits_dev: u64,
    /// Top-k expert indices: i32 `[1, k]`. Driven by router argmax.
    pub topk_idx_dev: u64,
    /// Top-k expert weights (post-softmax): f32 `[1, k]`.
    pub topk_w_dev: u64,
    /// Gate projection output: f16 `[1, intermediate]`.
    pub gate_dev: u64,
    /// Up projection output: f16 `[1, intermediate]`.
    pub up_dev: u64,
    /// SiLU(gate) ⊙ up: f16 `[1, intermediate]`.
    pub silu_mul_dev: u64,
    /// Down projection output: f16 `[1, hidden]`. Residual-added
    /// back to `hidden_dev`.
    pub down_dev: u64,

    // ──────────────────────────── Final closer scratch ─────────────────────
    /// Final RMSNorm output (post-final-norm): f16 `[1, hidden]`.
    pub final_norm_dev: u64,
    /// LM-head logits: f32 `[1, vocab]`. Argmax reads from here.
    pub logits_dev: u64,
    /// Output argmax token id: i32 `[1]`. The captured graph writes
    /// here; the outer loop does a single 4-byte DtoH AFTER replay
    /// (so the captured body never holds a DtoH).
    pub argmax_token_dev: u64,

    // ──────────────────────────── Frozen geometry ──────────────────────────
    pub hidden: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub num_experts: u32,
    pub top_k: u32,
}

impl Qwen36DecodeWorkspace {
    /// Allocate every per-step scratch slot on `arena`. Call ONCE
    /// per request before taking the scratch checkpoint so
    /// subsequent `arena.restore(ck)` calls don't reclaim these
    /// regions. Returns a workspace whose pointers stay valid until
    /// the next `arena.reset()` (i.e. across the entire decode
    /// loop for one request).
    ///
    /// The struct is `Copy` so the captured-graph machinery can
    /// snapshot the pointer set without re-borrowing the arena.
    pub fn alloc(
        arena: &HbmArena<'static>,
        arch: &Qwen36Arch,
    ) -> Result<Self> {
        let hidden = arch.base.hidden_size as u32;
        // Per-expert intermediate (MoE-MLP path); the dense FFN
        // path uses the same per-expert sizing for the routed
        // experts. The shared expert uses a separate larger size
        // (`shared_expert_intermediate_size`); commit 2 will add
        // a `shared_*_dev` slot if needed.
        let intermediate = arch.moe_intermediate_size as u32;
        let vocab = arch.base.vocab_size as u32;
        let num_heads = arch.base.num_attention_heads as u32;
        let num_kv_heads = arch.base.num_key_value_heads as u32;
        let head_dim = arch.base.head_dim as u32;
        let num_experts = arch.num_experts as u32;
        let top_k = arch.num_experts_per_tok as u32;

        // Sizes: every per-step buffer is sized for num_tokens = 1.
        let hb = (hidden as usize) * 2;                // f16 hidden
        let qkv_b = (num_heads as usize + 2 * num_kv_heads as usize)
            * (head_dim as usize) * 2;
        let q_b = (num_heads as usize) * (head_dim as usize) * 2;
        let kv_b = (num_kv_heads as usize) * (head_dim as usize) * 2;
        let inter_b = (intermediate as usize) * 2;
        let router_b = (num_experts as usize) * 4;
        let topk_i_b = (top_k as usize) * 4;
        let topk_w_b = (top_k as usize) * 4;
        let logits_b = (vocab as usize) * 4;

        // ── Scalars (token / pos / ctx / argmax_out) ─────────────
        let token_dev = arena.region("qwen36_ws_token", 4, 4)?.device_ptr();
        let pos_dev = arena.region("qwen36_ws_pos", 4, 4)?.device_ptr();
        let ctx_dev = arena.region("qwen36_ws_ctx", 4, 4)?.device_ptr();
        let argmax_token_dev =
            arena.region("qwen36_ws_argmax_tok", 4, 4)?.device_ptr();

        // ── Hidden flow ─────────────────────────────────────────
        let hidden_dev = arena.region("qwen36_ws_hidden", hb, 16)?.device_ptr();
        let residual_save_dev =
            arena.region("qwen36_ws_residual_save", hb, 16)?.device_ptr();
        let normed_dev = arena.region("qwen36_ws_normed", hb, 16)?.device_ptr();

        // ── Attention ───────────────────────────────────────────
        let qkv_dev = arena.region("qwen36_ws_qkv", qkv_b, 16)?.device_ptr();
        let q_dev = arena.region("qwen36_ws_q", q_b, 16)?.device_ptr();
        let k_dev = arena.region("qwen36_ws_k", kv_b, 16)?.device_ptr();
        let v_dev = arena.region("qwen36_ws_v", kv_b, 16)?.device_ptr();
        let attn_out_dev =
            arena.region("qwen36_ws_attn_out", q_b, 16)?.device_ptr();

        // ── Linear attn ─────────────────────────────────────────
        let linear_in_dev =
            arena.region("qwen36_ws_lin_in", hb, 16)?.device_ptr();
        let linear_out_dev =
            arena.region("qwen36_ws_lin_out", hb, 16)?.device_ptr();

        // ── MLP / MoE ───────────────────────────────────────────
        let router_logits_dev =
            arena.region("qwen36_ws_router_logits", router_b, 16)?.device_ptr();
        let topk_idx_dev =
            arena.region("qwen36_ws_topk_idx", topk_i_b, 16)?.device_ptr();
        let topk_w_dev =
            arena.region("qwen36_ws_topk_w", topk_w_b, 16)?.device_ptr();
        let gate_dev =
            arena.region("qwen36_ws_gate", inter_b, 16)?.device_ptr();
        let up_dev = arena.region("qwen36_ws_up", inter_b, 16)?.device_ptr();
        let silu_mul_dev =
            arena.region("qwen36_ws_silu_mul", inter_b, 16)?.device_ptr();
        let down_dev =
            arena.region("qwen36_ws_down", hb, 16)?.device_ptr();

        // ── Closer ──────────────────────────────────────────────
        let final_norm_dev =
            arena.region("qwen36_ws_final_norm", hb, 16)?.device_ptr();
        let logits_dev =
            arena.region("qwen36_ws_logits", logits_b, 16)?.device_ptr();

        Ok(Self {
            token_dev,
            pos_dev,
            ctx_dev,
            hidden_dev,
            residual_save_dev,
            normed_dev,
            qkv_dev,
            q_dev,
            k_dev,
            v_dev,
            attn_out_dev,
            linear_in_dev,
            linear_out_dev,
            router_logits_dev,
            topk_idx_dev,
            topk_w_dev,
            gate_dev,
            up_dev,
            silu_mul_dev,
            down_dev,
            final_norm_dev,
            logits_dev,
            argmax_token_dev,
            hidden,
            intermediate,
            vocab,
            num_heads,
            num_kv_heads,
            head_dim,
            num_experts,
            top_k,
        })
    }

    /// Total bytes the workspace claims from the arena. Sum of all
    /// per-slot sizes (excludes alignment padding the arena may
    /// insert between regions). Provided as a sanity check the
    /// caller can budget the arena around.
    pub fn approx_bytes(&self) -> usize {
        let hb = (self.hidden as usize) * 2;
        let qkv_b = (self.num_heads as usize + 2 * self.num_kv_heads as usize)
            * (self.head_dim as usize) * 2;
        let q_b = (self.num_heads as usize) * (self.head_dim as usize) * 2;
        let kv_b = (self.num_kv_heads as usize) * (self.head_dim as usize) * 2;
        let inter_b = (self.intermediate as usize) * 2;
        let router_b = (self.num_experts as usize) * 4;
        let topk_i_b = (self.top_k as usize) * 4;
        let topk_w_b = (self.top_k as usize) * 4;
        let logits_b = (self.vocab as usize) * 4;
        // 4 scalar slots (token/pos/ctx/argmax) + hidden×3 +
        // qkv + q + kv×2 + attn_out + lin×2 + router + topk_i +
        // topk_w + gate + up + silu_mul + down + final_norm + logits
        4 * 4
            + hb * 3
            + qkv_b
            + q_b
            + kv_b * 2
            + q_b  // attn_out same shape as q
            + hb * 2
            + router_b
            + topk_i_b
            + topk_w_b
            + inter_b * 3
            + hb
            + hb
            + logits_b
    }
}

/// Env gate for the Phase 8 captured decode path. Returns `true`
/// when `RVLLM_QWEN36_DECODE_GRAPH` is set truthy. Default off so
/// production decode remains on the eager path until the capture
/// machinery is fully wired (commits 2-3).
pub fn qwen36_decode_graph_enabled() -> bool {
    std::env::var("RVLLM_QWEN36_DECODE_GRAPH")
        .ok()
        .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}
