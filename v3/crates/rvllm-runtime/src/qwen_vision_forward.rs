//! Shareable Qwen-VL ViT forward path.
//!
//! Phase 3-a-i scaffolding for the Qwen 3.5 vision tower bring-up.
//!
//! The Qwen 3.5 27B dense and Qwen 3.6 35B-A3B checkpoints carry
//! byte-identical Qwen3-VL ViT geometry (27 blocks, hidden=1152,
//! intermediate=4304, 16 heads × head_dim=72, PatchMerger 2×2 →
//! out_hidden=2048 — note: out_hidden equals the *Qwen 3.6* text
//! hidden size, not Qwen 3.5's hidden=5120; the merger is sized
//! against the tower's intrinsic output, and the splice handles
//! any down-/up-projection if needed). The same forward kernel
//! chain works for both.
//!
//! Today the implementation still lives inline in
//! `qwen36_bring_up::Qwen36Bringup::forward_qwen_vision` (~1100 LOC).
//! This module introduces the **borrow bundle** that lets the
//! function operate on `(arena, stream, cublaslt, vision-tower,
//! kernel-handles)` without going through `&Qwen36Bringup`.
//! Phase 3-a-ii will move the function body itself; Phase 3-a-iii
//! adds the Qwen 3.5 caller.
//!
//! ## Why a borrow bundle and not a trait
//!
//! Two reasons:
//!  1. The forward path holds ~12 distinct `KernelFn` handles plus
//!     an arena, stream, cublaslt, and a vision-tower struct.
//!     Encoding each as a trait method gives a 16-method trait that
//!     gains nothing over a flat struct.
//!  2. `&dyn Trait` runs into lifetime trouble because every
//!     reference inside the forward function is borrowed from the
//!     bringup; a static borrow bundle propagates a single `'a`
//!     across all fields, which the compiler can see through.
//!
//! ## Lifetime story
//!
//! `'a` is the lifetime of the bringup that owns all the resources.
//! Every reference in `QwenVisionDeps` lives at least as long as
//! `'a`. Inside the free function, `arena.region(...)` returns
//! `Region<'a>` which works because we're borrowing the arena
//! through `'a`.

#![cfg(feature = "cuda")]

use rvllm_cutlass::CublasLt;
use rvllm_kernels::KernelFn;
use rvllm_loader::qwen36_weights::Qwen36Vision;
use rvllm_mem::{stream::Stream, HbmArena};

/// Borrow bundle handed to `forward_qwen_vision`. All fields are
/// borrowed from the owning bringup for the duration of one forward
/// pass.
pub struct QwenVisionDeps<'a> {
    pub vision: &'a Qwen36Vision,
    pub arena: &'a HbmArena<'a>,
    pub stream: &'a Stream,
    pub cublaslt: &'a CublasLt,
    // 12 kernels used by the ViT forward chain — verified against
    // qwen36_bring_up::forward_qwen_vision's `self.outside_kernels.fn_*`
    // call sites (2026-05-12). If a future kernel edit adds another
    // launch in the forward, add the handle here so the deps stay
    // exhaustive.
    pub fn_layernorm_inplace_f16: KernelFn,
    pub fn_gelu_tanh_f16: KernelFn,
    pub fn_softmax_row_f16: KernelFn,
    pub fn_vit_rotary_2d_f16: KernelFn,
    pub fn_vit_pos_embed_interp_f16: KernelFn,
    pub fn_scale_inplace_f16: KernelFn,
    pub fn_transpose_2d_f16: KernelFn,
    pub fn_add_bias_f16: KernelFn,
    pub fn_cast_f32_to_f16: KernelFn,
    pub fn_extract_head_f16: KernelFn,
    pub fn_scatter_head_f16: KernelFn,
    pub fn_vector_add_f16: KernelFn,
}
