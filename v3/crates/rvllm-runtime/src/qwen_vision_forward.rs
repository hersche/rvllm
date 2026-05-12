//! Shareable Qwen-VL ViT forward path.
//!
//! Qwen 3.5 27B dense and Qwen 3.6 35B-A3B share the same
//! Qwen3-VL ViT geometry (27 blocks, hidden=1152, intermediate=4304,
//! 16 heads × head_dim=72, PatchMerger 2×2). The merger's
//! `out_hidden` differs per family (Qwen 3.5: 5120, Qwen 3.6: 2048)
//! and is read at runtime from `vision.merger.fc2_w.shape[0]`
//! (Phase 3-a-vii), so the same forward fn drives both with no
//! family-specific branching.
//!
//! Both bringups expose `vision_deps()` to assemble the borrow
//! bundle below; `forward_qwen_vision(&deps, &bytes)` is the
//! shared entry point (smoke-validated against Qwen 3.6 on
//! GB10/sm_121 — "Kreis" on a 256×256 orange-disc PNG).
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
    // Phase-perf 2: batched-strided attention path (parity with the
    // Gemma 4 vision tower). Replaces a 16-head loop with a constant
    // number of launches per ViT block (1 batched QK^T GEMM, 1 fused
    // softmax over H×N rows, 1 transpose V, 1 batched scores @ V).
    pub fn_softmax_row_f32_to_f16: KernelFn,
    pub fn_transpose_heads_v_f16: KernelFn,
}

// ============================================================
// Phase 3-a-ii: shared Qwen-VL ViT forward (~1100 LOC).
//
// Body moved verbatim from `Qwen36Bringup::forward_qwen_vision`
// with the following mechanical substitutions:
//   self.outside_kernels.fn_X  →  deps.fn_X
//   self.arena                 →  deps.arena
//   self.stream                →  deps.stream
//   self.cublaslt              →  deps.cublaslt
//   self.model.vision.as_ref() →  deps.vision  (direct, taken at top)
// No algorithmic change. Both bringups now drive the same forward
// chain by handing in their `vision_deps()`.
// ============================================================

use crate::qwen36_bring_up::VisionForwardOutput;
use rvllm_core::Result;

pub fn forward_qwen_vision(
    deps: &QwenVisionDeps<'_>,
    image_bytes: &[u8],
) -> Result<VisionForwardOutput> {
    use crate::vision_preprocess::{
        decode_image, preprocess_qwen, QwenPreprocessConfig,
    };
    let vision = deps.vision;

    // ── Step 1: decode + preprocess (CPU). ───────────────────────
    let img = decode_image(image_bytes).map_err(|_e| {
        rvllm_core::RvllmError::cuda(
            "vision: image decode failed",
            rvllm_core::CudaErrorKind::Other,
            rvllm_core::CudaCtx::setup(),
        )
    })?;
    let cfg = QwenPreprocessConfig::default();
    let pp = preprocess_qwen(&img, &cfg).map_err(|_e| {
        rvllm_core::RvllmError::cuda(
            "vision: preprocess failed",
            rvllm_core::CudaErrorKind::Other,
            rvllm_core::CudaCtx::setup(),
        )
    })?;
    let [grid_t, grid_h, grid_w] = pp.grid_thw;
    let n_tokens = (grid_t as usize) * (grid_h as usize) * (grid_w as usize);
    let patch_dim: usize = 1536;
    let hidden: usize = 1152;
    let num_heads: usize = 16;
    let head_dim: usize = 72;
    let intermediate: usize = 4304;
    let merge: usize = 2;
    let merge_sq = merge * merge;
    let merger_in: usize = hidden * merge_sq; // 4608
    // Phase 3-a-vii: out_hidden is now data-driven from the
    // merger's fc2 weight row count, so Qwen 3.5 (out=5120) and
    // Qwen 3.6 (out=2048) share the same forward fn without a
    // family-specific branch. The fc2 weight is shape
    // [out_hidden, merger_in].
    let out_hidden: usize = vision.merger.fc2_w.shape[0];
    let n_merged = n_tokens / merge_sq;

    // ── Step 2: upload patches as f16. ──────────────────────────
    let patches_bytes = pp.to_f16_bytes();
    let patches_region =
        deps.arena.region("qvis_patches", patches_bytes.len(), 16)?;
    unsafe { patches_region.copy_from_host(&patches_bytes)? };

    let stream_raw = deps.stream.raw() as u64;

    // Helper: linear with bias (in_f16 [M,K] @ W^T [N,K] + b[N] →
    // out_f16 [M,N]). Implemented as cuBLASLt f16-GEMM-f32 + cast +
    // bias-add. The work-buffer is allocated by the caller because
    // ranges depend on M, N.
    let linear_with_bias = |in_dev: u64,
                            w_dev: u64,
                            b_dev: u64,
                            out_dev: u64,
                            f32_scratch: u64,
                            m: usize,
                            n: usize,
                            k: usize|
     -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            deps.cublaslt.f16_gemm_f32(
                in_dev, w_dev, f32_scratch,
                m as i32, n as i32, k as i32,
                stream_raw,
            )?;
            // Cast f32 → f16
            let n_elem = (m * n) as i32;
            let mut out = out_dev;
            let mut input = f32_scratch;
            let mut nn = n_elem;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
            let rc = cuLaunchKernel(
                deps.fn_cast_f32_to_f16.raw() as CUfunction,
                grid.0, grid.1, grid.2,
                block, 1, 1,
                0, deps.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        if rc != CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::cuda(
                "qwen-vit cast_f32_to_f16 launch",
                rvllm_core::CudaErrorKind::LaunchFailed,
                rvllm_core::CudaCtx::setup(),
            ));
        }
            // Add bias in-place
            let mut tensor = out_dev;
            let mut bias = b_dev;
            let mut dim = n as i32;
            let bargs = [
                (&mut tensor) as *mut u64 as *mut core::ffi::c_void,
                (&mut bias) as *mut u64 as *mut core::ffi::c_void,
                (&mut dim) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block_b: u32 = (n as u32).min(1024);
            let rc = cuLaunchKernel(
                deps.fn_add_bias_f16.raw() as CUfunction,
                m as u32, 1, 1,
                block_b, 1, 1,
                0, deps.stream.raw() as CUstream,
                bargs.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        if rc != CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::cuda(
                "qwen-vit add_bias_f16 launch",
                rvllm_core::CudaErrorKind::LaunchFailed,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        }
        // Phase-perf 1: no trailing fence. linear_with_bias is
        // called ~108×/forward (27 ViT blocks × 4 sites: QKV, proj,
        // fc1, fc2, plus 3 calls outside the loop). All call sites
        // either chain into the next stream-ordered device launch
        // or terminate at a synchronous cuMemcpyDtoH_v2 (which
        // carries its own implicit sync). The previous fence was
        // pure pipeline-drain overhead.
        Ok(())
    };

    // ── Step 3: patch_embed: patches [N, 1536] @ W^T [1152, 1536] + b. ─
    let hidden_bytes = n_tokens * hidden * 2;
    let f32_scratch = deps
        .arena
        .region("qvis_f32_scratch", n_tokens * intermediate.max(hidden) * 4, 16)?;
    let hidden_region = deps.arena.region("qvis_hidden", hidden_bytes, 16)?;
    linear_with_bias(
        patches_region.device_ptr(),
        vision.patch_embed.proj_weight.offset_bytes,
        vision.patch_embed.proj_bias.offset_bytes,
        hidden_region.device_ptr(),
        f32_scratch.device_ptr(),
        n_tokens, hidden, patch_dim,
    )?;

    // Stage dump: post patch_embed, pre pos_embed.
    if let Ok(path) = std::env::var("RVLLM_QWEN36_VIT_PATCH_EMBED_DUMP") {
        let bytes = n_tokens * hidden * 2;
        let mut host = vec![0u8; bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, hidden_region.device_ptr(), bytes);
        }
        let _ = std::fs::write(&path, &host);
        eprintln!("[qwen-vit] vit patch_embed dump: {} bytes ([{},{}]) → {}", bytes, n_tokens, hidden, path);
    }

    // ── Step 3.5: add learned absolute pos_embed (bilinear-interp). ─
    // vllm qwen3_vl.py:801: hidden_states += pos_embeds. The pos_embed
    // table is `[2304, 1152]` (num_grid_per_side²); we interpolate it
    // to `[grid_h * grid_w, 1152]` and add to hidden_region in place.
    const QWEN_VIT_NUM_GRID: i32 = 48; // sqrt(num_position_embeddings=2304)
    #[cfg(feature = "cuda")]
    unsafe {
        use cudarc::driver::sys::*;
        let mut hs = hidden_region.device_ptr();
        let mut tab = vision.pos_embed.offset_bytes;
        let mut gh = grid_h as i32;
        let mut gw = grid_w as i32;
        let mut ng = QWEN_VIT_NUM_GRID;
        let mut ms = merge as i32;
        let mut hd = hidden as i32;
        let args = [
            (&mut hs) as *mut u64 as *mut core::ffi::c_void,
            (&mut tab) as *mut u64 as *mut core::ffi::c_void,
            (&mut gh) as *mut i32 as *mut core::ffi::c_void,
            (&mut gw) as *mut i32 as *mut core::ffi::c_void,
            (&mut ng) as *mut i32 as *mut core::ffi::c_void,
            (&mut ms) as *mut i32 as *mut core::ffi::c_void,
            (&mut hd) as *mut i32 as *mut core::ffi::c_void,
        ];
        let block: u32 = 256;
        let rc = cuLaunchKernel(
            deps.fn_vit_pos_embed_interp_f16.raw() as CUfunction,
            n_tokens as u32, 1, 1,
            block, 1, 1,
            0, deps.stream.raw() as CUstream,
            args.as_ptr() as *mut *mut core::ffi::c_void,
            core::ptr::null_mut(),
        );
    if rc != CUresult::CUDA_SUCCESS {
        return Err(rvllm_core::RvllmError::cuda(
            "qwen-vit vit_pos_embed_interp_f16 launch",
            rvllm_core::CudaErrorKind::LaunchFailed,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    }

    // Stage dump: post pos_embed (before any block).
    if let Ok(path) = std::env::var("RVLLM_QWEN36_VIT_POSEMB_DUMP") {
        let bytes = n_tokens * hidden * 2;
        let mut host = vec![0u8; bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, hidden_region.device_ptr(), bytes);
        }
        let _ = std::fs::write(&path, &host);
        eprintln!("[qwen-vit] vit posemb dump: {} bytes → {}", bytes, path);
    }

    // ── Step 4: build per-token cos/sin tables for 2D rotary. ────
    // Matches HF transformers Qwen3VLVisionModel:
    //   freq_table = RotaryEmbedding(head_dim/2)(max_hw)   [max_hw, 18]
    //   embeddings = freq_table[pos_ids]                   [N, 2, 18]
    //   embeddings = embeddings.flatten(1)                 [N, 36]
    //   emb = cat([embeddings, embeddings], dim=-1)        [N, 72] (DUPLICATED)
    //   cos, sin = emb.cos(), emb.sin()                    [N, 72]
    // Rotary then applies rotate_half on the FULL head_dim=72 with these
    // tables. The "partial_rotary" interpretation from vLLM's get_rope
    // call was misleading — the effective rotary_dim is head_dim, not
    // head_dim/2. Within the [N, 72] cos table:
    //   cos[t, 0..18]   = cos(h_pos[t] * inv_freq[k])
    //   cos[t, 18..36]  = cos(w_pos[t] * inv_freq[k])
    //   cos[t, 36..54]  = cos(h_pos[t] * inv_freq[k])  (mirror of 0..18)
    //   cos[t, 54..72]  = cos(w_pos[t] * inv_freq[k])  (mirror of 18..36)
    // inv_freq dim = 18, with theta=10000 over (head_dim/2).
    let rotary_dim = head_dim;            // 72 (full head_dim — see HF cat-trick above)
    let inv_freq_dim = head_dim / 4;      // 18
    let inv_freq_dim_total = head_dim / 2; // 36 — pre-cat embedding width
    let inv_theta: Vec<f32> = (0..inv_freq_dim)
        .map(|k| 1.0 / 10_000.0_f32.powf(2.0 * k as f32 / inv_freq_dim_total as f32))
        .collect();
    let mut cos_table_host = vec![0u8; n_tokens * rotary_dim * 2];
    let mut sin_table_host = vec![0u8; n_tokens * rotary_dim * 2];
    // Determine per-token (h_pos, w_pos) following HF's rot_pos_emb
    // (block-internal merge-aware ordering).
    let mut pos_h = vec![0i32; n_tokens];
    let mut pos_w = vec![0i32; n_tokens];
    {
        let merged_h = (grid_h as usize) / merge;
        let merged_w = (grid_w as usize) / merge;
        let mut idx = 0usize;
        for _t in 0..(grid_t as usize) {
            for bh in 0..merged_h {
                for bw in 0..merged_w {
                    for ih in 0..merge {
                        for iw in 0..merge {
                            pos_h[idx] = (bh * merge + ih) as i32;
                            pos_w[idx] = (bw * merge + iw) as i32;
                            idx += 1;
                        }
                    }
                }
            }
        }
    }
    // Build [N, 72] cos/sin tables matching HF cat-of-itself layout:
    //   table[t, k]            = cos/sin(h_pos[t] * inv_freq[k])     for k ∈ [0, 18)
    //   table[t, 18+k]         = cos/sin(w_pos[t] * inv_freq[k])
    //   table[t, 36+k]         = same as table[t, k]                  (cat duplicate)
    //   table[t, 54+k]         = same as table[t, 18+k]               (cat duplicate)
    for t in 0..n_tokens {
        for k in 0..inv_freq_dim {
            let ah = (pos_h[t] as f32) * inv_theta[k];
            let aw = (pos_w[t] as f32) * inv_theta[k];
            let cos_h = half::f16::from_f32(ah.cos()).to_le_bytes();
            let sin_h = half::f16::from_f32(ah.sin()).to_le_bytes();
            let cos_w = half::f16::from_f32(aw.cos()).to_le_bytes();
            let sin_w = half::f16::from_f32(aw.sin()).to_le_bytes();
            let row_base = t * rotary_dim;
            for &mirror in &[0, inv_freq_dim_total] {
                let off_h = (row_base + mirror + k) * 2;
                let off_w = (row_base + mirror + inv_freq_dim + k) * 2;
                cos_table_host[off_h] = cos_h[0]; cos_table_host[off_h + 1] = cos_h[1];
                sin_table_host[off_h] = sin_h[0]; sin_table_host[off_h + 1] = sin_h[1];
                cos_table_host[off_w] = cos_w[0]; cos_table_host[off_w + 1] = cos_w[1];
                sin_table_host[off_w] = sin_w[0]; sin_table_host[off_w + 1] = sin_w[1];
            }
        }
    }
    let cos_region = deps.arena.region("qvis_cos", cos_table_host.len(), 16)?;
    let sin_region = deps.arena.region("qvis_sin", sin_table_host.len(), 16)?;
    unsafe {
        cos_region.copy_from_host(&cos_table_host)?;
        sin_region.copy_from_host(&sin_table_host)?;
    }

    // ── Step 5: 27-block transformer loop. ──────────────────────
    // Persistent scratch for QKV, attn-out, MLP intermediate, scores.
    let qkv_bytes = n_tokens * 3 * hidden * 2;
    let qkv_region = deps.arena.region("qvis_qkv", qkv_bytes, 16)?;
    let q_buf = deps.arena.region("qvis_q", n_tokens * hidden * 2, 16)?;
    let k_buf = deps.arena.region("qvis_k", n_tokens * hidden * 2, 16)?;
    let v_buf = deps.arena.region("qvis_v", n_tokens * hidden * 2, 16)?;
    let attn_out = deps.arena.region("qvis_attn_out", n_tokens * hidden * 2, 16)?;
    let mlp_buf = deps.arena.region("qvis_mlp", n_tokens * intermediate * 2, 16)?;
    let scores_bytes = n_tokens * n_tokens * 2;
    let scores_buf = deps.arena.region("qvis_scores", scores_bytes, 16)?;
    let scores_f32 = deps.arena.region("qvis_scores_f32", n_tokens * n_tokens * 4, 16)?;

    let qkv_eps = 1e-6f32;
    let blk_dump_dir = std::env::var("RVLLM_QWEN36_VIT_BLK_DUMP_DIR").ok();
    for (blk_idx, blk) in vision.blocks.iter().enumerate() {
        // ─ pre-attn LayerNorm on a copy ─
        let normed = deps.arena.region("qvis_normed", n_tokens * hidden * 2, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoDAsync_v2(
                normed.device_ptr(),
                hidden_region.device_ptr(),
                n_tokens * hidden * 2,
                deps.stream.raw() as _,
            );
            let mut x = normed.device_ptr();
            let mut g = blk.norm1_w.offset_bytes;
            let mut b = blk.norm1_b.offset_bytes;
            let mut eps = qkv_eps;
            let mut hd_i = hidden as i32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut g) as *mut u64 as *mut core::ffi::c_void,
                (&mut b) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps) as *mut f32 as *mut core::ffi::c_void,
                (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = (hidden as u32).min(1024);
            let rc = cuLaunchKernel(
                deps.fn_layernorm_inplace_f16.raw() as CUfunction,
                n_tokens as u32, 1, 1,
                block, 1, 1,
                0, deps.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        if rc != CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::cuda(
                "qwen-vit layernorm_inplace_f16 launch",
                rvllm_core::CudaErrorKind::LaunchFailed,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        }

        // Block 0 dump: norm1 output (= input to QKV proj).
        if blk_idx == 0 {
            if let Some(dir) = blk_dump_dir.as_deref() {
                let bytes = n_tokens * hidden * 2;
                let mut host = vec![0u8; bytes];
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, normed.device_ptr(), bytes);
                }
                let _ = std::fs::write(format!("{dir}/blk0_norm1_out.bin"), &host);
            }
        }

        // ─ QKV proj: normed [N, 1152] @ W^T [3456, 1152] + b. ─
        linear_with_bias(
            normed.device_ptr(),
            blk.qkv_w.offset_bytes,
            blk.qkv_b.offset_bytes,
            qkv_region.device_ptr(),
            f32_scratch.device_ptr(),
            n_tokens, 3 * hidden, hidden,
        )?;

        // ─ Split QKV → Q, K, V (each [N, 1152]). HF lays them out
        //   as [N, 3*hidden] = (Q[N,hidden], K[N,hidden], V[N,hidden]). ─
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let row_bytes = (hidden * 2) as u64;
            for t in 0..n_tokens {
                let src = qkv_region.device_ptr() + (t as u64) * 3 * row_bytes;
                let _ = cuMemcpyDtoDAsync_v2(
                    q_buf.device_ptr() + (t as u64) * row_bytes,
                    src,
                    row_bytes as usize,
                    deps.stream.raw() as _,
                );
                let _ = cuMemcpyDtoDAsync_v2(
                    k_buf.device_ptr() + (t as u64) * row_bytes,
                    src + row_bytes,
                    row_bytes as usize,
                    deps.stream.raw() as _,
                );
                let _ = cuMemcpyDtoDAsync_v2(
                    v_buf.device_ptr() + (t as u64) * row_bytes,
                    src + 2 * row_bytes,
                    row_bytes as usize,
                    deps.stream.raw() as _,
                );
            }
        }

        // ─ Apply 2D rotary to Q, K. ─
        for &qk_ptr in &[q_buf.device_ptr(), k_buf.device_ptr()] {
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut x = qk_ptr;
                let mut cos = cos_region.device_ptr();
                let mut sin = sin_region.device_ptr();
                let mut nh = num_heads as i32;
                let mut hd_i = head_dim as i32;
                let mut rd_i = rotary_dim as i32;
                let args = [
                    (&mut x) as *mut u64 as *mut core::ffi::c_void,
                    (&mut cos) as *mut u64 as *mut core::ffi::c_void,
                    (&mut sin) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
                    (&mut rd_i) as *mut i32 as *mut core::ffi::c_void,
                ];
                let rc = cuLaunchKernel(
                    deps.fn_vit_rotary_2d_f16.raw() as CUfunction,
                    n_tokens as u32, num_heads as u32, 1,
                    (rotary_dim / 2) as u32, 1, 1,
                    0, deps.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen-vit vit_rotary_2d_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            }
        }

        // Block 0: dump Q + K post-rotary (full [N, hidden] f16).
        if blk_idx == 0 {
            if let Some(dir) = blk_dump_dir.as_deref() {
                let bytes = n_tokens * hidden * 2;
                let mut q_host = vec![0u8; bytes];
                let mut k_host = vec![0u8; bytes];
                let mut v_host = vec![0u8; bytes];
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(q_host.as_mut_ptr() as *mut _, q_buf.device_ptr(), bytes);
                    let _ = cuMemcpyDtoH_v2(k_host.as_mut_ptr() as *mut _, k_buf.device_ptr(), bytes);
                    let _ = cuMemcpyDtoH_v2(v_host.as_mut_ptr() as *mut _, v_buf.device_ptr(), bytes);
                }
                let _ = std::fs::write(format!("{dir}/blk0_q_postrot.bin"), &q_host);
                let _ = std::fs::write(format!("{dir}/blk0_k_postrot.bin"), &k_host);
                let _ = std::fs::write(format!("{dir}/blk0_v.bin"), &v_host);
            }
        }

        // ─ Per-head attention: QK^T → softmax → @V. ─
        // Q, K, V layout is [N, num_heads*head_dim] = [N, 1152]
        // with each token's row containing all heads concatenated.
        // For head h, head data lives at offset h*head_dim within
        // each row, with stride hidden bytes.
        //
        // To get a contiguous [N, head_dim] per head, we copy each
        // head's slice into temporary buffers. For all 16 heads
        // we use a single round-robin scratch (q_h, k_h, v_h, out_h).
        let q_h = deps.arena.region("qvis_qh", n_tokens * head_dim * 2, 16)?;
        let k_h = deps.arena.region("qvis_kh", n_tokens * head_dim * 2, 16)?;
        let v_h = deps.arena.region("qvis_vh", n_tokens * head_dim * 2, 16)?;
        let v_h_t = deps.arena.region("qvis_vht", head_dim * n_tokens * 2, 16)?;
        let out_h = deps.arena.region("qvis_oh", n_tokens * head_dim * 2, 16)?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Gather/scatter helpers shared with the per-head attention
        // loop. Replaces the per-token DtoD memcpy schedule (~ N
        // memcpys × 4 directions × num_heads × 27 blocks = O(106k)
        // launches per image at N=196) with a single kernel
        // launch per (head, direction). Codex review #C round 3.
        let extract_head = |dst: u64, src: u64, head_idx: usize| -> Result<()> {
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut o = dst;
                let mut i = src;
                let mut hi = head_idx as i32;
                let mut nh = num_heads as i32;
                let mut hd = head_dim as i32;
                let args = [
                    (&mut o) as *mut u64 as *mut core::ffi::c_void,
                    (&mut i) as *mut u64 as *mut core::ffi::c_void,
                    (&mut hi) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                ];
                let rc = cuLaunchKernel(
                    deps.fn_extract_head_f16.raw() as CUfunction,
                    n_tokens as u32, 1, 1,
                    head_dim as u32, 1, 1,
                    0, deps.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qvis: extract_head launch failed",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            Ok(())
        };
        let scatter_head = |dst: u64, src: u64, head_idx: usize| -> Result<()> {
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut o = dst;
                let mut i = src;
                let mut hi = head_idx as i32;
                let mut nh = num_heads as i32;
                let mut hd = head_dim as i32;
                let args = [
                    (&mut o) as *mut u64 as *mut core::ffi::c_void,
                    (&mut i) as *mut u64 as *mut core::ffi::c_void,
                    (&mut hi) as *mut i32 as *mut core::ffi::c_void,
                    (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                    (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                ];
                let rc = cuLaunchKernel(
                    deps.fn_scatter_head_f16.raw() as CUfunction,
                    n_tokens as u32, 1, 1,
                    head_dim as u32, 1, 1,
                    0, deps.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qvis: scatter_head launch failed",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            Ok(())
        };
        for h in 0..num_heads {
            extract_head(q_h.device_ptr(), q_buf.device_ptr(), h)?;
            extract_head(k_h.device_ptr(), k_buf.device_ptr(), h)?;
            extract_head(v_h.device_ptr(), v_buf.device_ptr(), h)?;

            // QK^T: [N, head_dim] @ [N, head_dim]^T → [N, N] f32 → cast f16
            // Use cuBLASLt f16_gemm_f32 with N=N, M=N, K=head_dim.
            #[cfg(feature = "cuda")]
            unsafe {
                deps.cublaslt.f16_gemm_f32(
                    q_h.device_ptr(), k_h.device_ptr(),
                    scores_f32.device_ptr(),
                    n_tokens as i32, n_tokens as i32, head_dim as i32,
                    stream_raw,
                )?;
                // Apply scale + cast to f16: scores_f16[i] = (scores_f32[i] * scale) → f16
                // We do this by scaling f32 first (in-place via simple kernel — reuse cast
                // since we don't have a fused scale_cast: just bake scale into cos/sin? No,
                // just multiply scale into Q before GEMM. Move scaling there.)
                let n_elem = (n_tokens * n_tokens) as i32;
                let mut out = scores_buf.device_ptr();
                let mut input = scores_f32.device_ptr();
                let mut nn = n_elem;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
                use cudarc::driver::sys::*;
                let rc = cuLaunchKernel(
                    deps.fn_cast_f32_to_f16.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block, 1, 1,
                    0, deps.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen-vit cast_f32_to_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            }

            // Apply the standard 1/sqrt(head_dim) attention scale
            // to the f16 scores in place before softmax. (Without
            // this, softmax becomes degenerate — one token wins
            // ~all attention — and patches stop mixing, which
            // produces image-content-blind ViT output.)
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut x = scores_buf.device_ptr();
                let mut s = scale;
                let mut nn = (n_tokens * n_tokens) as i32;
                let args = [
                    (&mut x) as *mut u64 as *mut core::ffi::c_void,
                    (&mut s) as *mut f32 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = ((nn as u32 + block - 1) / block, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    deps.fn_scale_inplace_f16.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block, 1, 1,
                    0, deps.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen-vit scale_inplace_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            }

            // Softmax row-wise on scores [N, N]
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut x = scores_buf.device_ptr();
                let mut sl = n_tokens as i32;
                let args = [
                    (&mut x) as *mut u64 as *mut core::ffi::c_void,
                    (&mut sl) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = (n_tokens as u32).min(1024);
                let rc = cuLaunchKernel(
                    deps.fn_softmax_row_f16.raw() as CUfunction,
                    n_tokens as u32, 1, 1,
                    block, 1, 1,
                    0, deps.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen-vit softmax_row_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            }

            // scores @ V: [N, N] @ [N, head_dim] → [N, head_dim].
            // f16_gemm_f32 always computes input @ weight^T, so we
            // need V transposed [head_dim, N] for the call to give
            // sum_j scores[r,j] * V[j,c] (instead of scores @ V^T).
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut out_p = v_h_t.device_ptr();
                let mut in_p = v_h.device_ptr();
                let mut rows = n_tokens as i32;
                let mut cols = head_dim as i32;
                let args = [
                    (&mut out_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut in_p) as *mut u64 as *mut core::ffi::c_void,
                    (&mut rows) as *mut i32 as *mut core::ffi::c_void,
                    (&mut cols) as *mut i32 as *mut core::ffi::c_void,
                ];
                let gx = ((cols as u32) + 15) / 16;
                let gy = ((rows as u32) + 15) / 16;
                let rc = cuLaunchKernel(
                    deps.fn_transpose_2d_f16.raw() as CUfunction,
                    gx, gy, 1,
                    16, 16, 1,
                    0, deps.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen-vit transpose_2d_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            }
            #[cfg(feature = "cuda")]
            unsafe {
                deps.cublaslt.f16_gemm_f32(
                    scores_buf.device_ptr(), v_h_t.device_ptr(),
                    scores_f32.device_ptr(),
                    n_tokens as i32, head_dim as i32, n_tokens as i32,
                    stream_raw,
                )?;
                // cast f32 → f16 into out_h
                let n_elem = (n_tokens * head_dim) as i32;
                let mut out = out_h.device_ptr();
                let mut input = scores_f32.device_ptr();
                let mut nn = n_elem;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
                use cudarc::driver::sys::*;
                let rc = cuLaunchKernel(
                    deps.fn_cast_f32_to_f16.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block, 1, 1,
                    0, deps.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen-vit cast_f32_to_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            }

            // Scatter out_h back into attn_out at offset h*head_dim per row.
            scatter_head(attn_out.device_ptr(), out_h.device_ptr(), h)?;
        }

        // ─ O proj + residual: hidden += proj(attn_out). ─
        // Block 0 dump: attn output pre-O-proj.
        if blk_idx == 0 {
            if let Some(dir) = blk_dump_dir.as_deref() {
                let bytes = n_tokens * hidden * 2;
                let mut host = vec![0u8; bytes];
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, attn_out.device_ptr(), bytes);
                }
                let _ = std::fs::write(format!("{dir}/blk0_attn_pre_o_proj.bin"), &host);
            }
        }
        let proj_out = deps.arena.region("qvis_proj_out", n_tokens * hidden * 2, 16)?;
        linear_with_bias(
            attn_out.device_ptr(),
            blk.proj_w.offset_bytes,
            blk.proj_b.offset_bytes,
            proj_out.device_ptr(),
            f32_scratch.device_ptr(),
            n_tokens, hidden, hidden,
        )?;
        // Residual: hidden += proj_out via GPU vector_add_f16
        // (replaces the earlier DtoH-add-HtoD round-trip per
        // block — Codex review #3 round 4 follow-up).
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let n_elem = (n_tokens * hidden) as i32;
            let mut d = hidden_region.device_ptr();
            let mut s = proj_out.device_ptr();
            let mut nn = n_elem;
            let args = [
                (&mut d) as *mut u64 as *mut core::ffi::c_void,
                (&mut s) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
            let rc = cuLaunchKernel(
                deps.fn_vector_add_f16.raw() as CUfunction,
                grid.0, grid.1, grid.2,
                block, 1, 1,
                0, deps.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "vision: vector_add (attn residual) launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Block 0 dump: after attention residual (= input + attn_out).
        if blk_idx == 0 {
            if let Some(dir) = blk_dump_dir.as_deref() {
                let bytes = n_tokens * hidden * 2;
                let mut host = vec![0u8; bytes];
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, hidden_region.device_ptr(), bytes);
                }
                let _ = std::fs::write(format!("{dir}/blk0_post_attn.bin"), &host);
            }
        }

        // ─ pre-MLP LayerNorm (norm2) on a copy ─
        let normed2 = deps.arena.region("qvis_normed2", n_tokens * hidden * 2, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoDAsync_v2(
                normed2.device_ptr(),
                hidden_region.device_ptr(),
                n_tokens * hidden * 2,
                deps.stream.raw() as _,
            );
            let mut x = normed2.device_ptr();
            let mut g = blk.norm2_w.offset_bytes;
            let mut b = blk.norm2_b.offset_bytes;
            let mut eps = qkv_eps;
            let mut hd_i = hidden as i32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut g) as *mut u64 as *mut core::ffi::c_void,
                (&mut b) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps) as *mut f32 as *mut core::ffi::c_void,
                (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = (hidden as u32).min(1024);
            let rc = cuLaunchKernel(
                deps.fn_layernorm_inplace_f16.raw() as CUfunction,
                n_tokens as u32, 1, 1,
                block, 1, 1,
                0, deps.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        if rc != CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::cuda(
                "qwen-vit layernorm_inplace_f16 launch",
                rvllm_core::CudaErrorKind::LaunchFailed,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        }

        // ─ MLP: fc1 → GELU → fc2 ─
        linear_with_bias(
            normed2.device_ptr(),
            blk.fc1_w.offset_bytes,
            blk.fc1_b.offset_bytes,
            mlp_buf.device_ptr(),
            f32_scratch.device_ptr(),
            n_tokens, intermediate, hidden,
        )?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let n_elem = (n_tokens * intermediate) as i32;
            let mut x = mlp_buf.device_ptr();
            let mut nn = n_elem;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
            let rc = cuLaunchKernel(
                deps.fn_gelu_tanh_f16.raw() as CUfunction,
                grid.0, grid.1, grid.2,
                block, 1, 1,
                0, deps.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        if rc != CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::cuda(
                "qwen-vit gelu_tanh_f16 launch",
                rvllm_core::CudaErrorKind::LaunchFailed,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        }
        let mlp_out = deps.arena.region("qvis_mlp_out", n_tokens * hidden * 2, 16)?;
        linear_with_bias(
            mlp_buf.device_ptr(),
            blk.fc2_w.offset_bytes,
            blk.fc2_b.offset_bytes,
            mlp_out.device_ptr(),
            f32_scratch.device_ptr(),
            n_tokens, hidden, intermediate,
        )?;
        // Residual: hidden += mlp_out via GPU vector_add_f16.
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let n_elem = (n_tokens * hidden) as i32;
            let mut d = hidden_region.device_ptr();
            let mut s = mlp_out.device_ptr();
            let mut nn = n_elem;
            let args = [
                (&mut d) as *mut u64 as *mut core::ffi::c_void,
                (&mut s) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
            let rc = cuLaunchKernel(
                deps.fn_vector_add_f16.raw() as CUfunction,
                grid.0, grid.1, grid.2,
                block, 1, 1,
                0, deps.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "vision: vector_add (mlp residual) launch failed",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Block 0 sub-step dump (post-attn-residual, post-mlp-residual).
        if blk_idx == 0 {
            if let Some(dir) = blk_dump_dir.as_deref() {
                let bytes = n_tokens * hidden * 2;
                let mut host = vec![0u8; bytes];
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, hidden_region.device_ptr(), bytes);
                }
                let _ = std::fs::write(format!("{dir}/blk0_post_mlp.bin"), &host);
            }
        }

        // Per-block dump (only blocks 0, 13, 26 to keep io light).
        if let Some(dir) = blk_dump_dir.as_deref() {
            if blk_idx == 0 || blk_idx == 13 || blk_idx == 26 {
                let bytes = n_tokens * hidden * 2;
                let mut host = vec![0u8; bytes];
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, hidden_region.device_ptr(), bytes);
                }
                let path = format!("{dir}/blk{blk_idx}.bin");
                let _ = std::fs::write(&path, &host);
                eprintln!("[qwen-vit] vit blk{blk_idx} dump → {path}");
            }
        }
    }

    // Env-gated pre-merger dump for HF reference comparison.
    if let Ok(path) = std::env::var("RVLLM_QWEN36_VISION_PREMERGER_DUMP") {
        let bytes = n_tokens * hidden * 2;
        let mut host = vec![0u8; bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, hidden_region.device_ptr(), bytes);
        }
        let _ = std::fs::write(&path, &host);
        eprintln!(
            "[qwen-vit] pre-merger dump: {} bytes ([{}, {}] f16) → {}",
            bytes, n_tokens, hidden, path,
        );
    }

    // ── Step 6: PatchMerger. ────────────────────────────────────
    // (a) LayerNorm hidden in-place per-token (gamma/beta are 1152).
    #[cfg(feature = "cuda")]
    unsafe {
        use cudarc::driver::sys::*;
        let mut x = hidden_region.device_ptr();
        let mut g = vision.merger.norm_w.offset_bytes;
        let mut b = vision.merger.norm_b.offset_bytes;
        let mut eps = qkv_eps;
        let mut hd_i = hidden as i32;
        let args = [
            (&mut x) as *mut u64 as *mut core::ffi::c_void,
            (&mut g) as *mut u64 as *mut core::ffi::c_void,
            (&mut b) as *mut u64 as *mut core::ffi::c_void,
            (&mut eps) as *mut f32 as *mut core::ffi::c_void,
            (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
        ];
        let block: u32 = (hidden as u32).min(1024);
        let rc = cuLaunchKernel(
            deps.fn_layernorm_inplace_f16.raw() as CUfunction,
            n_tokens as u32, 1, 1,
            block, 1, 1,
            0, deps.stream.raw() as CUstream,
            args.as_ptr() as *mut *mut core::ffi::c_void,
            core::ptr::null_mut(),
        );
    if rc != CUresult::CUDA_SUCCESS {
        return Err(rvllm_core::RvllmError::cuda(
            "qwen-vit layernorm_inplace_f16 launch",
            rvllm_core::CudaErrorKind::LaunchFailed,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    }

    // (b) Spatial merge: every 4 spatial-neighbour tokens concat
    // into one row of width merger_in=4608. Token order from
    // pos_h/pos_w: pre-merge tokens are already in
    // (block, intra_h, intra_w) order, so 4 consecutive rows = one
    // 2×2 spatial cluster ⇒ direct concat works.
    let merged_bytes = n_merged * merger_in * 2;
    let merged_region = deps.arena.region("qvis_merged", merged_bytes, 16)?;
    #[cfg(feature = "cuda")]
    unsafe {
        use cudarc::driver::sys::*;
        let row_bytes = (hidden * 2) as u64;
        for m in 0..n_merged {
            for s in 0..merge_sq {
                let src = hidden_region.device_ptr() + ((m * merge_sq + s) as u64) * row_bytes;
                let dst = merged_region.device_ptr()
                    + (m as u64) * (merger_in as u64) * 2
                    + (s as u64) * row_bytes;
                let _ = cuMemcpyDtoDAsync_v2(dst, src, hidden * 2, deps.stream.raw() as _);
            }
        }
    }

    // (c) merger.linear_fc1 → GELU → linear_fc2.
    let merged_out_fc1 = deps.arena.region("qvis_mfc1", n_merged * merger_in * 2, 16)?;
    linear_with_bias(
        merged_region.device_ptr(),
        vision.merger.fc1_w.offset_bytes,
        vision.merger.fc1_b.offset_bytes,
        merged_out_fc1.device_ptr(),
        f32_scratch.device_ptr(),
        n_merged, merger_in, merger_in,
    )?;
    #[cfg(feature = "cuda")]
    unsafe {
        use cudarc::driver::sys::*;
        let n_elem = (n_merged * merger_in) as i32;
        let mut x = merged_out_fc1.device_ptr();
        let mut nn = n_elem;
        let args = [
            (&mut x) as *mut u64 as *mut core::ffi::c_void,
            (&mut nn) as *mut i32 as *mut core::ffi::c_void,
        ];
        let block: u32 = 256;
        let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
        let rc = cuLaunchKernel(
            deps.fn_gelu_tanh_f16.raw() as CUfunction,
            grid.0, grid.1, grid.2,
            block, 1, 1,
            0, deps.stream.raw() as CUstream,
            args.as_ptr() as *mut *mut core::ffi::c_void,
            core::ptr::null_mut(),
        );
    if rc != CUresult::CUDA_SUCCESS {
        return Err(rvllm_core::RvllmError::cuda(
            "qwen-vit gelu_tanh_f16 launch",
            rvllm_core::CudaErrorKind::LaunchFailed,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    }
    let final_region = deps.arena.region("qvis_final", n_merged * out_hidden * 2, 16)?;
    linear_with_bias(
        merged_out_fc1.device_ptr(),
        vision.merger.fc2_w.offset_bytes,
        vision.merger.fc2_b.offset_bytes,
        final_region.device_ptr(),
        f32_scratch.device_ptr(),
        n_merged, out_hidden, merger_in,
    )?;

    // ── Step 7: DtoH the final embeddings. ──────────────────────
    // Round-19 P1: previously the cuMemcpyDtoH return code was
    // discarded with `let _ = …`. Any CUDA fault, lost context,
    // or copy-size mismatch then handed an all-zero / stale
    // `out_bytes` back to the caller, which spliced silently into
    // the residual buffer and produced syntactically clean but
    // meaningless vision output. Propagate the error so the
    // request fails loudly.
    let mut out_bytes = vec![0u8; n_merged * out_hidden * 2];
    #[cfg(feature = "cuda")]
    unsafe {
        use cudarc::driver::sys::*;
        let r = cuMemcpyDtoH_v2(
            out_bytes.as_mut_ptr() as *mut _,
            final_region.device_ptr(),
            out_bytes.len(),
        );
        if r != CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::Cuda {
                kind: rvllm_core::CudaErrorKind::MemcpyFailed,
                op: "qwen_vision_output_dtoh",
                ctx: rvllm_core::CudaCtx {
                    stream: 0,
                    kernel: "",
                    launch: None,
                    device: 0,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
    }

    Ok(VisionForwardOutput {
        data: out_bytes,
        num_tokens: n_merged,
        hidden_dim: out_hidden,
        grid_thw: [grid_t, grid_h, grid_w],
    })
}
