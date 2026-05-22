//! Gemma 4 vision tower forward (SigLIP-style ViT).
//!
//! Extracted from `gemma4_bring_up.rs::Gemma4Bringup::forward_gemma_vision`
//! so the same body can be invoked from both the production fp8-block
//! `Gemma4Bringup` and the Option B `Gemma4Nvfp4Bringup` (Stream-7).
//! The body is verbatim from the original site; only field resolution
//! changes via the borrow-view struct below.
//!
//! Cross-model invariant: every kernel used here is in
//! [`Gemma4VisionKernels`] — both bringups load that struct once at
//! startup, mirroring the existing `Gemma4FusedModules` pattern.

use rvllm_core::{Result, RvllmError};
use rvllm_cutlass::CublasLt;
use rvllm_kernels::{KernelFn, KernelLoader, LoadedModule};
use rvllm_loader::gemma4_arch::Gemma4Arch;
use rvllm_loader::gemma4_weights::Gemma4Vision;
use rvllm_mem::{stream::Stream, HbmArena};

use crate::qwen36_bring_up::VisionForwardOutput;

/// Subset of [`crate::gemma4_bring_up::Gemma4FusedModules`] containing
/// only the kernels reachable through the Gemma 4 ViT forward path.
/// Loaded once per bringup at startup; both `Gemma4Bringup` (fp8-block)
/// and `Gemma4Nvfp4Bringup` (Option B) carry a copy.
pub struct Gemma4VisionKernels {
    pub fn_rmsnorm: KernelFn,
    pub fn_vnorm: KernelFn,
    pub fn_vector_add: KernelFn,
    pub fn_cast_f32_to_f16: KernelFn,
    pub fn_extract_head_f16: KernelFn,
    pub fn_scatter_head_f16: KernelFn,
    pub fn_scatter_heads_f16: KernelFn,
    pub fn_transpose_heads_v_f16: KernelFn,
    pub fn_softmax_row_f32_to_f16: KernelFn,
    pub fn_gelu_tanh_mul_f16: KernelFn,
    pub fn_scale_inplace_f32: KernelFn,
    pub fn_vit_avgpool_f16_to_f32: KernelFn,
    pub fn_vit_pos_emb_lookup_2d_f16: KernelFn,
    pub fn_vit_rotary_gemma4_2d_f16: KernelFn,
    pub fn_vit_standardize_f32_to_f16: KernelFn,
    // Anchor PTX modules so the function pointers stay valid.
    _modules: Vec<LoadedModule>,
}

impl Gemma4VisionKernels {
    pub fn load(loader: &KernelLoader) -> Result<Self> {
        let rmsnorm_inplace_mod = loader.load_ptx("rmsnorm_inplace_f16")?;
        let fn_rmsnorm =
            rmsnorm_inplace_mod.get_function("rmsnorm_inplace_f16_kernel")?;
        let vnorm_mod = loader.load_ptx("vnorm_f16")?;
        let fn_vnorm = vnorm_mod.get_function("vnorm_f16_kernel")?;
        let vector_add_mod = loader.load_ptx("vector_add_f16")?;
        let fn_vector_add = vector_add_mod.get_function("vector_add_f16_kernel")?;
        let cast_fp_mod = loader.load_ptx("cast_fp")?;
        let fn_cast_f32_to_f16 = cast_fp_mod.get_function("cast_f32_to_f16_kernel")?;
        let extract_head_f16_mod = loader.load_ptx("extract_head_f16")?;
        let fn_extract_head_f16 =
            extract_head_f16_mod.get_function("extract_head_f16_kernel")?;
        let fn_scatter_head_f16 =
            extract_head_f16_mod.get_function("scatter_head_f16_kernel")?;
        let scatter_heads_f16_mod = loader.load_ptx("scatter_heads_f16")?;
        let fn_scatter_heads_f16 =
            scatter_heads_f16_mod.get_function("scatter_heads_f16_kernel")?;
        let transpose_heads_v_f16_mod = loader.load_ptx("transpose_heads_v_f16")?;
        let fn_transpose_heads_v_f16 =
            transpose_heads_v_f16_mod.get_function("transpose_heads_v_f16_kernel")?;
        let softmax_row_f32_to_f16_mod = loader.load_ptx("softmax_row_f32_to_f16")?;
        let fn_softmax_row_f32_to_f16 = softmax_row_f32_to_f16_mod
            .get_function("softmax_row_f32_to_f16_kernel")?;
        let gelu_tanh_mul_f16_mod = loader.load_ptx("gelu_tanh_mul_f16")?;
        let fn_gelu_tanh_mul_f16 =
            gelu_tanh_mul_f16_mod.get_function("gelu_tanh_mul_f16_kernel")?;
        let scale_inplace_f32_mod = loader.load_ptx("scale_inplace_f32")?;
        let fn_scale_inplace_f32 =
            scale_inplace_f32_mod.get_function("scale_inplace_f32_kernel")?;
        let vit_avgpool_f16_to_f32_mod = loader.load_ptx("vit_avgpool_f16_to_f32")?;
        let fn_vit_avgpool_f16_to_f32 = vit_avgpool_f16_to_f32_mod
            .get_function("vit_avgpool_f16_to_f32_kernel")?;
        let vit_pos_emb_lookup_2d_f16_mod =
            loader.load_ptx("vit_pos_emb_lookup_2d_f16")?;
        let fn_vit_pos_emb_lookup_2d_f16 = vit_pos_emb_lookup_2d_f16_mod
            .get_function("vit_pos_emb_lookup_2d_f16_kernel")?;
        let vit_rotary_gemma4_2d_f16_mod =
            loader.load_ptx("vit_rotary_gemma4_2d_f16")?;
        let fn_vit_rotary_gemma4_2d_f16 = vit_rotary_gemma4_2d_f16_mod
            .get_function("vit_rotary_gemma4_2d_f16_kernel")?;
        let vit_standardize_f32_to_f16_mod =
            loader.load_ptx("vit_standardize_f32_to_f16")?;
        let fn_vit_standardize_f32_to_f16 = vit_standardize_f32_to_f16_mod
            .get_function("vit_standardize_f32_to_f16_kernel")?;
        Ok(Self {
            fn_rmsnorm,
            fn_vnorm,
            fn_vector_add,
            fn_cast_f32_to_f16,
            fn_extract_head_f16,
            fn_scatter_head_f16,
            fn_scatter_heads_f16,
            fn_transpose_heads_v_f16,
            fn_softmax_row_f32_to_f16,
            fn_gelu_tanh_mul_f16,
            fn_scale_inplace_f32,
            fn_vit_avgpool_f16_to_f32,
            fn_vit_pos_emb_lookup_2d_f16,
            fn_vit_rotary_gemma4_2d_f16,
            fn_vit_standardize_f32_to_f16,
            _modules: vec![
                rmsnorm_inplace_mod,
                vnorm_mod,
                vector_add_mod,
                cast_fp_mod,
                extract_head_f16_mod,
                scatter_heads_f16_mod,
                transpose_heads_v_f16_mod,
                softmax_row_f32_to_f16_mod,
                gelu_tanh_mul_f16_mod,
                scale_inplace_f32_mod,
                vit_avgpool_f16_to_f32_mod,
                vit_pos_emb_lookup_2d_f16_mod,
                vit_rotary_gemma4_2d_f16_mod,
                vit_standardize_f32_to_f16_mod,
            ],
        })
    }
}

/// Tiny adapter exposing the body's `self.model.vision` reference
/// pattern. Holds an `Option<&Gemma4Vision>` so callers can splice in
/// whichever weights container they own.
pub struct Gemma4VisionModelView<'a> {
    pub vision: Option<&'a Gemma4Vision>,
}

/// Borrow-view holding every reference [`Gemma4VisionRuntime::forward`]
/// needs. Constructed by each bringup at call time. Field names mirror
/// the original method's `self.X` lookups so the body stays
/// byte-identical except for the single `self.model.vision.as_ref()`
/// → `self.model.vision` substitution (the view's `vision` field is
/// already `Option<&...>`).
pub struct Gemma4VisionRuntime<'a> {
    pub arch: &'a Gemma4Arch,
    pub arena: &'a HbmArena<'static>,
    pub stream: &'a Stream,
    pub cublaslt: &'a CublasLt,
    pub fused: &'a Gemma4VisionKernels,
    pub model: Gemma4VisionModelView<'a>,
}

impl<'a> Gemma4VisionRuntime<'a> {
    /// Run the Gemma 4 SigLIP-style ViT over a single image. Returns
    /// f16 embeddings of shape `[num_pooled_tokens, hidden_size]`
    /// ready for splice into the post-embed text-side hidden buffer.
    /// Body verbatim from `Gemma4Bringup::forward_gemma_vision`
    /// (gemma4_bring_up.rs); the only substitution is that
    /// `self.model.vision` is already `Option<&Gemma4Vision>` here.
    #[cfg(feature = "cuda")]
    pub fn forward(
        &self,
        image_bytes: &[u8],
    ) -> Result<VisionForwardOutput> {
        use crate::vision_preprocess::{decode_image, preprocess_gemma, GemmaPreprocessConfig};
        use cudarc::driver::sys::*;

        let vision = self.model.vision.ok_or_else(|| {
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

    /// Non-CUDA build stub — vision forward is a CUDA-only path.
    #[cfg(not(feature = "cuda"))]
    pub fn forward(&self, _image_bytes: &[u8]) -> Result<VisionForwardOutput> {
        Err(RvllmError::cuda(
            "vision: forward_gemma_vision is CUDA-only",
            rvllm_core::CudaErrorKind::Other,
            rvllm_core::CudaCtx::setup(),
        ))
    }
}
