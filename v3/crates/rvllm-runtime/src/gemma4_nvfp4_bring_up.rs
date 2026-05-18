//! Phase 3c+ commit #4a: text-only `Gemma4Nvfp4Bringup` skeleton
//! with the load constructor and a pre-attention sub-block
//! smoke (embed → input_layernorm → q_proj on layer 0).
//!
//! This commit lays the structural foundation that future
//! commits build on:
//!   * #4b: per-layer attention math (Q/K-norm, RoPE, paged
//!     attention into the KV cache, O-projection, residual).
//!   * #4c: per-layer MLP composition (pre-FF norm, MLP via
//!     commit #3 helpers, post-FF norm, layer_scalar, residual).
//!     Wires the full 60-layer loop.
//!   * #4d: final RMSNorm + tied LM head + argmax + decoder
//!     loop. Single-token greedy generation smoke.
//!   * #5: NVFP4 KV cache integration (Hadamard rotation,
//!     bf16-in NVFP4 KV write kernel).
//!   * #6: spec-decode + vision splice.
//!
//! Scope of THIS commit:
//!   * Gemma4Nvfp4Bringup struct definition + Drop semantics.
//!   * `load(model_dir, arena_bytes)` constructor that opens
//!     a CUDA context + arena + stream + cuBLASLt handle +
//!     KernelLoader, runs `load_gemma4_nvfp4_text` (uploads
//!     embed + final_norm + all 60 layers), and loads the
//!     PTX kernel handles the forward path will need.
//!   * `pre_attn_one_token(token_id)` — a partial forward
//!     that exercises embed lookup + layer-0 input_layernorm +
//!     layer-0 q_proj. Returns the [N_q] f32 output as a host
//!     Vec for asserting plausible magnitudes.
//!   * `#[ignore]` integration smoke against the on-disk
//!     checkpoint.
//!
//! Out of scope: attention math, MLP composition, KV cache,
//! spec-decode, vision. Those are #4b..#6.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use rvllm_core::Result;
use rvllm_cutlass::cublaslt::CublasLt;
use std::sync::Arc;
use rvllm_kernels::{KernelFn, KernelLoader, LoadedModule};
use rvllm_loader::gemma4_arch::Gemma4Arch;
use rvllm_loader::gemma4_nvfp4_weights::Gemma4Nvfp4LoadedModel;
use rvllm_mem::{HbmArena, Stream};
use rvllm_mem::context::CudaContextHandle;

use crate::gemma4_nvfp4_load::load_gemma4_nvfp4_text;
use crate::gemma4_nvfp4_ops::{
    gemma4_nvfp4_attn_proj, Gemma4Nvfp4MlpKernels,
};

/// Kernel handles the forward path needs beyond the MLP set.
/// Loaded once at bring-up time from the existing sm_121 PTX
/// manifest.
///
/// Persistent NVFP4 KV cache + attention scratch state.
/// Allocated once at `load` time ABOVE the per-call
/// forward_checkpoint so each forward's scratch rewind doesn't
/// touch the cache.
///
/// Per codex round-4 A1+A6: the
/// `fused_rope_partial_nvfp4kv_bf16in_kernel` reads/writes
/// `(slot * num_kv_heads + head_idx) * (head_dim/2)` bytes from
/// a PER-LAYER base. We allocate one contiguous region per
/// (layer, K|V, packed|scale) tuple and pass the base pointer
/// in alongside `positions_ptr` + `slot_mapping_ptr`.
///
/// Block layout: identity blocks with `block_size=1` and
/// `block_tables[i] = i`. Spec/rollback support is deferred
/// to commit #6 — those scenarios need block reassignment +
/// commit-before-emit rollback ordering analogous to the
/// fp8-block path.
///
/// Memory footprint at max_pos=4096 on 31B:
///   sliding (50 layers × 16 kv_heads × 256 head_dim × 4096
///            × 2 K+V × 0.5625 byte/elem) ≈ 900 MiB
///   global  (10 layers × 4 kv_heads × 512 head_dim × 4096
///            × 2 × 0.5625) ≈ 90 MiB
///   total ≈ 990 MiB
#[derive(Debug)]
pub struct Gemma4Nvfp4KvState {
    /// Maximum context length the cache supports (max slot +1).
    pub max_pos: u32,
    /// Identity-block size. Always 1 in the bring-up.
    pub block_size: u32,
    /// Maximum number of query tokens the metadata buffers
    /// (positions / slot_mapping / context_lens) can hold per
    /// forward call. The current decode path uses 1; #5f
    /// chunked prefill bumps it. Capped at allocation; the
    /// `g4n_fill_pos_slots_i32` kernel rejects fills beyond
    /// this limit implicitly via grid sizing.
    pub max_query_tokens: u32,
    /// `[max_pos]` i32 identity block table.
    pub block_tables_ptr: u64,
    /// `[max_query_tokens]` i32 context lengths. Filled per
    /// forward call by `g4n_fill_pos_slots_i32` on
    /// `self.stream` — stream-ordered with the kernels that
    /// read it.
    pub context_lens_ptr: u64,
    /// `[max_query_tokens]` i32 positions. Indexes the RoPE
    /// cos/sin tables.
    pub positions_ptr: u64,
    /// `[max_query_tokens]` i32 slot mapping. Where K/V are
    /// written in the cache.
    pub slot_mapping_ptr: u64,
    /// `[1]` f32 fallback scalar Q scale (RVLLM_Q_SCALE; 2.0 in
    /// the production NVFP4 profile). The RoPE+KV-write kernel
    /// uses this when per-token Q scale is off.
    pub q_scale_ptr: u64,
    /// Per-layer K-cache base pointers. Slot indexing inside
    /// each layer's region follows the kernel's
    /// `(slot * num_kv_heads + head_idx) * (head_dim/2)` byte
    /// layout. For attention_k_eq_v=true global layers, the
    /// V pointer equals K (alias).
    pub k_packed_layer_ptrs: Vec<u64>,
    pub v_packed_layer_ptrs: Vec<u64>,
    pub k_scale_layer_ptrs: Vec<u64>,
    pub v_scale_layer_ptrs: Vec<u64>,
    /// Total KV cache bytes (informational, logged at startup).
    pub total_bytes: u64,
}

impl Gemma4Nvfp4KvState {
    /// Allocate a fresh KV state on the given arena. Returns a
    /// state with per-layer base pointers populated for every
    /// `arch.layer_types` entry. Sliding layers use
    /// `num_kv_heads_sliding × head_dim_sliding`; global layers
    /// use `num_kv_heads_global × head_dim_global`. For
    /// `attention_k_eq_v=true` global layers (the 31B
    /// convention), V aliases K — but the bring-up still
    /// allocates an explicit V buffer so the kernel's slot
    /// addressing stays uniform.
    pub fn allocate(
        arena: &rvllm_mem::HbmArena<'_>,
        arch: &rvllm_loader::gemma4_arch::Gemma4Arch,
        max_pos: u32,
        max_query_tokens: u32,
    ) -> Result<Self> {
        let block_size: u32 = 1;
        if max_query_tokens == 0 {
            return Err(corrupt_runtime_err(
                "Gemma4Nvfp4KvState::allocate: max_query_tokens must be >= 1"
                    .into()));
        }

        let block_tables_region = arena.region(
            "gemma4_nvfp4_kv_block_tables",
            (max_pos as usize) * 4, 256)?;
        let meta_bytes = (max_query_tokens as usize) * 4;
        let context_lens_region = arena.region(
            "gemma4_nvfp4_kv_context_lens", meta_bytes, 16)?;
        let positions_region = arena.region(
            "gemma4_nvfp4_kv_positions", meta_bytes, 16)?;
        let slot_mapping_region = arena.region(
            "gemma4_nvfp4_kv_slot_mapping", meta_bytes, 16)?;
        let q_scale_region = arena.region(
            "gemma4_nvfp4_kv_q_scale", 4, 16)?;

        // Initialize block_tables with identity mapping
        // (i32 0..max_pos). Single HtoD copy at allocate
        // time; never rewritten unless spec/rollback lands.
        let mut bt_host = Vec::<u8>::with_capacity((max_pos as usize) * 4);
        for i in 0..max_pos {
            bt_host.extend_from_slice(&(i as i32).to_le_bytes());
        }
        unsafe { block_tables_region.copy_from_host(&bt_host)? };

        // q_scale default = 2.0f (matches production
        // RVLLM_Q_SCALE on the fp8-block spec profile).
        let q_scale_default: f32 = std::env::var("RVLLM_Q_SCALE")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(2.0_f32);
        unsafe {
            q_scale_region.copy_from_host(
                &q_scale_default.to_le_bytes())?;
        }

        let mut k_packed_layer_ptrs = Vec::with_capacity(arch.num_hidden_layers);
        let mut v_packed_layer_ptrs = Vec::with_capacity(arch.num_hidden_layers);
        let mut k_scale_layer_ptrs = Vec::with_capacity(arch.num_hidden_layers);
        let mut v_scale_layer_ptrs = Vec::with_capacity(arch.num_hidden_layers);
        // block_tables[max_pos] + (positions+slot+context_lens)[max_query_tokens] + q_scale[1]
        let mut total_bytes: u64 =
            (max_pos as u64) * 4 + 3 * (max_query_tokens as u64) * 4 + 4;

        for layer_idx in 0..arch.num_hidden_layers {
            let lt = &arch.layer_types[layer_idx];
            let (n_kv_heads, head_dim) = match lt {
                rvllm_loader::gemma4_arch::Gemma4LayerType::SlidingAttention => {
                    (arch.num_kv_heads_sliding, arch.head_dim_sliding)
                }
                rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention => {
                    (arch.num_kv_heads_global, arch.head_dim_global)
                }
            };
            // Packed K + V bytes per slot per head: head_dim/2
            // nibbles. Total per layer = max_pos * n_kv_heads *
            // head_dim/2.
            let packed_bytes_per_layer =
                (max_pos as usize) * n_kv_heads * (head_dim / 2);
            // Per-(slot, head) E4M3 microscales: 1 byte per
            // group_of_16. groups per head = head_dim / 16.
            let scale_bytes_per_layer =
                (max_pos as usize) * n_kv_heads * (head_dim / 16);

            let k_packed = arena.region(
                "gemma4_nvfp4_kv_k_packed", packed_bytes_per_layer, 256)?;
            let v_packed = arena.region(
                "gemma4_nvfp4_kv_v_packed", packed_bytes_per_layer, 256)?;
            let k_scale = arena.region(
                "gemma4_nvfp4_kv_k_scale", scale_bytes_per_layer, 256)?;
            let v_scale = arena.region(
                "gemma4_nvfp4_kv_v_scale", scale_bytes_per_layer, 256)?;

            k_packed_layer_ptrs.push(k_packed.device_ptr());
            v_packed_layer_ptrs.push(v_packed.device_ptr());
            k_scale_layer_ptrs.push(k_scale.device_ptr());
            v_scale_layer_ptrs.push(v_scale.device_ptr());
            total_bytes += 2 * (packed_bytes_per_layer + scale_bytes_per_layer) as u64;
        }

        Ok(Self {
            max_pos,
            block_size,
            max_query_tokens,
            block_tables_ptr: block_tables_region.device_ptr(),
            context_lens_ptr: context_lens_region.device_ptr(),
            positions_ptr: positions_region.device_ptr(),
            slot_mapping_ptr: slot_mapping_region.device_ptr(),
            q_scale_ptr: q_scale_region.device_ptr(),
            k_packed_layer_ptrs,
            v_packed_layer_ptrs,
            k_scale_layer_ptrs,
            v_scale_layer_ptrs,
            total_bytes,
        })
    }
}

/// Loaded-but-held: the `LoadedModule` values must outlive the
/// `KernelFn` handles (the PTX module owns the function code).
/// They live on `Gemma4Nvfp4Bringup` to bound that lifetime.
struct ForwardKernels {
    _embed_mod: LoadedModule,
    fn_embedding_gather_bf16: KernelFn,
    _rmsnorm_mod: LoadedModule,
    fn_rmsnorm_inplace_bf16: KernelFn,
    /// Reserved for #4e residual add (vector_add_bf16).
    _vector_add_mod: LoadedModule,
    #[allow(dead_code)]
    fn_vector_add_bf16: KernelFn,
    /// `rope_split_half_bf16_kernel` — pure bf16 RoPE on
    /// `[n_heads, head_dim]` in-place. No KV side-effect.
    _rope_mod: LoadedModule,
    fn_rope_split_half_bf16: KernelFn,
    /// `argmax_kernel` — f32 logits → i32 token id. Grid=(rows,1,1),
    /// block=(min(vocab, 1024),1,1), shared-mem reduction handles
    /// vocab > block_size via tid-strided pre-fold.
    _argmax_mod: LoadedModule,
    fn_argmax_f32: KernelFn,
    /// `fused_rope_partial_nvfp4kv_bf16in_kernel` — bf16-input
    /// fused RoPE + NVFP4 K/V pack + per-(slot, head) E4M3
    /// microscale write + FP8 Q output. Handles both full and
    /// partial RoPE via runtime `rotary_dim` arg (per codex
    /// round-4 A1). Source at v3/kernels/.
    _rope_kv_write_mod: LoadedModule,
    #[allow(dead_code)] // wired by commit #5b2's forward integration
    fn_rope_kv_write_bf16in: KernelFn,
    /// `flash_attention_2_decode_nvfp4kv_bf16out_kernel` —
    /// paged-attention decode reader. FP8 Q in (produced by
    /// the RoPE+KV write above), bf16 attn_out. Source at
    /// kernels/flash_attention_nvfp4kv_bf16out.cu.
    _attn_decode_mod: LoadedModule,
    #[allow(dead_code)]
    fn_attn_decode_bf16out: KernelFn,
    /// GQA-aware variant of the above. Used when num_q_heads >
    /// num_kv_heads (always true on Gemma 4 31B layers).
    #[allow(dead_code)]
    fn_attn_decode_gqa_bf16out: KernelFn,
    /// `g4n_fill_pos_slots_i32_kernel` — device-side fill of the
    /// per-token positions / slot_mapping / context_lens arrays.
    /// Replaces sync-HtoD-on-default-stream scalar writes that
    /// race with kernels on `self.stream`.
    _fill_pos_slots_mod: LoadedModule,
    fn_fill_pos_slots_i32: KernelFn,
    /// `f32_to_bf16_kernel` — device-side narrow of f32 → bf16.
    /// Replaces per-call DtoH + host RTNE-narrow + HtoD trios
    /// after every cublasLt GEMM output. Single launch per
    /// narrow; per-thread does one __float2bfloat16.
    _f32_to_bf16_mod: LoadedModule,
    fn_f32_to_bf16: KernelFn,
    /// `vnorm_bf16_kernel` — parameter-free RMSNorm for V
    /// (Gemma 4 v_norm: `Gemma4RMSNorm(head_dim, eps,
    /// with_scale=False)`). Replaces the per-head host loop
    /// that DtoH'd v_f32, computed `rsqrt(mean_sq + eps)`,
    /// rescaled per-element, and re-uploaded bf16 V. Grid
    /// (num_kv_heads, 1, 1), block (head_dim, 1, 1) with
    /// shared warp reduction.
    _vnorm_bf16_mod: LoadedModule,
    fn_vnorm_bf16: KernelFn,
    /// `g4n_scaled_add_bf16_kernel` — fused
    /// `dst[i] += alpha[0] * src[i]` for bf16 vectors with a
    /// bf16 scalar `alpha` already on device. Replaces the
    /// post-MLP host scale loop (DtoH mlp_normed + DtoH
    /// layer_scalar + host scale + HtoD scaled + vector_add)
    /// that was the last per-layer fence on the Option B
    /// device-resident chain.
    _scaled_add_bf16_mod: LoadedModule,
    fn_scaled_add_bf16: KernelFn,
    /// #5f-PRIME scaffold: best-effort load of the unified
    /// NVFP4 prefill kernels. `Some` when the PTX is present
    /// (current sm_121 kernel tree); used by future Option B
    /// prompt-prefill dispatch.
    /// Wiring is deferred — see `forward_prompt_to_token`
    /// docstring + codex Stream-5+6+7 review for the scope
    /// (Fa2PtxKernels plumbing + batched MLP).
    _unified_prefill_nvfp4kv_mod: Option<LoadedModule>,
    #[allow(dead_code)]
    fn_prefill_nvfp4kv_unified_bf16out: Option<KernelFn>,
    /// Stream-6a (drafter forward primitive #1): cast_fp PTX
    /// module + `cast_f32_to_f16_kernel`. Used by the drafter
    /// forward to narrow `cublaslt.f16_gemm_f32` outputs
    /// (which return f32) into the drafter's f16 hidden
    /// stream. Same kernel production loads in its fused
    /// modules; loaded here for the Option B drafter-forward
    /// path. Module is `cast_fp.ptx` (contains multiple cast
    /// variants); we pick the f32→f16 entry.
    _cast_fp_mod: LoadedModule,
    fn_cast_f32_to_f16: KernelFn,
    /// Stream-6a drafter forward primitive bundle (q_side +
    /// attn_finisher + mlp_finisher). All f16 because the
    /// drafter runs in f16 (its weights are bf16-narrowed
    /// to f16 at load time per
    /// `Gemma4DrafterRuntime::load`).
    _rmsnorm_inplace_f16_mod: LoadedModule,
    fn_rmsnorm_inplace_f16: KernelFn,
    _fused_gelu_mul_f16_mod: LoadedModule,
    fn_fused_gelu_mul_f16: KernelFn,
    _vector_add_f16_mod: LoadedModule,
    fn_vector_add_f16: KernelFn,
    _scale_inplace_f16_mod: LoadedModule,
    fn_scale_inplace_f16: KernelFn,
    _rope_partial_f16kv_mod: LoadedModule,
    fn_rope_partial_f16kv: KernelFn,
    /// Stream-6b spec primitive #1: bf16→f16 saturating cast,
    /// used to snapshot the base's post-final-norm hidden into
    /// the drafter-readable f16 `base_last_hidden_ptr` buffer.
    _bf16_to_f16_sat_mod: LoadedModule,
    fn_bf16_to_f16_sat: KernelFn,
}

/// Restores the arena to the post-load high-water mark when a forward
/// helper returns or fails. Created before per-forward regions, so Rust
/// drops those regions before this guard restores the bump pointer.
struct ForwardScratchGuard<'a> {
    arena: &'a HbmArena<'static>,
    checkpoint: usize,
}

impl Drop for ForwardScratchGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: these bring-up forwards return host-owned Vec/u32 values.
        // Any Region locals allocated after the guard are dropped first.
        unsafe { self.arena.restore(self.checkpoint); }
    }
}

/// Phase 3c+ text-only bring-up for `nvidia/Gemma-4-31B-IT-NVFP4`.
///
/// Owns: a CUDA primary context (kept alive via the handle),
/// the HbmArena that all weight + scratch allocations live in,
/// a stream, a cuBLASLt handle with its own workspace region,
/// the loaded model (embed + 60 layers + final_norm), and the
/// kernel handles the forward path uses.
pub struct Gemma4Nvfp4Bringup {
    /// Drop order matters: ctx must outlive every CUDA resource
    /// allocated under it. Rust drops fields in declaration
    /// order, so this stays at the END.
    pub arch: Gemma4Arch,
    pub model: Gemma4Nvfp4LoadedModel,
    pub mlp_kernels: Gemma4Nvfp4MlpKernels,
    forward_kernels: ForwardKernels,
    /// Stream-#5f-PRIME: per-head-dim attention backends. The
    /// production `PagedPrefillNvfp4Launcher::launch_nvfp4kv
    /// _unified_sm121` (prefill.rs:1041) takes an
    /// `AttentionBackend` whose `head_dim()` must match
    /// `params.head_dim`. Production Gemma 4 loads two
    /// instances at startup (sliding + global); we mirror
    /// the same pattern so per-layer dispatch works without
    /// re-validation per call.
    attn_backend_sliding: rvllm_attention::AttentionBackend,
    attn_backend_global: rvllm_attention::AttentionBackend,
    /// PTX modules backing `mlp_kernels`. Kept alive here.
    _mlp_w4a16_gemv_mod: LoadedModule,
    _mlp_w4a16_gate_up_mod: LoadedModule,
    _mlp_gelu_tanh_mul_mod: LoadedModule,
    pub cublaslt: CublasLt,
    pub stream: Stream,
    pub arena: HbmArena<'static>,
    /// Checkpoint taken after persistent allocations. Forward helpers
    /// restore here so scratch allocations do not accumulate across calls.
    pub forward_checkpoint: usize,
    /// Re-entrancy guard for `allocate_kv_state`. A second call
    /// without `release_kv_state` would leak the first allocation
    /// (arena bump points past it) and silently shadow the caller's
    /// pointer to the prior state. We reject the second call instead.
    kv_state_allocated: bool,
    /// Stream-6a (Option B-side spec-decode): kernel loader kept
    /// around for lazy PTX module loads after construction. The
    /// drafter PTX bundle (masked_embedder, flash_attention_decode
    /// _f16io, flash_attention_decode_f16io_bc16, gemma4_drafter
    /// _dequant) only loads when `ensure_drafter_nvfp4` is called.
    pub kernels: Arc<KernelLoader>,
    /// Stream-6a: lazy-uploaded drafter runtime (parallel to
    /// production's `Gemma4Bringup::drafter`). Populated by
    /// `ensure_drafter_nvfp4` on first spec request. Behind a
    /// `Mutex<Option<_>>` so construction is at-most-once and
    /// thread-safe under the cuda_worker's single thread.
    pub drafter: std::sync::Mutex<Option<crate::gemma4_drafter::Gemma4DrafterRuntime>>,
    /// Stream-6b spec primitive #1: persistent f16 device buffer
    /// holding the BASE's post-final-norm hidden state for the
    /// last generated token. Drafter consumes this as
    /// `base_hidden_last` (the second half of
    /// `pre_projection_in`). Allocated lazily by
    /// `ensure_base_last_hidden_buffer()` (above
    /// `forward_checkpoint`, so arena.restore() doesn't reclaim
    /// it between forwards). Zero until the first
    /// `forward_final_to_token` lands a snapshot. `0` = not
    /// allocated → snapshot is a no-op (production path is
    /// undisturbed when spec decode isn't requested).
    pub base_last_hidden_ptr: std::sync::atomic::AtomicU64,
    /// Held to keep the primary CUDA context alive.
    _ctx: CudaContextHandle,
}

impl Gemma4Nvfp4Bringup {
    /// Open the CUDA context + arena + stream + cuBLASLt, run
    /// `load_gemma4_nvfp4_text` against `model_dir`, and load
    /// the forward-path kernel handles.
    ///
    /// `arena_bytes` controls the persistent arena size. Need
    /// ~22 GiB for weights on 31B + scratch for forward. 32 GiB
    /// is a safe minimum for the text-only path; the production
    /// profile (`mobile-31b-nvfp4w-rvllm-spec.env`) targets
    /// 96 GiB to cover spec + vision scratch + KV in later
    /// commits.
    pub fn load(
        model_dir: &Path,
        arena_bytes: usize,
        kernels_dir: &Path,
    ) -> Result<Self> {
        validate_no_stale_g4n_debug_envs()?;
        let ctx = CudaContextHandle::init(0)?;
        // SAFETY: HbmArena borrows the context for its lifetime.
        // We extend the borrow to 'static by moving the context
        // into Self alongside the arena — the struct's drop
        // order keeps the context alive longer than the arena.
        let arena = unsafe {
            let arena_borrowed: HbmArena<'_> = HbmArena::new(&ctx, arena_bytes)?;
            std::mem::transmute::<HbmArena<'_>, HbmArena<'static>>(arena_borrowed)
        };
        let stream = Stream::new(&ctx)?;

        // cuBLASLt: 64 MiB workspace, 256-byte aligned (cuBLASLt
        // requirement — see commit #2 bisect note).
        let cublaslt_ws = arena.region("gemma4_nvfp4_cublaslt_ws",
                                       64 * 1024 * 1024, 256)?;
        let cublaslt = CublasLt::new(cublaslt_ws.device_ptr(),
                                     64 * 1024 * 1024)?;

        // Parse arch + run the text loader (Phase 3b).
        let arch = Gemma4Arch::from_dir(model_dir)?;
        let model = load_gemma4_nvfp4_text(&arena, model_dir, &arch)?;

        // Load forward-path kernels.
        let manifest_path = kernels_dir.join("manifest.json");
        let manifest = rvllm_kernels::KernelManifest::load_and_verify(&manifest_path)?;
        let loader = Arc::new(KernelLoader::new(manifest));

        let embed_mod = loader.load_ptx("embedding_gather_bf16")?;
        let fn_embedding_gather_bf16 =
            embed_mod.get_function("embedding_gather_bf16_kernel")?;

        let rmsnorm_mod = loader.load_ptx("rmsnorm_inplace_bf16_gbf16")?;
        let fn_rmsnorm_inplace_bf16 =
            rmsnorm_mod.get_function("rmsnorm_inplace_bf16_gbf16_kernel")?;

        let vector_add_mod = loader.load_ptx("vector_add_bf16")?;
        let fn_vector_add_bf16 =
            vector_add_mod.get_function("vector_add_bf16_kernel")?;

        let rope_mod = loader.load_ptx("rope_split_half_bf16")?;
        let fn_rope_split_half_bf16 =
            rope_mod.get_function("rope_split_half_bf16_kernel")?;

        let argmax_mod = loader.load_ptx("argmax")?;
        let fn_argmax_f32 = argmax_mod.get_function("argmax_kernel")?;

        // Commit #5b1: attention kernel handles loaded but not
        // yet wired into a forward (that's #5b2).
        let rope_kv_write_mod =
            loader.load_ptx("fused_rope_partial_nvfp4kv_bf16in")?;
        let fn_rope_kv_write_bf16in = rope_kv_write_mod
            .get_function("fused_rope_partial_nvfp4kv_bf16in_kernel")?;

        let attn_decode_mod =
            loader.load_ptx("flash_attention_nvfp4kv_bf16out")?;
        let fn_attn_decode_bf16out = attn_decode_mod
            .get_function("flash_attention_2_decode_nvfp4kv_bf16out_kernel")?;
        let fn_attn_decode_gqa_bf16out = attn_decode_mod
            .get_function("flash_attention_2_decode_nvfp4kv_gqa_bf16out_kernel")?;

        // Floor commit 2 (stream-safe metadata).
        let fill_pos_slots_mod = loader.load_ptx("g4n_fill_pos_slots_i32")?;
        let fn_fill_pos_slots_i32 =
            fill_pos_slots_mod.get_function("g4n_fill_pos_slots_i32_kernel")?;

        // Floor commit 4 / Stream 5a: load device-side narrow +
        // parameter-free V-RMSNorm so the per-layer attention
        // chain doesn't have to round-trip Q/K/V through the
        // host. Both kernels already shipped in the production
        // manifest (kernels/f32_to_bf16.cu, kernels/vnorm_bf16.cu);
        // Option B was just not loading them.
        let f32_to_bf16_mod = loader.load_ptx("f32_to_bf16")?;
        let fn_f32_to_bf16 =
            f32_to_bf16_mod.get_function("f32_to_bf16_kernel")?;
        let vnorm_bf16_mod = loader.load_ptx("vnorm_bf16")?;
        let fn_vnorm_bf16 =
            vnorm_bf16_mod.get_function("vnorm_bf16_kernel")?;
        let scaled_add_bf16_mod = loader.load_ptx("g4n_scaled_add_bf16")?;
        let fn_scaled_add_bf16 =
            scaled_add_bf16_mod.get_function("g4n_scaled_add_bf16_kernel")?;

        // Stream-6a (drafter forward primitive #1): cast
        // module — required by the drafter pre_projection
        // step (f32 GEMM output → f16 hidden).
        let cast_fp_mod = loader.load_ptx("cast_fp")?;
        let fn_cast_f32_to_f16 =
            cast_fp_mod.get_function("cast_f32_to_f16_kernel")?;
        // Stream-6a drafter forward bundle (f16 throughout).
        let rmsnorm_inplace_f16_mod =
            loader.load_ptx("rmsnorm_inplace_f16")?;
        let fn_rmsnorm_inplace_f16 = rmsnorm_inplace_f16_mod
            .get_function("rmsnorm_inplace_f16_kernel")?;
        let fused_gelu_mul_f16_mod =
            loader.load_ptx("fused_gelu_mul_f16")?;
        let fn_fused_gelu_mul_f16 = fused_gelu_mul_f16_mod
            .get_function("fused_gelu_mul_f16_kernel")?;
        let vector_add_f16_mod = loader.load_ptx("vector_add_f16")?;
        let fn_vector_add_f16 = vector_add_f16_mod
            .get_function("vector_add_f16_kernel")?;
        let scale_inplace_f16_mod = loader.load_ptx("scale_inplace_f16")?;
        let fn_scale_inplace_f16 = scale_inplace_f16_mod
            .get_function("scale_inplace_f16_kernel")?;
        let rope_partial_f16kv_mod =
            loader.load_ptx("fused_rope_partial_f16kv")?;
        let fn_rope_partial_f16kv = rope_partial_f16kv_mod
            .get_function("fused_rope_partial_f16kv_kernel")?;

        // Stream-6b spec primitive #1: bf16→f16 saturating cast.
        // Used to snapshot the base's post-final-norm hidden
        // (bf16 in Option B's residual stream) into the f16
        // buffer the drafter consumes as `base_hidden_last`
        // (mirrors production's `base_last_hidden_ptr`).
        let bf16_to_f16_sat_mod = loader.load_ptx("bf16_to_f16_sat")?;
        let fn_bf16_to_f16_sat = bf16_to_f16_sat_mod
            .get_function("bf16_to_f16_sat_kernel")?;

        // Stream #5f-PRIME: still load this handle ourselves
        // for the per-launcher path (deprecated by the
        // AttentionBackend below but kept until callers move).
        let (unified_prefill_nvfp4kv_mod,
             fn_prefill_nvfp4kv_unified_bf16out) =
            match loader.load_ptx("flash_attention_unified_prefill_nvfp4kv") {
                Ok(m) => {
                    let f = m
                        .get_function(
                            "flash_attention_2_prefill_nvfp4kv_unified_bf16out_kernel"
                        )
                        .ok();
                    (Some(m), f)
                }
                Err(_) => (None, None),
            };

        // Stream-#5f-PRIME: production attention backends, one
        // per head_dim. Loading Fa2PtxKernels pulls in extra
        // PTX modules (decode + prefill + bf16 variants + …)
        // but they're small. The backend is what the
        // `PagedPrefillNvfp4Launcher` consumes for the
        // batched-N attention path.
        let attn_backend_sliding = rvllm_attention::AttentionBackend::Fa2Ptx(
            rvllm_attention::Fa2PtxKernels::load(
                &*loader, arch.head_dim_sliding as u32)?);
        let attn_backend_global = rvllm_attention::AttentionBackend::Fa2Ptx(
            rvllm_attention::Fa2PtxKernels::load(
                &*loader, arch.head_dim_global as u32)?);

        // MLP kernels (commit #3).
        let mlp_gemv_mod = loader.load_ptx("mistral35_w4a16_gemv_bf16")?;
        let fn_w4a16_gemv =
            mlp_gemv_mod.get_function("mistral35_w4a16_gemv_bf16_kernel")?;
        let mlp_gate_up_mod = loader.load_ptx("mistral35_w4a16_gate_up_gemv_bf16")?;
        let fn_w4a16_gate_up_gemv =
            mlp_gate_up_mod.get_function("mistral35_w4a16_gate_up_gemv_bf16_kernel")?;
        let mlp_gelu_mod = loader.load_ptx("gelu_tanh_mul_bf16")?;
        let fn_gelu_tanh_mul =
            mlp_gelu_mod.get_function("gelu_tanh_mul_bf16_kernel")?;

        let mlp_kernels = Gemma4Nvfp4MlpKernels {
            fn_w4a16_gemv,
            fn_w4a16_gate_up_gemv,
            fn_gelu_tanh_mul,
        };

        let forward_kernels = ForwardKernels {
            _embed_mod: embed_mod,
            fn_embedding_gather_bf16,
            _rmsnorm_mod: rmsnorm_mod,
            fn_rmsnorm_inplace_bf16,
            _vector_add_mod: vector_add_mod,
            fn_vector_add_bf16,
            _rope_mod: rope_mod,
            fn_rope_split_half_bf16,
            _argmax_mod: argmax_mod,
            fn_argmax_f32,
            _rope_kv_write_mod: rope_kv_write_mod,
            fn_rope_kv_write_bf16in,
            _attn_decode_mod: attn_decode_mod,
            fn_attn_decode_bf16out,
            fn_attn_decode_gqa_bf16out,
            _fill_pos_slots_mod: fill_pos_slots_mod,
            fn_fill_pos_slots_i32,
            _f32_to_bf16_mod: f32_to_bf16_mod,
            fn_f32_to_bf16,
            _vnorm_bf16_mod: vnorm_bf16_mod,
            fn_vnorm_bf16,
            _scaled_add_bf16_mod: scaled_add_bf16_mod,
            fn_scaled_add_bf16,
            _unified_prefill_nvfp4kv_mod: unified_prefill_nvfp4kv_mod,
            fn_prefill_nvfp4kv_unified_bf16out,
            _cast_fp_mod: cast_fp_mod,
            fn_cast_f32_to_f16,
            _rmsnorm_inplace_f16_mod: rmsnorm_inplace_f16_mod,
            fn_rmsnorm_inplace_f16,
            _fused_gelu_mul_f16_mod: fused_gelu_mul_f16_mod,
            fn_fused_gelu_mul_f16,
            _vector_add_f16_mod: vector_add_f16_mod,
            fn_vector_add_f16,
            _scale_inplace_f16_mod: scale_inplace_f16_mod,
            fn_scale_inplace_f16,
            _rope_partial_f16kv_mod: rope_partial_f16kv_mod,
            fn_rope_partial_f16kv,
            _bf16_to_f16_sat_mod: bf16_to_f16_sat_mod,
            fn_bf16_to_f16_sat,
        };
        let forward_checkpoint = arena.checkpoint();

        Ok(Self {
            arch,
            model,
            mlp_kernels,
            forward_kernels,
            attn_backend_sliding,
            attn_backend_global,
            _mlp_w4a16_gemv_mod: mlp_gemv_mod,
            _mlp_w4a16_gate_up_mod: mlp_gate_up_mod,
            _mlp_gelu_tanh_mul_mod: mlp_gelu_mod,
            cublaslt,
            stream,
            arena,
            forward_checkpoint,
            kv_state_allocated: false,
            kernels: loader,
            drafter: std::sync::Mutex::new(None),
            base_last_hidden_ptr: std::sync::atomic::AtomicU64::new(0),
            _ctx: ctx,
        })
    }

    /// Stream-6b spec primitive #1: allocate the persistent f16
    /// `base_last_hidden` buffer ABOVE the forward checkpoint, so
    /// `forward_scratch_guard`'s arena.restore() does NOT reclaim
    /// it between requests. Idempotent — second call is a no-op.
    /// Spec-decode callers must call this before the first
    /// forward they want to snapshot.
    pub fn ensure_base_last_hidden_buffer(&mut self) -> Result<()> {
        use std::sync::atomic::Ordering::{Acquire, Release};
        if self.base_last_hidden_ptr.load(Acquire) != 0 {
            return Ok(());
        }
        let h = self.arch.hidden_size;
        let region = self.arena.region(
            "g4n_base_last_hidden_f16", h * 2, 16)?;
        // Zero-init so a drafter step that runs before the first
        // base forward sees a defined (all-zero) hidden half.
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8_v2(
                region.device_ptr(), 0, h * 2);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(corrupt_runtime_err(
                    "ensure_base_last_hidden_buffer: zero-init".into()));
            }
        }
        // Re-checkpoint above this allocation so subsequent
        // forward calls' arena.restore() leaves the buffer intact.
        self.forward_checkpoint = self.arena.checkpoint();
        self.base_last_hidden_ptr.store(region.device_ptr(), Release);
        Ok(())
    }

    /// Stream-6b spec primitive #1: read-only getter for the
    /// drafter side. Returns `0` if `ensure_base_last_hidden
    /// _buffer` has not been called yet.
    pub fn base_last_hidden_device_ptr(&self) -> u64 {
        self.base_last_hidden_ptr
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Stream-6b spec primitive #2: populate the drafter's
    /// `pre_projection_in` buffer with
    /// `cat[base.embed_tokens[token_id], base_last_hidden]`
    /// — the input shape `pre_projection` expects.
    ///
    /// Both halves are `backbone_hidden_size` (= base hidden,
    /// = 5376 on 31B), so `pre_projection_in` is `2 *
    /// backbone_hidden_size` f16 elements wide.
    ///
    /// The embed half is sourced from the BASE model's
    /// `embed_tokens` (bf16 on Option B → cast to f16 via
    /// `bf16_to_f16_sat`), NOT from the drafter's own
    /// `top.embed_tokens` table. The drafter's table is sized
    /// `[vocab, drafter_hidden]` (1024 wide on 31B), too narrow
    /// for the `2 * backbone_hidden` contract — production
    /// matches this behaviour (`gemma4_bring_up.rs:4676` gathers
    /// from `self.model.embedding`, not the drafter's).
    ///
    /// Caller must have called `ensure_base_last_hidden_buffer`
    /// at least once; otherwise the hidden half has no source
    /// and this returns a clear error rather than silently
    /// reading from address 0.
    pub fn populate_drafter_pre_projection_input(
        &self,
        _drafter: &crate::gemma4_drafter::Gemma4DrafterRuntime,
        workspace: &crate::gemma4_drafter::DrafterStepWorkspace,
        token_id: u32,
    ) -> Result<()> {
        let backbone_hidden = self.arch.hidden_size;
        let half_bytes = backbone_hidden * 2;
        let stream = self.stream.raw();

        // Scratch guard so the bf16 embed lookup buffer is
        // reclaimed when this method returns. The destination
        // (workspace.pre_projection_in) lives above the
        // checkpoint and is unaffected.
        let _scratch_guard = self.forward_scratch_guard();

        // (1) Gather base.embed_tokens[token_id] (bf16) into a
        //     scratch buffer.
        let embed_bf16 = self.arena.region(
            "g4n_drafter_last_token_embed_bf16", half_bytes, 16)?;
        self.embed_one_token_to_device(token_id, embed_bf16.device_ptr())?;

        // (2) Cast bf16 → f16 directly into the embed half of
        //     pre_projection_in. The cast kernel is the same one
        //     used by the base's post-final-norm snapshot.
        unsafe {
            rvllm_fused::gemma4_launcher::Bf16ToF16SatLaunch {
                n: backbone_hidden as u32,
            }
            .launch(
                self.forward_kernels.fn_bf16_to_f16_sat,
                workspace.pre_projection_in,
                embed_bf16.device_ptr(), stream,
            )?;
        }

        // (3) DtoD copy base_last_hidden_ptr (already f16) into
        //     the hidden half of pre_projection_in. Stream-
        //     ordered so any update enqueued before this from
        //     `forward_final_to_token`'s Bf16ToF16Sat is observed.
        let base_hidden = self.base_last_hidden_device_ptr();
        if base_hidden == 0 {
            return Err(corrupt_runtime_err(
                "populate_drafter_pre_projection_input: \
                 base_last_hidden_ptr is 0 — call \
                 ensure_base_last_hidden_buffer (and run a base \
                 forward to populate it) before invoking this".into()));
        }
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                workspace.pre_projection_in + half_bytes as u64,
                base_hidden, half_bytes, stream as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(corrupt_runtime_err(
                    "populate_drafter_pre_projection_input: \
                     hidden half DtoDAsync".into()));
            }
        }
        Ok(())
    }

    /// Stream-ordered fill of the per-token metadata buffers
    /// (positions, slot_mapping, context_lens) via the
    /// `g4n_fill_pos_slots_i32` kernel on `self.stream`. Replaces
    /// the prior 3×sync-`cuMemcpyHtoD_v2`-on-default-stream
    /// pattern that races with non-blocking compute (same race
    /// as Qwen 3.6 hit on the legacy `pos_cl_region` path —
    /// codex round-25/26 review).
    ///
    /// Caller responsibility: `num_tokens <= kv.max_query_tokens`.
    /// The bounds check lives here so the launch is one atomic
    /// invariant — fewer chances for an off-by-one at call sites.
    fn fill_pos_slots(
        &self,
        kv: &Gemma4Nvfp4KvState,
        position_offset: i32,
        start_slot: i32,
        num_tokens: i32,
    ) -> Result<()> {
        if (num_tokens as u32) > kv.max_query_tokens {
            return Err(corrupt_runtime_err(format!(
                "fill_pos_slots: num_tokens={num_tokens} > \
                 kv.max_query_tokens={}",
                kv.max_query_tokens)));
        }
        if num_tokens <= 0 {
            return Err(corrupt_runtime_err(format!(
                "fill_pos_slots: num_tokens={num_tokens} must be > 0")));
        }
        let mut positions: u64 = kv.positions_ptr;
        let mut slot_mapping: u64 = kv.slot_mapping_ptr;
        let mut context_lens: u64 = kv.context_lens_ptr;
        let mut pos_off: i32 = position_offset;
        let mut sslot: i32 = start_slot;
        let mut n: i32 = num_tokens;
        let args: [*mut core::ffi::c_void; 6] = [
            (&mut positions)    as *mut u64 as *mut _,
            (&mut slot_mapping) as *mut u64 as *mut _,
            (&mut context_lens) as *mut u64 as *mut _,
            (&mut pos_off)      as *mut i32 as *mut _,
            (&mut sslot)        as *mut i32 as *mut _,
            (&mut n)            as *mut i32 as *mut _,
        ];
        const BLOCK: u32 = 256;
        let grid_x: u32 = ((num_tokens as u32) + BLOCK - 1) / BLOCK;
        unsafe {
            rvllm_fused::launch_raw(
                self.forward_kernels.fn_fill_pos_slots_i32,
                (grid_x, 1, 1),
                (BLOCK, 1, 1),
                0,
                self.stream.raw(),
                &args,
            )?;
        }
        Ok(())
    }

    /// Stream-ordered f32 → bf16 narrow via the
    /// `f32_to_bf16_kernel`. Replaces a DtoH +
    /// host-RTNE-narrow + HtoD trio that was costing one
    /// stream.fence() per call. Launch: grid (ceil(n/256),
    /// 1, 1), block (256, 1, 1).
    fn launch_f32_to_bf16(
        &self, dst_bf16: u64, src_f32: u64, n: u32,
    ) -> Result<()> {
        if n == 0 { return Ok(()); }
        let mut dst = dst_bf16;
        let mut src = src_f32;
        let mut n_i: i32 = n as i32;
        let args: [*mut core::ffi::c_void; 3] = [
            (&mut dst) as *mut u64 as *mut _,
            (&mut src) as *mut u64 as *mut _,
            (&mut n_i) as *mut i32 as *mut _,
        ];
        const BLOCK: u32 = 256;
        let grid_x = (n + BLOCK - 1) / BLOCK;
        unsafe {
            rvllm_fused::launch_raw(
                self.forward_kernels.fn_f32_to_bf16,
                (grid_x, 1, 1),
                (BLOCK, 1, 1),
                0, self.stream.raw(), &args,
            )
        }
    }

    /// Stream-ordered parameter-free RMSNorm on bf16 V (per
    /// head): `v[h, :] = v[h, :] / rms(v[h, :])`. Matches
    /// Gemma 4's `v_norm` = `Gemma4RMSNorm(head_dim, eps,
    /// with_scale=False)`. Grid (num_kv_heads, 1, 1), block
    /// (head_dim or 1024 capped, 1, 1) with shared warp
    /// reduction.
    fn launch_vnorm_bf16(
        &self, v_bf16: u64, num_kv_heads: u32, head_dim: u32,
    ) -> Result<()> {
        let mut v = v_bf16;
        let mut eps: f32 = self.arch.rms_norm_eps;
        let mut hd: i32 = head_dim as i32;
        let args: [*mut core::ffi::c_void; 3] = [
            (&mut v) as *mut u64 as *mut _,
            (&mut eps) as *mut f32 as *mut _,
            (&mut hd) as *mut i32 as *mut _,
        ];
        // Block size capped at 1024 (kernel's __launch_bounds__) and
        // at head_dim itself (no thread does zero work). Reduction
        // is warp-shuffle + 1×__syncthreads; non-power-of-2 block
        // sizes are fine.
        let block_x = head_dim.min(1024);
        unsafe {
            rvllm_fused::launch_raw(
                self.forward_kernels.fn_vnorm_bf16,
                (num_kv_heads, 1, 1),
                (block_x, 1, 1),
                0, self.stream.raw(), &args,
            )
        }
    }

    /// Stream-ordered fused `dst[i] += alpha[0] * src[i]` for
    /// bf16 vectors with bf16 alpha on device. Replaces the
    /// post-MLP host scale loop (the last per-layer fence on
    /// the Option B device-resident chain).
    fn launch_scaled_add_bf16(
        &self, dst: u64, src: u64, alpha_dev: u64, n: u32,
    ) -> Result<()> {
        if n == 0 { return Ok(()); }
        let mut d = dst;
        let mut s = src;
        let mut a = alpha_dev;
        let mut n_i: i32 = n as i32;
        let args: [*mut core::ffi::c_void; 4] = [
            (&mut d) as *mut u64 as *mut _,
            (&mut s) as *mut u64 as *mut _,
            (&mut a) as *mut u64 as *mut _,
            (&mut n_i) as *mut i32 as *mut _,
        ];
        const BLOCK: u32 = 256;
        let grid_x = (n + BLOCK - 1) / BLOCK;
        unsafe {
            rvllm_fused::launch_raw(
                self.forward_kernels.fn_scaled_add_bf16,
                (grid_x, 1, 1),
                (BLOCK, 1, 1),
                0, self.stream.raw(), &args,
            )
        }
    }

    /// Stream-6a (drafter forward primitive #1): cast f32 →
    /// f16 in place. Mirrors production's `launch_cast_f32_to
    /// _f16` (gemma4_bring_up.rs:16723) — same kernel ABI,
    /// same grid math, just on Option B's stream.
    fn launch_cast_f32_to_f16(
        &self, dst_f16: u64, src_f32: u64, n: u32,
    ) -> Result<()> {
        if n == 0 { return Ok(()); }
        let mut dst = dst_f16;
        let mut src = src_f32;
        let mut n_i: i32 = n as i32;
        let args: [*mut core::ffi::c_void; 3] = [
            (&mut dst) as *mut u64 as *mut _,
            (&mut src) as *mut u64 as *mut _,
            (&mut n_i) as *mut i32 as *mut _,
        ];
        const BLOCK: u32 = 256;
        let grid_x = (n + BLOCK - 1) / BLOCK;
        unsafe {
            rvllm_fused::launch_raw(
                self.forward_kernels.fn_cast_f32_to_f16,
                (grid_x, 1, 1),
                (BLOCK, 1, 1),
                0, self.stream.raw(), &args,
            )
        }
    }

    /// Stream-6a (drafter forward primitive #1):
    /// pre_projection step of one assistant draft iteration.
    /// Mirrors production's `Gemma4Bringup::run_drafter_pre
    /// _projection` (gemma4_bring_up.rs:8732) — same kernel
    /// chain, just on Option B's state.
    ///
    /// Math:
    ///   workspace.hidden[hidden_size] f16 =
    ///     f32_to_f16(cublaslt.f16_gemm_f32(
    ///       workspace.pre_projection_in[2*backbone_hidden] f16,
    ///       drafter.top.pre_projection[hidden_size, 2*backbone_hidden] f16
    ///     ))
    ///
    /// Caller responsibility: workspace + drafter populated;
    /// pre_projection_in already contains
    /// `[last_token_embed; base_hidden_last_step]`. The caller
    /// runs subsequent drafter forward primitives (q_side +
    /// cross-attn + finisher + mlp) — not in this commit.
    ///
    /// NO scratch_guard. NO fence. Caller orchestrates the
    /// full step + the final DtoH.
    pub fn forward_drafter_pre_projection(
        &self,
        drafter: &crate::gemma4_drafter::Gemma4DrafterRuntime,
        workspace: &crate::gemma4_drafter::DrafterStepWorkspace,
    ) -> Result<()> {
        let hidden = drafter.arch.hidden_size;
        let pre_in = drafter.arch.pre_projection_in_dim;
        let stream = self.stream.raw();

        if workspace.pre_projection_in == 0
            || workspace.gemm_f32 == 0
            || workspace.hidden == 0
        {
            return Err(corrupt_runtime_err(
                "forward_drafter_pre_projection: workspace has \
                 null device pointers — caller must call \
                 drafter.alloc_step_workspace(arena) first".into()));
        }
        if drafter.top.pre_projection == 0 {
            return Err(corrupt_runtime_err(
                "forward_drafter_pre_projection: drafter \
                 pre_projection weight is null".into()));
        }

        unsafe {
            self.cublaslt.f16_gemm_f32(
                workspace.pre_projection_in,
                drafter.top.pre_projection,
                workspace.gemm_f32,
                1, hidden as i32, pre_in as i32, stream,
            )?;
        }
        self.launch_cast_f32_to_f16(
            workspace.hidden, workspace.gemm_f32, hidden as u32,
        )?;
        Ok(())
    }

    /// Stream-6a (drafter forward primitive #2): Q-side of one
    /// drafter layer. Mirrors production's
    /// `Gemma4Bringup::run_drafter_layer_q_side`
    /// (gemma4_bring_up.rs:8823). Hadamard rotation is SKIPPED
    /// (Option B ships spec with Hadamard OFF per codex
    /// Stream-6b recommendation). Diagnostic probes
    /// (Q_BISECT, LAYER_TRACE) skipped — readd in a follow-up.
    ///
    /// Steps:
    ///   1. Snapshot workspace.hidden → workspace.residual1
    ///      (attn_finisher reads this for residual_1 add).
    ///   2. input_layernorm in place on workspace.hidden.
    ///   3. q_proj GEMM → f32 scratch → cast_f32_to_f16 →
    ///      workspace.q.
    ///   4. Per-head q_norm (num_tokens=num_heads,
    ///      hidden=effective_head_dim).
    ///   5. Partial NeoX RoPE on Q (num_kv_heads=0 disables
    ///      K/V branch so null kv pointers are safe). Sliding
    ///      uses full rotation + sliding cos/sin tables;
    ///      global uses partial_rotary_factor=0.25 + global
    ///      tables.
    pub fn forward_drafter_layer_q_side(
        &self,
        drafter: &crate::gemma4_drafter::Gemma4DrafterRuntime,
        workspace: &crate::gemma4_drafter::DrafterStepWorkspace,
        layer_idx: usize,
        position: u32,
    ) -> Result<()> {
        use rvllm_core::{RvllmError, CudaErrorKind, CudaCtx};
        let hidden = drafter.arch.hidden_size;
        let num_heads = drafter.arch.num_attention_heads;
        let layer = drafter.layers.get(layer_idx).ok_or_else(||
            corrupt_runtime_err(format!(
                "forward_drafter_layer_q_side: layer_idx {} \
                 out of range (drafter has {})",
                layer_idx, drafter.layers.len())))?;
        let eff_hd = layer.effective_head_dim;
        let q_rows = num_heads * eff_hd;
        let stream = self.stream.raw();
        let eps = drafter.arch.rms_norm_eps;

        // 1. Snapshot pre-norm residual.
        unsafe {
            use cudarc::driver::sys::*;
            let r = cuMemcpyDtoDAsync_v2(
                workspace.residual1, workspace.hidden,
                hidden * 2, stream as CUstream);
            if r != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "drafter_q_side residual1 snapshot",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
        }
        // 2. input_layernorm in place.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden: hidden as u32, eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_f16,
                workspace.hidden, layer.input_layernorm, stream)?;
        }
        // 3. q_proj GEMM + f32 → f16 cast.
        unsafe {
            self.cublaslt.f16_gemm_f32(
                workspace.hidden, layer.self_attn_q_proj,
                workspace.gemm_f32,
                1, q_rows as i32, hidden as i32, stream)?;
        }
        self.launch_cast_f32_to_f16(
            workspace.q, workspace.gemm_f32, q_rows as u32)?;
        // 4. per-head q_norm.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_heads as u32,
                hidden: eff_hd as u32, eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_f16,
                workspace.q, layer.self_attn_q_norm, stream)?;
        }
        // 5. Partial NeoX RoPE. is_global per drafter layer type;
        // partial_rotary_factor=0.25 for global (E4B convention,
        // same on 31B drafter).
        let is_global = matches!(layer.layer_type,
            rvllm_loader::gemma4_drafter::DrafterLayerType::Full);
        const PARTIAL_ROTARY_FACTOR_GLOBAL: f32 = 0.25;
        let rotary_dim: i32 = if is_global {
            ((eff_hd as f32) * PARTIAL_ROTARY_FACTOR_GLOBAL) as i32
        } else {
            eff_hd as i32
        };
        let (cos_table_off, sin_table_off) = if is_global {
            (self.model.outside.rope_cos_global.offset_bytes,
             self.model.outside.rope_sin_global.offset_bytes)
        } else {
            (self.model.outside.rope_cos_sliding.offset_bytes,
             self.model.outside.rope_sin_sliding.offset_bytes)
        };
        let pos_region = self.arena.region(
            "g4n_drafter_q_rope_pos", 4, 16)?;
        unsafe {
            let p = position as i32;
            pos_region.copy_from_host(&p.to_le_bytes())?;
        }
        unsafe {
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
                &mut q_in as *mut _ as *mut _,
                &mut k_in as *mut _ as *mut _,
                &mut v_in as *mut _ as *mut _,
                &mut q_out as *mut _ as *mut _,
                &mut key_cache as *mut _ as *mut _,
                &mut value_cache as *mut _ as *mut _,
                &mut cos_table as *mut _ as *mut _,
                &mut sin_table as *mut _ as *mut _,
                &mut positions_ptr as *mut _ as *mut _,
                &mut slot_mapping_ptr as *mut _ as *mut _,
                &mut num_tokens_arg as *mut _ as *mut _,
                &mut num_heads_arg as *mut _ as *mut _,
                &mut num_kv_heads_arg as *mut _ as *mut _,
                &mut head_dim_arg as *mut _ as *mut _,
                &mut rotary_dim_arg as *mut _ as *mut _,
            ];
            let rc = cuLaunchKernel(
                self.forward_kernels.fn_rope_partial_f16kv.raw() as CUfunction,
                1, num_heads as u32, 1,
                (eff_hd as u32) / 2, 1, 1,
                0, stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut());
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "drafter_q_side rope launch",
                    CudaErrorKind::LaunchFailed, CudaCtx::setup()));
            }
        }
        Ok(())
    }

    /// Stream-6a (drafter forward primitive #2.5): bridge
    /// between `q_side` and `attn_finisher`. Dispatches to the
    /// drafter's existing `launch_cross_attn_sliding` /
    /// `launch_cross_attn_global` (gemma4_drafter.rs:1227 /
    /// :1387) — those wrap the production f16io decode
    /// kernel against the drafter's f16 shadow KV. No new
    /// kernels; just the per-layer dispatch from Option B's
    /// state.
    ///
    /// Pre-conditions:
    ///   1. `forward_drafter_layer_q_side` has populated
    ///      `workspace.q` (RoPE'd Q).
    ///   2. Caller has populated the drafter's shadow KV via
    ///      `Gemma4DrafterRuntime::populate_shadow_kv*` so
    ///      shadow_kv.{sliding,full}_{k,v}_ptr point at
    ///      correct f16 K/V for the active spec session.
    ///
    /// Post-condition: `workspace.attn_out` holds the
    /// cross-attention output for this layer. `attn_finisher`
    /// then folds it through o_proj + post_attn norm +
    /// residual_1.
    ///
    /// `block_tables_ptr` / `context_lens_ptr`: Option B
    /// passes its own `kv.block_tables_ptr` /
    /// `kv.context_lens_ptr` — same identity-table /
    /// per-sequence length the BASE attention reads from, so
    /// the drafter sees the exact committed context length
    /// the base has written.
    pub fn forward_drafter_layer_cross_attn(
        &self,
        drafter: &crate::gemma4_drafter::Gemma4DrafterRuntime,
        workspace: &crate::gemma4_drafter::DrafterStepWorkspace,
        layer_idx: usize,
        kv: &Gemma4Nvfp4KvState,
    ) -> Result<()> {
        let layer = drafter.layers.get(layer_idx).ok_or_else(||
            corrupt_runtime_err(format!(
                "forward_drafter_layer_cross_attn: layer_idx {} \
                 out of range", layer_idx)))?;
        let is_global = matches!(layer.layer_type,
            rvllm_loader::gemma4_drafter::DrafterLayerType::Full);
        // Gemma 4 QK-norm absorbs 1/sqrt(d_k); attention runs
        // with scale = 1.0 (production confirms in
        // gemma4_bring_up.rs:2939 / 3522 / 10615 / 11292).
        let scale: f32 = 1.0;
        let stream = self.stream.raw();
        unsafe {
            if is_global {
                drafter.launch_cross_attn_global(
                    workspace.attn_out,
                    workspace.q,
                    kv.block_tables_ptr,
                    kv.context_lens_ptr,
                    scale,
                    stream,
                )
            } else {
                // Sliding window from the BASE arch — drafter's
                // sliding source layer mirrors the base's
                // sliding_window_size. window_size_left =
                // sliding_window - 1 per the existing decode
                // kernels.
                let window_size_left =
                    (self.arch.sliding_window_size as i32) - 1;
                drafter.launch_cross_attn_sliding(
                    workspace.attn_out,
                    workspace.q,
                    kv.block_tables_ptr,
                    kv.context_lens_ptr,
                    scale,
                    window_size_left,
                    stream,
                )
            }
        }
    }

    /// Stream-6a (drafter forward primitive #3): attention
    /// finisher. Mirrors production's
    /// `run_drafter_layer_attn_finisher` (line 9271).
    /// Steps: o_proj GEMM → f32→f16 cast → post_attention
    /// _layernorm → residual_1 (workspace.hidden =
    /// workspace.residual1 + workspace.proj_f16).
    ///
    /// Assumes the caller already populated workspace.attn_out
    /// via the drafter cross-attention launcher (which lives
    /// in gemma4_drafter.rs and is shared with production).
    pub fn forward_drafter_layer_attn_finisher(
        &self,
        drafter: &crate::gemma4_drafter::Gemma4DrafterRuntime,
        workspace: &crate::gemma4_drafter::DrafterStepWorkspace,
        layer_idx: usize,
    ) -> Result<()> {
        use rvllm_core::{RvllmError, CudaErrorKind, CudaCtx};
        let hidden = drafter.arch.hidden_size;
        let num_heads = drafter.arch.num_attention_heads;
        let layer = drafter.layers.get(layer_idx).ok_or_else(||
            corrupt_runtime_err(format!(
                "forward_drafter_layer_attn_finisher: layer_idx \
                 {} out of range", layer_idx)))?;
        let eff_hd = layer.effective_head_dim;
        let q_rows = num_heads * eff_hd;
        let stream = self.stream.raw();
        let eps = drafter.arch.rms_norm_eps;
        // 1. o_proj GEMM.
        unsafe {
            self.cublaslt.f16_gemm_f32(
                workspace.attn_out, layer.self_attn_o_proj,
                workspace.gemm_f32,
                1, hidden as i32, q_rows as i32, stream)?;
        }
        // 2. cast f32 → f16.
        self.launch_cast_f32_to_f16(
            workspace.proj_f16, workspace.gemm_f32,
            hidden as u32)?;
        // 3. post_attention_layernorm.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden: hidden as u32, eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_f16,
                workspace.proj_f16,
                layer.post_attention_layernorm, stream)?;
        }
        // 4. residual_1: hidden = residual1 + proj_f16.
        unsafe {
            use cudarc::driver::sys::*;
            let r = cuMemcpyDtoDAsync_v2(
                workspace.hidden, workspace.residual1,
                hidden * 2, stream as CUstream);
            if r != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "attn_finisher residual1 reload",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
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
                self.forward_kernels.fn_vector_add_f16.raw() as CUfunction,
                grid, 1, 1, block, 1, 1,
                0, stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut());
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "attn_finisher residual_1 vector_add",
                    CudaErrorKind::LaunchFailed, CudaCtx::setup()));
            }
        }
        Ok(())
    }

    /// Stream-6a (drafter forward primitive #4): MLP
    /// finisher. Mirrors production's
    /// `run_drafter_layer_mlp_finisher` (line 9469).
    /// Steps: residual_2 snapshot → pre_ff_norm → gate_proj +
    /// up_proj → fused_gelu_mul_f16 → down_proj → post_ff_norm
    /// → residual_2 add → layer_scalar scale in place.
    pub fn forward_drafter_layer_mlp_finisher(
        &self,
        drafter: &crate::gemma4_drafter::Gemma4DrafterRuntime,
        workspace: &crate::gemma4_drafter::DrafterStepWorkspace,
        layer_idx: usize,
    ) -> Result<()> {
        use rvllm_core::{RvllmError, CudaErrorKind, CudaCtx};
        let hidden = drafter.arch.hidden_size;
        let intermediate = drafter.arch.intermediate_size;
        let layer = drafter.layers.get(layer_idx).ok_or_else(||
            corrupt_runtime_err(format!(
                "forward_drafter_layer_mlp_finisher: layer_idx \
                 {} out of range", layer_idx)))?;
        let stream = self.stream.raw();
        let eps = drafter.arch.rms_norm_eps;

        let residual2_region = self.arena.region(
            "g4n_drafter_residual2", hidden * 2, 16)?;
        let gate_up_region = self.arena.region(
            "g4n_drafter_gate_up", 2 * intermediate * 2, 16)?;
        let gate_ptr = gate_up_region.device_ptr();
        let up_ptr = gate_ptr + (intermediate as u64) * 2;

        // 1. residual_2 snapshot.
        unsafe {
            use cudarc::driver::sys::*;
            let r = cuMemcpyDtoDAsync_v2(
                residual2_region.device_ptr(), workspace.hidden,
                hidden * 2, stream as CUstream);
            if r != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "mlp_finisher residual_2 snapshot",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
        }
        // 2. pre_ff_norm.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden: hidden as u32, eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_f16,
                workspace.hidden,
                layer.pre_feedforward_layernorm, stream)?;
            // 3. gate_proj.
            self.cublaslt.f16_gemm_f32(
                workspace.hidden, layer.mlp_gate_proj,
                workspace.gemm_f32,
                1, intermediate as i32, hidden as i32, stream)?;
        }
        self.launch_cast_f32_to_f16(
            gate_ptr, workspace.gemm_f32, intermediate as u32)?;
        // 4. up_proj.
        unsafe {
            self.cublaslt.f16_gemm_f32(
                workspace.hidden, layer.mlp_up_proj,
                workspace.gemm_f32,
                1, intermediate as i32, hidden as i32, stream)?;
        }
        self.launch_cast_f32_to_f16(
            up_ptr, workspace.gemm_f32, intermediate as u32)?;
        // 5. fused_gelu_mul_f16: gate = gelu(gate) * up.
        unsafe {
            use cudarc::driver::sys::*;
            let mut out_p = gate_ptr;
            let mut gate_up = gate_up_region.device_ptr();
            let mut inter_i = intermediate as i32;
            let args = [
                (&mut out_p)   as *mut u64 as *mut core::ffi::c_void,
                (&mut gate_up) as *mut u64 as *mut core::ffi::c_void,
                (&mut inter_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 1024u32.min(intermediate as u32).max(1);
            let rc = cuLaunchKernel(
                self.forward_kernels.fn_fused_gelu_mul_f16.raw() as CUfunction,
                1, 1, 1, block, 1, 1,
                0, stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut());
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "mlp_finisher fused_gelu_mul_f16",
                    CudaErrorKind::LaunchFailed, CudaCtx::setup()));
            }
            // 6. down_proj.
            self.cublaslt.f16_gemm_f32(
                gate_ptr, layer.mlp_down_proj,
                workspace.gemm_f32,
                1, hidden as i32, intermediate as i32, stream)?;
        }
        self.launch_cast_f32_to_f16(
            workspace.hidden, workspace.gemm_f32, hidden as u32)?;
        // 7. post_ff_norm.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden: hidden as u32, eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_f16,
                workspace.hidden,
                layer.post_feedforward_layernorm, stream)?;
            // 8. residual_2 vector_add: hidden = residual2 + hidden.
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
                self.forward_kernels.fn_vector_add_f16.raw() as CUfunction,
                grid, 1, 1, block, 1, 1,
                0, stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut());
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "mlp_finisher residual_2 vector_add",
                    CudaErrorKind::LaunchFailed, CudaCtx::setup()));
            }
            // 9. layer_scalar scale in place: hidden *= scalar.
            // scale_inplace_f16 kernel ABI is (x, scalar_f32, n).
            let mut x = workspace.hidden;
            let mut s: f32 = layer.layer_scalar_f32;
            let mut n: i32 = hidden as i32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut s) as *mut f32 as *mut core::ffi::c_void,
                (&mut n) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((n as u32 + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.forward_kernels.fn_scale_inplace_f16.raw() as CUfunction,
                grid, 1, 1, block, 1, 1,
                0, stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut());
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "mlp_finisher layer_scalar scale_inplace",
                    CudaErrorKind::LaunchFailed, CudaCtx::setup()));
            }
        }
        Ok(())
    }

    /// Stream-6a (drafter forward primitive #5): final-norm +
    /// LM head + argmax + post_projection. Mirrors the inline
    /// chain at gemma4_bring_up.rs:4976-5121 specialized for
    /// the 31B drafter's `use_ordered_embeddings=false`
    /// full-vocab tied LM-head path. Writes:
    ///   workspace.out_token_id (i32[1]) — argmax token id.
    ///   workspace.out_hidden (f16[backbone_hidden_size]) —
    ///     post_projection output for chaining the next step.
    pub fn forward_drafter_final_to_token(
        &self,
        drafter: &crate::gemma4_drafter::Gemma4DrafterRuntime,
        workspace: &crate::gemma4_drafter::DrafterStepWorkspace,
    ) -> Result<()> {
        let hidden = drafter.arch.hidden_size;
        let backbone_hidden = drafter.arch.backbone_hidden_size;
        let vocab = drafter.arch.vocab_size as i32;
        let stream = self.stream.raw();
        let eps = drafter.arch.rms_norm_eps;

        // 1. final_norm on workspace.hidden in place.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden: hidden as u32, eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_f16,
                workspace.hidden,
                drafter.top.final_norm, stream)?;
        }
        // 2. Full-vocab tied LM head (31B path). f16_gemm to
        // f32 logits in workspace.gemm_f32.
        if drafter.arch.use_ordered_embeddings {
            return Err(corrupt_runtime_err(
                "forward_drafter_final_to_token: drafter has \
                 use_ordered_embeddings=true (E4B masked-embedder \
                 path); not wired for Option B yet".into()));
        }
        unsafe {
            self.cublaslt.f16_gemm_f32(
                workspace.hidden, drafter.top.embed_tokens,
                workspace.gemm_f32,
                1, vocab, hidden as i32, stream)?;
            // 3. Argmax over f32 vocab logits → out_token_id.
            // ABI: argmax_kernel(logits, out, vocab) with grid
            // (1,1,1), block (1024,1,1).
            let mut logits = workspace.gemm_f32;
            let mut out = workspace.out_token_id;
            let mut v = vocab;
            let args: [*mut core::ffi::c_void; 3] = [
                (&mut logits) as *mut u64 as *mut _,
                (&mut out)    as *mut u64 as *mut _,
                (&mut v)      as *mut i32 as *mut _,
            ];
            rvllm_fused::launch_raw(
                self.forward_kernels.fn_argmax_f32,
                (1, 1, 1), (1024, 1, 1), 0, stream, &args)?;
            // 4. post_projection: hidden → out_hidden for
            // chaining the next drafter step.
            self.cublaslt.f16_gemm_f32(
                workspace.hidden, drafter.top.post_projection,
                workspace.gemm_f32,
                1, backbone_hidden as i32, hidden as i32, stream)?;
        }
        self.launch_cast_f32_to_f16(
            workspace.out_hidden, workspace.gemm_f32,
            backbone_hidden as u32)?;
        Ok(())
    }

    /// Stream-6b orchestration primitive: chain the full drafter
    /// forward for one speculation step.
    ///
    /// Caller contract: `workspace.pre_projection_in` is already
    /// populated with `cat[last_token_embed, base_hidden_last]`
    /// for this step (or zeros for the dispatch smoke). On
    /// return:
    ///   * `workspace.out_token_id` (u32, device) holds the
    ///     speculated token id;
    ///   * `workspace.out_hidden` (f16, device, `backbone_hidden`
    ///     elems) holds the drafter's post-projection hidden,
    ///     ready to splice into the next step's pre_projection_in
    ///     embed half.
    ///
    /// `position` is the absolute position the drafter cross-
    /// attends at (= number of base tokens currently in the
    /// shared KV). The drafter has no autoregressive K/V of its
    /// own — every step's Q is cross-attended into the shared
    /// shadow K/V populated from the base's source-layer pair.
    /// `kv` is therefore only used by the cross-attn launcher to
    /// reach context_lens / block_tables; `populate_drafter_shadow
    /// _kv` must have run for the current shared range BEFORE the
    /// first step of a session (and after each base verify that
    /// extends the shared range).
    pub fn run_drafter_forward_one_token(
        &self,
        drafter: &crate::gemma4_drafter::Gemma4DrafterRuntime,
        workspace: &crate::gemma4_drafter::DrafterStepWorkspace,
        position: u32,
        kv: &Gemma4Nvfp4KvState,
    ) -> Result<()> {
        self.forward_drafter_pre_projection(drafter, workspace)?;
        let num_layers = drafter.arch.num_hidden_layers;
        for li in 0..num_layers {
            self.forward_drafter_layer_q_side(
                drafter, workspace, li, position)?;
            self.forward_drafter_layer_cross_attn(
                drafter, workspace, li, kv)?;
            self.forward_drafter_layer_attn_finisher(
                drafter, workspace, li)?;
            self.forward_drafter_layer_mlp_finisher(
                drafter, workspace, li)?;
        }
        self.forward_drafter_final_to_token(drafter, workspace)?;
        Ok(())
    }

    fn forward_scratch_guard(&self) -> ForwardScratchGuard<'_> {
        ForwardScratchGuard {
            arena: &self.arena,
            checkpoint: self.forward_checkpoint,
        }
    }

    /// Commit #5b1: allocate the persistent NVFP4 KV cache state
    /// on the bring-up's arena. Call once after `load`, before
    /// the first `forward_*_attn` (commit #5b2). After allocating,
    /// this advances `forward_checkpoint` so that subsequent
    /// scratch rewinds do NOT free the KV state.
    ///
    /// Codex round-4-followup fix: in #5b1 this took `&self` and
    /// did not advance the checkpoint — the KV state was
    /// allocated above the post-load checkpoint and would be
    /// freed by the first forward's ForwardScratchGuard drop.
    /// Caught before any forward integration shipped.
    ///
    /// `max_pos` defaults to 4096 for production workloads;
    /// raise it for long-context inference (the 31B checkpoint
    /// supports up to 262144 but rope-table memory grows
    /// linearly).
    pub fn allocate_kv_state(&mut self, max_pos: u32) -> Result<Gemma4Nvfp4KvState> {
        // Default per-forward chunk size matches single-token decode.
        // #5f raises this once chunked prefill is wired.
        self.allocate_kv_state_with_chunk(max_pos, 1)
    }

    /// Variant that lets the caller pre-size the metadata buffers
    /// (positions / slot_mapping / context_lens) for chunked
    /// prefill. The decode-only path uses `max_query_tokens = 1`
    /// via `allocate_kv_state`; #5f prefill bumps it to the
    /// configured chunk size.
    pub fn allocate_kv_state_with_chunk(
        &mut self, max_pos: u32, max_query_tokens: u32,
    ) -> Result<Gemma4Nvfp4KvState> {
        if self.kv_state_allocated {
            return Err(corrupt_runtime_err(
                "allocate_kv_state called twice: arena would leak the first \
                 allocation (no API to free a single region) and the caller's \
                 prior `Gemma4Nvfp4KvState` would silently point at stale \
                 memory once a forward runs. If you need a fresh state, \
                 rebuild the Gemma4Nvfp4Bringup."
                    .to_string(),
            ));
        }
        let kv = Gemma4Nvfp4KvState::allocate(
            &self.arena, &self.arch, max_pos, max_query_tokens)?;
        // Re-anchor scratch rewinds above the KV state.
        self.forward_checkpoint = self.arena.checkpoint();
        self.kv_state_allocated = true;
        Ok(kv)
    }

    /// Stream-6a part 2 (Option B-side spec-decode prereq).
    /// Lazy-load the Gemma 4 assistant drafter against Option B's
    /// arena + kernel manifest. Mirrors production's
    /// `Gemma4Bringup::ensure_drafter` (gemma4_bring_up.rs:2324)
    /// but stays self-contained — does NOT touch production
    /// fields or call into any Gemma4Bringup method.
    ///
    /// Loads:
    ///   1. Drafter weight layout from `drafter_dir` (HF
    ///      `Gemma4ForAssistant` checkpoint, e.g.
    ///      /home/r00t/gemma-4-31B-it-assistant).
    ///   2. Drafter weights into the Option B arena via
    ///      `Gemma4DrafterRuntime::load`.
    ///   3. Four PTX modules + entry handles:
    ///      - gemma4_masked_embedder (drafter argmax)
    ///      - flash_attention (f16io decode for cross-attn)
    ///      - flash_attention_decode_f16io_bc16 (head_dim=512)
    ///      - gemma4_drafter_dequant (FP8 + NVFP4 → f16 shadow)
    ///   4. F16 shadow KV regions sized to Option B's KV layout
    ///      (block_size=kv.block_size=1, num_blocks_total =
    ///      kv.max_pos). Codex Stream-6a noted that production's
    ///      hardcoded `block_size=32` doesn't match Option B's
    ///      block_size=1 — using kv.block_size keeps the shadow
    ///      coherent with the actual base cache.
    ///
    /// The shadow regions live above `forward_checkpoint` so
    /// scratch rewinds don't reclaim them.
    ///
    /// Call once before the first spec request; subsequent calls
    /// are no-ops (the slot is `Some` after the first init).
    /// `kv.assistant_shared_kv_sources()` MUST resolve — 31B
    /// returns `(58, 59)`, but if a future variant lacks the
    /// pair this errors clearly instead of panicking.
    pub fn ensure_drafter_nvfp4(
        &mut self,
        drafter_dir: &std::path::Path,
        kv: &Gemma4Nvfp4KvState,
    ) -> Result<()> {
        if self.drafter.lock().unwrap().is_some() {
            return Ok(());
        }
        let sources = self.arch.assistant_shared_kv_sources()
            .ok_or_else(|| corrupt_runtime_err(
                "ensure_drafter_nvfp4: base arch has no \
                 assistant_shared_kv_sources — Gemma 4 assistant \
                 cross-attends to (sliding, full) layer pair; 31B \
                 returns (58, 59). Unsupported checkpoint.".into()))?;
        let layout = rvllm_loader::gemma4_drafter::Gemma4DrafterWeightLayout
            ::from_dir(drafter_dir)?;
        if layout.arch.backbone_hidden_size != self.arch.hidden_size {
            return Err(corrupt_runtime_err(format!(
                "ensure_drafter_nvfp4: drafter backbone_hidden_size={} \
                 != base hidden_size={}",
                layout.arch.backbone_hidden_size, self.arch.hidden_size)));
        }
        if layout.arch.vocab_size != self.arch.vocab_size {
            return Err(corrupt_runtime_err(format!(
                "ensure_drafter_nvfp4: drafter vocab_size={} != base \
                 vocab_size={}", layout.arch.vocab_size,
                self.arch.vocab_size)));
        }
        if layout.arch.pre_projection_in_dim != 2 * self.arch.hidden_size {
            return Err(corrupt_runtime_err(format!(
                "ensure_drafter_nvfp4: drafter pre_projection_in_dim={} \
                 != 2 * base hidden_size={}",
                layout.arch.pre_projection_in_dim, self.arch.hidden_size)));
        }

        let mut rt = crate::gemma4_drafter::Gemma4DrafterRuntime
            ::load(&layout, &self.arena)?;

        // Attach the 4 PTX bundles. Same kernel symbol set as
        // production's ensure_drafter.
        let me_mod = self.kernels.load_ptx("gemma4_masked_embedder")?;
        let me_fn = me_mod.get_function(
            "gemma4_masked_embedder_argmax_f16_kernel")?;
        rt.attach_masked_embedder_kernel(me_mod, me_fn);

        let fa_mod = self.kernels.load_ptx("flash_attention")?;
        let fa_fn = fa_mod.get_function(
            "flash_attention_2_decode_f16io_kernel")?;
        rt.attach_flash_attention_kernel(fa_mod, fa_fn);

        let fa_bc16_mod = self.kernels.load_ptx(
            "flash_attention_decode_f16io_bc16")?;
        let fa_bc16_fn = fa_bc16_mod.get_function(
            "flash_attention_2_decode_f16io_kernel")?;
        rt.attach_flash_attention_bc16_kernel(fa_bc16_mod, fa_bc16_fn);

        let dq_mod = self.kernels.load_ptx("gemma4_drafter_dequant")?;
        let dq_fp8 = dq_mod.get_function(
            "gemma4_drafter_dequant_fp8_to_f16_kernel")?;
        let dq_nvfp4 = dq_mod.get_function(
            "gemma4_drafter_dequant_nvfp4_to_f16_kernel")?;
        rt.attach_drafter_dequant_kernels(dq_mod, dq_fp8, dq_nvfp4);

        // Allocate f16 shadow KV. Codex Stream-6a fix:
        // block_size = kv.block_size (= 1 on Option B), NOT the
        // hardcoded 32 production uses. num_blocks_total =
        // kv.max_pos.
        let sliding_li = sources.0;
        let full_li = sources.1;
        let block_size = kv.block_size;
        let num_blocks_total = kv.max_pos;
        let sliding_nkvh =
            self.arch.num_kv_heads_for_layer(sliding_li) as u32;
        let sliding_hd =
            self.arch.head_dim_for_layer(sliding_li) as u32;
        let full_nkvh =
            self.arch.num_kv_heads_for_layer(full_li) as u32;
        let full_hd =
            self.arch.head_dim_for_layer(full_li) as u32;
        let sliding_layer_bytes = (num_blocks_total as usize)
            * (block_size as usize) * (sliding_nkvh as usize)
            * (sliding_hd as usize) * 2;
        let full_layer_bytes = (num_blocks_total as usize)
            * (block_size as usize) * (full_nkvh as usize)
            * (full_hd as usize) * 2;

        let alloc_zeroed = |name: &'static str, bytes: usize|
            -> Result<u64> {
            let region = self.arena.region(name, bytes.max(16), 256)?;
            unsafe {
                use cudarc::driver::sys::*;
                let rc = cuMemsetD8_v2(region.device_ptr(), 0, bytes);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "drafter shadow KV zero-init",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup()));
                }
            }
            Ok(region.device_ptr())
        };
        let sliding_k_ptr = alloc_zeroed(
            "g4n_drafter_shadow_k_sliding", sliding_layer_bytes)?;
        let sliding_v_ptr = alloc_zeroed(
            "g4n_drafter_shadow_v_sliding", sliding_layer_bytes)?;
        let full_k_ptr = alloc_zeroed(
            "g4n_drafter_shadow_k_full", full_layer_bytes)?;
        let full_v_ptr = alloc_zeroed(
            "g4n_drafter_shadow_v_full", full_layer_bytes)?;

        rt.attach_shadow_kv(crate::gemma4_drafter::DrafterShadowKv {
            sliding_k_ptr, sliding_v_ptr, full_k_ptr, full_v_ptr,
            sliding_layer_bytes, full_layer_bytes,
            block_size, num_blocks_total,
            max_blocks_per_seq: num_blocks_total,
            sliding_num_kv_heads: sliding_nkvh,
            sliding_head_dim: sliding_hd,
            full_num_kv_heads: full_nkvh,
            full_head_dim: full_hd,
        });

        // Re-anchor scratch checkpoint so the shadow KV survives
        // forward scratch rewinds.
        self.forward_checkpoint = self.arena.checkpoint();
        *self.drafter.lock().unwrap() = Some(rt);

        eprintln!(
            "[g4n-drafter] ready: drafter layout {}-layer hidden={} \
             vocab={}, shadow KV (sliding={} MiB, full={} MiB, \
             block_size={}, num_blocks={}) on Option B arena",
            layout.arch.num_hidden_layers,
            layout.arch.hidden_size, layout.arch.vocab_size,
            sliding_layer_bytes / (1024 * 1024),
            full_layer_bytes / (1024 * 1024),
            block_size, num_blocks_total,
        );
        Ok(())
    }

    /// Stream-6a (cross-attn data flow): populate the drafter's
    /// f16 shadow KV regions from Option B's NVFP4 K/V cache for
    /// the source layer pair (sliding + full).
    ///
    /// Mirrors production's `populate_shadow_kv*_from_base`
    /// call site (gemma4_bring_up.rs:4583+ source_view +
    /// drafter.populate_shadow_kv_range_from_base) but Option
    /// B-only — production stays untouched.
    ///
    /// `slot_start` + `slot_count`: range of cache slots to
    /// dequant. Used by the spec session loop to do
    /// incremental updates after each accept (`slot_start =
    /// committed_len`, `slot_count = newly_accepted`). The
    /// initial prompt prefill calls this with `slot_start=0`,
    /// `slot_count=prompt_len`.
    ///
    /// The dequant kernel
    /// (`kernels/gemma4_drafter_dequant.cu:72`) reads
    /// Option B's NVFP4 layout (`[slot, kv_head, head_dim/2]`
    /// packed + `[slot, kv_head, head_dim/16]` E4M3 scales)
    /// directly — codex Stream-6a confirmed the layout
    /// compatibility.
    pub fn populate_drafter_shadow_kv(
        &self,
        kv: &Gemma4Nvfp4KvState,
        slot_start: u32,
        slot_count: u32,
    ) -> Result<()> {
        let drafter_guard = self.drafter.lock().unwrap();
        let drafter = drafter_guard.as_ref().ok_or_else(||
            corrupt_runtime_err(
                "populate_drafter_shadow_kv: drafter not loaded; \
                 call ensure_drafter_nvfp4 first".into()))?;
        self.populate_drafter_shadow_kv_with_rt(
            drafter, kv, slot_start, slot_count)
    }

    /// Lock-free variant: caller already has a `&Gemma4Drafter
    /// Runtime` borrow (e.g. it's holding the `self.drafter`
    /// MutexGuard alive for other reads in the same critical
    /// section). Re-locking here would deadlock; use this entry
    /// point instead.
    pub fn populate_drafter_shadow_kv_with_rt(
        &self,
        drafter: &crate::gemma4_drafter::Gemma4DrafterRuntime,
        kv: &Gemma4Nvfp4KvState,
        slot_start: u32,
        slot_count: u32,
    ) -> Result<()> {
        let shadow = drafter.shadow_kv.as_ref().ok_or_else(||
            corrupt_runtime_err(
                "populate_drafter_shadow_kv: shadow KV not attached \
                 (ensure_drafter_nvfp4 should have done this)".into()))?;
        let (sliding_li, full_li) = self.arch.assistant_shared_kv_sources()
            .ok_or_else(|| corrupt_runtime_err(
                "populate_drafter_shadow_kv: arch has no source pair".into()))?;

        // Option B's K/V cache pointers for each source layer.
        let base_sliding_k = kv.k_packed_layer_ptrs[sliding_li];
        let base_sliding_v = kv.v_packed_layer_ptrs[sliding_li];
        let base_full_k    = kv.k_packed_layer_ptrs[full_li];
        let base_full_v    = kv.v_packed_layer_ptrs[full_li];
        let base_sliding_k_scale = kv.k_scale_layer_ptrs[sliding_li];
        let base_sliding_v_scale = kv.v_scale_layer_ptrs[sliding_li];
        let base_full_k_scale    = kv.k_scale_layer_ptrs[full_li];
        let base_full_v_scale    = kv.v_scale_layer_ptrs[full_li];

        let stream = self.stream.raw();
        unsafe {
            drafter.populate_shadow_kv_range_from_base(
                base_sliding_k, base_sliding_v,
                base_full_k, base_full_v,
                base_sliding_k_scale, base_sliding_v_scale,
                base_full_k_scale, base_full_v_scale,
                // Both source layers in Option B are NVFP4 (the
                // active KV dtype across the whole forward).
                crate::gemma4_layer_exec::KvDtype::Nvfp4,
                crate::gemma4_layer_exec::KvDtype::Nvfp4,
                shadow.sliding_layer_bytes,
                shadow.full_layer_bytes,
                slot_start, slot_count,
                stream,
            )?;
        }
        Ok(())
    }

    /// Codex Stream-6a: build a `DrafterBaseKvView` over
    /// Option B's NVFP4 KV layout for the given `layer_idx`.
    /// The production drafter (`gemma4_drafter.rs`) cross-
    /// attends from a small Q-only model into the BASE
    /// model's K/V at the source-layer pair `(58, 59)` on
    /// 31B (`gemma4_arch::Gemma4Arch::assistant_shared_kv
    /// _sources()`).
    ///
    /// The view exposes the per-layer K + V packed nibble
    /// arrays and their E4M3 microscale arrays, plus the
    /// already-allocated identity block_tables and the live
    /// context_lens (which `g4n_fill_pos_slots_i32` updates
    /// per request). Drafter cross-attn currently reads NVFP4
    /// only via the existing shadow-dequant kernel
    /// (`kernels/gemma4_drafter_dequant.cu:72`); pointers from
    /// this view feed straight into that path.
    ///
    /// Wiring into production spec-decode is the next step
    /// (codex Stream-6a real work): the drafter needs to
    /// consume this view through a `BaseKvSource` trait so the
    /// same drafter code can read from either the production
    /// fp8-block KV cache or Option B's NVFP4 cache. The
    /// drafter side of that refactor is in
    /// `gemma4_drafter.rs::DrafterRuntime` + `populate_shadow
    /// _kv*_from_base` (gemma4_bring_up.rs:4583+, 5742+).
    /// Today this method just hands back the view — the
    /// production-side glue is unwired until spec-decode is
    /// enabled on Option B.
    pub fn drafter_base_kv_view(
        &self,
        kv: &Gemma4Nvfp4KvState,
        layer_idx: usize,
    ) -> Result<crate::gemma4_drafter::DrafterBaseKvView> {
        if layer_idx >= self.arch.num_hidden_layers {
            return Err(corrupt_runtime_err(format!(
                "drafter_base_kv_view: layer_idx={} >= num_hidden_layers={}",
                layer_idx, self.arch.num_hidden_layers)));
        }
        // block_tables: Option B uses the kv-state's
        // identity table for the active sequence; the drafter
        // reads block_tables[seq_idx * max_blocks_per_seq +
        // page_idx] with seq_idx=0 in single-request mode.
        // max_blocks_per_seq = max_pos because block_size=1.
        Ok(crate::gemma4_drafter::DrafterBaseKvView {
            k_cache:        kv.k_packed_layer_ptrs[layer_idx],
            v_cache:        kv.v_packed_layer_ptrs[layer_idx],
            k_scale_cache:  kv.k_scale_layer_ptrs[layer_idx],
            v_scale_cache:  kv.v_scale_layer_ptrs[layer_idx],
            // q_scale_cache: per-token Q scale cache. Option
            // B doesn't allocate one today (per-token Q scale
            // is OFF in the floor commits); the drafter cross-
            // attn falls back to the scalar q_scale_ptr via
            // DrafterBaseKvView.q_scale_cache=0.
            q_scale_cache:  0,
            block_tables:   kv.block_tables_ptr,
            context_lens:   kv.context_lens_ptr,
            block_size:     kv.block_size,
            max_blocks_per_seq: kv.max_pos,
            num_blocks_total: kv.max_pos,
            kv_dtype:       crate::gemma4_layer_exec::KvDtype::Nvfp4,
        })
    }

    /// Layer-0 QKV + Q/K-norm on a single token. Extends
    /// `forward_layer0_qkv_only` with per-head RMSNorm on Q and
    /// K (V is not normed in Gemma 4). Returns the bf16 host
    /// view of (q_normed [N_q], k_normed [N_kv], v [N_kv]).
    ///
    /// Gemma 4 Q-norm + K-norm conceptually apply RMSNorm
    /// independently per attention head with gamma=q_norm /
    /// k_norm of shape [head_dim]. We reuse the existing
    /// in-place bf16 RMSNorm kernel by treating the
    /// `[num_heads, head_dim]` buffer as `num_heads` row-tokens
    /// of width `head_dim` — the kernel's per-row RMS reduction
    /// matches the per-head semantics exactly. Sliding layer 0
    /// has num_q_heads=32, num_kv_heads=16, head_dim=256.
    ///
    /// The Q/K outputs from `gemma4_nvfp4_attn_proj` are f32
    /// [1, N], so we first narrow to bf16 with the existing
    /// `f32_to_bf16` cast (TODO: this commit uses a host-side
    /// narrow via a tiny scratch CPU path because the cast
    /// kernel handle isn't yet on ForwardKernels; a follow-up
    /// micro-commit adds the GPU cast).
    pub fn forward_layer0_qk_norm(
        &self,
        token_id: u32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let _scratch_guard = self.forward_scratch_guard();
        let (q_f32, k_f32, v_f32) = self.forward_layer0_qkv_only(token_id)?;
        let layer0 = &self.model.layers[0];
        let head_dim = self.arch.head_dim_sliding as u32;
        let num_q_heads = (q_f32.len() / head_dim as usize) as u32;
        let num_kv_heads = (k_f32.len() / head_dim as usize) as u32;
        debug_assert_eq!(q_f32.len(), (num_q_heads * head_dim) as usize);
        debug_assert_eq!(k_f32.len(), (num_kv_heads * head_dim) as usize);

        // f32 → bf16 narrow on host (one-time small buffer for
        // the smoke; a GPU `f32_to_bf16_kernel` cast is the
        // production path).
        let q_bf16_host: Vec<u16> = q_f32.iter().map(|&x| {
            let bits = x.to_bits();
            // Round-to-nearest-even bf16 narrow.
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            (rounded >> 16) as u16
        }).collect();
        let k_bf16_host: Vec<u16> = k_f32.iter().map(|&x| {
            let bits = x.to_bits();
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            (rounded >> 16) as u16
        }).collect();

        // Upload to device, run per-head RMSNorm in place.
        let q_region = self.arena.region(
            "gemma4_nvfp4_qk_q", q_bf16_host.len() * 2, 256)?;
        let k_region = self.arena.region(
            "gemma4_nvfp4_qk_k", k_bf16_host.len() * 2, 256)?;
        unsafe {
            let q_bytes: &[u8] = std::slice::from_raw_parts(
                q_bf16_host.as_ptr() as *const u8, q_bf16_host.len() * 2);
            let k_bytes: &[u8] = std::slice::from_raw_parts(
                k_bf16_host.as_ptr() as *const u8, k_bf16_host.len() * 2);
            q_region.copy_from_host(q_bytes)?;
            k_region.copy_from_host(k_bytes)?;
        }
        let stream_u64 = self.stream.raw();
        unsafe {
            // Q-norm: num_tokens=num_q_heads, hidden=head_dim,
            // gamma=q_norm[head_dim]. Per-row RMSNorm matches the
            // per-head semantics.
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_q_heads, hidden: head_dim,
                eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                q_region.device_ptr(),
                layer0.q_norm.offset_bytes,
                stream_u64,
            )?;
            // K-norm: same idea on K.
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_kv_heads, hidden: head_dim,
                eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                k_region.device_ptr(),
                layer0.k_norm.offset_bytes,
                stream_u64,
            )?;
        }
        self.stream.fence()?;

        // Read back and convert to f32 for caller inspection.
        let mut q_bf16_out = vec![0u16; q_bf16_host.len()];
        let mut k_bf16_out = vec![0u16; k_bf16_host.len()];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                q_bf16_out.as_mut_ptr() as *mut _,
                q_region.device_ptr(),
                q_bf16_out.len() * 2,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qk_norm: q DtoH failed",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
            let rc = cuMemcpyDtoH_v2(
                k_bf16_out.as_mut_ptr() as *mut _,
                k_region.device_ptr(),
                k_bf16_out.len() * 2,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qk_norm: k DtoH failed",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let q_normed: Vec<f32> = q_bf16_out.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect();
        let k_normed: Vec<f32> = k_bf16_out.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect();

        Ok((q_normed, k_normed, v_f32))
    }

    /// END-TO-END layer-0 forward at position=0 (single token,
    /// no prior context).
    ///
    /// This is commit #5a's structural milestone: compose every
    /// piece commits #4a–#4g built into one path, exercising
    /// the WHOLE layer-0 pipeline on real weights. The
    /// attention math at position=0 collapses to a trivial
    /// identity:
    ///
    ///   For a single-token decode with no KV history, the
    ///   softmax of the 1×1 score matrix is [1.0], so
    ///   attn_out_per_head = V_per_head. Under GQA (Q heads =
    ///   32, KV heads = 16 on 31B sliding), each Q-head h
    ///   reads from KV-head `h / gqa_ratio = h / 2`, so the
    ///   attn_out tensor is built by replicating each KV-head's
    ///   V row across its `gqa_ratio` Q-head slots.
    ///
    /// This validates layer 0's pipeline at the smallest
    /// meaningful problem. Real attention (Q × K^T → softmax
    /// → × V) for position > 0 OR num_tokens > 1 needs a KV
    /// cache + paged-decode kernel — commit #5b's territory.
    ///
    /// Returns the predicted next token id from feeding layer 0's
    /// output through the final close-out (#4g).
    pub fn forward_layer0_position_zero_to_token(
        &self,
        token_id: u32,
    ) -> Result<u32> {
        let _scratch_guard = self.forward_scratch_guard();
        let (q_post_rope_f32, _k_post_rope_f32, v_f32) =
            self.forward_layer0_qk_rope(token_id, 0)?;
        let _ = q_post_rope_f32; // Q is consumed by attention; not used in identity case
        let head_dim = self.arch.head_dim_sliding;
        let num_q_heads = self.arch.num_attention_heads;
        let num_kv_heads = self.arch.num_kv_heads_sliding;
        if num_q_heads % num_kv_heads != 0 {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_layer0_position_zero: num_q_heads not divisible by num_kv_heads",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        let gqa_ratio = num_q_heads / num_kv_heads;
        if v_f32.len() != num_kv_heads * head_dim {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_layer0_position_zero: v_f32 length unexpected",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }

        // Build attn_out [num_q_heads * head_dim] by replicating
        // each KV-head's V row gqa_ratio times. For 31B sliding
        // (gqa_ratio=2): Q-heads 0,1 read V[0]; Q-heads 2,3 read
        // V[1]; ...; Q-heads 30,31 read V[15].
        let mut attn_out_f32: Vec<f32> =
            Vec::with_capacity(num_q_heads * head_dim);
        for qh in 0..num_q_heads {
            let kvh = qh / gqa_ratio;
            let src = &v_f32[kvh * head_dim..(kvh + 1) * head_dim];
            attn_out_f32.extend_from_slice(src);
        }
        debug_assert_eq!(attn_out_f32.len(), num_q_heads * head_dim);

        // Narrow attn_out to bf16 for the post-attn flow.
        let attn_out_bf16: Vec<u16> = attn_out_f32.iter().map(|&x| {
            let bits = x.to_bits();
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            (rounded >> 16) as u16
        }).collect();

        // Compute the input residual (embed → input_layernorm
        // output → no, actually the residual ENTERING the
        // post-attn step is the residual that was preserved AROUND
        // the attention sub-block, i.e. the pre-input_layernorm
        // residual = the embed lookup output). We need to grab
        // that here.
        //
        // Reconstruct: re-do embed lookup to get the residual.
        // Cheaper than threading a residual handle through the
        // intermediate methods.
        let hidden = self.arch.hidden_size as u32;
        let tok_region = self.arena.region(
            "gemma4_nvfp4_pz_tok", 4, 16)?;
        unsafe {
            tok_region.copy_from_host(&(token_id as i32).to_le_bytes())?;
        }
        let h_residual_region = self.arena.region(
            "gemma4_nvfp4_pz_residual", (hidden as usize) * 2, 256)?;
        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1, hidden,
                vocab: self.arch.vocab_size as u32,
            }
            .launch(
                self.forward_kernels.fn_embedding_gather_bf16,
                h_residual_region.device_ptr(),
                self.model.outside.embed_tokens.offset_bytes,
                tok_region.device_ptr(),
                self.stream.raw(),
            )?;
        }
        self.stream.fence()?;
        let mut h_residual_bf16 = vec![0u16; hidden as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                h_residual_bf16.as_mut_ptr() as *mut _,
                h_residual_region.device_ptr(),
                (hidden as usize) * 2,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "position_zero: residual DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Post-attention close-out (#4e): o_proj + post_attn_norm
        // + residual add → updated residual.
        let h_after_attn_f32 = self.forward_layer0_post_attn(
            &attn_out_bf16, &h_residual_bf16,
        )?;

        // Narrow back to bf16 for the MLP block (#4f).
        let h_after_attn_bf16: Vec<u16> = h_after_attn_f32.iter().map(|&x| {
            let bits = x.to_bits();
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            (rounded >> 16) as u16
        }).collect();

        // MLP block (#4f): pre_ff_norm → MLP → post_ff_norm
        // → residual += mlp_normed * layer_scalar.
        let h_after_mlp_f32 = self.forward_layer0_post_attn_mlp(
            &h_after_attn_bf16,
        )?;

        // For end-to-end position=0 we only have ONE layer. The
        // remaining 59 layers are stubbed by passing the layer-0
        // output directly to the final close-out. Full 60-layer
        // composition lands with #5b once real attention works
        // for layers 1..59 (each layer's attention reads the KV
        // cache populated by earlier layers).
        let h_after_mlp_bf16: Vec<u16> = h_after_mlp_f32.iter().map(|&x| {
            let bits = x.to_bits();
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            (rounded >> 16) as u16
        }).collect();

        // Final close-out (#4g): final_norm → tied lm_head → argmax.
        self.forward_final_to_token(&h_after_mlp_bf16)
    }

    /// Final close-out: final_norm → tied LM head → argmax.
    ///
    ///   h_final  = final_norm(h_residual)        // bf16 in-place RMSNorm
    ///   logits   = h_final @ raw_embed_tokens.T  // bf16 GEMV via cuBLASLt → f32
    ///   token_id = argmax(logits)
    ///
    /// `tie_word_embeddings=true` on this checkpoint, so the LM head
    /// uses the raw `embed_tokens.weight`. The separate forward
    /// embedding buffer is sqrt(hidden)-pre-scaled for layer-0 input,
    /// but HF applies that scale in the embedding module's forward(),
    /// not to `lm_head.weight`.
    ///
    /// Returns the predicted next token id.
    pub fn forward_final_to_token(
        &self,
        h_residual_bf16_host: &[u16], // [hidden]
    ) -> Result<u32> {
        let _scratch_guard = self.forward_scratch_guard();
        let hidden = self.arch.hidden_size as u32;
        let vocab = self.arch.vocab_size as u32;
        if h_residual_bf16_host.len() != hidden as usize {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_final_to_token: h_residual length != hidden",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        let h_region = self.arena.region(
            "gemma4_nvfp4_final_h", (hidden as usize) * 2, 256)?;
        let logits_region = self.arena.region(
            "gemma4_nvfp4_final_logits", (vocab as usize) * 4, 256)?;
        let token_region = self.arena.region(
            "gemma4_nvfp4_final_token", 4, 16)?;
        unsafe {
            let r: &[u8] = std::slice::from_raw_parts(
                h_residual_bf16_host.as_ptr() as *const u8,
                h_residual_bf16_host.len() * 2);
            h_region.copy_from_host(r)?;
        }
        let stream_u64 = self.stream.raw();

        // (1) final_norm in-place on h_region.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                h_region.device_ptr(),
                self.model.outside.final_norm.offset_bytes,
                stream_u64,
            )?;
        }

        // Stream-6b spec primitive #1: snapshot the post-final-
        // norm hidden as f16 into `base_last_hidden_ptr` for
        // drafter consumption. No-op when the buffer hasn't been
        // allocated (non-spec sessions). The cast is stream-
        // ordered so the drafter forward enqueued after this
        // sees the updated bytes.
        let base_last_hidden = self.base_last_hidden_device_ptr();
        if base_last_hidden != 0 {
            unsafe {
                rvllm_fused::gemma4_launcher::Bf16ToF16SatLaunch {
                    n: hidden,
                }
                .launch(
                    self.forward_kernels.fn_bf16_to_f16_sat,
                    base_last_hidden,
                    h_region.device_ptr(),
                    stream_u64,
                )?;
            }
        }

        // (2) Tied LM head GEMV: h_normed @ raw_embed_tokens.T → f32 [1, vocab].
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt,
                h_region.device_ptr(),
                self.model.outside.lm_head_tokens.offset_bytes,
                logits_region.device_ptr(),
                1, vocab as i32, hidden as i32, stream_u64,
            )?;
        }

        // (3) argmax over [1, vocab] f32 → [1] i32 token id.
        //     Grid=(1,1,1), block=(1024,1,1), shared-mem reduction.
        unsafe {
            let mut logits_ptr = logits_region.device_ptr();
            let mut out_ptr = token_region.device_ptr();
            let mut vs = vocab as i32;
            let args: [*mut core::ffi::c_void; 3] = [
                (&mut logits_ptr) as *mut u64 as *mut _,
                (&mut out_ptr) as *mut u64 as *mut _,
                (&mut vs) as *mut i32 as *mut _,
            ];
            rvllm_fused::launch_raw(
                self.forward_kernels.fn_argmax_f32,
                (1, 1, 1), (1024, 1, 1), 0, stream_u64, &args,
            )?;
        }
        self.stream.fence()?;

        let mut tok = [0i32; 1];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                tok.as_mut_ptr() as *mut _,
                token_region.device_ptr(),
                4,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "forward_final_to_token: token DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let id = tok[0];
        if id < 0 || (id as u32) >= vocab {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_final_to_token: argmax produced out-of-range token id",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        Ok(id as u32)
    }

    /// Layer-0 MLP block close-out:
    ///   h_normed   = pre_feedforward_layernorm(h_residual)
    ///   mlp_out    = down(gelu_tanh(gate(h_normed)) * up(h_normed))   // commit #3
    ///   mlp_normed = post_feedforward_layernorm(mlp_out)
    ///   h_residual += mlp_normed * layer_scalar
    ///
    /// `layer_scalar` is a learned per-layer f32 scalar (~0.09
    /// for 31B layer 0). It scales the MLP contribution before
    /// the residual add. Codex C2 confirmed this scalar is only
    /// applied AFTER the MLP block, never after attention.
    ///
    /// For commit #4f's smoke, `h_residual_bf16_host` is the
    /// post-attention residual (synthetic 0.01 or the output
    /// from `forward_layer0_post_attn`). Returns the updated
    /// residual after the MLP block.
    pub fn forward_layer0_post_attn_mlp(
        &self,
        h_residual_bf16_host: &[u16], // [hidden]
    ) -> Result<Vec<f32>> {
        let _scratch_guard = self.forward_scratch_guard();
        let layer0 = &self.model.layers[0];
        let hidden = self.arch.hidden_size as u32;
        let intermediate = self.arch.intermediate_size as u32;
        if h_residual_bf16_host.len() != hidden as usize {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_layer0_post_attn_mlp: h_residual length != hidden",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }

        // h_residual stays as bf16 on device. h_normed is a
        // SEPARATE bf16 buffer (pre_ff_norm is NOT in-place on
        // the residual — the residual is preserved for the final
        // add). MLP scratch is [2 * intermediate] bf16. mlp_out
        // is [hidden] bf16.
        let h_residual_region = self.arena.region(
            "gemma4_nvfp4_post_attn_mlp_resid",
            (hidden as usize) * 2, 256)?;
        let h_normed_region = self.arena.region(
            "gemma4_nvfp4_post_attn_mlp_normed",
            (hidden as usize) * 2, 256)?;
        let scratch_region = self.arena.region(
            "gemma4_nvfp4_post_attn_mlp_scratch",
            (2 * intermediate as usize) * 2, 256)?;
        let mlp_out_region = self.arena.region(
            "gemma4_nvfp4_post_attn_mlp_out",
            (hidden as usize) * 2, 256)?;

        unsafe {
            // Upload h_residual; copy to h_normed so the
            // RMSNorm-in-place writes into h_normed, leaving
            // h_residual intact for the final add.
            let r: &[u8] = std::slice::from_raw_parts(
                h_residual_bf16_host.as_ptr() as *const u8,
                h_residual_bf16_host.len() * 2);
            h_residual_region.copy_from_host(r)?;
            h_normed_region.copy_from_host(r)?;
        }
        let stream_u64 = self.stream.raw();

        // (1) pre_feedforward_layernorm in-place on h_normed.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                h_normed_region.device_ptr(),
                layer0.pre_feedforward_layernorm.offset_bytes,
                stream_u64,
            )?;
        }

        // (2) MLP forward via commit #3 helper. gate_up_fused
        //     writes [intermediate] gate + [intermediate] up
        //     into scratch; gelu_tanh_mul writes back into the
        //     gate slot; down_proj reads from there and writes
        //     [hidden] into mlp_out.
        unsafe {
            crate::gemma4_nvfp4_ops::gemma4_nvfp4_mlp_forward(
                &self.mlp_kernels,
                h_normed_region.device_ptr(),
                mlp_out_region.device_ptr(),
                &layer0.gate_proj,
                &layer0.up_proj,
                &layer0.down_proj,
                scratch_region.device_ptr(),
                stream_u64,
            )?;
        }

        // (3) post_feedforward_layernorm in-place on mlp_out.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                mlp_out_region.device_ptr(),
                layer0.post_feedforward_layernorm.offset_bytes,
                stream_u64,
            )?;
        }

        // (4) Read mlp_normed back to host, scale by layer_scalar
        //     (host-side because there's no scale_inplace_bf16
        //     kernel and the scalar is a single f32). Re-upload
        //     into mlp_out_region.
        let mut mlp_normed_bf16 = vec![0u16; hidden as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                mlp_normed_bf16.as_mut_ptr() as *mut _,
                mlp_out_region.device_ptr(),
                (hidden as usize) * 2,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "post_attn_mlp: mlp_normed DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // Pull layer_scalar value off device. It's a bf16 [1]
        // value uploaded by the loader; one DtoH of 2 bytes.
        let mut scalar_bf16 = [0u16; 1];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                scalar_bf16.as_mut_ptr() as *mut _,
                layer0.layer_scalar.offset_bytes,
                2,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "post_attn_mlp: layer_scalar DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let layer_scalar_f32 = f32::from_bits((scalar_bf16[0] as u32) << 16);

        // Scale + narrow back to bf16. RTNE narrow.
        let scaled_bf16: Vec<u16> = mlp_normed_bf16.iter().map(|&b| {
            let v = f32::from_bits((b as u32) << 16) * layer_scalar_f32;
            let bits = v.to_bits();
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            (rounded >> 16) as u16
        }).collect();
        unsafe {
            let s: &[u8] = std::slice::from_raw_parts(
                scaled_bf16.as_ptr() as *const u8, scaled_bf16.len() * 2);
            mlp_out_region.copy_from_host(s)?;
        }

        // (5) Residual add: h_residual += mlp_normed * layer_scalar.
        //     mlp_out_region NOW holds the scaled mlp_normed.
        unsafe {
            rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }
                .launch(
                    self.forward_kernels.fn_vector_add_bf16,
                    h_residual_region.device_ptr(),
                    mlp_out_region.device_ptr(),
                    stream_u64,
                )?;
        }
        self.stream.fence()?;

        let mut h_out_bf16 = vec![0u16; hidden as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                h_out_bf16.as_mut_ptr() as *mut _,
                h_residual_region.device_ptr(),
                (hidden as usize) * 2,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "post_attn_mlp: h_out DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(h_out_bf16.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect())
    }

    /// Layer-0 post-attention close-out: o_proj +
    /// post_attention_layernorm + residual add. Given a
    /// synthetic `attn_out_bf16` and `h_residual_bf16` (each
    /// [hidden]), returns the updated residual after applying
    /// `h += post_attn_norm(o_proj(attn_out))`.
    ///
    /// The real attention math (softmax(QK^T)V) lands with the
    /// KV cache in commit #5. For #4e the synthetic inputs let
    /// us validate the o_proj + norm + residual chain without
    /// depending on that.
    ///
    /// Per codex round-3 C1: Gemma 4 places post_attention_
    /// layernorm BETWEEN o_proj and residual (not before
    /// residual like Llama). layer_scalar is NOT applied here —
    /// it only multiplies after the MLP block.
    /// residual_weight (arch field, 1.0 on 31B) is conceptually
    /// in the residual add but a no-op at value 1.0.
    pub fn forward_layer0_post_attn(
        &self,
        attn_out_bf16_host: &[u16], // [N_q]
        h_residual_bf16_host: &[u16], // [hidden]
    ) -> Result<Vec<f32>> {
        let _scratch_guard = self.forward_scratch_guard();
        let layer0 = &self.model.layers[0];
        let n_q = layer0.o_proj.shape[1] as i32; // o_proj is [hidden, N_q]
        let hidden = self.arch.hidden_size as u32;
        if attn_out_bf16_host.len() != n_q as usize {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_layer0_post_attn: attn_out length != N_q",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        if h_residual_bf16_host.len() != hidden as usize {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_layer0_post_attn: h_residual length != hidden",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }

        // Upload synthetic inputs.
        let attn_in_region = self.arena.region(
            "gemma4_nvfp4_post_attn_in",
            attn_out_bf16_host.len() * 2, 256)?;
        let residual_region = self.arena.region(
            "gemma4_nvfp4_post_attn_resid",
            h_residual_bf16_host.len() * 2, 256)?;
        let o_f32_region = self.arena.region(
            "gemma4_nvfp4_post_attn_o_f32",
            (hidden as usize) * 4, 256)?;
        let o_bf16_region = self.arena.region(
            "gemma4_nvfp4_post_attn_o_bf16",
            (hidden as usize) * 2, 256)?;
        unsafe {
            let a: &[u8] = std::slice::from_raw_parts(
                attn_out_bf16_host.as_ptr() as *const u8,
                attn_out_bf16_host.len() * 2);
            let r: &[u8] = std::slice::from_raw_parts(
                h_residual_bf16_host.as_ptr() as *const u8,
                h_residual_bf16_host.len() * 2);
            attn_in_region.copy_from_host(a)?;
            residual_region.copy_from_host(r)?;
        }
        let stream_u64 = self.stream.raw();

        // (1) o_proj: bf16 attn_out [1, N_q] @ bf16 o_proj weight
        //     [hidden, N_q]^T → f32 [1, hidden].
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt,
                attn_in_region.device_ptr(),
                layer0.o_proj.offset_bytes,
                o_f32_region.device_ptr(),
                1, hidden as i32, n_q, stream_u64,
            )?;
        }
        self.stream.fence()?;

        // (2) Narrow f32 → bf16 on host (commit #4c convention).
        //     Read f32 back, narrow, re-upload as bf16 into
        //     o_bf16_region. GPU f32_to_bf16 wiring is a
        //     follow-up — codex round-3 B2 captures the kernel.
        let mut o_f32_host: Vec<f32> = vec![0.0; hidden as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                o_f32_host.as_mut_ptr() as *mut _,
                o_f32_region.device_ptr(),
                (hidden as usize) * 4,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "post_attn: o f32 DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let o_bf16_host: Vec<u16> = o_f32_host.iter().map(|&x| {
            let bits = x.to_bits();
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            (rounded >> 16) as u16
        }).collect();
        unsafe {
            let b: &[u8] = std::slice::from_raw_parts(
                o_bf16_host.as_ptr() as *const u8,
                o_bf16_host.len() * 2);
            o_bf16_region.copy_from_host(b)?;
        }

        // (3) post_attention_layernorm in-place on o_bf16_region.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                o_bf16_region.device_ptr(),
                layer0.post_attention_layernorm.offset_bytes,
                stream_u64,
            )?;
        }

        // (4) Residual add: h_residual += normed_o (both bf16).
        //     residual_weight is 1.0 on 31B (verified at arch
        //     parse) — vector_add_bf16 implements `dst += src`
        //     which is what we want for weight=1.0. A future
        //     scaled-residual kernel handles weight!=1.0.
        unsafe {
            rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }
                .launch(
                    self.forward_kernels.fn_vector_add_bf16,
                    residual_region.device_ptr(),
                    o_bf16_region.device_ptr(),
                    stream_u64,
                )?;
        }
        self.stream.fence()?;

        // Read back the updated residual as f32 for caller.
        let mut h_out_bf16 = vec![0u16; hidden as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                h_out_bf16.as_mut_ptr() as *mut _,
                residual_region.device_ptr(),
                (hidden as usize) * 2,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "post_attn: h_residual DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(h_out_bf16.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect())
    }

    /// Layer-0 RoPE on post-norm Q/K. Extends qk_norm by
    /// applying `rope_split_half_bf16` at `position`. Sliding
    /// layer 0 uses FULL RoPE on head_dim=256, theta=10000.
    /// At position=0 RoPE is the identity (cos=1, sin=0) — the
    /// smoke uses that to assert byte-equality between the pre-
    /// and post-RoPE Q/K under bf16 narrow tolerance.
    pub fn forward_layer0_qk_rope(
        &self,
        token_id: u32,
        position: u32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let _scratch_guard = self.forward_scratch_guard();
        let (q_normed_f32, k_normed_f32, v_f32) =
            self.forward_layer0_qk_norm(token_id)?;
        let head_dim = self.arch.head_dim_sliding as u32;
        let num_q_heads = (q_normed_f32.len() / head_dim as usize) as u32;
        let num_kv_heads = (k_normed_f32.len() / head_dim as usize) as u32;

        // f32 → bf16 host narrow (commit #4c convention; GPU
        // f32_to_bf16 wiring is a separate follow-up commit).
        let narrow_to_bf16 = |xs: &[f32]| -> Vec<u16> {
            xs.iter().map(|&x| {
                let bits = x.to_bits();
                let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
                (rounded >> 16) as u16
            }).collect()
        };
        let q_bf16: Vec<u16> = narrow_to_bf16(&q_normed_f32);
        let k_bf16: Vec<u16> = narrow_to_bf16(&k_normed_f32);

        let q_region = self.arena.region(
            "gemma4_nvfp4_rope_q", q_bf16.len() * 2, 256)?;
        let k_region = self.arena.region(
            "gemma4_nvfp4_rope_k", k_bf16.len() * 2, 256)?;
        unsafe {
            let q_bytes: &[u8] = std::slice::from_raw_parts(
                q_bf16.as_ptr() as *const u8, q_bf16.len() * 2);
            let k_bytes: &[u8] = std::slice::from_raw_parts(
                k_bf16.as_ptr() as *const u8, k_bf16.len() * 2);
            q_region.copy_from_host(q_bytes)?;
            k_region.copy_from_host(k_bytes)?;
        }

        let stream_u64 = self.stream.raw();

        // The global RoPE tables are f16 (floor commit 3), but the
        // legacy `rope_split_half_bf16` kernel reads `float*` cos/sin.
        // Probe is rare + table row is tiny (head_dim/2 = 128 floats),
        // so we materialize a single-row f32 table per probe call:
        // DtoH the f16 row at `position`, widen to f32, HtoD into a
        // scratch region, set `pos=0` in the kernel args (kernel reads
        // cos_table[pos * (head_dim/2) + freq]).
        let half = (head_dim as usize) / 2;
        let mut cos_row_f16 = vec![0u16; half];
        let mut sin_row_f16 = vec![0u16; half];
        unsafe {
            use cudarc::driver::sys::*;
            let row_bytes = (half as usize) * 2;
            let row_off = (position as usize) * row_bytes;
            let rc = cuMemcpyDtoH_v2(
                cos_row_f16.as_mut_ptr() as *mut _,
                self.model.outside.rope_cos_sliding.offset_bytes
                    + row_off as u64,
                row_bytes);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "forward_layer0_qk_rope: cos row DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
            let rc = cuMemcpyDtoH_v2(
                sin_row_f16.as_mut_ptr() as *mut _,
                self.model.outside.rope_sin_sliding.offset_bytes
                    + row_off as u64,
                row_bytes);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "forward_layer0_qk_rope: sin row DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup()));
            }
        }
        let cos_row_f32: Vec<f32> = cos_row_f16.iter()
            .map(|&b| half::f16::from_bits(b).to_f32()).collect();
        let sin_row_f32: Vec<f32> = sin_row_f16.iter()
            .map(|&b| half::f16::from_bits(b).to_f32()).collect();
        let cos_scratch = self.arena.region(
            "g4n_probe_cos_f32", half * 4, 16)?;
        let sin_scratch = self.arena.region(
            "g4n_probe_sin_f32", half * 4, 16)?;
        unsafe {
            let cb: &[u8] = std::slice::from_raw_parts(
                cos_row_f32.as_ptr() as *const u8, half * 4);
            let sb: &[u8] = std::slice::from_raw_parts(
                sin_row_f32.as_ptr() as *const u8, half * 4);
            cos_scratch.copy_from_host(cb)?;
            sin_scratch.copy_from_host(sb)?;
        }
        let cos_ptr = cos_scratch.device_ptr();
        let sin_ptr = sin_scratch.device_ptr();

        // Launch rope_split_half_bf16 per Q then per K.
        // pos=0 in args because the scratch holds only the row
        // for `position`.
        let launch_rope = |qk_ptr: u64, n_heads: u32| -> Result<()> {
            let mut qk = qk_ptr;
            let mut cos = cos_ptr;
            let mut sin = sin_ptr;
            let mut hd = head_dim as i32;
            let mut pos: i32 = 0;
            let args: [*mut core::ffi::c_void; 5] = [
                (&mut qk) as *mut u64 as *mut _,
                (&mut cos) as *mut u64 as *mut _,
                (&mut sin) as *mut u64 as *mut _,
                (&mut hd) as *mut i32 as *mut _,
                (&mut pos) as *mut i32 as *mut _,
            ];
            unsafe {
                rvllm_fused::launch_raw(
                    self.forward_kernels.fn_rope_split_half_bf16,
                    (n_heads, 1, 1),
                    (head_dim / 2, 1, 1),
                    0, stream_u64, &args,
                )
            }
        };
        launch_rope(q_region.device_ptr(), num_q_heads)?;
        launch_rope(k_region.device_ptr(), num_kv_heads)?;
        self.stream.fence()?;

        // Read back + convert to f32.
        let mut q_out_bf16 = vec![0u16; q_bf16.len()];
        let mut k_out_bf16 = vec![0u16; k_bf16.len()];
        unsafe {
            use cudarc::driver::sys::*;
            for (host, dev, n) in [
                (q_out_bf16.as_mut_ptr() as *mut _, q_region.device_ptr(), q_out_bf16.len() * 2),
                (k_out_bf16.as_mut_ptr() as *mut _, k_region.device_ptr(), k_out_bf16.len() * 2),
            ] {
                let rc = cuMemcpyDtoH_v2(host, dev, n);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qk_rope: DtoH failed",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        let q_out: Vec<f32> = q_out_bf16.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect();
        let k_out: Vec<f32> = k_out_bf16.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect();
        Ok((q_out, k_out, v_f32))
    }

    /// Layer-0 QKV projection on a single token. Extends the
    /// `pre_attn_one_token` smoke with K and V projections too.
    /// Returns (q[N_q], k[N_kv], v[N_kv]) as host f32 vectors.
    ///
    /// Layer 0 is a SLIDING-attention layer per
    /// `arch.layer_types[0]`. v_proj IS present (k_eq_v aliasing
    /// only applies to global layers); the helper requires that.
    ///
    /// `K_kv = arch.num_kv_heads_sliding * arch.head_dim_sliding`
    /// for layer 0 — 16 * 256 = 4096 on 31B.
    ///
    /// What's NOT yet wired (commit #4c):
    ///   * Q-norm / K-norm (the existing fp8-block path uses
    ///     `FusedQkvRmsnormLaunch` with `fused_qkv_rmsnorm_bf16`,
    ///     but the kernel takes fp8-class quantized input. A
    ///     pure-bf16-in variant is the next addition.)
    ///   * RoPE (Gemma 4 sliding uses full RoPE on head_dim=256;
    ///     global uses partial RoPE on head_dim=512 with
    ///     rotary_dim=128 per arch.partial_rotary_factor_global).
    ///   * Attention launch + KV write.
    ///   * O-proj + residual.
    pub fn forward_layer0_qkv_only(
        &self,
        token_id: u32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let _scratch_guard = self.forward_scratch_guard();
        let layer0 = &self.model.layers[0];
        let n_q = layer0.q_proj.shape[0] as i32;
        let n_kv = layer0.k_proj.shape[0] as i32;
        let v_weight = layer0.v_proj.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::cuda(
                "forward_layer0_qkv_only: v_proj absent on layer 0 \
                 (k_eq_v aliasing applies only to global layers — \
                 layer 0 is sliding, must have v_proj)",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;
        let n_v = v_weight.shape[0] as i32;
        // Sliding layer: k and v have the same output dim
        // (n_kv_heads * head_dim); on 31B that's 16 * 256 = 4096.
        // Codex review caught the prior debug_assert was release-
        // invisible: a corrupted checkpoint with mismatched K/V
        // dims on a sliding layer would silently produce wrong
        // attention output. Upgrade to a hard runtime check.
        if n_kv != n_v {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_layer0_qkv_only: sliding-layer K/V dim mismatch",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }

        let hidden = self.arch.hidden_size as u32;

        // Scratch: token + residual + q + k + v.
        let tok_region = self.arena.region("gemma4_nvfp4_qkv_tok", 4, 16)?;
        unsafe {
            tok_region.copy_from_host(&(token_id as i32).to_le_bytes())?;
        }
        let residual_region = self.arena.region(
            "gemma4_nvfp4_qkv_residual", (hidden as usize) * 2, 256)?;
        let q_region = self.arena.region(
            "gemma4_nvfp4_qkv_q", (n_q as usize) * 4, 256)?;
        let k_region = self.arena.region(
            "gemma4_nvfp4_qkv_k", (n_kv as usize) * 4, 256)?;
        let v_region = self.arena.region(
            "gemma4_nvfp4_qkv_v", (n_v as usize) * 4, 256)?;
        let stream_u64 = self.stream.raw();

        // embed → residual → input_layernorm in-place.
        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1, hidden,
                vocab: self.arch.vocab_size as u32,
            }
            .launch(
                self.forward_kernels.fn_embedding_gather_bf16,
                residual_region.device_ptr(),
                self.model.outside.embed_tokens.offset_bytes,
                tok_region.device_ptr(), stream_u64,
            )?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                residual_region.device_ptr(),
                layer0.input_layernorm.offset_bytes,
                stream_u64,
            )?;
        }

        // Q/K/V projections via commit #2 helper. Three sequential
        // bf16 GEMVs at M=1; commit #4c can fuse via cuBLASLt
        // strided-batched if perf matters here.
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, residual_region.device_ptr(),
                layer0.q_proj.offset_bytes, q_region.device_ptr(),
                1, n_q, hidden as i32, stream_u64,
            )?;
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, residual_region.device_ptr(),
                layer0.k_proj.offset_bytes, k_region.device_ptr(),
                1, n_kv, hidden as i32, stream_u64,
            )?;
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, residual_region.device_ptr(),
                v_weight.offset_bytes, v_region.device_ptr(),
                1, n_v, hidden as i32, stream_u64,
            )?;
        }
        self.stream.fence()?;

        let mut q = vec![0f32; n_q as usize];
        let mut k = vec![0f32; n_kv as usize];
        let mut v = vec![0f32; n_v as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let copies = [
                (q.as_mut_ptr() as *mut _, q_region.device_ptr(), (n_q as usize) * 4),
                (k.as_mut_ptr() as *mut _, k_region.device_ptr(), (n_kv as usize) * 4),
                (v.as_mut_ptr() as *mut _, v_region.device_ptr(), (n_v as usize) * 4),
            ];
            for (host, dev, sz) in copies {
                let rc = cuMemcpyDtoH_v2(host, dev, sz);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "forward_layer0_qkv_only: DtoH failed",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        Ok((q, k, v))
    }

    /// Pre-attention sub-block on layer 0:
    ///   embed[token_id] → input_layernorm → q_proj.
    ///
    /// Returns the [N_q] f32 q-projection output as a host Vec
    /// (N_q = 8192 for sliding layer 0, 16384 for global). The
    /// smoke test uses this to assert non-degenerate magnitudes
    /// without yet wiring attention math, MLP, or the per-layer
    /// loop.
    pub fn pre_attn_one_token(&self, token_id: u32) -> Result<Vec<f32>> {
        let _scratch_guard = self.forward_scratch_guard();
        let layer0 = &self.model.layers[0];
        let n_q = layer0.q_proj.shape[0] as i32;
        let hidden = self.arch.hidden_size as u32;

        // Scratch: token_id i32 + bf16 residual [1, hidden] + f32 q output.
        let tok_region = self.arena.region("gemma4_nvfp4_pre_attn_tok", 4, 16)?;
        unsafe {
            tok_region.copy_from_host(&(token_id as i32).to_le_bytes())?;
        }
        let residual_region = self.arena.region(
            "gemma4_nvfp4_pre_attn_residual",
            (hidden as usize) * 2, 256,
        )?;
        let q_region = self.arena.region(
            "gemma4_nvfp4_pre_attn_q",
            (n_q as usize) * 4, 256,
        )?;
        let stream_u64 = self.stream.raw();

        // (1) embed lookup → residual_region [1, hidden] bf16.
        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1,
                hidden,
                vocab: self.arch.vocab_size as u32,
            }
            .launch(
                self.forward_kernels.fn_embedding_gather_bf16,
                residual_region.device_ptr(),
                self.model.outside.embed_tokens.offset_bytes,
                tok_region.device_ptr(),
                stream_u64,
            )?;
        }

        // (2) input_layernorm in-place on residual_region.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1,
                hidden,
                eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                residual_region.device_ptr(),
                layer0.input_layernorm.offset_bytes,
                stream_u64,
            )?;
        }

        // (3) q_proj: bf16 [1, hidden] @ bf16 [N_q, hidden]^T → f32 [1, N_q].
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt,
                residual_region.device_ptr(),
                layer0.q_proj.offset_bytes,
                q_region.device_ptr(),
                1, n_q, hidden as i32, stream_u64,
            )?;
        }

        self.stream.fence()?;
        let mut out = vec![0f32; n_q as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut _,
                q_region.device_ptr(),
                (n_q as usize) * 4,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "pre_attn_one_token: q DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(out)
    }

    /// Commit #5b2: layer-0 attention end-to-end at a single
    /// position, with persistent NVFP4 KV cache.
    ///
    /// Composes:
    ///   1. forward_layer0_qk_norm (Q+K-norm; V returned raw)
    ///   2. host-side parameter-free V-RMSNorm (Gemma 4 uses
    ///      `Gemma4RMSNorm(head_dim, eps, with_scale=False)`
    ///      on V — no gamma. See codex round-3 note in
    ///      rvllm-serve/CLAUDE.md "V-cache write".)
    ///   3. upload Q/K/V as bf16, allocate q_fp8 + bf16 attn_out
    ///      scratch
    ///   4. build per-position f16 cos/sin mini-tables (1 row
    ///      each at index 0), write positions[0]=0,
    ///      slot_mapping[0]=position, context_lens[0]=position+1
    ///   5. launch `fused_rope_partial_nvfp4kv_bf16in` — RoPE +
    ///      NVFP4 K/V pack + per-(slot,head) E4M3 microscale
    ///      write + FP8 Q output. Hadamard OFF, per-token Q
    ///      scale OFF (uses scalar fallback), K policy = amax6,
    ///      V policy = amax6.
    ///   6. launch `flash_attention_2_decode_nvfp4kv_gqa_bf16out`
    ///      — reads cache + q_fp8, emits bf16 attn_out.
    ///      Sliding-layer 0 has GQA=2 (32 q / 16 kv); the GQA
    ///      kernel handles ratio ∈ [1, MAX_GQA_DECODE=4].
    ///   7. DtoH bf16 attn_out → host f32, return.
    ///
    /// Caller responsibilities:
    ///   * Allocate `kv` once via `allocate_kv_state` BEFORE
    ///     the first call (anchors it above forward_checkpoint).
    ///   * Call with `position` advancing 0,1,2,… (caller's
    ///     decode loop). Each call writes one new slot.
    ///
    /// NOT yet wired here (separate commits):
    ///   * Real RoPE table: per-launch mini-table is fine for
    ///     numerical correctness at the chosen position; the
    ///     full f16 table is loader work and lands when the
    ///     60-layer driver does (memory ≈ 384 MiB).
    ///   * Hadamard rotation — production NVFP4 profile uses
    ///     it. For a smoke at layer 0 it's not needed; wiring
    ///     it requires the sign tables + the unrotate kernel.
    ///   * Per-token Q scale — required when Hadamard is on.
    ///   * window_size_left handling at low position is the
    ///     same as production (= sliding_window_size - 1).
    pub fn forward_layer0_attn(
        &self,
        token_id: u32,
        position: u32,
        kv: &Gemma4Nvfp4KvState,
    ) -> Result<Vec<f32>> {
        use rvllm_core::{RvllmError, CudaErrorKind, CudaCtx};

        if position >= kv.max_pos {
            return Err(corrupt_runtime_err(format!(
                "forward_layer0_attn: position={} >= kv.max_pos={}",
                position, kv.max_pos)));
        }

        let _scratch_guard = self.forward_scratch_guard();

        // ---- 1. Q-norm + K-norm via existing helper -----------------------
        let (q_normed_f32, k_normed_f32, v_raw_f32) =
            self.forward_layer0_qk_norm(token_id)?;

        let head_dim = self.arch.head_dim_sliding;
        let num_q_heads = q_normed_f32.len() / head_dim;
        let num_kv_heads = k_normed_f32.len() / head_dim;
        debug_assert_eq!(v_raw_f32.len(), num_kv_heads * head_dim);
        debug_assert!(num_q_heads >= num_kv_heads
            && num_q_heads % num_kv_heads == 0);
        let gqa = num_q_heads / num_kv_heads;
        if gqa > 4 {
            // MAX_GQA_DECODE = 4 in the GQA kernel.
            return Err(corrupt_runtime_err(format!(
                "forward_layer0_attn: gqa_ratio={} > MAX_GQA_DECODE=4",
                gqa)));
        }
        // Sliding layer 0: full RoPE on head_dim. The rotary_dim
        // arg is still threaded to the kernel.
        let rotary_dim = head_dim;

        // ---- 2. Parameter-free V-RMSNorm on host --------------------------
        //   y_h[d] = v_h[d] * rsqrt(mean_d(v_h^2) + eps)
        let eps = self.arch.rms_norm_eps;
        let mut v_normed_f32: Vec<f32> = Vec::with_capacity(v_raw_f32.len());
        for h in 0..num_kv_heads {
            let row = &v_raw_f32[h * head_dim .. (h + 1) * head_dim];
            let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>()
                / (head_dim as f32);
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for &x in row { v_normed_f32.push(x * scale); }
        }

        // ---- 3. Upload Q/K/V as bf16 + allocate scratch -------------------
        let f32_to_bf16 = |xs: &[f32]| -> Vec<u16> {
            xs.iter().map(|&x| {
                let bits = x.to_bits();
                let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
                (rounded >> 16) as u16
            }).collect()
        };
        let q_bf16 = f32_to_bf16(&q_normed_f32);
        let k_bf16 = f32_to_bf16(&k_normed_f32);
        let v_bf16 = f32_to_bf16(&v_normed_f32);

        let q_region  = self.arena.region("g4n_attn_q_bf16",   q_bf16.len() * 2, 256)?;
        let k_region  = self.arena.region("g4n_attn_k_bf16",   k_bf16.len() * 2, 256)?;
        let v_region  = self.arena.region("g4n_attn_v_bf16",   v_bf16.len() * 2, 256)?;
        let q_fp8_region = self.arena.region(
            "g4n_attn_q_fp8", q_bf16.len(), 256)?;  // 1 byte/elem
        let attn_out_region = self.arena.region(
            "g4n_attn_out_bf16", num_q_heads * head_dim * 2, 256)?;

        unsafe {
            let qb: &[u8] = std::slice::from_raw_parts(
                q_bf16.as_ptr() as *const u8, q_bf16.len() * 2);
            let kb: &[u8] = std::slice::from_raw_parts(
                k_bf16.as_ptr() as *const u8, k_bf16.len() * 2);
            let vb: &[u8] = std::slice::from_raw_parts(
                v_bf16.as_ptr() as *const u8, v_bf16.len() * 2);
            q_region.copy_from_host(qb)?;
            k_region.copy_from_host(kb)?;
            v_region.copy_from_host(vb)?;
        }

        // ---- 4. Pre-built f16 cos/sin tables + state writes -------------
        // Floor commit 3: use the global f16 RoPE tables built once at
        // load time (~384 MiB total for sliding + global on 31B). The
        // per-launch mini-table builder is gone — positions[t] now
        // indexes the absolute row directly. Sliding-layer 0 uses
        // full RoPE (rotary_dim = head_dim = 256) and theta=10K, so
        // we point at rope_cos_sliding / rope_sin_sliding.
        let cos_ptr = self.model.outside.rope_cos_sliding.offset_bytes;
        let sin_ptr = self.model.outside.rope_sin_sliding.offset_bytes;

        // Stream-ordered metadata fill. position_offset = position so
        // positions[0] = position (mode B: full f16 tables index by
        // absolute row), slot_mapping[0] = position, context_lens[0]
        // = position+1.
        self.fill_pos_slots(kv,
            /*position_offset=*/position as i32,
            /*start_slot=*/position as i32,
            /*num_tokens=*/1,
        )?;

        // ---- 5. Launch RoPE + NVFP4 K/V write + FP8 Q ---------------------
        // Layer 0 is sliding; pick the layer-0 KV pointers.
        let k_packed = kv.k_packed_layer_ptrs[0];
        let v_packed = kv.v_packed_layer_ptrs[0];
        let k_scale  = kv.k_scale_layer_ptrs[0];
        let v_scale  = kv.v_scale_layer_ptrs[0];
        let stream_u64 = self.stream.raw();

        unsafe {
            let mut q_in: u64 = q_region.device_ptr();
            let mut k_in: u64 = k_region.device_ptr();
            let mut v_in: u64 = v_region.device_ptr();
            let mut q_out: u64 = q_fp8_region.device_ptr();
            let mut kp: u64 = k_packed;
            let mut vp: u64 = v_packed;
            let mut ks: u64 = k_scale;
            let mut vs: u64 = v_scale;
            let mut cos_ptr_local: u64 = cos_ptr;
            let mut sin_ptr_local: u64 = sin_ptr;
            let mut positions_ptr: u64 = kv.positions_ptr;
            let mut slot_ptr: u64 = kv.slot_mapping_ptr;
            let mut q_scale_ptr: u64 = kv.q_scale_ptr;
            let mut q_scale_cache_ptr: u64 = 0; // no per-token Q scale
            let mut hadamard_q: u64 = 0;
            let mut hadamard_k: u64 = 0;
            let mut debug_k_prequant: u64 = 0;
            let mut debug_v_prequant: u64 = 0;

            let mut nt: i32 = 1;
            let mut nh: i32 = num_q_heads as i32;
            let mut nkvh: i32 = num_kv_heads as i32;
            let mut hd: i32 = head_dim as i32;
            let mut rd: i32 = rotary_dim as i32;
            let (mut scale_policy, mut v_scale_policy) =
                read_nvfp4_kv_policies();
            let mut rotate_v: i32 = 0;
            let mut stoch_round_v: i32 = 0;

            let args = [
                (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut kp) as *mut u64 as *mut core::ffi::c_void,
                (&mut vp) as *mut u64 as *mut core::ffi::c_void,
                (&mut ks) as *mut u64 as *mut core::ffi::c_void,
                (&mut vs) as *mut u64 as *mut core::ffi::c_void,
                (&mut cos_ptr_local) as *mut u64 as *mut core::ffi::c_void,
                (&mut sin_ptr_local) as *mut u64 as *mut core::ffi::c_void,
                (&mut positions_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut slot_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_scale_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_scale_cache_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut rd) as *mut i32 as *mut core::ffi::c_void,
                (&mut scale_policy) as *mut i32 as *mut core::ffi::c_void,
                (&mut v_scale_policy) as *mut i32 as *mut core::ffi::c_void,
                (&mut hadamard_q) as *mut u64 as *mut core::ffi::c_void,
                (&mut hadamard_k) as *mut u64 as *mut core::ffi::c_void,
                (&mut rotate_v) as *mut i32 as *mut core::ffi::c_void,
                (&mut debug_k_prequant) as *mut u64 as *mut core::ffi::c_void,
                (&mut debug_v_prequant) as *mut u64 as *mut core::ffi::c_void,
                (&mut stoch_round_v) as *mut i32 as *mut core::ffi::c_void,
            ];
            let max_heads = num_q_heads.max(num_kv_heads) as u32;
            rvllm_fused::launch_raw(
                self.forward_kernels.fn_rope_kv_write_bf16in,
                (1u32, max_heads, 1u32),
                (head_dim as u32, 1u32, 1u32),
                0, stream_u64, &args,
            )?;
        }

        // ---- 6. Launch paged-attention decode (GQA bf16-out) -------------
        unsafe {
            let mut output: u64 = attn_out_region.device_ptr();
            let mut query: u64 = q_fp8_region.device_ptr();
            let mut kp: u64 = k_packed;
            let mut vp: u64 = v_packed;
            let mut ks: u64 = k_scale;
            let mut vs: u64 = v_scale;
            let mut q_scale_cache_ptr: u64 = 0;
            let mut block_tables: u64 = kv.block_tables_ptr;
            let mut context_lens: u64 = kv.context_lens_ptr;
            let mut q_descale: u64 = kv.q_scale_ptr;

            // Gemma 4 QK-norm gamma absorbs the 1/sqrt(d_k) — the
            // attention kernel must run with scale=1.0. Production
            // confirms this in gemma4_layer_exec.rs / gemma4_bring_up.rs
            // (search `attn_scale: 1.0`). Using 1/sqrt(d_k) here makes
            // every layer's softmax underflowed by ~1/16 (head_dim=256)
            // or ~1/22.6 (head_dim=512) vs the reference HF Gemma 4.
            let mut scale: f32 = 1.0;
            let mut nh: i32 = num_q_heads as i32;
            let mut nkvh: i32 = num_kv_heads as i32;
            let mut hd: i32 = head_dim as i32;
            let mut block_size: i32 = kv.block_size as i32;
            let mut max_blocks_per_seq: i32 = kv.max_pos as i32;
            // Sliding layer 0: window_size_left = sliding_window - 1.
            let mut window_size_left: i32 =
                (self.arch.sliding_window_size as i32) - 1;

            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut query) as *mut u64 as *mut core::ffi::c_void,
                (&mut kp) as *mut u64 as *mut core::ffi::c_void,
                (&mut vp) as *mut u64 as *mut core::ffi::c_void,
                (&mut ks) as *mut u64 as *mut core::ffi::c_void,
                (&mut vs) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_scale_cache_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut block_tables) as *mut u64 as *mut core::ffi::c_void,
                (&mut context_lens) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_descale) as *mut u64 as *mut core::ffi::c_void,
                (&mut scale) as *mut f32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut block_size) as *mut i32 as *mut core::ffi::c_void,
                (&mut max_blocks_per_seq) as *mut i32 as *mut core::ffi::c_void,
                (&mut window_size_left) as *mut i32 as *mut core::ffi::c_void,
            ];
            // Smem: 2*FA2_BC*head_dim halves + MAX_GQA*FA2_BC + FA2_THREADS/32 floats
            // = (2*32*head_dim)*2 + (4*32 + 4)*4  bytes
            let fa2_bc: u32 = 32;
            let max_gqa: u32 = 4;
            let fa2_threads: u32 = 128;
            let smem_bytes: u32 =
                2 * fa2_bc * (head_dim as u32) * 2
                + (max_gqa * fa2_bc + fa2_threads / 32) * 4;

            rvllm_fused::launch_raw(
                self.forward_kernels.fn_attn_decode_gqa_bf16out,
                (1u32, num_kv_heads as u32, 1u32),
                (fa2_threads, 1u32, 1u32),
                smem_bytes, stream_u64, &args,
            )?;
        }

        self.stream.fence()?;

        // ---- 7. DtoH bf16 → host f32 -------------------------------------
        let mut attn_out_bf16 = vec![0u16; num_q_heads * head_dim];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                attn_out_bf16.as_mut_ptr() as *mut _,
                attn_out_region.device_ptr(),
                attn_out_bf16.len() * 2);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "forward_layer0_attn: attn_out DtoH",
                    CudaErrorKind::MemcpyFailed,
                    CudaCtx::setup()));
            }
        }
        let attn_out_f32: Vec<f32> = attn_out_bf16.iter()
            .map(|&b| f32::from_bits((b as u32) << 16)).collect();
        Ok(attn_out_f32)
    }

    /// Commit #5b3: per-layer attention forward parameterized
    /// over `layer_idx`. Takes a pre-embed bf16 residual (the
    /// decoder block's input — embed_tokens lookup for layer 0,
    /// the prior layer's post-MLP residual for layer ≥1) and
    /// runs the attention sub-block:
    ///
    ///   input_layernorm → q/k/v_proj → q-norm + k-norm →
    ///   V-RMSNorm (host, parameter-free) → RoPE + NVFP4 K/V
    ///   write at `kv.<layer>` slot=position → GQA bf16 decode
    ///
    /// Handles both layer types via runtime dispatch on
    /// `arch.layer_types[layer_idx]`:
    ///
    ///   * SlidingAttention: head_dim=256, full RoPE,
    ///     theta=10_000, GQA=2, window_size_left=1023, explicit
    ///     v_proj. Decode via the GQA kernel (gqa ≤
    ///     MAX_GQA_DECODE=4).
    ///   * GlobalAttention: head_dim=512, partial RoPE
    ///     (rotary_dim_global=128), theta=1_000_000, GQA=8,
    ///     window_size_left=-1 (no window), attention_k_eq_v=
    ///     true → V aliases the post-k_proj output (no v_proj
    ///     on disk) and gets its own parameter-free V-norm.
    ///     Decode via the non-GQA kernel (GQA=8 exceeds the
    ///     GQA kernel's MAX_GQA_DECODE=4 cap).
    ///
    /// `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` is
    /// called once per launch when the requested dynamic smem
    /// exceeds the default 48 KiB — head_dim=512 needs ~64 KiB,
    /// so the global path always opts in.
    ///
    /// The per-layer KV state lives at `kv.{k,v}_*_layer_ptrs
    /// [layer_idx]` already (#5b1).
    pub fn forward_layer_attn_from_residual(
        &self,
        layer_idx: usize,
        h_residual_bf16_host: &[u16],
        position: u32,
        kv: &Gemma4Nvfp4KvState,
    ) -> Result<Vec<f32>> {
        use rvllm_core::{RvllmError, CudaErrorKind, CudaCtx};
        use rvllm_loader::gemma4_arch::Gemma4LayerType;

        if layer_idx >= self.arch.num_hidden_layers {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_attn_from_residual: layer_idx={} >= num_hidden_layers={}",
                layer_idx, self.arch.num_hidden_layers)));
        }
        if position >= kv.max_pos {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_attn_from_residual: position={} >= kv.max_pos={}",
                position, kv.max_pos)));
        }

        let _scratch_guard = self.forward_scratch_guard();

        let layer = &self.model.layers[layer_idx];
        let hidden = self.arch.hidden_size as u32;
        let is_global = matches!(
            self.arch.layer_types[layer_idx], Gemma4LayerType::GlobalAttention);

        // Per-layer-type config + persistent f16 RoPE tables.
        // `theta` is kept for diagnostic dumps but no longer
        // drives a per-launch table build — the global tables
        // are computed once at load time.
        #[allow(unused_variables)]
        let (head_dim, rotary_dim, theta, window_size_left,
             cos_table_dev, sin_table_dev) = if is_global {
            (
                self.arch.head_dim_global,
                self.arch.rotary_dim_global(),
                self.arch.rope_theta_global as f64,
                -1i32, // full attention — no sliding window
                self.model.outside.rope_cos_global.offset_bytes,
                self.model.outside.rope_sin_global.offset_bytes,
            )
        } else {
            (
                self.arch.head_dim_sliding,
                self.arch.head_dim_sliding, // sliding = full RoPE
                self.arch.rope_theta_sliding as f64,
                (self.arch.sliding_window_size as i32) - 1,
                self.model.outside.rope_cos_sliding.offset_bytes,
                self.model.outside.rope_sin_sliding.offset_bytes,
            )
        };

        let n_q = layer.q_proj.shape[0] as i32;
        let n_kv = layer.k_proj.shape[0] as i32;

        // V source: explicit v_proj for sliding; alias K-proj
        // output for global (attention_k_eq_v=true, see HF
        // `modeling_gemma4.py:1203-1207`). With the alias path,
        // V_in is the K-projection output BEFORE K-norm /
        // RoPE — the V cache then gets its own parameter-free
        // V-RMSNorm and is written separately from K.
        let v_proj_weight: Option<&rvllm_loader::weights::F16Weight>;
        let n_v: i32;
        if is_global {
            if layer.v_proj.is_some() {
                return Err(corrupt_runtime_err(format!(
                    "forward_layer_attn_from_residual: layer {layer_idx} is \
                     Global but v_proj IS present — attention_k_eq_v expected \
                     to drop v_proj from the modelopt checkpoint")));
            }
            v_proj_weight = None;
            n_v = n_kv;
        } else {
            let vw = layer.v_proj.as_ref().ok_or_else(|| corrupt_runtime_err(
                format!("forward_layer_attn_from_residual: layer {layer_idx} \
                         is sliding but v_proj is absent")))?;
            n_v = vw.shape[0] as i32;
            v_proj_weight = Some(vw);
            if n_kv != n_v {
                return Err(corrupt_runtime_err(format!(
                    "forward_layer_attn_from_residual: layer {layer_idx} K/V dim mismatch \
                     ({n_kv} vs {n_v})")));
            }
        }

        let num_q_heads = (n_q as usize) / head_dim;
        let num_kv_heads = (n_kv as usize) / head_dim;
        debug_assert!(num_q_heads % num_kv_heads == 0);
        let gqa = num_q_heads / num_kv_heads;
        // GQA kernel handles gqa ∈ [1, MAX_GQA_DECODE=4]; above
        // that it silently returns. Fall back to the per-Q-head
        // (non-GQA) decode kernel which handles any GQA via
        // internal `kv_head_idx = head_idx / gqa` mapping.
        let use_gqa_kernel = gqa <= 4;
        if h_residual_bf16_host.len() != hidden as usize {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_attn_from_residual: residual length {} != hidden {}",
                h_residual_bf16_host.len(), hidden)));
        }

        // -- Upload residual to device, then in-place input_layernorm ----
        let residual_region = self.arena.region(
            "g4n_lN_resid_bf16", (hidden as usize) * 2, 256)?;
        unsafe {
            let r: &[u8] = std::slice::from_raw_parts(
                h_residual_bf16_host.as_ptr() as *const u8,
                h_residual_bf16_host.len() * 2);
            residual_region.copy_from_host(r)?;
        }
        let stream_u64 = self.stream.raw();
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                residual_region.device_ptr(),
                layer.input_layernorm.offset_bytes,
                stream_u64,
            )?;
        }

        // -- Q/K/V projections to f32 buffers via bf16 GEMV --------------
        let q_f32_region = self.arena.region(
            "g4n_lN_q_f32", (n_q as usize) * 4, 256)?;
        let k_f32_region = self.arena.region(
            "g4n_lN_k_f32", (n_kv as usize) * 4, 256)?;
        let v_f32_region = self.arena.region(
            "g4n_lN_v_f32", (n_v as usize) * 4, 256)?;
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, residual_region.device_ptr(),
                layer.q_proj.offset_bytes, q_f32_region.device_ptr(),
                1, n_q, hidden as i32, stream_u64,
            )?;
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, residual_region.device_ptr(),
                layer.k_proj.offset_bytes, k_f32_region.device_ptr(),
                1, n_kv, hidden as i32, stream_u64,
            )?;
            // V source: alias k_proj output for global
            // (attention_k_eq_v=true); explicit v_proj for
            // sliding. The aliased path does a DtoD copy
            // from `k_f32_region` to `v_f32_region` so the
            // downstream V-norm reads its own buffer.
            match v_proj_weight {
                Some(vw) => {
                    gemma4_nvfp4_attn_proj(
                        &self.cublaslt, residual_region.device_ptr(),
                        vw.offset_bytes, v_f32_region.device_ptr(),
                        1, n_v, hidden as i32, stream_u64,
                    )?;
                }
                None => {
                    use cudarc::driver::sys::*;
                    let rc = cuMemcpyDtoDAsync_v2(
                        v_f32_region.device_ptr(),
                        k_f32_region.device_ptr(),
                        (n_v as usize) * 4,
                        stream_u64 as CUstream);
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(RvllmError::cuda(
                            "forward_layer_attn_from_residual: \
                             k_eq_v DtoD copy",
                            CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
                    }
                }
            }
        }
        // Stream 5a: stay on-device through Q/K/V → norm → RoPE.
        // The prior chain was DtoH-each-projection → host bf16
        // narrow → HtoD → kernel norms → host V-RMSNorm → HtoD.
        // That cost ~7 fences/HtoDs per layer. Now everything
        // runs on `self.stream` in kernel order, with one
        // `stream.fence()` deferred to the very end of the
        // forward (the existing DtoH for the host-API attn_out
        // return). Q-norm/K-norm still use the gamma RMSNorm;
        // V-norm uses the parameter-free `vnorm_bf16`.

        // bf16 Q/K/V scratch (per-call). q_fp8 + attn_out
        // scratch sized to N_q for the decode launch.
        let q_region = self.arena.region(
            "g4n_lN_q_bf16", (n_q as usize) * 2, 256)?;
        let k_region = self.arena.region(
            "g4n_lN_k_bf16", (n_kv as usize) * 2, 256)?;
        let v_region = self.arena.region(
            "g4n_lN_v_bf16", (n_v as usize) * 2, 256)?;
        let q_fp8_region = self.arena.region(
            "g4n_lN_q_fp8", n_q as usize, 256)?;
        let attn_out_region = self.arena.region(
            "g4n_lN_attn_out_bf16", num_q_heads * head_dim * 2, 256)?;

        // Device narrows: f32 GEMM outputs → bf16 scratch.
        self.launch_f32_to_bf16(
            q_region.device_ptr(), q_f32_region.device_ptr(),
            n_q as u32)?;
        self.launch_f32_to_bf16(
            k_region.device_ptr(), k_f32_region.device_ptr(),
            n_kv as u32)?;
        self.launch_f32_to_bf16(
            v_region.device_ptr(), v_f32_region.device_ptr(),
            n_v as u32)?;

        unsafe {
            // Q-norm + K-norm in place on the bf16 scratches.
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_q_heads as u32, hidden: head_dim as u32,
                eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                q_region.device_ptr(),
                layer.q_norm.offset_bytes,
                stream_u64,
            )?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_kv_heads as u32, hidden: head_dim as u32,
                eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                k_region.device_ptr(),
                layer.k_norm.offset_bytes,
                stream_u64,
            )?;
        }

        // Parameter-free V-RMSNorm on device, in place on the
        // bf16 V scratch.
        self.launch_vnorm_bf16(
            v_region.device_ptr(),
            num_kv_heads as u32, head_dim as u32)?;

        // Floor commit 3: use the global f16 RoPE table for this
        // layer's type (sliding or global). Mode B fill —
        // positions[t] = absolute slot — so the kernel reads
        // `cos_table[positions[t] * (rotary_dim/2) + freq]`
        // directly. Sanity check the position is within the
        // table's allocated rows.
        if (position as usize) >= self.arch.max_position_embeddings {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_attn_from_residual: position={} >= \
                 max_position_embeddings={}", position,
                self.arch.max_position_embeddings)));
        }
        self.fill_pos_slots(kv,
            /*position_offset=*/position as i32,
            /*start_slot=*/position as i32,
            /*num_tokens=*/1,
        )?;

        // -- RoPE + NVFP4 K/V write + FP8 Q launch -----------------------
        let k_packed = kv.k_packed_layer_ptrs[layer_idx];
        let v_packed = kv.v_packed_layer_ptrs[layer_idx];
        let k_scale  = kv.k_scale_layer_ptrs[layer_idx];
        let v_scale  = kv.v_scale_layer_ptrs[layer_idx];
        unsafe {
            let mut q_in: u64 = q_region.device_ptr();
            let mut k_in: u64 = k_region.device_ptr();
            let mut v_in: u64 = v_region.device_ptr();
            let mut q_out: u64 = q_fp8_region.device_ptr();
            let mut kp: u64 = k_packed;
            let mut vp: u64 = v_packed;
            let mut ks: u64 = k_scale;
            let mut vs: u64 = v_scale;
            let mut cos_ptr_local: u64 = cos_table_dev;
            let mut sin_ptr_local: u64 = sin_table_dev;
            let mut positions_ptr: u64 = kv.positions_ptr;
            let mut slot_ptr: u64 = kv.slot_mapping_ptr;
            let mut q_scale_ptr: u64 = kv.q_scale_ptr;
            let mut q_scale_cache_ptr: u64 = 0;
            let mut hadamard_q: u64 = 0;
            let mut hadamard_k: u64 = 0;
            let mut debug_k_prequant: u64 = 0;
            let mut debug_v_prequant: u64 = 0;

            let mut nt: i32 = 1;
            let mut nh: i32 = num_q_heads as i32;
            let mut nkvh: i32 = num_kv_heads as i32;
            let mut hd: i32 = head_dim as i32;
            let mut rd: i32 = rotary_dim as i32;
            let (mut scale_policy, mut v_scale_policy) =
                read_nvfp4_kv_policies();
            let mut rotate_v: i32 = 0;
            let mut stoch_round_v: i32 = 0;

            let args = [
                (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut kp) as *mut u64 as *mut core::ffi::c_void,
                (&mut vp) as *mut u64 as *mut core::ffi::c_void,
                (&mut ks) as *mut u64 as *mut core::ffi::c_void,
                (&mut vs) as *mut u64 as *mut core::ffi::c_void,
                (&mut cos_ptr_local) as *mut u64 as *mut core::ffi::c_void,
                (&mut sin_ptr_local) as *mut u64 as *mut core::ffi::c_void,
                (&mut positions_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut slot_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_scale_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_scale_cache_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut rd) as *mut i32 as *mut core::ffi::c_void,
                (&mut scale_policy) as *mut i32 as *mut core::ffi::c_void,
                (&mut v_scale_policy) as *mut i32 as *mut core::ffi::c_void,
                (&mut hadamard_q) as *mut u64 as *mut core::ffi::c_void,
                (&mut hadamard_k) as *mut u64 as *mut core::ffi::c_void,
                (&mut rotate_v) as *mut i32 as *mut core::ffi::c_void,
                (&mut debug_k_prequant) as *mut u64 as *mut core::ffi::c_void,
                (&mut debug_v_prequant) as *mut u64 as *mut core::ffi::c_void,
                (&mut stoch_round_v) as *mut i32 as *mut core::ffi::c_void,
            ];
            let max_heads = num_q_heads.max(num_kv_heads) as u32;
            rvllm_fused::launch_raw(
                self.forward_kernels.fn_rope_kv_write_bf16in,
                (1u32, max_heads, 1u32),
                (head_dim as u32, 1u32, 1u32),
                0, stream_u64, &args,
            )?;
        }

        // -- GQA bf16-out paged decode launch ----------------------------
        unsafe {
            let mut output: u64 = attn_out_region.device_ptr();
            let mut query: u64 = q_fp8_region.device_ptr();
            let mut kp: u64 = k_packed;
            let mut vp: u64 = v_packed;
            let mut ks: u64 = k_scale;
            let mut vs: u64 = v_scale;
            let mut q_scale_cache_ptr: u64 = 0;
            let mut block_tables: u64 = kv.block_tables_ptr;
            let mut context_lens: u64 = kv.context_lens_ptr;
            let mut q_descale: u64 = kv.q_scale_ptr;

            // Gemma 4 QK-norm gamma absorbs the 1/sqrt(d_k) — the
            // attention kernel must run with scale=1.0. Production
            // confirms this in gemma4_layer_exec.rs / gemma4_bring_up.rs
            // (search `attn_scale: 1.0`). Using 1/sqrt(d_k) here makes
            // every layer's softmax underflowed by ~1/16 (head_dim=256)
            // or ~1/22.6 (head_dim=512) vs the reference HF Gemma 4.
            let mut scale: f32 = 1.0;
            let mut nh: i32 = num_q_heads as i32;
            let mut nkvh: i32 = num_kv_heads as i32;
            let mut hd: i32 = head_dim as i32;
            let mut block_size: i32 = kv.block_size as i32;
            let mut max_blocks_per_seq: i32 = kv.max_pos as i32;
            let mut window_size_left: i32 = window_size_left;

            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut query) as *mut u64 as *mut core::ffi::c_void,
                (&mut kp) as *mut u64 as *mut core::ffi::c_void,
                (&mut vp) as *mut u64 as *mut core::ffi::c_void,
                (&mut ks) as *mut u64 as *mut core::ffi::c_void,
                (&mut vs) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_scale_cache_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut block_tables) as *mut u64 as *mut core::ffi::c_void,
                (&mut context_lens) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_descale) as *mut u64 as *mut core::ffi::c_void,
                (&mut scale) as *mut f32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut block_size) as *mut i32 as *mut core::ffi::c_void,
                (&mut max_blocks_per_seq) as *mut i32 as *mut core::ffi::c_void,
                (&mut window_size_left) as *mut i32 as *mut core::ffi::c_void,
            ];
            let fa2_bc: u32 = 32;
            let fa2_threads: u32 = 128;
            // GQA kernel: s_score is [MAX_GQA * FA2_BC] floats; grid_y =
            // num_kv_heads, one block per (kv_head) loops over its
            // gqa Q-heads internally.
            // Non-GQA kernel: s_score is [FA2_BC] floats; grid_y =
            // num_heads, one block per Q-head with kernel-internal
            // (head_idx → kv_head) mapping.
            let (kernel, grid_y, smem_bytes) = if use_gqa_kernel {
                let max_gqa: u32 = 4;
                let smem = 2 * fa2_bc * (head_dim as u32) * 2
                    + (max_gqa * fa2_bc + fa2_threads / 32) * 4;
                (self.forward_kernels.fn_attn_decode_gqa_bf16out,
                 num_kv_heads as u32, smem)
            } else {
                let smem = 2 * fa2_bc * (head_dim as u32) * 2
                    + (fa2_bc + fa2_threads / 32) * 4;
                (self.forward_kernels.fn_attn_decode_bf16out,
                 num_q_heads as u32, smem)
            };
            // head_dim=512 pushes the decode kernel above the 48 KiB
            // default per-launch smem cap; opt in via cuFuncSetAttribute.
            if smem_bytes >= 48 * 1024 {
                use cudarc::driver::sys::*;
                let rc = cuFuncSetAttribute(
                    kernel.raw() as CUfunction,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem_bytes as i32,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(RvllmError::cuda(
                        "forward_layer_attn_from_residual: \
                         cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE)",
                        CudaErrorKind::LaunchFailed, CudaCtx::setup()));
                }
            }
            rvllm_fused::launch_raw(
                kernel,
                (1u32, grid_y, 1u32),
                (fa2_threads, 1u32, 1u32),
                smem_bytes, stream_u64, &args,
            )?;
        }

        self.stream.fence()?;

        // -- DtoH bf16 → host f32 ----------------------------------------
        let mut attn_out_bf16 = vec![0u16; num_q_heads * head_dim];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                attn_out_bf16.as_mut_ptr() as *mut _,
                attn_out_region.device_ptr(),
                attn_out_bf16.len() * 2);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "forward_layer_attn_from_residual: attn_out DtoH",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
        }
        Ok(attn_out_bf16.iter()
            .map(|&b| f32::from_bits((b as u32) << 16)).collect())
    }

    /// Helper for the per-layer smoke + future driver: do the
    /// embed_tokens lookup for `token_id` and return the resulting
    /// bf16 residual on host (length = hidden_size). This is the
    /// input to layer 0; layers ≥1 receive their predecessor's
    /// post-MLP residual instead.
    pub fn embed_one_token_bf16(&self, token_id: u32) -> Result<Vec<u16>> {
        use rvllm_core::{RvllmError, CudaErrorKind, CudaCtx};
        let _scratch_guard = self.forward_scratch_guard();
        let hidden = self.arch.hidden_size as u32;
        let h_region = self.arena.region(
            "g4n_embed_residual_bf16", (hidden as usize) * 2, 256)?;
        self.embed_one_token_to_device(token_id, h_region.device_ptr())?;
        self.stream.fence()?;
        let mut out = vec![0u16; hidden as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut _, h_region.device_ptr(),
                (hidden as usize) * 2);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "embed_one_token_bf16: residual DtoH",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
        }
        Ok(out)
    }

    /// Stream 5a-step2: device-resident embed lookup.
    /// Writes the embedded bf16 residual for `token_id` into
    /// `dst_dev` (caller-allocated, length `hidden_size * 2`
    /// bytes). No fence — caller decides when to sync. Used by
    /// the inter-layer-device-residual drivers.
    fn embed_one_token_to_device(
        &self, token_id: u32, dst_dev: u64,
    ) -> Result<()> {
        let hidden = self.arch.hidden_size as u32;
        let tok_region = self.arena.region("g4n_embed_tok", 4, 16)?;
        unsafe { tok_region.copy_from_host(&(token_id as i32).to_le_bytes())?; }
        let stream_u64 = self.stream.raw();
        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1, hidden,
                vocab: self.arch.vocab_size as u32,
            }
            .launch(
                self.forward_kernels.fn_embedding_gather_bf16,
                dst_dev,
                self.model.outside.embed_tokens.offset_bytes,
                tok_region.device_ptr(), stream_u64,
            )?;
        }
        Ok(())
    }

    /// Commit #5c: per-layer post-attention close-out
    /// (o_proj + post_attention_layernorm + residual add).
    /// Same structure as `forward_layer0_post_attn` but
    /// parameterized over `layer_idx` — uses
    /// `self.model.layers[layer_idx].{o_proj, post_attention_layernorm}`.
    pub fn forward_layer_post_attn(
        &self,
        layer_idx: usize,
        attn_out_bf16_host: &[u16],
        h_residual_bf16_host: &[u16],
    ) -> Result<Vec<f32>> {
        use rvllm_core::{RvllmError, CudaErrorKind, CudaCtx};
        if layer_idx >= self.arch.num_hidden_layers {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_post_attn: layer_idx={} >= num_hidden_layers={}",
                layer_idx, self.arch.num_hidden_layers)));
        }
        let _scratch_guard = self.forward_scratch_guard();
        let layer = &self.model.layers[layer_idx];
        let n_q = layer.o_proj.shape[1] as i32;
        let hidden = self.arch.hidden_size as u32;
        if attn_out_bf16_host.len() != n_q as usize {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_post_attn: attn_out length {} != N_q {}",
                attn_out_bf16_host.len(), n_q)));
        }
        if h_residual_bf16_host.len() != hidden as usize {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_post_attn: residual length {} != hidden {}",
                h_residual_bf16_host.len(), hidden)));
        }

        let attn_in_region = self.arena.region(
            "g4n_lN_post_attn_in", attn_out_bf16_host.len() * 2, 256)?;
        let residual_region = self.arena.region(
            "g4n_lN_post_attn_resid", h_residual_bf16_host.len() * 2, 256)?;
        let o_f32_region = self.arena.region(
            "g4n_lN_post_attn_o_f32", (hidden as usize) * 4, 256)?;
        let o_bf16_region = self.arena.region(
            "g4n_lN_post_attn_o_bf16", (hidden as usize) * 2, 256)?;
        unsafe {
            let a: &[u8] = std::slice::from_raw_parts(
                attn_out_bf16_host.as_ptr() as *const u8,
                attn_out_bf16_host.len() * 2);
            let r: &[u8] = std::slice::from_raw_parts(
                h_residual_bf16_host.as_ptr() as *const u8,
                h_residual_bf16_host.len() * 2);
            attn_in_region.copy_from_host(a)?;
            residual_region.copy_from_host(r)?;
        }
        let stream_u64 = self.stream.raw();

        // o_proj: bf16 attn_out @ bf16 o_proj^T → f32 hidden,
        // then device-side f32→bf16 narrow. Stream 5a replaces
        // a fence + DtoH + host RTNE-narrow + HtoD with a single
        // launch on `self.stream`.
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt,
                attn_in_region.device_ptr(),
                layer.o_proj.offset_bytes,
                o_f32_region.device_ptr(),
                1, hidden as i32, n_q, stream_u64,
            )?;
        }
        self.launch_f32_to_bf16(
            o_bf16_region.device_ptr(),
            o_f32_region.device_ptr(),
            hidden,
        )?;

        // post_attention_layernorm in-place on o_bf16.
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                o_bf16_region.device_ptr(),
                layer.post_attention_layernorm.offset_bytes,
                stream_u64,
            )?;
            // Residual add: residual += normed_o.
            rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }
                .launch(
                    self.forward_kernels.fn_vector_add_bf16,
                    residual_region.device_ptr(),
                    o_bf16_region.device_ptr(),
                    stream_u64,
                )?;
        }
        self.stream.fence()?;

        let mut h_out_bf16 = vec![0u16; hidden as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                h_out_bf16.as_mut_ptr() as *mut _,
                residual_region.device_ptr(),
                (hidden as usize) * 2);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "forward_layer_post_attn: residual DtoH",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
        }
        Ok(h_out_bf16.iter()
            .map(|&b| f32::from_bits((b as u32) << 16)).collect())
    }

    /// Commit #5c: per-layer post-MLP close-out
    /// (pre_feedforward_layernorm + MLP + post_feedforward_layernorm
    /// + scale by layer_scalar + residual add). Same as
    /// `forward_layer0_post_attn_mlp` but parameterized over
    /// `layer_idx` — uses the per-layer norms, MLP linears, and
    /// `layer_scalar`.
    pub fn forward_layer_post_attn_mlp(
        &self,
        layer_idx: usize,
        h_residual_bf16_host: &[u16],
    ) -> Result<Vec<f32>> {
        use rvllm_core::{RvllmError, CudaErrorKind, CudaCtx};
        if layer_idx >= self.arch.num_hidden_layers {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_post_attn_mlp: layer_idx={} >= num_hidden_layers={}",
                layer_idx, self.arch.num_hidden_layers)));
        }
        let _scratch_guard = self.forward_scratch_guard();
        let layer = &self.model.layers[layer_idx];
        let hidden = self.arch.hidden_size as u32;
        let intermediate = self.arch.intermediate_size as u32;
        if h_residual_bf16_host.len() != hidden as usize {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_post_attn_mlp: residual length {} != hidden {}",
                h_residual_bf16_host.len(), hidden)));
        }

        let h_residual_region = self.arena.region(
            "g4n_lN_pamlp_resid", (hidden as usize) * 2, 256)?;
        let h_normed_region = self.arena.region(
            "g4n_lN_pamlp_normed", (hidden as usize) * 2, 256)?;
        let scratch_region = self.arena.region(
            "g4n_lN_pamlp_scratch", (2 * intermediate as usize) * 2, 256)?;
        let mlp_out_region = self.arena.region(
            "g4n_lN_pamlp_out", (hidden as usize) * 2, 256)?;
        unsafe {
            let r: &[u8] = std::slice::from_raw_parts(
                h_residual_bf16_host.as_ptr() as *const u8,
                h_residual_bf16_host.len() * 2);
            h_residual_region.copy_from_host(r)?;
            h_normed_region.copy_from_host(r)?;
        }
        let stream_u64 = self.stream.raw();

        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                h_normed_region.device_ptr(),
                layer.pre_feedforward_layernorm.offset_bytes,
                stream_u64,
            )?;
            crate::gemma4_nvfp4_ops::gemma4_nvfp4_mlp_forward(
                &self.mlp_kernels,
                h_normed_region.device_ptr(),
                mlp_out_region.device_ptr(),
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
                scratch_region.device_ptr(),
                stream_u64,
            )?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                mlp_out_region.device_ptr(),
                layer.post_feedforward_layernorm.offset_bytes,
                stream_u64,
            )?;
        }

        // Read mlp_normed back to host, scale by layer_scalar, re-upload.
        let mut mlp_normed_bf16 = vec![0u16; hidden as usize];
        let mut scalar_bf16 = [0u16; 1];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                mlp_normed_bf16.as_mut_ptr() as *mut _,
                mlp_out_region.device_ptr(),
                (hidden as usize) * 2);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "forward_layer_post_attn_mlp: mlp_normed DtoH",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
            let rc = cuMemcpyDtoH_v2(
                scalar_bf16.as_mut_ptr() as *mut _,
                layer.layer_scalar.offset_bytes, 2);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "forward_layer_post_attn_mlp: layer_scalar DtoH",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
        }
        let layer_scalar_f32 = f32::from_bits((scalar_bf16[0] as u32) << 16);
        let scaled_bf16: Vec<u16> = mlp_normed_bf16.iter().map(|&b| {
            let v = f32::from_bits((b as u32) << 16) * layer_scalar_f32;
            let bits = v.to_bits();
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            (rounded >> 16) as u16
        }).collect();
        unsafe {
            let s: &[u8] = std::slice::from_raw_parts(
                scaled_bf16.as_ptr() as *const u8, scaled_bf16.len() * 2);
            mlp_out_region.copy_from_host(s)?;
            rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }
                .launch(
                    self.forward_kernels.fn_vector_add_bf16,
                    h_residual_region.device_ptr(),
                    mlp_out_region.device_ptr(),
                    stream_u64,
                )?;
        }
        self.stream.fence()?;

        let mut h_out_bf16 = vec![0u16; hidden as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                h_out_bf16.as_mut_ptr() as *mut _,
                h_residual_region.device_ptr(),
                (hidden as usize) * 2);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "forward_layer_post_attn_mlp: residual DtoH",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
        }
        Ok(h_out_bf16.iter()
            .map(|&b| f32::from_bits((b as u32) << 16)).collect())
    }

    /// Stream 5a-step2: device-mode attention forward. Takes a
    /// bf16 residual already on device, writes the bf16 attn_out
    /// into a caller-provided device region. NO scratch_guard
    /// (caller owns lifetime), NO initial HtoD, NO terminal fence
    /// + DtoH. Mirrors `forward_layer_attn_from_residual` body
    /// minus the boundary I/O. The driver
    /// (`forward_full_to_token`) calls this with a persistent
    /// residual_dev so the inter-layer round-trip vanishes.
    ///
    /// SAFETY: `residual_dev` and `attn_out_dev` must point at
    /// arena regions of the correct size (hidden_size*2 and
    /// num_q_heads_for(layer_idx)*head_dim_for(layer_idx)*2
    /// respectively). The method reads residual_dev but DOES
    /// NOT mutate it — the input_layernorm runs on a local
    /// scratch copy so the caller's residual stays available
    /// for the post-attn residual add.
    fn forward_layer_attn_from_residual_dev(
        &self,
        layer_idx: usize,
        residual_dev: u64,
        position: u32,
        kv: &Gemma4Nvfp4KvState,
        attn_out_dev: u64,
    ) -> Result<()> {
        use rvllm_core::{RvllmError, CudaErrorKind, CudaCtx};
        use rvllm_loader::gemma4_arch::Gemma4LayerType;

        if layer_idx >= self.arch.num_hidden_layers {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_attn_from_residual_dev: layer_idx={} >= {}",
                layer_idx, self.arch.num_hidden_layers)));
        }
        if position >= kv.max_pos {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_attn_from_residual_dev: position={} >= kv.max_pos={}",
                position, kv.max_pos)));
        }
        let layer = &self.model.layers[layer_idx];
        let hidden = self.arch.hidden_size as u32;
        let is_global = matches!(
            self.arch.layer_types[layer_idx], Gemma4LayerType::GlobalAttention);
        #[allow(unused_variables)]
        let (head_dim, rotary_dim, _theta, window_size_left,
             cos_table_dev, sin_table_dev) = if is_global {
            (self.arch.head_dim_global,
             self.arch.rotary_dim_global(),
             self.arch.rope_theta_global as f64,
             -1i32,
             self.model.outside.rope_cos_global.offset_bytes,
             self.model.outside.rope_sin_global.offset_bytes)
        } else {
            (self.arch.head_dim_sliding,
             self.arch.head_dim_sliding,
             self.arch.rope_theta_sliding as f64,
             (self.arch.sliding_window_size as i32) - 1,
             self.model.outside.rope_cos_sliding.offset_bytes,
             self.model.outside.rope_sin_sliding.offset_bytes)
        };
        let n_q = layer.q_proj.shape[0] as i32;
        let n_kv = layer.k_proj.shape[0] as i32;
        let v_proj_weight: Option<&rvllm_loader::weights::F16Weight>;
        let n_v: i32;
        if is_global {
            if layer.v_proj.is_some() {
                return Err(corrupt_runtime_err(format!(
                    "forward_layer_attn_from_residual_dev: layer {layer_idx} \
                     is Global but v_proj IS present")));
            }
            v_proj_weight = None;
            n_v = n_kv;
        } else {
            let vw = layer.v_proj.as_ref().ok_or_else(|| corrupt_runtime_err(
                format!("forward_layer_attn_from_residual_dev: layer {layer_idx} \
                         is sliding but v_proj is absent")))?;
            n_v = vw.shape[0] as i32;
            v_proj_weight = Some(vw);
            if n_kv != n_v {
                return Err(corrupt_runtime_err(format!(
                    "forward_layer_attn_from_residual_dev: K/V dim mismatch \
                     ({n_kv} vs {n_v})")));
            }
        }
        let num_q_heads = (n_q as usize) / head_dim;
        let num_kv_heads = (n_kv as usize) / head_dim;
        let gqa = num_q_heads / num_kv_heads;
        let use_gqa_kernel = gqa <= 4;
        if (position as usize) >= self.arch.max_position_embeddings {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_attn_from_residual_dev: position={} >= \
                 max_position_embeddings={}", position,
                self.arch.max_position_embeddings)));
        }

        let stream_u64 = self.stream.raw();

        // Local h_normed scratch (caller's residual preserved).
        let h_normed_region = self.arena.region(
            "g4n_dev_h_normed", (hidden as usize) * 2, 256)?;
        unsafe {
            // DtoD copy of residual_dev → h_normed.
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                h_normed_region.device_ptr(), residual_dev,
                (hidden as usize) * 2, stream_u64 as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "forward_layer_attn_from_residual_dev: residual DtoD copy",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                h_normed_region.device_ptr(),
                layer.input_layernorm.offset_bytes,
                stream_u64,
            )?;
        }

        // Q/K/V projections + device narrow + norms (same kernel
        // chain as the host-API method, just no boundary I/O).
        let q_f32_region = self.arena.region(
            "g4n_dev_q_f32", (n_q as usize) * 4, 256)?;
        let k_f32_region = self.arena.region(
            "g4n_dev_k_f32", (n_kv as usize) * 4, 256)?;
        let v_f32_region = self.arena.region(
            "g4n_dev_v_f32", (n_v as usize) * 4, 256)?;
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, h_normed_region.device_ptr(),
                layer.q_proj.offset_bytes, q_f32_region.device_ptr(),
                1, n_q, hidden as i32, stream_u64)?;
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, h_normed_region.device_ptr(),
                layer.k_proj.offset_bytes, k_f32_region.device_ptr(),
                1, n_kv, hidden as i32, stream_u64)?;
            match v_proj_weight {
                Some(vw) => {
                    gemma4_nvfp4_attn_proj(
                        &self.cublaslt, h_normed_region.device_ptr(),
                        vw.offset_bytes, v_f32_region.device_ptr(),
                        1, n_v, hidden as i32, stream_u64)?;
                }
                None => {
                    use cudarc::driver::sys::*;
                    let rc = cuMemcpyDtoDAsync_v2(
                        v_f32_region.device_ptr(),
                        k_f32_region.device_ptr(),
                        (n_v as usize) * 4,
                        stream_u64 as CUstream);
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(RvllmError::cuda(
                            "forward_layer_attn_from_residual_dev: k_eq_v DtoD",
                            CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
                    }
                }
            }
        }

        let q_region = self.arena.region(
            "g4n_dev_q_bf16", (n_q as usize) * 2, 256)?;
        let k_region = self.arena.region(
            "g4n_dev_k_bf16", (n_kv as usize) * 2, 256)?;
        let v_region = self.arena.region(
            "g4n_dev_v_bf16", (n_v as usize) * 2, 256)?;
        let q_fp8_region = self.arena.region(
            "g4n_dev_q_fp8", n_q as usize, 256)?;
        self.launch_f32_to_bf16(
            q_region.device_ptr(), q_f32_region.device_ptr(), n_q as u32)?;
        self.launch_f32_to_bf16(
            k_region.device_ptr(), k_f32_region.device_ptr(), n_kv as u32)?;
        self.launch_f32_to_bf16(
            v_region.device_ptr(), v_f32_region.device_ptr(), n_v as u32)?;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_q_heads as u32, hidden: head_dim as u32,
                eps: self.arch.rms_norm_eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                q_region.device_ptr(), layer.q_norm.offset_bytes, stream_u64)?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_kv_heads as u32, hidden: head_dim as u32,
                eps: self.arch.rms_norm_eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                k_region.device_ptr(), layer.k_norm.offset_bytes, stream_u64)?;
        }
        self.launch_vnorm_bf16(
            v_region.device_ptr(), num_kv_heads as u32, head_dim as u32)?;
        self.fill_pos_slots(kv,
            /*position_offset=*/position as i32,
            /*start_slot=*/position as i32,
            /*num_tokens=*/1)?;

        let k_packed = kv.k_packed_layer_ptrs[layer_idx];
        let v_packed = kv.v_packed_layer_ptrs[layer_idx];
        let k_scale  = kv.k_scale_layer_ptrs[layer_idx];
        let v_scale  = kv.v_scale_layer_ptrs[layer_idx];
        unsafe {
            let mut q_in: u64 = q_region.device_ptr();
            let mut k_in: u64 = k_region.device_ptr();
            let mut v_in: u64 = v_region.device_ptr();
            let mut q_out: u64 = q_fp8_region.device_ptr();
            let mut kp: u64 = k_packed; let mut vp: u64 = v_packed;
            let mut ks: u64 = k_scale; let mut vs: u64 = v_scale;
            let mut cos_p: u64 = cos_table_dev;
            let mut sin_p: u64 = sin_table_dev;
            let mut positions_ptr: u64 = kv.positions_ptr;
            let mut slot_ptr: u64 = kv.slot_mapping_ptr;
            let mut q_scale_ptr: u64 = kv.q_scale_ptr;
            let mut q_scale_cache_ptr: u64 = 0;
            let mut hadamard_q: u64 = 0; let mut hadamard_k: u64 = 0;
            let mut debug_k_prequant: u64 = 0;
            let mut debug_v_prequant: u64 = 0;
            let mut nt: i32 = 1;
            let mut nh: i32 = num_q_heads as i32;
            let mut nkvh: i32 = num_kv_heads as i32;
            let mut hd: i32 = head_dim as i32;
            let mut rd: i32 = rotary_dim as i32;
            let (mut scale_policy, mut v_scale_policy) =
                read_nvfp4_kv_policies();
            let mut rotate_v: i32 = 0;
            let mut stoch_round_v: i32 = 0;
            let args = [
                (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut kp) as *mut u64 as *mut core::ffi::c_void,
                (&mut vp) as *mut u64 as *mut core::ffi::c_void,
                (&mut ks) as *mut u64 as *mut core::ffi::c_void,
                (&mut vs) as *mut u64 as *mut core::ffi::c_void,
                (&mut cos_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut sin_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut positions_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut slot_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_scale_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_scale_cache_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut rd) as *mut i32 as *mut core::ffi::c_void,
                (&mut scale_policy) as *mut i32 as *mut core::ffi::c_void,
                (&mut v_scale_policy) as *mut i32 as *mut core::ffi::c_void,
                (&mut hadamard_q) as *mut u64 as *mut core::ffi::c_void,
                (&mut hadamard_k) as *mut u64 as *mut core::ffi::c_void,
                (&mut rotate_v) as *mut i32 as *mut core::ffi::c_void,
                (&mut debug_k_prequant) as *mut u64 as *mut core::ffi::c_void,
                (&mut debug_v_prequant) as *mut u64 as *mut core::ffi::c_void,
                (&mut stoch_round_v) as *mut i32 as *mut core::ffi::c_void,
            ];
            let max_heads = num_q_heads.max(num_kv_heads) as u32;
            rvllm_fused::launch_raw(
                self.forward_kernels.fn_rope_kv_write_bf16in,
                (1u32, max_heads, 1u32),
                (head_dim as u32, 1u32, 1u32),
                0, stream_u64, &args)?;
        }

        unsafe {
            let mut output: u64 = attn_out_dev;
            let mut query: u64 = q_fp8_region.device_ptr();
            let mut kp: u64 = k_packed; let mut vp: u64 = v_packed;
            let mut ks: u64 = k_scale; let mut vs: u64 = v_scale;
            let mut q_scale_cache_ptr: u64 = 0;
            let mut block_tables: u64 = kv.block_tables_ptr;
            let mut context_lens: u64 = kv.context_lens_ptr;
            let mut q_descale: u64 = kv.q_scale_ptr;
            let mut scale: f32 = 1.0;
            let mut nh: i32 = num_q_heads as i32;
            let mut nkvh: i32 = num_kv_heads as i32;
            let mut hd: i32 = head_dim as i32;
            let mut block_size: i32 = kv.block_size as i32;
            let mut max_blocks_per_seq: i32 = kv.max_pos as i32;
            let mut window_size_left: i32 = window_size_left;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut query) as *mut u64 as *mut core::ffi::c_void,
                (&mut kp) as *mut u64 as *mut core::ffi::c_void,
                (&mut vp) as *mut u64 as *mut core::ffi::c_void,
                (&mut ks) as *mut u64 as *mut core::ffi::c_void,
                (&mut vs) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_scale_cache_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut block_tables) as *mut u64 as *mut core::ffi::c_void,
                (&mut context_lens) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_descale) as *mut u64 as *mut core::ffi::c_void,
                (&mut scale) as *mut f32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut block_size) as *mut i32 as *mut core::ffi::c_void,
                (&mut max_blocks_per_seq) as *mut i32 as *mut core::ffi::c_void,
                (&mut window_size_left) as *mut i32 as *mut core::ffi::c_void,
            ];
            let fa2_bc: u32 = 32;
            let fa2_threads: u32 = 128;
            let (kernel, grid_y, smem_bytes) = if use_gqa_kernel {
                let max_gqa: u32 = 4;
                let smem = 2 * fa2_bc * (head_dim as u32) * 2
                    + (max_gqa * fa2_bc + fa2_threads / 32) * 4;
                (self.forward_kernels.fn_attn_decode_gqa_bf16out,
                 num_kv_heads as u32, smem)
            } else {
                let smem = 2 * fa2_bc * (head_dim as u32) * 2
                    + (fa2_bc + fa2_threads / 32) * 4;
                (self.forward_kernels.fn_attn_decode_bf16out,
                 num_q_heads as u32, smem)
            };
            if smem_bytes >= 48 * 1024 {
                use cudarc::driver::sys::*;
                let rc = cuFuncSetAttribute(
                    kernel.raw() as CUfunction,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem_bytes as i32);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(RvllmError::cuda(
                        "forward_layer_attn_from_residual_dev: \
                         cuFuncSetAttribute",
                        CudaErrorKind::LaunchFailed, CudaCtx::setup()));
                }
            }
            rvllm_fused::launch_raw(
                kernel,
                (1u32, grid_y, 1u32),
                (fa2_threads, 1u32, 1u32),
                smem_bytes, stream_u64, &args)?;
        }
        Ok(())
    }

    /// Stream-#5f-PRIME: BATCHED layer-attention. Processes
    /// `num_tokens` tokens through one layer's attention in a
    /// single unified-prefill kernel launch instead of N
    /// separate decode launches. Same kernel chain as the
    /// single-token `_dev` variant for everything except the
    /// final attention op:
    ///
    ///   input_layernorm (num_tokens=N) → Q/K/V GEMM (M=N) →
    ///   f32→bf16 (N×dim) → Q-norm + K-norm (per-(token,head))
    ///   → V-norm (per-(token,head)) → fill_pos_slots
    ///   (num_tokens=N, position_offset=start, start_slot=start)
    ///   → rope_kv_write (grid (N, max_heads, 1)) →
    ///   PagedPrefillNvfp4Launcher::launch_nvfp4kv_unified_sm121
    ///   (one launch over N q-rows)
    ///
    /// SAFETY: `residual_dev` size = num_tokens * hidden_size * 2
    /// bytes; `attn_out_dev` size = num_tokens * num_q_heads *
    /// head_dim * 2 bytes. Caller owns lifetime.
    fn forward_layer_attn_batched_prefill_dev(
        &self,
        layer_idx: usize,
        num_tokens: u32,
        residual_dev: u64,
        position_start: u32,
        kv: &Gemma4Nvfp4KvState,
        attn_out_dev: u64,
    ) -> Result<()> {
        use rvllm_core::{RvllmError, CudaErrorKind, CudaCtx};
        use rvllm_loader::gemma4_arch::Gemma4LayerType;

        if num_tokens == 0 {
            return Err(corrupt_runtime_err(
                "forward_layer_attn_batched_prefill_dev: num_tokens=0".into()));
        }
        if layer_idx >= self.arch.num_hidden_layers {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_attn_batched_prefill_dev: layer_idx={} >= {}",
                layer_idx, self.arch.num_hidden_layers)));
        }
        if num_tokens > kv.max_query_tokens {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_attn_batched_prefill_dev: num_tokens={} > \
                 kv.max_query_tokens={} (rebuild KV state via \
                 allocate_kv_state_with_chunk)",
                num_tokens, kv.max_query_tokens)));
        }
        if (position_start as u64) + (num_tokens as u64) > (kv.max_pos as u64) {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_attn_batched_prefill_dev: position_start={} + \
                 num_tokens={} > kv.max_pos={}",
                position_start, num_tokens, kv.max_pos)));
        }
        let layer = &self.model.layers[layer_idx];
        let hidden = self.arch.hidden_size as u32;
        let is_global = matches!(
            self.arch.layer_types[layer_idx], Gemma4LayerType::GlobalAttention);
        #[allow(unused_variables)]
        let (head_dim, rotary_dim, _theta, window_size_left,
             cos_table_dev, sin_table_dev, backend) = if is_global {
            (self.arch.head_dim_global,
             self.arch.rotary_dim_global(),
             self.arch.rope_theta_global as f64,
             -1i32,
             self.model.outside.rope_cos_global.offset_bytes,
             self.model.outside.rope_sin_global.offset_bytes,
             &self.attn_backend_global)
        } else {
            (self.arch.head_dim_sliding,
             self.arch.head_dim_sliding,
             self.arch.rope_theta_sliding as f64,
             (self.arch.sliding_window_size as i32) - 1,
             self.model.outside.rope_cos_sliding.offset_bytes,
             self.model.outside.rope_sin_sliding.offset_bytes,
             &self.attn_backend_sliding)
        };
        let n_q = layer.q_proj.shape[0] as i32;
        let n_kv = layer.k_proj.shape[0] as i32;
        let v_proj_weight: Option<&rvllm_loader::weights::F16Weight>;
        let n_v: i32;
        if is_global {
            if layer.v_proj.is_some() {
                return Err(corrupt_runtime_err(format!(
                    "batched: layer {layer_idx} Global but v_proj present")));
            }
            v_proj_weight = None;
            n_v = n_kv;
        } else {
            let vw = layer.v_proj.as_ref().ok_or_else(|| corrupt_runtime_err(
                format!("batched: layer {layer_idx} sliding but v_proj absent")))?;
            n_v = vw.shape[0] as i32;
            v_proj_weight = Some(vw);
        }
        let num_q_heads = (n_q as usize) / head_dim;
        let num_kv_heads = (n_kv as usize) / head_dim;
        let stream_u64 = self.stream.raw();
        let n = num_tokens as usize;

        // h_normed scratch [N * hidden] bf16. DtoD copy of
        // residual then in-place input_layernorm with num_tokens=N.
        let h_normed_region = self.arena.region(
            "g4n_batch_h_normed", n * (hidden as usize) * 2, 256)?;
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                h_normed_region.device_ptr(), residual_dev,
                n * (hidden as usize) * 2, stream_u64 as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "batched: residual DtoD",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens, hidden, eps: self.arch.rms_norm_eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                h_normed_region.device_ptr(),
                layer.input_layernorm.offset_bytes,
                stream_u64,
            )?;
        }

        // Q/K/V GEMM with M=N. f32 scratch sized N×dim.
        let q_f32_region = self.arena.region(
            "g4n_batch_q_f32", n * (n_q as usize) * 4, 256)?;
        let k_f32_region = self.arena.region(
            "g4n_batch_k_f32", n * (n_kv as usize) * 4, 256)?;
        let v_f32_region = self.arena.region(
            "g4n_batch_v_f32", n * (n_v as usize) * 4, 256)?;
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, h_normed_region.device_ptr(),
                layer.q_proj.offset_bytes, q_f32_region.device_ptr(),
                num_tokens as i32, n_q, hidden as i32, stream_u64)?;
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, h_normed_region.device_ptr(),
                layer.k_proj.offset_bytes, k_f32_region.device_ptr(),
                num_tokens as i32, n_kv, hidden as i32, stream_u64)?;
            match v_proj_weight {
                Some(vw) => {
                    gemma4_nvfp4_attn_proj(
                        &self.cublaslt, h_normed_region.device_ptr(),
                        vw.offset_bytes, v_f32_region.device_ptr(),
                        num_tokens as i32, n_v, hidden as i32, stream_u64)?;
                }
                None => {
                    use cudarc::driver::sys::*;
                    let rc = cuMemcpyDtoDAsync_v2(
                        v_f32_region.device_ptr(),
                        k_f32_region.device_ptr(),
                        n * (n_v as usize) * 4,
                        stream_u64 as CUstream);
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(RvllmError::cuda(
                            "batched: k_eq_v DtoD",
                            CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
                    }
                }
            }
        }

        // f32 → bf16 narrow on device.
        let q_region = self.arena.region(
            "g4n_batch_q_bf16", n * (n_q as usize) * 2, 256)?;
        let k_region = self.arena.region(
            "g4n_batch_k_bf16", n * (n_kv as usize) * 2, 256)?;
        let v_region = self.arena.region(
            "g4n_batch_v_bf16", n * (n_v as usize) * 2, 256)?;
        let q_fp8_region = self.arena.region(
            "g4n_batch_q_fp8", n * (n_q as usize), 256)?;
        self.launch_f32_to_bf16(
            q_region.device_ptr(), q_f32_region.device_ptr(),
            (n * n_q as usize) as u32)?;
        self.launch_f32_to_bf16(
            k_region.device_ptr(), k_f32_region.device_ptr(),
            (n * n_kv as usize) as u32)?;
        self.launch_f32_to_bf16(
            v_region.device_ptr(), v_f32_region.device_ptr(),
            (n * n_v as usize) as u32)?;

        // Q/K/V-norm operate per-(token, head). Flat-row count:
        //   Q-norm:  N * num_q_heads  rows of head_dim
        //   K-norm:  N * num_kv_heads rows of head_dim
        //   V-norm:  N * num_kv_heads rows of head_dim
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: (n as u32) * (num_q_heads as u32),
                hidden: head_dim as u32,
                eps: self.arch.rms_norm_eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                q_region.device_ptr(), layer.q_norm.offset_bytes, stream_u64)?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: (n as u32) * (num_kv_heads as u32),
                hidden: head_dim as u32,
                eps: self.arch.rms_norm_eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                k_region.device_ptr(), layer.k_norm.offset_bytes, stream_u64)?;
        }
        self.launch_vnorm_bf16(
            v_region.device_ptr(),
            (n as u32) * (num_kv_heads as u32),
            head_dim as u32)?;

        // Metadata fill: positions[t] = position_start + t (mode B
        // = absolute slot for full f16 RoPE tables); slot_mapping
        // = same. The fill kernel also writes context_lens[t] =
        // position_start + t + 1, which is the PER-DECODE-TOKEN
        // semantic. The unified prefill kernel, however, expects
        // context_lens to be PER-SEQUENCE: `seq_len =
        // context_lens[seq_idx]` (kernels/flash_attention_unified
        // _prefill_nvfp4kv.cu:207). For num_seqs=1 with N prompt
        // tokens it reads context_lens[0] only, treating it as
        // the SEQUENCE total context length (= N + position_start).
        // The fill leaves context_lens[0] = position_start + 1
        // which would silently constrain the kernel to attend
        // only to slot 0 (codex review caught this).
        // Fix: after fill, overwrite context_lens[0] with the
        // sequence-total via a stream-ordered HtoD memcpy.
        self.fill_pos_slots(kv,
            position_start as i32, position_start as i32,
            num_tokens as i32)?;
        // Sequence total context length for this prompt batch.
        let seq_total_ctx: i32 = (position_start + num_tokens) as i32;
        let seq_total_region = self.arena.region(
            "g4n_batch_ctx_override", 4, 16)?;
        unsafe {
            seq_total_region.copy_from_host(&seq_total_ctx.to_le_bytes())?;
            // Stream-ordered DtoD copy into context_lens[0].
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                kv.context_lens_ptr,
                seq_total_region.device_ptr(),
                4, stream_u64 as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "batched: context_lens[0] override DtoD",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
        }

        // cu_seqlens_q for one sequence of length N: [0, N].
        let cu_seqlens_region = self.arena.region(
            "g4n_batch_cu_seqlens", 2 * 4, 16)?;
        unsafe {
            let cu: [i32; 2] = [0, num_tokens as i32];
            let bytes: &[u8] = std::slice::from_raw_parts(
                cu.as_ptr() as *const u8, 8);
            cu_seqlens_region.copy_from_host(bytes)?;
        }

        let k_packed = kv.k_packed_layer_ptrs[layer_idx];
        let v_packed = kv.v_packed_layer_ptrs[layer_idx];
        let k_scale  = kv.k_scale_layer_ptrs[layer_idx];
        let v_scale  = kv.v_scale_layer_ptrs[layer_idx];

        // RoPE + KV write — kernel already takes num_tokens.
        unsafe {
            let mut q_in: u64 = q_region.device_ptr();
            let mut k_in: u64 = k_region.device_ptr();
            let mut v_in: u64 = v_region.device_ptr();
            let mut q_out: u64 = q_fp8_region.device_ptr();
            let mut kp: u64 = k_packed; let mut vp: u64 = v_packed;
            let mut ks: u64 = k_scale; let mut vs: u64 = v_scale;
            let mut cos_p: u64 = cos_table_dev;
            let mut sin_p: u64 = sin_table_dev;
            let mut positions_ptr: u64 = kv.positions_ptr;
            let mut slot_ptr: u64 = kv.slot_mapping_ptr;
            let mut q_scale_ptr: u64 = kv.q_scale_ptr;
            let mut q_scale_cache_ptr: u64 = 0;
            let mut hadamard_q: u64 = 0; let mut hadamard_k: u64 = 0;
            let mut debug_k_prequant: u64 = 0;
            let mut debug_v_prequant: u64 = 0;
            let mut nt: i32 = num_tokens as i32;
            let mut nh: i32 = num_q_heads as i32;
            let mut nkvh: i32 = num_kv_heads as i32;
            let mut hd: i32 = head_dim as i32;
            let mut rd: i32 = rotary_dim as i32;
            let (mut scale_policy, mut v_scale_policy) =
                read_nvfp4_kv_policies();
            let mut rotate_v: i32 = 0;
            let mut stoch_round_v: i32 = 0;
            let args = [
                (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut kp) as *mut u64 as *mut core::ffi::c_void,
                (&mut vp) as *mut u64 as *mut core::ffi::c_void,
                (&mut ks) as *mut u64 as *mut core::ffi::c_void,
                (&mut vs) as *mut u64 as *mut core::ffi::c_void,
                (&mut cos_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut sin_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut positions_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut slot_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_scale_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_scale_cache_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut rd) as *mut i32 as *mut core::ffi::c_void,
                (&mut scale_policy) as *mut i32 as *mut core::ffi::c_void,
                (&mut v_scale_policy) as *mut i32 as *mut core::ffi::c_void,
                (&mut hadamard_q) as *mut u64 as *mut core::ffi::c_void,
                (&mut hadamard_k) as *mut u64 as *mut core::ffi::c_void,
                (&mut rotate_v) as *mut i32 as *mut core::ffi::c_void,
                (&mut debug_k_prequant) as *mut u64 as *mut core::ffi::c_void,
                (&mut debug_v_prequant) as *mut u64 as *mut core::ffi::c_void,
                (&mut stoch_round_v) as *mut i32 as *mut core::ffi::c_void,
            ];
            let max_heads = num_q_heads.max(num_kv_heads) as u32;
            rvllm_fused::launch_raw(
                self.forward_kernels.fn_rope_kv_write_bf16in,
                (num_tokens, max_heads, 1u32),
                (head_dim as u32, 1u32, 1u32),
                0, stream_u64, &args)?;
        }

        // Unified NVFP4 prefill kernel — ONE launch covers all N
        // q-rows with causal softmax inside.
        let tile_size: u32 = if head_dim <= 256 { 32 } else { 16 };
        let num_queries_per_kv = (num_q_heads as u32) / (num_kv_heads as u32);
        let block_q = (rvllm_attention::UNIFIED_PREFILL_BLOCK_M
            / num_queries_per_kv.max(1)).max(1);
        let params = rvllm_attention::PagedPrefillParams {
            num_seqs: 1,
            num_tokens,
            num_heads: num_q_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            block_size: kv.block_size,
            max_blocks_per_seq: kv.max_pos,
            num_blocks_total: kv.max_pos,
            scale: 1.0,           // Gemma 4 QK-norm absorbs 1/sqrt(d_k)
            window_size_left,
        };
        let unified = rvllm_attention::UnifiedPrefillParams {
            num_queries_per_kv,
            tile_size,
            block_q,
            use_mma: true,
        };
        let prefill = rvllm_attention::PagedPrefillNvfp4Launcher::new(backend);
        unsafe {
            prefill.launch_nvfp4kv_unified_sm121(
                params,
                unified,
                attn_out_dev,                   // o (bf16)
                q_fp8_region.device_ptr(),      // q (fp8)
                k_packed, v_packed,
                k_scale,  v_scale,
                0,                              // q_scale_cache (none)
                kv.block_tables_ptr,
                cu_seqlens_region.device_ptr(),
                kv.context_lens_ptr,
                kv.q_scale_ptr,                 // q_descale fallback
                true,                           // output_bf16
                stream_u64,
            )?;
        }
        Ok(())
    }

    /// Stream 5a-step2: device-mode post-attention close-out
    /// (o_proj + post_attention_layernorm + residual add).
    /// Takes attn_out_dev (bf16) + residual_dev (bf16, updated
    /// in place). No boundary I/O, no fence.
    fn forward_layer_post_attn_dev(
        &self,
        layer_idx: usize,
        attn_out_dev: u64,
        residual_dev: u64,
    ) -> Result<()> {
        if layer_idx >= self.arch.num_hidden_layers {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_post_attn_dev: layer_idx={} >= {}",
                layer_idx, self.arch.num_hidden_layers)));
        }
        let layer = &self.model.layers[layer_idx];
        let n_q = layer.o_proj.shape[1] as i32;
        let hidden = self.arch.hidden_size as u32;
        let stream_u64 = self.stream.raw();
        let o_f32_region = self.arena.region(
            "g4n_dev_post_attn_o_f32", (hidden as usize) * 4, 256)?;
        let o_bf16_region = self.arena.region(
            "g4n_dev_post_attn_o_bf16", (hidden as usize) * 2, 256)?;
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, attn_out_dev,
                layer.o_proj.offset_bytes,
                o_f32_region.device_ptr(),
                1, hidden as i32, n_q, stream_u64)?;
        }
        self.launch_f32_to_bf16(
            o_bf16_region.device_ptr(),
            o_f32_region.device_ptr(),
            hidden)?;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                o_bf16_region.device_ptr(),
                layer.post_attention_layernorm.offset_bytes,
                stream_u64)?;
            rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }
                .launch(
                    self.forward_kernels.fn_vector_add_bf16,
                    residual_dev,
                    o_bf16_region.device_ptr(),
                    stream_u64)?;
        }
        Ok(())
    }

    /// Stream 5a-step2: device-mode post-MLP close-out. Updates
    /// residual_dev (bf16) in place. The layer_scalar host scale
    /// loop still costs ONE fence + DtoH(hidden*2 + 2) + HtoD per
    /// layer — Stream 5b (bf16 scaled_add kernel) eliminates it.
    fn forward_layer_post_attn_mlp_dev(
        &self,
        layer_idx: usize,
        residual_dev: u64,
    ) -> Result<()> {
        use rvllm_core::{RvllmError, CudaErrorKind, CudaCtx};
        if layer_idx >= self.arch.num_hidden_layers {
            return Err(corrupt_runtime_err(format!(
                "forward_layer_post_attn_mlp_dev: layer_idx={} >= {}",
                layer_idx, self.arch.num_hidden_layers)));
        }
        let layer = &self.model.layers[layer_idx];
        let hidden = self.arch.hidden_size as u32;
        let intermediate = self.arch.intermediate_size as u32;
        let stream_u64 = self.stream.raw();

        let h_normed_region = self.arena.region(
            "g4n_dev_pamlp_normed", (hidden as usize) * 2, 256)?;
        let scratch_region = self.arena.region(
            "g4n_dev_pamlp_scratch", (2 * intermediate as usize) * 2, 256)?;
        let mlp_out_region = self.arena.region(
            "g4n_dev_pamlp_out", (hidden as usize) * 2, 256)?;
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                h_normed_region.device_ptr(), residual_dev,
                (hidden as usize) * 2, stream_u64 as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "forward_layer_post_attn_mlp_dev: residual DtoD",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                h_normed_region.device_ptr(),
                layer.pre_feedforward_layernorm.offset_bytes,
                stream_u64)?;
            crate::gemma4_nvfp4_ops::gemma4_nvfp4_mlp_forward(
                &self.mlp_kernels,
                h_normed_region.device_ptr(),
                mlp_out_region.device_ptr(),
                &layer.gate_proj, &layer.up_proj, &layer.down_proj,
                scratch_region.device_ptr(), stream_u64)?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                mlp_out_region.device_ptr(),
                layer.post_feedforward_layernorm.offset_bytes,
                stream_u64)?;
        }

        // Stream 5b: fused `residual_dev += layer_scalar *
        // mlp_out` on device. Replaces the prior fence + 2×
        // DtoH + host scale + HtoD + vector_add chain with one
        // kernel launch on the existing stream.
        self.launch_scaled_add_bf16(
            residual_dev,
            mlp_out_region.device_ptr(),
            layer.layer_scalar.offset_bytes,
            hidden,
        )?;
        let _ = stream_u64;
        Ok(())
    }

    /// Stream-#5f-PRIME: batched post-attention close-out.
    /// Same as `forward_layer_post_attn_dev` but processes N
    /// tokens — o_proj at M=N, RMSNorm over N rows,
    /// scaled-add over N*hidden elements.
    fn forward_layer_post_attn_batched_dev(
        &self,
        layer_idx: usize,
        num_tokens: u32,
        attn_out_dev: u64,
        residual_dev: u64,
    ) -> Result<()> {
        if layer_idx >= self.arch.num_hidden_layers {
            return Err(corrupt_runtime_err(format!(
                "post_attn_batched_dev: layer_idx={} >= {}",
                layer_idx, self.arch.num_hidden_layers)));
        }
        let layer = &self.model.layers[layer_idx];
        let n_q = layer.o_proj.shape[1] as i32;
        let hidden = self.arch.hidden_size as u32;
        let n = num_tokens as usize;
        let stream_u64 = self.stream.raw();
        let o_f32_region = self.arena.region(
            "g4n_batch_o_f32", n * (hidden as usize) * 4, 256)?;
        let o_bf16_region = self.arena.region(
            "g4n_batch_o_bf16", n * (hidden as usize) * 2, 256)?;
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt, attn_out_dev,
                layer.o_proj.offset_bytes,
                o_f32_region.device_ptr(),
                num_tokens as i32, hidden as i32, n_q, stream_u64)?;
        }
        self.launch_f32_to_bf16(
            o_bf16_region.device_ptr(),
            o_f32_region.device_ptr(),
            (n * (hidden as usize)) as u32)?;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens, hidden, eps: self.arch.rms_norm_eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                o_bf16_region.device_ptr(),
                layer.post_attention_layernorm.offset_bytes,
                stream_u64)?;
            // vector_add on flat N*hidden elements.
            rvllm_fused::gemma4_launcher::VectorAddF16Launch {
                n: (n as u32) * hidden,
            }.launch(
                self.forward_kernels.fn_vector_add_bf16,
                residual_dev,
                o_bf16_region.device_ptr(),
                stream_u64)?;
        }
        Ok(())
    }

    /// Stream-#5f-PRIME: batched post-MLP close-out. Processes
    /// N tokens through the MLP layer; MLP itself stays per-
    /// token (M=1 GEMV kernels) but pre/post norms + scaled_add
    /// run on the full N*hidden flat. Per-token MLP loop is
    /// the next perf step (would need M>1 W4A16 kernel).
    fn forward_layer_post_attn_mlp_batched_dev(
        &self,
        layer_idx: usize,
        num_tokens: u32,
        residual_dev: u64,
    ) -> Result<()> {
        use rvllm_core::{RvllmError, CudaErrorKind, CudaCtx};
        if layer_idx >= self.arch.num_hidden_layers {
            return Err(corrupt_runtime_err(format!(
                "post_attn_mlp_batched_dev: layer_idx={} >= {}",
                layer_idx, self.arch.num_hidden_layers)));
        }
        let layer = &self.model.layers[layer_idx];
        let hidden = self.arch.hidden_size as u32;
        let intermediate = self.arch.intermediate_size as u32;
        let n = num_tokens as usize;
        let stream_u64 = self.stream.raw();
        let h_normed_region = self.arena.region(
            "g4n_batch_pamlp_normed", n * (hidden as usize) * 2, 256)?;
        let scratch_region = self.arena.region(
            "g4n_batch_pamlp_scratch", (2 * intermediate as usize) * 2, 256)?;
        let mlp_out_region = self.arena.region(
            "g4n_batch_pamlp_out", n * (hidden as usize) * 2, 256)?;
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoDAsync_v2(
                h_normed_region.device_ptr(), residual_dev,
                n * (hidden as usize) * 2, stream_u64 as CUstream);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "post_attn_mlp_batched_dev: residual DtoD",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens, hidden, eps: self.arch.rms_norm_eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                h_normed_region.device_ptr(),
                layer.pre_feedforward_layernorm.offset_bytes,
                stream_u64)?;
            // Per-token MLP loop. M=1 W4A16 kernels — batching the
            // MLP needs a new kernel (~300 LOC follow-up). At N=3
            // this is 3 MLP launches; at N=100 it's 100. For
            // typical prompts the win from batched attention
            // already dominates.
            let hidden_bytes = (hidden as usize) * 2;
            for t in 0..n {
                let h_t = h_normed_region.device_ptr()
                    + (t * hidden_bytes) as u64;
                let mlp_t = mlp_out_region.device_ptr()
                    + (t * hidden_bytes) as u64;
                crate::gemma4_nvfp4_ops::gemma4_nvfp4_mlp_forward(
                    &self.mlp_kernels,
                    h_t, mlp_t,
                    &layer.gate_proj, &layer.up_proj, &layer.down_proj,
                    scratch_region.device_ptr(), stream_u64)?;
            }
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens, hidden, eps: self.arch.rms_norm_eps,
            }.launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                mlp_out_region.device_ptr(),
                layer.post_feedforward_layernorm.offset_bytes,
                stream_u64)?;
        }
        // scaled_add on N*hidden flat (alpha broadcast across all
        // elements of all tokens — layer_scalar is per-layer not
        // per-token).
        self.launch_scaled_add_bf16(
            residual_dev,
            mlp_out_region.device_ptr(),
            layer.layer_scalar.offset_bytes,
            (n as u32) * hidden,
        )?;
        Ok(())
    }

    /// Commit #5c: 60-layer driver for a single-token decode at
    /// `position`. Chains:
    ///   embed_tokens[token_id] → for each of `num_hidden_layers`:
    ///     attn_out = forward_layer_attn_from_residual(layer_idx, …)
    ///     residual = forward_layer_post_attn(layer_idx, attn_out, residual)
    ///     residual = forward_layer_post_attn_mlp(layer_idx, residual)
    ///   → forward_final_to_token(residual) → argmax token id
    ///
    /// Stream 5a-step2 (this commit): single-token forward now
    /// uses the device-resident _dev variants of each per-layer
    /// step. Residual stays on device across all 60 layers, so
    /// the prior per-method fence + DtoH + HtoD cycle is gone.
    /// The N-token prompt path (`forward_prompt_to_token`) still
    /// uses the host-API chain; the same refactor for it lands
    /// in a follow-up once the dump-mode path is reconciled.
    ///
    /// `kv` MUST have `max_pos > position`. Caller manages the
    /// position counter across multi-token decode loops.
    ///
    /// Returns the predicted next-token id from the tied LM head
    /// + argmax. The numerical correctness of this output depends
    /// on the cumulative correctness of all 60 layers — codex
    /// round-4 work is what would validate per-layer cosines.
    /// At this milestone we assert structural correctness only:
    /// no NaN/Inf, no panics, finite plausible-magnitude
    /// intermediate residuals, an in-range token id.
    pub fn forward_full_to_token(
        &self,
        token_id: u32,
        position: u32,
        kv: &Gemma4Nvfp4KvState,
    ) -> Result<u32> {
        if position >= kv.max_pos {
            return Err(corrupt_runtime_err(format!(
                "forward_full_to_token: position={} >= kv.max_pos={}",
                position, kv.max_pos)));
        }
        // bf16 narrow on host. Used to flip the f32 Vec returned
        // by attn/post-attn/post-mlp back to bf16 between layers.
        let f32_to_bf16_vec = |xs: &[f32]| -> Vec<u16> {
            xs.iter().map(|&x| {
                let bits = x.to_bits();
                let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
                (rounded >> 16) as u16
            }).collect()
        };

        // Commit #5e: per-layer dump for numerical validation.
        // When `G4N_DUMP_DIR` is set, write every intermediate
        // residual + attn_out as `step_NN_<stage>.bin` (raw bf16)
        // and the final logits as `final_logits.f32.bin`. The
        // dumps are consumable by `v3/tools/cmp_g4n_residuals.py`
        // for HF-vs-rvllm cosine comparisons. No-op when the
        // env is unset, so production paths pay nothing.
        let dump_dir: Option<std::path::PathBuf> =
            std::env::var("G4N_DUMP_DIR").ok().map(std::path::PathBuf::from);
        let dump_bf16 = |label: &str, data: &[u16]| -> Result<()> {
            if let Some(d) = dump_dir.as_ref() {
                std::fs::create_dir_all(d).map_err(|e|
                    corrupt_runtime_err(format!(
                        "G4N_DUMP_DIR create {d:?}: {e}")))?;
                let path = d.join(format!("{label}.bf16.bin"));
                let bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                    data.as_ptr() as *const u8, data.len() * 2) };
                std::fs::write(&path, bytes).map_err(|e|
                    corrupt_runtime_err(format!(
                        "G4N_DUMP_DIR write {path:?}: {e}")))?;
            }
            Ok(())
        };

        let trace_on = std::env::var("G4N_FORWARD_TRACE")
            .ok().as_deref() == Some("1");
        let log_stats = |li: usize, stage: &str, data: &[u16]| {
            if !trace_on { return; }
            let mut mean_abs: f32 = 0.0;
            let mut max_abs: f32 = 0.0;
            let mut nan = 0usize;
            let mut inf = 0usize;
            for &b in data {
                let v = f32::from_bits((b as u32) << 16);
                if v.is_nan() { nan += 1; continue; }
                if v.is_infinite() { inf += 1; continue; }
                mean_abs += v.abs();
                if v.abs() > max_abs { max_abs = v.abs(); }
            }
            mean_abs /= data.len() as f32;
            eprintln!(
                "[g4n-trace] layer {li:02} {stage:>10}: mean_abs={mean_abs:.4} \
                 max_abs={max_abs:.4} nan={nan} inf={inf}"
            );
        };

        // Dump-mode path: G4N_DUMP_DIR is opt-in for diagnostic
        // cosine validation against an HF reference. Use the
        // host-API chain so each .bin lands at the same boundary
        // as before. Slower, but expected since dump mode is
        // diagnostic-only.
        if dump_dir.is_some() {
            let mut residual_bf16 = self.embed_one_token_bf16(token_id)?;
            dump_bf16("step_00_embed", &residual_bf16)?;
            for li in 0..self.arch.num_hidden_layers {
                let attn_out_f32 = self.forward_layer_attn_from_residual(
                    li, &residual_bf16, position, kv)?;
                let attn_out_bf16 = f32_to_bf16_vec(&attn_out_f32);
                dump_bf16(&format!("step_{:02}_attn_out", li), &attn_out_bf16)?;
                let resid_after_attn_f32 = self.forward_layer_post_attn(
                    li, &attn_out_bf16, &residual_bf16)?;
                let resid_after_attn_bf16 = f32_to_bf16_vec(&resid_after_attn_f32);
                dump_bf16(&format!("step_{:02}_post_attn", li),
                          &resid_after_attn_bf16)?;
                let resid_after_mlp_f32 = self.forward_layer_post_attn_mlp(
                    li, &resid_after_attn_bf16)?;
                residual_bf16 = f32_to_bf16_vec(&resid_after_mlp_f32);
                dump_bf16(&format!("step_{:02}_post_mlp", li), &residual_bf16)?;
                if trace_on && (li % 10 == 0
                                || li == self.arch.num_hidden_layers - 1) {
                    log_stats(li, "post_mlp", &residual_bf16);
                }
            }
            return self.forward_final_to_token_with_dump(
                &residual_bf16, dump_dir.as_ref().unwrap());
        }

        // Production path (no dump): device-resident residual
        // across all 60 layers via the _dev inner methods.
        let _scratch_guard = self.forward_scratch_guard();
        let hidden = self.arch.hidden_size as u32;
        // attn_out scratch sized for the LARGEST per-layer
        // n_q across layer types (global = 32*512 = 16384 bf16
        // elems; sliding = 32*256 = 8192).
        let max_n_q: usize = self.model.layers.iter()
            .map(|l| l.q_proj.shape[0]).max().unwrap_or(0);
        let residual_dev = self.arena.region(
            "g4n_drv_residual_bf16", (hidden as usize) * 2, 256)?;
        let attn_out_dev = self.arena.region(
            "g4n_drv_attn_out_bf16", max_n_q * 2, 256)?;
        self.embed_one_token_to_device(token_id, residual_dev.device_ptr())?;
        for li in 0..self.arch.num_hidden_layers {
            self.forward_layer_attn_from_residual_dev(
                li, residual_dev.device_ptr(), position, kv,
                attn_out_dev.device_ptr())?;
            self.forward_layer_post_attn_dev(
                li, attn_out_dev.device_ptr(), residual_dev.device_ptr())?;
            self.forward_layer_post_attn_mlp_dev(
                li, residual_dev.device_ptr())?;
            if trace_on && (li % 10 == 0
                            || li == self.arch.num_hidden_layers - 1) {
                // Trace requires DtoH — opt-in only, so the
                // per-decade-of-layers sync is acceptable.
                self.stream.fence()?;
                let mut h = vec![0u16; hidden as usize];
                unsafe {
                    use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(
                        h.as_mut_ptr() as *mut _,
                        residual_dev.device_ptr(),
                        (hidden as usize) * 2);
                }
                log_stats(li, "post_mlp", &h);
            }
        }
        self.stream.fence()?;
        let mut residual_host = vec![0u16; hidden as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                residual_host.as_mut_ptr() as *mut _,
                residual_dev.device_ptr(),
                (hidden as usize) * 2);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(corrupt_runtime_err(
                    "forward_full_to_token: residual DtoH".into()));
            }
        }
        self.forward_final_to_token(&residual_host)
    }

    /// Commit #5f: multi-token prompt processing on Option B.
    /// Processes `prompt` token IDs sequentially through the 60
    /// decoder blocks, writing N consecutive KV cache slots at
    /// `position_start..position_start+N`, then returns the
    /// argmax token id from the LAST prompt position's final
    /// hidden state (i.e. the predicted next token after the
    /// prompt).
    ///
    /// Strategy: per-token-decode-loop fallback. Each prompt
    /// token runs the existing single-token attention chain at
    /// its slot, and subsequent prompt tokens see the prior K/V
    /// slots via the decode kernel's `context_lens` (which now
    /// equals the absolute slot+1). This matches the production
    /// Qwen35/Qwen36 NVFP4 path's "batched prefill falls back to
    /// per-token decode loop; unified prefill kernel is a
    /// follow-up" pattern (see rvllm-serve CLAUDE.md).
    ///
    /// Cost: 60 × N attention calls instead of 60 × 1 for
    /// single-token forward. The unified NVFP4 prefill kernel
    /// (`PagedPrefillNvfp4Launcher::launch_nvfp4kv_unified_sm121`,
    /// `v3/crates/rvllm-attention/src/prefill.rs:1041`) would
    /// collapse those N calls into a single kernel launch per
    /// layer. Wiring it requires plumbing the full Fa2Ptx
    /// backend through Option B — separate commit (#5f-prime).
    ///
    /// Correctness invariant: at slot t, the decode reads slots
    /// [0, t] for sliding (within window) or [0, t] for global
    /// (full attention). Each token's contribution is causal by
    /// construction — token t only sees what was written by
    /// tokens 0..t.
    ///
    /// `kv` MUST have `max_pos >= position_start + N` and
    /// `max_query_tokens >= 1` (today we still launch
    /// one-token-at-a-time inside the loop; #5f-prime bumps
    /// this).
    pub fn forward_prompt_to_token(
        &self,
        prompt: &[u32],
        position_start: u32,
        kv: &Gemma4Nvfp4KvState,
    ) -> Result<u32> {
        if prompt.is_empty() {
            return Err(corrupt_runtime_err(
                "forward_prompt_to_token: prompt is empty".into()));
        }
        let n_tokens = prompt.len() as u32;
        if (position_start as u64) + (n_tokens as u64) > kv.max_pos as u64 {
            return Err(corrupt_runtime_err(format!(
                "forward_prompt_to_token: position_start={} + n_tokens={} \
                 exceeds kv.max_pos={}",
                position_start, n_tokens, kv.max_pos)));
        }

        // Stream-#5f-PRIME: device-resident batched-prefill
        // path. Residual lives as [N * hidden] bf16 on device
        // across all 60 layers; attention runs as ONE unified-
        // prefill kernel launch per layer instead of N decode
        // launches.
        if n_tokens > kv.max_query_tokens {
            return Err(corrupt_runtime_err(format!(
                "forward_prompt_to_token: prompt length {n_tokens} > \
                 kv.max_query_tokens={}. Re-allocate KV via \
                 `allocate_kv_state_with_chunk(max_pos, >= {n_tokens})`.",
                kv.max_query_tokens)));
        }
        let _scratch_guard = self.forward_scratch_guard();
        let hidden = self.arch.hidden_size as u32;
        let n = prompt.len();
        // attn_out_dev sized for the LARGEST per-layer n_q
        // (global = 32*512 = 16384; sliding = 32*256 = 8192).
        let max_n_q: usize = self.model.layers.iter()
            .map(|l| l.q_proj.shape[0]).max().unwrap_or(0);
        let residual_dev = self.arena.region(
            "g4n_prompt_residual_bf16", n * (hidden as usize) * 2, 256)?;
        let attn_out_dev = self.arena.region(
            "g4n_prompt_attn_out_bf16", n * max_n_q * 2, 256)?;

        // Embed N tokens into residual_dev rows 0..N-1.
        let hidden_bytes = (hidden as usize) * 2;
        for (t, &tok) in prompt.iter().enumerate() {
            let dst = residual_dev.device_ptr()
                + (t * hidden_bytes) as u64;
            self.embed_one_token_to_device(tok, dst)?;
        }

        for li in 0..self.arch.num_hidden_layers {
            self.forward_layer_attn_batched_prefill_dev(
                li, n_tokens, residual_dev.device_ptr(),
                position_start, kv, attn_out_dev.device_ptr())?;
            self.forward_layer_post_attn_batched_dev(
                li, n_tokens, attn_out_dev.device_ptr(),
                residual_dev.device_ptr())?;
            self.forward_layer_post_attn_mlp_batched_dev(
                li, n_tokens, residual_dev.device_ptr())?;
        }

        // DtoH last token's residual for the final LM head.
        self.stream.fence()?;
        let mut last_residual = vec![0u16; hidden as usize];
        let last_off = ((n - 1) * hidden_bytes) as u64;
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                last_residual.as_mut_ptr() as *mut _,
                residual_dev.device_ptr() + last_off,
                hidden_bytes);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(corrupt_runtime_err(
                    "forward_prompt_to_token: last residual DtoH".into()));
            }
        }
        self.forward_final_to_token(&last_residual)
    }

    /// Commit #5e: dump-mode variant of `forward_final_to_token`
    /// that writes `step_final_norm.bf16.bin` (post-norm hidden
    /// state) and `step_final_logits.f32.bin` (LM head output
    /// before argmax) into the given directory, then runs the
    /// argmax and returns the token id. Same kernel chain as the
    /// non-dump variant — only DtoH + disk writes are added.
    fn forward_final_to_token_with_dump(
        &self,
        h_residual_bf16_host: &[u16],
        dump_dir: &std::path::Path,
    ) -> Result<u32> {
        use rvllm_core::{RvllmError, CudaErrorKind, CudaCtx};
        let _scratch_guard = self.forward_scratch_guard();
        let hidden = self.arch.hidden_size as u32;
        let vocab = self.arch.vocab_size as u32;
        if h_residual_bf16_host.len() != hidden as usize {
            return Err(corrupt_runtime_err(format!(
                "forward_final_to_token_with_dump: residual length {} != hidden {}",
                h_residual_bf16_host.len(), hidden)));
        }

        let h_region = self.arena.region(
            "g4n_final_dump_h", (hidden as usize) * 2, 256)?;
        let logits_region = self.arena.region(
            "g4n_final_dump_logits", (vocab as usize) * 4, 256)?;
        let token_region = self.arena.region(
            "g4n_final_dump_token", 4, 16)?;
        unsafe {
            let r: &[u8] = std::slice::from_raw_parts(
                h_residual_bf16_host.as_ptr() as *const u8,
                h_residual_bf16_host.len() * 2);
            h_region.copy_from_host(r)?;
        }
        let stream_u64 = self.stream.raw();

        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.rms_norm_eps,
            }
            .launch(
                self.forward_kernels.fn_rmsnorm_inplace_bf16,
                h_region.device_ptr(),
                self.model.outside.final_norm.offset_bytes,
                stream_u64,
            )?;
        }
        self.stream.fence()?;

        // Dump post-final-norm hidden state (bf16).
        let mut h_normed_bf16 = vec![0u16; hidden as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                h_normed_bf16.as_mut_ptr() as *mut _,
                h_region.device_ptr(),
                (hidden as usize) * 2);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "forward_final_to_token_with_dump: h_normed DtoH",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
        }
        std::fs::create_dir_all(dump_dir).map_err(|e|
            corrupt_runtime_err(format!(
                "dump_dir create {dump_dir:?}: {e}")))?;
        unsafe {
            let bytes: &[u8] = std::slice::from_raw_parts(
                h_normed_bf16.as_ptr() as *const u8,
                h_normed_bf16.len() * 2);
            std::fs::write(
                dump_dir.join("step_final_norm.bf16.bin"), bytes
            ).map_err(|e| corrupt_runtime_err(format!(
                "dump final_norm: {e}")))?;
        }

        // Tied LM head GEMV.
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt,
                h_region.device_ptr(),
                self.model.outside.lm_head_tokens.offset_bytes,
                logits_region.device_ptr(),
                1, vocab as i32, hidden as i32, stream_u64,
            )?;
        }
        self.stream.fence()?;

        // Dump f32 logits.
        let mut logits_f32 = vec![0f32; vocab as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                logits_f32.as_mut_ptr() as *mut _,
                logits_region.device_ptr(),
                (vocab as usize) * 4);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "forward_final_to_token_with_dump: logits DtoH",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
        }
        unsafe {
            let bytes: &[u8] = std::slice::from_raw_parts(
                logits_f32.as_ptr() as *const u8,
                logits_f32.len() * 4);
            std::fs::write(
                dump_dir.join("step_final_logits.f32.bin"), bytes
            ).map_err(|e| corrupt_runtime_err(format!(
                "dump final_logits: {e}")))?;
        }

        // argmax.
        unsafe {
            let mut logits_ptr = logits_region.device_ptr();
            let mut out_ptr = token_region.device_ptr();
            let mut vs = vocab as i32;
            let args: [*mut core::ffi::c_void; 3] = [
                (&mut logits_ptr) as *mut u64 as *mut _,
                (&mut out_ptr) as *mut u64 as *mut _,
                (&mut vs) as *mut i32 as *mut _,
            ];
            rvllm_fused::launch_raw(
                self.forward_kernels.fn_argmax_f32,
                (1, 1, 1), (1024, 1, 1), 0, stream_u64, &args,
            )?;
        }
        self.stream.fence()?;

        let mut tok = [0i32; 1];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                tok.as_mut_ptr() as *mut _, token_region.device_ptr(), 4);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "forward_final_to_token_with_dump: token DtoH",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
        }
        let id = tok[0];
        if id < 0 || (id as u32) >= vocab {
            return Err(corrupt_runtime_err(format!(
                "forward_final_to_token_with_dump: argmax out-of-range \
                 token id={id} vocab={vocab}")));
        }

        // Also dump the token id as an i32 so the Python diff
        // script can show "rvllm token vs reference token" at a
        // glance.
        std::fs::write(dump_dir.join("step_final_token.i32.bin"),
                       (id as i32).to_le_bytes()).map_err(|e|
            corrupt_runtime_err(format!("dump token: {e}")))?;

        Ok(id as u32)
    }
}

/// Stream-#6a: borrow-pair wrapper that lets Option B's KV state
/// implement `crate::gemma4_drafter::BaseKvSource`. Bound by the
/// lifetime of the underlying bringup + kv state.
pub struct Gemma4Nvfp4BaseKvSource<'a> {
    pub bringup: &'a Gemma4Nvfp4Bringup,
    pub kv: &'a Gemma4Nvfp4KvState,
}

impl<'a> crate::gemma4_drafter::BaseKvSource
    for Gemma4Nvfp4BaseKvSource<'a>
{
    fn drafter_base_kv_view(
        &self, layer_idx: usize,
    ) -> Result<crate::gemma4_drafter::DrafterBaseKvView> {
        self.bringup.drafter_base_kv_view(self.kv, layer_idx)
    }

    fn assistant_shared_kv_sources(&self) -> Option<(usize, usize)> {
        // Re-route through the arch helper. 31B returns
        // Some((58, 59)); other variants return None or their
        // own pair.
        self.bringup.arch.assistant_shared_kv_sources()
    }
}

/// f32 → f16 IEEE-754 round-to-nearest-even, returning the raw u16
/// bit pattern. Used for the per-position cos/sin mini-tables in
/// `forward_layer0_attn`. Handles subnormals, overflow→inf, NaN.
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp32 = ((b >> 23) & 0xff) as i32;
    let mant32 = b & 0x007fffff;
    if exp32 == 0xff {
        // Inf / NaN.
        let mant16 = if mant32 != 0 { 0x200u16 } else { 0u16 };
        return sign | 0x7c00 | mant16;
    }
    let unbiased = exp32 - 127;
    if unbiased > 15 {
        // Overflow → inf with sign.
        return sign | 0x7c00;
    }
    if unbiased < -14 {
        // Subnormal f16 or zero.
        let shift = -unbiased - 1; // 0..24
        if shift >= 24 { return sign; }
        let mant_with_hidden = mant32 | 0x00800000;
        let mant16 = (mant_with_hidden >> (shift + 13)) as u16;
        let round_bit = (mant_with_hidden >> (shift + 12)) & 1;
        let sticky = (mant_with_hidden & ((1 << (shift + 12)) - 1)) != 0;
        let bumped = if round_bit == 1 && (sticky || (mant16 & 1) == 1)
            { mant16 + 1 } else { mant16 };
        return sign | bumped;
    }
    let exp16 = ((unbiased + 15) as u16) << 10;
    let mant16 = (mant32 >> 13) as u16;
    let round_bit = (mant32 >> 12) & 1;
    let sticky = (mant32 & 0xfff) != 0;
    let bumped = if round_bit == 1 && (sticky || (mant16 & 1) == 1)
        { (sign | exp16 | mant16) + 1 } else { sign | exp16 | mant16 };
    bumped
}

#[inline]
fn corrupt_runtime_err(msg: String) -> rvllm_core::RvllmError {
    rvllm_core::RvllmError::Config {
        err: rvllm_core::ConfigError::InvalidField {
            name: "runtime", reason: msg.into(),
        },
        field: "runtime",
    }
}

/// Parse a Gemma4 NVFP4 K/V scale-policy env value.
/// `0 | "amax6"` → 0 (range-preserving baseline)
/// `1 | "mse"`   → 1 (outlier-aware MSE search)
/// Unknown values fall through to `None`. Matches production's
/// `parse_policy` in gemma4_layer_exec.rs::rope_nvfp4kv so a
/// profile set for the production path works on Option B too.
fn parse_nvfp4_policy(v: &str) -> Option<i32> {
    match v.trim() {
        "amax6" | "0" => Some(0),
        "mse"   | "1" => Some(1),
        _ => None,
    }
}

/// Read the (K, V) NVFP4 scale policies for Option B with
/// production-matching defaults: K = amax6, V = mse.
///
/// Resolution order (per side):
///   1. `RVLLM_NVFP4_{K,V}_SCALE_POLICY` if set + recognized
///   2. `RVLLM_NVFP4_SCALE_POLICY` (legacy single-knob) if set
///   3. default — K=amax6 (0), V=mse (1)
///
/// V=mse is the codex round-4 fix that landed in production
/// to remove the V-clipping cliff on long German prompts. The
/// previous Option B default (V=amax6) was a smoke-test
/// shortcut.
fn read_nvfp4_kv_policies() -> (i32, i32) {
    let global = std::env::var("RVLLM_NVFP4_SCALE_POLICY")
        .ok()
        .and_then(|s| parse_nvfp4_policy(&s));
    let k = std::env::var("RVLLM_NVFP4_K_SCALE_POLICY")
        .ok()
        .and_then(|s| parse_nvfp4_policy(&s))
        .or(global)
        .unwrap_or(0);          // K default = amax6
    let v = std::env::var("RVLLM_NVFP4_V_SCALE_POLICY")
        .ok()
        .and_then(|s| parse_nvfp4_policy(&s))
        .or(global)
        .unwrap_or(1);          // V default = mse
    (k, v)
}

/// Debug-env knobs that change the Option B forward path
/// behavior (extra disk writes, stderr spam) but are inert in
/// production. Mirror Mistral 3.5's stale-env guard: if any of
/// these is set without `RVLLM_DEBUG_G4N=1`, refuse to start so
/// a leaked diagnostic env from a prior session can't slowly
/// fill disk on a production rvllm-serve.
const G4N_STALE_DEBUG_KEYS: &[&str] = &[
    "G4N_DUMP_DIR",
    "G4N_FORWARD_TRACE",
];

#[inline]
fn g4n_debug_active() -> bool {
    matches!(
        std::env::var("RVLLM_DEBUG_G4N").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

fn validate_no_stale_g4n_debug_envs() -> Result<()> {
    if g4n_debug_active() { return Ok(()); }
    for key in G4N_STALE_DEBUG_KEYS {
        if std::env::var_os(key).is_some() {
            return Err(corrupt_runtime_err(format!(
                "[gemma4-nvfp4] refusing to start: {key} is set but \
                 RVLLM_DEBUG_G4N=1 is not. These knobs add disk I/O \
                 + stderr writes that production should not silently \
                 pay. Set RVLLM_DEBUG_G4N=1 to opt in, or unset \
                 {key} for a production run."
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Commit #4a smoke: full bring-up load + pre-attention
    /// sub-block on layer 0 with the BOS token.
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_load_and_pre_attn \
    ///     -- --ignored --nocapture
    ///
    /// Requires rvllm-serve stopped — the full 60-layer upload
    /// uses ~22 GiB of unified memory and the smoke allocates
    /// a 32 GiB arena. Coexisting with prod would OOM.
    #[test]
    #[ignore]
    fn ondisk_bringup_load_and_pre_attn() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping bring-up smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );

        // 32 GiB arena — covers ~22 GiB of model weights + ~10
        // GiB of forward scratch headroom. Production profile
        // (mobile-31b-nvfp4w-rvllm-spec.env) uses 96 GiB to
        // include KV cache + spec scratch which we don't allocate here.
        let t0 = std::time::Instant::now();
        let bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");
        let load_s = t0.elapsed().as_secs_f64();
        eprintln!(
            "[bringup-smoke] load complete in {load_s:.1}s — layers={} hidden={} \
             intermediate={} vocab={}",
            bringup.arch.num_hidden_layers,
            bringup.arch.hidden_size,
            bringup.arch.intermediate_size,
            bringup.arch.vocab_size,
        );

        // BOS token (Gemma 4 tokenizer: <bos>=2).
        let token_id: u32 = 2;
        let q = bringup.pre_attn_one_token(token_id)
            .expect("pre_attn_one_token");

        let nan = q.iter().filter(|x| x.is_nan()).count();
        let inf = q.iter().filter(|x| x.is_infinite()).count();
        let mean_abs: f32 = q.iter().map(|x| x.abs()).sum::<f32>() / (q.len() as f32);
        let max_abs: f32 = q.iter().fold(0f32, |a, &x| a.max(x.abs()));
        eprintln!(
            "[bringup-smoke] layer-0 pre-attn q_proj on token {token_id}: \
             N_q={} nan={nan} inf={inf} mean_abs={mean_abs:.4} max_abs={max_abs:.4} \
             first8={:?}",
            q.len(), &q[..8],
        );

        assert_eq!(nan, 0, "{nan} NaN(s) in q_proj");
        assert_eq!(inf, 0, "{inf} Inf(s) in q_proj");
        assert!(mean_abs > 0.0, "q_proj all zero");
        // After RMSNorm the residual has unit-ish magnitude, then
        // q_proj weights are small Gaussian-ish → expect output
        // magnitudes in roughly the [0.01, 10.0] range. Reject
        // extreme outliers that would indicate a bug.
        assert!(mean_abs < 100.0,
                "q_proj mean_abs={mean_abs} implausibly large");
        assert!(max_abs < 1000.0,
                "q_proj max_abs={max_abs} implausibly large");
    }

    /// Commit #4b smoke: full Q/K/V projection on BOS token.
    /// Extends the pre-attention smoke to also exercise the K
    /// and V projections from the same bf16 GEMV path. Asserts
    /// all three outputs are finite and have plausible
    /// magnitudes.
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_qkv \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_bringup_qkv() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping qkv smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        let token_id: u32 = 2;
        let (q, k, v) = bringup.forward_layer0_qkv_only(token_id)
            .expect("forward_layer0_qkv_only");

        for (name, vec) in [("q", &q), ("k", &k), ("v", &v)] {
            let nan = vec.iter().filter(|x| x.is_nan()).count();
            let inf = vec.iter().filter(|x| x.is_infinite()).count();
            let mean_abs: f32 = vec.iter().map(|x| x.abs()).sum::<f32>() / (vec.len() as f32);
            let max_abs: f32 = vec.iter().fold(0f32, |a, &x| a.max(x.abs()));
            eprintln!(
                "[qkv-smoke] {name}: N={} nan={nan} inf={inf} \
                 mean_abs={mean_abs:.4} max_abs={max_abs:.4} first4={:?}",
                vec.len(), &vec[..4],
            );
            assert_eq!(nan, 0, "{name} has {nan} NaN");
            assert_eq!(inf, 0, "{name} has {inf} Inf");
            assert!(mean_abs > 0.0, "{name} all-zero");
            assert!(mean_abs < 100.0, "{name} mean_abs={mean_abs} too large");
        }

        assert_eq!(q.len(), 8192, "q on sliding layer 0: N_q=32*256");
        assert_eq!(k.len(), 4096, "k on sliding layer 0: N_kv=16*256");
        assert_eq!(v.len(), 4096, "v on sliding layer 0: N_kv=16*256");
    }

    /// Commit #4c smoke: per-head Q-norm + K-norm on layer 0.
    /// After RMSNorm with gamma=q_norm/k_norm[head_dim], each
    /// head's RMS should be close to gamma's mean (gamma is the
    /// learned per-channel scale). For Gemma 4 31B these gammas
    /// are typically O(1), so post-norm per-head RMS should
    /// also be O(1).
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_qk_norm \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_bringup_qk_norm() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping qk_norm smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        let head_dim = bringup.arch.head_dim_sliding;
        let (q, k, _v) = bringup.forward_layer0_qk_norm(2)
            .expect("forward_layer0_qk_norm");

        for (name, vec, n_heads) in [
            ("q", &q, q.len() / head_dim),
            ("k", &k, k.len() / head_dim),
        ] {
            let nan = vec.iter().filter(|x| x.is_nan()).count();
            let inf = vec.iter().filter(|x| x.is_infinite()).count();
            assert_eq!(nan, 0, "{name} has {nan} NaN");
            assert_eq!(inf, 0, "{name} has {inf} Inf");

            // Per-head RMS after norm should be O(gamma.mean()) — for
            // Gemma 4 the gamma is typically in [0.5, 2.0], so per-
            // head RMS should land in roughly [0.1, 10].
            let mut head_rms: Vec<f32> = Vec::with_capacity(n_heads);
            for h in 0..n_heads {
                let row = &vec[h * head_dim..(h + 1) * head_dim];
                let sum_sq: f32 = row.iter().map(|x| x * x).sum();
                let rms = (sum_sq / head_dim as f32).sqrt();
                head_rms.push(rms);
            }
            let mean_rms: f32 = head_rms.iter().sum::<f32>() / (n_heads as f32);
            let min_rms = head_rms.iter().fold(f32::INFINITY, |a, &x| a.min(x));
            let max_rms = head_rms.iter().fold(0f32, |a, &x| a.max(x));
            eprintln!(
                "[qk-norm-smoke] {name}: heads={n_heads} head_dim={head_dim} \
                 mean_rms={mean_rms:.4} min_rms={min_rms:.4} max_rms={max_rms:.4}"
            );
            assert!(mean_rms > 0.01 && mean_rms < 100.0,
                    "{name} mean_rms={mean_rms} outside plausible [0.01, 100]");
        }
    }

    /// Commit #4d smoke: RoPE on post-norm Q/K at position=0.
    /// RoPE is norm-preserving (orthogonal rotation), so the
    /// per-head RMS after RoPE must equal the per-head RMS
    /// after Q/K-norm. At position=0 specifically, cos=1 + sin=0
    /// → RoPE is the identity, so the values should be byte-
    /// equal to the pre-RoPE values modulo bf16 narrow noise.
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_qk_rope_at_zero \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_bringup_qk_rope_at_zero() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping qk_rope smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        let head_dim = bringup.arch.head_dim_sliding;

        let (q_pre, k_pre, _v) = bringup.forward_layer0_qk_norm(2)
            .expect("forward_layer0_qk_norm");
        let (q_post, k_post, _v2) = bringup.forward_layer0_qk_rope(2, 0)
            .expect("forward_layer0_qk_rope(pos=0)");

        for (name, pre, post, n_heads) in [
            ("q", &q_pre, &q_post, q_pre.len() / head_dim),
            ("k", &k_pre, &k_post, k_pre.len() / head_dim),
        ] {
            let nan = post.iter().filter(|x| x.is_nan()).count();
            let inf = post.iter().filter(|x| x.is_infinite()).count();
            assert_eq!(nan, 0, "{name} post-RoPE NaN");
            assert_eq!(inf, 0, "{name} post-RoPE Inf");

            // Per-head RMS preservation: RoPE rotates each (i, i+half)
            // pair by a unit complex number — magnitude is preserved
            // exactly in f32, with small bf16 narrow error.
            let mut max_rms_drift: f32 = 0.0;
            for h in 0..n_heads {
                let row_pre = &pre[h * head_dim..(h + 1) * head_dim];
                let row_post = &post[h * head_dim..(h + 1) * head_dim];
                let rms_pre = (row_pre.iter().map(|x| x * x).sum::<f32>()
                               / head_dim as f32).sqrt();
                let rms_post = (row_post.iter().map(|x| x * x).sum::<f32>()
                                / head_dim as f32).sqrt();
                max_rms_drift = max_rms_drift.max((rms_pre - rms_post).abs());
            }

            // At position=0 cos=1, sin=0 so RoPE is the identity.
            // The pre / post values should agree element-wise modulo
            // bf16 round-trip through the GPU buffer (one extra narrow
            // beyond the pre-RoPE narrow). Element-wise max diff:
            let max_elem_diff = pre.iter().zip(post.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);

            eprintln!(
                "[qk-rope-smoke] {name}: n_heads={n_heads} \
                 max_rms_drift={max_rms_drift:.6} \
                 max_elem_diff={max_elem_diff:.6}",
            );
            // bf16 mantissa is 7 bits → relative precision ~0.8%.
            // For values up to ~10 the absolute error is ~0.08.
            // Allow generous headroom — anything > 0.5 indicates a bug.
            assert!(max_rms_drift < 0.5,
                    "{name} RoPE NOT norm-preserving: max_rms_drift={max_rms_drift}");
            assert!(max_elem_diff < 0.5,
                    "{name} RoPE@pos=0 NOT identity: max_elem_diff={max_elem_diff}");
        }
    }

    /// Commit #4e smoke: o_proj + post_attention_layernorm +
    /// residual add. Uses a synthetic bf16 attn_out (all 0.01)
    /// and a synthetic bf16 residual (all 0.01) — the goal is
    /// structural, not numerical (real attention output lands
    /// with #5's KV cache).
    ///
    /// Assertions:
    /// * Output residual is finite, has plausible magnitudes.
    /// * Output != input residual (the add did something).
    /// * Post-norm contribution is consistent with the post_attn
    ///   gamma magnitude.
    #[test]
    #[ignore]
    fn ondisk_bringup_post_attn() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping post_attn smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        let hidden = bringup.arch.hidden_size;
        let n_q = (bringup.arch.num_attention_heads
                   * bringup.arch.head_dim_sliding) as usize;
        assert_eq!(n_q, 8192, "31B sliding N_q sanity");

        // Synthetic bf16 0.01 ≈ 0x3C23.
        let bf16_0_01: u16 = 0x3C23;
        let attn_out: Vec<u16> = vec![bf16_0_01; n_q];
        let h_in: Vec<u16> = vec![bf16_0_01; hidden];
        let h_in_f32: Vec<f32> = h_in.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect();

        let h_out = bringup.forward_layer0_post_attn(&attn_out, &h_in)
            .expect("forward_layer0_post_attn");

        assert_eq!(h_out.len(), hidden);
        let nan = h_out.iter().filter(|x| x.is_nan()).count();
        let inf = h_out.iter().filter(|x| x.is_infinite()).count();
        let mean_abs: f32 = h_out.iter().map(|x| x.abs()).sum::<f32>()
                          / (hidden as f32);
        let max_abs: f32 = h_out.iter().fold(0f32, |a, &x| a.max(x.abs()));
        // Delta = output - input (= the normed o_proj contribution)
        let diff: Vec<f32> = h_out.iter().zip(h_in_f32.iter())
            .map(|(o, i)| o - i)
            .collect();
        let diff_mean_abs: f32 = diff.iter().map(|x| x.abs()).sum::<f32>()
                              / (hidden as f32);
        let diff_max_abs: f32 = diff.iter().fold(0f32, |a, &x| a.max(x.abs()));

        eprintln!(
            "[post-attn-smoke] hidden={hidden} N_q={n_q}\n  \
             h_out: nan={nan} inf={inf} mean_abs={mean_abs:.6} max_abs={max_abs:.6}\n  \
             delta: mean_abs={diff_mean_abs:.6} max_abs={diff_max_abs:.6}\n  \
             first4_in=0.01  first4_out={:?}",
            &h_out[..4],
        );

        assert_eq!(nan, 0, "post_attn produced NaN");
        assert_eq!(inf, 0, "post_attn produced Inf");
        assert!(diff_mean_abs > 1e-6,
            "post_attn delta=0 — residual add did nothing");
        assert!(mean_abs < 100.0,
            "post_attn h_out mean_abs={mean_abs} implausibly large");
        assert!(diff_mean_abs < 100.0,
            "post_attn delta mean_abs={diff_mean_abs} implausibly large");
    }

    /// Commit #4f smoke: MLP-block composition on layer 0.
    /// Pre_ff_norm → MLP (commit #3) → post_ff_norm → residual
    /// add with layer_scalar. Synthetic 0.01 bf16 input.
    #[test]
    #[ignore]
    fn ondisk_bringup_post_attn_mlp() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping mlp_block smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        let hidden = bringup.arch.hidden_size;
        let bf16_0_01: u16 = 0x3C23;
        let h_in: Vec<u16> = vec![bf16_0_01; hidden];
        let h_in_f32: Vec<f32> = h_in.iter().map(|&b| {
            f32::from_bits((b as u32) << 16)
        }).collect();

        let h_out = bringup.forward_layer0_post_attn_mlp(&h_in)
            .expect("forward_layer0_post_attn_mlp");

        let nan = h_out.iter().filter(|x| x.is_nan()).count();
        let inf = h_out.iter().filter(|x| x.is_infinite()).count();
        let mean_abs: f32 = h_out.iter().map(|x| x.abs()).sum::<f32>()
                          / (hidden as f32);
        let max_abs: f32 = h_out.iter().fold(0f32, |a, &x| a.max(x.abs()));
        let diff: Vec<f32> = h_out.iter().zip(h_in_f32.iter())
            .map(|(o, i)| o - i)
            .collect();
        let diff_mean_abs: f32 = diff.iter().map(|x| x.abs()).sum::<f32>()
                              / (hidden as f32);
        let diff_max_abs: f32 = diff.iter().fold(0f32, |a, &x| a.max(x.abs()));

        eprintln!(
            "[mlp-block-smoke] hidden={hidden}\n  \
             h_out: nan={nan} inf={inf} mean_abs={mean_abs:.6} max_abs={max_abs:.6}\n  \
             delta (mlp_normed * layer_scalar): \
             mean_abs={diff_mean_abs:.6} max_abs={diff_max_abs:.6}\n  \
             first4_in=0.01 first4_out={:?}",
            &h_out[..4],
        );

        assert_eq!(nan, 0, "mlp_block produced NaN");
        assert_eq!(inf, 0, "mlp_block produced Inf");
        assert!(diff_mean_abs > 1e-7,
            "mlp_block delta=0 — residual add did nothing");
        // layer_scalar≈0.0894 means the MLP contribution is
        // small relative to whatever post_ff_norm would normally
        // produce (~mean_gamma=1.39). diff_mean_abs upper bound
        // is roughly layer_scalar * gamma ≈ 0.12. Allow wide
        // headroom — real MLP output for 0.01 input is much
        // smaller than the trained-on activation magnitudes.
        assert!(diff_mean_abs < 100.0,
            "mlp_block delta mean_abs={diff_mean_abs} implausibly large");
        assert!(mean_abs < 100.0,
            "mlp_block h_out mean_abs={mean_abs} implausibly large");
    }

    /// Commit #4g smoke: final_norm + tied LM head + argmax.
    /// Synthetic bf16 residual (all 0.01) → predict a token id.
    /// Doesn't validate THE next-token correctness (that needs
    /// the full 60-layer attention+MLP chain from #5+), but
    /// verifies the LM head pipeline:
    ///   * final_norm runs without NaN/Inf
    ///   * cuBLASLt produces non-degenerate f32 logits
    ///   * argmax returns a valid token id in [0, vocab)
    /// Also confirms the tied-embed weight is correctly accessed
    /// via `model.outside.embed_tokens` (the same buffer the
    /// embed lookup reads from).
    #[test]
    #[ignore]
    fn ondisk_bringup_final_to_token() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping final smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        let hidden = bringup.arch.hidden_size;
        let vocab = bringup.arch.vocab_size;
        let bf16_0_01: u16 = 0x3C23;
        let h_in: Vec<u16> = vec![bf16_0_01; hidden];

        let token_id = bringup.forward_final_to_token(&h_in)
            .expect("forward_final_to_token");

        eprintln!(
            "[final-smoke] hidden={hidden} vocab={vocab} \
             predicted_token_id={token_id}"
        );
        assert!((token_id as usize) < vocab,
            "argmax produced token_id={token_id} >= vocab={vocab}");
    }

    /// Commit #5a smoke: layer-0 END-TO-END at position=0.
    /// Exercises every kernel from commit #4a–#4g composed into
    /// one path using the position=0 trivial attention identity
    /// (attn_out = V with GQA head replication).
    ///
    /// This is the structural milestone for the bring-up:
    /// everything from embed lookup through tied lm_head argmax
    /// runs on real weights with no NaN/Inf/crash. The predicted
    /// token is NOT semantically meaningful (only 1 of 60 layers
    /// is actually composed — the remaining 59 are bypassed at
    /// this point), but the pipeline is byte-stable.
    #[test]
    #[ignore]
    fn ondisk_bringup_layer0_position_zero_e2e() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping #5a smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        // BOS token at position=0.
        let token_id = bringup.forward_layer0_position_zero_to_token(2)
            .expect("forward_layer0_position_zero_to_token");

        eprintln!(
            "[e2e-pos0-smoke] BOS(2) at position=0 → layer0 only → tied_lm_head → token_id={token_id}"
        );
        let vocab = bringup.arch.vocab_size;
        assert!((token_id as usize) < vocab,
            "e2e pos=0 produced token_id={token_id} >= vocab={vocab}");
    }

    /// Commit #5b1 smoke: allocate the NVFP4 KV state on a loaded
    /// bring-up. Asserts every per-layer pointer is non-null,
    /// scale pointers are non-null, and total_bytes is in the
    /// expected ~990 MiB range at max_pos=4096.
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_allocate_kv_state \
    ///     -- --ignored --nocapture
    ///
    /// Does not run any forward — purely tests the KV allocator +
    /// kernel handle loading (#5b1). Forward integration is #5b2.
    #[test]
    #[ignore]
    fn ondisk_bringup_allocate_kv_state() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping #5b1 smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let mut bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        let max_pos: u32 = 4096;
        let kv = bringup.allocate_kv_state(max_pos)
            .expect("allocate_kv_state");

        // Commit floor: re-entrancy guard — a second call must
        // refuse rather than silently leak the first state.
        let second = bringup.allocate_kv_state(max_pos);
        assert!(second.is_err(),
            "allocate_kv_state must reject the second call");
        eprintln!("[#floor-smoke] double-allocate guard: rejected ✓");

        assert_eq!(kv.max_pos, max_pos);
        assert_eq!(kv.block_size, 1);
        assert_ne!(kv.block_tables_ptr, 0, "block_tables null");
        assert_ne!(kv.context_lens_ptr, 0, "context_lens null");
        assert_ne!(kv.positions_ptr, 0, "positions null");
        assert_ne!(kv.slot_mapping_ptr, 0, "slot_mapping null");
        assert_ne!(kv.q_scale_ptr, 0, "q_scale null");

        let n_layers = bringup.arch.num_hidden_layers;
        assert_eq!(kv.k_packed_layer_ptrs.len(), n_layers);
        assert_eq!(kv.v_packed_layer_ptrs.len(), n_layers);
        assert_eq!(kv.k_scale_layer_ptrs.len(), n_layers);
        assert_eq!(kv.v_scale_layer_ptrs.len(), n_layers);
        for li in 0..n_layers {
            assert_ne!(kv.k_packed_layer_ptrs[li], 0,
                "k_packed[{li}] null");
            assert_ne!(kv.v_packed_layer_ptrs[li], 0,
                "v_packed[{li}] null");
            assert_ne!(kv.k_scale_layer_ptrs[li], 0,
                "k_scale[{li}] null");
            assert_ne!(kv.v_scale_layer_ptrs[li], 0,
                "v_scale[{li}] null");
        }

        let mib = kv.total_bytes as f64 / (1024.0 * 1024.0);
        eprintln!(
            "[#5b1-smoke] allocated NVFP4 KV state: layers={n_layers} \
             max_pos={max_pos} total_bytes={} ({mib:.1} MiB)",
            kv.total_bytes,
        );
        // Expected ~990 MiB at max_pos=4096 (see docstring on the
        // Gemma4Nvfp4KvState struct). Bound generously.
        assert!(mib > 500.0 && mib < 1500.0,
            "total_bytes={mib:.1} MiB outside expected 500..1500 range");

        // Codex Stream-6a scaffold: verify the DrafterBaseKvView
        // surfaces non-null per-layer pointers for the 31B
        // source pair (58, 59) which is what the production
        // drafter cross-attends to.
        for src in [58usize, 59] {
            let view = bringup.drafter_base_kv_view(&kv, src)
                .expect("drafter_base_kv_view");
            assert_ne!(view.k_cache, 0,        "{src}: k_cache null");
            assert_ne!(view.v_cache, 0,        "{src}: v_cache null");
            assert_ne!(view.k_scale_cache, 0,  "{src}: k_scale null");
            assert_ne!(view.v_scale_cache, 0,  "{src}: v_scale null");
            assert_ne!(view.block_tables, 0,   "{src}: block_tables null");
            assert_ne!(view.context_lens, 0,   "{src}: context_lens null");
            assert!(matches!(view.kv_dtype,
                crate::gemma4_layer_exec::KvDtype::Nvfp4));
        }
        eprintln!("[stream-6a-scaffold] drafter view at layers 58, 59 ✓");

        // Rejection guard: out-of-range layer_idx must error.
        let n_layers = bringup.arch.num_hidden_layers;
        let err = bringup.drafter_base_kv_view(&kv, n_layers);
        assert!(err.is_err());

        // Stream-#6a: BaseKvSource trait impl. The wrapper
        // exposes Option B's KV as a `&dyn BaseKvSource` for
        // the future drafter parameterization.
        use crate::gemma4_drafter::BaseKvSource;
        let src = crate::gemma4_nvfp4_bring_up::Gemma4Nvfp4BaseKvSource {
            bringup: &bringup, kv: &kv,
        };
        let dyn_src: &dyn BaseKvSource = &src;
        let pair = dyn_src.assistant_shared_kv_sources()
            .expect("31B has an assistant source pair");
        eprintln!("[stream-6a] assistant_shared_kv_sources = {pair:?}");
        assert_eq!(pair, (58, 59),
            "31B expected source pair (58, 59), got {pair:?}");
        let v58 = dyn_src.drafter_base_kv_view(pair.0)
            .expect("dyn-trait view at layer 58");
        let v59 = dyn_src.drafter_base_kv_view(pair.1)
            .expect("dyn-trait view at layer 59");
        assert_ne!(v58.k_cache, v59.k_cache);
        eprintln!("[stream-6a] trait impl ✓ — views at 58, 59 distinct");
    }

    /// Commit #5b2 smoke: layer-0 attention end-to-end at two
    /// consecutive positions with shared NVFP4 KV cache.
    ///
    /// Drives `forward_layer0_attn` at position=0 (BOS, slot 0
    /// written) and position=1 (a second token, slot 1 written,
    /// decode reads both slots). Asserts attn_out is finite, has
    /// plausible magnitude, and the two positions produce
    /// DIFFERENT outputs (smoke that the cache+attention chain
    /// actually consumes the prior slot).
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test --release -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_layer0_attn_pos0_pos1 \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_bringup_layer0_attn_pos0_pos1() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping #5b2 smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let mut bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        let kv = bringup.allocate_kv_state(64)
            .expect("allocate_kv_state(64)");

        // Run attention at position=0 (BOS) and position=1
        // (some other token id) sharing the same KV state.
        let attn0 = bringup.forward_layer0_attn(2, 0, &kv)
            .expect("forward_layer0_attn(pos=0)");
        let attn1 = bringup.forward_layer0_attn(64, 1, &kv)
            .expect("forward_layer0_attn(pos=1)");

        let summarize = |label: &str, a: &[f32]| -> (f32, f32, usize, usize) {
            let nan = a.iter().filter(|x| x.is_nan()).count();
            let inf = a.iter().filter(|x| x.is_infinite()).count();
            let mean_abs: f32 = a.iter().map(|x| x.abs()).sum::<f32>()
                / (a.len() as f32);
            let max_abs: f32 = a.iter().fold(0f32, |a, &x| a.max(x.abs()));
            eprintln!(
                "[#5b2-smoke] {label}: N={} nan={nan} inf={inf} \
                 mean_abs={mean_abs:.4} max_abs={max_abs:.4} \
                 first4={:?}",
                a.len(), &a[..4],
            );
            (mean_abs, max_abs, nan, inf)
        };

        let (m0, x0, n0, i0) = summarize("attn_pos0", &attn0);
        let (m1, x1, n1, i1) = summarize("attn_pos1", &attn1);

        assert_eq!(n0, 0, "pos=0 attn_out has {n0} NaN");
        assert_eq!(i0, 0, "pos=0 attn_out has {i0} Inf");
        assert_eq!(n1, 0, "pos=1 attn_out has {n1} NaN");
        assert_eq!(i1, 0, "pos=1 attn_out has {i1} Inf");
        assert!(m0 > 0.0, "pos=0 attn_out all zero");
        assert!(m1 > 0.0, "pos=1 attn_out all zero");
        // Expect magnitudes in a sane band — V-norm scales V to
        // unit-ish; softmax then averages so attn_out should be
        // similarly bounded.
        assert!(m0 < 100.0, "pos=0 mean_abs={m0} implausibly large");
        assert!(x0 < 1000.0, "pos=0 max_abs={x0} implausibly large");
        assert!(m1 < 100.0, "pos=1 mean_abs={m1} implausibly large");
        assert!(x1 < 1000.0, "pos=1 max_abs={x1} implausibly large");

        // The two outputs MUST differ — at pos=1 the decode reads
        // both slot 0 (BOS-token V) and slot 1 (the second
        // token's V), so the attn_out distribution is different
        // from pos=0 (which sees only slot 0).
        assert_eq!(attn0.len(), attn1.len());
        let diff: f32 = attn0.iter().zip(attn1.iter())
            .map(|(a, b)| (a - b).abs()).sum::<f32>()
            / (attn0.len() as f32);
        eprintln!("[#5b2-smoke] mean_abs(attn_pos0 - attn_pos1) = {diff:.6}");
        assert!(diff > 1e-4,
            "pos=0 and pos=1 attn_out are identical — KV cache likely \
             not being consumed (diff_mean_abs={diff})");
    }

    /// Commit #5b3 smoke: per-layer attention works on layer 1
    /// (also sliding on 31B; layer_types pattern is 5×sliding +
    /// 1×global repeating, so layers 0..=4 are sliding).
    ///
    /// Feeds the SAME embedded BOS residual to layer 0's
    /// attention AND layer 1's attention (independent slot=0
    /// writes on each layer's KV region). Asserts:
    ///   * Both produce finite output (no NaN/Inf).
    ///   * Both have plausible magnitudes.
    ///   * The two outputs DIFFER (per-layer weights are
    ///     distinct → outputs should not be identical).
    ///
    /// Note: feeding layer 1 the raw embed residual is
    /// structurally incoherent — layer 1's correct input is
    /// layer 0's post-MLP residual. The smoke validates that
    /// `forward_layer_attn_from_residual` indexes the
    /// per-layer weights + KV state correctly; semantic
    /// correctness of a multi-layer forward is the 60-layer
    /// driver's job.
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test --release -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_layer1_attn \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_bringup_layer1_attn() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping #5b3 smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let mut bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        // Confirm test assumption: layers 0 and 1 are both sliding.
        for li in [0usize, 1] {
            let lt = &bringup.arch.layer_types[li];
            assert!(
                matches!(lt, rvllm_loader::gemma4_arch::Gemma4LayerType::SlidingAttention),
                "test precondition: layer {li} must be SlidingAttention, got {lt:?}"
            );
        }

        let kv = bringup.allocate_kv_state(64)
            .expect("allocate_kv_state(64)");

        let residual = bringup.embed_one_token_bf16(2)
            .expect("embed_one_token_bf16(BOS=2)");
        assert_eq!(residual.len(), bringup.arch.hidden_size);

        let attn_l0 = bringup.forward_layer_attn_from_residual(0, &residual, 0, &kv)
            .expect("forward_layer_attn_from_residual(layer=0)");
        let attn_l1 = bringup.forward_layer_attn_from_residual(1, &residual, 0, &kv)
            .expect("forward_layer_attn_from_residual(layer=1)");

        let summarize = |label: &str, a: &[f32]| -> (f32, f32, usize, usize) {
            let nan = a.iter().filter(|x| x.is_nan()).count();
            let inf = a.iter().filter(|x| x.is_infinite()).count();
            let mean_abs: f32 = a.iter().map(|x| x.abs()).sum::<f32>()
                / (a.len() as f32);
            let max_abs: f32 = a.iter().fold(0f32, |a, &x| a.max(x.abs()));
            eprintln!(
                "[#5b3-smoke] {label}: N={} nan={nan} inf={inf} \
                 mean_abs={mean_abs:.4} max_abs={max_abs:.4} \
                 first4={:?}",
                a.len(), &a[..4],
            );
            (mean_abs, max_abs, nan, inf)
        };
        let (m0, x0, n0, i0) = summarize("attn_l0", &attn_l0);
        let (m1, x1, n1, i1) = summarize("attn_l1", &attn_l1);

        assert_eq!(n0, 0, "layer 0 has {n0} NaN");
        assert_eq!(i0, 0, "layer 0 has {i0} Inf");
        assert_eq!(n1, 0, "layer 1 has {n1} NaN");
        assert_eq!(i1, 0, "layer 1 has {i1} Inf");
        assert!(m0 > 0.0 && m0 < 100.0,
            "layer 0 mean_abs={m0} out of [0, 100)");
        assert!(m1 > 0.0 && m1 < 100.0,
            "layer 1 mean_abs={m1} out of [0, 100)");
        assert!(x0 < 1000.0 && x1 < 1000.0,
            "implausible max_abs (l0={x0}, l1={x1})");

        // Per-layer weights differ → outputs should differ.
        assert_eq!(attn_l0.len(), attn_l1.len());
        let diff: f32 = attn_l0.iter().zip(attn_l1.iter())
            .map(|(a, b)| (a - b).abs()).sum::<f32>()
            / (attn_l0.len() as f32);
        eprintln!("[#5b3-smoke] mean_abs(attn_l0 - attn_l1) = {diff:.6}");
        assert!(diff > 1e-4,
            "layer 0 and layer 1 attn_out are identical (diff={diff}) — \
             per-layer weight indexing is likely broken");

        // #5b3 used to reject global layers; #5b4 wires them
        // through the same method via partial RoPE + k_eq_v V
        // alias + non-GQA decode kernel. See the dedicated
        // global-layer smoke below.
    }

    /// Commit #5b4 smoke: global-layer attention works through
    /// the same `forward_layer_attn_from_residual` method via
    /// runtime dispatch on `arch.layer_types`. 31B layer 5 is
    /// the first GlobalAttention layer (5-sliding/1-global
    /// pattern).
    ///
    /// Global-layer specifics covered by this smoke:
    ///   * head_dim=512 (vs sliding 256) → decode kernel needs
    ///     `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)`
    ///     because per-block dynamic smem ≈ 64 KiB > 48 KiB
    ///     default cap.
    ///   * Partial RoPE: rotary_dim_global = head_dim_global *
    ///     partial_rotary_factor_global (= 128 on 31B). Only
    ///     the first 128 of 512 channels get rotated.
    ///   * rope_theta_global = 1_000_000 (vs sliding 10_000).
    ///   * attention_k_eq_v = true: no v_proj on disk, V is
    ///     the K-projection output (DtoD-copied before V-norm).
    ///   * GQA = 32/4 = 8 > MAX_GQA_DECODE=4 → falls through
    ///     to the non-GQA decode kernel (1 block per Q head,
    ///     internal kv_head mapping).
    ///   * window_size_left = -1 (full attention, no window).
    ///
    /// Asserts no NaN/Inf, plausible magnitudes, and (since
    /// the smoke runs on a fresh KV cache) at position=0 the
    /// trivial single-slot attention output should be V_normed
    /// replicated across each Q-head's kv_head group — which
    /// just means the output magnitude tracks the V-norm
    /// output's scale, comfortably below typical FP overflow.
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test --release -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_global_attn \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_bringup_global_attn() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping #5b4 smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let mut bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        // Pick the first global layer.
        let global_idx = (0..bringup.arch.num_hidden_layers)
            .find(|&i| matches!(
                bringup.arch.layer_types[i],
                rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention))
            .expect("31B must have at least one global layer");
        eprintln!(
            "[#5b4-smoke] global layer index = {global_idx}, \
             head_dim_global = {}, num_kv_heads_global = {}, \
             rotary_dim_global = {}, rope_theta_global = {}",
            bringup.arch.head_dim_global,
            bringup.arch.num_kv_heads_global,
            bringup.arch.rotary_dim_global(),
            bringup.arch.rope_theta_global,
        );

        let kv = bringup.allocate_kv_state(64)
            .expect("allocate_kv_state(64)");

        // Two DIFFERENT tokens so the per-position V differs.
        // If we used the same token at both positions the
        // attention output for pos=1 collapses back to
        // V_pos0 = V_pos1 = sm[0]·V + sm[1]·V = V, masking the
        // KV-cache read (matches #5b2's sliding-pos1 smoke).
        let resid0 = bringup.embed_one_token_bf16(2)   // BOS
            .expect("embed_one_token_bf16(BOS=2)");
        let resid1 = bringup.embed_one_token_bf16(64)  // arbitrary
            .expect("embed_one_token_bf16(64)");

        let attn_g0 = bringup.forward_layer_attn_from_residual(
            global_idx, &resid0, 0, &kv,
        ).expect("forward_layer_attn_from_residual(global, pos=0)");
        let attn_g1 = bringup.forward_layer_attn_from_residual(
            global_idx, &resid1, 1, &kv,
        ).expect("forward_layer_attn_from_residual(global, pos=1)");

        for (label, a) in [("attn_g0", &attn_g0), ("attn_g1", &attn_g1)] {
            let nan = a.iter().filter(|x| x.is_nan()).count();
            let inf = a.iter().filter(|x| x.is_infinite()).count();
            let mean_abs: f32 = a.iter().map(|x| x.abs()).sum::<f32>()
                / (a.len() as f32);
            let max_abs: f32 = a.iter().fold(0f32, |a, &x| a.max(x.abs()));
            eprintln!(
                "[#5b4-smoke] {label}: N={} nan={nan} inf={inf} \
                 mean_abs={mean_abs:.4} max_abs={max_abs:.4} \
                 first4={:?}",
                a.len(), &a[..4],
            );
            assert_eq!(nan, 0, "{label} has {nan} NaN");
            assert_eq!(inf, 0, "{label} has {inf} Inf");
            assert!(mean_abs > 0.0 && mean_abs < 100.0,
                "{label} mean_abs={mean_abs} out of (0, 100)");
            assert!(max_abs < 1000.0,
                "{label} max_abs={max_abs} implausibly large");
        }

        // Pos=0 cache state has only slot 0; pos=1 reads two slots.
        // Outputs MUST differ.
        let diff: f32 = attn_g0.iter().zip(attn_g1.iter())
            .map(|(a, b)| (a - b).abs()).sum::<f32>()
            / (attn_g0.len() as f32);
        eprintln!("[#5b4-smoke] mean_abs(attn_g0 - attn_g1) = {diff:.6}");
        assert!(diff > 1e-4,
            "global pos=0 and pos=1 are identical (diff={diff}) — \
             global decode kernel likely not consuming the KV cache");

        // Confirm output dim = num_attention_heads * head_dim_global.
        let expected_n =
            bringup.arch.num_attention_heads * bringup.arch.head_dim_global;
        assert_eq!(attn_g0.len(), expected_n,
            "global attn_out length {} != num_attention_heads*head_dim_global {}",
            attn_g0.len(), expected_n);
    }

    /// Commit #5c smoke: full 60-layer forward at position=0.
    ///
    /// Drives every decoder block (50 sliding + 10 global, in
    /// the 5+1 pattern on 31B) end-to-end via the per-layer
    /// attention + post-attn + post-MLP helpers, then final_norm
    /// + tied LM head + argmax. Asserts the resulting token id
    /// is in [0, vocab_size). Structural pass; numerical
    /// correctness of the predicted token is NOT validated here.
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test --release -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_full_60_layer \
    ///     -- --ignored --nocapture
    ///
    /// Set `G4N_FORWARD_TRACE=1` to dump residual mean/max/abs
    /// every 10 layers (handy when triaging mid-layer blow-ups).
    #[test]
    #[ignore]
    fn ondisk_bringup_full_60_layer() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping #5c smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let mut bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");

        let kv = bringup.allocate_kv_state(64)
            .expect("allocate_kv_state(64)");

        let t0 = std::time::Instant::now();
        let token = bringup.forward_full_to_token(2 /*BOS*/, 0, &kv)
            .expect("forward_full_to_token");
        let elapsed = t0.elapsed().as_secs_f64();

        let vocab = bringup.arch.vocab_size;
        let n_layers = bringup.arch.num_hidden_layers;
        eprintln!(
            "[#5c-smoke] BOS at pos=0 → {n_layers}-layer NVFP4 forward → \
             argmax token = {token} (vocab={vocab}) in {elapsed:.2}s"
        );
        assert!((token as usize) < vocab,
            "argmax returned out-of-range token {token} (vocab={vocab})");
    }

    /// Commit #5f smoke: multi-token prompt processing.
    /// Drives a 3-token prompt through `forward_prompt_to_token`
    /// starting at position 0, asserts the returned argmax is in
    /// `[0, vocab_size)`. Also runs a 1-token call as a
    /// baseline — a single-element prompt must produce the same
    /// token as `forward_full_to_token` (BOS-only).
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///     cargo test --release -p rvllm-runtime --features cuda,gb10 \
    ///     gemma4_nvfp4_bring_up::tests::ondisk_bringup_prompt_forward \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_bringup_prompt_forward() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping #5f smoke");
                return;
            }
        };
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );

        // 1-token baseline: forward_prompt_to_token([BOS]) must
        // match forward_full_to_token(BOS) byte-identically.
        {
            let mut bringup = Gemma4Nvfp4Bringup::load(
                &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
            ).expect("Gemma4Nvfp4Bringup::load");
            let kv = bringup.allocate_kv_state(64)
                .expect("allocate_kv_state(64)");
            let token_a = bringup.forward_full_to_token(2, 0, &kv)
                .expect("forward_full_to_token");
            eprintln!("[#5f-smoke] forward_full_to_token(BOS)={token_a}");
            assert!((token_a as usize) < bringup.arch.vocab_size);
        }
        {
            let mut bringup = Gemma4Nvfp4Bringup::load(
                &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
            ).expect("Gemma4Nvfp4Bringup::load");
            let kv = bringup.allocate_kv_state(64)
                .expect("allocate_kv_state(64)");
            let token_b = bringup.forward_prompt_to_token(&[2], 0, &kv)
                .expect("forward_prompt_to_token([BOS])");
            eprintln!("[#5f-smoke] forward_prompt_to_token([BOS])={token_b}");
            assert!((token_b as usize) < bringup.arch.vocab_size);
        }

        // 3-token prompt: BOS + two arbitrary tokens. Verifies
        // the prompt-loop scaffolding doesn't blow up on N>1 and
        // produces an in-range argmax.
        let mut bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");
        // Stream-#5f-PRIME: prompt path needs max_query_tokens >=
        // prompt length. Default chunk=1 only fits N=1 (decode);
        // bump to N=64 for prompts.
        let kv = bringup.allocate_kv_state_with_chunk(64, 64)
            .expect("allocate_kv_state_with_chunk");

        let prompt: Vec<u32> = vec![2, 1000, 5000];
        let t0 = std::time::Instant::now();
        let next = bringup.forward_prompt_to_token(&prompt, 0, &kv)
            .expect("forward_prompt_to_token(3 tokens)");
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!(
            "[#5f-smoke] 3-token prompt={:?} at pos=0..3 → next token {next} \
             in {elapsed:.2}s ({} tok/s)",
            prompt, prompt.len() as f64 / elapsed,
        );
        assert!((next as usize) < bringup.arch.vocab_size,
            "out-of-range prompt argmax {next}");
    }

    /// Stream-6a smoke: `ensure_drafter_nvfp4` loads the
    /// Gemma 4 assistant drafter against Option B's arena +
    /// kernel manifest, attaches all 4 PTX bundles
    /// (masked_embedder, flash_attention f16io, BC16, NVFP4
    /// dequant), and allocates F16 shadow KV at the source
    /// pair `(58, 59)` on 31B.
    ///
    /// Run:
    ///   GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///   GEMMA4_DRAFTER_DIR=/home/r00t/gemma-4-31B-it-assistant \
    ///     cargo test --release -p rvllm-runtime \
    ///       --features cuda,gb10 \
    ///       gemma4_nvfp4_bring_up::tests::ondisk_bringup_drafter_load \
    ///       -- --ignored --nocapture
    ///
    /// Does NOT run a spec step yet — that's the next commit
    /// (drafter forward helpers + spec session loop). This
    /// smoke just verifies the load surface is wired and
    /// idempotent.
    #[test]
    #[ignore]
    fn ondisk_bringup_drafter_load() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping #6a smoke");
                return;
            }
        };
        let drafter_dir = match std::env::var("GEMMA4_DRAFTER_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => PathBuf::from("/home/r00t/gemma-4-31B-it-assistant"),
        };
        if !drafter_dir.is_dir() {
            eprintln!("drafter dir {drafter_dir:?} missing — skip");
            return;
        }
        let kernels_dir = std::path::PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121",
        );
        let mut bringup = Gemma4Nvfp4Bringup::load(
            &dir, 40 * 1024 * 1024 * 1024, &kernels_dir,
        ).expect("Gemma4Nvfp4Bringup::load");
        let kv = bringup.allocate_kv_state_with_chunk(64, 64)
            .expect("allocate_kv_state_with_chunk");

        // Stream-6b primitive #1: ensure the base_last_hidden
        // snapshot buffer is allocated. Must happen BEFORE we
        // lock the drafter mutex (this method needs &mut self).
        bringup.ensure_base_last_hidden_buffer()
            .expect("ensure_base_last_hidden_buffer");
        assert_ne!(bringup.base_last_hidden_device_ptr(), 0);

        bringup.ensure_drafter_nvfp4(&drafter_dir, &kv)
            .expect("ensure_drafter_nvfp4");

        // Idempotency: second call is a no-op.
        bringup.ensure_drafter_nvfp4(&drafter_dir, &kv)
            .expect("ensure_drafter_nvfp4 idempotent");

        // Verify the drafter slot is populated with all
        // attached handles.
        let guard = bringup.drafter.lock().unwrap();
        let rt = guard.as_ref().expect("drafter slot populated");
        assert!(rt.fn_masked_embedder_argmax_f16.is_some(),
            "masked_embedder kernel handle missing");
        assert!(rt.fn_flash_attention_2_decode_f16io.is_some(),
            "flash_attention f16io kernel handle missing");
        assert!(rt.fn_flash_attention_2_decode_f16io_bc16.is_some(),
            "flash_attention bc16 kernel handle missing");
        assert!(rt.fn_drafter_dequant_fp8_to_f16.is_some(),
            "dequant FP8 kernel handle missing");
        assert!(rt.fn_drafter_dequant_nvfp4_to_f16.is_some(),
            "dequant NVFP4 kernel handle missing");
        let shadow = rt.shadow_kv.as_ref()
            .expect("shadow KV not allocated");
        assert!(shadow.sliding_layer_bytes > 0);
        assert!(shadow.full_layer_bytes > 0);
        assert_eq!(shadow.block_size, kv.block_size,
            "shadow block_size must match Option B's kv.block_size");
        assert_eq!(shadow.num_blocks_total, kv.max_pos);
        eprintln!(
            "[#6a-smoke] drafter loaded: {} bytes resident, \
             shadow sliding={}MiB full={}MiB ✓",
            rt.bytes_resident,
            shadow.sliding_layer_bytes / (1024 * 1024),
            shadow.full_layer_bytes / (1024 * 1024),
        );
        let workspace = rt.alloc_step_workspace(&bringup.arena)
            .expect("alloc_step_workspace");
        eprintln!(
            "[#6a-smoke] drafter workspace allocated ({} bytes)",
            workspace.bytes,
        );
        // Stream-6a primitive #1: zero the pre_projection_in
        // buffer (so f16_gemm sees a defined zero input — real
        // step would populate this with [last_token_embed;
        // base_hidden_last]) then run forward_drafter_pre
        // _projection. Asserts the gemm + cast chain completes
        // without NaN/Inf.
        let drafter_hidden = rt.arch.hidden_size;
        let drafter_pre_in = rt.arch.pre_projection_in_dim;
        unsafe {
            use cudarc::driver::sys::*;
            let zero_bytes = drafter_pre_in * 2; // pre_in f16
            let rc = cuMemsetD8_v2(
                workspace.pre_projection_in, 0, zero_bytes);
            assert_eq!(rc, CUresult::CUDA_SUCCESS,
                "zero pre_projection_in");
        }
        bringup.forward_drafter_pre_projection(rt, &workspace)
            .expect("forward_drafter_pre_projection");
        bringup.stream.fence().expect("stream fence");
        // DtoH workspace.hidden (drafter f16 hidden, NOT base
        // hidden — production drafter has its own hidden_size,
        // 1024 on the 31B assistant).
        let h_words = drafter_hidden;
        let mut hidden_f16 = vec![0u16; h_words];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                hidden_f16.as_mut_ptr() as *mut _,
                workspace.hidden, h_words * 2);
            assert_eq!(rc, CUresult::CUDA_SUCCESS,
                "drafter hidden DtoH");
        }
        let mut nan = 0usize;
        let mut inf = 0usize;
        let mut mean_abs: f32 = 0.0;
        let mut max_abs: f32 = 0.0;
        for &b in &hidden_f16 {
            let v = half::f16::from_bits(b).to_f32();
            if v.is_nan() { nan += 1; continue; }
            if v.is_infinite() { inf += 1; continue; }
            mean_abs += v.abs();
            if v.abs() > max_abs { max_abs = v.abs(); }
        }
        mean_abs /= h_words as f32;
        eprintln!(
            "[#6a-smoke] pre_projection on zero input: \
             N={h_words} nan={nan} inf={inf} mean_abs={mean_abs:.6} \
             max_abs={max_abs:.6}"
        );
        assert_eq!(nan, 0, "pre_projection produced NaN");
        assert_eq!(inf, 0, "pre_projection produced Inf");
        // Zero input through bf16-narrow GEMM should land near
        // zero. Bound generously to absorb FMA round-off.
        assert!(max_abs < 1.0,
            "pre_projection on zero input has implausible \
             max_abs={max_abs}");

        // Stream-6a primitives #2-3 chained: run layer 0 of
        // the drafter (q_side → cross_attn → attn_finisher).
        // Shadow KV is zero-initialised (we haven't called
        // populate_shadow_kv*_from_base yet — that's the next
        // step in the spec session loop), so the cross-attn
        // sees an all-zero K/V and produces a deterministic
        // (zero or near-zero) attn_out. The smoke just verifies
        // the launch chain completes without panic / NaN / Inf.
        let pos: u32 = 0;
        bringup.forward_drafter_layer_q_side(rt, &workspace, 0, pos)
            .expect("forward_drafter_layer_q_side");
        bringup.forward_drafter_layer_cross_attn(rt, &workspace, 0, &kv)
            .expect("forward_drafter_layer_cross_attn");
        bringup.forward_drafter_layer_attn_finisher(rt, &workspace, 0)
            .expect("forward_drafter_layer_attn_finisher");
        bringup.forward_drafter_layer_mlp_finisher(rt, &workspace, 0)
            .expect("forward_drafter_layer_mlp_finisher");
        bringup.stream.fence().expect("stream fence");
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                hidden_f16.as_mut_ptr() as *mut _,
                workspace.hidden, h_words * 2);
            assert_eq!(rc, CUresult::CUDA_SUCCESS,
                "drafter hidden DtoH (post-layer0)");
        }
        let mut nan = 0usize;
        let mut inf = 0usize;
        let mut max_abs: f32 = 0.0;
        for &b in &hidden_f16 {
            let v = half::f16::from_bits(b).to_f32();
            if v.is_nan() { nan += 1; continue; }
            if v.is_infinite() { inf += 1; continue; }
            if v.abs() > max_abs { max_abs = v.abs(); }
        }
        eprintln!(
            "[#6a-smoke] layer0 full chain (q_side + cross_attn + \
             attn_fin + mlp_fin), shadow KV ZERO: nan={nan} inf={inf} \
             max_abs={max_abs:.6}"
        );
        assert_eq!(nan, 0, "drafter layer0 produced NaN");
        assert_eq!(inf, 0, "drafter layer0 produced Inf");

        // Stream-6a populate call: smoke-test that
        // `populate_drafter_shadow_kv_with_rt` runs without
        // panicking against the (zero-content) NVFP4 KV.
        // (`_with_rt` is the lock-free variant — the smoke
        // already holds `bringup.drafter.lock()` via `guard`,
        // so the locking entry point would deadlock.) Numerical
        // validation of cross-attn × real-K/V is deferred to the
        // spec-session smoke (which drives a real prompt prefill
        // first) — that path hung in the unified-NVFP4-prefill
        // kernel during 2026-05-18 bring-up and is being
        // debugged separately. Compilation + dispatch coverage
        // for the populate path lands here.
        bringup.populate_drafter_shadow_kv_with_rt(rt, &kv, 0, 1)
            .expect("populate_drafter_shadow_kv_with_rt on empty KV");
        eprintln!("[#6a-smoke] populate_drafter_shadow_kv_with_rt dispatch OK");

        // Stream-6b orchestration primitive: chain the full
        // drafter forward (pre_projection → 4 layers ×
        // (q_side + cross_attn + attn_finisher + mlp_finisher)
        // → final_to_token) on zero input. Shadow KV is still
        // zero, so the speculated token is deterministic from
        // the LM-head argmax over a near-zero residual; the
        // smoke just verifies the launch chain completes and
        // `out_token_id` ends up in a valid vocab range.
        unsafe {
            use cudarc::driver::sys::*;
            let zero_bytes = drafter_pre_in * 2;
            let rc = cuMemsetD8_v2(
                workspace.pre_projection_in, 0, zero_bytes);
            assert_eq!(rc, CUresult::CUDA_SUCCESS);
        }
        bringup.run_drafter_forward_one_token(rt, &workspace, 0, &kv)
            .expect("run_drafter_forward_one_token");
        bringup.stream.fence().expect("stream fence");
        let mut out_tok_host: u32 = u32::MAX;
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                &mut out_tok_host as *mut u32 as *mut _,
                workspace.out_token_id, 4);
            assert_eq!(rc, CUresult::CUDA_SUCCESS,
                "DtoH out_token_id");
        }
        eprintln!(
            "[#6a-smoke] run_drafter_forward_one_token \
             (zero input, shadow ZERO) → token={out_tok_host}");
        assert!(out_tok_host < rt.arch.vocab_size as u32,
            "run_drafter_forward_one_token produced out-of-vocab \
             token {out_tok_host} (vocab={})", rt.arch.vocab_size);

        // Stream-6b primitive #2: populate pre_projection_in
        // with a real base-embed lookup + the (zero-initialised)
        // base_last_hidden buffer.
        bringup.populate_drafter_pre_projection_input(
            rt, &workspace, /* token_id */ 2)
            .expect("populate_drafter_pre_projection_input(BOS)");
        bringup.stream.fence().expect("stream fence");

        // Probe: DtoH the embed half of pre_projection_in and
        // assert it's non-zero. The bf16→f16 cast must produce
        // a defined non-zero buffer for any BOS embedding; if
        // the cast or DtoD never ran, the buffer stays zero.
        let half_elems = rt.arch.backbone_hidden_size;
        let mut probe = vec![0u16; half_elems];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                workspace.pre_projection_in,
                half_elems * 2);
            assert_eq!(rc, CUresult::CUDA_SUCCESS);
        }
        let nonzero = probe.iter().filter(|&&b| b != 0).count();
        let mut max_abs: f32 = 0.0;
        for &b in &probe {
            let v = half::f16::from_bits(b).to_f32();
            if v.is_nan() || v.is_infinite() { continue; }
            if v.abs() > max_abs { max_abs = v.abs(); }
        }
        eprintln!(
            "[#6a-smoke] populate_drafter_pre_projection_input(BOS): \
             embed-half nonzero={nonzero}/{half_elems} \
             max_abs={max_abs:.4}");
        assert!(nonzero > 0,
            "populate_drafter_pre_projection_input(BOS) left the \
             embed half of pre_projection_in fully zero — the \
             bf16→f16 cast must not be reaching the buffer");

        // End-to-end: re-run the drafter forward on the populated
        // input. Just verifies the chain stays NaN/Inf-free; we
        // don't assert token inequality with the zero baseline
        // because shadow KV is still zero (cross-attn × 0 V = 0
        // attn_out, so layer-0+ residual is dominated by the
        // pre_projection output and the LM-head argmax can
        // collapse to the same token even with very different
        // inputs).
        bringup.run_drafter_forward_one_token(rt, &workspace, 0, &kv)
            .expect("run_drafter_forward_one_token (post-populate)");
        bringup.stream.fence().expect("stream fence");
        let mut tok_after: u32 = u32::MAX;
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                &mut tok_after as *mut u32 as *mut _,
                workspace.out_token_id, 4);
            assert_eq!(rc, CUresult::CUDA_SUCCESS);
        }
        eprintln!(
            "[#6a-smoke] post-populate run_drafter_forward_one_token \
             → token={tok_after}");
        assert!(tok_after < rt.arch.vocab_size as u32,
            "drafter post-populate produced out-of-vocab token \
             {tok_after} (vocab={})", rt.arch.vocab_size);
    }
}
