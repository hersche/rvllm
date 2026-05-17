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
    /// `[max_pos]` i32 identity block table.
    pub block_tables_ptr: u64,
    /// `[1]` i32 current context length. Caller writes
    /// `position + 1` before each decode launch.
    pub context_lens_ptr: u64,
    /// `[1]` i32 current position; written before RoPE+KV write.
    pub positions_ptr: u64,
    /// `[1]` i32 slot_mapping[0] = position; tells the
    /// RoPE+KV-write kernel where to write the new K/V slot.
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
    ) -> Result<Self> {
        let block_size: u32 = 1;

        let block_tables_region = arena.region(
            "gemma4_nvfp4_kv_block_tables",
            (max_pos as usize) * 4, 256)?;
        let context_lens_region = arena.region(
            "gemma4_nvfp4_kv_context_lens", 4, 16)?;
        let positions_region = arena.region(
            "gemma4_nvfp4_kv_positions", 4, 16)?;
        let slot_mapping_region = arena.region(
            "gemma4_nvfp4_kv_slot_mapping", 4, 16)?;
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
        let mut total_bytes: u64 = (max_pos as u64) * 4 + 4 * 4; // tables + scalars

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
        let loader = KernelLoader::new(manifest);

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
        };
        let forward_checkpoint = arena.checkpoint();

        Ok(Self {
            arch,
            model,
            mlp_kernels,
            forward_kernels,
            _mlp_w4a16_gemv_mod: mlp_gemv_mod,
            _mlp_w4a16_gate_up_mod: mlp_gate_up_mod,
            _mlp_gelu_tanh_mul_mod: mlp_gelu_mod,
            cublaslt,
            stream,
            arena,
            forward_checkpoint,
            kv_state_allocated: false,
            _ctx: ctx,
        })
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
        let kv = Gemma4Nvfp4KvState::allocate(&self.arena, &self.arch, max_pos)?;
        // Re-anchor scratch rewinds above the KV state.
        self.forward_checkpoint = self.arena.checkpoint();
        self.kv_state_allocated = true;
        Ok(kv)
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
        let cos_ptr = self.model.outside.rope_cos_sliding.offset_bytes;
        let sin_ptr = self.model.outside.rope_sin_sliding.offset_bytes;

        // Launch rope_split_half_bf16 per Q then per K.
        //   grid = (n_heads, 1, 1)
        //   block = (head_dim / 2, 1, 1)
        let launch_rope = |qk_ptr: u64, n_heads: u32| -> Result<()> {
            let mut qk = qk_ptr;
            let mut cos = cos_ptr;
            let mut sin = sin_ptr;
            let mut hd = head_dim as i32;
            let mut pos = position as i32;
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

        // ---- 4. Per-position f16 cos/sin mini-tables + state writes ------
        // Sliding-layer 0 uses FULL RoPE: rotary_dim = head_dim.
        let rotary_dim = head_dim;
        let half_rotary = rotary_dim / 2;
        let theta = self.arch.rope_theta_sliding as f64;
        let p = position as f64;
        let mut cos_f16: Vec<u16> = Vec::with_capacity(half_rotary);
        let mut sin_f16: Vec<u16> = Vec::with_capacity(half_rotary);
        for i in 0..half_rotary {
            let inv_freq = 1.0 / theta.powf((2 * i) as f64 / rotary_dim as f64);
            let angle = p * inv_freq;
            cos_f16.push(f32_to_f16_bits(angle.cos() as f32));
            sin_f16.push(f32_to_f16_bits(angle.sin() as f32));
        }
        let cos_region = self.arena.region(
            "g4n_attn_cos_f16", half_rotary * 2, 256)?;
        let sin_region = self.arena.region(
            "g4n_attn_sin_f16", half_rotary * 2, 256)?;
        unsafe {
            let cb: &[u8] = std::slice::from_raw_parts(
                cos_f16.as_ptr() as *const u8, cos_f16.len() * 2);
            let sb: &[u8] = std::slice::from_raw_parts(
                sin_f16.as_ptr() as *const u8, sin_f16.len() * 2);
            cos_region.copy_from_host(cb)?;
            sin_region.copy_from_host(sb)?;
        }

        // Write i32 scalars into the persistent KV state buffers.
        //   positions[0] = 0    (mini-table is 1 row at index 0)
        //   slot_mapping[0] = position  (actual cache slot to write)
        //   context_lens[0] = position + 1  (decoder reads this)
        let write_i32_devptr = |dst: u64, val: i32| -> Result<()> {
            let bytes = val.to_le_bytes();
            unsafe {
                use cudarc::driver::sys::*;
                let rc = cuMemcpyHtoD_v2(dst, bytes.as_ptr() as *const _, 4);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(RvllmError::cuda(
                        "forward_layer0_attn: kv state HtoD",
                        CudaErrorKind::MemcpyFailed,
                        CudaCtx::setup()));
                }
            }
            Ok(())
        };
        write_i32_devptr(kv.positions_ptr,     0)?;
        write_i32_devptr(kv.slot_mapping_ptr,  position as i32)?;
        write_i32_devptr(kv.context_lens_ptr,  (position as i32) + 1)?;

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
            let mut cos_ptr: u64 = cos_region.device_ptr();
            let mut sin_ptr: u64 = sin_region.device_ptr();
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
                (&mut cos_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut sin_ptr) as *mut u64 as *mut core::ffi::c_void,
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

        // Per-layer-type config.
        let (head_dim, rotary_dim, theta, window_size_left) = if is_global {
            (
                self.arch.head_dim_global,
                self.arch.rotary_dim_global(),
                self.arch.rope_theta_global as f64,
                -1i32, // full attention — no sliding window
            )
        } else {
            (
                self.arch.head_dim_sliding,
                self.arch.head_dim_sliding, // sliding = full RoPE
                self.arch.rope_theta_sliding as f64,
                (self.arch.sliding_window_size as i32) - 1,
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
        self.stream.fence()?;

        // DtoH the three projections so we can apply host-side Q/K-norm
        // (existing pattern from forward_layer0_qk_norm — GPU qk_norm
        // bf16-in-place launch happens after we re-upload Q/K as bf16).
        let mut q_f32 = vec![0f32; n_q as usize];
        let mut k_f32 = vec![0f32; n_kv as usize];
        let mut v_f32 = vec![0f32; n_v as usize];
        unsafe {
            use cudarc::driver::sys::*;
            for (host, dev, sz) in [
                (q_f32.as_mut_ptr() as *mut _, q_f32_region.device_ptr(), (n_q as usize) * 4),
                (k_f32.as_mut_ptr() as *mut _, k_f32_region.device_ptr(), (n_kv as usize) * 4),
                (v_f32.as_mut_ptr() as *mut _, v_f32_region.device_ptr(), (n_v as usize) * 4),
            ] {
                let rc = cuMemcpyDtoH_v2(host, dev, sz);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(RvllmError::cuda(
                        "forward_layer_attn_from_residual: qkv DtoH",
                        CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
                }
            }
        }

        // -- Q-norm + K-norm via fresh bf16 buffers ----------------------
        let f32_to_bf16 = |xs: &[f32]| -> Vec<u16> {
            xs.iter().map(|&x| {
                let bits = x.to_bits();
                let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
                (rounded >> 16) as u16
            }).collect()
        };
        let q_bf16 = f32_to_bf16(&q_f32);
        let k_bf16 = f32_to_bf16(&k_f32);
        let q_region = self.arena.region(
            "g4n_lN_q_bf16", q_bf16.len() * 2, 256)?;
        let k_region = self.arena.region(
            "g4n_lN_k_bf16", k_bf16.len() * 2, 256)?;
        unsafe {
            let qb: &[u8] = std::slice::from_raw_parts(
                q_bf16.as_ptr() as *const u8, q_bf16.len() * 2);
            let kb: &[u8] = std::slice::from_raw_parts(
                k_bf16.as_ptr() as *const u8, k_bf16.len() * 2);
            q_region.copy_from_host(qb)?;
            k_region.copy_from_host(kb)?;
        }
        unsafe {
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

        // -- V-RMSNorm on host (parameter-free, per-head) ----------------
        let eps = self.arch.rms_norm_eps;
        let mut v_normed_f32: Vec<f32> = Vec::with_capacity(v_f32.len());
        for h in 0..num_kv_heads {
            let row = &v_f32[h * head_dim .. (h + 1) * head_dim];
            let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>()
                / (head_dim as f32);
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for &x in row { v_normed_f32.push(x * scale); }
        }
        let v_bf16 = f32_to_bf16(&v_normed_f32);
        let v_region = self.arena.region(
            "g4n_lN_v_bf16", v_bf16.len() * 2, 256)?;
        unsafe {
            let vb: &[u8] = std::slice::from_raw_parts(
                v_bf16.as_ptr() as *const u8, v_bf16.len() * 2);
            v_region.copy_from_host(vb)?;
        }

        // -- q_fp8 scratch + attn_out scratch ----------------------------
        let q_fp8_region = self.arena.region(
            "g4n_lN_q_fp8", q_bf16.len(), 256)?;
        let attn_out_region = self.arena.region(
            "g4n_lN_attn_out_bf16", num_q_heads * head_dim * 2, 256)?;

        // -- Per-position f16 cos/sin mini-tables + i32 state writes -----
        let half_rotary = rotary_dim / 2;
        let p = position as f64;
        let mut cos_f16: Vec<u16> = Vec::with_capacity(half_rotary);
        let mut sin_f16: Vec<u16> = Vec::with_capacity(half_rotary);
        for i in 0..half_rotary {
            let inv_freq = 1.0 / theta.powf((2 * i) as f64 / rotary_dim as f64);
            let angle = p * inv_freq;
            cos_f16.push(f32_to_f16_bits(angle.cos() as f32));
            sin_f16.push(f32_to_f16_bits(angle.sin() as f32));
        }
        let cos_region = self.arena.region(
            "g4n_lN_cos_f16", half_rotary * 2, 256)?;
        let sin_region = self.arena.region(
            "g4n_lN_sin_f16", half_rotary * 2, 256)?;
        unsafe {
            let cb: &[u8] = std::slice::from_raw_parts(
                cos_f16.as_ptr() as *const u8, cos_f16.len() * 2);
            let sb: &[u8] = std::slice::from_raw_parts(
                sin_f16.as_ptr() as *const u8, sin_f16.len() * 2);
            cos_region.copy_from_host(cb)?;
            sin_region.copy_from_host(sb)?;
        }
        let write_i32_devptr = |dst: u64, val: i32| -> Result<()> {
            let bytes = val.to_le_bytes();
            unsafe {
                use cudarc::driver::sys::*;
                let rc = cuMemcpyHtoD_v2(dst, bytes.as_ptr() as *const _, 4);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(RvllmError::cuda(
                        "forward_layer_attn_from_residual: kv state HtoD",
                        CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
                }
            }
            Ok(())
        };
        write_i32_devptr(kv.positions_ptr,    0)?;
        write_i32_devptr(kv.slot_mapping_ptr, position as i32)?;
        write_i32_devptr(kv.context_lens_ptr, (position as i32) + 1)?;

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
            let mut cos_ptr: u64 = cos_region.device_ptr();
            let mut sin_ptr: u64 = sin_region.device_ptr();
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
                (&mut cos_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut sin_ptr) as *mut u64 as *mut core::ffi::c_void,
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
        let tok_region = self.arena.region("g4n_embed_tok", 4, 16)?;
        unsafe { tok_region.copy_from_host(&(token_id as i32).to_le_bytes())?; }
        let h_region = self.arena.region(
            "g4n_embed_residual_bf16", (hidden as usize) * 2, 256)?;
        let stream_u64 = self.stream.raw();
        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1, hidden,
                vocab: self.arch.vocab_size as u32,
            }
            .launch(
                self.forward_kernels.fn_embedding_gather_bf16,
                h_region.device_ptr(),
                self.model.outside.embed_tokens.offset_bytes,
                tok_region.device_ptr(), stream_u64,
            )?;
        }
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

        // o_proj: bf16 attn_out @ bf16 o_proj^T → f32 hidden.
        unsafe {
            gemma4_nvfp4_attn_proj(
                &self.cublaslt,
                attn_in_region.device_ptr(),
                layer.o_proj.offset_bytes,
                o_f32_region.device_ptr(),
                1, hidden as i32, n_q, stream_u64,
            )?;
        }
        self.stream.fence()?;

        // f32 → bf16 narrow on host.
        let mut o_f32_host: Vec<f32> = vec![0.0; hidden as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                o_f32_host.as_mut_ptr() as *mut _,
                o_f32_region.device_ptr(),
                (hidden as usize) * 4);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "forward_layer_post_attn: o f32 DtoH",
                    CudaErrorKind::MemcpyFailed, CudaCtx::setup()));
            }
        }
        let o_bf16_host: Vec<u16> = o_f32_host.iter().map(|&x| {
            let bits = x.to_bits();
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            (rounded >> 16) as u16
        }).collect();
        unsafe {
            let b: &[u8] = std::slice::from_raw_parts(
                o_bf16_host.as_ptr() as *const u8, o_bf16_host.len() * 2);
            o_bf16_region.copy_from_host(b)?;
        }

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

    /// Commit #5c: 60-layer driver for a single-token decode at
    /// `position`. Chains:
    ///   embed_tokens[token_id] → for each of `num_hidden_layers`:
    ///     attn_out = forward_layer_attn_from_residual(layer_idx, …)
    ///     residual = forward_layer_post_attn(layer_idx, attn_out, residual)
    ///     residual = forward_layer_post_attn_mlp(layer_idx, residual)
    ///   → forward_final_to_token(residual) → argmax token id
    ///
    /// All inter-layer state lives on the host as bf16 (DtoH ↔
    /// HtoD between layers). This is structurally correct but
    /// inefficient — each layer round-trips the residual through
    /// PCIe. A future "stay-on-device" forward keeps residual +
    /// attn_out in HBM and skips ~120 DtoH/HtoD copies per token.
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

            // Periodic trace (every 10 layers + last).
            if trace_on && (li % 10 == 0
                            || li == self.arch.num_hidden_layers - 1) {
                log_stats(li, "post_mlp", &residual_bf16);
            }
        }

        // Final-norm + tied LM head with optional dump of the
        // post-norm hidden state and the f32 logits before argmax.
        if let Some(d) = dump_dir.as_ref() {
            // We need the post-final-norm bf16 + the f32 logits in
            // SAME buffer addresses the production forward uses, so
            // we have to recreate that op chain here (the
            // `forward_final_to_token` method does it all in one
            // shot; dumping requires teeing intermediates).
            self.forward_final_to_token_with_dump(&residual_bf16, d)
        } else {
            self.forward_final_to_token(&residual_bf16)
        }
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
}
