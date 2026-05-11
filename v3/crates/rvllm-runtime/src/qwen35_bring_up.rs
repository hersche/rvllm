//! Qwen 3.5 27B dense — Phase 0 bring-up scaffolding.
//!
//! See `v3/QWEN35_BRINGUP_PLAN.md` for the phase plan. Today the
//! engine only:
//!
//!  1. Parses + validates the config.json via `Qwen35Arch::from_dir`.
//!  2. Logs the arch summary.
//!  3. Returns a `NotImplemented` error from every generate-path
//!     entry point so the operator gets a clean startup signal
//!     pointing at the unwired phases.
//!
//! Phases (to be implemented in subsequent sessions):
//!   * Phase 1 — FP8 weight inventory + safetensors loader
//!     (mirrors qwen36_load with no MoE expert tensors, no router).
//!   * Phase 2 — decoder forward (linear + full attn reusing
//!     qwen36's gated_delta_rule + FA2 kernels; dense MLP via
//!     gate/up/down/silu_mul reusing Mistral 3.5 / Gemma 4 kernels).
//!   * Phase 3 — Qwen3-VL ViT (27 blocks, hidden=1152, intermediate
//!     =4304, patch_embed [1152,3,2,16,16], patch merger
//!     [4608→4608→5120]).
//!   * Phase 4 — slot-aware vision splice (port from
//!     `Mistral35Bringup::generate_with_vision_slots`).
//!   * Phase 5 — perf optimisations (FA-decode, batched prefill,
//!     codex round on the new arch).

use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};
#[cfg(feature = "cuda")]
use rvllm_loader::qwen35_weights::Qwen35LoadedModel;
#[cfg(feature = "cuda")]
use rvllm_mem::{context::CudaContextHandle, stream::Stream, HbmArena};

use crate::qwen35_arch::Qwen35Arch;

/// Engine paths — identical shape to the other family bring-ups so
/// `cuda_worker` can spawn either through the same interface.
#[derive(Clone, Debug)]
pub struct Qwen35EnginePaths {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub cutlass_so: PathBuf,
    pub fa3_so: PathBuf,
    pub policy_json: PathBuf,
}

/// Per-full-attn-layer KV cache pair. Linear-attn layers don't
/// carry a positional KV cache (state is SSM-style in
/// `Qwen35LinearState`).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct Qwen35LayerKv {
    pub k_ptr: u64,
    pub v_ptr: u64,
}

/// Sparse BF16 KV cache — only the 16 full-attn layers carry a
/// `Qwen35LayerKv`. `layer_idx_to_full_idx[layer_idx]` returns
/// `Some(full_rank)` for full-attn layers and `None` for linear.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct Qwen35KvCache {
    pub layers: Vec<Qwen35LayerKv>,
    pub layer_idx_to_full_idx: Vec<Option<usize>>,
    pub max_pos: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub per_layer_bytes: usize,
}

/// Per-linear-attn-layer SSM state. Layout: one contiguous region
/// holding all 48 layers' states. Per-layer offset is
/// `linear_attn_rank × per_layer_bytes`. Zero-initialised at
/// bring-up (session reset). Phase 2c forward will update + read
/// this region via the existing `gated_delta_rule_decode_f16`
/// kernel.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct Qwen35LinearState {
    pub base_ptr: u64,
    pub per_layer_bytes: usize,
    pub n_linear_layers: usize,
    pub num_ssm_heads: usize,
    pub d_state: usize,
    pub layer_idx_to_linear_idx: Vec<Option<usize>>,
}

/// Host-built partial-rotary cos/sin tables. Shape
/// `[max_pos, rotary_dim/2]` F16. Indexed by `(pos, i)` where
/// `i < rotary_dim/2`.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct Qwen35RopeTables {
    pub cos_ptr: u64,
    pub sin_ptr: u64,
    pub max_pos: usize,
    pub rotary_dim: usize,
}

/// Per-decode-token forward scratch. Reusable across requests
/// via `arena.checkpoint() / restore()`. Phase 2c will allocate
/// a separate T-batched scratch for prefill.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct Qwen35Scratch {
    pub token_in_ptr: u64,     // i32 [1]
    pub h_residual_ptr: u64,   // F16 [hidden]
    pub h_work_ptr: u64,       // F16 [hidden]
    pub q_out_ptr: u64,        // F16 [n_q_heads * head_dim] (full-attn)
    pub q_gate_ptr: u64,       // F16 [n_q_heads * head_dim]  output gate slice
    pub k_out_ptr: u64,        // F16 [n_kv_heads * head_dim]
    pub v_out_ptr: u64,        // F16 [n_kv_heads * head_dim]
    pub attn_out_ptr: u64,     // F16 [n_q_heads * head_dim]
    pub o_out_ptr: u64,        // F16 [hidden]
    pub gate_out_ptr: u64,     // F16 [intermediate]
    pub up_out_ptr: u64,       // F16 [intermediate]
    pub silu_mid_ptr: u64,     // F16 [intermediate]
    pub down_out_ptr: u64,     // F16 [hidden]
    pub logits_ptr: u64,       // F32 [vocab]
    pub token_out_ptr: u64,    // i32 [1]
    pub scratch_bytes: usize,
}

/// Phase 2b engine handle. Adds scratch / KV cache / RoPE tables /
/// linear-attn state on top of Phase 1b (arch + weights).
pub struct Qwen35Bringup {
    pub paths: Qwen35EnginePaths,
    pub arch: Qwen35Arch,
    pub arena_bytes: usize,
    #[cfg(feature = "cuda")]
    pub ctx: Option<Arc<CudaContextHandle>>,
    #[cfg(feature = "cuda")]
    pub arena: Option<HbmArena<'static>>,
    #[cfg(feature = "cuda")]
    pub stream: Option<Stream>,
    /// Loaded weights (outside tensors + 64 per-layer slots). Access
    /// `.outside.embed_tokens` etc. through `model.as_ref()`.
    #[cfg(feature = "cuda")]
    pub model: Option<Qwen35LoadedModel>,
    #[cfg(feature = "cuda")]
    pub kv_cache: Option<Qwen35KvCache>,
    #[cfg(feature = "cuda")]
    pub linear_state: Option<Qwen35LinearState>,
    #[cfg(feature = "cuda")]
    pub rope_tables: Option<Qwen35RopeTables>,
    #[cfg(feature = "cuda")]
    pub scratch: Option<Qwen35Scratch>,
}

impl std::fmt::Debug for Qwen35Bringup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35Bringup")
            .field("paths", &self.paths)
            .field("arch", &self.arch)
            .field("arena_bytes", &self.arena_bytes)
            .finish_non_exhaustive()
    }
}

impl Qwen35Bringup {
    /// Phase 1a: parse + validate `config.json`, allocate the arena,
    /// upload outside-the-stack tensors. NO per-layer upload, NO
    /// forward path. See QWEN35_BRINGUP_PLAN.md.
    pub fn load(paths: Qwen35EnginePaths, arena_bytes: usize) -> Result<Self> {
        let arch = Qwen35Arch::from_dir(&paths.model_dir)?
            .ok_or_else(|| corrupt(
                paths.model_dir.clone(),
                "Qwen35Bringup::load: config.json does not match Qwen 3.5 \
                 dense markers (model_type==qwen3_5, attn_output_gate==true, \
                 num_experts absent/0, intermediate_size present)".into(),
            ))?;
        arch.log_summary();

        #[cfg(feature = "cuda")]
        {
            let ctx = Arc::new(CudaContextHandle::init(0)?);
            #[cfg(feature = "gb10")]
            let arena = {
                let (major, minor) = ctx.compute_capability();
                let target = rvllm_core::CompileTarget::from_compute_capability(
                    major, minor);
                if matches!(target, Some(rvllm_core::CompileTarget::Sm121)) {
                    rvllm_mem::UnifiedArena::new(&ctx, arena_bytes)?.into_inner()
                } else {
                    HbmArena::new(&ctx, arena_bytes)?
                }
            };
            #[cfg(not(feature = "gb10"))]
            let arena = HbmArena::new(&ctx, arena_bytes)?;
            let arena: HbmArena<'static> = unsafe { std::mem::transmute(arena) };
            let stream = Stream::new(&ctx)?;

            // Phase 1b: full per-layer upload. Reads `layer_types`
            // from the arch (3:1 linear:full pattern in the 27B
            // dense checkpoint) and routes each slot to the matching
            // builder. Dense MLP lands on every layer.
            let model = rvllm_loader::qwen35_load::load_qwen35_model(
                &paths.model_dir, &arena, &arch.base.layer_types,
            )?;

            // ── Phase 2b: scratch + KV + RoPE + linear-attn state ─
            let kv_max_pos: usize = std::env::var("RVLLM_QWEN35_KV_MAX_POS")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or(8192);

            // RoPE tables: partial rotary (rotary_dim = 64 of head_dim
            // = 256 for the 27B dense), [max_pos, rotary_dim/2] f16.
            let head_dim = arch.base.head_dim;
            let rotary_dim = (head_dim as f32 * arch.rope_partial_rotary_factor) as usize;
            let rope_tables = build_rope_tables(
                &arena, kv_max_pos, head_dim, rotary_dim,
                arch.rope_theta,
            )?;

            // KV cache: BF16 [max_pos, n_kv_heads, head_dim] per
            // FULL-attn layer.
            let n_kv_heads = arch.base.num_key_value_heads;
            let per_layer_kv_bytes = kv_max_pos * n_kv_heads * head_dim * 2;
            let mut kv_layers: Vec<Qwen35LayerKv> = Vec::new();
            let mut layer_idx_to_full_idx: Vec<Option<usize>> =
                vec![None; arch.base.num_hidden_layers];
            let mut next_full: usize = 0;
            for (li, ty) in arch.base.layer_types.iter().enumerate() {
                if matches!(ty, rvllm_loader::LayerAttnType::Full) {
                    let k_region = arena.region(
                        "qwen35_k_cache", per_layer_kv_bytes, 16)?;
                    let v_region = arena.region(
                        "qwen35_v_cache", per_layer_kv_bytes, 16)?;
                    kv_layers.push(Qwen35LayerKv {
                        k_ptr: k_region.device_ptr(),
                        v_ptr: v_region.device_ptr(),
                    });
                    layer_idx_to_full_idx[li] = Some(next_full);
                    next_full += 1;
                }
            }
            let kv_cache = Qwen35KvCache {
                layers: kv_layers,
                layer_idx_to_full_idx,
                max_pos: kv_max_pos,
                n_kv_heads,
                head_dim,
                per_layer_bytes: per_layer_kv_bytes,
            };
            eprintln!(
                "[qwen35] KV cache: {} full-attn layers × \
                 (k+v) {} MiB/layer = {:.2} GiB total \
                 (max_pos={}, n_kv_heads={}, head_dim={})",
                kv_cache.layers.len(),
                per_layer_kv_bytes / (1024 * 1024),
                (kv_cache.layers.len() * per_layer_kv_bytes * 2) as f64
                    / (1024.0 * 1024.0 * 1024.0),
                kv_max_pos, n_kv_heads, head_dim,
            );

            // Linear-attn SSM state: 48 layers × num_value_heads
            // × value_head_dim × value_head_dim × 2 bytes (f16).
            // Read from text_config; defaults match Qwen 3.5 27B.
            let (num_ssm_heads, d_state) = qwen35_linear_dims(&paths.model_dir);
            let per_layer_state_bytes = num_ssm_heads * d_state * d_state * 2;
            let mut layer_idx_to_linear_idx: Vec<Option<usize>> =
                vec![None; arch.base.num_hidden_layers];
            let mut n_linear_layers = 0usize;
            for (li, ty) in arch.base.layer_types.iter().enumerate() {
                if matches!(ty, rvllm_loader::LayerAttnType::Linear) {
                    layer_idx_to_linear_idx[li] = Some(n_linear_layers);
                    n_linear_layers += 1;
                }
            }
            let total_linear_state_bytes = n_linear_layers * per_layer_state_bytes;
            let linear_state_region = arena.region(
                "qwen35_linear_state", total_linear_state_bytes, 16)?;
            zero_region(linear_state_region.device_ptr(), total_linear_state_bytes)?;
            let linear_state = Qwen35LinearState {
                base_ptr: linear_state_region.device_ptr(),
                per_layer_bytes: per_layer_state_bytes,
                n_linear_layers,
                num_ssm_heads,
                d_state,
                layer_idx_to_linear_idx,
            };
            eprintln!(
                "[qwen35] linear-attn state: {} layers × \
                 {} MiB/layer = {:.2} GiB total \
                 (num_ssm_heads={}, d_state={}, zero-init)",
                n_linear_layers,
                per_layer_state_bytes / (1024 * 1024),
                total_linear_state_bytes as f64
                    / (1024.0 * 1024.0 * 1024.0),
                num_ssm_heads, d_state,
            );

            // Per-decode-token scratch. All buffers F16 except logits
            // (F32) and token_in/out (i32). Sized for M=1 decode.
            let hidden = arch.base.hidden_size;
            let intermediate = arch.base.intermediate_size;
            let n_q_heads = arch.base.num_attention_heads;
            let vocab = arch.base.vocab_size;
            let f16_bytes = |n: usize| n * 2;
            let scratch = Qwen35Scratch {
                token_in_ptr: arena.region("qwen35_token_in", 4, 4)?.device_ptr(),
                h_residual_ptr: arena.region(
                    "qwen35_h_residual", f16_bytes(hidden), 16)?.device_ptr(),
                h_work_ptr: arena.region(
                    "qwen35_h_work", f16_bytes(hidden), 16)?.device_ptr(),
                q_out_ptr: arena.region(
                    "qwen35_q_out", f16_bytes(n_q_heads * head_dim), 16)?.device_ptr(),
                q_gate_ptr: arena.region(
                    "qwen35_q_gate", f16_bytes(n_q_heads * head_dim), 16)?.device_ptr(),
                k_out_ptr: arena.region(
                    "qwen35_k_out", f16_bytes(n_kv_heads * head_dim), 16)?.device_ptr(),
                v_out_ptr: arena.region(
                    "qwen35_v_out", f16_bytes(n_kv_heads * head_dim), 16)?.device_ptr(),
                attn_out_ptr: arena.region(
                    "qwen35_attn_out", f16_bytes(n_q_heads * head_dim), 16)?.device_ptr(),
                o_out_ptr: arena.region(
                    "qwen35_o_out", f16_bytes(hidden), 16)?.device_ptr(),
                gate_out_ptr: arena.region(
                    "qwen35_gate_out", f16_bytes(intermediate), 16)?.device_ptr(),
                up_out_ptr: arena.region(
                    "qwen35_up_out", f16_bytes(intermediate), 16)?.device_ptr(),
                silu_mid_ptr: arena.region(
                    "qwen35_silu_mid", f16_bytes(intermediate), 16)?.device_ptr(),
                down_out_ptr: arena.region(
                    "qwen35_down_out", f16_bytes(hidden), 16)?.device_ptr(),
                logits_ptr: arena.region(
                    "qwen35_logits", vocab * 4, 16)?.device_ptr(),
                token_out_ptr: arena.region("qwen35_token_out", 4, 4)?.device_ptr(),
                scratch_bytes:
                    f16_bytes(hidden) * 4 +  // h_residual, h_work, o_out, down_out
                    f16_bytes(n_q_heads * head_dim) * 3 +  // q_out, q_gate, attn_out
                    f16_bytes(n_kv_heads * head_dim) * 2 +  // k_out, v_out
                    f16_bytes(intermediate) * 3 +  // gate, up, silu
                    vocab * 4 + 8,  // logits + tokens
            };
            eprintln!(
                "[qwen35] decode scratch allocated: {:.2} MiB \
                 (M=1, hidden={}, intermediate={}, vocab={})",
                scratch.scratch_bytes as f64 / (1024.0 * 1024.0),
                hidden, intermediate, vocab,
            );

            eprintln!(
                "[qwen35] Phase 2b complete: arch + outside + {} layers \
                 + KV cache + linear-attn state + RoPE tables + decode \
                 scratch. arena.used={:.2} GiB. Forward path still \
                 pending (Phase 2c). See v3/QWEN35_BRINGUP_PLAN.md.",
                arch.base.num_hidden_layers,
                arena.used() as f64 / (1024.0 * 1024.0 * 1024.0),
            );

            return Ok(Self {
                paths, arch, arena_bytes,
                ctx: Some(ctx),
                arena: Some(arena),
                stream: Some(stream),
                model: Some(model),
                kv_cache: Some(kv_cache),
                linear_state: Some(linear_state),
                rope_tables: Some(rope_tables),
                scratch: Some(scratch),
            });
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!(
                "[qwen35] Phase 0 ONLY (no-cuda build): arch validated, \
                 NO weight upload. See v3/QWEN35_BRINGUP_PLAN.md."
            );
            Ok(Self { paths, arch, arena_bytes })
        }
    }
}

#[cfg(feature = "cuda")]
fn build_rope_tables(
    arena: &HbmArena,
    max_pos: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
) -> Result<Qwen35RopeTables> {
    let half = rotary_dim / 2;
    // Proportional-RoPE frequency convention: divisor is head_dim
    // (NOT rotary_dim) — matches Qwen 3.6 and Gemma 4. So even
    // though only the first `rotary_dim` slots of each head are
    // rotated, the frequency spread is over the full head dim.
    let inv_theta: Vec<f32> = (0..half)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let table_elems = max_pos * half;
    let table_bytes = table_elems * 2; // f16
    let mut cos_bytes = Vec::with_capacity(table_bytes);
    let mut sin_bytes = Vec::with_capacity(table_bytes);
    for pos in 0..max_pos {
        for &freq in &inv_theta {
            let angle = pos as f32 * freq;
            let c = half::f16::from_f32(angle.cos()).to_bits();
            let s = half::f16::from_f32(angle.sin()).to_bits();
            cos_bytes.extend_from_slice(&c.to_le_bytes());
            sin_bytes.extend_from_slice(&s.to_le_bytes());
        }
    }
    let cos_region = arena.region("qwen35_rope_cos", table_bytes, 16)?;
    let sin_region = arena.region("qwen35_rope_sin", table_bytes, 16)?;
    unsafe {
        cos_region.copy_from_host(&cos_bytes)?;
        sin_region.copy_from_host(&sin_bytes)?;
    }
    eprintln!(
        "[qwen35] RoPE tables uploaded: max_pos={max_pos} head_dim={head_dim} \
         rotary_dim={rotary_dim} theta={rope_theta:.1e} ({} KiB cos + {} KiB sin)",
        table_bytes / 1024,
        table_bytes / 1024,
    );
    Ok(Qwen35RopeTables {
        cos_ptr: cos_region.device_ptr(),
        sin_ptr: sin_region.device_ptr(),
        max_pos,
        rotary_dim,
    })
}

/// Read the linear-attn SSM head config from `text_config`.
/// Qwen 3.5 27B dense ships `linear_num_value_heads=48,
/// linear_value_head_dim=128`. Falls back to Qwen 3.6's defaults
/// (32, 128) if the keys are missing — this is the right move for
/// non-canonical checkpoints; production-style configs always have
/// them.
#[cfg(feature = "cuda")]
fn qwen35_linear_dims(model_dir: &Path) -> (usize, usize) {
    let p = model_dir.join("config.json");
    let bytes = match std::fs::read(&p) {
        Ok(b) => b,
        Err(_) => return (32, 128),
    };
    let v: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => return (32, 128),
    };
    let tc = if v["text_config"]["hidden_size"].is_u64() {
        &v["text_config"]
    } else { &v };
    let nvh = tc["linear_num_value_heads"].as_u64().unwrap_or(32) as usize;
    let vhd = tc["linear_value_head_dim"].as_u64().unwrap_or(128) as usize;
    (nvh, vhd)
}

#[cfg(feature = "cuda")]
fn zero_region(dev_ptr: u64, nbytes: usize) -> Result<()> {
    use cudarc::driver::sys::*;
    let rc = unsafe { cuMemsetD8_v2(dev_ptr, 0, nbytes) };
    if rc != CUresult::CUDA_SUCCESS {
        return Err(rvllm_core::RvllmError::cuda(
            "qwen35_zero_region",
            rvllm_core::CudaErrorKind::MemcpyFailed,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    Ok(())
}


#[derive(Debug)]
pub enum Qwen35Error {
    /// Generation called before the GPU forward path is wired.
    ForwardNotImplemented,
    /// Weight loader not yet implemented.
    LoaderNotImplemented,
    /// Vision tower not yet implemented.
    VisionNotImplemented,
}

impl std::fmt::Display for Qwen35Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForwardNotImplemented => write!(
                f, "qwen35: forward path not wired yet (Phase 0 only — \
                    see QWEN35_BRINGUP_PLAN.md)"),
            Self::LoaderNotImplemented => write!(
                f, "qwen35: weight loader not wired yet (Phase 1 pending)"),
            Self::VisionNotImplemented => write!(
                f, "qwen35: vision tower not wired yet (Phase 3 pending)"),
        }
    }
}

impl std::error::Error for Qwen35Error {}

fn corrupt(path: PathBuf, detail: String) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt { detail },
        ctx: LoaderCtx { path, tensor: None },
        bt: std::backtrace::Backtrace::capture(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tempdir() -> PathBuf {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "rvllm-qwen35-{}-{}",
            std::process::id(),
            N.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_config(dir: &Path, body: &str) {
        let mut f = std::fs::File::create(dir.join("config.json")).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    /// Minimal qwen3_5 config sufficient for the arch parser. Layer
    /// pattern is the 3:1 linear:full mix the 27B-dense uses.
    fn qwen35_full() -> &'static str {
        r#"{
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "model_type": "qwen3_5",
          "image_token_id": 248056,
          "vision_config": {
            "hidden_size": 1152, "depth": 27, "out_hidden_size": 5120
          },
          "text_config": {
            "num_hidden_layers": 8, "hidden_size": 5120,
            "intermediate_size": 17408, "num_attention_heads": 24,
            "num_key_value_heads": 4, "head_dim": 256,
            "vocab_size": 248320, "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-5, "hidden_act": "silu",
            "tie_word_embeddings": false, "attn_output_gate": true,
            "rope_parameters": {"rope_theta": 5000000.0,
                                  "partial_rotary_factor": 0.25,
                                  "mrope_section": [11, 11, 10]},
            "layer_types": [
              "linear_attention","linear_attention","linear_attention","full_attention",
              "linear_attention","linear_attention","linear_attention","full_attention"
            ]
          }
        }"#
    }
    // Note: arch_parses_rope_fields expects rope_theta=5e6 because
    // the qwen35_full() fixture uses that value (vs the real
    // checkpoint's 1e7) for compactness.

    #[test]
    fn arch_parses_dense_qwen35() {
        let tmp = tempdir();
        write_config(&tmp, qwen35_full());
        let arch = Qwen35Arch::from_dir(&tmp).unwrap().expect("dense qwen35");
        assert_eq!(arch.base.num_hidden_layers, 8);
        assert_eq!(arch.n_linear, 6);
        assert_eq!(arch.n_full, 2);
        assert!(arch.attn_output_gate);
        assert_eq!(arch.image_token_id, Some(248056));
        assert_eq!(arch.vision_depth, Some(27));
    }

    #[test]
    fn arch_rejects_qwen36_moe_config() {
        let tmp = tempdir();
        write_config(&tmp, &qwen35_full().replace(
            "\"num_hidden_layers\": 8",
            "\"num_hidden_layers\": 40, \"num_experts\": 256",
        ));
        // num_experts != 0 → Qwen35 probe returns None, leaving room
        // for the Qwen36 probe in the family resolver.
        let arch = Qwen35Arch::from_dir(&tmp).unwrap();
        assert!(arch.is_none());
    }

    #[test]
    fn arch_rejects_mistral_config() {
        let tmp = tempdir();
        write_config(&tmp, r#"{
          "architectures": ["Mistral3ForConditionalGeneration"],
          "model_type": "mistral3",
          "text_config": {"hidden_size": 12288}
        }"#);
        let arch = Qwen35Arch::from_dir(&tmp).unwrap();
        assert!(arch.is_none());
    }

    #[test]
    fn arch_parses_rope_fields() {
        let tmp = tempdir();
        write_config(&tmp, qwen35_full());
        let arch = Qwen35Arch::from_dir(&tmp).unwrap().expect("dense qwen35");
        assert_eq!(arch.rope_partial_rotary_factor, 0.25);
        assert!((arch.rope_theta - 5_000_000.0).abs() < 1e-3);
    }

    #[test]
    fn typed_errors_render_helpful_messages() {
        // Verify the typed error messages still point at the plan
        // doc so operators get clear breadcrumbs.
        let fwd = Qwen35Error::ForwardNotImplemented.to_string();
        assert!(fwd.contains("Phase 0"));
        let ldr = Qwen35Error::LoaderNotImplemented.to_string();
        assert!(ldr.contains("Phase 1"));
        let vis = Qwen35Error::VisionNotImplemented.to_string();
        assert!(vis.contains("Phase 3"));
    }
}
