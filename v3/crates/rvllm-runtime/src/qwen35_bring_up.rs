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

/// Phase 1a engine handle. Holds the validated arch + paths +
/// outside-tensor uploads (embed_tokens, final_norm, lm_head, FP8
/// lm-head mirror). Phase 1b+ will fill the 64 per-layer slots
/// and the vision tower.
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

            eprintln!(
                "[qwen35] Phase 1b complete: arch + outside + all \
                 {} layers uploaded. Forward path still pending \
                 (Phase 2). See v3/QWEN35_BRINGUP_PLAN.md.",
                arch.base.num_hidden_layers,
            );

            return Ok(Self {
                paths, arch, arena_bytes,
                ctx: Some(ctx),
                arena: Some(arena),
                stream: Some(stream),
                model: Some(model),
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
            "rope_parameters": {"rope_theta": 5000000.0},
            "layer_types": [
              "linear_attention","linear_attention","linear_attention","full_attention",
              "linear_attention","linear_attention","linear_attention","full_attention"
            ]
          }
        }"#
    }

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
    fn phase0_load_succeeds_then_generate_path_is_not_implemented() {
        let tmp = tempdir();
        write_config(&tmp, qwen35_full());
        let paths = Qwen35EnginePaths {
            model_dir: tmp,
            kernels_dir: PathBuf::from("/tmp/k"),
            cutlass_so: PathBuf::from("/tmp/c"),
            fa3_so: PathBuf::from("/tmp/f"),
            policy_json: PathBuf::from("/tmp/p"),
        };
        let bringup = Qwen35Bringup::load(paths, 0).expect("Phase 0 load ok");
        assert_eq!(bringup.arch.n_linear, 6);
        // Forward not wired — sanity-check the typed error exists.
        let e = Qwen35Error::ForwardNotImplemented.to_string();
        assert!(e.contains("Phase 0"));
    }
}
