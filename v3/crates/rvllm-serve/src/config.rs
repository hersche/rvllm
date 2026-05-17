//! Server configuration. Populated from CLI flags + env vars in
//! `main.rs`; plain data beyond that. Kept in a separate module so
//! integration tests can construct configs directly without going
//! through `clap`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// Which model family the worker should bring up.
///
/// `Auto` (default) defers to `crate::family::resolve_model_family`,
/// which inspects the model dir's `config.json`. Explicit values take
/// precedence and require the dir to actually match — mismatch is a
/// startup error rather than silent fall-through. The CLI flag
/// `--model-family` and env var `RVLLM_MODEL_FAMILY` map onto this
/// enum directly (kebab-case lowercase).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ModelFamily {
    #[default]
    Auto,
    Qwen35,
    Qwen36,
    Gemma4,
    /// nvidia/Gemma-4-31B-IT-NVFP4 — same Gemma 4 architecture as
    /// `Gemma4` but with modelopt NVFP4 MLP weights + retained
    /// bf16 attention. Forward path lives behind feature parity
    /// with `Gemma4` and is staged in.
    Gemma4Nvfp4,
    Mistral35,
}

impl ModelFamily {
    /// Parse the CLI / env-var spelling. Accepts the canonical
    /// kebab-case ids plus a couple of obvious aliases. Unknown values
    /// produce a typed error so the operator gets a clear startup
    /// rejection instead of a silent default.
    pub fn parse(s: &str) -> Result<Self, ModelFamilyParseError> {
        let norm = s.trim().to_ascii_lowercase();
        match norm.as_str() {
            "auto" | "" => Ok(ModelFamily::Auto),
            "qwen35" | "qwen-35" | "qwen3.5" | "qwen-3.5" => Ok(ModelFamily::Qwen35),
            "qwen36" | "qwen-36" | "qwen3.6" | "qwen-3.6" => Ok(ModelFamily::Qwen36),
            "gemma4" | "gemma-4" => Ok(ModelFamily::Gemma4),
            "gemma4-nvfp4" | "gemma-4-nvfp4" | "gemma4nvfp4" => {
                Ok(ModelFamily::Gemma4Nvfp4)
            }
            "mistral35" | "mistral-35" | "mistral3.5" | "mistral-3.5" => {
                Ok(ModelFamily::Mistral35)
            }
            other => Err(ModelFamilyParseError(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ModelFamily::Auto => "auto",
            ModelFamily::Qwen35 => "qwen35",
            ModelFamily::Qwen36 => "qwen36",
            ModelFamily::Gemma4 => "gemma4",
            ModelFamily::Gemma4Nvfp4 => "gemma4-nvfp4",
            ModelFamily::Mistral35 => "mistral35",
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("--model-family must be one of auto|qwen35|qwen36|gemma4|gemma4-nvfp4|mistral35 (got: {0:?})")]
pub struct ModelFamilyParseError(pub String);

impl std::str::FromStr for ModelFamily {
    type Err = ModelFamilyParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Top-level server configuration.
///
/// All fields are public so `main.rs` and tests can populate them
/// directly. Invariants are checked in [`ServerConfig::validate`];
/// callers must call it before handing the config to the router.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// HTTP bind address. Default `127.0.0.1:8080`.
    pub bind: SocketAddr,
    /// Directory containing the HF model artefacts (config.json,
    /// tokenizer.json, safetensors shards). Required.
    pub model_dir: PathBuf,
    /// Model name advertised on `/v1/models` and echoed in response
    /// bodies. Defaults to the directory's last component.
    pub model_id: String,
    /// Max number of in-flight + queued generate requests. Full =
    /// admission returns 429. The worker pulls one request out of the
    /// queue and processes it on its dedicated thread; the channel
    /// buffer + admission permits are both sized to `max_queue_depth`,
    /// so depth=1 is strictly serial (1 permit + 1 channel slot,
    /// dequeued one at a time). Validate() only rejects 0.
    pub max_queue_depth: usize,
    /// Hard upper bound on `max_tokens` a request may ask for. Prevents
    /// a single client from pinning the worker.
    pub max_new_tokens_cap: u32,
    /// Per-request hard wall-clock cap. Worker aborts past this.
    pub request_timeout: Duration,
    /// SSE keep-alive interval. Proxies drop idle TCP.
    pub sse_keepalive: Duration,
    /// After the shutdown signal, wait at most this long for
    /// in-flight requests to finish before forcing the server down.
    pub shutdown_drain_timeout: Duration,
    /// Operator-supplied model family. `Auto` defers to
    /// config-marker detection. Anything else asserts the model
    /// matches and fails on mismatch.
    pub model_family: ModelFamily,
    /// Spec-decode commit 1: opt-in greedy speculative decoding via
    /// the Gemma 4 E4B assistant drafter. When enabled, requires
    /// `spec_drafter_dir` to point at a valid
    /// `Gemma4AssistantForCausalLM` checkpoint. Default OFF —
    /// resident-only in commit 1; the gate is parsed but no engine
    /// branches on it yet.
    pub spec_decode: bool,
    /// Drafter checkpoint directory (Gemma 4 E4B assistant).
    /// Required only when `spec_decode == true`. Read from
    /// `RVLLM_GEMMA4_DRAFTER_DIR`.
    pub spec_drafter_dir: PathBuf,
    /// Number of draft tokens K proposed per verification round
    /// (`RVLLM_GEMMA4_SPEC_K`, default 6). Read at config time but
    /// only consumed when `spec_decode == true`.
    pub spec_k: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], 8080)),
            model_dir: PathBuf::new(),
            model_id: String::from("rvllm"),
            max_queue_depth: 8,
            max_new_tokens_cap: 4096,
            request_timeout: Duration::from_secs(300),
            sse_keepalive: Duration::from_secs(15),
            shutdown_drain_timeout: Duration::from_secs(30),
            model_family: ModelFamily::Auto,
            spec_decode: false,
            spec_drafter_dir: PathBuf::new(),
            spec_k: 6,
        }
    }
}

impl ServerConfig {
    /// Check invariants after population. Called once at startup.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model_dir.as_os_str().is_empty() {
            return Err(ConfigError::MissingModelDir);
        }
        if !self.model_dir.is_dir() {
            return Err(ConfigError::ModelDirMissing(self.model_dir.clone()));
        }
        if self.max_queue_depth == 0 {
            return Err(ConfigError::InvalidQueueDepth);
        }
        // Both the cuda_worker and the mock spawn use
        // `mpsc::channel(queue_depth.max(1))` and the same number of
        // admission permits. With depth=1 that's exactly:
        //   1 admission permit + 1 channel slot + serial dequeue
        // = strictly serial, which is what the doc says it is. The
        // earlier ≥2 floor was a leftover from before the
        // admission/channel arithmetic was harmonised.
        if self.max_new_tokens_cap == 0 {
            return Err(ConfigError::InvalidMaxTokens);
        }
        if self.model_id.is_empty() {
            return Err(ConfigError::InvalidModelId);
        }
        // `request_timeout=0` was accepted before — clap parses any
        // u64 from `--request-timeout-secs` /
        // `RVLLM_REQUEST_TIMEOUT_SECS`. The handler uses it to build
        // `tokio::time::Instant::now() + request_timeout` as the
        // per-request deadline, so a 0-duration deadline rejects
        // every preprocess step (or hits an immediate timeout
        // post-submit) and the server is unusable. Refuse at
        // startup with a clear error instead of staying up and
        // 4xx-ing every request.
        if self.request_timeout.is_zero() {
            return Err(ConfigError::InvalidRequestTimeout);
        }
        // Spec-decode commit 1: when the gate is on, the drafter
        // directory must be set + readable. K must be >= 1.
        if self.spec_decode {
            if self.spec_drafter_dir.as_os_str().is_empty() {
                return Err(ConfigError::MissingSpecDrafterDir);
            }
            if !self.spec_drafter_dir.is_dir() {
                return Err(ConfigError::SpecDrafterDirMissing(
                    self.spec_drafter_dir.clone()));
            }
            if self.spec_k == 0 {
                return Err(ConfigError::InvalidSpecK);
            }
            // Codex review item #6: spec_k must fit the pre-allocated
            // K-hidden capture buffer (MAX_SPEC_K rows × hidden f16),
            // otherwise the capture hook overruns device memory.
            const MAX_SPEC_K: u32 =
                rvllm_runtime::gemma4_bring_up::MAX_SPEC_K as u32;
            if self.spec_k > MAX_SPEC_K {
                return Err(ConfigError::SpecKTooLarge {
                    requested: self.spec_k,
                    max: MAX_SPEC_K,
                });
            }
        }
        Ok(())
    }
}

/// Config validation errors. Distinct from [`crate::error::ApiError`]
/// because these fire at startup, not in the request path.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("--model-dir is required")]
    MissingModelDir,
    #[error("model dir does not exist or is not a directory: {0}")]
    ModelDirMissing(PathBuf),
    #[error("--max-queue-depth must be > 0")]
    InvalidQueueDepth,
    #[error("--max-new-tokens-cap must be > 0")]
    InvalidMaxTokens,
    #[error("--model-id must be non-empty")]
    InvalidModelId,
    #[error("--request-timeout-secs must be > 0 (zero deadlines reject every request)")]
    InvalidRequestTimeout,
    #[error("RVLLM_GEMMA4_SPEC_DECODE=1 set but RVLLM_GEMMA4_DRAFTER_DIR is empty")]
    MissingSpecDrafterDir,
    #[error("RVLLM_GEMMA4_DRAFTER_DIR does not exist or is not a directory: {0}")]
    SpecDrafterDirMissing(PathBuf),
    #[error("RVLLM_GEMMA4_SPEC_K must be >= 1")]
    InvalidSpecK,
    #[error("RVLLM_GEMMA4_SPEC_K={requested} exceeds MAX_SPEC_K={max}; recompile with a larger cap or reduce spec_k")]
    SpecKTooLarge { requested: u32, max: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_sane_values() {
        let c = ServerConfig::default();
        assert_eq!(c.max_queue_depth, 8);
        assert!(c.request_timeout >= Duration::from_secs(60));
    }

    #[test]
    fn validate_rejects_empty_model_dir() {
        let c = ServerConfig::default();
        assert!(matches!(c.validate(), Err(ConfigError::MissingModelDir)));
    }

    #[test]
    fn validate_accepts_queue_depth_one() {
        let here = std::env::current_dir().expect("cwd");
        let c = ServerConfig {
            model_dir: here,
            model_id: "test".into(),
            max_queue_depth: 1,
            max_new_tokens_cap: 32,
            request_timeout: Duration::from_secs(60),
            ..ServerConfig::default()
        };
        assert!(c.validate().is_ok(), "depth=1 should be allowed (strictly serial)");
    }

    #[test]
    fn validate_rejects_zero_request_timeout() {
        // Use std::env::current_dir() as a guaranteed-existing dir so
        // the model-dir check passes; we only want to exercise the
        // request_timeout branch.
        let here = std::env::current_dir().expect("cwd");
        let c = ServerConfig {
            model_dir: here,
            model_id: "test".into(),
            max_queue_depth: 2,
            max_new_tokens_cap: 32,
            request_timeout: Duration::from_secs(0),
            ..ServerConfig::default()
        };
        assert!(matches!(c.validate(), Err(ConfigError::InvalidRequestTimeout)));
    }

    // Spec-decode commit 1 validation.

    #[test]
    fn validate_rejects_spec_decode_without_drafter_dir() {
        let here = std::env::current_dir().expect("cwd");
        let c = ServerConfig {
            model_dir: here,
            model_id: "test".into(),
            max_queue_depth: 2,
            max_new_tokens_cap: 32,
            request_timeout: Duration::from_secs(60),
            spec_decode: true,
            spec_drafter_dir: PathBuf::new(),
            spec_k: 6,
            ..ServerConfig::default()
        };
        assert!(matches!(c.validate(), Err(ConfigError::MissingSpecDrafterDir)));
    }

    #[test]
    fn validate_rejects_spec_decode_with_bogus_drafter_dir() {
        let here = std::env::current_dir().expect("cwd");
        let c = ServerConfig {
            model_dir: here,
            model_id: "test".into(),
            max_queue_depth: 2,
            max_new_tokens_cap: 32,
            request_timeout: Duration::from_secs(60),
            spec_decode: true,
            spec_drafter_dir: PathBuf::from("/this/dir/does/not/exist/rvllm-test"),
            spec_k: 6,
            ..ServerConfig::default()
        };
        assert!(matches!(c.validate(), Err(ConfigError::SpecDrafterDirMissing(_))));
    }

    #[test]
    fn validate_rejects_spec_k_zero() {
        let here = std::env::current_dir().expect("cwd");
        let c = ServerConfig {
            model_dir: here.clone(),
            model_id: "test".into(),
            max_queue_depth: 2,
            max_new_tokens_cap: 32,
            request_timeout: Duration::from_secs(60),
            spec_decode: true,
            spec_drafter_dir: here, // existing dir, satisfies the path check
            spec_k: 0,
            ..ServerConfig::default()
        };
        assert!(matches!(c.validate(), Err(ConfigError::InvalidSpecK)));
    }

    #[test]
    fn validate_ignores_spec_fields_when_decode_off() {
        let here = std::env::current_dir().expect("cwd");
        let c = ServerConfig {
            model_dir: here,
            model_id: "test".into(),
            max_queue_depth: 2,
            max_new_tokens_cap: 32,
            request_timeout: Duration::from_secs(60),
            spec_decode: false,
            spec_drafter_dir: PathBuf::new(),
            spec_k: 0,
            ..ServerConfig::default()
        };
        assert!(c.validate().is_ok());
    }
}
