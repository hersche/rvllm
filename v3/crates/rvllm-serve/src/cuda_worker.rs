//! CUDA-backed worker. Feature-gated (`cuda` / `gb10`).
//!
//! Phase 2 scope: wrap the existing monolithic
//! `Gemma4Bringup::run_generate` synchronously. All generated tokens
//! are emitted to the `events_tx` channel **after** generation
//! completes — not token-by-token on the GPU's pace. True per-token
//! streaming is phase 5 (requires breaking `run_generate` apart into
//! `prefill()` + `decode_one()`).
//!
//! ## RAII gotcha fixed here
//!
//! `LoadedModule` is RAII: its `Drop` calls `cuModuleUnload`. A
//! `KernelFn` is only an opaque handle into that module. If the
//! module drops while the handle lives, the next `cuLaunchKernel`
//! on the handle fails with `LaunchFailed`. [`GenerateKernels`]
//! holds the module alongside the fn-handles to anchor its lifetime
//! — the first Gemma 4 chat request on GB10 died with exactly that
//! error before this fix.

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use tokio::sync::mpsc;
use tokio::sync::oneshot;

use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};

use crate::config::ModelFamily;
use crate::error::ApiError;
use crate::openai::types::FinishReason;
use crate::worker::{GenerateEvent, GenerateRequest, WorkerHandle};

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

/// Minimum config needed to bring up the CUDA worker. Mirrors the
/// `probe-gemma4-load` flags so operators can move directly from
/// probe to serve.
///
/// `Gemma4EnginePaths` is neither `Debug` nor `Clone` upstream, so
/// this wrapper is not either.
pub struct CudaWorkerConfig {
    pub paths: Gemma4EnginePaths,
    pub arena_bytes: usize,
    pub queue_depth: usize,
    /// Resolved model family. Drives explicit dispatch in
    /// [`spawn_cuda_worker`]. With `Auto` the worker still does
    /// per-family marker probing as before, but `Mistral35` /
    /// `Qwen36` / `Gemma4` short-circuit to the matching bring-up
    /// without re-probing.
    pub family: ModelFamily,
    /// Optional Gemma 4 E4B assistant-drafter speculative decode.
    /// Default-off; when enabled the drafter is loaded before the
    /// arena scratch checkpoint so its resident weights survive
    /// per-request arena restores.
    pub spec_decode: bool,
    pub spec_drafter_dir: PathBuf,
    pub spec_k: u32,
}

#[derive(Clone, Debug)]
struct WorkerSpecDecode {
    enabled: bool,
    k: u32,
}

/// Spawn the CUDA worker on a dedicated OS thread.
///
/// Returns once `Gemma4Bringup::load` has completed — the returned
/// `WorkerHandle` is immediately usable. `load` takes ~25 s for
/// Gemma 4 31B; the caller should render a "loading" log line before
/// awaiting.
pub async fn spawn_cuda_worker(
    cfg: CudaWorkerConfig,
) -> Result<(WorkerHandle, std::thread::JoinHandle<()>), ApiError> {
    // Channel buffer must equal admission-permit count. The earlier
    // arithmetic (`queue_depth - 1` plus an "in-flight slot on the
    // worker") was wrong: the worker only frees a buffer slot when
    // it pulls from the channel, so until that recv returns, the
    // buffer IS the cap. With `queue_depth` permits but a
    // `queue_depth - 1` buffer, a cold-burst of `queue_depth`
    // concurrent admissions could see one handler's `try_send` fail
    // with Busy after fetch+tokenize work — violating the "permit
    // reserves the lifecycle slot" contract. Clamp to >= 1 because
    // `mpsc::channel(0)` panics.
    let channel_buf = cfg.queue_depth.max(1);
    let (req_tx, mut req_rx) = mpsc::channel::<GenerateRequest>(channel_buf);
    let (ready_tx, ready_rx) = oneshot::channel::<Result<(), String>>();

    let CudaWorkerConfig {
        paths,
        arena_bytes,
        family,
        spec_decode,
        spec_drafter_dir,
        spec_k,
        ..
    } = cfg;
    let spec_cfg = WorkerSpecDecode {
        enabled: spec_decode,
        k: spec_k,
    };
    let join = std::thread::Builder::new()
        .name("rvllm-serve-cuda-worker".into())
        .spawn(move || {
            // RVLLM_GEMMA4_SPEC_DECODE applies to either the fp8-block
            // Gemma 4 path (production) or the Gemma4-NVFP4 weight
            // path (Phase 3c — still fails-fast at the forward-path
            // dispatch below, with a more specific message). Reject
            // explicitly for any other family so an operator doesn't
            // wonder why the flag is silently ignored on Qwen / Mistral.
            if spec_decode
                && !matches!(family, ModelFamily::Gemma4 | ModelFamily::Gemma4Nvfp4)
            {
                let _ = ready_tx.send(Err(format!(
                    "RVLLM_GEMMA4_SPEC_DECODE=1 is only supported for Gemma 4 \
                     (fp8-block and NVFP4 weight variants); resolved model \
                     family is {}",
                    family.as_str()
                )));
                return;
            }
            // Mistral 3.5: parse arch + validate inventory + assert
            // CUTLASS NVFP4 symbols present, then refuse per-request
            // generation cleanly until the GPU forward path is wired.
            // This covers steps 1-4 + 7 partial — the operator gets
            // concrete startup diagnostics on a real Mistral
            // checkpoint, and per-request errors carry the same
            // "kernel not implemented" reason instead of corrupting
            // arena state.
            // Qwen 3.5 27B dense — Phase 0 only. Parse arch + log
            // summary, then reject every generate request with a
            // typed error pointing at QWEN35_BRINGUP_PLAN.md.
            if matches!(family, ModelFamily::Qwen35) {
                use rvllm_runtime::qwen35_bring_up::{
                    Qwen35Bringup, Qwen35EnginePaths,
                };
                let q35_paths = Qwen35EnginePaths {
                    model_dir: paths.model_dir.clone(),
                    kernels_dir: paths.kernels_dir.clone(),
                    cutlass_so: paths.cutlass_so.clone(),
                    fa3_so: paths.fa3_so.clone(),
                    policy_json: paths.policy_json.clone(),
                };
                let bringup = match Qwen35Bringup::load(q35_paths, arena_bytes) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!(
                            "Qwen35Bringup::load: {e:?}"
                        )));
                        return;
                    }
                };
                if std::env::var("RVLLM_QWEN35_SPEC_STATE_SELFTEST").as_deref() == Ok("1") {
                    if let Err(e) = bringup.spec_state_snapshot_selftest() {
                        let _ = ready_tx.send(Err(format!(
                            "qwen35 spec-state snapshot selftest: {e:?}"
                        )));
                        return;
                    }
                    tracing::info!(
                        "qwen35 spec-state snapshot selftest passed"
                    );
                }
                if std::env::var("RVLLM_QWEN35_SPEC_PRIMITIVE_SELFTEST").as_deref() == Ok("1") {
                    let primitive_selftest = unsafe {
                        bringup.spec_decode_primitives_selftest()
                    };
                    if let Err(e) = primitive_selftest {
                        let _ = ready_tx.send(Err(format!(
                            "qwen35 spec primitive selftest: {e:?}"
                        )));
                        return;
                    }
                    tracing::info!(
                        "qwen35 spec primitive selftest passed"
                    );
                }
                let _ = ready_tx.send(Ok(()));
                tracing::info!(
                    "Qwen 3.5 dense — Phase 2c-A complete (substrate + \
                     outside kernels + cuBLASLt). Per-request generation \
                     drives embed → final-RMSNorm → lm_head → argmax \
                     with the 64 transformer layers SKIPPED — the output \
                     token is structurally valid but semantically \
                     meaningless until Phase 2c-B. See \
                     QWEN35_BRINGUP_PLAN.md."
                );
                let qwen35_fwd_mode = std::env::var("RVLLM_QWEN35_FWD")
                    .ok().unwrap_or_else(|| "outside".to_string());
                tracing::info!("qwen35 forward mode: {qwen35_fwd_mode}");
                while let Some(req) = req_rx.blocking_recv() {
                    // Phase 2c-A → 2c-E mode dispatch:
                    //   outside, dense_mlp, qkv_mlp, linear, all_layers
                    //     — single-token probes (1 Token + Done).
                    //   generate (default Phase 2c-E)
                    //     — full prefill + decode loop, streams Token
                    //       events up to max_new_tokens.
                    let last_tok = req.prompt_ids.last().copied().unwrap_or(1);
                    if qwen35_fwd_mode == "generate" {
                        // Codex review (drive-by fix): Qwen 3.5 allocates
                        // persistent KV + linear-attn delta state + conv1d
                        // state at load time but the worker never resets
                        // them per request, so state from request N can
                        // leak into request N+1. Qwen 3.6 has this reset
                        // — Qwen 3.5 did not. Match the Qwen 3.6 pattern.
                        let reset_ok = bringup.reset_linear_state()
                            .and_then(|_| bringup.reset_kv_cache())
                            .and_then(|_| bringup.reset_conv_state())
                            // Repetition-penalty histogram must start
                            // empty each request (counts never leak).
                            .and_then(|_| bringup.reset_rep_count());
                        if let Err(e) = reset_ok {
                            let _ = req.events_tx.send(GenerateEvent::Error(
                                format!("qwen35 per-request reset: {e:?}"),
                            ));
                            continue;
                        }

                        let prompt_ids = req.prompt_ids.clone();
                        let max_new = req.max_new_tokens.max(1);
                        let prompt_len = prompt_ids.len() as u32;
                        let events_tx = req.events_tx.clone();

                        // Phase 3-b: vision pre-pass. Decode each
                        // image once through the Qwen3-VL ViT and
                        // accumulate (token_start, vision_bytes)
                        // tuples for the splice into prefill.
                        // Mirrors the Qwen 3.6 path below.
                        let mut vision_outputs: Vec<
                            rvllm_runtime::qwen36_bring_up::VisionForwardOutput,
                        > = Vec::with_capacity(req.vision_items.len());
                        let mut vision_failed = false;
                        for (i, item) in req.vision_items.iter().enumerate() {
                            match bringup.forward_qwen_vision(&item.bytes) {
                                Ok(out) => {
                                    if out.num_tokens != item.num_tokens {
                                        let _ = req.events_tx.send(GenerateEvent::Error(
                                            format!("qwen35 vision: tokens mismatch \
                                                     (predicted {} got {})",
                                                    item.num_tokens, out.num_tokens),
                                        ));
                                        vision_failed = true;
                                        break;
                                    }
                                    tracing::info!(
                                        idx = i, tokens = out.num_tokens,
                                        hidden = out.hidden_dim,
                                        "qwen35 vision: ViT forward done"
                                    );
                                    vision_outputs.push(out);
                                }
                                Err(e) => {
                                    let _ = req.events_tx.send(GenerateEvent::Error(
                                        format!("qwen35 vision forward: {e:?}"),
                                    ));
                                    vision_failed = true;
                                    break;
                                }
                            }
                        }
                        if vision_failed {
                            continue;
                        }
                        let splices: Vec<(usize, &[u8])> = req.vision_slots
                            .iter()
                            .map(|s| (s.token_start,
                                      vision_outputs[s.vision_item_idx].data.as_slice()))
                            .collect();

                        let stop_ids: std::collections::HashSet<u32> =
                            req.stop_token_ids.iter().copied().collect();
                        let stopped_on_eos = std::cell::Cell::new(false);
                        let spec_min_prompt_tokens = std::env::var(
                            "RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS")
                            .ok()
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(1024);
                        let spec_min_max_new_tokens = std::env::var(
                            "RVLLM_QWEN35_SPEC_MIN_MAX_NEW_TOKENS")
                            .ok()
                            .and_then(|s| s.parse::<u32>().ok())
                            .unwrap_or(64);
                        let spec_min_full_draft_hits = std::env::var(
                            "RVLLM_QWEN35_SPEC_PREFLIGHT_MIN_FULL_DRAFTS")
                            .ok()
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(0);
                        let spec_k = std::env::var("RVLLM_QWEN35_SPEC_K")
                            .ok()
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(4)
                            .max(1);
                        let spec_ngram = std::env::var("RVLLM_QWEN35_SPEC_NGRAM")
                            .ok()
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(2)
                            .max(1);
                        let enough_full_draft_hits = if spec_min_full_draft_hits == 0 {
                            true
                        } else if prompt_ids.len() < spec_ngram + spec_k + 1 {
                            false
                        } else if spec_ngram == 2 {
                            let mut seen = std::collections::HashSet::<(u32, u32)>::new();
                            let mut hits = 0usize;
                            let last_start = prompt_ids.len() - spec_ngram - spec_k;
                            for start in 0..=last_start {
                                let key = (prompt_ids[start], prompt_ids[start + 1]);
                                if !seen.insert(key) {
                                    hits += 1;
                                    if hits >= spec_min_full_draft_hits {
                                        break;
                                    }
                                }
                            }
                            hits >= spec_min_full_draft_hits
                        } else {
                            let mut seen = std::collections::HashSet::<Vec<u32>>::new();
                            let mut hits = 0usize;
                            let last_start = prompt_ids.len() - spec_ngram - spec_k;
                            for start in 0..=last_start {
                                let key = prompt_ids[start..start + spec_ngram].to_vec();
                                if !seen.insert(key) {
                                    hits += 1;
                                    if hits >= spec_min_full_draft_hits {
                                        break;
                                    }
                                }
                            }
                            hits >= spec_min_full_draft_hits
                        };
                        // Publish per-request sampling params to the
                        // bringup. The decode token-selection site reads
                        // these once per token: temperature>0 routes to
                        // the top-k/top-p sampler kernel, temperature==0
                        // keeps the legacy greedy argmax (byte-identical).
                        // Qwen 3 degrades under greedy; this is what lets
                        // it run at its recommended temp=0.6/top_k=20/
                        // top_p=0.95 instead of being forced to argmax.
                        match req.sampling {
                            crate::sampling::SamplingDecision::Greedy => {
                                bringup.set_sampling(0.0, 0, 1.0, 0);
                            }
                            crate::sampling::SamplingDecision::Stochastic(s) => {
                                // Qwen 3 is documented to run best at
                                // top_k=20 / top_p=0.95. Clients that send
                                // only a temperature (e.g. zeroclaw, which
                                // exposes no top_k/top_p knob) would
                                // otherwise sample over the full top-64 cap
                                // with no nucleus → occasional tail
                                // degradation. Apply the recommended
                                // nucleus as the family default when the
                                // request leaves them unset; explicit
                                // request values always win.
                                let top_k = s.top_k.unwrap_or(20);
                                let top_p = if s.top_p >= 1.0 { 0.95 } else { s.top_p };
                                bringup.set_sampling(
                                    s.temperature, top_k, top_p, s.seed,
                                );
                            }
                        }
                        // Repetition penalty (design A). Driven by env
                        // (mirrors gemma's RVLLM_REPETITION_PENALTY) so
                        // clients that send no penalty fields — zeroclaw
                        // sends temperature only — still get the profile
                        // default. freq+presence both 0 → disabled →
                        // byte-identical decode. min_count gates which
                        // tokens are penalized (>=2 protects function
                        // words; raises only on genuine repetition).
                        {
                            let freq = std::env::var("RVLLM_QWEN35_FREQUENCY_PENALTY")
                                .ok().and_then(|s| s.parse::<f32>().ok())
                                .unwrap_or(0.0);
                            let presence = std::env::var("RVLLM_QWEN35_PRESENCE_PENALTY")
                                .ok().and_then(|s| s.parse::<f32>().ok())
                                .unwrap_or(0.0);
                            let min_count = std::env::var("RVLLM_QWEN35_REP_MIN_COUNT")
                                .ok().and_then(|s| s.parse::<u32>().ok())
                                .unwrap_or(2);
                            bringup.set_penalty(freq, presence, min_count);
                        }
                        let spec_decode_on = std::env::var("RVLLM_QWEN35_SPEC_DECODE")
                            .map(|v| v != "0" && !v.is_empty())
                            .unwrap_or(false)
                            && splices.is_empty()
                            && prompt_ids.len() >= spec_min_prompt_tokens
                            && max_new >= spec_min_max_new_tokens
                            && enough_full_draft_hits
                            // Prompt-lookup spec verification assumes greedy
                            // acceptance; never spec a stochastic request.
                            && req.sampling.is_greedy();
                        let result = if spec_decode_on {
                            rvllm_runtime::qwen35_spec_decode::run_qwen35_prompt_lookup_spec(
                                &bringup, &prompt_ids, max_new,
                                &req.stop_token_ids,
                                |tok_id, pos| {
                                    if stop_ids.contains(&tok_id) {
                                        stopped_on_eos.set(true);
                                        return false;
                                    }
                                    events_tx.send(GenerateEvent::Token {
                                        id: tok_id, position: pos,
                                    }).is_ok()
                                },
                            ).map(|(emitted, _reason)| emitted)
                        } else {
                            unsafe {
                                bringup.generate_session_with_vision(
                                    &prompt_ids, max_new, &splices,
                                    |tok_id, pos| {
                                        // EOS check: if this token is a
                                        // stop token, emit nothing and
                                        // signal short-circuit. The Done
                                        // event below will carry
                                        // FinishReason::Stop.
                                        if stop_ids.contains(&tok_id) {
                                            stopped_on_eos.set(true);
                                            return false;
                                        }
                                        events_tx.send(GenerateEvent::Token {
                                            id: tok_id, position: pos,
                                        }).is_ok()
                                    },
                                )
                            }
                        };
                        match result {
                            Ok(emitted) => {
                                let finish = if stopped_on_eos.get() {
                                    FinishReason::Stop
                                } else {
                                    FinishReason::Length
                                };
                                // When stop_on_eos fires the loop
                                // counts the EOS slot toward emitted
                                // but doesn't actually send it as a
                                // Token event — subtract it so the
                                // usage/completion_tokens matches the
                                // events the client received.
                                let user_emitted = if stopped_on_eos.get() {
                                    emitted.saturating_sub(1)
                                } else { emitted };
                                let _ = req.events_tx.send(GenerateEvent::Done {
                                    finish,
                                    completion_tokens: user_emitted,
                                    prompt_tokens: prompt_len,
                                });
                            }
                            Err(e) => {
                                let _ = req.events_tx.send(GenerateEvent::Error(
                                    format!("qwen35 generate: {e:?}"),
                                ));
                            }
                        }
                        continue;
                    }
                    let result = match qwen35_fwd_mode.as_str() {
                        "dense_mlp" => unsafe {
                            bringup.forward_one_dense_mlp_smoke(last_tok)
                        },
                        "qkv_mlp" => unsafe {
                            bringup.forward_layer3_qkv_plus_mlp_smoke(last_tok)
                        },
                        "linear" => unsafe {
                            bringup.forward_layer0_linear_smoke(last_tok)
                        },
                        "all_layers" => unsafe {
                            bringup.forward_all_layers_smoke(last_tok, 0)
                        },
                        _ => unsafe { bringup.forward_outside_only_smoke(last_tok) },
                    };
                    match result {
                        Ok(predicted) => {
                            let _ = req.events_tx.send(GenerateEvent::Token {
                                id: predicted,
                                position: req.prompt_ids.len() as u32,
                            });
                            let _ = req.events_tx.send(GenerateEvent::Done {
                                finish: FinishReason::Length,
                                completion_tokens: 1,
                                prompt_tokens: req.prompt_ids.len() as u32,
                            });
                        }
                        Err(e) => {
                            let _ = req.events_tx.send(GenerateEvent::Error(
                                format!("qwen35 single-tok smoke: {e:?}"),
                            ));
                        }
                    }
                }
                return;
            }
            if matches!(family, ModelFamily::Mistral35) {
                let bringup = match rvllm_runtime::mistral35_bring_up::Mistral35Bringup::load(
                    paths,
                    arena_bytes,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!(
                            "Mistral35Bringup::load: {e:?}"
                        )));
                        return;
                    }
                };
                tracing::info!(
                    nvfp4_active = bringup.nvfp4_active,
                    "Mistral 3.5 bring-up validated; forward path ready"
                );
                let _ = ready_tx.send(Ok(()));

                fn stage_stats(label: &str, v: &[f32]) {
                    let n = v.len();
                    // Round-9 #4 fix: stage dumps are only populated
                    // when RVLLM_SMOKE_FULL_DUMP=1 (or boundary dump).
                    // For normal runs they're empty, so the previous
                    // `(sumsq / 0).sqrt()` produced NaN min / -Inf max
                    // / NaN rms in the diagnostic log. Emit a single
                    // "not collected" line instead.
                    if n == 0 {
                        tracing::debug!(stage = label,
                            "stage not collected (RVLLM_SMOKE_FULL_DUMP=0)");
                        return;
                    }
                    let any_nan = v.iter().any(|x| x.is_nan());
                    let any_inf = v.iter().any(|x| x.is_infinite());
                    let mut min = f32::INFINITY;
                    let mut max = f32::NEG_INFINITY;
                    let mut nz = 0usize;
                    let mut sumsq = 0.0f64;
                    for &x in v {
                        if x < min { min = x; }
                        if x > max { max = x; }
                        if x != 0.0 { nz += 1; }
                        sumsq += (x as f64) * (x as f64);
                    }
                    let rms = (sumsq / (n as f64)).sqrt();
                    let head: Vec<String> = v.iter().take(8)
                        .map(|x| format!("{:+.4}", x)).collect();
                    tracing::info!(
                        n, any_nan, any_inf, nz, min, max, rms,
                        "mistral35 smoke {label} head[8]={}", head.join(",")
                    );
                }

                while let Some(req) = req_rx.blocking_recv() {
                    // Round-12 phase 5c codex review #3 — REVERTED.
                    // Reverted to the round-trip path
                    // (generate_with_vision + Vec<u8>) because the
                    // device-resident generate_with_images path
                    // regressed semantic correctness on real images
                    // ("orange ball" → "I need to see the image").
                    // Root cause needs more investigation. The
                    // forward_pixtral_vision_into helper stays available
                    // for future use; the cuda_worker uses the old
                    // forward_pixtral_vision (DtoH) + generate_with_vision
                    // (HtoD) for now to keep production correct.
                    //
                    // Codex review #3 (2026-05-11): re-probing the
                    // device-resident path under
                    // RVLLM_MISTRAL35_VISION_DEVICE_RESIDENT=1.
                    // ONLY enabled for single-slot-per-image cases
                    // (H=1) until the H>1 slot-aware variant is
                    // wired through generate_with_images. Falls
                    // back to the DtoH path on any anomaly.
                    let try_device_resident_vision =
                        std::env::var("RVLLM_MISTRAL35_VISION_DEVICE_RESIDENT")
                            .ok().as_deref()
                            .map(|s| s != "0" && !s.is_empty())
                            .unwrap_or(true)
                        && !req.vision_items.is_empty();
                    let mut vision_splices: Vec<(usize, usize, Vec<u8>)> = Vec::new();
                    // Slot-aware device-resident inputs: raw image
                    // bytes per UNIQUE image + (token_start,
                    // num_tokens, image_idx, row_offset) tuples
                    // honouring Pixtral H>1 row-separator layout.
                    let mut vision_images_raw: Vec<Vec<u8>> = Vec::new();
                    let mut vision_slot_tuples: Vec<(usize, usize, usize, usize)>
                        = Vec::new();
                    if try_device_resident_vision {
                        for item in req.vision_items.iter() {
                            vision_images_raw.push(item.bytes.clone());
                        }
                        for slot in req.vision_slots.iter() {
                            vision_slot_tuples.push((
                                slot.token_start,
                                slot.num_tokens,
                                slot.vision_item_idx,
                                slot.vision_row_offset,
                            ));
                        }
                    } else if !req.vision_items.is_empty() {
                        // Run the Pixtral vision tower ONCE per image
                        // and reuse the output across the H per-row
                        // VisionSlots. Without dedup the H slots emitted
                        // by the tokenizer for one Pixtral image would
                        // each trigger their own forward pass — H×
                        // wasted work.
                        let mut outputs: Vec<Option<rvllm_runtime::mistral35_bring_up
                            ::Mistral35VisionForwardOutput>> =
                            (0..req.vision_items.len()).map(|_| None).collect();
                        for slot in req.vision_slots.iter() {
                            let vi_idx = slot.vision_item_idx;
                            if outputs[vi_idx].is_some() { continue; }
                            let item = &req.vision_items[vi_idx];
                            tracing::info!(
                                "[mistral35-vision] image {}/{} ({} bytes, \
                                 {}x{}, merged={}x{} num_soft_tokens={})",
                                vi_idx + 1, req.vision_items.len(),
                                item.bytes.len(), item.width, item.height,
                                item.merged_h, item.merged_w, item.num_soft_tokens,
                            );
                            match bringup.forward_pixtral_vision(&item.bytes) {
                                Ok(out) => {
                                    if out.num_tokens != item.num_soft_tokens {
                                        tracing::warn!(
                                            "[mistral35-vision] image {} \
                                             produced {} soft tokens but \
                                             admission predicted {}",
                                            vi_idx + 1, out.num_tokens,
                                            item.num_soft_tokens,
                                        );
                                    }
                                    outputs[vi_idx] = Some(out);
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "[mistral35-vision] image {} forward \
                                         failed: {e:?}", vi_idx + 1,
                                    );
                                }
                            }
                        }
                        // Emit one splice per VisionSlot, slicing the
                        // shared image output by `vision_row_offset`.
                        for slot in req.vision_slots.iter() {
                            let vi_idx = slot.vision_item_idx;
                            let Some(out) = outputs[vi_idx].as_ref() else { continue };
                            let bytes_per_tok = out.hidden_dim * 2;
                            let row_start_bytes = slot.vision_row_offset * bytes_per_tok;
                            let take = slot.num_tokens * bytes_per_tok;
                            // Bounds check — admission predicts the
                            // grid, but a mismatch would silently splice
                            // garbage. Clamp + log instead of panic.
                            let avail_bytes = out.data.len();
                            if row_start_bytes + take > avail_bytes {
                                tracing::warn!(
                                    "[mistral35-vision] slot row_offset={} num_tokens={} \
                                     would read {}..{} of vision output ({} bytes); \
                                     skipping this slot",
                                    slot.vision_row_offset, slot.num_tokens,
                                    row_start_bytes, row_start_bytes + take, avail_bytes,
                                );
                                continue;
                            }
                            vision_splices.push((
                                slot.token_start,
                                slot.num_tokens,
                                out.data[row_start_bytes..row_start_bytes + take].to_vec(),
                            ));
                        }
                    }
                    // Multi-token autoregressive generation. Tokenize
                    // → prefill (one forward per prompt token, KV cache
                    // built up) → decode max_new tokens. Empty prompt
                    // falls back to RVLLM_SMOKE_TOKEN as a single seed.
                    let env_token: u32 = std::env::var("RVLLM_SMOKE_TOKEN")
                        .ok().and_then(|s| s.parse().ok()).unwrap_or(1);
                    let prompt: Vec<u32> = if req.prompt_ids.is_empty() {
                        vec![env_token]
                    } else {
                        req.prompt_ids.clone()
                    };
                    // Round-11 #1: production path no longer reads
                    // RVLLM_SMOKE_MAX_NEW. The bringup-load step
                    // already rejected any leaked SMOKE_* env without
                    // the explicit `RVLLM_DEBUG_MISTRAL35=1` gate, so
                    // by the time we get here the env is either unset
                    // (production) or the operator opted in to debug
                    // mode and accepts a possibly-truncated max_new.
                    let env_max: Option<usize> = rvllm_runtime::mistral35_bring_up
                        ::debug_env_str("RVLLM_SMOKE_MAX_NEW")
                        .and_then(|s| s.parse().ok());
                    let req_max = (req.max_new_tokens as usize).max(1);
                    let max_new: usize = match env_max {
                        Some(em) => req_max.min(em),
                        None => req_max,
                    };
                    // F1#1 fix: use the request's resolved stop_token_ids
                    // (handler already merged tokenizer EOS + user
                    // `stop` strings) so user stops actually halt
                    // generation. Falls back to canonical Mistral EOS=2
                    // if the handler somehow passed an empty list.
                    let eos: Vec<u32> = if req.stop_token_ids.is_empty() {
                        vec![2]
                    } else {
                        req.stop_token_ids.clone()
                    };
                    // F1#2 fix: rebind the CUDA context to this OS
                    // thread before each request. Mirrors the Qwen
                    // path's GB10 fix — after a long idle the worker
                    // thread otherwise hits cuLaunchKernel errors.
                    #[cfg(feature = "cuda")]
                    if let Some(ctx) = bringup.ctx.as_ref() {
                        if let Err(e) = ctx.bind_to_current_thread() {
                            let _ = req.events_tx.send(GenerateEvent::Error(
                                format!("mistral35 ctx rebind: {e:?}")));
                            continue;
                        }
                    }
                    // F1#1 fix: pass cancellation flag so client
                    // disconnects + request_timeout actually halt the
                    // 88-layer-per-token loop instead of waiting for it
                    // to finish.
                    // Round-9 #3: emit Token events from the on_token
                    // callback immediately (mirrors the Qwen / Gemma
                    // worker shape). Without this stream=true was
                    // effectively batched: the user only saw output
                    // once `generate` returned. Closing the SSE
                    // channel or hitting a stop token now flips
                    // `req.cancelled` so the next forward bails.
                    let stop_set: std::collections::HashSet<u32> =
                        req.stop_token_ids.iter().copied().collect();
                    let prompt_len_u32 = prompt.len() as u32;
                    let events_tx_inner = req.events_tx.clone();
                    let cancel_inner = std::sync::Arc::clone(&req.cancelled);
                    let mut emitted_in_closure: u32 = 0;
                    let mut step_in_closure: u32 = 0;
                    let mut hit_stop_in_closure = false;
                    let cancel_ref: &std::sync::atomic::AtomicBool = &*req.cancelled;
                    let on_token_cb = |tok: u32| {
                        let pos = prompt_len_u32 + step_in_closure;
                        step_in_closure += 1;
                        if stop_set.contains(&tok) {
                            hit_stop_in_closure = true;
                            cancel_inner.store(true, Ordering::Relaxed);
                            return;
                        }
                        if events_tx_inner.send(GenerateEvent::Token {
                            id: tok, position: pos,
                        }).is_err() {
                            cancel_inner.store(true, Ordering::Relaxed);
                            return;
                        }
                        emitted_in_closure += 1;
                    };
                    let gen = if try_device_resident_vision {
                        unsafe {
                            bringup.generate_with_vision_slots(
                                &prompt, max_new, &eos,
                                Some(cancel_ref), on_token_cb,
                                &vision_images_raw,
                                &vision_slot_tuples,
                            )
                        }
                    } else {
                        unsafe {
                            bringup.generate_with_vision(
                                &prompt, max_new, &eos,
                                Some(cancel_ref), on_token_cb,
                                &vision_splices,
                            )
                        }
                    };
                    let (gen_tokens, prompt_len_for_log, last_dump_opt, gen_err) = match gen {
                        Ok(r) => (r.tokens, r.prompt_len, r.last_dump, None),
                        Err(e) => (Vec::new(), prompt.len(), None, Some(e)),
                    };
                    if !gen_tokens.is_empty() {
                        tracing::info!(
                            "mistral35 generated tokens: prompt_len={} all={:?} new={:?}",
                            prompt_len_for_log,
                            &gen_tokens[..],
                            &gen_tokens.get(prompt_len_for_log..).unwrap_or(&[]));
                    }
                    let token_id: u32 = *prompt.last().unwrap();
                    // #3 fix: last_dump is Optional now (cancellation
                    // before the first forward leaves it None instead
                    // of panicking via .expect("at least one forward")).
                    // The dump-stages summary is purely diagnostic —
                    // skip it cleanly when missing.
                    let summary = match (&last_dump_opt, &gen_err) {
                        (Some(d), _) => {
                            // Optional: persist all stage vectors to disk
                            // for offline compare against a vllm reference.
                            // RVLLM_SMOKE_DUMP_DIR=/tmp/rvllm-mistral35-dump
                            if let Some(dir) = std::env::var_os("RVLLM_SMOKE_DUMP_DIR") {
                                let dir = std::path::PathBuf::from(dir);
                                let _ = std::fs::create_dir_all(&dir);
                                let dumps: &[(&str, &[f32])] = &[
                                    ("post_embed", &d.post_embed),
                                    ("post_rmsnorm", &d.post_rmsnorm),
                                    ("q_out", &d.q_out),
                                    ("k_out", &d.k_out),
                                    ("v_out", &d.v_out),
                                    ("attn_out", &d.attn_out),
                                    ("o_out", &d.o_out),
                                    ("h_after_attn", &d.h_after_attn),
                                    ("post_attn_norm", &d.post_attn_norm),
                                    ("gate_out", &d.gate_out),
                                    ("up_out", &d.up_out),
                                    ("silu_mid", &d.silu_mid),
                                    ("down_out", &d.down_out),
                                    ("h_after_layer0", &d.h_after_layer0),
                                    ("h_after_final_norm", &d.h_after_final_norm),
                                ];
                                for (name, v) in dumps {
                                    if v.is_empty() { continue; }
                                    let path = dir.join(format!("{name}.f32"));
                                    let bytes: Vec<u8> = v.iter()
                                        .flat_map(|x| x.to_le_bytes()).collect();
                                    if let Err(e) = std::fs::write(&path, &bytes) {
                                        tracing::error!("dump write {path:?}: {e}");
                                    }
                                }
                                tracing::info!(?dir,
                                    "mistral35 smoke stages written");
                            }
                            stage_stats("post_embed", &d.post_embed);
                            stage_stats("post_rmsnorm", &d.post_rmsnorm);
                            stage_stats("q_out", &d.q_out);
                            stage_stats("k_out", &d.k_out);
                            stage_stats("v_out", &d.v_out);
                            stage_stats("attn_out", &d.attn_out);
                            stage_stats("o_out", &d.o_out);
                            stage_stats("h_after_attn", &d.h_after_attn);
                            stage_stats("post_attn_norm", &d.post_attn_norm);
                            stage_stats("gate_out", &d.gate_out);
                            stage_stats("up_out", &d.up_out);
                            stage_stats("silu_mid", &d.silu_mid);
                            stage_stats("down_out", &d.down_out);
                            stage_stats("h_after_layer0", &d.h_after_layer0);
                            stage_stats("h_after_final_norm", &d.h_after_final_norm);
                            // Per-layer residual rms cascade (88 entries).
                            let lr_min = d.layer_residual_rms.iter().cloned().fold(f32::INFINITY, f32::min);
                            let lr_max = d.layer_residual_rms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let lr_first = &d.layer_residual_rms[..d.layer_residual_rms.len().min(4)];
                            let lr_last  = &d.layer_residual_rms[d.layer_residual_rms.len().saturating_sub(4)..];
                            tracing::info!(
                                n_layers = d.layer_residual_rms.len(),
                                lr_min, lr_max,
                                "mistral35 smoke layer_residual_rms first4={:.3?} last4={:.3?}",
                                lr_first, lr_last,
                            );
                            tracing::info!(
                                predicted_token = d.predicted_token,
                                "mistral35 smoke logits_top8={:?}", d.logits_top8,
                            );
                            let stages = [
                                ("embed", &d.post_embed[..]),
                                ("norm_in", &d.post_rmsnorm[..]),
                                ("q", &d.q_out[..]),
                                ("k", &d.k_out[..]),
                                ("v", &d.v_out[..]),
                                ("attn", &d.attn_out[..]),
                                ("o", &d.o_out[..]),
                                ("h+attn", &d.h_after_attn[..]),
                                ("norm_post", &d.post_attn_norm[..]),
                                ("gate", &d.gate_out[..]),
                                ("up", &d.up_out[..]),
                                ("silu", &d.silu_mid[..]),
                                ("down", &d.down_out[..]),
                                ("h_l0", &d.h_after_layer0[..]),
                                ("h_fnorm", &d.h_after_final_norm[..]),
                            ];
                            // Round-9 #4: skip stages that weren't
                            // collected (n=0 → previous code emitted
                            // NaN). Empty stages are the default unless
                            // RVLLM_SMOKE_FULL_DUMP=1.
                            let parts: Vec<String> = stages.iter().filter_map(|(name, v)| {
                                let n = v.len();
                                if n == 0 { return None; }
                                let mut sumsq = 0.0f64;
                                for &x in *v { sumsq += (x as f64) * (x as f64); }
                                let rms = (sumsq / (n as f64)).sqrt();
                                Some(format!("{name}={rms:.3}"))
                            }).collect();
                            format!(
                                "in={token_id} predicted={} rms[{}]",
                                d.predicted_token, parts.join(",")
                            )
                        }
                        (None, Some(e)) => format!("generate FAILED: {e:?}"),
                        (None, None) => format!(
                            "in={token_id} cancelled before any forward completed"
                        ),
                    };
                    // Round-9 #3 fix: tokens were already streamed by
                    // the on_token closure above. Only Done / Error
                    // remains.
                    if gen_err.is_none() && (emitted_in_closure > 0 || hit_stop_in_closure) {
                        let finish = if hit_stop_in_closure {
                            FinishReason::Stop
                        } else if req.cancelled.load(Ordering::Relaxed) {
                            FinishReason::Cancelled
                        } else {
                            FinishReason::Length
                        };
                        let _ = req.events_tx.send(GenerateEvent::Done {
                            finish,
                            completion_tokens: emitted_in_closure,
                            prompt_tokens: prompt_len_for_log as u32,
                        });
                        tracing::debug!("[mistral35 debug] {summary}");
                    } else {
                        // Generation failed before the first emitted
                        // token, or hit cancellation before any forward
                        // completed. Surface the diagnostic summary.
                        let _ = req.events_tx.send(GenerateEvent::Error(format!(
                            "[mistral35] {summary}"
                        )));
                    }
                }
                tracing::info!("Mistral 3.5 cuda worker queue closed, exiting");
                return;
            }

            // Qwen 3.6 detection — under explicit `--model-family`,
            // skip the probe; under Auto fall back to the marker
            // probe. `Gemma4` selection short-circuits straight to the
            // Gemma 4 loader.
            // Gemma4Nvfp4 — Option B native NVFP4-weights + NVFP4-KV
            // text-only worker. Floor → #5f sequence (commits
            // bc89650..d152cc9) wired the 60-layer forward and
            // multi-token prompt processing. This branch makes it
            // servable via the OpenAI API for the first time.
            //
            // Scope of this commit (codex Stream-5+6+7 review,
            // "fix current blocker"): greedy-only, text-only,
            // no spec-decode, no vision/audio. Each rejection has a
            // typed error message pointing at the corresponding
            // stream that lifts it.
            if matches!(family, ModelFamily::Gemma4Nvfp4) {
                use rvllm_runtime::gemma4_nvfp4_bring_up::Gemma4Nvfp4Bringup;
                // Spec-decode gate: when on, the drafter dir
                // must exist and the family-specific resolver
                // must yield a source-layer pair. Validate up
                // front so a bad config fails at ready_tx time
                // rather than mid-request.
                if spec_decode {
                    if spec_cfg.k == 0 {
                        let _ = ready_tx.send(Err(format!(
                            "RVLLM_GEMMA4_SPEC_K={} is invalid on the \
                             Option B Gemma4-NVFP4 path; K must be >= 1.",
                            spec_cfg.k)));
                        return;
                    }
                    if !spec_drafter_dir.is_dir() {
                        let _ = ready_tx.send(Err(format!(
                            "RVLLM_GEMMA4_SPEC_DECODE=1 but \
                             RVLLM_GEMMA4_DRAFTER_DIR={:?} is not a \
                             directory. Point it at the assistant \
                             checkpoint (e.g. \
                             /home/r00t/gemma-4-31B-it-assistant).",
                             spec_drafter_dir)));
                        return;
                    }
                }
                let _: &WorkerSpecDecode = &spec_cfg;
                // Conservative max_pos cap for KV cache. The
                // checkpoint advertises max_position_embeddings =
                // 262144 but per-layer NVFP4 KV at that cap would
                // be O(60 GiB). Default 4096 matches the production
                // NVFP4 smoke profile and fits comfortably alongside
                // ~22 GiB of model weights in a 40 GiB arena.
                // Override via env G4N_KV_MAX_POS (e.g. 16384 for
                // the full zeroclaw persona prompt).
                let g4n_kv_max_pos: u32 = std::env::var("G4N_KV_MAX_POS")
                    .ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
                // Stream-#5f-PRIME: prompt-path batched prefill needs
                // max_query_tokens >= prompt length per call. Cap at
                // g4n_kv_max_pos so the chunk equals the full context.
                let g4n_kv_max_query_tokens: u32 = g4n_kv_max_pos;
                let mut bringup = match Gemma4Nvfp4Bringup::load(
                    &paths.model_dir, arena_bytes, &paths.kernels_dir,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!(
                            "Gemma4Nvfp4Bringup::load: {e:?}"
                        )));
                        return;
                    }
                };
                let kv = match bringup.allocate_kv_state_with_chunk(
                    g4n_kv_max_pos, g4n_kv_max_query_tokens,
                ) {
                    Ok(k) => k,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!(
                            "Gemma4Nvfp4Bringup::allocate_kv_state: {e:?}"
                        )));
                        return;
                    }
                };

                // Prefix-cache clobber fix: optional SCRATCH KV region
                // for short divergent "auxiliary" requests (health
                // checks, reply-intent prechecks, slash commands). The
                // single shared main KV is overwritten front-to-back by
                // every request, so a short divergent request between
                // two conversation turns physically corrupts the
                // conversation's cached KV prefix → cold re-prefill.
                // Routing aux requests to this isolated region keeps the
                // main conversation KV + prefix cache intact across the
                // interleaving. Default OFF (allocate only when
                // `RVLLM_GEMMA4_NVFP4_AUX_KV=1`); allocated BELOW the
                // drafter pin so it persists across requests.
                const AUX_KV_MAX_POS: u32 = 16384;
                // Default-ON (2026-05-28, validated): opt-out via
                // `RVLLM_GEMMA4_NVFP4_AUX_KV=0`. Isolating short aux
                // requests is byte-equivalent for the conversation path
                // and only adds a 16k scratch KV region (~small vs the
                // 128k main); the win is keeping the prefix-cache hit
                // alive across interleaved health-checks / prechecks.
                let aux_kv_enabled = std::env::var("RVLLM_GEMMA4_NVFP4_AUX_KV")
                    .map(|s| !matches!(s.as_str(), "0" | "false" | "FALSE"))
                    .unwrap_or(true);
                let kv_aux: Option<_> = if aux_kv_enabled {
                    match bringup.allocate_aux_kv_state(
                        AUX_KV_MAX_POS, AUX_KV_MAX_POS,
                    ) {
                        Ok(k) => {
                            tracing::info!(
                                "gemma4-nvfp4 aux scratch KV allocated \
                                 (max_pos={AUX_KV_MAX_POS}) — short aux \
                                 requests isolated from the conversation KV");
                            Some(k)
                        }
                        Err(e) => {
                            tracing::warn!(
                                "aux scratch KV alloc failed ({e:?}); \
                                 aux requests fall back to main KV");
                            None
                        }
                    }
                } else {
                    None
                };

                // Spec-decode init (#6b-session): allocate the
                // persistent base_last_hidden snapshot buffer,
                // load the drafter checkpoint, and pin the arena
                // top so subsequent per-request scratch_guard
                // rewinds don't reclaim the drafter workspace
                // mid-forward. Done once at worker startup so
                // each request can call the Option B spec session
                // directly.
                // Split-KV decode workspace (long-context decode speedup,
                // gated RVLLM_GEMMA4_NVFP4_SPLIT_DECODE=1). Allocated
                // unconditionally (the single-token decode path it
                // accelerates is used by both spec bailout and non-spec
                // decode), BEFORE the drafter pins the arena top. A
                // failure here is non-fatal — the decode falls back to
                // the single-CTA path.
                if let Err(e) = bringup.ensure_split_decode_workspace(g4n_kv_max_pos) {
                    tracing::warn!(
                        "ensure_split_decode_workspace failed ({e:?}); \
                         long-context decode stays on the single-CTA path");
                }
                if spec_decode {
                    if let Err(e) = bringup.ensure_base_last_hidden_buffer() {
                        let _ = ready_tx.send(Err(format!(
                            "ensure_base_last_hidden_buffer: {e:?}")));
                        return;
                    }
                    if let Err(e) = bringup.ensure_drafter_nvfp4(
                        &spec_drafter_dir, &kv,
                    ) {
                        let _ = ready_tx.send(Err(format!(
                            "ensure_drafter_nvfp4({:?}): {e:?}",
                            spec_drafter_dir)));
                        return;
                    }
                }

                let _ = ready_tx.send(Ok(()));
                tracing::info!(
                    "gemma4-nvfp4 worker ready (Option B native; \
                     max_pos={g4n_kv_max_pos}, vocab={}, hidden={}, \
                     layers={}, spec_decode={spec_decode}, spec_k={}).",
                    bringup.arch.vocab_size,
                    bringup.arch.hidden_size,
                    bringup.arch.num_hidden_layers,
                    spec_cfg.k,
                );

                let spec_zero_accept_trip_reqs = std::env::var(
                    "G4N_SPEC_ZERO_ACCEPT_TRIP_REQUESTS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let spec_zero_accept_skip_reqs = std::env::var(
                    "G4N_SPEC_ZERO_ACCEPT_SKIP_REQUESTS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(16);
                let mut spec_zero_accept_req_streak = 0usize;
                let mut spec_circuit_skip_remaining = 0usize;

                while let Some(req) = req_rx.blocking_recv() {
                    let prompt_len = req.prompt_ids.len() as u32;
                    if req.prompt_ids.is_empty() {
                        let _ = req.events_tx.send(GenerateEvent::Error(
                            "gemma4-nvfp4: empty prompt".to_string()));
                        continue;
                    }
                    if !req.sampling.is_greedy() {
                        let _ = req.events_tx.send(GenerateEvent::Error(
                            "gemma4-nvfp4 path: non-greedy sampling \
                             (temperature>0 / top_p<1 / top_k / seed) \
                             is not yet supported. Set temperature=0 \
                             explicitly, or omit it. Sampling lands once \
                             logits-out + sampler are wired."
                                .to_string()));
                        continue;
                    }
                    // Stream-7: vision is now wired through
                    // `Gemma4Nvfp4Bringup::forward_gemma_vision` + the
                    // residual splice at the top of
                    // `forward_prompt_to_all_tokens_impl`. Audio remains
                    // unwired on Option B (E4B-only path).
                    if !req.audio_items.is_empty() || !req.audio_slots.is_empty() {
                        let _ = req.events_tx.send(GenerateEvent::Error(
                            "gemma4-nvfp4 path: audio is not wired on the \
                             Option B forward (E4B-only). Use the E4B \
                             profile for audio transcription."
                                .to_string()));
                        continue;
                    }
                    if req.cancelled.load(Ordering::Relaxed) {
                        let _ = req.events_tx.send(GenerateEvent::Done {
                            finish: FinishReason::Cancelled,
                            prompt_tokens: prompt_len,
                            completion_tokens: 0,
                        });
                        continue;
                    }
                    // Prompt overflow guard: we own one KV state for
                    // the worker's lifetime (no per-request reset
                    // yet — rollback on completion uses
                    // context_lens-only, but with text-only +
                    // greedy + no spec there's no rollback to do).
                    // Conservative reject when the prompt+decode
                    // would exceed the KV cap.
                    let max_new = req.max_new_tokens.max(1);
                    if (prompt_len as u32) + max_new > g4n_kv_max_pos {
                        let _ = req.events_tx.send(GenerateEvent::Error(
                            format!(
                                "gemma4-nvfp4: prompt_len({prompt_len}) + \
                                 max_new_tokens({max_new}) exceeds \
                                 max_pos({g4n_kv_max_pos}). Raise \
                                 G4N_KV_MAX_POS or shorten the prompt."
                            )));
                        continue;
                    }

                    // Stop tokens: union of the request's
                    // stop_token_ids with the model's EOS (Gemma 4
                    // EOS = 106 per the tokenizer). Empty
                    // stop_token_ids on the request still triggers
                    // EOS-only stop.
                    let stop_set: std::collections::HashSet<u32> =
                        req.stop_token_ids.iter().copied().chain([106u32]).collect();

                    let spec_max_prompt_tokens = std::env::var("G4N_SPEC_MAX_PROMPT_TOKENS")
                        .ok()
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(0);
                    let spec_prompt_allowed =
                        spec_max_prompt_tokens == 0 || prompt_len <= spec_max_prompt_tokens;
                    let spec_probe_this_request = spec_decode
                        && spec_prompt_allowed
                        && spec_circuit_skip_remaining == 0;
                    if spec_decode && spec_circuit_skip_remaining > 0 {
                        spec_circuit_skip_remaining -= 1;
                    }

                    // Stream-7: vision pre-pass. Run the native Gemma 4
                    // ViT for each image, narrow f16 → bf16 host-side,
                    // build the splice list for the prefill. Spec mode
                    // ALSO carries vision now (Stream-7 spec+vision
                    // unblock, 2026-05-23): the drafter cross-attends
                    // to the SAME shadow K/V the prefill populates, so
                    // vision-spliced base tokens propagate to the
                    // drafter for free. No paired drafter ViT needed
                    // because the drafter operates in token space, not
                    // pixel space.
                    // Build vision splice. Outputs from
                    // `forward_gemma_vision` are little-endian f16 in
                    // `out.data` of shape `[num_tokens, hidden]`;
                    // Option B's residual buffer is bf16, so we
                    // narrow f16 → bf16 here (host-side, one pass per
                    // image) before handing the byte chunks to the
                    // splice. Each ViT image already collapses to
                    // ~256 tokens × 5376 hidden = 1.3 M floats so the
                    // narrow is trivially cheap relative to the
                    // device forward.
                    let mut vision_splice_bytes: Vec<(usize, Vec<u8>)> =
                        Vec::with_capacity(req.vision_items.len());
                    let mut vision_failed_msg: Option<String> = None;
                    for (i, item) in req.vision_items.iter().enumerate() {
                        if req.cancelled.load(Ordering::Relaxed) {
                            vision_failed_msg = Some("cancelled".to_string());
                            break;
                        }
                        match bringup.forward_gemma_vision(&item.bytes) {
                            Ok(out) => {
                                tracing::info!(
                                    idx = i,
                                    tokens = out.num_tokens,
                                    hidden = out.hidden_dim,
                                    "vision: Gemma4Nvfp4 ViT forward done"
                                );
                                if out.num_tokens != item.num_tokens {
                                    vision_failed_msg = Some(format!(
                                        "vision tokens mismatch: predicted {} got {}",
                                        item.num_tokens, out.num_tokens));
                                    break;
                                }
                                // f16 → bf16 narrow.
                                let nrows = out.num_tokens;
                                let dim = out.hidden_dim;
                                let n_elems = nrows * dim;
                                let mut bf16_bytes = Vec::with_capacity(n_elems * 2);
                                let src = &out.data;
                                if src.len() != n_elems * 2 {
                                    vision_failed_msg = Some(format!(
                                        "vision data size mismatch: {} vs expected {}",
                                        src.len(), n_elems * 2));
                                    break;
                                }
                                for c in src.chunks_exact(2) {
                                    let bits = u16::from_le_bytes([c[0], c[1]]);
                                    let f = half::f16::from_bits(bits).to_f32();
                                    let bf16 = half::bf16::from_f32(f);
                                    bf16_bytes.extend_from_slice(&bf16.to_le_bytes());
                                }
                                // Use the i-th vision_slot's
                                // token_start (set by tokenizer
                                // chat-template expansion).
                                let slot = req.vision_slots.get(i).copied();
                                let token_start = match slot {
                                    Some(s) => s.token_start,
                                    None => {
                                        vision_failed_msg = Some(
                                            "vision_slots missing slot for image"
                                                .to_string());
                                        break;
                                    }
                                };
                                vision_splice_bytes.push((token_start, bf16_bytes));
                            }
                            Err(e) => {
                                vision_failed_msg = Some(format!(
                                    "gemma4-nvfp4 vision forward: {e:?}"));
                                break;
                            }
                        }
                    }
                    if let Some(msg) = vision_failed_msg {
                        let _ = req.events_tx.send(GenerateEvent::Error(msg));
                        let _ = req.events_tx.send(GenerateEvent::Done {
                            finish: FinishReason::Stop,
                            prompt_tokens: prompt_len,
                            completion_tokens: 0,
                        });
                        continue;
                    }
                    let vision_splice_refs: Vec<(usize, &[u8])> = vision_splice_bytes
                        .iter()
                        .map(|(ts, v)| (*ts, v.as_slice()))
                        .collect();

                    // Spec-decode branch: drive the full request
                    // through the Option B greedy spec loop, then
                    // post-process the emitted tokens for stop tokens.
                    // Cancellation isn't honored inside the session
                    // (single big call); coarse-grained cancel check +
                    // finer-grained cancellation inside the session is
                    // a follow-up.
                    if spec_probe_this_request {
                        let stop_vec: Vec<u32> = stop_set.iter().copied().collect();
                        // Prefix-cache clobber fix: route SHORT DIVERGENT
                        // requests to the scratch KV so they don't corrupt
                        // the conversation's cached KV prefix. Heuristic:
                        // a request is "auxiliary" when a much-longer
                        // conversation is cached (committed > 2×prompt_len)
                        // and this request fits the scratch region. That
                        // matches health-checks / prechecks / slash
                        // commands (tiny, divergent system prompt) but NOT
                        // a fresh conversation turn (prompt ≈ cached size)
                        // nor a continuation (high LCP, large prompt). Only
                        // active when the scratch region was allocated
                        // (`RVLLM_GEMMA4_NVFP4_AUX_KV=1`); aux_mode then
                        // suppresses cache lookup/publish for the call.
                        let plen = req.prompt_ids.len() as u32;
                        let committed = bringup.nvfp4_prefix_cache_committed_len();
                        let is_aux = kv_aux.is_some()
                            && plen + max_new < AUX_KV_MAX_POS
                            && committed > plen.saturating_mul(2);
                        let kv_ref = if is_aux { kv_aux.as_ref().unwrap() } else { &kv };
                        if is_aux {
                            bringup.set_nvfp4_aux_mode(true);
                            tracing::debug!(
                                "gemma4-nvfp4 aux-route: plen={plen} \
                                 committed={committed} → scratch KV");
                        }
                        // Cooperative cancel: hand the request's cancel
                        // flag to the spec session so a provider/client
                        // timeout (or disconnect) lets it bail at the
                        // next iteration boundary instead of running the
                        // whole reply uncancellable and wedging the
                        // single-in-flight queue. Cleared right after.
                        bringup.set_nvfp4_cancel(Some(std::sync::Arc::clone(&req.cancelled)));
                        // Stream-7 spec+vision: route the same
                        // vision_splice_refs the non-spec branch uses
                        // into the spec prefill. Empty slice reduces
                        // to the text-only path byte-identically.
                        let session_result = if spec_cfg.k == 1 {
                            bringup.run_spec_session_nvfp4_greedy_k1_with_vision(
                                &req.prompt_ids, max_new as usize,
                                &stop_vec, kv_ref, &vision_splice_refs)
                        } else {
                            bringup.run_spec_session_nvfp4_greedy_k_with_vision(
                                &req.prompt_ids, max_new as usize,
                                spec_cfg.k as usize, &stop_vec, kv_ref,
                                &vision_splice_refs)
                        };
                        bringup.set_nvfp4_cancel(None);
                        if is_aux {
                            bringup.set_nvfp4_aux_mode(false);
                        }
                        let stats = match session_result {
                            Ok(s) => s,
                            Err(e) => {
                                let _ = req.events_tx.send(GenerateEvent::Error(
                                    format!(
                                        "gemma4-nvfp4 spec session K={}: \
                                         {e:?}",
                                        spec_cfg.k)));
                                continue;
                            }
                        };
                        let mut completion_tokens: u32 = 0;
                        let mut finish = FinishReason::Length;
                        for (i, &t) in stats.emitted.iter().enumerate() {
                            let _ = req.events_tx.send(GenerateEvent::Token {
                                id: t, position: i as u32,
                            });
                            completion_tokens += 1;
                            if stop_set.contains(&t) {
                                finish = FinishReason::Stop;
                                break;
                            }
                        }
                        if std::env::var("RVLLM_GEMMA4_SPEC_PERF_TRACE")
                            .as_deref() == Ok("1")
                        {
                            tracing::debug!(
                                "gemma4-nvfp4 spec-session: prompt={} \
                                 emitted={} iters={} accepted={} \
                                 accept_rate={:.3}",
                                prompt_len, stats.emitted.len(),
                                stats.n_iters, stats.n_accepted,
                                if stats.n_iters > 0 {
                                    stats.n_accepted as f32
                                        / stats.n_iters as f32
                                } else { 0.0 },
                            );
                        }
                        if stats.n_iters > 0 && stats.n_accepted == 0 {
                            spec_zero_accept_req_streak += 1;
                        } else {
                            spec_zero_accept_req_streak = 0;
                        }
                        if spec_zero_accept_trip_reqs > 0
                            && spec_zero_accept_skip_reqs > 0
                            && spec_zero_accept_req_streak
                                >= spec_zero_accept_trip_reqs
                        {
                            tracing::warn!(
                                "gemma4-nvfp4 spec circuit open: \
                                 {} consecutive zero-accept requests; \
                                 routing next {} requests through \
                                 non-spec decode",
                                spec_zero_accept_req_streak,
                                spec_zero_accept_skip_reqs,
                            );
                            spec_zero_accept_req_streak = 0;
                            spec_circuit_skip_remaining =
                                spec_zero_accept_skip_reqs;
                        }
                        let _ = req.events_tx.send(GenerateEvent::Done {
                            finish, prompt_tokens: prompt_len,
                            completion_tokens,
                        });
                        continue;
                    }

                    // Non-spec branch (existing): prefill all
                    // prompt tokens at position 0. The first
                    // generation token is what
                    // forward_prompt_to_token returns (argmax of
                    // the last prompt token's final residual).
                    // Stream-7: vision splice (empty for text-only)
                    // overwrites the image-pad rows in residual_dev
                    // between embed and the layer loop.
                    let next_first = match bringup.forward_prompt_to_token_with_vision(
                        &req.prompt_ids, 0, &kv, &vision_splice_refs,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            let _ = req.events_tx.send(GenerateEvent::Error(
                                format!("forward_prompt_to_token: {e:?}")));
                            continue;
                        }
                    };
                    let _ = req.events_tx.send(GenerateEvent::Token {
                        id: next_first,
                        position: 0,
                    });

                    let mut completion_tokens: u32 = 1;
                    let mut finish = FinishReason::Length;
                    // CUDA-Graph foundation (step toward goal #2): the
                    // unified-NVFP4-prefill path uses a device-pointer
                    // positions buffer (`positions_dev` in
                    // gemma4_nvfp4_bring_up.rs's batched-prefill body),
                    // so routing single-token decode through
                    // `forward_prompt_to_token([token], pos, kv)` is
                    // graph-replay-friendly by design — the only
                    // per-iter input that's NOT a stable device
                    // pointer is the embedded token row, which we can
                    // shadow into a stable buffer in a follow-up.
                    // Enable for A/B with `G4N_DECODE_VIA_UNIFIED=1`.
                    // Defaults OFF until the equivalence + perf A/B
                    // lands a clear win.
                    let decode_via_unified = std::env::var("G4N_DECODE_VIA_UNIFIED")
                        .ok().as_deref() == Some("1");
                    // Device-only argmax path: skips the mid-forward
                    // sync DtoH of the residual back to host that
                    // blocks `cuStreamBeginCapture` recording. The
                    // host result extraction happens AFTER the
                    // device pipeline via a single sync fence + DtoH.
                    let decode_device_argmax = std::env::var("G4N_DECODE_DEVICE_ARGMAX")
                        .ok().as_deref() == Some("1");
                    // Final cuStreamBeginCapture wrap entry. When
                    // set, decode iter 1 captures the device-only
                    // forward, iter 2+ replays via cuGraphLaunch.
                    // Requires `G4N_DECODE_GRAPH_INDIRECT=1` co-set
                    // (position must reach kernels via stable
                    // device pointer for replay correctness).
                    let decode_graph_replay = std::env::var("G4N_DECODE_GRAPH_REPLAY")
                        .ok().as_deref() == Some("1");
                    if stop_set.contains(&next_first) {
                        finish = FinishReason::Stop;
                    } else {
                        // Decode loop: feed last token at the next
                        // position, get its successor, emit, until
                        // EOS / stop / max_new / cancellation.
                        let mut last_token = next_first;
                        for step in 1..max_new {
                            if req.cancelled.load(Ordering::Relaxed) {
                                finish = FinishReason::Cancelled;
                                break;
                            }
                            let position = prompt_len + (step - 1);
                            let single = [last_token];
                            let res = if decode_graph_replay {
                                // Final cuStreamBeginCapture wrap:
                                // captures iter 1, replays iter 2+.
                                // Per-iter ~660 cuLaunchKernel host
                                // dispatches collapse into a single
                                // cuGraphLaunch.
                                bringup.forward_full_to_token_captured(
                                    last_token, position, &kv,
                                )
                            } else if decode_device_argmax {
                                // Step 2/4 of the cuGraphLaunch wrap.
                                // Run the per-token forward fully on
                                // device, then extract argmax via a
                                // single sync fence + DtoH at the end.
                                // This is the body the captured graph
                                // will eventually record (the
                                // argmax_dev_to_host_token call lives
                                // OUTSIDE the recorded region).
                                match bringup.forward_full_to_token_device_argmax(
                                    last_token, position, &kv,
                                ) {
                                    Ok(dev_ptr) => bringup.argmax_dev_to_host_token(dev_ptr),
                                    Err(e) => Err(e),
                                }
                            } else if decode_via_unified {
                                // Same path as the multi-token prefill,
                                // exercised at N=1. Slower per-iter
                                // today (unified kernel is tuned for
                                // batched N), but device-pointer
                                // shape stability is the prerequisite
                                // for cuGraphLaunch replay.
                                bringup.forward_prompt_to_token(
                                    &single, position, &kv,
                                )
                            } else {
                                bringup.forward_full_to_token(
                                    last_token, position, &kv,
                                )
                            };
                            let next = match res {
                                Ok(t) => t,
                                Err(e) => {
                                    let _ = req.events_tx.send(GenerateEvent::Error(
                                        format!("decode: {e:?}")));
                                    break;
                                }
                            };
                            let _ = req.events_tx.send(GenerateEvent::Token {
                                id: next,
                                position: step,
                            });
                            completion_tokens += 1;
                            if stop_set.contains(&next) {
                                finish = FinishReason::Stop;
                                break;
                            }
                            last_token = next;
                        }
                    }
                    let _ = req.events_tx.send(GenerateEvent::Done {
                        finish,
                        prompt_tokens: prompt_len,
                        completion_tokens,
                    });
                    // No arena restore — Option B's forward methods
                    // already use ForwardScratchGuard per call which
                    // rewinds to the post-allocate_kv_state
                    // checkpoint. KV state itself persists for the
                    // worker's lifetime; commit 2 of codex's
                    // Stream-5+6+7 plan (BaseKvSource) will
                    // generalize this for the spec rollback path.
                }
                return;
            }

            let qwen_probe = match family {
                ModelFamily::Gemma4 | ModelFamily::Qwen35 | ModelFamily::Gemma4Nvfp4 => Ok(None),
                ModelFamily::Qwen36 => {
                    rvllm_runtime::qwen36_arch::Qwen36Arch::from_dir(&paths.model_dir)
                        .and_then(|opt| match opt {
                            Some(a) => Ok(Some(a)),
                            None => {
                                use rvllm_core::{LoaderCtx, LoaderError, RvllmError};
                                Err(RvllmError::Loader {
                                    err: LoaderError::Corrupt {
                                        detail:
                                            "operator forced --model-family=qwen36 but config.json \
                                             does not match the Qwen 3.6 markers"
                                                .into(),
                                    },
                                    ctx: LoaderCtx {
                                        path: paths.model_dir.join("config.json"),
                                        tensor: None,
                                    },
                                    bt: std::backtrace::Backtrace::capture(),
                                })
                            }
                        })
                }
                ModelFamily::Auto | ModelFamily::Mistral35 => {
                    rvllm_runtime::qwen36_arch::Qwen36Arch::from_dir(&paths.model_dir)
                }
            };
            match qwen_probe {
                Ok(Some(_)) => {
                    let qwen = match rvllm_runtime::qwen36_bring_up::Qwen36Bringup::load(
                        paths,
                        arena_bytes,
                    ) {
                        Ok(b) => b,
                        Err(e) => {
                            let _ = ready_tx.send(Err(format!(
                                "Qwen36Bringup::load: {e:?}"
                            )));
                            return;
                        }
                    };
                    // Phase 8 follow-on (graph-cache reuse):
                    // allocate the workspace ONCE at worker
                    // bring-up — BEFORE `scratch_ck` is taken —
                    // so its device pointers stay stable across
                    // every request. The captured decode-step
                    // graph holds those pointers; with the
                    // workspace persistent + decode_inner taking
                    // its own internal checkpoint+restore (added
                    // alongside this commit), every per-call
                    // arena region inside decode_inner lands at
                    // the same address on every call, every
                    // request. The captured graph from request N
                    // is therefore valid for request N+1 — no
                    // re-capture needed.
                    //
                    // Allocated unconditionally (env-independent):
                    // ~800KB overhead on a 65GB arena is trivial,
                    // and the env check decides whether to USE
                    // the workspace, not whether to allocate it.
                    let persistent_decode_workspace =
                        match qwen.alloc_decode_workspace() {
                            Ok(w) => Some(w),
                            Err(e) => {
                                tracing::warn!(
                                    "qwen36 persistent workspace alloc \
                                     failed: {e:?}. DECODE_GRAPH/WORKSPACE \
                                     paths unavailable for this worker."
                                );
                                None
                            }
                        };
                    let scratch_ck = qwen.arena.checkpoint();
                    tracing::info!(
                        "qwen36 cuda worker ready (Phase 5d: \
                         linear-attn layout rewritten per vLLM \
                         qwen3_next reference — Q[16,128]+K[16,128]+ \
                         V[32,128] split, L2-norm on Q/K, in_proj_a/b \
                         for α/β, GQA expansion for state update)"
                    );
                    let _ = ready_tx.send(Ok(()));

                    // Phase 5c request loop: prefill the prompt
                    // (state + KV accumulate causally), then decode
                    // step-by-step up to max_new_tokens, with EOS /
                    // stop-token early-exit. arena.restore between
                    // requests keeps memory bounded; the persistent
                    // linear-state + KV cache regions live ABOVE the
                    // scratch checkpoint so they survive that restore
                    // and only get reset by the explicit calls below.
                    while let Some(req) = req_rx.blocking_recv() {
                        // GB10 belt-and-suspenders: rebind the retained
                        // primary CUDA context to this worker thread once
                        // per request. `Qwen36Bringup::load` already
                        // bound it during init, but a context binding that
                        // sat idle across `blocking_recv` has been
                        // observed to drop on GB10, producing
                        // `cuLaunchKernel` failures on the next launch.
                        // One `cuCtxSetCurrent` per request is free next
                        // to decode cost. See
                        // rvllm-mem/src/context.rs:bind_to_current_thread.
                        if let Err(e) = qwen.ctx.bind_to_current_thread() {
                            let _ = req.events_tx.send(GenerateEvent::Error(
                                format!("qwen36 ctx rebind: {e:?}"),
                            ));
                            continue;
                        }
                        let prompt_len = req.prompt_ids.len() as u32;
                        // Qwen 3.6 path is greedy-only today: the
                        // forward_qwen36_decode runtime samples internally
                        // (returns the picked token, not logits). We reject
                        // non-greedy sampling explicitly so callers can't
                        // believe their `temperature`/`top_p`/`top_k`/`seed`
                        // were honoured. Lift the rejection when Qwen's
                        // bring-up grows a logits-out variant + sampler.
                        if !req.sampling.is_greedy() {
                            // Note: the absent-temperature default now
                            // resolves to 0.0 (greedy) globally; if the
                            // client lands here they explicitly asked
                            // for stochastic. The old hint mentioned
                            // "omit sampling params" — that's exactly
                            // the path that USED to fail; corrected.
                            let _ = req.events_tx.send(GenerateEvent::Error(
                                "qwen36 path: non-greedy sampling \
                                 (temperature>0 / top_p<1 / top_k / seed) \
                                 is not yet supported on Qwen 3.6. Set \
                                 temperature=0 explicitly, or omit it to \
                                 take the (now greedy) default."
                                    .to_string(),
                            ));
                            continue;
                        }
                        // Reset per-request transient state. cuMemsetD8Async
                        // can return real CUDA errors (e.g. context lost
                        // after a kernel fault); silently swallowing them
                        // would make the next request run on stale state.
                        let reset_ok = qwen.reset_linear_state()
                            .and_then(|_| qwen.reset_kv_cache())
                            .and_then(|_| qwen.reset_conv_state());
                        if let Err(e) = reset_ok {
                            let _ = req.events_tx.send(GenerateEvent::Error(
                                format!("qwen36 per-request reset: {e:?}"),
                            ));
                            unsafe { qwen.arena.restore(scratch_ck); }
                            continue;
                        }
                        if req.cancelled.load(Ordering::Relaxed) {
                            let _ = req.events_tx.send(GenerateEvent::Done {
                                finish: FinishReason::Cancelled,
                                prompt_tokens: prompt_len,
                                completion_tokens: 0,
                            });
                            unsafe { qwen.arena.restore(scratch_ck); }
                            continue;
                        }
                        let prompt_i32: Vec<i32> = req
                            .prompt_ids
                            .iter()
                            .map(|&t| t as i32)
                            .collect();

                        // Vision pre-pass: run native ViT forward on
                        // each image. Outputs accumulate per slot for
                        // splicing during the prefill embed step.
                        let mut vision_outputs: Vec<rvllm_runtime::qwen36_bring_up::VisionForwardOutput> =
                            Vec::with_capacity(req.vision_items.len());
                        let mut vision_failed = false;
                        for (i, item) in req.vision_items.iter().enumerate() {
                            if req.cancelled.load(Ordering::Relaxed) {
                                vision_failed = true;
                                break;
                            }
                            let vit_t0 = std::time::Instant::now();
                            match qwen.forward_qwen_vision(&item.bytes) {
                                Ok(out) => {
                                    if out.num_tokens != item.num_tokens {
                                        let _ = req.events_tx.send(
                                            GenerateEvent::Error(format!(
                                                "vision tokens mismatch: predicted {} got {}",
                                                item.num_tokens, out.num_tokens
                                            )),
                                        );
                                        vision_failed = true;
                                        break;
                                    }
                                    let vit_ms = vit_t0.elapsed().as_secs_f64() * 1000.0;
                                    tracing::info!(
                                        idx = i,
                                        tokens = out.num_tokens,
                                        hidden = out.hidden_dim,
                                        vit_ms = format!("{vit_ms:.1}"),
                                        "vision: ViT forward done"
                                    );
                                    vision_outputs.push(out);
                                }
                                Err(e) => {
                                    let _ = req.events_tx.send(GenerateEvent::Error(
                                        format!("vision forward: {e:?}"),
                                    ));
                                    vision_failed = true;
                                    break;
                                }
                            }
                        }
                        if vision_failed {
                            unsafe { qwen.arena.restore(scratch_ck); }
                            continue;
                        }

                        // Prefill: feed full prompt at start_position=0.
                        // For each vision slot, build (token_start,
                        // embedding bytes) tuple — the splice happens
                        // inside forward_qwen36_decode after embed_gather.
                        let vision_splice: Vec<(usize, &[u8])> = req
                            .vision_slots
                            .iter()
                            .map(|s| {
                                (s.token_start, vision_outputs[s.vision_item_idx].data.as_slice())
                            })
                            .collect();
                        let timing_on = std::env::var("RVLLM_QWEN36_TIMING")
                            .map(|s| matches!(s.as_str(), "1"|"true"|"TRUE"|"yes"))
                            .unwrap_or(false);
                        let prefill_start = if timing_on {
                            Some(std::time::Instant::now())
                        } else { None };
                        // RVLLM_QWEN36_SPEC_DECODE=1 → prompt-lookup
                        // speculative decoding (no drafter checkpoint
                        // needed). The session function owns the
                        // initial prefill + decode loop and emits each
                        // accepted token via the on_token callback,
                        // matching the cuda_worker SSE contract.
                        // Bypasses the per-token loop entirely.
                        let spec_decode_on = std::env::var("RVLLM_QWEN36_SPEC_DECODE")
                            .map(|s| matches!(s.as_str(), "1"|"true"|"TRUE"|"yes"))
                            .unwrap_or(false);
                        let spec_decode_on = if spec_decode_on {
                            let min_prompt_tokens = std::env::var(
                                "RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS",
                            )
                                .ok()
                                .and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(0);
                            let min_full_draft_hits = std::env::var(
                                "RVLLM_QWEN36_SPEC_PREFLIGHT_MIN_FULL_DRAFTS",
                            )
                                .ok()
                                .and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(0);
                            let min_max_new_tokens = std::env::var(
                                "RVLLM_QWEN36_SPEC_MIN_MAX_NEW_TOKENS",
                            )
                                .ok()
                                .and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(0);
                            let spec_k = std::env::var("RVLLM_QWEN36_SPEC_K")
                                .ok()
                                .and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(4)
                                .max(1);
                            let ngram = std::env::var("RVLLM_QWEN36_SPEC_NGRAM")
                                .ok()
                                .and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(2)
                                .max(1);

                            let prompt_len_usize = prompt_i32.len();
                            let requested_max_new = req.max_new_tokens.max(1) as usize;
                            let enough_prompt = prompt_len_usize >= min_prompt_tokens;
                            let enough_decode_len = requested_max_new >= min_max_new_tokens;
                            let enough_full_draft_hits = if min_full_draft_hits == 0 {
                                true
                            } else if prompt_len_usize < ngram + spec_k + 1 {
                                false
                            } else if ngram == 2 {
                                let mut seen = std::collections::HashSet::<(i32, i32)>::new();
                                let mut hits = 0usize;
                                let last_start = prompt_len_usize - ngram - spec_k;
                                for start in 0..=last_start {
                                    let key = (prompt_i32[start], prompt_i32[start + 1]);
                                    if !seen.insert(key) {
                                        hits += 1;
                                        if hits >= min_full_draft_hits {
                                            break;
                                        }
                                    }
                                }
                                hits >= min_full_draft_hits
                            } else {
                                let mut seen = std::collections::HashSet::<Vec<i32>>::new();
                                let mut hits = 0usize;
                                let last_start = prompt_len_usize - ngram - spec_k;
                                for start in 0..=last_start {
                                    let key = prompt_i32[start..start + ngram].to_vec();
                                    if !seen.insert(key) {
                                        hits += 1;
                                        if hits >= min_full_draft_hits {
                                            break;
                                        }
                                    }
                                }
                                hits >= min_full_draft_hits
                            };
                            let enabled =
                                enough_prompt && enough_decode_len && enough_full_draft_hits;
                            if !enabled && std::env::var("RVLLM_QWEN36_SPEC_PERF_TRACE")
                                .as_deref() == Ok("1")
                            {
                                tracing::debug!(
                                    prompt_tokens = prompt_len_usize,
                                    requested_max_new,
                                    min_prompt_tokens,
                                    min_max_new_tokens,
                                    min_full_draft_hits,
                                    "qwen36 spec preflight skipped; using native base decode",
                                );
                            }
                            enabled
                        } else {
                            false
                        };
                        let qwen36_repetition_guard_n = std::env::var(
                            "RVLLM_QWEN36_REPETITION_GUARD_N",
                        )
                            .ok()
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(0);
                        if spec_decode_on {
                            let max_new = req.max_new_tokens.max(1);
                            let events_tx = req.events_tx.clone();
                            let cancelled_for_cb = req.cancelled.clone();
                            let stop_tokens = req.stop_token_ids.clone();
                            let mut emitted_ids: Vec<u32> =
                                Vec::with_capacity(max_new as usize);
                            let result = rvllm_runtime::qwen36_spec_decode::
                                run_qwen36_prompt_lookup_spec(
                                    &qwen,
                                    &prompt_i32,
                                    max_new,
                                    &stop_tokens,
                                    &vision_splice,
                                    Some(&*req.cancelled),
                                    |tok, pos| {
                                        let _ = events_tx.send(GenerateEvent::Token {
                                            id: tok,
                                            position: pos,
                                        });
                                        emitted_ids.push(tok);
                                        if qwen36_repetition_guard_n >= 2
                                            && emitted_ids.len() >= qwen36_repetition_guard_n
                                        {
                                            let tail = &emitted_ids[
                                                emitted_ids.len() - qwen36_repetition_guard_n..
                                            ];
                                            if tail.iter().all(|&id| id == tail[0]) {
                                                tracing::warn!(
                                                    token_id = tail[0],
                                                    guard_n = qwen36_repetition_guard_n,
                                                    "qwen36 repetition guard stopped generation",
                                                );
                                                return false;
                                            }
                                        }
                                        !cancelled_for_cb.load(Ordering::Relaxed)
                                    },
                                );
                            unsafe { qwen.arena.restore(scratch_ck); }
                            match result {
                                Ok((completion_tokens, finish_label)) => {
                                    let finish = match finish_label {
                                        "stop" => FinishReason::Stop,
                                        "cancelled" => FinishReason::Cancelled,
                                        _ => FinishReason::Length,
                                    };
                                    let _ = req.events_tx.send(GenerateEvent::Done {
                                        finish,
                                        completion_tokens,
                                        prompt_tokens: prompt_len,
                                    });
                                }
                                Err(e) => {
                                    let _ = req.events_tx.send(GenerateEvent::Error(
                                        format!("qwen36 spec-decode: {e:?}"),
                                    ));
                                }
                            }
                            continue;
                        }

                        let mut next_token = match qwen
                            .forward_qwen36_decode_cancellable(
                                &prompt_i32, 0, &vision_splice,
                                Some(&*req.cancelled),
                            )
                        {
                            Ok(t) => t,
                            Err(e) => {
                                let _ = req.events_tx.send(GenerateEvent::Error(
                                    format!("qwen36 prefill: {e:?}"),
                                ));
                                unsafe { qwen.arena.restore(scratch_ck); }
                                continue;
                            }
                        };

                        // Phase 8 follow-up wiring: workspace alloc
                        // + dispatch. Two operator gates:
                        //   * `RVLLM_QWEN36_DECODE_GRAPH=1` — full
                        //     graph-capture path via
                        //     `decode_step_via_graph_or_eager`.
                        //   * `RVLLM_QWEN36_DECODE_WORKSPACE=1` —
                        //     workspace-eager ONLY (no capture
                        //     attempted). Isolates whether the
                        //     workspace path itself is correct across
                        //     multi-step decode, independent of the
                        //     capture machinery. Set automatically when
                        //     DECODE_GRAPH is on; can be set
                        //     standalone for debug.
                        let decode_graph_on =
                            rvllm_runtime::qwen36_decode_workspace
                                ::qwen36_decode_graph_enabled();
                        let decode_workspace_only =
                            std::env::var("RVLLM_QWEN36_DECODE_WORKSPACE")
                                .ok()
                                .map(|s| matches!(s.as_str(),
                                    "1"|"true"|"TRUE"|"yes"|"on"))
                                .unwrap_or(false);
                        let need_workspace =
                            decode_graph_on || decode_workspace_only;
                        // Phase 8 deeper: when multi-step env value
                        // changes mid-process OR the workspace's
                        // arena layout shifts, the cached multi-step
                        // macro-graph could go stale. Since we now
                        // keep workspace persistent + decode_inner
                        // takes an inner checkpoint, addresses are
                        // stable across requests — so we DON'T need
                        // to clear per request. But keep the clear
                        // method available for ops that change
                        // multi-step N at runtime (no current
                        // caller). The single-step
                        // `decode_capture` already enjoys
                        // cross-request reuse (commit bcdce94).

                        // Phase 8 follow-on: reuse the
                        // worker-persistent workspace (allocated
                        // above scratch_ck at worker bring-up).
                        // The captured graph carries over across
                        // requests since the workspace pointers
                        // and the decode_inner per-call arena
                        // addresses (stable via the inner
                        // checkpoint+restore added alongside this
                        // commit) are stable end-to-end. No
                        // per-request workspace allocation, no
                        // per-request `clear_decode_capture`.
                        let decode_workspace = if need_workspace {
                            persistent_decode_workspace.as_ref().copied()
                        } else { None };
                        if let Some(t0) = prefill_start {
                            let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            // Round-27d: gates are default-ON post-audit;
                            // log the RESOLVED state (env-var=0/false → off,
                            // anything else including "unset" → on) so the
                            // log reflects what the runtime actually
                            // dispatched, not just the literal env string.
                            let resolved = |name: &str| -> &'static str {
                                match std::env::var(name) {
                                    Ok(s) if matches!(s.as_str(),
                                        "0"|"false"|"FALSE"|"no") => "off",
                                    _ => "on",
                                }
                            };
                            tracing::info!(
                                prompt_tokens = prompt_len,
                                prefill_ms = format!("{dt_ms:.2}"),
                                batch_linear = resolved("RVLLM_QWEN36_BATCH_LINEAR_PREFILL"),
                                batch_full = resolved("RVLLM_QWEN36_BATCH_FULL_PREFILL"),
                                batch_moe = resolved("RVLLM_QWEN36_BATCH_MOE_PREFILL"),
                                batch_moe_routed_ffn = resolved("RVLLM_QWEN36_BATCH_MOE_ROUTED_FFN"),
                                batch_moe_shared = resolved("RVLLM_QWEN36_BATCH_MOE_SHARED"),
                                "[qwen36-timing] prefill done",
                            );
                        }
                        let mut completion_tokens: u32 = 0;
                        let mut finish = FinishReason::Length;
                        let max_new = req.max_new_tokens.max(1);
                        let mut emitted_ids: Vec<u32> = Vec::with_capacity(max_new as usize);
                        // Phase 8 deeper-optimization: multi-step
                        // macro-replay. When
                        // `RVLLM_QWEN36_DECODE_MULTI_STEP=<N>` is
                        // set (and N>1) AND the workspace is live,
                        // each macro-replay returns N tokens at
                        // once. `multi_step_buf` is the FIFO of
                        // pre-fetched tokens (reversed so we pop
                        // from the back in O(1)). When empty, we
                        // top it up with the next macro-block.
                        let multi_step_n = rvllm_runtime::qwen36_decode_workspace
                            ::qwen36_decode_multi_step_n();
                        let multi_step_active = multi_step_n > 1
                            && decode_workspace.is_some()
                            && decode_graph_on;
                        let mut multi_step_buf: Vec<i32> = Vec::new();
                        for step in 0..max_new {
                            let id = if next_token < 0 { 0u32 } else { next_token as u32 };
                            // Round-20 finding #2: stop-token check BEFORE
                            // emit + counter bump, so the EOS / stop
                            // token never reaches the SSE consumer and
                            // `completion_tokens` doesn't include it
                            // (mirrors mock-worker semantics; previously
                            // qwen leaked the stop token both ways).
                            if req.stop_token_ids.contains(&id) {
                                finish = FinishReason::Stop;
                                break;
                            }
                            let _ = req.events_tx.send(GenerateEvent::Token {
                                id,
                                position: prompt_len + step,
                            });
                            emitted_ids.push(id);
                            completion_tokens += 1;
                            if qwen36_repetition_guard_n >= 2
                                && emitted_ids.len() >= qwen36_repetition_guard_n
                            {
                                let tail = &emitted_ids[
                                    emitted_ids.len() - qwen36_repetition_guard_n..
                                ];
                                if tail.iter().all(|&id| id == tail[0]) {
                                    tracing::warn!(
                                        token_id = tail[0],
                                        guard_n = qwen36_repetition_guard_n,
                                        "qwen36 repetition guard stopped generation",
                                    );
                                    break;
                                }
                            }
                            if req.cancelled.load(Ordering::Relaxed) {
                                finish = FinishReason::Cancelled;
                                break;
                            }
                            if step + 1 >= max_new {
                                break;
                            }
                            // Decode step: feed just this token at
                            // start_position = prompt_len + step.
                            let pos = prompt_len + step;
                            let step_res = if multi_step_active {
                                // Pop from the pre-fetched batch;
                                // when empty, top up with the next
                                // macro-block of N tokens.
                                let ws = decode_workspace.as_ref().unwrap();
                                if multi_step_buf.is_empty() {
                                    let remaining = max_new - step;
                                    let n = multi_step_n.min(remaining);
                                    match qwen.decode_steps_n_via_graph_or_eager(
                                        ws, next_token as i32, pos, n)
                                    {
                                        Ok(mut toks) => {
                                            // Reverse so we pop from
                                            // the back in O(1).
                                            toks.reverse();
                                            multi_step_buf = toks;
                                        }
                                        Err(e) => {
                                            let _ = req.events_tx.send(
                                                GenerateEvent::Error(format!(
                                                    "qwen36 multi-step replay: {e:?}"
                                                )));
                                            break;
                                        }
                                    }
                                }
                                Ok(multi_step_buf.pop().unwrap_or(0))
                            } else if let Some(ws) = decode_workspace.as_ref() {
                                if decode_graph_on {
                                    qwen.decode_step_via_graph_or_eager(
                                        ws, next_token as i32, pos)
                                        .map(|t| t as i32)
                                } else {
                                    // Workspace-eager isolation mode:
                                    // write token + position to workspace
                                    // device slots, run the workspace-driven
                                    // step, extract via
                                    // argmax_dev_to_host_token. NO capture,
                                    // NO replay.
                                    qwen.write_token_to_workspace(ws, next_token as i32)
                                        .and_then(|()|
                                            qwen.write_position_to_workspace(ws, pos))
                                        .and_then(|()|
                                            qwen.forward_qwen36_decode_step_to_workspace(ws, pos))
                                        .and_then(|()| qwen.argmax_dev_to_host_token(
                                            ws.argmax_token_dev))
                                }
                            } else {
                                qwen.forward_qwen36_decode(
                                    &[next_token], pos, &[])
                            };
                            match step_res {
                                Ok(t) => {
                                    if std::env::var("RVLLM_QWEN36_DECODE_DEBUG_TOKENS").as_deref()
                                        == Ok("1")
                                    {
                                        tracing::info!(
                                            step, pos, token = t, "qwen36 decode step token"
                                        );
                                    }
                                    next_token = t;
                                }
                                Err(e) => {
                                    let _ = req.events_tx.send(GenerateEvent::Error(
                                        format!("qwen36 decode step {step}: {e:?}"),
                                    ));
                                    finish = FinishReason::Stop;
                                    break;
                                }
                            }
                        }
                        let _ = req.events_tx.send(GenerateEvent::Done {
                            finish,
                            prompt_tokens: prompt_len,
                            completion_tokens,
                        });
                        unsafe { qwen.arena.restore(scratch_ck); }
                    }
                    tracing::info!("qwen36 cuda worker queue closed, exiting");
                    return;
                }
                Ok(None) => {
                    // Not Qwen 3.6 — fall through to Gemma 4 path.
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(format!(
                        "Qwen36Arch::from_dir: {e:?}"
                    )));
                    return;
                }
            }

            // Bring-up on the worker thread so Gemma4Bringup (which
            // contains !Send CUDA state like streams) never crosses
            // a thread boundary.
            let bringup = match Gemma4Bringup::load(paths, arena_bytes) {
                Ok(b) => b,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("Gemma4Bringup::load: {e:?}")));
                    return;
                }
            };

            // Resolve kernel function pointers once. See the struct
            // doc — the LoadedModule MUST outlive the KernelFn.
            let kernels_ctx = match resolve_generate_kernels(&bringup) {
                Ok(k) => k,
                Err(msg) => {
                    let _ = ready_tx.send(Err(msg));
                    return;
                }
            };

            // Snapshot the arena's bump pointer here — everything
            // after this point (scratch regions allocated inside
            // `run_generate`) will be released back to this mark at
            // the end of each request. Without this the bump pointer
            // grows monotonically across requests and we'd hit
            // `HbmArena::region AllocFailed` after a handful of
            // calls (each `run_generate` allocates ~a dozen named
            // scratch regions sized for max_tokens + KV cache).
            //
            // Initialise the session-level prefix cache BEFORE the
            // scratch checkpoint so the persistent KV region sits
            // below the checkpoint and survives every
            // `arena.restore(scratch_ck)` between requests. This is
            // what makes the MVP vLLM-style prefix reuse work on
            // zeroclaw's "identical 15k-token persona every request"
            // pattern.
            // === DIAGNOSTIC: RVLLM_DISABLE_PREFIX_CACHE gate ===
            // When set, skip prefix-cache initialization entirely so
            // every request goes through the per-call fallback KV
            // allocation path (full prefill, no cross-request reuse).
            // Used to discriminate "prefix-cache reuse causes the
            // R1≠R2 divergence" from "deeper non-determinism." Remove
            // after the prefix-cache hypothesis is settled.
            let disable_prefix_cache = env_truthy("RVLLM_DISABLE_PREFIX_CACHE");
            if disable_prefix_cache {
                tracing::warn!("RVLLM_DISABLE_PREFIX_CACHE=1 — skipping init_prefix_cache (diagnostic mode)");
            } else {
                if let Err(e) = bringup.init_prefix_cache() {
                    let _ = ready_tx.send(Err(format!("init_prefix_cache: {e:?}")));
                    return;
                }
            }
            // === END DIAGNOSTIC ===

            // Spec-decode resident state must be allocated before the
            // scratch checkpoint. If the lazy upload happened inside a
            // request after `scratch_ck`, the worker's post-request
            // `arena.restore(scratch_ck)` would mark the drafter weights
            // as free while `bringup.drafter` still holds their device
            // pointers. That is the same lifetime contract as the
            // prefix-cache / hadamard / shadow allocations below.
            if spec_decode {
                if spec_k == 0 {
                    let _ = ready_tx.send(Err(
                        "RVLLM_GEMMA4_SPEC_K must be >= 1".to_string(),
                    ));
                    return;
                }
                match bringup.ensure_drafter(&spec_drafter_dir) {
                    Ok(()) => {
                        tracing::info!(
                            drafter_dir = %spec_drafter_dir.display(),
                            spec_k,
                            "Gemma 4 speculative drafter uploaded above scratch checkpoint",
                        );
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!(
                            "ensure_drafter({}): {e:?}",
                            spec_drafter_dir.display()
                        )));
                        return;
                    }
                }
            }

            // Pre-allocate persistent NVFP4 helper buffers BEFORE the
            // scratch checkpoint. The lazy allocation paths inside
            // `run_generate` had a fatal lifetime bug: they ran
            // AFTER the checkpoint, so `arena.restore(scratch_ck)`
            // at the end of every request marked the buffers' bytes
            // as free, and the SAME pointer in
            // `Gemma4Bringup::nvfp4_hadamard` (or `_shadow`) on the
            // next request silently aliased fresh scratch
            // allocations. With `RVLLM_NVFP4_HADAMARD=1` enabled in
            // production this could corrupt every Nth request's
            // attention math without crashing — the worst kind of
            // failure mode in an inference server.
            //
            // `build_nvfp4_hadamard_signs` is a no-op (returns
            // `Ok(None)`) when the env gate is off, so adding the
            // pre-allocation step here is free for the default
            // configuration.
            {
                let max_hd = bringup.arch.max_head_dim() as u32;
                let nl = bringup.arch.num_hidden_layers as u32;
                match rvllm_runtime::gemma4_bring_up::build_nvfp4_hadamard_signs(
                    nl, max_hd, &bringup.arena,
                ) {
                    Ok(alloc) => {
                        if alloc.is_some() {
                            tracing::info!(
                                "nvfp4_hadamard signs pre-allocated above scratch checkpoint"
                            );
                        }
                        *bringup.nvfp4_hadamard.lock().unwrap() = alloc;
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!(
                            "build_nvfp4_hadamard_signs: {e:?}"
                        )));
                        return;
                    }
                }
            }

            // Pre-allocate the NVFP4 shadow KV / Q / throwaway regions
            // BEFORE the scratch checkpoint, mirroring the hadamard
            // pre-alloc above. Same lifetime contract — the lazy alloc
            // inside `run_generate` had a known silent-corruption bug
            // from request 2 onward (regions allocated below the
            // checkpoint were marked free by `arena.restore(scratch_ck)`
            // and aliased into fresh scratch on the next request). The
            // helper returns `None` when `RVLLM_NVFP4_SHADOW_LAYERS` is
            // unset, so this is a no-op on production where the
            // diagnostic is disabled.
            {
                const BLOCK_SIZE: u32 = 32;
                let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
                    .ok().and_then(|s| s.parse().ok())
                    .unwrap_or(1024);
                // Sliding layers share the same block budget on the
                // current code path; if that ever diverges, mirror
                // the runtime's calculation here.
                let sliding_blocks: u32 = num_blocks_total;
                match rvllm_runtime::gemma4_bring_up::build_nvfp4_shadow_alloc(
                    &bringup.arch,
                    num_blocks_total,
                    sliding_blocks,
                    BLOCK_SIZE,
                    &bringup.arena,
                    bringup.assistant_kv_sources,
                ) {
                    Ok(alloc) => {
                        if alloc.is_some() {
                            tracing::info!(
                                "nvfp4_shadow regions pre-allocated above scratch checkpoint"
                            );
                        }
                        *bringup.nvfp4_shadow.lock().unwrap() = alloc;
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!(
                            "build_nvfp4_shadow_alloc: {e:?}"
                        )));
                        return;
                    }
                }
            }
            let scratch_ck = bringup.arena.checkpoint();
            tracing::info!(
                compute_cap = ?bringup.ctx.compute_capability(),
                arena_mib = bringup.arena.capacity() / (1024 * 1024),
                scratch_checkpoint = scratch_ck,
                "cuda worker ready",
            );
            let _ = ready_tx.send(Ok(()));

            // Main serve loop. One request at a time (single-seq).
            while let Some(req) = req_rx.blocking_recv() {
                // Same GB10 ctx-rebind guard as the qwen36 branch
                // (see rvllm-mem/src/context.rs::bind_to_current_thread).
                // One `cuCtxSetCurrent` per request is free; a dropped
                // binding after a long blocking_recv idle can produce
                // cuLaunchKernel failures on the next decode.
                if let Err(e) = bringup.ctx.bind_to_current_thread() {
                    let _ = req.events_tx.send(GenerateEvent::Error(
                        format!("gemma4 ctx rebind: {e:?}"),
                    ));
                    continue;
                }
                run_one(&bringup, &kernels_ctx, &spec_cfg, req);
                // SAFETY: `run_one` fully consumes the `Region`s it
                // allocated inside `bringup.run_generate` — they're
                // function-local there and drop before we return.
                // No region reference above the checkpoint survives,
                // so rewinding the bump pointer is safe.
                unsafe { bringup.arena.restore(scratch_ck); }
                // Commit 55: clear the per-request batched-verify
                // override so a non-spec request (= spec_cfg.enabled
                // off) sees a clean atomic. Re-armed at the top of
                // each spec-enabled request.
                bringup
                    .force_batched_verify
                    .store(false, std::sync::atomic::Ordering::Release);
            }

            tracing::info!("cuda worker queue closed, exiting");
        })
        .map_err(|e| ApiError::Internal(format!("spawn cuda worker: {e}")))?;

    match ready_rx.await {
        Ok(Ok(())) => Ok((WorkerHandle::new(req_tx, cfg.queue_depth.max(1)), join)),
        Ok(Err(msg)) => Err(ApiError::Internal(msg)),
        Err(_) => Err(ApiError::Internal(
            "cuda worker thread exited before signalling ready".into(),
        )),
    }
}

/// Held together intentionally: `LoadedModule` is RAII, its `Drop`
/// calls `cuModuleUnload` — if we drop the module but keep the
/// `KernelFn` around, the next `cuLaunchKernel` dereferences a
/// freed handle and fails with `LaunchFailed` (how this bug first
/// surfaced on the GB10 live-test smoke). Struct owns the module
/// to keep it alive alongside the function handles.
struct GenerateKernels {
    fn_embed: rvllm_kernels::KernelFn,
    fn_argmax: rvllm_kernels::KernelFn,
    // Never read — its lifetime anchors the module so `fn_embed`
    // stays valid across run_generate calls.
    _embed_mod: rvllm_kernels::LoadedModule,
}

fn resolve_generate_kernels(
    bringup: &Gemma4Bringup,
) -> Result<GenerateKernels, String> {
    let embed_mod = bringup
        .kernels
        .load_ptx("embedding_gather_f16")
        .map_err(|e| format!("load embedding_gather_f16: {e}"))?;
    let fn_embed = embed_mod
        .get_function("embedding_gather_f16_kernel")
        .map_err(|e| format!("resolve embedding_gather_f16_kernel: {e}"))?;
    let fn_argmax = bringup.fused.fn_argmax;
    Ok(GenerateKernels { fn_embed, fn_argmax, _embed_mod: embed_mod })
}

fn run_one(
    bringup: &Gemma4Bringup,
    kernels: &GenerateKernels,
    spec_cfg: &WorkerSpecDecode,
    req: GenerateRequest,
) {
    let prompt_len = req.prompt_ids.len() as u32;

    if req.cancelled.load(Ordering::Relaxed) {
        let _ = req.events_tx.send(GenerateEvent::Done {
            finish: FinishReason::Cancelled,
            prompt_tokens: prompt_len,
            completion_tokens: 0,
        });
        return;
    }

    tracing::debug!(
        request_id = %req.request_id,
        prompt_tokens = req.prompt_ids.len(),
        max_new = req.max_new_tokens,
        spec_decode = spec_cfg.enabled,
        "calling Gemma generate",
    );
    let sampling_cfg = match req.sampling {
        crate::sampling::SamplingDecision::Greedy => {
            rvllm_runtime::gemma4_bring_up::SamplingConfig::Greedy
        }
        crate::sampling::SamplingDecision::Stochastic(s) => {
            rvllm_runtime::gemma4_bring_up::SamplingConfig::Stochastic {
                temperature: s.temperature,
                top_p: s.top_p,
                top_k: s.top_k,
                seed: s.seed,
            }
        }
    };
    // Per-token streaming sink. Fires on every token the runtime
    // emits (prefill's first + every decode-loop token, in order).
    // Returning `false` stops generation — used for closed-channel
    // (client disconnect / handler exit) and for the existing
    // cancellation flag (timeout, stop-string match in handler,
    // SSE Drop). Counter `emitted` runs from 0 so SSE position
    // matches.
    //
    // This is the change that makes `stream=true` actually deliver
    // tokens incrementally instead of in a burst at the end of
    // run_generate, AND lets handler-side stop-string detection
    // cancel the worker within ~one decode step instead of after
    // all max_tokens are produced.
    let mut emitted: u32 = 0;
    let cancelled_ref = req.cancelled.clone();
    let events_tx_ref = req.events_tx.clone();
    let mut on_token = |id: u32| -> bool {
        if cancelled_ref.load(Ordering::Relaxed) {
            return false;
        }
        let pos = emitted;
        match events_tx_ref.send(GenerateEvent::Token { id, position: pos }) {
            Ok(()) => {
                emitted += 1;
                true
            }
            Err(_) => false,
        }
    };

    // Phase 3b: Gemma vision pre-pass. Run the ViT forward for each
    // image; collect device-side embeddings as raw f16 bytes for
    // splice-during-prefill inside `run_generate`.
    let mut vision_outputs: Vec<rvllm_runtime::qwen36_bring_up::VisionForwardOutput> =
        Vec::with_capacity(req.vision_items.len());
    let mut vision_failed_msg: Option<String> = None;
    for (i, item) in req.vision_items.iter().enumerate() {
        if req.cancelled.load(Ordering::Relaxed) {
            vision_failed_msg = Some("cancelled".to_string());
            break;
        }
        match bringup.forward_gemma_vision(&item.bytes) {
            Ok(out) => {
                tracing::info!(
                    idx = i,
                    tokens = out.num_tokens,
                    hidden = out.hidden_dim,
                    "vision: Gemma ViT forward done"
                );
                if out.num_tokens != item.num_tokens {
                    vision_failed_msg = Some(format!(
                        "vision tokens mismatch: predicted {} got {}",
                        item.num_tokens, out.num_tokens
                    ));
                    break;
                }
                vision_outputs.push(out);
            }
            Err(e) => {
                vision_failed_msg = Some(format!("gemma vision forward: {e:?}"));
                break;
            }
        }
    }
    if let Some(msg) = vision_failed_msg {
        let _ = req.events_tx.send(GenerateEvent::Error(msg));
        let _ = req.events_tx.send(GenerateEvent::Done {
            finish: FinishReason::Stop,
            prompt_tokens: prompt_len,
            completion_tokens: 0,
        });
        return;
    }
    let vision_splice: Vec<(usize, &[u8])> = req
        .vision_slots
        .iter()
        .map(|s| (s.token_start, vision_outputs[s.vision_item_idx].data.as_slice()))
        .collect();

    // Audio pre-pass: run forward_gemma_audio_to_host per audio item,
    // collect host bytes for splice in run_generate.
    let mut audio_failed_msg: Option<String> = None;
    let mut audio_host_bytes: Vec<Vec<u8>> = Vec::with_capacity(req.audio_items.len());
    for (_i, item) in req.audio_items.iter().enumerate() {
        if req.cancelled.load(Ordering::Relaxed) {
            audio_failed_msg = Some("cancelled".to_string());
            break;
        }
        match bringup.forward_gemma_audio_to_host(&item.samples_16k_mono) {
            Ok((buf, n_soft, _hidden)) => {
                if n_soft != item.num_soft_tokens {
                    audio_failed_msg = Some(format!(
                        "audio soft-token mismatch: predicted {} got {}",
                        item.num_soft_tokens, n_soft
                    ));
                    break;
                }
                audio_host_bytes.push(buf);
            }
            Err(e) => {
                audio_failed_msg = Some(format!("gemma audio forward: {e:?}"));
                break;
            }
        }
    }
    if let Some(msg) = audio_failed_msg {
        let _ = req.events_tx.send(GenerateEvent::Error(msg));
        let _ = req.events_tx.send(GenerateEvent::Done {
            finish: FinishReason::Stop,
            prompt_tokens: prompt_len,
            completion_tokens: 0,
        });
        return;
    }
    let audio_splice: Vec<(usize, &[u8])> = req
        .audio_slots
        .iter()
        .map(|s| (s.token_start, audio_host_bytes[s.audio_item_idx].as_slice()))
        .collect();

    let result = unsafe {
        if spec_cfg.enabled {
            // Commit 55 (codex review priority 0.1): when SPEC_DECODE
            // is on, route to the batched-verify path BY DEFAULT.
            // Previously, without an explicit RVLLM_GEMMA4_SPEC_BATCHED=1,
            // the worker fell into the legacy `run_generate_speculative`
            // path which runs warmup_max_new = max_new sequential base
            // decodes — pure overhead (drafter runs but base is still
            // sequential). The legacy single-step path is now only
            // reachable as an explicit opt-out via
            // RVLLM_GEMMA4_SPEC_LEGACY_DEBUG=1.
            let legacy_debug =
                std::env::var("RVLLM_GEMMA4_SPEC_LEGACY_DEBUG").as_deref() == Ok("1");
            // Iterative wrapper retained as a separate explicit opt-in
            // for debugging the legacy sequential-decode verify; not
            // a production path.
            let iterative = std::env::var("RVLLM_GEMMA4_SPEC_ITERATIVE").as_deref() == Ok("1");
            let batched = !legacy_debug && !iterative;
            if batched {
                // Codex Round 9 #4: arm the force_batched_verify atomic
                // ONLY for the batched wrapper, and clear it after the
                // call so the flag cannot leak into a subsequent
                // non-batched request handled by the same worker.
                // Earlier the store(true) was unconditionally above
                // the branch, which armed the flag even for the
                // legacy / iterative paths.
                bringup
                    .force_batched_verify
                    .store(true, std::sync::atomic::Ordering::Release);
                bringup.run_generate_speculative_batched(
                    kernels.fn_embed,
                    kernels.fn_argmax,
                    &req.prompt_ids,
                    req.max_new_tokens as usize,
                    &req.stop_token_ids,
                    spec_cfg.k,
                    sampling_cfg,
                    Some(req.cancelled.as_ref()),
                    Some(&mut on_token),
                    &vision_splice,
                    &audio_splice,
                )
            } else if iterative {
                bringup.run_generate_speculative_iterative(
                    kernels.fn_embed,
                    kernels.fn_argmax,
                    &req.prompt_ids,
                    req.max_new_tokens as usize,
                    &req.stop_token_ids,
                    spec_cfg.k,
                    sampling_cfg,
                    Some(req.cancelled.as_ref()),
                    Some(&mut on_token),
                    &vision_splice,
                    &audio_splice,
                )
            } else {
                bringup.run_generate_speculative(
                    kernels.fn_embed,
                    kernels.fn_argmax,
                    &req.prompt_ids,
                    req.max_new_tokens as usize,
                    &req.stop_token_ids,
                    spec_cfg.k,
                    sampling_cfg,
                    Some(req.cancelled.as_ref()),
                    Some(&mut on_token),
                    &vision_splice,
                    &audio_splice,
                )
            }
        } else {
            // Codex round 10 fix: defensively disarm ALL spec-decode
            // hook atomics before a non-spec run_generate call. Prior
            // sessions only cleared force_batched_verify here; the
            // other hooks (force_common_prefix_override,
            // force_prefill_only, skip_prefix_cache_publish, the two
            // base_last_*_snapshot_pending flags) could leak across
            // requests from an errored spec path and corrupt the next
            // non-spec request — e.g. force_common_prefix_override
            // makes run_generate skip real prompt tokens, reading
            // stale KV slots and producing garbage like
            // "Die,\n\n        *,\n\n        *,...".
            use std::sync::atomic::Ordering::Release;
            bringup.force_batched_verify.store(false, Release);
            bringup
                .force_common_prefix_override
                .store(u32::MAX, Release);
            bringup.force_prefill_only.store(false, Release);
            bringup.skip_prefix_cache_publish.store(false, Release);
            bringup.base_last_k_snapshot_pending.store(false, Release);
            bringup
                .base_last_hidden_snapshot_pending
                .store(false, Release);
            bringup.run_generate(
                kernels.fn_embed,
                kernels.fn_argmax,
                &req.prompt_ids,
                req.max_new_tokens as usize,
                &req.stop_token_ids,
                // shadow_requested: per-request header was removed; gate on
                // the env var directly so cycle-26 ShadowDumper analysis can
                // fire without needing to restore the header path.
                env_truthy("RVLLM_NVFP4_SHADOW_F16"),
                sampling_cfg,
                // Forward the request-level cancellation flag. The HTTP
                // handler sets this on client disconnect / wall-clock
                // timeout; the runtime breaks out of the decode loop on
                // the next step so the worker thread does not stay
                // blocked rendering tokens nobody will read.
                Some(req.cancelled.as_ref()),
                Some(&mut on_token),
                &vision_splice,
                &audio_splice,
            )
        }
    };

    // Codex Round 9 #4: clear force_batched_verify after the spec call
    // returns so the flag cannot leak into the next request handled
    // by this worker. No-op when the flag was never set.
    bringup
        .force_batched_verify
        .store(false, std::sync::atomic::Ordering::Release);

    match result {
        Ok(generated_ids) => {
            // Commit 25: drain per-request spec-decode accept stats
            // (if any) and emit as a SpeculativeStep event so the
            // HTTP handler can surface them via X-RVLLM-Accept-Rate.
            // No-op when spec-decode is off (stats stay None).
            if spec_cfg.enabled {
                if let Some(stats) = bringup.take_last_spec_stats() {
                    let _ = req.events_tx.send(GenerateEvent::SpeculativeStep {
                        drafted: stats.drafted,
                        accepted: stats.accepted,
                        cumulative_decoded: stats.cumulative_decoded,
                    });
                }
            }
            // Tokens were already emitted to the events channel via
            // the on_token callback during run_generate. We just
            // need the final Done event with the right finish_reason
            // and completion_tokens count.
            let final_emitted = emitted;
            let finish = if req.cancelled.load(Ordering::Relaxed) {
                FinishReason::Cancelled
            } else if generated_ids
                .last()
                .is_some_and(|id| req.stop_token_ids.contains(id))
            {
                FinishReason::Stop
            } else if final_emitted >= req.max_new_tokens {
                FinishReason::Length
            } else {
                FinishReason::Stop
            };
            let _ = req.events_tx.send(GenerateEvent::Done {
                finish,
                prompt_tokens: prompt_len,
                completion_tokens: final_emitted,
            });
        }
        Err(e) => {
            let msg = format!("run_generate: {e}");
            tracing::error!(request_id = %req.request_id, error = %msg, "generation failed");
            let _ = req.events_tx.send(GenerateEvent::Error(msg));
        }
    }
}

/// Resolve `Gemma4EnginePaths` from env vars + sensible fallbacks,
/// matching what `probe-gemma4-load` accepts (so operators can move
/// straight from probe to serve). On sm_121 the cutlass_so / fa3_so
/// are never opened; the policy file is parsed but its entries are
/// not consulted, so a minimal placeholder is acceptable.
pub fn resolve_paths(
    model_dir: PathBuf,
    kernels_dir: PathBuf,
    cutlass_so: Option<PathBuf>,
    fa3_so: Option<PathBuf>,
    policy_json: Option<PathBuf>,
) -> Result<Gemma4EnginePaths, ApiError> {
    const UNUSED: &str = "<unused-on-sm121>";

    // Codex31/34: detect sm_121 host so we can (a) point cutlass_so at
    // the real libcutlass_sm120.so (lib_so.rs's resolver anchors at
    // sm90_hint.parent().parent() — placeholder breaks the search),
    // and (b) skip the minimal-policy placeholder write on read-only
    // /tmp / restricted containers. Two signals, OR'd: nvidia-smi
    // (preferred — matches the runtime's actual device) and the
    // kernels_dir layout (deterministic fallback for containers
    // without nvidia-smi or whose first-line cap doesn't match the
    // CUDA device). Either is sufficient.
    fn is_sm121_via_nvidia_smi() -> bool {
        // Authoritative device-property probe via nvidia-smi
        // compute_cap. The earlier code OR'd this with a
        // kernels_dir-based detection that flipped to sm_121 if the
        // operator's kernel root happened to ship `sm_121/` alongside
        // other arch dirs — an H100 host with a shared multi-arch
        // kernel install would then be misclassified, the
        // sm_121-only `libcutlass_sm120.so` resolver would fire, and
        // startup would either ENOENT or load the wrong .so.
        let out = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output();
        match out {
            Ok(o) if o.status.success() => {
                let s = String::from_utf8_lossy(&o.stdout);
                s.lines().next().map(|l| l.trim() == "12.1").unwrap_or(false)
            }
            _ => false,
        }
    }
    fn is_sm121_via_env_override() -> bool {
        // Operator escape hatch: `RVLLM_FORCE_SM121=1` if nvidia-smi
        // is unavailable in the container but the device IS sm_121.
        // Replaces the implicit "directory exists" signal with an
        // explicit one.
        std::env::var("RVLLM_FORCE_SM121")
            .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
            .unwrap_or(false)
    }
    let sm121 = is_sm121_via_nvidia_smi() || is_sm121_via_env_override();

    // Resolve the policy-json path. We never write into `kernels_dir`
    // here — that path can be a system-wide read-only location
    // (read-only container layer, immutable Nix store, root-owned
    // shared install) and a resolver should not depend on being able
    // to write there.
    //
    // Order:
    //   1. Caller-supplied `policy_json` — used as-is.
    //   2. `RVLLM_MINIMAL_POLICY_PATH` env var — explicit operator
    //      override; useful for read-only filesystems where the
    //      operator wants the file in a known location.
    //   3. `std::env::temp_dir()` fallback — written once per process
    //      with the placeholder body. The file is small (~80 B) and
    //      idempotent, so a stale copy from a prior process is
    //      harmless.
    // The runtime loader at `Gemma4Bringup::load` reads
    // `paths.policy_json` strictly via `std::fs::read`, so the file
    // must exist on disk before bring-up. Both fallback branches
    // therefore write the placeholder body if the file is not
    // already there. A previous iteration only wrote the temp_dir
    // path and returned the env-override path raw — operators
    // following the "set RVLLM_MINIMAL_POLICY_PATH" hint then hit
    // ENOENT despite obeying the error message.
    let policy_json = match policy_json {
        Some(p) => p,
        None => {
            // On sm_121 the runtime (Codex30-2 / Codex31-2) skips
            // policy.json entirely — neither the generic nor the
            // gemma4 bring-up reads it. Synthesise a path so anything
            // that prints `policy_json` for diagnostics shows a
            // sensible location, but do NOT touch disk: a read-only
            // /tmp or a restricted container shouldn't abort startup
            // for a file that won't be read.
            //
            // On non-sm_121 (RTX 5090, RTX 6000 Blackwell, …) the
            // FP8 plan loader DOES read this file — and the legacy
            // fallback wrote a tiny placeholder body that
            // `Fp8GemmPlan::from_policy(...)` then rejects later, deep
            // inside autotune. That produced a confusing "plan
            // missing entry for shape X" instead of the truthful
            // "operator forgot to pass --policy-json". Refuse here
            // with a clear startup error.
            let p = std::env::var_os("RVLLM_MINIMAL_POLICY_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    std::env::temp_dir().join("rvllm-serve-minimal-policy.json")
                });
            if !sm121 {
                return Err(ApiError::Internal(format!(
                    "policy_json required on non-sm_121 hosts: pass an \
                     explicit FP8 policy via --policy-json (a real one \
                     produced by the autotune sweep, not a placeholder). \
                     Resolved would-be path was {} but no policy was \
                     supplied and synthesising a stub here only delays \
                     the failure to Fp8GemmPlan::from_policy(...)",
                    p.display()
                )));
            }
            p
        }
    };

    // Codex31-1: when the operator hasn't supplied an explicit
    // cutlass_so on a sm_121 host, point at the per-arch dir so
    // CutlassBackend::resolve_sm120_so_path (which anchors at
    // sm90_hint.parent().parent().join(arch)) actually finds the
    // shipped libcutlass_sm120.so. The legacy "<unused-on-sm121>"
    // placeholder broke that search and silently fell to
    // CutlassBackend::Absent — production was running without the
    // CUTLASS blockwise FP8 path unless RVLLM_CUTLASS_SM120_SO was
    // set explicitly. The file may still be absent (operators
    // without CUTLASS), in which case the resolver keeps falling
    // through to Absent — same outcome as before, just no longer
    // hidden behind a string-shaped tripwire.
    let cutlass_so = cutlass_so.unwrap_or_else(|| {
        if sm121 {
            // Codex34-2: when the operator already pointed kernels_dir
            // at the per-arch sm_121 subdir (Codex32-2 lets that work
            // for resolve_kernels_dir), don't append sm_121 again —
            // that would land at .../sm_121/sm_121/libcutlass_sm120.so
            // and the resolver's parent().parent() walk would still
            // land in the wrong dir, falling back to Absent.
            let arch_dir = if kernels_dir.file_name().and_then(|s| s.to_str()) == Some("sm_121") {
                kernels_dir.clone()
            } else {
                kernels_dir.join("sm_121")
            };
            arch_dir.join("libcutlass_sm120.so")
        } else {
            PathBuf::from(UNUSED)
        }
    });
    Ok(Gemma4EnginePaths {
        model_dir,
        kernels_dir,
        cutlass_so,
        fa3_so: fa3_so.unwrap_or_else(|| PathBuf::from(UNUSED)),
        policy_json,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    /// Both tests in this module manipulate the
    /// `RVLLM_MINIMAL_POLICY_PATH` env var. cargo runs lib tests in
    /// parallel by default, so without serialisation they race and
    /// one observes the other's env state. A process-wide lock keeps
    /// the env-var critical section ordered without needing
    /// `--test-threads=1` from the user.
    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Detect sm_121 the same way `resolve_paths` does, so tests
    /// stay portable between an x86 dev box and a GB10 CI host
    /// without pulling in a cudarc dep.
    fn detect_sm121_for_test() -> bool {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout).ok()
                } else { None }
            })
            .map(|s| s.lines().next().map(|l| l.trim() == "12.1").unwrap_or(false))
            .unwrap_or(false)
            || std::env::var("RVLLM_FORCE_SM121")
                .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false)
    }

    /// `resolve_paths` must NOT write into `kernels_dir`. A read-only
    /// kernel install (Nix store, container layer, root-owned shared
    /// dir) is a legitimate deploy shape; the resolver writing a
    /// `.serve-minimal-policy.json` there used to fail at startup
    /// with a confusing IO error.
    ///
    /// On sm_121 the runtime skips policy_json entirely so the
    /// resolver succeeds without touching disk. On non-sm_121 the
    /// resolver now refuses (Codex60 / round-16 finding #2): the
    /// loader needs a real autotune-produced policy, and the old
    /// behaviour of synthesising a placeholder only delayed the
    /// failure to `Fp8GemmPlan::from_policy(...)`.
    #[test]
    fn resolve_paths_does_not_write_into_kernels_dir() {
        let _g = env_lock().lock().expect("env lock");
        let tmp = std::env::temp_dir().join(format!(
            "rvllm-serve-resolve-paths-test-{}", std::process::id()
        ));
        std::fs::create_dir_all(&tmp).expect("setup tmp");
        let kernels_dir = tmp.join("kernels-readonly");
        std::fs::create_dir_all(&kernels_dir).expect("setup kernels");
        let model_dir = tmp.join("model");
        std::fs::create_dir_all(&model_dir).expect("setup model");

        std::env::remove_var("RVLLM_MINIMAL_POLICY_PATH");
        let result = resolve_paths(model_dir, kernels_dir.clone(), None, None, None);

        if detect_sm121_for_test() {
            let paths = result.expect("resolve on sm_121");
            assert!(!paths.policy_json.starts_with(&kernels_dir),
                "policy_json {:?} is inside kernels_dir {:?}",
                paths.policy_json, kernels_dir);
        } else {
            assert!(result.is_err(),
                "resolve_paths must refuse on non-sm_121 without --policy-json: {:?}",
                result.map(|p| p.policy_json));
        }

        // Postcondition: kernels_dir untouched in either branch.
        let after: Vec<_> = std::fs::read_dir(&kernels_dir)
            .expect("read kernels_dir")
            .collect();
        assert!(after.is_empty(),
            "resolve_paths wrote into kernels_dir: {:?}",
            after.into_iter().filter_map(|e| e.ok().map(|e| e.path())).collect::<Vec<_>>()
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Operator can override the policy path via env var, e.g. on a
    /// read-only host where /tmp is also read-only. The override path
    /// On sm_121 the env override decides where the (unread)
    /// policy_json path is recorded; the resolver doesn't touch
    /// disk. On non-sm_121 the resolver refuses regardless of the
    /// env override — the env var only steers the placeholder
    /// location, not a real policy.
    #[test]
    fn resolve_paths_honours_env_override() {
        let _g = env_lock().lock().expect("env lock");
        let tmp = std::env::temp_dir().join(format!(
            "rvllm-serve-resolve-env-test-{}", std::process::id()
        ));
        std::fs::create_dir_all(&tmp).expect("setup tmp");
        let kernels_dir = tmp.join("kernels");
        std::fs::create_dir_all(&kernels_dir).expect("setup kernels");
        let custom_policy = tmp.join("my-policy.json");

        std::env::set_var("RVLLM_MINIMAL_POLICY_PATH", &custom_policy);
        let result = resolve_paths(
            tmp.join("model"),
            kernels_dir.clone(),
            None,
            None,
            None,
        );
        std::env::remove_var("RVLLM_MINIMAL_POLICY_PATH");

        if detect_sm121_for_test() {
            let paths = result.expect("resolve on sm_121 with env override");
            assert_eq!(paths.policy_json, custom_policy,
                "env override not honoured: got {:?}", paths.policy_json);
        } else {
            assert!(result.is_err(),
                "non-sm_121 must refuse even with env override: {:?}",
                result.map(|p| p.policy_json));
        }

        let after: Vec<_> = std::fs::read_dir(&kernels_dir)
            .expect("read kernels_dir")
            .collect();
        assert!(after.is_empty(), "kernels_dir was written to despite env override");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Operator-only accept-rate probe for the Option B Gemma4-NVFP4
    /// spec path on a real text prompt. This lives in rvllm-serve so it
    /// can reuse the production tokenizer instead of adding tokenizers
    /// as a runtime-crate dependency.
    #[test]
    #[ignore]
    fn ondisk_gemma4_nvfp4_spec_real_prompt_accept_probe() {
        let _g = env_lock().lock().expect("env lock");
        let model_dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset - skip");
                return;
            }
        };
        let drafter_dir = std::env::var("GEMMA4_DRAFTER_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(
                "/home/r00t/gemma-4-31B-it-assistant"));
        if !drafter_dir.is_dir() {
            eprintln!("drafter dir {drafter_dir:?} missing - skip");
            return;
        }
        let kernels_dir = PathBuf::from(
            "/home/r00t/workspace/upstream/rvllm-serve/kernels/sm_121");
        let prompt = std::env::var("G4N_SPEC_ACCEPT_PROMPT")
            .unwrap_or_else(|_| {
                "Tell me a short story about a careful engineer who \
                 measures before optimizing."
                    .to_string()
            });
        let max_new = std::env::var("G4N_SPEC_ACCEPT_MAX_NEW")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64);
        let spec_k = std::env::var("G4N_SPEC_ACCEPT_K")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4);

        let tokenizer = crate::tokenize::TokenizerHandle::load(&model_dir)
            .expect("TokenizerHandle::load");
        let prompt_ids = tokenizer.encode(&prompt).expect("tokenize prompt");
        assert!(!prompt_ids.is_empty(), "tokenized prompt is empty");

        let mut bringup =
            rvllm_runtime::gemma4_nvfp4_bring_up::Gemma4Nvfp4Bringup::load(
                &model_dir, 40 * 1024 * 1024 * 1024, &kernels_dir)
                .expect("Gemma4Nvfp4Bringup::load");
        let max_pos = ((prompt_ids.len() + max_new + spec_k + 8)
            .next_power_of_two())
            .max(128)
            .min(4096) as u32;
        let kv = bringup
            .allocate_kv_state_with_chunk(max_pos, max_pos)
            .expect("allocate_kv_state_with_chunk");
        bringup
            .ensure_base_last_hidden_buffer()
            .expect("ensure_base_last_hidden_buffer");
        bringup
            .ensure_drafter_nvfp4(&drafter_dir, &kv)
            .expect("ensure_drafter_nvfp4");

        let plain_t0 = std::time::Instant::now();
        let mut plain_emitted = Vec::with_capacity(max_new);
        let mut last = bringup
            .forward_prompt_to_token(&prompt_ids, 0, &kv)
            .expect("plain forward_prompt_to_token");
        plain_emitted.push(last);
        for step in 1..max_new {
            let position = prompt_ids.len() as u32 + (step as u32) - 1;
            last = bringup
                .forward_full_to_token(last, position, &kv)
                .expect("plain forward_full_to_token");
            plain_emitted.push(last);
        }
        let plain_elapsed = plain_t0.elapsed().as_secs_f64();

        let old_bailout = std::env::var(
            "G4N_SPEC_ZERO_ACCEPT_BAILOUT_ITERS").ok();
        std::env::set_var("G4N_SPEC_ZERO_ACCEPT_BAILOUT_ITERS", "0");

        let spec_t0 = std::time::Instant::now();
        let spec_stats = bringup
            .run_spec_session_nvfp4_greedy_k(
                &prompt_ids, max_new, spec_k, &[], &kv)
            .expect("run_spec_session_nvfp4_greedy_k");
        let spec_elapsed = spec_t0.elapsed().as_secs_f64();

        match old_bailout {
            Some(v) => std::env::set_var(
                "G4N_SPEC_ZERO_ACCEPT_BAILOUT_ITERS", v),
            None => std::env::remove_var(
                "G4N_SPEC_ZERO_ACCEPT_BAILOUT_ITERS"),
        }

        eprintln!(
            "[g4n-spec-real-prompt-bench] prompt_tokens={} max_new={} \
             K={} plain_elapsed={:.2}s plain_tok_s={:.3} \
             spec_elapsed={:.2}s spec_tok_s={:.3} speedup={:.3}x \
             spec_emitted={} n_iters={} n_accepted={} \
             accept_per_iter={:.3} prompt={:?}",
            prompt_ids.len(),
            max_new,
            spec_k,
            plain_elapsed,
            max_new as f64 / plain_elapsed,
            spec_elapsed,
            max_new as f64 / spec_elapsed,
            plain_elapsed / spec_elapsed,
            spec_stats.emitted.len(),
            spec_stats.n_iters,
            spec_stats.n_accepted,
            if spec_stats.n_iters > 0 {
                spec_stats.n_accepted as f32 / spec_stats.n_iters as f32
            } else { 0.0 },
            prompt,
        );
        assert_eq!(plain_emitted.len(), max_new);
        assert_eq!(spec_stats.emitted.len(), max_new);
        assert!(spec_stats.n_accepted <= spec_stats.n_iters * spec_k);
    }
}
