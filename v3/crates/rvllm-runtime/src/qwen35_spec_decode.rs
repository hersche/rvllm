//! Qwen 3.5 / Qwen 3.6 27B prompt-lookup speculative decoding.
//!
//! This mirrors the Qwen 3.6 prompt-lookup policy but calls the
//! Qwen35 bring-up primitives. The key correctness requirement is
//! the same as Qwen36: linear-attn delta state and causal-conv
//! state are recurrent, so a verify chunk must snapshot before
//! running and replay only the accepted prefix after any partial
//! accept.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

use crate::qwen35_bring_up::Qwen35Bringup;

struct PromptLookupIndex {
    ngram: usize,
    next_start: usize,
    latest_start_by_ngram: HashMap<Vec<u32>, usize>,
}

impl PromptLookupIndex {
    fn new(ngram: usize) -> Self {
        Self {
            ngram,
            next_start: 0,
            latest_start_by_ngram: HashMap::new(),
        }
    }

    fn drafts(&mut self, committed: &[u32], max_drafts: usize) -> Vec<u32> {
        if self.ngram == 0 || max_drafts == 0 || committed.len() < self.ngram + 1 {
            return Vec::new();
        }
        let tail_start = committed.len() - self.ngram;
        if tail_start < self.ngram {
            return Vec::new();
        }
        let max_start = tail_start - self.ngram;
        while self.next_start <= max_start {
            let start = self.next_start;
            self.latest_start_by_ngram
                .insert(committed[start..start + self.ngram].to_vec(), start);
            self.next_start += 1;
        }
        let pattern = &committed[tail_start..];
        let Some(&start) = self.latest_start_by_ngram.get(pattern) else {
            return Vec::new();
        };
        let draft_start = start + self.ngram;
        let drafts_avail = tail_start.saturating_sub(draft_start);
        if drafts_avail == 0 {
            return Vec::new();
        }
        let n = drafts_avail.min(max_drafts);
        committed[draft_start..draft_start + n].to_vec()
    }
}

fn repeated_tail_pattern_drafts(
    committed: &[u32],
    min_repeats: usize,
    max_pattern: usize,
    max_drafts: usize,
) -> Vec<u32> {
    if min_repeats == 0 || max_pattern == 0 || max_drafts == 0 {
        return Vec::new();
    }
    for pattern_len in 1..=max_pattern.min(max_drafts) {
        if committed.len() < pattern_len * min_repeats {
            continue;
        }
        let pattern_start = committed.len() - pattern_len;
        let pattern = &committed[pattern_start..];
        let mut matched = true;
        for repeat_idx in 1..min_repeats {
            let start = committed.len() - pattern_len * (repeat_idx + 1);
            if &committed[start..start + pattern_len] != pattern {
                matched = false;
                break;
            }
        }
        if matched {
            let mut drafts = Vec::with_capacity(max_drafts);
            while drafts.len() < max_drafts {
                for &tok in pattern {
                    if drafts.len() >= max_drafts {
                        break;
                    }
                    drafts.push(tok);
                }
            }
            return drafts;
        }
    }
    Vec::new()
}

fn spec_err(op: &'static str) -> RvllmError {
    RvllmError::cuda(op, CudaErrorKind::Other, CudaCtx::setup())
}

pub fn run_qwen35_prompt_lookup_spec<F>(
    qwen: &Qwen35Bringup,
    prompt_ids: &[u32],
    max_new_tokens: u32,
    stop_token_ids: &[u32],
    mut on_token: F,
) -> Result<(u32, &'static str)>
where
    F: FnMut(u32, u32) -> bool,
{
    if prompt_ids.is_empty() {
        return Ok((0, "empty"));
    }
    let prompt_len = prompt_ids.len() as u32;
    let spec_k: usize = std::env::var("RVLLM_QWEN35_SPEC_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let ngram: usize = std::env::var("RVLLM_QWEN35_SPEC_NGRAM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let min_drafts_for_verify: usize = std::env::var("RVLLM_QWEN35_SPEC_MIN_DRAFTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(spec_k)
        .clamp(1, spec_k.max(1));
    let zero_accept_bailout_iters: u32 =
        std::env::var("RVLLM_QWEN35_SPEC_ZERO_ACCEPT_BAILOUT_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
    let repeat_run_min: usize = std::env::var("RVLLM_QWEN35_SPEC_REPEAT_RUN_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let repeat_pattern_max: usize = std::env::var("RVLLM_QWEN35_SPEC_REPEAT_PATTERN_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let perf_trace = std::env::var("RVLLM_QWEN35_SPEC_PERF_TRACE").as_deref() == Ok("1");

    let arena = qwen
        .arena
        .as_ref()
        .ok_or_else(|| spec_err("qwen35 spec: arena absent"))?;
    let (linear_bytes, conv_bytes) = qwen.recurrent_state_bytes();
    let snap_linear = arena.region("qwen35_spec_snap_linear", linear_bytes.max(1), 16)?;
    let snap_conv = arena.region("qwen35_spec_snap_conv", conv_bytes.max(1), 16)?;
    let snap_linear_ptr = snap_linear.device_ptr();
    let snap_conv_ptr = snap_conv.device_ptr();
    let iter_ck = arena.checkpoint();

    let mut current = unsafe {
        let t = qwen.forward_qwen35_prefill_batched(prompt_ids, &[])?;
        arena.restore(iter_ck);
        t
    };

    let mut committed: Vec<u32> =
        Vec::with_capacity(prompt_ids.len() + max_new_tokens as usize + 8);
    committed.extend_from_slice(prompt_ids);

    let mut completion_tokens: u32 = 0;
    let mut iter_count: u32 = 0;
    let mut verify_iters: u32 = 0;
    let mut total_drafted: u32 = 0;
    let mut total_accepted: u32 = 0;
    let mut zero_accept_iters: u32 = 0;
    let mut bailout_tokens: u32 = 0;
    let mut skipped_short_drafts: u32 = 0;
    let mut repeated_draft_iters: u32 = 0;
    let mut prompt_lookup = PromptLookupIndex::new(ngram);
    let t0_session = if perf_trace {
        Some(std::time::Instant::now())
    } else {
        None
    };

    macro_rules! finish {
        ($reason:expr) => {{
            unsafe {
                arena.restore(iter_ck);
            }
            if let Some(t0) = t0_session {
                let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "[qwen35-spec-perf] reason={} iters={iter_count} \
                     verify_iters={verify_iters} drafted={total_drafted} \
                     accepted={total_accepted} accept_per_verify={:.2} \
                     bailout_tokens={bailout_tokens} \
                     skipped_short_drafts={skipped_short_drafts} \
                     repeated_draft_iters={repeated_draft_iters} \
                     completion_tokens={completion_tokens} wall_ms={dt_ms:.2}",
                    $reason,
                    if verify_iters > 0 {
                        total_accepted as f32 / verify_iters as f32
                    } else {
                        0.0
                    },
                );
            }
            return Ok((completion_tokens, $reason));
        }};
    }

    while completion_tokens < max_new_tokens {
        if stop_token_ids.contains(&current) {
            finish!("stop");
        }
        if !on_token(current, prompt_len + completion_tokens) {
            finish!("cancelled");
        }
        committed.push(current);
        completion_tokens += 1;
        if completion_tokens >= max_new_tokens {
            break;
        }

        let mut drafts =
            repeated_tail_pattern_drafts(&committed, repeat_run_min, repeat_pattern_max, spec_k);
        if !drafts.is_empty() {
            repeated_draft_iters += 1;
        } else {
            drafts = prompt_lookup.drafts(&committed, spec_k);
        }

        if drafts.len() < min_drafts_for_verify {
            if !drafts.is_empty() {
                skipped_short_drafts += 1;
            }
            let pos = prompt_len + completion_tokens - 1;
            current = unsafe {
                let out = qwen.forward_qwen35_decode_argmax_all(&[current], pos)?;
                arena.restore(iter_ck);
                out.first().copied().unwrap_or(0).max(0) as u32
            };
            iter_count += 1;
            continue;
        }

        let base_pos = prompt_len + completion_tokens - 1;
        let kmax = drafts.len();
        total_drafted += kmax as u32;
        let mut verify_input: Vec<u32> = Vec::with_capacity(kmax + 1);
        verify_input.push(current);
        verify_input.extend(drafts.iter().copied());

        qwen.snapshot_recurrent_state(snap_linear_ptr, snap_conv_ptr)?;
        let argmax_at = unsafe {
            qwen.forward_qwen35_decode_argmax_all(&verify_input, base_pos)?
        };
        verify_iters += 1;

        let mut accept_len = 0usize;
        for i in 0..kmax {
            if argmax_at[i] >= 0 && argmax_at[i] as u32 == drafts[i] {
                accept_len += 1;
            } else {
                break;
            }
        }
        total_accepted += accept_len as u32;

        if accept_len < kmax {
            qwen.restore_recurrent_state(snap_linear_ptr, snap_conv_ptr)?;
            let mut commit_input: Vec<u32> = Vec::with_capacity(accept_len + 1);
            commit_input.push(current);
            for i in 0..accept_len {
                commit_input.push(drafts[i]);
            }
            unsafe {
                qwen.forward_qwen35_decode_commit_only(&commit_input, base_pos)?;
            }
        }
        current = argmax_at[accept_len].max(0) as u32;

        if accept_len == 0 {
            zero_accept_iters += 1;
        } else {
            zero_accept_iters = 0;
        }

        for &tok in drafts.iter().take(accept_len) {
            if stop_token_ids.contains(&tok) {
                finish!("stop");
            }
            if !on_token(tok, prompt_len + completion_tokens) {
                finish!("cancelled");
            }
            committed.push(tok);
            completion_tokens += 1;
            if completion_tokens >= max_new_tokens {
                break;
            }
        }
        if completion_tokens >= max_new_tokens {
            break;
        }

        iter_count += 1;
        unsafe {
            arena.restore(iter_ck);
        }
        if zero_accept_bailout_iters > 0 && zero_accept_iters >= zero_accept_bailout_iters {
            while completion_tokens < max_new_tokens {
                if stop_token_ids.contains(&current) {
                    finish!("stop");
                }
                if !on_token(current, prompt_len + completion_tokens) {
                    finish!("cancelled");
                }
                committed.push(current);
                completion_tokens += 1;
                bailout_tokens += 1;
                if completion_tokens >= max_new_tokens {
                    break;
                }
                let pos = prompt_len + completion_tokens - 1;
                current = unsafe {
                    let out = qwen.forward_qwen35_decode_argmax_all(&[current], pos)?;
                    arena.restore(iter_ck);
                    out.first().copied().unwrap_or(0).max(0) as u32
                };
            }
        }
    }

    finish!("length");
}
