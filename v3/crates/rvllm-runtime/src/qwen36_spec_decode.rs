//! Qwen 3.6 35B-A3B prompt-lookup speculative decoding.
//!
//! No drafter checkpoint required: drafts come from n-gram
//! matches inside the prompt + emitted-so-far buffer (the
//! "prompt-lookup decoding" / "prompt-LD" idea, also known as
//! reference-based speculative decoding). The base does the
//! verify step exactly like classical spec-decode.
//!
//! Why this is the right MVP for Qwen 3.6:
//! * The Qwen team has not published an MTP/EAGLE assistant
//!   head for 35B-A3B, so a classical drafter-model approach
//!   is blocked on missing weights.
//! * Prompt-lookup needs zero new model weights and integrates
//!   cleanly with the existing batched-prefill forward path —
//!   each verify step is just a multi-token call into
//!   `forward_qwen36_decode`.
//! * Real speedup on the workloads where Qwen 3.6 is used
//!   most (long structured prompts, code, tool calls, JSON):
//!   the output naturally echoes prompt content, so n-gram
//!   matches are common.
//!
//! Acceptance is GREEDY only (temperature=0 path). When the
//! base's argmax at position `P + i` equals the proposed draft
//! token at position `P + i`, accept it. Stop at the first
//! mismatch. Emit the accepted prefix + the base's next
//! prediction ("bonus") and advance the session.
//!
//! State semantics — there are TWO kinds of caches to rollback:
//!
//! 1. Full-attn KV cache (positional, slot-indexed). Each verify
//!    writes K+1 slots [P, P+K]. If accept_len = m, slots
//!    [P, P+m] are valid; [P+m+1, P+K] are stale but get
//!    overwritten by the next iter's verify starting at P+m+1.
//!    No explicit rollback needed for KV.
//!
//! 2. **Recurrent state** (Gated-DeltaNet linear state + Conv1d
//!    state). RECURRENT — each forward step ADVANCES the state
//!    in place. A verify forward over [current, d_0, …, d_{K-1}]
//!    has advanced the persistent recurrent state through ALL
//!    K drafts by the time we know accept_len. Rejected drafts
//!    are not optional — the state IS already corrupted past the
//!    accepted prefix. **No naïve "overwrite later" trick works
//!    here** — the state is a moving window, not a slot-indexed
//!    cache.
//!
//!    Fix (Codex review 2026-05-17): snapshot the linear + conv
//!    state into a scratch arena region BEFORE verify, run verify
//!    on persistent state (so its KV writes land in the real
//!    cache), then on ANY partial accept restore the snapshot
//!    and replay a commit-only forward over [current, d_0, …,
//!    d_{accept_len-1}]. Commit-only = full layer stack + KV
//!    writes + recurrent state update, but no closer/argmax.
//!    On accept_len == K nothing to roll back — state is correct.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

use rvllm_core::Result;

use crate::qwen36_bring_up::Qwen36Bringup;

/// Search a `[prompt + emitted]` haystack for the last N tokens
/// of `committed`, return up to `max_drafts` tokens that follow
/// the most recent match. If no match, return empty.
///
/// `committed`: full session-so-far (prompt + emitted), in token-id
/// order. The trailing `ngram` tokens are the search pattern.
/// `ngram`: window size (typical: 2). 1 is too sloppy on diverse
/// text; 3+ misses too many matches. 2 is the prompt-LD default.
/// `max_drafts`: cap on K (~4–8 typical).
///
/// Returns drafts in execution order: `[t_match+ngram, t_match+ngram+1, ...]`.
/// The match is taken from the *latest* occurrence in `committed`
/// (excluding the trailing pattern itself), since recent context
/// is the strongest predictor.
pub fn prompt_lookup_drafts(committed: &[u32], ngram: usize, max_drafts: usize) -> Vec<u32> {
    if ngram == 0 || max_drafts == 0 || committed.len() < ngram + 1 {
        return Vec::new();
    }
    let tail_start = committed.len() - ngram;
    let pattern = &committed[tail_start..];
    // Search backwards from `tail_start - 1` down to 0 so we
    // pick the most recent occurrence first.
    if tail_start == 0 {
        return Vec::new();
    }
    let mut i = tail_start;
    while i > 0 {
        i -= 1;
        if i + ngram > tail_start {
            // candidate end overlaps the pattern itself — skip
            continue;
        }
        if &committed[i..i + ngram] == pattern {
            let draft_start = i + ngram;
            let drafts_avail = tail_start.saturating_sub(draft_start);
            if drafts_avail == 0 {
                continue;
            }
            let n = drafts_avail.min(max_drafts);
            return committed[draft_start..draft_start + n].to_vec();
        }
    }
    Vec::new()
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_lookup_index_matches_backward_scan() {
        let seq = [
            10, 20, 30, 40, 20, 30, 41, 42, 20, 30, 50, 60, 41, 42, 70, 20, 30,
        ];
        for ngram in 1..=4 {
            for max_drafts in 1..=6 {
                let mut index = PromptLookupIndex::new(ngram);
                for len in 0..=seq.len() {
                    let committed = &seq[..len];
                    assert_eq!(
                        index.drafts(committed, max_drafts),
                        prompt_lookup_drafts(committed, ngram, max_drafts),
                        "len={len} ngram={ngram} max_drafts={max_drafts}"
                    );
                }
            }
        }
    }

    #[test]
    fn repeated_tail_pattern_drafts_require_min_repeats() {
        assert_eq!(
            repeated_tail_pattern_drafts(&[1, 2, 2, 2], 3, 4, 4),
            vec![2, 2, 2, 2],
        );
        assert_eq!(
            repeated_tail_pattern_drafts(&[7, 8, 7, 8, 7, 8], 3, 4, 5),
            vec![7, 8, 7, 8, 7],
        );
        assert!(repeated_tail_pattern_drafts(&[1, 2, 2], 3, 4, 4).is_empty());
        assert!(repeated_tail_pattern_drafts(&[1, 2, 2, 2], 3, 4, 0).is_empty());
    }
}

/// Run greedy speculative decoding for one Qwen 3.6 request.
///
/// Contract is intentionally similar to the per-token decode
/// loop in `cuda_worker.rs::qwen36`: caller passes the prompt,
/// max_new, stop-token list, and an `on_token` callback that
/// returns `false` to short-circuit. Returns `completion_tokens`
/// (matches the existing accounting).
///
/// Verify-step KV writes: see module doc — no explicit
/// rollback; the next iter's overlapping writes cover the
/// stale-draft slots.
pub fn run_qwen36_prompt_lookup_spec<F>(
    qwen: &Qwen36Bringup,
    prompt_ids: &[i32],
    max_new_tokens: u32,
    stop_token_ids: &[u32],
    vision_splice: &[(usize, &[u8])],
    cancel: Option<&AtomicBool>,
    mut on_token: F,
) -> Result<(u32, &'static str)>
where
    F: FnMut(u32, u32) -> bool,
{
    let prompt_len = prompt_ids.len() as u32;

    // Tuning knobs (env-gated so they can be A/B'd without
    // a rebuild).
    let spec_k: usize = std::env::var("RVLLM_QWEN36_SPEC_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let ngram: usize = std::env::var("RVLLM_QWEN36_SPEC_NGRAM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let min_drafts_for_verify: usize = std::env::var("RVLLM_QWEN36_SPEC_MIN_DRAFTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(spec_k)
        .clamp(1, spec_k.max(1));
    let zero_accept_bailout_iters: u32 =
        std::env::var("RVLLM_QWEN36_SPEC_ZERO_ACCEPT_BAILOUT_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
    let repeat_run_min: usize = std::env::var("RVLLM_QWEN36_SPEC_REPEAT_RUN_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let repeat_pattern_max: usize = std::env::var("RVLLM_QWEN36_SPEC_REPEAT_PATTERN_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let prompt_lookup_warmup_matches: u32 =
        std::env::var("RVLLM_QWEN36_SPEC_PROMPT_LOOKUP_WARMUP_MATCHES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
    let mtp_shadow = std::env::var("RVLLM_QWEN36_MTP_SHADOW").as_deref() == Ok("1");
    let mtp_draft = std::env::var("RVLLM_QWEN36_MTP_DRAFT").as_deref() == Ok("1");
    let perf_trace = std::env::var("RVLLM_QWEN36_SPEC_PERF_TRACE").as_deref() == Ok("1");

    // (0) Allocate scratch snapshot buffers for recurrent state.
    //     Sized to the bring-up's persistent linear+conv state
    //     so a full snapshot/restore round-trip preserves byte-
    //     for-byte equality. Lives on the qwen arena alongside
    //     the per-request scratch (cuda_worker restores below).
    let (linear_bytes, conv_bytes) = qwen.recurrent_state_bytes();
    let snap_linear = qwen
        .arena
        .region("qwen36_spec_snap_linear", linear_bytes, 16)?;
    let snap_conv = qwen.arena.region("qwen36_spec_snap_conv", conv_bytes, 16)?;
    let snap_linear_ptr = snap_linear.device_ptr();
    let snap_conv_ptr = snap_conv.device_ptr();

    // (1) Initial prefill: feed the whole prompt at start_position=0.
    //     Returns the argmax of the LAST prompt position — the
    //     first generated token.
    let mtp_active = mtp_shadow || mtp_draft;
    let (mut next_token, mut pending_mtp_shadow) = if mtp_active {
        qwen.forward_qwen36_decode_cancellable_with_mtp_shadow(
            prompt_ids,
            0,
            vision_splice,
            cancel,
        )?
    } else {
        (
            qwen.forward_qwen36_decode_cancellable(prompt_ids, 0, vision_splice, cancel)?,
            None,
        )
    };
    if next_token < 0 {
        next_token = 0;
    }
    // session.committed = prompt + bonus_so_far. Treated as u32 for
    // the drafter pattern match; the forward calls keep i32.
    let mut committed: Vec<u32> =
        Vec::with_capacity(prompt_ids.len() + (max_new_tokens as usize) + 8);
    committed.extend(
        prompt_ids
            .iter()
            .map(|&t| if t < 0 { 0u32 } else { t as u32 }),
    );
    // The first generated token isn't yet emitted to the caller —
    // we emit only after committed.push(token) below to keep the
    // ordering symmetric.

    let mut completion_tokens: u32 = 0;
    let mut iter_count: u32 = 0;
    let mut verify_iters: u32 = 0;
    let mut total_drafted: u32 = 0;
    let mut total_accepted: u32 = 0;
    let mut zero_accept_iters: u32 = 0;
    let mut bailout_tokens: u32 = 0;
    let mut skipped_short_drafts: u32 = 0;
    let mut repeated_draft_iters: u32 = 0;
    let mut prompt_lookup_probe_iters: u32 = 0;
    let mut prompt_lookup_probe_matches: u32 = 0;
    let mut prompt_lookup_match_streak: u32 = 0;
    let mut mtp_shadow_compares: u32 = 0;
    let mut mtp_shadow_matches: u32 = 0;
    let mut mtp_verify_iters: u32 = 0;
    let mut mtp_accepted: u32 = 0;
    let mut current = next_token; // base's last-emit-candidate
    let mut prompt_lookup = PromptLookupIndex::new(ngram);

    let t0_session = if perf_trace {
        Some(std::time::Instant::now())
    } else {
        None
    };

    // Per-iter arena checkpoint — placed AFTER the snap_linear /
    // snap_conv allocations so they survive every restore. Codex
    // review found that forward_qwen36_decode_argmax_all and the
    // closer K-times path allocate hidden_fp8 / hidden_scale /
    // logits / token regions per call without freeing them; over
    // many spec iters this grows the per-request arena unbounded.
    // Restoring at the end of each iter bounds peak usage to one
    // iter's footprint.
    let iter_ck = qwen.arena.checkpoint();

    macro_rules! finish {
        ($reason:expr) => {{
            unsafe {
                qwen.arena.restore(iter_ck);
            }
            if let Some(t0) = t0_session {
                let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "[qwen36-spec-perf] reason={} iters={iter_count} \
                     verify_iters={verify_iters} drafted={total_drafted} \
                     accepted={total_accepted} accept_per_verify={:.2} \
                     bailout_tokens={bailout_tokens} \
                     skipped_short_drafts={skipped_short_drafts} \
                     repeated_draft_iters={repeated_draft_iters} \
                     prompt_lookup_probe_iters={prompt_lookup_probe_iters} \
                     prompt_lookup_probe_matches={prompt_lookup_probe_matches} \
                     mtp_shadow_compares={mtp_shadow_compares} \
                     mtp_shadow_matches={mtp_shadow_matches} \
                     mtp_verify_iters={mtp_verify_iters} \
                     mtp_accepted={mtp_accepted} \
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
        if let Some(c) = cancel {
            if c.load(std::sync::atomic::Ordering::Relaxed) {
                finish!("cancelled");
            }
        }

        // Emit `current` first (the token at position prompt_len + completion_tokens).
        let current_u32 = if current < 0 { 0u32 } else { current as u32 };
        if stop_token_ids.contains(&current_u32) {
            finish!("stop");
        }
        if !on_token(current_u32, prompt_len + completion_tokens) {
            finish!("cancelled");
        }
        committed.push(current_u32);
        completion_tokens += 1;
        if completion_tokens >= max_new_tokens {
            break;
        }

        // (2) Propose drafts via prompt-lookup. If the match can not
        //     supply enough lookahead, fall back to a single 1-token
        //     decode. A short 1-2 token draft still pays the full
        //     recurrent-state snapshot/verify/rollback cost and has
        //     been slower than base greedy on ordinary chat; require
        //     enough draft depth before entering the verifier.
        let mut drafts =
            repeated_tail_pattern_drafts(&committed, repeat_run_min, repeat_pattern_max, spec_k);
        let drafts_from_prompt_lookup;
        if !drafts.is_empty() {
            repeated_draft_iters += 1;
            drafts_from_prompt_lookup = false;
        } else {
            drafts = prompt_lookup.drafts(&committed, spec_k);
            drafts_from_prompt_lookup = !drafts.is_empty();
        }

        if drafts.len() < min_drafts_for_verify {
            if !drafts.is_empty() {
                skipped_short_drafts += 1;
            }
            if mtp_draft {
                if let Some(mtp_tok_i32) = pending_mtp_shadow.take() {
                    let mtp_tok = if mtp_tok_i32 < 0 {
                        0u32
                    } else {
                        mtp_tok_i32 as u32
                    };
                    let base_pos = prompt_len + completion_tokens - 1;
                    let verify_input = [current, mtp_tok as i32];
                    total_drafted += 1;
                    qwen.snapshot_recurrent_state(snap_linear_ptr, snap_conv_ptr)?;
                    let argmax_at =
                        qwen.forward_qwen36_decode_argmax_all(&verify_input, base_pos, &[], cancel)?;
                    verify_iters += 1;
                    mtp_verify_iters += 1;
                    if let Some(&actual) = argmax_at.first() {
                        mtp_shadow_compares += 1;
                        if mtp_tok_i32 == actual {
                            mtp_shadow_matches += 1;
                        }
                    }
                    if argmax_at.first().copied().unwrap_or(-1) >= 0
                        && argmax_at[0] as u32 == mtp_tok
                    {
                        total_accepted += 1;
                        mtp_accepted += 1;
                        zero_accept_iters = 0;
                        current = argmax_at.get(1).copied().unwrap_or(0);
                        if stop_token_ids.contains(&mtp_tok) {
                            finish!("stop");
                        }
                        if !on_token(mtp_tok, prompt_len + completion_tokens) {
                            finish!("cancelled");
                        }
                        committed.push(mtp_tok);
                        completion_tokens += 1;
                        iter_count += 1;
                        unsafe {
                            qwen.arena.restore(iter_ck);
                        }
                        continue;
                    } else {
                        qwen.restore_recurrent_state(snap_linear_ptr, snap_conv_ptr)?;
                        qwen.forward_qwen36_decode_commit_only(&[current], base_pos, &[], cancel)?;
                        current = argmax_at.first().copied().unwrap_or(0);
                        zero_accept_iters += 1;
                        iter_count += 1;
                        unsafe {
                            qwen.arena.restore(iter_ck);
                        }
                        continue;
                    }
                }
            }
            // Fallback: single-token decode at current position.
            let pos = prompt_len + completion_tokens - 1;
            // `current` is the freshly-emitted token; its K/V at
            // slot `pos` was NOT yet written (only the prefill +
            // any prior verify steps populated slots up through
            // pos - 1). Run a 1-token forward to write slot `pos`
            // and get the next prediction.
            let (t, next_shadow) = if mtp_active {
                qwen.forward_qwen36_decode_cancellable_with_mtp_shadow(
                    &[current],
                    pos,
                    &[],
                    cancel,
                )?
            } else {
                (
                    qwen.forward_qwen36_decode_cancellable(&[current], pos, &[], cancel)?,
                    None,
                )
            };
            if let Some(pred) = pending_mtp_shadow.take() {
                mtp_shadow_compares += 1;
                if pred == t {
                    mtp_shadow_matches += 1;
                }
            }
            pending_mtp_shadow = next_shadow;
            current = if t < 0 { 0 } else { t };
            iter_count += 1;
            unsafe {
                qwen.arena.restore(iter_ck);
            }
            continue;
        }

        // Prompt-lookup drafts can be plentiful but wrong for summarisation
        // prompts: one zero-accept K-token verify is already more expensive
        // than a native greedy step. Optionally require a small streak of
        // one-token base predictions that agree with prompt lookup before
        // paying the full snapshot + K-token verifier cost. These warm-up
        // probes commit exactly the same token the native greedy path would
        // have emitted, so they cannot change output quality.
        if drafts_from_prompt_lookup
            && prompt_lookup_warmup_matches > 0
            && prompt_lookup_match_streak < prompt_lookup_warmup_matches
        {
            let pos = prompt_len + completion_tokens - 1;
            let (t, next_shadow) = if mtp_active {
                qwen.forward_qwen36_decode_cancellable_with_mtp_shadow(
                    &[current],
                    pos,
                    &[],
                    cancel,
                )?
            } else {
                (
                    qwen.forward_qwen36_decode_cancellable(&[current], pos, &[], cancel)?,
                    None,
                )
            };
            if let Some(pred) = pending_mtp_shadow.take() {
                mtp_shadow_compares += 1;
                if pred == t {
                    mtp_shadow_matches += 1;
                }
            }
            pending_mtp_shadow = next_shadow;
            current = if t < 0 { 0 } else { t };
            prompt_lookup_probe_iters += 1;
            if drafts[0] == current as u32 {
                prompt_lookup_probe_matches += 1;
                prompt_lookup_match_streak += 1;
            } else {
                prompt_lookup_match_streak = 0;
            }
            iter_count += 1;
            unsafe {
                qwen.arena.restore(iter_ck);
            }
            continue;
        }

        // (3) Verify: run base forward ONCE over
        //     [current, d0, d1, ..., d_{K-1}] starting at position
        //     `prompt_len + completion_tokens - 1`. Returns argmaxes
        //     at every input position via `forward_qwen36_decode_argmax_all`.
        //
        //     Qwen 3.6 has linear-attn (Gated DeltaNet) layers whose
        //     state is RECURRENT — a naive "call forward K times with
        //     growing prefixes" approach (idempotent under full-attn-
        //     only models like Gemma) corrupts the linear state. The
        //     single-call closer-all path is the only correct option.
        //
        //     KV writes: this single call writes K+1 slots
        //     [base_pos, base_pos+K]. If accept_len = m, slots
        //     [base_pos, base_pos+m] are valid (bonus + accepted
        //     drafts); [base_pos+m+1, base_pos+K] are stale from
        //     rejected drafts but the NEXT iter's verify starts at
        //     base_pos+m+1 and overwrites them.
        let base_pos = prompt_len + completion_tokens - 1;
        let kmax = drafts.len();
        total_drafted += kmax as u32;

        let mut verify_input: Vec<i32> = Vec::with_capacity(kmax + 1);
        verify_input.push(current);
        for d in &drafts {
            verify_input.push(*d as i32);
        }

        // SNAPSHOT recurrent state before verify (which would
        // otherwise advance it through ALL K drafts in place).
        qwen.snapshot_recurrent_state(snap_linear_ptr, snap_conv_ptr)?;

        let argmax_at =
            qwen.forward_qwen36_decode_argmax_all(&verify_input, base_pos, &[], cancel)?;
        verify_iters += 1;
        if let Some(pred) = pending_mtp_shadow.take() {
            if let Some(&actual) = argmax_at.first() {
                mtp_shadow_compares += 1;
                if pred == actual {
                    mtp_shadow_matches += 1;
                }
            }
        }

        // (4) Acceptance: argmax_at[i] is the prediction at position
        //     base_pos + i given inputs prefix[0..=i].
        //     - argmax_at[0]: predicts the token AFTER `current`. Compare
        //       to drafts[0] to decide acceptance of d0.
        //     - argmax_at[i]: predicts the token AFTER drafts[i-1]. Compare
        //       to drafts[i] to decide acceptance of d_i.
        //     - argmax_at[kmax]: the "bonus" prediction past all K drafts.
        let mut accept_len: usize = 0;
        for i in 0..kmax {
            if argmax_at[i] >= 0 && argmax_at[i] as u32 == drafts[i] {
                accept_len += 1;
            } else {
                break;
            }
        }
        total_accepted += accept_len as u32;

        // (4a) Recurrent-state rollback FIRST — codex review
        //      (MEDIUM): emitting accepted drafts before rollback
        //      means a stop/length/cancel mid-emit early-returns
        //      with the persistent state polluted by rejected
        //      drafts. Today the cross-request reset covers it,
        //      but a future session-continuation path would
        //      inherit the dirty state. Doing rollback first
        //      makes the path correct under all early-return
        //      cases.
        if accept_len < kmax {
            qwen.restore_recurrent_state(snap_linear_ptr, snap_conv_ptr)?;
            // commit prefix = [current] + accepted drafts.
            let mut commit_input: Vec<i32> = Vec::with_capacity(accept_len + 1);
            commit_input.push(current);
            for i in 0..accept_len {
                commit_input.push(drafts[i] as i32);
            }
            qwen.forward_qwen36_decode_commit_only(&commit_input, base_pos, &[], cancel)?;
        }
        // On accept_len == kmax: state is already correct (every
        // verify advance was a real commit), nothing to roll back.

        // (4b) Set the next `current` = base's prediction at
        //      accept_len. If all K accepted, that's argmax_at[kmax]
        //      (bonus past all drafts). Otherwise argmax_at[accept_len]
        //      (which diverged from drafts[accept_len]).
        current = argmax_at[accept_len];

        if accept_len == 0 {
            zero_accept_iters += 1;
            prompt_lookup_match_streak = 0;
        } else {
            zero_accept_iters = 0;
            if drafts_from_prompt_lookup {
                prompt_lookup_match_streak = prompt_lookup_warmup_matches;
            }
        }

        // (5) Emit accepted drafts. Safe to early-return now —
        //     recurrent state already matches the committed prefix.
        for i in 0..accept_len {
            let tok = drafts[i];
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
        if zero_accept_bailout_iters > 0 && zero_accept_iters >= zero_accept_bailout_iters {
            while completion_tokens < max_new_tokens {
                if let Some(c) = cancel {
                    if c.load(std::sync::atomic::Ordering::Relaxed) {
                        finish!("cancelled");
                    }
                }

                let current_u32 = if current < 0 { 0u32 } else { current as u32 };
                if stop_token_ids.contains(&current_u32) {
                    finish!("stop");
                }
                if !on_token(current_u32, prompt_len + completion_tokens) {
                    finish!("cancelled");
                }
                committed.push(current_u32);
                completion_tokens += 1;
                bailout_tokens += 1;
                if completion_tokens >= max_new_tokens {
                    break;
                }

                let pos = prompt_len + completion_tokens - 1;
                let (t, next_shadow) = if mtp_active {
                    qwen.forward_qwen36_decode_cancellable_with_mtp_shadow(
                        &[current],
                        pos,
                        &[],
                        cancel,
                    )?
                } else {
                    (
                        qwen.forward_qwen36_decode_cancellable(&[current], pos, &[], cancel)?,
                        None,
                    )
                };
                if let Some(pred) = pending_mtp_shadow.take() {
                    mtp_shadow_compares += 1;
                    if pred == t {
                        mtp_shadow_matches += 1;
                    }
                }
                pending_mtp_shadow = next_shadow;
                current = if t < 0 { 0 } else { t };
                unsafe {
                    qwen.arena.restore(iter_ck);
                }
            }
        }
        // End-of-iter arena restore — bounds peak per-request
        // arena usage to one iter's footprint regardless of how
        // many spec iters fire. The snap_linear / snap_conv
        // allocations are above this checkpoint, so they survive.
        unsafe {
            qwen.arena.restore(iter_ck);
        }
    }

    finish!("length");
}
