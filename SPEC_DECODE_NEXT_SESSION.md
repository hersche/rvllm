# Speculative-decoding next-session plan — Gemma 4 E4B

## Status (2026-05-22 session update — head `3a40819`)

* **#28 — Persistent identity block table — DONE** (`9845395`).

* **#26 — `verify_batched_suffix_k_only` — LANDED, BYTE-EQUIV PENDING**
  (`3f9febd` Phase 1 scratch + scaffold, `4449903` Phase 2 body,
  `4a43405` Phase 4 wire-in, `3a40819` debug). The new method exists
  on `Gemma4Bringup`, allocates K-sized scratch via
  `prepare_spec_prefill_scratch(MAX_SPEC_K=16)`, runs one chunk of
  prefill through `gemma4_forward_phase` directly (no run_generate),
  captures K post-layer-loop residual rows + K argmaxes. Wired
  through `RVLLM_GEMMA4_SPEC_NEW_PRIMITIVES=1` env knob, default
  OFF. **Known regression on NEW path** — accept_rate drops from
  87.5% (OLD) to 15.8% (NEW verify + OLD commit) to 0.9% (NEW
  verify + NEW commit). Bisect knob `RVLLM_GEMMA4_SPEC_NEW_COMMIT=1`
  added in `3a40819` to isolate verify vs commit contribution. The
  GPU output diverges from the run_generate-driven path despite a
  faithful per-layer setup mirror; root cause not yet identified.
  Plausible candidates: missing PLE precompute (E4B only), per-
  layer scratch field bind mismatch (38 fields), `q_scale_cache`
  init shape (memset already added in `3a40819`). Next step: dump
  K-row hidden + K argmax from BOTH paths on the SAME prompt and
  diff to localize the divergence.

* **#27 — `commit_base_tokens_from_state` — LANDED, BYTE-EQUIV PENDING**
  (`4449903` Phase 2 body, `4a43405` Phase 4 wire-in). K=1 wrapper
  around `verify_batched_suffix_k_only` that returns the next
  argmax (matches `prefill_one_from_state`'s contract). Same env
  knob + same regression status as #26.

Production safety: both env knobs default to OFF.
`verify_batched_from_state` + `prefill_one_from_state` remain the
production path. Persistent identity block table (#28) is in use
on the production path through `PrefixCacheState`.

## Previous status (head `9845395`)
  The per-call `gen_bt` arena region + sync HtoD of
  `(0..num_blocks_total).collect()` now lives once, in
  `init_prefix_cache`. Exposed through new fields
  `PrefixCacheState::identity_block_tables_ptr` +
  `identity_block_tables_len`. `run_generate` reads the ptr
  through the prefix-cache tuple and skips the per-call HtoD
  whenever the cache is initialised. Three internal usages
  (`gemma4_forward` meta-ptrs, `gemma4_forward_phase` meta-ptrs,
  and the `RVLLM_BOUNDARY_DUMP` DtoH staging) now read from the
  persistent ptr. Smoke on production qwen3-6-27b unchanged.

* **#26 — `verify_batched_suffix_k_only` (no run_generate) —
  NOT STARTED.** ~400-600 LOC extraction of the chunked-prefill
  body (lines ~11240-11700 in `gemma4_bring_up.rs`) into a new
  method that calls `gemma4_layer_exec::gemma4_forward_phase`
  directly with `Gemma4Phase::Prefill` for K tokens at
  `start_pos`. The extraction needs to faithfully replicate
  ~150 LOC of per-layer scratch + meta setup. Risk class is
  "correctness regression vs the in-place chunked-prefill" —
  byte-equivalence against the existing
  `verify_batched_from_state` path on the three regression
  prompts before any production flip.

* **#27 — `commit_base_tokens_from_state` (K=1 case) — NOT
  STARTED.** Per codex's note ("treat `prefill_one_from_state`
  as `verify_batched_suffix(K=1)`"), this is a ~50 LOC wrapper
  on top of #26 that skips the K-row argmax buffer (commit
  doesn't need per-row argmaxes).

What changed in the perf model since the plan was first
written: the new persistent identity block table (#28) removes
one of the per-call HtoDs but the dominant cost in
`verify_avg_ms ≈ 183 ms` is GPU compute on the chunked-prefill
kernels at K=4. run_generate's host overhead (25 env reads + 38
arena.region calls per call) adds <1 ms total. The per-token
cost amortises poorly on E4B's compute-bound dense path. Even a
perfect #26 extraction is unlikely to drop `verify_avg_ms`
below ~100 ms — the layer-loop kernels at K=4 themselves
dominate. Kernel-level work (decode-tuned K≤4 variants) is the
next-after step beyond the extraction.

## State at end of e4b spec session

Branch: `rusty_sm121_qwen36_26b` (not pushed). HEAD = `6eeefb5`.

Two correctness/safety commits added since `a3e9ea9`:

| Commit | What |
|---|---|
| `6eeefb5` | spec-decode: commit-before-emit + force_batched_verify lifetime (codex round 9 #2,#4) — commit 58 |

`6eeefb5` reorders the spec loop body so the bonus's base K/V +
shadow K/V are committed BEFORE `emitted.push` / `on_token` fires;
on mid-emit abort the internal state matches the externally-visible
state. It also scopes `bringup.force_batched_verify=true` to the
batched branch only and clears it after the spec call returns, so
the atomic cannot leak into a subsequent non-spec request handled
by the same worker. No perf impact (correctness/safety only).

## Smaller intermediate option (codex round 9 follow-up)

If the full ~500 LOC `verify_batched_suffix_k_only` extraction is too
risky to attempt in one session, codex's recommended smaller patch is
~150-250 LOC: a `SpecRequestContext` plus a `run_generate` fast-spec
setup bypass. Keep the existing chunked-prefill layer loop body in
place; factor only the top of `run_generate` so the spec path can skip:

* `prefix_cache.lock()` + `last_tokens` clone / restore (already
  partially neutralised by `force_common_prefix_override` +
  `skip_prefix_cache_publish`, but the mutex round-trip + provenance
  check + clones remain). Add a stronger hook
  `force_spec_prefix_ctx: Option<&SpecRequestContext>` that lets
  `run_generate` skip the lock entirely.
* `pc.kv_layer_offsets.clone()` / `pc.kv_scale_layer_offsets.clone()` —
  carry borrowed slices in `SpecRequestContext`.
* `block_tables = (0..max_blocks_per_seq)` HtoD — upload once into
  `SpecRequestContext`.
* Repeated `arena.region(...)` checkpoint churn — pre-allocate spec
  scratch sized at `MAX_SPEC_K` (gen_qkv, gen_q_normed, gen_k_normed,
  gen_v_normed, gen_q_fp8, gen_attn_out, gen_gate_up, gen_mlp,
  gen_delta, gen_gemm_f32, gen_pos, gen_slot, gen_ctx, gen_cu_seqlens,
  gen_tok_ids, gen_residual). Re-use across verify + commit calls.
* `q_scale`, `kv_scale` HtoD — upload once.
* For K≤4: build `positions/slot_mapping/context_lens/cu_seqlens` into
  fixed `[i32; MAX_SPEC_K]` stack arrays, not `Vec`.

Codex's note on NOT doing alone: env-var hoisting, stack-building
position vecs, prefix-cache token-match skip without the scratch
hoisting — each by itself is single-digit ms. Bundle them.

Codex's note on unification: treat `prefill_one_from_state` as
`verify_batched_suffix(K=1)`. Removing its second `run_generate`
entry + `last_tokens` mutation is the highest-leverage single delete.

5 spec-decode commits land on top of `3b5fb4d`:

| Commit | What |
|---|---|
| `0f19828` | Session-loop refactor + Rounds 1–5 structural cleanup (~1450 insertions) |
| `c9c0b21` | Drop `force_common_prefix_override` in `prefill_one_from_state` (correctness) |
| `dea6f54` | **Drop `sqrt(hidden_size)` scale on drafter inputs_embeds** — root cause of `accept_rate=0` |
| `0ffc5e2` | Classical `accept_len + 1` emit (codex Round 7 #2 + #5) |
| `a3e9ea9` | bf16-aware K-row final_norm + skip bonus decode step (codex Round 7 perf) |

There's also a `b4db277` revert in the history sandwiched between `a12fe83` and `a3e9ea9`; both are no-op pairs and can be squashed if the user prefers a cleaner log, but per the "no destructive git" rule they're left in place.

## Measured state

E4B fp8-block, `RVLLM_F16_KV=0`, `RVLLM_BATCH_PREFILL=1`, K=4, greedy, temperature=0:

| Workload | Non-spec | Spec ON | Ratio |
|---|---|---|---|
| 128-token creative continuation, `accept_per_verify ≈ 0.15` | ~25 s (median) | ~50–55 s (median) | ~2× SLOWER |
| Short factual ("Was ist die Hauptstadt von Italien?"), `accept_per_verify = 4.0` | ~450 ms / 2 tok | ~600 ms / 2 tok | ~1.35× SLOWER |

Correctness: byte-identical to non-spec on the three regression prompts (`Hallo, mein Name ist`, `The capital of France is`, `Once upon a time, in a`). Output coherence preserved on all tested prompts. No regressions vs non-spec path on the same binary.

## Why "huge perf boost" is NOT met

Mathematical cost model from measured numbers:

- `verify_batched_from_state` (K=4 chunked prefill via `run_generate` with `force_common_prefix_override`) ≈ **255 ms** — close to a single base-decode step due to GPU FFN amortization across K rows.
- `prefill_one_from_state` (K=1 chunked prefill, post `a3e9ea9` skips the decode-step's full forward) ≈ **240 ms** — basically one base-decode equivalent.
- Per-iter cost classical: 255 + 240 ≈ **495 ms**.
- Per-iter emit average at `accept_per_verify` = 0.15: 1 (bonus) + 0.15 (drafts) = **1.15 tokens**.
- Cost / token: 495 / 1.15 ≈ **430 ms** (matches measured ~500 ms with variance).

Non-spec is **240 ms/tok**. The architecture's cost ceiling at full-accept (`accept_len = K`) is `(K + 1) / (K + 1) = 1.0 decode-equiv/tok` — parity, never below. At partial accept it's strictly slower because each `prefill_one_from_state` adds ~1 decode-equivalent for one extra emitted token, plus the K-token verify cost which is already ~1 decode-equivalent.

The only way to beat parity is to make verify's per-call cost **less than one base-decode equivalent**. That requires bypassing `run_generate`'s host-side setup (env reads, region allocations sized to `prompt_len`, prefix-cache lookup, position/slot_mapping host vec builds, layer-loop dispatch overhead) — a ~400-600 LOC extraction of the chunked-prefill body that I judged too risky to attempt safely within a single session without breaking correctness.

## Concrete plan for the next session

### 1. Build `verify_batched_suffix_k_only`

New method on `Gemma4Bringup`:

```rust
unsafe fn verify_batched_suffix_k_only(
    &self,
    fn_embed: rvllm_kernels::KernelFn,
    ctx: &SpecRequestContext,        // see below — built once per request
    new_tokens: &[u32],              // K (or K=1) tokens to prefill
    start_pos: u32,                  // committed_len_kv
    k_hidden_out: u64,               // device ptr, sized to K * hidden * 2 bytes
    k_argmax_out: &mut [u32],        // host buf, len K
) -> Result<()>
```

The body is the chunked-prefill block from `run_generate` (currently at lines ~10091–10611 of `v3/crates/rvllm-runtime/src/gemma4_bring_up.rs`), adapted as follows:

- Scratch regions (`gen_qkv`, `gen_q_normed`, `gen_k_normed`, `gen_v_normed`, `gen_q_fp8`, `gen_attn_out`, `gen_attn_out_fp8`, `gen_attn_out_scale`, `gen_gate_up`, `gen_gate_up_fp8`, `gen_gate_up_scale`, `gen_mlp_fp8`, `gen_mlp_scale`, `gen_delta`, `gen_gemm_f32`) sized to **K**, not `prompt_len`.
- No chunk loop — exactly one chunk of K tokens.
- `chunk_start_abs = start_pos`, `chunk_end_abs = start_pos + K`. `pos = [start_pos..start_pos+K)`, `slot = same`, `context_lens = [start_pos + K]`, `cu_seqlens_q = [0, K]`.
- Reuse `ctx.block_tables_region.device_ptr()` (built once in `ctx`, identity `[0..num_blocks_total)`).
- Per-layer state: `ctx.kv_dtype_per_layer[layer_idx]`, `ctx.kv_layer_offsets[kv_idx]`, etc., all pre-computed.
- Embed N tokens directly into a K-sized residual region; bf16 widen if `bf16_residual_enabled()`.
- PLE precompute if `RVLLM_E4B_PLE=1` (only needed if start_pos == 0, which never happens in verify suffix; can skip — verify always runs with committed prefix already containing PLE-augmented embeddings).
- Layer loop: 42 iterations calling `gemma4_layer_exec::gemma4_forward_phase` with `Gemma4Phase::Prefill { cu_seqlens_q, max_seqlen_q: K, num_seqs: 1 }`.
- After layer loop: `cuMemcpyDtoDAsync` residual rows [0..K) → `k_hidden_out`.
- `final_norm_inplace_bf16/f16` over `k_hidden_out` rows + `bf16_to_f16_sat` narrow if bf16.
- `cublaslt.f16_gemm_f32` over `k_hidden_out` × `lm_head_f16` → K-row logits.
- Softcap if `logit_softcap > 0.0`.
- Argmax over K rows → device buf → DtoH → `k_argmax_out`.

### 2. Build `SpecRequestContext`

```rust
struct SpecRequestContext {
    block_tables_region: ArenaRegion,    // identity [0..num_blocks_total), built once
    kv_dtype_per_layer: Vec<KvDtype>,    // pre-computed
    kv_base_ptr: u64,
    kv_scale_base_ptr: u64,
    kv_layer_offsets: Vec<u64>,
    kv_scale_layer_offsets: Vec<u64>,
    num_blocks_total: u32,
    block_size: u32,
    sliding_blocks: u32,
    sources: Gemma4AssistantKvSources,
    sliding_source_view: (u64, u64, u64, u64),  // (k, v, k_scale, v_scale)
    full_source_view: (u64, u64, u64, u64),
    shadow_sliding_bytes: usize,
    shadow_full_bytes: usize,
}
```

Built once per request at the top of `run_generate_speculative_batched` (after warmup, before the spec loop). Passed by reference to `run_drafter_k_from_state` (replaces its in-method computation of the same data) and to the new `verify_batched_suffix_k_only` and a new `commit_one_token_suffix` (which is just `verify_batched_suffix_k_only` with K=1).

### 3. Use `verify_batched_suffix_k_only` in the spec loop

- Replace `verify_batched_from_state` call with the new K-only primitive.
- Replace `prefill_one_from_state`'s internal `run_generate` call with `verify_batched_suffix_k_only(K=1)`.
- Drop `force_common_prefix_override` / `skip_prefix_cache_publish` / `force_prefill_only` / `base_last_k_snapshot_pending` / `base_last_k_hidden_ptr.swap` atomics — they were workarounds for driving `run_generate` indirectly. With direct invocation they're not needed.
- Drop `SpecHookGuard` for the verify + commit paths (still needed for the warmup `run_generate` call).

### 4. Validation plan

- `cargo check -p rvllm-runtime --features cuda`.
- `cargo test -p rvllm-runtime spec_decode_session_tests --features cuda` (5/5 pass currently).
- `cargo build --release --bin rvllm-server --features cuda,gb10`.
- Hardware smoke: spec ON vs spec OFF on three test prompts (`Hallo, mein Name ist`, `The capital of France is`, `Once upon a time, in a`) at `max_tokens=24, temperature=0`. Expect byte-identical output.
- Hardware bench: 128-token creative continuation. Target: spec ON ≤ 25 s median (= parity with non-spec). Stretch: < 20 s (1.25× faster).

### 5. Risk and rollback

If correctness diverges or build doesn't stabilize, `git revert` the new commits — none touch the existing `run_generate` or `gemma4_layer_exec::gemma4_forward_phase` paths, so reverting cleanly returns to current `a3e9ea9` working state.

## What absolutely should NOT change next session

- The `dea6f54` embed-scale fix (it's the root-cause correctness fix that turned `accept_rate = 0` into `accept_rate > 0`).
- The F16-KV hard reject (Round 5 #1, in `0f19828`) — it prevents silent FP8/F16 cache corruption.
- Vision/audio splice hard reject (Round 4 #4).
- EOS-before-push parity (Round 4 #3).
- Drafter `"stable"` scale default (HF-parity verified, in `dea6f54`).
- `SpecHookGuard` (RAII for the legacy `run_generate`-based warmup path — keep it for that call only).

## Per-iter timing data (added end-of-session via `RVLLM_GEMMA4_SPEC_PERF_TRACE=1`)

Measured 128-token creative continuation, K=4, FP8 KV, bf16 residual, GB10:

```
[spec-perf] iters=109
            drafter_avg_ms=8.20
            verify_avg_ms=183.23      <- K=4 chunked prefill via run_generate
            prefill_one_avg_ms=213.98 <- K=1 chunked prefill via run_generate
            shadow_us_total=1365      <- negligible
```

**The decisive observation:** `prefill_one` at K=1 (214ms) is nearly as
expensive as `verify` at K=4 (183ms). The K-scaled portion of GPU
work is small; the dominant cost is `run_generate`'s per-call host
overhead — env reads, region allocations (mitigated by commit
`2fb7a14`), prefix-cache lookup, position vec build, kernel launch
dispatch.

Per-iter total: 8 + 183 + 214 = 405 ms. For 109 iters: 44.1 s wall
time. Matches measured.

Non-spec baseline: 240 ms/token.

**Implication for the kernel extraction**: even if a `verify_suffix_k_only`
primitive matches the K=4 GPU work cost exactly (183ms), and a
similar `commit_one_from_state_k_only` matches the K=1 GPU work
cost (~80ms — only the actual K=1 forward, no setup overhead), per
iter cost drops to ~270 ms (drafter 8 + verify 183 + commit 80 = 271).
At observed accept_per_verify=0.15 (creative): emit_per_iter = 1.15.
cost/token = 271 / 1.15 = 236 ms. **Matches non-spec 240 ms — JUST
at parity, not "huge boost".**

For real speedup, the setup overhead reduction has to be combined
with a higher accept rate workload (factual content) where
emit_per_iter ≥ 2. There, cost/token = 271/3 ≈ 90 ms, a clear ~2.6×
speedup vs non-spec.

**Recommended next-session ordering:**
1. First implement the perf trace as a permanent diagnostic surface
   (already done — commit `7f58464`).
2. Then attempt `verify_batched_suffix_k_only` extraction (codex Round
   7 #1) and measure: target `verify_avg_ms` ≤ 150 ms (vs current 183).
3. Then attempt `commit_base_token_from_state` (codex Round 7 #1 / K=1
   variant): target `prefill_one_avg_ms` ≤ 80 ms (vs current 214).
4. Validate spec ON vs OFF byte-identity on the three regression
   prompts after each step.
5. Final bench: target ≤ 28 s on the 128-token creative continuation
   (= parity with non-spec). Stretch: ≤ 20 s.
