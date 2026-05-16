# Speculative-decoding next-session plan — Gemma 4 E4B

## State at end of this session

Branch: `rusty_sm121_qwen36_26b` (not pushed). HEAD = `a3e9ea9`.

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
