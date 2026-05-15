# Spec-decode batched verify — design + status

## Where we are (post commit 35)

- Drafter forward is **correct** vs HF/PyTorch reference (commit 32 +
  33 fixed pre_projection sqrt scaling + wrong GELU kernel handle).
- Calibration knob `RVLLM_GEMMA4_SPEC_ACCEPT_BIAS` produces measurable
  accept_rate from 0% (bias=0) to 96% (bias=32) on the typical-
  acceptance ratio test (commit 34).
- `RVLLM_GEMMA4_SPEC_EMIT_ACCEPTED=1` makes
  `run_generate_speculative` return `accept_len + 1` tokens from
  the accepted drafter prefix (commit 35).
- **No wall-clock speedup yet**: `run_generate_speculative` still
  calls `self.run_generate(max_new=user_max)` internally, which
  runs N sequential base decodes regardless. Drafter forward
  adds overhead but saves no base work.

## What's missing for actual speedup

The inner loop needs to be:

```text
loop while emitted < max_new:
    // Drafter forward K=K times (chained), produce K candidate tokens
    drafts = [run_drafter_step(input_i) for i in 0..K]

    // Base batched verify in ONE forward pass:
    //   - Input: drafts[0..K]  (K tokens)
    //   - Positions: [current_pos, ..., current_pos + K - 1]
    //   - KV state: prompt + previously-committed tokens
    //   - Output: K logit rows (one per draft position)
    //   - Side effect: writes K new K/V slots to base cache
    logits[K, vocab] = base.verify_batched(drafts, current_pos)

    // Accept under typical-acceptance ratio test (commit 34):
    accept_len = longest_prefix_where(
        u < min(1, p_base(draft_i) / p_drafter(draft_i)) + bias
    )

    // Emit accepted prefix + 1 bonus base token at the divergence:
    emitted_now = drafts[0..accept_len] + argmax(logits[accept_len])
    output.extend(emitted_now)

    // Critical: roll back rejected tail K/V slots.
    // K - accept_len - 1 slots written but not committed.
    base.rewind_kv(rejected_count = K - accept_len - 1)

    current_pos += accept_len + 1
```

Per-iteration cost: ~1 batched prefill (the K-verify) + 1 drafter
forward (~600 µs for K=4). Per-token cost amortizes accept_len+1
tokens against ~one prefill of length K.

Per Qwen 3.6 batched-prefill numbers in this repo, batched prefill
of N tokens runs at **1.77×–2.69× the throughput of N sequential
decodes** at N=22..293. For E4B at K=4 that should land in the
2–3× window — matching the documented public benchmark.

## Implementation pieces required

### Piece A: base `verify_batched(input_ids[K], start_pos)` method

Build it by **reusing the existing chunked-prefill batch path**.
Vision splice already does `num_query_tokens > 1` with custom
positions; this is the analogous "extend from existing prompt KV"
path.

Surface (in `Gemma4Bringup`):

```rust
pub unsafe fn verify_batched(
    &self,
    fn_embed: KernelFn,
    fn_argmax: KernelFn,
    input_ids: &[u32],     // K verify tokens
    start_pos: u32,         // their absolute positions begin here
    // logits output buffer of shape [K, vocab]
    out_logits_f32: u64,
    stream: u64,
) -> Result<Vec<u32>> {     // returns K argmax tokens (or per-row top-1 if useful)
```

Internal flow (mirror prompt prefill but skip embedding-from-prefix
and use `start_pos` as `tok_offset`):

1. EmbeddingGather for K input_ids
2. Build per-token `positions[K] = [start_pos..start_pos+K]`
3. Build `slot_mapping[K]` (allocate K new slots after current
   committed-length)
4. Build `cu_seqlens_q[2] = [0, K]` and `cu_seqlens_k[2] = [0,
   start_pos+K]` for the FA-3 / FA-2 prefill kernel
5. Run the full layer stack via the same chunked-prefill machinery
   that prompt prefill uses
6. Final norm
7. lm_head GEMM for ALL K positions → `out_logits_f32[K, vocab]`
8. argmax over each row → return `Vec<u32>` of K base argmaxes

Risks:
- Chunked-prefill currently assumes `start_pos = 0`. The kernel
  reads `positions[t]` for RoPE which already supports any value,
  but the slot_mapping convention and cu_seqlens_k handling for
  pre-existing prefix may need tweaks.
- `prefix_cache` invalidation on each verify call (its committed
  length increases by accept_len+1 per spec iteration).

### Piece B: KV-slot rollback on rejection

When `accept_len < K - 1`, K - accept_len - 1 K/V slots were
written for nothing. Three options ranked by complexity:

i. **No rollback** (simplest, possibly incorrect): rely on the
   next iteration's slot allocator to overwrite. Works if slot
   allocator allocates linearly from current committed-length
   pointer, but the **rejected slots remain in the cache for the
   length of the next batched-prefill** — base reads them during
   attention as if committed. That gives wrong context.

ii. **Wind back committed_length pointer** to
    `start_pos + accept_len + 1` after each verify. The next
    iteration's verify will overwrite the rejected K/V slots.
    Needs careful audit of how `prefix_cache` tracks state.

iii. **Scratch region** for verify-K writes; copy in on accept.
     Doubles the prefill memory traffic. Safest.

Recommend (ii): wind back committed_length. ~50 LOC if
prefix_cache is amenable.

### Piece C: iterative spec-decode loop in `run_generate_speculative`

Replace the current `self.run_generate(max_new)` call with the
loop sketched at the top. Reuse:
- existing `run_drafter_*` helpers for the drafter forward
- existing logits-capture buffer (commit 26) for verify scores
- existing acceptance criterion (commit 29 + 34)
- existing emit-accepted mechanism (commit 35)

The only new wiring is calling `verify_batched` (piece A) and
`rewind_kv` (piece B) per iteration. ~100 LOC.

### Total scope estimate

- Piece A: 200-300 LOC (mostly mirror of prompt prefill but with
  start_pos)
- Piece B: 50 LOC
- Piece C: 100 LOC
- Tests / smokes / debug probes: 50 LOC

Realistic: 1 focused session (3-5 hours) to land safely with no
base-path regression.

## Why this session stopped before landing it

This session already landed **two real root-cause bug fixes**
(commit 32 pre_projection scale, commit 33 GELU kernel handle)
which together make the drafter mathematically correct vs HF
reference. That correctness work was the prerequisite for any
meaningful speedup measurement — without it, every speedup attempt
would be noise on top of a broken forward.

The remaining batched-verify work is well-scoped and isolated from
the correctness work. It should land in its own session with fresh
context and dedicated test cycles.
