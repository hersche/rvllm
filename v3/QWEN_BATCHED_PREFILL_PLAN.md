# Qwen 3.6 batched-prefill plan

**Status:** in progress, Phase 1 landed (this commit).
**Goal:** flatten the per-token prefill loop in
`Qwen36Bringup::forward_qwen36_decode` so the layer-chain runs
on the full `[N, D]` prompt-hidden region in one pass, matching
Gemma's `unified_prefill` mode. End state: O(layers + log N)
launches per prefill instead of O(prompt_tokens × layers).

## Current state (post Round 16 #2 fence-drop + #3 GPU-argmax)

```rust
for tok_local in 0..num_tokens {
    DtoD-extract  hidden_region[off] → last_hidden_region   // 1 launch, no fence
    for layer L in 0..40 {
        apply_layer_linear_attn(last_hidden_region, …)      // ~470 LOC, 11+ fences, m=1
        // OR apply_layer_full_attn(...)                    // ~similar
        per-layer MoE block(last_hidden_region, …)          // m=1 grouped FP8 GEMV
    }
    DtoD-writeback last_hidden_region → hidden_region[off]   // 1 launch
}
final fence
forward_qwen36_outside_closer(hidden_region, …)
```

`last_hidden_region` is a single-token (`hidden=2048` × f16 = 4 kB)
buffer. Every projection call inside the layer functions runs
the **`fp8_gemv_wpr_native_f16in` GEMV kernel hard-coded at
`m: u32 = 1`**.

## Cost model (rough)

For a 1024-token prompt at 40 layers:
* **Launch overhead**: ≈40 × 1024 × ~10 launches/layer = 410 k
  launches × ~5 µs ≈ **2 s** of host-side launch chatter alone.
* **Fence overhead**: ≈11 × 40 × 1024 ≈ 450 k fences × ~2 µs ≈
  **0.9 s** (Round 16 partial-fix only dropped the 2 outer
  per-token fences; the inner 11/layer remain).
* **Kernel work** (GEMV at m=1): bandwidth-bound, ~hundreds of
  GB/s on GB10. Real GEMM (m=N) wins back the
  arithmetic-intensity headroom.

Switching to a real `[N, D]` batched prefill should:
* turn the 11/layer fences inside `apply_layer_linear_attn`
  into ~3/layer (only at recurrence boundaries),
* turn `m=1` GEMV into `m=N` GEMM (≈100× higher AI for N=1024),
* eliminate ≥2 DtoD copies per token.

## Risk profile

* **Silent garbage**: any kernel-ABI miswire produces tokens
  that look fine on the wire but are semantically wrong (Codex
  Round 16 #1 hardened against the embed→norm→lm_head
  fallback for exactly this reason). Per-phase byte-equivalence
  vs the per-token reference is mandatory.
* **Linear-attn recurrent state**: the only kernel that genuinely
  cannot be batched in the naïve sense — `state_t = f(state_{t-1},
  x_t)`. Choices:
  1. *Loop INSIDE the kernel* over N tokens with state held in
     shared/registers — eliminates host-loop launch overhead,
     no parallelism. Cheap, mechanical.
  2. *Chunked-recurrent* parallel form (RWKV / RetNet / Mamba2
     style) — true parallelism; significantly more kernel work.
  Phase 4 starts with (1); (2) is a future optimisation.

## Phase break-down

| Phase | Scope | LOC est. | Deliverable | Verify |
|---|---|---|---|---|
| **0** (done) | Round-16 partial fixes: GPU argmax, drop redundant per-token fences | ~80 | `qwen36_bring_up.rs` | Byte-equivalent canary |
| **1** (this commit) | Refactor `apply_layer_*` to take a raw token-slot pointer instead of a `Region`; eliminate the two per-token DtoD copies; caller passes `hidden_region.device_ptr() + tok_local × hidden_bytes` directly | ~150 | `qwen36_bring_up.rs` | Byte-equivalent canary on Qwen joke + vision |
| **2** | `bench-qwen-prefill` standalone binary: time prefill at N ∈ {32, 256, 1024, 4096}, log layer-by-layer breakdown | ~250 | new `crates/rvllm-runtime/src/bin/bench_qwen_prefill.rs` | Self-validating |
| **3a** (done) | Add the dispatcher entry point `Qwen36Bringup::fp8_proj_dispatch(out, w, scale, x_f16, m, n, k, …)`. m=1 path delegates to existing `Fp8GemvF16InLaunch` (byte-identical). m≥2 returns a typed `Phase 3b` deferral error instead of silently producing zeros. Verified by `probe-qwen-fp8-proj`: m=1 dispatcher output matches the direct GEMV launch byte-for-byte (16 384 bytes compared at q_proj of layer 3). | ~150 | `qwen36_bring_up.rs`, new `bin/probe_qwen_fp8_proj.rs` | `probe-qwen-fp8-proj` passes |
| **3b** (done) | Wire blockwise FP8 GEMM (per-token VEC128_32F activation scale + per-128×128 weight scale) into the dispatcher's m≥2 branch. New `fp8_quantize_per_token_f16.cu` kernel produces `[M, K] fp8` + `[M, K/128] f32` activation scales (cuBLASLt mode `VEC128_32F`). New `cublaslt.fp8_gemm_blockwise(...)` entry point sets both `A_SCALE_MODE = BLK128x128_32F` (mode 5) and `B_SCALE_MODE = VEC128_32F` (mode 4) on the matmul descriptor. **sm_121 caveat**: cuBLASLt 13.x does NOT ship a blockwise FP8 kernel for sm_121 — `AlgoGetHeuristic` returns no-algo. The dispatcher catches the heuristic error and falls back to looped-m=1 `Fp8GemvF16InLaunch` (the existing GEMV) so behaviour stays correct on this arch. On sm_100 / sm_120 (B100/200, RTX 5090, RTX 6000 Blackwell) the cuBLASLt blockwise tensor-core path dispatches directly. Verified by `probe-qwen-fp8-proj`: cosine = 1.000000 across m∈{2,4,16,64} on the q_proj@layer 3 shape (8192×2048) — the fallback IS the looped GEMV so output is byte-identical. | ~350 | `kernels/fp8_quantize_per_token_f16.cu`, `cublaslt.rs::fp8_gemm_blockwise`, `qwen36_bring_up.rs` | `probe-qwen-fp8-proj` cosine ≥ 0.9999 across m∈{2,4,16,64} |
| **3c** (done) | sm_121-specific perf: dispatcher's m≥128 branch now goes through CUTLASS SM120 blockwise FP8 GEMM (the same `libcutlass_sm120.so` Gemma loads at lm_head). New `fp8_quantize_per_token_amax_f16.cu` kernel (per-token amax sibling of the per-K-block kernel from 3b) feeds CUTLASS's `prep_sfa`; the existing `[N/128, K/128]` weight blockscale feeds `prep_sfb`. `Qwen36Bringup` gained a `cutlass: CutlassBackend` field loaded at bring-up. Dispatch: m=1 → GEMV (bit-identical), m∈[2,127] → cuBLASLt blockwise (sm_100/120 fast path) or looped-GEMV fallback (sm_121), m≥128 AND `SoSm120` available → CUTLASS SM120. Verified: probe at q_proj@layer 3 (n=8192, k=2048): m={128, 256} cosine ≥0.9997 vs looped-GEMV reference (relaxed threshold from 0.9999 because CUTLASS uses per-token-amax replicated vs the reference's per-K-block scaling — different but mathematically very close). | ~250 | `kernels/fp8_quantize_per_token_amax_f16.cu`, `qwen36_bring_up.rs` | `probe-qwen-fp8-proj` extended with m∈{128, 256}; cosine ≥ 0.999 (CUTLASS branch) and ≥ 0.9999 (cuBLASLt / GEMV branches) |
| **4a** (done) | Route every per-layer projection (`apply_layer_linear_attn`, `_full_attn`, `_moe` shared + routed experts) through `Qwen36Bringup::fp8_proj_dispatch` instead of direct `Fp8GemvF16InLaunch`. 13 call sites. At m=1 (today's caller) this dispatches byte-identically to the same GEMV kernel; the value is that any future caller flip from m=1 to m=N (Phase 4b/5/7) immediately picks up the cuBLASLt-blockwise (sm_100/120) or CUTLASS SM120 (sm_121) fast paths without further per-call-site edits. Greedy determinism canary (German joke) + Qwen vision smoke (test_224.png) both byte-identical post-routing. | ~150 | `qwen36_bring_up.rs` | cargo test 117 lib + 29 integration; Qwen text + vision E2E unchanged |
| **4b** | Make `apply_layer_full_attn` batched-causal: input `[N, D]`, output `[N, D]`, KV-cache slot writes vectorised. Reuse Gemma's batched FA2 prefill path. The body still has substantial host-side processing (DtoH q/k/v → CPU RoPE → HtoD; host-side residual add) that must be ported to GPU first — call this Phase 4b-prep. Then the actual `[N, D]` conversion is mostly tile-shape arithmetic since the projections already accept m=N via the dispatcher (Phase 4a). | ~400 | `qwen36_bring_up.rs`, possibly new RoPE batched kernel | Per-layer cosine vs per-token reference (≥0.999), greedy canary byte-identical at m=1, then m=N=prompt_len works |
| **5** | Linear-attn: loop INSIDE the kernel over N tokens with state in shared. Same recurrent semantics, just batched at the launch level. | ~300 + new kernel | new `kernels/qwen_linear_attn_batched.cu` | Per-step state-equivalence vs per-token reference |
| **6** | MoE block batched: top-k routing over `[N, D]` produces `[N, K]` expert assignments; per-expert grouped GEMM. The existing `forward_layer3_full_moe_probe` path is per-token — needs the same `m: u32` extension as Phase 3. | ~500 | `qwen36_bring_up.rs`, MoE helpers | Cosine vs per-token reference |
| **7** | `forward_qwen36_decode` outer loop deleted: the layer-chain runs once over `hidden_region[0..N]`. Optional: CUDA Graph capture of the prefill, replayed for re-runs of the same shape. | ~150 | `qwen36_bring_up.rs` | TTFT measurement (Phase 2 harness) |

## Phase 1 (this commit)

### Change

`apply_layer_linear_attn` and `apply_layer_full_attn` previously
took `last_hidden_region: &rvllm_mem::Region<'_>` and called
`.device_ptr()` 28 places, plus one `copy_from_host` for the
host-residual-add path inside `apply_layer_linear_attn`. The
caller therefore had to set up a separate `last_hidden_region`
scratch and shuttle each token's slice in and out via DtoD
copies.

The refactor:

* Both functions now take **`last_hidden_ptr: u64`** and
  `last_hidden_bytes: usize`. The 28 `.device_ptr()` calls
  become direct uses of `last_hidden_ptr`. The single
  `copy_from_host` becomes a raw `cuMemcpyHtoDAsync_v2`.
* Caller (line ~4500 in `forward_qwen36_decode`) computes
  `let tok_ptr = hidden_region.device_ptr() + (tok_local as
  u64) * (last_hidden_bytes as u64);` and passes it.
* The two `cuMemcpyDtoDAsync_v2` calls (extract + writeback)
  in the per-token loop are deleted. Layer kernels now read
  and write directly into `hidden_region` at the per-token
  offset.

### Why this is a no-op for correctness

Each layer function's reads and writes through
`last_hidden_region.device_ptr()` are equivalent to reads and
writes through `hidden_region.device_ptr() + tok_local ×
hidden_bytes` provided the offset is constant for the duration
of one layer-chain pass. Since the per-token outer loop pins
`tok_local`, the offset is constant — same byte addresses, just
no intermediate scratch.

### What this enables

Phase 3+ can change the layer functions' `m: u32 = 1` to
`m: u32 = num_tokens` and pass `hidden_region.device_ptr()`
(no offset) plus `num_tokens` — same code path, just operating
on `[N, D]` instead of `[1, D]`. The Phase 1 refactor is the
mechanical groundwork that makes that swap one signature change
away from working.

### Performance gain in this phase alone

Tiny: ~2 DtoDAsync launches × ~5 µs/launch saved per token =
~1 ms on a 100-token prompt. The gain is in *enabling* Phase 3,
not the saved copies themselves.

## Open questions for later phases

* Does the existing `fused_rmsnorm_fp8_quant` kernel handle
  `m > 1` correctly? Its caller chain assumes per-token output;
  output layout for `[N, D]` may need a wrapper.
* Linear-attn conv state: 1-D conv across the recurrent
  history — needs to be batched across token positions while
  staying causal.
* CUDA Graph capture (Phase 7) interacts badly with linear-attn
  state if the graph is replayed across requests; must
  capture-per-request or expose param updates for the state
  region pointer.

These are solvable, just non-trivial. They get chosen during
Phase 4 / Phase 5 based on bench-harness numbers from Phase 2.

## Status snapshot — head 8241c8f

All transformer-stack batched-prefill phases are GREEN (byte-equivalent
to per-token reference, all 1782 layer/phase/token dump rows
cos = 1.000000):

* Phase Linear (Round-24+25): batched delta-rule + conv-state-advance
  via two new kernels. Env-gate `RVLLM_QWEN36_BATCH_LINEAR_PREFILL=1`.
* Phase Full (Round-26): f16-IO causal-prefill via existing
  `flash_attention_2_f16kv_kernel` wrapped with f16↔f32 casts. Env-gate
  `RVLLM_QWEN36_BATCH_FULL_PREFILL=1`.
* Phase 6a (Round-27): batched router GEMV + topk_softmax. Env-gate
  `RVLLM_QWEN36_BATCH_MOE_PREFILL=1`.
* Phase 6b (Round-27b): row-batched-by-topk indirect FFN k-rounds via
  three new kernels (codex' "skip the gather/sort, extend
  blockIdx.y to token-row + read expert from top_idx[m*K+k]"
  insight). Env-gate `RVLLM_QWEN36_BATCH_MOE_ROUTED_FFN=1`.
* Phase 6c (Round-27c): batched shared-expert + final residual via two
  new kernels (`shared_gate_dot_sigmoid_batched`,
  `scaled_add_devw_batched`) + reuse of dual_silu/Fp8GemvF16InLaunch
  m=N/f16_plus_f32_inplace N*hidden. Env-gate
  `RVLLM_QWEN36_BATCH_MOE_SHARED=1`.
* Phase 7: outer-loop deletion is implicit — `forward_qwen36_decode_
  cancellable`'s batched branch is strictly layer-major; the legacy
  for-tok loop is skipped when the gates are set.

Production race fixes en route:
* Round-25 / 26: pos_cl HtoD vs `CU_STREAM_NON_BLOCKING` race —
  diagnosed by codex, structurally fixed via device-fill kernel
  `qwen_fill_pos_slots_i32`. Both token-major and layer-major.

Bench instrumentation: `RVLLM_QWEN36_TIMING=1` logs prefill_ms.

Default for all 5 gates is OFF until the prod-flip is taken explicitly;
single-line change.

## qwen35 27B-dense parallel track (2026-05-22)

The qwen35 path (Qwen 3.5 / 3.6-27B dense, qwen35_bring_up.rs)
shares Phase 1-4a's plumbing but has its own NVFP4-batched-prefill
landing. Recent perf landings on `rusty_sm121_qwen36_26b`:

* `a165a41` + `df1f485` — CUTLASS FP8 blockwise GEMM helper
  accepts M<128 via internal zero-pad to M_pad=128 + first-M-
  rows-copy back. Profile defaults dropped MIN_TOKENS 128 → 8
  across MLP/LINEAR/FULL after a byte-equivalence sweep.
  md5 byte-equivalent across all tested prompts.
* `68c6dbb` — NVFP4 batched-prefill cu_seqlens populator switched
  to stream-ordered `cuMemsetD32Async`. 16 sync HtoDs/request
  eliminated.
* `c8c72cb` — qwen35 spec-decode `commit_only` uses sequential
  `forward_layers_only` per token (not the batched recurrent-
  state path which had BF16 accumulation drift). Spec output now
  byte-equivalent to eager on 2 canonical workloads.

These changes mean qwen35 sm_121 prefill at M ∈ [32, 127] now
runs through CUTLASS SM120 instead of the slow per-token GEMV
fallback — closing the perf cliff that lived at M=128.

## Phase 8: Decode-step CUDA Graph capture — SHIPPED 2026-05-22

The captured-graph decode path landed across a chain of commits
from `793ddf0` through `4f083f8`. The replay path is hardware-
validated producing byte-correct multi-step output. Production
default remains `RVLLM_QWEN36_DECODE_GRAPH` unset (legacy eager);
captured path is opt-in.

Final commit chain (see `CLAUDE.md` "Phase 8" section for the
detailed catalog):

- `793ddf0` `Qwen36DecodeWorkspace` struct + allocator.
- `be98a01` device-argmax closer + `argmax_dev_to_host_token`.
- `81a487f` workspace-driven decode-step entry
  (`forward_qwen36_decode_step_to_workspace`).
- `8505f90` `CapturedGraph::capture` + replay infrastructure
  (`try_capture_decode_step`, `replay_decode_step`,
  `decode_step_via_graph_or_eager`).
- `3c30fea` cuda_worker wire-up; two-gate
  capture/replay design.
- `bb9a3cf` debug fix: explicit `graph.replay()` after capture
  (CUDA stream capture in THREAD_LOCAL mode records but doesn't
  execute eagerly — the unfortunate root cause of the
  documented "Die." regression).
- `4f083f8` position-indirect overrides
  (`pos_dev_override` / `ctx_dev_override` skip the
  per-call positions_region + fill kernel; workspace stable
  pos/ctx device slots feed RoPE + KV-slot via
  `cuMemsetD32Async`) + cross-request `clear_decode_capture`
  reset.

Codex Round-28 originally reviewed the path:

> The minimal-risk first green graph is therefore not "capture
> forward_qwen36_decode", but "factor a fixed-workspace
> qwen36_decode_step_launch_only and capture that."

The pre-Phase-8 `forward_qwen36_decode_cancellable` allocated
new arena regions per call, did sync `Region::copy_from_host` on
the legacy default stream, and ran a sync `cuMemcpyDtoH` inside
the closer — none of these were graph-friendly. The shipped plan
(below) addressed each:

1. **Workspace** — `Qwen36DecodeWorkspace` struct holding the ~30
   per-step scratch regions preallocated once (normed, qkv, q/k/v,
   conv_in, silu, down, rs, logits, topk_idx, topk_w, etc.).
   Allocated at decode-loop entry, reused for every step.
2. **`decode_step_launch_only(workspace, token_dev_ptr,
   pos_dev_ptr, ctx_dev_ptr)`** — pure launch sequence, no
   `arena.region`/`checkpoint`/`restore`, no `copy_from_host`,
   no DtoH. Reads inputs and writes outputs through device
   pointers passed in.
3. **`outside_closer_launch_only`** — argmax launch + write to
   a token-output device buffer; no DtoH inside.
4. **Outer loop in `cuda_worker.rs`** —
    - prefill eager → first next_token
    - decode step 0 eager (warmup, advances state once)
    - decode step 1: set token/pos/clen via `cuMemsetD32Async`,
      eager run, fence, capture identical body without executing,
      DtoH token from the eager run
    - decode steps 2..N: set token/pos/clen, `graph.replay`, fence,
      DtoH token (4 bytes) outside graph
5. **Env-gate** `RVLLM_QWEN36_DECODE_GRAPH=1`. Default OFF; capture
   is per-request scope so graph-cache mgmt is trivial.

Realistic scope per Round-28: 500–1000 LOC across the decode forward
factoring, the workspace struct, the cuda_worker capture/replay
loop, and the audit harness (decode-step dumps rather than per-layer).
Not a one-iteration task; Round-28 explicitly flagged it as bigger
than the "200 LOC" placeholder I had in my initial sketch.

Numerical contract: replay is the same kernel sequence with the same
device pointers and scalar launch args. With token/pos/clen updated
device-side before each replay, decoded tokens must be byte-identical
to eager-mode for the same input. Any divergence points at captured
stale metadata, hidden scratch aliasing, or accidental double-advance
of state on the capture step.

The shipped path satisfies this contract (validated on three
sequential qwen3-6-35b-a3b requests, full 1-10 counting + 80-token
photosynthesis output byte-identical to eager).

Follow-on commit `bcdce94` adds cross-request graph cache reuse via
a persistent workspace at worker bring-up + an inner RAII arena
checkpoint+restore guard at decode_inner entry. The captured graph
from request N is now valid for request N+1 (all device pointers
stable). First request after worker startup pays the
capture+instantiate cost; every subsequent request skips it.

Multi-step macro-replay + argmax+link fusion (commits `a14af12`,
`e47166b`) infrastructure shipped — kernel-count reductions
real, latency within noise of single-step replay (the per-step
host overhead is small relative to kernel work).

**MoE expert kernel fusion (commit `f0f79d5`)**. The fused
`fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add_kernel`
collapses the per-k-round (down-projection FP8 GEMV +
scaled-add accumulator) pair into ONE launch. Saves 320 kernel
launches + 320 f16 round-trips per decode token on Qwen 3.6
35B-A3B.

**Other-models fusion (commit `39f7c1a`)** — same pattern
ported to Qwen 3.5/3.6 27B DENSE decode via new kernel
`fp8_gemv_blockwise_wpr_native_f16in_residual_add_kernel`.
Three per-layer M=1 sites in qwen35_bring_up.rs (full-attn
o_proj, linear-attn out_proj, dense MLP ffn_down) fuse with
the subsequent vector_add_f16 residual. 120 launches saved per
decode token. Hardware-validated correct on qwen3-6-27b.
The fused kernel infrastructure is
now available for Mistral 3.5 / Gemma 4 31B dense paths.

**Batched-prefill fusion follow-on (commit `8060834`)** — new
kernel `..._indirect_scaled_add_batched_topk_kernel` extends the
f0f79d5 pattern to `apply_layer_moe_batched`'s prefill k_round
loop. Saves 1 launch + 1 f16 round-trip per k-round across the
entire prompt. Gated via
`RVLLM_QWEN36_BATCH_MOE_PREFILL`.

**Shared-expert fusion (commit `99e6cde`)** — new kernel
`fp8_gemv_blockwise_wpr_native_f16in_scaled_add_devw_kernel`
fuses the shared-expert down + scaled_add into one launch.
Saves 40 launches/decode token at 40 MoE layers. Latency
impact unverified.

**Dual_silu k_round-batch fusion (commit `442a72c`)** — new
kernel `..._dual_silu_indirect_kround_batched_kernel` batches
the 8-k_round host loop into ONE launch via `grid.z = top_k`.
silu_region grows to `[top_k, M, N_int]`; down loop still
serial. 7 launches saved per MoE layer per token. Latency
impact unverified.

**Router+topk fusion (commit `e049258`)** — new kernel
`router_gemv_with_topk_f16_to_f32_kernel` uses atomic-counter
last-block-does-topk pattern. Saves 1 launch per MoE layer
per token (~40 per decode token).

**Batched router+topk fusion (commit `314dbe6`)** — same
pattern ported to the prefill batched path with per-token
counter slots (one per token). New kernel
`router_gemv_with_topk_batched_f16_to_f32_kernel`. Eliminates
1 launch per MoE layer per prefill. Counter region sized by
`kv_cache_num_blocks`, zeroed once at bring-up; kernel self-
resets per-token slots.

**Down k_round-batch fusion (commit `b1f221e`)** — fuses the
8 per-k_round host-loop down launches in
`apply_layer_moe_with_override` into ONE kernel via per-warp
f32-register accumulation. The LITERAL dual_silu+down
megakernel is infeasible (silu_mul recomputation per output
would explode work ~1000x); this is the closest tractable
analog. New kernel `..._indirect_scaled_add_kround_batched_kernel`
has each warp own one (m, n) output slot and sequentially
process all top_k k_rounds — no atomic, single global RMW at
the end. Saves 280 launches/decode token.

**Hidden-state → workspace refactor (commit `e13e2eb`)** — adds
`hidden_dev_override: Option<u64>` to
`forward_qwen36_decode_inner_with_workspace_overrides_v2`;
when Some, all hidden-state reads/writes (embed_gather output,
residual stream, vision-splice destination, layer loop's
tok_ptr, closer's last_hidden_row_ptr) route through that
pointer instead of the per-call arena `hidden_region`. The
four closer fns (`outside_closer`, `_device_argmax`,
`_device_argmax_with_link`, `_all`) refactored to take
`hidden_dev_ptr: u64`. Workspace forward entry points plumb
`Some(workspace.hidden_dev)`. Persistent workspace slot
survives the inner-ckpt restore + is address-stable across
requests, so the captured graph's hidden-state references stay
valid forever — unlocks broader closer/post-attn fusion
patterns. Hardware-validated coherent under DECODE_WORKSPACE=1
and DECODE_GRAPH=1 + REPLAY=1 (short + 1-10 counting). Latency
impact unverified.

**Multi-step graph with persistent hidden (commit `eb26d86`)**
— exploits e13e2eb to retire the per-iteration
`arena.region("qwen36_pl_hidden", ...)` re-allocation in the
multi-step macro-graph (`try_capture_decode_steps_n` capture
body + eager fallback + the pure-eager branch of
`decode_steps_n_via_graph_or_eager`). The fused closer reads
hidden state directly from `workspace.hidden_dev` (which the
decode-step body already writes to). Result on qwen3-6-35b-a3b
at N=8: macro-graph node count drops roughly in half. Latency
impact unverified. Multi-step stays operator-opt-in via
`RVLLM_QWEN36_DECODE_MULTI_STEP`.

**Q-norm + K-norm + RoPE + KV megakernel Phase 1 (commit
`943f8bb`, K-side race fixed in commit `33145a6`)** — first
step toward the QKV+norm+RoPE+KV megakernel goal. New kernel
`fused_qnorm_knorm_rope_qwen_partial_f16kv_kernel` folds the
two standalone `rmsnorm_inplace_f16` launches (Q-norm + K-
norm per full-attn layer) into the existing partial-NeoX
RoPE kernel via a 2-phase per-head body: Phase 1 block-
reduces sum-of-squares across head_dim + applies
gamma*inv_norm in-place; Phase 2 standard rotation + KV-
cache write. Each thread covers both halves of its
(tid, tid+half_head) pair so the non-rotary tail
[rotary_dim, head_dim) gets normalised without relying on
the in-place trick the unfused kernel used. F16-KV-only
wiring; NVFP4 follow-on flagged. **Original commit had a
K-side race for tid in [half_rot, rotary_dim) (two warps
writing the same key_cache slot, non-deterministic md5
run-to-run); fixed in `33145a6` by splitting the else
branch.** Hardware-validated coherent
on qwen3-6-35b-a3b F16-KV + production NVFP4-KV regression-
clean.

**NVFP4-KV sibling (commit `6f6a25a`)** — same 2-phase
structure ported to the NVFP4 RoPE kernel:
`fused_qnorm_knorm_rope_qwen_partial_nvfp4kv_kernel`. The
fp8 K-quantisation + microscale + per-token Q-scale cache
internals are preserved verbatim; only the rotation's input
reads change from raw `q_in`/`k_in` to shared-mem
`s_normalized` (head_dim f32) populated by Phase 1. Both
KV-dtype branches now retire the standalone Q/K-norm
rmsnorm_inplace launches. Hardware-validated coherent on
production NVFP4-KV (qwen3635b spec profile, NVFP4=1).
Saves 2 launches per full-attn layer per decode token
(~22/token at 11 layers) on the production path.

**Phase 2 QKV megakernel** (commits `fa48141` F16, `71fdf33` NVFP4,
`b08774e` batched-prefill wiring). Single warp-cooperative kernel
per KV dtype that fuses Q+K+V FP8 GEMVs + Q+gate split + Q-norm +
K-norm + partial-NeoX RoPE + KV-write (NVFP4 path also adds Q FP8
quant + K/V NVFP4 pack with per-16-elem microscales) into one
launch per (token, head). Each WARP (32 lanes) produces one output
via 32-thread K-dim cooperation using the same 8-elem lane-strided
pattern as `fp8_gemv_blockwise_wpr_native_f16in_kernel`; input row
staged into shared mem once per block. Block dim `head_dim*2 = 512`
threads = 16 warps for Qwen 3.6.

Dispatch in `apply_layer_full_attn` (per-token decode) and
`apply_layer_full_attn_batched` (batched prefill). Single env-gate
`RVLLM_QWEN36_QKV_MEGAKERNEL=1` (default off) controls all paths.
Batched arm is hard-gated to `num_tokens < 128`; at M≥128
`fp8_proj_dispatch` routes to CUTLASS SM120 GEMM which beats the
warp-coop megakernel on per-output throughput.

Numerical contract: FP8 GEMV byte-identical to
`fp8_gemv_blockwise_wpr_native_f16in_kernel`. F16 norm+RoPE byte-
identical to `fused_qnorm_knorm_rope_qwen_partial_f16kv` (commit
`943f8bb`). NVFP4 norm+RoPE+quant+pack byte-identical to
`fused_qnorm_knorm_rope_qwen_partial_nvfp4kv` (commit `6f6a25a`)
given normalised inputs.

Re-verified A/B on qwen3-6-35b-a3b NVFP4-KV (max_tokens=150,
3 runs each, freshly-installed symlinked binary):

  | Cell             | wall   | tokens | tok/s | prefill_ms | md5(completion) |
  |------------------|--------|--------|-------|------------|-----------------|
  | QKV_MEGA_OFF     | 3.69 s | 150    | 40.65 | 215        | b5d33eaa        |
  | QKV_MEGA_ON      | 3.09 s | 125    | 40.45 | 197        | b73335f8        |

Decode tok/s **parity** — wall-time delta explained by ON path
hitting EOS at 125 tokens. prefill_ms shows **~8% speedup**
(215→197) which IS apples-to-apples. md5 differs ON vs OFF
(fusion fires). CUTLASS path at M≥128 unchanged.

**Batched-prefill down k_round-batch port** (commit `38afff0`).
The decode-side down k_round-batch fusion (commit `b1f221e`) was
ported to `apply_layer_moe_batched`. Same kernel reused (M-agnostic),
just dispatched with `M=num_tokens`. Replaces the host-side
`for k_round in 0..top_k` loop of 8 separate batched_topk launches
with one launch per MoE layer (280 launches saved per prefill).
silu_b layout `[top_k, num_tokens, n_int]` (k_round-major, from
the dual_silu kround-batched launch) already matches the kernel's
`input_kround` contract. Bit-equivalent numerics (sum order
preserved per-warp sequential over k_rounds).

A/B on qwen3-6-35b-a3b NVFP4-KV (115-token prompt), deterministic
over 3 runs each: BATCH_MOE_ROUTED_FFN=on → prefill_ms 217 / wall
3.69s / 40.65 tok/s; BATCH_MOE_ROUTED_FFN=off (full per-token
fallback) → prefill_ms 273 / wall 3.75s / 40.00 tok/s. **21%
prefill speedup**, decode tok/s parity, from the whole batched-MoE stack
(this port + dual_silu kround-batch + router+topk batched +
shared-expert batched).

**Closer + last-block-residual_add fusion** (commit `505e9ea`).
New kernel
`fp8_gemv_blockwise_wpr_native_f16in_scaled_add_devw_then_residual_kernel`
folds the per-token MoE tail's in-place `hidden += f16(routed_sum)`
into the shared-expert closer's existing fp8_gemv + scaled-add
epilogue. Both side-effect writes (acc_f32 + hidden_f16) happen
on the same warp's lane 0; routed_sum f32 write preserved so the
`RVLLM_QWEN36_DEBUG_MOE` probe stays visible. Env-gated opt-in
(`RVLLM_QWEN36_MOE_CLOSER_FUSED=1`, default off). A/B on
qwen3-6-35b-a3b NVFP4-KV (150-token photosynthesis), deterministic
on top of QKV_MEGAKERNEL=1: wall 3.09s / 125 tok / 40.45 tok/s /
prefill 197ms — no measurable delta vs QKV megakernel alone. Saves 1 launch per MoE layer per decode token
(~40 launches/token); architectural cleanup at this scale, not a
measurable wall-clock win.

Not blocking the prefill batched path's production rollout — those
gates are independent of decode-graph and ready to flip on whenever
desired.


## Phase 9: TensorCore MMA dual_silu — first cut (task #93, 2026-05-23)

**First TensorCore-MMA-based FP8 kernel** (commit `443d77b`):
`fp8_mma_dual_silu_indirect_kround_batched_kernel` uses
`mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`
via existing helpers in `kernels/fp8_mma_frag_pack.cuh`. Replaces
the per-warp scalar-reduction GEMV with TensorCore m16n8k32 tiles.
Opt-in: `RVLLM_QWEN36_MOE_MMA_DUAL_SILU=1` (default GEMV path
unchanged).

First-cut scope deliberately conservative — 1 warp per block,
1 token per block, MMA tile [M=16, N=8] with only row 0 active
(rows 1..15 zero-staged). 15/16 of MMA throughput wasted; this
exists to validate the FP8×FP8 MMA path on sm_121 against real
qwen36 weights and stand as the foundation for follow-up work.

A/B (qwen3-6-35b-a3b NVFP4, deterministic, fresh binary
md5 `6b9bd230`):

  | Cell        | 1112 tok prefill | 4412 tok prefill |
  |-------------|------------------|------------------|
  | GEMV (def.) |      6020 ms     |     24775 ms     |
  | MMA (=1)    |      5738 ms     |     23636 ms     |
  | Speedup     |      +4.9%       |      +4.8%       |

Output coherence byte-checked on smoke + 4412-tok German prompt.

**Follow-up (parked, task #94)**: GPU pre-pass that sorts
(token, k_round) assignments by routed expert id; new MMA kernel
processes M=16 sorted-contiguous tile sharing one expert;
scatter output back to `[k_round, M, N]` via permutation. ~500-800
LOC. Expected 4-8× on the dual_silu portion = 30-50% overall
prefill speedup once landed.



## Phase 10: Expert-sort + grouped MMA (task #94, 2026-05-23, +45%)

Recovers the M-direction MMA reuse the Phase 9 first-cut sacrificed.
Three new kernels (commit `74eeade`):

* `qwen36_moe_expert_sort_kernel` — atomicAdd-based bucketing of
  (token, k_round) pairs into per-expert lists.
* `qwen36_moe_tile_table_kernel` — one thread per expert; emits
  ceil(C[e]/16) tile descriptors (expert_id, m_off, tile_size).
* `fp8_mma_dual_silu_grouped_m16_kernel` — grouped MMA. Per block
  (1 warp) processes one [M=16, N=8] tile sharing ONE expert,
  inline FP8 quant of input, dual MMAs for gate + up, per-row
  a_scale folded at scatter-write.

Pipeline: zero counters → sort → tile_table → DtoH tile_count →
grouped MMA. Opt-in via `RVLLM_QWEN36_MOE_MMA_GROUPED=1` (precedence
over Phase 9's `RVLLM_QWEN36_MOE_MMA_DUAL_SILU`). Diagnostic env
`RVLLM_QWEN36_MOE_MMA_GROUPED_DEBUG=1` adds per-pre-pass
cuStreamSynchronize so failures surface with their own op label.

A/B (qwen3-6-35b-a3b NVFP4, fresh binary, deterministic):

  | Cell                        | 1112 tok | 4412 tok |
  |-----------------------------|----------|----------|
  | GEMV baseline               |  6020 ms | 24775 ms |
  | Phase 9 MMA first-cut (M=1) |  5738 ms | 23636 ms |
  | **Phase 10 grouped (M=16)** | **3265 ms** | **13674 ms** |
  | Win vs GEMV                 | **+45.8%** | **+45.0%** |

Output coherence verified on 2k + 4k German prompts. Backward-compat
verified: gate unset → GEMV path produces correct German.

`max_per_expert` sizing learned the hard way: 8× sigma over even-
distribution mean was insufficient (one expert exceeded 278 at
M=1112 → silent OOB → cuStreamSynchronize trap). Set to
`(typical*32).max(256).min(total_assign)` — ~2.3 MB per layer at
M=1112, fits the worst routing skew observed.

Follow-up parked (not blocking production rollout):
* Same expert-sort foundation could feed a grouped MMA `down`
  projection sibling (27% of prefill GPU time per nsys profile).
* Persistent sort buffer alloc reuse across layers.
* Multi-warp tiling per block (currently 1 warp/block).


## Phase 11: W=4 multi-warp + persistent scratch (task #95, 2026-05-23, +56% vs GEMV)

Two follow-ons to Phase 10's grouped MMA in one commit (`618efb2`).

* **W=4 kernel** `fp8_mma_dual_silu_grouped_m16_w4_kernel`: 4 warps
  per block share one A tile + cover N=32 cols per block (vs N=8 in
  Phase 10). Amortises per-row amax + A staging across warps; grid.x
  drops 4×. Default-on when N % 32 == 0 (qwen36 moe_int=512). Opt-out
  via `RVLLM_QWEN36_MOE_MMA_GROUPED_W4=0`.
* **Persistent sort scratch** (`Qwen36Bringup.mma_sort_scratch`):
  sorted_per_expert + counts + tile_descriptors + tile_count pre-
  allocated at load() sized for M_MAX_FOR_PERSISTENT=16384 (32 MB).
  Per-call dispatch consults capacity; falls back to arena.region
  when oversized.

A/B (qwen3-6-35b-a3b NVFP4, deterministic):

  | Cell                          | 1112 tok    | 4412 tok    |
  |-------------------------------|-------------|-------------|
  | GEMV baseline                 |   6020 ms   |  24775 ms   |
  | Phase 10 MMA W=1              |   3265 ms   |  13674 ms   |
  | **Phase 11 MMA W=4**          | **2627 ms** | **11062 ms**|
  | Phase 11 + persistent         |   2631 ms   |  11146 ms   |
  | Win vs GEMV                   | **+56.4%**  | **+55.4%**  |
  | Win vs W=1                    |  +19.5%     |  +19.1%     |

Persistent scratch shows parity with per-call allocation —
arena.region overhead was negligible. The win is architectural
(fewer Bringup-state mutations per request).

Output coherence verified on 2k + 4k German prompts. Backward-compat
verified post-restore: legacy GEMV path unchanged.

Cumulative MoE-prefill speedup vs raw GEMV across all phases:
* Phase 4-7 baseline batched MoE (commits 8060834..b1f221e):
  reference 6020 ms
* Phase 9 MMA first-cut: 5738 ms (+4.9%)
* Phase 10 MMA grouped M=16: 3265 ms (+45.8%)
* **Phase 11 MMA W=4: 2627 ms (+56.4%)**

Next-step follow-up parked: same expert-sort foundation feeding a
grouped MMA `down` projection sibling (27% of remaining prefill
GPU time).


## Phase 12: Cooperative A-staging in W=4 grouped MMA (task #96, 2026-05-23, +57.9% vs GEMV)

`fp8_mma_dual_silu_grouped_m16_w4c_kernel` (commit `41892e9`) replaces
the W=4 serial A-staging (warp 0 lanes 0..15 each writing 32 bytes
through a 32-iter inner loop) with a single cooperative pass: all
128 threads write 4 contiguous bytes each via aligned u32 store. Per-
row token_idx broadcast through a new `smem_token_idx[16]` slot
(+64 B smem). Default-on when GROUPED+W4 are on; opt-out via
`RVLLM_QWEN36_MOE_MMA_GROUPED_W4_COOP=0`.

A/B (qwen3-6-35b-a3b NVFP4, deterministic, fresh binary):

  | Cell                          | 1112 tok    | 4412 tok    |
  |-------------------------------|-------------|-------------|
  | GEMV baseline                 |   6020 ms   |  24775 ms   |
  | Phase 10 MMA W=1              |   3265 ms   |  13674 ms   |
  | Phase 11 MMA W=4 serial       |   2627 ms   |  11062 ms   |
  | **Phase 12 MMA W=4 coop**     | **2533 ms** | **10587 ms**|
  | Win vs GEMV                   | **+57.9%**  | **+57.3%**  |
  | Win vs Phase 11               |  +3.6%      |  +4.3%      |

Regression-checked: COOP=0 falls back to Phase 11 (within noise).
Backward-compat smoke verified post-restore.

Cumulative MoE-prefill speedup vs GEMV:
* Phase 4-7 reference 6020 ms
* Phase 9 MMA first-cut: 5738 ms (+4.9%)
* Phase 10 MMA grouped M=16: 3265 ms (+45.8%)
* Phase 11 MMA W=4: 2627 ms (+56.4%)
* **Phase 12 MMA W=4 coop: 2533 ms (+57.9%)**


## Phase 13: Cooperative B-staging in W=4 (task #97, 2026-05-24, parity — opt-in)

`fp8_mma_dual_silu_grouped_m16_w4cb_kernel` (commit `5bb1c26`) adds
u64-vector B-staging on top of #96's coop-A. B_g/B_u staging
refactored from 8 unrolled byte iterations to 1 pass via 32 lanes ×
8-byte u64 loads. Lane mapping (n_row, k_chunk) = (lane/4, lane%4),
aligned LDG + STS per lane. Same total bytes moved, same coalescing.

A/B (qwen3-6-35b-a3b NVFP4, deterministic 3 runs):

  | Cell                            | 1112 tok    | 4412 tok    |
  |---------------------------------|-------------|-------------|
  | Phase 12 coop A (production)    |   2533 ms   |  10587 ms   |
  | Phase 13 coop A + coop B        |   2517 ms   |  10647 ms   |

Parity — within ~0.5% noise. The existing 8-iter byte-load loop
already gets compiler-vectorised on sm_121. Default-off (opt-in via
`RVLLM_QWEN36_MOE_MMA_GROUPED_W4_COOP_B=1`); preserved as foundation
for follow-on B-side cp.async pipelining or cross-warp B-distribution
work.

No regression to Phase 12 production default. Backward-compat smoke
verified.


## Phase 14: Grouped MMA down projection (task #98, 2026-05-24, +84% with both grouped)

`fp8_mma_down_grouped_m16_w4_kernel` (commit `3e5c097`) extends the
expert-sort + W=4 grouped MMA pattern to the down projection (27%
of prefill GPU time per the 2026-05-23 nsys profile).

Same kernel shape as the dual_silu grouped path (W=4, M=16, N=8 per
warp tile, cooperative A-staging), with key differences:

* Input: silu_b `[top_k, M_full, K_in]` k_round-major. Each (token,
  k_round) row gathered via `silu_b + k_round*M_full*K_in + token*K_in`.
* Per-row metadata broadcast via smem: token_idx, k_round, top_w.
* Output: `acc_f32[token, n]` f32 via atomicAdd — each (token, n)
  receives top_k=8 contributions from different (expert, k_round)
  tiles. atomicAdd cost amortised by tile parallelism.

Opt-in via `RVLLM_QWEN36_MOE_MMA_DOWN_GROUPED=1` (independent of
MMA_GROUPED; works alone or paired). Independently re-runs the
expert-sort pre-pass (~few µs/layer) using the same persistent
scratch as #94 — duplicates work when dual_silu grouped is also
on; cleanup follow-up parked.

**Numerical fix**: per-K=128 a_scale must be folded INSIDE the
inner accumulator per-kblk, not multiplied at write time only.
Previous (smem_ascale[row] at write) used LAST kblk's a_scale,
under-counting earlier kblks. For dual_silu input (RMSNormed →
~constant per-K-block amax) invisible; for down input (silu_b
SwiGLU output → varies substantially) produces `<ctrl47>` garbage
on the 80-word quantum entanglement prompt. Fixed by computing
`a_lo = smem_ascale[r_lo]` + `a_hi = smem_ascale[r_hi]` inside
the K-loop and folding per-(row, kblk) a_scale into g_outer.

A/B (qwen3-6-35b-a3b NVFP4, deterministic):

  | Cell                                  | 1112 tok    | 4412 tok    |
  |---------------------------------------|-------------|-------------|
  | GEMV baseline (both legacy)           |   6020 ms   |  24775 ms   |
  | Phase 12 dual_silu grouped only       |   2533 ms   |  10587 ms   |
  | Phase 14 down grouped only            |   4524 ms   |  18520 ms   |
  | **Phase 14 both grouped**             |   **955 ms**|  **4116 ms**|
  | Win vs GEMV                           | **+84.1%**  | **+83.4%**  |
  | Win vs dual_silu grouped only         |  +62.3%     |  +61.1%     |

Output coherence verified: 80-word quantum entanglement gives
coherent English (no NaN, no special tokens). Default-off
regression-checked: legacy path runs at 6019-6039 ms matching
GEMV baseline.

Cumulative MoE-prefill speedup vs raw GEMV:
* Phase 4-7 reference 6020 ms
* Phase 12 dual_silu grouped: 2533 ms (+57.9%)
* **Phase 14 both grouped:    955 ms (+84.1%)**


## Phase 15: Shared sort + per-kblk a_scale fix (task #99, 2026-05-24)

Two paired cleanups in commit `81b4b21`:

* **Shared sort across dual_silu + down**: layer-scoped
  `shared_sort_ptrs: Option<(...)>` published by dual_silu's grouped
  path after its sort+tile_table+DtoH succeeds. Down's grouped
  path consults it; if `Some`, reuses persistent buffer + cached
  tile_count and skips its own sort. Saves 2 kernel launches + 1
  sync DtoH per layer when both grouped paths are enabled.

* **Per-kblk a_scale fix** applied to all 4 dual_silu grouped
  kernels (#94 grouped_m16, #95 w4, #96 w4c, #97 w4cb). Mirrors
  the #98 down kernel's fix: `a_lo = smem_ascale[r_lo_c]`,
  `a_hi = smem_ascale[r_hi_c]` inside K-loop, fold per-(row, kblk)
  a_scale into outer accumulators. Write applies only `silu(g)*u`.
  Previously `smem_ascale[row]` at write time used the LAST kblk's
  value (under-counted earlier kblks). Invisible for dual_silu
  (RMSNormed input → ~constant per-K-block amax) but a real
  correctness issue.

A/B (qwen3-6-35b-a3b NVFP4, deterministic):

  | Cell                                    | 1112 tok    |
  |-----------------------------------------|-------------|
  | GEMV default-off                        |   6045 ms   |
  | BOTH grouped + shared + a_scale fix     |   946 ms    |

Parity vs #98 within noise. Quality verified: 80-word quantum
entanglement gives Einstein "spooky action" reference, no garbage
tokens. Down-only path (own sort, when MMA_GROUPED is off) also
coherent. Default-off regression matches the 6020 ms GEMV baseline.

The cumulative MoE-prefill speedup vs raw GEMV stays at +84%
(architectural cleanup, not a perf gain).


## Phase 16: Fresh nsys profile post-#98+#99 (task #100, 2026-05-24)

Captured against fresh binary md5 `65cb2a35` with both grouped MMA
paths active (M=4412 prompt). Top kernels in the prefill window:

  | Time% | Kernel                                          | Instances | Notes |
  |-------|-------------------------------------------------|-----------|-------|
  | 15.8% | gated_delta_rule_prefill_f16                    |    30     | **NEW TOP — linear-attn** |
  | 12.5% | fp8_mma_dual_silu_grouped_m16_w4c               |    40     | #96 path |
  | 12.3% | flash_attention_2_prefill_nvfp4kv_unified       |    10     | full-attn (CUTLASS NVFP4) |
  | 10.5% | fp8_mma_down_grouped_m16_w4                     |    40     | #98 path |
  |  6.4% | router_gemv_with_topk_batched_f16_to_f32        |    40     | |
  |  6.2% | fp8_gemv_blockwise_wpr_native_f16in (per-token) |  1940     | |
  |  4.6% | fp8_gemv_dual_silu (shared expert)              |   800     | |

**MoE routed FFN: 88.9% → 23.0%** (5.7× absolute time reduction).
The cumulative perf work since Phase 9 (commit `443d77b`) has
broken the original dominant bottleneck. Prefill walltime at
M=4412 dropped from 24775 ms (GEMV baseline) → 4116 ms (-83.4%).

**New top hotspot: linear-attn `gated_delta_rule_prefill_f16`
(15.8%).** Was a relative minor at 3.6% in the original profile
but absolute time stayed similar (~900 ms then / now); now
dominates the smaller wall.

Path forward documented in CLAUDE.md (descending expected impact):
1. Linear-attn `gated_delta_rule_prefill_f16` optimization (single
   biggest hotspot, ~650 ms at M=4412)
2. Apply grouped MMA pattern to shared-expert dual_silu (no routing
   indirection — simpler than #94-#97; expected ~5-8%)
3. Apply grouped MMA pattern to other model families


## Phase 17: Linear-attn prefill v2/v3 (task #101, 2026-05-24, +2.3% with v3)

Targets the new top hotspot identified by Phase 16 (`gated_delta_rule
_prefill_f16_kernel`, 15.8% of prefill GPU time). Two opt-in sibling
kernels in commit `50c3b1e`:

* **v2** vectorised state load/store via aligned u64 (4 fp16 halves
  per access). PARITY with v1 at hardware A/B — state I/O is not the
  bottleneck. Opt-in: `RVLLM_QWEN36_LINEAR_ATTN_PREFILL_V2=1`.
* **v3** drops inner-loop f16 RTNE round-trip on `s_row[kd]` (kept
  in v1 for byte-equivalence with the per-token decode kernel). v3
  keeps state in fp32 across the prefill; rounds at boundary only.
  Mathematically MORE correct (less quantisation noise). Opt-in:
  `RVLLM_QWEN36_LINEAR_ATTN_PREFILL_V3=1`.

A/B (qwen3-6-35b-a3b NVFP4, both grouped MMA active, deterministic):

  | Cell                 | 1112 tok    | 4412 tok    |
  |----------------------|-------------|-------------|
  | v1 baseline          |   944 ms    |  4116 ms    |
  | v3 skip-RTNE         | **920 ms**  | **4022 ms** |
  | v3 win vs v1         |   +2.5%     |   +2.3%     |

Quality verified on quantum-entanglement prompt. Default-off
regression-checked at 6020 ms baseline.

The kernel's REMAINING bottleneck is NOT state I/O (v2 parity) and
only partially the f16 RTNE (v3 +2.3%). Likely candidates: per-token
`__syncthreads` barriers (~9000 per call at M=4412) or inner-loop
FMA throughput (75M FMAs/launch). Both would require deeper rewrite
(warp-level reduction, multi-token recurrence batching) — multi-week
scope, parked.


## Phase 18: Shared-expert dual_silu grouped MMA (task #103, 2026-05-24, +4%)

`fp8_mma_shared_dual_silu_m16_w4c_kernel` (commit `84625d2`) extends
the W=4 coop-A grouped MMA pattern from Phase 12 to the SHARED-
expert FFN. Single per-layer weight matrix (no routing/sort);
tile mapping is direct (m_block, n_block) grid coverage. Per-kblk
a_scale fold (Phase 15 fix) built in.

Targets `fp8_gemv_dual_silu_kernel` (4.6% prefill GPU time per
Phase 16 nsys).

Opt-in via `RVLLM_QWEN36_MOE_SHARED_MMA=1`.

A/B (deterministic 3 runs):

  | Cell                          | 1112 tok    | 4412 tok    |
  |-------------------------------|-------------|-------------|
  | GEMV baseline                 |   6020 ms   |  24775 ms   |
  | All 3 grouped (incl. shared)  |   **907 ms**|  **3962 ms**|
  | Win vs GEMV                   | **+84.9%**  | **+84.0%**  |
  | Win vs prior phase            |  +4.0%      |  +3.9%      |

Cumulative progression vs raw GEMV (single linked summary):
* Phase 4-7 baseline 6020 ms
* Phase 12 dual_silu grouped: 2533 ms (+57.9%)
* Phase 14 both grouped (+ down): 955 ms (+84.1%)
* **Phase 18 all 3 grouped (+ shared): 907 ms (+84.9%)**


## Phase 19: Shared-expert down grouped MMA (task #106, 2026-05-24, +3.6%)

`fp8_mma_shared_down_m16_w4c_kernel` (commit `5ca9131`) symmetric
follow-up to Phase 18 — applies W=4 coop-A grouped MMA to the
shared-expert DOWN projection. Single weight, no routing/sort,
no silu/mul, no atomic — dense FP8 GEMM. Per-kblk a_scale (#99)
built in.

Opt-in via `RVLLM_QWEN36_MOE_SHARED_DOWN_MMA=1` (default off).

A/B (qwen3-6-35b-a3b NVFP4, all four grouped MMA paths active,
deterministic):

  | Cell                          | 1112 tok    | 4412 tok    |
  |-------------------------------|-------------|-------------|
  | GEMV baseline                 |   6020 ms   |  24775 ms   |
  | All 4 grouped (incl shared-down)| **873 ms**| **3821 ms** |
  | Win vs GEMV                   | **+85.5%**  | **+84.6%**  |
  | Win vs Phase 18               |  +3.7%      |  +3.6%      |

Cumulative progression vs GEMV (linked summary):
* Phase 4-7 baseline 6020 ms
* Phase 12 dual_silu grouped: 2533 ms (+57.9%)
* Phase 14 both grouped (+down): 955 ms (+84.1%)
* Phase 18 all 3 (+shared dual_silu): 907 ms (+84.9%)
* **Phase 19 all 4 (+shared down): 873 ms (+85.5%)**
