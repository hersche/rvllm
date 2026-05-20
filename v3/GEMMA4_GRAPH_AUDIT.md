# Gemma 4 31B NVFP4 — CUDA Graph audit (2026-05-20)

## Findings

The graph-replay infrastructure landed by commits `7737d35` +
`464e52d` is **dormant in production** for a structural reason, not
a profile-flip oversight.

`cuda_worker.rs:1219` dispatches `bringup.forward_full_to_token_captured`
ONLY inside the NON-spec branch (`spec_probe_this_request = false`).
The production profile `profiles/gb10/gemma4-31b-nvfp4` sets:

  RVLLM_GEMMA4_SPEC_DECODE=1   # K=8 spec, accept rate ~0.965

so every request enters the spec branch (`cuda_worker.rs:1076`) and
runs `run_spec_session_nvfp4_greedy_k_*`, which has its own decode
+ drafter loop. The captured-graph path is never reached.

## Decision

DO NOT add G4N_DECODE_GRAPH_REPLAY=1 / G4N_DECODE_GRAPH_INDIRECT=1
to the production profile. They have no effect under spec=on.

To actually use the captured-graph path on gemma4-nvfp4 we'd need
to either:
  (a) Drop spec decode (lose 1.5–2× steady-state, net negative).
  (b) Integrate graph replay INTO the spec session loop's per-iter
      verify/decode forward calls. Non-trivial — the spec loop
      makes K+1 base verify calls per iteration plus drafter
      forwards, each at a different position; capturing the whole
      thing as a single graph requires layout-stable K.

Option (b) is a multi-day refactor. Defer.

## What this means for the cross-family CUDA-Graph rollout

The mistral35 two-pass pattern (commit `6a9d43e` on
`perf/mistral35-cuda-graph`) is the only working capture wiring
today; mistral measured 0% gain at 88L 128B-param decode (compute-
bound). gemma4-31b-nvfp4 can't use it under spec. qwen36 35B-A3B
is the next candidate (MoE active params ~3B per token →
launch-overhead share is higher → CUDA-Graph win more likely).
