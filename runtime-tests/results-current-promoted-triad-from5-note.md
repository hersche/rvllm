# Current promoted triad smoke after Qwen36 from5

Generated: 2026-05-19 20:15 CEST

## Scope

Fresh text-only smoke of the three target promoted profiles after promoting
Qwen36 `RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_FROM=5`:

- `mobile-qwen-rvllm-nvfp4-spec`
- `mobile-qwen35-rvllm-nvfp4-spec`
- `mobile-31b-nvfp4w-rvllm-spec`

The run used `--include-repeat-probes`, so both Qwen profiles include the
repeat-heavy canary. Vision/audio were skipped because the active objective is
text-generation performance for these NVFP4 KV/spec/batching paths.

## Promoted Settings

Qwen36:

- `RVLLM_NVFP4_KV=1`, `RVLLM_BATCH_PREFILL=1`, `RVLLM_UNIFIED_PREFILL_MMA=1`
- `RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_FROM=5`
- `RVLLM_QWEN36_SPEC_K=4`, `RVLLM_QWEN36_SPEC_MIN_DRAFTS=4`
- `RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=1024`
- `RVLLM_QWEN36_SPEC_MIN_MAX_NEW_TOKENS=64`
- `RVLLM_QWEN36_SPEC_PREFLIGHT_MIN_FULL_DRAFTS=1`
- `RVLLM_QWEN36_REPETITION_GUARD_N=12`

Qwen35:

- `RVLLM_NVFP4_KV=1`, `RVLLM_BATCH_PREFILL=1`, `RVLLM_QWEN35_BATCHED_PREFILL=1`
- `RVLLM_NVFP4_HADAMARD=1`, `RVLLM_NVFP4_HADAMARD_V=1`
- `RVLLM_QWEN35_SPEC_K=6`, `RVLLM_QWEN35_SPEC_MIN_DRAFTS=6`
- `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=1024`
- `RVLLM_QWEN35_SPEC_MIN_MAX_NEW_TOKENS=64`

Gemma NVFP4W:

- `RVLLM_NVFP4_KV=1`, `RVLLM_BATCH_PREFILL=1`, `RVLLM_UNIFIED_PREFILL_MMA=1`
- `RVLLM_GEMMA4_NVFP4_MLP_MMA_V8=1`
- `RVLLM_PREFILL_CHUNK_SIZE=128`
- `RVLLM_GEMMA4_SPEC_K=5`
- `G4N_SPEC_ADAPTIVE_WINDOW_ITERS=3`
- `G4N_SPEC_ADAPTIVE_MIN_K=1`

## Result

| Profile | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| Qwen36 NVFP4 spec from5 | 11 | 0 | 74.735 |
| Qwen35 NVFP4 spec | 11 | 0 | 24.752 |
| Gemma 31B NVFP4W spec | 10 | 0 | 29.769 |

Key canaries:

- Qwen36 `qwen_repeat_160`: 14 completion tokens, 13.90s, guard12 fired by output shape.
- Qwen35 `qwen_repeat_160`: 160 completion tokens, 28.12s, exact repeating pattern preserved.
- Gemma long-summary and reasoning probes remained green under the post-MMA-v8 K5/window3 profile.

Compared with the prior consolidated triad baseline, Qwen36 improved from
73.317 to 74.735 mean decode tok/s after the from5 promotion; Qwen35 and Gemma
remain in their promoted ranges.

Raw evidence:

- `runtime-tests/results-current-promoted-triad-from5.md`
- `runtime-tests/results-current-promoted-triad-from5.jsonl`
