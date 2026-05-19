# Current promoted triad smoke

Generated: 2026-05-19 19:44 CEST

## Scope

Fresh text-only smoke of the three target promoted profiles after the latest
external profile changes:

- `mobile-qwen-rvllm-nvfp4-spec`
- `mobile-qwen35-rvllm-nvfp4-spec`
- `mobile-31b-nvfp4w-rvllm-spec`

The run used `--include-repeat-probes`, so both Qwen profiles include the
repeat-heavy canary. Vision/audio were skipped because the active objective is
text-generation performance for these NVFP4 KV/spec/batching paths.

## Promoted Settings

Qwen36:

- `RVLLM_NVFP4_KV=1`, `RVLLM_BATCH_PREFILL=1`, `RVLLM_UNIFIED_PREFILL_MMA=1`
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

| Profile | Pass | Fail | Mean combined tok/s |
|---|---:|---:|---:|
| Qwen36 NVFP4 spec | 11 | 0 | 73.3 |
| Qwen35 NVFP4 spec | 11 | 0 | 24.4 |
| Gemma 31B NVFP4W spec | 10 | 0 | 29.7 |

Key canaries:

- Qwen36 `qwen_repeat_160`: 14 completion tokens, 15.34s, guard12 fired by output shape.
- Qwen35 `qwen_repeat_160`: 160 completion tokens, 28.20s, exact repeating pattern preserved.
- Gemma long-summary and reasoning probes remained green under the post-MMA-v8 K5/window3 profile.

Raw evidence:

- `runtime-tests/results-current-promoted-triad.md`
- `runtime-tests/results-current-promoted-triad.jsonl`
