# Qwen36 repetition guard10 from5 probe

Generated: 2026-05-19 20:18 CEST

## Scope

Focused Qwen36 from5 profile probe for:

- `RVLLM_QWEN36_REPETITION_GUARD_N=10`

The current promoted profile uses `RVLLM_QWEN36_REPETITION_GUARD_N=12`.
All other promoted settings were kept:

- `RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_FROM=5`
- `RVLLM_QWEN36_SPEC_K=4`
- `RVLLM_QWEN36_SPEC_MIN_DRAFTS=4`
- `RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=1024`
- `RVLLM_QWEN36_SPEC_MIN_MAX_NEW_TOKENS=64`
- `RVLLM_QWEN36_SPEC_PREFLIGHT_MIN_FULL_DRAFTS=1`

The intent was to test whether a tighter repetition guard saves repeat-path
latency without cutting normal outputs.

## Result

Guard10 passed all quality canaries but did not improve performance:

| Setting | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| promoted guard12 | 11 | 0 | 74.735 |
| guard10 | 11 | 0 | 74.698 |

Key deltas:

- `qwen_repeat_160`: 13.90s / 14 tokens with guard12 vs 13.92s / 12 tokens with guard10
- `medium_explain`: 2244.8 ms with guard12 vs 2297.8 ms with guard10
- Long summaries and reasoning remained effectively unchanged.

The tighter guard stops two tokens earlier on the repeat canary but does not
reduce wall time and gives less margin for legitimate repeated-token outputs.

## Decision

Do not promote `RVLLM_QWEN36_REPETITION_GUARD_N=10`. The external Qwen36
profile was restored to `RVLLM_QWEN36_REPETITION_GUARD_N=12`.

Raw evidence:

- `runtime-tests/results-qwen36-repetition-guard10-from5-current.md`
- `runtime-tests/results-qwen36-repetition-guard10-from5-current.jsonl`

Baseline:

- `runtime-tests/results-current-promoted-triad-from5.jsonl`
