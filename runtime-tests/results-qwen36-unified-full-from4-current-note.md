# Qwen36 unified full-attention from4 probe

Generated: 2026-05-19 20:02 CEST

## Scope

Focused Qwen36 NVFP4 spec profile probe for:

- `RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_FROM=4`

The current promoted profile uses `RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_FROM=5`.
All other promoted settings were kept:

- `RVLLM_QWEN36_SPEC_K=4`
- `RVLLM_QWEN36_SPEC_MIN_DRAFTS=4`
- `RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=1024`
- `RVLLM_QWEN36_SPEC_MIN_MAX_NEW_TOKENS=64`
- `RVLLM_QWEN36_SPEC_PREFLIGHT_MIN_FULL_DRAFTS=1`
- `RVLLM_QWEN36_REPETITION_GUARD_N=12`

The intent was to test whether moving one more full-attention layer to unified
NVFP4 prefill could improve long-prompt speed without breaking quality.

## Result

From4 was faster but failed the quality smoke:

| Setting | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| promoted from5 baseline | 11 | 0 | 74.580 |
| from4 | 10 | 1 | 76.399 |

Speed wins:

- `long_summary_300`: 4679.7 ms from5 vs 4487.3 ms from4
- `long_summary_800`: 11833.7 ms from5 vs 10491.5 ms from4

Quality regressions:

- `reasoning_chain` failed: expected `6 Äpfel`, from4 produced `3 Äpfel`.
- `qwen_repeat_160` changed output shape from the guarded
  `mu alpha beta gamma gamma...` 14-token stop to a 54-token refusal-style
  answer. The harness marks it as ok because it is not a strict semantic probe,
  but it confirms from4 changes Qwen36 behavior.

## Decision

Do not promote `RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_FROM=4`. Keep the
external Qwen36 profile at the promoted from5 cutoff.

Raw evidence:

- `runtime-tests/results-qwen36-unified-full-from4-current.md`
- `runtime-tests/results-qwen36-unified-full-from4-current.jsonl`

Baseline:

- `runtime-tests/results-qwen36-unified-full-from5-current.jsonl`
