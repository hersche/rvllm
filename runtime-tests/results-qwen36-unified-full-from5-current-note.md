# Qwen36 unified full-attention from5 probe

Generated: 2026-05-19 19:59 CEST

## Scope

Focused Qwen36 NVFP4 spec profile probe for:

- `RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_FROM=5`

The previous promoted profile used `RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_FROM=6`.
All other promoted settings were kept:

- `RVLLM_QWEN36_SPEC_K=4`
- `RVLLM_QWEN36_SPEC_MIN_DRAFTS=4`
- `RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=1024`
- `RVLLM_QWEN36_SPEC_MIN_MAX_NEW_TOKENS=64`
- `RVLLM_QWEN36_SPEC_PREFLIGHT_MIN_FULL_DRAFTS=1`
- `RVLLM_QWEN36_REPETITION_GUARD_N=12`

The intent was to move one more full-attention layer onto the faster unified
NVFP4 prefill path while checking that the repeat canary output shape remains
guarded and quality-safe.

## Result

The probe passed all text and repeat canaries:

| Setting | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| promoted from6 baseline | 11 | 0 | 73.317 |
| from5 | 11 | 0 | 74.580 |

Notable wins:

- `long_summary_300`: 4879.7 ms baseline vs 4679.7 ms from5
- `long_summary_800`: 12291.5 ms baseline vs 11833.7 ms from5
- `qwen_repeat_160`: 15341.5 ms baseline vs 13916.2 ms from5

Short and medium prompts remained green and close to baseline. The repeat canary
still stopped at 14 completion tokens under `RVLLM_QWEN36_REPETITION_GUARD_N=12`
with the same `mu alpha beta gamma gamma...` output shape.

## Decision

Promote `RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_FROM=5` in the external Qwen36
profile. This keeps the first 5 full-attention layers on the byte-stable path
and uses unified NVFP4 prefill for the final 5 full-attention layers.

Raw evidence:

- `runtime-tests/results-qwen36-unified-full-from5-current.md`
- `runtime-tests/results-qwen36-unified-full-from5-current.jsonl`

Baseline:

- `runtime-tests/results-current-promoted-triad.jsonl`
