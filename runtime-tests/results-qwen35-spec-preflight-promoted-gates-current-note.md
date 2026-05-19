# Qwen35 preflight with promoted gates

Generated: 2026-05-19 19:50 CEST

## Scope

Focused Qwen35 probe for `RVLLM_QWEN35_SPEC_PREFLIGHT_MIN_FULL_DRAFTS=1`
while keeping the current promoted admission gates:

- `RVLLM_QWEN35_SPEC_K=6`
- `RVLLM_QWEN35_SPEC_MIN_DRAFTS=6`
- `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=1024`
- `RVLLM_QWEN35_SPEC_MIN_MAX_NEW_TOKENS=64`
- `RVLLM_QWEN35_SPEC_ZERO_ACCEPT_BAILOUT_ITERS=1`

The intent was to test whether the Qwen35 prompt-lookup preflight filter can
avoid low-value verification while preserving the repeat-heavy speedup.

## Result

The probe passed all text and repeat canaries:

| Setting | Pass | Fail | Mean decode tok/s | `qwen_repeat_160` |
|---|---:|---:|---:|---:|
| promoted baseline | 11 | 0 | 24.420 | 28.20s / 50.4 tok/s |
| preflight=1 | 11 | 0 | 24.419 | 28.63s / 49.6 tok/s |

Ordinary prompts remained native under the existing 1024-token prompt gate, and
the long summaries were effectively unchanged. The repeat-heavy canary regressed
slightly, so there is no performance case for promoting this line.

## Decision

Do not promote `RVLLM_QWEN35_SPEC_PREFLIGHT_MIN_FULL_DRAFTS=1` with the current
Qwen35 promoted gates. The external Qwen35 profile was restored to omit the line.

Raw evidence:

- `runtime-tests/results-qwen35-spec-preflight-promoted-gates-current.md`
- `runtime-tests/results-qwen35-spec-preflight-promoted-gates-current.jsonl`

Baseline:

- `runtime-tests/results-current-promoted-triad.jsonl`
