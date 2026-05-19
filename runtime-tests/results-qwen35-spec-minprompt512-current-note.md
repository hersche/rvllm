# Qwen35 spec min-prompt 512 probe

Generated: 2026-05-19 20:08 CEST

## Scope

Focused Qwen35 prompt-lookup spec admission probe for:

- `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=512`

The promoted profile uses `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=1024`. All other
promoted settings were kept:

- `RVLLM_QWEN35_SPEC_K=6`
- `RVLLM_QWEN35_SPEC_MIN_DRAFTS=6`
- `RVLLM_QWEN35_SPEC_MIN_MAX_NEW_TOKENS=64`
- `RVLLM_QWEN35_SPEC_ZERO_ACCEPT_BAILOUT_ITERS=1`

The intent was to keep short and medium requests native while allowing the
1004-token `long_summary_800` prompt into speculative decode.

## Result

The probe passed all quality canaries but regressed throughput:

| Setting | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| promoted 1024 gate | 11 | 0 | 24.420 |
| minprompt512 | 11 | 0 | 23.129 |

Key deltas:

- `long_summary_300`: 8236.3 ms baseline vs 8152.3 ms minprompt512
- `long_summary_800`: 9973.9 ms baseline vs 11636.9 ms minprompt512
- `qwen_repeat_160`: 28202.6 ms baseline vs 28162.5 ms minprompt512

The small repeat and `long_summary_300` changes do not offset the
`long_summary_800` regression. This confirms the current 1024-token prompt gate
is still the better performance setting for Qwen35.

## Decision

Do not promote `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=512`. The external Qwen35
profile was restored to `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=1024`.

Raw evidence:

- `runtime-tests/results-qwen35-spec-minprompt512-current.md`
- `runtime-tests/results-qwen35-spec-minprompt512-current.jsonl`

Baseline:

- `runtime-tests/results-current-promoted-triad.jsonl`
