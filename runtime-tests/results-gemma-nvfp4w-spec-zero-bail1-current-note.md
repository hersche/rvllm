# Gemma NVFP4W zero-accept bailout probe

Generated: 2026-05-19 19:55 CEST

## Scope

Focused Gemma NVFP4W spec-decode probe for:

- `G4N_SPEC_ZERO_ACCEPT_BAILOUT_ITERS=1`

All promoted settings were otherwise kept:

- `RVLLM_GEMMA4_NVFP4_MLP_MMA_V8=1`
- `RVLLM_PREFILL_CHUNK_SIZE=128`
- `RVLLM_GEMMA4_SPEC_K=5`
- `G4N_SPEC_ADAPTIVE_WINDOW_ITERS=3`
- `G4N_SPEC_ADAPTIVE_MIN_K=1`

The intent was to see whether falling back to native decode after the first
zero-accept speculative iteration reduces wasted verify work on low-acceptance
prompts.

## Result

The probe preserved smoke quality but regressed latency:

| Setting | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| promoted baseline | 10 | 0 | 29.735 |
| bailout=1 | 10 | 0 | 27.204 |

Notable regressions:

- `short_pangram`: 1199.7 ms promoted baseline vs 17664.1 ms with bailout=1
- `medium_translation`: 5545.2 ms promoted baseline vs 17872.3 ms with bailout=1
- `instruction`: 14192.1 ms promoted baseline vs 15289.8 ms with bailout=1

Small wins on `short_math`, `long_summary_800`, and `reasoning_chain` do not
offset the short/medium latency spikes.

## Decision

Do not promote `G4N_SPEC_ZERO_ACCEPT_BAILOUT_ITERS=1`. The external Gemma
NVFP4W spec profile was restored to omit the line and use the code default.

Raw evidence:

- `runtime-tests/results-gemma-nvfp4w-spec-zero-bail1-current.md`
- `runtime-tests/results-gemma-nvfp4w-spec-zero-bail1-current.jsonl`

Baseline:

- `runtime-tests/results-current-promoted-triad.jsonl`
