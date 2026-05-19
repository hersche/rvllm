# Gemma NVFP4W fixed K6 post-MMA probe

Generated: 2026-05-19 20:36 CEST

## Scope

Focused Gemma NVFP4W spec-decode probe for:

- `RVLLM_GEMMA4_SPEC_K=6`
- `G4N_SPEC_ADAPTIVE_K=0`

The promoted profile before this probe used fixed K5:

- `RVLLM_GEMMA4_SPEC_K=5`
- `G4N_SPEC_ADAPTIVE_K=0`

All other promoted settings were kept:

- `RVLLM_GEMMA4_NVFP4_MLP_MMA_V8=1`
- `RVLLM_PREFILL_CHUNK_SIZE=128`
- `RVLLM_NVFP4_KV=1`

The intent was to test whether the newly promoted fixed-K policy wants a wider
verifier window than K5.

## Result

Fixed K6 passed all quality canaries and improved aggregate throughput:

| Setting | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| fixed K5 | 10 | 0 | 30.617 |
| fixed K6 | 10 | 0 | 31.027 |

Notable wins:

- `medium_translation`: 4096.2 ms fixed K5 vs 3751.3 ms fixed K6
- `long_summary_300`: 6299.8 ms fixed K5 vs 5962.1 ms fixed K6
- `instruction`: 11201.0 ms fixed K5 vs 10927.6 ms fixed K6

Regressions:

- `medium_explain`: 9874.5 ms fixed K5 vs 10004.4 ms fixed K6
- `medium_code`: 7809.1 ms fixed K5 vs 7920.9 ms fixed K6
- `long_summary_800`: 25318.2 ms fixed K5 vs 25581.4 ms fixed K6
- Small short-prompt drift on `short_math` and `short_pangram`.

The aggregate win and 10/10 quality justify promoting fixed K6 for the current
Gemma NVFP4W post-MMA profile.

## Decision

Promote `RVLLM_GEMMA4_SPEC_K=6` while keeping `G4N_SPEC_ADAPTIVE_K=0` in the
external Gemma NVFP4W spec profile.

Raw evidence:

- `runtime-tests/results-gemma-nvfp4w-spec-fixed-k6-post-mma-current.md`
- `runtime-tests/results-gemma-nvfp4w-spec-fixed-k6-post-mma-current.jsonl`

Baseline:

- `runtime-tests/results-gemma-nvfp4w-spec-fixed-k5-post-mma-current.jsonl`
