# Gemma NVFP4W fixed K7 post-MMA probe

Generated: 2026-05-19 20:49 CEST

## Scope

Focused Gemma NVFP4W spec-decode probe for:

- `RVLLM_GEMMA4_SPEC_K=7`
- `G4N_SPEC_ADAPTIVE_K=0`

The promoted profile before this probe used fixed K6:

- `RVLLM_GEMMA4_SPEC_K=6`
- `G4N_SPEC_ADAPTIVE_K=0`

All other promoted settings were kept:

- `RVLLM_GEMMA4_NVFP4_MLP_MMA_V8=1`
- `RVLLM_PREFILL_CHUNK_SIZE=128`
- `RVLLM_NVFP4_KV=1`

The intent was to test whether fixed-K6 was the local optimum or whether a
wider verifier window still improved the current post-MMA profile.

## Result

Fixed K7 passed all quality canaries and improved aggregate throughput:

| Setting | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| fixed K6 | 10 | 0 | 31.027 |
| fixed K7 | 10 | 0 | 31.652 |

Notable wins:

- `medium_code`: 7920.9 ms fixed K6 vs 7165.3 ms fixed K7
- `long_summary_300`: 5962.1 ms fixed K6 vs 5578.3 ms fixed K7
- `short_capital`: 1553.2 ms fixed K6 vs 1544.3 ms fixed K7

Regressions:

- `medium_explain`: 10004.4 ms fixed K6 vs 10134.1 ms fixed K7
- `medium_translation`: 3751.3 ms fixed K6 vs 3784.9 ms fixed K7
- `long_summary_800`: 25581.4 ms fixed K6 vs 25699.3 ms fixed K7
- `instruction`: 10927.6 ms fixed K6 vs 11046.4 ms fixed K7
- Small short-prompt drift on `short_math` and `short_pangram`.

The aggregate win and 10/10 quality justify promoting fixed K7 for the current
Gemma NVFP4W post-MMA profile.

## Decision

Promote `RVLLM_GEMMA4_SPEC_K=7` while keeping `G4N_SPEC_ADAPTIVE_K=0` in the
external Gemma NVFP4W spec profile.

Raw evidence:

- `runtime-tests/results-gemma-nvfp4w-spec-fixed-k7-post-mma-current.md`
- `runtime-tests/results-gemma-nvfp4w-spec-fixed-k7-post-mma-current.jsonl`

Baseline:

- `runtime-tests/results-gemma-nvfp4w-spec-fixed-k6-post-mma-current.jsonl`
