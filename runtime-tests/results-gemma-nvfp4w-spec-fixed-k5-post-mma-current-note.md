# Gemma NVFP4W fixed K5 post-MMA probe

Generated: 2026-05-19 20:23 CEST

## Scope

Focused Gemma NVFP4W spec-decode probe for:

- `G4N_SPEC_ADAPTIVE_K=0`

This disables adaptive K and keeps verification at the promoted
`RVLLM_GEMMA4_SPEC_K=5`. All other promoted settings were kept:

- `RVLLM_GEMMA4_NVFP4_MLP_MMA_V8=1`
- `RVLLM_PREFILL_CHUNK_SIZE=128`
- `RVLLM_GEMMA4_SPEC_K=5`
- `RVLLM_NVFP4_KV=1`

The intent was to refresh the old fixed-K comparison against the current
post-MMA-v8 Gemma profile.

## Result

Fixed K5 passed all quality canaries and improved aggregate throughput:

| Setting | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| adaptive K5/window3/minK1 | 10 | 0 | 29.769 |
| fixed K5 | 10 | 0 | 30.617 |

Notable wins:

- `medium_explain`: 14903.2 ms adaptive vs 9874.5 ms fixed K5
- `medium_translation`: 5539.8 ms adaptive vs 4096.2 ms fixed K5
- `long_summary_300`: 6571.5 ms adaptive vs 6299.8 ms fixed K5
- `long_summary_800`: 25527.4 ms adaptive vs 25318.2 ms fixed K5
- `instruction`: 14188.5 ms adaptive vs 11201.0 ms fixed K5

Regressions were limited to small short-prompt drift and `medium_code`:

- `short_capital`: 1491.9 ms adaptive vs 1553.9 ms fixed K5
- `medium_code`: 6943.5 ms adaptive vs 7809.1 ms fixed K5

The aggregate and long/medium workload wins justify promoting fixed K5 for the
current Gemma NVFP4W post-MMA profile.

## Decision

Promote `G4N_SPEC_ADAPTIVE_K=0` in the external Gemma NVFP4W spec profile.

Raw evidence:

- `runtime-tests/results-gemma-nvfp4w-spec-fixed-k5-post-mma-current.md`
- `runtime-tests/results-gemma-nvfp4w-spec-fixed-k5-post-mma-current.jsonl`

Baseline:

- `runtime-tests/results-current-promoted-triad-from5.jsonl`
