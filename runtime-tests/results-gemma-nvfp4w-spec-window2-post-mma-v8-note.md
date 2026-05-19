# Gemma4 31B NVFP4W Spec Window-2 Post-MMA-v8 Probe

Generated: 2026-05-19 18:53 CEST

Change under test: the external Gemma NVFP4W profile was temporarily changed
from `G4N_SPEC_ADAPTIVE_WINDOW_ITERS=3` to
`G4N_SPEC_ADAPTIVE_WINDOW_ITERS=2`, keeping:

- `RVLLM_GEMMA4_NVFP4_MLP_MMA_V8=1`
- `RVLLM_GEMMA4_SPEC_K=5`
- `G4N_SPEC_ADAPTIVE_MIN_K=1`

The profile was restored to `G4N_SPEC_ADAPTIVE_WINDOW_ITERS=3` after the run.

| Probe | Window 2 total ms | Window 3 post-MMA total ms | Decision |
|---|---:|---:|---|
| Full text smoke | 10/10 pass | 10/10 pass | quality ok |
| Mean combined tok/s | 29.3 | 29.8 | keep window 3 |
| long_summary_300 | 7026.6 | 6538.3 | keep window 3 |
| long_summary_800 | 25490.9 | 25433.4 | keep window 3 |
| reasoning_chain | 852.3 | 848.8 | keep window 3 |

Result: window 2 preserves quality but does not improve the promoted
post-MMA-v8 Gemma NVFP4W spec profile. Keep
`G4N_SPEC_ADAPTIVE_WINDOW_ITERS=3`.

Raw evidence:

- `runtime-tests/results-gemma-nvfp4w-spec-window2-post-mma-v8-current.md`
- `runtime-tests/results-gemma-nvfp4w-spec-window2-post-mma-v8-current.jsonl`
- `runtime-tests/results-gemma-nvfp4w-spec-post-mma-v8-current.md`
- `runtime-tests/results-gemma-nvfp4w-spec-post-mma-v8-current.jsonl`
