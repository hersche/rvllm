# Gemma NVFP4W post-MMA adaptive window4 probe

Generated: 2026-05-19 19:23 CEST

## Configuration

- Temporary external profile: `mobile-31b-nvfp4w-rvllm-spec-window4-postmma`
- Base profile: `mobile-31b-nvfp4w-rvllm-spec`
- Only tested change: `G4N_SPEC_ADAPTIVE_WINDOW_ITERS=4`
- Kept promoted settings: `RVLLM_GEMMA4_SPEC_K=5`, `G4N_SPEC_ADAPTIVE_MIN_K=1`, `RVLLM_GEMMA4_NVFP4_MLP_MMA_V8=1`, `RVLLM_PREFILL_CHUNK_SIZE=128`, NVFP4 KV on.

## Result

- Smoke result: 10 pass, 0 fail
- Mean combined throughput: 29.5 tok/s vs 29.8 tok/s on the promoted window3 post-MMA run.
- Most probes were close to window3, but the profile did not produce a net win:
  - `medium_code`: 9078.9 ms vs 6925.0 ms promoted window3
  - `medium_translation`: 5147.5 ms vs 5522.5 ms promoted window3
  - `long_summary_300`: 6590.8 ms vs 6538.3 ms promoted window3
  - `long_summary_800`: 25465.9 ms vs 25433.4 ms promoted window3
- Quality stayed green across the suite.

## Decision

Do not promote `G4N_SPEC_ADAPTIVE_WINDOW_ITERS=4`. It remains quality-safe, but
the current post-MMA-v8 profile is marginally faster overall with window3 and
avoids the medium-code regression.

Raw evidence:

- `runtime-tests/results-gemma-nvfp4w-spec-window4-post-mma-v8-current.md`
- `runtime-tests/results-gemma-nvfp4w-spec-window4-post-mma-v8-current.jsonl`
- Baseline comparison: `runtime-tests/results-gemma-nvfp4w-spec-post-mma-v8-current.md`
