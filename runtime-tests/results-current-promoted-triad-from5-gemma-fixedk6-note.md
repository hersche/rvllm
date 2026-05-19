# Current promoted triad smoke after Gemma fixed K6

Generated: 2026-05-19 20:44 CEST

## Scope

Fresh text-only smoke of the three target promoted profiles after promoting
Gemma `RVLLM_GEMMA4_SPEC_K=6` with `G4N_SPEC_ADAPTIVE_K=0`:

- `mobile-qwen-rvllm-nvfp4-spec`
- `mobile-qwen35-rvllm-nvfp4-spec`
- `mobile-31b-nvfp4w-rvllm-spec`

The run used `--include-repeat-probes`, so both Qwen profiles include the
repeat-heavy canary. Vision/audio were skipped because the active objective is
text-generation performance for these NVFP4 KV/spec/batching paths.

## Promoted Settings

Qwen36:

- `RVLLM_NVFP4_KV=1`, `RVLLM_BATCH_PREFILL=1`, `RVLLM_UNIFIED_PREFILL_MMA=1`
- `RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_FROM=5`
- `RVLLM_QWEN36_SPEC_K=4`, `RVLLM_QWEN36_SPEC_MIN_DRAFTS=4`
- `RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=1024`
- `RVLLM_QWEN36_SPEC_MIN_MAX_NEW_TOKENS=64`
- `RVLLM_QWEN36_REPETITION_GUARD_N=12`

Qwen35:

- `RVLLM_NVFP4_KV=1`, `RVLLM_BATCH_PREFILL=1`, `RVLLM_QWEN35_BATCHED_PREFILL=1`
- `RVLLM_QWEN35_SPEC_K=6`, `RVLLM_QWEN35_SPEC_MIN_DRAFTS=6`
- `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=1024`
- `RVLLM_QWEN35_SPEC_MIN_MAX_NEW_TOKENS=64`

Gemma NVFP4W:

- `RVLLM_NVFP4_KV=1`, `RVLLM_BATCH_PREFILL=1`, `RVLLM_UNIFIED_PREFILL_MMA=1`
- `RVLLM_GEMMA4_NVFP4_MLP_MMA_V8=1`
- `RVLLM_PREFILL_CHUNK_SIZE=128`
- `RVLLM_GEMMA4_SPEC_K=6`
- `G4N_SPEC_ADAPTIVE_K=0`

## Result

| Profile | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| Qwen36 NVFP4 spec from5 | 11 | 0 | 74.734 |
| Qwen35 NVFP4 spec | 11 | 0 | 24.546 |
| Gemma 31B NVFP4W fixed K6 | 10 | 0 | 31.121 |

Key canaries:

- Qwen36 `qwen_repeat_160`: 14 completion tokens, 13.96s, guard12 stable.
- Qwen35 `qwen_repeat_160`: 160 completion tokens, 28.14s, exact repeating pattern preserved.
- Gemma fixed K6 reproduced the current gains in the consolidated smoke:
  `medium_translation` 3.74s, `long_summary_300` 5.96s, `instruction` 10.89s.

Compared with the fixed-K5 triad, Gemma improved from 30.591 to 31.121 mean
decode tok/s while preserving 10/10 quality. Qwen36 and Qwen35 remain green and
within their promoted performance ranges.

Raw evidence:

- `runtime-tests/results-current-promoted-triad-from5-gemma-fixedk6.md`
- `runtime-tests/results-current-promoted-triad-from5-gemma-fixedk6.jsonl`
