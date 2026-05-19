# Current promoted triad smoke after Gemma fixed K5

Generated: 2026-05-19 20:31 CEST

## Scope

Fresh text-only smoke of the three target promoted profiles after promoting
Gemma `G4N_SPEC_ADAPTIVE_K=0`:

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
- `RVLLM_GEMMA4_SPEC_K=5`
- `G4N_SPEC_ADAPTIVE_K=0`

## Result

| Profile | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| Qwen36 NVFP4 spec from5 | 11 | 0 | 74.496 |
| Qwen35 NVFP4 spec | 11 | 0 | 23.946 |
| Gemma 31B NVFP4W fixed K5 | 10 | 0 | 30.591 |

Key canaries:

- Qwen36 `qwen_repeat_160`: 14 completion tokens, 13.91s, guard12 stable.
- Qwen35 `qwen_repeat_160`: 160 completion tokens, 28.44s, exact repeating pattern preserved.
- Gemma fixed K5 reproduced the post-MMA gains in the consolidated smoke:
  `medium_explain` 9.88s, `medium_translation` 4.11s, `instruction` 11.20s.

Compared with the previous triad baseline, Gemma improved from 29.769 to 30.591
mean decode tok/s after the fixed-K5 promotion. Qwen36 remains stable; Qwen35
shows ordinary run-to-run variance but stays green.

Raw evidence:

- `runtime-tests/results-current-promoted-triad-from5-gemma-fixedk.md`
- `runtime-tests/results-current-promoted-triad-from5-gemma-fixedk.jsonl`
