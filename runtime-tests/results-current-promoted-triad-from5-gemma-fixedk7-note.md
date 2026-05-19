# Promoted triad after Gemma fixed K7

Generated: 2026-05-19 20:58:45 CEST

## Scope

Validated the current promoted runtime profiles after promoting Gemma NVFP4W speculative decode to fixed K=7:

- `mobile-qwen-rvllm-nvfp4-spec`: Qwen36 NVFP4 KV, unified batch full prefill from 5, spec K=4, guard N=12.
- `mobile-qwen35-rvllm-nvfp4-spec`: Qwen35 NVFP4 KV, spec K=6, min drafts 6.
- `mobile-31b-nvfp4w-rvllm-spec`: Gemma4 31B NVFP4 weights/KV, MLP MMA v8, fixed spec K=7, adaptive K disabled.

Command:

```bash
sudo /home/r00t/workspace/upstream/rvllm-serve/runtime-tests/run_smoke.py \
  --profiles mobile-qwen-rvllm-nvfp4-spec mobile-qwen35-rvllm-nvfp4-spec mobile-31b-nvfp4w-rvllm-spec \
  --skip-vision --skip-audio --include-repeat-probes \
  --text-max-tokens 80 --request-timeout 300 \
  --restore-profile mobile-qwen-rvllm-nvfp4-spec \
  --results runtime-tests/results-current-promoted-triad-from5-gemma-fixedk7.md \
  --jsonl runtime-tests/results-current-promoted-triad-from5-gemma-fixedk7.jsonl
```

## Results

| Profile | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| `mobile-qwen-rvllm-nvfp4-spec` | 11 | 0 | 74.565 |
| `mobile-qwen35-rvllm-nvfp4-spec` | 11 | 0 | 24.477 |
| `mobile-31b-nvfp4w-rvllm-spec` | 10 | 0 | 31.720 |

## Comparison to fixed K6 triad

| Profile | K6 mean decode tok/s | K7 mean decode tok/s | Delta |
|---|---:|---:|---:|
| `mobile-qwen-rvllm-nvfp4-spec` | 74.734 | 74.565 | -0.169 |
| `mobile-qwen35-rvllm-nvfp4-spec` | 24.546 | 24.477 | -0.069 |
| `mobile-31b-nvfp4w-rvllm-spec` | 31.121 | 31.720 | +0.599 |

Gemma fixed K7 improves the promoted Gemma triad aggregate over fixed K6 while preserving all quality canaries. Qwen36 and Qwen35 remain effectively flat relative to the previous triad, which is expected because their profiles did not change.

Key Gemma K7 canaries:

| Label | total ms | completion tokens | decode tok/s | ok |
|---|---:|---:|---:|:-:|
| `medium_explain` | 10137.2 | 68 | 9.4 | yes |
| `medium_code` | 7174.6 | 80 | 14.4 | yes |
| `medium_translation` | 3783.9 | 25 | 13.7 | yes |
| `long_summary_300` | 5574.6 | 43 | 93.8 | yes |
| `long_summary_800` | 25711.1 | 40 | 40.6 | yes |
| `reasoning_chain` | 872.5 | 5 | 72.2 | yes |
| `instruction` | 11029.4 | 80 | 10.4 | yes |

After the run, `rvllm-serve` was active with `/home/r00t/.rvllm/profiles/mobile-qwen-rvllm-nvfp4-spec.env` restored and serving `qwen3-6-35b-a3b`.

Raw evidence:

- `runtime-tests/results-current-promoted-triad-from5-gemma-fixedk7.md`
- `runtime-tests/results-current-promoted-triad-from5-gemma-fixedk7.jsonl`
