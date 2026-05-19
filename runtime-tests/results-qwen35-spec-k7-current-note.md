# Qwen35 speculative K7 probe

Generated: 2026-05-19 21:04:11 CEST

## Scope

Tested a temporary Qwen35 candidate profile copied from `mobile-qwen35-rvllm-nvfp4-spec.env` with:

- `RVLLM_QWEN35_SPEC_K=7`
- `RVLLM_QWEN35_SPEC_MIN_DRAFTS=7`

The promoted baseline remains K=6 / min drafts 6.

Command:

```bash
sudo /home/r00t/workspace/upstream/rvllm-serve/runtime-tests/run_smoke.py \
  --profiles mobile-qwen35-rvllm-nvfp4-spec-k7 \
  --skip-vision --skip-audio --include-repeat-probes \
  --text-max-tokens 80 --request-timeout 300 \
  --restore-profile mobile-qwen-rvllm-nvfp4-spec \
  --results runtime-tests/results-qwen35-spec-k7-current.md \
  --jsonl runtime-tests/results-qwen35-spec-k7-current.jsonl
```

## Result

| Profile | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| `mobile-qwen35-rvllm-nvfp4-spec-k7` | 11 | 0 | 24.398 |

## Comparison to promoted K6

Promoted K6 comparison source: `runtime-tests/results-current-promoted-triad-from5-gemma-fixedk7.jsonl`.

| Label | K6 tok/s | K7 tok/s | K7 total ms | Verdict |
|---|---:|---:|---:|---|
| `short_capital` | 6.477 | 6.437 | 4194.4 | slower |
| `short_math` | 6.518 | 6.510 | 3686.4 | flat |
| `short_pangram` | 6.657 | 6.655 | 3306.0 | flat |
| `medium_explain` | 6.251 | 6.249 | 16961.5 | flat |
| `medium_code` | 6.381 | 6.381 | 7052.4 | flat |
| `medium_translation` | 6.533 | 6.531 | 5512.4 | flat |
| `long_summary_300` | 61.836 | 62.788 | 8074.8 | faster |
| `long_summary_800` | 105.830 | 105.074 | 9973.9 | slower |
| `reasoning_chain` | 6.695 | 6.697 | 8959.9 | flat |
| `instruction` | 6.280 | 6.272 | 18175.9 | flat |
| `qwen_repeat_160` | 49.790 | 48.780 | 29110.1 | slower |

K7 passed all quality canaries, but it did not improve the aggregate and regressed the repeat-heavy probe. Reject K7 and keep Qwen35 at the promoted K6 / min drafts 6 setting.

After the run, `rvllm-serve` was active with `/home/r00t/.rvllm/profiles/mobile-qwen-rvllm-nvfp4-spec.env` restored and serving `qwen3-6-35b-a3b`.

Raw evidence:

- `runtime-tests/results-qwen35-spec-k7-current.md`
- `runtime-tests/results-qwen35-spec-k7-current.jsonl`
