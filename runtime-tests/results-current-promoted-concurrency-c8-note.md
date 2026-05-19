# Current promoted concurrency c8 probe

Generated: 2026-05-19 21:24:48 CEST

## Scope

Ran the promoted profile concurrency benchmark at `concurrency=8`, matching the current `RVLLM_QUEUE_DEPTH=8` setting in all three target profiles.

Command:

```bash
sudo python3 /home/r00t/workspace/upstream/rvllm-serve/runtime-tests/run_concurrency.py \
  --profiles mobile-qwen-rvllm-nvfp4-spec mobile-qwen35-rvllm-nvfp4-spec mobile-31b-nvfp4w-rvllm-spec \
  --concurrency 8 --requests 8 --max-tokens 32 \
  --request-timeout 300 \
  --restore-profile mobile-qwen-rvllm-nvfp4-spec \
  --results runtime-tests/results-current-promoted-concurrency-c8.md \
  --jsonl runtime-tests/results-current-promoted-concurrency-c8.jsonl
```

## Results

| Profile | c1 total tok/s | c4 total tok/s | c8 total tok/s | c8 pass | c8 verdict |
|---|---:|---:|---:|---:|---|
| `mobile-qwen-rvllm-nvfp4-spec` | 83.25 | 84.24 | 83.29 | 8/8 | flat |
| `mobile-qwen35-rvllm-nvfp4-spec` | 6.61 | 6.66 | 6.65 | 8/8 | flat |
| `mobile-31b-nvfp4w-rvllm-spec` | 24.51 | 27.29 | 23.72 | 8/8 | c4 better |

All c8 requests passed the semantic canaries. For this short-prompt concurrency mix, Qwen36 and Qwen35 remain throughput-flat as concurrency rises, while Gemma NVFP4W fixed K7 peaks at c4 and regresses at c8. This argues against treating full queue-depth saturation as a performance target for the current short-prompt workload.

After the run, `rvllm-serve` was active with `/home/r00t/.rvllm/profiles/mobile-qwen-rvllm-nvfp4-spec.env` restored and serving `qwen3-6-35b-a3b`.

Raw evidence:

- `runtime-tests/results-current-promoted-concurrency-c8.md`
- `runtime-tests/results-current-promoted-concurrency-c8.jsonl`
- c1/c4 comparison source: `runtime-tests/results-current-promoted-concurrency-c1-c4.jsonl`
