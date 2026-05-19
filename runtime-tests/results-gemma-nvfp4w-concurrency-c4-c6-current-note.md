# Gemma NVFP4W concurrency c4/c6 follow-up

Generated: 2026-05-19 21:27:31 CEST

## Scope

Ran a Gemma-only concurrency follow-up with 8 requests per level after the promoted c8 sweep. The goal was to check whether the earlier 4-request c4 win held with the same request count as c8 and whether c6 was a better middle point.

Command:

```bash
sudo python3 /home/r00t/workspace/upstream/rvllm-serve/runtime-tests/run_concurrency.py \
  --profiles mobile-31b-nvfp4w-rvllm-spec \
  --concurrency 4,6 --requests 8 --max-tokens 32 \
  --request-timeout 300 \
  --restore-profile mobile-qwen-rvllm-nvfp4-spec \
  --results runtime-tests/results-gemma-nvfp4w-concurrency-c4-c6-current.md \
  --jsonl runtime-tests/results-gemma-nvfp4w-concurrency-c4-c6-current.jsonl
```

## Results

| Profile | Concurrency | Requests | Pass | Fail | Total tok/s | Avg latency ms |
|---|---:|---:|---:|---:|---:|---:|
| `mobile-31b-nvfp4w-rvllm-spec` | 4 | 8 | 8 | 0 | 23.06 | 5461.9 |
| `mobile-31b-nvfp4w-rvllm-spec` | 6 | 8 | 8 | 0 | 21.10 | 7477.5 |

Comparison points:

- Earlier c4 with 4 requests: 27.29 total tok/s.
- Promoted c8 with 8 requests: 23.72 total tok/s.

All requests passed quality canaries. With equal 8-request runs, c4 and c8 are close and c6 is worse. The earlier c4 gain was sample-size sensitive, so no queue-depth or scheduling promotion follows from this short-prompt benchmark. Keep the current profile settings and use larger request counts for future concurrency decisions.

After the run, `rvllm-serve` was active with `/home/r00t/.rvllm/profiles/mobile-qwen-rvllm-nvfp4-spec.env` restored and serving `qwen3-6-35b-a3b`.

Raw evidence:

- `runtime-tests/results-gemma-nvfp4w-concurrency-c4-c6-current.md`
- `runtime-tests/results-gemma-nvfp4w-concurrency-c4-c6-current.jsonl`
