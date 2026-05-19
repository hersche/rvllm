# Current promoted long-prefill c4 8-request probe

Generated: 2026-05-19 21:49:36 CEST

## Scope

Re-ran the promoted long-prefill concurrency benchmark at `concurrency=4` with 8 requests per profile. This makes the c4 comparison fair against the existing c8 run, which also used 8 requests.

Command:

```bash
sudo python3 /home/r00t/workspace/upstream/rvllm-serve/runtime-tests/run_concurrency.py \
  --profiles mobile-qwen-rvllm-nvfp4-spec mobile-qwen35-rvllm-nvfp4-spec mobile-31b-nvfp4w-rvllm-spec \
  --prompt-set long \
  --concurrency 4 --requests 8 --max-tokens 48 \
  --request-timeout 300 \
  --restore-profile mobile-qwen-rvllm-nvfp4-spec \
  --results runtime-tests/results-current-promoted-concurrency-long-c4-8req.md \
  --jsonl runtime-tests/results-current-promoted-concurrency-long-c4-8req.jsonl
```

## Results

| Profile | c4 total tok/s | c8 total tok/s | c4 pass | c8 pass | Fair comparison |
|---|---:|---:|---:|---:|---|
| `mobile-qwen-rvllm-nvfp4-spec` | 95.76 | 95.81 | 8/8 | 8/8 | throughput flat |
| `mobile-qwen35-rvllm-nvfp4-spec` | 89.46 | 89.37 | 8/8 | 8/8 | throughput flat |
| `mobile-31b-nvfp4w-rvllm-spec` | 92.08 | 91.33 | 8/8 | 8/8 | c4 slightly faster |

All requests passed quality checks. With equal request counts, long-prefill c4 and c8 are throughput-equivalent for Qwen36 and Qwen35. Gemma is slightly faster at c4, but the gap is small. c4 consistently has lower average latency than c8:

- Qwen36: 25.69s at c4 vs 36.78s at c8.
- Qwen35: 28.13s at c4 vs 38.48s at c8.
- Gemma: 27.52s at c4 vs 35.81s at c8.

This corrects the earlier 4-request c4 sample: moderate concurrency still helps keep latency lower, but the larger-sample throughput gain over c8 is small or absent. No profile setting is promoted from this benchmark.

After the run, `rvllm-serve` was active with `/home/r00t/.rvllm/profiles/mobile-qwen-rvllm-nvfp4-spec.env` restored and serving `qwen3-6-35b-a3b`.

Raw evidence:

- `runtime-tests/results-current-promoted-concurrency-long-c4-8req.md`
- `runtime-tests/results-current-promoted-concurrency-long-c4-8req.jsonl`
- c8 comparison source: `runtime-tests/results-current-promoted-concurrency-long-c8.jsonl`
