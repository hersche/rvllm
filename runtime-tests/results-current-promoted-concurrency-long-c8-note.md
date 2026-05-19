# Current promoted long-prefill concurrency c8 probe

Generated: 2026-05-19 21:42:37 CEST

## Scope

Ran the promoted long-prefill concurrency benchmark at `concurrency=8`, matching the current `RVLLM_QUEUE_DEPTH=8` in all three target profiles. This extends the long c1/c4 evidence from `runtime-tests/results-current-promoted-concurrency-long-c1-c4.jsonl`.

Command:

```bash
sudo python3 /home/r00t/workspace/upstream/rvllm-serve/runtime-tests/run_concurrency.py \
  --profiles mobile-qwen-rvllm-nvfp4-spec mobile-qwen35-rvllm-nvfp4-spec mobile-31b-nvfp4w-rvllm-spec \
  --prompt-set long \
  --concurrency 8 --requests 8 --max-tokens 48 \
  --request-timeout 300 \
  --restore-profile mobile-qwen-rvllm-nvfp4-spec \
  --results runtime-tests/results-current-promoted-concurrency-long-c8.md \
  --jsonl runtime-tests/results-current-promoted-concurrency-long-c8.jsonl
```

## Results

| Profile | c1 total tok/s | c4 total tok/s | c8 total tok/s | c8 pass | c8 verdict |
|---|---:|---:|---:|---:|---|
| `mobile-qwen-rvllm-nvfp4-spec` | 95.78 | 95.91 | 95.81 | 8/8 | flat |
| `mobile-qwen35-rvllm-nvfp4-spec` | 88.48 | 90.84 | 89.37 | 8/8 | c4 better |
| `mobile-31b-nvfp4w-rvllm-spec` | 87.12 | 97.64 | 91.33 | 8/8 | c4 better |

All c8 requests passed quality checks. Full queue-depth saturation is not the best long-prefill throughput point in this benchmark:

- Qwen36 remains throughput-flat from c1 through c8.
- Qwen35 keeps a small positive batching effect, but c4 is better than c8.
- Gemma NVFP4W fixed K7 keeps a positive c8 gain over c1, but c4 remains the best observed point.

No profile setting is promoted from this benchmark. The result argues for future scheduling/queue work that can benefit from moderate concurrent long-prefill batching without blindly filling the full queue for these prompt shapes.

After the run, `rvllm-serve` was active with `/home/r00t/.rvllm/profiles/mobile-qwen-rvllm-nvfp4-spec.env` restored and serving `qwen3-6-35b-a3b`.

Raw evidence:

- `runtime-tests/results-current-promoted-concurrency-long-c8.md`
- `runtime-tests/results-current-promoted-concurrency-long-c8.jsonl`
- c1/c4 comparison source: `runtime-tests/results-current-promoted-concurrency-long-c1-c4.jsonl`
