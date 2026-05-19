# Current promoted long-prefill concurrency c1/c4 probe

Generated: 2026-05-19 21:35:23 CEST

## Scope

Extended `runtime-tests/run_concurrency.py` with `--prompt-set long`, using the same 300-word and 800-word summary prompt shapes as the sequential smoke harness. This measures batched long-prefill behavior rather than the short-canary queue behavior from the earlier c1/c4/c8 probes.

Command:

```bash
sudo python3 /home/r00t/workspace/upstream/rvllm-serve/runtime-tests/run_concurrency.py \
  --profiles mobile-qwen-rvllm-nvfp4-spec mobile-qwen35-rvllm-nvfp4-spec mobile-31b-nvfp4w-rvllm-spec \
  --prompt-set long \
  --concurrency 1,4 --requests 4 --max-tokens 48 \
  --request-timeout 300 \
  --restore-profile mobile-qwen-rvllm-nvfp4-spec \
  --results runtime-tests/results-current-promoted-concurrency-long-c1-c4.md \
  --jsonl runtime-tests/results-current-promoted-concurrency-long-c1-c4.jsonl
```

## Results

| Profile | c1 pass | c1 total tok/s | c4 pass | c4 total tok/s | c4 delta |
|---|---:|---:|---:|---:|---:|
| `mobile-qwen-rvllm-nvfp4-spec` | 4/4 | 95.78 | 4/4 | 95.91 | +0.1% |
| `mobile-qwen35-rvllm-nvfp4-spec` | 4/4 | 88.48 | 4/4 | 90.84 | +2.7% |
| `mobile-31b-nvfp4w-rvllm-spec` | 4/4 | 87.12 | 4/4 | 97.64 | +12.1% |

All long-prefill requests passed quality checks. The promoted Gemma NVFP4W fixed-K7 profile benefits materially from c4 long-prefill concurrency. Qwen35 shows a small positive c4 gain. Qwen36 remains throughput-flat on this prompt mix, so its current long-prefill path does not expose a useful c4 batching gain under this harness.

No profile setting is promoted from this benchmark. The result provides a targeted batching baseline for future scheduler/queue-depth work and shows that follow-up CUDA batching work is likely most valuable on Gemma long-prefill and then Qwen35.

After the run, `rvllm-serve` was active with `/home/r00t/.rvllm/profiles/mobile-qwen-rvllm-nvfp4-spec.env` restored and serving `qwen3-6-35b-a3b`.

Raw evidence:

- `runtime-tests/results-current-promoted-concurrency-long-c1-c4.md`
- `runtime-tests/results-current-promoted-concurrency-long-c1-c4.jsonl`
