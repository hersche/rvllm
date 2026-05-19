# Current promoted concurrency c1/c4 probe

Generated: 2026-05-19 21:19:35 CEST

## Scope

Added and ran `runtime-tests/run_concurrency.py`, a profile-aware concurrent chat benchmark for the active `rvllm-serve` service. Unlike the sequential smoke harness, this probe sends multiple simultaneous `/v1/chat/completions` requests while preserving simple semantic quality checks.

Command:

```bash
sudo python3 /home/r00t/workspace/upstream/rvllm-serve/runtime-tests/run_concurrency.py \
  --profiles mobile-qwen-rvllm-nvfp4-spec mobile-qwen35-rvllm-nvfp4-spec mobile-31b-nvfp4w-rvllm-spec \
  --concurrency 1,4 --requests 4 --max-tokens 32 \
  --request-timeout 300 \
  --restore-profile mobile-qwen-rvllm-nvfp4-spec \
  --results runtime-tests/results-current-promoted-concurrency-c1-c4.md \
  --jsonl runtime-tests/results-current-promoted-concurrency-c1-c4.jsonl
```

## Results

| Profile | c1 pass | c1 total tok/s | c4 pass | c4 total tok/s | c4 delta |
|---|---:|---:|---:|---:|---:|
| `mobile-qwen-rvllm-nvfp4-spec` | 4/4 | 83.2 | 4/4 | 84.2 | +1.2% |
| `mobile-qwen35-rvllm-nvfp4-spec` | 4/4 | 6.6 | 4/4 | 6.7 | +0.8% |
| `mobile-31b-nvfp4w-rvllm-spec` | 4/4 | 24.5 | 4/4 | 27.3 | +11.3% |

All profiles passed the concurrent quality canaries at both concurrency levels.

Interpretation:

- Qwen36 and Qwen35 are effectively flat on this short-prompt c1/c4 mix; concurrency increases per-request latency but does not hurt quality.
- Gemma NVFP4W fixed K7 shows a measurable short-prompt concurrency win, from 24.5 to 27.3 total tok/s.
- This establishes a reusable batching benchmark surface for follow-up queue-depth and prompt-mix probes. It does not by itself promote any profile changes.

After the run, `rvllm-serve` was active with `/home/r00t/.rvllm/profiles/mobile-qwen-rvllm-nvfp4-spec.env` restored and serving `qwen3-6-35b-a3b`.

Raw evidence:

- `runtime-tests/results-current-promoted-concurrency-c1-c4.md`
- `runtime-tests/results-current-promoted-concurrency-c1-c4.jsonl`
