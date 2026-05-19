# rvllm-serve runtime tests

Generated: 2026-05-19 23:02:55 CEST

## Summary

| Profile | Pass | Fail | Mean prefill tok/s | Mean decode tok/s |
|---|---:|---:|---:|---:|
| mobile-qwen35-rvllm-nvfp4-spec | 1 | 0 | nan | 47.6 |
| mobile-qwen35-rvllm-nvfp4-nospec-repeat | 1 | 0 | nan | 43.1 |

## Detailed records

| Profile | Kind | Label | Prompt (truncated) | Output (truncated) | prompt_tokens | completion_tokens | ttft ms | total ms | prefill tok/s | decode tok/s | ok |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|:-:|
| mobile-qwen35-rvllm-nvfp4-spec | text | qwen_repeat_160 | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta ga | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu | 1260 | 160 | — | 29817.1 | — | 47.6 | ✅ |
| mobile-qwen35-rvllm-nvfp4-nospec-repeat | text | qwen_repeat_160 | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta ga | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu | 1260 | 160 | — | 32941.0 | — | 43.1 | ✅ |
