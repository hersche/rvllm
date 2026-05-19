# rvllm-serve runtime tests

Generated: 2026-05-19 22:58:54 CEST

## Summary

| Profile | Pass | Fail | Mean prefill tok/s | Mean decode tok/s |
|---|---:|---:|---:|---:|
| mobile-qwen-rvllm-nvfp4-spec | 1 | 0 | nan | 91.4 |
| mobile-qwen-rvllm-nvfp4-nospec-repeat | 1 | 0 | nan | 92.5 |

## Detailed records

| Profile | Kind | Label | Prompt (truncated) | Output (truncated) | prompt_tokens | completion_tokens | ttft ms | total ms | prefill tok/s | decode tok/s | ok |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|:-:|
| mobile-qwen-rvllm-nvfp4-spec | text | qwen_repeat_160 | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta ga | mu alpha beta gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma | 1260 | 14 | — | 13941.9 | — | 91.4 | ✅ |
| mobile-qwen-rvllm-nvfp4-nospec-repeat | text | qwen_repeat_160 | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta ga | mu alpha beta gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma | 1260 | 15 | — | 13781.4 | — | 92.5 | ✅ |
