# rvllm-serve concurrent runtime benchmark

Generated: 2026-05-19 21:19:35 CEST

| Profile | Concurrency | Pass | Fail | Wall ms | total tok/s | completion tok/s | req/s | avg latency ms | p95 latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mobile-qwen-rvllm-nvfp4-spec | 1 | 4 | 0 | 1753.9 | 83.2 | 9.1 | 2.28 | 437.9 | 528.7 |
| mobile-qwen-rvllm-nvfp4-spec | 4 | 4 | 0 | 1733.1 | 84.2 | 9.2 | 2.31 | 1057.8 | 1442.4 |
| mobile-qwen35-rvllm-nvfp4-spec | 1 | 4 | 0 | 22102.6 | 6.6 | 0.7 | 0.18 | 5525.3 | 5513.5 |
| mobile-qwen35-rvllm-nvfp4-spec | 4 | 4 | 0 | 21937.6 | 6.7 | 0.7 | 0.18 | 11561.7 | 12970.6 |
| mobile-31b-nvfp4w-rvllm-spec | 1 | 4 | 0 | 6976.5 | 24.5 | 4.9 | 0.57 | 1743.7 | 1112.1 |
| mobile-31b-nvfp4w-rvllm-spec | 4 | 4 | 0 | 6265.1 | 27.3 | 5.4 | 0.64 | 2813.7 | 2512.5 |
