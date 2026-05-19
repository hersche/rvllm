# rvllm-serve concurrent runtime benchmark

Generated: 2026-05-19 21:35:23 CEST

| Profile | Prompt set | Concurrency | Pass | Fail | Wall ms | total tok/s | completion tok/s | req/s | avg latency ms | p95 latency ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mobile-qwen-rvllm-nvfp4-spec | long | 1 | 4 | 0 | 32680.2 | 95.8 | 5.8 | 0.12 | 8169.5 | 11635.2 |
| mobile-qwen-rvllm-nvfp4-spec | long | 4 | 4 | 0 | 32635.8 | 95.9 | 5.8 | 0.12 | 16909.3 | 20989.7 |
| mobile-qwen35-rvllm-nvfp4-spec | long | 1 | 4 | 0 | 35147.5 | 88.5 | 4.8 | 0.11 | 8786.5 | 9391.4 |
| mobile-qwen35-rvllm-nvfp4-spec | long | 4 | 4 | 0 | 34236.3 | 90.8 | 5.0 | 0.12 | 20983.3 | 24843.3 |
| mobile-31b-nvfp4w-rvllm-spec | long | 1 | 4 | 0 | 35972.4 | 87.1 | 4.6 | 0.11 | 8992.7 | 9191.9 |
| mobile-31b-nvfp4w-rvllm-spec | long | 4 | 4 | 0 | 32095.9 | 97.6 | 5.2 | 0.12 | 17778.0 | 21847.0 |
