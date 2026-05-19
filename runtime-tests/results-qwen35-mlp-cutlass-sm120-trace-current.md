# Qwen35 MLP CUTLASS SM120 trace

Generated: 2026-05-19 17:10 CEST

Profile: `mobile-qwen35-rvllm-nvfp4-spec` with:

- `RVLLM_QWEN35_PREFILL_PERF_TRACE=1`
- `RVLLM_QWEN35_MLP_PERF_TRACE=1`
- `RVLLM_QWEN35_MLP_CUTLASS_SM120=1`
- `RVLLM_QWEN35_MLP_CUTLASS_MIN_TOKENS=128`

Probe: 944 prompt tokens, `max_tokens=1`, temperature 0.

## Summary

| Stage | Total ms | Mean per layer ms | Share |
|---|---:|---:|---:|
| Full prefill | 42382.857 | - | 100.0% |
| Linear attention | 32204.152 | 670.920 over 48 linear layers | 76.0% |
| Full attention | 8897.477 | 556.092 over 16 full layers | 21.0% |
| Dense MLP | 1268.910 | 19.809 over 64 layers | 3.0% |
| Embed | 0.619 | - | 0.0% |
| Finalize | 10.918 | - | 0.0% |

## Dense MLP substages

| MLP substage | Total ms | Mean per layer ms |
|---|---:|---:|
| Residual copy | 93.402 | 1.459 |
| RMSNorm | 5.334 | 0.083 |
| Gate/up CUTLASS + SiLU | 831.548 | 12.993 |
| Down CUTLASS | 329.853 | 5.154 |
| Residual add | 7.605 | 0.119 |

## Comparison

Previous Qwen35 MLP trace baseline on the same 944-token probe:

- Full prefill: 141341.444 ms
- Dense MLP: 99240.222 ms
- Gate/up: 66211.872 ms
- Down: 32915.514 ms

CUTLASS SM120 result:

- Full prefill: 42382.857 ms
- Dense MLP: 1268.910 ms
- Gate/up + SiLU: 831.548 ms
- Down: 329.853 ms

Conclusion: the experimental MLP path removes the dense MLP as the long-prefill bottleneck. Remaining prefill work is now dominated by Qwen35 linear attention and full attention.
