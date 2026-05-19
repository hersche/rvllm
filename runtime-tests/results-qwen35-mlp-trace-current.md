# Qwen35 Batched MLP Trace

Generated: 2026-05-19 16:53 CEST

Profile: `mobile-qwen35-rvllm-nvfp4-spec`

Probe: one long text request against `qwen3-6-27b`, `max_tokens=1`, with
`RVLLM_QWEN35_PREFILL_PERF_TRACE=1` and `RVLLM_QWEN35_MLP_PERF_TRACE=1`.
This isolates Qwen35 batched prefill and splits dense MLP into its internal
substeps.

| Prompt tokens | Total prefill ms | Linear-attn ms | Full-attn ms | Dense MLP ms |
|---:|---:|---:|---:|---:|
| 944 | 141341.444 | 32991.125 | 9097.242 | 99240.222 |

Dense MLP subtotal across 64 layers:

| MLP substep | Total ms | Mean ms/layer | Share of MLP |
|---|---:|---:|---:|
| Gate/up FP8 dual-SiLU projection | 66211.872 | 1034.560 | 66.7% |
| Down FP8 projection | 32915.514 | 514.305 | 33.2% |
| Residual add | 8.132 | 0.127 | <0.1% |
| RMSNorm | 5.359 | 0.084 | <0.1% |
| Residual copy | 98.254 | 1.535 | 0.1% |

Decision: Qwen35 long-prefill optimization should target the row-batched FP8
projection kernels. The gate/up dual projection is the largest single target,
with the down projection second. RMSNorm, residual add, and copies are not
material bottlenecks. The raw trace is in
`runtime-tests/results-qwen35-mlp-trace-current.log`.
