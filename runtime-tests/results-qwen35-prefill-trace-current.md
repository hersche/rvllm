# Qwen35 Batched Prefill Trace

Generated: 2026-05-19 16:45 CEST

Profile: `mobile-qwen35-rvllm-nvfp4-spec`

Probe: one long text request against `qwen3-6-27b`, `max_tokens=1`, with
`RVLLM_QWEN35_PREFILL_PERF_TRACE=1`. This isolates the prompt prefill path;
speculative decode is not involved because `max_tokens` is below the spec gate.

| Prompt tokens | Total ms | Embed ms | Linear-attn ms | Full-attn ms | Dense MLP ms | Finalize ms |
|---:|---:|---:|---:|---:|---:|---:|
| 944 | 142016.982 | 0.634 | 33370.387 | 9217.224 | 99417.017 | 10.872 |

| Stage | Layers | Total ms | Mean ms/layer | Share |
|---|---:|---:|---:|---:|
| Dense MLP | 64 | 99417.017 | 1553.391 | 70.0% |
| Linear attention | 48 | 33370.387 | 695.216 | 23.5% |
| Full attention | 16 | 9217.224 | 576.077 | 6.5% |

Decision: the next Qwen35 performance target should be batched dense MLP, not
KV page geometry or full-attention prefill. The raw trace is in
`runtime-tests/results-qwen35-prefill-trace-current.log`.
