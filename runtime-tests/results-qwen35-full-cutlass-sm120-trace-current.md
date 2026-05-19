# Qwen35 full-attention CUTLASS SM120 trace

Generated: 2026-05-19 17:37 CEST

Profile: `mobile-qwen35-rvllm-nvfp4-spec` with:

- `RVLLM_QWEN35_MLP_CUTLASS_SM120=1`
- `RVLLM_QWEN35_LINEAR_CUTLASS_SM120=1`
- `RVLLM_QWEN35_FULL_CUTLASS_SM120=1`
- `RVLLM_QWEN35_FULL_CUTLASS_MIN_TOKENS=128`
- `RVLLM_QWEN35_PREFILL_PERF_TRACE=1`

Probe: 944 prompt tokens, `max_tokens=1`, temperature 0.

## Prefill Summary

| Stage | Total ms | Mean per layer ms | Share |
|---|---:|---:|---:|
| Full prefill | 2235.056 | - | 100.0% |
| Dense MLP | 1305.062 | 20.392 over 64 layers | 58.4% |
| Linear attention | 752.792 | 15.683 over 48 linear layers | 33.7% |
| Full attention | 165.057 | 10.316 over 16 full layers | 7.4% |
| Embed + finalize | 11.488 | - | 0.5% |

## Comparison

Before full-attention CUTLASS projection dispatch, the same 944-token
probe with MLP and linear CUTLASS enabled spent:

- Full prefill: 10826.681 ms
- Full attention: 8895.518 ms

With full-attention CUTLASS enabled:

- Full prefill: 2235.056 ms
- Full attention: 165.057 ms

Full-attention prefill is no longer the dominant Qwen35 cost. The
remaining long-prefill time is split between dense MLP and linear
attention overhead, both already using their CUTLASS projection paths.
