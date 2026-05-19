# Qwen35 linear CUTLASS SM120 trace

Generated: 2026-05-19 17:29 CEST

Profile: `mobile-qwen35-rvllm-nvfp4-spec` with:

- `RVLLM_QWEN35_MLP_CUTLASS_SM120=1`
- `RVLLM_QWEN35_LINEAR_CUTLASS_SM120=1`
- `RVLLM_QWEN35_LINEAR_CUTLASS_MIN_TOKENS=128`
- `RVLLM_QWEN35_PREFILL_PERF_TRACE=1`
- `RVLLM_QWEN35_LINEAR_PERF_TRACE=1`

Probe: 944 prompt tokens, `max_tokens=1`, temperature 0.

## Prefill Summary

| Stage | Total ms | Mean per layer ms | Share |
|---|---:|---:|---:|
| Full prefill | 10826.681 | - | 100.0% |
| Full attention | 8895.518 | 555.970 over 16 full layers | 82.2% |
| Dense MLP | 1193.088 | 18.642 over 64 layers | 11.0% |
| Linear attention | 725.573 | 15.116 over 48 linear layers | 6.7% |

## Linear-Attention Substages

| Linear substage | Total ms | Mean per linear layer ms | Share of linear |
|---|---:|---:|---:|
| RMSNorm copy + norm | 10.251 | 0.214 | 1.4% |
| QKV CUTLASS projection | 51.663 | 1.076 | 7.1% |
| Conv state advance | 15.634 | 0.326 | 2.2% |
| Causal conv1d | 15.485 | 0.323 | 2.1% |
| SiLU/L2/GQA split | 23.765 | 0.495 | 3.3% |
| Alpha/beta | 35.875 | 0.747 | 4.9% |
| Gated delta prefill | 374.344 | 7.799 | 51.6% |
| Z CUTLASS projection | 31.752 | 0.662 | 4.4% |
| RMS gated | 47.969 | 0.999 | 6.6% |
| Output CUTLASS projection | 112.145 | 2.336 | 15.5% |
| Residual add | 5.541 | 0.115 | 0.8% |

## Comparison

Before the linear CUTLASS branch, the same 944-token probe with the
MLP CUTLASS path active spent:

- Full prefill: 42329.648 ms
- Linear attention: 32218.965 ms
- QKV/Z/output projections: 31757.832 ms combined

With linear CUTLASS enabled:

- Full prefill: 10826.681 ms
- Linear attention: 725.573 ms
- QKV/Z/output projections: 195.560 ms combined

Conclusion: Qwen35 long-prefill is now dominated by full attention,
not dense MLP or linear-attention projections.
