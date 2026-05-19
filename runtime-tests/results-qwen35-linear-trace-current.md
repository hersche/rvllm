# Qwen35 linear-attention trace

Generated: 2026-05-19 17:24 CEST

Profile: `mobile-qwen35-rvllm-nvfp4-spec` with the promoted CUTLASS
MLP path enabled plus:

- `RVLLM_QWEN35_PREFILL_PERF_TRACE=1`
- `RVLLM_QWEN35_LINEAR_PERF_TRACE=1`

Probe: 944 prompt tokens, `max_tokens=1`, temperature 0.

## Prefill Summary

| Stage | Total ms | Mean per layer ms | Share |
|---|---:|---:|---:|
| Full prefill | 42329.648 | - | 100.0% |
| Linear attention | 32218.965 | 671.229 over 48 linear layers | 76.1% |
| Full attention | 8892.619 | 555.789 over 16 full layers | 21.0% |
| Dense MLP | 1205.832 | 18.841 over 64 layers | 2.8% |

## Linear-Attention Substages

| Linear substage | Total ms | Mean per linear layer ms | Share of linear |
|---|---:|---:|---:|
| RMSNorm copy + norm | 9.824 | 0.205 | 0.0% |
| QKV FP8 projection | 14424.998 | 300.521 | 44.8% |
| Conv state advance | 16.049 | 0.334 | 0.0% |
| Causal conv1d | 14.693 | 0.306 | 0.0% |
| SiLU/L2/GQA split | 20.850 | 0.434 | 0.1% |
| Alpha/beta | 35.204 | 0.733 | 0.1% |
| Gated delta prefill | 343.615 | 7.159 | 1.1% |
| Z FP8 projection | 8647.500 | 180.156 | 26.8% |
| RMS gated | 13.442 | 0.280 | 0.0% |
| Output FP8 projection | 8685.334 | 180.944 | 27.0% |
| Residual add | 6.033 | 0.126 | 0.0% |

## Conclusion

The Qwen35 long-prefill bottleneck moved from dense MLP to the
linear-attention projection triplet. QKV, Z, and output projection
account for about 98.6% of linear-attention time; the recurrent
gated-delta kernel is only about 1.1%.

Next target: apply the same M>=128 CUTLASS SM120 blockscale FP8
projection path to Qwen35 batched linear-attention QKV, Z, and output
projections, behind a quality gate.
