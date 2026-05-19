# Qwen35 Dual-SiLU `__expf` Probe

Generated: 2026-05-19 17:00 CEST

Profile: `mobile-qwen35-rvllm-nvfp4-spec`

Probe: one long text request against `qwen3-6-27b`, `max_tokens=1`, with
`RVLLM_QWEN35_PREFILL_PERF_TRACE=1` and `RVLLM_QWEN35_MLP_PERF_TRACE=1`.
The only code change under test was replacing `expf` with CUDA `__expf` in
`fp8_gemv_blockwise_wpr_native_f16in_dual_silu_kernel`.

| Variant | Total prefill ms | Dense MLP ms | Gate/up ms | Down ms |
|---|---:|---:|---:|---:|
| Baseline (`expf`) | 141341.444 | 99240.222 | 66211.872 | 32915.514 |
| Probe (`__expf`) | 141271.492 | 99206.093 | 66191.986 | 32903.279 |

Decision: rejected. The measured gate/up change is about 0.03%, which is below
the noise floor for this trace and does not justify changing numerics in a
shared kernel. The Qwen35 MLP bottleneck remains the row-batched FP8 projection
algorithm itself, not the SiLU exponential epilogue. The raw trace is in
`runtime-tests/results-qwen35-dualsilu-expf-current.log`.
