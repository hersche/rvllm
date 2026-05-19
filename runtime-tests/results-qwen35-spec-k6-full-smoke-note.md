# Qwen35 Spec K=6 Full-Smoke Probe

Generated: 2026-05-19 18:58 CEST

Change promoted in the external Qwen35 profile:

- `RVLLM_QWEN35_SPEC_K=6`
- `RVLLM_QWEN35_SPEC_MIN_DRAFTS=6`

The rest of the Qwen35 profile remains unchanged, including:

- `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=1024`
- `RVLLM_QWEN35_SPEC_MIN_MAX_NEW_TOKENS=64`
- Qwen35 MLP, linear-attention, and full-attention CUTLASS SM120 prefill paths

| Probe | K=6 full-smoke result | K=4 promoted baseline | Decision |
|---|---:|---:|---|
| Full text smoke | 11/11 pass | 11/11 pass | quality ok |
| Mean combined tok/s | 24.5 | 24.8 | acceptable; mostly native-route noise |
| qwen_repeat_160 | 28212.6 ms | 28449.4 ms | promote K=6 |

K=6 only affects spec-eligible requests under the current prompt/max-token
gates. The full smoke confirms no quality regression, and the repeat-heavy spec
case improves by 236.8 ms versus the current full-CUTLASS K=4 baseline.

Raw evidence:

- `runtime-tests/results-qwen35-spec-k6-full-smoke-current.md`
- `runtime-tests/results-qwen35-spec-k6-full-smoke-current.jsonl`
- `runtime-tests/results-qwen35-full-cutlass-sm120-current.md`
- `runtime-tests/results-qwen35-full-cutlass-sm120-current.jsonl`
