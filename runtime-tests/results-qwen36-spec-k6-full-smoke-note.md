# Qwen36 Spec K=6 Full-Smoke Probe

Generated: 2026-05-19 19:06 CEST

Change under test: the external Qwen36 profile was temporarily changed from:

- `RVLLM_QWEN36_SPEC_K=4`
- `RVLLM_QWEN36_SPEC_MIN_DRAFTS=4`

to:

- `RVLLM_QWEN36_SPEC_K=6`
- `RVLLM_QWEN36_SPEC_MIN_DRAFTS=6`

The rest of the Qwen36 profile remained unchanged, including
`RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=1024`,
`RVLLM_QWEN36_SPEC_MIN_MAX_NEW_TOKENS=64`, and
`RVLLM_QWEN36_SPEC_PREFLIGHT_MIN_FULL_DRAFTS=1`.

| Probe | K=6 total ms | K=4 promoted baseline total ms | Decision |
|---|---:|---:|---|
| Full text smoke | 11/11 pass | 11/11 pass | quality ok |
| Mean combined tok/s | 71.3 | 71.6 | keep K=4 |
| long_summary_800 | 12292.1 | 12273.8 | keep K=4 |
| qwen_repeat_160 | 20777.3 | 20747.7 | keep K=4 |

Result: K=6 preserves quality but does not improve the promoted Qwen36 profile.
The external Qwen36 profile was restored to K=4/min_drafts=4.

Raw evidence:

- `runtime-tests/results-qwen36-spec-k6-full-smoke-current.md`
- `runtime-tests/results-qwen36-spec-k6-full-smoke-current.jsonl`
- `runtime-tests/results-qwen-nvfp4-spec-minmax64-repeat-harness-current.md`
- `runtime-tests/results-qwen-nvfp4-spec-minmax64-repeat-harness-current.jsonl`
