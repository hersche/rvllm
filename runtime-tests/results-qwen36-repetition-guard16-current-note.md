# Qwen36 repetition guard 16 probe

Generated: 2026-05-19 19:18 CEST

## Configuration

- Temporary external profile: `mobile-qwen-rvllm-nvfp4-spec-guard16`
- Base profile: `mobile-qwen-rvllm-nvfp4-spec`
- Only tested change: `RVLLM_QWEN36_REPETITION_GUARD_N=16`
- Kept promoted spec settings: `RVLLM_QWEN36_SPEC_K=4`, `RVLLM_QWEN36_SPEC_MIN_DRAFTS=4`, `RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=1024`, `RVLLM_QWEN36_SPEC_MIN_MAX_NEW_TOKENS=64`, `RVLLM_QWEN36_SPEC_PREFLIGHT_MIN_FULL_DRAFTS=1`

## Result

- Smoke result: 11 pass, 0 fail
- Mean combined throughput remained 73.0 tok/s, matching the guard24 run.
- Normal probes stayed healthy: capital, math, pangram, explain, code, translation, summaries, reasoning, and instruction all passed with timings in the same band as guard24.
- `qwen_repeat_160` stopped after 18 completion tokens in 15.49s.
- Prior promoted guard24 run stopped after 26 completion tokens in 15.85s.

## Decision

Promote `RVLLM_QWEN36_REPETITION_GUARD_N=16` in the external Qwen36 profile.
This stops the known same-token loop sooner without changing normal smoke
quality. The repeat canary remains a pathology guard, not a semantic exact-
pattern fix: Qwen36 still collapses into `mu alpha beta gamma gamma ...`.

Journal evidence from the run:

```text
qwen36 repetition guard stopped generation token_id=20956 guard_n=16
```

Raw evidence:

- `runtime-tests/results-qwen36-repetition-guard16-current.md`
- `runtime-tests/results-qwen36-repetition-guard16-current.jsonl`
- Baseline comparison: `runtime-tests/results-qwen36-repetition-guard-current.md`
