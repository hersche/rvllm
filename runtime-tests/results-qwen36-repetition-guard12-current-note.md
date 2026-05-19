# Qwen36 repetition guard 12 probe

Generated: 2026-05-19 19:31 CEST

## Configuration

- Temporary external profile: `mobile-qwen-rvllm-nvfp4-spec-guard12`
- Base profile: `mobile-qwen-rvllm-nvfp4-spec`
- Only tested change: `RVLLM_QWEN36_REPETITION_GUARD_N=12`
- Kept promoted spec settings: `RVLLM_QWEN36_SPEC_K=4`, `RVLLM_QWEN36_SPEC_MIN_DRAFTS=4`, `RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=1024`, `RVLLM_QWEN36_SPEC_MIN_MAX_NEW_TOKENS=64`, `RVLLM_QWEN36_SPEC_PREFLIGHT_MIN_FULL_DRAFTS=1`

## Result

- Smoke result: 11 pass, 0 fail
- Mean combined throughput: 72.8 tok/s vs 73.0 tok/s on the guard16 run.
- Normal probes stayed healthy: capital, math, pangram, explain, code, translation, summaries, reasoning, and instruction all passed.
- `qwen_repeat_160` stopped after 14 completion tokens in 15.31s.
- Prior promoted guard16 run stopped after 18 completion tokens in 15.49s.

## Decision

Promote `RVLLM_QWEN36_REPETITION_GUARD_N=12` in the external Qwen36 profile.
This stops the known same-token loop sooner while preserving normal smoke
quality. As with guard16, this is a pathology guard, not an exact-pattern
semantic fix: Qwen36 still collapses into `mu alpha beta gamma gamma ...`.

Journal evidence from the run:

```text
qwen36 repetition guard stopped generation token_id=20956 guard_n=12
```

Raw evidence:

- `runtime-tests/results-qwen36-repetition-guard12-current.md`
- `runtime-tests/results-qwen36-repetition-guard12-current.jsonl`
- Baseline comparison: `runtime-tests/results-qwen36-repetition-guard16-current.md`
