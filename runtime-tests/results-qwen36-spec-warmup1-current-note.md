# Qwen36 prompt-lookup warmup1 probe

Generated: 2026-05-19 19:27 CEST

## Configuration

- Temporary external profile: `mobile-qwen-rvllm-nvfp4-spec-warmup1`
- Base profile: `mobile-qwen-rvllm-nvfp4-spec`
- Only tested change: `RVLLM_QWEN36_SPEC_PROMPT_LOOKUP_WARMUP_MATCHES=1`
- Kept promoted settings: `RVLLM_QWEN36_SPEC_K=4`, `RVLLM_QWEN36_SPEC_MIN_DRAFTS=4`, `RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=1024`, `RVLLM_QWEN36_SPEC_MIN_MAX_NEW_TOKENS=64`, `RVLLM_QWEN36_SPEC_PREFLIGHT_MIN_FULL_DRAFTS=1`, `RVLLM_QWEN36_REPETITION_GUARD_N=16`

## Result

- Smoke result: 11 pass, 0 fail
- Mean combined throughput: 72.8 tok/s vs 73.0 tok/s on the promoted guard16 profile.
- Normal quality stayed green.
- `long_summary_800` was essentially unchanged: 12271.2 ms vs 12276.6 ms.
- `qwen_repeat_160` stayed governed by the repetition guard: 18 completion tokens in 15.47s vs 18 tokens in 15.49s.

## Decision

Do not promote `RVLLM_QWEN36_SPEC_PROMPT_LOOKUP_WARMUP_MATCHES=1`. It is
quality-safe on this smoke, but does not deliver a measurable net performance
gain over the current promoted Qwen36 profile.

Raw evidence:

- `runtime-tests/results-qwen36-spec-warmup1-current.md`
- `runtime-tests/results-qwen36-spec-warmup1-current.jsonl`
- Baseline comparison: `runtime-tests/results-qwen36-repetition-guard16-current.md`
