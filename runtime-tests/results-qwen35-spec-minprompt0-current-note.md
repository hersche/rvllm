# Qwen35 spec min-prompt 0 probe

Generated: 2026-05-19 19:14 CEST

## Configuration

- Temporary external profile: `mobile-qwen35-rvllm-nvfp4-spec-minprompt0`
- Base profile: `mobile-qwen35-rvllm-nvfp4-spec`
- Only tested change: `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=0`
- Kept promoted spec settings: `RVLLM_QWEN35_SPEC_K=6`, `RVLLM_QWEN35_SPEC_MIN_DRAFTS=6`, `RVLLM_QWEN35_SPEC_MIN_MAX_NEW_TOKENS=64`

## Result

- Smoke result: 11 pass, 0 fail
- Mean combined throughput fell from 24.5 tok/s on the promoted K6 profile to 22.0 tok/s.
- Short and medium prompts did not improve:
  - `short_capital`: 5249.2 ms vs 4226.2 ms promoted K6
  - `short_math`: 4737.4 ms vs 3685.2 ms promoted K6
  - `medium_explain`: 19099.4 ms vs 16967.4 ms promoted K6
- Long prompts also regressed:
  - `long_summary_300`: 9406.1 ms vs 8254.3 ms promoted K6
  - `long_summary_800`: 11643.1 ms vs 9843.6 ms promoted K6
- Repeat-heavy probe was effectively unchanged:
  - `qwen_repeat_160`: 28168.9 ms vs 28212.6 ms promoted K6

## Decision

Do not promote `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=0`. The existing `1024`
prompt-token gate remains the better quality-preserving performance setting:
it keeps the repeat-heavy benefit while avoiding extra prompt-lookup spec
overhead on ordinary short and medium requests.

Raw evidence:

- `runtime-tests/results-qwen35-spec-minprompt0-current.md`
- `runtime-tests/results-qwen35-spec-minprompt0-current.jsonl`
- Baseline comparison: `runtime-tests/results-qwen35-spec-k6-full-smoke-current.md`
