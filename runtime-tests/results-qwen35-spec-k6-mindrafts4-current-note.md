# Qwen35 K6 min-drafts4 probe

Generated: 2026-05-19 19:36 CEST

## Configuration

- Temporary external profile: `mobile-qwen35-rvllm-nvfp4-spec-k6-mindrafts4`
- Base profile: `mobile-qwen35-rvllm-nvfp4-spec`
- Only tested change: `RVLLM_QWEN35_SPEC_MIN_DRAFTS=4`
- Kept promoted settings: `RVLLM_QWEN35_SPEC_K=6`, `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=1024`, `RVLLM_QWEN35_SPEC_MIN_MAX_NEW_TOKENS=64`, NVFP4 KV on, batched prefill on.

## Result

- Smoke result: 11 pass, 0 fail
- Mean combined throughput fell from 24.5 tok/s on promoted K6/min-drafts6 to 23.8 tok/s.
- Short and medium prompts were effectively unchanged because they stay below the spec gate.
- Mixed long-prompt result:
  - `long_summary_300`: 8037.2 ms vs 8254.3 ms promoted K6
  - `long_summary_800`: 10885.7 ms vs 9843.6 ms promoted K6
- Repeat-heavy probe was effectively tied:
  - `qwen_repeat_160`: 28204.1 ms vs 28212.6 ms promoted K6

## Decision

Do not promote `RVLLM_QWEN35_SPEC_MIN_DRAFTS=4`. The lower verifier threshold
is quality-safe on this smoke, but it does not produce a net performance win
over the current K6/min-drafts6 Qwen35 profile.

Raw evidence:

- `runtime-tests/results-qwen35-spec-k6-mindrafts4-current.md`
- `runtime-tests/results-qwen35-spec-k6-mindrafts4-current.jsonl`
- Baseline comparison: `runtime-tests/results-qwen35-spec-k6-full-smoke-current.md`
