# Qwen35 Spec Preflight Gate Probe

Generated: 2026-05-19 18:42 CEST

Change under test: Qwen35 prompt-lookup speculation can now require a cheap
prompt preflight before entering the speculative path, controlled by
`RVLLM_QWEN35_SPEC_PREFLIGHT_MIN_FULL_DRAFTS`. The default is `0`, preserving
the prior gate. For this probe the external profile was temporarily changed
from `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=1024` to:

- `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=0`
- `RVLLM_QWEN35_SPEC_PREFLIGHT_MIN_FULL_DRAFTS=1`

The profile was restored to `RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=1024` after
the run.

| Probe | Result | Comparison | Decision |
|---|---:|---:|---|
| Full text smoke | 11/11 pass | matches current quality gate | keep code |
| Mean combined tok/s | 23.3 | current full-CUTLASS smoke was 24.8 | do not promote profile |
| qwen_repeat_160 | 28486.2 ms | current K=4 baseline was 28449.4 ms | do not promote profile |

Result: keep the Qwen35 preflight capability in code so lower thresholds can be
tested safely, but leave the promoted external profile at the prior
`RVLLM_QWEN35_SPEC_MIN_PROMPT_TOKENS=1024` setting.

Raw evidence:

- `runtime-tests/results-qwen35-spec-preflight-current.md`
- `runtime-tests/results-qwen35-spec-preflight-current.jsonl`
