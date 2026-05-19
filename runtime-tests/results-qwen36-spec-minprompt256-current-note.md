# Qwen36 Spec Min-Prompt 256 Probe

Generated: 2026-05-19 18:58 CEST

Change under test: the external Qwen36 profile was temporarily changed from
`RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=1024` to
`RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=256`, keeping
`RVLLM_QWEN36_SPEC_PREFLIGHT_MIN_FULL_DRAFTS=1`.

The goal was to allow prompt-lookup speculation on the 466-token repetitive
summary prompt while preserving the native route for short prompts that fail
the preflight.

| Probe | Min-prompt 256 total ms | Promoted baseline total ms | Decision |
|---|---:|---:|---|
| long_summary_300 | 5000.0 | 4869.5 | reject |
| long_summary_800 | 12692.0 | 12273.8 | reject |
| qwen_repeat_160 | 20759.0 | 20747.7 | reject |

The run passed 11/11 harness checks, but it did not improve any target probe.
The external Qwen36 profile was restored to
`RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=1024`.

Additional focused canary: disabling `RVLLM_QWEN36_SPEC_REPEAT_RUN_MIN` did not
change the repeat probe's `mu alpha beta gamma gamma...` output, and fully
disabling Qwen36 speculation produced the same output. That output is therefore
not caused by the repeated-tail speculative fast path.

Raw evidence:

- `runtime-tests/results-qwen36-spec-minprompt256-current.md`
- `runtime-tests/results-qwen36-spec-minprompt256-current.jsonl`
