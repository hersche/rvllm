# Qwen36 repetition guard probe

Generated: 2026-05-19 19:05 CEST

## Configuration

- Profile: `mobile-qwen-rvllm-nvfp4-spec`
- Model: `qwen3-6-35b-a3b`
- Spec decode: enabled, `K=4`, `min_drafts=4`
- Gates: `min_prompt_tokens=1024`, `min_max_new_tokens=64`, `preflight_min_full_drafts=1`
- Added runtime guard: `RVLLM_QWEN36_REPETITION_GUARD_N=24`

## Result

- Smoke result: 11 pass, 0 fail
- Mean decode throughput: 73.0 tok/s
- Normal probes stayed healthy: capital, math, pangram, explain, code, translation, summaries, reasoning, and instruction all passed.
- `qwen_repeat_160` stopped after 26 completion tokens in 15.85s instead of burning the full 160-token budget.
- Prior comparable K4 run without the guard emitted the degenerate `gamma` loop through the full budget in 20.75s.

## Caveat

The repeat canary is still not semantically solved. Native and speculative Qwen36 decode both collapse into `mu alpha beta gamma gamma ...`; the guard only prevents pathological same-token continuation and avoids wasting the rest of the requested token budget.

Journal evidence from the run:

```text
qwen36 repetition guard stopped generation token_id=20956 guard_n=24
```
