# Qwen36 MTP Draft Canary

Generated: 2026-05-19 18:04 CEST

Focused canary: `qwen3-6-35b-a3b`, NVFP4 KV, Qwen36 profile with MTP loaded
and the spec loop forced onto ordinary prompts:

- `RVLLM_QWEN36_LOAD_MTP=1`
- `RVLLM_QWEN36_MTP_DRAFT=1`
- `RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=0`
- `RVLLM_QWEN36_SPEC_MIN_MAX_NEW_TOKENS=1`
- `RVLLM_QWEN36_SPEC_PREFLIGHT_MIN_FULL_DRAFTS=0`
- `RVLLM_QWEN36_SPEC_PERF_TRACE=1`

| Probe | Pass | Total ms | Prompt tokens | Completion tokens | Spec trace | Decision |
|---|:-:|---:|---:|---:|---|---|
| medium_code | yes | 957.6 | 23 | 22 | `mtp_accepted=8`, `mtp_shadow_matches=8/8`, `wall_ms=529.0` | reject |
| reasoning_chain | yes | 795.2 | 56 | 4 | `mtp_accepted=0`, `mtp_shadow_matches=0/2`, `wall_ms=216.6` | reject |
| qwen_repeat_64 | no | 18182.2 | 1259 | 64 | `mtp_accepted=11`, `mtp_shadow_matches=19/30`, `wall_ms=3598.8` | reject |

Result: verified MTP drafts are not ready for promotion. They add overhead on
ordinary prompts versus the current profile's native route (`medium_code`
baseline 767.5 ms, `reasoning_chain` baseline 668.1 ms) and changed the repeat
probe output while being slower than the current repeat-64 spec boundary result
(17173.1 ms). Keep MTP load/draft disabled in the promoted Qwen36 profile.
