# Qwen35 NVFP4 Spec K Sweep

Generated: 2026-05-19 16:07 CEST

Focused probe: `qwen_repeat_160`, model `qwen3-6-27b`, NVFP4 KV, batched prefill, prompt-lookup speculation.

| Variant | Pass | Total ms | Prompt tokens | Completion tokens | Combined tok/s | Spec wall ms | Verify iters | Drafted | Accepted | Accept / verify | Decision |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| K=4 min_drafts=4 | yes | 213806.3 | 1260 | 160 | 6.641 | n/a | n/a | n/a | n/a | n/a | keep current |
| K=5 min_drafts=5 | yes | 216024.9 | 1260 | 160 | 6.573 | 25869.0 | 27 | 135 | 135 | 5.00 | reject |
| K=6 min_drafts=6 | yes | 215118.7 | 1260 | 160 | 6.601 | 26422.7 | 23 | 138 | 136 | 5.91 | reject |
| K=8 min_drafts=8 | yes | 217257.6 | 1260 | 160 | 6.536 | 27583.8 | 19 | 152 | 146 | 7.68 | reject |

Result: larger K values reduce speculative verifier count and decode-segment wall time, but all lose end-to-end on the real repeat probe. The request remains dominated by long Qwen35 prefill, so the promoted profile should stay at `RVLLM_QWEN35_SPEC_K=4` and `RVLLM_QWEN35_SPEC_MIN_DRAFTS=4`.
