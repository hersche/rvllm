# Qwen35 NVFP4 Spec K Sweep

Generated: 2026-05-19 17:48 CEST

Focused probe: `qwen_repeat_160`, model `qwen3-6-27b`, NVFP4 KV, batched prefill, prompt-lookup speculation.
Current profile includes Qwen35 MLP, linear-attn, and full-attn prefill projection CUTLASS SM120 paths.

| Variant | Pass | Total ms | Prompt tokens | Completion tokens | Combined tok/s | Spec wall ms | Verify iters | Drafted | Accepted | Accept / verify | Decision |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| no spec | yes | 33636.8 | 1260 | 160 | 42.216 | n/a | n/a | n/a | n/a | n/a | reject |
| K=4 min_drafts=4 | yes | 29271.8 | 1260 | 160 | 48.510 | 26738.4 | 33 | 132 | 129 | 3.91 | keep current |
| K=6 min_drafts=6 | yes | 29141.0 | 1260 | 160 | 48.729 | 26391.9 | 23 | 138 | 136 | 5.91 | no promotion |
| K=8 min_drafts=8 | yes | 29832.5 | 1260 | 160 | 47.599 | 27604.1 | 19 | 152 | 146 | 7.68 | reject |

Result: post-CUTLASS prefill, prompt-lookup speculation still helps the repeat-heavy Qwen35 case
(29.3s K=4 versus 33.6s native). K=6 reduces verifier iterations from 33 to 23, but the
single-probe end-to-end gain is only 0.4% and has not been full-smoke quality-gated, so the
promoted profile should stay at `RVLLM_QWEN35_SPEC_K=4` and `RVLLM_QWEN35_SPEC_MIN_DRAFTS=4`.
