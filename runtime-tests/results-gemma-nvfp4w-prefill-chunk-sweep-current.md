# Gemma4 31B NVFP4W Prefill Chunk Sweep

Generated: 2026-05-19 17:59 CEST

Focused probe: `long_summary_800`, model `gemma-4-31b-it-nvfp4`, NVFP4 weights,
NVFP4 KV, Option B speculative profile, `max_tokens=1`, greedy decode.

| RVLLM_PREFILL_CHUNK_SIZE | Pass | Total ms | Prompt tokens | Completion tokens | First token | Decision |
|---:|:-:|---:|---:|---:|---|---|
| 128 | yes | 78213.6 | 1004 | 1 | Der | keep current |
| 256 | yes | 78601.1 | 1004 | 1 | Der | reject |
| 512 | yes | 78608.8 | 1004 | 1 | Der | reject |
| 0 | yes | 78766.0 | 1004 | 1 | Der | reject |

Result: Gemma4 Option B NVFP4-weight long-prompt prefill is flat across 2-8
chunks and unchunked execution for this 1004-token probe. Keep the promoted
`RVLLM_PREFILL_CHUNK_SIZE=128`; the next performance work should target the
Option B layer internals rather than this profile knob.
