# Gemma4 31B NVFP4W Final-Row Fast Path

Generated: 2026-05-19 18:54 CEST

Focused change: `gemma-4-31b-it-nvfp4`, NVFP4 weights, NVFP4 KV, Option B
spec profile. The ordinary prompt-prefill path now finalizes only the selected
last prompt row on device when the caller needs one next-token prediction,
instead of copying all prompt residual rows to host and running one LM-head GEMV
per prompt row.

| Probe | Pass | Total ms | Prompt tokens | Completion tokens | Output | Baseline | Decision |
|---|:-:|---:|---:|---:|---|---:|---|
| long_summary_800_max1 | yes | 62368.4 | 1004 | 1 | `Der` | 78213.6 | promote |
| short_math | yes | 1704.7 | 18 | 2 | `2` | 1814.9 | sanity |
| reasoning_chain | yes | 4014.0 | 58 | 5 | `6 Äpfel` | 4959.5 | sanity |
| medium_code_max32 | yes | 4360.3 | 23 | 32 | Python slicing answer | n/a | sanity |

Baseline for `long_summary_800_max1` is the kept-current chunk-size-128 result
from `results-gemma-nvfp4w-prefill-chunk-sweep-current.md` under the same
profile and `max_tokens=1`. Result: same first token, 15845.2 ms faster
(20.3%). Keep the final-row fast path.
