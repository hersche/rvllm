# Gemma4 31B NVFP4W MLP MMA-v8 Probe

Generated: 2026-05-19 19:18 CEST

Focused change: `gemma-4-31b-it-nvfp4`, NVFP4 weights, NVFP4 KV, Option B
spec profile, plus `RVLLM_GEMMA4_NVFP4_MLP_MMA_V8=1`. The Gemma NVFP4W MLP
M>1 W4A16 path can now route gate, up, and down projections through the
existing tensor-core persistent-CTA `mistral35_w4a16_gemm_mma_v8_bf16` kernel
instead of the CUDA-core M/N W4A16 kernel.

| Probe | Pass | Total ms | Prompt tokens | Completion tokens | Output | Baseline | Decision |
|---|:-:|---:|---:|---:|---|---:|---|
| long_summary_800_max1 | yes | 4153.1 | 1004 | 1 | `Während` | 62368.4 | promote |
| short_math | yes | 978.5 | 18 | 2 | `2` | 1704.7 | sanity |
| reasoning_chain | yes | 841.4 | 58 | 5 | `6 Äpfel` | 4014.0 | sanity |
| medium_translation | yes | 5504.6 | 27 | 25 | correct German idiom | n/a | sanity |
| long_summary_800_max32 | yes | 10264.9 | 1004 | 32 | coherent quantum/classical summary | n/a | sanity |

Baseline is the prior promoted final-row fast path. The open-ended summary
first token changes (`Der` -> `Während`), but sanity probes remain correct and
the 32-token long summary is coherent. Result: promote MMA-v8 for Gemma NVFP4W
MLP prefill.
