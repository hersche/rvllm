# Gemma4 31B NVFP4W Spec Post-MMA-v8 Sweep

Generated: 2026-05-19 18:34 CEST

Scope: `gemma-4-31b-it-nvfp4`, NVFP4 weights, NVFP4 KV, Gemma speculative
decode profile with `RVLLM_GEMMA4_NVFP4_MLP_MMA_V8=1`,
`G4N_SPEC_ADAPTIVE_WINDOW_ITERS=3`, and `G4N_SPEC_ADAPTIVE_MIN_K=1`.

The current promoted profile uses `RVLLM_GEMMA4_SPEC_K=5`. A post-MMA-v8
refresh was run against K=5, then a temporary external profile edit changed only
`RVLLM_GEMMA4_SPEC_K=6` for a comparison run. The external profile was restored
to K=5 afterward.

| Probe | K=5 total ms | K=6 total ms | K=5 output/quality | K=6 output/quality | Decision |
|---|---:|---:|---|---|---|
| short_capital | 1521.4 | 1531.0 | Paris, pass | Paris, pass | keep K=5 |
| short_math | 796.3 | 802.5 | `2`, pass | `2`, pass | keep K=5 |
| short_pangram | 1193.2 | 1202.2 | dog, pass | dog, pass | keep K=5 |
| medium_explain | 14869.6 | 14954.5 | coherent, pass | coherent, pass | keep K=5 |
| medium_code | 6925.0 | 6963.1 | slicing implementation, pass | slicing implementation, pass | keep K=5 |
| medium_translation | 5522.5 | 5537.4 | German idiom, pass | German idiom, pass | keep K=5 |
| long_summary_300 | 6538.3 | 6578.9 | coherent summary, pass | coherent summary, pass | keep K=5 |
| long_summary_800 | 25433.4 | 25526.0 | coherent summary, pass | coherent summary, pass | keep K=5 |
| reasoning_chain | 848.8 | 852.8 | `6 Äpfel`, pass | `6 Äpfel`, pass | keep K=5 |
| instruction | 14197.9 | 14238.5 | coherent list, pass | coherent list, pass | keep K=5 |

Both runs passed 10/10 quality checks. K=6 is consistently but only slightly
slower on this post-MMA-v8 smoke, so the promoted external profile remains at
`RVLLM_GEMMA4_SPEC_K=5`.

Raw evidence:

- `runtime-tests/results-gemma-nvfp4w-spec-post-mma-v8-current.md`
- `runtime-tests/results-gemma-nvfp4w-spec-post-mma-v8-current.jsonl`
- `runtime-tests/results-gemma-nvfp4w-spec-k6-post-mma-v8-current.md`
- `runtime-tests/results-gemma-nvfp4w-spec-k6-post-mma-v8-current.jsonl`
