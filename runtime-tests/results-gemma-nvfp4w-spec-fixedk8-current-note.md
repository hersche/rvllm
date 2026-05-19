# Gemma NVFP4W fixed K8 probe

Generated: 2026-05-19 21:07:48 CEST

## Scope

Tested a temporary Gemma candidate profile copied from the current promoted `mobile-31b-nvfp4w-rvllm-spec.env` with:

- `RVLLM_GEMMA4_SPEC_K=8`
- `G4N_SPEC_ADAPTIVE_K=0`

The promoted baseline remains fixed K=7 with adaptive K disabled.

Command:

```bash
sudo /home/r00t/workspace/upstream/rvllm-serve/runtime-tests/run_smoke.py \
  --profiles mobile-31b-nvfp4w-rvllm-spec-k8 \
  --skip-vision --skip-audio \
  --text-max-tokens 80 --request-timeout 300 \
  --restore-profile mobile-qwen-rvllm-nvfp4-spec \
  --results runtime-tests/results-gemma-nvfp4w-spec-fixedk8-current.md \
  --jsonl runtime-tests/results-gemma-nvfp4w-spec-fixedk8-current.jsonl
```

## Result

| Profile | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| `mobile-31b-nvfp4w-rvllm-spec-k8` | 10 | 0 | 31.361 |

## Comparison to promoted fixed K7

Promoted K7 comparison source: `runtime-tests/results-current-promoted-triad-from5-gemma-fixedk7.jsonl`.

| Label | K7 tok/s | K8 tok/s | K8 total ms | Verdict |
|---|---:|---:|---:|---|
| `short_capital` | 19.025 | 18.752 | 1546.5 | slower |
| `short_math` | 24.339 | 24.277 | 823.8 | flat |
| `short_pangram` | 19.311 | 19.253 | 1246.5 | flat |
| `medium_explain` | 9.371 | 9.322 | 10190.8 | slower |
| `medium_code` | 14.356 | 14.262 | 7221.8 | slower |
| `medium_translation` | 13.742 | 12.272 | 4237.3 | slower |
| `long_summary_300` | 93.818 | 92.740 | 5639.4 | slower |
| `long_summary_800` | 40.605 | 40.534 | 25756.1 | flat |
| `reasoning_chain` | 72.207 | 71.823 | 877.2 | flat |
| `instruction` | 10.427 | 10.373 | 11086.0 | flat |

K8 passed all quality canaries, but it reduced aggregate throughput from 31.720 to 31.361 mean decode tok/s and did not produce a compensating win on any key canary. Reject fixed K8 and keep Gemma at promoted fixed K7.

After the run, `rvllm-serve` was active with `/home/r00t/.rvllm/profiles/mobile-qwen-rvllm-nvfp4-spec.env` restored and serving `qwen3-6-35b-a3b`.

Raw evidence:

- `runtime-tests/results-gemma-nvfp4w-spec-fixedk8-current.md`
- `runtime-tests/results-gemma-nvfp4w-spec-fixedk8-current.jsonl`
