# Qwen36 spec min-prompt 512 from5 probe

Generated: 2026-05-19 21:12:08 CEST

## Scope

Tested a temporary Qwen36 candidate profile copied from the current promoted `mobile-qwen-rvllm-nvfp4-spec.env` with:

- `RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=512`

All other promoted Qwen36 settings were kept, including:

- `RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_FROM=5`
- `RVLLM_QWEN36_SPEC_K=4`
- `RVLLM_QWEN36_SPEC_MIN_DRAFTS=4`
- `RVLLM_QWEN36_SPEC_MIN_MAX_NEW_TOKENS=64`
- `RVLLM_QWEN36_REPETITION_GUARD_N=12`

The promoted baseline remains `RVLLM_QWEN36_SPEC_MIN_PROMPT_TOKENS=1024`.

Command:

```bash
sudo /home/r00t/workspace/upstream/rvllm-serve/runtime-tests/run_smoke.py \
  --profiles mobile-qwen-rvllm-nvfp4-spec-minprompt512 \
  --skip-vision --skip-audio --include-repeat-probes \
  --text-max-tokens 80 --request-timeout 300 \
  --restore-profile mobile-qwen-rvllm-nvfp4-spec \
  --results runtime-tests/results-qwen36-spec-minprompt512-from5-current.md \
  --jsonl runtime-tests/results-qwen36-spec-minprompt512-from5-current.jsonl
```

## Result

| Profile | Pass | Fail | Mean decode tok/s |
|---|---:|---:|---:|
| `mobile-qwen-rvllm-nvfp4-spec-minprompt512` | 11 | 0 | 74.358 |

## Comparison to promoted 1024 gate

Promoted comparison source: `runtime-tests/results-current-promoted-triad-from5-gemma-fixedk7.jsonl`.

| Label | 1024 tok/s | 512 tok/s | 512 total ms | Verdict |
|---|---:|---:|---:|---|
| `short_capital` | 65.217 | 65.344 | 413.2 | flat |
| `short_math` | 67.802 | 67.836 | 353.8 | flat |
| `short_pangram` | 81.976 | 81.921 | 268.6 | flat |
| `medium_explain` | 46.343 | 44.875 | 2362.1 | slower |
| `medium_code` | 58.590 | 58.615 | 767.7 | flat |
| `medium_translation` | 69.208 | 69.193 | 520.3 | flat |
| `long_summary_300` | 109.579 | 109.592 | 4681.0 | flat |
| `long_summary_800` | 89.253 | 88.465 | 11936.9 | slower |
| `reasoning_chain` | 90.223 | 90.121 | 665.8 | flat |
| `instruction` | 50.560 | 50.537 | 2255.8 | flat |
| `qwen_repeat_160` | 91.467 | 91.434 | 13933.6 | flat |

The 512-token prompt gate preserved all quality canaries, but it reduced aggregate throughput from 74.565 to 74.358 mean decode tok/s and made `long_summary_800` slower. Reject min-prompt 512 and keep the promoted 1024-token gate.

After the run, `rvllm-serve` was active with `/home/r00t/.rvllm/profiles/mobile-qwen-rvllm-nvfp4-spec.env` restored and serving `qwen3-6-35b-a3b`.

Raw evidence:

- `runtime-tests/results-qwen36-spec-minprompt512-from5-current.md`
- `runtime-tests/results-qwen36-spec-minprompt512-from5-current.jsonl`
