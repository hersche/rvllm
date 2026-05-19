# Qwen min-max-new-token boundary probes

| Probe | Route | Threshold | Prompt | Completion | Total ms | Combined tok/s |
|---|---|---:|---:|---:|---:|---:|
| repeat-40 ask160 | native | 64 | 1260 | 40 | 16170.3 | 80.39 |
| repeat-40 ask160 | spec | 40 | 1260 | 40 | 16270.8 | 79.90 |
| repeat-64 ask160 | native | 80 | 1260 | 64 | 17221.6 | 76.88 |
| repeat-64 ask160 | spec | 64 | 1260 | 64 | 17173.1 | 77.10 |
| repeat-80 ask160 | native | 128 | 1260 | 80 | 17799.4 | 75.28 |
| repeat-80 ask160 | spec | 80 | 1260 | 80 | 17740.4 | 75.53 |
| repeat-96 ask160 | native | 128 | 1260 | 96 | 18578.6 | 72.99 |
| repeat-96 ask160 | spec | 96 | 1260 | 96 | 18297.8 | 74.11 |
| repeat-128 | native | 160 | 1260 | 128 | 20030.6 | 69.29 |
| repeat-128 | spec | 128 | 1260 | 128 | 19644.5 | 70.66 |

Decision: promote `RVLLM_QWEN36_SPEC_MIN_MAX_NEW_TOKENS=64`. With the established ask-160 repeat prompt, 40-token requests are faster native, while 64/80/96/128-token repeat requests are equal-or-faster when speculation is allowed.
