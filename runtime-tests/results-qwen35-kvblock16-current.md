# rvllm-serve runtime tests

Generated: 2026-05-19 16:22:42 CEST

## Summary

| Profile | Pass | Fail | Mean prefill tok/s | Mean decode tok/s |
|---|---:|---:|---:|---:|
| mobile-qwen35-rvllm-nvfp4-spec | 11 | 0 | nan | 6.5 |

## Detailed records

| Profile | Kind | Label | Prompt (truncated) | Output (truncated) | prompt_tokens | completion_tokens | ttft ms | total ms | prefill tok/s | decode tok/s | ok |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|:-:|
| mobile-qwen35-rvllm-nvfp4-spec | text | short_capital | Was ist die Hauptstadt von Frankreich? | Die Hauptstadt von Frankreich ist **Paris**. | 19 | 8 | — | 4254.0 | — | 6.3 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | short_math | 1 + 1 = | 1 + 1 = 2 | 17 | 7 | — | 3682.7 | — | 6.5 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | short_pangram | The quick brown fox jumps over the lazy | dog. | 20 | 2 | — | 3307.4 | — | 6.7 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | medium_explain | Erkläre in drei Sätzen, was Photosynthese ist. | Photosynthese ist der biologische Prozess, bei dem Pflanzen, Algen und bestimmte Bakterien Lichtenergie in chemische Energie umwandeln. Dabei nutzen sie Sonnenlicht, um Kohlendioxid und Wasser in Gluk | 26 | 80 | — | 16967.5 | — | 6.2 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | medium_code | Write a short Python function that reverses a string. | ```python def reverse_string(s: str) -> str:     return s[::-1] ``` | 23 | 22 | — | 7044.3 | — | 6.4 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | medium_translation | Translate this sentence to German: 'The early bird catches the worm.' | Der frühe Vogel fängt den Wurm. | 26 | 10 | — | 5509.0 | — | 6.5 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | long_summary_300 | Im Frühling blühen die Kirschbäume und die Tage werden länger. Die Vögel kehren  | Der Text beschreibt wiederholt, dass im Frühling die Kirschbäume blühen, die Tage länger werden, die Vögel aus dem Süden zurückkehren und das Gras zu wachsen beginnt. | 466 | 41 | — | 77243.1 | — | 6.6 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | long_summary_800 | Quantum mechanics describes nature at the smallest scales of energy levels of at | Während die klassische Physik die Natur auf makroskopischer Ebene beschreibt, ist die Quantenmechanik notwendig, um die Natur auf der kleinsten Ebene der Atome und subatomaren Teilchen zu beschreiben. | 1004 | 44 | — | 158106.6 | — | 6.6 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | reasoning_chain | Anna hat 3 Äpfel. Sie gibt Tom 2 davon und kauft dann 5 weitere. Wie viele Äpfel | 6 Äpfel | 56 | 4 | — | 8959.7 | — | 6.7 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | instruction | Liste drei Dinge auf, die man beachten sollte, wenn man eine Sauerteig-Brot back | Hier sind drei wesentliche Punkte, die beim Backen von Sauerteigbrot entscheidend sind:  1. **Die richtige Teigtemperatur und Umgebungstemperatur**      Sauerteig ist temperaturabhängig. Die ideale Te | 34 | 80 | — | 18155.2 | — | 6.3 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | qwen_repeat_160 | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta ga | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu | 1260 | 160 | — | 214021.9 | — | 6.6 | ✅ |

## Decision

Rejected. Qwen35 KV `block_size=16` with identity paged slots preserved correctness
(11/11 pass) but regressed end-to-end throughput versus the promoted
`block_size=1` geometry. The focused repeat probe moved from 213806.3 ms
(`runtime-tests/results-qwen35-spec-k-sweep-current.md`, K=4 baseline) to
214021.9 ms, and the normal long prompts were materially slower. Keep the
current Qwen35 profile and runtime geometry.
