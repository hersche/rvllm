# rvllm-serve runtime tests

Generated: 2026-05-19 19:36:03 CEST

## Summary

| Profile | Pass | Fail | Mean prefill tok/s | Mean decode tok/s |
|---|---:|---:|---:|---:|
| mobile-qwen35-rvllm-nvfp4-spec-k6-mindrafts4 | 11 | 0 | nan | 23.8 |

## Detailed records

| Profile | Kind | Label | Prompt (truncated) | Output (truncated) | prompt_tokens | completion_tokens | ttft ms | total ms | prefill tok/s | decode tok/s | ok |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|:-:|
| mobile-qwen35-rvllm-nvfp4-spec-k6-mindrafts4 | text | short_capital | Was ist die Hauptstadt von Frankreich? | Die Hauptstadt von Frankreich ist **Paris**. | 19 | 8 | — | 4223.5 | — | 6.4 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-k6-mindrafts4 | text | short_math | 1 + 1 = | 1 + 1 = 2 | 17 | 7 | — | 3684.3 | — | 6.5 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-k6-mindrafts4 | text | short_pangram | The quick brown fox jumps over the lazy | dog. | 20 | 2 | — | 3304.5 | — | 6.7 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-k6-mindrafts4 | text | medium_explain | Erkläre in drei Sätzen, was Photosynthese ist. | Photosynthese ist der biologische Prozess, bei dem Pflanzen, Algen und bestimmte Bakterien Lichtenergie in chemische Energie umwandeln. Dabei nutzen sie Sonnenlicht, um Kohlendioxid und Wasser in Gluk | 26 | 80 | — | 16952.6 | — | 6.3 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-k6-mindrafts4 | text | medium_code | Write a short Python function that reverses a string. | ```python def reverse_string(s: str) -> str:     return s[::-1] ``` | 23 | 22 | — | 7044.0 | — | 6.4 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-k6-mindrafts4 | text | medium_translation | Translate this sentence to German: 'The early bird catches the worm.' | Der frühe Vogel fängt den Wurm. | 26 | 10 | — | 5509.5 | — | 6.5 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-k6-mindrafts4 | text | long_summary_300 | Im Frühling blühen die Kirschbäume und die Tage werden länger. Die Vögel kehren  | Der Text beschreibt wiederholt, dass im Frühling die Kirschbäume blühen, die Tage länger werden, die Vögel aus dem Süden zurückkehren und das Gras zu wachsen beginnt. | 466 | 41 | — | 8037.2 | — | 63.1 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-k6-mindrafts4 | text | long_summary_800 | Quantum mechanics describes nature at the smallest scales of energy levels of at | Während die klassische Physik die Natur auf makroskopischer Ebene beschreibt, ist die Quantenmechanik erforderlich, um die Natur auf der kleinsten Ebene der Atome und subatomaren Teilchen zu beschreib | 1004 | 44 | — | 10885.7 | — | 96.3 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-k6-mindrafts4 | text | reasoning_chain | Anna hat 3 Äpfel. Sie gibt Tom 2 davon und kauft dann 5 weitere. Wie viele Äpfel | 6 Äpfel | 56 | 4 | — | 8953.5 | — | 6.7 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-k6-mindrafts4 | text | instruction | Liste drei Dinge auf, die man beachten sollte, wenn man eine Sauerteig-Brot back | Hier sind drei wesentliche Punkte, die beim Backen von Sauerteigbrot entscheidend sind:  1. **Die richtige Teigtemperatur und Umgebungstemperatur**      Sauerteig ist temperaturabhängig. Die ideale Te | 34 | 80 | — | 18157.3 | — | 6.3 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-k6-mindrafts4 | text | qwen_repeat_160 | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta ga | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu | 1260 | 160 | — | 28204.1 | — | 50.3 | ✅ |
