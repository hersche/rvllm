# rvllm-serve runtime tests

Generated: 2026-05-19 19:13:57 CEST

## Summary

| Profile | Pass | Fail | Mean prefill tok/s | Mean decode tok/s |
|---|---:|---:|---:|---:|
| mobile-qwen35-rvllm-nvfp4-spec-minprompt0 | 11 | 0 | nan | 22.0 |

## Detailed records

| Profile | Kind | Label | Prompt (truncated) | Output (truncated) | prompt_tokens | completion_tokens | ttft ms | total ms | prefill tok/s | decode tok/s | ok |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|:-:|
| mobile-qwen35-rvllm-nvfp4-spec-minprompt0 | text | short_capital | Was ist die Hauptstadt von Frankreich? | Die Hauptstadt von Frankreich ist **Paris**. | 19 | 8 | — | 5249.2 | — | 5.1 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-minprompt0 | text | short_math | 1 + 1 = | 1 + 1 = 2 | 17 | 7 | — | 4737.4 | — | 5.1 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-minprompt0 | text | short_pangram | The quick brown fox jumps over the lazy | dog. | 20 | 2 | — | 3317.1 | — | 6.6 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-minprompt0 | text | medium_explain | Erkläre in drei Sätzen, was Photosynthese ist. | Photosynthese ist der biologische Prozess, bei dem Pflanzen, Algen und bestimmte Bakterien Lichtenergie in chemische Energie umwandeln. Dabei nutzen sie Sonnenlicht, um Kohlendioxid und Wasser in Gluk | 26 | 80 | — | 19099.4 | — | 5.5 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-minprompt0 | text | medium_code | Write a short Python function that reverses a string. | ```python def reverse_string(s: str) -> str:     return s[::-1] ``` | 23 | 22 | — | 7051.5 | — | 6.4 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-minprompt0 | text | medium_translation | Translate this sentence to German: 'The early bird catches the worm.' | Der frühe Vogel fängt den Wurm. | 26 | 10 | — | 5525.2 | — | 6.5 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-minprompt0 | text | long_summary_300 | Im Frühling blühen die Kirschbäume und die Tage werden länger. Die Vögel kehren  | Der Text beschreibt wiederholt, dass im Frühling die Kirschbäume blühen, die Tage länger werden, die Vögel aus dem Süden zurückkehren und das Gras zu wachsen beginnt. | 466 | 41 | — | 9406.1 | — | 53.9 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-minprompt0 | text | long_summary_800 | Quantum mechanics describes nature at the smallest scales of energy levels of at | Während die klassische Physik die Natur auf makroskopischer Ebene beschreibt, ist die Quantenmechanik erforderlich, um die Natur auf der kleinsten Ebene der Atome und subatomaren Teilchen zu beschreib | 1004 | 44 | — | 11643.1 | — | 90.0 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-minprompt0 | text | reasoning_chain | Anna hat 3 Äpfel. Sie gibt Tom 2 davon und kauft dann 5 weitere. Wie viele Äpfel | 6 Äpfel | 56 | 4 | — | 10027.8 | — | 6.0 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-minprompt0 | text | instruction | Liste drei Dinge auf, die man beachten sollte, wenn man eine Sauerteig-Brot back | Hier sind drei wesentliche Punkte, die beim Backen von Sauerteigbrot entscheidend sind:  1. **Die richtige Teigtemperatur und Umgebungstemperatur**      Sauerteig ist temperaturabhängig. Die ideale Te | 34 | 80 | — | 19297.9 | — | 5.9 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec-minprompt0 | text | qwen_repeat_160 | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta ga | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu | 1260 | 160 | — | 28168.9 | — | 50.4 | ✅ |
