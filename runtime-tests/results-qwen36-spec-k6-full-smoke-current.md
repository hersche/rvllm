# rvllm-serve runtime tests

Generated: 2026-05-19 18:59:43 CEST

## Summary

| Profile | Pass | Fail | Mean prefill tok/s | Mean decode tok/s |
|---|---:|---:|---:|---:|
| mobile-qwen-rvllm-nvfp4-spec | 11 | 0 | nan | 71.6 |

## Detailed records

| Profile | Kind | Label | Prompt (truncated) | Output (truncated) | prompt_tokens | completion_tokens | ttft ms | total ms | prefill tok/s | decode tok/s | ok |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|:-:|
| mobile-qwen-rvllm-nvfp4-spec | text | short_capital | Was ist die Hauptstadt von Frankreich? | Die Hauptstadt von Frankreich ist **Paris**. | 19 | 8 | — | 406.0 | — | 66.5 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | short_math | 1 + 1 = | 1 + 1 = 2 | 17 | 7 | — | 355.3 | — | 67.6 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | short_pangram | The quick brown fox jumps over the lazy | dog. | 20 | 2 | — | 270.0 | — | 81.5 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | medium_explain | Erkläre in drei Sätzen, was Photosynthese ist. | Die Photosynthese ist der biochemische Prozess, bei dem Pflanzen, Algen und einige Bakterien Lichtenergie nutzen, um aus Kohlendioxid und Wasser Glucose sowie Sauerstoff zu produzieren. Dieser Prozess | 26 | 80 | — | 2247.1 | — | 47.2 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | medium_code | Write a short Python function that reverses a string. | ```python def reverse_string(s: str) -> str:     return s[::-1] ``` | 23 | 22 | — | 770.3 | — | 58.4 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | medium_translation | Translate this sentence to German: 'The early bird catches the worm.' | Der frühe Vogel fängt den Wurm. | 26 | 10 | — | 523.4 | — | 68.8 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | long_summary_300 | Im Frühling blühen die Kirschbäume und die Tage werden länger. Die Vögel kehren  | Der Text beschreibt wiederholt die typischen Merkmale des Frühlings, wie das Blühen der Kirschbäume, das Längerwerden der Tage, die Rückkehr der Vögel und das Wachsen des Grases. | 466 | 47 | — | 4879.9 | — | 105.1 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | long_summary_800 | Quantum mechanics describes nature at the smallest scales of energy levels of at | Quantum mechanics describes nature at the smallest scales of atoms and subatomic particles, whereas classical physics describes nature at ordinary, macroscopic scales but is insufficient for describin | 1004 | 40 | — | 12292.0 | — | 84.9 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | reasoning_chain | Anna hat 3 Äpfel. Sie gibt Tom 2 davon und kauft dann 5 weitere. Wie viele Äpfel | 6 Äpfel | 56 | 4 | — | 672.4 | — | 89.2 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | instruction | Liste drei Dinge auf, die man beachten sollte, wenn man eine Sauerteig-Brot back | Hier sind drei wichtige Punkte, die man beim Backen von Sauerteigbrot beachten sollte:  1. **Geduld bei der Gehzeit (Gare):** Sauerteigbrot braucht deutlich mehr Zeit als Hefebrot. Der Teig braucht of | 34 | 80 | — | 2254.8 | — | 50.6 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | qwen_repeat_160 | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta ga | mu alpha beta gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma  | 1260 | 160 | — | 20777.1 | — | 68.3 | ✅ |
