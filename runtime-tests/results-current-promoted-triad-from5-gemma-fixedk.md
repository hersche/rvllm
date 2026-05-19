# rvllm-serve runtime tests

Generated: 2026-05-19 20:31:59 CEST

## Summary

| Profile | Pass | Fail | Mean prefill tok/s | Mean decode tok/s |
|---|---:|---:|---:|---:|
| mobile-qwen-rvllm-nvfp4-spec | 11 | 0 | nan | 74.5 |
| mobile-qwen35-rvllm-nvfp4-spec | 11 | 0 | nan | 23.9 |
| mobile-31b-nvfp4w-rvllm-spec | 10 | 0 | nan | 30.6 |

## Detailed records

| Profile | Kind | Label | Prompt (truncated) | Output (truncated) | prompt_tokens | completion_tokens | ttft ms | total ms | prefill tok/s | decode tok/s | ok |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|:-:|
| mobile-qwen-rvllm-nvfp4-spec | text | short_capital | Was ist die Hauptstadt von Frankreich? | Die Hauptstadt von Frankreich ist **Paris**. | 19 | 8 | — | 413.0 | — | 65.4 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | short_math | 1 + 1 = | 1 + 1 = 2 | 17 | 7 | — | 354.0 | — | 67.8 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | short_pangram | The quick brown fox jumps over the lazy | dog. | 20 | 2 | — | 269.1 | — | 81.7 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | medium_explain | Erkläre in drei Sätzen, was Photosynthese ist. | Die Photosynthese ist der biochemische Prozess, bei dem Pflanzen, Algen und einige Bakterien Lichtenergie nutzen, um aus Kohlendioxid und Wasser Glucose sowie Sauerstoff zu produzieren. Dieser Prozess | 26 | 80 | — | 2293.9 | — | 46.2 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | medium_code | Write a short Python function that reverses a string. | ```python def reverse_string(s: str) -> str:     return s[::-1] ``` | 23 | 22 | — | 768.9 | — | 58.5 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | medium_translation | Translate this sentence to German: 'The early bird catches the worm.' | Der frühe Vogel fängt den Wurm. | 26 | 10 | — | 520.4 | — | 69.2 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | long_summary_300 | Im Frühling blühen die Kirschbäume und die Tage werden länger. Die Vögel kehren  | Der Text beschreibt wiederholt die typischen Merkmale des Frühlings, wie das Blühen der Kirschbäume, das Längerwerden der Tage, die Rückkehr der Vögel und das Wachsen des Grases. | 466 | 47 | — | 4683.5 | — | 109.5 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | long_summary_800 | Quantum mechanics describes nature at the smallest scales of energy levels of at | Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles, whereas classical physics describes many aspects of nature at an ordinary (macroscopic) sca | 1004 | 52 | — | 11833.1 | — | 89.2 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | reasoning_chain | Anna hat 3 Äpfel. Sie gibt Tom 2 davon und kauft dann 5 weitere. Wie viele Äpfel | 6 Äpfel | 56 | 4 | — | 668.4 | — | 89.8 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | instruction | Liste drei Dinge auf, die man beachten sollte, wenn man eine Sauerteig-Brot back | Hier sind drei wichtige Punkte, die man beim Backen von Sauerteigbrot beachten sollte:  1. **Geduld bei der Gehzeit (Gare):** Sauerteigbrot braucht deutlich mehr Zeit als Hefebrot. Der Teig braucht of | 34 | 80 | — | 2257.1 | — | 50.5 | ✅ |
| mobile-qwen-rvllm-nvfp4-spec | text | qwen_repeat_160 | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta ga | mu alpha beta gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma | 1260 | 14 | — | 13912.2 | — | 91.6 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | short_capital | Was ist die Hauptstadt von Frankreich? | Die Hauptstadt von Frankreich ist **Paris**. | 19 | 8 | — | 4224.7 | — | 6.4 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | short_math | 1 + 1 = | 1 + 1 = 2 | 17 | 7 | — | 3684.4 | — | 6.5 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | short_pangram | The quick brown fox jumps over the lazy | dog. | 20 | 2 | — | 3307.7 | — | 6.7 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | medium_explain | Erkläre in drei Sätzen, was Photosynthese ist. | Photosynthese ist der biologische Prozess, bei dem Pflanzen, Algen und bestimmte Bakterien Lichtenergie in chemische Energie umwandeln. Dabei nutzen sie Sonnenlicht, um Kohlendioxid und Wasser in Gluk | 26 | 80 | — | 16965.5 | — | 6.2 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | medium_code | Write a short Python function that reverses a string. | ```python def reverse_string(s: str) -> str:     return s[::-1] ``` | 23 | 22 | — | 7044.8 | — | 6.4 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | medium_translation | Translate this sentence to German: 'The early bird catches the worm.' | Der frühe Vogel fängt den Wurm. | 26 | 10 | — | 5518.7 | — | 6.5 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | long_summary_300 | Im Frühling blühen die Kirschbäume und die Tage werden länger. Die Vögel kehren  | Der Text beschreibt wiederholt, dass im Frühling die Kirschbäume blühen, die Tage länger werden, die Vögel aus dem Süden zurückkehren und das Gras zu wachsen beginnt. | 466 | 41 | — | 8562.8 | — | 59.2 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | long_summary_800 | Quantum mechanics describes nature at the smallest scales of energy levels of at | Während die klassische Physik die Natur auf makroskopischer Ebene beschreibt, ist die Quantenmechanik erforderlich, um die Natur auf der kleinsten Ebene der Atome und subatomaren Teilchen zu beschreib | 1004 | 44 | — | 10217.3 | — | 102.6 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | reasoning_chain | Anna hat 3 Äpfel. Sie gibt Tom 2 davon und kauft dann 5 weitere. Wie viele Äpfel | 6 Äpfel | 56 | 4 | — | 8956.8 | — | 6.7 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | instruction | Liste drei Dinge auf, die man beachten sollte, wenn man eine Sauerteig-Brot back | Hier sind drei wesentliche Punkte, die beim Backen von Sauerteigbrot entscheidend sind:  1. **Die richtige Teigtemperatur und Umgebungstemperatur**      Sauerteig ist temperaturabhängig. Die ideale Te | 34 | 80 | — | 18139.1 | — | 6.3 | ✅ |
| mobile-qwen35-rvllm-nvfp4-spec | text | qwen_repeat_160 | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta ga | alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu | 1260 | 160 | — | 28441.0 | — | 49.9 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | short_capital | Was ist die Hauptstadt von Frankreich? | Die Hauptstadt von Frankreich ist **Paris**. | 20 | 9 | — | 1521.6 | — | 19.1 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | short_math | 1 + 1 = | 2 | 18 | 2 | — | 811.6 | — | 24.6 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | short_pangram | The quick brown fox jumps over the lazy | dog. | 21 | 3 | — | 1220.5 | — | 19.7 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | medium_explain | Erkläre in drei Sätzen, was Photosynthese ist. | Photosynthese ist der Prozess, bei dem Pflanzen, Algen und einige Bakterien Sonnenlicht nutzen, um Energie zu gewinnen. Dabei wandeln sie Wasser und Kohlendioxid in energiereiche Glukose (Zucker) um.  | 27 | 68 | — | 9875.2 | — | 9.6 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | medium_code | Write a short Python function that reverses a string. | The most efficient and "Pythonic" way to reverse a string is by using **slicing**.  ```python def reverse_string(text):     return text[::-1]  # Example usage: print(reverse_string("Hello World"))  #  | 23 | 80 | — | 7839.6 | — | 13.1 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | medium_translation | Translate this sentence to German: 'The early bird catches the worm.' | The direct translation of this idiom into German is:  **"Der frühe Vogel fängt den Wurm."** | 27 | 25 | — | 4113.3 | — | 12.6 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | long_summary_300 | Im Frühling blühen die Kirschbäume und die Tage werden länger. Die Vögel kehren  | Der Text beschreibt wiederholt die typischen Anzeichen des Frühlings, wie blühende Kirschbäume, längere Tage, die Rückkehr der Vögel und wachsendes Gras. | 480 | 43 | — | 6320.0 | — | 82.8 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | long_summary_800 | Quantum mechanics describes nature at the smallest scales of energy levels of at | Während die klassische Physik die makroskopische Welt beschreibt, ist die Quantenmechanik notwendig, um die Natur auf der Ebene von Atomen und subatomaren Teilchen zu erklären. | 1004 | 40 | — | 25419.7 | — | 41.1 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | reasoning_chain | Anna hat 3 Äpfel. Sie gibt Tom 2 davon und kauft dann 5 weitere. Wie viele Äpfel | 6 Äpfel | 58 | 5 | — | 862.3 | — | 73.1 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | instruction | Liste drei Dinge auf, die man beachten sollte, wenn man eine Sauerteig-Brot back | Hier sind drei der wichtigsten Dinge, die man beim Backen von Sauerteig-Brot beachten sollte:  1. **Die Aktivität des Starters (Vorteig):**    Ein Sauerteig-Brot gelingt nur, wenn der Starter „aktiv“  | 35 | 80 | — | 11204.3 | — | 10.3 | ✅ |
