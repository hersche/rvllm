# rvllm-serve runtime tests

Generated: 2026-05-19 18:32:00 CEST

## Summary

| Profile | Pass | Fail | Mean prefill tok/s | Mean decode tok/s |
|---|---:|---:|---:|---:|
| mobile-31b-nvfp4w-rvllm-spec | 10 | 0 | nan | 29.7 |

## Detailed records

| Profile | Kind | Label | Prompt (truncated) | Output (truncated) | prompt_tokens | completion_tokens | ttft ms | total ms | prefill tok/s | decode tok/s | ok |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|:-:|
| mobile-31b-nvfp4w-rvllm-spec | text | short_capital | Was ist die Hauptstadt von Frankreich? | Die Hauptstadt von Frankreich ist **Paris**. | 20 | 9 | — | 1531.0 | — | 18.9 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | short_math | 1 + 1 = | 2 | 18 | 2 | — | 802.5 | — | 24.9 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | short_pangram | The quick brown fox jumps over the lazy | dog. | 21 | 3 | — | 1202.2 | — | 20.0 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | medium_explain | Erkläre in drei Sätzen, was Photosynthese ist. | Photosynthese ist der Prozess, bei dem Pflanzen, Algen und einige Bakterien Sonnenlicht nutzen, um Energie zu gewinnen. Dabei wandeln sie Wasser und Kohlendioxid in energiereiche Glukose (Zucker) um.  | 27 | 68 | — | 14954.5 | — | 6.4 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | medium_code | Write a short Python function that reverses a string. | The most efficient and "Pythonic" way to reverse a string is by using **slicing**.  ```python def reverse_string(text):     return text[::-1]  # Example usage: print(reverse_string("Hello World"))  #  | 23 | 80 | — | 6963.1 | — | 14.8 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | medium_translation | Translate this sentence to German: 'The early bird catches the worm.' | The direct translation of this idiom into German is:  **"Der frühe Vogel fängt den Wurm."** | 27 | 25 | — | 5537.4 | — | 9.4 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | long_summary_300 | Im Frühling blühen die Kirschbäume und die Tage werden länger. Die Vögel kehren  | Der Text beschreibt wiederholt die typischen Anzeichen des Frühlings, wie blühende Kirschbäume, längere Tage, die Rückkehr der Vögel und wachsendes Gras. | 480 | 43 | — | 6578.9 | — | 79.5 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | long_summary_800 | Quantum mechanics describes nature at the smallest scales of energy levels of at | Während die klassische Physik die makroskopische Welt beschreibt, ist die Quantenmechanik notwendig, um die Natur auf der Ebene von Atomen und subatomaren Teilchen zu erklären. | 1004 | 40 | — | 25526.0 | — | 40.9 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | reasoning_chain | Anna hat 3 Äpfel. Sie gibt Tom 2 davon und kauft dann 5 weitere. Wie viele Äpfel | 6 Äpfel | 58 | 5 | — | 852.8 | — | 73.9 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | instruction | Liste drei Dinge auf, die man beachten sollte, wenn man eine Sauerteig-Brot back | Hier sind drei der wichtigsten Dinge, die man beim Backen von Sauerteig-Brot beachten sollte:  1. **Die Aktivität des Starters (Vorteig):**    Ein Sauerteig-Brot gelingt nur, wenn der Starter „aktiv“  | 35 | 80 | — | 14238.5 | — | 8.1 | ✅ |
