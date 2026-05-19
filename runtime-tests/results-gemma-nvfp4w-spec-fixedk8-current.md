# rvllm-serve runtime tests

Generated: 2026-05-19 21:07:48 CEST

## Summary

| Profile | Pass | Fail | Mean prefill tok/s | Mean decode tok/s |
|---|---:|---:|---:|---:|
| mobile-31b-nvfp4w-rvllm-spec-k8 | 10 | 0 | nan | 31.4 |

## Detailed records

| Profile | Kind | Label | Prompt (truncated) | Output (truncated) | prompt_tokens | completion_tokens | ttft ms | total ms | prefill tok/s | decode tok/s | ok |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|:-:|
| mobile-31b-nvfp4w-rvllm-spec-k8 | text | short_capital | Was ist die Hauptstadt von Frankreich? | Die Hauptstadt von Frankreich ist **Paris**. | 20 | 9 | — | 1546.5 | — | 18.8 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec-k8 | text | short_math | 1 + 1 = | 2 | 18 | 2 | — | 823.8 | — | 24.3 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec-k8 | text | short_pangram | The quick brown fox jumps over the lazy | dog. | 21 | 3 | — | 1246.5 | — | 19.3 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec-k8 | text | medium_explain | Erkläre in drei Sätzen, was Photosynthese ist. | Photosynthese ist der Prozess, bei dem Pflanzen, Algen und einige Bakterien Sonnenlicht nutzen, um Energie zu gewinnen. Dabei wandeln sie Wasser und Kohlendioxid in energiereiche Glukose (Zucker) um.  | 27 | 68 | — | 10190.8 | — | 9.3 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec-k8 | text | medium_code | Write a short Python function that reverses a string. | The most efficient and "Pythonic" way to reverse a string is by using **slicing**.  ```python def reverse_string(text):     return text[::-1]  # Example usage: print(reverse_string("Hello World"))  #  | 23 | 80 | — | 7221.8 | — | 14.3 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec-k8 | text | medium_translation | Translate this sentence to German: 'The early bird catches the worm.' | The direct translation of this idiom into German is:  **"Der frühe Vogel fängt den Wurm."** | 27 | 25 | — | 4237.3 | — | 12.3 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec-k8 | text | long_summary_300 | Im Frühling blühen die Kirschbäume und die Tage werden länger. Die Vögel kehren  | Der Text beschreibt wiederholt die typischen Anzeichen des Frühlings, wie blühende Kirschbäume, längere Tage, die Rückkehr der Vögel und wachsendes Gras. | 480 | 43 | — | 5639.4 | — | 92.7 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec-k8 | text | long_summary_800 | Quantum mechanics describes nature at the smallest scales of energy levels of at | Während die klassische Physik die makroskopische Welt beschreibt, ist die Quantenmechanik notwendig, um die Natur auf der Ebene von Atomen und subatomaren Teilchen zu erklären. | 1004 | 40 | — | 25756.1 | — | 40.5 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec-k8 | text | reasoning_chain | Anna hat 3 Äpfel. Sie gibt Tom 2 davon und kauft dann 5 weitere. Wie viele Äpfel | 6 Äpfel | 58 | 5 | — | 877.2 | — | 71.8 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec-k8 | text | instruction | Liste drei Dinge auf, die man beachten sollte, wenn man eine Sauerteig-Brot back | Hier sind drei der wichtigsten Dinge, die man beim Backen von Sauerteig-Brot beachten sollte:  1. **Die Aktivität des Starters (Vorteig):**    Ein Sauerteig-Brot gelingt nur, wenn der Starter „aktiv“  | 35 | 80 | — | 11086.0 | — | 10.4 | ✅ |
