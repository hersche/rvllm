# rvllm-serve runtime tests

Generated: 2026-05-19 13:01:10 CEST

## Summary

| Profile | Pass | Fail | Mean prefill tok/s | Mean decode tok/s |
|---|---:|---:|---:|---:|
| mobile-31b-nvfp4w-rvllm-spec | 10 | 0 | nan | 10.6 |

## Detailed records

| Profile | Kind | Label | Prompt (truncated) | Output (truncated) | prompt_tokens | completion_tokens | ttft ms | total ms | prefill tok/s | decode tok/s | ok |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|:-:|
| mobile-31b-nvfp4w-rvllm-spec | text | short_capital | Was ist die Hauptstadt von Frankreich? | Die Hauptstadt von Frankreich ist **Paris**. | 20 | 9 | — | 2602.7 | — | 11.1 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | short_math | 1 + 1 = | 2 | 18 | 2 | — | 1814.9 | — | 11.0 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | short_pangram | The quick brown fox jumps over the lazy | dog. | 21 | 3 | — | 2370.1 | — | 10.1 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | medium_explain | Erkläre in drei Sätzen, was Photosynthese ist. | Photosynthese ist der Prozess, bei dem Pflanzen aus Sonnenlicht, Wasser und Kohlendioxid energiereiche Glukose (Zucker) herstellen. Dabei nutzen sie das grüne Blattpigment Chlorophyll, um die Lichtene | 27 | 80 | — | 13225.4 | — | 8.1 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | medium_code | Write a short Python function that reverses a string. | The most efficient and "Pythonic" way to reverse a string is by using **slicing**.  ```python def reverse_string(text):     return text[::-1]  # Example usage: print(reverse_string("Hello World"))  #  | 23 | 80 | — | 9800.5 | — | 10.5 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | medium_translation | Translate this sentence to German: 'The early bird catches the worm.' | The direct translation of this idiom into German is:  **"Der frühe Vogel fängt den Wurm."** | 27 | 25 | — | 5845.8 | — | 8.9 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | long_summary_300 | Im Frühling blühen die Kirschbäume und die Tage werden länger. Die Vögel kehren  | Der Text beschreibt wiederholt die typischen Anzeichen des Frühlings, wie blühende Kirschbäume, längere Tage, die Rückkehr der Vögel und wachsendes Gras. | 480 | 43 | — | 41958.6 | — | 12.5 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | long_summary_800 | Quantum mechanics describes nature at the smallest scales of energy levels of at | Der Text erklärt, dass die Quantenmechanik die Natur auf atomarer und subatomarer Ebene beschreibt, während die klassische Physik für makroskopische Maßstäbe geeignet ist, aber bei kleinsten Skalen ve | 1004 | 51 | — | 86411.9 | — | 12.2 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | reasoning_chain | Anna hat 3 Äpfel. Sie gibt Tom 2 davon und kauft dann 5 weitere. Wie viele Äpfel | 6 Äpfel | 58 | 5 | — | 4959.5 | — | 12.7 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec | text | instruction | Liste drei Dinge auf, die man beachten sollte, wenn man eine Sauerteig-Brot back | Hier sind drei der wichtigsten Dinge, die du beim Backen von Sauerteig-Brot beachten solltest:  1. **Die Aktivität des Starters (Vorteig):** Ein Sauerteig-Brot gelingt nur, wenn der Starter „aktiv“ is | 35 | 80 | — | 12710.0 | — | 9.0 | ✅ |
