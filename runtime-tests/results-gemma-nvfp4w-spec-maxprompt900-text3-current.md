# rvllm-serve runtime tests

Generated: 2026-05-19 23:16:39 CEST

## Summary

| Profile | Pass | Fail | Mean prefill tok/s | Mean decode tok/s |
|---|---:|---:|---:|---:|
| mobile-31b-nvfp4w-rvllm-spec-maxprompt900 | 3 | 0 | nan | 55.2 |

## Detailed records

| Profile | Kind | Label | Prompt (truncated) | Output (truncated) | prompt_tokens | completion_tokens | ttft ms | total ms | prefill tok/s | decode tok/s | ok |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|:-:|
| mobile-31b-nvfp4w-rvllm-spec-maxprompt900 | text | long_summary_300 | Im Frühling blühen die Kirschbäume und die Tage werden länger. Die Vögel kehren  | Der Text beschreibt wiederholt die typischen Anzeichen des Frühlings, wie blühende Kirschbäume, längere Tage, die Rückkehr der Vögel und wachsendes Gras. | 480 | 43 | — | 5832.9 | — | 89.7 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec-maxprompt900 | text | long_summary_800 | Quantum mechanics describes nature at the smallest scales of energy levels of at | Während die klassische Physik die makroskopische Welt beschreibt, ist die Quantenmechanik notwendig, um die Natur auf der Ebene von Atomen und subatomaren Teilchen zu erklären. | 1004 | 40 | — | 16273.0 | — | 64.2 | ✅ |
| mobile-31b-nvfp4w-rvllm-spec-maxprompt900 | text | instruction | Liste drei Dinge auf, die man beachten sollte, wenn man eine Sauerteig-Brot back | Hier sind drei der wichtigsten Dinge, die man beim Backen von Sauerteig-Brot beachten sollte:  1. **Die Aktivität des Starters (Vorteig):**    Ein Sauerteig-Brot gelingt nur, wenn der Starter „aktiv“  | 35 | 80 | — | 9733.3 | — | 11.8 | ✅ |
