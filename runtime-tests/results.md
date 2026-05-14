# rvllm-serve runtime tests

Generated: 2026-05-14 03:36:56 CEST

## Summary

| Profile | Pass | Fail | Mean prefill tok/s | Mean decode tok/s |
|---|---:|---:|---:|---:|
| mobile-e4b-rvllm | 13 | 1 | nan | 71.9 |

## Detailed records

| Profile | Kind | Label | Prompt (truncated) | Output (truncated) | prompt_tokens | completion_tokens | ttft ms | total ms | prefill tok/s | decode tok/s | ok |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|:-:|
| mobile-e4b-rvllm | text | short_capital | Was ist die Hauptstadt von Frankreich? | Die Hauptstadt von Frankreich ist **Paris**. | 16 | 8 | — | 1411.3 | — | 17.0 | ✅ |
| mobile-e4b-rvllm | text | short_math | 1 + 1 = | 1 + 1 = **2** | 14 | 8 | — | 1291.6 | — | 17.0 | ✅ |
| mobile-e4b-rvllm | text | short_pangram | The quick brown fox jumps over the lazy | The quick brown fox jumps over the lazy **dog**. | 17 | 11 | — | 1643.9 | — | 17.0 | ✅ |
| mobile-e4b-rvllm | text | medium_explain | Erkläre in drei Sätzen, was Photosynthese ist. | Photosynthese ist der lebenswichtige Prozess, bei dem Pflanzen mithilfe von Sonnenlicht, Wasser und Kohlendioxid ihre eigene Nahrungenergie herstellen. Dabei wandeln sie Lichtenergie in chemische Ener | 23 | 80 | — | 14333.4 | — | 7.2 | ✅ |
| mobile-e4b-rvllm | text | medium_code | Write a short Python function that reverses a string. | Here are a few ways to write a short Python function to reverse a string, depending on which method you prefer:  ### 1. Using Slicing (The most "Pythonic" way)  This is the shortest and most idiomatic | 19 | 80 | — | 11005.3 | — | 9.0 | ✅ |
| mobile-e4b-rvllm | text | medium_translation | Translate this sentence to German: 'The early bird catches the worm.' | The most common and idiomatic translation of "The early bird catches the worm" in German is:  **Der frühe Vogel fängt den Wurm.**  This is a direct and widely understood equivalent. | 23 | 42 | — | 5998.0 | — | 10.8 | ✅ |
| mobile-e4b-rvllm | text | long_summary_300 | Im Frühling blühen die Kirschbäume und die Tage werden länger. Die Vögel kehren  | Im Frühling erwacht die Natur mit blühenden Kirschbäumen, längeren Tagen, zurückkehämmenden Vögeln und wachsendem Gras. | 476 | 35 | — | 5507.9 | — | 92.8 | ✅ |
| mobile-e4b-rvllm | text | long_summary_800 | Quantum mechanics describes nature at the smallest scales of energy levels of at | Quantum mechanics is necessary to describe the behavior of nature at the atomic and subatomic level, where classical physics, while sufficient for macroscopic observations, fails. | 1000 | 30 | — | 5019.2 | — | 205.2 | ✅ |
| mobile-e4b-rvllm | text | reasoning_chain | Anna hat 3 Äpfel. Sie gibt Tom 2 davon und kauft dann 5 weitere. Wie viele Äpfel | Anna hat jetzt **6** Äpfel.  **Rechenweg:**  1 **Start:** Anna hat 3 Äpfel. 2 **Abgeben:** Sie gibt Tom 2 Äpfel weg: $3 - 2 = 1$ Apfel. 3 **Kaufen:** Sie kauft dann 5 weitere Äpfel: $1 + 5 = | 49 | 80 | — | 11137.6 | — | 11.6 | ✅ |
| mobile-e4b-rvllm | text | instruction | Liste drei Dinge auf, die man beachten sollte, wenn man eine Sauerteig-Brot back | Hier sind drei wichtige Dinge, die man beachten sollte, wenn man Sauerteigbrot backt:  1. **Die Aktivität und Pflege des Sauerteigstarters:** Der Sauerteigstarter ist das Herzstück des Brotes. Er muss | 31 | 80 | — | 11552.2 | — | 9.6 | ✅ |
| mobile-e4b-rvllm | vision | orange_ball | <image_url> Welche Farbe hat das Hauptobjekt im Bild? Antworte in einem Satz. | Das Hauptobjekt im Bild hat die Farbe Orange. | 283 | 11 | — | 2011.5 | — | 146.2 | ✅ |
| mobile-e4b-rvllm | vision | solid_red | <image_url> Was ist die dominante Farbe im Bild? | Die dominante Farbe im Bild ist **Rot**. | 273 | 9 | — | 1689.9 | — | 166.9 | ✅ |
| mobile-e4b-rvllm | vision | solid_green | <image_url> Welche Farbe siehst du? | Ich sehe **Grün**. | 272 | 6 | — | 1241.9 | — | 223.9 | ✅ |
| mobile-e4b-rvllm | audio | 440hz_1s | <audio_url> Transcribe verbatim. | `HTTP 500: {"error":{"message":"internal error: audio_url admission accepted (1 item) but B6 encoder forward is not yet w` | — | — | — | 2.0 | — | — | ❌ |
