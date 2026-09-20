# MCGA-LM results (synthetic personas)


## Table 7 analogue - primary comparison

| System | WPM ↑ | SACT ↓ | IHR@3 ↑ | Hard halluc. ↓ | Soft halluc. | FAR ↓ | Abstain |
|---|---|---|---|---|---|---|---|
| MCGA-LM | 0.0 ± 0.2 | 2.4 ± 0.0 | 0 ± 1 | 4.2 ± 2.7 | 11.9 ± 5.8 | 99.9 ± 0.6 | 99.9 ± 0.6 |
| RAG-LLM | 0.1 ± 0.3 | 3.0 ± 0.0 | 0 ± 1 | 4.0 ± 2.6 | 11.8 ± 5.0 | 99.9 ± 0.6 | 99.9 ± 0.6 |
| M-LLM | 0.1 ± 0.3 | 2.4 ± 0.0 | 0 ± 1 | 10.4 ± 4.8 | 39.0 ± 7.3 | 99.9 ± 0.6 | 99.9 ± 0.6 |
| LLM-Only | 0.1 ± 0.3 | 2.3 ± 0.0 | 0 ± 1 | 10.4 ± 4.8 | 39.1 ± 7.4 | 99.9 ± 0.6 | 99.9 ± 0.6 |
| TouchChat | 12.5 ± 0.2 | 15.5 ± 0.5 | n/r | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | n/r |
| Static-WP-bigram | 6.0 ± 0.3 | n/r | n/r | 0.0 ± 0.0 | 0.0 ± 0.0 | 4.0 ± 0.0 | n/r |
| Adaptive-grid | 4.3 ± 0.2 | n/r | n/r | 0.0 ± 0.0 | 0.0 ± 0.0 | 4.0 ± 0.0 | n/r |
| Non-LLM-Intent | 0.3 ± 0.4 | 3.0 ± 0.0 | 25 ± 7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 99.0 ± 1.3 | n/r |

## Table 8 analogue - retrieval, fluency and calibration

| System | IHR@1 ↑ | IHR@5 ↑ | BLEU-4 ↑ | ROUGE-L ↑ | ECE ↓ | KSPC ↓ |
|---|---|---|---|---|---|---|
| MCGA-LM | 0 ± 1 | 0 ± 1 | n/r | n/r | 0.002 ± 0.005 | n/r |
| RAG-LLM | 0 ± 0 | 0 ± 1 | n/r | n/r | 0.002 ± 0.005 | n/r |
| M-LLM | 0 ± 0 | 0 ± 1 | n/r | n/r | 0.001 ± 0.006 | n/r |
| LLM-Only | 0 ± 0 | 0 ± 1 | n/r | n/r | 0.001 ± 0.006 | n/r |
| TouchChat | n/r | n/r | n/r | n/r | n/r | 0.58 ± 0.03 |
| Static-WP-bigram | n/r | n/r | n/r | n/r | n/r | 1.11 ± 0.03 |
| Adaptive-grid | n/r | n/r | n/r | n/r | n/r | 1.57 ± 0.06 |
| Non-LLM-Intent | 9 ± 5 | 36 ± 8 | 0.16 ± 0.34 | 0.28 ± 0.30 | n/r | n/r |

## Held-out split (context/interlocutor/topic absent from the graph)

| System | Hard halluc. ↓ | Soft halluc. | IHR@3 ↑ |
|---|---|---|---|
| MCGA-LM | 1.5 ± 2.1 | 4.0 ± 5.6 | 0 ± 1 |
| RAG-LLM | 1.2 ± 1.7 | 5.0 ± 5.3 | 0 ± 1 |
| M-LLM | 14.5 ± 5.5 | 71.5 ± 8.7 | 0 ± 1 |
| LLM-Only | 14.5 ± 5.5 | 71.6 ± 8.9 | 0 ± 1 |

## Statistics 

Repeated-measures ANOVA on SACT: F(1.0, 19.2) = 11583.95, p = 3.56e-28, partial eta^2 = 0.998 (Greenhouse-Geisser corrected)

| Contrast | Cohen's d_s | p (Holm) | |
|---|---|---|---|
| MCGA-LM vs RAG-LLM | -38.18 | 2.49e-28 | *** |
| MCGA-LM vs M-LLM | 0.05 | 6.66e-01 | n.s. |
| MCGA-LM vs LLM-Only | 0.34 | 1.79e-01 | n.s. |
| MCGA-LM vs TouchChat | -34.99 | 1.30e-27 | *** |
| MCGA-LM vs Non-LLM-Intent | -29.79 | 4.02e-28 | *** |
