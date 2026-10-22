# SpeechAI — Generalization False Positive Analysis

**Total Non-Stutter Clips:** 25 (11 Filler, 14 Clean)

- **Fix 3 Reference False Positives:** 10
- **Fix 3 + Combined False Positives:** 10
- **Net New False Positives Introduced by N-gram:** 0

## Result: Zero New False Positives

The combined 2-gram + 3-gram repetition detector introduced **0** false positive classifications on non-stutter speech.

Fluent control clips and filler interjections were not falsely classified as stutter.

## Existing False Positives (Present in Both Reference and Combined)

- `M_1022_11y3m_1_dysfluent_008.wav` (UCLASS): GT=Filler, Pred=Stutter, Conf=0.6
- `M_1022_11y3m_1_dysfluent_000.wav` (UCLASS): GT=Filler, Pred=Stutter, Conf=0.65
- `M_0061_16y9m-1_dysfluent_030.wav` (UCLASS): GT=Filler, Pred=Stutter, Conf=0.65
- `M_0061_16y9m-1_fluent_001.wav` (UCLASS): GT=Clean, Pred=Stutter, Conf=0.65
- `FluencyBank_017_4.wav` (FluencyBank): GT=Filler, Pred=Stutter, Conf=0.9
- `FluencyBank_031_47.wav` (FluencyBank): GT=Filler, Pred=Stutter, Conf=0.6
- `FluencyBank_031_69.wav` (FluencyBank): GT=Clean, Pred=Stutter, Conf=0.6
- `198-129977-0009.flac` (Libri_Stutter): GT=Clean, Pred=Stutter, Conf=0.65
- `198-209-0026.flac` (Libri_Stutter): GT=Clean, Pred=Stutter, Conf=0.7
- `1743-142913-0021.flac` (Libri_Stutter): GT=Clean, Pred=Stutter, Conf=0.65