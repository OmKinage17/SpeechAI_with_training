# SpeechAI — Generalization False Negative Analysis

**Total Stutter Clips:** 35

- **Fix 3 Reference False Negatives:** 15
- **Fix 3 + Combined False Negatives:** 13
- **True Positives Recovered by N-gram:** 2

## Recovered True Positives (Fixed by N-gram Repetition Detection)

### `1743-142914-0005.flac` (Libri_Stutter, Reader_1743)
- **Ground Truth:** Stutter
- **Fix 3 Ref Prediction:** Clean (conf: 0.0)
- **Combined Prediction:** Stutter (conf: 0.9)
  - **Matched Phrase:** `"he had ju"` (N=3, iter=2, span=0:5, conf=0.85)

### `1743-142913-0022.flac` (Libri_Stutter, Reader_1743)
- **Ground Truth:** Stutter
- **Fix 3 Ref Prediction:** Filler (conf: 0.0)
- **Combined Prediction:** Stutter (conf: 0.9)
  - **Matched Phrase:** `"the better"` (N=2, iter=2, span=4:7, conf=0.85)

## Remaining False Negatives in Combined Configuration

- `M_1022_11y3m_1_dysfluent_019.wav` (UCLASS, M_1022): GT=Stutter, Pred=Filler, Annotation={'sound_rep': 0, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 0}
- `M_0061_16y9m-1_dysfluent_081.wav` (UCLASS, M_0061): GT=Stutter, Pred=Filler, Annotation={'sound_rep': 0, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 0}
- `M_0061_16y9m-1_dysfluent_060.wav` (UCLASS, M_0061): GT=Stutter, Pred=Clean, Annotation={'sound_rep': 0, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 0}
- `M_0061_16y9m-1_dysfluent_046.wav` (UCLASS, M_0061): GT=Stutter, Pred=Filler, Annotation={'sound_rep': 1, 'word_rep': 1, 'prolongation': 1, 'block': 0, 'interjection': 0}
- `FluencyBank_017_15.wav` (FluencyBank, Ep_017): GT=Stutter, Pred=Clean, Annotation={'sound_rep': 0, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 1}
- `FluencyBank_017_42.wav` (FluencyBank, Ep_017): GT=Stutter, Pred=Clean, Annotation={'sound_rep': 0, 'word_rep': 0, 'prolongation': 0, 'block': 1, 'interjection': 0}
- `FluencyBank_017_34.wav` (FluencyBank, Ep_017): GT=Stutter, Pred=Clean, Annotation={'sound_rep': 0, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 0}
- `FluencyBank_017_52.wav` (FluencyBank, Ep_017): GT=Stutter, Pred=Clean, Annotation={'sound_rep': 0, 'word_rep': 0, 'prolongation': 0, 'block': 1, 'interjection': 0}
- `FluencyBank_017_2.wav` (FluencyBank, Ep_017): GT=Stutter, Pred=Clean, Annotation={'sound_rep': 1, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 1}
- `FluencyBank_031_14.wav` (FluencyBank, Ep_031): GT=Stutter, Pred=Clean, Annotation={'sound_rep': 0, 'word_rep': 0, 'prolongation': 0, 'block': 1, 'interjection': 0}
- `FluencyBank_031_9.wav` (FluencyBank, Ep_031): GT=Stutter, Pred=Clean, Annotation={'sound_rep': 1, 'word_rep': 0, 'prolongation': 0, 'block': 0, 'interjection': 0}
- `198-209-0025.flac` (Libri_Stutter, Reader_198): GT=Stutter, Pred=Filler, Annotation={'stutter_types': ['4', '0']}
- `1743-142912-0001.flac` (Libri_Stutter, Reader_1743): GT=Stutter, Pred=Filler, Annotation={'stutter_types': ['3', '0', '5']}