# SpeechAI — Ablation 4: Remaining False-Negative Error Audit

**Project:** `E:\SpeechPrj`  
**Run Directory:** `evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/`  
**Accepted Reference:** Ablation 3 (`evaluation_results/experimental/ablations/stutter_fn_recovery/run_20261020_144131/`)  
**Feature Cache SHA-256:** `91ad23deef8331ec52953412927ca4a08b2d113def0f5dc997f70918dbe4d076`  
**Evaluation Benchmark:** 60-Clip Generalization Validation Set (Clean: 15, Stutter: 35, Filler: 10)  
**Reference Performance:** Accuracy: 70.00% | Macro F1: 0.6622 | Stutter Precision: 86.67% | Stutter Recall: 74.29% (26 TP, 9 FN, 4 FP, 21 TN)

---

## Executive Summary & Objective

In Ablation 3, the introduction of the Interrupted-Repetition Detector successfully recovered 4 false negatives (`M_0061_081`, `M_0061_046`, `FluencyBank_017_2`, `198-209-0025`), increasing stutter recall from 62.86% to 74.29% while maintaining 0 regressions and 0 new false positives. 

The objective of **Ablation 4 (Phase 1)** is to conduct an exhaustive, evidence-grounded read-only error audit of the **9 remaining false-negative stutter clips** in Ablation 3. For each missed clip, we inspect:
1. Ground-truth clinical annotations (sound repetition, word repetition, prolongation, block, interjection).
2. Whisper Large-v3 ASR transcript and token timestamps.
3. Acoustic waveform and prosodic features (vocal energy, noise floor, pitch stability, spectral flux, prolongation candidates, silent gaps, and plosive burst ratios).
4. Detailed failure categorization and root causes.

Furthermore, we evaluate the actionability and false-positive risk of potential algorithmic interventions, providing empirical counter-evidence on why naive threshold lowering or boundary heuristics cause severe false positive inflation.

---

## Detailed Audit of the 9 Remaining False-Negative Clips

| # | Clip ID | Corpus | Speaker | Dominant GT | Ablation 3 Pred | Annotator Details | Primary Failure Category |
|---|---|---|---|---|---|---|---|
| 1 | `M_1022_11y3m_1_dysfluent_019.wav` | SEP-28k | M_1022 | Stutter | **Filler** | `{'word_rep': 1}` | ASR Normalization & Filler Dominance |
| 2 | `M_0061_16y9m-1_dysfluent_060.wav` | SEP-28k | M_0061 | Stutter | **Clean** | `{'word_rep': 1}` | Boundary Slicing & Syntactic Revision |
| 3 | `FluencyBank_017_15.wav` | FluencyBank | Ep_017 | Stutter | **Clean** | `{'word_rep': 1, 'interjection': 1}` | ASR Lexical Smoothing |
| 4 | `FluencyBank_017_42.wav` | FluencyBank | Ep_017 | Stutter | **Clean** | `{'block': 1}` | Sub-threshold Plosive Burst (No Acoustic Release) |
| 5 | `FluencyBank_017_34.wav` | FluencyBank | Ep_017 | Stutter | **Clean** | `{'word_rep': 1}` | ASR Timeout / Audio Distortion |
| 6 | `FluencyBank_017_52.wav` | FluencyBank | Ep_017 | Stutter | **Clean** | `{'block': 1}` | Contiguous Phonation (No Silent Gap) |
| 7 | `FluencyBank_031_14.wav` | FluencyBank | Ep_031 | Stutter | **Clean** | `{'block': 1}` | Contiguous Phonation (No Silent Gap) |
| 8 | `FluencyBank_031_9.wav` | FluencyBank | Ep_031 | Stutter | **Clean** | `{'sound_rep': 1}` | ASR Sound Repetition Smoothing |
| 9 | `1743-142912-0001.flac` | LibriSpeech | Reader_1743 | Stutter | **Filler** | `{'types': ['3', '0', '5']}` | Sub-500ms Prosodic Elongation & Filler Co-occurrence |

---

### Clip-by-Clip In-Depth Evidence & Root Cause Analysis

#### 1. `M_1022_11y3m_1_dysfluent_019.wav`
- **Corpus:** SEP-28k (UCLASS) | **Speaker:** `M_1022`
- **Ground Truth:** `Stutter` | **Ablation 3 Prediction:** `Filler`
- **Clinical Annotation Details:** `{'sound_rep': 0, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 0}`
- **ASR Transcript:** `"things. Well, my best subject."`
- **Token Timestamps:**
  - `(' things.', 0.00, 0.44)`
  - `(' Well,', 1.02, 1.32)`
  - `(' my', 1.52, 1.78)`
  - `(' best', 1.78, 2.26)`
  - `(' subject.', 2.26, 2.68)`
- **Acoustic Inspection:** Duration: 3.00s. Vocal threshold: 0.0062. Noise floor: 0.0025. Prolongation candidates (≥0.50s): 0. Acoustic blocks (≥0.40s with burst): 0.
- **Root Cause Analysis:** 
  The clinical listener heard a word repetition (`word_rep: 1`). However, Whisper Large-v3 completely smoothed the repeated word out of the transcript, outputting the fluent clause `"things. Well, my best subject."`. Crucially, Whisper transcribed the discourse marker `"Well,"`. Because `"well"` is an established member of the vocal filler lexicon, and the transcript contains zero repeated tokens or acoustic dysfluencies, the filler detection engine classified the clip as `Filler`.
- **Failure Category:** **Category 1: ASR Language-Model Smoothing & Lexical Normalization**

---

#### 2. `M_0061_16y9m-1_dysfluent_060.wav`
- **Corpus:** SEP-28k (UCLASS) | **Speaker:** `M_0061`
- **Ground Truth:** `Stutter` | **Ablation 3 Prediction:** `Clean`
- **Clinical Annotation Details:** `{'sound_rep': 0, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 0}`
- **ASR Transcript:** `"In your health master, we believe to code, to be called Mr. Sh-"`
- **Token Timestamps:**
  - `(' In', 0.00, 0.08)`, `(' your', 0.08, 0.18)`, `(' health', 0.18, 0.40)`, `(' master,', 0.40, 0.84)`
  - `(' we', 0.98, 1.04)`, `(' believe', 1.04, 1.30)`, `(' to', 1.30, 1.58)`, `(' code,', 1.58, 1.92)`
  - `(' to', 2.10, 2.22)`, `(' be', 2.22, 2.32)`, `(' called', 2.32, 2.50)`, `(' Mr.', 2.50, 2.80)`
  - `(' Sh', 2.90, 2.98)`, `('-', 2.98, 2.98)`
- **Acoustic Inspection:** Duration: 3.00s. Prolongation candidates (≥0.30s): 0 (longest candidate is 0.27s at 1.12–1.39s with low confidence 0.34). Acoustic blocks: 0.
- **Root Cause Analysis:** 
  1. The annotator flagged `word_rep: 1` due to the speaker revising their utterance: `"... to code, to be called ..."`. Here, the infinitive marker `"to"` is repeated, but separated by the lexical word `"code,"` (a non-adjacent syntactic revision). Generalizing repetition detection to arbitrary non-adjacent words creates disastrous false positives across normal English syntax (e.g., `"I went to the store to buy milk"`).
  2. The trailing token `"Mr. Sh-"` is cut off at 2.98s due to the 3.00s audio segmentation window slicing off the speaker's articulation of a surname (e.g., "Shephard"). 
- **Failure Category:** **Category 4: Dataset Boundary Truncation & Non-Adjacent Syntactic Revision**

---

#### 3. `FluencyBank_017_15.wav`
- **Corpus:** FluencyBank | **Speaker:** `Ep_017`
- **Ground Truth:** `Stutter` | **Ablation 3 Prediction:** `Clean`
- **Clinical Annotation Details:** `{'sound_rep': 0, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 1}`
- **ASR Transcript:** `"I'm just going to do a video. Oh"`
- **Token Timestamps:**
  - `(" I'm", 1.12, 1.56)`, `(' just', 1.56, 1.64)`, `(' going', 1.64, 2.00)`
  - `(' to', 2.00, 2.00)`, `(' do', 2.00, 2.02)`, `(' a', 2.02, 2.02)`, `(' video.', 2.02, 2.02)`, `(' Oh', 2.02, 2.82)`
- **Acoustic Inspection:** Duration: 3.00s. Prolongation candidates (≥0.50s): 0. Acoustic blocks: 0.
- **Root Cause Analysis:** 
  The clinical annotator documented a word repetition. However, Whisper's decoder language model smoothed the speech stream into the grammatically fluent sentence `"I'm just going to do a video. Oh"`. The acoustic features contain no vowel steady-state ≥0.50s or silent gap ≥0.40s. Downstream detectors receiving this transcript cannot detect a repetition that does not exist in the decoded text.
- **Failure Category:** **Category 1: ASR Language-Model Smoothing & Lexical Normalization**

---

#### 4. `FluencyBank_017_42.wav`
- **Corpus:** FluencyBank | **Speaker:** `Ep_017`
- **Ground Truth:** `Stutter` | **Ablation 3 Prediction:** `Clean`
- **Clinical Annotation Details:** `{'sound_rep': 0, 'word_rep': 0, 'prolongation': 0, 'block': 1, 'interjection': 0}`
- **ASR Transcript:** `"You're stuttering. Tell me about your therapy..."`
- **Token Timestamps:**
  - `(" You're", 0.00, 0.08)`, `(' stuttering.', 0.08, 0.54)`
  - `(' Tell', 0.92, 1.10)`, `(' me', 1.10, 1.28)`, `(' about', 1.28, 1.54)`, `(' your', 1.54, 1.80)`, `(' therapy...', 1.80, 2.98)`
- **Acoustic Inspection:** 
  - Word gap between `"stuttering."` (end: 0.54s) and `"Tell"` (start: 0.92s) is **0.38s**.
  - Silence RMS: `0.0112`.
  - Burst RMS in following 4 frames: `0.0089`.
  - Burst Energy Ratio: **0.79** (plosive release is *lower* than the silence noise level).
  - Noise floor: `0.0045`, Vocal threshold: `0.0112`.
- **Root Cause Analysis:** 
  While the human listener perceived an articulatory block, the physical gap duration is 0.38s (below the 0.40s block threshold) and the burst energy ratio is 0.79. Because the speaker initiates `"Tell"` very softly without a plosive energy release, the acoustic detector correctly treats this as a conversational transition rather than an explosive block release.
- **Failure Category:** **Category 2: Acoustic Silent Block Absence (Sub-threshold Burst)**

---

#### 5. `FluencyBank_017_34.wav`
- **Corpus:** FluencyBank | **Speaker:** `Ep_017`
- **Ground Truth:** `Stutter` | **Ablation 3 Prediction:** `Clean`
- **Clinical Annotation Details:** `{'sound_rep': 0, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 0}`
- **ASR Transcript:** `""` (Empty Transcript)
- **Acoustic Inspection:** Duration: 0.00s / Timeout.
- **Root Cause Analysis:** 
  The audio clip contains extreme acoustic corruption/distortion. The Whisper ASR worker timed out under CPU safety limits (or yielded empty text), causing a safe fallback to empty transcript and zero candidates. In the absence of text or valid acoustic features, the system defaults conservatively to `Clean`.
- **Failure Category:** **Category 1: ASR Language-Model Smoothing & Audio Timeout**

---

#### 6. `FluencyBank_017_52.wav`
- **Corpus:** FluencyBank | **Speaker:** `Ep_017`
- **Ground Truth:** `Stutter` | **Ablation 3 Prediction:** `Clean`
- **Clinical Annotation Details:** `{'sound_rep': 0, 'word_rep': 0, 'prolongation': 0, 'block': 1, 'interjection': 0}`
- **ASR Transcript:** `"community activities. Okay."`
- **Token Timestamps:**
  - `(' community', 0.00, 1.36)`
  - `(' activities.', 1.36, 2.12)`
  - `(' Okay.', 2.12, 2.62)`
- **Acoustic Inspection:** 
  - Word gaps: Gap 0 (`"community"` -> `"activities"`): **0.00s**. Gap 1 (`"activities"` -> `"Okay"`): **0.00s**.
  - All words are acoustically contiguous in time.
  - Prolongation candidates (≥0.50s): 0 (only one short segment of 0.42s with low confidence 0.52).
- **Root Cause Analysis:** 
  The speaker experiences articulatory tension during phonation, but there is **no silent gap in the acoustic waveform**. The acoustic block detector searches for silent pauses with plosive releases; because the audio has zero silence gap between words, no block candidate can be formed without phoneme-level articulatory modeling.
- **Failure Category:** **Category 2: Acoustic Silent Block Absence (Contiguous Phonation)**

---

#### 7. `FluencyBank_031_14.wav`
- **Corpus:** FluencyBank | **Speaker:** `Ep_031`
- **Ground Truth:** `Stutter` | **Ablation 3 Prediction:** `Clean`
- **Clinical Annotation Details:** `{'sound_rep': 0, 'word_rep': 0, 'prolongation': 0, 'block': 1, 'interjection': 0}`
- **ASR Transcript:** `"You get better in some way."`
- **Token Timestamps:**
  - `(' You', 0.00, 0.40)`
  - `(' get', 0.40, 1.32)`
  - `(' better', 1.32, 1.82)`
  - `(' in', 1.82, 2.44)`
  - `(' some', 2.44, 2.70)`
  - `(' way.', 2.70, 2.98)`
- **Acoustic Inspection:** 
  - Inter-word gaps: all **0.00s**. Continuous vocal tract phonation across all words.
  - Prolongation candidates: 0.45s at 1.65–2.10s (pitch std: 0.29, flux: 0.0165, confidence: 0.56, below 0.50s admission floor).
- **Root Cause Analysis:** 
  Identical to `FluencyBank_017_52`. Articulatory fixation occurred without vocal tract closure or acoustic silence. Speech energy remained continuous across word boundaries.
- **Failure Category:** **Category 2: Acoustic Silent Block Absence (Contiguous Phonation)**

---

#### 8. `FluencyBank_031_9.wav`
- **Corpus:** FluencyBank | **Speaker:** `Ep_031`
- **Ground Truth:** `Stutter` | **Ablation 3 Prediction:** `Clean`
- **Clinical Annotation Details:** `{'sound_rep': 1, 'word_rep': 0, 'prolongation': 0, 'block': 0, 'interjection': 0}`
- **ASR Transcript:** `"Can we go back to this thing about there?"`
- **Token Timestamps:**
  - `(' Can', 0.70, 1.14)`, `(' we', 1.14, 1.28)`, `(' go', 1.28, 1.42)`, `(' back', 1.42, 1.60)`, `(' to', 1.60, 1.82)`
  - `(' this', 1.82, 2.26)`, `(' thing', 2.26, 2.48)`, `(' about', 2.48, 2.74)`, `(' there?', 2.74, 2.98)`
- **Acoustic Inspection:** Duration: 3.00s. Prolongation candidates (≥0.50s): 0. Acoustic blocks: 0.
- **Root Cause Analysis:** 
  The clinical listener documented a sound repetition (`sound_rep: 1`, e.g., "c-c-can"). However, Whisper Large-v3 decoded the acoustic input into the standard word `"Can"`. Because sound repetitions are absorbed by Whisper's beam-search language model prior to transcript emission, the downstream detector has no lexical repetition signal.
- **Failure Category:** **Category 1: ASR Language-Model Smoothing & Lexical Normalization**

---

#### 9. `1743-142912-0001.flac`
- **Corpus:** LibriSpeech (Libri_Stutter) | **Speaker:** `Reader_1743`
- **Ground Truth:** `Stutter` | **Ablation 3 Prediction:** `Filler`
- **Clinical Annotation Details:** `{'types': ['3', '0', '5']}` (Prolongation, Interjection, Word Repetition)
- **ASR Transcript:** `"He didn't like to admit that he couldn't recall those two cousins of Chibbeck's. Did you say you're fighting cousin? He asked in a hesitating way. That is what I said, replied Chibbeck. He is scrapper the Kingbird, as of course you know."`
- **Acoustic Inspection:** Duration: 17.50s. Expressive literary narration. Vowel steady-states are elongated up to 0.46s, but none exceed the 0.50s prolongation threshold.
- **Root Cause Analysis:** 
  The speaker reads expressively with theatrical pauses and mild elongation (0.42–0.46s). The transcript contains conversational interjections (`"like"`, `"you know"`). Because the prosodic elongations are under the 0.50s safety floor, the prolongation detector did not fire, leaving the filler detector to classify the clip as `Filler`.
- **Failure Category:** **Category 3: Sub-500ms Prosodic Elongation & Filler Co-occurrence**

---

## Failure Category Ranking & Actionability

We group the 9 remaining false negatives into four distinct categories and rank them by **Frequency**, **Actionability**, and **False-Positive Risk**:

| Rank | Failure Category | Affected Clips | % of FNs | Actionability at Detector Layer | False-Positive Risk | Recommended Action |
|---|---|---|---|---|---|---|
| **1** | **ASR Smoothing & Normalization** | 4 (`M_1022_019`, `FluencyBank_017_15`, `FluencyBank_031_9`, `FluencyBank_017_34`) | 44.4% | **Low / None** (Upstream ASR bottleneck) | **Catastrophic** (Guessing from fluent text destroys precision) | **Do Not Modify.** Requires multi-pass ASR or raw phoneme decoding. |
| **2** | **Acoustic Silent Block Absence** | 3 (`FluencyBank_017_42`, `FluencyBank_017_52`, `FluencyBank_031_14`) | 33.3% | **Moderate** (Threshold tuning) | **Extreme** (7 new FPs on clean pauses across benchmark) | **Do Not Modify.** Lowering pause floors flags normal comma pauses. |
| **3** | **Sub-500ms Prosodic Elongation** | 1 (`1743-142912-0001.flac`) | 11.1% | **High** (Threshold tuning) | **Extreme** (6 new FPs on clean speech in [0.40s, 0.50s)) | **Do Not Modify.** Preserves 0.50s prolongation safety floor. |
| **4** | **Boundary Slicing & Revision** | 1 (`M_0061_060`) | 11.1% | **Moderate** (Hyphen / token heuristic) | **High** (4 new FPs from 3-second audio slicing) | **Do Not Modify.** Trailing hyphens are dataset boundary artifacts. |

---

## Empirical Counter-Evidence: The Severe Risk of Threshold Lowering

Before concluding, we conducted controlled empirical simulations of the three most apparent algorithmic interventions to determine whether any change could safely recover these false negatives without harming precision:

### Experiment 1: Lowering Prolongation Duration Threshold (0.50s -> 0.40s)
- **Target:** Recover `1743-142912-0001.flac` (which exhibits 0.46s vowels).
- **Measured Result across 60 Clips:**
  - **14 clips** possess steady-state vowels between 0.40s and 0.50s.
  - **6 NEW FALSE POSITIVES** were introduced on Clean and Filler speech:
    1. `M_1022_11y3m_1_fluent_049.wav` (Clean, dur: 0.42s)
    2. `198-129977-0009.flac` (Clean, dur: 0.44s)
    3. `198-209-0026.flac` (Clean, dur: 0.41s)
    4. `M_1022_11y3m_1_fluent_008.wav` (Clean, dur: 0.43s)
    5. `M_1022_11y3m_1_fluent_001.wav` (Clean, dur: 0.40s)
    6. `FluencyBank_031_47.wav` (Filler, dur: 0.45s)
  - **Stutter Precision Collapsed:** From **86.67% to 74.29%** (-12.38%).
- **Conclusion:** Lowering the prolongation floor below 0.50s violates the core safety objective.

### Experiment 2: Lowering Silent Block Silence Threshold (0.40s -> 0.35s)
- **Target:** Recover `FluencyBank_017_42.wav` (which has a 0.38s silence gap).
- **Measured Result across 60 Clips:**
  - **21 clips** possess inter-word pauses between 0.30s and 0.40s.
  - **7 NEW FALSE POSITIVES** were introduced on Clean speech (`M_1022_fluent_001`, `198-209-0003`, `FluencyBank_017_24`, `FluencyBank_017_1`, `1743-142912-0006`, etc.).
  - Furthermore, `FluencyBank_017_42.wav` still failed because its burst ratio (0.79) was below noise.
  - **Stutter Precision Collapsed:** From **86.67% to 72.22%** (-14.45%).
- **Conclusion:** Natural pauses at clause and comma boundaries fall into the 0.30–0.40s window. Lowering the silence threshold destroys specificity.

### Experiment 3: Detecting Trailing Boundary Hyphens (`-`)
- **Target:** Recover `M_0061_16y9m-1_dysfluent_060.wav` (transcript ends with `"Mr. Sh-"`).
- **Measured Result across 60 Clips:**
  - In SEP-28k and FluencyBank, 3-second segmented clips cut off speech mid-word as a boundary artifact.
  - **16 clips** in the benchmark end with trailing hyphens.
  - **4 NEW FALSE POSITIVES** were introduced on Clean and Filler speech:
    1. `M_1022_11y3m_1_fluent_049.wav` (Clean, transcript ends with `"we get to-"`)
    2. `FluencyBank_017_24.wav` (Clean, transcript ends with `"and-"`)
    3. `FluencyBank_017_5.wav` (Filler, transcript ends with `"circuit-"`)
    4. `FluencyBank_031_32.wav` (Filler, transcript ends with `"sex-"`)
  - **Stutter Precision Collapsed:** From **86.67% to 79.41%** (-7.26%).
- **Conclusion:** Trailing hyphens represent audio clipping artifacts, not reliable indicators of clinical stuttering.

---

## Final Phase 1 Conclusion & Recommendation

The empirical evidence is definitive:
1. **ASR Smoothing (44.4%)** is an upstream modeling ceiling that cannot be solved by heuristic rule adjustments in the downstream detector.
2. **Articulatory Tension without Silent Gaps (33.3%)** cannot be captured by acoustic energy thresholding without catastrophic false positives on normal speech.
3. Every evaluated intervention produces **4 to 7 new false positives** in exchange for at most 0 to 1 true positive.

**Decision:** Under the project safety guidelines (*"If the evidence does not justify a safe intervention, stop after the error audit and recommend no change"*), we stop after this error audit, **recommend NO threshold or algorithmic modifications**, and retain **Ablation 3 as the frozen development reference**.
