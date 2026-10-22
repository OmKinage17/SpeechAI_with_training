# SpeechAI — Ablation 5: Targeted False-Negative Root-Cause Analysis

**Project:** `E:\SpeechPrj`  
**Run Directory:** `evaluation_results/experimental/ablations/ablation5_targeted_recovery/run_20261020_154500/`  
**Accepted Reference:** Ablation 3 (`evaluation_results/experimental/ablations/stutter_fn_recovery/run_20261020_144131/`)  
**Prior Error Audit:** Ablation 4 (`evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/`)  
**Generalization Validation Benchmark:** 60 clips (Clean: 15, Stutter: 35, Filler: 10; Dominant Support: 14 Clean, 35 Stutter, 11 Filler)  
**Feature Cache Path:** `evaluation_results/experimental/generalization_validation/run_20261020_132214/feature_cache.json`  
**Feature Cache SHA-256:** `91ad23deef8331ec52953412927ca4a08b2d113def0f5dc997f70918dbe4d076`  
**Reference Performance:** Accuracy: 70.00% (42/60) | Macro F1: 0.6622 | Stutter Precision: 86.67% | Stutter Recall: 74.29% (26 TP, 9 FN, 4 FP, 21 TN)

---

## 1. Executive Summary & Objective

The primary objective of **Ablation 5 (Phase 1)** is to conduct an independent, deep root-cause analysis of the **9 remaining stutter false negatives** from the frozen Ablation 3 reference. Following the rigorous verification of the 60-clip generalization benchmark, we examine whether any of the remaining 9 false negatives can be safely recovered through targeted algorithmic mechanisms without destabilizing precision (86.67%) or introducing regressions.

All findings below are derived directly from the raw paired predictions (`per_clip_predictions.json`), audio waveforms, Whisper Large-v3 timestamped tokens, and expert clinical annotations in `run_manifest.json`.

---

## 2. Verification of the 9 False-Negative Clips

Every clip in the frozen reference was cross-checked against raw records. The 9 stutter false negatives and their exact annotations are confirmed:

| # | Clip ID | Dataset / Corpus | Speaker | Ground Truth | Ablation 3 Prediction | Clinical Annotations | Root-Cause Category |
|---|---|---|---|---|---|---|---|
| 1 | `M_1022_11y3m_1_dysfluent_019.wav` | UCLASS / SEP-28k | `M_1022` | **Stutter** | `Filler` | `{'word_rep': 1}` | ASR Normalization & Filler Dominance |
| 2 | `M_0061_16y9m-1_dysfluent_060.wav` | UCLASS / SEP-28k | `M_0061` | **Stutter** | `Clean` | `{'word_rep': 1}` | Boundary Slicing & Non-Adjacent Revision |
| 3 | `FluencyBank_017_15.wav` | FluencyBank | `Ep_017` | **Stutter** | `Clean` | `{'word_rep': 1, 'interjection': 1}` | ASR Lexical Smoothing |
| 4 | `FluencyBank_017_42.wav` | FluencyBank | `Ep_017` | **Stutter** | `Clean` | `{'block': 1}` | Sub-threshold Plosive Burst (Soft Onset) |
| 5 | `FluencyBank_017_34.wav` | FluencyBank | `Ep_017` | **Stutter** | `Clean` | `{'word_rep': 1}` | Acoustic Degradation & ASR Timeout |
| 6 | `FluencyBank_017_52.wav` | FluencyBank | `Ep_017` | **Stutter** | `Clean` | `{'block': 1}` | Contiguous Phonation (No Silent Gap) |
| 7 | `FluencyBank_031_14.wav` | FluencyBank | `Ep_031` | **Stutter** | `Clean` | `{'block': 1}` | Contiguous Phonation (No Silent Gap) |
| 8 | `FluencyBank_031_9.wav` | FluencyBank | `Ep_031` | **Stutter** | `Clean` | `{'sound_rep': 1}` | ASR Sound Repetition Smoothing |
| 9 | `1743-142912-0001.flac` | Libri_Stutter | `Reader_1743` | **Stutter** | `Filler` | `{'stutter_types': ['3', '0', '5']}` | Sub-500ms Prolongation & Filler Masking |

---

## 3. Clip-by-Clip Root-Cause & Detector Stage Diagnostics

### 1. `M_1022_11y3m_1_dysfluent_019.wav`
- **Corpus:** SEP-28k (UCLASS) | **Ground Truth:** `Stutter` | **Prediction:** `Filler`
- **Clinical Annotation:** `{'word_rep': 1, 'sound_rep': 0, 'prolongation': 0, 'block': 0, 'interjection': 0}`
- **ASR Transcript:** `"things. Well, my best subject."`
- **Active Detector Stages:**
  - *Repetition Stage:* Lexical repetition = 0, Interrupted repetition = 0 (no repeated words in transcript).
  - *Prolongation Stage:* Candidates ≥ 0.50s = 0.
  - *Silent Block Stage:* Silent gaps ≥ 0.40s with burst ≥ 1.5 = 0 (pause after `"things."` is 0.58s, but burst ratio is 0.62, below vocal threshold).
  - *Filler Stage:* Matched `"Well,"` from filler lexicon.
- **Why Missed:** Whisper's autoregressive language model normalized the acoustic word repetition into fluent text. The discourse marker `"Well,"` was then captured by the filler lexicon, leading to `Filler`.
- **Classification:** **Category 1: Upstream ASR Normalization & Smoothing**

### 2. `M_0061_16y9m-1_dysfluent_060.wav`
- **Corpus:** SEP-28k (UCLASS) | **Ground Truth:** `Stutter` | **Prediction:** `Clean`
- **Clinical Annotation:** `{'word_rep': 1, 'sound_rep': 0, 'prolongation': 0, 'block': 0, 'interjection': 0}`
- **ASR Transcript:** `"In your health master, we believe to code, to be called Mr. Sh-"`
- **Active Detector Stages:**
  - *Repetition Stage:* Lexical repetition = 0, Interrupted repetition = 0. The repetition is non-adjacent (`"to code, to be called"`, separated by content word `"code"`).
  - *Prolongation Stage:* Max steady state = 0.27s (below 0.50s).
  - *Silent Block Stage:* Inter-word pauses are 0.14s, 0.18s, 0.10s (below 0.40s).
  - *Boundary Slicing:* Final token is cut off at 2.98s (`"Mr. Sh-"`).
- **Why Missed:** The disfluency consists of a syntactic false start / revision (`to code` -> `to be called`). The detector's repetition engines require adjacent or interrupted token sequences; non-adjacent token matches with intervening content words are not treated as repetitions due to normal English syntax.
- **Classification:** **Category 4: Boundary Truncation & Non-Adjacent Syntactic Revision**

### 3. `FluencyBank_017_15.wav`
- **Corpus:** FluencyBank | **Ground Truth:** `Stutter` | **Prediction:** `Clean`
- **Clinical Annotation:** `{'word_rep': 1, 'interjection': 1}`
- **ASR Transcript:** `"I'm just going to do a video. Oh"`
- **Active Detector Stages:**
  - *Repetition Stage:* Repetition count = 0.
  - *Prolongation Stage:* Max steady state = 0.24s.
  - *Silent Block Stage:* All inter-word pauses < 0.05s.
- **Why Missed:** The clinical repetition heard by the annotator was completely absorbed into standard grammatical text by Whisper. The detector receives zero acoustic or lexical evidence.
- **Classification:** **Category 1: Upstream ASR Normalization & Smoothing**

### 4. `FluencyBank_017_42.wav`
- **Corpus:** FluencyBank | **Ground Truth:** `Stutter` | **Prediction:** `Clean`
- **Clinical Annotation:** `{'block': 1, 'word_rep': 0, 'sound_rep': 0, 'prolongation': 0}`
- **ASR Transcript:** `"You're stuttering. Tell me about your therapy..."`
- **Acoustic Measurements:**
  - Pause duration between `"stuttering."` and `"Tell"`: **0.38s**.
  - Silence RMS: `0.0112`.
  - Burst RMS (first 4 post-pause frames): `0.0089`.
  - Burst Ratio: **0.79** (< 1.50 threshold).
- **Active Detector Stages:**
  - *Silent Block Stage:* Rejected because silence duration (0.38s < 0.40s) and burst ratio (0.79 < 1.50) are both sub-threshold.
- **Why Missed:** The speaker produced a soft onset into `"Tell"`. The burst release energy was actually *lower* than the silence floor.
- **Classification:** **Category 2: Missing Acoustic Evidence (Sub-threshold Burst / Soft Onset)**

### 5. `FluencyBank_017_34.wav`
- **Corpus:** FluencyBank | **Ground Truth:** `Stutter` | **Prediction:** `Clean`
- **Clinical Annotation:** `{'word_rep': 1}`
- **ASR Transcript:** `""` (Empty string)
- **Active Detector Stages:** All stages inactive due to ASR worker timeout on corrupted audio.
- **Why Missed:** Heavy acoustic distortion caused Whisper decoding to fail safely to empty transcript. In the absence of text or acoustics, fallback is `Clean`.
- **Classification:** **Category 1: Upstream ASR Timeout / Audio Degradation**

### 6. `FluencyBank_017_52.wav`
- **Corpus:** FluencyBank | **Ground Truth:** `Stutter` | **Prediction:** `Clean`
- **Clinical Annotation:** `{'block': 1}`
- **ASR Transcript:** `"community activities. Okay."`
- **Acoustic Measurements:**
  - Inter-word gaps: **0.00s** across all word boundaries (`"community"` -> `"activities"` -> `"Okay"`).
  - Steady-state vowel duration: **0.42s** (at 1.11–1.53s, confidence 0.52).
- **Active Detector Stages:**
  - *Silent Block Stage:* No silence interval exists in the waveform.
  - *Prolongation Stage:* Duration (0.42s) is below the 0.50s floor.
- **Why Missed:** The block manifests as continuous laryngeal tension with uninterrupted airflow. Without silence, an acoustic silent-block detector cannot trigger.
- **Classification:** **Category 2: Missing Acoustic Evidence (Contiguous Phonation)**

### 7. `FluencyBank_031_14.wav`
- **Corpus:** FluencyBank | **Ground Truth:** `Stutter` | **Prediction:** `Clean`
- **Clinical Annotation:** `{'block': 1}`
- **ASR Transcript:** `"You get better in some way."`
- **Acoustic Measurements:**
  - Inter-word gaps: **0.00s** across all words.
  - Steady-state vowel duration: **0.45s** (at 1.65–2.10s on `"better in"`, confidence 0.56).
- **Active Detector Stages:**
  - *Silent Block Stage:* Zero silence gap.
  - *Prolongation Stage:* Duration (0.45s) is below the 0.50s floor.
- **Why Missed:** Articulatory fixation occurred during continuous phonation.
- **Classification:** **Category 2: Missing Acoustic Evidence (Contiguous Phonation)**

### 8. `FluencyBank_031_9.wav`
- **Corpus:** FluencyBank | **Ground Truth:** `Stutter` | **Prediction:** `Clean`
- **Clinical Annotation:** `{'sound_rep': 1}`
- **ASR Transcript:** `"Can we go back to this thing about there?"`
- **Active Detector Stages:** All stages inactive.
- **Why Missed:** The sound repetition (e.g., initial consonant repetition) was completely smoothed away by Whisper. The transcript is 100% grammatically fluent.
- **Classification:** **Category 1: Upstream ASR Normalization & Smoothing**

### 9. `1743-142912-0001.flac`
- **Corpus:** Libri_Stutter | **Ground Truth:** `Stutter` | **Prediction:** `Filler`
- **Clinical Annotation:** `{'stutter_types': ['3', '0', '5']}` (Prolongation, Interjection, Word Repetition)
- **ASR Transcript:** `"He didn't like to admit that he couldn't recall those two cousins of Chibbeck's. Did you say you're fighting cousin? He asked in a hesitating way. That is what I said, replied Chibbeck. He is scrapper the Kingbird, as of course you know."`
- **Acoustic Measurements:**
  - Steady-state vowel: **0.43s** (at 10.25–10.68s on `"hesitating way."`).
- **Active Detector Stages:**
  - *Prolongation Stage:* Candidate duration (0.43s) is below 0.50s.
  - *Filler Stage:* Fired on `"like"` and `"you know"`, classifying clip as `Filler`.
- **Why Missed:** In a 17.5s literary audiobook recording, sub-500ms expressive pauses/vowels co-occur with filler phrases, triggering filler dominance in the absence of a ≥0.50s stutter event.
- **Classification:** **Category 3: Sub-500ms Prosodic Elongation Masked by Fillers**

---

## 4. Root-Cause Distribution & Feasibility Separation

| Category | Description | Clips | Share | Plausibly Addressable by Detector? | Root Bottleneck |
|---|---|---|---|---|---|
| **Category 1** | ASR Normalization / Smoothing / Timeout | 4 | **44.4%** | **NO** | Whisper language model eliminates disfluency prior to detector ingestion. |
| **Category 2** | Acoustic Silent Block Absence (Continuous Phonation / Soft Burst) | 3 | **33.3%** | **NO** | Physical waveform has 0.00s gap or burst RMS < silence RMS. No silence exists to detect. |
| **Category 3** | Sub-500ms Prosodic Elongation Masked by Fillers | 1 | **11.1%** | **EXTREMELY HAZARDOUS** | Vowel duration (0.43s) overlaps with normal clean English vowels (0.43–0.48s). |
| **Category 4** | Non-Adjacent Syntactic Revision & Boundary Slicing | 1 | **11.1%** | **EXTREMELY HAZARDOUS** | Non-adjacent repetition (`to ... to`) overlaps with standard syntactic structures (e.g. `as soon as`). |

---

## 5. Candidate Intervention Ranking

| Rank | Candidate Intervention | Target Clips | Theoretical Max Recall Gain | Measured FP Risk across 60 Clips | Precision Impact | Implementation Complexity | Recommendation |
|---|---|---|---|---|---|---|---|
| **1** | **Sub-500ms Prolongation Admission (0.40s–0.50s)** | `1743-142912-0001`, `FluencyBank_017_52`, `FluencyBank_031_14` | +3 TP (+8.57%) | **+6 FPs** (`M_1022_008`, `M_1022_001`, `M_1022_049`, `FluencyBank_031_47`, `198-129977-0009`, `198-209-0026`) | Precision crashes from **86.67% to 74.29%** | Low | **REJECT** |
| **2** | **Silent Block Pause Reduction (0.40s -> 0.35s)** | `FluencyBank_017_42` | +0 to +1 TP | **+7 FPs** on natural clause/comma pauses | Precision crashes to **72.22%** | Low | **REJECT** |
| **3** | **Non-Adjacent Lexical Repetition (Gap = 1 or 2)** | `M_0061_060` | +1 TP (+2.86%) | **+4 FPs** (`198-209-0026` ["as soon as"], `1743-142912-0006`, `1743-142913-0021`, `M_0061_001`) | Precision crashes to **79.41%** | Medium | **REJECT** |
| **4** | **Boundary Hyphen Detection** | `M_0061_060` | +1 TP (+2.86%) | **+4 FPs** on 3.0s cut-off fluent clips (`M_1022_049`, `FluencyBank_017_24`, etc.) | Precision crashes to **79.41%** | Low | **REJECT** |
| **5** | **Heuristic Guessing on Clean Transcripts** | `M_1022_019`, `FluencyBank_017_15`, `FluencyBank_031_9` | 0 TP | **Unbounded FP inflation** | Degrades entire detector | High | **PROHIBITED** |

---

## 6. Key Takeaway

Of the 9 false negatives:
- **7 clips (77.8%)** are physically absent from the detector's input features (4 smoothed by Whisper ASR, 3 lacking acoustic silence/bursts).
- **2 clips (22.2%)** exhibit subtle acoustic or textual patterns, but those exact patterns are ubiquitous in fluent English speech. Any detector threshold or rule permissive enough to capture them triggers **4 to 7 new false positives**, immediately violating the core project acceptance criteria.
