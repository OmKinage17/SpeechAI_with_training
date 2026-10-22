# SpeechAI — Generalization Error Analysis & Diagnostic Audit

**Target Generalization Run:** `evaluation_results/experimental/generalization_validation/run_20261020_132214/`  
**Evaluated Candidate:** Fix 3 + Combined 2-gram/3-gram Repetition Detection  
**Evaluation Dataset:** Independent 60-Clip Generalization Benchmark ($N=60$ across 6 unseen source entities)  
**Audit Date:** 2026-10-20  
**Audit Nature:** Diagnostic, Read-Only Audit (No algorithms, thresholds, or code modified)  

---

## 1. Executive Summary

A comprehensive diagnostic audit was conducted on the completed Ablation 1 Generalization Run (`run_20261020_132214`). The run evaluated the accepted development candidate—**Fix 3 + Combined 2-gram/3-gram Repetition Detection**—against the Fix 3 Reference on $N=60$ audio clips drawn from speaker-disjoint splits of **UCLASS** (speakers `M_1022`, `M_0061`), **FluencyBank** (episodes `Ep_017`, `Ep_031`), and **Libri_Stutter** (readers `Reader_198`, `Reader_1743`).

### Key Findings of the Audit:
1. **Generalization Performance Confirmed:** The multi-word n-gram repetition detector successfully recovered 2 true stutter clips in Libri_Stutter (`1743-142914-0005.flac` and `1743-142913-0022.flac`) without introducing a single new false positive across any corpus.
2. **Current Performance Metrics:** On the 60 generalization clips, the combined detector achieves:
   - **3-Class Accuracy:** 53.33% (32/60)
   - **Stutter Recall:** 62.86% (22/35)
   - **Stutter Precision:** 68.75% (22/32)
   - **Stutter Specificity:** 60.00% (15/25)
   - **Binary Confusion Matrix:** TP=22, FP=10, FN=13, TN=15
3. **Core Diagnostic Takeaway:**
   - **False Negatives (13 clips):** Bottlenecked by **single-word repetitions collapsed/normalized by Whisper ASR** (4 clips, 30.8%), **phrase/word repetitions interrupted by conversational fillers** (3 clips, 23.1%), and **silent/tense acoustic blocks** (3 clips, 23.1%).
   - **False Positives (10 clips):** Heavily dominated by **conversational filler prolongations** (4 clips, 40.0%) and **normal syntactic/sentence-boundary pauses mistaken for blocks** (3 clips, 30.0%).
   - **Integrity Status:** 100% verified. All 12 baseline files remain strictly pristine, holdout immutability is intact, and the generalization feature cache SHA-256 (`91ad23deef8331ec52953412927ca4a08b2d113def0f5dc997f70918dbe4d076`) matches exactly.

---

## 2. Independent Metric Recomputation & Artifact Inconsistency Audit

All predictions from `per_clip_predictions.json` were independently extracted and recomputed.

### 2.1 Recomputed Multiclass Confusion Matrices ($N=60$)

#### Fix 3 Reference
| Ground Truth \ Predicted | Stutter | Filler | Clean | Total Support |
| :--- | :---: | :---: | :---: | :---: |
| **Stutter** | **20** | 6 | 9 | 35 |
| **Filler** | 5 | **4** | 2 | 11 |
| **Clean** | 5 | 3 | **6** | 14 |
| **Total Predicted** | 30 | 13 | 17 | 60 |

#### Fix 3 + Combined (2g+3g)
| Ground Truth \ Predicted | Stutter | Filler | Clean | Total Support | Delta vs Ref |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stutter** | **22** | 5 | 8 | 35 | **+2 TP (-1 Filler FN, -1 Clean FN)** |
| **Filler** | 5 | **4** | 2 | 11 | 0 (Unchanged) |
| **Clean** | 5 | 3 | **6** | 14 | 0 (Unchanged) |
| **Total Predicted** | 32 | 12 | 16 | 60 | +2 Stutter, -1 Filler, -1 Clean |

### 2.2 Recomputed Binary Stutter Metrics
- True Positives (TP) = 22 (Ref: 20, Delta = +2)
- False Positives (FP) = 10 (Ref: 10, Delta = 0)
- False Negatives (FN) = 13 (Ref: 15, Delta = -2)
- True Negatives (TN) = 15 (Ref: 15, Delta = 0)
- Stutter Recall = 22 / 35 = **62.86%** (Ref: 57.14%, +5.72 pp)
- Stutter Precision = 22 / 32 = **68.75%** (Ref: 66.67%, +2.08 pp)
- Stutter Specificity = 15 / 25 = **60.00%** (Ref: 60.00%, 0.00 pp)
- Stutter F1 Score = **0.6567** (Ref: 0.6154, +0.0413)

### 2.3 Artifact Inconsistency Documentation
During independent recomputation, a typographical discrepancy was detected in `GENERALIZATION_VALIDATION_REPORT.md`:
- **Discrepancy Location:** `GENERALIZATION_VALIDATION_REPORT.md` Section 5 ("Multiclass Confusion Matrices").
- **Observed Text Table in Report:**
  - Listed Ref Stutter row as: `20 Stutter, 4 Filler, 11 Clean`.
  - Listed Combined Stutter row as: `22 Stutter, 3 Filler, 10 Clean`.
  - Listed Filler rows as: `7 Stutter, 3 Filler, 1 Clean`.
  - Listed Clean rows as: `3 Stutter, 4 Filler, 7 Clean`.
- **True Mathematical Recomputation:**
  - Ground Truth Stutter predictions are actually: **22 Stutter, 5 Filler, 8 Clean** (Ref: 20 Stutter, 6 Filler, 9 Clean).
  - Ground Truth Filler predictions are actually: **5 Stutter, 4 Filler, 2 Clean**.
  - Ground Truth Clean predictions are actually: **5 Stutter, 3 Filler, 6 Clean**.
- **Root Cause & Impact:**
  - This was a transcription/typographical error in the markdown presentation table of the previous report.
  - The underlying serialized files—`confusion_matrices.json`, `generalization_metrics.json`, and `per_clip_predictions.json`—are **100% mathematically correct and consistent with each other**.
  - Table 4 in the report (the binary metric table: 22/10/13/15) is also completely accurate.

---

## 3. Stutter False Negative Root-Cause Categorization ($N=13$)

The 13 ground-truth stutter clips missed by Fix 3 Combined were categorized according to the strongest available acoustic, lexical, and timestamp evidence:

| Category | Count | Percentage | Primary Failure Mechanism |
| :--- | :---: | :---: | :--- |
| **Single-word repetition** | 4 | 30.8% | Whisper normalized/collapsed repeated single words or part-word sounds. |
| **Repeated phrase not detected by n-gram** | 3 | 23.1% | Phrase or word repeated across an intervening filler (`sort of`, `obviously`, `er`), breaking strict token contiguity. |
| **Block or abnormal pause** | 3 | 23.1% | Tense silent block / abnormal pause; standalone block detector is disabled in Fix 3. |
| **Acoustic prolongation** | 2 | 15.4% | Sound elongation annotated in ground truth fell below the 0.50s duration threshold or spectral flux gating. |
| **ASR transcription/timestamp limitation** | 1 | 7.7% | Extreme dysfluency exceeded Whisper 25.0s safety timeout; empty transcript generated. |
| **Boundary discount or contextual rejection** | 0 | 0.0% | No true positive was discarded due to contextual discount. |
| **Other / insufficient evidence** | 0 | 0.0% | All 13 false negatives have clearly verifiable failure mechanisms. |
| **Total False Negatives** | **13** | **100.0%** | |

### Detailed Breakdown of False Negative Categories:

#### 1. Single-Word Repetitions (4 Clips)
- **`M_1022_11y3m_1_dysfluent_019.wav` (UCLASS):** Ground truth indicates `word_rep=1`. Whisper transcript: `"things. Well, my best subject."`. The speaker repeated a monosyllabic word, but Whisper normalized the output into a single fluent token. Dominant prediction: **Filler** (due to `"well"`).
- **`M_0061_16y9m-1_dysfluent_060.wav` (UCLASS):** Ground truth indicates `word_rep=1`. Whisper transcript: `"In your health master, we believe to code, to be called Mr. Sh-"`. Partial false-start / cut-off syllable transcribed without repeated words. Dominant prediction: **Clean**.
- **`FluencyBank_017_15.wav` (FluencyBank):** Ground truth consensus `word_rep=1`. Whisper transcript: `"I'm just going to do a video. Oh"`. Monosyllabic word repetition smoothed by ASR. Dominant prediction: **Clean**.
- **`FluencyBank_031_9.wav` (FluencyBank):** Ground truth consensus `sound_rep=1`. Whisper transcript: `"Can we go back to this thing about there?"`. Sound/syllable repetition on onset suppressed by Whisper. Dominant prediction: **Clean**.

#### 2. Repeated Phrase Not Detected by N-Gram (3 Clips)
- **`M_0061_16y9m-1_dysfluent_081.wav` (UCLASS):** Ground truth indicates `word_rep=1`. Whisper transcript: `"Every lesson today, it depends, it sort of depends what day it is."`.
  - **Matched Span:** `"it depends, it sort of depends"`
  - **Mechanism:** The speaker repeated the 2-word phrase `"it depends"`, but inserted the filler phrase `"sort of"`. Because the repetition is non-contiguous (`[it depends] [sort of] [depends]`), the strict contiguous 2-gram detector could not match it. Dominant prediction: **Filler**.
- **`FluencyBank_017_2.wav` (FluencyBank):** Ground truth consensus `sound_rep=1, word_rep=1`. Whisper transcript: `"I still, obviously, still stutter, although I-"`.
  - **Matched Span:** `"still, obviously, still"`
  - **Mechanism:** The word `"still"` is repeated across an intervening parenthetical adverbial filler `"obviously,"`. The unigram detector requires strictly adjacent tokens ($w_i == w_{i+1}$), and the n-gram detector requires adjacent multi-word blocks. Dominant prediction: **Clean**.
- **`198-209-0025.flac` (Libri_Stutter):** Ground truth indicates `stutter_types=['4', '0']` (Type 4 = Phrase Repetition). Whisper transcript: `"And he being hurried for time, and, er, at time, and not caring to have it put off, made a point of her being denied. That was all I do assure you."`.
  - **Matched Span:** `"for time, and, er, at time"`
  - **Mechanism:** The reader experienced a phrase restart with the filler interjection `"er"`. The interruption prevented contiguous n-gram matching. Dominant prediction: **Filler** (due to `"er"`).

#### 3. Blocks or Abnormal Pauses (3 Clips)
- **`FluencyBank_017_42.wav` (FluencyBank):** Ground truth consensus `block=1`. Transcript: `"You're stuttering. Tell me about your therapy..."`. Speaker exhibits tense silent blockage during conversational turn. Fix 3 has no standalone acoustic block detector. Dominant prediction: **Clean**.
- **`FluencyBank_017_52.wav` (FluencyBank):** Ground truth consensus `block=1`. Transcript: `"community activities. Okay."`. Silent pre-utterance block. Dominant prediction: **Clean**.
- **`FluencyBank_031_14.wav` (FluencyBank):** Ground truth consensus `block=1`. Transcript: `"You get better in some way."`. Pre-phonatory block before onset. Dominant prediction: **Clean**.

#### 4. Acoustic Prolongations (2 Clips)
- **`M_0061_16y9m-1_dysfluent_046.wav` (UCLASS):** Ground truth indicates `prolongation=1, sound_rep=1, word_rep=1`. Transcript: `"Like this school, the schools, it's-"`. Prolongation candidate fell below duration threshold (0.50s) or spectral flux bounds. Dominant prediction: **Filler** (due to `"like"`).
- **`1743-142912-0001.flac` (Libri_Stutter):** Ground truth indicates `stutter_types=['3', '0', '5']` (Type 5 = Prolongation). Transcript: `"He didn't like to admit that he couldn't recall those two cousins of Chibbeck's..."`. Acoustic prolongation did not trigger above the standalone confidence threshold. Dominant prediction: **Filler** (due to `"you know"`, `"like"`).

#### 5. ASR Safety Timeout / Truncation (1 Clip)
- **`FluencyBank_017_34.wav` (FluencyBank):** Ground truth consensus `word_rep=1`. The clip audio contains severe dysfluent struggle and background room acoustics that caused Whisper `base` to exceed the 25.0s CPU safety timeout. In accordance with safety policies, an empty transcript was returned. Dominant prediction: **Clean** (confidence: 0.0).

---

## 4. Stutter False Positive Root-Cause Categorization ($N=10$)

The 10 ground-truth non-stutter clips incorrectly classified as Stutter by Fix 3 Combined were categorized as follows:

| Category | Count | Percentage | Primary Failure Mechanism |
| :--- | :---: | :---: | :--- |
| **Conversational filler** | 4 | 40.0% | Prolonged vocalization of filler vowels (`um`, `uh`) or repeated filler token (`uh uh`) triggered stutter detectors. |
| **Normal pause mistaken for a block** | 3 | 30.0% | Fluent hesitation silences or syntactic comma/sentence boundaries mistaken for tense blocks. |
| **Fluent repeated phrase** | 0 | 0.0% | Zero fluent phrases were falsely matched by the multi-word n-gram detector. |
| **ASR duplication or transcription error** | 0 | 0.0% | No Whisper hallucinated duplicate tokens observed. |
| **Other / insufficient evidence** | 3 | 30.0% | Expressive read speech in LibriSpeech with natural prosodic elongation exceeding duration thresholds. |
| **Total False Positives** | **10** | **100.0%** | |

### Detailed Breakdown of False Positive Categories:

#### 1. Conversational Fillers (4 Clips)
- **`M_1022_11y3m_1_dysfluent_008.wav` (UCLASS, GT: Filler):**
  - **Transcript:** `"High school, the, um, kid go..."`
  - **Detector Event:** `acoustic_prolongation` on `"um,"` (start: 1.31s, end: 1.85s, duration: 0.54s, confidence: 0.68) and `inter_word_acoustic_block` (dur: 0.78s).
  - **Cause:** Extended vocalization of the filler vowel `"um"` directly triggered acoustic prolongation.
- **`M_1022_11y3m_1_dysfluent_000.wav` (UCLASS, GT: Filler):**
  - **Transcript:** `"It's good, um, hand in school. Yeah."`
  - **Detector Event:** `acoustic_prolongation` on `"um,"` (start: 0.58s, end: 1.51s, duration: 0.93s, confidence: 1.00).
  - **Cause:** A drawn-out filler `"um"` lasting nearly 1 second triggered standalone prolongation with maximum confidence.
- **`FluencyBank_017_4.wav` (FluencyBank, GT: Filler):**
  - **Transcript:** `"And work, uh, environment, uh, uh."`
  - **Detector Event:** `word_repetition` on `"uh uh"` (start: 2.36s, end: 2.98s, duration: 0.62s, confidence: 0.85).
  - **Cause:** The speaker uttered a double filler interjection (`"uh, uh"`). The lexical repetition detector matched `"uh uh"` because conversational fillers were not filtered out of lexical repetition candidate matching.
- **`FluencyBank_031_47.wav` (FluencyBank, GT: Filler):**
  - **Transcript:** `"I don't have to learn. Um, so are there other?"`
  - **Detector Event:** `acoustic_prolongation` on `"Um,"` (dur: 0.55s, conf: 0.69) and `inter_word_acoustic_block` (dur: 0.74s).
  - **Cause:** Pre-sentence hesitation and vocalic elongation on `"Um,"`.

#### 2. Normal Pauses Mistaken for Blocks (3 Clips)
- **`M_0061_16y9m-1_dysfluent_030.wav` (UCLASS, GT: Filler):**
  - **Transcript:** `"Which is really good. Um, what about it?"`
  - **Detector Event:** `inter_word_acoustic_block` (dur: 0.56s and 0.86s, conf: 0.60).
  - **Cause:** Natural inter-sentence silence between `"good."` and `"Um,"` registered as an acoustic block.
- **`FluencyBank_031_69.wav` (FluencyBank, GT: Clean):**
  - **Transcript:** `"Oh, eight years ago."`
  - **Detector Event:** `acoustic_prolongation` on sentence terminal `"ago."` (dur: 0.50s, conf: 0.62) and `inter_word_acoustic_block` (dur: 0.86s).
  - **Cause:** Pre-speech delay (0.16s–1.02s) and sentence-final phonetic lengthening on `"ago."`.
- **`198-209-0026.flac` (Libri_Stutter, GT: Clean):**
  - **Transcript:** `"I meant to make her apology as soon as possible. Catherine's mind was greatly eased by this information, yet as something a solicitude remained, from which sprang the following question..."`
  - **Detector Event:** `acoustic_prolongation` on `"remained,"` (dur: 0.54s, conf: 0.68) and `inter_word_acoustic_block` (dur: 0.48s, 0.42s).
  - **Cause:** Syntactic comma pause after `"remained,"` in literary narration.

#### 3. Other / Natural Prosodic Elongation in Clean Speech (3 Clips)
- **`M_0061_16y9m-1_fluent_001.wav` (UCLASS, GT: Clean):** Prolongation detected on intra-phrase vowel (dur: 0.37s, conf: 0.46) in conversational speech.
- **`198-129977-0009.flac` (Libri_Stutter, GT: Clean):** Prolongation detected on `"moss"` (dur: 0.40s, conf: 0.50) in expressive literary reading.
- **`1743-142913-0021.flac` (Libri_Stutter, GT: Clean):** Prolongation detected on `"noticed"` (dur: 0.38s, conf: 0.47) due to prosodic focus stress.

---

## 5. Corpus-Level Performance & Error Discrepancy Breakdown

The error profiles diverge markedly across the three evaluation corpora, reflecting fundamental differences in speech register and annotation conventions:

| Corpus | Total Clips | Stutter GT | Stutter Recall | Stutter Precision | Dominant FN Cause | Dominant FP Cause |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Libri_Stutter** | 20 | 14 | **85.71% (12/14)** | **80.00% (12/15)** | Non-adjacent phrase repetition (1) | Prosodic elongation in read speech (2) |
| **UCLASS** | 20 | 11 | **63.64% (7/11)** | **63.64% (7/11)** | Single-word repetition collapsed by ASR (2) | Prolonged filler vowels (`um`) (2) |
| **FluencyBank** | 20 | 10 | **30.00% (3/10)** | **50.00% (3/6)** | Silent acoustic blocks (3) | Filler repetition & boundary pauses (2) |

### Corpus-Specific Insights:
1. **Libri_Stutter (Controlled Audiobook Reading):**
   - High baseline recall (85.71%) because dysfluencies in audiobooks predominantly manifest as overt multi-word restarts and repetitions.
   - The n-gram detector excelled here, recovering 2 full true positives.
   - Errors are confined to expressive reading prosody (readers elongating stressed syllables like `"moss"` or pausing at complex literary clause boundaries).
2. **UCLASS (Adolescent Spontaneous Monologues):**
   - Balanced performance (63.64% Recall, 63.64% Precision).
   - High density of conversational filler interjections (`"um"`, `"well"`, `"sort of"`).
   - Speakers frequently use fillers during lexical access, causing false alarms when filler vowels are drawn out, or masking repetitions when fillers interrupt repeated phrases (`"it depends, it sort of depends"`).
3. **FluencyBank (Clinical Stuttering Interviews):**
   - Severe recall bottleneck (30.00% Recall, 7 FNs out of 10 Stutter clips).
   - Speakers in FluencyBank exhibit clinical core stuttering behaviors—specifically **severe silent blocks** (3 FNs) and **prolonged tense articulatory posturing** that causes ASR safety timeouts (1 FN).
   - Because Fix 3 does not enable standalone block detection, these clinical blocks are completely invisible to the pipeline.

---

## 6. Consolidated Error Manifest ($N=23$)

Below is the complete, itemized audit manifest of all 13 False Negatives and 10 False Positives:

### 6.1 False Negatives ($N=13$)
| # | Clip ID | Corpus | GT Dominant | Annotation | Pred (Ref / Comb) | Category | Evidence Strength | Matched Tokens / Span | Supporting Evidence |
| :-: | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :--- | :--- |
| 1 | `M_1022_11y3m_1_dysfluent_019.wav` | UCLASS | Stutter | `word_rep=1` | Filler / Filler | Single-word repetition | PLAUSIBLE_HYPOTHESIS | `"Well, my best subject."` | Single-word repeat collapsed by Whisper into single fluent token; `"well"` triggered Filler. |
| 2 | `M_0061_16y9m-1_dysfluent_081.wav` | UCLASS | Stutter | `word_rep=1` | Filler / Filler | Repeated phrase not detected by n-gram | VERIFIED | `"it depends, it sort of depends"` | 2-word phrase repeated across intervening filler `"sort of"`, breaking contiguous adjacency. |
| 3 | `M_0061_16y9m-1_dysfluent_060.wav` | UCLASS | Stutter | `word_rep=1` | Clean / Clean | Single-word repetition | PLAUSIBLE_HYPOTHESIS | `"to code, to be called Mr. Sh-"` | Cut-off syllable and revision; no adjacent repeating word token generated. |
| 4 | `M_0061_16y9m-1_dysfluent_046.wav` | UCLASS | Stutter | `prol=1, s_rep=1, w_rep=1` | Filler / Filler | Acoustic prolongation | PLAUSIBLE_HYPOTHESIS | `"Like this school, the schools, it's-"` | Ground truth annotated prolongation; acoustic detector failed duration (0.50s) / flux gating. |
| 5 | `FluencyBank_017_15.wav` | FluencyBank | Stutter | `word_rep=1, int=1` | Clean / Clean | Single-word repetition | PLAUSIBLE_HYPOTHESIS | `"I'm just going to do a video. Oh"` | Ground truth consensus word repetition; Whisper smoothed output to clean speech. |
| 6 | `FluencyBank_017_42.wav` | FluencyBank | Stutter | `block=1` | Clean / Clean | Block or abnormal pause | VERIFIED | `"You're stuttering. Tell me..."` | Ground truth consensus Block >= 2; standalone block detector disabled in Fix 3. |
| 7 | `FluencyBank_017_34.wav` | FluencyBank | Stutter | `word_rep=1` | Clean / Clean | ASR transcription/timestamp limitation | VERIFIED | `""` (Empty transcript) | Audio dysfluency exceeded Whisper 25.0s safety timeout; 0 words generated. |
| 8 | `FluencyBank_017_52.wav` | FluencyBank | Stutter | `block=1` | Clean / Clean | Block or abnormal pause | VERIFIED | `"community activities. Okay."` | Ground truth consensus Block >= 2; silent blockage undetected. |
| 9 | `FluencyBank_017_2.wav` | FluencyBank | Stutter | `s_rep=1, w_rep=1, int=1` | Clean / Clean | Repeated phrase not detected by n-gram | VERIFIED | `"still, obviously, still"` | Word `"still"` repeated across parenthetical adverbial filler `"obviously,"`. |
| 10 | `FluencyBank_031_14.wav` | FluencyBank | Stutter | `block=1` | Clean / Clean | Block or abnormal pause | VERIFIED | `"You get better in some way."` | Ground truth consensus Block >= 2; pre-speech block undetected. |
| 11 | `FluencyBank_031_9.wav` | FluencyBank | Stutter | `sound_rep=1` | Clean / Clean | Single-word repetition | PLAUSIBLE_HYPOTHESIS | `"Can we go back to this thing..."` | Sound/syllable repetition normalized by ASR. |
| 12 | `198-209-0025.flac` | Libri_Stutter | Stutter | `stutter_types=['4', '0']` | Filler / Filler | Repeated phrase not detected by n-gram | VERIFIED | `"for time, and, er, at time"` | Phrase repetition interrupted by filler `"er"`, breaking strict n-gram contiguity. |
| 13 | `1743-142912-0001.flac` | Libri_Stutter | Stutter | `stutter_types=['3', '0', '5']` | Filler / Filler | Acoustic prolongation | PLAUSIBLE_HYPOTHESIS | `"He didn't like to admit that..."` | Annotated prolongation (type 5) failed acoustic candidate threshold. |

### 6.2 False Positives ($N=10$)
| # | Clip ID | Corpus | GT Dominant | Annotation | Pred (Ref / Comb) | Category | Evidence Strength | Triggering Feature & Span | Supporting Evidence |
| :-: | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :--- | :--- |
| 1 | `M_1022_11y3m_1_dysfluent_008.wav` | UCLASS | Filler | `interjection=1` | Stutter / Stutter | Conversational filler | VERIFIED | Prolongation on `"um,"` (dur: 0.54s, conf: 0.68) | Extended vocalization of filler vowel `"um"` directly triggered prolongation. |
| 2 | `M_1022_11y3m_1_dysfluent_000.wav` | UCLASS | Filler | `interjection=1` | Stutter / Stutter | Conversational filler | VERIFIED | Prolongation on `"um,"` (dur: 0.93s, conf: 1.00) | Long filler interjection `"um"` triggered standalone prolongation with max confidence. |
| 3 | `M_0061_16y9m-1_dysfluent_030.wav` | UCLASS | Filler | `interjection=1` | Stutter / Stutter | Normal pause mistaken for a block | VERIFIED | Inter-word block (dur: 0.56s, 0.86s, conf: 0.60) | Natural hesitation silences around `"Um"` mistaken for dysfluent blocks. |
| 4 | `M_0061_16y9m-1_fluent_001.wav` | UCLASS | Clean | All 0 | Stutter / Stutter | Other / insufficient evidence | VERIFIED | Prolongation intra-phrase (dur: 0.37s, conf: 0.46) | Natural phonetic duration variation in conversational monologue. |
| 5 | `FluencyBank_017_4.wav` | FluencyBank | Filler | `interjection=1` | Stutter / Stutter | Conversational filler | VERIFIED | Word repetition on `"uh uh"` (dur: 0.62s, conf: 0.85) | Lexical repetition detector triggered on repeated filler `"uh uh"`. |
| 6 | `FluencyBank_031_47.wav` | FluencyBank | Filler | `interjection=1` | Stutter / Stutter | Conversational filler | VERIFIED | Prolongation on `"Um,"` (dur: 0.55s, conf: 0.69) | Sentence-boundary hesitation vocalization on filler `"Um,"`. |
| 7 | `FluencyBank_031_69.wav` | FluencyBank | Clean | All 0 | Stutter / Stutter | Normal pause mistaken for a block | VERIFIED | Prolongation on `"ago."` (0.50s) & block (0.86s) | Normal pre-speech pause and sentence-final lengthening on terminal word `"ago."`. |
| 8 | `198-129977-0009.flac` | Libri_Stutter | Clean | `stutter_types=['0']` | Stutter / Stutter | Other / insufficient evidence | VERIFIED | Prolongation on `"moss"` (dur: 0.40s, conf: 0.50) | Prosodic emphasis on fricative noun in expressive audiobook reading. |
| 9 | `198-209-0026.flac` | Libri_Stutter | Clean | `stutter_types=['0']` | Stutter / Stutter | Normal pause mistaken for a block | VERIFIED | Prolongation on `"remained,"` (0.54s) & block (0.48s) | Syntactic comma pause and pre-pausal lengthening in literary prose. |
| 10 | `1743-142913-0021.flac` | Libri_Stutter | Clean | `stutter_types=['0']` | Stutter / Stutter | Other / insufficient evidence | VERIFIED | Prolongation on `"noticed"` (dur: 0.38s, conf: 0.47) | Natural articulatory duration on accented word in read speech. |

---

## 7. Methodological Distinction: Verified Root Causes vs Plausible Hypotheses

To maintain diagnostic rigor, causes are strictly partitioned into **Verified Causes** (grounded in directly observable acoustic/lexical telemetry in the cache) and **Plausible Hypotheses** (inferred from ground truth annotations where intermediate representations lack direct traces):

### 7.1 Verified Root Causes ($N=15$ errors)
1. **Filler Token Repetition (`FluencyBank_017_4.wav`):** Directly verified from token sequence `['uh', 'uh']` matching the word repetition detector.
2. **Prolonged Filler Vowels (3 clips: `M_1022_008`, `M_1022_000`, `FluencyBank_031_47`):** Directly verified from timestamp overlap between acoustic prolongation spans and ASR filler tokens `"um"`.
3. **Syntactic and Hesitation Pauses (3 clips: `M_0061_030`, `FluencyBank_031_69`, `198-209-0026`):** Directly verified from silence gap durations aligned with sentence boundaries or punctuation marks.
4. **Natural Prosodic Elongation in Clean Reading (3 clips: `M_0061_fluent_001`, `198-129977-0009`, `1743-142913-0021`):** Directly verified from acoustic prolongation events occurring on fluent words in clean ground-truth audio.
5. **Non-Adjacent Repetitions with Intervening Fillers (3 clips: `M_0061_081`, `FluencyBank_017_2`, `198-209-0025`):** Directly verified from the presence of repeated words/phrases (`"it depends"`, `"still"`, `"for time"`) interrupted by fillers (`"sort of"`, `"obviously"`, `"er"`) in the ASR transcript.
6. **ASR Safety Timeout (`FluencyBank_017_34.wav`):** Directly verified from `is_timeout: true` and empty transcript metadata in feature cache.
7. **Clinical Silent Blocks (3 clips: `FluencyBank_017_42`, `FluencyBank_017_52`, `FluencyBank_031_14`):** Directly verified from ground-truth consensus annotations (`block=1`) and absence of standalone block detector triggers in Fix 3.

### 7.2 Plausible Hypotheses ($N=8$ errors)
1. **Single-Word and Sound Repetition Collapsing by Whisper (4 clips: `M_1022_019`, `M_0061_060`, `FluencyBank_017_15`, `FluencyBank_031_9`):** Ground-truth annotations confirm the physical presence of sound/word repetitions, but the Whisper transcripts contain single fluent tokens. It is hypothesized based on known Whisper behavior that the sequence-to-sequence decoder normalized the stuttered repetitions into fluent grammar.
2. **Sub-Threshold Acoustic Prolongations (2 clips: `M_0061_046`, `1743-142912-0001`):** Ground-truth annotations indicate prolongations, but acoustic feature mining yielded no candidates >= 0.50s with acceptable flux. It is hypothesized that the physical prolongation duration was between 0.25s and 0.49s, or that high pitch variability prevented candidate admission.

---

## 8. Diagnostic Conclusions & Technical Implications

1. **Ablation 1 (N-gram Repetitions) is Solid:** Multi-word repetition detection has zero false alarms on non-stuttered multi-word sequences across all 60 generalization clips, while recovering 2 true positives in read speech. It should remain frozen and accepted.
2. **The Precision Bottleneck is Conversational Fillers (40% of FPs):** The detector currently lacks filler-awareness in both acoustic prolongation mining and lexical repetition matching. Drawing out `"um"` or repeating `"uh uh"` should never trigger a stutter diagnosis.
3. **The Recall Bottleneck is Bimodal:**
   - In spontaneous conversational speech (UCLASS, FluencyBank), repetitions are frequently **interrupted by filler interjections** (accounting for 3 FNs).
   - In clinical stuttering (FluencyBank), dysfluency manifests primarily as **silent blocks** (accounting for 3 FNs).
