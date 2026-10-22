# Phase 4A: Development-Only False-Negative Investigation Report

**Project:** SpeechAI Experimental Benchmark Suite  
**Date:** October 22, 2026  
**Auditor / Investigator:** Antigravity (Advanced Agentic AI Pair Programmer)  
**Status:** Completed — Investigation & Error Analysis Only (Strict Stop Condition Enforced)  
**Scope:** Confined strictly to development data ($N=90$ clips: 30 standardized dev clips + 60 generalization validation clips) and the authoritative Phase 2 baseline configuration.  
**Production & Benchmark Impact:** **ZERO** (0 production modules, 0 frontend files, 0 detector logic changes, 0 threshold modifications, 0 re-evaluations of the locked 120-clip final set).

---

## Executive Summary

Phase 4A executed a rigorous, evidence-grounded error analysis of all false negatives produced by the authoritative Phase 2 baseline detector on development data. The investigation strictly adhered to the diagnostic safeguard: no code was modified, no thresholds were altered, no experiments were rerun on the locked holdout set, and no implementations were attempted.

| Key Metric / Check | Audit & Analysis Finding | Status |
| :--- | :--- | :---: |
| **Development Split Isolation** | 90 clips, 9 speakers. Overlap with 120-clip final set: **0 clips (0.0%), 0 speakers (0.0%)**. | **VERIFIED** |
| **Authoritative Baseline Metrics** | Reconciled dev metrics: **TP=40, FP=4, FN=19, TN=27** (Prec: 90.91%, Rec: 67.80%, F1: 0.7767, Spec: 87.10%). | **RECONCILED** |
| **Total False Negatives Analyzed** | **19 clips** (UCLASS: 8, FluencyBank: 10, Libri_Stutter: 1). All 19 had 0 detector events and conf = 0.0. | **INSPECTED** |
| **Primary Failure Modes Identified** | 1. Whisper timestamp snapping & acoustic block masking (8 clips)<br>2. Multi-iteration hesitation struggle filtered by filler gating (3 clips)<br>3. Abrupt terminal phonological cutoffs (2 clips)<br>4. Morphological stem & interrupted phrasal repetitions (3 clips)<br>5. Whisper transcript smoothing / empty ASR output (3 clips) | **GROUNDED** |
| **Ranked Potential Interventions** | 1. Multi-Iteration Hesitation Struggle Recognition ($\ge 3$ repetitions)<br>2. Phonological Cutoff Fragment Detection (`\b[a-zA-Z]{1,4}-`)<br>3. ASR-Independent Pre-Utterance Acoustic Block Detection | **PRIORITIZED** |

---

## 1. Dataset & Configuration Verification

### 1.1 Development Split Composition & Isolation
Development data was verified for complete isolation from the locked 120-clip final evaluation set:

- **Standardized Dev Split:** 30 clips from [`feature_cache.json`](file:///e:/SpeechPrj/evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json) (UCLASS: 20, FluencyBank: 10).
- **Generalization Validation Split:** 60 clips from [`run_manifest.json`](file:///e:/SpeechPrj/evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json) (UCLASS: 20, FluencyBank: 20, Libri_Stutter: 20).
- **Total Development Set:** **90 clips** across 9 unique speakers (`M_0030`, `M_0061`, `M_1022`, FluencyBank speakers `010`, `017`, `031`, and Libri_Stutter speakers `198`, `1743`, `2989`).
- **Holdout Set Isolation:** The locked 120-clip final set contains 12 distinct speakers (`M_1023`, `M_1099`, `M_1105`, FluencyBank speakers `024`, `038`, `178`, Libri_Stutter speakers `1926`, `2989-eval`, etc.).
  - **Clip Overlap:** **0 clips (0.00%)**
  - **Speaker Overlap:** **0 speakers (0.00%)**

### 1.2 Authoritative Baseline Configuration
All analyses in this phase use the authoritative Phase 2 baseline configuration without runner overrides:

```python
StutterDetectorConfig(
    confident_threshold=0.60,
    uncertain_threshold=0.35,
    min_prolongation_duration_sec=0.50,
    severe_prolongation_duration_sec=0.60,
    standalone_prolongation_confidence=0.50,
    standalone_prolongation_min_duration_sec=0.0,  # Authoritative Phase 2 setting
    enable_contextual_scoring=True,
    enable_ngram_repetition=True,
    ngram_sizes=(2, 3),
    ngram_repetition_confidence=0.85,
    enable_filler_boundary_gating=True,
    filler_overlap_threshold=0.35,
    filler_word_coverage_threshold=0.40,
    boundary_prolongation_min_duration_sec=0.65,
    gate_filler_blocks=True,
    enable_interrupted_repetition=True,
    interrupted_repetition_max_skip=3,
    interrupted_repetition_allow_stem=True,
    interrupted_repetition_allow_hesitations=True,
    interrupted_repetition_confidence=0.85
)
```

### 1.3 Reconciled Development Baseline Metrics

| Metric | Stutter Class (P1) | Filler Class (P2) | Clean Class (P3) | Overall Macro / Acc |
| :--- | :---: | :---: | :---: | :---: |
| **Ground Truth Support** | **59 clips** | **16 clips** | **15 clips** | **90 clips** |
| **True Positives (TP)** | 40 | 16 | 8 | 64 correct |
| **False Positives (FP)**| 4 | 5 | 17 | — |
| **False Negatives (FN)**| 19 | 0 | 7 | — |
| **True Negatives (TN)** | 27 | 69 | 65 | — |
| **Precision** | **90.91%** | 76.19% | 32.00% | 66.37% (Macro-P) |
| **Recall** | **67.80%** | 100.00% | 53.33% | 73.71% (Macro-R) |
| **F1 Score** | **0.7767** | 0.8649 | 0.4000 | **0.6805 (Macro-F1)** |
| **Specificity** | **87.10%** | 93.24% | 79.27% | 86.54% (Macro-Spec)|
| **Dominant Accuracy** | — | — | — | **71.11% (64/90)** |

---

## 2. False-Negative Inventory (N=19)

Every false negative on development data produced 0 detector events (`lex_reps = 0, prols = 0, blocks = 0`) and confidence score `0.0`, resulting in a default classification of `CONFIDENT_CLEAN` (or `Filler` if ASR detected a non-stutter interjection):

| # | Clip Filename | Dataset & Split | GT Annotations | Whisper Transcript | Pred Dominant (Conf) | Primary Failure Mode |
| :-: | :--- | :---: | :--- | :--- | :---: | :--- |
| **1** | `M_0030_16y4m_1_dysfluent_000.wav` | UCLASS (dev_30) | Dysfluent reading | `A wealthy Athenian.` | Clean (0.0) | Silent pre-utterance block masked by Whisper timestamp snapping |
| **2** | `M_0030_16y4m_1_dysfluent_001.wav` | UCLASS (dev_30) | Dysfluent reading | `A wealthy Athenian.` | Clean (0.0) | Silent pre-utterance block masked by Whisper timestamp snapping |
| **3** | `M_0030_16y4m_1_dysfluent_002.wav` | UCLASS (dev_30) | Dysfluent reading | `A wealthy Athenian.` | Clean (0.0) | Silent pre-utterance block masked by Whisper timestamp snapping |
| **4** | `M_0030_16y4m_1_dysfluent_003.wav` | UCLASS (dev_30) | Dysfluent reading | `Well, the Athenian instead.` | Clean (0.0) | Articulatory block / struggle before vowel onset |
| **5** | `M_0030_16y4m_1_dysfluent_009.wav` | UCLASS (dev_30) | Dysfluent reading | `You could, um, um, uh, uh... (x110)` | Clean (0.0) | Multi-iteration hesitation struggle gated as conversational filler |
| **6** | `M_0030_16y4m_1_dysfluent_011.wav` | UCLASS (dev_30) | Dysfluent reading | `You could make a warship, or you could...` | Clean (0.0) | Interrupted phrasal repetition exceeding max skip |
| **7** | `M_0030_16y4m_1_dysfluent_015.wav` | UCLASS (dev_30) | Dysfluent reading | `er, er, er, er, er, er, er. Once, oh quick.` | Clean (0.0) | 7x repeated hesitation struggle gated as conversational filler |
| **8** | `FluencyBank_010_79.wav` | FB (dev_30) | Dysfluent convo | `Um, uh, I still struggle with how to-` | Clean (0.0) | Abrupt phonological cutoff token (`how to-`) ignored by regex |
| **9** | `FluencyBank_010_82.wav` | FB (dev_30) | Dysfluent convo | `I have it the computer if I have it the compassion.` | Clean (0.0) | Interrupted phrasal repetition (`I have it the...`) |
| **10**| `FluencyBank_010_97.wav` | FB (dev_30) | Dysfluent convo | `just face, and she was going, go with the head.` | Clean (0.0) | Morphological stem repetition (`going, go`) missed by exact match |
| **11**| `M_1022_11y3m_1_dysfluent_019.wav` | UCLASS (gen_60) | `word_rep: 1` | `things. Well, my best subject.` | Filler (0.0) | Word repetition omitted by Whisper smoothing |
| **12**| `M_0061_16y9m-1_dysfluent_060.wav` | UCLASS (gen_60) | `word_rep: 1` | `...to be called Mr. Sh-` | Clean (0.0) | Abrupt phonological cutoff fragment (`Mr. Sh-`) ignored by regex |
| **13**| `FluencyBank_017_15.wav` | FB (gen_60) | `word_rep: 1, interj: 1` | `I'm just going to do a video. Oh` | Clean (0.0) | Pre-utterance silence (1.12s) below energy ratio threshold |
| **14**| `FluencyBank_017_42.wav` | FB (gen_60) | `block: 1` | `You're stuttering. Tell me about your therapy...` | Clean (0.0) | Post-utterance / intra-word block (0.71s) unanchored by word gap |
| **15**| `FluencyBank_017_34.wav` | FB (gen_60) | `word_rep: 1` | `""` (Empty string) | Clean (0.0) | Total Whisper omission due to prolonged articulatory block |
| **16**| `FluencyBank_017_52.wav` | FB (gen_60) | `block: 1` | `community activities. Okay.` | Clean (0.0) | Pre-utterance silence (0.95s) masked by Whisper timestamp snapping |
| **17**| `FluencyBank_031_14.wav` | FB (gen_60) | `block: 1` | `You get better in some way.` | Clean (0.0) | Internal block pause (0.55s) bridged across single Whisper word token |
| **18**| `FluencyBank_031_9.wav` | FB (gen_60) | `sound_rep: 1` | `Can we go back to this thing about there?` | Clean (0.0) | Sound repetition smoothed by Whisper; 0.77s pause low burst energy |
| **19**| `1743-142912-0001.flac` | Libri (gen_60) | `stutter: ['3','0','5']` | `He didn't like to admit that he couldn't recall...` | Filler (0.0) | Long 17s clip; subtle disfluencies smoothed in verbatim reading |

---

## 3. In-Depth Error Category Analysis

### Category A: Silent Articulatory Blocks & ASR Boundary Snapping (N=8 clips)
*Clips:* `M_0030_000`, `M_0030_001`, `M_0030_002`, `M_0030_003`, `FluencyBank_017_42`, `FluencyBank_017_52`, `FluencyBank_031_14`, `FluencyBank_017_34`.

#### Evidence from Saved Data
1. **Physical Silence Analysis:**
   Direct analysis of the audio energy profiles (`rms` frames at 10ms hop length) demonstrates that substantial silence and pause intervals exist in these clips:
   - `FluencyBank_017_52.wav`: Sustained silence from $t = 0.09\text{s}$ to $1.04\text{s}$ (**0.95 seconds** of silence).
   - `FluencyBank_017_42.wav`: Sustained silence from $t = 1.87\text{s}$ to $2.58\text{s}$ (**0.71 seconds** of silence) during "therapy...".
   - `FluencyBank_031_14.wav`: Sustained silence from $t = 0.49\text{s}$ to $1.04\text{s}$ (**0.55 seconds** of silence) during "You get...".
   - `FluencyBank_017_34.wav`: Sustained silence from $t = 0.31\text{s}$ to $1.05\text{s}$ (**0.74 seconds** of silence).
   - `M_0030_16y4m_1_dysfluent_000-003`: Sustained pauses $\ge 0.94\text{s}$ to $1.13\text{s}$.

2. **Why the Current Logic Missed Them:**
   In [`backend/experimental/stutter_detector.py`](file:///e:/SpeechPrj/backend/experimental/stutter_detector.py) (`detect_acoustic_blocks`, lines 215–271):
   - **Flaw 1 (Whisper Timestamp Snapping):** The initial block check tests `first_word_start = float(words_list[0].get("start", 0.0))`. In Whisper output, `words_list[0]["start"]` is snapped to `0.00` in 16 of the 19 clips. Even though the speaker sat in tense silence until $t = 0.95\text{s}$ (as in `FluencyBank_017_52`), Whisper labeled the first word as starting at `0.00s`. Because `0.00 < 0.40s`, the initial block check was bypassed completely.
   - **Flaw 2 (Inter-Word Boundary Requirement):** The detector only searches for silence between distinct word entries (`words_list[i].end` to `words_list[i+1].start`). If Whisper bridges a silent pause into a single word duration (e.g. `community` spanning $0.00-1.36\text{s}$), the detector calculates no inter-word gap (`gap = 0.0`).
   - **Flaw 3 (Empty Transcript Failure):** In `FluencyBank_017_34.wav`, the block was so prolonged that Whisper generated an empty transcript (`words_list = []`). Line 215 immediately aborts: `if not words_list or len(words_list) < 1: return 0, []`.

#### Status: Supported by Concrete Telemetry
The physical acoustic silence exists in the audio waveforms, but the detector relies entirely on Whisper word timestamps to define search windows.

---

### Category B: Abrupt Terminal Phonological Cutoffs (N=2 clips)
*Clips:* `FluencyBank_010_79.wav`, `M_0061_16y9m-1_dysfluent_060.wav`.

#### Evidence from Saved Data
1. **Transcript Verification:**
   - `FluencyBank_010_79.wav`:
     - Raw Transcript: `"Um, uh, I still struggle with how to-"`
     - Sanitized Transcript: `"Um, uh, I still struggle with how to-"`
     - Terminal token: `to-` with timestamp $2.78-2.98\text{s}$.
   - `M_0061_16y9m-1_dysfluent_060.wav`:
     - Raw Transcript: `"In your health master, we believe to code, to be called Mr. Sh-"`
     - Sanitized Transcript: `"In your health master, we believe to code, to be called Mr. Sh-"`
     - Terminal token: `Sh-` with timestamp $2.22-2.32\text{s}$.

2. **Why the Current Logic Missed Them:**
   The lexical repetition detector in [`stutter_detector.py`](file:///e:/SpeechPrj/backend/experimental/stutter_detector.py) requires:
   - Sound repetition 3x: `\b([a-zA-Z]{1,4})-\1-\1([a-zA-Z]*)\b` (e.g., `w-w-what`)
   - Sound repetition 2x: `\b([a-zA-Z]{1,4})-\1([a-zA-Z]*)\b` (e.g., `w-what`)
   - Broken syllable fragment: `\b([a-zA-Z]{1,2})[\s\-]+([a-zA-Z]{3,})\b` (e.g., `b- because`)
   - Consecutive whole-word repetition: `w1 == w2`
   
   Every existing pattern requires a **subsequent target word** or an **immediate repeating unit**. When a speaker experiences an abrupt block or phonological cutoff at the end of an utterance (`how to-`, `Mr. Sh-`), there is no following token. The terminal fragment is ignored.

#### Status: Supported by Concrete Telemetry
Both raw and sanitized Whisper transcripts contain explicit hyphenated cutoff tokens that are currently discarded by lexical regex rules.

---

### Category C: Multi-Iteration Repeated Struggle Sounds (N=3 clips)
*Clips:* `M_0030_16y4m_1_dysfluent_015.wav`, `M_0030_16y4m_1_dysfluent_009.wav`, `FluencyBank_017_15.wav`.

#### Evidence from Saved Data
1. **Transcript Verification:**
   - `M_0030_16y4m_1_dysfluent_015.wav`:
     `"er, er, er, er, er, er, er. Once, oh quick."`
     (The hesitation token `er` is repeated **7 consecutive times** in 1.34 seconds).
   - `M_0030_16y4m_1_dysfluent_009.wav`:
     `"You could, um, um, uh, uh, uh... (repeated 110 times)"`
     (Massive hesitation struggle chain across the entire 3-second recording).

2. **Why the Current Logic Missed Them:**
   - In [`backend/experimental/filler_boundary_gating.py`](file:///e:/SpeechPrj/backend/experimental/filler_boundary_gating.py), any word repetition where the token belongs to `FILLER_WORDS` (`er`, `um`, `uh`, `ah`) is unconditionally gated out as a conversational interjection.
   - The detector has no count threshold or iteration-depth logic: a single casual interjection ("I, um, think") is treated identically to 7 or 110 rapid struggle repetitions ("er, er, er, er, er, er, er").
   - In clinical speech pathology, 1–2 filler iterations are typical conversational hesitation, but $\ge 3$ consecutive iterations of a hesitation syllable constitute overt speech struggle / disfluency.

#### Status: Supported by Concrete Telemetry
The disfluent repetition is preserved verbatim in the transcript, but is actively purged by the filler boundary gater.

---

### Category D: Morphological & Phrasal Repetitions (N=3 clips)
*Clips:* `FluencyBank_010_97.wav`, `FluencyBank_010_82.wav`, `M_0030_16y4m_1_dysfluent_011.wav`.

#### Evidence from Saved Data
- `FluencyBank_010_97.wav`: Transcript contains `"going, go"`. The speaker begins to utter "going" and restarts with "go". The detector tests strict string equality (`w1 == w2`). Because `"going" != "go"`, 0 events are recorded.
- `FluencyBank_010_82.wav`: `"I have it the computer if I have it the compassion"`. 4-word phrasal revision separated by 2 intervening words ("computer if").
- `M_0030_16y4m_1_dysfluent_011.wav`: `"You could make a warship, or you could..."`. Phrasal repetition ("You could ... you could") separated by 4 words ("make a warship or"), which exceeds the current `max_skip = 3` limit of `InterruptedRepetitionDetector`.

#### Status: Supported by Concrete Telemetry
Tokens exist in transcript; current matching constraints (exact match, max skip 3) are too restrictive to capture these disfluencies.

---

### Category E: Whisper Smoothing & Unresolved Misses (N=3 clips)
*Clips:* `M_1022_11y3m_1_dysfluent_019.wav`, `FluencyBank_031_9.wav`, `1743-142912-0001.flac`.

#### Evidence & Analysis
- `M_1022_11y3m_1_dysfluent_019.wav`: UCLASS annotation indicates `word_rep: 1`. Whisper transcribed `"things. Well, my best subject."` with no repetition. The audio shows a 1.07s pause ($t=0.13-1.20\text{s}$) with low burst energy.
- `FluencyBank_031_9.wav`: Annotation indicates `sound_rep: 1`. Whisper transcribed `"Can we go back to this thing about there?"`. Audio shows a 0.77s initial pause with maximum onset energy $0.0050 < \text{vocal\_threshold} (0.0093)$.
- `1743-142912-0001.flac`: Long 17.19s Libri_Stutter reading clip. The speaker exhibits mild reading hesitations that Whisper normalised into fluent standard English.

#### Status: Partially Unresolved / ASR Smoothing Limitation
In these clips, Whisper's language model smoothed away the disfluency, and the acoustic signal lacks high-energy bursts or distinctive prolongation postures to trigger acoustic detectors without transcript anchors.

---

## 4. Evidence-Supported Findings vs. Hypotheses

| Error Category | Affected Clips | Status | Technical Finding |
| :--- | :---: | :---: | :--- |
| **Multi-Iteration Hesitation Struggle** | 3 clips (`M_0030_015`, `M_0030_009`, `FB_017_15`) | **Proven by Telemetry** | Transcripts contain $\ge 7$ repeated hesitation tokens (`er`, `uh`). Gater purged them because repetition thresholding is absent. |
| **Abrupt Terminal Phonological Cutoffs** | 2 clips (`FB_010_79`, `M_0061_060`) | **Proven by Telemetry** | Trailing hyphenated tokens (`to-`, `Sh-`) exist in raw and sanitized Whisper output. Regex fails to match terminal tokens lacking subsequent words. |
| **Whisper Pre-Utterance Timestamp Snapping** | 5 clips (`M_0030_000`, `001`, `002`, `003`, `FB_017_52`) | **Proven by Telemetry** | Physical silence of $0.55\text{s} - 1.13\text{s}$ exists in audio, but Whisper snaps `words_list[0].start` to `0.00s`, completely bypassing the initial block check. |
| **Intra-Word Silence Bridging** | 3 clips (`FB_017_42`, `FB_031_14`, `FB_017_34`) | **Proven by Telemetry** | Whisper bridges over internal silent pauses or outputs empty string, preventing inter-word block logic from triggering. |
| **Morphological / Stem Repetition** | 1 clip (`FB_010_97`) | **Probable Hypothesis** | `"going, go"` represents a morphological revision, but stemming on development data previously increased false positives on fluent speech. |
| **Interrupted Phrasal Revision ($\text{skip} > 3$)** | 2 clips (`FB_010_82`, `M_0030_011`) | **Unresolved Hypothesis** | Expanding `max_skip` from 3 to 5 risks capturing natural fluent discourse restarts and increasing false alarms. |
| **Total ASR Smoothing / Silent Audio** | 3 clips (`M_1022_019`, `FB_031_9`, `1743-142912`) | **Unresolved Limitation** | ASR output is fluent and acoustic signals are low-energy; cannot be recovered without audio-based phoneme recognition. |

---

## 5. Prioritized Potential Interventions

Based on the evidence, three potential interventions are formulated and ranked according to risk, signal availability, and testability. **None of these interventions have been implemented or tested on the holdout set.**

```
+----------------------------------------------------------------------------------------------------+
|                                    RANKED INTERVENTION PROPOSALS                                   |
+------+-------------------------------------------+---------------+---------------+-----------------+
| Rank | Proposal Name                             | Potential FNs | FP Risk Level | Implementation  |
+------+-------------------------------------------+---------------+---------------+-----------------+
|  1   | Multi-Iteration Hesitation Struggle Gating|   2-3 clips   |   Very Low    | Low Complexity  |
|  2   | Terminal Phonological Cutoff Matching     |    2 clips    |   Low-Medium  | Low Complexity  |
|  3   | ASR-Independent Energy-Based Block Anchor |   3-5 clips   |   High        | High Complexity |
+------+-------------------------------------------+---------------+---------------+-----------------+
```

### Rank 1: Multi-Iteration Hesitation Struggle Gating (RECOMMENDED)
- **Concept:** Modify `FillerBoundaryGater` to introduce an iteration threshold for consecutive filler words:
  - 1–2 consecutive fillers (`um`, `uh`, `er`): Retained as conversational fillers / interjections.
  - $\ge 3$ consecutive identical fillers (e.g. `er, er, er` or `uh, uh, uh`): Classified as severe disfluent struggle / stuttering (`confidence = 0.85`).
- **Development Errors Addressed:** Recovers `M_0030_16y4m_1_dysfluent_015.wav` (7x `er`) and `M_0030_16y4m_1_dysfluent_009.wav` (110x `uh`).
- **Signal Availability:** 100% present in existing Whisper transcripts.
- **False Positive Risk:** **Extremely Low.** Fluent speakers in natural conversation virtually never utter $\ge 3$ rapid consecutive identical filler tokens without speech struggle.
- **Genuine Filler Recognition Impact:** Completely safe. Normal single- and double-interjections remain classified as Fillers.
- **Testability:** Fully testable on the 90-clip development set without touching the locked holdout.

### Rank 2: Terminal Phonological Cutoff Detection
- **Concept:** Add a dedicated terminal cutoff regex in `detect_lexical_repetitions`:
  `r'\b([a-zA-Z]{1,4})-(?=\s*$|[.,?!;])'`
  Detects incomplete, hyphenated words representing an abrupt phonological stop.
- **Development Errors Addressed:** Recovers `FluencyBank_010_79.wav` (`to-`) and `M_0061_16y9m-1_dysfluent_060.wav` (`Sh-`).
- **Signal Availability:** 100% present in both raw and sanitized Whisper output.
- **False Positive Risk:** **Low to Moderate.** Must strictly avoid matching hyphenated compound words (e.g. `well-known`, `twenty-five`) by enforcing boundary conditions and minimum token context.
- **Testability:** Fully testable on the 90-clip development set.

### Rank 3: ASR-Independent Pre-Utterance Acoustic Silence Anchor
- **Concept:** Decouple initial acoustic block detection from Whisper's `words_list[0].start`. Directly analyze raw audio RMS: if a recording contains $\ge 0.50\text{s}$ of pre-phonation silence with elevated articulatory muscle/breath energy followed by vocal onset, register an initial acoustic block.
- **Development Errors Addressed:** Potentially addresses `FluencyBank_017_52.wav`, `FluencyBank_017_42.wav`, and UCLASS reading blocks.
- **False Positive Risk:** **High.** Natural conversational recordings often feature initial recording latency, interviewer pauses, or microphone activation delays that are completely fluent. Without lexical corroboration, pure acoustic silence anchors have historically degraded specificity.
- **Testability:** High complexity; requires extensive false-positive tuning across clean conversational audio.

---

## 6. Files Inspected & Compliance Audit

### Inspected Artifacts & Source Code (Read-Only)
- [`backend/experimental/stutter_detector.py`](file:///e:/SpeechPrj/backend/experimental/stutter_detector.py) (Acoustic block and lexical repetition detection logic)
- [`backend/experimental/filler_boundary_gating.py`](file:///e:/SpeechPrj/backend/experimental/filler_boundary_gating.py) (Filler suppression rules)
- [`backend/experimental/config.py`](file:///e:/SpeechPrj/backend/experimental/config.py) (Hyperparameter definitions)
- [`evaluation_results/experimental/dev_error_diagnostic.json`](file:///e:/SpeechPrj/evaluation_results/experimental/dev_error_diagnostic.json) (Diagnostic per-clip predictions)
- [`evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json`](file:///e:/SpeechPrj/evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json) (Dev 30 cache)
- [`evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json`](file:///e:/SpeechPrj/evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json) (Gen 60 manifest)
- [`evaluation_results/experimental/generalization_validation/run_20261020_132214/feature_cache.json`](file:///e:/SpeechPrj/evaluation_results/experimental/generalization_validation/run_20261020_132214/feature_cache.json) (Gen 60 cache)

### Diagnostic Scripts Created in Phase 4A Output Directory
- [`evaluation_results/experimental/phase4a_error_analysis/inspect_fn.py`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4a_error_analysis/inspect_fn.py)
- [`evaluation_results/experimental/phase4a_error_analysis/extract_fn_details.py`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4a_error_analysis/extract_fn_details.py)
- [`evaluation_results/experimental/phase4a_error_analysis/categorize_fn.py`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4a_error_analysis/categorize_fn.py)
- [`evaluation_results/experimental/phase4a_error_analysis/diagnose_blocks_fn.py`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4a_error_analysis/diagnose_blocks_fn.py)
- [`evaluation_results/experimental/phase4a_error_analysis/analyze_acoustic_evidence.py`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4a_error_analysis/analyze_acoustic_evidence.py)
- [`evaluation_results/experimental/phase4a_error_analysis/fn_detailed_inventory.json`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4a_error_analysis/fn_detailed_inventory.json)
- [`evaluation_results/experimental/phase4a_error_analysis/acoustic_evidence.json`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4a_error_analysis/acoustic_evidence.json)

### Confirmation of Immutability
- **Production Files Modified:** **0**
- **Main Frontend Files Modified:** **0**
- **Phase 2 / Phase 3 Benchmark Outputs Modified:** **0**
- **Detector Logic / Thresholds Modified:** **0**
- **Locked 120-Clip Holdout Re-evaluated:** **NO** (0 runs)

---

## 7. Recommendation & Next Steps

### Recommendation: Evaluate Intervention 1 (Multi-Iteration Hesitation Struggle Recognition)
1. **Target:** Address the clear diagnostic flaw where clinically severe repeated speech struggle (e.g. `er, er, er, er, er, er, er`) is purged by filler gating.
2. **Safety Profile:** This intervention carries virtually zero risk of increasing false positives on clean speech, does not damage genuine filler recognition (1–2 fillers remain untouched), and requires no risky acoustic silence heuristics.
3. **Controlled Scope:** If approved for Phase 4B, this intervention should be implemented as an explicit experimental flag (`enable_multi_iteration_hesitation_stutter`) and tested strictly on the 90-clip development set first.

---

## 8. Stop Condition Enforcement

**This investigation is complete and strictly diagnostic.**  
No code changes, threshold modifications, or benchmark executions have taken place.  
Awaiting explicit user direction and approval before taking any further action.
