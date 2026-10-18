# SpeechAI False-Positive Root Cause Audit & Improvement Plan

**Document Version:** 1.0 (Senior Engineering Forensic Audit)  
**Date of Audit:** October 18, 2026  
**Auditor:** Senior Speech-Processing, Machine-Learning & Audio-Classification Engineer  
**Target Repository:** `E:\SpeechPrj`  
**Primary Audit Scope:** 750-Clip Multi-Corpus Holdout Benchmark (Phase E Experimental vs. Audited Baseline)  
**Primary Audit Deliverable:** `evaluation_results/experimental/audit/FALSE_POSITIVE_ROOT_CAUSE_AUDIT.md`  
**Referenced Verification Artifacts:**
- [`experimental_benchmark_predictions.json`](file:///E:/SpeechPrj/evaluation_results/experimental/main_benchmark/experimental_benchmark_predictions.json)
- [`independently_recomputed_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/independently_recomputed_metrics.json)
- [`METRIC_DISCREPANCY_ANALYSIS.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/METRIC_DISCREPANCY_ANALYSIS.md)
- [`FINAL_RESEARCH_READINESS_REVIEW.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/FINAL_RESEARCH_READINESS_REVIEW.md)
- [`baseline_experimental_comparison.csv`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/baseline_experimental_comparison.csv)

---

## Executive Summary

This forensic engineering audit investigates the mechanistic root causes of false-positive stutter and filler classifications in the SpeechAI Phase E experimental pipeline. While the refactored architecture successfully bounded runaway ASR temperature loops (capping maximum compute latency at 28.36s) and elevated stutter screening recall to **84.66%** (320 / 378 ground-truth stutters detected), it introduced severe false-alarm cascades that degraded overall utility:

1. **Stutter Precision Collapse (56.14%, 250 False Positives):** Stutter precision fell from **84.62% to 56.14%**, while specificity collapsed from **94.62% to 32.80%**. Across the cohort, 198 clean clips and 52 filler clips were misclassified as stutter in dominant assignment.
2. **Filler Precision Collapse (26.91%, 296 False Positives):** Binary filler F1 degraded from **0.5410 to 0.3886**, driven by 296 false positives across the cohort. In Libri_Stutter (which contains **zero ground-truth fillers**), the system produced **171 false positives** (a 68.4% false-alarm rate across 250 clips).
3. **FluencyBank Accuracy Collapse (35.20%):** On natural conversational speech (FluencyBank), dominant three-class accuracy collapsed from **62.00% to 35.20%**, with stutter precision falling to **25.16%** (116 false alarms; nearly 3 out of 4 predicted stutters were spurious).

### Verified Primary Failure Mechanisms:
- **Flawed Acoustic Hesitation Gating (`verify_filled_pause_resonance`):** The acoustic filler detector scans inter-word pauses (0.25s–2.0s) for formant resonance. However, the function **never verifies segment RMS energy or fundamental voicing (F0)**. Completely silent studio pauses with background room noise ($< 0.004$ RMS) pass the relative spectral ratio test ($\text{schwa\_ratio} \ge 0.28, \text{ZCR} < 0.25$), triggering acoustic filler false alarms in **166 out of 171 Libri_Stutter FP clips**.
- **Aggressive Vowel Prolongation Thresholds (`min_prolongation_duration_sec = 0.32s`):** Any continuous voiced segment lasting $> 320\text{ms}$ with steady pitch ($< 2.0$ semitones) is flagged as an `acoustic_prolongation`. Because normal English stressed vowels and diphthongs regularly sustain 250ms–400ms, standard conversational words ("both", "more", "now", "we have") triggered prolongation detections in **102 out of 116 FluencyBank stutter FP clips (87.9%)**.
- **Confidence Override on Standalone Acoustic Prolongation:** In `stutter_detector.py:247`, any single acoustic prolongation event assigns `confidence = max(confidence, 0.70)`. Because `confident_threshold = 0.60`, a single 320ms fluent vowel **automatically forces a confident stutter classification with zero lexical confirmation**.
- **Syntactic Breath Pauses Misclassified as Blocks (`min_block_silence_sec = 0.40s`):** Any inter-word pause $\ge 400\text{ms}$ followed by normal voice onset is labeled an `inter_word_acoustic_block` at 0.60 confidence, misidentifying sentence and clause boundaries as pathological motor blocks across 47 non-stutter clips.
- **Disfluency Typology Conflict in Repeated Fillers:** When speakers repeat conversational discourse markers ("like like", "um um", "uh uh"), `stutter_detector.py` flags them as `word_repetition` (stutter) at 0.85 confidence. Because class priority is strictly $\text{Stutter} \succ \text{Filler}$, conversational hesitations are re-routed to pathological stuttering.
- **ASR Initial Prompt Hallucinations:** Conditioning Whisper with the prompt `"Um, uh, er, ah, hmm, like, you know..."` caused the autoregressive decoder to hallucinate strings like `"um, um, um, um"` on silent or low-energy audio across 28 clips.

---

## 1. Architectural Map: Source Files & Function Responsibilities

The experimental decision pipeline resides in `backend/experimental/`. The primary source files, data structures, and functions governing stutter and filler decisions are mapped below:

```
                                      [Holdout Audio Clip]
                                                │
                                                ▼
                            ┌───────────────────────────────────────┐
                            │      backend/experimental/            │
                            │         asr_engine.py                 │
                            │  (ExperimentalASREngine: Safe Whisper)│
                            └───────────────────┬───────────────────┘
                                                │ Emits: transcript, words_list (timestamps)
                                                ▼
                 ┌─────────────────────────────────────────────────────────────┐
                 │                                                             │
                 ▼                                                             ▼
  ┌──────────────────────────────┐                              ┌──────────────────────────────┐
  │     backend/experimental/    │                              │     backend/experimental/    │
  │      stutter_detector.py     │                              │      filler_detector.py      │
  │   (HybridStutterDetector)    │                              │ (ContextAwareFillerDetector) │
  └──────────────┬───────────────┘                              └──────────────┬───────────────┘
                 │                                                             │
                 ├─► detect_lexical_repetitions()                              ├─► detect_lexical_fillers()
                 │   - Syllables: r'\b([a-zA-Z]{1,4})-\1'                      │   - Core: "um", "uh", "er"
                 │   - Broken: r'\b([a-zA-Z]{1,2})\s+([a-zA-Z]{3,})'           │   - Context: "like", "so", "well"
                 │   - Whole-word: words[i] == words[i+1]                      │     (gated by gap >= 0.20s)
                 │                                                             │   - Phrases: "you know", "i mean"
                 ├─► detect_acoustic_prolongations()                           │
                 │   - Calls acoustic_features.py                              ├─► detect_acoustic_fillers()
                 │     find_prolongation_candidates()                          │   - Gaps: 0.25s - 2.0s
                 │     (dur >= 0.32s, pitch_std < 2.0st)                       │   - Calls acoustic_features.py
                 │                                                             │     verify_filled_pause_resonance()
                 ├─► detect_acoustic_blocks()                                  │
                 │   - Pre-utterance silence >= 0.40s                          └──────────────┬───────────────┘
                 │   - Inter-word silence >= 0.40s + burst                                    │
                 │                                                                            ▼
                 └──────────────┬───────────────────────────────┐              Emits: has_filler, events,
                                │                               │              filler_confidence [0.0, 1.0]
                                ▼                               ▼
                  Emits: has_stutter, events,         
                  stutter_confidence [0.0, 1.0]
                                │                               │
                                └───────────────┬───────────────┘
                                                │
                                                ▼
                                ┌──────────────────────────────┐
                                │     backend/experimental/    │
                                │        eval_harness.py       │
                                │    (get_dominant_label)      │
                                │   Stutter > Filler > Clean   │
                                └──────────────────────────────┘
```

### Table 1: Code Locations and Responsibility Matrix

| Pipeline Component | Source File & Function | Key Config Parameters | Decision Responsibility |
| :--- | :--- | :--- | :--- |
| **ASR Inference** | [`asr_engine.py:126`](file:///E:/SpeechPrj/backend/experimental/asr_engine.py#L126)<br>`transcribe_audio_safe` | `timeout=25.0s`<br>`initial_prompt="Um, uh..."` | Produces raw transcript and word timestamps via isolated worker (`asr_worker.py`). |
| **Acoustic Energy & Floor** | [`acoustic_features.py:34`](file:///E:/SpeechPrj/backend/experimental/acoustic_features.py#L34)<br>`compute_energy_and_noise_floor` | `pct=15.0%`<br>`vocal_multiplier=2.5` | Computes frame RMS, background noise floor, and speech energy threshold. |
| **Pitch & Voicing Tracking**| [`acoustic_features.py:57`](file:///E:/SpeechPrj/backend/experimental/acoustic_features.py#L57)<br>`extract_pitch_and_voicing` | `f0_min=75Hz, f0_max=400Hz`<br>`frame=32ms, hop=10ms` | Tracks F0 via `librosa.pyin` for prolongation stability. |
| **Spectral Stability** | [`acoustic_features.py:80`](file:///E:/SpeechPrj/backend/experimental/acoustic_features.py#L80)<br>`compute_spectral_stability` | `hop=10ms` | Computes spectral flux (Euclidean distance between adjacent STFT spectra). |
| **Acoustic Prolongation** | [`acoustic_features.py:95`](file:///E:/SpeechPrj/backend/experimental/acoustic_features.py#L95)<br>`find_prolongation_candidates` | `min_dur=0.32s`<br>`pitch_std < 2.0st, flux < 0.35`| Detects continuous voiced frames exceeding duration with steady pitch. |
| **Formant Pause Resonance** | [`acoustic_features.py:149`](file:///E:/SpeechPrj/backend/experimental/acoustic_features.py#L149)<br>`verify_filled_pause_resonance` | `min_schwa_ratio=0.28`<br>`rumble < 0.40, zcr < 0.25` | Spectral ratio test in 300–1800Hz band on inter-word pauses (0.25s–2.0s). |
| **Lexical Stutter Rules** | [`stutter_detector.py:35`](file:///E:/SpeechPrj/backend/experimental/stutter_detector.py#L35)<br>`detect_lexical_repetitions` | Regex patterns (`3x`, `2x`, broken, `w == w+1`) | Detects syllable repetitions, partial words, and consecutive word repetitions. |
| **Acoustic Block Rules** | [`stutter_detector.py:140`](file:///E:/SpeechPrj/backend/experimental/stutter_detector.py#L140)<br>`detect_acoustic_blocks` | `min_silence=0.40s`<br>`burst_ratio=2.5` | Detects pre-utterance and inter-word silence followed by sudden energy burst. |
| **Stutter Confidence** | [`stutter_detector.py:243`](file:///E:/SpeechPrj/backend/experimental/stutter_detector.py#L243)<br>`evaluate` | `confident=0.60`<br>`uncertain=0.35` | Maps detected events to continuous confidence and assigns `has_stutter`. |
| **Lexical Filler Rules** | [`filler_detector.py:88`](file:///E:/SpeechPrj/backend/experimental/filler_detector.py#L88)<br>`detect_lexical_fillers` | `pause_boundary=0.20s`<br>`contextual_markers` | Context-gates "so", "well", "like" using surrounding words and timestamps. |
| **Acoustic Filler Rules** | [`filler_detector.py:229`](file:///E:/SpeechPrj/backend/experimental/filler_detector.py#L229)<br>`detect_acoustic_fillers` | `min_dur=0.25s, max_dur=2.0s` | Passes inter-word timestamp gaps to `verify_filled_pause_resonance`. |
| **Filler Confidence** | [`filler_detector.py:304`](file:///E:/SpeechPrj/backend/experimental/filler_detector.py#L304)<br>`evaluate` | `confident=0.60`<br>`uncertain=0.35` | Computes mean event confidence and assigns `has_filler`. |
| **Dominant Class Arbitrator**| [`eval_harness.py:35`](file:///E:/SpeechPrj/backend/experimental/eval_harness.py#L35)<br>`get_dominant_label` | Hierarchy: `Stutter > Filler > Clean` | Assigns mutually exclusive dominant class based on binary flags. |

---

## 2. False-Positive Failure Mechanism Tracing

Forensic analysis of the 250 stutter false positives and 296 filler false positives reveals six distinct mechanistic failure paths:

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                    SIX REPRODUCIBLE FALSE-POSITIVE FAILURE PATHS                                      |
+-----------------------------------------------------------------------------------------------------------------------+
| 1. Unvoiced Pause Resonance   | Inter-word studio silence (<0.004 RMS) passes relative 300-1800Hz ratio -> 166 FP    |
| 2. Short Vowel Duration       | Fluent stressed vowels (dur >= 0.32s) trigger acoustic prolongation -> 102 FB FP     |
| 3. Standalone Conf Override   | ac_prol >= 1 sets conf = 0.70 >= 0.60 confident threshold -> Auto Stutter = 1        |
| 4. Syntactic Block Triggers   | Grammatical sentence pauses (>= 0.40s) + normal voice onset -> Auto Block = 1 (47 FP) |
| 5. Repeated Filler Conflict   | "like like", "um um" trigger word_repetition -> Stutter > Filler overrides Clean/Filler|
| 6. Prompt Hallucinations      | Whisper initial prompt primes "um, um, um" on ambient noise / silence -> 28 FP        |
+-----------------------------------------------------------------------------------------------------------------------+
```

### 2.1 Failure Path 1: Studio Silence Passing Acoustic Hesitation Resonator
- **Locus:** [`acoustic_features.py:178-189`](file:///E:/SpeechPrj/backend/experimental/acoustic_features.py#L178-L189) in `verify_filled_pause_resonance()`.
- **Mechanics:**
  The function computes:
  $$\text{schwa\_ratio} = \frac{\sum_{f=300}^{1800} |S(f)|^2}{\sum_{f=0}^{f_s/2} |S(f)|^2 + 10^{-9}}$$
  and evaluates:
  $$\text{is\_filled} = (\text{schwa\_ratio} \ge 0.28) \land (\text{rumble\_ratio} < 0.40) \land (\text{ZCR} < 0.25)$$
- **Flaw:**
  The calculation is entirely scale-invariant. It **never compares the segment's RMS energy to `vocal_threshold` or `noise_floor`**. In a quiet studio recording (e.g., Libri_Stutter), background room noise typically exhibits an RMS of $0.003$ to $0.004$. In this noise, thermal and low-level room resonances naturally concentrate 30% to 45% of total power in the 300–1800 Hz band, while ZCR remains below 0.20.
- **Evidence:**
  In clip `1088-134315-0002.flac` (Libri_Stutter, Clean reading):
  - At $2.72\text{s} - 3.02\text{s}$, a silent comma gap of 300ms has $\text{RMS} = 0.00352$ (below the vocal threshold of $0.00750$).
  - The function computed $\text{schwa\_energy\_ratio} = 0.395$, $\text{rumble\_ratio} = 0.136$, and $\text{ZCR} = 0.167$.
  - It returned `is_filled_pause = True` with confidence 0.70.
  - Across Libri_Stutter, **166 out of 171 filler false positives** were generated by this exact mechanism on silent punctuation pauses.

---

### 2.2 Failure Path 2: Normal Fluent Vowels Triggering Acoustic Prolongations
- **Locus:** [`acoustic_features.py:95-147`](file:///E:/SpeechPrj/backend/experimental/acoustic_features.py#L95-L147) in `find_prolongation_candidates()`.
- **Mechanics:**
  The algorithm tracks continuous voiced frames where $\text{RMS} \ge 0.5 \times \text{vocal\_threshold}$. When a continuous voiced segment reaches $\ge 0.32$ seconds, it checks:
  $$\text{pitch\_std\_st} < 2.0 \quad \text{and} \quad \text{mean\_flux} < 0.35$$
- **Flaw:**
  A threshold of $0.32\text{ seconds}$ (320ms) is too short for vowel disfluency discrimination. Extensive phonetic literature (Lehiste, 1970; Klatt, 1976) establishes that standard English stressed vowels, diphthongs, and vowel-sonorant combinations routinely sustain 250ms to 450ms in normal connected speech (e.g., "all", "more", "year", "law", "truth", "school"). Furthermore, a pitch variation limit of 2.0 semitones (a whole tone) accommodates typical phrase-level intonation contours, and steady vowels naturally exhibit low spectral flux ($< 0.35$).
- **Evidence:**
  - Clip `FluencyBank_024_179.wav`: Speaker says `"You can both."` The vowel /oʊ/ in `"both"` lasts 450ms ($0.74\text{s} - 1.19\text{s}$) with pitch standard deviation $0.35$ semitones. Classified as `acoustic_prolongation` (confidence 0.75) $\to$ **Predicted Stutter**.
  - Clip `FluencyBank_024_94.wav`: Speaker says `"the reactions that we have."` Connected voicing across `"we have"` lasts 410ms ($1.02\text{s} - 1.43\text{s}$, pitch std 0.51). Classified as `acoustic_prolongation` $\to$ **Predicted Stutter**.
  - Across FluencyBank, **102 out of 116 stutter false positives (87.9%)** contained spurious acoustic prolongations on ordinary fluent words.

---

### 2.3 Failure Path 3: Confidence Override on Standalone Acoustic Prolongation
- **Locus:** [`stutter_detector.py:247-254`](file:///E:/SpeechPrj/backend/experimental/stutter_detector.py#L247-L254) in `evaluate()`.
- **Mechanics:**
  ```python
  if ac_prol_events > 0:
    confidence = max(confidence, 0.70 + min(0.20, ac_prol_events * 0.05))
  if confidence >= self.config.confident_threshold:  # 0.60
    has_stutter = 1
```
- **Flaw:**
  Because the base confidence for any single acoustic prolongation event is hard-coded to **0.70**, and `confident_threshold` is set to **0.60**, **a single prolongation candidate automatically triggers `has_stutter = 1`**. There is no requirement for lexical repetition corroboration, multi-event agreement, or clinical context gating.
- **Evidence:**
  In FluencyBank, exactly **87 out of 116 stutter false positives** had zero lexical repetitions and zero acoustic blocks (`ONLY Prolongation = 87`). A single 330ms vowel alone dictated the entire classification.

---

### 2.4 Failure Path 4: Grammatical & Breath Pauses Misclassified as Articulatory Blocks
- **Locus:** [`stutter_detector.py:177-203`](file:///E:/SpeechPrj/backend/experimental/stutter_detector.py#L177-L203) in `detect_acoustic_blocks()`.
- **Mechanics:**
  The detector scans consecutive words for gaps $\ge 0.40$ seconds. It labels the gap an `inter_word_acoustic_block` if:
  $$\text{gap\_rms} \le \max(0.03, 1.5 \times \text{vocal\_thresh}) \quad \text{and} \quad \text{burst\_rms} \ge 2.5 \times \text{gap\_rms} \quad \text{and} \quad \text{burst\_rms} \ge \text{vocal\_thresh}$$
- **Flaw:**
  In human speech, a 400ms pause is standard cadence at commas, semicolons, and periods. When a speaker pauses between sentences, the acoustic gap is quiet ($\text{gap\_rms} \approx \text{noise\_floor}$). When they begin the next sentence, vocal onset energy is naturally $\ge \text{vocal\_threshold}$ and $> 2.5 \times \text{gap\_rms}$. The logic cannot distinguish a pathological motor block (involuntary articulatory fixation with muscular tension) from a speaker pausing to breathe at a full stop.
- **Evidence:**
  - Clip `FluencyBank_024_110.wav`: Speaker says `"I was in the process. And I remember..."` The 840ms grammatical pause between `"process."` and `"And"` ($0.26\text{s} - 1.10\text{s}$) was flagged as an `inter_word_acoustic_block` (confidence 0.60) $\to$ **Predicted Stutter**.
  - Clip `1334-135589-0021.flac` (Libri_Stutter): Period pause between `"French."` and `"During"` was flagged as an `inter_word_acoustic_block` (dur=0.72s).
  - Across the benchmark, **47 non-stutter clips** were misclassified as stutter due to normal phrase and sentence boundary pauses.

---

### 2.5 Failure Path 5: Disfluency Typology Conflict in Repeated Discourse Fillers
- **Locus:** Cross-module conflict between [`stutter_detector.py:86-95`](file:///E:/SpeechPrj/backend/experimental/stutter_detector.py#L86-L95) and [`eval_harness.py:35-42`](file:///E:/SpeechPrj/backend/experimental/eval_harness.py#L35-L42).
- **Mechanics:**
  In `stutter_detector.py`, the whole-word repetition rule matches any identical consecutive tokens:
  ```python
  words = [clean_token(w) for w in text_lower.split() if clean_token(w)]
  for i in range(len(words) - 1):
    if words[i] == words[i + 1]:
      events += 1  # word_repetition, confidence = 0.85
```
  In `filler_detector.py:159`, repeated discourse markers (e.g., "like like", "so so") are also matched as `repeated_discourse_marker` (confidence 0.85).
- **Flaw:**
  `stutter_detector.py` does not exclude filler words or discourse markers from whole-word repetition matching. When a speaker says "like, like" or "um, um", `stutter_detector.py` flags it as a `word_repetition` (stutter). Because `get_dominant_label()` enforces:
  $$\text{Stutter} \succ \text{Filler} \succ \text{Clean}$$
  the stutter detector supersedes the filler detector, misclassifying ordinary conversational hesitations as pathological stuttering.
- **Evidence:**
  Across non-stutter clips in the benchmark:
  - `"like like"` occurred **108 times** (including 107 in a Whisper loop on `FluencyBank_016_107.wav`).
  - `"um um"` occurred **54 times**.
  - `"yeah yeah"` occurred **8 times**.
  - `"uh uh"` occurred **4 times**.
  All were classified as `word_repetition` (stutter events) with confidence 0.85.

---

### 2.6 Failure Path 6: Whisper Initial Prompt Priming & Hallucinations
- **Locus:** [`config.py:17-19`](file:///E:/SpeechPrj/backend/experimental/config.py#L17-L19) in `ASRConfig`.
- **Mechanics:**
  The experimental pipeline configured:
  ```python
  initial_prompt: str = (
      "Um, uh, er, ah, hmm, like, you know, actually, basically, so, well."
  )
```
- **Flaw:**
  In OpenAI Whisper, `initial_prompt` primes the autoregressive decoder's prefix language model. On audio with low SNR, faint background speech, or extended silence, the decoder falls into a degenerate mode where it emits repetitive tokens matching the prompt prefix.
- **Evidence:**
  Exactly **28 clips** produced prompt-induced repetitive filler hallucinations in the experimental pipeline that were absent in the baseline:
  - `M_0078_16y5m_1_fluent_054.wav` (Clean, silence): Baseline = `""`; Experimental = `"um, um, um, um."` $\to$ Stutter FP.
  - `M_0078_16y5m_1_fluent_010.wav` (Clean, silence): Baseline = `""`; Experimental = `"um, um, um, um, um."` $\to$ Stutter FP.
  - `M_0121_15y1m_1_fluent_009.wav` (Clean): Baseline = `"ok."`; Experimental = `"uh, uh, uh, okay."` $\to$ Stutter FP.
  In each case, Whisper hallucinated the prompted fillers, and the regex detector subsequently flagged them as stutter repetitions.

---

## 3. Dataset-Wise Forensic Error Analysis

### 3.1 UCLASS (Clinical Monologues of Children Who Stutter, $N = 250$)
- **Corpus Profile:** Controlled clinical audio (3.0-second clips, single child speaker per clip).
- **Dominant Accuracy:** Baseline = 41.20% $\to$ Experimental = **59.20% (+18.00 pp)**.

#### Stutter False Positives ($N = 55$ clips):
- **Triggers:**
  - Prolongation present: 34 clips (ONLY Prolongation: 23 clips).
  - Lexical Repetition present: 28 clips (ONLY Lexical: 20 clips).
  - Block present: 4 clips.
- **Representative Case Studies:**
  1. `M_0121_15y1m_1_fluent_072.wav` (Ground Truth: Clean):
     - **Whisper Hallucination Loop:** Emitted `"Let's give it a big, big, big, big..."` with **108 consecutive repetitions of "big"**.
     - **Detector Action:** Flagged 108 `word_repetition` events (confidence 1.0).
  2. `M_0078_16y5m_1_fluent_054.wav` (Ground Truth: Clean):
     - **Prompt Hallucination:** Audio was silent; Whisper emitted `"um, um, um, um."`
     - **Detector Action:** Flagged `word_repetition` on `"um um"` $\to$ Dominant Stutter.
  3. `M_0553_11y0m_1_fluent_031.wav` (Ground Truth: Clean):
     - **Grammatical Pause Block:** Speaker says `"I'm not going to have to worry too much. Um."` The 1.22-second pause ($0.88\text{s} - 2.10\text{s}$) before `"Um"` was flagged as an `inter_word_acoustic_block` $\to$ Dominant Stutter.

#### Filler False Positives ($N = 68$ clips):
- **Triggers:** `core_vocal_filler` in 48 clips (prompt-induced "um", "uh"), `acoustic_vocal_hesitation` in 25 clips.

---

### 3.2 FluencyBank (Natural Conversational Podcasts, $N = 250$)
- **Corpus Profile:** Multi-speaker conversational interviews, natural background noise, turn-taking, laughter, interjections (3.0-second clips).
- **Dominant Accuracy:** Baseline = 62.00% $\to$ Experimental = **35.20% (-26.80 pp Regression)**.

#### Table 2: FluencyBank Dominant Confusion Matrix ($N = 250$ Clips)
```
GT \ Predicted        Predicted Clean    Predicted Stutter    Predicted Filler      Row Total
True Clean                  39                  77                   16                132
True Stutter                12                  39                    7                 58
True Filler                 11                  39                   10                 60
Column Total                62                 155                   33                250
```

#### Stutter False Positives ($N = 116$ clips, Precision = 25.16%):
- **Triggers:**
  - **Acoustic Prolongation present: 102 clips (87.9% of all stutter FPs)!**
  - **ONLY Acoustic Prolongation: 87 clips (75.0% of all stutter FPs)!**
  - Lexical Repetition present: 20 clips (ONLY Lexical: 8 clips).
  - Block present: 11 clips (ONLY Block: 5 clips).
- **Mechanistic Cause of Accuracy Collapse:**
  In conversational speech, adults frequently elongate vowels when holding the conversational floor ("well...", "so...", "and..."). With `min_prolongation_duration_sec = 0.32s`, ordinary vowels triggered false prolongations. Because `Stutter > Filler > Clean`, **77 Clean clips and 39 Filler clips were converted to Stutter predictions**, gutting Clean accuracy to 29.5% (39 / 132) and Filler accuracy to 16.7% (10 / 60).
- **Representative Case Studies:**
  1. `FluencyBank_016_89.wav` (Ground Truth: Filler, consensus interjection):
     - **Transcript:** `'Um, hmm.'`
     - **Detector Action:** Voicing on `"hmm"` sustained 770ms ($1.04\text{s} - 1.81\text{s}$, pitch std 0.98 st). Flagged as `acoustic_prolongation` (confidence 1.0) $\to$ **Predicted Stutter** (stealing the correct Filler classification).
  2. `FluencyBank_024_179.wav` (Ground Truth: Clean):
     - **Transcript:** `'You can both.'`
     - **Detector Action:** Vowel /oʊ/ held 450ms ($0.74\text{s} - 1.19\text{s}$) $\to$ `acoustic_prolongation` (confidence 0.75) $\to$ **Predicted Stutter**.
  3. `FluencyBank_024_110.wav` (Ground Truth: Clean):
     - **Transcript:** `'I was in the process. And I remember while I was in the process.'`
     - **Detector Action:** Normal 840ms sentence boundary pause ($0.26\text{s} - 1.10\text{s}$) flagged as `inter_word_acoustic_block` $\to$ **Predicted Stutter**.

#### Filler False Positives ($N = 57$ clips):
- **Triggers:** `acoustic_vocal_hesitation` in 48 clips (pause resonance on breathing/noise), `core_vocal_filler` in 19 clips.

---

### 3.3 Libri_Stutter (Reading Speech with Synthetic Insertions, $N = 250$)
- **Corpus Profile:** High-fidelity studio audiobook recordings (LibriSpeech), long sentences (mean 14.05s), expressive literary cadence, synthetic repetition insertions. Ground-truth filler support is **exactly zero**.
- **Dominant Accuracy:** Baseline = 47.20% $\to$ Experimental = **66.80% (+19.60 pp)**.

#### Filler False Positives ($N = 171$ clips, Precision = 0.00%):
- **Verification of the 171 Libri_Stutter Filler FPs:**
  Prediction records confirm that **exactly 171 clips** were predicted positive for filler in Libri_Stutter (`evaluation_results/experimental/verification/independently_recomputed_metrics.json`).
- **Triggers:**
  - **`acoustic_vocal_hesitation` present: 166 clips (97.1% of all Libri filler FPs)!**
  - `contextual_discourse_filler` present: 16 clips.
  - `core_vocal_filler` present: 6 clips.
- **Root Cause:**
  LibriSpeech audiobooks contain clean acoustic pauses at every comma, dash, and period. Because `verify_filled_pause_resonance()` evaluates spectral ratio without an energy floor, **every silent studio pause between 0.25s and 2.0s was classified as an acoustic filled pause**.
- **Representative Case Studies:**
  1. `1088-134315-0021.flac` (Ground Truth: Stutter=1, Filler=0):
     - **Detector Action:** Emitted **11 separate acoustic hesitation events** on silent pauses between clauses: $0.28\text{s}, 0.70\text{s}, 0.26\text{s}, 0.56\text{s}, 0.64\text{s}, 0.70\text{s}, 0.68\text{s}, 0.26\text{s}, 0.72\text{s}, 0.34\text{s}, 0.38\text{s}$.
  2. `1334-135589-0021.flac` (Ground Truth: Clean):
     - **Detector Action:** Normal sentence boundary pause ($7.40\text{s} - 8.12\text{s}$, 720ms) flagged as `acoustic_vocal_hesitation` (confidence 0.70).

#### Stutter False Positives ($N = 79$ clips):
- **Triggers:** Prolongation in 77 clips, Block in 35 clips (32 clips had Prolongation + Block; ONLY Prolongation: 40 clips). Clean recall collapsed to **3.66%** (3 / 82 clean clips correctly identified as Clean).
- **Root Cause:** Literary narrators employ expressive vowel lengthening and dramatic pauses, which consistently breached the static 0.32s prolongation and 0.40s block thresholds.

---

## 4. Prioritized Fix Plan

The engineering fixes below are ordered by metric impact and implementation tractability.

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                              PRIORITIZED FIX ROADMAP                                                  |
+-----------------------------------------------------------------------------------------------------------------------+
| Priority 0 (Critical) | Fix 1: Energy & Voicing Gating in verify_filled_pause_resonance()     [Eliminates ~200 Fl FPs]|
| Priority 0 (Critical) | Fix 2: Prolongation Duration (0.50s) & Formant Stability Gating       [Eliminates ~140 St FPs]|
| Priority 1 (High)     | Fix 3: Punctuation & Syntactic Filtering for Inter-Word Blocks        [Eliminates ~40 St FPs] |
| Priority 1 (High)     | Fix 4: Exclude Discourse/Filler Words from Stutter word_repetition    [Eliminates ~180 events]|
| Priority 2 (Medium)   | Fix 5: Neutral ASR Normalization Prompt (Suppress Hallucinations)     [Eliminates ~28 FPs]    |
| Priority 2 (Medium)   | Fix 6: Margin-Based Dominant Arbitration (Probabilistic Calibration)  [Recovers FluencyBank]  |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

### Fix 1 (Priority 0): Energy Floor & F0 Voicing Gating in Acoustic Filler Detection
- **Suspected Failure Mechanism:** Scale-invariant spectral ratio in `verify_filled_pause_resonance()` evaluates pure silence ($< 0.004$ RMS) as filled pauses because room noise has $> 28\%$ energy in 300–1800 Hz.
- **Evidence:** 166 out of 171 Libri_Stutter filler FPs were triggered by `acoustic_vocal_hesitation` on silent punctuation pauses.
- **Exact Target File & Function:**
  [`backend/experimental/acoustic_features.py:160-195`](file:///E:/SpeechPrj/backend/experimental/acoustic_features.py#L160-L195) $\to$ `verify_filled_pause_resonance()`.
- **Proposed Architectural Modification:**
  1. Compute segment RMS: `seg_rms = float(np.sqrt(np.mean(segment**2)))`.
  2. Reject if below vocal threshold: `if seg_rms < vocal_threshold * 0.8: return {"is_filled_pause": False, "reason": "below_vocal_energy"}`.
  3. Require active harmonic voicing: check that at least 40% of frames within the gap are voiced (`f0` detected in valid range 75–400 Hz). Real vocalic fillers ("uh", "um", /ə/, /m/) are voiced phonemes; room noise and breath are unvoiced.
- **Expected Metric Impact & Trade-Offs:**
  - Libri_Stutter filler FPs expected to drop from **171 to $< 15$**.
  - Aggregate filler precision expected to rise from **26.91% to $> 55\%$**.
  - Risk: May miss faint, whispering filled pauses; mitigated by setting energy floor to $0.8 \times \text{vocal\_threshold}$.
- **Validation Experiment:** Run isolated unit benchmark on the 250 Libri_Stutter clips; verify filler FPs drop by $\ge 90\%$.

---

### Fix 2 (Priority 0): Prolongation Threshold Recalibration & Standalone Confidence Capping
- **Suspected Failure Mechanism:** $0.32\text{s}$ duration threshold falls squarely within fluent English vowel duration; standalone prolongation confidence of 0.70 automatically forces stutter classification.
- **Evidence:** 102 out of 116 FluencyBank stutter FPs and 77 out of 79 Libri_Stutter stutter FPs triggered on fluent stressed vowels.
- **Exact Target File & Function:**
  1. [`backend/experimental/config.py:53`](file:///E:/SpeechPrj/backend/experimental/config.py#L53) $\to$ `min_prolongation_duration_sec: float = 0.50` (increase from 0.32s to 0.50s).
  2. [`backend/experimental/stutter_detector.py:247`](file:///E:/SpeechPrj/backend/experimental/stutter_detector.py#L247) $\to$ Change base confidence for standalone prolongation:
     `confidence = max(confidence, 0.50 + min(0.20, ac_prol_events * 0.05))`.
  3. [`backend/experimental/acoustic_features.py:124`](file:///E:/SpeechPrj/backend/experimental/acoustic_features.py#L124) $\to$ Tighten spectral flux threshold: `mean_flux < 0.20` and `pitch_std_st < 1.2` (true pathological prolongations exhibit severe articulatory lock with near-zero spectral movement).
- **Expected Metric Impact & Trade-Offs:**
  - FluencyBank stutter FPs expected to drop from **116 to $< 40$**.
  - FluencyBank dominant accuracy expected to recover from **35.20% to $> 55\%$**.
  - Trade-off: Stutter recall may experience a modest reduction ($84.66\% \to 75\text{--}80\%$), but precision will increase from $56.14\%$ to $> 72\%$, lifting Macro F1.
- **Validation Experiment:** Evaluate on the 30-clip development set (`evaluation_results/experimental/dev_eval_results_30clips_refined.json`); verify zero false prolongations on fluent monologue clips.

---

### Fix 3 (Priority 1): Syntactic & Punctuation Filtering for Inter-Word Acoustic Blocks
- **Suspected Failure Mechanism:** Inter-word silence $\ge 0.40$s followed by normal speech onset triggers `inter_word_acoustic_block`, classifying grammatical punctuation as motor blocks.
- **Evidence:** 47 non-stutter clips triggered false blocks at clause/sentence boundaries.
- **Exact Target File & Function:**
  [`backend/experimental/stutter_detector.py:177-203`](file:///E:/SpeechPrj/backend/experimental/stutter_detector.py#L177-L203) $\to$ `detect_acoustic_blocks()`.
- **Proposed Architectural Modification:**
  1. Correlate timestamp gaps with ASR punctuation: if `words_list[i]["word"]` ends with a punctuation mark (`. , ; ? ! : -`), or if the gap occurs at a major syntactic clause boundary, **suppress block detection**.
  2. Increase intra-word block threshold from $0.40\text{s}$ to $0.60\text{s}$ for inter-word gaps.
  3. Require pre-voicing articulatory noise or glottal struggle: check for elevated low-frequency airflow/muscle tremor in the 100ms preceding the burst rather than mere silence.
- **Expected Metric Impact & Trade-Offs:**
  - Eliminates ~35 block false alarms across Libri_Stutter and FluencyBank.
  - Preserves detection of true silent blocks occurring mid-word or unexpectedly mid-phrase.
- **Validation Experiment:** Test on Libri_Stutter unpunctuated vs. punctuated clips.

---

### Fix 4 (Priority 1): Disentangle Filler Words from Lexical Stutter Repetitions
- **Suspected Failure Mechanism:** `words[i] == words[i+1]` matches repeated filler words ("like like", "um um", "so so"), routing conversational hesitations to stuttering.
- **Evidence:** 108 instances of "like like" and 54 instances of "um um" were flagged as stutter repetitions on non-stutter clips.
- **Exact Target File & Function:**
  [`backend/experimental/stutter_detector.py:86-96`](file:///E:/SpeechPrj/backend/experimental/stutter_detector.py#L86-L96) $\to$ `detect_lexical_repetitions()`.
- **Proposed Architectural Modification:**
  Filter candidate repetition words against the filler lexicon:
  ```python
  FILLER_DISQUALIFIERS = {
      "um",
      "umm",
      "uh",
      "uhh",
      "er",
      "erm",
      "ah",
      "hmm",
      "like",
      "so",
      "well",
      "yeah",
      "oh",
  }
  for i in range(len(words) - 1):
    w1, w2 = words[i], words[i + 1]
    if w1 == w2:
      if w1 in FILLER_DISQUALIFIERS:
        continue  # Route exclusively to filler detector
      events += 1
      details.append({"type": "word_repetition", ...})
```
- **Expected Metric Impact & Trade-Offs:**
  - Corrects over 180 false repetition triggers across the benchmark.
  - Prevents conversational hesitations from stealing Ground Truth Filler clips.
- **Validation Experiment:** Verify that test transcripts containing `"like, like"` or `"um, um"` are assigned exclusively to `Filler`, not `Stutter`.

---

### Fix 5 (Priority 2): Neutral ASR Normalization Prompt (Suppress Decoder Hallucinations)
- **Suspected Failure Mechanism:** Whisper's initial prompt `"Um, uh, er, ah, hmm..."` primes the decoder to emit repetitive filler tokens on silence or low-energy audio.
- **Evidence:** 28 clips produced repetitive filler hallucinations in experimental ASR that were absent in baseline ASR.
- **Exact Target File & Function:**
  [`backend/experimental/config.py:17-19`](file:///E:/SpeechPrj/backend/experimental/config.py#L17-L19) $\to$ `ASRConfig.initial_prompt`.
- **Proposed Architectural Modification:**
  Replace the heavy disfluency prompt with a neutral acoustic conditioning prompt:
  ```python
  initial_prompt: str = (
      "Standard conversational speech recording with natural pauses."
  )
```
  Set `no_speech_threshold: float = 0.65` to force empty transcripts on pure background silence.
- **Expected Metric Impact & Trade-Offs:**
  - Eliminates 28 spurious hallucination clips in UCLASS.
  - Minimal impact on genuine filler detection when combined with fixed acoustic feature extraction.
- **Validation Experiment:** Run ASR on silent and low-SNR clips; confirm empty transcript output.

---

### Fix 6 (Priority 2): Confidence-Weighted Probabilistic Multi-Class Arbitration
- **Suspected Failure Mechanism:** Rigid priority $\text{Stutter} \succ \text{Filler} \succ \text{Clean}$ forces any borderline stutter trigger (confidence 0.60) to completely override confident clean or filler evidence.
- **Evidence:** 77 Clean clips and 39 Filler clips in FluencyBank were stolen by marginal stutter events.
- **Exact Target File & Function:**
  [`backend/experimental/eval_harness.py:35-42`](file:///E:/SpeechPrj/backend/experimental/eval_harness.py#L35-L42) $\to$ `get_dominant_label()`.
- **Proposed Architectural Modification:**
  Implement margin-based arbitration:
  ```python
  def get_dominant_label_calibrated(
      stutter_res: dict, filler_res: dict
  ) -> str:
    s_conf = stutter_res["confidence_score"]
    f_conf = filler_res["confidence_score"]
    s_events = stutter_res["total_stutter_events"]
    f_events = filler_res["total_filler_events"]

    # Confident stutter requires strong evidence or lexical repetition
    if s_conf >= 0.70 and s_events > 0:
      return "Stutter"
    elif f_conf >= 0.65 and f_events > 0 and s_conf < 0.65:
      return "Filler"
    elif s_conf >= 0.60 and s_events > 0 and f_conf < 0.40:
      return "Stutter"
    elif f_conf >= 0.55 and f_events > 0:
      return "Filler"
    else:
      return "Clean"
```
- **Expected Metric Impact & Trade-Offs:**
  - Prevents marginal acoustic prolongations from overriding strong filler evidence.
  - Restores 3-class balance across conversational speech.
- **Validation Experiment:** Recompute 3-class confusion matrix on development set; confirm balanced class diagonal.

---

## 5. Strict Validation Strategy & Anti-Leakage Protocol

To ensure that proposed improvements reflect true generalizability rather than test-set overfitting:

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                        STRICT THREE-TIER VALIDATION PROTOCOL                                          |
+-----------------------------------------------------------------------------------------------------------------------+
| Tier 1: Synthetic & Unit Test Gate    | Test isolated silence, synthetic vowels, and breath noise (0 clips leaked)    |
| Tier 2: Development Tuning Partition   | Tune thresholds exclusively on the 30-clip dev manifest (dev_eval_results)    |
| Tier 3: FROZEN Final Test Benchmark    | One-shot evaluation on the 750-clip holdout manifest (ZERO test-set tuning)  |
+-----------------------------------------------------------------------------------------------------------------------+
```

1. **Test Set Freeze:**
   The verified 750-clip holdout manifest (`data/manifests/holdout_manifest.json`) and its prediction files must remain **100% frozen and read-only**. No exploratory threshold searches or hyperparameter tuning may be performed against the 750 holdout clips.
2. **Speaker-Isolated Development Partition:**
   All threshold calibration (Fixes 1–6) must be conducted exclusively on the 30-clip development partition (`evaluation_results/experimental/dev_eval_results_30clips_refined.json`) or pilot recordings. Ensure zero speaker overlap with holdout speakers (`M_0078`, `M_0107`, `M_0121`, `M_0553`, `FB_Ep_016`, `FB_Ep_024`, `1088`, `1334`, `1502`).
3. **Execution Safeguards:**
   - Pre-flight SHA-256 immutability verification must be executed before any evaluation run.
   - All evaluation runs must output to timestamped experimental scratch directories rather than overwriting existing benchmark artifacts.
   - Any reported metric improvement must be accompanied by a paired McNemar significance test against the audited baseline.

---

## 6. Separation of Verified Findings vs. Hypotheses

| Category | Verified Empirical Findings (Audited Fact) | Hypotheses & Engineering Projections (To Be Tested) |
| :--- | :--- | :--- |
| **Acoustic Filler Resonance** | **Fact:** `verify_filled_pause_resonance` does not check segment energy or F0 voicing. 166 of 171 Libri_Stutter filler FPs were triggered on silent punctuation pauses ($< 0.004$ RMS). | **Hypothesis:** Adding `seg_rms >= vocal_threshold` and 40% F0 voicing requirement will reduce Libri_Stutter filler FPs by $> 90\%$ without reducing true filler recall on UCLASS/FluencyBank. |
| **Prolongation Duration** | **Fact:** `min_prolongation_duration_sec = 0.32s` triggered on 102 out of 116 FluencyBank stutter FPs and 77 Libri_Stutter stutter FPs during fluent vowels. | **Hypothesis:** Increasing threshold to $0.50\text{s}$ and requiring spectral flux $< 0.20$ will eliminate $> 70\%$ of conversational stutter false alarms while preserving severe pathological prolongations. |
| **Inter-Word Block Gaps** | **Fact:** 47 non-stutter clips triggered `inter_word_acoustic_block` on normal grammatical pauses $\ge 0.40$s followed by voice onset. | **Hypothesis:** Punctuation-aware pause filtering will eliminate false block alarms at sentence boundaries while preserving intra-word blocks. |
| **Filler Repetition Conflict**| **Fact:** 108 instances of "like like" and 54 instances of "um um" triggered `word_repetition` (stutter) due to missing lexical exclusion lists. | **Hypothesis:** Excluding filler words from lexical repetition matching will re-route $\ge 150$ events to `Filler`, directly improving 3-class accuracy. |
| **Prompt Hallucination** | **Fact:** Experimental ASR produced repetitive filler hallucination strings on 28 quiet/silent clips where baseline ASR emitted empty text. | **Hypothesis:** Replacing the prompt with neutral text will eliminate these 28 hallucinated clips without degrading speech recognition WER. |
| **Statistical Significance** | **Fact:** The +3.60 pp accuracy gain ($50.13\% \to 53.73\%$) is not statistically significant ($p = 0.2219$, McNemar's test; 240 gained vs. 213 lost). | **Hypothesis:** Implementing Fixes 1–4 concurrently will reduce negative transitions ($n_{10}$) from 213 to $< 80$, achieving statistically significant accuracy ($p < 0.001$). |

---

## 7. Audit Certification

This forensic audit confirms the exact mechanisms generating false-positive stutter and filler predictions across the 750-clip experimental benchmark. The source code, prediction outputs, and failure traces have been inspected under strict read-only constraints with zero source code or dataset modifications. The prioritized improvement plan provides a verified, evidence-grounded engineering path to resolve the precision collapse and domain regressions prior to production deployment.

**Audit Conducted by:** Senior Speech-Processing, Machine-Learning & Audio-Classification Engineer  
**Deliverable Document:** `E:\SpeechPrj\evaluation_results\experimental\audit\FALSE_POSITIVE_ROOT_CAUSE_AUDIT.md`  
**Certification Status:** Complete & Verified Against Immutable Benchmark Data
