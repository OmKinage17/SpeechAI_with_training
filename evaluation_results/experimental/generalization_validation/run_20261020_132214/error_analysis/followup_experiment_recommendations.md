# SpeechAI — Follow-up Experiment Recommendations (Ablation 1 Generalization Audit)

**Reference Run:** `evaluation_results/experimental/generalization_validation/run_20261020_132214/`  
**Evaluated Pipeline:** Fix 3 + Combined 2-gram/3-gram Repetition Detection  
**Context:** Diagnostic Audit of 13 False Negatives and 10 False Positives on 60 Independent Generalization Clips  
**Status:** Recommendations Only — Strictly Non-Executable Diagnostic Planning  

---

## 1. Overview & Prioritization Strategy

Following the diagnostic audit of the 60-clip generalization run (`run_20261020_132214`), the pipeline exhibits two distinct error bottlenecks:
1. **Precision Ceiling (68.75% Precision, 10 False Positives):** 40% of false positives are caused by conversational filler vocalizations (`"um"`, `"uh"`), and 30% are caused by normal syntactic/boundary pauses mistaken for blocks.
2. **Recall Ceiling (62.86% Recall, 13 False Negatives):** In clinical stuttering (FluencyBank), 3 clips are missed due to undetectable silent blocks. In conversational speech, 3 clips are missed because repeated phrases are interrupted by filler interjections, breaking strict token contiguity.

To maximize clinical diagnostic reliability while avoiding pipeline instability, follow-up experiments are ranked according to **observed error counts**, **likely quantitative benefit**, and **false-positive explosion risk**.

---

## 2. Ranked Follow-up Experiments

| Rank | Experiment Title | Target Error Category | Error Pool Addressed | Expected Metric Impact | FP Risk | Complexity |
| :-: | :--- | :--- | :---: | :--- | :---: | :---: |
| **1** | **Filler-Aware & Boundary-Aware Prolongation Gating** | Conversational filler & normal pause FPs | 6 FPs (60% of FPs) | Precision: $68.8\% \to \mathbf{84.6\%}$ | **Zero** | Low |
| **2** | **Gated Inter-Word Acoustic Block Detector (Ablation 2)** | Clinical silent blocks | 3 FNs (23% of FNs) | Recall: $62.9\% \to \mathbf{71.4\%}$ | Moderate / High | Moderate |
| **3** | **Non-Adjacent Repetition with Single-Token Interruption** | Interrupted repeated phrases/words | 3 FNs (23% of FNs) | Recall: $62.9\% \to \mathbf{71.4\%}$ | Low / Moderate | Low |

---

### Experiment 1 (Priority 1 — HIGHEST): Filler-Aware & Boundary-Aware Acoustic Prolongation Gating

#### A. Objective & Motivation
Prevent conversational fillers and clause/sentence boundaries from triggering stuttering diagnoses. In the generalization run, 4 clips were falsely flagged as Stutter due to prolonged filler vowels or repeated filler tokens, and 2 clips were flagged due to normal boundary pauses.

#### B. Observed Empirical Evidence ($N=60$)
- **Filler Prolongation FPs (3 clips):**
  - `M_1022_11y3m_1_dysfluent_008.wav` (prolongation on `"um,"`, dur: 0.54s, conf: 0.68)
  - `M_1022_11y3m_1_dysfluent_000.wav` (prolongation on `"um,"`, dur: 0.93s, conf: 1.00)
  - `FluencyBank_031_47.wav` (prolongation on `"Um,"`, dur: 0.55s, conf: 0.69)
- **Repeated Filler Token FP (1 clip):**
  - `FluencyBank_017_4.wav` (repetition on `"uh uh"`, dur: 0.62s, conf: 0.85)
- **Normal Boundary Pause FPs (2 clips):**
  - `FluencyBank_031_69.wav` (sentence-final `"ago."` dur: 0.50s)
  - `198-209-0026.flac` (syntactic comma pause on `"remained,"` dur: 0.54s)

#### C. Proposed Technical Architecture
1. **Acoustic Filler Time-Masking:**
   - When mining acoustic prolongation candidates, cross-reference candidate time intervals $[t_{\text{start}}, t_{\text{end}}]$ against the timestamp spans of ASR-identified conversational fillers (`"um"`, `"uh"`, `"er"`, `"ah"`, `"erm"`).
   - If candidate temporal overlap with a filler exceeds 50%, suppress candidate admission or reclassify it as a conversational hesitation rather than stuttering.
2. **Lexical Repetition Filler Exclusion:**
   - In `detect_repetitions()` and `detect_ngram_repetitions()`, filter out recognized filler words from candidate repetition pairs.
   - Repeated conversational interjections (e.g., `"uh uh"`, `"um um"`) must trigger the **Filler** class, never the **Stutter** class.
3. **Punctuation & Terminal Syllable Gating:**
   - When an acoustic prolongation occurs on a word bearing terminal punctuation (`.`, `?`, `!`, `,`), require a higher duration threshold ($\ge 0.65$s vs $0.50$s) and tighter pitch stability to prevent normal pre-pausal lengthening from registering as a dysfluency.

#### D. Expected Benefit & Risk Assessment
- **Likely Benefit:** Directly eliminates **up to 6 False Positives**. Stutter Precision increases from **68.75% to 84.62%** ($22/26$), and Stutter Specificity increases from **60.00% to 84.00%** ($21/25$).
- **False-Positive Risk:** **Zero**. This is a purely subtractive precision-guard filter.
- **False-Negative Risk:** Negligible. Genuine stuttering prolongations occur on speech sounds within lexical content words (e.g., `s-s-school`, `m-m-mother`), not on filler hesitation vowels.

---

### Experiment 2 (Priority 2): Gated Inter-Word Acoustic Block Detector with Silence Context (Ablation 2)

#### A. Objective & Motivation
Recover genuine clinical stuttering blocks that are currently invisible to Fix 3. Fix 3 disables standalone acoustic block detection due to high false-alarm rates on ordinary speech pauses. Consequently, clinical interview data in FluencyBank suffers from a 30.00% recall ceiling.

#### B. Observed Empirical Evidence ($N=60$)
- **Silent / Tense Block FNs (3 clips in FluencyBank):**
  - `FluencyBank_017_42.wav` (GT consensus Block $\ge 2$, silent block during turn)
  - `FluencyBank_017_52.wav` (GT consensus Block $\ge 2$, pre-utterance block)
  - `FluencyBank_031_14.wav` (GT consensus Block $\ge 2$, pre-phonatory block before `"You get better"`)

#### C. Proposed Technical Architecture
1. **Silence Interval Mining:**
   - Extract inter-word silence intervals $T_{\text{silence}} \in [0.65\text{s}, 2.50\text{s}]$.
2. **Syntactic Boundary Masking (Mandatory Dependency on Experiment 1):**
   - Disqualify any silence that immediately follows a sentence terminator (`.`, `?`, `!`) or clause boundary (`,`, `;`).
3. **Phonation Onset Energy Flux Verification:**
   - Analyze the 100ms window immediately following silence termination (the onset of the subsequent word).
   - Require an abrupt spectral flux surge ($>0.08$) and RMS energy rise ($>0.10$), characteristic of an explosive articulatory release / tense onset after a dysfluent block, rather than smooth breathing.

#### D. Expected Benefit & Risk Assessment
- **Likely Benefit:** Recovers **3 False Negatives** in FluencyBank. Increases FluencyBank recall from **30.00% to 60.00%**, and lifts Overall Stutter Recall from **62.86% to 71.43%** ($25/35$).
- **False-Positive Risk:** **Moderate to High**. If phonation onset gating is tuned too loosely, natural conversational pauses will trigger false alarms. Experiment 1's boundary masking is a mandatory prerequisite.
- **Implementation Complexity:** Moderate. Requires audio silence segmentation and local spectral flux release profiling.

---

### Experiment 3 (Priority 3): Non-Adjacent Word / Phrase Repetition with Single-Token Interruption Bypass

#### A. Objective & Motivation
Recover phrase and word repetitions where the speaker inserts a single hesitation or conversational filler between the repeated elements.

#### B. Observed Empirical Evidence ($N=60$)
- **Interrupted Repetition FNs (3 clips):**
  - `M_0061_16y9m-1_dysfluent_081.wav` (`"it depends, [it sort of] depends"`)
  - `FluencyBank_017_2.wav` (`"still, [obviously,] still"`)
  - `198-209-0025.flac` (`"for time, and, [er,] at time"`)

#### C. Proposed Technical Architecture
1. **Token Skip Matching:**
   - Extend the n-gram matching window to allow matching identical words ($W_i == W_{i+2}$) or identical 2-grams across a single intervening token $W_{i+1}$, provided $W_{i+1}$ belongs to a recognized filler/interjection lexicon (`"sort of"`, `"obviously"`, `"er"`, `"um"`, `"like"`).
2. **Confidence Calibration:**
   - Assign a calibrated confidence of $0.70$ (compared to $0.85$ for contiguous n-grams) to account for conversational restart ambiguity.

#### D. Expected Benefit & Risk Assessment
- **Likely Benefit:** Recovers **up to 3 False Negatives** across all three corpora, boosting overall recall from **62.86% to 71.43%**.
- **False-Positive Risk:** **Low to Moderate**. Risk of false alarms on rhetorical repetition (e.g., `"very, you know, very"`).
- **Implementation Complexity:** Low. Self-contained extension of token indexing logic in `detect_ngram_repetitions()`.

---

## 3. Comparative Trade-off Analysis

| Feature / Dimension | Experiment 1: Filler/Boundary Gating | Experiment 2: Gated Block Detector | Experiment 3: Non-Adjacent Repetition |
| :--- | :--- | :--- | :--- |
| **Primary Target** | False Positives (Precision) | False Negatives (Recall) | False Negatives (Recall) |
| **Target Error Count** | 6 FPs (60% of FPs) | 3 FNs (23% of FNs) | 3 FNs (23% of FNs) |
| **Projected Stutter Precision** | **84.6% (+15.8 pp)** | 64.1% (-4.6 pp if unmitigated) | 68.7% (Unchanged) |
| **Projected Stutter Recall** | 62.9% (Unchanged) | **71.4% (+8.5 pp)** | **71.4% (+8.5 pp)** |
| **Projected Stutter F1** | **0.721** | **0.676** | **0.700** |
| **Regression Risk** | Negligible | High (if gating leaks pauses) | Low |
| **Prerequisites** | None | Requires Experiment 1 Gating | None |

---

## 4. Single Highest-Priority Follow-up Recommendation

### **Recommendation: Experiment 1 (Filler-Aware & Boundary-Aware Acoustic Prolongation Gating)**

### Detailed Empirical Rationale:
1. **Precision is Currently the Critical Bottleneck:**
   - On the 60 unseen generalization clips, Stutter Precision is currently **68.75%** ($	ext{TP}=22, 	ext{FP}=10$). For every 3 clips flagged as Stutter, 1 is a false alarm.
   - 40% of these false alarms (4 clips) are caused by conversational filler vowels (`"um"`, `"uh"`), and 20% (2 clips) are caused by normal terminal pauses.
   - Experiment 1 directly addresses **60% of all false positives** (6 out of 10), projecting an immediate jump in Stutter Precision from **68.75% to 84.62%** without degrading recall.
2. **Experiment 1 is an Architectural Prerequisite for Block Detection:**
   - While recovering the 3 silent blocks in FluencyBank (Experiment 2) is highly desirable, introducing acoustic block detection into a pipeline that already mistakes normal pauses for blocks will inevitably cause a false-positive explosion.
   - Implementing Experiment 1 first establishes robust filler suppression and syntactic boundary masking, creating the necessary protective barrier that makes acoustic block detection safe to deploy in a subsequent cycle.
3. **Minimal Implementation Risk:**
   - Experiment 1 is entirely subtractive: it introduces gating rules that suppress known non-stuttered phenomena (fillers and phrase boundaries). It does not alter core feature extraction, does not require retraining, and presents virtually zero risk of new regressions.

---

## 5. Next Steps

In accordance with strict integrity constraints:
- **No algorithms or thresholds will be implemented or tested at this stage.**
- This recommendation is submitted for user review. Implementation will commence only upon explicit authorization.
