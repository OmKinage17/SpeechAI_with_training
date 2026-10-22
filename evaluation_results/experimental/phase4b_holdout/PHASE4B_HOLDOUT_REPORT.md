# Phase 4B.2: One-Time Locked Holdout Evaluation Report

**Evaluation Title:** Multi-Iteration Hesitation Gating Candidate Evaluation on Locked 120-Clip Benchmark  
**Date & Timestamp:** 2026-10-22 13:09:14 – 13:10:08 UTC  
**Evaluation Scope:** Locked Holdout Evaluation ($N=120$ unseen clips)  
**Authoritative Baseline:** Phase 2 Corrected Benchmark Baseline  
**Candidate Under Test:** Phase 4B Multi-Iteration Hesitation Gating Candidate (Intervention 1)  
**Execution Status:** Completed (120/120 clips processed, 0 failures, 0 timeouts)  
**Output Directory:** `evaluation_results/experimental/phase4b_holdout/`  

---

## 1. Executive Summary & Final Verdict

### Final Verdict: `CANDIDATE NOT SUPPORTED`

The Phase 4B Multi-Iteration Hesitation Gating Candidate (Intervention 1) was evaluated on the locked 120-clip final benchmark under strict one-time execution governance. 

While the candidate successfully recovered **3 severe dysfluent struggle chains** that were missed by the Phase 2 baseline (+4.48% stutter recall), it simultaneously introduced **5 new false positives** on fluent and filler holdout clips (-4.44% stutter precision, -9.44% specificity, -0.84% multiclass accuracy). 

A paired McNemar test on binary stutter classification errors reveals **no statistically significant improvement** ($\chi^2 = 0.1250, p = 0.7237$). Furthermore, performance degraded on conversational speech (FluencyBank specificity dropped by -13.63%, and F1 dropped by -0.0062).

Consequently, the candidate **fails the production acceptance criteria** and is **NOT SUPPORTED** for production deployment. The Phase 2 corrected detector remains the authoritative production baseline.

---

## 2. Pre-Execution Safeguards Verification

All pre-execution safeguards were audited, recorded, and verified prior to executing the evaluation runner:

| Safeguard Check | Required Specification | Observed / Audited Value | Status |
| :--- | :--- | :--- | :--- |
| **Locked Manifest Path** | `evaluation_results/experimental/final_benchmark/run_final_20261020_complete/run_manifest.json` | `evaluation_results/experimental/final_benchmark/run_final_20261020_complete/run_manifest.json` | **PASS** |
| **Locked Manifest SHA-256** | `fdfa62ce5a85c22b57db013fbee8451f84777058e671ae2eb897f8c121e4fb00` | `fdfa62ce5a85c22b57db013fbee8451f84777058e671ae2eb897f8c121e4fb00` | **PASS** |
| **Clip Count & Uniqueness** | Exactly 120 unique clips (40 UCLASS, 40 FB, 40 Libri_Stutter) | Exactly 120 unique clips across 3 corpora | **PASS** |
| **Dataset Isolation** | 0 clip overlap with the 90 development clips | Overlap count = 0 (completely disjoint holdout) | **PASS** |
| **Candidate Configuration** | `enable_multi_iteration_filler_stutter = True`, `min_filler_repetition_count = 3`, `rapid_filler_isi_threshold_sec = 0.35` | Matches Phase 4B development configuration exactly | **PASS** |
| **Output Directory Pristine** | `evaluation_results/experimental/phase4b_holdout/` does not exist | Directory did not exist prior to runner launch | **PASS** |
| **Execution Governance** | Exactly one execution; no post-hoc parameter adjustments | Run executed once synchronously; runtime 54.73s | **PASS** |

---

## 3. Side-by-Side Performance Comparison ($N=120$)

### 3.1 Binary Stutter Classification Metrics

Evaluated on the locked 120 clips with identical evaluation definitions and thresholds:

| Metric | Phase 2 Baseline | Phase 4B Candidate | Absolute Delta | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **True Positives (TP)** | 43 | **46** | **+3** | +6.98% |
| **False Positives (FP)** | **15** | 20 | **+5** | +33.33% (Worse) |
| **False Negatives (FN)** | 24 | **21** | **-3** | -12.50% |
| **True Negatives (TN)** | **38** | 33 | **-5** | -13.16% (Worse) |
| **Precision** | **74.14%** (0.7414) | 69.70% (0.6970) | **-4.44%** (-0.0444) | Degradation |
| **Recall (Sensitivity)** | 64.18% (0.6418) | **68.66%** (0.6866) | **+4.48%** (+0.0448) | Improvement |
| **F1 Score** | 0.6880 | **0.6917** | **+0.0037** | Marginal (+0.54%) |
| **Specificity** | **71.70%** (0.7170) | 62.26% (0.6226) | **-9.44%** (-0.0944) | Severe Degradation |
| **Overall Binary Accuracy** | **67.50%** (81/120) | 65.83% (79/120) | **-1.67%** (-0.0167) | Degradation |

> **Key Takeaway:** The candidate achieved a modest recall increase (+4.48%), but at the expense of a severe 9.44% drop in specificity and a 4.44% drop in precision. For every 1 true positive recovered, the candidate introduced 1.67 false positives.

---

### 3.2 Multiclass Dominant Resolution Metrics (3-Class)

Clips classified into dominant category (**Stutter**, **Filler**, **Clean**):

| Metric / Class | Phase 2 Baseline | Phase 4B Candidate | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **59.17%** (71/120) | 58.33% (70/120) | **-0.84%** (-0.0084) |
| **Macro-Average F1** | **0.5530** | 0.5303 | **-0.0227** |
| **Stutter F1** | 0.6880 | **0.6917** | +0.0037 |
| - Stutter Precision | **74.14%** | 69.70% | -4.44% |
| - Stutter Recall | 64.18% | **68.66%** | +4.48% |
| **Filler F1** | **0.5424** | 0.4706 | **-0.0718** (Worse) |
| - Filler Precision | **42.11%** | 40.00% | -2.11% |
| - Filler Recall | **76.19%** | 57.14% | **-19.05%** (Worse) |
| **Clean F1** | 0.4286 | 0.4286 | 0.0000 |
| - Clean Precision | 50.00% | 50.00% | 0.0000 |
| - Clean Recall | 37.50% | 37.50% | 0.0000 |

#### Multiclass Confusion Matrix Comparison

**Phase 2 Baseline Confusion Matrix:**
```
                Predicted Stutter   Predicted Filler   Predicted Clean   Total
Actual Stutter         43                  14                 10           67
Actual Filler           3                  16                  2           21
Actual Clean           12                   8                 12           32
Total Predicted        58                  38                 24          120
```

**Phase 4B Candidate Confusion Matrix:**
```
                Predicted Stutter   Predicted Filler   Predicted Clean   Total
Actual Stutter         46 (+3)             11 (-3)            10           67
Actual Filler           7 (+4)             12 (-4)             2           21
Actual Clean           13 (+1)              7 (-1)            12           32
Total Predicted        66 (+8)             30 (-8)            24          120
```

> **Multiclass Insight:** The candidate cannibalized valid **Filler** dominant classifications, shifting 4 actual Filler clips and 1 actual Clean clip into false Stutter classifications. As a result, Filler recall plummeted from 76.19% down to 57.14%, dragging Macro-F1 down by -0.0227.

---

### 3.3 Filler Event-Level Detection Metrics

Using identical event-level definitions and boundary matching as Phase 2:

| Metric | Phase 2 Baseline | Phase 4B Candidate | Delta |
| :--- | :---: | :---: | :---: |
| **Event TP** | 24 | 24 | 0 |
| **Event FP** | 14 | 14 | 0 |
| **Event FN** | 9 | 9 | 0 |
| **Event TN** | 73 | 73 | 0 |
| **Precision** | 63.16% | 63.16% | 0.00% |
| **Recall** | 72.73% | 72.73% | 0.00% |
| **F1 Score** | 0.6761 | 0.6761 | 0.0000 |
| **Lost Filler Detections** | — | **0** | Confirmed 0 loss |
| **New Filler Event FPs** | — | **0** | Confirmed 0 new FPs |

*Note: The candidate retained all original filler event detections intact. The regression occurred at the clip-level gating level where multi-filler bursts over-promoted clips from non-stutter to stutter.*

---

## 4. Statistical Significance Testing (McNemar Paired Test)

To evaluate whether the observed shift in binary stutter errors represents a statistically significant improvement or random variation, an exact McNemar test with continuity correction was performed on the paired binary stutter predictions:

### 2x2 Contingency Matrix of Paired Predictions:
```
                                Candidate Correct    Candidate Incorrect
Baseline Correct                       76 (a)                5 (b)
Baseline Incorrect                      3 (c)               36 (d)
```

- **Clips where both models agreed and were correct ($a$):** 76
- **Clips where both models agreed and were incorrect ($d$):** 36
- **Discordant Pairs ($b + c$):** 8 clips
  - **$b$ (Baseline Correct, Candidate Incorrect):** 5 clips (Candidate created New False Positives)
  - **$c$ (Baseline Incorrect, Candidate Correct):** 3 clips (Candidate recovered False Negatives)

### McNemar Statistic:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} = \frac{(|5 - 3| - 1)^2}{5 + 3} = \frac{(2 - 1)^2}{8} = \frac{1}{8} = 0.1250$$

- **Degrees of Freedom:** 1
- **Two-Sided $p$-value:** **$0.7237$** ($p \gg 0.05$)
- **Statistically Significant:** **NO**

The difference between the Phase 2 baseline and the Phase 4B candidate on the locked holdout set is **statistically indistinguishable from chance**.

---

## 5. Per-Corpus Performance Breakdown

Evaluating performance across the three individual datasets reveals critical cross-corpus differences:

### 5.1 FluencyBank ($N=40$, Conversational Speech)

| Metric | Phase 2 Baseline | Phase 4B Candidate | Delta | Evaluation |
| :--- | :---: | :---: | :---: | :--- |
| **TP / FP / FN / TN** | 8 / 3 / 10 / 19 | 9 / 6 / 9 / 16 | +1 TP, +3 FP, -1 FN, -3 TN | 3 new FPs |
| **Precision** | **72.73%** | 60.00% | **-12.73%** | Major drop |
| **Recall** | 44.44% | **50.00%** | **+5.56%** | Minor gain |
| **F1 Score** | **0.5517** | 0.5455 | **-0.0062** | Net degradation |
| **Specificity** | **86.36%** | 72.73% | **-13.63%** | Severe drop |
| **Accuracy** | **67.50%** | 62.50% | **-5.00%** | Degradation |

### 5.2 Libri_Stutter ($N=40$, Read Speech with Synthetic / Controlled Dysfluencies)

| Metric | Phase 2 Baseline | Phase 4B Candidate | Delta | Evaluation |
| :--- | :---: | :---: | :---: | :--- |
| **TP / FP / FN / TN** | 26 / 8 / 2 / 4 | 26 / 8 / 2 / 4 | 0 / 0 / 0 / 0 | 100% Identical |
| **Precision / Recall / F1** | 76.47% / 92.86% / 0.8387 | 76.47% / 92.86% / 0.8387 | 0.00% / 0.00% / 0.0000 | Zero change |
| **Specificity / Accuracy** | 33.33% / 75.00% | 33.33% / 75.00% | 0.00% / 0.00% | Zero change |

### 5.3 UCLASS ($N=40$, Monologue & Clinical Speech)

| Metric | Phase 2 Baseline | Phase 4B Candidate | Delta | Evaluation |
| :--- | :---: | :---: | :---: | :--- |
| **TP / FP / FN / TN** | 9 / 4 / 12 / 15 | 11 / 6 / 10 / 13 | +2 TP, +2 FP, -2 FN, -2 TN | Even trade-off |
| **Precision** | **69.23%** | 64.71% | **-4.52%** | Moderate drop |
| **Recall** | 42.86% | **52.38%** | **+9.52%** | Solid gain |
| **F1 Score** | 0.5294 | **0.5789** | **+0.0495** | Positive gain |
| **Specificity** | **78.95%** | 68.42% | **-10.53%** | Degradation |
| **Accuracy** | 60.00% | 60.00% | 0.00% | Parity |

> **Corpus Diagnostic:** 
> 1. In **UCLASS**, the candidate demonstrated clinical utility by recovering 2 long struggle chains in dysfluent adult speakers, but also misclassified 2 clips.
> 2. In **FluencyBank**, the candidate failed completely: it recovered 1 TP while generating 3 new false alarms, driving specificity down by -13.63% and lowering F1.
> 3. In **Libri_Stutter**, no multi-iteration hesitation chains were present, resulting in complete parity.

---

## 6. Comprehensive Changed Clips Inventory ($N=8$)

Across the entire 120-clip locked holdout set, exactly 8 clips changed predictions between the Phase 2 baseline and the Phase 4B candidate. In all 8 clips, the Phase 2 baseline had predicted `has_stutter = 0` (and `pred_dominant = Filler`), while the Phase 4B candidate predicted `has_stutter = 1` (and `pred_dominant = Stutter`).

### 6.1 True Positive Recoveries ($N=3$ Clips)

These clips represent true perseverative stuttering struggle behaviors successfully recovered by the multi-iteration hesitation gating heuristic:

1. **`M_1099_25y0m_1_dysfluent_199.wav` (UCLASS)**
   - **Ground Truth:** `has_stutter = 1`, `gt_dominant = Stutter`
   - **Baseline:** `has_stutter = 0`, `pred_dominant = Filler` (FN)
   - **Candidate:** `has_stutter = 1`, `pred_dominant = Stutter` (**TP - Recovered**)
   - **Trigger Details:** 3x 'uh' (span: 0.0s, mean ISI: 0.0s, rapid: True, confidence: 0.60, classification: `RAPID_FILLER_STRUGGLE`).
   - **Clinical Description:** Adult male exhibiting rapid hesitation iterations at word onset before speech block release.

2. **`M_1104_40y0m_1_dysfluent_143.wav` (UCLASS)**
   - **Ground Truth:** `has_stutter = 1`, `gt_dominant = Stutter`
   - **Baseline:** `has_stutter = 0`, `pred_dominant = Filler` (FN)
   - **Candidate:** `has_stutter = 1`, `pred_dominant = Stutter` (**TP - Recovered**)
   - **Trigger Details:** 111x 'uh' (span: 2.0s, mean ISI: 0.018s, rapid: True, confidence: 0.85, classification: `SEVERE_FILLER_STRUGGLE_CHAIN`).
   - **Clinical Description:** Adult speaker with severe vocalic perseveration during prolonged hesitation episode.

3. **`FluencyBank_089_66.wav` (FluencyBank)**
   - **Ground Truth:** `has_stutter = 1`, `gt_dominant = Stutter`
   - **Baseline:** `has_stutter = 0`, `pred_dominant = Filler` (FN)
   - **Candidate:** `has_stutter = 1`, `pred_dominant = Stutter` (**TP - Recovered**)
   - **Trigger Details:** 106x 'uh' (span: 1.18s, mean ISI: 0.011s, rapid: True, confidence: 0.85, classification: `SEVERE_FILLER_STRUGGLE_CHAIN`).
   - **Clinical Description:** Child dysfluency episode containing rapid, repetitive vocalizations under conversational pressure.

---

### 6.2 New False Positives ($N=5$ Clips)

These clips represent fluent or normal hesitation utterances incorrectly classified as stuttering by the candidate:

1. **`M_1023_11y3m_1_fluent_014.wav` (UCLASS)**
   - **Ground Truth:** `has_stutter = 0`, `gt_dominant = Clean` (Fluent speaker control)
   - **Baseline:** `has_stutter = 0`, `pred_dominant = Filler` (TN)
   - **Candidate:** `has_stutter = 1`, `pred_dominant = Stutter` (**FP - New False Alarm**)
   - **Trigger Details:** 7x 'um' (span: 1.74s, mean ISI: 0.29s, rapid: True, confidence: 0.76, classification: `SEVERE_FILLER_STRUGGLE_CHAIN`).
   - **Failure Cause:** Natural deliberation hesitation in an 11-year-old fluent speaker. The speaker produced a sequence of deliberation fillers ("um ... um") with an average ISI of 0.29s (< 0.35s threshold). The candidate misclassified normal developmental cognitive pause as pathological struggle.

2. **`M_1099_25y0m_1_dysfluent_152.wav` (UCLASS)**
   - **Ground Truth:** `has_stutter = 0`, `gt_dominant = Filler` (Non-stuttered filler utterance)
   - **Baseline:** `has_stutter = 0`, `pred_dominant = Filler` (TN)
   - **Candidate:** `has_stutter = 1`, `pred_dominant = Stutter` (**FP - New False Alarm**)
   - **Trigger Details:** 107x 'uh' (span: 0.5s, mean ISI: 0.005s, rapid: True, confidence: 0.85, classification: `SEVERE_FILLER_STRUGGLE_CHAIN`).
   - **Failure Cause:** ASR transcript repetition artifact during trailing vocalization. A single prolonged filler sound caused the ASR decoder to hallucinate a repeated token chain, triggering the heuristic without acoustic verification.

3. **`FluencyBank_089_130.wav` (FluencyBank)**
   - **Ground Truth:** `has_stutter = 0`, `gt_dominant = Filler`
   - **Baseline:** `has_stutter = 0`, `pred_dominant = Filler` (TN)
   - **Candidate:** `has_stutter = 1`, `pred_dominant = Stutter` (**FP - New False Alarm**)
   - **Trigger Details:** 108x 'uh' (span: 0.0s, mean ISI: 0.0s, rapid: True, confidence: 0.85, classification: `SEVERE_FILLER_STRUGGLE_CHAIN`).
   - **Failure Cause:** Instantaneous ASR token collapse artifact on a normal conversational filler, erroneously detected as severe stuttering.

4. **`FluencyBank_089_30.wav` (FluencyBank)**
   - **Ground Truth:** `has_stutter = 0`, `gt_dominant = Filler`
   - **Baseline:** `has_stutter = 0`, `pred_dominant = Filler` (TN)
   - **Candidate:** `has_stutter = 1`, `pred_dominant = Stutter` (**FP - New False Alarm**)
   - **Trigger Details:** 3x 'um' (span: 0.36s, mean ISI: 0.18s, rapid: True, confidence: 0.60, classification: `RAPID_FILLER_STRUGGLE`) alongside ASR trailing loop.
   - **Failure Cause:** The speaker uttered a quick double/triple "um um" while thinking of words during a semi-structured interview. This natural hesitation fell below the 0.35s ISI window and triggered a stuttering detection.

5. **`FluencyBank_099_171.wav` (FluencyBank)**
   - **Ground Truth:** `has_stutter = 0`, `gt_dominant = Filler`
   - **Baseline:** `has_stutter = 0`, `pred_dominant = Filler` (TN)
   - **Candidate:** `has_stutter = 1`, `pred_dominant = Stutter` (**FP - New False Alarm**)
   - **Trigger Details:** 110x 'um' (span: 1.42s, mean ISI: 0.013s, rapid: True, confidence: 0.85, classification: `SEVERE_FILLER_STRUGGLE_CHAIN`).
   - **Failure Cause:** Acoustic hesitation held over a clause boundary triggering decoder repetition loop, classified by candidate as clinical stuttering.

---

## 7. Root Cause Analysis & Generalization Failure

Why did the candidate show **0 false positives on the 90 development clips**, but immediately produce **5 false positives on the 120 holdout clips**?

1. **Absence of Acoustic Anchoring (Tension & Pitch Jump):**
   - True stuttering struggle chains involve acoustic correlates of physical tension: elevated fundamental frequency ($F_0$), high energy, laryngeal glottalization, or sudden pitch breaks.
   - Normal conversational hesitations (even repetitive ones like "um... um...") have low, declining pitch and relaxed vocal tract acoustics.
   - Because Intervention 1 relied **purely on textual tokens and inter-segment timing** without acoustic tension verification, it was fundamentally incapable of distinguishing cognitive deliberation from physical speech block struggle.

2. **ASR Token Loop Vulnerability:**
   - In holdout clips containing prolonged or quiet hesitations, Whisper/ASR decoders occasionally generate repetitive token sequences (e.g. `uh uh uh...`).
   - The Phase 2 baseline safely ignored these because filler gating suppressed all hesitation tokens regardless of repetition count.
   - The Phase 4B candidate explicitly un-gated repeated filler chains, making it acutely vulnerable to ASR hallucination loops.

3. **Development Set Selection Bias:**
   - The 90 development clips happened to contain clean non-stutter audio without rapid conversational hesitations or ASR looping artifacts.
   - The holdout set revealed the true empirical distribution of conversational speech (particularly in FluencyBank), where rapid filler bursts and token loops occur regularly.

---

## 8. Post-Evaluation Integrity Audit

Following evaluation, system and artifact integrity was re-verified:

1. **Locked Manifest Invariance:**
   - File: `evaluation_results/experimental/final_benchmark/run_final_20261020_complete/run_manifest.json`
   - Post-run SHA-256: `fdfa62ce5a85c22b57db013fbee8451f84777058e671ae2eb897f8c121e4fb00`
   - Status: **VERIFIED UNCHANGED**

2. **Prior Phase Artifact Protection:**
   - Directories `evaluation_results/experimental/phase2_corrected/`, `phase3_improvement/`, `final_benchmark/`, and `phase4b_filler_gating/` were audited.
   - Status: **VERIFIED UNCHANGED**

3. **Production Isolation:**
   - Production files (`backend/main.py`, production configs, live models, UI frontend) were untouched.
   - Status: **VERIFIED UNCHANGED**

4. **No Post-Hoc Tuning:**
   - No detector code or configuration parameters were adjusted in response to holdout outcomes.
   - Status: **CONFIRMED**

---

## 9. Conclusion & Production Recommendations

1. **Reject Phase 4B Candidate:**
   - Due to the severe specificity penalty (-9.44%), drop in precision (-4.44%), multiclass macro-F1 decline (-0.0227), and lack of statistical significance ($p = 0.7237$), the Phase 4B candidate must not be integrated into production.

2. **Preserve Phase 2 Corrected Baseline:**
   - The Phase 2 corrected detector remains the authoritative, official production baseline.

3. **Future Research Direction:**
   - Any future iteration on hesitation gating must mandate **acoustic tension anchoring** (e.g., verifying $F_0$ rise, RMS energy spikes, or spectral flux across the hesitation chain) and **ASR hallucination filtering** before un-gating filler tokens.
   - In accordance with evaluation governance, any such changes must take place in a new, separately authorized development phase and will require a fresh, untouched evaluation split.
