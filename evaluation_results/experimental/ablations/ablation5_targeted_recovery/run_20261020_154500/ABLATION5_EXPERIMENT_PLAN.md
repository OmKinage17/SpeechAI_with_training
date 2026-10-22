# SpeechAI — Ablation 5: Targeted Experiment Plan & Benefit-Risk Evaluation

**Project:** `E:\SpeechPrj`  
**Run Directory:** `evaluation_results/experimental/ablations/ablation5_targeted_recovery/run_20261020_154500/`  
**Accepted Reference:** Ablation 3 (`evaluation_results/experimental/ablations/stutter_fn_recovery/run_20261020_144131/`)  
**Generalization Validation Benchmark:** 60 clips (Clean: 15, Stutter: 35, Filler: 10)  
**Feature Cache SHA-256:** `91ad23deef8331ec52953412927ca4a08b2d113def0f5dc997f70918dbe4d076`  
**Reference Baseline Metrics:**
- Accuracy: 70.00% (42/60)
- Macro F1: 0.6622
- Weighted F1: 0.7076
- Stutter Precision: 86.67% (26 TP / 30 positive predictions)
- Stutter Recall: 74.29% (26 TP / 35 positive GTs)
- Stutter Specificity: 84.00% (21 TN / 25 negative GTs)
- Confusion Matrix: TP=26, FP=4, FN=9, TN=21

---

## 1. Targeted Hypotheses Formulation

Based on the evidence gathered in Phase 1 (`FALSE_NEGATIVE_ROOT_CAUSE_ANALYSIS.md`), we formulated the two most plausible candidate hypotheses aimed at safely recovering remaining false negatives:

### Hypothesis 1: Gated Non-Adjacent Syntactic Repetition Detection
- **Target Phenomenon:** Non-adjacent repetition of grammatical markers across an intervening content token (`to [word] to`, `in [word] in`).
- **Target Clip:** `M_0061_16y9m-1_dysfluent_060.wav` (transcript contains `"to code, to be called"`).
- **Proposed Mechanism:**
  - Search transcript for non-adjacent identical lemmas separated by exactly 1 or 2 intervening tokens.
  - Require tight inter-word pause timing (≤ 0.25s) to signify speech revision.
  - Implement an exclusion blocklist for standard idiomatic structures (e.g., comparative `"as ... as"`, correlative conjunctions).
- **Target Recovery:** 1 FN -> TP (`M_0061_060`), theoretical recall gain: +2.86% (74.29% -> 77.14%).

### Hypothesis 2: Contextually Gated Sub-500ms Acoustic Prolongation
- **Target Phenomenon:** Mild or conversational phonatory elongation below the standard 0.50s threshold (specifically in the [0.40s, 0.50s) window).
- **Target Clips:** `1743-142912-0001.flac` (0.43s), `FluencyBank_017_52.wav` (0.42s), `FluencyBank_031_14.wav` (0.45s).
- **Proposed Mechanism:**
  - Lower the prolongation admission floor from 0.50s to 0.40s.
  - Apply strict acoustic gating: pitch standard deviation ≤ 1.2 st, mean spectral flux ≤ 0.03, confidence score ≥ 0.50.
  - Suppress sentence-terminal prolongation to mitigate phrase-final lengthening.
- **Target Recovery:** Up to 3 FNs -> TP, theoretical recall gain: +8.57% (74.29% -> 82.86%).

---

## 2. Rejection Criteria (Pre-Declared)

Under the project evaluation policy, an experimental candidate MUST be immediately rejected if any of the following occur:
1. **False-Positive Constraint:** Any increase in false positives on Clean or Filler speech (new FP ≥ 1).
2. **Precision Constraint:** Any regression in stutter precision below the reference baseline (Precision < 86.67%).
3. **Specificity Constraint:** Any regression in stutter specificity below the reference baseline (Specificity < 84.00%).
4. **Generalization Constraint:** Any regression on any sub-corpus (SEP-28k, FluencyBank, LibriSpeech).

---

## 3. Pre-Implementation Empirical Counter-Evidence

Prior to modifying production code or executing Phase 3, we simulated the exact behavior of both hypotheses across all 60 development benchmark clips using the verified feature cache and audio records:

### Empirical Evaluation of Hypothesis 1 (Non-Adjacent Repetition)
- **Results across 60 Clips:**
  - The pattern fires on `M_0061_16y9m-1_dysfluent_060.wav` (`"to code to"`), recovering 1 FN.
  - However, across the remaining 59 clips, identical token repetitions separated by 1 or 2 words are common features of fluent discourse and literature:
    1. `198-209-0026.flac` (Clean): `"as soon as"` -> **NEW FALSE POSITIVE**
    2. `1743-142912-0006.flac` (Clean): `"white throated white"` and `"back and forth and"` -> **NEW FALSE POSITIVE**
    3. `1743-142913-0021.flac` (Clean): `"and the rain and"` -> **NEW FALSE POSITIVE**
    4. `M_0061_16y9m-1_fluent_001.wav` (Clean): `"can do you can"` -> **NEW FALSE POSITIVE**
- **Metric Impact:**
  - Recovered TPs: +1 (26 -> 27)
  - New FPs: **+4 (4 -> 8)**
  - Stutter Precision: Crashes from **86.67% to 77.14% (-9.53%)**
  - Specificity: Drops from **84.00% to 68.00% (-16.00%)**
- **Verdict:** **VIOLATES CRITERIA 1, 2, AND 3.** Rejection is mandatory.

### Empirical Evaluation of Hypothesis 2 (Sub-500ms Prolongation)
- **Results across 60 Clips:**
  - 14 clips contain steady-state vowels between 0.40s and 0.50s.
  - Acoustic inspection confirms that vowels in clean speech have equal or lower pitch deviation and spectral flux than disfluent speech:
    - Clean clip `M_1022_11y3m_1_fluent_049.wav`: `"Learn"` has duration **0.48s**, pitch std **0.55 st**, flux **0.0297**, confidence **0.60**.
    - Stutter clip `FluencyBank_017_52.wav`: duration **0.42s**, pitch std **1.08 st**, flux **0.0281**, confidence **0.52**.
    - Clean speech displays *longer duration* and *tighter pitch stability* than the target stutter clip.
  - Across the 60 clips, admitting [0.40s, 0.50s) candidates generates **6 NEW FALSE POSITIVES**:
    1. `M_1022_11y3m_1_fluent_049.wav` (Clean, dur: 0.48s)
    2. `198-129977-0009.flac` (Clean, dur: 0.44s)
    3. `198-209-0026.flac` (Clean, dur: 0.41s)
    4. `M_1022_11y3m_1_fluent_008.wav` (Clean, dur: 0.43s)
    5. `M_1022_11y3m_1_fluent_001.wav` (Clean, dur: 0.40s)
    6. `FluencyBank_031_47.wav` (Filler, dur: 0.45s)
- **Metric Impact:**
  - Recovered TPs: +3 (26 -> 29)
  - New FPs: **+6 (4 -> 10)**
  - Stutter Precision: Crashes from **86.67% to 74.36% (-12.31%)**
  - Specificity: Drops from **84.00% to 60.00% (-24.00%)**
- **Verdict:** **VIOLATES CRITERIA 1, 2, AND 3.** Rejection is mandatory.

---

## 4. Benefit-Risk Synthesis & Formal Recommendation

| Dimension | Reference (Ablation 3) | Hypothesis 1 (Non-Adj Rep) | Hypothesis 2 (Sub-500ms Prol) |
|---|---|---|---|
| **Stutter Recall** | 74.29% (26/35) | 77.14% (+2.86%) | 82.86% (+8.57%) |
| **Stutter Precision** | **86.67%** (26/30) | 77.14% (**-9.53%**) | 74.36% (**-12.31%**) |
| **Stutter Specificity** | **84.00%** (21/25) | 68.00% (**-16.00%**) | 60.00% (**-24.00%**) |
| **New False Positives** | 0 | **+4** | **+6** |
| **Regressions on Clean** | 0 | **4 clean clips lost** | **5 clean clips lost** |
| **Acceptance Status** | **FROZEN REFERENCE** | **REJECT** | **REJECT** |

### Defensible Benefit-Risk Rationale:
- Neither hypothesis offers a defensible trade-off. In both cases, gaining 1 to 3 true positives causes 4 to 6 false positives, degrading overall system precision and violating the core design philosophy of SpeechAI.
- 77.8% of the remaining false negatives (7/9) stem from upstream ASR normalization or absent acoustic silence, which cannot be compensated for at the detector level without guessing.
- Attempting to force detection on these clips causes catastrophic degradation on clean conversational speech.

### Conclusion:
In strict compliance with Phase 2 instructions:
> *"If no intervention has a defensible benefit-risk rationale, stop and recommend no change."*

We formally **STOP** after Phase 2, execute **NO CODE MODIFICATIONS**, and recommend **NO CHANGE**. The frozen reference **Ablation 3** remains the authoritative, immutable development baseline.
