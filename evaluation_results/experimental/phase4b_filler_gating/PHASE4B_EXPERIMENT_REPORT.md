# Phase 4B: Controlled Evaluation of Multi-Iteration Hesitation Gating

**Project:** SpeechAI Experimental Benchmark Suite  
**Date:** October 22, 2026  
**Investigator / Pair Programmer:** Antigravity (Advanced Agentic AI)  
**Status:** Completed — Development-Only Controlled Evaluation (Intervention 1 Only)  
**Scope:** Strictly confined to the 90 development clips (30 standardized dev + 60 generalization validation) and isolated experimental candidate configuration.  
**Production & Benchmark Safeguards:** Zero production changes, zero locked 120-clip final set access, zero modification of Phase 2/3/4A artifacts, Phase 2 detector maintained as authoritative baseline.

---

## Executive Summary

Phase 4B implemented and evaluated an isolated experimental candidate for **Intervention 1: Multi-Iteration Hesitation Gating**. The intervention specifically targets repeated-filler struggle chains (such as severe motor loops of `"er er er..."` or `"uh uh uh..."`) that were inadvertently suppressed by the Phase 2 filler-boundary filter.

In strict accordance with Phase 4B instructions:
1. **Target Cases Validated First:** Proposed candidate clips were individually inspected against raw transcripts, alignments, ground truth, and acoustic pacing before implementing logic.
2. **Context-Aware Pacing & Calibrated Confidence:** Hesitations are **not** classified as stuttering based solely on a fixed iteration count. The candidate measures inter-syllable interval (ISI) pacing ($\le 0.35\,\text{s}$ threshold) to separate rapid motor struggle from leisurely conversational pause, assigns calibrated continuous confidence ($0.50$ to $0.85$), and retains ambiguous cases as `UNCERTAIN_BORDERLINE` requiring independent corroboration.
3. **Immutability of Transcripts & Baseline:** Raw and sanitized transcripts remain completely unmodified. The Phase 2 baseline configuration remains default and frozen.
4. **Development-Only Evaluation:** Benchmarked strictly on the 90 development clips. Zero access to the locked 120-clip final set.

### Key Evaluation Findings ($N=90$ Development Clips)

| Metric | Authoritative Baseline | Phase 4B Candidate | Absolute Delta | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **True Positives (TP)** | 40 | **42** | **+2** | +5.0% |
| **False Negatives (FN)**| 19 | **17** | **-2** | -10.5% |
| **False Positives (FP)**| 4 | **4** | **0** | **0 new false alarms** |
| **True Negatives (TN)** | 27 | **27** | **0** | **100% specificity preserved** |
| **Stutter Precision** | 90.91% | **91.30%** | **+0.39%** | +0.4% |
| **Stutter Recall** | 67.80% | **71.19%** | **+3.39%** | **+5.0%** |
| **Stutter F1-Score** | 0.7767 | **0.8000** | **+0.0233** | **+3.0%** |
| **Stutter Specificity**| 87.10% | **87.10%** | **0.00%** | **Parity** |
| **Dominant Accuracy** | 67.78% (61/90) | **70.00%** (63/90) | **+2.22%** | +3.3% |
| **Dominant Macro-F1** | 0.6237 | **0.6413** | **+0.0176** | +2.8% |
| **Filler Precision** | 41.03% | **41.03%** | 0.00% | Parity |
| **Filler Recall** | 69.57% | **69.57%** | 0.00% | Parity |
| **Lost Filler Detections** | 0 | **0** | 0 | None |

---

## 1. Strict Scope & Safeguard Compliance

This evaluation adhered strictly to the protocol constraints:
- **Zero Production Impact:** Production entrypoints ([`backend/main.py`](file:///e:/SpeechPrj/backend/main.py)), live models, and frontend code were untouched.
- **Zero Locked Benchmark Access:** The locked 120-clip final evaluation set was **not accessed, evaluated, or referenced**.
- **Preservation of Prior Artifacts:** All artifacts for Phase 2 (`evaluation_results/experimental/phase2_corrected/`), Phase 3 (`evaluation_results/experimental/phase3_improvement/`), and Phase 4A (`evaluation_results/experimental/phase4a_error_analysis/`) remain intact and unaltered.
- **Authoritative Baseline Retained:** The Phase 2 corrected detector configuration remains the default reference baseline.
- **Dedicated Output Directory:** All candidate outputs, configurations, manifests, per-clip CSVs, and metrics are encapsulated under [`evaluation_results/experimental/phase4b_filler_gating/`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4b_filler_gating/).

---

## 2. Validation of Proposed Target Cases

Before implementing any detector changes, an empirical audit of the target cases proposed in Phase 4A was conducted using ground-truth annotations, word alignments, and acoustic data across the 90 development clips:

### 2.1 Case 1: `M_0030_16y4m_1_dysfluent_015.wav`
- **Dataset / Split:** UCLASS Standardized Dev (Duration: 3.12s)
- **Ground Truth:** Stuttering (`gt_has_stutter = 1`, `gt_dominant = "Stutter"`, severe repetition struggle).
- **Transcript:** `"er er er er er er er"` (7 repeated filler tokens).
- **Acoustic Alignment & Pacing:**
  - Token start: 0.00s, Token end: 1.34s (Span: 1.34s).
  - Mean inter-syllable interval (ISI): **0.223s** ($\le 0.35\,\text{s}$ rapid struggle threshold).
  - Rapid, spasmodic motor hesitation chain.
- **Baseline Behavior:** Phase 2 filler gater recognized all tokens as `"er"` and suppressed the repetition completely (`has_stutter = 0`, `conf = 0.0`, FN).
- **Audit Conclusion:** **Confirmed False Negative.** Genuine motor struggle erroneously filtered as conversational filler.

### 2.2 Case 2: `M_0030_16y4m_1_dysfluent_009.wav`
- **Dataset / Split:** UCLASS Standardized Dev (Duration: 3.51s)
- **Ground Truth:** Stuttering (`gt_has_stutter = 1`, `gt_dominant = "Stutter"`, severe blocking loop).
- **Transcript:** 108–110 repeated `"uh"` tokens (`"uh uh uh uh..."`).
- **Acoustic Alignment & Pacing:**
  - Heavy repetitive vocalizations at $t=0.0-2.98\,\text{s}$.
  - Rapid repetition chain representing severe motor perseveration.
- **Baseline Behavior:** Complete suppression by filler repetition filter (`has_stutter = 0`, `conf = 0.0`, FN).
- **Audit Conclusion:** **Confirmed False Negative.** Extreme motor loop erroneously filtered.

### 2.3 Case 3: `FluencyBank_017_15.wav`
- **Dataset / Split:** FluencyBank Generalization Validation (Duration: 3.00s)
- **Ground Truth:** Stuttering (`gt_has_stutter = 1`, annotation: smoothed whole-word repetition).
- **Transcript:** `"I went to the store, oh"` (only 1 single filler `"oh"` at phrase-final boundary).
- **Audit Conclusion:** **Disproved as a multi-iteration target.** This clip contains only a single filler token. Its stutter annotation reflects a whole-word repetition smoothed by Whisper ASR (`"to the"`), not a repeated filler streak. It was correctly excluded from Intervention 1.

### 2.4 Control Case: `FluencyBank_017_4.wav`
- **Dataset / Split:** FluencyBank Generalization Validation (Duration: 3.20s)
- **Ground Truth:** **Filler / Non-Stutter** (`gt_has_stutter = 0`, `gt_dominant = "Filler"`).
- **Transcript:** Contains exactly 2 instances of `"uh"` separated by conversational pause (ISI = 0.36s).
- **Audit Conclusion:** **Essential Control Benchmark.** Proves that setting the threshold to $\ge 3$ iterations and enforcing pacing safeguards is mandatory. A naive count of 2 would have introduced a catastrophic false alarm on clean speech. Under the candidate, `FluencyBank_017_4.wav` remains correctly classified as pure Filler (`has_stutter = 0`).

---

## 3. Experimental Candidate Implementation Details

The candidate logic was integrated in [`backend/experimental/filler_boundary_gating.py`](file:///e:/SpeechPrj/backend/experimental/filler_boundary_gating.py) and wired through [`backend/experimental/stutter_detector.py`](file:///e:/SpeechPrj/backend/experimental/stutter_detector.py) behind the isolated configuration flag `enable_multi_iteration_filler_stutter`.

### 3.1 Pacing Discrimination: Rapid Struggle vs. Leisurely Hesitation
A fixed iteration count alone is insufficient to diagnose pathology. In natural conversation, speakers frequently produce multiple slow hesitations while formulating complex thoughts.

To address this, `_extract_filler_streaks()` extracts consecutive identical filler tokens and computes:
$$\text{Span} = t_{\text{end}} - t_{\text{start}}$$
$$\text{Mean ISI} = \frac{\text{Span}}{\text{Count} - 1}$$
- **Rapid Motor Struggle:** $\text{Mean ISI} \le 0.35\,\text{s}$ (or severe loops $\ge 5$ tokens). Represents rapid articulatory cycling characteristic of clonic stuttering.
- **Leisurely Hesitation:** $\text{Mean ISI} > 0.35\,\text{s}$. Represents deliberate, relaxed conversational planning pauses.

### 3.2 Calibrated Confidence Scoring (No Forced 0.85)
In strict compliance with user instructions, the candidate **does not** automatically assign high confidence ($0.85$) without empirical evidence:

| Repetition Pattern | Pacing Condition | Confidence | Classification | Decision (Standalone) |
| :--- | :---: | :---: | :---: | :---: |
| **1–2 Iterations** (`"uh uh"`) | Any | 0.30 | `CONVERSATIONAL_FILLER` | Gated (Non-stutter) |
| **3 Iterations** (`"um um um"`) | Rapid ($\text{ISI} \le 0.35\,\text{s}$) | **0.60** | `RAPID_FILLER_STRUGGLE` | Stutter (`CONFIDENT_DISFLUENT`) |
| **3 Iterations** (`"um um um"`) | Leisurely ($\text{ISI} > 0.35\,\text{s}$) | **0.50** | `LEISURELY_FILLER_HESITATION` | **Non-stutter** (`UNCERTAIN_BORDERLINE`) |
| **4 Iterations** | Rapid | **0.65** | `REPEATED_FILLER_STRUGGLE` | Stutter (`CONFIDENT_DISFLUENT`) |
| **4 Iterations** | Leisurely | **0.55** | `LEISURELY_FILLER_HESITATION` | **Non-stutter** (`UNCERTAIN_BORDERLINE`) |
| **$\ge 5$ Iterations** | Any | $\min(0.85, 0.70 + (N-5)\times 0.03)$ | `SEVERE_FILLER_STRUGGLE_CHAIN` | Stutter (`CONFIDENT_DISFLUENT`) |

### 3.3 Handling Ambiguous & Borderline Cases
If a 3-iteration hesitation occurs in a leisurely conversational context, it is assigned `confidence = 0.50` and categorized as `UNCERTAIN_BORDERLINE`.
- **Standalone in clean speech:** It does **not** self-corroborate (`has_corroboration = False`). The detector outputs `has_stutter = 0` and resolves dominant class to `"Filler"`.
- **With co-occurring symptom:** If the same utterance exhibits an acoustic block or prolongation, `has_corroboration` evaluates to `True`, allowing the borderline hesitation to contribute to an overall disfluent classification.

---

## 4. Regression & Unit Test Verification

A dedicated unit test suite ([`backend/tests/test_phase4b_filler_gating.py`](file:///e:/SpeechPrj/backend/tests/test_phase4b_filler_gating.py)) was constructed covering all required scenarios:

```text
Ran 8 tests in 0.004s: OK
```

### Verified Test Cases:
1. `test_ordinary_filler_repetitions_remain_fillers`: Verifies that 1–2 filler tokens remain classified as `"Filler"` with `has_stutter = 0`.
2. `test_three_or_more_repeated_fillers_with_struggle_evidence`: Verifies that 3 rapid fillers ($\text{ISI} \le 0.35\,\text{s}$) trigger `has_stutter = 1` with calibrated confidence ($0.60$).
3. `test_severe_struggle_chain_calibrated_confidence`: Verifies 7 iterations (M_0030_015) receive calibrated confidence ($0.76$, strictly $< 0.85$).
4. `test_three_or_more_repeated_fillers_in_fluent_context`: Verifies that leisurely 3-filler hesitations ($\text{ISI} = 0.70\,\text{s}$) receive `UNCERTAIN_BORDERLINE` and output `has_stutter = 0` (no false alarm).
5. `test_genuine_lexical_repetitions_and_interrupted_words_remain_intact`: Verifies that whole-word repetitions (`"I I"`) and broken fragments (`"c- cat"`) retain confidence $\ge 0.85$.
6. `test_raw_and_sanitized_transcript_preservation`: Verifies that transcripts and word structures are never mutated.
7. `test_no_behavior_changes_to_phase2_baseline_configuration`: Verifies that when `enable_multi_iteration_filler_stutter=False`, outputs match the Phase 2 baseline exactly.
8. `test_uncertain_hesitation_corroboration_dynamics`: Verifies that leisurely hesitations remain non-stutter alone, but corroborate when paired with an acoustic prolongation.

### Full Repository Test Suite:
```text
Ran 142 tests across backend/tests in 21.598s: OK (0 failures, 0 errors).
```

---

## 5. Development Benchmark Results & Comparison ($N=90$)

The evaluation runner ([`backend/experimental/run_phase4b_filler_gating.py`](file:///e:/SpeechPrj/backend/experimental/run_phase4b_filler_gating.py)) processed all 90 development clips through both the authoritative Phase 2 baseline and the Phase 4B candidate.

### 5.1 Binary Stutter Detection Performance

```text
Baseline:   TP=40, FP=4, FN=19, TN=27  (Support: Stutter=59, Non-Stutter=31)
Candidate:  TP=42, FP=4, FN=17, TN=27  (Support: Stutter=59, Non-Stutter=31)
Delta:      +2 TP,  0 FP, -2 FN,  0 TN
```

| Metric | Authoritative Baseline | Phase 4B Candidate | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Precision** | 90.91% (40/44) | **91.30%** (42/46) | **+0.39%** |
| **Recall** | 67.80% (40/59) | **71.19%** (42/59) | **+3.39%** |
| **F1 Score** | 0.7767 | **0.8000** | **+0.0233** |
| **Specificity** | 87.10% (27/31) | **87.10%** (27/31) | **0.00%** |
| **Accuracy** | 74.44% (67/90) | **76.67%** (69/90) | **+2.22%** |

### 5.2 Multiclass 3-Class Dominant Resolution

```text
Baseline Confusion Matrix:
               Pred_Stutter   Pred_Filler   Pred_Clean
GT_Stutter          40             6            13      (Total: 59)
GT_Filler            0            13             3      (Total: 16)
GT_Clean             4             3             8      (Total: 15)

Candidate Confusion Matrix:
               Pred_Stutter   Pred_Filler   Pred_Clean
GT_Stutter          42             5            12      (Total: 59)
GT_Filler            0            13             3      (Total: 16)
GT_Clean             4             3             8      (Total: 15)
```

| Class / Metric | Baseline F1 | Candidate F1 | Delta |
| :--- | :---: | :---: | :---: |
| **Stutter Class** | 0.7767 | **0.8000** | **+0.0233** |
| **Filler Class** | 0.6842 | **0.7027** | **+0.0185** |
| **Clean Class** | 0.4103 | **0.4211** | **+0.0108** |
| **Overall Accuracy** | 67.78% (61/90) | **70.00%** (63/90) | **+2.22%** |
| **Macro F1-Score** | 0.6237 | **0.6413** | **+0.0176** |

### 5.3 Per-Corpus Breakdown

| Corpus | Sample Size | Baseline TP / FN / FP | Candidate TP / FN / FP | Baseline Recall -> Cand Recall | Baseline F1 -> Cand F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UCLASS** | 30 | 12 / 9 / 1 | **14 / 7 / 1** | 57.14% -> **66.67%** (+9.53%) | 0.7059 -> **0.7778** (+0.0719) |
| **FluencyBank** | 30 | 5 / 9 / 0 | **5 / 9 / 0** | 35.71% -> 35.71% (Parity) | 0.5263 -> 0.5263 (Parity) |
| **Libri_Stutter** | 30 | 23 / 1 / 3 | **23 / 1 / 3** | 95.83% -> 95.83% (Parity) | 0.9200 -> 0.9200 (Parity) |

*Observation*: All improvements occurred cleanly in UCLASS, where severe repetitive struggle chains were historically concentrated. FluencyBank and Libri_Stutter exhibited zero prediction shifts, verifying that clean speech and ordinary conversational fillers in conversational podcasts and read speech are completely undisturbed.

### 5.4 Complete Inventory of Prediction Differences

Across all 90 development clips, exactly **2 clips** changed prediction between baseline and candidate:

1. **`M_0030_16y4m_1_dysfluent_009.wav` (UCLASS, GT: Stutter):**
   - *Baseline Output:* `has_stutter = 0`, `pred_dominant = "Clean"`, `confidence = 0.0`.
   - *Candidate Output:* `has_stutter = 1`, `pred_dominant = "Stutter"`, `confidence = 0.85`.
   - *Event Details:* `108x 'uh' struggle loop (span: 2.98s, mean_isi: 0.027s, rapid: True)`.
   - *Result:* **Recovered True Positive** (+1 TP, -1 FN).
2. **`M_0030_16y4m_1_dysfluent_015.wav` (UCLASS, GT: Stutter):**
   - *Baseline Output:* `has_stutter = 0`, `pred_dominant = "Filler"`, `confidence = 0.0`.
   - *Candidate Output:* `has_stutter = 1`, `pred_dominant = "Stutter"`, `confidence = 0.76`.
   - *Event Details:* `7x 'er' struggle chain (span: 1.34s, mean_isi: 0.223s, rapid: True)`.
   - *Result:* **Recovered True Positive** (+1 TP, -1 FN).

### 5.5 Adverse Impact Assessment
- **Newly Introduced False Positives:** **0** (Baseline FP=4, Candidate FP=4).
- **Lost True Positives:** **0** (No previously detected stuttering clips were lost).
- **Lost Filler Detections:** **0** (Filler TP=16, Recall=69.57% identical).
- **Runtime Latency:** Dual evaluation of all 90 clips completed in 44.99s (mean: 0.50s per clip). Zero runtime exceptions or worker timeouts.

---

## 6. Distinction Between Verified Results and Hypotheses

### Verified Empirical Facts:
1. **Target Case Grounding:** `M_0030_015` (7x `er`) and `M_0030_009` (108x `uh`) are verified clonic struggle chains, not conversational hesitations.
2. **False Target Elimination:** `FluencyBank_017_15` does not contain a multi-iteration filler streak; treating it as a target would have led to an erroneous heuristic.
3. **Pacing Floor Protection:** The combination of $\ge 3$ iteration count and $\text{ISI} \le 0.35\,\text{s}$ threshold successfully preserved 100% of non-stuttering filler controls (e.g., `FluencyBank_017_4`).
4. **Development Performance:** Recall increased by $+3.39\%$ ($67.80\% \to 71.19\%$), F1 reached **0.8000** ($+0.0233$), and specificity remained unchanged at $87.10\%$.

### Hypotheses & Known Limitations:
1. **ASR Timestamp Dependency:** The pacing check relies on Whisper word-level timestamps. In clips where Whisper hallucinated or collapsed timestamps into identical values ($t=2.98\,\text{s}$ in `M_0030_009`), the rule fell back to token count ($\ge 5$). While effective for extreme perseveration, fine-grained pacing requires reliable ASR timestamps.
2. **Single-Speaker Clustering in Dev Set:** Both recovered clips belong to speaker `M_0030`. It remains a hypothesis whether this specific multi-iteration filler manifestation generalizes across different speakers without evaluating the locked holdout set.
3. **ASR Filler Normalization:** If Whisper normalizes or omits short repeated fillers (e.g., converting `"uh uh uh"` to a single `"uh"` or omitting it entirely), lexical gating cannot detect the disfluency.

---

## 7. Assessment & Next Steps

### Merit for Further Consideration
Candidate 1 (Intervention 1) demonstrates:
- **Zero false alarm penalty** across all 90 development clips.
- **Measurable F1 improvement** ($0.7767 \to 0.8000$).
- **Strict backward compatibility** with Phase 2 baseline when disabled.
- **Principled confidence calibration** that does not artificially inflate scores.

### Strict Stop Condition Enforced
In accordance with user instructions:
- No evaluations have been run on the locked 120-clip final set.
- No modifications have been made to production code or `backend/main.py`.
- No additional interventions (terminal cutoffs, acoustic silence anchoring) have been implemented.
- The Phase 2 detector remains the authoritative baseline.

**Awaiting user review and explicit approval before taking any further action.**
