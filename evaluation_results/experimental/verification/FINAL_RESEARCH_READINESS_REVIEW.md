# SpeechAI Phase E Benchmark: Independent Research-Methodology Review
**Document Version:** 1.0  
**Audit & Review Authority:** Independent Research-Methodology Reviewer  
**Date of Review:** October 18, 2026  
**Target Artifact:** SpeechAI Phase E Experimental Benchmark & Implementation Paper  
**Repository:** `E:\SpeechPrj`  
**Primary Review Deliverable:** `evaluation_results/experimental/verification/FINAL_RESEARCH_READINESS_REVIEW.md`  

---

## 1. Executive Summary & Official Verdict

This independent research-methodology review evaluates the empirical integrity, statistical validity, and publication readiness of the completed SpeechAI Phase E 750-clip experimental benchmark. The review was conducted by cross-verifying all primary data records, audit logs, and recomputed metrics across the existing verification artifacts:
- [`METRIC_DISCREPANCY_ANALYSIS.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/METRIC_DISCREPANCY_ANALYSIS.md)
- [`TIMEOUT_LATENCY_AUDIT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/TIMEOUT_LATENCY_AUDIT.md)
- [`independently_recomputed_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/independently_recomputed_metrics.json)
- [`clip_completeness_audit.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/clip_completeness_audit.json)
- [`immutability_verification.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/immutability_verification.json)
- [`baseline_experimental_comparison.csv`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/baseline_experimental_comparison.csv)
- [`experimental_benchmark_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/main_benchmark/experimental_benchmark_metrics.json)
- [`EXPERIMENTAL_BENCHMARK_REPORT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/main_benchmark/EXPERIMENTAL_BENCHMARK_REPORT.md)

### Official Review Verdict
$$\mathbf{VERDICT: \quad READY \quad AFTER \quad CORRECTIONS}$$

### Verdict Rationale
The empirical foundation of the Phase E benchmark is **sound, complete, and reproducible**:
- All 750 clips (250 UCLASS, 250 FluencyBank, 250 Libri_Stutter) are accounted for with zero data loss, zero missing clips, and zero duplicates.
- All 12 protected baseline and production files are 100% byte-for-byte cryptographically intact.
- The process-level ASR isolation successfully eliminated the 1,917-second runaway temperature-fallback loops that plagued the baseline benchmark.
- All confusion matrices, class supports, and binary contingency tables reconcile mathematically across all 750 records.

However, the manuscript **cannot be submitted in its current state** due to six critical methodological and reporting deficiencies:
1. **Wall-Clock Reporting Conflation:** The benchmark report published `60.47 min (3628.21s)` as total benchmark wall-clock time, which represents only the second resumed session (395 clips), omitting the 122.1-minute first session.
2. **Uncorrected Latency Outlier:** The reported maximum latency of `2,658.687s` was caused by laptop sleep/hibernation during testing, not algorithmic execution, distorting mean latency and RTF.
3. **Unstated Text Normalization in WER:** The reported WER of 44.19% vs 45.05% omitted mention of punctuation normalization; under standard Whisper text cleaning, WER is 23.84% vs 24.93%.
4. **Lack of Statistical Significance Disclosure:** McNemar’s test reveals that the +3.60 percentage point gain in 3-class accuracy ($50.13\% \to 53.73\%$) is **not statistically significant** ($\chi^2 = 1.4923, p = 0.2219$). The paper must report this explicitly.
5. **Overstated Generalization & FluencyBank Collapse:** While the experimental pipeline improved child speech in UCLASS (+18.0 pp accuracy), it experienced a catastrophic accuracy collapse on conversational adult speech in FluencyBank ($62.00\% \to 35.20\%$) and high false positives on fillers.
6. **Mixed Table Reporting:** Table 5 in the benchmark report mixed 3-class overall accuracy with binary disfluency metrics without clear labeling.

Once these six prioritized corrections are incorporated into the manuscript text and tables, the research will meet the highest standards of peer-reviewed empirical AI publications.

---

## 2. Review Criterion 1: Tabular Consistency & Metric Alignment

We conducted a cell-by-cell cross-audit between the summary tables in [`EXPERIMENTAL_BENCHMARK_REPORT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/main_benchmark/EXPERIMENTAL_BENCHMARK_REPORT.md) and the machine-readable values in [`independently_recomputed_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/independently_recomputed_metrics.json).

### 2.1 Tabular Alignment Audit Matrix

| Benchmark Report Table / Section | Reported Metric | Benchmark Report Value | Recomputed JSON Metric | Recomputed Value | Consistency Status | Audit Observation / Discrepancy Detail |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **Table 1: Execution Status** | Total Attempted | 750 | `verified_sample_count` | 750 | **PASS** | Exact match. |
| | Successful | 706 | `execution_status.SUCCESS` | 706 | **PASS** | Exact match. |
| | Timed-Out | 44 | `execution_status.TIMEOUT` | 44 | **PASS** | Exact match. |
| | Errors / Crashes | 0 / 0 | `ERROR` / `CRASHED` | 0 / 0 | **PASS** | Zero unhandled exceptions. |
| **Table 2: Core Comparison** | Baseline Accuracy | 50.13% | `baseline.overall_accuracy` | 0.5013 | **PASS** | $376 / 750 = 50.1333\%$. |
| | Experimental Accuracy | 53.73% | `experimental.overall_accuracy`| 0.5373 | **PASS** | $403 / 750 = 53.7333\%$. |
| | Baseline Macro F1 | 0.4748 | `baseline.macro_f1` | 0.4748 | **PASS** | Exact match. |
| | Experimental Macro F1 | 0.4000 | `experimental.macro_f1` | 0.4000 | **PASS** | Exact match. |
| | Binary Stutter Prec (Base) | 84.62% | `binary_stutter.baseline.prec` | 0.8462 | **PASS** | Exact match. |
| | Binary Stutter Rec (Base) | 29.10% | `binary_stutter.baseline.rec` | 0.2910 | **PASS** | Exact match. |
| | Binary Stutter F1 (Base) | 0.4331 | `binary_stutter.baseline.f1` | 0.4331 | **PASS** | Exact match. |
| | Binary Stutter Prec (Exp) | 56.14% | `binary_stutter.exp.prec` | 0.5614 | **PASS** | Exact match. |
| | Binary Stutter Rec (Exp) | 84.66% | `binary_stutter.exp.rec` | 0.8466 | **PASS** | Exact match. |
| | Binary Stutter F1 (Exp) | 0.6751 | `binary_stutter.exp.f1` | 0.6751 | **PASS** | Exact match. |
| | Binary Filler Prec (Base) | 41.36% | `binary_filler.baseline.prec` | 0.4136 | **PASS** | Exact match. |
| | Binary Filler Rec (Base) | 78.21% | `binary_filler.baseline.rec` | 0.7821 | **PASS** | Exact match. |
| | Binary Filler F1 (Base) | 0.5410 | `binary_filler.baseline.f1` | 0.5411 | **MINOR ROUNDING** | $2 \cdot P \cdot R / (P+R) = 0.54106 \approx 0.5411$. |
| | Binary Filler Prec (Exp) | 26.91% | `binary_filler.exp.prec` | 0.2691 | **PASS** | Exact match. |
| | Binary Filler Rec (Exp) | 69.87% | `binary_filler.exp.rec` | 0.6987 | **PASS** | Exact match. |
| | Binary Filler F1 (Exp) | 0.3886 | `binary_filler.exp.f1` | 0.3886 | **PASS** | Exact match. |
| **Table 3: Confusion Matrix** | True Clean (C / S / F) | 70 / 198 / 24 | `cm.Clean` | 70 / 198 / 24 | **PASS** | Row sum = 292. |
| | True Stutter (C / S / F) | 34 / 320 / 24 | `cm.Stutter` | 34 / 320 / 24 | **PASS** | Row sum = 378. |
| | True Filler (C / S / F) | 15 / 52 / 13 | `cm.Filler` | 15 / 52 / 13 | **PASS** | Row sum = 80. |
| **Table 4: Binary Detection** | Stutter TP / FP / FN / TN | 320 / 250 / 58 / 122 | `binary_stutter.exp` | 320 / 250 / 58 / 122 | **PASS** | Sum = 750. |
| | Filler TP / FP / FN / TN | 109 / 296 / 47 / 298 | `binary_filler.exp` | 109 / 296 / 47 / 298 | **PASS** | Sum = 750. |
| **Table 5: Per-Corpus** | UCLASS Accuracy | 59.20% | `per_corpus.UCLASS.acc` | 0.5920 | **PASS** | Exact match. |
| | FluencyBank Accuracy | 35.20% | `per_corpus.FluencyBank.acc` | 0.3520 | **PASS** | Exact match. |
| | Libri_Stutter Accuracy | 66.80% | `per_corpus.Libri.acc` | 0.6680 | **PASS** | Exact match. |
| **Section 6: ASR WER** | Macro WER (Raw) | 0.4419 | `wer.raw.macro_wer` | 0.4419 | **OMISSION** | Report omitted stating this is unnormalized text. |
| | Evaluated Clips | 248 | `wer.clip_counts.eval` | 248 | **PASS** | Exactly 2 timeouts excluded. |
| **Table 7: Latency & RTF** | Median Latency | 6.173s | `latency.median_sec` | 6.173 | **PASS** | Exact match. |
| | Mean Latency | 13.434s | `latency.mean_sec` | 13.434 | **PASS** | Artificially inflated by sleep outlier. |
| | Maximum Latency | 2658.687s | `latency.max_sec` | 2658.687 | **CRITICAL DEFECT** | OS suspension artifact reported as compute latency. |
| | Total Wall-Clock Time | 60.47 min | `execution_meta.elapsed` | 3628.21s | **CRITICAL DEFECT** | Represents Session 2 only; omits Session 1 (122.1 min). |

*Reviewer Assessment:* Tabular values agree with underlying JSON records. The primary defects are contextual omissions (raw vs. normalized WER) and execution artifacts (suspension max latency, truncated wall-clock duration).

---

## 3. Review Criterion 2: Confusion Matrix & Class Support Integrity

We audited the internal consistency of the 3-class confusion matrix, marginal support totals, and binary contingency tables.

### 3.1 3-Class Confusion Matrix Marginal Verification

```
                      Predicted Clean    Predicted Stutter    Predicted Filler     Row Total (Ground Truth Support)
True Clean                  70                  198                  24                          292
True Stutter                34                  320                  24                          378
True Filler                 15                   52                  13                           80
Column Total (Pred)        119                  570                  61                          750
```

1. **Row Sums (Ground Truth Support):**
   - Clean Support: $70 + 198 + 24 = \mathbf{292}$
   - Stutter Support: $34 + 320 + 24 = \mathbf{378}$
   - Filler Support: $15 + 52 + 13 = \mathbf{80}$
   - Total Evaluated Clips: $292 + 378 + 80 = \mathbf{750}$
2. **Column Sums (Model Prediction Totals):**
   - Predicted Clean: $70 + 34 + 15 = \mathbf{119}$
   - Predicted Stutter: $198 + 320 + 52 = \mathbf{570}$
   - Predicted Filler: $24 + 24 + 13 = \mathbf{61}$
   - Total Predicted Clips: $119 + 570 + 61 = \mathbf{750}$
3. **Trace (Sum of Correct Predictions):**
   $$\text{Trace} = 70 (\text{Clean}) + 320 (\text{Stutter}) + 13 (\text{Filler}) = \mathbf{403}$$
   $$\text{Overall Accuracy} = \frac{403}{750} = \mathbf{53.7333\%} \approx \mathbf{53.73\%}$$

### 3.2 Binary Contingency Table Reconciliations

The benchmark evaluates two binary detection tasks: Stutter Presence and Filler Presence. We verified that positive support and negative support partition the entire 750-clip cohort without remainder.

#### Binary Stutter Contingency Table:
- $\text{TP} = 320, \quad \text{FP} = 250, \quad \text{FN} = 58, \quad \text{TN} = 122$
- Positive Condition Support: $\text{TP} + \text{FN} = 320 + 58 = \mathbf{378}$
- Negative Condition Support: $\text{TN} + \text{FP} = 122 + 250 = \mathbf{372}$
- Cohort Sum: $378 + 372 = \mathbf{750}$
- Specificity: $\text{TN} / (\text{TN} + \text{FP}) = 122 / 372 = \mathbf{32.80\%}$
- Sensitivity (Recall): $\text{TP} / (\text{TP} + \text{FN}) = 320 / 378 = \mathbf{84.66\%}$

#### Binary Filler Contingency Table:
- $\text{TP} = 109, \quad \text{FP} = 296, \quad \text{FN} = 47, \quad \text{TN} = 298$
- Positive Condition Support: $\text{TP} + \text{FN} = 109 + 47 = \mathbf{156}$
- Negative Condition Support: $\text{TN} + \text{FP} = 298 + 296 = \mathbf{594}$
- Cohort Sum: $156 + 594 = \mathbf{750}$
- Specificity: $\text{TN} / (\text{TN} + \text{FP}) = 298 / 594 = \mathbf{50.17\%}$
- Sensitivity (Recall): $\text{TP} / (\text{TP} + \text{FN}) = 109 / 156 = \mathbf{69.87\%}$

*Conclusion:* The confusion matrices and binary detection tables are mathematically closed and reconcile with zero discrepancy across all 750 clips.

---

## 4. Review Criterion 3: Ground-Truth Disfluency Taxonomy Consistency

A critical question raised in peer review is whether the comparative evaluation between Baseline and Experimental was conducted on identical ground-truth annotations.

### 4.1 Taxonomy Reconciliation (Definition A vs. Definition B)

As established in [`METRIC_DISCREPANCY_ANALYSIS.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/METRIC_DISCREPANCY_ANALYSIS.md):
- **Definition A (Historical Baseline Audit):** Evaluated Stutter as `gt_sound_rep or gt_word_rep or gt_prolongation`, completely omitting `gt_block`. This yielded:
  - Stutter Support = **344**
  - Baseline Accuracy = $383 / 750 = \mathbf{51.07\%}$
  - Experimental Accuracy = $386 / 750 = \mathbf{51.47\%}$
- **Definition B (SEP-28k Clinically Inclusive Baseline):** Evaluated Stutter as `gt_sound_rep or gt_word_rep or gt_prolongation or gt_block`. Exactly 34 clips in UCLASS (23) and FluencyBank (11) have `gt_block == 1` with 0 repetitions/prolongations. This shifted:
  - Stutter Support = $344 + 34 = \mathbf{378}$
  - Baseline Accuracy = $376 / 750 = \mathbf{50.13\%}$
  - Experimental Accuracy = $403 / 750 = \mathbf{53.73\%}$

### 4.2 Reviewer Finding on Methodological Consistency

1. In [`EXPERIMENTAL_BENCHMARK_REPORT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/main_benchmark/EXPERIMENTAL_BENCHMARK_REPORT.md), **Definition B was consistently applied to both Baseline and Experimental predictions.** Both models were evaluated against the exact same 378 ground-truth stutter clips, 80 filler clips, and 292 clean clips.
2. The comparison presented in Table 2 of the benchmark report (Baseline 50.13% vs Experimental 53.73%) is an **apples-to-apples comparison**.
3. **Mandatory Manuscript Requirement:** To avoid confusion with previously circulated project documentation citing 51.07% baseline accuracy, the implementation paper must explicitly define the inclusion of `gt_block` under SEP-28k taxonomy and present both taxonomies in an ablation or historical comparison footnote.

---

## 5. Review Criterion 4: Automatic Speech Recognition (ASR) & Word Error Rate (WER)

### 5.1 Text Normalization Reporting Transparency

Section 6 of `EXPERIMENTAL_BENCHMARK_REPORT.md` reported:
- Experimental Macro WER = **0.4419 (44.19%)**
- Baseline Macro WER = **0.4505 (45.05%)**
- Experimental Pooled Micro WER = **0.4298 (42.98%)**
- Baseline Pooled Micro WER = **0.4364 (43.64%)**

#### The Methodological Omission:
The report omitted that these figures represent **raw, unnormalized text evaluation**.
- Ground-truth reference strings in Libri_Stutter are unpunctuated uppercase strings (e.g., `"MISTRESS MARY QUITE CONTRARY HOW DOES YOUR GARDEN GROW"`).
- OpenAI Whisper generates standard English punctuation (commas, periods, question marks, apostrophes) and mixed-case tokens.
- Without text cleaning, Levenshtein distance treats every punctuation mark and capitalization mismatch as an insertion or substitution penalty, artificially inflating WER by nearly 20 percentage points.

#### Normalized WER Recomputation (`clean_text()`):
When standard Whisper text cleaning is applied (stripping punctuation marks, lowercasing, and normalizing whitespace):
- **Baseline Macro WER:** **0.2493 (24.93%)** (matching the baseline audit report down to the fourth decimal place)
- **Experimental Macro WER:** **0.2384 (23.84%)**
- **Absolute Improvement:** **-0.0109 (-1.09 percentage points)**
- **Baseline Pooled Micro WER:** **0.2430 (24.30%)**
- **Experimental Pooled Micro WER:** **0.2346 (23.46%)**
- **Absolute Improvement:** **-0.0084 (-0.84 percentage points)**

### 5.2 Denominator and Timeout Handling Transparency

- WER was evaluated strictly on the **248 Libri_Stutter clips** that completed ASR inference.
- Exactly 2 Libri_Stutter clips timed out (`1334-135589-0088.flac` and `1334-135589-0039.flac`) and were excluded from the evaluated denominator ($N=248$).
- Neither timed-out clip was assigned an artificial WER of `0.0` or `1.0`.
- **Manuscript Correction:** The paper must present Normalized WER (23.84% vs 24.93%) as the primary accuracy figure, cite raw WER (44.19% vs 45.05%) for completeness, and explicitly state the evaluation denominator ($N=248$, $0.8\%$ timeout rate).

---

## 6. Review Criterion 5: Latency, Real-Time Factor (RTF) & Throughput

As detailed in [`TIMEOUT_LATENCY_AUDIT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/TIMEOUT_LATENCY_AUDIT.md), the benchmark report contained two severe latency reporting defects that must be resolved prior to publication.

### 6.1 Tri-Fold Latency Taxonomy

Authors must distinguish three distinct timing dimensions:

```
+----------------------------------------------------------------------------------------------------+
| 1. ACTIVE COMPUTE LATENCY (Algorithmic Performance, N=746 clips)                                   |
|    Median: 6.093s | Mean: 8.994s | P95: 25.062s | Max: 28.361s | Active Micro RTF: 1.341          |
+----------------------------------------------------------------------------------------------------+
| 2. SUMMED CLIP LATENCY (All JSON Prediction Records, N=750 clips)                                  |
|    Sum: 10,075.79s (167.93 min) | Mean: 13.434s | Max: 2658.687s (includes 4 sleep outliers)      |
+----------------------------------------------------------------------------------------------------+
| 3. MULTI-SESSION ELAPSED WALL-CLOCK TIME (End-to-End Campaign Duration)                            |
|    Session 1: 7,326.43s (122.1 min) | Pause: 9,193.01s (153.2 min) | Session 2: 3,628.21s (60.5 min)|
|    Total Active Wall-Clock: 10,954.64s (182.58 min) | Gross Campaign Span: 20,147.65s (5h 35m)     |
+----------------------------------------------------------------------------------------------------+
```

### 6.2 The 2,658.687s Outlier & Hardware Sleep Disclosure

- **Report Error:** `EXPERIMENTAL_BENCHMARK_REPORT.md` Table 7 listed `Maximum Latency: 2658.687s` under the Experimental Pipeline without explanation.
- **Forensic Truth:** Clip 239 (`M_0107_07y7m_1_fluent_012.wav`) finished with status `SUCCESS` and transcript `"THE NEXT TWO YEARS"`. The 44-minute gap in the execution log (`15:05:47` to `15:50:07`) was caused by hardware sleep when the user temporarily closed the laptop.
- Because the OS halted CPU execution, the Whisper child process was frozen; upon wake-up, it finished inference in $< 1\text{s}$, but `time.perf_counter()` recorded the elapsed sleep interval.
- **True Algorithmic Maximum:** The actual maximum active latency of the experimental pipeline was **28.361 seconds** (25.0s timeout + Windows process reaping overhead).
- **Mandatory Correction:** The 2,658.687s figure must be excluded from active-processing statistics in the manuscript.

### 6.3 Wall-Clock Reporting Defect Reconciled

- `EXPERIMENTAL_BENCHMARK_REPORT.md` stated that the entire experimental benchmark completed in `60.47 min (3628.21s)`.
- This was an error caused by measuring elapsed time only from `18:51:26` when Session 2 resumed from the 355-clip checkpoint.
- Session 1 required `122.11 min (7326.43s)`.
- Total active wall-clock time was **182.58 min (10,954.64s)**.
- **Mandatory Correction:** The manuscript must report the full multi-session wall-clock duration of 182.58 minutes.

---

## 7. Review Criterion 6: Stutter Precision–Recall Trade-Off & Filler Degradation

A core responsibility of this review is to ensure the implementation paper does **not overstate performance** and transparently analyzes algorithmic regressions.

### 7.1 Stutter Detection: The Sensitivity Trade-Off

The experimental pipeline made an intentional architectural shift:

| Metric | Audited Baseline | Experimental Pipeline | Absolute Delta | Scientific Characterization |
| :--- | :---: | :---: | :---: | :--- |
| **Recall (Sensitivity)** | **29.10%** | **84.66%** | **+55.56 pp** | **Massive clinical screening gain** (320 / 378 stutters detected). |
| **Precision** | **84.62%** | **56.14%** | **-28.48 pp** | **Substantial increase in false positives** (FP rose from 20 to 250). |
| **F1-Score** | **0.4331** | **0.6751** | **+0.2420** | Net harmonic mean improvement due to dominant recall gain. |
| **Specificity** | **94.62%** | **32.80%** | **-61.82 pp** | Severe collapse in identifying non-stuttered speech as non-stuttered. |

#### Scientific Review Analysis:
- The baseline model was an extremely conservative detector that missed **70.9% of true stutters** (268 False Negatives). It achieved high precision (84.62%) only by refusing to predict stutter on all but the most obvious cases.
- The experimental model lowered acoustic thresholds (prolongation duration: 0.32s; block silence: 0.40s) and combined energy, spectral tilt, and repetition boundaries to prioritize sensitivity.
- **The Trade-Off:** The experimental model achieved clinical screening sensitivity (84.66% recall), but misclassified **198 clean clips and 52 filler clips as stutter**.
- **Manuscript Requirement:** Authors must NOT claim an unqualified improvement in stutter detection. The paper must explicitly frame the experimental model as a **high-sensitivity screening calibration** designed to minimize missed pathological cases at the cost of diagnostic precision.

---

### 7.2 Filler Word Detection: Severe Algorithmic Degradation

The experimental pipeline suffered an unambiguous regression in filler word detection:

| Metric | Audited Baseline | Experimental Pipeline | Absolute Delta | Scientific Characterization |
| :--- | :---: | :---: | :---: | :--- |
| **Precision** | **41.36%** | **26.91%** | **-14.45 pp** | Substantial degradation in confidence. |
| **Recall** | **78.21%** | **69.87%** | **-8.34 pp** | Modest loss in detection sensitivity. |
| **F1-Score** | **0.5410** | **0.3886** | **-0.1524** | **Significant overall performance collapse.** |
| **False Positives** | **173 clips** | **296 clips** | **+123 clips** | 71.1% increase in spurious filler detections. |

#### Root Cause of the Regression:
- In Libri_Stutter (synthetic read speech with **0 ground-truth fillers**), the experimental model produced **171 False Positives**, compared to baseline's 64 FP.
- The experimental context-aware filler detector utilized a fixed 0.20-second pause boundary rule. In audiobook recordings, natural syntactic breath pauses and expressive punctuation intervals consistently triggered the acoustic filler boundary detector.
- **Manuscript Requirement:** Authors must acknowledge this regression without minimization. It must be documented as a limitation of static acoustic pause thresholds on expressive speech, with a concrete recommendation for speaker-adaptive prosodic normalization in future work.

---

## 8. Review Criterion 7: Corpus-Level Generalization & Disconfirmation of Claims

The benchmark evaluated 250 clips each from three diverse speech corpora. We cross-examined whether performance claims generalize across corpora.

### 8.1 Corpus-Level Performance Audit Matrix

| Corpus | Target Population & Acoustic Profile | Baseline 3-Class Acc | Exp 3-Class Acc | Acc Delta ($\Delta$) | Exp Stutter F1 | Exp Stutter Rec | Exp Filler F1 | Validated Scientific Conclusion |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **UCLASS** | Clinical recordings of children who stutter (monologue / reading). | 41.20% | **59.20%** | **+18.00 pp** | **0.7222** | **76.97%** | **0.5860** | **SUCCESS:** Major, unambiguous improvement in pediatric screening and clinical disfluency detection. |
| **FluencyBank** | Adult conversational interviews (stuttering and typical speakers). | **62.00%** | **35.20%** | **-26.80 pp** | 0.3662 | 67.24% | 0.5257 | **SEVERE REGRESSION:** Massive failure on conversational speech. Over-predicted stutter on conversational pauses (Prec: 25.16%). |
| **Libri_Stutter** | Clean audiobook recordings with synthetically spliced repetitions. | 47.20% | **66.80%** | **+19.60 pp** | **0.7981** | **97.62%** | N/A (Supp=0) | **MIXED:** High sensitivity to synthetic repetitions (97.6% Rec), but clean recall collapsed to 3.66% (79 FP) and 171 filler FP. |

### 8.2 Unsupported Claims to Strike from Manuscript

The following claims are **empirically disconfirmed** by the benchmark data and must NOT appear in the implementation paper:
1. *"The experimental pipeline demonstrates generalized superiority across all speech domains."*  
   **Fact:** False. The pipeline experienced a 26.80 percentage point drop in accuracy on FluencyBank.
2. *"SpeechAI provides robust fluency classification for conversational speech."*  
   **Fact:** False. On conversational adult speech (FluencyBank), accuracy collapsed to 35.20%, and stutter precision was only 25.16% (3 out of 4 predicted stutters were false alarms).
3. *"Context-aware filler detection improves conversational interjection accuracy."*  
   **Fact:** False. Filler F1 degraded from 0.5410 to 0.3886 overall, and false alarms rose by 71%.

---

## 9. Review Criterion 8: Limitations, Constraints & Statistical Significance

### 9.1 Formal Statistical Significance Testing (McNemar’s Test)

In `EXPERIMENTAL_BENCHMARK_REPORT.md`, the authors highlighted the +3.60 percentage point accuracy increase ($50.13\% \to 53.73\%$). However, no statistical significance testing was performed.

This review conducted a formal **McNemar’s Test for Paired Nominal Data** on the 750 paired predictions under Definition B:

#### McNemar Contingency Table ($N=750$ paired clips):
```
                                Experimental Correct    Experimental Incorrect    Row Total
Baseline Correct                         163                     213                 376 (50.13%)
Baseline Incorrect                       240                     134                 374 (49.87%)
Column Total                             403 (53.73%)            347 (46.27%)        750
```

- $n_{01}$ (Baseline Incorrect, Experimental Correct): **240 clips**
- $n_{10}$ (Baseline Correct, Experimental Incorrect): **213 clips**
- Discordant Pairs: $n_{01} + n_{10} = 240 + 213 = 453 \text{ clips}$
- Net Correct Gain: $240 - 213 = \mathbf{27 \text{ clips}}$

#### McNemar's Test Statistic (with Edwards' Continuity Correction):
$$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}} = \frac{(|240 - 213| - 1)^2}{240 + 213} = \frac{(27 - 1)^2}{453} = \frac{26^2}{453} = \frac{676}{453} \approx \mathbf{1.4923}$$

#### P-Value Determination:
$$\mathbf{p\text{-value} = 0.2219} \quad (\mathbf{p > 0.05})$$

#### Critical Methodological Finding:
**The +3.60 percentage point gain in overall 3-class classification accuracy is NOT STATISTICALLY SIGNIFICANT at the $\alpha = 0.05$ significance level ($p = 0.2219$).**

*Scientific Explanation:* While the experimental model successfully gained 240 clips that baseline missed (primarily catching previously missed stutters), it simultaneously degraded on 213 clips that baseline correctly classified (primarily misclassifying clean and filler speech as stutter). The net difference of 27 clips across 453 discordant pairs is consistent with random variation under the null hypothesis.

---

### 9.2 Summary of Inherent Limitations to Include in the Paper

1. **Clip-Level Binary vs. Event-Level Boundaries:** Evaluations operate on 3-second segmented clips. The system assesses disfluency presence within a window, not millisecond-accurate acoustic boundary alignments (onset/offset timestamps).
2. **Synthetic vs. Natural Disfluency Divergence:** High performance on Libri_Stutter (synthetic splicing) does not predict performance on natural conversational speech (FluencyBank).
3. **ASR Process Timeouts:** Under CPU execution, 44 clips (5.87%) timed out at the 25.0s ASR cutoff. Full transcription failed on 5.87% of clips, although acoustic screening was successfully preserved.
4. **Hardware Environment:** All benchmarks were run on CPU; GPU acceleration will significantly reduce latency and RTF.
5. **No Clinical Validation:** SpeechAI is a proof-of-concept screening aid, not an FDA/CE-cleared diagnostic device. It has not undergone clinical trials or normative calibration across multi-ethnic dialects.

---

## 10. Prioritized Action Plan of Corrections for Manuscript Submission

To bring the implementation paper to publication standard, the authors must implement the following prioritized corrections:

### Priority 1: Mandatory Technical & Runtime Corrections (High Priority)
1. **Correct Benchmark Wall-Clock Reporting:**  
   Replace `60.47 min (3628.21s)` with the complete multi-session execution breakdown: Session 1 (122.11 min) + Session 2 (60.47 min) = **182.58 min (10,954.64s)** active wall-clock time across 750 clips.
   - *Supporting Artifact:* [`TIMEOUT_LATENCY_AUDIT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/TIMEOUT_LATENCY_AUDIT.md#8-investigation-7-summed-clip-latency-vs-elapsed-wall-clock-time-reconciliation).
2. **Exclude OS Sleep Outlier from Latency Reporting:**  
   Remove the 2,658.687s sleep interval from active compute latency tables. Report **Active Processing Latency** (Median: 6.093s, Mean: 8.994s, P95: 25.062s, Max: 28.361s, Active Micro RTF: 1.341 across 746 clips) and explain the 4 hardware suspension outliers in a dedicated telemetry footnote.
   - *Supporting Artifact:* [`TIMEOUT_LATENCY_AUDIT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/TIMEOUT_LATENCY_AUDIT.md#6-investigation-5-root-cause-of-the-2658687-second-maximum--suspension-outliers).

### Priority 2: Mandatory Scientific Integrity & Discrepancy Disclosures (High Priority)
3. **Transparently Report Both Stutter Taxonomies:**  
   Explicitly declare that baseline accuracy is 50.13% under the SEP-28k inclusive taxonomy (Definition B, Stutter Support = 378) and 51.07% under the narrow taxonomy (Definition A, Stutter Support = 344), citing the exact 34 `gt_block=1` clips.
   - *Supporting Artifact:* [`METRIC_DISCREPANCY_ANALYSIS.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/METRIC_DISCREPANCY_ANALYSIS.md#2-investigation-1-root-cause-analysis-of-baseline-accuracy-5013-vs-5107-and-stutter-support-378-vs-344).
4. **Standardize Word Error Rate (WER) Reporting:**  
   Present Normalized WER (23.84% Experimental vs. 24.93% Baseline) as the primary ASR metric, report Raw WER (44.19% vs. 45.05%) for unnormalized completeness, and declare the evaluation denominator ($N=248$ successful clips, 2 timeouts excluded).
   - *Supporting Artifact:* [`METRIC_DISCREPANCY_ANALYSIS.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/METRIC_DISCREPANCY_ANALYSIS.md#8-investigation-7-word-error-rate-wer-discrepancy-2493-vs-4505).
5. **Acknowledge Non-Significance on Accuracy:**  
   State clearly in the Results section that the +3.60 pp accuracy gain is not statistically significant ($p = 0.2219$, McNemar's test), framing the true contribution as the massive gain in stutter screening recall ($29.10\% \to 84.66\%$) and elimination of runaway ASR inference loops.
   - *Supporting Artifact:* Section 9.1 of this document; [`baseline_experimental_comparison.csv`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/baseline_experimental_comparison.csv).

### Priority 3: Methodological Polish & Discussion Adjustments (Medium Priority)
6. **Acknowledge Filler Regression and Domain Mismatch:**  
   Provide a transparent discussion of the filler detection collapse (F1 dropped to 0.3886, with 171 FP on Libri_Stutter) and the FluencyBank domain regression (accuracy dropped to 35.20%), identifying adaptive pause thresholding as necessary future work.
   - *Supporting Artifact:* Section 7.2 & 8.1 of this document; [`independently_recomputed_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/independently_recomputed_metrics.json).
7. **Disentangle Multi-Class and Binary Table Labels:**  
   Relabel Table 5 in `EXPERIMENTAL_BENCHMARK_REPORT.md` so that readers clearly understand that the Stutter and Filler columns report binary detection metrics, rather than 3-class dominant per-class metrics.
   - *Supporting Artifact:* [`independently_recomputed_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/independently_recomputed_metrics.json#line-290-345).

---

## 11. Final Auditor Certification

This research-methodology review confirms that the empirical Phase E benchmark data in `E:\SpeechPrj` is authentic, mathematically sound, and cryptographically verified. Upon implementation of the prioritized corrections detailed in Section 10, the manuscript will be fully prepared for submission to high-impact speech processing and clinical AI venues (e.g., IEEE/ACM TASLP, Interspeech, or ICASSP).

$$\mathbf{FINAL \quad VERDICT: \quad READY \quad AFTER \quad CORRECTIONS}$$

**Review Conducted by:** Independent Research-Methodology Reviewer  
**Primary Review Deliverable:** `evaluation_results/experimental/verification/FINAL_RESEARCH_READINESS_REVIEW.md`  
**Supporting Verification Artifacts:**
- Metric Discrepancies: [`METRIC_DISCREPANCY_ANALYSIS.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/METRIC_DISCREPANCY_ANALYSIS.md)
- Runtime & Timeouts: [`TIMEOUT_LATENCY_AUDIT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/TIMEOUT_LATENCY_AUDIT.md)
- Machine-Readable Dataset: [`independently_recomputed_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/independently_recomputed_metrics.json)
- Side-by-Side Comparison: [`baseline_experimental_comparison.csv`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/baseline_experimental_comparison.csv)
- Cryptographic Proof: [`immutability_verification.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/immutability_verification.json)
