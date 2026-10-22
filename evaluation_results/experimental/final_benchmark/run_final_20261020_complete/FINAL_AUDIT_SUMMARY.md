# SpeechAI — Final Verification and Benchmark Audit Summary

**Audit Timestamp:** 2026-10-20T18:55:00+05:30  
**Target Directory:** `E:\SpeechPrj\evaluation_results\experimental\final_benchmark\run_final_20261020_complete`  
**Audit Status:** **PASSED (100% Mathematical Reconciliation & Integrity Verified)**  
**Backend Test Suite:** **115 / 115 Tests Passing (100% OK, 0 Errors, 0 Failures)**

---

## 1. Verified Benchmark Metrics (Direct Recomputation)

All metrics were recomputed directly from [`per_clip_predictions.json`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/per_clip_predictions.json) and verified against [`confusion_matrices.json`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/confusion_matrices.json) and [`run_manifest.json`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/run_manifest.json):

### A. Multiclass 3-Class Aggregate ($N=120$)
- **Aggregate Accuracy:** $71 / 120 = \mathbf{59.17\%}$ (95% CI: [50.00%, 68.33%])
- **Macro F1 Score:** $\mathbf{0.5530}$ (95% CI: [0.4556, 0.6444])
- **Weighted F1 Score:** $\mathbf{0.5933}$
- **Multiclass Confusion Matrix:**
  * Actual Clean ($N=32$): Clean=12, Stutter=12, Filler=8
  * Actual Stutter ($N=67$): Clean=10, Stutter=43, Filler=14
  * Actual Filler ($N=21$): Clean=2, Stutter=3, Filler=16
- **Per-Class Breakdown:**
  * **Clean Speech:** Precision = **50.00%**, Recall = **37.50%**, F1 = **0.4286**, Specificity = **86.36%** ($\text{TP}=12, \text{FP}=12, \text{FN}=20, \text{TN}=76$)
  * **Stuttering:** Precision = **74.14%**, Recall = **64.18%**, F1 = **0.6880**, Specificity = **71.70%** ($\text{TP}=43, \text{FP}=15, \text{FN}=24, \text{TN}=38$)
  * **Filler (Dominant):** Precision = **42.11%**, Recall = **76.19%**, F1 = **0.5424**, Specificity = **77.78%** ($\text{TP}=16, \text{FP}=22, \text{FN}=5, \text{TN}=77$)

### B. Binary Stuttering Detection ($N=120$)
- **Task:** Discrimination of Stutter ($N=67$) vs. Non-Stutter ($N=53$).
- **Contingency:** $\text{TP} = 43$, $\text{FP} = 15$, $\text{FN} = 24$, $\text{TN} = 38$
- **Precision:** $\mathbf{74.14\%}$ (95% CI: [62.50%, 84.62%])
- **Recall:** $\mathbf{64.18\%}$ (95% CI: [51.56%, 75.71%])
- **F1 Score:** $\mathbf{0.6880}$ (95% CI: [0.5789, 0.7759])
- **Specificity:** $\mathbf{71.70\%}$ (95% CI: [57.89%, 83.33%])
- **Accuracy:** $\mathbf{67.50\%}$

### C. Filler Event Detection ($N=120$)
- **Task:** Detection of hesitation tokens (`filler_count > 0`) against all annotated interjection clips ($N=33$) and fluent controls.
- **Contingency:** $\text{TP} = 31$, $\text{FP} = 25$, $\text{FN} = 2$, $\text{TN} = 62$
- **Precision:** $\mathbf{55.36\%}$, **Recall:** $\mathbf{93.94\%}$, **F1:** $\mathbf{0.6966}$, **Specificity:** $\mathbf{71.26\%}$, **Accuracy:** $\mathbf{77.50\%}$
- **Libri_Stutter Negative Control:** 0 false positives across all 40 clips (**0.0% FPR**).

### D. Speech-Rate Estimation ($N=40$ Libri_Stutter with Verbatim Gold Reference)
- **Mean Absolute Error (MAE):** **17.84 WPM**
- **Mean Bias Error:** **+11.84 WPM**
- **RMSE:** **22.15 WPM**
- **Pearson Correlation ($r$):** $\mathbf{0.7547}$ ($p = 1.85 \times 10^{-8}$)
- **Spearman Correlation ($\rho$):** $\mathbf{0.6914}$ ($p = 7.82 \times 10^{-7}$)

### E. Automatic Speech Recognition ($N=40$ Libri_Stutter with Verbatim Gold Reference)
- **Reference Words:** 1,249
- **Total Word Edits:** 299
- **Normalized Corpus WER:** $\mathbf{23.94\%}$
- **Mean Clip CER:** $\mathbf{15.35\%}$
- **Timeouts / Failures:** **0 / 120 (0.0%)**

---

## 2. Baseline vs. Final Comparison & Gain Decomposition

### Comparability Verification
- **Sample Parity:** The baseline (`run_expanded_120`) and final optimized (`run_final_20261020_complete`) runs evaluate the **exact same 120 audio clips** with identical SHA-256 manifest hash (`fdfa62ce5a85c22b57db013fbee8451f84777058e671ae2eb897f8c121e4fb00`).
- **Label & Denominator Parity:** Ground-truth definitions and denominators are identical (67 Stutter, 32 Clean, 21 Filler).

### Clear Distinction: Multiclass Accuracy Gain vs. Binary Stuttering F1 Gain

```
+-----------------------------------------------------------------------------------------------------------------+
| Dimension                  | Baseline (Ablation 8) | Final Optimized | Delta / Gain      | Primary Attribution  |
+-----------------------------------------------------------------------------------------------------------------+
| Multiclass Accuracy        | 53.33% (64/120)       | 59.17% (71/120) | +5.84% (+7 clips) | Filler + Stutter Rec.|
| Binary Stutter F1          | 0.6557                | 0.6880          | +0.0323 (+3 TP)   | Timeout + Corrobor.  |
| Stutter Specificity        | 71.70% (38/53)        | 71.70% (38/53)  | +0.00% (Identical)| 0 Additional FPs     |
| ASR Watchdog Timeouts      | 13 clips (10.8%)      | 0 clips (0.0%)  | -13 (-100%)       | Greedy Decoding      |
+-----------------------------------------------------------------------------------------------------------------+
```

1. **Multiclass Accuracy Gain (+5.84%, Net +7 Clips):**
   - 8 clips transitioned from Incorrect $\to$ Correct:
     * **5 Filler Clips:** (`M_1099_..._152`, `FluencyBank_089_130`, `FluencyBank_089_30`, `FluencyBank_099_171`, `FluencyBank_178_75`) were previously misclassified as Clean; final filler boundary gating promoted them to Filler.
     * **3 Stutter Clips:** (`FluencyBank_089_78`, `FluencyBank_089_1`, `1926-143879-0007`) were correctly classified as Stutter (2 from timeout recovery, 1 from repetition corroboration).
   - 1 clip transitioned from Correct $\to$ Incorrect:
     * `M_1099_..._fluent_132` (Clean predicted as Filler).
   - Net gain = $8 - 1 = \mathbf{+7 \text{ correct clips}}$.

2. **Binary Stuttering F1 Gain (+0.0323, +3 True Positives):**
   - True Positives increased from 40 to 43 (+3 TP, -3 FN):
     * **Recovered ASR Timeouts (+2 TP, 66.7% of gain):** `FluencyBank_089_78` and `FluencyBank_089_1` previously timed out on CPU beam search; greedy decoding (`beam_size=1`) produced complete transcripts, allowing lexical repetitions to be identified.
     * **Algorithmic Corroboration (+1 TP, 33.3% of gain):** `1926-143879-0007` on Libri_Stutter was correctly corroborated as an interrupted repetition.
   - False Positives and True Negatives remained identical ($\text{FP}=15, \text{TN}=38$), preserving **71.70% specificity**.

---

## 3. Annotation Integrity: Zero Silent Negative Assumption

A rigorous audit was conducted to confirm that missing annotations were **never silently treated as negative ground truth**:

1. **Block Detection Scoping:**
   - **Clinical Ground Truth Availability:** UCLASS and FluencyBank ($N=80$) explicitly annotate articulatory blocks. Exactly 6 clips are block-positive (4 UCLASS, 2 FluencyBank).
   - **Schema Omission in Libri_Stutter:** Libri_Stutter's annotation taxonomy does not include articulatory blocks. Readers' unvoiced stop closures before plosives (`/p, t, k, b/`) trigger acoustic struggle ratios ($N=19$ clips).
   - **Audit Correction:** Libri_Stutter clips are formally designated as **Unannotated / Schema-Omitted** for block ground truth, rather than scored as negative ground truth.
   - **Reconciled Scoped Evaluation (UCLASS + FluencyBank, $N=80$):**
     * $\text{TP} = 0, \text{FP} = 4, \text{FN} = 6, \text{TN} = 70$
     * **Precision:** **0.00%**, **Recall:** **0.00%**, **F1:** **0.0000**
     * **Specificity:** $\frac{70}{74} = \mathbf{94.59\%}$, **Accuracy:** $\frac{70}{80} = \mathbf{87.50\%}$
2. **Filler Detection Ground Truth:**
   - UCLASS and FluencyBank contain clinician-annotated interjections ($N=33$ positive clips).
   - Libri_Stutter contains professional studio audiobook recordings of literature with 0 conversational fillers, establishing a valid **negative control** (0 false positives observed).
3. **ASR WER Scoping:**
   - Evaluated strictly on Libri_Stutter ($N=40$) where authoritative verbatim text transcripts exist.
   - UCLASS and FluencyBank lack verbatim word-aligned transcripts; reporting estimated WER on them was omitted to maintain scientific validity.

---

## 4. Cross-Artifact Agreement Audit

Automated verification confirmed **100% exact numerical agreement** across all deliverables:
- [`FINAL_BENCHMARK_REPORT.md`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/FINAL_BENCHMARK_REPORT.md)
- [`final_metrics.json`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/final_metrics.json)
- [`per_parameter_metrics.json`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/per_parameter_metrics.json)
- [`per_dataset_metrics.json`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/per_dataset_metrics.json)
- [`confusion_matrices.json`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/confusion_matrices.json)
- [`final_summary_table.csv`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/final_summary_table.csv)
- [`seven_parameters_summary.csv`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/seven_parameters_summary.csv)
- [`per_dataset_table.csv`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/per_dataset_table.csv)
- [`per_clip_results.csv`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/per_clip_results.csv)
- [`runtime_and_failures.json`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/runtime_and_failures.json)
- [`data_integrity_report.json`](file:///e:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/data_integrity_report.json)

---

## 5. Backend Test Suite Verification

```
Ran 115 tests in 19.509s
Status: OK (0 Failures, 0 Errors, 100% Passing)
```
- Module 1 Fluency & Feature Unit Tests: Passed
- Module 2 Coaching & Visual Unit Tests: Passed
- ASR Watchdog Timeout & Process Worker Isolation Tests: Passed
- Filler Boundary & Acoustic Gating Tests: Passed
- N-Gram & Interrupted Repetition Tests: Passed

---

## 6. Honest Limitations & Scope Boundaries

1. **No Claims of Clinical Diagnostic Effectiveness:** SpeechAI is an automated research and screening benchmark. Reported F1 scores (Binary: 0.6880, Macro: 0.5530) demonstrate algorithmic feasibility, not clinical diagnostic equivalence to certified speech-language pathologists.
2. **Acoustic Block Struggle Gap Sensitivity:** On short 3-second audio slices, articulatory blocks co-occur with audible sound repetitions or vocal fry. Pure silent pre-phonation struggle gaps $\ge 0.40$s followed by sudden explosive bursts are rare in short slices. While scoped specificity is high (**94.59%**), standalone acoustic sensitivity on co-occurring blocks is 0.00% without articulatory/visual motion tracking.
3. **Short-Duration Speech-Rate Bias:** Extrapolating WPM from 3-second slices inflates rates (360–400 WPM on clinical slices) compared to full passage reading (171.4 WPM on Libri_Stutter, MAE: 17.84 WPM). Stable conversational rate estimation requires 10–30s windows.
4. **Coaching Rule Engine Validation:** The coaching evaluation reflects **100% deterministic rule compliance** against predefined thresholds; longitudinal user satisfaction and therapeutic efficacy remain subjects for future clinical trials.
