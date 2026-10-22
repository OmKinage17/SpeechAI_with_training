# SpeechAI — Step 3: Independent Aggregate Audit Report

**Audit Target Directory:** [`evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000)  
**Accepted Reference Candidate:** Ablation 3 (`evaluation_results/experimental/ablations/stutter_fn_recovery/run_20261020_144131/`)  
**Raw Source of Truth:** [`per_clip_predictions.json`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/per_clip_predictions.json) & [`run_manifest.json`](file:///E:/SpeechPrj/evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json)  
**Feature Cache SHA-256:** `91ad23deef8331ec52953412927ca4a08b2d113def0f5dc997f70918dbe4d076` (Verified)  
**Audit Step:** Step 3 — Independent Aggregate Audit  
**Audit Status:** **PASS**

---

## 1. Executive Summary & Raw Input Verification

An independent, first-principles audit of the aggregate metrics was performed directly on the raw per-clip prediction records in [`per_clip_predictions.json`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/per_clip_predictions.json) and authoritative ground-truth labels from [`run_manifest.json`](file:///E:/SpeechPrj/evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json).

- **Total Clips Evaluated:** Exactly **60 unique clips**.
- **Ground-Truth Source Cross-Check:** 100% of all 60 clip ground-truth labels in `per_clip_predictions.json` match `run_manifest.json` `gt_dominant` annotations.
- **Ground-Truth Support Sum:** `Clean (14) + Stutter (35) + Filler (11) = 60` (Exact match).
- **Prediction Total Sum:** `Clean (16) + Stutter (30) + Filler (14) = 60` (Exact match).
- **Core Accuracy:** **70.00%** (42 / 60 correct predictions).
- **Binary Stutter Metrics:** **Precision: 86.67%**, **Recall: 74.29%**, **F1: 0.8000**, **Specificity: 84.00%** (TP=26, FP=4, FN=9, TN=21).
- **Verdict:** **PASS**. All calculations reconcile without error.

---

## 2. Recomputed 3-Class Confusion Matrix

Recomputed directly from the 60 raw paired records:

```
                  Predicted Class
                Clean   Stutter  Filler   Total (Support)
Actual Clean       7        4       3       14
Actual Stutter     7       26       2       35
Actual Filler      2        0       9       11
Total (Pred)      16       30      14       60
```

### Row & Column Reconciliation
- **Row Sums (Ground-Truth Class Supports):**
  - Actual Clean: `7 + 4 + 3 = 14`
  - Actual Stutter: `7 + 26 + 2 = 35`
  - Actual Filler: `2 + 0 + 9 = 11`
  - Total Ground-Truth Support: `14 + 35 + 11 = 60` (Reconciled).
- **Column Sums (Predicted Label Counts):**
  - Predicted Clean: `7 + 7 + 2 = 16`
  - Predicted Stutter: `4 + 26 + 0 = 30`
  - Predicted Filler: `3 + 2 + 9 = 14`
  - Total Predicted Count: `16 + 30 + 14 = 60` (Reconciled).
- **Diagonal Sum (Correct Classifications):**
  - `7 (Clean) + 26 (Stutter) + 9 (Filler) = 42`
  - **Overall Accuracy:** `42 / 60 = 0.7000` (**70.00%**)

---

## 3. Recomputed Per-Class Performance Metrics

Calculated using standard multi-class formulations:
- $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
- $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
- $\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- $\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$

| Class | Support | Pred Count | TP | FP | FN | TN | Precision | Recall | F1 Score | Specificity |
|---|---|---|---|---|---|---|---|---|---|---|
| **Clean** | 14 | 16 | 7 | 9 | 7 | 37 | **43.75%** (7/16) | **50.00%** (7/14) | **0.4667** | **80.43%** (37/46) |
| **Stutter** | 35 | 30 | 26 | 4 | 9 | 21 | **86.67%** (26/30) | **74.29%** (26/35) | **0.8000** | **84.00%** (21/25) |
| **Filler** | 11 | 14 | 9 | 5 | 2 | 44 | **64.29%** (9/14) | **81.82%** (9/11) | **0.7200** | **89.80%** (44/49) |

### Aggregate Summary Metrics
- **Macro F1 (Unweighted Arithmetic Mean):**  
  $$\text{Macro F1} = \frac{0.466667 + 0.800000 + 0.720000}{3} = \frac{1.986667}{3} = \mathbf{0.6622}$$
- **Support-Weighted F1:**  
  $$\text{Weighted F1} = \frac{0.466667 \times 14 + 0.800000 \times 35 + 0.720000 \times 11}{60} = \frac{6.533333 + 28.000000 + 7.920000}{60} = \frac{42.453333}{60} = \mathbf{0.7076}$$

---

## 4. Recomputed Binary Stutter Detection Metrics

Treating `Stutter` as the Positive class and `Clean + Filler` as the Negative class:

- **True Positives (TP):** `26` (Actual Stutter predicted as Stutter)
- **False Positives (FP):** `4` (Actual Clean: 4, Actual Filler: 0 predicted as Stutter)
- **False Negatives (FN):** `9` (Actual Stutter predicted as Clean: 7, as Filler: 2)
- **True Negatives (TN):** `21` (Actual Clean: 10, Actual Filler: 11 correctly non-stutter)
- **Total Samples:** `26 + 4 + 9 + 21 = 60`

### Binary Metrics
- **Stutter Precision:** $\frac{26}{26 + 4} = \frac{26}{30} = \mathbf{86.67\%}$ ($0.866667$)
- **Stutter Recall:** $\frac{26}{26 + 9} = \frac{26}{35} = \mathbf{74.29\%}$ ($0.742857$)
- **Stutter F1 Score:** $\frac{2 \times 0.866667 \times 0.742857}{0.866667 + 0.742857} = \mathbf{0.8000}$ ($0.800000$)
- **Stutter Specificity:** $\frac{21}{21 + 4} = \frac{21}{25} = \mathbf{84.00\%}$ ($0.840000$)

---

## 5. Cross-Comparison Against Project Artifacts & Discrepancy Audit

We cross-referenced every recomputed value against the three target artifacts:
1. [`ablation4_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/ablation4_metrics.json)
2. [`confusion_matrices.json`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/confusion_matrices.json)
3. [`ABLATION4_REPORT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/ABLATION4_REPORT.md)

| Metric / Record | Raw Recomputed | `ablation4_metrics.json` | `confusion_matrices.json` | `ABLATION4_REPORT.md` | Audit Status |
|---|---|---|---|---|---|
| **Accuracy** | **70.00%** | `0.7` (70.00%) | N/A | `70.00%` | **MATCH (100%)** |
| **Macro F1** | **0.6622** | `0.662222` | N/A | `0.6622` | **MATCH (100%)** |
| **Weighted F1** | **0.7076** | `0.707556` | N/A | `0.7076` | **MATCH (100%)** |
| **Stutter Precision** | **86.67%** | `0.866667` | N/A | `86.67%` | **MATCH (100%)** |
| **Stutter Recall** | **74.29%** | `0.742857` | N/A | `74.29%` | **MATCH (100%)** |
| **Stutter F1** | **0.8000** | `0.8` | N/A | `0.8000` | **MATCH (100%)** |
| **Stutter Specificity** | **84.00%** | `0.84` | N/A | `84.00%` | **MATCH (100%)** |
| **Stutter TP / FP / FN / TN** | **26 / 4 / 9 / 21** | `26 / 4 / 9 / 21` | N/A | `26 / 4 / 9 / 21` | **MATCH (100%)** |
| **Clean Prec / Rec / F1** | **43.75% / 50.00% / 0.4667** | `0.4375 / 0.50 / 0.4667` | N/A | `43.75% / 50.00% / 0.4667` | **MATCH (100%)** |
| **Filler Prec / Rec / F1** | **64.29% / 81.82% / 0.7200** | `0.6429 / 0.8182 / 0.7200` | N/A | `64.29% / 81.82% / 0.7200` | **MATCH (100%)** |
| **Overall Confusion Matrix** | Reconciled 3x3 | Matches Exactly | Matches Exactly | Matches Exactly | **MATCH (100%)** |

### Discrepancy Findings
- **Unresolved Discrepancies:** **0**
- All metrics in [`ablation4_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/ablation4_metrics.json), [`confusion_matrices.json`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/confusion_matrices.json), and [`ABLATION4_REPORT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/ABLATION4_REPORT.md) match the raw recomputations directly.

---

## 6. Step 3 Audit Verdict

### **STATUS: PASS**
Every required check is supported by raw prediction artifacts. All row/column supports reconcile to 60. All aggregate metrics are 100% verified.
