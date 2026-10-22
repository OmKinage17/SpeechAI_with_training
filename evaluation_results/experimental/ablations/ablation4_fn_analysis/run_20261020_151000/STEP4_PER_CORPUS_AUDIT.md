# SpeechAI — Step 4: Independent Per-Corpus Audit Report

**Audit Target Directory:** [`evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000)  
**Accepted Reference Candidate:** Ablation 3 (`evaluation_results/experimental/ablations/stutter_fn_recovery/run_20261020_144131/`)  
**Raw Source of Truth:** [`per_clip_predictions.json`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/per_clip_predictions.json) & [`run_manifest.json`](file:///E:/SpeechPrj/evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json)  
**Feature Cache SHA-256:** `91ad23deef8331ec52953412927ca4a08b2d113def0f5dc997f70918dbe4d076` (Verified)  
**Audit Step:** Step 4 — Independent Per-Corpus Audit  
**Audit Status:** **PASS**

---

## 1. Executive Summary & Corpus Partition Verification

The 60 benchmark clips were partitioned into their three constituent corpora using [`run_manifest.json`](file:///E:/SpeechPrj/evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json) (mapping `dataset: 'UCLASS'` to `SEP-28k`, `'FluencyBank'` to `FluencyBank`, and `'Libri_Stutter'` to `LibriSpeech`):

| Corpus | Target Count | Recomputed Unique Clips | Overlap | Missing Clips | Partition Status |
|---|---|---|---|---|---|
| **SEP-28k** (UCLASS) | 20 | 20 | 0 | 0 | **VERIFIED (20 Clips)** |
| **FluencyBank** | 20 | 20 | 0 | 0 | **VERIFIED (20 Clips)** |
| **LibriSpeech** (Libri_Stutter) | 20 | 20 | 0 | 0 | **VERIFIED (20 Clips)** |
| **Total Benchmark** | **60** | **60** | **0** | **0** | **EXACT PARTITION (100%)** |

- **Disjointness & Coverage:** $\text{SEP-28k} \cup \text{FluencyBank} \cup \text{LibriSpeech} = 60$ unique clips with zero pairwise intersection.
- **Verdict:** **PASS**. All per-corpus calculations reconcile and re-aggregate to the aggregate matrix without discrepancy.

---

## 2. Detailed Per-Corpus Recomputations

### 1. SEP-28k (UCLASS, 20 Clips)
- **Speakers Evaluated:** `M_1022`, `M_0061`
- **Ground-Truth Support:** Clean: **4**, Stutter: **11**, Filler: **5** (Total: 20)
- **Predicted Distribution:** Clean: **4**, Stutter: **10**, Filler: **6** (Total: 20)

#### 3x3 Confusion Matrix
```
                  Predicted Class
                Clean   Stutter  Filler   Total (Support)
Actual Clean       1        1       2        4
Actual Stutter     1        9       1       11
Actual Filler      2        0       3        5
Total (Pred)       4       10       6       20
```
- **Diagonal Sum (Correct Predictions):** `1 + 9 + 3 = 13`
- **Accuracy:** `13 / 20 = 0.6500` (**65.00%**)

#### Class-Wise Metrics
- **Clean:** TP=1, FP=3, FN=3, TN=13 | **Precision:** `1/4 = 25.00%` | **Recall:** `1/4 = 25.00%` | **F1:** `0.2500`
- **Stutter:** TP=9, FP=1, FN=2, TN=8 | **Precision:** `9/10 = 90.00%` | **Recall:** `9/11 = 81.82%` | **F1:** `0.8571`
- **Filler:** TP=3, FP=3, FN=2, TN=12 | **Precision:** `3/6 = 50.00%` | **Recall:** `3/5 = 60.00%` | **F1:** `0.5455`
- **Summary Metrics:**
  - **Macro F1 (All 3 Classes):** `(0.2500 + 0.8571 + 0.5455) / 3 =` **0.5509**
  - **Support-Weighted F1:** `(0.2500 * 4 + 0.8571 * 11 + 0.5455 * 5) / 20 = 13.1556 / 20 =` **0.6578**

---

### 2. FluencyBank (20 Clips)
- **Speakers Evaluated:** `Ep_017`, `Ep_031`
- **Ground-Truth Support:** Clean: **4**, Stutter: **10**, Filler: **6** (Total: 20)
- **Predicted Distribution:** Clean: **10**, Stutter: **4**, Filler: **6** (Total: 20)

#### 3x3 Confusion Matrix
```
                  Predicted Class
                Clean   Stutter  Filler   Total (Support)
Actual Clean       4        0       0        4
Actual Stutter     6        4       0       10
Actual Filler      0        0       6        6
Total (Pred)      10        4       6       20
```
- **Diagonal Sum (Correct Predictions):** `4 + 4 + 6 = 14`
- **Accuracy:** `14 / 20 = 0.7000` (**70.00%**)

#### Class-Wise Metrics
- **Clean:** TP=4, FP=6, FN=0, TN=10 | **Precision:** `4/10 = 40.00%` | **Recall:** `4/4 = 100.00%` | **F1:** `0.5714`
- **Stutter:** TP=4, FP=0, FN=6, TN=10 | **Precision:** `4/4 = 100.00%` | **Recall:** `4/10 = 40.00%` | **F1:** `0.5714`
- **Filler:** TP=6, FP=0, FN=0, TN=14 | **Precision:** `6/6 = 100.00%` | **Recall:** `6/6 = 100.00%` | **F1:** `1.0000`
- **Summary Metrics:**
  - **Macro F1 (All 3 Classes):** `(0.5714 + 0.5714 + 1.0000) / 3 =` **0.7143**
  - **Support-Weighted F1:** `(0.5714 * 4 + 0.5714 * 10 + 1.0000 * 6) / 20 = 14.0000 / 20 =` **0.7000**

---

### 3. LibriSpeech (Libri_Stutter, 20 Clips)
- **Speakers Evaluated:** `Reader_198`, `Reader_1743`
- **Ground-Truth Support:** Clean: **6**, Stutter: **14**, Filler: **0** (Total: 20)
- **Predicted Distribution:** Clean: **2**, Stutter: **16**, Filler: **2** (Total: 20)

#### 3x3 Confusion Matrix
```
                  Predicted Class
                Clean   Stutter  Filler   Total (Support)
Actual Clean       2        3       1        6
Actual Stutter     0       13       1       14
Actual Filler      0        0       0        0
Total (Pred)       2       16       2       20
```
- **Diagonal Sum (Correct Predictions):** `2 + 13 + 0 = 15`
- **Accuracy:** `15 / 20 = 0.7500` (**75.00%**)

#### Class-Wise Metrics & Zero-Support Handling
> [!IMPORTANT]
> **Metric Convention for Classes with Zero Ground-Truth Support:**
> In the LibriSpeech subset, there are **0 ground-truth Filler samples** ($\text{Support} = 0$). Two samples (1 Clean, 1 Stutter) were falsely predicted as Filler ($\text{FP} = 2, \text{TP} = 0$). Under standard scientific conventions:
> - $\text{Recall} = \frac{0}{0} \rightarrow 0.0$ (undefined/zero)
> - $\text{Precision} = \frac{0}{2} = 0.0\%$
> - $\text{F1} = 0.0000$
> 
> To provide rigorous reporting, we compute both standard macro averages:
> 1. **Active-Class Macro F1 (2 Classes: Clean, Stutter):** Averages only classes with non-zero support:
>    $$\text{Macro F1}_{\text{active}} = \frac{0.5000 + 0.8667}{2} = \mathbf{0.6833}$$
> 2. **Three-Class Macro F1 (All 3 Classes including zero-support Filler):**
>    $$\text{Macro F1}_{\text{all 3}} = \frac{0.5000 + 0.8667 + 0.0000}{3} = \frac{1.3667}{3} = \mathbf{0.4556}$$

- **Clean:** TP=2, FP=0, FN=4, TN=14 | **Precision:** `2/2 = 100.00%` | **Recall:** `2/6 = 33.33%` | **F1:** `0.5000`
- **Stutter:** TP=13, FP=3, FN=1, TN=3 | **Precision:** `13/16 = 81.25%` | **Recall:** `13/14 = 92.86%` | **F1:** `0.8667`
- **Filler (Zero Support):** TP=0, FP=2, FN=0, TN=18 | **Precision:** `0.00%` | **Recall:** `0.00%` | **F1:** `0.0000`
- **Support-Weighted F1:** `(0.5000 * 6 + 0.8667 * 14 + 0.0000 * 0) / 20 = 15.1333 / 20 =` **0.7567**

---

## 3. Matrix Re-Aggregation & Additivity Verification

Summing the three per-corpus confusion matrices element-wise:

$$C_{\text{overall}} = C_{\text{SEP-28k}} + C_{\text{FluencyBank}} + C_{\text{LibriSpeech}}$$

$$\begin{pmatrix}
1 & 1 & 2 \\
1 & 9 & 1 \\
2 & 0 & 3
\end{pmatrix} +
\begin{pmatrix}
4 & 0 & 0 \\
6 & 4 & 0 \\
0 & 0 & 6
\end{pmatrix} +
\begin{pmatrix}
2 & 3 & 1 \\
0 & 13 & 1 \\
0 & 0 & 0
\end{pmatrix} =
\begin{pmatrix}
7 & 4 & 3 \\
7 & 26 & 2 \\
2 & 0 & 9
\end{pmatrix}$$

### Reconciliation Check
- Re-aggregated Clean row: `[1+4+2, 1+0+3, 2+0+1] = [7, 4, 3]` (Sum: 14)
- Re-aggregated Stutter row: `[1+6+0, 9+4+13, 1+0+1] = [7, 26, 2]` (Sum: 35)
- Re-aggregated Filler row: `[2+0+0, 0+0+0, 3+6+0] = [2, 0, 9]` (Sum: 11)
- Re-aggregated Matrix Total: `14 + 35 + 11 = 60`
- Total Diagonal Sum: `7 + 26 + 9 = 42`
- Re-aggregated Accuracy: `42 / 60 = 70.00%`

The element-wise sum of the three per-corpus confusion matrices matches the Step 3 aggregate confusion matrix **100% exactly**.

---

## 4. Cross-Comparison Against Project Artifacts

We verified all per-corpus metrics against [`ablation4_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/ablation4_metrics.json), [`confusion_matrices.json`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/confusion_matrices.json), and [`ABLATION4_REPORT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/ABLATION4_REPORT.md):

| Corpus / Metric | Recomputed Value | `ablation4_metrics.json` | `confusion_matrices.json` | `ABLATION4_REPORT.md` | Audit Status |
|---|---|---|---|---|---|
| **SEP-28k Accuracy** | **65.00%** (13/20) | `0.65` | N/A | `65.00%` | **MATCH (100%)** |
| **SEP-28k Confusion Matrix** | `[1, 1, 2], [1, 9, 1], [2, 0, 3]` | Matches | Matches | Matches | **MATCH (100%)** |
| **SEP-28k Stutter Prec / Rec / F1** | **90.00% / 81.82% / 0.8571** | Matches | N/A | Matches | **MATCH (100%)** |
| **FluencyBank Accuracy** | **70.00%** (14/20) | `0.70` | N/A | `70.00%` | **MATCH (100%)** |
| **FluencyBank Confusion Matrix** | `[4, 0, 0], [6, 4, 0], [0, 0, 6]` | Matches | Matches | Matches | **MATCH (100%)** |
| **FluencyBank Stutter Prec / Rec / F1** | **100.00% / 40.00% / 0.5714** | Matches | N/A | Matches | **MATCH (100%)** |
| **LibriSpeech Accuracy** | **75.00%** (15/20) | `0.75` | N/A | `75.00%` | **MATCH (100%)** |
| **LibriSpeech Confusion Matrix** | `[2, 3, 1], [0, 13, 1], [0, 0, 0]` | Matches | Matches | Matches | **MATCH (100%)** |
| **LibriSpeech Stutter Prec / Rec / F1** | **81.25% / 92.86% / 0.8667** | Matches | N/A | Matches | **MATCH (100%)** |
| **LibriSpeech Zero-Support Filler** | `Support: 0, F1: 0.0000` | Matches | Matches | Matches | **MATCH (100%)** |

### Discrepancy Findings
- **Unresolved Discrepancies:** **0**
- All per-corpus values across all three corpora are completely reconciled and consistent with the raw per-clip prediction records.

---

## 5. Step 4 Audit Verdict

### **STATUS: PASS**
Partitioning is exact (20 clips per corpus, 0 overlap, 0 missing). Zero-support classes are explicitly handled and documented. Sum of per-corpus confusion matrices reproduces the aggregate matrix with 100% mathematical fidelity.
