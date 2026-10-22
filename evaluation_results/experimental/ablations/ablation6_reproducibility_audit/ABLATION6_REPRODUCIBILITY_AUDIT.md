# SpeechAI — Ablation 6: Independent Reproducibility Audit

**Audit Run Directory:** `evaluation_results/experimental/ablations/ablation6_reproducibility_audit/`  
**Timestamp:** `2026-10-20T15:50:00+05:30`  
**Auditor:** Antigravity Autonomous Agent (Pair Programming with User)  
**Primary Audit Target:** Ablation 3 Frozen Development Reference (`evaluation_results/experimental/ablations/stutter_fn_recovery/run_20261020_144131/`)  
**Validation Benchmark:** 60-Clip Generalization Validation Set (`evaluation_results/experimental/generalization_validation/run_20261020_132214/`)  
**Ablation 4 Reference:** `evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/`  
**Ablation 5 Reference:** `evaluation_results/experimental/ablations/ablation5_targeted_recovery/run_20261020_154500/`  
**Feature Cache Path:** `evaluation_results/experimental/generalization_validation/run_20261020_132214/feature_cache.json`  
**Expected Feature Cache SHA-256:** `91ad23deef8331ec52953412927ca4a08b2d113def0f5dc997f70918dbe4d076`  
**Final Audit Verdict:** **PASS**

---

## 1. Executive Summary & Audit Mandate

This report documents the independent, read-only reproducibility audit of the **Ablation 3 frozen development reference** for the SpeechAI system. 

In accordance with the audit protocol:
1. No detector code, thresholds, configurations, labels, manifests, or predictions were modified.
2. The frozen 750-clip holdout was not accessed or evaluated.
3. Every metric reported herein was recomputed directly from raw per-clip prediction records and authoritative manifest annotations. Precomputed summary JSON and Markdown files were treated strictly as unverified claims until mathematically proven.
4. All 10 tasks outlined in the audit specification were executed sequentially and verified.

**Summary of Core Findings:**
- **Feature Cache Integrity:** Exact match with SHA-256 `91ad23deef8331ec52953412927ca4a08b2d113def0f5dc997f70918dbe4d076`.
- **Dataset Alignment:** Exactly 60 unique clips with 100% 1-to-1 correspondence between ground truth and predictions.
- **Reference Accuracy:** **70.00%** (42 / 60 clips).
- **Macro F1:** **0.6622** | **Weighted F1:** **0.7076**.
- **Binary Stutter Metrics:** Precision: **86.67%** (26/30), Recall: **74.29%** (26/35), F1: **0.8000**, Specificity: **84.00%** (21/25).
- **Confusion Matrix Reconciliation:** The sum of the three per-corpus confusion matrices (`SEP-28k`, `FluencyBank`, `LibriSpeech`) matches the aggregate 60-clip confusion matrix cell-by-cell with **0 discrepancies**.
- **False Negatives:** Exactly the same 9 false-negative clips are verified.
- **Ablation 5 Isolation:** Verified that Ablation 5 halted after Phase 2 with decision `NO CHANGE` and zero changes to the frozen reference.
- **Test Suite Status:** 115 of 115 tests passed (`OK`, exit code 0).

---

## 2. Task 1: Cryptographic Hash Verification

All core artifacts were hashed using SHA-256 directly from disk:

| Artifact Description | File Path | Expected SHA-256 | Computed SHA-256 | Status |
|---|---|---|---|---|
| **Feature Cache** | `evaluation_results/experimental/generalization_validation/run_20261020_132214/feature_cache.json` | `91ad23deef8331ec52953412927ca4a08b2d113def0f5dc997f70918dbe4d076` | `91ad23deef8331ec52953412927ca4a08b2d113def0f5dc997f70918dbe4d076` | **EXACT MATCH** |
| **Generalization Manifest** | `evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json` | N/A (Authoritative) | `a86ec68d19cbd9430fe8a0c1d5fb927fcff73dd72f78090eb70d39cc841fd3e3` | **RECORDED** |
| **Ablation 3 Manifest** | `evaluation_results/experimental/ablations/stutter_fn_recovery/run_20261020_144131/run_manifest.json` | N/A (Authoritative) | `a3dee0199cac4b0ba84690b38e41f4941275f4a3034fa5fef248f734aca0acfe` | **RECORDED** |
| **Ablation 3 Predictions** | `evaluation_results/experimental/ablations/stutter_fn_recovery/run_20261020_144131/per_clip_predictions.json` | N/A (Authoritative) | `36cfc6e7811a28a621d163c47114b4bb6f2e3725a3fa763b3d1a92ecd48afa15` | **RECORDED** |
| **Ablation 3 Metrics JSON**| `evaluation_results/experimental/ablations/stutter_fn_recovery/run_20261020_144131/ablation3_metrics.json` | N/A (Authoritative) | `09e73ee68024a679ed20b8a813dee0c6a2fff5c874f8ab5eb94e8fed74b8babf` | **RECORDED** |
| **Ablation 4 Manifest** | `evaluation_results/experimental/ablations/ablation4_fn_analysis/run_20261020_151000/run_manifest.json` | N/A (Authoritative) | `d9c6aa85bb23f688d57d409d69c411beece11eeffa9fb25670eab603001b1d1f` | **RECORDED** |
| **Ablation 5 Manifest** | `evaluation_results/experimental/ablations/ablation5_targeted_recovery/run_20261020_154500/run_manifest.json` | N/A (Authoritative) | `9ea41247d7ff4fa5f3f9badd38186245ffc77aff269521d0584215d3aa55c70e` | **RECORDED** |

---

## 3. Task 3: Dataset Integrity, Supports & Speaker Separation

### 3.1 Clip Count and 1-to-1 Pairing
- **Generalization Manifest Count:** 60 unique clips.
- **Prediction Records Count:** 60 unique clips.
- **Set Difference:** $\emptyset$ (manifest filenames and prediction keys are 100% identical).

### 3.2 Authoritative Class Supports
In the authoritative manifest and raw prediction dictionaries:
- `gt_dominant == "Stutter"`: **35 clips** (58.33%)
- `gt_dominant == "Clean"`: **14 clips** (23.33%)
- `gt_dominant == "Filler"`: **11 clips** (18.33%)
- **Total Support:** $35 + 14 + 11 = 60$ clips.

Binary stutter ground truth:
- `Positive (Stutter)`: **35 clips**
- `Negative (Clean + Filler)`: **25 clips**

*(Note on reporting distinction in Task 8: In multi-label annotations where secondary fillers co-occur or are tallied independently, there are 15 purely fluent clips without any disfluency, 35 with stutter, and 10 with filler only. All formal 3-class matrix operations use the mutually exclusive dominant supports: Clean=14, Stutter=35, Filler=11).*

### 3.3 Corpus & Speaker Disjointness
The benchmark is balanced across three distinct corpora with zero speaker overlap:

| Corpus Name (Manifest / Pred) | Clip Count | Unique Speakers / Episodes | Speaker Identifiers |
|---|---|---|---|
| **SEP-28k / UCLASS** | 20 | 2 | `M_0061`, `M_1022` |
| **FluencyBank** | 20 | 2 | `Ep_017`, `Ep_031` |
| **LibriSpeech / Libri_Stutter** | 20 | 2 | `Reader_1743`, `Reader_198` |
| **Total** | **60** | **6** | **Disjoint across corpora** |

---

## 4. Tasks 2 & 4: Recomputed 3-Class Metrics & Confusion Matrix

All metrics were recomputed from raw pairs `(ground_truth, predicted_3class)`.

### 4.1 3-Class Confusion Matrix (Order: Clean, Stutter, Filler)

```
                 Predicted Clean   Predicted Stutter   Predicted Filler   Total True Support
True Clean:             7                  4                  3                   14
True Stutter:           7                 26                  2                   35
True Filler:            2                  0                  9                   11
Total Predicted:       16                 30                 14                   60
```

### 4.2 Recomputed 3-Class Metrics vs. Precomputed Reference

| Metric | Recomputed Value | Precomputed in `ablation3_metrics.json` | Discrepancy | Status |
|---|---|---|---|---|
| **Accuracy** | **0.7000** (42 / 60) | `0.7` | 0.0000 | **MATCH** |
| **Macro F1** | **0.6622** | `0.6622` | 0.0000 | **MATCH** |
| **Weighted F1** | **0.7076** | `0.7076` | 0.0000 | **MATCH** |
| **Clean Precision** | **0.4375** (7 / 16) | `0.4375` | 0.0000 | **MATCH** |
| **Clean Recall** | **0.5000** (7 / 14) | `0.5` | 0.0000 | **MATCH** |
| **Clean F1** | **0.4667** | `0.4667` | 0.0000 | **MATCH** |
| **Clean Specificity** | **0.8043** (37 / 46) | `0.8043` | 0.0000 | **MATCH** |
| **Stutter Precision** | **0.8667** (26 / 30) | `0.8667` | 0.0000 | **MATCH** |
| **Stutter Recall** | **0.7429** (26 / 35) | `0.7429` | 0.0000 | **MATCH** |
| **Stutter F1** | **0.8000** | `0.8` | 0.0000 | **MATCH** |
| **Stutter Specificity** | **0.8400** (21 / 25) | `0.84` | 0.0000 | **MATCH** |
| **Filler Precision** | **0.6429** (9 / 14) | `0.6429` | 0.0000 | **MATCH** |
| **Filler Recall** | **0.8182** (9 / 11) | `0.8182` | 0.0000 | **MATCH** |
| **Filler F1** | **0.7200** | `0.72` | 0.0000 | **MATCH** |
| **Filler Specificity** | **0.8980** (44 / 49) | `0.898` | 0.0000 | **MATCH** |

---

## 5. Task 5: Recomputed Binary Stutter Metrics

### 5.1 Explicit Positive-Class Definition
- **Positive Class (1):** Ground truth `Stutter`.
- **Negative Class (0):** Ground truth `Clean` or `Filler`.
- **Predicted Positive:** Prediction `Stutter`.
- **Predicted Negative:** Prediction `Clean` or `Filler`.

### 5.2 Contingency Table
- **True Positives (TP):** $cm[\text{Stutter}][\text{Stutter}] = \mathbf{26}$
- **False Positives (FP):** $cm[\text{Clean}][\text{Stutter}] + cm[\text{Filler}][\text{Stutter}] = 4 + 0 = \mathbf{4}$
- **False Negatives (FN):** $cm[\text{Stutter}][\text{Clean}] + cm[\text{Stutter}][\text{Filler}] = 7 + 2 = \mathbf{9}$
- **True Negatives (TN):** $cm[\text{Clean}][\text{Clean}] + cm[\text{Clean}][\text{Filler}] + cm[\text{Filler}][\text{Clean}] + cm[\text{Filler}][\text{Filler}] = 7 + 3 + 2 + 9 = \mathbf{21}$
- **Total:** $26 + 4 + 9 + 21 = \mathbf{60}$

### 5.3 Binary Metric Verification

| Metric | Formula | Recomputed Exact | Precomputed Reference | Status |
|---|---|---|---|---|
| **Binary Accuracy** | $(TP + TN) / \text{Total} = 47 / 60$ | **0.7833** | `0.7833` | **MATCH** |
| **Precision** | $TP / (TP + FP) = 26 / 30$ | **0.8667** | `0.8667` | **MATCH** |
| **Recall (Sensitivity)**| $TP / (TP + FN) = 26 / 35$ | **0.7429** | `0.7429` | **MATCH** |
| **F1 Score** | $2 \cdot P \cdot R / (P + R)$ | **0.8000** | `0.8000` | **MATCH** |
| **Specificity** | $TN / (TN + FP) = 21 / 25$ | **0.8400** | `0.8400` | **MATCH** |

---

## 6. Task 6: Per-Corpus Metrics & Matrix Sum Reconciliation

### 6.1 Per-Corpus Confusion Matrices

#### A. FluencyBank (20 clips)
- **Supports:** Stutter: 10, Filler: 6, Clean: 4
- **Confusion Matrix:**
  ```
  True Clean:    4 Clean,  0 Stutter,  0 Filler
  True Stutter:  6 Clean,  4 Stutter,  0 Filler
  True Filler:   0 Clean,  0 Stutter,  6 Filler
  ```
- **Accuracy:** 70.00% (14 / 20)
- **Binary Stutter:** TP=4, FP=0, FN=6, TN=10 | Prec=1.0000, Rec=0.4000, F1=0.5714, Spec=1.0000

#### B. LibriSpeech / Libri_Stutter (20 clips)
- **Supports:** Stutter: 14, Clean: 6, Filler: 0
- **Confusion Matrix:**
  ```
  True Clean:    2 Clean,  3 Stutter,  1 Filler
  True Stutter:  0 Clean, 13 Stutter,  1 Filler
  True Filler:   0 Clean,  0 Stutter,  0 Filler
  ```
- **Accuracy:** 75.00% (15 / 20)
- **Binary Stutter:** TP=13, FP=3, FN=1, TN=3 | Prec=0.8125, Rec=0.9286, F1=0.8667, Spec=0.5000

#### C. SEP-28k / UCLASS (20 clips)
- **Supports:** Stutter: 11, Filler: 5, Clean: 4
- **Confusion Matrix:**
  ```
  True Clean:    1 Clean,  1 Stutter,  2 Filler
  True Stutter:  1 Clean,  9 Stutter,  1 Filler
  True Filler:   2 Clean,  0 Stutter,  3 Filler
  ```
- **Accuracy:** 65.00% (13 / 20)
- **Binary Stutter:** TP=9, FP=1, FN=2, TN=8 | Prec=0.9000, Rec=0.8182, F1=0.8571, Spec=0.8889

### 6.2 Matrix Sum Reconciliation
Summing each cell across FluencyBank + LibriSpeech + SEP-28k:
- Clean $\rightarrow$ Clean: $4 + 2 + 1 = \mathbf{7}$ (Aggregate: 7) $\rightarrow$ **MATCH**
- Clean $\rightarrow$ Stutter: $0 + 3 + 1 = \mathbf{4}$ (Aggregate: 4) $\rightarrow$ **MATCH**
- Clean $\rightarrow$ Filler: $0 + 1 + 2 = \mathbf{3}$ (Aggregate: 3) $\rightarrow$ **MATCH**
- Stutter $\rightarrow$ Clean: $6 + 0 + 1 = \mathbf{7}$ (Aggregate: 7) $\rightarrow$ **MATCH**
- Stutter $\rightarrow$ Stutter: $4 + 13 + 9 = \mathbf{26}$ (Aggregate: 26) $\rightarrow$ **MATCH**
- Stutter $\rightarrow$ Filler: $0 + 1 + 1 = \mathbf{2}$ (Aggregate: 2) $\rightarrow$ **MATCH**
- Filler $\rightarrow$ Clean: $0 + 0 + 2 = \mathbf{2}$ (Aggregate: 2) $\rightarrow$ **MATCH**
- Filler $\rightarrow$ Stutter: $0 + 0 + 0 = \mathbf{0}$ (Aggregate: 0) $\rightarrow$ **MATCH**
- Filler $\rightarrow$ Filler: $6 + 0 + 3 = \mathbf{9}$ (Aggregate: 9) $\rightarrow$ **MATCH**

The per-corpus confusion matrices sum to the aggregate confusion matrix with **100% exact cell-by-cell equivalence**.

---

## 7. Task 7: Independent False-Negative & Ablation 5 Isolation Verification

### 7.1 Independent False-Negative Verification
Filtering all clips where `ground_truth == "Stutter"` and `ablation3.predicted_3class != "Stutter"` yields exactly **9 clips**:

| Clip ID | Predicted Class | Corpus | Clinical Annotations |
|---|---|---|---|
| `1743-142912-0001.flac` | `Filler` | LibriSpeech | `{'stutter_types': ['3', '0', '5']}` |
| `FluencyBank_017_15.wav` | `Clean` | FluencyBank | `{'sound_rep': 0, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 1}` |
| `FluencyBank_017_34.wav` | `Clean` | FluencyBank | `{'sound_rep': 0, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 0}` |
| `FluencyBank_017_42.wav` | `Clean` | FluencyBank | `{'sound_rep': 0, 'word_rep': 0, 'prolongation': 0, 'block': 1, 'interjection': 0}` |
| `FluencyBank_017_52.wav` | `Clean` | FluencyBank | `{'sound_rep': 0, 'word_rep': 0, 'prolongation': 0, 'block': 1, 'interjection': 0}` |
| `FluencyBank_031_14.wav` | `Clean` | FluencyBank | `{'sound_rep': 0, 'word_rep': 0, 'prolongation': 0, 'block': 1, 'interjection': 0}` |
| `FluencyBank_031_9.wav` | `Clean` | FluencyBank | `{'sound_rep': 1, 'word_rep': 0, 'prolongation': 0, 'block': 0, 'interjection': 0}` |
| `M_0061_16y9m-1_dysfluent_060.wav` | `Clean` | SEP-28k | `{'sound_rep': 0, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 0}` |
| `M_1022_11y3m_1_dysfluent_019.wav` | `Filler` | SEP-28k | `{'sound_rep': 0, 'word_rep': 1, 'prolongation': 0, 'block': 0, 'interjection': 0}` |

### 7.2 Ablation 5 Non-Destructive Isolation Check
Inspection of [`run_manifest.json`](file:///e:/SpeechPrj/evaluation_results/experimental/ablations/ablation5_targeted_recovery/run_20261020_154500/run_manifest.json) confirms:
- `execution_status`: `"Phase 1 & Phase 2 Completed. Phase 3 Halted."`
- `final_recommendation`: `"NO CHANGE"`
- `decision`: `"NO CHANGE"`
- Both candidate hypotheses (H1: Non-adjacent repetition, H2: Sub-500ms prolongation) were evaluated counterfactually and rejected prior to implementation.
- **Reference Immutability:** Zero prediction records, weights, thresholds, or detector code files were modified.

---

## 8. Task 8: Denominators, Zero-Division & Convention Checks

An audit of all calculation formulas across the evaluation pipeline confirmed:
1. **Denominators:**
   - Accuracy: Denominator is 60 (total samples).
   - Stutter Precision: Denominator is $TP + FP = 26 + 4 = 30$.
   - Stutter Recall: Denominator is $TP + FN = 26 + 9 = 35$.
   - Stutter Specificity: Denominator is $TN + FP = 21 + 4 = 25$.
   - Macro F1: Denominator is 3 (unweighted average of class F1 scores).
   - Weighted F1: Denominator is 60 (weighted by true class supports: Clean=14, Stutter=35, Filler=11).
2. **Zero-Division Handling:**
   - The evaluator enforces standard safe division (`0.0` if denominator is 0). In this benchmark, all classes have non-zero supports and positive predictions; zero division did not occur.
3. **Class Support Conventions:**
   - Dominant 3-class support is Clean: 14, Stutter: 35, Filler: 11.
   - Occasional narrative mentions in earlier informal notes of "Clean: 15, Stutter: 35, Filler: 10" reflect multi-label pure-fluent tallies (clips with zero dysfluency annotations of any kind).
   - All actual metrics in code, JSON, and confusion matrices use the mutually exclusive dominant labels (14 / 35 / 11), with zero mathematical inconsistency.

---

## 9. Task 9: Reference Immutability & Clean Workspace Audit

A git status check confirmed that the core reference run directories remain strictly read-only and uncommitted files are restricted to documentation and experimental audit scratchpads:
- `stutter_fn_recovery/run_20261020_144131/` is **completely untouched**.
- `generalization_validation/run_20261020_132214/` is **completely untouched**.
- All baseline detector files remain in their verified state.

---

## 10. Task 10: Backend Test Suite Verification

The backend test suite was executed in read-only verification mode:
- **Exact Command:** `backend/venv/Scripts/python.exe -m unittest discover -s backend/tests`
- **Execution Output:**
  ```
  Ran 115 tests in 27.570s
  OK
  ```
- **Exit Code:** `0`
- **Total Tests Run:** 115
- **Tests Passed:** 115 (100%)
- **Tests Failed:** 0
- **Tests Errored:** 0

---

## 11. Final Audit Verdict

| Task # | Audit Task Description | Result | Details |
|---|---|---|---|
| **1** | SHA-256 Hash Verification | **PASS** | Feature cache hash `91ad23...` and all manifests strictly verified |
| **2** | Independent Metric Recomputation | **PASS** | 100% agreement with raw paired prediction records |
| **3** | Dataset Supports & Clip Correspondence | **PASS** | Exactly 60 unique clips; 1-to-1 match across manifest & predictions |
| **4** | 3-Class Metrics & Confusion Matrix | **PASS** | Acc: 70.00%, Macro F1: 0.6622, Weighted F1: 0.7076 |
| **5** | Binary Stutter Metrics | **PASS** | Prec: 86.67%, Rec: 74.29%, F1: 0.8000, Spec: 84.00% (TP=26, FP=4, FN=9, TN=21) |
| **6** | Per-Corpus Matrices Reconciliation | **PASS** | Per-corpus confusion matrices sum cell-by-cell to aggregate matrix |
| **7** | False Negatives & Ablation 5 Isolation | **PASS** | Exactly 9 FNs verified; Ablation 5 confirmed non-destructive `NO CHANGE` |
| **8** | Denominators & Conventions Check | **PASS** | Mathematically rigorous, standard zero-division, supports reconciled |
| **9** | Reference File Immutability | **PASS** | All reference artifacts and baseline files unmodified |
| **10** | Backend Test Suite Execution | **PASS** | 115 of 115 tests passing (`OK`, exit code 0) |

**FINAL VERDICT:** **PASS**
The Ablation 3 development reference results are fully reproducible, internally consistent, mathematically sound, and correctly reported.
