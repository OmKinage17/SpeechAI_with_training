# SpeechAI — Ablation 8: Metric Consistency & Audit Report

**Audit Timestamp:** 2026-10-20T16:29:04.070904  
**Target Benchmark:** Expanded 120-Clip Validation Set (`run_expanded_120`)  
**Detector Reference:** Frozen Ablation 3 (`run_20261020_144131`)  
**Auditor:** Antigravity Autonomous Diagnostic Engine  
**Status:** **PASSED (100% RECONCILED & CONSISTENT)**

---

## 1. Executive Summary & Verification Matrix

An independent forensic audit was conducted on all raw prediction records in `per_clip_predictions.json` to verify that all reported multi-class, binary stutter, per-corpus, and per-entity metrics in `expanded_validation_metrics.json` and `confusion_matrices.json` reconcile with zero mathematical discrepancy.

| Verification Item | Specification / Pre-Registration | Verified Value | Discrepancy | Status |
|---|---|---|---|---|
| **Total Clip Count** | Exactly 120 unique clips | 120 clips | 0 | **PASS** |
| **Corpora Partition** | Exactly 40 clips/corpus (3 corpora) | 40 UCLASS, 40 FB, 40 Libri | 0 | **PASS** |
| **Source Entity Diversity** | Exactly 12 distinct entities (10 clips/ent) | 12 entities, 10 clips each | 0 | **PASS** |
| **Partition Disjointness** | 0 overlap with Dev30, Val60, Holdout750 | 0 overlap | 0 | **PASS** |
| **Holdout Protection** | Strict zero-touch guarantee | 0 accesses to Holdout750 | 0 | **PASS** |
| **Aggregate Accuracy** | (TP_clean + TP_stutter + TP_filler) / 120 | 64 / 120 = 53.33% | 0.00% | **PASS** |
| **Binary Stutter F1** | 2 * P * R / (P + R) | 0.6557 (TP=40, FP=15, FN=27, TN=38) | 0.0000 | **PASS** |
| **Cell-by-Cell Sum** | Corpus CMs sum to Aggregate CM | 100% Exact match | 0 | **PASS** |
| **Backend Test Suite** | Full regression pass | 115 / 115 Passed | 0 | **PASS** |

---

## 2. Recomputed Confusion Matrices

### 2.1 Multiclass 3-Class Confusion Matrix ($N=120$)
Class hierarchy: Stutter > Filler > Clean

| Ground Truth \ Predicted | Clean | Stutter | Filler | Total Support | Class Recall | Class Precision | Class F1 | Specificity |
|---|---|---|---|---|---|---|---|---|
| **Clean** | **13** | 12 | 7 | **32** | 40.62% | 36.11% | 0.3824 | 73.86% |
| **Stutter** | 16 | **40** | 11 | **67** | 59.70% | 72.73% | 0.6557 | 71.70% |
| **Filler** | 7 | 3 | **11** | **21** | 52.38% | 37.93% | 0.4400 | 81.82% |
| **Total Predicted** | **36** | **55** | **29** | **120** | **Accuracy:** 53.33% | **Macro F1:** 0.4927 | **Weighted F1:** 0.5451 | — |

### 2.2 Binary Stutter Contingency Table ($N=120$)
- **Positive Class:** Stutter
- **Negative Class:** Clean or Filler

| | Predicted Stutter | Predicted Non-Stutter | Total |
|---|---|---|---|
| **Actual Stutter** | **TP = 40** | **FN = 27** | **67** |
| **Actual Non-Stutter** | **FP = 15** | **TN = 38** | **53** |
| **Total** | **55** | **65** | **120** |

- **Binary Stutter Accuracy:** 65.00%
- **Binary Stutter Precision:** 72.73%
- **Binary Stutter Recall:** 59.70%
- **Binary Stutter F1:** 0.6557
- **Binary Stutter Specificity:** 71.70%

---

## 3. Per-Corpus Cell-by-Cell Reconciliation

| Corpus | Clips | Multiclass Acc | Macro F1 | TP | FP | FN | TN | Precision | Recall | Stutter F1 | Specificity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **UCLASS** | 40 | 42.5% | 0.4088 | 9 | 4 | 12 | 15 | 69.2% | 42.9% | 0.5294 | 79.0% |
| **FluencyBank** | 40 | 45.0% | 0.4511 | 6 | 3 | 12 | 19 | 66.7% | 33.3% | 0.4444 | 86.4% |
| **Libri_Stutter** | 40 | 72.5% | 0.4136 | 25 | 8 | 3 | 4 | 75.8% | 89.3% | 0.8197 | 33.3% |
| **Total / Reconciled** | **120** | **53.3%** | **0.4927** | **40** | **15** | **27** | **38** | **72.7%** | **59.7%** | **0.6557** | **71.7%** |

**Reconciliation Verification:**
- $\sum \text{TP} = 9 + 6 + 25 = 40 \equiv 40$ (Match)
- $\sum \text{FP} = 4 + 3 + 8 = 15 \equiv 15$ (Match)
- $\sum \text{FN} = 12 + 12 + 3 = 27 \equiv 27$ (Match)
- $\sum \text{TN} = 15 + 19 + 4 = 38 \equiv 38$ (Match)
- All matrix cells sum perfectly to the aggregate multi-class and binary tables with zero remainder.

---

## 4. Per-Entity Stratification (12 Entities)

| Entity / Speaker | Corpus | Clips | Accuracy | Macro F1 | TP | FP | FN | TN | Stutter F1 |
|---|---|---|---|---|---|---|---|---|---|
| `Ep_063` | FluencyBank | 10 | 60.0% | 0.5738 | 1 | 0 | 3 | 6 | 0.4000 |
| `Ep_089` | FluencyBank | 10 | 30.0% | 0.3360 | 1 | 1 | 4 | 4 | 0.2857 |
| `Ep_099` | FluencyBank | 10 | 30.0% | 0.3056 | 1 | 1 | 3 | 5 | 0.3333 |
| `Ep_178` | FluencyBank | 10 | 60.0% | 0.5778 | 3 | 1 | 2 | 4 | 0.6667 |
| `M_1023` | UCLASS | 10 | 40.0% | 0.3056 | 3 | 1 | 2 | 4 | 0.6667 |
| `M_1099` | UCLASS | 10 | 60.0% | 0.6032 | 2 | 0 | 3 | 5 | 0.5714 |
| `M_1104` | UCLASS | 10 | 20.0% | 0.1786 | 1 | 0 | 5 | 4 | 0.2857 |
| `M_1105` | UCLASS | 10 | 50.0% | 0.4818 | 3 | 3 | 2 | 2 | 0.5455 |
| `Reader_1867` | Libri_Stutter | 10 | 90.0% | 0.5778 | 7 | 1 | 0 | 2 | 0.9333 |
| `Reader_1926` | Libri_Stutter | 10 | 60.0% | 0.3492 | 5 | 2 | 2 | 1 | 0.7143 |
| `Reader_2289` | Libri_Stutter | 10 | 70.0% | 0.4000 | 6 | 2 | 1 | 1 | 0.8000 |
| `Reader_2989` | Libri_Stutter | 10 | 70.0% | 0.2745 | 7 | 3 | 0 | 0 | 0.8235 |

---

## 5. Artifact Cryptographic Hashes

| File | SHA-256 Hash | Integrity Verified |
|---|---|---|
| `feature_cache_expanded.json` | `9502a7783228175e8c22084df24cf51779f2d6761bb764f1b1eff49e3e4c733f` | **VERIFIED** |
| `run_manifest.json` | `fdfa62ce5a85c22b57db013fbee8451f84777058e671ae2eb897f8c121e4fb00` | **VERIFIED** |
| `per_clip_predictions.json` | `068c05c0339e28e1cafa3e55a37199f97748eccda93046fc09ca27467947ee5f` | **VERIFIED** |
| `expanded_validation_metrics.json` | `2c841226b7fa446e4b34aea6e61a939204282b0537112fcbf3420cd7d95e88f2` | **VERIFIED** |
| `confusion_matrices.json` | `b043383d43e13e396cc54eda4dbca461d08350eb6211407351a237cd7cb54368` | **VERIFIED** |

---

## 6. Audit Verdict

**FINAL VERDICT: PASS**  
All 120 prediction records, class assignments, confusion matrices, bootstrap confidence intervals, and per-entity metrics are mathematically verified, mutually disjoint, and internally consistent.
