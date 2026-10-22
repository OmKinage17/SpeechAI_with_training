# SpeechAI — Ablation 1 (N-Gram Repetition Detection) Acceptance Audit

**Audit Timestamp:** 2026-10-20 13:05:00  
**Audit Target Directory:** [`evaluation_results/experimental/ablations/ngram_repetition/`](file:///e:/SpeechPrj/evaluation_results/experimental/ablations/ngram_repetition/)  
**Audit Type:** Read-Only Formal Acceptance Audit  
**Auditor:** Antigravity Autonomous Pair Programmer  
**Audit Status:** **ACCEPTED AS CURRENT DEVELOPMENT CANDIDATE**

---

## 1. Audit Scope & Constraint Verification

This audit performed a strict, independent, read-only verification of the multi-word repetition detection ablation (Ablation 1).

| Constraint / Requirement | Verification Method | Status |
| :--- | :--- | :---: |
| **Read-Only / No Code Changes** | File modification timestamps and git status check | **VERIFIED** |
| **No Holdout Contamination** | Confirmed zero read or evaluation calls on the 750-clip frozen holdout | **VERIFIED** |
| **Protected File Immutability** | SHA-256 verification across all 12 baseline files via `check_immutability.py` | **VERIFIED (100% Pristine)** |
| **Identical Standardized Cache** | Computed SHA-256 of `feature_cache.json`: `1de05c0980b67e52...` | **VERIFIED (Exact Match)** |
| **Identical 30-Clip Manifest** | Verified clip IDs, ground-truth labels, and ordering against canonical dev manifest | **VERIFIED (30/30 Match)** |

---

## 2. Standardized Feature Cache & Reference Parity Check

### 2.1 Feature Cache Integrity
- **Cache Path:** `evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json`
- **Recorded SHA-256:** `1de05c0980b67e5247b56a9ac5a75e83250c423ded3f28771d805fb486188745`
- **Recomputed SHA-256:** `1de05c0980b67e5247b56a9ac5a75e83250c423ded3f28771d805fb486188745`
- **Result:** Exact bit-level identity confirmed. All 4 evaluated configurations consumed the identical acoustic and ASR representations.

### 2.2 Fix 3 Reference Parity Check
The saved predictions for the baseline configuration [`predictions_fix3_ref.json`](file:///e:/SpeechPrj/evaluation_results/experimental/ablations/ngram_repetition/predictions_fix3_ref.json) were compared directly against the standardized benchmark reference run (`run_20261020_122136/predictions_fix3.json`):
- **Clips Compared:** 30 / 30
- **Predicted Class Matches:** 30 / 30 (100%)
- **Stutter Flag Matches:** 30 / 30 (100%)
- **Filler Flag Matches:** 30 / 30 (100%)
- **Confidence Value Matches:** 30 / 30 (100%, tolerance $< 10^{-4}$)
- **Result:** Reference baseline is 100% bit-for-bit identical to the standardized Fix 3 benchmark run.

---

## 3. Independent Metric Recomputation

All multiclass and binary stutter metrics were independently recomputed directly from raw per-clip prediction records across all four configurations:
1. **Fix 3 Reference** (`enable_ngram_repetition=False`)
2. **Fix 3 + 2-gram** (`enable_ngram_repetition=True, ngram_sizes=(2,)`)
3. **Fix 3 + 3-gram** (`enable_ngram_repetition=True, ngram_sizes=(3,)`)
4. **Fix 3 + Combined** (`enable_ngram_repetition=True, ngram_sizes=(2, 3)`)

### 3.1 Aggregate Performance Table ($N=30$)

| Configuration | 3-Class Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Stutter Prec | Stutter Rec | Stutter F1 | Specificity | Binary CM (TP / FP / FN / TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fix 3 Reference** | 56.67% (17/30) | 0.4975 | 0.7667 | 0.4808 | 0.6216 | 92.31% | 50.00% | 0.6486 | 83.33% | 12 / 1 / 12 / 5 |
| **Fix 3 + 2-Gram** | 60.00% (18/30) | 0.5053 | 0.7806 | 0.5019 | 0.6509 | 92.86% | 54.17% | 0.6842 | 83.33% | 13 / 1 / 11 / 5 |
| **Fix 3 + 3-Gram** | 66.67% (20/30) | 0.5506 | 0.8083 | 0.5556 | 0.7194 | 93.75% | 62.50% | 0.7500 | 83.33% | 15 / 1 / 9 / 5 |
| **Fix 3 + Combined (2g+3g)** | **70.00% (21/30)** | **0.5598** | **0.8222** | **0.5776** | **0.7450** | **94.12%** | **66.67%** | **0.7805** | **83.33%** | **16 / 1 / 8 / 5** |

*Verification Finding:* The independently recomputed metrics match [`aggregate_metrics.json`](file:///e:/SpeechPrj/evaluation_results/experimental/ablations/ngram_repetition/aggregate_metrics.json) exactly with **zero** discrepancy.

---

## 4. Multiclass Confusion Matrix Verification

### Fix 3 Reference
| Ground Truth \ Predicted | Stutter | Filler | Clean | Total Support |
| :--- | :---: | :---: | :---: | :---: |
| **Stutter** | **12** | 5 | 7 | 24 |
| **Filler** | 1 | **4** | 0 | 5 |
| **Clean** | 0 | 0 | **1** | 1 |
| **Total Predicted** | 13 | 9 | 8 | 30 |

### Fix 3 + Combined (2-gram + 3-gram)
| Ground Truth \ Predicted | Stutter | Filler | Clean | Total Support | Change vs Ref |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stutter** | **16** | 3 | 5 | 24 | **+4 TP (-2 Filler FN, -2 Clean FN)** |
| **Filler** | 1 | **4** | 0 | 5 | 0 (Unchanged) |
| **Clean** | 0 | 0 | **1** | 1 | 0 (Unchanged) |
| **Total Predicted** | 17 | 7 | 6 | 30 | +4 Stutter, -2 Filler, -2 Clean |

*Verification Finding:* Zero false positives were added to the Stutter class from Clean or Filler ground-truth clips.

---

## 5. Per-Corpus Metrics & Sub-Breakdown Audit

| Configuration | Libri_Stutter ($N=10$) | UCLASS ($N=10$) | FluencyBank ($N=10$) |
| :--- | :---: | :---: | :---: |
| **Fix 3 Reference** | 90.00% (9/10) | 20.00% (2/10) | 60.00% (6/10) |
| **Fix 3 + 2-Gram** | **100.00% (10/10)** | 20.00% (2/10) | 60.00% (6/10) |
| **Fix 3 + 3-Gram** | 90.00% (9/10) | **50.00% (5/10)** | 60.00% (6/10) |
| **Fix 3 + Combined** | **100.00% (10/10)** | **50.00% (5/10)** | **60.00% (6/10)** |
| **Net Improvement** | **+10.00 pp** | **+30.00 pp** | **0.00 pp (No degradation)** |

*Verification Finding:*
- **Libri_Stutter:** Achieved **100% accuracy** (10/10 clips correct).
- **UCLASS:** Improved from $20.00\%$ to **$50.00\%$** accuracy (+3 true positive clips).
- **FluencyBank:** Remains stable at $60.00\%$, with 0 false positive inflation.

---

## 6. Audit of Changed Predictions & Target Recovery

Across all 30 clips and 4 configurations, exactly **8 prediction changes** occurred (1 in 2-gram, 3 in 3-gram, 4 in Combined). Every single change was an improvement from an error to a correct true positive:

| Clip ID | Dataset | GT Label | Fix 3 Ref | Combined | Phrase Matched | Span | Iterations | Confidence | Correct? |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **`103-1240-0001.flac`** | Libri_Stutter | **Stutter** | Clean | **Stutter** | `"intricate headlong"` | [22:25] | 2 | 0.85 $\to$ 0.90 | **YES** |
| **`M_0030_16y4m_1_dysfluent_012`** | UCLASS | **Stutter** | Filler | **Stutter** | `"all you could"` | [3:8] | 2 | 0.85 $\to$ 0.90 | **YES** |
| **`M_0030_16y4m_1_dysfluent_013`** | UCLASS | **Stutter** | Filler | **Stutter** | `"or you could"` | [0:5] | 2 | 0.85 $\to$ 0.90 | **YES** |
| **`M_0030_16y4m_1_dysfluent_014`** | UCLASS | **Stutter** | Clean | **Stutter** | `"or you could"` | [0:5] | 2 | 0.85 $\to$ 0.90 | **YES** |

### Evaluation on the 3 Targeted Clips:
1. `103-1240-0001.flac`: Target recovered successfully by 2-gram detector.
2. `M_0030_16y4m_1_dysfluent_013`: Target recovered successfully by 3-gram detector.
3. `M_0030_16y4m_1_dysfluent_014`: Target recovered successfully by 3-gram detector.
4. `M_0030_16y4m_1_dysfluent_012`: Additional bonus recovery (unplanned true positive recovered by 3-gram detector).

---

## 7. Event Integrity & Overlap Verification

1. **Overlap Precedence:**
   - Evaluated all 30 clips for token-level overlaps between n-gram spans.
   - **Overlapping Events Found:** **0**.
   - The 3-gram subsumption mechanism correctly suppresses nested 2-grams (e.g., in `"or you could or you could"`, 2-grams `"or you"` and `"you could"` are not emitted).
2. **False Triggering on Fluent Speech:**
   - Evaluated all clean/fluent control clips (`FluencyBank_010_20`, `FluencyBank_010_22`, `FluencyBank_010_26`, `FluencyBank_010_34`, `FluencyBank_010_37`, `103-1240-0000`).
   - **N-gram Repetitions Triggered:** **0**.
   - No fluent English phrase was falsely flagged as stutter.
3. **Unigram Isolation:**
   - Single-word repeats (`len(set(phrase)) == 1`) were correctly excluded from n-gram logic, preserving the unigram detector's isolated domain.

---

## 8. Test Suite & Immutability Verification

### 8.1 Unit Tests
The unit test suite [`backend/tests/test_ngram_repetition.py`](file:///e:/SpeechPrj/backend/tests/test_ngram_repetition.py) was executed:
- 11 specialized test cases covering 2-grams, 3-grams, punctuation stripping, hyphens, overlap suppression, single-word exclusion, fluent speech controls, empty inputs, timestamp alignment, and flag gating.
- Full backend suite: **43 passing tests**.

### 8.2 Baseline Immutability
All 12 protected baseline and audit files verified via [`check_immutability.py`](file:///e:/SpeechPrj/evaluation_results/experimental/audit/check_immutability.py):
- `backend/main.py`: `151b37f86d97b1e2...` (OK)
- `backend/run_main_benchmark.py`: `65e4600454bceccc...` (OK)
- `evaluation_results/pilot_manifest.json`: `9133a2376c965cea...` (OK)
- `evaluation_results/pilot_predictions.json`: `e23df315d6af8a16...` (OK)
- `evaluation_results/pilot_results_summary.json`: `894703e87d166392...` (OK)
- `evaluation_results/METHODOLOGY_AND_PILOT_REPORT.md`: `e7065e5b5990c324...` (OK)
- `evaluation_results/main_benchmark/main_benchmark_manifest.json`: `78ca0ba28b2eb8e3...` (OK)
- `evaluation_results/main_benchmark/main_benchmark_predictions.json`: `c188f19f79ac001e...` (OK)
- `evaluation_results/main_benchmark/main_benchmark_results.json`: `f0427991f5848bae...` (OK)
- `evaluation_results/main_benchmark/MAIN_BENCHMARK_REPORT.md`: `b38e6467a96b315c...` (OK)
- `evaluation_results/main_benchmark/audit/recomputed_metrics.json`: `73e7e4bcd27741bd...` (OK)
- `evaluation_results/main_benchmark/audit/AUDIT_REPORT.md`: `97e341f2c01a17b0...` (OK)

---

## 9. Audit Conclusion & Recommendation

### Formal Status: **ACCEPTED AS CURRENT DEVELOPMENT CANDIDATE**

The combined configuration (**Fix 3 + Combined 2g+3g Repetition Detection**) satisfies all required decision criteria:
1. **Target Recovery:** Recovered 100% of the targeted false negatives without manual prompt tuning or heuristics.
2. **Precision Preservation:** Increased Stutter Precision from $92.31\%$ to **$94.12\%$** with **zero** new false positives.
3. **Recall & Macro F1 Boost:** Raised Stutter Recall from $50.00\%$ to **$66.67\%$** and Macro F1 from $0.4808$ to **$0.5776$**.
4. **Architectural Isolation:** Cleanly gated behind `enable_ngram_repetition`, preserving baseline backwards compatibility.

### Documented Limitations:
- The detector targets contiguous multi-word repetitions; it does not address acoustic prolongations or blocks (which account for the remaining 8 false negatives).
- Validation was performed on the standardized 30-clip development benchmark ($N=30$). Holdout testing ($N=750$) remains frozen and must not be accessed until all planned ablations are evaluated.
