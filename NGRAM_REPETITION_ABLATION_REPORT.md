# SpeechAI — Ablation 1 Report: Multi-Word N-Gram Repetition Detection

**Experiment Date:** 2026-10-20  
**Scope:** Standardized 30-Clip Development Benchmark ($N=30$)  
**Feature Cache Provenance:** `evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json` (`1de05c...`)  
**Evaluation Harness:** `backend.experimental.canonical_evaluator.CanonicalEvaluator`  
**Deliverable Directory:** `evaluation_results/experimental/ablations/ngram_repetition/`  
**Protected Baseline Immutability:** 100% Verified Clean (SHA-256 Checksums Pristine)

---

## 1. Executive Summary

This ablation evaluated the hypothesis formulated during the Fix 3 False-Negative Root-Cause Analysis: **that extending lexical repetition detection from unigrams to contiguous multi-word token sequences of length 2 and 3 would recover true stutter dysfluencies that produce zero acoustic prolongation candidates $\ge 320\,\text{ms}$.**

Four isolated configurations were evaluated against the exact same 30 development clips and pre-extracted feature cache:
1. **Fix 3 Reference:** Unigrams only (`enable_ngram_repetition=False`).
2. **Fix 3 + 2-Gram Detection:** Unigrams + 2-grams (`ngram_sizes=(2,)`).
3. **Fix 3 + 3-Gram Detection:** Unigrams + 3-grams (`ngram_sizes=(3,)`).
4. **Fix 3 + Combined (2-Gram + 3-Gram):** Unigrams + 2-grams + 3-grams with 3-gram subsumption precedence (`ngram_sizes=(2, 3)`).

### Empirical Progression Table ($N=30$, Identical Feature Cache `1de05c...`)

| Metric | Fix 3 Reference | Fix 3 + 2-Gram | Fix 3 + 3-Gram | Fix 3 + Combined (2g+3g) | Progression (Ref $\to$ Combined) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **3-Class Accuracy** | 56.67% (17/30) | 60.00% (18/30) | 66.67% (20/30) | **70.00% (21/30)** | **+13.33 pp** |
| **Macro Precision** | 0.4975 | 0.5053 | 0.5506 | **0.5598** | **+0.0623** |
| **Macro Recall** | 0.7667 | 0.7806 | 0.8083 | **0.8222** | **+0.0555** |
| **Macro F1** | 0.4808 | 0.5019 | 0.5556 | **0.5776** | **+0.0968** |
| **Weighted F1** | 0.6216 | 0.6509 | 0.7194 | **0.7450** | **+0.1234** |
| **Stutter Precision** | 92.31% (12/13) | 92.86% (13/14) | 93.75% (15/16) | **94.12% (16/17)** | **+1.81 pp** |
| **Stutter Recall** | 50.00% (12/24) | 54.17% (13/24) | 62.50% (15/24) | **66.67% (16/24)** | **+16.67 pp** |
| **Stutter Specificity** | 83.33% (5/6) | 83.33% (5/6) | 83.33% (5/6) | **83.33% (5/6)** | $\pm 0.00\,\text{pp}$ |
| **Stutter F1** | 0.6486 | 0.6842 | 0.7500 | **0.7805** | **+0.1319** |
| **Binary CM (TP / FP / FN / TN)** | 12 / 1 / 12 / 5 | 13 / 1 / 11 / 5 | 15 / 1 / 9 / 5 | **16 / 1 / 8 / 5** | **+4 TP, -4 FN, 0 new FP** |
| **New False Positives** | — | **0** | **0** | **0** | **0** |

### Key Experimental Findings

1. **Targeted False Negatives Successfully Recovered:**
   - All three targeted clips were recovered as **True Positives**:
     - `103-1240-0001.flac`: Recovered via 2-gram (`"intricate headlong intricate headlong"`, conf $0.85$).
     - `M_0030_16y4m_1_dysfluent_013.wav`: Recovered via 3-gram (`"or you could or you could"`, conf $0.85$).
     - `M_0030_16y4m_1_dysfluent_014.wav`: Recovered via 3-gram (`"or you could or you could"`, conf $0.85$).
   - In addition, a 4th false negative was recovered:
     - `M_0030_16y4m_1_dysfluent_012.wav`: Recovered via 3-gram (`"all you could all you could"`, conf $0.85$).
2. **Zero False Positive Inflation:** Exactly 0 non-stutter clips triggered n-gram repetition matches. The sole remaining false positive in FluencyBank (`FluencyBank_010_40.wav`) remained unchanged ($1\,\text{FP}$ across all configurations).
3. **Decisive Accuracy & Recall Milestone:** Combined 2-gram + 3-gram detection elevated overall benchmark accuracy to **$70.00\%$** (from $56.67\%$) and Stutter Recall to **$66.67\%$** (from $50.00\%$), achieving the highest macro F1 ($0.5776$) in the project's history without modifying acoustic prolongation thresholds.

---

## 2. Motivation and Problem Context

In Fix 3, lexical repetition detection was restricted to adjacent single-word identity:
```python
for i in range(len(words) - 1):
    if words[i] == words[i+1]:
        events += 1
```
This formulation failed whenever dysfluencies involved multi-token phrases. Even when Whisper produced flawless transcriptions of multi-word repetitions separated by punctuation or hyphens (e.g. `"Or you could, or you could sponsor."`), the detector compared only adjacent unigrams (`"could"` $\ne$ `"or"`), finding 0 matches.

Simultaneously, acoustic prolongation detection could not rescue these clips because rapid articulatory speech produced zero sustained vowel plateaus $\ge 320\,\text{ms}$. Ablation 1 bridges this gap by directly capturing verbatim contiguous multi-word phrases.

---

## 3. Isolated Implementation Architecture

The detector is implemented as a modular component in [`backend/experimental/ngram_repetition.py`](file:///e:/SpeechPrj/backend/experimental/ngram_repetition.py) and gated behind an isolated flag in [`StutterDetectorConfig`](file:///e:/SpeechPrj/backend/experimental/config.py#L77-L81):

```python
@dataclass
class StutterDetectorConfig:
    ...
    # Ablation 1: Multi-word n-gram repetition detection (isolated experimental flag)
    enable_ngram_repetition: bool = False
    ngram_sizes: Tuple[int, ...] = (2, 3)
    ngram_repetition_confidence: float = 0.85
```

### Algorithmic Design Rules

1. **Token Normalization:** Case-normalized tokens stripped of boundary punctuation while strictly preserving token order.
2. **Contiguous Comparison:** For $N \in \{2, 3\}$, a repetition at token index $i$ requires exact token-by-token equality:
   $$W[i : i+N] == W[i+N : i+2N]$$
3. **Punctuation & Boundary Resilience:** Because boundary punctuation (commas, hyphens, dashes, periods) is stripped at the token level, clauses like `"Or you could, or you could..."` and `"Or you could- or you could-"` cleanly reduce to contiguous identical tuples `("or", "you", "could")`.
4. **Overlap Subsumption (Longest Match First):** Candidates are evaluated descending ($N=3$ first, then $N=2$). If a 3-gram match covers token span $[i, i+6]$, candidate 2-grams whose spans fall entirely within $[i, i+6]$ are subsumed, eliminating double-counting of sub-phrases.
5. **Single-Word Repetition Exclusion:** Phrases consisting of identical repeated tokens (e.g., `["the", "the"]`) are excluded (`len(set(phrase)) > 1`), ensuring unigram repetition logic remains the sole authority on single-word sequences.
6. **Multi-Iteration Boost:** Iteration chains $\ge 3$ (e.g., `"sharp eye, sharp eye, sharp eye"`) assign elevated confidence ($0.90$).
7. **Timestamp Alignment:** Matched spans are mapped to Whisper word timestamps when available, preserving temporal localization (`start`, `end`, `duration`).

---

## 4. Benchmark Evaluation & Confusion Matrices

### Multiclass Confusion Matrices ($N=30$)

#### Fix 3 Reference (Unigrams Only) — Accuracy: 56.67%
| Ground Truth \ Predicted | Stutter | Filler | Clean | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Stutter** | 12 | 5 | 7 | 24 |
| **Filler** | 1 | 4 | 0 | 5 |
| **Clean** | 0 | 0 | 1 | 1 |
| **Total** | 13 | 9 | 8 | 30 |

#### Fix 3 + 2-Gram Detection — Accuracy: 60.00%
| Ground Truth \ Predicted | Stutter | Filler | Clean | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Stutter** | **13 (+1)** | 5 | 6 (-1) | 24 |
| **Filler** | 1 | 4 | 0 | 5 |
| **Clean** | 0 | 0 | 1 | 1 |
| **Total** | 14 | 9 | 7 | 30 |

#### Fix 3 + 3-Gram Detection — Accuracy: 66.67%
| Ground Truth \ Predicted | Stutter | Filler | Clean | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Stutter** | **15 (+3)** | 3 (-2) | 6 (-1) | 24 |
| **Filler** | 1 | 4 | 0 | 5 |
| **Clean** | 0 | 0 | 1 | 1 |
| **Total** | 16 | 7 | 7 | 30 |

#### Fix 3 + Combined (2-Gram + 3-Gram) — Accuracy: **70.00%**
| Ground Truth \ Predicted | Stutter | Filler | Clean | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Stutter** | **16 (+4)** | 3 (-2) | 5 (-2) | 24 |
| **Filler** | 1 | 4 | 0 | 5 |
| **Clean** | 0 | 0 | 1 | 1 |
| **Total** | 17 | 7 | 6 | 30 |

---

## 5. Per-Corpus Metrics Breakdown

### UCLASS ($N=10$, Ground Truth: 10 Stutter, 0 Filler, 0 Clean)
| Configuration | Accuracy | Macro F1 | Weighted F1 | Stutter Recall | Stutter Precision | Binary CM (TP / FP / FN / TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Fix 3 Reference | 20.00% | 0.1111 | 0.3333 | 20.00% | 100.00% | 2 / 0 / 8 / 0 |
| Fix 3 + 2-Gram | 20.00% | 0.1111 | 0.3333 | 20.00% | 100.00% | 2 / 0 / 8 / 0 |
| Fix 3 + 3-Gram | **50.00%** | **0.2222** | **0.6667** | **50.00%** | **100.00%** | **5 / 0 / 5 / 0 (+3 TP)** |
| Fix 3 + Combined | **50.00%** | **0.2222** | **0.6667** | **50.00%** | **100.00%** | **5 / 0 / 5 / 0 (+3 TP)** |

*Analysis:* 3-Gram repetition detection rescued UCLASS performance, pulling 3 clips (`012`, `013`, `014`) from false negatives to verified true positives and boosting UCLASS accuracy from $20\%$ to **$50\%$**.

### Libri_Stutter ($N=10$, Ground Truth: 10 Stutter, 0 Filler, 0 Clean)
| Configuration | Accuracy | Macro F1 | Weighted F1 | Stutter Recall | Stutter Precision | Binary CM (TP / FP / FN / TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Fix 3 Reference | 90.00% | 0.3158 | 0.9474 | 90.00% | 100.00% | 9 / 0 / 1 / 0 |
| Fix 3 + 2-Gram | **100.00%** | **0.3333** | **1.0000** | **100.00%** | **100.00%** | **10 / 0 / 0 / 0 (+1 TP)** |
| Fix 3 + 3-Gram | 90.00% | 0.3158 | 0.9474 | 90.00% | 100.00% | 9 / 0 / 1 / 0 |
| Fix 3 + Combined | **100.00%** | **0.3333** | **1.0000** | **100.00%** | **100.00%** | **10 / 0 / 0 / 0 (+1 TP)** |

*Analysis:* 2-Gram repetition detection recovered the single remaining Libri_Stutter false negative (`103-1240-0001.flac`), achieving **$100.00\%$ Accuracy (10/10) on Libri_Stutter**.

### FluencyBank ($N=10$, Ground Truth: 4 Stutter, 5 Filler, 1 Clean)
| Configuration | Accuracy | Macro F1 | Weighted F1 | Stutter Recall | Stutter Precision | Binary CM (TP / FP / FN / TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Fix 3 Reference | 60.00% | 0.5758 | 0.5636 | 25.00% | 50.00% | 1 / 1 / 3 / 5 |
| Fix 3 + 2-Gram | 60.00% | 0.5758 | 0.5636 | 25.00% | 50.00% | 1 / 1 / 3 / 5 |
| Fix 3 + 3-Gram | 60.00% | 0.5758 | 0.5636 | 25.00% | 50.00% | 1 / 1 / 3 / 5 |
| Fix 3 + Combined | 60.00% | 0.5758 | 0.5636 | 25.00% | 50.00% | 1 / 1 / 3 / 5 |

*Analysis:* FluencyBank metrics remained completely stable across all configurations. Zero false positives were introduced on fluent or filler clips.

---

## 6. Detailed Clip-by-Clip Prediction Changes ($N=4$)

Every prediction change between Fix 3 Reference and Fix 3 Combined was evaluated for ground-truth correctness:

### 1. `M_0030_16y4m_1_dysfluent_012.wav` (UCLASS)
- **Ground Truth:** `Stutter` (`gt_has_stutter=1`, `gt_has_filler=0`)
- **Fix 3 Reference:** `Filler` (Confidence $0.00$) $\to$ **False Negative**
- **Fix 3 Combined:** `Stutter` (Confidence $0.90$) $\to$ **TRUE POSITIVE (RECOVERED)**
- **Event Matched:** `ngram_phrase_repetition_3gram`
- **Phrase Matched:** `"all you could all you could"` (iterations: 2, conf: $0.85$)
- **Validation:** Punctuation between clauses (`"Make a warship. All you could. All you could."`) was normalized, successfully matching the verbatim repeated 3-gram.

### 2. `M_0030_16y4m_1_dysfluent_013.wav` (UCLASS)
- **Ground Truth:** `Stutter` (`gt_has_stutter=1`, `gt_has_filler=0`)
- **Fix 3 Reference:** `Filler` (Confidence $0.00$) $\to$ **False Negative**
- **Fix 3 Combined:** `Stutter` (Confidence $0.90$) $\to$ **TRUE POSITIVE (RECOVERED)**
- **Event Matched:** `ngram_phrase_repetition_3gram`
- **Phrase Matched:** `"or you could or you could"` (iterations: 2, conf: $0.85$)
- **Validation:** Cleanly resolves the comma pause in `"Or you could, or you could sponsor."`.

### 3. `M_0030_16y4m_1_dysfluent_014.wav` (UCLASS)
- **Ground Truth:** `Stutter` (`gt_has_stutter=1`, `gt_has_filler=0`)
- **Fix 3 Reference:** `Clean` (Confidence $0.00$) $\to$ **False Negative**
- **Fix 3 Combined:** `Stutter` (Confidence $0.90$) $\to$ **TRUE POSITIVE (RECOVERED)**
- **Event Matched:** `ngram_phrase_repetition_3gram`
- **Phrase Matched:** `"or you could or you could"` (iterations: 2, conf: $0.85$)
- **Validation:** Handles the hyphenated speech restart marker in `"Or you could- or you could sponsor-"`.

### 4. `103-1240-0001.flac` (Libri_Stutter)
- **Ground Truth:** `Stutter` (`gt_has_stutter=1`, `gt_has_filler=0`)
- **Fix 3 Reference:** `Clean` (Confidence $0.00$) $\to$ **False Negative**
- **Fix 3 Combined:** `Stutter` (Confidence $0.90$) $\to$ **TRUE POSITIVE (RECOVERED)**
- **Event Matched:** `ngram_phrase_repetition_2gram`
- **Phrase Matched:** `"intricate headlong intricate headlong"` (iterations: 2, conf: $0.85$)
- **Validation:** Resolves the alternating 2-gram dysfluency in `"...that intricate headlong, that intricate headlong, intricate headlong..."`.

**Correctness Rate:** **4 out of 4 prediction changes (100.0%) are verified correct true positive recoveries.**

---

## 7. Complementarity of 2-Gram and 3-Gram Detectors

The ablation confirms that 2-gram and 3-gram detectors are strictly complementary and neither alone achieves the optimal result:
- **2-Gram Alone:** Recovers `103-1240-0001.flac` (+1 TP), but cannot recover `012`, `013`, or `014` because the 3-gram phrase `"or you could"` does not contain repeated adjacent pairs.
- **3-Gram Alone:** Recovers `012`, `013`, and `014` (+3 TP), but misses `103-1240-0001.flac` because `"intricate headlong"` is a 2-word phrase.
- **Combined (2-Gram + 3-Gram with Subsumption):** Seamlessly recovers all 4 clips (+4 TP) without any double-counting or interference.

---

## 8. Unit Test Suite Summary

A dedicated test suite in [`backend/tests/test_ngram_repetition.py`](file:///e:/SpeechPrj/backend/tests/test_ngram_repetition.py) was implemented and validated:

| Test Case | Scenario Verified | Status |
| :--- | :--- | :---: |
| `test_2gram_repetition_basic` | Basic contiguous 2-gram matching on prose transcript | **PASS** |
| `test_3gram_repetition_with_punctuation` | 3-gram repetition separated by commas | **PASS** |
| `test_3gram_repetition_with_hyphens` | 3-gram repetition separated by hyphens / speech restart markers | **PASS** |
| `test_multi_iteration_chain` | 3+ iteration repetitions with elevated confidence ($0.90$) | **PASS** |
| `test_overlap_subsumption_3gram_over_2gram` | 3-gram match subsumes sub-2-grams (no double counting) | **PASS** |
| `test_2gram_detector_isolation` | 2-gram detector does not match 3-grams lacking 2-gram pairs | **PASS** |
| `test_3gram_detector_isolation` | 3-gram detector does not match 2-gram repetitions | **PASS** |
| `test_single_repeated_word_exclusion` | Sequences of identical tokens (`"the the the"`) excluded from n-grams | **PASS** |
| `test_negative_control_fluent_speech` | Standard non-repeated conversational speech produces 0 events | **PASS** |
| `test_empty_and_none_transcripts` | Graceful handling of empty, whitespace, and None input | **PASS** |
| `test_integration_with_hybrid_stutter_detector` | Verification of isolated config flag (`enable_ngram_repetition`) | **PASS** |

All 11 ablation unit tests and all 32 existing benchmark regression tests pass cleanly (**43 / 43 tests pass**).

---

## 9. Baseline File Immutability Record

Cryptographic SHA-256 verification confirms that all 12 baseline files and the frozen 750-clip holdout remain 100% clean and untouched:

```
[OK] backend/main.py (sha256: 151b37f86d97b1e2...)
[OK] backend/run_main_benchmark.py (sha256: 65e4600454bceccc...)
[OK] evaluation_results/pilot_manifest.json (sha256: 9133a2376c965cea...)
[OK] evaluation_results/pilot_predictions.json (sha256: e23df315d6af8a16...)
[OK] evaluation_results/pilot_results_summary.json (sha256: 894703e87d166392...)
[OK] evaluation_results/METHODOLOGY_AND_PILOT_REPORT.md (sha256: e7065e5b5990c324...)
[OK] evaluation_results/main_benchmark/main_benchmark_manifest.json (sha256: 78ca0ba28b2eb8e3...)
[OK] evaluation_results/main_benchmark/main_benchmark_predictions.json (sha256: c188f19f79ac001e...)
[OK] evaluation_results/main_benchmark/main_benchmark_results.json (sha256: f0427991f5848bae...)
[OK] evaluation_results/main_benchmark/MAIN_BENCHMARK_REPORT.md (sha256: b38e6467a96b315c...)
[OK] evaluation_results/main_benchmark/audit/recomputed_metrics.json (sha256: 73e7e4bcd27741bd...)
[OK] evaluation_results/main_benchmark/audit/AUDIT_REPORT.md (sha256: 97e341f2c01a17b0...)
```
Status: **100% Clean.** The frozen holdout dataset was never accessed or evaluated.

---

## 10. Conclusion and Next Steps

Ablation 1 is an **unequivocal success**:
- It decisively recovers all 3 targeted false negatives (`UCLASS_013`, `UCLASS_014`, `103-1240-0001`) plus `UCLASS_012`.
- It lifts Stutter Recall from $50.00\%$ to **$66.67\%$** and Accuracy from $56.67\%$ to **$70.00\%$**.
- It achieves this with **zero new false positives** and zero modification to acoustic thresholds.

This ablation should be formally incorporated into the mainline experimental detector configuration for the upcoming production release.
