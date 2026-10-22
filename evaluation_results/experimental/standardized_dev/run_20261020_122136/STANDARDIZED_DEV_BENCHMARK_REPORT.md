# SpeechAI — Standardized 30-Clip Development Benchmark Report

**Benchmark Execution Date:** 2026-10-20  
**Run ID:** `run_20261020_122136`  
**Dataset:** 30-Clip Canonical Development Set ($N=30$)  
**Feature Cache SHA-256:** `1de05c0980b67e5247b56a9ac5a75e83250c423ded3f28771d805fb486188745`  
**Primary Evaluator:** `backend.experimental.canonical_evaluator.CanonicalEvaluator`  
**Evaluation Scope:** Strict development partition only. Frozen 750-clip holdout dataset remained 100% untouched and uninspected.

---

## 1. Executive Summary

This benchmark establishes the first **standardized, unified, and reproducible comparison** between:
1. **Fix 2B Baseline (Variant 7):** Duration $\ge 0.50\,\text{s}$, acoustic boundary drop corroboration ($\ge 2$ peaks or multi-vowel corroboration).
2. **Fix 3 Syllable Context:** Continuous syllable-rate normalization, relative duration elongation ratios, and phrase-final lengthening dampening.
3. **Fix 4 Lexical/Syntactic Context:** Token confidence integration, word boundary position penalties, function-word dampening, and syntactic context modeling.

To eliminate historical inconsistencies caused by divergent evaluation scripts, disparate silence detection models, or heuristic simulations, all three configurations were evaluated against an **identical, immutable feature cache** extracted in a single pass using real Whisper ASR word timestamps and acoustic feature analysis.

### Standardized Comparison (Identical Feature Cache, N=30)

| Metric | Fix 2B Baseline (Variant 7) | Fix 3 Syllable Context | Fix 4 Lexical/Syntactic Context | Fix 2B $\to$ Fix 3 | Fix 3 $\to$ Fix 4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **3-Class Accuracy** | 53.33% (16/30) | **56.67%** (17/30) | **56.67%** (17/30) | **+3.34 pp** | $\pm 0.00\,\text{pp}$ |
| **Macro Precision** | 0.4524 | **0.4975** | **0.4975** | **+0.0451** | $\pm 0.0000$ |
| **Macro Recall** | 0.7000 | **0.7667** | **0.7667** | **+0.0667** | $\pm 0.0000$ |
| **Macro F1** | 0.4384 | **0.4808** | **0.4808** | **+0.0424** | $\pm 0.0000$ |
| **Weighted F1** | 0.5896 | **0.6216** | **0.6216** | **+0.0320** | $\pm 0.0000$ |
| **Stutter Precision** | 0.8571 (12/14) | **0.9231** (12/13) | **0.9231** (12/13) | **+0.0660** | $\pm 0.0000$ |
| **Stutter Recall** | 50.00% (12/24) | 50.00% (12/24) | 50.00% (12/24) | $\pm 0.00\,\text{pp}$ | $\pm 0.00\,\text{pp}$ |
| **Stutter F1** | 0.6316 | **0.6486** | **0.6486** | **+0.0170** | $\pm 0.0000$ |
| **Stutter Binary (TP / FP / FN / TN)** | 12 / 2 / 12 / 4 | **12 / 1 / 12 / 5** | **12 / 1 / 12 / 5** | -1 FP, +1 TN | $\pm 0$ |
| **FluencyBank Stutter FPs** | 2 | **1** | **1** | **-50.0%** | $\pm 0.00$ |
| **ASR Timeouts** | 0 / 30 | 0 / 30 | 0 / 30 | 0 | 0 |

---

## 2. Shared Feature Extraction Protocol and Cache Provenance

Feature extraction was performed by `backend/experimental/run_dev_benchmark.py` and serialized to `feature_cache.json`:
- **Execution Run ID:** `run_20261020_122136`
- **Feature Cache Path:** `evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json`
- **Cache SHA-256:** `1de05c0980b67e5247b56a9ac5a75e83250c423ded3f28771d805fb486188745`
- **ASR Model:** OpenAI Whisper `base` with `word_timestamps=True`
- **Decoding Configuration:** Bounded temperature schedule `(0.0, 0.2)`
- **Timeout Threshold:** 25.0 seconds per clip (0 / 30 clips timed out)
- **Total Audio Processed:** 220.09 seconds across 30 clips
- **Total Extraction Latency:** 202.12 seconds (mean 6.74s/clip, range 1.57s – 20.38s)

Because all three configurations evaluate predictions strictly from this frozen feature cache, **all detector comparisons are 100% deterministic, variance-free, and directly aligned**.

---

## 3. Silence Detection Mechanics and Word Timestamps

In previous evaluations, inter-word silence intervals were either omitted or approximated, leading to discrepancies in block detection. In this standardized benchmark, silence intervals are derived directly from actual Whisper word timestamp boundaries:
- **Lead-in Silence:** $[0.0, w_{\text{first}}.\text{start}]$ if duration $> 0.25\,\text{s}$.
- **Inter-Word Silence (Gaps):** $[w_{i}.\text{end}, w_{i+1}.\text{start}]$ for all adjacent words where gap $> 0.20\,\text{s}$.
- **Terminal Silence:** $[w_{\text{last}}.\text{end}, T_{\text{audio}}]$ if duration $> 0.35\,\text{s}$.

### Empirical Silence Interval Summary
- **Total Silence Intervals Extracted:** 110 intervals across 30 clips
- **Mean Intervals per Clip:** 3.67 (range 0 to 12)
- **Total Silence Duration:** 48.79 seconds (mean 1.63s/clip, range 0.00s to 8.00s)
- **Clips with 0 Detected Silence:** `FluencyBank_010_4.wav` and `M_0030_16y4m_1_dysfluent_014.wav`
- **Clips with Highest Silence:** `103-1240-0002.flac` (12 intervals, 3.22s), `103-1240-0005.flac` (12 intervals, 2.96s), `103-1240-0007.flac` (5 intervals, 8.00s).

These verified silence intervals enabled legitimate acoustic block detection across Libri_Stutter clips (e.g., $14.16\text{--}14.60\,\text{s}$ in `103-1240-0005.flac`, $4.40\text{--}4.94\,\text{s}$ in `103-1240-0008.flac`), providing consistent evidence across all three pipelines.

---

## 4. 3-Class Confusion Matrices

### Fix 2B Baseline (Variant 7)
Overall Accuracy: 53.33% (16/30)

| Ground Truth \ Predicted | Stutter | Filler | Clean | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Stutter** | 12 | 5 | 7 | 24 |
| **Filler** | 2 | 3 | 0 | 5 |
| **Clean** | 0 | 0 | 1 | 1 |
| **Total Predicted** | 14 | 8 | 8 | 30 |

### Fix 3 Syllable Context
Overall Accuracy: **56.67%** (17/30)

| Ground Truth \ Predicted | Stutter | Filler | Clean | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Stutter** | 12 | 5 | 7 | 24 |
| **Filler** | **1** | **4** | 0 | 5 |
| **Clean** | 0 | 0 | 1 | 1 |
| **Total Predicted** | 13 | 9 | 8 | 30 |

### Fix 4 Lexical/Syntactic Context
Overall Accuracy: **56.67%** (17/30)

| Ground Truth \ Predicted | Stutter | Filler | Clean | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Stutter** | 12 | 5 | 7 | 24 |
| **Filler** | **1** | **4** | 0 | 5 |
| **Clean** | 0 | 0 | 1 | 1 |
| **Total Predicted** | 13 | 9 | 8 | 30 |

---

## 5. Per-Corpus Metrics

### FluencyBank ($N=10$, Ground Truth: 4 Stutter, 5 Filler, 1 Clean)
| Configuration | Accuracy | Macro Prec | Macro Rec | Macro F1 | Weighted F1 | Stutter Prec | Stutter Rec | Stutter F1 | Stutter CM (TP/FP/FN/TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fix 2B Baseline | 50.00% | 0.4778 | 0.6167 | 0.5175 | 0.4810 | 0.3333 | 25.00% | 0.2857 | 1 / 2 / 3 / 4 |
| Fix 3 Context | **60.00%** | **0.5556** | **0.6833** | **0.5758** | **0.5636** | **0.5000** | 25.00% | **0.3333** | **1 / 1 / 3 / 5** |
| Fix 4 Context | **60.00%** | **0.5556** | **0.6833** | **0.5758** | **0.5636** | **0.5000** | 25.00% | **0.3333** | **1 / 1 / 3 / 5** |

### Libri_Stutter ($N=10$, Ground Truth: 10 Stutter, 0 Filler, 0 Clean)
| Configuration | Accuracy | Macro Prec | Macro Rec | Macro F1 | Weighted F1 | Stutter Prec | Stutter Rec | Stutter F1 | Stutter CM (TP/FP/FN/TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fix 2B Baseline | 90.00% | 0.3333 | 0.3000 | 0.3158 | 0.9474 | 1.0000 | 90.00% | 0.9474 | 9 / 0 / 1 / 0 |
| Fix 3 Context | 90.00% | 0.3333 | 0.3000 | 0.3158 | 0.9474 | 1.0000 | 90.00% | 0.9474 | 9 / 0 / 1 / 0 |
| Fix 4 Context | 90.00% | 0.3333 | 0.3000 | 0.3158 | 0.9474 | 1.0000 | 90.00% | 0.9474 | 9 / 0 / 1 / 0 |

### UCLASS ($N=10$, Ground Truth: 10 Stutter, 0 Filler, 0 Clean)
| Configuration | Accuracy | Macro Prec | Macro Rec | Macro F1 | Weighted F1 | Stutter Prec | Stutter Rec | Stutter F1 | Stutter CM (TP/FP/FN/TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fix 2B Baseline | 20.00% | 0.3333 | 0.0667 | 0.1111 | 0.3333 | 1.0000 | 20.00% | 0.3333 | 2 / 0 / 8 / 0 |
| Fix 3 Context | 20.00% | 0.3333 | 0.0667 | 0.1111 | 0.3333 | 1.0000 | 20.00% | 0.3333 | 2 / 0 / 8 / 0 |
| Fix 4 Context | 20.00% | 0.3333 | 0.0667 | 0.1111 | 0.3333 | 1.0000 | 20.00% | 0.3333 | 2 / 0 / 8 / 0 |

---

## 6. Disagreement Analysis Across Configurations

Across all 30 development clips, exactly **1 clip** resulted in differing dominant class predictions:

### Clip: `FluencyBank_010_17.wav`
- **Ground Truth:** `Filler` (`has_stutter=0`, `has_filler=1`)
- **Fix 2B Prediction:** `Stutter` (Confidence 0.60) — **False Positive**
- **Fix 3 Prediction:** `Filler` (Confidence 0.50) — **Correct Prediction (True Negative for Stutter)**
- **Fix 4 Prediction:** `Filler` (Confidence 0.50) — **Correct Prediction (True Negative for Stutter)**
- **Diagnostic Root Cause:**
  - In Fix 2B, two prolonged vowel segments ($0.50\,\text{s}$ and $0.55\,\text{s}$) met acoustic duration and corroboration thresholds without contextual rate adjustment, triggering an uncorroborated prolongation stutter detection.
  - In Fix 3 and Fix 4, speech rate was calibrated to $2.58\,\text{syllables/s}$ (slow speech rate regime), raising the dynamic prolongation threshold above raw vowel duration. Furthermore, the candidate fell adjacent to the hesitation boundary on `"uh,"`, applying phrase-final lengthening dampening. This successfully suppressed the false alarm, shifting the classification from Stutter FP to Filler TP.

### Confidence Nuances (Without Class Flip)
- In `103-1240-0005.flac`: Fix 2B confidence was 0.60; Fix 3 confidence was 0.60; Fix 4 confidence increased to **0.65** due to syntactic function-word context corroborate scores on prolongation candidates.
- In `103-1240-0008.flac`: Fix 2B confidence was 0.60; Fix 3 and Fix 4 confidence increased to **0.70** due to syllable elongation calibration.

---

## 7. ASR Latency and Timeout Accounting

In previous runs, ASR timeout handling was inconsistent: some clips were flagged as timeouts and defaulted to `Clean` or `Stutter` under differing timeout heuristics.

In this standardized benchmark:
- **Total Clips Evaluated:** 30
- **Timeout Rate:** **0% (0 / 30 clips)**
- **Max Latency Observed:** 20.38 seconds for `M_0030_16y4m_1_dysfluent_009.wav` (threshold was 25.0 seconds).
- **Latency Distribution:**
  - $\le 5.0\,\text{s}$: 15 clips (50.0%)
  - $5.0\,\text{s} - 10.0\,\text{s}$: 8 clips (26.7%)
  - $10.0\,\text{s} - 15.0\,\text{s}$: 6 clips (20.0%)
  - $> 15.0\,\text{s}$: 1 clip (3.3%)
- **Impact of Successful Transcription on `M_0030_16y4m_1_dysfluent_009.wav`:**
  - Previously timed out in some unconstrained runs.
  - In this standardized run, complete decode yielded 112 words including dysfluent repeated fragments (`"uh uh"`), enabling lexical repetition detection and classifying it as a valid Stutter True Positive across all configurations.

---

## 8. Methodological Reconciliation with Historical Reports

| Item | Historical Fix 4 Report (`simulate_fix4_dev.py`) | Standardized Benchmark (`run_dev_benchmark.py`) | Methodological Explanation |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 63.33% (19/30) | **56.67% (17/30)** | Historical report used a simulated heuristic script that credited hypothetical candidate recoveries without running the full acoustic extraction pipeline. |
| **Stutter TPs** | 15 / 24 | **12 / 24** | The 3 hypothetical TPs (`FluencyBank_010_79`, `010_97`, `103-1240-0001`) in historical simulation did not produce acoustic prolongation candidates $\ge 0.50\,\text{s}$ on disk; in real extraction, they are legitimate FNs. |
| **Stutter FPs** | 1 (FluencyBank) | **1 (FluencyBank)** | Both reports agree on the single remaining false positive: `FluencyBank_010_40.wav`. |
| **Feature Origin** | Heterogeneous / Simulated | **Unified Feature Cache (`1de05c...`)** | Standardized benchmark runs real Whisper ASR word timestamps, real acoustic boundaries, and real silence detection. |

The standardized pipeline grounds all future development on **empirical reality** rather than simulation artifacts.

---

## 9. Baseline File Immutability Verification

All 12 protected baseline and holdout files were verified via SHA-256 pre-flight and post-flight checksums:
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
Status: **100% Clean — 0 baseline files modified.** The frozen 750-clip holdout dataset was never accessed or inspected.

---

## 10. Next Steps for Fix 5

1. **Target Unrecovered True Positives in UCLASS & FluencyBank:**
   - 8 UCLASS clips (`000`, `001`, `002`, `003`, `011`, `012`, `013`, `014`) remain false negatives because their dysfluent events are sub-0.50s sound repetitions or tense pauses rather than multi-peak sustained prolongations.
   - Investigate syllable-level acoustic repetition detection (e.g., energy envelope repetition matching) rather than relying exclusively on lexical ASR repetitions.
2. **Resolve the Remaining FluencyBank False Positive (`FluencyBank_010_40.wav`):**
   - Candidate segment contains sustained hesitation vocalization that narrowly evaded phrase-final dampening. Contextual vowel formant dispersion analysis could definitively differentiate hesitation lax vowels from tense prolonged syllables.
3. **Preserve Benchmark Integrity:**
   - Continue running `run_dev_benchmark.py` against the canonical development manifest before any consideration of holdout evaluation.
