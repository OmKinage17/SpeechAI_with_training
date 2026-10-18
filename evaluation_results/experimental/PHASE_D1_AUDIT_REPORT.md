# Phase D1 Audit Report: Independent Verification of SpeechAI Experimental Pipeline

## 1. Executive Summary & Audit Scope

This document provides an independent, reproducible audit of the Phase B/C experimental implementation and development evaluation conducted under `E:\SpeechPrj`. 

In accordance with Phase D1 instructions, this audit evaluated:
1. Exact metric reproducibility from saved raw predictions and ground-truth annotations.
2. Immutability of production code, benchmark holdout manifests, and prior audit artifacts.
3. Complete absence of data leakage or partition overlap between the 30 development clips and the 750-clip final holdout benchmark.
4. Concurrency safety, thread lifecycle, and resource isolation of the ASR timeout mechanism.
5. Independent re-execution of the 18 automated unit tests and evaluation of newly constructed edge-case stress tests.
6. Treatment of co-occurring disfluencies, timed-out clips, and latency distributions.

### Audit Summary Dashboard

| Audit Domain | Target / Standard | Audit Finding | Verdict |
|---|---|---|:---:|
| **Production Immutability** | `main.py`, `run_main_benchmark.py`, 750-clip holdout untouched | Hashes match original baseline files; zero git modifications | **PASSED** |
| **Partition Independence** | Zero overlap with 750-clip holdout at clip, speaker, or reader level | 0 clip overlaps, 0 filepath overlaps, 0 speaker/reader overlaps | **PASSED** |
| **Metric Reproducibility** | All reported metrics match recomputed values from raw records | Recomputed confusion matrices and PR-F1 match report exactly | **PASSED** |
| **Automated Unit Tests** | 18 automated tests passing without regression | 18/18 passed in 2.645s | **PASSED** |
| **Edge-Case Stress Tests** | Robustness to silence, clipping, rapid speech, contractions | 7/8 passed; 1 uncovered edge case identified (direct question) | **REVIEW** |
| **Timeout Safety** | Calling thread unblocks; no process crashes | Calling thread unblocks at 25.0s; worker persistence documented | **PASSED WITH ADVISORY** |
| **Co-occurrence Handling** | Binary metrics evaluate disfluencies independently | Stutter and Filler evaluated independently; dominant class uses priority rule | **PASSED** |

---

## 2. Verification of Production & Baseline Immutability

The SHA-256 hashes and file sizes of all production source files, benchmark manifests, baseline predictions, and prior audit artifacts were independently computed and verified:

| File Path | Verified Size (Bytes) | SHA-256 Hash Prefix | Immutability Status |
|---|:---:|:---:|:---:|
| `backend/main.py` | 78,947 | `151b37f86d97b1e2...` | **Unmodified** |
| `backend/run_main_benchmark.py` | 15,507 | `65e4600454bceccc...` | **Unmodified** |
| `evaluation_results/pilot_manifest.json` | 25,212 | `9133a2376c965cea...` | **Unmodified** |
| `evaluation_results/pilot_predictions.json` | 45,057 | `e23df315d6af8a16...` | **Unmodified** |
| `evaluation_results/pilot_results_summary.json` | 2,840 | `894703e87d166392...` | **Unmodified** |
| `evaluation_results/METHODOLOGY_AND_PILOT_REPORT.md` | 7,459 | `e7065e5b5990c324...` | **Unmodified** |
| `evaluation_results/main_benchmark/main_benchmark_manifest.json` | 356,338 | `78ca0ba28b2eb8e3...` | **Unmodified** |
| `evaluation_results/main_benchmark/main_benchmark_predictions.json` | 606,119 | `c188f19f79ac001e...` | **Unmodified** |
| `evaluation_results/main_benchmark/main_benchmark_results.json` | 6,543 | `f0427991f5848bae...` | **Unmodified** |
| `evaluation_results/main_benchmark/MAIN_BENCHMARK_REPORT.md` | 14,604 | `b38e6467a96b315c...` | **Unmodified** |
| `evaluation_results/main_benchmark/audit/recomputed_metrics.json` | 8,615 | `73e7e4bcd27741bd...` | **Unmodified** |
| `evaluation_results/main_benchmark/audit/AUDIT_REPORT.md` | 14,470 | `97e341f2c01a17b0...` | **Unmodified** |

*Verification*: Git status on all production directories confirms zero tracked modifications. All experimental work remains strictly encapsulated under `backend/experimental/`, `backend/tests/`, and `evaluation_results/experimental/`.

---

## 3. Partition Overlap & Data Leakage Audit

A cross-manifest audit script ([`evaluation_results/experimental/audit/audit_partition_overlap.py`](file:///e:/SpeechPrj/evaluation_results/experimental/audit/audit_partition_overlap.py)) evaluated all 30 development clips against the 750 holdout clips in `main_benchmark_manifest.json`:

```text
=== RUNNING PARTITION OVERLAP AUDIT ===
Loaded 30 development clips.
Loaded 750 main benchmark holdout clips.
[UCLASS] Dev entities: ['M_0030']
[UCLASS] Main entities count: 4, Overlap: []
[FluencyBank] Dev entities: ['FB_Ep_010']
[FluencyBank] Main entities count: 2, Overlap: []
[Libri_Stutter] Dev entities: ['103']
[Libri_Stutter] Main entities count: 1, Overlap: []

Audit completed. Status: PASSED.
```

### Partition Disjointness Summary
1. **Clip Filename Overlap**: Exactly **0** overlapping filenames ($0 / 30$).
2. **Filepath Overlap**: Exactly **0** overlapping absolute filepaths ($0 / 30$).
3. **Speaker / Episode / Reader Overlap**:
   - **UCLASS**: Development speaker `M_0030` is disjoint from all 4 holdout speakers in the main benchmark (`M_0034`, `M_0045`, `M_0052`, `M_0060`). Overlap: **0**.
   - **FluencyBank**: Development episode `FB_Ep_010` is disjoint from the 2 holdout episodes (`FB_Ep_008`, `FB_Ep_012`). Overlap: **0**.
   - **Libri_Stutter**: Development reader `103` is disjoint from the holdout reader (`121`). Overlap: **0**.
- **Audit Deliverable**: Saved to [`evaluation_results/experimental/partition_overlap_audit.json`](file:///e:/SpeechPrj/evaluation_results/experimental/partition_overlap_audit.json).

---

## 4. Recomputed Performance Metrics & Verification

All metrics were recomputed directly from the 30 raw clip records in [`evaluation_results/experimental/dev_eval_results_30clips.json`](file:///e:/SpeechPrj/evaluation_results/experimental/dev_eval_results_30clips.json) and matched against the audited baseline predictions in [`evaluation_results/pilot_predictions.json`](file:///e:/SpeechPrj/evaluation_results/pilot_predictions.json).

### 4.1 Side-by-Side Verified Metrics Table

| Metric Category | Metric Name | Baseline Value | Experimental Value | Absolute Change | Relative Change |
|---|---|:---:|:---:|:---:|:---:|
| **3-Class Dominant** | **Accuracy** | 33.33% (10/30) | **60.00%** (18/30) | **+26.67%** | **+80.0%** |
| | Macro Precision | 48.89% | 36.87% | -12.02% | -24.6% |
| | Macro Recall | 66.94% | 30.28% | -36.66% | -54.8% |
| | Macro F1-Score | 0.3344 | 0.3297 | -0.0047 | -1.4% |
| **Binary Stutter** | **Recall** | **20.83%** (5/24) | **70.83%** (17/24) | **+50.00%** | **+240.0%** |
| | Precision | 100.00% (5/5) | 77.27% (17/22) | -22.73% | -22.7% |
| | **F1-Score** | **0.3448** | **0.7391** | **+0.3943** | **+114.4%** |
| | True Positives (TP) | 5 | 17 | +12 | +240.0% |
| | False Negatives (FN) | 19 | 7 | -12 | **-63.2%** |
| | False Positives (FP) | 0 | 5 | +5 | — |
| | True Negatives (TN) | 6 | 1 | -5 | — |
| **Binary Filler** | **Recall** | 85.71% (6/7) | 85.71% (6/7) | 0.00% | Parity |
| | Precision | 50.00% (6/12) | 40.00% (6/15) | -10.00% | -20.0% |
| | F1-Score | 0.6316 | 0.5455 | -0.0861 | -13.6% |
| | True Positives (TP) | 6 | 6 | 0 | Parity |
| | False Negatives (FN) | 1 | 1 | 0 | Parity |
| | False Positives (FP) | 6 | 9 | +3 | — |
| | True Negatives (TN) | 17 | 14 | -3 | — |
| **Latency Statistics** | Median Latency | 2.893s | 8.139s | +5.246s | +181.3% |
| | Mean Latency | 3.972s | 10.734s | +6.762s | +170.2% |
| | Min Latency | 1.882s | 2.376s | +0.494s | +26.2% |
| | Max Latency | 19.146s | 25.015s | +5.869s | Capped by timeout |
| | P95 Latency | 11.696s | 25.004s | — | Capped by timeout |
| | Sum Latency ($\sum L$) | 119.164s | 322.025s | +202.861s | — |
| | Elapsed Wall Clock | N/A | 368.580s | — | — |
| | **Aggregate Micro RTF** | **0.591** | **1.463** | +0.872 | — |
| | **Macro RTF** | **1.274** | **2.905** | +1.631 | — |
| | **Wall-Clock RTF** | N/A | **1.675** | — | — |

- **Audit Deliverable**: Saved to [`evaluation_results/experimental/verified_dev_metrics.json`](file:///e:/SpeechPrj/evaluation_results/experimental/verified_dev_metrics.json).

---

### 4.2 Verified 3-Class Confusion Matrices

#### Baseline 3-Class Confusion Matrix (Ground Truth Rows × Predicted Columns)
```
               Predicted:
Ground Truth   Stutter   Filler   Clean   | Total
-------------------------------------------------
Stutter           5         6       13    |   24
Filler            0         4        1    |    5
Clean             0         0        1    |    1
-------------------------------------------------
Total Predicted   5        10       15    |   30
```

#### Experimental 3-Class Confusion Matrix (Ground Truth Rows × Predicted Columns)
```
               Predicted:
Ground Truth   Stutter   Filler   Clean   | Total
-------------------------------------------------
Stutter          17         2        5    |   24
Filler            4         1        0    |    5
Clean             1         0        0    |    1
-------------------------------------------------
Total Predicted  22         3        5    |   30
```

*Verification Finding*: Marginal sums match ground-truth supports ($24 + 5 + 1 = 30$) and predicted totals ($22 + 3 + 5 = 30$). The experimental pipeline correctly reclassified 12 previously missed disfluent clips as Stutter, achieving a 60.00% dominant accuracy.

---

## 5. Audit of Co-Occurring Disfluencies & Mutually Exclusive Labels

### 5.1 Distinction of Evaluation Units
In speech pathology, a single speech utterance can contain both a sound repetition/block and a vocal hesitation (e.g., *"w-w-what, um, happened?"*). The audit verified that:
1. **Binary Metrics Allow Co-Occurrence**:
   - Stutter presence ($y_{\text{stutter}} \in \{0, 1\}$) and Filler presence ($y_{\text{filler}} \in \{0, 1\}$) are computed as two independent binary tasks.
   - Ground truth support: 24 clips have stutter, 7 clips have filler. Exactly **2 clips** in the development ground truth contain **both** stutter and filler.
   - Experimental predictions: **12 clips** were predicted to contain both stutter and filler events.
2. **3-Class Dominant Classification Enforces Mutual Exclusion**:
   - The multi-class benchmark maps multi-label events into a single dominant label using the priority hierarchy: $\text{Stutter} > \text{Filler} > \text{Clean}$.
   - Consequently, when both disfluencies co-occur, the clip is labeled `Stutter`. This explains why only 1 clip was classified as dominant `Filler` in the experimental 3-class matrix, despite 6 true filler detections in the binary matrix.

---

## 6. Audit of Timed-Out Clips & Processing Statuses

The processing status breakdown across the 30 development clips was verified as follows:

| Processing Status | Clip Count | Percentage | Description |
|---|:---:|:---:|---|
| **Attempted** | 30 | 100.0% | Total clips submitted to the evaluation harness |
| **Successful (Full ASR + DSP)** | 27 | 90.0% | Complete ASR transcript, word timestamps, and acoustic DSP features |
| **Timed-Out (ASR Timeout Capped)** | 3 | 10.0% | Whisper inference exceeded 25.0s and was safely aborted |
| **Failed / Crashed** | 0 | 0.0% | Zero unhandled exceptions, zero process aborts |

### Detailed Audit of the 3 Timed-Out Clips:
All three timeouts occurred on UCLASS clips with severe disfluent blocks and prolongations:
1. `M_0030_16y4m_1_dysfluent_009.wav` (Latency: 25.003s)
2. `M_0030_16y4m_1_dysfluent_011.wav` (Latency: 25.004s)
3. `M_0030_16y4m_1_dysfluent_012.wav` (Latency: 25.015s)

#### Key Architectural Finding on Timeout Handling:
- In the baseline, clips like these caused 30-minute temperature fallback loops. In the experimental pipeline, the 25.0s safety cap prevented runaway execution.
- When ASR timed out, `transcript` returned empty (`""`). However, `eval_harness.py` was designed with **multimodal decoupling**: the `AcousticFeatureExtractor` proceeded to analyze the raw audio waveform directly.
- The acoustic prolongation detector detected sustained vowel pitch stability on all three clips, correctly predicting `has_stutter = 1` and `pred_dominant = "Stutter"`.
- If evaluating strictly on the **27 ASR-successful clips**:
  - Experimental Stutter Recall: $14 / 21 = 66.67\%$
  - Experimental Stutter Precision: $14 / 19 = 73.68\%$ ($F_1 = 0.7000$)
  - Experimental 3-Class Accuracy: $15 / 27 = 55.56\%$

---

## 7. Latency Metrics & RTF Definitions

### 7.1 RTF Metric Definitions
The audit independently verified the mathematical definitions of all reported Real-Time Factor (RTF) metrics:
1. **Aggregate Micro RTF**:
   $$\text{Micro RTF} = \frac{\sum_{i=1}^N \text{latency}_i}{\sum_{i=1}^N \text{duration}_i} = \frac{322.025\text{s}}{220.100\text{s}} = \mathbf{1.463}$$
2. **Macro RTF**:
   $$\text{Macro RTF} = \frac{1}{N} \sum_{i=1}^N \frac{\text{latency}_i}{\text{duration}_i} = \mathbf{2.905}$$
3. **Wall-Clock RTF**:
   $$\text{Wall-Clock RTF} = \frac{\text{Total Elapsed Time}}{\sum_{i=1}^N \text{duration}_i} = \frac{368.580\text{s}}{220.100\text{s}} = \mathbf{1.675}$$

### 7.2 Reconciliation of Wall-Clock vs. Summed Latency
- Summed clip latency: **322.025s**
- Total elapsed wall-clock time: **368.580s**
- Overhead difference: **46.555s** (12.6% of total runtime).
- *Root Cause Analysis*:
  1. **Model Weights Loading**: Initializing Whisper `base` on clip 0 required **10.256s**.
  2. **ThreadPool Queue Lag**: During the 3 timeouts, worker thread execution overlap and queue delay accounted for ~28s.
  3. **File I/O & Metric Aggregation**: Disk reads, resampling, and JSON serialization accounted for ~8.3s.

---

## 8. Test Suite Inspection & Edge-Case Stress Testing

### 8.1 Re-Execution of Original 18 Tests
The original test suite ([`backend/tests/test_experimental_pipeline.py`](file:///e:/SpeechPrj/backend/tests/test_experimental_pipeline.py)) was re-run via `e:\SpeechPrj\backend\venv\Scripts\python.exe`:
- **Result**: **18 passed, 0 failed in 2.645s** (100% passing).

### 8.2 Edge-Case Stress Test Suite (8 New Tests)
A dedicated edge-case test suite was created in [`backend/tests/test_experimental_edge_cases.py`](file:///e:/SpeechPrj/backend/tests/test_experimental_edge_cases.py):

| Test Case | Scenario Tested | Outcome | Details |
|---|---|:---:|---|
| `test_pure_digital_silence_no_crashes` | 2 seconds of pure zeros | **PASS** | No division by zero or NaN in noise floor or spectral flux |
| `test_clipped_saturated_audio` | Square wave at maximum amplitude | **PASS** | Bounded RMS and prolongation candidate handling |
| `test_contractions_with_discourse_words` | *"So's your brother"*, *"well's dry"* | **PASS** | Correctly rejected as non-fillers |
| `test_fillers_with_heavy_punctuation_and_caps` | *"UM... UH???"* | **PASS** | Detected core vocal hesitations |
| `test_contiguous_speech_no_gaps` | Contiguous words (0ms pause) | **PASS** | No array out-of-bounds; 0 block false alarms |
| `test_inverted_or_invalid_timestamps_handled_safely` | Corrupted word timestamps ($start > end$) | **PASS** | Safely handled without crashes |
| `test_short_sub_hundred_millisecond_audio` | 50ms audio snippet | **PASS** | Safely returned `too_short` |
| `test_ambiguous_like_at_utterance_end` | *"Do you know what it's like?"* | **FAIL** | Triggered by `"you know"`, not `"like"` |

### 8.3 Analysis of the Failed Edge-Case Check
- In *"Do you know what it's like?"*, the word `"like"` was correctly rejected as a non-filler.
- However, the phrase `"you know"` was matched because the sentence contained the interrogative question clause *"Do you know"*.
- *Diagnosis*: In English, when preceded by auxiliary verbs (`"do"`, `"did"`, `"does"`, `"don't"`, `"didn't"`), *"you know"* is the main verb phrase of a question, not a discourse filler.
- *Recommendation*: Add an auxiliary verb exclusion pattern (`\b(do|did|does|don't|dont|didn't|didnt)\s+you\s+know\b`) to `filler_detector.py` in Phase E.

---

## 9. Synthesis: Facts, Limitations, and Recommendations

### 9.1 Verified Facts
1. Production code (`main.py`, `run_main_benchmark.py`), baseline manifests, and audit reports are 100% pristine and unmodified.
2. The 30 development clips have **zero overlap** with the 750-clip final benchmark holdout at clip, speaker, episode, or reader levels.
3. Multimodal acoustic-lexical fusion increases stutter recall by **+50.00% absolute** (tripling recall from 20.83% to 70.83%) and reduces missed stutter false negatives by **63.2%**.
4. The 25.0s safety timeout prevents runaway temperature fallback loops and bounds P95 latency to 25.004s.

### 9.2 Limitations Discovered
1. **Development Set Sample Size ($N=30$)**: Designed strictly for debugging; class skew (24 stutter, 7 filler, 1 clean) limits statistical generalizability claims.
2. **Worker Thread Persistence in ThreadPoolExecutor**: When a timeout triggers, Python unblocks the caller, but the background worker thread completes its current forward pass, creating queue delay for subsequent clips when `max_workers=1`.
3. **Question Clause Ambiguity**: Direct questions starting with *"Do you know..."* can match phrase filler patterns unless explicitly gated by auxiliary verb rules.

### 9.3 Recommendations for Next Phase
1. **Approval to Proceed with Phase E (Full 750-Clip Holdout Benchmark)**:
   - With verified partition disjointness and proven safety caps, the pipeline is ready to run on the full 750-clip holdout.
2. **Upgrade to Process-Level Isolation (`multiprocessing.Process`)**:
   - Replace `ThreadPoolExecutor` with a dedicated child process so that timeouts can invoke `process.terminate()`, instantly killing runaway decodes without work queue backlog.
3. **Refine Interrogative Exclusions**:
   - Add `"do you know"` to the grammatical non-filler filter to resolve the edge-case finding.
