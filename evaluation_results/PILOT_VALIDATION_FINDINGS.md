# Phase 1: Pilot Benchmark Validation Findings

**Project**: SpeechAI Benchmark Testing  
**Validation Date**: October 17, 2026  
**Audited Artifacts**: `dataset_audit_report.json`, `pilot_manifest.json`, `pilot_predictions.json`, `pilot_results_summary.json`, `pilot_validation_report.json`  

---

## 1. Reconciliation: 60 Planned vs. 54 Executed Clips

A discrepancy existed between the planned 60 clips (20 per dataset) and the 54 clips executed in the pilot run:

| Dataset | Planned Clips | Executed Clips | Difference | Audit Finding |
| :--- | :--- | :--- | :--- | :--- |
| **UCLASS** | 20 | 20 | 0 | Exactly 20 clips executed (Speakers `M_0030`, `M_0061`). |
| **FluencyBank** | 20 | 14 | **-6** | **Root Cause**: The pilot sampling enforced strict episode isolation to `EpId == 010`. In episode 010, exactly 0 sound repetitions met the consensus threshold ($\ge 2$ annotators), and only 4 word repetitions met the threshold. The sampling filter was non-replacing and strictly scoped, resulting in 14 clips (0 sound repetitions, 4 word repetitions, 5 interjections, 5 clean). |
| **Libri_Stutter** | 20 | 20 | 0 | Exactly 20 clips executed (Speaker `103`, Chapter `1240`). |
| **Total** | **60** | **54** | **-6** | Fully reconciled. No audio files crashed or were dropped during inference. |

---

## 2. Metric Recalculation & Zero-Support Handling

In the original summary output, classes with zero ground-truth examples had precision, recall, and F1 reported as `0.000` via automated zero-division overrides. In rigorous academic evaluation, **zero-support classes must be explicitly designated as Undefined / N/A**, rather than interpreted as a detection failure.

### Validated Confusion Matrices and Per-Class Metrics:

#### A. UCLASS (Clinical Speech, N = 20)
* **True Distribution**: Clean = 5, Stutter = 15, Filler = 0 (Support = 0)
* **Confusion Matrix**:
  ```
               Pred Clean   Pred Stutter   Pred Filler   [Support]
  True Clean       2             2              1           [5]
  True Stutter     8             0              7          [15]
  True Filler      0             0              0           [0]
  ```
* **Accuracy**: **10.00%** (2 / 20 correct)
* **Class Performance**:
  * **Clean**: Support = 5 | Precision = **0.2000** | Recall = **0.4000** | F1-Score = **0.2667**
  * **Stutter**: Support = 15 | Precision = **0.0000** | Recall = **0.0000** | F1-Score = **0.0000**
  * **Filler**: Support = 0 | Precision = **0.0000** (8 false positives) | Recall = **Undefined (Support = 0)** | F1-Score = **Undefined (Support = 0)**
* **Diagnosis**: In UCLASS, the 5 selected filler clips co-occurred with sound repetitions and blocks in the clinical ground truth. Because the label hierarchy classified `Stutter > Filler`, ground-truth support for isolated fillers was 0. Furthermore, Whisper frequently normalized clinical stutter blocks into clean text, causing text-level regex to miss stutters.

#### B. FluencyBank (Natural Podcast Speech, N = 14)
* **True Distribution**: Clean = 5, Stutter = 4, Filler = 5
* **Confusion Matrix**:
  ```
               Pred Clean   Pred Stutter   Pred Filler   [Support]
  True Clean       4             0              1           [5]
  True Stutter     3             0              1           [4]
  True Filler      1             0              4           [5]
  ```
* **Accuracy**: **57.14%** (8 / 14 correct)
* **Class Performance**:
  * **Clean**: Support = 5 | Precision = **0.5000** | Recall = **0.8000** | F1-Score = **0.6154**
  * **Stutter**: Support = 4 | Precision = **Undefined (0 Predicted)** | Recall = **0.0000** | F1-Score = **0.0000**
  * **Filler**: Support = 5 | Precision = **0.6667** | Recall = **0.8000** | F1-Score = **0.7273**
* **Diagnosis**: High performance on real-world conversational fillers (F1 = 0.7273). Acoustic DSP energy detection paired with lexical regex successfully caught 4 out of 5 ground-truth fillers.

#### C. Libri_Stutter (Read Speech with Full Transcripts, N = 20)
* **True Distribution**: Clean = 3, Stutter = 17, Filler = 0 (Support = 0)
* **Confusion Matrix**:
  ```
               Pred Clean   Pred Stutter   Pred Filler   [Support]
  True Clean       3             0              0           [3]
  True Stutter     6             7              4          [17]
  True Filler      0             0              0           [0]
  ```
* **Accuracy**: **50.00%** (10 / 20 correct)
* **Mean Word Error Rate (WER)**: **30.44%**
* **Class Performance**:
  * **Clean**: Support = 3 | Precision = **0.3333** | Recall = **1.0000** | F1-Score = **0.5000**
  * **Stutter**: Support = 17 | Precision = **1.0000** | Recall = **0.4118** | F1-Score = **0.5833**
  * **Filler**: Support = 0 | Precision = **0.0000** (4 false positives) | Recall = **Undefined (Support = 0)** | F1-Score = **Undefined (Support = 0)**
* **Diagnosis**: **100% Precision on Stutters** (0 false alarms when predicting stutters). Transcripts enabled exact Levenshtein WER measurement (mean WER = 30.44%).

---

## 3. Speaker Isolation Audit

* **UCLASS**: 2 speakers (`M_0030`, `M_0061`) out of 16 available. Zero overlap with remaining 14 speakers.
* **FluencyBank**: 1 podcast episode (`Ep_010`) out of 8 available episodes. Zero overlap with remaining episodes.
* **Libri_Stutter**: 1 reader (`103`, Chapter `1240`) out of 42 available readers. Zero overlap with remaining 41 readers.

---

## 4. Hardware Latency & Real-Time Factor (RTF) Profile

* **Total Processed Clips**: 54 audio clips
* **Mean Inference Latency**: **4.395 seconds** per 3-second clip
* **Median Latency**: **2.895 seconds**
* **95th-Percentile Latency (p95)**: **14.319 seconds** (due to longer audio segments in LibriStutter)
* **Standard Deviation ($\sigma$)**: **4.162 seconds**
* **Mean Real-Time Factor (RTF)**: **1.265x** on CPU (FP32)
* **Empirical Throughput**: **13.65 clips / minute** (~819 clips / hour)
