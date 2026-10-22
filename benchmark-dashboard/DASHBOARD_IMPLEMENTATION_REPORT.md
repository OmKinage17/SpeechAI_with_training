# SpeechAI Benchmark Analytics Dashboard — Implementation Report

**Date:** 2026-10-20  
**Application Path:** `E:\SpeechPrj\benchmark-dashboard\`  
**Local URL:** `http://localhost:4173/`  
**Status:** ✅ Fully Implemented, Built, Verified (100% Tests Passed), and Running Live  

---

## 1. Executive Summary

A completely separate, interactive, research-grade analytics dashboard has been constructed in `E:\SpeechPrj\benchmark-dashboard` using **React.js + Vite + Tailwind CSS + Recharts**.

- **Complete Project Isolation:** The original SpeechAI application, backend code, detectors, thresholds, and artifact directories in `E:\SpeechPrj\` were preserved completely untouched without a single algorithm modification.
- **Dedicated Port:** Running on independent port `http://localhost:4173` (verified free before binding, and verified responding with HTTP 200).
- **Exact Data Integrity:** Built upon the verified benchmark artifacts from `evaluation_results/experimental/final_benchmark/run_final_20261020_complete/`. All 120 clips, per-clip predictions, confusion matrices, and audit scorecards were reconciled with 0 synthetic or fabricated metrics.
- **Dynamic Recomputations:** Dataset filters (Combined, UCLASS, FluencyBank, Libri_Stutter) dynamically recompute all metrics directly from per-clip predictions using a unified mathematical engine (`src/utils/metrics.js`).

---

## 2. Completed Pages & Implemented Visualizations

### Page A — Executive Overview (`src/pages/ExecutiveOverview.jsx`)
- **8 KPI Cards:** Total Clips (120), Evaluated Corpora (3), Multiclass Accuracy (59.17%), Macro F1 (55.30%), Binary Stuttering F1 (0.6880), Event-Level Filler F1 (0.6966), ASR WER (23.94%), and Benchmark Stability (115/115 Tests Passing, 0 Timeouts).
- **3×3 Multiclass Confusion Matrix:** Clean ($N=32$), Stutter ($N=67$), Filler ($N=21$) with per-row and per-column totals, precision, and recall.
- **Baseline vs Final Comparison Chart:** Bar chart showing gains across 6 dimensions (+5.84% accuracy, +6.03% macro F1, +3.23% binary stutter F1, 13 $\rightarrow$ 0 timeouts).
- **Per-Dataset Performance Comparison:** Grouped bar chart comparing precision, recall, F1, and specificity across UCLASS, FluencyBank, and Libri_Stutter.
- **7-Parameter Structured Summary:** Cleanly separating supervised classification, regression, ASR, and deterministic rule consistency metrics.
- **Dataset Distribution:** Ground-truth class breakdown across the three evaluation corpora.

### Page B — Parameter 1: Stuttering Detection (`src/pages/Parameter1Stuttering.jsx`)
- **Binary Confusion Matrix:** $\text{TP}=43$, $\text{FP}=15$, $\text{FN}=24$, $\text{TN}=38$ (Precision: $74.14\%$, Recall: $64.18\%$, $F_1: 0.6880$, Specificity: $71.70\%$).
- **Performance Metric Radar/Bar Chart:** Visualizing precision, recall, F1, and specificity.
- **Subtype Detection Counts:** 156 lexical repetitions, 56 acoustic prolongations, 26 acoustic struggle blocks.
- **Gain Attribution Analysis:** Decomposes the $+3.23\%$ binary F1 gain into $+2$ TP from recovered timeouts and $+1$ TP from algorithmic syllable/lexical context filtering.
- **Per-Clip Prediction & Error Table:** Searchable table filterable by False Positives ($N=15$) and False Negatives ($N=24$) with confidence scores and acoustic timestamps.

### Page C — Parameter 2: Filler Detection (`src/pages/Parameter2Fillers.jsx`)
- **Dual-Task Evaluation Banner:** Rigorously distinguishes **Event-Level Filler Detection** ($F_1: 0.6966$, Precision: $55.36\%$, Recall: $93.94\%$, Specificity: $71.26\%$) from **Dominant-Class Multiclass Classification** ($F_1: 0.5424$, Precision: $42.11\%$, Recall: $76.19\%$, Specificity: $77.78\%$).
- **Event-Level Confusion Matrix:** $\text{TP}=31$, $\text{FP}=25$, $\text{FN}=2$, $\text{TN}=62$.
- **Libri_Stutter Negative Control Verification:** Demonstrates $0.0\%$ False Positive Rate across 40 fluent audiobook clips.
- **Per-Dataset Filler Statistics:** UCLASS (Recall $92.9\%$), FluencyBank (Recall $94.7\%$), Libri_Stutter ($0$ false alarms).

### Page D — Parameter 3: Clean-Speech Detection (`src/pages/Parameter3CleanSpeech.jsx`)
- **Clean-Speech Confusion Matrix:** $\text{TP}=12$, $\text{FP}=12$, $\text{FN}=20$, $\text{TN}=76$ (Specificity: $86.36\%$, Precision: $50.00\%$, Recall: $37.50\%$, $F_1: 0.4286$).
- **Per-Dataset Clean Specificity Chart:** UCLASS ($86.2\%$), FluencyBank ($80.6\%$), Libri_Stutter ($92.9\%$).
- **False-Clean vs Missed-Clean Inspector:** Detailed error audit explaining the impact of severe disfluencies overwhelming clean speech thresholds.

### Page E — Parameter 4: Speech-Rate Estimation (`src/pages/Parameter4SpeechRate.jsx`)
- **Scatter Plot with $y = x$ Identity Line:** Evaluated strictly on Libri_Stutter ($N=40$ gold references) comparing estimated WPM against gold reference WPM.
- **Residual Error Plot:** Prediction error ($\Delta \text{WPM}$) vs reference WPM showing mean bias $+11.84$ WPM.
- **Error Distribution Histogram:** Visualizing the residual spread around the zero-error line.
- **Regression Metric Cards:** MAE: $17.84$ WPM, RMSE: $22.15$ WPM, Pearson $r: 0.7547$ ($p < 10^{-7}$), Spearman $\rho: 0.6914$.
- **Short-Window Scientific Disclosure:** Detailed explanation of the $+11.84$ WPM bias introduced by evaluating 3-second slices vs whole-paragraph reading.

### Page F — Parameter 5: Pause and Block Detection (`src/pages/Parameter5PauseBlock.jsx`)
- **Pause Analysis Section:** Continuous acoustic analysis of 200 silence events ($\ge 0.25$s), histogram of pause durations (mean: $0.94$s, median: $0.68$s), and dataset pause frequency table.
- **Clinical Block Analysis Section:** Scoped strictly to UCLASS + FluencyBank ($N=80$ clips with clinical block ground truth):
  - $\text{TP}=0$, $\text{FP}=4$, $\text{FN}=6$, $\text{TN}=70$, Specificity: $94.59\%$, $F_1: 0.0000$.
- **Libri_Stutter Schema Disclosure:** Block metrics explicitly marked `"Not evaluated — ground truth unavailable"` rather than assuming negative labels.
- **Audit Table of the 6 Ground-Truth Block Clips:** Explains why acoustic silence detectors failed to detect tense articulatory blocks (co-occurring sound repetitions and lack of isolated $>0.40$s pre-phonation silence).

### Page G — Parameter 6: ASR Evaluation (`src/pages/Parameter6ASR.jsx`)
- **WER & CER Cards:** Normalized WER: $23.94\%$, CER: $15.35\%$, $299$ word edits across $1,249$ reference words on Libri_Stutter ($N=40$).
- **Stability Metrics:** $0$ timeouts, $0$ failures ($100\%$ transcription completion rate).
- **Per-Clip WER Histogram:** Distribution of word error rates across 5 buckets.
- **Hypothesis vs Reference Transcript Inspector:** Interactive side-by-side comparison of recognized vs gold book sentences.
- **Corpus Exclusions:** UCLASS and FluencyBank are explicitly labeled as unaligned/conversational holdouts lacking gold verbatim transcripts.

### Page H — Parameter 7: Communication Coaching (`src/pages/Parameter7Coaching.jsx`)
- **Deterministic Rule Consistency KPI:** $100.0\%$ ($120/120$ clips evaluated without runtime errors).
- **Average Feedback Density:** $1.74$ advice items per clip ($209$ total items).
- **Rule Trigger Frequency Chart:** Speaking rate ($120$), filler reduction ($45$), repetition easing ($24$), pause rhythm ($20$).
- **Deterministic Logic vs Clinical Efficacy Disclosure:** Prominently highlights that the score measures software logic compliance, not clinical therapy effectiveness.

### Dataset Comparison (`src/pages/DatasetComparison.jsx`)
- **3-Column Comparative Matrix:** Side-by-side comparison of UCLASS vs FluencyBank vs Libri_Stutter across clip count, multiclass accuracy, macro F1, binary stuttering F1, clean specificity, speech rate, blocks, WER, and latency.
- **Grouped Bar Charts:** Multiclass accuracy, macro F1, and stuttering F1 across all three datasets.
- **Individual Dataset Confusion Matrices:** Discrete 3×3 heatmaps for each corpus.

### Per-Clip Explorer (`src/pages/ClipExplorer.jsx`)
- **Search & Multi-Filter Table:** Filter by dataset, true dominant class, predicted dominant class, correctness, and error category (False Alarm, Missed Stutter, etc.).
- **Drawer Detail Modal:** Clickable clip drawer displaying full acoustic details, pause durations, transcript text, gold reference, and coaching feedback.
- **CSV Exporter:** Exports filtered or full clip data with complete ground-truth labels and model predictions.

### About & Methodology (`src/pages/MethodologyLimitations.jsx`)
- **Evaluation Taxonomy:** 4-tier epistemological classification (Supervised Ground Truth, Gold Verbatim References, Signal Processing Proxies, Deterministic Logic Audits).
- **Verified 7-Parameter Reference Table:** Official values from `run_final_20261020_complete`.
- **4 Key Methodological Limitations:** 3.0s window bias, acoustic block detection blindspot, ASR ground truth scoping, and engineering vs clinical validation.

---

## 3. Data Sources Used

All data is drawn strictly from actual benchmark artifacts stored in `evaluation_results/experimental/final_benchmark/run_final_20261020_complete/`:
1. `per_clip_predictions.json` — 120 per-clip model outputs, latency, and classifications.
2. `run_manifest.json` — 120 clip ground truth labels, speaker IDs, and annotation details.
3. `per_parameter_metrics.json` — Official scorecard for Parameters 1 through 7.
4. `confusion_matrices.json` — Combined and per-dataset 3×3 multiclass confusion matrices.
5. `final_metrics.json` — Macro and per-class precision, recall, F1, and support.
6. `per_dataset_metrics.json` — Corpus-specific accuracy, macro F1, and stuttering metrics.

The data mapping layer (`generate_dashboard_dataset.py`) merges and enriches these files into `src/data/benchmark_unified_data.json` without modifying any raw score or metric value.

---

## 4. Unavailable Metrics & Explicit Disclosures

To prevent scientific misrepresentation, the following parameters are explicitly flagged as unavailable on unannotated datasets:
- **Block Detection on Libri_Stutter:** Marked as `"Not evaluated — ground truth unavailable"`. Libri_Stutter's annotation schema does not annotate blocks; acoustic struggle detections are not counted as false positives in the scoped benchmark.
- **ASR WER / CER on UCLASS & FluencyBank:** Marked as `"Not evaluated — verbatim reference transcripts unavailable"`. UCLASS and FluencyBank consist of spontaneous speech without gold reference texts.
- **Speech Rate Ground Truth on UCLASS & FluencyBank:** Only Libri_Stutter has gold text to compute reference WPM. Unreferenced speech rates are identified as acoustic estimates only.

---

## 5. Verification Test Results

An automated research audit test (`tests/verify_dashboard_integrity.js`) was executed via Node.js to verify the data integrity of all 120 clips and 7 project parameters:

```
====================================================
   SPEECHAI DASHBOARD DATA INTEGRITY AUDIT TEST    
====================================================
[PASS] Total clips count is exactly 120 (found: 120)
[PASS] UCLASS clip count is exactly 40 (found: 40)
[PASS] FluencyBank clip count is exactly 40 (found: 40)
[PASS] Libri_Stutter clip count is exactly 40 (found: 40)
[PASS] Ground-truth Clean count is 32 (found: 32)
[PASS] Ground-truth Stutter count is 67 (found: 67)
[PASS] Ground-truth Filler count is 21 (found: 21)
[PASS] Total correct multiclass predictions is 71 (found: 71)
[PASS] Multiclass accuracy is 59.17% (found: 59.17%)
[PASS] Binary Stutter TP is 43 (found: 43)
[PASS] Binary Stutter FP is 15 (found: 15)
[PASS] Binary Stutter FN is 24 (found: 24)
[PASS] Binary Stutter TN is 38 (found: 38)
[PASS] Stutter precision is 74.14% (found: 74.14%)
[PASS] Stutter recall is 64.18% (found: 64.18%)
[PASS] Stutter F1 is 0.6880 (found: 0.6880)
[PASS] Stutter specificity is 71.70% (found: 71.70%)
[PASS] Filler Event TP is 31 (found: 31)
[PASS] Filler Event FP is 25 (found: 25)
[PASS] Filler Event FN is 2 (found: 2)
[PASS] Filler Event TN is 62 (found: 62)
[PASS] Filler Event F1 is 0.6966 (found: 0.6966)
[PASS] Scoped block clips count is exactly 80 (found: 80)
[PASS] Scoped Block TP is 0 (found: 0)
[PASS] Scoped Block FP is 4 (found: 4)
[PASS] Scoped Block FN is 6 (found: 6)
[PASS] Scoped Block TN is 70 (found: 70)
[PASS] Scoped Block Specificity is 94.59% (found: 94.59%)
[PASS] Libri_Stutter block annotations correctly marked unavailable/null (found: 0)
[PASS] Libri_Stutter reference WPM count is exactly 40 (found: 40)
[PASS] Speech Rate MAE is 17.84 WPM (found: 17.85)
[PASS] Speech Rate RMSE is 22.15 WPM (found: 22.15)
[PASS] Coaching engine fired consistently on 120/120 clips (found: 120)
[PASS] Average coaching tips per clip is 1.74 (found: 1.74)
----------------------------------------------------
TEST SUMMARY: 34 / 34 TESTS PASSED (100% PASS RATE)
====================================================
```

---

## 6. Build & Server Status

- **Production Build:** `npm run build` compiled cleanly in $769$ms.
- **Server Execution:** Running as a background process via Vite preview on port `4173`.
- **Verified URL:** `http://localhost:4173/` (HTTP 200 OK verified via PowerShell).
- **Original SpeechAI Project:** 100% untouched and unchanged.
