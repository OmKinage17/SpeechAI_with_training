# SpeechAI Phase E: 750-Clip Experimental Benchmark Report

**Execution Timestamp**: 2026-10-18T18:51:26.215030 to 2026-10-18T19:51:54.428246
**Total Elapsed Wall-Clock Time**: 3628.21 seconds (60.47 minutes)
**Manifest Evaluated**: `E:\SpeechPrj\evaluation_results\main_benchmark\main_benchmark_manifest.json`

---

## 1. Executive Summary & Execution Status

| Metric | Experimental Count / Value |
|---|---:|
| Total Clips Attempted | 750 |
| Successful Inferences | 706 |
| Timed-Out Clips (Terminated & Reaped) | 44 |
| Execution Errors | 0 |
| Worker Crashes | 0 |
| Process Isolation Status | Enabled (25.0s per-clip hard timeout via OS process termination) |

---

## 2. Core Detection Comparison: Baseline vs. Experimental

| Classification Task / Metric | Audited Baseline | Experimental Pipeline | Absolute Delta |
|---|---:|---:|---:|
| **Dominant 3-Class Accuracy** | 50.13% | **53.73%** | +3.60% |
| **Dominant 3-Class Macro F1** | 0.4748 | **0.4000** | -0.0748 |
| **Dominant 3-Class Weighted F1** | 0.4948 | **0.4925** | -0.0023 |
| Binary Stutter Precision | 84.62% | 56.14% | -28.48% |
| **Binary Stutter Recall** | 29.10% | **84.66%** | +55.56% |
| **Binary Stutter F1-Score** | 0.4331 | **0.6751** | +0.2420 |
| **Binary Filler Precision** | 41.36% | **26.91%** | -14.45% |
| Binary Filler Recall | 78.21% | 69.87% | -8.34% |
| **Binary Filler F1-Score** | 0.5410 | **0.3886** | -0.1524 |

---

## 3. Dominant 3-Class Classification Details

### Confusion Matrix (Rows = Ground Truth, Columns = Predicted)

| Ground Truth \ Predicted | Pred Clean | Pred Stutter | Pred Filler | Total Support |
|---|---:|---:|---:|---:|
| **True Clean** | **70** | 198 | 24 | 292 |
| **True Stutter** | 34 | **320** | 24 | 378 |
| **True Filler** | 15 | 52 | **13** | 80 |

### Per-Class Detailed Performance

| Class | Precision | Recall | F1-Score | Support | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Clean** | 58.82% | 23.97% | 0.3406 | 292 | 70 | 49 | 222 |
| **Stutter** | 56.14% | 84.66% | 0.6751 | 378 | 320 | 250 | 58 |
| **Filler** | 21.31% | 16.25% | 0.1844 | 80 | 13 | 48 | 67 |

---

## 4. Binary Disfluency Presence Detection

Binary detection evaluates whether a specific disfluency type is present in the clip independently of other disfluency types.

| Disfluency Type | TP | FP | FN | TN | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Stuttering Presence** | 320 | 250 | 58 | 122 | 56.14% | 84.66% | 0.6751 | 378 |
| **Filler Word Presence** | 109 | 296 | 47 | 298 | 26.91% | 69.87% | 0.3886 | 156 |

---

## 5. Per-Corpus Breakdown

| Corpus | 3-Class Accuracy | Stutter Prec | Stutter Rec | Stutter F1 | Filler Prec | Filler Rec | Filler F1 | Clips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **UCLASS** | 59.20% | 68.02% | 76.97% | 0.7222 | 48.09% | 75.00% | 0.5860 | 250 |
| **FluencyBank** | 35.20% | 25.16% | 67.24% | 0.3662 | 44.66% | 63.89% | 0.5257 | 250 |
| **Libri_Stutter** | 66.80% | 67.49% | 97.62% | 0.7981 | 0.00% | N/A (Supp=0) | N/A (Supp=0) | 250 |

---

## 6. Automatic Speech Recognition (ASR) Word Error Rate

- **Evaluated Clips**: 248 Libri_Stutter clips with verified reference text.
- **Experimental Macro WER**: 0.4419 (Baseline: 0.4505)
- **Experimental Pooled Micro WER**: 0.4298 (Baseline: 0.4364)

> **Definitions**:
> - **Macro WER**: Mean of individual clip Word Error Rates across all evaluated clips.
> - **Pooled Micro WER**: $\frac{\sum (\text{Substitutions} + \text{Deletions} + \text{Insertions})}{\sum (\text{Hits} + \text{Substitutions} + \text{Deletions})}$, calculating global edit distance across the entire corpus.

---

## 7. Latency, Real-Time Factor (RTF) & Throughput

| Metric | Audited Baseline | Experimental Pipeline | Improvement / Notes |
|---|---:|---:|---|
| **Median Latency** | 2.521s | 6.173s | Per-clip typical response time |
| **Mean Latency** | 12.074s | 13.434s | Bounded execution |
| **P95 Latency** | 16.211s | 25.064s | High-percentile predictability |
| **Maximum Latency** | 1917.945s | **2658.687s** | Runaway inference eliminated |
| **Aggregate Micro RTF** | 1.806 | 2.010 | Total Processing Time / Audio Time |
| **Macro RTF** | 3.303 | 4.004 | Mean of per-clip RTFs |
| **Total Wall-Clock Time** | 150.92 min (9055s) | 60.47 min (3628.21s) | Full benchmark duration |

---

## 8. Dominant-Class Decision Rule & Co-Occurrence Explanation

In natural continuous speech, disfluencies frequently co-occur within the same audio segment (e.g., a speaker may produce both a prolongation and an interjection like 'um').

1. **Clip-Level Binary Detections**: The experimental pipeline evaluates stuttering presence and filler word presence independently. A clip can have both `pred_has_stutter = 1` and `pred_has_filler = 1`.
2. **Mutually Exclusive Dominant-Class Rule**: To assign a single categorical label per clip, the system applies the hierarchical clinical rule:
   - If `pred_has_stutter == 1` $\rightarrow$ Class is **Stutter** (stuttering events take clinical precedence).
   - Else if `pred_has_filler == 1` $\rightarrow$ Class is **Filler**.
   - Else $\rightarrow$ Class is **Clean**.

---

## 9. Limitations & Scope of Evaluation

- **Screening & Coaching Scope**: SpeechAI is designed as an asynchronous communication-support and speech fluency screening tool. It is **not** a clinically validated diagnostic instrument.
- **Clip-Level Binary vs. Event-Level Boundaries**: Evaluations are performed at the 3-second audio clip level. The system detects presence and frequency within clips, but does not provide millisecond-accurate acoustic boundary alignments.
- **Dataset Heterogeneity**: Libri_Stutter contains synthetic disfluency insertions that differ acoustically from natural disfluencies found in clinical datasets like UCLASS and FluencyBank.
- **Hardware Constraints**: Benchmarking was executed on CPU; GPU inference will yield significantly lower latencies and Real-Time Factors.

---

## 10. Pipeline Configuration Parameters
```json
{
  "asr_timeout_sec": 25.0,
  "asr_temperature": [
    0.0,
    0.2
  ],
  "asr_model": "base",
  "condition_on_previous_text": false,
  "stutter_confident_thresh": 0.6,
  "stutter_uncertain_thresh": 0.35,
  "prolongation_min_sec": 0.32,
  "block_silence_min_sec": 0.4,
  "filler_contextual_pause_sec": 0.2
}
```