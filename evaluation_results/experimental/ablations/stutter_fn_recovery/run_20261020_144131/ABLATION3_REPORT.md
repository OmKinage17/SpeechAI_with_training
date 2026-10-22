# SpeechAI — Ablation 3: Targeted Stutter False-Negative Recovery Report

**Evaluation Run:** `run_20261020_144131`  
**Evaluation Date:** October 20, 2026  
**Benchmark Set:** 60-Clip Generalization Validation Set  
**Reference Candidate:** Ablation 2 (Fix 3 Speech Context + Combined 2-gram/3-gram + Filler/Boundary Gating)  
**Ablation 3 Candidate:** Ablation 2 + Interrupted & Filler-Bridged Repetition Detection  
**Feature Cache SHA-256:** `91ad23deef8331ec52953412927ca4a08b2d113def0f5dc997f70918dbe4d076` (Verified Pristine)  
**Protected Baseline Immutability:** 100% Verified Clean  

---

## 1. Executive Summary

Ablation 3 introduces an isolated, targeted **Interrupted & Filler-Bridged Repetition Detector** designed to recover genuine stutter false negatives caused by discourse fillers, parentheticals, and hesitation connectors intervening between repeated words or phrases.

### Key Findings:
1. **Stutter Recall Improvement:** Stutter recall increased by **+11.43%** (62.86% $\rightarrow$ **74.29%**), recovering **4 genuine stutter false negatives** (true positives: 22 $\rightarrow$ **26**, false negatives: 13 $\rightarrow$ **9**).
2. **Rock-Solid Precision (Zero New False Positives):** Stutter precision rose from 84.62% to **86.67%** (+2.05%) with **0 new false positives** (FP count remained identical at **4**).
3. **Zero Lost Stutters / Regressions:** Zero true positive stutter clips regressed to Clean or Filler (4 net recoveries, 0 regressions).
4. **Overall Accuracy & F1 Gains:** 3-class accuracy advanced from 63.33% to **70.00%** (+6.67%), Macro F1 advanced from 0.6053 to **0.6622** (+0.0569), and Stutter F1 advanced from 0.7213 to **0.8000** (+0.0787).
5. **Multi-Corpus Representation:** Recoveries occurred across all three distinct evaluation corpora: UCLASS (+2 recoveries), FluencyBank (+1 recovery), and LibriSpeech / Libri_Stutter (+1 recovery).

---

## 2. Controlled Evaluation Metrics

### 2.1 Multiclass Metrics (3-Class: Clean / Stutter / Filler)

| Metric | Reference (Ablation 2) | Ablation 3 (Recovered) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 63.33% | **70.00%** | **+6.67%** |
| **Macro F1** | 0.6053 | **0.6622** | **+0.0569** |
| **Weighted F1** | 0.6440 | **0.7076** | **+0.0636** |
| **Stutter Precision** | 84.62% | **86.67%** | **+2.05%** |
| **Stutter Recall** | 62.86% | **74.29%** | **+11.43%** |
| **Stutter F1** | 0.7213 | **0.8000** | **+0.0787** |
| **Filler Precision** | 52.94% | **64.29%** | **+11.35%** |
| **Filler Recall** | 81.82% | **81.82%** | **+0.00%** |
| **Filler F1** | 0.6429 | **0.7200** | **+0.0771** |

### 2.2 Binary Stutter Detection Metrics

| Metric | Reference (Ablation 2) | Ablation 3 (Recovered) | Delta |
| :--- | :---: | :---: | :---: |
| **Binary Accuracy** | 71.67% | **78.33%** | **+6.67%** |
| **Precision** | 84.62% | **86.67%** | **+2.05%** |
| **Recall** | 62.86% | **74.29%** | **+11.43%** |
| **F1 Score** | 0.7213 | **0.8000** | **+0.0787** |
| **Specificity** | 84.00% | **84.00%** | **0.00%** |
| **True Positives (TP)** | 22 | **26** | **+4** |
| **False Positives (FP)** | 4 | **4** | **0** |
| **False Negatives (FN)** | 13 | **9** | **-4** |
| **True Negatives (TN)** | 21 | **21** | **0** |

---

## 3. Confusion Matrices (Recomputed from Raw Predictions)

### Multiclass 3-Class Confusion Matrix

#### Reference Candidate (Ablation 2):
```
                Predicted Clean  Predicted Stutter  Predicted Filler
True Clean             7                  4                  3
True Stutter           8                 22                  5
True Filler            2                  0                  9
```

#### Ablation 3 (Candidate):
```
                Predicted Clean  Predicted Stutter  Predicted Filler
True Clean             7                  4                  3
True Stutter           7                 26                  2
True Filler            2                  0                  9
```

---

## 4. Per-Corpus Breakdown

### Corpus: `SEP-28k`

| Metric | Reference (Ablation 2) | Ablation 3 | Delta |
| :--- | :---: | :---: | :---: |
| 3-Class Accuracy | 55.00% | **65.00%** | **+10.00%** |
| Stutter Precision | 87.50% | **90.00%** | **+2.50%** |
| Stutter Recall | 63.64% | **81.82%** | **+18.18%** |
| Stutter F1 | 0.7368 | **0.8571** | **+0.1203** |
| Stutter TP/FP/FN/TN | 7/1/4/8 | **9/1/2/8** | TP +2 |

### Corpus: `FluencyBank`

| Metric | Reference (Ablation 2) | Ablation 3 | Delta |
| :--- | :---: | :---: | :---: |
| 3-Class Accuracy | 65.00% | **70.00%** | **+5.00%** |
| Stutter Precision | 100.00% | **100.00%** | **+0.00%** |
| Stutter Recall | 30.00% | **40.00%** | **+10.00%** |
| Stutter F1 | 0.4615 | **0.5714** | **+0.1099** |
| Stutter TP/FP/FN/TN | 3/0/7/10 | **4/0/6/10** | TP +1 |

### Corpus: `LibriSpeech`

| Metric | Reference (Ablation 2) | Ablation 3 | Delta |
| :--- | :---: | :---: | :---: |
| 3-Class Accuracy | 70.00% | **75.00%** | **+5.00%** |
| Stutter Precision | 80.00% | **81.25%** | **+1.25%** |
| Stutter Recall | 85.71% | **92.86%** | **+7.15%** |
| Stutter F1 | 0.8276 | **0.8667** | **+0.0391** |
| Stutter TP/FP/FN/TN | 12/3/2/3 | **13/3/1/3** | TP +1 |

---

## 5. Changed Predictions & False-Negative Recovery Audit

A total of **4 predictions changed** between Ablation 2 and Ablation 3.
**Every single changed prediction was a verified true positive recovery (0 false positives introduced, 0 true positive regressions).**

| Clip ID | Corpus | GT | Ablation 2 Pred | Ablation 3 Pred | Category | Linguistic Evidence & Matched Pattern |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `M_0061_16y9m-1_dysfluent_081.wav` | SEP-28k | `Stutter` | `Filler` | **`Stutter`** | **recovered_false_negative** | 'it depends' across 'sort of' (interrupted_ngram_repetition_2gram) |
| `M_0061_16y9m-1_dysfluent_046.wav` | SEP-28k | `Stutter` | `Filler` | **`Stutter`** | **recovered_false_negative** | 'school' across 'the' (interrupted_word_repetition) |
| `FluencyBank_017_2.wav` | FluencyBank | `Stutter` | `Clean` | **`Stutter`** | **recovered_false_negative** | 'still' across 'obviously' (interrupted_word_repetition) |
| `198-209-0025.flac` | LibriSpeech | `Stutter` | `Filler` | **`Stutter`** | **recovered_false_negative** | 'time and' across 'er at' (interrupted_ngram_repetition_2gram) |

---

## 6. Implementation Architecture & Immutability

### Module Isolation:
1. `backend/experimental/interrupted_repetition.py`: Encapsulates `InterruptedRepetitionDetector` with parameterizable skip bounds, stem matching, and strict negative controls.
2. `backend/experimental/config.py`: Exposes `enable_interrupted_repetition: bool = False` (disabled by default, ensuring 100% backward compatibility).
3. `backend/experimental/stutter_detector.py`: Dispatches to `InterruptedRepetitionDetector` only when the experimental flag is explicitly enabled.
4. `check_immutability.py`: Confirmed all 12 protected baseline and production files remain 100% untouched.

---

## 7. Decision Recommendation: ACCEPT

> [!IMPORTANT]
> **Recommendation:** **ACCEPT**
> 
> **Evidence:**
> - Stutter Recall improved by **+11.43%** (62.86% $\rightarrow$ **74.29%**).
> - Stutter Precision improved by **+2.05%** (84.62% $\rightarrow$ **86.67%**).
> - **Zero new false positives** were introduced (FP count remained identical at **4**).
> - **Zero stutter regressions** occurred (0 lost stutter cases).
> - 3-class accuracy advanced from 63.33% to **70.00%** and Macro F1 rose from 0.6053 to **0.6622**.
> - Verified improvements across all three evaluation corpora (UCLASS, FluencyBank, LibriSpeech).
> - The evaluation fully satisfies all acceptance criteria.