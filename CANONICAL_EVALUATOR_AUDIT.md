# SpeechAI — Canonical Benchmark Evaluator and Fix 3/4 Reconciliation Audit

**Audit Date:** 2026-10-20  
**Project:** `E:\SpeechPrj`  
**Auditor:** Antigravity Autonomous Audit Suite  
**Scope:** Final reconciliation of evaluation pipelines across Fix 1, Fix 2B Variant 7, Fix 3, and Fix 4; establishment of a read-only canonical evaluation engine; and systematic assessment of historical artifact comparability.  
**Holdout Dataset Status:** **FROZEN & IMMUTABLE** (The 750-clip holdout dataset remained completely untouched and uninspected; all 12 protected baseline and benchmark files cryptographically verified).

---

## 1. Executive Summary & Canonical Recommendation

### 1.1 The Core Methodological Conflict
This audit conclusively resolves why development benchmark results diverged between `FIX3_SYLLABLE_CONTEXT_REPORT.md` (reporting $56.67\%$ Accuracy, $0.4794$ Macro F1, $50.00\%$ Stutter Recall) and `FIX4_LEXICAL_SYNTACTIC_CONTEXT_REPORT.md` (reporting $53.33\%$ Accuracy, $0.4606$ Macro F1, $45.83\%$ Stutter Recall):

1. **Pipeline Divergence:** Historical benchmarks were evaluated along two incompatible feature-extraction and caching pipelines:
   - **Live Acoustic Pipeline (`run_fix3_benchmark.py`):** Evaluated raw audio and live text sequentially. Because Whisper word-level timestamps were not saved in the baseline manifest (`words_list: None`), word boundaries were reconstructed synthetically. This produced zero inter-word silence gaps, disabling live acoustic block detection and causing genuine block-driven stutter clips (e.g., `103-1240-0005.flac`) to be missed as False Negatives ($10/24$ recall).
   - **Hybrid Simulation Pipeline (`simulate_fix4_dev.py`):** Recomputed acoustic prolongations dynamically while re-injecting preflight cached blocks and repetitions from `dev_eval_results_30clips_refined.json`. This restored `103-1240-0005.flac` to a True Positive ($11/24$ recall), but masked standalone acoustic prolongation recoveries (such as `103-1240-0008.flac`).
   - **Analytical Baseline Inheritance (Fix 3 Narrative):** Inherited 12 True Positives from the Fix 2B Variant 7 report (which had counted `M_0030_16y4m_1_dysfluent_009.wav` as a TP), and added $+1$ correct clip from eliminating `FluencyBank_010_17.wav`'s false alarm ($17/30 = 56.67\%$).
2. **Strict Denominator & Timeout Accounting:**
   - All 30 clips are preserved in the denominator ($N=30$) across all evaluations.
   - Timeouts were bounded by the $25.0\,\text{s}$ watchdog. Timeouts returned empty transcripts, preserved raw acoustic evaluation, and were scored strictly as non-stutter if uncorroborated (never treated as a separate class).

### 1.2 Formal Recommendation (Task 4)
> [!IMPORTANT]
> **Definitive Recommendation: Controlled Development-Set Rerun Required**  
> Historical artifacts **cannot be treated as equivalent without qualification** because `words_list` (inter-word silence intervals) was stripped from `dev_eval_results_30clips_refined.json`, forcing later scripts to either hallucinate uniform word intervals (breaking block detection) or rely on preflight cached block events.  
> 
> Before implementing Fix 5, the team must execute **one controlled rerun of the 30-clip development set** using a unified script that caches exact Whisper word timestamps with silence intervals. This will produce a pristine, deterministic baseline for Fix 2B, Fix 3, and Fix 4 under identical feature inputs.

---

## 2. Canonical Source of Truth Audit (Task 2)

Each input modality used by the stutter detection system was inspected to identify its canonical source of truth and verify whether it can be reconstructed:

| Modality / Feature | Historical Location | Reconstructible from Disk? | Canonical Source of Truth | Integrity Status |
| :--- | :--- | :---: | :--- | :---: |
| **Audio Waveforms** | `Datasets/<corpus>/<file>` | **YES (100%)** | Local WAV/FLAC files on disk | Verified pristine (all 30 exist) |
| **Ground-Truth Labels** | `dev_eval_results_30clips_refined.json` | **YES (100%)** | `p["gt_dominant"]`, `p["gt_has_stutter"]` | Verified invariant (24 Stutter, 5 Filler, 1 Clean) |
| **ASR Transcripts** | `p["pred_transcript"]` | **YES (100%)** | Whisper ASR output saved in JSON | Available (empty string on timeouts) |
| **Word Timestamps** | `p["words_list"]` | **NO (Stripped)** | **Missing from historical JSON (`None`)** | **Cannot be reconstructed without rerun** |
| **Inter-Word Silence** | Derived from `words_list` gaps | **NO (Stripped)** | Word onset/offset delta: $t_{i+1}^{\text{start}} - t_i^{\text{end}}$ | **Cannot be reconstructed without rerun** |
| **Acoustic Prolongations**| `p["details"]["stutter"]` | **YES (Dynamic)** | pYIN pitch stability + spectral flux | 100% reproducible via `AcousticFeatureExtractor` |
| **Acoustic Blocks** | `p["details"]["stutter"]` | **PARTIAL** | Preflight cached events only | Bypassed in live runs due to missing `words_list` |
| **Lexical Repetitions** | `p["details"]["stutter"]` | **YES (Dynamic)** | Regex / token repetition analyzer | 100% reproducible via `detect_lexical_repetitions` |

---

## 3. Canonical Metrics Engine (Task 3)

The canonical evaluation harness ([canonical_evaluator.py](file:///e:/SpeechPrj/evaluation_results/experimental/audit/canonical_eval/canonical_evaluator.py)) was constructed as an independent, read-only evaluation suite. It guarantees:
- Identical 30-clip manifest and ground-truth mapping across all configurations.
- Unchanging denominator ($N = 30$).
- Standard mathematical definitions without heuristic adjustments.
- Machine-readable JSON output for per-clip comparisons and aggregate summaries.

### 3.1 Aggregate Benchmark Comparison

| Metric | Fix 1 Baseline | Fix 2B Baseline (Variant 7) | Fix 3 Contextual (Live Saved) | Fix 3 Contextual (Hybrid Sim) | Fix 4 Contextual (Saved Artifact) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Harness Pipeline** | Baseline | Hybrid Simulated | Live Acoustic | Hybrid Simulated | Hybrid Simulated |
| **3-Class Accuracy** | **60.00%** (18/30) | **50.00%** (15/30) | **50.00%** (15/30) | **53.33%** (16/30) | **53.33%** (16/30) |
| **Macro Precision** | 0.4242 | 0.4639 | 0.5118 | 0.5148 | 0.5148 |
| **Macro Recall** | 0.3028 | 0.6861 | 0.7389 | 0.7528 | 0.7528 |
| **Macro F1** | 0.3416 | 0.4182 | 0.4421 | **0.4606** | **0.4606** |
| **Weighted F1** | 0.6389 | 0.6042 | 0.5714 | 0.6042 | 0.6042 |
| **Stutter Precision**| 0.7727 (17/22) | 0.9167 (11/12) | **1.0000** (10/10) | **1.0000** (11/11) | **1.0000** (11/11) |
| **Stutter Recall** | **70.83%** (17/24) | 45.83% (11/24) | 41.67% (10/24) | 45.83% (11/24) | **45.83%** (11/24) |
| **Stutter F1** | 0.7391 | 0.6111 | 0.5882 | 0.6286 | 0.6286 |
| **Stutter TP** | 17 | 11 | 10 | 11 | 11 |
| **Stutter FP** | 5 | 1 | **0** | **0** | **0** |
| **Stutter FN** | 7 | 13 | 14 | 13 | 13 |
| **Stutter TN** | 1 | 5 | 6 | 6 | 6 |
| **Filler Precision** | 0.5000 (1/2) | 0.3750 (3/8) | 0.4444 (4/9) | 0.4444 (4/9) | 0.4444 (4/9) |
| **Filler Recall** | 0.2000 (1/5) | 0.6000 (3/5) | 0.8000 (4/5) | 0.8000 (4/5) | 0.8000 (4/5) |
| **Filler F1** | 0.2857 | 0.4615 | 0.5714 | 0.5714 | 0.5714 |
| **Clean Precision** | 0.0000 (0/6) | 0.1000 (1/10) | 0.0909 (1/11) | 0.1000 (1/10) | 0.1000 (1/10) |
| **Clean Recall** | 0.0000 (0/1) | 1.0000 (1/1) | 1.0000 (1/1) | 1.0000 (1/1) | 1.0000 (1/1) |
| **Clean F1** | 0.0000 | 0.1818 | 0.1667 | 0.1818 | 0.1818 |
| **FluencyBank FPs** | 5 | 1 | **0 (Zero)** | **0 (Zero)** | **0 (Zero)** |

---

## 4. Deep-Dive Tracing of the Five Focal Audit Questions (Task 5)

### 4.1 Why `M_0030_16y4m_1_dysfluent_009.wav` Changes from TP to FN
- **Fix 1 Baseline Status:** True Positive (`pred_dominant = "Stutter"`).
  - *Mechanism:* Audio contains two prolongation candidates: $0.37\,\text{s}$ and $0.59\,\text{s}$. Under Fix 1's $0.32\,\text{s}$ duration floor, both candidates were admitted. Two co-occurring prolongations met the multi-event corroboration condition, boosting confidence to $0.60 \ge 0.60$.
- **Fix 2B / 3 / 4 Status:** False Negative (`pred_dominant = "Clean"`).
  - *Mechanism:* The minimum duration floor was raised to $0.50\,\text{s}$. Candidate 1 ($0.37\,\text{s}$) was rejected. Candidate 2 ($0.59\,\text{s}$) passed the floor but remained a solitary standalone candidate.
  - Candidate 2 failed the severe standalone threshold ($\ge 0.60\,\text{s}$ by $10\,\text{ms}$).
  - ASR decode timed out at $25.0\,\text{s}$, producing an empty transcript (`""`). There were 0 lexical repetitions and 0 blocks.
  - Standalone uncorroborated prolongation confidence was capped at $0.50$.
  - With confidence $0.50 < 0.60$, the clip was classified as negative (`Clean`).
- **Audit Reconciliation:** The Fix 3 narrative report had counted this clip as a TP based on historical manual tables ($12/24$ recall), whereas automated code evaluation strictly categorizes it as an FN ($11/24$ recall).

### 4.2 Why `103-1240-0005.flac` is Missed by Live Fix 3 Evaluation
- In `dev_eval_results_30clips_refined.json`, `words_list` was stored as `None`.
- In `run_fix3_benchmark.py`, word boundaries were created synthetically by dividing duration evenly: $w_{\text{dur}} = \text{duration} / N_{\text{words}}$.
- This created contiguous words with $\text{gap} = 0.00\,\text{s}$.
- The block detector requires silence $\ge 0.25\,\text{s}$. Consequently, live block detection found **0 blocks** across all clips.
- `103-1240-0005.flac` has 0 lexical repetitions, 0 admitted prolongations, and its block was undetected $\to$ confidence $0.0 \to$ predicted `Clean` (FN).
- In `simulate_fix4_dev.py`, the cached block at $4.54\text{--}5.10\,\text{s}$ ($0.56\,\text{s}$) was preserved, restoring it to a TP.

### 4.3 Verified Status of `103-1240-0008.flac`: Acoustic Recovery vs Block Masking
- **Acoustic Truth:** Prolongation on `"bloom,"` at $5.54\text{--}6.11\,\text{s}$ (duration $0.57\,\text{s}$, pitch std $1.13\,\text{st}$, flux $0.019$, elongation $2.51\times$).
- **Algorithm Action:** Fix 4's boundary attenuation formula attenuated the clause discount from $-0.20$ to $-0.03$, raising context score from $0.492$ (Fix 3) to $0.762 \ge 0.65$ (Fix 4). In standalone acoustic evaluation without preflight blocks, this **genuinely recovers the clip** ($+4.17\,\text{pp}$ recall).
- **Masking in Saved Artifact:** In `dev_eval_results_30clips_fix4.json`, the clip already had an `inter_word_acoustic_block` at $4.40\text{--}4.94\,\text{s}$ from preflight cache. That block alone assigned confidence $0.60$, marking it as Stutter in Fix 2B, Fix 3, and Fix 4. Thus, in the saved hybrid artifact, the acoustic recovery was masked and did not change aggregate metrics.

### 4.4 Why UCLASS has 10% Accuracy while Libri_Stutter has 90% in Fix 4
- **Phonetic & Morphological Disparity:**
  - **Libri_Stutter ($N=10$, all Stutter):** Characterized by explicit lexical repetitions (`0003`, `0004`, `0007`, `0009` have 2–3 repetitions each), clear inter-word blocks (`0000`, `0005`, `0006`, `0007`, `0008`), and severe acoustic prolongations (`0000`, `0002`, `0006` $\ge 0.60\,\text{s}$). Exactly 9 out of 10 clips trigger detection ($90\%$ accuracy).
  - **UCLASS ($N=10$, all Stutter):** Conversational speech of adolescent stutterers. Dominated by brief, isolated prolongations ($0.32\text{--}0.40\,\text{s}$ in `011`, `012`, `013`), single uncorroborated prolongations ($0.59\,\text{s}$ in `009`), and subtle tense pauses without repetitions. Under strict gating ($\ge 0.50\,\text{s}$ and corroboration), 9 out of 10 clips fail. Only `015` (which contains 6 lexical repetitions) is detected ($10\%$ accuracy).

### 4.5 Timeout Handling Equivalence
- **Equivalence Confirmed:** In all evaluated configurations, the three timeouts (`M_0030_009`, `_011`, `_012`) are retained in the denominator ($N=30$).
- When ASR times out, acoustic features are evaluated independently. Clips are classified as Stutter if acoustic criteria are met (as occurred in Fix 1), or classified as Clean/Filler if criteria fail. They are never discarded or assigned to a phantom "timeout" class.

---

## 5. Per-Clip Disagreement Matrix across All Configurations

Of the 30 development clips, exactly 18 have identical predictions across all five configurations, and **12 exhibit configuration-dependent disagreements**:

| Filename | Dataset | Ground Truth | Fix 1 Baseline | Fix 2B Baseline | Fix 3 Live Saved | Fix 3 Hybrid Sim | Fix 4 Saved Artifact | Root Cause of Disagreement |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `M_0030_009` | UCLASS | Stutter | **Stutter (TP)** | Clean (FN) | Clean (FN) | Clean (FN) | Clean (FN) | $0.37\,\text{s}$ dropped; $0.59\,\text{s} < 0.60\,\text{s}$ severe floor |
| `M_0030_011` | UCLASS | Stutter | **Stutter (TP)** | Clean (FN) | Clean (FN) | Clean (FN) | Clean (FN) | $0.40\,\text{s} < 0.50\,\text{s}$ duration floor |
| `M_0030_012` | UCLASS | Stutter | **Stutter (TP)** | Filler (FN) | Filler (FN) | Filler (FN) | Filler (FN) | $0.40\,\text{s} < 0.50\,\text{s}$ duration floor |
| `M_0030_013` | UCLASS | Stutter | **Stutter (TP)** | Filler (FN) | Filler (FN) | Filler (FN) | Filler (FN) | $0.32\,\text{s} < 0.50\,\text{s}$ duration floor |
| `FluencyBank_079` | FluencyBank | Stutter | **Stutter (TP)** | Filler (FN) | Filler (FN) | Filler (FN) | Filler (FN) | $0.32\,\text{s} < 0.50\,\text{s}$ duration floor |
| `FluencyBank_097` | FluencyBank | Stutter | **Stutter (TP)** | Filler (FN) | Filler (FN) | Filler (FN) | Filler (FN) | $0.38\,\text{s} < 0.50\,\text{s}$ duration floor |
| `FluencyBank_017` | FluencyBank | Filler | Stutter (FP) | Stutter (FP) | **Filler (TP)** | **Filler (TP)** | **Filler (TP)** | **Fix 3 Breakthrough:** Phrase-final lengthening damped |
| `FluencyBank_018` | FluencyBank | Filler | Stutter (FP) | **Filler (TP)** | **Filler (TP)** | **Filler (TP)** | **Filler (TP)** | Eliminated by Fix 2 pitch/flux rigidity |
| `FluencyBank_040` | FluencyBank | Filler | Stutter (FP) | Clean (TN) | Clean (TN) | Clean (TN) | Clean (TN) | Fluent $400\,\text{ms}$ vowel eliminated by $0.50\,\text{s}$ floor |
| `FluencyBank_041` | FluencyBank | Filler | Stutter (FP) | **Filler (TP)** | **Filler (TP)** | **Filler (TP)** | **Filler (TP)** | Fluent $390\,\text{ms}$ vowel eliminated by $0.50\,\text{s}$ floor |
| `FluencyBank_000` | FluencyBank | Clean | Stutter (FP) | **Clean (TP)** | **Clean (TP)** | **Clean (TP)** | **Clean (TP)** | Fluent $350\,\text{ms}$ vowel eliminated by $0.50\,\text{s}$ floor |
| `103-1240-0005` | Libri_Stutter | Stutter | **Stutter (TP)** | **Stutter (TP)** | Clean (FN) | **Stutter (TP)** | **Stutter (TP)** | Block missed in Live Fix 3 due to synthetic words |

---

## 6. Machine-Readable Audit Deliverables

The following canonical JSON artifacts have been generated and validated:
1. **Per-Clip Comparison JSON:**  
   [per_clip_comparison.json](file:///e:/SpeechPrj/evaluation_results/experimental/audit/canonical_eval/per_clip_comparison.json)  
   Contains aligned ground-truth labels, predicted dominant classes, stutter/filler flags, confidence scores, timeout indicators, and multiclass correctness across all 30 clips and 5 configurations.
2. **Aggregate Metrics Comparison JSON:**  
   [aggregate_metrics.json](file:///e:/SpeechPrj/evaluation_results/experimental/audit/canonical_eval/aggregate_metrics.json)  
   Contains precision, recall, macro F1, weighted F1, binary stutter confusion matrices, and 3x3 confusion matrices for all evaluated configurations.

---

## 7. Cryptographic Hash Verification

SHA-256 digests of all relevant files were computed before and after the audit. Zero baseline or holdout files were altered.

| File Path | SHA-256 Digest | Status |
| :--- | :--- | :---: |
| `backend/main.py` | `151b37f86d97b1e2e3f524e4604b344fe3be9b104975f42e47e5b5655519db12` | Unchanged |
| `backend/run_main_benchmark.py` | `65e4600454bceccce1fe54215f7ba11ecb5bfa0fec9eb482ae743a1ba7aa3432` | Unchanged |
| `evaluation_results/pilot_manifest.json` | `9133a2376c965cea5fa500e21a221f5df8b8ba00df9eb7614e2d3126f4f2ceea` | Unchanged |
| `evaluation_results/main_benchmark/main_benchmark_manifest.json` | `78ca0ba28b2eb8e3cb98be74beabde8b8f2762e15bc99d91f24d1a1b1a03975a` | Unchanged |
| `evaluation_results/main_benchmark/main_benchmark_predictions.json` | `c188f19f79ac001e309feee91c5324aa8f921ea0283c8cfaec65b2694931a293` | Unchanged |
| `dev_eval_results_30clips_refined.json` | `f70b1c9ce9e854fb4edb217ce77e63c55767e666a606317c3f38ff26b87ef932` | Unchanged |
| `dev_eval_results_30clips_fix3.json` | `fd71d7129bfa84438a9c08cbd789c0357820c7f77f212d12051eb4e00aceb08b` | Unchanged |
| `dev_eval_results_30clips_fix4.json` | `6bd8ff5c038fac8b1d21793a28befb8f602d2e664814183f14d21b4a04669252` | Unchanged |
| `backend/experimental/config.py` | `20a4cc5603e870a326a56c9437870218e2926375555e7fb1455eb8f34cd23434` | Unchanged |
| `backend/experimental/speech_context.py` | `7e2884859f6f3a67ebb7b5dfbb84845ed178dca783bbea2a87f67f1f919aeb81` | Unchanged |
| `backend/experimental/stutter_detector.py`| `015275351c6aa7454ec4ba13c5bc41e7147141893c81eb4d0557a1356014feca` | Unchanged |

All 12 protected production and benchmark files passed validation via `check_immutability.py`.

---

## 8. Exact Next Action

Do not modify detector algorithms, change thresholds, or evaluate on the frozen holdout dataset.

**Immediate Next Action:**
1. Implement a single, unified development benchmark runner (`backend/experimental/run_dev_benchmark.py`).
2. Have it run Whisper ASR with `word_timestamps=True` on the 30 development clips, caching both the transcript and the full `words_list` (including word start/end times) into the benchmark record.
3. Re-run Fix 2B, Fix 3, and Fix 4 sequentially through this canonical runner to establish an unassailable baseline before proceeding to Fix 5.
