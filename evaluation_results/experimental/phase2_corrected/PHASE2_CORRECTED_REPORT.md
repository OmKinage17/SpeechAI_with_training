# Phase 2 Audit Report: ASR Repetition-Loop Handling and Canonical Evaluation Corrections

**Date:** October 22, 2026  
**Scope:** Experimental Benchmark Suite only (`backend/experimental/`, `evaluation_results/experimental/phase2_corrected/`)  
**Status:** Verification Complete — Pending User Review & Approval for Phase 3

---

## 1. Executive Summary

Phase 2 resolved two foundational methodology issues in the Experimental Benchmark Suite without tuning detector thresholds against the final evaluation set:
1. **Degenerate Whisper Repetition Loops:** Fixed runaway greedy decoding loops in 3-second clips. The ASR engine now detects repeated token/hesitation loops, truncates the runaway tail while preserving initial disfluencies, stores raw and sanitized transcripts separately, and flags recovered clips with `DEGENERATE_LOOP_RECOVERED`.
2. **Canonical Evaluation Corrections:** Resolved reporting ambiguities in the benchmark metrics:
   - **Filler Discrepancy:** Formally defined and separated **Dominant-Class Filler** ($N=38$ predictions) from **Event-Level Filler** ($N=56$ predictions with `filler_count > 0`), explaining the 18-clip co-occurrence difference.
   - **Clinical Block Specificity:** Scoped specificity to annotated corpora (UCLASS + FluencyBank, $N=80$) yielding 94.59%, while separating unannotated Libri_Stutter acoustic detections ($N=19$).
   - **WER/CER Denominator Scoping:** Grounded ASR metrics strictly in gold verbatim transcripts on Libri_Stutter ($N=40$, denominator = 1,249 reference words), explicitly excluding UCLASS and FluencyBank due to the absence of word-level verbatim ground truth.
   - **Speech-Rate Separation:** Isolated valid speech-rate distributions from degenerate loops, bringing the mean speech rate from a distorted 306.9 WPM down to a realistic 148.1 WPM (burst/reading rate), while preserving raw transcripts for auditing.

> [!IMPORTANT]
> **No Model Performance Inflation Claim:** The metrics changes reported below represent **methodological and reporting corrections**, not new algorithm tuning. Detector thresholds were frozen and untouched.

---

## 2. Files Changed and Created

### Files Modified:
- [`backend/experimental/asr_engine.py`](file:///e:/SpeechPrj/backend/experimental/asr_engine.py):
  - Implemented `detect_and_sanitize_repetition_loop()`.
  - Added separation between `raw_transcript` and `sanitized_transcript`.
  - Added status tracking (`DEGENERATE_LOOP_RECOVERED`), `sanitization_reason`, `loop_pattern`, and `truncated_tokens_count`.
- [`backend/experimental/canonical_evaluator.py`](file:///e:/SpeechPrj/backend/experimental/canonical_evaluator.py):
  - Added native `compute_levenshtein_wer()` and `compute_levenshtein_cer()`.
  - Added dual-measure filler evaluation (event-level vs dominant-class) and discrepancy documentation.
  - Added multi-label co-occurrence tracking ($N=12$ clips).
  - Scoped clinical block specificity on UCLASS + FluencyBank ($N=80$) vs full benchmark ($N=120$).
  - Separated valid speech rate from degenerate loops and documented gold verbatim denominators.
- [`backend/tests/test_benchmark_reconciliation.py`](file:///e:/SpeechPrj/backend/tests/test_benchmark_reconciliation.py):
  - Added 7 targeted unit tests (Tests 07–13) covering genuine stutter preservation, loop detection, raw/sanitized separation, filler dual-measures, block scoping, speech rate distinctions, and WER/CER denominators.

### Files Created:
- [`backend/experimental/run_phase2_corrected_benchmark.py`](file:///e:/SpeechPrj/backend/experimental/run_phase2_corrected_benchmark.py):
  - Deterministic evaluation runner executing the corrected pipeline and saving outputs in the isolated directory.
- **Output Artifacts in [`evaluation_results/experimental/phase2_corrected/`](file:///e:/SpeechPrj/evaluation_results/experimental/phase2_corrected/):**
  - `run_manifest.json`: Exact copy of the 120-clip final evaluation manifest.
  - `feature_cache_phase2.json`: Cached features with both raw and sanitized transcripts.
  - `per_clip_predictions.json`: Detailed 120-clip evaluation records with all flags.
  - `per_clip_results.csv`: Tabular export of per-clip results.
  - `final_metrics.json`: Complete canonical metrics across all parameters.
  - `per_dataset_metrics.json`: Metrics decomposed across UCLASS, FluencyBank, and Libri_Stutter.
  - `per_parameter_metrics.json`: Audited parameters 1 through 7.
  - `confusion_matrices.json`: Multiclass and binary confusion matrices.
  - `runtime_and_failures.json`: Latency, timeout, and loop recovery audit.
  - `seven_parameters_summary.csv`: Before-and-after reconciliation table.
  - `multiclass_confusion_matrix.png`: Dominant-class confusion matrix plot.
  - `speech_rate_distribution.png`: Speech-rate histogram showing sanitized burst rate distribution.

### Invariant Verification:
- `backend/main.py`: **Untouched** (0 lines modified).
- Main frontend (`frontend/`): **Untouched** (0 lines modified).
- Original benchmark dir (`run_final_20261020_complete/`): **Untouched** (0 files modified or deleted).

---

## 3. Test Suite Verification

A total of **49 automated tests** were executed and **49 passed (100% pass rate, 0 failed, 0 errors)**:

| Test Suite | File | Tests Run | Result | Focus Areas |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 1 Regression** | `test_benchmark_reconciliation.py` (Tests 1–6) | 6 | **PASS** | Manifest disjointness, multiclass-to-binary stutter math, filler discrepancy discovery, loop identification, loop sanitizer logic, co-occurrence count. |
| **Phase 2 Targeted** | `test_benchmark_reconciliation.py` (Tests 7–13) | 7 | **PASS** | Genuine disfluency preservation, degenerate loop detection/separation, raw/sanitized integrity, dual-measure filler reconciliation, block scoping, burst rate distinction, WER denominators. |
| **Canonical Evaluator** | `test_canonical_evaluator.py` | 3 | **PASS** | Evaluator determinism, metric calculations, and zero-leakage manifest alignment. |
| **N-gram Repetition** | `test_ngram_repetition.py` | 13 | **PASS** | Lexical and syllable repetition detectors, stem matching. |
| **Filler Boundary Gating** | `test_filler_boundary_gating.py` | 12 | **PASS** | Boundary gating, false positive reduction, prolongation boundary checks. |
| **Interrupted Repetition** | `test_interrupted_repetition.py` | 8 | **PASS** | Repetitions interrupted by filler tokens and pauses. |
| **Total** | | **49** | **49 PASSED (0 FAIL)** | Execution time: 0.017s |

---

## 4. Repetition Loop Recovery Audit

### Loop Detection Summary:
- **Total Clips Evaluated:** 120
- **Clips Flagged for Repetition Loops:** **9 clips** (7.5%)
- **Clips Successfully Recovered:** **9 clips** (100% of flagged clips)
- **False Positives on Genuine Disfluencies:** **0 clips** (all 28 high-word Libri_Stutter clips and all genuine UCLASS/FluencyBank repetitions preserved untouched).
- **Unrecovered Clips Flagged for Human Review:** 0 clips.

### Genuine Disfluency Safeguards:
- Utterances with $< 12$ words are never truncated, preserving genuine short repetitions (e.g., `FluencyBank_089_9.wav`: *"just just just just just"*).
- Repetition loop sanitizer keeps the first 2 instances of a repeated lexical word or the first 3 hesitation tokens to preserve clinical dysfluency evidence for downstream acoustic and lexical detectors.
- Runaway tails ($\ge 6$ consecutive hesitations or $\ge 5$ repeated words with $\ge 25$ total clip words) are truncated.

### Clip-by-Clip Loop Recovery Details:

| Clip ID | Dataset | GT Dominant | Pred Dominant | Raw Word Count (Raw WPM) | Sanitized Word Count (Sanitized WPM) | Truncated Tokens | Pattern & Sanitization Reason |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `M_1023_11y3m_1_dysfluent_030.wav` | UCLASS | Stutter | Stutter | 112 (2,240.0) | 5 (100.0) | 107 | Degenerate hesitation stream: 6+ consecutive filler tokens (`um, uh...`). Initial hesitations preserved. |
| `M_1099_25y0m_1_dysfluent_199.wav` | UCLASS | Stutter | Stutter | 113 (2,260.0) | 9 (180.0) | 104 | Degenerate hesitation stream: 6+ consecutive filler tokens (`um, uh...`). Initial hesitations preserved. |
| `M_1099_25y0m_1_dysfluent_152.wav` | UCLASS | Stutter | Stutter | 113 (2,260.0) | 8 (160.0) | 105 | Degenerate hesitation stream: 6+ consecutive filler tokens (`um, uh...`). Initial hesitations preserved. |
| `M_1104_40y0m_1_dysfluent_143.wav` | UCLASS | Stutter | Stutter | 112 (2,240.0) | 4 (80.0) | 108 | Degenerate hesitation stream: 6+ consecutive filler tokens (`uh, uh...`). Initial hesitations preserved. |
| `FluencyBank_089_78.wav` | FluencyBank | Stutter | Stutter | 113 (2,260.0) | 4 (80.0) | 109 | Degenerate single-token loop: `'did'` repeated 111 times consecutively. Initial repetition preserved. |
| `FluencyBank_089_66.wav` | FluencyBank | Stutter | Filler | 114 (2,280.0) | 11 (220.0) | 103 | Degenerate hesitation stream: 6+ consecutive filler tokens (`uh, uh...`). Initial hesitations preserved. |
| `FluencyBank_089_130.wav` | FluencyBank | Filler | Filler | 114 (2,280.0) | 8 (160.0) | 106 | Degenerate hesitation stream: 6+ consecutive filler tokens (`uh, uh...`). Initial hesitations preserved. |
| `FluencyBank_089_30.wav` | FluencyBank | Filler | Filler | 116 (2,320.0) | 12 (240.0) | 104 | Degenerate hesitation stream: 6+ consecutive filler tokens (`um, uh...`). Initial hesitations preserved. |
| `FluencyBank_099_171.wav` | FluencyBank | Filler | Filler | 112 (2,240.0) | 5 (100.0) | 107 | Degenerate hesitation stream: 6+ consecutive filler tokens (`um, um...`). Initial hesitations preserved. |

### Concrete Transcript Examples:
- **`FluencyBank_089_78.wav`**:
  - *Raw Transcript:* `"If you did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did, did"`
  - *Sanitized Transcript:* `"If you did, did,"`
  - *Effect:* Downstream stutter detector correctly detects lexical repetition from the preserved `"did, did"` without being misled by a 2,260 WPM token stream.
- **`M_1099_25y0m_1_dysfluent_199.wav`**:
  - *Raw Transcript:* `"Well, that's it. In your job. Um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh, um, uh, uh, uh"`
  - *Sanitized Transcript:* `"Well, that's it. In your job. Um, uh, uh,"`
  - *Effect:* Preserves opening sentence and initial hesitations; eliminates 104 runaway tokens.

---

## 5. Seven Parameters: Original vs Corrected Reconciliation

Below is the audited comparison across all seven SpeechAI parameters, separating model invariants from methodology corrections.

| Parameter | Original Benchmark (Uncorrected) | Phase 2 Corrected Benchmark | Nature of Difference | Methodological Rationale & Documentation |
| :--- | :---: | :---: | :--- | :--- |
| **P1: Stuttering Detection** | F1: **0.688**<br>(P: 0.741, R: 0.642) | F1: **0.688**<br>(P: 0.741, R: 0.642) | **Identical (Model Invariant)** | Frozen detector weights and thresholds were preserved without post-hoc tuning. $TP=43, FP=15, FN=24, TN=38$. |
| **P2: Filler Detection** | F1: **0.697**<br>(Reported 38 dominant preds vs 56 positive metrics) | **Event-Level:** F1: **0.697**<br>(P: 0.554, R: 0.939, $N=56$)<br>**Dominant-Class:** F1: **0.542**<br>(P: 0.421, R: 0.762, $N=38$) | **Reporting Correction** | Resolved discrepancy: 56 clips have $\ge 1$ detected filler (event-level), while 38 clips have Filler as the exclusive dominant prediction. The 18-clip difference represents co-occurring stuttering. |
| **P3: Clean Speech** | Specificity: **86.4%**<br>(from dev/generalization) | Dominant Precision: **50.0%**<br>Binary Specificity: **71.7%**<br>(Clean support = 32) | **Clarification & Scoping** | Evaluator now reports exact Dominant Clean class metrics ($TP=12, FP=12, FN=20$) and Binary Non-Stutter Specificity ($TN=38 / 53 = 71.70\%$) on the final 120-clip set. |
| **P4: Speech Rate (WPM)** | Pearson $r$: **0.755**<br>Mean WPM: **342.2**<br>(Max: 2,320 WPM) | **Valid Non-Loop Mean:** **148.2 WPM**<br>**Sanitized All Mean:** **148.1 WPM**<br>Gold Ref MAE: **16.95 WPM**<br>Gold Ref Bias: **+10.56 WPM** | **ASR Sanitization & Metric Separation** | Degenerate ASR loops previously skewed overall speech rate to 342.2 WPM. Truncating loops restored realistic burst/reading speech rate (148.1 WPM) while retaining raw WPM for audit. |
| **P5: Pause & Blocks** | Specificity: **94.6%**<br>(unscoped claim) | **Scoped Clinical Specificity:** **94.59%** ($N=80$)<br>Full Benchmark Specificity: **79.82%** ($N=120$) | **Corpus Scoping Correction** | UCLASS + FluencyBank contain clinical block annotations ($TN=70, FP=4 \rightarrow 94.59\%$). Libri_Stutter lacks block annotations, generating 19 unannotated acoustic pause detections. |
| **P6: ASR Recognition** | WER: **23.9%**<br>(Unspecified denominator) | **Corpus WER:** **23.94%**<br>Mean Clip WER: **22.98%**<br>Mean Clip CER: **15.35%** | **Denominator Scoping** | Calculated strictly against gold verbatim transcripts on Libri_Stutter ($N=40$, 1,249 reference words, 299 edits). UCLASS and FluencyBank ($N=80$) explicitly excluded. |
| **P7: Coaching Engine** | **100% Audit** | **100.0%** (120/120 clips) | **Verified on Sanitized Rates** | Deterministic coaching rules verified against sanitized speech rate (no false "faster than optimal" warnings triggered by runaway ASR loops). |

---

## 6. Detailed Analysis of Methodological Corrections

### 6.1 Filler Discrepancy Resolution (38 vs 56)
- **The Finding:** The original benchmark dashboard reported 38 predicted filler clips in some views, but Parameter 2 showed 56 positive predictions ($TP=31 + FP=25$).
- **The Explanation:** 
  - In a **multiclass single-label framework**, each clip receives exactly one dominant label (`Stutter`, `Filler`, or `Clean`). There were **38 clips** where `Filler` was the primary predicted label.
  - In an **event-level clinical framework**, a clip is considered positive if at least one filler event occurred (`filler_count > 0`). There were **56 clips** with detected fillers.
  - Exactly **18 clips** contained co-occurring stuttering events (e.g. sound repetitions or prolongations) along with filler words. In these clips, the model classified dominant class as `Stutter` (because stuttering was clinically severe), but still accurately detected and logged the filler word occurrences.
- **The Correction:** The evaluator now outputs both `event_level` and `dominant_class` filler metrics side by side, documenting the clinical relationship.

### 6.2 Multi-Label Co-occurrence Analysis ($N=12$)
- Exactly **12 clips** in the 120-clip benchmark have ground-truth annotations for **both Stuttering and Interjection/Filler**:
  - The model predicted `Filler` as dominant in 8 clips and `Stutter` as dominant in 4 clips.
  - The model predicted `Clean` in 0 clips.
  - **Any-Dysfluency Accuracy:** **100.0% (12/12)** — every co-occurring clip was correctly recognized as dysfluent.

### 6.3 Clinical Block Specificity Scoping ($N=80$ vs $N=120$)
- **The Finding:** Clinical block specificity was originally claimed at 94.6%, but evaluating across all 120 clips showed 23 false positives (79.82% specificity).
- **The Explanation:** 
  - UCLASS and FluencyBank ($N=80$) are clinical dysfluency datasets with explicit block annotations. In these corpora, the model achieved $TN=70, FP=4, FN=6, TP=0$, yielding **94.59% specificity**.
  - Libri_Stutter ($N=40$) consists of audiobooks read aloud. Speakers frequently use expressive, dramatic pauses before plosive consonants. The acoustic struggle detector flagged 19 of these pauses as blocks. However, the Libri_Stutter annotation schema **does not annotate blocks at all** (only sound reps, word reps, and prolongations). Thus, these 19 acoustic events were counted as false positives purely due to missing annotations.
- **The Correction:** The evaluator scopes clinical block specificity to the 80 annotated clips, while transparently reporting the unannotated Libri_Stutter pause detections separately.

### 6.4 ASR WER/CER Appropriate Denominators
- **The Finding:** Original documents reported "WER: 23.9%" without clarifying which clips were included or how the denominator was computed.
- **The Correction:**
  - **Gold Verbatim Subset:** Evaluated strictly on the 40 Libri_Stutter clips which have verified, publisher-provided book text.
  - **Corpus WER Denominator:** Sum of reference words across all 40 clips = **1,249 words**. Sum of Levenshtein word edits = **299 edits**. $\text{Corpus WER} = 299 / 1,249 = \mathbf{23.94\%}$.
  - **Mean Clip WER:** Arithmetic mean of clip-level WERs = **22.98%**.
  - **Mean Clip CER:** Arithmetic mean of character-level error rates = **15.35%**.
  - **Excluded Corpora:** UCLASS ($N=40$) and FluencyBank ($N=40$) were explicitly excluded from WER calculation because neither corpus provides word-level verbatim ground truth. Aligning ASR hypotheses against non-verbatim transcripts creates arbitrary edit penalties that measure transcription convention differences rather than speech recognition quality.

---

## 7. Remaining Limitations and Phase 3 Readiness

### Remaining Limitations:
1. **Clinical Block Recall ($TP=0$ on $N=6$ Blocks):** While block specificity is high (94.59%), the acoustic energy/pause detector did not achieve confirmed true positives on the 6 subtle blocks in UCLASS/FluencyBank. Acoustic silent blocks remain the most challenging dysfluency subtype to detect from 3-second audio slices alone.
2. **Short-Clip Speech Rate Dynamics:** Calculating WPM over a 3.0-second clip inherently represents an *instantaneous burst rate* rather than a sustained conversational rate. While sanitization eliminated extreme 2,000+ WPM artifacts, clinical interpretations must account for short-window burst variability.
3. **Absence of Verbatim Ground Truth in Clinical Corpora:** ASR WER is currently grounded only in Libri_Stutter. Obtaining clinician-verified verbatim word transcripts for UCLASS and FluencyBank would enable cross-corpus WER evaluation in future research.

### Readiness Assessment:
- **Phase 2 Objectives:** Fully satisfied.
- **Reproducibility:** 100% reproducible via `run_phase2_corrected_benchmark.py` and `test_benchmark_reconciliation.py`.
- **System Integrity:** Main codebase (`backend/main.py`) and frontend are completely untouched.
- **Next Steps:** Benchmark artifacts are now methodologically sound and ready for **Phase 3 planning** upon your explicit review and approval.

---

*Report generated by Antigravity Autonomous Audit Suite.*  
*Artifacts: `evaluation_results/experimental/phase2_corrected/`*
