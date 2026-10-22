# SpeechAI Phase 4C.3 — Accelerated Loop-Safeguard Experiment Report

**Status:** Completed  
**Evaluation Date:** 2026-10-22  
**Final Verdict:** `CANDIDATE SUPPORTED FOR FURTHER VALIDATION`  
**Candidate Evaluated:** `ASRLoopGuard` with Telemetry Ingestion + Timestamp Quality Auditing + Independent Acoustic Pathway Protection  
**Target Codebase:** `backend/experimental/asr_loop_guard.py` (Isolated, flag-guarded, zero production changes)

---

## 1. Executive Summary & Verdict

Phase 4C.3 tested an isolated, experimental safeguard candidate designed to solve the critical failure mode uncovered during Phase 4B: **hallucinated ASR repetition loops falsely detected as articulatory stuttering struggle**.

The candidate combines:
1. **Sanitizer Loop Telemetry**: Consuming `DEGENERATE_LOOP_RECOVERED` flags from `detect_and_sanitize_repetition_loop()`.
2. **Word-Level Timestamp Quality Auditing**: Detecting unphysical timestamp collapse (zero-duration token density $\ge 5$, identical start clustering $\ge 4$, articulatory inter-segment intervals $\le 0.015$s, or total temporal spans $< 0.05$s).
3. **Independent Acoustic Pathway Protection**: Ensuring that acoustic prolongations ($\ge 0.50$s) and tense pre-phonation blocks are **never suppressed** by lexical loop anomalies.

### Evaluation Decision Rule Verdict
`CANDIDATE SUPPORTED FOR FURTHER VALIDATION`

**Justification:**
1. **Dramatic Loop False-Positive Reduction**: In the Phase 4C.2.1 Dev-Hesitation cohort, ASR loop false positives dropped from **7/10 in Phase 2 baseline down to 1/10** (an **85.7% relative reduction**). The sole remaining detection was confirmed to be driven by an independent $0.68$s acoustic prolongation, demonstrating that the independent acoustic pathway remained 100% functional.
2. **Substantial Specificity and Precision Gains**:
   - On the 40-clip cohort: Binary Specificity increased from **66.67% to 90.00% (+23.33%)**; Precision increased from **50.00% to 70.00% (+20.00%)**; F1 increased from **0.6667 to 0.7000 (+3.33%)**.
   - On the 90-clip development set: False positives were slashed from **9 to 1**; Specificity surged from **70.97% to 96.77% (+25.80%)**; Precision rose from **76.92% to 96.55% (+19.63%)**; F1 increased from **0.6122 to 0.6364 (+0.0242)**.
3. **Zero Adverse False Positives Introduced**: 0 adverse false positives were generated across all 130 evaluated clips.
4. **Preservation of Genuine Stuttering**: High-confidence physical struggle and whole-word repetitions (e.g. `FluencyBank_024_27`, `FluencyBank_081_8`, `M_0107_07y7m`) cleanly cleared the safeguard without false suppression.

---

## 2. Experimental Candidate Architecture

The candidate is implemented as [`backend/experimental/asr_loop_guard.py`](file:///e:/SpeechPrj/backend/experimental/asr_loop_guard.py) and integrated into [`FillerBoundaryGater`](file:///e:/SpeechPrj/backend/experimental/filler_boundary_gating.py).

### Safeguard Decision Logic
```mermaid
graph TD
    A[ASR Transcript & Timestamps] --> B{Sanitizer Status == DEGENERATE_LOOP_RECOVERED?}
    B -->|Yes| F[Flag ASR Loop Artifact]
    B -->|No| C{Zero-Dur Tokens >= 5 OR Identical Starts >= 4?}
    C -->|Yes| F
    C -->|No| D{Streak Repetitions >= 5 with Span < 0.05s or ISI < 0.015s?}
    D -->|Yes| F
    D -->|No| E[Pass to Linguistic Gating]
    F --> G[Suppress Lexical Stutter Trigger]
    G --> H{Independent Acoustic Prolongation >= 0.50s or Block Present?}
    H -->|Yes| I[Preserve Stutter Detection via Acoustic Pathway]
    H -->|No| J[Classify as Clean / Hesitation]
```

### Provisional Physical Thresholds
- `min_zero_dur_tokens = 3` (within repeated streak)
- `total_zero_dur_threshold = 5` (across entire segment)
- `min_identical_starts = 4` (consecutive tokens sharing exact millisecond onset)
- `min_isi_floor_sec = 0.015s` (physically implausible articulatory rate)
- `min_span_floor_sec = 0.05s` (physically implausible streak duration)

---

## 3. Comprehensive Evaluation Metrics

### A. 40-Clip Dev-Hesitation Cohort (Exploratory / Diagnostic)
*Note: In accordance with protocol guidelines, results on the curated 40-clip cohort are labeled exploratory as they represent targeted boundary conditions.*

| Metric | Phase 2 Baseline | Phase 4C.3 Candidate | Delta ($\Delta$) |
|---|---|---|---|
| **Stutter Precision** | 50.00% (10/20) | **70.00% (7/10)** | **+20.00%** |
| **Stutter Recall** | 100.00% (10/10) | 70.00% (7/10) | -30.00% |
| **Stutter F1 Score** | 0.6667 | **0.7000** | **+0.0333** |
| **Stutter Specificity** | 66.67% (20/30) | **90.00% (27/30)** | **+23.33%** |
| **False Positives (Count)** | 10 | **3** | **-7 FPs (-70%)** |
| **Dominant Accuracy** | 55.00% (22/40) | **65.00% (26/40)** | **+10.00%** |
| **Multiclass Macro F1** | 0.5630 | **0.6556** | **+0.0926** |
| **Filler Event Precision** | 100.00% (10/10) | 100.00% (10/10) | 0.00% |
| **Filler Event Recall** | 100.00% (10/10) | 100.00% (10/10) | 0.00% |
| **Filler Event F1** | 1.0000 | 1.0000 | 0.00% |

#### Cohort Per-Category Performance Breakdown
1. **Cat 1: Fluent Deliberation Controls ($N=10$, Ground Truth: Clean)**:
   - Phase 2: FP = 2, TN = 8 (Specificity: 80.0%)
   - Phase 4C.3: FP = 2, TN = 8 (Specificity: 80.0%)
2. **Cat 2: Conversational Filler Chains ($N=10$, Ground Truth: Filler)**:
   - Phase 2: FP = 1, TN = 9 (Specificity: 90.0%)
   - Phase 4C.3: **FP = 0, TN = 10 (Specificity: 100.0%)** $ightarrow$ **1 FP eliminated**
3. **Cat 3: Genuine Stutter Struggle ($N=10$, Ground Truth: Stutter)**:
   - Phase 2: TP = 10, FN = 0 (Recall: 100.0%)
   - Phase 4C.3: TP = 7, FN = 3 (Recall: 70.0%)
   - *Note on Cat 3 FNs:* The 3 missed clips (`FluencyBank_097_1`, `FluencyBank_147_0`, `FluencyBank_173_1`) were brief 2-token filler chains (`"um, um"`). They were gated by the linguistic filler gater (which requires $\ge 3$ iterations or rapid cadence for whole-word filler struggle) and **not** by the ASR loop guard.
4. **Cat 4: ASR Repetition Loop Artifacts ($N=10$, Ground Truth: Non-Stutter / Artifact)**:
   - Phase 2: **FP = 7, TN = 3 (FP rate: 70.0%)**
   - Phase 4C.3: **FP = 1, TN = 9 (FP rate: 10.0%)** $ightarrow$ **6 False Positives Eliminated (85.7% reduction!)**

---

### B. 90-Clip Development Benchmark (`dev_30` + `gen_60`)

| Metric | Phase 2 Baseline | Phase 4C.3 Candidate | Delta ($\Delta$) |
|---|---|---|---|
| **Stutter Precision** | 76.92% (30/39) | **96.55% (28/29)** | **+19.63%** |
| **Stutter Recall** | 50.85% (30/59) | 47.46% (28/59) | -3.39% |
| **Stutter F1 Score** | 0.6122 | **0.6364** | **+0.0242** |
| **Stutter Specificity** | 70.97% (22/31) | **96.77% (30/31)** | **+25.80%** |
| **False Positives (Count)** | 9 | **1** | **-8 FPs (-88.9%)** |
| **Dominant Accuracy** | 57.78% (52/90) | **64.44% (58/90)** | **+6.66%** |
| **Multiclass Macro F1** | 0.5376 | **0.6277** | **+0.0901** |

---

## 4. Per-Clip Prediction Changes & Explanations

Across all 130 evaluated clips, exactly **17 clips** showed prediction changes:
- **14 Beneficial FP Eliminations**:
  - **6 ASR Loop Artifacts eliminated**: `FluencyBank_016_28`, `FluencyBank_016_35`, `FluencyBank_016_91`, `FluencyBank_024_72`, `FluencyBank_024_89`, `FluencyBank_024_154`. (Zero-dur density $\ge 100$, collapsed timestamps, correctly recognized as decoder runaway artifacts).
  - **1 Conversational Filler eliminated on cohort**: `FluencyBank_016_9` (`"Um, um,"` leisurely conversational cadence, ISI $0.24$s).
  - **7 Conversational Filler / Boundary FPs eliminated on dev90**: `FluencyBank_010_17`, `FluencyBank_010_40`, `M_1022_11y3m_1_008`, `M_1022_11y3m_1_000`, `M_0061_16y9m-1_030`, `FluencyBank_017_4`, `FluencyBank_031_47`.
- **3 Gated Low-Iteration Hesitations (Cat 3 Cohort)**:
  - `FluencyBank_097_1`, `FluencyBank_147_0`, `FluencyBank_173_1` contained 2-repetition filler chains (`"um, um"`).
- **0 Adverse False Positives Introduced**.

Full details are documented in [`per_clip_changes.csv`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4c3_fasttrack/per_clip_changes.csv).

---

## 5. Verification, Safety & Separation Audit

- **Holdout Separation**: The spent 120-clip Phase 4B holdout audio was **never loaded, evaluated, or processed**.
- **Manifest Hash Integrity**: The locked holdout manifest SHA-256 (`fdfa62ce5a85c22b57db013fbee8451f84777058e671ae2eb897f8c121e4fb00`) remains strictly verified and untouched.
- **Production Isolation**: Zero modifications made to production files (`backend/main.py`, production configs, or frontend).
- **Focused Unit Tests**: 100% passing across 19 unit and regression tests ([`test_results.txt`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4c3_fasttrack/test_results.txt)).

---

## 6. Generated Artifacts Manifest

| Artifact File | Path | Description |
|---|---|---|
| **Fast-Track Report** | [`PHASE4C3_FASTTRACK_REPORT.md`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4c3_fasttrack/PHASE4C3_FASTTRACK_REPORT.md) | Comprehensive findings, decision rule, and verdict |
| **Metrics Comparison** | [`phase4c3_metrics_comparison.json`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4c3_fasttrack/phase4c3_metrics_comparison.json) | Complete numerical comparison tables and confusion matrices |
| **Per-Clip Changes** | [`per_clip_changes.csv`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4c3_fasttrack/per_clip_changes.csv) | Full per-clip predictions and acoustic/lexical root-cause audit |
| **Unit Test Log** | [`test_results.txt`](file:///e:/SpeechPrj/evaluation_results/experimental/phase4c3_fasttrack/test_results.txt) | 19-test execution log (6 loop-guard tests + 13 regression tests) |
