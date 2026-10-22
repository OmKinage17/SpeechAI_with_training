# Phase 4C.1: Read-Only ASR Repetition-Loop Error Analysis Report

**Project:** SpeechAI Experimental Benchmark Suite  
**Date:** October 22, 2026  
**Auditor / Investigator:** Antigravity (Advanced Agentic AI)  
**Status:** Analysis Complete — Read-Only Mode (No Detector Code or Thresholds Modified)  
**Scope:** Read-only inspection of the 90 development clips, existing Phase 4B holdout report (cited as historical failure evidence only), and existing experimental ASR/gating modules.  
**Output Location:** `evaluation_results/experimental/phase4c1_error_analysis/PHASE4C1_READ_ONLY_ERROR_ANALYSIS.md`

---

## 1. Scope, Integrity, and Non-Tuning Statement

This investigation was conducted strictly under read-only diagnostic governance:
- **No Holdout Re-Evaluation:** The locked 120-clip benchmark was **not re-run**, not processed, and its audio/features were not inspected. The locked manifest hash was re-verified and remains untouched at `fdfa62ce5a85c22b57db013fbee8451f84777058e671ae2eb897f8c121e4fb00`.
- **No Detector Modifications:** No thresholds were tuned, no heuristics were implemented, and no detector logic in `backend/experimental/` was altered.
- **No Production Impact:** Production backend, `backend/main.py`, the user interface, and live models remain completely untouched.
- **Strict Partition Separation:** All empirical inquiries, timestamp audits, token counts, and telemetry verifications were performed exclusively on the **90 development clips** (`dev_30` and `gen_60`). The spent holdout results documented in `PHASE4B_HOLDOUT_REPORT.md` are referenced solely as historical failure evidence to frame the research questions.

---

## 2. Failure Categorization from the Phase 4B Holdout Report

In Phase 4B.2, the Multi-Iteration Hesitation Gating candidate (Intervention 1: $N \ge 3$ consecutive filler repetitions with $\text{ISI} \le 0.35\,\text{s}$) recovered 3 true positives (+4.48% stutter recall), but introduced 5 false positives (-4.44% precision, -9.44% specificity).

Based on the documentation in [`PHASE4B_HOLDOUT_REPORT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/phase4b_holdout/PHASE4B_HOLDOUT_REPORT.md), the 5 holdout false-alarm failures fall into two distinct structural categories:

| Historical Failure Category | Mechanism | Documented Spent-Holdout Examples |
| :--- | :--- | :--- |
| **Category A: Degenerate ASR Token-Loop Hallucinations** | Whisper decoder generates 100+ identical filler tokens (`uh`, `um`) with collapsed identical timestamps ($\text{ISI} \le 0.015\,\text{s}$, span $\approx 0.0\text{--}1.4\,\text{s}$) during trailing silence or low-energy hesitation. | - `M_1099_25y0m_1_dysfluent_152.wav` (UCLASS, GT Filler): 107x `uh`, mean $\text{ISI}=0.005\,\text{s}$<br>- `FluencyBank_089_130.wav` (FB, GT Filler): 108x `uh`, mean $\text{ISI}=0.000\,\text{s}$<br>- `FluencyBank_089_30.wav` (FB, GT Filler): 102x `uh`, mean $\text{ISI}=0.000\,\text{s}$<br>- `FluencyBank_099_171.wav` (FB, GT Filler): 110x `um`, mean $\text{ISI}=0.013\,\text{s}$ |
| **Category B: Normal Conversational Hesitation Bursts** | Genuine human speakers produce multiple rapid deliberation/cognitive fillers ("um... um...") without physical block struggle. | - `M_1023_11y3m_1_fluent_014.wav` (UCLASS, GT Clean): 7x `um`, mean $\text{ISI}=0.29\,\text{s}$ in fluent child<br>- `FluencyBank_089_30.wav` (FB, GT Filler): 3x `um`, mean $\text{ISI}=0.18\,\text{s}$ during word search |

*Note: In accordance with data governance rules, these holdout clips are spent historical evidence. None of their audio was loaded, and none of their features were recomputed.*

---

## 3. Development-Only Dataset Composition and Available Evidence

The development dataset consists of **90 clips** combining the Standardized Development partition (`dev_30`, $N=30$) and the Generalization Validation partition (`gen_60`, $N=60$).

### 3.1 Corpus & Class Balance
- **Total Clips:** 90
- **Ground Truth Support:** 59 Stuttering (65.6%), 31 Non-Stuttering (16 Filler, 15 Clean).
- **Corpora:** UCLASS (40 clips), FluencyBank (30 clips), Libri_Stutter (20 clips).
- **Available Feature Fields:** Audio duration, Whisper raw transcript, word-level timestamps (`words_list` containing `word`, `start`, `end`, `probability`), silence intervals (`start`, `end`, `duration`), and acoustic energy/pitch features.

### 3.2 Global Filler Distribution in the 90 Development Clips
An exhaustive scan of `words_list` across all 90 development clips revealed:
- **Clips containing $\ge 1$ filler token (`um`, `uh`, `er`, `ah`, `oh`):** 25 clips (27.8%).
- **Clips containing $\ge 2$ consecutive repeated fillers:** 4 clips (`M_0030_009`, `M_0030_015`, `FluencyBank_017_4`, `FluencyBank_031_21`).
- **Clips containing $\ge 3$ consecutive repeated fillers:** **EXACTLY TWO CLIPS** (`M_0030_16y4m_1_dysfluent_009.wav` and `M_0030_16y4m_1_dysfluent_015.wav`).

---

## 4. Empirical Development-Only Findings

### Finding 1: ASR Token-Loop Hallucinations Exist Inside the Development Partition
In Phase 4B, clip **`M_0030_16y4m_1_dysfluent_009.wav`** was treated as a target recovery for Intervention 1. A detailed timestamp inspection of the development feature cache reveals:
- Total word count in transcript: **112 words**.
- Spoken prefix: `"You could, um, um, uh,"` ($t = 0.00\text{s}$ to $2.98\text{s}$).
- Runaway tail: **108 consecutive `"uh,"` tokens** spanning from word index 4 to 111.
- Timestamps: **Every single one of the 108 runaway tokens has `start: 2.98, end: 2.98`**.
- Articulatory span: $2.98 - 2.98 = \mathbf{0.00\,\text{s}}$. Mean $\text{ISI} = \mathbf{0.000\,\text{s}}$.
- Token probability: Individual token probabilities fluctuated wildly ($0.08 \to 0.95$).

**Empirical Conclusion:** The 108x `"uh"` sequence in `M_0030_009` is **100% an ASR decoder loop hallucination**, identical in structure to the holdout failures. Because `M_0030_009` had `gt_has_stutter = 1`, Phase 4B erroneously credited this artifact as a "true positive struggle recovery" (`SEVERE_FILLER_STRUGGLE_CHAIN`).

### Finding 2: Genuine Perseverative Struggle Has Real Acoustic & Temporal Extent
Contrasting `M_0030_16y4m_1_dysfluent_009.wav` against **`M_0030_16y4m_1_dysfluent_015.wav`** illustrates the difference between an ASR artifact and genuine clinical struggle:

| Structural Property | `M_0030_009` (ASR Loop Artifact) | `M_0030_015` (Genuine Articulatory Struggle) |
| :--- | :--- | :--- |
| **Token Sequence** | 108x `'uh'` | 7x `'er'` |
| **Total Transcript Length** | 112 words (runaway explosion) | 10 words (realistic length) |
| **Temporal Span** | $\mathbf{0.00\,\text{s}}$ ($t=2.98\text{s} \to 2.98\text{s}$) | $\mathbf{1.34\,\text{s}}$ ($t=0.00\text{s} \to 1.34\text{s}$) |
| **Individual Token Durations** | $0.00\,\text{s}$ (zero-duration collapsed tokens) | $0.08\,\text{s}, 0.24\,\text{s}, 0.02\,\text{s}, 0.02\,\text{s}\dots$ (finite) |
| **Mean Inter-Segment Interval** | $\mathbf{0.000\,\text{s}}$ | $\mathbf{0.223\,\text{s}}$ (physically plausible syllable rate) |
| **Acoustic Silences** | No silences during loop (all at 2.98s) | Measured inter-word pauses: $0.54\text{s}, 0.10\text{s}, 0.28\text{s}$ |
| **Subsequent Speech Context** | Trailing silence | Followed by lexical phrase: `"Once, oh quick."` |

### Finding 3: Existing Baseline Loop Sanitizer Successfully Detects the Dev Artifact
Running the existing baseline `detect_and_sanitize_repetition_loop` (from [`asr_engine.py`](file:///E:/SpeechPrj/backend/experimental/asr_engine.py)) on `M_0030_009` yields:
- `is_loop`: **`True`**
- `status`: **`"DEGENERATE_LOOP_RECOVERED"`**
- `reason`: `"Degenerate hesitation stream: 6+ consecutive filler tokens. Truncated 107 runaway tokens while preserving initial hesitations."`
- `sanitized_transcript`: `"You could, um, um, uh,"` (truncated from 112 words down to 5 words).

**The Architectural Disconnect:**
In Phase 4B development evaluation and holdout evaluation, the evaluation runners bypassed `detect_and_sanitize_repetition_loop` and passed raw cached Whisper transcripts directly into `HybridStutterDetector`. Furthermore, `detect_and_sanitize_repetition_loop` intentionally preserves 3 initial hesitation tokens in the sanitized transcript so as not to erase genuine dysfluency evidence. However, Intervention 1's gating rule treated $\ge 3$ filler repetitions as sufficient to trigger stuttering, creating a circular vulnerability.

### Finding 4: Multi-Gram Loops Can Bypass 1-Gram/2-Gram Sanitizers
Inspection of clip **`FluencyBank_031_21.wav`** in the development partition revealed a 149-word transcript repeating the 6-gram phrase:
`"I'm just going to say, um, um, I'm just going to say, um, I'm just going to say, um..."` (25 cycles).
- 148 of the 149 tokens are collapsed at `start: 2.98, end: 2.98`.
- Running the baseline `detect_and_sanitize_repetition_loop` on this clip returned `is_loop: False`.
- **Reason:** The baseline sanitizer only checks single-token runs and alternating 2-grams ($n=1, 2$); it does not evaluate higher-order $n$-grams ($n \ge 3$) or timestamp collapse density.

---

## 5. Summary Table of Relevant Development Clips

The following table presents all clips in the 90-clip development set exhibiting repetitive filler structures, timestamp collapses, or loop telemetry:

| Clip ID | Dataset & Split | GT Stutter | GT Dominant | Total Words | Filler Tokens & Streaks | Zero-Dur Tokens | Baseline Sanitizer Status | Phase 4B Baseline Pred | Phase 4B Candidate Pred | Finding / Diagnostic Classification |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **`M_0030_16y4m_1_dysfluent_009.wav`** | UCLASS (`dev_30`) | **1** | Stutter | 112 | 2x `um` ($1.04\text{--}2.98\text{s}$)<br>108x `uh` ($2.98\text{--}2.98\text{s}$) | 108 | `DEGENERATE_LOOP_RECOVERED` (Truncated 107) | 0 (Clean) | 1 (Stutter, 0.85) | **ASR 1-gram loop artifact**; erroneously rewarded as TP in Phase 4B. |
| **`M_0030_16y4m_1_dysfluent_015.wav`** | UCLASS (`dev_30`) | **1** | Stutter | 10 | 7x `er` ($0.00\text{--}1.34\text{s}$, $\text{ISI}=0.22\text{s}$) | 2 | `SUCCESS` (No loop) | 0 (Filler) | 1 (Stutter, 0.76) | **Genuine perseverative struggle**; physical duration, realistic pacing. |
| **`FluencyBank_017_4.wav`** | FluencyBank (`gen_60`) | **0** | Filler | 6 | 1x `uh`, then 2x `uh` ($2.36\text{--}2.98\text{s}$) | 2 | `SUCCESS` (No loop) | 0 (Filler) | 0 (Filler) | **Conversational filler**; streak count = 2 ($< 3$), safely ignored. |
| **`FluencyBank_031_21.wav`** | FluencyBank (`gen_60`) | **1** | Stutter | 149 | 25x `um` inside 25x repeating 6-gram | 148 | `SUCCESS` (Bypassed 1g/2g check) | 1 (Stutter) | 1 (Stutter) | **ASR 6-gram loop artifact**; baseline caught via acoustic candidate; candidate unchanged. |
| **`FluencyBank_031_19.wav`** | FluencyBank (`gen_60`) | **1** | Stutter | 6 | 2x `um` separated by `"definitely"` | 2 | `SUCCESS` (No loop) | 1 (Stutter) | 1 (Stutter) | Interrupted filler sequence; consecutive streak = 1. |
| **`103-1240-0002.flac`** | Libri_Stutter (`dev_30`) | **1** | Stutter | 56 | 0 fillers; 3-cycle 2-gram lexical loop | 1 | `DEGENERATE_LOOP_RECOVERED` (Truncated 11) | 1 (Stutter) | 1 (Stutter) | Lexical 2-gram loop (`sharp eye`); baseline sanitizer functions correctly. |
| **`103-1240-0006.flac`** | Libri_Stutter (`dev_30`) | **1** | Stutter | 35 | 0 fillers; 3-cycle 2-gram lexical loop | 0 | `DEGENERATE_LOOP_RECOVERED` (Truncated 31) | 1 (Stutter) | 1 (Stutter) | Lexical 2-gram loop (`as heavenly`); baseline sanitizer functions correctly. |

---

## 6. Analysis of Potential Artifact Indicators and Risks

To distinguish ASR repetition loops, conversational fillers, and genuine struggle, five potential indicators were analyzed conceptually against the development evidence:

### Indicator 1: Minimum Physical Temporal Span & Inter-Segment Interval Floor
- **Mechanism:** Reject any multi-token repetition where the temporal span between first and last token is below a physical articulatory threshold (e.g., span $< 0.15\,\text{s}$ or mean $\text{ISI} < 0.08\,\text{s}$).
- **Evidence from Dev Data:**
  - `M_0030_009` (Loop Artifact): Span $= 0.00\,\text{s}$, $\text{ISI} = 0.000\,\text{s}$ $\implies$ **Correctly rejected as artifact**.
  - `M_0030_015` (Genuine Struggle): Span $= 1.34\,\text{s}$, $\text{ISI} = 0.223\,\text{s}$ $\implies$ **Safely preserved**.
- **Known Risk:** If a speaker produces a rapid, tense clonic repetition (e.g. 3 iterations of /b-b-b/), syllable rate can reach 6–7 Hz ($\text{ISI} \approx 0.14\text{--}0.16\,\text{s}$). An overly aggressive ISI floor (e.g., $> 0.15\,\text{s}$) could suppress rapid real clonic repetitions. However, an ISI floor of $\approx 0.05\text{--}0.08\,\text{s}$ safely eliminates instantaneous ASR collapses ($\text{ISI} \le 0.02\,\text{s}$) without threatening physical human speech.

### Indicator 2: Zero-Duration Token Density Check
- **Mechanism:** Reject repetition chains where multiple consecutive tokens have identical timestamps (`start == end` or consecutive tokens share the same `start`).
- **Evidence from Dev Data:**
  - In `M_0030_009`, 108 tokens have identical start/end ($2.98\text{s}$).
  - In `FluencyBank_031_21`, 148 tokens have identical start/end ($2.98\text{s}$).
  - In genuine development disfluencies (`M_0030_015`), only 2 boundary tokens shared a timestamp, but the sequence progressed over 1.34s.
- **Known Risk:** Whisper occasionally assigns identical timestamps to two short adjacent syllables (e.g. at audio clip boundaries). Requiring $\ge 3$ consecutive identical timestamps prevents penalizing minor timestamp rounding errors.

### Indicator 3: Upstream Repetition-Loop Sanitizer Gating (`DEGENERATE_LOOP_RECOVERED`)
- **Mechanism:** If the upstream ASR engine marks a clip as `DEGENERATE_LOOP_RECOVERED`, multi-iteration filler gating should be strictly disabled for that clip.
- **Evidence from Dev Data:**
  - `M_0030_009` is cleanly caught by the baseline sanitizer with status `DEGENERATE_LOOP_RECOVERED`.
  - Passing the sanitized transcript (`"You could, um, um, uh,"`) eliminates the 108-token explosion.
- **Known Risk:** As demonstrated by `FluencyBank_031_21`, higher-order $n$-gram loops ($n \ge 3$) are not detected by the current 1-gram/2-gram sanitizer and would slip past this check unless the sanitizer is generalized or accompanied by timestamp checks.

### Indicator 4: Acoustic Tension & Energy Corroboration
- **Mechanism:** Do not allow text-only filler repetition to trigger stuttering without acoustic corroboration (e.g., verifying an energy increase, pitch rise, or glottalization across the hesitation).
- **Evidence from Dev Data:**
  - In `M_0030_009`, the acoustic candidate was a prolongation during the early part of the clip ($1.56\text{--}2.15\text{s}$), completely decoupled from the trailing ASR loop at $2.98\text{s}$.
  - Conversational hesitations (like `FluencyBank_017_4`) have low, declining pitch and relaxed vocal tract acoustics.
- **Known Risk:** Acoustic feature extraction can be noisy in conversational speech or low-SNR environments; requiring rigid acoustic thresholds could lower recall on quiet perseverations.

---

## 7. Evidence Gaps and Limitations of the Development Set

The most critical finding of Phase 4C.1 is a fundamental **dataset sample size limitation**:

1. **Extreme Small-Sample Size for Repetitive Fillers:**
   In the entire 90-clip development set, there are **only 2 clips** exhibiting $\ge 3$ repeated fillers:
   - `M_0030_009`: ASR loop artifact (Ground Truth: Stutter).
   - `M_0030_015`: Genuine struggle (Ground Truth: Stutter).
2. **Zero Non-Stutter Multi-Filler Examples:**
   There are **exactly 0 clips** in the 90 development clips where a fluent or filler-dominant speaker produced $\ge 3$ consecutive repeated fillers (e.g., "um um um").
   - `FluencyBank_017_4` had only 2 consecutive fillers at the end.
   - `FluencyBank_031_19` had fillers separated by lexical words.
3. **Overfitting Illusion:**
   Because the development set contained $2$ stutter clips and $0$ non-stutter clips with $\ge 3$ fillers, any rule that triggered on $\ge 3$ fillers achieved an apparent "100% precision" during development, blinding Phase 4B to the false-alarm danger on conversational speech.
4. **Conclusion on Development Sufficiency:**
   The 90-clip development set is **insufficient** on its own to validate a boundary between normal conversational hesitation bursts and genuine struggle chains. While ASR loop artifacts can be modeled and eliminated using physical timing constraints, evaluating whether ordinary $3\times$ fillers are benign or dysfluent requires additional development audio containing conversational deliberation chains.

---

## 8. Prioritized Recommendation for Next Experiment (Phase 4C.2)

### Recommended Experiment: "Physical Articulation Floor & Loop Sanitization Integration"

Before any further holdout evaluation can even be contemplated, a strictly controlled development experiment (Phase 4C.2) should be designed around two tightly scoped interventions:

1. **Intervention A: Physical Articulation Span & ISI Floor Gating**
   - Require that any candidate multi-iteration filler struggle chain must satisfy a minimum physical duration:
     - Total span: $\text{span} \ge 0.20\,\text{s}$.
     - Minimum mean ISI: $\text{mean\_isi} \ge 0.06\,\text{s}$ (rejecting sub-millisecond timestamp collapses).
     - Maximum consecutive zero-duration tokens: $< 3$.
   - **Hypothesis:** This cleanly eliminates ASR loop artifacts (`M_0030_009` and holdout Category A failures) without suppressing genuine physical struggle (`M_0030_015`).

2. **Intervention B: Telemetry Binding to Upstream Sanitizer**
   - Connect `asr_engine.py` loop telemetry directly into `HybridStutterDetector`. If `status == "DEGENERATE_LOOP_RECOVERED"`, suppress un-gating of filler repetitions in that segment.

3. **Requirement for Development Partition Expansion:**
   - Because the 90 dev clips contain 0 non-stutter multi-filler examples, Phase 4C.2 must source or annotate additional development clips (from available non-holdout UCLASS/FluencyBank sessions) containing fluent deliberation repetitions before testing conversational gating thresholds.

---

## 9. Final Integrity Audit

- **Locked Manifest Hash:** Recalculated SHA-256 is `fdfa62ce5a85c22b57db013fbee8451f84777058e671ae2eb897f8c121e4fb00` (**Verified Unchanged**).
- **Holdout Immutability:** The locked 120-clip benchmark was **not executed** and its files were not accessed or modified.
- **Prior Phase Artifacts:** All artifacts in `phase2_corrected/`, `phase3_improvement/`, `phase4a_error_analysis/`, `phase4b_filler_gating/`, and `phase4b_holdout/` remain pristine and unmodified.
- **Production Files:** `backend/main.py`, production configs, live models, and frontend were **not modified**.

---

## 10. Audit Verdict

### **`READY FOR PHASE 4C.2 EXPERIMENT DESIGN`**

**Summary of Verdict:**
The root causes of Phase 4B's holdout degradation have been definitively isolated on development data:
1. One of the two target recoveries in Phase 4B (`M_0030_009`) was actually an ASR token-loop artifact with 108 zero-duration tokens at $t=2.98\text{s}$, which the candidate misidentified as stuttering.
2. The 90-clip development partition suffered from severe sample scarcity ($N=2$ multi-filler clips, $0$ non-stutter multi-filler clips).
3. A principled physical articulation floor ($\text{span} \ge 0.20\,\text{s}$, $\text{mean\_isi} \ge 0.06\,\text{s}$) and integration of existing ASR loop telemetry provide a clear, testable development hypothesis for Phase 4C.2 without modifying production or touching the locked holdout.
