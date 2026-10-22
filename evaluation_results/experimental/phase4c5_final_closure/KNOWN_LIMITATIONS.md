# SpeechAI — Phase 4C.5 Known Technical and Methodological Limitations

**Status:** Frozen  
**Date:** 2026-10-22  
**Context:** Detector Improvement Cycle Closure (Phases 4A through 4C.5)

This document establishes the known technical, algorithmic, and methodological limitations of the experimental stutter detector and its ASR loop safeguard (`ASRLoopGuard`).

---

## 1. Algorithmic and Decoding Limitations

### 1.1 Whisper Base Repetition Loops on Low-Energy Audio
- **Phenomenon:** The upstream Whisper base model exhibits a known failure mode where low-energy trailing audio (breathing, faint background murmurs, or extended post-utterance silence) causes the autoregressive decoder to enter runaway repetition loops.
- **Observed Behavior:** The model outputs tens to hundreds of identical filler tokens (e.g., `"uh, uh, uh..."` or `"um, um, um..."`) with collapsed, identical, or zero timestamps ($0.00$s duration).
- **Mitigation & Boundary:** While `ASRLoopGuard` successfully detects these collapsed timestamps ($\text{zero\_dur} \ge 3$, $\text{identical\_starts} \ge 4$) and suppresses lexical repetition false positives, it is an output-filtering heuristic. It does not correct the underlying acoustic-to-phonetic alignment in the ASR decoder.

### 1.2 Two-Token Filler Gating Boundary
- **Phenomenon:** In the linguistic filler boundary gater, whole-word filler sequences require $\ge 3$ iterations (`min_filler_repetition_count = 3`) to be treated as physical struggle.
- **Observed Behavior:** Brief two-token filler repetitions (e.g., `"Um, um, raw butt."` in `FluencyBank_097_1`) are classified as conversational hesitations and gated out.
- **Impact on Recall:** In clinical corpora (such as FluencyBank), speech-language pathologists occasionally label 2-token filler interjections as stuttering when accompanied by observed tension or communicative difficulty. Because the lexical detector relies on word iteration counts without facial/visual tension cues, these 2-token events represent an irreducible lexical boundary between conversational hesitation and clinical dysfluency.

### 1.3 Co-Occurrence of Genuine Stutter and Degenerate ASR Loops
- **Phenomenon:** In clips such as `M_0030_16y4m_1_dysfluent_009`, a speaker exhibits genuine initial blocking or repetition, but the utterance audio subsequently decays into a trailing murmur that triggers an upstream Whisper loop.
- **Safeguard Behavior:** If the clip contains no independent acoustic prolongation $\ge 0.50$s or detected acoustic block, the safeguard suppresses the entire lexical repetition streak to prevent the hallucinated loop from generating false positives.
- **Limitation:** Genuine lexical stuttering that immediately precedes or co-occurs with an ASR decoder runaway loop may be suppressed unless corroborated by acoustic features.

### 1.4 Whisper Word-Boundary Timestamp Collapse
- **Phenomenon:** Whisper base word-level timestamp estimation (derived via cross-attention alignment) occasionally collapses adjacent token boundaries to the exact same millisecond timestamp (e.g., `FluencyBank_031_19` where both `' um,'` and `' definitely,'` share identical start and end times at $2.20$s).
- **Limitation:** When timestamps collapse at phrase boundaries, inter-segment interval (ISI) calculations and word duration metrics become artificially zero or near-zero, requiring minimum span floors ($0.05$s) to prevent false artifact rejection.

---

## 2. Evaluation and Dataset Limitations

### 2.1 The Spent 120-Clip Holdout
- **Status:** The 120-clip Phase 4B evaluation set was evaluated once in Phase 4B. Under strict methodological separation rules, **it is spent and cannot be re-used, tuned upon, or re-evaluated** to claim unbiased generalizability.
- **Preservation:** The locked manifest hash `fdfa62ce5a85c22b57db013fbee8451f84777058e671ae2eb897f8c121e4fb00` confirms that zero holdout audio or features were accessed in Phase 4C.

### 2.2 Exploratory Nature of the 40-Clip Dev-Hesitation Cohort
- **Cohort Purpose:** The 40-clip cohort (`DH_FLUE_01..10`, `DH_CONV_01..10`, `DH_GENU_01..10`, `DH_ASR__01..10`) was curated specifically to test hesitation boundaries and ASR loop failure modes during development.
- **Limitation:** Because this cohort was targeted at challenging edge cases, its metrics (e.g., $F_1 = 0.7000$) represent exploratory diagnostic behavior under stress conditions, **not natural prevalence or clinical population performance**.

### 2.3 Absence of Independent Out-of-Domain Clinical Holdout
- **Limitation:** All performance improvements observed in Phase 4C ($F_1 = 0.8119$ on Dev90, 80% reduction in loop FPs on the cohort) are **development-only metrics**.
- **Requirement:** These results do not constitute proof of clinical effectiveness or generalizability to unseen clinical populations. Production integration requires validation on an entirely new, unspent holdout set collected across independent clinical sites.

---

## 3. Deployment Constraints

1. **Experimental Decoupling:** The safeguard logic resides in `backend/experimental/asr_loop_guard.py` and is controlled by `enable_asr_loop_guard: bool = False` in `StutterDetectorConfig`. It must remain disabled by default.
2. **Production Code Isolation:** Production endpoints in `backend/main.py` and frontend UI components have **not** been modified and must not be updated without explicit, separate authorization following clinical validation.
