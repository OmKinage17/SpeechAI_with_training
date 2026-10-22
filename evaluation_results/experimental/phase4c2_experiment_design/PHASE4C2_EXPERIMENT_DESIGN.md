# Phase 4C.2: ASR Loop Guard Experiment Design

**Project:** SpeechAI Experimental Benchmark Suite  
**Date:** October 22, 2026  
**Status:** Pre-Registered Experiment Design (Design-Only Phase — No Code Modified)  
**Output Location:** `evaluation_results/experimental/phase4c2_experiment_design/PHASE4C2_EXPERIMENT_DESIGN.md`  
**Governing Objective:** Design a controlled development experiment to eliminate false-positive stutter classifications caused by ASR repetition loops while preserving true-positive detection of genuine articulatory struggle.

---

## 1. Scope and Integrity Governance

This design document adheres strictly to the project's pre-execution integrity standards:
- **Design-Only Scope:** No detector code, thresholds, model architectures, or production files (`backend/main.py`, production configs, frontend) are modified.
- **Holdout Immutability:** The locked 120-clip final benchmark remains untouched. No audio from the spent Phase 4B holdout was loaded, processed, or re-evaluated. The locked manifest hash is pre-registered and verified:
  `fdfa62ce5a85c22b57db013fbee8451f84777058e671ae2eb897f8c121e4fb00`
- **Data Governance:** No holdout labels, clips, or outcomes were used to select numerical thresholds or fit models. Historical holdout failure categories from Phase 4B are cited solely as qualitative problem definitions.
- **Prior Phase Protection:** All artifacts in Phase 2, Phase 3, Phase 4A, Phase 4B dev, and Phase 4B holdout remain unchanged.
- **No Active Audio Collection in Phase 4C.2:** This document details the protocol for dataset expansion; no audio is downloaded, processed, or annotated during this design-only phase.

---

## 2. Architectural Analysis of Current Code Paths & Telemetry

To design non-invasive guards, the current data flow from ASR generation to stutter classification was audited:

```
[Audio Input]
      │
      ▼
[ExperimentalASREngine] ──> Calls Whisper & word-level timestamping
      │
      ▼
[detect_and_sanitize_repetition_loop] (asr_engine.py)
      │
      ├─► Generates Telemetry: {is_loop: bool, status: "DEGENERATE_LOOP_RECOVERED", ...}
      └─► Truncates Runaway Tail, but intentionally preserves initial 3 tokens
      │
      ▼
[HybridStutterDetector.evaluate] (stutter_detector.py)
      │  * Current Signature: evaluate(transcript, words_list, audio_path, audio_data, pred_has_filler)
      │  * BUG: Loop telemetry (is_loop, status) is DROPPED; only raw/cached text is passed!
      │
      ▼
[FillerBoundaryGater.gate_details] (filler_boundary_gating.py)
      │
      ▼
[_extract_filler_streaks]
      │  * Evaluates filler streaks: count >= 3, mean_isi <= 0.35s
      │  * Vulnerability 1: count >= 5 unconditionally sets is_rapid=True, confidence=0.70-0.85
      │  * Vulnerability 2: Zero-duration collapsed tokens (ISI = 0.0s) satisfy mean_isi <= 0.35s!
      │
      ▼
[multi_iteration_filler_struggle Event] ──> Emitted with confidence >= 0.60
      │
      ▼
[Stutter Classification] ──> has_stutter = 1 (FALSE POSITIVE on ASR loops)
```

### Primary Structural Deficiencies Identified
1. **Telemetry Severance:** `HybridStutterDetector.evaluate()` does not accept or propagate `asr_status` or loop metadata. Upstream loop detection is discarded at the detector interface.
2. **Circular Interaction with Sanitizer Design:** `detect_and_sanitize_repetition_loop()` intentionally keeps 3 initial hesitation tokens to avoid erasing real disfluency evidence. However, Intervention 1's minimum count threshold was set to $N=3$, meaning sanitized loop remnants still satisfy the stuttering trigger.
3. **Absence of Articulatory Feasibility Checks:** In `_extract_filler_streaks()`, `is_rapid` is evaluated as `mean_isi <= 0.35s`. When timestamps collapse to zero duration ($t_{start} = t_{end} = 2.98\,\text{s}$), the computed $\text{ISI} = 0.000\,\text{s}$ mathematically satisfies $\le 0.35\,\text{s}$, falsely classifying a zero-duration decoder hallucination as "rapid human struggle."

---

## 3. Specification of Three Isolated Guard Variants

To evaluate solutions rigorously, three distinct experimental variants are designed. Each variant isolates a specific hypothesis and will be ablated against the authoritative Phase 2 baseline and the Phase 4B candidate.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │               Variant Comparison Scheme                 │
                  └─────────────────────────────────────────────────────────┘
                                               │
             ┌─────────────────────────────────┼─────────────────────────────────┐
             ▼                                 ▼                                 ▼
   [Variant A: Telemetry]             [Variant B: Quality]              [Variant C: Combined]
  • Propagate ASR sanitizer         • Evidence-quality checks         • Upstream telemetry priority
    telemetry to gater                inside filler gater             • Timestamp fallback
  • Suppress multi-filler           • Temporal span floor             • Independent acoustic
    stutter when loop flagged       • Reject collapsed timestamps       corroboration bypass
```

---

### 3.1 Variant A — Existing Loop-Recovery Telemetry Guard

#### Conceptual Hypothesis
The upstream ASR engine already possesses a specialized loop sanitizer (`detect_and_sanitize_repetition_loop`). If its diagnostic status (`DEGENERATE_LOOP_RECOVERED`) is wired directly to the gating logic, any segment identified as a runaway loop will be barred from asserting a stutter detection based on filler repetition.

#### Detailed Specification
1. **Interface Modification:**
   - Extend `HybridStutterDetector.evaluate()` to accept an optional dictionary parameter:
     `asr_telemetry: Optional[Dict[str, Any]] = None`
   - Pass `asr_telemetry` to `FillerBoundaryGater.gate_details()`.
2. **Gating Logic:**
   - In `FillerBoundaryGater._extract_filler_streaks()`:
     - If `asr_telemetry` is present and `asr_telemetry.get("status") == "DEGENERATE_LOOP_RECOVERED"`:
       - Any filler streak overlapping the reported loop region (or the whole clip if global) is tagged with `is_loop_tainted = True`.
       - Tainted streaks are downgraded: `is_struggle = False`, `confidence = 0.30`, `classification = "LOOP_TAINTED_HESITATION"`.
       - The streak is gated into `gated_events` rather than promoted to `multi_iteration_filler_struggle`.
3. **Preservation of Independent Stutter Evidence:**
   - If the clip contains genuine non-filler stuttering (e.g. an acoustic prolongation candidate with duration $\ge 0.50\,\text{s}$, an acoustic block silence $\ge 0.25\,\text{s}$ followed by energy burst, or a non-filler lexical repetition like "did did did"), those events are processed normally and can independently classify the clip as `has_stutter = 1`.
   - Only the multi-iteration filler mechanism is muted by the telemetry flag.
4. **False-Suppression Risks & Failure Modes:**
   - **ASR Sanitizer Blind Spots:** The existing sanitizer only inspects 1-grams and alternating 2-grams. As proven in Phase 4C.1, higher-order loops (e.g. the 6-gram loop in `FluencyBank_031_21`) return `status: "SUCCESS"` and would slip past Variant A completely.
   - **False Loop Flagging:** If the sanitizer incorrectly tags a genuine, highly perseverative clinical struggle as a degenerate loop, Variant A would suppress valid stuttering evidence unless independent acoustic candidates rescue it.

---

### 3.2 Variant B — Timestamp-Quality Guard

#### Conceptual Hypothesis
ASR repetition loops exhibit distinct non-physical temporal signatures: zero-duration tokens, identical sub-millisecond timestamps, and instantaneous inter-segment intervals. By enforcing timestamp evidence-quality checks within the gater, artifacts can be filtered locally without depending on upstream ASR flags.

#### Detailed Specification
1. **Interface:** Requires no changes to `HybridStutterDetector.evaluate()` signature; operates entirely on `words_list` timing metadata.
2. **Quality Checks in `_extract_filler_streaks()`:**
   - **Check B1: Collapsed Timestamp Density:**
     - For any consecutive streak of filler tokens $S = [w_1, w_2, \dots, w_n]$:
     - Count zero-duration tokens: $Z = \sum_{i=1}^n \mathbb{I}(w_i.\text{start} == w_i.\text{end})$.
     - Count identical timestamp occurrences: $M = \max_{\tau} \sum_{i=1}^n \mathbb{I}(w_i.\text{start} == \tau)$.
     - If $Z \ge 3$ or $M \ge 3$: The streak is classified as `DEGENERATE_TIMESTAMP_COLLAPSE`. It cannot trigger struggle, regardless of token count.
   - **Check B2: Physical Articulation Span Consistency:**
     - Compute total articulatory span: $\Delta T = w_n.\text{end} - w_1.\text{start}$.
     - Compute mean inter-segment interval: $\text{ISI}_{mean} = \frac{\Delta T}{n - 1}$.
     - **Pre-registered constraint:** Do not hardcode universal physiological constants. Check consistency: if $\Delta T < \epsilon_{span}$ or $\text{ISI}_{mean} < \epsilon_{isi}$, the tokens are deemed non-physical hallucinations.
     - *(Note: Specific values for $\epsilon_{span}$ and $\epsilon_{isi}$ will be ablated on dedicated calibration data; Phase 4C.1 values of $0.20\,\text{s}$ and $0.06\,\text{s}$ are treated strictly as unvalidated research hypotheses).*
   - **Check B3: Missing / Synthetic Timestamp Fallback:**
     - If all tokens have `start == 0.0, end == 0.0` (unaligned transcript), timestamp quality is tagged `TIMESTAMP_UNAVAILABLE`.
     - In this state, pacing cannot be proven rapid: `is_rapid` defaults to `False`, preventing unverified un-gating.
3. **Preservation of Genuine Articulatory Dysfluencies:**
   - Real blocks and prolongations are characterized by continuous physical duration ($\ge 0.50\,\text{s}$) or acoustic silence intervals ($0.25\text{--}0.60\,\text{s}$) followed by energy bursts. Because Check B only gates repetitive text tokens whose timestamps have collapsed, genuine physical dysfluencies (`M_0030_015`) are completely unaffected.
4. **False-Suppression Risks:**
   - Whisper sometimes assigns coarse timestamps to very brief syllable bursts. If a true clonic repetition occurs at a clip boundary where Whisper rounds all three bursts to the same second, Check B1 might discard it.

---

### 3.3 Variant C — Combined Evidence Guard

#### Conceptual Hypothesis
Maximum robustness is achieved through multi-layered defense: prioritizing explicit upstream ASR telemetry when present, falling back to local timestamp-quality validation when telemetry is absent or incomplete, and maintaining an acoustic corroboration bypass for high-severity dysfluencies.

#### Detailed Specification & Decision Hierarchy
For every candidate multi-iteration hesitation chain, apply the following strict precedence rules:

```
[Candidate Filler Streak Extracted]
                 │
                 ▼
    [1. Check ASR Telemetry]
       Is status == "DEGENERATE_LOOP_RECOVERED"?
         ├── YES ──► Tag as ASR_LOOP_ARTIFACT (Suppress Filler Stutter)
         └── NO  ──► Proceed to Check 2
                 │
                 ▼
    [2. Check Timestamp Evidence Quality]
       Does streak have >= 3 collapsed/zero-dur tokens OR span < eps_span?
         ├── YES ──► Tag as TIMESTAMP_COLLAPSE_ARTIFACT (Suppress Filler Stutter)
         └── NO  ──► Proceed to Check 3
                 │
                 ▼
    [3. Check Acoustic Corroboration Bypass]
       Does clip contain independent acoustic prolongation (dur >= 0.50s, conf >= 0.60)
       OR acoustic block silence with onset burst outside hesitation window?
         ├── YES ──► PRESERVE STUTTER (Corroborated by physical acoustic posture)
         └── NO  ──► Evaluate standard multi-iteration hesitation thresholds
```

1. **Precedence:** Upstream telemetry (Variant A) triggers immediate suppression. If upstream telemetry missed the loop (e.g. higher-order $n$-gram), timestamp checks (Variant B) catch the hallucination.
2. **Corroboration Bypass:** If a speaker genuinely stuttered (e.g., an acoustic prolongation) but Whisper also generated a trailing repetition loop at the end of the file, the clip's overall classification will NOT be suppressed because the independent acoustic detector confirms stuttering.
3. **Graceful Fallback:** When processing transcripts from external sources or cached benchmarks lacking ASR telemetry, the module defaults seamlessly to Variant B without throwing exceptions or changing behavior.

---

## 4. Dataset and Annotation Plan for Development Expansion

### 4.1 Root Cause of Current Evaluation Impasse
Phase 4C.1 proved that the current 90 development clips contain:
- Exactly 2 clips with $\ge 3$ repeated fillers (both GT Stutter: 1 ASR artifact, 1 real struggle).
- **Zero clips** with $\ge 3$ repeated fillers from fluent or non-stutter speakers.
- Testing any loop guard on the current 90 clips alone is statistically meaningless ($N=2$ cases) and cannot evaluate false-positive resistance on conversational speech.

### 4.2 Expansion Strategy: The "Dev-Hesitation" Cohort ($N=40$ Target)
To establish a statistically defensible development benchmark, a dedicated development cohort of **40 additional clips** must be sourced from eligible, unspent sessions:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                  Target Dev-Hesitation Composition (N=40)                  │
├──────────────────────────────────┬─────────────────────────────────────────┤
│ 1. Fluent Deliberation Controls  │ 10 clips from confirmed fluent speakers │
│    (Category B Controls)         │ with audible 3x/4x hesitations          │
├──────────────────────────────────┼─────────────────────────────────────────┤
│ 2. Conversational Non-Stutter    │ 10 clips from adult interviewees        │
│    Filler Chains                 │ with multiple deliberation markers      │
├──────────────────────────────────┼─────────────────────────────────────────┤
│ 3. Verified Genuine Stuttering   │ 10 clips containing physical persever-  │
│    Struggle Chains               │ ation with audible laryngeal tension    │
├──────────────────────────────────┼─────────────────────────────────────────┤
│ 4. Verified ASR Decoder Loop     │ 10 clips where Whisper produces token   │
│    Hallucinations (Category A)   │ loops during low-energy trailing pauses │
└──────────────────────────────────┴─────────────────────────────────────────┘
```

### 4.3 Data Sourcing & Partition Integrity Rules
1. **Strict Holdout Disjointness:**
   - Under no circumstances will any audio, speaker, or session from the locked 120-clip final benchmark (`run_manifest.json`, hash `fdfa62ce...`) be used.
   - Sourcing will be restricted to unspent recording sessions in UCLASS (e.g. speakers `M_0030`, `F_0012` non-holdout takes) and FluencyBank (e.g. sessions `010`, `017`, `031` non-holdout clips).
2. **Speaker-Level Separation:**
   - Clips selected for fluent controls must come from distinct speaker sessions not present in the dysfluent target evaluation set.
3. **Licensing Compliance:**
   - UCLASS is utilized under the UCL speech research agreement. FluencyBank is utilized under the TalkBank academic research license for non-commercial educational/scientific speech analysis.

### 4.4 Standardized Manual Annotation & Adjudication Protocol
Each candidate clip in the Dev-Hesitation cohort must undergo structured two-pass verification:

| Step | Verification Task | Objective & Standards |
| :---: | :--- | :--- |
| **Pass 1** | **Auditory Verification** | Listening to audio to verify: Is the repetitive sound audible in the acoustic wave, or is it a complete hallucination? If audible, is there physical struggle (pitch breaks, vocal tension) or relaxed cognitive pause? |
| **Pass 2** | **Timestamp Alignment** | Inspecting Whisper word timestamps against the acoustic spectrogram to determine whether tokens correspond to discrete acoustic bursts or collapsed artifacts. |
| **Adjudication** | **Metadata Recording** | Every clip must record: `corpus`, `speaker_id`, `session_id`, `ground_truth_label` (Clean/Filler/Stutter), `annotated_hesitation_type` (`AUDIBLE_STRUGGLE`, `CONVERSATIONAL_DELIBERATION`, `ASR_HALLUCINATION`), `raw_transcript`, `sanitized_transcript`, `timestamps`, `sanitizer_status`, and `adjudication_confidence` ($1.0 = \text{certain}, 0.5 = \text{ambiguous}$). |

*Constraint: Unannotated or ambiguous clips will not be treated as confirmed negatives. Only clips with adjudication confidence $\ge 0.90$ will be admitted.*

---

## 5. Evaluation Design & Comparative Protocol

### 5.1 Paired Comparison Matrix
The evaluation must be conducted simultaneously across all models on the identical expanded development dataset:
1. **Phase 2 Authoritative Baseline** (authoritative reference).
2. **Phase 4B Development Candidate** (Intervention 1 without guard).
3. **Phase 4C Candidate — Variant A** (Telemetry Guard).
4. **Phase 4C Candidate — Variant B** (Timestamp-Quality Guard).
5. **Phase 4C Candidate — Variant C** (Combined Guard).

### 5.2 Mandatory Performance Metrics

| Evaluation Dimension | Metrics to Report | Success Threshold / Target |
| :--- | :--- | :--- |
| **Binary Stuttering** | TP, FP, FN, TN, Precision, Recall, F1, Specificity, Accuracy | Specificity must not degrade relative to Phase 2 baseline. |
| **Multiclass 3-Class** | Clean, Stutter, Filler Precision, Recall, F1; Overall Accuracy; Macro-F1 | Macro-F1 must increase or match baseline; Filler recall must not collapse. |
| **Filler Event-Level** | Event TP, FP, FN, TN, Event F1, Lost Filler Detections, New Event FPs | Lost filler detections must remain exactly 0. |
| **Artifact Rejection Rate** | Percentage of verified ASR repetition loops correctly neutralized | $\ge 90.0\%$ rejection on Category A loops. |
| **False-Alarm Rate** | Number of false-positive stutter detections on confirmed fluent/filler clips | Exactly 0 new false positives on fluent controls. |
| **Struggle Preservation Rate** | Percentage of genuine physical struggle chains retained as stutter | $100\%$ retention of verified struggle (`M_0030_015`). |
| **Statistical Testing** | Exact McNemar test on paired binary stutter predictions | Required if discordant pairs $\ge 10$; report $\chi^2$ and two-sided $p$. |

*Requirement: All metrics must be computed strictly from saved per-clip prediction records. Confusion matrices must reconcile 100% to the total evaluated clip count.*

---

## 6. Pre-Registered Decision & Acceptance Criteria

To prevent confirmation bias, post-hoc metric rationalization, or threshold gaming, the following criteria are pre-registered:

### Gate 1: Non-Negotiable False-Alarm Rejection
- A variant is **IMMEDIATELY REJECTED** if it introduces even a single new false-positive stutter classification on confirmed fluent controls.
- On the confirmed conversational deliberation cohort, stutter false-alarm rate must be reduced by at least $80\%$ compared to the ungated Phase 4B candidate.

### Gate 2: True-Positive Struggle Preservation
- A variant is **IMMEDIATELY REJECTED** if it suppresses verified genuine clinical struggle (e.g. `M_0030_015` or equivalent physical perseverations).

### Gate 3: Macro-F1 Non-Degradation
- Overall multiclass Macro-F1 across the development partition must not decline relative to the Phase 2 baseline ($\Delta \text{Macro-F1} \ge 0.000$).
- Filler class recall must not suffer cannibalization ($\Delta \text{Filler Recall} \ge -0.020$).

### Gate 4: Sample Size Sufficiency for Threshold Approval
- **Numerical timing thresholds (e.g., minimum span, minimum ISI) CANNOT be finalized or declared validated without testing on at least 10 positive struggle chains and 10 negative conversational hesitation chains.**
- No claims of statistical significance will be made on sample sizes with fewer than 10 discordant pairs.

---

## 7. Risks, Failure Modes, and Fallback Strategy

| Identified Risk | Impact | Pre-Planned Mitigation / Fallback |
| :--- | :--- | :--- |
| **ASR Decoder Temperature Variance** | Whisper temperature fallback might alter token counts or timestamps between runs. | Enforce deterministic decoding in `ASRConfig` (`temperature = (0.0,)`, greedy decoding) during benchmark feature extraction. |
| **Acoustic Clonic Repetition Overlap** | High-speed clonic syllable repetitions (/p-p-p/) might approach the ISI floor. | Anchor timing floors strictly below human physiological minimums ($\le 0.06\,\text{s}$). If ambiguous, allow acoustic burst energy to corroborate disfluency. |
| **Missing Timing Metadata** | Transcripts generated without word-level timestamps (`word_timestamps=False`). | Graceful fallback in Variant C: if timestamps are absent, treat timestamp quality as `UNAVAILABLE` and rely on upstream sanitizer telemetry; do not assume rapid human struggle. |
| **Data Scarcity in Sourcing** | Inability to locate 10 non-holdout fluent deliberation chains. | If fewer than 10 confirmed non-stutter multi-filler clips are obtained, the evaluation must be declared exploratory and cannot approve production promotion. |

---

## 8. Exact Files Inspected and Created

### Files Inspected (Read-Only)
- [`backend/experimental/asr_engine.py`](file:///E:/SpeechPrj/backend/experimental/asr_engine.py) (lines 20–155: loop detection & sanitization logic)
- [`backend/experimental/filler_boundary_gating.py`](file:///E:/SpeechPrj/backend/experimental/filler_boundary_gating.py) (lines 44–300: `FillerBoundaryGater` & streak extraction)
- [`backend/experimental/stutter_detector.py`](file:///E:/SpeechPrj/backend/experimental/stutter_detector.py) (lines 270–375: `HybridStutterDetector.evaluate`)
- [`backend/experimental/config.py`](file:///E:/SpeechPrj/backend/experimental/config.py) (detector and ASR configuration dataclasses)
- [`evaluation_results/experimental/phase4c1_error_analysis/PHASE4C1_READ_ONLY_ERROR_ANALYSIS.md`](file:///E:/SpeechPrj/evaluation_results/experimental/phase4c1_error_analysis/PHASE4C1_READ_ONLY_ERROR_ANALYSIS.md)
- [`evaluation_results/experimental/final_benchmark/run_final_20261020_complete/run_manifest.json`](file:///E:/SpeechPrj/evaluation_results/experimental/final_benchmark/run_final_20261020_complete/run_manifest.json) (hash verification)

### Single Output File Created
- [`evaluation_results/experimental/phase4c2_experiment_design/PHASE4C2_EXPERIMENT_DESIGN.md`](file:///E:/SpeechPrj/evaluation_results/experimental/phase4c2_experiment_design/PHASE4C2_EXPERIMENT_DESIGN.md)

---

## 9. Final Phase 4C.2 Verdict

### **Verdict: `NEEDS MORE DEVELOPMENT DATA`**

#### Evidence & Justification:
1. **Design Completeness:** The experimental architecture, telemetry propagation, timestamp-quality checks, combined decision hierarchy, and pre-registered acceptance criteria are fully specified.
2. **Implementation Block:** However, Phase 4C.1 proved that the existing 90 development clips contain **only two clips** with $\ge 3$ repeated fillers and **zero non-stutter clips** with that pattern.
3. **Governance Requirement:** Implementing or tuning numerical guard thresholds against a development partition with zero non-stutter multi-filler examples would repeat the exact methodology error of Phase 4B.
4. **Prerequisite for Phase 4C.3:** Development data curation (sourcing the $N=40$ "Dev-Hesitation" cohort per Section 4) must be authorized and completed before Phase 4C.3 implementation and evaluation can be executed.
