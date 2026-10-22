# Dev-Hesitation Cohort Annotation Guidelines (Phase 4C.2.1)

## 1. Overview and Purpose
This document establishes the official adjudication protocol and rubric for curating the Dev-Hesitation cohort under Phase 4C.2.1. The objective is to establish an audited, reproducible benchmark partition of non-holdout speech clips containing hesitation, repetition, and temporal anomalies to evaluate ASR repetition-loop mitigation without compromising genuine stuttering detection.

All data curation strictly adheres to research integrity standards:
- **Zero Holdout Contamination:** No clips from the locked 120-clip Phase 4B benchmark (`fdfa62ce5a85c22b57db013fbee8451f84777058e671ae2eb897f8c121e4fb00`) may ever be included or referenced.
- **Strict Partition Independence:** Candidates are sourced exclusively from non-holdout and non-dev90 speakers and sessions.
- **Independence of Labels:** Annotations are established strictly through empirical two-pass review and must not be inferred solely from model predictions.

---

## 2. Target Categories and Cohort Quotas

The target cohort aims for up to 40 clips across four distinct categories (nominal target: 10 per category):

| Category | Description | Primary Distinguishing Criteria |
| :--- | :--- | :--- |
| **Category 1: Fluent Deliberation Controls** | Fluent speech containing rapid or repeated deliberation tokens (`um`, `uh`, `er`, `like`) without physical struggle or articulatory tension. | Ground-truth non-stutter (`Clean` / `Fluent`); speaker is deliberating/formulating ideas; smooth vocal tract transition; absence of blocks or physical tremors. |
| **Category 2: Conversational Non-Stutter Filler Chains** | Conversational dialogue where the speaker emits repeated hesitation tokens (`um... um`, `uh... uh`). | Ground-truth non-stutter (`Filler`); conversational pacing; natural pause intervals; non-pathological dysfluency. |
| **Category 3: Genuine Stuttering Struggle Chains** | Speech exhibiting genuine articulatory struggle, sound/syllable/word repetitions, blocks, or prolongations. | Ground-truth stuttering (`Stutter`); acoustic evidence of muscular tension, acoustic fragmentation, or repetitive phonatory arrests. |
| **Category 4: ASR Repetition-Loop Artifacts** | Speech where the ASR decoder generated repetitive token sequences that do not correspond to repeated audible acoustic events. | Severe timestamp collapse ($\ge 3$ zero-duration tokens or identical timestamps), degenerate runaway loops ($N \ge 6$ identical ungrounded tokens), or hallucinated $n$-gram cycles during low-energy trailing silence. |

*Quotas are target aspirations. If empirical data yields fewer than 10 verified clips in a category, the true shortfall must be reported without relaxing eligibility rules or mislabeling ambiguous clips.*

---

## 3. Two-Pass Annotation Protocol

Every candidate clip undergoes a rigorous two-pass adjudication procedure:

### Pass 1: Auditory Review
An auditory evaluation is performed using calibrated playback to evaluate acoustic reality:
1. **Audible Speech Reality:** Does the audio contain speech across the full 3.0s window, or does it transition into ambient noise/silence?
2. **Repetition Audit:** For any token repeated in the transcript, is there an audible acoustic repetition produced by the speaker?
   - *Audible repetition present:* Speaker audibly articulates the repeated syllable, word, or hesitation marker.
   - *Hallucinated repetition:* Audio is silent, static, or contains a single breath/murmur, while transcript lists multiple repeated tokens.
3. **Struggle and Tension Audit:** Are there audible signs of physical effort, glottal fry, abrupt phonatory arrest, prolongation, or muscular blocking?
4. **Deliberation vs. Struggle Distinction:**
   - *Fluent Deliberation:* Normal pitch contour, relaxed phonation, typical pauses between lexical formulations.
   - *Struggle:* High pitch variance, broken syllables, dysrhythmic phonation, audible struggle tension.

### Pass 2: Spectrogram and Temporal Alignment Review
A temporal and spectral alignment audit is performed using word-level alignment telemetry:
1. **Temporal Span:** The duration from `start` of the first repeated token to `end` of the last repeated token ($\Delta t = t_{\text{end}} - t_{\text{start}}$).
   - If $\Delta t < 0.05\,\text{s}$ for $\ge 3$ tokens, flag as **Severe Timestamp Collapse**.
2. **Inter-Segment Interval (ISI):** Time between the end of token $i$ and start of token $i+1$.
   - $\text{ISI} \le 0.015\,\text{s}$ across multiple tokens indicates decoder runaway rather than human articulatory capability.
3. **Zero-Duration Density:** Count of tokens where `start == end`.
   - $\ge 3$ zero-duration tokens indicates timestamp failure / ungrounded hallucination.
4. **Spectral Grounding:** Verify whether spectral energy peaks (formants, bursts) correspond to reported token boundaries.

---

## 4. Adjudication Confidence and Scoring Rubric

Annotator confidence must be rigorously justified based on observable acoustic and temporal evidence. Adjudication confidence $\ge 0.90$ is reserved exclusively for clips where both passes provide unequivocal evidence:

| Confidence Level | Criteria and Justification |
| :--- | :--- |
| **$0.95 - 1.00$ (Unequivocal)** | Both auditory and spectral evidence are in complete agreement; timestamps precisely map to audible events (or unequivocally collapse to zero during silence); zero disagreement between reviewers. |
| **$0.90 - 0.94$ (High Confidence)** | Auditory evidence is clear; minor timestamp offset ($\le 0.1\,\text{s}$) due to soft onset/offset; consensus reached on categorization without residual ambiguity. |
| **$0.75 - 0.89$ (Moderate Confidence)** | Subtle hesitation that borders between conversational filler and mild tension; minor ambient background noise; slight spectral blurring. |
| **$< 0.75$ (Ambiguous / Exclude)** | Unresolvable ambiguity between word-search pause and block; audio clipping or heavy distortion; lack of reviewer consensus. *These clips are marked as AMBIGUOUS and excluded from final cohort.* |

---

## 5. Reviewer Roles and Dispute Resolution

1. **Reviewer 1 (Primary Annotator):** Performs systematic pass 1 & pass 2 analysis, documents raw/sanitized transcripts, token timing indicators, and initial category recommendation.
2. **Reviewer 2 (Verification Auditor):** Independently inspects audio alignment, checks candidate eligibility records, verifies provenance, and scores confidence.
3. **Consensus Adjudication:**
   - If Reviewers 1 and 2 agree on category and confidence $\ge 0.90$, the classification is accepted.
   - If Reviewers disagree or confidence $< 0.90$, the clip is flagged for joint adjudication.
   - If joint review cannot establish unequivocal ground truth with $\ge 0.90$ confidence, the clip is marked `AMBIGUOUS` and omitted from the cohort.

---

## 6. Output Schema Definitions

The resulting `annotation_results.csv` records the following standardized fields for each candidate:
1. `clip_id`: Base filename of the 3.0-second WAV file.
2. `corpus`: Originating dataset (`UCLASS` or `FluencyBank`).
3. `speaker_session`: Speaker ID or Podcast Episode ID.
4. `source_ground_truth`: Authoritative label from original dataset (`Clean`, `Filler`, `Stutter`).
5. `adjudicated_category`: Assigned Dev-Hesitation category (`fluent_deliberation_control`, `conversational_filler_chain`, `genuine_stutter_struggle`, `asr_loop_artifact`, or `ambiguous`).
6. `raw_transcript`: Raw transcript generated by baseline Whisper base model.
7. `sanitized_transcript`: Transcript after processing by baseline `detect_and_sanitize_repetition_loop`.
8. `sanitizer_status`: Status returned by baseline sanitizer (`SUCCESS`, `DEGENERATE_LOOP_RECOVERED`, etc.).
9. `audible_repetition`: Binary indicator (1 = repeated tokens correspond to audible speech events; 0 = repeated tokens are hallucinated).
10. `struggle_present`: Binary indicator (1 = independently observable acoustic struggle present; 0 = absent).
11. `zero_dur_count`: Number of transcript tokens with `start == end`.
12. `max_identical_starts`: Maximum number of tokens sharing an identical `start` timestamp.
13. `temporal_span_sec`: Time elapsed between first and last repeated token (seconds).
14. `mean_isi_sec`: Mean inter-segment interval across repeated tokens (seconds).
15. `reviewer_1_category`: Initial category proposed by Reviewer 1.
16. `reviewer_2_category`: Independent category proposed by Reviewer 2.
17. `agreement_status`: Reviewer agreement status (`UNANIMOUS_CONSENSUS`, `ADJUDICATED_CONSENSUS`, `DISAGREEMENT_EXCLUDED`).
18. `adjudication_confidence`: Final calibrated confidence score ($0.00 - 1.00$).
19. `adjudication_notes`: Summary of empirical basis for adjudication decision.
