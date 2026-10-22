# SpeechAI Phase 5 — Independent Clinical Validation Protocol

**Protocol Version:** 1.0 (Draft / Pre-execution)  
**Date:** 2026-10-22  
**Target Evaluation Size:** $N \ge 150$ audio clips  
**Artifact Directory:** [`evaluation_results/experimental/post_phase4c5_planning/`](file:///e:/SpeechPrj/evaluation_results/experimental/post_phase4c5_planning/)

---

## 1. Protocol Objective & Ethical Premise

The objective of Phase 5 is to perform a rigorous, unbiased, out-of-domain evaluation of the SpeechAI stutter detection system and its experimental loop safeguard (`ASRLoopGuard`) against a completely fresh clinical dataset ($N \ge 150$).

### Fundamental Methodological Rules:
1. **Total Isolation from Spent Data:** The spent 120-clip Phase 4B holdout (`fdfa62ce5a85c22b57db013fbee8451f84777058e671ae2eb897f8c121e4fb00`) and the 40-clip development cohort are permanently excluded.
2. **Pre-Registered Evaluation:** All inclusion criteria, annotation rubrics, primary metrics, and acceptance thresholds are frozen *prior* to cohort assembly.
3. **Zero Post-Hoc Tuning:** No algorithmic parameters, weights, or thresholds may be modified after viewing Phase 5 evaluation results. A single evaluation run is permitted.

---

## 2. Ethical Oversight, Consent, and Data Privacy

Clinical audio recordings of individuals who stutter are sensitive biometric health data. The following compliance rules must be satisfied prior to collecting or processing any audio:

### 2.1 Regulatory and Institutional Governance
- **IRB / Ethics Approval:** Audio collection from human subjects must operate under an approved Institutional Review Board (IRB) or Clinical Ethics Committee protocol.
- **Informed Consent:** Every participant (or legal guardian for minors) must execute a written informed consent agreement explicitly permitting the use of their audio recordings for speech technology research and algorithm validation.
- **Right to Erasure:** The data management infrastructure must support cryptographic revocation and deletion if a participant withdraws consent.

### 2.2 Biometric Privacy and De-Identification
- **Acoustic Redaction:** Any spoken Personally Identifiable Information (PII) — including personal names, clinic names, addresses, or phone numbers — must be identified and scrubbed (silenced) prior to inclusion in the dataset.
- **Metadata Anonymization:** File identifiers must use pseudorandom UUIDs (e.g., `P5_SITE01_SPK042_001.wav`). All speaker demographic keys must be stored in an air-gapped, encrypted mapping table accessible only to the principal investigator.
- **Third-Party API Isolation:** Evaluation audio must be processed strictly through local models (local Whisper base, local acoustic feature extraction). **Raw patient audio must never be sent to external cloud APIs** (Groq, Gemini, or public transcription endpoints) during validation.

---

## 3. Cohort Composition and Inclusion/Exclusion Criteria

The validation dataset will comprise **$N \ge 150$ audio clips** sampled across multiple clinical collection sites and recording environments.

### 3.1 Stratified Cohort Target ($N = 160$ Target)

| Sub-Cohort | Category Label | Target Size | Clinical Description | Purpose |
|---|---|---|---|---|
| **Cohort A** | `P5_FLUENT_CONTROL` | 40 clips | Fluent speakers performing impromptu and reading tasks. | Specificity baseline across varied accents and microphones. |
| **Cohort B** | `P5_CONVERSATIONAL_HESITATION` | 40 clips | Fluent speakers with high rates of normal disfluencies (`"um"`, `"uh"`, pauses, self-repairs). | Validate linguistic hesitation gating and prevent false positives. |
| **Cohort C** | `P5_CLINICAL_STUTTER` | 50 clips | Individuals diagnosed with developmental or acquired stuttering (repetitions, prolongations, blocks). | Evaluate genuine dysfluency sensitivity across severity levels. |
| **Cohort D** | `P5_ACOUSTIC_CHALLENGE` | 30 clips | Low-energy murmurs, trailing vocal fry, breathy pauses, background ambient noise. | Stress-test the Whisper decoder and validate `ASRLoopGuard`. |

### 3.2 Speaker and Session Isolation Criteria
- **Speaker Isolation:** Zero speakers or sessions may overlap with:
  * Standardized Dev 30 (`dev_30`)
  * Generalization Validation 60 (`gen_60`)
  * 40-clip Dev-Hesitation Cohort (`DH_FLUE_01..10`, `DH_CONV_01..10`, etc.)
  * The spent 120-clip holdout.
- **Site Diversity:** Clips must be acquired across at least 3 distinct recording setups (e.g., studio dynamic microphone, laptop integrated microphone, mobile smartphone microphone) to prevent hardware acoustic bias.

### 3.3 Audio Inclusion & Exclusion Rules
- **Inclusion:**
  * Sample rate $\ge 16$ kHz, 16-bit depth (converted to 16 kHz mono WAV).
  * Clip duration: 3.0 seconds to 15.0 seconds.
  * Spoken in English (including international and Indian English varieties).
- **Exclusion:**
  * Severe clipping or digital distortion ($> 1\%$ clipped samples).
  * Extreme background cross-talk (multiple overlapping speakers).
  * Audio with unverified participant consent.

---

## 4. Clinical Annotation Standard Operating Procedure (SOP)

### 4.1 Independent Expert Annotation Panel
- Every clip must be annotated independently by **two certified Speech-Language Pathologists (SLPs)** who are blinded to:
  * The model's predictions.
  * The specific sub-cohort assignment (Cohorts A, B, C, D).
  * The presence or absence of the experimental safeguard.
- In cases of disagreement between the two primary annotators, an independent **Senior Clinical Adjudicator** will review the audio and render the binding ground-truth label.

### 4.2 Annotation Categories & Hierarchy
Annotators will label clips using a standardized taxonomy:
1. **Binary Ground Truth (`stutter_binary`):** `0` = No clinical stuttering; `1` = Clinical stuttering present.
2. **Dominant Event Type (`dominant_type`):**
   - `REPETITION`: Sound, syllable, or whole-word iteration accompanied by tension.
   - `PROLONGATION`: Involuntary lengthening of sound $\ge 0.50$s.
   - `BLOCK`: Involuntary silent or tense stoppage of airflow/phonation.
   - `NORMAL_HESITATION`: Unstressed filler interjection (`"um"`, `"uh"`, revision) without physical tension.
   - `FLUENT`: Smooth phonation.
3. **Timestamp Bounds:** Precise onset and offset timestamps for each dysfluent event.

### 4.3 Inter-Rater Reliability Threshold
- Inter-annotator agreement will be measured via Cohen’s Kappa ($\kappa$).
- Annotation quality threshold: $\kappa \ge 0.80$. Clips failing consensus after adjudication will be flagged for review or excluded before manifest locking.

---

## 5. Separation of Operational Roles & Manifest Locking

To ensure absolute scientific integrity, the workflow enforces strict role segregation:

```
┌────────────────────────────────┐
│   DATA & ETHICS CURATOR        │
│   - Acquires consented audio   │
│   - Anonymizes & scrubs PII    │
│   - Generates SHA-256 hashes   │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│   CLINICAL ANNOTATION TEAM     │
│   - Blinded dual SLP review    │
│   - Adjudication of conflicts  │
│   - Produces ground_truth.csv  │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐       LOCKED MANIFEST
│   MANIFEST LOCKING & VAULT     │ ────► phase5_manifest.json (SHA-256)
│   - Freezes audio & labels     │       (Engineers strictly blinded)
└───────────────┬────────────────┘
                │
                ▼ (Single Evaluation Execution)
┌────────────────────────────────┐
│   MODEL EVALUATION HARNESS     │
│   - Runs Canonical Baseline    │
│   - Runs Phase 4C Candidate    │
│   - Calculates Metrics         │
└────────────────────────────────┘
```

1. **Manifest File:** `phase5_evaluation_manifest.json` will list:
   - File path, clip UUID, file SHA-256 hash, duration, audio format.
   - Speaker UUID, recording site code, microphone type.
   - Ground truth binary label, dominant event type, annotator consensus score.
2. **Cryptographic Locking:** The manifest's root SHA-256 checksum will be published into the project repository prior to running the evaluation script.

---

## 6. Predefined Evaluation Metrics and Acceptance Thresholds

The evaluation harness will compute metrics comparing the **Canonical Phase 2 Baseline** vs the **Phase 4C.3 Candidate** (`ASRLoopGuard` active):

### 6.1 Primary Evaluation Metric
- **Binary Stutter Detection F1 Score ($F_1$)** across the overall $N \ge 150$ validation set.

### 6.2 Pre-Registered Acceptance Criteria
For the candidate to be authorized for production integration, it must satisfy **ALL** of the following criteria:

1. **Overall F1 Improvement / Preservation:**
   $$F_{1\text{ (Candidate)}} \ge F_{1\text{ (Baseline)}} \quad \text{and} \quad F_{1\text{ (Candidate)}} \ge 0.7500$$
2. **High Specificity Floor:**
   $$\text{Specificity}_{\text{Candidate}} \ge 90.0\% \quad (\text{False Positive Rate} \le 10.0\%)$$
3. **Loop False Positive Suppression (Cohort D):**
   $$\text{FP}_{\text{Cohort D (Candidate)}} \le 3 \quad (\le 10.0\% \text{ loop artifact false positives})$$
4. **Zero Clinically Significant Recall Regression (Cohort C):**
   $$\text{Recall}_{\text{Cohort C (Candidate)}} \ge \text{Recall}_{\text{Cohort C (Baseline)}} - 2.0\%$$
5. **No Independent Prolongation Suppression:**
   Acoustic prolongations verified by SLPs ($\ge 0.50$s) must exhibit zero suppression by the loop guard.

---

## 7. Execution and Post-Evaluation Rules

1. **Automated Report Generation:** The harness will generate `PHASE5_VALIDATION_REPORT.md` and `phase5_metrics.json` directly from the evaluation run.
2. **Failure Protocol:** If any acceptance criterion fails, the candidate is declared **REJECTED**. No tuning or parameter changes are permitted on Phase 5 data.
3. **Production Authorization:** Production integration into `backend/main.py` may only proceed upon formal review and sign-off of the generated Phase 5 report by the Lead Architect and Clinical Advisory Board.
