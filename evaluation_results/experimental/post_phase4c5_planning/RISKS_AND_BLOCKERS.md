# SpeechAI — Risks, Blockers, and Technical Debt Register

**Date:** 2026-10-22  
**Context:** Post-Phase 4C.5 Stabilization and Phase 5 Preparation  
**Artifact Directory:** [`evaluation_results/experimental/post_phase4c5_planning/`](file:///e:/SpeechPrj/evaluation_results/experimental/post_phase4c5_planning/)

---

## 1. Executive Risk Summary

This document catalogues the identified technical, operational, architectural, and clinical risks across the SpeechAI platform. Each risk is evaluated by severity, likelihood, operational impact, and mitigation roadmap.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                               RISK TAXONOMY                                      │
│                                                                                  │
│  [CRITICAL BLOCKERS]  ──► Issues preventing immediate production or validation   │
│  [HIGH RISKS]         ──► Clinical, ethical, or operational vulnerabilities      │
│  [MEDIUM RISKS]       ──► Infrastructure, latency, and third-party dependencies  │
│  [LOW / TECH DEBT]    ──► Code structure, test harnesses, and maintenance debt   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Critical Blockers (Pre-Phase 5 Prerequisites)

### B-01: Absence of an Unspent, Multi-Clinic Clinical Evaluation Dataset
- **Classification:** Critical Blocker (Methodological / Regulatory)
- **Description:** The 120-clip Phase 4B holdout dataset is officially spent and locked. The 40-clip Dev-Hesitation cohort was utilized for exploratory candidate discovery and cannot serve as an independent test set.
- **Impact:** The experimental candidate (`ASRLoopGuard`) cannot be authorized for production integration until evaluated against a completely fresh, unspent evaluation set.
- **Mitigation:** Execute the [Phase 5 Validation Protocol](file:///e:/SpeechPrj/evaluation_results/experimental/post_phase4c5_planning/PHASE5_VALIDATION_PROTOCOL.md) to recruit new clinical sites and assemble an unspent $N \ge 150$ cohort with dual-SLP annotation.

### B-02: Patient Consent and IRB Data Governance Requirements
- **Classification:** Critical Blocker (Legal / Ethical)
- **Description:** Speech recordings of individuals with speech disorders constitute sensitive biometric health information. Collecting, storing, or evaluating clinical patient recordings without documented IRB approval or written informed consent poses legal and institutional compliance risks.
- **Impact:** Any data gathered without formal consent cannot be utilized in published benchmarks or clinical evaluations.
- **Mitigation:** Require written informed consent, cryptographic pseudonymization, acoustic PII scrubbing, and secure local storage before ingesting any new audio.

---

## 3. High Operational and Architectural Risks

### R-01: Disconnection Between Production and Experimental Detection Engines
- **Classification:** High Risk (Architectural)
- **Description:** The live production endpoint (`POST /analyze/speech` in [`backend/main.py`](file:///e:/SpeechPrj/backend/main.py)) executes legacy regex and string-matching routines. The advanced research components (`HybridStutterDetector`, acoustic features, contextual scoring, and `ASRLoopGuard`) exist exclusively in `backend/experimental/`.
- **Impact:** Live users interacting with the frontend do not experience the accuracy improvements developed during Phases 1–4C.5.
- **Mitigation:** Maintain the freeze until Phase 5 validation is completed. Once formally authorized, execute an architectural integration PR that wires `HybridStutterDetector` directly into `POST /analyze/speech`.

### R-02: Third-Party Cloud API Latency and Rate Limiting (Gemini & Groq)
- **Classification:** High Risk (Operational / External Dependency)
- **Description:** Speech pathology suggestions rely on Google Gemini 2.5 Flash, and customized reading drills rely on Groq Cloud (`openai/gpt-oss-120b`). While local fallback templates exist, internet outages, expired API keys, or API quota exhaustion degrade the clinical user experience.
- **Impact:** Live demonstrations in environments without stable internet or valid keys will lose dynamic AI feedback.
- **Mitigation:** Implement proactive token caching, pre-warmed local clinical guidance templates, and an automated health-check banner in the UI alerting the user if external AI services are offline.

---

## 4. Medium Infrastructure and Performance Risks

### R-03: Local MongoDB Daemon Dependency
- **Classification:** Medium Risk (Operational / Deployment)
- **Description:** The backend connects to `mongodb://localhost:27017` on startup. If MongoDB is not running as an active background service on the host OS, the backend sets `db = None` and operates in an ephemeral, un-persisted state.
- **Impact:** User registration, login, report history, and streak analytics fail if MongoDB is offline.
- **Mitigation:** Provide a standalone Docker Compose configuration (`docker-compose up -d mongodb`) or an automated PowerShell verification script that checks `mongod.exe` status prior to launching `uvicorn`.

### R-04: Whisper Base CPU Inference Latency
- **Classification:** Medium Risk (Performance / User Experience)
- **Description:** The Whisper `base` model runs on CPU by default. Processing a 15-second user audio sample takes 2.5 to 4.5 seconds.
- **Impact:** Consecutive user submissions create perceived UI lag during live demonstrations.
- **Mitigation:** Configure `faster-whisper` (CTranslate2 backend) with 4-bit/8-bit quantization, or automatically leverage NVIDIA CUDA GPU acceleration when available (`torch.cuda.is_available()`).

---

## 5. Low Severity / Technical Debt Register

### T-01: Test Suite Harness Flaw in `test_auth_flow.py` [RESOLVED]
- **Classification:** Technical Debt (Harness Defect) — **RESOLVED & VERIFIED**
- **Description:** Previously in `backend/run_all_tests.py`, `test_auth_flow.py` made raw network calls via `requests` against `http://127.0.0.1:8000/auth/register`, assuming a live server was already running.
- **Resolution:** Refactored `test_auth_flow.py` to use FastAPI's in-memory `TestClient(app)`. Running `python backend/run_all_tests.py` now executes all 7 suites and 100% pass without requiring an external server process.

### T-02: Monolithic Frontend Structure (`frontend/src/App.tsx`)
- **Classification:** Technical Debt (Maintainability)
- **Description:** `App.tsx` contains 224KB of code (~3,600 lines) combining navigation, authentication modals, dashboard state, recording controls, audio playback, and canvas visualizers into a single file.
- **Impact:** Increases risk of merge conflicts and accidental regressions during UI enhancements.
- **Mitigation:** Refactor `App.tsx` into modular domain directories:
  * `frontend/src/components/dashboard/`
  * `frontend/src/components/fluency/`
  * `frontend/src/components/exercises/`
  * `frontend/src/components/auth/`

---

## 6. Demonstration Pre-Flight Checklist

To ensure a flawless live demonstration of the SpeechAI platform, the following pre-flight steps must be verified:

1. **MongoDB Service:** Verify MongoDB is active (`net start MongoDB` or `mongod`).
2. **Environment Variables:** Verify `.env` contains valid `GEMINI_API_KEY` and `GROQ_API_KEY` values.
3. **Microphone Permissions:** Verify the host browser (Chrome / Edge) has granted microphone and camera permissions for `http://localhost:5173`.
4. **Backend Launch:** Start backend via:
   ```powershell
   & "e:\SpeechPrj\backend\venv\Scripts\uvicorn.exe" main:app --reload --port 8000
   ```
5. **Frontend Launch:** Start frontend dev server via:
   ```powershell
   npm --prefix frontend run dev
   ```
