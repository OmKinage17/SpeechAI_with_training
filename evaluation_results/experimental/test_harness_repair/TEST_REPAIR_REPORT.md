# SpeechAI — Authentication Test Harness Repair Report

**Date:** 2026-10-22  
**Test File Modified:** [`backend/tests/test_auth_flow.py`](file:///e:/SpeechPrj/backend/tests/test_auth_flow.py)  
**Suite Runner Executed:** [`backend/run_all_tests.py`](file:///e:/SpeechPrj/backend/run_all_tests.py)  
**Deliverable Path:** [`evaluation_results/experimental/test_harness_repair/TEST_REPAIR_REPORT.md`](file:///e:/SpeechPrj/evaluation_results/experimental/test_harness_repair/TEST_REPAIR_REPORT.md)

---

## 1. Executive Summary

The reported failure in `backend/run_all_tests.py` originating from `test_auth_flow.py` has been resolved. The test harness was refactored from an external HTTP socket-dependent integration test to a deterministic in-memory `TestClient` architecture with full dependency isolation and automatic database teardown.

All 7 test suites in `backend/run_all_tests.py` now pass with **exit code 0 (100% pass rate)**. **No production files were modified**.

---

## 2. Root Cause Analysis

### 2.1 Why `test_auth_flow.py` Sent Requests to `127.0.0.1:8000`
In the original implementation of `backend/tests/test_auth_flow.py`:
```python
def test_endpoints():
    print("\nTesting registration & login endpoints...")
    base_url = "http://127.0.0.1:8000"
    ...
    reg_url = f"{base_url}/auth/register"
    response = requests.post(reg_url, json=reg_payload)
```
- The test was written as an out-of-process integration test using the `requests` library.
- It hardcoded `base_url = "http://127.0.0.1:8000"`, assuming an active, running Uvicorn server on port 8000.
- When `backend/run_all_tests.py` executed `test_auth_flow.py` as a standalone subprocess without a pre-existing background server, `requests.post()` failed immediately with:
  `requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /auth/register (Caused by NewConnectionError: [WinError 10061] No connection could be made because the target machine actively refused it)`

### 2.2 App and Database Dependency Inspection
- `backend/main.py` defines the global database handle `db = mongo_client["speechai_db"]` (or `None` if MongoDB is offline).
- The endpoints `/auth/register` and `/auth/login` access `users_collection = db["users"]`.
- The existing sister test suites (`test_groq_generation.py`, `test_speech_pathology_analysis.py`) use FastAPI's in-memory `from fastapi.testclient import TestClient` without opening network sockets.
- The authentication test lacked this in-memory harness pattern and lacked test database isolation.

---

## 3. Implementation Details of the Test Harness Repair

The test harness in [`backend/tests/test_auth_flow.py`](file:///e:/SpeechPrj/backend/tests/test_auth_flow.py) was refactored with the following enhancements:

1. **Adoption of FastAPI `TestClient`:**
   Replaced `requests.post("http://127.0.0.1:8000/...")` with `client = TestClient(app)`. Requests are dispatched directly through ASGI in-memory without network sockets or running server daemons.

2. **In-Memory Isolated Database Fixture (`InMemoryDatabase`):**
   Implemented lightweight `InMemoryUsersCollection` and `InMemoryDatabase` classes mimicking PyMongo collection semantics (`find_one`, `insert_one`, `delete_many`).
   - During `test_endpoints()`, `main.db` is temporarily swapped to `InMemoryDatabase()` inside a `try...finally` block.
   - Eliminates database pollution in the local `speechai_db.users` collection.
   - Guarantees tests pass in any CI/CD environment even if MongoDB is not installed or offline.

3. **Expanded Coverage of Authentication Flows & Error Contracts:**
   - **Password Hashing:** Verified PBKDF2 hashing, correct password verification, wrong password rejection, and malformed hash string handling.
   - **HMAC Tokens:** Verified token generation, valid user verification, signature tampering rejection, malformed string rejection, and token expiration rejection (>7 days).
   - **Endpoint Registration:** Verified 200 OK success and 400 Bad Request duplicate email rejection (`"Email already registered"`).
   - **Endpoint Login:** Verified 200 OK success (bearer token generation, user profile return), token validity matching user ID, 400 Bad Request on incorrect password, and 400 Bad Request on non-existent user.
   - **Database Unavailable Handling:** Added `test_database_unavailable_error()` verifying that when `main.db = None`, `/auth/register` and `/auth/login` cleanly return 500 Internal Server Error (`"Database not available"`).
   - **Live MongoDB Persistence & Teardown:** Added `test_live_mongodb_integration()` which, when MongoDB is detected, tests actual MongoDB persistence and guarantees document teardown (`delete_many`) in a `finally` block.

---

## 4. Verification and Execution Results

### 4.1 Standalone Test Execution
**Command:**
```powershell
& "e:\SpeechPrj\backend\venv\Scripts\python.exe" backend/tests/test_auth_flow.py
```
**Results:**
- Password Hashing PBKDF2: **PASS**
- HMAC Token Security (tamper, malformed, expiry): **PASS**
- In-Memory Registration & Login (`TestClient`): **PASS**
- Database Unavailable Contract (500 handling): **PASS**
- Live MongoDB Persistence & Teardown: **PASS** (1 test record created and deleted)
- **Exit Code:** `0` (Success)

### 4.2 Full Test Suite Execution
**Command:**
```powershell
& "e:\SpeechPrj\backend\venv\Scripts\python.exe" backend/run_all_tests.py
```
**Summary Table:**

| Test Suite | Purpose | Execution Mode | Status |
|---|---|---|---|
| `test_streak_logic.py` | Daily practice streak calculations | In-memory MongoDB | **PASS** |
| `test_fluency_analytics.py` | Filler word & repetition counting | Rule-based regex analytics | **PASS** |
| `test_auth_flow.py` | PBKDF2, HMAC, Register/Login routes | In-memory `TestClient` + live DB | **PASS** |
| `test_groq_generation.py` | AI reading passage generation | `TestClient` + Groq API | **PASS** |
| `test_pronunciation_alignment.py` | Word-level alignment & WER | String alignment logic | **PASS** |
| `test_speech_pathology_analysis.py` | Clinical AI feedback evaluation | `TestClient` + Gemini/Groq API | **PASS** |
| `test_feedback_evaluation.py` | Custom rules condition evaluation | AST condition parser | **PASS** |

**Final Suite Verdict:**
```
============================================================
[SUCCESS] ALL SPEECH_AI TEST SUITES PASSED METICULOUSLY!
============================================================
```
- Total Test Suites: 7
- Passed: 7
- Failed: 0
- Skipped: 0
- **Exit Code:** `0`

---

## 5. Production Code and Detector Freeze Integrity

- **Production Files Modified:** **0** (`backend/main.py` was not modified; API contracts, security controls, routes, and behavior remain identical).
- **Detector Files Modified:** **0** (`backend/experimental/asr_loop_guard.py` and `backend/experimental/stutter_detector.py` remain untouched).
- **Holdout Audio Isolation:** 100% maintained (spent 120-clip Phase 4B holdout audio was not accessed, loaded, or evaluated).
