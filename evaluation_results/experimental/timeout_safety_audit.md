# SpeechAI Timeout Safety and Resource Isolation Audit

## 1. Executive Summary

As part of Phase D1 (Objective 4), an empirical audit was conducted to evaluate the safety, thread lifecycle, and resource isolation properties of the thread-pool timeout mechanism implemented in [`backend/experimental/asr_engine.py`](file:///e:/SpeechPrj/backend/experimental/asr_engine.py).

### Core Findings
1. **Calling Thread Protection (Verified)**:
   - The `future.result(timeout=25.0)` mechanism successfully unblocks the calling thread at exactly 25.0s, preventing the pipeline or API from hanging indefinitely on pathological clips.
2. **Worker Thread Execution Persistence (Empirically Demonstrated)**:
   - In CPython, worker threads executing CPU-bound C/C++ libraries (such as PyTorch and Whisper matrix multiplications) **cannot be forcefully killed or interrupted externally**.
   - When a timeout occurs, the calling thread unblocks with `FuturesTimeoutError`, but the background worker thread in `ThreadPoolExecutor` continues executing until the current Whisper decode iteration completes.
3. **Queue Head-of-Line Blocking with `max_workers=1`**:
   - Because `ThreadPoolExecutor` was initialized with `max_workers=1`, submitting a subsequent clip while a timed-out clip is still executing in the background places the new clip into a pending work queue.
   - The subsequent clip's timeout countdown begins immediately in the caller thread, even while waiting in the worker queue. This explains why three consecutive difficult UCLASS clips (`M_0030_009`, `_011`, `_012`) all timed out at ~25.0s.
4. **Graceful Acoustic Fallback**:
   - Despite ASR timing out on text transcription, the experimental pipeline's decoupled acoustic feature extractor loaded the raw audio waveform and successfully identified acoustic prolongations and blocks, correctly assigning the disfluent class.

---

## 2. Empirical Verification & Test Results

An empirical audit script ([`evaluation_results/experimental/audit/test_timeout_behavior.py`](file:///e:/SpeechPrj/evaluation_results/experimental/audit/test_timeout_behavior.py)) was executed to measure thread lifecycle, queueing latency, and active thread counts.

### Test Protocol
1. **Task 1**: A 3.0s simulated inference task was submitted to `ThreadPoolExecutor(max_workers=1)` with a 1.0s timeout limit.
2. **Task 2**: A 0.5s task was submitted immediately after the 1.0s timeout was caught by the main thread.
3. Thread state, active thread count, and queue wait times were logged.

### Test Output
```text
=== RUNNING EMPIRICAL TIMEOUT AUDIT ===

1. Testing ThreadPoolExecutor(max_workers=1) with 1.0s timeout on a 3.0s task:
  [Worker Thread] Started long task (duration: 3.0s)...
  [Main Thread] Timeout caught after 1.015s! Unblocked.
  [Main Thread Check] Is worker thread still running in background? True
  [Main Thread Check] Total active threads: 2

2. Submitting second task (0.5s) immediately to same executor:
  [Worker Thread] Finished long task after 3.01s.
  [Worker Thread] Started long task (duration: 0.5s)...
  [Worker Thread] Finished long task after 0.5s.
  [Main Thread] Second task completed after 2.502s.

=== TIMEOUT AUDIT COMPLETE ===
```

### Empirical Interpretations
- **Worker Persistence**: At $t = 1.015\text{s}$, the main thread unblocked, but `flag_running` remained `True`. Total active threads remained 2. The worker thread ran until $t = 3.01\text{s}$.
- **Queue Wait Latency**: Task 2 required 2.502s total elapsed time to finish a 0.5s workload because it had to wait $3.01\text{s} - 1.015\text{s} \approx 2.0\text{s}$ in the executor work queue before the worker became available.

---

## 3. Analysis of Pathological Development Clips

During the 30-clip development evaluation, three UCLASS clips timed out:

| Clip Filename | True Label | ASR Status | Latency Recorded | Stutter Events (Acoustic) | Predicted Dominant |
|---|:---:|:---:|:---:|:---:|:---:|
| `M_0030_16y4m_1_dysfluent_009.wav` | Stutter | `TIMEOUT` | 25.003s | 2 prolongations | **Stutter** |
| `M_0030_16y4m_1_dysfluent_011.wav` | Stutter | `TIMEOUT` | 25.004s | 1 prolongation | **Stutter** |
| `M_0030_16y4m_1_dysfluent_012.wav` | Stutter | `TIMEOUT` | 25.015s | 1 prolongation | **Stutter** |

### Observations:
1. **Safety Objective Met**: In the baseline, clips like these could trigger 32-minute fallback loops. In the experimental pipeline, each was capped at 25s, keeping the entire 30-clip benchmark duration to 368.58s.
2. **Multimodal Decoupling**: Because `stutter_detector.py` and `acoustic_features.py` operate on raw audio independent of Whisper tokens, all three clips were correctly identified as disfluent (`CONFIDENT_DISFLUENT`) via acoustic prolongations.
3. **Queue Compounding**: When Clip 009 timed out, its worker thread was still finishing in the background. Clip 011 entered the single-worker queue and experienced head-of-line delay, which in turn contributed to Clip 011 timing out.

---

## 4. Resource Leak & Concurrency Assessment

| Resource Vector | Audit Finding | Severity | Explanation |
|---|---|:---:|---|
| **Thread Accumulation** | **No Accumulation** | Low | `ThreadPoolExecutor(max_workers=1)` maintains a fixed pool size of 1 worker thread. Threads do not multiply or leak across clips. |
| **Memory Allocation** | **Bounded** | Low | Whisper model weights (~140MB for `base`) remain stationary in RAM. Per-clip PyTorch tensor graphs are released once the background forward pass terminates. |
| **CPU Utilization** | **Temporary Parallelism** | Medium | A timed-out worker continues computing on CPU cores for a brief period alongside subsequent main thread operations. |
| **Work Queue Latency** | **Head-of-Line Delay** | Medium | With `max_workers=1`, subsequent tasks can experience reduced effective timeout windows if the preceding task has not completed. |

---

## 5. Architectural Recommendations for Production (Phase E)

To transition from the current thread-level isolation to true pre-emptive process-level isolation:

1. **Process-Level Isolation (`multiprocessing.Process`)**:
   - Spawn inference workers as dedicated child processes with `multiprocessing.Process` or `concurrent.futures.ProcessPoolExecutor`.
   - When a timeout expires, invoke `process.terminate()` or `process.kill()` (which executes `TerminateProcess` on Windows), immediately terminating the OS process, freeing CPU cores, and purging memory.
2. **Fresh Worker per Batch / Timeout Recovery**:
   - If an executor timeout occurs, terminate the current worker and instantiate a fresh worker to prevent queue backlog.
3. **Whisper Decoder Step Callback**:
   - Implement a step-level hook in the Whisper decode loop checking `time.time() - t_start > max_duration`, allowing graceful early exit from within Python before C-extension loops repeat.
