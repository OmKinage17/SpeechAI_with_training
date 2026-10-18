"""
Empirical Audit of ThreadPoolExecutor Timeout Behavior and Resource Isolation
=============================================================================
Tests:
1. Whether thread-pool timeout actually halts ongoing execution or leaves orphaned threads.
2. What happens to subsequent tasks submitted to ThreadPoolExecutor(max_workers=1) after a timeout.
3. Thread count, memory, and CPU queueing dynamics.
"""

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

print("=== RUNNING EMPIRICAL TIMEOUT AUDIT ===")

# Test 1: Simulating thread-pool timeout behavior
flag_running = False
task_completed = False

def long_running_task(duration_sec: float):
    global flag_running, task_completed
    flag_running = True
    print(f"  [Worker Thread] Started long task (duration: {duration_sec}s)...")
    t_start = time.time()
    while time.time() - t_start < duration_sec:
        time.sleep(0.1)
    flag_running = False
    task_completed = True
    print(f"  [Worker Thread] Finished long task after {round(time.time() - t_start, 2)}s.")
    return "SUCCESS"

executor = ThreadPoolExecutor(max_workers=1)

print("\n1. Testing ThreadPoolExecutor(max_workers=1) with 1.0s timeout on a 3.0s task:")
t0 = time.time()
future1 = executor.submit(long_running_task, 3.0)

try:
    res = future1.result(timeout=1.0)
    print("  [Main Thread] Completed:", res)
except FuturesTimeoutError:
    dt = round(time.time() - t0, 3)
    print(f"  [Main Thread] Timeout caught after {dt}s! Unblocked.")

print(f"  [Main Thread Check] Is worker thread still running in background? {flag_running}")
print(f"  [Main Thread Check] Total active threads: {threading.active_count()}")

# Test 2: Submitting second task immediately
print("\n2. Submitting second task (0.5s) immediately to same executor:")
t1 = time.time()
future2 = executor.submit(long_running_task, 0.5)

# Notice how long future2 takes to even START if worker is still busy!
try:
    res2 = future2.result(timeout=3.0)
    dt2 = round(time.time() - t1, 3)
    print(f"  [Main Thread] Second task completed after {dt2}s.")
except FuturesTimeoutError:
    print("  [Main Thread] Second task timed out!")

executor.shutdown(wait=True)
print("\n=== TIMEOUT AUDIT COMPLETE ===")
