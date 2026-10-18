import os
import hashlib
import subprocess
import json

audit_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(audit_dir)))

files_to_check = [
    "backend/main.py",
    "backend/run_main_benchmark.py",
    "evaluation_results/pilot_manifest.json",
    "evaluation_results/pilot_predictions.json",
    "evaluation_results/pilot_results_summary.json",
    "evaluation_results/METHODOLOGY_AND_PILOT_REPORT.md",
    "evaluation_results/main_benchmark/main_benchmark_manifest.json",
    "evaluation_results/main_benchmark/main_benchmark_predictions.json",
    "evaluation_results/main_benchmark/main_benchmark_results.json",
    "evaluation_results/main_benchmark/MAIN_BENCHMARK_REPORT.md",
    "evaluation_results/main_benchmark/audit/recomputed_metrics.json",
    "evaluation_results/main_benchmark/audit/AUDIT_REPORT.md",
]

print("=== CHECKING BASELINE & PRODUCTION FILE IMMUTABILITY ===")
status_report = {}
for rel_path in files_to_check:
    full_path = os.path.join(project_root, rel_path)
    if os.path.exists(full_path):
        size = os.path.getsize(full_path)
        with open(full_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        status_report[rel_path] = {
            "status": "EXISTS",
            "size_bytes": size,
            "sha256": h
        }
        print(f"[OK] {rel_path} (size: {size:,} bytes, sha256: {h[:16]}...)")
    else:
        status_report[rel_path] = {"status": "MISSING"}
        print(f"[MISSING] {rel_path}")

# Run git status on these files
print("\n=== GIT STATUS CHECK ON PRODUCTION FILES ===")
git_res = subprocess.run(
    ["git", "status", "--porcelain", "backend/main.py", "backend/run_main_benchmark.py", "evaluation_results/main_benchmark/"],
    capture_output=True, text=True, cwd=project_root
)
print("Git modified lines (should be empty for production/baseline files):")
print(git_res.stdout.strip() if git_res.stdout.strip() else "(Clean - No modifications)")
