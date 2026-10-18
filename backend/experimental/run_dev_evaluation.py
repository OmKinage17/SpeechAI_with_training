"""
Runner script for Development Evaluation of SpeechAI Experimental Pipeline
========================================================================
Evaluates a balanced 30-clip development set (10 UCLASS, 10 FluencyBank, 10 Libri_Stutter)
against the audited baseline predictions, computing side-by-side performance metrics.
"""

import os
import sys
import json
import time

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from experimental.config import ExperimentalConfig
from experimental.eval_harness import DevelopmentEvaluationHarness

def main():
    root_dir = os.path.dirname(backend_dir)
    manifest_path = os.path.join(root_dir, "evaluation_results", "pilot_manifest.json")
    baseline_pred_path = os.path.join(root_dir, "evaluation_results", "pilot_predictions.json")
    out_dir = os.path.join(root_dir, "evaluation_results", "experimental")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "dev_eval_results_30clips_refined.json")

    print("=" * 70)
    print("SPEECHAI EXPERIMENTAL PIPELINE — REFINED DEVELOPMENT EVALUATION (30 CLIPS)")
    print("=" * 70)

    with open(manifest_path, "r", encoding="utf-8") as f:
        full_pilot = json.load(f)

    # Partition by dataset: 10 UCLASS, 10 FluencyBank, 10 Libri_Stutter
    uclass_clips = [c for c in full_pilot if c["dataset"] == "UCLASS"][:10]
    fb_clips = [c for c in full_pilot if c["dataset"] == "FluencyBank"][:10]
    libri_clips = [c for c in full_pilot if c["dataset"] == "Libri_Stutter"][:10]

    dev_manifest = uclass_clips + fb_clips + libri_clips
    print(f"Selected {len(dev_manifest)} clips for development evaluation:")
    print(f" - UCLASS: {len(uclass_clips)} clips")
    print(f" - FluencyBank: {len(fb_clips)} clips")
    print(f" - Libri_Stutter: {len(libri_clips)} clips\n")

    # Initialize experimental harness
    config = ExperimentalConfig()
    harness = DevelopmentEvaluationHarness(config)

    # Run evaluation
    t0 = time.time()
    try:
        results = harness.run_benchmark_on_manifest(dev_manifest, baseline_predictions_path=baseline_pred_path)
    finally:
        harness.asr_engine.shutdown()
    elapsed = round(time.time() - t0, 2)
    print(f"\n[DONE] Evaluation completed in {elapsed}s.")

    # Save detailed JSON results
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to: {out_json}\n")

    # Display comparison
    base = results.get("baseline", {})
    exp = results.get("experimental", {})

    print("=" * 70)
    print("SIDE-BY-SIDE DEVELOPMENT BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<35} | {'Baseline':<15} | {'Experimental':<15}")
    print("-" * 70)

    # 3-Class Multiclass
    b_acc = base.get("multiclass_3class", {}).get("accuracy", "N/A")
    e_acc = exp.get("multiclass_3class", {}).get("accuracy", "N/A")
    print(f"{'3-Class Dominant Accuracy':<35} | {b_acc:<15} | {e_acc:<15}")

    b_mf1 = base.get("multiclass_3class", {}).get("macro_f1", "N/A")
    e_mf1 = exp.get("multiclass_3class", {}).get("macro_f1", "N/A")
    print(f"{'3-Class Macro F1-Score':<35} | {b_mf1:<15} | {e_mf1:<15}")

    # Stutter Binary
    b_s_prec = base.get("stutter_binary", {}).get("precision", "N/A")
    e_s_prec = exp.get("stutter_binary", {}).get("precision", "N/A")
    print(f"{'Stutter Precision':<35} | {b_s_prec:<15} | {e_s_prec:<15}")

    b_s_rec = base.get("stutter_binary", {}).get("recall", "N/A")
    e_s_rec = exp.get("stutter_binary", {}).get("recall", "N/A")
    print(f"{'Stutter Recall':<35} | {b_s_rec:<15} | {e_s_rec:<15}")

    b_s_f1 = base.get("stutter_binary", {}).get("f1", "N/A")
    e_s_f1 = exp.get("stutter_binary", {}).get("f1", "N/A")
    print(f"{'Stutter F1-Score':<35} | {b_s_f1:<15} | {e_s_f1:<15}")

    # Filler Binary
    b_f_prec = base.get("filler_binary", {}).get("precision", "N/A")
    e_f_prec = exp.get("filler_binary", {}).get("precision", "N/A")
    print(f"{'Filler Precision':<35} | {b_f_prec:<15} | {e_f_prec:<15}")

    b_f_rec = base.get("filler_binary", {}).get("recall", "N/A")
    e_f_rec = exp.get("filler_binary", {}).get("recall", "N/A")
    print(f"{'Filler Recall':<35} | {b_f_rec:<15} | {e_f_rec:<15}")

    b_f_f1 = base.get("filler_binary", {}).get("f1", "N/A")
    e_f_f1 = exp.get("filler_binary", {}).get("f1", "N/A")
    print(f"{'Filler F1-Score':<35} | {b_f_f1:<15} | {e_f_f1:<15}")

    # Latency Stats
    b_med = base.get("latency", {}).get("median_sec", "N/A")
    e_med = exp.get("latency", {}).get("median_sec", "N/A")
    print(f"{'Median Latency (s)':<35} | {b_med:<15} | {e_med:<15}")

    b_mean = base.get("latency", {}).get("mean_sec", "N/A")
    e_mean = exp.get("latency", {}).get("mean_sec", "N/A")
    print(f"{'Mean Latency (s)':<35} | {b_mean:<15} | {e_mean:<15}")

    b_p95 = base.get("latency", {}).get("p95_sec", "N/A")
    e_p95 = exp.get("latency", {}).get("p95_sec", "N/A")
    print(f"{'P95 Latency (s)':<35} | {b_p95:<15} | {e_p95:<15}")

    b_rtf = base.get("latency", {}).get("aggregate_rtf", "N/A")
    e_rtf = exp.get("latency", {}).get("aggregate_rtf", "N/A")
    print(f"{'Aggregate RTF':<35} | {b_rtf:<15} | {e_rtf:<15}")
    print("=" * 70)

if __name__ == "__main__":
    main()
