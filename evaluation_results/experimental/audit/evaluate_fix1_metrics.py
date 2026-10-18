"""
Evaluate Fix 1 Metrics Script
=============================
Computes side-by-side development metrics before vs after Fix 1
for the 30-clip development set, eliminating ASR timeout variance
by using the exact verified ASR transcripts and word boundaries.
"""

import os
import sys
import json
import statistics

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

project_root = os.path.dirname(backend_dir)
before_json = os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips_refined_before_fix1.json")
after_json = os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips_refined.json")

with open(before_json, "r", encoding="utf-8") as f:
    before_data = json.load(f)

with open(after_json, "r", encoding="utf-8") as f:
    after_data = json.load(f)

print("=" * 70)
print("DEVELOPMENT EVALUATION: BEFORE VS. AFTER FIX 1 (30 CLIPS)")
print("=" * 70)

# 1. Full pipeline run comparison (from dev_eval_results_30clips_refined.json)
b_exp = before_data["experimental"]
a_exp = after_data["experimental"]

print("\n--- 1. Live Runner Metrics (ASR + Acoustic + Lexical) ---")
b_acc = b_exp["multiclass_3class"]["accuracy"]
a_acc = a_exp["multiclass_3class"]["accuracy"]
b_mf1 = b_exp["multiclass_3class"]["macro_f1"]
a_mf1 = a_exp["multiclass_3class"]["macro_f1"]
print(f"3-Class Accuracy: Baseline={before_data['baseline']['multiclass_3class']['accuracy']:.4f} | Before Fix 1={b_acc:.4f} | After Fix 1={a_acc:.4f}")
print(f"3-Class Macro F1: Baseline={before_data['baseline']['multiclass_3class']['macro_f1']:.4f} | Before Fix 1={b_mf1:.4f} | After Fix 1={a_mf1:.4f}")

b_st = b_exp["stutter_binary"]
a_st = a_exp["stutter_binary"]
print(f"Stutter Precision: Before={b_st['precision']:.4f} -> After={a_st['precision']:.4f}")
print(f"Stutter Recall:    Before={b_st['recall']:.4f} -> After={a_st['recall']:.4f}")
print(f"Stutter F1:        Before={b_st['f1']:.4f} -> After={a_st['f1']:.4f}")
print(f"Stutter (TP/FP/FN/TN): ({b_st['tp']}/{b_st['fp']}/{b_st['fn']}/{b_st['tn']}) -> ({a_st['tp']}/{a_st['fp']}/{a_st['fn']}/{a_st['tn']})")

b_fl = b_exp["filler_binary"]
a_fl = a_exp["filler_binary"]
print(f"Filler Precision:  Before={b_fl['precision']:.4f} -> After={a_fl['precision']:.4f} (+{a_fl['precision'] - b_fl['precision']:.4f})")
print(f"Filler Recall:     Before={b_fl['recall']:.4f} -> After={a_fl['recall']:.4f}")
print(f"Filler F1:         Before={b_fl['f1']:.4f} -> After={a_fl['f1']:.4f}")
print(f"Filler (TP/FP/FN/TN): ({b_fl['tp']}/{b_fl['fp']}/{b_fl['fn']}/{b_fl['tn']}) -> ({a_fl['tp']}/{a_fl['fp']}/{a_fl['fn']}/{a_fl['tn']})")

b_fps = sum(1 for p in before_data["predictions"] if p["gt_has_filler"] == 0 and p["pred_has_filler"] == 1)
a_fps = sum(1 for p in after_data["predictions"] if p["gt_has_filler"] == 0 and p["pred_has_filler"] == 1)
print(f"Filler False Positives: {b_fps} -> {a_fps} (-{b_fps - a_fps})")

# Per-corpus breakdown
print("\n--- 2. Dataset Breakdown of Filler False Positives ---")
for ds in ["UCLASS", "FluencyBank", "Libri_Stutter"]:
    b_ds_fps = sum(1 for p in before_data["predictions"] if p["dataset"] == ds and p["gt_has_filler"] == 0 and p["pred_has_filler"] == 1)
    a_ds_fps = sum(1 for p in after_data["predictions"] if p["dataset"] == ds and p["gt_has_filler"] == 0 and p["pred_has_filler"] == 1)
    print(f"[{ds}] Filler False Positives: Before={b_ds_fps} -> After={a_ds_fps} (Delta: {a_ds_fps - b_ds_fps})")

# 2. Check WER availability
print("\n--- 3. ASR Word Error Rate (WER) ---")
wers_before = [p["wer"] for p in before_data["predictions"] if p.get("wer") is not None]
wers_after = [p["wer"] for p in after_data["predictions"] if p.get("wer") is not None]
if wers_after and wers_before:
    print(f"Mean Dev WER: Before={statistics.mean(wers_before):.4f} -> After={statistics.mean(wers_after):.4f}")
else:
    print("Dev WER: Metric unavailable (Ground truth reference transcripts not provided on 30-clip development manifest).")

print("=" * 70)
