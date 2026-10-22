"""
Evaluate Fix 2 Metrics Script
=============================
Computes side-by-side development metrics comparing Baseline, Fix 1, and Fix 2
for the 30-clip development set, detailing:
- 3-Class Dominant Accuracy & Macro F1
- Binary Stutter (Precision, Recall, F1, TP, FP, FN, TN)
- Binary Filler (Precision, Recall, F1, TP, FP, FN, TN)
- Per-dataset false positive breakdowns
- Genuine stutter events / clips lost
- Timeout and processing status accounting
"""

import os
import json
import statistics

audit_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(audit_dir)))

before_json = os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips_refined_before_fix2.json")
after_json = os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips_refined.json")

with open(before_json, "r", encoding="utf-8") as f:
    b_data = json.load(f)

with open(after_json, "r", encoding="utf-8") as f:
    a_data = json.load(f)

base = a_data["baseline"]
exp_b = b_data["experimental"]
exp_a = a_data["experimental"]

print("=" * 75)
print("DEVELOPMENT BENCHMARK: FIX 1 vs. FIX 2 COMPARISON (30 CLIPS)")
print("=" * 75)

# 1. 3-Class Dominant Classification
b_acc = exp_b["multiclass_3class"]["accuracy"]
a_acc = exp_a["multiclass_3class"]["accuracy"]
base_acc = base["multiclass_3class"]["accuracy"]

b_mf1 = exp_b["multiclass_3class"]["macro_f1"]
a_mf1 = exp_a["multiclass_3class"]["macro_f1"]
base_mf1 = base["multiclass_3class"]["macro_f1"]

print("\n--- 1. 3-Class Dominant Classification ---")
print(f"Accuracy: Baseline={base_acc:.4f} | Fix 1={b_acc:.4f} | Fix 2={a_acc:.4f} (Delta Fix2-Fix1: {a_acc - b_acc:+.4f})")
print(f"Macro F1: Baseline={base_mf1:.4f} | Fix 1={b_mf1:.4f} | Fix 2={a_mf1:.4f} (Delta Fix2-Fix1: {a_mf1 - b_mf1:+.4f})")

# Confusion Matrices
print("\nConfusion Matrix - Fix 1 (Rows: GT [Clean, Stutter, Filler], Cols: Pred):")
for r in exp_b["multiclass_3class"]["confusion_matrix"]:
    print(" ", r)
print("Confusion Matrix - Fix 2 (Rows: GT [Clean, Stutter, Filler], Cols: Pred):")
for r in exp_a["multiclass_3class"]["confusion_matrix"]:
    print(" ", r)

# 2. Stutter Binary
st_b = exp_b["stutter_binary"]
st_a = exp_a["stutter_binary"]
st_base = base["stutter_binary"]

print("\n--- 2. Stutter Binary Metrics ---")
print(f"Precision: Baseline={st_base['precision']:.4f} | Fix 1={st_b['precision']:.4f} -> Fix 2={st_a['precision']:.4f} (Delta: {st_a['precision'] - st_b['precision']:+.4f})")
print(f"Recall:    Baseline={st_base['recall']:.4f} | Fix 1={st_b['recall']:.4f} -> Fix 2={st_a['recall']:.4f} (Delta: {st_a['recall'] - st_b['recall']:+.4f})")
print(f"F1-Score:  Baseline={st_base['f1']:.4f} | Fix 1={st_b['f1']:.4f} -> Fix 2={st_a['f1']:.4f} (Delta: {st_a['f1'] - st_b['f1']:+.4f})")
print(f"Counts (TP/FP/FN/TN): Fix 1=({st_b['tp']}/{st_b['fp']}/{st_b['fn']}/{st_b['tn']}) -> Fix 2=({st_a['tp']}/{st_a['fp']}/{st_a['fn']}/{st_a['tn']})")

# 3. Filler Binary
fl_b = exp_b["filler_binary"]
fl_a = exp_a["filler_binary"]
fl_base = base["filler_binary"]

print("\n--- 3. Filler Binary Metrics ---")
print(f"Precision: Baseline={fl_base['precision']:.4f} | Fix 1={fl_b['precision']:.4f} -> Fix 2={fl_a['precision']:.4f} (Delta: {fl_a['precision'] - fl_b['precision']:+.4f})")
print(f"Recall:    Baseline={fl_base['recall']:.4f} | Fix 1={fl_b['recall']:.4f} -> Fix 2={fl_a['recall']:.4f} (Delta: {fl_a['recall'] - fl_b['recall']:+.4f})")
print(f"F1-Score:  Baseline={fl_base['f1']:.4f} | Fix 1={fl_b['f1']:.4f} -> Fix 2={fl_a['f1']:.4f} (Delta: {fl_a['f1'] - fl_b['f1']:+.4f})")
print(f"Counts (TP/FP/FN/TN): Fix 1=({fl_b['tp']}/{fl_b['fp']}/{fl_b['fn']}/{fl_b['tn']}) -> Fix 2=({fl_a['tp']}/{fl_a['fp']}/{fl_a['fn']}/{fl_a['tn']})")

# 4. False Positives Breakdown by Dataset
preds_b = b_data["predictions"]
preds_a = a_data["predictions"]

print("\n--- 4. Stutter False Positives Breakdown by Dataset ---")
for ds in ["UCLASS", "FluencyBank", "Libri_Stutter"]:
    fp_b = sum(1 for p in preds_b if p["dataset"] == ds and p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 1)
    fp_a = sum(1 for p in preds_a if p["dataset"] == ds and p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 1)
    print(f"[{ds}] Stutter FPs: Fix 1={fp_b} -> Fix 2={fp_a} (Delta: {fp_a - fp_b:+d})")

print("\n--- 5. Filler False Positives Breakdown by Dataset ---")
for ds in ["UCLASS", "FluencyBank", "Libri_Stutter"]:
    fl_fp_b = sum(1 for p in preds_b if p["dataset"] == ds and p["gt_has_filler"] == 0 and p["pred_has_filler"] == 1)
    fl_fp_a = sum(1 for p in preds_a if p["dataset"] == ds and p["gt_has_filler"] == 0 and p["pred_has_filler"] == 1)
    print(f"[{ds}] Filler FPs: Fix 1={fl_fp_b} -> Fix 2={fl_fp_a} (Delta: {fl_fp_a - fl_fp_b:+d})")

# 6. Detailed Genuine Stutter Losses
print("\n--- 6. Analysis of Genuine Stutter Events & Clips ---")
lost_tps = []
retained_tps = []
for pb, pa in zip(preds_b, preds_a):
    if pb["gt_has_stutter"] == 1:
        if pb["pred_has_stutter"] == 1 and pa["pred_has_stutter"] == 0:
            lost_tps.append({
                "filename": pa["filename"],
                "dataset": pa["dataset"],
                "fix1_events": pb["stutter_events"],
                "fix2_events": pa["stutter_events"],
                "fix1_details": pb["details"].get("stutter", []),
                "fix2_details": pa["details"].get("stutter", [])
            })
        elif pb["pred_has_stutter"] == 1 and pa["pred_has_stutter"] == 1:
            retained_tps.append(pa["filename"])

print(f"Retained Stutter True Positives: {len(retained_tps)} clips")
print(f"Lost Stutter True Positives:     {len(lost_tps)} clips")
for item in lost_tps:
    print(f"\n  [Lost TP] {item['filename']} ({item['dataset']}):")
    print(f"    Fix 1 Stutter Events: {item['fix1_events']} -> Fix 2 Stutter Events: {item['fix2_events']}")
    print(f"    Fix 1 Details: {item['fix1_details']}")
    print(f"    Fix 2 Details: {item['fix2_details']}")

# 7. Dissecting the Corrected Stutter False Positives
print("\n--- 7. Analysis of Corrected Stutter False Positives ---")
corrected_fps = []
for pb, pa in zip(preds_b, preds_a):
    if pb["gt_has_stutter"] == 0 and pb["pred_has_stutter"] == 1 and pa["pred_has_stutter"] == 0:
        corrected_fps.append({
            "filename": pa["filename"],
            "dataset": pa["dataset"],
            "gt_dominant": pa["gt_dominant"],
            "fix1_pred": pb["pred_dominant"],
            "fix2_pred": pa["pred_dominant"],
            "fix1_events": pb["stutter_events"],
            "fix2_events": pa["stutter_events"],
            "fix1_details": pb["details"].get("stutter", [])
        })

print(f"Corrected Stutter False Positives: {len(corrected_fps)} clips")
for item in corrected_fps:
    print(f"\n  [Corrected FP] {item['filename']} ({item['dataset']}):")
    print(f"    Ground Truth: {item['gt_dominant']} | Fix 1: {item['fix1_pred']} -> Fix 2: {item['fix2_pred']}")
    print(f"    Fix 1 Spurious Prolongations: {item['fix1_details']}")

# 8. Timeouts and Missing Predictions
print("\n--- 8. Runtime, Timeout & Integrity Verification ---")
print(f"Total clips evaluated: Fix 1={len(preds_b)}, Fix 2={len(preds_a)}")
missing_preds = [p["filename"] for p in preds_a if p.get("status") not in ("SUCCESS", "TIMEOUT")]
print(f"Missing predictions: {len(missing_preds)}")

to_b = [p["filename"] for p in preds_b if p.get("asr_status") == "TIMEOUT" or p.get("status") == "TIMEOUT"]
to_a = [p["filename"] for p in preds_a if p.get("asr_status") == "TIMEOUT" or p.get("status") == "TIMEOUT"]
print(f"Timeouts: Fix 1={len(to_b)} ({to_b}) -> Fix 2={len(to_a)} ({to_a})")

print("=" * 75)
