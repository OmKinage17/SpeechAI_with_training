import json
import csv
import hashlib
from pathlib import Path
import numpy as np

run_dir = Path("evaluation_results/experimental/final_benchmark/run_final_20261020_complete")
base_dir = Path("evaluation_results/experimental/expanded_validation/run_expanded_120")

# 1. Load artifacts
with open(run_dir / "run_manifest.json", "r", encoding="utf-8") as f:
    final_manifest = json.load(f)
with open(base_dir / "run_manifest.json", "r", encoding="utf-8") as f:
    base_manifest = json.load(f)
with open(run_dir / "per_clip_predictions.json", "r", encoding="utf-8") as f:
    final_preds = json.load(f)
with open(base_dir / "per_clip_predictions.json", "r", encoding="utf-8") as f:
    base_preds = json.load(f)
with open(run_dir / "final_metrics.json", "r", encoding="utf-8") as f:
    final_metrics = json.load(f)
with open(run_dir / "per_parameter_metrics.json", "r", encoding="utf-8") as f:
    per_param = json.load(f)
with open(run_dir / "confusion_matrices.json", "r", encoding="utf-8") as f:
    cm_data = json.load(f)

print("=== 1. MANIFEST & COMPARABILITY AUDIT ===")
base_ids = [m["filename"] for m in base_manifest]
final_ids = [m["filename"] for m in final_manifest]
print(f"Identical clip count: {len(base_ids) == len(final_ids) == 120}")
print(f"Identical clip IDs: {base_ids == final_ids}")

# Verify SHA256 of manifest
with open(run_dir / "run_manifest.json", "rb") as f:
    h_final = hashlib.sha256(f.read()).hexdigest()
with open(base_dir / "run_manifest.json", "rb") as f:
    h_base = hashlib.sha256(f.read()).hexdigest()
print(f"Manifest SHA256 match: {h_final == h_base} ({h_final})")

# Verify ground truth labels match
gt_match = True
for b, f_m in zip(base_manifest, final_manifest):
    if b["gt_dominant"] != f_m["gt_dominant"] or b["gt_has_stutter"] != f_m["gt_has_stutter"] or b["gt_has_filler"] != f_m["gt_has_filler"]:
        gt_match = False
        break
print(f"Ground truth labels identical between baseline and final: {gt_match}")

print("\n=== 2. METRIC RECOMPUTATION FROM per_clip_predictions.json ===")
classes = ["Clean", "Stutter", "Filler"]

# Multiclass
cm_recomputed = {c: {c2: 0 for c2 in classes} for c in classes}
for p in final_preds:
    cm_recomputed[p["gt_dominant"]][p["pred_dominant"]] += 1

print("Recomputed Multiclass CM:")
for c in classes:
    print(f"  {c}: {cm_recomputed[c]}")

correct_mc = sum(cm_recomputed[c][c] for c in classes)
acc_mc = correct_mc / len(final_preds)
print(f"Recomputed Accuracy: {acc_mc:.4f} (Saved: {final_metrics['multiclass_aggregate']['accuracy']:.4f})")

per_class = {}
macro_f1_sum = 0.0
for c in classes:
    tp = cm_recomputed[c][c]
    fp = sum(cm_recomputed[c_other][c] for c_other in classes if c_other != c)
    fn = sum(cm_recomputed[c][c_other] for c_other in classes if c_other != c)
    tn = sum(cm_recomputed[c1][c2] for c1 in classes if c1 != c for c2 in classes if c2 != c)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    per_class[c] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "p": p, "r": r, "f1": f1, "spec": spec}
    macro_f1_sum += f1
    print(f"  Class {c}: TP={tp}, FP={fp}, FN={fn}, TN={tn} | P={p:.4f}, R={r:.4f}, F1={f1:.4f}, Spec={spec:.4f}")

macro_f1 = macro_f1_sum / len(classes)
print(f"Recomputed Macro F1: {macro_f1:.4f} (Saved: {final_metrics['multiclass_aggregate']['macro_f1']:.4f})")

# Binary Stutter
bin_tp = sum(1 for p in final_preds if p["gt_has_stutter"] and p["pred_has_stutter"])
bin_fp = sum(1 for p in final_preds if not p["gt_has_stutter"] and p["pred_has_stutter"])
bin_fn = sum(1 for p in final_preds if p["gt_has_stutter"] and not p["pred_has_stutter"])
bin_tn = sum(1 for p in final_preds if not p["gt_has_stutter"] and not p["pred_has_stutter"])

bin_p = bin_tp / (bin_tp + bin_fp)
bin_r = bin_tp / (bin_tp + bin_fn)
bin_f1 = 2 * bin_p * bin_r / (bin_p + bin_r)
bin_spec = bin_tn / (bin_tn + bin_fp)
bin_acc = (bin_tp + bin_tn) / len(final_preds)
print(f"\nBinary Stutter: TP={bin_tp}, FP={bin_fp}, FN={bin_fn}, TN={bin_tn}")
print(f"  Acc={bin_acc:.4f}, P={bin_p:.4f}, R={bin_r:.4f}, F1={bin_f1:.4f}, Spec={bin_spec:.4f}")

# Filler Event Detection
fill_tp = 0
fill_fp = 0
fill_fn = 0
fill_tn = 0
for p, m in zip(final_preds, final_manifest):
    is_gt_filler = 1 if p["gt_dominant"] == "Filler" or m.get("annotation_details", {}).get("interjection", 0) > 0 else 0
    is_pr_filler = 1 if p.get("filler_count", 0) > 0 else 0
    if is_gt_filler and is_pr_filler: fill_tp += 1
    elif not is_gt_filler and is_pr_filler: fill_fp += 1
    elif is_gt_filler and not is_pr_filler: fill_fn += 1
    else: fill_tn += 1

fill_p = fill_tp / (fill_tp + fill_fp)
fill_r = fill_tp / (fill_tp + fill_fn)
fill_f1 = 2 * fill_p * fill_r / (fill_p + fill_r)
fill_spec = fill_tn / (fill_tn + fill_fp)
fill_acc = (fill_tp + fill_tn) / len(final_preds)
print(f"\nFiller Event: TP={fill_tp}, FP={fill_fp}, FN={fill_fn}, TN={fill_tn}")
print(f"  Acc={fill_acc:.4f}, P={fill_p:.4f}, R={fill_r:.4f}, F1={fill_f1:.4f}, Spec={fill_spec:.4f}")

print("\n=== 3. ANNOTATION INTEGRITY & SILENT NEGATIVE BIAS AUDIT ===")
# Check which datasets have annotations for each dysfluency subtype
for ds in ["UCLASS", "FluencyBank", "Libri_Stutter"]:
    ds_manifest = [m for m in final_manifest if m["dataset"] == ds]
    # Check block annotation keys
    has_block_key = any("block" in m.get("annotation_details", {}) for m in ds_manifest)
    block_pos = sum(1 for m in ds_manifest if m.get("annotation_details", {}).get("block", 0) > 0)
    # Check interjection key
    has_inj_key = any("interjection" in m.get("annotation_details", {}) for m in ds_manifest)
    inj_pos = sum(1 for m in ds_manifest if m.get("annotation_details", {}).get("interjection", 0) > 0)
    print(f"Dataset {ds} (N={len(ds_manifest)}):")
    print(f"  Block annotated schema: {has_block_key} | Positive block count: {block_pos}")
    print(f"  Filler/Interjection annotated schema: {has_inj_key} | Positive filler count: {inj_pos}")

print("\nBlock Evaluation Scoping Breakdown:")
# Scoped (UCLASS + FluencyBank)
scoped_manifest = [m for m in final_manifest if m["dataset"] in ["UCLASS", "FluencyBank"]]
scoped_preds = [p for p in final_preds if p["dataset"] in ["UCLASS", "FluencyBank"]]
bl_tp = sum(1 for p, m in zip(scoped_preds, scoped_manifest) if m.get("annotation_details", {}).get("block", 0) > 0 and p.get("blocks", 0) > 0)
bl_fp = sum(1 for p, m in zip(scoped_preds, scoped_manifest) if not (m.get("annotation_details", {}).get("block", 0) > 0) and p.get("blocks", 0) > 0)
bl_fn = sum(1 for p, m in zip(scoped_preds, scoped_manifest) if m.get("annotation_details", {}).get("block", 0) > 0 and not (p.get("blocks", 0) > 0))
bl_tn = sum(1 for p, m in zip(scoped_preds, scoped_manifest) if not (m.get("annotation_details", {}).get("block", 0) > 0) and not (p.get("blocks", 0) > 0))

bl_spec = bl_tn / (bl_tn + bl_fp)
bl_acc = (bl_tp + bl_tn) / len(scoped_manifest)
print(f"  Scoped UCLASS+FB (N=80): TP={bl_tp}, FP={bl_fp}, FN={bl_fn}, TN={bl_tn} | Spec={bl_spec:.4f}, Acc={bl_acc:.4f}")

# Libri_Stutter unannotated check
libri_manifest = [m for m in final_manifest if m["dataset"] == "Libri_Stutter"]
libri_preds = [p for p in final_preds if p["dataset"] == "Libri_Stutter"]
libri_block_preds = sum(1 for p in libri_preds if p.get("blocks", 0) > 0)
print(f"  Libri_Stutter (N=40): Block Ground Truth: NONE (Schema Omission) | Acoustic Detections: {libri_block_preds}")
