import json
from pathlib import Path

out_dir = Path("evaluation_results/experimental/final_benchmark/run_final_20261020_complete")
with open(out_dir / "per_clip_predictions.json", "r", encoding="utf-8") as f:
    preds = json.load(f)
with open(out_dir / "confusion_matrices.json", "r", encoding="utf-8") as f:
    cms = json.load(f)
with open(out_dir / "final_metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

print(f"Total prediction records: {len(preds)}")

# 1. Verify Multiclass Confusion Matrix
classes = ["Clean", "Stutter", "Filler"]
calc_cm = {c1: {c2: 0 for c2 in classes} for c1 in classes}
for p in preds:
    calc_cm[p["gt_dominant"]][p["pred_dominant"]] += 1

cm_match = (calc_cm == cms["aggregate"])
print(f"Multiclass CM Match: {cm_match}")
if not cm_match:
    print("Calculated CM:", calc_cm)
    print("Saved CM:", cms["aggregate"])

# 2. Recompute Multiclass Metrics
total = len(preds)
correct = sum(calc_cm[c][c] for c in classes)
accuracy = correct / total

per_class = {}
f1_list = []
for c in classes:
    tp = calc_cm[c][c]
    fp = sum(calc_cm[other][c] for other in classes if other != c)
    fn = sum(calc_cm[c][other] for other in classes if other != c)
    tn = total - (tp + fp + fn)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    per_class[c] = {"p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4), "spec": round(spec, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    f1_list.append(f1)

macro_f1 = sum(f1_list) / len(f1_list)
print(f"Recomputed Accuracy: {accuracy:.4f} (Saved: {metrics['multiclass_aggregate']['accuracy']})")
print(f"Recomputed Macro F1: {macro_f1:.4f} (Saved: {metrics['multiclass_aggregate']['macro_f1']})")
for c in classes:
    saved_pc = metrics["multiclass_aggregate"]["per_class"][c]
    print(f"Class {c}: P={per_class[c]['p']} (saved {saved_pc['precision']}), R={per_class[c]['r']} (saved {saved_pc['recall']}), F1={per_class[c]['f1']} (saved {saved_pc['f1']}), Spec={per_class[c]['spec']} (saved {saved_pc['specificity']})")

# 3. Verify Binary Stutter Metrics
bin_tp = sum(1 for p in preds if p["gt_has_stutter"] == 1 and p["pred_has_stutter"] == 1)
bin_fp = sum(1 for p in preds if p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 1)
bin_fn = sum(1 for p in preds if p["gt_has_stutter"] == 1 and p["pred_has_stutter"] == 0)
bin_tn = sum(1 for p in preds if p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 0)

b_prec = bin_tp / (bin_tp + bin_fp)
b_rec = bin_tp / (bin_tp + bin_fn)
b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec)
b_spec = bin_tn / (bin_tn + bin_fp)
b_acc = (bin_tp + bin_tn) / total

print(f"\nBinary Stutter: TP={bin_tp}, FP={bin_fp}, FN={bin_fn}, TN={bin_tn}")
print(f"Recomputed Stutter F1: {b_f1:.4f} (Saved: {metrics['binary_stutter_aggregate']['f1']})")
print(f"Recomputed Stutter Prec: {b_prec:.4f} (Saved: {metrics['binary_stutter_aggregate']['precision']})")
print(f"Recomputed Stutter Rec: {b_rec:.4f} (Saved: {metrics['binary_stutter_aggregate']['recall']})")
print(f"Recomputed Stutter Spec: {b_spec:.4f} (Saved: {metrics['binary_stutter_aggregate']['specificity']})")
print(f"Recomputed Stutter Acc: {b_acc:.4f} (Saved: {metrics['binary_stutter_aggregate']['accuracy']})")

# 4. Check Per-Dataset Consistency
for ds in ["UCLASS", "FluencyBank", "Libri_Stutter"]:
    ds_preds = [p for p in preds if p["dataset"] == ds]
    ds_tp = sum(1 for p in ds_preds if p["gt_has_stutter"] == 1 and p["pred_has_stutter"] == 1)
    ds_fp = sum(1 for p in ds_preds if p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 1)
    ds_fn = sum(1 for p in ds_preds if p["gt_has_stutter"] == 1 and p["pred_has_stutter"] == 0)
    ds_tn = sum(1 for p in ds_preds if p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 0)
    print(f"\nCorpus {ds} (N={len(ds_preds)}): TP={ds_tp}, FP={ds_fp}, FN={ds_fn}, TN={ds_tn}")
    print(f"  Stutter Prec: {ds_tp/(ds_tp+ds_fp):.4f}, Rec: {ds_tp/(ds_tp+ds_fn):.4f}, F1: {2*ds_tp/(2*ds_tp+ds_fp+ds_fn):.4f}, Spec: {ds_tn/(ds_tn+ds_fp):.4f}")
