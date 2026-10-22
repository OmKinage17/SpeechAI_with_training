"""
Independent recalculation and audit script for Phase 4B per-clip predictions.
"""
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
pred_file = project_root / "evaluation_results" / "experimental" / "phase4b_filler_gating" / "per_clip_predictions.json"
final_metrics_file = project_root / "evaluation_results" / "experimental" / "phase4b_filler_gating" / "final_metrics.json"

with open(pred_file, "r", encoding="utf-8") as f:
    preds = json.load(f)

with open(final_metrics_file, "r", encoding="utf-8") as f:
    saved_metrics = json.load(f)

print(f"Total clips in per_clip_predictions: {len(preds)}")
filenames = [p["filename"] for p in preds]
print(f"Unique filenames: {len(set(filenames))}")

# Split counts
dev30_count = sum(1 for p in preds if p["split"] == "dev_30")
gen60_count = sum(1 for p in preds if p["split"] == "gen_60")
print(f"Split breakdown: dev_30 = {dev30_count}, gen_60 = {gen60_count}")

# Recompute binary metrics directly from records
base_tp = sum(1 for p in preds if p["gt_has_stutter"] == 1 and p["baseline"]["has_stutter"] == 1)
base_fp = sum(1 for p in preds if p["gt_has_stutter"] == 0 and p["baseline"]["has_stutter"] == 1)
base_fn = sum(1 for p in preds if p["gt_has_stutter"] == 1 and p["baseline"]["has_stutter"] == 0)
base_tn = sum(1 for p in preds if p["gt_has_stutter"] == 0 and p["baseline"]["has_stutter"] == 0)

cand_tp = sum(1 for p in preds if p["gt_has_stutter"] == 1 and p["candidate"]["has_stutter"] == 1)
cand_fp = sum(1 for p in preds if p["gt_has_stutter"] == 0 and p["candidate"]["has_stutter"] == 1)
cand_fn = sum(1 for p in preds if p["gt_has_stutter"] == 1 and p["candidate"]["has_stutter"] == 0)
cand_tn = sum(1 for p in preds if p["gt_has_stutter"] == 0 and p["candidate"]["has_stutter"] == 0)

print(f"Baseline Binary:  TP={base_tp}, FP={base_fp}, FN={base_fn}, TN={base_tn}")
print(f"Candidate Binary: TP={cand_tp}, FP={cand_fp}, FN={cand_fn}, TN={cand_tn}")

b_prec = base_tp / (base_tp + base_fp) if (base_tp + base_fp) else 0.0
b_rec = base_tp / (base_tp + base_fn) if (base_tp + base_fn) else 0.0
b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec) if (b_prec + b_rec) else 0.0
b_spec = base_tn / (base_tn + base_fp) if (base_tn + base_fp) else 0.0

c_prec = cand_tp / (cand_tp + cand_fp) if (cand_tp + cand_fp) else 0.0
c_rec = cand_tp / (cand_tp + cand_fn) if (cand_tp + cand_fn) else 0.0
c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) else 0.0
c_spec = cand_tn / (cand_tn + cand_fp) if (cand_tn + cand_fp) else 0.0

print(f"Baseline Prec={b_prec:.4f}, Rec={b_rec:.4f}, F1={b_f1:.4f}, Spec={b_spec:.4f}")
print(f"Candidate Prec={c_prec:.4f}, Rec={c_rec:.4f}, F1={c_f1:.4f}, Spec={c_spec:.4f}")

# Verify exact match with saved_metrics
saved_st = saved_metrics["stutter_metrics"]
assert saved_st["baseline"]["tp"] == base_tp, "Baseline TP mismatch"
assert saved_st["baseline"]["fp"] == base_fp, "Baseline FP mismatch"
assert saved_st["baseline"]["fn"] == base_fn, "Baseline FN mismatch"
assert saved_st["baseline"]["tn"] == base_tn, "Baseline TN mismatch"
assert saved_st["candidate"]["tp"] == cand_tp, "Candidate TP mismatch"
assert saved_st["candidate"]["fp"] == cand_fp, "Candidate FP mismatch"
assert saved_st["candidate"]["fn"] == cand_fn, "Candidate FN mismatch"
assert saved_st["candidate"]["tn"] == cand_tn, "Candidate TN mismatch"
print(">>> Exact match verified for all binary confusion counts.")

# Dominant 3-class confusion matrices
labels = ["Stutter", "Filler", "Clean"]
base_cm = {r: {c: 0 for c in labels} for r in labels}
cand_cm = {r: {c: 0 for c in labels} for r in labels}

for p in preds:
    gt = p["gt_dominant"]
    b_pred = p["baseline"]["pred_dominant"]
    c_pred = p["candidate"]["pred_dominant"]
    base_cm[gt][b_pred] += 1
    cand_cm[gt][c_pred] += 1

print("\nBaseline 3x3 CM:")
for r in labels:
    print(f"  {r:<8}: {base_cm[r]}")

print("\nCandidate 3x3 CM:")
for r in labels:
    print(f"  {r:<8}: {cand_cm[r]}")

# Changed clips audit
changed = [p for p in preds if p["prediction_changed"]]
print(f"\nTotal clips with prediction_changed == True: {len(changed)}")
for c in changed:
    print(f"  Clip: {c['filename']} | GT: {c['gt_has_stutter']} (dom: {c['gt_dominant']})")
    print(f"    Baseline:  stutter={c['baseline']['has_stutter']}, dom={c['baseline']['pred_dominant']}, conf={c['baseline']['confidence']}")
    print(f"    Candidate: stutter={c['candidate']['has_stutter']}, dom={c['candidate']['pred_dominant']}, conf={c['candidate']['confidence']}")
    print(f"    Reason:    {c['change_reason']}")

# Verify that for all other clips, candidate and baseline predictions are 100% identical
non_changed = [p for p in preds if not p["prediction_changed"]]
print(f"Total non-changed clips: {len(non_changed)}")
for p in non_changed:
    assert p["baseline"]["has_stutter"] == p["candidate"]["has_stutter"], f"Inconsistent stutter prediction on {p['filename']}"
    assert p["baseline"]["pred_dominant"] == p["candidate"]["pred_dominant"], f"Inconsistent dominant prediction on {p['filename']}"

print(">>> Verified that all 88 non-changed clips have 100% identical predictions.")
