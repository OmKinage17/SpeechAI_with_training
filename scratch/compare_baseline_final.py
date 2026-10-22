import json

p_base_pred = "evaluation_results/experimental/expanded_validation/run_expanded_120/per_clip_predictions.json"
p_final_pred = "evaluation_results/experimental/final_benchmark/run_final_20261020_complete/per_clip_predictions.json"

with open(p_base_pred, "r", encoding="utf-8") as f:
    base_preds = json.load(f)
with open(p_final_pred, "r", encoding="utf-8") as f:
    final_preds = json.load(f)

base_map = {p["filename"]: p for p in base_preds}
final_map = {p["filename"]: p for p in final_preds}

# Identify the 13 timeout clips
timeout_fns = [fn for fn, bp in base_map.items() if bp.get("timed_out") or bp.get("classification") == "Timeout" or bp.get("transcript") == ""]
print(f"Total baseline timeouts: {len(timeout_fns)}")

print("\n--- PERFORMANCE ON THE 13 TIMEOUT CLIPS ---")
print("Baseline outcomes on the 13 clips:")
b_stutter_tp, b_stutter_fp, b_stutter_fn, b_stutter_tn = 0, 0, 0, 0
for fn in timeout_fns:
    bp = base_map[fn]
    gt_st = bp["gt_has_stutter"]
    pr_st = bp.get("pred_has_stutter", False)
    if gt_st and pr_st: b_stutter_tp += 1
    elif not gt_st and pr_st: b_stutter_fp += 1
    elif gt_st and not pr_st: b_stutter_fn += 1
    else: b_stutter_tn += 1
print(f"  Baseline 13 clips: TP={b_stutter_tp}, FP={b_stutter_fp}, FN={b_stutter_fn}, TN={b_stutter_tn}")

print("\nFinal outcomes on the 13 previously-timed-out clips:")
f_stutter_tp, f_stutter_fp, f_stutter_fn, f_stutter_tn = 0, 0, 0, 0
for fn in timeout_fns:
    fp = final_map[fn]
    gt_st = fp["gt_has_stutter"]
    pr_st = fp.get("pred_has_stutter", False)
    if gt_st and pr_st: f_stutter_tp += 1
    elif not gt_st and pr_st: f_stutter_fp += 1
    elif gt_st and not pr_st: f_stutter_fn += 1
    else: f_stutter_tn += 1
print(f"  Final 13 clips:    TP={f_stutter_tp}, FP={f_stutter_fp}, FN={f_stutter_fn}, TN={f_stutter_tn}")
print(f"  Delta on 13 clips: TP={f_stutter_tp - b_stutter_tp:+d}, FP={f_stutter_fp - b_stutter_fp:+d}, FN={f_stutter_fn - b_stutter_fn:+d}, TN={f_stutter_tn - b_stutter_tn:+d}")

print("\n--- PERFORMANCE ON THE 107 NON-TIMEOUT CLIPS ---")
non_timeout_fns = [fn for fn in base_map if fn not in timeout_fns]
b_nt_tp, b_nt_fp, b_nt_fn, b_nt_tn = 0, 0, 0, 0
f_nt_tp, f_nt_fp, f_nt_fn, f_nt_tn = 0, 0, 0, 0

for fn in non_timeout_fns:
    bp = base_map[fn]
    gt_st = bp["gt_has_stutter"]
    pr_st = bp.get("pred_has_stutter", False)
    if gt_st and pr_st: b_nt_tp += 1
    elif not gt_st and pr_st: b_nt_fp += 1
    elif gt_st and not pr_st: b_nt_fn += 1
    else: b_nt_tn += 1

    fp = final_map[fn]
    f_pr_st = fp.get("pred_has_stutter", False)
    if gt_st and f_pr_st: f_nt_tp += 1
    elif not gt_st and f_pr_st: f_nt_fp += 1
    elif gt_st and not f_pr_st: f_nt_fn += 1
    else: f_nt_tn += 1

print(f"  Baseline 107 clips: TP={b_nt_tp}, FP={b_nt_fp}, FN={b_nt_fn}, TN={b_nt_tn}")
print(f"  Final 107 clips:    TP={f_nt_tp}, FP={f_nt_fp}, FN={f_nt_fn}, TN={f_nt_tn}")
print(f"  Delta on 107 clips: TP={f_nt_tp - b_nt_tp:+d}, FP={f_nt_fp - b_nt_fp:+d}, FN={f_nt_fn - b_nt_fn:+d}, TN={f_nt_tn - b_nt_tn:+d}")

print("\n--- OVERALL (120 CLIPS) ---")
print(f"  Baseline (120): TP={b_stutter_tp + b_nt_tp}, FP={b_stutter_fp + b_nt_fp}, FN={b_stutter_fn + b_nt_fn}, TN={b_stutter_tn + b_nt_tn}")
print(f"  Final (120):    TP={f_stutter_tp + f_nt_tp}, FP={f_stutter_fp + f_nt_fp}, FN={f_stutter_fn + f_nt_fn}, TN={f_stutter_tn + f_nt_tn}")
