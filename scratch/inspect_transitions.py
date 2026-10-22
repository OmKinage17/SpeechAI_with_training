import json

base_dir = "evaluation_results/experimental/expanded_validation/run_expanded_120"
final_dir = "evaluation_results/experimental/final_benchmark/run_final_20261020_complete"

with open(f"{base_dir}/per_clip_predictions.json", "r", encoding="utf-8") as f:
    base_preds = json.load(f)
with open(f"{final_dir}/per_clip_predictions.json", "r", encoding="utf-8") as f:
    final_preds = json.load(f)

base_map = {p["filename"]: p for p in base_preds}
final_map = {p["filename"]: p for p in final_preds}

print("=== MULTICLASS ACCURACY TRANSITIONS ===")
mc_gained = []
mc_lost = []
for fn in base_map:
    b_corr = base_map[fn]["is_correct_multiclass"]
    f_corr = final_map[fn]["is_correct_multiclass"]
    if not b_corr and f_corr:
        mc_gained.append(fn)
    elif b_corr and not f_corr:
        mc_lost.append(fn)

print(f"Clips that became multiclass correct (+1): {len(mc_gained)}")
for fn in mc_gained:
    bp = base_map[fn]
    fp = final_map[fn]
    print(f"  {fn} ({fp['dataset']}): GT={fp['gt_dominant']} | BasePred={bp.get('pred_dominant')} -> FinalPred={fp['pred_dominant']}")

print(f"\nClips that became multiclass incorrect (-1): {len(mc_lost)}")
for fn in mc_lost:
    bp = base_map[fn]
    fp = final_map[fn]
    print(f"  {fn} ({fp['dataset']}): GT={fp['gt_dominant']} | BasePred={bp.get('pred_dominant')} -> FinalPred={fp['pred_dominant']}")

print(f"Net multiclass gain: {len(mc_gained) - len(mc_lost)} clips (64 -> 71 clips, +5.833%)")

print("\n=== BINARY STUTTERING TRANSITIONS ===")
bin_gained_tp = []
bin_lost_tp = []
for fn in base_map:
    bp = base_map[fn]
    fp = final_map[fn]
    gt_st = fp["gt_has_stutter"]
    b_pred_st = bp.get("pred_has_stutter", False)
    f_pred_st = fp.get("pred_has_stutter", False)
    if gt_st:
        if not b_pred_st and f_pred_st:
            bin_gained_tp.append(fn)
        elif b_pred_st and not f_pred_st:
            bin_lost_tp.append(fn)

print(f"Clips that gained Stutter True Positive (+1 TP): {len(bin_gained_tp)}")
for fn in bin_gained_tp:
    fp = final_map[fn]
    bp = base_map[fn]
    timed_out = bp.get("timed_out") or bp.get("classification") == "Timeout" or bp.get("transcript") == ""
    print(f"  {fn} ({fp['dataset']}): timed_out_in_baseline={timed_out}")

print(f"Clips that lost Stutter True Positive (-1 TP): {len(bin_lost_tp)}")
print(f"Net TP gain: {len(bin_gained_tp) - len(bin_lost_tp)} (40 -> 43 TP)")
