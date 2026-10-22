import json

with open("evaluation_results/experimental/phase4b_holdout/per_clip_predictions.json") as f:
    data = json.load(f)

for c in data:
    if c["baseline"]["has_stutter"] != c["candidate"]["has_stutter"]:
        fn = c["filename"]
        gt_dom = c["gt_dominant"]
        gt_st = c["gt_has_stutter"]
        b_dom = c["baseline"]["pred_dominant"]
        b_st = c["baseline"]["has_stutter"]
        c_dom = c["candidate"]["pred_dominant"]
        c_st = c["candidate"]["has_stutter"]
        print(f"{fn} | GT: dom={gt_dom}, st={gt_st} | Base: dom={b_dom}, st={b_st} | Cand: dom={c_dom}, st={c_st} | Reason: {c['change_reason']}")
