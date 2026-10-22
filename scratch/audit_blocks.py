import json
from pathlib import Path

out_dir = Path("evaluation_results/experimental/final_benchmark/run_final_20261020_complete")
with open(out_dir / "run_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)
with open(out_dir / "per_clip_predictions.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

print(f"Total manifest clips: {len(manifest)}")
print(f"Total predictions: {len(preds)}")

# Check block annotations in manifest
block_positive_manifest = [m for m in manifest if m.get("annotation_details", {}).get("block", 0) > 0]
print(f"\nClips with annotation_details.block > 0: {len(block_positive_manifest)}")
for m in block_positive_manifest:
    p = next(x for x in preds if x["filename"] == m["filename"])
    fn = m["filename"]
    ds = m["dataset"]
    gt_dom = m["gt_dominant"]
    ann = m.get("annotation_details")
    b_det = p["blocks"]
    pred_dom = p["pred_dominant"]
    print(f"  {fn} ({ds}) -> gt_dom={gt_dom}, ann={ann}, detected_blocks={b_det}, pred_dom={pred_dom}")

# Check clips where detector found blocks
detected_block_clips = [p for p in preds if p["blocks"] > 0]
print(f"\nClips with detector blocks > 0: {len(detected_block_clips)}")
for p in detected_block_clips:
    m = next(x for x in manifest if x["filename"] == p["filename"])
    fn = p["filename"]
    ds = p["dataset"]
    b_det = p["blocks"]
    gt_dom = p["gt_dominant"]
    ann = m.get("annotation_details")
    pred_dom = p["pred_dominant"]
    print(f"  {fn} ({ds}) -> detected_blocks={b_det}, gt_dom={gt_dom}, ann={ann}, pred_dom={pred_dom}")
