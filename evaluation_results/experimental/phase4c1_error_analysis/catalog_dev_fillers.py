import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

dev30_path = Path("evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json")
gen60_path = Path("evaluation_results/experimental/generalization_validation/run_20261020_132214/feature_cache.json")

with open(dev30_path, "r", encoding="utf-8") as f:
    dev30_cache = json.load(f)
with open(gen60_path, "r", encoding="utf-8") as f:
    gen60_cache = json.load(f)

all_dev = {}
for fn, c in dev30_cache.items():
    all_dev[fn] = dict(c, split="dev_30")
for fn, c in gen60_cache.items():
    all_dev[fn] = dict(c, split="gen_60")

fillers = {"um", "uh", "er", "erm", "ah", "hmm", "mm", "umm", "uhh", "err", "oh"}
import re
def clean_w(w):
    return re.sub(r'[^\w]', '', w).lower()

all_filler_instances = []
for fn, c in all_dev.items():
    wl = c.get("words_list", [])
    transcript = c.get("transcript", "")
    
    # count filler tokens
    filler_tokens = [w for w in wl if clean_w(w.get("word", "")) in fillers]
    
    if filler_tokens:
        all_filler_instances.append({
            "filename": fn,
            "split": c["split"],
            "dataset": c.get("dataset"),
            "gt_has_stutter": c.get("gt_has_stutter"),
            "gt_has_filler": c.get("gt_has_filler"),
            "gt_dominant": c.get("gt_dominant"),
            "total_words": len(wl),
            "filler_count": len(filler_tokens),
            "filler_words": [clean_w(w["word"]) for w in filler_tokens],
            "silence_intervals_count": c.get("silence_intervals_count", 0),
            "raw_candidates_count": c.get("raw_candidates_count", 0),
            "raw_candidates": c.get("raw_candidates", [])
        })

print(f"=== Filler Instances across 90 Development Clips ===")
print(f"Total clips with at least 1 filler token: {len(all_filler_instances)} / 90")

by_gt = Counter(c["gt_dominant"] for c in all_filler_instances)
print(f"Distribution of GT dominant: {by_gt}")
by_stutter = Counter(c["gt_has_stutter"] for c in all_filler_instances)
print(f"Distribution of GT has_stutter: {by_stutter}")

print("\n--- Detailed breakdown of all clips with filler tokens in 90 dev clips ---")
for c in all_filler_instances:
    print(f"{c['filename']} ({c['dataset']}) | GT: Stutter={c['gt_has_stutter']}, Dom={c['gt_dominant']} | Total words: {c['total_words']} | Fillers ({c['filler_count']}): {c['filler_words']}")
    if c['raw_candidates']:
        c_types = [cand.get('type') for cand in c['raw_candidates']]
        print(f"  Raw acoustic candidates: {c_types}")
