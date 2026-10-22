import json
from pathlib import Path
from collections import Counter

dev30_path = Path("evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json")
gen60_path = Path("evaluation_results/experimental/generalization_validation/run_20261020_132214/feature_cache.json")

with open(dev30_path, "r", encoding="utf-8") as f:
    dev30_cache = json.load(f)
with open(gen60_path, "r", encoding="utf-8") as f:
    gen60_cache = json.load(f)

print(f"Loaded {len(dev30_cache)} dev30 clips, {len(gen60_cache)} gen60 clips. Total: {len(dev30_cache) + len(gen60_cache)}")

all_dev = {}
for fn, c in dev30_cache.items():
    all_dev[fn] = dict(c, split="dev_30")
for fn, c in gen60_cache.items():
    all_dev[fn] = dict(c, split="gen_60")

# 1. Check ASR Status across all 90 dev clips
asr_statuses = Counter(c.get("asr_status") for c in all_dev.values())
print("\n=== ASR Status Distribution across 90 Dev Clips ===")
for status, count in asr_statuses.items():
    print(f"  {status}: {count}")

# 2. Check token counts across all 90 dev clips
word_counts = [(c["filename"], c.get("words_count", len(c.get("words_list", []))), c.get("gt_has_stutter"), c.get("gt_dominant")) for c in all_dev.values()]
word_counts.sort(key=lambda x: x[1], reverse=True)
print("\n=== Top 10 Longest Transcripts in 90 Dev Clips ===")
for fn, wc, gt_st, gt_dom in word_counts[:10]:
    print(f"  {fn}: {wc} words | GT Stutter: {gt_st}, Dominant: {gt_dom}")

# 3. Analyze filler tokens and filler streaks in words_list
fillers = {"um", "uh", "er", "erm", "ah", "hmm", "mm", "umm", "uhh", "err", "oh"}

def get_clean_word(w_str):
    import re
    return re.sub(r'[^\w]', '', w_str).lower()

clips_with_filler_streaks = []
for fn, c in all_dev.items():
    wl = c.get("words_list", [])
    streaks = []
    curr = []
    for w in wl:
        w_clean = get_clean_word(w.get("word", ""))
        if w_clean in fillers:
            if not curr or curr[-1]["clean"] == w_clean:
                curr.append(dict(w, clean=w_clean))
            else:
                if len(curr) >= 2:
                    streaks.append(curr)
                curr = [dict(w, clean=w_clean)]
        else:
            if len(curr) >= 2:
                streaks.append(curr)
            curr = []
    if len(curr) >= 2:
        streaks.append(curr)
        
    # Check for identical timestamps or zero duration
    zero_dur_count = sum(1 for w in wl if w.get("start") is not None and w.get("end") is not None and w.get("start") == w.get("end"))
    
    if streaks or zero_dur_count > 0:
        clips_with_filler_streaks.append({
            "filename": fn,
            "split": c["split"],
            "dataset": c.get("dataset"),
            "words_count": len(wl),
            "gt_has_stutter": c.get("gt_has_stutter"),
            "gt_dominant": c.get("gt_dominant"),
            "zero_dur_tokens": zero_dur_count,
            "streaks": [
                {
                    "word": s[0]["clean"],
                    "count": len(s),
                    "start": s[0].get("start"),
                    "end": s[-1].get("end"),
                    "span": round(s[-1].get("end", 0.0) - s[0].get("start", 0.0), 3) if s[0].get("start") is not None else None,
                    "all_zero_dur": all(w.get("start") == w.get("end") for w in s),
                    "identical_start": len(set(w.get("start") for w in s)) == 1
                }
                for s in streaks
            ]
        })

print(f"\n=== Clips with Filler Streaks (>=2) or Zero-Duration Tokens ({len(clips_with_filler_streaks)} clips) ===")
for c in clips_with_filler_streaks:
    print(f"\nClip: {c['filename']} ({c['split']} - {c['dataset']}) | Total Words: {c['words_count']} | Zero-dur tokens: {c['zero_dur_tokens']}")
    print(f"  GT Stutter: {c['gt_has_stutter']}, GT Dominant: {c['gt_dominant']}")
    for s in c['streaks']:
        print(f"  Streak: {s['count']}x '{s['word']}' | start={s['start']}, end={s['end']}, span={s['span']}s | all_zero_dur={s['all_zero_dur']}, identical_start={s['identical_start']}")
