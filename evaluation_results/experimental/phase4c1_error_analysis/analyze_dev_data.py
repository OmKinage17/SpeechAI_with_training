import json
from pathlib import Path
from collections import Counter

pred_path = Path("evaluation_results/experimental/phase4b_filler_gating/per_clip_predictions.json")
with open(pred_path, "r", encoding="utf-8") as f:
    dev_clips = json.load(f)

print(f"Loaded {len(dev_clips)} dev clips.")

# Inspect keys
print("Keys in first clip:", list(dev_clips[0].keys()))

# Check which clips have filler tokens, repeated fillers, or multi-iteration filler detections
filler_clips = []
for c in dev_clips:
    transcript = c.get("transcript", "")
    words = transcript.lower().split()
    cand_details = c.get("candidate", {}).get("details", [])
    
    # Check if candidate fired on multi_iteration_filler_struggle
    multi_filler_events = [d for d in cand_details if d.get("type") == "multi_iteration_filler_struggle"]
    
    # Check words_list for repeated fillers
    wl = c.get("words_list", [])
    
    # Look for any consecutive filler repetitions
    fillers = {"um", "uh", "er", "erm", "ah", "hmm", "mm", "umm", "uhh", "err", "oh"}
    filler_streaks = []
    curr = []
    for w_entry in wl:
        w_text = w_entry.get("word", "").lower().strip(".,?!;:")
        if w_text in fillers:
            if not curr or curr[-1]["word"] == w_text:
                curr.append(w_entry)
            else:
                if len(curr) >= 2:
                    filler_streaks.append(curr)
                curr = [w_entry]
        else:
            if len(curr) >= 2:
                filler_streaks.append(curr)
            curr = []
    if len(curr) >= 2:
        filler_streaks.append(curr)

    if multi_filler_events or filler_streaks or any(w in fillers for w in words):
        filler_clips.append({
            "filename": c["filename"],
            "dataset": c.get("dataset"),
            "gt_has_stutter": c["gt_has_stutter"],
            "gt_has_filler": c.get("gt_has_filler"),
            "gt_dominant": c.get("gt_dominant"),
            "baseline_stutter": c.get("baseline", {}).get("has_stutter"),
            "candidate_stutter": c.get("candidate", {}).get("has_stutter"),
            "baseline_dom": c.get("baseline", {}).get("pred_dominant"),
            "candidate_dom": c.get("candidate", {}).get("pred_dominant"),
            "multi_filler_events": multi_filler_events,
            "filler_streaks_count": len(filler_streaks),
            "max_streak_len": max([len(s) for s in filler_streaks]) if filler_streaks else 0,
            "streaks_info": [
                {
                    "word": s[0]["word"],
                    "count": len(s),
                    "start": s[0].get("start", 0.0),
                    "end": s[-1].get("end", 0.0),
                    "span": round(s[-1].get("end", 0.0) - s[0].get("start", 0.0), 3),
                    "mean_isi": round((s[-1].get("end", 0.0) - s[0].get("start", 0.0)) / (len(s) - 1), 3) if len(s) > 1 else 0.0
                }
                for s in filler_streaks
            ],
            "total_words": len(words),
            "transcript_snippet": transcript[:80]
        })

print(f"Total clips with any filler activity: {len(filler_clips)}")
print(f"Clips with multi_filler_events (candidate fired): {len([c for c in filler_clips if c['multi_filler_events']])}")
print(f"Clips with streak >= 2: {len([c for c in filler_clips if c['filler_streaks_count'] > 0])}")

print("\n--- Details of clips where streak >= 2 ---")
for c in filler_clips:
    if c["filler_streaks_count"] > 0:
        print(f"Clip: {c['filename']} ({c['dataset']}) | GT: St={c['gt_has_stutter']}, Dom={c['gt_dominant']} | Base: {c['baseline_stutter']}, Cand: {c['candidate_stutter']}")
        print(f"  Streaks: {c['streaks_info']}")
        if c["multi_filler_events"]:
            print(f"  Cand fired: {c['multi_filler_events']}")
