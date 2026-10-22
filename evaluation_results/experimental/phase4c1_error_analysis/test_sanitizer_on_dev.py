import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backend.experimental.asr_engine import detect_and_sanitize_repetition_loop

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

loop_clips = []
for fn, c in all_dev.items():
    raw_text = c.get("transcript", "")
    raw_wl = c.get("words_list", [])
    dur = c.get("audio_duration_sec", 3.0)
    
    loop_res = detect_and_sanitize_repetition_loop(raw_text, raw_wl, dur)
    if loop_res["is_loop"]:
        loop_clips.append({
            "filename": fn,
            "split": c["split"],
            "dataset": c.get("dataset"),
            "gt_has_stutter": c.get("gt_has_stutter"),
            "gt_dominant": c.get("gt_dominant"),
            "raw_words_count": len(raw_text.split()),
            "sanitized_words_count": len(loop_res["sanitized_transcript"].split()),
            "truncated_count": loop_res["truncated_tokens_count"],
            "reason": loop_res["reason"],
            "loop_pattern": loop_res["loop_pattern"],
            "raw_snippet": raw_text[:80],
            "sanitized_text": loop_res["sanitized_transcript"]
        })

print(f"=== Loop Sanitizer on All 90 Development Clips ===")
print(f"Total clips flagged as loops: {len(loop_clips)} / {len(all_dev)}")
for lc in loop_clips:
    print(f"\nClip: {lc['filename']} ({lc['split']} - {lc['dataset']})")
    print(f"  GT Stutter: {lc['gt_has_stutter']}, Dominant: {lc['gt_dominant']}")
    print(f"  Raw words: {lc['raw_words_count']} -> Sanitized words: {lc['sanitized_words_count']} (Truncated: {lc['truncated_count']})")
    print(f"  Pattern: {lc['loop_pattern']}")
    print(f"  Reason: {lc['reason']}")
    print(f"  Sanitized Text: {lc['sanitized_text']}")
