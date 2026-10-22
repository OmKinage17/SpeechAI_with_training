import csv
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, ".")
from backend.experimental.asr_engine import ExperimentalASREngine, detect_and_sanitize_repetition_loop

def main():
    elig_csv_path = Path("evaluation_results/experimental/phase4c2_1_dev_hesitation_cohort/eligibility_manifest.csv")
    eligible_clips = []
    with open(elig_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["is_eligible"] == "True":
                eligible_clips.append(r)

    print(f"Loaded {len(eligible_clips)} eligible clips.")

    # Prioritize clips with natural pauses or interjections (known Whisper loop triggers)
    fb_csv = Path("Datasets/Stuttering Events in Podcasts Dataset (SEP-28k)/fluencybank_labels.csv")
    pause_clips = set()
    with open(fb_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if int(r.get("NaturalPause", 0)) > 0 or int(r.get("Interjection", 0)) > 0:
                pause_clips.add(f"FluencyBank_{r['EpId'].strip()}_{r['ClipId'].strip()}.wav")

    print(f"Clips with pause/interjection markers in FB: {len(pause_clips)}")

    # Sort eligible clips so pause/interjection clips come first
    sample_clips = sorted(eligible_clips, key=lambda c: 0 if c["filename"] in pause_clips else 1)
    
    asr = ExperimentalASREngine()

    loop_clips_found = []
    tested = 0

    print("Starting scan for ASR loop clips...")
    for c in sample_clips:
        fpath = c["filepath"]
        fn = c["filename"]
        
        try:
            res = asr.transcribe(fpath, timeout_sec=10.0)
            raw_text = res.get("transcript", "")
            raw_wl = res.get("words_list", [])
            tested += 1
            
            loop_res = detect_and_sanitize_repetition_loop(raw_text, raw_wl, 3.0)
            if loop_res["is_loop"]:
                print(f"FOUND LOOP ({len(loop_clips_found)+1}/10): {fn} ({c['dataset']}) | Reason: {loop_res['reason']}")
                loop_clips_found.append({
                    "clip": c,
                    "raw_text": raw_text,
                    "sanitized_text": loop_res["sanitized_transcript"],
                    "reason": loop_res["reason"],
                    "pattern": loop_res["loop_pattern"],
                    "truncated": loop_res["truncated_tokens_count"],
                    "words_list": raw_wl
                })
                if len(loop_clips_found) >= 12:
                    break
                    
            if tested % 25 == 0:
                print(f"Tested {tested} clips, found {len(loop_clips_found)} loops...")
                
        except Exception as e:
            continue

    print(f"\nScan completed. Tested {tested} clips, found {len(loop_clips_found)} ASR loop clips.")
    with open("evaluation_results/experimental/phase4c2_1_dev_hesitation_cohort/discovered_loops.json", "w", encoding="utf-8") as f:
        json.dump(loop_clips_found, f, indent=2)

if __name__ == "__main__":
    main()
