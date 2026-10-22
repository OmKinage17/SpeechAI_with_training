import csv
import json
import sys
import os
import re
from pathlib import Path
from collections import defaultdict
import static_ffmpeg
static_ffmpeg.add_paths()
import whisper

sys.path.insert(0, ".")
from backend.experimental.asr_engine import detect_and_sanitize_repetition_loop

out_dir = Path("evaluation_results/experimental/phase4c2_1_dev_hesitation_cohort")
elig_csv_path = out_dir / "eligibility_manifest.csv"

eligible_clips = {}
with open(elig_csv_path, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["is_eligible"] == "True":
            eligible_clips[r["filename"]] = r

print(f"Loaded {len(eligible_clips)} eligible clips.", flush=True)

fb_csv = Path("Datasets/Stuttering Events in Podcasts Dataset (SEP-28k)/fluencybank_labels.csv")
fb_labels = {}
with open(fb_csv, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        fn = f"FluencyBank_{r['EpId'].strip()}_{r['ClipId'].strip()}.wav"
        fb_labels[fn] = r

uclass_meta_path = Path("Datasets/UCLASS Stuttered Speech Clips (SEP-28k Format)/clips/metadata.json")
with open(uclass_meta_path, "r", encoding="utf-8") as f:
    uclass_meta = {Path(item["file_path"]).name: item for item in json.load(f)}

print("Loading Whisper model...", flush=True)
model = whisper.load_model("base")
print("Whisper model loaded.", flush=True)

def clean_token(w):
    return re.sub(r'[^\w]', '', w).lower()

fillers = {"um", "uh", "er", "erm", "ah", "hmm", "mm", "umm", "uhh", "err", "oh", "like", "you know"}

def transcribe(fpath):
    res = model.transcribe(
        str(fpath),
        temperature=0.0,
        word_timestamps=True,
        initial_prompt="Um, uh, er, ah, hmm, like, you know, actually, basically, so, well."
    )
    raw_text = res.get("text", "").strip()
    words_list = []
    for seg in res.get("segments", []):
        if "words" in seg:
            words_list.extend(seg["words"])
    loop_res = detect_and_sanitize_repetition_loop(raw_text, words_list, 3.0)
    
    starts = [w.get("start") for w in words_list if w.get("start") is not None]
    zero_dur = sum(1 for w in words_list if w.get("start") is not None and w.get("end") is not None and w.get("start") == w.get("end"))
    max_identical_starts = max([starts.count(s) for s in set(starts)]) if starts else 0
    
    clean_words = [clean_token(w["word"]) for w in words_list]
    has_repeated_filler = False
    for i in range(len(clean_words)-1):
        if clean_words[i] in fillers and clean_words[i] == clean_words[i+1]:
            has_repeated_filler = True
            break
            
    return {
        "raw_transcript": raw_text,
        "sanitized_transcript": loop_res["sanitized_transcript"],
        "words_list": words_list,
        "clean_words": clean_words,
        "is_loop": loop_res["is_loop"],
        "loop_reason": loop_res["reason"],
        "loop_pattern": loop_res["loop_pattern"],
        "truncated_count": loop_res["truncated_tokens_count"],
        "zero_dur_count": zero_dur,
        "max_identical_starts": max_identical_starts,
        "has_repeated_filler": has_repeated_filler
    }

print("\n--- Scanning FB non-stutter interjection clips ---", flush=True)
fb_interj_clips = [eligible_clips[fn] for fn, r in fb_labels.items() 
                   if fn in eligible_clips and int(r.get("Interjection", 0)) > 0 and 
                   int(r.get("Block", 0)) == 0 and int(r.get("Prolongation", 0)) == 0 and 
                   int(r.get("SoundRep", 0)) == 0 and int(r.get("WordRep", 0)) == 0]

print(f"Total FB non-stutter interjection clips: {len(fb_interj_clips)}", flush=True)

results = []
for i, c in enumerate(fb_interj_clips[:60]):
    t_res = transcribe(c["filepath"])
    results.append({
        "clip": c,
        "transcription": t_res
    })
    if t_res["is_loop"] or t_res["zero_dur_count"] >= 3 or t_res["max_identical_starts"] >= 4:
        print(f"POTENTIAL LOOP in FB: {c['filename']} -> is_loop={t_res['is_loop']}, zero_dur={t_res['zero_dur_count']}, id_starts={t_res['max_identical_starts']} | {t_res['raw_transcript'][:60]}", flush=True)
    elif t_res["has_repeated_filler"]:
        print(f"REPEATED FILLER in FB: {c['filename']} -> {t_res['raw_transcript']}", flush=True)
    if (i+1) % 15 == 0:
        print(f"Processed {i+1} clips...", flush=True)

with open(out_dir / "sample_fb_interj_scan.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Scan sample complete.", flush=True)
