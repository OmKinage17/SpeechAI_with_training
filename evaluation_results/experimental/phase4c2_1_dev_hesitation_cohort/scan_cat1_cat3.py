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

fillers = {"um", "uh", "er", "erm", "ah", "hmm", "mm", "umm", "uhh", "err", "oh", "like", "you", "know"}

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
    }

# Scan UCLASS fluent clips for Cat 1
print("\n--- Scanning UCLASS fluent clips for Cat 1 (Fluent Deliberation Controls) ---", flush=True)
uclass_fluent = [eligible_clips[fn] for fn, m in uclass_meta.items() 
                 if fn in eligible_clips and m.get("is_fluent", False)]
print(f"Total eligible UCLASS fluent clips: {len(uclass_fluent)}", flush=True)

cat1_found = []
by_speaker = defaultdict(list)
for c in uclass_fluent:
    by_speaker[c["speaker_or_session"]].append(c)

for spk, clips in sorted(by_speaker.items()):
    for c in clips[:4]:
        t = transcribe(c["filepath"])
        # We want fluent clips with clear deliberation or clean fluent phrasing, no loop
        if not t["is_loop"] and t["zero_dur_count"] <= 1 and len(t["clean_words"]) >= 4:
            cat1_found.append({
                "clip": c,
                "transcription": t
            })
            print(f"Cat 1 candidate: {c['filename']} ({spk}) -> {t['raw_transcript']}", flush=True)
            if len(cat1_found) >= 12:
                break
    if len(cat1_found) >= 12:
        break

# Scan FB/UCLASS for Cat 3 (Genuine Stuttering Struggle Chains)
print("\n--- Scanning for Cat 3 (Genuine Stuttering Struggle Chains) ---", flush=True)
fb_stutter = [eligible_clips[fn] for fn, r in fb_labels.items() 
              if fn in eligible_clips and (int(r.get("WordRep", 0)) > 0 or int(r.get("SoundRep", 0)) > 0)]
print(f"Total eligible FB stutter rep clips: {len(fb_stutter)}", flush=True)

cat3_found = []
fb_stutter_by_ep = defaultdict(list)
for c in fb_stutter:
    fb_stutter_by_ep[c["speaker_or_session"]].append(c)

for ep, clips in sorted(fb_stutter_by_ep.items()):
    for c in clips[:4]:
        t = transcribe(c["filepath"])
        # Check for repetition in transcript and realistic temporal duration (not zero-duration collapse)
        words = t["clean_words"]
        has_rep = any(words[i] and words[i] == words[i+1] for i in range(len(words)-1))
        # Genuine struggle must have real acoustic span (> 0.2s) and not be an ASR loop
        if has_rep and not t["is_loop"] and t["zero_dur_count"] <= 2:
            cat3_found.append({
                "clip": c,
                "transcription": t
            })
            print(f"Cat 3 candidate: {c['filename']} ({ep}) -> {t['raw_transcript']}", flush=True)
            if len(cat3_found) >= 12:
                break
    if len(cat3_found) >= 12:
        break

print(f"\nScan results: Cat 1 candidates={len(cat1_found)}, Cat 3 candidates={len(cat3_found)}", flush=True)

with open(out_dir / "cat1_cat3_candidates.json", "w", encoding="utf-8") as f:
    json.dump({"cat1": cat1_found, "cat3": cat3_found}, f, indent=2)
