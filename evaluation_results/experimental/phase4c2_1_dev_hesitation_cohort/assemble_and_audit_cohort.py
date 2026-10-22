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
        "sanitizer_status": loop_res["status"]
    }

# Load previously scanned results
with open(out_dir / "sample_fb_interj_scan.json", "r", encoding="utf-8") as f:
    sample_fb = json.load(f)

with open(out_dir / "cat1_cat3_candidates.json", "r", encoding="utf-8") as f:
    c1_c3 = json.load(f)

print(f"Loaded existing samples: {len(sample_fb)} FB interj, {len(c1_c3['cat1'])} Cat 1, {len(c1_c3['cat3'])} Cat 3")

# Let's find more Cat 2 (Conversational non-stutter filler chains)
# We need 10 clips with conversational hesitation / filler chains from non-stutter ground truth
cat2_clips = []
# From sample_fb:
for item in sample_fb:
    c = item["clip"]
    t = item["transcription"]
    raw = t["raw_transcript"]
    # Look for clips that have fillers ('um', 'uh', 'hmm', etc.) and are NOT loops
    cw = t["clean_words"]
    has_fillers = any(w in {"um", "uh", "er", "erm", "ah", "hmm"} for w in cw)
    if has_fillers and not t["is_loop"] and t["zero_dur_count"] <= 2 and len(cw) >= 2:
        cat2_clips.append({
            "clip": c,
            "transcription": t,
            "category": "conversational_filler_chain",
            "ground_truth_class": "Filler"
        })
        print(f"Cat 2 candidate: {c['filename']} ({c['speaker_or_session']}) -> {raw}")
        if len(cat2_clips) >= 12:
            break

# If we need more Cat 2, scan additional FB non-stutter clips
if len(cat2_clips) < 10:
    print(f"Scanning more for Cat 2 (currently {len(cat2_clips)})...")
    fb_interj_clips = [eligible_clips[fn] for fn, r in fb_labels.items() 
                       if fn in eligible_clips and int(r.get("Interjection", 0)) > 0 and 
                       int(r.get("Block", 0)) == 0 and int(r.get("Prolongation", 0)) == 0 and 
                       int(r.get("SoundRep", 0)) == 0 and int(r.get("WordRep", 0)) == 0]
    for c in fb_interj_clips[60:120]:
        t = transcribe(c["filepath"])
        cw = t["clean_words"]
        has_fillers = any(w in {"um", "uh", "er", "erm", "ah", "hmm"} for w in cw)
        if has_fillers and not t["is_loop"] and t["zero_dur_count"] <= 2 and len(cw) >= 2:
            cat2_clips.append({
                "clip": c,
                "transcription": t,
                "category": "conversational_filler_chain",
                "ground_truth_class": "Filler"
            })
            print(f"Cat 2 candidate: {c['filename']} ({c['speaker_or_session']}) -> {t['raw_transcript']}")
            if len(cat2_clips) >= 12:
                break

# Check Cat 3 count: we need 10
cat3_clips = []
for item in c1_c3["cat3"]:
    cat3_clips.append({
        "clip": item["clip"],
        "transcription": item["transcription"],
        "category": "genuine_stutter_struggle",
        "ground_truth_class": "Stutter"
    })

# If Cat 3 has 9, find 1-3 more from UCLASS dysfluent eligible speakers (e.g. M_0553 or M_1103)
if len(cat3_clips) < 10:
    print(f"Scanning UCLASS for Cat 3 to reach target (currently {len(cat3_clips)})...")
    uclass_dys = [eligible_clips[fn] for fn, m in uclass_meta.items()
                  if fn in eligible_clips and not m.get("is_fluent", False)]
    for c in uclass_dys:
        m = uclass_meta[c["filename"]]
        labels = m.get("labels", {})
        if labels.get("WordRep", 0) > 0 or labels.get("SoundRep", 0) > 0:
            t = transcribe(c["filepath"])
            cw = t["clean_words"]
            has_rep = any(cw[i] and cw[i] == cw[i+1] for i in range(len(cw)-1))
            if has_rep and not t["is_loop"] and t["zero_dur_count"] <= 2:
                cat3_clips.append({
                    "clip": c,
                    "transcription": t,
                    "category": "genuine_stutter_struggle",
                    "ground_truth_class": "Stutter"
                })
                print(f"Cat 3 UCLASS candidate: {c['filename']} ({c['speaker_or_session']}) -> {t['raw_transcript']}")
                if len(cat3_clips) >= 12:
                    break

# Select Cat 1 (Fluent deliberation controls)
cat1_clips = []
# Prefer clips with deliberation markers, fluent pausing, or clean phrasing
for item in c1_c3["cat1"]:
    cat1_clips.append({
        "clip": item["clip"],
        "transcription": item["transcription"],
        "category": "fluent_deliberation_control",
        "ground_truth_class": "Clean"
    })

# Select Cat 4 (ASR repetition-loop artifacts)
cat4_clips = []
# From sample_fb: select 10 loops across different episodes/patterns
for item in sample_fb:
    c = item["clip"]
    t = item["transcription"]
    # We want confirmed loop: is_loop=True or zero_dur_count >= 10
    if (t["is_loop"] or t["zero_dur_count"] >= 50) and len(cat4_clips) < 10:
        cat4_clips.append({
            "clip": c,
            "transcription": t,
            "category": "asr_loop_artifact",
            "ground_truth_class": "Artifact"
        })
        print(f"Cat 4 selected: {c['filename']} ({c['speaker_or_session']}) -> zero_dur={t['zero_dur_count']}, is_loop={t['is_loop']} | {t['raw_transcript'][:60]}")

print(f"\nFinal Cohort Assembly Status:")
print(f"  Category 1 (Fluent deliberation controls): {len(cat1_clips[:10])}")
print(f"  Category 2 (Conversational non-stutter filler chains): {len(cat2_clips[:10])}")
print(f"  Category 3 (Genuine stuttering struggle chains): {len(cat3_clips[:10])}")
print(f"  Category 4 (ASR repetition-loop artifacts): {len(cat4_clips[:10])}")
total_curated = len(cat1_clips[:10]) + len(cat2_clips[:10]) + len(cat3_clips[:10]) + len(cat4_clips[:10])
print(f"  TOTAL: {total_curated}/40")

curated_all = {
    "cat1": cat1_clips[:10],
    "cat2": cat2_clips[:10],
    "cat3": cat3_clips[:10],
    "cat4": cat4_clips[:10]
}

with open(out_dir / "final_curated_pools.json", "w", encoding="utf-8") as f:
    json.dump(curated_all, f, indent=2)

print("Saved final_curated_pools.json successfully.")
