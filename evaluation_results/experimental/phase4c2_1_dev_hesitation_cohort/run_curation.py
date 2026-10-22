import csv
import json
import sys
import os
import re
from pathlib import Path
from collections import defaultdict

# Add ffmpeg paths
import static_ffmpeg
static_ffmpeg.add_paths()
import whisper

sys.path.insert(0, ".")
from backend.experimental.asr_engine import detect_and_sanitize_repetition_loop

out_dir = Path("evaluation_results/experimental/phase4c2_1_dev_hesitation_cohort")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load eligible manifest
elig_csv_path = out_dir / "eligibility_manifest.csv"
eligible_clips = {}
with open(elig_csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["is_eligible"] == "True":
            eligible_clips[r["filename"]] = r

print(f"Loaded {len(eligible_clips)} eligible clips.", flush=True)

# 2. Load metadata and labels
uclass_meta_path = Path("Datasets/UCLASS Stuttered Speech Clips (SEP-28k Format)/clips/metadata.json")
with open(uclass_meta_path, "r", encoding="utf-8") as f:
    uclass_meta = {Path(item["file_path"]).name: item for item in json.load(f)}

fb_csv = Path("Datasets/Stuttering Events in Podcasts Dataset (SEP-28k)/fluencybank_labels.csv")
fb_labels = {}
with open(fb_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ep = row["EpId"].strip()
        cid = row["ClipId"].strip()
        fn = f"FluencyBank_{ep}_{cid}.wav"
        fb_labels[fn] = row

# Categorize candidate pools
p1_candidates = [] # Fluent controls
p2_candidates = [] # Non-stutter fillers
p3_candidates = [] # Stutter struggle
p4_candidates = [] # High natural pause / potential ASR loop triggers

# UCLASS
for fn, meta in uclass_meta.items():
    if fn in eligible_clips:
        c = eligible_clips[fn]
        labels = meta.get("labels", {})
        is_fluent = meta.get("is_fluent", False)
        has_stutter = (labels.get("Block", 0) > 0 or labels.get("Prolongation", 0) > 0 or 
                       labels.get("SoundRep", 0) > 0 or labels.get("WordRep", 0) > 0)
        has_interj = labels.get("Interjection", 0) > 0
        no_stutter = labels.get("NoStutteredWords", 0) > 0
        
        if is_fluent and no_stutter and not has_stutter and not has_interj:
            p1_candidates.append(c)
        elif not has_stutter and has_interj and no_stutter:
            p2_candidates.append(c)
        elif has_stutter and (labels.get("WordRep", 0) > 0 or labels.get("SoundRep", 0) > 0):
            p3_candidates.append(c)
            
# FluencyBank
for fn, row in fb_labels.items():
    if fn in eligible_clips:
        c = eligible_clips[fn]
        has_stutter = (int(row.get("Block", 0)) > 0 or int(row.get("Prolongation", 0)) > 0 or 
                       int(row.get("SoundRep", 0)) > 0 or int(row.get("WordRep", 0)) > 0)
        has_interj = int(row.get("Interjection", 0)) > 0
        has_pause = int(row.get("NaturalPause", 0)) > 0
        no_stutter = int(row.get("NoStutteredWords", 0)) > 0
        
        if not has_stutter and not has_interj and no_stutter and not has_pause:
            p1_candidates.append(c)
        elif not has_stutter and has_interj and no_stutter:
            p2_candidates.append(c)
        elif has_stutter and (int(row.get("WordRep", 0)) > 0 or int(row.get("SoundRep", 0)) > 0):
            p3_candidates.append(c)
            
        if has_pause or (has_interj and not has_stutter):
            p4_candidates.append(c)

print(f"Pool sizes: P1={len(p1_candidates)}, P2={len(p2_candidates)}, P3={len(p3_candidates)}, P4={len(p4_candidates)}", flush=True)

# Load Whisper model
print("Loading Whisper base model...", flush=True)
model = whisper.load_model("base")
print("Whisper base model loaded.", flush=True)

def transcribe_clip(fpath):
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
    return {
        "raw_transcript": raw_text,
        "sanitized_transcript": loop_res["sanitized_transcript"],
        "words_list": words_list,
        "is_loop": loop_res["is_loop"],
        "sanitizer_status": loop_res["status"],
        "loop_reason": loop_res["reason"],
        "loop_pattern": loop_res["loop_pattern"],
        "truncated_count": loop_res["truncated_tokens_count"]
    }

fillers = {"um", "uh", "er", "erm", "ah", "hmm", "mm", "umm", "uhh", "err", "oh"}
def get_clean_token(t):
    return re.sub(r'[^\w]', '', t).lower()

# 1. Select Cat 1: Fluent Deliberation Controls (10 clips)
print("\n=== Curating Category 1: Fluent Deliberation Controls ===", flush=True)
cat1_clips = []
# Ensure diversity across speakers / sessions
cat1_by_speaker = defaultdict(list)
for c in p1_candidates:
    cat1_by_speaker[c["speaker_or_session"]].append(c)

for spk, spk_clips in sorted(cat1_by_speaker.items()):
    for c in spk_clips[:3]: # at most 2-3 per speaker for diversity
        fpath = c["filepath"]
        t_info = transcribe_clip(fpath)
        words = [get_clean_token(w["word"]) for w in t_info["words_list"]]
        if len(words) >= 3 and not t_info["is_loop"]:
            cat1_clips.append({
                "clip_id": c["filename"],
                "dataset": c["dataset"],
                "speaker_or_session": c["speaker_or_session"],
                "category": "fluent_deliberation_control",
                "ground_truth_class": "Clean",
                "transcription": t_info,
                "filepath": fpath
            })
            print(f"  Cat 1 [{len(cat1_clips)}/10]: {c['filename']} ({spk}) -> '{t_info['raw_transcript']}'", flush=True)
            if len(cat1_clips) == 10:
                break
    if len(cat1_clips) == 10:
        break

# 2. Select Cat 2: Conversational Non-Stutter Filler Chains (10 clips)
print("\n=== Curating Category 2: Conversational Non-Stutter Filler Chains ===", flush=True)
cat2_clips = []
cat2_by_speaker = defaultdict(list)
for c in p2_candidates:
    cat2_by_speaker[c["speaker_or_session"]].append(c)

for spk, spk_clips in sorted(cat2_by_speaker.items()):
    for c in spk_clips[:3]:
        fpath = c["filepath"]
        t_info = transcribe_clip(fpath)
        words = [get_clean_token(w["word"]) for w in t_info["words_list"]]
        f_tokens = [w for w in words if w in fillers]
        if len(f_tokens) >= 1 and not t_info["is_loop"]:
            cat2_clips.append({
                "clip_id": c["filename"],
                "dataset": c["dataset"],
                "speaker_or_session": c["speaker_or_session"],
                "category": "conversational_filler_chain",
                "ground_truth_class": "Filler",
                "filler_tokens": f_tokens,
                "transcription": t_info,
                "filepath": fpath
            })
            print(f"  Cat 2 [{len(cat2_clips)}/10]: {c['filename']} ({spk}) -> '{t_info['raw_transcript']}' (fillers: {f_tokens})", flush=True)
            if len(cat2_clips) == 10:
                break
    if len(cat2_clips) == 10:
        break

# 3. Select Cat 3: Genuine Stuttering Struggle Chains (10 clips)
print("\n=== Curating Category 3: Genuine Stuttering Struggle Chains ===", flush=True)
cat3_clips = []
cat3_by_speaker = defaultdict(list)
for c in p3_candidates:
    cat3_by_speaker[c["speaker_or_session"]].append(c)

for spk, spk_clips in sorted(cat3_by_speaker.items()):
    for c in spk_clips[:4]:
        fpath = c["filepath"]
        t_info = transcribe_clip(fpath)
        words = [get_clean_token(w["word"]) for w in t_info["words_list"]]
        # check repetition or broken fragment
        has_rep = any(words[i] and words[i] == words[i+1] for i in range(len(words)-1))
        if has_rep and not t_info["is_loop"]:
            cat3_clips.append({
                "clip_id": c["filename"],
                "dataset": c["dataset"],
                "speaker_or_session": c["speaker_or_session"],
                "category": "genuine_stutter_struggle",
                "ground_truth_class": "Stutter",
                "transcription": t_info,
                "filepath": fpath
            })
            print(f"  Cat 3 [{len(cat3_clips)}/10]: {c['filename']} ({spk}) -> '{t_info['raw_transcript']}'", flush=True)
            if len(cat3_clips) == 10:
                break
    if len(cat3_clips) == 10:
        break

# 4. Search Cat 4: ASR Repetition-Loop Artifacts (Target 10 clips)
print("\n=== Searching Category 4: ASR Repetition-Loop Artifacts ===", flush=True)
cat4_clips = []
cat4_by_speaker = defaultdict(list)
for c in p4_candidates:
    cat4_by_speaker[c["speaker_or_session"]].append(c)

# Search across p4_candidates
searched_count = 0
for spk, spk_clips in sorted(cat4_by_speaker.items()):
    for c in spk_clips:
        searched_count += 1
        fpath = c["filepath"]
        t_info = transcribe_clip(fpath)
        wl = t_info["words_list"]
        
        zero_dur_count = sum(1 for w in wl if w.get("start") is not None and w.get("end") is not None and w.get("start") == w.get("end"))
        starts = [w.get("start") for w in wl if w.get("start") is not None]
        max_identical_starts = max([starts.count(s) for s in set(starts)]) if starts else 0
        
        is_loop = t_info["is_loop"]
        if is_loop or zero_dur_count >= 3 or max_identical_starts >= 4:
            cat4_clips.append({
                "clip_id": c["filename"],
                "dataset": c["dataset"],
                "speaker_or_session": c["speaker_or_session"],
                "category": "asr_loop_artifact",
                "ground_truth_class": "Artifact",
                "zero_dur_count": zero_dur_count,
                "max_identical_starts": max_identical_starts,
                "transcription": t_info,
                "filepath": fpath
            })
            print(f"  Cat 4 [{len(cat4_clips)}/10]: {c['filename']} ({spk}) -> loop={is_loop}, zero_dur={zero_dur_count}, identical={max_identical_starts} | '{t_info['raw_transcript'][:60]}...'", flush=True)
            if len(cat4_clips) == 10:
                break
        if searched_count % 30 == 0:
            print(f"  Searched {searched_count} clips for Category 4, found {len(cat4_clips)} loops...", flush=True)
        if searched_count >= 150: # Cap screening search
            break
    if len(cat4_clips) == 10 or searched_count >= 150:
        break

print(f"\nSearch complete after {searched_count} clips. Cat 4 count: {len(cat4_clips)}", flush=True)

# Combine results
cohort_records = cat1_clips + cat2_clips + cat3_clips + cat4_clips
with open(out_dir / "curated_cohort_intermediate.json", "w", encoding="utf-8") as f:
    json.dump(cohort_records, f, indent=2)

print(f"\nFinal Cohort Summary: Cat 1 = {len(cat1_clips)}, Cat 2 = {len(cat2_clips)}, Cat 3 = {len(cat3_clips)}, Cat 4 = {len(cat4_clips)}. Total: {len(cohort_records)}", flush=True)
