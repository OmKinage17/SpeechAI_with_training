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

print(f"Loaded {len(eligible_clips)} eligible clips.")

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
# Pool 1: Fluent deliberation controls
# Sourced from fluent UCLASS speakers (e.g. M_0078, M_0121, M_0553, M_1100, M_1101, M_1103, M_1106) where is_fluent is True and no stutter labels exist.
p1_candidates = []
for fn, meta in uclass_meta.items():
    if fn in eligible_clips and meta.get("is_fluent", False):
        labels = meta.get("labels", {})
        if (labels.get("Block", 0) == 0 and labels.get("Prolongation", 0) == 0 and 
            labels.get("SoundRep", 0) == 0 and labels.get("WordRep", 0) == 0):
            p1_candidates.append(eligible_clips[fn])

# Pool 2: Conversational non-stutter filler chains
# Sourced from FluencyBank unspent episodes and UCLASS where Interjection >= 1 and no stutter events.
p2_candidates = []
for fn, row in fb_labels.items():
    if fn in eligible_clips:
        has_stutter = (int(row.get("Block", 0)) > 0 or int(row.get("Prolongation", 0)) > 0 or 
                       int(row.get("SoundRep", 0)) > 0 or int(row.get("WordRep", 0)) > 0)
        has_interj = int(row.get("Interjection", 0)) > 0
        no_stutter = int(row.get("NoStutteredWords", 0)) > 0
        if not has_stutter and has_interj and no_stutter:
            p2_candidates.append(eligible_clips[fn])

for fn, meta in uclass_meta.items():
    if fn in eligible_clips:
        labels = meta.get("labels", {})
        has_stutter = (labels.get("Block", 0) > 0 or labels.get("Prolongation", 0) > 0 or 
                       labels.get("SoundRep", 0) > 0 or labels.get("WordRep", 0) > 0)
        has_interj = labels.get("Interjection", 0) > 0
        no_stutter = labels.get("NoStutteredWords", 0) > 0
        if not has_stutter and has_interj and no_stutter:
            p2_candidates.append(eligible_clips[fn])

# Pool 3: Genuine stuttering struggle chains
# Sourced from unspent UCLASS and FluencyBank where WordRep >= 1 or SoundRep >= 1 or Block >= 1
p3_candidates = []
for fn, meta in uclass_meta.items():
    if fn in eligible_clips:
        labels = meta.get("labels", {})
        if labels.get("WordRep", 0) > 0 or labels.get("SoundRep", 0) > 0:
            p3_candidates.append(eligible_clips[fn])

for fn, row in fb_labels.items():
    if fn in eligible_clips:
        if int(row.get("WordRep", 0)) > 0 or int(row.get("SoundRep", 0)) > 0:
            p3_candidates.append(eligible_clips[fn])

print(f"P1 (Fluent) size: {len(p1_candidates)}")
print(f"P2 (Non-stutter fillers) size: {len(p2_candidates)}")
print(f"P3 (Stutter struggle) size: {len(p3_candidates)}")

# Initialize whisper model once in-process
print("Loading Whisper base model...")
model = whisper.load_model("base")
print("Model loaded successfully.")

# Helper to transcribe a clip directly
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

# 3. Select Category 1: Fluent Deliberation Controls (10 clips)
# Target: Fluent controls from unspent UCLASS fluent takes (e.g. M_0078, M_0121, M_1100, M_1103)
print("\n--- Selecting Category 1 (Fluent Deliberation Controls) ---")
selected_cat1 = []
for c in p1_candidates:
    fpath = c["filepath"]
    t_info = transcribe_clip(fpath)
    text = t_info["raw_transcript"]
    wl = t_info["words_list"]
    clean_words = [get_clean_token(w.get("word", "")) for w in wl]
    
    # Check for presence of fluent speech or deliberation hesitation
    if len(clean_words) >= 3 and not t_info["is_loop"]:
        has_filler = any(w in fillers for w in clean_words)
        selected_cat1.append({
            "clip_id": c["filename"],
            "dataset": c["dataset"],
            "speaker_or_session": c["speaker_or_session"],
            "category": "fluent_deliberation_control",
            "ground_truth_class": "Clean",
            "has_filler": has_filler,
            "transcription": t_info,
            "filepath": fpath
        })
        print(f"  Cat 1 [{len(selected_cat1)}/10]: {c['filename']} ({c['speaker_or_session']}) -> '{text}'")
        if len(selected_cat1) == 10:
            break

# 4. Select Category 2: Conversational Non-Stutter Filler Chains (10 clips)
# Target: Conversational hesitations containing filler words from unspent FB episodes / UCLASS
print("\n--- Selecting Category 2 (Conversational Non-Stutter Filler Chains) ---")
selected_cat2 = []
for c in p2_candidates:
    fpath = c["filepath"]
    t_info = transcribe_clip(fpath)
    text = t_info["raw_transcript"]
    wl = t_info["words_list"]
    clean_words = [get_clean_token(w.get("word", "")) for w in wl]
    
    filler_tokens = [w for w in clean_words if w in fillers]
    if len(filler_tokens) >= 1 and not t_info["is_loop"]:
        selected_cat2.append({
            "clip_id": c["filename"],
            "dataset": c["dataset"],
            "speaker_or_session": c["speaker_or_session"],
            "category": "conversational_filler_chain",
            "ground_truth_class": "Filler",
            "filler_count": len(filler_tokens),
            "transcription": t_info,
            "filepath": fpath
        })
        print(f"  Cat 2 [{len(selected_cat2)}/10]: {c['filename']} ({c['speaker_or_session']}) -> '{text}' (fillers: {filler_tokens})")
        if len(selected_cat2) == 10:
            break

# 5. Select Category 3: Genuine Stuttering Struggle Chains (10 clips)
# Target: Clinical perseverations (word or sound repetitions) from unspent UCLASS / FB dysfluent sessions
print("\n--- Selecting Category 3 (Genuine Stuttering Struggle Chains) ---")
selected_cat3 = []
for c in p3_candidates:
    fpath = c["filepath"]
    t_info = transcribe_clip(fpath)
    text = t_info["raw_transcript"]
    wl = t_info["words_list"]
    clean_words = [get_clean_token(w.get("word", "")) for w in wl]
    
    # Check for repeated words or audible stuttering repetitions
    has_rep = False
    for i in range(len(clean_words) - 1):
        if clean_words[i] and clean_words[i] == clean_words[i+1]:
            has_rep = True
            break
            
    if has_rep and not t_info["is_loop"]:
        selected_cat3.append({
            "clip_id": c["filename"],
            "dataset": c["dataset"],
            "speaker_or_session": c["speaker_or_session"],
            "category": "genuine_stutter_struggle",
            "ground_truth_class": "Stutter",
            "transcription": t_info,
            "filepath": fpath
        })
        print(f"  Cat 3 [{len(selected_cat3)}/10]: {c['filename']} ({c['speaker_or_session']}) -> '{text}'")
        if len(selected_cat3) == 10:
            break

# 6. Select Category 4: ASR Repetition-Loop Artifacts (10 clips)
# Search across all eligible clips for clips where is_loop is True OR zero-duration token collapse >= 3
print("\n--- Searching for Category 4 (ASR Repetition-Loop Artifacts) ---")
selected_cat4 = []
all_sample = list(eligible_clips.values())
for c in all_sample:
    fpath = c["filepath"]
    fn = c["filename"]
    
    # Skip if already selected in cat1-3
    if any(s["clip_id"] == fn for s in selected_cat1 + selected_cat2 + selected_cat3):
        continue
        
    t_info = transcribe_clip(fpath)
    wl = t_info["words_list"]
    
    # Check for loop sanitizer trigger OR zero-duration token collapse density >= 3
    is_loop = t_info["is_loop"]
    zero_dur_count = sum(1 for w in wl if w.get("start") is not None and w.get("end") is not None and w.get("start") == w.get("end"))
    
    # Check for identical timestamps
    starts = [w.get("start") for w in wl if w.get("start") is not None]
    max_identical_starts = max([starts.count(s) for s in set(starts)]) if starts else 0
    
    if is_loop or zero_dur_count >= 3 or max_identical_starts >= 4:
        selected_cat4.append({
            "clip_id": fn,
            "dataset": c["dataset"],
            "speaker_or_session": c["speaker_or_session"],
            "category": "asr_loop_artifact",
            "ground_truth_class": "Artifact",
            "zero_dur_count": zero_dur_count,
            "max_identical_starts": max_identical_starts,
            "transcription": t_info,
            "filepath": fpath
        })
        print(f"  Cat 4 [{len(selected_cat4)}/10]: {fn} ({c['speaker_or_session']}) -> loop={is_loop}, zero_dur={zero_dur_count}, identical={max_identical_starts} | '{t_info['raw_transcript'][:60]}...'")
        if len(selected_cat4) == 10:
            break

print(f"\nFinal Cohort Selection Complete:")
print(f"  Cat 1: {len(selected_cat1)}")
print(f"  Cat 2: {len(selected_cat2)}")
print(f"  Cat 3: {len(selected_cat3)}")
print(f"  Cat 4: {len(selected_cat4)}")

all_selected = selected_cat1 + selected_cat2 + selected_cat3 + selected_cat4
with open(out_dir / "curated_raw_cohort.json", "w", encoding="utf-8") as f:
    json.dump(all_selected, f, indent=2)

print("Saved curated raw cohort to curated_raw_cohort.json")
