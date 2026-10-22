import csv
import json
import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")
from backend.experimental.asr_engine import ExperimentalASREngine, detect_and_sanitize_repetition_loop

# Output directory
out_dir = Path("evaluation_results/experimental/phase4c2_1_dev_hesitation_cohort")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Exclusions
locked_manifest_path = Path("evaluation_results/experimental/final_benchmark/run_final_20261020_complete/run_manifest.json")
with open(locked_manifest_path, "r", encoding="utf-8") as f:
    locked_items = json.load(f)
locked_filenames = set(item["filename"] for item in locked_items)

dev30_path = Path("evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json")
with open(dev30_path, "r", encoding="utf-8") as f:
    dev30_filenames = set(json.load(f).keys())

gen60_path = Path("evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json")
with open(gen60_path, "r", encoding="utf-8") as f:
    gen60_filenames = set(item["filename"] for item in json.load(f))

dev90_filenames = dev30_filenames.union(gen60_filenames)

holdout_uclass_spks = {'M_1099', 'M_1023', 'M_1104', 'M_1105'}
holdout_fb_eps = {'063', '089', '178', '099'}

dev90_uclass_spks = {'M_0030', 'M_0061', 'M_1022'}
dev90_fb_eps = {'010', '017', '031'}

print(f"Exclusions: {len(locked_filenames)} locked holdout clips, {len(dev90_filenames)} dev90 clips.")

# 2. Build Eligibility Database
uclass_meta_path = Path("Datasets/UCLASS Stuttered Speech Clips (SEP-28k Format)/clips/metadata.json")
uclass_clips_dir = Path("Datasets/UCLASS Stuttered Speech Clips (SEP-28k Format)/clips/clips")

with open(uclass_meta_path, "r", encoding="utf-8") as f:
    uclass_meta = json.load(f)

fb_csv = Path("Datasets/Stuttering Events in Podcasts Dataset (SEP-28k)/fluencybank_labels.csv")
fb_clips_dir = Path("Datasets/Stuttering Events in Podcasts Dataset (SEP-28k)/clips/stuttering-clips/clips")

eligibility_records = []

# Scan UCLASS
for item in uclass_meta:
    fn = Path(item["file_path"]).name
    fpath = uclass_clips_dir / fn
    parts = fn.split("_")
    spk = f"{parts[0]}_{parts[1]}"
    
    is_eligible = True
    exclusion_reason = "ELIGIBLE"
    
    if fn in locked_filenames:
        is_eligible = False
        exclusion_reason = "EXCLUDED_LOCKED_HOLDOUT_CLIP"
    elif fn in dev90_filenames:
        is_eligible = False
        exclusion_reason = "EXCLUDED_DEV90_CLIP"
    elif spk in holdout_uclass_spks:
        is_eligible = False
        exclusion_reason = "EXCLUDED_HOLDOUT_SPEAKER_OVERLAP"
    elif spk in dev90_uclass_spks:
        is_eligible = False
        exclusion_reason = "EXCLUDED_DEV90_SPEAKER_OVERLAP"
    elif not fpath.exists():
        is_eligible = False
        exclusion_reason = "AUDIO_FILE_MISSING"
        
    eligibility_records.append({
        "filename": fn,
        "dataset": "UCLASS",
        "speaker_or_session": spk,
        "filepath": str(fpath),
        "is_eligible": is_eligible,
        "exclusion_reason": exclusion_reason,
        "labels": item["labels"],
        "is_fluent": item.get("is_fluent", False)
    })

# Scan FluencyBank
with open(fb_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ep = row["EpId"].strip()
        cid = row["ClipId"].strip()
        fn = f"FluencyBank_{ep}_{cid}.wav"
        fpath = fb_clips_dir / f"{ep}_{cid}.wav"
        if not fpath.exists():
            fpath = fb_clips_dir / fn
            
        is_eligible = True
        exclusion_reason = "ELIGIBLE"
        
        if fn in locked_filenames:
            is_eligible = False
            exclusion_reason = "EXCLUDED_LOCKED_HOLDOUT_CLIP"
        elif fn in dev90_filenames:
            is_eligible = False
            exclusion_reason = "EXCLUDED_DEV90_CLIP"
        elif ep in holdout_fb_eps:
            is_eligible = False
            exclusion_reason = "EXCLUDED_HOLDOUT_SESSION_OVERLAP"
        elif ep in dev90_fb_eps:
            is_eligible = False
            exclusion_reason = "EXCLUDED_DEV90_SESSION_OVERLAP"
        elif int(row.get("PoorAudioQuality", 0)) > 0:
            is_eligible = False
            exclusion_reason = "EXCLUDED_POOR_AUDIO_QUALITY"
        elif int(row.get("Unsure", 0)) > 0:
            is_eligible = False
            exclusion_reason = "EXCLUDED_ANNOTATOR_UNSURE"
        elif int(row.get("NoSpeech", 0)) > 0:
            is_eligible = False
            exclusion_reason = "EXCLUDED_NO_SPEECH"
        elif int(row.get("Music", 0)) > 0:
            is_eligible = False
            exclusion_reason = "EXCLUDED_MUSIC"
        elif not fpath.exists():
            is_eligible = False
            exclusion_reason = "AUDIO_FILE_MISSING"
            
        eligibility_records.append({
            "filename": fn,
            "dataset": "FluencyBank",
            "speaker_or_session": ep,
            "filepath": str(fpath),
            "is_eligible": is_eligible,
            "exclusion_reason": exclusion_reason,
            "labels": {
                "Block": int(row.get("Block", 0)),
                "Prolongation": int(row.get("Prolongation", 0)),
                "SoundRep": int(row.get("SoundRep", 0)),
                "WordRep": int(row.get("WordRep", 0)),
                "Interjection": int(row.get("Interjection", 0)),
                "NoStutteredWords": int(row.get("NoStutteredWords", 0))
            },
            "is_fluent": False
        })

print(f"Total evaluated clips in pool: {len(eligibility_records)}")
eligible_count = sum(1 for r in eligibility_records if r["is_eligible"])
print(f"Total eligible clips: {eligible_count}")

# Write eligibility manifest
elig_csv_path = out_dir / "eligibility_manifest.csv"
with open(elig_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "filename", "dataset", "speaker_or_session", "is_eligible", "exclusion_reason", "filepath"
    ])
    writer.writeheader()
    for r in eligibility_records:
        writer.writerow({
            "filename": r["filename"],
            "dataset": r["dataset"],
            "speaker_or_session": r["speaker_or_session"],
            "is_eligible": r["is_eligible"],
            "exclusion_reason": r["exclusion_reason"],
            "filepath": r["filepath"]
        })

print(f"Saved eligibility manifest to {elig_csv_path}")
