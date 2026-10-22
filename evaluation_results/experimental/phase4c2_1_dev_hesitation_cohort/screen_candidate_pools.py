import csv
import json
import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")
from backend.experimental.asr_engine import ExperimentalASREngine, detect_and_sanitize_repetition_loop

# Load eligible manifest
elig_csv_path = Path("evaluation_results/experimental/phase4c2_1_dev_hesitation_cohort/eligibility_manifest.csv")
eligible_clips = []
with open(elig_csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["is_eligible"] == "True":
            eligible_clips.append(r)

print(f"Total eligible clips available: {len(eligible_clips)}")

# Load original labels to categorize candidate pools
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

# Categorize eligible candidates into candidate pools
pool_cat1 = [] # Fluent controls
pool_cat2 = [] # Non-stutter fillers
pool_cat3 = [] # Stutter struggle
pool_cat4 = [] # Candidates for ASR loop search (e.g. low speech, trailing silence, or interjection)

for c in eligible_clips:
    fn = c["filename"]
    ds = c["dataset"]
    fpath = c["filepath"]
    spk = c["speaker_or_session"]
    
    if ds == "UCLASS":
        meta = uclass_meta.get(fn, {})
        labels = meta.get("labels", {})
        is_fluent = meta.get("is_fluent", False)
        
        has_stutter = (labels.get("Block", 0) > 0 or labels.get("Prolongation", 0) > 0 or 
                       labels.get("SoundRep", 0) > 0 or labels.get("WordRep", 0) > 0)
        has_interj = labels.get("Interjection", 0) > 0
        no_stutter = labels.get("NoStutteredWords", 0) > 0
        
        if is_fluent and no_stutter and not has_stutter and not has_interj:
            pool_cat1.append((fn, fpath, ds, spk, labels))
        elif not has_stutter and has_interj and no_stutter:
            pool_cat2.append((fn, fpath, ds, spk, labels))
        elif has_stutter and (labels.get("WordRep", 0) > 0 or labels.get("SoundRep", 0) > 0):
            pool_cat3.append((fn, fpath, ds, spk, labels))
            
    elif ds == "FluencyBank":
        row = fb_labels.get(fn, {})
        has_stutter = (int(row.get("Block", 0)) > 0 or int(row.get("Prolongation", 0)) > 0 or 
                       int(row.get("SoundRep", 0)) > 0 or int(row.get("WordRep", 0)) > 0)
        has_interj = int(row.get("Interjection", 0)) > 0
        no_stutter = int(row.get("NoStutteredWords", 0)) > 0
        
        if not has_stutter and not has_interj and no_stutter:
            pool_cat1.append((fn, fpath, ds, spk, row))
        elif not has_stutter and has_interj and no_stutter:
            pool_cat2.append((fn, fpath, ds, spk, row))
        elif has_stutter and (int(row.get("WordRep", 0)) > 0 or int(row.get("SoundRep", 0)) > 0):
            pool_cat3.append((fn, fpath, ds, spk, row))

print(f"Pool Cat 1 (Fluent Controls): {len(pool_cat1)} clips")
print(f"Pool Cat 2 (Non-Stutter Fillers): {len(pool_cat2)} clips")
print(f"Pool Cat 3 (Stutter Struggle): {len(pool_cat3)} clips")
