import csv
import json
from pathlib import Path
from collections import defaultdict

# Exclusions
holdout_uclass_spks = {'M_1099', 'M_1023', 'M_1104', 'M_1105'}
holdout_fb_eps = {'063', '089', '178', '099'}

dev90_uclass_spks = {'M_0030', 'M_0061', 'M_1022'}
dev90_fb_eps = {'010', '017', '031'}

# 1. Load UCLASS metadata
uclass_meta_path = Path("Datasets/UCLASS Stuttered Speech Clips (SEP-28k Format)/clips/metadata.json")
with open(uclass_meta_path, "r", encoding="utf-8") as f:
    uclass_meta = json.load(f)

uclass_candidates = defaultdict(list)
for item in uclass_meta:
    fn = Path(item["file_path"]).name
    parts = fn.split("_")
    spk = f"{parts[0]}_{parts[1]}"
    
    # Check exclusions
    if spk in holdout_uclass_spks or spk in dev90_uclass_spks:
        continue
    
    labels = item["labels"]
    is_fluent = item.get("is_fluent", False)
    has_stutter = (labels.get("Block", 0) > 0 or labels.get("Prolongation", 0) > 0 or 
                   labels.get("SoundRep", 0) > 0 or labels.get("WordRep", 0) > 0)
    has_interj = labels.get("Interjection", 0) > 0
    no_stutter = labels.get("NoStutteredWords", 0) > 0
    
    if is_fluent and not has_stutter and not has_interj:
        uclass_candidates["fluent_controls"].append((fn, item))
    elif not has_stutter and has_interj:
        uclass_candidates["non_stutter_interj"].append((fn, item))
    elif has_stutter and (labels.get("WordRep", 0) > 0 or labels.get("SoundRep", 0) > 0):
        uclass_candidates["stutter_rep"].append((fn, item))
    elif has_stutter:
        uclass_candidates["stutter_other"].append((fn, item))

print("=== UCLASS Eligible Candidates ===")
for cat, lst in uclass_candidates.items():
    print(f"  {cat}: {len(lst)} candidate clips")

# 2. Load FluencyBank metadata
fb_csv = Path("Datasets/Stuttering Events in Podcasts Dataset (SEP-28k)/fluencybank_labels.csv")
fb_candidates = defaultdict(list)
with open(fb_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ep = row["EpId"].strip()
        cid = row["ClipId"].strip()
        fn = f"FluencyBank_{ep}_{cid}.wav"
        
        # Check exclusions
        if ep in holdout_fb_eps or ep in dev90_fb_eps:
            continue
        if int(row.get("PoorAudioQuality", 0)) > 0 or int(row.get("Unsure", 0)) > 0 or int(row.get("NoSpeech", 0)) > 0 or int(row.get("Music", 0)) > 0:
            continue
            
        has_stutter = (int(row.get("Block", 0)) > 0 or int(row.get("Prolongation", 0)) > 0 or 
                       int(row.get("SoundRep", 0)) > 0 or int(row.get("WordRep", 0)) > 0)
        has_interj = int(row.get("Interjection", 0)) > 0
        no_stutter = int(row.get("NoStutteredWords", 0)) > 0
        
        if not has_stutter and not has_interj and no_stutter:
            fb_candidates["fluent_controls"].append((fn, row))
        elif not has_stutter and has_interj:
            fb_candidates["non_stutter_interj"].append((fn, row))
        elif has_stutter and (int(row.get("WordRep", 0)) > 0 or int(row.get("SoundRep", 0)) > 0):
            fb_candidates["stutter_rep"].append((fn, row))
        elif has_stutter:
            fb_candidates["stutter_other"].append((fn, row))

print("\n=== FluencyBank Eligible Candidates ===")
for cat, lst in fb_candidates.items():
    print(f"  {cat}: {len(lst)} candidate clips")
