import csv
import json
from pathlib import Path

# 1. Load locked 120-clip final benchmark filenames (metadata only - filenames for exclusion)
locked_manifest_path = Path("evaluation_results/experimental/final_benchmark/run_final_20261020_complete/run_manifest.json")
with open(locked_manifest_path, "r", encoding="utf-8") as f:
    locked_manifest = json.load(f)
if isinstance(locked_manifest, list):
    locked_filenames = set(item["filename"] for item in locked_manifest)
else:
    locked_filenames = set(item["filename"] for item in locked_manifest.get("items", []))
print(f"Total locked holdout clips to exclude: {len(locked_filenames)}")

# 2. Load existing 90 development clips
dev30_path = Path("evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json")
gen60_path = Path("evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json")

with open(dev30_path, "r", encoding="utf-8") as f:
    dev30_cache = json.load(f)
dev30_filenames = set(dev30_cache.keys())

with open(gen60_path, "r", encoding="utf-8") as f:
    gen60_manifest = json.load(f)
if isinstance(gen60_manifest, list):
    gen60_filenames = set(item["filename"] for item in gen60_manifest)
else:
    gen60_filenames = set(item["filename"] for item in gen60_manifest.get("items", []))

dev90_filenames = dev30_filenames.union(gen60_filenames)
print(f"Total dev90 clips: {len(dev90_filenames)}")
print(f"Overlap between dev90 and locked: {len(dev90_filenames.intersection(locked_filenames))}")

# 3. Check UCLASS clips pool
uclass_csv = Path("Datasets/UCLASS Stuttered Speech Clips (SEP-28k Format)/clips/labels.csv")
uclass_clips_dir = Path("Datasets/UCLASS Stuttered Speech Clips (SEP-28k Format)/clips/clips")

uclass_all = []
with open(uclass_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fp = row["filepath"]
        fn = Path(fp).name
        uclass_all.append({
            "filename": fn,
            "filepath": str(uclass_clips_dir / fn),
            "dataset": "UCLASS",
            "block": int(row.get("Block", 0)),
            "prolongation": int(row.get("Prolongation", 0)),
            "sound_rep": int(row.get("SoundRep", 0)),
            "word_rep": int(row.get("WordRep", 0)),
            "interjection": int(row.get("Interjection", 0)),
            "no_stuttered_words": int(row.get("NoStutteredWords", 0))
        })
print(f"Total UCLASS clips in labels.csv: {len(uclass_all)}")

# 4. Check FluencyBank clips pool
fb_csv = Path("Datasets/Stuttering Events in Podcasts Dataset (SEP-28k)/fluencybank_labels.csv")
fb_clips_dir = Path("Datasets/Stuttering Events in Podcasts Dataset (SEP-28k)/clips/stuttering-clips/clips")

fb_all = []
with open(fb_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ep = row["EpId"].strip()
        cid = row["ClipId"].strip()
        fn = f"FluencyBank_{ep}_{cid}.wav"
        fb_all.append({
            "filename": fn,
            "filepath": str(fb_clips_dir / f"{ep}_{cid}.wav"),
            "dataset": "FluencyBank",
            "ep_id": ep,
            "clip_id": cid,
            "block": int(row.get("Block", 0)),
            "prolongation": int(row.get("Prolongation", 0)),
            "sound_rep": int(row.get("SoundRep", 0)),
            "word_rep": int(row.get("WordRep", 0)),
            "interjection": int(row.get("Interjection", 0)),
            "natural_pause": int(row.get("NaturalPause", 0)),
            "no_stuttered_words": int(row.get("NoStutteredWords", 0)),
            "unsure": int(row.get("Unsure", 0)),
            "poor_audio": int(row.get("PoorAudioQuality", 0))
        })
print(f"Total FluencyBank clips in labels.csv: {len(fb_all)}")
