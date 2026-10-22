import json
import csv
from pathlib import Path

# 1. Inspect locked holdout (120 clips)
locked_manifest_path = Path("evaluation_results/experimental/final_benchmark/run_final_20261020_complete/run_manifest.json")
with open(locked_manifest_path, "r", encoding="utf-8") as f:
    locked_items = json.load(f)

locked_info = {}
for it in locked_items:
    fn = it["filename"]
    ds = it["dataset"]
    spk = it.get("speaker_id")
    # extract speaker/session from filename if not in it
    if ds == "UCLASS":
        # e.g., M_1099_25y0m_1_dysfluent_000.wav -> speaker is M_1099
        parts = fn.split("_")
        spk_derived = f"{parts[0]}_{parts[1]}"
    elif ds == "FluencyBank":
        # e.g., FluencyBank_010_4.wav -> ep is 010
        parts = fn.split("_")
        spk_derived = parts[1]
    else:
        spk_derived = fn.split("-")[0]
    locked_info[fn] = {
        "dataset": ds,
        "speaker": spk or spk_derived
    }

print("Locked holdout UCLASS speakers:", set(v["speaker"] for v in locked_info.values() if v["dataset"] == "UCLASS"))
print("Locked holdout FluencyBank sessions:", set(v["speaker"] for v in locked_info.values() if v["dataset"] == "FluencyBank"))

# 2. Inspect 90 dev clips
dev30_path = Path("evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json")
gen60_path = Path("evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json")

with open(dev30_path, "r", encoding="utf-8") as f:
    dev30_cache = json.load(f)

dev_info = {}
for fn, c in dev30_cache.items():
    ds = c.get("dataset", "UCLASS")
    if ds == "UCLASS":
        parts = fn.split("_")
        spk = f"{parts[0]}_{parts[1]}"
    elif ds == "FluencyBank":
        parts = fn.split("_")
        spk = parts[1]
    else:
        spk = fn.split("-")[0]
    dev_info[fn] = {"dataset": ds, "speaker": spk, "split": "dev_30"}

with open(gen60_path, "r", encoding="utf-8") as f:
    gen60_items = json.load(f)

for it in gen60_items:
    fn = it["filename"]
    ds = it["dataset"]
    if ds == "UCLASS":
        parts = fn.split("_")
        spk = f"{parts[0]}_{parts[1]}"
    elif ds == "FluencyBank":
        parts = fn.split("_")
        spk = parts[1]
    else:
        spk = fn.split("-")[0]
    dev_info[fn] = {"dataset": ds, "speaker": spk, "split": "gen_60"}

print("\nDev90 UCLASS speakers:", set(v["speaker"] for v in dev_info.values() if v["dataset"] == "UCLASS"))
print("Dev90 FluencyBank sessions:", set(v["speaker"] for v in dev_info.values() if v["dataset"] == "FluencyBank"))
