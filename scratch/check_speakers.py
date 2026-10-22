import json
from pathlib import Path
from collections import Counter

# Inspect spent holdout 120
holdout_path = Path(r"E:\SpeechPrj\evaluation_results\experimental\expanded_validation\run_expanded_120\run_manifest.json")
if holdout_path.exists():
    with open(holdout_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Spent Holdout 120 items: {len(data)}")
    speakers = Counter()
    datasets = Counter()
    for item in data:
        spk = item.get("speaker_id") or item.get("speaker")
        speakers[spk] += 1
        datasets[item.get("dataset")] += 1
    print(f"Holdout 120 datasets: {dict(datasets)}")
    print(f"Holdout 120 speakers: {dict(speakers)}")

# Inspect dev error diagnostic (dev_30 + gen_60)
diag_path = Path(r"E:\SpeechPrj\evaluation_results\experimental\dev_error_diagnostic.json")
if diag_path.exists():
    with open(diag_path, "r", encoding="utf-8") as f:
        diag = json.load(f)
    dev30 = diag.get("dev_30_records", {})
    gen60 = diag.get("gen_60_records", {})
    print(f"\nDev 30 items: {len(dev30)}")
    print(f"Gen 60 items: {len(gen60)}")
    
    # Check speakers in dev30 and gen60
    dev_speakers = Counter()
    for k, v in dev30.items():
        dev_speakers[v.get("speaker", "unknown")] += 1
    print(f"Dev 30 speakers: {dict(dev_speakers)}")
    
    gen_speakers = Counter()
    for k, v in gen60.items():
        gen_speakers[v.get("speaker", "unknown")] += 1
    print(f"Gen 60 speakers: {dict(gen_speakers)}")
