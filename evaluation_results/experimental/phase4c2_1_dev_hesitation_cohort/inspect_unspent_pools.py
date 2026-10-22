import csv
from pathlib import Path
from collections import Counter, defaultdict

# Excluded speakers / sessions
holdout_uclass_speakers = {'M_1099', 'M_1023', 'M_1104', 'M_1105'}
holdout_fb_episodes = {'063', '089', '178', '099', 'Ep_063', 'Ep_089', 'Ep_178', 'Ep_099'}

dev90_uclass_speakers = {'M_0061', 'M_0030', 'M_1022'}
dev90_fb_episodes = {'010', '017', '031', 'Ep_010', 'Ep_017', 'Ep_031'}

# 1. Scan UCLASS
uclass_csv = Path("Datasets/UCLASS Stuttered Speech Clips (SEP-28k Format)/clips/labels.csv")
uclass_by_speaker = defaultdict(list)
with open(uclass_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fn = Path(row["filepath"]).name
        parts = fn.split("_")
        spk = f"{parts[0]}_{parts[1]}"
        uclass_by_speaker[spk].append(dict(row, filename=fn))

print("=== UCLASS Speakers Summary ===")
for spk, clips in sorted(uclass_by_speaker.items()):
    status = "EXCLUDED_HOLDOUT" if spk in holdout_uclass_speakers else ("EXCLUDED_DEV90" if spk in dev90_uclass_speakers else "ELIGIBLE_UNSPENT")
    interj_count = sum(1 for c in clips if int(c.get("Interjection", 0)) > 0)
    fluent_count = sum(1 for c in clips if int(c.get("NoStutteredWords", 0)) > 0 and int(c.get("Interjection", 0)) == 0 and int(c.get("Block", 0)) == 0 and int(c.get("Prolongation", 0)) == 0 and int(c.get("SoundRep", 0)) == 0 and int(c.get("WordRep", 0)) == 0)
    stutter_count = sum(1 for c in clips if int(c.get("Block", 0)) > 0 or int(c.get("Prolongation", 0)) > 0 or int(c.get("SoundRep", 0)) > 0 or int(c.get("WordRep", 0)) > 0)
    print(f"  {spk:10s} [{status:17s}]: {len(clips):4d} clips | Interjections: {interj_count:3d} | Fluent: {fluent_count:3d} | Stutter: {stutter_count:3d}")

# 2. Scan FluencyBank
fb_csv = Path("Datasets/Stuttering Events in Podcasts Dataset (SEP-28k)/fluencybank_labels.csv")
fb_by_ep = defaultdict(list)
with open(fb_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ep = row["EpId"].strip()
        cid = row["ClipId"].strip()
        fn = f"FluencyBank_{ep}_{cid}.wav"
        fb_by_ep[ep].append(dict(row, filename=fn))

print("\n=== FluencyBank Episodes Summary (Top 25) ===")
for ep, clips in sorted(fb_by_ep.items(), key=lambda x: len(x[1]), reverse=True)[:25]:
    status = "EXCLUDED_HOLDOUT" if (ep in holdout_fb_episodes or f"Ep_{ep}" in holdout_fb_episodes) else ("EXCLUDED_DEV90" if (ep in dev90_fb_episodes or f"Ep_{ep}" in dev90_fb_episodes) else "ELIGIBLE_UNSPENT")
    interj_count = sum(1 for c in clips if int(c.get("Interjection", 0)) > 0)
    clean_count = sum(1 for c in clips if int(c.get("NoStutteredWords", 0)) > 0 and int(c.get("Interjection", 0)) == 0 and int(c.get("Block", 0)) == 0 and int(c.get("Prolongation", 0)) == 0 and int(c.get("SoundRep", 0)) == 0 and int(c.get("WordRep", 0)) == 0)
    stutter_count = sum(1 for c in clips if int(c.get("Block", 0)) > 0 or int(c.get("Prolongation", 0)) > 0 or int(c.get("SoundRep", 0)) > 0 or int(c.get("WordRep", 0)) > 0)
    print(f"  Ep {ep:4s} [{status:17s}]: {len(clips):3d} clips | Interjections: {interj_count:2d} | Clean: {clean_count:2d} | Stutter: {stutter_count:2d}")
