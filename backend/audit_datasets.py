import os
import sys
import glob
import csv
import json

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation_results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"=== SPEECHAI DATASET AUDIT ===")
print(f"Dataset root: {DATASET_ROOT}")

audit_summary = {}

# 1. AUDIT SEP-28k / FluencyBank
sep_dir = os.path.join(DATASET_ROOT, "Stuttering Events in Podcasts Dataset (SEP-28k)")
sep_audio_dir = os.path.join(sep_dir, "clips", "stuttering-clips", "clips")
sep_labels_path = os.path.join(sep_dir, "SEP-28k_labels.csv")
fb_labels_path = os.path.join(sep_dir, "fluencybank_labels.csv")

print("\n--- Auditing SEP-28k & FluencyBank ---")
sep_audio_files = glob.glob(os.path.join(sep_audio_dir, "*.wav")) if os.path.exists(sep_audio_dir) else []
print(f"Total audio files found in clips folder: {len(sep_audio_files)}")

valid_sep_audio = {}
corrupted_sep_audio = []
for f in sep_audio_files:
    sz = os.path.getsize(f)
    bname = os.path.basename(f)
    if sz <= 44:
        corrupted_sep_audio.append(bname)
    else:
        valid_sep_audio[bname] = f

print(f"Valid audio clips (>44 bytes): {len(valid_sep_audio)}")
print(f"Corrupted/placeholder clips (<=44 bytes): {len(corrupted_sep_audio)}")

# Read fluencybank_labels.csv
fb_rows = []
if os.path.exists(fb_labels_path):
    with open(fb_labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fb_rows.append({k.strip(): v.strip() for k, v in r.items()})

print(f"fluencybank_labels.csv rows: {len(fb_rows)}")

# Check matching with audio
fb_matched = []
fb_missing = []
fb_corrupted_matched = []

for r in fb_rows:
    show = r.get('Show', '')
    ep = r.get('EpId', '')
    clip = r.get('ClipId', '')
    candidate = f"{show}_{ep}_{clip}.wav"
    if candidate in valid_sep_audio:
        fb_matched.append((candidate, r))
    elif candidate in corrupted_sep_audio:
        fb_corrupted_matched.append(candidate)
    else:
        fb_missing.append(candidate)

print(f"FluencyBank labels matched with valid playable audio: {len(fb_matched)}")
print(f"FluencyBank matched with corrupted (44-byte) clips: {len(fb_corrupted_matched)}")
print(f"FluencyBank completely missing audio: {len(fb_missing)}")

# Analyze class distribution in valid FluencyBank clips
fb_class_dist = {
    "Prolongation": 0,
    "Block": 0,
    "SoundRep": 0,
    "WordRep": 0,
    "Interjection": 0,
    "NoStutteredWords": 0
}
for _, r in fb_matched:
    for k in fb_class_dist.keys():
        try:
            val = int(r.get(k, 0))
            if val >= 2: # Consensus of at least 2 of 3 annotators
                fb_class_dist[k] += 1
        except ValueError:
            pass

print("FluencyBank class distribution (consensus >= 2 annotators):")
for k, v in fb_class_dist.items():
    print(f"  - {k}: {v} clips")

audit_summary["FluencyBank"] = {
    "total_audio_found": len(sep_audio_files),
    "valid_audio": len(valid_sep_audio),
    "corrupted_audio": len(corrupted_sep_audio),
    "labels_count": len(fb_rows),
    "valid_matched_clips": len(fb_matched),
    "class_distribution_consensus_ge2": fb_class_dist
}

# 2. AUDIT UCLASS
print("\n--- Auditing UCLASS ---")
uclass_dir = os.path.join(DATASET_ROOT, "UCLASS Stuttered Speech Clips (SEP-28k Format)")
uclass_clips_dir = os.path.join(uclass_dir, "clips", "clips")
uclass_labels_path = os.path.join(uclass_dir, "clips", "labels.csv")

uclass_audio_files = glob.glob(os.path.join(uclass_clips_dir, "*.wav")) if os.path.exists(uclass_clips_dir) else []
print(f"Total UCLASS audio files found: {len(uclass_audio_files)}")

valid_uclass_audio = {}
corrupted_uclass_audio = []
uclass_speakers = set()

for f in uclass_audio_files:
    sz = os.path.getsize(f)
    bname = os.path.basename(f)
    parts = bname.split("_")
    if len(parts) >= 2:
        spk = f"{parts[0]}_{parts[1]}"
        uclass_speakers.add(spk)
    if sz <= 44:
        corrupted_uclass_audio.append(bname)
    else:
        valid_uclass_audio[bname] = f

print(f"Valid UCLASS audio clips (>44 bytes): {len(valid_uclass_audio)}")
print(f"Corrupted UCLASS clips (<=44 bytes): {len(corrupted_uclass_audio)}")
print(f"Unique UCLASS speakers identified: {len(uclass_speakers)}: {sorted(list(uclass_speakers))}")

uclass_rows = []
if os.path.exists(uclass_labels_path):
    with open(uclass_labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            uclass_rows.append({k.strip(): v.strip() for k, v in r.items()})

print(f"UCLASS labels.csv rows: {len(uclass_rows)}")

uclass_matched = []
uclass_class_dist = {
    "Block": 0,
    "Prolongation": 0,
    "SoundRep": 0,
    "WordRep": 0,
    "Interjection": 0,
    "NoStutteredWords": 0
}

uclass_speaker_clips = {}
for r in uclass_rows:
    fp = r.get('filepath', '')
    bname = os.path.basename(fp)
    if bname in valid_uclass_audio:
        uclass_matched.append((bname, r))
        # Speaker tracking
        parts = bname.split("_")
        spk = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "unknown"
        uclass_speaker_clips[spk] = uclass_speaker_clips.get(spk, 0) + 1
        for k in uclass_class_dist.keys():
            try:
                if int(r.get(k, 0)) == 1:
                    uclass_class_dist[k] += 1
            except ValueError:
                pass

print(f"UCLASS valid matched clips: {len(uclass_matched)}")
print("UCLASS class distribution:")
for k, v in uclass_class_dist.items():
    print(f"  - {k}: {v} clips")

print("Clips per speaker in UCLASS:")
for spk, cnt in sorted(uclass_speaker_clips.items()):
    print(f"  - {spk}: {cnt} clips")

audit_summary["UCLASS"] = {
    "total_audio_found": len(uclass_audio_files),
    "valid_audio": len(valid_uclass_audio),
    "corrupted_audio": len(corrupted_uclass_audio),
    "unique_speakers": len(uclass_speakers),
    "speaker_distribution": uclass_speaker_clips,
    "labels_count": len(uclass_rows),
    "valid_matched_clips": len(uclass_matched),
    "class_distribution": uclass_class_dist
}

# 3. AUDIT Libri_Stutter
print("\n--- Auditing Libri_Stutter ---")
libri_dir = os.path.join(DATASET_ROOT, "Libri_Stutter")
libri_audio_dir = os.path.join(libri_dir, "LibriStutter Audio")
libri_annot_dir = os.path.join(libri_dir, "LibriStutter Annotations")
libri_transcript_dir = os.path.join(libri_dir, "LibriStutter Transcripts")

libri_speakers = [d for d in os.listdir(libri_audio_dir) if os.path.isdir(os.path.join(libri_audio_dir, d))] if os.path.exists(libri_audio_dir) else []
print(f"Libri_Stutter speaker folders: {len(libri_speakers)}")

libri_audio_files = []
if os.path.exists(libri_audio_dir):
    for root, _, files in os.walk(libri_audio_dir):
        for f in files:
            if f.endswith(".flac") or f.endswith(".wav"):
                libri_audio_files.append(os.path.join(root, f))

print(f"Total Libri_Stutter audio files: {len(libri_audio_files)}")

# Check paired annotation files
libri_paired = []
for af in libri_audio_files:
    stem = os.path.splitext(os.path.basename(af))[0]
    parts = stem.split("-")
    if len(parts) >= 3:
        spk, ch, utt = parts[0], parts[1], parts[2]
        expected_csv = os.path.join(libri_annot_dir, spk, ch, f"{stem}.csv")
        if os.path.exists(expected_csv) and os.path.getsize(expected_csv) > 0:
            libri_paired.append((af, expected_csv, spk))

print(f"Fully paired Libri_Stutter audio + time-aligned CSV files: {len(libri_paired)}")

audit_summary["Libri_Stutter"] = {
    "speaker_folders": len(libri_speakers),
    "total_audio_files": len(libri_audio_files),
    "paired_audio_and_annotation": len(libri_paired)
}

# Save complete audit report
audit_out_path = os.path.join(RESULTS_DIR, "dataset_audit_report.json")
with open(audit_out_path, "w", encoding="utf-8") as f:
    json.dump(audit_summary, f, indent=2)

print(f"\n==========================================")
print(f"AUDIT SUMMARY COMPLETE!")
print(f"Saved to: {audit_out_path}")
print(f"==========================================")
