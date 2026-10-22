import os
import glob
import pandas as pd

base = r"e:\SpeechPrj\Datasets"
print("=== RAW / SOURCE DATASETS ON DISK ===")

# 1. Libri_Stutter
libri_dir = os.path.join(base, "Libri_Stutter")
libri_audio = os.path.join(libri_dir, "LibriStutter Audio")
libri_clips = glob.glob(os.path.join(libri_audio, "**", "*.wav"), recursive=True) + glob.glob(os.path.join(libri_audio, "**", "*.flac"), recursive=True)
print(f"Libri_Stutter: {len(libri_clips)} audio clips found in '{libri_audio}'")
# Check subdirectories in LibriStutter Audio
if os.path.exists(libri_audio):
    subdirs = [d for d in os.listdir(libri_audio) if os.path.isdir(os.path.join(libri_audio, d))]
    for sd in subdirs:
        count = len(glob.glob(os.path.join(libri_audio, sd, "**", "*.*"), recursive=True))
        print(f"  - {sd}: {count} files")

# 2. SEP-28k
sep_dir = os.path.join(base, "Stuttering Events in Podcasts Dataset (SEP-28k)")
sep_clips_dir = os.path.join(sep_dir, "clips")
sep_clips = glob.glob(os.path.join(sep_clips_dir, "**", "*.wav"), recursive=True)
print(f"\nSEP-28k clips on disk: {len(sep_clips)} audio clips found in '{sep_clips_dir}'")
if os.path.exists(sep_clips_dir):
    subdirs = [d for d in os.listdir(sep_clips_dir) if os.path.isdir(os.path.join(sep_clips_dir, d))]
    for sd in subdirs:
        count = len(glob.glob(os.path.join(sep_clips_dir, sd, "*.wav")))
        print(f"  - {sd}: {count} wav files")

# Labels in SEP-28k
sep_labels_path = os.path.join(sep_dir, "SEP-28k_labels.csv")
if os.path.exists(sep_labels_path):
    df_sep = pd.read_csv(sep_labels_path)
    print(f"  SEP-28k_labels.csv entries: {len(df_sep)} rows")
fb_labels_path = os.path.join(sep_dir, "fluencybank_labels.csv")
if os.path.exists(fb_labels_path):
    df_fb = pd.read_csv(fb_labels_path)
    print(f"  fluencybank_labels.csv entries: {len(df_fb)} rows")

# 3. UCLASS
uclass_dir = os.path.join(base, "UCLASS Stuttered Speech Clips (SEP-28k Format)")
uclass_clips_dir = os.path.join(uclass_dir, "clips")
uclass_clips = glob.glob(os.path.join(uclass_clips_dir, "**", "*.wav"), recursive=True)
print(f"\nUCLASS clips on disk: {len(uclass_clips)} audio clips found in '{uclass_clips_dir}'")
if os.path.exists(uclass_clips_dir):
    subdirs = [d for d in os.listdir(uclass_clips_dir) if os.path.isdir(os.path.join(uclass_clips_dir, d))]
    for sd in subdirs:
        count = len(glob.glob(os.path.join(uclass_clips_dir, sd, "*.wav")))
        print(f"  - {sd}: {count} wav files")
uclass_raw_audio = os.path.join(uclass_dir, "audio")
if os.path.exists(uclass_raw_audio):
    raw_files = glob.glob(os.path.join(uclass_raw_audio, "**", "*.wav"), recursive=True)
    print(f"  UCLASS raw full recordings in 'audio/': {len(raw_files)} wav files")

print("\n=== PROJECT BENCHMARK & EVALUATION SETS ===")
# Check evaluation_results cohorts
eval_dir = r"e:\SpeechPrj\evaluation_results"
for root, dirs, files in os.walk(eval_dir):
    for f in files:
        if f.endswith(".csv") and ("manifest" in f.lower() or "cohort" in f.lower() or "benchmark" in f.lower() or "holdout" in f.lower() or "labels" in f.lower()):
            fpath = os.path.join(root, f)
            try:
                df = pd.read_csv(fpath)
                rel = os.path.relpath(fpath, r"e:\SpeechPrj")
                print(f"- {rel}: {len(df)} rows")
            except Exception:
                pass
