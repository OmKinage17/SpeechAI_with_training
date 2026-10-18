import os
import sys
import glob
import csv
import json
import random
import wave
import soundfile as sf
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation_results", "main_benchmark"))
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 65)
print("MAIN BENCHMARK PREFLIGHT CHECKS (750 CLIPS PROPOSED)")
print("Random Seed: 42 | Strict Speaker Isolation")
print("=" * 65)

preflight_log = {
    "random_seed": RANDOM_SEED,
    "pilot_holdouts_excluded": {
        "UCLASS": ["M_0030", "M_0061"],
        "FluencyBank": ["FB_Ep_010"],
        "Libri_Stutter": ["103_1240"]
    },
    "datasets": {}
}

# ---------------------------------------------------------
# 1. UCLASS PREFLIGHT (Target: 250 clips)
# ---------------------------------------------------------
print("\n[Check 1/3] UCLASS Holdout Verification...")
uclass_clips_dir = os.path.join(DATASET_ROOT, "UCLASS Stuttered Speech Clips (SEP-28k Format)", "clips", "clips")
uclass_labels_path = os.path.join(DATASET_ROOT, "UCLASS Stuttered Speech Clips (SEP-28k Format)", "clips", "labels.csv")

# Available candidate speakers excluding pilot (M_0030, M_0061):
# We use M_0078, M_0107, M_0553, M_0121 (clinical patients)
uclass_holdout_speakers = ["M_0078", "M_0107", "M_0553", "M_0121"]
uclass_candidates = []

with open(uclass_labels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        fp = r.get("filepath", "")
        bname = os.path.basename(fp)
        fpath = os.path.join(uclass_clips_dir, bname)
        
        # Check speaker prefix
        spk = None
        for s in uclass_holdout_speakers:
            if bname.startswith(s):
                spk = s
                break
                
        if spk and os.path.exists(fpath) and os.path.getsize(fpath) > 44:
            # Check wave readability
            try:
                with wave.open(fpath, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    dur = frames / float(rate)
                uclass_candidates.append({
                    "dataset": "UCLASS",
                    "file_path": fpath,
                    "filename": bname,
                    "speaker": spk,
                    "duration_sec": round(dur, 3),
                    "gt_sound_rep": int(r.get("SoundRep", 0)),
                    "gt_word_rep": int(r.get("WordRep", 0)),
                    "gt_prolongation": int(r.get("Prolongation", 0)),
                    "gt_block": int(r.get("Block", 0)),
                    "gt_filler": int(r.get("Interjection", 0)),
                    "gt_clean": int(r.get("NoStutteredWords", 0))
                })
            except Exception as e:
                print(f"Warning: Corrupted WAV {bname}: {e}")

print(f"  Total valid verified candidates in holdout speakers {uclass_holdout_speakers}: {len(uclass_candidates)}")
assert len(uclass_candidates) >= 250, f"Insufficient UCLASS candidates ({len(uclass_candidates)} < 250)"

# Natural random sample of 250 clips without replacement
random.shuffle(uclass_candidates)
uclass_selected = uclass_candidates[:250]

# Tally support
uclass_support = {
    "Total": len(uclass_selected),
    "SoundRep": sum(1 for c in uclass_selected if c["gt_sound_rep"] == 1),
    "WordRep": sum(1 for c in uclass_selected if c["gt_word_rep"] == 1),
    "Prolongation": sum(1 for c in uclass_selected if c["gt_prolongation"] == 1),
    "Block": sum(1 for c in uclass_selected if c["gt_block"] == 1),
    "Interjection": sum(1 for c in uclass_selected if c["gt_filler"] == 1),
    "NoStutteredWords": sum(1 for c in uclass_selected if c["gt_clean"] == 1)
}
print(f"  UCLASS Selected: 250 clips. Natural Event Support: {uclass_support}")

preflight_log["datasets"]["UCLASS"] = {
    "holdout_speakers": uclass_holdout_speakers,
    "available_candidates": len(uclass_candidates),
    "selected_count": len(uclass_selected),
    "class_support": uclass_support
}

# ---------------------------------------------------------
# 2. FLUENCYBANK PREFLIGHT (Target: 250 clips)
# ---------------------------------------------------------
print("\n[Check 2/3] FluencyBank Holdout Verification...")
fb_labels_path = os.path.join(DATASET_ROOT, "Stuttering Events in Podcasts Dataset (SEP-28k)", "fluencybank_labels.csv")
fb_audio_dir = os.path.join(DATASET_ROOT, "Stuttering Events in Podcasts Dataset (SEP-28k)", "clips", "stuttering-clips", "clips")

# Available candidate episodes excluding pilot Ep_010:
# Ep_016 (115 valid clips) and Ep_024 (197 valid clips) = 312 clips total
fb_holdout_episodes = ["016", "024"]
fb_candidates = []

with open(fb_labels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        ep = str(r.get("EpId", "")).strip()
        if ep in fb_holdout_episodes:
            show = str(r.get("Show", "")).strip()
            clip = str(r.get("ClipId", "")).strip()
            fname = f"{show}_{ep}_{clip}.wav"
            fpath = os.path.join(fb_audio_dir, fname)
            
            if os.path.exists(fpath) and os.path.getsize(fpath) > 44:
                try:
                    with wave.open(fpath, "rb") as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        dur = frames / float(rate)
                    
                    # Multi-annotator consensus threshold >= 2
                    s_rep = 1 if int(r.get("SoundRep", 0)) >= 2 else 0
                    w_rep = 1 if int(r.get("WordRep", 0)) >= 2 else 0
                    prol = 1 if int(r.get("Prolongation", 0)) >= 2 else 0
                    blk = 1 if int(r.get("Block", 0)) >= 2 else 0
                    fil = 1 if int(r.get("Interjection", 0)) >= 2 else 0
                    cln = 1 if int(r.get("NoStutteredWords", 0)) >= 2 else 0
                    
                    fb_candidates.append({
                        "dataset": "FluencyBank",
                        "file_path": fpath,
                        "filename": fname,
                        "speaker": f"FB_Ep_{ep}",
                        "duration_sec": round(dur, 3),
                        "gt_sound_rep": s_rep,
                        "gt_word_rep": w_rep,
                        "gt_prolongation": prol,
                        "gt_block": blk,
                        "gt_filler": fil,
                        "gt_clean": cln
                    })
                except Exception as e:
                    print(f"Warning: Corrupted WAV {fname}: {e}")

print(f"  Total valid verified candidates in holdout episodes {fb_holdout_episodes}: {len(fb_candidates)}")
assert len(fb_candidates) >= 250, f"Insufficient FluencyBank candidates ({len(fb_candidates)} < 250)"

random.shuffle(fb_candidates)
fb_selected = fb_candidates[:250]

fb_support = {
    "Total": len(fb_selected),
    "SoundRep": sum(1 for c in fb_selected if c["gt_sound_rep"] == 1),
    "WordRep": sum(1 for c in fb_selected if c["gt_word_rep"] == 1),
    "Prolongation": sum(1 for c in fb_selected if c["gt_prolongation"] == 1),
    "Block": sum(1 for c in fb_selected if c["gt_block"] == 1),
    "Interjection": sum(1 for c in fb_selected if c["gt_filler"] == 1),
    "NoStutteredWords": sum(1 for c in fb_selected if c["gt_clean"] == 1)
}
print(f"  FluencyBank Selected: 250 clips. Natural Event Support: {fb_support}")

preflight_log["datasets"]["FluencyBank"] = {
    "holdout_episodes": fb_holdout_episodes,
    "available_candidates": len(fb_candidates),
    "selected_count": len(fb_selected),
    "class_support": fb_support
}

# ---------------------------------------------------------
# 3. LIBRI_STUTTER PREFLIGHT (Target: 250 clips)
# ---------------------------------------------------------
print("\n[Check 3/3] Libri_Stutter Holdout Verification...")
libri_audio_dir = os.path.join(DATASET_ROOT, "Libri_Stutter", "LibriStutter Audio")
libri_annot_dir = os.path.join(DATASET_ROOT, "Libri_Stutter", "LibriStutter Annotations")

# Available candidate readers excluding pilot 103:
# Readers 1088, 1334, 1502
libri_holdout_readers = ["1088", "1334", "1502"]
libri_candidates = []

for rdr in libri_holdout_readers:
    rdr_audio_path = os.path.join(libri_audio_dir, rdr)
    if os.path.exists(rdr_audio_path):
        for root, _, files in os.walk(rdr_audio_path):
            for f in files:
                if f.endswith(".flac"):
                    fpath = os.path.join(root, f)
                    stem = os.path.splitext(f)[0]
                    parts = stem.split("-")
                    if len(parts) >= 3:
                        spk, ch, utt = parts[0], parts[1], parts[2]
                        annot_csv = os.path.join(libri_annot_dir, spk, ch, f"{stem}.csv")
                        if os.path.exists(annot_csv) and os.path.getsize(annot_csv) > 0:
                            # Verify readability with soundfile
                            try:
                                info = sf.info(fpath)
                                dur = info.duration
                                
                                # Parse reference text and stutter codes
                                ref_words = []
                                s_rep = 0
                                w_rep = 0
                                prol = 0
                                fil = 0
                                
                                with open(annot_csv, "r", encoding="utf-8") as cf:
                                    for line in cf:
                                        c_parts = line.strip().split(",")
                                        if len(c_parts) >= 4:
                                            word = c_parts[0].strip()
                                            code = c_parts[3].strip()
                                            if word != "STUTTER":
                                                ref_words.append(word)
                                            if code == "1": fil = 1
                                            elif code == "2": s_rep = 1
                                            elif code == "3": w_rep = 1
                                            elif code == "5": prol = 1
                                
                                cln = 1 if (s_rep == 0 and w_rep == 0 and prol == 0 and fil == 0) else 0
                                
                                libri_candidates.append({
                                    "dataset": "Libri_Stutter",
                                    "file_path": fpath,
                                    "filename": f,
                                    "speaker": f"Libri_{rdr}_{ch}",
                                    "reference_text": " ".join(ref_words),
                                    "duration_sec": round(dur, 3),
                                    "gt_sound_rep": s_rep,
                                    "gt_word_rep": w_rep,
                                    "gt_prolongation": prol,
                                    "gt_filler": fil,
                                    "gt_clean": cln
                                })
                            except Exception as e:
                                print(f"Warning: Corrupted FLAC {f}: {e}")

print(f"  Total valid verified candidates in holdout readers {libri_holdout_readers}: {len(libri_candidates)}")
assert len(libri_candidates) >= 250, f"Insufficient Libri_Stutter candidates ({len(libri_candidates)} < 250)"

random.shuffle(libri_candidates)
libri_selected = libri_candidates[:250]

libri_support = {
    "Total": len(libri_selected),
    "SoundRep": sum(1 for c in libri_selected if c["gt_sound_rep"] == 1),
    "WordRep": sum(1 for c in libri_selected if c["gt_word_rep"] == 1),
    "Prolongation": sum(1 for c in libri_selected if c["gt_prolongation"] == 1),
    "Interjection": sum(1 for c in libri_selected if c["gt_filler"] == 1),
    "NoStutteredWords": sum(1 for c in libri_selected if c["gt_clean"] == 1)
}
print(f"  Libri_Stutter Selected: 250 clips. Natural Event Support: {libri_support}")

preflight_log["datasets"]["Libri_Stutter"] = {
    "holdout_readers": libri_holdout_readers,
    "available_candidates": len(libri_candidates),
    "selected_count": len(libri_selected),
    "class_support": libri_support
}

# ---------------------------------------------------------
# COMBINE AND SAVE VERIFIED MANIFEST (EXACT 750 CLIPS)
# ---------------------------------------------------------
main_manifest = uclass_selected + fb_selected + libri_selected
print(f"\n[Verification] Total Combined Verified Clips: {len(main_manifest)}")
assert len(main_manifest) == 750, f"Total clips != 750 (got {len(main_manifest)})"

manifest_path = os.path.join(RESULTS_DIR, "main_benchmark_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(main_manifest, f, indent=2)

preflight_path = os.path.join(RESULTS_DIR, "preflight_check_report.json")
with open(preflight_path, "w", encoding="utf-8") as f:
    json.dump(preflight_log, f, indent=2)

print("\n" + "=" * 65)
print("ALL PREFLIGHT CHECKS PASSED SUCCESSFULLY (100% VERIFIED)!")
print(f"Verified Manifest saved to: {manifest_path}")
print(f"Preflight Report saved to:   {preflight_path}")
print("=" * 65)
