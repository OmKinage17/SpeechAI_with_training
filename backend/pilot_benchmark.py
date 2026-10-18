import os
import sys
import glob
import csv
import json
import time
import re
import numpy as np
import wave
import jiwer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

# Add backend to path to import main components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    get_whisper_model,
    detect_fillers,
    detect_stammering,
    clean_text
)

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation_results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("SPEECHAI METHODOLOGY PILOT BENCHMARK RUN (60 CLIPS)")
print("Hardware & Runtime Profiling + Dataset-Separated Evaluation")
print("=" * 60)

# Load Whisper model and measure initialization time
print("\n[1/5] Initializing Whisper model...")
t_model_start = time.time()
model = get_whisper_model()
t_model_load = time.time() - t_model_start
print(f"Whisper model loaded in {t_model_load:.2f} seconds.")

# -------------------------------------------------------------
# STEP 1: PILOT SAMPLE SELECTION WITH ZERO SPEAKER LEAKAGE
# -------------------------------------------------------------
print("\n[2/5] Constructing balanced pilot test split with zero speaker leakage...")

# UCLASS: Holdout speakers M_0030 (adult male, severe stutter) & M_0061 (teen male)
uclass_clips_dir = os.path.join(DATASET_ROOT, "UCLASS Stuttered Speech Clips (SEP-28k Format)", "clips", "clips")
uclass_labels_path = os.path.join(DATASET_ROOT, "UCLASS Stuttered Speech Clips (SEP-28k Format)", "clips", "labels.csv")

uclass_candidates = []
with open(uclass_labels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        fp = r.get("filepath", "")
        bname = os.path.basename(fp)
        full_path = os.path.join(uclass_clips_dir, bname)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 44:
            # Check speaker isolation
            if bname.startswith("M_0030") or bname.startswith("M_0061"):
                uclass_candidates.append({
                    "dataset": "UCLASS",
                    "file_path": full_path,
                    "filename": bname,
                    "speaker": "M_0030" if bname.startswith("M_0030") else "M_0061",
                    "gt_sound_rep": int(r.get("SoundRep", 0)),
                    "gt_word_rep": int(r.get("WordRep", 0)),
                    "gt_prolongation": int(r.get("Prolongation", 0)),
                    "gt_block": int(r.get("Block", 0)),
                    "gt_filler": int(r.get("Interjection", 0)),
                    "gt_clean": int(r.get("NoStutteredWords", 0))
                })

# Balanced selection of 20 UCLASS clips:
# 5 SoundRep, 5 WordRep, 5 Interjection/Filler, 5 Clean
uclass_pilot = []
s_reps = [c for c in uclass_candidates if c["gt_sound_rep"] == 1][:5]
w_reps = [c for c in uclass_candidates if c["gt_word_rep"] == 1 and c not in s_reps][:5]
fillers = [c for c in uclass_candidates if c["gt_filler"] == 1 and c not in s_reps and c not in w_reps][:5]
cleans = [c for c in uclass_candidates if c["gt_clean"] == 1 and c not in s_reps and c not in w_reps and c not in fillers][:5]
uclass_pilot = s_reps + w_reps + fillers + cleans

print(f"UCLASS Pilot Clips: {len(uclass_pilot)} (Speakers: M_0030, M_0061)")

# FluencyBank / SEP-28k: Holdout Episode '010'
fb_labels_path = os.path.join(DATASET_ROOT, "Stuttering Events in Podcasts Dataset (SEP-28k)", "fluencybank_labels.csv")
fb_audio_dir = os.path.join(DATASET_ROOT, "Stuttering Events in Podcasts Dataset (SEP-28k)", "clips", "stuttering-clips", "clips")

fb_candidates = []
with open(fb_labels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        ep = str(r.get("EpId", "")).strip()
        if ep == "010": # Holdout episode for clean test boundary
            show = str(r.get("Show", "")).strip()
            clip = str(r.get("ClipId", "")).strip()
            fname = f"{show}_{ep}_{clip}.wav"
            fpath = os.path.join(fb_audio_dir, fname)
            if os.path.exists(fpath) and os.path.getsize(fpath) > 44:
                fb_candidates.append({
                    "dataset": "FluencyBank",
                    "file_path": fpath,
                    "filename": fname,
                    "speaker": f"FB_Ep_{ep}",
                    "gt_sound_rep": 1 if int(r.get("SoundRep", 0)) >= 2 else 0,
                    "gt_word_rep": 1 if int(r.get("WordRep", 0)) >= 2 else 0,
                    "gt_prolongation": 1 if int(r.get("Prolongation", 0)) >= 2 else 0,
                    "gt_block": 1 if int(r.get("Block", 0)) >= 2 else 0,
                    "gt_filler": 1 if int(r.get("Interjection", 0)) >= 2 else 0,
                    "gt_clean": 1 if int(r.get("NoStutteredWords", 0)) >= 2 else 0
                })

fb_s_reps = [c for c in fb_candidates if c["gt_sound_rep"] == 1][:5]
fb_w_reps = [c for c in fb_candidates if c["gt_word_rep"] == 1 and c not in fb_s_reps][:5]
fb_fillers = [c for c in fb_candidates if c["gt_filler"] == 1 and c not in fb_s_reps and c not in fb_w_reps][:5]
fb_cleans = [c for c in fb_candidates if c["gt_clean"] == 1 and c not in fb_s_reps and c not in fb_w_reps and c not in fb_fillers][:5]
fb_pilot = fb_s_reps + fb_w_reps + fb_fillers + fb_cleans

print(f"FluencyBank Pilot Clips: {len(fb_pilot)} (Holdout EpId: 010)")

# Libri_Stutter: Holdout Speaker 103 (Chapter 1240)
libri_audio_dir = os.path.join(DATASET_ROOT, "Libri_Stutter", "LibriStutter Audio")
libri_annot_dir = os.path.join(DATASET_ROOT, "Libri_Stutter", "LibriStutter Annotations")

libri_pilot = []
spk_103_audio = glob.glob(os.path.join(libri_audio_dir, "103", "1240", "*.flac"))
for af in spk_103_audio[:20]:
    bname = os.path.basename(af)
    stem = os.path.splitext(bname)[0]
    csv_path = os.path.join(libri_annot_dir, "103", "1240", f"{stem}.csv")
    
    # Parse annotations to get reference text and stutter label types
    ref_words = []
    has_sound_rep = 0
    has_word_rep = 0
    has_filler = 0
    has_prolongation = 0
    
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as cf:
            for line in cf:
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    word = parts[0].strip()
                    code = parts[3].strip()
                    if word != "STUTTER":
                        ref_words.append(word)
                    if code == "1": has_filler = 1
                    elif code == "2": has_sound_rep = 1
                    elif code == "3": has_word_rep = 1
                    elif code == "5": has_prolongation = 1

    libri_pilot.append({
        "dataset": "Libri_Stutter",
        "file_path": af,
        "filename": bname,
        "speaker": "103_1240",
        "reference_text": " ".join(ref_words),
        "gt_sound_rep": has_sound_rep,
        "gt_word_rep": has_word_rep,
        "gt_prolongation": has_prolongation,
        "gt_filler": has_filler,
        "gt_clean": 1 if (has_sound_rep == 0 and has_word_rep == 0 and has_filler == 0 and has_prolongation == 0) else 0
    })

print(f"Libri_Stutter Pilot Clips: {len(libri_pilot)} (Holdout Speaker 103, Chapter 1240)")

full_pilot_manifest = uclass_pilot + fb_pilot + libri_pilot
print(f"Total Combined Pilot Manifest: {len(full_pilot_manifest)} clips.")

manifest_path = os.path.join(RESULTS_DIR, "pilot_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(full_pilot_manifest, f, indent=2)

# -------------------------------------------------------------
# STEP 2: INFERENCE EXECUTION & RUNTIME MEASUREMENT
# -------------------------------------------------------------
print("\n[3/5] Executing SpeechAI inference on pilot clips...")

predictions = []
latencies = []
audio_durations = []

for idx, item in enumerate(full_pilot_manifest, 1):
    fpath = item["file_path"]
    t0 = time.time()
    
    # 1. Transcribe with Whisper
    whisper_res = model.transcribe(
        fpath,
        initial_prompt="Um, uh, er, ah, hmm, like, actually, basically, so, well.",
        condition_on_previous_text=False,
        word_timestamps=True,
        language="en"
    )
    transcript = whisper_res.get("text", "").strip()
    segments = whisper_res.get("segments", [])
    
    # Extract word timestamps
    words_list = []
    duration = 3.0 # default for SEP/UCLASS
    if segments:
        duration = max(0.5, segments[-1].get("end", 0.0) - segments[0].get("start", 0.0))
        for seg in segments:
            if "words" in seg:
                words_list.extend(seg["words"])
                
    # 2. Run SpeechAI disfluency feature extractors
    # Lexical + DSP Acoustic Fillers
    filler_count, filler_words_found = detect_fillers(
        transcript=transcript,
        wav_path=fpath if fpath.endswith(".wav") else None,
        words_list=words_list,
        return_details=False
    )
    
    # Stammering Repetitions
    stammer_events = detect_stammering(
        transcript=transcript,
        words_list=words_list,
        return_details=False
    )
    
    dt = time.time() - t0
    latencies.append(dt)
    audio_durations.append(duration)
    
    # Calculate WER if reference text is present (Libri_Stutter)
    wer_score = None
    if item.get("reference_text"):
        ref_c = clean_text(item["reference_text"])
        hyp_c = clean_text(transcript)
        if ref_c:
            wer_score = round(jiwer.wer(ref_c, hyp_c), 3)

    pred_record = {
        **item,
        "pred_transcript": transcript,
        "pred_stammers": stammer_events,
        "pred_fillers": filler_count,
        "pred_has_stutter": 1 if stammer_events > 0 else 0,
        "pred_has_filler": 1 if filler_count > 0 else 0,
        "pred_is_clean": 1 if (stammer_events == 0 and filler_count == 0) else 0,
        "wer": wer_score,
        "latency_sec": round(dt, 3),
        "duration_sec": round(duration, 3),
        "rtf": round(dt / duration, 3) if duration > 0 else 0.0
    }
    predictions.append(pred_record)
    
    if idx % 10 == 0 or idx == len(full_pilot_manifest):
        print(f"  Processed {idx}/{len(full_pilot_manifest)} | Last clip RTF: {pred_record['rtf']:.2f}x | Avg Latency: {np.mean(latencies):.2f}s")

# Save predictions
pred_path = os.path.join(RESULTS_DIR, "pilot_predictions.json")
with open(pred_path, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2)

# -------------------------------------------------------------
# STEP 3: STATISTICAL METRICS PER DATASET (SEPARATE EVALUATION)
# -------------------------------------------------------------
print("\n[4/5] Computing formal metrics per dataset...")

results_by_dataset = {}

for ds_name in ["UCLASS", "FluencyBank", "Libri_Stutter"]:
    ds_preds = [p for p in predictions if p["dataset"] == ds_name]
    if not ds_preds:
        continue
        
    # Multi-class ground truth category mapping
    # 0: Clean, 1: Stutter (Repetition/Prolongation), 2: Filler
    y_true = []
    y_pred = []
    
    for p in ds_preds:
        # Ground truth hierarchy: Stutter > Filler > Clean
        if p["gt_sound_rep"] == 1 or p["gt_word_rep"] == 1 or p["gt_prolongation"] == 1:
            gt_label = "Stutter"
        elif p["gt_filler"] == 1:
            gt_label = "Filler"
        else:
            gt_label = "Clean"
            
        # Prediction hierarchy
        if p["pred_stammers"] > 0:
            pr_label = "Stutter"
        elif p["pred_fillers"] > 0:
            pr_label = "Filler"
        else:
            pr_label = "Clean"
            
        y_true.append(gt_label)
        y_pred.append(pr_label)

    classes = ["Clean", "Stutter", "Filler"]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    
    wers = [p["wer"] for p in ds_preds if p.get("wer") is not None]
    avg_wer = round(float(np.mean(wers)), 3) if wers else None

    results_by_dataset[ds_name] = {
        "sample_count": len(ds_preds),
        "overall_accuracy": round(float(acc), 4),
        "average_wer": avg_wer,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "metrics_per_class": {
            classes[i]: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1_score": round(float(f1[i]), 4)
            } for i in range(len(classes))
        }
    }
    
    print(f"\n--- {ds_name} Results (N={len(ds_preds)}) ---")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    if avg_wer is not None:
        print(f"Mean Word Error Rate (WER): {avg_wer * 100:.2f}%")
    for cls in classes:
        m = results_by_dataset[ds_name]["metrics_per_class"][cls]
        print(f"  Class '{cls:7s}': Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, F1={m['f1_score']:.3f}")

# Runtime Profile Summary
runtime_profile = {
    "total_clips": len(predictions),
    "whisper_model_load_sec": round(t_model_load, 2),
    "mean_latency_per_clip_sec": round(float(np.mean(latencies)), 3),
    "median_latency_per_clip_sec": round(float(np.median(latencies)), 3),
    "min_latency_sec": round(float(np.min(latencies)), 3),
    "max_latency_sec": round(float(np.max(latencies)), 3),
    "std_latency_sec": round(float(np.std(latencies)), 3),
    "mean_rtf": round(float(np.mean([p['rtf'] for p in predictions])), 3),
    "throughput_clips_per_minute": round(60.0 / float(np.mean(latencies)), 1)
}

final_report = {
    "runtime_profile": runtime_profile,
    "per_dataset_results": results_by_dataset
}

res_path = os.path.join(RESULTS_DIR, "pilot_results_summary.json")
with open(res_path, "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2)

# -------------------------------------------------------------
# STEP 4: PLOT PUBLICATION-GRADE FIGURES
# -------------------------------------------------------------
print("\n[5/5] Generating publication-grade graphs...")

# Plot 1: Runtime Latency Distribution Histogram
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
ax.hist(latencies, bins=12, color="#4F46E5", edgecolor="black", alpha=0.8)
ax.axvline(np.mean(latencies), color="#DC2626", linestyle="--", linewidth=1.5, label=f"Mean Latency ({np.mean(latencies):.2f}s)")
ax.set_title("SpeechAI Inference Latency Distribution (Pilot Hardware Profile)", fontsize=12, fontweight="bold")
ax.set_xlabel("Latency per 3-Second Audio Clip (seconds)", fontsize=10)
ax.set_ylabel("Clip Count", fontsize=10)
ax.legend(frameon=True)
ax.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
fig_latency_path = os.path.join(RESULTS_DIR, "pilot_latency_distribution.png")
plt.savefig(fig_latency_path)
plt.close()

# Plot 2: Class-Wise F1-Score Comparison across Datasets
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
x = np.arange(len(classes))
width = 0.25

ds_names = list(results_by_dataset.keys())
colors = ["#2563EB", "#059669", "#D97706"]

for i, ds in enumerate(ds_names):
    f1s = [results_by_dataset[ds]["metrics_per_class"][c]["f1_score"] for c in classes]
    ax.bar(x + (i - 1) * width, f1s, width, label=ds, color=colors[i % len(colors)], edgecolor="black", alpha=0.85)

ax.set_ylabel("F1-Score", fontsize=11, fontweight="bold")
ax.set_title("Class-Wise F1-Score Evaluation Across Datasets (Zero Speaker Leakage)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(frameon=True)
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
fig_f1_path = os.path.join(RESULTS_DIR, "pilot_classwise_f1_comparison.png")
plt.savefig(fig_f1_path)
plt.close()

print("\n" + "=" * 60)
print("PILOT BENCHMARK COMPLETE!")
print(f"Mean Latency: {runtime_profile['mean_latency_per_clip_sec']}s per clip")
print(f"Throughput:   {runtime_profile['throughput_clips_per_minute']} clips/minute")
print(f"Artifacts saved in: {RESULTS_DIR}")
print("=" * 60)
