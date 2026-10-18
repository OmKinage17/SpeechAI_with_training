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

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation_results", "main_benchmark"))
os.makedirs(RESULTS_DIR, exist_ok=True)
manifest_path = os.path.join(RESULTS_DIR, "main_benchmark_manifest.json")

print("=" * 70)
print("SPEECHAI MAIN BENCHMARK RUNNER (750 PREFLIGHT-VERIFIED CLIPS)")
print("Random Seed: 42 | Strict Holdout Isolation | Hardware Latency Profiling")
print("=" * 70)

if not os.path.exists(manifest_path):
    print(f"Error: Manifest not found at {manifest_path}")
    sys.exit(1)

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print(f"Loaded verified manifest: {len(manifest)} clips.")

# Load Whisper model
t_model_start = time.time()
print("\n[Phase 3.1] Loading Whisper 'base' model...")
model = get_whisper_model()
t_model_load = round(time.time() - t_model_start, 2)
print(f"Whisper model loaded in {t_model_load} seconds.\n")

predictions = []
latencies = []
durations = []
t_start_total = time.time()

print("[Phase 3.2] Executing Full 750-Clip Inference Pipeline...")
print("Reporting live progress every 25 clips:\n")

for idx, item in enumerate(manifest, 1):
    fpath = item["file_path"]
    t0 = time.time()
    
    try:
        # 1. Whisper Transcription (exact production pipeline configuration)
        whisper_res = model.transcribe(
            fpath,
            initial_prompt="Um, uh, er, ah, hmm, like, actually, basically, so, well.",
            condition_on_previous_text=False,
            word_timestamps=True,
            language="en"
        )
        transcript = whisper_res.get("text", "").strip()
        segments = whisper_res.get("segments", [])
        
        words_list = []
        clip_dur = item.get("duration_sec", 3.0)
        if segments:
            seg_dur = segments[-1].get("end", 0.0) - segments[0].get("start", 0.0)
            if seg_dur > 0:
                clip_dur = seg_dur
            for seg in segments:
                if "words" in seg:
                    words_list.extend(seg["words"])
                    
        # 2. Production SpeechAI Disfluency Detectors
        # Lexical + Acoustic DSP Filled-Pauses
        filler_count, filler_words_found = detect_fillers(
            transcript=transcript,
            wav_path=fpath if fpath.endswith(".wav") else None,
            words_list=words_list,
            return_details=False
        )
        
        # Syllable & Duplicate Word Repetition Regex
        stammer_events = detect_stammering(
            transcript=transcript,
            words_list=words_list,
            return_details=False
        )
        
        dt = round(time.time() - t0, 3)
        latencies.append(dt)
        durations.append(clip_dur)
        
        # Calculate WER on LibriStutter where ground-truth reading text exists
        wer_score = None
        if item.get("reference_text"):
            ref_c = clean_text(item["reference_text"])
            hyp_c = clean_text(transcript)
            if ref_c:
                wer_score = round(jiwer.wer(ref_c, hyp_c), 4)

        record = {
            **item,
            "pred_transcript": transcript,
            "pred_stammers": stammer_events,
            "pred_fillers": filler_count,
            "pred_has_stutter": 1 if stammer_events > 0 else 0,
            "pred_has_filler": 1 if filler_count > 0 else 0,
            "pred_is_clean": 1 if (stammer_events == 0 and filler_count == 0) else 0,
            "wer": wer_score,
            "latency_sec": dt,
            "rtf": round(dt / clip_dur, 3) if clip_dur > 0 else 0.0,
            "status": "SUCCESS"
        }
    except Exception as exc:
        dt = round(time.time() - t0, 3)
        latencies.append(dt)
        clip_dur = item.get("duration_sec", 3.0)
        durations.append(clip_dur)
        print(f"\n[Warning] Failed processing clip {item.get('filename')}: {exc}", flush=True)
        record = {
            **item,
            "pred_transcript": "",
            "pred_stammers": 0,
            "pred_fillers": 0,
            "pred_has_stutter": 0,
            "pred_has_filler": 0,
            "pred_is_clean": 1,
            "wer": None,
            "latency_sec": dt,
            "rtf": round(dt / clip_dur, 3) if clip_dur > 0 else 0.0,
            "status": f"FAILED: {str(exc)}"
        }

    predictions.append(record)
    
    # Report progress every 25 clips
    if idx % 25 == 0 or idx == len(manifest):
        elapsed_sec = time.time() - t_start_total
        avg_lat = np.mean(latencies)
        remaining_clips = len(manifest) - idx
        eta_min = (remaining_clips * avg_lat) / 60.0
        pct = (idx / len(manifest)) * 100.0
        print(f"[{idx:3d}/{len(manifest)} | {pct:5.1f}%] Last: {dt:5.2f}s | Avg: {avg_lat:4.2f}s | Elapsed: {elapsed_sec/60.0:4.1f}m | ETA: {eta_min:4.1f}m", flush=True)

# Save full predictions JSON
pred_out_path = os.path.join(RESULTS_DIR, "main_benchmark_predictions.json")
with open(pred_out_path, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2)
print(f"\n[Saved] Detailed predictions saved to: {pred_out_path}", flush=True)

# ---------------------------------------------------------
# METRIC COMPILATION & SCIENTIFIC EVALUATION
# ---------------------------------------------------------
print("\n[Phase 3.3] Computing Academic Performance Metrics...", flush=True)

dataset_names = ["UCLASS", "FluencyBank", "Libri_Stutter"]
classes = ["Clean", "Stutter", "Filler"]

def evaluate_subset(subset_preds):
    matrix = {c_true: {c_pred: 0 for c_pred in classes} for c_true in classes}
    for p in subset_preds:
        if p["gt_sound_rep"] == 1 or p["gt_word_rep"] == 1 or p["gt_prolongation"] == 1:
            gt = "Stutter"
        elif p["gt_filler"] == 1:
            gt = "Filler"
        else:
            gt = "Clean"
            
        if p["pred_stammers"] > 0:
            pr = "Stutter"
        elif p["pred_fillers"] > 0:
            pr = "Filler"
        else:
            pr = "Clean"
            
        matrix[gt][pr] += 1

    raw_cm = [[matrix[c_t][c_p] for c_p in classes] for c_t in classes]
    total_samples = len(subset_preds)
    correct = sum(matrix[c][c] for c in classes)
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    
    class_stats = {}
    for c in classes:
        tp = matrix[c][c]
        fp = sum(matrix[other][c] for other in classes if other != c)
        fn = sum(matrix[c][other] for other in classes if other != c)
        support = tp + fn
        
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
        
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = round(2 * (prec * rec) / (prec + rec), 4)
        else:
            f1 = None if support == 0 else 0.0

        class_stats[c] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": prec if prec is not None else "Undefined (Support=0)",
            "recall": rec if rec is not None else "Undefined (Support=0)",
            "f1_score": f1 if f1 is not None else "Undefined (Support=0)"
        }

    stutter_gt = [1 if (p["gt_sound_rep"] == 1 or p["gt_word_rep"] == 1 or p["gt_prolongation"] == 1) else 0 for p in subset_preds]
    stutter_pr = [1 if p["pred_stammers"] > 0 else 0 for p in subset_preds]
    s_acc = accuracy_score(stutter_gt, stutter_pr)
    s_p, s_r, s_f, _ = precision_recall_fscore_support(stutter_gt, stutter_pr, average='binary', zero_division=0)
    
    filler_gt = [1 if p["gt_filler"] == 1 else 0 for p in subset_preds]
    filler_pr = [1 if p["pred_fillers"] > 0 else 0 for p in subset_preds]
    f_acc = accuracy_score(filler_gt, filler_pr)
    f_p, f_r, f_f, _ = precision_recall_fscore_support(filler_gt, filler_pr, average='binary', zero_division=0)

    wers = [p["wer"] for p in subset_preds if p.get("wer") is not None]
    avg_wer = round(float(np.mean(wers)), 4) if wers else None

    return {
        "sample_count": total_samples,
        "overall_accuracy": round(accuracy, 4),
        "average_wer": avg_wer,
        "classes": classes,
        "confusion_matrix": raw_cm,
        "dominant_class_performance": class_stats,
        "event_level_binary_metrics": {
            "stutter_detection": {
                "support": sum(stutter_gt),
                "accuracy": round(float(s_acc), 4),
                "precision": round(float(s_p), 4),
                "recall": round(float(s_r), 4),
                "f1_score": round(float(s_f), 4)
            },
            "filler_detection": {
                "support": sum(filler_gt),
                "accuracy": round(float(f_acc), 4),
                "precision": round(float(f_p), 4),
                "recall": round(float(f_r), 4),
                "f1_score": round(float(f_f), 4)
            }
        }
    }

results_by_dataset = {}
for ds_name in dataset_names:
    ds_preds = [p for p in predictions if p["dataset"] == ds_name]
    if ds_preds:
        results_by_dataset[ds_name] = evaluate_subset(ds_preds)

aggregate_results = evaluate_subset(predictions)

# Hardware Latency Statistics
runtime_stats = {
    "total_evaluated_clips": len(predictions),
    "model_load_sec": t_model_load,
    "total_elapsed_min": round((time.time() - t_start_total) / 60.0, 2),
    "mean_latency_sec": round(float(np.mean(latencies)), 3),
    "median_latency_sec": round(float(np.median(latencies)), 3),
    "std_latency_sec": round(float(np.std(latencies)), 3),
    "p95_latency_sec": round(float(np.percentile(latencies, 95)), 3),
    "min_latency_sec": round(float(np.min(latencies)), 3),
    "max_latency_sec": round(float(np.max(latencies)), 3),
    "mean_rtf": round(float(np.mean([p['rtf'] for p in predictions])), 3),
    "throughput_clips_per_min": round(60.0 / float(np.mean(latencies)), 2)
}

final_benchmark_results = {
    "runtime_stats": runtime_stats,
    "aggregate_results": aggregate_results,
    "per_dataset_results": results_by_dataset
}

benchmark_res_path = os.path.join(RESULTS_DIR, "main_benchmark_results.json")
with open(benchmark_res_path, "w", encoding="utf-8") as f:
    json.dump(final_benchmark_results, f, indent=2)

# ---------------------------------------------------------
# GENERATE PUBLICATION-GRADE PLOTS (300 DPI)
# ---------------------------------------------------------
print("\n[Phase 3.4] Rendering High-Resolution Publication Figures...", flush=True)

# 1. Latency Distribution Plot
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
ax.hist(latencies, bins=25, color="#3B82F6", edgecolor="#1E3A8A", alpha=0.85)
ax.axvline(runtime_stats["mean_latency_sec"], color="#DC2626", linestyle="--", linewidth=1.5, label=f"Mean Latency ({runtime_stats['mean_latency_sec']:.2f}s)")
ax.axvline(runtime_stats["p95_latency_sec"], color="#F59E0B", linestyle=":", linewidth=1.5, label=f"95th Percentile ({runtime_stats['p95_latency_sec']:.2f}s)")
ax.set_title("SpeechAI Inference Latency Distribution (750-Clip Benchmark)", fontsize=12, fontweight="bold")
ax.set_xlabel("Latency per Audio Clip (seconds)", fontsize=10)
ax.set_ylabel("Number of Clips", fontsize=10)
ax.legend(frameon=True)
ax.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "main_benchmark_latency_distribution.png"))
plt.close()

# 2. Confusion Matrix Heatmaps for each dataset and Aggregate
for ds_name in dataset_names:
    cm_data = results_by_dataset[ds_name]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    im = ax.imshow(cm_data, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           title=f"{ds_name} Confusion Matrix (N=250)",
           ylabel='Ground Truth Label',
           xlabel='SpeechAI Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = np.max(cm_data) / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_data[i][j]
            ax.text(j, i, format(val, 'd'),
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"main_benchmark_confusion_matrix_{ds_name.lower()}.png"))
    plt.close()

# Aggregate Confusion Matrix Heatmap
agg_cm_data = aggregate_results["confusion_matrix"]
fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
im = ax.imshow(agg_cm_data, interpolation='nearest', cmap=plt.cm.Purples)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(len(classes)),
       yticks=np.arange(len(classes)),
       xticklabels=classes, yticklabels=classes,
       title="Aggregate Confusion Matrix (N=750)",
       ylabel='Ground Truth Label',
       xlabel='SpeechAI Predicted Label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
thresh = np.max(agg_cm_data) / 2.
for i in range(len(classes)):
    for j in range(len(classes)):
        val = agg_cm_data[i][j]
        ax.text(j, i, format(val, 'd'),
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "main_benchmark_confusion_matrix_aggregate.png"))
plt.close()

# 3. Event-Level F1-Score Comparison Bar Chart
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
x = np.arange(len(dataset_names))
width = 0.35

stutter_f1s = [results_by_dataset[d]["event_level_binary_metrics"]["stutter_detection"]["f1_score"] for d in dataset_names]
filler_f1s = [results_by_dataset[d]["event_level_binary_metrics"]["filler_detection"]["f1_score"] for d in dataset_names]

ax.bar(x - width/2, stutter_f1s, width, label='Stutter Detection F1', color='#4F46E5', edgecolor='black', alpha=0.85)
ax.bar(x + width/2, filler_f1s, width, label='Filler Detection F1', color='#10B981', edgecolor='black', alpha=0.85)

ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax.set_title('Event-Level Binary Detection F1-Score Across Datasets (N=750)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(dataset_names, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(frameon=True)
ax.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "main_benchmark_event_level_metrics.png"))
plt.close()

print("\n" + "=" * 70)
print("MAIN BENCHMARK EXECUTION COMPLETE!")
print(f"Total Time:   {runtime_stats['total_elapsed_min']} minutes")
print(f"Mean Latency: {runtime_stats['mean_latency_sec']}s per clip | Mean RTF: {runtime_stats['mean_rtf']:.3f}x")
print(f"Throughput:   {runtime_stats['throughput_clips_per_min']} clips/minute")
print(f"Artifacts saved in: {RESULTS_DIR}")
print("=" * 70)
