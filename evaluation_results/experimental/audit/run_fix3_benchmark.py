"""
Run Fix 3 Benchmark on 30-Clip Development Benchmark
=====================================================
Evaluates:
- Baseline Fix 2B (Variant 7, enable_contextual_scoring=False)
- Fix 3 (enable_contextual_scoring=True)

Produces evaluation_results/experimental/dev_eval_results_30clips_fix3.json
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

audit_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(audit_dir)))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.experimental.config import ExperimentalConfig, StutterDetectorConfig, SpeechContextConfig
from backend.experimental.acoustic_features import AcousticFeatureExtractor
from backend.experimental.speech_context import SpeechContextAnalyzer
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.eval_harness import compute_binary_metrics, compute_multiclass_metrics, get_dominant_label


def run_benchmark():
    dev_json = os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips_refined.json")
    with open(dev_json, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    cfg = ExperimentalConfig()
    ac = AcousticFeatureExtractor(cfg.acoustic)
    context_analyzer = SpeechContextAnalyzer(cfg.context)
    
    # Detector instances
    detector_base = HybridStutterDetector(
        StutterDetectorConfig(enable_contextual_scoring=False),
        ac,
        context_analyzer
    )
    detector_fix3 = HybridStutterDetector(
        StutterDetectorConfig(enable_contextual_scoring=True),
        ac,
        context_analyzer
    )

    print("Pre-loading audio waveforms and extracting candidates for 30 dev clips...")
    clip_cache = []
    for idx, p in enumerate(dev_data["predictions"], 1):
        fname = p["filename"]
        fpath = p["file_path"]
        if not os.path.exists(fpath):
            fpath = os.path.join(project_root, "Datasets", p["dataset"], fname)
        
        y, sr = ac.load_audio(fpath)
        audio_dur = len(y) / sr if sr > 0 else 0.0
        transcript = p.get("pred_transcript", "")
        words_list = p.get("words_list", [])
        
        # Word timestamps fallback from transcript if empty
        if not words_list and transcript:
            words = [w for w in transcript.split() if w]
            if words:
                w_dur = audio_dur / len(words)
                words_list = [
                    {"word": w, "start": round(i * w_dur, 2), "end": round((i + 1) * w_dur, 2), "probability": 0.95}
                    for i, w in enumerate(words)
                ]
        
        clip_cache.append({
            "p": p,
            "filename": fname,
            "dataset": p["dataset"],
            "file_path": fpath,
            "y": y,
            "sr": sr,
            "audio_dur": audio_dur,
            "transcript": transcript,
            "words_list": words_list
        })
        if idx % 10 == 0 or idx == len(dev_data["predictions"]):
            print(f"  Loaded {idx}/{len(dev_data['predictions'])} clips...")

    # Run evaluations
    print("\n--- Evaluating Fix 2B Baseline & Fix 3 Contextual Calibration ---")
    results_base = []
    results_fix3 = []

    for item in clip_cache:
        p = item["p"]
        y = item["y"]
        sr = item["sr"]
        t = item["transcript"]
        wl = item["words_list"]

        # 1. Base Fix 2B
        res_b = detector_base.evaluate(transcript=t, words_list=wl, audio_data=(y, sr))
        has_stutter_b = res_b["has_stutter"]
        has_filler_b = p["pred_has_filler"]
        dom_b = get_dominant_label(has_stutter_b, has_filler_b)
        results_base.append({
            "filename": item["filename"],
            "dataset": item["dataset"],
            "gt_has_stutter": p["gt_has_stutter"],
            "gt_has_filler": p["gt_has_filler"],
            "gt_dominant": p["gt_dominant"],
            "pred_has_stutter": has_stutter_b,
            "pred_has_filler": has_filler_b,
            "pred_dominant": dom_b,
            "confidence": res_b["confidence_score"],
            "details": res_b["details"]
        })

        # 2. Fix 3 Context
        res_c = detector_fix3.evaluate(transcript=t, words_list=wl, audio_data=(y, sr))
        has_stutter_c = res_c["has_stutter"]
        has_filler_c = p["pred_has_filler"]
        dom_c = get_dominant_label(has_stutter_c, has_filler_c)
        results_fix3.append({
            "filename": item["filename"],
            "dataset": item["dataset"],
            "gt_has_stutter": p["gt_has_stutter"],
            "gt_has_filler": p["gt_has_filler"],
            "gt_dominant": p["gt_dominant"],
            "pred_has_stutter": has_stutter_c,
            "pred_has_filler": has_filler_c,
            "pred_dominant": dom_c,
            "confidence": res_c["confidence_score"],
            "details": res_c["details"]
        })

    # Compute Metrics
    labels = ["Stutter", "Filler", "Clean"]
    mc_b = compute_multiclass_metrics([r["gt_dominant"] for r in results_base], [r["pred_dominant"] for r in results_base], labels)
    st_b = compute_binary_metrics([r["gt_has_stutter"] for r in results_base], [r["pred_has_stutter"] for r in results_base])
    fl_b = compute_binary_metrics([r["gt_has_filler"] for r in results_base], [r["pred_has_filler"] for r in results_base])
    fb_fps_b = sum(1 for r in results_base if r["dataset"] == "FluencyBank" and r["pred_has_stutter"] == 1 and r["gt_has_stutter"] == 0)

    mc_c = compute_multiclass_metrics([r["gt_dominant"] for r in results_fix3], [r["pred_dominant"] for r in results_fix3], labels)
    st_c = compute_binary_metrics([r["gt_has_stutter"] for r in results_fix3], [r["pred_has_stutter"] for r in results_fix3])
    fl_c = compute_binary_metrics([r["gt_has_filler"] for r in results_fix3], [r["pred_has_filler"] for r in results_fix3])
    fb_fps_c = sum(1 for r in results_fix3 if r["dataset"] == "FluencyBank" and r["pred_has_stutter"] == 1 and r["gt_has_stutter"] == 0)

    print("\n================== BENCHMARK COMPARISON (N=30) ==================")
    print(f"Fix 2B Baseline : Acc={mc_b['accuracy']:.4f}, MacF1={mc_b['macro_f1']:.4f}, StPrec={st_b['precision']:.4f}, StRec={st_b['recall']:.4f}, StF1={st_b['f1']:.4f}, FB_FPs={fb_fps_b}")
    print(f"Fix 3 Context   : Acc={mc_c['accuracy']:.4f}, MacF1={mc_c['macro_f1']:.4f}, StPrec={st_c['precision']:.4f}, StRec={st_c['recall']:.4f}, StF1={st_c['f1']:.4f}, FB_FPs={fb_fps_c}")
    print("=================================================================\n")

    # Save to dev_eval_results_30clips_fix3.json
    output_payload = {
        "benchmark_name": "SpeechAI Fix 3 Syllable-Rate and Speech-Context Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_clips": len(clip_cache),
        "fix2b_baseline": {
            "multiclass_3class": mc_b,
            "stutter_binary": st_b,
            "filler_binary": fl_b,
            "fluencybank_stutter_fps": fb_fps_b
        },
        "fix3_contextual": {
            "multiclass_3class": mc_c,
            "stutter_binary": st_c,
            "filler_binary": fl_c,
            "fluencybank_stutter_fps": fb_fps_c
        },
        "predictions_fix3": results_fix3
    }

    out_path = os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips_fix3.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Saved Fix 3 benchmark results to: {out_path}")


if __name__ == "__main__":
    run_benchmark()
