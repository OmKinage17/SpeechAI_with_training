"""
Run Fix 4 Benchmark on 30-Clip Development Benchmark
=====================================================
Evaluates:
- Baseline Fix 2B (Variant 7, enable_contextual_scoring=False)
- Fix 3 Syllable Context (enable_contextual_scoring=True, enable_lexical_syntactic_context=False)
- Fix 4 Lexical & Syntactic Context (enable_contextual_scoring=True, enable_lexical_syntactic_context=True)

Produces evaluation_results/experimental/dev_eval_results_30clips_fix4.json
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
    
    # 1. Base Fix 2B (Context disabled)
    ctx_base = SpeechContextAnalyzer(SpeechContextConfig(enabled=False))
    det_base = HybridStutterDetector(
        StutterDetectorConfig(enable_contextual_scoring=False),
        ac,
        ctx_base
    )
    
    # 2. Fix 3 (Speech rate + Elongation + Flat boundary penalty, Lexical context disabled)
    ctx_fix3 = SpeechContextAnalyzer(SpeechContextConfig(enabled=True, enable_lexical_syntactic_context=False))
    det_fix3 = HybridStutterDetector(
        StutterDetectorConfig(enable_contextual_scoring=True),
        ac,
        ctx_fix3
    )
    
    # 3. Fix 4 (Speech rate + Elongation + Attenuated boundary + Lexical/Positional Context)
    ctx_fix4 = SpeechContextAnalyzer(SpeechContextConfig(enabled=True, enable_lexical_syntactic_context=True))
    det_fix4 = HybridStutterDetector(
        StutterDetectorConfig(enable_contextual_scoring=True),
        ac,
        ctx_fix4
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
    print("\n--- Evaluating Fix 2B Baseline, Fix 3, and Fix 4 Contextual Calibration ---")
    results_base = []
    results_fix3 = []
    results_fix4 = []

    for item in clip_cache:
        p = item["p"]
        y = item["y"]
        sr = item["sr"]
        t = item["transcript"]
        wl = item["words_list"]

        # 1. Base Fix 2B
        res_b = det_base.evaluate(transcript=t, words_list=wl, audio_data=(y, sr))
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

        # 2. Fix 3
        res_3 = det_fix3.evaluate(transcript=t, words_list=wl, audio_data=(y, sr))
        has_stutter_3 = res_3["has_stutter"]
        has_filler_3 = p["pred_has_filler"]
        dom_3 = get_dominant_label(has_stutter_3, has_filler_3)
        results_fix3.append({
            "filename": item["filename"],
            "dataset": item["dataset"],
            "gt_has_stutter": p["gt_has_stutter"],
            "gt_has_filler": p["gt_has_filler"],
            "gt_dominant": p["gt_dominant"],
            "pred_has_stutter": has_stutter_3,
            "pred_has_filler": has_filler_3,
            "pred_dominant": dom_3,
            "confidence": res_3["confidence_score"],
            "details": res_3["details"]
        })

        # 3. Fix 4
        res_4 = det_fix4.evaluate(transcript=t, words_list=wl, audio_data=(y, sr))
        has_stutter_4 = res_4["has_stutter"]
        has_filler_4 = p["pred_has_filler"]
        dom_4 = get_dominant_label(has_stutter_4, has_filler_4)
        results_fix4.append({
            "filename": item["filename"],
            "dataset": item["dataset"],
            "gt_has_stutter": p["gt_has_stutter"],
            "gt_has_filler": p["gt_has_filler"],
            "gt_dominant": p["gt_dominant"],
            "pred_has_stutter": has_stutter_4,
            "pred_has_filler": has_filler_4,
            "pred_dominant": dom_4,
            "confidence": res_4["confidence_score"],
            "details": res_4["details"]
        })

    # Compute Metrics
    labels = ["Stutter", "Filler", "Clean"]
    
    # Multiclass & binary for all three variants
    mc_b = compute_multiclass_metrics([r["gt_dominant"] for r in results_base], [r["pred_dominant"] for r in results_base], labels)
    st_b = compute_binary_metrics([r["gt_has_stutter"] for r in results_base], [r["pred_has_stutter"] for r in results_base])
    fl_b = compute_binary_metrics([r["gt_has_filler"] for r in results_base], [r["pred_has_filler"] for r in results_base])
    fb_fps_b = sum(1 for r in results_base if r["dataset"] == "FluencyBank" and r["pred_has_stutter"] == 1 and r["gt_has_stutter"] == 0)

    mc_3 = compute_multiclass_metrics([r["gt_dominant"] for r in results_fix3], [r["pred_dominant"] for r in results_fix3], labels)
    st_3 = compute_binary_metrics([r["gt_has_stutter"] for r in results_fix3], [r["pred_has_stutter"] for r in results_fix3])
    fl_3 = compute_binary_metrics([r["gt_has_filler"] for r in results_fix3], [r["pred_has_filler"] for r in results_fix3])
    fb_fps_3 = sum(1 for r in results_fix3 if r["dataset"] == "FluencyBank" and r["pred_has_stutter"] == 1 and r["gt_has_stutter"] == 0)

    mc_4 = compute_multiclass_metrics([r["gt_dominant"] for r in results_fix4], [r["pred_dominant"] for r in results_fix4], labels)
    st_4 = compute_binary_metrics([r["gt_has_stutter"] for r in results_fix4], [r["pred_has_stutter"] for r in results_fix4])
    fl_4 = compute_binary_metrics([r["gt_has_filler"] for r in results_fix4], [r["pred_has_filler"] for r in results_fix4])
    fb_fps_4 = sum(1 for r in results_fix4 if r["dataset"] == "FluencyBank" and r["pred_has_stutter"] == 1 and r["gt_has_stutter"] == 0)

    # Per-corpus breakdown for Fix 4
    corpora = sorted(list(set(r["dataset"] for r in results_fix4)))
    corpus_metrics = {}
    for c in corpora:
        sub = [r for r in results_fix4 if r["dataset"] == c]
        mc_c = compute_multiclass_metrics([r["gt_dominant"] for r in sub], [r["pred_dominant"] for r in sub], labels)
        st_c = compute_binary_metrics([r["gt_has_stutter"] for r in sub], [r["pred_has_stutter"] for r in sub])
        fl_c = compute_binary_metrics([r["gt_has_filler"] for r in sub], [r["pred_has_filler"] for r in sub])
        corpus_metrics[c] = {
            "n_clips": len(sub),
            "accuracy": mc_c["accuracy"],
            "macro_f1": mc_c["macro_f1"],
            "stutter": st_c,
            "filler": fl_c
        }

    print("\n================== BENCHMARK COMPARISON (N=30) ==================")
    print(f"Fix 2B Baseline : Acc={mc_b['accuracy']:.4f}, MacF1={mc_b['macro_f1']:.4f}, StPrec={st_b['precision']:.4f}, StRec={st_b['recall']:.4f}, StF1={st_b['f1']:.4f}, FB_FPs={fb_fps_b}")
    print(f"Fix 3 Context   : Acc={mc_3['accuracy']:.4f}, MacF1={mc_3['macro_f1']:.4f}, StPrec={st_3['precision']:.4f}, StRec={st_3['recall']:.4f}, StF1={st_3['f1']:.4f}, FB_FPs={fb_fps_3}")
    print(f"Fix 4 Context   : Acc={mc_4['accuracy']:.4f}, MacF1={mc_4['macro_f1']:.4f}, StPrec={st_4['precision']:.4f}, StRec={st_4['recall']:.4f}, StF1={st_4['f1']:.4f}, FB_FPs={fb_fps_4}")
    print("=================================================================\n")

    output_payload = {
        "benchmark_name": "SpeechAI Fix 4 Lexical and Syntactic Context Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_clips": len(clip_cache),
        "fix2b_baseline": {
            "multiclass_3class": mc_b,
            "stutter_binary": st_b,
            "filler_binary": fl_b,
            "fluencybank_stutter_fps": fb_fps_b
        },
        "fix3_contextual": {
            "multiclass_3class": mc_3,
            "stutter_binary": st_3,
            "filler_binary": fl_3,
            "fluencybank_stutter_fps": fb_fps_3
        },
        "fix4_contextual": {
            "multiclass_3class": mc_4,
            "stutter_binary": st_4,
            "filler_binary": fl_4,
            "fluencybank_stutter_fps": fb_fps_4,
            "per_corpus_metrics": corpus_metrics
        },
        "predictions_fix4": results_fix4
    }

    out_path = os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips_fix4.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Saved Fix 4 benchmark results to: {out_path}")

if __name__ == "__main__":
    run_benchmark()
