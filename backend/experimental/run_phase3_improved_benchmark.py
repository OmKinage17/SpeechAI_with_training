"""
SpeechAI — Phase 3 Improved Benchmark Runner (Final Evaluation Gate)
===================================================================
Strict Scope: Confined to backend/experimental/ and evaluation_results/experimental/phase3_improvement/.
Does NOT modify any production files, live models, or the main frontend.
Does NOT modify original benchmark artifacts in run_final_20261020_complete/ or phase2_corrected/.

Evaluates Candidate 1 (Standalone Short Prolongation Calibration) on the locked 120-clip holdout set.
"""

import os
import re
import sys
import time
import json
import csv
import hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent.parent
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.experimental.config import (
    ASRConfig,
    StutterDetectorConfig,
    SpeechContextConfig,
    AcousticConfig
)
from backend.experimental.asr_engine import detect_and_sanitize_repetition_loop
from backend.experimental.acoustic_features import AcousticFeatureExtractor
from backend.experimental.speech_context import SpeechContextAnalyzer
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.canonical_evaluator import (
    CanonicalEvaluator,
    compute_levenshtein_wer,
    compute_levenshtein_cer
)
from backend.video_module.feedback import generate_multimodal_feedback

# Directories
ORIGINAL_BENCHMARK_DIR = project_root / "evaluation_results" / "experimental" / "final_benchmark" / "run_final_20261020_complete"
PHASE2_DIR = project_root / "evaluation_results" / "experimental" / "phase2_corrected"
OUTPUT_DIR = project_root / "evaluation_results" / "experimental" / "phase3_improvement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_SRC = ORIGINAL_BENCHMARK_DIR / "run_manifest.json"
CACHE_SRC = ORIGINAL_BENCHMARK_DIR / "feature_cache_final.json"

MANIFEST_COPY_PATH = OUTPUT_DIR / "run_manifest.json"
FEATURE_CACHE_PATH = OUTPUT_DIR / "feature_cache_phase3.json"
PER_CLIP_PRED_PATH = OUTPUT_DIR / "per_clip_predictions.json"
PER_CLIP_CSV = OUTPUT_DIR / "per_clip_results.csv"
FINAL_METRICS_PATH = OUTPUT_DIR / "final_metrics.json"
PER_DATASET_METRICS_PATH = OUTPUT_DIR / "per_dataset_metrics.json"
PER_PARAM_METRICS_PATH = OUTPUT_DIR / "per_parameter_metrics.json"
CONFUSION_MATRICES_PATH = OUTPUT_DIR / "confusion_matrices.json"
RUNTIME_FAILURES_PATH = OUTPUT_DIR / "runtime_and_failures.json"
SEVEN_PARAM_CSV = OUTPUT_DIR / "seven_parameters_summary.csv"
PHASE3_REPORT_MD = OUTPUT_DIR / "PHASE3_IMPROVEMENT_REPORT.md"

CM_PLOT_PATH = OUTPUT_DIR / "multiclass_confusion_matrix.png"
SPEECH_RATE_PLOT_PATH = OUTPUT_DIR / "speech_rate_distribution.png"

DATASETS_DIR = project_root / "Datasets"
LIBRI_TRANSCRIPTS_DIR = DATASETS_DIR / "Libri_Stutter" / "LibriStutter Transcripts"


def detect_fillers_canonical(transcript: str, words_list: Optional[list] = None) -> Tuple[int, List[str]]:
    """Standard baseline conversational and acoustic filler detection."""
    text_lower = transcript.lower()
    filler_count = 0
    fillers_found = []

    phrase_fillers = [
        "you know", "i mean", "sort of", "kind of",
        "you see", "as in"
    ]
    remaining_text = text_lower
    for phrase in phrase_fillers:
        pattern = rf'\b{re.escape(phrase)}\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            cnt = len(matches)
            filler_count += cnt
            fillers_found.extend([phrase] * cnt)
            remaining_text = re.sub(pattern, " ", remaining_text)

    single_fillers = ["um", "uh", "er", "erm", "ah", "hmm", "like", "well", "so"]
    words = re.findall(r"\b[a-z']+\b", remaining_text)
    for w in words:
        if w in single_fillers:
            filler_count += 1
            fillers_found.append(w)

    return filler_count, fillers_found


def run_phase3_locked_evaluation():
    print("=" * 80)
    print("SPEECHAI PHASE 3: LOCKED 120-CLIP CANDIDATE EVALUATION")
    print("Target Candidate: Standalone Short Prolongation Calibration (0.45s floor)")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 80)

    # 1. Load Locked Manifest and Pre-extracted Feature Cache
    assert MANIFEST_SRC.exists(), f"Missing original locked manifest: {MANIFEST_SRC}"
    assert CACHE_SRC.exists(), f"Missing original feature cache: {CACHE_SRC}"

    with open(MANIFEST_SRC, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(CACHE_SRC, "r", encoding="utf-8") as f:
        feature_cache = json.load(f)

    assert len(manifest) == 120, f"Expected 120 clips in manifest, got {len(manifest)}"

    # Copy manifest into Phase 3 output directory
    with open(MANIFEST_COPY_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 2. Initialize Engines with Phase 3 Configuration
    acoustic_cfg = AcousticConfig()
    acoustic_extractor = AcousticFeatureExtractor(acoustic_cfg)

    context_cfg = SpeechContextConfig(
        enabled=True,
        fast_speech_threshold=5.5,
        slow_speech_threshold=2.8,
        min_context_score_for_admission=0.65,
        elongation_ratio_threshold=2.0
    )
    context_analyzer = SpeechContextAnalyzer(context_cfg)

    # Phase 3 Candidate 1 Configuration
    stutter_config = StutterDetectorConfig(
        enable_contextual_scoring=True,
        confident_threshold=0.65,
        uncertain_threshold=0.50,
        min_prolongation_duration_sec=0.50,
        severe_prolongation_duration_sec=0.60,
        standalone_prolongation_confidence=0.50,
        standalone_prolongation_min_duration_sec=0.45,  # Phase 3 Improvement
        enable_ngram_repetition=True,
        ngram_sizes=(2, 3),
        ngram_repetition_confidence=0.85,
        enable_filler_boundary_gating=True,
        filler_overlap_threshold=0.35,
        filler_word_coverage_threshold=0.40,
        boundary_prolongation_min_duration_sec=0.65,
        gate_filler_blocks=True,
        enable_interrupted_repetition=True,
        interrupted_repetition_max_skip=3,
        interrupted_repetition_allow_stem=True,
        interrupted_repetition_allow_hesitations=True,
        interrupted_repetition_confidence=0.85
    )
    detector = HybridStutterDetector(
        config=stutter_config,
        acoustic_extractor=acoustic_extractor,
        context_analyzer=context_analyzer
    )

    # 3. Process all 120 clips with ASR Loop Sanitizer + Calibrated Detector
    print("\nProcessing all 120 locked clips through Repetition Loop Handler and Calibrated Detector...")
    phase3_cache = {}
    predictions = []
    latencies = []
    loop_recovered_count = 0

    t_start = time.time()
    for item in manifest:
        fn = item["filename"]
        cached = feature_cache[fn]
        fpath = item["file_path"]
        corpus = item["dataset"]
        speaker_id = item["speaker_id"]
        gt_dom = item["gt_dominant"]
        gt_st = item["gt_has_stutter"]
        gt_fl = item["gt_has_filler"]
        ann_details = item.get("annotation_details", {})

        raw_transcript = cached.get("transcript", "")
        raw_words = cached.get("words_list", [])

        # Load audio for acoustic features
        y, sr = acoustic_extractor.load_audio(fpath)
        clip_dur = len(y) / sr if sr > 0 else 3.0

        # Run ASR Repetition Loop Detection and Sanitization (identical to Phase 2)
        loop_res = detect_and_sanitize_repetition_loop(
            raw_transcript=raw_transcript,
            raw_words_list=raw_words,
            clip_duration_sec=clip_dur
        )

        is_loop = loop_res["is_loop"]
        if is_loop:
            loop_recovered_count += 1

        sanitized_transcript = loop_res["sanitized_transcript"]
        sanitized_words = loop_res["sanitized_words_list"]

        # Dual filler measures
        filler_cnt, filler_types = detect_fillers_canonical(sanitized_transcript, sanitized_words)
        pred_has_filler_event = 1 if filler_cnt > 0 else 0

        # Run Calibrated Hybrid Stutter Detector
        t_inf0 = time.time()
        det_res = detector.evaluate(
            transcript=sanitized_transcript,
            words_list=sanitized_words,
            audio_data=(y, sr),
            pred_has_filler=pred_has_filler_event
        )
        dt_inf = time.time() - t_inf0
        latencies.append(dt_inf)

        pred_dominant = det_res["pred_dominant"]
        pred_has_stutter = det_res["has_stutter"]
        pred_has_filler_dominant = 1 if pred_dominant == "Filler" else 0
        conf = det_res["confidence_score"]

        # Speech rate calculations
        raw_wc = len(raw_transcript.split())
        san_wc = len(sanitized_transcript.split())
        dur_min = clip_dur / 60.0 if clip_dur > 0 else 0.05
        raw_wpm = round(raw_wc / dur_min, 1)
        sanitized_wpm = round(san_wc / dur_min, 1)

        pauses = []
        total_pause_sec = 0.0
        if sanitized_words and len(sanitized_words) > 1:
            for w_idx in range(len(sanitized_words) - 1):
                gap = float(sanitized_words[w_idx + 1].get("start", 0.0)) - float(sanitized_words[w_idx].get("end", 0.0))
                if gap >= 0.25:
                    pauses.append(gap)
                    total_pause_sec += gap

        articulation_dur_sec = max(0.5, clip_dur - total_pause_sec)
        articulation_wpm = round(san_wc / (articulation_dur_sec / 60.0), 1)

        # Ground truth transcript for Libri_Stutter
        ref_text = None
        ref_wpm = None
        wer_val, cer_val = None, None
        edits_val, ref_words_val = None, None

        if corpus == "Libri_Stutter":
            parts = fn.replace(".flac", "").replace(".wav", "").split("-")
            if len(parts) >= 3:
                r_id, ch_id, u_id = parts[0], parts[1], parts[2]
                ref_path = LIBRI_TRANSCRIPTS_DIR / r_id / ch_id / f"{r_id}-{ch_id}-{u_id}.txt"
                if ref_path.exists():
                    with open(ref_path, "r", encoding="utf-8") as rf:
                        ref_text = rf.read().strip()
                    ref_word_count = len(ref_text.split())
                    ref_wpm = round(ref_word_count / dur_min, 1)
                    wer_val, edits_val, ref_words_val, _, _ = compute_levenshtein_wer(ref_text, sanitized_transcript)
                    cer_val, _, _ = compute_levenshtein_cer(ref_text, sanitized_transcript)

        gt_block = 1 if ann_details.get("block", 0) > 0 else 0
        pred_block = 1 if det_res["acoustic_block_events"] > 0 else 0

        # Communication Coaching Verification
        speech_feat_dict = {
            "transcript": sanitized_transcript,
            "word_count": san_wc,
            "wpm": sanitized_wpm,
            "filler_count": filler_cnt,
            "filler_types": filler_types,
            "repetition_count": det_res["lexical_repetition_events"],
            "long_pauses": len([p for p in pauses if p >= 1.5]),
            "pause_ratio": total_pause_sec / max(0.1, clip_dur)
        }
        feedback_items = generate_multimodal_feedback(
            speech=speech_feat_dict,
            visual={"face_presence_ratio": 0.8, "camera_facing_ratio": 0.8, "head_motion": 0.02},
            scores={"clarity": 85.0}
        )
        coaching_valid = True
        if sanitized_wpm > 155.0 and not any("faster than optimal" in fb["message"] for fb in feedback_items):
            coaching_valid = False
        if sanitized_wpm < 110.0 and sanitized_wpm > 0.0 and not any("slower than typical" in fb["message"] for fb in feedback_items):
            coaching_valid = False
        if (filler_cnt / max(1, san_wc)) * 100.0 > 4.0 and not any("filler words" in fb["message"] for fb in feedback_items):
            coaching_valid = False

        phase3_cache[fn] = {
            "filename": fn,
            "file_path": fpath,
            "dataset": corpus,
            "speaker_id": speaker_id,
            "gt_dominant": gt_dom,
            "gt_has_stutter": gt_st,
            "gt_has_filler": gt_fl,
            "raw_transcript": raw_transcript,
            "sanitized_transcript": sanitized_transcript,
            "raw_words_list": raw_words,
            "sanitized_words_list": sanitized_words,
            "is_loop_detected": is_loop,
            "status": loop_res["status"],
            "sanitization_reason": loop_res["reason"],
            "loop_pattern": loop_res["loop_pattern"],
            "truncated_tokens_count": loop_res["truncated_tokens_count"],
            "filler_count": filler_cnt,
            "filler_types": filler_types,
            "pred_has_filler_event": pred_has_filler_event,
            "latency_sec": round(dt_inf, 3)
        }

        predictions.append({
            "filename": fn,
            "dataset": corpus,
            "speaker_id": speaker_id,
            "clip_duration_sec": round(clip_dur, 2),
            "gt_dominant": gt_dom,
            "gt_has_stutter": gt_st,
            "gt_has_filler": gt_fl,
            "pred_dominant": pred_dominant,
            "pred_has_stutter": pred_has_stutter,
            "pred_has_filler": pred_has_filler_dominant,
            "pred_has_filler_event": pred_has_filler_event,
            "filler_count": filler_cnt,
            "gt_block": gt_block,
            "pred_block": pred_block,
            "raw_wpm": raw_wpm,
            "sanitized_wpm": sanitized_wpm,
            "articulation_wpm": articulation_wpm,
            "is_loop_detected": is_loop,
            "status": loop_res["status"],
            "confidence": round(conf, 3),
            "classification": det_res["classification"],
            "lexical_repetition_events": det_res["lexical_repetition_events"],
            "acoustic_prolongation_events": det_res["acoustic_prolongation_events"],
            "acoustic_block_events": det_res["acoustic_block_events"],
            "raw_transcript": raw_transcript,
            "sanitized_transcript": sanitized_transcript,
            "ref_transcript": ref_text,
            "ref_wpm": ref_wpm,
            "wer": round(wer_val, 4) if wer_val is not None else None,
            "cer": round(cer_val, 4) if cer_val is not None else None,
            "edits": edits_val,
            "ref_words": ref_words_val,
            "coaching_rules_valid": coaching_valid,
            "details": det_res["details"]
        })

    total_proc_time = time.time() - t_start
    print(f"Processed 120 clips in {total_proc_time:.2f}s (Avg inference: {np.mean(latencies)*1000:.1f}ms/clip)")

    # Save feature cache and per-clip predictions
    with open(FEATURE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(phase3_cache, f, indent=2)
    with open(PER_CLIP_PRED_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    # Export CSV
    csv_fields = [
        "filename", "dataset", "speaker_id", "clip_duration_sec",
        "gt_dominant", "pred_dominant", "gt_has_stutter", "pred_has_stutter",
        "gt_has_filler", "pred_has_filler_event", "filler_count",
        "gt_block", "pred_block", "raw_wpm", "sanitized_wpm", "articulation_wpm",
        "is_loop_detected", "status", "confidence", "classification",
        "lexical_repetition_events", "acoustic_prolongation_events", "acoustic_block_events",
        "wer", "cer", "coaching_rules_valid"
    ]
    with open(PER_CLIP_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)

    # 4. Canonical Metrics Aggregation
    classes = ["Stutter", "Filler", "Clean"]
    agg = {
        "total_clips": len(predictions),
        "dominant_multiclass": {
            "classes": classes,
            "confusion_matrix": {c1: {c2: 0 for c2 in classes} for c1 in classes},
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0
        },
        "class_metrics": {},
        "stutter_binary": {},
        "filler_evaluation": {
            "event_level": {},
            "dominant_class": {}
        },
        "speech_rate": {},
        "block_detection": {
            "scoped_clinical_uclass_fluencybank": {},
            "full_benchmark": {}
        },
        "asr_recognition": {},
        "coaching_compliance": {}
    }

    # Dominant Multiclass CM
    for p in predictions:
        gt = p["gt_dominant"]
        pred = p["pred_dominant"]
        agg["dominant_multiclass"]["confusion_matrix"][gt][pred] += 1

    cm = agg["dominant_multiclass"]["confusion_matrix"]
    correct_dom = sum(cm[c][c] for c in classes)
    dom_acc = correct_dom / len(predictions)
    agg["dominant_multiclass"]["accuracy"] = round(dom_acc, 4)

    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in classes if other != c)
        fn = sum(cm[c][other] for other in classes if other != c)
        tn = sum(cm[o1][o2] for o1 in classes if o1 != c for o2 in classes if o2 != c)
        p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_val = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
        spec_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sup = sum(cm[c][other] for other in classes)
        agg["class_metrics"][c] = {
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "f1": round(f_val, 4),
            "specificity": round(spec_val, 4),
            "support": sup,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        }

    macro_f1 = np.mean([agg["class_metrics"][c]["f1"] for c in classes])
    weighted_f1 = sum(agg["class_metrics"][c]["f1"] * agg["class_metrics"][c]["support"] for c in classes) / len(predictions)
    agg["dominant_multiclass"]["macro_f1"] = round(float(macro_f1), 4)
    agg["dominant_multiclass"]["weighted_f1"] = round(float(weighted_f1), 4)

    # Binary Stutter Metrics
    s_tp = sum(1 for p in predictions if p["gt_has_stutter"] == 1 and p["pred_has_stutter"] == 1)
    s_fp = sum(1 for p in predictions if p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 1)
    s_fn = sum(1 for p in predictions if p["gt_has_stutter"] == 1 and p["pred_has_stutter"] == 0)
    s_tn = sum(1 for p in predictions if p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 0)
    s_p = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
    s_r = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
    s_f = 2 * s_p * s_r / (s_p + s_r) if (s_p + s_r) > 0 else 0.0
    s_spec = s_tn / (s_tn + s_fp) if (s_tn + s_fp) > 0 else 0.0
    agg["stutter_binary"] = {
        "tp": s_tp, "fp": s_fp, "fn": s_fn, "tn": s_tn,
        "precision": round(s_p, 4),
        "recall": round(s_r, 4),
        "f1": round(s_f, 4),
        "specificity": round(s_spec, 4),
        "support": s_tp + s_fn
    }

    # Filler Evaluation
    fe_tp = sum(1 for p in predictions if p["gt_has_filler"] == 1 and p["pred_has_filler_event"] == 1)
    fe_fp = sum(1 for p in predictions if p["gt_has_filler"] == 0 and p["pred_has_filler_event"] == 1)
    fe_fn = sum(1 for p in predictions if p["gt_has_filler"] == 1 and p["pred_has_filler_event"] == 0)
    fe_tn = sum(1 for p in predictions if p["gt_has_filler"] == 0 and p["pred_has_filler_event"] == 0)
    fe_p = fe_tp / (fe_tp + fe_fp) if (fe_tp + fe_fp) > 0 else 0.0
    fe_r = fe_tp / (fe_tp + fe_fn) if (fe_tp + fe_fn) > 0 else 0.0
    fe_f = 2 * fe_p * fe_r / (fe_p + fe_r) if (fe_p + fe_r) > 0 else 0.0
    fe_spec = fe_tn / (fe_tn + fe_fp) if (fe_tn + fe_fp) > 0 else 0.0
    agg["filler_evaluation"]["event_level"] = {
        "tp": fe_tp, "fp": fe_fp, "fn": fe_fn, "tn": fe_tn,
        "precision": round(fe_p, 4), "recall": round(fe_r, 4),
        "f1": round(fe_f, 4), "specificity": round(fe_spec, 4),
        "support": fe_tp + fe_fn,
        "total_positive_predictions": fe_tp + fe_fp
    }

    fd_tp = sum(1 for p in predictions if p["gt_dominant"] == "Filler" and p["pred_dominant"] == "Filler")
    fd_fp = sum(1 for p in predictions if p["gt_dominant"] != "Filler" and p["pred_dominant"] == "Filler")
    fd_fn = sum(1 for p in predictions if p["gt_dominant"] == "Filler" and p["pred_dominant"] != "Filler")
    fd_tn = sum(1 for p in predictions if p["gt_dominant"] != "Filler" and p["pred_dominant"] != "Filler")
    fd_p = fd_tp / (fd_tp + fd_fp) if (fd_tp + fd_fp) > 0 else 0.0
    fd_r = fd_tp / (fd_tp + fd_fn) if (fd_tp + fd_fn) > 0 else 0.0
    fd_f = 2 * fd_p * fd_r / (fd_p + fd_r) if (fd_p + fd_r) > 0 else 0.0
    fd_spec = fd_tn / (fd_tn + fd_fp) if (fd_tn + fd_fp) > 0 else 0.0
    agg["filler_evaluation"]["dominant_class"] = {
        "tp": fd_tp, "fp": fd_fp, "fn": fd_fn, "tn": fd_tn,
        "precision": round(fd_p, 4), "recall": round(fd_r, 4),
        "f1": round(fd_f, 4), "specificity": round(fd_spec, 4),
        "support": fd_tp + fd_fn,
        "total_positive_predictions": fd_tp + fd_fp
    }

    # Speech Rate
    valid_clips = [p for p in predictions if not p["is_loop_detected"]]
    deg_clips = [p for p in predictions if p["is_loop_detected"]]
    san_wpms_all = [p["sanitized_wpm"] for p in predictions]
    raw_wpms_all = [p["raw_wpm"] for p in predictions]
    valid_wpms = [p["sanitized_wpm"] for p in valid_clips]

    libri_preds = [p for p in predictions if p["dataset"] == "Libri_Stutter" and p["ref_wpm"] is not None and not p["is_loop_detected"]]
    corr_val, p_val = None, None
    if len(libri_preds) >= 5:
        ref_rates = [p["ref_wpm"] for p in libri_preds]
        hyp_rates = [p["sanitized_wpm"] for p in libri_preds]
        corr_val, p_val = stats.pearsonr(ref_rates, hyp_rates)

    agg["speech_rate"] = {
        "valid_non_loop_clips_count": len(valid_clips),
        "degenerate_loop_clips_count": len(deg_clips),
        "valid_mean_wpm": round(float(np.mean(valid_wpms)), 2) if valid_wpms else 0.0,
        "valid_median_wpm": round(float(np.median(valid_wpms)), 2) if valid_wpms else 0.0,
        "valid_std_wpm": round(float(np.std(valid_wpms)), 2) if valid_wpms else 0.0,
        "all_clips_sanitized_mean_wpm": round(float(np.mean(san_wpms_all)), 2),
        "all_clips_raw_unfiltered_mean_wpm": round(float(np.mean(raw_wpms_all)), 2),
        "libri_verbatim_pearson_r": round(float(corr_val), 4) if corr_val is not None else None,
        "libri_verbatim_p_value": float(p_val) if p_val is not None else None,
        "libri_evaluated_clips": len(libri_preds)
    }

    # Scoped Block Detection
    clinical_preds = [p for p in predictions if p["dataset"] in ("UCLASS", "FluencyBank")]
    b_tp = sum(1 for p in clinical_preds if p["gt_block"] == 1 and p["pred_block"] == 1)
    b_fp = sum(1 for p in clinical_preds if p["gt_block"] == 0 and p["pred_block"] == 1)
    b_fn = sum(1 for p in clinical_preds if p["gt_block"] == 1 and p["pred_block"] == 0)
    b_tn = sum(1 for p in clinical_preds if p["gt_block"] == 0 and p["pred_block"] == 0)
    b_p = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
    b_r = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
    b_f = 2 * b_p * b_r / (b_p + b_r) if (b_p + b_r) > 0 else 0.0
    b_spec = b_tn / (b_tn + b_fp) if (b_tn + b_fp) > 0 else 0.0
    agg["block_detection"]["scoped_clinical_uclass_fluencybank"] = {
        "clips_evaluated": len(clinical_preds),
        "tp": b_tp, "fp": b_fp, "fn": b_fn, "tn": b_tn,
        "precision": round(b_p, 4), "recall": round(b_r, 4),
        "f1": round(b_f, 4), "specificity": round(b_spec, 4),
        "support": b_tp + b_fn
    }

    # Full benchmark block metrics
    fb_tp = sum(1 for p in predictions if p["gt_block"] == 1 and p["pred_block"] == 1)
    fb_fp = sum(1 for p in predictions if p["gt_block"] == 0 and p["pred_block"] == 1)
    fb_fn = sum(1 for p in predictions if p["gt_block"] == 1 and p["pred_block"] == 0)
    fb_tn = sum(1 for p in predictions if p["gt_block"] == 0 and p["pred_block"] == 0)
    fb_spec = fb_tn / (fb_tn + fb_fp) if (fb_tn + fb_fp) > 0 else 0.0
    agg["block_detection"]["full_benchmark"] = {
        "clips_evaluated": len(predictions),
        "tp": fb_tp, "fp": fb_fp, "fn": fb_fn, "tn": fb_tn,
        "specificity": round(fb_spec, 4)
    }

    # ASR WER/CER
    libri_all = [p for p in predictions if p["dataset"] == "Libri_Stutter" and p["wer"] is not None]
    total_edits = sum(p["edits"] for p in libri_all)
    total_ref_words = sum(p["ref_words"] for p in libri_all)
    macro_wer = np.mean([p["wer"] for p in libri_all]) if libri_all else 0.0
    micro_wer = total_edits / total_ref_words if total_ref_words > 0 else 0.0
    agg["asr_recognition"] = {
        "libri_clips_evaluated": len(libri_all),
        "total_reference_words": total_ref_words,
        "total_edit_distance": total_edits,
        "micro_wer": round(float(micro_wer), 4),
        "macro_wer": round(float(macro_wer), 4),
        "cer": round(float(np.mean([p["cer"] for p in libri_all])), 4) if libri_all else 0.0
    }

    # Coaching Compliance
    coaching_pass = sum(1 for p in predictions if p["coaching_rules_valid"])
    agg["coaching_compliance"] = {
        "total_clips": len(predictions),
        "compliant_clips": coaching_pass,
        "compliance_rate": round(coaching_pass / len(predictions), 4)
    }

    # 5. Load Phase 2 Corrected Baseline for Pairwise Comparison
    phase2_final_file = PHASE2_DIR / "final_metrics.json"
    p2_metrics = {}
    if phase2_final_file.exists():
        with open(phase2_final_file, "r", encoding="utf-8") as f:
            p2_metrics = json.load(f)

    # 6. Save JSON Metrics
    final_metrics_output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": "phase3_improved_holdout_120",
        "benchmark_clips": len(predictions),
        "configuration": {
            "standalone_prolongation_min_duration_sec": stutter_config.standalone_prolongation_min_duration_sec,
            "severe_prolongation_duration_sec": stutter_config.severe_prolongation_duration_sec,
            "confident_threshold": stutter_config.confident_threshold,
            "uncertain_threshold": stutter_config.uncertain_threshold,
            "enable_contextual_scoring": stutter_config.enable_contextual_scoring,
            "enable_ngram_repetition": stutter_config.enable_ngram_repetition,
            "enable_filler_boundary_gating": stutter_config.enable_filler_boundary_gating,
            "enable_interrupted_repetition": stutter_config.enable_interrupted_repetition
        },
        "multiclass_classification": agg["dominant_multiclass"],
        "per_class_metrics": agg["class_metrics"],
        "binary_stutter_metrics": agg["stutter_binary"],
        "filler_evaluation": agg["filler_evaluation"],
        "speech_rate": agg["speech_rate"],
        "block_detection": agg["block_detection"],
        "asr_recognition": agg["asr_recognition"],
        "coaching_engine": agg["coaching_compliance"],
        "runtime": {
            "total_processing_time_sec": round(total_proc_time, 2),
            "mean_latency_ms": round(float(np.mean(latencies)) * 1000.0, 1),
            "max_latency_ms": round(float(np.max(latencies)) * 1000.0, 1),
            "timeout_count": 0,
            "loop_recovered_count": loop_recovered_count
        }
    }

    with open(FINAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics_output, f, indent=2)

    # Confusion matrices
    confusion_matrices_dict = {
        "multiclass_dominant": agg["dominant_multiclass"]["confusion_matrix"],
        "stutter_binary_tp_fp_fn_tn": agg["stutter_binary"],
        "filler_event_tp_fp_fn_tn": agg["filler_evaluation"]["event_level"],
        "filler_dominant_tp_fp_fn_tn": agg["filler_evaluation"]["dominant_class"],
        "block_clinical_scoped_tp_fp_fn_tn": agg["block_detection"]["scoped_clinical_uclass_fluencybank"],
        "block_full_benchmark_tp_fp_fn_tn": agg["block_detection"]["full_benchmark"]
    }
    with open(CONFUSION_MATRICES_PATH, "w", encoding="utf-8") as f:
        json.dump(confusion_matrices_dict, f, indent=2)

    # Seven parameters summary
    p1 = agg["stutter_binary"]
    p2_ev = agg["filler_evaluation"]["event_level"]
    p2_dom = agg["filler_evaluation"]["dominant_class"]
    p3 = agg["class_metrics"]["Clean"]
    p4 = agg["speech_rate"]
    p5 = agg["block_detection"]["scoped_clinical_uclass_fluencybank"]
    p6 = agg["asr_recognition"]
    p7 = agg["coaching_compliance"]

    seven_params = {
        "parameter_1_stuttering_detection": {
            "metric": "Binary Stutter F1",
            "value": p1["f1"],
            "precision": p1["precision"],
            "recall": p1["recall"],
            "specificity": p1["specificity"],
            "support": p1["support"],
            "tp": p1["tp"], "fp": p1["fp"], "fn": p1["fn"], "tn": p1["tn"]
        },
        "parameter_2_filler_detection_event_level": {
            "metric": "Event-Level Filler F1 (presence in clip)",
            "value": p2_ev["f1"],
            "precision": p2_ev["precision"],
            "recall": p2_ev["recall"],
            "specificity": p2_ev["specificity"],
            "support": p2_ev["support"],
            "tp": p2_ev["tp"], "fp": p2_ev["fp"], "fn": p2_ev["fn"], "tn": p2_ev["tn"]
        },
        "parameter_2_filler_detection_dominant_class": {
            "metric": "Dominant-Class Filler F1 (primary label)",
            "value": p2_dom["f1"],
            "precision": p2_dom["precision"],
            "recall": p2_dom["recall"],
            "specificity": p2_dom["specificity"],
            "support": p2_dom["support"],
            "tp": p2_dom["tp"], "fp": p2_dom["fp"], "fn": p2_dom["fn"], "tn": p2_dom["tn"]
        },
        "parameter_3_clean_speech_discrimination": {
            "metric": "Clean Speech Specificity",
            "value": p3["specificity"],
            "clean_precision": p3["precision"],
            "clean_recall": p3["recall"],
            "clean_f1": p3["f1"],
            "clean_support": p3["support"],
            "tp": p3["tp"], "fp": p3["fp"], "fn": p3["fn"], "tn": p3["tn"]
        },
        "parameter_4_speech_rate_estimation": {
            "metric": "Gold Verbatim Correlation & Sanitized Distribution",
            "valid_mean_wpm": p4["valid_mean_wpm"],
            "valid_median_wpm": p4["valid_median_wpm"],
            "valid_std_wpm": p4["valid_std_wpm"],
            "libri_verbatim_pearson_r": p4["libri_verbatim_pearson_r"]
        },
        "parameter_5_pause_and_block_analysis": {
            "metric": "Scoped Clinical Block Specificity (UCLASS + FluencyBank)",
            "value": p5["specificity"],
            "precision": p5["precision"],
            "recall": p5["recall"],
            "f1": p5["f1"],
            "clips_evaluated": p5["clips_evaluated"],
            "support": p5["support"],
            "tp": p5["tp"], "fp": p5["fp"], "fn": p5["fn"], "tn": p5["tn"]
        },
        "parameter_6_asr_transcription_fidelity": {
            "metric": "Libri_Stutter Verbatim Word Error Rate",
            "micro_wer": p6["micro_wer"],
            "macro_wer": p6["macro_wer"],
            "cer": p6["cer"],
            "clips_evaluated": p6["libri_clips_evaluated"]
        },
        "parameter_7_coaching_rule_compliance": {
            "metric": "Rule Compliance Rate",
            "value": p7["compliance_rate"],
            "compliant_clips": p7["compliant_clips"],
            "total_clips": p7["total_clips"]
        }
    }
    with open(PER_PARAM_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(seven_params, f, indent=2)

    # Per-dataset breakdowns
    per_ds = {}
    for corpus in ["UCLASS", "FluencyBank", "Libri_Stutter"]:
        ds_preds = [p for p in predictions if p["dataset"] == corpus]
        ds_s_tp = sum(1 for p in ds_preds if p["gt_has_stutter"] == 1 and p["pred_has_stutter"] == 1)
        ds_s_fp = sum(1 for p in ds_preds if p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 1)
        ds_s_fn = sum(1 for p in ds_preds if p["gt_has_stutter"] == 1 and p["pred_has_stutter"] == 0)
        ds_s_tn = sum(1 for p in ds_preds if p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 0)
        ds_p = ds_s_tp / (ds_s_tp + ds_s_fp) if (ds_s_tp + ds_s_fp) > 0 else 0.0
        ds_r = ds_s_tp / (ds_s_tp + ds_s_fn) if (ds_s_tp + ds_s_fn) > 0 else 0.0
        ds_f = 2 * ds_p * ds_r / (ds_p + ds_r) if (ds_p + ds_r) > 0 else 0.0
        ds_spec = ds_s_tn / (ds_s_tn + ds_s_fp) if (ds_s_tn + ds_s_fp) > 0 else 0.0
        ds_acc = sum(1 for p in ds_preds if p["gt_dominant"] == p["pred_dominant"]) / len(ds_preds)
        per_ds[corpus] = {
            "clips": len(ds_preds),
            "dominant_accuracy": round(ds_acc, 4),
            "stutter_binary": {
                "tp": ds_s_tp, "fp": ds_s_fp, "fn": ds_s_fn, "tn": ds_s_tn,
                "precision": round(ds_p, 4), "recall": round(ds_r, 4),
                "f1": round(ds_f, 4), "specificity": round(ds_spec, 4)
            }
        }
    with open(PER_DATASET_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(per_ds, f, indent=2)

    # Runtime & Failures
    runtime_info = {
        "status": "COMPLETED",
        "benchmark_clips": 120,
        "successful_inferences": 120,
        "failed_inferences": 0,
        "timeout_count": 0,
        "total_processing_time_sec": round(total_proc_time, 2),
        "mean_latency_ms": round(float(np.mean(latencies)) * 1000.0, 1),
        "max_latency_ms": round(float(np.max(latencies)) * 1000.0, 1),
        "median_latency_ms": round(float(np.median(latencies)) * 1000.0, 1),
        "loop_recovered_count": loop_recovered_count
    }
    with open(RUNTIME_FAILURES_PATH, "w", encoding="utf-8") as f:
        json.dump(runtime_info, f, indent=2)

    # Generate Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_matrix = np.array([[cm[gt][pr] for pr in classes] for gt in classes])
    im = ax.imshow(cm_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm_matrix.shape[1]),
           yticks=np.arange(cm_matrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="Phase 3 Multiclass Confusion Matrix (N=120)",
           ylabel="Ground Truth",
           xlabel="Predicted")
    for i in range(cm_matrix.shape[0]):
        for j in range(cm_matrix.shape[1]):
            ax.text(j, i, format(cm_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_matrix[i, j] > cm_matrix.max() / 2. else "black")
    fig.tight_layout()
    plt.savefig(CM_PLOT_PATH, dpi=150)
    plt.close()

    # Generate Speech Rate Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(valid_wpms, bins=15, color="#1976D2", edgecolor="black", alpha=0.7)
    ax.set_title("Phase 3 Speech Rate Distribution (Valid Non-Loop Clips, N=111)")
    ax.set_xlabel("Words Per Minute (WPM)")
    ax.set_ylabel("Clip Count")
    ax.axvline(np.mean(valid_wpms), color="red", linestyle="dashed", linewidth=1.5, label=f"Mean ({np.mean(valid_wpms):.1f} WPM)")
    ax.axvline(np.median(valid_wpms), color="green", linestyle="dotted", linewidth=1.5, label=f"Median ({np.median(valid_wpms):.1f} WPM)")
    ax.legend()
    fig.tight_layout()
    plt.savefig(SPEECH_RATE_PLOT_PATH, dpi=150)
    plt.close()

    print("\n" + "=" * 80)
    print("PHASE 3 FINAL LOCKED EVALUATION PERFORMANCE")
    print("=" * 80)
    print(f"Dominant Multiclass Accuracy: {agg['dominant_multiclass']['accuracy']*100:.2f}%")
    print(f"Dominant Multiclass Macro-F1: {agg['dominant_multiclass']['macro_f1']:.4f}")
    print(f"Binary Stutter (P1) F1:       {agg['stutter_binary']['f1']:.4f}")
    print(f"  Precision:                  {agg['stutter_binary']['precision']*100:.2f}% (TP={s_tp}, FP={s_fp})")
    print(f"  Recall:                     {agg['stutter_binary']['recall']*100:.2f}% (FN={s_fn})")
    print(f"  Specificity:                {agg['stutter_binary']['specificity']*100:.2f}% (TN={s_tn})")
    print(f"Event-Level Filler (P2) F1:   {agg['filler_evaluation']['event_level']['f1']:.4f}")
    print(f"Clean Specificity (P3):       {agg['class_metrics']['Clean']['specificity']*100:.2f}%")
    print(f"Scoped Block Specificity (P5):{agg['block_detection']['scoped_clinical_uclass_fluencybank']['specificity']*100:.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    run_phase3_locked_evaluation()
