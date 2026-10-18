"""
Development Evaluation Harness Module for SpeechAI Experimental Pipeline
========================================================================
Runs development/pilot evaluation clips through the experimental pipeline
and computes side-by-side comparative metrics against the audited baseline:
- 3-Class Dominant Classification (Stutter, Filler, Clean)
- Binary Stutter Detection (Precision, Recall, F1)
- Binary Filler Detection (Precision, Recall, F1)
- Latency & RTF Distribution (Median, Mean, P95, Max)
- ASR Word Error Rate (WER) where ground truth text is available
- Detailed error, timeout, and failure tracking
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    from jiwer import wer as compute_wer
except ImportError:
    compute_wer = None

from .config import ExperimentalConfig
from .asr_engine import ExperimentalASREngine
from .acoustic_features import AcousticFeatureExtractor
from .stutter_detector import HybridStutterDetector
from .filler_detector import ContextAwareFillerDetector

logger = logging.getLogger("SpeechAI_Dev_Eval_Harness")

def get_dominant_label(has_stutter: int, has_filler: int) -> str:
    """Standardized 3-class dominant label assignment (Stutter > Filler > Clean)."""
    if has_stutter:
        return "Stutter"
    elif has_filler:
        return "Filler"
    else:
        return "Clean"

def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Calculates TP, FP, TN, FN, Precision, Recall, and F1."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "support": tp + fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

def compute_multiclass_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    """Calculates multiclass accuracy, classwise PR-F1, macro F1, and confusion matrix."""
    n = len(y_true)
    if n == 0:
        return {}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    matrix = {row: {col: 0 for col in labels} for row in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1

    class_metrics = {}
    f1_list = []
    p_list = []
    r_list = []

    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        support = sum(matrix[label].values())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        class_metrics[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support
        }
        f1_list.append(f1)
        p_list.append(prec)
        r_list.append(rec)

    macro_p = float(np.mean(p_list))
    macro_r = float(np.mean(r_list))
    macro_f1 = float(np.mean(f1_list))

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "class_metrics": class_metrics,
        "confusion_matrix": matrix
    }

class DevelopmentEvaluationHarness:
    def __init__(self, config: Optional[ExperimentalConfig] = None):
        self.config = config or ExperimentalConfig()
        self.asr_engine = ExperimentalASREngine(self.config.asr)
        self.acoustic_extractor = AcousticFeatureExtractor(self.config.acoustic)
        self.stutter_detector = HybridStutterDetector(self.config.stutter, self.acoustic_extractor)
        self.filler_detector = ContextAwareFillerDetector(self.config.filler, self.acoustic_extractor)

    def run_development_clip(self, clip_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Runs a single clip through the experimental pipeline."""
        file_path = clip_meta["file_path"]
        t_start = time.perf_counter()

        # Step 1: Safe ASR decoding with bounded temperature & timeout
        asr_res = self.asr_engine.transcribe_audio_safe(file_path)
        latency = round(time.perf_counter() - t_start, 3)

        transcript = asr_res.get("text", "")
        words_list = asr_res.get("words", [])
        status = asr_res.get("status", "UNKNOWN")

        # Step 2: Load audio data once for acoustic feature extraction
        audio_data = None
        duration_sec = 0.0
        try:
            audio_data = self.acoustic_extractor.load_audio(file_path)
            y, sr = audio_data
            duration_sec = round(len(y) / sr, 2)
        except Exception as e:
            logger.warning(f"Error loading audio for {file_path}: {e}")

        # Step 3: Hybrid Stutter Detection
        stutter_res = self.stutter_detector.evaluate(
            transcript=transcript,
            words_list=words_list,
            audio_data=audio_data
        )

        # Step 4: Context-Aware Filler Detection
        filler_res = self.filler_detector.evaluate(
            transcript=transcript,
            words_list=words_list,
            audio_data=audio_data
        )

        pred_has_stutter = stutter_res["has_stutter"]
        pred_has_filler = filler_res["has_filler"]
        pred_dominant = get_dominant_label(pred_has_stutter, pred_has_filler)

        # Step 5: Optional WER computation
        wer_val = None
        ref_text = clip_meta.get("ref_text") or clip_meta.get("reference_text")
        if ref_text and compute_wer and transcript and status == "SUCCESS":
            try:
                wer_val = round(float(compute_wer(ref_text, transcript)), 4)
            except Exception:
                pass

        rtf = round(latency / duration_sec, 3) if duration_sec > 0 else 0.0

        return {
            "dataset": clip_meta.get("dataset"),
            "file_path": file_path,
            "filename": clip_meta.get("filename", os.path.basename(file_path)),
            "speaker": clip_meta.get("speaker"),
            # Ground truth
            "gt_has_stutter": 1 if (
                clip_meta.get("gt_sound_rep", 0) or 
                clip_meta.get("gt_word_rep", 0) or 
                clip_meta.get("gt_prolongation", 0) or 
                clip_meta.get("gt_block", 0)
            ) else 0,
            "gt_has_filler": 1 if clip_meta.get("gt_filler", 0) else 0,
            "gt_is_clean": 1 if clip_meta.get("gt_clean", 0) else 0,
            "gt_dominant": get_dominant_label(
                1 if (clip_meta.get("gt_sound_rep", 0) or clip_meta.get("gt_word_rep", 0) or clip_meta.get("gt_prolongation", 0) or clip_meta.get("gt_block", 0)) else 0,
                1 if clip_meta.get("gt_filler", 0) else 0
            ),
            # Experimental predictions
            "pred_transcript": transcript,
            "pred_has_stutter": pred_has_stutter,
            "pred_has_filler": pred_has_filler,
            "pred_is_clean": 1 if (pred_has_stutter == 0 and pred_has_filler == 0) else 0,
            "pred_dominant": pred_dominant,
            "stutter_events": stutter_res["total_stutter_events"],
            "stutter_confidence": stutter_res["confidence_score"],
            "stutter_classification": stutter_res["classification"],
            "filler_events": filler_res["total_filler_events"],
            "filler_confidence": filler_res["confidence_score"],
            "filler_classification": filler_res["classification"],
            "status": status,
            "asr_status": status,
            "asr_error": asr_res.get("error"),
            "latency_sec": latency,
            "duration_sec": duration_sec,
            "rtf": rtf,
            "wer": wer_val,
            "details": {
                "stutter": stutter_res["details"],
                "filler": filler_res["details"]
            }
        }

    def run_benchmark_on_manifest(
        self,
        manifest: List[Dict[str, Any]],
        baseline_predictions_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs evaluation over a development manifest and produces side-by-side comparison.
        """
        logger.info(f"Starting development evaluation on {len(manifest)} clips...")
        exp_predictions = []
        for i, clip in enumerate(manifest):
            logger.info(f"[{i+1}/{len(manifest)}] Processing: {clip.get('filename')}")
            res = self.run_development_clip(clip)
            exp_predictions.append(res)

        # Load baseline predictions for side-by-side comparison if available
        base_dict = {}
        if baseline_predictions_path and os.path.exists(baseline_predictions_path):
            with open(baseline_predictions_path, "r", encoding="utf-8") as f:
                raw_base = json.load(f)
                for b in raw_base:
                    base_dict[b["filename"]] = b

        # Align predictions for comparative metrics
        gt_dominant = [p["gt_dominant"] for p in exp_predictions]
        exp_dominant = [p["pred_dominant"] for p in exp_predictions]

        gt_stutter = [p["gt_has_stutter"] for p in exp_predictions]
        exp_stutter = [p["pred_has_stutter"] for p in exp_predictions]

        gt_filler = [p["gt_has_filler"] for p in exp_predictions]
        exp_filler = [p["pred_has_filler"] for p in exp_predictions]

        labels = ["Stutter", "Filler", "Clean"]
        exp_3class = compute_multiclass_metrics(gt_dominant, exp_dominant, labels)
        exp_stutter_binary = compute_binary_metrics(gt_stutter, exp_stutter)
        exp_filler_binary = compute_binary_metrics(gt_filler, exp_filler)

        # Experimental latency stats
        latencies = [p["latency_sec"] for p in exp_predictions]
        rtfs = [p["rtf"] for p in exp_predictions if p["duration_sec"] > 0]
        total_audio = sum(p["duration_sec"] for p in exp_predictions)
        total_time = sum(latencies)

        exp_latency_stats = {
            "median_sec": round(float(np.median(latencies)), 3),
            "mean_sec": round(float(np.mean(latencies)), 3),
            "min_sec": round(float(np.min(latencies)), 3),
            "max_sec": round(float(np.max(latencies)), 3),
            "p95_sec": round(float(np.percentile(latencies, 95)), 3),
            "aggregate_rtf": round(total_time / total_audio, 3) if total_audio > 0 else 0.0
        }

        # Baseline metrics over the exact same clips
        base_3class, base_stutter_binary, base_filler_binary, base_latency_stats = None, None, None, None
        if base_dict:
            matched_base = [base_dict[p["filename"]] for p in exp_predictions if p["filename"] in base_dict]
            if len(matched_base) == len(exp_predictions):
                base_dom = [get_dominant_label(b.get("pred_has_stutter", 0), b.get("pred_has_filler", 0)) for b in matched_base]
                base_stut = [b.get("pred_has_stutter", 0) for b in matched_base]
                base_fill = [b.get("pred_has_filler", 0) for b in matched_base]
                
                base_3class = compute_multiclass_metrics(gt_dominant, base_dom, labels)
                base_stutter_binary = compute_binary_metrics(gt_stutter, base_stut)
                base_filler_binary = compute_binary_metrics(gt_filler, base_fill)

                b_lats = [b.get("latency_sec", 0.0) for b in matched_base]
                b_aud = sum(b.get("duration_sec", 0.0) for b in matched_base)
                b_time = sum(b_lats)
                base_latency_stats = {
                    "median_sec": round(float(np.median(b_lats)), 3),
                    "mean_sec": round(float(np.mean(b_lats)), 3),
                    "min_sec": round(float(np.min(b_lats)), 3),
                    "max_sec": round(float(np.max(b_lats)), 3),
                    "p95_sec": round(float(np.percentile(b_lats, 95)), 3),
                    "aggregate_rtf": round(b_time / b_aud, 3) if b_aud > 0 else 0.0
                }

        summary = {
            "n_clips": len(exp_predictions),
            "baseline": {
                "multiclass_3class": base_3class,
                "stutter_binary": base_stutter_binary,
                "filler_binary": base_filler_binary,
                "latency": base_latency_stats
            },
            "experimental": {
                "multiclass_3class": exp_3class,
                "stutter_binary": exp_stutter_binary,
                "filler_binary": exp_filler_binary,
                "latency": exp_latency_stats
            },
            "predictions": exp_predictions
        }

        return summary
