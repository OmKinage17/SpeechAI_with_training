"""
Phase 4C Holdout Evaluation Runner (PREPARED ONLY — DO NOT EXECUTE WITHOUT USER APPROVAL)
========================================================================================
Strict Scope & Safeguards:
- Evaluates the frozen Phase 2 baseline and Phase 4B candidate on the locked 120-clip holdout set.
- Reads manifest and feature cache from evaluation_results/experimental/final_benchmark/run_final_20261020_complete/.
- Writes all candidate outputs exclusively to evaluation_results/experimental/phase4b_holdout/.
- Preserves raw and sanitized transcripts, per-clip predictions, configuration, and metrics.
- ZERO modifications to production modules, main frontend, or locked reference files.
"""

import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, Any, List

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.experimental.config import StutterDetectorConfig
from backend.experimental.stutter_detector import HybridStutterDetector


def calculate_binary_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Any]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "specificity": round(spec, 4),
        "accuracy": round(acc, 4)
    }


def calculate_multiclass_metrics(cm: Dict[str, Dict[str, int]], labels: List[str]) -> Dict[str, Any]:
    total = sum(sum(cm[row].values()) for row in labels)
    correct = sum(cm[label][label] for label in labels)
    accuracy = correct / total if total > 0 else 0.0

    per_class = {}
    f1s = []
    for label in labels:
        tp = cm[label][label]
        fp = sum(cm[other][label] for other in labels if other != label)
        fn = sum(cm[label][other] for other in labels if other != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        per_class[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class
    }


def run_holdout_evaluation():
    start_time = time.time()
    out_dir = project_root / "evaluation_results" / "experimental" / "phase4b_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Starting Phase 4B Final Holdout Evaluation (Locked 120 Clips) ===")

    # 1. Candidate Configuration
    baseline_cfg = StutterDetectorConfig(
        confident_threshold=0.60,
        uncertain_threshold=0.35,
        min_prolongation_duration_sec=0.50,
        severe_prolongation_duration_sec=0.60,
        standalone_prolongation_confidence=0.50,
        standalone_prolongation_min_duration_sec=0.0,
        enable_contextual_scoring=True,
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
        interrupted_repetition_confidence=0.85,
        enable_multi_iteration_filler_stutter=False
    )
    candidate_cfg = StutterDetectorConfig(
        confident_threshold=0.60,
        uncertain_threshold=0.35,
        min_prolongation_duration_sec=0.50,
        severe_prolongation_duration_sec=0.60,
        standalone_prolongation_confidence=0.50,
        standalone_prolongation_min_duration_sec=0.0,
        enable_contextual_scoring=True,
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
        interrupted_repetition_confidence=0.85,
        enable_multi_iteration_filler_stutter=True,
        min_filler_repetition_count=3,
        rapid_filler_isi_threshold_sec=0.35
    )

    with open(out_dir / "candidate_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "candidate_name": "Phase 4B: Multi-Iteration Hesitation Gating",
            "intervention": "Intervention 1 (Repeated-Filler Gating with Pacing & Calibrated Confidence)",
            "scope": "Locked Final Evaluation Set (N=120 clips)",
            "baseline_config": {
                "confident_threshold": baseline_cfg.confident_threshold,
                "uncertain_threshold": baseline_cfg.uncertain_threshold,
                "min_prolongation_duration_sec": baseline_cfg.min_prolongation_duration_sec,
                "severe_prolongation_duration_sec": baseline_cfg.severe_prolongation_duration_sec,
                "standalone_prolongation_confidence": baseline_cfg.standalone_prolongation_confidence,
                "standalone_prolongation_min_duration_sec": baseline_cfg.standalone_prolongation_min_duration_sec,
                "enable_contextual_scoring": baseline_cfg.enable_contextual_scoring,
                "enable_ngram_repetition": baseline_cfg.enable_ngram_repetition,
                "enable_filler_boundary_gating": baseline_cfg.enable_filler_boundary_gating,
                "enable_interrupted_repetition": baseline_cfg.enable_interrupted_repetition,
                "enable_multi_iteration_filler_stutter": baseline_cfg.enable_multi_iteration_filler_stutter
            },
            "candidate_config": {
                "confident_threshold": candidate_cfg.confident_threshold,
                "uncertain_threshold": candidate_cfg.uncertain_threshold,
                "min_prolongation_duration_sec": candidate_cfg.min_prolongation_duration_sec,
                "severe_prolongation_duration_sec": candidate_cfg.severe_prolongation_duration_sec,
                "standalone_prolongation_confidence": candidate_cfg.standalone_prolongation_confidence,
                "standalone_prolongation_min_duration_sec": candidate_cfg.standalone_prolongation_min_duration_sec,
                "enable_contextual_scoring": candidate_cfg.enable_contextual_scoring,
                "enable_ngram_repetition": candidate_cfg.enable_ngram_repetition,
                "enable_filler_boundary_gating": candidate_cfg.enable_filler_boundary_gating,
                "enable_interrupted_repetition": candidate_cfg.enable_interrupted_repetition,
                "enable_multi_iteration_filler_stutter": candidate_cfg.enable_multi_iteration_filler_stutter,
                "min_filler_repetition_count": candidate_cfg.min_filler_repetition_count,
                "rapid_filler_isi_threshold_sec": candidate_cfg.rapid_filler_isi_threshold_sec
            }
        }, f, indent=2)

    # 2. Load Locked 120-clip final benchmark
    holdout_manifest_path = project_root / "evaluation_results" / "experimental" / "final_benchmark" / "run_final_20261020_complete" / "run_manifest.json"
    holdout_cache_path = project_root / "evaluation_results" / "experimental" / "final_benchmark" / "run_final_20261020_complete" / "feature_cache_final.json"
    holdout_preds_path = project_root / "evaluation_results" / "experimental" / "final_benchmark" / "run_final_20261020_complete" / "per_clip_predictions.json"

    with open(holdout_manifest_path, "r", encoding="utf-8") as f:
        manifest_entries = json.load(f)
    with open(holdout_cache_path, "r", encoding="utf-8") as f:
        feature_cache = json.load(f)
    with open(holdout_preds_path, "r", encoding="utf-8") as f:
        ref_preds = {p["filename"]: p for p in json.load(f)}

    print(f"Loaded {len(manifest_entries)} clips from locked final benchmark manifest.")

    baseline_detector = HybridStutterDetector(baseline_cfg)
    candidate_detector = HybridStutterDetector(candidate_cfg)

    per_clip_predictions = []
    changed_clips = []
    failures = []

    eval_start = time.time()
    for m in manifest_entries:
        fn = m["filename"]
        fc = feature_cache.get(fn, {})
        rp = ref_preds.get(fn, {})
        tx = fc.get("transcript", m.get("transcript", ""))
        wlist = fc.get("words_list", [])
        fpath = m.get("file_path", fc.get("file_path"))
        pred_has_filler = rp.get("pred_has_filler", 0)

        try:
            b_res = baseline_detector.evaluate(
                transcript=tx,
                words_list=wlist,
                audio_path=fpath,
                pred_has_filler=pred_has_filler
            )
            c_res = candidate_detector.evaluate(
                transcript=tx,
                words_list=wlist,
                audio_path=fpath,
                pred_has_filler=pred_has_filler
            )

            is_changed = (
                b_res["has_stutter"] != c_res["has_stutter"] or
                b_res["pred_dominant"] != c_res["pred_dominant"]
            )

            change_reason = ""
            if is_changed:
                if b_res["has_stutter"] == 0 and c_res["has_stutter"] == 1:
                    struggle_events = [d for d in c_res["details"] if d.get("type") == "multi_iteration_filler_struggle"]
                    if struggle_events:
                        ev = struggle_events[0]
                        change_reason = f"Recovered FN via multi-iteration struggle: {ev.get('evidence', '')} (conf: {ev.get('confidence')})"
                    else:
                        change_reason = "Recovered FN via corroborated hesitation"
                elif b_res["has_stutter"] == 1 and c_res["has_stutter"] == 0:
                    change_reason = "Candidate suppressed baseline detection"
                changed_clips.append({
                    "filename": fn,
                    "dataset": m["dataset"],
                    "gt_has_stutter": m["gt_has_stutter"],
                    "gt_dominant": m["gt_dominant"],
                    "baseline_stutter": b_res["has_stutter"],
                    "candidate_stutter": c_res["has_stutter"],
                    "baseline_dominant": b_res["pred_dominant"],
                    "candidate_dominant": c_res["pred_dominant"],
                    "reason": change_reason,
                    "candidate_details": c_res["details"]
                })

            per_clip_predictions.append({
                "filename": fn,
                "dataset": m["dataset"],
                "file_path": fpath,
                "audio_duration_sec": m.get("audio_duration_sec", 0.0),
                "transcript": tx,
                "gt_has_stutter": m["gt_has_stutter"],
                "gt_has_filler": m["gt_has_filler"],
                "gt_dominant": m["gt_dominant"],
                "pred_has_filler": pred_has_filler,
                "baseline": {
                    "has_stutter": b_res["has_stutter"],
                    "pred_dominant": b_res["pred_dominant"],
                    "confidence": b_res["confidence_score"],
                    "classification": b_res["classification"],
                    "total_events": b_res["total_stutter_events"],
                    "details": b_res["details"]
                },
                "candidate": {
                    "has_stutter": c_res["has_stutter"],
                    "pred_dominant": c_res["pred_dominant"],
                    "confidence": c_res["confidence_score"],
                    "classification": c_res["classification"],
                    "total_events": c_res["total_stutter_events"],
                    "details": c_res["details"],
                    "gated_events": c_res.get("gated_events", [])
                },
                "prediction_changed": is_changed,
                "change_reason": change_reason
            })
        except Exception as e:
            failures.append({"filename": fn, "error": str(e)})

    eval_duration = time.time() - eval_start

    # Metrics
    base_tp = sum(1 for p in per_clip_predictions if p["gt_has_stutter"] == 1 and p["baseline"]["has_stutter"] == 1)
    base_fp = sum(1 for p in per_clip_predictions if p["gt_has_stutter"] == 0 and p["baseline"]["has_stutter"] == 1)
    base_fn = sum(1 for p in per_clip_predictions if p["gt_has_stutter"] == 1 and p["baseline"]["has_stutter"] == 0)
    base_tn = sum(1 for p in per_clip_predictions if p["gt_has_stutter"] == 0 and p["baseline"]["has_stutter"] == 0)
    base_stutter_metrics = calculate_binary_metrics(base_tp, base_fp, base_fn, base_tn)

    cand_tp = sum(1 for p in per_clip_predictions if p["gt_has_stutter"] == 1 and p["candidate"]["has_stutter"] == 1)
    cand_fp = sum(1 for p in per_clip_predictions if p["gt_has_stutter"] == 0 and p["candidate"]["has_stutter"] == 1)
    cand_fn = sum(1 for p in per_clip_predictions if p["gt_has_stutter"] == 1 and p["candidate"]["has_stutter"] == 0)
    cand_tn = sum(1 for p in per_clip_predictions if p["gt_has_stutter"] == 0 and p["candidate"]["has_stutter"] == 0)
    cand_stutter_metrics = calculate_binary_metrics(cand_tp, cand_fp, cand_fn, cand_tn)

    labels = ["Stutter", "Filler", "Clean"]
    base_cm = {row: {col: 0 for col in labels} for row in labels}
    cand_cm = {row: {col: 0 for col in labels} for row in labels}

    for p in per_clip_predictions:
        gt = p["gt_dominant"]
        b_pred = p["baseline"]["pred_dominant"]
        c_pred = p["candidate"]["pred_dominant"]
        if gt in labels and b_pred in labels:
            base_cm[gt][b_pred] += 1
        if gt in labels and c_pred in labels:
            cand_cm[gt][c_pred] += 1

    base_mc_metrics = calculate_multiclass_metrics(base_cm, labels)
    cand_mc_metrics = calculate_multiclass_metrics(cand_cm, labels)

    with open(out_dir / "per_clip_predictions.json", "w", encoding="utf-8") as f:
        json.dump(per_clip_predictions, f, indent=2)

    with open(out_dir / "per_clip_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "dataset", "gt_has_stutter", "gt_has_filler", "gt_dominant",
            "base_has_stutter", "base_pred_dominant", "base_conf",
            "cand_has_stutter", "cand_pred_dominant", "cand_conf",
            "changed", "change_reason"
        ])
        for p in per_clip_predictions:
            writer.writerow([
                p["filename"], p["dataset"], p["gt_has_stutter"], p["gt_has_filler"], p["gt_dominant"],
                p["baseline"]["has_stutter"], p["baseline"]["pred_dominant"], p["baseline"]["confidence"],
                p["candidate"]["has_stutter"], p["candidate"]["pred_dominant"], p["candidate"]["confidence"],
                p["prediction_changed"], p["change_reason"]
            ])

    # Standalone confusion matrices artifact
    confusion_matrices_dict = {
        "stutter_binary": {
            "baseline": {"tp": base_tp, "fp": base_fp, "fn": base_fn, "tn": base_tn},
            "candidate": {"tp": cand_tp, "fp": cand_fp, "fn": cand_fn, "tn": cand_tn}
        },
        "dominant_multiclass": {
            "baseline": base_cm,
            "candidate": cand_cm
        }
    }
    with open(out_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(confusion_matrices_dict, f, indent=2)

    # Filler binary metrics
    filler_tp = sum(1 for p in per_clip_predictions if p["gt_has_filler"] == 1 and p["pred_has_filler"] == 1)
    filler_fp = sum(1 for p in per_clip_predictions if p["gt_has_filler"] == 0 and p["pred_has_filler"] == 1)
    filler_fn = sum(1 for p in per_clip_predictions if p["gt_has_filler"] == 1 and p["pred_has_filler"] == 0)
    filler_tn = sum(1 for p in per_clip_predictions if p["gt_has_filler"] == 0 and p["pred_has_filler"] == 0)
    filler_metrics = calculate_binary_metrics(filler_tp, filler_fp, filler_fn, filler_tn)

    # Per-dataset breakdown
    datasets = sorted(list(set(p["dataset"] for p in per_clip_predictions)))
    per_dataset = {}
    for ds in datasets:
        ds_preds = [p for p in per_clip_predictions if p["dataset"] == ds]
        b_tp = sum(1 for p in ds_preds if p["gt_has_stutter"] == 1 and p["baseline"]["has_stutter"] == 1)
        b_fp = sum(1 for p in ds_preds if p["gt_has_stutter"] == 0 and p["baseline"]["has_stutter"] == 1)
        b_fn = sum(1 for p in ds_preds if p["gt_has_stutter"] == 1 and p["baseline"]["has_stutter"] == 0)
        b_tn = sum(1 for p in ds_preds if p["gt_has_stutter"] == 0 and p["baseline"]["has_stutter"] == 0)
        b_m = calculate_binary_metrics(b_tp, b_fp, b_fn, b_tn)

        c_tp = sum(1 for p in ds_preds if p["gt_has_stutter"] == 1 and p["candidate"]["has_stutter"] == 1)
        c_fp = sum(1 for p in ds_preds if p["gt_has_stutter"] == 0 and p["candidate"]["has_stutter"] == 1)
        c_fn = sum(1 for p in ds_preds if p["gt_has_stutter"] == 1 and p["candidate"]["has_stutter"] == 0)
        c_tn = sum(1 for p in ds_preds if p["gt_has_stutter"] == 0 and p["candidate"]["has_stutter"] == 0)
        c_m = calculate_binary_metrics(c_tp, c_fp, c_fn, c_tn)

        per_dataset[ds] = {
            "total_clips": len(ds_preds),
            "baseline": b_m,
            "candidate": c_m,
            "delta": {
                "tp": c_m["tp"] - b_m["tp"],
                "fp": c_m["fp"] - b_m["fp"],
                "fn": c_m["fn"] - b_m["fn"],
                "tn": c_m["tn"] - b_m["tn"],
                "precision": round(c_m["precision"] - b_m["precision"], 4),
                "recall": round(c_m["recall"] - b_m["recall"], 4),
                "f1": round(c_m["f1"] - b_m["f1"], 4)
            }
        }

    with open(out_dir / "per_dataset_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_dataset, f, indent=2)

    # McNemar's test for paired binary classification errors
    # b_disc: baseline correct, candidate incorrect
    # c_disc: baseline incorrect, candidate correct
    b_disc = sum(1 for p in per_clip_predictions if (p["baseline"]["has_stutter"] == p["gt_has_stutter"]) and (p["candidate"]["has_stutter"] != p["gt_has_stutter"]))
    c_disc = sum(1 for p in per_clip_predictions if (p["baseline"]["has_stutter"] != p["gt_has_stutter"]) and (p["candidate"]["has_stutter"] == p["gt_has_stutter"]))
    both_correct = sum(1 for p in per_clip_predictions if (p["baseline"]["has_stutter"] == p["gt_has_stutter"]) and (p["candidate"]["has_stutter"] == p["gt_has_stutter"]))
    both_incorrect = sum(1 for p in per_clip_predictions if (p["baseline"]["has_stutter"] != p["gt_has_stutter"]) and (p["candidate"]["has_stutter"] != p["gt_has_stutter"]))

    # Edwards continuity-corrected McNemar chi-squared
    total_disc = b_disc + c_disc
    if total_disc > 0:
        mcnemar_stat = round(float((max(0, abs(b_disc - c_disc) - 1)) ** 2 / total_disc), 4)
        from math import erf, sqrt
        # 1-df chi-square p-value
        p_val = round(1.0 - erf(sqrt(mcnemar_stat / 2.0)), 4) if mcnemar_stat > 0 else 1.0
    else:
        mcnemar_stat = 0.0
        p_val = 1.0

    mcnemar_results = {
        "contingency_table": {
            "both_correct": both_correct,
            "both_incorrect": both_incorrect,
            "baseline_correct_candidate_incorrect": b_disc,
            "baseline_incorrect_candidate_correct": c_disc,
            "total_discordant": total_disc
        },
        "chi2_stat_continuity_corrected": mcnemar_stat,
        "p_value": p_val,
        "is_statistically_significant_p05": p_val < 0.05
    }

    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "evaluation_name": "Phase 4C Locked Holdout Evaluation (Intervention 1)",
            "total_clips": len(manifest_entries),
            "stutter_metrics": {
                "baseline": base_stutter_metrics,
                "candidate": cand_stutter_metrics,
                "delta": {
                    "tp": cand_stutter_metrics["tp"] - base_stutter_metrics["tp"],
                    "fp": cand_stutter_metrics["fp"] - base_stutter_metrics["fp"],
                    "fn": cand_stutter_metrics["fn"] - base_stutter_metrics["fn"],
                    "tn": cand_stutter_metrics["tn"] - base_stutter_metrics["tn"],
                    "precision": round(cand_stutter_metrics["precision"] - base_stutter_metrics["precision"], 4),
                    "recall": round(cand_stutter_metrics["recall"] - base_stutter_metrics["recall"], 4),
                    "f1": round(cand_stutter_metrics["f1"] - base_stutter_metrics["f1"], 4),
                    "specificity": round(cand_stutter_metrics["specificity"] - base_stutter_metrics["specificity"], 4)
                }
            },
            "multiclass_dominant_metrics": {
                "baseline": base_mc_metrics,
                "candidate": cand_mc_metrics,
                "delta": {
                    "accuracy": round(cand_mc_metrics["accuracy"] - base_mc_metrics["accuracy"], 4),
                    "macro_f1": round(cand_mc_metrics["macro_f1"] - base_mc_metrics["macro_f1"], 4)
                }
            },
            "filler_metrics": {
                "baseline": filler_metrics,
                "candidate": filler_metrics,
                "lost_filler_detections": 0,
                "new_filler_false_positives": 0
            },
            "mcnemar_test": mcnemar_results,
            "changed_clips": changed_clips
        }, f, indent=2)

    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "run_id": "phase4c_holdout_120clips",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_split": "Locked Holdout (N=120)",
            "clips_count": len(manifest_entries),
            "eval_duration_sec": round(eval_duration, 2),
            "failures_count": len(failures)
        }, f, indent=2)

    with open(out_dir / "runtime_and_failures.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_clips": len(manifest_entries),
            "failures_count": len(failures),
            "failures": failures,
            "total_duration_sec": round(time.time() - start_time, 2),
            "evaluation_duration_sec": round(eval_duration, 2)
        }, f, indent=2)

    print("=== Holdout Evaluation Complete ===")
    print(f"Stutter Binary: TP {base_tp}->{cand_tp} (+{cand_tp-base_tp}), FP {base_fp}->{cand_fp} ({cand_fp-base_fp:+d}), FN {base_fn}->{cand_fn} ({cand_fn-base_fn:+d}), TN {base_tn}->{cand_tn}")
    print(f"Precision: {base_stutter_metrics['precision']*100:.2f}% -> {cand_stutter_metrics['precision']*100:.2f}%")
    print(f"Recall:    {base_stutter_metrics['recall']*100:.2f}% -> {cand_stutter_metrics['recall']*100:.2f}% ({cand_stutter_metrics['recall']*100 - base_stutter_metrics['recall']*100:+.2f}%)")
    print(f"F1 Score:  {base_stutter_metrics['f1']*100:.2f}% -> {cand_stutter_metrics['f1']*100:.2f}% ({cand_stutter_metrics['f1']*100 - base_stutter_metrics['f1']*100:+.2f}%)")
    print(f"Specificity: {base_stutter_metrics['specificity']*100:.2f}% -> {cand_stutter_metrics['specificity']*100:.2f}%")
    print(f"Discordant clips: {total_disc} (b={b_disc}, c={c_disc}), McNemar p-value: {p_val}")


if __name__ == "__main__":
    print("Holdout runner is prepared. Awaiting user authorization before execution.")
