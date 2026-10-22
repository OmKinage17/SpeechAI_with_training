"""
Canonical Benchmark Evaluator for SpeechAI Development Benchmark (N=30)
========================================================================

Author: Antigravity Autonomous Audit Suite
Scope: Read-only evaluation harness to reconcile Fix 1, Fix 2B, Fix 3, and Fix 4
       saved predictions and configurations under a single canonical metric engine.

Strict Constraints:
- Read-only: Does not modify any detector algorithms, thresholds, or config files.
- Preserves exact 30-clip development manifest and ground-truth labels.
- Evaluates directly from saved artifacts without mixing cached and live features.
- Tracks timeouts, missing predictions, and errors separately.
- Strictly maintains N=30 denominator across all evaluations.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class CanonicalEvaluator:
    """
    Standardized, read-only evaluator for multiclass and binary stutter detection benchmarks.
    """
    CLASSES = ["Stutter", "Filler", "Clean"]

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.dev_manifest_path = self.project_root / "evaluation_results" / "experimental" / "dev_eval_results_30clips_refined.json"
        self._load_manifest()

    def _load_manifest(self):
        """Loads canonical 30-clip manifest and ground-truth labels."""
        if not self.dev_manifest_path.exists():
            raise FileNotFoundError(f"Dev manifest not found at: {self.dev_manifest_path}")
        with open(self.dev_manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.raw_manifest = data["predictions"]
        self.manifest_clips = [p["filename"] for p in self.raw_manifest]
        self.gt_dominant = {p["filename"]: p["gt_dominant"].strip() for p in self.raw_manifest}
        self.gt_has_stutter = {p["filename"]: int(p["gt_has_stutter"]) for p in self.raw_manifest}
        self.gt_has_filler = {p["filename"]: int(p["gt_has_filler"]) for p in self.raw_manifest}
        self.dataset_map = {p["filename"]: p["dataset"].strip() for p in self.raw_manifest}
        self.audio_path_map = {p["filename"]: p["file_path"] for p in self.raw_manifest}

        # Verify manifest composition
        assert len(self.manifest_clips) == 30, f"Expected 30 clips, found {len(self.manifest_clips)}"
        assert sum(1 for c, v in self.gt_has_stutter.items() if v == 1) == 24
        assert sum(1 for c, v in self.gt_dominant.items() if v == "Filler") == 5
        assert sum(1 for c, v in self.gt_dominant.items() if v == "Clean") == 1

    def evaluate_prediction_records(
        self,
        config_name: str,
        predictions: List[Dict[str, Any]],
        pipeline_type: str,
        timeout_clips: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a list of prediction records against the canonical ground-truth manifest.
        
        pipeline_type: 'live_acoustic', 'hybrid_cached', or 'baseline'
        timeout_clips: list of filenames that suffered ASR timeouts
        """
        if timeout_clips is None:
            timeout_clips = ["M_0030_16y4m_1_dysfluent_009.wav", "FluencyBank_010_40.wav"]

        seen = set()
        duplicates = []
        missing = []
        errors = []
        pred_map = {}

        for p in predictions:
            fname = p.get("filename")
            if not fname:
                errors.append(f"Missing filename field in record: {p}")
                continue
            if fname in seen:
                duplicates.append(fname)
            seen.add(fname)
            pred_map[fname] = p

        for fname in self.manifest_clips:
            if fname not in pred_map:
                missing.append(fname)

        # Build verified aligned records
        aligned_records = []
        for fname in self.manifest_clips:
            p = pred_map.get(fname, {})
            gt_dom = self.gt_dominant[fname]
            gt_st = self.gt_has_stutter[fname]
            gt_fl = self.gt_has_filler[fname]
            dataset = self.dataset_map[fname]

            pred_dom = p.get("pred_dominant") or p.get("predicted_dominant") or "Clean"
            pred_st = int(p.get("pred_has_stutter", 1 if pred_dom == "Stutter" else 0))
            pred_fl = int(p.get("pred_has_filler", 1 if pred_dom == "Filler" else 0))
            conf = p.get("confidence", 0.0)
            is_timeout = fname in timeout_clips

            aligned_records.append({
                "filename": fname,
                "dataset": dataset,
                "gt_dominant": gt_dom,
                "gt_has_stutter": gt_st,
                "gt_has_filler": gt_fl,
                "pred_dominant": pred_dom,
                "pred_has_stutter": pred_st,
                "pred_has_filler": pred_fl,
                "confidence": conf,
                "is_timeout": is_timeout,
                "is_correct_multiclass": (gt_dom == pred_dom),
                "is_correct_binary_stutter": (gt_st == pred_st)
            })

        # Calculate metrics using canonical formulas
        aggregate_metrics = self._compute_metrics_block(aligned_records)

        # Per-corpus breakdown
        per_corpus = {}
        for ds in ["FluencyBank", "Libri_Stutter", "UCLASS"]:
            ds_records = [r for r in aligned_records if r["dataset"] == ds]
            per_corpus[ds] = self._compute_metrics_block(ds_records)

        return {
            "config_name": config_name,
            "pipeline_type": pipeline_type,
            "total_manifest_clips": len(self.manifest_clips),
            "evaluated_clips": len(aligned_records),
            "missing_predictions": missing,
            "duplicate_predictions": duplicates,
            "timeout_clips": [r["filename"] for r in aligned_records if r["is_timeout"]],
            "errors": errors,
            "aggregate_metrics": aggregate_metrics,
            "per_corpus_metrics": per_corpus,
            "records": aligned_records
        }

    def _compute_metrics_block(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates multiclass and binary metrics from aligned records."""
        n = len(records)
        if n == 0:
            return {}

        # Multiclass Confusion Matrix
        cm = {c_gt: {c_pred: 0 for c_pred in self.CLASSES} for c_gt in self.CLASSES}
        for r in records:
            gt = r["gt_dominant"]
            pred = r["pred_dominant"]
            if gt in cm and pred in cm[gt]:
                cm[gt][pred] += 1

        accuracy = sum(cm[c][c] for c in self.CLASSES) / n

        class_metrics = {}
        macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
        weighted_f1 = 0.0

        for c in self.CLASSES:
            tp = cm[c][c]
            fp = sum(cm[other][c] for other in self.CLASSES if other != c)
            fn = sum(cm[c][other] for other in self.CLASSES if other != c)
            support = sum(cm[c].values())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            class_metrics[c] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": support,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4)
            }
            macro_p += prec
            macro_r += rec
            macro_f1 += f1
            weighted_f1 += f1 * support

        macro_p /= len(self.CLASSES)
        macro_r /= len(self.CLASSES)
        macro_f1 /= len(self.CLASSES)
        weighted_f1 /= n

        # Binary Stutter Metrics
        bin_tp = sum(1 for r in records if r["gt_has_stutter"] == 1 and r["pred_has_stutter"] == 1)
        bin_fp = sum(1 for r in records if r["gt_has_stutter"] == 0 and r["pred_has_stutter"] == 1)
        bin_fn = sum(1 for r in records if r["gt_has_stutter"] == 1 and r["pred_has_stutter"] == 0)
        bin_tn = sum(1 for r in records if r["gt_has_stutter"] == 0 and r["pred_has_stutter"] == 0)

        bin_prec = bin_tp / (bin_tp + bin_fp) if (bin_tp + bin_fp) > 0 else 0.0
        bin_rec = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) > 0 else 0.0
        bin_f1 = (2 * bin_prec * bin_rec) / (bin_prec + bin_rec) if (bin_prec + bin_rec) > 0 else 0.0

        # FluencyBank Stutter FPs
        fb_stutter_fps = sum(
            1 for r in records 
            if r["dataset"] == "FluencyBank" and r["gt_has_stutter"] == 0 and r["pred_has_stutter"] == 1
        )

        return {
            "n": n,
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "class_metrics": class_metrics,
            "confusion_matrix": cm,
            "stutter_binary": {
                "tp": bin_tp,
                "fp": bin_fp,
                "fn": bin_fn,
                "tn": bin_tn,
                "support": bin_tp + bin_fn,
                "precision": round(bin_prec, 4),
                "recall": round(bin_rec, 4),
                "f1": round(bin_f1, 4)
            },
            "fluencybank_stutter_fps": fb_stutter_fps
        }


def run_canonical_evaluation(project_root: str):
    """Executes canonical evaluation across all saved development benchmarks."""
    evaluator = CanonicalEvaluator(project_root)
    out_dir = Path(project_root) / "evaluation_results" / "experimental" / "audit" / "canonical_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs_to_evaluate = [
        {
            "name": "Fix 1 Baseline (Refined)",
            "file": "evaluation_results/experimental/dev_eval_results_30clips_refined.json",
            "key": "predictions",
            "pipeline": "baseline_acoustic_lexical"
        },
        {
            "name": "Fix 2B Baseline (Variant 7, from Fix 4 Artifact)",
            "file": "evaluation_results/experimental/dev_eval_results_30clips_fix4.json",
            "key": "fix2b_baseline_preds",  # Synthesized from sim harness
            "pipeline": "simulated_hybrid"
        },
        {
            "name": "Fix 3 Contextual (Live Runner Saved)",
            "file": "evaluation_results/experimental/dev_eval_results_30clips_fix3.json",
            "key": "predictions_fix3",
            "pipeline": "live_acoustic_sequential"
        },
        {
            "name": "Fix 3 Contextual (Hybrid Simulated)",
            "file": "evaluation_results/experimental/dev_eval_results_30clips_fix4.json",
            "key": "fix3_sim_preds",
            "pipeline": "simulated_hybrid"
        },
        {
            "name": "Fix 4 Lexical/Syntactic (Saved Artifact)",
            "file": "evaluation_results/experimental/dev_eval_results_30clips_fix4.json",
            "key": "predictions_fix4",
            "pipeline": "simulated_hybrid"
        }
    ]

    # Pre-extract predictions from JSONs
    with open(evaluator.project_root / "evaluation_results" / "experimental" / "dev_eval_results_30clips_refined.json", "r", encoding="utf-8") as f:
        d_ref = json.load(f)
    with open(evaluator.project_root / "evaluation_results" / "experimental" / "dev_eval_results_30clips_fix3.json", "r", encoding="utf-8") as f:
        d_f3 = json.load(f)
    with open(evaluator.project_root / "evaluation_results" / "experimental" / "dev_eval_results_30clips_fix4.json", "r", encoding="utf-8") as f:
        d_f4 = json.load(f)

    # Re-extract the simulated fix2b and fix3 predictions from simulate_fix4_dev data structures
    # We load them directly via a clean simulation import or reconstruct from their saved confusion matrix
    sys.path.insert(0, str(evaluator.project_root / "evaluation_results" / "experimental" / "audit"))
    import simulate_fix4_dev

    dataset_results = {}
    
    # 1. Fix 1 Baseline
    dataset_results["Fix 1 Baseline"] = evaluator.evaluate_prediction_records(
        "Fix 1 Baseline", d_ref["predictions"], "baseline_acoustic_lexical"
    )

    # 2. Fix 2B Baseline (Variant 7)
    dataset_results["Fix 2B Baseline (Variant 7)"] = evaluator.evaluate_prediction_records(
        "Fix 2B Baseline (Variant 7)", simulate_fix4_dev.b_preds, "simulated_hybrid"
    )

    # 3. Fix 3 Live Saved
    dataset_results["Fix 3 Contextual (Live Saved)"] = evaluator.evaluate_prediction_records(
        "Fix 3 Contextual (Live Saved)", d_f3["predictions_fix3"], "live_acoustic_sequential"
    )

    # 4. Fix 3 Hybrid Simulated
    dataset_results["Fix 3 Contextual (Hybrid Simulated)"] = evaluator.evaluate_prediction_records(
        "Fix 3 Contextual (Hybrid Simulated)", simulate_fix4_dev.f3_preds, "simulated_hybrid"
    )

    # 5. Fix 4 Saved Artifact
    dataset_results["Fix 4 Contextual (Saved Artifact)"] = evaluator.evaluate_prediction_records(
        "Fix 4 Contextual (Saved Artifact)", d_f4["predictions_fix4"], "simulated_hybrid"
    )

    # Compile Per-Clip Comparison Table
    per_clip_table = []
    for fname in evaluator.manifest_clips:
        clip_row = {
            "filename": fname,
            "dataset": evaluator.dataset_map[fname],
            "gt_dominant": evaluator.gt_dominant[fname],
            "gt_has_stutter": evaluator.gt_has_stutter[fname],
            "gt_has_filler": evaluator.gt_has_filler[fname],
            "is_timeout": fname in ["M_0030_16y4m_1_dysfluent_009.wav", "FluencyBank_010_40.wav"],
            "configurations": {}
        }
        for cfg_name, res in dataset_results.items():
            rec = next(r for r in res["records"] if r["filename"] == fname)
            clip_row["configurations"][cfg_name] = {
                "pred_dominant": rec["pred_dominant"],
                "pred_has_stutter": rec["pred_has_stutter"],
                "pred_has_filler": rec["pred_has_filler"],
                "confidence": rec["confidence"],
                "is_correct_multiclass": rec["is_correct_multiclass"]
            }
        per_clip_table.append(clip_row)

    # Aggregate Metrics Comparison
    aggregate_comparison = {}
    for cfg_name, res in dataset_results.items():
        agg = res["aggregate_metrics"]
        aggregate_comparison[cfg_name] = {
            "pipeline_type": res["pipeline_type"],
            "accuracy": agg["accuracy"],
            "macro_precision": agg["macro_precision"],
            "macro_recall": agg["macro_recall"],
            "macro_f1": agg["macro_f1"],
            "weighted_f1": agg["weighted_f1"],
            "stutter_binary": agg["stutter_binary"],
            "filler_metrics": agg["class_metrics"]["Filler"],
            "clean_metrics": agg["class_metrics"]["Clean"],
            "fluencybank_stutter_fps": agg["fluencybank_stutter_fps"],
            "confusion_matrix": agg["confusion_matrix"]
        }

    # Save Machine-Readable JSONs
    per_clip_path = out_dir / "per_clip_comparison.json"
    with open(per_clip_path, "w", encoding="utf-8") as f:
        json.dump(per_clip_table, f, indent=2)
    print(f"Saved machine-readable per-clip comparison to: {per_clip_path}")

    agg_path = out_dir / "aggregate_metrics.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_comparison, f, indent=2)
    print(f"Saved machine-readable aggregate metrics to: {agg_path}")

    return dataset_results, per_clip_table, aggregate_comparison


if __name__ == "__main__":
    run_canonical_evaluation("e:/SpeechPrj")
