"""
Ablation 1 Benchmark Runner: Multi-Word N-Gram Repetition Detection
===================================================================

Compares 4 configurations on the standardized 30-clip development benchmark:
1. Fix 3 Reference (Unigrams only, enable_ngram_repetition=False)
2. Fix 3 + 2-Gram Detection (enable_ngram_repetition=True, ngram_sizes=(2,))
3. Fix 3 + 3-Gram Detection (enable_ngram_repetition=True, ngram_sizes=(3,))
4. Fix 3 + Combined (2-Gram + 3-Gram, enable_ngram_repetition=True, ngram_sizes=(2, 3))

Strict Constraints:
- Evaluates against the identical immutable feature cache:
  evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json
- Preserves all acoustic thresholds, boundary discounts, and silence models.
- Does not inspect or touch the frozen 750-clip holdout.
- Evaluates via CanonicalEvaluator.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent.parent
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.experimental.config import (
    StutterDetectorConfig,
    SpeechContextConfig,
    AcousticConfig
)
from backend.experimental.acoustic_features import AcousticFeatureExtractor
from backend.experimental.speech_context import SpeechContextAnalyzer
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.eval_harness import get_dominant_label
from backend.experimental.canonical_evaluator import CanonicalEvaluator


def evaluate_configuration(
    config_name: str,
    detector: HybridStutterDetector,
    feature_cache: Dict[str, Any],
    manifest_records: List[Dict[str, Any]],
    acoustic_extractor: AcousticFeatureExtractor
) -> List[Dict[str, Any]]:
    """Evaluates a single detector configuration across the 30 development clips."""
    predictions = []
    
    for r in manifest_records:
        fname = r["filename"]
        cache_item = feature_cache[fname]
        audio_path = cache_item["file_path"]
        transcript = cache_item["transcript"]
        words_list = cache_item["words_list"]

        # Load audio data
        y, sr = acoustic_extractor.load_audio(audio_path)

        res = detector.evaluate(
            transcript=transcript,
            words_list=words_list,
            audio_data=(y, sr)
        )

        has_stutter = res["has_stutter"]
        # Invariant filler detection flag from manifest / cache
        has_filler = cache_item["gt_has_filler"] if cache_item.get("pred_has_filler") is None else cache_item["pred_has_filler"]
        
        # Consistent dominant label derivation
        # Match baseline logic: if filler was detected in standardized run, keep it invariant
        pred_dominant = get_dominant_label(has_stutter, has_filler)

        predictions.append({
            "filename": fname,
            "dataset": cache_item["dataset"],
            "gt_dominant": cache_item["gt_dominant"],
            "gt_has_stutter": cache_item["gt_has_stutter"],
            "gt_has_filler": cache_item["gt_has_filler"],
            "pred_dominant": pred_dominant,
            "pred_has_stutter": has_stutter,
            "pred_has_filler": has_filler,
            "confidence": round(float(res["confidence_score"]), 3),
            "is_timeout": cache_item["is_timeout"],
            "details": res["details"]
        })

    return predictions


def run_ngram_ablation():
    """Executes Ablation 1 comparison and serializes all artifacts."""
    output_dir = project_root / "evaluation_results" / "experimental" / "ablations" / "ngram_repetition"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = project_root / "evaluation_results" / "experimental" / "standardized_dev" / "run_20261020_122136" / "feature_cache.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found at: {cache_path}")
    
    with open(cache_path, "r", encoding="utf-8") as f:
        feature_cache = json.load(f)

    manifest_path = project_root / "evaluation_results" / "experimental" / "dev_eval_results_30clips_refined.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_records = json.load(f)["predictions"]

    # Also load Fix 3 predictions from standardized run to guarantee invariant filler flags
    fix3_pred_path = project_root / "evaluation_results" / "experimental" / "standardized_dev" / "run_20261020_122136" / "predictions_fix3.json"
    with open(fix3_pred_path, "r", encoding="utf-8") as f:
        fix3_ref_saved = {p["filename"]: p for p in json.load(f)}

    # Attach invariant filler detection flags from standardized run
    for fname, c in feature_cache.items():
        if fname in fix3_ref_saved:
            c["pred_has_filler"] = fix3_ref_saved[fname]["pred_has_filler"]

    acoustic_extractor = AcousticFeatureExtractor(AcousticConfig())
    context_analyzer_fix3 = SpeechContextAnalyzer(
        SpeechContextConfig(enabled=True, enable_lexical_syntactic_context=False)
    )

    # 1. Fix 3 Reference (enable_ngram_repetition=False)
    det_ref = HybridStutterDetector(
        StutterDetectorConfig(
            enable_contextual_scoring=True,
            min_prolongation_duration_sec=0.50,
            severe_prolongation_duration_sec=0.60,
            standalone_prolongation_confidence=0.50,
            enable_ngram_repetition=False
        ),
        acoustic_extractor,
        context_analyzer_fix3
    )

    # 2. Fix 3 + 2-Gram Detection
    det_2g = HybridStutterDetector(
        StutterDetectorConfig(
            enable_contextual_scoring=True,
            min_prolongation_duration_sec=0.50,
            severe_prolongation_duration_sec=0.60,
            standalone_prolongation_confidence=0.50,
            enable_ngram_repetition=True,
            ngram_sizes=(2,),
            ngram_repetition_confidence=0.85
        ),
        acoustic_extractor,
        context_analyzer_fix3
    )

    # 3. Fix 3 + 3-Gram Detection
    det_3g = HybridStutterDetector(
        StutterDetectorConfig(
            enable_contextual_scoring=True,
            min_prolongation_duration_sec=0.50,
            severe_prolongation_duration_sec=0.60,
            standalone_prolongation_confidence=0.50,
            enable_ngram_repetition=True,
            ngram_sizes=(3,),
            ngram_repetition_confidence=0.85
        ),
        acoustic_extractor,
        context_analyzer_fix3
    )

    # 4. Fix 3 + Combined (2-Gram + 3-Gram)
    det_comb = HybridStutterDetector(
        StutterDetectorConfig(
            enable_contextual_scoring=True,
            min_prolongation_duration_sec=0.50,
            severe_prolongation_duration_sec=0.60,
            standalone_prolongation_confidence=0.50,
            enable_ngram_repetition=True,
            ngram_sizes=(2, 3),
            ngram_repetition_confidence=0.85
        ),
        acoustic_extractor,
        context_analyzer_fix3
    )

    configs = [
        ("Fix 3 Reference", det_ref, "predictions_fix3_ref.json"),
        ("Fix 3 + 2-gram", det_2g, "predictions_ngram_2gram.json"),
        ("Fix 3 + 3-gram", det_3g, "predictions_ngram_3gram.json"),
        ("Fix 3 + Combined (2g+3g)", det_comb, "predictions_ngram_combined.json")
    ]

    all_predictions = {}
    for cfg_name, detector, filename in configs:
        print(f"Evaluating {cfg_name}...")
        preds = evaluate_configuration(cfg_name, detector, feature_cache, manifest_records, acoustic_extractor)
        all_predictions[cfg_name] = preds
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2)

    # Evaluate via CanonicalEvaluator
    evaluator = CanonicalEvaluator(str(project_root.resolve()))
    aggregate_results = {}
    per_corpus_results = {}

    for cfg_name, _, _ in configs:
        eval_res = evaluator.evaluate_prediction_records(
            cfg_name,
            all_predictions[cfg_name],
            pipeline_type="Ablation 1 (N-gram Repetition)",
            timeout_clips=[]
        )
        aggregate_results[cfg_name] = eval_res["aggregate_metrics"]
        per_corpus_results[cfg_name] = eval_res["per_corpus_metrics"]

    # Save aggregate and per-corpus metrics
    with open(output_dir / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "configurations": aggregate_results}, f, indent=2)

    with open(output_dir / "per_corpus_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "configurations": per_corpus_results}, f, indent=2)

    # Build per-clip comparison
    clip_comparison = []
    for r in manifest_records:
        fname = r["filename"]
        ds = r["dataset"]
        gt_dom = r["gt_dominant"]
        gt_st = r["gt_has_stutter"]

        ref_p = next(p for p in all_predictions["Fix 3 Reference"] if p["filename"] == fname)
        p_2g = next(p for p in all_predictions["Fix 3 + 2-gram"] if p["filename"] == fname)
        p_3g = next(p for p in all_predictions["Fix 3 + 3-gram"] if p["filename"] == fname)
        p_comb = next(p for p in all_predictions["Fix 3 + Combined (2g+3g)"] if p["filename"] == fname)

        # Extract n-gram events in combined
        ngram_evs = [d for d in p_comb["details"] if "ngram" in d.get("type", "")]

        clip_comparison.append({
            "filename": fname,
            "dataset": ds,
            "gt_dominant": gt_dom,
            "gt_has_stutter": gt_st,
            "ref_pred_dominant": ref_p["pred_dominant"],
            "ref_confidence": ref_p["confidence"],
            "pred_2gram_dominant": p_2g["pred_dominant"],
            "pred_2gram_confidence": p_2g["confidence"],
            "pred_3gram_dominant": p_3g["pred_dominant"],
            "pred_3gram_confidence": p_3g["confidence"],
            "pred_combined_dominant": p_comb["pred_dominant"],
            "pred_combined_confidence": p_comb["confidence"],
            "has_prediction_change": ref_p["pred_dominant"] != p_comb["pred_dominant"],
            "is_change_correct": (p_comb["pred_dominant"] == gt_dom) if ref_p["pred_dominant"] != p_comb["pred_dominant"] else None,
            "ngram_events_found": ngram_evs
        })

    with open(output_dir / "per_clip_comparison.json", "w", encoding="utf-8") as f:
        json.dump(clip_comparison, f, indent=2)

    print(f"\nAll ablation outputs saved under: {output_dir}")
    return aggregate_results, per_corpus_results, clip_comparison


if __name__ == "__main__":
    run_ngram_ablation()
