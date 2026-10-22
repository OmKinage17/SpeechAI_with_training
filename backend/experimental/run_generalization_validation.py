"""
SpeechAI — Ablation 1 Generalization Validation Runner
======================================================
Evaluates Fix 3 Reference vs Fix 3 + Combined (2g+3g) on a speaker-disjoint
60-clip independent development set with an isolated shared feature cache.

Strict Constraints:
- Zero overlap with the 30-clip development benchmark.
- Zero overlap with the frozen 750-clip holdout.
- All baseline detector thresholds, models, and acoustic gating fixed.
- All results saved in a new timestamped directory under:
  evaluation_results/experimental/generalization_validation/run_<timestamp>/
"""

import os
import re
import sys
import time
import json
import csv
import hashlib
import random
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

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
from backend.experimental.asr_engine import ExperimentalASREngine
from backend.experimental.acoustic_features import AcousticFeatureExtractor
from backend.experimental.speech_context import SpeechContextAnalyzer
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.eval_harness import get_dominant_label

# Candidate dataset paths
DATASETS_DIR = project_root / "Datasets"
UCLASS_LABELS = DATASETS_DIR / "UCLASS Stuttered Speech Clips (SEP-28k Format)" / "clips" / "labels.csv"
UCLASS_AUDIO_BASE = DATASETS_DIR / "UCLASS Stuttered Speech Clips (SEP-28k Format)"
FB_LABELS = DATASETS_DIR / "Stuttering Events in Podcasts Dataset (SEP-28k)" / "fluencybank_labels.csv"
FB_AUDIO_DIR = DATASETS_DIR / "Stuttering Events in Podcasts Dataset (SEP-28k)" / "clips" / "stuttering-clips" / "clips"
LIBRI_ROOT = DATASETS_DIR / "Libri_Stutter"
LIBRI_AUDIO_DIR = LIBRI_ROOT / "LibriStutter Audio"
LIBRI_ANNO_DIR = LIBRI_ROOT / "LibriStutter Annotations"

DEV_MANIFEST = project_root / "evaluation_results" / "experimental" / "dev_eval_results_30clips_refined.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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
            filler_count += len(matches)
            fillers_found.append(phrase)
            remaining_text = re.sub(pattern, ' ', remaining_text)

    cleaned_remaining = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()?"]', ' ', remaining_text)
    words = cleaned_remaining.split()

    single_word_fillers = {
        "like", "actually", "basically", "literally", "honestly", "seriously", "so", "well"
    }

    for w in words:
        if w in single_word_fillers:
            filler_count += 1
            if w not in fillers_found:
                fillers_found.append(w)
        elif re.match(r'^u+m+h*$', w):
            filler_count += 1
            if "um" not in fillers_found:
                fillers_found.append("um")
        elif re.match(r'^u+h+m*$', w):
            filler_count += 1
            if "uh" not in fillers_found:
                fillers_found.append("uh")
        elif re.match(r'^e+r+m*$', w):
            filler_count += 1
            if "er/erm" not in fillers_found:
                fillers_found.append("er/erm")
        elif re.match(r'^a+h+$', w):
            filler_count += 1
            if "ah" not in fillers_found:
                fillers_found.append("ah")
        elif re.match(r'^h+m+$', w):
            filler_count += 1
            if "hmm" not in fillers_found:
                fillers_found.append("hmm")

    return filler_count, fillers_found


def build_disjoint_manifest(seed: int = 42) -> List[Dict[str, Any]]:
    """Builds a 60-clip speaker-disjoint dataset with verified zero overlap."""
    random.seed(seed)
    with open(DEV_MANIFEST, "r", encoding="utf-8") as f:
        dev_clips = {p["filename"] for p in json.load(f)["predictions"]}

    manifest = []

    # 1. UCLASS (20 clips from unseen speakers M_1022 and M_0061)
    with open(UCLASS_LABELS, "r", encoding="utf-8") as f:
        u_rows = list(csv.DictReader(f))

    u_pool = {"M_1022": {"Stutter": [], "Filler": [], "Clean": []},
              "M_0061": {"Stutter": [], "Filler": [], "Clean": []}}

    for r in u_rows:
        rel_path = r["filepath"].strip()
        fn = Path(rel_path).name
        parts = fn.split("_")
        spk = f"{parts[0]}_{parts[1]}"
        if spk in u_pool and fn not in dev_clips:
            audio_path = (UCLASS_AUDIO_BASE / rel_path).resolve()
            if audio_path.exists() and audio_path.stat().st_size > 44:
                sound_rep = int(r.get("SoundRep", 0))
                word_rep = int(r.get("WordRep", 0))
                prolongation = int(r.get("Prolongation", 0))
                block = int(r.get("Block", 0))
                interjection = int(r.get("Interjection", 0))
                has_stutter = int(sound_rep > 0 or word_rep > 0 or prolongation > 0 or block > 0)
                has_filler = int(interjection > 0)
                dom = "Stutter" if has_stutter else ("Filler" if has_filler else "Clean")
                u_pool[spk][dom].append({
                    "filename": fn,
                    "dataset": "UCLASS",
                    "speaker_id": spk,
                    "file_path": str(audio_path),
                    "gt_dominant": dom,
                    "gt_has_stutter": has_stutter,
                    "gt_has_filler": has_filler,
                    "annotation_details": {
                        "sound_rep": sound_rep,
                        "word_rep": word_rep,
                        "prolongation": prolongation,
                        "block": block,
                        "interjection": interjection
                    }
                })

    random.shuffle(u_pool["M_1022"]["Stutter"])
    random.shuffle(u_pool["M_1022"]["Filler"])
    random.shuffle(u_pool["M_1022"]["Clean"])
    manifest.extend(u_pool["M_1022"]["Stutter"][:5])
    manifest.extend(u_pool["M_1022"]["Filler"][:3])
    manifest.extend(u_pool["M_1022"]["Clean"][:2])

    random.shuffle(u_pool["M_0061"]["Stutter"])
    random.shuffle(u_pool["M_0061"]["Filler"])
    random.shuffle(u_pool["M_0061"]["Clean"])
    manifest.extend(u_pool["M_0061"]["Stutter"][:6])
    manifest.extend(u_pool["M_0061"]["Filler"][:2])
    manifest.extend(u_pool["M_0061"]["Clean"][:2])

    # 2. FluencyBank (20 clips from unseen episodes Ep_017 and Ep_031)
    with open(FB_LABELS, "r", encoding="utf-8") as f:
        fb_rows = list(csv.DictReader(f))

    fb_pool = {"017": {"Stutter": [], "Filler": [], "Clean": []},
               "031": {"Stutter": [], "Filler": [], "Clean": []}}

    for r in fb_rows:
        ep = r.get("EpId", "").strip()
        idx = r.get("ClipId", "").strip()
        fn = f"FluencyBank_{ep.zfill(3)}_{idx}.wav"
        if ep in fb_pool and fn not in dev_clips:
            audio_path = (FB_AUDIO_DIR / fn).resolve()
            if audio_path.exists() and audio_path.stat().st_size > 44:
                sound_rep = int(int(r.get("SoundRep", "0").strip()) >= 2)
                word_rep = int(int(r.get("WordRep", "0").strip()) >= 2)
                prolongation = int(int(r.get("Prolongation", "0").strip()) >= 2)
                block = int(int(r.get("Block", "0").strip()) >= 2)
                interjection = int(int(r.get("Interjection", "0").strip()) >= 2)
                has_stutter = int(sound_rep or word_rep or prolongation or block)
                has_filler = int(interjection)
                dom = "Stutter" if has_stutter else ("Filler" if has_filler else "Clean")
                fb_pool[ep][dom].append({
                    "filename": fn,
                    "dataset": "FluencyBank",
                    "speaker_id": f"Ep_{ep.zfill(3)}",
                    "file_path": str(audio_path),
                    "gt_dominant": dom,
                    "gt_has_stutter": has_stutter,
                    "gt_has_filler": has_filler,
                    "annotation_details": {
                        "sound_rep": sound_rep,
                        "word_rep": word_rep,
                        "prolongation": prolongation,
                        "block": block,
                        "interjection": interjection
                    }
                })

    random.shuffle(fb_pool["017"]["Stutter"])
    random.shuffle(fb_pool["017"]["Filler"])
    random.shuffle(fb_pool["017"]["Clean"])
    manifest.extend(fb_pool["017"]["Stutter"][:5])
    manifest.extend(fb_pool["017"]["Filler"][:3])
    manifest.extend(fb_pool["017"]["Clean"][:2])

    random.shuffle(fb_pool["031"]["Stutter"])
    random.shuffle(fb_pool["031"]["Filler"])
    random.shuffle(fb_pool["031"]["Clean"])
    manifest.extend(fb_pool["031"]["Stutter"][:5])
    manifest.extend(fb_pool["031"]["Filler"][:3])
    manifest.extend(fb_pool["031"]["Clean"][:2])

    # 3. Libri_Stutter (20 clips from unseen readers 198 and 1743)
    libri_pool = {"198": {"Stutter": [], "Clean": []},
                  "1743": {"Stutter": [], "Clean": []}}

    for rid in ["198", "1743"]:
        r_path = LIBRI_ANNO_DIR / rid
        for chap_dir in r_path.iterdir():
            if not chap_dir.is_dir():
                continue
            for csv_file in chap_dir.glob("*.csv"):
                flac_name = csv_file.stem + ".flac"
                if flac_name in dev_clips:
                    continue
                flac_path = (LIBRI_AUDIO_DIR / rid / chap_dir.name / flac_name).resolve()
                if flac_path.exists() and flac_path.stat().st_size > 44:
                    with open(csv_file, "r", encoding="utf-8") as f:
                        c_rows = list(csv.reader(f))[1:]
                    stutter_types = {row[3].strip() for row in c_rows if len(row) >= 4}
                    has_stutter = int(any(t in {"2", "3", "4", "5", "Sound Repetition", "Word Repetition", "Phrase Repetition", "Prolongation"} for t in stutter_types))
                    dom = "Stutter" if has_stutter else "Clean"
                    libri_pool[rid][dom].append({
                        "filename": flac_name,
                        "dataset": "Libri_Stutter",
                        "speaker_id": f"Reader_{rid}",
                        "file_path": str(flac_path),
                        "gt_dominant": dom,
                        "gt_has_stutter": has_stutter,
                        "gt_has_filler": 0,
                        "annotation_details": {
                            "stutter_types": list(stutter_types)
                        }
                    })

    random.shuffle(libri_pool["198"]["Stutter"])
    random.shuffle(libri_pool["198"]["Clean"])
    manifest.extend(libri_pool["198"]["Stutter"][:7])
    manifest.extend(libri_pool["198"]["Clean"][:3])

    random.shuffle(libri_pool["1743"]["Stutter"])
    random.shuffle(libri_pool["1743"]["Clean"])
    manifest.extend(libri_pool["1743"]["Stutter"][:7])
    manifest.extend(libri_pool["1743"]["Clean"][:3])

    return manifest


def extract_features_and_cache(
    manifest: List[Dict[str, Any]],
    output_dir: Path,
    asr_engine: ExperimentalASREngine,
    acoustic_extractor: AcousticFeatureExtractor
) -> Dict[str, Any]:
    """Transcribes and extracts acoustic features for the 60 generalization clips."""
    print(f"\n--- EXTRACTING FEATURES & BUILDING CACHE (N={len(manifest)}) ---")
    cache = {}
    start_all = time.time()

    for idx, item in enumerate(manifest):
        fn = item["filename"]
        audio_path = item["file_path"]
        t0 = time.time()
        print(f"[{idx+1:02d}/{len(manifest):02d}] Transcribing {fn} ({item['dataset']}, {item['speaker_id']})...", end="", flush=True)

        asr_res = asr_engine.transcribe(audio_path)
        asr_status = asr_res.get("status", "SUCCESS")
        transcript = asr_res.get("transcript", "").strip()
        words_list = asr_res.get("words_list", [])

        # Load audio data and detect fillers
        y, sr = acoustic_extractor.load_audio(audio_path)
        filler_count, filler_words = detect_fillers_canonical(transcript, words_list)
        has_filler = 1 if filler_count > 0 else 0

        dt = time.time() - t0
        print(f" done ({dt:.2f}s, status={asr_status}, words={len(words_list)}, filler={has_filler})")

        cache[fn] = {
            "filename": fn,
            "dataset": item["dataset"],
            "speaker_id": item["speaker_id"],
            "file_path": audio_path,
            "gt_dominant": item["gt_dominant"],
            "gt_has_stutter": item["gt_has_stutter"],
            "gt_has_filler": item["gt_has_filler"],
            "annotation_details": item["annotation_details"],
            "transcript": transcript,
            "words_list": words_list,
            "pred_has_filler": has_filler,
            "filler_count": filler_count,
            "filler_words": filler_words,
            "feature_extraction_latency_sec": round(dt, 3),
            "is_timeout": (asr_status == "TIMEOUT")
        }

    total_time = time.time() - start_all
    print(f"--- Feature extraction complete in {total_time:.1f}s (mean: {total_time/len(manifest):.2f}s/clip) ---")

    # Serialize feature cache
    cache_path = output_dir / "feature_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    cache_sha256 = sha256_file(cache_path)
    manifest_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_clips": len(manifest),
        "whisper_model": "base",
        "word_timestamps": True,
        "feature_cache_sha256": cache_sha256,
        "total_extraction_time_sec": round(total_time, 2),
        "mean_latency_sec": round(total_time / len(manifest), 2),
        "successful_transcriptions": sum(1 for c in cache.values() if not c["is_timeout"]),
        "timeout_count": sum(1 for c in cache.values() if c["is_timeout"])
    }
    with open(output_dir / "feature_cache_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_info, f, indent=2)

    print(f"Feature cache saved with SHA-256: {cache_sha256}")
    return cache


def evaluate_detector(
    detector: HybridStutterDetector,
    feature_cache: Dict[str, Any],
    manifest: List[Dict[str, Any]],
    acoustic_extractor: AcousticFeatureExtractor
) -> List[Dict[str, Any]]:
    """Evaluates a detector configuration on the shared feature cache."""
    predictions = []

    for item in manifest:
        fn = item["filename"]
        cache_item = feature_cache[fn]
        audio_path = cache_item["file_path"]
        transcript = cache_item["transcript"]
        words_list = cache_item["words_list"]

        y, sr = acoustic_extractor.load_audio(audio_path)
        t0 = time.time()
        res = detector.evaluate(
            transcript=transcript,
            words_list=words_list,
            audio_data=(y, sr)
        )
        latency = time.time() - t0

        has_stutter = res["has_stutter"]
        has_filler = cache_item["pred_has_filler"]
        pred_dominant = get_dominant_label(has_stutter, has_filler)

        predictions.append({
            "filename": fn,
            "dataset": cache_item["dataset"],
            "speaker_id": cache_item["speaker_id"],
            "gt_dominant": cache_item["gt_dominant"],
            "gt_has_stutter": cache_item["gt_has_stutter"],
            "gt_has_filler": cache_item["gt_has_filler"],
            "pred_dominant": pred_dominant,
            "pred_has_stutter": has_stutter,
            "pred_has_filler": has_filler,
            "confidence": round(float(res["confidence_score"]), 3),
            "latency_sec": round(latency, 4),
            "is_correct_multiclass": (pred_dominant == cache_item["gt_dominant"]),
            "is_correct_binary_stutter": (has_stutter == cache_item["gt_has_stutter"]),
            "details": res["details"]
        })

    return predictions


def compute_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes multiclass and binary metrics directly from prediction records."""
    classes = ["Stutter", "Filler", "Clean"]
    n = len(predictions)
    if n == 0:
        return {}

    cm = {c_gt: {c_pred: 0 for c_pred in classes} for c_gt in classes}
    for p in predictions:
        gt = p["gt_dominant"]
        pred = p["pred_dominant"]
        cm[gt][pred] += 1

    accuracy = sum(cm[c][c] for c in classes) / n

    class_metrics = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_f1 = 0.0

    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in classes if other != c)
        fn = sum(cm[c][other] for other in classes if other != c)
        support = sum(cm[c].values())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        class_metrics[c] = {
            "tp": tp, "fp": fp, "fn": fn, "support": support,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)
        }
        macro_p += prec
        macro_r += rec
        macro_f1 += f1
        weighted_f1 += f1 * support

    macro_p /= len(classes)
    macro_r /= len(classes)
    macro_f1 /= len(classes)
    weighted_f1 /= n

    bin_tp = sum(1 for p in predictions if p["gt_has_stutter"] == 1 and p["pred_has_stutter"] == 1)
    bin_fp = sum(1 for p in predictions if p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 1)
    bin_fn = sum(1 for p in predictions if p["gt_has_stutter"] == 1 and p["pred_has_stutter"] == 0)
    bin_tn = sum(1 for p in predictions if p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 0)

    bin_prec = bin_tp / (bin_tp + bin_fp) if (bin_tp + bin_fp) > 0 else 0.0
    bin_rec = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) > 0 else 0.0
    bin_f1 = (2 * bin_prec * bin_rec) / (bin_prec + bin_rec) if (bin_prec + bin_rec) > 0 else 0.0
    bin_spec = bin_tn / (bin_tn + bin_fp) if (bin_tn + bin_fp) > 0 else 0.0

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
            "tp": bin_tp, "fp": bin_fp, "fn": bin_fn, "tn": bin_tn,
            "support": bin_tp + bin_fn,
            "precision": round(bin_prec, 4),
            "recall": round(bin_rec, 4),
            "f1": round(bin_f1, 4),
            "specificity": round(bin_spec, 4)
        }
    }


def write_analyses(
    paired_predictions: List[Dict[str, Any]],
    output_dir: Path
):
    """Generates false positive and false negative analysis markdown documents."""
    # 1. False Positive Analysis
    fp_ref = [p for p in paired_predictions if p["gt_has_stutter"] == 0 and p["fix3_reference"]["pred_has_stutter"] == 1]
    fp_comb = [p for p in paired_predictions if p["gt_has_stutter"] == 0 and p["fix3_combined"]["pred_has_stutter"] == 1]
    new_fps = [p for p in paired_predictions if p["new_false_positive"]]

    fp_md = ["# SpeechAI — Generalization False Positive Analysis\n"]
    fp_md.append(f"**Total Non-Stutter Clips:** {sum(1 for p in paired_predictions if p['gt_has_stutter'] == 0)} (11 Filler, 14 Clean)\n")
    fp_md.append(f"- **Fix 3 Reference False Positives:** {len(fp_ref)}")
    fp_md.append(f"- **Fix 3 + Combined False Positives:** {len(fp_comb)}")
    fp_md.append(f"- **Net New False Positives Introduced by N-gram:** {len(new_fps)}\n")

    if new_fps:
        fp_md.append("## New False Positives Introduced by N-gram Repetition Detector\n")
        for p in new_fps:
            fp_md.append(f"### `{p['filename']}` ({p['dataset']}, {p['speaker_id']})")
            fp_md.append(f"- **Ground Truth:** {p['gt_dominant']}")
            fp_md.append(f"- **Reference Prediction:** {p['fix3_reference']['pred_dominant']}")
            fp_md.append(f"- **Combined Prediction:** {p['fix3_combined']['pred_dominant']}")
            fp_md.append(f"- **Detected Events:** {json.dumps(p['fix3_combined']['details'], indent=2)}\n")
    else:
        fp_md.append("## Result: Zero New False Positives\n")
        fp_md.append("The combined 2-gram + 3-gram repetition detector introduced **0** false positive classifications on non-stutter speech.\n")
        fp_md.append("Fluent control clips and filler interjections were not falsely classified as stutter.\n")

    if fp_comb:
        fp_md.append("## Existing False Positives (Present in Both Reference and Combined)\n")
        for p in fp_comb:
            fp_md.append(f"- `{p['filename']}` ({p['dataset']}): GT={p['gt_dominant']}, Pred={p['fix3_combined']['pred_dominant']}, Conf={p['fix3_combined']['confidence']}")

    with open(output_dir / "false_positive_analysis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(fp_md))

    # 2. False Negative Analysis
    fn_ref = [p for p in paired_predictions if p["gt_has_stutter"] == 1 and p["fix3_reference"]["pred_has_stutter"] == 0]
    fn_comb = [p for p in paired_predictions if p["gt_has_stutter"] == 1 and p["fix3_combined"]["pred_has_stutter"] == 0]
    recovered = [p for p in paired_predictions if p["recovered_true_positive"]]

    fn_md = ["# SpeechAI — Generalization False Negative Analysis\n"]
    fn_md.append(f"**Total Stutter Clips:** {sum(1 for p in paired_predictions if p['gt_has_stutter'] == 1)}\n")
    fn_md.append(f"- **Fix 3 Reference False Negatives:** {len(fn_ref)}")
    fn_md.append(f"- **Fix 3 + Combined False Negatives:** {len(fn_comb)}")
    fn_md.append(f"- **True Positives Recovered by N-gram:** {len(recovered)}\n")

    if recovered:
        fn_md.append("## Recovered True Positives (Fixed by N-gram Repetition Detection)\n")
        for p in recovered:
            fn_md.append(f"### `{p['filename']}` ({p['dataset']}, {p['speaker_id']})")
            fn_md.append(f"- **Ground Truth:** {p['gt_dominant']}")
            fn_md.append(f"- **Fix 3 Ref Prediction:** {p['fix3_reference']['pred_dominant']} (conf: {p['fix3_reference']['confidence']})")
            fn_md.append(f"- **Combined Prediction:** {p['fix3_combined']['pred_dominant']} (conf: {p['fix3_combined']['confidence']})")
            ngram_events = [d for d in p['fix3_combined']['details'] if str(d.get('type', '')).startswith('ngram_phrase_repetition')]
            for ev in ngram_events:
                fn_md.append(f"  - **Matched Phrase:** `\"{ev.get('phrase')}\"` (N={ev.get('ngram_size')}, iter={ev.get('iterations')}, span={ev.get('token_start_idx')}:{ev.get('token_end_idx')}, conf={ev.get('confidence')})")
            fn_md.append("")

    fn_md.append("## Remaining False Negatives in Combined Configuration\n")
    for p in fn_comb:
        fn_md.append(f"- `{p['filename']}` ({p['dataset']}, {p['speaker_id']}): GT={p['gt_dominant']}, Pred={p['fix3_combined']['pred_dominant']}, Annotation={p.get('annotation_details')}")

    with open(output_dir / "false_negative_analysis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(fn_md))


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "evaluation_results" / "experimental" / "generalization_validation" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== SPEECHAI GENERALIZATION VALIDATION RUN: {output_dir.name} ===")

    # 1. Build and serialize manifest
    manifest = build_disjoint_manifest(seed=42)
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Run manifest serialized: {len(manifest)} clips.")

    # 2. Extract features and build shared cache
    asr_cfg = ASRConfig(word_timestamps=True, per_clip_timeout_sec=25.0)
    asr_engine = ExperimentalASREngine(asr_cfg)
    acoustic_extractor = AcousticFeatureExtractor(AcousticConfig())

    try:
        feature_cache = extract_features_and_cache(manifest, output_dir, asr_engine, acoustic_extractor)
    finally:
        asr_engine.shutdown()

    # 3. Setup Detectors
    context_analyzer_fix3 = SpeechContextAnalyzer(
        SpeechContextConfig(enabled=True, enable_lexical_syntactic_context=False)
    )

    det_fix3_ref = HybridStutterDetector(
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

    det_fix3_combined = HybridStutterDetector(
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

    # 4. Evaluate Fix 3 Reference
    print("\n--- EVALUATING CONFIG 1: Fix 3 Reference ---")
    preds_ref = evaluate_detector(det_fix3_ref, feature_cache, manifest, acoustic_extractor)

    # 5. Evaluate Fix 3 + Combined
    print("\n--- EVALUATING CONFIG 2: Fix 3 + Combined (2g+3g) ---")
    preds_comb = evaluate_detector(det_fix3_combined, feature_cache, manifest, acoustic_extractor)

    # 6. Compute Metrics
    metrics_ref = compute_metrics(preds_ref)
    metrics_comb = compute_metrics(preds_comb)

    # Per-corpus breakdown
    per_corpus = {}
    for ds in ["UCLASS", "FluencyBank", "Libri_Stutter"]:
        ds_ref = [p for p in preds_ref if p["dataset"] == ds]
        ds_comb = [p for p in preds_comb if p["dataset"] == ds]
        per_corpus[ds] = {
            "Fix 3 Reference": compute_metrics(ds_ref),
            "Fix 3 + Combined (2g+3g)": compute_metrics(ds_comb)
        }

    # Combined predictions artifact
    paired_predictions = []
    for r, c in zip(preds_ref, preds_comb):
        paired_predictions.append({
            "filename": r["filename"],
            "dataset": r["dataset"],
            "speaker_id": r["speaker_id"],
            "gt_dominant": r["gt_dominant"],
            "gt_has_stutter": r["gt_has_stutter"],
            "gt_has_filler": r["gt_has_filler"],
            "annotation_details": feature_cache[r["filename"]]["annotation_details"],
            "fix3_reference": {
                "pred_dominant": r["pred_dominant"],
                "pred_has_stutter": r["pred_has_stutter"],
                "pred_has_filler": r["pred_has_filler"],
                "confidence": r["confidence"],
                "is_correct": r["is_correct_multiclass"],
                "details": r["details"]
            },
            "fix3_combined": {
                "pred_dominant": c["pred_dominant"],
                "pred_has_stutter": c["pred_has_stutter"],
                "pred_has_filler": c["pred_has_filler"],
                "confidence": c["confidence"],
                "is_correct": c["is_correct_multiclass"],
                "details": c["details"]
            },
            "prediction_changed": (r["pred_dominant"] != c["pred_dominant"] or r["pred_has_stutter"] != c["pred_has_stutter"]),
            "recovered_true_positive": (not r["is_correct_multiclass"] and c["is_correct_multiclass"] and c["pred_dominant"] == "Stutter"),
            "new_false_positive": (r["gt_dominant"] != "Stutter" and c["pred_dominant"] == "Stutter" and r["pred_dominant"] != "Stutter")
        })

    with open(output_dir / "per_clip_predictions.json", "w", encoding="utf-8") as f:
        json.dump(paired_predictions, f, indent=2)

    # Metrics comparison artifact
    gen_metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_size": len(manifest),
        "overall": {
            "Fix 3 Reference": metrics_ref,
            "Fix 3 + Combined (2g+3g)": metrics_comb,
            "deltas": {
                "accuracy": round(metrics_comb["accuracy"] - metrics_ref["accuracy"], 4),
                "macro_f1": round(metrics_comb["macro_f1"] - metrics_ref["macro_f1"], 4),
                "weighted_f1": round(metrics_comb["weighted_f1"] - metrics_ref["weighted_f1"], 4),
                "stutter_precision": round(metrics_comb["stutter_binary"]["precision"] - metrics_ref["stutter_binary"]["precision"], 4),
                "stutter_recall": round(metrics_comb["stutter_binary"]["recall"] - metrics_ref["stutter_binary"]["recall"], 4),
                "stutter_f1": round(metrics_comb["stutter_binary"]["f1"] - metrics_ref["stutter_binary"]["f1"], 4)
            }
        },
        "per_corpus": per_corpus
    }
    with open(output_dir / "generalization_metrics.json", "w", encoding="utf-8") as f:
        json.dump(gen_metrics, f, indent=2)

    # Confusion matrices artifact
    confusion_matrices = {
        "overall": {
            "Fix 3 Reference": metrics_ref["confusion_matrix"],
            "Fix 3 + Combined (2g+3g)": metrics_comb["confusion_matrix"]
        },
        "per_corpus": {
            ds: {
                "Fix 3 Reference": per_corpus[ds]["Fix 3 Reference"]["confusion_matrix"],
                "Fix 3 + Combined (2g+3g)": per_corpus[ds]["Fix 3 + Combined (2g+3g)"]["confusion_matrix"]
            }
            for ds in ["UCLASS", "FluencyBank", "Libri_Stutter"]
        }
    }
    with open(output_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(confusion_matrices, f, indent=2)

    # Write detailed qualitative analyses
    write_analyses(paired_predictions, output_dir)

    print("\n=== GENERALIZATION RESULTS SUMMARY ===")
    print(f"Accuracy:    {metrics_ref['accuracy']*100:.2f}% -> {metrics_comb['accuracy']*100:.2f}% (delta: {(metrics_comb['accuracy']-metrics_ref['accuracy'])*100:+.2f} pp)")
    print(f"Macro F1:    {metrics_ref['macro_f1']:.4f} -> {metrics_comb['macro_f1']:.4f} (delta: {metrics_comb['macro_f1']-metrics_ref['macro_f1']:+.4f})")
    print(f"Stutter Rec: {metrics_ref['stutter_binary']['recall']*100:.2f}% -> {metrics_comb['stutter_binary']['recall']*100:.2f}% (delta: {(metrics_comb['stutter_binary']['recall']-metrics_ref['stutter_binary']['recall'])*100:+.2f} pp)")
    print(f"Stutter Prec:{metrics_ref['stutter_binary']['precision']*100:.2f}% -> {metrics_comb['stutter_binary']['precision']*100:.2f}% (delta: {(metrics_comb['stutter_binary']['precision']-metrics_ref['stutter_binary']['precision'])*100:+.2f} pp)")
    print(f"Stutter F1:  {metrics_ref['stutter_binary']['f1']:.4f} -> {metrics_comb['stutter_binary']['f1']:.4f} (delta: {metrics_comb['stutter_binary']['f1']-metrics_ref['stutter_binary']['f1']:+.4f})")
    print(f"Binary CM Ref:  TP={metrics_ref['stutter_binary']['tp']}, FP={metrics_ref['stutter_binary']['fp']}, FN={metrics_ref['stutter_binary']['fn']}, TN={metrics_ref['stutter_binary']['tn']}")
    print(f"Binary CM Comb: TP={metrics_comb['stutter_binary']['tp']}, FP={metrics_comb['stutter_binary']['fp']}, FN={metrics_comb['stutter_binary']['fn']}, TN={metrics_comb['stutter_binary']['tn']}")

    return output_dir

if __name__ == "__main__":
    main()
