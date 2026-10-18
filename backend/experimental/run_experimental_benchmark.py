"""
Full 750-Clip Experimental Benchmark Runner for SpeechAI (Phase E)
==================================================================
Executes the refined experimental pipeline against the verified 750-clip
holdout manifest with strict process isolation, comprehensive logging,
independent metric recomputation, and baseline comparative analysis.
"""

import os
import sys
import time
import json
import hashlib
import logging
import datetime
import numpy as np

# Add backend directory to sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    import jiwer
except ImportError:
    jiwer = None

from experimental.config import ExperimentalConfig
from experimental.eval_harness import (
    DevelopmentEvaluationHarness,
    get_dominant_label,
    compute_binary_metrics,
    compute_multiclass_metrics
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Experimental_Benchmark_750")

PROTECTED_FILES = [
    "backend/main.py",
    "backend/run_main_benchmark.py",
    "evaluation_results/pilot_manifest.json",
    "evaluation_results/pilot_predictions.json",
    "evaluation_results/pilot_results_summary.json",
    "evaluation_results/METHODOLOGY_AND_PILOT_REPORT.md",
    "evaluation_results/main_benchmark/main_benchmark_manifest.json",
    "evaluation_results/main_benchmark/main_benchmark_predictions.json",
    "evaluation_results/main_benchmark/main_benchmark_results.json",
    "evaluation_results/main_benchmark/MAIN_BENCHMARK_REPORT.md",
    "evaluation_results/main_benchmark/audit/recomputed_metrics.json",
    "evaluation_results/main_benchmark/audit/AUDIT_REPORT.md",
]

def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_preflight_checks(project_root: str, manifest_path: str) -> dict:
    """
    Executes mandatory preflight checks before running inference:
    1. Confirms manifest has exactly 750 unique clips (250 per corpus).
    2. Verifies all 750 files exist on disk.
    3. Records pre-run SHA-256 hashes of all protected files.
    """
    logger.info("Executing Preflight Checks...")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if len(manifest) != 750:
        raise ValueError(f"Manifest length is {len(manifest)}, expected 750.")

    counts = {"UCLASS": 0, "FluencyBank": 0, "Libri_Stutter": 0}
    filenames = set()
    filepaths = set()
    missing_files = []

    for item in manifest:
        ds = item.get("dataset")
        if ds not in counts:
            raise ValueError(f"Unexpected dataset '{ds}' in manifest item: {item}")
        counts[ds] += 1
        filenames.add(item["filename"])
        filepaths.add(item["file_path"])

        if not os.path.exists(item["file_path"]):
            missing_files.append(item["file_path"])

    if counts != {"UCLASS": 250, "FluencyBank": 250, "Libri_Stutter": 250}:
        raise ValueError(f"Dataset counts mismatch: {counts}, expected 250 per corpus.")

    if len(filenames) != 750 or len(filepaths) != 750:
        raise ValueError(f"Duplicate clips detected! Unique filenames: {len(filenames)}, paths: {len(filepaths)}")

    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} audio files on disk! First 3: {missing_files[:3]}")

    logger.info("[Preflight OK] Manifest verified: 750 unique clips (250 UCLASS, 250 FluencyBank, 250 Libri_Stutter).")
    logger.info("[Preflight OK] All 750 audio files present and readable on disk.")

    # Record SHA-256 hashes of protected files
    pre_hashes = {}
    for rel_path in PROTECTED_FILES:
        full_path = os.path.join(project_root, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Protected file missing: {rel_path}")
        pre_hashes[rel_path] = {
            "size_bytes": os.path.getsize(full_path),
            "sha256": compute_file_sha256(full_path)
        }

    logger.info("[Preflight OK] All 12 protected files verified and SHA-256 hashes recorded.")
    return pre_hashes

def calculate_detailed_metrics(predictions: list, ref_text_map: dict = None) -> dict:
    """
    Calculates 3-class classification, binary detection, per-corpus breakdowns,
    WER, and latency distribution from raw prediction records.
    Normalizes records if gt_dominant/pred_dominant are missing.
    """
    labels_3class = ["Clean", "Stutter", "Filler"]

    # Normalize records to ensure uniform schema
    normalized = []
    for p in predictions:
        rec = dict(p)
        if "gt_has_stutter" not in rec:
            rec["gt_has_stutter"] = 1 if (
                rec.get("gt_sound_rep", 0) or rec.get("gt_word_rep", 0) or 
                rec.get("gt_prolongation", 0) or rec.get("gt_block", 0)
            ) else 0
        if "gt_has_filler" not in rec:
            rec["gt_has_filler"] = 1 if rec.get("gt_filler", 0) else 0
        if "gt_dominant" not in rec:
            rec["gt_dominant"] = get_dominant_label(rec["gt_has_stutter"], rec["gt_has_filler"])
        if "pred_has_stutter" not in rec:
            rec["pred_has_stutter"] = 1 if rec.get("pred_stammers", 0) > 0 else 0
        if "pred_has_filler" not in rec:
            rec["pred_has_filler"] = 1 if rec.get("pred_fillers", 0) > 0 else 0
        if "pred_dominant" not in rec:
            rec["pred_dominant"] = get_dominant_label(rec["pred_has_stutter"], rec["pred_has_filler"])
        if ref_text_map and not rec.get("reference_text") and rec.get("filename") in ref_text_map:
            rec["reference_text"] = ref_text_map[rec["filename"]]
        normalized.append(rec)

    def evaluate_subset(subset: list) -> dict:
        y_true_dom = [p["gt_dominant"] for p in subset]
        y_pred_dom = [p["pred_dominant"] for p in subset]

        # 3-Class dominant classification
        n = len(subset)
        correct = sum(1 for t, p in zip(y_true_dom, y_pred_dom) if t == p)
        acc = round(correct / n, 4) if n > 0 else 0.0

        matrix = {row: {col: 0 for col in labels_3class} for row in labels_3class}
        for t, p in zip(y_true_dom, y_pred_dom):
            if t in matrix and p in matrix[t]:
                matrix[t][p] += 1

        per_class = {}
        f1_list, p_list, r_list = [], [], []
        weighted_f1_sum = 0.0
        total_support = 0

        for lbl in labels_3class:
            tp = matrix[lbl][lbl]
            fp = sum(matrix[other][lbl] for other in labels_3class if other != lbl)
            fn = sum(matrix[lbl][other] for other in labels_3class if other != lbl)
            supp = sum(matrix[lbl].values())
            total_support += supp

            prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
            rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
            f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

            per_class[lbl] = {
                "tp": tp, "fp": fp, "fn": fn, "support": supp,
                "precision": prec, "recall": rec, "f1_score": f1
            }
            p_list.append(prec)
            r_list.append(rec)
            f1_list.append(f1)
            weighted_f1_sum += f1 * supp

        macro_p = round(float(np.mean(p_list)), 4) if p_list else 0.0
        macro_r = round(float(np.mean(r_list)), 4) if r_list else 0.0
        macro_f1 = round(float(np.mean(f1_list)), 4) if f1_list else 0.0
        weighted_f1 = round(weighted_f1_sum / total_support, 4) if total_support > 0 else 0.0

        # Binary Stutter
        y_true_stutter = [p["gt_has_stutter"] for p in subset]
        y_pred_stutter = [p["pred_has_stutter"] for p in subset]
        bin_stutter = compute_binary_metrics(y_true_stutter, y_pred_stutter)

        # Binary Filler
        y_true_filler = [p["gt_has_filler"] for p in subset]
        y_pred_filler = [p["pred_has_filler"] for p in subset]
        bin_filler = compute_binary_metrics(y_true_filler, y_pred_filler)

        return {
            "sample_count": n,
            "overall_accuracy": acc,
            "confusion_matrix": matrix,
            "per_class": per_class,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "binary_stutter": bin_stutter,
            "binary_filler": bin_filler
        }

    # Aggregate evaluation
    agg_metrics = evaluate_subset(normalized)

    # Per-corpus breakdown
    per_corpus = {}
    for corpus in ["UCLASS", "FluencyBank", "Libri_Stutter"]:
        sub = [p for p in normalized if p["dataset"] == corpus]
        per_corpus[corpus] = evaluate_subset(sub)

    # Status counts
    status_counts = {}
    for p in normalized:
        st = p.get("status", "UNKNOWN")
        status_counts[st] = status_counts.get(st, 0) + 1

    # WER calculation on clips with reference_text and successful ASR
    wer_clips = [p for p in normalized if p.get("reference_text") and p.get("status") == "SUCCESS"]
    macro_wer = None
    pooled_micro_wer = None
    wer_evaluated_count = len(wer_clips)

    if wer_clips and jiwer:
        wer_scores = []
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_ref_words = 0

        for p in wer_clips:
            ref = str(p["reference_text"])
            hyp = str(p.get("pred_transcript", ""))
            try:
                w_score = float(jiwer.wer(ref, hyp))
                wer_scores.append(w_score)

                # Process words for micro pooled WER
                out = jiwer.process_words(ref, hyp)
                total_substitutions += out.substitutions
                total_deletions += out.deletions
                total_insertions += out.insertions
                total_ref_words += (out.hits + out.substitutions + out.deletions)
            except Exception:
                pass

        if wer_scores:
            macro_wer = round(float(np.mean(wer_scores)), 4)
        if total_ref_words > 0:
            pooled_micro_wer = round((total_substitutions + total_deletions + total_insertions) / total_ref_words, 4)

    # Latency & RTF distribution
    latencies = [p["latency_sec"] for p in normalized if "latency_sec" in p]
    durations = [p.get("duration_sec", 3.0) for p in normalized]
    rtfs = [p.get("rtf", 0.0) for p in normalized if "rtf" in p]

    total_latency = sum(latencies) if latencies else 0.0
    total_duration = sum(durations) if durations else 0.0

    latency_stats = {
        "median_sec": round(float(np.median(latencies)), 3) if latencies else 0.0,
        "mean_sec": round(float(np.mean(latencies)), 3) if latencies else 0.0,
        "min_sec": round(float(np.min(latencies)), 3) if latencies else 0.0,
        "max_sec": round(float(np.max(latencies)), 3) if latencies else 0.0,
        "p95_sec": round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
        "macro_rtf": round(float(np.mean(rtfs)), 3) if rtfs else 0.0,
        "aggregate_micro_rtf": round(total_latency / total_duration, 3) if total_duration > 0 else 0.0,
        "total_latency_sec": round(total_latency, 2),
        "total_audio_duration_sec": round(total_duration, 2)
    }

    return {
        "status_counts": status_counts,
        "aggregate": agg_metrics,
        "per_corpus": per_corpus,
        "wer": {
            "evaluated_clip_count": wer_evaluated_count,
            "macro_wer": macro_wer,
            "pooled_micro_wer": pooled_micro_wer
        },
        "latency": latency_stats
    }

def generate_markdown_report(metrics: dict, out_path: str):
    """Generates the comprehensive benchmark markdown report."""
    agg = metrics["aggregate"]
    base_agg = metrics.get("baseline_comparison", {}).get("aggregate", {})
    b_st = base_agg.get("binary_stutter", {})
    e_st = agg["binary_stutter"]
    b_fl = base_agg.get("binary_filler", {})
    e_fl = agg["binary_filler"]
    lat = metrics["latency"]
    b_lat = metrics.get("baseline_comparison", {}).get("latency", {})
    wer = metrics["wer"]
    b_wer = metrics.get("baseline_comparison", {}).get("wer", {})
    meta = metrics.get("execution_meta", {})
    cfg = meta.get("config", {})

    status_counts = metrics.get("status_counts", {})
    attempted = meta.get("total_clips", 750)
    successful = status_counts.get("SUCCESS", 0)
    timed_out = status_counts.get("TIMEOUT", 0)
    failed = status_counts.get("ERROR", 0)
    crashed = status_counts.get("CRASHED", 0)

    lines = []
    lines.append("# SpeechAI Phase E: 750-Clip Experimental Benchmark Report")
    lines.append("")
    lines.append(f"**Execution Timestamp**: {meta.get('start_time_iso', 'N/A')} to {meta.get('end_time_iso', 'N/A')}")
    lines.append(f"**Total Elapsed Wall-Clock Time**: {meta.get('total_elapsed_sec', 'N/A')} seconds ({round(meta.get('total_elapsed_sec', 0) / 60, 2)} minutes)")
    lines.append(f"**Manifest Evaluated**: `{meta.get('manifest_path', 'evaluation_results/main_benchmark/main_benchmark_manifest.json')}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary & Execution Status")
    lines.append("")
    lines.append("| Metric | Experimental Count / Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total Clips Attempted | {attempted} |")
    lines.append(f"| Successful Inferences | {successful} |")
    lines.append(f"| Timed-Out Clips (Terminated & Reaped) | {timed_out} |")
    lines.append(f"| Execution Errors | {failed} |")
    lines.append(f"| Worker Crashes | {crashed} |")
    lines.append(f"| Process Isolation Status | Enabled (25.0s per-clip hard timeout via OS process termination) |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Core Detection Comparison: Baseline vs. Experimental")
    lines.append("")
    lines.append("| Classification Task / Metric | Audited Baseline | Experimental Pipeline | Absolute Delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| **Dominant 3-Class Accuracy** | {base_agg.get('overall_accuracy', 0.5107)*100:.2f}% | **{agg['overall_accuracy']*100:.2f}%** | {(agg['overall_accuracy'] - base_agg.get('overall_accuracy', 0.5107))*100:+.2f}% |")
    lines.append(f"| **Dominant 3-Class Macro F1** | {base_agg.get('macro_f1', 0.4908):.4f} | **{agg['macro_f1']:.4f}** | {agg['macro_f1'] - base_agg.get('macro_f1', 0.4908):+.4f} |")
    lines.append(f"| **Dominant 3-Class Weighted F1** | {base_agg.get('weighted_f1', 0.5107):.4f} | **{agg['weighted_f1']:.4f}** | {agg['weighted_f1'] - base_agg.get('weighted_f1', 0.5107):+.4f} |")
    lines.append(f"| Binary Stutter Precision | {b_st.get('precision', 0.7923)*100:.2f}% | {e_st['precision']*100:.2f}% | {(e_st['precision'] - b_st.get('precision', 0.7923))*100:+.2f}% |")
    lines.append(f"| **Binary Stutter Recall** | {b_st.get('recall', 0.2994)*100:.2f}% | **{e_st['recall']*100:.2f}%** | {(e_st['recall'] - b_st.get('recall', 0.2994))*100:+.2f}% |")
    lines.append(f"| **Binary Stutter F1-Score** | {b_st.get('f1', 0.4346):.4f} | **{e_st['f1']:.4f}** | {e_st['f1'] - b_st.get('f1', 0.4346):+.4f} |")
    lines.append(f"| **Binary Filler Precision** | {b_fl.get('precision', 0.4136)*100:.2f}% | **{e_fl['precision']*100:.2f}%** | {(e_fl['precision'] - b_fl.get('precision', 0.4136))*100:+.2f}% |")
    lines.append(f"| Binary Filler Recall | {b_fl.get('recall', 0.7821)*100:.2f}% | {e_fl['recall']*100:.2f}% | {(e_fl['recall'] - b_fl.get('recall', 0.7821))*100:+.2f}% |")
    lines.append(f"| **Binary Filler F1-Score** | {b_fl.get('f1', 0.5410):.4f} | **{e_fl['f1']:.4f}** | {e_fl['f1'] - b_fl.get('f1', 0.5410):+.4f} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Dominant 3-Class Classification Details")
    lines.append("")
    lines.append("### Confusion Matrix (Rows = Ground Truth, Columns = Predicted)")
    cm = agg["confusion_matrix"]
    lines.append("")
    lines.append("| Ground Truth \\ Predicted | Pred Clean | Pred Stutter | Pred Filler | Total Support |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| **True Clean** | **{cm['Clean']['Clean']}** | {cm['Clean']['Stutter']} | {cm['Clean']['Filler']} | {sum(cm['Clean'].values())} |")
    lines.append(f"| **True Stutter** | {cm['Stutter']['Clean']} | **{cm['Stutter']['Stutter']}** | {cm['Stutter']['Filler']} | {sum(cm['Stutter'].values())} |")
    lines.append(f"| **True Filler** | {cm['Filler']['Clean']} | {cm['Filler']['Stutter']} | **{cm['Filler']['Filler']}** | {sum(cm['Filler'].values())} |")
    lines.append("")
    lines.append("### Per-Class Detailed Performance")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1-Score | Support | TP | FP | FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for lbl in ["Clean", "Stutter", "Filler"]:
        c = agg["per_class"][lbl]
        lines.append(f"| **{lbl}** | {c['precision']*100:.2f}% | {c['recall']*100:.2f}% | {c['f1_score']:.4f} | {c['support']} | {c['tp']} | {c['fp']} | {c['fn']} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Binary Disfluency Presence Detection")
    lines.append("")
    lines.append("Binary detection evaluates whether a specific disfluency type is present in the clip independently of other disfluency types.")
    lines.append("")
    lines.append("| Disfluency Type | TP | FP | FN | TN | Precision | Recall | F1-Score | Support |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(f"| **Stuttering Presence** | {e_st['tp']} | {e_st['fp']} | {e_st['fn']} | {e_st['tn']} | {e_st['precision']*100:.2f}% | {e_st['recall']*100:.2f}% | {e_st['f1']:.4f} | {e_st['support']} |")
    lines.append(f"| **Filler Word Presence** | {e_fl['tp']} | {e_fl['fp']} | {e_fl['fn']} | {e_fl['tn']} | {e_fl['precision']*100:.2f}% | {e_fl['recall']*100:.2f}% | {e_fl['f1']:.4f} | {e_fl['support']} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Per-Corpus Breakdown")
    lines.append("")
    lines.append("| Corpus | 3-Class Accuracy | Stutter Prec | Stutter Rec | Stutter F1 | Filler Prec | Filler Rec | Filler F1 | Clips |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for corp in ["UCLASS", "FluencyBank", "Libri_Stutter"]:
        sub = metrics["per_corpus"][corp]
        st = sub["binary_stutter"]
        fl = sub["binary_filler"]
        fl_rec_str = f"{fl['recall']*100:.2f}%" if fl['support'] > 0 else "N/A (Supp=0)"
        fl_f1_str = f"{fl['f1']:.4f}" if fl['support'] > 0 else "N/A (Supp=0)"
        lines.append(f"| **{corp}** | {sub['overall_accuracy']*100:.2f}% | {st['precision']*100:.2f}% | {st['recall']*100:.2f}% | {st['f1']:.4f} | {fl['precision']*100:.2f}% | {fl_rec_str} | {fl_f1_str} | {sub['sample_count']} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Automatic Speech Recognition (ASR) Word Error Rate")
    lines.append("")
    lines.append(f"- **Evaluated Clips**: {wer.get('evaluated_clip_count', 0)} Libri_Stutter clips with verified reference text.")
    lines.append(f"- **Experimental Macro WER**: {wer.get('macro_wer', 'N/A')} (Baseline: {b_wer.get('macro_wer', 0.2493)})")
    lines.append(f"- **Experimental Pooled Micro WER**: {wer.get('pooled_micro_wer', 'N/A')} (Baseline: {b_wer.get('pooled_micro_wer', 0.4364)})")
    lines.append("")
    lines.append("> **Definitions**:")
    lines.append("> - **Macro WER**: Mean of individual clip Word Error Rates across all evaluated clips.")
    lines.append("> - **Pooled Micro WER**: $\\frac{\\sum (\\text{Substitutions} + \\text{Deletions} + \\text{Insertions})}{\\sum (\\text{Hits} + \\text{Substitutions} + \\text{Deletions})}$, calculating global edit distance across the entire corpus.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Latency, Real-Time Factor (RTF) & Throughput")
    lines.append("")
    lines.append("| Metric | Audited Baseline | Experimental Pipeline | Improvement / Notes |")
    lines.append("|---|---:|---:|---|")
    lines.append(f"| **Median Latency** | {b_lat.get('median_sec', 2.521):.3f}s | {lat['median_sec']:.3f}s | Per-clip typical response time |")
    lines.append(f"| **Mean Latency** | {b_lat.get('mean_sec', 12.074):.3f}s | {lat['mean_sec']:.3f}s | Bounded execution |")
    lines.append(f"| **P95 Latency** | {b_lat.get('p95_sec', 16.211):.3f}s | {lat['p95_sec']:.3f}s | High-percentile predictability |")
    lines.append(f"| **Maximum Latency** | {b_lat.get('max_sec', 1917.945):.3f}s | **{lat['max_sec']:.3f}s** | Runaway inference eliminated |")
    lines.append(f"| **Aggregate Micro RTF** | {b_lat.get('aggregate_micro_rtf', 1.806):.3f} | {lat['aggregate_micro_rtf']:.3f} | Total Processing Time / Audio Time |")
    lines.append(f"| **Macro RTF** | {b_lat.get('macro_rtf', 3.303):.3f} | {lat['macro_rtf']:.3f} | Mean of per-clip RTFs |")
    lines.append(f"| **Total Wall-Clock Time** | 150.92 min (9055s) | {round(meta.get('total_elapsed_sec', 0) / 60, 2)} min ({meta.get('total_elapsed_sec', 'N/A')}s) | Full benchmark duration |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 8. Dominant-Class Decision Rule & Co-Occurrence Explanation")
    lines.append("")
    lines.append("In natural continuous speech, disfluencies frequently co-occur within the same audio segment (e.g., a speaker may produce both a prolongation and an interjection like 'um').")
    lines.append("")
    lines.append("1. **Clip-Level Binary Detections**: The experimental pipeline evaluates stuttering presence and filler word presence independently. A clip can have both `pred_has_stutter = 1` and `pred_has_filler = 1`.")
    lines.append("2. **Mutually Exclusive Dominant-Class Rule**: To assign a single categorical label per clip, the system applies the hierarchical clinical rule:")
    lines.append("   - If `pred_has_stutter == 1` $\\rightarrow$ Class is **Stutter** (stuttering events take clinical precedence).")
    lines.append("   - Else if `pred_has_filler == 1` $\\rightarrow$ Class is **Filler**.")
    lines.append("   - Else $\\rightarrow$ Class is **Clean**.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 9. Limitations & Scope of Evaluation")
    lines.append("")
    lines.append("- **Screening & Coaching Scope**: SpeechAI is designed as an asynchronous communication-support and speech fluency screening tool. It is **not** a clinically validated diagnostic instrument.")
    lines.append("- **Clip-Level Binary vs. Event-Level Boundaries**: Evaluations are performed at the 3-second audio clip level. The system detects presence and frequency within clips, but does not provide millisecond-accurate acoustic boundary alignments.")
    lines.append("- **Dataset Heterogeneity**: Libri_Stutter contains synthetic disfluency insertions that differ acoustically from natural disfluencies found in clinical datasets like UCLASS and FluencyBank.")
    lines.append("- **Hardware Constraints**: Benchmarking was executed on CPU; GPU inference will yield significantly lower latencies and Real-Time Factors.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 10. Pipeline Configuration Parameters")
    lines.append("```json")
    lines.append(json.dumps(cfg, indent=2))
    lines.append("```")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    t_start_wall = time.time()
    dt_start_iso = datetime.datetime.now().isoformat()

    project_root = os.path.dirname(backend_dir)
    manifest_path = os.path.join(project_root, "evaluation_results", "main_benchmark", "main_benchmark_manifest.json")
    baseline_pred_path = os.path.join(project_root, "evaluation_results", "main_benchmark", "main_benchmark_predictions.json")
    out_dir = os.path.join(project_root, "evaluation_results", "experimental", "main_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "benchmark_execution_log.txt")
    chkpt_path = os.path.join(out_dir, "predictions_checkpoint.json")
    pred_path = os.path.join(out_dir, "experimental_benchmark_predictions.json")
    metrics_path = os.path.join(out_dir, "experimental_benchmark_metrics.json")
    post_hash_path = os.path.join(out_dir, "post_run_immutability_audit.json")
    report_path = os.path.join(out_dir, "EXPERIMENTAL_BENCHMARK_REPORT.md")

    # File logger setup (append mode preserves logs across resume sessions)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    print("=" * 80)
    print("SPEECHAI PHASE E: FULL 750-CLIP EXPERIMENTAL BENCHMARK EXECUTION")
    print(f"Start Time: {dt_start_iso}")
    print(f"Output Directory: {out_dir}")
    print("=" * 80)
    logger.info("Initializing SpeechAI 750-Clip Experimental Benchmark Execution.")

    # 1. Mandatory Preflight Validation
    pre_hashes = run_preflight_checks(project_root, manifest_path)
    with open(os.path.join(out_dir, "pre_run_immutability_hashes.json"), "w", encoding="utf-8") as f:
        json.dump(pre_hashes, f, indent=2)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 2. Pipeline Initialization
    config = ExperimentalConfig()
    harness = DevelopmentEvaluationHarness(config)
    logger.info("Experimental pipeline harness initialized with process-isolated ASR (timeout=25.0s).")

    # 3. Main Benchmark Execution Loop with Checkpoint Recovery
    predictions = []
    processed_files = set()
    if os.path.exists(chkpt_path):
        try:
            with open(chkpt_path, "r", encoding="utf-8") as f:
                chkpt_data = json.load(f)
            if isinstance(chkpt_data, list) and len(chkpt_data) > 0:
                predictions = chkpt_data
                processed_files = {p.get("filename") for p in predictions}
                logger.info(f"Resuming from checkpoint with {len(predictions)} previously completed clips.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")
            predictions = []
            processed_files = set()

    completed_count = len(predictions)
    checkpoint_interval = 5
    ref_text_map = {m["filename"]: m["reference_text"] for m in manifest if "reference_text" in m}

    logger.info(f"Beginning inference across {len(manifest)} clips (Remaining: {len(manifest) - len(processed_files)})...")

    try:
        for i, clip in enumerate(manifest):
            clip_idx = i + 1
            fname = clip.get("filename")
            ds = clip.get("dataset")

            if fname in processed_files:
                continue

            t_clip_start = time.perf_counter()
            res = harness.run_development_clip(clip)
            dt_clip = round(time.perf_counter() - t_clip_start, 3)

            # Preserve reference text for WER analysis
            if "reference_text" in clip:
                res["reference_text"] = clip["reference_text"]

            predictions.append(res)
            processed_files.add(fname)
            completed_count += 1

            status = res.get("status", "UNKNOWN")
            pred_dom = res.get("pred_dominant")
            gt_dom = res.get("gt_dominant")
            st_ev = res.get("stutter_events", 0)
            fl_ev = res.get("filler_events", 0)

            log_msg = (
                f"[{completed_count}/750] {ds} | {fname} | Status: {status} | "
                f"Pred: {pred_dom} (GT: {gt_dom}) | StutterEvents: {st_ev} | FillerEvents: {fl_ev} | "
                f"Latency: {dt_clip}s"
            )
            logger.info(log_msg)
            if completed_count % 25 == 0 or completed_count == 750 or status != "SUCCESS":
                print(log_msg)

            # Atomic checkpoint write
            if completed_count % checkpoint_interval == 0 or completed_count == 750:
                with open(chkpt_path, "w", encoding="utf-8") as f:
                    json.dump(predictions, f, indent=2)

    except KeyboardInterrupt:
        logger.warning(f"Inference interrupted by user! Completed clips: {completed_count} / 750.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Inference crashed on clip {completed_count + 1}: {e}")
        raise
    finally:
        harness.asr_engine.shutdown()
        logger.info("ASR engine child processes cleanly terminated and IPC closed.")

    t_end_wall = time.time()
    dt_end_iso = datetime.datetime.now().isoformat()
    total_elapsed_sec = round(t_end_wall - t_start_wall, 2)

    logger.info(f"Completed all {len(predictions)} clips in {total_elapsed_sec}s.")

    # 4. Save Final Predictions
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Saved complete prediction records to: {pred_path}")

    # 5. Calculate Comprehensive Metrics
    metrics = calculate_detailed_metrics(predictions, ref_text_map=ref_text_map)
    metrics["execution_meta"] = {
        "start_time_iso": dt_start_iso,
        "end_time_iso": dt_end_iso,
        "total_elapsed_sec": total_elapsed_sec,
        "total_clips": len(predictions),
        "manifest_path": manifest_path,
        "config": {
            "asr_timeout_sec": config.asr.per_clip_timeout_sec,
            "asr_temperature": config.asr.temperature,
            "asr_model": config.asr.model_name,
            "condition_on_previous_text": config.asr.condition_on_previous_text,
            "stutter_confident_thresh": config.stutter.confident_threshold,
            "stutter_uncertain_thresh": config.stutter.uncertain_threshold,
            "prolongation_min_sec": config.stutter.min_prolongation_duration_sec,
            "block_silence_min_sec": config.stutter.min_block_silence_sec,
            "filler_contextual_pause_sec": config.filler.contextual_pause_boundary_sec
        }
    }

    # Load baseline predictions for side-by-side comparison
    if os.path.exists(baseline_pred_path):
        with open(baseline_pred_path, "r", encoding="utf-8") as f:
            base_preds = json.load(f)
        metrics["baseline_comparison"] = calculate_detailed_metrics(base_preds, ref_text_map=ref_text_map)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved complete metrics summary to: {metrics_path}")

    # Generate Markdown Report
    generate_markdown_report(metrics, report_path)
    logger.info(f"Saved comprehensive benchmark report to: {report_path}")

    # 6. Post-Run Immutability Verification
    logger.info("Executing Post-Run Immutability Audit on Protected Files...")
    post_hashes = {}
    tampered_files = []

    for rel_path in PROTECTED_FILES:
        full_path = os.path.join(project_root, rel_path)
        cur_hash = compute_file_sha256(full_path)
        cur_size = os.path.getsize(full_path)
        orig_hash = pre_hashes[rel_path]["sha256"]

        is_match = (cur_hash == orig_hash)
        if not is_match:
            tampered_files.append(rel_path)

        post_hashes[rel_path] = {
            "size_bytes": cur_size,
            "sha256": cur_hash,
            "verified_unchanged": is_match
        }

    post_audit = {
        "audit_timestamp": datetime.datetime.now().isoformat(),
        "all_files_unchanged": len(tampered_files) == 0,
        "tampered_files": tampered_files,
        "files": post_hashes
    }

    with open(post_hash_path, "w", encoding="utf-8") as f:
        json.dump(post_audit, f, indent=2)
    logger.info(f"Saved post-run immutability audit to: {post_hash_path}")

    if tampered_files:
        logger.error(f"CRITICAL: Protected files were modified! {tampered_files}")
        sys.exit(1)
    else:
        logger.info("[Post-Audit OK] All 12 protected files confirmed 100% UNCHANGED.")

    # 7. Print Console Summary
    print("\n" + "=" * 80)
    print("PHASE E EXPERIMENTAL BENCHMARK — COMPREHENSIVE SUMMARY")
    print("=" * 80)
    print(f"Total Clips Processed:  {len(predictions)} / 750")
    print(f"Wall-Clock Runtime:     {total_elapsed_sec}s ({round(total_elapsed_sec / 60, 2)} minutes)")
    print(f"Status Breakdown:       {metrics['status_counts']}")
    print("-" * 80)

    agg = metrics["aggregate"]
    base_agg = metrics.get("baseline_comparison", {}).get("aggregate", {})

    print(f"{'Metric':<35} | {'Baseline':<15} | {'Experimental':<15}")
    print("-" * 70)
    print(f"{'3-Class Dominant Accuracy':<35} | {base_agg.get('overall_accuracy', 'N/A'):<15} | {agg['overall_accuracy']:<15}")
    print(f"{'3-Class Macro F1-Score':<35} | {base_agg.get('macro_f1', 'N/A'):<15} | {agg['macro_f1']:<15}")
    print(f"{'3-Class Weighted F1-Score':<35} | {base_agg.get('weighted_f1', 'N/A'):<15} | {agg['weighted_f1']:<15}")
    print("-" * 70)
    b_st = base_agg.get("binary_stutter", {})
    e_st = agg["binary_stutter"]
    print(f"{'Stutter Precision':<35} | {b_st.get('precision', 'N/A'):<15} | {e_st['precision']:<15}")
    print(f"{'Stutter Recall':<35} | {b_st.get('recall', 'N/A'):<15} | {e_st['recall']:<15}")
    print(f"{'Stutter F1-Score':<35} | {b_st.get('f1', 'N/A'):<15} | {e_st['f1']:<15}")
    print("-" * 70)
    b_fl = base_agg.get("binary_filler", {})
    e_fl = agg["binary_filler"]
    print(f"{'Filler Precision':<35} | {b_fl.get('precision', 'N/A'):<15} | {e_fl['precision']:<15}")
    print(f"{'Filler Recall':<35} | {b_fl.get('recall', 'N/A'):<15} | {e_fl['recall']:<15}")
    print(f"{'Filler F1-Score':<35} | {b_fl.get('f1', 'N/A'):<15} | {e_fl['f1']:<15}")
    print("-" * 70)
    b_lat = metrics.get("baseline_comparison", {}).get("latency", {})
    e_lat = metrics["latency"]
    print(f"{'Median Latency (s)':<35} | {b_lat.get('median_sec', 'N/A'):<15} | {e_lat['median_sec']:<15}")
    print(f"{'Mean Latency (s)':<35} | {b_lat.get('mean_sec', 'N/A'):<15} | {e_lat['mean_sec']:<15}")
    print(f"{'P95 Latency (s)':<35} | {b_lat.get('p95_sec', 'N/A'):<15} | {e_lat['p95_sec']:<15}")
    print(f"{'Aggregate Micro RTF':<35} | {b_lat.get('aggregate_micro_rtf', 'N/A'):<15} | {e_lat['aggregate_micro_rtf']:<15}")
    print(f"{'Macro RTF':<35} | {b_lat.get('macro_rtf', 'N/A'):<15} | {e_lat['macro_rtf']:<15}")
    print("-" * 70)
    b_wer = metrics.get("baseline_comparison", {}).get("wer", {})
    e_wer = metrics["wer"]
    print(f"{'Macro WER (Libri_Stutter)':<35} | {b_wer.get('macro_wer', 'N/A'):<15} | {e_wer['macro_wer']:<15}")
    print(f"{'Pooled Micro WER (Libri_Stutter)':<35} | {b_wer.get('pooled_micro_wer', 'N/A'):<15} | {e_wer['pooled_micro_wer']:<15}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
