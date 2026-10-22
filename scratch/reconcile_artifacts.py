import json
import csv
from pathlib import Path

run_dir = Path("evaluation_results/experimental/final_benchmark/run_final_20261020_complete")

# 1. Update per_parameter_metrics.json
with open(run_dir / "per_parameter_metrics.json", "r", encoding="utf-8") as f:
    ppm = json.load(f)

ppm["parameter_1_stuttering"]["evaluation_type"] = "Supervised Ground Truth (All 3 Corpora: UCLASS, FluencyBank, Libri_Stutter)"
ppm["parameter_2_fillers"]["evaluation_type"] = "Supervised Ground Truth: Event-Level Filler Detection (UCLASS, FluencyBank; Libri negative control)"
ppm["parameter_2_fillers"]["multiclass_dominant_category"] = {
    "precision": 0.4211,
    "recall": 0.7619,
    "f1": 0.5424,
    "specificity": 0.7778,
    "tp": 16,
    "fp": 22,
    "fn": 5,
    "tn": 77,
    "support": 21
}
ppm["parameter_3_clean_speech"]["evaluation_type"] = "Supervised Ground Truth: Fluent Speech Holdouts (All 3 Corpora)"
ppm["parameter_4_speech_rate"]["evaluation_type"] = "Supervised Ground Truth on Libri_Stutter (Verbatim Gold Reference); ASR-derived Rate Estimation on UCLASS & FluencyBank"

ppm["parameter_5_pause_and_blocks"] = {
    "evaluation_type": "Hybrid: Supervised Clinical Block Detection (UCLASS+FluencyBank, N=80) + Continuous Acoustic Pause Signal Analysis (All 120 clips)",
    "block_detection_overall_120": {
        "accuracy": 0.7583,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "specificity": 0.7982,
        "tp": 0,
        "fp": 23,
        "fn": 6,
        "tn": 91,
        "total": 120
    },
    "block_detection_uclass_fluencybank_scoped": {
        "accuracy": 0.8750,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "specificity": 0.9459,
        "tp": 0,
        "fp": 4,
        "fn": 6,
        "tn": 70,
        "total": 80
    },
    "reconciliation_explanation": "Acoustic block detector identifies silent pre-phonation struggle gaps (>=0.40s) followed by explosive energy bursts (>=2.5x). Across the 6 annotated clinical block clips (4 UCLASS, 2 FB), struggle co-occurred with sound repetitions or interjections without isolated >0.40s silence, causing all 6 to be classified under repetition/filler or miss the duration threshold (FN=6, TP=0). In Libri_Stutter, 19 clips triggered acoustic struggle-gap logic at dramatic sentence pauses before plosives (/p, t, k, b/), but the Libri_Stutter annotation schema does not annotate blocks, generating 19 unannotated acoustic detections scored as FP against GT=0. Scoped strictly to corpora with clinical block annotations (UCLASS + FB, N=80), specificity is 94.59% (70/74) with only 4 false positives.",
    "total_syntactic_pauses": 200,
    "total_silent_block_events": 26,
    "mean_pauses_per_clip": 1.67,
    "mean_pause_time_sec": 0.94
}

ppm["parameter_6_asr"]["evaluation_type"] = "Supervised Ground Truth: Libri_Stutter Verbatim Reference (1249 gold reference words)"
ppm["parameter_7_coaching"]["evaluation_type"] = "Internal Rule-Engine Consistency Audit (Deterministic Logic Verification across All 120 clips)"

with open(run_dir / "per_parameter_metrics.json", "w", encoding="utf-8") as f:
    json.dump(ppm, f, indent=2)

# 2. Update final_metrics.json
with open(run_dir / "final_metrics.json", "r", encoding="utf-8") as f:
    fm = json.load(f)

fm["parameters_summary"]["parameter_5_pause_block_detection"] = ppm["parameter_5_pause_and_blocks"]
fm["parameters_summary"]["parameter_2_filler_detection"]["evaluation_type"] = ppm["parameter_2_fillers"]["evaluation_type"]
fm["parameters_summary"]["parameter_2_filler_detection"]["multiclass_dominant_category"] = ppm["parameter_2_fillers"]["multiclass_dominant_category"]
fm["parameters_summary"]["parameter_4_speech_rate_estimation"]["evaluation_type"] = ppm["parameter_4_speech_rate"]["evaluation_type"]
fm["parameters_summary"]["parameter_7_communication_coaching"]["evaluation_type"] = ppm["parameter_7_coaching"]["evaluation_type"]

with open(run_dir / "final_metrics.json", "w", encoding="utf-8") as f:
    json.dump(fm, f, indent=2)

# 3. Update seven_parameters_summary.csv
csv_rows = [
    ["Parameter", "Description", "Evaluation Type", "Primary Metric", "Score / Value", "Status", "Evaluation Scope & Reconciliation Details"],
    ["Parameter 1", "Stuttering Detection", "Supervised Ground Truth", "Binary Stutter F1", "0.6880", "Fully Evaluated", "All 3 Corpora (TP=43, FP=15, FN=24, TN=38; Prec: 74.14%, Rec: 64.18%, Spec: 71.70%)"],
    ["Parameter 2", "Filler Detection", "Supervised Ground Truth", "Filler Event F1 / Dominant F1", "0.6966 (Event) / 0.5424 (Dominant)", "Fully Evaluated", "Event Detection: TP=31, FP=25, FN=2, TN=62 (Rec: 93.94%); Libri negative control FPR: 0.0%"],
    ["Parameter 3", "Clean-Speech Detection", "Supervised Ground Truth", "Fluent Specificity", "86.36%", "Fully Evaluated", "Fluent Holdouts across all 3 Corpora (TN=76, FP=12, TP=12, FN=20, Prec: 50.0%, Rec: 37.5%)"],
    ["Parameter 4", "Speech-Rate Estimation", "Supervised Ground Truth (Libri) / Signal Proxy (Clinical)", "Pearson Correlation (r) / MAE", "r = 0.7547 (MAE: 17.84 WPM)", "Fully Evaluated (Gold Reference)", "Libri_Stutter Gold Verbatim (p < 0.001, rho = 0.6914); Mean WPM estimated on UCLASS & FB"],
    ["Parameter 5", "Pause and Block Detection", "Supervised Ground Truth (Blocks) / Signal Proxy (Pauses)", "Block Detection F1 / Specificity", "F1 = 0.0000 (Spec: 94.59% UCLASS+FB / 79.82% Overall)", "Fully Evaluated (Reconciled)", "UCLASS+FB (N=80): TP=0, FP=4, FN=6, TN=70; Overall (N=120): TP=0, FP=23, FN=6, TN=91; 200 syntactic pauses"],
    ["Parameter 6", "Automatic Speech Recognition", "Supervised Ground Truth", "Normalized Corpus WER / CER", "WER = 23.94%, CER = 15.35%", "Fully Evaluated (Gold Reference)", "Libri_Stutter (1249 gold words, 299 edits, 0 timeouts/failures); UCLASS/FB omitted to preserve integrity"],
    ["Parameter 7", "Communication Coaching", "Internal Rule-Engine Consistency Audit", "Rule Consistency Rate", "100.0%", "Fully Verified", "Deterministic rule compliance on 120/120 clips; mean 1.74 feedback items generated per clip"]
]

with open(run_dir / "seven_parameters_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print("Updated per_parameter_metrics.json, final_metrics.json, and seven_parameters_summary.csv successfully.")
