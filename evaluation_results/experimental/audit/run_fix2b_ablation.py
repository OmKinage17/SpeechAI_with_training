"""
Script: run_fix2b_ablation.py
=============================
Executes the controlled factorial ablation for SpeechAI Fix 2B across the
30-clip development set, evaluating exactly:
1. Fix 1 configuration (Baseline for Fix 2)
2. Complete Fix 2 configuration
3. Variant 3: Fix 2 with ONLY duration reverted to 0.32s
4. Variant 4: Fix 2 with ONLY standalone confidence cap / corroboration relaxed (0.70 / uncorroborated)
5. Variant 5: Fix 2 with ONLY acoustic rigidity relaxed (pitch std 2.0 st, spectral flux 0.35)
Supplementary:
6. Variant 3B: Intermediate duration (0.40s) with Fix 2 rigidity & corroboration
7. Variant 4B: Standalone severe prolongation (duration >= 0.60s) allowed to qualify

Computes side-by-side:
- 3-class dominant accuracy & macro F1
- Stutter precision, recall, F1, TP, FP, FN, TN
- Filler precision, recall, F1, TP, FP, FN, TN
- False positive counts by dataset
- Genuine stutter events/clips recovered vs lost compared with Fix 1
- Detailed diagnostic breakdown of missed clips (including 103-1240-0002.flac)
"""

import os
import sys
import json
# Add project root and backend to path
audit_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(audit_dir)))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.experimental.config import ExperimentalConfig, StutterDetectorConfig
from backend.experimental.acoustic_features import AcousticFeatureExtractor
from backend.experimental.eval_harness import compute_binary_metrics, compute_multiclass_metrics, get_dominant_label

# Load verified dev eval results
dev_json_path = os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips_refined.json")
with open(dev_json_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

predictions_cache = dev_data["predictions"]
baseline_metrics = dev_data["baseline"]

# Initialize acoustic extractor
ac = AcousticFeatureExtractor(ExperimentalConfig().acoustic)
sr = ac.config.sample_rate

# Define the Ablation Variants
VARIANTS = {
    "1. Fix 1 (Base)": {
        "min_duration_sec": 0.32,
        "max_pitch_std_st": 2.0,
        "max_spectral_flux": 0.35,
        "standalone_confidence": 0.70,
        "require_corroboration": False,
        "desc": "Fix 1 baseline (dur=0.32s, pitch=2.0st, flux=0.35, standalone conf=0.70, no corroboration required)"
    },
    "2. Complete Fix 2": {
        "min_duration_sec": 0.50,
        "max_pitch_std_st": 1.2,
        "max_spectral_flux": 0.20,
        "standalone_confidence": 0.50,
        "require_corroboration": True,
        "desc": "Complete Fix 2 (dur=0.50s, pitch=1.2st, flux=0.20, standalone conf=0.50, corroboration required)"
    },
    "3. Revert Duration (0.32s)": {
        "min_duration_sec": 0.32,
        "max_pitch_std_st": 1.2,
        "max_spectral_flux": 0.20,
        "standalone_confidence": 0.50,
        "require_corroboration": True,
        "desc": "Fix 2 with ONLY duration reverted to 0.32s (pitch=1.2st, flux=0.20, corroboration required)"
    },
    "4. Relax Standalone Conf (0.70)": {
        "min_duration_sec": 0.50,
        "max_pitch_std_st": 1.2,
        "max_spectral_flux": 0.20,
        "standalone_confidence": 0.70,
        "require_corroboration": False,
        "desc": "Fix 2 with ONLY standalone conf cap relaxed to 0.70 (dur=0.50s, pitch=1.2st, flux=0.20, uncorroborated qualifies)"
    },
    "5. Relax Rigidity (2.0st/0.35flux)": {
        "min_duration_sec": 0.50,
        "max_pitch_std_st": 2.0,
        "max_spectral_flux": 0.35,
        "standalone_confidence": 0.50,
        "require_corroboration": True,
        "desc": "Fix 2 with ONLY acoustic rigidity relaxed (dur=0.50s, pitch=2.0st, flux=0.35, corroboration required)"
    },
    "6. Intermediate Dur (0.40s)": {
        "min_duration_sec": 0.40,
        "max_pitch_std_st": 1.2,
        "max_spectral_flux": 0.20,
        "standalone_confidence": 0.50,
        "require_corroboration": True,
        "desc": "Exploratory: Intermediate duration floor 0.40s with Fix 2 rigidity and corroboration"
    },
    "7. Severe Standalone Allowed (>=0.60s)": {
        "min_duration_sec": 0.50,
        "max_pitch_std_st": 1.2,
        "max_spectral_flux": 0.20,
        "standalone_confidence": 0.50,
        "require_corroboration": True,
        "allow_severe_standalone": True,  # If a single prolongation >= 0.60s, it self-corroborates
        "desc": "Exploratory: Fix 2, but standalone prolongation >= 0.60s self-corroborates without other events"
    }
}

# Pre-load audio waveforms and pre-extract raw candidate segments once for all 30 clips
print("Pre-extracting acoustic prolongation segments for all 30 development clips...")
clip_audio_cache = {}
clip_raw_cands = {}
for idx, p in enumerate(predictions_cache, 1):
    fpath = p["file_path"]
    if not os.path.exists(fpath):
        fpath = os.path.join(project_root, "Datasets", p["dataset"], p["filename"])
    y, _ = ac.load_audio(fpath)
    clip_audio_cache[p["filename"]] = (y, fpath)
    # Extract with permissive bounds so all variants can filter instantly in memory
    cands = ac.find_prolongation_candidates(
        y,
        min_duration_sec=0.20,
        max_pitch_std_st=10.0,
        max_spectral_flux=10.0
    )
    clip_raw_cands[p["filename"]] = cands
    if idx % 10 == 0 or idx == len(predictions_cache):
        print(f"  [Pre-extraction] Cached {idx}/{len(predictions_cache)} clips...")

def evaluate_variant(var_name, var_cfg):
    min_dur = var_cfg["min_duration_sec"]
    max_pitch = var_cfg["max_pitch_std_st"]
    max_flux = var_cfg["max_spectral_flux"]
    standalone_conf = var_cfg["standalone_confidence"]
    req_corrob = var_cfg["require_corroboration"]
    allow_severe = var_cfg.get("allow_severe_standalone", False)

    variant_predictions = []
    
    for p in predictions_cache:
        fname = p["filename"]
        y, fpath = clip_audio_cache[fname]
        
        # 1. Non-prolongation stutter events (lexical repetitions & blocks) are fixed from preflight
        fixed_stutter_events = [ev for ev in p["details"].get("stutter", []) if ev["type"] != "acoustic_prolongation"]
        lex_events = sum(1 for ev in fixed_stutter_events if ev["type"] in ("word_repetition", "broken_syllable_fragment", "phrase_repetition"))
        block_events = sum(1 for ev in fixed_stutter_events if ev["type"] == "inter_word_acoustic_block")
        
        # 2. Filter precomputed acoustic prolongations under variant thresholds
        raw_cands = clip_raw_cands[fname]
        prol_cands = [
            c for c in raw_cands
            if c["duration"] >= min_dur and c["pitch_std_st"] <= max_pitch and c["mean_flux"] <= max_flux
        ]
        ac_prol_events = len(prol_cands)
        
        # 3. Decision & Confidence logic
        confidence = 0.0
        if lex_events > 0:
            confidence = max(confidence, 0.85 + min(0.15, lex_events * 0.05))
            
        if ac_prol_events > 0:
            if req_corrob:
                prol_conf = standalone_conf
                if ac_prol_events >= 2:
                    prol_conf += min(0.20, (ac_prol_events - 1) * 0.10)
                elif allow_severe:
                    # Check if the single prolongation was severe (>= 0.60s)
                    if any(c["duration"] >= 0.60 for c in prol_cands):
                        prol_conf = 0.65  # qualifies as confident
                confidence = max(confidence, prol_conf)
            else:
                # Standalone uncorroborated prolongation allowed to reach confidence >= 0.70
                confidence = max(confidence, standalone_conf + min(0.20, ac_prol_events * 0.05))

        if block_events > 0:
            confidence = max(confidence, 0.55 + min(0.15, block_events * 0.05))

        # Corroborating evidence
        if req_corrob:
            has_corroboration = (
                lex_events > 0 or 
                (ac_prol_events > 0 and block_events > 0) or 
                (ac_prol_events >= 2) or
                (allow_severe and ac_prol_events >= 1 and any(c["duration"] >= 0.60 for c in prol_cands))
            )
        else:
            has_corroboration = (lex_events > 0 or ac_prol_events > 0 or block_events > 0)

        # Cross-modal boost if prolongation co-occurs with block
        if ac_prol_events > 0 and block_events > 0 and confidence < 0.60:
            confidence = min(1.0, confidence + 0.10)

        # Categorize
        if confidence >= 0.60:
            classification = "CONFIDENT_DISFLUENT"
            has_stutter = 1
        elif confidence >= 0.35:
            classification = "UNCERTAIN_BORDERLINE"
            has_stutter = 1 if has_corroboration else 0
        else:
            classification = "CONFIDENT_CLEAN"
            has_stutter = 0

        # Filler status is fixed from Fix 1
        has_filler = p["pred_has_filler"]
        dominant = get_dominant_label(has_stutter, has_filler)

        variant_predictions.append({
            "filename": fname,
            "dataset": p["dataset"],
            "gt_has_stutter": p["gt_has_stutter"],
            "gt_has_filler": p["gt_has_filler"],
            "gt_dominant": p["gt_dominant"],
            "pred_has_stutter": has_stutter,
            "pred_has_filler": has_filler,
            "pred_dominant": dominant,
            "stutter_events": lex_events + block_events + ac_prol_events,
            "ac_prol_events": ac_prol_events,
            "confidence": round(confidence, 3),
            "classification": classification,
            "prol_cands": prol_cands,
            "fixed_events": fixed_stutter_events,
            "status": p.get("status", "SUCCESS"),
            "asr_status": p.get("asr_status", "SUCCESS")
        })

    # Compute metrics
    multiclass = compute_multiclass_metrics(
        [p["gt_dominant"] for p in variant_predictions],
        [p["pred_dominant"] for p in variant_predictions],
        labels=["Clean", "Stutter", "Filler"]
    )
    stutter_bin = compute_binary_metrics(
        [p["gt_has_stutter"] for p in variant_predictions],
        [p["pred_has_stutter"] for p in variant_predictions]
    )
    filler_bin = compute_binary_metrics(
        [p["gt_has_filler"] for p in variant_predictions],
        [p["pred_has_filler"] for p in variant_predictions]
    )

    # Per-dataset false positives
    st_fps_by_ds = {}
    fl_fps_by_ds = {}
    for ds in ["UCLASS", "FluencyBank", "Libri_Stutter"]:
        st_fps_by_ds[ds] = sum(1 for p in variant_predictions if p["dataset"] == ds and p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 1)
        fl_fps_by_ds[ds] = sum(1 for p in variant_predictions if p["dataset"] == ds and p["gt_has_filler"] == 0 and p["pred_has_filler"] == 1)

    return {
        "name": var_name,
        "config": var_cfg,
        "multiclass": multiclass,
        "stutter_binary": stutter_bin,
        "filler_binary": filler_bin,
        "st_fps_by_ds": st_fps_by_ds,
        "fl_fps_by_ds": fl_fps_by_ds,
        "predictions": variant_predictions
    }

# Execute all variants
all_results = {}
for name, cfg in VARIANTS.items():
    all_results[name] = evaluate_variant(name, cfg)

# Print Summary Table
print("=" * 115)
print("SPEECHAI FIX 2B: CONTROLLED FACTORIAL ABLATION RESULTS (30 CLIPS)")
print("=" * 115)
header = f"{'Variant':<35} | {'Acc':<6} | {'MacF1':<6} | {'StPrec':<6} | {'StRec':<6} | {'StF1':<6} | {'TP/FP/FN/TN':<14} | {'FB StFPs':<8} | {'FlF1':<6}"
print(header)
print("-" * 115)

fix1_tps = set(p["filename"] for p in all_results["1. Fix 1 (Base)"]["predictions"] if p["gt_has_stutter"] == 1 and p["pred_has_stutter"] == 1)

for name, res in all_results.items():
    m = res["multiclass"]
    st = res["stutter_binary"]
    fl = res["filler_binary"]
    fb_fps = res["st_fps_by_ds"]["FluencyBank"]
    counts = f"{st['tp']}/{st['fp']}/{st['fn']}/{st['tn']}"
    row = f"{name:<35} | {m['accuracy']:<6.4f} | {m['macro_f1']:<6.4f} | {st['precision']:<6.4f} | {st['recall']:<6.4f} | {st['f1']:<6.4f} | {counts:<14} | {fb_fps:<8} | {fl['f1']:<6.4f}"
    print(row)

print("=" * 115)

# Detailed analysis of missed cases
print("\n" + "=" * 80)
print("INSPECTION OF SPECIFIC MISSED STUTTER CASES")
print("=" * 80)

fix1_preds = {p["filename"]: p for p in all_results["1. Fix 1 (Base)"]["predictions"]}
fix2_preds = {p["filename"]: p for p in all_results["2. Complete Fix 2"]["predictions"]}

missed_filenames = [fname for fname in fix1_tps if fix2_preds[fname]["pred_has_stutter"] == 0]

print(f"Total True Positive clips lost in Complete Fix 2 compared to Fix 1: {len(missed_filenames)}")

for fname in sorted(missed_filenames):
    p1 = fix1_preds[fname]
    p2 = fix2_preds[fname]
    y, fpath = clip_audio_cache[fname]
    
    print(f"\n--- Clip: {fname} [{p1['dataset']}] ---")
    print(f"  Fix 1 Stutter Events: {p1['stutter_events']} (has_stutter={p1['pred_has_stutter']}, conf={p1['confidence']})")
    print(f"  Fix 2 Stutter Events: {p2['stutter_events']} (has_stutter={p2['pred_has_stutter']}, conf={p2['confidence']})")
    print(f"  Fixed Events (Lexical/Block): {p1['fixed_events']}")
    print(f"  Fix 1 Prolongations (0.32s, 2.0st, 0.35flux): {p1['prol_cands']}")
    print(f"  Fix 2 Prolongations (0.50s, 1.2st, 0.20flux): {p2['prol_cands']}")
    
    # Check behavior across each isolated ablation variant:
    reasons = []
    # 1. Did duration alone reject it?
    cands_dur_reverted = all_results["3. Revert Duration (0.32s)"]["predictions"]
    p3 = next(x for x in cands_dur_reverted if x["filename"] == fname)
    if p3["pred_has_stutter"] == 1:
        reasons.append("Recovered by reverting duration to 0.32s")
    
    # 2. Did confidence capping alone reject it?
    cands_conf_relaxed = all_results["4. Relax Standalone Conf (0.70)"]["predictions"]
    p4 = next(x for x in cands_conf_relaxed if x["filename"] == fname)
    if p4["pred_has_stutter"] == 1:
        reasons.append("Recovered by relaxing standalone confidence to 0.70")
        
    # 3. Did rigidity alone reject it?
    cands_rig_relaxed = all_results["5. Relax Rigidity (2.0st/0.35flux)"]["predictions"]
    p5 = next(x for x in cands_rig_relaxed if x["filename"] == fname)
    if p5["pred_has_stutter"] == 1:
        reasons.append("Recovered by relaxing rigidity (pitch std < 2.0, flux < 0.35)")

    # 4. Severe standalone?
    p7 = next(x for x in all_results["7. Severe Standalone Allowed (>=0.60s)"]["predictions"] if x["filename"] == fname)
    if p7["pred_has_stutter"] == 1:
        reasons.append("Recovered by allowing severe standalone (>= 0.60s) prolongation")

    print(f"  Root Cause Diagnosis: {reasons if reasons else ['Requires multiple threshold relaxations']}")

# Save complete ablation JSON results for permanent artifact
out_ablation_json = os.path.join(project_root, "evaluation_results", "experimental", "audit", "fix2b_ablation_results.json")
# Strip unneeded large audio cache before dumping
json_dumpable = {}
for k, v in all_results.items():
    json_dumpable[k] = {
        "name": v["name"],
        "config": v["config"],
        "multiclass": v["multiclass"],
        "stutter_binary": v["stutter_binary"],
        "filler_binary": v["filler_binary"],
        "st_fps_by_ds": v["st_fps_by_ds"],
        "fl_fps_by_ds": v["fl_fps_by_ds"]
    }

with open(out_ablation_json, "w", encoding="utf-8") as f:
    json.dump(json_dumpable, f, indent=2)

print(f"\n[DONE] Saved complete ablation metrics to: {out_ablation_json}")
