import os
import sys
import json

audit_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(audit_dir)))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.experimental.config import ExperimentalConfig, StutterDetectorConfig, SpeechContextConfig
from backend.experimental.acoustic_features import AcousticFeatureExtractor
from backend.experimental.speech_context import SpeechContextAnalyzer
from backend.experimental.eval_harness import compute_binary_metrics, compute_multiclass_metrics, get_dominant_label

# Load dev data
dev_json = os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips_refined.json")
with open(dev_json, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

ac = AcousticFeatureExtractor(ExperimentalConfig().acoustic)
analyzer = SpeechContextAnalyzer(SpeechContextConfig())

# Load waveforms and pre-extract candidates
print("Caching waveforms & raw candidates (down to 0.30s)...")
clip_cache = {}
for p in dev_data["predictions"]:
    fpath = p["file_path"]
    if not os.path.exists(fpath):
        fpath = os.path.join(project_root, "Datasets", p["dataset"], p["filename"])
    y, sr = ac.load_audio(fpath)
    
    # Pre-extract with generous bounds
    raw_cands = ac.find_prolongation_candidates(
        y, 
        min_duration_sec=0.25,
        max_pitch_std_st=5.0,
        max_spectral_flux=1.0
    )
    clip_cache[p["filename"]] = (y, sr, fpath, raw_cands)

def run_simulation(enable_context=False):
    preds = []
    
    for p in dev_data["predictions"]:
        fname = p["filename"]
        y, sr, fpath, raw_cands = clip_cache[fname]
        audio_dur = len(y) / sr
        transcript = p.get("pred_transcript", "")
        words_list = p.get("words_list", [])  # Might be empty in json, we can synthesize pseudo word from transcript if needed
        
        # If words_list is empty, construct word-level approximations from transcript words across audio duration
        if not words_list and transcript:
            words = [w for w in transcript.split() if w]
            if words:
                word_dur = audio_dur / len(words)
                words_list = []
                for i, w in enumerate(words):
                    words_list.append({
                        "word": w,
                        "start": round(i * word_dur, 2),
                        "end": round((i + 1) * word_dur, 2),
                        "probability": 0.95
                    })
        
        # Compute global context
        global_ctx = analyzer.estimate_syllable_rate(
            words_list=words_list,
            transcript=transcript,
            duration_sec=audio_dur
        )
        
        # Non-prolongation stutter events (lexical repetitions & blocks) are fixed from preflight
        fixed_stutter_events = [ev for ev in p["details"].get("stutter", []) if ev["type"] != "acoustic_prolongation"]
        lex_events = sum(1 for ev in fixed_stutter_events if ev["type"] in ("word_repetition", "broken_syllable_fragment", "phrase_repetition"))
        block_events = sum(1 for ev in fixed_stutter_events if ev["type"] == "inter_word_acoustic_block")
        
        # Filter candidates under rigidity thresholds
        rigid_cands = [
            c for c in raw_cands 
            if c["pitch_std_st"] <= 1.2 and c["mean_flux"] <= 0.20
        ]
        
        admitted_prols = []
        for c in rigid_cands:
            c_dur = c["duration"]
            if not enable_context:
                # Fix 2B baseline: duration floor 0.50s
                if c_dur >= 0.50:
                    admitted_prols.append(c)
            else:
                # Fix 3: Context-aware analysis
                ctx_res = analyzer.analyze_candidate(
                    c, 
                    audio_duration_sec=audio_dur,
                    words_list=words_list,
                    transcript=transcript,
                    global_context=global_ctx
                )
                if ctx_res["is_admitted"]:
                    c_copy = dict(c)
                    c_copy.update(ctx_res)
                    admitted_prols.append(c_copy)
        
        ac_prol_events = len(admitted_prols)
        
        # Decision & Confidence Logic
        confidence = 0.0
        if lex_events > 0:
            confidence = max(confidence, 0.85 + min(0.15, lex_events * 0.05))
            
        severe_dur_floor = 0.60
        has_severe_standalone = any(c.get("duration", 0.0) >= severe_dur_floor for c in admitted_prols)
        has_context_support = enable_context and any(c.get("context_score", 0.0) >= 0.65 for c in admitted_prols)
        
        if ac_prol_events > 0:
            if has_severe_standalone:
                prol_conf = 0.65
            elif ac_prol_events >= 2:
                # In Fix 3, if all prolongations have very low context score (< 0.40) or are in slow speech, don't corroborate!
                if enable_context and all(c.get("context_score", 0.50) < 0.40 for c in admitted_prols):
                    prol_conf = 0.50  # Damped due to slow fluent speech / phrase-final lengthening
                else:
                    prol_conf = 0.50 + min(0.20, (ac_prol_events - 1) * 0.10)
            elif has_context_support:
                prol_conf = 0.65
            else:
                prol_conf = 0.50
            confidence = max(confidence, prol_conf)
            
        if block_events > 0:
            confidence = max(confidence, 0.55 + min(0.15, block_events * 0.05))
            
        has_corroboration = (
            lex_events > 0 or 
            (ac_prol_events > 0 and block_events > 0) or 
            (ac_prol_events >= 2 and not (enable_context and all(c.get("context_score", 0.50) < 0.40 for c in admitted_prols))) or
            has_severe_standalone or
            has_context_support
        )
        
        if ac_prol_events > 0 and block_events > 0 and confidence < 0.60:
            confidence = min(1.0, confidence + 0.10)
            
        if confidence >= 0.60:
            has_stutter = 1
        elif confidence >= 0.35:
            has_stutter = 1 if has_corroboration else 0
        else:
            has_stutter = 0
            
        # Filler prediction (invariant)
        has_filler = p["pred_has_filler"]
        pred_dominant = get_dominant_label(has_stutter, has_filler)
        
        preds.append({
            "filename": fname,
            "dataset": p["dataset"],
            "pred_has_stutter": has_stutter,
            "pred_has_filler": has_filler,
            "pred_dominant": pred_dominant,
            "gt_dominant": p["gt_dominant"],
            "gt_has_stutter": p["gt_has_stutter"],
            "gt_has_filler": p["gt_has_filler"],
            "confidence": confidence,
            "ac_prol_events": ac_prol_events,
            "admitted_prols": admitted_prols
        })
        
    return preds

print("\n--- Running Baseline Fix 2B (Context Disabled) ---")
b_preds = run_simulation(enable_context=False)
st_cm_b = compute_binary_metrics([p["gt_has_stutter"] for p in b_preds], [p["pred_has_stutter"] for p in b_preds])
mc_b = compute_multiclass_metrics([p["gt_dominant"] for p in b_preds], [p["pred_dominant"] for p in b_preds], labels=["Stutter", "Filler", "Clean"])
fb_fps_b = sum(1 for p in b_preds if p["dataset"] == "FluencyBank" and p["pred_has_stutter"] == 1 and p["gt_has_stutter"] == 0)

print(f"Fix 2B -> Acc: {mc_b['accuracy']:.4f}, MacF1: {mc_b['macro_f1']:.4f}, StPrec: {st_cm_b['precision']:.4f}, StRec: {st_cm_b['recall']:.4f}, StF1: {st_cm_b['f1']:.4f}")
print(f"Fix 2B -> Stutter TP: {st_cm_b['tp']}, FP: {st_cm_b['fp']}, FN: {st_cm_b['fn']}, TN: {st_cm_b['tn']} | FB FPs: {fb_fps_b}")

print("\n--- Running Fix 3 (Context Enabled) ---")
c_preds = run_simulation(enable_context=True)
st_cm_c = compute_binary_metrics([p["gt_has_stutter"] for p in c_preds], [p["pred_has_stutter"] for p in c_preds])
mc_c = compute_multiclass_metrics([p["gt_dominant"] for p in c_preds], [p["pred_dominant"] for p in c_preds], labels=["Stutter", "Filler", "Clean"])
fb_fps_c = sum(1 for p in c_preds if p["dataset"] == "FluencyBank" and p["pred_has_stutter"] == 1 and p["gt_has_stutter"] == 0)

print(f"Fix 3  -> Acc: {mc_c['accuracy']:.4f}, MacF1: {mc_c['macro_f1']:.4f}, StPrec: {st_cm_c['precision']:.4f}, StRec: {st_cm_c['recall']:.4f}, StF1: {st_cm_c['f1']:.4f}")
print(f"Fix 3  -> Stutter TP: {st_cm_c['tp']}, FP: {st_cm_c['fp']}, FN: {st_cm_c['fn']}, TN: {st_cm_c['tn']} | FB FPs: {fb_fps_c}")

# Check clip transitions
print("\n--- Clip Transitions: Fix 2B vs Fix 3 ---")
for pb, pc in zip(b_preds, c_preds):
    if pb["pred_has_stutter"] != pc["pred_has_stutter"]:
        gt = pb["gt_has_stutter"]
        trans = "TP RECOVERED!" if pc["pred_has_stutter"] == 1 and gt == 1 else ("FP INTRODUCED!" if pc["pred_has_stutter"] == 1 and gt == 0 else ("FP ELIMINATED!" if pc["pred_has_stutter"] == 0 and gt == 0 else "TP LOST!"))
        print(f"  [{pb['dataset']}] {pb['filename']} -> Fix2B={pb['pred_has_stutter']}, Fix3={pc['pred_has_stutter']} (GT={gt}) ==> {trans}")
        for c in pc["admitted_prols"]:
            print(f"      Cand: {c['duration']}s, score={c.get('context_score')}, elong={c.get('elongation_ratio')}x, pf={c.get('is_phrase_final')}")

target_fnames = ["FluencyBank_010_79.wav", "FluencyBank_010_97.wav", "M_0030_16y4m_1_dysfluent_011.wav", "M_0030_16y4m_1_dysfluent_012.wav", "M_0030_16y4m_1_dysfluent_013.wav"]
print("\n--- Diagnostic of 5 Missed Clips in Fix 3 ---")
for p in dev_data["predictions"]:
    if p["filename"] in target_fnames:
        fname = p["filename"]
        y, sr, fpath, raw_cands = clip_cache[fname]
        audio_dur = len(y) / sr
        transcript = p.get("pred_transcript", "")
        words_list = p.get("words_list", [])
        if not words_list and transcript:
            words = [w for w in transcript.split() if w]
            if words:
                word_dur = audio_dur / len(words)
                words_list = [{"word": w, "start": round(i*word_dur, 2), "end": round((i+1)*word_dur, 2), "probability": 0.95} for i, w in enumerate(words)]
        
        global_ctx = analyzer.estimate_syllable_rate(words_list=words_list, transcript=transcript, duration_sec=audio_dur)
        print(f"\nClip: {fname} ({p['dataset']}) | Text: '{transcript}'")
        print(f"  Global Rate: {global_ctx['global_speech_rate']} syll/s | Mean Syll: {global_ctx['mean_syllable_duration_sec']}s")
        rigid_cands = [c for c in raw_cands if c["pitch_std_st"] <= 1.2 and c["mean_flux"] <= 0.20]
        print(f"  Rigid Candidates (dur >= 0.25s): {len(rigid_cands)}")
        for c in rigid_cands:
            ctx_res = analyzer.analyze_candidate(c, audio_duration_sec=audio_dur, words_list=words_list, transcript=transcript, global_context=global_ctx)
            print(f"    * Cand {c['start']:.2f}-{c['end']:.2f}s (dur={c['duration']:.2f}s, pitch_std={c['pitch_std_st']}, flux={c['mean_flux']}):")
            print(f"        Elongation: {ctx_res['elongation_ratio']}x, PhraseFinal: {ctx_res['is_phrase_final']} ({ctx_res['phrase_final_reason']})")
            print(f"        Context Score: {ctx_res['context_score']}, Admitted: {ctx_res['is_admitted']}, Reason: {ctx_res['rejection_reason']}")
