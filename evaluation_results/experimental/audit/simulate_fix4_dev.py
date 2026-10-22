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

dev_json = os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips_refined.json")
with open(dev_json, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

ac = AcousticFeatureExtractor(ExperimentalConfig().acoustic)
analyzer_fix3 = SpeechContextAnalyzer(SpeechContextConfig(enabled=True, enable_lexical_syntactic_context=False))
analyzer_fix4 = SpeechContextAnalyzer(SpeechContextConfig(enabled=True, enable_lexical_syntactic_context=True))

print("Caching waveforms & raw candidates...")
clip_cache = {}
for p in dev_data["predictions"]:
    fpath = p["file_path"]
    if not os.path.exists(fpath):
        fpath = os.path.join(project_root, "Datasets", p["dataset"], p["filename"])
    y, sr = ac.load_audio(fpath)
    
    raw_cands = ac.find_prolongation_candidates(
        y, 
        min_duration_sec=0.25,
        max_pitch_std_st=5.0,
        max_spectral_flux=1.0
    )
    clip_cache[p["filename"]] = (y, sr, fpath, raw_cands)

def run_eval(mode="fix2b"):
    """
    mode: 'fix2b', 'fix3', or 'fix4'
    """
    preds = []
    analyzer = analyzer_fix4 if mode == "fix4" else analyzer_fix3
    enable_context = (mode in ("fix3", "fix4"))
    
    for p in dev_data["predictions"]:
        fname = p["filename"]
        y, sr, fpath, raw_cands = clip_cache[fname]
        audio_dur = len(y) / sr
        transcript = p.get("pred_transcript", "")
        words_list = p.get("words_list", [])
        
        if not words_list and transcript:
            words = [w for w in transcript.split() if w]
            if words:
                word_dur = audio_dur / len(words)
                words_list = [
                    {"word": w, "start": round(i * word_dur, 2), "end": round((i + 1) * word_dur, 2), "probability": 0.95}
                    for i, w in enumerate(words)
                ]
        
        global_ctx = analyzer.estimate_syllable_rate(
            words_list=words_list,
            transcript=transcript,
            duration_sec=audio_dur
        )
        
        fixed_stutter_events = [ev for ev in p["details"].get("stutter", []) if ev["type"] != "acoustic_prolongation"]
        lex_events = sum(1 for ev in fixed_stutter_events if ev["type"] in ("word_repetition", "broken_syllable_fragment", "phrase_repetition"))
        block_events = sum(1 for ev in fixed_stutter_events if ev["type"] == "inter_word_acoustic_block")
        
        rigid_cands = [
            c for c in raw_cands 
            if c["pitch_std_st"] <= 1.2 and c["mean_flux"] <= 0.20
        ]
        
        admitted_prols = []
        for c in rigid_cands:
            c_dur = c["duration"]
            if not enable_context:
                if c_dur >= 0.50:
                    admitted_prols.append(c)
            else:
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
        confidence = 0.0
        if lex_events > 0:
            confidence = max(confidence, 0.85 + min(0.15, lex_events * 0.05))
            
        severe_dur_floor = 0.60
        has_severe_standalone = any(c.get("duration", 0.0) >= severe_dur_floor for c in admitted_prols)
        has_context_support = enable_context and any(c.get("context_score", 0.0) >= 0.65 for c in admitted_prols)
        
        if ac_prol_events > 0:
            if has_severe_standalone and ac_prol_events == 1:
                prol_conf = 0.65
            elif ac_prol_events >= 2:
                if enable_context and all(c.get("context_score", 0.50) < 0.40 for c in admitted_prols):
                    prol_conf = 0.50
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
            "confidence": round(confidence, 3),
            "ac_prol_events": ac_prol_events,
            "admitted_prols": admitted_prols
        })
        
    return preds

labels = ["Stutter", "Filler", "Clean"]

b_preds = run_eval("fix2b")
mc_b = compute_multiclass_metrics([p["gt_dominant"] for p in b_preds], [p["pred_dominant"] for p in b_preds], labels)
st_b = compute_binary_metrics([p["gt_has_stutter"] for p in b_preds], [p["pred_has_stutter"] for p in b_preds])
fl_b = compute_binary_metrics([p["gt_has_filler"] for p in b_preds], [p["pred_has_filler"] for p in b_preds])
fb_fps_b = sum(1 for p in b_preds if p["dataset"] == "FluencyBank" and p["pred_has_stutter"] == 1 and p["gt_has_stutter"] == 0)

f3_preds = run_eval("fix3")
mc_3 = compute_multiclass_metrics([p["gt_dominant"] for p in f3_preds], [p["pred_dominant"] for p in f3_preds], labels)
st_3 = compute_binary_metrics([p["gt_has_stutter"] for p in f3_preds], [p["pred_has_stutter"] for p in f3_preds])
fl_3 = compute_binary_metrics([p["gt_has_filler"] for p in f3_preds], [p["pred_has_filler"] for p in f3_preds])
fb_fps_3 = sum(1 for p in f3_preds if p["dataset"] == "FluencyBank" and p["pred_has_stutter"] == 1 and p["gt_has_stutter"] == 0)

f4_preds = run_eval("fix4")
mc_4 = compute_multiclass_metrics([p["gt_dominant"] for p in f4_preds], [p["pred_dominant"] for p in f4_preds], labels)
st_4 = compute_binary_metrics([p["gt_has_stutter"] for p in f4_preds], [p["pred_has_stutter"] for p in f4_preds])
fl_4 = compute_binary_metrics([p["gt_has_filler"] for p in f4_preds], [p["pred_has_filler"] for p in f4_preds])
fb_fps_4 = sum(1 for p in f4_preds if p["dataset"] == "FluencyBank" and p["pred_has_stutter"] == 1 and p["gt_has_stutter"] == 0)

print("\n========================= 30-CLIP BENCHMARK SUMMARY =========================")
print(f"Fix 2B Baseline : Acc={mc_b['accuracy']:.4f}, MacF1={mc_b['macro_f1']:.4f}, StPrec={st_b['precision']:.4f}, StRec={st_b['recall']:.4f}, StF1={st_b['f1']:.4f}, FB_FPs={fb_fps_b}")
print(f"Fix 3 Context   : Acc={mc_3['accuracy']:.4f}, MacF1={mc_3['macro_f1']:.4f}, StPrec={st_3['precision']:.4f}, StRec={st_3['recall']:.4f}, StF1={st_3['f1']:.4f}, FB_FPs={fb_fps_3}")
print(f"Fix 4 Lex+Syn   : Acc={mc_4['accuracy']:.4f}, MacF1={mc_4['macro_f1']:.4f}, StPrec={st_4['precision']:.4f}, StRec={st_4['recall']:.4f}, StF1={st_4['f1']:.4f}, FB_FPs={fb_fps_4}")
print("==============================================================================\n")

print("--- Binary Confusion Matrix ---")
print(f"Fix 2B: TP={st_b['tp']}, FP={st_b['fp']}, FN={st_b['fn']}, TN={st_b['tn']}")
print(f"Fix 3:  TP={st_3['tp']}, FP={st_3['fp']}, FN={st_3['fn']}, TN={st_3['tn']}")
print(f"Fix 4:  TP={st_4['tp']}, FP={st_4['fp']}, FN={st_4['fn']}, TN={st_4['tn']}")

print("\n--- Differences between Fix 3 and Fix 4 ---")
for p3, p4 in zip(f3_preds, f4_preds):
    if p3["pred_has_stutter"] != p4["pred_has_stutter"] or p3["pred_dominant"] != p4["pred_dominant"]:
        print(f"Clip: {p3['filename']} ({p3['dataset']}) | GT: {p3['gt_dominant']}")
        print(f"   Fix 3: Stutter={p3['pred_has_stutter']}, Dom={p3['pred_dominant']}, Conf={p3['confidence']}")
        print(f"   Fix 4: Stutter={p4['pred_has_stutter']}, Dom={p4['pred_dominant']}, Conf={p4['confidence']}")
        for c in p4["admitted_prols"]:
            print(f"      Cand: dur={c['duration']}s, score={c.get('context_score')}, ratio={c.get('elongation_ratio')}x, boundary={c.get('boundary_type')}, lex={c.get('lexical_class')}, reasons={c.get('decision_reasons')}")

# Save the official Fix 4 artifact payload
payload = {
    "benchmark_name": "SpeechAI Fix 4 Lexical and Syntactic Context Benchmark",
    "timestamp": "2026-10-20 11:15:00",
    "n_clips": 30,
    "fix2b_baseline": {
        "multiclass_3class": mc_b,
        "stutter_binary": st_b,
        "filler_binary": fl_b,
        "fluencybank_stutter_fps": fb_fps_b
    },
    "fix3_contextual": {
        "multiclass_3class": mc_3,
        "stutter_binary": st_3,
        "filler_binary": fl_3,
        "fluencybank_stutter_fps": fb_fps_3
    },
    "fix4_contextual": {
        "multiclass_3class": mc_4,
        "stutter_binary": st_4,
        "filler_binary": fl_4,
        "fluencybank_stutter_fps": fb_fps_4
    },
    "predictions_fix4": f4_preds
}

out_path = os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips_fix4.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"\nSaved official Fix 4 evaluation artifact to: {out_path}")
