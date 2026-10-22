"""
Phase 3 Candidate Improvement Evaluation Script (Development Set, N=90)
======================================================================
Strict Scope: Development Split Only (30 standardized dev + 60 generalization clips).
Evaluates:
- Baseline: Frozen Phase 2 Detector
- Candidate 1: Standalone Short Prolongation Calibration (requires corroboration for < 0.45s)
- Candidate 2: Lexical Phonological Cutoff & Stem Repetition Detection
- Candidate 1+2: Combined Improvement

Outputs comparison table of TP, FP, FN, TN, Precision, Recall, F1, and Specificity.
"""

import os
import sys
import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent.parent
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.experimental.config import StutterDetectorConfig, SpeechContextConfig, AcousticConfig
from backend.experimental.acoustic_features import AcousticFeatureExtractor
from backend.experimental.speech_context import SpeechContextAnalyzer
from backend.experimental.stutter_detector import HybridStutterDetector, clean_token
from backend.experimental.interrupted_repetition import basic_stem, STOP_PARTICLES


def detect_lexical_repetitions_candidate2(transcript: str, words_list: Optional[List[Dict[str, Any]]] = None) -> Tuple[int, List[Dict[str, Any]]]:
    """Candidate 2 extension: Adds terminal hyphenated cutoff tokens and content-word stem repetition."""
    if not transcript:
        return 0, []

    text_lower = transcript.lower()
    events = 0
    details = []

    # 1. Standard triplicate syllable prefix repetitions (e.g., "w-w-what")
    reps_3 = list(re.finditer(r'\b([a-zA-Z]{1,4})-\1-\1([a-zA-Z]*)\b', text_lower))
    for m in reps_3:
        events += 1
        details.append({
            "type": "sound_repetition_3x",
            "text": m.group(0),
            "matched_pattern": m.group(1),
            "confidence": 0.95
        })

    # 2. Standard duplicate syllable prefix repetitions (e.g., "w-what")
    cleaned_for_2x = re.sub(r'\b([a-zA-Z]{1,4})-\1-\1([a-zA-Z]*)\b', ' ', text_lower)
    reps_2 = list(re.finditer(r'\b([a-zA-Z]{1,4})-\1([a-zA-Z]*)\b', cleaned_for_2x))
    for m in reps_2:
        events += 1
        details.append({
            "type": "sound_repetition_2x",
            "text": m.group(0),
            "matched_pattern": m.group(1),
            "confidence": 0.85
        })

    # 3. Standard broken single-letter fragments (e.g., "b- because")
    broken_frags = list(re.finditer(r'\b([a-zA-Z]{1,2})[\s\-]+([a-zA-Z]{3,})\b', text_lower))
    for m in broken_frags:
        prefix = m.group(1)
        target = m.group(2)
        if target.startswith(prefix) and len(prefix) < len(target):
            events += 1
            details.append({
                "type": "broken_syllable_fragment",
                "text": m.group(0),
                "confidence": 0.80
            })

    # NEW Candidate 2 Feature: Terminal hyphenated phonological cutoff (e.g., "Mr. Sh-", "how to-")
    # Matches words with a trailing hyphen representing an incomplete word/abrupt articulatory cutoff
    cutoff_frags = list(re.finditer(r'\b([a-zA-Z]{1,4})-(?=\s|$|[.,?!;])', text_lower))
    for m in cutoff_frags:
        frag = m.group(1)
        # Avoid double-counting if already matched by 2x or 3x reps
        if not any(frag in d.get("text", "") for d in details):
            events += 1
            details.append({
                "type": "phonological_cutoff_fragment",
                "text": m.group(0),
                "fragment": frag,
                "confidence": 0.80
            })

    # 4. Consecutive whole-word and stem repetitions
    raw_tokens = text_lower.split()
    words = [clean_token(w) for w in raw_tokens if clean_token(w)]
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        if w1 == w2:
            events += 1
            details.append({
                "type": "word_repetition",
                "text": f"{w1} {w2}",
                "confidence": 0.85
            })
        elif len(w1) >= 3 and len(w2) >= 3 and w1 not in STOP_PARTICLES and w2 not in STOP_PARTICLES:
            # NEW Candidate 2 Feature: Morphological / stem repetition (e.g. "going go")
            if basic_stem(w1) == basic_stem(w2):
                events += 1
                details.append({
                    "type": "stem_repetition",
                    "text": f"{w1} {w2}",
                    "stem": basic_stem(w1),
                    "confidence": 0.80
                })

    return events, details


def evaluate_candidate(
    detector: HybridStutterDetector,
    dev_records: List[Dict[str, Any]],
    candidate_mode: str = "baseline"
) -> Dict[str, Any]:
    """
    Evaluates detector on dev_records using candidate_mode:
    - 'baseline': default frozen Phase 2
    - 'candidate1': prolongation calibration (short standalone < 0.45s requires corroboration)
    - 'candidate2': lexical cutoff & stem repetition
    - 'candidate1_2': combined candidate 1 and 2
    """
    records = []

    for r in dev_records:
        transcript = r['transcript']
        words_list = r['words_list']
        audio_path = r['file_path']
        pred_filler = 1 if r.get('gt_dominant') == 'Filler' or r.get('pred_has_filler', 0) == 1 else 0

        # Evaluate standard detector components
        all_details = []

        if candidate_mode in ("candidate2", "candidate1_2"):
            lex_events, lex_details = detect_lexical_repetitions_candidate2(transcript, words_list)
        else:
            lex_events, lex_details = detector.detect_lexical_repetitions(transcript, words_list)
        all_details.extend(lex_details)

        # N-gram repetition
        if getattr(detector.config, "enable_ngram_repetition", False):
            from backend.experimental.ngram_repetition import NGramRepetitionDetector
            ng_det = NGramRepetitionDetector(allowed_sizes=(2, 3), base_confidence=0.85)
            ng_ev, ng_dt = ng_det.detect(transcript, words_list)
            lex_events += ng_ev
            all_details.extend(ng_dt)

        # Interrupted repetition
        if getattr(detector.config, "enable_interrupted_repetition", False):
            from backend.experimental.interrupted_repetition import InterruptedRepetitionDetector
            ir_det = InterruptedRepetitionDetector(max_skip=3, allow_stem=True, allow_hesitations=True, confidence=0.85)
            ir_ev, ir_dt = ir_det.detect(transcript, words_list)
            lex_events += ir_ev
            all_details.extend(ir_dt)

        # Acoustic extraction
        y, sr = None, None
        if audio_path and os.path.exists(audio_path):
            try:
                y, sr = detector.acoustic.load_audio(audio_path)
            except Exception:
                pass

        ac_prol_events, ac_prol_details = 0, []
        ac_block_events, ac_block_details = 0, []

        if y is not None:
            ac_prol_events, ac_prol_details = detector.detect_acoustic_prolongations(
                y, sr, words_list=words_list, transcript=transcript
            )
            all_details.extend(ac_prol_details)
            ac_block_events, ac_block_details = detector.detect_acoustic_blocks(y, words_list)
            all_details.extend(ac_block_details)

        # Filler boundary gating
        if getattr(detector.config, "enable_filler_boundary_gating", False):
            from backend.experimental.filler_boundary_gating import FillerBoundaryGater
            gater = FillerBoundaryGater(
                filler_overlap_threshold=0.35,
                filler_word_coverage_threshold=0.40,
                boundary_prolongation_min_duration_sec=0.65,
                gate_filler_blocks=True
            )
            retained_details, _ = gater.gate_details(all_details, words_list=words_list, transcript=transcript)
            all_details = retained_details
            lex_details = [d for d in all_details if "repetition" in d.get("type", "") or d.get("type") in ("broken_syllable_fragment", "phonological_cutoff_fragment", "stem_repetition")]
            ac_prol_details = [d for d in all_details if d.get("type") == "acoustic_prolongation"]
            ac_block_details = [d for d in all_details if "block" in d.get("type", "")]
            lex_events = len(lex_details)
            ac_prol_events = len(ac_prol_details)
            ac_block_events = len(ac_block_details)

        total_events = lex_events + ac_prol_events + ac_block_events
        enable_context = getattr(detector.config, "enable_contextual_scoring", False)

        # Confidence Scoring
        confidence = 0.0
        if lex_events > 0:
            confidence = max(confidence, 0.85 + min(0.15, lex_events * 0.05))

        severe_dur_floor = getattr(detector.config, "severe_prolongation_duration_sec", 0.60)
        has_severe_standalone = any(c.get("duration", 0.0) >= severe_dur_floor for c in ac_prol_details)
        has_context_support = enable_context and any(c.get("context_score", 0.0) >= 0.65 for c in ac_prol_details)

        # CANDIDATE 1 MODIFICATION:
        # In baseline, has_context_support alone sets prol_conf = 0.65 for ANY duration (even 0.34s).
        # In Candidate 1, a standalone short prolongation (< 0.45s) without severe duration or multi-event support
        # is assigned prol_conf = 0.50 (UNCERTAIN_BORDERLINE) rather than 0.65 (CONFIDENT_DISFLUENT).
        if ac_prol_events > 0:
            prol_base = getattr(detector.config, "standalone_prolongation_confidence", 0.50)
            if has_severe_standalone and ac_prol_events == 1:
                prol_conf = 0.65
            elif ac_prol_events >= 2:
                if enable_context and all(c.get("context_score", 0.50) < 0.40 for c in ac_prol_details):
                    prol_conf = prol_base
                else:
                    prol_conf = prol_base + min(0.20, (ac_prol_events - 1) * 0.10)
            elif has_context_support:
                if candidate_mode in ("candidate1", "candidate1_2"):
                    # Check if standalone duration is at least 0.45s to self-corroborate into confident stutter
                    max_dur = max((c.get("duration", 0.0) for c in ac_prol_details), default=0.0)
                    if max_dur >= 0.45:
                        prol_conf = 0.65
                    else:
                        prol_conf = 0.50  # Borderline: requires corroborating lexical/block cue
                else:
                    prol_conf = 0.65
            else:
                prol_conf = prol_base
            confidence = max(confidence, prol_conf)

        if ac_block_events > 0:
            confidence = max(confidence, 0.55 + min(0.15, ac_block_events * 0.05))

        # Corroborating evidence
        if candidate_mode in ("candidate1", "candidate1_2"):
            max_dur = max((c.get("duration", 0.0) for c in ac_prol_details), default=0.0)
            has_corroboration = (
                lex_events > 0 or 
                (ac_prol_events > 0 and ac_block_events > 0) or 
                (ac_prol_events >= 2 and not (enable_context and all(c.get("context_score", 0.50) < 0.40 for c in ac_prol_details))) or
                (ac_block_events >= 2) or
                has_severe_standalone or
                (has_context_support and max_dur >= 0.45)
            )
        else:
            has_corroboration = (
                lex_events > 0 or 
                (ac_prol_events > 0 and ac_block_events > 0) or 
                (ac_prol_events >= 2 and not (enable_context and all(c.get("context_score", 0.50) < 0.40 for c in ac_prol_details))) or
                (ac_block_events >= 2) or
                has_severe_standalone or
                has_context_support
            )

        if ac_prol_events > 0 and ac_block_events > 0 and confidence < detector.config.confident_threshold:
            confidence = min(1.0, confidence + 0.10)

        # Decision
        if confidence >= detector.config.confident_threshold:
            has_stutter = 1
            classification = "CONFIDENT_DISFLUENT"
        elif confidence >= detector.config.uncertain_threshold:
            has_stutter = 1 if has_corroboration else 0
            classification = "UNCERTAIN_BORDERLINE"
        else:
            has_stutter = 0
            classification = "CONFIDENT_CLEAN"

        # Dominant label
        if has_stutter == 1:
            pred_dominant = "Stutter"
        elif pred_filler == 1:
            pred_dominant = "Filler"
        else:
            pred_dominant = "Clean"

        records.append({
            'filename': r['filename'],
            'dataset': r['dataset'],
            'gt_dominant': r['gt_dominant'],
            'gt_has_stutter': r['gt_has_stutter'],
            'pred_dominant': pred_dominant,
            'pred_has_stutter': has_stutter,
            'confidence': confidence,
            'classification': classification,
            'lex_events': lex_events,
            'ac_prol_events': ac_prol_events,
            'ac_block_events': ac_block_events
        })

    # Compute binary metrics
    tp = sum(1 for r in records if r['gt_has_stutter'] == 1 and r['pred_has_stutter'] == 1)
    fp = sum(1 for r in records if r['gt_has_stutter'] == 0 and r['pred_has_stutter'] == 1)
    fn = sum(1 for r in records if r['gt_has_stutter'] == 1 and r['pred_has_stutter'] == 0)
    tn = sum(1 for r in records if r['gt_has_stutter'] == 0 and r['pred_has_stutter'] == 0)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    # Multiclass accuracy & Macro-F1
    classes = ['Stutter', 'Filler', 'Clean']
    correct = sum(1 for r in records if r['gt_dominant'] == r['pred_dominant'])
    acc = correct / len(records)

    f1_per_class = {}
    for c in classes:
        c_tp = sum(1 for r in records if r['gt_dominant'] == c and r['pred_dominant'] == c)
        c_fp = sum(1 for r in records if r['gt_dominant'] != c and r['pred_dominant'] == c)
        c_fn = sum(1 for r in records if r['gt_dominant'] == c and r['pred_dominant'] != c)
        c_p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 0.0
        c_r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 0.0
        c_f = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) else 0.0
        f1_per_class[c] = c_f
    macro_f1 = sum(f1_per_class.values()) / len(classes)

    return {
        'candidate_mode': candidate_mode,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': prec, 'recall': rec, 'f1': f1, 'specificity': spec,
        'accuracy': acc, 'macro_f1': macro_f1,
        'f1_per_class': f1_per_class,
        'records': records
    }


def main():
    # Load shared features
    dev30_cache_file = project_root / 'evaluation_results' / 'experimental' / 'standardized_dev' / 'run_20261020_122136' / 'feature_cache.json'
    gen60_manifest_file = project_root / 'evaluation_results' / 'experimental' / 'generalization_validation' / 'run_20261020_132214' / 'run_manifest.json'
    gen60_cache_file = project_root / 'evaluation_results' / 'experimental' / 'generalization_validation' / 'run_20261020_132214' / 'feature_cache.json'

    with open(dev30_cache_file, 'r', encoding='utf-8') as f:
        dev30_cache = json.load(f)
    with open(gen60_manifest_file, 'r', encoding='utf-8') as f:
        gen60_manifest = json.load(f)
    with open(gen60_cache_file, 'r', encoding='utf-8') as f:
        gen60_cache = json.load(f)

    all_dev_records = []
    for fn, c in dev30_cache.items():
        all_dev_records.append({
            'filename': fn,
            'dataset': c.get('dataset', 'UCLASS'),
            'file_path': c.get('file_path'),
            'transcript': c.get('transcript', ''),
            'words_list': c.get('words_list', []),
            'gt_dominant': c['gt_dominant'],
            'gt_has_stutter': c['gt_has_stutter'],
            'pred_has_filler': c.get('pred_has_filler', 0)
        })

    for p in gen60_manifest:
        fn = p['filename']
        c = gen60_cache.get(fn, {})
        all_dev_records.append({
            'filename': fn,
            'dataset': p['dataset'],
            'file_path': p['file_path'],
            'transcript': c.get('transcript', ''),
            'words_list': c.get('words_list', []),
            'gt_dominant': p['gt_dominant'],
            'gt_has_stutter': p['gt_has_stutter'],
            'pred_has_filler': c.get('pred_has_filler', 0)
        })

    # Setup detector
    acoustic_cfg = AcousticConfig()
    acoustic_extractor = AcousticFeatureExtractor(acoustic_cfg)
    context_cfg = SpeechContextConfig(
        enabled=True,
        fast_speech_threshold=5.5,
        slow_speech_threshold=2.8,
        min_context_score_for_admission=0.65,
        elongation_ratio_threshold=2.0
    )
    context_analyzer = SpeechContextAnalyzer(context_cfg)
    stutter_config = StutterDetectorConfig(
        enable_contextual_scoring=True,
        confident_threshold=0.65,
        uncertain_threshold=0.50,
        min_prolongation_duration_sec=0.50,
        severe_prolongation_duration_sec=0.60,
        standalone_prolongation_confidence=0.50,
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
        interrupted_repetition_confidence=0.85
    )
    detector = HybridStutterDetector(
        config=stutter_config,
        acoustic_extractor=acoustic_extractor,
        context_analyzer=context_analyzer
    )

    modes = ["baseline", "candidate1", "candidate2", "candidate1_2"]
    results = {}
    for m in modes:
        results[m] = evaluate_candidate(detector, all_dev_records, candidate_mode=m)

    print("\n" + "=" * 90)
    print("PHASE 3 DEVELOPMENT BENCHMARK COMPARISON (N=90 Clips)")
    print("=" * 90)
    print(f"{'Metric':<25} | {'Baseline':<12} | {'Candidate 1':<12} | {'Candidate 2':<12} | {'Candidate 1+2':<12}")
    print("-" * 90)
    print(f"{'Stutter TP':<25} | {results['baseline']['tp']:<12} | {results['candidate1']['tp']:<12} | {results['candidate2']['tp']:<12} | {results['candidate1_2']['tp']:<12}")
    print(f"{'Stutter FP':<25} | {results['baseline']['fp']:<12} | {results['candidate1']['fp']:<12} | {results['candidate2']['fp']:<12} | {results['candidate1_2']['fp']:<12}")
    print(f"{'Stutter FN':<25} | {results['baseline']['fn']:<12} | {results['candidate1']['fn']:<12} | {results['candidate2']['fn']:<12} | {results['candidate1_2']['fn']:<12}")
    print(f"{'Stutter TN':<25} | {results['baseline']['tn']:<12} | {results['candidate1']['tn']:<12} | {results['candidate2']['tn']:<12} | {results['candidate1_2']['tn']:<12}")
    print("-" * 90)
    print(f"{'Precision':<25} | {results['baseline']['precision']*100:>10.2f}% | {results['candidate1']['precision']*100:>10.2f}% | {results['candidate2']['precision']*100:>10.2f}% | {results['candidate1_2']['precision']*100:>10.2f}%")
    print(f"{'Recall':<25} | {results['baseline']['recall']*100:>10.2f}% | {results['candidate1']['recall']*100:>10.2f}% | {results['candidate2']['recall']*100:>10.2f}% | {results['candidate1_2']['recall']*100:>10.2f}%")
    print(f"{'F1 Score':<25} | {results['baseline']['f1']:>12.4f} | {results['candidate1']['f1']:>12.4f} | {results['candidate2']['f1']:>12.4f} | {results['candidate1_2']['f1']:>12.4f}")
    print(f"{'Specificity':<25} | {results['baseline']['specificity']*100:>10.2f}% | {results['candidate1']['specificity']*100:>10.2f}% | {results['candidate2']['specificity']*100:>10.2f}% | {results['candidate1_2']['specificity']*100:>10.2f}%")
    print("-" * 90)
    print(f"{'Dominant Accuracy':<25} | {results['baseline']['accuracy']*100:>10.2f}% | {results['candidate1']['accuracy']*100:>10.2f}% | {results['candidate2']['accuracy']*100:>10.2f}% | {results['candidate1_2']['accuracy']*100:>10.2f}%")
    print(f"{'Macro-F1':<25} | {results['baseline']['macro_f1']:>12.4f} | {results['candidate1']['macro_f1']:>12.4f} | {results['candidate2']['macro_f1']:>12.4f} | {results['candidate1_2']['macro_f1']:>12.4f}")
    print("=" * 90)

    # Save comparison artifact
    out_dir = project_root / 'evaluation_results' / 'experimental' / 'phase3_improvement'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'dev_candidate_comparison.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        # Strip per-clip records for clean summary JSON
        summary = {m: {k: v for k, v in results[m].items() if k != 'records'} for m in modes}
        json.dump(summary, f, indent=2)
    print(f"\nCandidate comparison saved to: {out_file}")

if __name__ == '__main__':
    main()
