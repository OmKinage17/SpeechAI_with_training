"""
SpeechAI — Phase 3 Development Error Diagnostic Script
======================================================
Evaluates the frozen Phase 2 detector exclusively across development data (N=90):
- 30-clip standardized dev set
- 60-clip generalization validation set
Zero evaluation-set leakage: 0 clips, 0 speakers from final 120-clip evaluation set.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import json
from collections import Counter
from backend.experimental.config import StutterDetectorConfig, SpeechContextConfig
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.acoustic_features import AcousticFeatureExtractor
from backend.experimental.speech_context import SpeechContextAnalyzer

def run_diagnostic():
    # Setup frozen detector with exact Phase 2 parameters
    acoustic_extractor = AcousticFeatureExtractor()
    context_analyzer = SpeechContextAnalyzer(
        SpeechContextConfig(enabled=True, enable_lexical_syntactic_context=True)
    )
    stutter_config = StutterDetectorConfig(
        enable_contextual_scoring=True,
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

    # 1. Dev 30 (from standardized_dev feature cache)
    dev30_cache_file = project_root / 'evaluation_results' / 'experimental' / 'standardized_dev' / 'run_20261020_122136' / 'feature_cache.json'
    with open(dev30_cache_file, 'r', encoding='utf-8') as f:
        dev30_cache = json.load(f)

    # 2. Gen 60
    gen60_manifest_file = project_root / 'evaluation_results' / 'experimental' / 'generalization_validation' / 'run_20261020_132214' / 'run_manifest.json'
    gen60_cache_file = project_root / 'evaluation_results' / 'experimental' / 'generalization_validation' / 'run_20261020_132214' / 'feature_cache.json'
    with open(gen60_manifest_file, 'r', encoding='utf-8') as f:
        gen60_manifest = json.load(f)
    with open(gen60_cache_file, 'r', encoding='utf-8') as f:
        gen60_cache = json.load(f)

    all_dev_records = []

    for fn, c in dev30_cache.items():
        audio_path = c.get('file_path')
        transcript = c.get('transcript', '')
        words_list = c.get('words_list', [])
        pred_filler = 1 if c.get('gt_dominant') == 'Filler' or c.get('pred_has_filler', 0) == 1 else 0
        res = detector.evaluate(transcript, words_list, audio_path=audio_path, pred_has_filler=pred_filler)
        all_dev_records.append({
            'source_split': 'dev_30',
            'filename': fn,
            'dataset': c.get('dataset', 'UCLASS'),
            'gt_dominant': c['gt_dominant'],
            'gt_has_stutter': c['gt_has_stutter'],
            'annotation_details': c.get('annotation_details', {}),
            'transcript': transcript,
            'pred_dominant': res['pred_dominant'],
            'pred_has_stutter': res['has_stutter'],
            'lex_reps': res['lexical_repetition_events'],
            'prols': res['acoustic_prolongation_events'],
            'blocks': res['acoustic_block_events'],
            'conf': res['confidence_score'],
            'classification': res['classification'],
            'details': res['details']
        })

    for p in gen60_manifest:
        fn = p['filename']
        c = gen60_cache.get(fn, {})
        audio_path = p['file_path']
        transcript = c.get('transcript', '')
        words_list = c.get('words_list', [])
        pred_filler = 1 if p.get('gt_dominant') == 'Filler' or c.get('pred_has_filler', 0) == 1 else 0
        res = detector.evaluate(transcript, words_list, audio_path=audio_path, pred_has_filler=pred_filler)
        all_dev_records.append({
            'source_split': 'gen_60',
            'filename': fn,
            'dataset': p['dataset'],
            'gt_dominant': p['gt_dominant'],
            'gt_has_stutter': p['gt_has_stutter'],
            'annotation_details': p.get('annotation_details', {}),
            'transcript': transcript,
            'pred_dominant': res['pred_dominant'],
            'pred_has_stutter': res['has_stutter'],
            'lex_reps': res['lexical_repetition_events'],
            'prols': res['acoustic_prolongation_events'],
            'blocks': res['acoustic_block_events'],
            'conf': res['confidence_score'],
            'classification': res['classification'],
            'details': res['details']
        })

    print(f"Total development clips evaluated: {len(all_dev_records)}")

    # Binary stutter metrics
    st_tp = [r for r in all_dev_records if r['gt_has_stutter'] == 1 and r['pred_has_stutter'] == 1]
    st_fp = [r for r in all_dev_records if r['gt_has_stutter'] == 0 and r['pred_has_stutter'] == 1]
    st_fn = [r for r in all_dev_records if r['gt_has_stutter'] == 1 and r['pred_has_stutter'] == 0]
    st_tn = [r for r in all_dev_records if r['gt_has_stutter'] == 0 and r['pred_has_stutter'] == 0]

    prec = len(st_tp) / (len(st_tp) + len(st_fp)) if (st_tp or st_fp) else 0.0
    rec = len(st_tp) / (len(st_tp) + len(st_fn)) if (st_tp or st_fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = len(st_tn) / (len(st_tn) + len(st_fp)) if (st_tn or st_fp) else 0.0

    print("\n" + "=" * 60)
    print("DEVELOPMENT BENCHMARK PERFORMANCE (N=90 Clips)")
    print("=" * 60)
    print(f"Binary Stutter (P1): TP={len(st_tp)}, FP={len(st_fp)}, FN={len(st_fn)}, TN={len(st_tn)} (Total Stutter Support={len(st_tp)+len(st_fn)})")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Specificity:{spec:.4f} ({spec*100:.2f}%)")

    # Multiclass confusion matrix
    classes = ['Stutter', 'Filler', 'Clean']
    cm = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for r in all_dev_records:
        cm[r['gt_dominant']][r['pred_dominant']] += 1

    print("\nMulticlass Confusion Matrix (Dominant Label, N=90):")
    header = f"{'':12s} | " + " | ".join(f"{c:10s}" for c in classes)
    print(header)
    print("-" * len(header))
    for c in classes:
        row = f"GT_{c:9s} | " + " | ".join(f"{cm[c][pred]:10d}" for pred in classes)
        print(row)

    # Detailed False Negative Analysis
    print("\n" + "=" * 60)
    print(f"FALSE NEGATIVE ANALYSIS (Missed Stuttering Events, N={len(st_fn)})")
    print("=" * 60)
    fn_subtypes = Counter()
    for r in st_fn:
        ann = r['annotation_details']
        annotated_types = [k for k, v in ann.items() if (isinstance(v, (int, float)) and v > 0) or (isinstance(v, list) and len(v) > 0)]
        fn_subtypes.update(annotated_types)
        print(f"\nClip: {r['filename']} ({r['dataset']}, Split: {r['source_split']})")
        print(f"  GT Dominant: {r['gt_dominant']} | Predicted: {r['pred_dominant']} (Conf: {r['conf']}, Class: {r['classification']})")
        print(f"  Annotated Subtypes: {annotated_types}")
        print(f"  Transcript: '{r['transcript']}'")
        print(f"  Detector Events: LexReps={r['lex_reps']}, Prols={r['prols']}, Blocks={r['blocks']}")
        if r['details']:
            print(f"  Uncorroborated Detections in Clip: {r['details']}")

    print(f"\nBreakdown of annotated dysfluency subtypes in False Negatives: {dict(fn_subtypes)}")

    # Detailed False Positive Analysis
    print("\n" + "=" * 60)
    print(f"FALSE POSITIVE ANALYSIS (Spurious Stuttering Events, N={len(st_fp)})")
    print("=" * 60)
    for r in st_fp:
        print(f"\nClip: {r['filename']} ({r['dataset']}, Split: {r['source_split']})")
        print(f"  GT Dominant: {r['gt_dominant']} | Predicted: {r['pred_dominant']} (Conf: {r['conf']}, Class: {r['classification']})")
        print(f"  Transcript: '{r['transcript']}'")
        print(f"  Trigger Details: LexReps={r['lex_reps']}, Prols={r['prols']}, Blocks={r['blocks']}")
        print(f"  Detection Details: {r['details']}")

    # Save Diagnostic JSON
    diag_out = project_root / 'evaluation_results' / 'experimental' / 'dev_error_diagnostic.json'
    with open(diag_out, 'w', encoding='utf-8') as f:
        json.dump({
            'total_dev_clips': len(all_dev_records),
            'metrics': {'precision': prec, 'recall': rec, 'f1': f1, 'specificity': spec, 'tp': len(st_tp), 'fp': len(st_fp), 'fn': len(st_fn), 'tn': len(st_tn)},
            'confusion_matrix': cm,
            'false_negatives': st_fn,
            'false_positives': st_fp
        }, f, indent=2)
    print(f"\nDiagnostic report saved to: {diag_out}")

if __name__ == '__main__':
    run_diagnostic()
