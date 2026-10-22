import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
from backend.experimental.config import StutterDetectorConfig, SpeechContextConfig
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.speech_context import SpeechContextAnalyzer

# Canonical Phase 2 Detector from diagnose_dev_errors.py
context_analyzer = SpeechContextAnalyzer(
    SpeechContextConfig(enabled=True, enable_lexical_syntactic_context=True)
)
cfg_canon_p2 = StutterDetectorConfig(
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
    interrupted_repetition_confidence=0.85,
    enable_multi_iteration_filler_stutter=False,
    enable_asr_loop_guard=False
)
det_canon_p2 = HybridStutterDetector(
    config=cfg_canon_p2,
    context_analyzer=context_analyzer
)

# Canonical Phase 4C.3 Candidate (Same canonical pipeline + Loop Guard + Multi-iteration hesitation)
cfg_canon_p4c3 = StutterDetectorConfig(
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
    interrupted_repetition_confidence=0.85,
    enable_multi_iteration_filler_stutter=True,
    enable_asr_loop_guard=True
)
det_canon_p4c3 = HybridStutterDetector(
    config=cfg_canon_p4c3,
    context_analyzer=context_analyzer
)

dev30_cache = json.load(open('evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json', encoding='utf-8'))
gen60_manifest = json.load(open('evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json', encoding='utf-8'))
gen60_cache = json.load(open('evaluation_results/experimental/generalization_validation/run_20261020_132214/feature_cache.json', encoding='utf-8'))

records = []
for fn, c in dev30_cache.items():
    records.append({
        'filename': fn,
        'dataset': c.get('dataset', 'UCLASS'),
        'split': 'dev_30',
        'gt_dominant': c['gt_dominant'],
        'gt_has_stutter': c['gt_has_stutter'],
        'file_path': c.get('file_path') or c.get('filepath'),
        'transcript': c.get('transcript', ''),
        'words_list': c.get('words_list', []),
        'pred_filler': 1 if c.get('gt_dominant') == 'Filler' or c.get('pred_has_filler', 0) == 1 else 0
    })

gen60_map = {p['filename']: p for p in gen60_manifest}
for fn, m in gen60_map.items():
    c = gen60_cache.get(fn, {})
    records.append({
        'filename': fn,
        'dataset': m['dataset'],
        'split': 'gen_60',
        'gt_dominant': m['gt_dominant'],
        'gt_has_stutter': m['gt_has_stutter'],
        'file_path': m.get('file_path') or m.get('filepath'),
        'transcript': c.get('transcript', ''),
        'words_list': c.get('words_list', []),
        'pred_filler': 1 if m.get('gt_dominant') == 'Filler' or c.get('pred_has_filler', 0) == 1 else 0
    })

results = []
for r in records:
    r_p2 = det_canon_p2.evaluate(r['transcript'], r['words_list'], audio_path=r['file_path'], pred_has_filler=r['pred_filler'])
    r_p4c3 = det_canon_p4c3.evaluate(r['transcript'], r['words_list'], audio_path=r['file_path'], pred_has_filler=r['pred_filler'])
    results.append({
        'filename': r['filename'],
        'gt': r['gt_has_stutter'],
        'p2': r_p2['has_stutter'],
        'p4c3': r_p4c3['has_stutter']
    })

def print_bin(name, p_key):
    tp = sum(1 for r in results if r['gt'] == 1 and r[p_key] == 1)
    fp = sum(1 for r in results if r['gt'] == 0 and r[p_key] == 1)
    fn = sum(1 for r in records if r['gt_has_stutter'] == 1 and [x for x in results if x['filename'] == r['filename']][0][p_key] == 0)
    tn = sum(1 for r in records if r['gt_has_stutter'] == 0 and [x for x in results if x['filename'] == r['filename']][0][p_key] == 0)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"\n--- {name} (Dev90, N=90) ---")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision:   {p:.4f}")
    print(f"  Recall:      {r:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Specificity: {spec:.4f}")

print_bin("Canonical Phase 2 (Re-evaluated with Context)", "p2")
print_bin("Canonical Phase 4C.3 Candidate (with Context & Loop Guard)", "p4c3")

diffs = [r for r in results if r['p2'] != r['p4c3']]
print(f"\nTotal differences between Canonical P2 and Canonical P4C.3: {len(diffs)}")
for d in diffs:
    print(f"  {d['filename']}: GT={d['gt']} | P2={d['p2']} -> P4C3={d['p4c3']}")
