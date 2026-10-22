import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.config import StutterDetectorConfig

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
        'gt_has_filler': c.get('gt_has_filler', 0),
        'file_path': c.get('file_path') or c.get('filepath'),
        'transcript': c.get('transcript', ''),
        'words_list': c.get('words_list', [])
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
        'gt_has_filler': m.get('gt_has_filler', 0),
        'file_path': m.get('file_path') or m.get('filepath'),
        'transcript': c.get('transcript', ''),
        'words_list': c.get('words_list', [])
    })

print(f"Loaded {len(records)} development clips.")

cfg_p2 = StutterDetectorConfig(
    enable_filler_boundary_gating=False,
    enable_multi_iteration_filler_stutter=False,
    enable_asr_loop_guard=False
)
det_p2 = HybridStutterDetector(config=cfg_p2)

cfg_p4c3 = StutterDetectorConfig(
    enable_filler_boundary_gating=True,
    enable_multi_iteration_filler_stutter=True,
    enable_asr_loop_guard=True
)
det_p4c3 = HybridStutterDetector(config=cfg_p4c3)

results = []
for r in records:
    tx = r['transcript']
    wl = r['words_list']
    fp = r['file_path']
    gt_st = r['gt_has_stutter']
    gt_dom = r['gt_dominant']
    
    r_p2 = det_p2.evaluate(transcript=tx, words_list=wl, audio_path=fp)
    r_p4c3 = det_p4c3.evaluate(transcript=tx, words_list=wl, audio_path=fp)
    
    results.append({
        'filename': r['filename'],
        'dataset': r['dataset'],
        'split': r['split'],
        'gt_dominant': gt_dom,
        'gt_has_stutter': gt_st,
        'gt_has_filler': r['gt_has_filler'],
        'p2_stutter': r_p2['has_stutter'],
        'p2_dom': r_p2['pred_dominant'],
        'p4c3_stutter': r_p4c3['has_stutter'],
        'p4c3_dom': r_p4c3['pred_dominant'],
        'p4c3_gated': len(r_p4c3.get('gated_events', []))
    })

for label, p_key in [("Phase 2 Baseline (Dev90)", "p2_stutter"), ("Phase 4C.3 Candidate (Dev90)", "p4c3_stutter")]:
    tp = sum(1 for r in results if r['gt_has_stutter'] == 1 and r[p_key] == 1)
    fp = sum(1 for r in results if r['gt_has_stutter'] == 0 and r[p_key] == 1)
    fn = sum(1 for r in results if r['gt_has_stutter'] == 1 and r[p_key] == 0)
    tn = sum(1 for r in results if r['gt_has_stutter'] == 0 and r[p_key] == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"\n--- {label} ---")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Specificity: {spec:.4f}")

# Differences
diffs = [r for r in results if r['p2_stutter'] != r['p4c3_stutter']]
print(f"\nTotal clips with prediction changes on Dev90: {len(diffs)}")
for d in diffs:
    print(f"  {d['filename']}: GT={d['gt_has_stutter']} (dom: {d['gt_dominant']}) | P2={d['p2_stutter']} -> P4C3={d['p4c3_stutter']}")
