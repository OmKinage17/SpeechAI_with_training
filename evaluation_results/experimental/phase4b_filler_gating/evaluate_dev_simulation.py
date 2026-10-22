import json
import re
from pathlib import Path
from typing import Dict, Any, List

# Load dev caches and manifests
dev30_cache_file = Path('evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json')
gen60_manifest_file = Path('evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json')
gen60_cache_file = Path('evaluation_results/experimental/generalization_validation/run_20261020_132214/feature_cache.json')

with open(dev30_cache_file, 'r', encoding='utf-8') as f:
    dev30_cache = json.load(f)
with open(gen60_manifest_file, 'r', encoding='utf-8') as f:
    gen60_manifest = {p['filename']: p for p in json.load(f)}
with open(gen60_cache_file, 'r', encoding='utf-8') as f:
    gen60_cache = json.load(f)

# Load diagnostic baseline predictions
with open('evaluation_results/experimental/dev_error_diagnostic.json', 'r', encoding='utf-8') as f:
    diag = json.load(f)
diag_fn = {x['filename']: x for x in diag['false_negatives']}
diag_fp = {x['filename']: x for x in diag['false_positives']}

import sys
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evaluation_results.experimental.phase4b_filler_gating.test_concept import analyze_filler_streaks

# Evaluate all 90 dev clips
all_dev_records = []
for fn, c in dev30_cache.items():
    all_dev_records.append({
        'filename': fn,
        'dataset': c.get('dataset', 'UCLASS'),
        'split': 'dev_30',
        'gt_dominant': c['gt_dominant'],
        'gt_has_stutter': c['gt_has_stutter'],
        'transcript': c.get('transcript', ''),
        'words_list': c.get('words_list', [])
    })

for fn, m in gen60_manifest.items():
    c = gen60_cache.get(fn, {})
    all_dev_records.append({
        'filename': fn,
        'dataset': m['dataset'],
        'split': 'gen_60',
        'gt_dominant': m['gt_dominant'],
        'gt_has_stutter': m['gt_has_stutter'],
        'transcript': c.get('transcript', ''),
        'words_list': c.get('words_list', [])
    })

print(f"Total dev clips to evaluate: {len(all_dev_records)}")

# Compare Baseline vs Candidate
baseline_preds = []
candidate_preds = []
diff_clips = []

for r in all_dev_records:
    fn = r['filename']
    gt_st = r['gt_has_stutter']
    tx = r['transcript']
    wlist = r['words_list']
    
    # Baseline prediction:
    # If it was in diag false negatives, baseline_st = 0
    # If it was in diag false positives, baseline_st = 1
    # Otherwise: baseline_st == gt_st
    if fn in diag_fn:
        base_st = 0
    elif fn in diag_fp:
        base_st = 1
    else:
        base_st = gt_st
    baseline_preds.append(base_st)
    
    # Candidate evaluation:
    # Run multi-iteration hesitation struggle analysis
    streaks = analyze_filler_streaks(wlist, tx)
    cand_stutter_events = [s for s in streaks if s['is_stutter_candidate']]
    
    cand_st = base_st
    if cand_stutter_events:
        max_conf = max(s['confidence'] for s in cand_stutter_events)
        if max_conf >= 0.60:
            cand_st = 1
            if base_st != 1:
                diff_clips.append({
                    'filename': fn,
                    'gt_has_stutter': gt_st,
                    'gt_dominant': r['gt_dominant'],
                    'base_st': base_st,
                    'cand_st': cand_st,
                    'events': cand_stutter_events,
                    'transcript': tx
                })
    candidate_preds.append(cand_st)

print(f"Total clips flipped by candidate: {len(diff_clips)}")
for d in diff_clips:
    print(f"Clip: {d['filename']} ({d['gt_dominant']})")
    print(f"  GT: {d['gt_has_stutter']} | Baseline: {d['base_st']} -> Candidate: {d['cand_st']}")
    print(f"  Events: {d['events']}")
    print(f"  Transcript: '{d['transcript'][:100]}'")
    print("-" * 60)

# Compute metrics for Baseline
def get_metrics(preds, records):
    tp = sum(1 for p, r in zip(preds, records) if r['gt_has_stutter'] == 1 and p == 1)
    fp = sum(1 for p, r in zip(preds, records) if r['gt_has_stutter'] == 0 and p == 1)
    fn = sum(1 for p, r in zip(preds, records) if r['gt_has_stutter'] == 1 and p == 0)
    tn = sum(1 for p, r in zip(preds, records) if r['gt_has_stutter'] == 0 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'prec': prec, 'rec': rec, 'f1': f1, 'spec': spec}

base_m = get_metrics(baseline_preds, all_dev_records)
cand_m = get_metrics(candidate_preds, all_dev_records)

print("\n=== METRIC COMPARISON (N=90) ===")
print(f"{'Metric':<15} | {'Baseline':<12} | {'Candidate':<12} | {'Delta':<12}")
print("-" * 55)
for k in ('tp', 'fp', 'fn', 'tn'):
    print(f"{k.upper():<15} | {base_m[k]:<12} | {cand_m[k]:<12} | {cand_m[k]-base_m[k]:+d}")
for k, name in [('prec', 'Precision'), ('rec', 'Recall'), ('f1', 'F1 Score'), ('spec', 'Specificity')]:
    print(f"{name:<15} | {base_m[k]*100:>10.2f}% | {cand_m[k]*100:>10.2f}% | {(cand_m[k]-base_m[k])*100:>+10.2f}%")
