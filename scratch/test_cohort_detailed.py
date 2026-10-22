import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from collections import Counter
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.config import StutterDetectorConfig

data = json.load(open('evaluation_results/experimental/phase4c2_1_dev_hesitation_cohort/final_curated_pools.json', encoding='utf-8'))

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
for cat_name, items in data.items():
    for item in items:
        fn = item['clip']['filename']
        san_tx = item['transcription']['sanitized_transcript']
        wlist = item['transcription']['words_list']
        is_loop = item['transcription']['is_loop']
        san_stat = 'DEGENERATE_LOOP_RECOVERED' if is_loop else None
        audio_p = item['clip']['filepath']
        gt_dom = item['ground_truth_class']
        gt_st = 1 if gt_dom == 'Stutter' else 0
        
        # Audio is available, so evaluate full hybrid
        r_p2 = det_p2.evaluate(transcript=san_tx, words_list=wlist, audio_path=audio_p, sanitizer_status=san_stat)
        r_p4c3 = det_p4c3.evaluate(transcript=san_tx, words_list=wlist, audio_path=audio_p, sanitizer_status=san_stat)
        
        results.append({
            'filename': fn,
            'category': cat_name,
            'gt_dominant': gt_dom,
            'gt_has_stutter': gt_st,
            'p2_stutter': r_p2['has_stutter'],
            'p2_dom': r_p2['pred_dominant'],
            'p2_conf': r_p2['confidence_score'],
            'p2_events': r_p2['total_stutter_events'],
            'p4c3_stutter': r_p4c3['has_stutter'],
            'p4c3_dom': r_p4c3['pred_dominant'],
            'p4c3_conf': r_p4c3['confidence_score'],
            'p4c3_events': r_p4c3['total_stutter_events'],
            'p4c3_gated': len(r_p4c3.get('gated_events', []))
        })

print(f"Total cohort evaluated: {len(results)}")

for cat in ['cat1', 'cat2', 'cat3', 'cat4']:
    cat_res = [r for r in results if r['category'] == cat]
    p2_fps = sum(1 for r in cat_res if r['gt_has_stutter'] == 0 and r['p2_stutter'] == 1)
    p4c3_fps = sum(1 for r in cat_res if r['gt_has_stutter'] == 0 and r['p4c3_stutter'] == 1)
    p2_tps = sum(1 for r in cat_res if r['gt_has_stutter'] == 1 and r['p2_stutter'] == 1)
    p4c3_tps = sum(1 for r in cat_res if r['gt_has_stutter'] == 1 and r['p4c3_stutter'] == 1)
    print(f"\n[{cat}] N={len(cat_res)}")
    print(f"  P2 Baseline:   TP={p2_tps}, FP={p2_fps}")
    print(f"  P4C3 Candidate: TP={p4c3_tps}, FP={p4c3_fps}")

# Aggregate Cohort Binary Metrics
for label, p_key in [("Phase 2 Baseline", "p2_stutter"), ("Phase 4C.3 Candidate", "p4c3_stutter")]:
    tp = sum(1 for r in results if r['gt_has_stutter'] == 1 and r[p_key] == 1)
    fp = sum(1 for r in results if r['gt_has_stutter'] == 0 and r[p_key] == 1)
    fn = sum(1 for r in results if r['gt_has_stutter'] == 1 and r[p_key] == 0)
    tn = sum(1 for r in results if r['gt_has_stutter'] == 0 and r[p_key] == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"\n--- Cohort Aggregate ({label}) ---")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Specificity: {spec:.4f}")
