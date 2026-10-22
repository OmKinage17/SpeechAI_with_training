import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.config import StutterDetectorConfig

data = json.load(open('evaluation_results/experimental/phase4c2_1_dev_hesitation_cohort/final_curated_pools.json', encoding='utf-8'))

cfg_phase2 = StutterDetectorConfig(
    enable_filler_boundary_gating=False,
    enable_multi_iteration_filler_stutter=False,
    enable_asr_loop_guard=False
)
det_phase2 = HybridStutterDetector(config=cfg_phase2)

cfg_phase4b = StutterDetectorConfig(
    enable_filler_boundary_gating=True,
    enable_multi_iteration_filler_stutter=True,
    enable_asr_loop_guard=False
)
det_phase4b = HybridStutterDetector(config=cfg_phase4b)

cfg_phase4c3 = StutterDetectorConfig(
    enable_filler_boundary_gating=True,
    enable_multi_iteration_filler_stutter=True,
    enable_asr_loop_guard=True
)
det_phase4c3 = HybridStutterDetector(config=cfg_phase4c3)

print("--- EVALUATION PREVIEW ACROSS 4 COHORT CATEGORIES ---")
for cat_name, items in data.items():
    print(f"\nCategory: {cat_name} (N={len(items)})")
    for i, item in enumerate(items):
        fn = item['clip']['filename']
        san_tx = item['transcription']['sanitized_transcript']
        wlist = item['transcription']['words_list']
        san_stat = 'DEGENERATE_LOOP_RECOVERED' if item['transcription']['is_loop'] else None
        
        # In actual pipeline, if words_list is passed, it is sanitized words or raw words:
        r2 = det_phase2.evaluate(transcript=san_tx, words_list=wlist, sanitizer_status=san_stat)
        r4b = det_phase4b.evaluate(transcript=san_tx, words_list=wlist, sanitizer_status=san_stat)
        r4c = det_phase4c3.evaluate(transcript=san_tx, words_list=wlist, sanitizer_status=san_stat)
        
        gt = item['ground_truth_class']
        p2 = r2['has_stutter']
        p4b = r4b['has_stutter']
        p4c = r4c['has_stutter']
        
        print(f"[{i+1:02d}] {fn[:25]:25s} | GT:{gt:7s} | P2:{p2} | P4B:{p4b} | P4C3:{p4c}")
