import sys
sys.path.insert(0, ".")
import json
import csv
from pathlib import Path
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.config import StutterDetectorConfig

# 1. Canonical Phase 2 Baseline
cfg_canon = StutterDetectorConfig(
    enable_filler_boundary_gating=True,
    enable_multi_iteration_filler_stutter=False,
    enable_asr_loop_guard=False
)
det_canon = HybridStutterDetector(config=cfg_canon)

# 2. Phase 4B Candidate
cfg_p4b = StutterDetectorConfig(
    enable_filler_boundary_gating=True,
    enable_multi_iteration_filler_stutter=True,
    enable_asr_loop_guard=False
)
det_p4b = HybridStutterDetector(config=cfg_p4b)

# 3. Phase 4C.3 Candidate
cfg_p4c3 = StutterDetectorConfig(
    enable_filler_boundary_gating=True,
    enable_multi_iteration_filler_stutter=True,
    enable_asr_loop_guard=True
)
det_p4c3 = HybridStutterDetector(config=cfg_p4c3)

cohort_data = json.load(open('evaluation_results/experimental/phase4c2_1_dev_hesitation_cohort/final_curated_pools.json', encoding='utf-8'))
cohort_clips = []
for cat_k, items in cohort_data.items():
    for item in items:
        tx = item['transcription']['sanitized_transcript']
        wl = item['transcription']['words_list']
        fp = item['clip']['filepath']
        stat = 'DEGENERATE_LOOP_RECOVERED' if item['transcription']['is_loop'] else None
        gt_class = item['ground_truth_class']
        gt_dom = "Clean" if gt_class == "Artifact" else gt_class
        gt_st = 1 if gt_dom == "Stutter" else 0
        cohort_clips.append({
            'fn': item['clip']['filename'],
            'cat': cat_k,
            'gt_has_stutter': gt_st,
            'tx': tx, 'wl': wl, 'fp': fp, 'stat': stat
        })

print(f"Total cohort clips: {len(cohort_clips)}")

# Evaluate Cohort
c_res = []
for c in cohort_clips:
    r_canon = det_canon.evaluate(c['tx'], c['wl'], audio_path=c['fp'], sanitizer_status=c['stat'])['has_stutter']
    r_p4b = det_p4b.evaluate(c['tx'], c['wl'], audio_path=c['fp'], sanitizer_status=c['stat'])['has_stutter']
    r_p4c3 = det_p4c3.evaluate(c['tx'], c['wl'], audio_path=c['fp'], sanitizer_status=c['stat'])['has_stutter']
    c_res.append({
        'fn': c['fn'],
        'cat': c['cat'],
        'gt': c['gt_has_stutter'],
        'canon': r_canon,
        'p4b': r_p4b,
        'p4c3': r_p4c3
    })

def print_binary_table(name, records, p_key):
    tp = sum(1 for r in records if r['gt'] == 1 and r[p_key] == 1)
    fp = sum(1 for r in records if r['gt'] == 0 and r[p_key] == 1)
    fn = sum(1 for r in records if r['gt'] == 1 and r[p_key] == 0)
    tn = sum(1 for r in records if r['gt'] == 0 and r[p_key] == 0)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"  [{name:25s}] TP={tp:2d}, FP={fp:2d}, FN={fn:2d}, TN={tn:2d} | P={p:.4f}, R={r:.4f}, F1={f1:.4f}, Spec={spec:.4f}")

print("\n--- COHORT (N=40) RECONCILED PERFORMANCE ---")
print_binary_table("Canonical Phase 2", c_res, "canon")
print_binary_table("Phase 4B Candidate", c_res, "p4b")
print_binary_table("Phase 4C.3 Candidate", c_res, "p4c3")

print("\nCohort Cat 3 (Genuine Stutter, N=10):")
cat3 = [r for r in c_res if r['cat'] == 'cat3']
print_binary_table("Canonical Phase 2", cat3, "canon")
print_binary_table("Phase 4B Candidate", cat3, "p4b")
print_binary_table("Phase 4C.3 Candidate", cat3, "p4c3")

print("\nCohort Cat 4 (ASR Loop Artifact, N=10):")
cat4 = [r for r in c_res if r['cat'] == 'cat4']
print_binary_table("Canonical Phase 2", cat4, "canon")
print_binary_table("Phase 4B Candidate", cat4, "p4b")
print_binary_table("Phase 4C.3 Candidate", cat4, "p4c3")
