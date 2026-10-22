import json
import csv
from pathlib import Path

# Paths
root = Path("e:/SpeechPrj")
json_path = root / "evaluation_results/experimental/phase4c3_fasttrack/phase4c3_metrics_comparison.json"
csv_path = root / "evaluation_results/experimental/phase4c3_fasttrack/per_clip_changes.csv"
report_path = root / "evaluation_results/experimental/phase4c3_fasttrack/PHASE4C3_FASTTRACK_REPORT.md"
test_log_path = root / "evaluation_results/experimental/phase4c3_fasttrack/test_results.txt"

print("--- 1. AUDITING JSON METRICS ---")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def audit_binary_block(name, block):
    tp = block["tp"]
    fp = block["fp"]
    fn = block["fn"]
    tn = block["tn"]
    total = tp + fp + fn + tn
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    rep_p = block["precision"]
    rep_r = block["recall"]
    rep_f1 = block["f1"]
    rep_spec = block["specificity"]
    
    p_diff = abs(p - rep_p)
    r_diff = abs(r - rep_r)
    f1_diff = abs(f1 - rep_f1)
    spec_diff = abs(spec - rep_spec)
    
    is_ok = (p_diff < 1e-4 and r_diff < 1e-4 and f1_diff < 1e-4 and spec_diff < 1e-4)
    print(f"[{name}] N={total}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"   Calc: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, Spec={spec:.4f}")
    print(f"   Rep:  P={rep_p:.4f}, R={rep_r:.4f}, F1={rep_f1:.4f}, Spec={rep_spec:.4f} -> OK: {is_ok}")
    return is_ok, total

print("\n--- COHORT ---")
audit_binary_block("Cohort P2", data["cohort_metrics"]["phase2_baseline"]["binary_stutter"])
audit_binary_block("Cohort P4C3", data["cohort_metrics"]["phase4c3_candidate"]["binary_stutter"])

print("\n--- DEV90 ---")
audit_binary_block("Dev90 P2", data["dev90_metrics"]["phase2_baseline"]["binary_stutter"])
audit_binary_block("Dev90 P4C3", data["dev90_metrics"]["phase4c3_candidate"]["binary_stutter"])

print("\n--- COMBINED 130 ---")
audit_binary_block("Combined P2", data["combined_metrics"]["phase2_baseline"]["binary_stutter"])
audit_binary_block("Combined P4C3", data["combined_metrics"]["phase4c3_candidate"]["binary_stutter"])

print("\n--- 2. AUDITING PER_CLIP_CHANGES.CSV ---")
changes = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        changes.append(row)

print(f"Total rows in CSV: {len(changes)}")
by_split = {}
by_category = {}
for c in changes:
    s = c["split"]
    cat = c["change_category"]
    by_split[s] = by_split.get(s, 0) + 1
    by_category[cat] = by_category.get(cat, 0) + 1

print(f"By split: {by_split}")
print(f"By change category: {by_category}")

print("\nDetail of all 20 changes:")
for i, c in enumerate(changes):
    gt = c["gt_has_stutter"]
    p2 = c["p2_has_stutter"]
    p4 = c["p4c3_has_stutter"]
    print(f"{i+1:02d}. {c['clip_id']} ({c['split']}): GT={gt}, P2={p2} -> P4C3={p4} | Cat={c['change_category']}")
