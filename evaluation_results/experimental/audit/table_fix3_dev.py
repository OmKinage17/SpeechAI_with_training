import json
import os
import sys

audit_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(audit_dir)))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open(os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips_fix3.json"), "r") as f:
    fix3_data = json.load(f)

print(f"{'Filename':<35} | {'Dataset':<12} | {'GT':<7} | {'Fix3':<7} | {'Conf':<6} | {'Details':<30}")
print("-" * 110)
for p in fix3_data["predictions_fix3"]:
    fname = p["filename"]
    ds = p["dataset"]
    gt = p["gt_dominant"]
    pred = p["pred_dominant"]
    conf = p["confidence"]
    prol_events = sum(1 for d in p["details"] if d.get("type") == "acoustic_prolongation")
    lex_events = sum(1 for d in p["details"] if "repetition" in d.get("type", ""))
    block_events = sum(1 for d in p["details"] if "block" in d.get("type", ""))
    summary = f"prol={prol_events}, lex={lex_events}, blk={block_events}"
    mismatch = " <--- MISMATCH" if gt != pred else ""
    print(f"{fname:<35} | {ds:<12} | {gt:<7} | {pred:<7} | {conf:<6.2f} | {summary:<30}{mismatch}")
