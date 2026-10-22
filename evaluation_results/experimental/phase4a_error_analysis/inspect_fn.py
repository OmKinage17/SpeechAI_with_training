import json
from pathlib import Path

diag_path = Path('evaluation_results/experimental/dev_error_diagnostic.json')
with open(diag_path, 'r', encoding='utf-8') as f:
    diag = json.load(f)

fn_list = diag['false_negatives']
print(f"Total False Negatives: {len(fn_list)}")

for i, fn in enumerate(fn_list, 1):
    print(f"[{i}] {fn['filename']} ({fn['dataset']}, Split: {fn.get('source_split')})")
    print(f"    GT Dominant: {fn['gt_dominant']} | Pred Dominant: {fn['pred_dominant']} | Conf: {fn['conf']} | Class: {fn['classification']}")
    print(f"    Transcript: '{fn['transcript']}'")
    print(f"    Detector Events: LexReps={fn['lex_reps']}, Prols={fn['prols']}, Blocks={fn['blocks']}")
    print(f"    Annotations: {fn['annotation_details']}")
    print(f"    Details: {fn['details']}")
    print("-" * 80)
