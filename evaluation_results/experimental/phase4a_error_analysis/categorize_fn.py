import json
from pathlib import Path

inventory_path = Path('evaluation_results/experimental/phase4a_error_analysis/fn_detailed_inventory.json')
with open(inventory_path, 'r', encoding='utf-8') as f:
    records = json.load(f)

print(f"Loaded {len(records)} FN records.")

for r in records:
    dur_str = f"{r['duration']:.2f}s" if r['duration'] else "N/A"
    print(f"=== {r['filename']} | {r['dataset']} | Split: {r['split']} | Dur: {dur_str} ===")
    print(f"  Annotations: {r['annotation_details']}")
    print(f"  Raw Tx:   '{r['raw_transcript']}'")
    print(f"  Sanit Tx: '{r['sanitized_transcript']}'")
    if r['words_list']:
        w_summary = ' '.join([f"{w.get('word')}({w.get('start', 0.0):.2f}-{w.get('end', 0.0):.2f})" for w in r['words_list'][:10]])
        print(f"  Words ({len(r['words_list'])} total): {w_summary}")
    else:
        print("  Words: [EMPTY]")
    print()
