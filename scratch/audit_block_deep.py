import json
from collections import Counter

with open('evaluation_results/experimental/final_benchmark/run_final_20261020_complete/per_clip_predictions.json') as f:
    preds = json.load(f)

gt_blocks_filenames = [
    'M_1023_11y3m_1_dysfluent_027.wav',
    'M_1104_40y0m_1_dysfluent_037.wav',
    'M_1104_40y0m_1_dysfluent_143.wav',
    'M_1104_40y0m_1_dysfluent_105.wav',
    'FluencyBank_063_45.wav',
    'FluencyBank_099_150.wav'
]

print('=== Predictions for the 6 GT block clips ===')
for p in preds:
    if p['filename'] in gt_blocks_filenames:
        print(f"{p['filename']} ({p['dataset']}): blocks_pred={p.get('blocks', 0)}, lex_reps={p.get('lex_reps', 0)}, prolongations={p.get('prolongations', 0)}, pred_dominant={p.get('pred_dominant')}, gt_dominant={p.get('gt_dominant')}")

print('\n=== Total clips with blocks > 0 in per_clip_predictions ===')
pred_block_clips = [p for p in preds if p.get('blocks', 0) > 0]
print(f'Count of clips with blocks > 0: {len(pred_block_clips)}')
print(f'Total block events detected: {sum(p.get("blocks", 0) for p in preds)}')

counts = Counter(p['dataset'] for p in pred_block_clips)
print('Dataset breakdown of clips with blocks > 0:', dict(counts))

event_counts = Counter()
for p in preds:
    if p.get('blocks', 0) > 0:
        event_counts[p['dataset']] += p['blocks']
print('Dataset breakdown of block EVENTS:', dict(event_counts))

# Let's check overlap
overlap = [p for p in pred_block_clips if p['filename'] in gt_blocks_filenames]
print(f'Overlap between predicted block clips and GT block clips: {len(overlap)}')
