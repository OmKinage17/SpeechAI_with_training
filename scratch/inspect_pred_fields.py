import json

with open(r'E:\SpeechPrj\evaluation_results\experimental\phase5_benchmark\per_clip_predictions.json', 'r') as f:
    preds = json.load(f)
print('Total prediction records:', len(preds))
print('Keys in first record:', list(preds[0].keys()))

sample = preds[0]
gt_keys = [k for k in sample.keys() if k.startswith('gt_')]
print('GT fields in predictions:', gt_keys)
print('gt_dominant:', sample.get('gt_dominant'))
print('gt_has_stutter:', sample.get('gt_has_stutter'))

any_phrase_rep = any('gt_phrase_rep' in r for r in preds)
print('gt_phrase_rep in any prediction record:', any_phrase_rep)

uclass_preds = [r for r in preds if r.get('dataset') == 'UCLASS']
if uclass_preds:
    print('UCLASS pred has gt_clean field:', 'gt_clean' in uclass_preds[0])

# Check for None values in gt_has_stutter (ambiguous leaked)
none_stutter = [r for r in preds if r.get('gt_has_stutter') is None]
print('Pred records with gt_has_stutter=None:', len(none_stutter))

# Verify datasets in predictions
from collections import Counter
datasets = Counter(r.get('dataset') for r in preds)
print('Datasets in predictions:', dict(datasets))

# Check canonical evaluator field: int(p['gt_has_stutter']) - will fail if None
# Test that all gt_has_stutter are int-coercible
bad = [r for r in preds if not isinstance(r.get('gt_has_stutter'), (int, float))]
print('Records where gt_has_stutter is not numeric:', len(bad))
