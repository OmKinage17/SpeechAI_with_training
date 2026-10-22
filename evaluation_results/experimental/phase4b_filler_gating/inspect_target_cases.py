import json
import re
from pathlib import Path

# Load dev clips
dev30_cache_file = Path('evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json')
gen60_manifest_file = Path('evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json')
gen60_cache_file = Path('evaluation_results/experimental/generalization_validation/run_20261020_132214/feature_cache.json')

with open(dev30_cache_file, 'r', encoding='utf-8') as f:
    dev30_cache = json.load(f)
with open(gen60_manifest_file, 'r', encoding='utf-8') as f:
    gen60_manifest = {p['filename']: p for p in json.load(f)}
with open(gen60_cache_file, 'r', encoding='utf-8') as f:
    gen60_cache = json.load(f)

all_dev = []
for fn, c in dev30_cache.items():
    all_dev.append({
        'filename': fn,
        'dataset': c.get('dataset', 'UCLASS'),
        'split': 'dev_30',
        'gt_dominant': c.get('gt_dominant'),
        'gt_has_stutter': c.get('gt_has_stutter'),
        'gt_has_filler': c.get('gt_has_filler', 0),
        'annotation_details': c.get('annotation_details', {}),
        'transcript': c.get('transcript', ''),
        'words_list': c.get('words_list', []),
        'file_path': c.get('file_path')
    })

for fn, m in gen60_manifest.items():
    c = gen60_cache.get(fn, {})
    all_dev.append({
        'filename': fn,
        'dataset': m.get('dataset'),
        'split': 'gen_60',
        'gt_dominant': m.get('gt_dominant'),
        'gt_has_stutter': m.get('gt_has_stutter'),
        'gt_has_filler': m.get('gt_has_filler', 0),
        'annotation_details': m.get('annotation_details', {}),
        'transcript': c.get('transcript', ''),
        'words_list': c.get('words_list', []),
        'file_path': m.get('file_path')
    })

FILLER_WORDS = {"um", "uh", "er", "ah", "hmm", "like", "well", "you know"}

def clean_tok(w):
    return re.sub(r'[^a-zA-Z]', '', w).lower()

# Check target cases and any cases with repeated fillers
repeated_filler_clips = []
for r in all_dev:
    tx = r['transcript']
    tokens = [clean_tok(w) for w in tx.split() if clean_tok(w)]
    
    # scan for consecutive fillers
    filler_streaks = []
    curr_filler = None
    curr_streak = 0
    
    for t in tokens:
        if t in FILLER_WORDS:
            if t == curr_filler:
                curr_streak += 1
            else:
                if curr_streak >= 2:
                    filler_streaks.append((curr_filler, curr_streak))
                curr_filler = t
                curr_streak = 1
        else:
            if curr_streak >= 2:
                filler_streaks.append((curr_filler, curr_streak))
            curr_filler = None
            curr_streak = 0
    if curr_streak >= 2:
        filler_streaks.append((curr_filler, curr_streak))
        
    if filler_streaks or r['filename'] in ('M_0030_16y4m_1_dysfluent_015.wav', 'M_0030_16y4m_1_dysfluent_009.wav', 'FluencyBank_017_15.wav'):
        repeated_filler_clips.append({
            'clip': r['filename'],
            'dataset': r['dataset'],
            'split': r['split'],
            'gt_dominant': r['gt_dominant'],
            'gt_has_stutter': r['gt_has_stutter'],
            'gt_has_filler': r['gt_has_filler'],
            'annotation_details': r['annotation_details'],
            'transcript': tx,
            'filler_streaks': filler_streaks
        })

print(f"Total dev clips: {len(all_dev)}")
print(f"Found {len(repeated_filler_clips)} clips with repeated filler streaks or target names:\n")
for c in repeated_filler_clips:
    print(f"Clip: {c['clip']} ({c['dataset']}, {c['split']})")
    print(f"  GT Dominant: {c['gt_dominant']} | Has Stutter: {c['gt_has_stutter']} | Has Filler: {c['gt_has_filler']}")
    print(f"  Annotations: {c['annotation_details']}")
    print(f"  Filler streaks: {c['filler_streaks']}")
    print(f"  Transcript: '{c['transcript'][:120]}...'")
    print("-" * 80)
