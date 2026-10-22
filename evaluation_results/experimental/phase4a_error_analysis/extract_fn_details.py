import json
import os
from pathlib import Path
import soundfile as sf
import numpy as np

# Load diagnostic
diag_path = Path('evaluation_results/experimental/dev_error_diagnostic.json')
with open(diag_path, 'r', encoding='utf-8') as f:
    diag = json.load(f)

fn_list = diag['false_negatives']

# Load caches
dev30_cache_file = Path('evaluation_results/experimental/standardized_dev/run_20261020_122136/feature_cache.json')
gen60_cache_file = Path('evaluation_results/experimental/generalization_validation/run_20261020_132214/feature_cache.json')
gen60_manifest_file = Path('evaluation_results/experimental/generalization_validation/run_20261020_132214/run_manifest.json')

with open(dev30_cache_file, 'r', encoding='utf-8') as f:
    dev30_cache = json.load(f)
with open(gen60_cache_file, 'r', encoding='utf-8') as f:
    gen60_cache = json.load(f)
with open(gen60_manifest_file, 'r', encoding='utf-8') as f:
    gen60_manifest = {p['filename']: p for p in json.load(f)}

results = []

for r in fn_list:
    fn = r['filename']
    ds = r['dataset']
    split = r['source_split']
    
    cache_item = dev30_cache.get(fn) if split == 'dev_30' else gen60_cache.get(fn)
    manifest_item = gen60_manifest.get(fn, {}) if split == 'gen_60' else {}
    
    audio_path = None
    if cache_item and cache_item.get('file_path'):
        audio_path = cache_item['file_path']
    elif manifest_item.get('file_path'):
        audio_path = manifest_item['file_path']
        
    duration = None
    audio_exists = False
    if audio_path and os.path.exists(audio_path):
        audio_exists = True
        try:
            info = sf.info(audio_path)
            duration = info.duration
        except Exception:
            pass

    words = cache_item.get('words_list', []) if cache_item else []
    raw_transcript = cache_item.get('raw_transcript', cache_item.get('transcript', '')) if cache_item else ''
    sanitized_transcript = cache_item.get('sanitized_transcript', cache_item.get('transcript', '')) if cache_item else ''
    
    results.append({
        'filename': fn,
        'dataset': ds,
        'split': split,
        'audio_exists': audio_exists,
        'audio_path': audio_path,
        'duration': duration,
        'gt_dominant': r['gt_dominant'],
        'annotation_details': r.get('annotation_details', {}),
        'raw_transcript': raw_transcript,
        'sanitized_transcript': sanitized_transcript,
        'words_count': len(words),
        'words_list': words,
    })

out_json = Path('evaluation_results/experimental/phase4a_error_analysis/fn_detailed_inventory.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"Dumped {len(results)} detailed FN records to {out_json}")
