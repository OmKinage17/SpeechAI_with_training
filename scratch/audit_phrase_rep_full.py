"""
Full PhraseRep audit across all 500 Libri clips in the Phase 5 manifest.
READ-ONLY: no manifest or files are modified.
"""
import os, json
from collections import Counter

PROJ = r"E:\SpeechPrj"

with open(os.path.join(PROJ, "evaluation_results", "experimental", "phase5_benchmark", "phase5_benchmark_manifest.json"), 'r') as f:
    manifest = json.load(f)

libri_clips = [x for x in manifest if x['dataset'] == 'Libri_Stutter']
libri_annot_dir = os.path.join(PROJ, "datasets", "Libri_Stutter", "LibriStutter Annotations")

code4_correct = 0      # has code 4, gt_has_stutter=1 in manifest (was already stutter via other code)
code4_mismatched = 0   # has code 4, gt_has_stutter=0 in manifest (PhraseRep-only, mislabeled)
code4_phrase_only = 0  # has code 4 but NO codes 2,3,5 -> PhraseRep is the ONLY stutter signal
checked = 0
not_found = 0
phrase_only_mismatched = []

for clip in libri_clips:
    fname = clip['filename']
    stem = os.path.splitext(fname)[0]
    parts = stem.split('-')
    if len(parts) < 3:
        continue
    spk, ch = parts[0], parts[1]
    annot_csv = os.path.join(libri_annot_dir, spk, ch, f"{stem}.csv")
    if not os.path.exists(annot_csv):
        not_found += 1
        continue
    
    checked += 1
    has_code4 = False
    has_other_stutter = False  # codes 2, 3, 5
    try:
        with open(annot_csv, 'r', encoding='utf-8') as f:
            for line in f:
                cols = line.strip().split(',')
                if len(cols) >= 4:
                    code = cols[3].strip()
                    if code == '4':
                        has_code4 = True
                    elif code in ('2', '3', '5'):
                        has_other_stutter = True
    except Exception as e:
        continue
    
    if has_code4:
        manifest_says_stutter = clip.get('gt_has_stutter', 0)
        if manifest_says_stutter == 1:
            code4_correct += 1
        else:
            code4_mismatched += 1
            if not has_other_stutter:
                code4_phrase_only += 1
                phrase_only_mismatched.append(fname)

print(f"Total Libri clips: {len(libri_clips)}")
print(f"Annotation files found: {checked} / not found: {not_found}")
print(f"\nClips containing code 4 (PhraseRep):")
print(f"  -> gt_has_stutter=1 in manifest (correct, also had codes 2/3/5): {code4_correct}")
print(f"  -> gt_has_stutter=0 in manifest (MISLABELED by old code): {code4_mismatched}")
print(f"     of which PhraseRep was the ONLY stutter signal: {code4_phrase_only}")
print(f"\nPhraseRep-only mislabeled clips (first 10 of {code4_phrase_only}):")
for fn in phrase_only_mismatched[:10]:
    print(f"  {fn}")
