"""
Audit: Check whether the existing Phase 5 manifest was built with the 
old (pre-PhraseRep fix) or new (post-fix) code, by inspecting annotated
Libri CSV files directly and comparing what code 4 occurrences exist.
READ-ONLY: no manifest or files are modified.
"""
import os, csv, json
from collections import Counter

PROJ = r"E:\SpeechPrj"

# Load the manifest to get the Libri clips that were selected
with open(os.path.join(PROJ, "evaluation_results", "experimental", "phase5_benchmark", "phase5_benchmark_manifest.json"), 'r') as f:
    manifest = json.load(f)

libri_clips = [x for x in manifest if x['dataset'] == 'Libri_Stutter']
print(f"Libri clips in manifest: {len(libri_clips)}")

# Sample a subset of Libri annotation files directly
# and count how many have code 4 (PhraseRep) among the selected clips
libri_annot_dir = os.path.join(PROJ, "Datasets", "Libri_Stutter", "LibriStutter Annotations")
if not os.path.exists(libri_annot_dir):
    # Try alternate path
    libri_annot_dir = os.path.join(PROJ, "datasets", "Libri_Stutter", "LibriStutter Annotations")

print(f"Annotation dir exists: {os.path.exists(libri_annot_dir)}")

# For the first 50 clips, check if any annotation CSV contains code 4
code4_clips = 0
no_code4_but_stutter = 0
code4_mismatched = 0  # clip has code 4 but gt_has_stutter=0 in manifest (pre-fix mislabeling)
code4_correct = 0  # clip has code 4 and gt_has_stutter=1 in manifest
checked = 0

for clip in libri_clips[:100]:
    fname = clip['filename']
    fpath = clip['file_path']
    stem = os.path.splitext(fname)[0]
    parts = stem.split('-')
    if len(parts) < 3:
        continue
    spk, ch = parts[0], parts[1]
    annot_csv = os.path.join(libri_annot_dir, spk, ch, f"{stem}.csv")
    if not os.path.exists(annot_csv):
        continue
    
    checked += 1
    has_code4 = False
    try:
        with open(annot_csv, 'r', encoding='utf-8') as f:
            for line in f:
                parts2 = line.strip().split(',')
                if len(parts2) >= 4 and parts2[3].strip() == '4':
                    has_code4 = True
                    break
    except Exception as e:
        print(f"Error reading {annot_csv}: {e}")
        continue
    
    if has_code4:
        code4_clips += 1
        manifest_says_stutter = clip.get('gt_has_stutter', 0)
        if manifest_says_stutter == 1:
            code4_correct += 1
        else:
            code4_mismatched += 1  # PhraseRep but labeled NOT stutter in manifest

print(f"\nChecked {checked} annotation files for first 100 Libri clips")
print(f"Clips with code 4 (PhraseRep): {code4_clips}")
print(f"  -> manifest gt_has_stutter=1 (correct): {code4_correct}")
print(f"  -> manifest gt_has_stutter=0 (MISLABELED by old code): {code4_mismatched}")
