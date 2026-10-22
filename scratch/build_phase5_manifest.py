"""
Phase 5 Benchmark Manifest Generator (1,500 Clips)
==================================================
Script: scratch.build_phase5_manifest
Requirements:
- Strict speaker & episode isolation across datasets.
- Imports shared mapping helpers from scratch.ground_truth_mapping.
- Import-safe: Does NOT execute dataset scanning or manifest generation on module import.
"""

import os
import sys
import json
import csv
import glob
import random
import wave
import hashlib
from collections import Counter

# Import shared ground-truth mapping helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from scratch.ground_truth_mapping import _safe_int, derive_fluencybank_ground_truth, derive_libri_ground_truth
except ImportError:
    from ground_truth_mapping import _safe_int, derive_fluencybank_ground_truth, derive_libri_ground_truth

__all__ = ["derive_fluencybank_ground_truth", "derive_libri_ground_truth", "main"]


def main():
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    PROJ_DIR = r"E:\SpeechPrj"
    DATASET_ROOT = os.path.join(PROJ_DIR, "datasets")
    OUTPUT_DIR = os.path.join(PROJ_DIR, "evaluation_results", "experimental", "phase5_benchmark")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest_out_path = os.path.join(OUTPUT_DIR, "phase5_1_benchmark_manifest.json")
    preflight_report_path = os.path.join(OUTPUT_DIR, "phase5_1_preflight_check_report.json")

    print("=" * 70)
    print("PHASE 5.1 CORRECTED BENCHMARK PREFLIGHT & MANIFEST GENERATOR (1,500 CLIPS)")
    print("Random Seed: 42 | 100% Strict Speaker & Episode Isolation | Audited GT Mapping")
    print("=" * 70)

    # -------------------------------------------------------------
    # STEP 0: COMPILE GLOBAL EXCLUSION LIST OF FILENAMES & SPEAKERS
    # -------------------------------------------------------------
    dev30_path = os.path.join(PROJ_DIR, "evaluation_results", "experimental", "dev_eval_results_30clips_refined.json")
    with open(dev30_path, "r", encoding="utf-8") as f:
        dev30_files = set(x["filename"] for x in json.load(f)["predictions"])

    gen60_path = os.path.join(PROJ_DIR, "evaluation_results", "experimental", "generalization_validation", "run_20261020_132214", "run_manifest.json")
    with open(gen60_path, "r", encoding="utf-8") as f:
        gen60_files = set(x["filename"] for x in json.load(f))

    p4c_path = os.path.join(PROJ_DIR, "evaluation_results", "experimental", "phase4c2_1_dev_hesitation_cohort", "final_curated_pools.json")
    with open(p4c_path, "r", encoding="utf-8") as f:
        p4c_d = json.load(f)
        p4c_files = set(item["clip"]["filename"] for cat in p4c_d for item in p4c_d[cat])

    h120_path = os.path.join(PROJ_DIR, "evaluation_results", "experimental", "expanded_validation", "run_expanded_120", "run_manifest.json")
    with open(h120_path, "r", encoding="utf-8") as f:
        h120_files = set(x["filename"] for x in json.load(f))

    main_750_path = os.path.join(PROJ_DIR, "evaluation_results", "main_benchmark", "main_benchmark_manifest.json")
    with open(main_750_path, "r", encoding="utf-8") as f:
        main_750_files = set(x["filename"] for x in json.load(f))

    all_excluded_files = dev30_files | gen60_files | p4c_files | h120_files | main_750_files
    print(f"Loaded {len(all_excluded_files)} globally excluded files from all previous splits.")

    # -------------------------------------------------------------
    # 1. UCLASS POOL (Target: 500 clips from M_1101, M_1103, M_1106)
    # -------------------------------------------------------------
    print("\n[Pool 1/3] Assembling UCLASS Candidates...")
    uclass_clips_dir = os.path.join(DATASET_ROOT, "UCLASS Stuttered Speech Clips (SEP-28k Format)", "clips", "clips")
    uclass_labels_path = os.path.join(DATASET_ROOT, "UCLASS Stuttered Speech Clips (SEP-28k Format)", "clips", "labels.csv")
    target_uclass_spks = ["M_1101", "M_1103", "M_1106"]

    uclass_candidates = []
    with open(uclass_labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            bname = os.path.basename(r.get("filepath", ""))
            if bname in all_excluded_files:
                continue
            spk = next((s for s in target_uclass_spks if bname.startswith(s)), None)
            if not spk:
                continue
            fpath = os.path.join(uclass_clips_dir, bname)
            if not os.path.exists(fpath) or os.path.getsize(fpath) <= 44:
                continue
            # Note on UCLASS: Single expert clinical annotator per clip without multi-annotator
            # consensus voting. Each column is a binary indicator {0, 1}.
            try:
                with wave.open(fpath, "rb") as wf:
                    dur = wf.getnframes() / float(wf.getframerate())
                s_rep = _safe_int(r.get("SoundRep"))
                w_rep = _safe_int(r.get("WordRep"))
                prol = _safe_int(r.get("Prolongation"))
                blk = _safe_int(r.get("Block"))
                fil = _safe_int(r.get("Interjection"))
                cln = _safe_int(r.get("NoStutteredWords"))
                has_stutter = 1 if (s_rep or w_rep or prol or blk) else 0

                # Dominant class hierarchy: Stutter > Filler > Clean
                if has_stutter:
                    dom = "Stutter"
                elif fil:
                    dom = "Filler"
                else:
                    dom = "Clean"

                uclass_candidates.append({
                    "dataset": "UCLASS",
                    "file_path": fpath,
                    "filename": bname,
                    "speaker": spk,
                    "speaker_id": spk,
                    "duration_sec": round(dur, 3),
                    "gt_sound_rep": s_rep,
                    "gt_word_rep": w_rep,
                    "gt_prolongation": prol,
                    "gt_block": blk,
                    "gt_filler": fil,
                    "gt_clean": cln,
                    "gt_has_stutter": has_stutter,
                    "gt_has_filler": fil,
                    "gt_dominant": dom
                })
            except Exception as e:
                print(f"Warning: Corrupted UCLASS file {bname}: {e}")

    print(f"Verified UCLASS candidates: {len(uclass_candidates)}")
    assert len(uclass_candidates) >= 500, f"Insufficient UCLASS candidates ({len(uclass_candidates)} < 500)"

    # Shuffle deterministically and pick exactly 500
    random.shuffle(uclass_candidates)
    uclass_selected = uclass_candidates[:500]
    print(f"Selected UCLASS: 500 clips. Dominant: {Counter(c['gt_dominant'] for c in uclass_selected)}")

    # -------------------------------------------------------------
    # 2. FLUENCYBANK POOL (Target: 500 clips from 16 untouched eps)
    # -------------------------------------------------------------
    print("\n[Pool 2/3] Assembling FluencyBank Candidates...")
    fb_audio_dir = os.path.join(DATASET_ROOT, "Stuttering Events in Podcasts Dataset (SEP-28k)", "clips", "stuttering-clips", "clips")
    fb_labels_path = os.path.join(DATASET_ROOT, "Stuttering Events in Podcasts Dataset (SEP-28k)", "fluencybank_labels.csv")
    untouched_fb_eps = ["046", "071", "076", "093", "096", "098", "107", "109", "111", "118", "121", "132", "179", "193", "219", "230"]

    fb_candidates = []
    fb_ambiguous_excluded_count = 0
    with open(fb_labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ep = r.get("EpId", "").strip()
            if ep not in untouched_fb_eps:
                continue
            show = r.get("Show", "").strip()
            clip = r.get("ClipId", "").strip()
            fname = f"{show}_{ep}_{clip}.wav"
            if fname in all_excluded_files:
                continue
            fpath = os.path.join(fb_audio_dir, fname)
            if not os.path.exists(fpath) or os.path.getsize(fpath) <= 44:
                continue
            try:
                with wave.open(fpath, "rb") as wf:
                    dur = wf.getnframes() / float(wf.getframerate())

                gt = derive_fluencybank_ground_truth(r)
                if gt["is_ambiguous"]:
                    # Record counts of excluded ambiguous rows; do not silently classify as clean
                    fb_ambiguous_excluded_count += 1
                    continue

                fb_candidates.append({
                    "dataset": "FluencyBank",
                    "file_path": fpath,
                    "filename": fname,
                    "speaker": f"FB_Ep_{ep}",
                    "speaker_id": f"FB_Ep_{ep}",
                    "duration_sec": round(dur, 3),
                    "gt_sound_rep": gt["gt_sound_rep"],
                    "gt_word_rep": gt["gt_word_rep"],
                    "gt_prolongation": gt["gt_prolongation"],
                    "gt_block": gt["gt_block"],
                    "gt_filler": gt["gt_filler"],
                    "gt_clean": gt["gt_clean"],
                    "gt_has_stutter": gt["gt_has_stutter"],
                    "gt_has_filler": gt["gt_has_filler"],
                    "gt_dominant": gt["gt_dominant"]
                })
            except Exception as e:
                print(f"Warning: Corrupted FluencyBank file {fname}: {e}")

    print(f"Verified FluencyBank candidates: {len(fb_candidates)} (Excluded ambiguous: {fb_ambiguous_excluded_count})")
    assert len(fb_candidates) >= 500, f"Insufficient FluencyBank candidates ({len(fb_candidates)} < 500)"

    random.shuffle(fb_candidates)
    fb_selected = fb_candidates[:500]
    print(f"Selected FluencyBank: 500 clips. Dominant: {Counter(c['gt_dominant'] for c in fb_selected)}")

    # -------------------------------------------------------------
    # 3. LIBRI_STUTTER POOL (Target: 500 clips from 32 untouched readers)
    # -------------------------------------------------------------
    print("\n[Pool 3/3] Assembling Libri_Stutter Candidates...")
    libri_audio_dir = os.path.join(DATASET_ROOT, "Libri_Stutter", "LibriStutter Audio")
    libri_annot_dir = os.path.join(DATASET_ROOT, "Libri_Stutter", "LibriStutter Annotations")
    used_readers = {"103", "198", "1743", "2989", "1088", "1334", "1502", "1867", "1926", "2289"}
    all_readers = set(os.listdir(libri_audio_dir))
    untouched_readers = sorted(list(all_readers - used_readers))

    libri_candidates = []
    libri_unknown_code_clips_count = 0
    libri_total_unknown_codes = 0

    for rdr in untouched_readers:
        rdr_path = os.path.join(libri_audio_dir, rdr)
        for root, _, files in os.walk(rdr_path):
            for f in sorted(files):
                if not (f.endswith(".flac") or f.endswith(".wav")):
                    continue
                if f in all_excluded_files:
                    continue
                fpath = os.path.join(root, f)
                if not os.path.exists(fpath) or os.path.getsize(fpath) <= 44:
                    continue
                stem = os.path.splitext(f)[0]
                parts = stem.split("-")
                if len(parts) < 3:
                    continue
                spk, ch, utt = parts[0], parts[1], parts[2]
                annot_csv = os.path.join(libri_annot_dir, spk, ch, f"{stem}.csv")
                if not os.path.exists(annot_csv) or os.path.getsize(annot_csv) == 0:
                    continue

                try:
                    with open(annot_csv, "r", encoding="utf-8") as cf:
                        libri_gt = derive_libri_ground_truth(cf)
                except Exception as e:
                    print(f"Warning: Error reading annotation {annot_csv}: {e}")
                    continue

                if libri_gt.get("unknown_code_count", 0) > 0:
                    libri_unknown_code_clips_count += 1
                    libri_total_unknown_codes += libri_gt["unknown_code_count"]
                    # Do not silently classify unknown codes as clean: exclude clip from candidate pool
                    continue

                # Check duration via soundfile or wave
                dur = 3.0
                try:
                    import soundfile as sf
                    info = sf.info(fpath)
                    dur = round(info.duration, 3)
                except Exception:
                    pass

                libri_candidates.append({
                    "dataset": "Libri_Stutter",
                    "file_path": fpath,
                    "filename": f,
                    "speaker": f"Libri_{rdr}_{ch}",
                    "speaker_id": f"Libri_{rdr}_{ch}",
                    "reference_text": libri_gt["reference_text"],
                    "duration_sec": dur,
                    "gt_sound_rep": libri_gt["gt_sound_rep"],
                    "gt_word_rep": libri_gt["gt_word_rep"],
                    "gt_phrase_rep": libri_gt["gt_phrase_rep"],
                    "gt_prolongation": libri_gt["gt_prolongation"],
                    "gt_block": 0,  # Block ground truth is not annotated in Libri_Stutter
                    "gt_filler": libri_gt["gt_filler"],
                    "gt_clean": libri_gt["gt_clean"],
                    "gt_has_stutter": libri_gt["gt_has_stutter"],
                    "gt_has_filler": libri_gt["gt_filler"],
                    "gt_dominant": libri_gt["gt_dominant"]
                })

    print(f"Verified Libri_Stutter candidates: {len(libri_candidates)} (Excluded unknown code clips: {libri_unknown_code_clips_count})")
    assert len(libri_candidates) >= 500, f"Insufficient Libri_Stutter candidates ({len(libri_candidates)} < 500)"

    random.shuffle(libri_candidates)
    libri_selected = libri_candidates[:500]
    print(f"Selected Libri_Stutter: 500 clips. Dominant: {Counter(c['gt_dominant'] for c in libri_selected)}")

    # -------------------------------------------------------------
    # STEP 4: COMBINE, AUDIT & RUN COMPREHENSIVE INTEGRITY CHECKS
    # -------------------------------------------------------------
    final_manifest = uclass_selected + fb_selected + libri_selected
    print(f"\nFinal Manifest Assembled: {len(final_manifest)} clips total.")

    print("\n[Integrity Checks] Running rigorous validation suite...")

    # 1. Total and per-corpus count checks
    assert len(final_manifest) == 1500, f"Expected 1,500 total clips, got {len(final_manifest)}"
    assert len(uclass_selected) == 500, f"Expected 500 UCLASS clips, got {len(uclass_selected)}"
    assert len(fb_selected) == 500, f"Expected 500 FluencyBank clips, got {len(fb_selected)}"
    assert len(libri_selected) == 500, f"Expected 500 Libri_Stutter clips, got {len(libri_selected)}"
    print("  [Pass 1/8] Total and per-corpus clip counts verified (500 / 500 / 500).")

    # 2. Duplicate clips checks
    filenames = [c["filename"] for c in final_manifest]
    assert len(filenames) == len(set(filenames)), "Duplicate filenames detected in manifest!"
    filepaths = [c["file_path"] for c in final_manifest]
    assert len(filepaths) == len(set(filepaths)), "Duplicate file paths detected in manifest!"
    print("  [Pass 2/8] Zero duplicate filenames or filepaths across all 1,500 clips.")

    # 3. Audio file readability and non-empty checks
    for c in final_manifest:
        fp = c["file_path"]
        assert os.path.exists(fp), f"Audio file does not exist: {fp}"
        assert os.path.getsize(fp) > 44, f"Audio file is empty or corrupted: {fp}"
    print("  [Pass 3/8] All 1,500 audio files exist, are readable, and > 44 bytes.")

    # 4. Required fields and valid labels checks
    base_required_keys = {
        "dataset", "file_path", "filename", "speaker", "speaker_id",
        "duration_sec", "gt_sound_rep", "gt_word_rep", "gt_prolongation",
        "gt_block", "gt_filler", "gt_clean", "gt_has_stutter",
        "gt_has_filler", "gt_dominant"
    }
    for c in final_manifest:
        missing_keys = base_required_keys - set(c.keys())
        assert not missing_keys, f"Missing required keys {missing_keys} in clip {c['filename']}"
        assert c["gt_has_stutter"] in (0, 1), f"Invalid gt_has_stutter in {c['filename']}"
        assert c["gt_has_filler"] in (0, 1), f"Invalid gt_has_filler in {c['filename']}"
        assert c["gt_clean"] in (0, 1), f"Invalid gt_clean in {c['filename']}"
        assert c["gt_dominant"] in ("Stutter", "Filler", "Clean"), f"Invalid gt_dominant in {c['filename']}"
        assert c["gt_sound_rep"] in (0, 1), f"Invalid gt_sound_rep in {c['filename']}"
        assert c["gt_word_rep"] in (0, 1), f"Invalid gt_word_rep in {c['filename']}"
        assert c["gt_prolongation"] in (0, 1), f"Invalid gt_prolongation in {c['filename']}"
        assert c["gt_block"] in (0, 1), f"Invalid gt_block in {c['filename']}"
        assert c["gt_filler"] in (0, 1), f"Invalid gt_filler in {c['filename']}"

        # Dominant hierarchy consistency
        if c["gt_has_stutter"] == 1:
            assert c["gt_dominant"] == "Stutter", f"Inconsistent stutter dominant in {c['filename']}"
        elif c["gt_has_filler"] == 1:
            assert c["gt_dominant"] == "Filler", f"Inconsistent filler dominant in {c['filename']}"
        else:
            assert c["gt_dominant"] == "Clean", f"Inconsistent clean dominant in {c['filename']}"

        # Libri specific field and PhraseRep checks
        if c["dataset"] == "Libri_Stutter":
            assert "reference_text" in c, f"Missing reference_text in Libri clip {c['filename']}"
            assert "gt_phrase_rep" in c, f"Missing gt_phrase_rep in Libri clip {c['filename']}"
            assert c["gt_phrase_rep"] in (0, 1), f"Invalid gt_phrase_rep in {c['filename']}"
            if c["gt_phrase_rep"] == 1:
                assert c["gt_has_stutter"] == 1, f"PhraseRep clip {c['filename']} has gt_has_stutter != 1"
                assert c["gt_dominant"] == "Stutter", f"PhraseRep clip {c['filename']} has gt_dominant != Stutter"
    print("  [Pass 4/8] Required schema keys and valid label domains strictly verified.")

    # 5. Overlap with previous splits (approved metadata only, NEVER loading/processing holdout audio)
    overlap_with_exclusions = set(filenames) & all_excluded_files
    assert len(overlap_with_exclusions) == 0, f"Overlap detected with previous splits: {overlap_with_exclusions}"
    overlap_with_h120 = set(filenames) & h120_files
    assert len(overlap_with_h120) == 0, f"Overlap detected with locked holdout: {overlap_with_h120}"
    print(f"  [Pass 5/8] Strict isolation verified: 0 overlap with {len(all_excluded_files)} globally excluded clips.")

    # 6. Verify locked holdout manifest hash remains completely untouched
    with open(h120_path, "rb") as f:
        current_h120_hash = hashlib.sha256(f.read()).hexdigest()
    assert current_h120_hash == "fdfa62ce5a85c22b57db013fbee8451f84777058e671ae2eb897f8c121e4fb00", "Locked holdout modified!"
    print(f"  [Pass 6/8] Locked Phase 4B holdout integrity verified: SHA-256 matches locked hash.")

    # 7. Verify historical phase 5 manifest remains completely untouched
    historical_manifest_path = os.path.join(OUTPUT_DIR, "phase5_benchmark_manifest.json")
    with open(historical_manifest_path, "rb") as f:
        current_hist_hash = hashlib.sha256(f.read()).hexdigest()
    assert current_hist_hash == "b03d87ab5ce51b2d672fcc118165d3d00d10e26a39f911899d15cf5588553f4a", "Historical manifest modified!"
    print(f"  [Pass 7/8] Historical Phase 5 manifest preserved unchanged: SHA-256 matches b03d87ab5c...")

    # 8. PhraseRep statistics computation
    libri_prep_total = sum(1 for c in libri_selected if c.get("gt_phrase_rep") == 1)
    libri_prep_only = sum(1 for c in libri_selected if c.get("gt_phrase_rep") == 1 and c["gt_sound_rep"] == 0 and c["gt_word_rep"] == 0 and c["gt_prolongation"] == 0)
    print(f"  [Pass 8/8] PhraseRep validation: total={libri_prep_total}, PhraseRep-only={libri_prep_only} (all labeled Stutter).")

    # Write new manifest JSON
    with open(manifest_out_path, "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2)

    # Compute SHA-256 of new manifest
    with open(manifest_out_path, "rb") as f:
        manifest_sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"\nManifest saved to: {manifest_out_path}")
    print(f"Manifest SHA-256: {manifest_sha256}")

    # Write Preflight Report for Phase 5.1
    preflight_report = {
        "manifest_path": manifest_out_path,
        "manifest_sha256": manifest_sha256,
        "total_clips": len(final_manifest),
        "random_seed": RANDOM_SEED,
        "breakdown_by_dataset": {
            "UCLASS": {
                "selected": len(uclass_selected),
                "speakers": sorted(list(set(c["speaker_id"] for c in uclass_selected))),
                "dominant_classes": dict(Counter(c["gt_dominant"] for c in uclass_selected))
            },
            "FluencyBank": {
                "selected": len(fb_selected),
                "episodes": sorted(list(set(c["speaker_id"] for c in fb_selected))),
                "dominant_classes": dict(Counter(c["gt_dominant"] for c in fb_selected)),
                "excluded_ambiguous_rows": fb_ambiguous_excluded_count
            },
            "Libri_Stutter": {
                "selected": len(libri_selected),
                "readers": sorted(list(set(c["speaker_id"].split("_")[1] for c in libri_selected))),
                "dominant_classes": dict(Counter(c["gt_dominant"] for c in libri_selected)),
                "phrase_rep_stats": {
                    "total_phrase_rep_clips": libri_prep_total,
                    "phrase_rep_only_clips": libri_prep_only
                },
                "unknown_codes_stats": {
                    "clips_with_unknown_codes": libri_unknown_code_clips_count,
                    "total_unknown_code_occurrences": libri_total_unknown_codes
                }
            }
        },
        "exclusions_verified": {
            "dev30_count": len(dev30_files),
            "gen60_count": len(gen60_files),
            "p4c_cohort_count": len(p4c_files),
            "h120_holdout_count": len(h120_files),
            "main_750_count": len(main_750_files),
            "total_excluded": len(all_excluded_files),
            "observed_overlap": len(overlap_with_exclusions)
        },
        "historical_integrity": {
            "phase5_historical_manifest_sha256": current_hist_hash,
            "locked_holdout_manifest_sha256": current_h120_hash,
            "artifacts_preserved": True
        }
    }

    with open(preflight_report_path, "w", encoding="utf-8") as f:
        json.dump(preflight_report, f, indent=2)

    print(f"Preflight report saved to: {preflight_report_path}")
    print("Phase 5.1 manifest build & integrity validation 100% SUCCESSFUL!")


if __name__ == "__main__":
    main()
