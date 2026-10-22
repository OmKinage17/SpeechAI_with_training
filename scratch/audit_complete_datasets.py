import os
import glob
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

def audit_datasets():
    datasets_root = Path(r"E:\SpeechPrj\Datasets")
    print(f"=== AUDITING DATASETS ROOT: {datasets_root} ===")
    
    if not datasets_root.exists():
        print(f"ERROR: {datasets_root} does not exist!")
        return

    subdirs = [d for d in datasets_root.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} subdirectories in Datasets:")
    for d in subdirs:
        print(f" - {d.name}")

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    dataset_audio_counts = {}
    dataset_audio_files = {}

    for d in subdirs:
        files = []
        for ext in audio_extensions:
            files.extend(list(d.rglob(f"*{ext}")))
            files.extend(list(d.rglob(f"*{ext.upper()}")))
        dataset_audio_counts[d.name] = len(files)
        dataset_audio_files[d.name] = files
        print(f"\nTotal audio files in {d.name}: {len(files)}")
        if files:
            sample_paths = [str(f.relative_to(datasets_root)) for f in files[:2]]
            print(f" Sample paths: {sample_paths}")

    # Inspect key metadata and annotation files specifically
    print("\n--- Key Annotation & Metadata Files ---")
    key_files = [
        datasets_root / "Stuttering Events in Podcasts Dataset (SEP-28k)" / "SEP-28k_labels.csv",
        datasets_root / "Stuttering Events in Podcasts Dataset (SEP-28k)" / "SEP-28k_episodes.csv",
        datasets_root / "Stuttering Events in Podcasts Dataset (SEP-28k)" / "fluencybank_labels.csv",
        datasets_root / "Stuttering Events in Podcasts Dataset (SEP-28k)" / "fluencybank_episodes.csv",
        datasets_root / "UCLASS Stuttered Speech Clips (SEP-28k Format)" / "clips" / "metadata.json",
        datasets_root / "Libri_Stutter" / "README.md",
        datasets_root / "UCLASS Stuttered Speech Clips (SEP-28k Format)" / "README.md",
    ]
    for kf in key_files:
        if kf.exists():
            size = kf.stat().st_size
            with open(kf, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            print(f"{kf.name} ({kf.parent.name}): {size} bytes, {len(lines)} lines")
            if lines and kf.suffix == ".csv":
                print(f"  Columns: {lines[0].strip()}")

    # 4. Check for spent / previously used cohorts
    print("\n--- Auditing Previously Evaluated Cohorts (Strict Benchmark Protection) ---")
    
    # 4a. Main Benchmark 750 clips
    main_bench_path = Path(r"E:\SpeechPrj\evaluation_results\main_benchmark\main_benchmark_manifest.json")
    main_bench_clips = set()
    main_bench_speakers = set()
    if main_bench_path.exists():
        with open(main_bench_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                main_bench_clips.add(item.get("filename"))
                main_bench_speakers.add(item.get("speaker"))
        print(f"Main Benchmark (750): {len(main_bench_clips)} clips, {len(main_bench_speakers)} speakers: {sorted(list(main_bench_speakers))}")

    # 4b. Dev 30
    dev30_clips = set()
    diag_path = Path(r"E:\SpeechPrj\evaluation_results\experimental\dev_error_diagnostic.json")
    if diag_path.exists():
        with open(diag_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            dev30_clips = set(data.get("dev_30_records", {}).keys())
        print(f"Dev 30: {len(dev30_clips)} clips")

    # 4c. Gen 60
    gen60_clips = set()
    if diag_path.exists():
        with open(diag_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            gen60_clips = set(data.get("gen_60_records", {}).keys())
        print(f"Gen 60: {len(gen60_clips)} clips")

    # 4d. Dev-Hesitation 40
    cohort40_path = Path(r"E:\SpeechPrj\evaluation_results\experimental\phase4c2_1_dev_hesitation_cohort\cohort_manifest.csv")
    cohort40_clips = set()
    cohort40_speakers = set()
    if cohort40_path.exists():
        with open(cohort40_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cohort40_clips.add(row.get("original_filename"))
                cohort40_speakers.add(row.get("speaker_or_session"))
        print(f"Dev-Hesitation 40: {len(cohort40_clips)} clips, {len(cohort40_speakers)} speakers: {sorted(list(cohort40_speakers))}")

    # 4e. Spent Holdout 120 (Metadata only, DO NOT touch audio!)
    holdout120_path = Path(r"E:\SpeechPrj\evaluation_results\experimental\expanded_validation\run_expanded_120\run_manifest.json")
    holdout120_clips = set()
    holdout120_speakers = set()
    if holdout120_path.exists():
        with open(holdout120_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            manifest_items = data if isinstance(data, list) else data.get("manifest", [])
            for item in manifest_items:
                holdout120_clips.add(item.get("filename"))
                holdout120_speakers.add(item.get("speaker"))
        print(f"Spent Holdout 120: {len(holdout120_clips)} clips, {len(holdout120_speakers)} speakers: {sorted(list(holdout120_speakers))}")

    all_prior_clips = main_bench_clips | dev30_clips | gen60_clips | cohort40_clips | holdout120_clips
    all_prior_speakers = {s for s in (main_bench_speakers | cohort40_speakers | holdout120_speakers) if s is not None}
    print(f"\nTotal Unique Prior Evaluated Clips across all phases: {len(all_prior_clips)}")
    print(f"Total Unique Prior Speakers/Episodes: {len(all_prior_speakers)}: {sorted(list(all_prior_speakers))}")

    # 5. Analyze Total Available Audio vs Prior Evaluated
    all_dataset_audio_basenames = {}
    for dname, fpaths in dataset_audio_files.items():
        basenames = set(f.name for f in fpaths)
        overlap = basenames & all_prior_clips
        fresh = basenames - all_prior_clips
        all_dataset_audio_basenames[dname] = basenames
        print(f"\nDataset: {dname}")
        print(f"  Total audio clips on disk: {len(basenames)}")
        print(f"  Clips overlapping prior evaluations: {len(overlap)}")
        print(f"  FRESH (unspent) audio clips on disk: {len(fresh)}")

    # 6. Check UCLASS clips metadata
    uclass_meta_path = datasets_root / "UCLASS Stuttered Speech Clips (SEP-28k Format)" / "clips" / "metadata.json"
    if uclass_meta_path.exists():
        with open(uclass_meta_path, "r", encoding="utf-8") as f:
            uclass_meta = json.load(f)
        print(f"\nUCLASS metadata.json entries: {len(uclass_meta)}")
        uclass_speakers = set()
        uclass_dys_counts = Counter()
        uclass_fresh_count = 0
        uclass_fresh_speakers = set()
        for entry in uclass_meta:
            fname = entry.get("filename")
            spk = entry.get("speaker")
            uclass_speakers.add(spk)
            # Check speaker disjointness
            if spk not in all_prior_speakers:
                uclass_fresh_speakers.add(spk)
            if fname not in all_prior_clips and spk not in all_prior_speakers:
                uclass_fresh_count += 1
                dys = entry.get("dysfluency_type", "clean")
                uclass_dys_counts[dys] += 1
        print(f"UCLASS total speakers: {len(uclass_speakers)}: {sorted(list(uclass_speakers))}")
        print(f"UCLASS fresh (unspent & disjoint speakers) count: {uclass_fresh_count} from speakers: {sorted(list(uclass_fresh_speakers))}")
        print(f"UCLASS fresh clip types: {dict(uclass_dys_counts)}")

    # 7. Check SEP-28k / FluencyBank
    sep_clips_dir = datasets_root / "Stuttering Events in Podcasts Dataset (SEP-28k)" / "clips"
    sep_clip_files = list(sep_clips_dir.rglob("*.wav"))
    print(f"\nSEP-28k clips folder contains: {len(sep_clip_files)} wav files")
    
    # Check FluencyBank labels vs SEP-28k labels
    fb_labels_path = datasets_root / "Stuttering Events in Podcasts Dataset (SEP-28k)" / "fluencybank_labels.csv"
    if fb_labels_path.exists():
        with open(fb_labels_path, "r", encoding="utf-8") as f:
            fb_reader = csv.DictReader(f)
            fb_rows = list(fb_reader)
            fb_episodes = set(r.get("EpId", "") for r in fb_rows)
            print(f"FluencyBank labels rows: {len(fb_rows)}, Episodes: {len(fb_episodes)}")
            fb_fresh_rows = [r for r in fb_rows if r.get("EpId") not in all_prior_speakers]
            print(f"FluencyBank fresh (disjoint episode) rows: {len(fb_fresh_rows)}")

    sep_labels_path = datasets_root / "Stuttering Events in Podcasts Dataset (SEP-28k)" / "SEP-28k_labels.csv"
    if sep_labels_path.exists():
        with open(sep_labels_path, "r", encoding="utf-8") as f:
            sep_reader = csv.DictReader(f)
            sep_rows = list(sep_reader)
            sep_shows = set(r.get("Show", "") for r in sep_rows)
            sep_episodes = set(r.get("EpId", "") for r in sep_rows)
            print(f"SEP-28k labels rows: {len(sep_rows)}, Shows: {len(sep_shows)}, Episodes: {len(sep_episodes)}")
            sep_fresh_rows = [r for r in sep_rows if r.get("EpId") not in all_prior_speakers]
            print(f"SEP-28k fresh (disjoint episode) rows: {len(sep_fresh_rows)}")

    # 8. Check LibriStutter
    libri_audio_dir = datasets_root / "Libri_Stutter" / "LibriStutter Audio"
    libri_files = list(libri_audio_dir.rglob("*.wav")) + list(libri_audio_dir.rglob("*.flac"))
    print(f"\nLibriStutter audio folder contains: {len(libri_files)} audio files")

if __name__ == "__main__":
    audit_datasets()
