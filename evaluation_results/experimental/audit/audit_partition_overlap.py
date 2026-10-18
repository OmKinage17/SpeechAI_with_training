"""
Partition Overlap Audit Script for SpeechAI Experimental Pipeline
================================================================
Verifies that the 30 development clips do NOT overlap with the 750-clip
main benchmark holdout at the clip level, speaker level, episode level, or reader level.
Saves structured report to: evaluation_results/experimental/partition_overlap_audit.json
"""

import os
import json

audit_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(audit_dir)))

dev_results_path = os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips.json")
main_manifest_path = os.path.join(project_root, "evaluation_results", "main_benchmark", "main_benchmark_manifest.json")
output_audit_path = os.path.join(project_root, "evaluation_results", "experimental", "partition_overlap_audit.json")

print("=== RUNNING PARTITION OVERLAP AUDIT ===")

with open(dev_results_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)
dev_clips = dev_data["predictions"]

with open(main_manifest_path, "r", encoding="utf-8") as f:
    main_clips = json.load(f)

print(f"Loaded {len(dev_clips)} development clips.")
print(f"Loaded {len(main_clips)} main benchmark holdout clips.")

# 1. Clip-level filename overlap
dev_filenames = set(c["filename"].lower() for c in dev_clips)
main_filenames = set(c["filename"].lower() for c in main_clips)
filename_overlap = dev_filenames.intersection(main_filenames)

# 2. Filepath overlap
dev_filepaths = set(os.path.normpath(c["file_path"]).lower() for c in dev_clips)
main_filepaths = set(os.path.normpath(c["file_path"]).lower() for c in main_clips)
filepath_overlap = dev_filepaths.intersection(main_filepaths)

# 3. Dataset-specific speaker / episode / reader overlap
dev_by_ds = {}
for c in dev_clips:
    dev_by_ds.setdefault(c["dataset"], []).append(c)

main_by_ds = {}
for c in main_clips:
    main_by_ds.setdefault(c["dataset"], []).append(c)

speaker_audits = {}
total_speaker_leakages = 0

for ds in ["UCLASS", "FluencyBank", "Libri_Stutter"]:
    d_clips = dev_by_ds.get(ds, [])
    m_clips = main_by_ds.get(ds, [])

    if ds == "UCLASS":
        dev_spks = set(c.get("speaker", "") for c in d_clips)
        main_spks = set(c.get("speaker", "") for c in m_clips)
    elif ds == "FluencyBank":
        dev_spks = set(c.get("speaker", "") for c in d_clips)
        main_spks = set(c.get("speaker", "") for c in m_clips)
    elif ds == "Libri_Stutter":
        # Reader is first part of speaker / filename before '-' or '_'
        dev_spks = set(c.get("speaker", "").split("_")[0] for c in d_clips)
        main_spks = set(c.get("speaker", "").split("_")[0] for c in m_clips)

    overlap = dev_spks.intersection(main_spks)
    total_speaker_leakages += len(overlap)

    speaker_audits[ds] = {
        "dev_clip_count": len(d_clips),
        "main_clip_count": len(m_clips),
        "dev_entities": sorted(list(dev_spks)),
        "main_entities_count": len(main_spks),
        "overlapping_entities": sorted(list(overlap)),
        "has_overlap": len(overlap) > 0
    }
    print(f"[{ds}] Dev entities: {sorted(list(dev_spks))}")
    print(f"[{ds}] Main entities count: {len(main_spks)}, Overlap: {list(overlap)}")

audit_result = {
    "audit_status": "PASSED" if (len(filename_overlap) == 0 and len(filepath_overlap) == 0 and total_speaker_leakages == 0) else "FAILED",
    "clip_level_overlap": {
        "dev_total_clips": len(dev_clips),
        "main_total_clips": len(main_clips),
        "filename_overlap_count": len(filename_overlap),
        "filename_overlaps": list(filename_overlap),
        "filepath_overlap_count": len(filepath_overlap),
        "filepath_overlaps": list(filepath_overlap),
        "is_disjoint": len(filename_overlap) == 0 and len(filepath_overlap) == 0
    },
    "speaker_level_overlap": {
        "total_overlapping_entities": total_speaker_leakages,
        "is_disjoint": total_speaker_leakages == 0,
        "datasets": speaker_audits
    }
}

with open(output_audit_path, "w", encoding="utf-8") as f:
    json.dump(audit_result, f, indent=2)

print(f"\nAudit completed. Status: {audit_result['audit_status']}.")
print(f"Saved audit report to: {output_audit_path}")
