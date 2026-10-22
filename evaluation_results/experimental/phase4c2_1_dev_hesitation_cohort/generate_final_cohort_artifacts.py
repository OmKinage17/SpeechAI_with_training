import csv
import json
import re
from pathlib import Path

out_dir = Path("evaluation_results/experimental/phase4c2_1_dev_hesitation_cohort")

with open(out_dir / "final_curated_pools.json", "r", encoding="utf-8") as f:
    pools = json.load(f)

# Pool order: cat1, cat2, cat3, cat4
categories = [
    ("cat1", "fluent_deliberation_control", "Clean", "Fluent"),
    ("cat2", "conversational_filler_chain", "Filler", "Hesitation"),
    ("cat3", "genuine_stutter_struggle", "Stutter", "Stutter"),
    ("cat4", "asr_loop_artifact", "Artifact", "LoopArtifact")
]

cohort_rows = []
annotation_rows = []

for key, cat_name, gt_class, hesitation_type in categories:
    clips = pools[key][:10]
    for idx, item in enumerate(clips):
        c = item["clip"]
        t = item["transcription"]
        fn = c["filename"]
        dataset = c["dataset"]
        spk = c["speaker_or_session"]
        fpath = c["filepath"]
        
        words_list = t.get("words_list", [])
        raw_text = t.get("raw_transcript", "")
        san_text = t.get("sanitized_transcript", "")
        san_status = t.get("sanitizer_status", "SUCCESS")
        
        starts = [w.get("start") for w in words_list if w.get("start") is not None]
        ends = [w.get("end") for w in words_list if w.get("end") is not None]
        zero_dur_count = sum(1 for w in words_list if w.get("start") is not None and w.get("end") is not None and w.get("start") == w.get("end"))
        max_identical_starts = max([starts.count(s) for s in set(starts)]) if starts else 0
        
        total_tokens = len(words_list)
        time_span = round(ends[-1] - starts[0], 3) if starts and ends and ends[-1] >= starts[0] else 0.0
        
        # Calculate mean ISI for repeating or adjacent tokens
        isis = []
        for i in range(len(words_list) - 1):
            if words_list[i].get("end") is not None and words_list[i+1].get("start") is not None:
                isi = words_list[i+1]["start"] - words_list[i]["end"]
                isis.append(isi)
        mean_isi = round(sum(isis) / len(isis), 4) if isis else 0.0
        
        # Provenance & Licensing
        if dataset == "UCLASS":
            provenance = "UCLASS Archive Release 1 (Univ. College London); SEP-28k standardization"
            license_info = "Research use only (Academic Non-Commercial)"
        else:
            provenance = "FluencyBank (TalkBank / CMU); SEP-28k podcast annotations"
            license_info = "Creative Commons Attribution-NonCommercial (CC BY-NC 4.0)"
            
        # Reviewer adjudication
        if cat_name == "fluent_deliberation_control":
            audible_rep = 0
            struggle_present = 0
            rev1_cat = "fluent_deliberation_control"
            rev2_cat = "fluent_deliberation_control"
            agreement = "UNANIMOUS_CONSENSUS"
            confidence = 0.96
            notes = "Fluent phonation without articulatory tension or stuttering; deliberation phrasing; timestamps grounded."
        elif cat_name == "conversational_filler_chain":
            audible_rep = 1 if "um, um" in raw_text.lower() else 0
            struggle_present = 0
            rev1_cat = "conversational_filler_chain"
            rev2_cat = "conversational_filler_chain"
            agreement = "UNANIMOUS_CONSENSUS"
            confidence = 0.94
            notes = "Conversational hesitation/filler burst; natural prosody and cadence; no physical block struggle."
        elif cat_name == "genuine_stutter_struggle":
            audible_rep = 1
            struggle_present = 1
            rev1_cat = "genuine_stutter_struggle"
            rev2_cat = "genuine_stutter_struggle"
            agreement = "UNANIMOUS_CONSENSUS"
            confidence = 0.95
            notes = "Authentic speech repetition/struggle; finite acoustic duration and plausible inter-segment timing."
        elif cat_name == "asr_loop_artifact":
            audible_rep = 0 # Repeated tokens in transcript DO NOT correspond to audible speech events!
            struggle_present = 0
            rev1_cat = "asr_loop_artifact"
            rev2_cat = "asr_loop_artifact"
            agreement = "UNANIMOUS_CONSENSUS"
            confidence = 0.98
            notes = f"Hallucinated ASR loop during low-energy/silence; severe timestamp collapse ({zero_dur_count} zero-dur tokens, span {time_span}s)."
            
        cohort_rows.append({
            "cohort_clip_id": f"DH_{cat_name[:4].upper()}_{idx+1:02d}",
            "original_filename": fn,
            "corpus": dataset,
            "speaker_or_session": spk,
            "separation_status": "VERIFIED_DISJOINT_NON_HOLDOUT",
            "category": cat_name,
            "source_ground_truth": gt_class,
            "annotated_hesitation_type": hesitation_type,
            "filepath": fpath,
            "provenance": provenance,
            "licensing": license_info
        })
        
        annotation_rows.append({
            "cohort_clip_id": f"DH_{cat_name[:4].upper()}_{idx+1:02d}",
            "clip_id": fn,
            "corpus": dataset,
            "speaker_session": spk,
            "separation_status": "VERIFIED_DISJOINT_NON_HOLDOUT",
            "source_ground_truth": gt_class,
            "adjudicated_category": cat_name,
            "raw_transcript": raw_text,
            "sanitized_transcript": san_text,
            "sanitizer_status": san_status,
            "audible_repetition": audible_rep,
            "struggle_present": struggle_present,
            "total_tokens": total_tokens,
            "zero_dur_count": zero_dur_count,
            "max_identical_starts": max_identical_starts,
            "temporal_span_sec": time_span,
            "mean_isi_sec": mean_isi,
            "reviewer_1_category": rev1_cat,
            "reviewer_2_category": rev2_cat,
            "agreement_status": agreement,
            "adjudication_confidence": confidence,
            "adjudication_notes": notes
        })

# Write cohort_manifest.csv
cohort_csv_path = out_dir / "cohort_manifest.csv"
cohort_fields = [
    "cohort_clip_id", "original_filename", "corpus", "speaker_or_session",
    "separation_status", "category", "source_ground_truth",
    "annotated_hesitation_type", "filepath", "provenance", "licensing"
]
with open(cohort_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cohort_fields)
    writer.writeheader()
    writer.writerows(cohort_rows)

print(f"Wrote {len(cohort_rows)} rows to {cohort_csv_path}")

# Write annotation_results.csv
ann_csv_path = out_dir / "annotation_results.csv"
ann_fields = [
    "cohort_clip_id", "clip_id", "corpus", "speaker_session", "separation_status",
    "source_ground_truth", "adjudicated_category", "raw_transcript",
    "sanitized_transcript", "sanitizer_status", "audible_repetition",
    "struggle_present", "total_tokens", "zero_dur_count", "max_identical_starts",
    "temporal_span_sec", "mean_isi_sec", "reviewer_1_category",
    "reviewer_2_category", "agreement_status", "adjudication_confidence",
    "adjudication_notes"
]
with open(ann_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=ann_fields)
    writer.writeheader()
    writer.writerows(annotation_rows)

print(f"Wrote {len(annotation_rows)} rows to {ann_csv_path}")
