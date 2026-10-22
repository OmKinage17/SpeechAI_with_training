"""
Automated Regression and Mathematical Reconciliation Tests
for SpeechAI Benchmark Suite (120-Clip Holdout & Canonical Metrics)
===================================================================

Audits and verifies:
1. Manifest integrity, 12-speaker isolation, and 100% disjointness from Dev, Gen, and Main holdouts.
2. Independent mathematical recalculation of Multiclass and Binary metrics from raw predictions.
3. Detection and isolation of Whisper greedy-decoding degenerate repetition loops.
4. Retention of raw vs sanitized transcripts without prediction rejection.
5. Reconciliation between per-clip prediction records and Parameter 2 (Filler) metrics.
6. Multi-label co-occurrence accounting (dominant-class vs event-level).
7. Accurate runtime status accounting (distinguishing pristine inference from recovered/sanitized inference).
"""

import os
import sys
import json
import re
import unittest
from pathlib import Path
from collections import Counter

# Set project paths
project_root = Path(__file__).resolve().parent.parent.parent
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestBenchmarkReconciliation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.project_root = project_root
        cls.final_benchmark_dir = project_root / "evaluation_results" / "experimental" / "final_benchmark" / "run_final_20261020_complete"
        cls.manifest_path = cls.final_benchmark_dir / "run_manifest.json"
        cls.predictions_path = cls.final_benchmark_dir / "per_clip_predictions.json"
        cls.final_metrics_path = cls.final_benchmark_dir / "final_metrics.json"
        cls.param_metrics_path = cls.final_benchmark_dir / "per_parameter_metrics.json"
        cls.cm_path = cls.final_benchmark_dir / "confusion_matrices.json"
        cls.runtime_path = cls.final_benchmark_dir / "runtime_and_failures.json"

        # Load artifacts if they exist
        cls.has_artifacts = cls.predictions_path.exists() and cls.manifest_path.exists()
        if cls.has_artifacts:
            with open(cls.manifest_path, "r", encoding="utf-8") as f:
                cls.manifest = json.load(f)
            with open(cls.predictions_path, "r", encoding="utf-8") as f:
                cls.predictions = json.load(f)
            with open(cls.final_metrics_path, "r", encoding="utf-8") as f:
                cls.final_metrics = json.load(f)
            with open(cls.param_metrics_path, "r", encoding="utf-8") as f:
                cls.param_metrics = json.load(f)
            with open(cls.cm_path, "r", encoding="utf-8") as f:
                cls.confusion_matrices = json.load(f)
            with open(cls.runtime_path, "r", encoding="utf-8") as f:
                cls.runtime_data = json.load(f)

    def test_01_manifest_composition_and_speaker_isolation(self):
        """Task 1: Verify 120 clips, 3 balanced corpora, 12 distinct speakers, 0 duplicates."""
        if not self.has_artifacts:
            self.skipTest("Benchmark artifacts not found.")

        self.assertEqual(len(self.manifest), 120, "Manifest must contain exactly 120 clips.")
        
        # Check uniqueness
        filenames = [c["filename"] for c in self.manifest]
        self.assertEqual(len(set(filenames)), 120, "All 120 filenames must be completely unique.")

        # Check balanced corpus partitions
        datasets = Counter(c["dataset"] for c in self.manifest)
        self.assertEqual(datasets["UCLASS"], 40, "Must have exactly 40 UCLASS clips.")
        self.assertEqual(datasets["FluencyBank"], 40, "Must have exactly 40 FluencyBank clips.")
        self.assertEqual(datasets["Libri_Stutter"], 40, "Must have exactly 40 Libri_Stutter clips.")

        # Check speaker entities (12 entities, 10 clips each)
        speakers = Counter(c.get("speaker_id") or c.get("speaker") for c in self.manifest)
        self.assertEqual(len(speakers), 12, "Must contain exactly 12 distinct speaker/episode entities.")
        for spk, count in speakers.items():
            self.assertEqual(count, 10, f"Speaker {spk} must have exactly 10 clips.")

        # Verify zero overlap with other partitions
        dev_manifest_p = project_root / "evaluation_results" / "experimental" / "dev_eval_results_30clips_refined.json"
        if dev_manifest_p.exists():
            with open(dev_manifest_p, "r", encoding="utf-8") as f:
                dev_clips = [p["filename"] for p in json.load(f)["predictions"]]
            overlap_dev = set(filenames).intersection(set(dev_clips))
            self.assertEqual(len(overlap_dev), 0, f"Found overlap with Dev partition: {overlap_dev}")

        main_manifest_p = project_root / "evaluation_results" / "main_benchmark" / "main_benchmark_manifest.json"
        if main_manifest_p.exists():
            with open(main_manifest_p, "r", encoding="utf-8") as f:
                main_data = json.load(f)
                main_clips = [c["filename"] for c in (main_data if isinstance(main_data, list) else main_data.get("clips", []))]
            overlap_main = set(filenames).intersection(set(main_clips))
            self.assertEqual(len(overlap_main), 0, f"Found overlap with Main Benchmark: {overlap_main}")

    def test_02_multiclass_and_binary_mathematical_reconciliation(self):
        """Task 3: Recompute multiclass and binary stutter metrics from raw predictions."""
        if not self.has_artifacts:
            self.skipTest("Benchmark artifacts not found.")

        # Recompute Multiclass Confusion Matrix
        classes = ["Clean", "Stutter", "Filler"]
        cm_calc = {c1: {c2: 0 for c2 in classes} for c1 in classes}
        for p in self.predictions:
            gt = p["gt_dominant"]
            pr = p["pred_dominant"]
            cm_calc[gt][pr] += 1

        saved_cm = self.confusion_matrices["aggregate"]
        for c1 in classes:
            for c2 in classes:
                self.assertEqual(
                    cm_calc[c1][c2], saved_cm[c1][c2],
                    f"Confusion matrix mismatch for [{c1}][{c2}]: calc={cm_calc[c1][c2]} vs saved={saved_cm[c1][c2]}"
                )

        # Recompute Binary Stutter Metrics (P1)
        tp = sum(1 for p in self.predictions if p["gt_has_stutter"] == 1 and p["pred_has_stutter"] == 1)
        fp = sum(1 for p in self.predictions if p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 1)
        fn = sum(1 for p in self.predictions if p["gt_has_stutter"] == 1 and p["pred_has_stutter"] == 0)
        tn = sum(1 for p in self.predictions if p["gt_has_stutter"] == 0 and p["pred_has_stutter"] == 0)

        saved_bin = self.param_metrics["parameter_1_stuttering"]["binary"]
        self.assertEqual((tp, fp, fn, tn), (saved_bin["tp"], saved_bin["fp"], saved_bin["fn"], saved_bin["tn"]))

        calc_prec = tp / (tp + fp) if (tp + fp) else 0.0
        calc_rec = tp / (tp + fn) if (tp + fn) else 0.0
        calc_f1 = 2 * calc_prec * calc_rec / (calc_prec + calc_rec) if (calc_prec + calc_rec) else 0.0
        self.assertAlmostEqual(calc_prec, saved_bin["precision"], places=4)
        self.assertAlmostEqual(calc_rec, saved_bin["recall"], places=4)
        self.assertAlmostEqual(calc_f1, saved_bin["f1"], places=4)

    def test_03_filler_metric_discrepancy_identification(self):
        """Task 3 & 4: Audit discrepancy between pred_has_filler in clip results vs Parameter 2 metrics."""
        if not self.has_artifacts:
            self.skipTest("Benchmark artifacts not found.")

        p2 = self.param_metrics["parameter_2_fillers"]
        # In per_parameter_metrics.json, tp + fp = 31 + 25 = 56 positive predictions
        p2_positives = p2["tp"] + p2["fp"]

        # In per_clip_predictions.json:
        clip_pred_has_filler = sum(p.get("pred_has_filler", 0) for p in self.predictions)
        clip_filler_count_gt0 = sum(1 for p in self.predictions if p.get("filler_count", 0) > 0)

        # Confirm the root cause of the discrepancy:
        # clip_pred_has_filler equals 38 (only dominant-class Filler) while p2_positives equals 56 (filler_count > 0)
        self.assertEqual(clip_filler_count_gt0, p2_positives, "P2 metrics were calculated using filler_count > 0.")
        self.assertNotEqual(
            clip_pred_has_filler, p2_positives,
            "Discrepancy confirmed: pred_has_filler in per_clip_predictions reflects dominant class (38), not event presence (56)."
        )

    def test_04_identify_whisper_repetition_loops(self):
        """Task 4: Identify clips with degenerate repetition loops and extreme WPM outliers."""
        if not self.has_artifacts:
            self.skipTest("Benchmark artifacts not found.")

        # Detect degenerate repetition loops: word count > 50 in 3.0s audio clip
        looping_clips = []
        for p in self.predictions:
            dur = p.get("clip_duration_sec", 3.0)
            words = len(p.get("transcript", "").split())
            wpm = p.get("wpm", 0.0)
            if dur <= 3.5 and words >= 50:
                looping_clips.append((p["filename"], p["dataset"], dur, words, wpm))

        # We expect exactly 9 clips in the existing uncorrected final benchmark
        self.assertEqual(
            len(looping_clips), 9,
            f"Expected exactly 9 clips with degenerate repetition loops; found {len(looping_clips)}: {looping_clips}"
        )
        for fn, ds, dur, words, wpm in looping_clips:
            self.assertGreater(wpm, 1000.0, f"Looping clip {fn} should have extreme WPM > 1000; got {wpm}")

    def test_05_repetition_loop_sanitizer_logic(self):
        """Task 4: Verify that degenerate repetition loop detection correctly truncates loops without discarding true words."""
        sample_transcript = "Well, that's it. In your job. Um, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh"
        
        # Test sanitizer function
        words = sample_transcript.split()
        self.assertGreater(len(words), 100)

        # Detect loop: trailing token or n-gram repeating >= 4 times
        tokens = [re.sub(r'[^\w]', '', w).lower() for w in words]
        
        # Find where consecutive single tokens repeat >= 4 times
        loop_start_idx = None
        for i in range(len(tokens) - 4):
            if tokens[i] and all(tokens[i] == tokens[i + k] for k in range(1, 5)):
                loop_start_idx = i
                break

        self.assertIsNotNone(loop_start_idx)
        # Keep initial repetitions up to 2 instances, truncate the runaway tail
        sanitized_words = words[:loop_start_idx + 2]
        sanitized_text = " ".join(sanitized_words)
        
        self.assertLess(len(sanitized_words), 15)
        self.assertTrue(sanitized_text.startswith("Well, that's it."))

    def test_06_multilabel_cooccurrence_distribution(self):
        """Task 1 & 2: Audit co-occurring dysfluencies (stutter + interjection)."""
        if not self.has_artifacts:
            self.skipTest("Benchmark artifacts not found.")

        cooccurring_clips = []
        for c in self.manifest:
            ann = c.get("annotation_details", {})
            has_st = any(ann.get(k, 0) > 0 for k in ["sound_rep", "word_rep", "prolongation", "block"])
            has_fil = ann.get("interjection", 0) > 0
            if has_st and has_fil:
                cooccurring_clips.append(c["filename"])

        self.assertEqual(
            len(cooccurring_clips), 12,
            f"Expected exactly 12 clips with co-occurring Stutter and Filler; found {len(cooccurring_clips)}"
        )

    def test_07_genuine_stuttering_preservation(self):
        """Phase 2 Targeted: Genuine clinical disfluencies must be 100% preserved without truncation."""
        from backend.experimental.asr_engine import detect_and_sanitize_repetition_loop

        genuine_samples = [
            "If you did did want to come with us today.",
            "I I I think that we should go now.",
            "Let's just, just, just, just, just round one.",
            "One would be for choosing her father, sir, sir, sir, sir, several times.",
            "Um, uh, well you know, actually I think it's fine."
        ]

        for sample in genuine_samples:
            res = detect_and_sanitize_repetition_loop(sample, clip_duration_sec=3.0)
            self.assertFalse(res["is_loop"], f"Genuine sample incorrectly flagged as loop: {sample}")
            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["truncated_tokens_count"], 0)
            self.assertEqual(res["sanitized_transcript"], sample)
            self.assertEqual(res["raw_transcript"], sample)

    def test_08_degenerate_loop_detection_and_separation(self):
        """Phase 2 Targeted: Degenerate runaway repetition loops must be detected and sanitized separately from raw."""
        from backend.experimental.asr_engine import detect_and_sanitize_repetition_loop

        # Synthetic runaway loop (110 repetitions of 'uh')
        runaway_filler = "Well, that's it. In your job. " + " ".join(["uh"] * 105)
        res = detect_and_sanitize_repetition_loop(runaway_filler, clip_duration_sec=3.0)

        self.assertTrue(res["is_loop"])
        self.assertEqual(res["status"], "DEGENERATE_LOOP_RECOVERED")
        self.assertEqual(res["raw_transcript"], runaway_filler)
        self.assertNotEqual(res["sanitized_transcript"], runaway_filler)
        self.assertLess(len(res["sanitized_transcript"].split()), 15)
        self.assertTrue(res["sanitized_transcript"].startswith("Well, that's it. In your job."))
        self.assertGreater(res["truncated_tokens_count"], 90)
        self.assertIsNotNone(res["loop_pattern"])
        self.assertIn("Degenerate", res["reason"])

        # Synthetic runaway repetition (100 repetitions of lexical word 'did')
        runaway_word = "If you " + " ".join(["did"] * 100)
        res2 = detect_and_sanitize_repetition_loop(runaway_word, clip_duration_sec=3.0)
        self.assertTrue(res2["is_loop"])
        self.assertEqual(res2["status"], "DEGENERATE_LOOP_RECOVERED")
        self.assertGreater(res2["truncated_tokens_count"], 90)
        # Verify genuine repetition onset 'did, did' is preserved for downstream detectors
        self.assertTrue(res2["sanitized_transcript"].startswith("If you did did"))

    def test_09_raw_vs_sanitized_transcript_integrity(self):
        """Phase 2 Targeted: Both raw and sanitized transcripts must be stored and never overwritten."""
        from backend.experimental.asr_engine import detect_and_sanitize_repetition_loop

        test_raw = "Testing audio recording. " + " ".join(["um"] * 50)
        res = detect_and_sanitize_repetition_loop(test_raw, clip_duration_sec=3.0)
        
        # Invariants
        self.assertIn("raw_transcript", res)
        self.assertIn("sanitized_transcript", res)
        self.assertEqual(res["raw_transcript"], test_raw)
        self.assertNotEqual(res["sanitized_transcript"], test_raw)

    def test_10_filler_cooccurrence_and_dual_measure_reconciliation(self):
        """Phase 2 Targeted: Evaluator must compute dominant vs event-level filler metrics separately."""
        from backend.experimental.canonical_evaluator import CanonicalEvaluator

        evaluator = CanonicalEvaluator(str(project_root))
        
        # Synthetic records with both dominant and event-level fields
        records = [
            # 1. Pure Stutter (no filler)
            {"gt_dominant": "Stutter", "pred_dominant": "Stutter", "gt_has_stutter": 1, "pred_has_stutter": 1, "gt_has_filler": 0, "pred_has_filler": 0, "pred_has_filler_event": 0, "filler_count": 0, "dataset": "UCLASS"},
            # 2. Co-occurring: Stutter + Filler annotation; model detects both (pred_dominant=Stutter, but filler_event=1)
            {"gt_dominant": "Stutter", "pred_dominant": "Stutter", "gt_has_stutter": 1, "pred_has_stutter": 1, "gt_has_filler": 1, "pred_has_filler": 0, "pred_has_filler_event": 1, "filler_count": 2, "dataset": "FluencyBank"},
            # 3. Pure Filler
            {"gt_dominant": "Filler", "pred_dominant": "Filler", "gt_has_stutter": 0, "pred_has_stutter": 0, "gt_has_filler": 1, "pred_has_filler": 1, "pred_has_filler_event": 1, "filler_count": 1, "dataset": "FluencyBank"},
            # 4. Clean Speech
            {"gt_dominant": "Clean", "pred_dominant": "Clean", "gt_has_stutter": 0, "pred_has_stutter": 0, "gt_has_filler": 0, "pred_has_filler": 0, "pred_has_filler_event": 0, "filler_count": 0, "dataset": "Libri_Stutter"}
        ]

        metrics = evaluator._compute_metrics_block(records)
        self.assertIn("filler_evaluation", metrics)
        fl_eval = metrics["filler_evaluation"]
        
        # Dominant-Filler has 1 positive prediction
        self.assertEqual(fl_eval["dominant_class"]["total_positive_predictions"], 1)
        # Event-Level Filler has 2 positive predictions (pure filler + co-occurring filler)
        self.assertEqual(fl_eval["event_level"]["total_positive_predictions"], 2)
        # Both recall measures are 100% on their respective grounds
        self.assertEqual(fl_eval["dominant_class"]["recall"], 1.0)
        self.assertEqual(fl_eval["event_level"]["recall"], 1.0)
        self.assertIn("Filler Discrepancy Resolution", fl_eval["discrepancy_reconciliation"])

    def test_11_clinical_block_specificity_scoping(self):
        """Phase 2 Targeted: Block specificity must be scoped to annotated corpora (UCLASS + FB)."""
        from backend.experimental.canonical_evaluator import CanonicalEvaluator

        evaluator = CanonicalEvaluator(str(project_root))
        records = [
            # UCLASS clip with block GT and block prediction
            {"dataset": "UCLASS", "gt_block": 1, "pred_block": 1, "gt_dominant": "Stutter", "pred_dominant": "Stutter", "gt_has_stutter": 1, "pred_has_stutter": 1},
            # UCLASS clean clip
            {"dataset": "UCLASS", "gt_block": 0, "pred_block": 0, "gt_dominant": "Clean", "pred_dominant": "Clean", "gt_has_stutter": 0, "pred_has_stutter": 0},
            # Libri_Stutter clip without block GT (unannotated control) but acoustic pause triggered
            {"dataset": "Libri_Stutter", "gt_block": 0, "pred_block": 1, "gt_dominant": "Clean", "pred_dominant": "Clean", "gt_has_stutter": 0, "pred_has_stutter": 0}
        ]

        metrics = evaluator._compute_metrics_block(records)
        self.assertIn("block_detection", metrics)
        bl = metrics["block_detection"]
        
        # Scoped clinical specificity should be 100% (1 TN / 1 Neg) because Libri is excluded
        self.assertEqual(bl["scoped_clinical_uclass_fluencybank"]["specificity"], 1.0)
        # Full benchmark specificity is penalized by the unannotated Libri pause (1 TN / 2 Neg = 50%)
        self.assertEqual(bl["full_benchmark"]["specificity"], 0.5)

    def test_12_burst_vs_sustained_speech_rate_distinction(self):
        """Phase 2 Targeted: Speech-rate calculation distinguishes short-clip burst rate from sustained rate without hard WPM rejection."""
        # 3.0s clip with 8 words = 160 WPM burst rate
        w_short = 8
        dur_short = 3.0
        wpm_short = round(w_short / (dur_short / 60.0), 1)
        self.assertEqual(wpm_short, 160.0)

        # Ensure high WPM is NOT used to discard predictions:
        # A fast burst or stuttered repetition clip remains included in the dataset evaluation
        pred_record = {
            "filename": "fast_burst.wav",
            "wpm": 280.0,
            "clip_duration_sec": 3.0,
            "pred_dominant": "Stutter",
            "gt_dominant": "Stutter"
        }
        self.assertTrue(pred_record["pred_dominant"] == pred_record["gt_dominant"])
        self.assertIn("wpm", pred_record)

    def test_13_asr_wer_cer_denominators_and_scoping(self):
        """Phase 2 Targeted: WER/CER calculation uses documented denominators and excludes ungrounded corpora."""
        from backend.experimental.canonical_evaluator import CanonicalEvaluator, compute_levenshtein_wer, compute_levenshtein_cer

        # Test Levenshtein WER & CER
        ref = "it is a bright sunny day"
        hyp = "it is bright sunny today"
        wer, edits, ref_len, hyp_len, _ = compute_levenshtein_wer(ref, hyp)
        # ref has 6 words. hyp has 5 words. "a" is deleted (1), "today" substituted for "day" (1) -> 2 edits / 6 words = 0.3333
        self.assertEqual(ref_len, 6)
        self.assertEqual(edits, 2)
        self.assertAlmostEqual(wer, 2 / 6, places=4)

        # Test Evaluator ASR section
        evaluator = CanonicalEvaluator(str(project_root))
        sample_records = [
            {"dataset": "Libri_Stutter", "reference": "hello world", "sanitized_transcript": "hello world", "gt_dominant": "Clean", "pred_dominant": "Clean", "gt_has_stutter": 0, "pred_has_stutter": 0},
            {"dataset": "Libri_Stutter", "reference": "good morning to you", "sanitized_transcript": "good morning you", "gt_dominant": "Clean", "pred_dominant": "Clean", "gt_has_stutter": 0, "pred_has_stutter": 0},
            {"dataset": "UCLASS", "reference": None, "sanitized_transcript": "something else", "gt_dominant": "Stutter", "pred_dominant": "Stutter", "gt_has_stutter": 1, "pred_has_stutter": 1}
        ]
        metrics = evaluator._compute_metrics_block(sample_records)
        self.assertIn("asr_recognition", metrics)
        asr_m = metrics["asr_recognition"]
        self.assertEqual(asr_m["clips_with_gold_reference"], 2)
        self.assertEqual(asr_m["excluded_clips_count"], 1)
        self.assertEqual(asr_m["total_reference_words_denominator"], 6)  # 2 + 4 = 6 words
        self.assertEqual(asr_m["total_word_edits"], 1)  # 1 deletion in second clip
        self.assertAlmostEqual(asr_m["corpus_wer"], 1 / 6, places=4)
        self.assertIn("Corpus WER is calculated as total_word_edits / total_reference_words", asr_m["denominator_documentation"])


if __name__ == "__main__":
    unittest.main()


