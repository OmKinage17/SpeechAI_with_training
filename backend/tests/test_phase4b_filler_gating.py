"""
Regression and validation test suite for Phase 4B: Multi-Iteration Hesitation Gating.

Strict Scope Safeguards:
- Development suite only.
- Validates isolated experimental candidate against authoritative Phase 2 baseline.
- Tests ordinary filler gating, rapid struggle detection, leisurely hesitation handling,
  lexical repetition preservation, transcript immutability, and calibrated confidence.
"""

import unittest
import numpy as np
from backend.experimental.config import StutterDetectorConfig
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.filler_boundary_gating import FillerBoundaryGater


class TestPhase4BFillerGating(unittest.TestCase):
    def setUp(self):
        # Baseline configuration (Phase 2 authoritative baseline: multi-iteration disabled)
        self.baseline_config = StutterDetectorConfig(
            enable_filler_boundary_gating=True,
            enable_multi_iteration_filler_stutter=False
        )
        self.baseline_detector = HybridStutterDetector(self.baseline_config)

        # Candidate configuration (Phase 4B experimental candidate: multi-iteration enabled)
        self.candidate_config = StutterDetectorConfig(
            enable_filler_boundary_gating=True,
            enable_multi_iteration_filler_stutter=True,
            min_filler_repetition_count=3,
            rapid_filler_isi_threshold_sec=0.35
        )
        self.candidate_detector = HybridStutterDetector(self.candidate_config)

    def test_ordinary_filler_repetitions_remain_fillers(self):
        """1 or 2 ordinary filler repetitions must remain fillers and not trigger stutter."""
        # Pair of fillers with leisurely or normal timing
        words_list = [
            {"word": "uh", "start": 0.5, "end": 0.7},
            {"word": "uh", "start": 1.1, "end": 1.3},
            {"word": "I", "start": 1.5, "end": 1.7},
            {"word": "think", "start": 1.8, "end": 2.1},
        ]
        transcript = "uh uh I think"

        # Baseline: gated out as filler
        base_res = self.baseline_detector.evaluate(
            transcript=transcript, words_list=words_list, pred_has_filler=1
        )
        self.assertEqual(base_res["has_stutter"], 0)
        self.assertEqual(base_res["pred_dominant"], "Filler")

        # Candidate: still gated out as filler because count=2 < min_filler_repetition_count(3)
        cand_res = self.candidate_detector.evaluate(
            transcript=transcript, words_list=words_list, pred_has_filler=1
        )
        self.assertEqual(cand_res["has_stutter"], 0)
        self.assertEqual(cand_res["pred_dominant"], "Filler")
        self.assertEqual(cand_res["classification"], "CONFIDENT_CLEAN")

    def test_three_or_more_repeated_fillers_with_struggle_evidence(self):
        """3 or more rapid repeated fillers with struggle pacing (ISI <= 0.35s) trigger stutter."""
        # 3 rapid 'uh' tokens: ISI = 0.25s (rapid motor struggle loop)
        words_list = [
            {"word": "uh", "start": 0.10, "end": 0.30},
            {"word": "uh", "start": 0.35, "end": 0.55},
            {"word": "uh", "start": 0.60, "end": 0.80},
            {"word": "today", "start": 1.00, "end": 1.40},
        ]
        transcript = "uh uh uh today"

        # Baseline gates out all fillers -> Misses struggle FN
        base_res = self.baseline_detector.evaluate(
            transcript=transcript, words_list=words_list, pred_has_filler=1
        )
        self.assertEqual(base_res["has_stutter"], 0)
        self.assertEqual(base_res["pred_dominant"], "Filler")

        # Candidate detects rapid struggle loop
        cand_res = self.candidate_detector.evaluate(
            transcript=transcript, words_list=words_list, pred_has_filler=1
        )
        self.assertEqual(cand_res["has_stutter"], 1)
        self.assertEqual(cand_res["pred_dominant"], "Stutter")
        self.assertEqual(cand_res["classification"], "CONFIDENT_DISFLUENT")
        # Calibrated confidence for 3 rapid iterations is 0.60 (NOT hardcoded 0.85)
        self.assertEqual(cand_res["confidence_score"], 0.60)
        self.assertIn("multi_iteration_filler_struggle", [d.get("type") for d in cand_res["details"]])

    def test_severe_struggle_chain_calibrated_confidence(self):
        """7 or more rapid repeated fillers (like M_0030_015) receive higher calibrated confidence."""
        # 7 rapid 'er' tokens (similar to M_0030_015, mean ISI ~0.22s)
        words_list = [
            {"word": "er", "start": 0.00, "end": 0.15},
            {"word": "er", "start": 0.20, "end": 0.35},
            {"word": "er", "start": 0.40, "end": 0.55},
            {"word": "er", "start": 0.60, "end": 0.75},
            {"word": "er", "start": 0.80, "end": 0.95},
            {"word": "er", "start": 1.00, "end": 1.15},
            {"word": "er", "start": 1.20, "end": 1.34},
        ]
        transcript = "er er er er er er er"

        cand_res = self.candidate_detector.evaluate(
            transcript=transcript, words_list=words_list, pred_has_filler=1
        )
        self.assertEqual(cand_res["has_stutter"], 1)
        self.assertEqual(cand_res["pred_dominant"], "Stutter")
        # 7 iterations: 0.70 + (7 - 5) * 0.03 = 0.76 (calibrated, strictly < 0.85)
        self.assertAlmostEqual(cand_res["confidence_score"], 0.76, places=2)

    def test_three_or_more_repeated_fillers_in_fluent_context(self):
        """3 or more repeated fillers in a leisurely conversational context remain non-stutter."""
        # 3 leisurely 'um' tokens: mean ISI = 0.70s (> 0.35s threshold, leisurely hesitation)
        words_list = [
            {"word": "um", "start": 0.20, "end": 0.50},
            {"word": "um", "start": 1.20, "end": 1.50},
            {"word": "um", "start": 2.20, "end": 2.50},
            {"word": "let", "start": 3.00, "end": 3.20},
            {"word": "see", "start": 3.25, "end": 3.60},
        ]
        transcript = "um um um let see"

        # Baseline: gated out as filler
        base_res = self.baseline_detector.evaluate(
            transcript=transcript, words_list=words_list, pred_has_filler=1
        )
        self.assertEqual(base_res["has_stutter"], 0)
        self.assertEqual(base_res["pred_dominant"], "Filler")

        # Candidate: leisurely repetition is treated as UNCERTAIN_BORDERLINE (confidence 0.50)
        # Without corroboration, it does NOT predict stuttering!
        cand_res = self.candidate_detector.evaluate(
            transcript=transcript, words_list=words_list, pred_has_filler=1
        )
        self.assertEqual(cand_res["has_stutter"], 0)
        self.assertEqual(cand_res["pred_dominant"], "Filler")
        self.assertEqual(cand_res["classification"], "UNCERTAIN_BORDERLINE")
        self.assertEqual(cand_res["confidence_score"], 0.50)
        self.assertIn("multi_iteration_filler_hesitation", [d.get("type") for d in cand_res["details"]])

    def test_genuine_lexical_repetitions_and_interrupted_words_remain_intact(self):
        """Genuine word repetitions and broken syllable fragments must remain detected with high confidence."""
        # True whole-word repetition
        rep_transcript = "I I want to go"
        rep_words = [
            {"word": "I", "start": 0.1, "end": 0.3},
            {"word": "I", "start": 0.4, "end": 0.6},
            {"word": "want", "start": 0.7, "end": 0.9},
            {"word": "to", "start": 1.0, "end": 1.1},
            {"word": "go", "start": 1.2, "end": 1.4},
        ]
        cand_rep = self.candidate_detector.evaluate(rep_transcript, rep_words)
        self.assertEqual(cand_rep["has_stutter"], 1)
        self.assertEqual(cand_rep["pred_dominant"], "Stutter")
        self.assertGreaterEqual(cand_rep["confidence_score"], 0.85)

        # Interrupted syllable fragment
        frag_transcript = "c- cat on the mat"
        cand_frag = self.candidate_detector.evaluate(frag_transcript)
        self.assertEqual(cand_frag["has_stutter"], 1)
        self.assertEqual(cand_frag["pred_dominant"], "Stutter")
        self.assertGreaterEqual(cand_frag["confidence_score"], 0.85)

    def test_raw_and_sanitized_transcript_preservation(self):
        """Detector evaluation must not mutate or alter input transcripts or word structures."""
        original_transcript = "um   um   um  hello  world!  "
        copied_transcript = str(original_transcript)
        words_list = [
            {"word": "um", "start": 0.0, "end": 0.2},
            {"word": "um", "start": 0.25, "end": 0.45},
            {"word": "um", "start": 0.50, "end": 0.70},
            {"word": "hello", "start": 0.80, "end": 1.10},
            {"word": "world", "start": 1.20, "end": 1.50},
        ]
        words_copy = [dict(w) for w in words_list]

        _ = self.candidate_detector.evaluate(original_transcript, words_list)

        self.assertEqual(original_transcript, copied_transcript)
        self.assertEqual(words_list, words_copy)

    def test_no_behavior_changes_to_phase2_baseline_configuration(self):
        """When enable_multi_iteration_filler_stutter=False, behavior is identical to Phase 2 baseline."""
        # 7x 'er' streak should be completely gated out under baseline
        words_list = [
            {"word": "er", "start": 0.00, "end": 0.15},
            {"word": "er", "start": 0.20, "end": 0.35},
            {"word": "er", "start": 0.40, "end": 0.55},
            {"word": "er", "start": 0.60, "end": 0.75},
            {"word": "er", "start": 0.80, "end": 0.95},
            {"word": "er", "start": 1.00, "end": 1.15},
            {"word": "er", "start": 1.20, "end": 1.34},
        ]
        transcript = "er er er er er er er"

        base_res = self.baseline_detector.evaluate(transcript, words_list, pred_has_filler=1)
        self.assertEqual(base_res["has_stutter"], 0)
        self.assertEqual(base_res["pred_dominant"], "Filler")
        self.assertEqual(base_res["lexical_repetition_events"], 0)

    def test_uncertain_hesitation_corroboration_dynamics(self):
        """Leisurely hesitation alone is non-stutter; with independent acoustic symptom it corroborates."""
        words_list = [
            {"word": "um", "start": 0.20, "end": 0.50},
            {"word": "um", "start": 1.20, "end": 1.50},
            {"word": "um", "start": 2.20, "end": 2.50},
            {"word": "people", "start": 3.00, "end": 3.50},
        ]
        transcript = "um um um people"

        # Case 1: Leisurely hesitation alone -> uncorroborated, has_stutter = 0
        res_alone = self.candidate_detector.evaluate(transcript, words_list, pred_has_filler=1)
        self.assertEqual(res_alone["has_stutter"], 0)
        self.assertEqual(res_alone["classification"], "UNCERTAIN_BORDERLINE")

        # Case 2: Injected acoustic prolongation (duration 0.55s, not severe enough alone to trigger confident disfluent)
        # We simulate the details to test corroboration logic directly
        gater = FillerBoundaryGater(
            enable_multi_iteration_filler_stutter=True,
            min_filler_repetition_count=3,
            rapid_filler_isi_threshold_sec=0.35
        )
        cands = [
            {"type": "phrase_repetition", "phrase": "um um", "start": 0.2, "end": 2.5},
            {"type": "acoustic_prolongation", "start": 3.0, "end": 3.55, "duration": 0.55, "pitch_std_st": 0.5, "confidence": 0.50}
        ]
        retained, gated = gater.gate_details(cands, words_list=words_list, transcript=transcript)
        # Should retain both the hesitation and the prolongation
        types = [d.get("type") for d in retained]
        self.assertIn("multi_iteration_filler_hesitation", types)
        self.assertIn("acoustic_prolongation", types)


if __name__ == "__main__":
    unittest.main()
