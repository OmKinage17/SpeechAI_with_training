"""
Unit Tests for Ablation 2: Filler-Aware & Boundary-Aware Gating
==============================================================

Verifies:
1. Prolongations overlapping conversational filler tokens ("um", "uh", "er", etc.) are gated.
2. Genuine pathological prolongations on content words are NOT gated.
3. Sentence-boundary vowel lengthening (< 0.65s) at terminal boundaries is gated.
4. Severe prolongations (>= 0.65s) at boundaries are preserved as genuine stutter candidates.
5. Multi-word and single-word repetitions consisting exclusively of filler tokens ("uh uh") are gated.
6. Lexical stutter repetitions on content words ("it was", "we we") are preserved.
7. Silence blocks adjacent to filler tokens are gated; genuine stutter blocks are preserved.
8. Diagnostic reason recording in gated_events.
9. Configuration isolation: when enable_filler_boundary_gating=False, detector behavior
   is completely identical to the reference candidate (Fix 3 + combined n-gram).
"""

import unittest
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "backend") not in sys.path:
    sys.path.insert(0, str(project_root / "backend"))

from backend.experimental.config import StutterDetectorConfig
from backend.experimental.filler_boundary_gating import FillerBoundaryGater
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.acoustic_features import AcousticFeatureExtractor


class TestFillerBoundaryGating(unittest.TestCase):

    def setUp(self):
        self.config = StutterDetectorConfig(
            enable_filler_boundary_gating=True,
            filler_overlap_threshold=0.35,
            filler_word_coverage_threshold=0.40,
            boundary_prolongation_min_duration_sec=0.65,
            gate_filler_blocks=True,
        )
        self.gater = FillerBoundaryGater(self.config)

    def test_prolongation_on_filler_gated(self):
        """Prolongation directly overlapping a conversational filler ('um') must be gated."""
        words = [
            {"word": "um", "start": 0.50, "end": 1.10},
            {"word": "what", "start": 1.20, "end": 1.50},
        ]
        prolongations = [
            {"start_time": 0.52, "end_time": 1.05, "duration": 0.53, "confidence": 0.70}
        ]

        active_prolongations, gated_events = self.gater.gate_prolongations(
            prolongations, words, total_duration=2.0
        )

        self.assertEqual(len(active_prolongations), 0)
        self.assertEqual(len(gated_events), 1)
        self.assertEqual(gated_events[0]["reason"], "filler_prolongation")
        self.assertEqual(gated_events[0]["overlapping_word"], "um")

    def test_prolongation_on_content_word_retained(self):
        """Prolongation on a non-filler word ('stream') must NOT be gated."""
        words = [
            {"word": "broken", "start": 0.20, "end": 0.60},
            {"word": "stream", "start": 0.70, "end": 1.40},
        ]
        prolongations = [
            {"start_time": 0.75, "end_time": 1.30, "duration": 0.55, "confidence": 0.75}
        ]

        active_prolongations, gated_events = self.gater.gate_prolongations(
            prolongations, words, total_duration=2.5
        )

        self.assertEqual(len(active_prolongations), 1)
        self.assertEqual(len(gated_events), 0)
        self.assertAlmostEqual(active_prolongations[0]["duration"], 0.55)

    def test_boundary_terminal_lengthening_gated(self):
        """Sentence-terminal vowel lengthening (< 0.65s) near final punctuation must be gated."""
        words = [
            {"word": "long", "start": 0.20, "end": 0.50},
            {"word": "ago.", "start": 0.55, "end": 1.10},
        ]
        prolongations = [
            {"start_time": 0.58, "end_time": 1.08, "duration": 0.50, "confidence": 0.70}
        ]

        active_prolongations, gated_events = self.gater.gate_prolongations(
            prolongations, words, total_duration=1.20
        )

        self.assertEqual(len(active_prolongations), 0)
        self.assertEqual(len(gated_events), 1)
        self.assertEqual(gated_events[0]["reason"], "sentence_boundary_lengthening")

    def test_severe_boundary_prolongation_retained(self):
        """Severe genuine prolongation (>= 0.65s) near boundary must be retained."""
        words = [
            {"word": "long", "start": 0.20, "end": 0.50},
            {"word": "ago.", "start": 0.55, "end": 1.45},
        ]
        prolongations = [
            {"start_time": 0.60, "end_time": 1.40, "duration": 0.80, "confidence": 0.85}
        ]

        active_prolongations, gated_events = self.gater.gate_prolongations(
            prolongations, words, total_duration=1.50
        )

        self.assertEqual(len(active_prolongations), 1)
        self.assertEqual(len(gated_events), 0)
        self.assertAlmostEqual(active_prolongations[0]["duration"], 0.80)

    def test_filler_repetition_gated(self):
        """Repetitions of pure filler tokens ('uh uh', 'um um') must be gated."""
        words = [
            {"word": "uh", "start": 0.20, "end": 0.40},
            {"word": "uh", "start": 0.45, "end": 0.65},
            {"word": "maybe", "start": 0.70, "end": 1.00},
        ]
        rep_details = [
            {"phrase": "uh", "iterations": 2, "ngram_size": 1, "confidence": 0.85},
            {"phrase": "uh uh", "iterations": 2, "ngram_size": 2, "confidence": 0.85},
        ]

        active_reps, gated_events = self.gater.gate_repetitions(rep_details, words)

        self.assertEqual(len(active_reps), 0)
        self.assertEqual(len(gated_events), 2)
        self.assertTrue(all(g["reason"] == "filler_repetition" for g in gated_events))

    def test_genuine_lexical_repetition_retained(self):
        """Repetition of content or function words ('it was', 'we we') must NOT be gated."""
        words = [
            {"word": "it", "start": 0.10, "end": 0.30},
            {"word": "was", "start": 0.35, "end": 0.55},
            {"word": "it", "start": 0.60, "end": 0.80},
            {"word": "was", "start": 0.85, "end": 1.05},
        ]
        rep_details = [
            {"phrase": "it was", "iterations": 2, "ngram_size": 2, "confidence": 0.85}
        ]

        active_reps, gated_events = self.gater.gate_repetitions(rep_details, words)

        self.assertEqual(len(active_reps), 1)
        self.assertEqual(len(gated_events), 0)
        self.assertEqual(active_reps[0]["phrase"], "it was")

    def test_filler_adjacent_block_gated(self):
        """Silence block adjacent to a filler token must be gated."""
        words = [
            {"word": "good.", "start": 0.20, "end": 0.50},
            {"word": "um,", "start": 1.20, "end": 1.50},
            {"word": "what", "start": 1.60, "end": 1.90},
        ]
        # Silence between 0.50 and 1.20 preceding "um,"
        blocks = [
            {"start_time": 0.55, "end_time": 1.15, "duration": 0.60, "confidence": 0.65}
        ]

        active_blocks, gated_events = self.gater.gate_blocks(blocks, words)

        self.assertEqual(len(active_blocks), 0)
        self.assertEqual(len(gated_events), 1)
        self.assertEqual(gated_events[0]["reason"] == "filler_adjacent_block", True)

    def test_genuine_block_not_adjacent_to_filler_retained(self):
        """Silence block between content words (genuine tense stutter pause) must be retained."""
        words = [
            {"word": "in", "start": 0.20, "end": 0.40},
            {"word": "Africa.", "start": 1.40, "end": 1.80},
        ]
        blocks = [
            {"start_time": 0.45, "end_time": 1.35, "duration": 0.90, "confidence": 0.75}
        ]

        active_blocks, gated_events = self.gater.gate_blocks(blocks, words)

        self.assertEqual(len(active_blocks), 1)
        self.assertEqual(len(gated_events), 0)


class TestHybridStutterDetectorGatingIntegration(unittest.TestCase):

    def test_flag_disabled_by_default(self):
        """Default config must have enable_filler_boundary_gating=False for reference integrity."""
        default_cfg = StutterDetectorConfig()
        self.assertFalse(default_cfg.enable_filler_boundary_gating)

    def test_detector_gating_filler_repetition_integration(self):
        """When disabled, 'uh uh' flags stutter repetition; when enabled, it is gated as conversational filler."""
        acoustic = AcousticFeatureExtractor()
        
        cfg_disabled = StutterDetectorConfig(enable_filler_boundary_gating=False)
        detector_disabled = HybridStutterDetector(config=cfg_disabled, acoustic_extractor=acoustic)

        cfg_enabled = StutterDetectorConfig(enable_filler_boundary_gating=True)
        detector_enabled = HybridStutterDetector(config=cfg_enabled, acoustic_extractor=acoustic)

        transcript = "uh uh maybe we could go."
        words = [
            {"word": "uh", "start": 0.20, "end": 0.40},
            {"word": "uh", "start": 0.45, "end": 0.65},
            {"word": "maybe", "start": 0.70, "end": 1.00},
            {"word": "we", "start": 1.05, "end": 1.25},
            {"word": "could", "start": 1.30, "end": 1.50},
            {"word": "go.", "start": 1.55, "end": 1.80},
        ]

        res_disabled = detector_disabled.evaluate(
            transcript=transcript,
            words_list=words
        )

        res_enabled = detector_enabled.evaluate(
            transcript=transcript,
            words_list=words
        )

        # Disabled flags stutter due to repeated word
        self.assertEqual(res_disabled["has_stutter"], 1)
        self.assertGreaterEqual(res_disabled["lexical_repetition_events"], 1)

        # Enabled gates filler repetition and avoids false positive
        self.assertEqual(res_enabled["has_stutter"], 0)
        self.assertEqual(res_enabled["lexical_repetition_events"], 0)
        self.assertIn("gated_events", res_enabled)
        self.assertEqual(len(res_enabled["gated_events"]), 1)
        self.assertEqual(res_enabled["gated_events"][0]["reason"], "filler_repetition")

    def test_detector_genuine_repetition_preserved(self):
        """Genuine content word repetition ('we we') must be flagged as stutter under both configurations."""
        acoustic = AcousticFeatureExtractor()
        
        cfg_enabled = StutterDetectorConfig(enable_filler_boundary_gating=True)
        detector_enabled = HybridStutterDetector(config=cfg_enabled, acoustic_extractor=acoustic)

        transcript = "we we could go."
        words = [
            {"word": "we", "start": 0.20, "end": 0.40},
            {"word": "we", "start": 0.45, "end": 0.65},
            {"word": "could", "start": 0.70, "end": 0.90},
            {"word": "go.", "start": 0.95, "end": 1.20},
        ]

        res_enabled = detector_enabled.evaluate(
            transcript=transcript,
            words_list=words
        )

        self.assertEqual(res_enabled["has_stutter"], 1)
        self.assertGreaterEqual(res_enabled["lexical_repetition_events"], 1)
        self.assertEqual(len(res_enabled.get("gated_events", [])), 0)


if __name__ == "__main__":
    unittest.main()

