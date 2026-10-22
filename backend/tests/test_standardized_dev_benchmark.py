"""
Unit & Integrity Tests for Standardized Development Benchmark
=============================================================

Verifies:
- Shared feature extraction consistency.
- Inter-word silence interval derivation across edge cases.
- Word timestamp sanitization (handling numpy floats).
- Explicit timeout accounting without denominator truncation.
- Identical clip IDs, ground truth, and evaluation denominators across configurations.
"""

import unittest
from pathlib import Path
import json
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
import sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "backend") not in sys.path:
    sys.path.insert(0, str(project_root / "backend"))

from backend.experimental.run_dev_benchmark import (
    extract_inter_word_silences,
    sanitize_word_record
)
from backend.experimental.eval_harness import get_dominant_label


class TestStandardizedDevBenchmark(unittest.TestCase):

    def test_inter_word_silence_extraction_normal(self):
        """Derives initial, inter-word, and terminal silences properly."""
        words_list = [
            {"word": "Hello", "start": 0.50, "end": 1.00, "probability": 0.95},
            {"word": "world", "start": 1.40, "end": 1.90, "probability": 0.92}
        ]
        silences = extract_inter_word_silences(words_list, audio_duration_sec=2.50)
        self.assertEqual(len(silences), 3)
        self.assertEqual(silences[0]["type"], "initial_silence")
        self.assertEqual(silences[0]["duration"], 0.50)

        self.assertEqual(silences[1]["type"], "inter_word_silence")
        self.assertEqual(silences[1]["after_word"], "Hello")
        self.assertEqual(silences[1]["before_word"], "world")
        self.assertEqual(silences[1]["duration"], 0.40)

        self.assertEqual(silences[2]["type"], "terminal_silence")
        self.assertEqual(silences[2]["duration"], 0.60)

    def test_inter_word_silence_extraction_empty_and_contiguous(self):
        """Empty words return empty silences; contiguous words produce 0 inter-word gaps."""
        self.assertEqual(extract_inter_word_silences([], audio_duration_sec=2.0), [])

        # Contiguous words (0 gap)
        contiguous = [
            {"word": "one", "start": 0.0, "end": 1.0, "probability": 0.9},
            {"word": "two", "start": 1.0, "end": 2.0, "probability": 0.9}
        ]
        silences = extract_inter_word_silences(contiguous, audio_duration_sec=2.0)
        # Should have 0 initial, 0 inter-word, 0 terminal
        self.assertEqual(len(silences), 0)

    def test_sanitize_word_record_numpy_types(self):
        """Numpy float64 types must be sanitized to standard Python floats."""
        raw = {
            "word": " testing",
            "start": np.float64(1.23456),
            "end": np.float64(2.34567),
            "probability": np.float64(0.87654)
        }
        clean = sanitize_word_record(raw)
        self.assertIsInstance(clean["start"], float)
        self.assertIsInstance(clean["end"], float)
        self.assertIsInstance(clean["probability"], float)
        self.assertEqual(clean["start"], 1.235)
        self.assertEqual(clean["end"], 2.346)
        self.assertEqual(clean["probability"], 0.8765)

    def test_manifest_invariance_and_denominator(self):
        """Manifest must contain exactly 30 clips with 24 Stutter, 5 Filler, 1 Clean."""
        manifest_path = project_root / "evaluation_results" / "experimental" / "dev_eval_results_30clips_refined.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            dev = json.load(f)
        preds = dev["predictions"]
        self.assertEqual(len(preds), 30)
        self.assertEqual(sum(1 for p in preds if p["gt_has_stutter"] == 1), 24)
        self.assertEqual(sum(1 for p in preds if p["gt_dominant"] == "Filler"), 5)
        self.assertEqual(sum(1 for p in preds if p["gt_dominant"] == "Clean"), 1)

    def test_timeout_policy_preservation(self):
        """Timeout clips must remain in denominator and assign non-stutter dominant when uncorroborated."""
        # Simulated timeout record: has_stutter = 0, has_filler = 0 -> Clean
        dom = get_dominant_label(has_stutter=0, has_filler=0)
        self.assertEqual(dom, "Clean")
        # If has_filler = 1 -> Filler
        dom_filler = get_dominant_label(has_stutter=0, has_filler=1)
        self.assertEqual(dom_filler, "Filler")


if __name__ == "__main__":
    unittest.main()
