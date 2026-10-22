"""
Unit Tests for Multi-Word N-Gram Repetition Detector (Ablation 1)
================================================================

Verifies:
1. Contiguous 2-gram phrase repetition detection.
2. Contiguous 3-gram phrase repetition detection.
3. Robustness to punctuation, hyphens, and clause boundaries.
4. Overlap and subsumption handling (3-gram subsumes sub-2-gram, no double counting).
5. Single repeated token exclusion (e.g., "the the the" handled by unigram, not n-gram).
6. Multi-iteration chains (>= 3 iterations with confidence boost).
7. Negative controls on fluent non-repeated speech.
8. Empty, whitespace-only, and None transcripts.
9. Integration with HybridStutterDetector (flag isolation and confidence elevation).
"""

import unittest
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "backend") not in sys.path:
    sys.path.insert(0, str(project_root / "backend"))

from backend.experimental.ngram_repetition import NGramRepetitionDetector, clean_token
from backend.experimental.config import StutterDetectorConfig
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.acoustic_features import AcousticFeatureExtractor
from backend.experimental.speech_context import SpeechContextAnalyzer


class TestNGramRepetitionDetector(unittest.TestCase):

    def setUp(self):
        self.detector_combined = NGramRepetitionDetector(allowed_sizes=(2, 3))
        self.detector_2gram = NGramRepetitionDetector(allowed_sizes=(2,))
        self.detector_3gram = NGramRepetitionDetector(allowed_sizes=(3,))

    def test_2gram_repetition_basic(self):
        text = "It was reputed to be an intricate headlong, intricate headlong broken stream."
        events, details = self.detector_combined.detect(text)
        self.assertEqual(events, 1)
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["ngram_size"], 2)
        self.assertEqual(details[0]["phrase"], "intricate headlong")
        self.assertEqual(details[0]["iterations"], 2)
        self.assertAlmostEqual(details[0]["confidence"], 0.85)

    def test_3gram_repetition_with_punctuation(self):
        text = "Or you could, or you could sponsor."
        events, details = self.detector_combined.detect(text)
        self.assertEqual(events, 1)
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["ngram_size"], 3)
        self.assertEqual(details[0]["phrase"], "or you could")
        self.assertEqual(details[0]["iterations"], 2)
        self.assertAlmostEqual(details[0]["confidence"], 0.85)

    def test_3gram_repetition_with_hyphens(self):
        text = "Or you could- or you could sponsor-"
        events, details = self.detector_combined.detect(text)
        self.assertEqual(events, 1)
        self.assertEqual(details[0]["phrase"], "or you could")
        self.assertEqual(details[0]["iterations"], 2)

    def test_multi_iteration_chain(self):
        text = "sharp eye, sharp eye, sharp eye looking ahead"
        events, details = self.detector_combined.detect(text)
        self.assertEqual(events, 1)
        self.assertEqual(details[0]["ngram_size"], 2)
        self.assertEqual(details[0]["iterations"], 3)
        self.assertAlmostEqual(details[0]["confidence"], 0.90)

    def test_overlap_subsumption_3gram_over_2gram(self):
        # In "all you could all you could", a 3-gram matches.
        # Ensure it does NOT also emit overlapping sub-2-grams from the same phrase.
        text = "Make a warship. All you could. All you could."
        events, details = self.detector_combined.detect(text)
        self.assertEqual(events, 1)
        self.assertEqual(details[0]["ngram_size"], 3)
        self.assertEqual(details[0]["phrase"], "all you could")

    def test_2gram_detector_isolation(self):
        # 2-gram only should NOT match 3-grams that do not contain 2-gram repetitions
        text = "Or you could, or you could sponsor."
        events, details = self.detector_2gram.detect(text)
        self.assertEqual(events, 0)

        text_2g = "an intricate headlong, intricate headlong broken stream"
        events_2g, details_2g = self.detector_2gram.detect(text_2g)
        self.assertEqual(events_2g, 1)
        self.assertEqual(details_2g[0]["ngram_size"], 2)

    def test_3gram_detector_isolation(self):
        # 3-gram only should NOT match 2-gram repetitions
        text_2g = "an intricate headlong, intricate headlong broken stream"
        events_2g, details_2g = self.detector_3gram.detect(text_2g)
        self.assertEqual(events_2g, 0)

        text_3g = "Or you could, or you could sponsor."
        events_3g, details_3g = self.detector_3gram.detect(text_3g)
        self.assertEqual(events_3g, 1)
        self.assertEqual(details_3g[0]["ngram_size"], 3)

    def test_single_repeated_word_exclusion(self):
        # "the the the the" is a repeated unigram, NOT a distinct n-gram phrase
        text = "The the the the dog was barking."
        events, details = self.detector_combined.detect(text)
        self.assertEqual(events, 0)
        self.assertEqual(len(details), 0)

    def test_negative_control_fluent_speech(self):
        text = "A wealthy Athenian was walking through the Agora yesterday morning."
        events, details = self.detector_combined.detect(text)
        self.assertEqual(events, 0)
        self.assertEqual(len(details), 0)

    def test_empty_and_none_transcripts(self):
        self.assertEqual(self.detector_combined.detect("")[0], 0)
        self.assertEqual(self.detector_combined.detect("   ")[0], 0)
        self.assertEqual(self.detector_combined.detect(None)[0], 0)

    def test_integration_with_hybrid_stutter_detector(self):
        # Baseline Fix 3 config (enable_ngram_repetition=False by default)
        acoustic = AcousticFeatureExtractor()
        context = SpeechContextAnalyzer()
        cfg_off = StutterDetectorConfig(enable_ngram_repetition=False)
        det_off = HybridStutterDetector(cfg_off, acoustic, context)

        text = "Or you could, or you could sponsor."
        events_off, details_off = det_off.detect_lexical_repetitions(text)
        self.assertEqual(events_off, 0)

        # Ablation config (enable_ngram_repetition=True)
        cfg_on = StutterDetectorConfig(enable_ngram_repetition=True, ngram_sizes=(2, 3))
        det_on = HybridStutterDetector(cfg_on, acoustic, context)

        events_on, details_on = det_on.detect_lexical_repetitions(text)
        self.assertEqual(events_on, 1)
        self.assertEqual(details_on[0]["type"], "ngram_phrase_repetition_3gram")

        res = det_on.evaluate(transcript=text)
        self.assertEqual(res["has_stutter"], 1)
        self.assertEqual(res["classification"], "CONFIDENT_DISFLUENT")
        self.assertGreaterEqual(res["confidence_score"], 0.85)


if __name__ == "__main__":
    unittest.main()
