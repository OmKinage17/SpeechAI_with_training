"""
Unit Tests for Ablation 3: Interrupted & Filler-Bridged Repetition Detection
=============================================================================
Tests:
1. Interrupted 2-gram repetitions (e.g. "it depends, it sort of depends").
2. Interrupted word repetitions (e.g. "still, obviously, still").
3. Stem-level repetition variations (e.g. "school, the schools").
4. Multiple hesitation tokens between repeated phrases (e.g. "time, and, er, at time").
5. Strict negative controls on fluent speech with conversational fillers.
6. Exclusion of grammatical stop particles from single-word repetition matching.
7. Proper timestamp alignment from words_list.
8. Backward compatibility when disabled by default in HybridStutterDetector.
9. Integration with HybridStutterDetector when enabled.
"""

import unittest
from backend.experimental.interrupted_repetition import InterruptedRepetitionDetector, clean_token, basic_stem
from backend.experimental.config import StutterDetectorConfig
from backend.experimental.stutter_detector import HybridStutterDetector


class TestInterruptedRepetitionDetector(unittest.TestCase):
    def setUp(self):
        self.detector = InterruptedRepetitionDetector(
            max_skip=3,
            allow_stem=True,
            allow_hesitations=True,
            confidence=0.85
        )

    def test_clean_token(self):
        self.assertEqual(clean_token("Depends,"), "depends")
        self.assertEqual(clean_token("obviously..."), "obviously")
        self.assertEqual(clean_token("don't"), "don't")
        self.assertEqual(clean_token(""), "")

    def test_basic_stem(self):
        self.assertEqual(basic_stem("schools"), "school")
        self.assertEqual(basic_stem("stuttering"), "stutter")
        self.assertEqual(basic_stem("hurried"), "hurry")
        self.assertEqual(basic_stem("cat"), "cat")

    def test_interrupted_2gram_repetition_with_discourse_filler(self):
        transcript = "Every lesson today, it depends, it sort of depends what day it is."
        words_list = [
            {"word": "Every", "start": 0.0, "end": 0.14},
            {"word": "lesson", "start": 0.14, "end": 0.30},
            {"word": "today,", "start": 0.30, "end": 0.66},
            {"word": "it", "start": 0.94, "end": 0.98},
            {"word": "depends,", "start": 0.98, "end": 1.26},
            {"word": "it", "start": 1.54, "end": 1.62},
            {"word": "sort", "start": 1.62, "end": 1.76},
            {"word": "of", "start": 1.76, "end": 1.86},
            {"word": "depends", "start": 1.86, "end": 2.08},
            {"word": "what", "start": 2.08, "end": 2.28},
            {"word": "day", "start": 2.28, "end": 2.50},
            {"word": "it", "start": 2.50, "end": 2.60},
            {"word": "is.", "start": 2.60, "end": 2.74},
        ]
        events, details = self.detector.detect(transcript, words_list)
        self.assertGreaterEqual(events, 1)
        self.assertTrue(any("it depends" in d.get("phrase", "") for d in details))
        self.assertTrue(any("sort of" in d.get("intervening_fillers", "") for d in details))
        self.assertEqual(details[0]["confidence"], 0.85)
        self.assertIn("start", details[0])
        self.assertIn("end", details[0])

    def test_interrupted_single_word_repetition(self):
        transcript = "I still, obviously, still stutter, although I-"
        words_list = [
            {"word": "I", "start": 0.0, "end": 0.14},
            {"word": "still,", "start": 0.14, "end": 1.30},
            {"word": "obviously,", "start": 1.46, "end": 1.98},
            {"word": "still", "start": 2.20, "end": 2.28},
            {"word": "stutter,", "start": 2.28, "end": 2.58},
            {"word": "although", "start": 2.68, "end": 2.80},
            {"word": "I-", "start": 2.80, "end": 2.98},
        ]
        events, details = self.detector.detect(transcript, words_list)
        self.assertGreaterEqual(events, 1)
        self.assertEqual(details[0]["phrase"], "still")
        self.assertIn("obviously", details[0]["intervening_fillers"])

    def test_hesitation_phrase_repetition(self):
        transcript = "for time, and, er, at time, and not caring"
        events, details = self.detector.detect(transcript)
        self.assertGreaterEqual(events, 1)
        self.assertTrue(any("time" in d.get("phrase", "") for d in details))

    def test_stem_level_repetition(self):
        transcript = "Like this school, the schools, it's-"
        events, details = self.detector.detect(transcript)
        self.assertGreaterEqual(events, 1)
        self.assertTrue(any("school" in d.get("phrase", "") for d in details))

    def test_negative_fluent_controls(self):
        # Fluent sentences with discourse fillers should NOT match
        fluent_examples = [
            "I really like pizza and pasta.",
            "Um, well, actually that was an amazing movie yesterday.",
            "You know what I mean about the weather.",
            "Basically we went to the market and then to the library.",
            "She thinks so because of the results."
        ]
        for ex in fluent_examples:
            events, details = self.detector.detect(ex)
            self.assertEqual(events, 0, f"False positive detected on fluent text: '{ex}' -> {details}")

    def test_stop_particles_not_matched(self):
        # Closed-class particles like 'to', 'the', 'in' should NOT trigger single-word matches
        examples = [
            "We went to the store and to the gym.",
            "The dog chased the cat into the garden.",
            "It was in the box and in the bag."
        ]
        for ex in examples:
            events, details = self.detector.detect(ex)
            # Ensure no single-word repetition triggers on 'to', 'the', 'in'
            single_word_events = [d for d in details if d["type"] == "interrupted_word_repetition"]
            self.assertEqual(len(single_word_events), 0, f"Stop particle triggered on: '{ex}' -> {details}")


class TestHybridStutterDetectorAblation3Integration(unittest.TestCase):
    def test_disabled_by_default(self):
        # Default config has enable_interrupted_repetition=False
        cfg = StutterDetectorConfig()
        self.assertFalse(cfg.enable_interrupted_repetition)
        detector = HybridStutterDetector(config=cfg)
        
        # Transcript with interrupted repetition
        text = "Every lesson today, it depends, it sort of depends what day it is."
        events, details = detector.detect_lexical_repetitions(text)
        # Should be 0 because interrupted repetition is disabled
        self.assertEqual(events, 0)
        self.assertEqual(len(details), 0)

    def test_enabled_integration(self):
        # When enabled, detector captures interrupted repetitions
        cfg = StutterDetectorConfig(
            enable_interrupted_repetition=True,
            interrupted_repetition_max_skip=3,
            interrupted_repetition_allow_stem=True,
            interrupted_repetition_allow_hesitations=True
        )
        detector = HybridStutterDetector(config=cfg)
        
        text = "Every lesson today, it depends, it sort of depends what day it is."
        events, details = detector.detect_lexical_repetitions(text)
        self.assertGreaterEqual(events, 1)
        self.assertTrue(any("interrupted" in d.get("type", "") for d in details))

        # Full evaluation produces confident stutter
        res = detector.evaluate(transcript=text)
        self.assertEqual(res["has_stutter"], 1)
        self.assertGreaterEqual(res["confidence_score"], 0.85)


if __name__ == "__main__":
    unittest.main()
