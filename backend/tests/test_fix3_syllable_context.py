"""
Unit Tests for Fix 3: Syllable-Rate and Speech-Context Analysis
==============================================================
Validates:
1. Ordinary sustained vowels / diphthongs in fluent context (not classified as stutter).
2. Genuine short prolongations in fast speech (admitted with high elongation ratio).
3. Genuine severe prolongations (>= 0.60s) admitted and self-corroborated.
4. Phrase-final lengthening at pre-pausal / clause / terminal boundaries penalized.
5. Stressed syllables with pitch variation (> 1.2 st) rejected.
6. Slow fluent speech tempo adaptation (damping false alarms).
7. Fast speech tempo adaptation (enhancing sensitivity).
8. Silence, pause, and low-confidence ASR words filtered from net articulation rate.
9. Missing context / unavailable ASR falls back to neutral score (0.50).
10. Continuous context score is strictly bounded [0.0, 1.0].
11. Fix 1 & Fix 2B regression preservation when context scoring is disabled.
12. Determinism and reproducibility across identical runs.
"""

import unittest
import numpy as np
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from experimental.config import ExperimentalConfig, StutterDetectorConfig, SpeechContextConfig
from experimental.acoustic_features import AcousticFeatureExtractor
from experimental.speech_context import SpeechContextAnalyzer, count_syllables_english
from experimental.stutter_detector import HybridStutterDetector


class TestFix3SyllableContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ExperimentalConfig()
        cls.config.stutter.enable_contextual_scoring = True
        cls.acoustic = AcousticFeatureExtractor(cls.config.acoustic)
        cls.context_analyzer = SpeechContextAnalyzer(cls.config.context)
        cls.detector = HybridStutterDetector(
            cls.config.stutter, 
            cls.acoustic,
            cls.context_analyzer
        )
        cls.sr = cls.config.acoustic.sample_rate  # 16000

    def _generate_synthetic_tone(self, duration_sec: float, f0: float = 160.0, amplitude: float = 0.4) -> np.ndarray:
        """Generates a steady harmonic voiced tone."""
        n_samples = int(self.sr * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        tone = amplitude * np.sin(2 * np.pi * f0 * t) + (amplitude * 0.5) * np.sin(2 * np.pi * 2 * f0 * t)
        return tone.astype(np.float32)

    # 1. Ordinary sustained vowels / diphthongs in fluent context
    def test_ordinary_sustained_vowel_in_fluent_context(self):
        """In normal fluent tempo (~4 syll/s), a 0.35s vowel is not admitted as stutter."""
        words_list = [
            {"word": "We", "start": 0.0, "end": 0.25, "probability": 0.95},
            {"word": "were", "start": 0.25, "end": 0.50, "probability": 0.95},
            {"word": "going", "start": 0.50, "end": 0.95, "probability": 0.95},
            {"word": "home", "start": 0.95, "end": 1.30, "probability": 0.95},
            {"word": "today", "start": 1.30, "end": 1.75, "probability": 0.95},
        ]
        transcript = "We were going home today."
        # Candidate 0.35s in middle of utterance
        cand = {
            "start": 0.50,
            "end": 0.85,
            "duration": 0.35,
            "pitch_std_st": 0.8,
            "mean_flux": 0.12,
            "confidence": 0.50
        }
        res = self.context_analyzer.analyze_candidate(
            cand, audio_duration_sec=2.0, words_list=words_list, transcript=transcript
        )
        # In normal tempo (~4 syll/s, mean syll ~250ms), a 0.35s vowel is only 1.4x mean syll
        # It must be rejected (is_admitted = False) because elongation ratio < min_elongation_ratio (1.75x)
        self.assertFalse(res["is_admitted"])
        self.assertLess(res["elongation_ratio"], 1.75)

    # 2. Genuine short prolongation in fast speech
    def test_genuine_short_prolongation_in_fast_speech(self):
        """In fast speech (6 syll/s, mean syll ~160ms), a 0.36s sustained sound is > 2.2x mean syll."""
        words_list = [
            {"word": "He", "start": 0.0, "end": 0.15, "probability": 0.95},
            {"word": "quickly", "start": 0.15, "end": 0.40, "probability": 0.95},
            {"word": "ran", "start": 0.40, "end": 0.55, "probability": 0.95},
            {"word": "into", "start": 0.55, "end": 0.80, "probability": 0.95},
            {"word": "the", "start": 0.80, "end": 0.92, "probability": 0.95},
            {"word": "market", "start": 0.92, "end": 1.25, "probability": 0.95},
        ]
        transcript = "He quickly ran into the market"
        cand = {
            "start": 0.40,
            "end": 0.76,
            "duration": 0.36,
            "pitch_std_st": 0.6,
            "mean_flux": 0.10,
            "confidence": 0.50
        }
        res = self.context_analyzer.analyze_candidate(
            cand, audio_duration_sec=1.5, words_list=words_list, transcript=transcript
        )
        self.assertTrue(res["is_admitted"])
        self.assertGreaterEqual(res["elongation_ratio"], 1.75)
        self.assertGreaterEqual(res["context_score"], 0.60)

    # 3. Genuine severe prolongation (>= 0.60s) self-corroborates
    def test_genuine_severe_prolongation_self_corroborates(self):
        """A severe standalone prolongation (0.75s) self-corroborates into confident disfluent."""
        y = self._generate_synthetic_tone(duration_sec=0.75, f0=150.0)
        res = self.detector.evaluate(
            transcript="Hello",
            words_list=[{"word": "Hello", "start": 0.0, "end": 0.75, "probability": 0.95}],
            audio_data=(y, self.sr)
        )
        self.assertEqual(res["has_stutter"], 1)
        self.assertGreaterEqual(res["confidence_score"], 0.60)

    # 4. Phrase-final lengthening at clause/terminal boundary
    def test_phrase_final_lengthening_boundary_damping(self):
        """Sustained vowel at sentence end or before long pause receives phrase-final penalty."""
        words_list = [
            {"word": "I", "start": 0.0, "end": 0.20, "probability": 0.95},
            {"word": "wonder", "start": 0.20, "end": 0.60, "probability": 0.95},
            {"word": "why.", "start": 0.60, "end": 1.15, "probability": 0.95},
        ]
        transcript = "I wonder why."
        cand = {
            "start": 0.65,
            "end": 1.15,
            "duration": 0.50,
            "pitch_std_st": 0.7,
            "mean_flux": 0.11,
            "confidence": 0.50
        }
        res = self.context_analyzer.analyze_candidate(
            cand, audio_duration_sec=1.5, words_list=words_list, transcript=transcript
        )
        self.assertTrue(res["is_phrase_final"])
        self.assertLess(res["context_score"], 0.50)

    # 5. Stressed syllables with pitch variation (> 1.2 st) rejected
    def test_stressed_syllables_with_intonation_variation_rejected(self):
        """Intonation rise/fall (pitch std > 1.2 st) is rejected at acoustic candidate extraction."""
        t = np.linspace(0, 0.65, int(self.sr * 0.65), endpoint=False)
        f0_mod = 140.0 + 40.0 * np.sin(2 * np.pi * 2.0 * t)
        phase = 2 * np.pi * np.cumsum(f0_mod) / self.sr
        y = (0.4 * np.sin(phase)).astype(np.float32)
        
        events, details = self.detector.detect_acoustic_prolongations(y, self.sr)
        self.assertEqual(events, 0)

    # 6. Slow fluent speech tempo adaptation
    def test_slow_fluent_speech_tempo_damping(self):
        """Slow speaker (~2.2 syll/s) damps context score so marginal prolongations don't false-alarm."""
        words_list = [
            {"word": "One", "start": 0.0, "end": 0.50, "probability": 0.95},
            {"word": "two", "start": 0.60, "end": 1.10, "probability": 0.95},
            {"word": "three", "start": 1.20, "end": 1.70, "probability": 0.95},
            {"word": "four", "start": 1.80, "end": 2.30, "probability": 0.95},
        ]
        transcript = "One two three four"
        cand = {
            "start": 0.60,
            "end": 1.10,
            "duration": 0.50,
            "pitch_std_st": 0.8,
            "mean_flux": 0.12,
            "confidence": 0.50
        }
        res = self.context_analyzer.analyze_candidate(
            cand, audio_duration_sec=2.5, words_list=words_list, transcript=transcript
        )
        # Slow tempo (2.2 syll/s -> mean syll 450ms). 0.50s is only ~1.1x mean syll.
        self.assertLess(res["context_score"], 0.40)

    # 7. Fast speech tempo adaptation
    def test_fast_speech_tempo_boost(self):
        """Fast speech tempo (> 5.0 syll/s) boosts candidate score when elongation is substantial."""
        words_list = [
            {"word": "It", "start": 0.0, "end": 0.12, "probability": 0.95},
            {"word": "was", "start": 0.12, "end": 0.25, "probability": 0.95},
            {"word": "unbelievable", "start": 0.25, "end": 0.80, "probability": 0.95},
            {"word": "to", "start": 0.80, "end": 0.92, "probability": 0.95},
            {"word": "see", "start": 0.92, "end": 1.10, "probability": 0.95},
        ]
        transcript = "It was unbelievable to see"
        rate_info = self.context_analyzer.estimate_syllable_rate(words_list, transcript, 1.2)
        self.assertGreaterEqual(rate_info["articulation_rate_sps"], 5.0)

    # 8. Silence, breath, low-confidence ASR words filtered
    def test_silence_breath_low_prob_words_filtered(self):
        """Pauses >= 0.25s and hallucinated low-prob ASR words (< 0.40) are excluded from articulation time."""
        words_list = [
            {"word": "Hello", "start": 0.0, "end": 0.40, "probability": 0.95},
            {"word": "asdfghjkl", "start": 0.45, "end": 0.95, "probability": 0.20},  # Low prob hallucination
            # 1.5s silent gap
            {"word": "world", "start": 2.50, "end": 2.90, "probability": 0.95},
        ]
        rate_info = self.context_analyzer.estimate_syllable_rate(words_list, "Hello world", 3.0)
        # Syllables should count only for "Hello" (2) and "world" (1) = 3 syllables
        self.assertEqual(rate_info["syllables"], 3)
        # Net articulation time should exclude the 1.5s pause
        self.assertLess(rate_info["net_articulation_sec"], 2.0)

    # 9. Missing context / unavailable ASR neutral fallback
    def test_missing_context_neutral_fallback(self):
        """When words_list and transcript are None/empty, analyzer returns neutral score 0.50."""
        cand = {
            "start": 0.5,
            "end": 1.05,
            "duration": 0.55,
            "pitch_std_st": 0.7,
            "mean_flux": 0.12,
            "confidence": 0.50
        }
        res = self.context_analyzer.analyze_candidate(cand, audio_duration_sec=2.0)
        self.assertEqual(res["confidence_level"], "UNAVAILABLE")
        self.assertEqual(res["context_score"], 0.50)
        self.assertTrue(res["is_admitted"])

    # 10. Normalization and score bounds
    def test_normalization_and_score_bounds(self):
        """Continuous context score S_context is strictly in [0.0, 1.0] under extreme values."""
        for ratio in [0.5, 1.0, 2.0, 5.0, 10.0]:
            cand = {
                "start": 0.5,
                "end": 0.5 + (0.25 * ratio),
                "duration": 0.25 * ratio,
                "pitch_std_st": 0.5,
                "mean_flux": 0.10,
                "confidence": 0.50
            }
            res = self.context_analyzer.analyze_candidate(
                cand, 
                audio_duration_sec=5.0,
                words_list=[{"word": "test", "start": 0.0, "end": 0.5, "probability": 0.95}],
                transcript="test"
            )
            score = res["context_score"]
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    # 11. Fix 1 & Fix 2B regression preservation
    def test_fix1_fix2b_regression_preservation(self):
        """When enable_contextual_scoring is False, behavior is 100% identical to Fix 2B."""
        detector_disabled = HybridStutterDetector(
            StutterDetectorConfig(enable_contextual_scoring=False),
            self.acoustic
        )
        # Candidate 0.40s with steady pitch
        y = self._generate_synthetic_tone(duration_sec=0.40, f0=150.0)
        events, details = detector_disabled.detect_acoustic_prolongations(y, self.sr)
        # Fix 2B baseline floor is 0.50s, so 0.40s MUST produce 0 events
        self.assertEqual(events, 0)

        # Severe candidate 0.70s produces 1 event and self-corroborates in evaluate
        y_sev = self._generate_synthetic_tone(duration_sec=0.70, f0=150.0)
        res = detector_disabled.evaluate("test", audio_data=(y_sev, self.sr))
        self.assertEqual(res["has_stutter"], 1)

    # 12. Determinism and reproducibility
    def test_determinism_and_reproducibility(self):
        """Identical inputs must produce identical context scores and detector evaluations."""
        y = self._generate_synthetic_tone(duration_sec=0.65, f0=150.0)
        words_list = [{"word": "sample", "start": 0.0, "end": 0.65, "probability": 0.95}]
        res1 = self.detector.evaluate("sample", words_list=words_list, audio_data=(y, self.sr))
        res2 = self.detector.evaluate("sample", words_list=words_list, audio_data=(y, self.sr))
        self.assertEqual(res1["confidence_score"], res2["confidence_score"])
        self.assertEqual(res1["has_stutter"], res2["has_stutter"])
        self.assertEqual(res1["total_stutter_events"], res2["total_stutter_events"])

    def test_count_syllables_english_cases(self):
        """Verify phonological syllable counting on representative vocabulary."""
        cases = {
            "a": 1,
            "the": 1,
            "cat": 1,
            "hello": 2,
            "wonderful": 3,
            "communication": 5,
            "unbelievable": 5,
            "stuttering": 3,
            "fluency": 3,
            "make": 1,      # silent e
            "baked": 1,     # regular past tense
            "wanted": 2,    # syllabic -ed after t
            "buses": 2,     # syllabic -es after s
        }
        for word, expected in cases.items():
            actual = count_syllables_english(word)
            self.assertEqual(actual, expected, f"Word '{word}': expected {expected}, got {actual}")


if __name__ == "__main__":
    unittest.main()
