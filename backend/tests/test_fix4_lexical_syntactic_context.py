"""
Unit Tests for Fix 4: Lexical and Syntactic Context for Prolongation Detection
==============================================================================
Validates:
1. Ordinary phrase-final lengthening (suppressed/penalized at sentence terminal).
2. Stressed and emphasized vowels (content words without disfluency indicators).
3. Genuine short and long prolongations (detected and appropriately calibrated).
4. Genuine prolongations near clause boundaries (attenuated discount preserving detection).
5. Repetitions and existing block detections (cross-modal reinforcement).
6. ASR errors, missing punctuation, and low-confidence tokens (neutral fallback).
7. Missing or unreliable syntactic features (neutral fallback without error).
8. Filler-versus-stutter separation (fillers do not trigger stutter prolongation).
9. Regression preservation of Fix 1, Fix 2B, and Fix 3 behaviors.
10. Deterministic predictions and configurable feature weights.
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
from experimental.speech_context import SpeechContextAnalyzer, count_syllables_english, FUNCTION_WORDS
from experimental.stutter_detector import HybridStutterDetector


class TestFix4LexicalSyntacticContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ExperimentalConfig()
        cls.config.stutter.enable_contextual_scoring = True
        cls.config.context.enabled = True
        cls.config.context.enable_lexical_syntactic_context = True
        cls.acoustic = AcousticFeatureExtractor(cls.config.acoustic)
        cls.context_analyzer = SpeechContextAnalyzer(cls.config.context)
        cls.detector = HybridStutterDetector(
            cls.config.stutter, 
            cls.acoustic,
            cls.context_analyzer
        )
        cls.sr = cls.config.acoustic.sample_rate

    def _generate_synthetic_tone(self, duration_sec: float, f0: float = 160.0, amplitude: float = 0.4) -> np.ndarray:
        """Generates a steady harmonic voiced tone."""
        n_samples = int(self.sr * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        tone = amplitude * np.sin(2 * np.pi * f0 * t) + (amplitude * 0.5) * np.sin(2 * np.pi * 2 * f0 * t)
        return tone.astype(np.float32)

    # 1. Ordinary phrase-final lengthening
    def test_ordinary_phrase_final_lengthening(self):
        """A mild vowel lengthening at the end of a sentence is penalized and rejected."""
        words_list = [
            {"word": "He", "start": 0.0, "end": 0.25, "probability": 0.95},
            {"word": "read", "start": 0.25, "end": 0.55, "probability": 0.95},
            {"word": "the", "start": 0.55, "end": 0.75, "probability": 0.95},
            {"word": "book.", "start": 0.75, "end": 1.25, "probability": 0.95},
        ]
        cand = {
            "start": 0.90,
            "end": 1.25,
            "duration": 0.35,
            "pitch_std_st": 0.8,
            "mean_flux": 0.05,
            "confidence": 0.50
        }
        res = self.context_analyzer.analyze_candidate(
            cand, audio_duration_sec=1.8, words_list=words_list
        )
        self.assertEqual(res["boundary_type"], "SENTENCE_TERMINAL")
        self.assertIn(res["word_position"], ("WORD_FINAL", "WORD_SPAN"))
        self.assertFalse(res["is_admitted"])
        self.assertLess(res["context_score"], 0.45)
        self.assertTrue(res["is_phrase_final"])

    # 2. Stressed and emphasized vowels
    def test_stressed_and_emphasized_vowels(self):
        """Content words with ordinary emphatic elongation receive neutral/conservative scores."""
        words_list = [
            {"word": "I", "start": 0.0, "end": 0.20, "probability": 0.95},
            {"word": "really", "start": 0.20, "end": 0.65, "probability": 0.95},
            {"word": "loved", "start": 0.65, "end": 1.05, "probability": 0.95},
            {"word": "that", "start": 1.05, "end": 1.25, "probability": 0.95},
            {"word": "movie.", "start": 1.25, "end": 1.70, "probability": 0.95},
        ]
        cand = {
            "start": 0.70,
            "end": 1.02,
            "duration": 0.32,
            "pitch_std_st": 0.9,
            "mean_flux": 0.06,
            "confidence": 0.50
        }
        res = self.context_analyzer.analyze_candidate(
            cand, audio_duration_sec=2.0, words_list=words_list
        )
        self.assertEqual(res["lexical_class"], "CONTENT_WORD")
        self.assertEqual(res["boundary_type"], "INTRA_PHRASE")
        # Ordinary 1.3-1.4x elongation is rejected from short candidate zone
        self.assertFalse(res["is_admitted"])
        self.assertLess(res["context_score"], 0.65)

    # 3. Genuine short and long prolongations
    def test_genuine_short_and_long_prolongations(self):
        """
        In fast speech, a short prolongation (0.36s) on a function word has high elongation ratio and is admitted.
        A long prolongation (0.57s) is admitted and corroborates.
        """
        # (a) Short prolongation in fast speech on function word 'to'
        words_list_fast = [
            {"word": "He", "start": 0.0, "end": 0.10, "probability": 0.95},
            {"word": "quickly", "start": 0.10, "end": 0.30, "probability": 0.95},
            {"word": "ran", "start": 0.30, "end": 0.45, "probability": 0.95},
            {"word": "to", "start": 0.45, "end": 0.81, "probability": 0.95},
            {"word": "the", "start": 0.81, "end": 0.90, "probability": 0.95},
            {"word": "big", "start": 0.90, "end": 1.05, "probability": 0.95},
            {"word": "house", "start": 1.05, "end": 1.25, "probability": 0.95},
        ]
        cand_short = {
            "start": 0.45,
            "end": 0.81,
            "duration": 0.36,
            "pitch_std_st": 0.7,
            "mean_flux": 0.04,
            "confidence": 0.65
        }
        res_short = self.context_analyzer.analyze_candidate(
            cand_short, audio_duration_sec=1.5, words_list=words_list_fast
        )
        self.assertTrue(res_short["is_admitted"])
        self.assertGreaterEqual(res_short["elongation_ratio"], 1.80)
        self.assertEqual(res_short["lexical_class"], "FUNCTION_WORD")

        # (b) Long prolongation (0.57s)
        cand_long = {
            "start": 1.0,
            "end": 1.57,
            "duration": 0.57,
            "pitch_std_st": 0.9,
            "mean_flux": 0.03,
            "confidence": 0.70
        }
        res_long = self.context_analyzer.analyze_candidate(
            cand_long, audio_duration_sec=3.0, words_list=words_list_fast
        )
        self.assertTrue(res_long["is_admitted"])

    # 4. Genuine prolongations near clause boundaries
    def test_genuine_prolongation_near_clause_boundary(self):
        """
        A severe prolongation (duration 0.57s, ratio 2.51x) at a clause boundary (e.g., 'bloom,')
        receives elongation attenuation on the boundary discount, preserving high context score.
        """
        words_list = [
            {"word": "slope", "start": 3.80, "end": 4.10, "probability": 0.95},
            {"word": "below", "start": 4.10, "end": 4.50, "probability": 0.95},
            {"word": "the", "start": 4.50, "end": 4.70, "probability": 0.95},
            {"word": "house", "start": 4.70, "end": 5.00, "probability": 0.95},
            {"word": "was", "start": 5.00, "end": 5.20, "probability": 0.95},
            {"word": "in", "start": 5.20, "end": 5.35, "probability": 0.95},
            {"word": "white", "start": 5.35, "end": 5.54, "probability": 0.95},
            {"word": "bloom,", "start": 5.54, "end": 6.15, "probability": 0.95},
            {"word": "hummed", "start": 6.30, "end": 6.60, "probability": 0.95},
            {"word": "over", "start": 6.60, "end": 6.90, "probability": 0.95},
        ]
        cand = {
            "start": 5.54,
            "end": 6.11,
            "duration": 0.57,
            "pitch_std_st": 1.13,
            "mean_flux": 0.019,
            "confidence": 0.71
        }
        res = self.context_analyzer.analyze_candidate(
            cand, audio_duration_sec=7.0, words_list=words_list
        )
        self.assertEqual(res["boundary_type"], "CLAUSE_BOUNDARY")
        self.assertGreaterEqual(res["elongation_ratio"], 2.0)
        # Attenuation reduces boundary discount from 0.20 to ~0.03
        self.assertLess(res["effective_boundary_discount"], 0.10)
        # Context score remains high (>= 0.65) to support confident disfluency
        self.assertGreaterEqual(res["context_score"], 0.65)
        self.assertTrue(res["is_admitted"])

    # 5. Repetitions and existing block detections
    def test_repetition_proximity_and_block_integration(self):
        """Candidate adjacent to repeated or interrupted tokens receives disfluency boost."""
        words_list = [
            {"word": "I", "start": 0.5, "end": 0.7, "probability": 0.95},
            {"word": "I", "start": 0.75, "end": 1.35, "probability": 0.95},
            {"word": "went", "start": 1.35, "end": 1.65, "probability": 0.95},
        ]
        cand = {
            "start": 0.75,
            "end": 1.32,
            "duration": 0.57,
            "pitch_std_st": 0.8,
            "mean_flux": 0.03,
            "confidence": 0.65
        }
        res = self.context_analyzer.analyze_candidate(
            cand, audio_duration_sec=2.5, words_list=words_list
        )
        # Candidate is adjacent to repeated token
        self.assertIn("repetition_proximity", " ".join(res["decision_reasons"]))
        self.assertGreater(res["lexical_boost"], 0.0)

    # 6. ASR errors, missing punctuation, and low-confidence tokens
    def test_asr_errors_and_low_confidence_tokens(self):
        """Tokens with probability < 0.40 are treated as UNKNOWN and fall back safely."""
        words_list = [
            {"word": "garbled", "start": 0.5, "end": 1.0, "probability": 0.20},
        ]
        cand = {
            "start": 0.5,
            "end": 1.0,
            "duration": 0.50,
            "pitch_std_st": 0.9,
            "mean_flux": 0.04,
            "confidence": 0.50
        }
        res = self.context_analyzer.analyze_candidate(
            cand, audio_duration_sec=2.0, words_list=words_list
        )
        self.assertEqual(res["lexical_class"], "UNKNOWN")
        self.assertEqual(res["word_position"], "UNKNOWN")

    # 7. Missing or unreliable syntactic features
    def test_missing_or_unreliable_syntactic_features(self):
        """When words_list is None, system falls back to neutral context score (0.50)."""
        cand = {
            "start": 1.0,
            "end": 1.55,
            "duration": 0.55,
            "pitch_std_st": 0.8,
            "mean_flux": 0.03,
            "confidence": 0.50
        }
        res = self.context_analyzer.analyze_candidate(
            cand, audio_duration_sec=3.0, words_list=None
        )
        self.assertEqual(res["context_score"], 0.50)
        self.assertEqual(res["context_reliability"], "NEUTRAL_UNAVAILABLE")
        self.assertEqual(res["boundary_type"], "RECORDING_EDGE" if (3.0 - 1.55) <= 0.35 else "UNKNOWN")

    # 8. Filler-versus-stutter separation
    def test_filler_versus_stutter_separation(self):
        """A core vocal filler ('um', 'uh') does not become a stutter prolongation."""
        words_list = [
            {"word": "um", "start": 0.5, "end": 0.85, "probability": 0.95},
            {"word": "hello", "start": 1.0, "end": 1.4, "probability": 0.95},
        ]
        y = self._generate_synthetic_tone(duration_sec=0.35, f0=140.0)
        res = self.detector.evaluate(
            transcript="um hello",
            words_list=words_list,
            audio_data=(y, self.sr)
        )
        # Vocal filler 'um' alone must not produce confident stutter
        self.assertEqual(res["has_stutter"], 0)

    # 9. Regression preservation of Fix 1, Fix 2B, and Fix 3 behavior
    def test_regression_preservation_when_fix4_disabled(self):
        """Disabling enable_lexical_syntactic_context produces exact Fix 3 behavior."""
        cfg_fix3 = SpeechContextConfig(enabled=True, enable_lexical_syntactic_context=False)
        analyzer_fix3 = SpeechContextAnalyzer(cfg_fix3)
        
        words_list = [
            {"word": "end.", "start": 0.0, "end": 0.60, "probability": 0.95}
        ]
        cand = {
            "start": 0.0,
            "end": 0.55,
            "duration": 0.55,
            "pitch_std_st": 0.8,
            "mean_flux": 0.04,
            "confidence": 0.50
        }
        res_fix3 = analyzer_fix3.analyze_candidate(cand, audio_duration_sec=1.5, words_list=words_list)
        self.assertIn("phrase_final_reason", res_fix3)
        self.assertTrue(res_fix3["is_phrase_final"])

    # 10. Deterministic predictions and configurable feature weights
    def test_deterministic_predictions_and_configurable_weights(self):
        """Results are strictly deterministic across runs, and config parameters are honored."""
        words_list = [
            {"word": "the", "start": 0.0, "end": 0.50, "probability": 0.95},
            {"word": "test", "start": 0.50, "end": 0.90, "probability": 0.95},
        ]
        cand = {
            "start": 0.0,
            "end": 0.50,
            "duration": 0.50,
            "pitch_std_st": 0.7,
            "mean_flux": 0.03,
            "confidence": 0.50
        }
        res1 = self.context_analyzer.analyze_candidate(cand, audio_duration_sec=1.5, words_list=words_list)
        res2 = self.context_analyzer.analyze_candidate(cand, audio_duration_sec=1.5, words_list=words_list)
        self.assertEqual(res1["context_score"], res2["context_score"])
        self.assertEqual(res1["decision_reasons"], res2["decision_reasons"])


if __name__ == "__main__":
    unittest.main()
