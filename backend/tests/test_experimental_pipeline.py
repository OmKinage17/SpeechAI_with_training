"""
Automated Unit and Integration Tests for SpeechAI Experimental Pipeline
======================================================================
Tests:
1. Contextual filler gating (ensuring 'so', 'well', 'like' are not falsely flagged in standard grammar).
2. Acoustic filled-pause validation (schwa formant resonance vs unvoiced breathing/noise).
3. Stutter repetition detection (syllables, words, partial fragments).
4. Acoustic prolongation detection (sustained steady pitch vs dynamic intonation).
5. Acoustic block detection (tense silence + burst vs natural pause).
6. Safe ASR bounded configuration and timeout recovery.
7. Continuous confidence scoring and uncertainty classification.
"""

import os
import sys
import unittest
import numpy as np

# Ensure backend directory is in path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from experimental.config import ExperimentalConfig, ASRConfig, StutterDetectorConfig, FillerDetectorConfig, AcousticConfig
from experimental.acoustic_features import AcousticFeatureExtractor
from experimental.stutter_detector import HybridStutterDetector
from experimental.filler_detector import ContextAwareFillerDetector
from experimental.asr_engine import ExperimentalASREngine

class TestContextualFillerDetector(unittest.TestCase):
    def setUp(self):
        self.config = FillerDetectorConfig()
        self.detector = ContextAwareFillerDetector(self.config)

    def test_ordinary_so_not_counted_as_filler(self):
        """Ordinary grammatical usages of 'so' must NOT be flagged as fillers."""
        test_sentences = [
            "I don't think so.",
            "I hope so too.",
            "It was so good to see you.",
            "He ran so fast that he won the race.",
            "We left early so that we could arrive on time.",
            "There were five or so people in the room.",
            "Why is the weather so cold today?"
        ]
        for sent in test_sentences:
            count, details = self.detector.detect_lexical_fillers(sent)
            self.assertEqual(count, 0, f"False filler detected in ordinary 'so': '{sent}' (details: {details})")

    def test_ordinary_well_not_counted_as_filler(self):
        """Ordinary grammatical usages of 'well' must NOT be flagged as fillers."""
        test_sentences = [
            "She speaks Spanish very well.",
            "He played soccer quite well yesterday.",
            "The patient is getting well soon.",
            "We invited Mary as well as John.",
            "That was a job well done.",
            "This is a well-known scientific fact.",
            "The old water well was twenty feet deep."
        ]
        for sent in test_sentences:
            count, details = self.detector.detect_lexical_fillers(sent)
            self.assertEqual(count, 0, f"False filler detected in ordinary 'well': '{sent}' (details: {details})")

    def test_ordinary_like_not_counted_as_filler(self):
        """Ordinary grammatical usages of 'like' must NOT be flagged as fillers."""
        test_sentences = [
            "I like reading historical novels.",
            "They would like to order dinner now.",
            "It looks like it might rain soon.",
            "The cake tastes like fresh strawberries.",
            "What does the new building look like?",
            "She sings just like her mother."
        ]
        for sent in test_sentences:
            count, details = self.detector.detect_lexical_fillers(sent)
            self.assertEqual(count, 0, f"False filler detected in ordinary 'like': '{sent}' (details: {details})")

    def test_genuine_vocal_fillers_detected(self):
        """Unconditional core vocal fillers must be accurately detected."""
        test_sentences = [
            ("Um, I need a minute to think.", "um"),
            ("The answer is, uh, forty two.", "uh"),
            ("Er, I am not entirely sure.", "er"),
            ("Hmm, let me consider that option.", "hmm"),
            ("Umm, that was strange.", "um")
        ]
        for sent, expected in test_sentences:
            count, details = self.detector.detect_lexical_fillers(sent)
            self.assertGreater(count, 0, f"Expected vocal filler in: '{sent}'")
            matched_texts = [d["text"] for d in details]
            self.assertTrue(any(expected in t for t in matched_texts), f"Expected '{expected}' in {matched_texts}")

    def test_genuine_phrase_fillers_detected(self):
        """Multi-word discourse phrase fillers must be detected."""
        test_sentences = [
            "You know, this is a difficult problem.",
            "I mean, we did our best.",
            "It was sort of an unexpected result.",
            "We were kind of hoping for better weather."
        ]
        for sent in test_sentences:
            count, details = self.detector.detect_lexical_fillers(sent)
            self.assertGreater(count, 0, f"Expected phrase filler in: '{sent}'")

    def test_repeated_discourse_markers_detected(self):
        """Consecutive repeated markers like 'like like' or 'so so' must be detected."""
        sent = "I was like like going to tell him."
        count, details = self.detector.detect_lexical_fillers(sent)
        self.assertGreater(count, 0, f"Expected repeated marker filler in: '{sent}'")

    def test_filler_uncertainty_and_confidence(self):
        """Borderline vs confident filler decisions must have calibrated categories."""
        # Clean sentence
        clean_res = self.detector.evaluate("I really enjoy working on complex engineering challenges.")
        self.assertEqual(clean_res["has_filler"], 0)
        self.assertEqual(clean_res["classification"], "CONFIDENT_CLEAN")

        # Confident filler
        filler_res = self.detector.evaluate("Um, well, uh, you know, we should go.")
        self.assertEqual(filler_res["has_filler"], 1)
        self.assertIn(filler_res["classification"], ["CONFIDENT_FILLER", "UNCERTAIN_BORDERLINE"])
        self.assertGreaterEqual(filler_res["confidence_score"], 0.60)

class TestAcousticFillerDiscrimination(unittest.TestCase):
    def setUp(self):
        self.extractor = AcousticFeatureExtractor(AcousticConfig(sample_rate=16000))
        self.sr = 16000

    def test_formant_resonance_schwa_verification(self):
        """A synthetic /ə/ vocalization (central formant resonance) must pass as filled pause."""
        duration_sec = 0.5
        t = np.linspace(0, duration_sec, int(self.sr * duration_sec), endpoint=False)
        # F0 = 130 Hz with strong F1 (~500 Hz) and F2 (~1500 Hz) formant resonances
        harmonic_f0 = 0.2 * np.sin(2 * np.pi * 130 * t)
        formant_f1 = 0.5 * np.sin(2 * np.pi * 500 * t)
        formant_f2 = 0.3 * np.sin(2 * np.pi * 1500 * t)
        synthetic_schwa = harmonic_f0 + formant_f1 + formant_f2

        res = self.extractor.verify_filled_pause_resonance(synthetic_schwa, 0.0, duration_sec)
        self.assertTrue(res["is_filled_pause"], f"Synthetic schwa failed verification: {res}")
        self.assertGreaterEqual(res["schwa_energy_ratio"], 0.28)
        self.assertLess(res["zcr"], 0.25)

    def test_unvoiced_breathing_noise_rejected(self):
        """Unvoiced breath noise (high ZCR white noise) must be rejected as non-filler."""
        duration_sec = 0.5
        # High-frequency unvoiced turbulence / breath noise
        white_noise = 0.1 * np.random.normal(0, 1, int(self.sr * duration_sec))
        
        res = self.extractor.verify_filled_pause_resonance(white_noise, 0.0, duration_sec)
        self.assertFalse(res["is_filled_pause"], f"Breathing noise was falsely verified as filled pause: {res}")
        self.assertEqual(res.get("reason"), "high_zcr_unvoiced")

    def test_low_frequency_rumble_rejected(self):
        """Microphone handling rumble (< 60 Hz) must be rejected as non-filler."""
        duration_sec = 0.5
        t = np.linspace(0, duration_sec, int(self.sr * duration_sec), endpoint=False)
        rumble = 0.5 * np.sin(2 * np.pi * 45 * t)

        res = self.extractor.verify_filled_pause_resonance(rumble, 0.0, duration_sec)
        self.assertFalse(res["is_filled_pause"], f"Low rumble was falsely verified as filled pause: {res}")

class TestStutterDetection(unittest.TestCase):
    def setUp(self):
        self.detector = HybridStutterDetector()
        self.sr = 16000

    def test_lexical_sound_repetitions(self):
        """Sound repetitions ('w-w-what', 'c-c-can') must be detected."""
        res = self.detector.evaluate("w-w-what did you say to me?")
        self.assertEqual(res["has_stutter"], 1)
        self.assertGreater(res["lexical_repetition_events"], 0)
        self.assertEqual(res["classification"], "CONFIDENT_DISFLUENT")

    def test_lexical_broken_syllable_fragments(self):
        """Broken syllable fragments ('b- because', 't- today') must be detected."""
        res = self.detector.evaluate("I stayed home b- because of the rain.")
        self.assertEqual(res["has_stutter"], 1)
        self.assertGreater(res["lexical_repetition_events"], 0)

    def test_lexical_word_repetitions(self):
        """Whole-word repetitions ('the the', 'I I') must be detected."""
        res = self.detector.evaluate("the the cat ran across the street.")
        self.assertEqual(res["has_stutter"], 1)
        self.assertGreater(res["lexical_repetition_events"], 0)

    def test_fluent_clean_speech_no_stutter(self):
        """Fluent speech without disfluency must have 0 stutters and CONFIDENT_CLEAN."""
        res = self.detector.evaluate("The quick brown fox jumps over the lazy dog.")
        self.assertEqual(res["has_stutter"], 0)
        self.assertEqual(res["total_stutter_events"], 0)
        self.assertEqual(res["classification"], "CONFIDENT_CLEAN")

    def test_acoustic_prolongation_sustained_vowel(self):
        """A sustained steady-state vowel (450ms, stable F0) must be detected as prolongation."""
        duration_sec = 0.50
        t = np.linspace(0, duration_sec, int(self.sr * duration_sec), endpoint=False)
        # 160 Hz steady tone with harmonics
        vowel = 0.4 * np.sin(2 * np.pi * 160 * t) + 0.2 * np.sin(2 * np.pi * 320 * t)

        events, details = self.detector.detect_acoustic_prolongations(vowel, self.sr)
        self.assertGreater(events, 0, f"Failed to detect acoustic prolongation: {details}")
        self.assertEqual(details[0]["type"], "acoustic_prolongation")

    def test_acoustic_block_detection(self):
        """A silent gap (500ms) followed by an explosive burst must be detected as block."""
        total_sec = 1.0
        audio = np.zeros(int(self.sr * total_sec))
        # Silence from 0.0 to 0.5s, then high-energy plosive burst from 0.5s to 0.6s
        burst_len = int(0.1 * self.sr)
        burst = 0.6 * np.sin(2 * np.pi * 250 * np.linspace(0, 0.1, burst_len, endpoint=False))
        audio[int(0.5 * self.sr):int(0.5 * self.sr) + burst_len] = burst

        words_list = [
            {"word": "wordone", "start": 0.0, "end": 0.1},
            {"word": "wordtwo", "start": 0.5, "end": 0.9}
        ]
        events, details = self.detector.detect_acoustic_blocks(audio, words_list)
        self.assertGreater(events, 0, f"Failed to detect acoustic block: {details}")
        self.assertEqual(details[0]["type"], "inter_word_acoustic_block")

class TestASRBoundedDecodingAndResilience(unittest.TestCase):
    def test_asr_config_bounds(self):
        """ASR decoding temperatures must be bounded to at most (0.0, 0.2)."""
        cfg = ASRConfig()
        self.assertEqual(cfg.temperature, (0.0, 0.2))
        self.assertLessEqual(cfg.per_clip_timeout_sec, 30.0)

    def test_asr_safe_execution_nonexistent_file(self):
        """Nonexistent audio files must return structured error without crashing."""
        engine = ExperimentalASREngine()
        res = engine.transcribe_audio_safe("nonexistent_test_audio_12345.wav")
        self.assertEqual(res["status"], "ERROR")
        self.assertEqual(res["text"], "")
        self.assertIn("error", res)

def run_tests():
    suite = unittest.TestLoader().loadTestsFromNames([
        __name__ + ".TestContextualFillerDetector",
        __name__ + ".TestAcousticFillerDiscrimination",
        __name__ + ".TestStutterDetection",
        __name__ + ".TestASRBoundedDecodingAndResilience"
    ])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
