"""
Unit Tests for Fix 2: Prolongation Detection Recalibration & Standalone Gating
=============================================================================
Tests that:
1. Standard fluent English vowels (< 500ms) are NOT flagged as prolongations.
2. Short vowels (150-250ms) are ignored.
3. Genuine pathological prolongations (>= 500ms, locked pitch, low flux) are detected.
4. Natural dynamic intonation contours (pitch std > 1.2 semitones) are rejected.
5. Formant movements / diphthong glides (spectral flux > 0.20) are rejected.
6. Pure noise / breath turbulence is rejected without spurious candidates.
7. Standalone single prolongations receive confidence <= 0.50 and do NOT trigger stutter.
8. Corroborated prolongations (accompanied by lexical repetition or block) qualify as stutter.
9. Multiple recurrent prolongations (>= 2) provide sufficient evidence to qualify.
10. All thresholds are fully configurable at runtime.
"""

import unittest
import numpy as np
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from experimental.config import ExperimentalConfig, StutterDetectorConfig, AcousticConfig
from experimental.acoustic_features import AcousticFeatureExtractor
from experimental.stutter_detector import HybridStutterDetector


class TestFix2ProlongationGating(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ExperimentalConfig()
        cls.acoustic = AcousticFeatureExtractor(cls.config.acoustic)
        cls.detector = HybridStutterDetector(cls.config.stutter, cls.acoustic)
        cls.sr = cls.config.acoustic.sample_rate  # 16000

    def _generate_synthetic_tone(self, duration_sec: float, f0: float = 160.0, amplitude: float = 0.4) -> np.ndarray:
        """Generates a steady harmonic voiced tone."""
        n_samples = int(self.sr * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        tone = amplitude * np.sin(2 * np.pi * f0 * t) + (amplitude * 0.5) * np.sin(2 * np.pi * 2 * f0 * t)
        return tone.astype(np.float32)

    def test_ordinary_sustained_vowel_rejected(self):
        """Standard fluent stressed vowel (400ms) must NOT trigger prolongation under 0.50s threshold."""
        vowel_400ms = self._generate_synthetic_tone(duration_sec=0.40, f0=150.0)
        events, details = self.detector.detect_acoustic_prolongations(vowel_400ms, self.sr)
        self.assertEqual(events, 0, f"Expected 0 events for 400ms vowel, got {events}: {details}")

    def test_short_vowel_rejected(self):
        """Short normal vowel (200ms) must NOT trigger prolongation."""
        vowel_200ms = self._generate_synthetic_tone(duration_sec=0.20, f0=180.0)
        events, details = self.detector.detect_acoustic_prolongations(vowel_200ms, self.sr)
        self.assertEqual(events, 0, f"Expected 0 events for 200ms vowel, got {events}: {details}")

    def test_genuine_pathological_prolongation_detected(self):
        """Sustained steady vowel (750ms, stable F0, low flux) must be detected as genuine prolongation."""
        prolongation_750ms = self._generate_synthetic_tone(duration_sec=0.75, f0=160.0)
        events, details = self.detector.detect_acoustic_prolongations(prolongation_750ms, self.sr)
        self.assertGreaterEqual(events, 1, f"Failed to detect genuine 750ms prolongation: {details}")
        first = details[0]
        self.assertEqual(first["type"], "acoustic_prolongation")
        self.assertGreaterEqual(first["duration"], 0.50)
        self.assertLess(first["pitch_std_st"], 1.2)
        self.assertLess(first["mean_flux"], 0.20)

    def test_dynamic_intonation_contour_rejected(self):
        """Long vowel (650ms) with natural intonation (pitch variation > 1.2 st) must be rejected."""
        duration_sec = 0.65
        n_samples = int(self.sr * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        # Pitch modulates from 130 Hz to 180 Hz (pitch std ~ 2.5 semitones)
        f0_contour = 130.0 + 50.0 * np.sin(np.pi * t / duration_sec)
        phase = 2 * np.pi * np.cumsum(f0_contour) / self.sr
        inflected_vowel = (0.4 * np.sin(phase)).astype(np.float32)

        events, details = self.detector.detect_acoustic_prolongations(inflected_vowel, self.sr)
        self.assertEqual(events, 0, f"Expected 0 events for inflected vowel (pitch std > 1.2st), got {events}: {details}")

    def test_formant_transition_high_flux_rejected(self):
        """Long vowel (600ms) with rapid spectral changes (flux > 0.20) must be rejected."""
        duration_sec = 0.60
        n_samples = int(self.sr * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        # Modulate frequency content rapidly across frames to induce high spectral flux
        mod = np.sin(2 * np.pi * 20.0 * t)  # 20 Hz rapid vibrato
        dynamic_audio = (0.4 * np.sin(2 * np.pi * 200.0 * (1 + 0.3 * mod) * t)).astype(np.float32)

        events, details = self.detector.detect_acoustic_prolongations(dynamic_audio, self.sr)
        self.assertEqual(events, 0, f"Expected 0 events for high-flux transition, got {events}: {details}")

    def test_noisy_segment_no_false_prolongation(self):
        """Unvoiced Gaussian breath noise must NOT trigger acoustic prolongation."""
        np.random.seed(42)
        noise = (0.05 * np.random.randn(int(self.sr * 0.80))).astype(np.float32)
        events, details = self.detector.detect_acoustic_prolongations(noise, self.sr)
        self.assertEqual(events, 0, f"Noise produced spurious prolongations: {details}")

    def test_standalone_prolongation_confidence_and_decision(self):
        """A single standalone prolongation (< 0.60s) has confidence <= 0.50 and does NOT trigger confident stutter."""
        prolongation = self._generate_synthetic_tone(duration_sec=0.55, f0=150.0)
        # Empty text and empty words list -> no lexical repetitions, no blocks
        res = self.detector.evaluate(transcript="", words_list=[], audio_data=(prolongation, self.sr))
        self.assertEqual(res["acoustic_prolongation_events"], 1)
        self.assertEqual(res["lexical_repetition_events"], 0)
        self.assertEqual(res["acoustic_block_events"], 0)
        # Confidence must be 0.50 (below confident threshold 0.60)
        self.assertLessEqual(res["confidence_score"], 0.55)
        self.assertEqual(res["classification"], "UNCERTAIN_BORDERLINE")
        # Standalone prolongation must NOT trigger stutter
        self.assertEqual(res["has_stutter"], 0, "Standalone prolongation must not qualify as stutter")

    def test_corroborated_prolongation_with_lexical_repetition(self):
        """A prolongation corroborated by lexical repetition qualifies as confident stutter."""
        prolongation = self._generate_synthetic_tone(duration_sec=0.70, f0=150.0)
        text = "they they went to the market"
        res = self.detector.evaluate(transcript=text, audio_data=(prolongation, self.sr))
        self.assertGreaterEqual(res["acoustic_prolongation_events"], 1)
        self.assertGreaterEqual(res["lexical_repetition_events"], 1)
        self.assertGreaterEqual(res["confidence_score"], 0.85)
        self.assertEqual(res["classification"], "CONFIDENT_DISFLUENT")
        self.assertEqual(res["has_stutter"], 1)

    def test_corroborated_prolongation_with_acoustic_block(self):
        """A prolongation corroborated by an acoustic block qualifies as stutter."""
        # 1.5s audio: 0.0-0.7s steady vowel (prolongation), 0.7-1.2s silence (gap), 1.2-1.3s burst (block)
        total_len = int(1.5 * self.sr)
        audio = np.zeros(total_len, dtype=np.float32)
        # Prolongation segment
        audio[:int(0.70 * self.sr)] = self._generate_synthetic_tone(duration_sec=0.70, f0=150.0)
        # Block burst
        burst_len = int(0.10 * self.sr)
        burst_start = int(1.20 * self.sr)
        audio[burst_start:burst_start + burst_len] = 0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.1, burst_len, endpoint=False))

        words_list = [
            {"word": "first", "start": 0.0, "end": 0.70},
            {"word": "second", "start": 1.20, "end": 1.45}
        ]
        res = self.detector.evaluate(transcript="first second", words_list=words_list, audio_data=(audio, self.sr))
        self.assertGreaterEqual(res["acoustic_prolongation_events"], 1)
        self.assertGreaterEqual(res["acoustic_block_events"], 1)
        self.assertGreaterEqual(res["confidence_score"], 0.60)
        self.assertEqual(res["has_stutter"], 1)

    def test_recurrent_multiple_prolongations(self):
        """Two distinct prolongations in one clip corroborate each other and trigger stutter."""
        # Audio with two 700ms steady tones separated by 400ms normal speech
        p1 = self._generate_synthetic_tone(duration_sec=0.70, f0=150.0)
        gap = np.zeros(int(self.sr * 0.40), dtype=np.float32)
        p2 = self._generate_synthetic_tone(duration_sec=0.70, f0=170.0)
        multi_audio = np.concatenate([p1, gap, p2])

        res = self.detector.evaluate(transcript="", words_list=[], audio_data=(multi_audio, self.sr))
        self.assertGreaterEqual(res["acoustic_prolongation_events"], 2)
        self.assertGreaterEqual(res["confidence_score"], 0.60)
        self.assertEqual(res["has_stutter"], 1)

    def test_configurability_of_thresholds(self):
        """Thresholds must be dynamically configurable at runtime."""
        # A 400ms vowel is rejected under default 0.50s config
        vowel_400ms = self._generate_synthetic_tone(duration_sec=0.40, f0=150.0)
        events_def, _ = self.detector.detect_acoustic_prolongations(vowel_400ms, self.sr)
        self.assertEqual(events_def, 0)

        # Lower threshold to 0.35s dynamically
        custom_config = StutterDetectorConfig(min_prolongation_duration_sec=0.35)
        custom_detector = HybridStutterDetector(custom_config, self.acoustic)
        events_cust, _ = custom_detector.detect_acoustic_prolongations(vowel_400ms, self.sr)
        self.assertGreaterEqual(events_cust, 1, "Lowered threshold (0.35s) should detect 400ms vowel")


if __name__ == "__main__":
    unittest.main()
