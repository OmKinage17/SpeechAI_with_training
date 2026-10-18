"""
Unit Tests for Fix 1: Acoustic Filler Energy Floor and Voicing Gating
=====================================================================
Tests that verify_filled_pause_resonance() and detect_acoustic_fillers():
1. Reject pure silence (below RMS floor).
2. Reject near-silence room tone (below RMS floor).
3. Reject unvoiced breath noise (high ZCR / zero voicing).
4. Reject ordinary quiet speech pauses between words.
5. Accept genuine voiced vocalic hesitations (/ə/ "uh" and /m/ "um").
6. Preserve lexical filler detection for audible words ("um", "uh", "er").
7. Scale energy gating adaptively when vocal_threshold is provided.
"""

import unittest
import numpy as np
from scipy.signal import butter, lfilter

from experimental.config import ExperimentalConfig, AcousticConfig
from experimental.acoustic_features import AcousticFeatureExtractor
from experimental.filler_detector import ContextAwareFillerDetector


class TestFix1AcousticFillerGating(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentalConfig()
        self.acoustic = AcousticFeatureExtractor(self.config.acoustic)
        self.filler_detector = ContextAwareFillerDetector(self.config.filler, self.acoustic)
        self.sr = self.config.acoustic.sample_rate  # 16000

    def test_pure_silence_rejected(self):
        """Pure digital silence (zeros) must be rejected with below_energy_floor."""
        dur_sec = 0.5
        silence = np.zeros(int(self.sr * dur_sec), dtype=np.float32)
        res = self.acoustic.verify_filled_pause_resonance(silence, 0.0, dur_sec)

        self.assertFalse(res["is_filled_pause"])
        self.assertEqual(res.get("reason"), "below_energy_floor")
        self.assertEqual(res.get("seg_rms"), 0.0)
        self.assertLess(res.get("seg_rms", 0.0), self.config.acoustic.min_filled_pause_rms)

    def test_near_silence_studio_room_tone_rejected(self):
        """Quiet studio room tone (RMS ~0.002 - 0.0035) must be rejected with below_energy_floor."""
        dur_sec = 0.5
        # Simulate typical LibriSpeech studio silence: Gaussian room tone with RMS = 0.003
        np.random.seed(42)
        room_tone = np.random.normal(0, 0.003, int(self.sr * dur_sec)).astype(np.float32)
        
        res = self.acoustic.verify_filled_pause_resonance(room_tone, 0.0, dur_sec)
        self.assertFalse(res["is_filled_pause"])
        self.assertEqual(res.get("reason"), "below_energy_floor")
        self.assertLess(res.get("seg_rms"), self.config.acoustic.min_filled_pause_rms)

    def test_unvoiced_white_breath_noise_rejected(self):
        """Unvoiced breath / aspiration noise must be rejected as unvoiced / high ZCR."""
        dur_sec = 0.5
        np.random.seed(123)
        # Moderate energy breath turbulence (RMS ~0.05, above floor, but unvoiced with high ZCR)
        breath_noise = np.random.normal(0, 0.05, int(self.sr * dur_sec)).astype(np.float32)
        
        res = self.acoustic.verify_filled_pause_resonance(breath_noise, 0.0, dur_sec)
        self.assertFalse(res["is_filled_pause"])
        self.assertIn(res.get("reason"), ["high_zcr_unvoiced", "unvoiced_insufficient_pitch"])

    def test_shaped_unvoiced_breath_turbulence_rejected(self):
        """Shaped unvoiced exhalation (bandpassed white noise in 1-3 kHz) must fail voicing check."""
        dur_sec = 0.5
        np.random.seed(456)
        white = np.random.normal(0, 0.06, int(self.sr * dur_sec)).astype(np.float32)
        nyq = self.sr / 2.0
        b, a = butter(2, [1000.0 / nyq, 3000.0 / nyq], btype="band")
        shaped_breath = lfilter(b, a, white).astype(np.float32)

        res = self.acoustic.verify_filled_pause_resonance(shaped_breath, 0.0, dur_sec)
        self.assertFalse(res["is_filled_pause"])
        self.assertIn(res.get("reason"), ["high_zcr_unvoiced", "unvoiced_insufficient_pitch"])
        self.assertLess(res.get("voicing_ratio", 0.0), self.config.acoustic.min_filled_pause_voicing_ratio)

    def test_ordinary_speech_pause_rejected_by_detector(self):
        """An ordinary silent speech pause between words must produce 0 acoustic filler events."""
        # Create 3-second audio: word 1 (0.0 - 0.8s), silent pause (0.8 - 1.5s), word 2 (1.5 - 2.5s)
        total_samples = int(self.sr * 3.0)
        y = np.zeros(total_samples, dtype=np.float32)
        
        # Word 1: 0.0 to 0.8s (sine tones at speech frequencies, RMS ~0.05)
        t1 = np.linspace(0, 0.8, int(self.sr * 0.8), endpoint=False)
        y[:len(t1)] = 0.05 * np.sin(2 * np.pi * 200 * t1)
        
        # Silent pause with studio noise (0.8s to 1.5s, duration 0.70s): RMS ~0.002
        p_start = int(self.sr * 0.8)
        p_end = int(self.sr * 1.5)
        np.random.seed(789)
        y[p_start:p_end] = np.random.normal(0, 0.002, p_end - p_start).astype(np.float32)
        
        # Word 2: 1.5 to 2.5s
        t2 = np.linspace(0, 1.0, int(self.sr * 1.0), endpoint=False)
        w2_start = int(self.sr * 1.5)
        y[w2_start:w2_start + len(t2)] = 0.05 * np.sin(2 * np.pi * 250 * t2)

        words_list = [
            {"word": "hello", "start": 0.0, "end": 0.8},
            {"word": "world", "start": 1.5, "end": 2.5}
        ]

        events, details = self.filler_detector.detect_acoustic_fillers(y, self.sr, words_list)
        self.assertEqual(events, 0, f"Ordinary speech pause falsely detected as acoustic filler: {details}")
        self.assertEqual(len(details), 0)

    def test_genuine_voiced_schwa_hesitation_accepted(self):
        """A genuine voiced /ə/ vocalization (F0=130Hz, formants=500/1500Hz, RMS=0.04) must pass."""
        dur_sec = 0.5
        t = np.linspace(0, dur_sec, int(self.sr * dur_sec), endpoint=False)
        # Periodic harmonic phonation with vocal tract resonance in schwa band
        f0 = 0.025 * np.sin(2 * np.pi * 130 * t)
        f1 = 0.045 * np.sin(2 * np.pi * 500 * t)
        f2 = 0.030 * np.sin(2 * np.pi * 1500 * t)
        synthetic_uh = (f0 + f1 + f2).astype(np.float32)

        res = self.acoustic.verify_filled_pause_resonance(synthetic_uh, 0.0, dur_sec)
        self.assertTrue(res["is_filled_pause"], f"Genuine voiced hesitation rejected: {res}")
        self.assertEqual(res.get("reason"), "verified_vocalic_resonance")
        self.assertGreaterEqual(res.get("seg_rms", 0.0), self.config.acoustic.min_filled_pause_rms)
        self.assertGreaterEqual(res.get("voicing_ratio", 0.0), self.config.acoustic.min_filled_pause_voicing_ratio)
        self.assertGreaterEqual(res.get("schwa_energy_ratio", 0.0), self.config.acoustic.min_formant_schwa_ratio)

    def test_genuine_voiced_nasal_murmur_accepted(self):
        """A genuine voiced nasal murmur (/m/, F0=115Hz with low harmonic rumble < 0.40) must pass."""
        dur_sec = 0.5
        t = np.linspace(0, dur_sec, int(self.sr * dur_sec), endpoint=False)
        # Nasal /m/ has periodic F0 and low-frequency resonance (300-600 Hz)
        f0 = 0.020 * np.sin(2 * np.pi * 115 * t)
        f1 = 0.040 * np.sin(2 * np.pi * 350 * t)
        f2 = 0.025 * np.sin(2 * np.pi * 700 * t)
        synthetic_um = (f0 + f1 + f2).astype(np.float32)

        res = self.acoustic.verify_filled_pause_resonance(synthetic_um, 0.0, dur_sec)
        self.assertTrue(res["is_filled_pause"], f"Genuine nasal murmur rejected: {res}")
        self.assertEqual(res.get("reason"), "verified_vocalic_resonance")
        self.assertGreaterEqual(res.get("voicing_ratio", 0.0), self.config.acoustic.min_filled_pause_voicing_ratio)

    def test_lexical_filler_detection_preserved(self):
        """Lexical filler detection for audible words ('um', 'uh', 'er') must remain fully intact."""
        transcript = "Well, um, I was thinking that, uh, er, maybe we should start."
        res = self.filler_detector.evaluate(transcript)
        
        self.assertEqual(res["has_filler"], 1)
        self.assertGreaterEqual(res["lexical_filler_events"], 3)
        self.assertEqual(res["classification"], "CONFIDENT_FILLER")
        self.assertGreaterEqual(res["confidence_score"], 0.70)
        
        # Verify specific core lexical filler details are present
        filler_types = [d.get("type") for d in res["details"]]
        self.assertIn("core_vocal_filler", filler_types)
        filler_words = [d.get("text") for d in res["details"] if d.get("type") == "core_vocal_filler"]
        self.assertIn("um", filler_words)
        self.assertIn("uh", filler_words)
        self.assertIn("er", filler_words)

    def test_adaptive_vocal_threshold_scaling(self):
        """When vocal_threshold is high, segment RMS below 0.75 * vocal_threshold is rejected."""
        dur_sec = 0.5
        t = np.linspace(0, dur_sec, int(self.sr * dur_sec), endpoint=False)
        # Moderate amplitude sound (RMS ~0.015)
        sound = 0.015 * np.sqrt(2) * np.sin(2 * np.pi * 500 * t)
        
        # In quiet environment (vocal_threshold = 0.010): floor is max(0.008, 0.0075) = 0.008. Sound (0.015) passes energy check.
        res_quiet = self.acoustic.verify_filled_pause_resonance(sound, 0.0, dur_sec, vocal_threshold=0.010)
        self.assertNotEqual(res_quiet.get("reason"), "below_energy_floor")
        
        # In loud environment (vocal_threshold = 0.040): floor is max(0.008, 0.030) = 0.030. Sound (0.015) is rejected as below floor.
        res_loud = self.acoustic.verify_filled_pause_resonance(sound, 0.0, dur_sec, vocal_threshold=0.040)
        self.assertFalse(res_loud["is_filled_pause"])
        self.assertEqual(res_loud.get("reason"), "below_energy_floor")
        self.assertGreater(res_loud.get("energy_floor"), res_loud.get("seg_rms"))


if __name__ == "__main__":
    unittest.main()
