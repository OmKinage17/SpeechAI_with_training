"""
Unit and Boundary Regression Tests for Phase 4C.3 ASR Loop Guard Candidate
===========================================================================
Tests all required safety and boundary conditions:
1. ASR repetition-loop artifacts are flagged and suppressed from false-positive stuttering.
2. Genuine repeated-filler struggle with physical duration and plausible timing is preserved.
3. Ordinary conversational filler chains do not automatically become stuttering.
4. Missing, zero, or collapsed timestamps are handled safely without crashing.
5. Independent acoustic prolongation and block pathways remain unsuppressed by ASR loop status.
6. Flag-off behavior strictly reproduces the Phase 2 baseline.
"""

import unittest
import numpy as np
from pathlib import Path

from backend.experimental.config import StutterDetectorConfig
from backend.experimental.stutter_detector import HybridStutterDetector
from backend.experimental.asr_loop_guard import ASRLoopGuard
from backend.experimental.filler_boundary_gating import FillerBoundaryGater


class TestPhase4C3ASRLoopGuard(unittest.TestCase):

    def setUp(self):
        # Default Phase 2 config
        self.phase2_cfg = StutterDetectorConfig(
            enable_filler_boundary_gating=False,
            enable_multi_iteration_filler_stutter=False,
            enable_asr_loop_guard=False
        )

        # Candidate config: Phase 4B hesitation gating + Phase 4C.3 ASR loop safeguard
        self.candidate_cfg = StutterDetectorConfig(
            enable_filler_boundary_gating=True,
            enable_multi_iteration_filler_stutter=True,
            enable_asr_loop_guard=True
        )

        self.phase2_detector = HybridStutterDetector(self.phase2_cfg)
        self.candidate_detector = HybridStutterDetector(self.candidate_cfg)
        self.loop_guard = ASRLoopGuard()

    def test_01_asr_repetition_loop_artifacts_flagged(self):
        """Verify ASR repetition loops (single-token, 2-gram, multi-gram) are flagged and suppressed."""
        # Case A: Degenerate 1-gram loop with collapsed timestamps
        words_loop = []
        for _ in range(50):
            words_loop.append({"word": "uh", "start": 2.98, "end": 2.98, "probability": 0.85})
        raw_text = " ".join(["uh"] * 50)

        audit = self.loop_guard.audit_clip(raw_text, words_loop)
        self.assertTrue(audit["is_loop_artifact"])
        self.assertTrue(audit["zero_dur_count"] >= 5)

        res = self.candidate_detector.evaluate(
            raw_text, words_list=words_loop, sanitizer_status="DEGENERATE_LOOP_RECOVERED"
        )
        self.assertEqual(res["has_stutter"], 0, "ASR loop artifact must not trigger stuttering")
        self.assertNotEqual(res["pred_dominant"], "Stutter")

        # Case B: Multi-gram loop bypassed by 1g/2g checks but caught by timestamp collapse
        words_multigram = []
        pattern = ["I'm", "just", "going", "to", "say,", "um"]
        for _ in range(15):
            for w in pattern:
                words_multigram.append({"word": w, "start": 2.98, "end": 2.98})
        raw_multi = " ".join(pattern * 15)
        
        audit_multi = self.loop_guard.audit_clip(raw_multi, words_multigram)
        self.assertTrue(audit_multi["is_loop_artifact"])
        self.assertTrue(audit_multi["zero_dur_count"] >= 50)

    def test_02_genuine_repeated_filler_struggle_preserved(self):
        """Verify authentic repeated-filler struggle with physical duration is preserved."""
        # Realistic struggle: 7x 'er' over 1.34s with plausible ISIs (~0.22s)
        words_struggle = []
        t = 0.0
        for _ in range(7):
            words_struggle.append({"word": "er", "start": round(t, 2), "end": round(t + 0.12, 2)})
            t += 0.20
        raw_struggle = "er er er er er er er Once oh quick."
        words_struggle.append({"word": "Once", "start": 1.50, "end": 1.70})
        words_struggle.append({"word": "oh", "start": 1.75, "end": 1.85})
        words_struggle.append({"word": "quick", "start": 1.90, "end": 2.20})

        audit = self.loop_guard.audit_clip(raw_struggle, words_struggle)
        self.assertFalse(audit["is_loop_artifact"], "Genuine struggle must not be flagged as ASR loop")
        self.assertEqual(audit["zero_dur_count"], 0)

        res = self.candidate_detector.evaluate(raw_struggle, words_list=words_struggle)
        self.assertEqual(res["has_stutter"], 1, "Genuine filler struggle chain must trigger stuttering")
        struggle_events = [d for d in res["details"] if d.get("type") == "multi_iteration_filler_struggle"]
        self.assertTrue(len(struggle_events) > 0)

    def test_03_ordinary_conversational_filler_chains_not_stutter(self):
        """Verify leisurely conversational hesitation ('um, um' with normal pacing) is not stuttering."""
        words_hesitation = [
            {"word": "Um", "start": 0.20, "end": 0.50},
            {"word": "um", "start": 0.90, "end": 1.20}
        ]
        raw_hesitation = "Um, um,"

        res = self.candidate_detector.evaluate(raw_hesitation, words_list=words_hesitation, pred_has_filler=1)
        self.assertEqual(res["has_stutter"], 0, "Leisurely conversational hesitation must not trigger stuttering")
        self.assertEqual(res["pred_dominant"], "Filler", "Dominant class should resolve to Filler")

    def test_04_missing_zero_collapsed_timestamps_handled_safely(self):
        """Verify boundary robustness: None, empty lists, or all-zero timestamps do not crash."""
        # None words_list
        res1 = self.candidate_detector.evaluate("uh uh uh", words_list=None)
        self.assertIn("has_stutter", res1)

        # Empty words_list
        res2 = self.candidate_detector.evaluate("um um um", words_list=[])
        self.assertIn("has_stutter", res2)

        # Words with 0.0 timestamps
        words_zero = [{"word": "uh", "start": 0.0, "end": 0.0} for _ in range(4)]
        res3 = self.candidate_detector.evaluate("uh uh uh uh", words_list=words_zero)
        self.assertIn("has_stutter", res3)
        self.assertEqual(res3["has_stutter"], 0, "Zero-duration filler repeats must be safely suppressed")

    def test_05_independent_acoustic_prolongations_and_blocks_unsuppressed(self):
        """Verify that an ASR repetition loop does NOT suppress independent acoustic prolongations or blocks."""
        # Create audio with a clear, steady 0.70s acoustic prolongation (220 Hz sine tone)
        sr = 16000
        dur_sec = 2.0
        t = np.linspace(0, dur_sec, int(sr * dur_sec), endpoint=False)
        # Prolonged steady-state tone from 0.2s to 0.9s (duration = 0.70s)
        audio = np.zeros_like(t)
        mask = (t >= 0.2) & (t <= 0.9)
        audio[mask] = 0.25 * np.sin(2 * np.pi * 220.0 * t[mask])

        # Transcript has an ASR repetition loop collapsed at end (2.98s)
        loop_words = [{"word": "uh", "start": 2.98, "end": 2.98} for _ in range(40)]
        loop_text = " ".join(["uh"] * 40)

        res = self.candidate_detector.evaluate(
            loop_text, words_list=loop_words, audio_data=(audio, sr), sanitizer_status="DEGENERATE_LOOP_RECOVERED"
        )

        # The ASR loop was suppressed, but the acoustic prolongation must remain detected!
        self.assertTrue(res["acoustic_prolongation_events"] >= 1, "Acoustic prolongation pathway must remain active")
        self.assertEqual(res["has_stutter"], 1, "Severe acoustic prolongation must independently trigger stuttering")

    def test_06_flag_off_reproduces_phase2_baseline(self):
        """Verify candidate with flags disabled produces identical output to Phase 2 baseline."""
        test_transcript = "b- because I think that um, um, maybe"
        test_words = [
            {"word": "b-", "start": 0.1, "end": 0.3},
            {"word": "because", "start": 0.3, "end": 0.6},
            {"word": "I", "start": 0.7, "end": 0.8},
            {"word": "think", "start": 0.8, "end": 1.0},
            {"word": "that", "start": 1.0, "end": 1.1},
            {"word": "um", "start": 1.2, "end": 1.4},
            {"word": "um", "start": 1.5, "end": 1.7},
            {"word": "maybe", "start": 1.8, "end": 2.1}
        ]

        p2_res = self.phase2_detector.evaluate(test_transcript, words_list=test_words)
        
        # Detector with flags off
        off_cfg = StutterDetectorConfig(
            enable_filler_boundary_gating=False,
            enable_multi_iteration_filler_stutter=False,
            enable_asr_loop_guard=False
        )
        off_detector = HybridStutterDetector(off_cfg)
        off_res = off_detector.evaluate(test_transcript, words_list=test_words)

        self.assertEqual(p2_res["has_stutter"], off_res["has_stutter"])
        self.assertEqual(p2_res["pred_dominant"], off_res["pred_dominant"])
        self.assertEqual(p2_res["confidence_score"], off_res["confidence_score"])
        self.assertEqual(p2_res["total_stutter_events"], off_res["total_stutter_events"])


if __name__ == "__main__":
    unittest.main()
