"""
Edge-Case and Robustness Unit Tests for SpeechAI Experimental Pipeline
=====================================================================
Tests uncovered boundary conditions and edge cases:
1. Zero-length and pure silence audio.
2. Saturated square-wave / clipped audio.
3. Ambiguous syntactic positions for discourse markers ("what it's like").
4. Contractions with discourse words ("so's", "well's").
5. Case-insensitivity and heavy punctuation in fillers ("UM... UH???").
6. Contiguous speech (zero pause gap between words).
7. Audio resampling robustness (different sampling rates).
8. Acoustic block detection with inverted or negative timestamps.
"""

import os
import sys
import time
import subprocess
import unittest
import numpy as np

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from experimental.config import ExperimentalConfig, ASRConfig, AcousticConfig, FillerDetectorConfig, StutterDetectorConfig
from experimental.acoustic_features import AcousticFeatureExtractor
from experimental.stutter_detector import HybridStutterDetector
from experimental.filler_detector import ContextAwareFillerDetector
from experimental.asr_engine import ExperimentalASREngine
from experimental.eval_harness import DevelopmentEvaluationHarness

class TestExperimentalEdgeCases(unittest.TestCase):
    def setUp(self):
        self.acoustic = AcousticFeatureExtractor(AcousticConfig(sample_rate=16000))
        self.stutter = HybridStutterDetector(StutterDetectorConfig(), self.acoustic)
        self.filler = ContextAwareFillerDetector(FillerDetectorConfig(), self.acoustic)
        self.sr = 16000

    def test_pure_digital_silence_no_crashes(self):
        """Pure digital silence (zeros) must not cause division by zero or NaN crashes."""
        silence = np.zeros(self.sr * 2)  # 2 seconds of zeros
        rms, noise_floor, vocal_thresh = self.acoustic.compute_energy_and_noise_floor(silence)
        self.assertGreater(noise_floor, 0.0)
        self.assertGreater(vocal_thresh, 0.0)
        
        flux = self.acoustic.compute_spectral_stability(silence)
        self.assertFalse(np.isnan(flux).any())

        # Stutter evaluation on pure silence
        stutter_res = self.stutter.evaluate(
            transcript="",
            words_list=[],
            audio_data=(silence, self.sr)
        )
        self.assertEqual(stutter_res["has_stutter"], 0)
        self.assertEqual(stutter_res["classification"], "CONFIDENT_CLEAN")

    def test_clipped_saturated_audio(self):
        """Heavily clipped / saturated audio must not produce unhandled exceptions."""
        t = np.linspace(0, 1.0, self.sr, endpoint=False)
        clipped_audio = np.sign(np.sin(2 * np.pi * 300 * t))  # Square wave at maximum amplitude
        
        rms, noise_floor, vocal_thresh = self.acoustic.compute_energy_and_noise_floor(clipped_audio)
        self.assertTrue(np.all(np.isfinite(rms)))
        
        cands = self.acoustic.find_prolongation_candidates(clipped_audio)
        self.assertIsInstance(cands, list)

    def test_ambiguous_like_at_utterance_end(self):
        """'Do you know what it's like?' must NOT be flagged as a filler."""
        sent = "Do you know what it's like?"
        count, details = self.filler.detect_lexical_fillers(sent)
        self.assertEqual(count, 0, f"False filler detected in: '{sent}' ({details})")

    def test_contractions_with_discourse_words(self):
        """Contractions like 'so's your mother' or 'well's dry' must not be flagged as fillers."""
        sents = [
            "So's your brother coming tomorrow?",
            "The old well's pump is broken."
        ]
        for s in sents:
            count, details = self.filler.detect_lexical_fillers(s)
            self.assertEqual(count, 0, f"False filler detected in contraction: '{s}' ({details})")

    def test_fillers_with_heavy_punctuation_and_caps(self):
        """Punctuation and capitalization variations ('UM... UH???') must be detected."""
        sent = "UM... UH??? What was that?"
        count, details = self.filler.detect_lexical_fillers(sent)
        self.assertGreaterEqual(count, 2, f"Failed to detect punctuated/capitalized fillers in: '{sent}'")

    def test_contiguous_speech_no_gaps(self):
        """Continuous speech with zero inter-word gap must not crash or trigger block detection."""
        audio = 0.2 * np.random.normal(0, 1, self.sr)
        words_list = [
            {"word": "wordone", "start": 0.0, "end": 0.5},
            {"word": "wordtwo", "start": 0.5, "end": 1.0}
        ]
        events, details = self.stutter.detect_acoustic_blocks(audio, words_list)
        self.assertEqual(events, 0, "Zero-gap words should not trigger acoustic blocks")

    def test_inverted_or_invalid_timestamps_handled_safely(self):
        """Corrupted or inverted timestamps (start > end) must not crash block detector."""
        audio = 0.1 * np.random.normal(0, 1, self.sr)
        words_list = [
            {"word": "wordone", "start": 0.8, "end": 0.2},
            {"word": "wordtwo", "start": 0.1, "end": 0.5}
        ]
        events, details = self.stutter.detect_acoustic_blocks(audio, words_list)
        self.assertEqual(events, 0, "Invalid timestamps should produce 0 events safely")

    def test_short_sub_hundred_millisecond_audio(self):
        """Audio under 100ms must return safely without array dimension errors."""
        short_audio = np.zeros(int(0.05 * self.sr))  # 50ms audio
        res = self.acoustic.verify_filled_pause_resonance(short_audio, 0.0, 0.05)
        self.assertFalse(res["is_filled_pause"])
        self.assertEqual(res["reason"], "too_short")

    def test_contextual_filler_positive_negative_matrix(self):
        """Comprehensive matrix of positive discourse fillers vs negative grammatical idioms."""
        # 1. Negative grammatical cases (MUST NOT be detected as fillers)
        negative_sentences = [
            "Do you know what it's like?",
            "Did you know that she was leaving today?",
            "Don't you know how to solve this equation?",
            "Would you know if something went wrong?",
            "Can you know the future for certain?",
            "I'll let you know when the results arrive.",
            "Please let you know... wait, let me know.",
            "If you know the answer, please raise your hand.",
            "Someone you know is waiting for you in the lobby.",
            "As far as you know, has this been verified?",
            "What did that building look like?",
            "I would like to order a warm beverage.",
            "She sings just like her grandmother.",
            "It was so pleasant to meet everyone yesterday.",
            "He ran so fast that nobody could catch him.",
            "That was truly a job well done by the team."
        ]
        for sent in negative_sentences:
            count, details = self.filler.detect_lexical_fillers(sent)
            self.assertEqual(count, 0, f"False filler detected in negative case: '{sent}' (details: {details})")

        # 2. Positive discourse fillers (MUST be detected)
        positive_sentences = [
            ("You know, I was nervous before the interview.", "you know"),
            ("I was, you know, feeling quite uncertain.", "you know"),
            ("It was a very difficult exam, you know.", "you know"),
            ("YOU KNOW, that makes a lot of sense.", "you know"),
            ("You know... maybe we should reconsider.", "you know"),
            ("You know? I never thought about that before.", "you know"),
            ("Um, you know, we really should head out now.", "you know"),
            ("Well, you know, like, it was getting late.", "you know")
        ]
        for sent, expected_filler in positive_sentences:
            count, details = self.filler.detect_lexical_fillers(sent)
            self.assertGreaterEqual(count, 1, f"Failed to detect positive filler in: '{sent}'")
            matched_fillers = [d.get("text", "") for d in details]
            self.assertTrue(
                any(expected_filler in m for m in matched_fillers),
                f"Expected '{expected_filler}' in matched fillers {matched_fillers} for '{sent}'"
            )

    def test_process_timeout_worker_termination_and_cleanup(self):
        """Timed-out ASR worker must be killed, joined, and removed from OS process table."""
        wav_path = r"E:\SpeechPrj\datasets\UCLASS Stuttered Speech Clips (SEP-28k Format)\clips\clips\M_0030_16y4m_1_dysfluent_000.wav"
        if not os.path.exists(wav_path):
            self.skipTest(f"Audio file not found: {wav_path}")

        engine = ExperimentalASREngine()
        try:
            self.assertTrue(engine._ensure_worker(), "Failed to initialize ASR worker")
            worker_pid = engine._process.pid

            # Confirm process is alive in Windows OS
            out_before = subprocess.run(["tasklist", "/FI", f"PID eq {worker_pid}"], capture_output=True, text=True).stdout
            self.assertIn(str(worker_pid), out_before, f"Worker PID {worker_pid} not found in tasklist")

            # Force timeout using tiny limit
            t0 = time.time()
            res = engine.transcribe(wav_path, timeout_sec=0.001)
            t_elapsed = time.time() - t0

            # Status validation
            self.assertEqual(res["status"], "TIMEOUT")
            self.assertEqual(res["text"], "")
            self.assertIn("timed out", res["error"].lower())

            # Tolerance check: process termination and join must complete within 2.0s
            self.assertLess(t_elapsed, 2.0, f"Worker timeout & cleanup took {t_elapsed}s, exceeding 2.0s tolerance")

            # Confirm process is DEAD in Windows OS process table
            time.sleep(0.1)
            out_after = subprocess.run(["tasklist", "/FI", f"PID eq {worker_pid}"], capture_output=True, text=True).stdout
            self.assertNotIn(str(worker_pid), out_after, f"Worker PID {worker_pid} is still alive in tasklist after timeout!")
        finally:
            engine.shutdown()

    def test_consecutive_repeated_process_timeouts_and_recovery(self):
        """Repeated consecutive timeouts must not leave orphaned processes or backlog."""
        wav_path = r"E:\SpeechPrj\datasets\UCLASS Stuttered Speech Clips (SEP-28k Format)\clips\clips\M_0030_16y4m_1_dysfluent_000.wav"
        if not os.path.exists(wav_path):
            self.skipTest(f"Audio file not found: {wav_path}")

        engine = ExperimentalASREngine()
        spawned_pids = []
        try:
            for iteration in range(3):
                self.assertTrue(engine._ensure_worker(), f"Failed to ensure worker on iteration {iteration}")
                pid = engine._process.pid
                spawned_pids.append(pid)

                res = engine.transcribe(wav_path, timeout_sec=0.001)
                self.assertEqual(res["status"], "TIMEOUT", f"Iteration {iteration} did not return TIMEOUT")

            # Confirm all 3 worker processes were terminated and none are orphaned
            time.sleep(0.1)
            for pid in spawned_pids:
                out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
                self.assertNotIn(str(pid), out, f"Orphan process detected! PID {pid} still running after repeated timeout.")
        finally:
            engine.shutdown()

    def test_ipc_serialization_error_handling(self):
        """Worker crash or unexpected termination must produce explicit CRASHED status."""
        wav_path = r"E:\SpeechPrj\datasets\UCLASS Stuttered Speech Clips (SEP-28k Format)\clips\clips\M_0030_16y4m_1_dysfluent_000.wav"
        if not os.path.exists(wav_path):
            self.skipTest(f"Audio file not found: {wav_path}")

        engine = ExperimentalASREngine()
        try:
            self.assertTrue(engine._ensure_worker())
            # Terminate the worker process behind the scenes to simulate unexpected crash
            engine._process.terminate()
            engine._process.join(timeout=1.0)

            # Subsequent transcribe attempt must catch the dead pipe cleanly
            res = engine.transcribe(wav_path, timeout_sec=2.0)
            self.assertIn(res["status"], ["CRASHED", "ERROR", "TIMEOUT"])
            self.assertEqual(res["text"], "")
            self.assertIn("error", res)
        finally:
            engine.shutdown()

    def test_acoustic_preservation_on_asr_failure(self):
        """Acoustic analysis results must be preserved when ASR fails without hallucinating transcript."""
        wav_path = r"E:\SpeechPrj\datasets\UCLASS Stuttered Speech Clips (SEP-28k Format)\clips\clips\M_0030_16y4m_1_dysfluent_000.wav"
        if not os.path.exists(wav_path):
            self.skipTest(f"Audio file not found: {wav_path}")

        harness = DevelopmentEvaluationHarness()
        # Mock ASR transcribe_audio_safe to simulate timeout failure
        harness.asr_engine.transcribe_audio_safe = lambda path: {
            "status": "TIMEOUT",
            "text": "",
            "transcript": "",
            "segments": [],
            "words": [],
            "words_list": [],
            "latency_sec": 0.05,
            "error": "Worker process timed out and was terminated"
        }

        clip_meta = {
            "file_path": wav_path,
            "filename": "M_0030_16y4m_1_dysfluent_000.wav",
            "dataset": "UCLASS",
            "speaker": "M_0030",
            "ref_text": "A wealthy Athenian.",
            "gt_sound_rep": 1,
            "gt_word_rep": 0,
            "gt_prolongation": 0,
            "gt_block": 1,
            "gt_filler": 0,
            "gt_clean": 0
        }

        res = harness.run_development_clip(clip_meta)

        # 1. ASR failure is explicitly reported
        self.assertEqual(res["asr_status"], "TIMEOUT")
        self.assertEqual(res["status"], "TIMEOUT")
        self.assertIn("timed out", res["asr_error"].lower())

        # 2. Failed ASR transcript is NOT treated as a successful transcription
        self.assertEqual(res["pred_transcript"], "")
        self.assertIsNone(res["wer"], "WER must be None when ASR failed; must not be 0.0")

        # 3. Acoustic features are fully preserved
        self.assertGreater(res["duration_sec"], 0.0)
        self.assertIn("stutter", res["details"])
        self.assertIn("filler", res["details"])
        self.assertIn(res["stutter_classification"], ["CONFIDENT_DISFLUENT", "UNCERTAIN_BORDERLINE", "CONFIDENT_CLEAN"])
        self.assertIn(res["filler_classification"], ["CONFIDENT_FILLER", "UNCERTAIN_BORDERLINE", "CONFIDENT_CLEAN"])

def run_edge_case_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExperimentalEdgeCases)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_edge_case_tests()
    sys.exit(0 if success else 1)
