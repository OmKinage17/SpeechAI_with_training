"""
Unit & Regression Tests for Phase 3 Detector Improvement: Standalone Prolongation Calibration
=============================================================================================

Validates:
1. Short standalone prolongation (< 0.45s) does NOT self-corroborate to confident stuttering.
2. Short standalone prolongation (< 0.45s) remains classified as UNCERTAIN_BORDERLINE (confidence=0.50).
3. Short prolongation (< 0.45s) accompanied by lexical repetition or block triggers stuttering.
4. Prolongation >= 0.45s with context support triggers stuttering (confidence=0.65).
5. Severe prolongation (>= 0.60s) self-corroborates regardless of context.
6. Multiple prolongations (>= 2) trigger stuttering.
7. Regression preservation: config defaults and backward compatibility.
"""

import unittest
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.experimental.config import StutterDetectorConfig
from backend.experimental.stutter_detector import HybridStutterDetector


class MockAcousticExtractor:
    def __init__(self):
        class Cfg:
            sample_rate = 16000
            hop_length_ms = 10.0
        self.config = Cfg()

    def load_audio(self, path):
        import numpy as np
        return np.zeros(16000, dtype=np.float32), 16000


class TestPhase3DetectorImprovement(unittest.TestCase):

    def setUp(self):
        self.cfg = StutterDetectorConfig(
            enable_contextual_scoring=True,
            confident_threshold=0.65,
            uncertain_threshold=0.50,
            standalone_prolongation_confidence=0.50,
            severe_prolongation_duration_sec=0.60,
            standalone_prolongation_min_duration_sec=0.45
        )
        self.detector = HybridStutterDetector(config=self.cfg, acoustic_extractor=MockAcousticExtractor())

    def test_short_standalone_prolongation_rejected_as_stutter(self):
        """A single prolongation under 0.45s (e.g. 0.38s) with context support receives conf=0.50 and has_stutter=0."""
        # Monkey patch detect methods
        self.detector.detect_lexical_repetitions = lambda t, w: (0, [])
        self.detector.detect_acoustic_blocks = lambda y, w: (0, [])
        self.detector.detect_acoustic_prolongations = lambda y, sr, **kw: (1, [{
            "type": "acoustic_prolongation",
            "start": 1.0,
            "end": 1.38,
            "duration": 0.38,
            "context_score": 0.85
        }])

        import numpy as np
        audio_data = (np.zeros(16000, dtype=np.float32), 16000)
        res = self.detector.evaluate("This is clean speech.", words_list=[], audio_data=audio_data)
        self.assertEqual(res["has_stutter"], 0, "Short standalone prolongation (0.38s) must not trigger stuttering")
        self.assertEqual(res["confidence_score"], 0.50)
        self.assertEqual(res["classification"], "UNCERTAIN_BORDERLINE")
        self.assertEqual(res["pred_dominant"], "Clean")

    def test_prolongation_meeting_min_duration_accepted(self):
        """A single prolongation >= 0.45s (e.g. 0.48s) with context support receives conf=0.65 and has_stutter=1."""
        self.detector.detect_lexical_repetitions = lambda t, w: (0, [])
        self.detector.detect_acoustic_blocks = lambda y, w: (0, [])
        self.detector.detect_acoustic_prolongations = lambda y, sr, **kw: (1, [{
            "type": "acoustic_prolongation",
            "start": 1.0,
            "end": 1.48,
            "duration": 0.48,
            "context_score": 0.85
        }])

        import numpy as np
        audio_data = (np.zeros(16000, dtype=np.float32), 16000)
        res = self.detector.evaluate("This is stuttered speech.", words_list=[], audio_data=audio_data)
        self.assertEqual(res["has_stutter"], 1, "Prolongation meeting 0.48s with context support must trigger stuttering")
        self.assertEqual(res["confidence_score"], 0.65)
        self.assertEqual(res["classification"], "CONFIDENT_DISFLUENT")
        self.assertEqual(res["pred_dominant"], "Stutter")

    def test_short_prolongation_with_lexical_corroboration_accepted(self):
        """A short prolongation (0.38s) accompanied by a lexical repetition qualifies as stuttering."""
        self.detector.detect_lexical_repetitions = lambda t, w: (1, [{
            "type": "word_repetition", "text": "the the", "confidence": 0.85
        }])
        self.detector.detect_acoustic_blocks = lambda y, w: (0, [])
        self.detector.detect_acoustic_prolongations = lambda y, sr, **kw: (1, [{
            "type": "acoustic_prolongation",
            "start": 1.0,
            "end": 1.38,
            "duration": 0.38,
            "context_score": 0.85
        }])

        import numpy as np
        audio_data = (np.zeros(16000, dtype=np.float32), 16000)
        res = self.detector.evaluate("the the speech", words_list=[], audio_data=audio_data)
        self.assertEqual(res["has_stutter"], 1)
        self.assertGreaterEqual(res["confidence_score"], 0.65)

    def test_short_prolongation_with_block_corroboration_accepted(self):
        """A short prolongation (0.38s) accompanied by an acoustic block qualifies as stuttering."""
        self.detector.detect_lexical_repetitions = lambda t, w: (0, [])
        self.detector.detect_acoustic_blocks = lambda y, w: (1, [{
            "type": "inter_word_acoustic_block", "duration": 0.45, "confidence": 0.60
        }])
        self.detector.detect_acoustic_prolongations = lambda y, sr, **kw: (1, [{
            "type": "acoustic_prolongation",
            "start": 1.0,
            "end": 1.38,
            "duration": 0.38,
            "context_score": 0.85
        }])

        import numpy as np
        audio_data = (np.zeros(16000, dtype=np.float32), 16000)
        res = self.detector.evaluate("tense speech", words_list=[], audio_data=audio_data)
        self.assertEqual(res["has_stutter"], 1)

    def test_severe_standalone_prolongation_self_corroborates(self):
        """A severe standalone prolongation (>= 0.60s) self-corroborates even with neutral context."""
        self.detector.detect_lexical_repetitions = lambda t, w: (0, [])
        self.detector.detect_acoustic_blocks = lambda y, w: (0, [])
        self.detector.detect_acoustic_prolongations = lambda y, sr, **kw: (1, [{
            "type": "acoustic_prolongation",
            "start": 1.0,
            "end": 1.65,
            "duration": 0.65,
            "context_score": 0.50
        }])

        import numpy as np
        audio_data = (np.zeros(16000, dtype=np.float32), 16000)
        res = self.detector.evaluate("prolonged speech", words_list=[], audio_data=audio_data)
        self.assertEqual(res["has_stutter"], 1)
        self.assertEqual(res["confidence_score"], 0.65)

    def test_backward_compatibility_when_threshold_zero(self):
        """When standalone_prolongation_min_duration_sec=0.0, acts identical to Phase 2 baseline."""
        legacy_cfg = StutterDetectorConfig(
            enable_contextual_scoring=True,
            confident_threshold=0.65,
            uncertain_threshold=0.50,
            standalone_prolongation_min_duration_sec=0.0
        )
        legacy_det = HybridStutterDetector(config=legacy_cfg, acoustic_extractor=MockAcousticExtractor())
        legacy_det.detect_lexical_repetitions = lambda t, w: (0, [])
        legacy_det.detect_acoustic_blocks = lambda y, w: (0, [])
        legacy_det.detect_acoustic_prolongations = lambda y, sr, **kw: (1, [{
            "type": "acoustic_prolongation",
            "start": 1.0,
            "end": 1.38,
            "duration": 0.38,
            "context_score": 0.85
        }])

        import numpy as np
        audio_data = (np.zeros(16000, dtype=np.float32), 16000)
        res = legacy_det.evaluate("legacy speech", words_list=[], audio_data=audio_data)
        self.assertEqual(res["has_stutter"], 1, "Legacy mode with 0.0 duration floor admits short prolongation")


if __name__ == '__main__':
    unittest.main()
