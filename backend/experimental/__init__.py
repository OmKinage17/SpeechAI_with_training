"""
SpeechAI Experimental Pipeline Package
======================================
Modular, evidence-based speech disfluency analysis architecture.
Designed to address stutter recall, filler precision, and latency tail spikes
without modifying production baseline files.
"""

from .config import ExperimentalConfig
from .asr_engine import ExperimentalASREngine
from .acoustic_features import AcousticFeatureExtractor
from .stutter_detector import HybridStutterDetector
from .filler_detector import ContextAwareFillerDetector
from .eval_harness import DevelopmentEvaluationHarness

__all__ = [
    "ExperimentalConfig",
    "ExperimentalASREngine",
    "AcousticFeatureExtractor",
    "HybridStutterDetector",
    "ContextAwareFillerDetector",
    "DevelopmentEvaluationHarness"
]
