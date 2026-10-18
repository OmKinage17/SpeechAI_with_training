"""
Configuration and Hyperparameters for SpeechAI Experimental Pipeline
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple

@dataclass
class ASRConfig:
    model_name: str = "base"
    language: str = "en"
    # Bounded temperature decoding to prevent the 30-minute fallback loops
    # Uses greedy decoding first (0.0), with at most one fallback (0.2) if compression ratio fails
    temperature: Tuple[float, ...] = (0.0, 0.2)
    # Safe per-clip processing timeout in seconds
    per_clip_timeout_sec: float = 25.0
    initial_prompt: str = (
        "Um, uh, er, ah, hmm, like, you know, actually, basically, so, well."
    )
    condition_on_previous_text: bool = False
    word_timestamps: bool = True
    beam_size: int = 5
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

@dataclass
class AcousticConfig:
    sample_rate: int = 16000
    frame_length_ms: float = 32.0
    hop_length_ms: float = 10.0
    # Pitch extraction parameters (human vocal range)
    f0_min_hz: float = 75.0
    f0_max_hz: float = 400.0
    # Minimum silence duration to be considered a pause (seconds)
    min_pause_duration_sec: float = 0.25
    # Energy thresholds for voice activity
    noise_floor_percentile: float = 15.0
    vocal_energy_multiplier: float = 2.5
    # Formant resonance bands for central vowels / nasals (/ə/, /m/, /n/)
    f1_schwa_band_hz: Tuple[float, float] = (300.0, 750.0)
    f2_schwa_band_hz: Tuple[float, float] = (1000.0, 1800.0)
    min_formant_schwa_ratio: float = 0.28
    # Acoustic filled pause energy floor and harmonic voicing gating (Fix 1)
    min_filled_pause_rms: float = 0.008
    min_filled_pause_voicing_ratio: float = 0.30
    vocal_energy_floor_multiplier: float = 0.75
    autocorr_voicing_threshold: float = 0.35

@dataclass
class StutterDetectorConfig:
    # Decision confidence thresholds
    confident_threshold: float = 0.60
    uncertain_threshold: float = 0.35
    
    # Acoustic prolongation parameters
    # A single continuous phone or vowel sustained for > 350ms with stable pitch is candidate prolongation
    min_prolongation_duration_sec: float = 0.32
    max_pitch_variation_st: float = 1.8  # semitones variation for steady-state vowel
    min_spectral_stability: float = 0.70 # low spectral flux indicates held posture
    
    # Acoustic block parameters
    # Silent gap (> 400ms) with elevated pre-phonation muscle/airflow energy or sudden plosive onset
    min_block_silence_sec: float = 0.40
    plosive_burst_energy_ratio: float = 3.0

    # Lexical repetition parameters
    min_syllable_len: int = 1
    max_syllable_len: int = 4

@dataclass
class FillerDetectorConfig:
    # Unconditional vocal fillers (always counted as filled pause / interjection)
    core_vocal_fillers: Set[str] = field(default_factory=lambda: {
        "um", "umm", "ummm", "uh", "uhh", "uhhh", "er", "err", "erm", 
        "ah", "ahh", "hmm", "hmmm"
    })
    
    # Contextual discourse markers (only counted if surrounded by pauses or repeated)
    contextual_markers: Set[str] = field(default_factory=lambda: {
        "like", "actually", "basically", "literally", "honestly", "seriously", "so", "well"
    })
    
    # Multi-word phrase fillers
    phrase_fillers: List[str] = field(default_factory=lambda: [
        "you know", "i mean", "sort of", "kind of", "you see", "as in"
    ])
    
    # Context requirements for "so", "well", "like"
    # Word must be preceded OR followed by a pause of at least this duration
    contextual_pause_boundary_sec: float = 0.20
    
    # Acoustic filled pause parameters
    min_filled_pause_dur_sec: float = 0.25
    max_filled_pause_dur_sec: float = 2.0
    min_formant_schwa_ratio: float = 0.28  # ratio of energy in schwa band vs total energy

@dataclass
class ExperimentalConfig:
    asr: ASRConfig = field(default_factory=ASRConfig)
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    stutter: StutterDetectorConfig = field(default_factory=StutterDetectorConfig)
    filler: FillerDetectorConfig = field(default_factory=FillerDetectorConfig)
