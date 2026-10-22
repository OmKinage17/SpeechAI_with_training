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
    per_clip_timeout_sec: float = 30.0
    initial_prompt: str = (
        "Um, uh, er, ah, hmm, like, you know, actually, basically, so, well."
    )
    condition_on_previous_text: bool = False
    word_timestamps: bool = True
    beam_size: int = 1
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
    
    # Acoustic prolongation parameters (Fix 2: Recalibrated thresholds)
    # A single continuous phone or vowel sustained for >= 500ms with rigid articulatory posture
    min_prolongation_duration_sec: float = 0.50  # Increased from 0.32s to 0.50s to avoid fluent vowel overlap
    max_pitch_variation_st: float = 1.2  # Tightened from 1.8/2.0 to 1.2 semitones variation for steady-state vowel
    max_spectral_flux: float = 0.20  # Tightened from 0.35 to 0.20 (low spectral flux indicates held posture)
    min_spectral_stability: float = 0.70 # retained for backwards compatibility
    standalone_prolongation_confidence: float = 0.50  # Reduced from 0.70 to 0.50 (standalone prolongation requires corroboration)
    severe_prolongation_duration_sec: float = 0.60  # Fix 2B: Standalone prolongation >= 0.60s self-corroborates
    standalone_prolongation_min_duration_sec: float = 0.45  # Phase 3: Standalone prolongation must be >= 0.45s to self-corroborate via context
    
    # Fix 3: Context-aware prolongation scoring
    enable_contextual_scoring: bool = False  # Preserves Fix 2B baseline when False

    # Acoustic block parameters
    # Silent gap (> 400ms) with elevated pre-phonation muscle/airflow energy or sudden plosive onset
    min_block_silence_sec: float = 0.40
    plosive_burst_energy_ratio: float = 3.0

    # Lexical repetition parameters
    min_syllable_len: int = 1
    max_syllable_len: int = 4

    # Ablation 1: Multi-word n-gram repetition detection (isolated flag)
    enable_ngram_repetition: bool = False
    ngram_sizes: Tuple[int, ...] = (2, 3)
    ngram_repetition_confidence: float = 0.85

    # Ablation 2: Filler-aware and boundary-aware gating (isolated flag)
    enable_filler_boundary_gating: bool = False
    filler_overlap_threshold: float = 0.35
    filler_word_coverage_threshold: float = 0.40
    boundary_prolongation_min_duration_sec: float = 0.65
    gate_filler_blocks: bool = True

    # Ablation 3: Targeted stutter false-negative recovery (isolated flag)
    enable_interrupted_repetition: bool = False
    interrupted_repetition_max_skip: int = 3
    interrupted_repetition_allow_stem: bool = True
    interrupted_repetition_allow_hesitations: bool = True
    interrupted_repetition_confidence: float = 0.85

    # Phase 4B: Multi-iteration filler struggle recognition (isolated experimental candidate)
    enable_multi_iteration_filler_stutter: bool = False
    min_filler_repetition_count: int = 3
    rapid_filler_isi_threshold_sec: float = 0.35

    # Phase 4C.3: ASR repetition-loop safeguard (isolated experimental candidate)
    enable_asr_loop_guard: bool = False
    asr_loop_min_zero_dur_tokens: int = 3
    asr_loop_min_identical_starts: int = 4
    asr_loop_min_isi_floor_sec: float = 0.015
    asr_loop_min_span_floor_sec: float = 0.05
    asr_loop_min_repetition_streak: int = 5
    asr_loop_total_zero_dur_threshold: int = 5



@dataclass
class SpeechContextConfig:
    """Configuration options for Fix 3 contextual speech analysis."""
    enabled: bool = True
    
    # Syllable rate estimation
    min_word_probability: float = 0.40       # Exclude low-confidence ASR hallucinations
    pause_threshold_sec: float = 0.25        # Pauses >= 250ms excluded from articulation time
    local_window_sec: float = 2.50           # Temporal window radius around candidate (±2.5s)
    min_words_for_local_rate: int = 2        # Need at least 2 words in window for local rate
    min_articulation_time_sec: float = 0.35   # Minimum voiced/speech time to compute a valid rate
    
    # Phrase-final boundary detection
    phrase_final_pause_sec: float = 0.28     # Post-word pause indicating phrase-final boundary
    phrase_final_edge_margin_sec: float = 0.35 # Distance from recording end to consider terminal
    terminal_punctuation: str = ".,?!;:…-"   # Punctuation marks indicating syntactic boundaries
    phrase_final_score_discount: float = 0.35 # Score discount applied to phrase-final candidates
    
    # Relative elongation scoring
    elongation_ratio_threshold: float = 1.80  # Candidate duration / local mean syllable duration
    min_context_score_for_admission: float = 0.65 # Context score threshold to admit short candidate
    short_candidate_min_sec: float = 0.32    # Short candidate lower bound
    short_candidate_max_sec: float = 0.49    # Short candidate upper bound
    
    # Tempo bounds (syllables/second)
    slow_speech_threshold: float = 2.50      # Below this, speakers speak slowly (vowels naturally long)
    fast_speech_threshold: float = 4.50      # Above this, speech is brisk (shorter vowels stand out)
    
    # Fix 4: Lexical and Syntactic Context Analysis
    enable_lexical_syntactic_context: bool = True
    clause_boundary_discount: float = 0.20        # Minor clause punctuation (, ;)
    sentence_terminal_discount: float = 0.35      # Major sentence terminal punctuation (. ? !)
    recording_edge_discount: float = 0.25         # Proximity to end of recording (< margin)
    interrupted_speech_boost: float = 0.10        # Hyphenated / truncated word (e.g., "she-")
    function_word_elongation_boost: float = 0.08  # Closed-class function words elongated >= 1.7x
    word_initial_boost: float = 0.08              # Candidate at word-initial onset
    word_final_extra_discount: float = 0.05       # Candidate at word-final coda at phrase boundary
    repetition_proximity_boost: float = 0.12      # Adjacent to repeated token or partial repetition
    max_boundary_attenuation_ratio: float = 2.30  # Elongation ratio where boundary penalty decays to minimum floor
    min_boundary_attenuation_floor: float = 0.15  # Attenuation decay floor for genuine prolongations at boundaries

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
    context: SpeechContextConfig = field(default_factory=SpeechContextConfig)
