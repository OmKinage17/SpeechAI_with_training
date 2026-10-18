"""
Hybrid Stutter Detector Module for SpeechAI Experimental Pipeline
=================================================================
Combines:
1. Multi-level lexical repetition detection (syllables, words, partial-word fragments).
2. Acoustic sound prolongation detection (sustained pitch and spectral stability via librosa).
3. Acoustic block detection (pre-phonation tense silence and struggle gaps).
4. Calibrated continuous confidence scoring [0.0, 1.0] with explicit uncertainty classification.
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .config import StutterDetectorConfig
from .acoustic_features import AcousticFeatureExtractor

logger = logging.getLogger("SpeechAI_Hybrid_Stutter_Detector")

def clean_token(text: str) -> str:
    """Removes non-alphanumeric punctuation from a single token."""
    return re.sub(r'[^\w]', '', text).lower()

class HybridStutterDetector:
    def __init__(
        self, 
        config: Optional[StutterDetectorConfig] = None,
        acoustic_extractor: Optional[AcousticFeatureExtractor] = None
    ):
        self.config = config or StutterDetectorConfig()
        self.acoustic = acoustic_extractor or AcousticFeatureExtractor()

    def detect_lexical_repetitions(
        self, 
        transcript: str, 
        words_list: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Detects syllable-level, partial-word, and whole-word repetitions in the transcript.
        """
        if not transcript:
            return 0, []

        text_lower = transcript.lower()
        events = 0
        details = []

        # 1. Triplicate syllable prefix repetitions (e.g., "w-w-what", "c-c-cat")
        reps_3 = list(re.finditer(r'\b([a-zA-Z]{1,4})-\1-\1([a-zA-Z]*)\b', text_lower))
        for m in reps_3:
            events += 1
            details.append({
                "type": "sound_repetition_3x",
                "text": m.group(0),
                "matched_pattern": m.group(1),
                "confidence": 0.95
            })

        # Remove 3x repetitions before checking 2x repetitions
        cleaned_for_2x = re.sub(r'\b([a-zA-Z]{1,4})-\1-\1([a-zA-Z]*)\b', ' ', text_lower)
        reps_2 = list(re.finditer(r'\b([a-zA-Z]{1,4})-\1([a-zA-Z]*)\b', cleaned_for_2x))
        for m in reps_2:
            events += 1
            details.append({
                "type": "sound_repetition_2x",
                "text": m.group(0),
                "matched_pattern": m.group(1),
                "confidence": 0.85
            })

        # 2. Broken single-letter fragments (e.g., "b- because", "t- today")
        broken_frags = list(re.finditer(r'\b([a-zA-Z]{1,2})[\s\-]+([a-zA-Z]{3,})\b', text_lower))
        for m in broken_frags:
            prefix = m.group(1)
            target = m.group(2)
            if target.startswith(prefix) and len(prefix) < len(target):
                events += 1
                details.append({
                    "type": "broken_syllable_fragment",
                    "text": m.group(0),
                    "confidence": 0.80
                })

        # 3. Consecutive whole-word repetitions (e.g., "the the", "I I I")
        words = [clean_token(w) for w in text_lower.split() if clean_token(w)]
        for i in range(len(words) - 1):
            if words[i] == words[i+1]:
                events += 1
                details.append({
                    "type": "word_repetition",
                    "text": f"{words[i]} {words[i+1]}",
                    "confidence": 0.85
                })

        # Align with timestamps if words_list is available
        if words_list and len(words_list) > 1:
            for i in range(len(words_list) - 1):
                w1 = clean_token(words_list[i].get("word", ""))
                w2 = clean_token(words_list[i+1].get("word", ""))
                if w1 and w1 == w2:
                    s_time = round(float(words_list[i].get("start", 0.0)), 2)
                    e_time = round(float(words_list[i+1].get("end", 0.0)), 2)
                    # Update timestamp in details
                    for d in details:
                        if d["type"] == "word_repetition" and words[i] in d["text"] and "start" not in d:
                            d["start"] = s_time
                            d["end"] = e_time
                            d["duration"] = round(max(0.1, e_time - s_time), 2)
                            break

        return events, details

    def detect_acoustic_prolongations(
        self, 
        y: np.ndarray, 
        sr: int
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Detects acoustic prolongations (unnatural continuous vowel/voiced consonant elongation).
        """
        cands = self.acoustic.find_prolongation_candidates(
            y, 
            min_duration_sec=self.config.min_prolongation_duration_sec
        )
        events = len(cands)
        details = []
        for c in cands:
            details.append({
                "type": "acoustic_prolongation",
                "start": c["start"],
                "end": c["end"],
                "duration": c["duration"],
                "pitch_std_st": c["pitch_std_st"],
                "confidence": c["confidence"]
            })
        return events, details

    def detect_acoustic_blocks(
        self, 
        y: np.ndarray, 
        words_list: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Detects silent articulatory blocks (tense silence before word onsets).
        """
        if not words_list or len(words_list) < 1:
            return 0, []

        sr = self.acoustic.config.sample_rate
        rms, noise_floor, vocal_thresh = self.acoustic.compute_energy_and_noise_floor(y)
        frame_sec = self.acoustic.config.hop_length_ms / 1000.0

        events = 0
        details = []

        # Check pre-utterance block (silence before the very first word)
        first_word_start = float(words_list[0].get("start", 0.0))
        if first_word_start >= self.config.min_block_silence_sec:
            # Check if this initial gap is silence followed by a plosive burst
            first_frame = int(first_word_start / frame_sec)
            if first_frame < len(rms):
                burst_energy = float(np.max(rms[first_frame:min(len(rms), first_frame + 4)]))
                pre_energy = float(np.mean(rms[:first_frame])) if first_frame > 0 else 0.0
                if burst_energy >= self.config.plosive_burst_energy_ratio * (pre_energy + 1e-4) and burst_energy >= vocal_thresh:
                    events += 1
                    details.append({
                        "type": "initial_acoustic_block",
                        "start": 0.0,
                        "end": round(first_word_start, 2),
                        "duration": round(first_word_start, 2),
                        "confidence": 0.65
                    })

        # Check inter-word blocks
        for i in range(len(words_list) - 1):
            curr_end = float(words_list[i].get("end", 0.0))
            next_start = float(words_list[i+1].get("start", 0.0))
            gap = next_start - curr_end

            if gap >= self.config.min_block_silence_sec:
                # Analyze energy in this gap (excluding 30ms margin from word boundaries)
                margin_sec = 0.03
                s_frame = int((curr_end + margin_sec) / frame_sec)
                e_frame = int((next_start - margin_sec) / frame_sec)
                burst_start_frame = int(next_start / frame_sec)
                
                if e_frame > s_frame and burst_start_frame < len(rms):
                    gap_rms = float(np.mean(rms[s_frame:e_frame]))
                    burst_rms = float(np.max(rms[burst_start_frame:min(len(rms), burst_start_frame + 4)]))
                    
                    # True block is silence (near noise floor) followed by sudden onset burst
                    if gap_rms <= max(0.03, vocal_thresh * 1.5) and burst_rms >= 2.5 * (gap_rms + 1e-4) and burst_rms >= vocal_thresh:
                        events += 1
                        details.append({
                            "type": "inter_word_acoustic_block",
                            "start": round(curr_end, 2),
                            "end": round(next_start, 2),
                            "duration": round(gap, 2),
                            "confidence": 0.60
                        })

        return events, details

    def evaluate(
        self,
        transcript: str,
        words_list: Optional[List[Dict[str, Any]]] = None,
        audio_path: Optional[str] = None,
        audio_data: Optional[Tuple[np.ndarray, int]] = None
    ) -> Dict[str, Any]:
        """
        Full hybrid stutter detection evaluation combining lexical and acoustic modalities.
        """
        all_details = []
        lex_events, lex_details = self.detect_lexical_repetitions(transcript, words_list)
        all_details.extend(lex_details)

        ac_prol_events, ac_prol_details = 0, []
        ac_block_events, ac_block_details = 0, []

        # Load audio if provided
        y, sr = None, None
        if audio_data:
            y, sr = audio_data
        elif audio_path and os.path.exists(audio_path):
            try:
                y, sr = self.acoustic.load_audio(audio_path)
            except Exception as e:
                logger.warning(f"Could not load audio for stutter detection: {e}")

        if y is not None:
            ac_prol_events, ac_prol_details = self.detect_acoustic_prolongations(y, sr)
            all_details.extend(ac_prol_details)

            ac_block_events, ac_block_details = self.detect_acoustic_blocks(y, words_list)
            all_details.extend(ac_block_details)

        total_events = lex_events + ac_prol_events + ac_block_events

        # Compute continuous confidence score [0.0, 1.0]
        confidence = 0.0
        if lex_events > 0:
            confidence = max(confidence, 0.85 + min(0.15, lex_events * 0.05))
        if ac_prol_events > 0:
            confidence = max(confidence, 0.70 + min(0.20, ac_prol_events * 0.05))
        if ac_block_events > 0:
            confidence = max(confidence, 0.55 + min(0.15, ac_block_events * 0.05))

        # Categorize decision
        if confidence >= self.config.confident_threshold:
            classification = "CONFIDENT_DISFLUENT"
            has_stutter = 1
        elif confidence >= self.config.uncertain_threshold:
            classification = "UNCERTAIN_BORDERLINE"
            # In binary thresholding, borderline events with acoustic evidence count as candidate detections
            has_stutter = 1 if (lex_events > 0 or ac_prol_events > 0) else 0
        else:
            classification = "CONFIDENT_CLEAN"
            has_stutter = 0

        return {
            "has_stutter": has_stutter,
            "total_stutter_events": total_events,
            "lexical_repetition_events": lex_events,
            "acoustic_prolongation_events": ac_prol_events,
            "acoustic_block_events": ac_block_events,
            "confidence_score": round(confidence, 3),
            "classification": classification,
            "details": all_details
        }
