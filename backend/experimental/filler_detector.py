"""
Context-Aware Filler Detector Module for SpeechAI Experimental Pipeline
========================================================================
Combines:
1. Unconditional core vocal fillers ("um", "uh", "er", "erm", "ah", "hmm").
2. Context-gated discourse markers ("so", "well", "like", "actually", "basically")
   using lexical grammatical patterns and acoustic pause boundaries.
3. Multi-word phrase fillers ("you know", "i mean", "sort of", "kind of").
4. Formant-verified acoustic filled pause validation in speech gaps (rejecting breath/noise).
5. Continuous confidence scoring [0.0, 1.0] and explicit uncertainty classification.
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np

from .config import FillerDetectorConfig
from .acoustic_features import AcousticFeatureExtractor

logger = logging.getLogger("SpeechAI_Context_Filler_Detector")

def clean_token(text: str) -> str:
    """Removes non-alphanumeric punctuation from a single token."""
    return re.sub(r'[^\w]', '', text).lower()

class ContextAwareFillerDetector:
    def __init__(
        self,
        config: Optional[FillerDetectorConfig] = None,
        acoustic_extractor: Optional[AcousticFeatureExtractor] = None
    ):
        self.config = config or FillerDetectorConfig()
        self.acoustic = acoustic_extractor or AcousticFeatureExtractor()

        # Regular expressions for common non-filler usages of "so", "well", "like"
        self._non_filler_so_patterns = [
            re.compile(r'\b(think|thought|hope|hoped|said|believe|suppose|guess)\s+so\b', re.IGNORECASE),
            re.compile(r'\bso\s+(much|many|good|bad|big|small|far|fast|slow|long|short|well|great|hard|easy|nice|cold|hot|warm|tired|happy|sad|late|early)\b', re.IGNORECASE),
            re.compile(r'\b(or|and|even)\s+so\b', re.IGNORECASE),
            re.compile(r'\bso\s+(that|as\s+to)\b', re.IGNORECASE),
            re.compile(r'\bnot\s+so\b', re.IGNORECASE),
            re.compile(r'\bhow\s+so\b', re.IGNORECASE),
        ]

        self._non_filler_well_patterns = [
            re.compile(r'\bas\s+well(\s+as)?\b', re.IGNORECASE),
            re.compile(r'\b(very|quite|pretty|fairly|how|so|doing|did|does|do|done|sleep|slept|perform|performed|know|known)\s+well\b', re.IGNORECASE),
            re.compile(r'\bwell\s+(done|known|being|defined|made|built|off|thought|adjusted)\b', re.IGNORECASE),
            re.compile(r'\b(get|getting|feeling|feels|feel)\s+well\b', re.IGNORECASE),
            re.compile(r'\b(oil|water|deep)\s+well\b', re.IGNORECASE),
        ]

        self._non_filler_like_patterns = [
            re.compile(r'\b(i|you|we|they|he|she|who|would|to|d)\s+like\b', re.IGNORECASE),
            re.compile(r'\blike\s+(to|it|that|this|him|her|them|me|us)\b', re.IGNORECASE),
            re.compile(r'\b(looks?|sounded?|sounds?|feels?|felt|smells?|tastes?|just|more|much|something|nothing|anything|acts?|acted|seem|seemed|seems)\s+like\b', re.IGNORECASE),
            re.compile(r'\bwhat(\s+\w+)?\s+like\b', re.IGNORECASE),
        ]

        self._non_filler_you_know_patterns = [
            re.compile(r'\b(do|did|does|don\'t|dont|didn\'t|didnt|will|would|can|could|should|might|may)\s+you\s+know\b', re.IGNORECASE),
            re.compile(r'\b(as|if|whether|that|lest)\s+you\s+know\b', re.IGNORECASE),
            re.compile(r'\b(someone|anyone|everyone|no\s+one|nobody|people|who|whom|which)\s+you\s+know\b', re.IGNORECASE),
            re.compile(r'\b(let|lets|letting)\s+you\s+know\b', re.IGNORECASE),
        ]

    def is_lexical_context_non_filler(self, word: str, surrounding_text: str) -> bool:
        """
        Determines whether an occurrence of 'so', 'well', or 'like' is standard grammatical usage.
        """
        w_lower = word.lower()
        if w_lower == "so":
            for pat in self._non_filler_so_patterns:
                if pat.search(surrounding_text):
                    return True
        elif w_lower == "well":
            for pat in self._non_filler_well_patterns:
                if pat.search(surrounding_text):
                    return True
        elif w_lower == "like":
            for pat in self._non_filler_like_patterns:
                if pat.search(surrounding_text):
                    return True
        return False

    def detect_lexical_fillers(
        self,
        transcript: str,
        words_list: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Detects lexical fillers in transcript using context gating.
        """
        if not transcript:
            return 0, []

        text_lower = transcript.lower()
        events = 0
        details = []

        # 1. Multi-word phrase fillers (e.g., "you know", "i mean", "sort of")
        working_text = text_lower
        for phrase in self.config.phrase_fillers:
            pat = rf'\b{re.escape(phrase)}\b'
            for m in re.finditer(pat, text_lower):
                # Check grammatical exclusions for "you know"
                if phrase == "you know":
                    prefix_snippet = text_lower[max(0, m.start() - 25):m.end()]
                    is_grammatical = any(p.search(prefix_snippet) for p in self._non_filler_you_know_patterns)
                    if is_grammatical:
                        continue

                events += 1
                details.append({
                    "type": "phrase_filler",
                    "text": phrase,
                    "confidence": 0.80
                })
            working_text = re.sub(pat, ' ', working_text)

        # 2. Tokenize remaining words
        cleaned_tokens = [clean_token(w) for w in working_text.split() if clean_token(w)]

        # Map words to timestamps if available
        aligned_words = []
        if words_list:
            for w_dict in words_list:
                cleaned = clean_token(w_dict.get("word", ""))
                if cleaned:
                    aligned_words.append({
                        "word": cleaned,
                        "start": float(w_dict.get("start", 0.0)),
                        "end": float(w_dict.get("end", 0.0))
                    })

        # Process each token
        for idx, w in enumerate(cleaned_tokens):
            # A. Core vocal fillers (unconditional)
            if (
                w in self.config.core_vocal_fillers or
                re.match(r'^u+m+h*$', w) or
                re.match(r'^u+h+m*$', w) or
                re.match(r'^e+r+m*$', w) or
                re.match(r'^a+h+$', w) or
                re.match(r'^h+m+$', w)
            ):
                events += 1
                details.append({
                    "type": "core_vocal_filler",
                    "text": w,
                    "confidence": 0.90
                })
                continue

            # B. Contextual discourse markers ("so", "well", "like", "actually", etc.)
            if w in self.config.contextual_markers:
                # Check consecutive repetition (e.g., "like like", "so so") -> Always filler
                is_repeated = (
                    (idx > 0 and cleaned_tokens[idx - 1] == w) or
                    (idx < len(cleaned_tokens) - 1 and cleaned_tokens[idx + 1] == w)
                )
                if is_repeated:
                    events += 1
                    details.append({
                        "type": "repeated_discourse_marker",
                        "text": f"{w} {w}",
                        "confidence": 0.85
                    })
                    continue

                # Context check with surrounding words
                window_start = max(0, idx - 2)
                window_end = min(len(cleaned_tokens), idx + 3)
                surrounding_snippet = " ".join(cleaned_tokens[window_start:window_end])

                if self.is_lexical_context_non_filler(w, surrounding_snippet):
                    # Grammatical usage -> reject
                    continue

                # Timestamp pause boundary check if aligned words exist
                has_pause_boundary = False
                if aligned_words and idx < len(aligned_words):
                    curr_word_info = aligned_words[idx]
                    if curr_word_info["word"] == w:
                        pre_gap = 0.0
                        post_gap = 0.0
                        if idx > 0:
                            pre_gap = curr_word_info["start"] - aligned_words[idx - 1]["end"]
                        if idx < len(aligned_words) - 1:
                            post_gap = aligned_words[idx + 1]["start"] - curr_word_info["end"]
                        
                        if (
                            pre_gap >= self.config.contextual_pause_boundary_sec or
                            post_gap >= self.config.contextual_pause_boundary_sec
                        ):
                            has_pause_boundary = True

                # If we have timestamps, require the pause boundary
                if aligned_words:
                    if has_pause_boundary:
                        events += 1
                        details.append({
                            "type": "contextual_discourse_filler",
                            "text": w,
                            "confidence": 0.65
                        })
                else:
                    # Without timestamps, only count words like "actually", "basically", "literally"
                    # For "so", "well", "like" without timestamps, only count if at start of utterance
                    if w in {"actually", "basically", "literally", "honestly", "seriously"}:
                        events += 1
                        details.append({
                            "type": "contextual_discourse_filler",
                            "text": w,
                            "confidence": 0.55
                        })
                    elif idx == 0 and len(cleaned_tokens) > 1 and cleaned_tokens[1] in self.config.core_vocal_fillers:
                        events += 1
                        details.append({
                            "type": "initial_discourse_filler",
                            "text": w,
                            "confidence": 0.70
                        })

        return events, details

    def detect_acoustic_fillers(
        self,
        y: np.ndarray,
        sr: int,
        words_list: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Scans inter-word gaps for vocalized filled pauses (/ə/, /m/) using formant verification.
        Rejects silent pauses, breathing noise, and microphone room tone.
        """
        if not words_list or len(words_list) < 2:
            return 0, []

        events = 0
        details = []

        # Compute clip-level energy profile to provide adaptive vocal threshold for pause gating (Fix 1)
        _, _, vocal_thresh = self.acoustic.compute_energy_and_noise_floor(y)

        for i in range(len(words_list) - 1):
            curr_end = float(words_list[i].get("end", 0.0))
            next_start = float(words_list[i + 1].get("start", 0.0))
            gap = next_start - curr_end

            # Analyze gaps between 0.25s and 2.0s
            if self.config.min_filled_pause_dur_sec <= gap <= self.config.max_filled_pause_dur_sec:
                # Add small 30ms margin from word boundaries
                margin = 0.03
                s_sec = curr_end + margin
                e_sec = next_start - margin

                if e_sec > s_sec:
                    res = self.acoustic.verify_filled_pause_resonance(
                        y, s_sec, e_sec, vocal_threshold=vocal_thresh
                    )
                    if res.get("is_filled_pause", False):
                        events += 1
                        details.append({
                            "type": "acoustic_vocal_hesitation",
                            "start": round(curr_end, 2),
                            "end": round(next_start, 2),
                            "duration": round(gap, 2),
                            "schwa_ratio": res.get("schwa_energy_ratio", 0.0),
                            "seg_rms": res.get("seg_rms", 0.0),
                            "voicing_ratio": res.get("voicing_ratio", 0.0),
                            "confidence": 0.70
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
        Full context-aware filler detection evaluation combining lexical and acoustic verification.
        """
        all_details = []
        lex_events, lex_details = self.detect_lexical_fillers(transcript, words_list)
        all_details.extend(lex_details)

        ac_events, ac_details = 0, []

        y, sr = None, None
        if audio_data:
            y, sr = audio_data
        elif audio_path and os.path.exists(audio_path):
            try:
                y, sr = self.acoustic.load_audio(audio_path)
            except Exception as e:
                logger.warning(f"Could not load audio for filler detection: {e}")

        if y is not None and words_list:
            ac_events, ac_details = self.detect_acoustic_fillers(y, sr, words_list)
            all_details.extend(ac_details)

        total_events = lex_events + ac_events

        # Confidence calculation
        confidence = 0.0
        if total_events > 0:
            conf_scores = [d.get("confidence", 0.5) for d in all_details]
            confidence = float(np.mean(conf_scores))
            # Boost if multiple independent filler events occur
            if total_events >= 2:
                confidence = min(1.0, confidence + 0.10)

        # Categorize
        if confidence >= 0.60 and total_events > 0:
            classification = "CONFIDENT_FILLER"
            has_filler = 1
        elif confidence >= 0.35 and total_events > 0:
            classification = "UNCERTAIN_BORDERLINE"
            has_filler = 1
        else:
            classification = "CONFIDENT_CLEAN"
            has_filler = 0

        return {
            "has_filler": has_filler,
            "total_filler_events": total_events,
            "lexical_filler_events": lex_events,
            "acoustic_filler_events": ac_events,
            "confidence_score": round(confidence, 3),
            "classification": classification,
            "details": all_details
        }
