"""
Speech Context and Syllable-Rate Analyzer Module for SpeechAI (Fix 3)
=====================================================================
Provides context-aware temporal analysis to distinguish genuine articulatory
prolongations from ordinary vowel lengthening (e.g., phrase-final lengthening,
slow conversational tempo, stressed syllables).

Components:
1. SyllableCounter: Lightweight phonological syllabification for English words.
2. SyllableRateEstimator: Estimates local and global articulation rates (syllables/sec)
   excluding non-speech pauses and low-confidence ASR hallucinations.
3. SpeechContextAnalyzer: Assesses candidate prolongation elongation relative to local
   syllabic tempo, detects phrase-final boundary lengthening, and computes a
   continuous, diagnostic contextual evidence score.
"""

import os
import re
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np

logger = logging.getLogger("SpeechAI_SpeechContext")

FUNCTION_WORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "so", "for", "in", "on", "at", 
    "to", "from", "by", "with", "about", "into", "of", "off", "up", "down",
    "he", "she", "it", "they", "we", "you", "i", "me", "him", "her", "us", "them",
    "my", "his", "their", "our", "your", "its", "mine", "yours", "hers", "ours", "theirs",
    "who", "whom", "whose", "which", "what",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might", "must"
}

def count_syllables_english(word: str) -> int:
    """
    Estimates the syllable count of an English word using deterministic
    phonological vowel-cluster rules, diphthong merging, and common morphological
    suffix adjustments (silent 'e', '-ed', '-es').
    
    Returns at least 1 syllable for any non-empty alphabetical token.
    """
    w = re.sub(r'[^a-z]', '', word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1

    # Specific common contractions / monosyllables
    monosyllables = {
        "the", "and", "that", "have", "for", "not", "with", "you", "this",
        "but", "his", "from", "they", "say", "her", "she", "will", "one",
        "all", "would", "there", "their", "what", "out", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your", "good",
        "some", "could", "them", "see", "other", "than", "then", "now", "look"
    }
    if w in monosyllables:
        # Standard lookup check
        if w in ("people", "into", "other", "about"):
            return 2
        return 1

    # Remove non-syllabic trailing 'e' (e.g., 'make', 'wrote', 'time')
    # Keep 'le' (e.g., 'little', 'middle') and 'ee' (e.g., 'free')
    work = w
    if work.endswith('e') and not work.endswith('le') and not work.endswith('ee'):
        work = work[:-1]

    # Handle past tense -ed (e.g., 'walked' -> 1, but 'wanted' -> 2)
    if work.endswith('ed') and len(work) > 3:
        if not (work.endswith('ted') or work.endswith('ded')):
            work = work[:-2]

    # Handle plural -es (e.g., 'passes' -> 2, but 'babies' -> 2 handled by y)
    if work.endswith('es') and len(work) > 3:
        if not (work.endswith('ches') or work.endswith('shes') or work.endswith('sses') or work.endswith('zes') or work.endswith('xes')):
            work = work[:-1]

    # Count vowel groups [aeiouy]+, accounting for hiatus vowel pairs
    vowels = "aeiouy"
    count = 0
    prev_char = ""
    for idx, char in enumerate(work):
        is_vowel = char in vowels
        prev_is_vowel = bool(prev_char) and (prev_char in vowels)
        if is_vowel:
            if not prev_is_vowel:
                count += 1
            else:
                # Hiatus pairs that split across syllables (e.g., 'ia', 'io', 'eo', 'ua', 'uo', 'ui', 'ue' in non-monosyllables)
                pair = prev_char + char
                is_tion = (pair == "io" and (idx >= 2 and work[idx-2] in ('t', 's', 'c')))
                if (pair in ("ia", "eo", "ua", "uo", "ui") or (pair == "io" and not is_tion) or (pair == "ue" and not w.endswith("ue") and len(w) > 4)):
                    count += 1
        prev_char = char

    # Common prefixes that add a syllable
    if w.startswith(('re', 'pre', 'de')) and len(w) > 4 and w[2] in vowels:
        count += 1

    return max(1, count)


from .config import SpeechContextConfig


class SpeechContextAnalyzer:
    """
    Analyzes local speech tempo, relative vowel elongation, and syntactic/phrase-final
    boundaries to provide evidence for or against genuine articulatory prolongations.
    """
    def __init__(self, config: Optional[SpeechContextConfig] = None):
        self.config = config or SpeechContextConfig()

    def estimate_syllable_rate(
        self,
        words_list: Optional[List[Dict[str, Any]]] = None,
        transcript: Optional[str] = None,
        duration_sec: float = 3.0,
        energy_rms: Optional[np.ndarray] = None,
        vocal_threshold: Optional[float] = None,
        hop_length_ms: float = 10.0
    ) -> Dict[str, Any]:
        """
        Estimates global articulation rate (syllables per second of active speech)
        accounting for pauses, transcript confidence, and energy.
        
        Returns:
            {
                "global_speech_rate": float or None,
                "mean_syllable_duration_sec": float or None,
                "total_syllables": int,
                "articulation_time_sec": float,
                "confidence": "HIGH" | "MEDIUM" | "ACOUSTIC_ESTIMATE" | "UNAVAILABLE"
            }
        """
        if not self.config.enabled:
            return {
                "global_speech_rate": None,
                "mean_syllable_duration_sec": None,
                "total_syllables": 0,
                "articulation_time_sec": 0.0,
                "confidence": "UNAVAILABLE"
            }

        # 1. Primary Method: Align word timestamps and syllable counts
        valid_words = []
        if words_list:
            for w in words_list:
                prob = float(w.get("probability", 1.0))
                word_txt = w.get("word", "").strip()
                s = float(w.get("start", 0.0))
                e = float(w.get("end", 0.0))
                if prob >= self.config.min_word_probability and word_txt and e > s:
                    sylls = count_syllables_english(word_txt)
                    valid_words.append({
                        "word": word_txt,
                        "start": s,
                        "end": e,
                        "duration": e - s,
                        "syllables": sylls,
                        "probability": prob
                    })

        if len(valid_words) >= 2:
            total_sylls = sum(w["syllables"] for w in valid_words)
            # Compute net articulation time by summing word durations, merging overlapping words
            valid_words.sort(key=lambda x: x["start"])
            net_time = 0.0
            cur_start = valid_words[0]["start"]
            cur_end = valid_words[0]["end"]
            
            for w in valid_words[1:]:
                if w["start"] <= cur_end:
                    cur_end = max(cur_end, w["end"])
                else:
                    net_time += max(0.05, cur_end - cur_start)
                    cur_start = w["start"]
                    cur_end = w["end"]
            net_time += max(0.05, cur_end - cur_start)

            if net_time >= self.config.min_articulation_time_sec and total_sylls > 0:
                rate = round(total_sylls / net_time, 2)
                mean_dur = round(net_time / total_sylls, 3)
                conf = "HIGH" if len(valid_words) >= 4 else "MEDIUM"
                return {
                    "global_speech_rate": rate,
                    "articulation_rate_sps": rate,
                    "mean_syllable_duration_sec": mean_dur,
                    "total_syllables": total_sylls,
                    "syllables": total_sylls,
                    "articulation_time_sec": round(net_time, 2),
                    "net_articulation_sec": round(net_time, 2),
                    "confidence": conf
                }

        # 2. Secondary Method: Acoustic Voiced Energy Nuclei Fallback
        # If words_list is missing but audio energy is provided
        if energy_rms is not None and len(energy_rms) > 0 and vocal_threshold is not None:
            frame_sec = hop_length_ms / 1000.0
            voiced_frames = np.where(energy_rms >= vocal_threshold * 0.5)[0]
            voiced_time = len(voiced_frames) * frame_sec

            if transcript and voiced_time >= self.config.min_articulation_time_sec:
                # Use transcript syllable count distributed across voiced time
                words = [w for w in transcript.split() if w]
                total_sylls = sum(count_syllables_english(w) for w in words)
                if total_sylls >= 2:
                    rate = round(total_sylls / voiced_time, 2)
                    mean_dur = round(voiced_time / total_sylls, 3)
                    return {
                        "global_speech_rate": rate,
                        "articulation_rate_sps": rate,
                        "mean_syllable_duration_sec": mean_dur,
                        "total_syllables": total_sylls,
                        "syllables": total_sylls,
                        "articulation_time_sec": round(voiced_time, 2),
                        "net_articulation_sec": round(voiced_time, 2),
                        "confidence": "ACOUSTIC_ESTIMATE"
                    }

        # 3. Explicit low-confidence / unknown state
        return {
            "global_speech_rate": None,
            "articulation_rate_sps": None,
            "mean_syllable_duration_sec": None,
            "total_syllables": 0,
            "syllables": 0,
            "articulation_time_sec": 0.0,
            "net_articulation_sec": 0.0,
            "confidence": "UNAVAILABLE"
        }

    def estimate_local_speech_rate(
        self,
        center_sec: float,
        words_list: Optional[List[Dict[str, Any]]] = None,
        global_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Estimates the local speech rate in a temporal window [center - W, center + W]
        around a candidate prolongation. Falls back gracefully to global rate if local
        words are sparse.
        """
        if not self.config.enabled or not words_list:
            if global_context and global_context.get("global_speech_rate") is not None:
                return {
                    "local_speech_rate": global_context["global_speech_rate"],
                    "mean_syllable_duration_sec": global_context["mean_syllable_duration_sec"],
                    "confidence": "FALLBACK_GLOBAL",
                    "n_local_words": 0
                }
            return {
                "local_speech_rate": None,
                "mean_syllable_duration_sec": None,
                "confidence": "UNAVAILABLE",
                "n_local_words": 0
            }

        w_radius = self.config.local_window_sec
        win_start = max(0.0, center_sec - w_radius)
        win_end = center_sec + w_radius

        local_words = []
        for w in words_list:
            s = float(w.get("start", 0.0))
            e = float(w.get("end", 0.0))
            prob = float(w.get("probability", 1.0))
            word_txt = w.get("word", "").strip()
            
            # Words overlapping the window
            if s < win_end and e > win_start and prob >= self.config.min_word_probability and word_txt:
                local_words.append({
                    "word": word_txt,
                    "start": s,
                    "end": e,
                    "syllables": count_syllables_english(word_txt)
                })

        if len(local_words) >= self.config.min_words_for_local_rate:
            total_sylls = sum(w["syllables"] for w in local_words)
            # Net local articulation time
            local_words.sort(key=lambda x: x["start"])
            net_time = 0.0
            cur_s = max(win_start, local_words[0]["start"])
            cur_e = min(win_end, local_words[0]["end"])
            
            for w in local_words[1:]:
                ws = max(win_start, w["start"])
                we = min(win_end, w["end"])
                if ws <= cur_e:
                    cur_e = max(cur_e, we)
                else:
                    net_time += max(0.04, cur_e - cur_s)
                    cur_s = ws
                    cur_e = we
            net_time += max(0.04, cur_e - cur_s)

            if net_time >= 0.25 and total_sylls > 0:
                rate = round(total_sylls / net_time, 2)
                mean_dur = round(net_time / total_sylls, 3)
                return {
                    "local_speech_rate": rate,
                    "mean_syllable_duration_sec": mean_dur,
                    "confidence": "HIGH" if len(local_words) >= 4 else "MEDIUM",
                    "n_local_words": len(local_words)
                }

        # Fallback to global rate if local words are insufficient
        if global_context and global_context.get("global_speech_rate") is not None:
            return {
                "local_speech_rate": global_context["global_speech_rate"],
                "mean_syllable_duration_sec": global_context["mean_syllable_duration_sec"],
                "confidence": "FALLBACK_GLOBAL",
                "n_local_words": len(local_words)
            }

        return {
            "local_speech_rate": None,
            "mean_syllable_duration_sec": None,
            "confidence": "UNAVAILABLE",
            "n_local_words": len(local_words)
        }

    def classify_lexical_context(
        self,
        word_obj: Optional[Dict[str, Any]],
        words_list: Optional[List[Dict[str, Any]]] = None,
        word_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extracts lexical evidence for a candidate word from ASR token alignment.
        Differentiates closed-class function words from open-class content words,
        detects truncated/interrupted tokens, and identifies proximity to repeated words.
        """
        if not word_obj or float(word_obj.get("probability", 1.0)) < self.config.min_word_probability:
            return {
                "lexical_class": "UNKNOWN",
                "is_function_word": False,
                "is_interrupted": False,
                "is_repetition_adjacent": False,
                "clean_word": "",
                "word_text": "",
                "reliability": "UNKNOWN"
            }
        
        raw_word = word_obj.get("word", "").strip()
        clean_word = "".join(c for c in raw_word if c.isalnum()).lower()
        is_interrupted = raw_word.endswith("-") or raw_word.endswith("—") or raw_word.endswith("…")
        is_func = clean_word in FUNCTION_WORDS
        
        # Check repeated token or stutter cutoff in adjacent tokens
        is_rep_adj = False
        if words_list and word_idx is not None:
            # Check preceding token
            if word_idx > 0:
                prev_w = words_list[word_idx - 1]
                prev_raw = prev_w.get("word", "").strip()
                prev_clean = "".join(c for c in prev_raw if c.isalnum()).lower()
                if prev_clean and (prev_clean == clean_word or (len(prev_clean) >= 2 and clean_word.startswith(prev_clean))):
                    is_rep_adj = True
                elif prev_raw.endswith("-") or prev_raw.endswith("—"):
                    is_rep_adj = True
            # Check following token
            if not is_rep_adj and word_idx < len(words_list) - 1:
                next_w = words_list[word_idx + 1]
                next_raw = next_w.get("word", "").strip()
                next_clean = "".join(c for c in next_raw if c.isalnum()).lower()
                if next_clean and (next_clean == clean_word or (len(clean_word) >= 2 and next_clean.startswith(clean_word))):
                    is_rep_adj = True

        prob = float(word_obj.get("probability", 1.0))
        return {
            "lexical_class": "FUNCTION_WORD" if is_func else "CONTENT_WORD",
            "is_function_word": is_func,
            "is_interrupted": is_interrupted,
            "is_repetition_adjacent": is_rep_adj,
            "clean_word": clean_word,
            "word_text": raw_word,
            "word_probability": round(prob, 3),
            "reliability": "HIGH" if prob >= 0.70 else "MEDIUM"
        }

    def classify_word_position(
        self,
        cand_start: float,
        cand_end: float,
        word_obj: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Determines the candidate's positional relationship to the aligned word.
        Articulatory stutter prolongations overwhelmingly occur on word-initial onsets,
        whereas natural pre-pausal lengthening occurs exclusively on word-final rhymes/codas.
        """
        if not word_obj or float(word_obj.get("probability", 1.0)) < self.config.min_word_probability:
            return {
                "word_position": "UNKNOWN",
                "relative_start_sec": None,
                "relative_end_sec": None,
                "reliability": "UNKNOWN"
            }
        
        ws = float(word_obj.get("start", 0.0))
        we = float(word_obj.get("end", 0.0))
        wdur = we - ws
        if wdur <= 0.0:
            return {
                "word_position": "UNKNOWN",
                "relative_start_sec": None,
                "relative_end_sec": None,
                "reliability": "UNKNOWN"
            }

        margin = min(0.08, wdur * 0.20)
        is_at_start = cand_start <= ws + margin
        is_at_end = cand_end >= we - margin

        if is_at_start and is_at_end:
            pos = "WORD_SPAN"
        elif is_at_start:
            pos = "WORD_INITIAL"
        elif is_at_end:
            pos = "WORD_FINAL"
        else:
            pos = "WORD_MEDIAL"

        prob = float(word_obj.get("probability", 1.0))
        return {
            "word_position": pos,
            "relative_start_sec": round(cand_start - ws, 3),
            "relative_end_sec": round(we - cand_end, 3),
            "reliability": "HIGH" if prob >= 0.70 else "MEDIUM"
        }

    def classify_syntactic_boundary(
        self,
        cand_start: float,
        cand_end: float,
        audio_duration_sec: float,
        words_list: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Identifies whether candidate aligns with a syntactic boundary (sentence-terminal,
        clause boundary, pre-pausal silence, or recording edge) vs intra-phrase position.
        Uses maximum temporal overlap to select the aligned word.
        """
        edge_margin = getattr(self.config, "phrase_final_edge_margin_sec", 0.35)
        pause_thresh = getattr(self.config, "phrase_final_pause_sec", 0.28)
        
        # 1. Proximity to end of recording / utterance
        if (audio_duration_sec - cand_end) <= edge_margin:
            return {
                "boundary_type": "RECORDING_EDGE",
                "raw_discount": getattr(self.config, "recording_edge_discount", 0.25),
                "reason": "near_audio_end",
                "target_word_idx": None,
                "target_word": None
            }

        if not words_list:
            return {
                "boundary_type": "UNKNOWN",
                "raw_discount": 0.0,
                "reason": "no_words_list",
                "target_word_idx": None,
                "target_word": None
            }

        # 2. Find matching word by maximum temporal overlap
        target_idx = None
        best_overlap = -1.0
        cand_mid = (cand_start + cand_end) / 2.0
        
        for i, w in enumerate(words_list):
            ws = float(w.get("start", 0.0))
            we = float(w.get("end", 0.0))
            overlap = min(we, cand_end) - max(ws, cand_start)
            if overlap > best_overlap:
                best_overlap = overlap
                target_idx = i
            elif best_overlap <= 0.0 and ws <= cand_mid <= we:
                target_idx = i
                best_overlap = 0.01

        if target_idx is None:
            # Fallback to closest word by midpoint
            target_idx = min(
                range(len(words_list)),
                key=lambda i: abs(((float(words_list[i].get("start", 0.0)) + float(words_list[i].get("end", 0.0))) / 2.0) - cand_mid)
            )

        curr_w = words_list[target_idx]
        curr_text = curr_w.get("word", "").strip()

        # Interrupted speech with hyphen or dash
        if curr_text.endswith("-") or curr_text.endswith("—") or curr_text.endswith("…"):
            return {
                "boundary_type": "DISFLUENT_BREAK",
                "raw_discount": 0.0,
                "reason": f"interrupted_token('{curr_text}')",
                "target_word_idx": target_idx,
                "target_word": curr_w
            }

        # Major sentence terminal (. ? !)
        if any(curr_text.endswith(p) for p in [".", "?", "!"]):
            return {
                "boundary_type": "SENTENCE_TERMINAL",
                "raw_discount": getattr(self.config, "sentence_terminal_discount", 0.35),
                "reason": f"sentence_terminal('{curr_text}')",
                "target_word_idx": target_idx,
                "target_word": curr_w
            }

        # Minor clause boundary (, ; :)
        if any(curr_text.endswith(p) for p in [",", ";", ":"]):
            return {
                "boundary_type": "CLAUSE_BOUNDARY",
                "raw_discount": getattr(self.config, "clause_boundary_discount", 0.20),
                "reason": f"clause_boundary('{curr_text}')",
                "target_word_idx": target_idx,
                "target_word": curr_w
            }

        # Pre-pausal silence gap to next word
        if target_idx < len(words_list) - 1:
            next_w = words_list[target_idx + 1]
            curr_end = float(curr_w.get("end", 0.0))
            next_start = float(next_w.get("start", 0.0))
            pause_gap = next_start - curr_end
            if pause_gap >= pause_thresh:
                return {
                    "boundary_type": "PRE_PAUSAL_SILENCE",
                    "raw_discount": getattr(self.config, "clause_boundary_discount", 0.20),
                    "reason": f"pre_pausal_silence({pause_gap:.2f}s)",
                    "target_word_idx": target_idx,
                    "target_word": curr_w
                }
        else:
            return {
                "boundary_type": "LAST_WORD",
                "raw_discount": getattr(self.config, "sentence_terminal_discount", 0.35),
                "reason": "last_word_in_transcript",
                "target_word_idx": target_idx,
                "target_word": curr_w
            }

        return {
            "boundary_type": "INTRA_PHRASE",
            "raw_discount": 0.0,
            "reason": "intra_phrase",
            "target_word_idx": target_idx,
            "target_word": curr_w
        }

    def check_phrase_final_lengthening(
        self,
        cand_start: float,
        cand_end: float,
        audio_duration_sec: float,
        words_list: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, str]:
        """
        Determines whether a candidate prolongation coincides with a natural
        phrase-final or sentence-final pre-pausal lengthening boundary.
        Maintained for backwards compatibility with Fix 3.
        """
        boundary_info = self.classify_syntactic_boundary(
            cand_start, cand_end, audio_duration_sec, words_list
        )
        is_pf = boundary_info["boundary_type"] not in ("INTRA_PHRASE", "DISFLUENT_BREAK", "UNKNOWN")
        return is_pf, boundary_info["reason"]

    def analyze_candidate(
        self,
        cand: Dict[str, Any],
        audio_duration_sec: float,
        words_list: Optional[List[Dict[str, Any]]] = None,
        transcript: Optional[str] = None,
        global_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive contextual scoring of a single prolongation candidate.
        Integrates Fix 3 acoustic-temporal features with Fix 4 lexical and syntactic context.
        """
        c_dur = float(cand["duration"])
        c_start = float(cand["start"])
        c_end = float(cand["end"])
        c_mid = (c_start + c_end) / 2.0

        if not self.config.enabled:
            return {
                "context_score": 0.50,
                "context_reliability": "DISABLED",
                "elongation_ratio": 1.0,
                "is_phrase_final": False,
                "phrase_final_reason": "context_disabled",
                "boundary_type": "DISABLED",
                "lexical_class": "DISABLED",
                "word_position": "DISABLED",
                "local_speech_rate": None,
                "mean_syllable_dur_sec": None,
                "is_admitted": c_dur >= 0.50,
                "rejection_reason": "below_duration_floor" if c_dur < 0.50 else None
            }

        # 1. Local speech rate and mean syllable duration
        local_ctx = self.estimate_local_speech_rate(c_mid, words_list, global_context)
        mean_syll_dur = local_ctx.get("mean_syllable_duration_sec")
        local_rate = local_ctx.get("local_speech_rate")
        local_conf = local_ctx.get("confidence", "UNAVAILABLE")

        # Check if Fix 4 lexical/syntactic context is enabled
        use_fix4 = getattr(self.config, "enable_lexical_syntactic_context", True)

        if not use_fix4:
            # Exact Fix 3 branch
            is_pf, pf_reason = self.check_phrase_final_lengthening(
                c_start, c_end, audio_duration_sec, words_list
            )
            if local_conf == "UNAVAILABLE" or mean_syll_dur is None or mean_syll_dur <= 0.0:
                context_score = 0.50
                elongation_ratio = 1.0
                reliability = "NEUTRAL_UNAVAILABLE"
                admitted = c_dur >= 0.50
                rejection_reason = "insufficient_context_for_short_candidate" if c_dur < 0.50 else None
            else:
                reliability = local_conf
                elongation_ratio = round(c_dur / mean_syll_dur, 2)
                base_score = 1.0 / (1.0 + math.exp(-2.2 * (elongation_ratio - 1.75)))
                if is_pf:
                    base_score = max(0.05, base_score - self.config.phrase_final_score_discount)
                if local_rate is not None and local_rate < self.config.slow_speech_threshold:
                    slow_penalty = min(0.25, (self.config.slow_speech_threshold - local_rate) * 0.15)
                    base_score = max(0.05, base_score - slow_penalty)
                if local_rate is not None and local_rate > self.config.fast_speech_threshold:
                    fast_bonus = min(0.15, (local_rate - self.config.fast_speech_threshold) * 0.05)
                    base_score = min(1.0, base_score + fast_bonus)
                context_score = round(base_score, 3)
                if c_dur >= 0.50:
                    admitted = True
                    rejection_reason = None
                else:
                    if elongation_ratio >= self.config.elongation_ratio_threshold and not is_pf and context_score >= self.config.min_context_score_for_admission:
                        admitted = True
                        rejection_reason = None
                    else:
                        admitted = False
                        if is_pf:
                            rejection_reason = f"phrase_final_lengthening({pf_reason})"
                        elif elongation_ratio < self.config.elongation_ratio_threshold:
                            rejection_reason = f"insufficient_elongation_ratio({elongation_ratio:.2f}x < {self.config.elongation_ratio_threshold}x)"
                        else:
                            rejection_reason = f"low_context_score({context_score:.2f} < {self.config.min_context_score_for_admission})"

            return {
                "context_score": context_score,
                "context_reliability": reliability,
                "confidence_level": "UNAVAILABLE" if reliability == "NEUTRAL_UNAVAILABLE" else reliability,
                "elongation_ratio": elongation_ratio,
                "is_phrase_final": is_pf,
                "phrase_final_reason": pf_reason,
                "local_speech_rate": local_rate,
                "mean_syllable_dur_sec": mean_syll_dur,
                "is_admitted": admitted,
                "rejection_reason": rejection_reason
            }

        # Fix 4 Context-Aware Scoring Branch
        boundary_info = self.classify_syntactic_boundary(
            c_start, c_end, audio_duration_sec, words_list
        )
        boundary_type = boundary_info["boundary_type"]
        raw_boundary_discount = boundary_info["raw_discount"]
        target_w = boundary_info["target_word"]
        target_idx = boundary_info["target_word_idx"]

        word_pos_info = self.classify_word_position(c_start, c_end, target_w)
        word_pos = word_pos_info["word_position"]

        lex_info = self.classify_lexical_context(target_w, words_list, target_idx)
        lex_class = lex_info["lexical_class"]

        decision_reasons = []

        if local_conf == "UNAVAILABLE" or mean_syll_dur is None or mean_syll_dur <= 0.0:
            context_score = 0.50
            elongation_ratio = 1.0
            reliability = "NEUTRAL_UNAVAILABLE"
            admitted = c_dur >= 0.50
            rejection_reason = "insufficient_context_for_short_candidate" if c_dur < 0.50 else None
            effective_boundary_discount = 0.0
            lexical_boost = 0.0
            decision_reasons.append("neutral_unavailable_context")
        else:
            reliability = local_conf
            elongation_ratio = round(c_dur / mean_syll_dur, 2)
            base_score = 1.0 / (1.0 + math.exp(-2.2 * (elongation_ratio - 1.75)))

            # Boundary attenuation
            attenuation = 1.0
            if boundary_type in ("SENTENCE_TERMINAL", "CLAUSE_BOUNDARY", "RECORDING_EDGE", "PRE_PAUSAL_SILENCE", "LAST_WORD"):
                min_attenuation = getattr(self.config, "min_boundary_attenuation_floor", 0.15)
                max_ratio = getattr(self.config, "max_boundary_attenuation_ratio", 2.30)
                
                # Word-initial onsets at boundaries are NOT pre-pausal lengthening
                if word_pos == "WORD_INITIAL":
                    attenuation = min_attenuation
                    decision_reasons.append("word_initial_boundary_onset")
                elif elongation_ratio <= 1.60:
                    attenuation = 1.0
                    decision_reasons.append(f"natural_phrase_lengthening({boundary_type})")
                elif elongation_ratio >= max_ratio:
                    attenuation = min_attenuation
                    decision_reasons.append(f"high_elongation_boundary_attenuation({elongation_ratio:.2f}x)")
                else:
                    attenuation = 1.0 - (1.0 - min_attenuation) * ((elongation_ratio - 1.60) / (max_ratio - 1.60))
                    decision_reasons.append(f"partial_boundary_attenuation({elongation_ratio:.2f}x)")
                effective_boundary_discount = round(raw_boundary_discount * attenuation, 3)
            else:
                effective_boundary_discount = 0.0

            score_after_boundary = max(0.05, base_score - effective_boundary_discount)

            # Lexical and positional evidence
            lexical_boost = 0.0
            if lex_info["is_interrupted"]:
                b = getattr(self.config, "interrupted_speech_boost", 0.10)
                lexical_boost += b
                decision_reasons.append(f"interrupted_token(+{b:.2f})")

            if lex_info["is_repetition_adjacent"]:
                b = getattr(self.config, "repetition_proximity_boost", 0.12)
                lexical_boost += b
                decision_reasons.append(f"repetition_proximity(+{b:.2f})")

            if lex_info["is_function_word"] and elongation_ratio >= 1.70:
                b = getattr(self.config, "function_word_elongation_boost", 0.08)
                lexical_boost += b
                decision_reasons.append(f"function_word_elongation(+{b:.2f})")

            if word_pos == "WORD_INITIAL" and elongation_ratio >= 1.70:
                b = getattr(self.config, "word_initial_boost", 0.08)
                lexical_boost += b
                decision_reasons.append(f"word_initial_onset(+{b:.2f})")

            if word_pos == "WORD_FINAL" and boundary_type not in ("INTRA_PHRASE", "DISFLUENT_BREAK", "UNKNOWN"):
                d = getattr(self.config, "word_final_extra_discount", 0.05)
                lexical_boost -= d
                decision_reasons.append(f"word_final_coda_discount(-{d:.2f})")

            # Tempo adjustments
            tempo_adj = 0.0
            if local_rate is not None and local_rate < self.config.slow_speech_threshold:
                tempo_adj = -min(0.25, (self.config.slow_speech_threshold - local_rate) * 0.15)
                decision_reasons.append(f"slow_speech_penalty({tempo_adj:.2f})")
            elif local_rate is not None and local_rate > self.config.fast_speech_threshold:
                tempo_adj = min(0.15, (local_rate - self.config.fast_speech_threshold) * 0.05)
                decision_reasons.append(f"fast_speech_bonus(+{tempo_adj:.2f})")

            final_score = max(0.05, min(1.0, score_after_boundary + lexical_boost + tempo_adj))
            context_score = round(final_score, 3)

            # Admission decision logic
            is_pf = boundary_type not in ("INTRA_PHRASE", "DISFLUENT_BREAK", "UNKNOWN")
            if c_dur >= 0.50:
                admitted = True
                rejection_reason = None
            else:
                if elongation_ratio >= self.config.elongation_ratio_threshold and not is_pf and context_score >= self.config.min_context_score_for_admission:
                    admitted = True
                    rejection_reason = None
                else:
                    admitted = False
                    if is_pf:
                        rejection_reason = f"phrase_final_lengthening({boundary_type})"
                    elif elongation_ratio < self.config.elongation_ratio_threshold:
                        rejection_reason = f"insufficient_elongation_ratio({elongation_ratio:.2f}x < {self.config.elongation_ratio_threshold}x)"
                    else:
                        rejection_reason = f"low_context_score({context_score:.2f} < {self.config.min_context_score_for_admission})"

        is_pf = boundary_type not in ("INTRA_PHRASE", "DISFLUENT_BREAK", "UNKNOWN")
        return {
            "context_score": context_score,
            "context_reliability": reliability,
            "confidence_level": "UNAVAILABLE" if reliability == "NEUTRAL_UNAVAILABLE" else reliability,
            "elongation_ratio": elongation_ratio,
            "is_phrase_final": is_pf,
            "phrase_final_reason": boundary_info["reason"],
            "boundary_type": boundary_type,
            "lexical_class": lex_class,
            "word_position": word_pos,
            "raw_boundary_discount": round(raw_boundary_discount, 3),
            "effective_boundary_discount": round(effective_boundary_discount, 3),
            "lexical_boost": round(lexical_boost, 3),
            "decision_reasons": decision_reasons,
            "local_speech_rate": local_rate,
            "mean_syllable_dur_sec": mean_syll_dur,
            "is_admitted": admitted,
            "rejection_reason": rejection_reason
        }
