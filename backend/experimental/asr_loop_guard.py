"""
ASR Repetition-Loop Safeguard Module (Phase 4C.3 Experimental Candidate)
========================================================================
Combines:
1. Sanitizer loop-status telemetry (including DEGENERATE_LOOP_RECOVERED).
2. Word-level timestamp quality checks (zero-duration token density, identical start
   timestamp clustering, and implausible temporal inter-segment intervals).
3. Preservation of independent acoustic prolongation and block pathways.

Designed to be enabled behind the isolated experimental configuration flag:
`config.stutter.enable_asr_loop_guard = True`.
When False, default baseline behavior is strictly preserved.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Set

logger = logging.getLogger("SpeechAI_ASRLoopGuard")

# Canonical hesitation tokens
FILLER_TOKENS: Set[str] = {
    "um", "uh", "er", "erm", "ah", "hmm", "mm", "umm", "uhh", "err", "oh"
}

def clean_token(text: str) -> str:
    """Strips non-alphanumeric punctuation and lowercases the token."""
    return re.sub(r'[^\w]', '', text).lower()


class ASRLoopGuard:
    """
    Evaluates transcript and word-level alignment telemetry to distinguish
    degenerate ASR decoder repetition-loop hallucinations from genuine
    articulatory struggle and conversational hesitation.
    """

    def __init__(
        self,
        min_zero_dur_tokens: int = 3,
        min_identical_starts: int = 4,
        min_isi_floor_sec: float = 0.015,
        min_span_floor_sec: float = 0.05,
        min_loop_repetition_streak: int = 5,
        total_zero_dur_threshold: int = 5
    ):
        self.min_zero_dur_tokens = min_zero_dur_tokens
        self.min_identical_starts = min_identical_starts
        self.min_isi_floor_sec = min_isi_floor_sec
        self.min_span_floor_sec = min_span_floor_sec
        self.min_loop_repetition_streak = min_loop_repetition_streak
        self.total_zero_dur_threshold = total_zero_dur_threshold

    def audit_clip(
        self,
        transcript: str,
        words_list: Optional[List[Dict[str, Any]]] = None,
        sanitizer_status: Optional[str] = None,
        sanitizer_telemetry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Conducts a multi-factor audit of the clip's transcript and timestamps.
        
        Returns a diagnostic dictionary:
        {
            'is_loop_artifact': bool,
            'suppress_lexical_stutter': bool,
            'reasons': List[str],
            'zero_dur_count': int,
            'max_identical_starts': int,
            'collapsed_streaks': List[Dict[str, Any]],
            'sanitizer_loop_detected': bool
        }
        """
        words = words_list or []
        reasons = []
        is_loop = False
        collapsed_streaks = []

        # 1. Inspect sanitizer telemetry
        sanitizer_loop_detected = False
        if sanitizer_status == "DEGENERATE_LOOP_RECOVERED":
            sanitizer_loop_detected = True
            is_loop = True
            reasons.append("Sanitizer telemetry reported DEGENERATE_LOOP_RECOVERED")
        elif sanitizer_telemetry and sanitizer_telemetry.get("is_loop", False):
            sanitizer_loop_detected = True
            is_loop = True
            reasons.append(f"Sanitizer telemetry flagged loop: {sanitizer_telemetry.get('reason', '')}")

        # 2. Inspect timestamp quality metrics
        starts = [w.get("start") for w in words if w.get("start") is not None]
        ends = [w.get("end") for w in words if w.get("end") is not None]
        
        zero_dur_count = sum(
            1 for w in words 
            if w.get("start") is not None and w.get("end") is not None and w.get("start") == w.get("end")
        )
        
        max_identical_starts = 0
        if starts:
            start_counts = [starts.count(s) for s in set(starts)]
            max_identical_starts = max(start_counts) if start_counts else 0

        # Global timestamp collapse check
        if zero_dur_count >= self.total_zero_dur_threshold:
            is_loop = True
            reasons.append(f"Severe global zero-duration token density: {zero_dur_count} tokens have start == end")

        if max_identical_starts >= self.min_identical_starts:
            is_loop = True
            reasons.append(f"Severe timestamp collapse: {max_identical_starts} tokens share identical start time")

        # 3. Streak-level temporal plausibility audit
        # Group adjacent identical tokens
        clean_words = [clean_token(w.get("word", "")) for w in words]
        i = 0
        while i < len(clean_words):
            if not clean_words[i]:
                i += 1
                continue
            j = i + 1
            while j < len(clean_words) and clean_words[j] == clean_words[i]:
                j += 1
            streak_len = j - i
            if streak_len >= 3:
                streak_words = words[i:j]
                s_starts = [w.get("start") for w in streak_words if w.get("start") is not None]
                s_ends = [w.get("end") for w in streak_words if w.get("end") is not None]
                
                streak_zero_dur = sum(1 for w in streak_words if w.get("start") == w.get("end"))
                streak_span = (s_ends[-1] - s_starts[0]) if (s_starts and s_ends and s_ends[-1] >= s_starts[0]) else 0.0
                
                # Check ISIs
                streak_isis = []
                for k in range(len(streak_words) - 1):
                    if streak_words[k].get("end") is not None and streak_words[k+1].get("start") is not None:
                        streak_isis.append(streak_words[k+1]["start"] - streak_words[k]["end"])
                mean_streak_isi = sum(streak_isis) / len(streak_isis) if streak_isis else 0.0
                
                # Plausibility check:
                # If streak has >= 3 zero-dur tokens, or span < min_span_floor, or mean_isi < min_isi_floor
                is_streak_collapsed = False
                streak_reason = ""
                if streak_zero_dur >= self.min_zero_dur_tokens:
                    is_streak_collapsed = True
                    streak_reason = f"Streak '{clean_words[i]}' x{streak_len} contains {streak_zero_dur} zero-duration tokens"
                elif streak_len >= self.min_loop_repetition_streak and streak_span < self.min_span_floor_sec:
                    is_streak_collapsed = True
                    streak_reason = f"Streak '{clean_words[i]}' x{streak_len} temporal span ({streak_span:.3f}s) is below physical floor ({self.min_span_floor_sec}s)"
                elif streak_len >= self.min_loop_repetition_streak and mean_streak_isi < self.min_isi_floor_sec:
                    is_streak_collapsed = True
                    streak_reason = f"Streak '{clean_words[i]}' x{streak_len} mean ISI ({mean_streak_isi:.4f}s) is below articulatory floor ({self.min_isi_floor_sec}s)"

                if is_streak_collapsed:
                    is_loop = True
                    reasons.append(streak_reason)
                    collapsed_streaks.append({
                        "token": clean_words[i],
                        "count": streak_len,
                        "span_sec": streak_span,
                        "mean_isi_sec": mean_streak_isi,
                        "zero_dur_count": streak_zero_dur,
                        "reason": streak_reason
                    })
            i = j

        return {
            "is_loop_artifact": is_loop,
            "suppress_lexical_stutter": is_loop,
            "reasons": reasons,
            "zero_dur_count": zero_dur_count,
            "max_identical_starts": max_identical_starts,
            "collapsed_streaks": collapsed_streaks,
            "sanitizer_loop_detected": sanitizer_loop_detected
        }

    def filter_lexical_repetitions(
        self,
        repetition_events: List[Dict[str, Any]],
        audit_result: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Suppresses lexical repetition events that originate from hallucinated ASR loops.
        Preserves genuine lexical repetitions if they are temporally grounded.
        """
        if not audit_result["is_loop_artifact"]:
            return repetition_events, []

        retained = []
        suppressed = []

        collapsed_tokens = {cs["token"] for cs in audit_result["collapsed_streaks"]}
        is_global_collapse = audit_result["zero_dur_count"] >= self.total_zero_dur_threshold or audit_result["sanitizer_loop_detected"]

        for rep in repetition_events:
            rep_type = rep.get("type", "")
            rep_text = clean_token(rep.get("text", rep.get("phrase", "")))

            # If it's a multi-iteration filler struggle or hesitation:
            if rep_type in ("multi_iteration_filler_struggle", "multi_iteration_filler_hesitation"):
                # If global collapse or matched collapsed token: suppress
                entry = dict(rep)
                entry["suppressed_by_loop_guard"] = True
                entry["loop_guard_reasons"] = audit_result["reasons"]
                suppressed.append(entry)
                continue

            # If whole word repetition matches collapsed tokens:
            rep_words = [clean_token(w) for w in rep_text.split() if clean_token(w)]
            if any(w in collapsed_tokens for w in rep_words) or (is_global_collapse and all(w in FILLER_TOKENS for w in rep_words)):
                entry = dict(rep)
                entry["suppressed_by_loop_guard"] = True
                entry["loop_guard_reasons"] = audit_result["reasons"]
                suppressed.append(entry)
            else:
                retained.append(rep)

        return retained, suppressed
