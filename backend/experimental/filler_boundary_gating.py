"""
Filler-Aware and Boundary-Aware Gating Module for SpeechAI (Ablation 2)
=======================================================================
Provides isolated post-candidate gating to eliminate false positives caused by:
1. Prolonged conversational filler vowels ("um", "uh", "er", "ah", "oh").
2. Whole-word lexical repetition of pure conversational fillers ("uh uh", "um um").
3. Pre-pausal and sentence-boundary vowel lengthening under 0.65s.
4. Inter-word hesitation silences immediately adjacent to conversational filler tokens.

This module is designed to be invoked behind the disabled-by-default configuration flag:
`config.enable_filler_boundary_gating = True`.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Set, Optional

from .asr_loop_guard import ASRLoopGuard

logger = logging.getLogger("SpeechAI_FillerBoundaryGating")

# Phonologically standard English conversational hesitation markers
FILLER_WORDS: Set[str] = {
    "um", "uh", "er", "erm", "ah", "hmm", "mm", "umm", "uhh", "err", "oh"
}

def clean_token(text: str) -> str:
    """Strips non-alphanumeric punctuation and lowercases the token."""
    return re.sub(r'[^\w]', '', text).lower()

def is_filler_token(text: str) -> bool:
    """
    Determines if a word token represents a conversational filler interjection.
    Supports canonical forms and elongated orthographic variants (e.g., 'uuuh', 'uumm').
    """
    clean = clean_token(text)
    if not clean:
        return False
    if clean in FILLER_WORDS:
        return True
    if re.match(r'^u+h+m*$', clean) or re.match(r'^u+m+$', clean) or re.match(r'^e+r+m*$', clean) or re.match(r'^a+h+$', clean) or re.match(r'^o+h+$', clean):
        return True
    return False


class FillerBoundaryGater:
    """
    Applies linguistic and temporal gating rules to separate genuine articulatory
    stuttering from normal conversational hesitations and phrase boundary dynamics.
    """

    def __init__(
        self,
        config: Any = None,
        filler_overlap_threshold: float = 0.35,
        filler_word_coverage_threshold: float = 0.40,
        boundary_prolongation_min_duration_sec: float = 0.65,
        boundary_punctuation: str = ".,?!;:",
        gate_filler_blocks: bool = True,
        enable_multi_iteration_filler_stutter: bool = False,
        min_filler_repetition_count: int = 3,
        rapid_filler_isi_threshold_sec: float = 0.35,
        enable_asr_loop_guard: bool = False,
        asr_loop_min_zero_dur_tokens: int = 3,
        asr_loop_min_identical_starts: int = 4,
        asr_loop_min_isi_floor_sec: float = 0.015,
        asr_loop_min_span_floor_sec: float = 0.05,
        asr_loop_min_repetition_streak: int = 5,
        asr_loop_total_zero_dur_threshold: int = 5
    ):
        if config is not None and not isinstance(config, (int, float, str)):
            self.filler_overlap_threshold = getattr(config, "filler_overlap_threshold", filler_overlap_threshold)
            self.filler_word_coverage_threshold = getattr(config, "filler_word_coverage_threshold", filler_word_coverage_threshold)
            self.boundary_prolongation_min_duration_sec = getattr(config, "boundary_prolongation_min_duration_sec", boundary_prolongation_min_duration_sec)
            self.boundary_punctuation = getattr(config, "boundary_punctuation", boundary_punctuation)
            self.gate_filler_blocks = getattr(config, "gate_filler_blocks", gate_filler_blocks)
            self.enable_multi_iteration_filler_stutter = getattr(config, "enable_multi_iteration_filler_stutter", enable_multi_iteration_filler_stutter)
            self.min_filler_repetition_count = getattr(config, "min_filler_repetition_count", min_filler_repetition_count)
            self.rapid_filler_isi_threshold_sec = getattr(config, "rapid_filler_isi_threshold_sec", rapid_filler_isi_threshold_sec)
            self.enable_asr_loop_guard = getattr(config, "enable_asr_loop_guard", enable_asr_loop_guard)
            self.asr_loop_min_zero_dur_tokens = getattr(config, "asr_loop_min_zero_dur_tokens", asr_loop_min_zero_dur_tokens)
            self.asr_loop_min_identical_starts = getattr(config, "asr_loop_min_identical_starts", asr_loop_min_identical_starts)
            self.asr_loop_min_isi_floor_sec = getattr(config, "asr_loop_min_isi_floor_sec", asr_loop_min_isi_floor_sec)
            self.asr_loop_min_span_floor_sec = getattr(config, "asr_loop_min_span_floor_sec", asr_loop_min_span_floor_sec)
            self.asr_loop_min_repetition_streak = getattr(config, "asr_loop_min_repetition_streak", asr_loop_min_repetition_streak)
            self.asr_loop_total_zero_dur_threshold = getattr(config, "asr_loop_total_zero_dur_threshold", asr_loop_total_zero_dur_threshold)
        else:
            self.filler_overlap_threshold = config if isinstance(config, float) else filler_overlap_threshold
            self.filler_word_coverage_threshold = filler_word_coverage_threshold
            self.boundary_prolongation_min_duration_sec = boundary_prolongation_min_duration_sec
            self.boundary_punctuation = boundary_punctuation
            self.gate_filler_blocks = gate_filler_blocks
            self.enable_multi_iteration_filler_stutter = enable_multi_iteration_filler_stutter
            self.min_filler_repetition_count = min_filler_repetition_count
            self.rapid_filler_isi_threshold_sec = rapid_filler_isi_threshold_sec
            self.enable_asr_loop_guard = enable_asr_loop_guard
            self.asr_loop_min_zero_dur_tokens = asr_loop_min_zero_dur_tokens
            self.asr_loop_min_identical_starts = asr_loop_min_identical_starts
            self.asr_loop_min_isi_floor_sec = asr_loop_min_isi_floor_sec
            self.asr_loop_min_span_floor_sec = asr_loop_min_span_floor_sec
            self.asr_loop_min_repetition_streak = asr_loop_min_repetition_streak
            self.asr_loop_total_zero_dur_threshold = asr_loop_total_zero_dur_threshold

        if self.enable_asr_loop_guard:
            self.loop_guard = ASRLoopGuard(
                min_zero_dur_tokens=self.asr_loop_min_zero_dur_tokens,
                min_identical_starts=self.asr_loop_min_identical_starts,
                min_isi_floor_sec=self.asr_loop_min_isi_floor_sec,
                min_span_floor_sec=self.asr_loop_min_span_floor_sec,
                min_loop_repetition_streak=self.asr_loop_min_repetition_streak,
                total_zero_dur_threshold=self.asr_loop_total_zero_dur_threshold
            )
        else:
            self.loop_guard = None

    def gate_prolongations(
        self,
        prolongations: List[Dict[str, Any]],
        words_list: Optional[List[Dict[str, Any]]] = None,
        transcript: Optional[str] = None,
        total_duration: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filters prolongations overlapping fillers or subthreshold boundary lengthening."""
        retained = []
        gated = []
        words = words_list or []

        for p in prolongations:
            c_start = float(p.get("start_time", p.get("start", 0.0)))
            c_end = float(p.get("end_time", p.get("end", 0.0)))
            c_dur = float(p.get("duration", max(0.0, c_end - c_start)))
            reason = str(p.get("phrase_final_reason", p.get("reason", "")))

            # 1. Overlap with conversational filler words
            is_on_filler = False
            matched_filler = ""
            for w in words:
                w_str = w.get("word", "")
                if is_filler_token(w_str):
                    w_s = float(w.get("start", 0.0))
                    w_e = float(w.get("end", 0.0))
                    ov_s = max(c_start, w_s)
                    ov_e = min(c_end, w_e)
                    ov = max(0.0, ov_e - ov_s)
                    c_cov = ov / max(0.01, c_dur)
                    w_cov = ov / max(0.01, w_e - w_s)
                    if c_cov >= self.filler_overlap_threshold or w_cov >= self.filler_word_coverage_threshold:
                        is_on_filler = True
                        matched_filler = w_str.strip()
                        break

            if not is_on_filler:
                for fw in FILLER_WORDS:
                    pattern = rf"['\"]{fw}[,\.?!;:…\-]?['\"]"
                    if re.search(pattern, reason.lower()):
                        is_on_filler = True
                        matched_filler = fw
                        break

            if is_on_filler:
                entry = dict(p)
                entry["reason"] = "filler_prolongation"
                entry["overlapping_word"] = matched_filler
                entry["gated_reason"] = f"filler_vowel_prolongation('{matched_filler}')"
                gated.append(entry)
                continue

            # 2. Boundary lengthening subthreshold check
            is_boundary = (
                "sentence_terminal" in reason or 
                "clause_boundary" in reason or 
                p.get("is_phrase_final", False)
            )
            # Also check if it ends near the final word containing punctuation or total audio duration
            if not is_boundary and words:
                last_w = words[-1]
                last_w_str = last_w.get("word", "")
                last_w_end = float(last_w.get("end", 0.0))
                if any(last_w_str.endswith(punct) for punct in self.boundary_punctuation):
                    if abs(c_end - last_w_end) < 0.20 or (total_duration and abs(total_duration - c_end) < 0.35):
                        is_boundary = True

            if is_boundary and c_dur < self.boundary_prolongation_min_duration_sec:
                entry = dict(p)
                entry["reason"] = "sentence_boundary_lengthening"
                entry["gated_reason"] = (
                    f"boundary_lengthening_subthreshold("
                    f"{c_dur:.2f}s < {self.boundary_prolongation_min_duration_sec:.2f}s)"
                )
                gated.append(entry)
                continue

            retained.append(p)

        return retained, gated

    def _extract_filler_streaks(
        self,
        words_list: Optional[List[Dict[str, Any]]] = None,
        transcript: Optional[str] = None,
        sanitizer_status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Identifies consecutive streaks of identical filler tokens with timing and pacing."""
        tokens_info = []
        if words_list and len(words_list) > 0:
            for idx, w in enumerate(words_list):
                w_str = w.get("word", "")
                c = clean_token(w_str)
                if c:
                    tokens_info.append({
                        "word": c,
                        "raw": w_str,
                        "start": float(w.get("start", 0.0)),
                        "end": float(w.get("end", 0.0)),
                        "is_filler": is_filler_token(w_str),
                        "orig_idx": idx
                    })
        elif transcript:
            for idx, w in enumerate(transcript.split()):
                c = clean_token(w)
                if c:
                    tokens_info.append({
                        "word": c,
                        "raw": w,
                        "start": 0.0,
                        "end": 0.0,
                        "is_filler": is_filler_token(w),
                        "orig_idx": idx
                    })

        streaks = []
        curr_streak = []
        for t in tokens_info:
            if t["is_filler"]:
                if not curr_streak or t["word"] == curr_streak[-1]["word"]:
                    curr_streak.append(t)
                else:
                    if len(curr_streak) >= 2:
                        streaks.append(curr_streak)
                    curr_streak = [t]
            else:
                if len(curr_streak) >= 2:
                    streaks.append(curr_streak)
                curr_streak = []
        if len(curr_streak) >= 2:
            streaks.append(curr_streak)

        analyzed_streaks = []
        for s in streaks:
            count = len(s)
            filler_word = s[0]["word"]
            has_timestamps = any(t["end"] > 0.0 for t in s)
            total_span = max(0.0, s[-1]["end"] - s[0]["start"]) if has_timestamps else 0.0
            mean_isi = total_span / (count - 1) if (has_timestamps and count > 1) else 0.0

            # Rapid struggle repetition vs leisurely conversational hesitation
            is_rapid = (round(mean_isi, 4) <= self.rapid_filler_isi_threshold_sec and has_timestamps) or (count >= 5)

            # Phase 4C.3: ASR repetition-loop safeguard check
            is_loop_artifact = False
            loop_reason = ""
            if self.enable_asr_loop_guard and self.loop_guard:
                streak_zero_dur = sum(1 for t in s if t.get("start") == t.get("end"))
                if sanitizer_status == "DEGENERATE_LOOP_RECOVERED":
                    is_loop_artifact = True
                    loop_reason = "Sanitizer reported DEGENERATE_LOOP_RECOVERED"
                elif streak_zero_dur >= self.loop_guard.min_zero_dur_tokens:
                    is_loop_artifact = True
                    loop_reason = f"Streak contains {streak_zero_dur} zero-duration tokens"
                elif count >= self.loop_guard.min_loop_repetition_streak and total_span < self.loop_guard.min_span_floor_sec:
                    is_loop_artifact = True
                    loop_reason = f"Streak span ({total_span:.3f}s) below physical floor ({self.loop_guard.min_span_floor_sec}s)"
                elif count >= self.loop_guard.min_loop_repetition_streak and mean_isi < self.loop_guard.min_isi_floor_sec:
                    is_loop_artifact = True
                    loop_reason = f"Streak mean ISI ({mean_isi:.4f}s) below articulatory floor ({self.loop_guard.min_isi_floor_sec}s)"

            if is_loop_artifact:
                conf = 0.05
                is_struggle = False
                classification = "ASR_LOOP_ARTIFACT"
            elif count < self.min_filler_repetition_count:
                conf = 0.30
                is_struggle = False
                classification = "CONVERSATIONAL_FILLER"
            elif count == 3:
                if is_rapid:
                    conf = 0.60
                    is_struggle = True
                    classification = "RAPID_FILLER_STRUGGLE"
                else:
                    conf = 0.50
                    is_struggle = False
                    classification = "LEISURELY_FILLER_HESITATION"
            elif count == 4:
                conf = 0.65 if is_rapid else 0.55
                is_struggle = True if is_rapid else False
                classification = "REPEATED_FILLER_STRUGGLE"
            else: # count >= 5
                conf = round(min(0.85, 0.70 + (count - 5) * 0.03), 2)
                is_struggle = True
                classification = "SEVERE_FILLER_STRUGGLE_CHAIN"

            analyzed_streaks.append({
                "filler_word": filler_word,
                "count": count,
                "has_timestamps": has_timestamps,
                "total_span_sec": round(total_span, 2) if has_timestamps else None,
                "mean_isi_sec": round(mean_isi, 3) if has_timestamps else None,
                "is_rapid": is_rapid,
                "confidence": conf,
                "is_struggle": is_struggle,
                "classification": classification,
                "is_loop_artifact": is_loop_artifact,
                "loop_reason": loop_reason,
                "start": s[0]["start"],
                "end": s[-1]["end"],
                "tokens": [t["word"] for t in s]
            })

        return analyzed_streaks

    def gate_repetitions(
        self,
        repetitions: List[Dict[str, Any]],
        words_list: Optional[List[Dict[str, Any]]] = None,
        transcript: Optional[str] = None,
        sanitizer_status: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filters pure filler token repetitions ('uh uh', 'um um')."""
        retained = []
        gated = []

        if not self.enable_multi_iteration_filler_stutter:
            for r in repetitions:
                rep_text = r.get("phrase", r.get("text", ""))
                tokens = [clean_token(w) for w in rep_text.split() if clean_token(w)]
                if len(tokens) > 0 and all(is_filler_token(tok) for tok in tokens):
                    entry = dict(r)
                    entry["reason"] = "filler_repetition"
                    entry["gated_reason"] = f"filler_token_repetition('{rep_text}')"
                    gated.append(entry)
                else:
                    retained.append(r)
            return retained, gated

        # When enable_multi_iteration_filler_stutter is True:
        streaks = self._extract_filler_streaks(words_list, transcript, sanitizer_status=sanitizer_status)
        emitted_streaks: Set[str] = set()

        for r in repetitions:
            rep_text = r.get("phrase", r.get("text", ""))
            tokens = [clean_token(w) for w in rep_text.split() if clean_token(w)]
            is_filler_rep = len(tokens) > 0 and all(is_filler_token(tok) for tok in tokens)

            if not is_filler_rep:
                retained.append(r)
                continue

            filler_word = tokens[0]
            # Match to an extracted streak of this filler
            matching_streak = next((s for s in streaks if s["filler_word"] == filler_word and s["count"] >= self.min_filler_repetition_count), None)

            if matching_streak:
                if matching_streak.get("is_loop_artifact", False):
                    entry = dict(r)
                    entry["reason"] = "asr_loop_artifact"
                    entry["gated_reason"] = f"asr_loop_artifact_suppressed('{filler_word}' x{matching_streak['count']}: {matching_streak.get('loop_reason', '')})"
                    gated.append(entry)
                    continue
                streak_id = f"{matching_streak['filler_word']}_{matching_streak['count']}"
                if streak_id not in emitted_streaks:
                    emitted_streaks.add(streak_id)
                    entry = dict(r)
                    if matching_streak["is_struggle"]:
                        entry["type"] = "multi_iteration_filler_struggle"
                        entry["confidence"] = matching_streak["confidence"]
                        entry["iteration_count"] = matching_streak["count"]
                        entry["mean_isi_sec"] = matching_streak["mean_isi_sec"]
                        entry["is_rapid"] = matching_streak["is_rapid"]
                        entry["classification"] = matching_streak["classification"]
                        entry["evidence"] = (
                            f"{matching_streak['count']}x '{filler_word}' "
                            f"(span: {matching_streak['total_span_sec']}s, "
                            f"mean_isi: {matching_streak['mean_isi_sec']}s, "
                            f"rapid: {matching_streak['is_rapid']})"
                        )
                    else:
                        # Leisurely / ambiguous hesitation: retain as uncertain candidate
                        entry["type"] = "multi_iteration_filler_hesitation"
                        entry["confidence"] = matching_streak["confidence"]  # calibrated 0.50
                        entry["iteration_count"] = matching_streak["count"]
                        entry["mean_isi_sec"] = matching_streak["mean_isi_sec"]
                        entry["is_rapid"] = matching_streak["is_rapid"]
                        entry["classification"] = "LEISURELY_FILLER_HESITATION"
                        entry["evidence"] = (
                            f"{matching_streak['count']}x '{filler_word}' "
                            f"(span: {matching_streak['total_span_sec']}s, "
                            f"mean_isi: {matching_streak['mean_isi_sec']}s, "
                            f"leisurely_conversational_hesitation)"
                        )
                    retained.append(entry)
                else:
                    # Duplicate pair within already emitted streak: subsume
                    entry = dict(r)
                    entry["reason"] = "filler_repetition_subsumed_by_streak"
                    gated.append(entry)
            else:
                # Ordinary filler repetition (< min_filler_repetition_count)
                entry = dict(r)
                entry["reason"] = "filler_repetition"
                entry["gated_reason"] = f"filler_token_repetition('{rep_text}')"
                gated.append(entry)

        return retained, gated

    def gate_blocks(
        self,
        blocks: List[Dict[str, Any]],
        words_list: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filters silence blocks immediately adjacent to conversational filler tokens."""
        if not self.gate_filler_blocks:
            return list(blocks), []

        retained = []
        gated = []
        words = words_list or []

        for b in blocks:
            b_start = float(b.get("start_time", b.get("start", 0.0)))
            b_end = float(b.get("end_time", b.get("end", 0.0)))

            is_gated_pause = False
            gated_why = ""

            for idx in range(len(words) - 1):
                w1 = words[idx]
                w2 = words[idx + 1]
                w1_e = float(w1.get("end", 0.0))
                w2_s = float(w2.get("start", 0.0))
                if abs(w1_e - b_start) < 0.25 and abs(w2_s - b_end) < 0.25:
                    w1_tok = w1.get("word", "").strip()
                    w2_tok = w2.get("word", "").strip()

                    # Pause adjacent to conversational filler
                    if is_filler_token(w1_tok) or is_filler_token(w2_tok):
                        is_gated_pause = True
                        gated_why = f"hesitation_pause_adjacent_to_filler('{w1_tok} -> {w2_tok}')"
                        break

                    # Syntactic pause preceding a filler or functional token
                    if any(w1_tok.endswith(p) for p in [",", "."]) and (
                        is_filler_token(w2_tok) or (idx + 2 < len(words) and is_filler_token(words[idx + 2].get("word", "")))
                    ):
                        is_gated_pause = True
                        gated_why = f"syntactic_pause_preceding_filler('{w1_tok} -> {w2_tok}')"
                        break

            if is_gated_pause:
                entry = dict(b)
                entry["reason"] = "filler_adjacent_block"
                entry["gated_reason"] = gated_why
                gated.append(entry)
            else:
                retained.append(b)

        return retained, gated

    def gate_details(
        self,
        details: List[Dict[str, Any]],
        words_list: Optional[List[Dict[str, Any]]] = None,
        transcript: Optional[str] = None,
        sanitizer_status: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filters a mixed list of candidate disfluency event dictionaries."""
        prolongations = [d for d in details if d.get("type") == "acoustic_prolongation"]
        repetitions = [d for d in details if "repetition" in d.get("type", "") or d.get("type") == "broken_syllable_fragment"]
        blocks = [d for d in details if "block" in d.get("type", "")]
        others = [d for d in details if d not in prolongations and d not in repetitions and d not in blocks]

        ret_prol, gate_prol = self.gate_prolongations(prolongations, words_list, transcript)
        ret_rep, gate_rep = self.gate_repetitions(repetitions, words_list, transcript=transcript, sanitizer_status=sanitizer_status)
        ret_block, gate_block = self.gate_blocks(blocks, words_list)

        # Global loop audit on any remaining repetitions if loop guard is enabled
        if self.enable_asr_loop_guard and self.loop_guard:
            audit = self.loop_guard.audit_clip(transcript or "", words_list, sanitizer_status=sanitizer_status)
            if audit["is_loop_artifact"]:
                ret_rep, extra_suppressed = self.loop_guard.filter_lexical_repetitions(ret_rep, audit)
                gate_rep.extend(extra_suppressed)

        retained = ret_prol + ret_rep + ret_block + others
        gated = gate_prol + gate_rep + gate_block
        return retained, gated
