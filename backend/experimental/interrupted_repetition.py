"""
SpeechAI Interrupted & Filler-Bridged Repetition Detector (Ablation 3)
======================================================================
Detects lexical phrase and word repetitions separated by intervening
conversational fillers, discourse markers, and hesitation tokens:
- Pattern A: [w1, w2] + [1..k fillers] + [w1, w2]
- Pattern B: [w1, w2, w1] + [1..k fillers] + [w2] (e.g. "it depends, it sort of depends")
- Pattern C: [w1] + [1..k fillers] + [w1] (non-stopword content words, e.g. "still, obviously, still")
- Pattern D: Stem-level variants (e.g. "school, the schools") across hesitation boundaries.

Guarantees:
- Negative fluent controls are preserved by restricting skip tokens to validated filler/hesitation sets.
- Closed-class grammatical stop particles (e.g. "the ... the", "to ... to") are excluded from unigram matches.
- Fully isolated and disabled by default.
"""

import re
from typing import Dict, Any, List, Optional, Tuple, Set


def clean_token(token: str) -> str:
    """Normalizes token by lowercasing and stripping boundary punctuation."""
    t = re.sub(r"[^a-zA-Z0-9']", "", token)
    return t.lower()


# Core conversational fillers and discourse markers
CORE_FILLER_TOKENS = {
    "um", "uh", "er", "erm", "ah", "hmm",
    "like", "actually", "basically",
    "sort", "of", "kind", "you", "know",
    "well", "obviously"
}

# Hesitation connectors commonly present during speech restarts
HESITATION_CONNECTORS = {"and", "at", "the", "or", "so"}

# Closed-class grammatical particles excluded from single-word repetition matching
STOP_PARTICLES = {
    "the", "a", "an", "to", "in", "on", "at", "by",
    "for", "with", "is", "was", "it", "of", "as"
}


def basic_stem(word: str) -> str:
    """Lightweight English stemmer for morphological repetition variants (plurals, participles)."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    return word


class InterruptedRepetitionDetector:
    """
    Detects phrase and word repetitions interrupted by conversational fillers or hesitation words.
    """

    def __init__(
        self,
        max_skip: int = 3,
        allow_stem: bool = True,
        allow_hesitations: bool = True,
        confidence: float = 0.85
    ):
        self.max_skip = max_skip
        self.allow_stem = allow_stem
        self.allow_hesitations = allow_hesitations
        self.confidence = confidence

        self.allowed_fillers = set(CORE_FILLER_TOKENS)
        if self.allow_hesitations:
            self.allowed_fillers.update(HESITATION_CONNECTORS)

    def detect(
        self,
        transcript: str,
        words_list: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Scans transcript for filler-bridged or interrupted repetitions.

        Returns:
            events_count: Number of independent interrupted repetition events detected.
            details: List of event dictionaries describing each match.
        """
        if not transcript or not isinstance(transcript, str):
            return 0, []

        raw_words = transcript.strip().split()
        if not raw_words:
            return 0, []

        tokens = [clean_token(w) for w in raw_words]
        valid_tokens = [(t, i) for i, t in enumerate(tokens) if t]
        if len(valid_tokens) < 3:
            return 0, []

        words_seq = [t for t, _ in valid_tokens]
        n_words = len(words_seq)

        events = 0
        details: List[Dict[str, Any]] = []
        claimed_indices: Set[int] = set()

        # 1. Interrupted 2-gram repetitions
        # Pattern A: [w1, w2] + [1..max_skip fillers] + [w1, w2]
        for i in range(n_words - 4):
            for skip in range(1, self.max_skip + 1):
                if i + 2 + skip + 2 <= n_words:
                    p1 = words_seq[i : i + 2]
                    skipped = words_seq[i + 2 : i + 2 + skip]
                    p2 = words_seq[i + 2 + skip : i + 4 + skip]

                    if all(tok in self.allowed_fillers for tok in skipped):
                        is_match = (p1 == p2) or (
                            self.allow_stem and [basic_stem(w) for w in p1] == [basic_stem(w) for w in p2]
                        )
                        if is_match and len(set(p1)) > 1:
                            span = set(range(i, i + 4 + skip))
                            if not span.issubset(claimed_indices):
                                claimed_indices.update(span)
                                ev_detail: Dict[str, Any] = {
                                    "type": "interrupted_ngram_repetition_2gram",
                                    "phrase": " ".join(p1),
                                    "intervening_fillers": " ".join(skipped),
                                    "text": " ".join(words_seq[i : i + 4 + skip]),
                                    "confidence": self.confidence,
                                    "token_start_idx": i,
                                    "token_end_idx": i + 4 + skip - 1
                                }
                                self._attach_timestamps(ev_detail, valid_tokens, words_list, i, i + 4 + skip - 1)
                                events += 1
                                details.append(ev_detail)

            # Pattern B: [w1, w2, w1] + [1..max_skip fillers] + [w2] (e.g. "it depends it sort of depends")
            for skip in range(1, self.max_skip + 1):
                if i + 3 + skip + 1 <= n_words:
                    w1a, w2a, w1b = words_seq[i], words_seq[i + 1], words_seq[i + 2]
                    skipped = words_seq[i + 3 : i + 3 + skip]
                    w2b = words_seq[i + 3 + skip]

                    w2_match = (w2a == w2b) or (self.allow_stem and basic_stem(w2a) == basic_stem(w2b))
                    if w1a == w1b and w2_match and w1a != w2a:
                        if all(tok in self.allowed_fillers for tok in skipped):
                            span = set(range(i, i + 4 + skip))
                            if not span.issubset(claimed_indices):
                                claimed_indices.update(span)
                                ev_detail = {
                                    "type": "interrupted_ngram_repetition_2gram",
                                    "phrase": f"{w1a} {w2a}",
                                    "intervening_fillers": " ".join(skipped),
                                    "text": " ".join(words_seq[i : i + 4 + skip]),
                                    "confidence": self.confidence,
                                    "token_start_idx": i,
                                    "token_end_idx": i + 4 + skip - 1
                                }
                                self._attach_timestamps(ev_detail, valid_tokens, words_list, i, i + 4 + skip - 1)
                                events += 1
                                details.append(ev_detail)

        # 2. Interrupted single-word repetitions
        # Pattern C: [w1] + [1..max_skip fillers] + [w1]
        for i in range(n_words - 2):
            for skip in range(1, self.max_skip + 1):
                if i + 1 + skip + 1 <= n_words:
                    w1 = words_seq[i]
                    skipped = words_seq[i + 1 : i + 1 + skip]
                    w2 = words_seq[i + 1 + skip]

                    is_match = (w1 == w2) or (self.allow_stem and basic_stem(w1) == basic_stem(w2))
                    if is_match and w1 not in STOP_PARTICLES and len(w1) >= 3:
                        if all(tok in self.allowed_fillers for tok in skipped):
                            span = set(range(i, i + 2 + skip))
                            if not span.issubset(claimed_indices):
                                claimed_indices.update(span)
                                ev_detail = {
                                    "type": "interrupted_word_repetition",
                                    "phrase": w1,
                                    "intervening_fillers": " ".join(skipped),
                                    "text": " ".join(words_seq[i : i + 2 + skip]),
                                    "confidence": self.confidence,
                                    "token_start_idx": i,
                                    "token_end_idx": i + 2 + skip - 1
                                }
                                self._attach_timestamps(ev_detail, valid_tokens, words_list, i, i + 2 + skip - 1)
                                events += 1
                                details.append(ev_detail)

        return events, details

    def _attach_timestamps(
        self,
        ev_detail: Dict[str, Any],
        valid_tokens: List[Tuple[str, int]],
        words_list: Optional[List[Dict[str, Any]]],
        start_idx: int,
        end_idx: int
    ) -> None:
        """Aligns detected token indices with word timestamps when available."""
        if words_list and len(words_list) > 0:
            try:
                orig_s_idx = valid_tokens[start_idx][1]
                orig_e_idx = valid_tokens[end_idx][1]
                if orig_s_idx < len(words_list) and orig_e_idx < len(words_list):
                    s_time = float(words_list[orig_s_idx].get("start", 0.0))
                    e_time = float(words_list[orig_e_idx].get("end", 0.0))
                    ev_detail["start"] = round(s_time, 2)
                    ev_detail["end"] = round(e_time, 2)
                    ev_detail["duration"] = round(max(0.1, e_time - s_time), 2)
            except Exception:
                pass
