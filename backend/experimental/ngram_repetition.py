"""
SpeechAI Multi-Word N-Gram Repetition Detector (Ablation 1)
===========================================================

Deterministic detection of contiguous repeated token sequences of length 2 and 3:
- Normalizes case and boundary punctuation while preserving token sequence.
- Matches contiguous repeated phrases (e.g. "or you could or you could",
  "an intricate headlong, that intricate headlong, intricate headlong").
- Prevents overlap double-counting: evaluates 3-grams first, subsuming sub-2-grams.
- Excludes single-token repeated sequences (e.g. "the the the the"), deferring them
  strictly to unigram repetition logic.
- Aligns matched token spans with word timestamps when available.
"""

import re
from typing import Dict, Any, List, Optional, Tuple, Set


def clean_token(token: str) -> str:
    """Normalizes token by lowercasing and stripping boundary punctuation."""
    t = re.sub(r"[^a-zA-Z0-9']", "", token)
    return t.lower()


class NGramRepetitionDetector:
    """
    Deterministic detector for contiguous 2-word and 3-word phrase repetitions.
    """

    def __init__(
        self,
        allowed_sizes: Tuple[int, ...] = (2, 3),
        base_confidence: float = 0.85,
        multi_iteration_confidence: float = 0.90
    ):
        self.allowed_sizes = tuple(sorted([s for s in allowed_sizes if s in (2, 3)], reverse=True))
        self.base_confidence = base_confidence
        self.multi_iteration_confidence = multi_iteration_confidence

    def detect(
        self,
        transcript: str,
        words_list: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Detects contiguous repeated n-gram phrases in the transcript.
        
        Returns:
            events_count: Number of independent repeated phrase events detected.
            details: List of event dictionaries describing each match.
        """
        if not transcript or not isinstance(transcript, str):
            return 0, []

        raw_words = transcript.strip().split()
        if not raw_words:
            return 0, []

        tokens = [clean_token(w) for w in raw_words]
        valid_tokens = [(t, i) for i, t in enumerate(tokens) if t]
        if len(valid_tokens) < 4:  # Minimum 4 tokens required for a 2-gram repetition
            return 0, []

        words_seq = [t for t, _ in valid_tokens]
        n_words = len(words_seq)

        events = 0
        details: List[Dict[str, Any]] = []
        claimed_indices: Set[int] = set()

        # Process descending order: 3-grams first, then 2-grams
        for n in self.allowed_sizes:
            i = 0
            while i <= n_words - 2 * n:
                # Check if current window start is already subsumed by a longer match
                span_indices = set(range(i, i + 2 * n))
                if span_indices.issubset(claimed_indices):
                    i += 1
                    continue

                phrase1 = words_seq[i : i + n]
                phrase2 = words_seq[i + n : i + 2 * n]

                # Exclude trivial unigram repetitions (e.g. "uh uh" or "odd odd")
                # Such repetitions are handled exclusively by the unigram matcher
                if len(set(phrase1)) <= 1:
                    i += 1
                    continue

                if phrase1 == phrase2:
                    # Count consecutive iterations (2, 3, etc.)
                    iterations = 2
                    while i + (iterations + 1) * n <= n_words:
                        next_phrase = words_seq[i + iterations * n : i + (iterations + 1) * n]
                        if next_phrase == phrase1:
                            iterations += 1
                        else:
                            break

                    total_span_indices = set(range(i, i + iterations * n))
                    claimed_indices.update(total_span_indices)

                    matched_text = " ".join(words_seq[i : i + iterations * n])
                    phrase_text = " ".join(phrase1)
                    conf = self.multi_iteration_confidence if iterations >= 3 else self.base_confidence

                    ev_detail: Dict[str, Any] = {
                        "type": f"ngram_phrase_repetition_{n}gram",
                        "phrase": phrase_text,
                        "text": matched_text,
                        "ngram_size": n,
                        "iterations": iterations,
                        "confidence": conf,
                        "token_start_idx": i,
                        "token_end_idx": i + iterations * n - 1
                    }

                    # Align with timestamps if words_list is provided
                    if words_list and len(words_list) >= len(raw_words):
                        try:
                            orig_s_idx = valid_tokens[i][1]
                            orig_e_idx = valid_tokens[i + iterations * n - 1][1]
                            if orig_e_idx < len(words_list):
                                s_time = float(words_list[orig_s_idx].get("start", 0.0))
                                e_time = float(words_list[orig_e_idx].get("end", 0.0))
                                ev_detail["start"] = round(s_time, 2)
                                ev_detail["end"] = round(e_time, 2)
                                ev_detail["duration"] = round(max(0.1, e_time - s_time), 2)
                        except Exception:
                            pass

                    events += 1
                    details.append(ev_detail)
                    i += iterations * n
                else:
                    i += 1

        return events, details
