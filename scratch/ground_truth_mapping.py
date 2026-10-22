"""
SpeechAI Ground-Truth Mapping Logic
===================================
Module: scratch.ground_truth_mapping
Scope: Shared, import-safe functions for deriving clip-level ground-truth
       labels from raw dataset annotations (FluencyBank / SEP-28k and Libri_Stutter).

Design Principles:
- 100% Import Safe: No top-level execution, dataset scanning, or audio processing.
- Explicit Consensus Semantics: Differentiates category-level majority agreement (>= 2 annotators)
  from overall clip-level stutter presence consensus.
- Safe Missing/Malformed Value Handling: Coerces invalid, float, empty string, or None values gracefully.
"""

import csv
from typing import Dict, Any, Iterable, Union, List, Tuple


def _safe_int(val: Any, default: int = 0) -> int:
    """
    Safely converts an annotation value to integer, handling None,
    empty strings, float/numeric, and non-numeric inputs.
    """
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    try:
        s = str(val).strip()
        if not s:
            return default
        try:
            return int(s)
        except ValueError:
            return int(float(s))
    except (ValueError, TypeError):
        return default


def derive_fluencybank_ground_truth(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derives clip-level ground truth from a 3-annotator FluencyBank / SEP-28k record.

    Consensus Rules:
    1. Category-level majority: Any dysfluency category with >= 2 annotators is flagged as positive.
    2. Split-vote stutter consensus: If sum of stutter category counts >= 2 AND NoStutteredWords <= 1,
       the clip is flagged as Stutter (majority of annotators heard stuttering, but split on subtype).
    3. Explicit fluent majority: If NoStutteredWords >= 2, the clip is Clean (or Filler if Interjection >= 2).
    4. Explicit filler majority: If Interjection >= 2 and NoStutteredWords <= 1, the clip is Filler.
    5. Ambiguous / Conflicting evidence: If neither stuttering nor fluency reaches majority consensus,
       the clip is marked 'Ambiguous' (is_ambiguous=True) so it can be safely excluded from benchmarks.

    Established Branch Priority for Contradictions:
    - If an annotator row contains contradictory majority votes (e.g. NoStutteredWords >= 2 AND
      a stutter category >= 2), the positive stutter category takes precedence (clinical safety rule).
    """
    s_rep = 1 if _safe_int(row.get("SoundRep")) >= 2 else 0
    w_rep = 1 if _safe_int(row.get("WordRep")) >= 2 else 0
    prol = 1 if _safe_int(row.get("Prolongation")) >= 2 else 0
    blk = 1 if _safe_int(row.get("Block")) >= 2 else 0
    fil = 1 if _safe_int(row.get("Interjection")) >= 2 else 0
    cln_explicit = 1 if _safe_int(row.get("NoStutteredWords")) >= 2 else 0

    raw_s = _safe_int(row.get("SoundRep"))
    raw_w = _safe_int(row.get("WordRep"))
    raw_p = _safe_int(row.get("Prolongation"))
    raw_b = _safe_int(row.get("Block"))
    raw_cln = _safe_int(row.get("NoStutteredWords"))
    sum_stutter_votes = raw_s + raw_w + raw_p + raw_b

    # 1. Any individual stutter category has majority (>= 2 annotators)
    if (s_rep or w_rep or prol or blk):
        has_stutter = 1
        dom = "Stutter"
        status = "ACCEPTED"
    # 2. Stutter votes split across categories, with majority rejecting fluency (NoStutteredWords <= 1)
    elif sum_stutter_votes >= 2 and raw_cln <= 1:
        has_stutter = 1
        dom = "Stutter"
        status = "ACCEPTED_SPLIT_STUTTER"
    # 3. Explicit fluent majority (NoStutteredWords >= 2)
    elif cln_explicit:
        has_stutter = 0
        status = "ACCEPTED_CLEAN"
        if fil:
            dom = "Filler"
        else:
            dom = "Clean"
    # 4. Explicit filler majority with non-fluent speech
    elif fil and raw_cln <= 1:
        has_stutter = 0
        dom = "Filler"
        status = "ACCEPTED_FILLER"
    # 5. Ambiguous / Conflicting evidence (no consensus for stutter or clean)
    else:
        has_stutter = None
        dom = "Ambiguous"
        status = "EXCLUDED_AMBIGUOUS"

    return {
        "status": status,
        "is_ambiguous": (dom == "Ambiguous"),
        "gt_sound_rep": s_rep,
        "gt_word_rep": w_rep,
        "gt_prolongation": prol,
        "gt_block": blk,
        "gt_filler": fil,
        "gt_clean": 1 if dom == "Clean" else 0,
        "gt_has_stutter": has_stutter,
        "gt_has_filler": fil,
        "gt_dominant": dom,
    }


def derive_libri_ground_truth(token_rows: Iterable[Union[str, List[Any], Tuple[Any, ...]]]) -> Dict[str, Any]:
    """
    Derives clip-level ground truth and reference transcript from Libri_Stutter token annotations.

    Libri_Stutter Annotation Schema:
    - Column 0: Word text or 'STUTTER' marker
    - Column 1: Start timestamp (seconds)
    - Column 2: End timestamp (seconds)
    - Column 3: Dysfluency code:
        '0': Clean word
        '1': Interjection (Filler)
        '2': Sound repetition (Stutter)
        '3': Word repetition (Stutter)
        '4': Phrase repetition (Stutter)
        '5': Prolongation (Stutter)
        Other/corrupted codes: safely ignored (treated as non-stuttering)

    Note on Blocks:
    - Blocks were never synthesized in Libri_Stutter. gt_block is strictly 0.
    """
    ref_words: List[str] = []
    s_rep = 0
    w_rep = 0
    p_rep = 0
    prol = 0
    fil = 0
    unknown_codes: List[str] = []

    for item in token_rows:
        if isinstance(item, str):
            s = item.strip()
            if not s:
                continue
            c_parts = next(csv.reader([s]))
        else:
            c_parts = item
        if len(c_parts) >= 4:
            word = str(c_parts[0]).strip()
            code = str(c_parts[3]).strip()
            if word != "STUTTER":
                ref_words.append(word)
            if code == "0":
                pass
            elif code == "1":
                fil = 1
            elif code == "2":
                s_rep = 1
            elif code == "3":
                w_rep = 1
            elif code == "4":
                p_rep = 1
            elif code == "5":
                prol = 1
            else:
                unknown_codes.append(code)

    has_stutter = 1 if (s_rep or w_rep or p_rep or prol) else 0
    has_clean = 1 if (has_stutter == 0 and fil == 0) else 0
    dom = "Stutter" if has_stutter else ("Filler" if fil else "Clean")

    return {
        "reference_text": " ".join(ref_words),
        "gt_sound_rep": s_rep,
        "gt_word_rep": w_rep,
        "gt_phrase_rep": p_rep,
        "gt_prolongation": prol,
        "gt_block": 0,  # Block ground truth is not annotated in Libri_Stutter
        "gt_filler": fil,
        "gt_clean": has_clean,
        "gt_has_stutter": has_stutter,
        "gt_has_filler": fil,
        "gt_dominant": dom,
        "unknown_codes": unknown_codes,
        "unknown_code_count": len(unknown_codes),
    }
