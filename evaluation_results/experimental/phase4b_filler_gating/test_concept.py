import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Words considered conversational fillers
FILLER_WORDS = {"um", "uh", "er", "erm", "ah", "hmm", "mm", "umm", "uhh", "err", "oh"}

def clean_tok(text: str) -> str:
    return re.sub(r'[^\w]', '', text).lower()

def is_filler_token(text: str) -> bool:
    c = clean_tok(text)
    if not c:
        return False
    if c in FILLER_WORDS:
        return True
    if re.match(r'^u+h+m*$', c) or re.match(r'^u+m+$', c) or re.match(r'^e+r+m*$', c) or re.match(r'^a+h+$', c) or re.match(r'^o+h+$', c):
        return True
    return False

def analyze_filler_streaks(words_list: List[Dict[str, Any]], transcript: str) -> List[Dict[str, Any]]:
    """
    Analyzes consecutive filler repetitions in words_list and transcript.
    Extracts streak count, inter-syllable interval (ISI), pacing, and calibrated confidence.
    """
    # Prefer words_list with timestamps if available, otherwise tokenize transcript
    if words_list and len(words_list) > 0:
        tokens_info = []
        for w in words_list:
            w_str = w.get("word", "")
            c = clean_tok(w_str)
            if c:
                tokens_info.append({
                    "word": c,
                    "raw": w_str,
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                    "is_filler": is_filler_token(w_str)
                })
    else:
        tokens_info = [{
            "word": clean_tok(w),
            "raw": w,
            "start": 0.0,
            "end": 0.0,
            "is_filler": is_filler_token(w)
        } for w in transcript.split() if clean_tok(w)]

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

    results = []
    for s in streaks:
        count = len(s)
        filler_word = s[0]["word"]
        
        # Calculate pacing / inter-syllable intervals if timestamps exist
        durations = [t["end"] - t["start"] for t in s if t["end"] > t["start"]]
        mean_dur = sum(durations) / len(durations) if durations else 0.0
        
        has_timestamps = any(t["end"] > 0.0 for t in s)
        total_time = s[-1]["end"] - s[0]["start"] if has_timestamps else 0.0
        mean_isi = total_time / (count - 1) if has_timestamps and count > 1 else 0.0

        # Pacing classification:
        # Rapid struggle repetition: mean_isi <= 0.35s or total_time very compressed
        # Leisurely conversational hesitation: mean_isi > 0.35s
        is_rapid = (mean_isi <= 0.35 and has_timestamps) or (count >= 5)

        # Calibrated Confidence
        if count < 3:
            # 1-2 repetitions: purely conversational filler
            conf = 0.30
            classification = "CONVERSATIONAL_FILLER"
            is_stutter_candidate = False
        elif count == 3:
            if is_rapid:
                conf = 0.60  # Meets confident threshold
                classification = "RAPID_FILLER_STRUGGLE"
                is_stutter_candidate = True
            else:
                conf = 0.50  # Borderline uncertain; requires corroboration
                classification = "LEISURELY_FILLER_HESITATION"
                is_stutter_candidate = False
        elif count == 4:
            conf = 0.65 if is_rapid else 0.55
            classification = "REPEATED_FILLER_STRUGGLE"
            is_stutter_candidate = True if is_rapid else False
        else: # count >= 5
            # Severe motor struggle / block chain
            conf = round(min(0.85, 0.70 + (count - 5) * 0.03), 2)
            classification = "SEVERE_FILLER_STRUGGLE_CHAIN"
            is_stutter_candidate = True

        results.append({
            "filler_word": filler_word,
            "count": count,
            "has_timestamps": has_timestamps,
            "mean_isi_sec": round(mean_isi, 3) if has_timestamps else None,
            "total_span_sec": round(total_time, 2) if has_timestamps else None,
            "is_rapid": is_rapid,
            "confidence": conf,
            "classification": classification,
            "is_stutter_candidate": is_stutter_candidate,
            "evidence": f"{count}x '{filler_word}' (span: {total_time:.2f}s, mean_isi: {mean_isi:.2f}s, rapid: {is_rapid})"
        })

    return results

# Test on dev target cases
test_clips = [
    ('M_0030_16y4m_1_dysfluent_015.wav', 'er, er, er, er, er, er, er. Once, oh quick.', [
        {'word': 'er,', 'start': 0.00, 'end': 0.08},
        {'word': 'er,', 'start': 0.62, 'end': 0.86},
        {'word': 'er,', 'start': 0.96, 'end': 0.96},
        {'word': 'er,', 'start': 1.24, 'end': 1.26},
        {'word': 'er,', 'start': 1.28, 'end': 1.30},
        {'word': 'er,', 'start': 1.32, 'end': 1.34},
        {'word': 'er.', 'start': 1.34, 'end': 1.34},
        {'word': 'Once,', 'start': 1.34, 'end': 1.68},
        {'word': 'oh', 'start': 2.10, 'end': 2.56},
        {'word': 'quick.', 'start': 2.56, 'end': 2.84}
    ]),
    ('FluencyBank_017_4.wav', 'And work, uh, environment, uh, uh.', [
        {'word': 'And', 'start': 0.0, 'end': 0.0},
        {'word': 'work,', 'start': 0.0, 'end': 0.28},
        {'word': 'uh,', 'start': 0.98, 'end': 1.18},
        {'word': 'environment,', 'start': 1.86, 'end': 1.86},
        {'word': 'uh,', 'start': 2.36, 'end': 2.56},
        {'word': 'uh.', 'start': 2.92, 'end': 2.98}
    ]),
    ('M_0030_16y4m_1_dysfluent_009.wav', 'You could, um, um, uh, uh, uh, uh, uh', [
        {'word': 'um', 'start': 1.04, 'end': 1.88},
        {'word': 'um', 'start': 2.28, 'end': 2.98},
        {'word': 'uh', 'start': 2.98, 'end': 2.98},
        {'word': 'uh', 'start': 2.98, 'end': 2.98},
        {'word': 'uh', 'start': 2.98, 'end': 2.98},
        {'word': 'uh', 'start': 2.98, 'end': 2.98},
        {'word': 'uh', 'start': 2.98, 'end': 2.98}
    ]),
    ('Fluent_Hesitation_Sample', 'Well, uh, uh, uh, let me see', [
        {'word': 'Well,', 'start': 0.0, 'end': 0.30},
        {'word': 'uh,', 'start': 0.80, 'end': 1.10},
        {'word': 'uh,', 'start': 1.60, 'end': 1.90},
        {'word': 'uh,', 'start': 2.40, 'end': 2.70},
        {'word': 'let', 'start': 3.20, 'end': 3.40}
    ])
]

print("=== TEST CASES ===")
for name, tx, wlist in test_clips:
    res = analyze_filler_streaks(wlist, tx)
    print(f"Clip: {name}")
    print(f"  Transcript: '{tx}'")
    for r in res:
        print(f"  Streak: {r['evidence']}")
        print(f"    Confidence: {r['confidence']}, StutterCandidate: {r['is_stutter_candidate']}, Classification: {r['classification']}")
    print("-" * 60)
