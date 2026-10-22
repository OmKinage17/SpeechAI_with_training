"""
Isolated Deterministic Unit Test: Ground-Truth Mapping Rules
============================================================
Scope: Validates shared ground-truth mapping helpers imported directly from
scratch.ground_truth_mapping (and re-exported by scratch.build_phase5_manifest).

STRICT CONSTRAINTS:
- Tests import the actual shared implementation functions.
- No duplicate test-only mapping functions.
- 100% in-memory synthetic inputs only.
- No audio files, manifests, holdouts, or benchmarks accessed or executed.
"""

import os
import sys
import unittest

# Ensure scratch directory is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from scratch.ground_truth_mapping import (
        derive_fluencybank_ground_truth,
        derive_libri_ground_truth,
    )
except ImportError:
    from ground_truth_mapping import (
        derive_fluencybank_ground_truth,
        derive_libri_ground_truth,
    )


class TestGroundTruthMapping(unittest.TestCase):
    """Deterministic validation of actual ground-truth mapping rules on synthetic data."""

    # -------------------------------------------------------------
    # 1. Libri_Stutter Tests (Retained Original Cases)
    # -------------------------------------------------------------
    def test_libri_phrase_rep_only(self):
        """Test 1: A PhraseRep-only clip (Code 4) produces gt_has_stutter = 1 and dom = Stutter."""
        tokens = [
            ("the", 0.0, 0.3, "0"),
            ("STUTTER", 0.3, 1.2, "4"),  # Phrase repetition
            ("morning", 1.2, 1.8, "0"),
        ]
        res = derive_libri_ground_truth(tokens)
        self.assertEqual(res["gt_has_stutter"], 1, "PhraseRep-only must be flagged as stutter")
        self.assertEqual(res["gt_dominant"], "Stutter", "PhraseRep-only must have dominant Stutter")
        self.assertEqual(res["gt_phrase_rep"], 1, "gt_phrase_rep must be 1")
        self.assertEqual(res["gt_sound_rep"], 0, "gt_sound_rep must remain 0")
        self.assertEqual(res["gt_clean"], 0, "gt_clean must be 0")
        self.assertEqual(res["gt_block"], 0, "gt_block must strictly be 0 (unannotated in Libri)")
        self.assertEqual(res["reference_text"], "the morning", "Reference text must omit STUTTER token")

    def test_libri_mixed_labels(self):
        """Test 2: A mixed clip (Interjection Code 1 + PhraseRep Code 4) preserves all flags."""
        tokens = [
            ("STUTTER", 0.0, 0.5, "1"),  # Interjection
            ("he", 0.5, 0.8, "0"),
            ("STUTTER", 0.8, 1.6, "4"),  # Phrase repetition
            ("went", 1.6, 2.0, "0"),
        ]
        res = derive_libri_ground_truth(tokens)
        self.assertEqual(res["gt_has_stutter"], 1)
        self.assertEqual(res["gt_has_filler"], 1)
        self.assertEqual(res["gt_filler"], 1)
        self.assertEqual(res["gt_phrase_rep"], 1)
        self.assertEqual(res["gt_dominant"], "Stutter", "Stutter takes precedence in dominant hierarchy")

    def test_libri_unaffected_stutter_types(self):
        """Test 3: Existing unaffected codes (2: SoundRep, 3: WordRep, 5: Prolongation) retain behavior."""
        for code, key in [("2", "gt_sound_rep"), ("3", "gt_word_rep"), ("5", "gt_prolongation")]:
            tokens = [("STUTTER", 0.0, 0.5, code), ("word", 0.5, 1.0, "0")]
            res = derive_libri_ground_truth(tokens)
            self.assertEqual(res["gt_has_stutter"], 1)
            self.assertEqual(res[key], 1)
            self.assertEqual(res["gt_phrase_rep"], 0)
            self.assertEqual(res["gt_dominant"], "Stutter")

    def test_libri_clean_speech(self):
        """Test 4: Pure fluent speech (Code 0 only) produces clean ground truth."""
        tokens = [("this", 0.0, 0.3, "0"), ("is", 0.3, 0.5, "0"), ("clean", 0.5, 1.0, "0")]
        res = derive_libri_ground_truth(tokens)
        self.assertEqual(res["gt_has_stutter"], 0)
        self.assertEqual(res["gt_clean"], 1)
        self.assertEqual(res["gt_dominant"], "Clean")

    # -------------------------------------------------------------
    # 2. FluencyBank Multi-Annotator Consensus Tests (Retained Original Cases)
    # -------------------------------------------------------------
    def test_fluencybank_split_stutter_votes(self):
        """Test 5: Split stutter votes (1 SoundRep, 1 Prolongation, 1 Block; NoStutteredWords=0) is STUTTER, not Clean."""
        row = {
            "SoundRep": "1",
            "WordRep": "0",
            "Prolongation": "1",
            "Block": "1",
            "Interjection": "0",
            "NoStutteredWords": "0",
        }
        res = derive_fluencybank_ground_truth(row)
        self.assertEqual(res["gt_has_stutter"], 1, "Split stutter votes must be recognized as Stutter")
        self.assertEqual(res["gt_dominant"], "Stutter", "Dominant class must be Stutter")
        self.assertEqual(res["gt_clean"], 0, "Split stutter must NOT be labeled Clean")
        self.assertEqual(res["status"], "ACCEPTED_SPLIT_STUTTER")
        # Distinct concept check: individual category flags remain 0 because none achieved >= 2 majority
        self.assertEqual(res["gt_sound_rep"], 0)
        self.assertEqual(res["gt_prolongation"], 0)
        self.assertEqual(res["gt_block"], 0)

    def test_fluencybank_explicit_clean_supported(self):
        """Test 6: Supported fluent row (NoStutteredWords=3, 0 stutter) remains Clean."""
        row = {
            "SoundRep": "0",
            "WordRep": "0",
            "Prolongation": "0",
            "Block": "0",
            "Interjection": "0",
            "NoStutteredWords": "3",
        }
        res = derive_fluencybank_ground_truth(row)
        self.assertEqual(res["gt_has_stutter"], 0)
        self.assertEqual(res["gt_clean"], 1)
        self.assertEqual(res["gt_dominant"], "Clean")
        self.assertEqual(res["status"], "ACCEPTED_CLEAN")
        self.assertFalse(res["is_ambiguous"])

    def test_fluencybank_category_majority_stutter(self):
        """Test 7: Category majority stutter (SoundRep=2, NoStutteredWords=1) is Stutter."""
        row = {
            "SoundRep": "2",
            "WordRep": "0",
            "Prolongation": "0",
            "Block": "0",
            "Interjection": "0",
            "NoStutteredWords": "1",
        }
        res = derive_fluencybank_ground_truth(row)
        self.assertEqual(res["gt_has_stutter"], 1)
        self.assertEqual(res["gt_sound_rep"], 1)
        self.assertEqual(res["gt_dominant"], "Stutter")
        self.assertEqual(res["status"], "ACCEPTED")

    def test_fluencybank_explicit_filler(self):
        """Test 8: Supported filler row (Interjection=3, NoStutteredWords=3, 0 stutter) is Filler."""
        row = {
            "SoundRep": "0",
            "WordRep": "0",
            "Prolongation": "0",
            "Block": "0",
            "Interjection": "3",
            "NoStutteredWords": "3",
        }
        res = derive_fluencybank_ground_truth(row)
        self.assertEqual(res["gt_has_stutter"], 0)
        self.assertEqual(res["gt_has_filler"], 1)
        self.assertEqual(res["gt_dominant"], "Filler")
        self.assertEqual(res["status"], "ACCEPTED_CLEAN")

    def test_fluencybank_ambiguous_tie_excluded(self):
        """Test 9: Ambiguous/conflicting row (1 SoundRep, 1 Clean, 1 Unsure) is flagged Ambiguous and excluded."""
        row = {
            "SoundRep": "1",
            "WordRep": "0",
            "Prolongation": "0",
            "Block": "0",
            "Interjection": "0",
            "NoStutteredWords": "1",
            "Unsure": "1",
        }
        res = derive_fluencybank_ground_truth(row)
        self.assertIsNone(res["gt_has_stutter"], "Ambiguous row must not have forced binary stutter")
        self.assertEqual(res["gt_dominant"], "Ambiguous", "Dominant class must be Ambiguous")
        self.assertEqual(res["status"], "EXCLUDED_AMBIGUOUS", "Ambiguous clip must be excluded from candidate pool")
        self.assertTrue(res["is_ambiguous"])

    # -------------------------------------------------------------
    # 3. New Edge-Case Tests (Requirements 1 - 6)
    # -------------------------------------------------------------
    def test_fluencybank_conflicting_majority_stutter_precedence(self):
        """
        New Test 1: Conflicting majorities (NoStutteredWords >= 2 and a stutter category >= 2).
        Preserves established branch priority: positive stutter category takes precedence (clinical safety).
        """
        row = {
            "SoundRep": "2",
            "WordRep": "0",
            "Prolongation": "0",
            "Block": "0",
            "Interjection": "0",
            "NoStutteredWords": "2",
        }
        res = derive_fluencybank_ground_truth(row)
        self.assertEqual(
            res["gt_has_stutter"],
            1,
            "Established branch priority dictates positive stutter category takes precedence over NoStutteredWords >= 2",
        )
        self.assertEqual(res["gt_dominant"], "Stutter")
        self.assertEqual(res["gt_sound_rep"], 1)
        self.assertEqual(res["gt_clean"], 0, "Contradictory row with stutter majority must not be labeled clean")
        self.assertEqual(res["status"], "ACCEPTED")
        self.assertFalse(res["is_ambiguous"])

    def test_fluencybank_filler_majority_with_split_stutter_votes(self):
        """
        New Test 2: Filler majority (Interjection >= 2) combined with split stutter votes (sum >= 2, NoStutteredWords <= 1).
        Dominant class hierarchy specifies Stutter > Filler; category majority remains distinct.
        """
        row = {
            "SoundRep": "1",
            "WordRep": "0",
            "Prolongation": "0",
            "Block": "1",
            "Interjection": "2",
            "NoStutteredWords": "0",
        }
        res = derive_fluencybank_ground_truth(row)
        self.assertEqual(res["gt_has_stutter"], 1, "Split stutter votes must register stutter presence")
        self.assertEqual(res["gt_dominant"], "Stutter", "Stutter takes dominant precedence over Filler")
        self.assertEqual(res["gt_filler"], 1, "Interjection majority must be preserved on gt_filler")
        self.assertEqual(res["gt_has_filler"], 1, "Interjection majority must set gt_has_filler")
        self.assertEqual(res["gt_clean"], 0)
        self.assertEqual(res["status"], "ACCEPTED_SPLIT_STUTTER")
        # Verify category flags are NOT set when votes are split
        self.assertEqual(res["gt_sound_rep"], 0)
        self.assertEqual(res["gt_block"], 0)

    def test_libri_corrupted_or_unknown_code_six(self):
        """
        New Test 3: Unknown or corrupted Libri annotation code '6'.
        Must be safely ignored without raising exceptions, setting stutter flags, or corrupting reference text.
        """
        tokens = [
            ("hello", 0.0, 0.4, "0"),
            ("STUTTER", 0.4, 0.9, "6"),  # Corrupted / unknown code 6
            ("world", 0.9, 1.4, "0"),
        ]
        res = derive_libri_ground_truth(tokens)
        self.assertEqual(res["gt_has_stutter"], 0, "Unknown code 6 must not trigger stutter")
        self.assertEqual(res["gt_sound_rep"], 0)
        self.assertEqual(res["gt_word_rep"], 0)
        self.assertEqual(res["gt_phrase_rep"], 0)
        self.assertEqual(res["gt_prolongation"], 0)
        self.assertEqual(res["gt_block"], 0)
        self.assertEqual(res["gt_filler"], 0)
        self.assertEqual(res["gt_clean"], 1, "Clip with only clean words and code 6 marker must be Clean")
        self.assertEqual(res["gt_dominant"], "Clean")
        self.assertEqual(res["reference_text"], "hello world", "Reference text must omit STUTTER marker")

    def test_fluencybank_missing_and_malformed_values_safe_handling(self):
        """
        New Test 4: Missing (None, empty string), whitespace, or non-numeric annotation values.
        Production mapping must coerce values safely to 0 and exclude non-consensus rows as Ambiguous.
        """
        malformed_row = {
            "SoundRep": None,
            "WordRep": "   ",
            "Prolongation": "not_a_number",
            "Block": "",
            "Interjection": None,
            "NoStutteredWords": " ",
        }
        res = derive_fluencybank_ground_truth(malformed_row)
        self.assertTrue(res["is_ambiguous"], "Empty/malformed row must be flagged as ambiguous")
        self.assertEqual(res["status"], "EXCLUDED_AMBIGUOUS")
        self.assertIsNone(res["gt_has_stutter"], "Malformed row must have None for gt_has_stutter")
        self.assertEqual(res["gt_dominant"], "Ambiguous")
        self.assertEqual(res["gt_clean"], 0)
        self.assertEqual(res["gt_sound_rep"], 0)

        # Test partially malformed row with valid numeric string surrounded by whitespace
        partially_valid_row = {
            "SoundRep": "  2  ",
            "WordRep": "",
            "Prolongation": None,
            "Block": 0,
            "Interjection": 0,
            "NoStutteredWords": "0",
        }
        res2 = derive_fluencybank_ground_truth(partially_valid_row)
        self.assertEqual(res2["gt_has_stutter"], 1)
        self.assertEqual(res2["gt_sound_rep"], 1)
        self.assertEqual(res2["gt_dominant"], "Stutter")
        self.assertFalse(res2["is_ambiguous"])

    def test_libri_phrase_rep_only_and_mixed_cases(self):
        """
        New Test 5: Exhaustive PhraseRep-only and mixed PhraseRep clips.
        Verifies PhraseRep alone flags Stutter, and PhraseRep combined with Prolongation/Filler preserves all flags.
        """
        # 5A: Pure PhraseRep clip
        pure_prep = [
            ("the", 0.0, 0.2, "0"),
            ("STUTTER", 0.2, 0.8, "4"),
            ("end", 0.8, 1.2, "0"),
        ]
        res_pure = derive_libri_ground_truth(pure_prep)
        self.assertEqual(res_pure["gt_phrase_rep"], 1)
        self.assertEqual(res_pure["gt_sound_rep"], 0)
        self.assertEqual(res_pure["gt_word_rep"], 0)
        self.assertEqual(res_pure["gt_prolongation"], 0)
        self.assertEqual(res_pure["gt_filler"], 0)
        self.assertEqual(res_pure["gt_has_stutter"], 1)
        self.assertEqual(res_pure["gt_dominant"], "Stutter")
        self.assertEqual(res_pure["reference_text"], "the end")

        # 5B: Mixed PhraseRep + Prolongation + Interjection in CSV line string format
        csv_lines = [
            "STUTTER,0.0,0.4,1\n",   # Interjection
            "STUTTER,0.4,0.9,4\n",   # PhraseRep
            "STUTTER,0.9,1.5,5\n",   # Prolongation
            "morning,1.5,2.0,0\n",   # Clean word
        ]
        res_mixed = derive_libri_ground_truth(csv_lines)
        self.assertEqual(res_mixed["gt_phrase_rep"], 1)
        self.assertEqual(res_mixed["gt_prolongation"], 1)
        self.assertEqual(res_mixed["gt_filler"], 1)
        self.assertEqual(res_mixed["gt_has_filler"], 1)
        self.assertEqual(res_mixed["gt_has_stutter"], 1)
        self.assertEqual(res_mixed["gt_dominant"], "Stutter")
        self.assertEqual(res_mixed["reference_text"], "morning")

    def test_fluencybank_ambiguous_rows_explicitly_excluded_not_clean(self):
        """
        New Test 6: Ambiguous FluencyBank rows being explicitly excluded rather than silently labeled Clean.
        When annotator votes are split across stutter, filler, and clean without any consensus,
        is_ambiguous must be True and status must be EXCLUDED_AMBIGUOUS, never Clean.
        """
        split_no_consensus_row = {
            "SoundRep": "1",
            "Interjection": "1",
            "NoStutteredWords": "1",
            "WordRep": "0",
            "Prolongation": "0",
            "Block": "0",
        }
        res = derive_fluencybank_ground_truth(split_no_consensus_row)
        self.assertTrue(res["is_ambiguous"], "Row with no consensus must have is_ambiguous = True")
        self.assertEqual(res["status"], "EXCLUDED_AMBIGUOUS")
        self.assertEqual(res["gt_dominant"], "Ambiguous")
        self.assertEqual(res["gt_clean"], 0, "Non-consensus clip must NEVER be silently labeled Clean")
        self.assertIsNone(res["gt_has_stutter"], "Non-consensus clip must not have binary stutter forced")

    def test_libri_quoted_comma_in_word_parsing(self):
        """
        New Test 7: Handles words with internal punctuation/commas inside quotes safely.
        CSV parsing must not cause column-shifting or spurious code interpretation.
        """
        csv_lines = [
            '"Father,",1.4,2.1,0\n',
            'STUTTER,2.1,3.0,4\n',
            'spoke,3.0,3.5,0\n',
        ]
        res = derive_libri_ground_truth(csv_lines)
        self.assertEqual(res["gt_phrase_rep"], 1)
        self.assertEqual(res["gt_has_stutter"], 1)
        self.assertEqual(res["reference_text"], "Father, spoke")
        self.assertEqual(res["unknown_code_count"], 0)

    def test_libri_unknown_codes_tracked(self):
        """
        New Test 8: Unknown annotation codes are recorded with occurrences tracked.
        """
        tokens = [
            ("hello", 0.0, 0.4, "0"),
            ("STUTTER", 0.4, 0.9, "9"),
            ("world", 0.9, 1.4, "0"),
        ]
        res = derive_libri_ground_truth(tokens)
        self.assertEqual(res["unknown_code_count"], 1)
        self.assertEqual(res["unknown_codes"], ["9"])


if __name__ == "__main__":
    unittest.main()
