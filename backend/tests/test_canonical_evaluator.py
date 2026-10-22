"""
Unit Tests for Canonical Benchmark Evaluator
===========================================

Verifies:
- 30-clip manifest integrity and class balance.
- Canonical metric calculation formulas (accuracy, macro/weighted F1, binary stutter).
- Separate tracking of timeouts and strict N=30 denominator preservation.
- Reproduction of baseline metrics from saved prediction records.
"""

import unittest
from pathlib import Path
import os
import sys

project_root = Path(__file__).resolve().parent.parent.parent
audit_canonical_dir = project_root / "evaluation_results" / "experimental" / "audit" / "canonical_eval"
if str(audit_canonical_dir) not in sys.path:
    sys.path.insert(0, str(audit_canonical_dir))

from backend.experimental.canonical_evaluator import CanonicalEvaluator


class TestCanonicalEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = CanonicalEvaluator(str(project_root))

    def test_manifest_composition(self):
        """Manifest must contain exactly 30 clips with 24 Stutter, 5 Filler, 1 Clean."""
        self.assertEqual(len(self.evaluator.manifest_clips), 30)
        stutter_count = sum(1 for c, v in self.evaluator.gt_has_stutter.items() if v == 1)
        filler_count = sum(1 for c, v in self.evaluator.gt_dominant.items() if v == "Filler")
        clean_count = sum(1 for c, v in self.evaluator.gt_dominant.items() if v == "Clean")
        self.assertEqual(stutter_count, 24)
        self.assertEqual(filler_count, 5)
        self.assertEqual(clean_count, 1)

    def test_synthetic_metrics_calculation(self):
        """Mathematical formulas must yield exact precision, recall, macro F1, and accuracy."""
        # 30 synthetic records: 20 Stutter, 5 Filler, 5 Clean
        records = [
            {"filename": f"clip_{i}", "dataset": "Test", "gt_dominant": "Stutter", "pred_dominant": "Stutter" if i < 15 else "Clean",
             "gt_has_stutter": 1, "pred_has_stutter": 1 if i < 15 else 0, "gt_has_filler": 0, "pred_has_filler": 0, "confidence": 0.8, "is_timeout": False}
            for i in range(20)
        ] + [
            {"filename": f"filler_{i}", "dataset": "Test", "gt_dominant": "Filler", "pred_dominant": "Filler" if i < 4 else "Clean",
             "gt_has_stutter": 0, "pred_has_stutter": 0, "gt_has_filler": 1, "pred_has_filler": 1 if i < 4 else 0, "confidence": 0.7, "is_timeout": False}
            for i in range(5)
        ] + [
            {"filename": f"clean_{i}", "dataset": "Test", "gt_dominant": "Clean", "pred_dominant": "Clean",
             "gt_has_stutter": 0, "pred_has_stutter": 0, "gt_has_filler": 0, "pred_has_filler": 0, "confidence": 0.0, "is_timeout": False}
            for i in range(5)
        ]

        metrics = self.evaluator._compute_metrics_block(records)
        # Total correct: 15 (stutter) + 4 (filler) + 5 (clean) = 24 / 30 = 0.8000
        self.assertEqual(metrics["accuracy"], 0.8000)
        self.assertEqual(metrics["stutter_binary"]["tp"], 15)
        self.assertEqual(metrics["stutter_binary"]["fn"], 5)
        self.assertEqual(metrics["stutter_binary"]["fp"], 0)
        self.assertEqual(metrics["stutter_binary"]["tn"], 10)
        self.assertEqual(metrics["stutter_binary"]["precision"], 1.0)
        self.assertEqual(metrics["stutter_binary"]["recall"], 0.75)

    def test_strict_denominator_preservation_with_timeouts(self):
        """Timeouts must remain in the N=30 denominator and not distort class counts."""
        raw_manifest = self.evaluator.raw_manifest
        res = self.evaluator.evaluate_prediction_records(
            "Test Baseline", raw_manifest, "baseline", timeout_clips=["M_0030_16y4m_1_dysfluent_009.wav"]
        )
        self.assertEqual(res["total_manifest_clips"], 30)
        self.assertEqual(res["evaluated_clips"], 30)
        self.assertIn("M_0030_16y4m_1_dysfluent_009.wav", res["timeout_clips"])
        self.assertEqual(len(res["missing_predictions"]), 0)
        self.assertEqual(len(res["duplicate_predictions"]), 0)

    def test_baseline_exact_reproduction(self):
        """Fix 1 baseline saved predictions must evaluate to exactly 60% accuracy, 17 TPs, 5 FPs."""
        raw_manifest = self.evaluator.raw_manifest
        res = self.evaluator.evaluate_prediction_records("Fix 1 Baseline", raw_manifest, "baseline")
        agg = res["aggregate_metrics"]
        self.assertEqual(agg["accuracy"], 0.6000)
        self.assertEqual(agg["macro_f1"], 0.3416)
        self.assertEqual(agg["stutter_binary"]["tp"], 17)
        self.assertEqual(agg["stutter_binary"]["fp"], 5)
        self.assertEqual(agg["stutter_binary"]["fn"], 7)
        self.assertEqual(agg["stutter_binary"]["tn"], 1)


if __name__ == "__main__":
    unittest.main()
