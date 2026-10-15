"""The vendored upstream scorer and gate behave as the subnet documents them."""
import sys
import unittest

from tests import helpers

sys.path.insert(0, helpers.TRACK_DIR)
from vendor import gate  # noqa: E402
from vendor.settle_v2 import score_entry_v2  # noqa: E402

ACTUAL = {"cost_delta_pct": 0.1, "conversions_delta_pct": -0.2, "efficiency_delta_pct": 0.3}


class ScoringTest(unittest.TestCase):
    def test_exact_point_prediction_scores_one(self):
        pred = {m: {"p10": a, "p50": a, "p90": a} for m, a in ACTUAL.items()}
        self.assertEqual(score_entry_v2(pred, ACTUAL), 1.0)

    def test_abstention_scores_zero(self):
        self.assertEqual(score_entry_v2({}, ACTUAL), 0.0)

    def test_wrong_goal_direction_costs_score(self):
        right = {m: {"p10": a - 0.1, "p50": a, "p90": a + 0.1} for m, a in ACTUAL.items()}
        wrong = dict(right, efficiency_delta_pct={"p10": -0.4, "p50": -0.3, "p90": -0.2})
        self.assertGreater(score_entry_v2(right, ACTUAL), score_entry_v2(wrong, ACTUAL))

    def test_baseline_is_not_admitted_against_itself(self):
        rows = [gate.OutcomeRow(f"e{i}", 7, 0.1 * i, -0.05 * i, 0.02 * i - 0.1) for i in range(1, 20)]
        base = gate.naive_baseline_prediction(gate.corpus_spread(rows))
        preds = {(o.episode_id, o.horizon_days): base for o in rows}
        result = gate.gate_score(rows, preds)
        self.assertFalse(gate.admission_verdict(result, result)["admitted"])


if __name__ == "__main__":
    unittest.main()
