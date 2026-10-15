"""train.py and evaluate.py run end to end on a v4-shaped corpus and on the bundled fixture."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from tests import helpers


def run(script: str, *args: str) -> dict:
    proc = subprocess.run([sys.executable, os.path.join(helpers.TRACK_DIR, script), *args],
                          capture_output=True, text=True, timeout=600, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TrainEvaluateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = os.path.join(cls.tmp.name, "synthetic_v4.jsonl.gz")
        helpers.write_synthetic_v4(cls.data)
        cls.params = os.path.join(cls.tmp.name, "params.json")
        cls.trained = run("train.py", "--data", cls.data, "--out", cls.params)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_training_learns_the_budget_elasticity(self):
        with open(self.params) as handle:
            params = json.load(handle)
        beta = params["budget_elasticity"]["7"]["cost_delta_pct"]
        self.assertGreater(beta, 0.3)
        self.assertIn("family:NEGATIVE_KEYWORD_ADD", params["quantiles"]["14"])

    def test_trained_params_beat_the_untrained_defaults_on_held_out_data(self):
        trained = run("evaluate.py", "--data", self.data, "--params", self.params)
        untrained = run("evaluate.py", "--data", self.data)
        self.assertGreater(trained["score"], untrained["score"])
        self.assertTrue(trained["metrics"]["deterministic"])
        self.assertEqual(trained["metrics"]["coverage"], 1.0)

    def test_evaluate_on_fixture_prints_score_json_last(self):
        result = run("evaluate.py")
        self.assertIsInstance(result["score"], float)
        self.assertTrue(0.0 <= result["score"] <= 1.0)
        for key in ("gate_score", "baseline_gate_score", "coverage", "deterministic", "reference_v2"):
            self.assertIn(key, result["metrics"])


if __name__ == "__main__":
    unittest.main()
