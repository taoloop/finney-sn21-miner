"""The container contract from upstream docs/MINER_MODEL_SPEC.md, checked on the real entrypoint."""
import json
import os
import sys
import unittest

from tests import helpers

sys.path.insert(0, helpers.TRACK_DIR)
from model import predict as model  # noqa: E402

METRICS = ("cost_delta_pct", "conversions_delta_pct", "efficiency_delta_pct")


def parse(stdout: str) -> list[dict]:
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


class ContractTest(unittest.TestCase):
    def assert_block(self, block):
        self.assertIsInstance(block, dict)
        for metric in METRICS:
            t = block[metric]
            self.assertLessEqual(t["p10"], t["p50"])
            self.assertLessEqual(t["p50"], t["p90"])
        self.assertTrue(0 <= block["goal_miss_probability"] <= 1)
        self.assertTrue(0 <= block["instability_risk"] <= 1)

    def test_every_fixture_episode_is_answered_with_all_horizons(self):
        episodes = helpers.fixture_episodes()
        out = parse(helpers.run_model([json.dumps(e) for e in episodes]))
        self.assertEqual([o["episode_id"] for o in out], [e["episode_id"] for e in episodes])
        for o in out:
            self.assertEqual(sorted(o["horizons"], key=int), ["7", "14", "28"])
            for block in o["horizons"].values():
                self.assert_block(block)

    def test_output_is_byte_identical_across_hash_seeds(self):
        lines = [json.dumps(e) for e in helpers.fixture_episodes()]
        lines.append(json.dumps(helpers.live_episode("live-1")))
        self.assertEqual(helpers.run_model(lines, hashseed="1"), helpers.run_model(lines, hashseed="4242"))

    def test_malformed_lines_are_skipped_without_crashing(self):
        lines = ["not json", "[1, 2]", json.dumps({"no_id": True}), json.dumps(helpers.live_episode("ok"))]
        out = parse(helpers.run_model(lines))
        self.assertEqual([o["episode_id"] for o in out], ["ok"])

    def test_extra_requested_horizon_gets_a_block(self):
        ep = helpers.live_episode("h21", horizons=(7, 21))
        out = parse(helpers.run_model([json.dumps(ep)]))[0]
        self.assertEqual(sorted(out["horizons"], key=int), ["7", "14", "21", "28"])
        self.assert_block(out["horizons"]["21"])

    def test_broken_payload_still_gets_a_fallback_answer(self):
        ep = {"episode_id": "weird", "payload": {"action_bundle": {"actions": "not a list"},
                                                 "episode_metadata": {"outcome_horizons_days": [7]}}}
        out = parse(helpers.run_model([json.dumps(ep)]))
        self.assertEqual(out[0]["episode_id"], "weird")
        self.assert_block(out[0]["horizons"]["28"])

    def test_episode_id_is_read_from_every_known_layout(self):
        self.assertEqual(model.episode_id_of({"episode_id": "a"}), "a")
        self.assertEqual(model.episode_id_of({"input": {"episode_id": "b", "payload": {}}}), "b")
        self.assertEqual(model.episode_id_of({"episode_metadata": {"episode_id": "c"}}), "c")
        self.assertIsNone(model.episode_id_of({"payload": {}}))

    def test_budget_change_is_read_from_live_magnitudes_and_flat_fields(self):
        self.assertAlmostEqual(model.budget_change_pct(helpers.live_episode("x", prev=100, new=50)), -0.5)
        self.assertAlmostEqual(model.budget_change_pct({"action_type": "BUDGET_CHANGE",
                                                        "from_value": 100, "to_value": 120}), 0.2)
        self.assertIsNone(model.budget_change_pct(helpers.live_episode("y", key="NEGATIVE_KEYWORD_ADD")))

    def test_budget_increase_raises_cost_median(self):
        params = model.load_params()
        up = model.predict(helpers.live_episode("u", prev=100, new=150), params)
        flat = model.predict(helpers.live_episode("f", key="NEGATIVE_KEYWORD_ADD"), params)
        self.assertGreater(up["7"]["cost_delta_pct"]["p50"], flat["7"]["cost_delta_pct"]["p50"])

    def test_goal_basis_selects_the_efficiency_quantiles(self):
        params = {"width_scale": 1.0, "lean_cap": 2.0, "budget_elasticity": {},
                  "fallback": {m: {"p10": -1, "p50": 0, "p90": 1} for m in METRICS},
                  "quantiles": {"7": {"global": {m: {"p10": -1, "p50": 0, "p90": 1} for m in METRICS},
                                      "global|conversion_value": {"efficiency_delta_pct":
                                                                  {"p10": 0.1, "p50": 0.2, "p90": 0.3}}}}}
        value = model.predict(helpers.live_episode("v", key="X", basis="conversion_value"), params)
        cpa = model.predict(helpers.live_episode("c", key="X", basis="cpa"), params)
        self.assertEqual(value["28"]["efficiency_delta_pct"]["p50"], 0.2)
        self.assertEqual(cpa["28"]["efficiency_delta_pct"]["p50"], 0.0)

    def test_shipped_params_cover_every_horizon(self):
        params = model.load_params()
        self.assertEqual(sorted(params["quantiles"], key=int), ["7", "14", "28"])
        for metric in METRICS:
            self.assertIn(metric, params["fallback"])

    def test_model_has_no_third_party_imports(self):
        with open(helpers.ENTRYPOINT) as handle:
            imports = {line.split()[1].split(".")[0] for line in handle
                       if line.startswith(("import ", "from "))}
        self.assertLessEqual(imports, {"__future__", "json", "math", "os", "sys"})


class RepositoryLimitsTest(unittest.TestCase):
    def test_committed_fixture_stays_under_the_per_file_limit(self):
        self.assertLess(os.path.getsize(helpers.FIXTURE), 256 * 1024)


if __name__ == "__main__":
    unittest.main()
