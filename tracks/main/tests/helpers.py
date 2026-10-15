"""Shared helpers for the track's tests: paths, a subprocess runner, a synthetic v4 corpus."""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys

TRACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRYPOINT = os.path.join(TRACK_DIR, "model", "predict.py")
FIXTURE = os.path.join(TRACK_DIR, "data", "fixture_weekly.jsonl")


def run_model(lines: list[str], *args: str, hashseed: str | None = None) -> str:
    env = dict(os.environ)
    if hashseed is not None:
        env["PYTHONHASHSEED"] = hashseed
    proc = subprocess.run([sys.executable, ENTRYPOINT, *args], input="\n".join(lines) + "\n",
                          capture_output=True, text=True, env=env, timeout=300, check=True)
    return proc.stdout


def fixture_episodes() -> list[dict]:
    with open(FIXTURE) as handle:
        return [json.loads(line)["input"] for line in handle if line.strip()]


def live_episode(eid: str, key: str = "BUDGET:up", prev: int = 1_000_000_000, new: int = 1_200_000_000,
                 basis: str = "cpa", horizons=(7, 14, 28), when: str = "2026-07-01T00:00:00+00:00") -> dict:
    actions = []
    if key.startswith("BUDGET"):
        actions.append({"type": "BUDGET_CHANGE", "magnitude": {
            "previous_amount_micros": prev, "new_amount_micros": new}})
    else:
        actions.append({"type": key.split(":")[0]})
    return {"episode_id": eid, "transition_key": key, "system_estimate": None,
            "payload": {"episode_metadata": {"episode_id": eid, "outcome_horizons_days": list(horizons),
                                             "action_window_end": when},
                        "account_state": {"goal_basis": {"basis": basis, "source": "default",
                                                         "guarded": False}},
                        "action_bundle": {"bundle_summary": {"transition_key": key}, "actions": actions}}}


def write_synthetic_v4(path: str, n: int = 600) -> None:
    """A deterministic v4-shaped corpus: cost follows budget changes, NEGATIVE_KEYWORD_ADD trims cost."""
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"_manifest": {"records": n}}) + "\n")
        for i in range(n):
            noise = ((i * 7919) % 101 - 50) / 250.0
            day = 1 + i % 28
            if i % 3:
                pct = ((i * 37) % 81 - 40) / 100.0
                ep = live_episode(f"s{i:05d}", "BUDGET:up" if pct >= 0 else "BUDGET:down",
                                  1_000_000_000, int(1_000_000_000 * (1 + pct)),
                                  when=f"2026-07-{day:02d}T00:00:00+00:00")
                cost = 0.6 * pct + noise
            else:
                ep = live_episode(f"s{i:05d}", "NEGATIVE_KEYWORD_ADD", when=f"2026-07-{day:02d}T00:00:00+00:00")
                cost = -0.15 + noise
            labels = {h: {"cost_delta_pct": round(cost, 6),
                          "conversions_delta_pct": round(0.5 * cost - 0.05, 6),
                          "efficiency_delta_pct": round(0.5 * cost + 0.05, 6),
                          "goal_basis": "cpa", "finalized_on": "2026-09-01"}
                      for h in ("7", "14", "28")}
            handle.write(json.dumps({"episode_id": ep["episode_id"], "input": ep, "labels": labels}) + "\n")
