#!/usr/bin/env python3
"""Score the miner the way SN21 does, locally, through its real stdin/stdout entrypoint.

    python3 tracks/main/evaluate.py [--data PATH] [--holdout 0.2] [--limit 2000] [--params PATH]

Data: --data, else $SN21_TRAINING_DATA, else the bundled weekly-era fixture
(30 episodes, 7/14-day only; a smoke test, not a benchmark). On a training
corpus only the latest --holdout fraction by time is scored, the part
train.py leaves out. The fixture is scored whole.

What it does, mirroring upstream hope/backtest and hope/scoring:
1. runs model/predict.py twice over the same episodes and compares the
   output byte for byte (admission rejects `nondeterministic`);
2. parses lines as container_runner._parse_output does: a prediction counts
   only with a non-empty block for every required horizon;
3. scores each labelled (episode, horizon) with the vendored v2 entry
   formula; uncovered entries score 0.0 (the absence penalty), weighted by
   the high-resolution horizon blend 0.20 / 0.35 / 0.45;
4. computes the admission gate (vendored gate.py) against the persistence
   baseline with its 5% margin, and the same numbers for the upstream
   reference model for comparison.

Last line: {"score": <weighted mean v2 entry score>, "metrics": {...}}.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

TRACK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRACK_DIR)

import corpus  # noqa: E402
from vendor import gate  # noqa: E402
from vendor.settle_v2 import score_entry_v2  # noqa: E402

ENTRYPOINT = os.path.join(TRACK_DIR, "model", "predict.py")
REFERENCE = os.path.join(TRACK_DIR, "vendor", "reference_model.py")
HORIZON_WEIGHTS = {7: 0.20, 14: 0.35, 28: 0.45}
DEFAULT_HORIZONS = (7, 14)   # upstream container_runner fallback when metadata is silent
BASKET = 250                 # episodes per daily basket; the CPU budget is 15 min for one


def run(command: list[str], episodes: list[dict]) -> tuple[str, float]:
    blob = "".join(json.dumps(e, separators=(",", ":")) + "\n" for e in episodes)
    # Unset PYTHONHASHSEED so each run gets a fresh hash seed, as two sandbox runs would.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    t0 = time.perf_counter()
    proc = subprocess.run(command, input=blob.encode(), capture_output=True, env=env,
                          timeout=3600, check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"{command[-1]} exited {proc.returncode}: {proc.stderr[-500:]!r}")
    return proc.stdout.decode(errors="replace"), elapsed


def required(episode: dict) -> list[str]:
    payload = episode.get("payload") if isinstance(episode.get("payload"), dict) else episode
    meta = episode.get("episode_metadata") or payload.get("episode_metadata") or {}
    return [str(int(d)) for d in (meta.get("outcome_horizons_days") or DEFAULT_HORIZONS)]


def parse(stdout: str, records: list[dict]) -> dict:
    need = {r["episode_id"]: required(r["episode"]) for r in records}
    preds = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        eid = obj.get("episode_id") if isinstance(obj, dict) else None
        horizons = obj.get("horizons") if isinstance(obj, dict) else None
        if eid in need and isinstance(horizons, dict) and all(
                isinstance(horizons.get(h), dict) and horizons.get(h) for h in need[eid]):
            preds[eid] = horizons
    return preds


def monotone(preds: dict) -> bool:
    for horizons in preds.values():
        for block in horizons.values():
            for metric in gate.METRICS:
                t = (block or {}).get(metric)
                if t and not float(t["p10"]) <= float(t["p50"]) <= float(t["p90"]):
                    return False
    return True


def score(preds: dict, records: list[dict]) -> dict:
    total = weight = 0.0
    per_h: dict = {}
    for r in records:
        horizons = preds.get(r["episode_id"]) or {}
        for h, actual in r["labels"].items():
            s = score_entry_v2(horizons.get(str(h)) or {}, actual)
            w = HORIZON_WEIGHTS.get(h, 0.0)
            total += w * s
            weight += w
            agg = per_h.setdefault(str(h), [0.0, 0])
            agg[0] += s
            agg[1] += 1
    return {"v2": round(total / weight, 6) if weight else 0.0,
            "by_horizon": {h: round(a / n, 6) for h, (a, n) in sorted(per_h.items(), key=lambda kv: int(kv[0]))}}


def gate_verdict(preds: dict, records: list[dict]) -> dict:
    rows = [gate.OutcomeRow(r["episode_id"], h, a["cost_delta_pct"], a["conversions_delta_pct"],
                            a["efficiency_delta_pct"])
            for r in records for h, a in sorted(r["labels"].items())]
    model = {(r["episode_id"], h): preds[r["episode_id"]][str(h)]
             for r in records if r["episode_id"] in preds
             for h in r["labels"] if preds[r["episode_id"]].get(str(h))}
    base_pred = gate.naive_baseline_prediction(gate.corpus_spread(rows))
    baseline = {(o.episode_id, o.horizon_days): base_pred for o in rows}
    m = gate.gate_score(rows, model)
    b = gate.gate_score(rows, baseline)
    if m is None or b is None:
        return {"admitted": False, "model_gate_score": 0.0, "baseline_gate_score": b and b["gate_score"]}
    return gate.admission_verdict(m, b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.environ.get("SN21_TRAINING_DATA") or corpus.FIXTURE)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=2000, help="max held-out episodes (deterministic stride)")
    parser.add_argument("--params", default=None, help="params.json to evaluate (default: the shipped one)")
    args = parser.parse_args()

    records = corpus.load(args.data)
    if os.path.abspath(args.data) != corpus.FIXTURE:
        records = corpus.split(records, args.holdout)[1]
    records = corpus.stride(records, args.limit)
    if not records:
        raise SystemExit(f"no labelled records in {args.data}")
    episodes = [r["episode"] for r in records]

    command = [sys.executable, ENTRYPOINT] + (["--params", args.params] if args.params else [])
    out1, seconds = run(command, episodes)
    out2, _ = run(command, episodes)
    preds = parse(out1, records)
    ref_preds = parse(run([sys.executable, REFERENCE], episodes)[0], records)

    mine = score(preds, records)
    ref = score(ref_preds, records)
    verdict = gate_verdict(preds, records)
    ref_verdict = gate_verdict(ref_preds, records)
    metrics = {
        "data": os.path.basename(args.data),
        "episodes": len(records),
        "entries": sum(len(r["labels"]) for r in records),
        "coverage": round(len(preds) / len(records), 6),
        "deterministic": out1 == out2,
        "monotone": monotone(preds),
        "v2_by_horizon": mine["by_horizon"],
        "gate_score": verdict["model_gate_score"],
        "baseline_gate_score": verdict["baseline_gate_score"],
        "required_gate_score": verdict.get("required_gate_score"),
        "gate_admitted_estimate": bool(verdict["admitted"]) and out1 == out2,
        "reference_v2": ref["v2"],
        "reference_gate_score": ref_verdict["model_gate_score"],
        "v2_minus_reference": round(mine["v2"] - ref["v2"], 6),
        "seconds_per_basket": round(seconds * BASKET / len(records), 3),
    }
    print(json.dumps({"score": mine["v2"], "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
