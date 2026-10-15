#!/usr/bin/env python3
"""Fit model/params.json from a labelled SN21 corpus (stdlib only, deterministic).

    python3 tracks/main/train.py --data SN21_training_v4.jsonl.gz [--holdout 0.2] [--out PATH]

Per horizon it fits, on the training part only (the latest --holdout fraction
by time is left for evaluate.py):
1. a budget elasticity per metric: the shrunk median of delta / budget change
   over budget episodes;
2. empirical p10/p50/p90 of the residual (delta - elasticity x budget change)
   for the global pool, each change family and each transition key, shrunk
   toward the parent pool by PRIOR_MASS; efficiency is split by goal basis;
3. a width scale for the p10-p90 band, picked on the latest 20% of the
   training part by the upstream v2 entry score.
Use --holdout 0 for the image you ship, after you have read evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

TRACK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRACK_DIR)

import corpus  # noqa: E402
from model import predict as model  # noqa: E402
from vendor.settle_v2 import score_entry_v2  # noqa: E402

METRICS = model.METRICS
QUANTILES = (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))
HORIZON_WEIGHTS = {7: 0.20, 14: 0.35, 28: 0.45}
PRIOR_MASS = 40.0
MIN_FAMILY = 25
MIN_KEY = 60
MIN_BUDGET = 30
ELASTICITY_PRIOR = 30.0
MIN_BUDGET_SIZE = 0.05
WIDTH_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5)


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already sorted list."""
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def triple(values: list[float]) -> dict:
    s = sorted(values)
    return {k: quantile(s, q) for k, q in QUANTILES}


def shrink(own: dict, n: int, parent: dict) -> dict:
    w = n / (n + PRIOR_MASS)
    return {k: round(w * own[k] + (1 - w) * parent[k], 6) for k, _ in QUANTILES}


def median(values: list[float]) -> float:
    return quantile(sorted(values), 0.5)


def fit_elasticity(rows: list[tuple]) -> dict:
    out = {}
    for metric in METRICS:
        ratios = [lab[metric] / f["budget_pct"] for f, lab in rows
                  if f["budget_pct"] is not None and abs(f["budget_pct"]) >= MIN_BUDGET_SIZE]
        if len(ratios) < MIN_BUDGET:
            out[metric] = 0.0
            continue
        ratios = [max(-1.5, min(r, 1.5)) for r in ratios]
        out[metric] = round(median(ratios) * len(ratios) / (len(ratios) + ELASTICITY_PRIOR), 6)
    return out


def fit_horizon(rows: list[tuple], elasticity: dict) -> dict:
    """{pool name: {metric: triple}} for one horizon."""
    def residual(f, lab, metric):
        lean = elasticity.get(metric, 0.0) * f["budget_pct"] if f["budget_pct"] is not None else 0.0
        return lab[metric] - lean

    table: dict = {}
    pools = {"global": rows}
    for f, lab in rows:
        pools.setdefault("family:" + f["family"], []).append((f, lab))
        pools.setdefault("key:" + f["key"], []).append((f, lab))
    glob = {}
    for metric in METRICS:
        glob[metric] = triple([residual(f, lab, metric) for f, lab in rows])
    for basis in model.BASES:
        sub = [residual(f, lab, "efficiency_delta_pct") for f, lab in rows if f["basis"] == basis]
        if sub:
            table["global|" + basis] = {"efficiency_delta_pct": shrink(triple(sub), len(sub), glob["efficiency_delta_pct"])}
    table["global"] = {m: {k: round(v, 6) for k, v in t.items()} for m, t in glob.items()}
    for name in sorted(pools):
        members = pools[name]
        if name == "global":
            continue
        minimum = MIN_FAMILY if name.startswith("family:") else MIN_KEY
        if len(members) < minimum:
            continue
        parent_name = "global"
        if name.startswith("key:"):
            fam = "family:" + model.family_of(name[4:])
            parent_name = fam if fam in table else "global"
        entry = {}
        for metric in ("cost_delta_pct", "conversions_delta_pct"):
            entry[metric] = shrink(triple([residual(f, lab, metric) for f, lab in members]),
                                   len(members), table[parent_name][metric])
        entry["efficiency_delta_pct"] = shrink(
            triple([residual(f, lab, "efficiency_delta_pct") for f, lab in members]),
            len(members), table[parent_name]["efficiency_delta_pct"])
        table[name] = entry
        for basis in model.BASES:
            sub = [residual(f, lab, "efficiency_delta_pct") for f, lab in members if f["basis"] == basis]
            parent = (table.get(parent_name + "|" + basis) or table[parent_name])["efficiency_delta_pct"]
            if len(sub) >= minimum:
                table[name + "|" + basis] = {"efficiency_delta_pct": shrink(triple(sub), len(sub), parent)}
    return table


def fit(records: list[dict], width_scale: float = 1.0) -> dict:
    by_h: dict = {}
    for r in records:
        f = model.features(r["episode"])
        for h, lab in r["labels"].items():
            by_h.setdefault(h, []).append((f, lab))
    if not by_h:
        raise SystemExit("no labelled records to fit")
    quantiles, elasticity = {}, {}
    for h in sorted(by_h):
        elasticity[str(h)] = fit_elasticity(by_h[h])
        quantiles[str(h)] = fit_horizon(by_h[h], elasticity[str(h)])
    mid = str(14 if 14 in by_h else sorted(by_h)[0])
    fallback = {m: dict(quantiles[mid]["global"][m]) for m in METRICS}
    return {"version": 1, "width_scale": width_scale, "lean_cap": 2.0,
            "budget_elasticity": elasticity, "quantiles": quantiles, "fallback": fallback,
            "trained_on": {"records": len(records),
                           "entries": {str(h): len(v) for h, v in sorted(by_h.items())}}}


def score(params: dict, records: list[dict]) -> float:
    total = weight = 0.0
    for r in records:
        pred = model.predict(r["episode"], params)
        for h, lab in r["labels"].items():
            w = HORIZON_WEIGHTS.get(h, 0.0)
            total += w * score_entry_v2(pred.get(str(h)) or {}, lab)
            weight += w
    return total / weight if weight else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.environ.get("SN21_TRAINING_DATA") or corpus.FIXTURE)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--out", default=model.PARAMS_PATH)
    args = parser.parse_args()
    train, _held = corpus.split(corpus.load(args.data), args.holdout)
    inner, tune = corpus.split(train, 0.2)
    best = 1.0
    if len(inner) >= 50 and tune:
        probe = fit(inner)
        scores = {w: score(dict(probe, width_scale=w), tune) for w in WIDTH_GRID}
        best = max(WIDTH_GRID, key=lambda w: (round(scores[w], 6), -abs(w - 1.0)))
    params = fit(train, best)
    params["trained_on"]["holdout"] = args.holdout
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"out": args.out, "records": len(train), "width_scale": best,
                      "in_sample_v2": round(score(params, train), 6)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
