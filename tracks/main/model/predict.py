#!/usr/bin/env python3
"""SN21 (AdTAO) miner model: one episode JSON per stdin line in, one prediction per stdout line out.

Contract (upstream docs/MINER_MODEL_SPEC.md): for every episode the line must
carry a non-empty block for every horizon in
episode_metadata.outcome_horizons_days, each with monotone p10/p50/p90 for
cost_delta_pct, conversions_delta_pct and efficiency_delta_pct. Anything less
is an abstention, and abstentions are charged as 0.0 in standing, so this
model answers every episode it can name and never crashes the basket.

Strategy: per-(change family, horizon) empirical quantiles fitted by
train.py and stored in params.json, with the efficiency quantiles split by
the episode's frozen goal_basis (cpa or conversion_value), plus a fitted
budget elasticity that shifts the cost and conversions quantiles by the size
of a budget change. Stdlib only, no clock, no randomness, no network.
"""
from __future__ import annotations

import json
import math
import os
import sys

METRICS = ("cost_delta_pct", "conversions_delta_pct", "efficiency_delta_pct")
HORIZONS = (7, 14, 28)
BASES = ("cpa", "conversion_value")
PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json")

# Weekly-era payloads carry no transition_key; map their action types to the
# family names live baskets use.
ACTION_FAMILY = {
    "BUDGET_CHANGE": "BUDGET",
}


def load_params(path: str = PARAMS_PATH) -> dict:
    with open(path) as handle:
        return json.load(handle)


def unwrap(episode: dict) -> dict:
    """The served episode, whether it arrives as served or wrapped in a training record."""
    inner = episode.get("input")
    if isinstance(inner, dict) and ("payload" in inner or "episode_id" in inner):
        return inner
    return episode


def payload_of(episode: dict) -> dict:
    """Payload fields: live baskets nest them under `payload`, weekly exports under `input`."""
    ep = unwrap(episode)
    for key in ("payload", "input"):
        if isinstance(ep.get(key), dict):
            return ep[key]
    return ep


def episode_id_of(episode: dict):
    for source in (episode, unwrap(episode), payload_of(episode)):
        if isinstance(source, dict) and source.get("episode_id") not in (None, ""):
            return source["episode_id"]
        meta = source.get("episode_metadata") if isinstance(source, dict) else None
        if isinstance(meta, dict) and meta.get("episode_id") not in (None, ""):
            return meta["episode_id"]
    return None


def metadata_of(episode: dict) -> dict:
    for source in (unwrap(episode), payload_of(episode)):
        meta = source.get("episode_metadata")
        if isinstance(meta, dict):
            return meta
    return {}


def required_horizons(episode: dict) -> list[int]:
    """HORIZONS plus any extra horizon the episode asks for, ascending."""
    wanted = set(HORIZONS)
    for day in metadata_of(episode).get("outcome_horizons_days") or ():
        try:
            wanted.add(int(day))
        except (TypeError, ValueError):
            continue
    return sorted(wanted)


def transition_key(episode: dict) -> str:
    ep = unwrap(episode)
    bundle = payload_of(episode).get("action_bundle") or {}
    key = (ep.get("transition_key")
           or (bundle.get("bundle_summary") or {}).get("transition_key"))
    if key:
        return str(key)
    types = sorted({str(a.get("type")) for a in bundle.get("actions") or [] if a.get("type")})
    if not types:
        return ""
    if len(types) == 1:
        return ACTION_FAMILY.get(types[0], types[0])
    return "COMPOSITE:" + ACTION_FAMILY.get(types[0], types[0]) + "+" + str(len(types))


def family_of(key: str) -> str:
    return key.split(":", 1)[0] if key else ""


def goal_basis(episode: dict) -> str:
    state = payload_of(episode).get("account_state") or {}
    basis = (state.get("goal_basis") or {}).get("basis")
    return basis if basis in BASES else "cpa"


def budget_change_pct(episode: dict):
    """Relative size of the episode's budget change (new/old - 1), or None.

    Reads live magnitudes (previous/new_amount_micros), training-v2
    from_value/to_value, and the flat weekly-era fields.
    """
    if episode.get("action_type") == "BUDGET_CHANGE":
        pairs = [(episode.get("from_value"), episode.get("to_value"))]
    else:
        pairs = []
        bundle = payload_of(episode).get("action_bundle") or {}
        for action in bundle.get("actions") or []:
            if action.get("type") != "BUDGET_CHANGE":
                continue
            mag = action.get("magnitude") or {}
            if mag.get("previous_amount_micros") is not None and mag.get("new_amount_micros") is not None:
                pairs.append((mag["previous_amount_micros"], mag["new_amount_micros"]))
            elif action.get("from_value") is not None and action.get("to_value") is not None:
                pairs.append((action["from_value"], action["to_value"]))
    before = after = 0.0
    for old, new in pairs:
        try:
            old, new = float(old), float(new)
        except (TypeError, ValueError):
            continue
        if math.isfinite(old) and math.isfinite(new) and old > 0 and new >= 0:
            before += old
            after += new
    if before <= 0:
        return None
    return max(-1.0, min(after / before - 1.0, 5.0))


def features(episode: dict) -> dict:
    key = transition_key(episode)
    return {
        "key": key,
        "family": family_of(key),
        "basis": goal_basis(episode),
        "budget_pct": budget_change_pct(episode),
    }


def nearest_horizon(horizon: int, available) -> str:
    ordered = sorted(int(h) for h in available)
    return str(min(ordered, key=lambda h: (abs(h - horizon), h)))


def lookup(table: dict, feats: dict, metric: str):
    """Most specific quantile triple for (key, family, global), basis-split for efficiency."""
    suffix = "|" + feats["basis"] if metric == "efficiency_delta_pct" else ""
    for name in ("key:" + feats["key"], "family:" + feats["family"], "global"):
        entry = table.get(name + suffix) or table.get(name)
        if entry and metric in entry:
            return entry[metric]
    return None


def predict(episode: dict, params: dict) -> dict:
    """{horizon: {metric: {p10, p50, p90}, goal_miss_probability, instability_risk}}."""
    feats = features(episode)
    tables = params["quantiles"]
    elasticity = params.get("budget_elasticity", {})
    scale = float(params.get("width_scale", 1.0))
    cap = float(params.get("lean_cap", 2.0))
    out = {}
    for horizon in required_horizons(episode):
        h = nearest_horizon(horizon, tables)
        block = {}
        for metric in METRICS:
            trio = lookup(tables[h], feats, metric) or params["fallback"][metric]
            p10, p50, p90 = (float(trio[k]) for k in ("p10", "p50", "p90"))
            lean = 0.0
            if feats["budget_pct"] is not None:
                beta = float((elasticity.get(h) or {}).get(metric, 0.0))
                lean = max(-cap, min(beta * feats["budget_pct"], cap))
            lo = p50 + lean - (p50 - p10) * scale
            mid = p50 + lean
            hi = p50 + lean + (p90 - p50) * scale
            lo, mid, hi = sorted((lo, mid, hi))
            block[metric] = {"p10": round(lo, 6), "p50": round(mid, 6), "p90": round(hi, 6)}
        block["goal_miss_probability"] = 0.5
        block["instability_risk"] = 0.3
        out[str(horizon)] = block
    return out


def fallback_prediction(episode: dict, params: dict) -> dict:
    """Global quantiles only: used when feature extraction fails, so the episode stays covered."""
    out = {}
    for horizon in required_horizons(episode):
        block = {m: {k: round(float(params["fallback"][m][k]), 6) for k in ("p10", "p50", "p90")}
                 for m in METRICS}
        block["goal_miss_probability"] = 0.5
        block["instability_risk"] = 0.3
        out[str(horizon)] = block
    return out


def answer(line: str, params: dict):
    """The output line for one input line, or None when no episode id can be read."""
    try:
        episode = json.loads(line)
    except ValueError:
        return None
    if not isinstance(episode, dict):
        return None
    eid = episode_id_of(episode)
    if eid is None:
        return None
    try:
        horizons = predict(episode, params)
    except Exception:  # noqa: BLE001 - one bad episode must never cost the basket
        try:
            horizons = fallback_prediction(episode, params)
        except Exception:  # noqa: BLE001
            horizons = fallback_prediction({}, params)
    return json.dumps({"episode_id": eid, "horizons": horizons}, separators=(",", ":"))


def main(argv: list[str]) -> int:
    path = argv[argv.index("--params") + 1] if "--params" in argv else PARAMS_PATH
    params = load_params(path)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        out = answer(line, params)
        if out is not None:
            sys.stdout.write(out + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
