#!/usr/bin/env python3
"""Rebuild data/fixture_weekly.jsonl from an SN21-adtao checkout.

The fixture is every weekly-era episode the upstream repository publishes
with both inputs and settled labels (data/training/training_episodes.json,
plus data/episodes/* joined to data/outcomes/*), rewritten in the training-v4
record shape: {"episode_id", "input": {"episode_id", "payload"}, "labels"}.
Records are kept in episode_id order while they fit a byte budget (default
160 KiB) so the committed file stays far below the 256 KiB per-file limit;
records that would overflow it are skipped. Weekly-era episodes are 7/14-day only and graded on CPA, so every label row
carries goal_basis "cpa" and efficiency_delta_pct = the CPA delta.

    python3 tracks/main/tools/build_fixture.py --upstream /path/to/SN21-adtao
"""
from __future__ import annotations

import argparse
import glob
import json
import os

MAX_BYTES = 160 * 1024
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fixture_weekly.jsonl")


def label_rows(outcomes: dict) -> dict:
    rows = {}
    for key, row in sorted(outcomes.items()):
        if not key.startswith("t") or not isinstance(row, dict):
            continue
        eff = row.get("efficiency_delta_pct", row.get("cpa_delta_pct"))
        if eff is None or row.get("cost_delta_pct") is None or row.get("conversions_delta_pct") is None:
            continue
        rows[key[1:]] = {"cost_delta_pct": row["cost_delta_pct"],
                         "conversions_delta_pct": row["conversions_delta_pct"],
                         "efficiency_delta_pct": eff, "goal_basis": "cpa"}
    return rows


def record(episode_id: str, payload: dict, outcomes: dict):
    labels = label_rows(outcomes)
    if not labels:
        return None
    return {"episode_id": episode_id,
            "input": {"episode_id": episode_id, "payload": payload},
            "labels": labels}


def build(upstream: str) -> list[dict]:
    out = {}
    for rec in json.load(open(os.path.join(upstream, "data", "training", "training_episodes.json"))):
        r = record(rec["episode_id"], rec["input"], rec.get("outcome") or {})
        if r:
            out[r["episode_id"]] = r
    for ep_path in sorted(glob.glob(os.path.join(upstream, "data", "episodes", "*.json"))):
        out_path = os.path.join(upstream, "data", "outcomes", os.path.basename(ep_path))
        if not os.path.exists(out_path):
            continue
        inputs = {e["episode_id"]: e["input"] for e in json.load(open(ep_path))["episodes"]}
        for o in json.load(open(out_path))["episodes"]:
            if o["episode_id"] in inputs and o["episode_id"] not in out:
                r = record(o["episode_id"], inputs[o["episode_id"]],
                           (o.get("validator_only_outcomes") or {}).get("outcomes") or {})
                if r:
                    out[r["episode_id"]] = r
    return [out[k] for k in sorted(out)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", required=True, help="path to an SN21-adtao checkout")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    args = parser.parse_args()
    lines, size, skipped = [], 0, 0
    for r in build(args.upstream):
        line = json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
        if size + len(line.encode("utf-8")) > args.max_bytes:
            skipped += 1
            continue
        lines.append(line)
        size += len(line.encode("utf-8"))
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    print(json.dumps({"records": len(lines), "skipped": skipped, "bytes": size, "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
