"""Labelled SN21 corpora, normalised for train.py and evaluate.py.

Reads, one JSON object per line (plain or .gz, `_manifest` lines skipped):
- training v4 (live basket shape): {"episode_id", "input": <served episode>,
  "labels": {"7"|"14"|"28": {cost_delta_pct, conversions_delta_pct,
  efficiency_delta_pct, goal_basis, finalized_on}}}
- rich training v2/v3: {"episode_id", "input": {"payload", "transition_key"},
  "labels": {h: {cost_delta_pct, conversions_delta_pct, cpa_delta_pct,
  conversion_value_delta_pct}}}; the graded efficiency label is chosen by
  payload.account_state.goal_basis (cpa when absent)
- data/fixture_weekly.jsonl in this track (v4 shape, built by
  tools/build_fixture.py from the upstream weekly-era exports)

Every record becomes {"episode_id", "episode": <what the container is fed>,
"labels": {int horizon: {metric: float}}, "basis", "order"}.
"""
from __future__ import annotations

import gzip
import json
import os

METRICS = ("cost_delta_pct", "conversions_delta_pct", "efficiency_delta_pct")
TRACK_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(TRACK_DIR, "data", "fixture_weekly.jsonl")


def _open(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _served(record: dict) -> dict:
    inner = record.get("input")
    if not isinstance(inner, dict):
        inner = {k: v for k, v in record.items() if k not in ("labels", "outcome")}
    episode = dict(inner)
    episode.setdefault("episode_id", record.get("episode_id"))
    return episode


def _basis(episode: dict) -> str:
    payload = episode.get("payload") if isinstance(episode.get("payload"), dict) else episode
    basis = ((payload.get("account_state") or {}).get("goal_basis") or {}).get("basis")
    return basis if basis in ("cpa", "conversion_value") else "cpa"


def _labels(raw: dict, basis: str) -> dict:
    out = {}
    for key, row in (raw or {}).items():
        if not isinstance(row, dict):
            continue
        try:
            horizon = int(str(key).lstrip("t"))
        except ValueError:
            continue
        eff = row.get("efficiency_delta_pct")
        if eff is None:
            eff = row.get("conversion_value_delta_pct" if basis == "conversion_value" else "cpa_delta_pct")
        values = {"cost_delta_pct": row.get("cost_delta_pct"),
                  "conversions_delta_pct": row.get("conversions_delta_pct"),
                  "efficiency_delta_pct": eff}
        try:
            out[horizon] = {m: float(values[m]) for m in METRICS}
        except (TypeError, ValueError):
            continue
    return out


def _order(episode: dict, raw_labels: dict) -> str:
    payload = episode.get("payload") if isinstance(episode.get("payload"), dict) else episode
    meta = payload.get("episode_metadata") or episode.get("episode_metadata") or {}
    when = meta.get("action_window_end") or meta.get("action_window_start")
    if not when:
        finals = sorted(str(r.get("finalized_on")) for r in (raw_labels or {}).values()
                        if isinstance(r, dict) and r.get("finalized_on"))
        when = finals[0] if finals else ""
    return str(when)


def load(path: str) -> list[dict]:
    """Labelled records sorted by time (action window end), ties by episode id."""
    records = []
    with _open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict) or "_manifest" in record:
                continue
            episode = _served(record)
            if episode.get("episode_id") in (None, ""):
                continue
            basis = _basis(episode)
            labels = _labels(record.get("labels") or record.get("outcome"), basis)
            if not labels:
                continue
            records.append({"episode_id": str(episode["episode_id"]), "episode": episode,
                            "labels": labels, "basis": basis,
                            "order": _order(episode, record.get("labels"))})
    records.sort(key=lambda r: (r["order"], r["episode_id"]))
    return records


def split(records: list[dict], holdout: float) -> tuple[list[dict], list[dict]]:
    """(train, held-out): the held-out part is the latest `holdout` fraction by time."""
    if holdout <= 0:
        return list(records), []
    cut = len(records) - max(1, int(round(len(records) * holdout)))
    return records[:cut], records[cut:]


def stride(records: list[dict], limit: int) -> list[dict]:
    """A deterministic, evenly spread subset of at most `limit` records."""
    if limit <= 0 or len(records) <= limit:
        return list(records)
    step = len(records) / limit
    return [records[int(i * step)] for i in range(limit)]
