"""SN21 per-(episode, horizon) entry scorer, formula v2.

Copied from ippcteam/SN21-adtao@d0fad2db190b8a67f4e7ef2698f4814e32d3f0ea,
hope/scoring/settle_day_flow.py (MIT, see LICENSE-SN21-adtao at the repository
root): the constants and the functions entry_components, _coverage_score,
_goal_p50_score, _goal_direction_score, entry_components_v2 and
score_entry_v2, unchanged. The ledger, settle clock and goal-basis resolution
in the same module are left out; the import of gate is made relative.
"""
from __future__ import annotations

from .gate import METRICS, QUANTILES, pinball

MIN_ENTRY_SCALE = 0.01
GOAL_METRIC = "efficiency_delta_pct"
INTERVAL_WIDTH_SCALE = 1.0
P50_GOAL_SCALE = 0.40
DIRECTION_FLAT_BAND = 0.005
W_QUANTILE = 0.50
W_COVERAGE = 0.10      # = 0.20 calibration × the computable half
W_DIRECTION = 0.15
W_GOAL = 0.15
W_TOTAL = W_QUANTILE + W_COVERAGE + W_DIRECTION + W_GOAL   # 0.90


def entry_components(pred: dict, actual: dict[str, float]) -> tuple[float, float]:
    """(quantile, direction) components of score_entry, same arithmetic.

    Exposed for the J3 vertical-error series, which stores RAW components
    so the weekly aggregate survives a formula recomposition (the goal-
    metric term is under governance review). Kept as a separate walk of the same
    logic rather than a refactor of score_entry so the scoring hot path
    and its tests stay byte-identical.
    """
    losses: list[float] = []
    scales: list[float] = []
    dir_hits = 0
    dir_total = 0
    for m in METRICS:
        a = actual[m]
        trio = pred.get(m)
        if trio:
            for q, key in QUANTILES:
                losses.append(pinball(a, float(trio[key]), q))
            scales.append(abs(a))
        if a != 0:
            dir_total += 1
            p50 = float(trio["p50"]) if trio else 0.0
            if p50 != 0 and (a > 0) == (p50 > 0):
                dir_hits += 1
    if not losses:
        return 0.0, 0.0
    scale = max(sum(scales) / len(scales), MIN_ENTRY_SCALE)
    quantile = max(0.0, 1.0 - (sum(losses) / len(losses)) / scale)
    direction = (dir_hits / dir_total) if dir_total else 0.0
    return round(quantile, 6), round(direction, 6)


def _coverage_score(trio: dict | None, actual: float) -> float:
    """Does [P10,P90] contain the actual, less a convex width penalty.

    Note what the legacy formula does and does not do (measured, not
    assumed): the penalty saturates at 1.0 and carries a 0.5 coefficient,
    so ANY covering band scores at least 0.5 no matter how absurdly wide.
    Coverage alone therefore does not punish a band-inflation strategy —
    the QUANTILE term does, and decisively. A [-5.0, +5.0] band against a
    0.4 actual takes coverage 0.5 but drives pinball loss to the
    normalisation scale, collapsing the 0.50-weighted quantile term to
    0.0. Net, inflation is heavily loss-making. Ported unchanged from
    onchain_adapter._coverage_score — the shape is the system's own.
    """
    if not trio:
        return 0.0
    p10, p90 = float(trio["p10"]), float(trio["p90"])
    covered = 1.0 if p10 <= actual <= p90 else 0.0
    width_penalty = min(1.0, max(0.0, p90 - p10) / INTERVAL_WIDTH_SCALE)
    return max(0.0, covered - 0.5 * width_penalty)


def _goal_p50_score(trio: dict | None, actual: float) -> float:
    """P50 accuracy on the goal metric: 1.0 exact, decaying linearly to 0."""
    if not trio:
        return 0.0
    return max(0.0, 1.0 - abs(float(trio["p50"]) - actual) / P50_GOAL_SCALE)


def _goal_direction_score(trio: dict | None, actual: float) -> float:
    """Direction of the goal metric only (spec:137), legacy semantics.

    Both-flat earns HALF, not full: predicting flat is the trivial call and
    must not score like a committed up/down one. The legacy comment records
    why — giving it 1.0 floored the predict-zero baseline higher than it
    deserved on stable epochs. This is the v2 form of the same abstention
    discipline the gate formula enforces with its zero-p50 rule.
    """
    if not trio:
        return 0.0
    p50 = float(trio["p50"])
    sp = 0 if abs(p50) < DIRECTION_FLAT_BAND else (1 if p50 > 0 else -1)
    st = 0 if abs(actual) < DIRECTION_FLAT_BAND else (1 if actual > 0 else -1)
    if sp != st:
        return 0.0
    return 0.5 if sp == 0 else 1.0


def entry_components_v2(pred: dict, actual: dict[str, float]) -> dict:
    """The four spec components for one (episode, horizon) entry.

    Quantile is this module's per-entry-normalised term (shared with the v1
    scorer — one implementation, no drift). Coverage averages the three
    metrics; direction and goal score the GOAL metric alone, per spec:137-138.
    """
    quantile, _v1_direction = entry_components(pred, actual)
    coverage = sum(_coverage_score(pred.get(m), actual[m])
                   for m in METRICS) / len(METRICS)
    goal_trio = pred.get(GOAL_METRIC)
    goal_actual = actual[GOAL_METRIC]
    return {
        "quantile": quantile,
        "coverage": round(coverage, 6),
        "direction": round(_goal_direction_score(goal_trio, goal_actual), 6),
        "goal": round(_goal_p50_score(goal_trio, goal_actual), 6),
    }


def score_entry_v2(pred: dict, actual: dict[str, float]) -> float:
    """Spec-restored formula for ONE (episode, horizon). See module docstring.

    An empty prediction still scores exactly 0.0, as under v1.
    """
    c = entry_components_v2(pred, actual)
    blended = (
        W_QUANTILE * c["quantile"]
        + W_COVERAGE * c["coverage"]
        + W_DIRECTION * c["direction"]
        + W_GOAL * c["goal"]
    ) / W_TOTAL
    return round(max(0.0, min(1.0, blended)), 6)


