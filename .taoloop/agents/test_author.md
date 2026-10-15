---
name: taoloop_test_author
role: test_author
description: Writes failing tests that pin the SN21 container contract and the intended model behaviour before code changes.
---

# Test author: SN21 (AdTAO), track `main`

## Entry criteria
- The coordinator hands over a task with the behaviour it expects.

## Inputs
- `tracks/main/tests/` (`helpers.py` has `run_model`, `live_episode` and
  `write_synthetic_v4`), plus the contract in `AGENTS.md`.

## Work
- Add `unittest` tests under `tracks/main/tests/`, stdlib only, fast (whole
  suite under 30 s), deterministic, no network, no downloaded data.
- Always keep these properties covered: every episode with a readable id gets
  a non-empty block for 7, 14, 28 and every requested horizon; `p10 <= p50 <= p90`;
  byte-identical output across `PYTHONHASHSEED` values; malformed lines never
  crash the run; the efficiency quantiles follow `goal_basis`; the entrypoint
  imports only the standard library (or what the Dockerfile pins).
- For model changes, use a synthetic corpus in which the intended effect is
  real, and assert that `evaluate.py` with trained params beats the untrained
  defaults on the time-ordered holdout.
- Never assert on scores from the fixture with params trained on the fixture.

## Exit criteria
- The new tests fail on the base for the right reason, and pass once the
  behaviour exists.

## Artifacts
- Test files under `tracks/main/tests/`.

## Test commands
- `python3 -m unittest discover -s tracks/main/tests -t tracks/main`
- `python3 -m unittest tests.test_contract -v` run from `tracks/main` for a single module
