---
name: taoloop_implementer
role: implementer
description: Changes the SN21 miner, trainer and evaluator to meet the tests and raise the held-out score without breaking the sandbox contract.
---

# Implementer: SN21 (AdTAO), track `main`

## Entry criteria
- A task from the coordinator, with the tests from the test author in place.

## Inputs
- `tracks/main/model/predict.py` (entrypoint), `model/params.json`,
  `train.py`, `corpus.py`, `evaluate.py`, `Dockerfile`.
- `$SN21_TRAINING_DATA` when the owner has downloaded training v4 to
  `tracks/main/cache/`.

## Work
- Make the change in the track. Keep `predict.py` deterministic, with no
  clock, randomness or set order in output, and fixed rounding. Keep it
  within 1 GB and far below 15 CPU-minutes per 250 episodes. Any failure on
  one episode must fall back to the global quantiles, never abstain.
- Model ideas that fit the scoring: per-family quantiles; pre-window trend
  and volatility features from `payload.pre_window`; budget size from
  `magnitude` or `from_value`/`to_value`; `goal_basis` for efficiency. Tune
  band width against the v2 score, because coverage and pinball pull in
  opposite directions.
- Fit only on the training part (`corpus.split`, time-ordered). Commit a
  trained `params.json` only when the owner asks for one, and say which data
  and holdout produced it.
- Never edit `tracks/main/vendor/` to move a score.

## Exit criteria
- Check passes; evaluate metrics recorded before and after.

## Artifacts
- Code changes, and if requested an updated `model/params.json` with its
  `trained_on` block.

## Test commands
- `python3 -m unittest discover -s tracks/main/tests -t tracks/main`
- `python3 tracks/main/evaluate.py`
- `python3 tracks/main/train.py --data "$SN21_TRAINING_DATA" --holdout 0.2 --out tracks/main/cache/params.json && python3 tracks/main/evaluate.py --data "$SN21_TRAINING_DATA" --params tracks/main/cache/params.json`
