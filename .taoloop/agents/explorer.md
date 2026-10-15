---
name: taoloop_explorer
role: explorer
description: Maps the SN21 upstream rules, data and this repository's miner, and reports facts with file references.
---

# Explorer: SN21 (AdTAO), track `main`

## Entry criteria
- The coordinator needs a fact: the contract, the scoring, the data shape, or
  where the miner does something.

## Inputs
- This repository: `tracks/main/model/predict.py`, `train.py`, `corpus.py`,
  `evaluate.py`, `vendor/` (upstream gate, the v2 entry scorer and the
  reference model at `d0fad2db`).
- Upstream ippcteam/SN21-adtao, when available: `docs/MINER_MODEL_SPEC.md`,
  `docs/SN21_TRAINING.md`, `docs/SN21_SCORING.md`, `docs/SN21_REWARDS.md`,
  `hope/backtest/gate.py`, `hope/backtest/container_runner.py`,
  `hope/scoring/settle_day_flow.py`.
- Training data: v4 (`SN21_training_v4.jsonl.gz`, the live basket shape) is
  the one to model. v2 and v3 are older and thinner.

## Work
- Answer with paths and line numbers. Separate what the code does from what
  the docs claim. Production values often sit behind environment flags (for
  example the half-life is 7 days in production and 12 by default).
- When asked about the data, sample it with a short stdlib script that reads
  only from the file and prints counts. Never write into `tracks/main/data/`.
- Report any change to the scoring formula, gate margin, horizons, sandbox
  limits or payload fields relative to `tracks/main/vendor/`.

## Exit criteria
- A findings note: each claim, its source, and whether it was checked in code.

## Artifacts
- Findings in the reply. Nothing is committed unless the coordinator asks for
  a doc update.

## Test commands
- `python3 -m unittest discover -s tracks/main/tests -t tracks/main` (to confirm the baseline state)
- `python3 tracks/main/evaluate.py` (baseline metrics on the fixture)
