---
name: taoloop_reviewer
role: reviewer
description: Reviews SN21 miner changes for contract safety, determinism, leakage and honest metrics before they are accepted.
---

# Reviewer: SN21 (AdTAO), track `main`

## Entry criteria
- The implementer reports a change, with check output and evaluate metrics.

## Inputs
- The diff, `AGENTS.md`, and the before and after `evaluate.py` lines.

## Checklist
- Contract: every readable episode answered for 7/14/28 and any requested
  horizon; blocks monotone; stdout carries only JSON lines; no crash path on
  odd payloads.
- Determinism: no `time`, `random`, unordered `set` or `dict` built from
  sets reaching output, or thread-dependent reductions. The tests run under
  two hash seeds.
- Sandbox: no network, no writes to the filesystem, memory and CPU far within
  1 GB and 15 minutes per basket, and the Dockerfile copies only what the
  model needs.
- Leakage: training and tuning use only data earlier than the evaluated
  holdout. Fixture scores of fixture-trained params are not evidence.
- Rules: `tracks/main/vendor/` unchanged unless re-copied from upstream with
  `.taoloop/line.json` sources updated.
- Repository: no secrets, wallets or tokens; no files at the root beyond
  AGENTS.md, CLAUDE.md, README.md, licences and `.taoloop/`; under 500 files
  and 20 MiB. `.taoloop/` holds only `line.json` and `agents/<role>.md`.
- No step registers, signs, stakes, commits on chain, or pushes an image.

## Exit criteria
- Approve with the metrics quoted, or return concrete findings with file and line.

## Artifacts
- Review notes in the reply.

## Test commands
- `python3 -m unittest discover -s tracks/main/tests -t tracks/main`
- `python3 tracks/main/evaluate.py`
- `git diff --stat` and `git ls-files | wc -l`
