# SN21 (AdTAO) miner line

A starting point for mining Bittensor subnet 21 (AdTAO, finney netuid 21).
Every owner of this subnet starts from this repository.

AdTAO asks miners to forecast what a Google Ads change will do. For each
change it wants the 10th, 50th and 90th percentile of the cost, conversions
and efficiency change 7, 14 and 28 days later. You don't run a server. You
publish a container image, commit its digest on chain, and the subnet runs it
each day on that day's basket of about 250 changes, offline, in a sandbox.
Forecasts are scored when the outcomes settle.

## What's here

One track, `tracks/main/`:

- **The miner** (`model/predict.py`) is a small, deterministic, stdlib-only
  forecaster. It looks up quantiles fitted per kind of change and per
  horizon, chooses efficiency quantiles by the account's goal (CPA or value),
  and shifts cost by the size of any budget change.
- **The trainer** (`train.py`) fits those parameters from the subnet's
  published training data.
- **The evaluator** (`evaluate.py`) runs the miner exactly as the sandbox
  does. It scores it with the subnet's own formula and admission gate, copied
  from upstream, and compares it with the upstream reference model.

The shipped parameters are the upstream reference model's constants. **Expect
the miner as shipped to fail admission.** The subnet requires beating a naive
baseline by 5% on its own held-out data, and it set that margin so that the
reference model fails. The gate estimate on the small bundled sample may say
"admitted"; that sample is too small and too old to count as evidence. Train
the miner and evaluate it on training v4 first.

## Quick start

```bash
# tests (Python 3.12, no installs needed)
python3 -m unittest discover -s tracks/main/tests -t tracks/main

# score the shipped miner on the small bundled sample
python3 tracks/main/evaluate.py

# train on real data, then evaluate on the newest 20% you held out
mkdir -p tracks/main/cache
curl -L -o tracks/main/cache/SN21_training_v4.jsonl.gz \
  https://github.com/ippcteam/SN21-adtao/releases/download/training-v4-2026-10/SN21_training_v4.jsonl.gz
python3 tracks/main/train.py --data tracks/main/cache/SN21_training_v4.jsonl.gz --holdout 0.2
python3 tracks/main/evaluate.py --data tracks/main/cache/SN21_training_v4.jsonl.gz
```

## Going live (your steps)

These steps sign transactions, spend funds or publish an image, so you do
them yourself. The exact commands are in `tracks/main/.taoloop/track.json`.

1. Register a hotkey on netuid 21 (`btcli subnet register`) and add the
   ed25519 binding described in upstream's quickstart.
2. Retrain with `--holdout 0` once the evaluation looks good. Ship only if
   it reports deterministic, full coverage and a gate pass with margin.
3. Build `tracks/main/Dockerfile` and push it to a **private** repository.
   Note the digest.
4. Run it three times under the sandbox flags and check that the outputs
   match byte for byte.
5. Commit `sn21-model:v1:<repo>@sha256:<digest>` from your hotkey.
6. Then make the image public. The earliest on-chain commit of a digest is
   the one that gets paid, so don't publish first.
7. Hold 1,000 SN21 alpha on the hotkey to be eligible for pay.
8. Watch for the gate verdict (Discord, adtao.io) and your daily coverage,
   and check settled days with upstream's `scripts/verify_day.py`.

Payment needs 250 scored predictions, 7 settled days and at least 75% daily
coverage. Every episode you don't answer counts as a zero, so a crash or
timeout is expensive.

## Caveats

- Scoring is centralised: the operator holds the outcome data and computes
  the weights.
- Rules change by dated amendment, and the documents disagree on some
  numbers (for example a 7-day or 12-day half-life). Daily receipts record
  the values in force.
- The image must be public, so anyone can copy your model. The only
  protection is committing the digest before you publish.

## Licence

MIT (`LICENSE`). Code and data copied from
[ippcteam/SN21-adtao](https://github.com/ippcteam/SN21-adtao) are MIT too
(`LICENSE-SN21-adtao`). They are listed in `.taoloop/line.json` and
`tracks/main/vendor/README.md`.
