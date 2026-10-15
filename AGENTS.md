# AGENTS.md: SN21 (AdTAO) miner line, finney netuid 21

Instructions for any coding agent or IDE working in this repository. The
person who owns the miner reads README.md.

## What a miner is here

SN21 is a **coding-submission** subnet. A miner does not serve an axon and
does not POST predictions. It ships a **public, digest-pinned container
image** and commits `sn21-model:v1:<repo>@sha256:<64hex>` on chain. Each day
the operator runs that image in a sandbox on the day's basket of roughly 250
Google Ads change episodes:

- `--network=none`, 1 GB memory (no swap), 1 CPU, 15 CPU-minutes per basket,
  read-only root, 256 pids, no-new-privileges. Breaking a limit voids the day.
- stdin: one served episode per line, `{"episode_id", "transition_key",
  "payload": {...}, "system_estimate"}` (Episode payload v2.0).
- stdout: one line per episode, `{"episode_id", "horizons": {"7", "14",
  "28"}}`. Each block has monotone `p10 <= p50 <= p90` for `cost_delta_pct`,
  `conversions_delta_pct` and `efficiency_delta_pct`, plus
  `goal_miss_probability` and `instability_risk` (accepted, not scored).
- A missing, empty or null block for any horizon the episode asks for
  (`episode_metadata.outcome_horizons_days`) is an **abstention**, and every
  uncovered episode is charged as a full-weight 0.0. Always answer.
- Output must be **byte-identical** on a re-run. Admission runs a sample
  twice and rejects `nondeterministic`.

Deltas are fractional changes against the 60-day pre-window
(`-0.05` = 5% lower). `efficiency_delta_pct` is graded as the CPA delta or the
conversion-value delta, chosen by `payload.account_state.goal_basis.basis`,
which is frozen at reveal.

## Layout

| Path | What it is |
| :--- | :--- |
| `tracks/main/model/predict.py` | The miner: the container entrypoint. Stdlib only. |
| `tracks/main/model/params.json` | Fitted parameters baked into the image. The shipped file is the **untrained** upstream reference constants. |
| `tracks/main/train.py` | Fits `params.json`: per-(change family, horizon) residual quantiles, efficiency split by goal basis, and budget elasticity. |
| `tracks/main/corpus.py` | Loads training v4 / v2 / v3 bundles and the fixture, with a time-ordered split. |
| `tracks/main/evaluate.py` | Local scorer: runs the real entrypoint twice and scores with the subnet's v2 formula and admission gate. |
| `tracks/main/vendor/` | Upstream scorer, gate and reference model, copied unchanged (MIT). These are the rules: do not edit them to gain score. |
| `tracks/main/data/fixture_weekly.jsonl` | 30 labelled weekly-era episodes (7/14-day only, CPA basis), capped at 160 KiB by `tools/build_fixture.py`. A smoke test, not a benchmark. |
| `tracks/main/tests/` | Contract, scorer and train/evaluate tests (`unittest`). |
| `tracks/main/Dockerfile` | The image. It copies only `model/predict.py` and `model/params.json`. |
| `.taoloop/line.json`, `tracks/main/.taoloop/track.json` | Manifests: commands, settings, delivery steps, upstream sources. |

There is one track, `main`, so there is no `shared/` folder.

## How the track is scored

Upstream `ippcteam/SN21-adtao` at `d0fad2db`. Production values are set by
environment flags and published amendments, so where docs and code disagree
the daily receipt is authoritative.

1. **Admission, once per digest.** On a held-out corpus,
   `gate = 0.7 * quantile + 0.3 * direction` (pinball loss normalised by the
   mean absolute actual; a zero p50 on a nonzero actual is a direction miss).
   The model must exceed the persistence baseline (p50 = 0, band = the corpus
   median absolute delta) by **5%**, reach at least 90% of its coverage, and
   be deterministic. The upstream reference model was ruled **not**
   admissible on purpose, so an untrained copy of this repository will not
   earn.
2. **Per (episode, horizon) entry, formula v2:**
   `(0.50 Q + 0.10 C + 0.15 D + 0.15 G) / 0.90`, clipped to [0, 1]. Q is
   normalised pinball over the three metrics. C is interval coverage less
   0.5 x min(1, p90 - p10). D is efficiency direction (both inside ±0.005
   earns 0.5). G is `max(0, 1 - |p50 - actual| / 0.40)` on efficiency.
3. **Weights:** horizons 7/14/28 at 0.20/0.35/0.45. Standing is the
   age-weighted mean of own score minus the field mean on the same cell.
   Production since September 2026: half-life 7 days, 28-day window, prior
   mass 250 (docs elsewhere still say 12 days and 35 days). Uncovered episodes
   count as 0.0.
4. **Paid:** at least 250 evidence mass, 7 scored settle days, 75% daily
   coverage, one seat per coldkey, the earliest submitter of a copied model,
   and the 1,000-alpha hold. The top 20 get 50/25/10%, each later rank 0.8 of
   the one before, renormalised.

`evaluate.py` reproduces steps 1 and 2 locally. It cannot reproduce the
relative-to-field standing, because that needs other miners' predictions.

## Commands (run from the repository root)

```bash
# check: contract, determinism, scorer and train/evaluate tests (stdlib, ~2 s)
python3 -m unittest discover -s tracks/main/tests -t tracks/main

# run: the miner over a basket on stdin
python3 tracks/main/model/predict.py < basket.jsonl > predictions.jsonl

# evaluate: last stdout line is {"score": <v2 mean>, "metrics": {...}}
python3 tracks/main/evaluate.py                          # bundled fixture
SN21_TRAINING_DATA=tracks/main/cache/SN21_training_v4.jsonl.gz \
  python3 tracks/main/evaluate.py                        # latest 20% by time

# train: fit on the earliest 80%, leaving the rest for evaluate
python3 tracks/main/train.py --data tracks/main/cache/SN21_training_v4.jsonl.gz --holdout 0.2
# final image params: refit on everything, after evaluate looked good
python3 tracks/main/train.py --data tracks/main/cache/SN21_training_v4.jsonl.gz --holdout 0
```

There is no build step for local work. The image build is a delivery step,
because it needs Docker and a registry. Training data v4 (27,768 episodes, live
basket shape, 61 MB gzipped) is at
`https://github.com/ippcteam/SN21-adtao/releases/download/training-v4-2026-10/SN21_training_v4.jsonl.gz`.
Download it to `tracks/main/cache/`, which `tracks/main/.gitignore` keeps out of git.

Read the `evaluate` metrics together, not just `score`:
`deterministic` must be true, `coverage` 1.0 and `monotone` true.
`gate_admitted_estimate` should be true with room to spare over
`required_gate_score`, and `score` should beat `reference_v2`.
`seconds_per_basket` must stay far below 900. The fixture (30 old episodes) is a smoke test. Its gate
estimate passes even the untrained reference constants, so it is no evidence
of admission. Any params trained on the fixture are scored in-sample there.

## Rules for changing the miner

- `model/predict.py` stays **stdlib-only** and must import nothing outside
  the image. If you add a dependency, pin it in the Dockerfile and keep the
  image well under 1 GB of resident memory.
- No clock, randomness, set-order or thread-count dependence in output. Use
  sorted iteration and fixed rounding. The tests run under two hash seeds.
- Never let one bad episode crash the basket or turn into an abstention. Fall
  back to global quantiles.
- Train and tune only on data earlier than what you evaluate on
  (`corpus.split` is time-ordered). The subnet's held-out set always lies
  after the newest published slice.
- Re-copy `vendor/` from upstream if the formula changes, and update the
  `sources` entry in `.taoloop/line.json`.

## Delivery: a person's steps, never an agent's

These are listed in `tracks/main/.taoloop/track.json`. Do **not** run them;
describe them:

1. Register a hotkey on finney netuid 21 and add the ed25519 binding
   (`btcli`, upstream `scripts/sn21_keys.py`). Both burn or sign.
2. Train on the newest bundle and keep the evaluate output.
3. `docker build` and `docker push` to a **private** repository, then read the
   digest.
4. Run the image three times under the sandbox flags and compare the outputs.
5. Commit `sn21-model:v1:<repo>@sha256:<64hex>` from the hotkey (signed).
6. Only then make the repository public, and check an anonymous pull.
7. Hold 1,000 SN21 alpha on the hotkey (spends funds).
8. Watch the gate verdict, coverage and receipts (`scripts/verify_day.py`).

## Off limits

- Wallets, mnemonics, coldkey passwords, registry tokens, API keys: never in
  the repository, a setting default, a log or a commit.
- Registering, staking, committing, pushing or publishing images, and
  anything that signs or spends, belong to a person.
- Do not edit `vendor/` to raise a score, and do not train on evaluation data.
- Do not copy another miner's image or weights. Copies are detected and only
  the earliest submitter is paid.
- Do not add `.claude/`, `.codex/`, `.github/`, hooks, MCP configuration or
  code at the repository root.
