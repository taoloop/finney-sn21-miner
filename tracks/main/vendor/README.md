# Vendored upstream code

From https://github.com/ippcteam/SN21-adtao at d0fad2db190b8a67f4e7ef2698f4814e32d3f0ea,
MIT licence (`LICENSE-SN21-adtao` at the repository root).

| File | Upstream path | Changes |
| :--- | :--- | :--- |
| `gate.py` | `hope/backtest/gate.py` | none |
| `settle_v2.py` | `hope/scoring/settle_day_flow.py` | only the v2 entry scorer and its constants; relative import of `gate` |
| `reference_model.py` | `reference_model/model.py` | none (used by `evaluate.py` as the comparison line) |

`data/fixture_weekly.jsonl` is derived from upstream `data/training/`,
`data/episodes/` and `data/outcomes/` by `tools/build_fixture.py`.

Do not edit these files to make a score go up: they are the subnet's rules.
Re-copy them when upstream changes the formula.
