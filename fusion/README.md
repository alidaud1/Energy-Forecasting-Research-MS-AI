# Load–solar fusion and decision support

This module connects probabilistic load and solar forecasts through a checked
forecast contract, time/unit alignment, scenario-based fusion, and deterministic
decision indicators. It is a research demonstration: the included synchronized
synthetic benchmark supports controlled fusion evaluation. The GEFCom load and
solar competition tracks are separate data sets and do not form a measured,
coincident grid net-load record.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r fusion/requirements.txt
```

## Reproduce the synchronized synthetic demonstration

```bash
python -m fusion.synthetic_benchmark \
  --output fusion/examples/synthetic_benchmark \
  --start 2020-01-01 --days 14 --capacity-mw 60 \
  --scenarios 5000 --seed 2026
python -m fusion.synthetic_eda \
  --input fusion/examples/synthetic_benchmark
```

This writes the synthetic load/solar forecast inputs, aligned index and actuals,
fused net-load quantiles, evaluation metrics, decision report, run manifests,
and exploratory plots under the chosen output directory. A frozen example run
is included in `examples/synthetic_benchmark/`.

## Run the agents on forecast files

The forecast files must cover the same timestamps, location and forecast horizon.
The load input is MW; solar input must declare a capacity conversion in the
adapter configuration. Provide the matching solar timestamp/split file and state
the assumed installed capacity explicitly:

```bash
python -m fusion.run_existing \
  --load /path/to/load_predictions.csv \
  --solar /path/to/solar_predictions.csv \
  --solar-timestamps /path/to/matching_solar_test_index.csv \
  --solar-capacity-mw 60 \
  --output joint_outputs/run --scenarios 5000 --seed 42
```

Review `joint_manifest.json` and `decision_report.json` with every result. The
decision agent reports transparent peak, ramp, reserve and risk indicators for
human review; they are not dispatch instructions. Do not treat cyclic alignment
of the non-overlapping GEFCom load and solar tracks as measured joint grid
evidence.

## Tests

```bash
python -m pytest fusion/ -q
```

The journal ablation regression tests use the included
`elsevier_paper/experiments/run_ablations.py` analysis module. The broader paper
and all generated training artifacts are not included in this repository branch.

## Optional language summary

Deterministic numeric outputs do not require an API key. To request the optional
short OpenAI explanation, copy `fusion/.env.example` to `fusion/.env` and add
your own key locally. The key file is ignored by Git and must never be committed.
