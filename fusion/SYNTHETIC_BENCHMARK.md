# Synchronized joint benchmark

`synthetic_benchmark.py` creates the controlled benchmark needed by the joint
Fusion and Decision Agents when the available GEFCom load and solar tracks do
not share a calendar. It is intentionally labelled synthetic: it demonstrates
the integration and decision logic, but is not evidence that the simulated
values are measured utility behaviour.

## Reproduce

From the repository root:

```bash
agentic/.venv/bin/python -m fusion.synthetic_benchmark \
  --output joint_outputs/synthetic_benchmark --days 14 --seed 2026
```

The command writes synchronized hourly inputs, a 5,000-scenario fused
forecast, a deterministic decision report, manifests, and evaluation metrics.
By default the first seven days calibrate load--solar residual dependence and
the following seven days are held out for evaluation. Use a fresh output
directory when rerunning; pre-existing outputs may belong to the earlier
same-period-calibration version. The seed makes every artifact reproducible.

To create the optional supervisor-facing narrative after the numeric run:

```bash
agentic/.venv/bin/python -m fusion.openai_summary \
  --run joint_outputs/synthetic_benchmark
```

The key is loaded from `fusion/.env` via `OPENAI_API_KEY`. The API is used only
to explain the already-computed report; it does not make forecasts or
operational decisions. It writes `openai_summary.txt` and
`openai_summary.json`. If the key, network or quota is unavailable, it records
the skipped/failed status while leaving all deterministic outputs intact.

## Inputs and outputs

- `synthetic_actuals.csv`: labelled load, solar and ground-truth net load.
- `load_predictions.csv`: load MW forecasts (`actual_LOAD`, `q01`--`q99`).
- `solar_predictions.csv`: solar capacity-fraction forecasts (`q01`--`q99`).
- `solar_index.csv`: the matching timestamps, zone and solar actual fraction.
- `load_predictions_test.csv`, `solar_predictions_test.csv`, and
  `solar_index_test.csv`: the later-period inputs passed to fusion.
- `joint_run/net_load_forecast.csv`: Fusion Agent net-load Q01--Q99 in MW.
- `joint_run/decision_report.json`: reserve, peaks, ramps, flags and actions.
- `joint_evaluation.json`: pinball loss, interval coverage, sharpness and
  quantile-crossing checks.
- `synthetic_benchmark_manifest.json`: assumptions, scenarios and limitations.

The scenarios are normal operation, cloudy/low solar, high demand, solar
outage and evening ramp. Load and solar residuals share a common shock. Their
rank dependence is estimated from the earlier calibration block and supplied
explicitly to the Fusion Agent for the held-out hours; held-out actuals do not
calibrate fusion. The forecast quantiles are scripted by the simulator, not
produced by the separately trained GEFCom models.

## Thesis wording

Use this as a **controlled synchronized proof-of-concept**. GEFCom results
remain the empirical validation of the individual load and solar forecasters.
Do not describe these simulated observations as measured grid data; state that
measured co-located utility load/generation data are future validation work.
