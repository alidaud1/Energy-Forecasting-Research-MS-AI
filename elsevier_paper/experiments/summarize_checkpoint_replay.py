"""Verify all keyed CPU replays and construct the recorded solar ensemble."""
import json
import pickle

import numpy as np
import pandas as pd

from .replay_solar_checkpoints import MODELS, OUTPUT, SOLAR, QCOLS, metrics, sha


def run():
    summaries = {}
    for name in MODELS:
        folder = OUTPUT / name
        report = json.loads((folder / "replay_report.json").read_text())
        if sha(folder / "configuration.json") != report["configuration_sha256"]:
            raise ValueError(f"changed configuration {name}")
        for split in ("val", "test"):
            path = folder / f"{split}_keyed_q01_q99.csv"
            if sha(path) != report["splits"][split]["forecast_sha256"]:
                raise ValueError(f"changed forecast {name}")
            frame = pd.read_csv(path)
            q = frame[QCOLS].to_numpy()
            assert len(frame) == (6624 if split == "val" else 2160)
            assert not frame.duplicated(["ZONEID", "TIMESTAMP"]).any()
            assert np.isfinite(q).all() and (q >= 0).all() and (q <= 1).all()
            assert (np.diff(q, axis=1) >= 0).all()
            for key, value in metrics(frame.POWER.to_numpy(), q).items():
                np.testing.assert_allclose(value, report["splits"][split]["metrics"][key], atol=3e-5)
        summaries[name] = report
    with (SOLAR / "pipeline/saved_models/Ensemble_best.pkl").open("rb") as stream:
        weights = pickle.load(stream)["weights"]
    ensemble = OUTPUT / "Ensemble"
    ensemble.mkdir(exist_ok=True)
    for split in ("val", "test"):
        frames = {name: pd.read_csv(OUTPUT / name / f"{split}_keyed_q01_q99.csv") for name in weights}
        keys = next(iter(frames.values()))[["ZONEID", "TIMESTAMP", "POWER"]]
        for frame in frames.values():
            pd.testing.assert_frame_equal(keys, frame[["ZONEID", "TIMESTAMP", "POWER"]])
        q = sum(weights[name]*frame[QCOLS].to_numpy() for name, frame in frames.items())
        path = ensemble / f"{split}_keyed_q01_q99.csv"
        pd.concat([keys, pd.DataFrame(q, columns=QCOLS)], axis=1).to_csv(path, index=False)
        summaries.setdefault("Ensemble", {"recorded_weights": weights, "splits": {}})["splits"][split] = {
            "forecast_sha256": sha(path), "metrics": metrics(keys.POWER.to_numpy(), q)}
    result = {"scope": "keyed CPU checkpoint replay; no fresh solar retraining claimed",
              "historical_training_environment": "not recovered from checkpoint weights",
              "models": summaries}
    (OUTPUT / "verified_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    pd.DataFrame([{"model": name, **report["splits"]["test"]["metrics"]}
                  for name, report in summaries.items()]).to_csv(OUTPUT / "verified_test_metrics.csv", index=False)
    print(json.dumps({name: report["splits"]["test"]["metrics"] for name, report in summaries.items()}, indent=2))
    return result


if __name__ == "__main__":
    run()
