import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path

from elsevier_paper.experiments.run_full_budget_load import causal_features, vectorized_loss


def test_vectorized_native_loss_and_gradients_are_equivalent():
    torch.manual_seed(42)
    prediction = torch.randn(10, 99, requires_grad=True)
    target = torch.randn(10)
    quantiles = np.arange(1, 100)/100
    error = target[:, None] - prediction
    reference = sum(torch.maximum(q*error[:, i], (q-1)*error[:, i]).mean()
                    for i, q in enumerate(quantiles))/99
    measured = vectorized_loss(prediction, target, quantiles)
    torch.testing.assert_close(measured, reference)
    grad1 = torch.autograd.grad(measured, prediction, retain_graph=True)[0]
    grad2 = torch.autograd.grad(reference, prediction)[0]
    torch.testing.assert_close(grad1, grad2)


def test_strict24h_features_do_not_consume_unavailable_values():
    n = 600
    frame = pd.DataFrame({"TIMESTAMP": pd.date_range("2005-01-01", periods=n, freq="h"),
                          "LOAD": np.arange(n, dtype=float),
                          **{f"w{i}": np.arange(n, dtype=float) for i in range(1, 26)}})
    before = causal_features(frame)
    changed = frame.copy()
    # Target at hour 500 is issued at hour 476. Values at or after that issue
    # hour are deliberately changed; all observed features remain invariant.
    changed.loc[476:, "LOAD"] += 10000
    changed.loc[476:, [f"w{i}" for i in range(1, 26)]] += 10000
    after = causal_features(changed)
    np.testing.assert_allclose(before.loc[500], after.loc[500])
    assert before.loc[500, "load_lag25"] == 475
    assert before.loc[500, "temperature_mean_lag25"] == 475


def test_new_load_models_have_full_budgets_and_complete_test_hours():
    folder = Path(__file__).resolve().parent / "load_full_budget_24h"
    manifest = json.loads((folder / "input_manifest.json").read_text())
    assert len(manifest["feature_columns"]) == 17
    assert manifest["latest_observed_feature_lag_hours"] == 25
    for name in ("Linear", "GBR"):
        cfg = json.loads((folder / name / "configuration.json").read_text())
        assert cfg["training_rows"] == 34726
        assert cfg["training_row_limit"] is None
        assert cfg["horizon_hours"] == 24
        assert cfg["settings"].get("max_epochs", 300) == 300
        assert cfg["settings"].get("estimators_per_quantile", 50) == 50
        frame = pd.read_csv(folder / name / "test_q01_q99.csv")
        assert len(frame) == 6552
        assert frame.TIMESTAMP.iloc[-1] == "2010-09-30 23:00:00"
        assert len([c for c in frame if c.startswith("q")]) == 99
