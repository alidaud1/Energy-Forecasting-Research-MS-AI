import ast
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from elsevier_paper.experiments.build_logged_colab_notebooks import build, ROOT
from elsevier_paper.experiments.colab_provenance_runner import keyed_forecasts, verify_inputs, run_logged_solar, instrument_epoch_history


def write_splits(root):
    for zone in (1, 2, 3):
        folder = root / f"zone{zone}"
        folder.mkdir(parents=True)
        for split, start, rows in (("train", "2014-02-28 12:00", 1),
                                   ("val", "2014-03-01 01:00", 2208),
                                   ("test", "2014-06-01 01:00", 720)):
            pd.DataFrame({"ZONEID": zone, "TIMESTAMP": pd.date_range(start, periods=rows, freq="h"),
                          "POWER": .2}).to_csv(folder / f"{split}.csv", index=False)


def test_notebooks_embed_exact_original_code_and_isolated_wrapper(tmp_path):
    paths = build(tmp_path)
    assert len(paths) == 5
    for path in paths:
        nb = json.loads(path.read_text())
        sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for source in sources:
            ast.parse(source)
        constants = {node.targets[0].id: ast.literal_eval(node.value)
                     for node in ast.parse(sources[1]).body
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in ("ORIGINAL_RUNNER_SOURCE", "PROVENANCE_WRAPPER_SOURCE")}
        assert constants["ORIGINAL_RUNNER_SOURCE"] == (ROOT / "agentic_solar/solar-forecasting-GEFCom2014/colab/solar_gpu_training.py").read_text()
        assert "def run_logged_solar" in constants["PROVENANCE_WRAPPER_SOURCE"]
        assert "run_logged_solar" in sources[-1]
        assert "journal_repro_archives" in sources[-1]
        assert not any(c.get("outputs") for c in nb["cells"])


def test_input_hashes_keys_and_partitions_are_checked(tmp_path):
    write_splits(tmp_path)
    index, evidence = verify_inputs(SimpleNamespace(ZONES=(1, 2, 3)), tmp_path)
    assert len(index["val"]) == 6624 and len(index["test"]) == 2160
    assert len(evidence) == 9
    path = tmp_path / "zone1/test.csv"
    frame = pd.read_csv(path)
    frame.loc[1, "TIMESTAMP"] = frame.loc[0, "TIMESTAMP"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unordered/duplicate"):
        verify_inputs(SimpleNamespace(ZONES=(1, 2, 3)), tmp_path)


def test_keyed_forecasts_do_not_drop_quantiles_or_zone_identifiers():
    index = pd.DataFrame({"ZONEID": [1, 2], "TIMESTAMP": ["2014-06-01", "2014-06-01"], "POWER": [.2, .4]})
    frame = keyed_forecasts(index, np.tile(np.arange(1, 100)/100, (2, 1)))
    assert list(frame)[:3] == ["ZONEID", "TIMESTAMP", "POWER"]
    assert frame.shape == (2, 102)
    with pytest.raises(ValueError, match="shape"):
        keyed_forecasts(index, np.zeros((2, 97)))


def test_fresh_training_refuses_a_silent_cpu_substitution(monkeypatch, tmp_path):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Colab GPU"):
        run_logged_solar(SimpleNamespace(MODEL_CONFIG={"LSTM": {}}), "LSTM", "", tmp_path)
    assert not list(tmp_path.iterdir())


def test_epoch_logging_preserves_native_loss_and_fitted_weights(tmp_path):
    import types
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    source = (ROOT / "agentic_solar/solar-forecasting-GEFCom2014/colab/solar_gpu_training.py").read_text()
    original, logged = types.ModuleType("original_test"), types.ModuleType("logged_test")
    exec(source, original.__dict__)
    exec(source, logged.__dict__)
    instrument_epoch_history(logged, source, tmp_path)
    data = TensorDataset(torch.ones(4, 2), torch.ones(4)*.2)
    loaders = {name: DataLoader(data, batch_size=2, shuffle=False) for name in ("train", "val")}
    cfg = {"optimizer": "adam", "scheduler": "plateau", "epochs": 2, "patience": 10, "amp": False}
    fitted, histories = [], []
    for module, name in ((original, "original"), (logged, "logged")):
        torch.manual_seed(42)
        model = torch.nn.Linear(2, 99)
        model, history = module.train_model("test", model, loaders, cfg, tmp_path / f"{name}.pt", torch.device("cpu"))
        fitted.append(model.state_dict())
        histories.append(history)
    assert histories[0] == histories[1]
    for key in fitted[0]:
        torch.testing.assert_close(fitted[0][key], fitted[1][key], rtol=0, atol=0)
    assert len(pd.read_csv(tmp_path / "training_history.csv")) == 2
