import math

from src.agents.base_agent import AgentConfig
from src.agents.message_pool import MessagePool
from src.agents.model_manager_agent import ModelManagerAgent


def _mk_agent(tmp_path):
    pool = MessagePool(log_path=str(tmp_path / "pool.jsonl"), load_existing=False)
    cfg = AgentConfig(agent_id="mm", role="model_manager", run_id="run_test")
    return ModelManagerAgent(cfg, pool)


def _init_hpo(a: ModelManagerAgent, model_type: str):
    base = {
        "command": "train",
        "track": "load",
        "paths": {},
        "columns": {},
        "model_type": model_type,
        "hyperparams": a._make_default_params(model_type),
        "window": {"W": 168, "H": 24, "stride": 24},
        "training": {"seed": 42},
        "hpo": {"enabled": True, "n_trials": 3},
    }
    a._init_optuna(base)
    return base


def test_table_ii_xgboost_bounds_and_steps(tmp_path):
    a = _mk_agent(tmp_path)
    base = _init_hpo(a, "xgboost")
    p = a._sample_params_optuna("xgboost", dict(base["hyperparams"]))
    assert 10 <= int(p["n_estimators"]) <= 300
    assert int(p["n_estimators"]) % 10 == 0
    assert 4 <= int(p["max_depth"]) <= 16
    assert int(p["max_depth"]) % 2 == 0
    lr = float(p["learning_rate"])
    assert 1e-4 <= lr <= 1e-1


def test_table_ii_linear_space(tmp_path):
    a = _mk_agent(tmp_path)
    base = _init_hpo(a, "linear")
    p = a._sample_params_optuna("linear", dict(base["hyperparams"]))
    assert p["regularization"] in ("none", "ridge")
    alpha = float(p["alpha"])
    assert 1e-4 <= alpha <= 1.0
    assert math.isfinite(alpha)


def test_table_ii_svr_space(tmp_path):
    a = _mk_agent(tmp_path)
    base = _init_hpo(a, "svr")
    p = a._sample_params_optuna("svr", dict(base["hyperparams"]))
    C = float(p["C"])
    g = float(p["gamma"])
    assert 1e-3 <= C <= 1e3
    assert 1e-3 <= g <= 1e3


def test_table_ii_mlp_space(tmp_path):
    a = _mk_agent(tmp_path)
    base = _init_hpo(a, "mlp")
    p = a._sample_params_optuna("mlp", dict(base["hyperparams"]))
    assert 2 <= int(p["hidden_layers"]) <= 5
    hs = int(p["hidden_size"])
    assert 16 <= hs <= 512
    lr = float(p["learning_rate"])
    assert 1e-4 <= lr <= 1e-1
    dr = float(p["dropout"])
    assert 0.0 <= dr <= 0.5


def test_table_ii_rnn_spaces(tmp_path):
    a = _mk_agent(tmp_path)
    base = _init_hpo(a, "lstm")
    for mt in ("lstm", "gru"):
        p = a._sample_params_optuna(mt, dict(base["hyperparams"]))
        assert 1 <= int(p["layers"]) <= 3
        assert 16 <= int(p["hidden_size"]) <= 512
        assert 16 <= int(p["fc_size"]) <= 512
        lr = float(p["learning_rate"])
        assert 1e-4 <= lr <= 1e-1


def test_table_ii_cnn_space(tmp_path):
    a = _mk_agent(tmp_path)
    base = _init_hpo(a, "cnn")
    p = a._sample_params_optuna("cnn", dict(base["hyperparams"]))
    assert 1 <= int(p["layers"]) <= 3
    assert 1 <= int(p["kernel_size"]) <= 5
    assert 16 <= int(p["filters"]) <= 128
    assert 16 <= int(p["fc_size"]) <= 512
    lr = float(p["learning_rate"])
    assert 1e-4 <= lr <= 1e-1

