"""Journal ablations must use the same held-out timestamps and fixed seed."""

import pandas as pd
import pytest

from elsevier_paper.experiments.run_ablations import _same_rank, run
from fusion.synthetic_benchmark import create_benchmark


def test_ablation_reproduces_frozen_holdout_baseline(tmp_path):
    benchmark = create_benchmark(tmp_path, days=4, calibration_days=2,
                                 seed=55, n_scenarios=100)
    result = run(tmp_path, make_figures=False)
    assert result["evaluation_hours"] == 48
    assert result["calibration_rank_correlation"] == pytest.approx(
        benchmark["evaluation"]["calibration_rank_correlation"])
    bounded = result["methods"]["copula_bounded"]
    assert bounded["mean_pinball_loss_mw"] == pytest.approx(
        benchmark["evaluation"]["mean_pinball_loss_mw"])
    assert bounded["coverage_q05_q95"] == pytest.approx(
        benchmark["evaluation"]["coverage_q05_q95"])
    assert bounded["quantile_crossing_pairs"] == 0
    assert (tmp_path / "ablation_results.json").is_file()
    assert (tmp_path / "ablation_results.csv").is_file()


def test_same_rank_is_literal_quantile_subtraction():
    left = pd.DataFrame({"TIMESTAMP": ["2020-01-01"],
                         **{f"q{i:02d}": [100 + i] for i in range(1, 100)}})
    right = pd.DataFrame({"TIMESTAMP": ["2020-01-01"],
                          **{f"q{i:02d}": [20 + i / 2] for i in range(1, 100)}})
    result = _same_rank(left, right)
    assert result.loc[0, "q01"] == pytest.approx(80.5)
    assert result.loc[0, "q50"] == pytest.approx(105.0)


def test_ablation_refuses_old_same_period_protocol(tmp_path):
    create_benchmark(tmp_path, days=4, calibration_days=2,
                     seed=55, n_scenarios=100)
    path = tmp_path / "joint_evaluation.json"
    import json
    content = json.loads(path.read_text())
    content["dependence_uses_evaluation_actuals"] = True
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="leakage-free"):
        run(tmp_path, make_figures=False)
