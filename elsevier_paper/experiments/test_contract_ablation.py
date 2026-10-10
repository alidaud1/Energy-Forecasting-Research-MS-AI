import numpy as np
import pandas as pd

from fusion.forecast_contract import QUANTILE_COLUMNS
from elsevier_paper.experiments.run_contract_ablation import checked_fusion, run, unchecked_positional_fusion


def test_numerical_control_matches_existing_fusion():
    stamp = pd.date_range("2020-01-01", periods=3, freq="h")
    load = pd.DataFrame({"TIMESTAMP": stamp, **{q: [80+i, 90+i, 100+i]
                         for i, q in enumerate(QUANTILE_COLUMNS)}})
    solar = pd.DataFrame({"TIMESTAMP": stamp, **{q: [i/3, i/4, i/5]
                          for i, q in enumerate(QUANTILE_COLUMNS)}})
    np.testing.assert_allclose(checked_fusion(load, solar, capacity=60),
                               unchecked_positional_fusion(load, solar, capacity=60), rtol=0, atol=1e-9)


def test_archived_fault_injection_and_negative_control():
    result = run()
    assert result["summary"]["malformed_cases"] == 9
    assert result["summary"]["checked_rejections"] == 9
    assert result["clean_max_abs_difference_mw"] < 1e-9
    assert result["declared_unit_case"]["checked_max_abs_difference_mw"] < 1e-9
    assert result["declared_unit_case"]["unchecked_max_abs_difference_mw"] > 1
    assert result["negative_control"]["checked"] == "finite_output"
    assert result["negative_control"]["unchecked"] == "finite_output"
