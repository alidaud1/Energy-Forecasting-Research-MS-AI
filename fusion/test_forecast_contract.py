import numpy as np
import pandas as pd
import pytest

from fusion.forecast_contract import ForecastBundle, ForecastContractError, QUANTILE_COLUMNS


def valid_frame(n=3):
    columns = {"TIMESTAMP": pd.date_range("2024-01-01", periods=n, freq="h")}
    columns.update({q: [float(i)] * n for i, q in enumerate(QUANTILE_COLUMNS)})
    columns["actual"] = [49.0] * n
    return pd.DataFrame(columns)


def test_contract_accepts_canonical_forecast():
    bundle = ForecastBundle("load", valid_frame(), "MW", "grid-a", "24h", "actual")
    assert bundle.quantiles().shape == (3, 99)
    assert list(bundle.with_prefix("load").columns) == [
        "TIMESTAMP", *[f"load_q{i:02d}" for i in range(1, 100)]]


def test_timestamps_are_normalised_to_datetime():
    """A CSV-loaded bundle must merge against a datetime bundle."""
    frame = valid_frame()
    frame["TIMESTAMP"] = frame["TIMESTAMP"].astype(str)
    bundle = ForecastBundle("load", frame, "MW", "grid-a", "24h", "actual")
    assert pd.api.types.is_datetime64_any_dtype(bundle.frame["TIMESTAMP"])
    other = ForecastBundle("solar", valid_frame(), "MW", "z", "24h")
    merged = bundle.frame[["TIMESTAMP"]].merge(other.frame[["TIMESTAMP"]], on="TIMESTAMP")
    assert len(merged) == 3


def test_normalisation_does_not_mutate_the_caller_frame():
    frame = valid_frame()
    frame["TIMESTAMP"] = frame["TIMESTAMP"].astype(str)
    before = frame["TIMESTAMP"].dtype
    ForecastBundle("load", frame, "MW", "grid-a", "24h", "actual")
    assert frame["TIMESTAMP"].dtype == before
    assert not pd.api.types.is_datetime64_any_dtype(frame["TIMESTAMP"])


def test_residuals_require_actuals():
    assert ForecastBundle("s", valid_frame(), "MW", "z", "24h").residuals() is None
    bundle = ForecastBundle("l", valid_frame(), "MW", "z", "24h", "actual")
    np.testing.assert_allclose(bundle.residuals(), 49.0 - 49.0)


def test_physical_bounds_must_be_ordered():
    with pytest.raises(ForecastContractError):
        ForecastBundle("s", valid_frame(), "MW", "z", "24h",
                       physical_min=10.0, physical_max=1.0)


def test_replace_frame_revalidates_and_keeps_bounds():
    bundle = ForecastBundle("s", valid_frame(), "MW", "z", "24h", physical_min=0.0, physical_max=99.0)
    other = bundle.replace_frame(valid_frame(4))
    assert len(other.frame) == 4 and other.physical_max == 99.0
    with pytest.raises(ForecastContractError):
        bundle.replace_frame(valid_frame().assign(q50=np.nan))


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("mutate", [
    lambda f: f.drop(columns="q50"),
    lambda f: f.assign(q50=np.nan),
    lambda f: f.assign(q50=-1),                                    # crossing quantiles
    lambda f: f.assign(TIMESTAMP=["bad", "2024-01-01", "2024-01-02"]),
    lambda f: f.assign(TIMESTAMP=pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-01"])),
    lambda f: f.assign(TIMESTAMP=pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])),
    lambda f: f.iloc[0:0],
])
def test_contract_rejects_invalid_frame(mutate):
    with pytest.raises(ForecastContractError):
        ForecastBundle("solar", mutate(valid_frame()), "kW", "zone-1", "24h")
