import pandas as pd

from src.phase_c_precheck import _gap_hours_total, _non_1h_steps


def test_gap_and_step_metrics():
    ts = pd.date_range("2024-01-01", periods=10, freq="h")
    assert _gap_hours_total(ts) == 0
    assert _non_1h_steps(ts) == 0

    # Remove one timestamp to simulate a gap
    ts2 = ts.delete(5)
    assert _gap_hours_total(ts2) == 1

    # Create a non-1h step
    ts3 = pd.DatetimeIndex(list(ts[:5]) + [ts[5] + pd.Timedelta(hours=2)] + list(ts[6:]))
    assert _non_1h_steps(ts3) >= 1

