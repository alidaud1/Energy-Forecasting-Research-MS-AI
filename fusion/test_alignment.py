import numpy as np
import pandas as pd
import pytest

from fusion.alignment import align_cyclically, describe_overlap, solar_local_offset
from fusion.forecast_contract import ForecastContractError


def test_offset_is_measured_from_the_generation_curve(solar_shaped):
    """A profile already peaking at local noon needs no shift."""
    assert solar_local_offset(solar_shaped(peak_hour=12)) == 0


@pytest.mark.parametrize("peak, expected", [(2, 10), (6, 6), (18, 18), (12, 0)])
def test_offset_recovers_an_arbitrary_clock_shift(solar_shaped, peak, expected):
    assert solar_local_offset(solar_shaped(peak_hour=peak)) == expected


def test_overlap_detection(normal_bundle, solar_shaped):
    load = normal_bundle("load", 150, 10, n=48)
    assert describe_overlap(load, solar_shaped())["requires_alignment"] is True
    assert describe_overlap(load, load)["shared_steps"] == 48


def test_alignment_spans_the_load_timeline(normal_bundle, solar_shaped):
    load = normal_bundle("load", 150, 10, n=6552)
    aligned = align_cyclically(load, solar_shaped())
    assert len(aligned.frame) == 6552
    assert aligned.timestamps().equals(load.timestamps())
    assert aligned.metadata["solar_profile_repeats"] == pytest.approx(6552 / 720, abs=1e-3)


def test_alignment_puts_solar_generation_in_daylight(normal_bundle, solar_shaped):
    """The whole point: solar must not land in the load's overnight trough."""
    load = normal_bundle("load", 150, 10, n=6552)
    aligned = align_cyclically(load, solar_shaped(peak_hour=2))   # UTC-stamped array
    hours = aligned.timestamps().hour
    by_hour = aligned.frame.groupby(hours)["q50"].mean()
    assert by_hour.idxmax() == 12
    assert by_hour[0] == pytest.approx(0.0, abs=1e-9)
    assert aligned.metadata["hour_of_day_aligned"] is True
    assert aligned.metadata["solar_local_offset_hours"] == 10


def test_naive_alignment_would_have_misplaced_the_curve(normal_bundle, solar_shaped):
    load = normal_bundle("load", 150, 10, n=6552)
    naive = align_cyclically(load, solar_shaped(peak_hour=2), solar_utc_offset_hours=0)
    by_hour = naive.frame.groupby(naive.timestamps().hour)["q50"].mean()
    assert by_hour.idxmax() != 12          # this is the bug the offset fixes


def test_alignment_records_its_assumption(normal_bundle, solar_shaped):
    aligned = align_cyclically(normal_bundle("load", 150, 10, n=100), solar_shaped())
    assert aligned.metadata["alignment"] == "cyclical"
    assert "mod 720" in aligned.metadata["alignment_rule"]
    assert "cyclically" in aligned.metadata["caveat"]


def test_alignment_preserves_bounds_and_actuals(normal_bundle, solar_shaped):
    aligned = align_cyclically(normal_bundle("load", 150, 10, n=100), solar_shaped())
    assert aligned.physical_min == 0.0 and aligned.physical_max == 60.0


def test_alignment_rejects_an_empty_solar_bundle(normal_bundle, solar_shaped):
    solar = solar_shaped()
    empty = object.__new__(type(solar))
    object.__setattr__(empty, "frame", solar.frame.iloc[0:0])
    with pytest.raises(ForecastContractError):
        align_cyclically(normal_bundle("load", 150, 10), empty)


def test_non_whole_day_window_skips_hour_alignment(normal_bundle, solar_shaped):
    solar = solar_shaped(n=700)                      # not a multiple of 24
    aligned = align_cyclically(normal_bundle("load", 150, 10, n=100), solar)
    assert aligned.metadata["alignment_phase"] == 0
    assert "not a whole number of days" in aligned.metadata["alignment_note"]
