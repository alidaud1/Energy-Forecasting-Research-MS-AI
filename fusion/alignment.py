"""Cyclical alignment of non-coincident load and solar test windows.

GEFCom2014 publishes the load and solar tracks as independent competitions.
Their released test windows do not overlap:

    load   2010-01-01 00:00 .. 2010-09-30 23:00   (6552 hourly steps)
    solar  2014-06-01 01:00 .. 2014-07-01 00:00   (720 hourly steps per zone)

There is therefore no shared calendar on which a genuine grid-level net load
could be measured, and a literal timestamp join returns nothing.  The agreed
resolution (supervisor sign-off, 2026-08-24) is to demonstrate the *method* on
a cyclically repeated solar profile rather than to claim a real coincident
series.  This module implements that assumption in one auditable place so it
can be cited in the methodology and reproduced exactly.

    solar_index = (load_index + phase) mod n_solar

Two corrections determine ``phase``, and both matter:

1.  CLOCK BASIS.  The solar track records three Australian zones stamped in
    UTC: generation runs 22:00--07:00 UTC and peaks at 02:00 UTC.  The load
    track is a utility series on its own local clock, peaking at 20:00 and
    troughing at 05:00 -- an ordinary local-time demand curve.  Pairing the raw
    hour labels would place the entire solar generation curve inside the load's
    overnight trough, producing zero solar through the working day and full
    output at midnight.  Every ramp and reserve figure computed from that would
    be meaningless.  The solar series is therefore shifted to its own local
    time before alignment.  The offset is measured from the data (the circular
    mean of the generation profile is moved to solar noon) rather than assumed,
    and the measured value is recorded.

2.  DIURNAL PHASE.  Load begins at 00:00 and solar at 01:00, so a bare
    ``mod n_solar`` would still slip the curve one hour.  Because n_solar (720)
    is an exact multiple of 24, a constant phase repairs this.

WHAT THIS IS NOT: the reused solar profile carries the calendar of June 2014,
not of the load year.  Season, weather and day of week are not synchronised,
and the two series describe different continents.  Results demonstrate the
coordination mechanism; they are not an empirical net-load forecast for any
real system and must not be reported as one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast_contract import ForecastBundle, ForecastContractError, QUANTILE_COLUMNS

HOURS_PER_DAY = 24
SOLAR_NOON = 12


def solar_local_offset(solar: ForecastBundle) -> int:
    """Hours to add to the solar stamps to reach the array's local clock.

    Derived from the shape of the generation curve itself: the circular mean of
    median output over hour of day is the profile's centre of mass, and for a
    daylight-driven series that centre is solar noon.  The offset is whatever
    moves it there.  Measuring this beats hardcoding a timezone because it stays
    correct if the underlying artifact is ever regenerated for other zones.
    """
    hours = solar.timestamps().hour.to_numpy()
    output = solar.frame["q50"].to_numpy(dtype=float)
    profile = np.array([output[hours == h].mean() if (hours == h).any() else 0.0
                        for h in range(HOURS_PER_DAY)])
    profile = np.clip(profile, 0.0, None)
    if not profile.any():
        return 0
    angles = 2.0 * np.pi * np.arange(HOURS_PER_DAY) / HOURS_PER_DAY
    centre = np.arctan2((profile * np.sin(angles)).sum(),
                        (profile * np.cos(angles)).sum())
    centre_hour = (centre / (2.0 * np.pi) * HOURS_PER_DAY) % HOURS_PER_DAY
    return int(round(SOLAR_NOON - centre_hour)) % HOURS_PER_DAY


def align_cyclically(load: ForecastBundle, solar: ForecastBundle, *,
                     match_hour_of_day: bool = True,
                     solar_utc_offset_hours: int | None = None) -> ForecastBundle:
    """Repeat ``solar`` across ``load``'s timeline under the documented rule.

    Returns a new solar bundle carrying load's timestamps.  Its metadata records
    the rule, both corrections and the verification result, so the assumption
    travels with the data into the run manifest.
    """
    if len(solar.frame) == 0:
        raise ForecastContractError("solar bundle is empty; nothing to align")
    load_ts, solar_ts = load.timestamps(), solar.timestamps()
    n_solar = len(solar_ts)
    offset = (solar_local_offset(solar) if solar_utc_offset_hours is None
              else int(solar_utc_offset_hours) % HOURS_PER_DAY)
    if match_hour_of_day and n_solar % HOURS_PER_DAY == 0:
        phase = int((int(load_ts.hour[0]) - int(solar_ts.hour[0]) - offset) % HOURS_PER_DAY)
        note = "solar shifted to local clock, then hour of day aligned"
    elif match_hour_of_day:
        phase, note = 0, (f"solar window ({n_solar} steps) is not a whole number of "
                          "days; hour-of-day alignment skipped")
    else:
        phase, note = 0, "hour-of-day alignment disabled by caller"
    index = (np.arange(len(load_ts)) + phase) % n_solar
    frame = solar.frame.iloc[index].reset_index(drop=True).copy()
    frame["TIMESTAMP"] = load_ts.to_numpy()
    solar_local = (solar_ts.hour.to_numpy()[index] + offset) % HOURS_PER_DAY
    verified = bool(np.array_equal(load_ts.hour.to_numpy(), solar_local))
    metadata = dict(solar.metadata)
    metadata.update({
        "alignment": "cyclical",
        "alignment_rule": f"solar_index = (load_index + {phase}) mod {n_solar}",
        "alignment_phase": phase,
        "alignment_note": note,
        "solar_local_offset_hours": offset,
        "hour_of_day_aligned": verified,
        "solar_window": f"{solar_ts.min()} .. {solar_ts.max()}",
        "load_window": f"{load_ts.min()} .. {load_ts.max()}",
        "solar_profile_repeats": round(len(load_ts) / n_solar, 4),
        "caveat": ("solar profile is reused cyclically; season, weather and day of "
                   "week are not synchronised with the load year"),
    })
    return ForecastBundle(solar.source, frame, solar.unit, solar.location,
                          solar.horizon, solar.actual_column, metadata,
                          solar.physical_min, solar.physical_max)


def describe_overlap(load: ForecastBundle, solar: ForecastBundle) -> dict:
    """Report calendar overlap, used to decide whether alignment is needed."""
    left, right = set(load.timestamps()), set(solar.timestamps())
    shared = left & right
    return {"load_steps": len(left), "solar_steps": len(right),
            "shared_steps": len(shared), "requires_alignment": len(shared) == 0}
