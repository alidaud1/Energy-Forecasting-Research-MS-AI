from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class LoadTimestampParseResult:
    timestamps: pd.DatetimeIndex
    prefer_two_digit_month_for_len3: bool
    non_hourly_steps: int


def _parse_load_timestamp_single(ts: str, prefer_two_digit_len3: bool) -> datetime:
    """
    Parse a single GEFCom load timestamp.

    Observed format examples:
      - "112001 1:00"      -> (date_part="112001",  time_part="1:00")
      - "1112011 1:00"     -> (date_part="1112011", time_part="1:00")
      - "11272011 0:00"
      - "1012010 0:00"

    date_part is month+day+year without separators; year is last 4 digits.
    For md length==3, ambiguity exists: e.g. "1112011" can be 1/11/2011 or 11/1/2011.
    """
    date_part, time_part = ts.split()
    year = int(date_part[-4:])
    md = date_part[:-4]

    hour_str, min_str = time_part.split(":")
    hour = int(hour_str)
    minute = int(min_str)

    def try_build(m: int, d: int) -> datetime | None:
        try:
            return datetime(year, m, d, hour, minute)
        except ValueError:
            return None

    # Unambiguous cases
    if len(md) == 4:
        dt = try_build(int(md[:2]), int(md[2:]))
        if dt is None:
            raise ValueError(f"Invalid md=4 parse for {ts}")
        return dt

    if len(md) == 2:
        dt = try_build(int(md[0]), int(md[1]))
        if dt is None:
            raise ValueError(f"Invalid md=2 parse for {ts}")
        return dt

    # Ambiguous case: len(md) == 3
    if len(md) != 3:
        raise ValueError(f"Unexpected load timestamp format: {ts}")

    # Candidate A: 1-digit month + 2-digit day
    cand_a = try_build(int(md[0]), int(md[1:]))
    # Candidate B: 2-digit month + 1-digit day
    cand_b = try_build(int(md[:2]), int(md[2]))

    preferred = cand_b if prefer_two_digit_len3 else cand_a
    fallback = cand_a if prefer_two_digit_len3 else cand_b

    if preferred is not None:
        return preferred
    if fallback is not None:
        return fallback

    raise ValueError(f"Both md=3 parses invalid for {ts}")


def parse_gefcom_load_timestamps(ts_series: Iterable[str]) -> LoadTimestampParseResult:
    """
    Robustly parse a full timestamp series by selecting the md=3 preference
    that yields the fewest non-hourly steps (i.e., best continuity).
    """
    ts_list = list(ts_series)

    def parse_with_pref(pref: bool) -> pd.DatetimeIndex:
        dts = [_parse_load_timestamp_single(ts, pref) for ts in ts_list]
        return pd.DatetimeIndex(dts)

    idx_a = parse_with_pref(pref=False)
    idx_b = parse_with_pref(pref=True)

    def non_hourly_steps(idx: pd.DatetimeIndex) -> int:
        diffs = pd.Series(idx).diff().dropna()
        return int((diffs != pd.Timedelta(hours=1)).sum())

    n_a = non_hourly_steps(idx_a)
    n_b = non_hourly_steps(idx_b)

    if n_b < n_a:
        return LoadTimestampParseResult(
            timestamps=idx_b,
            prefer_two_digit_month_for_len3=True,
            non_hourly_steps=n_b,
        )
    return LoadTimestampParseResult(
        timestamps=idx_a,
        prefer_two_digit_month_for_len3=False,
        non_hourly_steps=n_a,
    )


def parse_gefcom_solar_timestamps(ts_series: Iterable[str]) -> pd.DatetimeIndex:
    """
    Solar timestamps are consistently formatted like: "20120401 01:00"
    => %Y%m%d %H:%M (deterministic).
    """
    return pd.to_datetime(list(ts_series), format="%Y%m%d %H:%M", errors="raise")

