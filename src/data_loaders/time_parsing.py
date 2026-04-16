from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from datetime import timedelta
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
    Robustly parse a full timestamp series by resolving ambiguous md=3 cases
    (e.g. "1292010" => 1/29/2010 vs 12/9/2010) using a continuity objective.

    The load track is hourly; for each ambiguous timestamp we choose the candidate
    that best preserves an overall 1-hour step sequence across the entire series.
    """
    ts_list = list(ts_series)

    def candidates_for(ts: str) -> list[datetime]:
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

        if len(md) == 4:
            dt = try_build(int(md[:2]), int(md[2:]))
            if dt is None:
                raise ValueError(f"Invalid md=4 parse for {ts}")
            return [dt]

        if len(md) == 2:
            dt = try_build(int(md[0]), int(md[1]))
            if dt is None:
                raise ValueError(f"Invalid md=2 parse for {ts}")
            return [dt]

        if len(md) != 3:
            raise ValueError(f"Unexpected load timestamp format: {ts}")

        cand_a = try_build(int(md[0]), int(md[1:]))   # 1-digit month + 2-digit day
        cand_b = try_build(int(md[:2]), int(md[2]))   # 2-digit month + 1-digit day
        cands = [c for c in (cand_a, cand_b) if c is not None]
        if not cands:
            raise ValueError(f"Both md=3 parses invalid for {ts}")
        # If only one is valid (e.g. month 13), it's effectively unambiguous.
        return cands

    all_cands = [candidates_for(ts) for ts in ts_list]

    # Viterbi-style DP over candidates to minimize deviation from hourly steps.
    # State size is at most 2 per timestep, so this stays fast for large series.
    one_hour = timedelta(hours=1)
    big = 10**12
    dp: list[list[float]] = []
    prev: list[list[int]] = []

    dp.append([0.0 for _ in all_cands[0]])
    prev.append([-1 for _ in all_cands[0]])

    def step_cost(dt_prev: datetime, dt_cur: datetime) -> float:
        diff = dt_cur - dt_prev
        if diff == one_hour:
            return 0.0
        # Strongly discourage non-forward time.
        if diff.total_seconds() <= 0:
            return float(big)
        # Penalize distance from 1 hour in hours.
        return float(abs(diff.total_seconds() / 3600.0 - 1.0))

    for i in range(1, len(all_cands)):
        dp_i = [float(big) for _ in all_cands[i]]
        prev_i = [-1 for _ in all_cands[i]]
        for j, dt_cur in enumerate(all_cands[i]):
            best_cost = float(big)
            best_k = -1
            for k, dt_prev in enumerate(all_cands[i - 1]):
                cost = dp[i - 1][k] + step_cost(dt_prev, dt_cur)
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            dp_i[j] = best_cost
            prev_i[j] = best_k
        dp.append(dp_i)
        prev.append(prev_i)

    # Backtrack best path
    last_j = min(range(len(dp[-1])), key=lambda j: dp[-1][j])
    chosen: list[datetime] = [all_cands[-1][last_j]]
    j = last_j
    for i in range(len(all_cands) - 1, 0, -1):
        j = prev[i][j]
        if j < 0:
            break
        chosen.append(all_cands[i - 1][j])
    chosen.reverse()

    idx = pd.DatetimeIndex(chosen)
    diffs = pd.Series(idx).diff().dropna()
    non_hourly = int((diffs != pd.Timedelta(hours=1)).sum())

    # Legacy field retained for compatibility; strategy is now dynamic.
    return LoadTimestampParseResult(
        timestamps=idx,
        prefer_two_digit_month_for_len3=False,
        non_hourly_steps=non_hourly,
    )


def parse_gefcom_solar_timestamps(ts_series: Iterable[str]) -> pd.DatetimeIndex:
    """
    Solar timestamps are consistently formatted like: "20120401 01:00"
    => %Y%m%d %H:%M (deterministic).
    """
    return pd.to_datetime(list(ts_series), format="%Y%m%d %H:%M", errors="raise")
