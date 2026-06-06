from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

EVENT_COLUMNS = [
    "event_id",
    "event_start_utc",
    "event_end_utc",
    "exceed_start_utc",
    "exceed_end_utc",
    "window_start_utc",
    "window_end_utc",
    "peak_time_utc",
    "peak_water_level_ft",
    "start_water_level_ft",
    "end_water_level_ft",
    "rise_ft",
    "duration_hours",
    "source_event_count",
]


@dataclass(frozen=True)
class EventSettings:
    threshold_ft: float = 2.0
    merge_gap_hours: float = 6.0
    padding_hours: float = 72.0
    smooth_hours: float = 1.0
    start_search_hours: float = 96.0
    end_search_hours: float = 96.0
    rate_threshold_ft_per_hr: float = 0.02
    quiet_hours: float = 3.0
    safety_buffer_hours: float = 6.0


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _prep_water(water: pd.DataFrame, smooth_hours: float) -> pd.DataFrame:
    df = water.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc").drop_duplicates("datetime_utc").reset_index(drop=True)
    if df.empty:
        return df
    median_step_hr = df["datetime_utc"].diff().dt.total_seconds().dropna().median() / 3600
    if not median_step_hr or pd.isna(median_step_hr) or median_step_hr <= 0:
        median_step_hr = 0.1
    smooth_n = max(1, int(round(smooth_hours / median_step_hr)))
    df["smooth_water_level_ft"] = df["water_level_ft"].rolling(smooth_n, center=True, min_periods=1).mean()
    dt_hr = df["datetime_utc"].diff().dt.total_seconds() / 3600
    df["rate_ft_per_hr"] = (df["smooth_water_level_ft"].diff() / dt_hr).fillna(0)
    return df


def _find_rise_start(df: pd.DataFrame, first_exceed_time: pd.Timestamp, settings: EventSettings) -> pd.Timestamp:
    search_start = first_exceed_time - pd.Timedelta(hours=settings.start_search_hours)
    segment = df[(df["datetime_utc"] >= search_start) & (df["datetime_utc"] <= first_exceed_time)].copy()
    if segment.empty:
        return first_exceed_time - pd.Timedelta(hours=settings.safety_buffer_hours)
    median_step_hr = segment["datetime_utc"].diff().dt.total_seconds().dropna().median() / 3600
    if not median_step_hr or pd.isna(median_step_hr) or median_step_hr <= 0:
        median_step_hr = 0.1
    quiet_n = max(1, int(round(settings.quiet_hours / median_step_hr)))
    for i in range(len(segment) - 1, quiet_n - 1, -1):
        rates = segment.iloc[i - quiet_n : i]["rate_ft_per_hr"]
        if bool((rates <= settings.rate_threshold_ft_per_hr).all()):
            return segment.iloc[i]["datetime_utc"] - pd.Timedelta(hours=settings.safety_buffer_hours)
    min_idx = segment["smooth_water_level_ft"].idxmin()
    return segment.loc[min_idx, "datetime_utc"] - pd.Timedelta(hours=settings.safety_buffer_hours)


def _find_recovery_end(df: pd.DataFrame, last_exceed_time: pd.Timestamp, settings: EventSettings) -> pd.Timestamp:
    search_end = last_exceed_time + pd.Timedelta(hours=settings.end_search_hours)
    segment = df[(df["datetime_utc"] >= last_exceed_time) & (df["datetime_utc"] <= search_end)].copy()
    if segment.empty:
        return last_exceed_time + pd.Timedelta(hours=settings.safety_buffer_hours)
    median_step_hr = segment["datetime_utc"].diff().dt.total_seconds().dropna().median() / 3600
    if not median_step_hr or pd.isna(median_step_hr) or median_step_hr <= 0:
        median_step_hr = 0.1
    quiet_n = max(1, int(round(settings.quiet_hours / median_step_hr)))
    below = segment[segment["smooth_water_level_ft"] < settings.threshold_ft]
    start_idx = below.index.min() if not below.empty else segment.index.min()
    segment2 = segment.loc[start_idx:].copy()
    for i in range(quiet_n, len(segment2)):
        rates = segment2.iloc[i - quiet_n : i]["rate_ft_per_hr"]
        if bool((rates >= -settings.rate_threshold_ft_per_hr).all()):
            return segment2.iloc[i]["datetime_utc"] + pd.Timedelta(hours=settings.safety_buffer_hours)
    min_idx = segment["smooth_water_level_ft"].idxmin()
    return segment.loc[min_idx, "datetime_utc"] + pd.Timedelta(hours=settings.safety_buffer_hours)


def _merge_overlapping_windows(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return _empty_events()
    events = events.sort_values("window_start_utc").reset_index(drop=True)
    merged = []
    current = events.iloc[0].to_dict()
    current["source_event_count"] = int(current.get("source_event_count", 1))
    for _, row in events.iloc[1:].iterrows():
        row_dict = row.to_dict()
        if row_dict["window_start_utc"] <= current["window_end_utc"]:
            current["event_start_utc"] = min(current["event_start_utc"], row_dict["event_start_utc"])
            current["event_end_utc"] = max(current["event_end_utc"], row_dict["event_end_utc"])
            current["exceed_start_utc"] = min(current["exceed_start_utc"], row_dict["exceed_start_utc"])
            current["exceed_end_utc"] = max(current["exceed_end_utc"], row_dict["exceed_end_utc"])
            current["window_start_utc"] = min(current["window_start_utc"], row_dict["window_start_utc"])
            current["window_end_utc"] = max(current["window_end_utc"], row_dict["window_end_utc"])
            current["duration_hours"] = (current["event_end_utc"] - current["event_start_utc"]).total_seconds() / 3600
            current["source_event_count"] += int(row_dict.get("source_event_count", 1))
            if row_dict["peak_water_level_ft"] > current["peak_water_level_ft"]:
                current["peak_water_level_ft"] = row_dict["peak_water_level_ft"]
                current["peak_time_utc"] = row_dict["peak_time_utc"]
            current["start_water_level_ft"] = min(current["start_water_level_ft"], row_dict["start_water_level_ft"])
            current["end_water_level_ft"] = min(current["end_water_level_ft"], row_dict["end_water_level_ft"])
            current["rise_ft"] = current["peak_water_level_ft"] - current["start_water_level_ft"]
        else:
            current["event_id"] = f"NWCL1_{current['event_start_utc']:%Y%m%d_%H%M}"
            merged.append(current)
            current = row_dict
            current["source_event_count"] = int(current.get("source_event_count", 1))
    current["event_id"] = f"NWCL1_{current['event_start_utc']:%Y%m%d_%H%M}"
    merged.append(current)
    return pd.DataFrame(merged)[EVENT_COLUMNS].sort_values("event_start_utc").reset_index(drop=True)


def find_exceedance_events(water: pd.DataFrame, settings: EventSettings) -> pd.DataFrame:
    df = _prep_water(water, settings.smooth_hours)
    if df.empty:
        return _empty_events()
    exceed = df[df["smooth_water_level_ft"] >= settings.threshold_ft].copy()
    if exceed.empty:
        return _empty_events()
    gap = pd.Timedelta(hours=settings.merge_gap_hours)
    exceed["new_event"] = exceed["datetime_utc"].diff().gt(gap).fillna(True)
    exceed["event_num"] = exceed["new_event"].cumsum()
    rows = []
    for _, group in exceed.groupby("event_num"):
        exceed_start = group["datetime_utc"].min()
        exceed_end = group["datetime_utc"].max()
        event_start = _find_rise_start(df, exceed_start, settings)
        event_end = _find_recovery_end(df, exceed_end, settings)
        window_start = max(df["datetime_utc"].min(), event_start)
        window_end = min(df["datetime_utc"].max(), event_end)
        event_df = df[(df["datetime_utc"] >= window_start) & (df["datetime_utc"] <= window_end)].copy()
        if event_df.empty:
            continue
        peak_idx = event_df["smooth_water_level_ft"].idxmax()
        start_level = float(event_df.iloc[0]["smooth_water_level_ft"])
        end_level = float(event_df.iloc[-1]["smooth_water_level_ft"])
        peak_level = float(event_df.loc[peak_idx, "smooth_water_level_ft"])
        rows.append(
            {
                "event_id": f"NWCL1_{window_start:%Y%m%d_%H%M}",
                "event_start_utc": window_start,
                "event_end_utc": window_end,
                "exceed_start_utc": exceed_start,
                "exceed_end_utc": exceed_end,
                "window_start_utc": window_start,
                "window_end_utc": window_end,
                "peak_time_utc": event_df.loc[peak_idx, "datetime_utc"],
                "peak_water_level_ft": peak_level,
                "start_water_level_ft": start_level,
                "end_water_level_ft": end_level,
                "rise_ft": peak_level - start_level,
                "duration_hours": (window_end - window_start).total_seconds() / 3600,
                "source_event_count": 1,
            }
        )
    if not rows:
        return _empty_events()
    raw_events = pd.DataFrame(rows).sort_values("event_start_utc").reset_index(drop=True)
    return _merge_overlapping_windows(raw_events)


def subset_water_to_event_windows(water: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    frames = []
    water = water.copy()
    water["datetime_utc"] = pd.to_datetime(water["datetime_utc"], utc=True)
    for _, ev in events.iterrows():
        mask = (water["datetime_utc"] >= ev["window_start_utc"]) & (water["datetime_utc"] <= ev["window_end_utc"])
        tmp = water.loc[mask].copy()
        tmp["event_id"] = ev["event_id"]
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
