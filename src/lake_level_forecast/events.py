from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

EVENT_COLUMNS = [
    "event_id",
    "event_start_utc",
    "event_end_utc",
    "window_start_utc",
    "window_end_utc",
    "peak_time_utc",
    "peak_water_level_ft",
    "duration_hours",
    "source_event_count",
]


@dataclass(frozen=True)
class EventSettings:
    threshold_ft: float = 2.0
    merge_gap_hours: float = 6.0
    padding_hours: float = 72.0


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _merge_overlapping_windows(events: pd.DataFrame) -> pd.DataFrame:
    """Merge padded windows that overlap.

    A long lake setup can repeatedly dip just below 2 ft, creating many tiny
    exceedance events. Once 72 hours of padding is added, those windows often
    overlap heavily. Treat those as one compound high-water episode so we do
    not download the same KNEW wind window over and over.
    """
    if events.empty:
        return _empty_events()

    events = events.sort_values("window_start_utc").reset_index(drop=True)
    merged = []
    current = events.iloc[0].to_dict()
    current["source_event_count"] = 1

    for _, row in events.iloc[1:].iterrows():
        row_dict = row.to_dict()
        if row_dict["window_start_utc"] <= current["window_end_utc"]:
            current["event_end_utc"] = max(current["event_end_utc"], row_dict["event_end_utc"])
            current["window_end_utc"] = max(current["window_end_utc"], row_dict["window_end_utc"])
            current["duration_hours"] = (current["event_end_utc"] - current["event_start_utc"]).total_seconds() / 3600
            current["source_event_count"] += 1
            if row_dict["peak_water_level_ft"] > current["peak_water_level_ft"]:
                current["peak_water_level_ft"] = row_dict["peak_water_level_ft"]
                current["peak_time_utc"] = row_dict["peak_time_utc"]
        else:
            current["event_id"] = f"NWCL1_{current['event_start_utc']:%Y%m%d_%H%M}"
            merged.append(current)
            current = row_dict
            current["source_event_count"] = 1

    current["event_id"] = f"NWCL1_{current['event_start_utc']:%Y%m%d_%H%M}"
    merged.append(current)
    return pd.DataFrame(merged)[EVENT_COLUMNS].sort_values("event_start_utc").reset_index(drop=True)


def find_exceedance_events(water: pd.DataFrame, settings: EventSettings) -> pd.DataFrame:
    df = water.copy()
    if df.empty:
        return _empty_events()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc")
    exceed = df[df["water_level_ft"] >= settings.threshold_ft].copy()
    if exceed.empty:
        return _empty_events()

    gap = pd.Timedelta(hours=settings.merge_gap_hours)
    exceed["new_event"] = exceed["datetime_utc"].diff().gt(gap).fillna(True)
    exceed["event_num"] = exceed["new_event"].cumsum()

    rows = []
    pad = pd.Timedelta(hours=settings.padding_hours)
    for _, group in exceed.groupby("event_num"):
        start = group["datetime_utc"].min()
        end = group["datetime_utc"].max()
        peak_idx = group["water_level_ft"].idxmax()
        rows.append(
            {
                "event_id": f"NWCL1_{start:%Y%m%d_%H%M}",
                "event_start_utc": start,
                "event_end_utc": end,
                "window_start_utc": start - pad,
                "window_end_utc": end + pad,
                "peak_time_utc": group.loc[peak_idx, "datetime_utc"],
                "peak_water_level_ft": group.loc[peak_idx, "water_level_ft"],
                "duration_hours": (end - start).total_seconds() / 3600,
            }
        )

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
