from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EventSettings:
    threshold_ft: float = 2.0
    merge_gap_hours: float = 6.0
    padding_hours: float = 72.0


def find_exceedance_events(water: pd.DataFrame, settings: EventSettings) -> pd.DataFrame:
    df = water.copy()
    if df.empty:
        return pd.DataFrame()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc")
    exceed = df[df["water_level_ft"] >= settings.threshold_ft].copy()
    if exceed.empty:
        return pd.DataFrame()
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
    return pd.DataFrame(rows).sort_values("event_start_utc").reset_index(drop=True)


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
