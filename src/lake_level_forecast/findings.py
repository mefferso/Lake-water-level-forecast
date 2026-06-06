from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _round_numeric(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    return out


def _direction_bin(direction: float) -> str:
    if pd.isna(direction):
        return "missing"
    d = direction % 360
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((d + 22.5) % 360) // 45)
    return labels[idx]


def _speed_bin(speed: float) -> str:
    if pd.isna(speed):
        return "missing"
    bins = [0, 10, 15, 20, 25, 30, 35, 1000]
    labels = ["0-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35+"]
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        if lo <= speed < hi:
            return label
    return "missing"


def _rise_bin(rise: float) -> str:
    if pd.isna(rise):
        return "missing"
    lower = np.floor(rise * 2) / 2
    upper = lower + 0.5
    return f"{lower:.1f}-{upper:.1f}"


def _mean_direction_deg(directions: pd.Series) -> float:
    vals = pd.to_numeric(directions, errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return np.nan
    rad = np.deg2rad(vals)
    s = np.sin(rad).mean()
    c = np.cos(rad).mean()
    return float((np.rad2deg(np.arctan2(s, c)) + 360) % 360)


def _event_peak_time(ev: pd.Series) -> pd.Timestamp:
    if "surge_peak_time_utc" in ev.index and pd.notna(ev["surge_peak_time_utc"]):
        return ev["surge_peak_time_utc"]
    return ev["peak_time_utc"]


def _event_peak_level(ev: pd.Series) -> float:
    if "surge_peak_water_level_ft" in ev.index and pd.notna(ev["surge_peak_water_level_ft"]):
        return float(ev["surge_peak_water_level_ft"])
    return float(ev["peak_water_level_ft"])


def make_event_wind_summary(events: pd.DataFrame, wind: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    if events.empty or wind.empty:
        return pd.DataFrame()

    events = events.copy()
    wind = wind.copy()
    time_cols = [
        "event_start_utc",
        "event_end_utc",
        "peak_time_utc",
        "surge_peak_time_utc",
        "water_level_peak_time_utc",
        "exceed_start_utc",
        "exceed_end_utc",
    ]
    for col in time_cols:
        if col in events.columns:
            events[col] = pd.to_datetime(events[col], utc=True)
    wind["datetime_utc"] = pd.to_datetime(wind["datetime_utc"], utc=True)

    rows = []
    for _, ev in events.iterrows():
        event_id = ev["event_id"]
        w = wind[wind["event_id"] == event_id].copy()
        if w.empty:
            continue

        peak_time = _event_peak_time(ev)
        peak_level = _event_peak_level(ev)
        setup = w[(w["datetime_utc"] >= ev["event_start_utc"]) & (w["datetime_utc"] <= peak_time)]
        if setup.empty:
            setup = w

        rows.append(
            {
                "event_id": event_id,
                "event_start_utc": ev["event_start_utc"],
                "surge_peak_time_utc": peak_time,
                "event_end_utc": ev["event_end_utc"],
                "rise_ft": ev["rise_ft"],
                "surge_peak_water_level_ft": peak_level,
                "water_level_peak_ft": ev.get("water_level_peak_ft", peak_level),
                "post_surge_dip_ft": ev.get("post_surge_dip_ft", np.nan),
                "duration_to_surge_peak_hr": (peak_time - ev["event_start_utc"]).total_seconds() / 3600,
                "mean_wind_speed_kt": setup["wind_speed_kt"].mean(),
                "max_wind_speed_kt": setup["wind_speed_kt"].max(),
                "mean_wind_gust_kt": setup["wind_gust_kt"].mean(),
                "max_wind_gust_kt": setup["wind_gust_kt"].max(),
                "mean_wind_dir_deg": _mean_direction_deg(setup["wind_dir_deg"]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["mean_wind_dir_bin"] = out["mean_wind_dir_deg"].apply(_direction_bin)
    out["mean_wind_speed_bin_kt"] = out["mean_wind_speed_kt"].apply(_speed_bin)
    out["rise_bin_ft"] = out["rise_ft"].apply(_rise_bin)
    return _round_numeric(out.sort_values("rise_ft", ascending=False), decimals)


def make_wind_rise_lookup(event_summary: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    if event_summary.empty:
        return event_summary
    group_cols = ["mean_wind_dir_bin", "mean_wind_speed_bin_kt"]
    rows = []
    for keys, g in event_summary.groupby(group_cols):
        direction, speed_bin = keys
        rows.append(
            {
                "wind_direction_bin": direction,
                "mean_wind_speed_bin_kt": speed_bin,
                "event_count": len(g),
                "median_rise_ft": g["rise_ft"].median(),
                "mean_rise_ft": g["rise_ft"].mean(),
                "p75_rise_ft": g["rise_ft"].quantile(0.75),
                "max_rise_ft": g["rise_ft"].max(),
                "median_surge_peak_water_level_ft": g["surge_peak_water_level_ft"].median(),
                "median_duration_to_surge_peak_hr": g["duration_to_surge_peak_hr"].median(),
            }
        )
    out = pd.DataFrame(rows)
    direction_order = {"N": 0, "NE": 1, "E": 2, "SE": 3, "S": 4, "SW": 5, "W": 6, "NW": 7, "missing": 8}
    speed_order = {"0-10": 0, "10-15": 1, "15-20": 2, "20-25": 3, "25-30": 4, "30-35": 5, "35+": 6, "missing": 7}
    out["_dir_order"] = out["wind_direction_bin"].map(direction_order).fillna(99)
    out["_speed_order"] = out["mean_wind_speed_bin_kt"].map(speed_order).fillna(99)
    out = out.sort_values(["_dir_order", "_speed_order"]).drop(columns=["_dir_order", "_speed_order"])
    return _round_numeric(out, decimals)


def make_rise_threshold_findings(event_summary: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    if event_summary.empty:
        return event_summary
    thresholds = np.arange(0.5, max(0.5, float(event_summary["rise_ft"].max())) + 0.5, 0.5)
    rows = []
    for threshold in thresholds:
        g = event_summary[event_summary["rise_ft"] >= threshold]
        if g.empty:
            continue
        dir_counts = g["mean_wind_dir_bin"].value_counts()
        rows.append(
            {
                "rise_threshold_ft": threshold,
                "event_count": len(g),
                "median_mean_wind_speed_kt": g["mean_wind_speed_kt"].median(),
                "median_max_wind_speed_kt": g["max_wind_speed_kt"].median(),
                "median_max_gust_kt": g["max_wind_gust_kt"].median(),
                "most_common_direction_bin": dir_counts.index[0] if not dir_counts.empty else "missing",
                "median_duration_to_surge_peak_hr": g["duration_to_surge_peak_hr"].median(),
                "median_surge_peak_water_level_ft": g["surge_peak_water_level_ft"].median(),
            }
        )
    return _round_numeric(pd.DataFrame(rows), decimals)


def write_findings_summary(lookup: pd.DataFrame, thresholds: pd.DataFrame, output_path: str | Path) -> None:
    lines = ["# NWCL1 Wind/Rise Findings", ""]
    lines.append("## Rise thresholds")
    lines.append("")
    if thresholds.empty:
        lines.append("No threshold findings available.")
    else:
        for _, row in thresholds.iterrows():
            lines.append(
                f"- Surge rise >= {row['rise_threshold_ft']:.2f} ft: median mean wind {row['median_mean_wind_speed_kt']:.2f} kt, "
                f"median max wind {row['median_max_wind_speed_kt']:.2f} kt, common direction {row['most_common_direction_bin']}, "
                f"median time to surge peak {row['median_duration_to_surge_peak_hr']:.2f} hr."
            )
    lines.append("")
    lines.append("## Wind-bin lookup")
    lines.append("")
    if lookup.empty:
        lines.append("No wind-bin lookup available.")
    else:
        for _, row in lookup.iterrows():
            if row["event_count"] < 2:
                continue
            lines.append(
                f"- {row['wind_direction_bin']} wind, {row['mean_wind_speed_bin_kt']} kt mean: "
                f"median surge rise {row['median_rise_ft']:.2f} ft from {int(row['event_count'])} events."
            )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
