from __future__ import annotations

import numpy as np
import pandas as pd


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    direction_rad = np.deg2rad(out["wind_dir_deg"])
    speed = out["wind_speed_kt"]
    out["wind_u_kt"] = -speed * np.sin(direction_rad)
    out["wind_v_kt"] = -speed * np.cos(direction_rad)
    out["wind_u_stress"] = out["wind_u_kt"] * speed
    out["wind_v_stress"] = out["wind_v_kt"] * speed
    return out


def resample_event_data(water: pd.DataFrame, wind: pd.DataFrame, minutes: int = 15) -> pd.DataFrame:
    frames = []
    freq = f"{minutes}min"
    event_ids = sorted(set(water["event_id"]).intersection(set(wind["event_id"])))
    for event_id in event_ids:
        wlev = water[water["event_id"] == event_id].copy()
        wnd = wind[wind["event_id"] == event_id].copy()
        wlev["datetime_utc"] = pd.to_datetime(wlev["datetime_utc"], utc=True)
        wnd["datetime_utc"] = pd.to_datetime(wnd["datetime_utc"], utc=True)
        wlev = wlev.set_index("datetime_utc").sort_index()
        wnd = wnd.set_index("datetime_utc").sort_index()
        wlev_rs = wlev[["water_level_ft"]].resample(freq).mean().interpolate(limit=4)
        wind_cols = ["wind_speed_kt", "wind_dir_deg", "wind_gust_kt", "air_temp_f", "pressure_mb"]
        for col in wind_cols:
            if col not in wnd.columns:
                wnd[col] = pd.NA
        wnd_rs = wnd[wind_cols].resample(freq).mean()
        wnd_rs[["wind_speed_kt", "wind_dir_deg", "wind_gust_kt"]] = wnd_rs[["wind_speed_kt", "wind_dir_deg", "wind_gust_kt"]].interpolate(limit=4)
        joined = wlev_rs.join(wnd_rs, how="inner")
        joined["event_id"] = event_id
        frames.append(joined.reset_index())
    if not frames:
        return pd.DataFrame()
    return add_wind_components(pd.concat(frames, ignore_index=True))


def add_lagged_wind_features(df: pd.DataFrame, lag_hours: list[int], minutes: int = 15) -> pd.DataFrame:
    out_frames = []
    periods_per_hour = int(round(60 / minutes))
    base_cols = ["wind_speed_kt", "wind_gust_kt", "wind_u_kt", "wind_v_kt", "wind_u_stress", "wind_v_stress"]
    for event_id, group in df.groupby("event_id"):
        g = group.sort_values("datetime_utc").copy()
        for hr in lag_hours:
            window = max(1, hr * periods_per_hour)
            min_periods = max(1, window // 2)
            for col in base_cols:
                g[f"{col}_mean_{hr}h"] = g[col].rolling(window=window, min_periods=min_periods).mean()
                g[f"{col}_max_{hr}h"] = g[col].rolling(window=window, min_periods=min_periods).max()
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


def add_targets(df: pd.DataFrame, forecast_hours: list[int], minutes: int = 15) -> pd.DataFrame:
    out_frames = []
    periods_per_hour = int(round(60 / minutes))
    for event_id, group in df.groupby("event_id"):
        g = group.sort_values("datetime_utc").copy()
        for hr in forecast_hours:
            shift_periods = hr * periods_per_hour
            g[f"water_level_t_plus_{hr}h_ft"] = g["water_level_ft"].shift(-shift_periods)
            g[f"delta_{hr}h_ft"] = g[f"water_level_t_plus_{hr}h_ft"] - g["water_level_ft"]
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


def build_training_table(water_events: pd.DataFrame, wind_events: pd.DataFrame, resample_minutes: int, lag_hours: list[int], forecast_hours: list[int]) -> pd.DataFrame:
    joined = resample_event_data(water_events, wind_events, resample_minutes)
    if joined.empty:
        return joined
    featured = add_lagged_wind_features(joined, lag_hours=lag_hours, minutes=resample_minutes)
    return add_targets(featured, forecast_hours=forecast_hours, minutes=resample_minutes)


def feature_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("water_level_ft", "wind_speed_kt", "wind_gust_kt", "wind_u_kt", "wind_v_kt", "wind_u_stress", "wind_v_stress", "air_temp_f", "pressure_mb")
    cols = []
    for col in df.columns:
        if col == "datetime_utc" or col.startswith("water_level_t_plus_") or col.startswith("delta_"):
            continue
        if col.startswith(prefixes):
            cols.append(col)
    return cols
