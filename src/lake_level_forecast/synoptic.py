from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

SYNOPTIC_TIMESERIES_API = "https://api.synopticdata.com/v2/stations/timeseries"


@dataclass(frozen=True)
class SynopticRequest:
    station: str
    token: str
    units: str = "english"


def _as_utc_timestamp(dt: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _fmt(dt: datetime | pd.Timestamp) -> str:
    return _as_utc_timestamp(dt).strftime("%Y%m%d%H%M")


def _first_existing(obs: dict[str, Any], base: str) -> list[Any] | None:
    for key, val in obs.items():
        if key == base or key.startswith(f"{base}_set_"):
            return val
    return None


def _series_value(values: list[Any] | None, idx: int) -> Any:
    if values is None or idx >= len(values):
        return None
    return values[idx]


def parse_synoptic_timeseries(payload: dict[str, Any], station: str) -> pd.DataFrame:
    stations = payload.get("STATION", [])
    if not stations:
        return pd.DataFrame(columns=["station", "datetime_utc", "wind_speed_kt", "wind_dir_deg", "wind_gust_kt", "air_temp_f", "pressure_mb"])
    st = stations[0]
    obs = st.get("OBSERVATIONS", {})
    times = obs.get("date_time", [])
    speed = _first_existing(obs, "wind_speed")
    direction = _first_existing(obs, "wind_direction")
    gust = _first_existing(obs, "wind_gust")
    temp = _first_existing(obs, "air_temp")
    pressure = _first_existing(obs, "pressure") or _first_existing(obs, "sea_level_pressure") or _first_existing(obs, "altimeter")
    rows = []
    for i, time in enumerate(times):
        rows.append(
            {
                "station": station,
                "datetime_utc": time,
                "wind_speed_kt": _series_value(speed, i),
                "wind_dir_deg": _series_value(direction, i),
                "wind_gust_kt": _series_value(gust, i),
                "air_temp_f": _series_value(temp, i),
                "pressure_mb": _series_value(pressure, i),
            }
        )
    df = pd.DataFrame(rows)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    for col in ["wind_speed_kt", "wind_dir_deg", "wind_gust_kt", "air_temp_f", "pressure_mb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc")


def fetch_wind_timeseries(start: datetime | pd.Timestamp, end: datetime | pd.Timestamp, req: SynopticRequest) -> pd.DataFrame:
    params = {
        "stid": req.station,
        "start": _fmt(start),
        "end": _fmt(end),
        "vars": "wind_speed,wind_direction,wind_gust,air_temp,pressure,sea_level_pressure,altimeter",
        "units": req.units,
        "token": req.token,
        "hfmetars": 1,
        "obtimezone": "UTC",
    }
    r = requests.get(SYNOPTIC_TIMESERIES_API, params=params, timeout=90)
    r.raise_for_status()
    payload = r.json()
    status = payload.get("SUMMARY", {}).get("RESPONSE_CODE")
    if status not in (1, None):
        raise RuntimeError(f"Synoptic API error: {payload.get('SUMMARY')}")
    return parse_synoptic_timeseries(payload, req.station)


def fetch_wind_timeseries_chunked(
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
    req: SynopticRequest,
    chunk_hours: int = 24,
) -> pd.DataFrame:
    start_ts = _as_utc_timestamp(start)
    end_ts = _as_utc_timestamp(end)
    cursor = start_ts
    frames = []
    while cursor < end_ts:
        chunk_end = min(cursor + pd.Timedelta(hours=chunk_hours), end_ts)
        print(f"  chunk {cursor} to {chunk_end}")
        frames.append(fetch_wind_timeseries(cursor, chunk_end, req))
        cursor = chunk_end
    if not frames:
        return pd.DataFrame(columns=["station", "datetime_utc", "wind_speed_kt", "wind_dir_deg", "wind_gust_kt", "air_temp_f", "pressure_mb"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("datetime_utc").sort_values("datetime_utc")


def fetch_wind_for_events(events: pd.DataFrame, station: str, token: str, raw_dir: str | Path | None = None) -> pd.DataFrame:
    frames = []
    req = SynopticRequest(station=station, token=token)
    for _, ev in events.iterrows():
        event_id = ev["event_id"]
        start = pd.Timestamp(ev["window_start_utc"])
        end = pd.Timestamp(ev["window_end_utc"])
        print(f"Synoptic {station} {event_id} {start} to {end}")
        df = fetch_wind_timeseries_chunked(start, end, req)
        df["event_id"] = event_id
        if raw_dir is not None:
            raw_path = Path(raw_dir)
            raw_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(raw_path / f"synoptic_{station}_{event_id}.csv", index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(["event_id", "datetime_utc"]).sort_values(["event_id", "datetime_utc"])
