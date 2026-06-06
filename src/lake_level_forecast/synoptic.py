from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd
import requests

SYNOPTIC_TIMESERIES_API = "https://api.synopticdata.com/v2/stations/timeseries"
WIND_COLUMNS = [
    "station",
    "datetime_utc",
    "wind_speed_kt",
    "wind_dir_deg",
    "wind_gust_kt",
    "air_temp_f",
    "pressure_mb",
]


@dataclass(frozen=True)
class SynopticRequest:
    station: str
    token: str
    units: str = "english"


def _empty_wind_df() -> pd.DataFrame:
    return pd.DataFrame(columns=WIND_COLUMNS)


def _as_utc_timestamp(dt: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _now_utc_safe() -> pd.Timestamp:
    # Small buffer prevents API errors when GitHub runner clock and API clock differ by a minute or two.
    return pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=10)


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
        return _empty_wind_df()
    st = stations[0]
    obs = st.get("OBSERVATIONS", {})
    times = obs.get("date_time", [])
    if not times:
        return _empty_wind_df()

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


def _get_json_with_retries(params: dict[str, Any], attempts: int = 4) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(SYNOPTIC_TIMESERIES_API, params=params, timeout=90)
            if r.status_code in {429, 500, 502, 503, 504} and attempt < attempts:
                wait = 2 * attempt
                print(f"  warning: Synoptic HTTP {r.status_code}; retrying in {wait}s")
                sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < attempts:
                wait = 2 * attempt
                print(f"  warning: Synoptic request failed ({exc}); retrying in {wait}s")
                sleep(wait)
                continue
            raise RuntimeError(f"Synoptic request failed after {attempts} attempts: {last_error}") from exc
    raise RuntimeError(f"Synoptic request failed: {last_error}")


def fetch_wind_timeseries(start: datetime | pd.Timestamp, end: datetime | pd.Timestamp, req: SynopticRequest) -> pd.DataFrame:
    start_ts = _as_utc_timestamp(start)
    end_ts = _as_utc_timestamp(end)
    safe_now = _now_utc_safe()

    if start_ts >= safe_now:
        print(f"  warning: requested Synoptic start {start_ts} is in the future; skipping chunk")
        return _empty_wind_df()
    if end_ts > safe_now:
        print(f"  warning: clamping Synoptic end from {end_ts} to {safe_now}")
        end_ts = safe_now
    if end_ts <= start_ts:
        return _empty_wind_df()

    params = {
        "stid": req.station,
        "start": _fmt(start_ts),
        "end": _fmt(end_ts),
        "vars": "wind_speed,wind_direction,wind_gust,air_temp,pressure,sea_level_pressure,altimeter",
        "units": req.units,
        "token": req.token,
        "hfmetars": 1,
        "obtimezone": "UTC",
    }
    payload = _get_json_with_retries(params)
    summary = payload.get("SUMMARY", {})
    status = summary.get("RESPONSE_CODE")
    message = str(summary.get("RESPONSE_MESSAGE", ""))

    # Synoptic returns RESPONSE_CODE 2 for no data/no station access in a specific window.
    # For archive building, empty chunks should not kill the whole run.
    if status == 2 or "No stations found" in message:
        print(f"  warning: no Synoptic data for {req.station} {_fmt(start_ts)}-{_fmt(end_ts)}; skipping chunk")
        return _empty_wind_df()

    # If a padded event runs beyond the current time, Synoptic may return this instead of data.
    # Treat it as empty after the safe-now clamp above.
    if status == -1 and "START cannot be in the future" in message:
        print(f"  warning: Synoptic rejected future request for {req.station}; skipping chunk")
        return _empty_wind_df()

    if status not in (1, None):
        raise RuntimeError(f"Synoptic API error: {summary}")
    return parse_synoptic_timeseries(payload, req.station)


def fetch_wind_timeseries_chunked(
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
    req: SynopticRequest,
    chunk_hours: int = 24,
) -> pd.DataFrame:
    start_ts = _as_utc_timestamp(start)
    end_ts = min(_as_utc_timestamp(end), _now_utc_safe())

    if start_ts >= end_ts:
        print(f"  warning: event window {start_ts} to {end_ts} has no past time to request; skipping")
        return _empty_wind_df()

    cursor = start_ts
    frames = []
    while cursor < end_ts:
        chunk_end = min(cursor + pd.Timedelta(hours=chunk_hours), end_ts)
        print(f"  chunk {cursor} to {chunk_end}")
        frames.append(fetch_wind_timeseries(cursor, chunk_end, req))
        cursor = chunk_end

    if not frames:
        return _empty_wind_df()
    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        return _empty_wind_df()
    return out.drop_duplicates("datetime_utc").sort_values("datetime_utc")


def fetch_wind_for_events(events: pd.DataFrame, station: str, token: str, raw_dir: str | Path | None = None) -> pd.DataFrame:
    frames = []
    req = SynopticRequest(station=station, token=token)
    skipped_events = 0

    for _, ev in events.iterrows():
        event_id = ev["event_id"]
        start = pd.Timestamp(ev["window_start_utc"])
        end = pd.Timestamp(ev["window_end_utc"])
        print(f"Synoptic {station} {event_id} {start} to {end}")
        df = fetch_wind_timeseries_chunked(start, end, req)
        if df.empty:
            skipped_events += 1
            print(f"  warning: no usable wind rows for {event_id}; event will be skipped downstream")
            continue
        df["event_id"] = event_id
        if raw_dir is not None:
            raw_path = Path(raw_dir)
            raw_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(raw_path / f"synoptic_{station}_{event_id}.csv", index=False)
        frames.append(df)

    if not frames:
        print(f"No Synoptic wind data found for any event. Skipped events: {skipped_events}")
        return pd.DataFrame(columns=WIND_COLUMNS + ["event_id"])
    print(f"Synoptic wind fetch complete. Events with no wind data skipped: {skipped_events}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["event_id", "datetime_utc"]).sort_values(["event_id", "datetime_utc"])
