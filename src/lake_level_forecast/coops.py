from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

COOPS_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


@dataclass(frozen=True)
class CoopsRequest:
    station: str
    product: str = "water_level"
    datum: str = "MLLW"
    units: str = "english"
    time_zone: str = "gmt"
    application: str = "LakeWaterLevelForecast"


def month_chunks(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    if cursor < start:
        cursor = start
    while cursor <= end:
        if cursor.month == 12:
            next_month = datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            next_month = datetime(cursor.year, cursor.month + 1, 1, tzinfo=timezone.utc)
        chunk_end = min(end, next_month - timedelta(minutes=1))
        yield cursor, chunk_end
        cursor = next_month


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d %H:%M")


def fetch_water_level(start: datetime, end: datetime, req: CoopsRequest) -> pd.DataFrame:
    params = {
        "begin_date": _fmt(start),
        "end_date": _fmt(end),
        "station": req.station,
        "product": req.product,
        "datum": req.datum,
        "units": req.units,
        "time_zone": req.time_zone,
        "format": "json",
        "application": req.application,
    }
    r = requests.get(COOPS_API, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(f"CO-OPS error for {start} to {end}: {payload['error']}")
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame(columns=["datetime_utc", "water_level_ft", "sigma", "flags"])
    df = pd.DataFrame(rows)
    df["datetime_utc"] = pd.to_datetime(df["t"], utc=True)
    df["water_level_ft"] = pd.to_numeric(df["v"], errors="coerce")
    df["sigma"] = pd.to_numeric(df.get("s"), errors="coerce") if "s" in df.columns else pd.NA
    df["flags"] = df["f"].astype(str) if "f" in df.columns else ""
    return df[["datetime_utc", "water_level_ft", "sigma", "flags"]].dropna(subset=["water_level_ft"])


def fetch_water_level_archive(
    station: str,
    start: datetime,
    end: datetime,
    datum: str = "MLLW",
    units: str = "english",
    time_zone: str = "gmt",
    raw_dir: str | Path | None = None,
) -> pd.DataFrame:
    req = CoopsRequest(station=station, datum=datum, units=units, time_zone=time_zone)
    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in month_chunks(start, end):
        print(f"CO-OPS {station} {chunk_start:%Y-%m-%d} to {chunk_end:%Y-%m-%d}")
        df = fetch_water_level(chunk_start, chunk_end, req)
        if raw_dir is not None:
            raw_path = Path(raw_dir)
            raw_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(raw_path / f"coops_{station}_{datum}_{chunk_start:%Y%m}.csv", index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["datetime_utc", "water_level_ft", "sigma", "flags"])
    out_df = pd.concat(frames, ignore_index=True)
    return out_df.drop_duplicates("datetime_utc").sort_values("datetime_utc")


def fetch_latest_water_level(station: str = "8761927", datum: str = "MLLW") -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=6)
    return fetch_water_level(start, end, CoopsRequest(station=station, datum=datum))
