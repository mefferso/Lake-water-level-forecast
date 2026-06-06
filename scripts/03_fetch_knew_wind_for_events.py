from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv

from lake_level_forecast.settings import REPO_ROOT, ensure_dirs, load_config
from lake_level_forecast.synoptic import fetch_wind_for_events


def main() -> None:
    ensure_dirs()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config()
    token = os.getenv("SYNOPTIC_TOKEN")
    if token is None or token == "":
        raise RuntimeError("SYNOPTIC_TOKEN is not set.")
    events_path = REPO_ROOT / "data/processed/nwcl1_high_water_events.csv"
    events = pd.read_csv(events_path, parse_dates=["event_start_utc", "event_end_utc", "window_start_utc", "window_end_utc", "peak_time_utc"])
    wind = fetch_wind_for_events(events=events, station=cfg["wind"]["station_id"], token=token, raw_dir=REPO_ROOT / "data/raw/synoptic")
    wind.to_parquet(REPO_ROOT / "data/processed/knew_wind_event_windows.parquet", index=False)
    wind.to_csv(REPO_ROOT / "data/processed/knew_wind_event_windows.csv", index=False)
    print(f"Saved {len(wind)} wind rows")


if __name__ == "__main__":
    main()
