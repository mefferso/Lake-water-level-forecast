from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from lake_level_forecast.settings import REPO_ROOT, ensure_dirs, load_config
from lake_level_forecast.synoptic import SynopticRequest, fetch_wind_timeseries_chunked


def main() -> None:
    ensure_dirs()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config()
    token = os.getenv("SYNOPTIC_TOKEN")
    if not token:
        raise RuntimeError("SYNOPTIC_TOKEN is not set.")
    wc = cfg["water_level"]
    station = cfg["wind"]["station_id"]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365.25 * int(wc["years_back"]))
    req = SynopticRequest(station=station, token=token)
    wind = fetch_wind_timeseries_chunked(start, end, req, chunk_hours=24)
    wind.to_parquet(REPO_ROOT / "data/processed/knew_wind_archive_5yr.parquet", index=False)
    wind.to_csv(REPO_ROOT / "data/processed/knew_wind_archive_5yr.csv", index=False)
    print(f"Saved {len(wind)} KNEW archive rows")


if __name__ == "__main__":
    main()
