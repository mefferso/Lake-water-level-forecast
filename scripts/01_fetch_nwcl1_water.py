from __future__ import annotations

from datetime import datetime, timedelta, timezone

from lake_level_forecast.coops import fetch_water_level_archive
from lake_level_forecast.settings import REPO_ROOT, ensure_dirs, load_config


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    wc = cfg["water_level"]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365.25 * int(wc["years_back"]))
    df = fetch_water_level_archive(
        station=wc["station_id"],
        start=start,
        end=end,
        datum=wc["datum"],
        units=wc["units"],
        time_zone=wc["time_zone"],
        raw_dir=REPO_ROOT / "data/raw/coops",
    )
    df.to_parquet(REPO_ROOT / "data/processed/nwcl1_water_level_5yr_mllw.parquet", index=False)
    df.to_csv(REPO_ROOT / "data/processed/nwcl1_water_level_5yr_mllw.csv", index=False)
    print(f"Saved {len(df)} rows")


if __name__ == "__main__":
    main()
