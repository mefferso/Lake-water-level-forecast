from __future__ import annotations

import argparse

import pandas as pd

from lake_level_forecast.coops import fetch_latest_water_level
from lake_level_forecast.forecast import build_forecast_feature_row, predict_water_levels
from lake_level_forecast.settings import REPO_ROOT, ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast NWCL1 water level from a KNEW wind forecast CSV.")
    parser.add_argument("--wind-csv", required=True)
    parser.add_argument("--current-water-level-ft", type=float, default=None)
    args = parser.parse_args()
    ensure_dirs()
    cfg = load_config()
    wc = cfg["water_level"]
    mc = cfg["model"]
    if args.current_water_level_ft is None:
        latest = fetch_latest_water_level(station=wc["station_id"], datum=wc["datum"])
        if latest.empty:
            raise RuntimeError("Could not fetch latest NWCL1 water level.")
        current = float(latest.sort_values("datetime_utc").tail(1)["water_level_ft"].iloc[0])
    else:
        current = args.current_water_level_ft
    wind_fcst = pd.read_csv(args.wind_csv)
    row = build_forecast_feature_row(
        current_water_level_ft=current,
        wind_forecast=wind_fcst,
        resample_minutes=int(mc["resample_minutes"]),
        lag_hours=list(mc["wind_lag_hours"]),
    )
    forecast = predict_water_levels(row, REPO_ROOT / "models")
    forecast.to_csv(REPO_ROOT / "outputs/tables/latest_forecast.csv", index=False)
    print(forecast)


if __name__ == "__main__":
    main()
