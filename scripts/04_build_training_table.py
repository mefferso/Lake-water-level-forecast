from __future__ import annotations

import pandas as pd

from lake_level_forecast.features import build_training_table
from lake_level_forecast.settings import REPO_ROOT, ensure_dirs, load_config


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    mc = cfg["model"]
    water = pd.read_parquet(REPO_ROOT / "data/processed/nwcl1_water_event_windows.parquet")
    wind = pd.read_parquet(REPO_ROOT / "data/processed/knew_wind_event_windows.parquet")
    table = build_training_table(
        water_events=water,
        wind_events=wind,
        resample_minutes=int(mc["resample_minutes"]),
        lag_hours=list(mc["wind_lag_hours"]),
        forecast_hours=list(mc["forecast_hours"]),
    )
    table.to_parquet(REPO_ROOT / "data/processed/training_table.parquet", index=False)
    table.head(5000).to_csv(REPO_ROOT / "data/processed/training_table_sample.csv", index=False)
    print(f"Saved training table with {len(table)} rows and {len(table.columns)} columns")


if __name__ == "__main__":
    main()
