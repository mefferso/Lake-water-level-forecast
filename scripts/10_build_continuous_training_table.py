from __future__ import annotations

import pandas as pd

from lake_level_forecast.continuous import build_continuous_training_table
from lake_level_forecast.settings import REPO_ROOT, ensure_dirs, load_config


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    mc = cfg["model"]
    water = pd.read_parquet(REPO_ROOT / "data/processed/nwcl1_water_level_5yr_mllw.parquet")
    wind = pd.read_parquet(REPO_ROOT / "data/processed/knew_wind_archive_5yr.parquet")
    future_hours = sorted(set(list(mc["forecast_hours"]) + [48]))
    past_hours = sorted(set(list(mc["wind_lag_hours"])))
    table = build_continuous_training_table(water, wind, int(mc["resample_minutes"]), past_hours, future_hours)
    table.to_parquet(REPO_ROOT / "data/processed/continuous_training_table.parquet", index=False)
    table.head(5000).to_csv(REPO_ROOT / "data/processed/continuous_training_table_sample.csv", index=False)
    print(f"Saved continuous table with {len(table)} rows and {len(table.columns)} columns")


if __name__ == "__main__":
    main()
