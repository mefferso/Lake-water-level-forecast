from __future__ import annotations

import pandas as pd

from lake_level_forecast.continuous import train_continuous_models
from lake_level_forecast.settings import REPO_ROOT, ensure_dirs, load_config


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    mc = cfg["model"]
    table = pd.read_parquet(REPO_ROOT / "data/processed/continuous_training_table.parquet")
    hours = sorted(set(list(mc["forecast_hours"]) + [48]))
    metrics = train_continuous_models(table, hours, str(mc["train_test_split_date"]), REPO_ROOT / "models", decimals=2)
    metrics.to_csv(REPO_ROOT / "outputs/tables/continuous_model_metrics.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
