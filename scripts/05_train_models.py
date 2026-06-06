from __future__ import annotations

import pandas as pd

from lake_level_forecast.settings import REPO_ROOT, ensure_dirs, load_config
from lake_level_forecast.train import train_models


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    mc = cfg["model"]
    table = pd.read_parquet(REPO_ROOT / "data/processed/training_table.parquet")
    metrics = train_models(
        table=table,
        forecast_hours=list(mc["forecast_hours"]),
        split_date=str(mc["train_test_split_date"]),
        model_dir=REPO_ROOT / "models",
        plot_dir=REPO_ROOT / "outputs/plots",
    )
    metrics.to_csv(REPO_ROOT / "outputs/tables/model_metrics.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
