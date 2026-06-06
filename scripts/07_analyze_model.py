from __future__ import annotations

import argparse

import pandas as pd

from lake_level_forecast.analysis import make_event_verification, make_metrics_from_predictions, make_model_coefficients, make_prediction_table
from lake_level_forecast.settings import REPO_ROOT, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create rounded model analysis tables.")
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args()
    ensure_dirs()
    table_path = REPO_ROOT / "data/processed/training_table.parquet"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing {table_path}. Run script 04 first.")
    table = pd.read_parquet(table_path)
    coeffs = make_model_coefficients(REPO_ROOT / "models", decimals=args.decimals)
    predictions = make_prediction_table(table, REPO_ROOT / "models", decimals=args.decimals)
    events = make_event_verification(predictions, decimals=args.decimals)
    metrics = make_metrics_from_predictions(predictions, decimals=args.decimals)
    out_dir = REPO_ROOT / "outputs/tables"
    coeffs.to_csv(out_dir / "model_coefficients.csv", index=False)
    predictions.to_csv(out_dir / "predictions_all_rows.csv", index=False)
    events.to_csv(out_dir / "event_verification.csv", index=False)
    metrics.to_csv(out_dir / "model_metrics_rounded.csv", index=False)
    print("Saved rounded analysis tables")
    print(metrics)


if __name__ == "__main__":
    main()
