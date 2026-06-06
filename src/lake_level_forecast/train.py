from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lake_level_forecast.features import feature_columns


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_models(table: pd.DataFrame, forecast_hours: list[int], split_date: str, model_dir: str | Path, plot_dir: str | Path | None = None) -> pd.DataFrame:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    table = table.copy()
    table["datetime_utc"] = pd.to_datetime(table["datetime_utc"], utc=True)
    split_ts = pd.Timestamp(split_date, tz="UTC")
    features = feature_columns(table)
    rows = []
    for hr in forecast_hours:
        target = f"delta_{hr}h_ft"
        data = table.dropna(subset=[target]).copy()
        if data.empty:
            rows.append({"horizon_hr": hr, "error": "no training rows"})
            continue
        train = data[data["datetime_utc"] < split_ts]
        test = data[data["datetime_utc"] >= split_ts]
        if len(test) < 20:
            cutoff = int(len(data) * 0.8)
            train = data.iloc[:cutoff]
            test = data.iloc[cutoff:]
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
            ]
        )
        model.fit(train[features], train[target])
        pred = model.predict(test[features])
        y_test = test[target]
        mae = mean_absolute_error(y_test, pred)
        rmse = _rmse(y_test, pred)
        baseline = np.zeros_like(y_test, dtype=float)
        baseline_mae = mean_absolute_error(y_test, baseline)
        baseline_rmse = _rmse(y_test, baseline)
        joblib.dump({"model": model, "features": features, "target": target, "horizon_hr": hr}, model_path / f"ridge_delta_{hr}h.joblib")
        rows.append(
            {
                "horizon_hr": hr,
                "n_train": len(train),
                "n_test": len(test),
                "mae_ft": mae,
                "rmse_ft": rmse,
                "r2": r2_score(y_test, pred) if len(y_test) > 1 else np.nan,
                "baseline_mae_ft": baseline_mae,
                "baseline_rmse_ft": baseline_rmse,
                "skill_vs_baseline_mae_pct": 100 * (1 - mae / baseline_mae) if baseline_mae else np.nan,
            }
        )
    return pd.DataFrame(rows)
