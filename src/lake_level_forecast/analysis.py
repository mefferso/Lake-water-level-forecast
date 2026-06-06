from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lake_level_forecast.features import feature_columns


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _round_numeric(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    return out


def load_model_payloads(model_dir: str | Path) -> dict[int, dict]:
    payloads = {}
    for path in Path(model_dir).glob("ridge_delta_*h.joblib"):
        payload = joblib.load(path)
        payloads[int(payload["horizon_hr"])] = payload
    return dict(sorted(payloads.items()))


def make_model_coefficients(model_dir: str | Path, decimals: int = 2) -> pd.DataFrame:
    rows = []
    for hr, payload in load_model_payloads(model_dir).items():
        model = payload["model"]
        features = payload["features"]
        ridge = model.named_steps["ridge"]
        for feature, coef in zip(features, ridge.coef_):
            rows.append(
                {
                    "horizon_hr": hr,
                    "feature": feature,
                    "coefficient": coef,
                    "abs_coefficient": abs(coef),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["horizon_hr", "abs_coefficient"], ascending=[True, False])
    return _round_numeric(df, decimals)


def make_prediction_table(table: pd.DataFrame, model_dir: str | Path, decimals: int = 2) -> pd.DataFrame:
    rows = []
    table = table.copy()
    table["datetime_utc"] = pd.to_datetime(table["datetime_utc"], utc=True)

    for hr, payload in load_model_payloads(model_dir).items():
        target = f"delta_{hr}h_ft"
        if target not in table.columns:
            continue
        model = payload["model"]
        features = payload["features"]
        data = table.dropna(subset=[target]).copy()
        if data.empty:
            continue
        pred_delta = model.predict(data[features])
        rows.append(
            pd.DataFrame(
                {
                    "event_id": data["event_id"].to_numpy(),
                    "datetime_utc": data["datetime_utc"].to_numpy(),
                    "horizon_hr": hr,
                    "current_water_level_ft": data["water_level_ft"].to_numpy(),
                    "observed_delta_ft": data[target].to_numpy(),
                    "predicted_delta_ft": pred_delta,
                    "observed_future_water_level_ft": data["water_level_ft"].to_numpy() + data[target].to_numpy(),
                    "predicted_future_water_level_ft": data["water_level_ft"].to_numpy() + pred_delta,
                    "error_ft": pred_delta - data[target].to_numpy(),
                    "abs_error_ft": np.abs(pred_delta - data[target].to_numpy()),
                }
            )
        )

    if not rows:
        return pd.DataFrame()
    return _round_numeric(pd.concat(rows, ignore_index=True), decimals)


def make_event_verification(predictions: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    rows = []
    for (event_id, hr), group in predictions.groupby(["event_id", "horizon_hr"]):
        obs_peak = group["observed_future_water_level_ft"].max()
        pred_peak = group["predicted_future_water_level_ft"].max()
        rows.append(
            {
                "event_id": event_id,
                "horizon_hr": hr,
                "n_rows": len(group),
                "mean_error_ft": group["error_ft"].mean(),
                "mae_ft": group["abs_error_ft"].mean(),
                "max_abs_error_ft": group["abs_error_ft"].max(),
                "observed_peak_ft": obs_peak,
                "predicted_peak_ft": pred_peak,
                "peak_error_ft": pred_peak - obs_peak,
            }
        )
    df = pd.DataFrame(rows).sort_values(["horizon_hr", "mae_ft"], ascending=[True, False])
    return _round_numeric(df, decimals)


def make_metrics_from_predictions(predictions: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    rows = []
    for hr, group in predictions.groupby("horizon_hr"):
        y = group["observed_delta_ft"]
        pred = group["predicted_delta_ft"]
        baseline = np.zeros_like(y, dtype=float)
        mae = mean_absolute_error(y, pred)
        baseline_mae = mean_absolute_error(y, baseline)
        rows.append(
            {
                "horizon_hr": hr,
                "n_rows": len(group),
                "mae_ft": mae,
                "rmse_ft": _rmse(y, pred),
                "r2": r2_score(y, pred) if len(group) > 1 else np.nan,
                "baseline_mae_ft": baseline_mae,
                "baseline_rmse_ft": _rmse(y, baseline),
                "skill_vs_baseline_mae_pct": 100 * (1 - mae / baseline_mae) if baseline_mae else np.nan,
            }
        )
    return _round_numeric(pd.DataFrame(rows), decimals)
