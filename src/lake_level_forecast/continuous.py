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


def _round_numeric(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    return out


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    direction_rad = np.deg2rad(out["wind_dir_deg"])
    speed = out["wind_speed_kt"]
    out["wind_u_kt"] = -speed * np.sin(direction_rad)
    out["wind_v_kt"] = -speed * np.cos(direction_rad)
    out["wind_u_stress"] = out["wind_u_kt"] * speed
    out["wind_v_stress"] = out["wind_v_kt"] * speed
    return out


def resample_join_water_wind(water: pd.DataFrame, wind: pd.DataFrame, minutes: int = 15) -> pd.DataFrame:
    water = water.copy()
    wind = wind.copy()
    water["datetime_utc"] = pd.to_datetime(water["datetime_utc"], utc=True)
    wind["datetime_utc"] = pd.to_datetime(wind["datetime_utc"], utc=True)
    water = water.drop_duplicates("datetime_utc").set_index("datetime_utc").sort_index()
    wind = wind.drop_duplicates("datetime_utc").set_index("datetime_utc").sort_index()
    freq = f"{minutes}min"
    wlev = water[["water_level_ft"]].resample(freq).mean().interpolate(limit=4)
    wind_cols = ["wind_speed_kt", "wind_dir_deg", "wind_gust_kt", "air_temp_f", "pressure_mb"]
    for col in wind_cols:
        if col not in wind.columns:
            wind[col] = pd.NA
    wnd = wind[wind_cols].resample(freq).mean()
    wnd[["wind_speed_kt", "wind_dir_deg", "wind_gust_kt"]] = wnd[["wind_speed_kt", "wind_dir_deg", "wind_gust_kt"]].interpolate(limit=4)
    out = wlev.join(wnd, how="inner").reset_index()
    return add_wind_components(out)


def _add_rolling_features(df: pd.DataFrame, hours: list[int], minutes: int, direction: str) -> pd.DataFrame:
    out = df.copy()
    periods_per_hour = int(round(60 / minutes))
    cols = ["wind_speed_kt", "wind_gust_kt", "wind_u_kt", "wind_v_kt", "wind_u_stress", "wind_v_stress"]
    for hr in hours:
        window = max(1, hr * periods_per_hour)
        min_periods = max(1, window // 2)
        for col in cols:
            if direction == "past":
                roll = out[col].rolling(window=window, min_periods=min_periods)
            else:
                roll = out[col].shift(-window + 1).rolling(window=window, min_periods=min_periods)
            out[f"{direction}_{hr}h_mean_{col}"] = roll.mean()
            out[f"{direction}_{hr}h_max_{col}"] = roll.max()
    return out


def build_continuous_training_table(water: pd.DataFrame, wind: pd.DataFrame, resample_minutes: int, past_hours: list[int], future_hours: list[int]) -> pd.DataFrame:
    df = resample_join_water_wind(water, wind, minutes=resample_minutes)
    df = _add_rolling_features(df, past_hours, resample_minutes, "past")
    df = _add_rolling_features(df, future_hours, resample_minutes, "future")
    periods_per_hour = int(round(60 / resample_minutes))
    for hr in future_hours:
        shift = hr * periods_per_hour
        df[f"water_level_t_plus_{hr}h_ft"] = df["water_level_ft"].shift(-shift)
        df[f"delta_{hr}h_ft"] = df[f"water_level_t_plus_{hr}h_ft"] - df["water_level_ft"]
    return df


def continuous_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = ["water_level_ft"]
    for col in df.columns:
        if col.startswith("past_") or col.startswith("future_"):
            cols.append(col)
    return cols


def train_continuous_models(table: pd.DataFrame, forecast_hours: list[int], split_date: str, model_dir: str | Path, decimals: int = 2) -> pd.DataFrame:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    table = table.copy()
    table["datetime_utc"] = pd.to_datetime(table["datetime_utc"], utc=True)
    split_ts = pd.Timestamp(split_date, tz="UTC")
    features = continuous_feature_columns(table)
    rows = []
    for hr in forecast_hours:
        target = f"delta_{hr}h_ft"
        data = table.dropna(subset=[target, "water_level_ft"]).copy()
        if data.empty:
            rows.append({"horizon_hr": hr, "error": "no rows"})
            continue
        train = data[data["datetime_utc"] < split_ts]
        test = data[data["datetime_utc"] >= split_ts]
        if len(test) < 100:
            cutoff = int(len(data) * 0.8)
            train = data.iloc[:cutoff]
            test = data.iloc[cutoff:]
        model = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25)))])
        model.fit(train[features], train[target])
        pred = model.predict(test[features])
        y = test[target]
        baseline = np.zeros_like(y, dtype=float)
        mae = mean_absolute_error(y, pred)
        baseline_mae = mean_absolute_error(y, baseline)
        joblib.dump({"model": model, "features": features, "target": target, "horizon_hr": hr}, model_path / f"continuous_delta_{hr}h.joblib")
        rows.append({"horizon_hr": hr, "n_train": len(train), "n_test": len(test), "mae_ft": mae, "rmse_ft": _rmse(y, pred), "r2": r2_score(y, pred) if len(y) > 1 else np.nan, "baseline_mae_ft": baseline_mae, "baseline_rmse_ft": _rmse(y, baseline), "skill_vs_baseline_mae_pct": 100 * (1 - mae / baseline_mae) if baseline_mae else np.nan})
    return _round_numeric(pd.DataFrame(rows), decimals)


def make_scenario_row(current_water_level_ft: float, wind_speed_kt: float, wind_dir_deg: float, duration_hours: int, resample_minutes: int, past_hours: list[int], future_hours: list[int]) -> pd.DataFrame:
    periods = int(max(duration_hours, max(future_hours)) * 60 / resample_minutes) + 1
    times = pd.date_range(pd.Timestamp.now(tz="UTC").floor(f"{resample_minutes}min"), periods=periods, freq=f"{resample_minutes}min")
    df = pd.DataFrame({"datetime_utc": times, "water_level_ft": current_water_level_ft, "wind_speed_kt": wind_speed_kt, "wind_dir_deg": wind_dir_deg, "wind_gust_kt": wind_speed_kt, "air_temp_f": pd.NA, "pressure_mb": pd.NA})
    df = add_wind_components(df)
    df = _add_rolling_features(df, past_hours, resample_minutes, "past")
    df = _add_rolling_features(df, future_hours, resample_minutes, "future")
    return df.head(1).copy()


def scenario_forecast(current_water_level_ft: float, wind_speed_kt: float, wind_dir_deg: float, duration_hours: int, resample_minutes: int, past_hours: list[int], future_hours: list[int], model_dir: str | Path, decimals: int = 2) -> pd.DataFrame:
    row = make_scenario_row(current_water_level_ft, wind_speed_kt, wind_dir_deg, duration_hours, resample_minutes, past_hours, future_hours)
    rows = []
    for path in sorted(Path(model_dir).glob("continuous_delta_*h.joblib")):
        payload = joblib.load(path)
        hr = int(payload["horizon_hr"])
        if hr > duration_hours:
            continue
        features = payload["features"]
        for col in features:
            if col not in row.columns:
                row[col] = pd.NA
        delta = float(payload["model"].predict(row[features])[0])
        rows.append({"horizon_hr": hr, "current_water_level_ft_mllw": current_water_level_ft, "scenario_wind_speed_kt": wind_speed_kt, "scenario_wind_dir_deg": wind_dir_deg, "predicted_rise_ft": delta, "forecast_water_level_ft_mllw": current_water_level_ft + delta})
    return _round_numeric(pd.DataFrame(rows), decimals)
