from __future__ import annotations

import argparse

from lake_level_forecast.continuous import scenario_forecast
from lake_level_forecast.settings import REPO_ROOT, ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario forecast for NWCL1 lake rise from constant wind.")
    parser.add_argument("--current-water-level-ft", type=float, required=True)
    parser.add_argument("--wind-speed-kt", type=float, required=True)
    parser.add_argument("--wind-dir-deg", type=float, required=True)
    parser.add_argument("--duration-hours", type=int, default=48)
    args = parser.parse_args()
    ensure_dirs()
    cfg = load_config()
    mc = cfg["model"]
    past_hours = sorted(set(list(mc["wind_lag_hours"])))
    future_hours = sorted(set(list(mc["forecast_hours"]) + [48]))
    out = scenario_forecast(
        current_water_level_ft=args.current_water_level_ft,
        wind_speed_kt=args.wind_speed_kt,
        wind_dir_deg=args.wind_dir_deg,
        duration_hours=args.duration_hours,
        resample_minutes=int(mc["resample_minutes"]),
        past_hours=past_hours,
        future_hours=future_hours,
        model_dir=REPO_ROOT / "models",
        decimals=2,
    )
    out.to_csv(REPO_ROOT / "outputs/tables/scenario_forecast.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
