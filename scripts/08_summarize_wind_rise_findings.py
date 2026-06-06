from __future__ import annotations

import argparse

import pandas as pd

from lake_level_forecast.findings import make_event_wind_summary, make_rise_threshold_findings, make_wind_rise_lookup, write_findings_summary
from lake_level_forecast.settings import REPO_ROOT, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize wind conditions associated with NWCL1 lake rises.")
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args()
    ensure_dirs()
    events_path = REPO_ROOT / "data/processed/nwcl1_high_water_events.csv"
    wind_path = REPO_ROOT / "data/processed/knew_wind_event_windows.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing {events_path}. Run script 02 first.")
    if not wind_path.exists():
        raise FileNotFoundError(f"Missing {wind_path}. Run script 03 first.")
    events = pd.read_csv(events_path)
    wind = pd.read_parquet(wind_path)
    summary = make_event_wind_summary(events, wind, decimals=args.decimals)
    lookup = make_wind_rise_lookup(summary, decimals=args.decimals)
    thresholds = make_rise_threshold_findings(summary, decimals=args.decimals)
    out_dir = REPO_ROOT / "outputs/tables"
    summary.to_csv(out_dir / "event_wind_rise_summary.csv", index=False)
    lookup.to_csv(out_dir / "wind_direction_speed_rise_lookup.csv", index=False)
    thresholds.to_csv(out_dir / "rise_threshold_wind_findings.csv", index=False)
    write_findings_summary(lookup, thresholds, REPO_ROOT / "outputs/wind_rise_findings.md")
    print("Saved wind/rise findings")
    print(thresholds)


if __name__ == "__main__":
    main()
