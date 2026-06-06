from __future__ import annotations

import pandas as pd

from lake_level_forecast.events import EventSettings, find_exceedance_events, subset_water_to_event_windows
from lake_level_forecast.settings import REPO_ROOT, ensure_dirs, load_config


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    wc = cfg["water_level"]
    water_path = REPO_ROOT / "data/processed/nwcl1_water_level_5yr_mllw.parquet"
    if not water_path.exists():
        raise FileNotFoundError(f"Missing {water_path}. Run script 01 first.")
    water = pd.read_parquet(water_path)
    settings = EventSettings(
        threshold_ft=float(wc["threshold_ft"]),
        merge_gap_hours=float(wc["event_merge_gap_hours"]),
        padding_hours=float(wc["event_padding_hours"]),
    )
    events = find_exceedance_events(water, settings)
    windows = subset_water_to_event_windows(water, events)
    events.to_csv(REPO_ROOT / "data/processed/nwcl1_high_water_events.csv", index=False)
    windows.to_parquet(REPO_ROOT / "data/processed/nwcl1_water_event_windows.parquet", index=False)
    print(f"Found {len(events)} events")
    print(events.tail(20))


if __name__ == "__main__":
    main()
