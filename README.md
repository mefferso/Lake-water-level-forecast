# Lake Water Level Forecast

Experimental Python workflow for relating **NWCL1 / New Canal Station water levels** to **KNEW wind observations**.

The first version does four things:

1. Downloads 5 years of 6-minute NOAA CO-OPS water-level data for New Canal Station.
2. Finds high-water events where NWCL1 reaches or exceeds **2.0 ft MLLW**.
3. Downloads KNEW wind observations from Synoptic for those event windows.
4. Builds lagged wind features and trains simple transparent forecast models for +6, +12, +18, and +24 hours.

## Stations

| Type | Station | ID |
|---|---:|---:|
| Water level | New Canal Station | NOAA CO-OPS `8761927` / `NWCL1` |
| Wind | Lakefront Airport | Synoptic/METAR `KNEW` |

## Basic model idea

The model predicts future water-level **change** instead of raw future water level:

```text
delta_6hr = water_level_t+6hr - water_level_now
forecast_6hr = current_water_level + predicted_delta_6hr
```

That lets the model focus on wind-driven rise/fall instead of trying to rediscover the current lake level every run.

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Then install:

```bash
pip install -r requirements.txt
pip install -e .
```

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Then add your Synoptic token inside `.env`:

```text
SYNOPTIC_TOKEN=your_token_here
```

Never commit `.env`.

## GitHub secret

For GitHub Actions, add a repository secret:

```text
Settings → Secrets and variables → Actions → New repository secret
Name: SYNOPTIC_TOKEN
Value: your Synoptic token
```

## Run locally

From the repo root:

```bash
python scripts/01_fetch_nwcl1_water.py
python scripts/02_find_high_water_events.py
python scripts/03_fetch_knew_wind_for_events.py
python scripts/04_build_training_table.py
python scripts/05_train_models.py
```

## Run in GitHub Actions

Go to:

```text
Actions → Build archive and train models → Run workflow
```

The workflow uploads generated data, models, and metrics as an artifact named:

```text
nwcl1-training-output
```

## Make a forecast from a wind forecast CSV

Create a CSV with at least these columns:

```csv
datetime_utc,wind_speed_kt,wind_dir_deg,wind_gust_kt
2026-06-06 12:00,18,080,24
2026-06-06 13:00,21,090,27
```

Then run:

```bash
python scripts/06_make_forecast.py --wind-csv path/to/forecast_wind.csv
```

The forecast script fetches the latest NWCL1 water level from CO-OPS, builds the same wind features, and writes forecast output to:

```text
outputs/tables/latest_forecast.csv
```

## Important caveats

This is an experimental statistical tool. It is not a replacement for PETSS/ESTOFS/OFS guidance or operational judgment. The first goal is to quantify how much of NWCL1 rise can be explained by KNEW wind speed, direction, and duration.
