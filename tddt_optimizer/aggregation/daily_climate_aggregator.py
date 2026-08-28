from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def saturation_vapor_pressure_kpa(temp_c: pd.Series | float) -> pd.Series | float:
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def aggregate_5m_to_growth_daily(df5: pd.DataFrame, season_start_date: str | None = None, output_csv: str | Path | None = None) -> pd.DataFrame:
    df = df5.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    start = pd.to_datetime(season_start_date).date() if season_start_date else min(df["date"])
    g = df.groupby("date", sort=True)
    daily = pd.DataFrame({
        "date": [d for d, _ in g],
        "mint": g["outdoor_temp_c"].min().values,
        "maxt": g["outdoor_temp_c"].max().values,
        "wind": g["outdoor_wind_m_s"].mean().values,
        "rain": g["outdoor_rain_mm_day"].mean().values,
        "okta": g["outdoor_cloud_okta"].mean().round().clip(0, 8).values,
        "rad": (g["outdoor_solar_w_m2"].mean().values * 86.4),
    })
    daily["date"] = pd.to_datetime(daily["date"])
    daily["fattening_day"] = (daily["date"].dt.date - start).apply(lambda x: x.days + 1)
    daily = daily[daily["fattening_day"] >= 1].copy()
    mean_temp = (daily["mint"] + daily["maxt"]) / 2.0
    daily["vpr"] = saturation_vapor_pressure_kpa(mean_temp) * 0.65
    daily["aha"] = 1.30
    daily["doy"] = daily["date"].dt.dayofyear.astype(int)
    daily["yr"] = daily["date"].dt.year.astype(int)
    daily["is_observed"] = 1
    daily["source_day"] = daily["fattening_day"]
    daily["season_start_date"] = str(start)
    cols = ["fattening_day", "yr", "doy", "rad", "mint", "maxt", "vpr", "wind", "rain", "aha", "okta", "is_observed", "source_day", "season_start_date"]
    daily = daily[cols].reset_index(drop=True)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(output_csv, index=False)
    return daily
