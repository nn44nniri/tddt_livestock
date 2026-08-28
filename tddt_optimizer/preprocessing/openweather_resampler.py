from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import sys

NUMERIC_INTERPOLATE = ["temp", "humidity", "wind_speed", "clouds_all", "rain_1h", "rain_3h", "snow_1h", "snow_3h", "dew_point", "pressure"]


def _parse_dt_iso(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.replace(" UTC", "", regex=False)
    return pd.to_datetime(raw, utc=True, errors="coerce").dt.tz_convert(None)


def load_openweather_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "dt_iso" not in df.columns:
        raise ValueError("Input dataset must contain dt_iso column.")
    df["timestamp"] = _parse_dt_iso(df["dt_iso"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    return df


def resample_to_5min(path: str | Path, output_csv: str | Path | None = None) -> pd.DataFrame:
    """Expand the hourly OpenWeather file to 5-minute cadence.

    This implementation interpolates each numeric hour-to-hour segment into 12
    five-minute samples and preserves all available source rows. It is faster
    and more deterministic than reindexing a 10+ year DatetimeIndex on small
    edge devices.
    """
    hourly = load_openweather_csv(path)
    n = len(hourly)
    if n < 2:
        raise ValueError("At least two hourly rows are required for 5-minute interpolation.")
    steps = 12
    # 12 five-minute samples per source hour-to-next-hour segment, plus final source row.
    seg_count = n - 1
    frac = np.tile(np.arange(steps, dtype=float) / steps, seg_count)
    base_idx = np.repeat(np.arange(seg_count), steps)
    timestamps = hourly["timestamp"].iloc[base_idx].to_numpy() + pd.to_timedelta(np.tile(np.arange(steps) * 5, seg_count), unit="min")
    out = pd.DataFrame({"timestamp": timestamps})
    for col in NUMERIC_INTERPOLATE:
        if col in hourly.columns:
            vals = pd.to_numeric(hourly[col], errors="coerce").ffill().bfill().to_numpy(dtype=float)
            out[col] = vals[base_idx] + (vals[base_idx + 1] - vals[base_idx]) * frac
    for col in ["city_name", "weather_main", "weather_description", "weather_icon"]:
        if col in hourly.columns:
            vals = hourly[col].ffill().bfill().to_numpy()
            out[col] = vals[base_idx]
    # Append exact final row so the available dataset end is represented.
    tail = {"timestamp": hourly["timestamp"].iloc[-1]}
    for col in NUMERIC_INTERPOLATE:
        if col in hourly.columns:
            tail[col] = pd.to_numeric(hourly[col], errors="coerce").ffill().bfill().iloc[-1]
    for col in ["city_name", "weather_main", "weather_description", "weather_icon"]:
        if col in hourly.columns:
            tail[col] = hourly[col].ffill().bfill().iloc[-1]
    out = pd.concat([out, pd.DataFrame([tail])], ignore_index=True)
    out["outdoor_temp_c"] = out.get("temp", pd.Series(0, index=out.index)).astype(float)
    out["outdoor_rh_pct"] = out.get("humidity", pd.Series(60, index=out.index)).astype(float).clip(0, 100)
    out["outdoor_wind_m_s"] = out.get("wind_speed", pd.Series(1.0, index=out.index)).astype(float).clip(lower=0)
    out["outdoor_solar_w_m2"] = _solar_proxy(pd.DatetimeIndex(out["timestamp"]), out.get("clouds_all", pd.Series(50, index=out.index)))
    out["outdoor_cloud_okta"] = (out.get("clouds_all", pd.Series(100, index=out.index)).astype(float).clip(0, 100) / 12.5).round().clip(0, 8)
    rain = out.get("rain_1h", pd.Series(0, index=out.index)).fillna(0) + out.get("rain_3h", pd.Series(0, index=out.index)).fillna(0) / 3.0
    out["outdoor_rain_mm_day"] = rain * 24.0
    ts = pd.to_datetime(out["timestamp"])
    out["fattening_day"] = ((ts.dt.normalize() - ts.dt.normalize().min()).dt.days + 1).astype(int)
    keep = ["timestamp", "outdoor_temp_c", "outdoor_rh_pct", "outdoor_wind_m_s", "outdoor_solar_w_m2", "outdoor_cloud_okta", "outdoor_rain_mm_day", "fattening_day"]
    out = out[[c for c in keep if c in out.columns]]
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_size = 100_000
        for start in range(0, len(out), chunk_size):
            chunk = out.iloc[start:start + chunk_size]
            chunk.to_csv(output_path, index=False, mode="w" if start == 0 else "a", header=(start == 0), float_format="%.4f")
            print(f"wrote 5m rows {min(start + chunk_size, len(out))}/{len(out)}", file=sys.stderr, flush=True)
    return out

def _solar_proxy(index: pd.DatetimeIndex, clouds_pct: pd.Series) -> pd.Series:
    hour = index.hour + index.minute / 60.0
    daylight = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
    cloud_factor = 1.0 - pd.Series(clouds_pct).astype(float).fillna(100).clip(0, 100).to_numpy() / 120.0
    return pd.Series(800.0 * daylight * cloud_factor, index=index).clip(lower=0).reset_index(drop=True)
