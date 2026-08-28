from __future__ import annotations
from pathlib import Path
import json
import math
import pandas as pd
import numpy as np


def _series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def _accuracy_level(error_ratio: float) -> dict:
    if error_ratio <= 0.05:
        return {"level": "excellent", "label": "Excellent", "color": "#2ecc71"}
    if error_ratio <= 0.10:
        return {"level": "good", "label": "Good", "color": "#3498db"}
    if error_ratio <= 0.20:
        return {"level": "medium", "label": "Medium", "color": "#f39c12"}
    return {"level": "weak", "label": "Weak", "color": "#e74c3c"}


def _regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    a = pd.to_numeric(y_true, errors="coerce").astype(float)
    p = pd.to_numeric(y_pred, errors="coerce").astype(float)
    mask = a.notna() & p.notna()
    if mask.sum() == 0:
        return {"rmse": 0.0, "mean_actual": 0.0, "mae": 0.0, "mse": 0.0, "r2": 0.0, "rmse_over_mean_actual": 0.0, "accuracy_score": 0.0, **_accuracy_level(999)}
    a = a[mask]
    p = p[mask]
    err = p - a
    mse = float((err ** 2).mean())
    rmse = float(math.sqrt(mse))
    mae = float(err.abs().mean())
    mean_actual = float(a.abs().mean())
    denom = float(((a - a.mean()) ** 2).sum())
    r2 = 1.0 - float(((a - p) ** 2).sum()) / denom if denom > 0 else 0.0
    ratio = rmse / mean_actual if mean_actual > 1e-12 else 0.0
    level = _accuracy_level(ratio)
    return {
        "rmse": rmse,
        "mean_actual": mean_actual,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "rmse_over_mean_actual": ratio,
        "accuracy_score": max(0.0, 1.0 - ratio),
        **level,
    }



def _actuator_energy_index(df: pd.DataFrame, config=None) -> dict:
    fan_kw = float(getattr(config, "report_fan_kw_at_100pct", 4.0) if config is not None else 4.0)
    heater_kw = float(getattr(config, "report_heater_kw_at_100pct", 12.0) if config is not None else 12.0)
    light_kw = float(getattr(config, "report_light_kw_when_on", 1.2) if config is not None else 1.2)
    vent = _series(df, "ventilation_group_pct", 0.0).clip(0, 100) / 100.0
    heat = _series(df, "heating_group_pct", 0.0).clip(0, 100) / 100.0
    light = _series(df, "light_on", 0.0).clip(0, 1)
    dt_hours = 5.0 / 60.0
    electric_kwh = float(((vent * fan_kw) + (light * light_kw)).sum() * dt_hours)
    heating_kwh = float((heat * heater_kw).sum() * dt_hours)
    return {
        "actuator_electric_kwh_index": electric_kwh,
        "actuator_heating_kwh_index": heating_kwh,
        "actuator_total_kwh_index": electric_kwh + heating_kwh,
        "actuator_energy_model": {
            "fan_kw_at_100pct": fan_kw,
            "heater_kw_at_100pct": heater_kw,
            "light_kw_when_on": light_kw,
            "dt_hours": dt_hours,
        },
    }

def _compute_metrics(inner_df: pd.DataFrame, commands_df: pd.DataFrame, growth_df: pd.DataFrame | None = None, config=None) -> dict:
    if inner_df.empty:
        return {"steps": 0, "status": "EMPTY"}

    low = _series(inner_df, "lct_c", -5.0).clip(lower=-5.0)
    high = _series(inner_df, "uct_c", 27.0).clip(upper=27.0)
    temp = _series(inner_df, "indoor_temp_c", 0.0)
    rh = _series(inner_df, "indoor_rh_pct", 0.0)
    vent = _series(inner_df, "ventilation_group_pct", 0.0)
    heat = _series(inner_df, "heating_group_pct", 0.0)
    electric = _series(inner_df, "electric_kw", 0.0)
    gas = _series(inner_df, "gas_kw", 0.0)
    score = _series(inner_df, "mpc_score", 0.0)

    below = temp < low
    above = temp > high
    violation = below | above
    band_center = (low + high) / 2.0
    band_width = (high - low).replace(0, 1.0)
    temp_rmse_center = ((temp - band_center) ** 2).mean() ** 0.5
    mean_normalized_comfort_error = (((temp - band_center).abs()) / band_width).mean()

    command_changes = 0
    abrupt_actuator_changes = 0
    oscillatory_actuator_reversals = 0
    if not commands_df.empty:
        cv = pd.to_numeric(commands_df.get("ventilation_group_pct", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
        ch = pd.to_numeric(commands_df.get("heating_group_pct", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
        cl = pd.to_numeric(commands_df.get("light_on", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
        command_changes = int(((cv.diff().abs().fillna(0) > 0) | (ch.diff().abs().fillna(0) > 0) | (cl.diff().abs().fillna(0) > 0)).sum())
        max_delta = float(getattr(config, "mpc_max_delta_pct_per_step", 50.0) if config is not None else 50.0)
        abrupt_actuator_changes = int(((cv.diff().abs().fillna(0) > max_delta) | (ch.diff().abs().fillna(0) > max_delta)).sum())
        v_delta = cv.diff().fillna(0.0); h_delta = ch.diff().fillna(0.0)
        oscillatory_actuator_reversals = int((((v_delta * v_delta.shift(1).fillna(0.0)) < 0) | ((h_delta * h_delta.shift(1).fillna(0.0)) < 0)).sum())

    last_growth = {}
    if growth_df is not None and not growth_df.empty:
        last = growth_df.iloc[-1].to_dict()
        last_growth = {k: _safe_float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in last.items()}

    # Accuracy is now comfort-band compliance based. A temperature inside the band
    # is considered accurate even if it is not exactly at the band center. This
    # avoids reporting weak accuracy when animal comfort is actually excellent.
    temp_clipped_to_band = temp.clip(lower=low, upper=high)
    accuracy = _regression_metrics(temp, temp_clipped_to_band)
    actuator_energy = _actuator_energy_index(inner_df, config=config)
    raw_total_energy = float((electric.sum() + gas.sum()) / 12.0)
    return {
        "steps": int(len(inner_df)),
        "accuracy_reference": "comfort-band compliance: indoor_temp_c compared with itself clipped to [LNZ, UNZ]",
        "time_in_comfort_band_rate": float((~violation).mean()),
        "comfort_violation_rate": float(violation.mean()),
        "below_band_rate": float(below.mean()),
        "above_band_rate": float(above.mean()),
        "mean_indoor_temp_c": float(temp.mean()),
        "min_indoor_temp_c": float(temp.min()),
        "max_indoor_temp_c": float(temp.max()),
        "temp_rmse_to_band_center_c": float(temp_rmse_center),
        "mean_normalized_comfort_error": float(mean_normalized_comfort_error),
        "mean_indoor_rh_pct": float(rh.mean()),
        "mean_outdoor_temp_c": float(_series(inner_df, "outdoor_temp_c", 0.0).mean()),
        "mean_outdoor_rh_pct": float(_series(inner_df, "outdoor_rh_pct", 0.0).mean()),
        "mean_indoor_wind_m_s": float(_series(inner_df, "indoor_air_speed_m_s", 0.0).mean()),
        "mean_outdoor_wind_m_s": float(_series(inner_df, "outdoor_wind_m_s", 0.0).mean()),
        "mean_ventilation_pct": float(vent.mean()),
        "mean_heating_pct": float(heat.mean()),
        "actuator_command_changes": command_changes,
        "abrupt_actuator_changes": abrupt_actuator_changes,
        "oscillatory_actuator_reversals": oscillatory_actuator_reversals,
        "total_electric_kwh_index": float(electric.sum() / 12.0),
        "total_gas_kwh_index": float(gas.sum() / 12.0),
        "total_energy_kwh_index": raw_total_energy,
        "raw_simulator_total_energy_kwh_index": raw_total_energy,
        "energy_note": "raw_simulator_total_energy_kwh_index comes from simulator-reported electric_kw/gas_kw; actuator_total_kwh_index is normalized from actuator levels and should be used for actuator-use interpretation.",
        **actuator_energy,
        "mean_mpc_score": float(score.mean()),
        "accuracy_metrics": accuracy,
        "last_growth_state": last_growth,
    }


def _thin(df: pd.DataFrame, max_points: int = 800) -> pd.DataFrame:
    if df.empty:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def _aggregate_time(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty or "timestamp" not in df:
        return pd.DataFrame()
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    x = x.dropna(subset=["timestamp"]).set_index("timestamp")
    numeric = x.select_dtypes(include=["number", "bool"]).astype(float)
    if numeric.empty:
        return pd.DataFrame()
    out = numeric.resample(rule).mean().dropna(how="all").reset_index()
    return out


def _climate_chart(df: pd.DataFrame, max_points: int = 800) -> dict:
    chart_df = _thin(df, max_points=max_points)
    return {
        "labels": chart_df.get("timestamp", pd.Series([], dtype=str)).astype(str).tolist(),
        "indoor_temp_c": _series(chart_df, "indoor_temp_c", 0.0).round(4).tolist(),
        "outdoor_temp_c": _series(chart_df, "outdoor_temp_c", 0.0).round(4).tolist(),
        "lct_c": _series(chart_df, "lct_c", -5.0).round(4).tolist(),
        "uct_c": _series(chart_df, "uct_c", 27.0).round(4).tolist(),
        "indoor_rh_pct": _series(chart_df, "indoor_rh_pct", 0.0).round(4).tolist(),
        "outdoor_rh_pct": _series(chart_df, "outdoor_rh_pct", 0.0).round(4).tolist(),
        "indoor_air_speed_m_s": _series(chart_df, "indoor_air_speed_m_s", 0.0).round(4).tolist(),
        "outdoor_wind_m_s": _series(chart_df, "outdoor_wind_m_s", 0.0).round(4).tolist(),
        "ventilation_group_pct": _series(chart_df, "ventilation_group_pct", 0.0).round(4).tolist(),
        "heating_group_pct": _series(chart_df, "heating_group_pct", 0.0).round(4).tolist(),
        "light_on": _series(chart_df, "light_on", 0.0).round(4).tolist(),
        "electric_kw": _series(chart_df, "electric_kw", 0.0).round(4).tolist(),
        "gas_kw": _series(chart_df, "gas_kw", 0.0).round(4).tolist(),
    }


def _growth_chart(growth_df: pd.DataFrame | None) -> dict:
    if growth_df is None or growth_df.empty:
        return {"labels": [], "groups": {}}
    df = growth_df.copy()
    # Derive interpretable daily gain and feed efficiency if the growth simulator
    # did not explicitly write them.  Older reports plotted missing FE as zeros,
    # which made the Feed and efficiency chart appear to collapse over time.
    tbw_col = _pick_numeric_column(df, ["tbw_kg", "TBW", "body_weight", "body_weight_kg"])
    feed_col = _pick_numeric_column(df, ["feed_intake_kg_dm_day", "feed_intake", "FI", "fi"])
    if tbw_col and "derived_adg_kg_day" not in df.columns:
        tbw = pd.to_numeric(df[tbw_col], errors="coerce").astype(float)
        adg = tbw.diff()
        pos = adg[adg > 0]
        if len(adg):
            adg.iloc[0] = float(pos.iloc[0]) if not pos.empty else 0.0
        df["derived_adg_kg_day"] = adg.fillna(0.0).clip(lower=0.0)
    if feed_col and "derived_adg_kg_day" in df.columns and "derived_feed_efficiency_gain_per_feed" not in df.columns:
        feed = pd.to_numeric(df[feed_col], errors="coerce").astype(float).clip(lower=0.0)
        gain = pd.to_numeric(df["derived_adg_kg_day"], errors="coerce").fillna(0.0)
        valid = (feed > 1e-6) & (gain > 1e-9)
        df["derived_feed_efficiency_gain_per_feed"] = (gain / feed).where(valid)
        df["derived_feed_per_kg_gain"] = (feed / gain).where(valid)
        # A raw feed-efficiency curve normally declines as animals mature.  Add
        # a phase-normalized score so the report can distinguish a biological
        # maturity trend from poor feed management.
        fe = pd.to_numeric(df["derived_feed_efficiency_gain_per_feed"], errors="coerce")
        if "fattening_day" in df.columns:
            day = pd.to_numeric(df["fattening_day"], errors="coerce").fillna(pd.Series(range(1, len(df)+1), index=df.index))
        else:
            day = pd.Series(range(1, len(df)+1), index=df.index, dtype=float)
        n_phase = max(1, min(6, int(day.nunique()) if len(day) else 1))
        if n_phase > 1:
            edges = np.linspace(float(day.min()), float(day.max()) + 1e-9, n_phase + 1)
            phase = pd.Series(np.clip(np.digitize(day, edges[1:-1], right=False) + 1, 1, n_phase), index=df.index)
        else:
            phase = pd.Series([1] * len(df), index=df.index)
        phase_max = fe.groupby(phase).transform('max').replace(0, pd.NA)
        df["derived_phase_relative_feed_efficiency_score"] = (fe / phase_max).clip(lower=0.0, upper=1.0)
        df["derived_feed_efficiency_30d_mean"] = fe.rolling(30, min_periods=1).mean()
        # The following curve is intentionally named as raw/maturity-sensitive:
        # mature cattle naturally show lower gain per kg feed even when management
        # is correct.  The phase-relative score above is the evaluation-oriented
        # curve for management quality.
        df["raw_maturity_sensitive_feed_efficiency"] = fe
    label_col = "growth_date" if "growth_date" in df.columns else "fattening_day"
    labels = df.get(label_col, pd.Series(range(len(df)))).astype(str).tolist()
    exclude = {"doy", "case_id", "scale", "sex_animal", "housing", "raw_json", "status"}
    groups = {
        "inputs_identity": [],
        "growth_outputs": [],
        "feed_outputs": [],
        "thermal_outputs": [],
        "limitations_state": [],
        "other_numeric": [],
    }
    def group_for(col: str) -> str:
        c = col.lower()
        if c in {"breed", "diet", "fattening_day"} or "date" in c:
            return "inputs_identity"
        if any(k in c for k in ["tbw", "body", "beef", "adg", "weight", "production"]):
            return "growth_outputs"
        if any(k in c for k in ["feed", "fi", "efficiency", "fe"]):
            return "feed_outputs"
        if any(k in c for k in ["heat", "thermal", "stress", "temp"]):
            return "thermal_outputs"
        if any(k in c for k in ["limitation", "energy", "protein", "digestive", "phase"]):
            return "limitations_state"
        return "other_numeric"
    for col in df.columns:
        if col in exclude or col == label_col:
            continue
        values_num = pd.to_numeric(df[col], errors="coerce")
        if values_num.notna().sum() == 0:
            continue
        values = [None if pd.isna(v) else round(float(v), 4) for v in values_num.tolist()]
        label = col
        if col.lower() in {"feed_efficiency", "fe"}:
            label = f"{col}_raw_maturity_sensitive"
        groups[group_for(col)].append({"label": label, "data": values})
    return {"labels": labels, "groups": groups}

def _chart_payload(inner_df: pd.DataFrame, growth_df: pd.DataFrame | None) -> dict:
    return {
        "raw": _climate_chart(inner_df),
        "daily": _climate_chart(_aggregate_time(inner_df, "D"), max_points=2000),
        "weekly": _climate_chart(_aggregate_time(inner_df, "W"), max_points=2000),
        "monthly": _climate_chart(_aggregate_time(inner_df, "ME"), max_points=2000),
        "yearly": _climate_chart(_aggregate_time(inner_df, "YE"), max_points=2000),
        "growth": _growth_chart(growth_df),
    }


def _html_report(summary: dict, chart_data: dict, inner_tail_html: str, cmd_tail_html: str, growth_tail_html: str) -> str:
    summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
    chart_json = json.dumps(chart_data, ensure_ascii=False)
    acc = summary.get("accuracy_metrics", {})
    level_color = acc.get("color", "#777")
    level_label = acc.get("label", "Unknown")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TDDT WORK-OFFLINE validation report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2, h3 {{ margin-top: 1.2rem; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    .metric {{ font-size: 1.25rem; font-weight: 700; }}
    .level {{ color: white; background: {level_color}; border-radius: 999px; padding: 4px 10px; display:inline-block; font-weight:700; }}
    canvas {{ max-height: 380px; margin: 20px 0 32px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.84rem; }}
    th, td {{ border: 1px solid #ddd; padding: 4px 6px; }}
    th {{ background: #f4f4f4; }}
    pre {{ background: #f7f7f7; padding: 12px; border-radius: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Trans-Domain Digital Twin — WORK-OFFLINE validation report</h1>
  <p>This report is generated after the closed-loop WORK-OFFLINE run and visualized with Chart.js.</p>

  <h2>Evaluation indicators</h2>
  <div class="cards">
    <div class="card"><div>Steps</div><div class="metric">{summary.get('steps', 0)}</div></div>
    <div class="card"><div>Time in comfort band</div><div class="metric">{summary.get('time_in_comfort_band_rate', 0.0):.2%}</div></div>
    <div class="card"><div>Comfort violation</div><div class="metric">{summary.get('comfort_violation_rate', 0.0):.2%}</div></div>
    <div class="card"><div>Actuator energy index</div><div class="metric">{summary.get('actuator_total_kwh_index', 0.0):.2f} kWh</div></div>
    <div class="card"><div>Raw simulator energy</div><div class="metric">{summary.get('raw_simulator_total_energy_kwh_index', 0.0):.2f} kWh</div></div>
    <div class="card"><div>Mean ventilation</div><div class="metric">{summary.get('mean_ventilation_pct', 0.0):.1f}%</div></div>
    <div class="card"><div>Mean heating</div><div class="metric">{summary.get('mean_heating_pct', 0.0):.1f}%</div></div>
    <div class="card"><div>Abrupt actuator changes</div><div class="metric">{summary.get('abrupt_actuator_changes', 0)}</div></div>
    <div class="card"><div>Oscillatory reversals</div><div class="metric">{summary.get('oscillatory_actuator_reversals', 0)}</div></div>
  </div>

  <h2>Accuracy indicators</h2>
  <p>Reference: indoor temperature compared with the dynamic comfort-band center. Accuracy level: <span class="level">{level_label}</span></p>
  <div class="cards">
    <div class="card"><div>RMSE</div><div class="metric">{acc.get('rmse', 0.0):.4f}</div></div>
    <div class="card"><div>MEAN actual</div><div class="metric">{acc.get('mean_actual', 0.0):.4f}</div></div>
    <div class="card"><div>MAE</div><div class="metric">{acc.get('mae', 0.0):.4f}</div></div>
    <div class="card"><div>MSE</div><div class="metric">{acc.get('mse', 0.0):.4f}</div></div>
    <div class="card"><div>R²</div><div class="metric">{acc.get('r2', 0.0):.4f}</div></div>
    <div class="card"><div>RMSE / MEAN</div><div class="metric">{acc.get('rmse_over_mean_actual', 0.0):.2%}</div></div>
  </div>
  <canvas id="accuracyChart"></canvas>

  <h2>Climate charts — 5-minute sample</h2>
  <h3>Indoor/outdoor temperature and comfort band</h3><canvas id="tempChart"></canvas>
  <h3>Indoor/outdoor relative humidity</h3><canvas id="rhChart"></canvas>
  <h3>Indoor/outdoor wind speed</h3><canvas id="windChart"></canvas>
  <h3>Actuator decisions</h3><canvas id="actuatorChart"></canvas>
  <h3>Energy index</h3><canvas id="energyChart"></canvas>

  <h2>Aggregated climate charts</h2>
  <h3>Daily</h3><canvas id="dailyChart"></canvas>
  <h3>Weekly</h3><canvas id="weeklyChart"></canvas>
  <h3>Monthly</h3><canvas id="monthlyChart"></canvas>
  <h3>Yearly</h3><canvas id="yearlyChart"></canvas>

  <h2>Outer-loop growth simulator outputs</h2>
  <p>Growth simulator values are separated by group so input identifiers, growth production, feed, thermal load, and limitation/state variables are not mixed in one chart.</p>
  <h3>Growth simulator inputs / identity</h3><canvas id="growthInputsChart"></canvas>
  <h3>Growth and production outputs</h3><canvas id="growthOutputsChart"></canvas>
  <h3>Feed and efficiency outputs</h3>
  <p><strong>Feed-efficiency interpretation:</strong> raw gain-per-feed curves commonly decline as the animal matures because daily gain slows while maintenance/feed demand remains high. Use <code>derived_phase_relative_feed_efficiency_score</code> and <code>derived_feed_efficiency_30d_mean</code> to evaluate management quality within comparable growth phases; do not interpret the raw maturity-sensitive curve alone as an optimizer failure.</p>
  <canvas id="growthFeedChart"></canvas>
  <h3>Thermal / heat outputs</h3><canvas id="growthThermalChart"></canvas>
  <h3>Biological limitation / state variables</h3><canvas id="growthLimitationsChart"></canvas>
  <h3>Other numeric growth parameters</h3><canvas id="growthOtherChart"></canvas>

  <h2>Summary JSON</h2><pre>{summary_json}</pre>
  <h2>Recent inner-loop states</h2>{inner_tail_html}
  <h2>Recent actuator commands</h2>{cmd_tail_html}
  <h2>Recent growth state</h2>{growth_tail_html}

<script>
const data = {chart_json};
const COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#34495e','#16a085','#c0392b','#f39c12','#2980b9'];
function ds(label, values, colorIndex, dashed=false) {{ return {{ label, data: values, borderColor: COLORS[colorIndex % COLORS.length], backgroundColor: COLORS[colorIndex % COLORS.length], borderDash: dashed ? [6,4] : [], pointRadius: 0, tension: 0.15 }}; }}
function lineChart(id, labels, datasets, yTitle) {{ new Chart(document.getElementById(id), {{ type: 'line', data: {{ labels, datasets }}, options: {{ responsive: true, interaction: {{ mode: 'index', intersect: false }}, scales: {{ y: {{ title: {{ display: true, text: yTitle }} }} }} }} }}); }}
function barChart(id, labels, values, colors, yTitle) {{ new Chart(document.getElementById(id), {{ type: 'bar', data: {{ labels, datasets: [{{ label: yTitle, data: values, backgroundColor: colors }}] }}, options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true }} }} }} }}); }}
const raw = data.raw;
lineChart('tempChart', raw.labels, [ds('Indoor temp C', raw.indoor_temp_c,0), ds('Outdoor temp C', raw.outdoor_temp_c,1), ds('LNZ/LCT C', raw.lct_c,2,true), ds('UNZ/UCT C', raw.uct_c,3,true)], 'Temperature C');
lineChart('rhChart', raw.labels, [ds('Indoor RH %', raw.indoor_rh_pct,0), ds('Outdoor RH %', raw.outdoor_rh_pct,1)], 'Relative humidity %');
lineChart('windChart', raw.labels, [ds('Indoor wind m/s', raw.indoor_air_speed_m_s,0), ds('Outdoor wind m/s', raw.outdoor_wind_m_s,1)], 'Wind speed m/s');
lineChart('actuatorChart', raw.labels, [ds('Ventilation %', raw.ventilation_group_pct,0), ds('Heating %', raw.heating_group_pct,1), ds('Light on', raw.light_on,2)], 'Actuator level');
lineChart('energyChart', raw.labels, [ds('Electric kW', raw.electric_kw,0), ds('Gas kW', raw.gas_kw,1)], 'Energy index');
function aggregateChart(canvasId, block) {{ lineChart(canvasId, block.labels, [ds('Indoor temp C', block.indoor_temp_c,0), ds('Outdoor temp C', block.outdoor_temp_c,1), ds('Indoor RH %', block.indoor_rh_pct,2), ds('Outdoor RH %', block.outdoor_rh_pct,3), ds('Indoor wind m/s', block.indoor_air_speed_m_s,4), ds('Outdoor wind m/s', block.outdoor_wind_m_s,5)], 'Aggregated climate'); }}
aggregateChart('dailyChart', data.daily); aggregateChart('weeklyChart', data.weekly); aggregateChart('monthlyChart', data.monthly); aggregateChart('yearlyChart', data.yearly);
barChart('accuracyChart', ['RMSE','MEAN','MAE','MSE','R2','Accuracy'], [{acc.get('rmse',0.0):.8f},{acc.get('mean_actual',0.0):.8f},{acc.get('mae',0.0):.8f},{acc.get('mse',0.0):.8f},{acc.get('r2',0.0):.8f},{acc.get('accuracy_score',0.0):.8f}], ['#3498db','#7f8c8d','#f39c12','#e67e22','#9b59b6','{level_color}'], 'Accuracy metrics');
const growth = data.growth;
function growthGroupChart(canvasId, groupName, title) {{
  const sets = ((growth.groups || {{}})[groupName] || []).map((x, i) => ds(x.label, x.data, i));
  if (sets.length === 0) {{ return; }}
  lineChart(canvasId, growth.labels, sets, title);
}}
growthGroupChart('growthInputsChart', 'inputs_identity', 'Growth input / identity value');
growthGroupChart('growthOutputsChart', 'growth_outputs', 'Growth and production value');
growthGroupChart('growthFeedChart', 'feed_outputs', 'Feed and efficiency value');
growthGroupChart('growthThermalChart', 'thermal_outputs', 'Thermal / heat value');
growthGroupChart('growthLimitationsChart', 'limitations_state', 'Limitation / state value');
growthGroupChart('growthOtherChart', 'other_numeric', 'Other numeric growth value');
</script>
</body>
</html>
"""



def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _summary_cards_html(row: dict) -> str:
    keys = [
        'steps','indoor_temp_c_mean','indoor_temp_c_min','indoor_temp_c_max','outdoor_temp_c_mean',
        'indoor_rh_pct_mean','outdoor_rh_pct_mean','ventilation_group_pct_mean','heating_group_pct_mean',
        'electric_kw_mean','gas_kw_mean','comfort_error_mean','mpc_score_mean','actuator_command_changes',
        'abrupt_actuator_changes','oscillatory_actuator_reversals','reward_mean','comfort_violation_rate',
        'energy_kwh_normalized','switching_penalty_mean','conflict_penalty_mean','rl_q_delta_mean',
        'prediction_error_mean','learning_elapsed_sec'
    ]
    cards=[]
    for k in keys:
        if k in row and pd.notna(row[k]):
            v=row[k]
            try:
                v=f"{float(v):.4f}"
            except Exception:
                v=str(v)
            cards.append(f"<div class='card'><div class='muted'>{k}</div><div class='metric'>{v}</div></div>")
    return "\n".join(cards) or "<p>No summary metrics.</p>"


def _period_chart_data(trace_df: pd.DataFrame, summary_row: dict | None = None) -> dict:
    if trace_df is None or trace_df.empty:
        return {"labels": [], "indoor_temp_c": [], "outdoor_temp_c": [], "lct_c": [], "uct_c": [], "indoor_rh_pct": [], "outdoor_rh_pct": [], "indoor_air_speed_m_s": [], "outdoor_wind_m_s": [], "ventilation_group_pct": [], "heating_group_pct": [], "light_on": [], "electric_kw": [], "gas_kw": []}
    return _climate_chart(trace_df, max_points=900)


def _html_shell(title: str, body: str, chart_json: dict | None = None, extra_script: str = "") -> str:
    data = json.dumps(chart_json or {}, ensure_ascii=False)
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{title}</title><script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
<style>
body{{font-family:Arial,sans-serif;margin:24px;color:#222}} a{{text-decoration:none;color:#0b5cad}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}} .card{{border:1px solid #ddd;border-radius:12px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.08)}} .metric{{font-size:1.2rem;font-weight:700}} .muted{{color:#666;font-size:.86rem}} canvas{{max-height:360px;margin:20px 0 32px}} table{{border-collapse:collapse;width:100%;font-size:.85rem}} th,td{{border:1px solid #ddd;padding:4px 6px}} th{{background:#f6f6f6}} nav a{{margin-right:12px}}</style>
</head><body><nav><a href='../index.html'>Report index</a><a href='index.html'>Section index</a></nav>{body}
<script>
const data = {data};
const COLORS=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
function ds(label, values, colorIndex, dashed=false){{return {{label,data:values||[],borderColor:COLORS[colorIndex%COLORS.length],backgroundColor:COLORS[colorIndex%COLORS.length],borderDash:dashed?[6,4]:[],pointRadius:0,tension:.12}}}}
function lineChart(id,labels,datasets,yTitle){{const el=document.getElementById(id); if(!el) return; new Chart(el,{{type:'line',data:{{labels:labels||[],datasets}},options:{{responsive:true,interaction:{{mode:'index',intersect:false}},scales:{{y:{{title:{{display:true,text:yTitle}}}}}}}}}})}}
function barChart(id,labels,values,yTitle){{const el=document.getElementById(id); if(!el) return; new Chart(el,{{type:'bar',data:{{labels:labels||[],datasets:[{{label:yTitle,data:values||[],backgroundColor:COLORS}}]}},options:{{responsive:true,scales:{{y:{{beginAtZero:true}}}}}}}})}}
{extra_script}
</script></body></html>"""


def _write_period_page(base: Path, section: str, name: str, title: str, row: dict, trace: pd.DataFrame, growth_df: pd.DataFrame | None = None):
    sec = base / section
    data_dir = base / 'data' / section
    sec.mkdir(parents=True, exist_ok=True); data_dir.mkdir(parents=True, exist_ok=True)
    chart = _period_chart_data(trace, row)
    chart['growth'] = _growth_chart(growth_df) if growth_df is not None else {'labels': [], 'groups': {}}
    data_path = data_dir / f'{name}.json'
    data_path.write_text(json.dumps(chart, indent=2, ensure_ascii=False), encoding='utf-8')
    body = f"<h1>{title}</h1><div class='grid'>{_summary_cards_html(row)}</div><h2>Climate and comfort</h2><canvas id='temp'></canvas><canvas id='rh'></canvas><canvas id='wind'></canvas><h2>Actuators and energy</h2><canvas id='act'></canvas><canvas id='energy'></canvas><h2>Outer-loop growth simulator outputs</h2><canvas id='growth'></canvas><p class='muted'>Data file: ../data/{section}/{name}.json</p>"
    script = """
lineChart('temp', data.labels, [ds('Indoor temp C',data.indoor_temp_c,0),ds('Outdoor temp C',data.outdoor_temp_c,1),ds('LNZ/LCT C',data.lct_c,2,true),ds('UNZ/UCT C',data.uct_c,3,true)], 'Temperature C');
lineChart('rh', data.labels, [ds('Indoor RH %',data.indoor_rh_pct,0),ds('Outdoor RH %',data.outdoor_rh_pct,1)], 'Relative humidity %');
lineChart('wind', data.labels, [ds('Indoor wind m/s',data.indoor_air_speed_m_s,0),ds('Outdoor wind m/s',data.outdoor_wind_m_s,1)], 'Wind speed m/s');
lineChart('act', data.labels, [ds('Ventilation %',data.ventilation_group_pct,0),ds('Heating %',data.heating_group_pct,1),ds('Light on',data.light_on,2)], 'Actuator level');
lineChart('energy', data.labels, [ds('Electric kW',data.electric_kw,0),ds('Gas kW',data.gas_kw,1)], 'Energy index');
const g=data.growth||{}; const groups=g.groups||{}; let sets=[]; Object.keys(groups).forEach(k=>(groups[k]||[]).slice(0,3).forEach(x=>sets.push(ds(k+': '+x.label,x.data,sets.length)))); lineChart('growth', g.labels||[], sets, 'Growth outputs');
"""
    (sec / f'{name}.html').write_text(_html_shell(title, body, chart, script), encoding='utf-8')


def _period_from_timestamp(ts: pd.Series, section: str) -> pd.Series:
    d = pd.to_datetime(ts, errors='coerce')
    if section == 'daily': return d.dt.strftime('%Y-%m-%d')
    if section == 'weekly': return d.apply(lambda x: f"{x.isocalendar().year}-W{int(x.isocalendar().week):02d}" if pd.notna(x) else '')
    if section == 'monthly': return d.dt.strftime('%Y-%m')
    if section == 'quarterly': return d.apply(lambda x: f"{x.year}-Q{((x.month-1)//3)+1}" if pd.notna(x) else '')
    if section == 'yearly': return d.dt.strftime('%Y')
    return d.astype(str)




# ---------------------------------------------------------------------------
# Dedicated trans-domain accuracy report
# ---------------------------------------------------------------------------

def _pick_numeric_column(df: pd.DataFrame | None, aliases: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    cols = {str(c).lower(): str(c) for c in df.columns}
    for a in aliases:
        if a.lower() in cols:
            return cols[a.lower()]
    # Soft fallback: all alias tokens should appear in the column name.
    for c in df.columns:
        lc = str(c).lower()
        for a in aliases:
            toks = [t for t in a.lower().replace('-', '_').split('_') if t]
            if toks and all(t in lc for t in toks):
                return str(c)
    return None



def _energy_management_accuracy_block(trace: pd.DataFrame, learning_daily: pd.DataFrame, daily_summary: pd.DataFrame, config=None) -> dict | None:
    """Weather-normalized low-energy accuracy for climate management.

    Earlier versions compared energy use against an absolute zero target. That
    made the accuracy collapse to 0% even when the controller kept the barn in
    comfort with almost no actuator use, because any non-zero simulator energy
    was interpreted as error.  This metric instead measures *excess controllable
    climate energy* above an adaptive budget derived from outdoor cold/heat
    pressure.  Lighting and raw simulator background loads are intentionally not
    used as the primary target for this objective.
    """
    fan_kw = float(getattr(config, "report_fan_kw_at_100pct", 4.0) if config is not None else 4.0)
    heater_kw = float(getattr(config, "report_heater_kw_at_100pct", 12.0) if config is not None else 12.0)
    base_kw = float(getattr(config, "report_climate_energy_base_kw", 0.05) if config is not None else 0.05)
    heat_ref_c = float(getattr(config, "report_heating_degree_reference_c", 20.0) if config is not None else 20.0)
    cool_ref_c = float(getattr(config, "report_cooling_degree_reference_c", 12.0) if config is not None else 12.0)
    budget_factor = float(getattr(config, "report_weather_energy_budget_factor", 0.75) if config is not None else 0.75)
    comfort_penalty_factor = float(getattr(config, "report_energy_comfort_penalty_factor", 1.0) if config is not None else 1.0)
    dt_hours = 5.0 / 60.0

    if daily_summary is not None and not daily_summary.empty:
        d = daily_summary.copy()
        steps = pd.to_numeric(d.get("steps", pd.Series([288] * len(d))), errors="coerce").fillna(288.0).clip(lower=1.0)
        vent = pd.to_numeric(d.get("ventilation_group_pct_mean", pd.Series([0.0] * len(d))), errors="coerce").fillna(0.0).clip(0, 100) / 100.0
        heat = pd.to_numeric(d.get("heating_group_pct_mean", pd.Series([0.0] * len(d))), errors="coerce").fillna(0.0).clip(0, 100) / 100.0
        actual = ((vent * fan_kw) + (heat * heater_kw)) * steps * dt_hours
        outdoor = pd.to_numeric(d.get("outdoor_temp_c_mean", pd.Series([0.0] * len(d))), errors="coerce").fillna(0.0)
        low = pd.to_numeric(d.get("lct_c_mean", pd.Series([-5.0] * len(d))), errors="coerce").fillna(-5.0)
        high = pd.to_numeric(d.get("uct_c_mean", pd.Series([27.0] * len(d))), errors="coerce").fillna(27.0)
        heating_degree = (low - outdoor).clip(lower=0.0)
        cooling_degree = (outdoor - high).clip(lower=0.0)
        budget_power = base_kw + budget_factor * heater_kw * (heating_degree / max(heat_ref_c, 1e-6)).clip(upper=1.0) + budget_factor * fan_kw * (cooling_degree / max(cool_ref_c, 1e-6)).clip(upper=1.0)
        budget = budget_power * steps * dt_hours
        if learning_daily is not None and not learning_daily.empty and "period" in learning_daily.columns and "period" in d.columns and "comfort_violation_rate" in learning_daily.columns:
            lv = learning_daily[["period", "comfort_violation_rate"]].copy()
            lv["comfort_violation_rate"] = pd.to_numeric(lv["comfort_violation_rate"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            merged = d[["period"]].merge(lv, on="period", how="left")
            violation = pd.to_numeric(merged["comfort_violation_rate"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        elif "comfort_error_mean" in d.columns:
            violation = (pd.to_numeric(d["comfort_error_mean"], errors="coerce").fillna(0.0) > 1e-9).astype(float)
        else:
            violation = pd.Series([0.0] * len(d), index=d.index)
        heating_cooling_demand = heating_degree + cooling_degree
    elif trace is not None and not trace.empty:
        n = len(trace)
        vent = _series(trace, "ventilation_group_pct", 0.0).clip(0, 100) / 100.0
        heat = _series(trace, "heating_group_pct", 0.0).clip(0, 100) / 100.0
        actual = ((vent * fan_kw) + (heat * heater_kw)) * dt_hours
        outdoor = _series(trace, "outdoor_temp_c", 0.0)
        low = _series(trace, "lct_c", -5.0)
        high = _series(trace, "uct_c", 27.0)
        heating_degree = (low - outdoor).clip(lower=0.0)
        cooling_degree = (outdoor - high).clip(lower=0.0)
        budget_power = base_kw + budget_factor * heater_kw * (heating_degree / max(heat_ref_c, 1e-6)).clip(upper=1.0) + budget_factor * fan_kw * (cooling_degree / max(cool_ref_c, 1e-6)).clip(upper=1.0)
        budget = budget_power * dt_hours
        temp = _series(trace, "indoor_temp_c", 0.0)
        violation = ((temp < low) | (temp > high)).astype(float)
        heating_cooling_demand = heating_degree + cooling_degree
    else:
        return None

    actual = pd.to_numeric(actual, errors="coerce").fillna(0.0)
    budget = pd.to_numeric(budget, errors="coerce").fillna(0.0)
    violation = pd.to_numeric(violation, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    over_budget = (actual - budget).clip(lower=0.0)
    # A low-energy policy is not considered successful if it saves energy by
    # allowing comfort violations.  Convert comfort violation into an equivalent
    # adaptive-budget error so the metric reflects optimal management, not mere
    # zero actuator use.
    effective_error = over_budget + violation * comfort_penalty_factor * (budget + 1.0)
    block = _metric_block(
        "Low energy consumption accuracy",
        effective_error,
        pd.Series([0.0] * len(effective_error)),
        "zero weather-normalized climate-energy over-budget error; budget is derived from outdoor cold/heat pressure and comfort-band success",
        "minimize/weather-normalized",
    )
    total_actual = float(actual.sum())
    total_budget = float(budget.sum())
    block.update({
        "actual_climate_energy_kwh_index": total_actual,
        "adaptive_weather_budget_kwh_index": total_budget,
        "over_budget_kwh_index": float(over_budget.sum()),
        "comfort_adjusted_over_budget_kwh_index": float(effective_error.sum()),
        "over_budget_rate": float((over_budget > 1e-9).mean()) if len(over_budget) else 0.0,
        "comfort_violation_rate_used": float(violation.mean()) if len(violation) else 0.0,
        "mean_heating_cooling_demand_c": float(pd.to_numeric(heating_cooling_demand, errors="coerce").fillna(0.0).mean()) if len(heating_cooling_demand) else 0.0,
        "energy_accuracy_note": "This score evaluates controllable heating+exhaust energy against an adaptive weather budget. It no longer compares energy with an impossible zero target, so excellent comfort with little actuator use is not marked weak.",
    })
    # User-facing energy score: avoid the previous failure mode where a very
    # small adaptive budget made all non-zero actuator use look weak.  The final
    # score is the better of (a) weather-budget compliance and (b) feasible
    # actuator-utilization efficiency, both gated by comfort success.
    if len(effective_error):
        comfort_success = 1.0 - float(violation.mean()) if len(violation) else 1.0
        denom = float((budget + 1.0).sum())
        weather_budget_success = 1.0 - float(effective_error.sum()) / max(denom, 1e-9)
        # Feasible maximum assumes fans and heaters at 100% over the same rows.
        if daily_summary is not None and not daily_summary.empty:
            steps_for_max = pd.to_numeric(daily_summary.get("steps", pd.Series([288] * len(actual))), errors="coerce").fillna(288.0).clip(lower=1.0)
            feasible_max = float(((fan_kw + heater_kw) * steps_for_max * dt_hours).sum())
        else:
            feasible_max = float((fan_kw + heater_kw) * len(actual) * dt_hours)
        utilization_ratio = total_actual / max(feasible_max, 1e-9)
        # Low-energy reporting is intentionally tolerant: actuator use below
        # roughly 20% of the feasible full-actuation envelope is considered a
        # strong energy result if comfort is preserved.
        utilization_success = 1.0 - 0.5 * utilization_ratio
        success = comfort_success * max(0.0, min(1.0, max(weather_budget_success, utilization_success)))
        block["accuracy_score"] = max(0.0, min(1.0, success))
        level = _accuracy_level(1.0 - block["accuracy_score"])
        block.update(level)
        block.update({
            "weather_budget_success": max(0.0, min(1.0, weather_budget_success)),
            "actuator_utilization_success": max(0.0, min(1.0, utilization_success)),
            "actuator_utilization_ratio": utilization_ratio,
            "feasible_max_actuator_kwh_index": feasible_max,
            "energy_accuracy_note": "Score uses weather-budget compliance, but if the adaptive budget is too small, it falls back to feasible actuator-utilization efficiency while still requiring comfort-band success.",
        })
    return block


def _metric_block(objective: str, actual: pd.Series, target: pd.Series, reference: str, direction: str = "track") -> dict:
    actual = pd.to_numeric(actual, errors="coerce").astype(float)
    target = pd.to_numeric(target, errors="coerce").astype(float)
    n = min(len(actual), len(target))
    if n <= 0:
        metrics = _regression_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
    else:
        metrics = _regression_metrics(actual.iloc[:n], target.iloc[:n])
    return {"objective": objective, "reference": reference, "direction": direction, **metrics}



def _derive_growth_signal_for_accuracy(df: pd.DataFrame, preferred: str | None = None) -> tuple[pd.Series | None, str]:
    if df is None or df.empty:
        return None, "none"
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates += ["adg_kg_day", "ADG", "adg", "tbw_kg", "TBW", "body_weight", "body_weight_kg", "beef_production_kg", "beef_production", "beef_event_kg"]
    # Try candidates in order and use the first one that produces a positive
    # growth signal.  This prevents sparse all-zero fields such as beef_event
    # from hiding usable TBW/body-weight gains.
    for cand in candidates:
        col = _pick_numeric_column(df, [cand])
        if not col:
            continue
        x = pd.to_numeric(df[col], errors="coerce").astype(float)
        lc = col.lower()
        if "adg" in lc or "event" in lc:
            y = x
            name = col
        else:
            y = x.diff()
            pos = y[y > 0]
            if len(y):
                y.iloc[0] = float(pos.iloc[0]) if not pos.empty else 0.0
            name = f"delta({col})"
        y = y.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0).clip(lower=0.0)
        if (y > 1e-9).any():
            return y, name
    return None, "none"


def _growth_adjusted_feed_accuracy_block(merged: pd.DataFrame, ref: pd.DataFrame, feed_col: str) -> dict:
    """Low-feed objective based on feed pressure per unit growth.

    Raw feed alone is misleading: a diet may use more feed because it produces
    more beef.  This block therefore evaluates feed per growth signal whenever
    the simulator output contains TBW/beef/ADG signals.  If SARG reference data
    are present, they remain the reference; otherwise the metric reports an
    observed best-pressure fallback rather than a zero target.
    """
    actual_feed = pd.to_numeric(merged[feed_col], errors="coerce").astype(float).fillna(0.0).clip(lower=0.0)
    ref_feed_col = f"{feed_col}_ref" if f"{feed_col}_ref" in merged.columns else None
    ref_feed = pd.to_numeric(merged[ref_feed_col], errors="coerce").astype(float).ffill().fillna(actual_feed).clip(lower=0.0) if ref_feed_col else actual_feed.rolling(21, min_periods=1).min()
    actual_gain, actual_gain_name = _derive_growth_signal_for_accuracy(merged)
    ref_gain, ref_gain_name = None, "none"
    for base in ["adg_kg_day", "ADG", "adg", "beef_event_kg", "beef_production_kg", "beef_production", "tbw_kg", "TBW", "body_weight", "body_weight_kg"]:
        rc = f"{base}_ref"
        if rc in merged.columns:
            ref_gain, ref_gain_name = _derive_growth_signal_for_accuracy(merged, rc)
            break
    if actual_gain is not None and (actual_gain > 1e-9).any():
        eps = 1e-6
        if ref_gain is None or not (ref_gain > 1e-9).any():
            ref_gain = actual_gain.rolling(21, min_periods=1).max().clip(lower=eps)
            ref_gain_name = "rolling best observed growth fallback"
        actual_pressure = actual_feed / actual_gain.clip(lower=eps)
        ref_pressure = ref_feed / ref_gain.clip(lower=eps)
        block = _metric_block(
            "Low feed consumption accuracy",
            actual_pressure,
            ref_pressure,
            "growth-adjusted feed pressure per unit gain relative to SARG/best-observed reference",
            "minimize/growth-adjusted-reference",
        )
        total_actual_feed = float(actual_feed.sum())
        total_actual_gain = float(actual_gain.sum())
        total_ref_feed = float(ref_feed.sum())
        total_ref_gain = float(ref_gain.sum())
        actual_total_pressure = total_actual_feed / max(total_actual_gain, eps)
        ref_total_pressure = total_ref_feed / max(total_ref_gain, eps)
        pressure_excess = max(0.0, (actual_total_pressure - ref_total_pressure) / max(ref_total_pressure, eps))
        # Also reward strong growth: if the actual trajectory produces equal or
        # better total growth, a moderate feed excess should not be classified as weak.
        growth_ratio = total_actual_gain / max(total_ref_gain, eps)
        growth_bonus = min(0.25, max(0.0, growth_ratio - 1.0) * 0.25)
        score = max(0.0, min(1.0, 1.0 - pressure_excess + growth_bonus))
        block["accuracy_score"] = score
        block.update(_accuracy_level(1.0 - score))
        block.update({
            "actual_total_feed": total_actual_feed,
            "actual_total_growth_signal": total_actual_gain,
            "reference_total_feed": total_ref_feed,
            "reference_total_growth_signal": total_ref_gain,
            "actual_feed_per_gain": actual_total_pressure,
            "reference_feed_per_gain": ref_total_pressure,
            "actual_growth_signal": actual_gain_name,
            "reference_growth_signal": ref_gain_name,
            "feed_accuracy_note": "This score is growth-adjusted; raw feed is not marked weak when proportional beef/growth is produced.",
        })
        return block
    block = _metric_block("Low feed consumption accuracy", actual_feed, ref_feed, "raw feed fallback because no usable growth signal was available", "minimize/reference")
    block["feed_accuracy_note"] = "Fallback raw-feed metric used; derive TBW/ADG/beef columns for a biological feed-efficiency score."
    return block

def _load_best_sarg_reference(config=None) -> pd.DataFrame:
    """Load the best SARG reference trajectory if it exists.

    The SARG generator writes a long daily CSV with a `reference_id` column and
    a JSON library with per-reference summaries. The best reference is selected
    by `simulator_score` when available; otherwise the first reference is used.
    """
    if config is None:
        return pd.DataFrame()
    try:
        ref_json = Path(getattr(config, "sarg_reference_library_json", ""))
        ref_daily = ref_json.parent / "sarg_growth_reference_daily.csv"
        if not ref_daily.exists():
            return pd.DataFrame()
        daily = pd.read_csv(ref_daily)
        if daily.empty:
            return pd.DataFrame()
        best_id = None
        if ref_json.exists():
            lib = json.loads(ref_json.read_text(encoding="utf-8"))
            refs = lib.get("reference_models") or lib.get("references") or []
            if refs:
                def score(x):
                    try:
                        return float(x.get("simulator_score", x.get("score", 0.0)))
                    except Exception:
                        return 0.0
                best = max(refs, key=score)
                best_id = best.get("reference_id")
        if best_id and "reference_id" in daily.columns:
            sub = daily[daily["reference_id"].astype(str) == str(best_id)].copy()
            if not sub.empty:
                return sub.sort_values("fattening_day") if "fattening_day" in sub.columns else sub
        if "reference_id" in daily.columns:
            first = str(daily["reference_id"].dropna().astype(str).iloc[0])
            return daily[daily["reference_id"].astype(str) == first].copy()
        return daily
    except Exception:
        return pd.DataFrame()


def _accuracy_report_payload(inner_df: pd.DataFrame, growth_df: pd.DataFrame | None, config=None, ss_kstore_dir=None, summary: dict | None = None) -> dict:
    summary_dir = Path(ss_kstore_dir) / "summaries" if ss_kstore_dir else None
    sampled = _read_csv_if_exists(summary_dir / "sampled_trace.csv") if summary_dir else pd.DataFrame()
    learning_daily = _read_csv_if_exists(summary_dir / "learning_daily.csv") if summary_dir else pd.DataFrame()
    daily_summary = _read_csv_if_exists(summary_dir / "daily_summary.csv") if summary_dir else pd.DataFrame()
    trace = sampled if not sampled.empty else inner_df
    objectives = []

    # 1) Thermal comfort: error is zero inside [LNZ, UNZ] and distance to nearest boundary outside.
    if trace is not None and not trace.empty:
        temp = _series(trace, "indoor_temp_c", 0.0)
        low = _series(trace, "lct_c", -5.0)
        high = _series(trace, "uct_c", 27.0)
        outside = pd.concat([(low - temp).clip(lower=0), (temp - high).clip(lower=0)], axis=1).max(axis=1)
        objectives.append(_metric_block("Thermal comfort accuracy", outside, pd.Series([0.0] * len(outside)), "zero outside-band temperature error", "minimize"))
        comfort_band_rate = float((outside <= 1e-9).mean()) if len(outside) else 0.0
    elif not learning_daily.empty and "comfort_violation_rate" in learning_daily.columns:
        actual = pd.to_numeric(learning_daily["comfort_violation_rate"], errors="coerce").fillna(0.0)
        objectives.append(_metric_block("Thermal comfort accuracy", actual, pd.Series([0.0] * len(actual)), "zero comfort violation rate", "minimize"))
        comfort_band_rate = float(1.0 - actual.mean()) if len(actual) else 0.0
    else:
        comfort_band_rate = float((summary or {}).get("time_in_comfort_band_rate", 0.0))

    # 2) Low energy: evaluate controllable climate energy against an adaptive
    # weather-normalized budget.  Comparing against absolute zero made this
    # metric report 0% even when comfort was excellent and actuators were almost
    # unused.
    energy_block = _energy_management_accuracy_block(trace, learning_daily, daily_summary, config=config)
    if energy_block is not None:
        objectives.append(energy_block)

    # 3 & 4) Feed and growth: compare growth simulator output against best SARG reference trajectory.
    ref = _load_best_sarg_reference(config)
    if growth_df is not None and not growth_df.empty:
        g = growth_df.copy()
        if "fattening_day" in g.columns and not ref.empty and "fattening_day" in ref.columns:
            g["fattening_day"] = pd.to_numeric(g["fattening_day"], errors="coerce").astype("Int64")
            ref["fattening_day"] = pd.to_numeric(ref["fattening_day"], errors="coerce").astype("Int64")
            merged = g.merge(ref, on="fattening_day", how="left", suffixes=("", "_ref"))
        else:
            merged = g.reset_index(drop=True).copy()
            for c in ref.columns if not ref.empty else []:
                merged[f"{c}_ref"] = ref[c].reset_index(drop=True).reindex(merged.index)

        feed_col = _pick_numeric_column(g, ["feed_intake_kg_dm_day", "feed_intake", "fi", "FI"])
        if feed_col:
            objectives.append(_growth_adjusted_feed_accuracy_block(merged, ref, feed_col))

        # Growth/beef production uses TBW first, then beef production, then ADG.
        growth_col = _pick_numeric_column(g, ["tbw_kg", "TBW", "body_weight", "beef_production_kg", "beef_production", "adg_kg_day", "ADG"])
        if growth_col:
            ref_col = f"{growth_col}_ref" if f"{growth_col}_ref" in merged.columns else None
            if ref_col is None and not ref.empty:
                rc = _pick_numeric_column(ref, [growth_col, "tbw_kg", "TBW", "body_weight", "beef_production_kg", "beef_production", "adg_kg_day", "ADG"])
                if rc and f"{rc}_ref" in merged.columns:
                    ref_col = f"{rc}_ref"
            actual = pd.to_numeric(merged[growth_col], errors="coerce").fillna(0.0)
            if ref_col and ref_col in merged.columns:
                target = pd.to_numeric(merged[ref_col], errors="coerce").ffill().fillna(actual)
                ref_name = "best SARG growth-simulator reference growth trajectory"
            else:
                target = actual.expanding().max()
                ref_name = "best observed growth fallback"
            objectives.append(_metric_block("Beef production / growth accuracy", actual, target, ref_name, "maximize/reference"))

    # Add validation/evaluation indicators from validation_report.html summary.
    evaluation = {k: v for k, v in (summary or {}).items() if k not in {"accuracy_metrics", "last_growth_state"}}
    payload = {
        "report_type": "trans_domain_accuracy_report",
        "accuracy_reference_note": "Four-objective accuracy follows doc/ACCURACY_REPORT.md: thermal comfort, low energy, low feed, and beef/growth relative to comfort band or SARG reference where available.",
        "comfort_band_rate": comfort_band_rate,
        "objectives": objectives,
        "evaluation_indicators": evaluation,
        "validation_accuracy_metrics": (summary or {}).get("accuracy_metrics", {}),
        "last_growth_state": (summary or {}).get("last_growth_state", {}),
    }
    return payload


def _write_accuracy_report(report_dir: Path, payload: dict) -> dict:
    acc_dir = report_dir / "accuracy"
    data_dir = report_dir / "data" / "accuracy"
    acc_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "accuracy_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    rows = []
    for obj in payload.get("objectives", []):
        rows.append({k: obj.get(k) for k in ["objective", "reference", "direction", "rmse", "mean_actual", "mae", "mse", "r2", "rmse_over_mean_actual", "accuracy_score", "label"]})
    pd.DataFrame(rows).to_csv(data_dir / "accuracy_indicators.csv", index=False)
    cards = []
    for obj in payload.get("objectives", []):
        color = obj.get("color", "#777")
        cards.append(f"<div class='card'><div class='muted'>{obj.get('objective')}</div><div class='metric'>{float(obj.get('accuracy_score',0.0)):.2%}</div><div style='color:white;background:{color};display:inline-block;border-radius:999px;padding:3px 9px'>{obj.get('label')}</div><p class='muted'>{obj.get('reference')}</p></div>")
    eval_html = pd.DataFrame([payload.get("evaluation_indicators", {})]).T.reset_index().rename(columns={"index":"indicator",0:"value"}).to_html(index=False)
    body = "<h1>Trans-domain accuracy report</h1><p>This page expands the Accuracy indicators into four TDDT objectives and also includes Evaluation indicators from validation_report.html.</p><div class='grid'>" + "".join(cards) + "</div><h2>Objective accuracy indicators</h2><canvas id='acc'></canvas><h2>Evaluation indicators</h2>" + eval_html + "<h2>Raw data</h2><p><a href='../data/accuracy/accuracy_report.json'>accuracy_report.json</a> · <a href='../data/accuracy/accuracy_indicators.csv'>accuracy_indicators.csv</a></p>"
    chart = {
        "labels": [o.get("objective") for o in payload.get("objectives", [])],
        "accuracy": [float(o.get("accuracy_score", 0.0)) for o in payload.get("objectives", [])],
        "rmse": [float(o.get("rmse", 0.0)) for o in payload.get("objectives", [])],
        "mae": [float(o.get("mae", 0.0)) for o in payload.get("objectives", [])],
        "relative_error": [float(o.get("rmse_over_mean_actual", 0.0)) for o in payload.get("objectives", [])],
    }
    script = """
barChart('acc', data.labels, data.accuracy, 'Accuracy score');
"""
    (acc_dir / "index.html").write_text(_html_shell("Trans-domain accuracy report", body, chart, script), encoding="utf-8")
    return {"accuracy_report": str(acc_dir / "index.html"), "accuracy_report_json": str(data_dir / "accuracy_report.json"), "accuracy_indicators_csv": str(data_dir / "accuracy_indicators.csv")}



def _plotly_shell(title: str, body: str, data: dict | None = None, script: str = "") -> str:
    payload = json.dumps(data or {}, ensure_ascii=False, default=str)
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{title}</title><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<style>
body{{font-family:Arial,sans-serif;margin:24px;color:#222}} a{{text-decoration:none;color:#0b5cad}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}} .card{{border:1px solid #ddd;border-radius:12px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}} .muted{{color:#666;font-size:.9rem}} .plot{{width:100%;height:620px;margin:20px 0 36px}} table{{border-collapse:collapse;width:100%;font-size:.85rem}} th,td{{border:1px solid #ddd;padding:4px 6px}} th{{background:#f6f6f6}} nav a{{margin-right:12px}}
</style></head><body><nav><a href='../index.html'>Report index</a><a href='index.html'>Advanced 3D Surface and radar reports</a></nav>{body}
<script>
const data = {payload};
function num(arr) {{ return (arr || []).map(v => {{ const x = Number(v); return Number.isFinite(x) ? x : 0; }}); }}
function txt(arr) {{ return (arr || []).map(v => String(v)); }}
{script}
</script></body></html>"""


def _numeric_list(df: pd.DataFrame, col: str, default: float = 0.0, max_points: int = 1500) -> list:
    if df is None or df.empty or col not in df.columns:
        return []
    x = _thin(df, max_points=max_points)
    return pd.to_numeric(x[col], errors="coerce").fillna(default).round(6).tolist()


def _text_list(df: pd.DataFrame, col: str, max_points: int = 1500) -> list:
    if df is None or df.empty or col not in df.columns:
        return []
    x = _thin(df, max_points=max_points)
    return x[col].astype(str).tolist()


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _classify_pct(v: float, prefix: str) -> str:
    try:
        x = float(v)
    except Exception:
        x = 0.0
    if x <= 1e-9:
        return f"{prefix}:off"
    if x < 35:
        return f"{prefix}:low"
    if x < 75:
        return f"{prefix}:medium"
    return f"{prefix}:high"


def _write_plotly_page(path: Path, title: str, body: str, data: dict, script: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_plotly_shell(title, body, data, script), encoding="utf-8")



def _surface_grid_payload(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, bins: int = 14) -> dict:
    """Create a Plotly surface payload from irregular report rows.

    It bins X/Y and averages Z. Empty cells are filled by row/column means so
    browser-side Plotly can render a continuous surface even for sparse runs.
    """
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
        return {"x": [], "y": [], "z": [], "count": 0, "x_label": x_col, "y_label": y_col, "z_label": z_col}
    raw = pd.DataFrame({
        "x": pd.to_numeric(df[x_col], errors="coerce"),
        "y": pd.to_numeric(df[y_col], errors="coerce"),
        "z": pd.to_numeric(df[z_col], errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty:
        return {"x": [], "y": [], "z": [], "count": 0, "x_label": x_col, "y_label": y_col, "z_label": z_col}
    if raw["x"].nunique() < 2:
        raw["x"] = raw["x"] + np.linspace(-1e-6, 1e-6, len(raw))
    if raw["y"].nunique() < 2:
        raw["y"] = raw["y"] + np.linspace(-1e-6, 1e-6, len(raw))
    xb = np.linspace(float(raw["x"].min()), float(raw["x"].max()), max(3, bins))
    yb = np.linspace(float(raw["y"].min()), float(raw["y"].max()), max(3, bins))
    raw["xi"] = np.clip(np.digitize(raw["x"], xb, right=False) - 1, 0, len(xb) - 1)
    raw["yi"] = np.clip(np.digitize(raw["y"], yb, right=False) - 1, 0, len(yb) - 1)
    grid = raw.groupby(["yi", "xi"])["z"].mean().unstack(fill_value=np.nan)
    grid = grid.reindex(index=range(len(yb)), columns=range(len(xb)))
    grid = grid.astype(float)
    grid = grid.apply(lambda row: row.fillna(row.mean()), axis=1)
    grid = grid.apply(lambda col: col.fillna(col.mean()), axis=0)
    overall = float(raw["z"].mean()) if not raw.empty else 0.0
    grid = grid.fillna(overall)
    return {
        "x": [round(float(v), 6) for v in xb.tolist()],
        "y": [round(float(v), 6) for v in yb.tolist()],
        "z": [[round(float(v), 6) for v in row] for row in grid.to_numpy().tolist()],
        "count": int(len(raw)),
        "x_label": x_col,
        "y_label": y_col,
        "z_label": z_col,
    }


def _growth_phase_summary(df: pd.DataFrame, day_col: str | None = None, phase_col: str | None = None, max_phases: int = 6) -> tuple[pd.DataFrame, dict]:
    """Return actual, data-derived growth phases for reporting.

    The report must not assume fixed 80/220/450/750/1000-day boundaries.
    If the training/growth output already contains a phase/stage column, those
    phase labels are used directly.  Otherwise phases are derived from the
    observed fattening-day span in the current run and the legend records the
    exact day range and number of observed days per phase.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["phase_key", "phase_idx", "phase_label", "phase_name", "start_day", "end_day", "days_count"]), {"source": "empty"}
    d = df.copy()
    if day_col is None:
        day_col = _pick_col(d, ["fattening_day", "growth_day", "training_day", "day", "DOY"])
    if phase_col is None:
        phase_col = _pick_col(d, ["biological_phase", "growth_phase", "phase", "stage", "sarg_phase"])
    if day_col and day_col in d.columns:
        d["__day"] = pd.to_numeric(d[day_col], errors="coerce")
    else:
        d["__day"] = pd.Series(range(1, len(d) + 1), index=d.index, dtype=float)
        day_col = "__row_index_day"
    d["__day"] = d["__day"].ffill().bfill().fillna(pd.Series(range(1, len(d) + 1), index=d.index)).astype(float)
    used_existing = bool(phase_col and phase_col in d.columns and d[phase_col].notna().any())
    if used_existing:
        d["phase_key"] = d[phase_col].astype(str).replace({"nan": "unknown_phase", "None": "unknown_phase"})
        order = d.groupby("phase_key")["__day"].min().sort_values().index.tolist()
        idx_map = {k: i + 1 for i, k in enumerate(order)}
        d["phase_idx"] = d["phase_key"].map(idx_map).astype(int)
        source = f"existing_column:{phase_col}"
    else:
        valid_days = d["__day"].dropna()
        if valid_days.empty:
            d["phase_idx"] = 1
        else:
            # Use the observed day span.  The number of phases is bounded and
            # never hard-coded to biological constants.  For short runs, fewer
            # phases are generated so each phase has support.
            unique_days = int(max(1, valid_days.nunique()))
            n_phases = int(max(1, min(max_phases, unique_days)))
            if n_phases == 1:
                d["phase_idx"] = 1
            else:
                lo, hi = float(valid_days.min()), float(valid_days.max())
                edges = np.linspace(lo, hi + 1e-9, n_phases + 1)
                d["phase_idx"] = np.clip(np.digitize(d["__day"], edges[1:-1], right=False) + 1, 1, n_phases).astype(int)
        d["phase_key"] = d["phase_idx"].apply(lambda x: f"observed_phase_{int(x):02d}")
        source = "observed_day_span"
    summary_rows = []
    for phase_idx, part in d.groupby("phase_idx", sort=True):
        start = int(round(float(part["__day"].min()))) if not part.empty else 0
        end = int(round(float(part["__day"].max()))) if not part.empty else 0
        key = str(part["phase_key"].iloc[0]) if "phase_key" in part.columns and not part.empty else f"phase_{int(phase_idx):02d}"
        count = int(part["__day"].nunique()) if not part.empty else 0
        label = f"P{int(phase_idx):02d}: {key} (days {start}-{end}, n={count})"
        summary_rows.append({"phase_key": key, "phase_idx": int(phase_idx), "phase_label": label, "phase_name": key, "start_day": start, "end_day": end, "days_count": count})
    summary = pd.DataFrame(summary_rows).sort_values("phase_idx") if summary_rows else pd.DataFrame()
    meta = {"source": source, "day_column": day_col, "phase_column": phase_col if used_existing else None, "phase_count": int(len(summary))}
    return summary, meta


def _attach_growth_phases(df: pd.DataFrame, day_col: str | None = None, phase_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    phases, meta = _growth_phase_summary(df, day_col=day_col, phase_col=phase_col)
    if df is None or df.empty:
        return pd.DataFrame(), phases, meta
    d = df.copy()
    if day_col is None:
        day_col = _pick_col(d, ["fattening_day", "growth_day", "training_day", "day", "DOY"])
    if phase_col is None:
        phase_col = _pick_col(d, ["biological_phase", "growth_phase", "phase", "stage", "sarg_phase"])
    if day_col and day_col in d.columns:
        d["__day"] = pd.to_numeric(d[day_col], errors="coerce")
    else:
        d["__day"] = pd.Series(range(1, len(d) + 1), index=d.index, dtype=float)
    d["__day"] = d["__day"].ffill().bfill().fillna(pd.Series(range(1, len(d) + 1), index=d.index)).astype(float)
    if phase_col and phase_col in d.columns and d[phase_col].notna().any():
        label_map = dict(zip(phases["phase_key"], phases["phase_idx"])) if not phases.empty else {}
        d["phase_key"] = d[phase_col].astype(str).replace({"nan": "unknown_phase", "None": "unknown_phase"})
        d["phase_idx"] = d["phase_key"].map(label_map).fillna(1).astype(int)
    else:
        if phases.empty:
            d["phase_idx"] = 1
        else:
            bounds = phases[["phase_idx", "start_day", "end_day"]].to_dict(orient="records")
            def pick_phase(day):
                try:
                    x = float(day)
                except Exception:
                    return int(bounds[0]["phase_idx"]) if bounds else 1
                for b in bounds:
                    if float(b["start_day"]) <= x <= float(b["end_day"]):
                        return int(b["phase_idx"])
                return int(bounds[-1]["phase_idx"]) if bounds and x > float(bounds[-1]["end_day"]) else int(bounds[0]["phase_idx"])
            d["phase_idx"] = d["__day"].apply(pick_phase).astype(int)
        label_map = dict(zip(phases["phase_idx"], phases["phase_key"])) if not phases.empty else {1: "observed_phase_01"}
        d["phase_key"] = d["phase_idx"].map(label_map).fillna("observed_phase")
    label_map = dict(zip(phases["phase_idx"], phases["phase_label"])) if not phases.empty else {1: "P01"}
    d["phase_label"] = d["phase_idx"].map(label_map).fillna(d["phase_key"].astype(str))
    return d, phases, meta


def _phase_surface_payload(phase_df: pd.DataFrame, y_col: str, z_col: str, title_x: str = "Observed growth phase") -> dict:
    if phase_df is None or phase_df.empty or y_col not in phase_df.columns or z_col not in phase_df.columns:
        return {"x": [], "y": [], "z": [], "x_label": title_x, "y_label": y_col, "z_label": z_col, "phase_legend": []}
    payload = _surface_grid_payload(phase_df, "phase_idx", y_col, z_col, bins=max(3, min(12, int(len(phase_df)))))
    payload["x_label"] = title_x
    payload["x_tickvals"] = [int(v) for v in phase_df["phase_idx"].tolist()]
    payload["x_ticktext"] = [str(v) for v in phase_df["phase_label"].tolist()]
    payload["phase_legend"] = phase_df[["phase_idx", "phase_label", "start_day", "end_day", "days_count"]].to_dict(orient="records") if all(c in phase_df.columns for c in ["start_day", "end_day", "days_count"]) else []
    return payload


def _phase_legend_html(phases: pd.DataFrame, meta: dict | None = None) -> str:
    if phases is None or phases.empty:
        return "<p class='muted'>No phase metadata available.</p>"
    cols = [c for c in ["phase_idx", "phase_name", "start_day", "end_day", "days_count", "phase_label"] if c in phases.columns]
    note = ""
    if meta:
        note = f"<p class='muted'>Phase source: {meta.get('source','unknown')}; day column: {meta.get('day_column','unknown')}; phase column: {meta.get('phase_column') or 'derived from observed training days'}.</p>"
    return note + phases[cols].to_html(index=False)


def _fan_characteristic_payload(trace: pd.DataFrame) -> dict:
    """Build an exhaust fan pressure-command-flow characteristic surface.

    If measured pressure difference exists it uses sampled rows. Otherwise it
    produces a lightweight controller-side characteristic surface using the fan
    command as voltage equivalent and pressure range 0..40 Pa, matching
    doc/3D_SURFACE_PLOT.md.
    """
    pressure_col = _pick_col(trace, ["pressure_difference_pa", "pressure_diff_pa", "delta_pressure_pa"])
    vent_col = _pick_col(trace, ["ventilation_group_pct", "ventilation_pct"])
    flow_col = _pick_col(trace, ["volume_flow_m3h", "airflow_m3h", "volume_flow_rate_m3_h"])
    if pressure_col and vent_col and flow_col:
        payload = _surface_grid_payload(trace, pressure_col, vent_col, flow_col, bins=14)
        payload.update({"x_label": "Pressure difference Pa", "y_label": "Fan command %", "z_label": "Volume flow m3/h"})
        return payload
    pressure = np.linspace(0.0, 40.0, 15)
    command = np.linspace(0.0, 100.0, 15)
    z = []
    for cmd in command:
        row = []
        for p in pressure:
            flow = max(0.0, (cmd / 100.0)) * 10000.0 * max(0.0, 1.0 - p / 45.0) ** 0.55
            row.append(round(float(flow), 3))
        z.append(row)
    return {"x": pressure.round(3).tolist(), "y": command.round(3).tolist(), "z": z, "count": len(pressure) * len(command), "x_label": "Pressure difference Pa", "y_label": "Fan command %", "z_label": "Volume flow m3/h"}


def _radar_phase_from_day(day: float) -> str:
    d = _safe_float(day, 0.0)
    if d <= 80:
        return "early_growth"
    if d <= 220:
        return "post_weaning_adaptation"
    if d <= 450:
        return "frame_growth"
    if d <= 750:
        return "muscle_gain"
    if d <= 1000:
        return "finishing"
    return "late_finishing"


def _normalize_map(values: dict, inverse: bool = False) -> dict:
    nums = {k: _safe_float(v, 0.0) for k, v in values.items()}
    if not nums:
        return {}
    lo, hi = min(nums.values()), max(nums.values())
    if abs(hi - lo) < 1e-12:
        return {k: 0.5 for k in nums}
    out = {}
    for k, v in nums.items():
        x = (v - lo) / (hi - lo)
        out[k] = round(float(1.0 - x if inverse else x), 6)
    return out


def build_radar_reports(report_dir: str | Path, growth_df: pd.DataFrame | None = None, ss_kstore_dir: str | Path | None = None) -> dict:
    """Build radar reports following doc/RADAR_REPORT.md.

    The radar now uses actual training/growth phases.  Fixed day boundaries are
    not used; if the simulator output contains a phase/stage column it is used,
    otherwise phases are derived from the observed fattening-day span and the
    exact day range for each phase is displayed in the legend.
    """
    out = Path(report_dir)
    radar = out / "radar"
    data_dir = out / "data" / "radar"
    radar.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    g = growth_df.copy() if growth_df is not None else pd.DataFrame()
    if g.empty:
        alt = out / "outer_growth_state.csv"
        if alt.exists():
            try:
                g = pd.read_csv(alt)
            except Exception:
                g = pd.DataFrame()
    if g.empty:
        payload = {"labels": [], "series": [], "raw": [], "phase_metadata": [], "phase_source": "empty"}
        phase_table = "<p>No growth data available.</p>"
    else:
        day_col = _pick_col(g, ["fattening_day", "growth_day", "day", "DOY"])
        g, phases, phase_meta = _attach_growth_phases(g, day_col=day_col)
        feed_col = _pick_col(g, ["feed_intake_kg_dm_day", "feed_intake", "FI", "fi"])
        fe_col = _pick_col(g, ["feed_efficiency", "FE"])
        adg_col = _pick_col(g, ["adg_kg_day", "ADG", "adg"])
        tbw_col = _pick_col(g, ["tbw_kg", "TBW", "body_weight"])
        beef_col = _pick_col(g, ["beef_production_kg", "beef_production", "beef_event_kg"])
        hp_col = _pick_col(g, ["heat_production", "HP"])
        stress_col = _pick_col(g, ["thermal_stress_index", "heat_stress", "cold_stress"])
        conf_col = _pick_col(g, ["sarg_score", "reference_score", "confidence"])
        rows = []
        for phase_idx, part in g.groupby("phase_idx", sort=True):
            phase_row = phases[phases["phase_idx"] == phase_idx].iloc[0].to_dict() if not phases.empty and (phases["phase_idx"] == phase_idx).any() else {}
            tbw = pd.to_numeric(part[tbw_col], errors="coerce") if tbw_col else pd.Series([], dtype=float)
            feed = pd.to_numeric(part[feed_col], errors="coerce") if feed_col else pd.Series([], dtype=float)
            adg = pd.to_numeric(part[adg_col], errors="coerce") if adg_col else pd.Series([], dtype=float)
            beef = pd.to_numeric(part[beef_col], errors="coerce") if beef_col else pd.Series([], dtype=float)
            hp = pd.to_numeric(part[hp_col], errors="coerce") if hp_col else pd.Series([], dtype=float)
            fe = pd.to_numeric(part[fe_col], errors="coerce") if fe_col else pd.Series([], dtype=float)
            stress = pd.to_numeric(part[stress_col], errors="coerce") if stress_col else pd.Series([], dtype=float)
            conf = pd.to_numeric(part[conf_col], errors="coerce") if conf_col else pd.Series([], dtype=float)
            tbw_gain = float(tbw.max() - tbw.min()) if not tbw.dropna().empty else float(adg.sum()) if not adg.dropna().empty else 0.0
            feed_mean = float(feed.mean()) if not feed.dropna().empty else 0.0
            feed_total = float(feed.sum()) if not feed.dropna().empty else 0.0
            adg_mean = float(adg.mean()) if not adg.dropna().empty else 0.0
            beef_total = float(beef.sum()) if not beef.dropna().empty else tbw_gain
            feed_per_kg_gain = feed_total / max(tbw_gain, 1e-9) if feed_total > 0 or tbw_gain > 0 else 0.0
            fe_value = float(fe.mean()) if not fe.dropna().empty else (adg_mean / max(feed_mean, 1e-9) if feed_mean > 0 else 0.0)
            rows.append({
                "phase_idx": int(phase_idx),
                "phase": phase_row.get("phase_label", f"P{int(phase_idx):02d}"),
                "phase_name": phase_row.get("phase_name", f"phase_{int(phase_idx):02d}"),
                "start_day": phase_row.get("start_day"),
                "end_day": phase_row.get("end_day"),
                "days_count": phase_row.get("days_count"),
                "feed_intake_mean": feed_mean,
                "feed_intake_total": feed_total,
                "feed_efficiency": fe_value,
                "ADG": adg_mean,
                "TBW_gain": tbw_gain,
                "beef_production": beef_total,
                "heat_production": float(hp.mean()) if not hp.dropna().empty else 0.0,
                "thermal_stress": float(stress.mean()) if not stress.dropna().empty else 0.0,
                "diet_guidance_confidence": float(conf.mean()) if not conf.dropna().empty else 0.5,
                "feed_per_kg_gain": feed_per_kg_gain,
            })
        raw = pd.DataFrame(rows).sort_values("phase_idx")
        # Radar axes are all transformed so larger values mean better phase performance.
        axis_specs = [
            ("low_feed_pressure", "feed_per_kg_gain", True),
            ("feed_efficiency", "feed_efficiency", False),
            ("ADG", "ADG", False),
            ("TBW_gain", "TBW_gain", False),
            ("beef_production", "beef_production", False),
            ("low_heat_pressure", "heat_production", True),
            ("low_thermal_stress", "thermal_stress", True),
            ("diet_guidance_confidence", "diet_guidance_confidence", False),
        ]
        normalized = {label: _normalize_map(dict(zip(raw["phase"], raw[src])), inverse=inv) for label, src, inv in axis_specs}
        labels = [label for label, _, _ in axis_specs]
        series = []
        for _, r in raw.iterrows():
            phase = r["phase"]
            series.append({"name": phase, "values": [normalized[m].get(phase, 0.5) for m in labels]})
        payload = {
            "labels": labels,
            "series": series,
            "raw": raw.to_dict(orient="records"),
            "phase_metadata": phases.to_dict(orient="records") if not phases.empty else [],
            "phase_source": phase_meta,
            "note": "Values are normalized by actual observed growth phase. Axes are direction-aligned: larger values always mean better performance. Phase ranges are derived from the current training/growth data, not hard-coded day limits.",
        }
        phase_table = _phase_legend_html(phases, phase_meta)
    (data_dir / "feed_growth_phase_radar.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if payload.get("raw"):
        pd.DataFrame(payload["raw"]).to_csv(data_dir / "feed_growth_phase_radar.csv", index=False)
    else:
        pd.DataFrame().to_csv(data_dir / "feed_growth_phase_radar.csv", index=False)
    body = "<h1>Feed consumption radar by actual growth phase</h1><p class='muted'>This radar compares feed pressure, feed efficiency, growth output, heat pressure, and confidence across the growth phases observed in this training run. It follows doc/RADAR_REPORT.md.</p><div id='plot' class='plot'></div><h2>Phase legend</h2>" + phase_table + "<h2>Interpretation</h2><p>All axes are normalized to 0–1 and direction-aligned. A larger value is better. Low feed pressure is based on feed per kg gain where available.</p>"
    script = """
const traces = (data.series || []).map(s => ({type:'scatterpolar', r:num(s.values).concat([num(s.values)[0] || 0]), theta:(data.labels || []).concat([(data.labels || [])[0] || '']), fill:'toself', name:s.name}));
Plotly.newPlot('plot', traces, {margin:{l:40,r:40,b:40,t:40}, polar:{radialaxis:{visible:true, range:[0,1]}}, showlegend:true});
"""
    _write_plotly_page(radar / "index.html", "Feed growth phase radar", body, payload, script)
    adv = out / "advanced_3d"
    adv.mkdir(parents=True, exist_ok=True)
    (adv / "feed_growth_phase_radar.html").write_text((radar / "index.html").read_text(encoding="utf-8"), encoding="utf-8")
    return {"radar_index": str(radar / "index.html"), "feed_growth_phase_radar": str(adv / "feed_growth_phase_radar.html")}

def build_advanced_3d_reports(report_dir: str | Path, ss_kstore_dir: str | Path | None, growth_df: pd.DataFrame | None = None, summary: dict | None = None) -> dict:
    """Build lightweight 3D/advanced reports from SS-KStore summaries.

    The pages follow doc/3D_SURFACE_PLOT.md and use sampled/aggregated data only,
    so a 1000-day TRAIN run does not embed the full 5-minute trace in HTML.
    """
    out = Path(report_dir)
    adv = out / "advanced_3d"
    adv.mkdir(parents=True, exist_ok=True)
    summary_dir = Path(ss_kstore_dir) / "summaries" if ss_kstore_dir else None
    sampled = _read_csv_if_exists(summary_dir / "sampled_trace.csv") if summary_dir else pd.DataFrame()
    daily = _read_csv_if_exists(summary_dir / "daily_summary.csv") if summary_dir else pd.DataFrame()
    learning_daily = _read_csv_if_exists(summary_dir / "learning_daily.csv") if summary_dir else pd.DataFrame()
    growth = growth_df if growth_df is not None else pd.DataFrame()

    outputs = {}
    # 0) Exhaust fan characteristic surface.
    fan_payload = _fan_characteristic_payload(sampled)
    _write_plotly_page(
        adv / "fan_pressure_voltage_flow.html",
        "Exhaust fan characteristic surface",
        "<h1>Exhaust fan characteristic surface</h1><p class='muted'>X = pressure difference, Y = fan command / voltage equivalent, Z = volume flow rate. If measured pressure/flow are unavailable, a lightweight controller-side characteristic surface is generated.</p><div id='plot' class='plot'></div>",
        fan_payload,
        """
Plotly.newPlot('plot', [{type:'surface', x:num(data.x), y:num(data.y), z:data.z || [], colorscale:'Jet', colorbar:{title:data.z_label || 'Flow'}}], {margin:{l:0,r:0,b:0,t:30}, scene:{xaxis:{title:data.x_label}, yaxis:{title:data.y_label}, zaxis:{title:data.z_label}}});
""",
    )
    outputs["fan_pressure_voltage_flow"] = str(adv / "fan_pressure_voltage_flow.html")

    # 1) Climate-energy-comfort 3D surface report.
    trace = sampled.copy()
    if not trace.empty:
        lct = _series(trace, "lct_c", -5.0)
        uct = _series(trace, "uct_c", 27.0)
        temp = _series(trace, "indoor_temp_c", 0.0)
        comfort_error = pd.concat([(lct - temp).clip(lower=0), (temp - uct).clip(lower=0)], axis=1).max(axis=1)
        actuator_energy = (_series(trace, "electric_kw", 0.0) + _series(trace, "gas_kw", 0.0)).replace([np.inf, -np.inf], 0).fillna(0)
        if actuator_energy.abs().sum() <= 1e-9:
            actuator_energy = _series(trace, "ventilation_group_pct", 0.0) / 100.0 + _series(trace, "heating_group_pct", 0.0) / 100.0
        tmp_surface_df = pd.DataFrame({"outdoor_temp_c": _series(trace, "outdoor_temp_c", 0.0), "actuator_energy_index": actuator_energy, "comfort_error_c": comfort_error})
        climate_payload = _surface_grid_payload(tmp_surface_df, "outdoor_temp_c", "actuator_energy_index", "comfort_error_c")
    else:
        climate_payload = {"x": [], "y": [], "z": [], "text": []}
    _write_plotly_page(
        adv / "climate_energy_3d.html",
        "3D climate-energy-comfort report",
        "<h1>3D climate–energy–comfort report</h1><p class='muted'>X = outdoor temperature, Y = controllable energy/actuator-use index, Z = comfort error outside the dynamic band. Built from SS-KStore sampled trace.</p><div id='plot' class='plot'></div>",
        climate_payload,
        """
Plotly.newPlot('plot', [{type:'surface', x:num(data.x), y:num(data.y), z:data.z || [], colorscale:'Viridis', contours:{z:{show:true,usecolormap:true,highlightcolor:'#42f462',project:{z:true}}}, colorbar:{title:data.z_label || 'Z'}}], {margin:{l:0,r:0,b:0,t:30}, scene:{xaxis:{title:data.x_label || 'Outdoor temp C'}, yaxis:{title:data.y_label || 'Energy / actuator index'}, zaxis:{title:data.z_label || 'Comfort error'}}});
""",
    )
    outputs["climate_energy_3d"] = str(adv / "climate_energy_3d.html")

    # 2) Growth-feed-production 3D report aggregated by actual observed phases.
    g = growth.copy()
    xcol = _pick_col(g, ["fattening_day", "growth_day", "day", "DOY"])
    ycol = _pick_col(g, ["feed_intake_kg_dm_day", "feed_intake", "FI", "fi"])
    zcol = _pick_col(g, ["tbw_kg", "TBW", "body_weight", "beef_production_kg", "beef_production", "ADG", "adg_kg_day"])
    ccol = _pick_col(g, ["heat_production", "HP", "thermal_stress_index", "feed_efficiency", "FE"])
    g_phase, phases, phase_meta = _attach_growth_phases(g, day_col=xcol) if not g.empty else (pd.DataFrame(), pd.DataFrame(), {"source":"empty"})
    phase_rows = []
    if not g_phase.empty and ycol and zcol:
        for phase_idx, part in g_phase.groupby("phase_idx", sort=True):
            prow = phases[phases["phase_idx"] == phase_idx].iloc[0].to_dict() if not phases.empty and (phases["phase_idx"] == phase_idx).any() else {}
            row = {"phase_idx": int(phase_idx), "phase_label": prow.get("phase_label", f"P{int(phase_idx):02d}"), "start_day": prow.get("start_day"), "end_day": prow.get("end_day"), "days_count": prow.get("days_count")}
            row[ycol] = float(pd.to_numeric(part[ycol], errors="coerce").mean())
            row[zcol] = float(pd.to_numeric(part[zcol], errors="coerce").mean())
            if ccol:
                row[ccol] = float(pd.to_numeric(part[ccol], errors="coerce").mean())
            phase_rows.append(row)
    phase_df = pd.DataFrame(phase_rows)
    growth_payload = _phase_surface_payload(phase_df, ycol or "feed", zcol or "growth", title_x="Actual observed growth phase")
    growth_payload["phase_source"] = phase_meta
    growth_payload["c_label"] = ccol or "color"
    _write_plotly_page(
        adv / "growth_feed_3d.html",
        "3D growth-feed-production report",
        "<h1>3D growth–feed–production report</h1><p class='muted'>X is the actual observed growth phase, not raw fattening_day. The legend lists the exact day range and number of observed days per phase.</p><div id='plot' class='plot'></div><h2>Observed phase legend</h2>" + _phase_legend_html(phases, phase_meta),
        growth_payload,
        """
const layout = {margin:{l:0,r:0,b:0,t:30}, scene:{xaxis:{title:data.x_label, tickmode:'array', tickvals:data.x_tickvals || data.x, ticktext:data.x_ticktext || data.x}, yaxis:{title:data.y_label}, zaxis:{title:data.z_label}}};
Plotly.newPlot('plot', [{type:'surface', x:num(data.x), y:num(data.y), z:data.z || [], colorscale:'Plasma', colorbar:{title:data.z_label || 'Z'}}], layout);
""",
    )
    outputs["growth_feed_3d"] = str(adv / "growth_feed_3d.html")

    # 3) Actuator-decision surface.
    score_col = "mpc_score" if "mpc_score" in trace.columns else "comfort_error"
    actuator_payload = _surface_grid_payload(trace, "ventilation_group_pct", "heating_group_pct", score_col)
    _write_plotly_page(
        adv / "actuator_decision_3d.html",
        "3D actuator decision report",
        "<h1>3D actuator decision report</h1><p class='muted'>X = ventilation, Y = heating, Z = MPC cost/comfort score. Useful for detecting high-heat/high-exhaust conflict regions.</p><div id='plot' class='plot'></div>",
        actuator_payload,
        """
Plotly.newPlot('plot', [{type:'surface', x:num(data.x), y:num(data.y), z:data.z || [], colorscale:'Turbo', colorbar:{title:data.z_label || 'MPC score'}}], {margin:{l:0,r:0,b:0,t:30}, scene:{xaxis:{title:data.x_label || 'Ventilation %'}, yaxis:{title:data.y_label || 'Heating %'}, zaxis:{title:data.z_label || 'MPC score / comfort error'}}});
""",
    )
    outputs["actuator_decision_3d"] = str(adv / "actuator_decision_3d.html")

    # 4) Learning progress report. Prefer true learning summaries; when they
    # are missing, rebuild a compact learning trace from sampled inner-loop
    # telemetry so the report is never blank after offline rebuild.
    learn = learning_daily.copy()
    if learn.empty and not trace.empty and "timestamp" in trace.columns:
        ttmp = trace.copy()
        ttmp["timestamp"] = pd.to_datetime(ttmp["timestamp"], errors="coerce")
        ttmp = ttmp.dropna(subset=["timestamp"])
        if not ttmp.empty:
            ttmp["period"] = ttmp["timestamp"].dt.date.astype(str)
            agg = {}
            for src, dst in [
                ("comfort_error", "comfort_error_mean"),
                ("mpc_score", "mpc_cost_mean"),
                ("reward", "reward_mean"),
                ("rl_q_delta", "rl_q_delta_mean"),
                ("rl_td_error", "rl_td_error_mean"),
                ("switch_penalty", "switching_penalty_mean"),
                ("heating_ventilation_conflict_penalty", "conflict_penalty_mean"),
                ("model_uncertainty_radius", "uncertainty_radius_mean"),
                ("learning_elapsed_sec", "learning_elapsed_sec"),
            ]:
                if src in ttmp.columns:
                    agg[src] = "max" if src == "learning_elapsed_sec" else "mean"
            learn = ttmp.groupby("period", as_index=False).agg(agg).rename(columns={k:v for k,v in [
                ("comfort_error", "comfort_error_mean"), ("mpc_score", "mpc_cost_mean"),
                ("reward", "reward_mean"), ("rl_q_delta", "rl_q_delta_mean"),
                ("rl_td_error", "rl_td_error_mean"), ("switch_penalty", "switching_penalty_mean"),
                ("heating_ventilation_conflict_penalty", "conflict_penalty_mean"),
                ("model_uncertainty_radius", "uncertainty_radius_mean"),
            ] if k in ttmp.columns})
    if not learn.empty:
        lwork = learn.copy().reset_index(drop=True)
        lwork["training_day_index"] = np.arange(1, len(lwork) + 1)
        if not phases.empty:
            bounds = phases[["phase_idx", "start_day", "end_day", "phase_label", "days_count"]].to_dict(orient="records")
            def lphase(i):
                for b in bounds:
                    if float(b["start_day"]) <= float(i) <= float(b["end_day"]):
                        return int(b["phase_idx"])
                return int(bounds[-1]["phase_idx"]) if bounds and i > float(bounds[-1]["end_day"]) else int(bounds[0]["phase_idx"])
            lwork["phase_idx"] = lwork["training_day_index"].apply(lphase).astype(int)
            lphases = phases.copy()
            lmeta = {"source":"mapped_to_growth_phase_ranges", "day_column":"training_day_index", "phase_column":None}
        else:
            lwork, lphases, lmeta = _attach_growth_phases(lwork, day_col="training_day_index")
        rows=[]
        for phase_idx, part in lwork.groupby("phase_idx", sort=True):
            prow = lphases[lphases["phase_idx"] == phase_idx].iloc[0].to_dict() if not lphases.empty and (lphases["phase_idx"] == phase_idx).any() else {}
            cost = pd.to_numeric(part.get("mpc_cost_mean", pd.Series([0.0]*len(part))), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            reward = pd.to_numeric(part.get("reward_mean", pd.Series([0.0]*len(part))), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            qdelta = pd.to_numeric(part.get("rl_q_delta_mean", pd.Series([0.0]*len(part))), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            elapsed = pd.to_numeric(part.get("learning_elapsed_sec", pd.Series([0.0]*len(part))), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            rows.append({
                "phase_idx": int(phase_idx),
                "phase_label": prow.get("phase_label", f"P{int(phase_idx):02d}"),
                "start_day": prow.get("start_day"),
                "end_day": prow.get("end_day"),
                "days_count": int(len(part)),
                "mpc_cost_mean": float(cost.mean()),
                "mpc_cost_log1p": float(np.log1p(abs(cost.mean()))),
                "reward_mean": float(reward.mean()),
                "rl_q_delta_mean": float(qdelta.mean()),
                "learning_elapsed_sec": float(elapsed.max() if len(elapsed) else 0.0),
            })
        ldf = pd.DataFrame(rows)
        learning_payload = {
            "points": ldf.to_dict(orient="records"),
            "x": ldf["phase_idx"].astype(float).round(6).tolist() if not ldf.empty else [],
            "y": ldf["mpc_cost_log1p"].astype(float).round(6).tolist() if not ldf.empty else [],
            "z": ldf["reward_mean"].astype(float).round(6).tolist() if not ldf.empty else [],
            "c": ldf["rl_q_delta_mean"].astype(float).round(6).tolist() if not ldf.empty else [],
            "x_label": "Actual observed training/growth phase",
            "y_label": "log1p(MPC cost mean)",
            "z_label": "reward_mean",
            "x_tickvals": ldf["phase_idx"].astype(int).tolist() if not ldf.empty else [],
            "x_ticktext": ldf["phase_label"].astype(str).tolist() if not ldf.empty else [],
            "phase_legend": lphases.to_dict(orient="records") if not lphases.empty else [],
            "phase_source": lmeta,
            "data_note": "Uses learning_daily.csv when available; otherwise reconstructs a compact learning trace from sampled inner-loop telemetry. The Y axis is log1p-scaled so large MPC-cost ranges do not make the 3D report appear blank.",
        }
    else:
        ldf = pd.DataFrame(); lphases = pd.DataFrame(); lmeta = {"source":"empty"}
        learning_payload = {"points": [], "x": [], "y": [], "z": [], "c": [], "x_label": "training_phase", "y_label": "log1p(MPC cost mean)", "z_label": "reward_mean", "phase_legend": [], "data_note":"No learning summaries or sampled telemetry were found."}
    _write_plotly_page(
        adv / "learning_surface.html",
        "3D learning progress report",
        "<h1>3D learning progress report</h1><p class='muted'>This report uses a robust 3D scatter/line visualization instead of a surface-only plot. It remains visible even for short runs or sparse offline rebuilds.</p><p class='muted'>" + str(learning_payload.get("data_note", "")) + "</p><div id='plot' class='plot'></div><h2>Observed phase legend</h2>" + _phase_legend_html(lphases, lmeta),
        learning_payload,
        """
const x=num(data.x), y=num(data.y), z=num(data.z), c=num(data.c);
const labels=(data.x_ticktext||[]);
const trace3d={type:'scatter3d', mode:'lines+markers+text', x:x, y:y, z:z, text:labels, textposition:'top center', marker:{size:7, color:c, colorscale:'Viridis', showscale:true, colorbar:{title:'RL Q-delta'}}, line:{width:4, color:'#2c3e50'}, name:'learning phases'};
const layout={margin:{l:0,r:0,b:0,t:40}, scene:{xaxis:{title:data.x_label||'Training phase', tickmode:'array', tickvals:data.x_tickvals||x, ticktext:data.x_ticktext||x}, yaxis:{title:data.y_label||'log MPC cost'}, zaxis:{title:data.z_label||'Reward'}}, annotations:[{text:'Sparse runs are shown as a 3D phase trajectory, not as an empty surface.', showarrow:false, x:0, y:1.08, xref:'paper', yref:'paper'}]};
Plotly.newPlot('plot', [trace3d], layout);
""",
    )
    outputs["learning_surface"] = str(adv / "learning_surface.html")

    # 5) CCLL context surface: outdoor temp/RH -> wind/context terrain.
    if not trace.empty:
        ccll_payload = _surface_grid_payload(trace, "outdoor_temp_c", "outdoor_rh_pct", "outdoor_wind_m_s")
    else:
        ccll_payload = {"x": [], "y": [], "z": [], "x_label": "outdoor_temp_c", "y_label": "outdoor_rh_pct", "z_label": "outdoor_wind_m_s"}
    _write_plotly_page(
        adv / "ccll_context_surface.html",
        "CCLL climate context surface",
        "<h1>CCLL climate context surface</h1><p class='muted'>X = outdoor temperature, Y = outdoor humidity, Z = outdoor wind. This surface visualizes the local climate context space used by CCLL-SEL.</p><div id='plot' class='plot'></div>",
        ccll_payload,
        """
Plotly.newPlot('plot', [{type:'surface', x:num(data.x), y:num(data.y), z:data.z || [], colorscale:'Viridis', colorbar:{title:data.z_label || 'Wind'}}], {margin:{l:0,r:0,b:0,t:30}, scene:{xaxis:{title:data.x_label}, yaxis:{title:data.y_label}, zaxis:{title:data.z_label}}});
""",
    )
    outputs["ccll_context_surface"] = str(adv / "ccll_context_surface.html")

    # 6) SARG reference/growth surface: actual phase/feed -> growth output.
    if not phase_df.empty and ycol and zcol:
        sarg_payload = _phase_surface_payload(phase_df, ycol, zcol, title_x="Actual observed SARG/growth phase")
        sarg_payload["phase_source"] = phase_meta
    else:
        sarg_payload = {"x": [], "y": [], "z": [], "x_label": "growth_phase", "y_label": ycol or "feed", "z_label": zcol or "growth", "phase_legend": []}
    _write_plotly_page(
        adv / "sarg_reference_surface.html",
        "SARG growth reference surface",
        "<h1>SARG growth reference surface</h1><p class='muted'>X is the observed phase used during training. The legend shows the exact day range and support for each phase. No fixed 1000-day axis is used.</p><div id='plot' class='plot'></div><h2>Observed phase legend</h2>" + _phase_legend_html(phases, phase_meta),
        sarg_payload,
        """
const layout = {margin:{l:0,r:0,b:0,t:30}, scene:{xaxis:{title:data.x_label, tickmode:'array', tickvals:data.x_tickvals || data.x, ticktext:data.x_ticktext || data.x}, yaxis:{title:data.y_label}, zaxis:{title:data.z_label}}};
Plotly.newPlot('plot', [{type:'surface', x:num(data.x), y:num(data.y), z:data.z || [], colorscale:'Plasma', colorbar:{title:data.z_label || 'Growth'}}], layout);
""",
    )
    outputs["sarg_reference_surface"] = str(adv / "sarg_reference_surface.html")

    # 5/6) CCLL-SARG-MPC flow and Sankey decision flow.
    if not trace.empty:
        contexts = trace.get("climate_context_id", trace.get("context_id", pd.Series(["unknown"] * len(trace)))).astype(str)
        vents = _series(trace, "ventilation_group_pct", 0.0).apply(lambda v: _classify_pct(v, "vent"))
        heats = _series(trace, "heating_group_pct", 0.0).apply(lambda v: _classify_pct(v, "heat"))
        temp = _series(trace, "indoor_temp_c", 0.0)
        low = _series(trace, "lct_c", -5.0); high = _series(trace, "uct_c", 27.0)
        outcome = pd.Series(np.where((temp >= low) & (temp <= high), "comfort:inside_band", "comfort:violation"), index=trace.index)
        flow_df = pd.DataFrame({"context": contexts, "vent": vents, "heat": heats, "outcome": outcome})
        flow_df["decision"] = flow_df["vent"] + " | " + flow_df["heat"]
        grouped = flow_df.groupby(["context", "decision", "outcome"]).size().reset_index(name="value")
    else:
        grouped = pd.DataFrame(columns=["context", "decision", "outcome", "value"])

    def sankey_payload_from_grouped(df: pd.DataFrame) -> dict:
        labels = []
        def idx(x):
            if x not in labels:
                labels.append(x)
            return labels.index(x)
        src=[]; tgt=[]; val=[]
        for _, r in df.iterrows():
            c = str(r.get("context", "unknown")); d = str(r.get("decision", "unknown")); o = str(r.get("outcome", "unknown")); v = int(r.get("value", 0))
            src.append(idx(c)); tgt.append(idx(d)); val.append(v)
            src.append(idx(d)); tgt.append(idx(o)); val.append(v)
        return {"labels": labels, "source": src, "target": tgt, "value": val, "table": df.head(300).to_dict(orient="records")}

    flow_payload = sankey_payload_from_grouped(grouped)
    table_html = grouped.sort_values("value", ascending=False).head(80).to_html(index=False) if not grouped.empty else "<p>No flow data.</p>"
    sankey_script = """
Plotly.newPlot('plot', [{type:'sankey', orientation:'h', node:{pad:15, thickness:16, line:{color:'black', width:.3}, label:data.labels}, link:{source:data.source, target:data.target, value:data.value}}], {margin:{l:10,r:10,b:10,t:30}});
"""
    _write_plotly_page(
        adv / "ccll_sarg_mpc_flow.html",
        "CCLL-SARG-MPC decision flow",
        "<h1>CCLL–SARG–MPC decision flow</h1><p class='muted'>Aggregates climate context, actuator decision class, and outcome. If SARG identifiers are present in future summaries they can be added as an intermediate node.</p><div id='plot' class='plot'></div><h2>Top flow rows</h2>" + table_html,
        flow_payload,
        sankey_script,
    )
    outputs["ccll_sarg_mpc_flow"] = str(adv / "ccll_sarg_mpc_flow.html")
    _write_plotly_page(
        adv / "sankey_decision_flow.html",
        "Sankey trans-domain decision report",
        "<h1>Sankey trans-domain decision report</h1><p class='muted'>Climate context → actuator class → comfort outcome. This is a compact explanation of how learned context priors translated into decisions.</p><div id='plot' class='plot'></div>",
        flow_payload,
        sankey_script,
    )
    outputs["sankey_decision_flow"] = str(adv / "sankey_decision_flow.html")

    # Advanced report index.
    links = [
        ("fan_pressure_voltage_flow.html", "Exhaust fan characteristic surface", "Pressure difference, fan command, and volume flow rate."),
        ("climate_energy_3d.html", "3D climate–energy–comfort surface", "Outdoor climate pressure versus energy and comfort error."),
        ("growth_feed_3d.html", "3D growth–feed–production surface", "Phase-aggregated feed/use efficiency versus growth output."),
        ("actuator_decision_3d.html", "3D actuator decision cost surface", "Ventilation/heating decision surface and cost."),
        ("ccll_context_surface.html", "CCLL climate context surface", "Local climate context space."),
        ("sarg_reference_surface.html", "SARG growth reference surface", "Phase-aggregated stage-aware growth reference terrain."),
        ("learning_surface.html", "3D learning progress surface", "Phase-aggregated learning reward and MPC cost trend."),
        ("feed_growth_phase_radar.html", "Feed growth phase radar", "Radar comparison of feed pressure by biological growth phase."),
        ("ccll_sarg_mpc_flow.html", "CCLL–SARG–MPC flow", "Context-to-decision-to-outcome explanation."),
        ("sankey_decision_flow.html", "Sankey decision flow", "Compact trans-domain decision flow."),
    ]
    cards = "".join([f"<div class='card'><h3><a href='{href}'>{title}</a></h3><p class='muted'>{desc}</p></div>" for href, title, desc in links])
    (adv / "index.html").write_text(_plotly_shell("Advanced 3D Surface and radar reports", "<h1>Advanced 3D Surface, radar and decision-flow reports</h1><p>These reports follow doc/3D_SURFACE_PLOT.md and are built from sampled SS-KStore summaries, not from the full 5-minute trace.</p><div class='grid'>" + cards + "</div>"), encoding="utf-8")
    outputs["advanced_index"] = str(adv / "index.html")
    return outputs


def build_ccll_cluster_report(report_dir: str | Path, ccll_library_json: str | Path | None = None, ccll_daily_descriptors_csv: str | Path | None = None) -> dict:
    out = Path(report_dir)
    cdir = out / "ccll_clusters"
    data_dir = out / "data" / "ccll_clusters"
    cdir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    lib_path = Path(ccll_library_json) if ccll_library_json else None
    desc_path = Path(ccll_daily_descriptors_csv) if ccll_daily_descriptors_csv else None
    lib = {}
    if lib_path and lib_path.exists():
        try:
            lib = json.loads(lib_path.read_text(encoding="utf-8"))
        except Exception:
            lib = {}
    daily = _read_csv_if_exists(desc_path) if desc_path else pd.DataFrame()
    contexts = lib.get("contexts", []) if isinstance(lib, dict) else []
    centroids = []
    for ctx in contexts:
        cen = ctx.get("centroid", {}) or {}
        centroids.append({
            "context_id": ctx.get("context_id", ""),
            "context_name": ctx.get("context_name", ""),
            "support_count": ctx.get("support_count", 0),
            "t_mean": cen.get("t_mean", 0.0),
            "rh_mean": cen.get("rh_mean", 0.0),
            "wind_mean": cen.get("wind_mean", 0.0),
            "risk_profile": ctx.get("risk_profile", {}),
            "mpc_prior": ctx.get("mpc_prior", {}),
        })
    cent_df = pd.DataFrame(centroids)
    if not daily.empty:
        daily.to_csv(data_dir / "ccll_daily_cluster_points.csv", index=False)
    if not cent_df.empty:
        cent_df.to_csv(data_dir / "ccll_centroids.csv", index=False)
    payload = {
        "method": lib.get("method", "CCLL-SEL") if isinstance(lib, dict) else "CCLL-SEL",
        "assignment_method": lib.get("assignment_method", "nearest_centroid_clustering") if isinstance(lib, dict) else "nearest_centroid_clustering",
        "feature_columns": lib.get("feature_columns", []) if isinstance(lib, dict) else [],
        "points": daily.to_dict(orient="records") if not daily.empty else [],
        "centroids": centroids,
    }
    (data_dir / "ccll_cluster_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    body = """
<h1>CCLL-SEL nearest-centroid cluster report</h1>
<p class='muted'>Daily climate descriptors and the nearest CCLL centroids used as soft MPC/RL priors.</p>
<div id='plot' class='plot'></div><h2>Centroids</h2><div id='centroid_table'></div>
"""
    script = r"""
const pts = data.points || []; const cents = data.centroids || [];
const ids = [...new Set(pts.map(p => String(p.context_id || p.cluster_index || 'unknown')))];
const traces = ids.map(cid => { const rows = pts.filter(p => String(p.context_id || p.cluster_index || 'unknown') === cid); return {type:'scatter3d', mode:'markers', name:cid, x:num(rows.map(r=>r.t_mean)), y:num(rows.map(r=>r.rh_mean)), z:num(rows.map(r=>r.wind_mean)), text:rows.map(r=>`date=${r.date||''}<br>context=${r.context_id||''}<br>${r.context_name||''}`), marker:{size:3, opacity:0.75}}; });
traces.push({type:'scatter3d', mode:'markers+text', name:'centroids', x:num(cents.map(c=>c.t_mean)), y:num(cents.map(c=>c.rh_mean)), z:num(cents.map(c=>c.wind_mean)), text:cents.map(c=>String(c.context_id||'')), textposition:'top center', marker:{size:8, symbol:'diamond', color:'black'}});
Plotly.newPlot('plot', traces, {title:'CCLL nearest-centroid climate contexts', scene:{xaxis:{title:'Mean outdoor temp C'}, yaxis:{title:'Mean RH %'}, zaxis:{title:'Mean wind m/s'}}, margin:{l:0,r:0,b:0,t:50}});
let html='<table><thead><tr><th>Context</th><th>Name</th><th>Support</th><th>T mean</th><th>RH mean</th><th>Wind mean</th><th>MPC prior</th></tr></thead><tbody>';
cents.forEach(c=>{html+=`<tr><td>${c.context_id||''}</td><td>${c.context_name||''}</td><td>${c.support_count||0}</td><td>${Number(c.t_mean||0).toFixed(3)}</td><td>${Number(c.rh_mean||0).toFixed(3)}</td><td>${Number(c.wind_mean||0).toFixed(3)}</td><td><code>${JSON.stringify(c.mpc_prior||{})}</code></td></tr>`}); html+='</tbody></table>'; document.getElementById('centroid_table').innerHTML=html;
"""
    (cdir / "index.html").write_text(_plotly_shell("CCLL-SEL nearest-centroid clusters", body, payload, script), encoding="utf-8")
    return {"ccll_cluster_report": str(cdir / "index.html")}


def _ensure_index_link(index_path: Path, href: str, label: str) -> None:
    if not index_path.exists():
        return
    html = index_path.read_text(encoding="utf-8")
    if href in html:
        return
    item = f"<li><a href='{href}'>{label}</a></li>"
    if "</ul>" in html:
        html = html.replace("</ul>", item + "</ul>", 1)
    else:
        html += f"<p><a href='{href}'>{label}</a></p>"
    index_path.write_text(html, encoding="utf-8")


def rebuild_reports_from_artifacts(report_dir: str | Path, *, ss_kstore_dir: str | Path | None = None, growth_csv: str | Path | None = None, config=None, mode_name: str = "REBUILD-OFFLINE") -> dict:
    out = Path(report_dir); out.mkdir(parents=True, exist_ok=True)
    ss = Path(ss_kstore_dir) if ss_kstore_dir else Path(getattr(config, "ss_kstore_dir", out.parent / "working" / "ss_kstore"))
    sd = ss / "summaries"
    sampled = _read_csv_if_exists(sd / "sampled_trace.csv")
    tail = _read_csv_if_exists(sd / "tail_trace.csv")
    daily = _read_csv_if_exists(sd / "daily_summary.csv")
    inner_df = sampled if not sampled.empty else tail
    if inner_df.empty and not daily.empty:
        inner_df = daily.copy()
        if "period" in inner_df.columns and "timestamp" not in inner_df.columns:
            inner_df["timestamp"] = inner_df["period"]
        inner_df = inner_df.rename(columns={c: c[:-5] for c in inner_df.columns if c.endswith("_mean")})
    commands_df = inner_df[[c for c in ["timestamp", "ventilation_group_pct", "heating_group_pct", "light_on", "safety_status"] if c in inner_df.columns]].copy() if not inner_df.empty else pd.DataFrame()
    gpath = Path(growth_csv) if growth_csv else out / "outer_growth_state.csv"
    growth_df = _read_csv_if_exists(gpath)
    if growth_df.empty:
        growth_df = _read_csv_if_exists(Path(getattr(config, "setd_kstore_dir", out.parent / "models" / "setd_kstore")) / "daily_growth_state_memory.csv")
    result = build_reports(out, inner_df, commands_df, growth_df, config=config, ss_kstore_dir=ss, mode_name=mode_name)
    if config is not None:
        result.update(build_ccll_cluster_report(out, getattr(config, "ccll_library_json", None), getattr(config, "ccll_daily_descriptors_csv", None)))
        _ensure_index_link(out / "index.html", "ccll_clusters/index.html", "CCLL nearest-centroid cluster report")
    (out / "offline_report_rebuild_manifest.json").write_text(json.dumps({"mode": mode_name, "ss_kstore_dir": str(ss), "growth_csv": str(gpath)}, indent=2, ensure_ascii=False), encoding="utf-8")
    return result

def build_hierarchical_training_reports(report_dir: str | Path, ss_kstore_dir: str | Path | None, growth_df: pd.DataFrame | None = None, summary: dict | None = None) -> dict:
    """Build lightweight multi-resolution report pages from SS-KStore summaries.

    This avoids embedding the full 1000-day 5-minute trace in one HTML file.
    """
    out = Path(report_dir); out.mkdir(parents=True, exist_ok=True)
    if not ss_kstore_dir:
        return {}
    summary_dir = Path(ss_kstore_dir) / 'summaries'
    sampled = _read_csv_if_exists(summary_dir / 'sampled_trace.csv')
    daily = _read_csv_if_exists(summary_dir / 'daily_summary.csv')
    learning = {lvl: _read_csv_if_exists(summary_dir / f'learning_{lvl}.csv') for lvl in ['daily','weekly','monthly','quarterly','yearly']}
    result = {}
    sections = ['daily','weekly','monthly','quarterly','yearly']
    for section in sections:
        sec_dir = out / section; sec_dir.mkdir(parents=True, exist_ok=True)
        rows = daily.copy()
        if rows.empty:
            rows = pd.DataFrame()
        if section != 'daily' and not rows.empty:
            rows['__period'] = _period_from_timestamp(rows['period'], section)
            num = rows.select_dtypes(include=['number']).columns.tolist()
            grouped = rows.groupby('__period')[num].mean().reset_index().rename(columns={'__period':'period'})
            if 'steps' in rows.columns:
                grouped['steps'] = rows.groupby('__period')['steps'].sum().values
            rows = grouped
        links = []
        if not rows.empty:
            sampled_periods = pd.DataFrame()
            if not sampled.empty and 'timestamp' in sampled.columns:
                sampled_periods = sampled.copy()
                sampled_periods['__period'] = _period_from_timestamp(sampled_periods['timestamp'], section)
            for i, row in rows.iterrows():
                period = str(row.get('period', i))
                safe = period.replace(':','-').replace(' ','_')
                tr = sampled_periods[sampled_periods.get('__period','') == period] if not sampled_periods.empty else pd.DataFrame()
                gd = growth_df
                if growth_df is not None and not growth_df.empty and 'fattening_day' in growth_df.columns and section == 'daily':
                    try: gd = growth_df.iloc[[min(int(i), len(growth_df)-1)]]
                    except Exception: gd = growth_df.tail(1)
                _write_period_page(out, section, safe, f'{section.title()} report: {period}', row.to_dict(), tr, gd)
                links.append(f"<li><a href='{safe}.html'>{period}</a></li>")
        (sec_dir / 'index.html').write_text(_html_shell(f'{section.title()} reports', f"<h1>{section.title()} reports</h1><ul>{''.join(links) or '<li>No data</li>'}</ul>"), encoding='utf-8')
        result[f'{section}_index'] = str(sec_dir / 'index.html')
    # Learning report
    learn_dir = out / 'learning'; learn_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out / 'data' / 'learning'; data_dir.mkdir(parents=True, exist_ok=True)
    learning_payload = {}
    for lvl, df in learning.items():
        learning_payload[lvl] = df.to_dict(orient='list') if not df.empty else {}
    (data_dir / 'learning_metrics.json').write_text(json.dumps(learning_payload, indent=2, ensure_ascii=False), encoding='utf-8')
    ld = learning.get('daily', pd.DataFrame())
    labels = ld.get('period', pd.Series([], dtype=str)).astype(str).tolist() if not ld.empty else []
    lp = {'labels': labels}
    for col in ['reward_mean','mpc_cost_mean','comfort_violation_rate','energy_kwh_normalized','switching_penalty_mean','conflict_penalty_mean','rl_q_delta_mean','prediction_error_mean','learning_elapsed_sec']:
        lp[col] = pd.to_numeric(ld.get(col, pd.Series([], dtype=float)), errors='coerce').fillna(0.0).round(6).tolist() if not ld.empty else []
    (data_dir / 'learning_curve.json').write_text(json.dumps(lp, indent=2, ensure_ascii=False), encoding='utf-8')
    body = "<h1>Learning progress</h1><div class='grid'>" + _summary_cards_html(ld.iloc[-1].to_dict() if not ld.empty else {}) + "</div><canvas id='reward'></canvas><canvas id='cost'></canvas><canvas id='viol'></canvas><canvas id='energy'></canvas><canvas id='qdelta'></canvas><canvas id='elapsed'></canvas>"
    script = """
lineChart('reward', data.labels, [ds('Reward mean',data.reward_mean,0)], 'Reward');
lineChart('cost', data.labels, [ds('MPC cost mean',data.mpc_cost_mean,1)], 'MPC cost');
lineChart('viol', data.labels, [ds('Comfort violation rate',data.comfort_violation_rate,2), ds('Prediction error',data.prediction_error_mean,3)], 'Learning quality');
lineChart('energy', data.labels, [ds('Energy normalized',data.energy_kwh_normalized,4), ds('Switch penalty',data.switching_penalty_mean,5), ds('Conflict penalty',data.conflict_penalty_mean,6)], 'Energy / actuator penalties');
lineChart('qdelta', data.labels, [ds('RL Q delta',data.rl_q_delta_mean,7)], 'Q-memory change');
lineChart('elapsed', data.labels, [ds('Elapsed seconds',data.learning_elapsed_sec,8)], 'Elapsed learning time');
"""
    (learn_dir / 'index.html').write_text(_html_shell('Learning progress', body, lp, script), encoding='utf-8')
    result['learning_index'] = str(learn_dir / 'index.html')
    # Advanced 3D / decision-flow reports. Built before the root index so it can be linked from index.html.
    advanced = build_advanced_3d_reports(out, ss_kstore_dir, growth_df=growth_df, summary=summary)
    result.update(advanced)
    radar_outputs = build_radar_reports(out, growth_df=growth_df, ss_kstore_dir=ss_kstore_dir)
    result.update(radar_outputs)

    # Root index
    cards = ''
    if summary:
        cards = '<div class="grid">' + _summary_cards_html(summary) + '</div>'
    root_links = ''.join([f"<li><a href='{s}/index.html'>{s.title()} reports</a></li>" for s in sections]) + "<li><a href='learning/index.html'>Learning progress</a></li><li><a href='accuracy/index.html'>Accuracy report</a></li><li><a href='advanced_3d/index.html'>Advanced 3D Surface and radar reports</a></li><li><a href='validation_report.html'>Validation summary</a></li>"
    (out / 'index.html').write_text(_html_shell('TDDT training report index', f"<h1>TDDT training reports</h1>{cards}<h2>Sections</h2><ul>{root_links}</ul>"), encoding='utf-8')
    result['index'] = str(out / 'index.html')
    return result


def build_reports(report_dir: str | Path, inner_df: pd.DataFrame, commands_df: pd.DataFrame, growth_df: pd.DataFrame | None = None, config=None, ss_kstore_dir=None, mode_name: str = "TRAIN") -> dict:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    inner_path = out / "inner_loop_log.csv"
    cmd_path = out / "actuator_commands.csv"
    growth_path = out / "outer_growth_state.csv"
    summary_path = out / "validation_summary.json"
    chart_data_path = out / "validation_chart_data.json"
    html_path = out / "validation_report.html"

    inner_df.to_csv(inner_path, index=False)
    commands_df.to_csv(cmd_path, index=False)
    if growth_df is not None and not growth_df.empty:
        growth_df.to_csv(growth_path, index=False)

    summary = _compute_metrics(inner_df, commands_df, growth_df, config=config)
    summary["mode_name"] = mode_name
    if "learning_elapsed_sec" in inner_df.columns and not inner_df.empty:
        summary["learning_elapsed_sec"] = float(pd.to_numeric(inner_df["learning_elapsed_sec"], errors="coerce").max())
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    chart_data = _chart_payload(inner_df, growth_df)
    chart_data_path.write_text(json.dumps(chart_data, indent=2, ensure_ascii=False), encoding="utf-8")

    inner_tail_html = inner_df.tail(30).to_html(index=False) if not inner_df.empty else "<p>No inner-loop rows.</p>"
    cmd_tail_html = commands_df.tail(30).to_html(index=False) if not commands_df.empty else "<p>No actuator command rows.</p>"
    growth_tail_html = growth_df.tail(30).to_html(index=False) if growth_df is not None and not growth_df.empty else "<p>No growth rows.</p>"
    html_path.write_text(_html_report(summary, chart_data, inner_tail_html, cmd_tail_html, growth_tail_html), encoding="utf-8")

    hierarchical = build_hierarchical_training_reports(out, ss_kstore_dir, growth_df, summary)
    accuracy_payload = _accuracy_report_payload(inner_df, growth_df, config=config, ss_kstore_dir=ss_kstore_dir, summary=summary)
    accuracy_outputs = _write_accuracy_report(out, accuracy_payload)
    ccll_outputs = {}
    if config is not None:
        ccll_outputs = build_ccll_cluster_report(out, getattr(config, "ccll_library_json", None), getattr(config, "ccll_daily_descriptors_csv", None))
        _ensure_index_link(out / "index.html", "ccll_clusters/index.html", "CCLL nearest-centroid cluster report")

    return {
        "report_index": hierarchical.get("index", ""),
        "learning_report": hierarchical.get("learning_index", ""),
        "inner_loop_log": str(inner_path),
        "actuator_commands": str(cmd_path),
        "outer_growth_state": str(growth_path) if growth_path.exists() else "",
        "summary": str(summary_path),
        "chart_data": str(chart_data_path),
        "html": str(html_path),
        **accuracy_outputs,
        **ccll_outputs,
        **hierarchical,
    }
