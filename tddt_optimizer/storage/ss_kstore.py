from __future__ import annotations
from pathlib import Path
from datetime import datetime
import csv, gzip, json, math, shutil
from collections import defaultdict, Counter
from typing import Any
import pandas as pd

INNER_COLUMNS = [
    "timestamp","mode","growth_day","climate_context_id","climate_context_name",
    "outdoor_temp_c","outdoor_rh_pct","outdoor_wind_m_s","outdoor_solar_w_m2","outdoor_cloud_okta","outdoor_rain_mm_day",
    "indoor_temp_c","indoor_rh_pct","indoor_air_speed_m_s","lct_c","uct_c",
    "ventilation_group_pct","heating_group_pct","light_on","electric_kw","gas_kw","reward",
    "mpc_score","comfort_error","energy_cost","gas_penalty","context_penalty","guidance_penalty","rl_prior_cost",
    "state_estimation_temp_error_c","state_estimation_uncertainty_radius","bayesian_profile",
    "switch_penalty","oscillation_penalty","heating_ventilation_conflict_penalty","robust_margin_penalty",
    "rl_q_delta","rl_td_error","learning_elapsed_sec",
    "rl_state_key","rl_action_key","safety_status"
]
RL_COLUMNS = ["timestamp","state_key","action_key","reward","old_q","new_q","td_error","layer"]

NUMERIC_SUMMARY_COLS = [
    "outdoor_temp_c","outdoor_rh_pct","outdoor_wind_m_s","outdoor_solar_w_m2","outdoor_cloud_okta","outdoor_rain_mm_day",
    "indoor_temp_c","indoor_rh_pct","indoor_air_speed_m_s","lct_c","uct_c",
    "ventilation_group_pct","heating_group_pct","light_on","electric_kw","gas_kw","mpc_score","comfort_error",
    "energy_cost","switch_penalty","oscillation_penalty","heating_ventilation_conflict_penalty",
    "rl_q_delta","rl_td_error","learning_elapsed_sec",
    "state_estimation_temp_error_c","state_estimation_uncertainty_radius"
]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def _jsonable(v: Any) -> Any:
    try:
        if hasattr(v, "isoformat"):
            return v.isoformat()
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return v


class _OnlineAgg:
    def __init__(self):
        self.n = 0
        self.sum = defaultdict(float)
        self.min = {}
        self.max = {}
        self.context_counts = Counter()
        self.first_ts = None
        self.last_ts = None
        self.command_changes = 0
        self.abrupt_changes = 0
        self.osc_reversals = 0
        self.prev_vent = None
        self.prev_heat = None
        self.prev_light = None
        self.prev_dvent = 0.0
        self.prev_dheat = 0.0

    def update(self, row: dict, max_delta: float = 50.0):
        self.n += 1
        ts = str(row.get("timestamp", ""))
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts
        cid = row.get("climate_context_id")
        if cid:
            self.context_counts[str(cid)] += 1
        for c in NUMERIC_SUMMARY_COLS:
            x = _safe_float(row.get(c), 0.0)
            self.sum[c] += x
            self.min[c] = x if c not in self.min else min(self.min[c], x)
            self.max[c] = x if c not in self.max else max(self.max[c], x)
        vent = _safe_float(row.get("ventilation_group_pct"), 0.0)
        heat = _safe_float(row.get("heating_group_pct"), 0.0)
        light = _safe_float(row.get("light_on"), 0.0)
        if self.prev_vent is not None:
            dvent = vent - self.prev_vent
            dheat = heat - self.prev_heat
            changed = abs(dvent) > 0 or abs(dheat) > 0 or abs(light - self.prev_light) > 0
            self.command_changes += int(changed)
            self.abrupt_changes += int(abs(dvent) > max_delta or abs(dheat) > max_delta)
            self.osc_reversals += int((dvent * self.prev_dvent < 0) or (dheat * self.prev_dheat < 0))
            self.prev_dvent, self.prev_dheat = dvent, dheat
        self.prev_vent, self.prev_heat, self.prev_light = vent, heat, light

    def to_row(self, key: str) -> dict:
        row = {"period": key, "steps": self.n, "first_timestamp": self.first_ts, "last_timestamp": self.last_ts}
        for c in NUMERIC_SUMMARY_COLS:
            row[f"{c}_mean"] = self.sum[c] / max(self.n, 1)
            row[f"{c}_min"] = self.min.get(c)
            row[f"{c}_max"] = self.max.get(c)
        row["dominant_climate_context_id"] = self.context_counts.most_common(1)[0][0] if self.context_counts else ""
        row["context_counts_json"] = json.dumps(dict(self.context_counts), ensure_ascii=False)
        row["actuator_command_changes"] = self.command_changes
        row["abrupt_actuator_changes"] = self.abrupt_changes
        row["oscillatory_actuator_reversals"] = self.osc_reversals
        return row



class _LearningAgg:
    """Constant-memory learning-progress accumulator for TRAIN reports."""
    def __init__(self):
        self.n = 0
        self.sum = defaultdict(float)
        self.min_elapsed = None
        self.max_elapsed = None
        self.first_ts = None
        self.last_ts = None
        self.context_counts = Counter()

    def update(self, row: dict):
        self.n += 1
        ts = str(row.get("timestamp", ""))
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts
        cid = row.get("climate_context_id")
        if cid:
            self.context_counts[str(cid)] += 1
        low = _safe_float(row.get("lct_c"), -999.0)
        high = _safe_float(row.get("uct_c"), 999.0)
        temp = _safe_float(row.get("indoor_temp_c"), 0.0)
        comfort_violation = 1.0 if (temp < low or temp > high) else 0.0
        elapsed = _safe_float(row.get("learning_elapsed_sec"), 0.0)
        if self.min_elapsed is None:
            self.min_elapsed = elapsed
        self.max_elapsed = elapsed
        fields = {
            "reward_mean": _safe_float(row.get("reward"), 0.0),
            "mpc_cost_mean": _safe_float(row.get("mpc_score"), 0.0),
            "comfort_error_mean": _safe_float(row.get("comfort_error"), 0.0),
            "comfort_violation_rate": comfort_violation,
            "energy_kwh_normalized": (_safe_float(row.get("electric_kw"), 0.0) + _safe_float(row.get("gas_kw"), 0.0)) / 12.0,
            "switching_penalty_mean": _safe_float(row.get("switch_penalty"), 0.0),
            "oscillation_penalty_mean": _safe_float(row.get("oscillation_penalty"), 0.0),
            "conflict_penalty_mean": _safe_float(row.get("heating_ventilation_conflict_penalty"), 0.0),
            "rl_q_delta_mean": _safe_float(row.get("rl_q_delta"), 0.0),
            "rl_td_error_mean": abs(_safe_float(row.get("rl_td_error"), 0.0)),
            "prediction_error_mean": abs(_safe_float(row.get("state_estimation_temp_error_c"), 0.0)),
            "uncertainty_radius_mean": _safe_float(row.get("state_estimation_uncertainty_radius"), 0.0),
            "learning_elapsed_sec": elapsed,
        }
        for k, v in fields.items():
            self.sum[k] += v

    def to_row(self, key: str) -> dict:
        row = {"period": key, "steps": self.n, "first_timestamp": self.first_ts, "last_timestamp": self.last_ts}
        for k, v in self.sum.items():
            if k == "energy_kwh_normalized":
                row[k] = v
            elif k == "learning_elapsed_sec":
                row[k] = self.max_elapsed or 0.0
            else:
                row[k] = v / max(self.n, 1)
        row["dominant_climate_context_id"] = self.context_counts.most_common(1)[0][0] if self.context_counts else ""
        row["context_counts_json"] = json.dumps(dict(self.context_counts), ensure_ascii=False)
        return row


def _period_keys(ts) -> dict:
    if pd.isna(ts):
        return {}
    return {
        "daily": ts.strftime("%Y-%m-%d"),
        "weekly": f"{ts.isocalendar().year}-W{int(ts.isocalendar().week):02d}",
        "monthly": ts.strftime("%Y-%m"),
        "quarterly": f"{ts.year}-Q{((ts.month - 1)//3) + 1}",
        "yearly": ts.strftime("%Y"),
    }

class SegmentedStreamingKnowledgeStore:
    """Append-only segmented store for TRAIN traces.

    The control loop writes small in-memory buffers to gzip CSV segments. Online
    hourly/daily aggregators and a bounded down-sampled trace are kept for reports.
    SQLite is intentionally not used for the high-frequency 5-minute trace.
    """
    def __init__(self, root: str | Path, *, segment_size: int = 1000, sample_every: int = 50, max_delta_pct: float = 50.0):
        self.root = Path(root)
        self.inner_dir = self.root / "segments" / "inner_loop"
        self.rl_dir = self.root / "segments" / "rl_updates"
        self.summary_dir = self.root / "summaries"
        for p in [self.inner_dir, self.rl_dir, self.summary_dir]:
            p.mkdir(parents=True, exist_ok=True)
        self.segment_size = int(segment_size)
        self.sample_every = max(1, int(sample_every))
        self.max_delta_pct = float(max_delta_pct)
        self.buffer: list[dict] = []
        self.rl_buffer: list[dict] = []
        self.segment_idx = 0
        self.rl_segment_idx = 0
        self.row_count = 0
        self.sample_rows: list[dict] = []
        self.tail_rows: list[dict] = []
        self.daily = {}
        self.hourly = {}
        self.learning = {"daily": {}, "weekly": {}, "monthly": {}, "quarterly": {}, "yearly": {}}
        self.started_at = datetime.utcnow().isoformat() + "Z"

    def _minimal_row(self, row: dict) -> dict:
        out = {c: _jsonable(row.get(c)) for c in INNER_COLUMNS}
        comps = row.get("mpc_components") or {}
        for name in [
            "comfort_error","energy_cost","gas_penalty","context_penalty","guidance_penalty",
            "switch_penalty","oscillation_penalty","heating_ventilation_conflict_penalty",
            "robust_margin_penalty","model_uncertainty_radius",
            "state_estimation_temp_error_c","state_estimation_uncertainty_radius"
        ]:
            if out.get(name) is None and isinstance(comps, dict):
                out[name] = comps.get(name)
        return out

    def append_inner(self, row: dict):
        r = self._minimal_row(row)
        self.row_count += 1
        self.buffer.append(r)
        if self.row_count == 1 or self.row_count % self.sample_every == 0:
            self.sample_rows.append(r.copy())
        self.tail_rows.append(r.copy())
        if len(self.tail_rows) > 500:
            self.tail_rows = self.tail_rows[-500:]
        ts = pd.to_datetime(r.get("timestamp"), errors="coerce")
        if not pd.isna(ts):
            dkey = ts.strftime("%Y-%m-%d")
            hkey = ts.strftime("%Y-%m-%d %H:00")
            self.daily.setdefault(dkey, _OnlineAgg()).update(r, self.max_delta_pct)
            self.hourly.setdefault(hkey, _OnlineAgg()).update(r, self.max_delta_pct)
            for level, key in _period_keys(ts).items():
                self.learning[level].setdefault(key, _LearningAgg()).update(r)
        if len(self.buffer) >= self.segment_size:
            self.flush_inner()

    def append_rl_update(self, update: dict):
        d = update if isinstance(update, dict) else getattr(update, "__dict__", {})
        row = {c: _jsonable(d.get(c)) for c in RL_COLUMNS}
        row["timestamp"] = row.get("timestamp") or datetime.utcnow().isoformat() + "Z"
        row["layer"] = row.get("layer") or "climate_mpc_bias"
        self.rl_buffer.append(row)
        if len(self.rl_buffer) >= self.segment_size:
            self.flush_rl()

    def _write_gzip_csv(self, path: Path, rows: list[dict], columns: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    def flush_inner(self):
        if not self.buffer:
            return
        self.segment_idx += 1
        path = self.inner_dir / f"part-{self.segment_idx:06d}.csv.gz"
        self._write_gzip_csv(path, self.buffer, INNER_COLUMNS)
        self.buffer.clear()

    def flush_rl(self):
        if not self.rl_buffer:
            return
        self.rl_segment_idx += 1
        path = self.rl_dir / f"part-{self.rl_segment_idx:06d}.csv.gz"
        self._write_gzip_csv(path, self.rl_buffer, RL_COLUMNS)
        self.rl_buffer.clear()

    def _write_summary(self, name: str, rows: list[dict]):
        path = self.summary_dir / name
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        cols = sorted(set().union(*(r.keys() for r in rows)))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    def close(self):
        self.flush_inner(); self.flush_rl()
        daily_rows = [agg.to_row(k) for k, agg in sorted(self.daily.items())]
        hourly_rows = [agg.to_row(k) for k, agg in sorted(self.hourly.items())]
        self._write_summary("daily_summary.csv", daily_rows)
        self._write_summary("hourly_summary.csv", hourly_rows)
        for level, aggs in self.learning.items():
            rows = [agg.to_row(k) for k, agg in sorted(aggs.items())]
            self._write_summary(f"learning_{level}.csv", rows)
        self._write_summary("sampled_trace.csv", self.sample_rows)
        self._write_summary("tail_trace.csv", self.tail_rows)
        manifest = {
            "storage_name": "SS-KStore",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "started_at": self.started_at,
            "episode_trace_rows": self.row_count,
            "segment_size": self.segment_size,
            "sample_every": self.sample_every,
            "inner_segments": self.segment_idx,
            "rl_segments": self.rl_segment_idx,
            "paths": {
                "inner_segments": str(self.inner_dir),
                "rl_segments": str(self.rl_dir),
                "daily_summary": str(self.summary_dir / "daily_summary.csv"),
                "hourly_summary": str(self.summary_dir / "hourly_summary.csv"),
                "sampled_trace": str(self.summary_dir / "sampled_trace.csv"),
                "tail_trace": str(self.summary_dir / "tail_trace.csv"),
                "learning_daily": str(self.summary_dir / "learning_daily.csv"),
                "learning_weekly": str(self.summary_dir / "learning_weekly.csv"),
                "learning_monthly": str(self.summary_dir / "learning_monthly.csv"),
                "learning_quarterly": str(self.summary_dir / "learning_quarterly.csv"),
                "learning_yearly": str(self.summary_dir / "learning_yearly.csv"),
            },
        }
        (self.root / "ss_kstore_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest

    def sampled_dataframe(self) -> pd.DataFrame:
        p = self.summary_dir / "sampled_trace.csv"
        if p.exists() and p.stat().st_size > 0:
            return pd.read_csv(p, parse_dates=["timestamp"])
        return pd.DataFrame(self.sample_rows)

    def daily_summary_dataframe(self) -> pd.DataFrame:
        p = self.summary_dir / "daily_summary.csv"
        if p.exists() and p.stat().st_size > 0:
            return pd.read_csv(p)
        return pd.DataFrame([agg.to_row(k) for k, agg in sorted(self.daily.items())])

    def hourly_summary_dataframe(self) -> pd.DataFrame:
        p = self.summary_dir / "hourly_summary.csv"
        if p.exists() and p.stat().st_size > 0:
            return pd.read_csv(p)
        return pd.DataFrame([agg.to_row(k) for k, agg in sorted(self.hourly.items())])

    def growth_daily_dataframe(self, growth_start_date: str) -> pd.DataFrame:
        daily = self.daily_summary_dataframe()
        if daily.empty:
            return pd.DataFrame()
        start = pd.to_datetime(growth_start_date).date()
        dates = pd.to_datetime(daily["period"]).dt.date
        out = pd.DataFrame({
            "date": pd.to_datetime(daily["period"]),
            "mint": pd.to_numeric(daily.get("indoor_temp_c_min", daily.get("outdoor_temp_c_min", 0)), errors="coerce").fillna(0.0),
            "maxt": pd.to_numeric(daily.get("indoor_temp_c_max", daily.get("outdoor_temp_c_max", 0)), errors="coerce").fillna(0.0),
            "wind": pd.to_numeric(daily.get("indoor_air_speed_m_s_mean", daily.get("outdoor_wind_m_s_mean", 0.25)), errors="coerce").fillna(0.25),
            "rain": pd.to_numeric(daily.get("outdoor_rain_mm_day_mean", 0.0), errors="coerce").fillna(0.0),
            "okta": pd.to_numeric(daily.get("outdoor_cloud_okta_mean", 8.0), errors="coerce").fillna(8.0).round().clip(0, 8),
            "rad": pd.to_numeric(daily.get("outdoor_solar_w_m2_mean", 0.0), errors="coerce").fillna(0.0) * 86.4,
        })
        out["fattening_day"] = [(d - start).days + 1 for d in dates]
        out = out[out["fattening_day"] >= 1].copy()
        mean_temp = (out["mint"] + out["maxt"]) / 2.0
        out["vpr"] = 0.6108 * (2.718281828 ** ((17.27 * mean_temp) / (mean_temp + 237.3))) * 0.65
        out["aha"] = 1.30
        out["doy"] = out["date"].dt.dayofyear.astype(int)
        out["yr"] = out["date"].dt.year.astype(int)
        out["is_observed"] = 1
        out["source_day"] = out["fattening_day"]
        out["season_start_date"] = str(start)
        cols = ["fattening_day", "yr", "doy", "rad", "mint", "maxt", "vpr", "wind", "rain", "aha", "okta", "is_observed", "source_day", "season_start_date"]
        return out[cols].reset_index(drop=True)

    def daily_context_for_day(self, fattening_day: int, contexts: dict) -> dict:
        daily = self.daily_summary_dataframe()
        if daily.empty:
            return {}
        idx = max(0, min(int(fattening_day) - 1, len(daily) - 1))
        cid = str(daily.iloc[idx].get("dominant_climate_context_id", ""))
        return contexts.get(cid, {}) if cid else {}


def _copy_if_exists(src: Path, dst: Path):
    """Copy a compact artifact only when source and destination differ.

    Some export paths intentionally point to models/setd_kstore itself. In that
    case, copying edge policy snapshots from models/setd_kstore back into the
    same directory raises shutil.SameFileError and aborts TRAIN after the inner
    loop has already completed. This guard keeps export idempotent and safe.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src.resolve() == dst.resolve():
            return
    except FileNotFoundError:
        # dst may not exist yet; normal copy should proceed.
        pass
    shutil.copy2(src, dst)


def export_ss_setd_kstore(sqlite_path: str | Path, output_dir: str | Path, ss_root: str | Path, *, ccll_path: str | Path | None = None, sarg_path: str | Path | None = None, tddt_version: str = "unknown") -> dict:
    import sqlite3
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    ss_root = Path(ss_root)
    summary_dir = ss_root / "summaries"
    # Keep the bulky trace as external segments; copy only compact report inputs.
    for name in ["daily_summary.csv", "hourly_summary.csv", "sampled_trace.csv", "tail_trace.csv",
                 "learning_daily.csv", "learning_weekly.csv", "learning_monthly.csv", "learning_quarterly.csv", "learning_yearly.csv"]:
        _copy_if_exists(summary_dir / name, out / name)
    # Small tables from SQLite are safe to materialize.
    conn = sqlite3.connect(sqlite_path)
    def table(name: str) -> pd.DataFrame:
        try:
            return pd.read_sql_query(f"SELECT * FROM {name}", conn)
        except Exception:
            return pd.DataFrame()
    growth = table("outer_growth_state")
    guidance = table("growth_climate_guidance")
    sarg_ledger = table("sarg_objective_score_ledger")
    formal_links = table("setd_formal_links")
    for df, name in [(growth,"daily_growth_state_memory.csv"),(guidance,"transdomain_guidance_memory.csv"),(sarg_ledger,"objective_score_ledger.csv"),(formal_links,"formal_relation_ledger.csv")]:
        df.to_csv(out / name, index=False)
    try:
        ccll = json.loads(Path(ccll_path).read_text(encoding="utf-8")) if ccll_path and Path(ccll_path).exists() else {}
    except Exception:
        ccll = {}
    all_contexts = sorted([str(c.get("context_id")) for c in ccll.get("contexts", []) if c.get("context_id")]) if isinstance(ccll, dict) else []
    daily = pd.read_csv(summary_dir / "daily_summary.csv") if (summary_dir / "daily_summary.csv").exists() else pd.DataFrame()
    seen_contexts = sorted(daily.get("dominant_climate_context_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) if not daily.empty else []
    rules = []
    if not guidance.empty:
        for _, row in guidance.tail(5000).iterrows():
            rules.append({
                "biological_phase": row.get("biological_phase"),
                "climate_context_id": row.get("climate_context_id"),
                "best_reference_id": row.get("best_reference_id"),
                "preferred_temp_low_c": row.get("preferred_temp_low_c"),
                "preferred_temp_high_c": row.get("preferred_temp_high_c"),
                "comfort_weight_bias": row.get("comfort_weight_bias"),
                "energy_weight_bias": row.get("energy_weight_bias"),
                "ventilation_bias": row.get("ventilation_bias"),
                "heating_bias": row.get("heating_bias"),
                "diet_phase_guidance": row.get("diet_phase_guidance"),
                "confidence": row.get("confidence"),
            })
    snapshot = {
        "policy_type": "single_episode_transdomain_policy_snapshot",
        "storage_method": "SS-KStore",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "learning_mode": "segmented_streaming_single_episode",
        "segment_store_path": str(ss_root),
        "best_stage_context_guidance_rules": rules,
        "fallback_safety_policy": {"if_context_unseen": "nearest_context_low_confidence", "if_state_unseen": "phase_level_policy", "if_both_unseen": "safe_mpc"},
    }
    (out / "learned_policy_snapshot.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "stage_context_policy.json").write_text(json.dumps({"policy_type":"stage_context_guidance_policy","rules":rules}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    manifest = json.loads((ss_root / "ss_kstore_manifest.json").read_text(encoding="utf-8")) if (ss_root / "ss_kstore_manifest.json").exists() else {}
    coverage = {
        "coverage": {
            "storage_method": "SS-KStore",
            "episode_trace_rows": int(manifest.get("episode_trace_rows", 0)),
            "inner_trace_storage": str(ss_root / "segments" / "inner_loop"),
            "growth_rows": int(len(growth)),
            "guidance_rows": int(len(guidance)),
            "sarg_score_rows": int(len(sarg_ledger)),
            "climate_contexts_seen_in_episode": seen_contexts,
            "climate_contexts_from_library": all_contexts,
            "unseen_contexts": sorted(list(set(all_contexts) - set(seen_contexts))),
        },
        "uncertainty_policy": snapshot["fallback_safety_policy"],
    }
    (out / "coverage_uncertainty_report.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    # Copy compact online learning states if available. These make the formal
    # layers executable in WORK-ONLINE without reading raw segments.
    root = Path(__file__).resolve().parents[2]
    for name in ["edge_multilayer_rl_policy.json", "edge_contextual_bayesian_tuner.json", "edge_state_estimator.json"]:
        _copy_if_exists(root / "models" / "setd_kstore" / name, out / name)

    final_manifest = {
        "storage_name": "SETD-KStore backed by SS-KStore segments",
        "tddt_version": tddt_version,
        "created_at": snapshot["created_at"],
        "learning_mode": "single_episode_segmented_streaming",
        "raw_trace_policy": "append_only_segments_not_loaded_into_RAM",
        "files": {
            "ss_kstore_manifest": str(ss_root / "ss_kstore_manifest.json"),
            "inner_segments": str(ss_root / "segments" / "inner_loop"),
            "daily_summary": str(out / "daily_summary.csv"),
            "hourly_summary": str(out / "hourly_summary.csv"),
            "sampled_trace": str(out / "sampled_trace.csv"),
            "daily_growth_state_memory": str(out / "daily_growth_state_memory.csv"),
            "transdomain_guidance_memory": str(out / "transdomain_guidance_memory.csv"),
            "objective_score_ledger": str(out / "objective_score_ledger.csv"),
            "formal_relation_ledger": str(out / "formal_relation_ledger.csv"),
            "learned_policy_snapshot": str(out / "learned_policy_snapshot.json"),
            "coverage_uncertainty_report": str(out / "coverage_uncertainty_report.json"),
            "edge_rl_policy": str(out / "edge_multilayer_rl_policy.json"),
            "edge_bayesian_tuner": str(out / "edge_contextual_bayesian_tuner.json"),
            "edge_state_estimator": str(out / "edge_state_estimator.json"),
        },
    }
    (out / "manifest.json").write_text(json.dumps(final_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS learned_model_registry (model_id TEXT PRIMARY KEY, created_at TEXT, tddt_version TEXT, storage_method TEXT, policy_snapshot_path TEXT, manifest_path TEXT)")
        conn.execute("INSERT OR REPLACE INTO learned_model_registry VALUES (?,?,?,?,?,?)", ("setd_kstore_latest", snapshot["created_at"], tddt_version, "SS-KStore", str(out / "learned_policy_snapshot.json"), str(out / "manifest.json")))
        conn.commit()
    except Exception:
        pass
    conn.close()
    return final_manifest
