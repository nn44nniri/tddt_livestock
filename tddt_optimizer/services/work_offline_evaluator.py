from __future__ import annotations
from pathlib import Path
import json
import time
import pandas as pd
from ..adapters.climate_adapter import ClimateAdapter
from ..database.async_sqlite_store import AsyncSQLiteStore
from ..database.sqlite_store import SQLiteStore
from ..evaluation.reports import build_reports
from ..optimizer.economic_mpc import EconomicMPC, MPCWeights
from ..optimizer.safety_filter import apply_safety


class WorkOfflineEvaluator:
    """Offline evaluation/replay mode that does not use the funnel.

    This mode is intentionally not a training loop. It reads prepared climate rows,
    loads the SETD-KStore learned policy snapshot when available, applies the best
    learned guidance/actuator priors, writes every step to SQLite through the async
    writer, and builds reports from the database at the end.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.project_root = Path(cfg.project_root)
        self.climate = ClimateAdapter(self.project_root, tmp_dir=getattr(cfg, "tmp_dir", "/tmp/tddt_livestock/"), cleanup_interval_steps=getattr(cfg, "tmp_cleanup_interval_steps", 4000))
        self.snapshot = self._load_json(Path(cfg.setd_kstore_dir) / "learned_policy_snapshot.json")
        self.ccll = self._load_json(cfg.ccll_library_json)
        self.contexts = {c.get("context_id"): c for c in self.ccll.get("contexts", [])} if isinstance(self.ccll, dict) else {}
        self.mpc = EconomicMPC(
            self.climate, cfg.candidate_ventilation, cfg.candidate_heating,
            MPCWeights(cfg.comfort_weight, cfg.energy_weight, cfg.gas_weight, cfg.mpc_context_weight, cfg.mpc_guidance_weight),
            rl_agent=None,
        )

    def run(self, growth_start_date: str, end_date: str | None = None, max_steps: int | None = None, case_id: int = 1, show_progress: bool = True, light_on_hour: int = 5, light_on_minute: int = 0, light_hours_on: float = 16.0, light_hours_off: float | None = 8.0) -> dict:
        self.mpc.configure_light_schedule(light_on_hour, light_on_minute, light_hours_on, light_hours_off)
        path = Path(self.cfg.prepared_ccll_5m_csv) if Path(self.cfg.prepared_ccll_5m_csv).exists() else Path(self.cfg.prepared_5m_csv)
        df5 = pd.read_csv(path, parse_dates=["timestamp"])
        start = pd.to_datetime(growth_start_date)
        df5 = df5[df5["timestamp"] >= start].reset_index(drop=True)
        selected_end = None
        if end_date:
            selected_end = self._parse_end_datetime(end_date)
            df5 = df5[df5["timestamp"] <= selected_end].reset_index(drop=True)
        if max_steps:
            df5 = df5.head(max_steps).copy()
        if df5.empty:
            raise ValueError(f"No prepared rows found for WORK-OFFLINE from {growth_start_date}")
        total = len(df5)
        if show_progress:
            print("[TDDT WORK-OFFLINE] Starting learned-policy offline evaluation", flush=True)
            print(f"  source_dataset    : {path}", flush=True)
            print(f"  policy_snapshot   : {Path(self.cfg.setd_kstore_dir) / 'learned_policy_snapshot.json'}", flush=True)
            print(f"  steps             : {total}", flush=True)
            print("  funnel            : disabled", flush=True)
        store = AsyncSQLiteStore(self.cfg.sqlite_path)
        started = time.time()
        for idx, (_, row) in enumerate(df5.iterrows(), start=1):
            state = row.to_dict()
            ctx = self.contexts.get(str(state.get("context_id") or state.get("climate_context_id")), {})
            if ctx:
                state["climate_context_id"] = ctx.get("context_id")
                state["climate_context_name"] = ctx.get("context_name")
            guidance = self._guidance_for_context(str(state.get("climate_context_id") or ""), ctx)
            command = self._learned_command(str(state.get("climate_context_id") or ""))
            if command is None:
                command, pred = self.mpc.choose(state, guidance)
            else:
                pred = {"mpc_score": None, "mpc_components": {"source": "learned_policy_snapshot"}}
            command, safety = apply_safety(command)
            result = self.climate.simulate_step(state, command, write_report=False)
            result.update({"mpc_score": pred.get("mpc_score"), "mpc_components": pred.get("mpc_components", {})})
            log_row = {**state, **result, **command, "mode": "WORK-OFFLINE", "safety_status": safety}
            store.log_command(str(state["timestamp"]), "WORK-OFFLINE", command, safety)
            store.log_inner(log_row)
            if show_progress and (idx == 1 or idx == total or idx % max(1, total // 50) == 0):
                pct = 100.0 * idx / max(total, 1)
                print(f"[TDDT WORK-OFFLINE] step {idx:>6}/{total:<6} ({pct:6.2f}%) | {state.get('timestamp')} | context={state.get('climate_context_id')} | vent={float(command.get('ventilation_group_pct',0)):5.1f}% heat={float(command.get('heating_group_pct',0)):5.1f}% | safety={safety} | elapsed={time.time()-started:6.1f}s", flush=True)
        store.close()
        sync = SQLiteStore(self.cfg.sqlite_path)
        inner_df = sync.table("inner_sensor_log")
        commands_df = sync.table("actuator_commands")
        growth_df = sync.table("outer_growth_state")
        reports = build_reports(self.cfg.report_dir, inner_df, commands_df, growth_df)
        sync.close()
        return {"mode": "WORK-OFFLINE", "purpose": "learned_policy_evaluation", "steps": int(total), "sqlite": self.cfg.sqlite_path, "reports": reports}

    def _load_json(self, path) -> dict:
        p = Path(path)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _guidance_for_context(self, context_id: str, ctx: dict) -> dict:
        base = {"preferred_temp_low_c": 5.0, "preferred_temp_high_c": 24.0, "comfort_weight_bias": 1.0, "energy_weight_bias": 1.0, "gas_weight_bias": 1.0, "status": "WORK_OFFLINE_EVAL"}
        if ctx:
            base["climate_context_id"] = ctx.get("context_id")
            base["ccll_prior"] = ctx.get("mpc_prior", {})
        for rule in self.snapshot.get("best_stage_context_guidance_rules", []) if isinstance(self.snapshot, dict) else []:
            if str(rule.get("climate_context_id")) == str(context_id):
                rec = rule.get("recommended_guidance") or {}
                base.update({k: v for k, v in rec.items() if v is not None})
                base["best_reference_id"] = rule.get("best_reference_id")
                base["confidence"] = rule.get("confidence")
                break
        return base

    def _learned_command(self, context_id: str) -> dict | None:
        for row in self.snapshot.get("best_actuator_lookup", []) if isinstance(self.snapshot, dict) else []:
            if str(row.get("climate_context_id")) == str(context_id):
                return {"ventilation_group_pct": float(row.get("ventilation_group_pct") or 0.0), "heating_group_pct": float(row.get("heating_group_pct") or 0.0), "light_on": bool(row.get("light_on"))}
        return None

    def _parse_end_datetime(self, value: str) -> pd.Timestamp:
        ts = pd.to_datetime(value)
        if len(str(value).strip()) <= 10:
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return ts
