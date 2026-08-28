from __future__ import annotations
from pathlib import Path
import json
import time
import hashlib
import pandas as pd
from ..adapters.climate_adapter import ClimateAdapter
from ..adapters.growth_adapter import GrowthAdapter
from ..database.sqlite_store import SQLiteStore
from ..storage.ss_kstore import SegmentedStreamingKnowledgeStore, export_ss_setd_kstore
from ..learning.sarg_sel import SARGOuterPolicy, identify_stage
from ..evaluation.reports import build_reports
from funnel.simulator_funnel import SimulatorFunnel
from ..optimizer.economic_mpc import EconomicMPC, MPCWeights
from ..optimizer.safety_filter import apply_safety
from ..rl.edge_multilayer_rl import EdgeMultiLayerRL
from ..rl.contextual_tuner import EdgeContextualBayesianTuner
from ..estimation.state_estimator import EdgeStateEstimator


class OfflineWorker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.project_root = Path(cfg.project_root)
        self.climate = ClimateAdapter(self.project_root, tmp_dir=getattr(cfg, "tmp_dir", "/tmp/tddt_livestock/"), cleanup_interval_steps=getattr(cfg, "tmp_cleanup_interval_steps", 4000))
        self.growth = GrowthAdapter(self.project_root, tmp_dir=getattr(cfg, "tmp_dir", "/tmp/tddt_livestock/"))
        self.store = None
        self.ccll_library = self._load_json(cfg.ccll_library_json)
        self.contexts = {c.get("context_id"): c for c in self.ccll_library.get("contexts", [])} if isinstance(self.ccll_library, dict) else {}
        self.sarg = SARGOuterPolicy(cfg.sarg_reference_library_json, weights={
            "growth": cfg.sarg_growth_weight,
            "feed_efficiency": cfg.sarg_feed_efficiency_weight,
            "feed_cost": cfg.sarg_feed_cost_weight,
            "heat_production": cfg.sarg_heat_production_weight,
            "thermal_stress": cfg.sarg_thermal_stress_weight,
            "climate_energy": cfg.sarg_climate_energy_weight,
            "distance": cfg.sarg_state_distance_weight,
        }) if Path(cfg.sarg_reference_library_json).exists() else None
        self.rl = EdgeMultiLayerRL.from_file(
            cfg.rl_state_path,
            alpha=cfg.rl_alpha, gamma=cfg.rl_gamma, epsilon=cfg.rl_epsilon, rl_weight=cfg.rl_weight,
        ) if getattr(cfg, "rl_enabled", True) else None
        self.bayes = EdgeContextualBayesianTuner.from_file(
            getattr(cfg, "bayesian_tuner_path", Path(cfg.setd_kstore_dir) / "edge_contextual_bayesian_tuner.json"),
            alpha=getattr(cfg, "bayesian_tuner_alpha", 0.05),
            exploration_weight=getattr(cfg, "bayesian_tuner_exploration_weight", 0.15),
            gaussian_alpha=getattr(cfg, "gaussian_tuner_alpha", 0.025),
            gaussian_tau=getattr(cfg, "gaussian_tuner_tau", 7.0),
            safe_blend_max=getattr(cfg, "gaussian_tuner_safe_blend_max", 0.85),
        ) if getattr(cfg, "bayesian_tuner_enabled", True) else None
        self.estimator = EdgeStateEstimator.from_file(
            getattr(cfg, "state_estimator_path", Path(cfg.setd_kstore_dir) / "edge_state_estimator.json"),
            alpha=getattr(cfg, "state_estimator_alpha", 0.025),
            uncertainty_decay=getattr(cfg, "state_estimator_uncertainty_decay", 0.995),
        ) if getattr(cfg, "state_estimator_enabled", True) else None
        self.mpc = EconomicMPC(
            self.climate, cfg.candidate_ventilation, cfg.candidate_heating,
            MPCWeights(
                cfg.comfort_weight, cfg.energy_weight, cfg.gas_weight, cfg.mpc_context_weight, cfg.mpc_guidance_weight,
                getattr(cfg, "mpc_switch_weight", 2.5), getattr(cfg, "mpc_max_delta_pct_per_step", 50.0), getattr(cfg, "mpc_oscillation_weight", 4.0),
                getattr(cfg, "mpc_heating_ventilation_conflict_weight", 25.0),
            ),
            rl_agent=self.rl,
        )

    def run(
        self,
        growth_start_date: str,
        end_date: str | None = None,
        max_steps: int | None = None,
        case_id: int = 1,
        show_progress: bool = True,
        mode_name: str = "TRAIN",
        light_on_hour: int = 5,
        light_on_minute: int = 0,
        light_hours_on: float = 16.0,
        light_hours_off: float | None = 8.0,
    ) -> dict:
        self.mpc.configure_light_schedule(light_on_hour, light_on_minute, light_hours_on, light_hours_off)
        ccll_5m = Path(getattr(self.cfg, "prepared_ccll_5m_csv", self.project_root / "prepared" / "climate_5m_ccll_all_rows.csv"))
        prepared_path = ccll_5m if ccll_5m.exists() else Path(self.cfg.prepared_5m_csv)
        start = pd.to_datetime(growth_start_date)
        selected_end = self._parse_end_datetime(end_date) if end_date else None
        total_steps, first_ts, last_ts = self._scan_prepared_window(prepared_path, start, selected_end, max_steps=max_steps)
        if total_steps <= 0:
            raise ValueError(f"No prepared 5-minute rows found from growth_start_date={growth_start_date}, end_date={end_date}")

        progress_mode = self._progress_mode_from_span(first_ts, last_ts)
        if show_progress:
            self._progress_header(growth_start_date, selected_end, total_steps, case_id, progress_mode, light_on_hour, light_on_minute, light_hours_on, light_hours_off)

        funnel = SimulatorFunnel(self._iter_prepared_rows(prepared_path, start, selected_end, max_steps=max_steps))
        guidance = self._base_guidance()
        ss_store = SegmentedStreamingKnowledgeStore(
            getattr(self.cfg, "ss_kstore_dir", self.project_root / "working" / "ss_kstore"),
            segment_size=getattr(self.cfg, "ss_segment_size", 1000),
            sample_every=getattr(self.cfg, "ss_sample_every", 50),
            max_delta_pct=getattr(self.cfg, "mpc_max_delta_pct_per_step", 50.0),
        )
        started = time.time()
        progress_seen: set[str] = set()
        last_state_key = None
        last_action_key = None
        for step_idx in range(1, total_steps + 1):
            state = funnel.read_sensor_packet()
            # L2 formal layer: update recursive estimator from the previous
            # simulator prediction vs. the newly observed/streamed state, then
            # enrich the current state for MPC/RL. This keeps state estimation
            # online and constant-memory.
            estimator_update = self.estimator.update_from_observation(state) if self.estimator else None
            state = self.estimator.enrich_state(state) if self.estimator else state
            ctx = self._context_for_state(state)
            if ctx:
                state.setdefault("climate_context_id", ctx.get("context_id"))
                state.setdefault("climate_context_name", ctx.get("context_name"))
                guidance["climate_context_id"] = ctx.get("context_id")
                guidance["ccll_prior"] = ctx.get("mpc_prior", {})
                # Climate library can directly bias early MPC even before daily growth is available.
                guidance["comfort_weight_bias"] = max(float(guidance.get("comfort_weight_bias", 1.0)), float(ctx.get("mpc_prior", {}).get("comfort_weight_bias", 0.5)))
                guidance["energy_weight_bias"] = min(float(guidance.get("energy_weight_bias", 1.0)), max(0.25, float(ctx.get("mpc_prior", {}).get("energy_weight_bias", 0.7))))
            if self.estimator:
                est_adj = self.estimator.guidance_adjustment()
                guidance["comfort_weight_bias"] = float(guidance.get("comfort_weight_bias", 1.0)) * float(est_adj.get("comfort_weight_bias", 1.0))
                guidance["ventilation_bias"] = max(float(guidance.get("ventilation_bias", 0.0)), float(est_adj.get("ventilation_bias", 0.0)))
                guidance["heating_bias"] = max(float(guidance.get("heating_bias", 0.0)), float(est_adj.get("heating_bias", 0.0)))
                guidance["model_uncertainty_radius"] = est_adj.get("estimator_uncertainty_radius")
                guidance["estimator_theta"] = est_adj.get("estimator_theta")
            if self.bayes:
                guidance = self.bayes.apply_to_guidance(state, guidance)
            command, pred = self.mpc.choose(state, guidance)
            command, safety = apply_safety(command)
            self.mpc.observe_command(command)
            result = self.climate.simulate_step(state, command, write_report=False)
            if self.estimator:
                self.estimator.remember_prediction(result, command)
            # Preserve selected MPC diagnostics in the stored row.
            components = pred.get("mpc_components", {}) if isinstance(pred, dict) else {}
            if estimator_update is not None:
                components["state_estimation_temp_error_c"] = estimator_update.temp_error_c
                components["state_estimation_uncertainty_radius"] = estimator_update.uncertainty_radius
            result.update({
                "mpc_components": components,
                "rl_prior_cost": pred.get("rl_prior_cost") if isinstance(pred, dict) else None,
                "rl_state_key": pred.get("rl_state_key") if isinstance(pred, dict) else None,
                "rl_action_key": pred.get("rl_action_key") if isinstance(pred, dict) else None,
                "mpc_score": pred.get("mpc_score") if isinstance(pred, dict) else None,
            })
            if self.rl:
                state_key = result.get("rl_state_key") or self.rl.encode_state(state, guidance)
                action_key = result.get("rl_action_key") or self.rl.encode_action(command)
                reward = self.rl.reward_from_components(components)
                update = self.rl.update(state_key, action_key, reward, next_state_key=None)
                result["reward"] = reward
                result["rl_q_delta"] = abs(float(update.new_q) - float(update.old_q))
                result["rl_td_error"] = float(update.td_error)
                ss_store.append_rl_update(update.__dict__)
                if self.bayes:
                    bayes_update = self.bayes.update(reward, guidance.get("bayesian_context_key"), guidance.get("bayesian_profile"))
                    if bayes_update:
                        ss_store.append_rl_update(bayes_update)
                        result["bayesian_profile"] = guidance.get("bayesian_profile")
                last_state_key, last_action_key = state_key, action_key
            funnel.update_from_climate_result(result)
            row = {**state, **result, **command, "mode": mode_name, "safety_status": safety, "mpc_score": result.get("mpc_score"), "learning_elapsed_sec": time.time() - started}
            ss_store.append_inner(row)
            if show_progress:
                self._progress_step(step_idx, total_steps, state, result, command, safety, started, progress_mode, progress_seen)

        if show_progress:
            print(f"\n[TDDT {mode_name}] Inner-loop simulation completed. Running daily growth simulator and building reports...", flush=True)

        ss_manifest = ss_store.close()
        if self.rl:
            self.rl.save(self.cfg.rl_state_path)
        if self.bayes:
            self.bayes.save(getattr(self.cfg, "bayesian_tuner_path", Path(self.cfg.setd_kstore_dir) / "edge_contextual_bayesian_tuner.json"))
        if self.estimator:
            self.estimator.save(getattr(self.cfg, "state_estimator_path", Path(self.cfg.setd_kstore_dir) / "edge_state_estimator.json"))

        # SS-KStore is the primary high-frequency storage. SQLite is now used only
        # for compact formal ledgers and model registry entries, avoiding the
        # 288,000-row inner-loop table and preventing RAM growth at report time.
        sync_store = SQLiteStore(self.cfg.sqlite_path)
        if self.estimator:
            sync_store.log_formal_link("STATE_ESTIMATOR_TO_MPC", "L2_state_estimator", "L4_adaptive_economic_mpc", "theta_{t+1}=theta_t+K_t e_t", self.estimator.to_dict())
        if self.bayes:
            sync_store.log_formal_link("BAYESIAN_TUNER_TO_MPC_WEIGHTS", "slow_contextual_bayesian_rl", "L4_mpc_weight_bias", "posterior_profile->MPC_weight_bias", self.bayes.to_dict())
        if self.rl:
            sync_store.log_formal_link("RL_Q_MEMORY_TO_MPC_PRIOR", "edge_multilayer_rl", "L4_mpc_action_prior", "Q(s,a)->soft_prior_cost", self.rl.to_dict())
        daily_df = ss_store.growth_daily_dataframe(growth_start_date)
        daily_df["season_start_date"] = growth_start_date
        growth_out_dir = Path(self.cfg.report_dir) / "growth_outputs"
        growth_last = self.growth.run_end_of_cycle(
            daily_df, growth_out_dir,
            breed=self.cfg.default_breed, diet=self.cfg.default_diet, scale=self.cfg.default_scale,
            sex_animal=self.cfg.default_sex_animal, housing=self.cfg.default_housing, case_id=case_id,
            keep_output_files=True,
        )
        growth_db_df = self._load_growth_output(growth_out_dir, case_id, pd.DataFrame([growth_last]))
        growth_memory_rows = self._normalize_growth_rows(growth_db_df, daily_df, growth_start_date)
        last_guidance = None
        for row in growth_memory_rows:
            ctx = ss_store.daily_context_for_day(int(row.get("fattening_day", 1)), self.contexts)
            row["daily_climate_context_id"] = ctx.get("context_id") if ctx else None
            sync_store.log_growth(row)
            if self.sarg:
                g, scores = self.sarg.guidance_from_scores(row, ctx)
                g["growth_state_hash"] = self._hash_state(row)
                sync_store.log_sarg_scores(scores)
                sync_store.log_guidance(int(row.get("fattening_day", 1) or 1), g)
                sync_store.log_formal_link("SARG_SCORE_TO_GUIDANCE", g["growth_state_hash"], g.get("best_reference_id", "R"), "Score_r(t)->g_t", g)
                last_guidance = g
        if last_guidance is None:
            last_guidance = self._growth_to_guidance(growth_last)
            sync_store.log_growth(growth_last)
            sync_store.log_guidance(int(growth_last.get("fattening_day", 1) or 1), last_guidance)

        sample_df = ss_store.sampled_dataframe()
        commands_df = sample_df[[c for c in ["timestamp","ventilation_group_pct","heating_group_pct","light_on","safety_status"] if c in sample_df.columns]].copy() if not sample_df.empty else pd.DataFrame()
        growth_db_df = sync_store.table("outer_growth_state")
        growth_full_df = self._load_growth_output(growth_out_dir, case_id, growth_db_df)
        reports = build_reports(self.cfg.report_dir, sample_df, commands_df, growth_full_df, config=self.cfg, ss_kstore_dir=getattr(self.cfg, "ss_kstore_dir", None), mode_name=mode_name)
        kstore = export_ss_setd_kstore(
            self.cfg.sqlite_path, self.cfg.setd_kstore_dir, getattr(self.cfg, "ss_kstore_dir", self.project_root / "working" / "ss_kstore"),
            ccll_path=self.cfg.ccll_library_json, sarg_path=self.cfg.sarg_reference_library_json, tddt_version=self.project_root.name,
        )
        reports["ss_kstore_manifest"] = ss_manifest.get("paths", {}).get("sampled_trace", str(Path(getattr(self.cfg, "ss_kstore_dir", "working/ss_kstore")) / "ss_kstore_manifest.json"))
        reports["setd_kstore_manifest"] = kstore.get("files", {}).get("learned_policy_snapshot", str(Path(self.cfg.setd_kstore_dir) / "manifest.json"))
        if show_progress:
            print(f"[TDDT {mode_name}] Reports created:", flush=True)
            for name, path in reports.items(): print(f"  - {name}: {path}", flush=True)
        sync_store.close()
        return {"steps": int(ss_manifest.get("episode_trace_rows", total_steps)), "growth_last": growth_last, "guidance": last_guidance, "reports": reports, "sqlite": self.cfg.sqlite_path, "ss_kstore": str(getattr(self.cfg, "ss_kstore_dir", "working/ss_kstore")), "knowledge_store": str(self.cfg.setd_kstore_dir), "rl_policy": self.cfg.rl_state_path}

    def _scan_prepared_window(self, prepared_path: Path, start: pd.Timestamp, end: pd.Timestamp | None, max_steps: int | None = None, chunksize: int = 50000) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
        """Count the selected training rows without loading the prepared dataset into RAM.

        This is intentionally streaming-first: TRAIN may cover 288,000+ rows for a
        1000-day episode, so the control loop must not materialize the full 5-minute
        prepared dataset before it starts.
        """
        count = 0
        first_ts = None
        last_ts = None
        for chunk in pd.read_csv(prepared_path, usecols=["timestamp"], parse_dates=["timestamp"], chunksize=chunksize):
            ts = chunk["timestamp"]
            mask = ts >= start
            if end is not None:
                mask &= ts <= end
            selected = ts[mask]
            if selected.empty:
                continue
            if max_steps is not None and count + len(selected) > max_steps:
                selected = selected.iloc[: max_steps - count]
            if selected.empty:
                break
            if first_ts is None:
                first_ts = pd.to_datetime(selected.iloc[0])
            last_ts = pd.to_datetime(selected.iloc[-1])
            count += int(len(selected))
            if max_steps is not None and count >= max_steps:
                break
        return count, first_ts, last_ts

    def _iter_prepared_rows(self, prepared_path: Path, start: pd.Timestamp, end: pd.Timestamp | None, max_steps: int | None = None, chunksize: int = 5000):
        """Yield selected 5-minute rows one by one from CSV chunks.

        The yielded rows feed SimulatorFunnel, while the outputs are written to
        SQLite through AsyncSQLiteStore. No full episode list/DataFrame is retained.
        """
        emitted = 0
        for chunk in pd.read_csv(prepared_path, parse_dates=["timestamp"], chunksize=chunksize):
            mask = chunk["timestamp"] >= start
            if end is not None:
                mask &= chunk["timestamp"] <= end
            selected = chunk.loc[mask]
            if selected.empty:
                continue
            for row in selected.to_dict(orient="records"):
                yield row
                emitted += 1
                if max_steps is not None and emitted >= max_steps:
                    return

    def _progress_mode_from_span(self, first_ts: pd.Timestamp | None, last_ts: pd.Timestamp | None) -> str:
        if first_ts is None or last_ts is None:
            return "step"
        span = pd.to_datetime(last_ts) - pd.to_datetime(first_ts)
        if span > pd.Timedelta(days=7):
            return "day"
        if span > pd.Timedelta(days=1):
            return "hour"
        return "step"

    def _load_json(self, path: str | Path) -> dict:
        p = Path(path)
        if p.exists():
            try: return json.loads(p.read_text(encoding="utf-8"))
            except Exception: return {}
        return {}

    def _base_guidance(self) -> dict:
        return {"preferred_temp_low_c": 5.0, "preferred_temp_high_c": 24.0, "comfort_weight_bias": 1.0, "gas_weight_bias": 1.0, "energy_weight_bias": 1.0, "status": "INITIAL", "biological_phase": "P_UNKNOWN"}

    def _context_for_state(self, state: dict) -> dict:
        cid = state.get("context_id") or state.get("climate_context_id")
        return self.contexts.get(cid, {}) if cid else {}

    def _daily_context_for_day(self, fattening_day: int, inner_df: pd.DataFrame) -> dict:
        if inner_df.empty or "climate_context_id" not in inner_df:
            return {}
        start_day = int(max(1, fattening_day))
        # approximate day slice using row index; 288 rows/day.
        lo, hi = (start_day - 1) * 288, start_day * 288
        x = inner_df.iloc[lo:hi]
        if x.empty: x = inner_df.tail(288)
        cid = x["climate_context_id"].dropna().mode()
        return self.contexts.get(str(cid.iloc[0]), {}) if not cid.empty else {}

    def _hash_state(self, row: dict) -> str:
        payload = json.dumps({k: row.get(k) for k in ["fattening_day","biological_phase","tbw_kg","adg_kg_day","feed_intake_kg_dm_day","heat_production"]}, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def _normalize_growth_rows(self, df: pd.DataFrame, daily_df: pd.DataFrame, growth_start_date: str) -> list[dict]:
        if df is None or df.empty:
            df = pd.DataFrame()
        rows = []
        n = max(len(daily_df), len(df), 1)
        prev_tbw = None
        for i in range(n):
            src = df.iloc[i].to_dict() if i < len(df) else {}
            day = int(src.get("fattening_day") or src.get("day") or src.get("DOY") or (i+1))
            tbw = float(pd.to_numeric(pd.Series([src.get("tbw_kg", src.get("TBW", src.get("body_weight", 0.0)))]), errors="coerce").fillna(0.0).iloc[0])
            if tbw <= 0: tbw = 45.0 + 0.8 * day
            fi = float(pd.to_numeric(pd.Series([src.get("feed_intake_kg_dm_day", src.get("feed_intake", src.get("FI", 5.0))) ]), errors="coerce").fillna(5.0).iloc[0])
            beef = float(pd.to_numeric(pd.Series([src.get("beef_production_kg", src.get("beef_production", src.get("beef", max(tbw-45.0,0.0))))]), errors="coerce").fillna(max(tbw-45.0,0.0)).iloc[0])
            hp = float(pd.to_numeric(pd.Series([src.get("heat_production", src.get("HP", 10.0 + 0.01 * tbw))]), errors="coerce").fillna(10.0+0.01*tbw).iloc[0])
            adg = tbw - prev_tbw if prev_tbw is not None else float(src.get("adg_kg_day", src.get("ADG", 0.0)) or 0.0)
            if adg == 0.0: adg = 0.8
            prev_tbw = tbw
            fe = adg / max(fi, 0.1)
            gdate = str((pd.to_datetime(growth_start_date) + pd.Timedelta(days=day-1)).date())
            phase = identify_stage(day, tbw)
            dominant = "heat_stress" if hp > 25 else ("energy" if fe < 0.08 else "balanced")
            rows.append({"fattening_day": day, "growth_date": src.get("growth_date", gdate), "tbw_kg": tbw, "feed_intake_kg_dm_day": fi, "beef_production_kg": beef, "heat_production": hp, "diet": src.get("diet", self.cfg.default_diet), "breed": src.get("breed", self.cfg.default_breed), "biological_phase": phase, "adg_kg_day": adg, "feed_efficiency": fe, "dominant_limitation": dominant, "status": "OK", **src})
        return rows

    def _parse_end_datetime(self, value: str) -> pd.Timestamp:
        ts = pd.to_datetime(value)
        if len(str(value).strip()) <= 10:
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return ts

    def _progress_mode(self, df5: pd.DataFrame) -> str:
        ts = pd.to_datetime(df5["timestamp"]); span = ts.max() - ts.min()
        if span > pd.Timedelta(days=7): return "day"
        if span > pd.Timedelta(days=1): return "hour"
        return "step"

    def _load_growth_output(self, out_dir: Path, case_id: int, fallback: pd.DataFrame) -> pd.DataFrame:
        p = out_dir / f"growth_optimizer_outputs_case_{case_id}.csv"
        if p.exists():
            try: return pd.read_csv(p)
            except Exception: pass
        return fallback

    def _progress_header(self, growth_start_date: str, end_date, total_steps: int, case_id: int, progress_mode: str, light_h: int, light_m: int, light_on: float, light_off: float | None) -> None:
        print("[TDDT TRAIN] Starting closed-loop trans-domain training", flush=True)
        print(f"  growth_start_date : {growth_start_date}", flush=True)
        if end_date is not None: print(f"  growth_end_date   : {end_date}", flush=True)
        print(f"  case_id           : {case_id}", flush=True)
        print(f"  inner_loop_steps  : {total_steps}", flush=True)
        print(f"  timestep          : {self.cfg.dt_seconds} seconds", flush=True)
        print(f"  progress_mode     : {progress_mode}", flush=True)
        print(f"  lighting_schedule : ON at {light_h:02d}:{light_m:02d}, on={light_on:g}h, off={0 if light_off is None else light_off:g}h", flush=True)
        print("  progress columns  : step, percent, timestamp, indoor_temp, comfort_band, ventilation, heating, light, safety, elapsed", flush=True)

    def _progress_step(self, step_idx: int, total_steps: int, state: dict, result: dict, command: dict, safety: str, started: float, progress_mode: str, seen: set[str]) -> None:
        timestamp = pd.to_datetime(state.get("timestamp"))
        if progress_mode == "hour":
            key = timestamp.strftime("%Y-%m-%d %H")
            if key in seen and step_idx != total_steps: return
            seen.add(key)
        elif progress_mode == "day":
            key = timestamp.strftime("%Y-%m-%d")
            if key in seen and step_idx != total_steps: return
            seen.add(key)
        else:
            interval = 1 if total_steps <= 60 else max(1, total_steps // 100)
            if step_idx != 1 and step_idx != total_steps and step_idx % interval != 0: return
        pct = 100.0 * step_idx / max(total_steps, 1); elapsed = time.time() - started
        temp = float(result.get("indoor_temp_c", state.get("indoor_temp_c", 0.0)))
        low = float(result.get("lct_c", result.get("safe_lct_c", -8.0))); high = float(result.get("uct_c", result.get("safe_uct_c", 26.0)))
        print(f"[TDDT TRAIN] step {step_idx:>6}/{total_steps:<6} ({pct:6.2f}%) | {timestamp} | T={temp:6.2f}C band=[{low:5.1f},{high:5.1f}] | vent={float(command.get('ventilation_group_pct',0)):5.1f}% heat={float(command.get('heating_group_pct',0)):5.1f}% light={int(bool(command.get('light_on')))} | safety={safety} | elapsed={elapsed:6.1f}s", flush=True)

    def _growth_to_guidance(self, growth_last: dict) -> dict:
        hp = float(growth_last.get("heat_production", 0.0) or 0.0)
        return {"preferred_temp_low_c": 5.0, "preferred_temp_high_c": 24.0, "heat_production_feedback": hp, "comfort_weight_bias": 1.1 if hp > 0 else 1.0, "gas_weight_bias": 1.0, "energy_weight_bias": 0.9, "status": "FALLBACK_GROWTH_GUIDANCE"}
