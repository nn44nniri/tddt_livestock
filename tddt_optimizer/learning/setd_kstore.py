from __future__ import annotations
from pathlib import Path
import json
import sqlite3
from datetime import datetime
import pandas as pd


def _read_table(conn, name: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(f"SELECT * FROM {name}", conn)
    except Exception:
        return pd.DataFrame()


def _write_df(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except Exception:
            path = path.with_suffix(".csv")
            df.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)
    return str(path)


def _read_json(path: str | Path | None) -> dict:
    if not path: return {}
    p = Path(path)
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}


def _policy_rules(guidance: pd.DataFrame, sarg_ledger: pd.DataFrame) -> list[dict]:
    rules = []
    if guidance.empty:
        return rules
    sort_cols = [c for c in ["biological_phase", "climate_context_id", "reference_score", "confidence"] if c in guidance.columns]
    g = guidance.copy()
    if "reference_score" in g: g["reference_score"] = pd.to_numeric(g["reference_score"], errors="coerce").fillna(0)
    if "confidence" in g: g["confidence"] = pd.to_numeric(g["confidence"], errors="coerce").fillna(0.5)
    group_cols = [c for c in ["biological_phase", "climate_context_id"] if c in g.columns]
    if group_cols:
        idx = g.groupby(group_cols, dropna=False)["reference_score"].idxmax() if "reference_score" in g else g.groupby(group_cols, dropna=False).tail(1).index
        selected = g.loc[idx].copy()
    else:
        selected = g.tail(200)
    for _, row in selected.iterrows():
        rules.append({
            "biological_phase": row.get("biological_phase"),
            "climate_context_id": row.get("climate_context_id"),
            "best_reference_id": row.get("best_reference_id"),
            "top_k_reference_ids": json.loads(row.get("top_k_reference_ids") or "[]") if isinstance(row.get("top_k_reference_ids"), str) else row.get("top_k_reference_ids"),
            "recommended_guidance": {
                "preferred_temp_low_c": row.get("preferred_temp_low_c"),
                "preferred_temp_high_c": row.get("preferred_temp_high_c"),
                "preferred_temp_position": row.get("preferred_temp_position"),
                "comfort_weight_bias": row.get("comfort_weight_bias"),
                "energy_weight_bias": row.get("energy_weight_bias"),
                "gas_weight_bias": row.get("gas_weight_bias"),
                "ventilation_bias": row.get("ventilation_bias"),
                "heating_bias": row.get("heating_bias"),
                "diet_phase_guidance": row.get("diet_phase_guidance"),
            },
            "confidence": row.get("confidence", 0.5),
            "support_count": int(len(g[(g.get("biological_phase") == row.get("biological_phase")) & (g.get("climate_context_id") == row.get("climate_context_id"))])) if group_cols else 1,
            "reason_code": row.get("reason_code"),
        })
    return rules


def export_setd_kstore(sqlite_path: str | Path, output_dir: str | Path = "models/setd_kstore", *, ccll_path: str | Path | None = None, sarg_path: str | Path | None = None, tddt_version: str = "unknown") -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    inner = _read_table(conn, "inner_sensor_log")
    commands = _read_table(conn, "actuator_commands")
    growth = _read_table(conn, "outer_growth_state")
    guidance = _read_table(conn, "growth_climate_guidance")
    sarg_ledger = _read_table(conn, "sarg_objective_score_ledger")
    rl_memory = _read_table(conn, "rl_policy_memory")
    formal_links = _read_table(conn, "setd_formal_links")

    single_episode_path = _write_df(inner, out / "single_episode_trace.parquet")
    decision = inner[[c for c in inner.columns if c in ["timestamp","climate_context_id","climate_context_name","ventilation_group_pct","heating_group_pct","light_on","comfort_error","energy_cost","gas_penalty","context_penalty","guidance_penalty","rl_prior_cost","mpc_score","rl_state_key","rl_action_key"]]].copy() if not inner.empty else commands.copy()
    if not commands.empty and "timestamp" in commands:
        decision = decision.merge(commands[["timestamp","safety_status","command_json"]], on="timestamp", how="left") if not decision.empty and "timestamp" in decision else commands
    decision_path = _write_df(decision, out / "climate_decision_memory.parquet")
    growth_path = _write_df(growth, out / "daily_growth_state_memory.parquet")
    td_path = _write_df(guidance, out / "transdomain_guidance_memory.parquet")
    ledger_path = _write_df(sarg_ledger, out / "objective_score_ledger.parquet")
    rl_path = _write_df(rl_memory, out / "rl_policy_memory.parquet")
    formal_path = _write_df(formal_links, out / "formal_relation_ledger.parquet")

    ccll = _read_json(ccll_path)
    sarg = _read_json(sarg_path)
    ccll_contexts = ccll.get("contexts", []) if isinstance(ccll, dict) else []
    seen_contexts = sorted([str(x) for x in inner.get("climate_context_id", pd.Series(dtype=str)).dropna().unique().tolist()]) if not inner.empty and "climate_context_id" in inner else []
    all_contexts = sorted([str(c.get("context_id")) for c in ccll_contexts if c.get("context_id")])
    unseen = sorted(list(set(all_contexts) - set(seen_contexts)))

    rules = _policy_rules(guidance, sarg_ledger)
    stage_policy = {"policy_type": "stage_context_guidance_policy", "selection_rule": "max_reference_score_per_stage_context", "rules": rules}
    (out / "stage_context_policy.json").write_text(json.dumps(stage_policy, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Best learned actuator lookup by climate context and biological/RL state.
    actuator_lookup = []
    if not decision.empty:
        d = decision.copy()
        if "mpc_score" in d: d["mpc_score"] = pd.to_numeric(d["mpc_score"], errors="coerce").fillna(float("inf"))
        group_cols = [c for c in ["climate_context_id", "rl_state_key"] if c in d.columns]
        if group_cols and "mpc_score" in d:
            for _, row in d.loc[d.groupby(group_cols, dropna=False)["mpc_score"].idxmin()].iterrows():
                actuator_lookup.append({"climate_context_id": row.get("climate_context_id"), "rl_state_key": row.get("rl_state_key"), "ventilation_group_pct": row.get("ventilation_group_pct"), "heating_group_pct": row.get("heating_group_pct"), "light_on": row.get("light_on"), "mpc_score": row.get("mpc_score")})

    snapshot = {
        "policy_type": "single_episode_transdomain_policy_snapshot",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "learning_mode": "single_episode_reference_and_context_guided_rl",
        "climate_context_library_path": str(ccll_path or "prepared/climate_context_local_library.json"),
        "growth_reference_library_path": str(sarg_path or "prepared/sarg_growth_reference_library.json"),
        "guidance_rows": int(len(guidance)), "decision_rows": int(len(decision)), "episode_trace_rows": int(len(inner)),
        "best_stage_context_guidance_rules": rules,
        "best_actuator_lookup": actuator_lookup[:5000],
        "rl_policy_memory_path": str(rl_path),
        "fallback_safety_policy": {"if_context_unseen": "nearest_context_low_confidence", "if_state_unseen": "phase_level_policy", "if_both_unseen": "safe_mpc"},
    }
    (out / "learned_policy_snapshot.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    coverage = {
        "coverage": {"episode_rows": int(len(inner)), "growth_rows": int(len(growth)), "guidance_rows": int(len(guidance)), "sarg_score_rows": int(len(sarg_ledger)), "rl_update_rows": int(len(rl_memory)), "climate_contexts_seen_in_episode": seen_contexts, "climate_contexts_from_library": all_contexts, "unseen_contexts": unseen, "uses_ccll_sel": bool(ccll), "uses_sarg_sel": bool(sarg)},
        "uncertainty_policy": snapshot["fallback_safety_policy"],
    }
    (out / "coverage_uncertainty_report.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "storage_name": "SETD-KStore",
        "tddt_version": tddt_version,
        "created_at": snapshot["created_at"],
        "learning_mode": "single_episode",
        "growth_method": "SARG-SEL",
        "climate_method": "CCLL-SEL",
        "rl_method": "edge_multi_layer_tabular_rl_with_mpc_bias_adapter",
        "main_episode_count": 1,
        "formal_relations_enforced": ["CCLL context -> MPC prior", "SARG score -> biological guidance", "guidance -> MPC weights", "MPC decision -> RL update", "best learned policy -> snapshot"],
        "files": {
            "single_episode_trace": single_episode_path,
            "climate_decision_memory": decision_path,
            "daily_growth_state_memory": growth_path,
            "transdomain_guidance_memory": td_path,
            "objective_score_ledger": ledger_path,
            "rl_policy_memory": rl_path,
            "formal_relation_ledger": formal_path,
            "learned_policy_snapshot": str(out / "learned_policy_snapshot.json"),
            "stage_context_policy": str(out / "stage_context_policy.json"),
            "coverage_uncertainty_report": str(out / "coverage_uncertainty_report.json"),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS learned_model_registry (model_id TEXT PRIMARY KEY, created_at TEXT, tddt_version TEXT, storage_method TEXT, policy_snapshot_path TEXT, manifest_path TEXT)")
        conn.execute("INSERT OR REPLACE INTO learned_model_registry VALUES (?,?,?,?,?,?)", ("setd_kstore_latest", snapshot["created_at"], tddt_version, "SETD-KStore", str(out / "learned_policy_snapshot.json"), str(out / "manifest.json")))
        conn.commit()
    except Exception:
        pass
    conn.close()
    return manifest
