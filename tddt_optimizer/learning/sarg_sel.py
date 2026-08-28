from __future__ import annotations

from pathlib import Path
import itertools
import json
import math
import tempfile
import shutil
import sys
from typing import Any

import pandas as pd

DEFAULT_PHASES = [
    {"phase_id": "P1", "name": "early_growth", "start_day": 1, "end_day": 80},
    {"phase_id": "P2", "name": "post_weaning_adaptation", "start_day": 81, "end_day": 220},
    {"phase_id": "P3", "name": "frame_and_muscle_growth", "start_day": 221, "end_day": 450},
    {"phase_id": "P4", "name": "muscle_gain", "start_day": 451, "end_day": 750},
    {"phase_id": "P5", "name": "finishing", "start_day": 751, "end_day": 1000},
]

GROWTH_COLUMNS = {
    "tbw": "tbw_kg",
    "adg": "adg_kg_day",
    "fi": "feed_intake_kg_dm_day",
    "fe": "feed_efficiency",
    "hp": "heat_production",
    "beef": "beef_production_kg",
}


def identify_stage(growth_day: int, tbw: float | None = None) -> str:
    for p in DEFAULT_PHASES:
        if p["start_day"] <= int(growth_day) <= p["end_day"]:
            return p["phase_id"]
    return DEFAULT_PHASES[-1]["phase_id"]


def _sf(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    den = float(den)
    if abs(den) < 1e-9:
        return default
    return float(num) / den


def _programs(diets: list[int], phases: list[dict], max_programs: int | None = None) -> list[dict]:
    programs: list[dict] = []
    ref_id = 1
    for d in diets:
        programs.append({
            "reference_id": f"R{ref_id:03d}",
            "kind": "baseline_full_diet",
            "diet_phase_program": {p["phase_id"]: int(d) for p in phases},
        })
        ref_id += 1
    baseline = diets[len(diets) // 2]
    for p in phases:
        for d in diets:
            if d == baseline:
                continue
            prog = {ph["phase_id"]: int(baseline) for ph in phases}
            prog[p["phase_id"]] = int(d)
            programs.append({
                "reference_id": f"R{ref_id:03d}",
                "kind": "single_phase_swap",
                "swapped_phase": p["phase_id"],
                "diet_phase_program": prog,
            })
            ref_id += 1
    for combo in itertools.product(diets, repeat=len(phases)):
        if list(combo) != sorted(combo):
            continue
        prog = {phases[i]["phase_id"]: int(combo[i]) for i in range(len(phases))}
        if any(prog == p["diet_phase_program"] for p in programs):
            continue
        programs.append({
            "reference_id": f"R{ref_id:03d}",
            "kind": "monotonic_mixed_program",
            "diet_phase_program": prog,
        })
        ref_id += 1
        if max_programs is not None and len(programs) >= max_programs:
            break
    if max_programs is not None:
        programs = programs[:max_programs]
        # keep stable ids after truncation
        for i, item in enumerate(programs, start=1):
            item["reference_id"] = f"R{i:03d}"
    return programs


def _ensure_growth_input(daily_csv: str | Path | None, output_dir: Path, horizon_days: int) -> Path:
    """Return a daily climate input for LiGAPS. Falls back to bundled sample."""
    if daily_csv and Path(daily_csv).exists():
        df = pd.read_csv(daily_csv)
    else:
        # The bundled LiGAPS input already contains the expected domain schema.
        root = Path(__file__).resolve().parents[2]
        sample = root / "growth_simulator" / "FRACHA19982012_growth_input_1000d_observed_until_day_560.csv"
        if not sample.exists():
            raise FileNotFoundError("No prepared daily growth CSV and no bundled LiGAPS sample input were found")
        df = pd.read_csv(sample)
    if df.empty:
        raise ValueError("Growth reference daily input is empty")
    # Normalize expected columns and force all rows used for reference generation to observed.
    lower = {str(c).strip().lower(): c for c in df.columns}
    if "fattening_day" not in lower:
        df["fattening_day"] = range(1, len(df) + 1)
    else:
        tmp_day = pd.to_numeric(df[lower["fattening_day"]], errors="coerce")
        fallback_day = pd.Series(range(1, len(df) + 1), index=df.index)
        df["fattening_day"] = tmp_day.fillna(fallback_day).astype(int)
    if len(df) < horizon_days:
        # Extend with the last day so simulator has the requested full horizon.
        last = df.iloc[-1].copy()
        extras = []
        for day in range(len(df) + 1, horizon_days + 1):
            row = last.copy()
            row["fattening_day"] = day
            if "doy" in df.columns:
                row["doy"] = ((day - 1) % 365) + 1
            if "source_day" in df.columns:
                row["source_day"] = int(last.get("source_day", len(df)))
            extras.append(row)
        df = pd.concat([df, pd.DataFrame(extras)], ignore_index=True)
    df = df.iloc[:horizon_days].copy()
    df["fattening_day"] = range(1, len(df) + 1)
    df["is_observed"] = 1
    # LiGAPS expects lowercase climate fields. Fill missing ones with indoor/stable-safe defaults.
    defaults = {"yr": 1998, "doy": None, "rad": 0.0, "mint": 10.0, "maxt": 18.0, "vpr": 0.9, "wind": 1.0, "rain": 0.0, "aha": 0.8, "okta": 8.0, "source_day": None}
    for col, default in defaults.items():
        if col not in df.columns:
            if col == "doy":
                df[col] = ((df["fattening_day"] - 1) % 365) + 1
            elif col == "source_day":
                df[col] = df["fattening_day"]
            else:
                df[col] = default
    out = output_dir / "sarg_growth_reference_input_daily.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def _run_growth_for_diet(input_csv: Path, *, output_dir: Path, diet: int, breed: int, scale: int, sex_animal: int, housing: int, case_id: int, horizon_days: int, timeout_seconds: int) -> pd.DataFrame:
    root = Path(__file__).resolve().parents[2]
    growth_dir = root / "growth_simulator"
    sys.path.insert(0, str(growth_dir))
    from ligaps_growth_library import run_growth_endOf_cycle  # type: ignore

    df = pd.read_csv(input_csv).iloc[:horizon_days].copy()
    df["is_observed"] = 1
    run_dir = output_dir / "sarg_growth_simulator_runs" / f"diet_{int(diet)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # run_growth_endOf_cycle preserves the full daily CSV when keep_output_files=True.
    run_growth_endOf_cycle(
        df,
        output_dir=run_dir,
        breed=int(breed),
        diet=int(diet),
        scale=int(scale),
        sex_animal=int(sex_animal),
        housing=int(housing),
        case_id=int(case_id),
        observed_only=True,
        imax=int(horizon_days),
        timeout_seconds=int(timeout_seconds),
        generate_report=False,
        keep_output_files=True,
    )
    daily_path = run_dir / f"growth_optimizer_outputs_case_{int(case_id)}.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Growth simulator did not create expected reference trajectory: {daily_path}")
    traj = pd.read_csv(daily_path)
    if "fattening_day" not in traj.columns:
        traj["fattening_day"] = range(1, len(traj) + 1)
    traj["fattening_day"] = pd.to_numeric(traj["fattening_day"], errors="coerce").fillna(0).astype(int)
    traj["reference_source_diet"] = int(diet)
    if "feed_efficiency" not in traj.columns:
        adg = pd.to_numeric(traj.get("adg_kg_day", pd.Series(0, index=traj.index)), errors="coerce").fillna(0)
        if "adg_kg_day" not in traj.columns and "tbw_kg" in traj.columns:
            adg = pd.to_numeric(traj["tbw_kg"], errors="coerce").diff().fillna(0).clip(lower=0)
            traj["adg_kg_day"] = adg
        fi = pd.to_numeric(traj.get("feed_intake_kg_dm_day", pd.Series(0, index=traj.index)), errors="coerce").fillna(0)
        traj["feed_efficiency"] = adg / fi.replace(0, pd.NA)
        traj["feed_efficiency"] = traj["feed_efficiency"].fillna(0)
    return traj


def _summarize_reference(ref_id: str, ref_daily: pd.DataFrame, program: dict) -> dict:
    numeric = lambda c: pd.to_numeric(ref_daily.get(c, pd.Series(0, index=ref_daily.index)), errors="coerce").fillna(0)
    tbw = numeric("tbw_kg")
    adg = numeric("adg_kg_day") if "adg_kg_day" in ref_daily.columns else tbw.diff().fillna(0).clip(lower=0)
    fi = numeric("feed_intake_kg_dm_day")
    fe = numeric("feed_efficiency") if "feed_efficiency" in ref_daily.columns else adg / fi.replace(0, pd.NA)
    hp = numeric("heat_production")
    beef = numeric("beef_production_kg")
    heat_stress = numeric("heat_stress")
    cold_stress = numeric("cold_stress")
    final_tbw = float(tbw.iloc[-1]) if len(tbw) else 0.0
    mean_adg = float(adg.mean()) if len(adg) else 0.0
    total_fi = float(fi.sum()) if len(fi) else 0.0
    mean_fe = float(fe.replace([math.inf, -math.inf], 0).fillna(0).mean()) if len(fe) else 0.0
    mean_hp = float(hp.mean()) if len(hp) else 0.0
    final_beef = float(beef.iloc[-1]) if len(beef) else 0.0
    score = float(0.35 * final_tbw / 700.0 + 0.25 * mean_adg / 1.5 + 0.20 * mean_fe / 0.20 - 0.10 * total_fi / max(1.0, 10000.0) - 0.07 * mean_hp / 30.0 - 0.03 * float(heat_stress.mean() if len(heat_stress) else 0.0))
    return {
        "reference_id": ref_id,
        "diet_phase_program": program,
        "final_TBW": round(final_tbw, 6),
        "mean_ADG": round(mean_adg, 6),
        "total_feed_intake": round(total_fi, 6),
        "mean_FE": round(mean_fe, 6),
        "final_beef_production": round(final_beef, 6),
        "mean_heat_production": round(mean_hp, 6),
        "mean_heat_stress": round(float(heat_stress.mean()) if len(heat_stress) else 0.0, 6),
        "mean_cold_stress": round(float(cold_stress.mean()) if len(cold_stress) else 0.0, 6),
        "simulator_score": round(score, 6),
    }


def _compose_program_trajectory(program_item: dict, diet_trajectories: dict[int, pd.DataFrame], phases: list[dict], horizon_days: int) -> pd.DataFrame:
    rows = []
    prog = {str(k): int(v) for k, v in (program_item.get("diet_phase_program") or {}).items()}
    for phase in phases:
        pid = phase["phase_id"]
        d = int(prog.get(pid, 2))
        traj = diet_trajectories[d]
        seg = traj[(traj["fattening_day"] >= int(phase["start_day"])) & (traj["fattening_day"] <= int(phase["end_day"]))].copy()
        seg["reference_id"] = program_item["reference_id"]
        seg["reference_kind"] = program_item.get("kind", "mixed_program")
        seg["biological_phase"] = pid
        seg["program_diet"] = d
        rows.append(seg)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = out[out["fattening_day"] <= int(horizon_days)].sort_values("fattening_day").reset_index(drop=True)
    return out


def build_sarg_reference_library(
    output_dir: str | Path = "prepared",
    diets: list[int] | None = None,
    top_k: int = 3,
    *,
    daily_csv: str | Path | None = None,
    simulator_mode: bool = True,
    horizon_days: int = 1000,
    breed: int = 6,
    scale: int = 1,
    sex_animal: int = 0,
    housing: int = 0,
    case_id: int = 1,
    timeout_seconds: int = 180,
    max_programs: int | None = None,
) -> dict:
    """Build the SARG-SEL reference library from growth simulator queries.

    The reference library is no longer a heuristic design-only artifact. In the
    default ``simulator_mode=True``, the function queries the LiGAPS-Beef growth
    simulator once per candidate diet over the full horizon. Mixed phase
    reference programs are then composed from those simulator-generated daily
    trajectories, so every stored growth value originates from the domain growth
    simulator rather than from hand-written growth heuristics.

    Notes
    -----
    The current growth simulator API accepts one FEEDNR/diet per run. Therefore
    true dynamic diet switching inside one LiGAPS execution is not imposed on the
    monolithic model. Instead, this library stores full-horizon per-diet
    simulator trajectories and phase-wise compositions derived from them. The
    provenance fields in JSON and CSV make this explicit and auditable.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    diets = [int(x) for x in (diets or [1, 2, 3, 4, 5])]
    phases = DEFAULT_PHASES
    programs = _programs(diets, phases, max_programs=max_programs)
    if not simulator_mode:
        raise ValueError("Heuristic SARG reference generation has been disabled. Use simulator_mode=True.")

    input_csv = _ensure_growth_input(daily_csv, out, int(horizon_days))
    diet_trajectories: dict[int, pd.DataFrame] = {}
    per_diet_paths: dict[int, str] = {}
    for d in diets:
        traj = _run_growth_for_diet(
            input_csv,
            output_dir=out,
            diet=d,
            breed=breed,
            scale=scale,
            sex_animal=sex_animal,
            housing=housing,
            case_id=case_id,
            horizon_days=horizon_days,
            timeout_seconds=timeout_seconds,
        )
        diet_trajectories[d] = traj
        path = out / "sarg_growth_simulator_runs" / f"diet_{d}" / f"growth_optimizer_outputs_case_{int(case_id)}.csv"
        per_diet_paths[d] = str(path)

    daily_rows = []
    summary_rows = []
    for item in programs:
        ref_daily = _compose_program_trajectory(item, diet_trajectories, phases, horizon_days)
        summary = _summarize_reference(item["reference_id"], ref_daily, item["diet_phase_program"])
        item["reference_summary"] = summary
        item["reference_generation"] = {
            "mode": "growth_simulator_generated_phase_composition",
            "simulator": "LiGAPS-Beef Python port",
            "per_diet_trajectory_sources": {str(k): v for k, v in per_diet_paths.items()},
            "dynamic_switching_note": "The monolithic growth simulator accepts one diet per run; mixed programs are composed phase-wise from full-horizon simulator outputs.",
        }
        summary_rows.append({"reference_id": item["reference_id"], "kind": item.get("kind", ""), "simulator_score": summary["simulator_score"], **item["diet_phase_program"], **{k: v for k, v in summary.items() if k not in {"reference_id", "diet_phase_program"}}})
        keep_cols = [c for c in ["reference_id", "reference_kind", "biological_phase", "program_diet", "fattening_day", "growth_date", "tbw_kg", "adg_kg_day", "feed_intake_kg_dm_day", "feed_efficiency", "beef_production_kg", "heat_production", "heat_stress", "cold_stress", "energy", "protein", "digestion_cap"] if c in ref_daily.columns]
        daily_rows.append(ref_daily[keep_cols].copy())

    summary_df = pd.DataFrame(summary_rows).sort_values("simulator_score", ascending=False).reset_index(drop=True)
    daily_df = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()

    top_programs = summary_df.head(max(top_k * len(phases), top_k)).copy()
    phase_policy: dict[str, list[dict]] = {}
    for p in phases:
        pid = p["phase_id"]
        vc = top_programs[pid].value_counts().head(top_k)
        phase_policy[pid] = [
            {"diet": int(k), "support_count": int(v), "source": "growth_simulator_generated_reference"}
            for k, v in vc.items()
        ]

    library = {
        "method": "SARG-SEL",
        "library_type": "growth_simulator_generated_stage_aware_reference_library",
        "description": "SARG-SEL reference library generated by querying the LiGAPS-Beef growth simulator; no heuristic growth summaries are used.",
        "reference_generation_mode": "growth_simulator_generated",
        "main_tddt_episode_count": 1,
        "growth_simulator": {
            "name": "LiGAPS-Beef Python port",
            "query_count": len(diets),
            "per_diet_queries": diets,
            "breed": int(breed),
            "scale": int(scale),
            "sex_animal": int(sex_animal),
            "housing": int(housing),
            "case_id": int(case_id),
            "horizon_days": int(horizon_days),
            "input_csv": str(input_csv),
        },
        "phases": phases,
        "diets": diets,
        "reference_count": len(programs),
        "reference_daily_csv": "sarg_growth_reference_daily.csv",
        "reference_programs_csv": "sarg_reference_programs.csv",
        "reference_models": programs,
        "top_k_diet_phase_policy": phase_policy,
    }

    lib_path = out / "sarg_growth_reference_library.json"
    csv_path = out / "sarg_reference_programs.csv"
    topk_path = out / "sarg_diet_phase_topk_policy.json"
    daily_path = out / "sarg_growth_reference_daily.csv"
    lib_path.write_text(json.dumps(library, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_df.to_csv(csv_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    topk_path.write_text(json.dumps(phase_policy, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "library_path": str(lib_path),
        "reference_programs_csv": str(csv_path),
        "reference_daily_csv": str(daily_path),
        "topk_policy_path": str(topk_path),
        "reference_count": len(programs),
        "growth_simulator_query_count": len(diets),
        "reference_generation_mode": "growth_simulator_generated",
    }


class SARGOuterPolicy:
    """Stage-aware reference-guided outer-loop policy backed by simulator references."""

    def __init__(self, library_path: str | Path, weights: dict | None = None):
        self.path = Path(library_path)
        self.library = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"reference_models": [], "phases": DEFAULT_PHASES}
        self.refs = list(self.library.get("reference_models", []))
        self.phases = list(self.library.get("phases", DEFAULT_PHASES))
        self.weights = weights or {
            "growth": 0.35,
            "feed_efficiency": 0.25,
            "feed_cost": 0.12,
            "heat_production": 0.10,
            "thermal_stress": 0.08,
            "climate_energy": 0.05,
            "distance": 0.05,
        }
        self._daily_index: dict[tuple[str, int], dict] = {}
        daily_csv = self.library.get("reference_daily_csv")
        daily_path = (self.path.parent / daily_csv) if daily_csv else None
        if daily_path and daily_path.exists():
            df = pd.read_csv(daily_path)
            if "reference_id" in df.columns and "fattening_day" in df.columns:
                for row in df.to_dict("records"):
                    self._daily_index[(str(row.get("reference_id")), int(_sf(row.get("fattening_day"), 0)))] = row

    def phase(self, growth_day: int, tbw: float | None = None) -> str:
        return identify_stage(int(growth_day or 1), tbw)

    def reference_state(self, ref: dict, growth_day: int) -> dict:
        day = max(1, int(growth_day))
        rid = str(ref.get("reference_id", "R_DEFAULT"))
        row = self._daily_index.get((rid, day))
        p = self.phase(day)
        d = int((ref.get("diet_phase_program") or {}).get(p, 2))
        if row:
            tbw = _sf(row.get("tbw_kg"), 0.0)
            adg = _sf(row.get("adg_kg_day"), 0.0)
            fi = _sf(row.get("feed_intake_kg_dm_day"), 0.0)
            fe = _sf(row.get("feed_efficiency"), _safe_ratio(adg, fi, 0.0))
            hp = _sf(row.get("heat_production"), 0.0)
            return {"reference_id": rid, "phase": p, "diet": d, "TBW": tbw, "ADG": adg, "FI": fi, "FE": fe, "HP": hp}
        # Fallback only if old/partial library is loaded; mark low trust by conservative values.
        summary = ref.get("reference_summary", {}) or ref.get("reference_summary_prior", {}) or {}
        return {
            "reference_id": rid,
            "phase": p,
            "diet": d,
            "TBW": _sf(summary.get("final_TBW", summary.get("final_TBW_prior", 0.0)), 0.0),
            "ADG": _sf(summary.get("mean_ADG", summary.get("mean_ADG_prior", 0.0)), 0.0),
            "FI": _sf(summary.get("total_feed_intake", summary.get("total_feed_intake_prior", 0.0)), 0.0) / max(1, day),
            "FE": _sf(summary.get("mean_FE", summary.get("mean_FE_prior", 0.0)), 0.0),
            "HP": _sf(summary.get("mean_heat_production", summary.get("mean_heat_production_prior", 0.0)), 0.0),
        }

    def score_references(self, state: dict, climate_context: dict | None = None) -> tuple[dict, list[dict]]:
        climate_context = climate_context or {}
        day = int(_sf(state.get("fattening_day", state.get("growth_day", 1)), 1))
        tbw = _sf(state.get("tbw_kg", state.get("TBW", 0.0)), 0.0)
        adg = _sf(state.get("adg_kg_day", state.get("ADG", 0.0)), 0.0)
        fi = _sf(state.get("feed_intake_kg_dm_day", state.get("FI", 0.0)), 0.0)
        fe = _sf(state.get("feed_efficiency", state.get("FE", 0.0)), 0.0)
        hp = _sf(state.get("heat_production", state.get("HP", 0.0)), 0.0)
        phase = self.phase(day, tbw)
        risk = climate_context.get("risk_profile", {}) if isinstance(climate_context, dict) else {}
        heat_risk = 1.0 if risk.get("heat_stress") == "high" else (0.5 if risk.get("heat_stress") == "medium" else 0.0)
        cold_risk = 1.0 if risk.get("cold_stress") == "high" else 0.0
        rows: list[dict] = []
        refs = self.refs or [{"reference_id": "R_DEFAULT", "diet_phase_program": {p["phase_id"]: 2 for p in self.phases}}]
        for ref in refs:
            r = self.reference_state(ref, day)
            atbw = tbw if tbw > 0 else r["TBW"] * 0.95
            aadg = adg if adg > 0 else r["ADG"] * 0.90
            afi = fi if fi > 0 else r["FI"]
            afe = fe if fe > 0 else _safe_ratio(aadg, afi, 0.0)
            ahp = hp if hp > 0 else r["HP"]
            growth_score = max(0.0, min(1.2, _safe_ratio(atbw, max(r["TBW"], 1e-6), 0.0))) / 1.2 * 0.55 + max(0.0, min(1.2, _safe_ratio(aadg, max(r["ADG"], 1e-6), 0.0))) / 1.2 * 0.45
            fe_score = max(0.0, min(1.5, _safe_ratio(afe, max(r["FE"], 1e-6), 0.0))) / 1.5
            feed_cost = max(0.0, _safe_ratio(afi, max(r["FI"], 1e-6), 0.0) - 0.85)
            hp_pen = max(0.0, _safe_ratio(ahp, max(r["HP"], 1e-6), 0.0) - 0.85) * (1.0 + heat_risk)
            ts_pen = heat_risk * max(0.0, ahp - r["HP"] * 0.85) / max(r["HP"], 1.0)
            climate_energy = cold_risk * 0.25 + heat_risk * 0.20 + 0.02 * r["diet"]
            dist = (
                abs(atbw - r["TBW"]) / max(abs(r["TBW"]), 1.0)
                + abs(aadg - r["ADG"]) / max(abs(r["ADG"]), 0.1)
                + abs(afi - r["FI"]) / max(abs(r["FI"]), 0.1)
                + abs(afe - r["FE"]) / max(abs(r["FE"]), 0.001)
            ) / 4.0
            w = self.weights
            final = w["growth"] * growth_score + w["feed_efficiency"] * fe_score - w["feed_cost"] * feed_cost - w["heat_production"] * hp_pen - w["thermal_stress"] * ts_pen - w["climate_energy"] * climate_energy - w["distance"] * dist
            rows.append({
                "growth_day": day,
                "biological_phase": phase,
                "climate_context_id": climate_context.get("context_id"),
                "reference_id": str(ref.get("reference_id", "R_DEFAULT")),
                "growth_score": growth_score,
                "feed_efficiency_score": fe_score,
                "feed_cost_penalty": feed_cost,
                "heat_production_penalty": hp_pen,
                "thermal_stress_penalty": ts_pen,
                "climate_energy_penalty": climate_energy,
                "state_distance_delta": dist,
                "final_score": final,
                "diet": r["diet"],
                "reference_state": r,
                "reference_generation_mode": self.library.get("reference_generation_mode", "unknown"),
            })
        rows.sort(key=lambda x: x["final_score"], reverse=True)
        for i, row in enumerate(rows, start=1):
            row["rank"] = i
            row["selected"] = 1 if i == 1 else 0
        return rows[0], rows

    def guidance_from_scores(self, state: dict, climate_context: dict | None = None) -> tuple[dict, list[dict]]:
        best, rows = self.score_references(state, climate_context)
        risk = (climate_context or {}).get("risk_profile", {}) if isinstance(climate_context, dict) else {}
        prior = (climate_context or {}).get("mpc_prior", {}) if isinstance(climate_context, dict) else {}
        phase = best["biological_phase"]
        diet = int(best.get("diet", 2))
        heat_risk = risk.get("heat_stress") == "high"
        cold_risk = risk.get("cold_stress") == "high"
        margin = best["final_score"] - (rows[1]["final_score"] if len(rows) > 1 else 0.0)
        conf = max(0.05, min(0.98, 0.50 + 0.50 * margin))
        if heat_risk:
            temp_pos = "lower_middle_band"
            v_bias = max(0.65, _sf(prior.get("ventilation_bias"), 0.4))
            h_bias = 0.0
        elif cold_risk:
            temp_pos = "center_to_upper_band"
            v_bias = min(0.25, _sf(prior.get("ventilation_bias"), 0.2))
            h_bias = max(0.55, _sf(prior.get("heating_bias"), 0.5))
        elif best["state_distance_delta"] > 0.20:
            temp_pos = "center_to_upper_band"
            v_bias = 0.25
            h_bias = 0.35
        else:
            temp_pos = "center_band"
            v_bias = _sf(prior.get("ventilation_bias"), 0.30)
            h_bias = _sf(prior.get("heating_bias"), 0.10)
        preferred_low = 7.0 if cold_risk else (4.0 if heat_risk else 5.0)
        preferred_high = 22.0 if heat_risk else (25.0 if cold_risk else 24.0)
        guidance = {
            "preferred_temp_low_c": preferred_low,
            "preferred_temp_high_c": preferred_high,
            "preferred_temp_position": temp_pos,
            "heat_production_feedback": best.get("heat_production_penalty", 0.0),
            "comfort_weight_bias": 1.15 if heat_risk or cold_risk else 1.0,
            "gas_weight_bias": 1.05 if heat_risk else 1.0,
            "energy_weight_bias": 0.75 if heat_risk or cold_risk else 1.0,
            "ventilation_bias": v_bias,
            "heating_bias": h_bias,
            "diet_phase_guidance": f"phase={phase};diet_like={diet}",
            "growth_priority": "recover_growth_gap" if best["state_distance_delta"] > 0.2 else "maintain_efficiency",
            "best_reference_id": best["reference_id"],
            "reference_score": best["final_score"],
            "reference_distance_delta": best["state_distance_delta"],
            "confidence": conf,
            "biological_phase": phase,
            "climate_context_id": (climate_context or {}).get("context_id"),
            "top_k_reference_ids": [r["reference_id"] for r in rows[:3]],
            "reason_code": f"SARG:{phase}:simulator_reference:risk_heat={heat_risk}:risk_cold={cold_risk}:diet={diet}",
            "reference_generation_mode": self.library.get("reference_generation_mode", "unknown"),
            "status": "SARG_POLICY",
            "ccll_prior": prior,
        }
        return guidance, rows
