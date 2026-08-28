from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import json
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPTIMIZER_JSON = ROOT / "optimizer.json"
LEGACY_OPTIMIZER_JSON = ROOT / "tddt_optimizer" / "configs" / "optimizer.json"

PATH_FIELDS = {
    "dataset_csv", "prepared_dir", "prepared_5m_csv", "prepared_daily_csv",
    "prepared_ccll_5m_csv", "ccll_library_json", "ccll_daily_descriptors_csv",
    "sarg_reference_library_json", "sarg_reference_programs_csv",
    "sarg_diet_phase_topk_policy_json", "sqlite_path", "report_dir",
    "setd_kstore_dir", "rl_state_path", "bayesian_tuner_path", "state_estimator_path", "ss_kstore_dir",
}

RELATIVE_DEFAULTS = {
    "optimizer_json": "optimizer.json",
    "dataset_csv": "dataset/CR_7R7_ZARGAR_36085925_50391289.csv",
    "prepared_dir": "prepared",
    "prepared_5m_csv": "prepared/climate_5m_all_rows.csv",
    "prepared_daily_csv": "prepared/growth_daily_all_rows.csv",
    "prepared_ccll_5m_csv": "prepared/climate_5m_ccll_all_rows.csv",
    "ccll_library_json": "prepared/climate_context_local_library.json",
    "ccll_daily_descriptors_csv": "prepared/climate_context_daily_descriptors.csv",
    "sarg_reference_library_json": "prepared/sarg_growth_reference_library.json",
    "sarg_reference_programs_csv": "prepared/sarg_reference_programs.csv",
    "sarg_diet_phase_topk_policy_json": "prepared/sarg_diet_phase_topk_policy.json",
    "sqlite_path": "database/tddt_runtime.sqlite",
    "report_dir": "reports",
    "setd_kstore_dir": "models/setd_kstore",
    "rl_state_path": "models/setd_kstore/edge_multilayer_rl_policy.json",
    "bayesian_tuner_path": "models/setd_kstore/edge_contextual_bayesian_tuner.json",
    "state_estimator_path": "models/setd_kstore/edge_state_estimator.json",
    "ss_kstore_dir": "working/ss_kstore",
}


def _rooted(rel: str) -> str:
    return str(ROOT / rel)


@dataclass
class TDDTConfig:
    project_root: str = str(ROOT)
    optimizer_json: str = str(DEFAULT_OPTIMIZER_JSON)

    dataset_csv: str = _rooted(RELATIVE_DEFAULTS["dataset_csv"])
    prepared_dir: str = _rooted(RELATIVE_DEFAULTS["prepared_dir"])
    prepared_5m_csv: str = _rooted(RELATIVE_DEFAULTS["prepared_5m_csv"])
    prepared_daily_csv: str = _rooted(RELATIVE_DEFAULTS["prepared_daily_csv"])
    prepared_ccll_5m_csv: str = _rooted(RELATIVE_DEFAULTS["prepared_ccll_5m_csv"])
    ccll_library_json: str = _rooted(RELATIVE_DEFAULTS["ccll_library_json"])
    ccll_daily_descriptors_csv: str = _rooted(RELATIVE_DEFAULTS["ccll_daily_descriptors_csv"])
    sarg_reference_library_json: str = _rooted(RELATIVE_DEFAULTS["sarg_reference_library_json"])
    sarg_reference_programs_csv: str = _rooted(RELATIVE_DEFAULTS["sarg_reference_programs_csv"])
    sarg_diet_phase_topk_policy_json: str = _rooted(RELATIVE_DEFAULTS["sarg_diet_phase_topk_policy_json"])

    sqlite_path: str = _rooted(RELATIVE_DEFAULTS["sqlite_path"])
    report_dir: str = _rooted(RELATIVE_DEFAULTS["report_dir"])
    setd_kstore_dir: str = _rooted(RELATIVE_DEFAULTS["setd_kstore_dir"])

    # Runtime temporary workspace. This is intentionally allowed to be an
    # absolute path and is not normalized under project_root. Users can move it
    # to a large disk by editing optimizer.json.
    tmp_dir: str = "/tmp/tddt_livestock/"
    tmp_cleanup_interval_steps: int = 4000

    dt_seconds: int = 300
    climate_horizon_seconds: int = 900

    default_breed: int = 6
    default_diet: int = 2
    default_scale: int = 1
    default_sex_animal: int = 0
    default_housing: int = 0
    default_case_id: int = 1

    cattle_count: int = 4
    average_weight_kg: float = 508.5
    average_heat_multiplier: float = 1.415

    candidate_ventilation: tuple = (0.0, 25.0, 50.0, 75.0, 100.0)
    candidate_heating: tuple = (0.0, 25.0, 50.0, 75.0, 100.0)
    comfort_weight: float = 100.0
    energy_weight: float = 0.05
    gas_weight: float = 0.2

    light_on_hour: int = 5
    light_on_minute: int = 0
    light_hours_on: float = 16.0
    light_hours_off: float = 8.0

    ccll_context_count: int = 12
    ccll_max_iter: int = 80
    ccll_random_seed: int = 42
    ccll_feature_columns: tuple = ("t_min", "t_max", "t_mean", "rh_mean", "wind_mean", "rad_sum", "rain_sum", "t_range")

    sarg_diets: tuple = (1, 2, 3, 4, 5)
    sarg_top_k: int = 3
    sarg_reference_horizon_days: int = 1000
    sarg_reference_timeout_seconds: int = 180
    sarg_reference_max_programs: int | None = None
    sarg_reference_generation_mode: str = "growth_simulator_generated"

    async_sqlite_batch_size: int = 256
    async_sqlite_flush_seconds: float = 1.0

    rl_enabled: bool = True
    rl_alpha: float = 0.08
    rl_gamma: float = 0.90
    rl_epsilon: float = 0.02
    rl_weight: float = 8.0
    rl_state_path: str = _rooted(RELATIVE_DEFAULTS["rl_state_path"])
    bayesian_tuner_path: str = _rooted(RELATIVE_DEFAULTS["bayesian_tuner_path"])
    state_estimator_path: str = _rooted(RELATIVE_DEFAULTS["state_estimator_path"])
    bayesian_tuner_enabled: bool = True
    bayesian_tuner_alpha: float = 0.05
    bayesian_tuner_exploration_weight: float = 0.15
    # LBG-OT: lightweight unsupervised Gaussian context layer for online tuning.
    gaussian_tuner_alpha: float = 0.025
    gaussian_tuner_tau: float = 7.0
    gaussian_tuner_safe_blend_max: float = 0.85
    state_estimator_enabled: bool = True
    state_estimator_alpha: float = 0.025
    state_estimator_uncertainty_decay: float = 0.995
    ss_kstore_dir: str = _rooted(RELATIVE_DEFAULTS["ss_kstore_dir"])
    ss_segment_size: int = 1000
    ss_sample_every: int = 50

    sarg_growth_weight: float = 0.35
    sarg_feed_efficiency_weight: float = 0.25
    sarg_feed_cost_weight: float = 0.12
    sarg_heat_production_weight: float = 0.10
    sarg_thermal_stress_weight: float = 0.08
    sarg_climate_energy_weight: float = 0.05
    sarg_state_distance_weight: float = 0.05
    mpc_context_weight: float = 15.0
    mpc_guidance_weight: float = 8.0
    # Actuator anti-wear terms: soft penalties against abrupt jumps and oscillatory commands.
    mpc_switch_weight: float = 2.5
    mpc_max_delta_pct_per_step: float = 50.0
    mpc_oscillation_weight: float = 4.0
    mpc_heating_ventilation_conflict_weight: float = 25.0
    report_fan_kw_at_100pct: float = 4.0
    report_heater_kw_at_100pct: float = 12.0
    report_light_kw_when_on: float = 1.2
    # Weather-normalized low-energy accuracy report parameters.
    # These are used only in reports, not in the control policy.
    report_climate_energy_base_kw: float = 0.05
    report_heating_degree_reference_c: float = 20.0
    report_cooling_degree_reference_c: float = 12.0
    report_weather_energy_budget_factor: float = 0.75
    report_energy_comfort_penalty_factor: float = 1.0

    def to_dict(self) -> dict:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, tuple):
                data[key] = list(value)
        return data

    def to_relative_dict(self) -> dict:
        data = self.to_dict()
        data["project_root"] = "."
        data["optimizer_json"] = "optimizer.json"
        for key in PATH_FIELDS:
            if key in RELATIVE_DEFAULTS:
                data[key] = RELATIVE_DEFAULTS[key]
        return data


def _coerce_types(data: dict[str, Any]) -> dict[str, Any]:
    field_map = {f.name: f for f in fields(TDDTConfig)}
    tuple_fields = {"candidate_ventilation", "candidate_heating", "ccll_feature_columns", "sarg_diets"}
    out = {}
    for key, value in data.items():
        if key not in field_map:
            continue
        if key in tuple_fields and isinstance(value, list):
            out[key] = tuple(value)
        else:
            out[key] = value
    return out


def _resolve_config_path(path: str | Path | None = None) -> Path:
    if path:
        p = Path(path)
        return p if p.is_absolute() else ROOT / p
    if DEFAULT_OPTIMIZER_JSON.exists():
        return DEFAULT_OPTIMIZER_JSON
    if LEGACY_OPTIMIZER_JSON.exists():
        return LEGACY_OPTIMIZER_JSON
    return DEFAULT_OPTIMIZER_JSON


def _normalize_loaded_paths(data: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    out = dict(data)
    old_root = Path(str(out.get("project_root", root)))
    if not old_root.is_absolute():
        old_root = root / old_root
    out["project_root"] = str(root)
    out["optimizer_json"] = str(DEFAULT_OPTIMIZER_JSON)
    for key in PATH_FIELDS:
        rel_default = RELATIVE_DEFAULTS.get(key)
        val = out.get(key, rel_default)
        if val is None:
            continue
        p = Path(str(val))
        if p.is_absolute():
            try:
                rel = p.relative_to(old_root)
            except Exception:
                rel = Path(rel_default) if rel_default else Path(p.name)
            out[key] = str(root / rel)
        else:
            out[key] = str(root / p)
    return out


def load_config(path: str | Path | None = None) -> TDDTConfig:
    p = _resolve_config_path(path)
    cfg = TDDTConfig()
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        data = _normalize_loaded_paths(data, ROOT)
        base = cfg.to_dict()
        base.update(_coerce_types(data))
        cfg = TDDTConfig(**_coerce_types(base))
    return cfg


def write_default_config(path: str | Path | None = None) -> None:
    p = _resolve_config_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = TDDTConfig(optimizer_json=str(p))
    p.write_text(json.dumps(cfg.to_relative_dict(), indent=2), encoding="utf-8")
