from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
BUILD_DIR_CANDIDATES = [REPO_ROOT / "build_updated", REPO_ROOT / "build"]
DEFAULT_IFC = REPO_ROOT / "Building_Information" / "beef_hall_120.ifc"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default_hall.cfg"
DEFAULT_OUTDOOR_CSV = REPO_ROOT / "data" / "sample_outdoor.csv"
DEFAULT_CONTROLS_CSV = REPO_ROOT / "data" / "sample_controls.csv"
DEFAULT_HERD_CFG = REPO_ROOT / "configs" / "herd_inventory.cfg"
DEFAULT_HERD_PROCESSED_CFG = REPO_ROOT / "configs" / "herd_inventory_processed.cfg"

@dataclass
class ExecutablePaths:
    beef_climate_sim: Path
    herd_inventory_cli: Path
    ifc_probe: Path


def _find_executables() -> ExecutablePaths:
    for build_dir in BUILD_DIR_CANDIDATES:
        beef = build_dir / "beef_climate_sim"
        herd = build_dir / "herd_inventory_cli"
        ifc = build_dir / "ifc_probe"
        if beef.exists() and herd.exists() and ifc.exists():
            return ExecutablePaths(beef_climate_sim=beef, herd_inventory_cli=herd, ifc_probe=ifc)
    raise FileNotFoundError("Could not locate bundled executables in build_updated/ or build/.")


def build_cython_extension() -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=REPO_ROOT, check=True, text=True, capture_output=True)


def _import_package() -> Any | None:
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        return importlib.import_module("climate_hall_simulator")
    except Exception:
        return None


def _api_function(name: str):
    module = _import_package()
    if module is None:
        return None
    return getattr(module, name, None)


def _run_cli(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True, capture_output=True)


def _parse_ifc_probe_output(stdout: str) -> pd.DataFrame:
    rows = []
    section = "general"
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":"):
            section = line[:-1].strip().lower().replace(" ", "_")
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            rows.append({"section": section, "key": "line", "value": line})
            continue
        rows.append({"section": section, "key": key.strip(), "value": value.strip()})
    return pd.DataFrame(rows)


def _read_processed_herd_cfg(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, cohort_rows = [], []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("cohort="):
            breed, avg_weight_kg, avg_heat_multiplier, count, breed_library = [x.strip() for x in line.split("=", 1)[1].split(",")]
            cohort_rows.append({"breed": breed, "avg_weight_kg": float(avg_weight_kg), "avg_heat_multiplier": float(avg_heat_multiplier), "count": int(count), "breed_library": int(breed_library)})
        elif "=" in line:
            key, value = line.split("=", 1)
            summary_rows.append({"key": key.strip(), "value": value.strip()})
    return pd.DataFrame(summary_rows), pd.DataFrame(cohort_rows)


def _normalize_frame_output(result: Any) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, dict):
        if result and all(isinstance(v, list) for v in result.values()):
            return pd.DataFrame(result)
        return pd.DataFrame([result])
    if isinstance(result, list):
        return pd.DataFrame(result)
    return pd.DataFrame([{"result": result}])


def inspect_ifc_file(ifc_path: str | Path = DEFAULT_IFC) -> pd.DataFrame:
    fn = _api_function("inspect_ifc_file")
    if callable(fn):
        return _normalize_frame_output(fn(str(ifc_path)))
    completed = _run_cli([str(_find_executables().ifc_probe), str(Path(ifc_path))])
    return _parse_ifc_probe_output(completed.stdout)


def process_herd_inventory_file(raw_cfg: str | Path = DEFAULT_HERD_CFG, processed_cfg: str | Path = DEFAULT_HERD_PROCESSED_CFG) -> dict[str, pd.DataFrame]:
    raw_cfg = Path(raw_cfg)
    processed_cfg = Path(processed_cfg)
    fn = _api_function("process_herd_inventory_file")
    if callable(fn):
        result = fn(str(raw_cfg), str(processed_cfg))
        summary, cohorts = _read_processed_herd_cfg(processed_cfg)
        return {"api_result": _normalize_frame_output(result), "summary": summary, "cohorts": cohorts}
    _run_cli([str(_find_executables().herd_inventory_cli), "--file", str(raw_cfg), "--processed", str(processed_cfg), "--process"])
    summary, cohorts = _read_processed_herd_cfg(processed_cfg)
    return {"summary": summary, "cohorts": cohorts}


def write_precomputed_processed_herd(processed_cfg: str | Path, cattle_count: int, average_weight_kg: float, average_heat_multiplier: float, cohorts: list[str]) -> dict[str, pd.DataFrame]:
    processed_cfg = Path(processed_cfg)
    cmd = [str(_find_executables().herd_inventory_cli), "--update-file", str(processed_cfg), "--summary.cattle_count", str(cattle_count), "--summary.average_weight_kg", str(average_weight_kg), "--summary.average_heat_multiplier", str(average_heat_multiplier), "--summary.cohort_count", str(len(cohorts))]
    for cohort in cohorts:
        cmd.extend(["--cohort", cohort])
    completed = _run_cli(cmd)
    summary, cohort_df = _read_processed_herd_cfg(processed_cfg)
    return {"confirmation": pd.DataFrame([{"stdout": completed.stdout.strip() or "updated"}]), "summary": summary, "cohorts": cohort_df}


def run_csv_mode(ifc_path: str | Path = DEFAULT_IFC, outdoor_csv: str | Path = DEFAULT_OUTDOOR_CSV, controls_csv: str | Path = DEFAULT_CONTROLS_CSV, output_dir: str | Path = REPO_ROOT / "python_runs" / "out_csv", dt_seconds: int = 300, write_report: bool = True) -> pd.DataFrame:
    fn = _api_function("run_simulation")
    if callable(fn):
        try:
            return _normalize_frame_output(fn(ifc_path=str(ifc_path), outdoor_csv=str(outdoor_csv), controls_csv=str(controls_csv), dt_seconds=dt_seconds, output_dir=str(output_dir), write_report=write_report))
        except TypeError:
            pass
    cmd = [str(_find_executables().beef_climate_sim), "--ifc", str(Path(ifc_path)), "--outdoor", str(Path(outdoor_csv)), "--controls", str(Path(controls_csv)), "--output", str(Path(output_dir)), "--dt-seconds", str(dt_seconds)]
    if write_report:
        cmd.append("--write-report")
    _run_cli(cmd)
    return pd.read_csv(Path(output_dir) / "simulation_results.csv")


def run_single_step(ifc_path: str | Path = DEFAULT_IFC, timestamp: str = "2026-01-01T00:00:00", outdoor_temp_c: float = -6.0, outdoor_rh_pct: float = 78.0, outdoor_wind_m_s: float = 3.2, outdoor_solar_w_m2: float = 120.0, outdoor_cloud_okta: float = 6.0, outdoor_rain_mm_day: float = 0.0, ventilation_group_pct: float = 40.0, heating_group_pct: float = 55.0, light_on: bool = True, output_dir: str | Path = REPO_ROOT / "python_runs" / "out_single_step", write_report: bool = True) -> pd.DataFrame:
    fn = _api_function("run_single_step")
    if callable(fn):
        try:
            return _normalize_frame_output(fn(ifc_path=str(ifc_path), timestamp=timestamp, outdoor_temp_c=outdoor_temp_c, outdoor_rh_pct=outdoor_rh_pct, outdoor_wind_m_s=outdoor_wind_m_s, outdoor_solar_w_m2=outdoor_solar_w_m2, outdoor_cloud_okta=outdoor_cloud_okta, outdoor_rain_mm_day=outdoor_rain_mm_day, ventilation_group_pct=ventilation_group_pct, heating_group_pct=heating_group_pct, light_on=light_on, output_dir=str(output_dir), write_report=write_report))
        except TypeError:
            pass
    cmd = [str(_find_executables().beef_climate_sim), "--ifc", str(Path(ifc_path)), "--timestamp", timestamp, "--outdoor-temp-c", str(outdoor_temp_c), "--outdoor-rh-pct", str(outdoor_rh_pct), "--outdoor-wind-m-s", str(outdoor_wind_m_s), "--outdoor-solar-w-m2", str(outdoor_solar_w_m2), "--outdoor-cloud-okta", str(outdoor_cloud_okta), "--outdoor-rain-mm-day", str(outdoor_rain_mm_day), "--ventilation-group", str(ventilation_group_pct), "--heating-group", str(heating_group_pct), "--light-on", "1" if light_on else "0", "--output", str(Path(output_dir))]
    if write_report:
        cmd.append("--write-report")
    _run_cli(cmd)
    return pd.read_csv(Path(output_dir) / "simulation_results.csv")


def run_sensor_seeded_step(config_path: str | Path = DEFAULT_CONFIG, timestamp: str = "2026-01-01T00:00:00", sensor_indoor_temp_c: float = 18.4, sensor_indoor_rh_pct: float = 67.0, sensor_indoor_wind_m_s: float = 0.42, sensor_indoor_co2_ppm: float = 1850.0, sensor_indoor_nh3_ppm: float = 12.0, sensor_indoor_h2o_g_m3: float = 14.8, sensor_indoor_rad_kj_m2_day: float = 220.0, sensor_indoor_okta: float = 8.0, sensor_indoor_aha: float = 1.35, outdoor_temp_c: float = -6.0, outdoor_rh_pct: float = 78.0, outdoor_wind_m_s: float = 3.2, outdoor_solar_w_m2: float = 120.0, outdoor_cloud_okta: float = 6.0, outdoor_rain_mm_day: float = 0.0, ventilation_group_pct: float = 40.0, heating_group_pct: float = 55.0, light_on: bool = True, output_dir: str | Path = REPO_ROOT / "python_runs" / "out_sensor", write_report: bool = True) -> pd.DataFrame:
    cmd = [str(_find_executables().beef_climate_sim), "--config", str(Path(config_path)), "--timestamp", timestamp, "--sensor-indoor-temp-c", str(sensor_indoor_temp_c), "--sensor-indoor-rh-pct", str(sensor_indoor_rh_pct), "--sensor-indoor-wind-m-s", str(sensor_indoor_wind_m_s), "--sensor-indoor-co2-ppm", str(sensor_indoor_co2_ppm), "--sensor-indoor-nh3-ppm", str(sensor_indoor_nh3_ppm), "--sensor-indoor-h2o-g-m3", str(sensor_indoor_h2o_g_m3), "--sensor-indoor-rad-kj-m2-day", str(sensor_indoor_rad_kj_m2_day), "--sensor-indoor-okta", str(sensor_indoor_okta), "--sensor-indoor-aha", str(sensor_indoor_aha), "--outdoor-temp-c", str(outdoor_temp_c), "--outdoor-rh-pct", str(outdoor_rh_pct), "--outdoor-wind-m-s", str(outdoor_wind_m_s), "--outdoor-solar-w-m2", str(outdoor_solar_w_m2), "--outdoor-cloud-okta", str(outdoor_cloud_okta), "--outdoor-rain-mm-day", str(outdoor_rain_mm_day), "--ventilation-group", str(ventilation_group_pct), "--heating-group", str(heating_group_pct), "--light-on", "1" if light_on else "0", "--output", str(Path(output_dir))]
    if write_report:
        cmd.append("--write-report")
    _run_cli(cmd)
    return pd.read_csv(Path(output_dir) / "simulation_results.csv")


def run_forecast_mode(ifc_path: str | Path = DEFAULT_IFC, timestamp: str = "2026-01-01T00:00:00", sensor_indoor_temp_c: float = 18.4, sensor_indoor_rh_pct: float = 67.0, sensor_indoor_wind_m_s: float = 0.42, sensor_indoor_co2_ppm: float = 1850.0, sensor_indoor_nh3_ppm: float = 12.0, sensor_indoor_h2o_g_m3: float = 14.8, outdoor_temp_c: float = -6.0, outdoor_rh_pct: float = 78.0, outdoor_wind_m_s: float = 3.2, outdoor_solar_w_m2: float = 120.0, outdoor_cloud_okta: float = 6.0, outdoor_rain_mm_day: float = 0.0, ventilation_group_pct: float = 40.0, heating_group_pct: float = 55.0, light_on: bool = True, forecast_horizon_seconds: int = 7200, dt_seconds: int = 300, output_dir: str | Path = REPO_ROOT / "python_runs" / "out_forecast", write_report: bool = False) -> pd.DataFrame:
    cmd = [str(_find_executables().beef_climate_sim), "--ifc", str(Path(ifc_path)), "--timestamp", timestamp, "--sensor-indoor-temp-c", str(sensor_indoor_temp_c), "--sensor-indoor-rh-pct", str(sensor_indoor_rh_pct), "--sensor-indoor-wind-m-s", str(sensor_indoor_wind_m_s), "--sensor-indoor-co2-ppm", str(sensor_indoor_co2_ppm), "--sensor-indoor-nh3-ppm", str(sensor_indoor_nh3_ppm), "--sensor-indoor-h2o-g-m3", str(sensor_indoor_h2o_g_m3), "--outdoor-temp-c", str(outdoor_temp_c), "--outdoor-rh-pct", str(outdoor_rh_pct), "--outdoor-wind-m-s", str(outdoor_wind_m_s), "--outdoor-solar-w-m2", str(outdoor_solar_w_m2), "--outdoor-cloud-okta", str(outdoor_cloud_okta), "--outdoor-rain-mm-day", str(outdoor_rain_mm_day), "--ventilation-group", str(ventilation_group_pct), "--heating-group", str(heating_group_pct), "--light-on", "1" if light_on else "0", "--forecast-horizon-seconds", str(forecast_horizon_seconds), "--dt-seconds", str(dt_seconds), "--output", str(Path(output_dir))]
    if write_report:
        cmd.append("--write-report")
    _run_cli(cmd)
    return pd.read_csv(Path(output_dir) / "simulation_results.csv")


def response_stdout_csv_example(ifc_path: str | Path = DEFAULT_IFC, timestamp: str = "2026-01-01T00:00:00") -> pd.DataFrame:
    from io import StringIO
    cmd = [str(_find_executables().beef_climate_sim), "--ifc", str(Path(ifc_path)), "--timestamp", timestamp, "--outdoor-temp-c", "-6", "--outdoor-rh-pct", "78", "--outdoor-wind-m-s", "3.2", "--outdoor-solar-w-m2", "120", "--outdoor-cloud-okta", "6", "--outdoor-rain-mm-day", "0", "--ventilation-group", "40", "--heating-group", "55", "--light-on", "1", "--response-stdout", "--response-format", "csv"]
    completed = _run_cli(cmd)
    return pd.read_csv(StringIO(completed.stdout))


def response_stdout_json_example(ifc_path: str | Path = DEFAULT_IFC, timestamp: str = "2026-01-01T00:00:00") -> pd.DataFrame:
    cmd = [str(_find_executables().beef_climate_sim), "--ifc", str(Path(ifc_path)), "--timestamp", timestamp, "--outdoor-temp-c", "-6", "--outdoor-rh-pct", "78", "--outdoor-wind-m-s", "3.2", "--outdoor-solar-w-m2", "120", "--outdoor-cloud-okta", "6", "--outdoor-rain-mm-day", "0", "--ventilation-group", "40", "--heating-group", "55", "--light-on", "1", "--response-stdout", "--response-format", "json"]
    completed = _run_cli(cmd)
    return _normalize_frame_output(json.loads(completed.stdout))


def _print_frame(df: pd.DataFrame, limit: int = 20) -> None:
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.head(limit).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Python entrypoint for the Cython-based climate_hall_simulator package.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["build-cython", "inspect-ifc", "process-herd", "run-csv", "run-single-step", "run-sensor-step", "run-forecast", "stdout-csv", "stdout-json"]:
        sub.add_parser(name)
    args = parser.parse_args()
    if args.command == "build-cython":
        print(build_cython_extension().stdout)
    elif args.command == "inspect-ifc":
        _print_frame(inspect_ifc_file())
    elif args.command == "process-herd":
        result = process_herd_inventory_file()
        for name, frame in result.items():
            print(f"\n[{name}]")
            _print_frame(frame)
    elif args.command == "run-csv":
        _print_frame(run_csv_mode())
    elif args.command == "run-single-step":
        _print_frame(run_single_step())
    elif args.command == "run-sensor-step":
        _print_frame(run_sensor_seeded_step())
    elif args.command == "run-forecast":
        _print_frame(run_forecast_mode())
    elif args.command == "stdout-csv":
        _print_frame(response_stdout_csv_example())
    elif args.command == "stdout-json":
        _print_frame(response_stdout_json_example())

if __name__ == "__main__":
    main()
