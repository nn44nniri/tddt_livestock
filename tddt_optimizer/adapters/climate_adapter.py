from __future__ import annotations
from pathlib import Path
import sys
import tempfile
import shutil
import time
import pandas as pd


class ClimateAdapter:
    """Thin adapter around climate_simulator/main.py.

    Temporary files are written under a user-configurable directory instead of
    the system default /tmp.  Per-step temporary directories are removed as soon
    as the simulator returns, and a periodic stale-directory cleanup protects
    long TRAIN runs from filling persistent storage.
    """

    def __init__(self, project_root: str | Path, tmp_dir: str | Path | None = None, cleanup_interval_steps: int = 4000):
        self.project_root = Path(project_root)
        self.climate_root = self.project_root / "climate_simulator"
        self.tmp_dir = Path(tmp_dir or "/tmp/tddt_livestock/")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup_interval_steps = max(1, int(cleanup_interval_steps or 4000))
        self._step_counter = 0
        if str(self.climate_root) not in sys.path:
            sys.path.insert(0, str(self.climate_root))
        import main as climate_main
        self.api = climate_main

    def simulate_step(self, state: dict, command: dict, write_report: bool=False, output_dir: str | Path | None=None) -> dict:
        auto_tmp = output_dir is None
        outdir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="tddt_climate_", dir=str(self.tmp_dir)))
        try:
            df = self.api.run_sensor_seeded_step(
                timestamp=str(state.get("timestamp")),
                sensor_indoor_temp_c=float(state.get("indoor_temp_c", 18.0)),
                sensor_indoor_rh_pct=float(state.get("indoor_rh_pct", 60.0)),
                sensor_indoor_wind_m_s=float(state.get("indoor_air_speed_m_s", 0.25)),
                sensor_indoor_co2_ppm=float(state.get("indoor_co2_ppm", 900.0)),
                sensor_indoor_nh3_ppm=float(state.get("indoor_nh3_ppm", 2.0)),
                sensor_indoor_h2o_g_m3=float(state.get("indoor_h2o_g_m3", 8.0)),
                sensor_indoor_rad_kj_m2_day=float(state.get("indoor_rad_kj_m2_day", 200.0)),
                sensor_indoor_okta=float(state.get("indoor_okta", 8.0)),
                sensor_indoor_aha=float(state.get("indoor_aha", 1.3)),
                outdoor_temp_c=float(state.get("outdoor_temp_c", 0.0)),
                outdoor_rh_pct=float(state.get("outdoor_rh_pct", 60.0)),
                outdoor_wind_m_s=float(state.get("outdoor_wind_m_s", 1.0)),
                outdoor_solar_w_m2=float(state.get("outdoor_solar_w_m2", 0.0)),
                outdoor_cloud_okta=float(state.get("outdoor_cloud_okta", 8.0)),
                outdoor_rain_mm_day=float(state.get("outdoor_rain_mm_day", 0.0)),
                ventilation_group_pct=float(command.get("ventilation_group_pct", 40.0)),
                heating_group_pct=float(command.get("heating_group_pct", 0.0)),
                light_on=bool(command.get("light_on", False)),
                output_dir=outdir,
                write_report=write_report,
            )
            row = df.iloc[-1].to_dict()
            row["timestamp"] = str(row.get("timestamp", state.get("timestamp")))
            return row
        finally:
            self._step_counter += 1
            # TRAIN calls pass write_report=False and do not need per-step report files.
            # If an explicit output_dir is passed, the caller owns that directory.
            if auto_tmp and not write_report:
                shutil.rmtree(outdir, ignore_errors=True)
            if self._step_counter % self.cleanup_interval_steps == 0:
                self.cleanup_tmp(max_age_seconds=0)

    def cleanup_tmp(self, max_age_seconds: int = 0) -> int:
        """Remove stale tddt_climate_* directories under tmp_dir.

        max_age_seconds=0 means remove every matching directory immediately.  A
        positive value preserves directories newer than that age.
        """
        removed = 0
        now = time.time()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        for p in self.tmp_dir.glob("tddt_climate_*"):
            if not p.is_dir():
                continue
            try:
                if max_age_seconds and now - p.stat().st_mtime < max_age_seconds:
                    continue
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
            except Exception:
                pass
        return removed
