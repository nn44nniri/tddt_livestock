from __future__ import annotations
from collections.abc import Iterable, Iterator
import pandas as pd
from .funnel_protocol import FunnelBase

class SimulatorFunnel(FunnelBase):
    """Software funnel for TRAIN/WORK-OFFLINE without holding a full episode in RAM.

    The constructor accepts either a DataFrame (legacy behavior) or any iterator of
    row dictionaries. TRAIN passes a CSV chunk iterator so a 1000-day episode can be
    streamed from disk while computed states are flushed to SQLite.
    """
    def __init__(self, rows: pd.DataFrame | Iterable[dict]):
        if isinstance(rows, pd.DataFrame):
            self._rows: Iterator[dict] = iter(rows.to_dict(orient="records"))
        else:
            self._rows = iter(rows)
        self.i = 0
        self.last_indoor = {
            "indoor_temp_c": 18.0, "indoor_rh_pct": 60.0, "indoor_air_speed_m_s": 0.25,
            "indoor_co2_ppm": 900.0, "indoor_nh3_ppm": 2.0, "indoor_h2o_g_m3": 8.0,
            "indoor_rad_kj_m2_day": 200.0, "indoor_okta": 8.0, "indoor_aha": 1.30,
        }

    def read_sensor_packet(self) -> dict:
        try:
            row = next(self._rows)
        except StopIteration:
            raise StopIteration("SimulatorFunnel reached end of prepared climate dataset.")
        self.i += 1
        state = dict(self.last_indoor)
        state.update(row)
        state["timestamp"] = str(row["timestamp"])
        return state

    def update_from_climate_result(self, result: dict) -> None:
        self.last_indoor.update({
            "indoor_temp_c": float(result.get("indoor_temp_c", self.last_indoor["indoor_temp_c"])),
            "indoor_rh_pct": float(result.get("indoor_rh_pct", self.last_indoor["indoor_rh_pct"])),
            "indoor_air_speed_m_s": float(result.get("indoor_air_speed_m_s", result.get("air_speed_m_s", self.last_indoor["indoor_air_speed_m_s"]))),
            "indoor_co2_ppm": float(result.get("indoor_co2_ppm", self.last_indoor["indoor_co2_ppm"])),
            "indoor_nh3_ppm": float(result.get("indoor_nh3_ppm", self.last_indoor["indoor_nh3_ppm"])),
            "indoor_h2o_g_m3": float(result.get("indoor_h2o_g_m3", self.last_indoor["indoor_h2o_g_m3"])),
            "indoor_rad_kj_m2_day": float(result.get("indoor_rad_kj_m2_day", self.last_indoor["indoor_rad_kj_m2_day"])),
            "indoor_okta": float(result.get("indoor_okta", self.last_indoor["indoor_okta"])),
            "indoor_aha": float(result.get("indoor_aha", self.last_indoor["indoor_aha"])),
        })
