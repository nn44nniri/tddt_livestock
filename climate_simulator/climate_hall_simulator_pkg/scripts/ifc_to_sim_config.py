#!/usr/bin/env python3
"""Extracts a simulator config from an IFC file using IfcOpenShell.

The output is a flat key=value config consumed by the C++ simulator.
This script intentionally keeps the extracted schema small and aligned to the
simulation engine's grey-box parameterization.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def require_ifcopenshell():
    try:
        import ifcopenshell  # type: ignore
        import ifcopenshell.util.element  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "IfcOpenShell is required for IFC extraction. Install it with: pip install ifcopenshell"
        ) from exc
    return ifcopenshell


def first_pset_value(psets: dict, set_name: str, prop: str, default=None):
    return psets.get(set_name, {}).get(prop, default)


def write_cfg(path: Path, rows: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated from IFC with IfcOpenShell\n")
        for key, value in rows.items():
            f.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifc", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ifcopenshell = require_ifcopenshell()
    from ifcopenshell.util.element import get_psets  # type: ignore

    model = ifcopenshell.open(args.ifc)
    project = model.by_type("IfcProject")[0]
    psets = get_psets(project)

    hall = psets.get("Pset_HallInfo", {})
    env = psets.get("Pset_EnvelopeThermalDesign", {})
    loss = psets.get("Pset_EnergyLossAssumptions", {})

    fans = []
    heaters = []
    lights = []
    intakes = []

    for obj in model.by_type("IfcFlowMovingDevice"):
        p = get_psets(obj)
        if "Pset_FanPerformance" in p:
            fans.append(p["Pset_FanPerformance"])
    for obj in model.by_type("IfcSpaceHeater"):
        p = get_psets(obj)
        if "Pset_HeaterPerformance" in p:
            heaters.append(p["Pset_HeaterPerformance"])
    for obj in model.by_type("IfcLightFixture"):
        p = get_psets(obj)
        if "Pset_LightFixtureInfo" in p:
            lights.append(p["Pset_LightFixtureInfo"])
    for obj in model.by_type("IfcWindow"):
        p = get_psets(obj)
        if "Pset_IntakeWindowInfo" in p:
            intakes.append(p["Pset_IntakeWindowInfo"])

    def mean(items, key, default=0.0):
        vals = [float(x.get(key, default)) for x in items if x.get(key) is not None]
        return sum(vals) / len(vals) if vals else default

    rows = {
        "hall.length_m": hall.get("BuildingLength_m", 48.0),
        "hall.width_m": hall.get("BuildingWidth_m", 35.0),
        "hall.eave_height_m": hall.get("EaveHeight_m", 5.2),
        "hall.ridge_height_m": hall.get("RidgeHeight_m", 7.3),
        "hall.volume_m3": hall.get("ApproxEnclosedVolume_m3", 10500.0),
        "hall.pen_count": hall.get("PenCount", 4),
        "hall.cattle_per_pen": hall.get("CattlePerPen", 30),
        "envelope.wall_u_w_m2k": env.get("WallUValue_W_m2K", 0.278),
        "envelope.roof_u_w_m2k": env.get("RoofUValue_W_m2K", 0.222),
        "envelope.door_u_w_m2k": env.get("ServiceDoorUValue_W_m2K", 0.625),
        "envelope.wall_leak_w_m2k": loss.get("WallLeakageCoeff_W_m2K_equiv", 0.5),
        "envelope.roof_leak_w_m2k": loss.get("RoofLeakageCoeff_W_m2K_equiv", 0.4),
        "envelope.window_leak_w_m2k": loss.get("WindowLeakageCoeff_W_m2K_equiv", 6.0),
        "envelope.door_leak_w_m2k": loss.get("DoorLeakageCoeff_W_m2K_equiv", 8.0),
        "actuator.intake_count": len(intakes) or 28,
        "actuator.intake_width_m": mean(intakes, "Width_m", 1.2),
        "actuator.intake_height_m": mean(intakes, "Height_m", 1.0),
        "actuator.intake_discharge_coeff": mean(intakes, "DischargeCoefficient", 0.62),
        "actuator.fan_count": len(fans) or 28,
        "actuator.fan_power_w_each": mean(fans, "ElectricalDemand_W", 370.0),
        "actuator.fan_flow_m3h_each": mean(fans, "FlowAt20Pa_m3h", 15700.0),
        "actuator.heater_count": len(heaters) or 6,
        "actuator.heater_gas_input_kw_each": mean(heaters, "RatedGasInput_kW", 43.96),
        "actuator.heater_useful_kw_each": mean(heaters, "UsefulHeatOutput_kW", 35.17),
        "actuator.heater_airflow_m3h_each": mean(heaters, "PlanningAirflow_m3h", 3704.0),
        "actuator.light_count": len(lights) or 36,
        "actuator.light_power_w_each": mean(lights, "ElectricalLoad_W", 49.0),
        "actuator.light_luminous_flux_lm_each": mean(lights, "LuminousFlux_lm", 6000.0),
        "actuator.light_visible_fraction": 0.10,
        "actuator.light_longwave_fraction": 0.60,
        "cattle.count": int(hall.get("PenCount", 4)) * int(hall.get("CattlePerPen", 30)),
        "cattle.average_weight_kg": 450.0,
        "calibration.theta_ua": 1.0,
        "calibration.theta_cap": 1.0,
        "calibration.theta_vent": 1.0,
        "calibration.theta_cattle": 1.0,
        "calibration.theta_humidity": 1.0,
        "calibration.theta_gas": 1.0,
        "calibration.theta_light": 1.0,
        "calibration.effective_thermal_mass_j_k": 550000000.0,
        "initial.indoor_temp_c": first_pset_value(psets, "Pset_EnergyLossAssumptions", "IndoorDesignTemp_C", 18.0),
        "initial.indoor_rh_pct": 70.0,
        "initial.gas_index": 30.0,
    }

    write_cfg(Path(args.output), rows)


if __name__ == "__main__":
    main()
