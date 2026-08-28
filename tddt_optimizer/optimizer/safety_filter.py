from __future__ import annotations

def apply_safety(command: dict) -> tuple[dict, str]:
    cmd = dict(command)
    cmd["ventilation_group_pct"] = min(100.0, max(0.0, float(cmd.get("ventilation_group_pct", 0.0))))
    cmd["heating_group_pct"] = min(100.0, max(0.0, float(cmd.get("heating_group_pct", 0.0))))
    cmd["light_on"] = bool(cmd.get("light_on", False))
    safety_notes = []
    # Hard anti-waste interlock: high exhaust ventilation and heating must not
    # be requested together. High exhaust already removes heated air rapidly;
    # if ventilation is above 75%, heating is forced off. Lower ventilation can
    # still coexist with heating for minimum hygienic air exchange.
    if cmd["ventilation_group_pct"] > 75.0 and cmd["heating_group_pct"] > 0.0:
        cmd["heating_group_pct"] = 0.0
        safety_notes.append("HEAT_OFF_HIGH_EXHAUST")
    # Hard compatibility rule: if ventilation is requested, the opposite dampers are considered open at same percentage.
    cmd["opposite_damper_group_pct"] = cmd["ventilation_group_pct"]
    return cmd, ";".join(safety_notes) if safety_notes else "OK"
