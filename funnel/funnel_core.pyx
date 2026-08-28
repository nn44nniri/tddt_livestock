# cython: language_level=3
from libc.stdlib cimport rand

def pack_command(double ventilation_group_pct, double heating_group_pct, bint light_on):
    return {
        "ventilation_group_pct": max(0.0, min(100.0, ventilation_group_pct)),
        "heating_group_pct": max(0.0, min(100.0, heating_group_pct)),
        "light_on": bool(light_on),
    }

def health_check():
    return {"funnel": "cython", "status": "OK"}
