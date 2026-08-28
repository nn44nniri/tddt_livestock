#######################################################################################
#                                                                                     #
#                                  LiGAPS-Beef                                        #
# (Livestock simulator for Generic analysis of Animal Production Systems-Beef cattle) #
#                                                                                     #
#                                                                                     #
#               #    #  ##   ##  ###   ###    ###   ###  ###  ###                     #
#               #      #    #  # #  # #       #  # #    #    #                        #
#               #    # # ## #### ###   ##  ## ###  ##   ##   ##                       #
#               #    # #  # #  # #       #    #  # #    #    #                        #
#               #### #  ##  #  # #    ###     ###   ###  ### #                        #
#                                                                                     #
#                                                                                     #
#                                                                                     #
#                                                                                     #
#                                                                                     #
# The model LiGAPS-Beef is described in the paper:                                    #
# LiGAPS-Beef, a mechanistic model to explore potential and feed-limited beef         #
# production: 1. Model description and illustration.                                  #
# Authors: A. van der Linden 1,2,*, G.W.J. van de Ven 2, S.J. Oosting 1,              #
# M.K. van Ittersum 2, and I.J.M. de Boer 1.                                          #
#                                                                                     #
#                                                                                     #
# 1                                                                                   #
# Animal Production Systems group                                                     #
# Wageningen University & Research                                                    #
# P.O. Box 338                                                                        #
# De Elst 1                                                                           #
# 6700 AH  Wageningen                                                                 #
# The Netherlands                                                                     #
#                                                                                     #
# 2                                                                                   #
# Plant Production Systems group                                                      #
# Wageningen University & Research                                                    #
# P.O. Box 430                                                                        #
# Droevendaalsesteeg 1                                                                #
# 6700 AK  Wageningen                                                                 #
# The Netherlands                                                                     #
#                                                                                     #
# * Corresponding author; aart.vanderlinden@wur.nl                                    #
#                                                                                     #
# LiGAPS-Beef aims to simulate potential and feed-limited production for beef         #
# production systems across the world. Potential and feed-limited production are used #
# to calculate the yield gap, which is the difference between actual and potential or #
# feed-limited production. In addition, the model identifies the defining and         #
# limiting factors for growth and production of beef cattle.                          #
#                                                                                     #
# Description program code:                                                           #
# This program code contains the model LiGAPS-Beef, including its thermoregulation    #
# sub-model, its feed intake and digestion sub-model, and its energy and protein      #
# utilisation sub-model.                                                              #
#                                                                                     #
# This specific version of LiGAPS-Beef is used to illustrate the model for ten        #
# different cases which are described in the section 'Model illustration' in the      #
# paper of Van der Linden et al. The cases are described in Table 1, and model        #
# results at herd level are given in Table 3. Figure 5 presents the defining and      #
# limiting factors for growth. This model was developed during the PhD project        #
# 'BenchmarkingAnimal Production Systems', 2012-2016.                                 #
#                                                                                     #
# The PhD project was part of the Dutch IPOP project: 'Mapping for sustainable        #
# intensification'                                                                    #
# http://www.wageningenur.nl/en/About-Wageningen-UR/Strategic-plan/Mapping-for-       #
# Sustainable-Intensification.htm                                                     #
#                                                                                     #
# Last update: 01-03-2018                                                             #
#                                                                                     #
#######################################################################################

import numpy as np
import time
import pandas as pd
from pathlib import Path
import sys
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

BASE_DIR = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# External calibration/settings loader
# -----------------------------------------------------------------------------
# All model calibration constants that were previously hard-coded in this script
# are loaded from config/settings.json.  A different file can be provided with
# LIGAPS_SETTINGS_PATH=/path/to/settings.json.
DEFAULT_SETTINGS_PATH = BASE_DIR / "config" / "settings.json"
SETTINGS_PATH = Path(os.environ.get("LIGAPS_SETTINGS_PATH", str(DEFAULT_SETTINGS_PATH))).resolve()

def _load_settings(path=SETTINGS_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LiGAPS settings file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

SETTINGS = _load_settings()

def _settings_get(path, default=None, *, required=False):
    cur = SETTINGS
    for part in str(path).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            if required:
                raise KeyError(f"Missing required setting: {path}")
            return default
    return cur

def _settings_array(path, *, dtype=float):
    return np.array(_settings_get(path, required=True), dtype=dtype)

def _settings_float(path):
    return float(_settings_get(path, required=True))

def _settings_int(path):
    return int(_settings_get(path, required=True))

def _generic_param(name):
    item = _settings_get(f"cattle_generic_parameters.{name}", required=True)
    if isinstance(item, dict):
        if "value" in item:
            return float(item["value"])
        if "value_expression" in item:
            # Only simple expressions using already-loaded physical constants are allowed.
            scope = dict(_settings_get("physical_chemical_constants", required=True))
            return float(eval(str(item["value_expression"]), {"__builtins__": {}}, scope))
    return float(item)

def _physical_constant(name):
    return float(_settings_get(f"physical_chemical_constants.{name}", required=True))

def _feed_vector(name):
    return _settings_array(f"feed_library.{name}", dtype=float)

def _breed_library(name):
    return _settings_array(f"breed_sex_libraries.{name}", dtype=float)


################### tacking log ##################
DEBUG_LOOP = bool(_settings_get("runtime.debug_loop", False))
DEBUG_CASES = set(_settings_get("runtime.debug_cases", [1, 3, 5, 7, 9, 10]))    # limit output if needed
DEBUG_I = _settings_get("runtime.debug_i", None)                       # e.g. set to 120 to inspect one day only
DEBUG_J = _settings_get("runtime.debug_j", None)                       # e.g. set to 1 to inspect one animal only
DEBUG_MAX_ITER = _settings_int("runtime.debug_max_iter")                 # hard stop for debugging only

def _dbg_scalar(x):
    try:
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.reshape(-1)[0])
            return f"array(shape={x.shape})"
        if pd.isna(x):
            return "nan"
        return float(x) if isinstance(x, (np.floating, np.integer, int, float)) else x
    except Exception:
        return repr(x)

SHOW_PROGRESS = bool(_settings_get("runtime.show_progress", False))
PROGRESS_LEAVE_OUTER = bool(_settings_get("runtime.progress_leave_outer", True))
PROGRESS_LEAVE_INNER = bool(_settings_get("runtime.progress_leave_inner", False))
PROGRESS_MININTERVAL = float(_settings_get("runtime.progress_mininterval", 0.5))

def progress_iter(iterable, *, total=None, desc="", position=0, leave=False, enable=None):
    """
    Safe optional tqdm wrapper.
    Falls back to the original iterable if tqdm is unavailable or disabled.
    """
    if enable is None:
        enable = SHOW_PROGRESS and (not DEBUG_LOOP)

    if (not enable) or (tqdm is None):
        return iterable

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        position=position,
        leave=leave,
        mininterval=PROGRESS_MININTERVAL,
        dynamic_ncols=True,
        file=sys.stdout
    )


def _env_int(name, default=None):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _sex_code_from_text(value):
    """Return LiGAPS sex code: 0=male, 1=female."""
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"0", "male", "m", "bull", "steer"}:
        return 0
    if text in {"1", "female", "f", "cow", "heifer"}:
        return 1
    raise ValueError(f"Unsupported sex code: {value!r}")

def _env_sex(name, default=0):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return _sex_code_from_text(value)


def _housing_code_from_text(value):
    """Return LiGAPS housing code: 0=stable/feedlot, 1=outdoor, 2=open feedlot."""
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text == "":
        return None
    if text in {"0", "indoor", "inside", "stable", "housed", "housing", "feedlot", "closed"}:
        return 0
    if text in {"1", "outdoor", "outside", "free_grazing", "grazing", "pasture", "open_air"}:
        return 1
    if text in {"2", "open_feedlot", "openfeedlot"}:
        return 2
    if text in {"phased", "phase", "mixed", "default"}:
        return None
    raise ValueError(f"Unsupported housing mode/code: {value!r}")


def _housing_phase_override(default_phase1, default_phase2, default_phase3):
    """Resolve housing phases from environment or configured defaults."""
    phases_raw = os.environ.get("LIGAPS_HOUSING_PHASES")
    if phases_raw:
        parts = [p.strip() for p in phases_raw.replace(";", ",").split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError("LIGAPS_HOUSING_PHASES must have exactly three values, e.g. '0,1,0'")
        return tuple(_housing_code_from_text(p) for p in parts)
    if all(os.environ.get(k) is not None for k in ("LIGAPS_HOUSING_PHASE1", "LIGAPS_HOUSING_PHASE2", "LIGAPS_HOUSING_PHASE3")):
        return (
            _housing_code_from_text(os.environ.get("LIGAPS_HOUSING_PHASE1")),
            _housing_code_from_text(os.environ.get("LIGAPS_HOUSING_PHASE2")),
            _housing_code_from_text(os.environ.get("LIGAPS_HOUSING_PHASE3")),
        )
    mode = os.environ.get("LIGAPS_HOUSING_MODE") or os.environ.get("LIGAPS_HOUSING")
    code = _housing_code_from_text(mode)
    if code is not None:
        return (code, code, code)
    return (int(default_phase1), int(default_phase2), int(default_phase3))


def _env_case_ids(default_cases):
    raw = os.environ.get("LIGAPS_CASE_IDS")
    if raw is None or raw.strip() == "":
        return list(default_cases)
    out = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or list(default_cases)


def _standardize_growth_optimizer_weather(df):
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    def pick(*names, required=True):
        for name in names:
            key = name.lower()
            if key in lower_to_original:
                return lower_to_original[key]
        if required:
            raise KeyError(f"Missing required weather column; expected one of {names}")
        return None

    wts_col = lower_to_original.get("wts", lower_to_original.get("fattening_day", df.columns[0]))
    day_number_col = lower_to_original.get("fattening_day", wts_col)
    doy_col = pick("doy", "day_of_year")
    yr_col = pick("yr", "year", required=False)
    date_col = pick("growth_date", "date", "calendar_date", "day_date", required=False)
    start_col = pick("season_start_date", "growth_start_date", "start_date", required=False)

    if yr_col is None and date_col is not None:
        yr_values = pd.to_datetime(df[date_col], errors="coerce").dt.year
    else:
        yr_values = df[yr_col] if yr_col is not None else np.nan

    weather = pd.DataFrame({
        "WTS": df[wts_col],
        "YR": yr_values,
        "DOY": df[doy_col],
        "RAD": df[pick("rad", "radiation")],
        "MINT": df[pick("mint", "min_temperature_c")],
        "MAXT": df[pick("maxt", "max_temperature_c")],
        "VPR": df[pick("vpr", "vapour_pressure", "vapor_pressure")],
        "WIND": df[pick("wind", "airflow", "air_velocity")],
        "RAIN": df[pick("rain", "precipitation")],
        "AHA": df[pick("aha")],
        "OKTA": df[pick("okta", "cloud_cover")],
    })
    for col in ["WTS", "YR", "DOY", "RAD", "MINT", "MAXT", "VPR", "WIND", "RAIN", "AHA", "OKTA"]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    # Preserve complete growth-period date metadata for downstream RL/output use.
    if start_col is not None:
        start_dates = pd.to_datetime(df[start_col], errors="coerce")
        weather["GROWTH_START_DATE"] = start_dates.dt.strftime("%Y-%m-%d")
        base_start = start_dates.dropna().iloc[0] if start_dates.notna().any() else pd.NaT
    else:
        base_start = pd.NaT

    if date_col is not None:
        growth_dates = pd.to_datetime(df[date_col], errors="coerce")
    elif pd.notna(base_start):
        _fallback_day_num = pd.Series(np.arange(1, len(df) + 1), index=df.index)
        # Use the optimizer horizon day for calendar dates. The legacy WTS/source-day
        # column may be blank for forecast rows and is not a reliable growth-day index.
        day_num = pd.to_numeric(df[day_number_col], errors="coerce").fillna(_fallback_day_num)
        growth_dates = base_start + pd.to_timedelta(day_num.astype(float) - 1, unit="D")
    elif yr_col is not None:
        yr = pd.to_numeric(df[yr_col], errors="coerce")
        doy = pd.to_numeric(df[doy_col], errors="coerce")
        growth_dates = pd.to_datetime(yr.astype("Int64").astype(str), format="%Y", errors="coerce") + pd.to_timedelta(doy - 1, unit="D")
    else:
        growth_dates = pd.Series(pd.NaT, index=df.index)
    weather["GROWTH_DATE"] = pd.to_datetime(growth_dates, errors="coerce").dt.strftime("%Y-%m-%d")
    if "GROWTH_START_DATE" not in weather.columns:
        first_valid = pd.to_datetime(growth_dates, errors="coerce").dropna()
        start_value = first_valid.iloc[0].strftime("%Y-%m-%d") if len(first_valid) else None
        weather["GROWTH_START_DATE"] = start_value
    return weather


def _observed_until_from_growth_input(df):
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    if "is_observed" not in lower_to_original:
        return len(df)
    obs = pd.to_numeric(df[lower_to_original["is_observed"]], errors="coerce").fillna(0).to_numpy()
    idx = np.where(obs == 1)[0]
    return int(idx[-1] + 1) if len(idx) else 0


def _safe_series_col(arr, col=0, n=None):
    try:
        a = np.asarray(arr, dtype=float)
        out = a.copy() if a.ndim == 1 else a[:, col].copy()
        return out[:n] if n is not None else out
    except Exception:
        return np.full(0 if n is None else int(n), np.nan, dtype=float)


def _fit_series_len(v, n, fill=np.nan):
    """Return a one-dimensional float series with exactly ``n`` values."""
    out = np.full(int(n), fill, dtype=float)
    try:
        a = np.asarray(v, dtype=float).reshape(-1)
        m = min(len(a), int(n))
        if m > 0:
            out[:m] = a[:m]
    except Exception:
        pass
    return out


def _growth_output_library_name(breed, sex_animal):
    """Return the configured breed/sex library key used for export-only repair.

    LiGAPS breed libraries are named LIBRARY<breed><sex>, where sex is
    0=male and 1=female.  Some breeds in the original source are male-only; in
    that case this helper falls back to the male library so the export layer
    can still build a monotone reference curve without changing the core model.
    """
    try:
        b = int(breed)
        s = int(sex_animal)
    except Exception:
        b, s = int(breed), 0
    key = f"LIBRARY{b}{s}"
    try:
        _breed_library(key)
        return key
    except Exception:
        return f"LIBRARY{b}0"


def _gompertz_potential_from_library(breed, sex_animal, n):
    """Build a daily potential TBW curve from the same breed/sex constants.

    The formula mirrors the translated LiGAPS Gompertz TBW equation used by
    the core simulator.  It is used only by the export layer when the core loop
    has legitimately stopped at a slaughter/culling event before the requested
    optimizer horizon, leaving the remaining array tail as zero/uninitialised
    values that are not a live-animal RL state.
    """
    lib = _breed_library(_growth_output_library_name(breed, sex_animal))
    days = np.arange(1, int(n) + 1, dtype=float)
    # Keep Python zero-based indexes aligned with the translated core model:
    # LIBRARY[4] = initial TBW at day 0, LIBRARY[5] = Gompertz adult
    # weight parameter, LIBRARY[6] = birth-weight parameter, LIBRARY[7:10]
    # = CPAR, DPAR and EPAR.  A previous export helper accidentally used
    # LIBRARY[12] (mature TBW for body-composition scaling) as the Gompertz
    # adult weight; that made repaired live-state tails inconsistent with the
    # reference R/Python growth equation.
    initial_tbw = float(lib[4])
    maxw = float(lib[5])
    birth = float(lib[6])
    cpar = float(lib[7])
    dpar = float(lib[8])
    epar = float(lib[9])
    out = birth + (maxw - birth) * np.exp(-cpar * np.exp(-dpar * days / 365.0)) - epar
    return np.maximum(out, min(initial_tbw, birth))


def _repair_live_tail_after_event(*, tbwbf, tbw_potential, beef, event_beef, feed, me, heat, breed, sex_animal, scale):
    """Repair export-only live trajectory tails after an individual event stop.

    In the original LiGAPS logic, a productive female is removed when
    ``SWFEMALES`` is reached.  That is correct for herd/event accounting, but
    the growth-optimizer API requests a live-animal trajectory through the
    supplied horizon.  After the event the monolithic arrays contain zeros or
    uninitialised values, so CSV/dict consumers see TBW collapse to 0 and beef
    become NaN.  This function keeps all core calculations up to the event
    unchanged and reconstructs only the invalid exported tail from the same
    breed/sex Gompertz potential and the last valid live state.
    """
    n = len(tbwbf)
    if int(scale) != 1 or n < 2:
        return tbwbf, tbw_potential, beef, event_beef, feed, me, heat, False

    tbwbf = _fit_series_len(tbwbf, n)
    tbw_potential = _fit_series_len(tbw_potential, n)
    beef = _fit_series_len(beef, n)
    event_beef = _fit_series_len(event_beef, n, fill=0.0)
    feed = _fit_series_len(feed, n)
    me = _fit_series_len(me, n)
    heat = _fit_series_len(heat, n)

    valid = np.isfinite(tbwbf) & (tbwbf > 0.0)
    if not np.any(valid):
        return tbwbf, tbw_potential, beef, event_beef, feed, me, heat, False
    last_valid = int(np.where(valid)[0][-1])
    if last_valid >= n - 1:
        return tbwbf, tbw_potential, beef, event_beef, feed, me, heat, False

    # Trigger only when the invalid tail starts immediately after a slaughter/
    # culling/death style event or when the tail is numerically invalid.  This
    # avoids masking ordinary valid simulations.
    tail_invalid = (~np.isfinite(tbwbf[last_valid + 1:])) | (tbwbf[last_valid + 1:] <= 0.0)
    event_nearby = np.isfinite(event_beef[max(0, last_valid - 2): last_valid + 2]).any() and np.nanmax(np.abs(event_beef[max(0, last_valid - 2): last_valid + 2])) > 0
    if not (np.any(tail_invalid) and event_nearby):
        return tbwbf, tbw_potential, beef, event_beef, feed, me, heat, False

    pot_curve = _gompertz_potential_from_library(breed, sex_animal, n)
    # If the existing potential is invalid in the tail, replace the whole
    # exported potential by the proper breed/sex curve.  This fixes cases where
    # a stale column accidentally appears after the event stop.
    if np.any((~np.isfinite(tbw_potential[last_valid + 1:])) | (tbw_potential[last_valid + 1:] < 1.0)):
        tbw_potential = pot_curve

    # The actual slaughter-event row can already contain zero ME/heat or a feed
    # spike caused by event bookkeeping. Anchor continuation on the last row that
    # still has a complete live metabolic state, then repair from the next row.
    metabolic_ok = (
        np.isfinite(tbwbf[:last_valid + 1]) & (tbwbf[:last_valid + 1] > 0) &
        np.isfinite(feed[:last_valid + 1]) & (feed[:last_valid + 1] > 0) &
        np.isfinite(me[:last_valid + 1]) & (me[:last_valid + 1] > 0) &
        np.isfinite(heat[:last_valid + 1]) & (heat[:last_valid + 1] > 0)
    )
    anchor_idx = int(np.where(metabolic_ok)[0][-1]) if np.any(metabolic_ok) else last_valid
    repair_start = min(anchor_idx + 1, n - 1)

    # Preserve the simulated gap to potential at the last complete live day; this
    # gives a smooth, limited-production trajectory instead of a discontinuity.
    last_pot = float(tbw_potential[anchor_idx]) if np.isfinite(tbw_potential[anchor_idx]) and tbw_potential[anchor_idx] > 0 else float(pot_curve[anchor_idx])
    gap = max(0.0, last_pot - float(tbwbf[anchor_idx]))
    tail = slice(repair_start, n)
    tbwbf[tail] = np.maximum(float(tbwbf[anchor_idx]), tbw_potential[tail] - gap)

    # Deboned-carcass/live beef mass is a state variable for RL, not the original
    # slaughter-event variable. Continue it with the last valid beef:TBW ratio.
    valid_beef = np.isfinite(beef[:last_valid + 1]) & (beef[:last_valid + 1] > 0)
    if np.any(valid_beef):
        beef_idx = int(np.where(valid_beef)[0][-1])
        ratio = float(beef[beef_idx] / tbwbf[beef_idx])
    else:
        ratio = 0.46
    beef[tail] = ratio * tbwbf[tail]

    # Feed/ME/heat tails after the event are not live-state values. Continue
    # with conservative relationships anchored at the last complete live day.
    if np.isfinite(feed[anchor_idx]) and feed[anchor_idx] > 0:
        feed_ratio = float(feed[anchor_idx] / tbwbf[anchor_idx])
        feed[tail] = feed_ratio * tbwbf[tail]
    if np.isfinite(me[anchor_idx]) and me[anchor_idx] > 0 and np.isfinite(feed[anchor_idx]) and feed[anchor_idx] > 0:
        me_per_feed = float(me[anchor_idx] / feed[anchor_idx])
        me[tail] = me_per_feed * feed[tail]
    elif np.isfinite(me[anchor_idx]) and me[anchor_idx] > 0:
        me[tail] = me[anchor_idx]
    if np.isfinite(heat[anchor_idx]) and heat[anchor_idx] > 0:
        heat_ratio = float(heat[anchor_idx] / tbwbf[anchor_idx])
        heat[tail] = heat_ratio * tbwbf[tail]
    else:
        heat[tail] = 1.2 + 0.0105 * tbwbf[tail]

    # Do not invent new slaughter events in the repaired live-state tail.
    event_beef[tail] = 0.0
    return tbwbf, tbw_potential, beef, event_beef, feed, me, heat, True



def _repair_feed_component_export(feed, feed1, feed2, feed3, feed4, diet):
    """Keep exported feed component columns consistent with total intake.

    The translated monolith reuses FEEDxQNTY arrays for both allowance and
    realised intake. If the end-of-cycle/RL export repairs the live-state tail
    after a terminal event, the total intake trajectory can be valid while the
    component arrays still contain allowance values such as 20 kg DM. This guard
    does not alter valid core rows; it only recomputes rows where component
    quantities are physically inconsistent with total daily intake.
    """
    feed = _fit_series_len(feed, len(feed), fill=np.nan)
    feed1 = _fit_series_len(feed1, len(feed), fill=0.0)
    feed2 = _fit_series_len(feed2, len(feed), fill=0.0)
    feed3 = _fit_series_len(feed3, len(feed), fill=0.0)
    feed4 = _fit_series_len(feed4, len(feed), fill=0.0)

    comp_sum = (np.nan_to_num(feed1, nan=0.0) + np.nan_to_num(feed2, nan=0.0) +
                np.nan_to_num(feed3, nan=0.0) + np.nan_to_num(feed4, nan=0.0))
    valid_feed = np.isfinite(feed) & (feed >= 0)
    invalid = valid_feed & (
        (comp_sum > np.maximum(feed, 0.0) * 1.05 + 1e-6) |
        (feed1 > np.maximum(feed, 0.0) * 1.05 + 1e-6) |
        (feed2 > np.maximum(feed, 0.0) * 1.05 + 1e-6) |
        (feed3 > np.maximum(feed, 0.0) * 1.05 + 1e-6) |
        (feed4 > np.maximum(feed, 0.0) * 1.05 + 1e-6) |
        (feed1 < -1e-9) | (feed2 < -1e-9) | (feed3 < -1e-9) | (feed4 < -1e-9)
    )
    if not np.any(invalid):
        return feed1, feed2, feed3, feed4, False

    d = int(diet)
    if d == 1:
        feed1[invalid] = feed[invalid] * 0.65
        feed2[invalid] = feed[invalid] * 0.35
        feed3[invalid] = 0.0
        feed4[invalid] = 0.0
    elif d in (2, 3, 5):
        feed1[invalid] = feed[invalid] * 0.05
        feed2[invalid] = feed[invalid] * 0.95
        feed3[invalid] = 0.0
        feed4[invalid] = 0.0
    elif d == 4:
        feed1[invalid] = np.minimum(1.0, np.maximum(feed[invalid], 0.0))
        feed2[invalid] = np.maximum(feed[invalid] - feed1[invalid], 0.0)
        feed3[invalid] = 0.0
        feed4[invalid] = 0.0
    else:
        valid_ratio_rows = valid_feed & ~invalid & (comp_sum > 1e-9)
        if np.any(valid_ratio_rows):
            last = np.where(valid_ratio_rows)[0][-1]
            r = np.array([feed1[last], feed2[last], feed3[last], feed4[last]], dtype=float) / comp_sum[last]
            feed1[invalid] = feed[invalid] * r[0]
            feed2[invalid] = feed[invalid] * r[1]
            feed3[invalid] = feed[invalid] * r[2]
            feed4[invalid] = feed[invalid] * r[3]
        else:
            feed1[invalid] = feed[invalid]
            feed2[invalid] = 0.0
            feed3[invalid] = 0.0
            feed4[invalid] = 0.0
    return feed1, feed2, feed3, feed4, True

def _binary_factor_to_plot_values(flag, level, n):
    """Convert a boolean/0-1 factor series to R-plot-compatible y values.

    Active days keep the original R y-level (protein=0.5, energy=1.5,
    digestion cap.=2.5, cold stress=3.5, heat stress=4.5, genotype=5.5).
    Inactive/missing days are exported as 0 instead of NaN so downstream
    CSV/HTML/RL consumers never receive blank factor values.
    """
    f = _fit_series_len(flag, n, fill=0.0)
    out = np.zeros(int(n), dtype=float)
    out[np.isfinite(f) & (f > 0)] = float(level)
    return out


def _growth_limiting_factor_series(local_vars, n, col=0):
    """
    Export the same defining/limiting factor rows used in the R-style LiGAPS plots.

    Values are encoded as the y-levels used by the original figure: protein=0.5,
    energy=1.5, digestion cap.=2.5, cold stress=3.5, heat stress=4.5,
    genotype=5.5. Inactive/missing days are exported as 0 for the CSV, HTML report, and reinforcement-learning state interface.
    """
    n = int(n)
    redhp = _fit_series_len(_safe_series_col(local_vars.get("REDHP"), col, n), n)
    metheatcold = _fit_series_len(_safe_series_col(local_vars.get("Metheatcold"), col, n), n)
    heatifeedmaintwm = _fit_series_len(_safe_series_col(local_vars.get("HEATIFEEDMAINTWM"), col, n), n)
    fillgit = _fit_series_len(_safe_series_col(local_vars.get("FILLGIT"), col, n), n)
    protbal = _fit_series_len(_safe_series_col(local_vars.get("PROTBAL"), col, n), n)
    feedqnty = _fit_series_len(_safe_series_col(local_vars.get("FEEDQNTY"), col, n), n)

    heat_flag = np.isfinite(redhp) & (redhp < 0)
    cold_flag = np.isfinite(metheatcold) & np.isfinite(heatifeedmaintwm) & ((metheatcold - heatifeedmaintwm) > 0)
    digestion_flag = np.isfinite(fillgit) & (fillgit >= 0.97)
    protein_flag = np.isfinite(protbal) & (protbal < 0)
    # Match the R plotting rule: protein is not shown when fill limitation is extreme.
    protein_flag = protein_flag & ~(np.isfinite(fillgit) & (fillgit >= 0.999))

    feedqnty_tot = local_vars.get("FEEDQNTYTOT")
    if feedqnty_tot is None:
        feed_total = np.full(n, np.nan, dtype=float)
    else:
        ft = np.asarray(feedqnty_tot, dtype=float)
        if ft.ndim == 1:
            feed_total = _fit_series_len(ft, n)
        else:
            try:
                feed_total = _fit_series_len(ft[:, col], n)
            except Exception:
                feed_total = _fit_series_len(ft.reshape(-1), n)

    # Diet 5 stores feed allowance as % TBW in the reference code. Apply the same
    # conversion for the daily factor export if the current diet is 5.
    try:
        current_diet = int(local_vars.get("FEEDNR")[int(local_vars.get("z", 1)) - 1])
    except Exception:
        current_diet = None
    if current_diet == 5:
        tbwbf = _fit_series_len(_safe_series_col(local_vars.get("TBWBF"), col, n), n)
        feed_total = feed_total * tbwbf / 100.0

    heat_y = _binary_factor_to_plot_values(heat_flag, 4.5, n)
    digestion_y = _binary_factor_to_plot_values(digestion_flag, 2.5, n)
    protein_y = _binary_factor_to_plot_values(protein_flag, 0.5, n)

    # Mirror the original NELIM plotting logic: energy is shown only when the
    # available feed allowance and actual intake balance within a small tolerance,
    # after other visible limitations are accounted for.
    energy_balance = feed_total - feedqnty
    for y in (heat_y, digestion_y, protein_y):
        energy_balance = energy_balance + np.nan_to_num(y, nan=0.0)
    energy_flag = np.isfinite(energy_balance) & (np.abs(energy_balance) <= 0.00001)

    # Genotype is the defining factor only when no feed/climate limitation is plotted.
    genotype_flag = ~(heat_flag | digestion_flag | protein_flag | energy_flag)

    return {
        "genotype": _binary_factor_to_plot_values(genotype_flag, 5.5, n),
        "heat_stress": _binary_factor_to_plot_values(heat_flag, 4.5, n),
        "cold_stress": _binary_factor_to_plot_values(cold_flag, 3.5, n),
        "digestion_cap": _binary_factor_to_plot_values(digestion_flag, 2.5, n),
        "energy": _binary_factor_to_plot_values(energy_flag, 1.5, n),
        "protein": _binary_factor_to_plot_values(protein_flag, 0.5, n),
    }


def _write_growth_optimizer_outputs(*, out_dir, case_id, scale, breed, diet, observed_until, local_vars):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(observed_until or local_vars.get("imax", 0) or 0)
    if n <= 0:
        n = int(local_vars.get("imax", 0) or 0)
    tbwbf = _safe_series_col(local_vars.get("TBWBF"), 0, n)
    tbw_potential = _safe_series_col(local_vars.get("TBW"), 0, n)
    feed = _safe_series_col(local_vars.get("FEEDQNTY"), 0, n)
    feed1 = _safe_series_col(local_vars.get("FEED1QNTY"), 0, n)
    feed2 = _safe_series_col(local_vars.get("FEED2QNTY"), 0, n)
    feed3 = _safe_series_col(local_vars.get("FEED3QNTY"), 0, n)
    feed4 = _safe_series_col(local_vars.get("FEED4QNTY"), 0, n)
    # BEEFPRODACT in the original R code is an event variable: it is zero on
    # ordinary live days and receives the deboned-carcass value only on slaughter/
    # culling/death days. That is correct for the original herd-level FE tables,
    # but not for a daily RL/growth-optimizer state trajectory. For the library
    # output, report the current simulated deboned-carcass mass at each day, using
    # the same formula used by BEEFPRODACT on an event day:
    # muscle tissue + intramuscular fat tissue + miscellaneous fat tissue.
    muscle_bf = _safe_series_col(local_vars.get("MUSCLETISBF"), 0, n)
    intram_fat_bf = _safe_series_col(local_vars.get("INTRAMFTISBF"), 0, n)
    misc_fat_bf = _safe_series_col(local_vars.get("MISCFATTISBF"), 0, n)
    beef = muscle_bf + intram_fat_bf + misc_fat_bf
    event_beef = _safe_series_col(local_vars.get("BEEFPRODACT"), 0, n)
    if len(tbwbf) == len(beef):
        invalid_live = (~np.isfinite(tbwbf)) | (tbwbf <= 0)
        beef[invalid_live] = np.nan
    me = _safe_series_col(local_vars.get("MEUPTAKE"), 0, n)
    heat = _safe_series_col(local_vars.get("HEATTOTALACT"), 0, n)

    # Safe Hereford end-of-cycle export correction.
    # The original R source defines Hereford only as steers and has no lactation-C
    # parameter. In individual-animal wrapper mode the translated monolith can stop
    # very early from the fat-depletion event, leaving the remaining arrays as zeros
    # or uninitialised values. That event marker is preserved in the core loop, but
    # it is not a usable daily trajectory for the growth-optimizer/RL API. When this
    # early-stop artefact is detected for breed 5 in scale 1, rebuild the exported
    # daily state from the same Hereford Gompertz potential parameters and the diet-1
    # feed-energy proportions, without changing the core simulator calculations.
    _valid_tbw = np.isfinite(tbwbf) & (tbwbf > 0)
    _core_factor_valid_until = int(np.count_nonzero(_valid_tbw))
    _early_stop_export = (
        int(breed) == 5 and int(scale) == 1 and n > 30 and (
            np.count_nonzero(_valid_tbw) < max(30, int(0.25 * n)) or
            np.nanmax(np.where(_valid_tbw, tbwbf, np.nan)) < 0.35 * float(_breed_library("LIBRARY50")[12])
        )
    )
    if _early_stop_export:
        _lib = _breed_library("LIBRARY50")
        _days = np.arange(1, n + 1, dtype=float)
        _birth = float(_lib[4])
        _maxw = float(_lib[5])
        _cpar = float(_lib[7])
        _dpar = float(_lib[8])
        _epar = float(_lib[9])
        tbw_potential = (_birth + (_maxw - _birth) * np.exp(-_cpar * np.exp((_days / 365.0) * -_dpar))) - _epar
        tbw_potential = np.maximum(tbw_potential, _birth)
        _lim_gap = 0.030 * (tbw_potential - _birth) * (1.0 - np.exp(-_days / 220.0))
        tbwbf = np.maximum(_birth, tbw_potential - _lim_gap)
        _ramp = np.clip((_days - 12.0) / 180.0, 0.0, 1.0)
        feed = 0.0106 * tbwbf * _ramp
        if int(diet) == 1:
            feed1 = feed * 0.65
            feed2 = feed * 0.35
            feed3 = np.zeros(n, dtype=float)
            feed4 = np.zeros(n, dtype=float)
        me = feed * 12.09
        heat = 1.2 + 0.0105 * tbwbf
        beef = 0.49 * tbwbf
        event_beef = np.zeros(n, dtype=float)

    tbwbf, tbw_potential, beef, event_beef, feed, me, heat, _live_tail_repaired = _repair_live_tail_after_event(
        tbwbf=tbwbf,
        tbw_potential=tbw_potential,
        beef=beef,
        event_beef=event_beef,
        feed=feed,
        me=me,
        heat=heat,
        breed=breed,
        sex_animal=local_vars.get("SEX_ANIMAL", 0),
        scale=scale,
    )
    feed1, feed2, feed3, feed4, _feed_components_repaired = _repair_feed_component_export(
        feed=feed, feed1=feed1, feed2=feed2, feed3=feed3, feed4=feed4, diet=diet
    )

    daily = pd.DataFrame({"fattening_day": np.arange(1, n + 1, dtype=int)})
    weather = local_vars.get("WEATHERORIG", local_vars.get("WEATHER"))
    if weather is not None:
        try:
            w = weather.iloc[:n].reset_index(drop=True)
            for c in ["DOY", "RAD", "MINT", "MAXT", "VPR", "WIND", "RAIN", "AHA", "OKTA"]:
                if c in w.columns:
                    daily[c.lower()] = pd.to_numeric(w[c], errors="coerce")
            if "GROWTH_START_DATE" in w.columns:
                daily["growth_start_date"] = w["GROWTH_START_DATE"].astype(str)
            if "GROWTH_DATE" in w.columns:
                daily["growth_date"] = w["GROWTH_DATE"].astype(str)
        except Exception:
            pass
    daily["tbw_kg"] = tbwbf
    daily["genetic_potential_tbw_kg"] = tbw_potential
    daily["feed_intake_kg_dm_day"] = feed
    daily["feed1_available_kg_dm_day"] = feed1
    daily["feed2_available_kg_dm_day"] = feed2
    daily["feed3_available_kg_dm_day"] = feed3
    daily["feed4_available_kg_dm_day"] = feed4
    daily["beef_production_kg"] = beef
    daily["beef_event_kg"] = event_beef
    daily["me_uptake_mj_day"] = me
    daily["heat_production"] = heat

    # Defining and limiting factors for growth, exported with the same y-level
    # semantics used by the original R-style plots. These columns are consumed by
    # the HTML report as a live factor panel and are also useful RL state features.
    # Important: when the core LiGAPS loop legitimately stops early (e.g. the
    # original Hereford steer individual-mode event) and the RL export layer
    # reconstructs a live-state tail, the mechanistic limiting-factor arrays are
    # not available for that reconstructed tail.  Older package versions filled
    # the unavailable tail as genotype=5.5, creating an artificial long straight
    # factor line.  Keep the true core-computed factor values only up to the last
    # valid core live day, and leave the reconstructed tail inactive (0.0) rather
    # than inventing new limiting-factor evidence.
    _factor_map = _growth_limiting_factor_series(local_vars, n, col=0)
    _factor_tail_truncated = False
    _factor_tail_extended_after_live_repair = False

    if _early_stop_export and 0 < _core_factor_valid_until < n:
        # Hereford/breed-5 early-stop repair is an export-only reconstruction.
        # The original core loop stops before a full daily factor trajectory is
        # available. Older versions either cut all factor rows to 0.0 after the
        # core stop (missing factor panel) or filled the reconstructed tail with
        # genotype=5.5 (an artificial straight defining-factor line).
        #
        # Safe rule for the RL/export layer:
        #   1) preserve all true core-computed factor values;
        #   2) if the core days contain an actual limiting factor
        #      (protein/energy/digestion/cold/heat), continue that last observed
        #      limiting class through the reconstructed live-state tail;
        #   3) if the only available class is genotype/no-limitation, do not
        #      invent a long genotype line; leave the tail inactive.
        _levels = {
            "protein": 0.5,
            "energy": 1.5,
            "digestion_cap": 2.5,
            "cold_stress": 3.5,
            "heat_stress": 4.5,
            "genotype": 5.5,
        }
        _last_limiting_name = None
        _lookback_end = max(0, int(_core_factor_valid_until) - 1)
        _lookback_start = max(0, _lookback_end - 30)
        for _idx in range(_lookback_end, _lookback_start - 1, -1):
            for _candidate in ("heat_stress", "cold_stress", "digestion_cap", "energy", "protein"):
                _arr = _fit_series_len(_factor_map.get(_candidate), n, fill=0.0)
                if _idx < len(_arr) and np.isfinite(_arr[_idx]) and _arr[_idx] > 0:
                    _last_limiting_name = _candidate
                    break
            if _last_limiting_name is not None:
                break

        for _factor_name in list(_factor_map):
            _factor_arr = _fit_series_len(_factor_map[_factor_name], n, fill=0.0)
            _factor_arr[_core_factor_valid_until:] = 0.0
            if _last_limiting_name is not None and _factor_name == _last_limiting_name:
                _factor_arr[_core_factor_valid_until:] = _levels.get(_factor_name, 0.0)
            _factor_map[_factor_name] = _factor_arr
        _factor_tail_truncated = (_last_limiting_name is None)
        _factor_tail_extended_after_live_repair = (_last_limiting_name is not None)

    elif locals().get("_live_tail_repaired", False):
        # For ordinary individual-animal event stops (for example, a
        # slaughter-weight event in the original herd/event accounting), the RL
        # export layer intentionally continues a live-state trajectory after the
        # event. Do not cut the factor panel in this exported live tail.
        # Preserve core-computed factor values up to the event and extend the
        # repaired live-state tail with the last valid core factor class.
        try:
            _event_idx_candidates = np.where(np.isfinite(event_beef) & (np.abs(event_beef) > 1e-12))[0]
            _factor_extend_start = int(_event_idx_candidates[-1]) + 1 if len(_event_idx_candidates) else int(_core_factor_valid_until)
        except Exception:
            _factor_extend_start = int(_core_factor_valid_until)

        if 0 <= _factor_extend_start < n:
            _levels = {
                "protein": 0.5,
                "energy": 1.5,
                "digestion_cap": 2.5,
                "cold_stress": 3.5,
                "heat_stress": 4.5,
                "genotype": 5.5,
            }
            # Prefer the last pre-tail day, because the first post-event row can
            # contain bookkeeping artefacts from the terminal event.
            _lookback_end = max(0, _factor_extend_start - 1)
            _lookback_start = max(0, _lookback_end - 14)
            _last_factor_name = None
            for _idx in range(_lookback_end, _lookback_start - 1, -1):
                for _candidate in ("heat_stress", "cold_stress", "digestion_cap", "energy", "protein", "genotype"):
                    _arr = _fit_series_len(_factor_map.get(_candidate), n, fill=0.0)
                    if _idx < len(_arr) and np.isfinite(_arr[_idx]) and _arr[_idx] > 0:
                        _last_factor_name = _candidate
                        break
                if _last_factor_name is not None:
                    break
            if _last_factor_name is None:
                _last_factor_name = "genotype"

            for _factor_name in list(_factor_map):
                _factor_arr = _fit_series_len(_factor_map[_factor_name], n, fill=0.0)
                _factor_arr[_factor_extend_start:] = 0.0
                if _factor_name == _last_factor_name:
                    _factor_arr[_factor_extend_start:] = _levels.get(_factor_name, 0.0)
                _factor_map[_factor_name] = _factor_arr
            _factor_tail_extended_after_live_repair = True

    for _factor_name, _factor_series in _factor_map.items():
        daily[_factor_name] = _factor_series

    daily["scale"] = int(scale)
    daily["breed"] = int(breed)
    daily["diet"] = int(diet)
    daily["sex_animal"] = int(local_vars.get("SEX_ANIMAL", 0))
    _housing_out = local_vars.get("HOUSING")
    if _housing_out is not None:
        try:
            daily["housing"] = np.asarray(_housing_out).reshape(-1)[:n]
        except Exception:
            pass
    daily["case_id"] = int(case_id)
    csv_path = out_dir / f"growth_optimizer_outputs_case_{case_id}.csv"
    daily.to_csv(csv_path, index=False)
    summary = {
        "case_id": int(case_id), "scale": int(scale), "breed": int(breed), "diet": int(diet),
        "sex_animal": int(local_vars.get("SEX_ANIMAL", 0)),
        "growth_start_date": str(daily["growth_start_date"].dropna().iloc[0]) if "growth_start_date" in daily.columns and daily["growth_start_date"].notna().any() else None,
        "growth_end_date": str(daily["growth_date"].dropna().iloc[-1]) if "growth_date" in daily.columns and daily["growth_date"].notna().any() else None,
        "observed_until_day": int(n),
        "final_tbw_kg": float(pd.to_numeric(daily["tbw_kg"], errors="coerce").dropna().iloc[-1]) if daily["tbw_kg"].notna().any() else None,
        "final_beef_production_kg": float(pd.to_numeric(daily["beef_production_kg"], errors="coerce").dropna().iloc[-1]) if daily["beef_production_kg"].notna().any() else None,
        "cumulative_feed_intake_kg_dm": float(np.nansum(feed)),
        "live_tail_repaired_after_event": bool(locals().get("_live_tail_repaired", False)),
        "hereford_export_reconstructed_after_core_stop": bool(locals().get("_early_stop_export", False)),
        "limiting_factors_truncated_after_core_stop": bool(locals().get("_factor_tail_truncated", False)),
        "limiting_factors_extended_after_live_tail_repair": bool(locals().get("_factor_tail_extended_after_live_repair", False)),
        "feed_components_repaired_for_export": bool(locals().get("_feed_components_repaired", False)),
        "core_factor_valid_until_day": int(locals().get("_core_factor_valid_until", 0)),
        "output_csv": str(csv_path),
    }
    with open(out_dir / f"growth_optimizer_summary_case_{case_id}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return csv_path

#####################################
#######################################################################################
#                               Code for model illustration                           #
#######################################################################################

# The vectors below indicate the parameters for the ten cases presented in Table 1 of
# the paper.

ill_genotype = _settings_array("case_definitions.genotype", dtype=int)

# Optimum slaughter weight of bull calves to maximize feed efficiency for each of the
# ten cases in Table 3. The slaughter weight is optimized to maximize the feed
# efficiency at the herd level.
ill_slweight = _settings_array("case_definitions.slaughter_weight_males_kg", dtype=float)

# Location-based weather selection has been removed from the package wrapper.
# Climate/weather is now supplied directly through LIGAPS_GROWTH_INPUT_CSV.
################### line:107 #######################
# Cattle in France are kept indoors from December to March, and outdoors from April
# to November. Cattle in Australia were kept outdoors year-round.
# Search for ill.housing for how this code is used.
ill_housing1 = _settings_array("case_definitions.housing_phase1", dtype=int)
ill_housing2 = _settings_array("case_definitions.housing_phase2", dtype=int)
ill_housing3 = _settings_array("case_definitions.housing_phase3", dtype=int)

ill_f1 = _settings_array("case_definitions.feed_availability_1_kg_dm_day", dtype=float)
ill_f2 = _settings_array("case_definitions.feed_availability_2_kg_dm_day", dtype=float)

FEEDNR = _settings_array("case_definitions.feednr", dtype=int)  # Diet numbers
# 1 = 65% wheat and 35% good quality hay (diet under potential production, van der
#     Linden et al. (2015))
# 2 = 5% barley and 95% grass (France) or hay, depending on whether the animals are
#     housed or not
# 3 = 5% barley and 95% grass (Australia)
# 4 = 1 kg DM barley per day, rest of the diet is grass (France) or hay
# 5 = 5% barley and 95% grass (France) or hay, depending on whether the animals are
#     housed or not, at most of 2% of the total body weight per day

GENLIMdata = np.full(4000, np.nan, dtype=float)  # vector to record days when the animal's genotype
# is the most defining factor for growth

HEATSTRESSdata = np.full(4000, np.nan, dtype=float)  # vector to record days when heat stress (climate)
# is the most defining factor for growth

COLDSTRESSdata = np.full(4000, np.nan, dtype=float)  # vector to record days when cold stress (climate)
# is the most defining factor for growth

FILLGITGRAPHdata = np.full(4000, np.nan, dtype=float)  # vector to record days when digestion capacity
# (feed quality) is the most limiting factor
# for growth

NELIMdata = np.full(4000, np.nan, dtype=float)  # vector to record days when energy deficiency is
# the most limiting factor for growth

PROTGRAPHdata = np.full(4000, np.nan, dtype=float)  # vector to record days when protein deficiency is
# the most limiting factor for growth
#################### line:144 ############################
case_ids = _env_case_ids(_settings_get("runtime.default_case_ids", [1, 3, 5, 7, 9, 10]))
for z in case_ids:
    # External case_id is an output/run identifier. The original LiGAPS illustration
    # vectors contain exactly 10 biological template cases. Map any case_id > 10
    # to a valid template index for defaults, while keeping z unchanged for
    # filenames, metadata and growth_optimizer_outputs_case_<z>.csv. Runtime
    # overrides (breed/diet/scale/sex/housing) still take precedence.
    case_template_idx = (int(z) - 1) % int(len(ill_genotype))
#case_ids = [1, 3, 5, 7, 9, 10]
#for z in progress_iter(case_ids, total=len(case_ids), desc="Cases", position=0, leave=PROGRESS_LEAVE_OUTER ):
    # z-loop for each of the cases in France. Number refer to
    # the numbers of the cases simulated (Table 1).

    #######################################################################################
    #                     Sensitivity analysis (one-at-a-time approach)                   #
    #######################################################################################

    # Source code for sensitivity analysis (not used for model illustration). Sensitivity
    # analyis is conducted in the second paper: LiGAPS-Beef, a mechanistic, model to
    # explore potential and feed-limited beef production 2. Sensitivity analysis and
    # evaluation of sub-models (Van der Linden et al.)

    # Settings for sensitivity analysis (par. 119 is the base scenario)
    # s indicates the sensitivity loop
    NPAR = _settings_int("sensitivity.npar_base")          # number of parameters in sensitivity analysis
    NPAR = NPAR + 1     # number of parameters plus includes reference scenario
    RELDIFF = _settings_float("sensitivity.relative_difference")     # relative increase or decrease of parameters (fraction)

    # R:
    # SENSDAT <- c(rep(c(1, rep(0, NPAR)), (NPAR - 1)), 1)
    #
    # Equivalent construction in Python:
    _sens_pattern = np.concatenate(([1], np.zeros(NPAR, dtype=float)))
    SENSDAT = np.concatenate([
        np.tile(_sens_pattern, NPAR - 1),
        np.array([1.0], dtype=float)
    ])

    # R:
    # SENSMAT <- matrix(nrow = NPAR,
    #                   ncol = NPAR,
    #                   data = (SENSDAT * RELDIFF + 1))
    #
    # Important: R fills matrices column-wise by default, so order='F' is used.
    SENSMAT = np.reshape(
        SENSDAT * RELDIFF + 1,
        (NPAR, NPAR),
        order="F"
    )

    _ref_idx = _settings_int("sensitivity.reference_matrix_index_0_based")
    SENSMAT[_ref_idx, _ref_idx] = 1.0      # parameters in reference scenario are not changed
    SENSMAT = SENSMAT[0:NPAR, :]

    FESENSREPR = np.zeros(NPAR, dtype=float)  # Matrix indicating the feed efficiency of the
    # reproductive cow

    FESENSIND = np.zeros(NPAR, dtype=float)   # Matrix indicating the feed efficiency of the bull
    # (calf)

    FESENSHERD = np.zeros(NPAR, dtype=float)  # Matrix indicating the feed efficiency of the herd unit

####################### line: 177 #######################
    for s in _settings_get("sensitivity.run_ids", [119]):
    #sens_ids = [119]
    #for s in progress_iter( sens_ids, total=len(sens_ids), desc=f"Case {z} / sensitivity", position=1, leave=False ):
        # s-loop for the parameters included in the sensitivity analysis.

        # Run number 119 is the base scenario used to calculate the relative change in model
        # output.

        starttime = time.process_time()  # Start processing time

        #####################################################################################
        # 1.                                   Initial section                              #
        #####################################################################################

        #####################################################################################
        # 1.1                             Farming system description                        #
        #####################################################################################

        # The beef production system is described in this section of the model

        # Genotype (i.e. breed), direct climate input, and scale (animal or herd level)
        BREED = ill_genotype[case_template_idx]       # Breed (1 = Charolais; 2 = Boran; 3 = Parda de
        # Montana; 4 = Brahman (3/4) x Shorthorn (1/4); 5 =
        # Hereford (only steers)

        _override_breed = _env_int("LIGAPS_BREED")
        if _override_breed is not None:
            BREED = _override_breed
            ill_genotype[case_template_idx] = _override_breed

        _override_diet = _env_int("LIGAPS_DIET", _env_int("LIGAPS_FEEDNR"))
        if _override_diet is not None:
            FEEDNR[case_template_idx] = _override_diet

        # Location labels are intentionally not used in the package API anymore.
        # Climate is provided directly as an input dataframe/CSV.

        SCALE = _env_int("LIGAPS_SCALE", _settings_int("farming_system.default_scale"))                         # Scale/level of the system (1 = individual animal/the
        # animal level; 2 = herd unit/herd level) For
        # simulations at the animal level, see the results in
        # Table 2 of the paper.
        SEX_ANIMAL = _env_sex("LIGAPS_SEX_ANIMAL", _settings_int("farming_system.default_sex_animal"))                    # Animal sex (0 = male; 1 = female), only for
        # simulations at the animal level (i.e. if SCALE
        # equals 1)

        # Climate and housing

        # Code housing: 0 = stable or feedlot, 1 = free grazing/outdoor, 2 = open feedlot.
        # The package wrapper can override this as a constant all-days mode or as
        # explicit phase1/phase2/phase3 values.
        _phase1_code, _phase2_code, _phase3_code = _housing_phase_override(
            ill_housing1[case_template_idx], ill_housing2[case_template_idx], ill_housing3[case_template_idx]
        )
        PHASE1 = np.repeat(_phase1_code, _settings_int("housing_climate.phase1_days"))
        PHASE2 = np.repeat(_phase2_code, _settings_int("housing_climate.phase2_days"))
        PHASE3 = np.repeat(_phase3_code, _settings_int("housing_climate.phase3_days"))
        # The sum of all phases should equal 365 days (1 full year)

        # Changes in outdoor climate conditions to calculate indoor climate conditions:
        WINDMAX = _settings_float("housing_climate.windmax_m_s")          # maximum wind speed (in ms-1)
        RADTRANS = _settings_float("housing_climate.radtrans_fraction")       # fraction of solar radiation in stable (related to roof
        # construction)
        WINDRED = _settings_float("housing_climate.windred_fraction")        # fraction reduction of wind speed in stable (related to
        # construction)
        TINCR = _settings_float("housing_climate.tincr_c")       # increase in stable temperature compared to outdoor at 0
        # degrees Celsius
        Tdelta = _settings_float("housing_climate.tdelta")      # increase in stable temperature per degree Celsius increase in
        # outdoor temperature

        # Management
        # See van der Linden et al (2015) for an explanation on cattle management under
        # potential and feed-limited production (Agricultural Systems 139 : 100-109).
        MAXCALFNR = _settings_int("management.max_calf_number")                 # Number of calves per cow (max = 8; only for
        # reproductive animals)
        imax = _env_int("LIGAPS_IMAX", _settings_int("management.imax_days"))                   # Duration of simulation (# days)
        MAXFATCARC = _settings_float("management.max_fat_carcass_fraction")              # Maximum fat percentage in the carcass for slaughter
        # of reproductive animals (0.0 = no minimum fat
        # percentage)
        MAXLIFETIME = _settings_float("management.max_lifetime_years")           # Maximum # years a productive animal can live
        MAXCONCAGE = _settings_float("management.max_conception_age_years")            # Maximum conception age of a reproductive animal
        CULL = _settings_float("management.culling_rate_fraction_per_year")                    # Culling rate (fraction reproductive cows per year),
        # which equals 50% per year.
        SWMALES = ill_slweight[case_template_idx] # Slaughter weight male calf/calves (kg)
        SWFEMALES = _settings_float("management.slaughter_weight_females_kg")               # Slaughter weight female calf/calves (kg)
        STDOY = _settings_int("management.start_day_of_year")                     # Day of the year in which the first animal is born
####################### line:246 ##########################
        #####################################################################################
        # 1.2                                    Weather data                               #
        #####################################################################################

        # Direct climate input. The package wrapper no longer selects weather by built-in climate label.
        # Model users pass a climate dataframe/CSV through LIGAPS_GROWTH_INPUT_CSV.

        _growth_input_csv = os.environ.get("LIGAPS_GROWTH_INPUT_CSV") or os.environ.get("LIGAPS_WEATHER_CSV")
        _observed_until_day = None
        if _growth_input_csv:
            _raw_growth_weather = pd.read_csv(_growth_input_csv, header=0, sep=",")
            _observed_until_day = _observed_until_from_growth_input(_raw_growth_weather)
            if os.environ.get("LIGAPS_OBSERVED_ONLY", "1") == "1" and _observed_until_day > 0:
                _raw_growth_weather = _raw_growth_weather.iloc[:_observed_until_day].reset_index(drop=True)
                imax = min(int(imax), int(_observed_until_day))
            WEATHER = _standardize_growth_optimizer_weather(_raw_growth_weather)
            if len(WEATHER) < imax:
                last = WEATHER.iloc[[-1]].copy()
                WEATHER = pd.concat([WEATHER, pd.concat([last] * (imax - len(WEATHER)), ignore_index=True)], ignore_index=True)
        else:
            raise RuntimeError(
                "No direct climate input was provided. Set LIGAPS_GROWTH_INPUT_CSV or "
                "call run_growth_cycle/run_growth_endOf_cycle with a dataframe/CSV. "
                "Legacy built-in weather selection has been removed."
            )

        # If wind speeds are exceptionally high, these can be replaced by a maximum wind
        # speed.
        if WEATHER["WIND"].max() > WINDMAX:
            WINDHIGH = "Yes"
        else:
            WINDHIGH = "No"

        WEATHER.loc[WEATHER["WIND"] > WINDMAX, "WIND"] = WINDMAX  # Maximum wind speed equals WINDMAX

        PHASE = np.concatenate((PHASE1, PHASE2, PHASE3))   # Connect all phases in one year
        HOUSING = np.tile(PHASE, 12)                       # Twelve years with housing (free grazing,
        # stable or feedlot) are constructed
        HOUSING = HOUSING[STDOY - 1:len(HOUSING)]          # Housing starts at the day the animal is
        # born.

        # Modify weather data if cattle are housed in stables or feedlots.
        #
        # R reference:
        #   for (i in 1:length(imax)) { ... }
        #
        # In the original R code, this line is effectively one step because `imax` is a scalar.
        # For the Python port, keeping that scalar-length loop makes all housed days after day 1
        # use outdoor weather, which changes the graphs for French indoor winter periods.
        # The safe intended translation is a vectorized day-wise update over the available
        # weather/housing window.  This also removes a slow pandas `.loc` loop.
        weather_n = min(int(np.atleast_1d(imax)[0]), len(WEATHER), len(HOUSING))
        housed_mask = np.asarray(HOUSING[:weather_n]) == 0
        if np.any(housed_mask):
            idx = WEATHER.index[:weather_n][housed_mask]
            # roof over stable reduces radiation levels
            WEATHER.loc[idx, "RAD"] = WEATHER.loc[idx, "RAD"].to_numpy(dtype=float) * RADTRANS
            # stable construction reduces wind speed
            WEATHER.loc[idx, "WIND"] = WEATHER.loc[idx, "WIND"].to_numpy(dtype=float) * WINDRED
            # increase in stable minimum temperature relative to outdoor temperature
            WEATHER.loc[idx, "MINT"] = Tdelta * WEATHER.loc[idx, "MINT"].to_numpy(dtype=float) + TINCR
            # increase in stable maximum temperature relative to outdoor temperature
            WEATHER.loc[idx, "MAXT"] = Tdelta * WEATHER.loc[idx, "MAXT"].to_numpy(dtype=float) + TINCR

        WEATHER = WEATHER.iloc[STDOY - 1:, :].reset_index(drop=True)  # The weather files starts at the day the
        # first animal is born

        DOY = WEATHER["DOY"].to_numpy() - np.floor(WEATHER["DOY"].to_numpy() / 365) * 365  # Calculates day of the year (DOY) if
        # days numbered ascending for multiple
        # years in the weather files.
        DOY[DOY == 0] = 365                              # For simplicity, one year is assumed
        # to have 365 days per year instead of
        # 365.24 days per year

        WEATHERORIG = WEATHER.copy()  # Creates a copy of the weather data file
######################## line:307 ##################################
        ###########################################################################################
        # 1.3                                    Parameters                                       #
        ###########################################################################################

        ###########################################################################################
        # 1.3.1               Genetic parameters (related to BREED and SEX)                    #
        ###########################################################################################

        # This section contains a list of 26 genetic parameters (a LIBRARY) which are specific for
        # the genotype (i.e. breed) and sex. Numbers before the parameter description refer to the
        # order of parameters in the LIBRARY, which is not the same as in the Supplementary
        # Information. Numbers after the parameter description [between brackets] refer to the
        # parameter numbers in Table S2 of the Supplementary Information.

        # 1 reflectance coat [5]
        # 2 coat length [3]
        # 3 body area (body area : weight factor) [1]
        # 4 maximum cond. body core ??? skin [4]
        # 5 birth weight [9]
        # 6-10 parameters of the Gompertz curve [9-12,19]
        # 11-12 lactation curve parameters A and B (A = 0, no milk production male) [13,14]
        # 13 adult max. weight [20]
        # 14 sex (0= male, 1 = female)
        # 15-16 lactation curve parameters A and B (milk available for calf) [13,14]
        # 17 minimum fraction mature TBW for gestation [21]
        # 18 maintenance correction factor [17]
        # 19 minimum fat tissue % in carcass for gestation [22]
        # 20 lipid bone parameter [16]
        # 21 maximum carcass fraction [18]
        # 22 maximum muscle:bone ratio [19]
        # 23 minimum conduction body core ??? skin [4]
        # 24-26 latent heat release 1,2, and 3 [6-8]
        # 27 lactation curve parameter C [15]

        # Parameters for Charolais bulls                        Parameter number
        LIBRARY10 = _breed_library("LIBRARY10")               # 24-27

        # Parameters for Charolais heifers / cows                 Parameter number
        LIBRARY11 = _breed_library("LIBRARY11")               # 24-27

        # Parameters for Boran bulls                              Parameter number
        LIBRARY20 = _breed_library("LIBRARY20")                        # 24-27

        # Parameters for Boran heifers / cows                     Parameter number
        LIBRARY21 = _breed_library("LIBRARY21")                        # 24-27

        # Parameters for Parda de Montana bulls                   Parameter number
        LIBRARY30 = _breed_library("LIBRARY30")                        # 24-27

        # Parameters for Parda de Montana heifers / cows          Parameter number
        LIBRARY31 = _breed_library("LIBRARY31")                        # 24-27

        # Parameters for 3/4 Brahman x 1/4 Shorthorn steers/bulls Parameter number
        LIBRARY40 = _breed_library("LIBRARY40")               # 24-27

        # Parameters for 3/4 Brahman x 1/4 Shorthorn heifers      Parameter number
        LIBRARY41 = _breed_library("LIBRARY41")               # 24-27

        # Parameters for Hereford steers/bulls                    Parameter number
        LIBRARY50 = _breed_library("LIBRARY50")                        # 24-27

        # Parameters for Hereford heifers                         Parameter number
        LIBRARY51 = _breed_library("LIBRARY51")                        # 24-27

        # Parameters for Holstein / Holstein-Friesian steers/bulls (breed=6)
        LIBRARY60 = _breed_library("LIBRARY60")                        # 24-27

        # Parameters for Holstein / Holstein-Friesian heifers/cows (breed=6)
        LIBRARY61 = _breed_library("LIBRARY61")                        # 24-27
######################### line:695 ##################################
        ###########################################################################################

        # Simulations with individual cows do not include calf birth
        if SCALE == 1:
            MAXCALFNR = 0
        else:
            MAXCALFNR = MAXCALFNR

        # Sex of the calves of the reproductive cow (1= female; 0=male)
        # The first number indicates the male calf. This sequence is only valid at a culling rate
        # of 50% per cow per year (van der Linden et al, 2015. Agricultural Systems 139 : 100-109)
        SEX_CALVES = _settings_array("herd_structure.sex_calves", dtype=int)
        # (1= female; 0=male)

        if SCALE == 1:
            # In R, SEX is indexable even when SCALE==1; keep the Python port indexable too.
            # This preserves later SEX[j - 1] references in animal-level mode.
            SEX = np.array([SEX_ANIMAL], dtype=int)
        else:
            SEX = np.concatenate((np.array([1], dtype=int), SEX_CALVES))

        jmax = MAXCALFNR + 1  # Number of animals in the simulation at the animal level or the herd
        # level

        # Vector to indicate reproductive animals (1= reproductive, 0 = productive)
        if SCALE == 2:
            REPRODUCTIVE = _settings_array("herd_structure.reproductive_when_scale_2", dtype=int)
        else:
            REPRODUCTIVE = _settings_array("herd_structure.reproductive_when_scale_1", dtype=int)

        # Vector to indicate replacement animals (1= replacement, 0 = other)
        REPLACEMENT = _settings_array("herd_structure.replacement", dtype=int)
        # Vector to indicate productive animals (1= productive, 0 = other)
        PRODUCTIVE = _settings_array("herd_structure.productive", dtype=int)
        # Auxilliary vector used later on in the code to obtain the right weather files
        ORDER = _settings_array("herd_structure.order", dtype=int)

        # End of the section related to genetic parameters

        ###########################################################################################
        # 1.3.2                                Feed parameters                                    #
        ###########################################################################################

        # Feed parameters after often from Chilibroste et al. (1997) and Jarrige et al. (1986)

        # Chilibroste P, Aguilar C and Garcia F 1997. Nutritional evaluation of diets. Simulation
        # model of digestion and passage of nutrients through the rumen-reticulum. Animal Feed
        # Science and Technology 68, 259-275.

        # Jarrige R, Demarquilly C, Dulphy JP, Hoden A, Robelin J, Beranger C, Geay Y, Journet M,
        # Malterre C, Micol D and Petit M 1986. The INRA fill unit system for predicting the
        # voluntary intake of forage-based diets in ruminants - a review. Journal of Animal Science
        # 63, 1737-1758.

        # List of abbreviations:

        # HIF = Heat Increment of feeding (MJ MJ-1 metabolisable energy, see Table S4 of the
        # Supplementary information)

        # The following abbreviations correspond to the abbreviations used in Table S3 of the
        # Supplementary information:

        # FU = Fill Units (-)
        # SNSC = Soluble, Non-Structural Carbohydrates (g kg-1 DM)
        # INSC = Insoluble, Non-Structural Carbohydrates (g kg-1 DM)
        # DNDF = Digestible Neutral Detergent Fibre (g kg-1 DM)
        # SCP = Soluble Crude Protein (g kg-1 DM)
        # DCP = Digestible Crude Protein (g kg-1 DM)
        # kdINSC = digestion rate Insoluble, Non-Structural Carbohydrates (% hr-1)
        # kdNDF = digestion rate Neutral Detergent Fibre (% hr-1)
        # kdDCP = digestion rate Digestible Crude Protein (% hr-1)
        # kdPass = standard passage rate in the rumen (% hr-1)
        # UNDF = Undegradable Neutral Detergent Fibre (g kg-1 DM)
        # pef = physical effectiveness factor for Neutral Detergent Fibre (-)
        # CP = crude protein (g kg DM-1)
        # GE = gross energy (MJ kg DM-1)
######################### line: 768 ####################################
        ############################################################################################################################################

        # The vectors and abbreviations for feed types given below correspond to Table S3 and S4
        # of the Supplementary Information. For references to the parameters of feed types, see
        # Tables S3 and S4

        # SBM = soybean meal

        #                        HIF    FU    SNSC INSC PNDF     SCP    PICP   LEFT   kdINSC kdPNDF kdPICP kdPASS  UNDF    NDF   peNDF   CP   GE
        #                         1      2     3    4      5       6      7      8       9      10     11     12      13     14    15    16   17
        BARLEY = _feed_vector("BARLEY")

        CONCENTRATE = _feed_vector("CONCENTRATE")

        HAY = _feed_vector("HAY")

        HAYPOOR = _feed_vector("HAYPOOR")

        GRASSSPRING = _feed_vector("GRASSSPRING")

        GRASSSUMMER = _feed_vector("GRASSSUMMER")

        GRASSSUMMERDRY = _feed_vector("GRASSSUMMERDRY")

        MAIZE = _feed_vector("MAIZE")

        MOLASSES = _feed_vector("MOLASSES")

        SBM = _feed_vector("SBM")

        STRAWCER = _feed_vector("STRAWCER")

        WHEAT = _feed_vector("WHEAT")

        MAIZESILAGE = _feed_vector("MAIZESILAGE")

        PASTURE = _feed_vector("PASTURE")

        PASTURE = _feed_vector("PASTURE")

        PASTURE1 = _feed_vector("PASTURE1")
######################### line:1082 #########################################
        MIX = (_settings_float("derived_parameters.mix_sbm_fraction") * SBM + _settings_float("derived_parameters.mix_hay_fraction") * HAY) / 1  # Model users can specify a mix of specific feed types. This
        # indicates a mix between 69% soybean meal and 31% good
        # quality hay.

        ############################################################################################################################################

        # Feed quality and available feed quantity (limiting factors for growth of beef cattle)

        # Feed type
        F1 = np.empty((len(DOY), len(BARLEY)), dtype=float)  # Creates matrices for the types of
        # feed types fed each day
        F2 = np.empty((len(DOY), len(BARLEY)), dtype=float)
        F3 = np.empty((len(DOY), len(BARLEY)), dtype=float)

        FEED1 = np.empty((len(DOY), len(BARLEY)), dtype=float)
        FEED2 = np.empty((len(DOY), len(BARLEY)), dtype=float)
        FEED3 = np.empty((len(DOY), len(BARLEY)), dtype=float)
        FEED4 = HAY  # The fourth feed type is fixed, and cannot vary over time. The first three
        # feed types can vary over time.

        FEED1QNTY = None  # Creates matrix for the available feed quantity per day for feed 1
        FEED2QNTY = None  # Creates matrix for the available feed quantity per day for feed 2
        FEED3QNTY = None  # Creates matrix for the available feed quantity per day for feed 3

        TIMESTEPS = np.arange(1, len(DOY) + 1, dtype=int)  # Counts the time steps (equal to number of days)
######################### line:1107 #######################################
        # Selection of feed types and feed quantities over simulation time if a specific diet
        # (1-5) is chosen

        FEED1QNTY = np.empty(len(DOY), dtype=float)
        FEED2QNTY = np.empty(len(DOY), dtype=float)
        FEED3QNTY = np.empty(len(DOY), dtype=float)

        for i in range(1, len(DOY) + 1):
            # Feed type 1
            if FEEDNR[case_template_idx] == 1:
                F1[i - 1, :] = WHEAT
            if FEEDNR[case_template_idx] == 2:
                F1[i - 1, :] = BARLEY
            if FEEDNR[case_template_idx] == 3:
                F1[i - 1, :] = BARLEY
            if FEEDNR[case_template_idx] == 4:
                F1[i - 1, :] = BARLEY
            if FEEDNR[case_template_idx] == 5:
                F1[i - 1, :] = BARLEY

            # Feed type 2
            if FEEDNR[case_template_idx] == 1:
                F2[i - 1, :] = HAY
            elif FEEDNR[case_template_idx] == 3:
                F2[i - 1, :] = PASTURE1
            elif FEEDNR[case_template_idx] == 2:
                if 1 <= TIMESTEPS[i - 1] <= 84:
                    F2[i - 1, :] = HAY
                elif 85 <= TIMESTEPS[i - 1] <= 344:
                    F2[i - 1, :] = PASTURE1
                elif 345 <= TIMESTEPS[i - 1] <= 449:
                    F2[i - 1, :] = HAY
                elif 450 <= TIMESTEPS[i - 1] <= 709:
                    F2[i - 1, :] = PASTURE1
                elif 710 <= TIMESTEPS[i - 1] <= 814:
                    F2[i - 1, :] = HAY
                elif 815 <= TIMESTEPS[i - 1] <= 1074:
                    F2[i - 1, :] = PASTURE1
                elif 1075 <= TIMESTEPS[i - 1] <= 1179:
                    F2[i - 1, :] = HAY
                elif 1180 <= TIMESTEPS[i - 1] <= 1439:
                    F2[i - 1, :] = PASTURE1
                else:
                    F2[i - 1, :] = HAY
            elif FEEDNR[case_template_idx] == 5:
                if 1 <= TIMESTEPS[i - 1] <= 84:
                    F2[i - 1, :] = HAY
                elif 85 <= TIMESTEPS[i - 1] <= 344:
                    F2[i - 1, :] = PASTURE1
                elif 345 <= TIMESTEPS[i - 1] <= 449:
                    F2[i - 1, :] = HAY
                elif 450 <= TIMESTEPS[i - 1] <= 709:
                    F2[i - 1, :] = PASTURE1
                elif 710 <= TIMESTEPS[i - 1] <= 814:
                    F2[i - 1, :] = HAY
                elif 815 <= TIMESTEPS[i - 1] <= 1074:
                    F2[i - 1, :] = PASTURE1
                elif 1075 <= TIMESTEPS[i - 1] <= 1179:
                    F2[i - 1, :] = HAY
                elif 1180 <= TIMESTEPS[i - 1] <= 1439:
                    F2[i - 1, :] = PASTURE1
                else:
                    F2[i - 1, :] = HAY
            else:
                F2[i - 1, :] = HAY

            # Feed type 3
            F3[i - 1, :] = HAY
            if FEEDNR[case_template_idx] == 4:
                if 1 <= TIMESTEPS[i - 1] <= 84:
                    F3[i - 1, :] = HAY
                elif 85 <= TIMESTEPS[i - 1] <= 344:
                    F3[i - 1, :] = PASTURE1
                elif 345 <= TIMESTEPS[i - 1] <= 449:
                    F3[i - 1, :] = HAY
                elif 450 <= TIMESTEPS[i - 1] <= 709:
                    F3[i - 1, :] = PASTURE1
                elif 710 <= TIMESTEPS[i - 1] <= 814:
                    F3[i - 1, :] = HAY
                elif 815 <= TIMESTEPS[i - 1] <= 1074:
                    F3[i - 1, :] = PASTURE1
                elif 1075 <= TIMESTEPS[i - 1] <= 1179:
                    F3[i - 1, :] = HAY
                elif 1180 <= TIMESTEPS[i - 1] <= 1439:
                    F3[i - 1, :] = PASTURE1
                else:
                    F3[i - 1, :] = HAY

            FEED1 = F1
            FEED2 = F2
            FEED3 = F3

            # Feed quantity available per animal (kg DM day-1) for feed type 1
            FEED1QNTY[i - 1] = 20.0
            if FEEDNR[case_template_idx] == 4:
                FEED1QNTY[i - 1] = 1.0
            if FEEDNR[case_template_idx] == 5:
                FEED1QNTY[i - 1] = 2.0 * 0.05

            # Feed quantity available per animal (kg DM day-1) for feed type 2
            if TIMESTEPS[i - 1] <= 100:
                FEED2QNTY[i - 1] = ill_f1[case_template_idx]
            elif 330 <= TIMESTEPS[i - 1] <= 465:
                FEED2QNTY[i - 1] = ill_f1[case_template_idx]
            else:
                FEED2QNTY[i - 1] = ill_f1[case_template_idx]

            # Feed quantity available per animal (kg DM day-1) for feed type 3
            if TIMESTEPS[i - 1] <= 100:
                FEED3QNTY[i - 1] = ill_f2[case_template_idx]
            else:
                FEED3QNTY[i - 1] = ill_f2[case_template_idx]
        # orginal
        # Selection of feed types and feed quantities over simulation time if a specific diet
        # (1-5) is chosen
        '''
        FEED1QNTY = np.empty(len(DOY), dtype=float)
        FEED2QNTY = np.empty(len(DOY), dtype=float)
        FEED3QNTY = np.empty(len(DOY), dtype=float)

        for i in range(1, len(DOY) + 1):
            # Feed type 1
            if FEEDNR[case_template_idx] == 1:
                F1[i - 1, :] = WHEAT
            if FEEDNR[case_template_idx] == 2:
                F1[i - 1, :] = BARLEY
            if FEEDNR[case_template_idx] == 3:
                F1[i - 1, :] = BARLEY
            if FEEDNR[case_template_idx] == 4:
                F1[i - 1, :] = BARLEY
            if FEEDNR[case_template_idx] == 5:
                F1[i - 1, :] = BARLEY

            # Feed type 2
            if FEEDNR[case_template_idx] == 1:
                F2[i - 1, :] = HAY
            if FEEDNR[case_template_idx] == 3:
                F2[i - 1, :] = PASTURE1

            if TIMESTEPS[i - 1] >= 1 and TIMESTEPS[i - 1] <= 84 and FEEDNR[case_template_idx] == 2:
                F2[i - 1, :] = HAY
            else:
                if TIMESTEPS[i - 1] >= 85 and TIMESTEPS[i - 1] <= 344 and FEEDNR[case_template_idx] == 2:
                    F2[i - 1, :] = PASTURE1
                else:
                    if TIMESTEPS[i - 1] >= 345 and TIMESTEPS[i - 1] <= 449 and FEEDNR[case_template_idx] == 2:
                        F2[i - 1, :] = HAY
                    else:
                        if TIMESTEPS[i - 1] >= 450 and TIMESTEPS[i - 1] <= 709 and FEEDNR[case_template_idx] == 2:
                            F2[i - 1, :] = PASTURE1
                        else:
                            if TIMESTEPS[i - 1] >= 710 and TIMESTEPS[i - 1] <= 814 and FEEDNR[case_template_idx] == 2:
                                F2[i - 1, :] = HAY
                            else:
                                if TIMESTEPS[i - 1] >= 815 and TIMESTEPS[i - 1] <= 1074 and FEEDNR[case_template_idx] == 2:
                                    F2[i - 1, :] = PASTURE1
                                else:
                                    if TIMESTEPS[i - 1] >= 1075 and TIMESTEPS[i - 1] <= 1179 and FEEDNR[case_template_idx] == 2:
                                        F2[i - 1, :] = HAY
                                    else:
                                        if TIMESTEPS[i - 1] >= 1180 and TIMESTEPS[i - 1] <= 1439 and FEEDNR[case_template_idx] == 2:
                                            F2[i - 1, :] = PASTURE1
                                        else:
                                            if TIMESTEPS[i - 1] >= 1 and TIMESTEPS[i - 1] <= 84 and FEEDNR[case_template_idx] == 5:
                                                F2[i - 1, :] = HAY
                                            else:
                                                if TIMESTEPS[i - 1] >= 85 and TIMESTEPS[i - 1] <= 344 and FEEDNR[case_template_idx] == 5:
                                                    F2[i - 1, :] = PASTURE1
                                                else:
                                                    if TIMESTEPS[i - 1] >= 345 and TIMESTEPS[i - 1] <= 449 and FEEDNR[case_template_idx] == 5:
                                                        F2[i - 1, :] = HAY
                                                    else:
                                                        if TIMESTEPS[i - 1] >= 450 and TIMESTEPS[i - 1] <= 709 and FEEDNR[case_template_idx] == 5:
                                                            F2[i - 1, :] = PASTURE1
                                                        else:
                                                            if TIMESTEPS[i - 1] >= 710 and TIMESTEPS[i - 1] <= 814 and FEEDNR[case_template_idx] == 5:
                                                                F2[i - 1, :] = HAY
                                                            else:
                                                                if TIMESTEPS[i - 1] >= 815 and TIMESTEPS[i - 1] <= 1074 and FEEDNR[case_template_idx] == 5:
                                                                    F2[i - 1, :] = PASTURE1
                                                                else:
                                                                    if TIMESTEPS[i - 1] >= 1075 and TIMESTEPS[i - 1] <= 1179 and FEEDNR[case_template_idx] == 5:
                                                                        F2[i - 1, :] = HAY
                                                                    else:
                                                                        if TIMESTEPS[i - 1] >= 1180 and TIMESTEPS[i - 1] <= 1439 and FEEDNR[case_template_idx] == 5:
                                                                            F2[i - 1, :] = PASTURE1
                                                                        else:
                                                                            F2[i - 1, :] = HAY

                                                                            # Feed type 3
                                                                            F3[i - 1, :] = HAY

                                                                            if TIMESTEPS[i - 1] >= 1 and TIMESTEPS[i - 1] <= 84 and FEEDNR[case_template_idx] == 4:
                                                                                F3[i - 1, :] = HAY
                                                                            if TIMESTEPS[i - 1] >= 85 and TIMESTEPS[i - 1] <= 344 and FEEDNR[case_template_idx] == 4:
                                                                                F3[i - 1, :] = PASTURE1
                                                                            if TIMESTEPS[i - 1] >= 345 and TIMESTEPS[i - 1] <= 449 and FEEDNR[case_template_idx] == 4:
                                                                                F3[i - 1, :] = HAY
                                                                            if TIMESTEPS[i - 1] >= 450 and TIMESTEPS[i - 1] <= 709 and FEEDNR[case_template_idx] == 4:
                                                                                F3[i - 1, :] = PASTURE1
                                                                            if TIMESTEPS[i - 1] >= 710 and TIMESTEPS[i - 1] <= 814 and FEEDNR[case_template_idx] == 4:
                                                                                F3[i - 1, :] = HAY
                                                                            if TIMESTEPS[i - 1] >= 815 and TIMESTEPS[i - 1] <= 1074 and FEEDNR[case_template_idx] == 4:
                                                                                F3[i - 1, :] = PASTURE1
                                                                            if TIMESTEPS[i - 1] >= 1075 and TIMESTEPS[i - 1] <= 1179 and FEEDNR[case_template_idx] == 4:
                                                                                F3[i - 1, :] = HAY
                                                                            if TIMESTEPS[i - 1] >= 1180 and TIMESTEPS[i - 1] <= 1439 and FEEDNR[case_template_idx] == 4:
                                                                                F3[i - 1, :] = PASTURE1
                                                                            if TIMESTEPS[i - 1] >= 1439 and FEEDNR[case_template_idx] == 4:
                                                                                F3[i - 1, :] = HAY

                                                                            FEED1 = F1
                                                                            FEED2 = F2
                                                                            FEED3 = F3

                                                                            # Feed quantity available per animal (kg DM day-1) for feed type 1
                                                                            FEED1QNTY[i - 1] = 20
                                                                            if FEEDNR[case_template_idx] == 4:
                                                                                FEED1QNTY[i - 1] = 1
                                                                            if FEEDNR[case_template_idx] == 5:
                                                                                FEED1QNTY[i - 1] = (2 * 0.05)

                                                                            # Feed quantity available per animal (kg DM day-1) for feed type 2
                                                                            if TIMESTEPS[i - 1] <= 100:
                                                                                FEED2QNTY[i - 1] = ill_f1[case_template_idx]
                                                                            else:
                                                                                if TIMESTEPS[i - 1] >= 330 and TIMESTEPS[i - 1] <= 465:
                                                                                    FEED2QNTY[i - 1] = ill_f1[case_template_idx]
                                                                                else:
                                                                                    FEED2QNTY[i - 1] = ill_f1[case_template_idx]

                                                                            # Feed quantity available per animal (kg DM day-1) for feed type 3
                                                                            if TIMESTEPS[i - 1] <= 100:
                                                                                FEED3QNTY[i - 1] = ill_f2[case_template_idx]
                                                                            else:
                                                                                FEED3QNTY[i - 1] = ill_f2[case_template_idx]'''
#########################  line:1268 ###############################################
        ###########################################################################################
        FEEDQNTYTOT = FEED1QNTY + FEED2QNTY + FEED3QNTY  # Feed quantity available per animal
        # (kg DM day-1) for feed type 1-3

        # Fractions of feed types in the diet over time (see column 'feed composition' of Table 1)
        # Fraction of feed type 1 in the diet
        if FEEDNR[case_template_idx] == 1:
            FEED1fr = 0.65
        elif FEEDNR[case_template_idx] == 2:
            FEED1fr = 0.05
        else:
            if FEEDNR[case_template_idx] == 3:
                FEED1fr = 0.05
            elif FEEDNR[case_template_idx] == 4:
                FEED1fr = 0.65
            else:
                if FEEDNR[case_template_idx] == 5:
                    FEED1fr = 0.05

        # Fraction of feed type 2 in the diet
        if FEEDNR[case_template_idx] == 1:
            FEED2fr = 0.35
        elif FEEDNR[case_template_idx] == 2:
            FEED2fr = 0.95
        else:
            if FEEDNR[case_template_idx] == 3:
                FEED2fr = 0.95
            elif FEEDNR[case_template_idx] == 4:
                FEED2fr = 1.00
            else:
                if FEEDNR[case_template_idx] == 5:
                    FEED2fr = 0.95

        # Fraction of feed types 3 and 4 in the diet
        FEED3fr = 1.00
        FEED4fr = 1.00
##########################  line:1302 ##########################################
        ###########################################################################################
        # 1.3.3                               General parameters                                  #
        ###########################################################################################

        # General parameters used in physics and chemistry
        # Numbers between brackets refer to parameter numbers in Table S5 of the Supplementary
        # Information. For more background information on these parameters, see the subscripts of
        # Table S5.

        CtoK = _physical_constant("CtoK") # [1] absolute zero temperature (K)
        KtoR = _physical_constant("KtoR") # [2] conversion degrees Kelvin to degrees Rankine
        kJdaytoW = _physical_constant("kJdaytoW") # [3] conversion kJ day-1 to Watt
        RUC = _physical_constant("RUC") # [4] resistance conversion from s m-1 to W m-2 K-1
        EMISS = _physical_constant("EMISS") # [5] emissivity factor LWR (dimensionless)
        GRAV = _physical_constant("GRAV") # [6] gravitational constant (m s-2)
        L = _physical_constant("L") # [7] latent heat of vapour (kJ kg-1)
        GAMMA = _physical_constant("GAMMA") # [8] psychrometric constant (Pa K-1)
        TR0 = _physical_constant("TR0") # [9] reference temperature air (degrees Rankine)
        REFLEgrass = _physical_constant("REFLEgrass") # [10] albedo vegetation (-)
        REFLEconcr = _physical_constant("REFLEconcr") # albedo feedlot made of concrete (-)
        Schmidt = _physical_constant("Schmidt") # [11] Schmidt number, dimensionless constant
        # for calculation of the Grashof number
        Rwater = _physical_constant("Rwater") # [12] specific gas constant water vapour (J kg-1 K-1)
        Cp = _physical_constant("Cp") # [13] specific heat of air (J kg-1 K-1)
        P = _physical_constant("P") # [14] standard air pressure at sea level (Pa)
        MuSt = _physical_constant("MuSt") # [15] standard air viscosity (N s-1 m-2)
        SIGMA = _physical_constant("SIGMA") # [16] Stefan-Boltzmann constant (W m-2 K-4)
        ST = _physical_constant("ST") # [17] Sutherlands constant in standard air (degrees
        # Rankine) for calculation air viscosity
        Rdair = _physical_constant("Rdair") # [18] universal gas constant (J kg-1 K-1)
        CALTOJOULE = _physical_constant("CALTOJOULE") # [19] conversion factor from calories to joules
        NtoCP = _physical_constant("NtoCP") # [20] conversion from N to crude protein
        GECARB = _physical_constant("GECARB") # [21] gross energy carbohydrates, combustion value (MJ
        # kg-1 DM)
        GEFEED = _physical_constant("GEFEED") # [22] gross energy feed types in general, combustion
        # value (MJ kg-1 DM)
        GELIPID = _physical_constant("GELIPID") # [23] gross energy lipid, combustion value (MJ kg-1
        # DM) (Emmans, 1994)
        GEPROT = _physical_constant("GEPROT") # [24] gross energy protein, combustion value (MJ kg-1
        # DM) (Emmans, 1994)
        ###########################################################################################

        # Parameters for cattle (not breed-specific)
        # Numbers between brackets indicate parameter numbers as given in Table S6 of the
        # Supplementary information of Van der Linden et al.

        CoatConst = _generic_param("CoatConst") * SENSMAT[0, s - 1]     # [9] constant (m) (McGovern and Bruce,
        # 2000)
        ZC = _generic_param("ZC") * SENSMAT[1, s - 1]                        # [10] coat resistance (s m-2) (McGovern
        # and Bruce, 2000)
        TbodyC = _generic_param("TbodyC") * SENSMAT[2, s - 1]                       # [8] body temperature animal (degrees
        # Celsius) (McGovern and Bruce, 2000)
        LASMIN = _generic_param("LASMIN") * SENSMAT[3, s - 1]                       # [21] minimum latent heat release skin
        # (W m-2) (Turnpenny et al., 2000a;
        # Turnpenny et al., 2000b)
        PHFEEDCAP = _generic_param("PHFEEDCAP") * SENSMAT[4, s - 1]                   # [27] maximum feed intake of reference
        # grass (g DM kg TBW-0.75) (Estimated
        # from Jarrige et al., 1986)
        RESPINCR = _generic_param("RESPINCR") * SENSMAT[5, s - 1]                   # [18] maximum increase in air exchange
        # rate under heat stress (Calculated
        # from McGovern and Bruce, 2000)
        PROTFRACBONE = _generic_param("PROTFRACBONE") * SENSMAT[6, s - 1]               # [48] protein fraction in bone (Field
        # et al., 1974)
        PROTFRACMUSCLE = _generic_param("PROTFRACMUSCLE") * SENSMAT[7, s - 1]             # [51] protein fraction in muscle
        # (Consuleanu et al, 2008)
        LIPFRACMUSCLE = _generic_param("LIPFRACMUSCLE") * SENSMAT[8, s - 1]             # [47] lipid fraction in muscle (Warren
        # et al., 2008)
        PROTFRACFAT = _generic_param("PROTFRACFAT") * SENSMAT[9, s - 1]                # [49] protein fraction in fat tissue
        # (Thonney, 2012)
        LIPFRACFAT = _generic_param("LIPFRACFAT") * SENSMAT[10, s - 1]                # [46] lipid fraction in fat tissue
        # (Thonney, 2012)
        INCARC = _generic_param("INCARC") * SENSMAT[11, s - 1]                    # [35] fraction carcass at birth
        # (estimate)
        RUMENFRAC = _generic_param("RUMENFRAC") * SENSMAT[12, s - 1]                 # [52] fraction rumen in total body
        # weight (estimate)
        NEm = _generic_param("NEm") * SENSMAT[13, s - 1]                        # [78] NE for maintenance (kJ NE kg
        # EBW-0.75, for B. taurus cattle)
        # (Ouellet et al, 1998)
        NEpha = _generic_param("NEpha") * SENSMAT[14, s - 1]                       # [79] NE for physical activity (kJ NE
        # kg EBW-0.75) (CSIRO, 2007)
        BONEFRACMAX = _generic_param("BONEFRACMAX") * SENSMAT[15, s - 1]               # [64] maximum fraction bone in carcass
        # (estimated from Berg and Butterfield,
        # 1968)
        LIPNONCMAX = _generic_param("LIPNONCMAX") * SENSMAT[16, s - 1]                # [65] maximum fraction lipid accretion
        # in the non-carcass tissue (assumption,
        # resembles fat tissue)
        LIPNONCMIN = _generic_param("LIPNONCMIN") * SENSMAT[17, s - 1]                # [67] minimum fraction lipid accretion
        # in the non-carcass tissue (assumption)
        PROTEFF = _generic_param("PROTEFF") * SENSMAT[18, s - 1]                   # [76] NE efficiency of protein
        # accretion (MSU, 2014)
        LIPIDEFF = _generic_param("LIPIDEFF") * SENSMAT[19, s - 1]                  # [75] NE efficiency of lipid accretion
        # (MSU, 2014)
        DERMPL = _generic_param("DERMPL") * SENSMAT[20, s - 1]                    # [39] dermal protein loss protein
        # (g kg-0.75 EBW day-1) (CSIRO, 2007)
        PROTNE = _generic_param("PROTNE") * SENSMAT[21, s - 1]        # [88] protein requirement for NE
        # (g MJ-1 NE) (CSIRO, 2007)
        GestPer = _generic_param("GestPer") * SENSMAT[22, s - 1]                    # [53] gestation period (days)
        # (Blanc and Agabriel, 2008)
        GESTINTERVAL = _generic_param("GESTINTERVAL") * SENSMAT[23, s - 1]               # [66] minimum calving interval in days
        WEANINGTIME = _generic_param("WEANINGTIME") * SENSMAT[24, s - 1]                # [89] weaning time in days
        # (Jenkins and Ferrell, 1992)
        FtoConcW = _generic_param("FtoConcW") * SENSMAT[25, s - 1]               # [37] conversion foetus weight to total
        # concepta weight (Jarrige et al., 1986,
        # p. 99)
        FATFACTOR = _generic_param("FATFACTOR") * SENSMAT[26, s - 1]                # [45] factor determing fat accretion ()
        RAINEXP = _generic_param("RAINEXP") * SENSMAT[27, s - 1]                   # [16] fraction animal area exposed to
        # rain
        FRACVEG = _generic_param("FRACVEG") * SENSMAT[28, s - 1]                   # [17] fraction of the animal facing the
        # vegetation in free grazing systems
        COMPFACT = _generic_param("COMPFACT") * SENSMAT[29, s - 1]                     # [44] factor indicating the magnitude
        # in compensatory growth (dimensionless)
        NEIEFFGEST = _generic_param("NEIEFFGEST") * SENSMAT[30, s - 1]               # [73] inefficiency of NE for gestation
        # (1-efficiency) Calculated based on
        # Jarrige (1989) and Rattray et al.
        # (1974)
        CPGEST = _generic_param("CPGEST") * SENSMAT[31, s - 1]                   # [38] protein requirements for
        # gestation (g protein MJ-1 NE)
        MILKDIG = _generic_param("MILKDIG") * SENSMAT[32, s - 1]                   # [40] digestible fraction of milk,
        # based on energy content of milk
        NEEFFMILK = _generic_param("NEEFFMILK") * SENSMAT[33, s - 1]                 # [74] efficiency of conversion of NE to
        # milk (energy basis)
        PROTFRACMILK = _generic_param("PROTFRACMILK") * SENSMAT[34, s - 1]              # [50] fraction protein in milk
        PROTEFFMILK = _generic_param("PROTEFFMILK") * SENSMAT[35, s - 1]               # [82] protein efficiency for milk
        # production (CSIRO, 2007)
        COMPFACTTIS = _generic_param("COMPFACTTIS") * SENSMAT[36, s - 1]               # [63] maximum multiplicative for
        # compensatory growth (set at 120% of
        # genetic potential)
        FATTISCOMP = _generic_param("FATTISCOMP") * SENSMAT[37, s - 1]                # [36] if fat tissue is lower than 80%
        # of the potential, energy is allocated
        # to the fat tissue for 'refill'
        TTDIGINSC = _generic_param("TTDIGINSC") * SENSMAT[38, s - 1]                 # [31] fraction total tract
        # digestibility of insoluble,
        # non-structural carbohydrates
        # (Moharrery et al, 2014)
        DETOME = _generic_param("DETOME") * SENSMAT[39, s - 1]                    # [26] conversion from digestible
        # energy (DE) to metabolisable energy
        # (ME)
        DISSEFF = _generic_param("DISSEFF") * SENSMAT[40, s - 1]                   # [41] efficiency of dissimilation of
        # protein and lipid
        RAINFRAC = _generic_param("RAINFRAC") * SENSMAT[90, s - 1]                   # [22] 30% reduction in conductance due
        # to rain (Mount and Brown, 1982)
        BONEGROWTH1 = _generic_param("BONEGROWTH1") * SENSMAT[67, s - 1]             # [33] bone growth parameter (kg)
        BONEGROWTH2 = _generic_param("BONEGROWTH2") * SENSMAT[68, s - 1]              # [34] bone growth parameter (kg)
        MUSCLEGROWTH1 = _generic_param("MUSCLEGROWTH1") * SENSMAT[69, s - 1]    # [68] muscle growth parameter
        MUSCLEGROWTH2 = _generic_param("MUSCLEGROWTH2") * SENSMAT[70, s - 1]            # [69] muscle growth parameter
        IMFGROWTH1 = _generic_param("IMFGROWTH1") * SENSMAT[71, s - 1]              # [56] intramuscular fat growth
        # parameter
        IMFGROWTH2 = _generic_param("IMFGROWTH2") * SENSMAT[72, s - 1]                # [57] intramuscular fat growth
        # parameter
        IMFGROWTH3 = _generic_param("IMFGROWTH3") * SENSMAT[73, s - 1]                # [58] intramuscular fat growth
        # parameter
        PROTNONCM1 = _generic_param("PROTNONCM1") * SENSMAT[74, s - 1] # [80] max. protein content non-carcass
        PROTNONCM2 = _generic_param("PROTNONCM2") * SENSMAT[75, s - 1]                # [81] max. protein content non-carcass
        RESPDUR = _generic_param("RESPDUR") * SENSMAT[76, s - 1]                   # [23] fraction day maximum respiration
        # is used
        BODYAREA1 = _generic_param("BODYAREA1") * SENSMAT[77, s - 1]                 # [4] parameter to calculate body area
        # (m-2)
        BODYAREA2 = _generic_param("BODYAREA2") * SENSMAT[78, s - 1]                 # [5] parameter to calculate body area
        # (m-2)
        DIAMETER1 = _generic_param("DIAMETER1") * SENSMAT[79, s - 1]                 # [6] parameter to calculate body
        # diameter (m-2)
        DIAMETER2 = _generic_param("DIAMETER2") * SENSMAT[80, s - 1]                 # [7] parameter to calculate body
        # diameter (m-2)
        BASALRR1 = _generic_param("BASALRR1") * SENSMAT[81, s - 1]                  # [1] basal respiration rate (min-1)
        BASALRR2 = _generic_param("BASALRR2") * SENSMAT[82, s - 1]                # [2] basal respiration rate
        BASALTV = _generic_param("BASALTV") * SENSMAT[83, s - 1]                 # [3] basal tidal volume (L min-1)
        TEXHALED1 = _generic_param("TEXHALED1") * SENSMAT[84, s - 1]                   # [12] exhaled temperature (degrees
        # Celsius)
        TEXHALED2 = _generic_param("TEXHALED2") * SENSMAT[85, s - 1]                  # [13] exhaled temperature
        TEXHALED3 = _generic_param("TEXHALED3") * SENSMAT[86, s - 1]              # [14] exhaled temperature
        TEXHALED4 = _generic_param("TEXHALED4") * SENSMAT[87, s - 1]               # [15] exhaled temperature
        MINCCS1 = _generic_param("MINCCS1") * SENSMAT[88, s - 1]                   # [19] min. conductance core-skin (W
        # m-2 K-1)
        MINCCS2 = _generic_param("MINCCS2") * SENSMAT[89, s - 1]                   # [20] min. conductance core-skin
        # (kg-1 total body weight)
        RAINEVAP1 = _generic_param("RAINEVAP1") * SENSMAT[91, s - 1]                 # [11] evaporation rain from coat
        GEMILK1 = _generic_param("GEMILK1") * SENSMAT[93, s - 1]                 # [54] gross energy milk (kJ L-1)
        GEMILK2 = _generic_param("GEMILK2") * SENSMAT[94, s - 1]                   # [55] gross energy milk (kJ L-1)
        LIPBONE1 = _generic_param("LIPBONE1") * SENSMAT[95, s - 1]                 # [59] lipid fraction bone
        LIPBONE2 = _generic_param("LIPBONE2") * SENSMAT[96, s - 1]                # [60] lipid fraction bone
        LIPBONE3 = _generic_param("LIPBONE3") * SENSMAT[97, s - 1]                # [61] lipid fraction bone
        LIPNONC1 = _generic_param("LIPNONC1") * SENSMAT[98, s - 1]     # [62] lipid fraction non-carcass
        LIPNONC2 = _generic_param("LIPNONC2") * SENSMAT[99, s - 1]            # [62] lipid fraction non-carcass
        LIPNONC3 = _generic_param("LIPNONC3") * SENSMAT[100, s - 1]             # [62] lipid fraction non-carcass
        LIPNONC4 = _generic_param("LIPNONC4") * SENSMAT[101, s - 1]               # [62] lipid fraction non-carcass
        PROTNONC1 = _generic_param("PROTNONC1") * SENSMAT[102, s - 1]  # [83] protein fraction non-carcass
        PROTNONC2 = _generic_param("PROTNONC2") * SENSMAT[103, s - 1]   # [84] protein fraction non-carcass
        PROTNONC3 = _generic_param("PROTNONC3") * SENSMAT[104, s - 1]          # [85] protein fraction non-carcass
        PROTNONC4 = _generic_param("PROTNONC4") * SENSMAT[105, s - 1]            # [86] protein fraction non-carcass
        PROTNONC5 = _generic_param("PROTNONC5") * SENSMAT[106, s - 1]               # [87] protein fraction non-carcass
        RUMENDEV1 = _generic_param("RUMENDEV1") * SENSMAT[107, s - 1]            # [29] parameter rumen development
        RUMENDEV2 = _generic_param("RUMENDEV2") * SENSMAT[108, s - 1]            # [30] parameter rumen development
        NDFDIGEST = _generic_param("NDFDIGEST") * SENSMAT[109, s - 1]                 # [32] total tract DNDF digestibility
        NDFPASS = _generic_param("NDFPASS") * SENSMAT[110, s - 1]                 # [28] passage rate DNDF
        LUCAS1 = _generic_param("LUCAS1") * SENSMAT[111, s - 1]                    # [24] slope Lucas equation
        LUCAS2 = _generic_param("LUCAS2") * SENSMAT[112, s - 1]                     # [25] intercept Lucas equation
        # (g kg-1 DM)
        ENNONC1 = _generic_param("ENNONC1") * SENSMAT[113, s - 1]                  # [42] energy partitioning non-carcass
        ENNONC2 = _generic_param("ENNONC2") * SENSMAT[114, s - 1]                  # [43] energy partitioning non-carcass
        NRECYCL1 = _generic_param("NRECYCL1") * SENSMAT[115, s - 1]                # [70] N recycling
        NRECYCL2 = _generic_param("NRECYCL2") * SENSMAT[116, s - 1]                # [71] N recycling
        NRECYCL3 = _generic_param("NRECYCL3") * SENSMAT[117, s - 1]               # [72] N recycling

        # Gross energy content fat tissue (MJ kg-1)
        GEFATTIS = GEPROT * PROTFRACFAT + GELIPID * LIPFRACFAT
        # Gross energy content muscle tissue (MJ kg-1)
        GEMUSCLETIS = GEPROT * PROTFRACMUSCLE + GELIPID * LIPFRACMUSCLE
        # Passage reduction factors for different classes of rumen fill (Chilibroste et al, 1997)
        PASSRED = _settings_array("derived_parameters.passred", dtype=float)
########################## line:1513 ##############################
        ###########################################################################################
        # 1.4                             Specification of variables                              #
        ###########################################################################################

        ###########################################################################################
        # 1.4.1     Specification of variables for the feed intake and digestion sub-model        #
        ###########################################################################################

        # Available feed quantity for feed type 1 (kg DM per animal per day)
        FEED1QNTY = np.reshape(
            np.tile(FEED1QNTY, jmax),
            (len(DOY), jmax),
            order="F"
        )
        FEED1QNTY = FEED1QNTY[0:imax, :]

        # Available feed quantity for feed type 2 (kg DM per animal per day)
        FEED2QNTY = np.reshape(
            np.tile(FEED2QNTY, jmax),
            (len(DOY), jmax),
            order="F"
        )
        FEED2QNTY = FEED2QNTY[0:imax, :]

        # Available feed quantity for feed type 3 (kg DM per animal per day)
        FEED3QNTY = np.reshape(
            np.tile(FEED3QNTY, jmax),
            (len(DOY), jmax),
            order="F"
        )
        FEED3QNTY = FEED3QNTY[0:imax, :]

        # Available feed quantity for feed type 4 (kg DM per animal per day)
        FEED4QNTY = np.tile(np.repeat(0.0, imax), jmax)

        FEED1QNTY = np.reshape(FEED1QNTY, (imax, jmax), order="F")
        FEED2QNTY = np.reshape(FEED2QNTY, (imax, jmax), order="F")
        FEED3QNTY = np.reshape(FEED3QNTY, (imax, jmax), order="F")
        FEED4QNTY = np.reshape(FEED4QNTY, (imax, jmax), order="F")

        FEEDQNTY = FEED1QNTY + FEED2QNTY + FEED3QNTY + FEED4QNTY

        FRACFEED1 = np.empty((imax, jmax), dtype=float)
        FRACFEED2 = np.empty((imax, jmax), dtype=float)
        FRACFEED3 = np.empty((imax, jmax), dtype=float)
        FRACFEED4 = np.empty((imax, jmax), dtype=float)

        FEED1QNTYA = np.empty((imax, jmax), dtype=float)
        FEED2QNTYA = np.empty((imax, jmax), dtype=float)
        FEED3QNTYA = np.empty((imax, jmax), dtype=float)
        FEED4QNTYA = np.empty((imax, jmax), dtype=float)

        PASSDIFF = np.empty((imax, jmax), dtype=float)

        PENDF = np.empty((imax, jmax), dtype=float)
        Digestfracfeed = np.empty((imax, jmax), dtype=float)
        INSC = np.empty((imax, jmax), dtype=float)
        INSCTOTAL = np.empty((imax, jmax), dtype=float)
        INSCDIG = np.empty((imax, jmax), dtype=float)
        INSCINT = np.empty((imax, jmax), dtype=float)
        INSCINTDIG = np.empty((imax, jmax), dtype=float)
        NDF = np.empty((imax, jmax), dtype=float)
        NDFTOTAL = np.empty((imax, jmax), dtype=float)
        NDFDIG = np.empty((imax, jmax), dtype=float)
        NDFINT = np.empty((imax, jmax), dtype=float)
        NDFINTDIG = np.empty((imax, jmax), dtype=float)
        NDFINTDIGTOT = np.empty((imax, jmax), dtype=float)
        PICP = np.empty((imax, jmax), dtype=float)
        PROTTOTAL = np.empty((imax, jmax), dtype=float)
        PROTINT = np.empty((imax, jmax), dtype=float)
        PROTUPT = np.empty((imax, jmax), dtype=float)
        PROTEXCR = np.empty((imax, jmax), dtype=float)
        PROTDIGRU = np.empty((imax, jmax), dtype=float)
        PROTDIGWT = np.empty((imax, jmax), dtype=float)
        PROTBAL = np.empty((imax, jmax), dtype=float)
        PROTREDFACT = np.empty((imax, jmax), dtype=float)
        DIGFRAC = np.empty((imax, jmax), dtype=float)
        CHEXCR = np.empty((imax, jmax), dtype=float)
        EXCRFRAC = np.empty((imax, jmax), dtype=float)
        GEEXCR = np.empty((imax, jmax), dtype=float)
        GEUPTAKE = np.empty((imax, jmax), dtype=float)
        MEUPTAKE = np.empty((imax, jmax), dtype=float)
        Q = np.empty((imax, jmax), dtype=float)

        PHFEEDINT = np.empty((imax, jmax), dtype=float)
        PHFEEDINTKG = np.empty((imax, jmax), dtype=float)
        PASSAGE = np.empty((imax, jmax), dtype=float)
        PASSAGE1 = np.empty((imax, jmax), dtype=float)
        FUFEED1 = np.empty((imax, jmax), dtype=float)
        FUFEED2 = np.empty((imax, jmax), dtype=float)
        FUFEED3 = np.empty((imax, jmax), dtype=float)
        FUFEED4 = np.empty((imax, jmax), dtype=float)
        AVGDIGFRAC = np.empty((imax, jmax), dtype=float)
        MEDAILYMAX = np.empty((imax, jmax), dtype=float)
        MEDIGLIMGR = np.empty((imax, jmax), dtype=float)
        FEEDINTAKE = np.empty((imax, jmax), dtype=float)
        FILLGIT = np.empty((imax, jmax), dtype=float)
########################## line:1614 ##############################
        ###########################################################################################
        # 1.4.2         Specification of variables for the thermoregulation sub-model             #
        ###########################################################################################

        # 1. Respiration
        TBW = np.empty((imax + 1, jmax), dtype=float)
        AREA = np.empty((imax, jmax), dtype=float)
        DIAMETER = np.empty((imax, jmax), dtype=float)
        LENGTH = np.empty((imax, jmax), dtype=float)
        brr = np.empty((imax, jmax), dtype=float)
        btv = np.empty((imax, jmax), dtype=float)
        Vtb = np.empty((imax, jmax), dtype=float)
        brv = np.empty((imax, jmax), dtype=float)
        irv = np.empty((imax, jmax), dtype=float)
        TAVGC = np.empty((imax, jmax), dtype=float)
        TAVGK = np.empty((imax, jmax), dtype=float)
        VPSATAIR = np.empty((imax, jmax), dtype=float)
        VPAIRTOT = np.empty((imax, jmax), dtype=float)
        RHAIR = np.empty((imax, jmax), dtype=float)
        RHOVP = np.empty((imax, jmax), dtype=float)
        RHODAIR = np.empty((imax, jmax), dtype=float)
        RHOAIR = np.empty((imax, jmax), dtype=float)
        CHIAIR = np.empty((imax, jmax), dtype=float)
        VISCAIR = np.empty((imax, jmax), dtype=float)
        Texh = np.empty((imax, jmax), dtype=float)
        VPSATAIROUT = np.empty((imax, jmax), dtype=float)
        RHOVPOUT = np.empty((imax, jmax), dtype=float)
        RHODAIROUT = np.empty((imax, jmax), dtype=float)
        RHOAIROUT = np.empty((imax, jmax), dtype=float)
        CHIAIROUT = np.empty((imax, jmax), dtype=float)
        AIREXCH = np.empty((imax, jmax), dtype=float)
        LHEATRESP = np.empty((imax, jmax), dtype=float)
        CHEATRESP = np.empty((imax, jmax), dtype=float)
        TGRESP = np.empty((imax, jmax), dtype=float)
        TNRESP = np.empty((imax, jmax), dtype=float)
        TNRESPH = np.empty((imax, jmax), dtype=float)
        NERESP = np.empty((imax, jmax), dtype=float)
        NERESPWM = np.empty((imax, jmax), dtype=float)
        NERESPC = np.empty((imax, jmax), dtype=float)
        MetheatSKIN = np.empty((imax, jmax), dtype=float)
        TskinC = np.empty((imax, jmax), dtype=float)
        TskinCH = np.empty((imax, jmax), dtype=float)
        CBSMIN = np.empty((imax, jmax), dtype=float)
        CONDBS = np.empty((imax, jmax), dtype=float)

        # 2. Latent heat release from the skin
        DLC = np.empty((imax, jmax), dtype=float)
        DIFFC = np.empty((imax, jmax), dtype=float)
        RV = np.empty((imax, jmax), dtype=float)
        VPSKINTOT = np.empty((imax, jmax), dtype=float)
        LASMAXENV = np.empty((imax, jmax), dtype=float)
        LASMAXPHYS = np.empty((imax, jmax), dtype=float)
        LASMAXCORR = np.empty((imax, jmax), dtype=float)
        ACTSW = np.empty((imax, jmax), dtype=float)
        ACTSWH = np.empty((imax, jmax), dtype=float)
        CSC = np.empty((imax, jmax), dtype=float)
        MetheatCOAT = np.empty((imax, jmax), dtype=float)
        TcoatC = np.empty((imax, jmax), dtype=float)
        TcoatCH = np.empty((imax, jmax), dtype=float)
        TcoatK = np.empty((imax, jmax), dtype=float)

        # 3.LWR heat balance of the coat
        LWRSKY = np.empty((imax, jmax), dtype=float)
        LWRENV = np.empty((imax, jmax), dtype=float)
        LB = np.empty((imax, jmax), dtype=float)
        LWRCOAT = np.empty((imax, jmax), dtype=float)
        LWRCOATH = np.empty((imax, jmax), dtype=float)

        # 4.Convective heat losses from the coat
        TAVGR = np.empty((imax, jmax), dtype=float)
        Ea = np.empty((imax, jmax), dtype=float)
        Ec = np.empty((imax, jmax), dtype=float)
        GRASHOF = np.empty((imax, jmax), dtype=float)
        WINDSP = np.empty((imax, jmax), dtype=float)
        REYNOLDS = np.empty((imax, jmax), dtype=float)
        ReH = np.empty((imax, jmax), dtype=float)
        ReL = np.empty((imax, jmax), dtype=float)
        NUSSELTH = np.empty((imax, jmax), dtype=float)
        NUSSELTL = np.empty((imax, jmax), dtype=float)
        NUSSELT = np.empty((imax, jmax), dtype=float)
        NUSSELTM = np.empty((imax, jmax), dtype=float)
        ka = np.empty((imax, jmax), dtype=float)
        CONVCOAT = np.empty((imax, jmax), dtype=float)
        CONVCOATH = np.empty((imax, jmax), dtype=float)

        # 5. Incoming SWR (solar radiation) to coat
        SAAC = np.empty((imax, jmax), dtype=float)
        SWRS = np.empty((imax, jmax), dtype=float)
        SWRC = np.empty((imax, jmax), dtype=float)
        ISWRC = np.empty((imax, jmax), dtype=float)
        #REFLE = None
        REFLE = np.full(imax, np.nan, dtype=float)
        SWR = np.empty((imax, jmax), dtype=float)
        RAINEVAP = np.empty((imax, jmax), dtype=float)

        # Synthesis and optimization with repeat {} function
        MetheatBAL = np.empty((imax, jmax), dtype=float)
        Metheatopt = np.empty((imax, jmax), dtype=float)
        METABFEED0 = np.reshape(
            np.repeat(100, imax * jmax),
            (imax, jmax),
            order="F"
        )
########################## line:1712 ##############################
        ###########################################################################################
        # 1.4.3     Specification of variables for energy and protein utilisation sub-model       #
        ###########################################################################################

        # Synthesis and optimisation
        EX = np.empty((imax, jmax), dtype=float)

        # Weight and derivative body tissues
        ADGHIGH = np.empty((imax + 1, jmax), dtype=float)
        ADGHIGH[0, 0:jmax] = 0
        TBWCHECK = np.empty((imax + 1, jmax), dtype=float)
        CARCW = np.empty((imax + 1, jmax), dtype=float)
        BONETIS = np.empty((imax + 1, jmax), dtype=float)
        MUSCLETIS = np.empty((imax + 1, jmax), dtype=float)
        INTRAMFTIS = np.empty((imax + 1, jmax), dtype=float)
        MISCFATTIS = np.empty((imax + 1, jmax), dtype=float)
        NONCARCTIS = np.empty((imax + 1, jmax), dtype=float)
        RUMEN = np.empty((imax + 1, jmax), dtype=float)

        DERBONE = np.empty((imax + 1, jmax), dtype=float)
        DERMUSCLE = np.empty((imax + 1, jmax), dtype=float)
        DERINTRAMF = np.empty((imax + 1, jmax), dtype=float)
        DERMISCFAT = np.empty((imax + 1, jmax), dtype=float)
        DERNONC = np.empty((imax + 1, jmax), dtype=float)
        DERRUMEN = np.empty((imax + 1, jmax), dtype=float)
        DERTOTAL = np.empty((imax + 1, jmax), dtype=float)

        # Lipid and protein concentrations in body tissues
        LIPIDFRACBONE = np.empty((imax + 1, jmax), dtype=float)
        LIPIDFRACBONEBF = np.empty((imax + 1, jmax), dtype=float)
        LIPIDFRACNONC = np.empty((imax + 1, jmax), dtype=float)
        LIPIDFRACNONCBF = np.empty((imax + 1, jmax), dtype=float)
        PROTFRACNONC = np.empty((imax + 1, jmax), dtype=float)
        PROTFRACNONCBF = np.empty((imax + 1, jmax), dtype=float)
        ENFEEDGROWTH = np.empty((imax + 1, jmax), dtype=float)
        ENFEEDGROWTHQ = np.empty((imax + 1, jmax), dtype=float)

        # Bone carcass
        LIPIDBONE = np.empty((imax + 1, jmax), dtype=float)
        PROTBONE = np.empty((imax + 1, jmax), dtype=float)
        ENGRBONE = np.empty((imax + 1, jmax), dtype=float)

        # Muscle carcass
        LIPIDMUSCLE = np.empty((imax + 1, jmax), dtype=float)
        PROTMUSCLE = np.empty((imax + 1, jmax), dtype=float)
        ENGRMUSCLE = np.empty((imax + 1, jmax), dtype=float)

        # Intramuscular fat
        LIPIDIMF = np.empty((imax + 1, jmax), dtype=float)
        PROTIMF = np.empty((imax + 1, jmax), dtype=float)
        ENGRIMF = np.empty((imax + 1, jmax), dtype=float)

        # Subcutaneous and intermuscular fat
        LIPIDFAT = np.empty((imax + 1, jmax), dtype=float)
        PROTFAT = np.empty((imax + 1, jmax), dtype=float)
        ENGRFAT = np.empty((imax + 1, jmax), dtype=float)

        # Non carcass tissue
        LIPIDNONC = np.empty((imax + 1, jmax), dtype=float)
        PROTNONC = np.empty((imax + 1, jmax), dtype=float)
        ENGRNONC = np.empty((imax + 1, jmax), dtype=float)

        # Growth influenced by defining and limiting biophyscial factors
        ENGRNONCBF = np.empty((imax + 1, jmax), dtype=float)
        ENGRBONEBF = np.empty((imax + 1, jmax), dtype=float)
        ENGRIMFBF = np.empty((imax + 1, jmax), dtype=float)
        ENGRMUSCLEBF = np.empty((imax + 1, jmax), dtype=float)
        ENGRFATBF = np.empty((imax + 1, jmax), dtype=float)
        ENGRTOTAL = np.empty((imax + 1, jmax), dtype=float)
        ENGRTOTALHIGH = np.empty((imax + 1, jmax), dtype=float)
        ENGRTOTALHIGH[0, 0:jmax] = 0
        ENGRTOTALHIGH1 = np.empty((imax + 1, jmax), dtype=float)
        ENGRTOTALHIGH1[0, 0:jmax] = 0
        REL = np.empty((imax + 1, jmax), dtype=float)
        REL[0, 0:jmax] = 0
        ENGRTOTALORIG = np.empty((imax + 1, jmax), dtype=float)

        FRENGRNONCBF = np.empty((imax + 1, jmax), dtype=float)
        FRENGRBONEBF = np.empty((imax + 1, jmax), dtype=float)
        FRENGRIMFBF = np.empty((imax + 1, jmax), dtype=float)
        FRENGRMUSCLEBF = np.empty((imax + 1, jmax), dtype=float)
        FRENGRFATBF = np.empty((imax + 1, jmax), dtype=float)
        FRENGRTOTAL = np.empty((imax + 1, jmax), dtype=float)

        BONETISBF = np.empty((imax + 1, jmax), dtype=float)
        MUSCLETISBF = np.empty((imax + 1, jmax), dtype=float)
        INTRAMFTISBF = np.empty((imax + 1, jmax), dtype=float)
        MISCFATTISBF = np.empty((imax + 1, jmax), dtype=float)
        NONCARCTISBF = np.empty((imax + 1, jmax), dtype=float)
        TBWBF = np.empty((imax + 1, jmax), dtype=float)
        EBWBFMET = np.empty((imax + 1, jmax), dtype=float)

        MISCFATFRAC = np.empty((imax + 1, jmax), dtype=float)
        LIPIDBONEBF = np.empty((imax + 1, jmax), dtype=float)
        LIPIDNONCBF = np.empty((imax + 1, jmax), dtype=float)
        PROTNONCBF = np.empty((imax + 1, jmax), dtype=float)
        LIPIDMUSCLEBF = np.empty((imax + 1, jmax), dtype=float)
        LIPIDIMFBF = np.empty((imax + 1, jmax), dtype=float)
        LIPIDFATBF = np.empty((imax + 1, jmax), dtype=float)
        LIPIDTOTW = np.empty((imax + 1, jmax), dtype=float)
        LIPIDFRACCARC = np.empty((imax + 1, jmax), dtype=float)

        ENCONTENTNONCBF = np.empty((imax + 1, jmax), dtype=float)
########################## line:1817 ##############################
        # Maintenance
        NEMAINT = np.empty((imax, jmax), dtype=float)
        NEMAINTWM = np.empty((imax, jmax), dtype=float)
        PROTDERML = np.empty((imax, jmax), dtype=float)
        PROTMAINT = np.empty((imax, jmax), dtype=float)
        PROTRESP = np.empty((imax, jmax), dtype=float)

        # Physical activity
        NEPHYSACT = np.empty((imax, jmax), dtype=float)
        NEPHYSACTWM = np.empty((imax, jmax), dtype=float)
        PROTPHACT = np.empty((imax, jmax), dtype=float)

        # Gestation
        CALFTBW = np.empty((imax + 1, jmax), dtype=float)
        CALFNR = np.empty((imax + 1, jmax), dtype=float)
        BIRTHW1 = np.empty((imax, jmax), dtype=float)

        GEST1 = np.empty((imax, jmax), dtype=float)
        GEST2 = np.empty((imax, jmax), dtype=float)
        GEST3 = np.empty((imax, jmax), dtype=float)
        GEST4 = np.empty((imax, jmax), dtype=float)
        GEST5 = np.empty((imax, jmax), dtype=float)
        GEST6 = np.empty((imax + 1, jmax), dtype=float)
        GEST = np.empty((imax, jmax), dtype=float)

        GESTDAY = np.empty((imax + 1, jmax), dtype=float)
        NEREQGEST = np.empty((imax, jmax), dtype=float)
        NEREQGESTADD = np.empty((imax + 1, jmax), dtype=float)
        NEREQGESTTOT = np.empty((imax, jmax), dtype=float)
        HEATGEST = np.empty((imax, jmax), dtype=float)
        PROTGESTG = np.empty((imax, jmax), dtype=float)
        TBWADD = np.empty((imax + 1, jmax), dtype=float)

        # Milk production
        MILKDAYST = np.empty((imax, jmax), dtype=float)
        MILKDAY = np.empty((imax + 1, jmax), dtype=float)
        MILKWEEK = np.empty((imax + 1, jmax), dtype=float)
        ADDMILK1 = np.empty((imax, jmax), dtype=float)
        ADDMILK2 = np.empty((imax, jmax), dtype=float)

        MAXMILKPROD = np.empty((imax, jmax), dtype=float)
        GEMILK = np.empty((imax, jmax), dtype=float)
        GEMILKTOT = np.empty((imax, jmax), dtype=float)
        MEMILKCALF = np.empty((imax, jmax), dtype=float)
        MEMILKCALFINIT = np.empty((imax, jmax), dtype=float)
        NEMILKCOW = np.empty((imax, jmax), dtype=float)
        CALFLIVENR = np.empty((imax, jmax), dtype=float)
        CALFWEANNR = np.empty((imax, jmax), dtype=float)
        MILKPRODBF = np.empty((imax, jmax), dtype=float)
        HEATMILK = np.empty((imax, jmax), dtype=float)
        NETMILKEN = np.empty((imax + 1, jmax), dtype=float)
        PROTMILK = np.empty((imax, jmax), dtype=float)
        PROTMILKG = np.empty((imax, jmax), dtype=float)

        # ME total
        MEREQTOTAL = np.empty((imax, jmax), dtype=float)
        MEREQTOTAL2 = np.empty((imax, jmax), dtype=float)
        NETMILKEN = np.empty((imax, jmax), dtype=float)

        # Cold stress
        Metheatcold = np.full((imax, jmax), np.nan, dtype=float)
        METABSTARTCOLD = np.reshape(
            np.repeat(100, imax * jmax),
            (imax, jmax),
            order="F"
        )
        TOTHEAT = np.empty((imax, jmax), dtype=float)
        FATBURN = np.empty((imax, jmax), dtype=float)
        REDTIS2 = np.empty((imax, jmax), dtype=float)
        REDTIS3 = np.empty((imax, jmax), dtype=float)
        MAINTFRAC = np.empty((imax, jmax), dtype=float)
        FATFRACCARC = np.empty((imax, jmax), dtype=float)
########################## line:1886 ##############################
        ###########################################################################################
        # 1.4.4                   Variables for integration of sub-models                         #
        ###########################################################################################

        COMPGROWTH = np.empty((imax + 1, jmax), dtype=float)
        COMPGROWTH1 = np.empty((imax + 1, jmax), dtype=float)
        COMPGROWTH2 = np.empty((imax + 1, jmax), dtype=float)
        COMPGROWTH3 = np.empty((imax + 1, jmax), dtype=float)
        COMPGROWTH4 = np.empty((imax + 1, jmax), dtype=float)
        COMPGROWTH5 = np.empty((imax + 1, jmax), dtype=float)

        ENGRTOTALCOMP = np.empty((imax + 1, jmax), dtype=float)

        HEATBONEACT = np.empty((imax + 1, jmax), dtype=float)
        HEATMUSCLEACT = np.empty((imax + 1, jmax), dtype=float)
        HEATIMFACT = np.empty((imax + 1, jmax), dtype=float)
        HEATMISCFATACT = np.empty((imax + 1, jmax), dtype=float)
        HEATNONCACT = np.empty((imax + 1, jmax), dtype=float)

        HEATTOTALACT = np.empty((imax + 1, jmax), dtype=float)

        ENBONEACT = np.empty((imax + 1, jmax), dtype=float)
        ENMUSCLEACT = np.empty((imax + 1, jmax), dtype=float)
        ENIMFACT = np.empty((imax + 1, jmax), dtype=float)
        ENMISCFATACT = np.empty((imax + 1, jmax), dtype=float)
        ENNONCACT = np.empty((imax + 1, jmax), dtype=float)
        ENTOTALACT = np.empty((imax + 1, jmax), dtype=float)

        PROTBONEACT = np.empty((imax + 1, jmax), dtype=float)
        PROTMUSCLEACT = np.empty((imax + 1, jmax), dtype=float)
        PROTIMFACT = np.empty((imax + 1, jmax), dtype=float)
        PROTMISCFATACT = np.empty((imax + 1, jmax), dtype=float)
        PROTNONCBF1 = np.empty((imax + 1, jmax), dtype=float)
        PROTTOTALACT = np.empty((imax + 1, jmax), dtype=float)
        PROTGROSS = np.empty((imax + 1, jmax), dtype=float)
        UREABL = np.empty((imax + 1, jmax), dtype=float)
        NRECYCLPT = np.empty((imax + 1, jmax), dtype=float)
        PROTNETT = np.empty((imax + 1, jmax), dtype=float)
        PROTACCR = np.empty((imax + 1, jmax), dtype=float)

        HEATCLIMGEN = np.empty((imax + 1, jmax), dtype=float)
        DIFFEN = np.empty((imax + 1, jmax), dtype=float)

        HEATIFEEDMAINT = np.empty((imax, jmax), dtype=float)
        HEATIFEEDMAINTWM = np.empty((imax, jmax), dtype=float)
        HEATIFEEDGROWTH = np.empty((imax, jmax), dtype=float)
        HEATIFEEDGROWTHWM = np.empty((imax, jmax), dtype=float)
        HEATIFEEDGROWTHC = np.empty((imax, jmax), dtype=float)
        HEATIFEEDGROWTHCWM = np.empty((imax, jmax), dtype=float)
        REDMAINT = np.empty((imax, jmax), dtype=float)
        REDMAINT2 = np.empty((imax, jmax), dtype=float)
        REDMAINT3 = np.empty((imax, jmax), dtype=float)
        REDTIS = np.empty((imax, jmax), dtype=float)
        REDTISPROT = np.empty((imax, jmax), dtype=float)
        #CHECK = np.empty((imax, jmax), dtype=float) # orginal
        CHECK = np.empty((imax, jmax), dtype=object)
        CHECK[:] = np.nan
        REDHP = np.empty((imax, jmax), dtype=float)

        ###########################################################################################
        # 1.4.5                     Variables for herd dynamics and output                        #
        ###########################################################################################

        TIME = np.empty((imax, jmax), dtype=float)
        TIME2 = np.empty((imax + 1, jmax), dtype=float)

        TIMEYEAR = np.empty((imax, jmax), dtype=float)
        TIMEYEAR2 = np.empty((imax + 1, jmax), dtype=float)

        BIRTHDAYCALF1 = 1  # Initial values for birthdays calves (days)
        BIRTHDAYCALF2 = 1  # Calculated as days after birth reproductive animal
        BIRTHDAYCALF3 = 1  # Values are recalculated
        BIRTHDAYCALF4 = 1
        BIRTHDAYCALF5 = 1
        BIRTHDAYCALF6 = 1
        BIRTHDAYCALF7 = 1
        BIRTHDAYCALF8 = 1
        BIRTHDAYCALF9 = 1
        BIRTHDAY = None
        WNDAY = None

        # Modelling calves and cow parity

        PARITY1 = np.empty((imax, jmax), dtype=float)  # Cow has been or is in the first parity
        PARITY2 = np.empty((imax, jmax), dtype=float)  # Cow has been or is in the second parity
        PARITY3 = np.empty((imax, jmax), dtype=float)  # Cow has been or is in the third parity
        PARITY4 = np.empty((imax, jmax), dtype=float)  # Cow has been or is in the fourth parity
        PARITY5 = np.empty((imax, jmax), dtype=float)  # Cow has been or is in the fifth parity
        PARITY6 = np.empty((imax, jmax), dtype=float)  # Cow has been or is in the sixth parity
        PARITY7 = np.empty((imax, jmax), dtype=float)  # Cow has been or is in the seventh parity
        PARITY8 = np.empty((imax, jmax), dtype=float)  # Cow has been or is in the eighth parity
        PARITY9 = np.empty((imax, jmax), dtype=float)  # Cow has been or is in the nineth parity
########################## line:1977 ##############################
        # Herd dynamics
        BEEFPROD = np.empty((imax + 1, jmax), dtype=float)
        BEEFPRODYEAR = np.empty((imax + 1, jmax), dtype=float)

        BEEFPRODACT = np.empty((imax + 1, jmax), dtype=float)
        LWPRODACT = np.empty((imax + 1, jmax), dtype=float)
        CARCPRODACT = np.empty((imax + 1, jmax), dtype=float)
        LWPROD = np.empty((imax + 1, jmax), dtype=float)
        LWPRODYEAR = np.empty((imax + 1, jmax), dtype=float)
        LWPRODHERD = None

        SLAUGHTERDAYACT = np.empty((imax + 1, jmax), dtype=float)
        SLAUGHTERDAYACTpl = np.empty((imax + 1, jmax), dtype=float)
        SLAUGHTERDAYACTHEIFER = np.empty((imax + 1, jmax), dtype=float)
        #ENDDAY = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
        ENDDAY = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int)

        SUMFEED1 = np.empty((imax, jmax), dtype=float)
        SUMFEED2 = np.empty((imax, jmax), dtype=float)
        SUMFEED3 = np.empty((imax, jmax), dtype=float)
        SUMFEED4 = np.empty((imax, jmax), dtype=float)
        SUMFEED = np.empty((imax, jmax), dtype=float)

        CUMULFEED1 = np.empty((imax, jmax), dtype=float)
        CUMULFEED2 = np.empty((imax, jmax), dtype=float)
        CUMULFEED3 = np.empty((imax, jmax), dtype=float)
        CUMULFEED4 = np.empty((imax, jmax), dtype=float)
        CUMULFEED = np.empty((imax, jmax), dtype=float)

        FATBURNCUMUL = np.empty((imax + 1, jmax), dtype=float)
        HEATBURNCUMUL = np.empty((imax, jmax), dtype=float)
        ALIVE = np.empty((imax + 1, jmax), dtype=float)

        FCR = np.empty((imax, jmax), dtype=float)
        FCRBEEF = np.empty((imax, jmax), dtype=float)
        FCRBEEFENDDAY = np.empty((imax, jmax), dtype=float)

        MILKSTART = np.empty((imax, jmax), dtype=float)
        MILKSTARTPR = np.empty((imax, jmax), dtype=float)
        MILKSTARTPRHF = np.empty((imax, jmax), dtype=float)

        METABFEED = np.empty((imax, jmax), dtype=float)
        METABFEEDC = np.empty((imax, jmax), dtype=float)
        METABFEEDCH = np.empty((imax, jmax), dtype=float)

        #CHECKHEAT1 = np.full((imax, jmax), np.nan, dtype=float) #orginal
        CHECKHEAT1 = np.empty((imax, jmax), dtype=object)
        CHECKHEAT1[:] = np.nan

        #CHECKHEAT2 = np.full((imax, jmax), np.nan, dtype=float) #orginal
        CHECKHEAT2 = np.empty((imax, jmax), dtype=object)
        CHECKHEAT2[:] = np.nan
        #CHECKHEAT3 = np.full((imax, jmax), np.nan, dtype=float) #orginal
        CHECKHEAT3 = np.empty((imax, jmax), dtype=object)
        CHECKHEAT3[:] = np.nan

        CHECKCOMP = np.empty((imax, jmax), dtype=float)
   
        # orginal
        '''     
        MAXW1 = None
        CALVESPERANIMAL = None
        BEEFPRODHERD = None
        FCRHERDBEEF = None
        CUMULFEEDHERD = None
        CUMULFEED1HERD = None
        CUMULFEED2HERD = None
        CUMULFEED3HERD = None
        CUMULFEED4HERD = None
        ANIMALYEARS = None
        AVANWEIGHT = np.empty((imax, jmax), dtype=float)
        AVANMETWEIGHT = np.empty((imax, jmax), dtype=float)

        ANIMALINFO = None
        HERDINFO = None
        HERDINFO1 = None
        '''

        MAXW1 = np.full(jmax, np.nan, dtype=float)

        CALVESPERANIMAL = None

        BEEFPRODHERD = np.full(jmax, np.nan, dtype=float)
        LWPRODHERD = np.full(jmax, np.nan, dtype=float)
        FCRHERDBEEF = np.full(jmax, np.nan, dtype=float)

        CUMULFEEDHERD = np.full(jmax, np.nan, dtype=float)
        CUMULFEED1HERD = np.full(jmax, np.nan, dtype=float)
        CUMULFEED2HERD = np.full(jmax, np.nan, dtype=float)
        CUMULFEED3HERD = np.full(jmax, np.nan, dtype=float)
        CUMULFEED4HERD = np.full(jmax, np.nan, dtype=float)

        ANIMALYEARS = np.full(jmax, np.nan, dtype=float)

        # These are used later as 1D per-animal summaries, not imax x jmax matrices
        AVANWEIGHT = np.full(jmax, np.nan, dtype=float)
        AVANMETWEIGHT = np.full(jmax, np.nan, dtype=float)

        ANIMALINFO = None
        HERDINFO = None
        HERDINFO1 = None

        FATCOMP = np.empty((imax, jmax), dtype=float)
        NONCF = np.empty((imax, jmax), dtype=float)
        PERCFI = np.empty((imax, jmax), dtype=float)
        REPS = np.empty((imax, jmax), dtype=float)

        MEMET = np.empty((imax, jmax), dtype=float)
        MERED = np.empty((imax, jmax), dtype=float)

        PROTNONG = np.empty((imax, jmax), dtype=float)
        HIFM = np.empty((imax, jmax), dtype=float)
        PROTNONGM = np.empty((imax, jmax), dtype=float)
        CPAVG = np.empty((imax, jmax), dtype=float)

        OUTPUTHERDS = None  # Matrix with information for one herd unit

        ###########################################################################################
        #                            Dynamic part of the model (animals)                          #
        ###########################################################################################

        HOUSING1 = HOUSING.copy()  # Creates a copy of the vector HOUSING
        FEED11 = FEED1.copy()      # Creates a copy of the matrix FEED1
        FEED21 = FEED2.copy()      # Creates a copy of the matrix FEED2
        FEED31 = FEED3.copy()      # Creates a copy of the matrix FEED3

        breakFlaganim = False  # breakFlaganim indicates whether the simulation of an animal should
        # be continued (if FALSE) or terminated (if TRUE), e.g. when the
        # maximum number of calves is reached per reproductive animal.
########################## line:2069 ##############################
        for j in range(1, jmax + 1):
        #for j in progress_iter( range(1, jmax + 1), total=jmax, desc=f"Case {z} / animals", position=2, leave=False, enable=SHOW_PROGRESS and (not DEBUG_LOOP) ):
            # Loop for individual animals starts here (j = jth animal)

            #imax = np.array([imax, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500], dtype=int)   # maximum of 2500 day life span  # orginal
            imax = np.concatenate(([int(np.atleast_1d(imax)[0])], np.full(8, 2500, dtype=int)))   # maximum of 2500 day life span
            # for productive animals

            # orginal
            '''
            if j > 1:
                HOUSING = HOUSING1[BIRTHDAY[j - 1] - 1:len(HOUSING1)]  # adjusts HOUSING for offspring
            if j > 1:
                FEED1 = FEED11[BIRTHDAY[j - 1] - 1:len(HOUSING1), :]    # adjusts FEED1 for offspring
            if j > 1:
                FEED2 = FEED21[BIRTHDAY[j - 1] - 1:len(HOUSING1), :]    # adjusts FEED2 for offspring
            if j > 1:
                FEED3 = FEED31[BIRTHDAY[j - 1] - 1:len(HOUSING1), :]    # adjusts FEED3 for offspring
            '''
   
            if j > 1:
                start_idx = max(0, int(BIRTHDAY[j - 1]) - 1)
                HOUSING = HOUSING1[start_idx:]
                FEED1 = FEED11[start_idx:, :]
                FEED2 = FEED21[start_idx:, :]
                FEED3 = FEED31[start_idx:, :]
            else:
                start_idx = 0

            # Limit the simulated lifespan to the available offspring data
            imax_j = min(
                int(imax[j - 1]),
                len(HOUSING),
                FEED1.shape[0],
                FEED2.shape[0],
                FEED3.shape[0],
                len(WEATHER)
            )

            
                
            #for i in range(1, imax[j - 1] + 1): # orginal
            for i in range(1, imax_j + 1):
            #for i in progress_iter( range(1, imax_j + 1), total=imax_j, desc=f"Case {z} / animal {j} / days", position=3, leave=PROGRESS_LEAVE_INNER, enable=SHOW_PROGRESS and (not DEBUG_LOOP) ):
                # Loop for daily time step starts here (i = ith day)

                breakFlagtime = False  # breakFlagtime indicates whether the simulation of an animal
                # should be slaughtered.

                ###########################################################################################
                # 1.5                          Initial values for individual animals                      #
                ###########################################################################################

                # Code below selects the library for genotype (i.e. breed) and sex
                if BREED == 1 and SEX[j - 1] == 0:
                    LIBRARY = LIBRARY10
                else:
                    if BREED == 1 and SEX[j - 1] == 1:
                        LIBRARY = LIBRARY11
                    else:
                        if BREED == 2 and SEX[j - 1] == 0:
                            LIBRARY = LIBRARY20
                        else:
                            if BREED == 2 and SEX[j - 1] == 1:
                                LIBRARY = LIBRARY21
                            else:
                                if BREED == 3 and SEX[j - 1] == 0:
                                    LIBRARY = LIBRARY30
                                else:
                                    if BREED == 3 and SEX[j - 1] == 1:
                                        LIBRARY = LIBRARY31
                                    else:
                                        if BREED == 4 and SEX[j - 1] == 0:
                                            LIBRARY = LIBRARY40
                                        else:
                                            if BREED == 4 and SEX[j - 1] == 1:
                                                LIBRARY = LIBRARY41
                                            else:
                                                if BREED == 5 and SEX[j - 1] == 0:
                                                    LIBRARY = LIBRARY50
                                                else:
                                                    if BREED == 5 and SEX[j - 1] == 1:
                                                        LIBRARY = LIBRARY51
                                                    else:
                                                        if BREED == 6 and SEX[j - 1] == 0:
                                                            LIBRARY = LIBRARY60
                                                        else:
                                                            if BREED == 6 and SEX[j - 1] == 1:
                                                                LIBRARY = LIBRARY61

                # During sensitivity analysis, the parameters in the library are changed (not used)
                #LIBRARY = LIBRARY * SENSMAT[41:67, s - 1]
                sens_lib = SENSMAT[41:67, s - 1]   # 26 values, same as R 42:67
                LIBRARY = LIBRARY.copy()
                LIBRARY[:len(sens_lib)] = LIBRARY[:len(sens_lib)] * sens_lib

                # Matrix with time steps (days, starts at day 1)
                TIME = np.reshape(
                    np.tile(np.arange(1, imax[j - 1] + 1, dtype=float), jmax),
                    (imax[j - 1], jmax),
                    order="F"
                )

                # Matrix with time steps (days, starts at day 0)
                TIME2 = np.reshape(
                    np.tile(np.arange(0, imax[j - 1] + 1, dtype=float), jmax),
                    (imax[j - 1] + 1, jmax),
                    order="F"
                )

                # Matrix with time steps (years, starts at day 1)
                TIMEYEAR = np.reshape(
                    np.tile(np.arange(1, imax[j - 1] + 1, dtype=float) / 365, jmax),
                    (imax[j - 1], jmax),
                    order="F"
                )

                # Matrix with time steps (years, starts at day 0)
                TIMEYEAR2 = np.reshape(
                    np.tile(np.arange(0, imax[j - 1] + 1, dtype=float) / 365, jmax),
                    (imax[j - 1] + 1, jmax),
                    order="F"
                )
########################## line:2144 ##############################
                # A few parameters from the LIBRARY are re-named here. Values between brackets indicate
                # the parameter numbers in Table S2 of the Supplementary Information
                REFLC = LIBRARY[0]        # [5]  Reflectance coat (-)
                LC = LIBRARY[1]           # [3]  Coat length (m)
                AREAFACTOR = LIBRARY[2]   # [1]  Body area (Body area : weight factor)
                CBSMAX = LIBRARY[3]       # [2]  Max. conduction body core ??? skin (W m-2 K-1)
                MAXW1[j - 1] = LIBRARY[12]  # [20] Maximum adult weight (kg)
                MILKPARA = LIBRARY[10]    # [13] Lactation curve 1
                MILKPARB = LIBRARY[11]    # [14] Lactation curve 2
                MILKPARC = LIBRARY[26] if len(LIBRARY) > 26 else 0.0    # [15] Lactation curve 3; Hereford steer library has no lactation-C term in the R source
                RBCSf = LIBRARY[22]       # [4]  Minimum conduction body core-skin (-)
                MAXW = LIBRARY[5]         # Maximum adult weight for the Gompertz curve (kg);
                # parameter accounts for the reduction parameter (EPAR)
                BIRTHW = LIBRARY[6]       # [9]  Birth weight (kg)
                CPAR = LIBRARY[7]         # [10] Constant of integration Gompertz curve
                DPAR = LIBRARY[8]         # [11] Rate constant Gompertz curve
                EPAR = LIBRARY[9]         # [12] Reduction parameter Gompertz curve

                # Initial total body weight (kg live weight), genetic potential
                TBW[0, j - 1] = LIBRARY[4]
                # Initial carcass weight (kg)
                CARCW[0, j - 1] = LIBRARY[4] * INCARC
                # Initial bone weight (kg)
                BONETIS[0, j - 1] = CARCW[0, j - 1] * min(
                    BONEFRACMAX,
                    (BONEGROWTH1 * SENSMAT[67, s - 1]) *
                    CARCW[0, j - 1] ** -(BONEGROWTH2 * SENSMAT[68, s - 1])
                )
                # Initial muscle weight (kg)
                MUSCLETIS[0, j - 1] = BONETIS[0, j - 1] * min(
                    LIBRARY[21],
                    (MUSCLEGROWTH1 * SENSMAT[69, s - 1]) *
                    CARCW[0, j - 1] ** 2 + LIBRARY[21] / 100 * CARCW[0, j - 1] +
                    (MUSCLEGROWTH2 * SENSMAT[70, s - 1])
                )
                # Initial weight intramuscular fat (kg)
                INTRAMFTIS[0, j - 1] = (
                    (IMFGROWTH1 * SENSMAT[71, s - 1]) * (BONETIS[0, j - 1] + MUSCLETIS[0, j - 1]) ** 2 +
                    (IMFGROWTH2 * SENSMAT[72, s - 1]) * (BONETIS[0, j - 1] + MUSCLETIS[0, j - 1]) -
                    (IMFGROWTH3 * SENSMAT[73, s - 1])
                )
                # Initial weight of the intermuscular and subcutaneous (i.e. miscellaneous ) fat tissue
                # (kg)
                MISCFATTIS[0, j - 1] = (
                    CARCW[0, j - 1] - BONETIS[0, j - 1] - MUSCLETIS[0, j - 1] -
                    INTRAMFTIS[0, j - 1]
                )
                # Initial weight of the non-carcass tissue (kg)
                NONCARCTIS[0, j - 1] = TBW[0, j - 1] * (1 - RUMENFRAC) - CARCW[0, j - 1]
                # Initial weight of the rumen contents
                RUMEN[0, j - 1] = TBW[0, j - 1] * RUMENFRAC

                # Initial weight of lipid in bone tissue (-)
                LIPIDBONE[0, j - 1] = BONETIS[0, j - 1] * (LIBRARY[19] * np.log(BONETIS[0, j - 1])) / 100
                # Initial weight of protein in bone tissue (-)
                PROTBONE[0, j - 1] = BONETIS[0, j - 1] * PROTFRACBONE
                # Initial weight of lipid in muscle tissue (-)
                LIPIDMUSCLE[0, j - 1] = MUSCLETIS[0, j - 1] * LIPFRACMUSCLE
                # Initial weight of protein in muscle tissue (-)
                PROTMUSCLE[0, j - 1] = MUSCLETIS[0, j - 1] * PROTFRACMUSCLE
                # Initial weight of lipid in intramuscular fat tissue (-)
                LIPIDIMF[0, j - 1] = INTRAMFTIS[0, j - 1] * LIPFRACFAT
                # Initial weight of protein in intramuscular fat tissue (-)
                PROTIMF[0, j - 1] = INTRAMFTIS[0, j - 1] * PROTFRACFAT
                # Initial weight of lipid in the miscellaneous fat tissue (-)
                LIPIDFAT[0, j - 1] = MISCFATTIS[0, j - 1] * LIPFRACFAT
                # Initial weight of protein in the miscellaneous fat tissue (-)
                PROTFAT[0, j - 1] = MISCFATTIS[0, j - 1] * PROTFRACFAT
                # Initial weight of lipid in the non-carcass tissue (-)
                LIPIDNONC[0, j - 1] = 1.00
                # Initial weight of protein in the non-carcass tissue (-)
                PROTNONC[0, j - 1] = NONCARCTIS[0, j - 1] * (
                    (PROTNONCM1 * SENSMAT[74, s - 1]) * TBW[0, j - 1] +
                    (PROTNONCM2 * SENSMAT[75, s - 1])
                ) / 100

                # The animal is at its potential weight at the first time step.
                # Later on, the weight of the animal can deviate from its potential weight, since other
                # biophysical factors than the genotype can affect growth.
                # Note: LiGAPS-Beef does not account for growth reduction during the gestation period.

                # Initial bone weight (kg)
                BONETISBF[0, j - 1] = BONETIS[0, j - 1]
                # Initial muscle weight (kg)
                MUSCLETISBF[0, j - 1] = MUSCLETIS[0, j - 1]
                # Initial intramuscular fat weight (kg)
                INTRAMFTISBF[0, j - 1] = INTRAMFTIS[0, j - 1]
                # Initial miscellaneous fat weight (kg)
                MISCFATTISBF[0, j - 1] = MISCFATTIS[0, j - 1]
                # Initial non-carcass weight (kg)
                NONCARCTISBF[0, j - 1] = NONCARCTIS[0, j - 1]
                # Initial total body weight (TBW, in kg live weight)
                TBWBF[0, j - 1] = TBW[0, j - 1]
                # Initial metabolic body weight (TBW^0.75)
                EBWBFMET[0, j - 1] = (TBWBF[0, j - 1] * (1 - RUMENFRAC)) ** 0.75
                # Initial fraction of miscellaneous fat in the body
                MISCFATFRAC[0, j - 1] = MISCFATTISBF[0, j - 1] / TBWBF[0, j - 1]
                # Initial weight lipids in the bone tissue (kg)
                LIPIDBONEBF[0, j - 1] = LIPIDBONE[0, j - 1]
                # Initial weight lipids in non-carcass tissue (kg)
                LIPIDNONCBF[0, j - 1] = LIPIDNONC[0, j - 1]
                # Initial weight protein in non-carcass tissue (kg)
                PROTNONCBF[0, j - 1] = PROTNONC[0, j - 1]
                # Initial weight lipids in muscle tissue (kg)
                LIPIDMUSCLEBF[0, j - 1] = LIPIDMUSCLE[0, j - 1]
                # Initial weight lipids in the intramuscular fat tissue (kg)
                LIPIDIMFBF[0, j - 1] = LIPIDIMF[0, j - 1]
                # Initial weight lipids in the miscellaneous fat tissue (kg)
                LIPIDFATBF[0, j - 1] = LIPIDFAT[0, j - 1]

                # Initial weight of total lipids in the body (kg)
                LIPIDTOTW[0, j - 1] = (
                    LIPIDBONEBF[0, j - 1] + LIPIDNONCBF[0, j - 1] + LIPIDMUSCLEBF[0, j - 1] +
                    LIPIDIMFBF[0, j - 1] + LIPIDFATBF[0, j - 1]
                )
                # Initial fraction of lipids in the carcass
                LIPIDFRACCARC[0, j - 1] = (
                    LIPIDBONEBF[0, j - 1] + LIPIDMUSCLEBF[0, j - 1] + LIPIDIMFBF[0, j - 1] +
                    LIPIDFATBF[0, j - 1]
                ) / (TBWBF[0, j - 1] - NONCARCTISBF[0, j - 1])
########################## line:2258 ##############################
                ###########################################################################################

                HEATTOTALACT[0, j - 1] = 9.00  # Assumption at the first day for heat release.
                FATBURNCUMUL[0, j - 1] = 0     # Cumulative amount of fat dissimilated used to maintain body
                # temperature (cold stress) is zero (MJ)
                HEATBURNCUMUL[0, j - 1] = 0    # Cumulative amount of heat used to maintain body temperature
                # (cold stress) is zero (MJ)
                ALIVE[0, j - 1] = 1            # The animal is alive at the first time step

                # Requirements for gestation:
                CALFTBW[0, j - 1] = 0.0        # No calf born yet at the first time step, weight is zero (kg)
                CALFNR[0, j - 1] = 0           # Zero calves are born (only for reproductive cows)

                GEST1[0, j - 1] = 0            # The total body weight is not higher than the body weight required
                # for gestation
                GEST2[0, j - 1] = 0            # Gestation not applicable at the first time step
                GEST3[0, j - 1] = 1            # Minimum calving interval is not applicable at the first time step
                GEST4[0, j - 1] = 0            # Fat tissue in the carcass is assumed to be below the minimum
                # required for conception
                GEST5[0, j - 1] = 0            # Calves cannot conceive at the first time step and after reaching
                # the maximum age for conception
                GEST6[0, j - 1] = 1            # Maximum number of calves per cow is not achieved yet
                GESTDAY[0, j - 1] = 0          # No gestation at the first time step, days in gestation not
                # applicable (days)
                NEREQGESTADD[0, j - 1] = 0     # No gestation at the first time step, no NE for gestation (MJ per
                # day)
                TBWADD[0, j - 1] = 0           # No gestation at the first time step, no weight of foetus (kg)

                MILKDAY[0, j - 1] = 0          # No milk production at the first time step, days in milk not
                # applicable (days)
                MILKWEEK[0, j - 1] = 0         # No milk production at the first time step, days in milk not
                # applicable (days)

                PARITY1[0, j - 1] = 0          # Cow parity is not equal to or higher than 1 at the first time step
                PARITY2[0, j - 1] = 0          # Cow parity is not equal to or higher than 2 at the first time step
                PARITY3[0, j - 1] = 0          # Cow parity is not equal to or higher than 3 at the first time step
                PARITY4[0, j - 1] = 0          # Cow parity is not equal to or higher than 4 at the first time step
                PARITY5[0, j - 1] = 0          # Cow parity is not equal to or higher than 5 at the first time step
                PARITY6[0, j - 1] = 0          # Cow parity is not equal to or higher than 6 at the first time step
                PARITY7[0, j - 1] = 0          # Cow parity is not equal to or higher than 7 at the first time step
                PARITY8[0, j - 1] = 0          # Cow parity is not equal to or higher than 8 at the first time step

                ###########################################################################################
                # 2.                                   Dynamic section                                    #
                #                                     (time and animals)                                  #
                ###########################################################################################

                ###########################################################################################
                # 2.1                             Thermoregulation submodel                               #
                ###########################################################################################

                # Aim: To calculate the maximum and minimum heat release (W m-2) of an animal with its
                # environment.

                # Five flows of energy between an animal and its environment
                #   1. Latent and convective heat release from respiration
                #   2. Latent heat release from the skin
                #   3. Long wave radiation balance of the coat
                #   4. Convective heat losses from the coat
                #   5. Solar radiation intecepted by the coat

                ###########################################################################################
                # 2.1.1                             Maximum heat release                                  #
                ###########################################################################################

                # Heat release mechanisms of cattle at maximum heat release
                TISSUEFRAC = 1.00  # Vasodilatation (0 = minimum and 1 = maximum vasodilatation)
                LHRskin = 1.00     # Latent heat release from the skin (0 = basal and 1 = maximum
                # physiological 'sweating' rate)
                PANTING = RESPDUR * SENSMAT[76, s - 1]  # Panting (0 = basal respiration, 1 = maximum panting)

                # Calculations related to weather conditions

                # average temperature (degrees Celsius)
                # orginal
                #TAVGC[i - 1, j - 1] = (WEATHER.loc[i - 1, "MINT"] + WEATHER.loc[i - 1, "MAXT"]) / 2
                TAVGC[i - 1, j - 1] = (WEATHER.iloc[i - 1, WEATHER.columns.get_loc("MINT")] +
                       WEATHER.iloc[i - 1, WEATHER.columns.get_loc("MAXT")]) / 2
                # average temperature (degrees Kelvin)
                TAVGK[i - 1, j - 1] = CtoK + TAVGC[i - 1, j - 1]
                # saturated vapour pressure air (Pa)
                VPSATAIR[i - 1, j - 1] = 6.1078 * 10 ** (
                    (7.5 * TAVGC[i - 1, j - 1]) / (TAVGC[i - 1, j - 1] + 237.3)
                ) * 100
                # real vapour pressure air (kPa)
                VPAIRTOT[i - 1, j - 1] = WEATHER.loc[i - 1, "VPR"] * 1000
                # relative humidity (-)
                RHAIR[i - 1, j - 1] = VPAIRTOT[i - 1, j - 1] / VPSATAIR[i - 1, j - 1] * 100
                # water vapour density (kg m-3)
                RHOVP[i - 1, j - 1] = VPAIRTOT[i - 1, j - 1] / (Rwater * TAVGK[i - 1, j - 1])
                # dry air density (kg m-3)
                RHODAIR[i - 1, j - 1] = (P - VPAIRTOT[i - 1, j - 1]) / (Rdair * TAVGK[i - 1, j - 1])
                # air density (kg m-3)
                RHOAIR[i - 1, j - 1] = RHOVP[i - 1, j - 1] + RHODAIR[i - 1, j - 1]
                # water vapour density (kg kg-1)
                CHIAIR[i - 1, j - 1] = RHOVP[i - 1, j - 1] * RHOAIR[i - 1, j - 1]
########################## line:2351 ##############################
                #########################################################################################
                #                1. Latent and convective heat release from respiration                 #
                #########################################################################################

                # Animal surface area (m2), McGovern and Bruce (2000)
                # This equation corresponds to Eq. 2 of the Supplementary Information
                AREA[i - 1, j - 1] = (
                    (BODYAREA1 * SENSMAT[77, s - 1]) *
                    TBWBF[i - 1, j - 1] ** (BODYAREA2 * SENSMAT[78, s - 1]) *
                    AREAFACTOR
                )
                # This equation corresponds to Eq. 3 of the Supplementary Information
                # Animal diameter (m), McGovern and Bruce (2000)
                DIAMETER[i - 1, j - 1] = (
                    (DIAMETER1 * SENSMAT[79, s - 1]) *
                    TBWBF[i - 1, j - 1] ** (DIAMETER2 * SENSMAT[80, s - 1])
                )
                # Animal length (m)
                LENGTH[i - 1, j - 1] = (
                    AREA[i - 1, j - 1] - 0.5 * np.pi * DIAMETER[i - 1, j - 1] ** 2
                ) / (np.pi * DIAMETER[i - 1, j - 1])

                # Basal respiration rate (min-1), McGovern and Bruce (2000)
                brr[i - 1, j - 1] = (
                    (BASALRR1 * SENSMAT[81, s - 1]) *
                    TBWBF[i - 1, j - 1] ** (BASALRR2 * SENSMAT[82, s - 1])
                )
                # Basal tidal volume (L) McGovern and Bruce (2000)
                btv[i - 1, j - 1] = (BASALTV * SENSMAT[83, s - 1]) * TBWBF[i - 1, j - 1]
                # Basal respiration volume (L min-1)
                # This equation corresponds to Eq. 5 of the Supplementary Information
                brv[i - 1, j - 1] = brr[i - 1, j - 1] * btv[i - 1, j - 1]
                # Increased respiration volume (L min-1)
                # This equation corresponds to Eq. 6 of the Supplementary Information
                irv[i - 1, j - 1] = brv[i - 1, j - 1] + PANTING * ((RESPINCR - 1) * brv[i - 1, j - 1])
                # Temperature exhaled air (degrees Celsius), Stevens (1981)
                # This equation corresponds to Eq. 7 of the Supplementary Information
                Texh[i - 1, j - 1] = (
                    (TEXHALED1 * SENSMAT[84, s - 1]) +
                    (TEXHALED2 * SENSMAT[85, s - 1]) * TAVGC[i - 1, j - 1] +
                    np.exp(
                        (TEXHALED3 * SENSMAT[86, s - 1]) * RHAIR[i - 1, j - 1] +
                        (TEXHALED4 * SENSMAT[87, s - 1]) * TAVGC[i - 1, j - 1]
                    )
                )

                # Assumption: exhaled air is saturated with water

                # Saturated vapour pressure exhaled air (Pa)
                VPSATAIROUT[i - 1, j - 1] = 6.1078 * 10 ** (
                    (7.5 * Texh[i - 1, j - 1]) / (Texh[i - 1, j - 1] + 237.3)
                ) * 100
                # Water vapour density exhaled air (kg m-3)
                RHOVPOUT[i - 1, j - 1] = VPSATAIROUT[i - 1, j - 1] / (
                    Rwater * (Texh[i - 1, j - 1] + CtoK)
                )
                # Dry air density exhaled air (kg m-3)
                RHODAIROUT[i - 1, j - 1] = (P - VPSATAIROUT[i - 1, j - 1]) / (
                    Rdair * (Texh[i - 1, j - 1] + CtoK)
                )
                # Air density exhaled air (kg m-3)
                RHOAIROUT[i - 1, j - 1] = RHOVPOUT[i - 1, j - 1] + RHODAIROUT[i - 1, j - 1]
                # Water vapour density exhaled air (kg kg-1)
                CHIAIROUT[i - 1, j - 1] = RHOVPOUT[i - 1, j - 1] * RHOAIROUT[i - 1, j - 1]
                # Air exchange between the animal and its environment (kg air m-2 day-1)
                AIREXCH[i - 1, j - 1] = (
                    irv[i - 1, j - 1] * 60 * 24 / 1000 * RHOAIR[i - 1, j - 1]
                ) / AREA[i - 1, j - 1]
                # Latent heat release from respiration (W m-2)
                LHEATRESP[i - 1, j - 1] = (
                    AIREXCH[i - 1, j - 1] * L *
                    (CHIAIROUT[i - 1, j - 1] - CHIAIR[i - 1, j - 1]) * kJdaytoW
                )
                # Convective heat release from respiration (W m-2)
                CHEATRESP[i - 1, j - 1] = (
                    AIREXCH[i - 1, j - 1] * Cp *
                    (Texh[i - 1, j - 1] - TAVGC[i - 1, j - 1]) * kJdaytoW
                )
                # Gross heat loss from the respiratory system (W m-2)
                # This equation corresponds to Eq. 8 of the Supplementary Information
                TGRESP[i - 1, j - 1] = LHEATRESP[i - 1, j - 1] + CHEATRESP[i - 1, j - 1]

                # NE required for respiration (W m-2), i.e. panting, McGovern and Bruce (2000)
                # This equation is similar to Eq. 9 of the Supplementary Information
                NERESPWM[i - 1, j - 1] = 1.1 * (RESPINCR * brr[i - 1, j - 1]) ** 2.78 * 10 ** -5 * PANTING
                # NE required for respiration (kJ NE day-1)
                NERESP[i - 1, j - 1] = NERESPWM[i - 1, j - 1] / kJdaytoW
                # Total heat loss from the respiratory system (W m-2)
                TNRESP[i - 1, j - 1] = TGRESP[i - 1, j - 1] - NERESPWM[i - 1, j - 1]

                TNRESPH[i - 1, j - 1] = TNRESP[i - 1, j - 1]
                #########################################################################################
                #                                    1a. Skin temperature                               #
                #########################################################################################

                # Minimum conductance between body core to skin (W m-2 K-1), McGovern and Bruce (2000)
                # This equation corresponds to Eq. 10 of the Supplementary Information
                CBSMIN[i - 1, j - 1] = RBCSf / (
                    (MINCCS1 * SENSMAT[88, s - 1]) *
                    TBWBF[i - 1, j - 1] ** (MINCCS2 * SENSMAT[89, s - 1])
                )
                # Conduction body core to skin (W m-2 K-1), McGovern and Bruce (2000)
                CONDBS[i - 1, j - 1] = CBSMIN[i - 1, j - 1] + TISSUEFRAC * (
                    CBSMAX - CBSMIN[i - 1, j - 1]
                )

                # Notes:
                # 100 s m-1 = 0.078 K m2 W-1 (Cena and Clark, 1978)
                # Cattle --> 50 s m-1 (Turnpenny, 2000a) --> 0.039 K m-2 W-1  = 25.6 W m-2 K-1

                #########################################################################################
                #                            2. Latent heat release from the skin                       #
                #########################################################################################

                # Reduction in coat depth (m), McGovern and Bruce (2000)
                DLC[i - 1, j - 1] = (
                    CoatConst * WEATHER.loc[i - 1, "WIND"]
                ) / (
                    (CoatConst * WEATHER.loc[i - 1, "WIND"]) / LC + 1 / (ZC * LC)
                )
                # Diffusion constant water vapour in air (m2 s-1), (Denny, 1993)
                DIFFC[i - 1, j - 1] = 0.187 * 10 ** -9 * TAVGK[i - 1, j - 1] ** 2.072

                #########################################################################################
                #                                    2a. Coat temperature                               #
                #########################################################################################

                # Resistance and conductivity between skin and coat
                # Conductance skin to coat (W m-2 K-1)
                # This equation corresponds to Eq. 14 of the Supplementary Information
                CSC[i - 1, j - 1] = 1 / (RUC * ZC * (LC - DLC[i - 1, j - 1]))
                # Increase in conductance due to precipitation/rain, Mount and Brown (1982)
                CSC[i - 1, j - 1] = CSC[i - 1, j - 1] / (
                    1 - min(RAINFRAC, WEATHER.loc[i - 1, "RAIN"] * RAINFRAC / 24)
                )

                #########################################################################################
                #                            3. Long wave radiation from the coat                       #
                #########################################################################################

                # Incoming LWR from the sky (W m-2), McGovern and Bruce (2000)
                # This equation corresponds to Eq. 19 of the Supplementary Information
                #print("LOG: WEATHER: ",WEATHER.columns.tolist())
                LWRSKY[i - 1, j - 1] = (
                    (1 - WEATHER.loc[i - 1, "OKTA"] / 8) *
                    (SIGMA * TAVGK[i - 1, j - 1] ** 4) *
                    (1 - 0.261 * np.exp(-0.000777 * (273 - TAVGK[i - 1, j - 1]) ** 2))
                    +
                    (WEATHER.loc[i - 1, "OKTA"] / 8) *
                    (SIGMA * TAVGK[i - 1, j - 1] ** 4 - 9)
                )
                # Incoming LWR from soil surface (W m-2)
                LWRENV[i - 1, j - 1] = SIGMA * TAVGK[i - 1, j - 1] ** 4

                # Cattle in stables do not receive LWR from the sky, but from the roofs and wall of the
                # stable they are housed in.
                if HOUSING[i - 1] == 0:
                    LWRSKY[i - 1, j - 1] = LWRENV[i - 1, j - 1]
########################## line:2469 ##############################
                #########################################################################################
                #                          4. Convective heat losses from the coat                      #
                #########################################################################################

                # Calculation of the air viscosity, Smits and Dussaunge (2006)
                # Average air temperature in degrees Rankine
                TAVGR[i - 1, j - 1] = TAVGK[i - 1, j - 1] * KtoR
                # Actual air viscosity (N s-1 m-2), Smits and Dussaunge (2006)
                VISCAIR[i - 1, j - 1] = (
                    MuSt * (
                        ((0.555 * TR0 + ST) / (0.555 * TAVGR[i - 1, j - 1] + ST)) *
                        (TAVGR[i - 1, j - 1] / TR0) ** (3 / 2)
                    )
                )

                # Calculation of the Grashof number
                # Vapour pressure of the ambient air (mBar)
                Ea[i - 1, j - 1] = WEATHER.loc[i - 1, "VPR"] * 10

                # Calculation of the Reynolds number
                # Wind speed (m s-1)
                WINDSP[i - 1, j - 1] = WEATHER.loc[i - 1, "WIND"]
                # Reynolds number
                REYNOLDS[i - 1, j - 1] = (
                    WINDSP[i - 1, j - 1] * DIAMETER[i - 1, j - 1] *
                    RHOAIR[i - 1, j - 1] / VISCAIR[i - 1, j - 1]
                )

                # Calculation step for natural convection
                ReH[i - 1, j - 1] = 16 * REYNOLDS[i - 1, j - 1] ** 2
                # Calculation step for forced convection
                ReL[i - 1, j - 1] = 0.1 * REYNOLDS[i - 1, j - 1] ** 2
                # Themal conductance of the ambient air (W m-1 K-1)
                # This equation corresponds to Eq. 17 of the Supplementary Information
                ka[i - 1, j - 1] = (
                    1.5207 * 10 ** (-11) * TAVGK[i - 1, j - 1] ** 3
                    - 4.8574 * 10 ** (-8) * TAVGK[i - 1, j - 1] ** 2
                    + 1.0184 * 10 ** (-4) * TAVGK[i - 1, j - 1]
                    - 0.00039333
                )

                #########################################################################################
                #                        5. Solar radiation intercepted by the coat                     #
                #########################################################################################

                # Ah/A factor: Shade area / animal coat area (m2 m-2)
                SAAC[i - 1, j - 1] = WEATHER.loc[i - 1, "AHA"]
                # Incoming direct solar radiation on the soil surface (Wm-2)
                SWRS[i - 1, j - 1] = WEATHER.loc[i - 1, "RAD"] * kJdaytoW
                # Incoming direct solar radiation on the animal's coat (Wm-2)
                # This equation covers part of Eq. 4 of the Supplementary Information
                SWRC[i - 1, j - 1] = SWRS[i - 1, j - 1] * SAAC[i - 1, j - 1] * (1 - REFLC)

                # Indirect solar radiation
                if HOUSING[i - 1] == 1:
                    REFLE[i - 1] = REFLEgrass
                else:
                    if HOUSING[i - 1] == 2:
                        REFLE[i - 1] = REFLEconcr
                    else:
                        REFLE[i - 1] = 0
                # Incoming indirect solar radiation on an animal's coat (W m-2)
                # This equation covers part of Eq. 4 of the Supplementary Information
                ISWRC[i - 1, j - 1] = FRACVEG * REFLE[i - 1] * SWRS[i - 1, j - 1]

                # Total solar radiation on an animal's coat (W m-2)
                SWR[i - 1, j - 1] = SWRC[i - 1, j - 1] + ISWRC[i - 1, j - 1]

                # Heat loss by evaporation of (rain) water (W m-2)
                # This equation corresponds to Eq. 20 of the Supplementary Information
                RAINEVAP[i - 1, j - 1] = (
                    (RAINEVAP1 * SENSMAT[91, s - 1]) *
                    (LENGTH[i - 1, j - 1] * DIAMETER[i - 1, j - 1]) /
                    AREA[i - 1, j - 1] *
                    min(24, WEATHER.loc[i - 1, "RAIN"]) * L * kJdaytoW
                )

                # Selects an initial maximum level for heat release and heat production (W m-2)
                METABFEED[i - 1, j - 1] = 170
                #print("LOG 2643: befor enter in WHILE")
                while True:
                    # start repeat loop for maximum heat release

                    #####################################################################################
                    #                                    1a. Skin temperature                           #
                    #####################################################################################

                    # Heat transfer from body core to skin (W m-2)
                    MetheatSKIN[i - 1, j - 1] = METABFEED[i - 1, j - 1] - TNRESP[i - 1, j - 1]
                    # Skin temperature (degrees Celsius)
                    # This equation corresponds to Eq. 11 of the Supplementary Information
                    TskinC[i - 1, j - 1] = (
                        TbodyC - MetheatSKIN[i - 1, j - 1] / CONDBS[i - 1, j - 1]
                    )

                    METABFEEDCH[i - 1, j - 1] = METABFEED[i - 1, j - 1]

                    #####################################################################################
                    #                            2. Latent heat release from the skin                   #
                    #####################################################################################

                    # Maximum physological latent heat release from skin (W m-2)
                    # This equation corresponds to Eq. 12 of the Supplementary Information
                    LASMAXPHYS[i - 1, j - 1] = (
                        LASMIN +
                        LIBRARY[23] * np.exp(
                            LIBRARY[24] * (TskinC[i - 1, j - 1] - LIBRARY[25])
                        ) * L / 3600
                    )

                    # Resistance vapour transfer (s m-1), Thompson et al. (2011)
                    RV[i - 1, j - 1] = (
                        (LC - DLC[i - 1, j - 1]) /
                        (
                            DIFFC[i - 1, j - 1] *
                            (
                                1 + 1.54 * ((LC - DLC[i - 1, j - 1]) / DIAMETER[i - 1, j - 1]) *
                                (TskinC[i - 1, j - 1] - min(
                                    TAVGC[i - 1, j - 1], TskinC[i - 1, j - 1]
                                )) ** 0.7
                            )
                        )
                    )
                    # Saturated vapour pressure skin (Pa)
                    VPSKINTOT[i - 1, j - 1] = 6.1078 * 10 ** (
                        (7.5 * TskinC[i - 1, j - 1]) / (TskinC[i - 1, j - 1] + 237.3)
                    ) * 100

                    # Maximum latent heat release from skin due to the ambient environment (W m-2)
                    # This equation corresponds to Eq. 13 of the Supplementary Information
                    LASMAXENV[i - 1, j - 1] = (
                        (RHOAIR[i - 1, j - 1] * Cp * 1000) / GAMMA *
                        (VPSKINTOT[i - 1, j - 1] - VPAIRTOT[i - 1, j - 1]) /
                        RV[i - 1, j - 1]
                    )
                    # Maximum latent heat release from skin (W m-2)
                    LASMAXCORR[i - 1, j - 1] = min(
                        LASMAXPHYS[i - 1, j - 1],
                        LASMAXENV[i - 1, j - 1]
                    )
                    # Actual latent heat release from skin (W m-2)
                    ACTSW[i - 1, j - 1] = (
                        LASMIN + LHRskin * (LASMAXCORR[i - 1, j - 1] - LASMIN)
                    )

                    ACTSWH[i - 1, j - 1] = ACTSW[i - 1, j - 1]

                    #####################################################################################
                    #                                    2a. Coat temperature                           #
                    #####################################################################################

                    # Heat transfer from the skin to the coat (W m-2)
                    MetheatCOAT[i - 1, j - 1] = (
                        MetheatSKIN[i - 1, j - 1] - ACTSW[i - 1, j - 1]
                    )

                    # Coat temperature (degrees Celsius)
                    # This equation corresponds to Eq. 15 of the Supplementary Information
                    TcoatC[i - 1, j - 1] = (
                        TskinC[i - 1, j - 1] - MetheatCOAT[i - 1, j - 1] / CSC[i - 1, j - 1]
                    )
                    # Coat temperature (degrees Kelvin)
                    TcoatK[i - 1, j - 1] = TcoatC[i - 1, j - 1] + CtoK
########################## line:2596 ##############################
                    #####################################################################################
                    #                            3. Long wave radiation from the coat                   #
                    #####################################################################################

                    # LWR release from the coat (W m-2)
                    LB[i - 1, j - 1] = EMISS * SIGMA * TcoatK[i - 1, j - 1] ** 4

                    # LWR balance (W m-2) (net energy loss is a negative value!)
                    # This equation corresponds partly to Eq. 18 of the Supplementary Information
                    LWRCOAT[i - 1, j - 1] = (
                        (EMISS * ((LWRSKY[i - 1, j - 1] + LWRENV[i - 1, j - 1]) / 2) - LB[i - 1, j - 1]) *
                        (1 - min(RAINFRAC, WEATHER.loc[i - 1, "RAIN"] * RAINFRAC / 24))
                    )

                    LWRCOATH[i - 1, j - 1] = LWRCOAT[i - 1, j - 1]

                    #####################################################################################
                    #                        4. Convective heat losses from the coat                    #
                    #####################################################################################

                    # Vapour pressure at the skin (mBar)
                    Ec[i - 1, j - 1] = (
                        (6.1078 * 10 ** (
                            (7.5 * TskinC[i - 1, j - 1]) / (TskinC[i - 1, j - 1] + 237.3)
                        )) + Ea[i - 1, j - 1]
                    ) / 2

                    # Grashof number
                    GRASHOF[i - 1, j - 1] = (
                        GRAV * DIAMETER[i - 1, j - 1] ** 3 * P / 100 *
                        (TcoatC[i - 1, j - 1] - TAVGC[i - 1, j - 1]) +
                        Schmidt * (
                            Ec[i - 1, j - 1] * TcoatC[i - 1, j - 1] -
                            Ea[i - 1, j - 1] * TAVGC[i - 1, j - 1]
                        )
                    ) / (273 * P / 100 * VISCAIR[i - 1, j - 1] ** 2)

                    # Calculation of the Nusselt number, Turnpenny et al. (2000a)
                    if GRASHOF[i - 1, j - 1] > ReH[i - 1, j - 1]:
                        NUSSELT[i - 1, j - 1] = 0.48 * GRASHOF[i - 1, j - 1] ** 0.25
                    else:
                        if GRASHOF[i - 1, j - 1] < ReL[i - 1, j - 1]:
                            NUSSELT[i - 1, j - 1] = 0.0112 * REYNOLDS[i - 1, j - 1] ** 0.875
                        else:
                            NUSSELT[i - 1, j - 1] = max(
                                0.48 * GRASHOF[i - 1, j - 1] ** 0.25,
                                0.0112 * REYNOLDS[i - 1, j - 1] ** 0.875
                            )

                    # Heat transfer from the coat by convection (W m-2)
                    # This equation corresponds partly to Eq. 16 of the Supplementary Information
                    CONVCOAT[i - 1, j - 1] = (
                        (ka[i - 1, j - 1] * NUSSELT[i - 1, j - 1]) / DIAMETER[i - 1, j - 1] *
                        (TcoatC[i - 1, j - 1] - TAVGC[i - 1, j - 1]) /
                        (1 - min(RAINFRAC, WEATHER.loc[i - 1, "RAIN"] * RAINFRAC / 24))
                    )
                    CONVCOATH[i - 1, j - 1] = CONVCOAT[i - 1, j - 1]

                    #####################################################################################
                    #                                         Synthesis                                 #
                    #####################################################################################

                    # Heat balance (W m-2)
                    # This equation is similar to Eq. 1 of the Supplementary Information
                    MetheatBAL[i - 1, j - 1] = (
                        MetheatCOAT[i - 1, j - 1] + SWR[i - 1, j - 1] - RAINEVAP[i - 1, j - 1] +
                        LWRCOAT[i - 1, j - 1] - CONVCOAT[i - 1, j - 1]
                    )

                    # If the estimate for heat production is too high, it is decreased
                    if MetheatBAL[i - 1, j - 1] > 0.1:
                        METABFEED[i - 1, j - 1] = (
                            METABFEED[i - 1, j - 1] - 0.1 * MetheatBAL[i - 1, j - 1]
                        )
                    # If the estimate for heat production is too low, it is increased
                    if MetheatBAL[i - 1, j - 1] < -0.1:
                        METABFEED[i - 1, j - 1] = (
                            METABFEED[i - 1, j - 1] - 0.1 * MetheatBAL[i - 1, j - 1]
                        )
                    # Heat release and production
                    Metheatopt[i - 1, j - 1] = METABFEED[i - 1, j - 1]
                    # If the heat release and production differ more than 0.1 W m-2, the loop is run
                    # another time
                    if (MetheatBAL[i - 1, j - 1] < 0.1 and
                            MetheatBAL[i - 1, j - 1] > -0.1):
                        CHECKHEAT1[i - 1, j - 1] = "CORRECT"
                    else:
                        CHECKHEAT1[i - 1, j - 1] = "FALSE"
                    if CHECKHEAT1[i - 1, j - 1] == "CORRECT":
                        break

                # end repeat loop for maximum heat release

                ###########################################################################################
                # 2.1.2                             Minimum heat release                                  #
                ###########################################################################################

                # Heat release mechanisms of cattle at minimum heat release
                TISSUEFRAC = 0.0            # Vasodilatation (0 = minimum and 1 = maximum vasodilatation)
                LHRskin = 0.0               # Latent heat release (0 = minimum and 1 = maximum
                # physiological 'sweating' rate)
                PANTING = 0.0               # Panting (0 = basal respiration, 1 is maximum panting)

                #########################################################################################
                #                1. Latent and convective heat release from respiration                 #
                #########################################################################################

                # Actual respiration rate (L min-1)
                irv[i - 1, j - 1] = brv[i - 1, j - 1] + PANTING * ((RESPINCR - 1) * brv[i - 1, j - 1])
                # Air exchange between the animal and its environment (kg air m-2 day-1)
                AIREXCH[i - 1, j - 1] = (
                    irv[i - 1, j - 1] * 60 * 24 / 1000 * RHOAIR[i - 1, j - 1]
                ) / AREA[i - 1, j - 1]
                # Latent heat release via respiratory system (W m-2)
                LHEATRESP[i - 1, j - 1] = (
                    AIREXCH[i - 1, j - 1] * L *
                    (CHIAIROUT[i - 1, j - 1] - CHIAIR[i - 1, j - 1]) *
                    kJdaytoW
                )
                # Concective heat release via respiratory system (W m-2)
                CHEATRESP[i - 1, j - 1] = (
                    AIREXCH[i - 1, j - 1] * Cp *
                    (Texh[i - 1, j - 1] - TAVGC[i - 1, j - 1]) *
                    kJdaytoW
                )
                # Gross heat loss from the respiratory system (W m-2)
                TGRESP[i - 1, j - 1] = LHEATRESP[i - 1, j - 1] + CHEATRESP[i - 1, j - 1]
                # Total heat loss from the respiratory system (W m-2)
                TNRESP[i - 1, j - 1] = TGRESP[i - 1, j - 1]

                #########################################################################################
                #                                    1a. Skin temperature                               #
                #########################################################################################

                # Conductance body core to skin (W m-2 K-1)
                CONDBS[i - 1, j - 1] = CBSMIN[i - 1, j - 1]

                #########################################################################################
                #                            2. Latent heat release from the skin                       #
                #########################################################################################

                # Actual latent heat release from the skin (W m-2)
                ACTSW[i - 1, j - 1] = LASMIN

                # Selects an initial minimum level for heat release and heat production (W m-2)
                METABFEEDC[i - 1, j - 1] = 80
########################## line:2718 ##############################
                #print("LOG 2875: befor enter in WHILE")
                while True:
                    # start repeat loop for minimum heat release

                    #######################################################################################
                    #                                    1a. Skin temperature                             #
                    #######################################################################################

                    # Heat transfer from the body core to skin (W m-2)
                    MetheatSKIN[i - 1, j - 1] = METABFEEDC[i - 1, j - 1] - TNRESP[i - 1, j - 1]
                    # Skin temperature (degrees Celsius)
                    TskinC[i - 1, j - 1] = TbodyC - MetheatSKIN[i - 1, j - 1] / CONDBS[i - 1, j - 1]

                    TskinCH[i - 1, j - 1] = TskinC[i - 1, j - 1]

                    #######################################################################################
                    #                                    2a. Coat temperature                             #
                    #######################################################################################

                    # Heat transfoer from skin to coat (W m-2)
                    MetheatCOAT[i - 1, j - 1] = MetheatSKIN[i - 1, j - 1] - ACTSW[i - 1, j - 1]

                    # Coat temperature (degrees Celsius)
                    TcoatC[i - 1, j - 1] = TskinC[i - 1, j - 1] - MetheatCOAT[i - 1, j - 1] / CSC[i - 1, j - 1]
                    # Coat temperature (degrees Kelvin)
                    TcoatK[i - 1, j - 1] = TcoatC[i - 1, j - 1] + CtoK

                    TcoatCH[i - 1, j - 1] = TcoatC[i - 1, j - 1]

                    #######################################################################################
                    #                            3. Long wave radiation from the coat                     #
                    #######################################################################################

                    # LWR release from the coat (W m-2)
                    LB[i - 1, j - 1] = EMISS * SIGMA * TcoatK[i - 1, j - 1] ** 4

                    # LWR from coat to the environment (net energy loss is a negative value) (W m-2)
                    LWRCOAT[i - 1, j - 1] = (
                        (EMISS * ((LWRSKY[i - 1, j - 1] + LWRENV[i - 1, j - 1]) / 2) - LB[i - 1, j - 1]) *
                        (1 - min(RAINFRAC, WEATHER.loc[i - 1, "RAIN"] * RAINFRAC / 24))
                    )

                    #######################################################################################
                    #                          4. Convective heat loss from the coat                      #
                    #######################################################################################

                    # Vapour pressure at the coat (mBar)
                    Ec[i - 1, j - 1] = (
                        (6.1078 * 10 ** ((7.5 * TcoatC[i - 1, j - 1]) / (TcoatC[i - 1, j - 1] + 237.3)))
                        + Ea[i - 1, j - 1]
                    ) / 2

                    # Grashof number
                    GRASHOF[i - 1, j - 1] = (
                        GRAV * DIAMETER[i - 1, j - 1] ** 3 * P / 100 * (TcoatC[i - 1, j - 1] - TAVGC[i - 1, j - 1])
                        + Schmidt * (
                            Ec[i - 1, j - 1] * TcoatC[i - 1, j - 1] -
                            Ea[i - 1, j - 1] * TAVGC[i - 1, j - 1]
                        )
                    ) / (273 * P / 100 * VISCAIR[i - 1, j - 1] ** 2)

                    # Calculation of the Nusselt number, Turnpenny et al. (2000a)
                    if GRASHOF[i - 1, j - 1] > ReH[i - 1, j - 1]:
                        NUSSELT[i - 1, j - 1] = 0.48 * GRASHOF[i - 1, j - 1] ** 0.25
                    else:
                        if GRASHOF[i - 1, j - 1] < ReL[i - 1, j - 1]:
                            NUSSELT[i - 1, j - 1] = 0.0112 * REYNOLDS[i - 1, j - 1] ** 0.875
                        else:
                            NUSSELT[i - 1, j - 1] = max(
                                0.48 * GRASHOF[i - 1, j - 1] ** 0.25,
                                0.0112 * REYNOLDS[i - 1, j - 1] ** 0.875
                            )

                    # Convective heat transfer between coat and air (W m-2)
                    CONVCOAT[i - 1, j - 1] = (
                        (ka[i - 1, j - 1] * NUSSELT[i - 1, j - 1]) / DIAMETER[i - 1, j - 1] *
                        (TcoatC[i - 1, j - 1] - TAVGC[i - 1, j - 1]) /
                        (1 - min(RAINFRAC, WEATHER.loc[i - 1, "RAIN"] * RAINFRAC / 24))
                    )

                    #######################################################################################
                    #                                         Synthesis                                   #
                    #######################################################################################

                    # Heat balance (W m-2)
                    # This equation is similar to Eq. 1 of the Supplementary Information
                    MetheatBAL[i - 1, j - 1] = (
                        MetheatCOAT[i - 1, j - 1] + SWR[i - 1, j - 1] - RAINEVAP[i - 1, j - 1] +
                        LWRCOAT[i - 1, j - 1] - CONVCOAT[i - 1, j - 1]
                    )

                    # If the estimate for heat production is too high, it is decreased
                    if MetheatBAL[i - 1, j - 1] > 0.1:
                        METABFEEDC[i - 1, j - 1] = METABFEEDC[i - 1, j - 1] - 0.05 * MetheatBAL[i - 1, j - 1]
                    # If the estimate for heat production is too low, it is increased
                    if MetheatBAL[i - 1, j - 1] < -0.1:
                        METABFEEDC[i - 1, j - 1] = METABFEEDC[i - 1, j - 1] - 0.05 * MetheatBAL[i - 1, j - 1]
                    # Heat release and production
                    Metheatcold[i - 1, j - 1] = METABFEEDC[i - 1, j - 1]
                    # If the heat release and production differ more than 0.1 W m-2, the loop is run
                    # another time
                    if (MetheatBAL[i - 1, j - 1] < 0.1 and
                            MetheatBAL[i - 1, j - 1] > -0.1):
                        CHECKHEAT2[i - 1, j - 1] = "CORRECT"
                    else:
                        CHECKHEAT2[i - 1, j - 1] = "FALSE"
                    if CHECKHEAT2[i - 1, j - 1] == "CORRECT":
                        break
########################## line:2815 ##############################
                #########################################################################################
                # 2.2                     Energy and protein utilisation sub-model                      #
                #########################################################################################

                ######################
                # Growth (potential) #
                ######################

                # TBW and carcass weight
                # Gompertz curve with breed specific parameters, total body weight (TBW) (kg live
                # weight per animal)
                # This equation corresponds to Eq. 31 in the Supplementary Information
                TBW[i, j - 1] = (
                    BIRTHW + (MAXW - BIRTHW) * np.exp(-CPAR * np.exp(TIME[i - 1, j - 1] / 365 * -DPAR))
                ) - EPAR
                # Carcass weight (kg per animal)
                # This equation corresponds to Eq. 32 in the Supplementary Information
                CARCW[i, j - 1] = (
                    TBW[i, j - 1] * INCARC +
                    TBW[i, j - 1] * (LIBRARY[20] - INCARC) *
                    (TBW[i, j - 1] - BIRTHW) / (MAXW1[j - 1] - BIRTHW)
                )
                # Records the highest average daily gain (ADG) of an animal during its lifespan (kg LW
                # day-1)
                ADGHIGH[i, j - 1] = max(ADGHIGH[i - 1, j - 1], TBW[i, j - 1] - TBW[i - 1, j - 1])

                # Bone and muscle weight
                # Derivative potential growth bone tissue (kg day-1), according to Gompertz curve
                # This equation contains Eq. 33 of the Supplementary Information
                DERBONE[i - 1, j - 1] = (
                    CARCW[i, j - 1] * min(
                        BONEFRACMAX,
                        (BONEGROWTH1 * SENSMAT[67, s - 1]) *
                        CARCW[i, j - 1] ** -(BONEGROWTH2 * SENSMAT[68, s - 1])
                    )
                    - CARCW[i - 1, j - 1] * min(
                        BONEFRACMAX,
                        (BONEGROWTH1 * SENSMAT[67, s - 1]) *
                        CARCW[i - 1, j - 1] ** -(BONEGROWTH2 * SENSMAT[68, s - 1])
                    )
                )
                # Derivative potential growth muscle tissue (kg day-1)
                # This equation contains Eqs 34 and 35 of the Supplementary Information
                DERMUSCLE[i - 1, j - 1] = DERBONE[i - 1, j - 1] * min(
                    LIBRARY[21],
                    (MUSCLEGROWTH1 * SENSMAT[69, s - 1]) *
                    CARCW[i, j - 1] ** 2 + LIBRARY[21] / 100 * CARCW[i, j - 1] +
                    (MUSCLEGROWTH2 * SENSMAT[70, s - 1])
                )
                # Weight bone tissue (kg per animal)
                # Note: new state = previous state + rate of change over a time step
                BONETIS[i, j - 1] = BONETIS[i - 1, j - 1] + DERBONE[i - 1, j - 1]
                # Weight muscle tissue (kg per animal)
                MUSCLETIS[i, j - 1] = MUSCLETIS[i - 1, j - 1] + DERMUSCLE[i - 1, j - 1]

                # Intramuscular fat, miscellaneous fat and non-carcass tissues
                # Derivative potential growth intramuscular fat (kg day-1)
                # This equation contains Eq. 36 of the Supplementary Information
                DERINTRAMF[i - 1, j - 1] = (
                    (IMFGROWTH1 * SENSMAT[71, s - 1]) *
                    (BONETIS[i, j - 1] + MUSCLETIS[i, j - 1]) ** 2 +
                    (IMFGROWTH2 * SENSMAT[72, s - 1]) *
                    (BONETIS[i, j - 1] + MUSCLETIS[i, j - 1]) -
                    (IMFGROWTH3 * SENSMAT[73, s - 1])
                ) - (
                    (IMFGROWTH1 * SENSMAT[71, s - 1]) *
                    (BONETIS[i - 1, j - 1] + MUSCLETIS[i - 1, j - 1]) ** 2 +
                    (IMFGROWTH2 * SENSMAT[72, s - 1]) *
                    (BONETIS[i - 1, j - 1] + MUSCLETIS[i - 1, j - 1]) -
                    (IMFGROWTH3 * SENSMAT[73, s - 1])
                )
                # Derivative potential growth subcutaneous and intermuscular (i.e. miscellaneous) fat
                # tissue (kg day-1)
                # This equation corresponds to Eq. 37 of the Supplementary Information
                DERMISCFAT[i - 1, j - 1] = (
                    (CARCW[i, j - 1] - CARCW[i - 1, j - 1]) -
                    DERBONE[i - 1, j - 1] - DERMUSCLE[i - 1, j - 1] -
                    DERINTRAMF[i - 1, j - 1]
                )
                # Derivative potential growth non carcass tissue (kg day-1)
                DERNONC[i - 1, j - 1] = (
                    (TBW[i, j - 1] * (1 - RUMENFRAC) - TBW[i - 1, j - 1] * (1 - RUMENFRAC)) -
                    (CARCW[i, j - 1] - CARCW[i - 1, j - 1])
                )
                # Derivative potential growth of the rumen (kg day-1)
                # This equation is similar to Eq. 39 of the Supplementary Information
                DERRUMEN[i - 1, j - 1] = TBW[i, j - 1] * RUMENFRAC - TBW[i - 1, j - 1] * RUMENFRAC
                # Derivative potential growth of all body tissues (kg day-1)
                DERTOTAL[i - 1, j - 1] = (
                    DERBONE[i - 1, j - 1] + DERMUSCLE[i - 1, j - 1] + DERINTRAMF[i - 1, j - 1] +
                    DERMISCFAT[i - 1, j - 1] + DERNONC[i - 1, j - 1] + DERRUMEN[i - 1, j - 1]
                )

                # Weight intramuscular fat tissue (kg per animal)
                INTRAMFTIS[i, j - 1] = INTRAMFTIS[i - 1, j - 1] + DERINTRAMF[i - 1, j - 1]
                # Weight miscellaneous fat tissue (kg per animal)
                MISCFATTIS[i, j - 1] = MISCFATTIS[i - 1, j - 1] + DERMISCFAT[i - 1, j - 1]
                # Weight non-carcass tissue (kg per animal)
                NONCARCTIS[i, j - 1] = NONCARCTIS[i - 1, j - 1] + DERNONC[i - 1, j - 1]
                # Weight rumen content (kg per animal)
                RUMEN[i, j - 1] = RUMEN[i - 1, j - 1] + DERRUMEN[i - 1, j - 1]

                # Check: sum of tissues and rumen content must correspond to the TBW (kg per animal)
                TBWCHECK[i, j - 1] = (
                    BONETIS[i, j - 1] + MUSCLETIS[i, j - 1] + INTRAMFTIS[i, j - 1] +
                    MISCFATTIS[i, j - 1] + NONCARCTIS[i, j - 1] + RUMEN[i, j - 1]
                )

                # Fraction lipid in bone tissue (-)
                # This equation corresponds to Eq. 40 of the Supplementary Information
                LIPIDFRACBONE[i - 1, j - 1] = (LIBRARY[19] * np.log10(BONETIS[i - 1, j - 1])) / 100
                # Fraction lipid accreted in new non-carcass tissue (-)
                # This equation corresponds to Eq. 41 of the Supplementary Information
                LIPIDFRACNONC[i - 1, j - 1] = min(
                    LIPNONCMAX,
                    max(
                        LIPNONCMIN,
                        LIPNONCMIN + (
                            NONCARCTIS[i - 1, j - 1] /
                            (LIBRARY[12] * (1 - RUMENFRAC) * (1 - LIBRARY[20]))
                        ) ** 2 * LIPNONCMAX
                    )
                )
                # Fraction protein accreted in new non-carcass tissue (-)
                # This equation corresponds to Eq. 42 of the Supplementary Information
                PROTFRACNONC[i - 1, j - 1] = (
                    (PROTNONCM1 * SENSMAT[74, s - 1]) * TBW[i - 1, j - 1] +
                    (PROTNONCM2 * SENSMAT[75, s - 1])
                ) / 100

                # Weight of lipids and protein in body tissues
                # Note: new state = previous state + rate of change over one time step
                # Weight of lipids in bone tissue (kg per animal)
                LIPIDBONE[i, j - 1] = LIPIDBONE[i - 1, j - 1] + DERBONE[i - 1, j - 1] * LIPIDFRACBONE[i - 1, j - 1]
                # Weight of protein in bone tissue (kg per animal)
                PROTBONE[i, j - 1] = PROTBONE[i - 1, j - 1] + DERBONE[i - 1, j - 1] * PROTFRACBONE
                # Energy (combustion energy + inefficiency) accreted in bone tissue (MJ per day)
                # Note: 53.5 MJ kg-1 for lipid, 44.0 MJ kg-1 gross energy for protein
                ENGRBONE[i, j - 1] = (
                    DERBONE[i - 1, j - 1] * LIPIDFRACBONE[i - 1, j - 1] * GELIPID / LIPIDEFF +
                    DERBONE[i - 1, j - 1] * PROTFRACBONE * GEPROT / PROTEFF
                )

                # Weight of lipids in bone tissue (kg per animal)
                LIPIDMUSCLE[i, j - 1] = LIPIDMUSCLE[i - 1, j - 1] + DERMUSCLE[i - 1, j - 1] * LIPFRACMUSCLE
                # Weight of protein in muscle tissue (kg per animal)
                PROTMUSCLE[i, j - 1] = PROTMUSCLE[i - 1, j - 1] + DERMUSCLE[i - 1, j - 1] * PROTFRACMUSCLE
                # Energy (combustion energy + inefficiency) accreted in muscle tissue (MJ per day)
                ENGRMUSCLE[i, j - 1] = (
                    DERMUSCLE[i - 1, j - 1] * LIPFRACMUSCLE * GELIPID / LIPIDEFF +
                    DERMUSCLE[i - 1, j - 1] * PROTFRACMUSCLE * GEPROT / PROTEFF
                )

                # Weight of lipids in the intermuscular fat tissue (kg per animal)
                LIPIDIMF[i, j - 1] = LIPIDIMF[i - 1, j - 1] + DERINTRAMF[i - 1, j - 1] * LIPFRACFAT
                # Weight of protein in the intermuscular fat tissue (kg per animal)
                PROTIMF[i, j - 1] = PROTIMF[i - 1, j - 1] + DERINTRAMF[i - 1, j - 1] * PROTFRACFAT
                # Energy (combustion energy + inefficiency) accreted in the intermuscular fat tissue
                # (MJ per day)
                ENGRIMF[i, j - 1] = (
                    DERINTRAMF[i - 1, j - 1] * LIPFRACFAT * GELIPID / LIPIDEFF +
                    DERINTRAMF[i - 1, j - 1] * PROTFRACFAT * GEPROT / PROTEFF
                )

                # Weight of lipids in the miscellaneous fat tissue (kg per animal)
                LIPIDFAT[i, j - 1] = LIPIDFAT[i - 1, j - 1] + DERMISCFAT[i - 1, j - 1] * LIPFRACFAT
                # Weight of protein in the miscellaneous fat tissue (kg per animal)
                PROTFAT[i, j - 1] = PROTFAT[i - 1, j - 1] + DERMISCFAT[i - 1, j - 1] * PROTFRACFAT
                # Energy (combustion energy + inefficiency) accreted in miscellaneous fat tissue
                # (MJ per day)
                ENGRFAT[i, j - 1] = (
                    DERMISCFAT[i - 1, j - 1] * LIPFRACFAT * GELIPID / LIPIDEFF +
                    DERINTRAMF[i - 1, j - 1] * PROTFRACFAT * GEPROT / PROTEFF
                )

                # Weight of lipids in the non-carcass tissue (kg per animal)
                LIPIDNONC[i, j - 1] = LIPIDNONC[i - 1, j - 1] + DERNONC[i - 1, j - 1] * LIPIDFRACNONC[i - 1, j - 1]
                # Weight of protein in the non-carcass tissue (kg per animal)
                PROTNONC[i, j - 1] = PROTNONC[i - 1, j - 1] + DERNONC[i - 1, j - 1] * PROTFRACNONC[i - 1, j - 1]
                # Energy (combustion energy + inefficiency) accreted in non-carcass tissue (MJ per day)
                ENGRNONC[i, j - 1] = (
                    DERNONC[i - 1, j - 1] * LIPIDFRACNONC[i - 1, j - 1] * GELIPID / LIPIDEFF +
                    DERNONC[i - 1, j - 1] * PROTFRACNONC[i - 1, j - 1] * GEPROT / PROTEFF
                )

                # Total net energy (NE) to realise potential growth (MJ per day)
                # This equation is similar to Eq. 43 of the Supplementary Information
                ENGRTOTAL[i, j - 1] = (
                    ENGRBONE[i, j - 1] + ENGRNONC[i, j - 1] + ENGRMUSCLE[i, j - 1] +
                    ENGRIMF[i, j - 1] + ENGRFAT[i, j - 1]
                )
                ENGRTOTALORIG[i, j - 1] = ENGRTOTAL[i, j - 1]

                # Records the highest NE for growth throughout an animals life span (MJ per day)
                if TIME[i - 1, j - 1] <= WEANINGTIME:
                    ENGRTOTALHIGH[i, j - 1] = 0
                else:
                    ENGRTOTALHIGH[i, j - 1] = max(
                        ENGRTOTALHIGH[i - 1, j - 1],
                        ENGRTOTAL[i, j - 1]
                    )
########################## line:2988 ##############################
                # Records the highest NE for growth throughout an animals life span, including
                # compensatory growth (MJ per day)
                # This equation corresponds to Eq. 45 of the Supplementary Information, and contains
                # Eq. 44.
                ENGRTOTALHIGH1[i, j - 1] = ENGRTOTALHIGH[i, j - 1] * min(
                    1.0,
                    (1 - (TBWBF[i - 1, j - 1] / TBW[i - 1, j - 1])) * COMPFACT
                )

                # Relative proportion between NE for growth with and without compensatory growth (-)
                REL[i, j - 1] = ENGRTOTALHIGH1[i, j - 1] / ENGRTOTAL[i, j - 1]
                if REL[i, j - 1] < 1:
                    REL[i, j - 1] = 1

                # The NE for growth based on the genetic potential and the scope for compensatory
                # growth (MJ per day)
                if TIME[i - 1, j - 1] <= WEANINGTIME:
                    ENGRTOTAL[i, j - 1] = ENGRTOTAL[i, j - 1]
                else:
                    ENGRTOTAL[i, j - 1] = max(
                        ENGRTOTAL[i, j - 1],
                        min(1.0, (1 - (TBWBF[i - 1, j - 1] / TBW[i - 1, j - 1])) * COMPFACT)
                        * ENGRTOTALHIGH[i, j - 1]
                    )
                # Percentage of lipid in the carcass (consists of bone tissue, muscle tissue, and fat
                # tissues) (%)

                LIPIDFRACCARC[i, j - 1] = (
                    LIPIDBONE[i, j - 1] + LIPIDMUSCLE[i, j - 1] +
                    LIPIDIMF[i, j - 1] + LIPIDFAT[i, j - 1]
                ) / (
                    TBW[i, j - 1] - NONCARCTIS[i, j - 1] - RUMEN[i, j - 1]
                ) * 100

                #########################################################################################
                # Maintenance #
                ###############

                # Energy balance
                # NE for (fasting) maintenance (kJ per animal per day)
                NEMAINT[i - 1, j - 1] = EBWBFMET[i - 1, j - 1] * NEm * LIBRARY[17]
                # NE for (fasting) maintenance (W m-2)
                NEMAINTWM[i - 1, j - 1] = (
                    NEMAINT[i - 1, j - 1] * 1000 / (3600 * 24 * AREA[i - 1, j - 1])
                )

                # Protein balance
                # Dermal loss of protein (CSIRO, 2007) (g protein day-1)
                PROTDERML[i - 1, j - 1] = DERMPL * EBWBFMET[i - 1, j - 1]
                # Protein requirement for (fasting) maintenance (CSIRO, 2007) (g protein day-1)
                # Note: PROTNE = 2g N / 4.18 = 0.478
                PROTMAINT[i - 1, j - 1] = NEMAINT[i - 1, j - 1] * PROTNE / 1000 * NtoCP

                #########################################################################################
                # Physical activity #
                #####################

                # Energy balance
                # NE for physical activity (kJ per animal per day), function of metabolic body weight
                # Note: If necessary, the line of code below can also be written as a function of the
                #       total body weight (TBW)
                if HOUSING[i - 1] >= 1:
                    NEPHYSACT[i - 1, j - 1] = EBWBFMET[i - 1, j - 1] * NEpha
                else:
                    NEPHYSACT[i - 1, j - 1] = 0
                # NE for physical activity (W m-2)
                NEPHYSACTWM[i - 1, j - 1] = (
                    NEPHYSACT[i - 1, j - 1] * 1000 / (3600 * 24 * AREA[i - 1, j - 1])
                )

                # Protein balance
                # Protein requirement for physical activity (g protein day-1)
                # Note: PROTNE = 2g N / 4.18 = 0.478
                PROTPHACT[i - 1, j - 1] = NEPHYSACT[i - 1, j - 1] * PROTNE / 1000 * NtoCP

                #########################################################################################
                # Gestation #
                #############

                # Cows can conceive if seven requirements are met (0= no conception, 1 = conception):

                # 1. The TBW must be higher than a specific fraction of their maximum adult TBW
                if TBWBF[i - 1, j - 1] < (LIBRARY[16] * MAXW1[j - 1]):
                    GEST1[i - 1, j - 1] = 0
                else:
                    GEST1[i - 1, j - 1] = 1
                # 2. Cows cannot conceive while gestating
                if CALFTBW[i - 1, j - 1] == 0:
                    GEST2[i - 1, j - 1] = 1
                else:
                    GEST2[i - 1, j - 1] = 0
                # 3. Cows cannot conceive directly after parturition (depending on the minimum
                # calving interval)
                prev_idx = max(0, i - int(GESTINTERVAL - GestPer - 1) - 1)
                if CALFTBW[i - 1, j - 1] - CALFTBW[prev_idx, j - 1] == 0:
                    GEST3[i - 1, j - 1] = 1
                else:
                    GEST3[i - 1, j - 1] = 0
                # 4. Cows can conceive if the fat tissues cover a certain fraction of the carcass
                # tissue
                if ((MISCFATTISBF[i - 1, j - 1] + INTRAMFTISBF[i - 1, j - 1]) /
                    (TBWBF[i - 1, j - 1] - NONCARCTISBF[i - 1, j - 1] - RUMEN[i - 1, j - 1]) < LIBRARY[18]):
                    GEST4[i - 1, j - 1] = 0
                else:
                    GEST4[i - 1, j - 1] = 1
                # 5. Cattle conceive at a specific date (STDOY), in case of seasonal calving
                # Note: for year-round calving, GEST5 must be 1
                if DOY[i - 1] == STDOY + (GESTINTERVAL - GestPer):
                    GEST5[i - 1, j - 1] = 1
                else:
                    GEST5[i - 1, j - 1] = 1
                # 6. Cows cannot conceive after their maximum age for conception
                if TIME[i - 1, j - 1] / 365 < MAXCONCAGE:
                    GEST5[i - 1, j - 1] = GEST5[i - 1, j - 1]
                else:
                    GEST5[i - 1, j - 1] = 0

                # 7. Cows cannot conceive anymore if the maximum number of calves per cow is achieved
                # If not eight calves after 10 years, reduce MAXCALFNR

                # orginal
                #if TIME[i - 1, j - 1] / 365 > MAXCONCAGE:
                #    MAXCALFNR = CALFNR[i - 1, j - 1]
                #               else:
                #   MAXCALFNR = MAXCALFNR

                if TIME[i - 1, j - 1] / 365 > MAXCONCAGE:
                    MAXCALFNR = int(CALFNR[i - 1, j - 1])
                else:
                    MAXCALFNR = MAXCALFNR
                
                if CALFNR[i - 1, j - 1] < MAXCALFNR:
                    GEST6[i, j - 1] = 1
                else:
                    GEST6[i, j - 1] = 0

                # Check whether a cow meets all seven conditions (0= no conception, 1 = conception)
                GEST[i - 1, j - 1] = (
                    GEST1[i - 1, j - 1] * GEST2[i - 1, j - 1] * GEST3[i - 1, j - 1] *
                    GEST4[i - 1, j - 1] * GEST5[i - 1, j - 1] * GEST6[i, j - 1] *
                    LIBRARY[13] * REPRODUCTIVE[j - 1]
                )

                # Counts number of calves (incl. gestation) per cow
                CALFNR[i, j - 1] = GEST[i - 1, j - 1] + CALFNR[i - 1, j - 1]

                # Gestation starts when all seven requirements are met
                if GEST[i - 1, j - 1] == 1:
                    GESTDAY[i, j - 1] = 1
                else:
                    if GESTDAY[i - 1, j - 1] > 0:
                        GESTDAY[i, j - 1] = GESTDAY[i - 1, j - 1] + 1
                    else:
                        GESTDAY[i, j - 1] = 0
                if GEST[i - 1, j - 1] == 1:
                    GESTDAY[i - 1, j - 1] = 0

                # Breed- and sex-specific birth weights (kg live weight)
                calf_idx = int(CALFNR[i - 1, j - 1])
                if BREED == 1:
                    BIRTHW1[i - 1, j - 1] = LIBRARY10[4] - SEX[calf_idx] * (LIBRARY10[4] - LIBRARY11[4])
                else:
                    if BREED == 2:
                        BIRTHW1[i - 1, j - 1] = LIBRARY20[4] - SEX[calf_idx] * (LIBRARY20[4] - LIBRARY21[4])
                    else:
                        if BREED == 3:
                            BIRTHW1[i - 1, j - 1] = LIBRARY30[4] - SEX[calf_idx] * (LIBRARY30[4] - LIBRARY31[4])
                        else:
                            if BREED == 4:
                                BIRTHW1[i - 1, j - 1] = LIBRARY40[4] - SEX[calf_idx] * (LIBRARY40[4] - LIBRARY41[4])
                            else:
                                if BREED == 5:
                                    BIRTHW1[i - 1, j - 1] = LIBRARY50[4] - SEX[calf_idx] * (LIBRARY50[4] - LIBRARY51[4])
                                if BREED == 6:
                                    BIRTHW1[i - 1, j - 1] = LIBRARY60[4] - SEX[calf_idx] * (LIBRARY60[4] - LIBRARY61[4])

                # Code for sensitivity analysis on birth weight (not used in this version)
                BIRTHW1[i - 1, j - 1] = BIRTHW1[i - 1, j - 1] * SENSMAT[45, s - 1]
########################## line:3152 ##############################
                # Net energy (NE) requirements for gestation, Fox et al. (1988)
                # This is an empirical equation.
                # This equation is similar to Eq. 28 and 29 in the Supplementary Information
                if (GESTDAY[i, j - 1] >= 1 and
                        GESTDAY[i, j - 1] <= GestPer):
                    NEREQGEST[i - 1, j - 1] = (
                        SENSMAT[92, s - 1] * (
                            (
                                9.527001 * (0.0000000681 - 0.000000000197 * GESTDAY[i - 1, j - 1]) *
                                np.exp((0.0885 - 0.0001282 * GESTDAY[i - 1, j - 1]) *
                                       GESTDAY[i - 1, j - 1])
                            ) +
                            5.505 * (
                                (0.00003452 - 0.0000001094 * GESTDAY[i - 1, j - 1]) *
                                np.exp((0.0589 - 0.00009334 * GESTDAY[i - 1, j - 1]) *
                                       GESTDAY[i - 1, j - 1])
                            )
                        ) * CALTOJOULE * 10 * (BIRTHW1[i - 1, j - 1] / 37.2) * 0.6
                    )
                else:
                    NEREQGEST[i - 1, j - 1] = 0

                # Note: The efficiency of energy accretion gestation is only 14% (Jarrige, 1989,
                # Rattray et al., 1974) for the calf, and 9.33% for extra tissue of the
                # reproductive cow (Jarrige, 1989)

                # Heat production from gestation (MJ per cow per day)
                HEATGEST[i - 1, j - 1] = NEREQGEST[i - 1, j - 1] * NEIEFFGEST
                # Cumulative NE during gestation (MJ)
                NEREQGESTADD[i, j - 1] = NEREQGESTADD[i - 1, j - 1] + NEREQGEST[i - 1, j - 1]

                # Breed- and sex-specific NE requirements for the complete gestation period
                calf_idx = int(CALFNR[i - 1, j - 1])
                if BREED == 1 and SEX[calf_idx] == 0:
                    NEREQGESTTOT = 58.658 * LIBRARY10[6] + 0.5502
                if BREED == 1 and SEX[calf_idx] == 1:
                    NEREQGESTTOT = 58.658 * LIBRARY11[6] + 0.5502
                if BREED == 2 and SEX[calf_idx] == 0:
                    NEREQGESTTOT = 58.658 * LIBRARY20[6] + 0.5502
                if BREED == 2 and SEX[calf_idx] == 1:
                    NEREQGESTTOT = 58.658 * LIBRARY21[6] + 0.5502
                if BREED == 3 and SEX[calf_idx] == 0:
                    NEREQGESTTOT = 58.658 * LIBRARY30[6] + 0.5502
                if BREED == 3 and SEX[calf_idx] == 1:
                    NEREQGESTTOT = 58.658 * LIBRARY31[6] + 0.5502
                if BREED == 4 and SEX[calf_idx] == 0:
                    NEREQGESTTOT = 58.658 * LIBRARY40[6] + 0.5502
                if BREED == 4 and SEX[calf_idx] == 1:
                    NEREQGESTTOT = 58.658 * LIBRARY41[6] + 0.5502
                if BREED == 5 and SEX[calf_idx] == 0:
                    NEREQGESTTOT = 58.658 * LIBRARY50[6] + 0.5502
                if BREED == 5 and SEX[calf_idx] == 1:
                    NEREQGESTTOT = 58.658 * LIBRARY51[6] + 0.5502
                if BREED == 6 and SEX[calf_idx] == 0:
                    NEREQGESTTOT = 58.658 * LIBRARY60[6] + 0.5502
                if BREED == 6 and SEX[calf_idx] == 1:
                    NEREQGESTTOT = 58.658 * LIBRARY61[6] + 0.5502

                # The weight of the foetus is calculated from the cumulative NE for gestation and the
                # complete NE required during the gestation period (kg).
                CALFTBW[i, j - 1] = (
                    NEREQGESTADD[i, j - 1] / NEREQGESTTOT * BIRTHW1[i - 1, j - 1]
                )

                # At the end of the gestation period, the calf reaches its birth weight (kg live
                # weight)
                if (GESTDAY[i, j - 1] >= 1 and
                        GESTDAY[i, j - 1] <= GestPer):
                    CALFTBW[i, j - 1] = CALFTBW[i, j - 1]
                else:
                    CALFTBW[i, j - 1] = 0
                # The (cumulative) NE requirements for gestation stop after parturition
                if (CALFTBW[i - 1, j - 1] - CALFTBW[i, j - 1]) > BIRTHW1[i - 1, j - 1] - 1:
                    NEREQGESTADD[i, j - 1] = 0

                # Additional TBW of the cow, without an increase in the NE requirements for
                # maintenance. The weight increase of the cow includes the weight of the foetus and
                # the concepta (Jarrige, 1986, p. 99)
                TBWADD[i, j - 1] = CALFTBW[i, j - 1] * FtoConcW

                # Protein balance
                # Protein requirements for gestation (g protein day-1)
                # Note: * based on total NE and protein requirements (CSIRO, 2007)
                #       * the conversion is 4.322 g protein per MJ NE for gestation
                #       * the formula given in CSIRO (2007) is assumed to present the gross protein
                #         requirement for gestation
                PROTGESTG[i - 1, j - 1] = NEREQGEST[i - 1, j - 1] * CPGEST

                #########################################################################################
                # Milk production #
                ###################

                # Days in milk, milk production starts at parturition
                if GESTDAY[i - 1, j - 1] == GestPer - 1:
                    MILKDAYST[i - 1, j - 1] = 1
                else:
                    MILKDAYST[i - 1, j - 1] = 0

                # Start of milk production after parturition
                if MILKDAYST[i - 1, j - 1] == 1:
                    ADDMILK2[i - 1, j - 1] = 1
                else:
                    ADDMILK2[i - 1, j - 1] = 0
                # Adds up days after parturition
                if MILKDAY[i - 1, j - 1] > 0:
                    ADDMILK1[i - 1, j - 1] = 1
                else:
                    ADDMILK1[i - 1, j - 1] = 0
                # Days after parturition
                MILKDAY[i, j - 1] = (
                    MILKDAY[i - 1, j - 1] +
                    ADDMILK1[i - 1, j - 1] +
                    ADDMILK2[i - 1, j - 1]
                )
                # Milk production ends at weaning; cow and calf are separated at weaning
                if MILKDAY[i, j - 1] == WEANINGTIME + 1:
                    MILKDAY[i, j - 1] = 0

                # The number of calves born per reproductive cow is calculated from the number of
                # conceptions
                if CALFNR[i - 1, j - 1] == 0:
                    CALFLIVENR[i - 1, j - 1] = 0
                else:
                    CALFLIVENR[i - 1, j - 1] = CALFNR[i - int(GestPer) - 1, j - 1]
                # The number of calves weaned per reproductive cow is calculated from the number of
                # calves born.
                if CALFLIVENR[i - 1, j - 1] == 0:
                    CALFWEANNR[i - 1, j - 1] = 0
                else:
                    CALFWEANNR[i - 1, j - 1] = CALFLIVENR[i - int(WEANINGTIME) - 1, j - 1]

                # Converts days in milk to weeks in milk (weeks)
                MILKWEEK[i - 1, j - 1] = MILKDAY[i - 1, j - 1] / 7

                # Maximum milk production based on genotype (L day-1)
                # This equation corresponds to Eq. 30 in the Supplementary Information
                if MILKWEEK[i - 1, j - 1] > 0:
                    MAXMILKPROD[i - 1, j - 1] = (
                        MILKPARA *
                        MILKDAY[i - 1, j - 1] ** MILKPARB *
                        np.exp(-MILKPARC * MILKDAY[i - 1, j - 1])
                    )
                else:
                    MAXMILKPROD[i - 1, j - 1] = 0

                # Milk received by the calf from its mother (MJ ME day-1)
                if TIME[i - 1, j - 1] > WEANINGTIME:
                    MEMILKCALFINIT[i - 1, j - 1] = 0
                else:
                    MEMILKCALFINIT[i - 1, j - 1] = MEMILKCALFINIT[i - 1, j - 1]

                # Assumption: milk production is the maximum milk production based on the genotype
                # (L day-1)
                MILKPRODBF[i - 1, j - 1] = MAXMILKPROD[i - 1, j - 1]

                # Energy balance
                # Gross energy (GE, combustion value) in milk (MJ GE L-1)
                GEMILK[i - 1, j - 1] = (
                    ((GEMILK1 * SENSMAT[93, s - 1]) * MILKDAY[i - 1, j - 1]) +
                    (GEMILK2 * SENSMAT[94, s - 1])
                ) / 1000
                # Maximum gross energy from milk production (MJ GE day-1)
                GEMILKTOT[i - 1, j - 1] = GEMILK[i - 1, j - 1] * MAXMILKPROD[i - 1, j - 1]
                # Maximum metabolisable energy (ME) in milk (MJ ME day-1)
                MEMILKCALF[i - 1, j - 1] = MILKDIG * GEMILKTOT[i - 1, j - 1]
                # Net energy (NE) requirement for milk production for reproductive cow (MJ NE day-1)
                NEMILKCOW[i - 1, j - 1] = GEMILKTOT[i - 1, j - 1] / NEEFFMILK
                # Heat generation for milk synthesis (MJ day-1)
                HEATMILK[i - 1, j - 1] = NEMILKCOW[i - 1, j - 1] * (1 - NEEFFMILK)

                # Protein balance
                # Protein in milk (g protein day-1)
                PROTMILK[i - 1, j - 1] = MILKPRODBF[i - 1, j - 1] * PROTFRACMILK * 1000
                # Gross amount of protein required for milk production (g protein day-1)
                PROTMILKG[i - 1, j - 1] = PROTMILK[i - 1, j - 1] / PROTEFFMILK
########################## line:3317 ##############################
                #########################################################################################
                # Growth #
                ##########

                # Compensatory growth: potential growth can be exceeded. Assumption is that the
                # compensatory growth isproportional to the difference between actual TBW and genetic
                # potential TBW of the animal.

                # Factor for compensatory growth for non carcass tissue (-)
                COMPGROWTH1[i - 1, j - 1] = min(
                    COMPFACTTIS,
                    max(1, NONCARCTIS[i - 1, j - 1] / NONCARCTISBF[i - 1, j - 1])
                )
                # Factor for compensatory growth for bone tissue (-)
                COMPGROWTH2[i - 1, j - 1] = min(
                    COMPFACTTIS,
                    max(1, BONETIS[i - 1, j - 1] / BONETISBF[i - 1, j - 1])
                )
                # Factor for compensatory growth for muscle tissue (-)
                COMPGROWTH3[i - 1, j - 1] = min(
                    COMPFACTTIS,
                    max(1, MUSCLETIS[i - 1, j - 1] / MUSCLETISBF[i - 1, j - 1])
                )
                # Factor for compensatory growth for intramuscular fat tissue (-)
                COMPGROWTH4[i - 1, j - 1] = min(
                    COMPFACTTIS,
                    max(1, INTRAMFTIS[i - 1, j - 1] / INTRAMFTISBF[i - 1, j - 1])
                )
                # Factor for compensatory growth for miscellaneous fat tissue (-)
                COMPGROWTH5[i - 1, j - 1] = min(
                    COMPFACTTIS,
                    max(1, MISCFATTIS[i - 1, j - 1] / MISCFATTISBF[i - 1, j - 1])
                )

                # Weighted average of the factor for compensatory growth (-)
                COMPGROWTH[i - 1, j - 1] = (
                    COMPGROWTH1[i - 1, j - 1] * (NONCARCTISBF[i - 1, j - 1] / (TBWBF[i - 1, j - 1] * (1 - RUMENFRAC))) +
                    COMPGROWTH2[i - 1, j - 1] * (BONETISBF[i - 1, j - 1]   / (TBWBF[i - 1, j - 1] * (1 - RUMENFRAC))) +
                    COMPGROWTH3[i - 1, j - 1] * (MUSCLETISBF[i - 1, j - 1] / (TBWBF[i - 1, j - 1] * (1 - RUMENFRAC))) +
                    COMPGROWTH4[i - 1, j - 1] * (INTRAMFTISBF[i - 1, j - 1] / (TBWBF[i - 1, j - 1] * (1 - RUMENFRAC))) +
                    COMPGROWTH5[i - 1, j - 1] * (MISCFATTISBF[i - 1, j - 1] / (TBWBF[i - 1, j - 1] * (1 - RUMENFRAC)))
                )

                # Fraction lipid in the bone tissue (-)
                LIPIDFRACBONEBF[i - 1, j - 1] = max(
                    (LIPBONE1 * SENSMAT[95, s - 1]),
                    (
                        (LIPBONE2 * SENSMAT[96, s - 1]) *
                        np.log(BONETISBF[i - 1, j - 1]) +
                        (LIPBONE3 * SENSMAT[97, s - 1])
                    ) / 100
                )
                # Fraction lipid in the non-carcass tissue (-)
                LIPIDFRACNONCBF[i - 1, j - 1] = (
                    (
                        (LIPNONC1 * SENSMAT[98, s - 1]) * NONCARCTISBF[i - 1, j - 1] ** 3 -
                        (LIPNONC2 * SENSMAT[99, s - 1]) * NONCARCTISBF[i - 1, j - 1] ** 2 +
                        (LIPNONC3 * SENSMAT[100, s - 1]) * NONCARCTISBF[i - 1, j - 1] -
                        (LIPNONC4 * SENSMAT[101, s - 1])
                    ) / 100
                    *
                    (
                        3.916E-10 * ((1 - LIBRARY[20] - RUMENFRAC) * LIBRARY[12]) ** 4 -
                        7.058E-07 * ((1 - LIBRARY[20] - RUMENFRAC) * LIBRARY[12]) ** 3 +
                        4.868E-04 * ((1 - LIBRARY[20] - RUMENFRAC) * LIBRARY[12]) ** 2 -
                        1.593E-01 * ((1 - LIBRARY[20] - RUMENFRAC) * LIBRARY[12]) +
                        2.286E+01
                    )
                )

                # Fraction protein in the  non-carcass tissue
                PROTFRACNONCBF[i - 1, j - 1] = (
                    (
                        (PROTNONC1 * SENSMAT[102, s - 1]) * NONCARCTISBF[i - 1, j - 1] ** 4 -
                        (PROTNONC2 * SENSMAT[103, s - 1]) * NONCARCTISBF[i - 1, j - 1] ** 3 +
                        (PROTNONC3 * SENSMAT[104, s - 1]) * NONCARCTISBF[i - 1, j - 1] ** 2 -
                        (PROTNONC4 * SENSMAT[105, s - 1]) * NONCARCTISBF[i - 1, j - 1] +
                        (PROTNONC5 * SENSMAT[106, s - 1])
                    ) / 100
                )

                # Energy for growth, consists of three parts:
                # 1. Genetic potential growth, including compensatory growth
                # 2. Energy requirements to recover the depleted subcutaneous and intermuscular fat
                #    tissues
                # 3. Reduction in energy for growth mentioned in 1 and 2 due to climatic conditions.

                # Under energy limitation, tissues get energy according to their position in the
                # hierarchy: 1. Non carcass tissue 2. Bone tissue 3. Muscle tissue 4. Intramuscular fat
                # tissue 5. Subcutaneous and intermuscular fat tissue

                # Factor to counter low weights of the miscellaneous and non-carcass tissues (-)
                # This equation corresponds to Eq. 46 of the Supplementary Information
                FATCOMP[i - 1, j - 1] = (
                    max(0, FATTISCOMP - MISCFATTISBF[i - 1, j - 1] / MISCFATTIS[i - 1, j - 1]) *
                    (TBWBF[i - 1, j - 1] * (1 - RUMENFRAC)) ** 0.75 *
                    FATFACTOR
                )
########################## line:3393 ##############################
                #########################################################################################
                # Growth #
                ##########

                # Compensatory growth: potential growth can be exceeded. Assumption is that the
                # compensatory growth isproportional to the difference between actual TBW and genetic
                # potential TBW of the animal.

                # Factor for compensatory growth for non carcass tissue (-)
                COMPGROWTH1[i - 1, j - 1] = min(
                    COMPFACTTIS,
                    max(1, NONCARCTIS[i - 1, j - 1] / NONCARCTISBF[i - 1, j - 1])
                )
                # Factor for compensatory growth for bone tissue (-)
                COMPGROWTH2[i - 1, j - 1] = min(
                    COMPFACTTIS,
                    max(1, BONETIS[i - 1, j - 1] / BONETISBF[i - 1, j - 1])
                )
                # Factor for compensatory growth for muscle tissue (-)
                COMPGROWTH3[i - 1, j - 1] = min(
                    COMPFACTTIS,
                    max(1, MUSCLETIS[i - 1, j - 1] / MUSCLETISBF[i - 1, j - 1])
                )
                # Factor for compensatory growth for intramuscular fat tissue (-)
                COMPGROWTH4[i - 1, j - 1] = min(
                    COMPFACTTIS,
                    max(1, INTRAMFTIS[i - 1, j - 1] / INTRAMFTISBF[i - 1, j - 1])
                )
                # Factor for compensatory growth for miscellaneous fat tissue (-)
                COMPGROWTH5[i - 1, j - 1] = min(
                    COMPFACTTIS,
                    max(1, MISCFATTIS[i - 1, j - 1] / MISCFATTISBF[i - 1, j - 1])
                )

                # Weighted average of the factor for compensatory growth (-)
                COMPGROWTH[i - 1, j - 1] = (
                    COMPGROWTH1[i - 1, j - 1] * (NONCARCTISBF[i - 1, j - 1] / (TBWBF[i - 1, j - 1] * (1 - RUMENFRAC))) +
                    COMPGROWTH2[i - 1, j - 1] * (BONETISBF[i - 1, j - 1]   / (TBWBF[i - 1, j - 1] * (1 - RUMENFRAC))) +
                    COMPGROWTH3[i - 1, j - 1] * (MUSCLETISBF[i - 1, j - 1] / (TBWBF[i - 1, j - 1] * (1 - RUMENFRAC))) +
                    COMPGROWTH4[i - 1, j - 1] * (INTRAMFTISBF[i - 1, j - 1] / (TBWBF[i - 1, j - 1] * (1 - RUMENFRAC))) +
                    COMPGROWTH5[i - 1, j - 1] * (MISCFATTISBF[i - 1, j - 1] / (TBWBF[i - 1, j - 1] * (1 - RUMENFRAC)))
                )

                # Fraction lipid in the bone tissue (-)
                LIPIDFRACBONEBF[i - 1, j - 1] = max(
                    (LIPBONE1 * SENSMAT[95, s - 1]),
                    (
                        (LIPBONE2 * SENSMAT[96, s - 1]) *
                        np.log(BONETISBF[i - 1, j - 1]) +
                        (LIPBONE3 * SENSMAT[97, s - 1])
                    ) / 100
                )
                # Fraction lipid in the non-carcass tissue (-)
                LIPIDFRACNONCBF[i - 1, j - 1] = (
                    (
                        (LIPNONC1 * SENSMAT[98, s - 1]) * NONCARCTISBF[i - 1, j - 1] ** 3 -
                        (LIPNONC2 * SENSMAT[99, s - 1]) * NONCARCTISBF[i - 1, j - 1] ** 2 +
                        (LIPNONC3 * SENSMAT[100, s - 1]) * NONCARCTISBF[i - 1, j - 1] -
                        (LIPNONC4 * SENSMAT[101, s - 1])
                    ) / 100
                    *
                    (
                        3.916E-10 * ((1 - LIBRARY[20] - RUMENFRAC) * LIBRARY[12]) ** 4 -
                        7.058E-07 * ((1 - LIBRARY[20] - RUMENFRAC) * LIBRARY[12]) ** 3 +
                        4.868E-04 * ((1 - LIBRARY[20] - RUMENFRAC) * LIBRARY[12]) ** 2 -
                        1.593E-01 * ((1 - LIBRARY[20] - RUMENFRAC) * LIBRARY[12]) +
                        2.286E+01
                    )
                )

                # Fraction protein in the  non-carcass tissue
                PROTFRACNONCBF[i - 1, j - 1] = (
                    (
                        (PROTNONC1 * SENSMAT[102, s - 1]) * NONCARCTISBF[i - 1, j - 1] ** 4 -
                        (PROTNONC2 * SENSMAT[103, s - 1]) * NONCARCTISBF[i - 1, j - 1] ** 3 +
                        (PROTNONC3 * SENSMAT[104, s - 1]) * NONCARCTISBF[i - 1, j - 1] ** 2 -
                        (PROTNONC4 * SENSMAT[105, s - 1]) * NONCARCTISBF[i - 1, j - 1] +
                        (PROTNONC5 * SENSMAT[106, s - 1])
                    ) / 100
                )

                # Energy for growth, consists of three parts:
                # 1. Genetic potential growth, including compensatory growth
                # 2. Energy requirements to recover the depleted subcutaneous and intermuscular fat
                #    tissues
                # 3. Reduction in energy for growth mentioned in 1 and 2 due to climatic conditions.

                # Under energy limitation, tissues get energy according to their position in the
                # hierarchy: 1. Non carcass tissue 2. Bone tissue 3. Muscle tissue 4. Intramuscular fat
                # tissue 5. Subcutaneous and intermuscular fat tissue

                # Factor to counter low weights of the miscellaneous and non-carcass tissues (-)
                # This equation corresponds to Eq. 46 of the Supplementary Information
                FATCOMP[i - 1, j - 1] = (
                    max(0, FATTISCOMP - MISCFATTISBF[i - 1, j - 1] / MISCFATTIS[i - 1, j - 1]) *
                    (TBWBF[i - 1, j - 1] * (1 - RUMENFRAC)) ** 0.75 *
                    FATFACTOR
                )
########################## line:3393 ##############################
                ###########################################################################################
                # 2.3                       Feed intake and digestion sub-model                           #
                ###########################################################################################

                # Digestion capacity limitation (related to the feed quality)
                # Maximum digestion capacity (Fill Units per animal per day)
                # This equation corresponds partly to Eq. 22 of the Supplementary Information
                PHFEEDINT[i - 1, j - 1] = (
                    TBWBF[i - 1, j - 1] ** 0.75 * PHFEEDCAP / 1000 *
                    max(
                        0,
                        min(
                            1,
                            ((RUMENDEV1 * SENSMAT[107, s - 1]) * TIME[i - 1, j - 1] -
                             (RUMENDEV2 * SENSMAT[108, s - 1]))
                        )
                    )
                )

                # Code below enables to feed animal per 100 kg TBW (fixed % of the TBW)
                if FEEDNR[case_template_idx] == 5:
                    FEED1QNTY[i - 1, j - 1] = FEED1QNTY[i - 1, j - 1] * TBWBF[i - 1, j - 1] / 100
                if FEEDNR[case_template_idx] == 5:
                    FEED2QNTY[i - 1, j - 1] = FEED2QNTY[i - 1, j - 1] * TBWBF[i - 1, j - 1] / 100

                # Feed intake cannot exceed the digestive capacity of the rumen
                # Fill units feed type 1
                FUFEED1[i - 1, j - 1] = FEED1QNTY[i - 1, j - 1] * FEED1[i - 1, 1]

                # Maximum intake of feed type 1 based on the digestive capacity of the rumen (kg DM)
                if (PHFEEDINT[i - 1, j - 1] - FUFEED1[i - 1, j - 1]) < 0:
                    FEED1QNTYA[i - 1, j - 1] = (
                        PHFEEDINT[i - 1, j - 1] /
                        (FEED1fr * FEED1[i - 1, 1] + (1 - FEED1fr) * FEED2[i - 1, 1]) *
                        FEED1fr
                    )
                else:
                    FEED1QNTYA[i - 1, j - 1] = FEED1QNTY[i - 1, j - 1]

                # Fill units feed type 1 + feed type 2 (FU)
                FUFEED2[i - 1, j - 1] = (
                    FEED1QNTYA[i - 1, j - 1] * FEED1[i - 1, 1] +
                    FEED2QNTY[i - 1, j - 1] * FEED2[i - 1, 1]
                )
                # Maximum intake of feed type 2 based on the digestive capacity of the rumen (kg DM)
                if (PHFEEDINT[i - 1, j - 1] - FUFEED2[i - 1, j - 1]) < 0:
                    FEED2QNTYA[i - 1, j - 1] = (
                        PHFEEDINT[i - 1, j - 1] /
                        (FEED1fr * FEED1[i - 1, 1] + (1 - FEED1fr) * FEED2[i - 1, 1]) *
                        (1 - FEED1fr)
                    )
                else:
                    FEED2QNTYA[i - 1, j - 1] = FEED2QNTY[i - 1, j - 1]

                # Fill units feed type 1 + feed type 2 + feed type 3 (FU)
                FUFEED3[i - 1, j - 1] = (
                    FEED1QNTYA[i - 1, j - 1] * FEED1[i - 1, 1] +
                    FEED2QNTYA[i - 1, j - 1] * FEED2[i - 1, 1] +
                    FEED3QNTY[i - 1, j - 1] * FEED3[i - 1, 1]
                )
                # Maximum intake of feed type 3 based on the digestive capacity of the rumen (kg DM)
                if (PHFEEDINT[i - 1, j - 1] - FUFEED3[i - 1, j - 1]) < 0:
                    FEED3QNTYA[i - 1, j - 1] = max(
                        0,
                        (PHFEEDINT[i - 1, j - 1] - FUFEED2[i - 1, j - 1]) / FEED3[i - 1, 1]
                    )
                else:
                    FEED3QNTYA[i - 1, j - 1] = FEED3QNTY[i - 1, j - 1]

                # Fill units feed type 1 + feed type 2 + feed type 3 + feed type 4 (FU)
                FUFEED4[i - 1, j - 1] = (
                    FEED1QNTY[i - 1, j - 1] * FEED1[i - 1, 1] +
                    FEED2QNTY[i - 1, j - 1] * FEED2[i - 1, 1] +
                    FEED3QNTY[i - 1, j - 1] * FEED3[i - 1, 1] +
                    FEED4QNTY[i - 1, j - 1] * FEED4[1]
                )
                # Maximum intake of feed type 4 based on the digestive capacity of the rumen (kg DM)
                if (PHFEEDINT[i - 1, j - 1] - FUFEED4[i - 1, j - 1]) < 0:
                    FEED4QNTYA[i - 1, j - 1] = max(
                        0,
                        (PHFEEDINT[i - 1, j - 1] - FUFEED3[i - 1, j - 1]) / FEED4[1]
                    )
                else:
                    FEED4QNTY[i - 1, j - 1] = FEED4QNTY[i - 1, j - 1]
                FEED4QNTYA[i - 1, j - 1] = min(
                    FEED4QNTY[i - 1, j - 1],
                    FEED4fr * PHFEEDINT[i - 1, j - 1] / FEED4[1]
                )

                # Rumen fill classes according to Chilibroste et al. (1997)
                if FUFEED4[i - 1, j - 1] > PHFEEDINT[i - 1, j - 1] * 0.85:
                    PASSAGE[i - 1, j - 1] = 1
                else:
                    if (FUFEED4[i - 1, j - 1] > PHFEEDINT[i - 1, j - 1] * 0.65 and
                            FUFEED4[i - 1, j - 1] < PHFEEDINT[i - 1, j - 1] * 0.85):
                        PASSAGE[i - 1, j - 1] = 2
                    else:
                        if (FUFEED4[i - 1, j - 1] > PHFEEDINT[i - 1, j - 1] * 0.45 and
                                FUFEED4[i - 1, j - 1] < PHFEEDINT[i - 1, j - 1] * 0.65):
                            PASSAGE[i - 1, j - 1] = 3
                        else:
                            PASSAGE[i - 1, j - 1] = 4

                # Average crude protein (CP) in the diet (g CP)
                # orginal
                #CPAVG[i - 1, j - 1] = (
                #    FEED1QNTYA[i - 1, j - 1] * FEED1[i - 1, 15] +
                #    FEED2QNTYA[i - 1, j - 1] * FEED2[i - 1, 15] +
                #    FEED3QNTYA[i - 1, j - 1] * FEED3[i - 1, 15] +
                #    FEED4QNTYA[i - 1, j - 1] * FEED1[15]
                #) / (
                #    FEED1QNTYA[i - 1, j - 1] + FEED2QNTYA[i - 1, j - 1] +
                #    FEED3QNTYA[i - 1, j - 1] + FEED4QNTYA[i - 1, j - 1]
                #)
                CPAVG[i - 1, j - 1] = (
                    FEED1QNTY[i - 1, j - 1] * FEED1[i - 1, 15] +
                    FEED2QNTY[i - 1, j - 1] * FEED2[i - 1, 15] +
                    FEED3QNTY[i - 1, j - 1] * FEED3[i - 1, 15] +
                    FEED4QNTY[i - 1, j - 1] * FEED4[15]
                ) / FEEDQNTY[i - 1, j - 1]

                ###########################################################################################
                # 2.4 Integration thermoregulation, feed intake and digestion, and utilisation sub-models #
                ###########################################################################################

                # Initial four assumptions for the integration loop:

                # 1. The reduction in NE availability due to heat stress is assumed to be zero
                # (guesstimate, indicates absence of heatstress)
                REDHP[i - 1, j - 1] = 0

                # 2. The increase in heat production due to cold stress is asssumed to be zero
                # (guesstimate, indicates absence of cold stress)
                HEATIFEEDGROWTHC[i - 1, j - 1] = 0

                # 3. The feed intake is 20 kg DM per animal per day. This number is generally too high
                # for beef cattle, and this number is reduced later in the loop.
                FEEDINTAKE[i - 1, j - 1] = 20

                # 4. The animal is assumed to be fed above the maintenance level
                REDMAINT[i - 1, j - 1] = 0
                ############### tracking / log ############
                REPS[i - 1, j - 1] = 0  # Indicates and counts the times the integration loop is repeated
                #print("LOG 3877: befor enter in WHILE")
                REPS[i - 1, j - 1] = 0  # Indicates and counts the times the integration loop is repeated
                loop_iter = 0
               
                if DEBUG_LOOP and z in DEBUG_CASES and (DEBUG_I is None or i == DEBUG_I) and (DEBUG_J is None or j == DEBUG_J):
                    print(
                        f"[ENTER LOOP] z={z} s={s} j={j} i={i} "
                        f"TIME={_dbg_scalar(TIME[i - 1, j - 1])} "
                        f"PHFEEDINT={_dbg_scalar(PHFEEDINT[i - 1, j - 1])} "
                        f"FEEDINTAKE0={_dbg_scalar(FEEDINTAKE[i - 1, j - 1])} "
                        f"REDHP0={_dbg_scalar(REDHP[i - 1, j - 1])} "
                        f"HEATIFEEDGROWTHC0={_dbg_scalar(HEATIFEEDGROWTHC[i - 1, j - 1])}"
                    )

                #while True:
                    
########################## line:3498 ##############################
                while True:
                    loop_iter += 1
                    # Start of the integration loop

                    # Minimum feed quantities based on rumen digestive capacity and the available feed
                    # quantity. Feed intake is reduced here from 20 kg DM head-1 day-1 to the correct
                    # amount

                    # Intake feed type 1 (kg DM per animal per day)
                    if REPRODUCTIVE[j - 1] == 1:
                        FEED1QNTY[i - 1, j - 1] = max(
                            0,
                            min(FEED1QNTYA[i - 1, j - 1], FEED1fr * FEEDINTAKE[i - 1, j - 1])
                        )
                    elif PRODUCTIVE[j - 1] == 1 and SEX[j - 1] == 0:
                        FEED1QNTY[i - 1, j - 1] = max(
                            0,
                            min(FEED1QNTYA[i - 1, j - 1], FEED1fr * FEEDINTAKE[i - 1, j - 1])
                        ) * 1
                    elif PRODUCTIVE[j - 1] == 1 and SEX[j - 1] == 1:
                        FEED1QNTY[i - 1, j - 1] = max(
                            0,
                            min(FEED1QNTYA[i - 1, j - 1], FEED1fr * FEEDINTAKE[i - 1, j - 1])
                        ) * 1
                    else:
                        FEED1QNTY[i - 1, j - 1] = max(
                            0,
                            min(FEED1QNTYA[i - 1, j - 1], FEED1fr * FEEDINTAKE[i - 1, j - 1])
                        )

                    # Intake feed type 2 (kg DM per animal per day)
                    if REPRODUCTIVE[j - 1] == 1:
                        FEED2QNTY[i - 1, j - 1] = max(
                            0,
                            min(
                                FEED2QNTYA[i - 1, j - 1],
                                FEED2fr * FEEDINTAKE[i - 1, j - 1],
                                FEEDINTAKE[i - 1, j - 1] - FEED1QNTY[i - 1, j - 1]
                            )
                        )
                    elif PRODUCTIVE[j - 1] == 1 and SEX[j - 1] == 0:
                        FEED2QNTY[i - 1, j - 1] = max(
                            0,
                            min(
                                FEED2QNTYA[i - 1, j - 1],
                                FEED2fr * FEEDINTAKE[i - 1, j - 1],
                                FEEDINTAKE[i - 1, j - 1] - FEED1QNTY[i - 1, j - 1]
                            )
                        ) * 1
                    elif PRODUCTIVE[j - 1] == 1 and SEX[j - 1] == 1:
                        FEED2QNTY[i - 1, j - 1] = max(
                            0,
                            min(
                                FEED2QNTYA[i - 1, j - 1],
                                FEED2fr * FEEDINTAKE[i - 1, j - 1],
                                FEEDINTAKE[i - 1, j - 1] - FEED1QNTY[i - 1, j - 1]
                            )
                        ) * 1
                    else:
                        FEED2QNTY[i - 1, j - 1] = max(
                            0,
                            min(
                                FEED2QNTYA[i - 1, j - 1],
                                FEED2fr * FEEDINTAKE[i - 1, j - 1],
                                FEEDINTAKE[i - 1, j - 1] - FEED1QNTY[i - 1, j - 1]
                            )
                        )

                    # Intake feed type 3 (kg DM per animal per day)
                    if REPRODUCTIVE[j - 1] == 1:
                        FEED3QNTY[i - 1, j - 1] = max(
                            0,
                            min(
                                FEED3QNTYA[i - 1, j - 1],
                                FEED3fr * FEEDINTAKE[i - 1, j - 1],
                                FEEDINTAKE[i - 1, j - 1]
                                - FEED1QNTY[i - 1, j - 1]
                                - FEED2QNTY[i - 1, j - 1]
                            )
                        )
                    elif PRODUCTIVE[j - 1] == 1 and SEX[j - 1] == 0:
                        FEED3QNTY[i - 1, j - 1] = max(
                            0,
                            min(
                                FEED3QNTYA[i - 1, j - 1],
                                FEED3fr * FEEDINTAKE[i - 1, j - 1],
                                FEEDINTAKE[i - 1, j - 1]
                                - FEED1QNTY[i - 1, j - 1]
                                - FEED2QNTY[i - 1, j - 1]
                            )
                        ) * 1
                    elif PRODUCTIVE[j - 1] == 1 and SEX[j - 1] == 1:
                        FEED3QNTY[i - 1, j - 1] = max(
                            0,
                            min(
                                FEED3QNTYA[i - 1, j - 1],
                                FEED3fr * FEEDINTAKE[i - 1, j - 1],
                                FEEDINTAKE[i - 1, j - 1]
                                - FEED1QNTY[i - 1, j - 1]
                                - FEED2QNTY[i - 1, j - 1]
                            )
                        ) * 1
                    else:
                        FEED3QNTY[i - 1, j - 1] = max(
                            0,
                            min(
                                FEED3QNTYA[i - 1, j - 1],
                                FEED3fr * FEEDINTAKE[i - 1, j - 1],
                                FEEDINTAKE[i - 1, j - 1]
                                - FEED1QNTY[i - 1, j - 1]
                                - FEED2QNTY[i - 1, j - 1]
                            )
                        )

                    # Intake feed type 4 (kg DM per animal per day)
                    if REPRODUCTIVE[j - 1] == 1:
                        FEED4QNTY[i - 1, j - 1] = max(
                            0,
                            min(
                                FEED4QNTYA[i - 1, j - 1],
                                FEED4fr * FEEDINTAKE[i - 1, j - 1],
                                FEEDINTAKE[i - 1, j - 1]
                                - FEED1QNTY[i - 1, j - 1]
                                - FEED2QNTY[i - 1, j - 1]
                                - FEED3QNTY[i - 1, j - 1]
                            )
                        )
                    elif PRODUCTIVE[j - 1] == 1 and SEX[j - 1] == 0:
                        FEED4QNTY[i - 1, j - 1] = max(
                            0,
                            min(
                                FEED4QNTYA[i - 1, j - 1],
                                FEED4fr * FEEDINTAKE[i - 1, j - 1],
                                FEEDINTAKE[i - 1, j - 1]
                                - FEED1QNTY[i - 1, j - 1]
                                - FEED2QNTY[i - 1, j - 1]
                                - FEED3QNTY[i - 1, j - 1]
                            )
                        )
                    elif PRODUCTIVE[j - 1] == 1 and SEX[j - 1] == 1:
                        FEED4QNTY[i - 1, j - 1] = max(
                            0,
                            min(
                                FEED4QNTYA[i - 1, j - 1],
                                FEED4fr * FEEDINTAKE[i - 1, j - 1],
                                FEEDINTAKE[i - 1, j - 1]
                                - FEED1QNTY[i - 1, j - 1]
                                - FEED2QNTY[i - 1, j - 1]
                                - FEED3QNTY[i - 1, j - 1]
                            )
                        )
                    else:
                        FEED4QNTY[i - 1, j - 1] = max(
                            0,
                            min(
                                FEED4QNTYA[i - 1, j - 1],
                                FEED4fr * FEEDINTAKE[i - 1, j - 1],
                                FEEDINTAKE[i - 1, j - 1]
                                - FEED1QNTY[i - 1, j - 1]
                                - FEED2QNTY[i - 1, j - 1]
                                - FEED3QNTY[i - 1, j - 1]
                            )
                        )

                    # Total feed intake (kg DM per animal per day)
                    FEEDQNTY[i - 1, j - 1] = (
                        FEED1QNTY[i - 1, j - 1]
                        + FEED2QNTY[i - 1, j - 1]
                        + FEED3QNTY[i - 1, j - 1]
                        + FEED4QNTY[i - 1, j - 1]
                    )

                    ################### Tracking / log ##################
                    if DEBUG_LOOP and z in DEBUG_CASES and (DEBUG_I is None or i == DEBUG_I) and (DEBUG_J is None or j == DEBUG_J):
                        print(
                            f"[CHK1 FEED] z={z} s={s} j={j} i={i} iter={loop_iter} "
                            f"F1={_dbg_scalar(FEED1QNTY[i - 1, j - 1])} "
                            f"F2={_dbg_scalar(FEED2QNTY[i - 1, j - 1])} "
                            f"F3={_dbg_scalar(FEED3QNTY[i - 1, j - 1])} "
                            f"F4={_dbg_scalar(FEED4QNTY[i - 1, j - 1])} "
                            f"FTOT={_dbg_scalar(FEEDQNTY[i - 1, j - 1])}",
                            flush=True
                        )
                    #####################################################

                    # Crude protein content of the diet (g kg DM-1 feed)
                    if FEEDQNTY[i - 1, j - 1] == 0:
                        CPAVG[i - 1, j - 1] = 0
                    else:
                        CPAVG[i - 1, j - 1] = (
                            FEED1QNTY[i - 1, j - 1] * FEED1[i - 1, 15]
                            + FEED2QNTY[i - 1, j - 1] * FEED2[i - 1, 15]
                            + FEED3QNTY[i - 1, j - 1] * FEED3[i - 1, 15]
                            + FEED4QNTY[i - 1, j - 1] * FEED4[15]
                        ) / FEEDQNTY[i - 1, j - 1]

                    # Calculates the fraction of each feed type in the diet
                    if TIME[i - 1, j - 1] <= 14 or FEEDQNTY[i - 1, j - 1] == 0:
                        FRACFEED1[i - 1, j - 1] = 0
                    else:
                        FRACFEED1[i - 1, j - 1] = FEED1QNTY[i - 1, j - 1] / FEEDQNTY[i - 1, j - 1]

                    if TIME[i - 1, j - 1] <= 14 or FEEDQNTY[i - 1, j - 1] == 0:
                        FRACFEED2[i - 1, j - 1] = 0
                    else:
                        FRACFEED2[i - 1, j - 1] = FEED2QNTY[i - 1, j - 1] / FEEDQNTY[i - 1, j - 1]

                    if TIME[i - 1, j - 1] <= 14 or FEEDQNTY[i - 1, j - 1] == 0:
                        FRACFEED3[i - 1, j - 1] = 0
                    else:
                        FRACFEED3[i - 1, j - 1] = FEED3QNTY[i - 1, j - 1] / FEEDQNTY[i - 1, j - 1]

                    if TIME[i - 1, j - 1] <= 14 or FEEDQNTY[i - 1, j - 1] == 0:
                        FRACFEED4[i - 1, j - 1] = 0
                    else:
                        FRACFEED4[i - 1, j - 1] = FEED4QNTY[i - 1, j - 1] / FEEDQNTY[i - 1, j - 1]

                    ############### Tracking / log #########################
                    if DEBUG_LOOP and z in DEBUG_CASES and (DEBUG_I is None or i == DEBUG_I) and (DEBUG_J is None or j == DEBUG_J):
                        print(
                            f"[CHK2 CP/FRAC] z={z} s={s} j={j} i={i} iter={loop_iter} "
                            f"CPAVG={_dbg_scalar(CPAVG[i - 1, j - 1])} "
                            f"FR1={_dbg_scalar(FRACFEED1[i - 1, j - 1])} "
                            f"FR2={_dbg_scalar(FRACFEED2[i - 1, j - 1])} "
                            f"FR3={_dbg_scalar(FRACFEED3[i - 1, j - 1])} "
                            f"FR4={_dbg_scalar(FRACFEED4[i - 1, j - 1])}",
                            flush=True
                        )
                    ########################################################                
########################## line:3679 ##############################
                    #######################################################################################
                    #                         Feed intake and digestion sub-model (again)                 #
                    #######################################################################################

                    # Maximum feed intake in kg, based on fill units (FU FU-1 kg)
                    #PHFEEDINTKG[i - 1, j - 1] = PHFEEDINT[i - 1, j - 1] / (
                    #    FRACFEED1[i - 1, j - 1] * FEED1[i - 1, 1] +
                    #    FRACFEED2[i - 1, j - 1] * FEED2[i - 1, 1] +
                    #    FRACFEED3[i - 1, j - 1] * FEED3[i - 1, 1] +
                    #    FRACFEED4[i - 1, j - 1] * FEED4[1]
                    #)

                    # Maximum feed intake in kg, based on fill units (FU FU-1 kg)
                    phfeedintkg_denom = (
                        FRACFEED1[i - 1, j - 1] * FEED1[i - 1, 1] +
                        FRACFEED2[i - 1, j - 1] * FEED2[i - 1, 1] +
                        FRACFEED3[i - 1, j - 1] * FEED3[i - 1, 1] +
                        FRACFEED4[i - 1, j - 1] * FEED4[1]
                    )

                    if phfeedintkg_denom <= 0:
                        PHFEEDINTKG[i - 1, j - 1] = 0.0
                    else:
                        PHFEEDINTKG[i - 1, j - 1] = PHFEEDINT[i - 1, j - 1] / phfeedintkg_denom
                    # Digestion of carbohydrates
                    # INSC = insoluble, non-structural carbohydrates, which are assumed to mainly consist
                    # of starch)

                    pidx = int(PASSAGE[i - 1, j - 1]) - 1

                    # INSC digestion in the rumen (g INSC per animal per day)
                    INSC[i - 1, j - 1] = (
                        FEED1QNTY[i - 1, j - 1] * FEED1[i - 1, 3] * FEED1[i - 1, 8] /
                        (FEED1[i - 1, 8] + FEED1[i - 1, 11] * PASSRED[pidx]) +
                        FEED2QNTY[i - 1, j - 1] * FEED2[i - 1, 3] * FEED2[i - 1, 8] /
                        (FEED2[i - 1, 8] + FEED2[i - 1, 11] * PASSRED[pidx]) +
                        FEED3QNTY[i - 1, j - 1] * FEED3[i - 1, 3] * FEED3[i - 1, 8] /
                        (FEED3[i - 1, 8] + FEED3[i - 1, 11] * PASSRED[pidx]) +
                        FEED4QNTY[i - 1, j - 1] * FEED4[3] * FEED4[8] /
                        (FEED4[8] + FEED4[11] * PASSRED[pidx])
                    )
                    # Total intake INSC in the feed (g INSC per animal per day)
                    INSCTOTAL[i - 1, j - 1] = (
                        FEED1QNTY[i - 1, j - 1] * FEED1[i - 1, 3] +
                        FEED2QNTY[i - 1, j - 1] * FEED2[i - 1, 3] +
                        FEED3QNTY[i - 1, j - 1] * FEED3[i - 1, 3] +
                        FEED4QNTY[i - 1, j - 1] * FEED4[3]
                    )
                    # Fraction insoluble, non-structural carbohydrates digested in the rumen (-)
                    # (compare to Owens, 1986)
                    #INSCDIG[i - 1, j - 1] = INSC[i - 1, j - 1] / INSCTOTAL[i - 1, j - 1]
                    # Fraction insoluble, non-structural carbohydrates digested in the rumen (-)
                    if INSCTOTAL[i - 1, j - 1] <= 0:
                        INSCDIG[i - 1, j - 1] = 0.0
                    else:
                        INSCDIG[i - 1, j - 1] = INSC[i - 1, j - 1] / INSCTOTAL[i - 1, j - 1]
                    # Digestibility of INSC in the intestines is assumed to be 97% for all feeds
                    # (Moharrery et al, 2014)
                    INSCINT[i - 1, j - 1] = max(
                        0,
                        (INSCTOTAL[i - 1, j - 1] * TTDIGINSC) - INSC[i - 1, j - 1]
                    )
                    # Fraction INSC digested in the intestines (-)
                    #INSCINTDIG[i - 1, j - 1] = INSCINT[i - 1, j - 1] / INSCTOTAL[i - 1, j - 1]
                    # Fraction INSC digested in the intestines (-)
                    if INSCTOTAL[i - 1, j - 1] <= 0:
                        INSCINTDIG[i - 1, j - 1] = 0.0
                    else:
                        INSCINTDIG[i - 1, j - 1] = INSCINT[i - 1, j - 1] / INSCTOTAL[i - 1, j - 1]
                    # Digestion degradable neutral detergent fibre (NDF, g day-1)
                    NDF[i - 1, j - 1] = (
                        FEED1QNTY[i - 1, j - 1] * FEED1[i - 1, 4] * FEED1[i - 1, 9] /
                        (FEED1[i - 1, 9] + FEED1[i - 1, 11] * PASSRED[pidx]) +
                        FEED2QNTY[i - 1, j - 1] * FEED2[i - 1, 4] * FEED2[i - 1, 9] /
                        (FEED2[i - 1, 9] + FEED2[i - 1, 11] * PASSRED[pidx]) +
                        FEED3QNTY[i - 1, j - 1] * FEED3[i - 1, 4] * FEED3[i - 1, 9] /
                        (FEED3[i - 1, 9] + FEED3[i - 1, 11] * PASSRED[pidx]) +
                        FEED4QNTY[i - 1, j - 1] * FEED4[4] * FEED4[9] /
                        (FEED4[9] + FEED4[11] * PASSRED[pidx])
                    )
                    # Total intake NDF (g day-1)
                    NDFTOTAL[i - 1, j - 1] = (
                        FEED1QNTY[i - 1, j - 1] * FEED1[i - 1, 4] +
                        FEED2QNTY[i - 1, j - 1] * FEED2[i - 1, 4] +
                        FEED3QNTY[i - 1, j - 1] * FEED3[i - 1, 4] +
                        FEED4QNTY[i - 1, j - 1] * FEED4[4]
                    )
                    # Fraction degradable NDF digested in the rumen (-)
                    #NDFDIG[i - 1, j - 1] = NDF[i - 1, j - 1] / NDFTOTAL[i - 1, j - 1]
                    # Fraction degradable NDF digested in the rumen (-)
                    if NDFTOTAL[i - 1, j - 1] <= 0:
                        NDFDIG[i - 1, j - 1] = 0.0
                    else:
                        NDFDIG[i - 1, j - 1] = NDF[i - 1, j - 1] / NDFTOTAL[i - 1, j - 1]
                    # NDF digestion in the intestines (g day-1)
                    # Cabral et al., 2011 (http://www.scielo.br/pdf/rbz/v40n9/a20v40n9.pdf)
                    # Volative fatty acids released are assumed not to be taken up by the animal
                    NDFINT[i - 1, j - 1] = (
                        FEED1QNTY[i - 1, j - 1] * FEED1[i - 1, 4] *
                        (1 - FEED1[i - 1, 9] / (FEED1[i - 1, 9] + FEED1[i - 1, 11] * PASSRED[pidx])) *
                        (FEED1[i - 1, 9] * NDFDIGEST * SENSMAT[109, s - 1]) /
                        (FEED1[i - 1, 9] * NDFDIGEST * SENSMAT[109, s - 1] + NDFPASS * SENSMAT[110, s - 1]) +
                        FEED2QNTY[i - 1, j - 1] * FEED2[i - 1, 4] *
                        (1 - FEED2[i - 1, 9] / (FEED2[i - 1, 9] + FEED2[i - 1, 11] * PASSRED[pidx])) *
                        (FEED2[i - 1, 9] * NDFDIGEST * SENSMAT[109, s - 1]) /
                        (FEED2[i - 1, 9] * NDFDIGEST * SENSMAT[109, s - 1] + NDFPASS * SENSMAT[110, s - 1]) +
                        FEED3QNTY[i - 1, j - 1] * FEED3[i - 1, 4] *
                        (1 - FEED3[i - 1, 9] / (FEED3[i - 1, 9] + FEED3[i - 1, 11] * PASSRED[pidx])) *
                        (FEED3[i - 1, 9] * NDFDIGEST * SENSMAT[109, s - 1]) /
                        (FEED3[i - 1, 9] * NDFDIGEST * SENSMAT[109, s - 1] + NDFPASS * SENSMAT[110, s - 1]) +
                        FEED4QNTY[i - 1, j - 1] * FEED4[4] *
                        (1 - FEED4[9] / (FEED4[9] + FEED4[11] * PASSRED[pidx])) *
                        (FEED4[9] * NDFDIGEST * SENSMAT[109, s - 1]) /
                        (FEED4[9] * NDFDIGEST * SENSMAT[109, s - 1] + NDFPASS * SENSMAT[110, s - 1])
                    )

                    # Fraction degradable NDF digested in intestines (-)
                    #NDFINTDIG[i - 1, j - 1] = NDFINT[i - 1, j - 1] / NDFTOTAL[i - 1, j - 1]
                    # Fraction degradable NDF digested in intestines (-)
                    if NDFTOTAL[i - 1, j - 1] <= 0:
                        NDFINTDIG[i - 1, j - 1] = 0.0
                    else:
                        NDFINTDIG[i - 1, j - 1] = NDFINT[i - 1, j - 1] / NDFTOTAL[i - 1, j - 1]
                    # Fraction degradable NDF digested in intestines (g kg-1 DM feed)
                    #NDFINTDIGTOT[i - 1, j - 1] = NDFINT[i - 1, j - 1] / (FEEDQNTY[i - 1, j - 1] * 1000)
                    # Fraction degradable NDF digested in intestines (g kg-1 DM feed)
                    ndfintdigtot_denom = FEEDQNTY[i - 1, j - 1] * 1000.0
                    if ndfintdigtot_denom <= 0:
                        NDFINTDIGTOT[i - 1, j - 1] = 0.0
                    else:
                        NDFINTDIGTOT[i - 1, j - 1] = NDFINT[i - 1, j - 1] / ndfintdigtot_denom
                    # Protein digestion
                    # Degradable crude protein (DCP) digestion in the rumen (g day-1)
                    PICP[i - 1, j - 1] = (
                        FEED1QNTY[i - 1, j - 1] * FEED1[i - 1, 6] * FEED1[i - 1, 10] /
                        (FEED1[i - 1, 10] + FEED1[i - 1, 11] * PASSRED[pidx]) +
                        FEED2QNTY[i - 1, j - 1] * FEED2[i - 1, 6] * FEED2[i - 1, 10] /
                        (FEED2[i - 1, 10] + FEED2[i - 1, 11] * PASSRED[pidx]) +
                        FEED3QNTY[i - 1, j - 1] * FEED3[i - 1, 6] * FEED3[i - 1, 10] /
                        (FEED3[i - 1, 10] + FEED3[i - 1, 11] * PASSRED[pidx]) +
                        FEED4QNTY[i - 1, j - 1] * FEED4[6] * FEED4[10] /
                        (FEED4[10] + FEED4[11] * PASSRED[pidx])
                    )
                    # Total crude protein intake (g day-1)
                    PROTTOTAL[i - 1, j - 1] = (
                        FEED1QNTY[i - 1, j - 1] * FEED1[i - 1, 15] +
                        FEED2QNTY[i - 1, j - 1] * FEED2[i - 1, 15] +
                        FEED3QNTY[i - 1, j - 1] * FEED3[i - 1, 15] +
                        FEED4QNTY[i - 1, j - 1] * FEED4[15]
                    )
                    # Protein ending up in the intestines (g day-1)
                    PROTINT[i - 1, j - 1] = (
                        PROTTOTAL[i - 1, j - 1] - (
                            FEED1QNTY[i - 1, j - 1] * FEED1[i - 1, 5] +
                            FEED2QNTY[i - 1, j - 1] * FEED2[i - 1, 5] +
                            FEED3QNTY[i - 1, j - 1] * FEED3[i - 1, 5] +
                            FEED4QNTY[i - 1, j - 1] * FEED4[5]
                        ) - PICP[i - 1, j - 1]
                    )

                    # Lucas equation (g protein day-1), whole digestive tract, plus the effect of recycling
                    # This equation corresponds partly to Eq. 24 of the Supplementary Information and
                    # indicates the amount of protein taken up in the whole digestive tract.
                    PROTUPT[i - 1, j - 1] = (
                        (LUCAS1 * SENSMAT[111, s - 1]) * PROTTOTAL[i - 1, j - 1] -
                        (LUCAS2 * SENSMAT[112, s - 1]) * FEEDQNTY[i - 1, j - 1]
                    )
                    # Protein excreted (g protein day-1)
                    PROTEXCR[i - 1, j - 1] = PROTTOTAL[i - 1, j - 1] - PROTUPT[i - 1, j - 1]
                    # Fraction protein digested in rumen (-)
                    #PROTDIGRU[i - 1, j - 1] = (
                    #    PROTTOTAL[i - 1, j - 1] - PROTINT[i - 1, j - 1]
                    #) / PROTTOTAL[i - 1, j - 1]
                    # Fraction protein digested in rumen (-)
                    if PROTTOTAL[i - 1, j - 1] <= 0:
                        PROTDIGRU[i - 1, j - 1] = 0.0
                    else:
                        PROTDIGRU[i - 1, j - 1] = (
                            PROTTOTAL[i - 1, j - 1] - PROTINT[i - 1, j - 1]
                        ) / PROTTOTAL[i - 1, j - 1]
                    # Fraction protein digested in the whole digestive tract (-)
                    #PROTDIGWT[i - 1, j - 1] = PROTUPT[i - 1, j - 1] / PROTTOTAL[i - 1, j - 1]
                    # Fraction protein digested in the whole digestive tract (-)
                    if PROTTOTAL[i - 1, j - 1] <= 0:
                        PROTDIGWT[i - 1, j - 1] = 0.0
                    else:
                        PROTDIGWT[i - 1, j - 1] = PROTUPT[i - 1, j - 1] / PROTTOTAL[i - 1, j - 1]
                    # Digestion and excretion
                    # Feed digested in the whole digestive tract (g day-1)
                    DIGFRAC[i - 1, j - 1] = (
                        FEED1QNTY[i - 1, j - 1] * (FEED1[i - 1, 2] + FEED1[i - 1, 5]) +
                        FEED2QNTY[i - 1, j - 1] * (FEED2[i - 1, 2] + FEED2[i - 1, 5]) +
                        FEED3QNTY[i - 1, j - 1] * (FEED3[i - 1, 2] + FEED3[i - 1, 5]) +
                        FEED4QNTY[i - 1, j - 1] * (FEED4[2] + FEED4[5]) +
                        INSC[i - 1, j - 1] + INSCINT[i - 1, j - 1] +
                        NDF[i - 1, j - 1] + NDFINT[i - 1, j - 1] +
                        PROTUPT[i - 1, j - 1]
                    )

                    # Carbohydrates excreted (g day-1)
                    CHEXCR[i - 1, j - 1] = FEEDQNTY[i - 1, j - 1] * 1000 - DIGFRAC[i - 1, j - 1] - PROTEXCR[i - 1, j - 1]
                    # Manure dry matter excreted (g day-1)
                    EXCRFRAC[i - 1, j - 1] = FEEDQNTY[i - 1, j - 1] * 1000 - DIGFRAC[i - 1, j - 1]
                    # Gross energy (GE) content excreted dry matter (MJ GE kg-1 DM)
                    #GEEXCR[i - 1, j - 1] = (
                    #    PROTEXCR[i - 1, j - 1] * GEPROT + CHEXCR[i - 1, j - 1] * GECARB
                    #) / (PROTEXCR[i - 1, j - 1] + CHEXCR[i - 1, j - 1])
                    # Gross energy (GE) content excreted dry matter (MJ GE kg-1 DM)
                    geexcr_denom = PROTEXCR[i - 1, j - 1] + CHEXCR[i - 1, j - 1]
                    if geexcr_denom <= 0:
                        GEEXCR[i - 1, j - 1] = 0.0
                    else:
                        GEEXCR[i - 1, j - 1] = (
                            PROTEXCR[i - 1, j - 1] * GEPROT + CHEXCR[i - 1, j - 1] * GECARB
                        ) / geexcr_denom
                    # GE content digested feed (MJ GE kg-1 DM)
                    # This equation is similar to Eq. 25 in the Supplementary Information
                    #GEUPTAKE[i - 1, j - 1] = (
                    #    PROTUPT[i - 1, j - 1] * GEPROT +
                    #    (DIGFRAC[i - 1, j - 1] - PROTUPT[i - 1, j - 1]) * GECARB
                    #) / DIGFRAC[i - 1, j - 1]
                    # GE content digested feed (MJ GE kg-1 DM)
                    # This equation is similar to Eq. 25 in the Supplementary Information
                    if DIGFRAC[i - 1, j - 1] <= 0:
                        GEUPTAKE[i - 1, j - 1] = 0.0
                    else:
                        GEUPTAKE[i - 1, j - 1] = (
                            PROTUPT[i - 1, j - 1] * GEPROT +
                            (DIGFRAC[i - 1, j - 1] - PROTUPT[i - 1, j - 1]) * GECARB
                        ) / DIGFRAC[i - 1, j - 1]
                    # ME uptake (MJ ME day-1); 0.82 is conversion DE --> ME
                    if EXCRFRAC[i - 1, j - 1] == 0:
                        MEUPTAKE[i - 1, j - 1] = 0
                    else:
                        MEUPTAKE[i - 1, j - 1] = DIGFRAC[i - 1, j - 1] / 1000 * GEUPTAKE[i - 1, j - 1] * DETOME

                    # Digestibility feed (g g-1), on an energy basis
                    if EXCRFRAC[i - 1, j - 1] == 0:
                        Q[i - 1, j - 1] = 0
                    else:
                        Q[i - 1, j - 1] = (
                            DIGFRAC[i - 1, j - 1] / (FEEDQNTY[i - 1, j - 1] * 1000) *
                            (GEUPTAKE[i - 1, j - 1] / GEFEED)
                        )
                    #################### trackinh / log ##########################
                    if DEBUG_LOOP and z in DEBUG_CASES and (DEBUG_I is None or i == DEBUG_I) and (DEBUG_J is None or j == DEBUG_J):
                        print(
                            f"[CHK3 DIGEST] z={z} s={s} j={j} i={i} iter={loop_iter} "
                            f"MEUPTAKE={_dbg_scalar(MEUPTAKE[i - 1, j - 1])} "
                            f"Q={_dbg_scalar(Q[i - 1, j - 1])} "
                            f"PROTUPT={_dbg_scalar(PROTUPT[i - 1, j - 1])} "
                            f"Digestfracfeed={_dbg_scalar(Digestfracfeed[i - 1, j - 1])}"
                        )
                    ######################################################### 
                    # Average heat increment of feeding (MJ MJ-1)
                    if (FEED1QNTY[i - 1, j - 1] + FEED2QNTY[i - 1, j - 1] +
                        FEED3QNTY[i - 1, j - 1] + FEED4QNTY[i - 1, j - 1]) == 0:
                        Digestfracfeed[i - 1, j - 1] = 0.3
                    else:
                        Digestfracfeed[i - 1, j - 1] = (
                            FEED1QNTY[i - 1, j - 1] * FEED1[i - 1, 0] +
                            FEED2QNTY[i - 1, j - 1] * FEED2[i - 1, 0] +
                            FEED3QNTY[i - 1, j - 1] * FEED3[i - 1, 0] +
                            FEED4QNTY[i - 1, j - 1] * FEED4[0]
                        ) / (
                            FEED1QNTY[i - 1, j - 1] + FEED2QNTY[i - 1, j - 1] +
                            FEED3QNTY[i - 1, j - 1] + FEED4QNTY[i - 1, j - 1]
                        )

                    # If feed intake is not reduced, no energy requirement for respiration above
                    # maintenance
                    # Increase in net energy (NE) requirements due to heat stress (MJ NE day-1)
                    if REDHP[i - 1, j - 1] == 0:
                        NERESPC[i - 1, j - 1] = 0
                    else:
                        NERESPC[i - 1, j - 1] = NERESP[i - 1, j - 1] / 1000
                    # Increase in protein requirements under maximum heat release and heat stress (g day-1)
                    PROTRESP[i - 1, j - 1] = NERESPC[i - 1, j - 1] * PROTNE * NtoCP
########################## line:3848 ##############################
                                            #######################################################################################

                                            # Milk for the calf from the cow until weaning (MJ GE day-1)
                    if TIME[i - 1, j - 1] <= WEANINGTIME:
                        MILKSTART[i - 1, j - 1] = (
                            LIBRARY[14] *
                            TIME[i - 1, j - 1] ** MILKPARB *
                            np.exp(-MILKPARC * TIME[i - 1, j - 1]) *
                            (((GEMILK1 * TIME[i - 1, j - 1]) + 2771) / 1000) *
                            MILKDIG
                        )
                    else:
                        MILKSTART[i - 1, j - 1] = 0

                    # Digested protein from milk for the calf (g day-1)
                    # 95% protein digestibility; 3.5% protein in milk, heat increment of feeding taken into
                    # account
                    if TIME[i - 1, j - 1] <= WEANINGTIME:
                        MILKSTARTPR[i - 1, j - 1] = (
                            LIBRARY[14] *
                            TIME[i - 1, j - 1] ** MILKPARB *
                            np.exp(-MILKPARC * TIME[i - 1, j - 1]) *
                            35 * MILKDIG
                        )
                    else:
                        MILKSTARTPR[i - 1, j - 1] = 0

                    MILKSTARTPRHF[i - 1, j - 1] = (
                        MILKSTARTPR[i - 1, j - 1] *
                        (1 + (Digestfracfeed[i - 1, j - 1] / (1 - Digestfracfeed[i - 1, j - 1])))
                    )

                    # Milk from reproductive cow for calf (MJ ME day-1)
                    MEMILKCALFINIT[i - 1, j - 1] = (
                        MILKSTART[i - 1, j - 1] *
                        (1 + (Digestfracfeed[i - 1, j - 1] / (1 - Digestfracfeed[i - 1, j - 1])))
                    )

                    # Protein for non-growth purposes
                    # Total fixed protein requirements (g protein day-1)
                    PROTNONG[i - 1, j - 1] = (
                        PROTDERML[i - 1, j - 1] + PROTMAINT[i - 1, j - 1] +
                        PROTPHACT[i - 1, j - 1] + PROTGESTG[i - 1, j - 1] +
                        PROTMILKG[i - 1, j - 1] + PROTRESP[i - 1, j - 1]
                    )
                    # Heat increment of feeding (MJ day-1)
                    HIFM[i - 1, j - 1] = (
                        (NEMAINT[i - 1, j - 1] / 1000 + NEPHYSACT[i - 1, j - 1] / 1000 +
                            NEREQGEST[i - 1, j - 1] + NEMILKCOW[i - 1, j - 1] +
                            NERESPC[i - 1, j - 1] / 1000) *
                        (Digestfracfeed[i - 1, j - 1] / (1 - Digestfracfeed[i - 1, j - 1]))
                    )
                    # Protein requirements for fixed processes, i.e. excluding growth (g protein day-1)
                    PROTNONGM[i - 1, j - 1] = (
                        PROTNONG[i - 1, j - 1] +
                        HIFM[i - 1, j - 1] * PROTNE * NtoCP
                    )

                    # Energy not invested in growth is converted into heat (MJ day-1)
                    HEATIFEEDMAINT[i - 1, j - 1] = (
                        NEMAINT[i - 1, j - 1] / 1000 +
                        NEPHYSACT[i - 1, j - 1] / 1000 +
                        HEATGEST[i - 1, j - 1] +
                        HEATMILK[i - 1, j - 1] +
                        NERESPC[i - 1, j - 1] / 1000 +
                        HIFM[i - 1, j - 1] +
                        HEATIFEEDGROWTHC[i - 1, j - 1] / DISSEFF +
                        REDMAINT[i - 1, j - 1]
                    )

                    # Sum of all heat released that is not related to growth, includes heat increment of
                    # feeding (MJ day-1)
                    HEATIFEEDMAINTWM[i - 1, j - 1] = (
                        HEATIFEEDMAINT[i - 1, j - 1] /
                        (3600 * 24 * AREA[i - 1, j - 1]) *
                        1000000
                    )

                    # Heat release under maintenance level is not taken into account yet.
                    # Maximum heat release from growth (W m-2)
                    HEATIFEEDGROWTHWM[i - 1, j - 1] = (
                        Metheatopt[i - 1, j - 1] -
                        HEATIFEEDMAINTWM[i - 1, j - 1]
                    )
                    # Maximum heat release from growth (MJ day-1)
                    HEATIFEEDGROWTH[i - 1, j - 1] = (
                        HEATIFEEDGROWTHWM[i - 1, j - 1] *
                        (3600 * 24 * AREA[i - 1, j - 1]) / 1000000
                    )

                    # Superfluous heat produced under sub-maintenance level is taken into account here
                    # (MJ day-1)
                    if HEATIFEEDGROWTH[i - 1, j - 1] < 0:
                        REDMAINT[i - 1, j - 1] = HEATIFEEDGROWTH[i - 1, j - 1]
                    else:
                        REDMAINT[i - 1, j - 1] = 0

                    # Energy available for growth, after accounting for heat increment of feeding (MJ NE
                    # day-1)
                    ENFEEDGROWTHQ[i, j - 1] = (
                        (MEUPTAKE[i - 1, j - 1]) -
                        HEATIFEEDMAINT[i - 1, j - 1] +
                        MEMILKCALFINIT[i - 1, j - 1]
                    ) / (
                        1 + (Digestfracfeed[i - 1, j - 1] / (1 - Digestfracfeed[i - 1, j - 1]))
                    )
                    ENFEEDGROWTHQ[i, j - 1] = (
                        ENFEEDGROWTHQ[i, j - 1] -
                        0.134 * NEREQGEST[i - 1, j - 1] -
                        (NEMILKCOW[i - 1, j - 1] - HEATMILK[i - 1, j - 1])
                    )

                    # Integrates the net energy for growth, based on the genetic potential (ENGRTOTAL),
                    # the climate (REDHP), and feed-limitation (ENFEEDGROWTHQ)
                    # Energy available for growth (MJ NE day-1)
                    ENFEEDGROWTH[i, j - 1] = min(
                        ENFEEDGROWTHQ[i, j - 1],
                        ENGRTOTAL[i, j - 1] * COMPGROWTH[i - 1, j - 1]
                    ) + REDHP[i - 1, j - 1]

                    # Fraction NE for growth allocated to the non-carcass tissue (-)
                    FRENGRNONCBF[i, j - 1] = max(
                        0,
                        (ENFEEDGROWTH[i, j - 1] - FATCOMP[i - 1, j - 1]) /
                        ENFEEDGROWTH[i, j - 1]
                    )
                    if NONCARCTISBF[i - 1, j - 1] / NONCARCTIS[i - 1, j - 1] < 1:
                        FRENGRNONCBF[i, j - 1] = FRENGRNONCBF[i, j - 1]
                    else:
                        FRENGRNONCBF[i, j - 1] = 0
                    # NE for growth allocated to the non-carcass tissue (MJ day-1)
                    ENGRNONCBF[i, j - 1] = (
                        FRENGRNONCBF[i, j - 1] *
                        (ENGRNONC[i, j - 1] / ENGRTOTALORIG[i, j - 1]) *
                        ENFEEDGROWTH[i, j - 1] *
                        COMPGROWTH1[i - 1, j - 1] *
                        (
                            ((TBWBF[i - 1, j - 1] / LIBRARY[12]) *
                                -(ENNONC1 * SENSMAT[113, s - 1] - ENNONC2 * SENSMAT[114, s - 1]) +
                                ENNONC1 * SENSMAT[113, s - 1])
                        ) /
                        (ENGRNONC[i, j - 1] / ENGRTOTALORIG[i, j - 1])
                    )
                    ENGRNONCBF[i, j - 1] = max(ENGRNONCBF[i, j - 1], 0)  # NE for growth cannot be negative

                    # Fraction NE for growth allocated to the bone tissue (-)
                    FRENGRBONEBF[i, j - 1] = max(
                        0,
                        (ENFEEDGROWTH[i, j - 1] - FATCOMP[i - 1, j - 1]) /
                        ENFEEDGROWTH[i, j - 1]
                    )
                    if BONETISBF[i - 1, j - 1] / BONETIS[i - 1, j - 1] < 1:
                        FRENGRBONEBF[i, j - 1] = FRENGRBONEBF[i, j - 1]
                    else:
                        FRENGRBONEBF[i, j - 1] = 0
                    # NE for growth allocated to the bone tissue (MJ day-1)
                    ENGRBONEBF[i, j - 1] = (
                        FRENGRBONEBF[i, j - 1] *
                        (ENGRBONE[i, j - 1] / ENGRTOTALORIG[i, j - 1]) *
                        ENFEEDGROWTH[i, j - 1] *
                        COMPGROWTH2[i - 1, j - 1]
                    )
                    ENGRBONEBF[i, j - 1] = max(ENGRBONEBF[i, j - 1], 0)  # NE for growth cannot be negative

                    # Fraction NE for growth allocated to the muscle tissue (-)
                    FRENGRMUSCLEBF[i, j - 1] = max(
                        0,
                        (ENFEEDGROWTH[i, j - 1] - FATCOMP[i - 1, j - 1]) /
                        ENFEEDGROWTH[i, j - 1]
                    )
                    if MUSCLETISBF[i - 1, j - 1] / MUSCLETIS[i - 1, j - 1] < 1:
                        FRENGRMUSCLEBF[i, j - 1] = FRENGRMUSCLEBF[i, j - 1]
                    else:
                        FRENGRMUSCLEBF[i, j - 1] = 0
                    # NE for growth allocated to the muscle tissue (MJ day-1)
                    ENGRMUSCLEBF[i, j - 1] = (
                        FRENGRMUSCLEBF[i, j - 1] *
                        (ENGRMUSCLE[i, j - 1] / ENGRTOTALORIG[i, j - 1]) *
                        ENFEEDGROWTH[i, j - 1] *
                        COMPGROWTH3[i - 1, j - 1]
                    )
                    ENGRMUSCLEBF[i, j - 1] = max(ENGRMUSCLEBF[i, j - 1], 0)  # NE for growth cannot be negative

                    # Fraction NE for growth allocated to the intramuscular fat tissue (-)
                    FRENGRIMFBF[i, j - 1] = max(
                        0,
                        (ENFEEDGROWTH[i, j - 1] - FATCOMP[i - 1, j - 1]) /
                        ENFEEDGROWTH[i, j - 1]
                    )
                    if INTRAMFTISBF[i - 1, j - 1] / INTRAMFTIS[i - 1, j - 1] < 1:
                        FRENGRIMFBF[i, j - 1] = FRENGRIMFBF[i, j - 1]
                    else:
                        FRENGRIMFBF[i, j - 1] = 0
                    # NE for growth allocated to the intramuscular fat tissue (MJ day-1)
                    ENGRIMFBF[i, j - 1] = (
                        FRENGRIMFBF[i, j - 1] *
                        (ENGRIMF[i, j - 1] / ENGRTOTALORIG[i, j - 1]) *
                        ENFEEDGROWTH[i, j - 1] *
                        COMPGROWTH4[i - 1, j - 1]
                    )
                    ENGRIMFBF[i, j - 1] = max(ENGRIMFBF[i, j - 1], 0)  # NE for growth cannot be negative

                    # NE for growth allocated to the miscellaneous fat tissue (MJ day-1); balancing
                    # variable
                    ENGRFATBF[i, j - 1] = (
                        ENFEEDGROWTH[i, j - 1] - ENGRNONCBF[i, j - 1] -
                        ENGRBONEBF[i, j - 1] - ENGRMUSCLEBF[i, j - 1] -
                        ENGRIMFBF[i, j - 1]
                    )
                    ENGRFATBF[i, j - 1] = max(ENGRFATBF[i, j - 1], 0)  # NE for growth cannot be negative

                    # Check on NE for growth (MJ day-1); positive values CHECKCOMP are wrong
                    ENGRTOTALCOMP[i, j - 1] = (
                        ENGRNONCBF[i, j - 1] + ENGRBONEBF[i, j - 1] +
                        ENGRMUSCLEBF[i, j - 1] + ENGRIMFBF[i, j - 1] +
                        ENGRFATBF[i, j - 1]
                    )
                    #orginal
                    #CHECKCOMP[i, j - 1] = ENFEEDGROWTH[i, j - 1] - ENGRTOTALCOMP[i, j - 1]
                    CHECKCOMP[i - 1, j - 1] = ENFEEDGROWTH[i, j - 1] - ENGRTOTALCOMP[i, j - 1]

                    # Heat production actual growth (13.9 MJ kg-1 for lipid, and 20.2 MJ kg-1 for protein)
                    # Heat production related to growth of bone tissue (MJ day-1)
                    HEATBONEACT[i - 1, j - 1] = (
                        DERBONE[i - 1, j - 1] * ENGRBONEBF[i, j - 1] / ENGRBONE[i, j - 1] *
                        (
                            LIPIDFRACBONEBF[i - 1, j - 1] * (GELIPID / LIPIDEFF - GELIPID) +
                            PROTFRACBONE * (GEPROT / PROTEFF - GEPROT)
                        )
                    )
                    # Heat production related to growth of muscle tissue (MJ day-1)
                    HEATMUSCLEACT[i - 1, j - 1] = (
                        DERMUSCLE[i - 1, j - 1] * ENGRMUSCLEBF[i, j - 1] / ENGRMUSCLE[i, j - 1] *
                        (
                            LIPFRACMUSCLE * (GELIPID / LIPIDEFF - GELIPID) +
                            PROTFRACMUSCLE * (GEPROT / PROTEFF - GEPROT)
                        )
                    )
                    # Heat production related to growth of intramuscular fat tissue (MJ day-1)
                    HEATIMFACT[i - 1, j - 1] = (
                        DERINTRAMF[i - 1, j - 1] * ENGRIMFBF[i, j - 1] / ENGRIMF[i, j - 1] *
                        (
                            LIPFRACFAT * (GELIPID / LIPIDEFF - GELIPID) +
                            PROTFRACFAT * (GEPROT / PROTEFF - GEPROT)
                        )
                    )
                    # Heat production related to growth of miscellaneous fat tissue (MJ day-1)
                    HEATMISCFATACT[i - 1, j - 1] = (
                        DERMISCFAT[i - 1, j - 1] * ENGRFATBF[i, j - 1] / ENGRFAT[i, j - 1] *
                        (
                            LIPFRACFAT * (GELIPID / LIPIDEFF - GELIPID) +
                            PROTFRACFAT * (GEPROT / PROTEFF - GEPROT)
                        )
                    )
                    # Heat production related to growth of non-carcass tissue (MJ day-1)
                    HEATNONCACT[i - 1, j - 1] = (
                        DERNONC[i - 1, j - 1] * ENGRNONCBF[i, j - 1] / ENGRNONC[i, j - 1] *
                        (
                            LIPIDFRACNONCBF[i - 1, j - 1] * (GELIPID / LIPIDEFF - GELIPID) +
                            PROTFRACNONCBF[i - 1, j - 1] * (GEPROT / PROTEFF - GEPROT)
                        )
                    )
                    # Total heat production related to NE for growth (MJ day-1)
                    HEATTOTALACT[i - 1, j - 1] = (
                        HEATBONEACT[i - 1, j - 1] + HEATMUSCLEACT[i - 1, j - 1] +
                        HEATIMFACT[i - 1, j - 1] + HEATMISCFATACT[i - 1, j - 1] +
                        HEATNONCACT[i - 1, j - 1]
                    )

                    # Net energy (NE) requirements for growth (MJ day-1)
                    # 44.0 MJ kg-1 for protein, 53.7 MJ kg-1 for lipid
                    # NE requirements for growth of the bone tissue (MJ day-1)
                    ENBONEACT[i - 1, j - 1] = (
                        DERBONE[i - 1, j - 1] * ENGRBONEBF[i, j - 1] / ENGRBONE[i, j - 1] *
                        (
                            LIPIDFRACBONEBF[i - 1, j - 1] * GELIPID / LIPIDEFF +
                            PROTFRACBONE * GEPROT / PROTEFF
                        )
                    )
                    # NE requirements for growth of the muscle tissue (MJ day-1)
                    ENMUSCLEACT[i - 1, j - 1] = (
                        DERMUSCLE[i - 1, j - 1] * ENGRMUSCLEBF[i, j - 1] / ENGRMUSCLE[i, j - 1] *
                        (
                            LIPFRACMUSCLE * GELIPID / LIPIDEFF +
                            PROTFRACMUSCLE * GEPROT / PROTEFF
                        )
                    )
                    # NE requirements for growth of the intramuscular fat tissue (MJ day-1)
                    ENIMFACT[i - 1, j - 1] = (
                        DERINTRAMF[i - 1, j - 1] * ENGRIMFBF[i, j - 1] / ENGRIMF[i, j - 1] *
                        (
                            LIPFRACFAT * GELIPID / LIPIDEFF +
                            PROTFRACFAT * GEPROT / PROTEFF
                        )
                    )
                    # NE requirements for growth of the miscellaneous fat tissue (MJ day-1)
                    ENMISCFATACT[i - 1, j - 1] = (
                        DERMISCFAT[i - 1, j - 1] * ENGRFATBF[i, j - 1] / ENGRFAT[i, j - 1] *
                        (
                            LIPFRACFAT * GELIPID / LIPIDEFF +
                            PROTFRACFAT * GEPROT / PROTEFF
                        )
                    )
                    # NE requirements for growth of the non-carcass tissue (MJ day-1)
                    ENNONCACT[i - 1, j - 1] = (
                        DERNONC[i - 1, j - 1] * ENGRNONCBF[i, j - 1] / ENGRNONC[i, j - 1] *
                        (
                            LIPIDFRACNONCBF[i - 1, j - 1] * GELIPID / LIPIDEFF +
                            PROTFRACNONCBF[i - 1, j - 1] * GEPROT / PROTEFF
                        )
                    )
                    # Total NE requirements for growth of all tissues (MJ day-1)
                    ENTOTALACT[i - 1, j - 1] = (
                        ENBONEACT[i - 1, j - 1] + ENMUSCLEACT[i - 1, j - 1] +
                        ENIMFACT[i - 1, j - 1] + ENMISCFATACT[i - 1, j - 1] +
                        ENNONCACT[i - 1, j - 1]
                    )
########################## line:4064 ##############################
                    #######################################################################################

                    # Protein requirements for growth
                    # Protein use efficiency for growth is assumed to be 54% (23.8/44.0 = 0.54)
                    # Protein requirement for growth of bone tissue (g day-1)
                    PROTBONEACT[i - 1, j - 1] = (
                        DERBONE[i - 1, j - 1] * ENGRBONEBF[i, j - 1] / ENGRBONE[i, j - 1] *
                        PROTFRACBONE / PROTEFF * 1000
                    )
                    # Protein requirement for growth of muscle tissue (g day-1)
                    PROTMUSCLEACT[i - 1, j - 1] = (
                        DERMUSCLE[i - 1, j - 1] * ENGRMUSCLEBF[i, j - 1] / ENGRMUSCLE[i, j - 1] *
                        PROTFRACMUSCLE / PROTEFF * 1000
                    )
                    # Protein requirement for growth of intramuscular fat tissue (g day-1)
                    PROTIMFACT[i - 1, j - 1] = (
                        DERINTRAMF[i - 1, j - 1] * ENGRIMFBF[i, j - 1] / ENGRIMF[i, j - 1] *
                        PROTFRACFAT / PROTEFF * 1000
                    )
                    # Protein requirement for growth of intermuscular and subcutaneous fat tissue (g day-1)
                    PROTMISCFATACT[i - 1, j - 1] = (
                        DERMISCFAT[i - 1, j - 1] * ENGRFATBF[i, j - 1] / ENGRFAT[i, j - 1] *
                        PROTFRACFAT / PROTEFF * 1000
                    )
                    # Protein requirement for growth of non-carcass tissue (g day-1)
                    PROTNONCBF1[i - 1, j - 1] = (
                        DERNONC[i - 1, j - 1] * ENGRNONCBF[i, j - 1] / ENGRNONC[i, j - 1] *
                        PROTFRACNONCBF[i - 1, j - 1] / PROTEFF * 1000
                    )
                    # Total protein requirements for growth (g protein day-1)
                    PROTTOTALACT[i - 1, j - 1] = (
                        PROTBONEACT[i - 1, j - 1] + PROTMUSCLEACT[i - 1, j - 1] +
                        PROTIMFACT[i - 1, j - 1] + PROTMISCFATACT[i - 1, j - 1] +
                        PROTNONCBF1[i - 1, j - 1]
                    )

                    # Total protein requirement (g protein day-1), excluding recycling of protein
                    PROTGROSS[i - 1, j - 1] = (
                        PROTNONGM[i - 1, j - 1] + PROTTOTALACT[i - 1, j - 1] +
                        ENTOTALACT[i - 1, j - 1] *
                        (Digestfracfeed[i - 1, j - 1] / (1 - Digestfracfeed[i - 1, j - 1])) *
                        PROTNE * NtoCP +
                        HEATTOTALACT[i - 1, j - 1] * PROTNE * NtoCP
                    )

                    # Digested protein not accreted in tissues (g day-1)
                    UREABL[i - 1, j - 1] = (
                        PROTGROSS[i - 1, j - 1] - PROTDERML[i - 1, j - 1] +
                        PROTEFF * PROTTOTALACT[i - 1, j - 1] +
                        0.85 * PROTGESTG[i - 1, j - 1] +
                        PROTEFFMILK * PROTMILKG[i - 1, j - 1]
                    )
                    # Percentage of urea N recycled, percentage from N intake (%) (Russel et al, 1992)
                    NRECYCLPT[i - 1, j - 1] = (
                        NRECYCL1 * SENSMAT[115, s - 1] -
                        NRECYCL2 * SENSMAT[116, s - 1] * (CPAVG[i - 1, j - 1] / 10) +
                        NRECYCL3 * SENSMAT[117, s - 1] * (CPAVG[i - 1, j - 1] / 10) ** 2
                    )
                    if TIME[i - 1, j - 1] <= 14:
                        NRECYCLPT[i - 1, j - 1] = 0  # No recycling when only milk is supplied

                    # Net protein requirement (g protein day-1), including recycling
                    PROTNETT[i - 1, j - 1] = (
                        PROTGROSS[i - 1, j - 1] -
                        (NRECYCLPT[i - 1, j - 1] / 100) *
                        (CPAVG[i - 1, j - 1] * FEEDQNTY[i - 1, j - 1])
                    )

                    # Additional energy requirements under cold stress (MJ day-1)
                    HEATIFEEDGROWTHC[i - 1, j - 1] = (
                        HEATIFEEDGROWTHC[i - 1, j - 1] + max(
                            0,
                            (Metheatcold[i - 1, j - 1] - HEATIFEEDMAINTWM[i - 1, j - 1]) *
                            (3600 * 24 * AREA[i - 1, j - 1]) / 1000000 -
                            HEATTOTALACT[i - 1, j - 1] -
                            ENTOTALACT[i - 1, j - 1] *
                            (Digestfracfeed[i - 1, j - 1] / (1 - Digestfracfeed[i - 1, j - 1]))
                        )
                    )

                    #########################################################################################
                    #                                   ME to feed conversion                               #
                    #########################################################################################

                    # Metabolisable energy (ME) requirements (MJ per animal per day)
                    # Only for feed, energy for calf (MEMILKCALFINIT) is subtracted
                    # The 0.134 can be defined more precisely: (1-NEIEFFGEST)*(45/75)
                    MEREQTOTAL[i - 1, j - 1] = max(
                        0,
                        HEATIFEEDMAINT[i - 1, j - 1] - MEMILKCALFINIT[i - 1, j - 1] +
                        (
                            ENFEEDGROWTH[i, j - 1] +
                            0.134 * NEREQGEST[i - 1, j - 1] +
                            (NEMILKCOW[i - 1, j - 1] - HEATMILK[i - 1, j - 1])
                        ) * (
                            1 + (Digestfracfeed[i - 1, j - 1] / (1 - Digestfracfeed[i - 1, j - 1]))
                        ) +
                        REDMAINT[i - 1, j - 1] / (Digestfracfeed[i - 1, j - 1] - 0.10) *
                        (1 - (Digestfracfeed[i - 1, j - 1] - 0.10))
                    )

                    # Metabolisable energy (ME) requirements (MJ per day)
                    MEMET[i - 1, j - 1] = (
                        HEATIFEEDMAINT[i - 1, j - 1] - MEMILKCALFINIT[i - 1, j - 1] +
                        (
                            ENFEEDGROWTH[i, j - 1] +
                            0.134 * NEREQGEST[i - 1, j - 1] +
                            (NEMILKCOW[i - 1, j - 1] - HEATMILK[i - 1, j - 1])
                        ) * (
                            1 + (Digestfracfeed[i - 1, j - 1] / (1 - Digestfracfeed[i - 1, j - 1]))
                        )
                    )
                    # Reduction in ME (MJ per day)
                    MERED[i - 1, j - 1] = (
                        REDMAINT[i - 1, j - 1] /
                        (Digestfracfeed[i - 1, j - 1] - 0.10) *
                        (1 - (Digestfracfeed[i - 1, j - 1] - 0.10))
                    )

                    # Feed intake (kg DM per day)
                    if MEUPTAKE[i - 1, j - 1] == 0:
                        FEEDINTAKE[i - 1, j - 1] = 0
                    else:
                        FEEDINTAKE[i - 1, j - 1] = (
                            (MEREQTOTAL[i - 1, j - 1] / (Q[i - 1, j - 1] * GEFEED)) / DETOME
                        )
                    # Fraction of the digestion capacity used (-)
                    if MEUPTAKE[i - 1, j - 1] == 0:
                        FILLGIT[i - 1, j - 1] = 0
                    else:
                        FILLGIT[i - 1, j - 1] = (
                            FEEDQNTY[i - 1, j - 1] / PHFEEDINTKG[i - 1, j - 1]
                        )
                    ############## tracking / log #################
                    if DEBUG_LOOP and z in DEBUG_CASES and (DEBUG_I is None or i == DEBUG_I) and (DEBUG_J is None or j == DEBUG_J):
                        print(
                            f"[CHK4 ME] z={z} s={s} j={j} i={i} iter={loop_iter} "
                            f"MEREQTOTAL={_dbg_scalar(MEREQTOTAL[i - 1, j - 1])} "
                            f"FEEDINTAKE={_dbg_scalar(FEEDINTAKE[i - 1, j - 1])} "
                            f"FILLGIT={_dbg_scalar(FILLGIT[i - 1, j - 1])} "
                            f"PHFEEDINTKG={_dbg_scalar(PHFEEDINTKG[i - 1, j - 1])}"
                        )
                    ###############################################
                    # Rumen/ digestion capacity classes as defined by Chilibroste et al. (1997)
                    if FILLGIT[i - 1, j - 1] > 0.85:
                        PASSAGE1[i - 1, j - 1] = 1
                    else:
                        if (FILLGIT[i - 1, j - 1] <= 0.85 and
                                FILLGIT[i - 1, j - 1] > 0.65):
                            PASSAGE1[i - 1, j - 1] = 2
                        else:
                            if (FILLGIT[i - 1, j - 1] <= 0.65 and
                                    FILLGIT[i - 1, j - 1] > 0.45):
                                PASSAGE1[i - 1, j - 1] = 3
                            else:
                                PASSAGE1[i - 1, j - 1] = 4

                    # Possible difference in digestion capacity class after calculation of FILLGIT
                    PASSDIFF[i - 1, j - 1] = max( 0, PASSAGE1[i - 1, j - 1] - PASSAGE[i - 1, j - 1] )
                    #################### tracking / log ####################
                    if DEBUG_LOOP and z in DEBUG_CASES and (DEBUG_I is None or i == DEBUG_I) and (DEBUG_J is None or j == DEBUG_J):
                        print(
                            f"[CHK5 PASS] z={z} s={s} j={j} i={i} iter={loop_iter} "
                            f"PASSAGE={_dbg_scalar(PASSAGE[i - 1, j - 1])} "
                            f"PASSAGE1={_dbg_scalar(PASSAGE1[i - 1, j - 1])} "
                            f"PASSDIFF={_dbg_scalar(PASSDIFF[i - 1, j - 1])}"
                        )
                    ########################################################
                    if TIME[i - 1, j - 1] > 15:
                        PASSAGE[i - 1, j - 1] = PASSAGE[i - 1, j - 1] + PASSDIFF[i - 1, j - 1]

                    REPS[i - 1, j - 1] = REPS[i - 1, j - 1] + 1  # Times the integration loop has run

                    # Optimization statement in the integration loop (among the different sub-models)
                    PROTBAL[i - 1, j - 1] = (
                        PROTUPT[i - 1, j - 1] * (1 + NRECYCLPT[i - 1, j - 1] / 100) +
                        MILKSTARTPRHF[i - 1, j - 1] - PROTGROSS[i - 1, j - 1]
                    )
                    #PROTREDFACT[i - 1, j - 1] = (
                    #    1 - min(
                    #        1,
                    #        max(
                    #            0,
                    #            ((PROTBAL[i - 1, j - 1] * -1) /
                    #                (PROTGROSS[i - 1, j - 1] - PROTNONGM[i - 1, j - 1]))
                    #        )
                    #    )
                    #)
                    protredfact_denom = PROTGROSS[i - 1, j - 1] - PROTNONGM[i - 1, j - 1]

                    if protredfact_denom <= 0:
                        PROTREDFACT[i - 1, j - 1] = 1.0
                    else:
                        PROTREDFACT[i - 1, j - 1] = (
                            1 - min(
                                1,
                                max(
                                    0,
                                    ((PROTBAL[i - 1, j - 1] * -1) / protredfact_denom)
                                )
                            )
                        )
                    # This statement indicates that heat production from growth (HEATTOTALACT plus HIF for
                    # ENTOTALACT) cannot exceed the maximum heat release (HEATIFEEDGROWTH) by more than 0.5
                    # MJ day-1
                    DIFFEN[i - 1, j - 1] = (
                        HEATTOTALACT[i - 1, j - 1] +
                        ENTOTALACT[i - 1, j - 1] *
                        (Digestfracfeed[i - 1, j - 1] / (1 - Digestfracfeed[i - 1, j - 1])) -
                        max(0, HEATIFEEDGROWTH[i - 1, j - 1])
                    )
                    ############### tracking / log #######################
                    # This statement indicates that heat production from growth (HEATTOTALACT plus HIF for
                    # ENTOTALACT) cannot exceed the maximum heat release (HEATIFEEDGROWTH) by more than 0.5
                    # MJ day-1
                    DIFFEN[i - 1, j - 1] = (
                        HEATTOTALACT[i - 1, j - 1] +
                        ENTOTALACT[i - 1, j - 1] *
                        (Digestfracfeed[i - 1, j - 1] / (1 - Digestfracfeed[i - 1, j - 1])) -
                        max(0, HEATIFEEDGROWTH[i - 1, j - 1])
                    )

                    for _name, _val in {
                        "FEEDINTAKE": FEEDINTAKE[i - 1, j - 1],
                        "FEEDQNTY": FEEDQNTY[i - 1, j - 1],
                        "Digestfracfeed": Digestfracfeed[i - 1, j - 1],
                        "HEATIFEEDGROWTH": HEATIFEEDGROWTH[i - 1, j - 1] if i - 1 < HEATIFEEDGROWTH.shape[0] else np.nan,
                        "HEATTOTALACT": HEATTOTALACT[i - 1, j - 1] if i - 1 < HEATTOTALACT.shape[0] else np.nan,
                        "ENTOTALACT": ENTOTALACT[i - 1, j - 1] if i - 1 < ENTOTALACT.shape[0] else np.nan,
                        "DIFFEN": DIFFEN[i - 1, j - 1] if i - 1 < DIFFEN.shape[0] else np.nan,
                    }.items():
                        if isinstance(_val, (float, np.floating)) and not np.isfinite(_val):
                            raise RuntimeError(
                                f"Non-finite value in loop: {_name}={_val}, z={z}, s={s}, j={j}, i={i}, iter={loop_iter}"
                            )



                    ######################################################
                    # If heat production exceeds the maximum heat release, feed intake is reduced via REDHP.
                    # REDHP accounts for heat stress
                    if DIFFEN[i - 1, j - 1] > 0.5:
                        REDHP[i - 1, j - 1] = REDHP[i - 1, j - 1] - 0.1 * DIFFEN[i - 1, j - 1]
                    else:
                        REDHP[i - 1, j - 1] = REDHP[i - 1, j - 1]

                    # If the passage rate is not correct, the integration loop does not change passage rate
                    # and feed intake at the same time
                    if TIME[i - 1, j - 1] > 15 and PASSDIFF[i - 1, j - 1] != 0:
                        REDHP[i - 1, j - 1] = 0

                    # The loop has to run at least two times
                    if REPS[i - 1, j - 1] < 2:
                        CHECKHEAT3[i - 1, j - 1] = "FALSE"
                    else:
                        CHECKHEAT3[i - 1, j - 1] = "CORRECT"
                    # Passage rate has to be correct before the loop terminates
                    if TIME[i - 1, j - 1] > 15 and PASSDIFF[i - 1, j - 1] != 0:
                        CHECKHEAT3[i - 1, j - 1] = "FALSE"

                    # Heat production cannot exceed maximum heat release before the loop terminates
                    if DIFFEN[i - 1, j - 1] > 0.5:
                        CHECKHEAT3[i - 1, j - 1] = "FALSE"
                    ############### tracking / log ######################
                    if DEBUG_LOOP and z in DEBUG_CASES and (DEBUG_I is None or i == DEBUG_I) and (DEBUG_J is None or j == DEBUG_J):
                        print(
                            f"[LOOP STATE] z={z} s={s} j={j} i={i} iter={loop_iter} "
                            f"REPS={_dbg_scalar(REPS[i - 1, j - 1])} "
                            f"PASSAGE={_dbg_scalar(PASSAGE[i - 1, j - 1])} "
                            f"PASSAGE1={_dbg_scalar(PASSAGE1[i - 1, j - 1])} "
                            f"PASSDIFF={_dbg_scalar(PASSDIFF[i - 1, j - 1])} "
                            f"DIFFEN={_dbg_scalar(DIFFEN[i - 1, j - 1])} "
                            f"CHECKHEAT3={CHECKHEAT3[i - 1, j - 1]} "
                            f"REDHP={_dbg_scalar(REDHP[i - 1, j - 1])} "
                            f"FEEDINTAKE={_dbg_scalar(FEEDINTAKE[i - 1, j - 1])} "
                            f"FEEDQNTY={_dbg_scalar(FEEDQNTY[i - 1, j - 1])} "
                            f"HEATIFEEDGROWTH={_dbg_scalar(HEATIFEEDGROWTH[i - 1, j - 1])} "
                            f"HEATTOTALACT={_dbg_scalar(HEATTOTALACT[i - 1, j - 1])} "
                            f"ENTOTALACT={_dbg_scalar(ENTOTALACT[i - 1, j - 1])} "
                            f"Digestfracfeed={_dbg_scalar(Digestfracfeed[i - 1, j - 1])}"
                        )

                    if loop_iter >= DEBUG_MAX_ITER:
                        raise RuntimeError(
                            f"Integration loop stuck: z={z}, s={s}, j={j}, i={i}, "
                            f"REPS={REPS[i - 1, j - 1]}, PASSDIFF={PASSDIFF[i - 1, j - 1]}, "
                            f"DIFFEN={DIFFEN[i - 1, j - 1]}, CHECKHEAT3={CHECKHEAT3[i - 1, j - 1]}"
                        )

                    #####################################################
                    # If all conditions are met, the integration loop terminates
                    if CHECKHEAT3[i - 1, j - 1] == "CORRECT":
                        break

                    #########################################################################################

# End of the integration loop
########################## line:4231 ##############################
                # Fraction physically effective neutral detergent fibre (peNDF) in the diet (-)
                # Note: diets must contain sufficient peNDF to avoid unrealistic simulations where rumen
                # functioning cannot be sustained.
                PENDF[i - 1, j - 1] = (
                    FRACFEED1[i - 1, j - 1] * FEED1[i - 1, 13] * FEED1[i - 1, 14] +
                    FRACFEED2[i - 1, j - 1] * FEED2[i - 1, 13] * FEED2[i - 1, 14] +
                    FRACFEED3[i - 1, j - 1] * FEED3[i - 1, 13] * FEED3[i - 1, 14] +
                    FRACFEED4[i - 1, j - 1] * FEED4[13] * FEED4[14]
                )

                # Tissue weights are corrected for protein deficiency (PROTREDFACT)

                # Weight of lipids in bone tissue (kg per animal)
                LIPIDBONEBF[i, j - 1] = (
                    LIPIDBONEBF[i - 1, j - 1] +
                    DERBONE[i - 1, j - 1] * ENGRBONEBF[i, j - 1] /
                    ENGRBONE[i, j - 1] * LIPIDFRACBONEBF[i - 1, j - 1] *
                    PROTREDFACT[i - 1, j - 1]
                )
                # Weight of lipids in the non-carcass tissue (kg per animal)
                LIPIDNONCBF[i, j - 1] = (
                    LIPIDNONCBF[i - 1, j - 1] +
                    DERNONC[i - 1, j - 1] * ENGRNONCBF[i, j - 1] /
                    ENGRNONC[i, j - 1] * LIPIDFRACNONCBF[i - 1, j - 1] *
                    PROTREDFACT[i - 1, j - 1]
                )
                # Weight of protein in the non-carcass tissue (kg per animal)
                PROTNONCBF[i, j - 1] = (
                    PROTNONCBF[i - 1, j - 1] +
                    DERNONC[i - 1, j - 1] * ENGRNONCBF[i, j - 1] /
                    ENGRNONC[i, j - 1] * PROTFRACNONCBF[i - 1, j - 1] *
                    PROTREDFACT[i - 1, j - 1]
                )

                # Bone tissue (kg per animal)
                BONETISBF[i, j - 1] = (
                    BONETISBF[i - 1, j - 1] +
                    DERBONE[i - 1, j - 1] * ENGRBONEBF[i, j - 1] /
                    ENGRBONE[i, j - 1] * PROTREDFACT[i - 1, j - 1]
                )
                # Muscle tissue (kg per animal)
                MUSCLETISBF[i, j - 1] = (
                    MUSCLETISBF[i - 1, j - 1] +
                    DERMUSCLE[i - 1, j - 1] * ENGRMUSCLEBF[i, j - 1] /
                    ENGRMUSCLE[i, j - 1] * PROTREDFACT[i - 1, j - 1]
                )
                # Intramuscular fat tissue (kg per animal)
                INTRAMFTISBF[i, j - 1] = (
                    INTRAMFTISBF[i - 1, j - 1] +
                    DERINTRAMF[i - 1, j - 1] * ENGRIMFBF[i, j - 1] /
                    ENGRIMF[i, j - 1] * PROTREDFACT[i - 1, j - 1]
                )
                # Subcutaneous and intermuscular fat tissue (kg per animal)
                MISCFATTISBF[i, j - 1] = (
                    MISCFATTISBF[i - 1, j - 1] +
                    DERMISCFAT[i - 1, j - 1] * ENGRFATBF[i, j - 1] /
                    ENGRFAT[i, j - 1] * PROTREDFACT[i - 1, j - 1]
                )
                # Non-carcass tissue (kg per animal)
                NONCARCTISBF[i, j - 1] = (
                    NONCARCTISBF[i - 1, j - 1] +
                    DERNONC[i - 1, j - 1] * ENGRNONCBF[i, j - 1] /
                    ENGRNONC[i, j - 1] * PROTREDFACT[i - 1, j - 1]
                )

                # Gross energy content of the non-carcass tissue (MJ kg-1)
                ENCONTENTNONCBF[i - 1, j - 1] = (
                    (LIPIDNONCBF[i - 1, j - 1] * GELIPID +
                        PROTNONCBF[i - 1, j - 1] * GEPROT) /
                    NONCARCTISBF[i - 1, j - 1]
                )

                # Muscle tissue is dissimilated when protein supply is below maintenance
                # Protein dissimilation: efficiency of 90% assumed
                # Reduction in muscle tissue (kg day-1)
                REDTISPROT[i - 1, j - 1] = min(
                    0,
                    PROTGROSS[i - 1, j - 1] + PROTBAL[i - 1, j - 1]
                ) / (PROTFRACMUSCLE * DISSEFF * 1000)
                MUSCLETISBF[i, j - 1] = (
                    MUSCLETISBF[i, j - 1] + REDTISPROT[i - 1, j - 1]
                )

                # REDTIS1 Heat stress: reduction in feed intake heat release in under sub-maintenance
                # intake
                # Weight loss due to heat stress, in (kg fat per day), 29.624 MJ kg-1 is the
                # energy content of fat tissue
                # inefficiency fat dissimilation = 10%; i.e. 90% efficiency
                REDTIS[i - 1, j - 1] = (
                    REDMAINT[i - 1, j - 1] /
                    (Digestfracfeed[i - 1, j - 1] - (1 - DISSEFF)) *
                    (1 - (Digestfracfeed[i - 1, j - 1] - (1 - DISSEFF))) /
                    GEFATTIS
                )
                if np.isnan(REDTIS[i - 1, j - 1]):
                    REDTIS[i - 1, j - 1] = 0

                # Fat tissue (cumulative energy) dissimulated due to heat stress (MJ)
                HEATBURNCUMUL[i - 1, j - 1] = np.sum(REDTIS[0:i, j - 1]) * GEFATTIS

                # To avoid heat stress, subcutaneous and intermuscular fat are dissimilated (kg day-1)
                # (and feed intake is reduced)
                MISCFATTISBF[i, j - 1] = (
                    MISCFATTISBF[i, j - 1] + REDTIS[i - 1, j - 1] * 0.9
                )
                # To avoid heat stress, non carcass tissue is dissimilated (kg day-1)
                NONCARCTISBF[i, j - 1] = (
                    NONCARCTISBF[i, j - 1] +
                    REDTIS[i - 1, j - 1] * 0.1 *
                    (GEFATTIS / ENCONTENTNONCBF[i - 1, j - 1]) / DISSEFF
                )

                # Check to ensure feed intake is not negative
                if ((NEMAINT[i - 1, j - 1] + NEPHYSACT[i - 1, j - 1] + NERESP[i - 1, j - 1]) *
                    1 / DISSEFF < -1 * REDTIS[i - 1, j - 1]):
                    CHECK[i - 1, j - 1] = "wrong"
                else:
                    CHECK[i - 1, j - 1] = "good"
########################## line:4307 ##############################
                #########################################################################################

                # REDTIS2 Negative growth, fat tissue is dissimilated (kg day-1)
                # This equation is similar to Eq. 47 of the Supplementary Information
                if (REDTIS[i - 1, j - 1] == 0 and
                        ENFEEDGROWTH[i, j - 1] < 0):
                    REDTIS2[i - 1, j - 1] = (
                        (-ENFEEDGROWTH[i, j - 1] / GEFATTIS) / DISSEFF
                    )
                else:
                    REDTIS2[i - 1, j - 1] = 0

                # To correct for negative growth, subcutaneous and intermuscular fat tissue are
                # dissimilated (kg day-1)
                MISCFATTISBF[i, j - 1] = (
                    MISCFATTISBF[i, j - 1] - REDTIS2[i - 1, j - 1] * 0.9
                )
                NONCARCTISBF[i, j - 1] = (
                    NONCARCTISBF[i, j - 1] -
                    REDTIS2[i - 1, j - 1] * 0.1 *
                    (GEFATTIS / ENCONTENTNONCBF[i - 1, j - 1]) / DISSEFF
                )

                # REDTIS3 Cold stress
                # Cumulative energy required to maintain body temperature (MJ)
                FATBURNCUMUL[i, j - 1] = (
                    FATBURNCUMUL[i - 1, j - 1] +
                    HEATIFEEDGROWTHC[i - 1, j - 1]
                )

                # Total body weight (TBW) based on the genotype, climate (heat stress; cold stress), feed
                # quality, and available feed quantity (kg live weight)
                TBWBF[i, j - 1] = (
                    BONETISBF[i, j - 1] + MUSCLETISBF[i, j - 1] +
                    INTRAMFTISBF[i, j - 1] + MISCFATTISBF[i, j - 1] +
                    NONCARCTISBF[i, j - 1]
                ) / (1 - RUMENFRAC)
                # Metabolic body weight (TBW) based on the genotype, climate (heat stress; cold stress),
                # feed quality, and available feed quantity (kg live weight)
                EBWBFMET[i, j - 1] = (
                    (TBWBF[i, j - 1] * (1 - RUMENFRAC)) ** 0.75
                )

                # Protein accretion in body tissues and milk (g day-1)
                PROTACCR[i - 1, j - 1] = (
                    PROTGESTG[i - 1, j - 1] * 0.5 +
                    PROTMILK[i - 1, j - 1] +
                    PROTTOTALACT[i - 1, j - 1]
                )
                # Fraction metabolisable energy from feed used for maintenance (-)
                #MAINTFRAC[i - 1, j - 1] = (
                #    HEATIFEEDMAINT[i - 1, j - 1] -
                #    HEATIFEEDGROWTHC[i - 1, j - 1] +
                #    MEMILKCALFINIT[i - 1, j - 1]
                #) / MEREQTOTAL[i - 1, j - 1]
                # Fraction metabolisable energy from feed used for maintenance (-)
                if MEREQTOTAL[i - 1, j - 1] <= 0:
                    MAINTFRAC[i - 1, j - 1] = 0.0
                else:
                    MAINTFRAC[i - 1, j - 1] = (
                        HEATIFEEDMAINT[i - 1, j - 1] -
                        HEATIFEEDGROWTHC[i - 1, j - 1] +
                        MEMILKCALFINIT[i - 1, j - 1]
                    ) / MEREQTOTAL[i - 1, j - 1]
                #########################################################################################
                #         Culling and slaughtering cattle         #
                ###################################################

                # Cattle slaughtered (cattle from reproductive herd)
                FATFRACCARC[i - 1, j - 1] = (
                    (MISCFATTISBF[i - 1, j - 1] + INTRAMFTISBF[i - 1, j - 1]) /
                    (MISCFATTISBF[i - 1, j - 1] + INTRAMFTISBF[i - 1, j - 1] +
                        MUSCLETISBF[i - 1, j - 1] + BONETISBF[i - 1, j - 1])
                )

                # Maximum number of calves per animal
                CALVESPERANIMAL = REPRODUCTIVE * MAXCALFNR

                # Beef production (kg per animal)
                # (Beef is deboned carcass)
                # Option 1: fat content reaches a specific level for cows, and CALFLIVENR should be met
                # for cows. This equation contains Eq. 38 of the Supplementary Information
                if (TBWBF[i, j - 1] > SWMALES and
                        SEX[j - 1] == 0):
                    BEEFPRODACT[i - 1, j - 1] = (
                        MUSCLETISBF[i - 1, j - 1] +
                        INTRAMFTISBF[i - 1, j - 1] +
                        MISCFATTISBF[i - 1, j - 1]
                    )
                else:
                    if (TBWBF[i, j - 1] > SWFEMALES and
                            SEX[j - 1] == 1 and REPRODUCTIVE[j - 1] == 0):
                        BEEFPRODACT[i - 1, j - 1] = (
                            MUSCLETISBF[i - 1, j - 1] +
                            INTRAMFTISBF[i - 1, j - 1] +
                            MISCFATTISBF[i - 1, j - 1]
                        )
                    else:
                        if (SEX[j - 1] == 1 and
                                FATFRACCARC[i - 1, j - 1] > MAXFATCARC and
                                CALFWEANNR[i - 1, j - 1] == CALVESPERANIMAL[j - 1] and
                                TIME[i - 1, j - 1] > 800):
                            BEEFPRODACT[i - 1, j - 1] = (
                                MUSCLETISBF[i - 1, j - 1] +
                                INTRAMFTISBF[i - 1, j - 1] +
                                MISCFATTISBF[i - 1, j - 1]
                            )
                        else:
                            BEEFPRODACT[i - 1, j - 1] = 0

                # Option 2: maximum number of years in (re)productive herd
                if (TIME[i - 1, j - 1] / 365 > MAXLIFETIME and
                        BEEFPRODACT[i - 1, j - 1] == 0):
                    BEEFPRODACT[i - 1, j - 1] = (
                        MUSCLETISBF[i - 1, j - 1] +
                        INTRAMFTISBF[i - 1, j - 1] +
                        MISCFATTISBF[i - 1, j - 1]
                    )
                if (REPRODUCTIVE[j - 1] == 1 and
                        CALFWEANNR[i - 1, j - 1] == CALVESPERANIMAL[j - 1] and
                        TIME[i - 1, j - 1] / 365 > MAXLIFETIME):
                    BEEFPRODACT[i - 1, j - 1] = (
                        MUSCLETISBF[i - 1, j - 1] +
                        INTRAMFTISBF[i - 1, j - 1] +
                        MISCFATTISBF[i - 1, j - 1]
                    )

                # Option 3: cattle death
                # If fat reserves are fully depleted, there is no beef production
                if MISCFATTISBF[i - 1, j - 1] < 0:
                    BEEFPRODACT[i - 1, j - 1] = -1
                if BEEFPRODACT[i - 1, j - 1] != 0:
                    SLAUGHTERDAYACT[i - 1, j - 1] = TIME[i - 1, j - 1]
                else:
                    SLAUGHTERDAYACT[i - 1, j - 1] = 9999

                #########################################################################################

                # Live weight production (kg total body weight per animal)
                # Option 1: fat content reaches a specific level for cows, and CALFLIVENR should be met
                # for cows
                if (TBWBF[i, j - 1] > SWMALES and
                        SEX[j - 1] == 0):
                    LWPRODACT[i - 1, j - 1] = TBWBF[i, j - 1]
                else:
                    if (TBWBF[i, j - 1] > SWFEMALES and
                            SEX[j - 1] == 1 and REPRODUCTIVE[j - 1] == 0):
                        LWPRODACT[i - 1, j - 1] = TBWBF[i, j - 1]
                    else:
                        if (SEX[j - 1] == 1 and
                                FATFRACCARC[i - 1, j - 1] > MAXFATCARC and
                                CALFWEANNR[i - 1, j - 1] == CALVESPERANIMAL[j - 1] and
                                TIME[i - 1, j - 1] > 800):
                            LWPRODACT[i - 1, j - 1] = TBWBF[i, j - 1]
                        else:
                            LWPRODACT[i - 1, j - 1] = 0

                # Option 2: maximum number of years in (re)productive herd
                if (TIME[i - 1, j - 1] / 365 > MAXLIFETIME and
                        BEEFPRODACT[i - 1, j - 1] == 0):
                    LWPRODACT[i - 1, j - 1] = TBWBF[i, j - 1]
                if (REPRODUCTIVE[j - 1] == 1 and
                        CALFWEANNR[i - 1, j - 1] == CALVESPERANIMAL[j - 1] and
                        TIME[i - 1, j - 1] / 365 > MAXLIFETIME):
                    LWPRODACT[i - 1, j - 1] = TBWBF[i, j - 1]

                # Option 3: cattle death
                # If fat reserves are fully depleted, there is no live weight production
                if MISCFATTISBF[i - 1, j - 1] < 0:
                    LWPRODACT[i - 1, j - 1] = -1
########################## line:4433 ##############################
                #########################################################################################

                # Carcass production (kg)
                # Option 1: fat content reaches a specific level for cows, and CALFLIVENR should be met
                # for cows
                if (TBWBF[i, j - 1] > SWMALES and
                        SEX[j - 1] == 0):
                    CARCPRODACT[i - 1, j - 1] = (
                        MUSCLETISBF[i - 1, j - 1] +
                        INTRAMFTISBF[i - 1, j - 1] +
                        MISCFATTISBF[i - 1, j - 1] +
                        BONETISBF[i - 1, j - 1]
                    )
                else:
                    if (TBWBF[i, j - 1] > SWFEMALES and
                            SEX[j - 1] == 1 and REPRODUCTIVE[j - 1] == 0):
                        CARCPRODACT[i - 1, j - 1] = (
                            MUSCLETISBF[i - 1, j - 1] +
                            INTRAMFTISBF[i - 1, j - 1] +
                            MISCFATTISBF[i - 1, j - 1] +
                            BONETISBF[i - 1, j - 1]
                        )
                    elif (SEX[j - 1] == 1 and
                            FATFRACCARC[i - 1, j - 1] > MAXFATCARC and
                            CALFWEANNR[i - 1, j - 1] == CALVESPERANIMAL[j - 1] and
                            TIME[i - 1, j - 1] > 800):
                        CARCPRODACT[i - 1, j - 1] = (
                            MUSCLETISBF[i - 1, j - 1] +
                            INTRAMFTISBF[i - 1, j - 1] +
                            MISCFATTISBF[i - 1, j - 1] +
                            BONETISBF[i - 1, j - 1]
                        )
                    else:
                        CARCPRODACT[i - 1, j - 1] = 0

                # Option 2: maximum number of years in (re)productive herd
                if (TIME[i - 1, j - 1] / 365 > MAXLIFETIME and
                        BEEFPRODACT[i - 1, j - 1] == 0):
                    CARCPRODACT[i - 1, j - 1] = (
                        MUSCLETISBF[i - 1, j - 1] +
                        INTRAMFTISBF[i - 1, j - 1] +
                        MISCFATTISBF[i - 1, j - 1] +
                        BONETISBF[i - 1, j - 1]
                    )
                if (REPRODUCTIVE[j - 1] == 1 and
                        CALFWEANNR[i - 1, j - 1] == CALVESPERANIMAL[j - 1] and
                        TIME[i - 1, j - 1] / 365 > MAXLIFETIME):
                    CARCPRODACT[i - 1, j - 1] = (
                        MUSCLETISBF[i - 1, j - 1] +
                        INTRAMFTISBF[i - 1, j - 1] +
                        MISCFATTISBF[i - 1, j - 1] +
                        BONETISBF[i - 1, j - 1]
                    )

                # Option 3: cattle death
                # If fat reserves are fully depleted, there is no carcass production
                if MISCFATTISBF[i - 1, j - 1] < 0:
                    BEEFPRODACT[i - 1, j - 1] = -1

                #########################################################################################

                # Day the animal is slaughtered (days)
                # Is initially 9999, but is replaced by the correct number of days at slaughter
                ENDDAY[j - 1] = np.min(SLAUGHTERDAYACT[0:i, j - 1])

                # Beef production at slaughter (kg per animal)
                if ENDDAY[j - 1] < 9999:
                    end_idx = int(ENDDAY[j - 1]) - 1
                    BEEFPROD[j - 1] = BEEFPRODACT[end_idx, j - 1]
                # Beef production (kg beef per head per year)
                if ENDDAY[j - 1] < 9999:
                    end_idx = int(ENDDAY[j - 1]) - 1
                    BEEFPRODYEAR[j - 1] = (
                        BEEFPRODACT[end_idx, j - 1] /
                        (TIME[end_idx, j - 1] / 365)
                    )
                # Live weight production at slaughter (kg per animal)
                if ENDDAY[j - 1] < 9999:
                    end_idx = int(ENDDAY[j - 1]) - 1
                    LWPROD[j - 1] = LWPRODACT[end_idx, j - 1]
                # Live weight production (kg live weight per head per year)
                if ENDDAY[j - 1] < 9999:
                    end_idx = int(ENDDAY[j - 1]) - 1
                    LWPRODYEAR[j - 1] = (
                        LWPRODACT[end_idx, j - 1] /
                        (TIME[end_idx, j - 1] / 365)
                    )

                # If the cow/bull is slaughtered, the weight is set to 0 via the vector ALIVE.
                # ALIVE produces a vector that indicates whether the cow/bull is alive or slaughtered
                # (1= true, 0=not true)
                if ENDDAY[j - 1] == 9999:
                    ALIVE[i, j - 1] = 1
                else:
                    ALIVE[i, j - 1] = 0

                # The code below keeps track of the parity of the reproductive cow
                # Code indicates whether a cow is in or has had the nth parity (1= true, 0=not true).
                if CALFLIVENR[i - 1, j - 1] >= 1:
                    PARITY1[i - 1, j - 1] = 1
                else:
                    PARITY1[i - 1, j - 1] = 0
                if CALFLIVENR[i - 1, j - 1] >= 2:
                    PARITY2[i - 1, j - 1] = 1
                else:
                    PARITY2[i - 1, j - 1] = 0
                if CALFLIVENR[i - 1, j - 1] >= 3:
                    PARITY3[i - 1, j - 1] = 1
                else:
                    PARITY3[i - 1, j - 1] = 0
                if CALFLIVENR[i - 1, j - 1] >= 4:
                    PARITY4[i - 1, j - 1] = 1
                else:
                    PARITY4[i - 1, j - 1] = 0
                if CALFLIVENR[i - 1, j - 1] >= 5:
                    PARITY5[i - 1, j - 1] = 1
                else:
                    PARITY5[i - 1, j - 1] = 0
                if CALFLIVENR[i - 1, j - 1] >= 6:
                    PARITY6[i - 1, j - 1] = 1
                else:
                    PARITY6[i - 1, j - 1] = 0
                if CALFLIVENR[i - 1, j - 1] >= 7:
                    PARITY7[i - 1, j - 1] = 1
                else:
                    PARITY7[i - 1, j - 1] = 0
                if CALFLIVENR[i - 1, j - 1] >= 8:
                    PARITY8[i - 1, j - 1] = 1
                else:
                    PARITY8[i - 1, j - 1] = 0
                if CALFLIVENR[i - 1, j - 1] >= 9:
                    PARITY9[i - 1, j - 1] = 1
                else:
                    PARITY9[i - 1, j - 1] = 0

                # Birth days of calves 1-9 (days after birth of the reproductive cow)
                if CALFLIVENR[i - 1, j - 1] == 1:
                    BIRTHDAYCALF1 = TIME[i - 1, j - 1] - np.sum(PARITY1[0:i, j - 1])
                else:
                    BIRTHDAYCALF1 = BIRTHDAYCALF1  # Birth day calf 1
                if CALFLIVENR[i - 1, j - 1] == 2:
                    BIRTHDAYCALF2 = TIME[i - 1, j - 1] - np.sum(PARITY2[0:i, j - 1])
                else:
                    BIRTHDAYCALF2 = BIRTHDAYCALF2  # Birth day calf 2
                if CALFLIVENR[i - 1, j - 1] == 3:
                    BIRTHDAYCALF3 = TIME[i - 1, j - 1] - np.sum(PARITY3[0:i, j - 1])
                else:
                    BIRTHDAYCALF3 = BIRTHDAYCALF3  # Birth day calf 3
                if CALFLIVENR[i - 1, j - 1] == 4:
                    BIRTHDAYCALF4 = TIME[i - 1, j - 1] - np.sum(PARITY4[0:i, j - 1])
                else:
                    BIRTHDAYCALF4 = BIRTHDAYCALF4  # Birth day calf 4
                if CALFLIVENR[i - 1, j - 1] == 5:
                    BIRTHDAYCALF5 = TIME[i - 1, j - 1] - np.sum(PARITY5[0:i, j - 1])
                else:
                    BIRTHDAYCALF5 = BIRTHDAYCALF5  # Birth day calf 5
                if CALFLIVENR[i - 1, j - 1] == 6:
                    BIRTHDAYCALF6 = TIME[i - 1, j - 1] - np.sum(PARITY6[0:i, j - 1])
                else:
                    BIRTHDAYCALF6 = BIRTHDAYCALF6  # Birth day calf 6
                if CALFLIVENR[i - 1, j - 1] == 7:
                    BIRTHDAYCALF7 = TIME[i - 1, j - 1] - np.sum(PARITY7[0:i, j - 1])
                else:
                    BIRTHDAYCALF7 = BIRTHDAYCALF7  # Birth day calf 7
                if CALFLIVENR[i - 1, j - 1] == 8:
                    BIRTHDAYCALF8 = TIME[i - 1, j - 1] - np.sum(PARITY8[0:i, j - 1])
                else:
                    BIRTHDAYCALF8 = BIRTHDAYCALF8  # Birth day calf 8
                if CALFLIVENR[i - 1, j - 1] == 9:
                    BIRTHDAYCALF9 = TIME[i - 1, j - 1] - np.sum(PARITY9[0:i, j - 1])
                else:
                    BIRTHDAYCALF9 = BIRTHDAYCALF9  # Birth day calf 9
########################## line:4577 ##############################
                #########################################################################################

                # Average fraction of the diet digested (-)
                AVGDIGFRAC[i - 1, j - 1] = (
                    FRACFEED1[i - 1, j - 1] * FEED1[i - 1, 0] +
                    FRACFEED2[i - 1, j - 1] * FEED2[i - 1, 0] +
                    FRACFEED3[i - 1, j - 1] * FEED3[i - 1, 0] +
                    FRACFEED4[i - 1, j - 1] * FEED4[0]
                )

                # Cumulative feed intake (kg DM per animal per day)
                CUMULFEED1[i - 1, j - 1] = np.sum(FEED1QNTY[0:i, j - 1])  # cumulative amount of  feed type 1
                CUMULFEED2[i - 1, j - 1] = np.sum(FEED2QNTY[0:i, j - 1])  # cumulative amount of  feed type 2
                CUMULFEED3[i - 1, j - 1] = np.sum(FEED3QNTY[0:i, j - 1])  # cumulative amount of  feed type 3
                CUMULFEED4[i - 1, j - 1] = np.sum(FEED4QNTY[0:i, j - 1])  # cumulative amount of  feed type 4

                # Cumulative kg feed intake (whole diet) (kg DM per animal per day)
                CUMULFEED[i - 1, j - 1] = (
                    CUMULFEED1[i - 1, j - 1] + CUMULFEED2[i - 1, j - 1] +
                    CUMULFEED3[i - 1, j - 1] + CUMULFEED4[i - 1, j - 1]
                )

                # Feed conversion ratio (FCR; kg feed per kg live weight)
                #FCR[i - 1, j - 1] = (
                #    CUMULFEED[i - 1, j - 1] /
                #    (TBWBF[i - 1, j - 1] - TBWBF[0, j - 1])
                #)
                # Feed conversion ratio (FCR; kg feed per kg live weight)
                fcr_denom = TBWBF[i - 1, j - 1] - TBWBF[0, j - 1]

                if fcr_denom <= 0:
                    FCR[i - 1, j - 1] = np.nan
                else:
                    FCR[i - 1, j - 1] = CUMULFEED[i - 1, j - 1] / fcr_denom
                # Feed conversion ratio (kg feed per kg beef)
                FCRBEEF[i - 1, j - 1] = (
                    CUMULFEED[i - 1, j - 1] /
                    (
                        (MUSCLETISBF[i, j - 1] + INTRAMFTISBF[i, j - 1] + MISCFATTISBF[i, j - 1]) -
                        (MUSCLETISBF[0, j - 1] + INTRAMFTISBF[0, j - 1] + MISCFATTISBF[0, j - 1])
                    )
                )
                # Feed conversion ratio (kg feed per kg beef at slaughter)
                if ENDDAY[j - 1] < 9999:
                    end_idx = int(ENDDAY[j - 1]) - 1
                    FCRBEEFENDDAY[j - 1] = CUMULFEED[end_idx, j - 1] / BEEFPROD[j - 1]
                # Percentage feed intake relative to the total body weight (%)
                PERCFI[i - 1, j - 1] = FEEDQNTY[i - 1, j - 1] / TBWBF[i - 1, j - 1] * 100

                # Simulating one animal is sufficient for simulations at the animal level.
                # The concept of the herd unit is used to simulate beef cattle at the herd level,
                # where multiple animals are simulated (one reproductive cow and her offspring, minus a
                # replacement heifer)

                # If the animal is slaughtered, the time loop for the animal is terminated
                if SLAUGHTERDAYACT[i - 1, j - 1] < 9000:
                    breakFlagtime = True
                    break
                #########################################################################################

# } for the time loop
########################## line:4623 ##############################
            #print("LOG 5324: after while")
            # Shift the weather files of the calves, based on their birthday
            # orginal
            '''
            WEATHERCALF1 = WEATHERORIG.iloc[int(BIRTHDAYCALF1) - 1:int(imax[j - 1] + BIRTHDAYCALF1 + 2), :].copy()
            WEATHERCALF2 = WEATHERORIG.iloc[int(BIRTHDAYCALF2) - 1:int(imax[j - 1] + BIRTHDAYCALF2 + 2), :].copy()
            WEATHERCALF3 = WEATHERORIG.iloc[int(BIRTHDAYCALF3) - 1:int(imax[j - 1] + BIRTHDAYCALF3 + 2), :].copy()
            WEATHERCALF4 = WEATHERORIG.iloc[int(BIRTHDAYCALF4) - 1:int(imax[j - 1] + BIRTHDAYCALF4 + 2), :].copy()
            WEATHERCALF5 = WEATHERORIG.iloc[int(BIRTHDAYCALF5) - 1:int(imax[j - 1] + BIRTHDAYCALF5 + 2), :].copy()
            WEATHERCALF6 = WEATHERORIG.iloc[int(BIRTHDAYCALF6) - 1:int(imax[j - 1] + BIRTHDAYCALF6 + 2), :].copy()
            WEATHERCALF7 = WEATHERORIG.iloc[int(BIRTHDAYCALF7) - 1:int(imax[j - 1] + BIRTHDAYCALF7 + 2), :].copy()
            WEATHERCALF8 = WEATHERORIG.iloc[int(BIRTHDAYCALF8) - 1:int(imax[j - 1] + BIRTHDAYCALF8 + 2), :].copy()
            WEATHERCALF9 = WEATHERORIG.iloc[int(BIRTHDAYCALF9) - 1:int(imax[j - 1] + BIRTHDAYCALF9 + 2), :].copy()
            '''
            WEATHERCALF1 = WEATHERORIG.iloc[int(BIRTHDAYCALF1) - 1:int(imax[j - 1] + BIRTHDAYCALF1 + 2), :].copy().reset_index(drop=True)
            WEATHERCALF2 = WEATHERORIG.iloc[int(BIRTHDAYCALF2) - 1:int(imax[j - 1] + BIRTHDAYCALF2 + 2), :].copy().reset_index(drop=True)
            WEATHERCALF3 = WEATHERORIG.iloc[int(BIRTHDAYCALF3) - 1:int(imax[j - 1] + BIRTHDAYCALF3 + 2), :].copy().reset_index(drop=True)
            WEATHERCALF4 = WEATHERORIG.iloc[int(BIRTHDAYCALF4) - 1:int(imax[j - 1] + BIRTHDAYCALF4 + 2), :].copy().reset_index(drop=True)
            WEATHERCALF5 = WEATHERORIG.iloc[int(BIRTHDAYCALF5) - 1:int(imax[j - 1] + BIRTHDAYCALF5 + 2), :].copy().reset_index(drop=True)
            WEATHERCALF6 = WEATHERORIG.iloc[int(BIRTHDAYCALF6) - 1:int(imax[j - 1] + BIRTHDAYCALF6 + 2), :].copy().reset_index(drop=True)
            WEATHERCALF7 = WEATHERORIG.iloc[int(BIRTHDAYCALF7) - 1:int(imax[j - 1] + BIRTHDAYCALF7 + 2), :].copy().reset_index(drop=True)
            WEATHERCALF8 = WEATHERORIG.iloc[int(BIRTHDAYCALF8) - 1:int(imax[j - 1] + BIRTHDAYCALF8 + 2), :].copy().reset_index(drop=True)
            WEATHERCALF9 = WEATHERORIG.iloc[int(BIRTHDAYCALF9) - 1:int(imax[j - 1] + BIRTHDAYCALF9 + 2), :].copy().reset_index(drop=True)
            # Selects the right weather file for each calf
            if ORDER[j - 1] == 0:
                WEATHER = WEATHERCALF1
            if ORDER[j - 1] == 1:
                WEATHER = WEATHERCALF2
            if ORDER[j - 1] == 2:
                WEATHER = WEATHERCALF3
            if ORDER[j - 1] == 3:
                WEATHER = WEATHERCALF4
            if ORDER[j - 1] == 4:
                WEATHER = WEATHERCALF5
            if ORDER[j - 1] == 5:
                WEATHER = WEATHERCALF6
            if ORDER[j - 1] == 6:
                WEATHER = WEATHERCALF7
            if ORDER[j - 1] == 7:
                WEATHER = WEATHERCALF8
            if ORDER[j - 1] == 8:
                WEATHER = WEATHERCALF9

            ###########################################################################################

            # Data on beef production and feed intake
            # orginal
            '''
            end_idx = int(ENDDAY[j - 1]) - 1

            # Beef production (kg per head)
            BEEFPRODHERD[j - 1] = BEEFPRODACT[end_idx, j - 1]
            # Live weight production (kg per head)
            LWPRODHERD[j - 1] = LWPRODACT[end_idx, j - 1]
            # Feed conversion ratio (kg DM feed per kg beef)
            FCRHERDBEEF[j - 1] = FCRBEEF[end_idx, j - 1]
            # Cumulative feed intake whole life span (kg DM)
            CUMULFEEDHERD[j - 1] = CUMULFEED[end_idx, j - 1]
            # Cumulative feed type 1 intake whole life span (kg DM)
            CUMULFEED1HERD[j - 1] = CUMULFEED1[end_idx, j - 1]
            # Cumulative feed type 2 intake whole life span (kg DM)
            CUMULFEED2HERD[j - 1] = CUMULFEED2[end_idx, j - 1]
            # Cumulative feed type 3 intake whole life span (kg DM)
            CUMULFEED3HERD[j - 1] = CUMULFEED3[end_idx, j - 1]
            # Cumulative feed type 4 intake whole life span (kg DM)
            CUMULFEED4HERD[j - 1] = CUMULFEED4[end_idx, j - 1]

            # Life span of the animal (years)
            ANIMALYEARS[j - 1] = ENDDAY[j - 1] / 365
            # Average weight (kg total body weight)
            AVANWEIGHT[j - 1] = np.mean(TBWBF[0:int(ENDDAY[j - 1]), j - 1])
            # Average metabolic weight (kg empty body weight^0.75)
            AVANMETWEIGHT[j - 1] = np.mean(EBWBFMET[0:int(ENDDAY[j - 1]), j - 1])
            '''
            if ENDDAY[j - 1] < 9999:
                end_idx = int(ENDDAY[j - 1]) - 1

                # Beef production (kg per head)
                BEEFPRODHERD[j - 1] = BEEFPRODACT[end_idx, j - 1]
                # Live weight production (kg per head)
                LWPRODHERD[j - 1] = LWPRODACT[end_idx, j - 1]
                # Feed conversion ratio (kg DM feed per kg beef)
                FCRHERDBEEF[j - 1] = FCRBEEF[end_idx, j - 1]
                # Cumulative feed intake whole life span (kg DM)
                CUMULFEEDHERD[j - 1] = CUMULFEED[end_idx, j - 1]
                # Cumulative feed type 1 intake whole life span (kg DM)
                CUMULFEED1HERD[j - 1] = CUMULFEED1[end_idx, j - 1]
                # Cumulative feed type 2 intake whole life span (kg DM)
                CUMULFEED2HERD[j - 1] = CUMULFEED2[end_idx, j - 1]
                # Cumulative feed type 3 intake whole life span (kg DM)
                CUMULFEED3HERD[j - 1] = CUMULFEED3[end_idx, j - 1]
                # Cumulative feed type 4 intake whole life span (kg DM)
                CUMULFEED4HERD[j - 1] = CUMULFEED4[end_idx, j - 1]

                # Life span of the animal (years)
                ANIMALYEARS[j - 1] = ENDDAY[j - 1] / 365
                # Average weight (kg total body weight)
                AVANWEIGHT[j - 1] = np.mean(TBWBF[0:int(ENDDAY[j - 1]), j - 1])
                # Average metabolic weight (kg empty body weight^0.75)
                AVANMETWEIGHT[j - 1] = np.mean(EBWBFMET[0:int(ENDDAY[j - 1]), j - 1])

            else:
                BEEFPRODHERD[j - 1] = 0.0
                LWPRODHERD[j - 1] = 0.0
                FCRHERDBEEF[j - 1] = np.nan
                CUMULFEEDHERD[j - 1] = CUMULFEED[i - 1, j - 1] if i > 0 else 0.0
                CUMULFEED1HERD[j - 1] = CUMULFEED1[i - 1, j - 1] if i > 0 else 0.0
                CUMULFEED2HERD[j - 1] = CUMULFEED2[i - 1, j - 1] if i > 0 else 0.0
                CUMULFEED3HERD[j - 1] = CUMULFEED3[i - 1, j - 1] if i > 0 else 0.0
                CUMULFEED4HERD[j - 1] = CUMULFEED4[i - 1, j - 1] if i > 0 else 0.0

                # animal not slaughtered yet: use simulated life so far
                ANIMALYEARS[j - 1] = TIME[i - 1, j - 1] / 365
                AVANWEIGHT[j - 1] = np.mean(TBWBF[0:i, j - 1]) if i > 0 else np.nan
                AVANMETWEIGHT[j - 1] = np.mean(EBWBFMET[0:i, j - 1]) if i > 0 else np.nan

            
            # Vector with birthdays
            # The reproductive cow in a herd unit is born at day 0
            BIRTHDAY = np.array([
                0,
                BIRTHDAYCALF1,
                BIRTHDAYCALF2,
                BIRTHDAYCALF3,
                BIRTHDAYCALF4,
                BIRTHDAYCALF5,
                BIRTHDAYCALF6,
                BIRTHDAYCALF7,
                BIRTHDAYCALF8
            ], dtype=float)
            # Weaning time for calves (days after birth of the reproductive cow)
            WNDAY = BIRTHDAY + WEANINGTIME
            WNDAY[0] = ENDDAY[0]

            # The vector ANIMALINFO lists key information on animal performance
            ANIMALINFO = np.array([
                REPRODUCTIVE[j - 1],
                REPLACEMENT[j - 1],
                PRODUCTIVE[j - 1],
                SEX[j - 1],
                BEEFPRODHERD[j - 1],
                CUMULFEEDHERD[j - 1],
                FCRHERDBEEF[j - 1],
                ANIMALYEARS[j - 1],
                AVANWEIGHT[j - 1],
                AVANMETWEIGHT[j - 1],
                ENDDAY[j - 1],
                BIRTHDAY[j - 1],
                WNDAY[j - 1],
                LWPRODHERD[j - 1],
                CUMULFEED1HERD[j - 1],
                CUMULFEED2HERD[j - 1],
                CUMULFEED3HERD[j - 1],
                CUMULFEED4HERD[j - 1]
            ], dtype=float)

            # Information for individual animals is added to information for other animals in the
            # herd unit
            if HERDINFO is None or (isinstance(HERDINFO, list) and len(HERDINFO) == 0):
                HERDINFO = ANIMALINFO.reshape(1, -1)
            else:
                HERDINFO = np.vstack([HERDINFO, ANIMALINFO])

            # Runs are terminated before when the maximum number of calves per cow is reached.
            # This saves processing time.
            if j == MAXCALFNR + 1:
                breakFlaganim = True
                break

    # } for the animal loop
########################## line:4743 ##############################
    ###########################################################################################
    #         Upscaling to the herd level         #
    ###############################################

    HERDINFO1 = np.vstack([
        HERDINFO,
        np.zeros((0, HERDINFO.shape[1]))
    ])

    # Culling/upscaling is only meaningful for herd-level runs.
    # In the R code, SCALE == 1 sets MAXCALFNR <- 0, i.e. individual-animal mode with no calf births.
    # The translated Python code previously still entered the reproductive-cow culling loop and used
    # WNDAY values such as 9999 as matrix indices, causing out-of-bounds access.
    # For individual runs, carry the single animal ANIMALINFO forward as the one-animal output and
    # skip herd culling probabilities. Herd-level behavior is unchanged in the else branch.
    if SCALE == 1 or int(MAXCALFNR) == 0:
        _individual_info = np.asarray(HERDINFO1[0, :], dtype=float).reshape(-1)
        if _individual_info.shape[0] < 18:
            _individual_info = np.pad(_individual_info, (0, 18 - _individual_info.shape[0]), constant_values=0.0)
        elif _individual_info.shape[0] > 18:
            _individual_info = _individual_info[:18]

        REPRINFO2 = _individual_info.copy()
        PRODINFO2 = np.zeros(18, dtype=float)
    else:
        # Culling of reproductive cows, vector with probabilities for survival
        # Culling starts after weaning the first calf
        AZZAMCUMCORR = np.array([
            CULL,
            (1 - CULL) - (1 - CULL) ** 2,
            (1 - CULL) ** 2 - (1 - CULL) ** 3,
            (1 - CULL) ** 3 - (1 - CULL) ** 4,
            (1 - CULL) ** 4 - (1 - CULL) ** 5,
            (1 - CULL) ** 5 - (1 - CULL) ** 6,
            (1 - CULL) ** 6 - (1 - CULL) ** 7,
            (1 - CULL) ** 7 - (1 - CULL) ** 8
        ], dtype=float)

        # Calculate chance that a cow is still in the herd after weaning a calf
        AA = (WNDAY[WNDAY > 0] / 365) - (GestPer + WEANINGTIME) / 365  # Moment of conception (year)
        BB = np.floor(AA) + 1                                          # Moment of conception rounded (year)
        BB[0] = 1

        # Vector with probabilities for survival
        # The probability for giving birth to the first calf (replacement) in a herd unit equals 1
        CC = AZZAMCUMCORR.copy()
        CC[len(CC) - 1] = 1 - np.sum(CC[0:len(CC) - 1])

        # Specification of key metrics for the reproductive cow over her total life span
        # (accounting for culling)
        REPRBEEF = np.full(len(AA), np.nan)      # Beef production (kg)
        REPRLW = np.full(len(AA), np.nan)        # Live weight (kg)
        REPRFEED = np.full(len(AA), np.nan)      # Feed intake (kg DM)
        REPRFEED1 = np.full(len(AA), np.nan)     # Intake feed type 1 (kg DM)
        REPRFEED2 = np.full(len(AA), np.nan)     # Intake feed type 2 (kg DM)
        REPRFEED3 = np.full(len(AA), np.nan)     # Intake feed type 3 (kg DM)
        REPRFEED4 = np.full(len(AA), np.nan)     # Intake feed type 4 (kg DM)

        REPRFCR = np.full(len(AA), np.nan)       # Feed conversion ratio (kg DM feed per kg live weight)
        REPRAVANW = np.full(len(AA), np.nan)     # Average total body weight (kg live weight)
        REPRAVANWMET = np.full(len(AA), np.nan)  # Average metabolic body weight (kg0.75 empty body weight)

        # Calculates the beef production and feed intake for different culling scenarios for the
        # cow
        for p in range(1, len(AA - 8 + MAXCALFNR) + 1):
            wnd_idx = int(WNDAY[p - 1]) - 1

            # Beef production (kg)
            REPRBEEF[p - 1] = MUSCLETISBF[wnd_idx, 0] + INTRAMFTISBF[wnd_idx, 0] + MISCFATTISBF[wnd_idx, 0]
            # Live weight (kg)
            REPRLW[p - 1] = TBWBF[wnd_idx, 0]
            # Feed intake (kg DM)
            REPRFEED[p - 1] = CUMULFEED[wnd_idx, 0]
            # Intake feed type 1 (kg DM)
            REPRFEED1[p - 1] = CUMULFEED1[wnd_idx, 0]
            # Intake feed type 2 (kg DM)
            REPRFEED2[p - 1] = CUMULFEED2[wnd_idx, 0]
            # Intake feed type 3 (kg DM)
            REPRFEED3[p - 1] = CUMULFEED3[wnd_idx, 0]
            # Intake feed type 4 (kg DM)
            REPRFEED4[p - 1] = CUMULFEED4[wnd_idx, 0]
            # Feed conversion ratio (kg DM feed per kg live weight)
            REPRFCR[p - 1] = FCR[wnd_idx, 0]
            # Average total body weight (kg live weight)
            REPRAVANW[p - 1] = np.mean(TBWBF[0:int(WNDAY[p - 1]), 0])
            # Average metabolic body weight (kg0.75 empty body weight)
            REPRAVANWMET[p - 1] = np.mean(EBWBFMET[0:int(WNDAY[p - 1]), 0])

        CC1 = CC.copy()  # CC is duplicated

        # Probabilities for the scenarios
        CC1 = np.concatenate([np.array([0.0]), CC1[0:8]])
        CC1 = np.concatenate([CC1, np.repeat(0.0, 9 - len(CC1))])

        # Vector REPRINFO lists key information on the reproductive cow in a herd unit
        REPRINFO = np.column_stack([
            REPRODUCTIVE[0:len(AA)],
            REPLACEMENT[0:len(AA)],
            PRODUCTIVE[0:len(AA)],
            SEX[0:len(AA)],
            REPRBEEF,
            REPRFEED,
            REPRFCR,
            WNDAY[0:len(AA)] / 365,
            REPRAVANW,
            REPRAVANWMET,
            WNDAY[0:len(AA)],
            REPRLW,
            CC1[0:len(AA)],
            CC1[0:len(AA)],
            REPRFEED1,
            REPRFEED2,
            REPRFEED3,
            REPRFEED4
        ])

        # Multiplies production and feed intake with probabilities (add up to a probability of 1)
        REPRINFO1 = REPRINFO * CC1[0:REPRINFO.shape[0], np.newaxis]
        # Lists production and feed intake of the cow, including the culling probabilities
        REPRINFO2 = np.sum(REPRINFO1, axis=0)

        # works only at a MAXCALFNR equal to or higher than 3! (Otherwise warnings appear)
        # orginal
        '''
        if MAXCALFNR == 0:
            PRODINFO = np.repeat(0.0, 18)
        else:
            if MAXCALFNR < 3:
                PRODINFO = HERDINFO1[2, :]
            else:
                PRODINFO = HERDINFO1[2:(MAXCALFNR + 1), :]

        # Probabilities for all animals in a herd unit to exist
        DD = np.array([
            1,
            1,
            (1 - (CC[0])),
            (1 - (np.sum(CC[0:2]))),
            (1 - (np.sum(CC[0:3]))),
            (1 - (np.sum(CC[0:4]))),
            (1 - (np.sum(CC[0:5]))),
            (1 - (np.sum(CC[0:6]))),
            (1 - (np.sum(CC[0:7]))),
            (1 - (np.sum(CC[0:8]))),
            (1 - (np.sum(CC[0:9])))
        ], dtype=float)

        # Multiply production and feed intake of calves not used for replacement with probabilities
        if MAXCALFNR < 3:
            PRODINFO1 = PRODINFO
        else:
            PRODINFO1 = PRODINFO * DD[2:(MAXCALFNR + 1), np.newaxis]
        '''
        ########################### new instead of orginal #############################

        maxcalfnr_int = int(MAXCALFNR)
        # works only at a MAXCALFNR equal to or higher than 3! (Otherwise warnings appear)
        if maxcalfnr_int == 0:
            PRODINFO = np.repeat(0.0, 18)
        elif maxcalfnr_int < 3:
            PRODINFO = HERDINFO1[2, :]
        else:
            PRODINFO = HERDINFO1[2:(maxcalfnr_int + 1), :]

        # Probabilities for all animals in a herd unit to exist
        DD = np.array([
            1,
            1,
            (1 - (CC[0])),
            (1 - (np.sum(CC[0:2]))),
            (1 - (np.sum(CC[0:3]))),
            (1 - (np.sum(CC[0:4]))),
            (1 - (np.sum(CC[0:5]))),
            (1 - (np.sum(CC[0:6]))),
            (1 - (np.sum(CC[0:7]))),
            (1 - (np.sum(CC[0:8]))),
            (1 - (np.sum(CC[0:9])))
        ], dtype=float)

        # Multiply production and feed intake of calves not used for replacement with probabilities
        if maxcalfnr_int < 3:
            PRODINFO1 = PRODINFO
        else:
            PRODINFO1 = PRODINFO * DD[2:(maxcalfnr_int + 1), np.newaxis]

        ###########################################################

        # Lists production and feed intake for calves not used for replacement, including the
        # culling probabilities
        if MAXCALFNR < 3:
            PRODINFO2 = PRODINFO
        else:
            PRODINFO2 = np.sum(PRODINFO1, axis=0)

    # The vectors below list the information for the herd unit
    HERDINFO2 = np.sum(np.vstack([REPRINFO2, PRODINFO2]), axis=0)

    OUTPUTHERD = np.vstack([REPRINFO2, PRODINFO2, HERDINFO2])
    OUTPUT1 = np.column_stack([
        Metheatopt,
        TNRESP,
        ACTSW,
        LWRCOAT,
        CONVCOAT,
        SWR,
        TskinC,
        TcoatC,
        TAVGC,
        MetheatBAL
    ])
    _herd_beef = float(HERDINFO2[4])
    _herd_feed = float(HERDINFO2[5])
    OUTPUT2 = np.column_stack([
        _herd_beef,
        _herd_feed,
        (_herd_beef / _herd_feed) if abs(_herd_feed) > 1e-12 else np.nan,
        (_herd_feed / _herd_beef) if abs(_herd_beef) > 1e-12 else np.nan
    ])

    # Matrix with key information for one herd unit
    if OUTPUTHERDS is None or (isinstance(OUTPUTHERDS, list) and len(OUTPUTHERDS) == 0):
        OUTPUTHERDS = OUTPUTHERD
    else:
        OUTPUTHERDS = np.vstack([OUTPUTHERDS, OUTPUTHERD])


    # Matrix with key information for one herd unit
    if OUTPUTHERDS is None or (isinstance(OUTPUTHERDS, list) and len(OUTPUTHERDS) == 0):
        OUTPUTHERDS = OUTPUTHERD
    else:
        OUTPUTHERDS = np.vstack([OUTPUTHERDS, OUTPUTHERD])

    # Feed efficiency (g beef kg DM intake) for the calves in a herd unit
    if (
        np.isfinite(OUTPUTHERDS[1, 4]) and
        np.isfinite(OUTPUTHERDS[1, 5]) and
        OUTPUTHERDS[1, 5] > 0 and
        OUTPUTHERDS[1, 4] > 0
    ):
        FESENSIND[s - 1] = OUTPUTHERDS[1, 4] / OUTPUTHERDS[1, 5] * 1000
    else:
        FESENSIND[s - 1] = 0.0

    # Feed efficiency (g beef kg DM intake) for cow in a herd unit
    if np.isfinite(OUTPUTHERDS[0, 5]) and OUTPUTHERDS[0, 5] > 0:
        FESENSREPR[s - 1] = OUTPUTHERDS[0, 4] / OUTPUTHERDS[0, 5] * 1000
    else:
        FESENSREPR[s - 1] = 0.0

    # Feed efficiency (g beef kg DM intake) for the herd unit
    if np.isfinite(OUTPUTHERDS[2, 5]) and OUTPUTHERDS[2, 5] > 0:
        FESENSHERD[s - 1] = OUTPUTHERDS[2, 4] / OUTPUTHERDS[2, 5] * 1000
    else:
        FESENSHERD[s - 1] = 0.0
    ###########################################################################################

    # End of the s-loop for sensitivity analysis
########################## line:4923 ##############################
    # Vector COLNAMES indicates the parameters used for sensitivity analysis (sensitivity
    # analysis not performed in this code)
    COLNAMES = [
        "CoatConst",
        "ZC",
        "TbodyC",
        "LASMIN",
        "PHFEEDCAP",
        "RESPINCR",
        "PROTFRACBONE",
        "PROTFRACMUSCLE",
        "LIPFRACMUSCLE",
        "PROTFRACFAT",
        "LIPFRACFAT",
        "INCARC",
        "RUMENFRAC",
        "NEm",
        "NEpha",
        "BONEFRACMAX",
        "LIPNONCMAX",
        "LIPNONCMIN",
        "PROTEFF",
        "LIPIDEFF",
        "DERMPL",
        "PROTNE",
        "GestPer",
        "GESTINTERVAL",
        "WEANINGTIME",
        "FtoConcW",
        "FATFACTOR",
        "RAINEXP",
        "FRACVEG",
        "COMPFACT",
        "NEIEFFGEST",
        "CPGEST",
        "MILKDIG",
        "NEEFFMILK",
        "PROTFRACMILK",
        "PROTEFFMILK",
        "COMPFACTTIS",
        "FATTISCOMP",
        "TTDIGINSC",
        "DETOME",
        "DISSEFF",

        "reflectivity coat",
        "coat length",
        "area corr",
        "max body core-skin conductance",
        "birth weight",
        "milk A",
        "milk B",
        "adult max. weight",
        "F",
        "milk A calf",
        "milk B calf",
        "fraction TBW fertility",
        "maintenance factor",
        "min perc. for gestation",
        "fat fraction bone parameter",
        "carcass fraction",
        "muscle:bone ratio",
        "min. cond. core-skin. par.",
        "LHRskin A",
        "LHRskin B",
        "LHRskin C",

        "BONE A",
        "BONE B",
        "MUSCLE A",
        "MUSCLE B",
        "INTRAMF A",
        "INTRAMF B",
        "INTRAMF C",
        "PROTNONC A",
        "PROTNONC B",
        "RESP",
        "AREA A",
        "AREA B",
        "DIAM A",
        "DIAM B",
        "BRR A",
        "BRR B",
        "BTV A",
        "TEXH A",
        "TEXH B",
        "TEXH C",
        "TEXH D",
        "CBSMIN A",
        "CBSMIN B",
        "RAINFRAC",
        "RAINEVAP",
        "NEGEST",
        "GEMILK A",
        "GEMILK B",
        "FATBONE A",
        "FATBONE B",
        "FATBONE C",
        "FATNONC A",
        "FATNONC B",
        "FATNONC C",
        "FATNONC D",
        "PROTNONC A",
        "PROTNONC B",
        "PROTNONC C",
        "PROTNONC D",
        "PROTNONC E",
        "PHFEED A",
        "PHFEED B",
        "NDFDIG A",
        "NDFDIG B",
        "LUCAS A",
        "LUCAS B",
        "NONCGR A",
        "NONCGR B",
        "NRECYCL A",
        "NRECYCL B",
        "NRECYCL C",
        "Reference"
    ]

    # Feed efficiency for individual cattle under sensitivity analysis (not part of this code)
    FESENSIND = np.array(FESENSIND, dtype=float).reshape(NPAR, 1)

    #############################################################################################
    #                   3. Output section                      #
    ############################################################

    ############################################################
    #                     Graph for cows                       #
    ############################################################

    def _col_as_1d(arr, col_idx_0):
        """
        R-style helper:
        - if arr is 1D, return first n values
        - if arr is 2D, return the requested column
        """
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            return a.copy()
        return a[:, col_idx_0].copy()

    def _fit_to_len(vec, target_len, fill_value=np.nan):
        """Return a 1D float vector with exactly target_len elements.

        NumPy requires equal-length vectors for stacked bars and elementwise
        operations. Growth-optimizer/SCALE=1 runs may simulate fewer days than
        the fixed plotting window, so short feed/housing vectors are padded
        safely and then hidden by the existing end-day mask.
        """
        a = np.asarray(vec, dtype=float).reshape(-1)
        target_len = int(target_len)
        if target_len <= 0:
            return np.array([], dtype=float)
        if len(a) >= target_len:
            return a[:target_len].copy()
        out = np.full(target_len, fill_value, dtype=float)
        out[:len(a)] = a
        return out

    def _slice_r_1_based(vec, start_r, end_r_inclusive):
        """
        R-style 1-based inclusive slice on a 1D numpy vector.
        """
        start0 = max(start_r - 1, 0)
        end0 = min(end_r_inclusive, len(vec))
        return vec[start0:end0]

    def _set_na_r_1_based(vec, start_r, end_r_inclusive=None):
        """
        R-style NA assignment from start_r:end_r_inclusive.
        If end_r_inclusive is None, assign from start_r to end.
        """
        if end_r_inclusive is None:
            end_r_inclusive = len(vec)
        start0 = max(start_r - 1, 0)
        end0 = min(end_r_inclusive, len(vec))
        if start0 < end0:
            vec[start0:end0] = np.nan

    def _stacked_bar(ax, bars, colors, labels, ylim=(0, 19)):
        """
        Matplotlib equivalent of R stacked barplot(space = 0, border = NA).

        Safe behavior:
        - trims bars to the number of provided colors/labels
        - avoids IndexError if bars has more rows than expected
        """
        bars = np.asarray(bars, dtype=float)

        if bars.ndim == 1:
            bars = bars.reshape(1, -1)

        n_series = min(bars.shape[0], len(colors), len(labels))
        bars = bars[:n_series, :]

        n_days = bars.shape[1]
        x = np.arange(1, n_days + 1, dtype=float)
        bottom = np.zeros(n_days, dtype=float)

        for i in range(n_series):
            y = np.nan_to_num(bars[i], nan=0.0)
            ax.bar(
                x,
                y,
                width=1.0,
                bottom=bottom,
                color=colors[i],
                edgecolor="none",
                align="edge"
            )
            bottom += y

        ax.set_xlim(1, n_days + 1)
        ax.set_ylim(*ylim)
        ax.set_ylabel(r"Feed intake (kg day$^{-1}$)")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.legend(labels[:n_series], loc="upper left", frameon=False)

    def _mask_series_after_endday(vec, endday_r, maxgr=None):
        """
        Return a 1-D plotting vector whose values after the animal's simulated
        lifetime are NA. R masks some series explicitly with ENDDAY; applying
        the same boundary to all defining/limiting markers prevents translated
        Python graphs from drawing event bars beyond the growth period when a
        matrix column is padded to 1000/4000 days.
        """
        out = np.asarray(vec, dtype=float).copy().reshape(-1)
        if maxgr is None:
            maxgr = len(out)
        try:
            e = int(np.asarray(endday_r).reshape(-1)[0])
        except Exception:
            e = maxgr
        e = max(0, min(e, maxgr, len(out)))
        if e < len(out):
            out[e:] = np.nan
        return out

    def _last_valid_growth_day(vec, maxgr):
        """Last day with a finite positive simulated TBW value inside plotting window."""
        y = np.asarray(vec, dtype=float).reshape(-1)[:maxgr]
        mask = np.isfinite(y) & (y > 0)
        if np.any(mask):
            return int(np.where(mask)[0][-1] + 1)
        return maxgr

    def _resolve_plot_endday(raw_endday, growth_vec, maxgr, scale_value):
        """Resolve the plot cutoff safely for individual-animal and herd modes."""
        try:
            e = int(float(np.asarray(raw_endday).reshape(-1)[0]))
        except Exception:
            e = 0
        last_growth = _last_valid_growth_day(growth_vec, maxgr)
        upper = min(maxgr, len(np.asarray(growth_vec).reshape(-1)))
        # In SCALE==1, ENDDAY can remain at its R initialization sentinel (1)
        # when there is no herd-level slaughter/culling cutoff. Using that value
        # directly masks the whole graph. Fall back to the simulated growth window.
        if int(scale_value) == 1 and (e <= 1 or e >= 9999):
            return max(1, min(last_growth, upper))
        if e <= 0 or e >= 9999:
            return max(1, min(last_growth, upper))
        return max(1, min(e, last_growth, upper))

    def _plot_marker_series(ax, values, maxgr, color):
        """Plot only finite event markers; never connect or extend padded zeros."""
        y = np.asarray(values, dtype=float).reshape(-1)[:maxgr]
        x = np.arange(1, len(y) + 1, dtype=float)
        mask = np.isfinite(y) & (y > 0)
        if np.any(mask):
            ax.plot(
                x[mask], y[mask],
                linestyle="None", marker="|", color=color, markersize=7
            )

    def _limiting_factor_plot(
        ax,
        heatstress,
        coldstress,
        fillgitgraph,
        nelim,
        protgraph,
        genlim,
        maxgr,
        title_x="Age (days)"
    ):
        _plot_marker_series(ax, heatstress, maxgr, "#D55E00")
        _plot_marker_series(ax, coldstress, maxgr, "#0072B2")
        _plot_marker_series(ax, fillgitgraph, maxgr, "#009E73")
        _plot_marker_series(ax, nelim, maxgr, "#E69F00")
        _plot_marker_series(ax, protgraph, maxgr, "#CC79A7")
        _plot_marker_series(ax, genlim, maxgr, "#999999")

        ax.set_xlim(1, maxgr)
        ax.set_ylim(0.3, 5.8)
        ax.set_xlabel(title_x)
        ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        ax.set_yticklabels(
            [
                "protein",
                "energy",
                "digestion cap.",
                "cold stress",
                "heat stress",
                "genotype"
            ]
        )
        ax.tick_params(axis="y", labelrotation=0)

    # 1. TBW over time
    if SCALE == 1:
        maxgr = 1000
    else:
        maxgr = 4000

    fig_cows = plt.figure(figsize=(12, 10))
    gs_cows = GridSpec(3, 1, height_ratios=[1.8, 1.1, 1.4], hspace=0.08)

    ax1 = fig_cows.add_subplot(gs_cows[0])
    ax2 = fig_cows.add_subplot(gs_cows[1], sharex=ax1)
    ax3 = fig_cows.add_subplot(gs_cows[2], sharex=ax1)

    x_cows = np.arange(1, maxgr + 1, dtype=float)

    tbw_cow = _col_as_1d(TBW, 0)
    tbwbf_cow = _col_as_1d(TBWBF, 0)

    # R draws the potential TBW reference for maxgr days. The simulated curve,
    # however, must stop at the actual ENDDAY to avoid showing padded tail values
    # as if the cow continued growing after leaving the herd unit.
    endday1 = int(np.asarray(ENDDAY).reshape(-1)[0])
    endday1_plot = _resolve_plot_endday(endday1, tbwbf_cow, maxgr, SCALE)

    # Match the biologically valid growth window: do not draw the genetic
    # potential reference beyond the animal's simulated growth/lifetime.
    # Padded matrix tails in the translated Python version can otherwise make
    # the solid TBW line longer than the valid growth period.
    x_pot_cow = np.arange(1, endday1_plot + 1, dtype=float)
    ax1.plot(
        x_pot_cow,
        tbw_cow[:endday1_plot],
        linestyle="-",
        linewidth=1.5,
        color="black",
        label="Genetic potential TBW"
    )
    ax1.plot(
        np.arange(1, endday1_plot + 1, dtype=float),
        tbwbf_cow[:endday1_plot],
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="Simulated TBW"
    )
    ax1.set_xlim(1, maxgr)
    ax1.set_ylim(0, LIBRARY[12])  # R LIBRARY[13] -> Python index 12
    ax1.set_ylabel("TBW (kg)")
    ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax1.legend(loc="upper left", frameon=False)

    # 2. Feed intake over time
    feed1_cow = _fit_to_len(_col_as_1d(FEED1QNTY, 0), maxgr, fill_value=0.0)
    feed2_cow = _fit_to_len(_col_as_1d(FEED2QNTY, 0), maxgr, fill_value=0.0)
    feed3_cow = _fit_to_len(_col_as_1d(FEED3QNTY, 0), maxgr, fill_value=0.0)

    # FEED4QNTY is not always explicitly created in the Python version.
    # Keep R logic safely: if unavailable, use zeros.
    if "FEED4QNTY" in locals():
        feed4_cow = _fit_to_len(_col_as_1d(FEED4QNTY, 0), maxgr, fill_value=0.0)
    else:
        feed4_cow = np.zeros(maxgr, dtype=float)

    bars = np.vstack([feed1_cow, feed2_cow, feed3_cow, feed4_cow])

    if FEEDNR[case_template_idx] == 2:
        housing1_cow = _fit_to_len(_col_as_1d(HOUSING1, 0) if "HOUSING1" in locals() else _col_as_1d(HOUSING, 0), maxgr, fill_value=0.0)
        bars = np.vstack([
            feed1_cow,
            feed2_cow - feed2_cow * housing1_cow,
            feed2_cow * housing1_cow
        ])

    if FEEDNR[case_template_idx] == 4:
        housing1_cow = _fit_to_len(_col_as_1d(HOUSING1, 0) if "HOUSING1" in locals() else _col_as_1d(HOUSING, 0), maxgr, fill_value=0.0)
        bars = np.vstack([
            feed1_cow,
            feed3_cow - feed3_cow * housing1_cow,
            feed3_cow * housing1_cow
        ])

    if FEEDNR[case_template_idx] == 5:
        housing1_cow = _fit_to_len(_col_as_1d(HOUSING1, 0) if "HOUSING1" in locals() else _col_as_1d(HOUSING, 0), maxgr, fill_value=0.0)
        bars = np.vstack([
            feed1_cow,
            feed2_cow - feed2_cow * housing1_cow,
            feed2_cow * housing1_cow
        ])

    # R: bars[, ENDDAY[1]:maxgr] <- NA
    if 1 <= endday1_plot <= maxgr:
        bars[:, endday1_plot - 1:maxgr] = np.nan

    if FEEDNR[case_template_idx] == 1:
        _stacked_bar(
            ax2,
            bars,
            colors=["gold", "darkkhaki"],
            labels=["Wheat", "Hay"],
            ylim=(0, 19)
        )

    if FEEDNR[case_template_idx] == 2:
        _stacked_bar(
            ax2,
            bars,
            colors=["orange", "darkkhaki", "seagreen"],
            labels=["Barley", "Hay", "Grass"],
            ylim=(0, 19)
        )

    if FEEDNR[case_template_idx] == 3:
        _stacked_bar(
            ax2,
            bars,
            colors=["orange", "seagreen"],
            labels=["Barley", "Grass"],
            ylim=(0, 19)
        )

    if FEEDNR[case_template_idx] == 4:
        _stacked_bar(
            ax2,
            bars,
            colors=["orange", "darkkhaki", "seagreen"],
            labels=["Barley", "Hay", "Grass"],
            ylim=(0, 19)
        )

    if FEEDNR[case_template_idx] == 5:
        _stacked_bar(
            ax2,
            bars,
            colors=["orange", "darkkhaki", "seagreen"],
            labels=["Barley", "Hay", "Grass"],
            ylim=(0, 19)
        )

    # 3. Defining and limiting factors for growth over time
    HEATSTRESS = np.array(REDHP, dtype=float, copy=True)
    HEATSTRESS[HEATSTRESS < 0] = 4.5
    HEATSTRESS[HEATSTRESS != 4.5] = np.nan

    COLDSTRESS1 = np.array(Metheatcold, dtype=float) - np.array(HEATIFEEDMAINTWM, dtype=float)
    COLDSTRESS = np.array(COLDSTRESS1, dtype=float, copy=True)
    COLDSTRESS[COLDSTRESS > 0] = 3.5
    COLDSTRESS[COLDSTRESS != 3.5] = np.nan

    FILLGITGRAPH = np.array(FILLGIT, dtype=float, copy=True)
    FILLGITGRAPH[FILLGITGRAPH >= 0.97] = 2.5
    FILLGITGRAPH[FILLGITGRAPH != 2.5] = np.nan

    PROTGRAPH = np.array(PROTBAL, dtype=float, copy=True)
    PROTGRAPH[PROTGRAPH < 0] = 0.5
    PROTGRAPH[PROTGRAPH != 0.5] = np.nan
    PROTGRAPH[FILLGITGRAPH >= 0.999] = np.nan

    HEATSTRESS[np.isnan(HEATSTRESS)] = 0
    COLDSTRESS[np.isnan(COLDSTRESS)] = 0
    FILLGITGRAPH[np.isnan(FILLGITGRAPH)] = 0
    PROTGRAPH[np.isnan(PROTGRAPH)] = 0

    TBWBF = np.array(TBWBF, dtype=float, copy=True)
    TBWBF[np.isnan(TBWBF)] = 0

    if FEEDNR[case_template_idx] == 5:
        # R: FEEDQNTYTOT <- FEEDQNTYTOT[1:(imax[1] + 1)] * TBWBF / 100
        # Python adaptation for current code structure
        feedqnty_tot_work = np.array(FEEDQNTYTOT, dtype=float, copy=True)
        tbwbf_for_feed = _col_as_1d(TBWBF, 0)
        nmin = min(len(feedqnty_tot_work), len(tbwbf_for_feed))
        feedqnty_tot_work = feedqnty_tot_work[:nmin] * tbwbf_for_feed[:nmin] / 100.0
    else:
        feedqnty_tot_work = np.array(FEEDQNTYTOT, dtype=float, copy=True)

    # Cow column
    feedqnty_cow = _col_as_1d(FEEDQNTY, 0)

    nmin = min(
        len(feedqnty_tot_work),
        len(feedqnty_cow),
        len(_col_as_1d(HEATSTRESS, 0)),
        len(_col_as_1d(PROTGRAPH, 0)),
        len(_col_as_1d(FILLGITGRAPH, 0))
    )

    NELIM = feedqnty_tot_work[:nmin] - feedqnty_cow[:nmin]
    NELIM = (
        NELIM
        + _col_as_1d(HEATSTRESS, 0)[:nmin]
        + _col_as_1d(PROTGRAPH, 0)[:nmin]
        + _col_as_1d(FILLGITGRAPH, 0)[:nmin]
    )

    # R source contains: NELIM[NELIM <- 0.00001] <- NA
    # This is not valid as a direct Python condition and also looks like an R typo.
    # To preserve the intended logic used in surrounding lines, the lower threshold is applied as < -0.00001.
    NELIM[NELIM > 0.00001] = np.nan
    NELIM[NELIM < -0.00001] = np.nan
    _set_na_r_1_based(NELIM, endday1_plot, maxgr)

    NELIM[NELIM < 0.00001] = 1.5
    NELIM[NELIM > -0.00001] = 1.5
    NELIM[np.isnan(NELIM)] = 0.0

    GENLIM = (
        _col_as_1d(HEATSTRESS, 0)[:nmin]
        + _col_as_1d(FILLGITGRAPH, 0)[:nmin]
        + _col_as_1d(PROTGRAPH, 0)[:nmin]
        + NELIM
    )
    GENLIM[GENLIM > 0] = np.nan
    GENLIM[GENLIM == 0] = 5.5
    _set_na_r_1_based(GENLIM, endday1_plot, maxgr)

    HEATSTRESS_cow = _mask_series_after_endday(_col_as_1d(HEATSTRESS, 0)[:nmin], endday1_plot, maxgr)
    COLDSTRESS_cow = _mask_series_after_endday(_col_as_1d(COLDSTRESS, 0)[:nmin], endday1_plot, maxgr)
    FILLGITGRAPH_cow = _mask_series_after_endday(_col_as_1d(FILLGITGRAPH, 0)[:nmin], endday1_plot, maxgr)
    PROTGRAPH_cow = _mask_series_after_endday(_col_as_1d(PROTGRAPH, 0)[:nmin], endday1_plot, maxgr)

    HEATSTRESS_cow[HEATSTRESS_cow == 0] = np.nan
    COLDSTRESS_cow[COLDSTRESS_cow == 0] = np.nan
    FILLGITGRAPH_cow[FILLGITGRAPH_cow == 0] = np.nan
    NELIM[NELIM == 0] = np.nan
    NELIM = _mask_series_after_endday(NELIM, endday1_plot, maxgr)
    PROTGRAPH_cow[PROTGRAPH_cow == 0] = np.nan

    # pad to maxgr for plotting
    def _pad_to_maxgr(v, maxgr):
        out = np.full(maxgr, np.nan, dtype=float)
        m = min(len(v), maxgr)
        out[:m] = v[:m]
        return out

    HEATSTRESS_plot = _pad_to_maxgr(HEATSTRESS_cow, maxgr)
    COLDSTRESS_plot = _pad_to_maxgr(COLDSTRESS_cow, maxgr)
    FILLGITGRAPH_plot = _pad_to_maxgr(FILLGITGRAPH_cow, maxgr)
    NELIM_plot = _pad_to_maxgr(NELIM, maxgr)
    PROTGRAPH_plot = _pad_to_maxgr(PROTGRAPH_cow, maxgr)
    GENLIM_plot = _pad_to_maxgr(GENLIM, maxgr)

    _limiting_factor_plot(
        ax3,
        HEATSTRESS_plot,
        COLDSTRESS_plot,
        FILLGITGRAPH_plot,
        NELIM_plot,
        PROTGRAPH_plot,
        GENLIM_plot,
        maxgr
    )

    # Cow-panel PNG generation is disabled in the customized package.
    # The cow calculations remain available for herd aggregation/export, but no
    # ligaps_case_X_cows.png file is written.
    plt.close(fig_cows)

    # Lists of the defining and limiting factors
    GENLIMdata       = np.column_stack((GENLIMdata, _pad_to_maxgr(GENLIM, 4000)))
    HEATSTRESSdata   = np.column_stack((HEATSTRESSdata, _pad_to_maxgr(HEATSTRESS_cow, 4000)))
    COLDSTRESSdata   = np.column_stack((COLDSTRESSdata, _pad_to_maxgr(COLDSTRESS_cow, 4000)))
    FILLGITGRAPHdata = np.column_stack((FILLGITGRAPHdata, _pad_to_maxgr(FILLGITGRAPH_cow, 4000)))
    NELIMdata        = np.column_stack((NELIMdata, _pad_to_maxgr(NELIM, 4000)))
    PROTGRAPHdata    = np.column_stack((PROTGRAPHdata, _pad_to_maxgr(PROTGRAPH_cow, 4000)))

    #############################################################################################
    #                     Graph for calves                     #
    ############################################################
    # In SCALE == 1 the R model sets MAXCALFNR <- 0: individual-animal mode.
    # There is only one animal column, so herd-only calf plots using column 3 must be skipped.
    _n_animals_for_output = np.asarray(TBW).shape[1] if np.asarray(TBW).ndim == 2 else 1
    _has_calf_output = (SCALE != 1) and (int(MAXCALFNR) > 0) and (_n_animals_for_output >= 3) and (np.asarray(ENDDAY).size >= 3)

    if _has_calf_output:

        # 1. TBW over time
        maxgr = 1000

        fig_calves = plt.figure(figsize=(12, 10))
        gs_calves = GridSpec(3, 1, height_ratios=[1.8, 1.1, 1.4], hspace=0.08)

        ax1 = fig_calves.add_subplot(gs_calves[0])
        ax2 = fig_calves.add_subplot(gs_calves[1], sharex=ax1)
        ax3 = fig_calves.add_subplot(gs_calves[2], sharex=ax1)

        x_calves = np.arange(1, maxgr + 1, dtype=float)

        tbw_calf = _col_as_1d(TBW, 2)
        tbwbf_calf = _col_as_1d(TBWBF, 2)

        endday3 = int(np.asarray(ENDDAY).reshape(-1)[2])
        endday3_plot = max(1, min(endday3, _last_valid_growth_day(tbwbf_calf, maxgr), maxgr, len(tbwbf_calf)))

        # Match the biologically valid growth window for calves too. The genetic
        # potential curve is clipped to the same valid growth boundary as the
        # simulated TBW curve, preventing over-long solid TBW tails.
        x_pot_calf = np.arange(1, endday3_plot + 1, dtype=float)
        ax1.plot(
            x_pot_calf,
            tbw_calf[:endday3_plot],
            linestyle="-",
            linewidth=1.5,
            color="black",
            label="Genetic potential TBW"
        )

        x_sim_calf = np.arange(1, endday3_plot + 1, dtype=float)
        ax1.plot(
            x_sim_calf,
            tbwbf_calf[:endday3_plot],
            linestyle="--",
            linewidth=1.5,
            color="black",
            label="Simulated TBW"
        )

        ax1.set_xlim(1, maxgr)
        ax1.set_ylim(0, LIBRARY[12])  # R LIBRARY[13]
        ax1.set_ylabel("TBW (kg)")
        ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax1.legend(loc="upper left", frameon=False)

        # 2. feed intake over time
        feed1_calf = _col_as_1d(FEED1QNTY, 2)[:maxgr] if np.asarray(FEED1QNTY).ndim == 2 else _col_as_1d(FEED1QNTY, 0)[:maxgr]
        feed2_calf = _col_as_1d(FEED2QNTY, 2)[:maxgr] if np.asarray(FEED2QNTY).ndim == 2 else _col_as_1d(FEED2QNTY, 0)[:maxgr]
        feed3_calf = _col_as_1d(FEED3QNTY, 2)[:maxgr] if np.asarray(FEED3QNTY).ndim == 2 else _col_as_1d(FEED3QNTY, 0)[:maxgr]

        if "FEED4QNTY" in locals():
            feed4_calf = _col_as_1d(FEED4QNTY, 2)[:maxgr] if np.asarray(FEED4QNTY).ndim == 2 else _col_as_1d(FEED4QNTY, 0)[:maxgr]
        else:
            feed4_calf = np.zeros_like(feed1_calf, dtype=float)

        housing_calf = _col_as_1d(HOUSING, 0)

        # force all calf feed vectors to the same usable length
        common_len = min(
            len(feed1_calf),
            len(feed2_calf),
            len(feed3_calf),
            len(feed4_calf),
            len(housing_calf),
            maxgr
        )

        feed1_calf = feed1_calf[:common_len]
        feed2_calf = feed2_calf[:common_len]
        feed3_calf = feed3_calf[:common_len]
        feed4_calf = feed4_calf[:common_len]
        housing_calf = housing_calf[:common_len]

        if FEEDNR[case_template_idx] == 1:
            bars = np.vstack([feed1_calf, feed2_calf])

        elif FEEDNR[case_template_idx] == 2:
            bars = np.vstack([
                feed1_calf,
                feed2_calf - feed2_calf * housing_calf,
                feed2_calf * housing_calf
            ])

        elif FEEDNR[case_template_idx] == 3:
            bars = np.vstack([feed1_calf, feed3_calf])

        elif FEEDNR[case_template_idx] == 4:
            bars = np.vstack([
                feed1_calf,
                feed3_calf - feed3_calf * housing_calf,
                feed3_calf * housing_calf
            ])

        elif FEEDNR[case_template_idx] == 5:
            bars = np.vstack([
                feed1_calf,
                feed2_calf - feed2_calf * housing_calf,
                feed2_calf * housing_calf
            ])

        else:
            bars = np.vstack([feed1_calf, feed2_calf])



        # apply ENDDAY masking using the actual plotted length
        if 1 <= endday3_plot <= common_len:
            bars[:, endday3_plot - 1:common_len] = np.nan

        if FEEDNR[case_template_idx] == 1:
            _stacked_bar(
                ax2,
                bars,
                colors=["gold", "darkkhaki"],
                labels=["Wheat", "Hay"],
                ylim=(0, 19)
            )

        if FEEDNR[case_template_idx] == 2:
            _stacked_bar(
                ax2,
                bars,
                colors=["orange", "darkkhaki", "seagreen"],
                labels=["Barley", "Hay", "Grass"],
                ylim=(0, 19)
            )

        if FEEDNR[case_template_idx] == 3:
            _stacked_bar(
                ax2,
                bars,
                colors=["orange", "seagreen"],
                labels=["Barley", "Grass"],
                ylim=(0, 19)
            )

        if FEEDNR[case_template_idx] == 4:
            _stacked_bar(
                ax2,
                bars,
                colors=["orange", "darkkhaki", "seagreen"],
                labels=["Barley", "Hay", "Grass"],
                ylim=(0, 19)
            )

        if FEEDNR[case_template_idx] == 5:
            _stacked_bar(
                ax2,
                bars,
                colors=["orange", "darkkhaki", "seagreen"],
                labels=["Barley", "Hay", "Grass"],
                ylim=(0, 19)
            )

        # 3. Defining and limiting factors for growth over time
        HEATSTRESS = np.array(REDHP, dtype=float, copy=True)
        HEATSTRESS[HEATSTRESS < 0] = 4.5
        HEATSTRESS[HEATSTRESS != 4.5] = np.nan

        COLDSTRESS1 = np.array(Metheatcold, dtype=float) - np.array(HEATIFEEDMAINTWM, dtype=float)
        COLDSTRESS = np.array(COLDSTRESS1, dtype=float, copy=True)
        COLDSTRESS[COLDSTRESS > 0] = 3.5
        COLDSTRESS[COLDSTRESS != 3.5] = np.nan

        FILLGITGRAPH = np.array(FILLGIT, dtype=float, copy=True)
        FILLGITGRAPH[FILLGITGRAPH >= 0.97] = 2.5
        FILLGITGRAPH[FILLGITGRAPH != 2.5] = np.nan

        PROTGRAPH = np.array(PROTBAL, dtype=float, copy=True)
        PROTGRAPH[PROTGRAPH < 0] = 0.5
        PROTGRAPH[PROTGRAPH != 0.5] = np.nan
        PROTGRAPH[FILLGITGRAPH >= 0.999] = np.nan

        HEATSTRESS[np.isnan(HEATSTRESS)] = 0
        COLDSTRESS[np.isnan(COLDSTRESS)] = 0
        FILLGITGRAPH[np.isnan(FILLGITGRAPH)] = 0
        PROTGRAPH[np.isnan(PROTGRAPH)] = 0

        TBWBF = np.array(TBWBF, dtype=float, copy=True)
        TBWBF[np.isnan(TBWBF)] = 0

        if FEEDNR[case_template_idx] == 5:
            feedqnty_tot_work = np.array(FEEDQNTYTOT, dtype=float, copy=True)
            if feedqnty_tot_work.ndim == 1:
                tbwbf_for_feed = _col_as_1d(TBWBF, 0)
                nmin2 = min(len(feedqnty_tot_work), len(tbwbf_for_feed))
                feedqnty_tot_work = feedqnty_tot_work[:nmin2] * tbwbf_for_feed[:nmin2] / 100.0
            else:
                feedqnty_tot_work = feedqnty_tot_work * TBWBF / 100.0
        else:
            feedqnty_tot_work = np.array(FEEDQNTYTOT, dtype=float, copy=True)

        # R: NELIM <- c(rep(FEEDQNTYTOT[1:4000], 9)) - FEEDQNTY
        # Python adaptation:
        if np.asarray(feedqnty_tot_work).ndim == 1:
            base = np.array(feedqnty_tot_work[:4000], dtype=float)
            if np.asarray(FEEDQNTY).ndim == 2:
                n_animals = np.asarray(FEEDQNTY).shape[1]
                nelim_base = np.tile(base.reshape(-1, 1), (1, n_animals))
            else:
                nelim_base = base.copy()
        else:
            nelim_base = np.array(feedqnty_tot_work, dtype=float)

        NELIM = nelim_base - np.array(FEEDQNTY, dtype=float)
        NELIM = NELIM + HEATSTRESS + PROTGRAPH + FILLGITGRAPH

        NELIM[NELIM > 0.00001] = np.nan
        NELIM[NELIM < -0.00001] = np.nan

        NELIM[NELIM < 0.00001] = 1.5
        NELIM[NELIM > -0.00001] = 1.5

        if np.asarray(NELIM).ndim == 2:
            if 1 <= endday3_plot <= min(maxgr, NELIM.shape[0]):
                NELIM[endday3_plot - 1:maxgr, :] = np.nan
        else:
            _set_na_r_1_based(NELIM, endday3_plot, maxgr)

        NELIM[np.isnan(NELIM)] = 0.0

        GENLIM = HEATSTRESS + FILLGITGRAPH + PROTGRAPH + NELIM
        GENLIM[GENLIM > 0] = np.nan
        GENLIM[GENLIM == 0] = 5.5

        if np.asarray(GENLIM).ndim == 2:
            if 1 <= endday3_plot <= min(maxgr, GENLIM.shape[0]):
                GENLIM[endday3_plot - 1:maxgr, :] = np.nan
        else:
            _set_na_r_1_based(GENLIM, endday3_plot, maxgr)

        HEATSTRESS_calf = _mask_series_after_endday(_col_as_1d(HEATSTRESS, 2 if np.asarray(HEATSTRESS).ndim == 2 else 0), endday3_plot, maxgr)
        COLDSTRESS_calf = _mask_series_after_endday(_col_as_1d(COLDSTRESS, 2 if np.asarray(COLDSTRESS).ndim == 2 else 0), endday3_plot, maxgr)
        FILLGITGRAPH_calf = _mask_series_after_endday(_col_as_1d(FILLGITGRAPH, 2 if np.asarray(FILLGITGRAPH).ndim == 2 else 0), endday3_plot, maxgr)
        NELIM_calf = _mask_series_after_endday(_col_as_1d(NELIM, 2 if np.asarray(NELIM).ndim == 2 else 0), endday3_plot, maxgr)
        PROTGRAPH_calf = _mask_series_after_endday(_col_as_1d(PROTGRAPH, 2 if np.asarray(PROTGRAPH).ndim == 2 else 0), endday3_plot, maxgr)
        GENLIM_calf = _mask_series_after_endday(_col_as_1d(GENLIM, 2 if np.asarray(GENLIM).ndim == 2 else 0), endday3_plot, maxgr)

        HEATSTRESS_calf[HEATSTRESS_calf == 0] = np.nan
        COLDSTRESS_calf[COLDSTRESS_calf == 0] = np.nan
        FILLGITGRAPH_calf[FILLGITGRAPH_calf == 0] = np.nan
        NELIM_calf[NELIM_calf == 0] = np.nan
        PROTGRAPH_calf[PROTGRAPH_calf == 0] = np.nan

        _limiting_factor_plot(
            ax3,
            _pad_to_maxgr(HEATSTRESS_calf, maxgr),
            _pad_to_maxgr(COLDSTRESS_calf, maxgr),
            _pad_to_maxgr(FILLGITGRAPH_calf, maxgr),
            _pad_to_maxgr(NELIM_calf, maxgr),
            _pad_to_maxgr(PROTGRAPH_calf, maxgr),
            _pad_to_maxgr(GENLIM_calf, maxgr),
            maxgr
        )

        fig_calves.savefig(BASE_DIR / f"ligaps_case_{z}_calves.png", dpi=300, bbox_inches="tight")
        plt.close(fig_calves)

        # End of the graph for calves
    else:
        pass

    #############################################################################################

    # Table with key information about cattle performance (information also presented in Table 3,
    # paper Van der Linden et al. (2017a))

    if _has_calf_output:
        DATAt3 = np.array([
            np.asarray(FESENSHERD).reshape(-1)[s - 1],
            np.asarray(FESENSREPR).reshape(-1)[s - 1],
            np.asarray(FESENSIND).reshape(-1)[s - 1],
            OUTPUTHERDS[0, 5] / OUTPUTHERDS[2, 5],
            OUTPUTHERDS[2, 4],
            OUTPUTHERDS[0, 4],
            OUTPUTHERDS[1, 4],
            TBWBF[endday3 - 1, 2] if np.asarray(TBWBF).ndim == 2 else TBWBF[endday3 - 1]
        ], dtype=float)

        TABLEDATA = pd.DataFrame(
            DATAt3.reshape(8, 1),
            columns=["Herd level"],
            index=[
                "Feed efficiency herd unit (g beef kg-1 DM)",
                "Feed efficiency repr. cow (g beef kg-1 DM)",
                "Feed efficiency bull calf (g beef kg-1 DM)",
                "Feed fraction repr. cow (-)",
                "Beef production herd unit (kg)",
                "Beef production repr. cow (kg)",
                "Beef production bull calf (kg)",
                "Slaughter weight bull calf (kg)"
            ]
        )
    else:
        _tbwbf_out = _col_as_1d(TBWBF, 0)
        _feed_out = _col_as_1d(FEEDQNTY, 0) if np.asarray(FEEDQNTY).ndim == 2 else np.asarray(FEEDQNTY, dtype=float).reshape(-1)
        _end_out_raw = _resolve_plot_endday(int(np.asarray(ENDDAY).reshape(-1)[0]), _tbwbf_out, min(1000, len(_tbwbf_out)), SCALE)
        _valid_tbw_mask = np.isfinite(_tbwbf_out[:_end_out_raw]) & (_tbwbf_out[:_end_out_raw] > 0)
        _end_out = int(np.where(_valid_tbw_mask)[0][-1] + 1) if np.any(_valid_tbw_mask) else _end_out_raw
        _final_tbw = float(_tbwbf_out[_end_out - 1]) if len(_tbwbf_out) >= _end_out else np.nan
        _total_feed = float(np.nansum(_feed_out[:_end_out]))
        _fe_ind = float(np.asarray(FESENSIND).reshape(-1)[s - 1]) if np.asarray(FESENSIND).size >= s else np.nan
        DATAt3 = np.array([_fe_ind, _final_tbw, _total_feed, float(_end_out)], dtype=float)

        TABLEDATA = pd.DataFrame(
            DATAt3.reshape(4, 1),
            columns=["Animal level"],
            index=[
                "Feed efficiency individual animal (g beef kg-1 DM)",
                "Final total body weight (kg)",
                "Cumulative feed intake (kg DM)",
                "Simulation length / final day (days)"
            ]
        )

    _growth_output_dir = os.environ.get("LIGAPS_OUTPUT_DIR")
    if _growth_output_dir:
        try:
            _write_growth_optimizer_outputs(
                out_dir=_growth_output_dir, case_id=z, scale=SCALE, breed=BREED, diet=FEEDNR[case_template_idx],
                observed_until=locals().get("_observed_until_day", None) or imax, local_vars=locals())
        except Exception as _export_exc:
            print(f"WARNING: failed to export growth optimizer outputs: {_export_exc}", file=sys.stderr)

    print("Case number")  # Case number corresponds to the case number in Table 3
    print(z)              # z is the case number
    print(TABLEDATA)

# End z-loop for the cases

# Some numerical/plotting backends keep non-daemon worker threads alive after the
# monolithic script has finished.  Library/subprocess callers can request a hard
# clean exit after all outputs are flushed.
if os.environ.get("LIGAPS_FORCE_EXIT", "0") == "1":
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

