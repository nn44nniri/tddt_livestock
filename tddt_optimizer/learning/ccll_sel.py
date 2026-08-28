from __future__ import annotations
from pathlib import Path
import json
import math
import pandas as pd
import numpy as np


FEATURE_COLUMNS = ["t_min", "t_max", "t_mean", "rh_mean", "wind_mean", "rad_sum", "rain_sum", "t_range"]


def _safe_float(v, default=0.0):
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _risk_and_prior_from_centroid(centroid: dict) -> tuple[str, dict, dict]:
    """Convert a numeric CCLL centroid to a readable name, risk profile, and soft MPC prior."""
    t = _safe_float(centroid.get("t_mean"), 0.0)
    rh = _safe_float(centroid.get("rh_mean"), 60.0)
    wind = _safe_float(centroid.get("wind_mean"), 1.0)
    if t < -5:
        temp_cls = "very_cold"
    elif t < 5:
        temp_cls = "cold"
    elif t < 15:
        temp_cls = "mild"
    elif t < 25:
        temp_cls = "warm"
    else:
        temp_cls = "hot"
    hum_cls = "dry" if rh < 40 else ("humid" if rh > 70 else "normal_humidity")
    wind_cls = "low_wind" if wind < 1 else ("windy" if wind > 4 else "normal_wind")
    name = f"{temp_cls}_{hum_cls}_{wind_cls}"
    risk = {
        "heat_stress": "high" if temp_cls == "hot" and hum_cls == "humid" else ("medium" if temp_cls in {"warm", "hot"} else "low"),
        "cold_stress": "high" if temp_cls in {"very_cold", "cold"} else "low",
        "draft_risk": "high" if wind_cls == "windy" and temp_cls in {"very_cold", "cold"} else "low",
        "humidity_pressure": "high" if hum_cls == "humid" else "low",
    }
    high_risk = any(v == "high" for v in risk.values())
    prior = {
        "ventilation_bias": 0.80 if risk["heat_stress"] == "high" else (0.45 if temp_cls in {"warm", "hot"} else 0.15),
        "heating_bias": 0.85 if risk["cold_stress"] == "high" else 0.0,
        "comfort_weight_bias": 0.90 if high_risk else 0.55,
        "energy_weight_bias": 0.35 if high_risk else 0.70,
        "context_penalty_weight": 0.25 if high_risk else 0.10,
    }
    return name, risk, prior


def _daily_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["date"] = x["timestamp"].dt.date
    for col in ["outdoor_temp_c", "outdoor_rh_pct", "outdoor_wind_m_s", "outdoor_solar_w_m2", "outdoor_rain_mm_day"]:
        if col not in x.columns:
            x[col] = 0.0
    g = x.groupby("date", sort=True)
    daily = pd.DataFrame({
        "date": [str(d) for d, _ in g],
        "t_min": g["outdoor_temp_c"].min().values,
        "t_max": g["outdoor_temp_c"].max().values,
        "t_mean": g["outdoor_temp_c"].mean().values,
        "rh_mean": g["outdoor_rh_pct"].mean().values,
        "wind_mean": g["outdoor_wind_m_s"].mean().values,
        "rad_sum": g["outdoor_solar_w_m2"].sum().values,
        "rain_sum": g["outdoor_rain_mm_day"].mean().values,
    })
    daily["t_range"] = daily["t_max"] - daily["t_min"]
    return daily


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std[std < 1e-9] = 1.0
    scaled = (values - mean) / std
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled, mean, std


def _init_centroids_farthest(x: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Deterministic farthest-point initialization; avoids adding sklearn as dependency."""
    n = x.shape[0]
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n)) if n else 0
    chosen = [first]
    while len(chosen) < k:
        existing = x[chosen]
        dist2 = ((x[:, None, :] - existing[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        for idx in chosen:
            dist2[idx] = -1.0
        chosen.append(int(np.argmax(dist2)))
    return x[chosen].copy()


def _kmeans_nearest_centroid(values: np.ndarray, k: int, max_iter: int = 80, seed: int = 42) -> tuple[np.ndarray, np.ndarray, float]:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("CCLL clustering requires at least one descriptor row")
    k = max(1, min(int(k), values.shape[0]))
    centroids = _init_centroids_farthest(values, k, seed=seed)
    labels = np.zeros(values.shape[0], dtype=int)
    for _ in range(max(1, int(max_iter))):
        dist2 = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = values[mask].mean(axis=0)
            else:
                # Re-seed empty cluster with the worst represented point.
                nearest = dist2.min(axis=1)
                centroids[j] = values[int(nearest.argmax())]
    final_dist2 = ((values - centroids[labels]) ** 2).sum(axis=1)
    inertia = float(final_dist2.sum())
    return labels, centroids, inertia


def _representative_days(grp: pd.DataFrame, centroid: dict, limit: int = 10) -> list[str]:
    cols = [c for c in FEATURE_COLUMNS if c in grp.columns]
    if not cols:
        return grp.sort_values("date").head(limit)["date"].astype(str).tolist()
    center = np.array([_safe_float(centroid.get(c)) for c in cols], dtype=float)
    vals = grp[cols].fillna(0.0).to_numpy(dtype=float)
    dist = ((vals - center[None, :]) ** 2).sum(axis=1)
    idx = np.argsort(dist)[:limit]
    return grp.iloc[idx]["date"].astype(str).tolist()


def build_ccll_from_5m(
    prepared_5m_csv: str | Path,
    output_dir: str | Path = "prepared",
    k_hint: int | None = None,
    max_iter: int = 80,
    seed: int = 42,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Build CCLL-SEL artifacts using nearest-centroid clustering.

    The 11-year/local 5-minute climate sequence is first aggregated into daily
    descriptors. A lightweight K-means implementation creates local climate
    contexts, and each day is assigned to its nearest centroid. The resulting
    context is mapped back to every 5-minute row for inner-loop MPC/RL priors.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(prepared_5m_csv, parse_dates=["timestamp"])
    if df.empty:
        raise ValueError("prepared 5-minute climate file is empty")
    daily = _daily_descriptors(df)
    cols = list(feature_columns or FEATURE_COLUMNS)
    cols = [c for c in cols if c in daily.columns]
    if not cols:
        raise ValueError("No valid CCLL feature columns found")
    values = daily[cols].fillna(0.0).to_numpy(dtype=float)
    scaled, mean, std = _standardize(values)
    if k_hint is None:
        k_hint = int(max(3, min(12, round(math.sqrt(len(daily))))))
    labels, centroids_scaled, inertia = _kmeans_nearest_centroid(scaled, k_hint, max_iter=max_iter, seed=seed)
    daily["cluster_index"] = labels.astype(int)
    # Build centroids in original feature space.
    centroids_original = centroids_scaled * std + mean
    contexts = []
    for j in sorted(set(labels.tolist())):
        cid = f"C{j:02d}"
        grp = daily[daily["cluster_index"] == j].copy()
        centroid = {col: _safe_float(centroids_original[j, i]) for i, col in enumerate(cols)}
        # Include all canonical feature keys for stable downstream usage.
        for c in FEATURE_COLUMNS:
            centroid.setdefault(c, _safe_float(grp[c].mean()) if c in grp else 0.0)
        cname_base, risk, prior = _risk_and_prior_from_centroid(centroid)
        context_name = f"cluster_{j:02d}_{cname_base}"
        daily.loc[daily["cluster_index"] == j, "context_id"] = cid
        daily.loc[daily["cluster_index"] == j, "context_name"] = context_name
        contexts.append({
            "context_id": cid,
            "context_name": context_name,
            "support_count": int(len(grp)),
            "assignment_method": "nearest_centroid_clustering",
            "centroid": centroid,
            "centroid_scaled": {col: _safe_float(centroids_scaled[j, i]) for i, col in enumerate(cols)},
            "risk_profile": risk,
            "mpc_prior": prior,
            "representative_days": _representative_days(grp, centroid),
        })
    library = {
        "method": "CCLL-SEL",
        "library_type": "climate_context_local_library",
        "source_5m_csv": str(prepared_5m_csv),
        "assignment_method": "nearest_centroid_clustering",
        "context_count": len(contexts),
        "requested_context_count": int(k_hint),
        "feature_columns": cols,
        "normalization": {
            "mean": {col: _safe_float(mean[i]) for i, col in enumerate(cols)},
            "std": {col: _safe_float(std[i]) for i, col in enumerate(cols)},
        },
        "inertia": inertia,
        "time_resolution": "5min",
        "contexts": contexts,
    }
    # Map daily cluster context back to every 5-minute row.
    x = df.copy()
    x["date"] = x["timestamp"].dt.date.astype(str)
    day_map = daily[["date", "context_id", "context_name", "cluster_index"]].copy()
    df_aug = x.merge(day_map, on="date", how="left").drop(columns=["date"])
    aug_path = out / "climate_5m_ccll_all_rows.csv"
    lib_path = out / "climate_context_local_library.json"
    desc_path = out / "climate_context_daily_descriptors.csv"
    # Write the small library/descriptor artifacts before the large augmented CSV.
    # If a long preparation is interrupted, the clustering model is still inspectable.
    lib_path.write_text(json.dumps(library, indent=2, ensure_ascii=False), encoding="utf-8")
    daily.to_csv(desc_path, index=False)
    df_aug.to_csv(aug_path, index=False)
    return {
        "library_path": str(lib_path),
        "daily_descriptors_csv": str(desc_path),
        "prepared_5m_with_context_csv": str(aug_path),
        "assignment_method": "nearest_centroid_clustering",
        "context_count": len(contexts),
        "rows_5m": len(df_aug),
        "inertia": inertia,
    }


def nearest_context(library: dict, vector: dict) -> dict:
    """Map a new climate vector to the nearest CCLL centroid."""
    best = None
    best_dist = float("inf")
    keymap = {
        "outdoor_temp_c": "t_mean",
        "outdoor_rh_pct": "rh_mean",
        "outdoor_wind_m_s": "wind_mean",
        "outdoor_solar_w_m2": "rad_sum",
        "outdoor_rain_mm_day": "rain_sum",
    }
    for ctx in library.get("contexts", []):
        centroid = ctx.get("centroid", {})
        dist = 0.0
        used = 0
        for vkey, ckey in keymap.items():
            if vkey not in vector and ckey not in vector:
                continue
            v = _safe_float(vector.get(vkey, vector.get(ckey)), 0.0)
            c = _safe_float(centroid.get(ckey), 0.0)
            dist += (v - c) ** 2
            used += 1
        if used == 0:
            continue
        if dist < best_dist:
            best_dist = dist
            best = ctx
    out = dict(best or {})
    out["distance"] = math.sqrt(best_dist) if best is not None else None
    return out
