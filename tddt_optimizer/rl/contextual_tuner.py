from __future__ import annotations
from pathlib import Path
from typing import Any
import json, math


def _f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


class EdgeContextualBayesianTuner:
    """Lightweight Gaussian-Bayesian Online Tuning (LBG-OT).

    This class is intentionally edge-oriented.  It does not implement a full
    Gaussian process or a heavy Bayesian optimizer.  It implements the formal
    LBG-OT layer from doc/GAUSSIAN_BAYESIAN.md:

    1) Unsupervised diagonal-Gaussian context layer
       - observes only online state vectors z_t;
       - updates per-context means/variances by EMA;
       - returns novelty/uncertainty u_t from the normalized diagonal distance.

    2) Supervised Bayesian profile tuner
       - keeps a small discrete set of MPC weight/bias profiles;
       - updates profile posterior probabilities from observed reward/error;
       - never directly drives actuators; it only soft-tunes MPC guidance.

    3) Safe fusion
       - if the Gaussian layer is uncertain, the final guidance is pulled toward
         a conservative safe profile.

    The public class name is kept compatible with previous packages.
    """

    PROFILES = {
        "balanced": {
            "comfort_weight_bias": 1.00, "energy_weight_bias": 1.00, "gas_weight_bias": 1.00,
            "ventilation_bias": 0.00, "heating_bias": 0.00,
        },
        "comfort_safe": {
            "comfort_weight_bias": 1.30, "energy_weight_bias": 0.95, "gas_weight_bias": 1.05,
            "ventilation_bias": 0.05, "heating_bias": 0.05,
        },
        "energy_saver": {
            "comfort_weight_bias": 0.95, "energy_weight_bias": 0.65, "gas_weight_bias": 1.00,
            "ventilation_bias": -0.05, "heating_bias": -0.05,
        },
        "air_quality": {
            "comfort_weight_bias": 1.05, "energy_weight_bias": 1.00, "gas_weight_bias": 1.40,
            "ventilation_bias": 0.20, "heating_bias": 0.00,
        },
        "anti_waste": {
            "comfort_weight_bias": 1.05, "energy_weight_bias": 0.80, "gas_weight_bias": 1.05,
            "ventilation_bias": -0.10, "heating_bias": -0.10,
        },
    }

    SAFE_PROFILE = {
        "comfort_weight_bias": 1.35,
        "energy_weight_bias": 1.00,
        "gas_weight_bias": 1.15,
        "ventilation_bias": 0.00,
        "heating_bias": 0.00,
    }

    FEATURE_NAMES = [
        "indoor_temp_c", "indoor_rh_pct", "indoor_air_speed_m_s",
        "outdoor_temp_c", "outdoor_rh_pct", "outdoor_wind_m_s", "heat_production",
    ]

    FEATURE_SCALE = [20.0, 100.0, 2.0, 40.0, 100.0, 10.0, 50.0]

    def __init__(
        self,
        alpha: float = 0.05,
        exploration_weight: float = 0.15,
        gaussian_alpha: float = 0.025,
        gaussian_tau: float = 7.0,
        safe_blend_max: float = 0.85,
    ):
        self.alpha = float(alpha)
        self.exploration_weight = float(exploration_weight)
        self.gaussian_alpha = float(gaussian_alpha)
        self.gaussian_tau = float(max(1e-6, gaussian_tau))
        self.safe_blend_max = float(_clip(safe_blend_max, 0.0, 1.0))
        self.gaussian_contexts: dict[str, dict[str, Any]] = {}
        self.profile_stats: dict[str, dict[str, dict[str, float]]] = {}
        self.posteriors: dict[str, dict[str, float]] = {}
        self.last_context: str | None = None
        self.last_profile: str | None = None
        self.last_gaussian_key: str | None = None
        self.last_uncertainty: float = 0.0
        self.last_z: list[float] | None = None

    # ------------------------------------------------------------------
    # Context/state representation
    # ------------------------------------------------------------------
    def _context_key(self, state: dict, guidance: dict | None = None) -> str:
        guidance = guidance or {}
        ctx = str(state.get("climate_context_id") or guidance.get("climate_context_id") or "C_UNKNOWN")
        phase = str(guidance.get("biological_phase") or guidance.get("phase") or state.get("growth_phase") or "P_UNKNOWN")
        temp = _f(state.get("indoor_temp_c"), 18.0)
        low = _f(guidance.get("preferred_temp_low_c"), _f(state.get("lct_c"), -5.0))
        high = _f(guidance.get("preferred_temp_high_c"), _f(state.get("uct_c"), 27.0))
        if temp < low:
            band = "below"
        elif temp > high:
            band = "above"
        else:
            mid = (low + high) / 2.0
            band = "inside_low" if temp < mid else "inside_high"
        hum = "humid" if _f(state.get("indoor_rh_pct"), 60.0) > 75.0 else "normal"
        return f"ctx={ctx}|phase={phase}|band={band}|hum={hum}"

    def _state_vector(self, state: dict, guidance: dict | None = None) -> list[float]:
        guidance = guidance or {}
        hp = _f(state.get("heat_production"), _f(guidance.get("heat_production_feedback"), 0.0))
        raw = [
            _f(state.get("indoor_temp_c"), 18.0),
            _f(state.get("indoor_rh_pct"), 60.0),
            _f(state.get("indoor_air_speed_m_s"), _f(state.get("air_speed_m_s"), 0.2)),
            _f(state.get("outdoor_temp_c"), 12.0),
            _f(state.get("outdoor_rh_pct"), 60.0),
            _f(state.get("outdoor_wind_m_s"), _f(state.get("wind_m_s"), 1.0)),
            hp,
        ]
        return [raw[i] / self.FEATURE_SCALE[i] for i in range(len(raw))]

    # ------------------------------------------------------------------
    # Gaussian unsupervised layer
    # ------------------------------------------------------------------
    def _diag_distance(self, z: list[float], g: dict[str, Any]) -> float:
        mu = g.get("mu") or [0.0] * len(z)
        var = g.get("var") or [1.0] * len(z)
        total = 0.0
        for zi, mi, vi in zip(z, mu, var):
            total += (float(zi) - float(mi)) ** 2 / max(float(vi), 1e-6)
        return float(total)

    def _nearest_gaussian(self, z: list[float]) -> tuple[str | None, float]:
        if not self.gaussian_contexts:
            return None, self.gaussian_tau
        best_key, best_d = None, 1e18
        for key, g in self.gaussian_contexts.items():
            d = self._diag_distance(z, g)
            if d < best_d:
                best_key, best_d = key, d
        return best_key, float(best_d)

    def _update_gaussian(self, key: str, z: list[float]) -> dict[str, Any]:
        g = self.gaussian_contexts.get(key)
        if not g:
            g = {"mu": list(z), "var": [0.25] * len(z), "n": 1.0}
            self.gaussian_contexts[key] = g
            return g
        a = self.gaussian_alpha
        mu = [float(x) for x in g.get("mu", z)]
        var = [max(1e-6, float(x)) for x in g.get("var", [0.25] * len(z))]
        new_mu, new_var = [], []
        for zi, mi, vi in zip(z, mu, var):
            nm = (1.0 - a) * mi + a * zi
            nv = max(1e-6, (1.0 - a) * vi + a * (zi - mi) ** 2)
            new_mu.append(nm); new_var.append(nv)
        g.update({"mu": new_mu, "var": new_var, "n": float(g.get("n", 0.0)) + 1.0})
        return g

    def gaussian_observe(self, state: dict, guidance: dict | None = None) -> dict:
        key = self._context_key(state, guidance)
        z = self._state_vector(state, guidance)
        nearest_key, dist_before = self._nearest_gaussian(z)
        # Update the nominal context after measuring novelty. This keeps the
        # unsupervised layer independent from reward labels.
        self._update_gaussian(key, z)
        nearest_key = nearest_key or key
        u = _clip(dist_before / self.gaussian_tau, 0.0, 1.0)
        self.last_context = key
        self.last_gaussian_key = nearest_key
        self.last_uncertainty = u
        self.last_z = z
        return {
            "context_key": key,
            "nearest_gaussian_key": nearest_key,
            "gaussian_distance": float(dist_before),
            "gaussian_uncertainty": float(u),
            "feature_names": list(self.FEATURE_NAMES),
        }

    # ------------------------------------------------------------------
    # Bayesian supervised profile layer
    # ------------------------------------------------------------------
    def _ensure_profile_state(self, context_key: str) -> None:
        if context_key not in self.profile_stats:
            self.profile_stats[context_key] = {name: {"mean": 0.0, "var": 1.0, "n": 0.0} for name in self.PROFILES}
        if context_key not in self.posteriors:
            p = 1.0 / max(1, len(self.PROFILES))
            self.posteriors[context_key] = {name: p for name in self.PROFILES}

    def choose_profile(self, state: dict, guidance: dict | None = None) -> tuple[str, dict, dict]:
        ginfo = self.gaussian_observe(state, guidance)
        key = ginfo["context_key"]
        self._ensure_profile_state(key)
        posterior = self.posteriors[key]
        stats = self.profile_stats[key]
        best_name, best_score = "balanced", -1e18
        for name in self.PROFILES:
            s = stats.get(name, {"mean": 0.0, "var": 1.0, "n": 0.0})
            p = float(posterior.get(name, 0.0))
            ucb = self.exploration_weight * math.sqrt(max(1e-6, float(s.get("var", 1.0)))) / math.sqrt(float(s.get("n", 0.0)) + 1.0)
            score = p + float(s.get("mean", 0.0)) + ucb
            if score > best_score:
                best_name, best_score = name, score
        self.last_profile = best_name
        return best_name, dict(self.PROFILES[best_name]), ginfo

    def _fuse_profile_with_safe(self, profile: dict, uncertainty: float) -> dict:
        u = _clip(uncertainty, 0.0, self.safe_blend_max)
        fused = {}
        for k, pv in profile.items():
            sv = float(self.SAFE_PROFILE.get(k, pv))
            fused[k] = (1.0 - u) * float(pv) + u * sv
        return fused

    def apply_to_guidance(self, state: dict, guidance: dict) -> dict:
        out = dict(guidance or {})
        profile_name, profile, ginfo = self.choose_profile(state, out)
        uncertainty = float(ginfo.get("gaussian_uncertainty", 0.0))
        fused = self._fuse_profile_with_safe(profile, uncertainty)
        # Multiplicative weights for existing MPC biases; additive for actuator
        # bias terms because guidance uses them as soft desired fractions.
        for k in ("comfort_weight_bias", "energy_weight_bias", "gas_weight_bias"):
            out[k] = _f(out.get(k), 1.0) * float(fused.get(k, 1.0))
        for k in ("ventilation_bias", "heating_bias"):
            out[k] = _clip(_f(out.get(k), 0.0) + float(fused.get(k, 0.0)), -1.0, 1.0)
        out.update({
            "bayesian_profile": profile_name,
            "bayesian_context_key": self.last_context,
            "lbg_ot_enabled": True,
            "lbg_gaussian_context_key": ginfo.get("nearest_gaussian_key"),
            "lbg_gaussian_distance": ginfo.get("gaussian_distance", 0.0),
            "lbg_gaussian_uncertainty": uncertainty,
            "model_uncertainty_radius": max(_f(out.get("model_uncertainty_radius"), 0.0), uncertainty),
            "lbg_posterior": self.posteriors.get(self.last_context or "", {}),
        })
        return out

    def update(self, reward: float, context_key: str | None = None, profile: str | None = None) -> dict | None:
        key = context_key or self.last_context
        selected = profile or self.last_profile
        if not key or not selected:
            return None
        self._ensure_profile_state(key)
        reward = float(reward)
        stats = self.profile_stats[key]
        posterior = self.posteriors[key]
        # Update reward moments for the actually selected profile.
        s = stats.setdefault(selected, {"mean": 0.0, "var": 1.0, "n": 0.0})
        old_mean = float(s.get("mean", 0.0))
        a = self.alpha
        new_mean = old_mean + a * (reward - old_mean)
        new_var = max(1e-6, (1.0 - a) * float(s.get("var", 1.0)) + a * (reward - new_mean) ** 2)
        s.update({"mean": new_mean, "var": new_var, "n": float(s.get("n", 0.0)) + 1.0})
        # Lightweight Bayes posterior over discrete profiles.
        numer = {}
        compat = math.exp(-0.5 * max(0.0, float(self.last_uncertainty)))
        for name in self.PROFILES:
            st = stats.get(name, {"mean": 0.0, "var": 1.0, "n": 0.0})
            mean = float(st.get("mean", 0.0))
            var = max(1e-6, float(st.get("var", 1.0)))
            likelihood = math.exp(-((reward - mean) ** 2) / (2.0 * var + 1e-6))
            numer[name] = max(1e-12, float(posterior.get(name, 1.0 / len(self.PROFILES))) * likelihood * compat)
        denom = sum(numer.values()) or 1.0
        self.posteriors[key] = {name: float(val / denom) for name, val in numer.items()}
        return {
            "layer": "lbg_ot",
            "context_key": key,
            "profile": selected,
            "reward": reward,
            "profile_mean": new_mean,
            "profile_var": new_var,
            "profile_n": s["n"],
            "gaussian_uncertainty": self.last_uncertainty,
            "posterior": self.posteriors[key],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "algorithm": "LBG-OT",
            "formal_layer": "lightweight_gaussian_bayesian_online_tuning",
            "description": "Unsupervised diagonal-Gaussian novelty/uncertainty + supervised discrete Bayesian MPC-profile tuner.",
            "alpha": self.alpha,
            "exploration_weight": self.exploration_weight,
            "gaussian_alpha": self.gaussian_alpha,
            "gaussian_tau": self.gaussian_tau,
            "safe_blend_max": self.safe_blend_max,
            "feature_names": self.FEATURE_NAMES,
            "feature_scale": self.FEATURE_SCALE,
            "profiles": self.PROFILES,
            "safe_profile": self.SAFE_PROFILE,
            "gaussian_contexts": self.gaussian_contexts,
            "profile_stats": self.profile_stats,
            "posteriors": self.posteriors,
        }

    @classmethod
    def from_file(cls, path: str | Path, **defaults) -> "EdgeContextualBayesianTuner":
        p = Path(path)
        obj = cls(**defaults)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                obj.alpha = float(data.get("alpha", obj.alpha))
                obj.exploration_weight = float(data.get("exploration_weight", obj.exploration_weight))
                obj.gaussian_alpha = float(data.get("gaussian_alpha", obj.gaussian_alpha))
                obj.gaussian_tau = float(data.get("gaussian_tau", obj.gaussian_tau))
                obj.safe_blend_max = float(data.get("safe_blend_max", obj.safe_blend_max))
                # Backward compatibility with the older conceptual tuner.
                obj.profile_stats = data.get("profile_stats") or data.get("stats", {}) or {}
                obj.gaussian_contexts = data.get("gaussian_contexts", {}) if isinstance(data.get("gaussian_contexts", {}), dict) else {}
                obj.posteriors = data.get("posteriors", {}) if isinstance(data.get("posteriors", {}), dict) else {}
            except Exception:
                pass
        return obj

    def save(self, path: str | Path) -> str:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return str(p)
