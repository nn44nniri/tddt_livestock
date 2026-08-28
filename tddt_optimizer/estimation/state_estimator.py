from __future__ import annotations
from dataclasses import dataclass, asdict
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


@dataclass
class EstimatorUpdate:
    timestamp: str | None
    temp_error_c: float
    rh_error_pct: float
    model_bias_temp_c: float
    model_bias_rh_pct: float
    theta_vent: float
    theta_heat: float
    theta_cattle: float
    uncertainty_radius: float
    layer: str = "state_estimation_identification"


class EdgeStateEstimator:
    """Edge-light implementation of Layer 2 in LAYERS_FORMALISM.

    It is not a heavy EKF/MHE. It keeps a bounded recursive bias/parameter memory
    that corrects the climate simulator with the most recent prediction residuals:

        e_t = y_real_t - y_sim_t
        theta_{t+1} = theta_t + K_t e_t

    For TRAIN, ``y_real`` is the next streamed sensor row while ``y_sim`` is the
    simulator prediction from the previously selected actuator. For WORK-ONLINE,
    it can be updated in the same way from real funnel packets.
    """
    def __init__(self, alpha: float = 0.025, uncertainty_decay: float = 0.995, min_uncertainty: float = 0.05):
        self.alpha = float(alpha)
        self.uncertainty_decay = float(uncertainty_decay)
        self.min_uncertainty = float(min_uncertainty)
        self.model_bias_temp_c = 0.0
        self.model_bias_rh_pct = 0.0
        self.theta = {
            "theta_UA": 1.0,
            "theta_vent": 1.0,
            "theta_heat": 1.0,
            "theta_cattle": 1.0,
            "theta_humidity": 1.0,
            "theta_gas": 1.0,
        }
        self.uncertainty_radius = 1.0
        self.last_prediction: dict | None = None
        self.last_command: dict | None = None
        self.updates = 0

    def enrich_state(self, state: dict) -> dict:
        out = dict(state)
        # Simulator-corrected state used by MPC scoring. Keep raw values too.
        if "indoor_temp_c" in out:
            out["estimated_indoor_temp_c"] = _f(out.get("indoor_temp_c"), 18.0) + self.model_bias_temp_c
        if "indoor_rh_pct" in out:
            out["estimated_indoor_rh_pct"] = max(0.0, min(100.0, _f(out.get("indoor_rh_pct"), 60.0) + self.model_bias_rh_pct))
        out["model_uncertainty_radius"] = self.uncertainty_radius
        out["theta_vent"] = self.theta["theta_vent"]
        out["theta_heat"] = self.theta["theta_heat"]
        out["theta_cattle"] = self.theta["theta_cattle"]
        return out

    def guidance_adjustment(self) -> dict:
        # Robust adaptive MPC-inspired soft adjustment: when uncertainty is high,
        # comfort and safety margins are emphasized; when prediction bias suggests
        # systematic heat/cool mismatch, shift actuator priors softly.
        u = max(self.min_uncertainty, self.uncertainty_radius)
        return {
            "estimator_uncertainty_radius": u,
            "comfort_weight_bias": 1.0 + min(0.5, 0.10 * u),
            "heating_bias": max(0.0, -self.model_bias_temp_c / 10.0),
            "ventilation_bias": max(0.0, self.model_bias_temp_c / 10.0),
            "estimator_theta": dict(self.theta),
        }

    def remember_prediction(self, pred: dict, command: dict) -> None:
        self.last_prediction = dict(pred or {})
        self.last_command = dict(command or {})

    def update_from_observation(self, observed_state: dict) -> EstimatorUpdate | None:
        if not self.last_prediction:
            return None
        pred = self.last_prediction
        temp_error = _f(observed_state.get("indoor_temp_c"), _f(pred.get("indoor_temp_c"), 18.0)) - _f(pred.get("indoor_temp_c"), 18.0)
        rh_error = _f(observed_state.get("indoor_rh_pct"), _f(pred.get("indoor_rh_pct"), 60.0)) - _f(pred.get("indoor_rh_pct"), 60.0)
        a = self.alpha
        self.model_bias_temp_c = (1.0 - a) * self.model_bias_temp_c + a * temp_error
        self.model_bias_rh_pct = (1.0 - a) * self.model_bias_rh_pct + a * rh_error
        cmd = self.last_command or {}
        vent = _f(cmd.get("ventilation_group_pct"), 0.0) / 100.0
        heat = _f(cmd.get("heating_group_pct"), 0.0) / 100.0
        # Signed lightweight identification gains. Keep bounded for edge stability.
        self.theta["theta_vent"] = max(0.2, min(3.0, self.theta["theta_vent"] + a * (-temp_error) * vent * 0.01))
        self.theta["theta_heat"] = max(0.2, min(3.0, self.theta["theta_heat"] + a * (temp_error) * heat * 0.01))
        self.theta["theta_humidity"] = max(0.2, min(3.0, self.theta["theta_humidity"] + a * rh_error * 0.001))
        # Uncertainty shrinks with stable small residuals and expands for large mismatch.
        err_norm = abs(temp_error) / 5.0 + abs(rh_error) / 30.0
        self.uncertainty_radius = max(self.min_uncertainty, min(5.0, self.uncertainty_decay * self.uncertainty_radius + a * err_norm))
        self.updates += 1
        return EstimatorUpdate(
            timestamp=str(observed_state.get("timestamp")) if observed_state.get("timestamp") is not None else None,
            temp_error_c=float(temp_error),
            rh_error_pct=float(rh_error),
            model_bias_temp_c=float(self.model_bias_temp_c),
            model_bias_rh_pct=float(self.model_bias_rh_pct),
            theta_vent=float(self.theta["theta_vent"]),
            theta_heat=float(self.theta["theta_heat"]),
            theta_cattle=float(self.theta["theta_cattle"]),
            uncertainty_radius=float(self.uncertainty_radius),
        )

    def to_dict(self) -> dict:
        return {
            "algorithm": "edge_recursive_state_estimator",
            "formal_layer": "L2_online_state_estimation_and_system_identification",
            "alpha": self.alpha,
            "uncertainty_decay": self.uncertainty_decay,
            "min_uncertainty": self.min_uncertainty,
            "model_bias_temp_c": self.model_bias_temp_c,
            "model_bias_rh_pct": self.model_bias_rh_pct,
            "theta": self.theta,
            "uncertainty_radius": self.uncertainty_radius,
            "updates": self.updates,
        }

    @classmethod
    def from_file(cls, path: str | Path, **defaults) -> "EdgeStateEstimator":
        p = Path(path)
        obj = cls(**defaults)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                obj.alpha = float(data.get("alpha", obj.alpha))
                obj.uncertainty_decay = float(data.get("uncertainty_decay", obj.uncertainty_decay))
                obj.min_uncertainty = float(data.get("min_uncertainty", obj.min_uncertainty))
                obj.model_bias_temp_c = float(data.get("model_bias_temp_c", 0.0))
                obj.model_bias_rh_pct = float(data.get("model_bias_rh_pct", 0.0))
                obj.theta.update({str(k): float(v) for k, v in data.get("theta", {}).items()})
                obj.uncertainty_radius = float(data.get("uncertainty_radius", 1.0))
                obj.updates = int(data.get("updates", 0))
            except Exception:
                pass
        return obj

    def save(self, path: str | Path) -> str:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return str(p)
