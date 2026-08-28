from __future__ import annotations
from dataclasses import dataclass
import math
from datetime import datetime
from typing import Any


def _f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


@dataclass
class MPCWeights:
    comfort_weight: float = 100.0
    energy_weight: float = 0.05
    gas_weight: float = 0.2
    context_weight: float = 15.0
    guidance_weight: float = 8.0
    switch_weight: float = 2.5
    max_delta_pct_per_step: float = 50.0
    oscillation_weight: float = 4.0
    heating_ventilation_conflict_weight: float = 25.0


class EconomicMPC:
    """Economic MPC with TDDT coupling and actuator anti-wear penalties.

    The cost terms are explicit so SETD-KStore can reconstruct why the chosen
    action was preferred:

        J = Wc Ec + We Ee + Wg Eg + Wctx Ectx + Wbio Ebio
            + Wsw Esw + Wosc Eosc + Wconf Econf + Wrl Erl

    * Esw penalizes large actuator jumps between consecutive 5-minute steps.
    * Eosc penalizes immediate reversals such as 0→100→25→100.
    * These terms are soft constraints; safety filtering remains a separate
      hard layer after MPC selection.
    * Econf penalizes anti-waste conflicts where high heating and high exhaust
      ventilation are requested at the same time.
    """

    def __init__(self, climate_adapter, ventilation_candidates, heating_candidates, weights: MPCWeights | None=None, rl_agent=None):
        self.climate = climate_adapter
        self.ventilation_candidates = list(ventilation_candidates)
        self.heating_candidates = list(heating_candidates)
        self.weights = weights or MPCWeights()
        self.rl_agent = rl_agent
        self.light_on_hour = 5
        self.light_on_minute = 0
        self.light_hours_on = 16.0
        self.light_hours_off = 8.0
        self.last_command: dict | None = None
        self.prev_command: dict | None = None

    def configure_light_schedule(self, on_hour: int = 5, on_minute: int = 0, hours_on: float = 16.0, hours_off: float | None = None) -> None:
        self.light_on_hour = int(max(0, min(23, on_hour)))
        self.light_on_minute = int(max(0, min(59, on_minute)))
        self.light_hours_on = float(max(0.0, min(24.0, hours_on)))
        self.light_hours_off = float(hours_off) if hours_off is not None else max(0.0, 24.0 - self.light_hours_on)

    def observe_command(self, command: dict) -> None:
        """Store the accepted/safety-filtered command for the next MPC step."""
        self.prev_command = self.last_command
        self.last_command = dict(command)

    def choose(self, state: dict, guidance: dict | None=None) -> tuple[dict, dict]:
        guidance = guidance or {}
        best_score = math.inf
        best_cmd, best_pred = None, None
        candidates_for_explore = []
        light_on = self._light_schedule(str(state.get("timestamp")))
        state_key = self.rl_agent.encode_state(state, guidance) if self.rl_agent else None
        for vent in self.ventilation_candidates:
            for heat in self.heating_candidates:
                cmd = {"ventilation_group_pct": float(vent), "heating_group_pct": float(heat), "light_on": light_on}
                action_key = self.rl_agent.encode_action(cmd) if self.rl_agent else None
                try:
                    pred = self.climate.simulate_step(state, cmd, write_report=False)
                    components = self._score_components(pred, guidance, cmd, state)
                    rl_prior_cost = self.rl_agent.action_prior_cost(state_key, action_key) if self.rl_agent else 0.0
                    score = components["weighted_total_without_rl"] + rl_prior_cost
                    pred.update({
                        "mpc_score": score,
                        "rl_prior_cost": rl_prior_cost,
                        "rl_state_key": state_key,
                        "rl_action_key": action_key,
                        "mpc_components": components,
                    })
                    candidates_for_explore.append({"cmd": cmd, "pred": pred, "score": score})
                except Exception as exc:
                    pred = {"error": str(exc)}
                    score = math.inf
                if score < best_score:
                    best_score, best_cmd, best_pred = score, cmd, pred
        # Optional epsilon exploration from edge RL; it is still passed through safety.
        if self.rl_agent:
            picked = self.rl_agent.maybe_explore(candidates_for_explore)
            if picked is not None:
                best_cmd, best_pred, best_score = picked["cmd"], picked["pred"], picked["score"]
                best_pred["rl_exploration"] = True
        best_cmd = best_cmd or {"ventilation_group_pct": 40.0, "heating_group_pct": 0.0, "light_on": light_on}
        best_pred = best_pred or {}
        best_pred["mpc_score"] = best_score
        return best_cmd, best_pred

    def _score_components(self, pred: dict, guidance: dict, cmd: dict, state: dict) -> dict:
        temp = _f(pred.get("indoor_temp_c"), 18.0)
        low = _f(guidance.get("preferred_temp_low_c"), _f(pred.get("lct_c", pred.get("safe_lct_c")), -5.0))
        high = _f(guidance.get("preferred_temp_high_c"), _f(pred.get("uct_c", pred.get("safe_uct_c")), 27.0))
        low = max(low, -8.0); high = min(high, 28.0)
        center = (low + high) / 2.0
        under = max(0.0, low - temp)
        over = max(0.0, temp - high)
        center_error = abs(temp - center) / max(high - low, 1.0)
        comfort_error = (under + over) ** 2 + 0.05 * center_error
        energy_cost = _f(pred.get("electric_kw"), 0.0) + _f(pred.get("gas_kw"), 0.0)
        gas_penalty = max(0.0, _f(pred.get("indoor_co2_ppm"), 0.0) - 1800.0) / 100.0 + max(0.0, _f(pred.get("indoor_nh3_ppm"), 0.0) - 20.0)
        context_penalty = self._context_penalty(guidance, cmd)
        guidance_penalty = self._growth_guidance_penalty(guidance, cmd, temp, low, high)
        # Robust-adaptive layer: uncertainty is a soft margin, not a hard stop.
        # It increases the cost of predictions close to comfort limits when the
        # estimator reports larger model uncertainty.
        uncertainty = _f(guidance.get("model_uncertainty_radius"), _f(pred.get("model_uncertainty_radius"), 0.0))
        robust_margin_penalty = 0.0
        if uncertainty > 0:
            distance_to_edge = min(abs(temp - low), abs(high - temp))
            robust_margin_penalty = max(0.0, uncertainty - distance_to_edge) ** 2 / max(high - low, 1.0)
        switch_penalty = self._switch_penalty(cmd)
        oscillation_penalty = self._oscillation_penalty(cmd)
        conflict_penalty = self._heating_ventilation_conflict_penalty(cmd)
        cw = self.weights.comfort_weight * _f(guidance.get("comfort_weight_bias"), 1.0)
        ew = self.weights.energy_weight * _f(guidance.get("energy_weight_bias"), 1.0)
        gw = self.weights.gas_weight * _f(guidance.get("gas_weight_bias"), 1.0)
        total = (
            cw * comfort_error
            + ew * energy_cost
            + gw * gas_penalty
            + self.weights.context_weight * context_penalty
            + self.weights.guidance_weight * (guidance_penalty + robust_margin_penalty)
            + self.weights.switch_weight * switch_penalty
            + self.weights.oscillation_weight * oscillation_penalty
            + self.weights.heating_ventilation_conflict_weight * conflict_penalty
        )
        return {
            "comfort_error": comfort_error,
            "energy_cost": energy_cost,
            "gas_penalty": gas_penalty,
            "context_penalty": context_penalty,
            "guidance_penalty": guidance_penalty,
            "robust_margin_penalty": robust_margin_penalty,
            "model_uncertainty_radius": uncertainty,
            "switch_penalty": switch_penalty,
            "oscillation_penalty": oscillation_penalty,
            "heating_ventilation_conflict_penalty": conflict_penalty,
            "comfort_weight_effective": cw,
            "energy_weight_effective": ew,
            "gas_weight_effective": gw,
            "switch_weight_effective": self.weights.switch_weight,
            "oscillation_weight_effective": self.weights.oscillation_weight,
            "heating_ventilation_conflict_weight_effective": self.weights.heating_ventilation_conflict_weight,
            "max_delta_pct_per_step": self.weights.max_delta_pct_per_step,
            "weighted_total_without_rl": total,
        }

    def _switch_penalty(self, cmd: dict) -> float:
        if not self.last_command:
            return 0.0
        max_delta = max(1.0, _f(self.weights.max_delta_pct_per_step, 50.0))
        total = 0.0
        for key in ("ventilation_group_pct", "heating_group_pct"):
            delta = abs(_f(cmd.get(key), 0.0) - _f(self.last_command.get(key), 0.0))
            # Smooth quadratic penalty, with a stronger term beyond the configured soft ramp.
            total += (delta / 100.0) ** 2
            if delta > max_delta:
                total += ((delta - max_delta) / 100.0) ** 2 * 4.0
        if bool(cmd.get("light_on")) != bool(self.last_command.get("light_on")):
            total += 0.10
        return total

    def _oscillation_penalty(self, cmd: dict) -> float:
        if not self.last_command or not self.prev_command:
            return 0.0
        total = 0.0
        for key in ("ventilation_group_pct", "heating_group_pct"):
            a = _f(self.prev_command.get(key), 0.0)
            b = _f(self.last_command.get(key), 0.0)
            c = _f(cmd.get(key), 0.0)
            prev_delta = b - a
            next_delta = c - b
            # Penalize immediate direction reversal and repeated large swings.
            if prev_delta * next_delta < 0:
                total += min(1.0, abs(prev_delta) / 100.0) * min(1.0, abs(next_delta) / 100.0)
            if max(abs(prev_delta), abs(next_delta)) >= 75.0:
                total += 0.25
        return total

    def _heating_ventilation_conflict_penalty(self, cmd: dict) -> float:
        """Soft anti-waste penalty for simultaneous heating and exhaust ventilation.

        The term follows the requested formalism:

            conflict_penalty = heating_pct/100 * ventilation_pct/100

        It is intentionally soft here, so low hygienic ventilation can still
        coexist with heating when needed. The hard interlock is implemented in
        safety_filter.apply_safety for extreme exhaust ventilation.
        """
        heat = _f(cmd.get("heating_group_pct"), 0.0) / 100.0
        vent = _f(cmd.get("ventilation_group_pct"), 0.0) / 100.0
        return max(0.0, min(1.0, heat)) * max(0.0, min(1.0, vent))

    def _context_penalty(self, guidance: dict, cmd: dict) -> float:
        prior = guidance.get("ccll_prior") or guidance.get("mpc_prior") or {}
        vent = _f(cmd.get("ventilation_group_pct"), 0.0) / 100.0
        heat = _f(cmd.get("heating_group_pct"), 0.0) / 100.0
        v_bias = _f(prior.get("ventilation_bias"), _f(guidance.get("ventilation_bias"), 0.0))
        h_bias = _f(prior.get("heating_bias"), _f(guidance.get("heating_bias"), 0.0))
        return (max(0.0, v_bias - vent) ** 2 + max(0.0, h_bias - heat) ** 2) * _f(prior.get("context_penalty_weight"), 0.20)

    def _growth_guidance_penalty(self, guidance: dict, cmd: dict, temp: float, low: float, high: float) -> float:
        vent = _f(cmd.get("ventilation_group_pct"), 0.0) / 100.0
        heat = _f(cmd.get("heating_group_pct"), 0.0) / 100.0
        v_bias = _f(guidance.get("ventilation_bias"), 0.0)
        h_bias = _f(guidance.get("heating_bias"), 0.0)
        temp_pos = str(guidance.get("preferred_temp_position", "center"))
        center = (low + high) / 2.0
        target = center
        if "upper" in temp_pos:
            target = low + 0.70 * (high - low)
        elif "lower" in temp_pos:
            target = low + 0.30 * (high - low)
        return 0.5 * (max(0.0, v_bias - vent) ** 2 + max(0.0, h_bias - heat) ** 2) + 0.05 * ((temp - target) / max(high - low, 1.0)) ** 2

    def _light_schedule(self, timestamp: str) -> bool:
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", ""))
            minute_of_day = dt.hour * 60 + dt.minute
        except Exception:
            minute_of_day = 12 * 60
        start = self.light_on_hour * 60 + self.light_on_minute
        duration = int(round(self.light_hours_on * 60.0))
        if duration <= 0:
            return False
        if duration >= 24 * 60:
            return True
        elapsed = (minute_of_day - start) % (24 * 60)
        return elapsed < duration
