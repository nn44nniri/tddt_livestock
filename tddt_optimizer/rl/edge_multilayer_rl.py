from __future__ import annotations
from pathlib import Path
import json, math, random
from dataclasses import dataclass, asdict
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
class RLUpdate:
    state_key: str
    action_key: str
    reward: float
    old_q: float
    new_q: float
    td_error: float
    layer: str = "climate_mpc_bias"


class EdgeMultiLayerRL:
    """Lightweight multi-layer RL for edge execution.

    The implementation is deliberately small: a tabular/EMA action-value memory
    with three explicit layers:
      1) context encoder: continuous state -> compact key,
      2) policy memory: Q(context, discrete actuator candidate),
      3) MPC bias adapter: Q is injected as a soft cost prior, not a hard rule.

    It is designed for one long single-episode run. The value memory is updated
    online from the actually selected MPC action, and can be persisted in
    SETD-KStore for WORK-ONLINE warm-start.
    """

    def __init__(self, alpha: float = 0.08, gamma: float = 0.90, epsilon: float = 0.03, rl_weight: float = 10.0, seed: int = 42):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.rl_weight = float(rl_weight)
        self.rng = random.Random(seed)
        self.q: dict[str, float] = {}
        self.visit: dict[str, int] = {}

    def encode_state(self, state: dict, guidance: dict | None = None) -> str:
        guidance = guidance or {}
        ctx = str(state.get("climate_context_id") or guidance.get("climate_context_id") or "C_UNKNOWN")
        phase = str(guidance.get("biological_phase") or guidance.get("phase") or "P_UNKNOWN")
        temp = _f(state.get("indoor_temp_c"), 18.0)
        low = _f(guidance.get("preferred_temp_low_c"), _f(state.get("lct_c"), -5.0))
        high = _f(guidance.get("preferred_temp_high_c"), _f(state.get("uct_c"), 27.0))
        if temp < low:
            band = "below"
        elif temp > high:
            band = "above"
        else:
            # keep center information; it helps avoid unnecessary heating/cooling.
            center = (low + high) / 2.0
            band = "inside_low" if temp < center else "inside_high"
        rh = _f(state.get("indoor_rh_pct"), 60.0)
        hum = "humid" if rh > 75 else ("dry" if rh < 40 else "normal")
        return f"ctx={ctx}|phase={phase}|band={band}|hum={hum}"

    def encode_action(self, command: dict) -> str:
        vent = int(round(_f(command.get("ventilation_group_pct"), 0.0) / 5.0) * 5)
        heat = int(round(_f(command.get("heating_group_pct"), 0.0) / 5.0) * 5)
        light = 1 if bool(command.get("light_on")) else 0
        return f"V{vent:03d}|H{heat:03d}|L{light}"

    def action_prior_cost(self, state_key: str, action_key: str) -> float:
        # High Q should reduce MPC cost. Unknown actions get zero bias.
        return -self.rl_weight * self.q.get(f"{state_key}::{action_key}", 0.0)

    def maybe_explore(self, candidates: list[dict]) -> dict | None:
        if not candidates or self.rng.random() >= self.epsilon:
            return None
        return self.rng.choice(candidates)

    def reward_from_components(self, components: dict) -> float:
        comfort = _f(components.get("comfort_error"), 0.0)
        energy = _f(components.get("energy_cost"), 0.0)
        gas = _f(components.get("gas_penalty"), 0.0)
        context = _f(components.get("context_penalty"), 0.0)
        guidance = _f(components.get("guidance_penalty"), 0.0)
        # Bounded negative cost reward. Edge-friendly and stable over long runs.
        total = comfort + 0.02 * energy + gas + context + guidance
        return 1.0 / (1.0 + total)

    def update(self, state_key: str, action_key: str, reward: float, next_state_key: str | None = None) -> RLUpdate:
        key = f"{state_key}::{action_key}"
        old = self.q.get(key, 0.0)
        # One-step bootstrap from best known action in next encoded state.
        next_best = 0.0
        if next_state_key:
            prefix = f"{next_state_key}::"
            vals = [v for k, v in self.q.items() if k.startswith(prefix)]
            next_best = max(vals) if vals else 0.0
        target = float(reward) + self.gamma * next_best
        new = old + self.alpha * (target - old)
        self.q[key] = new
        self.visit[key] = self.visit.get(key, 0) + 1
        return RLUpdate(state_key=state_key, action_key=action_key, reward=float(reward), old_q=old, new_q=new, td_error=target - old)

    def to_dict(self) -> dict:
        return {"algorithm": "edge_tabular_multilayer_rl", "alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon, "rl_weight": self.rl_weight, "q": self.q, "visit": self.visit}

    @classmethod
    def from_file(cls, path: str | Path, **defaults) -> "EdgeMultiLayerRL":
        p = Path(path)
        obj = cls(**defaults)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            obj.alpha = float(data.get("alpha", obj.alpha)); obj.gamma = float(data.get("gamma", obj.gamma)); obj.epsilon = float(data.get("epsilon", obj.epsilon)); obj.rl_weight = float(data.get("rl_weight", obj.rl_weight))
            obj.q = {str(k): float(v) for k, v in data.get("q", {}).items()}
            obj.visit = {str(k): int(v) for k, v in data.get("visit", {}).items()}
        return obj

    def save(self, path: str | Path) -> str:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return str(p)
