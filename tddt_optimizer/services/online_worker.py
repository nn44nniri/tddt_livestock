from __future__ import annotations
from pathlib import Path
import time
from funnel.funnel_protocol import FunnelBase
from ..adapters.climate_adapter import ClimateAdapter
from ..optimizer.economic_mpc import EconomicMPC, MPCWeights
from ..optimizer.safety_filter import apply_safety
from ..rl.edge_multilayer_rl import EdgeMultiLayerRL
from ..rl.contextual_tuner import EdgeContextualBayesianTuner
from ..estimation.state_estimator import EdgeStateEstimator


class OnlineWorker:
    """WORK-ONLINE edge loop.

    This implementation loads learned SETD/SS-KStore states and updates the
    small online learning layers incrementally. It avoids writing high-frequency
    traces by default; the edge memory is updated in compact JSON checkpoints.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.project_root = Path(cfg.project_root)
        self.funnel = FunnelBase()
        self.climate = ClimateAdapter(self.project_root, tmp_dir=getattr(cfg, "tmp_dir", "/tmp/tddt_livestock/"), cleanup_interval_steps=getattr(cfg, "tmp_cleanup_interval_steps", 4000))
        self.rl = EdgeMultiLayerRL.from_file(cfg.rl_state_path, alpha=cfg.rl_alpha, gamma=cfg.rl_gamma, epsilon=0.0, rl_weight=cfg.rl_weight) if getattr(cfg, "rl_enabled", True) else None
        self.bayes = EdgeContextualBayesianTuner.from_file(
            getattr(cfg, "bayesian_tuner_path", Path(cfg.setd_kstore_dir) / "edge_contextual_bayesian_tuner.json"),
            alpha=getattr(cfg, "bayesian_tuner_alpha", 0.05),
            exploration_weight=getattr(cfg, "bayesian_tuner_exploration_weight", 0.15),
            gaussian_alpha=getattr(cfg, "gaussian_tuner_alpha", 0.025),
            gaussian_tau=getattr(cfg, "gaussian_tuner_tau", 7.0),
            safe_blend_max=getattr(cfg, "gaussian_tuner_safe_blend_max", 0.85),
        ) if getattr(cfg, "bayesian_tuner_enabled", True) else None
        self.estimator = EdgeStateEstimator.from_file(getattr(cfg, "state_estimator_path", Path(cfg.setd_kstore_dir) / "edge_state_estimator.json"), alpha=getattr(cfg, "state_estimator_alpha", 0.025), uncertainty_decay=getattr(cfg, "state_estimator_uncertainty_decay", 0.995)) if getattr(cfg, "state_estimator_enabled", True) else None
        self.mpc = EconomicMPC(
            self.climate, cfg.candidate_ventilation, cfg.candidate_heating,
            MPCWeights(cfg.comfort_weight, cfg.energy_weight, cfg.gas_weight, cfg.mpc_context_weight, cfg.mpc_guidance_weight, getattr(cfg, "mpc_switch_weight", 2.5), getattr(cfg, "mpc_max_delta_pct_per_step", 50.0), getattr(cfg, "mpc_oscillation_weight", 4.0), getattr(cfg, "mpc_heating_ventilation_conflict_weight", 25.0)),
            rl_agent=self.rl,
        )

    def run(self, growth_start_date: str, max_steps: int | None=None) -> dict:
        steps = max(1, int(max_steps or 1))
        last = None
        for i in range(steps):
            packet = self.funnel.read_sensor_packet() or {}
            if self.estimator:
                self.estimator.update_from_observation(packet)
                packet = self.estimator.enrich_state(packet)
            guidance = {"preferred_temp_low_c": packet.get("lct_c", 5.0), "preferred_temp_high_c": packet.get("uct_c", 24.0), "status": "WORK_ONLINE"}
            if self.estimator:
                adj = self.estimator.guidance_adjustment()
                guidance.update({k: v for k, v in adj.items() if k in {"model_uncertainty_radius", "estimator_theta"}})
                guidance["comfort_weight_bias"] = adj.get("comfort_weight_bias", 1.0)
            if self.bayes:
                guidance = self.bayes.apply_to_guidance(packet, guidance)
            command, pred = self.mpc.choose(packet, guidance)
            command, safety = apply_safety(command)
            self.mpc.observe_command(command)
            ack = self.funnel.send_command({**command, "mode": "WORK-ONLINE", "safety_status": safety})
            # If simulator prediction exists, use it as the next estimator prior.
            if self.estimator:
                self.estimator.remember_prediction(pred, command)
            if self.rl and isinstance(pred, dict):
                comps = pred.get("mpc_components", {})
                reward = self.rl.reward_from_components(comps)
                state_key = pred.get("rl_state_key") or self.rl.encode_state(packet, guidance)
                action_key = pred.get("rl_action_key") or self.rl.encode_action(command)
                self.rl.update(state_key, action_key, reward)
                if self.bayes:
                    self.bayes.update(reward, guidance.get("bayesian_context_key"), guidance.get("bayesian_profile"))
            last = {"step": i + 1, "packet": packet, "command": command, "safety_status": safety, "ack": ack, "guidance": guidance, "prediction": pred}
            time.sleep(0.01)
        if self.rl: self.rl.save(self.cfg.rl_state_path)
        if self.bayes: self.bayes.save(getattr(self.cfg, "bayesian_tuner_path", Path(self.cfg.setd_kstore_dir) / "edge_contextual_bayesian_tuner.json"))
        if self.estimator: self.estimator.save(getattr(self.cfg, "state_estimator_path", Path(self.cfg.setd_kstore_dir) / "edge_state_estimator.json"))
        return {"mode": "WORK-ONLINE", "growth_start_date": growth_start_date, "steps": steps, "last": last, "status": "ONLINE_POLICY_UPDATE_READY"}
