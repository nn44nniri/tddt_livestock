#pragma once

#include "herd_inventory.hpp"
#include "types.hpp"

namespace beefclimate {

class ClimateSimulator {
  public:
    explicit ClimateSimulator(HallConfig cfg);

    // Formal interface aligned with SIMULATOR_OFFICIAL_SPECIFICATIONS.md
    State initialize();
    State reset();
    State reset(const State& initial_state);
    Observation observe(const Disturbance& disturbance, const Control& control) const;
    StepResult propagate(const Disturbance& disturbance, const Control& control, double dt_seconds);
    SimulationHistory rollout(const DisturbanceSeries& disturbances, const ControlSeries& controls, double dt_seconds);

    // Backward-compatible aliases.
    StepResult step(const Disturbance& disturbance, const Control& control, double dt_seconds) {
        return propagate(disturbance, control, dt_seconds);
    }
    SimulationHistory run(const DisturbanceSeries& disturbances, const ControlSeries& controls, double dt_seconds) {
        return rollout(disturbances, controls, dt_seconds);
    }

    const HallConfig& config() const { return cfg_; }
    void set_processed_herd(const HerdProcessedSummary& herd) { herd_ = herd; }

  private:
    HallConfig cfg_;
    HerdProcessedSummary herd_{};
    State state_{};

    double floor_area_m2() const;
    double roof_area_m2() const;
    double wall_area_m2() const;
    double total_ua_w_k() const;
    std::map<LayerId, std::string> summarize_layers(const StepResult& step) const;
};

}  // namespace beefclimate
