#pragma once

#include "types.hpp"

#include <string>

namespace beefclimate {

DisturbanceSeries load_disturbances_csv(const std::string& path);
ControlSeries load_controls_csv(const std::string& path, int fan_pair_count_hint = 0, int heater_count_hint = 0);
void save_results_csv(const SimulationHistory& history, const std::string& path);

}  // namespace beefclimate
