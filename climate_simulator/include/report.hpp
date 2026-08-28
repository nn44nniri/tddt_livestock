#pragma once

#include "types.hpp"

#include <string>

namespace beefclimate {

void write_html_report(const SimulationHistory& history, const HallConfig& cfg, const std::string& path);

}  // namespace beefclimate
