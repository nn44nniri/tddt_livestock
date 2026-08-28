#pragma once

#include "herd_inventory.hpp"
#include "thermoregulation_config.hpp"
#include "types.hpp"

namespace beefclimate {

// Adapted from the uploaded LiGAPS-Beef thermoregulation main.cpp.
ThermoregulationState evaluate_thermoregulation(const HallConfig& cfg,
                                                double ambient_temp_c,
                                                double relative_humidity_pct,
                                                double wind_m_s,
                                                double rad_kj_m2_day,
                                                double cloud_okta,
                                                double rain_mm_day,
                                                double aha,
                                                double housing_mode = 0.0,
                                                int breed_library = 3,
                                                double body_weight_kg = -1.0,
                                                double heat_multiplier = -1.0,
                                                const std::string& breed_name = "default");

ThermoregulationState evaluate_heterogeneous_thermoregulation(const HallConfig& cfg,
                                                              const HerdProcessedSummary& herd,
                                                              double ambient_temp_c,
                                                              double relative_humidity_pct,
                                                              double wind_m_s,
                                                              double rad_kj_m2_day,
                                                              double cloud_okta,
                                                              double rain_mm_day,
                                                              double aha,
                                                              double housing_mode = 0.0);

}  // namespace beefclimate
