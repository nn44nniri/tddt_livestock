#pragma once

#include "types.hpp"

namespace beefclimate {

LayerContributions compute_animal_loads(const HallConfig& cfg,
                                        const ThermoregulationState& thermo,
                                        double indoor_air_speed_m_s,
                                        double indoor_radiation_w);

}  // namespace beefclimate
