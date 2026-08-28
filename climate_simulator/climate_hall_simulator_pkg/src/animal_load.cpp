#include "animal_load.hpp"

#include <algorithm>
#include <cmath>

namespace beefclimate {

LayerContributions compute_animal_loads(const HallConfig& cfg,
                                        const ThermoregulationState& thermo,
                                        double indoor_air_speed_m_s,
                                        double indoor_radiation_w) {
    LayerContributions layer;
    const double wt = std::max(50.0, cfg.average_weight_kg);
    const double area_m2 = 0.14 * std::pow(wt, 0.57) * 1.09;
    const double per_head_total_w = thermo.animal_heat_threshold_w_m2 * area_m2;

    double sensible_fraction = 0.72;
    if (thermo.zone == ThermalZoneClass::AboveTNZ) sensible_fraction = 0.50;
    if (thermo.zone == ThermalZoneClass::BelowTNZ) sensible_fraction = 0.82;
    sensible_fraction -= std::clamp(indoor_radiation_w / 60000.0, 0.0, 0.10);
    sensible_fraction = std::clamp(sensible_fraction, 0.35, 0.90);

    layer.cattle_sensible_w = cfg.cattle_count * per_head_total_w * sensible_fraction * cfg.theta_cattle;
    const double latent_w = cfg.cattle_count * per_head_total_w * (1.0 - sensible_fraction) * cfg.theta_humidity;
    layer.cattle_latent_kg_s = latent_w / 2.45e6;

    const double speed_gain = 1.0 + std::clamp(indoor_air_speed_m_s, 0.0, 3.0) * 0.05;
    const double heat_gain = 1.0 + std::max(0.0, thermo.heat_production_multiplier - 1.0) * 0.35;
    layer.cattle_co2_ppm_per_h = cfg.theta_gas * cfg.cattle_count * (7.5 + 0.010 * wt) * speed_gain * heat_gain;
    layer.cattle_nh3_ppm_per_h = cfg.theta_gas * cfg.cattle_count * (0.020 + 0.00004 * wt) * speed_gain;
    layer.cattle_h2o_g_m3_per_h = cfg.theta_humidity * layer.cattle_latent_kg_s * 1000.0 * 3600.0 / std::max(1.0, cfg.volume_m3);
    layer.cattle_gas_index_per_h = 0.010 * layer.cattle_co2_ppm_per_h + 2.0 * layer.cattle_nh3_ppm_per_h + 0.8 * layer.cattle_h2o_g_m3_per_h;
    return layer;
}

}  // namespace beefclimate
