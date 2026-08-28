#pragma once

#include <string>
#include <unordered_map>

namespace beefclimate {

struct ThermoregulationBreedParams {
    std::string name;
    int library_id = 3;
    double reflectance = 0.56;
    double coat_depth_m = 0.012;
    double area_factor = 1.09;
    double cbs_max = 64.1;
    double latent_a = 4.44;
    double latent_b = 1.03;
    double rbcsf = 1.0;
    double latent_ref_temp_c = 34.7;
    double exhaled_temp_factor = 0.93;
};

struct ThermoregulationModelConfig {
    double body_temp_c = 39.0;
    double lasmin_w_m2 = 10.0;
    double respiration_increase_factor = 7.64;
    double rainfrac = 0.3;
    double maintenance_me_coeff = 311.0;
    double metab_weight_factor = 0.9;
    double body_area_coeff = 0.14;
    double body_area_exp = 0.57;
    double hif = 0.30;
    double zc = 11000.0;
    double coat_const = 1.90e-5;
    double gamma = 66.0;
    double emissivity = 0.98;
    double schmidt = 0.61;
    double mu_st = 1.827e-5;
    double tr0 = 527.0;
    double cconv1 = 120.0;
    double refle_grass = 0.10;
    double refle_concrete = 0.50;
    std::unordered_map<std::string, ThermoregulationBreedParams> breeds;
};

ThermoregulationModelConfig default_thermoregulation_config();
ThermoregulationModelConfig load_thermoregulation_config_file(const std::string& path);
void save_thermoregulation_config_file(const ThermoregulationModelConfig& cfg, const std::string& path);
void set_active_thermoregulation_config(ThermoregulationModelConfig cfg);
const ThermoregulationModelConfig& active_thermoregulation_config();
const ThermoregulationBreedParams& resolve_thermoregulation_breed(const std::string& breed_name, int breed_library_hint);

} // namespace beefclimate
