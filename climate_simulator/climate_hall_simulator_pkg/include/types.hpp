#pragma once

#include <map>
#include <string>
#include <vector>

namespace beefclimate {

enum class LayerId {
    BML, ETL, OCL, ALL, AOL, LPL, ICS, TEL, EML, CCL, CEL, CIL, VML
};

struct HallConfig {
    double length_m = 48.0;
    double width_m = 35.0;
    double eave_height_m = 5.2;
    double ridge_height_m = 7.3;
    double ridge_opening_m = 0.0;
    double volume_m3 = 10500.0;
    int pen_count = 4;
    int cattle_per_pen = 30;
    int cattle_count = 120;
    double average_weight_kg = 450.0;

    double wall_u_w_m2k = 0.278;
    double roof_u_w_m2k = 0.222;
    double personnel_door_u_w_m2k = 0.556;
    double service_door_u_w_m2k = 0.625;
    double wall_leak_w_m2k = 0.5;
    double roof_leak_w_m2k = 0.4;
    double window_leak_w_m2k = 6.0;
    double door_leak_w_m2k = 8.0;
    double design_indoor_temp_c = 18.0;
    double design_outdoor_temp_c = -10.0;
    double design_delta_t_c = 28.0;
    double effective_thermal_mass_j_k = 5.5e8;

    int intake_count = 28;
    double intake_width_m = 1.2;
    double intake_height_m = 1.0;
    double intake_discharge_coeff = 0.62;
    double intake_center_z_m = 2.8;

    int fan_count = 28;
    double fan_power_w_each = 370.0;
    double fan_flow_m3h_each = 15700.0;
    double fan_free_air_flow_m3h_each = 17900.0;
    double fan_air_speed_mps_each = 6.63;
    double fan_center_z_m = 2.8;

    int heater_count = 6;
    double heater_gas_input_kw_each = 43.96;
    double heater_useful_kw_each = 35.17;
    double heater_airflow_m3h_each = 3704.0;
    double heater_center_z_m = 3.2;

    int light_count = 36;
    double light_power_w_each = 49.0;
    double light_luminous_flux_lm_each = 6000.0;
    double light_visible_fraction = 0.10;
    double light_longwave_fraction = 0.60;

    double theta_ua = 1.0;
    double theta_cap = 1.0;
    double theta_vent = 1.0;
    double theta_cattle = 1.0;
    double theta_humidity = 1.0;
    double theta_gas = 1.0;
    double theta_light = 1.0;
    double theta_heat = 1.0;

    double initial_indoor_temp_c = 16.0;
    double initial_indoor_rh_pct = 70.0;
    double initial_gas_index = 30.0;
    double initial_air_speed_m_s = 0.05;
    double initial_radiation_w = 0.0;
    double initial_co2_ppm = 900.0;
    double initial_nh3_ppm = 3.0;
    double initial_h2o_g_m3 = 9.0;
    double effective_transmitting_area_m2 = 0.0;
    double envelope_solar_transmittance = 0.15;
    double indoor_view_factor = 0.85;
};

struct Disturbance {
    std::string timestamp;
    double outdoor_temp_c = 0.0;
    double outdoor_rh_pct = 70.0;
    double outdoor_wind_m_s = 1.0;
    double outdoor_solar_w_m2 = 0.0;
    double outdoor_cloud_okta = 4.0;
    double outdoor_rain_mm_day = 0.0;
    double outdoor_co2_ppm = 420.0;
    double outdoor_nh3_ppm = 0.2;
    double outdoor_h2o_g_m3 = -1.0;
    double sensor_indoor_temp_c = -999.0;
    double sensor_indoor_rh_pct = -1.0;
    double sensor_indoor_wind_m_s = -1.0;
    double sensor_indoor_co2_ppm = -1.0;
    double sensor_indoor_nh3_ppm = -1.0;
    double sensor_indoor_h2o_g_m3 = -1.0;
    double sensor_indoor_rad_kj_m2_day = -1.0;
    double sensor_indoor_okta = -1.0;
    double sensor_indoor_aha = -1.0;
};

struct Control {
    std::string timestamp;
    double ventilation_group_pct = 0.0;
    double heating_group_pct = 0.0;
    int light_on = 0;
    double average_fan_pair_pct() const { return ventilation_group_pct; }
    double average_heater_pct() const { return heating_group_pct; }
};

enum class ThermalZoneClass { BelowTNZ = 0, InTNZ = 1, AboveTNZ = 2 };

struct ThermoregulationState {
    double lower_critical_c = -1.0;
    double upper_critical_c = 30.5;
    double herd_safe_lower_c = -1.0;
    double herd_safe_upper_c = 30.5;
    int herd_cohort_count = 0;
    double animal_heat_threshold_w_m2 = 0.0;
    double min_heat_release_w_m2 = 0.0;
    double max_heat_release_w_m2 = 0.0;
    double skin_temp_min_c = 0.0;
    double skin_temp_max_c = 0.0;
    double total_body_weight_kg = 450.0;
    double heat_production_multiplier = 1.36;
    ThermalZoneClass zone = ThermalZoneClass::InTNZ;
};

struct LayerContributions {
    double cattle_sensible_w = 0.0;
    double cattle_latent_kg_s = 0.0;
    double cattle_gas_index_per_h = 0.0;
    double cattle_co2_ppm_per_h = 0.0;
    double cattle_nh3_ppm_per_h = 0.0;
    double cattle_h2o_g_m3_per_h = 0.0;
    double heater_useful_w = 0.0;
    double heater_fuel_w = 0.0;
    double light_radiant_w = 0.0;
    double light_convective_w = 0.0;
    double envelope_loss_w = 0.0;
    double ventilation_loss_w = 0.0;
    double solar_gain_w = 0.0;
    double infiltration_flow_m3_s = 0.0;
    double mechanical_flow_m3_s = 0.0;
    double total_flow_m3_s = 0.0;
    double air_speed_m_s = 0.0;
    double fan_power_w = 0.0;
    double light_power_w = 0.0;
};

struct State {
    double indoor_temp_c = 16.0;
    double indoor_rh_pct = 70.0;
    double gas_index = 30.0;
    double air_speed_m_s = 0.05;
    double internal_heat_w = 0.0;
    double indoor_radiation_w = 0.0;
    double humidity_ratio_kgkg = 0.0;
    double mass_temperature_c = 16.0;
    double cattle_heat_w = 0.0;
    double lamp_heat_w = 0.0;
    double generated_moisture_kg_s = 0.0;
    double generated_gas_index_per_h = 0.0;
    double co2_ppm = 900.0;
    double nh3_ppm = 3.0;
    double h2o_g_m3 = 9.0;
    double indoor_rad_kj_m2_day = 0.0;
    double indoor_okta = 8.0;
    double indoor_aha = 0.0;
    double cumulative_fan_energy_kwh = 0.0;
    double cumulative_heater_energy_kwh = 0.0;
    double cumulative_light_energy_kwh = 0.0;
};

struct OutputMetrics {
    double indoor_temp_c = 0.0;
    double indoor_rh_pct = 0.0;
    double gas_index = 0.0;
    double co2_ppm = 0.0;
    double nh3_ppm = 0.0;
    double h2o_g_m3 = 0.0;
    double indoor_rad_kj_m2_day = 0.0;
    double indoor_okta = 0.0;
    double indoor_aha = 0.0;
    double air_speed_m_s = 0.0;
    double fan_power_w = 0.0;
    double heater_power_w = 0.0;
    double light_power_w = 0.0;
    double comfort_violation = 0.0;
    double air_quality_violation = 0.0;
    double energy_cost = 0.0;
    double reward = 0.0;
};

struct Observation { State x; OutputMetrics y; Control u; Disturbance d; };
struct StepResult {
    std::string timestamp;
    State state;
    Control control;
    Disturbance disturbance;
    LayerContributions layers;
    ThermoregulationState thermoregulation;
    OutputMetrics outputs;
    std::map<LayerId, std::string> layer_notes;
    bool done = false;
};

using DisturbanceSeries = std::vector<Disturbance>;
using ControlSeries = std::vector<Control>;
using SimulationHistory = std::vector<StepResult>;

}  // namespace beefclimate
