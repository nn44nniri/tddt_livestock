#include "simulator.hpp"

#include "animal_load.hpp"
#include "psychrometrics.hpp"
#include "thermoregulation.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace beefclimate {
namespace {

double h2o_g_m3_from_trh(double temp_c, double rh_pct) {
    const double w = psychrometrics::humidity_ratio_from_rh(temp_c, rh_pct);
    const double rho = psychrometrics::air_density_kg_m3(temp_c);
    return std::max(0.0, rho * w * 1000.0);
}

double bounded_mix(double current, double outdoor, double generation_per_h, double removal_per_h, double dt_seconds) {
    return current + (generation_per_h - removal_per_h * (current - outdoor)) * dt_seconds / 3600.0;
}

double assimilate_sensor(double predicted, double measured, double alpha) {
    if (measured < 0.0 || measured < -900.0) return predicted;
    return (1.0 - alpha) * predicted + alpha * measured;
}


double body_surface_area_m2(double weight_kg) { return 0.14 * std::pow(std::max(50.0, weight_kg), 0.57) * 1.09; }
double body_projected_area_m2(double weight_kg) { return 0.055 * std::pow(std::max(50.0, weight_kg), 0.67); }
double roof_openness_fraction(const HallConfig& cfg, double floor_area) { const double ridge_open_area = std::max(0.0, cfg.ridge_opening_m) * cfg.length_m; return std::clamp(ridge_open_area / std::max(1.0, floor_area), 0.0, 1.0); }
double indoor_okta_surrogate(const HallConfig& cfg, double floor_area) { return std::clamp(8.0 * (1.0 - roof_openness_fraction(cfg, floor_area)), 0.0, 8.0); }
double indoor_aha_surrogate(const HallConfig& cfg) { const double body = body_surface_area_m2(cfg.average_weight_kg); const double proj = std::max(0.1, body_projected_area_m2(cfg.average_weight_kg)); return std::clamp(cfg.indoor_view_factor * body / proj, 0.0, 5.0); }
double indoor_rad_kj_m2_day_from_loads(const HallConfig& cfg, double floor_area, double outdoor_solar_w_m2, double light_radiant_w) { const double transmitting_area = cfg.effective_transmitting_area_m2 > 0.0 ? cfg.effective_transmitting_area_m2 : 0.12 * floor_area; const double solar_transmitted_w = cfg.envelope_solar_transmittance * outdoor_solar_w_m2 * transmitting_area; const double total_indoor_radiant_w = std::max(0.0, solar_transmitted_w + light_radiant_w); return total_indoor_radiant_w * 86.4 / std::max(1.0, floor_area); }

}  // namespace

ClimateSimulator::ClimateSimulator(HallConfig cfg) : cfg_(std::move(cfg)) { initialize(); }

State ClimateSimulator::initialize() { return reset(); }

State ClimateSimulator::reset() {
    state_ = {};
    state_.indoor_temp_c = cfg_.initial_indoor_temp_c;
    state_.indoor_rh_pct = cfg_.initial_indoor_rh_pct;
    state_.gas_index = cfg_.initial_gas_index;
    state_.air_speed_m_s = cfg_.initial_air_speed_m_s;
    state_.indoor_radiation_w = cfg_.initial_radiation_w;
    state_.mass_temperature_c = cfg_.initial_indoor_temp_c;
    state_.humidity_ratio_kgkg = psychrometrics::humidity_ratio_from_rh(state_.indoor_temp_c, state_.indoor_rh_pct);
    state_.co2_ppm = cfg_.initial_co2_ppm;
    state_.nh3_ppm = cfg_.initial_nh3_ppm;
    state_.h2o_g_m3 = cfg_.initial_h2o_g_m3;
    const double floor = floor_area_m2();
    state_.indoor_okta = indoor_okta_surrogate(cfg_, floor);
    state_.indoor_aha = indoor_aha_surrogate(cfg_);
    state_.indoor_rad_kj_m2_day = 0.0;
    return state_;
}

State ClimateSimulator::reset(const State& initial_state) {
    state_ = initial_state;
    state_.humidity_ratio_kgkg = psychrometrics::humidity_ratio_from_rh(state_.indoor_temp_c, state_.indoor_rh_pct);
    return state_;
}

Observation ClimateSimulator::observe(const Disturbance& d, const Control& u) const {
    Observation o;
    o.x = state_;
    o.u = u;
    o.d = d;
    o.y.indoor_temp_c = state_.indoor_temp_c;
    o.y.indoor_rh_pct = state_.indoor_rh_pct;
    o.y.gas_index = state_.gas_index;
    o.y.co2_ppm = state_.co2_ppm;
    o.y.nh3_ppm = state_.nh3_ppm;
    o.y.h2o_g_m3 = state_.h2o_g_m3;
    o.y.indoor_rad_kj_m2_day = state_.indoor_rad_kj_m2_day;
    o.y.indoor_okta = state_.indoor_okta;
    o.y.indoor_aha = state_.indoor_aha;
    o.y.air_speed_m_s = state_.air_speed_m_s;
    return o;
}

double ClimateSimulator::floor_area_m2() const { return cfg_.length_m * cfg_.width_m; }

double ClimateSimulator::roof_area_m2() const {
    const double rise = std::max(0.0, cfg_.ridge_height_m - cfg_.eave_height_m);
    const double half_span = cfg_.width_m / 2.0;
    const double slope_len = std::sqrt(half_span * half_span + rise * rise);
    return 2.0 * slope_len * cfg_.length_m;
}

double ClimateSimulator::wall_area_m2() const {
    const double rise = std::max(0.0, cfg_.ridge_height_m - cfg_.eave_height_m);
    const double base = 2.0 * cfg_.length_m * cfg_.eave_height_m + 2.0 * cfg_.width_m * cfg_.eave_height_m;
    const double gable = 2.0 * 0.5 * cfg_.width_m * rise;
    return base + gable;
}

double ClimateSimulator::total_ua_w_k() const {
    const double wall = wall_area_m2() * (cfg_.wall_u_w_m2k + cfg_.wall_leak_w_m2k);
    const double roof = roof_area_m2() * (cfg_.roof_u_w_m2k + cfg_.roof_leak_w_m2k);
    const double intake_area = cfg_.intake_count * cfg_.intake_width_m * cfg_.intake_height_m;
    const double intake = intake_area * cfg_.window_leak_w_m2k;
    const double service_doors_area = 4.0 * 4.5;
    const double doors = service_doors_area * (cfg_.service_door_u_w_m2k + cfg_.door_leak_w_m2k);
    return cfg_.theta_ua * (wall + roof + intake + doors);
}

StepResult ClimateSimulator::propagate(const Disturbance& d, const Control& u, double dt_seconds) {
    const double floor = floor_area_m2();
    const double rho = psychrometrics::air_density_kg_m3(state_.indoor_temp_c);
    const double cp = psychrometrics::moist_air_cp_j_kgk();
    const double intake_area_m2 = cfg_.intake_count * cfg_.intake_width_m * cfg_.intake_height_m;

    StepResult result;
    result.timestamp = d.timestamp;
    result.control = u;
    result.disturbance = d;

    if (!herd_.cohorts.empty()) {
        result.thermoregulation = evaluate_heterogeneous_thermoregulation(cfg_, herd_, state_.indoor_temp_c, state_.indoor_rh_pct,
                                                                          std::max(0.1, state_.air_speed_m_s),
                                                                          state_.indoor_rad_kj_m2_day,
                                                                          state_.indoor_okta, d.outdoor_rain_mm_day, state_.indoor_aha);
    } else {
        result.thermoregulation = evaluate_thermoregulation(cfg_, state_.indoor_temp_c, state_.indoor_rh_pct,
                                                            std::max(0.1, state_.air_speed_m_s),
                                                            state_.indoor_rad_kj_m2_day,
                                                            state_.indoor_okta, d.outdoor_rain_mm_day, state_.indoor_aha);
    }

    result.layers = compute_animal_loads(cfg_, result.thermoregulation, state_.air_speed_m_s, state_.indoor_radiation_w);
    state_.cattle_heat_w = result.layers.cattle_sensible_w;
    state_.generated_moisture_kg_s = result.layers.cattle_latent_kg_s;
    state_.generated_gas_index_per_h = result.layers.cattle_gas_index_per_h;

    const double vent_pct = std::clamp(u.ventilation_group_pct, 0.0, 100.0);
    const double heat_pct = std::clamp(u.heating_group_pct, 0.0, 100.0);

    result.layers.mechanical_flow_m3_s = cfg_.theta_vent * (vent_pct / 100.0) * cfg_.fan_count * cfg_.fan_flow_m3h_each / 3600.0;
    const double wind_driven = cfg_.intake_discharge_coeff * intake_area_m2 * std::sqrt(std::max(0.0, d.outdoor_wind_m_s * d.outdoor_wind_m_s + 0.15));
    result.layers.infiltration_flow_m3_s = 0.18 * wind_driven + 0.015 * cfg_.volume_m3 / 3600.0;
    result.layers.total_flow_m3_s = std::max(0.0, result.layers.mechanical_flow_m3_s + result.layers.infiltration_flow_m3_s);
    result.layers.air_speed_m_s = std::clamp(0.08 + 0.22 * result.layers.total_flow_m3_s / std::max(1.0, floor / 10.0), 0.02, 2.5);
    state_.air_speed_m_s = assimilate_sensor(result.layers.air_speed_m_s, d.sensor_indoor_wind_m_s, 0.35);

    result.layers.fan_power_w = cfg_.fan_count * cfg_.fan_power_w_each * std::pow(vent_pct / 100.0, 3.0);
    result.layers.heater_fuel_w = cfg_.heater_count * cfg_.heater_gas_input_kw_each * 1000.0 * (heat_pct / 100.0);
    result.layers.heater_useful_w = cfg_.theta_heat * cfg_.heater_count * cfg_.heater_useful_kw_each * 1000.0 * (heat_pct / 100.0);
    result.layers.light_power_w = cfg_.light_count * cfg_.light_power_w_each * (u.light_on ? 1.0 : 0.0);
    result.layers.light_radiant_w = cfg_.theta_light * result.layers.light_power_w * cfg_.light_visible_fraction;
    result.layers.light_convective_w = cfg_.theta_light * result.layers.light_power_w * (1.0 - cfg_.light_visible_fraction);
    state_.lamp_heat_w = result.layers.light_convective_w + result.layers.light_radiant_w;

    const double ua = total_ua_w_k();
    result.layers.envelope_loss_w = ua * (state_.indoor_temp_c - d.outdoor_temp_c);
    result.layers.ventilation_loss_w = rho * cp * result.layers.total_flow_m3_s * (state_.indoor_temp_c - d.outdoor_temp_c);
    result.layers.solar_gain_w = 0.08 * d.outdoor_solar_w_m2 * floor;

    const double q_internal = result.layers.cattle_sensible_w + result.layers.heater_useful_w +
                              result.layers.light_convective_w + result.layers.light_radiant_w + result.layers.solar_gain_w;
    state_.internal_heat_w = q_internal;
    state_.indoor_radiation_w = result.layers.light_radiant_w + 0.3 * result.layers.solar_gain_w;
    state_.indoor_okta = indoor_okta_surrogate(cfg_, floor);
    state_.indoor_aha = indoor_aha_surrogate(cfg_);
    state_.indoor_rad_kj_m2_day = indoor_rad_kj_m2_day_from_loads(cfg_, floor, d.outdoor_solar_w_m2, result.layers.light_radiant_w);
    state_.indoor_rad_kj_m2_day = assimilate_sensor(state_.indoor_rad_kj_m2_day, d.sensor_indoor_rad_kj_m2_day, 0.35);
    state_.indoor_okta = assimilate_sensor(state_.indoor_okta, d.sensor_indoor_okta, 0.35);
    state_.indoor_aha = assimilate_sensor(state_.indoor_aha, d.sensor_indoor_aha, 0.35);

    const double cap = cfg_.effective_thermal_mass_j_k * cfg_.theta_cap;
    const double q_mass_coupling = 0.08 * cap / std::max(1.0, dt_seconds) * (state_.mass_temperature_c - state_.indoor_temp_c);
    const double dT = (q_internal + q_mass_coupling - result.layers.envelope_loss_w - result.layers.ventilation_loss_w) * dt_seconds / std::max(1.0, cap);
    state_.indoor_temp_c = std::clamp(state_.indoor_temp_c + dT, -20.0, 60.0);
    state_.indoor_temp_c = assimilate_sensor(state_.indoor_temp_c, d.sensor_indoor_temp_c, 0.30);
    state_.mass_temperature_c += 0.05 * (state_.indoor_temp_c - state_.mass_temperature_c);

    const double outdoor_w = psychrometrics::humidity_ratio_from_rh(d.outdoor_temp_c, d.outdoor_rh_pct);
    const double air_mass_kg = std::max(1.0, rho * cfg_.volume_m3);
    const double mix_rate = std::clamp(result.layers.total_flow_m3_s * dt_seconds / std::max(1.0, cfg_.volume_m3), 0.0, 1.0);
    double humidity_ratio = state_.humidity_ratio_kgkg;
    humidity_ratio = (1.0 - mix_rate) * humidity_ratio + mix_rate * outdoor_w;
    humidity_ratio += result.layers.cattle_latent_kg_s * dt_seconds / air_mass_kg;
    humidity_ratio = std::clamp(humidity_ratio, 0.0005, 0.05);
    state_.humidity_ratio_kgkg = humidity_ratio;
    state_.indoor_rh_pct = psychrometrics::rh_from_humidity_ratio(state_.indoor_temp_c, humidity_ratio);
    state_.indoor_rh_pct = assimilate_sensor(state_.indoor_rh_pct, d.sensor_indoor_rh_pct, 0.25);
    state_.humidity_ratio_kgkg = psychrometrics::humidity_ratio_from_rh(state_.indoor_temp_c, state_.indoor_rh_pct);

    const double ach_removal_per_h = 0.25 * result.layers.total_flow_m3_s * 3600.0 / std::max(1.0, cfg_.volume_m3);
    const double outdoor_h2o = d.outdoor_h2o_g_m3 >= 0.0 ? d.outdoor_h2o_g_m3 : h2o_g_m3_from_trh(d.outdoor_temp_c, d.outdoor_rh_pct);
    state_.co2_ppm = std::clamp(bounded_mix(state_.co2_ppm, d.outdoor_co2_ppm, result.layers.cattle_co2_ppm_per_h, ach_removal_per_h, dt_seconds), 300.0, 15000.0);
    state_.nh3_ppm = std::clamp(bounded_mix(state_.nh3_ppm, d.outdoor_nh3_ppm, result.layers.cattle_nh3_ppm_per_h, ach_removal_per_h, dt_seconds), 0.0, 200.0);
    state_.h2o_g_m3 = std::clamp(bounded_mix(state_.h2o_g_m3, outdoor_h2o, result.layers.cattle_h2o_g_m3_per_h, ach_removal_per_h, dt_seconds), 0.1, 60.0);

    state_.co2_ppm = assimilate_sensor(state_.co2_ppm, d.sensor_indoor_co2_ppm, 0.35);
    state_.nh3_ppm = assimilate_sensor(state_.nh3_ppm, d.sensor_indoor_nh3_ppm, 0.35);
    state_.h2o_g_m3 = assimilate_sensor(state_.h2o_g_m3, d.sensor_indoor_h2o_g_m3, 0.35);

    const double co2_excess = std::max(0.0, state_.co2_ppm - 1500.0) / 500.0;
    const double nh3_excess = std::max(0.0, state_.nh3_ppm - 10.0) / 5.0;
    const double h2o_excess = std::max(0.0, state_.h2o_g_m3 - 18.0) / 4.0;
    state_.gas_index = std::clamp(25.0 * co2_excess + 40.0 * nh3_excess + 15.0 * h2o_excess, 0.0, 1000.0);

    state_.cumulative_fan_energy_kwh += result.layers.fan_power_w * dt_seconds / 3600000.0;
    state_.cumulative_heater_energy_kwh += result.layers.heater_fuel_w * dt_seconds / 3600000.0;
    state_.cumulative_light_energy_kwh += result.layers.light_power_w * dt_seconds / 3600000.0;

    if (!herd_.cohorts.empty()) {
        result.thermoregulation = evaluate_heterogeneous_thermoregulation(cfg_, herd_, state_.indoor_temp_c, state_.indoor_rh_pct,
                                                                          std::max(0.1, state_.air_speed_m_s),
                                                                          state_.indoor_rad_kj_m2_day,
                                                                          state_.indoor_okta, d.outdoor_rain_mm_day, state_.indoor_aha);
    } else {
        result.thermoregulation = evaluate_thermoregulation(cfg_, state_.indoor_temp_c, state_.indoor_rh_pct,
                                                            std::max(0.1, state_.air_speed_m_s),
                                                            state_.indoor_rad_kj_m2_day,
                                                            state_.indoor_okta, d.outdoor_rain_mm_day, state_.indoor_aha);
    }

    const double comfort_low = std::max(0.0, result.thermoregulation.lower_critical_c - state_.indoor_temp_c);
    const double comfort_high = std::max(0.0, state_.indoor_temp_c - result.thermoregulation.upper_critical_c);
    const double rh_penalty = std::max(0.0, state_.indoor_rh_pct - 85.0) + std::max(0.0, 45.0 - state_.indoor_rh_pct);
    const double gas_penalty = co2_excess + 2.0 * nh3_excess + 0.6 * h2o_excess;
    const double energy_penalty = (result.layers.fan_power_w + result.layers.heater_fuel_w + result.layers.light_power_w) / 10000.0;
    const double comfort_index = comfort_low + comfort_high + 0.05 * rh_penalty;
    const double reward = -(2.0 * comfort_index + gas_penalty + 0.25 * energy_penalty);

    result.state = state_;
    result.outputs.indoor_temp_c = state_.indoor_temp_c;
    result.outputs.indoor_rh_pct = state_.indoor_rh_pct;
    result.outputs.gas_index = state_.gas_index;
    result.outputs.co2_ppm = state_.co2_ppm;
    result.outputs.nh3_ppm = state_.nh3_ppm;
    result.outputs.h2o_g_m3 = state_.h2o_g_m3;
    result.outputs.indoor_rad_kj_m2_day = state_.indoor_rad_kj_m2_day;
    result.outputs.indoor_okta = state_.indoor_okta;
    result.outputs.indoor_aha = state_.indoor_aha;
    result.outputs.air_speed_m_s = state_.air_speed_m_s;
    result.outputs.fan_power_w = result.layers.fan_power_w;
    result.outputs.heater_power_w = result.layers.heater_fuel_w;
    result.outputs.light_power_w = result.layers.light_power_w;
    result.outputs.comfort_violation = comfort_index;
    result.outputs.air_quality_violation = gas_penalty;
    result.outputs.energy_cost = energy_penalty;
    result.outputs.reward = reward;
    result.done = false;
    result.layer_notes = summarize_layers(result);
    return result;
}

SimulationHistory ClimateSimulator::rollout(const DisturbanceSeries& disturbances, const ControlSeries& controls,
                                           double dt_seconds) {
    SimulationHistory history;
    const std::size_t steps = std::max(disturbances.size(), controls.size());
    history.reserve(steps);
    reset();

    if (steps == 0) return history;

    // Keep the simulator usable when the climate and control CSV files do not have
    // the same number of rows. The longer series defines the rollout horizon and the
    // last available value of the shorter series is held constant (zero-order hold).
    // This is important for grouped actuator traces: otherwise late control changes
    // such as 100% ventilation could be silently ignored when controls.csv has more
    // rows than outdoor.csv.
    for (std::size_t i = 0; i < steps; ++i) {
        Disturbance d = disturbances.empty() ? Disturbance{} : disturbances[std::min(i, disturbances.size() - 1)];
        Control ctrl = controls.empty() ? Control{} : controls[std::min(i, controls.size() - 1)];
        if (d.timestamp.empty()) d.timestamp = ctrl.timestamp;
        history.push_back(propagate(d, ctrl, dt_seconds));
    }
    return history;
}

std::map<LayerId, std::string> ClimateSimulator::summarize_layers(const StepResult& step) const {
    std::map<LayerId, std::string> notes;
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);

    ss << "BML | Building model: hall " << cfg_.length_m << "x" << cfg_.width_m << " m, volume " << cfg_.volume_m3
       << " m3, " << cfg_.cattle_count << " cattle, " << cfg_.fan_count << " fans, " << cfg_.heater_count
       << " heaters, " << cfg_.light_count << " lights.";
    notes[LayerId::BML] = ss.str(); ss.str(""); ss.clear();
    ss << "ETL | Envelope + inertia: UA=" << total_ua_w_k() << " W/K, mass=" << cfg_.effective_thermal_mass_j_k << " J/K.";
    notes[LayerId::ETL] = ss.str(); ss.str(""); ss.clear();
    ss << "OCL | Outdoor climate/sensors: T=" << step.disturbance.outdoor_temp_c << " C, RH=" << step.disturbance.outdoor_rh_pct
       << " %, wind=" << step.disturbance.outdoor_wind_m_s << " m/s, solar=" << step.disturbance.outdoor_solar_w_m2 << " W/m2, CO2=" << step.disturbance.outdoor_co2_ppm
       << " ppm, NH3=" << step.disturbance.outdoor_nh3_ppm << " ppm.";
    notes[LayerId::OCL] = ss.str(); ss.str(""); ss.clear();
    ss << "ALL | Animal load: sensible=" << step.layers.cattle_sensible_w << " W, latent=" << step.layers.cattle_latent_kg_s
       << " kg/s, CO2 gen=" << step.layers.cattle_co2_ppm_per_h << " ppm/h, NH3 gen=" << step.layers.cattle_nh3_ppm_per_h
       << " ppm/h, H2O gen=" << step.layers.cattle_h2o_g_m3_per_h << " g/m3/h.";
    notes[LayerId::ALL] = ss.str(); ss.str(""); ss.clear();
    ss << "AOL | Grouped actuators: vent=" << step.control.ventilation_group_pct << " %, heat=" << step.control.heating_group_pct
       << " %, light=" << step.control.light_on << ".";
    notes[LayerId::AOL] = ss.str(); ss.str(""); ss.clear();
    ss << "LPL | Physics: mech flow=" << step.layers.mechanical_flow_m3_s << " m3/s, infil=" << step.layers.infiltration_flow_m3_s
       << " m3/s, vent loss=" << step.layers.ventilation_loss_w << " W, envelope loss=" << step.layers.envelope_loss_w << " W.";
    notes[LayerId::LPL] = ss.str(); ss.str(""); ss.clear();
    ss << "ICS | State x_k: Tin=" << step.state.indoor_temp_c << " C, RHin=" << step.state.indoor_rh_pct << " %, CO2="
       << step.state.co2_ppm << " ppm, NH3=" << step.state.nh3_ppm << " ppm, H2O=" << step.state.h2o_g_m3
       << " g/m3, RAD=" << step.state.indoor_rad_kj_m2_day << " kJ/m2/day, OKTA=" << step.state.indoor_okta << ", AHA=" << step.state.indoor_aha << ", Gin=" << step.state.gas_index << ", Vin=" << step.state.air_speed_m_s << " m/s.";
    notes[LayerId::ICS] = ss.str(); ss.str(""); ss.clear();
    ss << "TEL | Time evolution: x(k+1)=f(x,u,d,p) advanced with dt and corrected by available indoor sensor measurements for T/RH/wind/CO2/NH3/H2O.";
    notes[LayerId::TEL] = ss.str(); ss.str(""); ss.clear();
    ss << "EML | Energy: fan=" << step.layers.fan_power_w << " W, heater fuel=" << step.layers.heater_fuel_w
       << " W, light=" << step.layers.light_power_w << " W.";
    notes[LayerId::EML] = ss.str(); ss.str(""); ss.clear();
    ss << "CCL | Constraints: TNZ [" << step.thermoregulation.lower_critical_c << ", " << step.thermoregulation.upper_critical_c
       << "] C, safe bounds [" << step.thermoregulation.herd_safe_lower_c << ", " << step.thermoregulation.herd_safe_upper_c
       << "] C across " << step.thermoregulation.herd_cohort_count << " cohorts, CO2/NH3/H2O bounded, RH bounded, energy non-negative.";
    notes[LayerId::CCL] = ss.str(); ss.str(""); ss.clear();
    ss << "CEL | Control evaluation: Jcomfort=" << step.outputs.comfort_violation << ", Jair=" << step.outputs.air_quality_violation
       << ", Jenergy=" << step.outputs.energy_cost << ", reward=" << step.outputs.reward << ".";
    notes[LayerId::CEL] = ss.str(); ss.str(""); ss.clear();
    ss << "CIL | Control interface: initialize/reset/observe/propagate/rollout methods expose RL/MPC-friendly interaction.";
    notes[LayerId::CIL] = ss.str(); ss.str(""); ss.clear();
    ss << "VML | Visualisation: CSV + optional SVG report provide monitoring for grouped layers, TNZ overlays, and harmful gas trends CO2/NH3/H2O plus indoor RAD/OKTA/AHA estimates.";
    notes[LayerId::VML] = ss.str();
    return notes;
}

}  // namespace beefclimate
