#include "config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace beefclimate {
namespace {

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

std::unordered_map<std::string, std::string> read_kv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open config file: " + path);
    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';' || line[0] == '[') continue;
        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        kv[trim(line.substr(0, pos))] = trim(line.substr(pos + 1));
    }
    return kv;
}

template <typename T>
void assign_num(const std::unordered_map<std::string, std::string>& kv, const std::string& key, T& target) {
    const auto it = kv.find(key);
    if (it == kv.end()) return;
    std::istringstream ss(it->second);
    ss >> target;
}

}  // namespace

HallConfig load_config_file(const std::string& path) {
    HallConfig cfg;
    const auto kv = read_kv(path);

    assign_num(kv, "hall.length_m", cfg.length_m);
    assign_num(kv, "hall.width_m", cfg.width_m);
    assign_num(kv, "hall.eave_height_m", cfg.eave_height_m);
    assign_num(kv, "hall.ridge_height_m", cfg.ridge_height_m);
    assign_num(kv, "hall.volume_m3", cfg.volume_m3);
    assign_num(kv, "hall.pen_count", cfg.pen_count);
    assign_num(kv, "hall.cattle_per_pen", cfg.cattle_per_pen);

    assign_num(kv, "envelope.wall_u_w_m2k", cfg.wall_u_w_m2k);
    assign_num(kv, "envelope.roof_u_w_m2k", cfg.roof_u_w_m2k);
    assign_num(kv, "envelope.personnel_door_u_w_m2k", cfg.personnel_door_u_w_m2k);
    assign_num(kv, "envelope.service_door_u_w_m2k", cfg.service_door_u_w_m2k);
    assign_num(kv, "envelope.wall_leak_w_m2k", cfg.wall_leak_w_m2k);
    assign_num(kv, "envelope.roof_leak_w_m2k", cfg.roof_leak_w_m2k);
    assign_num(kv, "envelope.window_leak_w_m2k", cfg.window_leak_w_m2k);
    assign_num(kv, "envelope.door_leak_w_m2k", cfg.door_leak_w_m2k);

    assign_num(kv, "actuator.intake_count", cfg.intake_count);
    assign_num(kv, "actuator.intake_width_m", cfg.intake_width_m);
    assign_num(kv, "actuator.intake_height_m", cfg.intake_height_m);
    assign_num(kv, "actuator.intake_discharge_coeff", cfg.intake_discharge_coeff);

    assign_num(kv, "actuator.fan_count", cfg.fan_count);
    assign_num(kv, "actuator.fan_power_w_each", cfg.fan_power_w_each);
    assign_num(kv, "actuator.fan_flow_m3h_each", cfg.fan_flow_m3h_each);

    assign_num(kv, "actuator.heater_count", cfg.heater_count);
    assign_num(kv, "actuator.heater_gas_input_kw_each", cfg.heater_gas_input_kw_each);
    assign_num(kv, "actuator.heater_useful_kw_each", cfg.heater_useful_kw_each);
    assign_num(kv, "actuator.heater_airflow_m3h_each", cfg.heater_airflow_m3h_each);

    assign_num(kv, "actuator.light_count", cfg.light_count);
    assign_num(kv, "actuator.light_power_w_each", cfg.light_power_w_each);
    assign_num(kv, "actuator.light_luminous_flux_lm_each", cfg.light_luminous_flux_lm_each);
    assign_num(kv, "actuator.light_visible_fraction", cfg.light_visible_fraction);
    assign_num(kv, "actuator.light_longwave_fraction", cfg.light_longwave_fraction);

    assign_num(kv, "cattle.count", cfg.cattle_count);
    assign_num(kv, "cattle.average_weight_kg", cfg.average_weight_kg);

    assign_num(kv, "calibration.theta_ua", cfg.theta_ua);
    assign_num(kv, "calibration.theta_cap", cfg.theta_cap);
    assign_num(kv, "calibration.theta_vent", cfg.theta_vent);
    assign_num(kv, "calibration.theta_cattle", cfg.theta_cattle);
    assign_num(kv, "calibration.theta_humidity", cfg.theta_humidity);
    assign_num(kv, "calibration.theta_gas", cfg.theta_gas);
    assign_num(kv, "calibration.theta_light", cfg.theta_light);
    assign_num(kv, "calibration.theta_heat", cfg.theta_heat);
    assign_num(kv, "calibration.effective_thermal_mass_j_k", cfg.effective_thermal_mass_j_k);

    assign_num(kv, "initial.indoor_temp_c", cfg.initial_indoor_temp_c);
    assign_num(kv, "initial.indoor_rh_pct", cfg.initial_indoor_rh_pct);
    assign_num(kv, "initial.gas_index", cfg.initial_gas_index);
    assign_num(kv, "initial.air_speed_m_s", cfg.initial_air_speed_m_s);
    assign_num(kv, "initial.radiation_w", cfg.initial_radiation_w);

    if (cfg.cattle_count <= 0) cfg.cattle_count = cfg.pen_count * cfg.cattle_per_pen;
    return cfg;
}

void save_config_file(const HallConfig& cfg, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write config file: " + path);
    out << "# Generated simulator configuration\n";
    out << "hall.length_m=" << cfg.length_m << '\n';
    out << "hall.width_m=" << cfg.width_m << '\n';
    out << "hall.eave_height_m=" << cfg.eave_height_m << '\n';
    out << "hall.ridge_height_m=" << cfg.ridge_height_m << '\n';
    out << "hall.volume_m3=" << cfg.volume_m3 << '\n';
    out << "hall.pen_count=" << cfg.pen_count << '\n';
    out << "hall.cattle_per_pen=" << cfg.cattle_per_pen << '\n';

    out << "envelope.wall_u_w_m2k=" << cfg.wall_u_w_m2k << '\n';
    out << "envelope.roof_u_w_m2k=" << cfg.roof_u_w_m2k << '\n';
    out << "envelope.personnel_door_u_w_m2k=" << cfg.personnel_door_u_w_m2k << '\n';
    out << "envelope.service_door_u_w_m2k=" << cfg.service_door_u_w_m2k << '\n';
    out << "envelope.wall_leak_w_m2k=" << cfg.wall_leak_w_m2k << '\n';
    out << "envelope.roof_leak_w_m2k=" << cfg.roof_leak_w_m2k << '\n';
    out << "envelope.window_leak_w_m2k=" << cfg.window_leak_w_m2k << '\n';
    out << "envelope.door_leak_w_m2k=" << cfg.door_leak_w_m2k << '\n';

    out << "actuator.intake_count=" << cfg.intake_count << '\n';
    out << "actuator.intake_width_m=" << cfg.intake_width_m << '\n';
    out << "actuator.intake_height_m=" << cfg.intake_height_m << '\n';
    out << "actuator.intake_discharge_coeff=" << cfg.intake_discharge_coeff << '\n';
    out << "actuator.fan_count=" << cfg.fan_count << '\n';
    out << "actuator.fan_power_w_each=" << cfg.fan_power_w_each << '\n';
    out << "actuator.fan_flow_m3h_each=" << cfg.fan_flow_m3h_each << '\n';
    out << "actuator.heater_count=" << cfg.heater_count << '\n';
    out << "actuator.heater_gas_input_kw_each=" << cfg.heater_gas_input_kw_each << '\n';
    out << "actuator.heater_useful_kw_each=" << cfg.heater_useful_kw_each << '\n';
    out << "actuator.heater_airflow_m3h_each=" << cfg.heater_airflow_m3h_each << '\n';
    out << "actuator.light_count=" << cfg.light_count << '\n';
    out << "actuator.light_power_w_each=" << cfg.light_power_w_each << '\n';
    out << "actuator.light_luminous_flux_lm_each=" << cfg.light_luminous_flux_lm_each << '\n';
    out << "actuator.light_visible_fraction=" << cfg.light_visible_fraction << '\n';
    out << "actuator.light_longwave_fraction=" << cfg.light_longwave_fraction << '\n';

    out << "cattle.count=" << cfg.cattle_count << '\n';
    out << "cattle.average_weight_kg=" << cfg.average_weight_kg << '\n';

    out << "calibration.theta_ua=" << cfg.theta_ua << '\n';
    out << "calibration.theta_cap=" << cfg.theta_cap << '\n';
    out << "calibration.theta_vent=" << cfg.theta_vent << '\n';
    out << "calibration.theta_cattle=" << cfg.theta_cattle << '\n';
    out << "calibration.theta_humidity=" << cfg.theta_humidity << '\n';
    out << "calibration.theta_gas=" << cfg.theta_gas << '\n';
    out << "calibration.theta_light=" << cfg.theta_light << '\n';
    out << "calibration.theta_heat=" << cfg.theta_heat << '\n';
    out << "calibration.effective_thermal_mass_j_k=" << cfg.effective_thermal_mass_j_k << '\n';

    out << "initial.indoor_temp_c=" << cfg.initial_indoor_temp_c << '\n';
    out << "initial.indoor_rh_pct=" << cfg.initial_indoor_rh_pct << '\n';
    out << "initial.gas_index=" << cfg.initial_gas_index << '\n';
    out << "initial.air_speed_m_s=" << cfg.initial_air_speed_m_s << '\n';
    out << "initial.radiation_w=" << cfg.initial_radiation_w << '\n';
}

}  // namespace beefclimate
