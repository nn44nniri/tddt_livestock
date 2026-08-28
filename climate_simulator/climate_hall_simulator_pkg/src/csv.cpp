#include "csv.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace beefclimate {
namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields;
    std::string item;
    std::stringstream ss(line);
    while (std::getline(ss, item, ',')) fields.push_back(item);
    return fields;
}

int find_col(const std::vector<std::string>& header, const std::string& name) {
    const std::string target = lower(name);
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (lower(header[i]) == target) return static_cast<int>(i);
    }
    return -1;
}

double cell_as_double(const std::vector<std::string>& row, int idx, double fallback = 0.0) {
    if (idx < 0 || static_cast<std::size_t>(idx) >= row.size() || row[idx].empty()) return fallback;
    return std::stod(row[idx]);
}

int cell_as_int(const std::vector<std::string>& row, int idx, int fallback = 0) {
    if (idx < 0 || static_cast<std::size_t>(idx) >= row.size() || row[idx].empty()) return fallback;
    return std::stoi(row[idx]);
}

}  // namespace

DisturbanceSeries load_disturbances_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open disturbance csv: " + path);
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("Disturbance csv is empty: " + path);
    const auto header = split(line);
    const int ts_col = find_col(header, "timestamp");
    const int t_col = find_col(header, "outdoor_temp_c");
    const int rh_col = find_col(header, "outdoor_rh_pct");
    const int wind_col = find_col(header, "outdoor_wind_m_s");
    const int solar_col = find_col(header, "outdoor_solar_w_m2");
    const int cloud_col = find_col(header, "outdoor_cloud_okta");
    const int rain_col = find_col(header, "outdoor_rain_mm_day");
    const int out_co2_col = find_col(header, "outdoor_co2_ppm");
    const int out_nh3_col = find_col(header, "outdoor_nh3_ppm");
    const int out_h2o_col = find_col(header, "outdoor_h2o_g_m3");
    const int s_t_col = find_col(header, "sensor_indoor_temp_c");
    const int s_rh_col = find_col(header, "sensor_indoor_rh_pct");
    const int s_wind_col = find_col(header, "sensor_indoor_wind_m_s");
    const int s_co2_col = find_col(header, "sensor_indoor_co2_ppm");
    const int s_nh3_col = find_col(header, "sensor_indoor_nh3_ppm");
    const int s_h2o_col = find_col(header, "sensor_indoor_h2o_g_m3");
    const int s_rad_col = find_col(header, "sensor_indoor_rad_kj_m2_day");
    const int s_okta_col = find_col(header, "sensor_indoor_okta");
    const int s_aha_col = find_col(header, "sensor_indoor_aha");
    DisturbanceSeries rows;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = split(line);
        Disturbance d;
        d.timestamp = (ts_col >= 0 && static_cast<std::size_t>(ts_col) < f.size()) ? f[ts_col] : "step";
        d.outdoor_temp_c = cell_as_double(f, t_col, 0.0);
        d.outdoor_rh_pct = cell_as_double(f, rh_col, 70.0);
        d.outdoor_wind_m_s = cell_as_double(f, wind_col, 1.0);
        d.outdoor_solar_w_m2 = cell_as_double(f, solar_col, 0.0);
        d.outdoor_cloud_okta = cell_as_double(f, cloud_col, 4.0);
        d.outdoor_rain_mm_day = cell_as_double(f, rain_col, 0.0);
        d.outdoor_co2_ppm = cell_as_double(f, out_co2_col, 420.0);
        d.outdoor_nh3_ppm = cell_as_double(f, out_nh3_col, 0.2);
        d.outdoor_h2o_g_m3 = cell_as_double(f, out_h2o_col, -1.0);
        d.sensor_indoor_temp_c = cell_as_double(f, s_t_col, -999.0);
        d.sensor_indoor_rh_pct = cell_as_double(f, s_rh_col, -1.0);
        d.sensor_indoor_wind_m_s = cell_as_double(f, s_wind_col, -1.0);
        d.sensor_indoor_co2_ppm = cell_as_double(f, s_co2_col, -1.0);
        d.sensor_indoor_nh3_ppm = cell_as_double(f, s_nh3_col, -1.0);
        d.sensor_indoor_h2o_g_m3 = cell_as_double(f, s_h2o_col, -1.0);
        d.sensor_indoor_rad_kj_m2_day = cell_as_double(f, s_rad_col, -1.0);
        d.sensor_indoor_okta = cell_as_double(f, s_okta_col, -1.0);
        d.sensor_indoor_aha = cell_as_double(f, s_aha_col, -1.0);
        rows.push_back(d);
    }
    return rows;
}

ControlSeries load_controls_csv(const std::string& path, int fan_pair_count_hint, int heater_count_hint) {
    (void)fan_pair_count_hint;
    (void)heater_count_hint;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open control csv: " + path);
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("Control csv is empty: " + path);
    const auto header = split(line);
    const int ts_col = find_col(header, "timestamp");
    const int vent_col = find_col(header, "ventilation_group_pct");
    const int vent_legacy_col = find_col(header, "vent_pct");
    const int heat_col = find_col(header, "heating_group_pct");
    const int heat_legacy_col = find_col(header, "heat_pct");
    const int light_col = find_col(header, "light_on");

    std::vector<int> fan_cols;
    std::vector<int> heater_cols;
    for (std::size_t i = 0; i < header.size(); ++i) {
        const std::string h = lower(header[i]);
        if (h.rfind("fan_pair_", 0) == 0) fan_cols.push_back(static_cast<int>(i));
        if (h.rfind("heater_", 0) == 0) heater_cols.push_back(static_cast<int>(i));
    }

    auto average_cols = [](const std::vector<std::string>& row, const std::vector<int>& cols) {
        if (cols.empty()) return 0.0;
        double sum = 0.0;
        int count = 0;
        for (int col : cols) {
            if (col >= 0 && static_cast<std::size_t>(col) < row.size() && !row[col].empty()) {
                sum += std::stod(row[col]);
                ++count;
            }
        }
        return count > 0 ? sum / static_cast<double>(count) : 0.0;
    };

    ControlSeries rows;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = split(line);
        Control c;
        c.timestamp = (ts_col >= 0 && static_cast<std::size_t>(ts_col) < f.size()) ? f[ts_col] : "step";
        c.light_on = cell_as_int(f, light_col, 0) ? 1 : 0;
        if (vent_col >= 0) c.ventilation_group_pct = std::clamp(cell_as_double(f, vent_col, 0.0), 0.0, 100.0);
        else if (vent_legacy_col >= 0) c.ventilation_group_pct = std::clamp(cell_as_double(f, vent_legacy_col, 0.0), 0.0, 100.0);
        else if (!fan_cols.empty()) c.ventilation_group_pct = std::clamp(average_cols(f, fan_cols), 0.0, 100.0);

        if (heat_col >= 0) c.heating_group_pct = std::clamp(cell_as_double(f, heat_col, 0.0), 0.0, 100.0);
        else if (heat_legacy_col >= 0) c.heating_group_pct = std::clamp(cell_as_double(f, heat_legacy_col, 0.0), 0.0, 100.0);
        else if (!heater_cols.empty()) c.heating_group_pct = std::clamp(average_cols(f, heater_cols), 0.0, 100.0);
        rows.push_back(c);
    }
    return rows;
}

void save_results_csv(const SimulationHistory& history, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write results csv: " + path);
    out << "timestamp,outdoor_temp_c,indoor_temp_c,lct_c,uct_c,tnz_zone,outdoor_rh_pct,indoor_rh_pct,outdoor_wind_m_s,indoor_air_speed_m_s,outdoor_co2_ppm,indoor_co2_ppm,outdoor_nh3_ppm,indoor_nh3_ppm,outdoor_h2o_g_m3,indoor_h2o_g_m3,indoor_rad_kj_m2_day,indoor_okta,indoor_aha,outdoor_cloud_okta,outdoor_rain_mm_day,ventilation_group_pct,heating_group_pct,light_on,electric_kw,gas_kw,fan_power_w,heater_power_w,light_power_w,gas_index,reward,comfort_index,air_quality_penalty,cum_fan_kwh,cum_heater_kwh,cum_light_kwh\n";
    for (const auto& row : history) {
        const double electric_kw = (row.layers.fan_power_w + row.layers.light_power_w) / 1000.0;
        const double gas_kw = row.layers.heater_fuel_w / 1000.0;
        out << row.timestamp << ',' << row.disturbance.outdoor_temp_c << ',' << row.state.indoor_temp_c << ','
            << row.thermoregulation.lower_critical_c << ',' << row.thermoregulation.upper_critical_c << ',' << static_cast<int>(row.thermoregulation.zone) << ','
            << row.disturbance.outdoor_rh_pct << ',' << row.state.indoor_rh_pct << ','
            << row.disturbance.outdoor_wind_m_s << ',' << row.state.air_speed_m_s << ','
            << row.disturbance.outdoor_co2_ppm << ',' << row.state.co2_ppm << ','
            << row.disturbance.outdoor_nh3_ppm << ',' << row.state.nh3_ppm << ','
            << row.disturbance.outdoor_h2o_g_m3 << ',' << row.state.h2o_g_m3 << ','
            << row.state.indoor_rad_kj_m2_day << ',' << row.state.indoor_okta << ',' << row.state.indoor_aha << ','
            << row.disturbance.outdoor_cloud_okta << ',' << row.disturbance.outdoor_rain_mm_day << ','
            << row.control.average_fan_pair_pct() << ',' << row.control.average_heater_pct() << ',' << row.control.light_on
            << ',' << electric_kw << ',' << gas_kw << ',' << row.layers.fan_power_w << ','
            << row.layers.heater_fuel_w << ',' << row.layers.light_power_w << ',' << row.state.gas_index << ','
            << row.outputs.reward << ',' << row.outputs.comfort_violation << ',' << row.outputs.air_quality_violation << ','
            << row.state.cumulative_fan_energy_kwh << ',' << row.state.cumulative_heater_energy_kwh << ','
            << row.state.cumulative_light_energy_kwh << '\n';
    }
}

}  // namespace beefclimate
