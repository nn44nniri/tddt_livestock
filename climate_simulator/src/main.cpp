#include "config.hpp"
#include "csv.hpp"
#include "ifc_reader.hpp"
#include "herd_inventory.hpp"
#include "thermoregulation_config.hpp"
#include "report.hpp"
#include "simulator.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace beefclimate;

namespace {

std::vector<std::string> split_csv_list(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(item);
    return out;
}

double parse_group_command(const std::string& s) {
    double sum = 0.0;
    int count = 0;
    for (const auto& item : split_csv_list(s)) {
        if (!item.empty()) {
            sum += std::stod(item);
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  beef_climate_sim [--config path | --ifc path] --outdoor outdoor.csv --controls controls.csv --output outdir [--dt-seconds 300] [--write-report]\n"
        << "  beef_climate_sim [--config path | --ifc path] --timestamp 2026-01-01T00:00:00 --outdoor-temp-c 0 --outdoor-rh-pct 75 --outdoor-wind-m-s 1.5 --ventilation-group 25 --heating-group 40 --light-on 1 --output outdir [--write-report]\n"
        << "  beef_climate_sim [--config path | --ifc path] --timestamp 2026-01-01T00:00:00 --sensor-indoor-temp-c 18.4 --sensor-indoor-rh-pct 67 --outdoor-temp-c -6 --outdoor-rh-pct 78 --outdoor-wind-m-s 3.2 --ventilation-group 40 --heating-group 55 --light-on 1 --forecast-horizon-seconds 7200 --dt-seconds 300 [--response-stdout --response-format json]\n\n"
        << "Defaults:\n"
        << "  --ifc Building_Information/beef_hall_120.ifc\n"
        << "  HTML report is optional and generated with --write-report or legacy --write-svg.\n";
}

std::string resolve_ifc_path(std::string path, const char* argv0) {
    if (std::filesystem::exists(path)) return path;
    const std::vector<std::string> candidates = {
        (std::filesystem::path("..") / path).string(),
        (std::filesystem::path(argv0).parent_path() / path).string(),
        (std::filesystem::path(argv0).parent_path() / ".." / path).string()
    };
    for (const auto& c : candidates) {
        if (std::filesystem::exists(c)) return c;
    }
    return path;
}


std::string resolve_support_path(std::string path, const char* argv0, const std::string& config_path) {
    if (std::filesystem::exists(path)) return path;
    std::vector<std::string> candidates;
    if (!config_path.empty()) candidates.push_back((std::filesystem::path(config_path).parent_path() / path).string());
    candidates.push_back((std::filesystem::path("..") / path).string());
    candidates.push_back((std::filesystem::path(argv0).parent_path() / path).string());
    candidates.push_back((std::filesystem::path(argv0).parent_path() / ".." / path).string());
    for (const auto& c : candidates) if (std::filesystem::exists(c)) return c;
    return path;
}

Control normalize_cli_control(Control c, const HallConfig&) {
    c.ventilation_group_pct = std::clamp(c.ventilation_group_pct, 0.0, 100.0);
    c.heating_group_pct = std::clamp(c.heating_group_pct, 0.0, 100.0);
    c.light_on = c.light_on ? 1 : 0;
    return c;
}

State seed_state_from_sensors(State state, const Disturbance& d, const HallConfig& cfg) {
    if (d.sensor_indoor_temp_c > -900.0) {
        state.indoor_temp_c = d.sensor_indoor_temp_c;
        state.mass_temperature_c = d.sensor_indoor_temp_c;
    }
    if (d.sensor_indoor_rh_pct >= 0.0) state.indoor_rh_pct = d.sensor_indoor_rh_pct;
    if (d.sensor_indoor_wind_m_s >= 0.0) state.air_speed_m_s = d.sensor_indoor_wind_m_s;
    if (d.sensor_indoor_co2_ppm >= 0.0) state.co2_ppm = d.sensor_indoor_co2_ppm;
    if (d.sensor_indoor_nh3_ppm >= 0.0) state.nh3_ppm = d.sensor_indoor_nh3_ppm;
    if (d.sensor_indoor_h2o_g_m3 >= 0.0) state.h2o_g_m3 = d.sensor_indoor_h2o_g_m3;
    if (d.sensor_indoor_rad_kj_m2_day >= 0.0) state.indoor_rad_kj_m2_day = d.sensor_indoor_rad_kj_m2_day;
    if (d.sensor_indoor_okta >= 0.0) state.indoor_okta = d.sensor_indoor_okta;
    if (d.sensor_indoor_aha >= 0.0) state.indoor_aha = d.sensor_indoor_aha;
    state.gas_index = std::clamp(25.0 * std::max(0.0, state.co2_ppm - 1500.0) / 500.0 +
                                 40.0 * std::max(0.0, state.nh3_ppm - 10.0) / 5.0 +
                                 15.0 * std::max(0.0, state.h2o_g_m3 - 18.0) / 4.0,
                                 0.0, 1000.0);
    (void)cfg;
    return state;
}

std::string step_timestamp(const std::string& base, std::size_t k, double dt_seconds) {
    if (base.empty()) return "forecast_step_" + std::to_string(k + 1);
    // lightweight label; keep original timestamp plus offset so no datetime parser is needed
    std::ostringstream ss;
    ss << base << "+" << static_cast<long long>(std::llround((k + 1) * dt_seconds)) << "s";
    return ss.str();
}

void write_response_stdout(const SimulationHistory& history, const std::string& fmt) {
    if (fmt == "json") {
        std::cout << "{\n  \"steps\": [\n";
        for (std::size_t i = 0; i < history.size(); ++i) {
            const auto& r = history[i];
            std::cout << "    {\"timestamp\":\"" << r.timestamp
                      << "\",\"indoor_temp_c\":" << r.state.indoor_temp_c
                      << ",\"indoor_rh_pct\":" << r.state.indoor_rh_pct
                      << ",\"indoor_co2_ppm\":" << r.state.co2_ppm
                      << ",\"indoor_nh3_ppm\":" << r.state.nh3_ppm
                      << ",\"indoor_h2o_g_m3\":" << r.state.h2o_g_m3
                      << ",\"air_speed_m_s\":" << r.state.air_speed_m_s
                      << ",\"lct_c\":" << r.thermoregulation.lower_critical_c
                      << ",\"uct_c\":" << r.thermoregulation.upper_critical_c
                      << ",\"ventilation_group_pct\":" << r.control.ventilation_group_pct
                      << ",\"heating_group_pct\":" << r.control.heating_group_pct
                      << ",\"light_on\":" << r.control.light_on
                      << ",\"fan_power_w\":" << r.layers.fan_power_w
                      << ",\"heater_power_w\":" << r.layers.heater_fuel_w
                      << ",\"light_power_w\":" << r.layers.light_power_w
                      << ",\"reward\":" << r.outputs.reward << "}";
            if (i + 1 != history.size()) std::cout << ',';
            std::cout << "\n";
        }
        std::cout << "  ]\n}\n";
    } else {
        std::cout << "timestamp,indoor_temp_c,indoor_rh_pct,indoor_co2_ppm,indoor_nh3_ppm,indoor_h2o_g_m3,air_speed_m_s,lct_c,uct_c,safe_lct_c,safe_uct_c,cohort_count,ventilation_group_pct,heating_group_pct,light_on,fan_power_w,heater_power_w,light_power_w,reward\n";
        for (const auto& r : history) {
            std::cout << r.timestamp << ',' << r.state.indoor_temp_c << ',' << r.state.indoor_rh_pct << ','
                      << r.state.co2_ppm << ',' << r.state.nh3_ppm << ',' << r.state.h2o_g_m3 << ','
                      << r.state.air_speed_m_s << ',' << r.thermoregulation.lower_critical_c << ','
                      << r.thermoregulation.upper_critical_c << ',' << r.thermoregulation.herd_safe_lower_c << ','
                      << r.thermoregulation.herd_safe_upper_c << ',' << r.thermoregulation.herd_cohort_count << ','
                      << r.control.ventilation_group_pct << ',' << r.control.heating_group_pct << ',' << r.control.light_on << ',' << r.layers.fan_power_w << ','
                      << r.layers.heater_fuel_w << ',' << r.layers.light_power_w << ',' << r.outputs.reward << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string config_path;
        std::string ifc_path = "Building_Information/beef_hall_120.ifc";
        std::string outdoor_csv;
        std::string controls_csv;
        std::string output_dir = "output";
        bool output_explicit = false;
        bool response_format_explicit = false;
        std::string herd_processed_path = "configs/herd_inventory_processed.cfg";
        std::string thermoregulation_cfg_path = "configs/thermoregulation.cfg";
        std::string timestamp = "single_step";
        double dt_seconds = 300.0;
        double forecast_horizon_seconds = 0.0;
        bool write_report = false;
        bool response_stdout = false;
        std::string response_format = "csv";
        bool use_cli_outdoor = false;
        bool use_cli_control = false;

        Disturbance cli_disturbance;
        cli_disturbance.timestamp = timestamp;
        Control cli_control;
        cli_control.timestamp = timestamp;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
            else if (arg == "--ifc" && i + 1 < argc) ifc_path = argv[++i];
            else if (arg == "--outdoor" && i + 1 < argc) outdoor_csv = argv[++i];
            else if (arg == "--controls" && i + 1 < argc) controls_csv = argv[++i];
            else if (arg == "--herd-processed" && i + 1 < argc) herd_processed_path = argv[++i];
            else if (arg == "--thermoregulation-config" && i + 1 < argc) thermoregulation_cfg_path = argv[++i];
            else if (arg == "--output" && i + 1 < argc) { output_dir = argv[++i]; output_explicit = true; }
            else if (arg == "--dt-seconds" && i + 1 < argc) dt_seconds = std::stod(argv[++i]);
            else if (arg == "--forecast-horizon-seconds" && i + 1 < argc) forecast_horizon_seconds = std::stod(argv[++i]);
            else if (arg == "--write-svg" || arg == "--write-report") write_report = true;
            else if (arg == "--response-stdout") response_stdout = true;
            else if (arg == "--response-format" && i + 1 < argc) { response_format = argv[++i]; response_format_explicit = true; }
            else if (arg == "--timestamp" && i + 1 < argc) {
                timestamp = argv[++i];
                cli_disturbance.timestamp = timestamp;
                cli_control.timestamp = timestamp;
            } else if (arg == "--outdoor-temp-c" && i + 1 < argc) {
                cli_disturbance.outdoor_temp_c = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--outdoor-rh-pct" && i + 1 < argc) {
                cli_disturbance.outdoor_rh_pct = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--outdoor-wind-m-s" && i + 1 < argc) {
                cli_disturbance.outdoor_wind_m_s = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--outdoor-solar-w-m2" && i + 1 < argc) {
                cli_disturbance.outdoor_solar_w_m2 = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--outdoor-cloud-okta" && i + 1 < argc) {
                cli_disturbance.outdoor_cloud_okta = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--outdoor-rain-mm-day" && i + 1 < argc) {
                cli_disturbance.outdoor_rain_mm_day = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--outdoor-co2-ppm" && i + 1 < argc) {
                cli_disturbance.outdoor_co2_ppm = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--outdoor-nh3-ppm" && i + 1 < argc) {
                cli_disturbance.outdoor_nh3_ppm = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--outdoor-h2o-g-m3" && i + 1 < argc) {
                cli_disturbance.outdoor_h2o_g_m3 = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--sensor-indoor-temp-c" && i + 1 < argc) {
                cli_disturbance.sensor_indoor_temp_c = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--sensor-indoor-rh-pct" && i + 1 < argc) {
                cli_disturbance.sensor_indoor_rh_pct = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--sensor-indoor-wind-m-s" && i + 1 < argc) {
                cli_disturbance.sensor_indoor_wind_m_s = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--sensor-indoor-co2-ppm" && i + 1 < argc) {
                cli_disturbance.sensor_indoor_co2_ppm = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--sensor-indoor-nh3-ppm" && i + 1 < argc) {
                cli_disturbance.sensor_indoor_nh3_ppm = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--sensor-indoor-h2o-g-m3" && i + 1 < argc) {
                cli_disturbance.sensor_indoor_h2o_g_m3 = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--sensor-indoor-rad-kj-m2-day" && i + 1 < argc) {
                cli_disturbance.sensor_indoor_rad_kj_m2_day = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--sensor-indoor-okta" && i + 1 < argc) {
                cli_disturbance.sensor_indoor_okta = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if (arg == "--sensor-indoor-aha" && i + 1 < argc) {
                cli_disturbance.sensor_indoor_aha = std::stod(argv[++i]);
                use_cli_outdoor = true;
            } else if ((arg == "--ventilation-group" || arg == "--fan-pairs") && i + 1 < argc) {
                cli_control.ventilation_group_pct = parse_group_command(argv[++i]);
                use_cli_control = true;
            } else if ((arg == "--heating-group" || arg == "--heaters") && i + 1 < argc) {
                cli_control.heating_group_pct = parse_group_command(argv[++i]);
                use_cli_control = true;
            } else if (arg == "--light-on" && i + 1 < argc) {
                cli_control.light_on = std::stoi(argv[++i]);
                use_cli_control = true;
            } else if (arg == "--help") {
                print_usage();
                return 0;
            }
        }

        if (response_format_explicit) response_stdout = true;
        const bool write_files = output_explicit || write_report;
        if (write_files) fs::create_directories(output_dir);

        HallConfig cfg;
        std::string resolved_ifc_path;
        if (!config_path.empty()) {
            cfg = load_config_file(config_path);
        } else {
            ifc_path = resolve_ifc_path(ifc_path, argv[0]);
            if (!fs::exists(ifc_path)) throw std::runtime_error("IFC file not found: " + ifc_path);
            cfg = load_config_from_ifc(ifc_path);
            resolved_ifc_path = ifc_path;
        }

        DisturbanceSeries disturbances;
        ControlSeries controls;
        if (!outdoor_csv.empty()) disturbances = load_disturbances_csv(outdoor_csv);
        if (!controls_csv.empty()) controls = load_controls_csv(controls_csv, cfg.fan_count, cfg.heater_count);
        if (disturbances.empty() && use_cli_outdoor) disturbances.push_back(cli_disturbance);
        if (controls.empty() && use_cli_control) controls.push_back(normalize_cli_control(cli_control, cfg));

        if (disturbances.empty()) {
            print_usage();
            throw std::runtime_error("Outdoor climate input is required, either by --outdoor CSV or by CLI values.");
        }
        if (controls.empty()) {
            print_usage();
            throw std::runtime_error("Actuator input is required, either by --controls CSV or by grouped CLI values.");
        }

        thermoregulation_cfg_path = resolve_support_path(thermoregulation_cfg_path, argv[0], config_path);
        if (fs::exists(thermoregulation_cfg_path)) set_active_thermoregulation_config(load_thermoregulation_config_file(thermoregulation_cfg_path));
        herd_processed_path = resolve_support_path(herd_processed_path, argv[0], config_path);
        ClimateSimulator sim(cfg);
        auto herd_processed = load_herd_processed_cfg(herd_processed_path);
        if (!herd_processed.cohorts.empty()) sim.set_processed_herd(herd_processed);
        SimulationHistory history;

        if (forecast_horizon_seconds > 0.0) {
            const std::size_t steps = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(forecast_horizon_seconds / dt_seconds)));
            Disturbance d0 = disturbances.front();
            State initial = sim.initialize();
            initial = seed_state_from_sensors(initial, d0, cfg);
            sim.reset(initial);

            DisturbanceSeries forecast_disturbances;
            ControlSeries forecast_controls;
            forecast_disturbances.reserve(steps);
            forecast_controls.reserve(steps);
            for (std::size_t i = 0; i < steps; ++i) {
                Disturbance d = disturbances[std::min(i, disturbances.size() - 1)];
                d.timestamp = step_timestamp(!timestamp.empty() ? timestamp : disturbances.front().timestamp, i, dt_seconds);
                if (i > 0) {
                    d.sensor_indoor_temp_c = -999.0;
                    d.sensor_indoor_rh_pct = -1.0;
                    d.sensor_indoor_wind_m_s = -1.0;
                    d.sensor_indoor_co2_ppm = -1.0;
                    d.sensor_indoor_nh3_ppm = -1.0;
                    d.sensor_indoor_h2o_g_m3 = -1.0;
                    d.sensor_indoor_rad_kj_m2_day = -1.0;
                    d.sensor_indoor_okta = -1.0;
                    d.sensor_indoor_aha = -1.0;
                }
                forecast_disturbances.push_back(d);
                Control c = controls[std::min(i, controls.size() - 1)];
                c.timestamp = d.timestamp;
                forecast_controls.push_back(c);
            }
            history = sim.rollout(forecast_disturbances, forecast_controls, dt_seconds);
        } else {
            history = sim.rollout(disturbances, controls, dt_seconds);
        }

        const fs::path out_base(output_dir);
        if (write_files) {
            save_results_csv(history, (out_base / "simulation_results.csv").string());
            save_config_file(cfg, (out_base / "resolved_config.cfg").string());
            if (write_report) {
                write_html_report(history, cfg, (out_base / "simulation_report.html").string());
                const fs::path exe_dir = fs::path(argv[0]).parent_path();
                std::vector<fs::path> candidates = {exe_dir / ".." / "third_party" / "chart.min.js", fs::current_path() / "third_party" / "chart.min.js", fs::current_path() / "../third_party/chart.min.js"};
                for (const auto& cand : candidates) {
                    if (fs::exists(cand)) {
                        fs::copy_file(cand, out_base / "chart.min.js", fs::copy_options::overwrite_existing);
                        break;
                    }
                }
            }

            std::ofstream meta(out_base / "run_summary.txt");
            meta << "Simulation completed with " << history.size() << " steps.\n";
            meta << "Output directory: " << output_dir << "\n";
            if (!resolved_ifc_path.empty()) meta << "Resolved IFC: " << resolved_ifc_path << "\n";
            if (forecast_horizon_seconds > 0.0) meta << "Forecast horizon seconds: " << forecast_horizon_seconds << "\n";
            meta << "dt_seconds: " << dt_seconds << "\n";
            if (forecast_horizon_seconds > 0.0) {
                meta << "Forecast mode: first indoor sensor row seeds initial state only; future steps are model predictions.\n";
            }
        }

        if (response_stdout) write_response_stdout(history, response_format);

        if (response_stdout && response_format == "json") {
            if (write_files) std::cerr << "Simulation completed. Results written to: " << output_dir << "\n";
        } else if (write_files) {
            std::cout << "Simulation completed. Results written to: " << output_dir << "\n";
        } else {
            std::cout << "Simulation completed.\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
