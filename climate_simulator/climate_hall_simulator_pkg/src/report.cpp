#include "report.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <functional>

namespace beefclimate {
namespace {

std::string zone_label(ThermalZoneClass z) {
    switch (z) {
        case ThermalZoneClass::BelowTNZ: return "below TNZ";
        case ThermalZoneClass::AboveTNZ: return "above TNZ";
        default: return "inside TNZ";
    }
}

std::string json_array(const std::vector<double>& values, int precision = 4) {
    std::ostringstream oss;
    oss << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) oss << ',';
        oss << std::fixed << std::setprecision(precision) << values[i];
    }
    oss << ']';
    return oss.str();
}

std::string json_strings(const std::vector<std::string>& values) {
    std::ostringstream oss;
    oss << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) oss << ',';
        oss << '"';
        for (char c : values[i]) {
            if (c == '"' || c == '\\') oss << '\\';
            oss << c;
        }
        oss << '"';
    }
    oss << ']';
    return oss.str();
}

void add_dataset(std::ostringstream& js, bool& first,
                 const std::string& label,
                 const std::vector<double>& values,
                 const std::string& border,
                 bool dashed = false,
                 const std::string& fill_target = "",
                 const std::string& background = "") {
    if (!first) js << ',';
    first = false;
    js << "{label:" << '"' << label << '"'
       << ",data:" << json_array(values)
       << ",borderColor:'" << border << "'"
       << ",backgroundColor:'" << (background.empty() ? border : background) << "'"
       << ",borderWidth:2,pointRadius:0,tension:0.25";
    if (dashed) js << ",borderDash:[6,5]";
    if (!fill_target.empty()) js << ",fill:'" << fill_target << "'";
    else js << ",fill:false";
    js << '}';
}

void write_chart_block(std::ofstream& out, const std::string& canvas_id, const std::string& title) {
    out << "<section class='panel'><h2>" << title
        << "</h2><div class='chart-wrap'><canvas id='" << canvas_id
        << "'></canvas></div></section>\n";
}

}  // namespace

void write_html_report(const SimulationHistory& history, const HallConfig&, const std::string& path) {
    if (history.empty()) throw std::runtime_error("Cannot create report from empty history");

    std::vector<std::string> labels;
    std::vector<double> tin, tout, rhin, rhout, windin, windout, vent, heat, light, fan_kw, led_kw, gas_kw, lct, uct;
    std::vector<double> co2, nh3, h2o, rad, okta, aha;

    for (const auto& r : history) {
        labels.push_back(r.disturbance.timestamp.empty() ? r.control.timestamp : r.disturbance.timestamp);
        tin.push_back(r.state.indoor_temp_c);
        tout.push_back(r.disturbance.outdoor_temp_c);
        rhin.push_back(r.state.indoor_rh_pct);
        rhout.push_back(r.disturbance.outdoor_rh_pct);
        windin.push_back(r.state.air_speed_m_s);
        windout.push_back(r.disturbance.outdoor_wind_m_s);
        vent.push_back(r.control.ventilation_group_pct);
        heat.push_back(r.control.heating_group_pct);
        light.push_back(r.control.light_on ? 100.0 : 0.0);
        fan_kw.push_back(r.layers.fan_power_w / 1000.0);
        led_kw.push_back(r.layers.light_power_w / 1000.0);
        gas_kw.push_back(r.layers.heater_fuel_w / 1000.0);
        lct.push_back(r.thermoregulation.lower_critical_c);
        uct.push_back(r.thermoregulation.upper_critical_c);
        co2.push_back(r.state.co2_ppm);
        nh3.push_back(r.state.nh3_ppm);
        h2o.push_back(r.state.h2o_g_m3);
        rad.push_back(r.state.indoor_rad_kj_m2_day);
        okta.push_back(r.state.indoor_okta);
        aha.push_back(r.state.indoor_aha);
    }

    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write html report: " + path);

    const auto& last = history.back();

    out << "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>\n";
    out << "<meta name='viewport' content='width=device-width,initial-scale=1'>\n";
    out << "<title>Beef Hall Climate Simulator Report</title>\n";
    out << "<style>"
           "body{font-family:Arial,sans-serif;background:#f3f4f6;color:#111827;margin:0;padding:24px;}"
           "header{margin-bottom:20px;}"
           "header h1{margin:0 0 8px 0;font-size:28px;}"
           "header p{margin:0;color:#374151;font-size:14px;}"
           ".panel{background:#fff;border:1px solid #d1d5db;border-radius:12px;padding:14px 16px;margin:0 0 16px 0;box-shadow:0 1px 2px rgba(0,0,0,.05);overflow:hidden;}"
           ".panel h2{margin:0 0 10px 0;font-size:17px;}"
           ".chart-wrap{position:relative;width:100%;height:260px;min-height:260px;max-height:260px;}"
           "canvas{display:block;width:100% !important;height:100% !important;max-height:260px !important;}"
           ".summary p{margin:8px 0;font-size:13px;color:#374151;}"
           ".layer-note{font-size:12.4px;padding:6px 0;border-top:1px solid #e5e7eb;}"
           "</style>\n";
    out << "<script src='chart.min.js'></script>\n";
    out << "</head><body>\n";
    out << "<header><h1>Beef Hall Climate Simulator Report</h1>"
           "<p>Offline HTML report powered by Chart.js | grouped lights/ventilation/heating | harmful gases CO2/NH3/H2O | separate fan and LED electricity rows</p></header>\n";

    write_chart_block(out, "tempChart", "Temperature (outdoor/indoor + TNZ)");
    write_chart_block(out, "humidityChart", "Humidity (outdoor/indoor)");
    write_chart_block(out, "windChart", "Wind (outdoor/indoor)");
    write_chart_block(out, "actuatorChart", "Actuator states");
    write_chart_block(out, "gasChart", "Harmful gases in hall (indoor)");
    write_chart_block(out, "fanChart", "Ventilation fan electricity consumption");
    write_chart_block(out, "ledChart", "LED lighting electricity consumption");
    write_chart_block(out, "fuelChart", "Gas consumption");
    write_chart_block(out, "radChart", "Indoor RAD");
    write_chart_block(out, "oktaChart", "Indoor OKTA");
    write_chart_block(out, "ahaChart", "Indoor AHA");

    out << "<section class='panel summary'><h2>Latest-step Summary</h2>\n";
    out << "<p>Indoor temperature: " << last.state.indoor_temp_c << " C | Outdoor temperature: " << last.disturbance.outdoor_temp_c
        << " C | LCT: " << last.thermoregulation.lower_critical_c << " C | UCT: " << last.thermoregulation.upper_critical_c
        << " C | Zone: " << zone_label(last.thermoregulation.zone) << "</p>\n";
    out << "<p>Indoor RH: " << last.state.indoor_rh_pct << " % | Outdoor RH: " << last.disturbance.outdoor_rh_pct
        << " % | Indoor air speed: " << last.state.air_speed_m_s << " m/s | Outdoor wind: " << last.disturbance.outdoor_wind_m_s << " m/s</p>\n";
    out << "<p>CO2: " << last.state.co2_ppm << " ppm | NH3: " << last.state.nh3_ppm << " ppm | H2O: " << last.state.h2o_g_m3
        << " g/m3 | Gas index: " << last.state.gas_index << "</p>\n";
    out << "<p>RAD: " << last.state.indoor_rad_kj_m2_day << " kJ/m2/day | OKTA: " << last.state.indoor_okta
        << " | AHA: " << last.state.indoor_aha << "</p>\n";
    out << "<p>Ventilation group: " << last.control.ventilation_group_pct << " % | Heating group: " << last.control.heating_group_pct
        << " % | Light: " << last.control.light_on << "</p>\n";
    out << "<p>Fan electric power: " << last.layers.fan_power_w / 1000.0 << " kW | LED electric power: "
        << last.layers.light_power_w / 1000.0 << " kW | Gas power: " << last.layers.heater_fuel_w / 1000.0
        << " kW | Reward: " << last.outputs.reward << "</p>\n";
    const std::vector<LayerId> ordered{LayerId::BML, LayerId::ETL, LayerId::OCL, LayerId::ALL, LayerId::AOL, LayerId::LPL,
                                       LayerId::ICS, LayerId::TEL, LayerId::EML, LayerId::CCL, LayerId::CEL, LayerId::CIL, LayerId::VML};
    for (LayerId lid : ordered) {
        auto it = last.layer_notes.find(lid);
        if (it != last.layer_notes.end()) out << "<div class='layer-note'>" << it->second << "</div>\n";
    }
    out << "</section>\n";

    out << "<script>\n";
    out << "const labels=" << json_strings(labels) << ";\n";
    out << R"JS(
function panelConfig(datasets, yText) {
  return {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 10 } },
        tooltip: { enabled: true }
      },
      scales: {
        x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } },
        y: { beginAtZero: false, title: { display: true, text: yText } }
      },
      elements: { point: { radius: 0 } }
    }
  };
}
function mk(id, datasets, yText) {
  new Chart(document.getElementById(id), panelConfig(datasets, yText));
}
)JS";

    auto write_mk = [&](const std::string& id, const std::string& ytext, const std::function<void(std::ostringstream&)>& datasets_fn) {
        std::ostringstream ds;
        bool first = true;
        datasets_fn(ds);
        out << "mk('" << id << "',[" << ds.str() << "],'" << ytext << "');\n";
    };

    write_mk("tempChart", "deg C", [&](std::ostringstream& ds){
        bool first=true;
        add_dataset(ds, first, "Outdoor T", tout, "#2563eb");
        add_dataset(ds, first, "Indoor T", tin, "#dc2626");
        add_dataset(ds, first, "LCT", lct, "#2563eb", true);
        add_dataset(ds, first, "UCT", uct, "#b91c1c", true);
    });
    write_mk("humidityChart", "%", [&](std::ostringstream& ds){ bool first=true; add_dataset(ds, first, "Outdoor RH", rhout, "#0f766e"); add_dataset(ds, first, "Indoor RH", rhin, "#0891b2"); });
    write_mk("windChart", "m/s", [&](std::ostringstream& ds){ bool first=true; add_dataset(ds, first, "Outdoor wind", windout, "#7c3aed"); add_dataset(ds, first, "Indoor air speed", windin, "#f97316"); });
    write_mk("actuatorChart", "%", [&](std::ostringstream& ds){ bool first=true; add_dataset(ds, first, "Ventilation group %", vent, "#059669"); add_dataset(ds, first, "Heating group %", heat, "#d97706"); add_dataset(ds, first, "Light on", light, "#111827"); });
    write_mk("gasChart", "ppm / g/m3", [&](std::ostringstream& ds){ bool first=true; add_dataset(ds, first, "CO2 ppm", co2, "#1d4ed8"); add_dataset(ds, first, "NH3 ppm", nh3, "#b45309"); add_dataset(ds, first, "H2O g/m3", h2o, "#0f766e"); });
    write_mk("fanChart", "kW", [&](std::ostringstream& ds){ bool first=true; add_dataset(ds, first, "Fan electric kW", fan_kw, "#111827"); });
    write_mk("ledChart", "kW", [&](std::ostringstream& ds){ bool first=true; add_dataset(ds, first, "LED electric kW", led_kw, "#7c3aed"); });
    write_mk("fuelChart", "kW", [&](std::ostringstream& ds){ bool first=true; add_dataset(ds, first, "Gas kW", gas_kw, "#be123c"); });
    write_mk("radChart", "kJ/m2/day", [&](std::ostringstream& ds){ bool first=true; add_dataset(ds, first, "RAD", rad, "#7c2d12"); });
    write_mk("oktaChart", "okta", [&](std::ostringstream& ds){ bool first=true; add_dataset(ds, first, "OKTA", okta, "#1f2937"); });
    write_mk("ahaChart", "AHA", [&](std::ostringstream& ds){ bool first=true; add_dataset(ds, first, "AHA", aha, "#0f766e"); });

    out << "</script>\n</body></html>\n";
}

}  // namespace beefclimate
