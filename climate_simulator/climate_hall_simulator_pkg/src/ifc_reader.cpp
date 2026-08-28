#include "ifc_reader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef BEEFCLIMATE_WITH_IFCOPENSHELL
#include <ifcparse/IfcFile.h>
#include <ifcparse/Ifc2x3.h>
#include <ifcparse/Ifc4.h>
#include <ifcparse/Ifc4x3_add2.h>
#endif

namespace beefclimate {
namespace {

struct StepEntity {
    int id = 0;
    std::string type;
    std::vector<std::string> args;
    std::string raw;
};

struct StepGraph {
    std::unordered_map<int, StepEntity> by_id;
    std::unordered_map<std::string, std::vector<int>> ids_by_type;
    std::string schema_name = "UNKNOWN";
};

std::string trim(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

std::string dequote(std::string s) {
    s = trim(s);
    if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') {
        s = s.substr(1, s.size() - 2);
        std::string out;
        out.reserve(s.size());
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '\'' && i + 1 < s.size() && s[i + 1] == '\'') { out.push_back('\''); ++i; }
            else out.push_back(s[i]);
        }
        return out;
    }
    return s;
}

std::optional<int> parse_ref(const std::string& s) {
    std::string t = trim(s);
    if (!t.empty() && t[0] == '#') {
        try { return std::stoi(t.substr(1)); } catch (...) { return std::nullopt; }
    }
    return std::nullopt;
}

std::vector<std::string> split_top_level_args(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    int depth_paren = 0;
    int depth_list = 0;
    bool in_string = false;
    for (std::size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '\'') {
            cur.push_back(c);
            if (i + 1 < s.size() && s[i + 1] == '\'') {
                cur.push_back(s[i + 1]);
                ++i;
            } else {
                in_string = !in_string;
            }
            continue;
        }
        if (!in_string) {
            if (c == '(') ++depth_paren;
            else if (c == ')') --depth_paren;
            else if (c == '[') ++depth_list;
            else if (c == ']') --depth_list;
            else if (c == ',' && depth_paren == 0 && depth_list == 0) {
                out.push_back(trim(cur));
                cur.clear();
                continue;
            }
        }
        cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(trim(cur));
    return out;
}

std::vector<int> parse_ref_list(std::string s) {
    s = trim(s);
    if (!s.empty() && s.front() == '(' && s.back() == ')') s = s.substr(1, s.size() - 2);
    std::vector<int> out;
    for (const auto& item : split_top_level_args(s)) {
        if (auto id = parse_ref(item)) out.push_back(*id);
    }
    return out;
}

std::string slurp_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open IFC file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

StepGraph parse_step_graph(const std::string& path) {
    const std::string text = slurp_file(path);
    StepGraph g;
    {
        const std::string marker = "FILE_SCHEMA((";
        auto pos = text.find(marker);
        if (pos != std::string::npos) {
            auto end = text.find(")", pos + marker.size());
            if (end != std::string::npos) g.schema_name = dequote(text.substr(pos + marker.size(), end - pos - marker.size()));
        }
    }

    std::size_t i = 0;
    while (i < text.size()) {
        if (text[i] != '#') { ++i; continue; }
        std::size_t j = i + 1;
        while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
        if (j >= text.size() || text[j] != '=') { ++i; continue; }
        int id = std::stoi(text.substr(i + 1, j - i - 1));
        std::size_t k = j + 1;
        while (k < text.size() && std::isspace(static_cast<unsigned char>(text[k]))) ++k;
        std::size_t type_beg = k;
        while (k < text.size() && (std::isalpha(static_cast<unsigned char>(text[k])) || std::isdigit(static_cast<unsigned char>(text[k])) || text[k] == '_')) ++k;
        std::string type = upper(text.substr(type_beg, k - type_beg));
        while (k < text.size() && std::isspace(static_cast<unsigned char>(text[k]))) ++k;
        if (k >= text.size() || text[k] != '(') { i = k; continue; }
        std::size_t args_beg = k + 1;
        int depth = 1;
        bool in_string = false;
        ++k;
        while (k < text.size() && depth > 0) {
            char c = text[k];
            if (c == '\'') {
                if (in_string && k + 1 < text.size() && text[k + 1] == '\'') { k += 2; continue; }
                in_string = !in_string;
            } else if (!in_string) {
                if (c == '(') ++depth;
                else if (c == ')') --depth;
            }
            ++k;
        }
        if (depth != 0) break;
        const std::size_t args_end = k - 1;
        const std::string args_text = text.substr(args_beg, args_end - args_beg);
        const std::size_t semi = text.find(';', k);
        StepEntity e;
        e.id = id;
        e.type = type;
        e.args = split_top_level_args(args_text);
        e.raw = text.substr(i, semi == std::string::npos ? std::string::npos : semi - i + 1);
        g.ids_by_type[type].push_back(id);
        g.by_id[id] = std::move(e);
        i = semi == std::string::npos ? k : semi + 1;
    }
    return g;
}

std::optional<double> parse_ifc_numeric_literal(const std::string& s) {
    std::string t = trim(s);
    auto open = t.find('(');
    auto close = t.rfind(')');
    if (open != std::string::npos && close != std::string::npos && close > open + 1) t = t.substr(open + 1, close - open - 1);
    t = trim(t);
    if (t == "$" || t.empty()) return std::nullopt;
    try { return std::stod(t); } catch (...) { return std::nullopt; }
}

std::optional<int> parse_ifc_integer_literal(const std::string& s) {
    if (auto v = parse_ifc_numeric_literal(s)) return static_cast<int>(*v);
    return std::nullopt;
}

std::optional<double> property_numeric_value(const StepGraph& g, int prop_id) {
    auto it = g.by_id.find(prop_id);
    if (it == g.by_id.end()) return std::nullopt;
    const StepEntity& e = it->second;
    if (e.type != "IFCPROPERTYSINGLEVALUE" || e.args.size() < 3) return std::nullopt;
    return parse_ifc_numeric_literal(e.args[2]);
}

std::optional<std::string> property_name(const StepGraph& g, int prop_id) {
    auto it = g.by_id.find(prop_id);
    if (it == g.by_id.end()) return std::nullopt;
    const StepEntity& e = it->second;
    if (e.type != "IFCPROPERTYSINGLEVALUE" || e.args.empty()) return std::nullopt;
    return dequote(e.args[0]);
}

std::map<std::string, double> collect_pset_numeric_properties(const StepGraph& g, int pset_id) {
    std::map<std::string, double> props;
    auto it = g.by_id.find(pset_id);
    if (it == g.by_id.end()) return props;
    const StepEntity& pset = it->second;
    if (pset.type != "IFCPROPERTYSET" || pset.args.size() < 5) return props;
    for (int prop_id : parse_ref_list(pset.args[4])) {
        auto name = property_name(g, prop_id);
        auto value = property_numeric_value(g, prop_id);
        if (name && value) props[*name] = *value;
    }
    return props;
}

std::string pset_name(const StepGraph& g, int pset_id) {
    auto it = g.by_id.find(pset_id);
    if (it == g.by_id.end()) return {};
    const StepEntity& e = it->second;
    if (e.type != "IFCPROPERTYSET" || e.args.size() < 3) return {};
    return dequote(e.args[2]);
}

std::vector<int> attached_psets(const StepGraph& g, int object_id) {
    std::vector<int> out;
    auto rels_it = g.ids_by_type.find("IFCRELDEFINESBYPROPERTIES");
    if (rels_it == g.ids_by_type.end()) return out;
    for (int rel_id : rels_it->second) {
        const StepEntity& rel = g.by_id.at(rel_id);
        if (rel.args.size() < 6) continue;
        const auto related = parse_ref_list(rel.args[4]);
        if (std::find(related.begin(), related.end(), object_id) == related.end()) continue;
        if (auto pset_id = parse_ref(rel.args[5])) out.push_back(*pset_id);
    }
    return out;
}

bool has_attached_pset_named(const StepGraph& g, int object_id, const std::string& wanted_name) {
    for (int pset_id : attached_psets(g, object_id)) {
        if (pset_name(g, pset_id) == wanted_name) return true;
    }
    return false;
}

std::optional<double> lookup_attached_numeric_property(const StepGraph& g, int object_id, const std::string& wanted) {
    for (int pset_id : attached_psets(g, object_id)) {
        const auto props = collect_pset_numeric_properties(g, pset_id);
        auto it = props.find(wanted);
        if (it != props.end()) return it->second;
    }
    return std::nullopt;
}

std::vector<int> ids_for(const StepGraph& g, const std::string& type) {
    auto it = g.ids_by_type.find(upper(type));
    if (it == g.ids_by_type.end()) return {};
    return it->second;
}

void assign_if_present(double& target, const std::optional<double>& v) { if (v) target = *v; }
void assign_if_present(int& target, const std::optional<int>& v) { if (v) target = *v; }

IfcValidationSummary summarize_from_graph(const StepGraph& g, const std::string& parser_mode) {
    IfcValidationSummary s;
    s.parser_mode = parser_mode;
    s.schema_name = g.schema_name;
    s.window_count = static_cast<int>(ids_for(g, "IFCWINDOW").size());
    s.fan_count = static_cast<int>(ids_for(g, "IFCFLOWMOVINGDEVICE").size());
    s.heater_count = static_cast<int>(ids_for(g, "IFCSPACEHEATER").size());
    s.light_count = static_cast<int>(ids_for(g, "IFCLIGHTFIXTURE").size());
    for (const auto& [type, ids] : g.ids_by_type) {
        if (type == "IFCPROPERTYSET") {
            for (int id : ids) s.property_sets[pset_name(g, id)] += 1;
        }
    }
    for (int pset_id : ids_for(g, "IFCPROPERTYSET")) {
        for (const auto& [k, v] : collect_pset_numeric_properties(g, pset_id)) s.numeric_properties.emplace(k, v);
    }
    return s;
}

HallConfig build_config_from_graph(const StepGraph& g) {
    HallConfig cfg;

    // Building-level psets attached to the single IfcBuilding template object.
    const auto buildings = ids_for(g, "IFCBUILDING");
    if (!buildings.empty()) {
        const int building_id = buildings.front();
        assign_if_present(cfg.length_m, lookup_attached_numeric_property(g, building_id, "BuildingLength_m"));
        assign_if_present(cfg.width_m, lookup_attached_numeric_property(g, building_id, "BuildingWidth_m"));
        assign_if_present(cfg.eave_height_m, lookup_attached_numeric_property(g, building_id, "EaveHeight_m"));
        assign_if_present(cfg.ridge_height_m, lookup_attached_numeric_property(g, building_id, "RidgeHeight_m"));
        assign_if_present(cfg.ridge_opening_m, lookup_attached_numeric_property(g, building_id, "RidgeOpening_m"));
        assign_if_present(cfg.volume_m3, lookup_attached_numeric_property(g, building_id, "ApproxEnclosedVolume_m3"));
        if (auto v = lookup_attached_numeric_property(g, building_id, "PenCount")) cfg.pen_count = static_cast<int>(*v);
        if (auto v = lookup_attached_numeric_property(g, building_id, "CattlePerPen")) cfg.cattle_per_pen = static_cast<int>(*v);
        cfg.cattle_count = cfg.pen_count * cfg.cattle_per_pen;
        assign_if_present(cfg.wall_u_w_m2k, lookup_attached_numeric_property(g, building_id, "WallUValue_W_m2K"));
        assign_if_present(cfg.roof_u_w_m2k, lookup_attached_numeric_property(g, building_id, "RoofUValue_W_m2K"));
        assign_if_present(cfg.personnel_door_u_w_m2k, lookup_attached_numeric_property(g, building_id, "PersonnelDoorUValue_W_m2K"));
        assign_if_present(cfg.service_door_u_w_m2k, lookup_attached_numeric_property(g, building_id, "ServiceDoorUValue_W_m2K"));
        assign_if_present(cfg.design_indoor_temp_c, lookup_attached_numeric_property(g, building_id, "IndoorDesignTemp_C"));
        assign_if_present(cfg.design_outdoor_temp_c, lookup_attached_numeric_property(g, building_id, "OutdoorDesignTemp_C"));
        assign_if_present(cfg.design_delta_t_c, lookup_attached_numeric_property(g, building_id, "DesignDeltaT_C"));
        assign_if_present(cfg.door_leak_w_m2k, lookup_attached_numeric_property(g, building_id, "DoorLeakageCoeff_W_m2K_equiv"));
        assign_if_present(cfg.window_leak_w_m2k, lookup_attached_numeric_property(g, building_id, "WindowLeakageCoeff_W_m2K_equiv"));
        assign_if_present(cfg.wall_leak_w_m2k, lookup_attached_numeric_property(g, building_id, "WallLeakageCoeff_W_m2K_equiv"));
        assign_if_present(cfg.roof_leak_w_m2k, lookup_attached_numeric_property(g, building_id, "RoofLeakageCoeff_W_m2K_equiv"));
    }

    const auto windows = ids_for(g, "IFCWINDOW");
    const auto fans = ids_for(g, "IFCFLOWMOVINGDEVICE");
    const auto heaters = ids_for(g, "IFCSPACEHEATER");
    const auto lights = ids_for(g, "IFCLIGHTFIXTURE");
    cfg.intake_count = 0;
    for (int id : windows) {
        if (has_attached_pset_named(g, id, "Pset_IntakeWindowInfo")) ++cfg.intake_count;
    }
    if (cfg.intake_count == 0) cfg.intake_count = static_cast<int>(windows.size());
    cfg.fan_count = static_cast<int>(fans.size());
    cfg.heater_count = static_cast<int>(heaters.size());
    cfg.light_count = static_cast<int>(lights.size());

    if (!windows.empty()) {
        assign_if_present(cfg.intake_width_m, lookup_attached_numeric_property(g, windows.front(), "OpeningWidth_m"));
        assign_if_present(cfg.intake_height_m, lookup_attached_numeric_property(g, windows.front(), "OpeningHeight_m"));
        assign_if_present(cfg.intake_center_z_m, lookup_attached_numeric_property(g, windows.front(), "CenterZ_m"));
    }
    if (!fans.empty()) {
        assign_if_present(cfg.fan_power_w_each, lookup_attached_numeric_property(g, fans.front(), "ElectricalDemand_W"));
        assign_if_present(cfg.fan_free_air_flow_m3h_each, lookup_attached_numeric_property(g, fans.front(), "FreeAirFlow_m3h"));
        assign_if_present(cfg.fan_flow_m3h_each, lookup_attached_numeric_property(g, fans.front(), "FlowAt20Pa_m3h"));
        assign_if_present(cfg.fan_air_speed_mps_each, lookup_attached_numeric_property(g, fans.front(), "ApproxAirSpeedAt20Pa_mps"));
        assign_if_present(cfg.fan_center_z_m, lookup_attached_numeric_property(g, fans.front(), "CenterZ_m"));
    }
    if (!heaters.empty()) {
        assign_if_present(cfg.heater_gas_input_kw_each, lookup_attached_numeric_property(g, heaters.front(), "RatedGasInput_kW"));
        assign_if_present(cfg.heater_useful_kw_each, lookup_attached_numeric_property(g, heaters.front(), "UsefulHeatOutput_kW"));
        assign_if_present(cfg.heater_airflow_m3h_each, lookup_attached_numeric_property(g, heaters.front(), "ApproxAirFlow_m3h"));
        if (cfg.heater_airflow_m3h_each <= 0.0) assign_if_present(cfg.heater_airflow_m3h_each, lookup_attached_numeric_property(g, heaters.front(), "ApproxGasFlow_m3h"));
        assign_if_present(cfg.heater_center_z_m, lookup_attached_numeric_property(g, heaters.front(), "CenterZ_m"));
    }
    if (!lights.empty()) {
        assign_if_present(cfg.light_power_w_each, lookup_attached_numeric_property(g, lights.front(), "ElectricalLoad_W"));
        assign_if_present(cfg.light_luminous_flux_lm_each, lookup_attached_numeric_property(g, lights.front(), "LuminousFlux_lm"));
    }
    return cfg;
}

#ifdef BEEFCLIMATE_WITH_IFCOPENSHELL

template <typename Schema>
IfcValidationSummary summarize_ifcopenshell_schema(IfcParse::IfcFile& file) {
    IfcValidationSummary s;
    s.parser_mode = "IfcOpenShell-C++";
    s.schema_name = file.schema()->name();

    s.window_count = static_cast<int>(file.template instances_by_type<typename Schema::IfcWindow>()->size());
    s.fan_count = static_cast<int>(file.template instances_by_type<typename Schema::IfcFlowMovingDevice>()->size());
    s.heater_count = static_cast<int>(file.template instances_by_type<typename Schema::IfcSpaceHeater>()->size());
    s.light_count = static_cast<int>(file.template instances_by_type<typename Schema::IfcLightFixture>()->size());

    const auto psets = file.template instances_by_type<typename Schema::IfcPropertySet>();
    for (auto it = psets->begin(); it != psets->end(); ++it) {
        auto* pset = *it;
        if (!pset) continue;
        s.property_sets[pset->Name()] += 1;
    }
    return s;
}

IfcValidationSummary inspect_ifc_ifcopenshell(const std::string& ifc_path) {
    IfcParse::IfcFile file(ifc_path);
    if (!file.good()) throw std::runtime_error("IfcOpenShell C++ could not parse IFC file: " + ifc_path);
    auto schema_version = file.schema()->name();
    if (schema_version == "IFC2X3") return summarize_ifcopenshell_schema<Ifc2x3>(file);
    if (schema_version == "IFC4") return summarize_ifcopenshell_schema<Ifc4>(file);
    if (schema_version == "IFC4X3_ADD2" || schema_version == "IFC4X3") return summarize_ifcopenshell_schema<Ifc4x3_add2>(file);
    IfcValidationSummary s;
    s.parser_mode = "IfcOpenShell-C++";
    s.schema_name = schema_version;
    return s;
}
#endif

}  // namespace

bool ifcopenshell_cxx_enabled() {
#ifdef BEEFCLIMATE_WITH_IFCOPENSHELL
    return true;
#else
    return false;
#endif
}

std::string ifcopenshell_validation_status() {
#ifdef BEEFCLIMATE_WITH_IFCOPENSHELL
    return "enabled-and-linked";
#else
    return "not-linked-in-this-build";
#endif
}

IfcValidationSummary inspect_ifc(const std::string& ifc_path) {
    const StepGraph graph = parse_step_graph(ifc_path);
#ifdef BEEFCLIMATE_WITH_IFCOPENSHELL
    try {
        auto s = inspect_ifc_ifcopenshell(ifc_path);
        // Enrich official-API summary with parsed scalar values from the same IFC model.
        auto parsed = summarize_from_graph(graph, s.parser_mode);
        s.property_sets = std::move(parsed.property_sets);
        s.numeric_properties = std::move(parsed.numeric_properties);
        return s;
    } catch (...) {
        return summarize_from_graph(graph, "STEP-graph-fallback-after-IfcOpenShell-error");
    }
#else
    return summarize_from_graph(graph, "STEP-graph-fallback");
#endif
}

HallConfig load_config_from_ifc(const std::string& ifc_path) {
    const StepGraph graph = parse_step_graph(ifc_path);
#ifdef BEEFCLIMATE_WITH_IFCOPENSHELL
    // Parse validity is checked with the official library first. Detailed scalar extraction still relies on
    // the STEP graph because the configured building template stores custom numeric fields in property sets,
    // and this path remains schema-agnostic in the absence of generated typed value visitors.
    IfcParse::IfcFile file(ifc_path);
    if (!file.good()) throw std::runtime_error("IfcOpenShell C++ could not parse IFC file: " + ifc_path);
#endif
    return build_config_from_graph(graph);
}

}  // namespace beefclimate
