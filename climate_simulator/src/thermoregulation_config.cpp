#include "thermoregulation_config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace beefclimate {
namespace {

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

template <typename T>
void assign_num(const std::unordered_map<std::string, std::string>& kv, const std::string& key, T& target) {
    const auto it = kv.find(key);
    if (it == kv.end()) return;
    std::istringstream ss(it->second);
    ss >> target;
}

ThermoregulationModelConfig& active_cfg_storage() {
    static ThermoregulationModelConfig cfg = default_thermoregulation_config();
    return cfg;
}

} // namespace

ThermoregulationModelConfig default_thermoregulation_config() {
    ThermoregulationModelConfig cfg;
    cfg.breeds["default"] = ThermoregulationBreedParams{"default", 3, 0.56, 0.012, 1.09, 64.1, 4.44, 1.03, 1.0, 34.7, 0.93};
    cfg.breeds["hereford"] = ThermoregulationBreedParams{"hereford", 3, 0.56, 0.012, 1.09, 64.1, 4.44, 1.03, 1.0, 34.7, 0.93};
    cfg.breeds["angus"] = ThermoregulationBreedParams{"angus", 3, 0.56, 0.012, 1.09, 64.1, 4.44, 1.03, 1.0, 34.7, 0.93};
    cfg.breeds["shorthorn"] = ThermoregulationBreedParams{"shorthorn", 3, 0.56, 0.012, 1.09, 64.1, 4.44, 1.03, 1.0, 34.7, 0.93};
    cfg.breeds["brahman"] = ThermoregulationBreedParams{"brahman", 2, 0.60, 0.012, 1.12, 64.1, 4.89, 0.80, 1.30, 34.5, 0.91};
    cfg.breeds["brahman_cross"] = ThermoregulationBreedParams{"brahman_cross", 2, 0.60, 0.012, 1.12, 64.1, 4.89, 0.80, 1.30, 34.5, 0.91};
    cfg.breeds["charolais"] = ThermoregulationBreedParams{"charolais", 1, 0.60, 0.012, 1.00, 64.1, 3.08, 1.73, 1.00, 35.3, 1.00};
    return cfg;
}

ThermoregulationModelConfig load_thermoregulation_config_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open thermoregulation config file: " + path);
    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';' || line[0] == '[') continue;
        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        kv[trim(line.substr(0, pos))] = trim(line.substr(pos + 1));
    }

    ThermoregulationModelConfig cfg = default_thermoregulation_config();
    assign_num(kv, "global.body_temp_c", cfg.body_temp_c);
    assign_num(kv, "global.lasmin_w_m2", cfg.lasmin_w_m2);
    assign_num(kv, "global.respiration_increase_factor", cfg.respiration_increase_factor);
    assign_num(kv, "global.rainfrac", cfg.rainfrac);
    assign_num(kv, "global.maintenance_me_coeff", cfg.maintenance_me_coeff);
    assign_num(kv, "global.metab_weight_factor", cfg.metab_weight_factor);
    assign_num(kv, "global.body_area_coeff", cfg.body_area_coeff);
    assign_num(kv, "global.body_area_exp", cfg.body_area_exp);
    assign_num(kv, "global.hif", cfg.hif);
    assign_num(kv, "global.zc", cfg.zc);
    assign_num(kv, "global.coat_const", cfg.coat_const);
    assign_num(kv, "global.gamma", cfg.gamma);
    assign_num(kv, "global.emissivity", cfg.emissivity);
    assign_num(kv, "global.schmidt", cfg.schmidt);
    assign_num(kv, "global.mu_st", cfg.mu_st);
    assign_num(kv, "global.tr0", cfg.tr0);
    assign_num(kv, "global.cconv1", cfg.cconv1);
    assign_num(kv, "global.refle_grass", cfg.refle_grass);
    assign_num(kv, "global.refle_concrete", cfg.refle_concrete);

    std::vector<std::string> breed_names;
    for (const auto& [k, _] : kv) {
        const std::string prefix = "breed.";
        if (k.rfind(prefix, 0) != 0) continue;
        const auto p = k.find('.', prefix.size());
        if (p == std::string::npos) continue;
        const std::string breed = lower(k.substr(prefix.size(), p - prefix.size()));
        if (std::find(breed_names.begin(), breed_names.end(), breed) == breed_names.end()) breed_names.push_back(breed);
    }
    for (const auto& breed : breed_names) {
        auto params = cfg.breeds.count(breed) ? cfg.breeds[breed] : ThermoregulationBreedParams{};
        params.name = breed;
        assign_num(kv, "breed." + breed + ".library_id", params.library_id);
        assign_num(kv, "breed." + breed + ".reflectance", params.reflectance);
        assign_num(kv, "breed." + breed + ".coat_depth_m", params.coat_depth_m);
        assign_num(kv, "breed." + breed + ".area_factor", params.area_factor);
        assign_num(kv, "breed." + breed + ".cbs_max", params.cbs_max);
        assign_num(kv, "breed." + breed + ".latent_a", params.latent_a);
        assign_num(kv, "breed." + breed + ".latent_b", params.latent_b);
        assign_num(kv, "breed." + breed + ".rbcsf", params.rbcsf);
        assign_num(kv, "breed." + breed + ".latent_ref_temp_c", params.latent_ref_temp_c);
        assign_num(kv, "breed." + breed + ".exhaled_temp_factor", params.exhaled_temp_factor);
        cfg.breeds[breed] = params;
    }
    return cfg;
}

void save_thermoregulation_config_file(const ThermoregulationModelConfig& cfg, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write thermoregulation config file: " + path);
    out << "# Thermoregulation tunables for breed-specific calibration\n";
    out << "global.body_temp_c=" << cfg.body_temp_c << '\n';
    out << "global.lasmin_w_m2=" << cfg.lasmin_w_m2 << '\n';
    out << "global.respiration_increase_factor=" << cfg.respiration_increase_factor << '\n';
    out << "global.rainfrac=" << cfg.rainfrac << '\n';
    out << "global.maintenance_me_coeff=" << cfg.maintenance_me_coeff << '\n';
    out << "global.metab_weight_factor=" << cfg.metab_weight_factor << '\n';
    out << "global.body_area_coeff=" << cfg.body_area_coeff << '\n';
    out << "global.body_area_exp=" << cfg.body_area_exp << '\n';
    out << "global.hif=" << cfg.hif << '\n';
    out << "global.zc=" << cfg.zc << '\n';
    out << "global.coat_const=" << cfg.coat_const << '\n';
    out << "global.gamma=" << cfg.gamma << '\n';
    out << "global.emissivity=" << cfg.emissivity << '\n';
    out << "global.schmidt=" << cfg.schmidt << '\n';
    out << "global.mu_st=" << cfg.mu_st << '\n';
    out << "global.tr0=" << cfg.tr0 << '\n';
    out << "global.cconv1=" << cfg.cconv1 << '\n';
    out << "global.refle_grass=" << cfg.refle_grass << '\n';
    out << "global.refle_concrete=" << cfg.refle_concrete << "\n\n";
    std::vector<std::string> keys;
    keys.reserve(cfg.breeds.size());
    for (const auto& [k, _] : cfg.breeds) keys.push_back(k);
    std::sort(keys.begin(), keys.end());
    for (const auto& k : keys) {
        const auto& b = cfg.breeds.at(k);
        out << "# breed=" << b.name << '\n';
        out << "breed." << b.name << ".library_id=" << b.library_id << '\n';
        out << "breed." << b.name << ".reflectance=" << b.reflectance << '\n';
        out << "breed." << b.name << ".coat_depth_m=" << b.coat_depth_m << '\n';
        out << "breed." << b.name << ".area_factor=" << b.area_factor << '\n';
        out << "breed." << b.name << ".cbs_max=" << b.cbs_max << '\n';
        out << "breed." << b.name << ".latent_a=" << b.latent_a << '\n';
        out << "breed." << b.name << ".latent_b=" << b.latent_b << '\n';
        out << "breed." << b.name << ".rbcsf=" << b.rbcsf << '\n';
        out << "breed." << b.name << ".latent_ref_temp_c=" << b.latent_ref_temp_c << '\n';
        out << "breed." << b.name << ".exhaled_temp_factor=" << b.exhaled_temp_factor << "\n\n";
    }
}

void set_active_thermoregulation_config(ThermoregulationModelConfig cfg) { active_cfg_storage() = std::move(cfg); }
const ThermoregulationModelConfig& active_thermoregulation_config() { return active_cfg_storage(); }

const ThermoregulationBreedParams& resolve_thermoregulation_breed(const std::string& breed_name, int breed_library_hint) {
    const auto& cfg = active_thermoregulation_config();
    const std::string key = lower(breed_name);
    auto it = cfg.breeds.find(key);
    if (it != cfg.breeds.end()) return it->second;
    if (breed_library_hint == 1) {
        auto jt = cfg.breeds.find("charolais"); if (jt != cfg.breeds.end()) return jt->second;
    } else if (breed_library_hint == 2) {
        auto jt = cfg.breeds.find("brahman"); if (jt != cfg.breeds.end()) return jt->second;
    } else {
        auto jt = cfg.breeds.find("hereford"); if (jt != cfg.breeds.end()) return jt->second;
    }
    return cfg.breeds.at("default");
}

} // namespace beefclimate
