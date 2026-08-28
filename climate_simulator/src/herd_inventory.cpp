#include "herd_inventory.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

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

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) out.push_back(trim(item));
    return out;
}

std::string cohort_key(const CowRecord& c) {
    const int wbin = static_cast<int>(std::floor(c.weight_kg / 50.0));
    const int hbin = static_cast<int>(std::floor(c.heat_multiplier * 10.0 + 1e-9));
    return lower(c.breed) + "|" + std::to_string(wbin) + "|" + std::to_string(hbin);
}

} // namespace

int infer_breed_library(const std::string& breed_name) {
    const std::string b = lower(breed_name);
    if (b.find("brahman") != std::string::npos || b.find("bos indicus") != std::string::npos) return 1;
    if (b.find("shorthorn") != std::string::npos || b.find("friesian") != std::string::npos || b.find("jersey") != std::string::npos) return 2;
    return 3; // generic Bos taurus / Hereford-like default
}

std::vector<CowRecord> load_herd_inventory_cfg(const std::string& path) {
    std::ifstream in(path);
    if (!in) return {};
    std::vector<CowRecord> cows;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        const std::string key = trim(line.substr(0, pos));
        if (key != "cow") continue;
        const auto parts = split(line.substr(pos + 1), ',');
        if (parts.size() < 4) continue;
        CowRecord c;
        c.id = parts[0];
        c.breed = parts[1];
        c.weight_kg = std::stod(parts[2]);
        c.heat_multiplier = std::stod(parts[3]);
        cows.push_back(c);
    }
    return cows;
}

void save_herd_inventory_cfg(const std::vector<CowRecord>& cows, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write herd inventory file: " + path);
    out << "# format: cow=<id>,<breed>,<weight_kg>,<heat_multiplier>\n";
    for (const auto& c : cows) {
        out << "cow=" << c.id << ',' << c.breed << ',' << c.weight_kg << ',' << c.heat_multiplier << '\n';
    }
}

void reset_herd_inventory_cfg(const std::string& path) {
    save_herd_inventory_cfg({}, path);
}

void add_cow_record(const CowRecord& cow, const std::string& path) {
    auto cows = load_herd_inventory_cfg(path);
    for (const auto& c : cows) {
        if (c.id == cow.id) throw std::runtime_error("Cow id already exists: " + cow.id);
    }
    if (cows.size() >= 120) throw std::runtime_error("Herd inventory already contains 120 cows");
    cows.push_back(cow);
    save_herd_inventory_cfg(cows, path);
}

void update_cow_record(const CowRecord& cow, const std::string& path) {
    auto cows = load_herd_inventory_cfg(path);
    bool found = false;
    for (auto& c : cows) {
        if (c.id == cow.id) {
            c = cow;
            found = true;
            break;
        }
    }
    if (!found) throw std::runtime_error("Cow id not found: " + cow.id);
    save_herd_inventory_cfg(cows, path);
}

void save_herd_processed_cfg(const HerdProcessedSummary& summary, const std::string& path) {
    std::ofstream file(path, std::ios::trunc);
    if (!file) throw std::runtime_error("Cannot write processed herd file: " + path);
    file << "# processed herd inventory\n";
    file << "summary.cattle_count=" << summary.cattle_count << '\n';
    file << "summary.average_weight_kg=" << summary.average_weight_kg << '\n';
    file << "summary.average_heat_multiplier=" << summary.average_heat_multiplier << '\n';
    file << "summary.cohort_count=" << summary.cohorts.size() << '\n';
    for (const auto& c : summary.cohorts) {
        file << "cohort=" << c.breed << ',' << c.avg_weight_kg << ',' << c.avg_heat_multiplier << ',' << c.count << ',' << c.breed_library << '\n';
    }
}

HerdProcessedSummary process_herd_inventory(const std::string& input_path, const std::string& output_path) {
    const auto cows = load_herd_inventory_cfg(input_path);
    HerdProcessedSummary out;
    if (cows.empty()) {
        out.cattle_count = 0;
        out.average_weight_kg = 450.0;
        out.average_heat_multiplier = 1.36;
        save_herd_processed_cfg(out, output_path);
        return out;
    }

    struct Accum { std::string breed; double sum_w = 0.0; double sum_h = 0.0; int count = 0; int lib = 3; };
    std::map<std::string, Accum> bins;
    for (const auto& c : cows) {
        auto& a = bins[cohort_key(c)];
        a.breed = c.breed;
        a.sum_w += c.weight_kg;
        a.sum_h += c.heat_multiplier;
        a.count += 1;
        a.lib = infer_breed_library(c.breed);
    }

    double total_w = 0.0;
    double total_h = 0.0;
    out.cattle_count = static_cast<int>(cows.size());
    for (const auto& [k, a] : bins) {
        HerdCohort c;
        c.breed = a.breed;
        c.avg_weight_kg = a.sum_w / std::max(1, a.count);
        c.avg_heat_multiplier = a.sum_h / std::max(1, a.count);
        c.count = a.count;
        c.breed_library = a.lib;
        out.cohorts.push_back(c);
        total_w += a.sum_w;
        total_h += a.sum_h;
    }
    out.average_weight_kg = total_w / out.cattle_count;
    out.average_heat_multiplier = total_h / out.cattle_count;

    std::ofstream file(output_path);
    if (!file) throw std::runtime_error("Cannot write processed herd file: " + output_path);
    file << "# processed herd inventory\n";
    file << "summary.cattle_count=" << out.cattle_count << '\n';
    file << "summary.average_weight_kg=" << out.average_weight_kg << '\n';
    file << "summary.average_heat_multiplier=" << out.average_heat_multiplier << '\n';
    file << "summary.cohort_count=" << out.cohorts.size() << '\n';
    for (const auto& c : out.cohorts) {
        file << "cohort=" << c.breed << ',' << c.avg_weight_kg << ',' << c.avg_heat_multiplier << ',' << c.count << ',' << c.breed_library << '\n';
    }
    return out;
}

HerdProcessedSummary load_herd_processed_cfg(const std::string& path) {
    std::ifstream in(path);
    if (!in) return {};
    HerdProcessedSummary out;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        const std::string key = trim(line.substr(0, pos));
        const std::string val = trim(line.substr(pos + 1));
        if (key == "summary.cattle_count") out.cattle_count = std::stoi(val);
        else if (key == "summary.average_weight_kg") out.average_weight_kg = std::stod(val);
        else if (key == "summary.average_heat_multiplier") out.average_heat_multiplier = std::stod(val);
        else if (key == "cohort") {
            auto p = split(val, ',');
            if (p.size() >= 5) {
                HerdCohort c;
                c.breed = p[0];
                c.avg_weight_kg = std::stod(p[1]);
                c.avg_heat_multiplier = std::stod(p[2]);
                c.count = std::stoi(p[3]);
                c.breed_library = std::stoi(p[4]);
                out.cohorts.push_back(c);
            }
        }
    }
    return out;
}

} // namespace beefclimate
