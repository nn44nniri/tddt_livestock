#include "herd_inventory.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace beefclimate;
namespace fs = std::filesystem;

namespace {
void usage() {
    std::cout << "Usage:\n"
              << "  herd_inventory_cli [--file configs/herd_inventory.cfg] [--processed configs/herd_inventory_processed.cfg] --reset\n"
              << "  herd_inventory_cli [--file ...] --add --id C001 --breed Hereford --weight-kg 480 --heat-multiplier 1.36\n"
              << "  herd_inventory_cli [--file ...] --update --id C001 --breed Hereford --weight-kg 500 --heat-multiplier 1.4\n"
              << "  herd_inventory_cli [--file ...] [--processed ...] --process\n"
              << "  herd_inventory_cli --update-file configs/herd_inventory_processed.cfg --summary.cattle_count 4 --summary.average_weight_kg 508.5 --summary.average_heat_multiplier 1.415 --summary.cohort_count 2 --cohort BxS,495,1.4,1,3 --cohort Hereford,513,1.42,3,3\n";
}

std::string normalize_key(const std::string& arg) {
    if (arg.rfind("--", 0) == 0) return arg.substr(2);
    return arg;
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string item;
    for (char ch : s) {
        if (ch == ","[0]) {
            out.push_back(item);
            item.clear();
        } else {
            item.push_back(ch);
        }
    }
    out.push_back(item);
    return out;
}

HerdCohort parse_cohort_value(const std::string& value) {
    const auto parts = split_csv(value);
    if (parts.size() != 5) {
        throw std::runtime_error("--cohort requires <breed>,<avg_weight_kg>,<avg_heat_multiplier>,<count>,<breed_library>");
    }
    HerdCohort c;
    c.breed = parts[0];
    c.avg_weight_kg = std::stod(parts[1]);
    c.avg_heat_multiplier = std::stod(parts[2]);
    c.count = std::stoi(parts[3]);
    c.breed_library = std::stoi(parts[4]);
    return c;
}
}

int main(int argc, char** argv) {
    try {
        std::string file = "configs/herd_inventory.cfg";
        std::string processed = "configs/herd_inventory_processed.cfg";
        std::string update_file;
        bool do_reset = false, do_add = false, do_update = false, do_process = false;
        CowRecord cow;
        HerdProcessedSummary direct_summary;
        bool direct_cattle_count = false;
        bool direct_avg_weight = false;
        bool direct_avg_heat = false;
        bool direct_cohort_count = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if ((arg == "--file" || arg == "file") && i + 1 < argc) file = argv[++i];
            else if ((arg == "--processed" || arg == "processed") && i + 1 < argc) processed = argv[++i];
            else if ((arg == "--update-file" || arg == "update-file") && i + 1 < argc) update_file = argv[++i];
            else if (arg == "--reset") do_reset = true;
            else if (arg == "--add") do_add = true;
            else if (arg == "--update") do_update = true;
            else if (arg == "--process") do_process = true;
            else if (arg == "--id" && i + 1 < argc) cow.id = argv[++i];
            else if (arg == "--breed" && i + 1 < argc) cow.breed = argv[++i];
            else if (arg == "--weight-kg" && i + 1 < argc) cow.weight_kg = std::stod(argv[++i]);
            else if (arg == "--heat-multiplier" && i + 1 < argc) cow.heat_multiplier = std::stod(argv[++i]);
            else if ((normalize_key(arg) == "summary.cattle_count") && i + 1 < argc) {
                direct_summary.cattle_count = std::stoi(argv[++i]);
                direct_cattle_count = true;
            }
            else if ((normalize_key(arg) == "summary.average_weight_kg") && i + 1 < argc) {
                direct_summary.average_weight_kg = std::stod(argv[++i]);
                direct_avg_weight = true;
            }
            else if ((normalize_key(arg) == "summary.average_heat_multiplier") && i + 1 < argc) {
                direct_summary.average_heat_multiplier = std::stod(argv[++i]);
                direct_avg_heat = true;
            }
            else if ((normalize_key(arg) == "summary.cohort_count") && i + 1 < argc) {
                (void)std::stoi(argv[++i]);
                direct_cohort_count = true;
            }
            else if ((normalize_key(arg) == "cohort") && i + 1 < argc) {
                direct_summary.cohorts.push_back(parse_cohort_value(argv[++i]));
            }
            else if (arg == "--help") { usage(); return 0; }
        }
        if (!fs::path(file).parent_path().empty()) fs::create_directories(fs::path(file).parent_path());
        if (!fs::path(processed).parent_path().empty()) fs::create_directories(fs::path(processed).parent_path());
        if (!update_file.empty() && !fs::path(update_file).parent_path().empty()) fs::create_directories(fs::path(update_file).parent_path());

        if (!update_file.empty()) {
            if (!direct_cattle_count || !direct_avg_weight || !direct_avg_heat) {
                throw std::runtime_error("--update-file requires summary.cattle_count, summary.average_weight_kg and summary.average_heat_multiplier");
            }
            if (direct_cohort_count && static_cast<int>(direct_summary.cohorts.size()) < 0) {
                throw std::runtime_error("Invalid cohort configuration");
            }
            save_herd_processed_cfg(direct_summary, update_file);
            std::cout << "updated\n";
            return 0;
        }
        if (do_reset) {
            reset_herd_inventory_cfg(file);
            std::cout << "reset " << file << "\n";
            return 0;
        }
        if (do_add) {
            if (cow.id.empty() || cow.breed.empty()) throw std::runtime_error("--add requires --id, --breed, --weight-kg, --heat-multiplier");
            add_cow_record(cow, file);
            std::cout << "added " << cow.id << "\n";
            return 0;
        }
        if (do_update) {
            if (cow.id.empty() || cow.breed.empty()) throw std::runtime_error("--update requires --id, --breed, --weight-kg, --heat-multiplier");
            update_cow_record(cow, file);
            std::cout << "updated " << cow.id << "\n";
            return 0;
        }
        if (do_process) {
            const auto summary = process_herd_inventory(file, processed);
            std::cout << "processed cattle=" << summary.cattle_count << " cohorts=" << summary.cohorts.size() << " -> " << processed << "\n";
            return 0;
        }
        usage();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
