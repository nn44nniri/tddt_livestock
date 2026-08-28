#pragma once

#include "types.hpp"

#include <string>
#include <vector>

namespace beefclimate {

struct CowRecord {
    std::string id;
    std::string breed;
    double weight_kg = 450.0;
    double heat_multiplier = 1.36;
};

struct HerdCohort {
    std::string breed;
    double avg_weight_kg = 450.0;
    double avg_heat_multiplier = 1.36;
    int count = 0;
    int breed_library = 3;
};

struct HerdProcessedSummary {
    std::vector<HerdCohort> cohorts;
    int cattle_count = 0;
    double average_weight_kg = 450.0;
    double average_heat_multiplier = 1.36;
};

std::vector<CowRecord> load_herd_inventory_cfg(const std::string& path);
void save_herd_inventory_cfg(const std::vector<CowRecord>& cows, const std::string& path);
void reset_herd_inventory_cfg(const std::string& path);
void add_cow_record(const CowRecord& cow, const std::string& path);
void update_cow_record(const CowRecord& cow, const std::string& path);
HerdProcessedSummary process_herd_inventory(const std::string& input_path, const std::string& output_path);
void save_herd_processed_cfg(const HerdProcessedSummary& summary, const std::string& path);
HerdProcessedSummary load_herd_processed_cfg(const std::string& path);
int infer_breed_library(const std::string& breed_name);

} // namespace beefclimate
