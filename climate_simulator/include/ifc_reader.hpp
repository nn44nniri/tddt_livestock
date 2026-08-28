#pragma once

#include "types.hpp"

#include <map>
#include <string>
#include <vector>

namespace beefclimate {

struct IfcValidationSummary {
    std::string parser_mode;
    std::string schema_name;
    int window_count = 0;
    int fan_count = 0;
    int heater_count = 0;
    int light_count = 0;
    std::map<std::string, int> property_sets;
    std::map<std::string, double> numeric_properties;
};

HallConfig load_config_from_ifc(const std::string& ifc_path);
IfcValidationSummary inspect_ifc(const std::string& ifc_path);
bool ifcopenshell_cxx_enabled();
std::string ifcopenshell_validation_status();

}  // namespace beefclimate
