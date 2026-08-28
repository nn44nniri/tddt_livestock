#pragma once
#include "ifc_reader.hpp"
#include <utility>
#include <vector>
namespace beefclimate {
std::vector<std::pair<std::string,int>> ifc_property_sets_items(const IfcValidationSummary& s);
std::vector<std::pair<std::string,double>> ifc_numeric_properties_items(const IfcValidationSummary& s);
}
