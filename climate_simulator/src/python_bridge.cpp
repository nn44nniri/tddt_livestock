#include "python_bridge.hpp"
namespace beefclimate {
std::vector<std::pair<std::string,int>> ifc_property_sets_items(const IfcValidationSummary& s) { return {s.property_sets.begin(), s.property_sets.end()}; }
std::vector<std::pair<std::string,double>> ifc_numeric_properties_items(const IfcValidationSummary& s) { return {s.numeric_properties.begin(), s.numeric_properties.end()}; }
}
