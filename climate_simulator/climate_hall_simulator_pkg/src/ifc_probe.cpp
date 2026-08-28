#include "ifc_reader.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace beefclimate;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ifc_probe <path-to-ifc>\n";
        return 2;
    }
    try {
        const auto s = inspect_ifc(argv[1]);
        std::cout << "parser_mode=" << s.parser_mode << "\n";
        std::cout << "schema_name=" << s.schema_name << "\n";
        std::cout << "window_count=" << s.window_count << "\n";
        std::cout << "fan_count=" << s.fan_count << "\n";
        std::cout << "heater_count=" << s.heater_count << "\n";
        std::cout << "light_count=" << s.light_count << "\n";
        for (const auto& [k, v] : s.property_sets) {
            std::cout << "pset." << k << '=' << v << "\n";
        }
        auto print_num = [&](const char* key) {
            auto it = s.numeric_properties.find(key);
            if (it != s.numeric_properties.end()) {
                std::cout << "prop." << key << '=' << std::fixed << std::setprecision(6) << it->second << "\n";
            }
        };
        print_num("BuildingLength_m");
        print_num("BuildingWidth_m");
        print_num("ApproxEnclosedVolume_m3");
        print_num("ElectricalDemand_W");
        print_num("UsefulHeatOutput_kW");
        print_num("LuminousFlux_lm");
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
