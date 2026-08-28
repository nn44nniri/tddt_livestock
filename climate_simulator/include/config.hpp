#pragma once

#include "types.hpp"

#include <string>

namespace beefclimate {

HallConfig load_config_file(const std::string& path);
void save_config_file(const HallConfig& cfg, const std::string& path);

}  // namespace beefclimate
