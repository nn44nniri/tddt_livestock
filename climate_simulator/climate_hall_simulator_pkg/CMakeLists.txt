cmake_minimum_required(VERSION 3.16)
project(climate_hall_simulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BEEFCLIMATE_WITH_IFCOPENSHELL "Enable direct IFC parsing through official IfcOpenShell C++" OFF)
set(BEEFCLIMATE_IFCOPENSHELL_ROOT "" CACHE PATH "Root of an IfcOpenShell C++ installation")

set(BEEFCLIMATE_IFCOPENSHELL_ENABLED OFF)
if(BEEFCLIMATE_WITH_IFCOPENSHELL)
  if(BEEFCLIMATE_IFCOPENSHELL_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${BEEFCLIMATE_IFCOPENSHELL_ROOT}")
  endif()
  find_package(IfcOpenShell QUIET)
  if(IfcOpenShell_FOUND)
    set(BEEFCLIMATE_IFCOPENSHELL_ENABLED ON)
    message(STATUS "IfcOpenShell C++ integration: enabled")
  else()
    message(WARNING "BEEFCLIMATE_WITH_IFCOPENSHELL=ON was requested, but official IfcOpenShell C++ headers/libs were not found. Falling back to the built-in STEP/IFC extractor. Set BEEFCLIMATE_IFCOPENSHELL_ROOT to the install prefix to enable official integration.")
  endif()
else()
  message(STATUS "IfcOpenShell C++ integration: disabled (using built-in STEP/IFC extractor)")
endif()

set(BEEFCLIMATE_CORE_SOURCES
    src/config.cpp
    src/csv.cpp
    src/animal_load.cpp
    src/thermoregulation.cpp
    src/simulator.cpp
    src/report.cpp
    src/ifc_reader.cpp
    src/herd_inventory.cpp
)

add_library(beefclimate_core ${BEEFCLIMATE_CORE_SOURCES})
target_include_directories(beefclimate_core PUBLIC include)

if(BEEFCLIMATE_IFCOPENSHELL_ENABLED)
  target_compile_definitions(beefclimate_core PUBLIC BEEFCLIMATE_WITH_IFCOPENSHELL=1)
  target_compile_definitions(beefclimate_core PUBLIC BEEFCLIMATE_IFCOPENSHELL_ENABLED=1)
  target_link_libraries(beefclimate_core PUBLIC IfcOpenShell::IfcParse)
else()
  target_compile_definitions(beefclimate_core PUBLIC BEEFCLIMATE_IFCOPENSHELL_ENABLED=0)
endif()

add_executable(beef_climate_sim src/main.cpp)
target_link_libraries(beef_climate_sim PRIVATE beefclimate_core)

add_executable(ifc_probe src/ifc_probe.cpp)
target_link_libraries(ifc_probe PRIVATE beefclimate_core)

add_executable(herd_inventory_cli src/herd_inventory_tool.cpp)
target_link_libraries(herd_inventory_cli PRIVATE beefclimate_core)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME smoke_ifc_probe COMMAND ifc_probe ${CMAKE_SOURCE_DIR}/Building_Information/beef_hall_120.ifc)

  add_test(
    NAME smoke_simulation_csv_mode
    COMMAND beef_climate_sim
      --config ${CMAKE_SOURCE_DIR}/configs/default_hall.cfg
      --outdoor ${CMAKE_SOURCE_DIR}/data/sample_outdoor.csv
      --controls ${CMAKE_SOURCE_DIR}/data/sample_controls.csv
      --output ${CMAKE_BINARY_DIR}/smoke_output
      --dt-seconds 300)

  add_test(
    NAME smoke_simulation_cli_mode
    COMMAND beef_climate_sim
      --config ${CMAKE_SOURCE_DIR}/configs/default_hall.cfg
      --timestamp 2026-01-01T00:00:00
      --outdoor-temp-c -4
      --outdoor-rh-pct 78
      --outdoor-wind-m-s 1.5
      --outdoor-solar-w-m2 0
      --outdoor-cloud-okta 4
      --outdoor-rain-mm-day 0
      --ventilation-group 20
      --heating-group 45
      --light-on 1
      --output ${CMAKE_BINARY_DIR}/smoke_output_cli
      --dt-seconds 300)

  add_test(NAME smoke_herd_inventory_process COMMAND herd_inventory_cli --file ${CMAKE_BINARY_DIR}/herd_inventory.cfg --processed ${CMAKE_BINARY_DIR}/herd_inventory_processed.cfg --reset)

  add_test(
    NAME smoke_herd_inventory_direct_update
    COMMAND herd_inventory_cli
      --update-file ${CMAKE_BINARY_DIR}/herd_inventory_processed.cfg
      --summary.cattle_count 4
      --summary.average_weight_kg 508.5
      --summary.average_heat_multiplier 1.415
      --summary.cohort_count 2
      --cohort BxS,495,1.4,1,3
      --cohort Hereford,513,1.42,3,3)
endif()
