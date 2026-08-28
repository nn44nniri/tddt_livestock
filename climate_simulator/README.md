# climate_hall_simulator

A C++ climate simulator for the beef hall template `Building_Information/beef_hall_120.ifc`.

This revision focuses on **IfcOpenShell C++ integration readiness** and on replacing the old text-regex IFC parsing with a **relationship-aware STEP graph extractor** that reads:

- entities
- property sets
- `IfcRelDefinesByProperties`
- object-to-property assignments
- grouped equipment template values

## What is included in this revision

- full grouped-control simulator (`light`, `ventilation_group_pct`, `heating_group_pct`)
- explicit harmful-gas state for `CO2 / NH3 / H2O` with optional indoor sensor assimilation
- LiGAPS-inspired thermoregulation module in `src/thermoregulation.cpp`
- 13-layer engine structure aligned with the uploaded design documents
- new IFC inspection utility: `ifc_probe`
- official IfcOpenShell C++ **CMake integration hooks**
- robust fallback IFC extraction path that is fully tested in this environment

## Project structure

```text
climate_hall_simulator/
├── Building_Information/
│   └── beef_hall_120.ifc
├── cmake/
│   └── FindIfcOpenShell.cmake
├── configs/
│   └── default_hall.cfg
├── data/
│   ├── sample_controls.csv
│   └── sample_outdoor.csv
├── include/
│   ├── animal_load.hpp
│   ├── config.hpp
│   ├── csv.hpp
│   ├── ifc_reader.hpp
│   ├── psychrometrics.hpp
│   ├── report.hpp
│   ├── simulator.hpp
│   ├── thermoregulation.hpp
│   └── types.hpp
├── src/
│   ├── animal_load.cpp
│   ├── config.cpp
│   ├── csv.cpp
│   ├── ifc_probe.cpp
│   ├── ifc_reader.cpp
│   ├── main.cpp
│   ├── report.cpp
│   ├── simulator.cpp
│   └── thermoregulation.cpp
├── CMakeLists.txt
└── VALIDATION_IFCOPENSHELL.md
```

## IFC / BIM extraction model

The runtime IFC loader now works in two modes:

### 1) Official IfcOpenShell C++ mode

Enabled by:

```bash
cmake .. -DBEEFCLIMATE_WITH_IFCOPENSHELL=ON -DBEEFCLIMATE_IFCOPENSHELL_ROOT=/path/to/install
```

CMake then requires the official library and headers. If they are missing, configuration fails immediately.

The code path uses:

- `IfcParse::IfcFile`
- schema detection via `file.schema()->name()`
- typed entity counting via `instances_by_type<Schema::IfcWindow>()`, etc.
- property-set discovery through typed relationship traversal hooks prepared for the official API

### 2) STEP graph fallback mode

This mode is used when the official library is not linked.

It is **not regex-based anymore**. Instead it builds a small IFC STEP graph and extracts:

- all entity instances by STEP id
- entity type
- top-level arguments
- property sets (`IFCPROPERTYSET`)
- scalar properties (`IFCPROPERTYSINGLEVALUE`)
- property relations (`IFCRELDEFINESBYPROPERTIES`)
- attached PSet values for building objects and equipment

This mode is the one fully built and tested in this environment.

## Building

### Standard build

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
ctest --output-on-failure
```

### Build with official IfcOpenShell C++

```bash
mkdir -p build
cd build
cmake .. \
  -DBEEFCLIMATE_WITH_IFCOPENSHELL=ON \
  -DBEEFCLIMATE_IFCOPENSHELL_ROOT=/path/to/ifcopenshell/install
cmake --build . -j
ctest --output-on-failure
```

## IFC validation utility

The new `ifc_probe` executable inspects the IFC template and prints parser mode, schema, counts, PSet names, and selected numeric properties.

Example:

```bash
./ifc_probe ../Building_Information/beef_hall_120.ifc
```

## Running the simulator

### CSV mode

```bash
./beef_climate_sim \
  --ifc ../Building_Information/beef_hall_120.ifc \
  --outdoor ../data/sample_outdoor.csv \
  --controls ../data/sample_controls.csv \
  --output out_csv \
  --dt-seconds 300 \
  --write-report (or legacy --write-svg)
```

### CLI mode

```bash
./beef_climate_sim \
  --ifc ../Building_Information/beef_hall_120.ifc \
  --timestamp 2026-01-01T00:00:00 \
  --outdoor-temp-c -4 \
  --outdoor-rh-pct 78 \
  --outdoor-wind-m-s 1.5 \
  --outdoor-solar-w-m2 0 \
  --outdoor-cloud-okta 4 \
  --outdoor-rain-mm-day 0 \
  --ventilation-group 20 \
  --heating-group 45 \
  --light-on 1 \
  --output out_cli \
  --write-report (or legacy --write-svg)
```

```bash
./beef_climate_sim \
  --config ../configs/default_hall.cfg \
  --sensor-indoor-temp-c 18.4 \
  --sensor-indoor-rh-pct 67 \
  --sensor-indoor-wind-m-s 0.42 \
  --sensor-indoor-co2-ppm 1850 \
  --sensor-indoor-nh3-ppm 12 \
  --sensor-indoor-h2o-g-m3 14.8 \
  --sensor-indoor-rad-kj-m2-day 220 \
  --sensor-indoor-okta 8 \
  --sensor-indoor-aha 1.35 \
  --outdoor-temp-c -6 \
  --outdoor-rh-pct 78 \
  --outdoor-wind-m-s 3.2 \
  --outdoor-solar-w-m2 120 \
  --outdoor-cloud-okta 6 \
  --outdoor-rain-mm-day 0 \
  --ventilation-group 40 \
  --heating-group 55 \
  --light-on 1 \
  --write-report (or legacy --write-svg)
```

## Outputs

- `simulation_results.csv`
- `resolved_config.cfg`
- `run_summary.txt`
- `simulation_report.html` only when `--write-report (or legacy --write-svg)` is requested

## Validation status in this environment

See `VALIDATION_IFCOPENSHELL.md`.

The short version is:

- **the simulator, fallback STEP graph extractor, gas-state extension, and tests are fully validated here**
- **the official IfcOpenShell C++ path is fully prepared in code and CMake, but could not be runtime-validated here because the official dev package is not installed in this environment**

That limitation is environmental, not architectural: once the official install prefix is provided, CMake will switch to the official path and fail fast if the package is incomplete.


## CMake behavior for IfcOpenShell

- `-DBEEFCLIMATE_WITH_IFCOPENSHELL=ON` now requests official IfcOpenShell C++ integration **without making configure fail** when the library is unavailable. In that case, CMake prints a warning and automatically falls back to the built-in STEP/IFC extractor.
- To enable the official library when available, set `-DBEEFCLIMATE_IFCOPENSHELL_ROOT=/your/install/prefix`. Typical prefixes contain `include/ifcparse/IfcFile.h` and `lib/libIfcParse.so` (or equivalent).
- The release tarball does **not** ship a `build/` directory, so stale CMake cache values do not leak into a fresh configuration.

Recommended clean configure:

```bash
mkdir -p build
cd build
cmake .. -DBEEFCLIMATE_WITH_IFCOPENSHELL=ON -DBEEFCLIMATE_IFCOPENSHELL_ROOT=/your/install/prefix
cmake --build . -j
ctest --output-on-failure
```

If the official library is not found, the project still configures, builds, and tests with the fallback IFC reader.


## Indoor gas model

The simulator now tracks three indoor gas-related state variables required by the design specifications:

- `CO2` in `ppm`
- `NH3` in `ppm`
- `H2O` vapour concentration in `g/m3`

These states are used directly in the hall calculations and also contribute to the aggregate air-quality state `gas_index` used by the reward and constraint logic. This aligns with the minimum gas-model requirement in the uploaded specifications and with the official state/output requirement for `G_in,k` and `J^air_k`.

### Disturbance CSV columns

The outdoor CSV may now include both boundary values and indoor sensor measurements:

```text
timestamp,
outdoor_temp_c,outdoor_rh_pct,outdoor_wind_m_s,outdoor_solar_w_m2,outdoor_cloud_okta,outdoor_rain_mm_day,
outdoor_co2_ppm,outdoor_nh3_ppm,outdoor_h2o_g_m3,
sensor_indoor_temp_c,sensor_indoor_rh_pct,sensor_indoor_wind_m_s,
sensor_indoor_co2_ppm,sensor_indoor_nh3_ppm,sensor_indoor_h2o_g_m3
```

Sensor columns are optional. When present, the simulator blends them into the propagated indoor state as a grey-box correction term.

### New gas chart in offline HTML report (Chart.js)

When `--write-report (or legacy --write-svg)` is used, the report now includes a dedicated gas panel for:

- `CO2 ppm`
- `NH3 ppm`
- `H2O g/m3`

### CLI gas/sensor options

```bash
--outdoor-co2-ppm 420
--outdoor-nh3-ppm 0.2
--outdoor-h2o-g-m3 3.0
--sensor-indoor-temp-c 16.5
--sensor-indoor-rh-pct 72
--sensor-indoor-wind-m-s 0.15
--sensor-indoor-co2-ppm 1650
--sensor-indoor-nh3-ppm 5.8
--sensor-indoor-h2o-g-m3 10.0
```


## Indoor RAD / OKTA / AHA

The simulator now estimates three indoor thermoregulation inputs for enclosed halls:

- `RAD_in,d` in `kJ m^-2 day^-1` from transmitted solar plus indoor lamp radiation over floor area.
- `OKTA_in` as an indoor sky-obstruction surrogate from enclosure openness.
- `AHA_in` as an effective coat-exposure conversion factor from indoor view factor and animal geometry.

These values are stored in the state, exported to `simulation_results.csv`, and shown in the offline HTML report (Chart.js). Optional indoor sensor corrections can be supplied through the outdoor CSV columns `sensor_indoor_rad_kj_m2_day`, `sensor_indoor_okta`, and `sensor_indoor_aha`.


### CSV horizon alignment

If `sample_outdoor.csv` and `sample_controls.csv` do not have the same number of rows, the simulator now runs for the **longer** series and holds the last value of the shorter series constant. This prevents late actuator changes, such as `ventilation_group_pct=100`, from being ignored when the controls CSV is longer than the outdoor CSV.


## Forecast mode from current indoor/outdoor conditions

The simulator now supports horizon forecasting from the current indoor and outdoor state.
This is intended for predictive control and RL rollouts.

Example: predict the next 2 hours with 5-minute steps, using the indoor sensor values only as the initial state seed:

```bash
./beef_climate_sim   --ifc Building_Information/beef_hall_120.ifc   --timestamp 2026-01-01T00:00:00   --sensor-indoor-temp-c 18.4   --sensor-indoor-rh-pct 67   --sensor-indoor-wind-m-s 0.42   --sensor-indoor-co2-ppm 1850   --sensor-indoor-nh3-ppm 12   --sensor-indoor-h2o-g-m3 14.8   --sensor-indoor-rad-kj-m2-day 220   --sensor-indoor-okta 8   --sensor-indoor-aha 1.35   --outdoor-temp-c -6   --outdoor-rh-pct 78   --outdoor-wind-m-s 3.2   --outdoor-solar-w-m2 120   --outdoor-cloud-okta 6   --outdoor-rain-mm-day 0   --ventilation-group 40   --heating-group 55   --light-on 1   --forecast-horizon-seconds 7200   --dt-seconds 300   --output out_forecast   --write-report   --response-stdout   --response-format json
```

Behavior:
- the first indoor sensor sample seeds the initial simulator state
- future steps are model predictions
- indoor sensor measurements are not re-applied after the first forecast step unless they are explicitly passed again as a disturbance series
- when `--response-stdout` is enabled, the simulator prints either CSV or JSON to stdout for downstream RL or orchestration use

## Heterogeneous herd inventory and lightweight cohort processing

This version adds a lightweight heterogeneous-herd pathway for thermoregulation.

Files:
- `configs/herd_inventory.cfg`
- `configs/herd_inventory_processed.cfg`

Inventory line format:
```text
# format: cow=<id>,<breed>,<weight_kg>,<heat_multiplier>
```

A dedicated CLI manages the inventory:
```bash
./herd_inventory_cli --file ../configs/herd_inventory.cfg --reset
./herd_inventory_cli --file ../configs/herd_inventory.cfg --add --id C001 --breed Hereford --weight-kg 500 --heat-multiplier 1.40
./herd_inventory_cli --file ../configs/herd_inventory.cfg --add --id C002 --breed Hereford --weight-kg 511 --heat-multiplier 1.41
./herd_inventory_cli --file ../configs/herd_inventory.cfg --add --id C003 --breed Hereford --weight-kg 508 --heat-multiplier 1.43
./herd_inventory_cli --file ../configs/herd_inventory.cfg --update --id C001 --breed Hereford --weight-kg 520 --heat-multiplier 1.42
./herd_inventory_cli --file ../configs/herd_inventory.cfg --add --id C004 --breed BxS --weight-kg 495 --heat-multiplier 1.40

./herd_inventory_cli --file ../configs/herd_inventory.cfg --processed ../configs/herd_inventory_processed.cfg --process
```

Processing compresses the herd into representative cohorts so the simulator does not need to re-group the full 120-head inventory every rollout. The climate simulator automatically loads `configs/herd_inventory_processed.cfg` when present and uses weighted-quantile herd TNZ bounds:
- operational herd LNZ = weighted quantile 0.90 of cohort LCTs
- operational herd UNZ = weighted quantile 0.10 of cohort UCTs
- safe bounds are also retained using max(LCT) and min(UCT)

You may override the processed file path from the simulator CLI with:
```bash
--herd-processed ../configs/herd_inventory_processed.cfg
```

Update herd_inventory_processed.cfg file witout internal process (directly):
```bash
./herd_inventory_cli \
  --update-file ../configs/herd_inventory_processed.cfg \
  --summary.cattle_count 4 \
  --summary.average_weight_kg 508.5 \
  --summary.average_heat_multiplier 1.415 \
  --summary.cohort_count 2 \
  --cohort BxS,495,1.4,1,3 \
  --cohort Hereford,513,1.42,3,3
```

### Breed-calibrated thermoregulation config

Thermoregulation tunables are now loaded from `configs/thermoregulation.cfg`.
You can override the default path with:

```bash
./beef_climate_sim --thermoregulation-config ../configs/thermoregulation.cfg ...
```

This file contains:
- global thermoregulation tunables (for example `global.body_temp_c`, `global.hif`, `global.body_area_coeff`)
- breed-specific calibration blocks such as `breed.hereford.*`, `breed.brahman.*`, and `breed.charolais.*`

If `herd_inventory_processed.cfg` contains heterogeneous cohorts, each cohort uses the breed-specific thermoregulation constants resolved from `thermoregulation.cfg` before the weighted herd LNZ/UNZ aggregation is computed.


## Python/Cython direct API

This revision adds a Python/Cython execution path so the core simulator can be invoked from Python instead of only through the C++ executable path, while keeping the existing CLI tools in place.
The intended direct API returns `pandas.DataFrame` outputs for single-step and rollout simulation.

Build path:

```bash
python3 -m pip install -U pip setuptools wheel Cython pandas
python3 setup.py build_ext --inplace
```

Planned public functions after build:

- `get_config(...)`
- `save_config(...)`
- `inspect_ifc_file(...)`
- `process_herd_inventory_file(...)`
- `load_processed_herd(...)`
- `run_single_step(...)`
- `run_simulation(...)`

The Cython bridge sources are added under `climate_hall_simulator/` and the helper C++ bridge is added as `include/python_bridge.hpp` and `src/python_bridge.cpp`.


## Python main entrypoint


This package now includes a root-level `main.py` that exposes the documented simulator workflows as separate Python-callable functions and as subcommands.
It first attempts the direct Cython API when the public function exists, and otherwise falls back to the equivalent bundled executable while still returning `pandas.DataFrame` outputs.

Examples:

```bash
python3 main.py build-cython
python3 main.py inspect-ifc
python3 main.py process-herd
python3 main.py run-csv
python3 main.py run-single-step
python3 main.py run-sensor-step
python3 main.py run-forecast
python3 main.py stdout-csv
python3 main.py stdout-json
```

Important public functions defined in `main.py`:

- `build_cython_extension()`
- `inspect_ifc_file()`
- `process_herd_inventory_file()`
- `write_precomputed_processed_herd()`
- `run_csv_mode()`
- `run_single_step()`
- `run_sensor_seeded_step()`
- `run_forecast_mode()`
- `response_stdout_csv_example()`
- `response_stdout_json_example()`
