# Tutorial.md

## Overview

`climate_hall_simulator` is a lightweight C++ hall-climate simulator for the beef hall BIM template in `Building_Information/beef_hall_120.ifc`.
It supports:
- IFC/BIM-driven hall configuration
- grouped actuator control (`ventilation`, `heating`, `light`)
- indoor/outdoor sensor-driven simulation
- forecast mode for future rollout
- heterogeneous herd processing for thermoregulation
- offline HTML reporting with Chart.js
- CSV/JSON stdout responses for orchestration or RL

The main executables are:
- `beef_climate_sim` — run the simulator
- `herd_inventory_cli` — manage the heterogeneous herd inventory
- `ifc_probe` — inspect the IFC/BIM file and extracted properties

---

## Build

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

---

## File layout

Important paths:
- `Building_Information/beef_hall_120.ifc` — IFC/BIM reference hall
- `configs/default_hall.cfg` — default hall and actuator configuration
- `configs/thermoregulation.cfg` — tunable thermoregulation constants by breed
- `configs/herd_inventory.cfg` — raw herd inventory (`cow=<id>,<breed>,<weight_kg>,<heat_multiplier>`)
- `configs/herd_inventory_processed.cfg` — processed herd cohorts for fast runtime use
- `data/sample_outdoor.csv` — outdoor and optional indoor sensor input
- `data/sample_controls.csv` — grouped actuator schedule

---

## 1) IFC inspection with `ifc_probe`

Use this tool to inspect the IFC template and extracted properties.

```bash
./ifc_probe ../Building_Information/beef_hall_120.ifc
```

---

## 2) Run the simulator in CSV mode

This mode reads time series from `sample_outdoor.csv` and `sample_controls.csv`.

```bash
./beef_climate_sim \
  --ifc ../Building_Information/beef_hall_120.ifc \
  --outdoor ../data/sample_outdoor.csv \
  --controls ../data/sample_controls.csv \
  --output out_csv \
  --dt-seconds 300 \
  --write-report
```

Use this mode when you already have time-stamped schedules and boundary conditions for multiple time steps.

---

## 3) Run the simulator in single-step CLI mode

This mode is useful for one-step evaluation from direct CLI inputs.

```bash
./beef_climate_sim \
  --ifc ../Building_Information/beef_hall_120.ifc \
  --timestamp 2026-01-01T00:00:00 \
  --outdoor-temp-c -6 \
  --outdoor-rh-pct 78 \
  --outdoor-wind-m-s 3.2 \
  --outdoor-solar-w-m2 120 \
  --outdoor-cloud-okta 6 \
  --outdoor-rain-mm-day 0 \
  --ventilation-group 40 \
  --heating-group 55 \
  --light-on 1 \
  --output out_cli \
  --write-report
```

---

## 4) Run the simulator with indoor sensor inputs

This mode seeds the simulator with measured indoor values.

```bash
./beef_climate_sim \
  --config ../configs/default_hall.cfg \
  --timestamp 2026-01-01T00:00:00 \
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
  --output out_sensor \
  --write-report
```

---

## 5) Forecast mode for future prediction

This mode uses the current indoor/outdoor conditions and actuator state as the starting point, then predicts into the future.

Example: forecast 2 hours ahead with 5-minute steps.

```bash
./beef_climate_sim \
  --ifc ../Building_Information/beef_hall_120.ifc \
  --timestamp 2026-01-01T00:00:00 \
  --sensor-indoor-temp-c 18.4 \
  --sensor-indoor-rh-pct 67 \
  --sensor-indoor-wind-m-s 0.42 \
  --sensor-indoor-co2-ppm 1850 \
  --sensor-indoor-nh3-ppm 12 \
  --sensor-indoor-h2o-g-m3 14.8 \
  --outdoor-temp-c -6 \
  --outdoor-rh-pct 78 \
  --outdoor-wind-m-s 3.2 \
  --outdoor-solar-w-m2 120 \
  --outdoor-cloud-okta 6 \
  --outdoor-rain-mm-day 0 \
  --ventilation-group 40 \
  --heating-group 55 \
  --light-on 1 \
  --forecast-horizon-seconds 7200 \
  --dt-seconds 300 \
  --output out_forecast \
  --write-report
```

---

## 6) Send simulator response to stdout for orchestration or RL

Use `--response-stdout` to send rollout data to stdout.

### CSV response

```bash
./beef_climate_sim ... --response-stdout --response-format csv
```

### JSON response

```bash
./beef_climate_sim ... --response-stdout --response-format json
```

This is useful for future RL integration and external controllers.

---

## 7) Manage herd inventory with `herd_inventory_cli`

### Reset the raw herd inventory

```bash
./herd_inventory_cli --file ../configs/herd_inventory.cfg --reset
```

### Add one cow

```bash
./herd_inventory_cli \
  --file ../configs/herd_inventory.cfg \
  --add \
  --id C001 \
  --breed Hereford \
  --weight-kg 500 \
  --heat-multiplier 1.40
```

### Update one cow

```bash
./herd_inventory_cli \
  --file ../configs/herd_inventory.cfg \
  --update \
  --id C001 \
  --breed Hereford \
  --weight-kg 520 \
  --heat-multiplier 1.45
```

### Process raw inventory into fast runtime cohorts

```bash
./herd_inventory_cli \
  --file ../configs/herd_inventory.cfg \
  --processed ../configs/herd_inventory_processed.cfg \
  --process
```

The processed file is used by `beef_climate_sim` to avoid repeated herd aggregation work during runtime.

### Write a precomputed processed herd file directly

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

This mode truncates `herd_inventory_processed.cfg`, rewrites it completely with externally precomputed values, and prints the one-word confirmation `updated` on success.

---

## 8) Run the simulator with processed heterogeneous herd data

```bash
./beef_climate_sim \
  --config ../configs/default_hall.cfg \
  --herd-processed ../configs/herd_inventory_processed.cfg \
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
  --output out_herd \
  --dt-seconds 300 \
  --response-stdout \
  --response-format csv
```

---

## 9) Thermoregulation calibration by breed

The file `configs/thermoregulation.cfg` stores thermoregulation constants and breed-specific overrides.

Run the simulator with the explicit file path if needed:

```bash
./beef_climate_sim \
  --config ../configs/default_hall.cfg \
  --thermoregulation-config ../configs/thermoregulation.cfg \
  --outdoor ../data/sample_outdoor.csv \
  --controls ../data/sample_controls.csv \
  --output out_calibrated \
  --dt-seconds 300 \
  --write-report
```

---

## 10) Output files

Typical outputs in the selected output folder:
- `simulation_results.csv` — time series results
- `resolved_config.cfg` — resolved runtime configuration snapshot
- `run_summary.txt` — short run summary
- `simulation_report.html` — offline Chart.js report
- `chart.min.js` — local Chart.js dependency for the offline report

---

## CSV file formats

### `sample_controls.csv`

```csv
timestamp,ventilation_group_pct,heating_group_pct,light_on
2026-01-01T00:00:00,20,42,1
2026-01-01T00:05:00,24,38,1
```

### `sample_outdoor.csv`

Minimum outdoor-only version:

```csv
timestamp,outdoor_temp_c,outdoor_rh_pct,outdoor_wind_m_s,outdoor_solar_w_m2,outdoor_cloud_okta,outdoor_rain_mm_day,outdoor_co2_ppm,outdoor_nh3_ppm,outdoor_h2o_g_m3
2026-01-01T00:00:00,-6,78,3.2,120,6,0,420,1.0,4.2
```

Extended version with optional indoor sensor values:

```csv
timestamp,outdoor_temp_c,outdoor_rh_pct,outdoor_wind_m_s,outdoor_solar_w_m2,outdoor_cloud_okta,outdoor_rain_mm_day,outdoor_co2_ppm,outdoor_nh3_ppm,outdoor_h2o_g_m3,sensor_indoor_temp_c,sensor_indoor_rh_pct,sensor_indoor_wind_m_s,sensor_indoor_co2_ppm,sensor_indoor_nh3_ppm,sensor_indoor_h2o_g_m3,sensor_indoor_rad_kj_m2_day,sensor_indoor_okta,sensor_indoor_aha
2026-01-01T00:00:00,-6,78,3.2,120,6,0,420,1.0,4.2,18.4,67,0.42,1850,12,14.8,220,8,1.35
```

---

## Switch reference

Below, each switch has a one-line explanation.

### `beef_climate_sim`

- `--help` — Print a short usage summary and exit.
- `--config <path>` — Load hall settings from a CFG file instead of relying only on IFC extraction.
- `--ifc <path>` — Load the BIM/IFC hall template used to derive building and equipment properties.
- `--outdoor <path>` — Read time-series boundary conditions and optional indoor sensor values from CSV.
- `--controls <path>` — Read grouped actuator commands from CSV.
- `--herd-processed <path>` — Load preprocessed heterogeneous herd cohorts for fast thermoregulation use.
- `--thermoregulation-config <path>` — Load tunable thermoregulation constants and breed overrides from CFG.
- `--output <dir>` — Write all generated result files into the selected output directory.
- `--dt-seconds <value>` — Set the simulation time step in seconds.
- `--forecast-horizon-seconds <value>` — Predict forward for the requested horizon using the selected time step.
- `--write-report` — Generate the offline HTML report with Chart.js.
- `--write-svg` — Legacy alias that also triggers HTML report generation for compatibility.
- `--response-stdout` — Print the simulated rollout directly to stdout for orchestration or RL integration.
- `--response-format csv|json` — Choose whether stdout response is emitted as CSV or JSON.
- `--timestamp <iso8601>` — Set the starting timestamp label for CLI-driven runs.
- `--outdoor-temp-c <value>` — Set outdoor air temperature from CLI.
- `--outdoor-rh-pct <value>` — Set outdoor relative humidity from CLI.
- `--outdoor-wind-m-s <value>` — Set outdoor wind speed from CLI.
- `--outdoor-solar-w-m2 <value>` — Set outdoor solar input from CLI.
- `--outdoor-cloud-okta <value>` — Set outdoor cloud cover in okta from CLI.
- `--outdoor-rain-mm-day <value>` — Set outdoor rain input from CLI.
- `--outdoor-co2-ppm <value>` — Set outdoor CO2 concentration from CLI.
- `--outdoor-nh3-ppm <value>` — Set outdoor NH3 concentration from CLI.
- `--outdoor-h2o-g-m3 <value>` — Set outdoor water-vapour concentration from CLI.
- `--sensor-indoor-temp-c <value>` — Seed or assimilate indoor air temperature from a sensor.
- `--sensor-indoor-rh-pct <value>` — Seed or assimilate indoor relative humidity from a sensor.
- `--sensor-indoor-wind-m-s <value>` — Seed or assimilate indoor air speed from a sensor.
- `--sensor-indoor-co2-ppm <value>` — Seed or assimilate indoor CO2 from a sensor.
- `--sensor-indoor-nh3-ppm <value>` — Seed or assimilate indoor NH3 from a sensor.
- `--sensor-indoor-h2o-g-m3 <value>` — Seed or assimilate indoor water vapour from a sensor.
- `--sensor-indoor-rad-kj-m2-day <value>` — Seed or assimilate indoor RAD from a sensor or cyber-physical estimator.
- `--sensor-indoor-okta <value>` — Seed or assimilate indoor OKTA surrogate from a sensor or estimator.
- `--sensor-indoor-aha <value>` — Seed or assimilate indoor AHA surrogate from a sensor or estimator.
- `--ventilation-group <value>` — Set grouped ventilation command as a single hall-level percentage.
- `--fan-pairs <list>` — Legacy alias that averages a list of fan-pair values into one grouped ventilation command.
- `--heating-group <value>` — Set grouped heating command as a single hall-level percentage.
- `--heaters <list>` — Legacy alias that averages a list of heater values into one grouped heating command.
- `--light-on 0|1` — Turn the grouped hall lighting command off or on.

### `herd_inventory_cli`

- `--help` — Print herd inventory CLI usage and exit.
- `--file <path>` — Select the raw herd inventory CFG file to edit or process.
- `--processed <path>` — Select the destination CFG file for processed herd cohorts.
- `--update-file <path>` — Rewrite a processed herd CFG directly from externally precomputed summary and cohort values.
- `--reset` — Clear the raw herd inventory file.
- `--add` — Add a new cow entry to the raw herd inventory file.
- `--update` — Update an existing cow entry in the raw herd inventory file.
- `--process` — Convert the raw inventory into processed cohorts for fast runtime use.
- `--summary.cattle_count <int>` — Direct-write summary cattle count when `--update-file` is used.
- `--summary.average_weight_kg <float>` — Direct-write summary average weight when `--update-file` is used.
- `--summary.average_heat_multiplier <float>` — Direct-write summary average heat multiplier when `--update-file` is used.
- `--summary.cohort_count <int>` — Optional compatibility switch for external writers; the file output still uses the supplied cohort entries.
- `--cohort <breed,avg_weight_kg,avg_heat_multiplier,count,breed_library>` — Append one processed cohort row when `--update-file` is used.
- `--id <value>` — Set the cow identifier used by `--add` or `--update`.
- `--breed <value>` — Set the breed used by `--add` or `--update`.
- `--weight-kg <value>` — Set the cow live weight used by `--add` or `--update`.
- `--heat-multiplier <value>` — Set the thermoregulation heat multiplier used by `--add` or `--update`.

---

## Recommended workflow

1. Build the project.
2. Inspect the hall IFC with `ifc_probe`.
3. Prepare or edit `default_hall.cfg` and `thermoregulation.cfg` if needed.
4. Optionally define the herd with `herd_inventory_cli` and run `--process`.
5. Run `beef_climate_sim` in CSV mode, CLI mode, or forecast mode.
6. Check `simulation_results.csv` and open `simulation_report.html` in a browser.
7. Use `--response-stdout --response-format json` when integrating with RL or another controller.



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


## 11) Use the new Python `main.py`


The project now includes a root `main.py` with separate functions for the documented workflows.
Each function is callable from Python, and the same file can also be used as a command-line entrypoint.
Returned simulator results are normalized to `pandas.DataFrame`.

### Build the Cython extension

```bash
python3 main.py build-cython
```

### Inspect the IFC file

```bash
python3 main.py inspect-ifc
```

### Process the herd inventory

```bash
python3 main.py process-herd
```

### Run the simulator in CSV mode

```bash
python3 main.py run-csv
```

### Run one single-step simulation

```bash
python3 main.py run-single-step
```

### Run one indoor-sensor-seeded simulation step

```bash
python3 main.py run-sensor-step
```

### Run forecast mode

```bash
python3 main.py run-forecast
```

### Request stdout responses in CSV or JSON form

```bash
python3 main.py stdout-csv
python3 main.py stdout-json
```
