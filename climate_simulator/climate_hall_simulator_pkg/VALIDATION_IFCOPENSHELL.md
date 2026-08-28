# IfcOpenShell integration validation report

## Environment result

This package was rebuilt and tested in the current environment.

### Validated successfully

- `cmake ..`
- `cmake --build . -j`
- `ctest --output-on-failure`
- `ifc_probe Building_Information/beef_hall_120.ifc`
- `beef_climate_sim` in CSV mode
- `beef_climate_sim` in CLI mode

### Test result summary

All three tests passed:

- `smoke_ifc_probe`
- `smoke_simulation_csv_mode`
- `smoke_simulation_cli_mode`

## Official IfcOpenShell C++ path

The source tree and CMake are prepared for the official IfcOpenShell C++ integration:

- `BEEFCLIMATE_WITH_IFCOPENSHELL=ON`
- `BEEFCLIMATE_IFCOPENSHELL_ROOT=/path/to/install`
- `FindIfcOpenShell.cmake`
- hard failure when the official package is requested but missing

## Important limitation in this environment

The official IfcOpenShell C++ development package was **not installed in this environment**.

Because of that, the following could **not** be runtime-validated here:

- linking against the real official `IfcParse` library
- compiling the typed schema traversal path against local official headers/libs
- running the executable with `BEEFCLIMATE_WITH_IFCOPENSHELL=ON`

## What was validated instead

The IFC loader itself was upgraded from a regex-based approach to a relationship-aware STEP graph extractor. This extractor now reads:

- STEP entities by id
- IFC types
- `IFCPROPERTYSET`
- `IFCPROPERTYSINGLEVALUE`
- `IFCRELDEFINESBYPROPERTIES`
- object-to-property relations

This path is fully validated here and is the active path for this environment.

## Probe output summary

Observed from the uploaded `beef_hall_120.ifc`:

- parser mode: `STEP-graph-fallback`
- schema: `IFC4`
- windows: `36`
- fans: `28`
- heaters: `6`
- lights: `36`
- `BuildingLength_m = 48`
- `BuildingWidth_m = 35`
- `ApproxEnclosedVolume_m3 = 10500`
- `ElectricalDemand_W = 370`
- `UsefulHeatOutput_kW = 35.17`
- `LuminousFlux_lm = 6000`

## Practical conclusion

This package is ready for official IfcOpenShell C++ use, but the final runtime proof of that path requires an environment where the official development package is actually present.
