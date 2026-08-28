from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map as cppmap
from libcpp.utility cimport pair

cdef extern from "include/types.hpp" namespace "beefclimate":
    cdef cppclass HallConfig:
        HallConfig() except +
        double length_m
        double width_m
        double eave_height_m
        double ridge_height_m
        double ridge_opening_m
        double volume_m3
        int pen_count
        int cattle_per_pen
        int cattle_count
        double average_weight_kg
        double wall_u_w_m2k
        double roof_u_w_m2k
        double personnel_door_u_w_m2k
        double service_door_u_w_m2k
        double wall_leak_w_m2k
        double roof_leak_w_m2k
        double window_leak_w_m2k
        double door_leak_w_m2k
        double design_indoor_temp_c
        double design_outdoor_temp_c
        double design_delta_t_c
        double effective_thermal_mass_j_k
        int intake_count
        double intake_width_m
        double intake_height_m
        double intake_discharge_coeff
        double intake_center_z_m
        int fan_count
        double fan_power_w_each
        double fan_flow_m3h_each
        double fan_free_air_flow_m3h_each
        double fan_air_speed_mps_each
        double fan_center_z_m
        int heater_count
        double heater_gas_input_kw_each
        double heater_useful_kw_each
        double heater_airflow_m3h_each
        double heater_center_z_m
        int light_count
        double light_power_w_each
        double light_luminous_flux_lm_each
        double light_visible_fraction
        double light_longwave_fraction
        double theta_ua
        double theta_cap
        double theta_vent
        double theta_cattle
        double theta_humidity
        double theta_gas
        double theta_light
        double theta_heat
        double initial_indoor_temp_c
        double initial_indoor_rh_pct
        double initial_gas_index
        double initial_air_speed_m_s
        double initial_radiation_w
        double initial_co2_ppm
        double initial_nh3_ppm
        double initial_h2o_g_m3
        double effective_transmitting_area_m2
        double envelope_solar_transmittance
        double indoor_view_factor
    cdef cppclass Disturbance:
        Disturbance() except +
        string timestamp
        double outdoor_temp_c
        double outdoor_rh_pct
        double outdoor_wind_m_s
        double outdoor_solar_w_m2
        double outdoor_cloud_okta
        double outdoor_rain_mm_day
        double outdoor_co2_ppm
        double outdoor_nh3_ppm
        double outdoor_h2o_g_m3
        double sensor_indoor_temp_c
        double sensor_indoor_rh_pct
        double sensor_indoor_wind_m_s
        double sensor_indoor_co2_ppm
        double sensor_indoor_nh3_ppm
        double sensor_indoor_h2o_g_m3
        double sensor_indoor_rad_kj_m2_day
        double sensor_indoor_okta
        double sensor_indoor_aha
    cdef cppclass Control:
        Control() except +
        string timestamp
        double ventilation_group_pct
        double heating_group_pct
        int light_on
    cdef cppclass ThermoregulationState:
        ThermoregulationState() except +
        double lower_critical_c
        double upper_critical_c
        double herd_safe_lower_c
        double herd_safe_upper_c
        int herd_cohort_count
        double animal_heat_threshold_w_m2
        double min_heat_release_w_m2
        double max_heat_release_w_m2
        double skin_temp_min_c
        double skin_temp_max_c
        double total_body_weight_kg
        double heat_production_multiplier
    cdef cppclass LayerContributions:
        LayerContributions() except +
        double cattle_sensible_w
        double cattle_latent_kg_s
        double cattle_gas_index_per_h
        double cattle_co2_ppm_per_h
        double cattle_nh3_ppm_per_h
        double cattle_h2o_g_m3_per_h
        double heater_useful_w
        double heater_fuel_w
        double light_radiant_w
        double light_convective_w
        double envelope_loss_w
        double ventilation_loss_w
        double solar_gain_w
        double infiltration_flow_m3_s
        double mechanical_flow_m3_s
        double total_flow_m3_s
        double air_speed_m_s
        double fan_power_w
        double light_power_w
    cdef cppclass State:
        State() except +
        double indoor_temp_c
        double indoor_rh_pct
        double gas_index
        double air_speed_m_s
        double internal_heat_w
        double indoor_radiation_w
        double humidity_ratio_kgkg
        double mass_temperature_c
        double cattle_heat_w
        double lamp_heat_w
        double generated_moisture_kg_s
        double generated_gas_index_per_h
        double co2_ppm
        double nh3_ppm
        double h2o_g_m3
        double indoor_rad_kj_m2_day
        double indoor_okta
        double indoor_aha
        double cumulative_fan_energy_kwh
        double cumulative_heater_energy_kwh
        double cumulative_light_energy_kwh
    cdef cppclass OutputMetrics:
        OutputMetrics() except +
        double indoor_temp_c
        double indoor_rh_pct
        double gas_index
        double co2_ppm
        double nh3_ppm
        double h2o_g_m3
        double indoor_rad_kj_m2_day
        double indoor_okta
        double indoor_aha
        double air_speed_m_s
        double fan_power_w
        double heater_power_w
        double light_power_w
        double comfort_violation
        double air_quality_violation
        double energy_cost
        double reward
    cdef cppclass StepResult:
        StepResult() except +
        string timestamp
        State state
        Control control
        Disturbance disturbance
        LayerContributions layers
        ThermoregulationState thermoregulation
        OutputMetrics outputs
        bint done
    ctypedef vector[Disturbance] DisturbanceSeries
    ctypedef vector[Control] ControlSeries
    ctypedef vector[StepResult] SimulationHistory

cdef extern from "include/herd_inventory.hpp" namespace "beefclimate":
    cdef cppclass HerdCohort:
        HerdCohort() except +
        string breed
        double avg_weight_kg
        double avg_heat_multiplier
        int count
        int breed_library
    cdef cppclass HerdProcessedSummary:
        HerdProcessedSummary() except +
        vector[HerdCohort] cohorts
        int cattle_count
        double average_weight_kg
        double average_heat_multiplier
    HerdProcessedSummary process_herd_inventory(const string& input_path, const string& output_path) except +
    HerdProcessedSummary load_herd_processed_cfg(const string& path) except +

cdef extern from "include/ifc_reader.hpp" namespace "beefclimate":
    cdef cppclass IfcValidationSummary:
        IfcValidationSummary() except +
        string parser_mode
        string schema_name
        int window_count
        int fan_count
        int heater_count
        int light_count
        cppmap[string, int] property_sets
        cppmap[string, double] numeric_properties
    HallConfig load_config_from_ifc(const string& ifc_path) except +
    IfcValidationSummary inspect_ifc(const string& ifc_path) except +
    bint ifcopenshell_cxx_enabled()
    string ifcopenshell_validation_status()

cdef extern from "include/config.hpp" namespace "beefclimate":
    HallConfig load_config_file(const string& path) except +
    void save_config_file(const HallConfig& cfg, const string& path) except +

cdef extern from "include/csv.hpp" namespace "beefclimate":
    DisturbanceSeries load_disturbances_csv(const string& path) except +
    ControlSeries load_controls_csv(const string& path, int fan_pair_count_hint, int heater_count_hint) except +
    void save_results_csv(const SimulationHistory& history, const string& path) except +

cdef extern from "include/report.hpp" namespace "beefclimate":
    void write_html_report(const SimulationHistory& history, const HallConfig& cfg, const string& path) except +

cdef extern from "include/thermoregulation_config.hpp" namespace "beefclimate":
    cdef cppclass ThermoregulationModelConfig:
        ThermoregulationModelConfig() except +
    ThermoregulationModelConfig load_thermoregulation_config_file(const string& path) except +
    void set_active_thermoregulation_config(ThermoregulationModelConfig cfg)

cdef extern from "include/python_bridge.hpp" namespace "beefclimate":
    vector[pair[string,int]] ifc_property_sets_items(const IfcValidationSummary& s) except +
    vector[pair[string,double]] ifc_numeric_properties_items(const IfcValidationSummary& s) except +

cdef extern from "include/simulator.hpp" namespace "beefclimate":
    cdef cppclass ClimateSimulator:
        ClimateSimulator(HallConfig cfg) except +
        State initialize()
        State reset()
        State reset(const State& initial_state)
        StepResult propagate(const Disturbance& disturbance, const Control& control, double dt_seconds)
        SimulationHistory rollout(const DisturbanceSeries& disturbances, const ControlSeries& controls, double dt_seconds)
        void set_processed_herd(const HerdProcessedSummary& herd)
