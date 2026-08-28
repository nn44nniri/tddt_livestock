#pragma once

#include <algorithm>
#include <cmath>

namespace beefclimate::psychrometrics {

inline double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(v, hi));
}

/*
Reference:
  Bring, A., Sahlin, P., and Vuolle, M. (1999).
  "Models for Building Indoor Climate and Energy Simulation",
  IEA SHC Task 22, Section 4.2 Humidity, page 12.

Purpose in this library:
  Compute the saturated water-vapour pressure p_sat(T) [Pa] needed by the
  indoor humidity model. This is used before converting relative humidity
  to humidity ratio and when reconstructing RH from humidity ratio.

Formula numbers in the reference:
  - Eq. (5): ln(P_sat) for air temperature below 0 °C
  - Eq. (6): ln(P_sat) for air temperature above 0 °C

Variables from the reference:
  - T     : air temperature [°C] in the article description, converted here
            to absolute temperature [K] as T = temp_c + 273.15 because the
            polynomial/exponential form is evaluated in Kelvin.
  - P_sat : saturated vapour pressure [Pa]
  - C1..C13 : empirical coefficients listed in the reference table.

How it is used here:
  The simulator calls this function as the psychrometric core of the indoor
  moisture balance. It is a direct implementation of the ASHRAE Toolkit form
  reproduced by Bring et al. in Section 4.2.
*/
inline double saturation_pressure_pa(double temp_c) {
    if (temp_c < 0.0) {
        const double T = temp_c + 273.15;
        return std::exp(
            /* Eq. (5) in Bring et al. (1999), Section 4.2, for T < 0 °C:
               ln(P_sat) = C1/T + C2 + C3*T + C4*T^2 + C5*T^3 + C6*T^4 + C7*ln(T)
               where:
                 - P_sat = saturated vapour pressure [Pa]
                 - T     = absolute air temperature [K]
               Use in simulator:
                 - humidity conversion for cold-weather barn conditions. */
            -5674.5359 / T + 6.3925247 - 0.009677843 * T + 0.00000062215701 * T * T +
            0.0000000020747825 * T * T * T - 0.0000000000009484024 * T * T * T * T +
            4.1635019 * std::log(T));
    }
    const double T = temp_c + 273.15;
    return std::exp(
        /* Eq. (6) in Bring et al. (1999), Section 4.2, for T >= 0 °C:
           ln(P_sat) = C8/T + C9 + C10*T + C11*T^2 + C12*T^3 + C13*ln(T)
           where:
             - P_sat = saturated vapour pressure [Pa]
             - T     = absolute air temperature [K]
           Use in simulator:
             - humidity conversion for typical indoor and mild outdoor conditions. */
        -5800.2206 / T + 1.3914993 - 0.04860239 * T + 0.000041764768 * T * T -
        0.000000014452093 * T * T * T + 6.5459673 * std::log(T));
}

/*
Reference:
  Bring, A., Sahlin, P., and Vuolle, M. (1999), Section 4.2 Humidity.

Purpose in this library:
  Convert air temperature and relative humidity to humidity ratio w [kg/kg dry air]
  for the indoor and outdoor air-mixing calculations.

Formula numbers in the reference:
  - Eq. (2): P_sat = SatPres(T_air)
  - Eq. (3): P_vap = P_sat * RelHum
  - Eq. (4): HumAir = HumRat(P_air, P_vap)
  - Eq. (7): HumRat = 0.62198 * p_vap / (p - p_vap)

Variables from the reference:
  - T_air   : air temperature [°C]
  - RelHum  : relative humidity [- or %]
  - P_sat   : saturated vapour pressure [Pa]
  - P_vap   : partial vapour pressure [Pa]
  - p       : atmospheric pressure [Pa]
  - HumAir  : humidity ratio [kg H2O / kg dry air]

How it is used here:
  This is the main entry point for initializing and updating moisture states in
  the hall simulator. The code applies Eqs. (2)-(4) and Eq. (7) in sequence.
*/
inline double humidity_ratio_from_rh(double temp_c, double rh_pct, double pressure_pa = 101325.0) {
    /* Eq. (2): P_sat = SatPres(T_air)
       Here: p_sat [Pa] = saturation_pressure_pa(temp_c). */
    const double p_sat = saturation_pressure_pa(temp_c);

    /* Eq. (3): P_vap = P_sat * RelHum
       Here RelHum is provided in percent, so it is first converted to [0,1]. */
    const double p_v = clamp(rh_pct / 100.0, 0.0, 1.0) * p_sat;

    /* Eq. (4) + explicit HumRat expression Eq. (7):
         HumAir = HumRat(p, p_vap) = 0.62198 * p_vap / (p - p_vap)
       where:
         - HumAir = humidity ratio [kg/kg dry air]
         - p      = atmospheric pressure [Pa]
         - p_vap  = partial vapour pressure [Pa]
       Use in simulator:
         - outdoor-to-indoor humidity mixing,
         - initialization of indoor moisture state,
         - conversion of user-provided RH forcing into mass-based humidity. */
    return 0.62198 * p_v / std::max(1.0, pressure_pa - p_v);
}

/*
Reference basis:
  This function is the algebraic inverse of Eq. (7) from Bring et al. (1999),
  Section 4.2 Humidity, combined with Eq. (3).

Purpose in this library:
  Reconstruct relative humidity [%] from the simulator's internal humidity-ratio
  state after the moisture balance has been advanced.

Formula basis:
  Starting from Eq. (7):
    w = 0.62198 * p_v / (p - p_v)
  solve for p_v:
    p_v = p * w / (0.62198 + w)
  then using Eq. (3):
    RH = 100 * p_v / p_sat

Variables:
  - w       : humidity ratio [kg/kg dry air]
  - p_v     : partial vapour pressure [Pa]
  - p       : atmospheric pressure [Pa]
  - p_sat   : saturated vapour pressure [Pa]
  - RH      : relative humidity [%]

Use in simulator:
  Convert the mass-based moisture state back into the human-readable indoor RH
  value used in outputs, constraints, and plots.
*/
inline double rh_from_humidity_ratio(double temp_c, double humidity_ratio, double pressure_pa = 101325.0) {
    const double p_sat = saturation_pressure_pa(temp_c);
    const double p_v = pressure_pa * humidity_ratio / (0.62198 + humidity_ratio);
    return clamp(100.0 * p_v / std::max(1.0, p_sat), 0.0, 100.0);
}

inline double air_density_kg_m3(double temp_c, double pressure_pa = 101325.0) {
    return pressure_pa / (287.058 * (temp_c + 273.15));
}

inline double moist_air_cp_j_kgk() {
    return 1005.0;
}

}  // namespace beefclimate::psychrometrics
