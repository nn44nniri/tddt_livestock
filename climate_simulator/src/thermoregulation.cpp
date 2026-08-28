#include "thermoregulation.hpp"
#include "thermoregulation_config.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace beefclimate {
namespace {

static constexpr double PI = 3.14159265;
static constexpr double P = 101325.0;
static constexpr double RDAIR = 287.058;
static constexpr double RWATER = 461.495;
static constexpr double CTOK = 273.15;
static constexpr double CP = 1.005;
static constexpr double L = 2260.0;
static constexpr double GRAV = 9.81;
static constexpr double SIGMA = 5.67037e-8;
static constexpr double KJDAY_TO_W = 1000.0 / (3600.0 * 24.0);
static constexpr double KTO_R = 9.0 / 5.0;

struct HeatPair {
    double min_heat_w_m2 = 0.0;
    double max_heat_w_m2 = 0.0;
    double skin_t_min_c = 0.0;
    double skin_t_max_c = 0.0;
};

double sat_vapor_pressure_pa(double temp_c) {
    return 6.1078 * std::pow(10.0, (7.5 * temp_c) / (temp_c + 237.3)) * 100.0;
}

HeatPair evaluate_at_temperature(const HallConfig& cfg,
                                 double air_temp_c,
                                 double rh_pct,
                                 double wind_m_s,
                                 double rad_kj_m2_day,
                                 double cloud_okta,
                                 double rain_mm_day,
                                 double aha,
                                 double housing_mode,
                                 const ThermoregulationBreedParams& breed,
                                 double body_weight_kg,
                                 double heat_multiplier) {
    const auto& model = active_thermoregulation_config();

    const double REFLC = breed.reflectance;
    const double LC = breed.coat_depth_m;
    const double AREAFACTOR = breed.area_factor;
    const double CBSMAX = breed.cbs_max;
    const double RBCSf = breed.rbcsf;

    const double TAVGK = CTOK + air_temp_c;
    const double wind = std::max(0.1, wind_m_s);
    const double rh = std::clamp(rh_pct, 10.0, 100.0);
    const double swrad = std::max(0.0, rad_kj_m2_day);
    const double octa = std::clamp(cloud_okta, 0.0, 8.0);
    const double rain = std::max(0.0, rain_mm_day);
    const double ahadata = std::clamp(aha, 0.0, 5.0);

    const double VPSATAIR = sat_vapor_pressure_pa(air_temp_c);
    const double VPAIRTOT = VPSATAIR * rh / 100.0;
    const double RHAIR_FRACTION = (VPSATAIR > 0.0) ? VPAIRTOT / VPSATAIR : 0.0;
    const double RHOVP = VPAIRTOT / (RWATER * TAVGK);
    const double RHODAIR = (P - VPAIRTOT) / (RDAIR * TAVGK);
    const double RHOAIR = RHOVP + RHODAIR;
    const double CHIAIR = RHOVP * RHOAIR;

    const double tbw = std::max(50.0, body_weight_kg > 0.0 ? body_weight_kg : cfg.average_weight_kg);
    const double AREA = model.body_area_coeff * std::pow(tbw, model.body_area_exp) * AREAFACTOR;
    const double DIAMETER = 0.06 * std::pow(tbw, 0.39);
    const double LENGTH = (AREA - PI * DIAMETER * DIAMETER / 2.0) / (PI * DIAMETER);

    const double brr = 73.8 * std::pow(tbw, -0.286);
    const double btv = 0.0117 * tbw;
    const double brv = brr * btv;

    double panting = 0.25;
    double irv = brv + panting * ((model.respiration_increase_factor - 1.0) * brv);
    const double Texh = 17.0 + 0.3 * air_temp_c + std::exp(0.01611 * RHAIR_FRACTION + 0.0387 * air_temp_c);
    const double VPSATAIROUT = sat_vapor_pressure_pa(Texh);
    const double RHOVPOUT = VPSATAIROUT / (RWATER * (Texh + CTOK));
    const double RHODAIROUT = (P - VPSATAIROUT) / (RDAIR * (Texh + CTOK));
    const double RHOAIROUT = RHOVPOUT + RHODAIROUT;
    const double CHIAIROUT = RHOVPOUT * RHOAIROUT;

    double AIREXCH = (irv * 60.0 * 24.0 / 1000.0 * RHOAIR) / AREA;
    double LHEATRESP = AIREXCH * L * (CHIAIROUT - CHIAIR) * KJDAY_TO_W;
    double CHEATRESP = AIREXCH * CP * (Texh - air_temp_c) * KJDAY_TO_W;
    double TGRESP = LHEATRESP + CHEATRESP;
    double NERESPWM = 1.1 * std::pow(model.respiration_increase_factor * brr, 2.78) * 1e-5 * panting;
    double TNRESP = TGRESP - NERESPWM;

    const double CBSMIN = RBCSf / (0.03 * std::pow(tbw, 0.33));
    double tissueFrac = 1.0;
    double CONDBS = CBSMIN + tissueFrac * (CBSMAX - CBSMIN);

    const double DLC = (model.coat_const * wind) / (((model.coat_const * wind) / LC) + 1.0 / (model.zc * LC));
    const double DIFFC = 0.187e-9 * std::pow(TAVGK, 2.072);
    double rainReduction = 1.0 - std::min(model.rainfrac, rain * model.rainfrac / 24.0);
    if (rainReduction <= 1e-9) rainReduction = 1e-9;
    const double CSC = (1.0 / (model.zc * (LC - DLC) * (0.078 / 100.0))) / rainReduction;

    double LWRSKY = (1.0 - octa / 8.0) * (SIGMA * std::pow(TAVGK, 4.0)) *
                    (1.0 - 0.261 * std::exp(-0.000777 * std::pow(273.0 - TAVGK, 2.0))) +
                    (octa / 8.0) * (SIGMA * std::pow(TAVGK, 4.0) - 9.0);
    const double LWRENV = SIGMA * std::pow(TAVGK, 4.0);
    if (housing_mode == 0.0) LWRSKY = LWRENV;

    const double TAVGR = TAVGK * KTO_R;
    const double VISCAIR = model.mu_st * (((0.555 * model.tr0 + model.cconv1) / (0.555 * TAVGR + model.cconv1)) * std::pow(TAVGR / model.tr0, 1.5));
    const double Ea = VPAIRTOT * 10.0;
    const double REYNOLDS = wind * DIAMETER * RHOAIR / VISCAIR;
    const double ReH = 16.0 * REYNOLDS * REYNOLDS;
    const double ReL = 0.1 * REYNOLDS * REYNOLDS;
    const double ka = 1.5207e-11 * std::pow(TAVGK, 3.0) - 4.8574e-8 * std::pow(TAVGK, 2.0) +
                      1.0184e-4 * TAVGK - 0.00039333;

    const double SAAC = ahadata;
    const double SWRS = swrad * 1000.0 / (3600.0 * 24.0);
    const double SWRC = SWRS * SAAC * (1.0 - REFLC);
    double REFLE = 0.0;
    if (housing_mode == 1.0) REFLE = model.refle_grass;
    else if (housing_mode == 2.0) REFLE = model.refle_concrete;
    const double ISWRC = 0.5 * REFLE * swrad * 1000.0 / (3600.0 * 24.0);
    const double SWR = SWRC + ISWRC;
    const double RAINEVAP = 0.15 * (LENGTH * DIAMETER) / AREA * std::min(24.0, rain) * L * KJDAY_TO_W;

    auto coat_conv = [&](double TcoatC) {
        const double Ec = (sat_vapor_pressure_pa(TcoatC) / 100.0 + Ea) / 2.0;
        const double GRASHOF = (GRAV * std::pow(DIAMETER, 3.0) * P / 100.0 * (TcoatC - air_temp_c) +
                                model.schmidt * (Ec * TcoatC - Ea * air_temp_c)) /
                               (273.0 * P / 100.0 * VISCAIR * VISCAIR);
        double NUSSELT = 0.0;
        if (GRASHOF > ReH) NUSSELT = 0.48 * std::pow(std::max(0.0, GRASHOF), 0.25);
        else if (GRASHOF < ReL) NUSSELT = 0.0112 * std::pow(std::max(0.0, REYNOLDS), 0.875);
        else NUSSELT = std::max(0.48 * std::pow(std::max(0.0, GRASHOF), 0.25), 0.0112 * std::pow(std::max(0.0, REYNOLDS), 0.875));
        return (ka * NUSSELT) / DIAMETER * (TcoatC - air_temp_c) / rainReduction;
    };

    double METABFEED = 170.0;
    double Metheatopt = METABFEED;
    double TskinC_max = model.body_temp_c;
    for (int iter = 0; iter < 50000; ++iter) {
        const double MetheatSKIN = METABFEED - TNRESP;
        const double TskinC = model.body_temp_c - MetheatSKIN / CONDBS;
        const double LASMAXPHYS = model.lasmin_w_m2 + breed.latent_a * std::exp(breed.latent_b * (TskinC - breed.latent_ref_temp_c)) * L / 3600.0;
        const double deltaT = TskinC - std::min(air_temp_c, TskinC);
        const double RV = (LC - DLC) / (DIFFC * (1.0 + 1.54 * ((LC - DLC) / DIAMETER) * std::pow(std::max(0.0, deltaT), 0.7)));
        const double VPSKINTOT = sat_vapor_pressure_pa(TskinC);
        const double LASMAXENV = (RHOAIR * CP * 1000.0 / model.gamma) * (VPSKINTOT - VPAIRTOT) / RV;
        const double LASMAXCORR = std::min(LASMAXPHYS, LASMAXENV);
        const double ACTSW = model.lasmin_w_m2 + 1.0 * (LASMAXCORR - model.lasmin_w_m2);
        const double MetheatCOAT = MetheatSKIN - ACTSW;
        const double TcoatC = TskinC - MetheatCOAT / CSC;
        const double TcoatK = TcoatC + CTOK;
        const double LB = model.emissivity * SIGMA * std::pow(TcoatK, 4.0);
        const double LWRCOAT = (model.emissivity * ((LWRSKY + LWRENV) / 2.0) - LB) / rainReduction;
        const double CONVCOAT = coat_conv(TcoatC);
        const double MetheatAIR = (MetheatCOAT + SWR - RAINEVAP + LWRCOAT - CONVCOAT);
        METABFEED -= 0.01 * MetheatAIR;
        Metheatopt = METABFEED;
        TskinC_max = TskinC;
        if (MetheatAIR < 1.0 && MetheatAIR > -1.0) break;
    }

    tissueFrac = 0.0;
    panting = 0.0;
    CONDBS = CBSMIN;
    irv = brv + panting * (6.64 * brv);
    AIREXCH = (irv * 60.0 * 24.0 / 1000.0 * RHOAIR) / AREA;
    LHEATRESP = AIREXCH * L * (CHIAIROUT - CHIAIR) * KJDAY_TO_W;
    CHEATRESP = AIREXCH * CP * (Texh - air_temp_c) * KJDAY_TO_W;
    TGRESP = LHEATRESP + CHEATRESP;
    TNRESP = TGRESP;

    double METABFEEDC = 80.0;
    double Metheatcold = METABFEEDC;
    double TskinC_min = model.body_temp_c;
    for (int iter = 0; iter < 50000; ++iter) {
        const double MetheatSKIN = METABFEEDC - TNRESP;
        const double TskinC = model.body_temp_c - MetheatSKIN / CONDBS;
        const double ACTSW = model.lasmin_w_m2;
        const double MetheatCOAT = MetheatSKIN - ACTSW;
        const double TcoatC = TskinC - MetheatCOAT / CSC;
        const double TcoatK = TcoatC + CTOK;
        const double LB = SIGMA * std::pow(TcoatK, 4.0);
        const double LWRCOAT = (model.emissivity * ((LWRSKY + LWRENV) / 2.0) - model.emissivity * LB) / rainReduction;
        const double CONVCOAT = coat_conv(TcoatC);
        const double MetheatAIR = (MetheatCOAT + SWR - RAINEVAP + LWRCOAT - CONVCOAT);
        METABFEEDC -= 0.01 * MetheatAIR;
        Metheatcold = METABFEEDC;
        TskinC_min = TskinC;
        if (MetheatAIR < 1.0 && MetheatAIR > -1.0) break;
    }

    return HeatPair{Metheatcold, Metheatopt, TskinC_min, TskinC_max};
}

}  // namespace

ThermoregulationState evaluate_thermoregulation(const HallConfig& cfg,
                                                double ambient_temp_c,
                                                double relative_humidity_pct,
                                                double wind_m_s,
                                                double rad_kj_m2_day,
                                                double cloud_okta,
                                                double rain_mm_day,
                                                double aha,
                                                double housing_mode,
                                                int breed_library,
                                                double body_weight_kg,
                                                double heat_multiplier,
                                                const std::string& breed_name) {
    ThermoregulationState out;
    const auto& model = active_thermoregulation_config();
    const auto& breed = resolve_thermoregulation_breed(breed_name, breed_library);
    const double effective_weight = body_weight_kg > 0.0 ? body_weight_kg : cfg.average_weight_kg;
    out.total_body_weight_kg = effective_weight;
    out.heat_production_multiplier = (heat_multiplier > 0.0 ? heat_multiplier : 1.36) * cfg.theta_cattle;

    const HeatPair current = evaluate_at_temperature(cfg, ambient_temp_c, relative_humidity_pct, wind_m_s, rad_kj_m2_day,
                                                     cloud_okta, rain_mm_day, aha, housing_mode, breed, effective_weight, out.heat_production_multiplier);
    out.min_heat_release_w_m2 = current.min_heat_w_m2;
    out.max_heat_release_w_m2 = current.max_heat_w_m2;
    out.skin_temp_min_c = current.skin_t_min_c;
    out.skin_temp_max_c = current.skin_t_max_c;

    const double METTBWACT = std::pow(std::max(50.0, effective_weight) * model.metab_weight_factor, 0.75);
    const double MAINTME = model.maintenance_me_coeff * METTBWACT;
    const double TOTNE = MAINTME + MAINTME * (out.heat_production_multiplier - 1.0);
    const double area = model.body_area_coeff * std::pow(std::max(50.0, effective_weight), model.body_area_exp) * breed.area_factor;
    out.animal_heat_threshold_w_m2 = TOTNE * (1.0 + model.hif / (1.0 - model.hif)) / area * 1000.0 / (3600.0 * 24.0);

    out.lower_critical_c = -40.0;
    out.upper_critical_c = 40.0;
    bool seen_low = false;
    bool seen_high = false;
    for (double t = -40.0; t <= 40.0; t += 0.25) {
        const HeatPair hp = evaluate_at_temperature(cfg, t, relative_humidity_pct, wind_m_s, rad_kj_m2_day,
                                                    cloud_okta, rain_mm_day, aha, housing_mode, breed, effective_weight, out.heat_production_multiplier);
        const bool below = hp.min_heat_w_m2 > out.animal_heat_threshold_w_m2;
        const bool above = hp.max_heat_w_m2 < out.animal_heat_threshold_w_m2;
        const bool in_tnz = !below && !above;
        if (in_tnz && !seen_low) {
            out.lower_critical_c = t;
            seen_low = true;
        }
        if (seen_low && in_tnz) {
            out.upper_critical_c = t;
            seen_high = true;
        }
    }
    if (!seen_low) out.lower_critical_c = ambient_temp_c;
    if (!seen_high) out.upper_critical_c = ambient_temp_c;

    if (ambient_temp_c < out.lower_critical_c) out.zone = ThermalZoneClass::BelowTNZ;
    else if (ambient_temp_c > out.upper_critical_c) out.zone = ThermalZoneClass::AboveTNZ;
    else out.zone = ThermalZoneClass::InTNZ;

    return out;
}

namespace {

double weighted_quantile(std::vector<std::pair<double,double>> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
    double total = 0.0;
    for (const auto& v : values) total += v.second;
    const double target = std::clamp(q, 0.0, 1.0) * total;
    double acc = 0.0;
    for (const auto& v : values) {
        acc += v.second;
        if (acc >= target) return v.first;
    }
    return values.back().first;
}

}

ThermoregulationState evaluate_heterogeneous_thermoregulation(const HallConfig& cfg,
                                                              const HerdProcessedSummary& herd,
                                                              double ambient_temp_c,
                                                              double relative_humidity_pct,
                                                              double wind_m_s,
                                                              double rad_kj_m2_day,
                                                              double cloud_okta,
                                                              double rain_mm_day,
                                                              double aha,
                                                              double housing_mode) {
    if (herd.cohorts.empty()) {
        return evaluate_thermoregulation(cfg, ambient_temp_c, relative_humidity_pct, wind_m_s, rad_kj_m2_day, cloud_okta, rain_mm_day, aha, housing_mode);
    }

    std::vector<std::pair<double,double>> lcts;
    std::vector<std::pair<double,double>> ucts;
    double sum_min = 0.0, sum_max = 0.0, sum_thresh = 0.0, sum_weight = 0.0;
    ThermoregulationState out;
    out.herd_cohort_count = static_cast<int>(herd.cohorts.size());
    out.total_body_weight_kg = herd.average_weight_kg;
    out.heat_production_multiplier = herd.average_heat_multiplier;
    out.herd_safe_lower_c = -1e9;
    out.herd_safe_upper_c = 1e9;
    for (const auto& c : herd.cohorts) {
        auto t = evaluate_thermoregulation(cfg, ambient_temp_c, relative_humidity_pct, wind_m_s, rad_kj_m2_day, cloud_okta, rain_mm_day, aha, housing_mode, c.breed_library, c.avg_weight_kg, c.avg_heat_multiplier, c.breed);
        const double w = static_cast<double>(std::max(1, c.count));
        lcts.push_back({t.lower_critical_c, w});
        ucts.push_back({t.upper_critical_c, w});
        sum_min += w * t.min_heat_release_w_m2;
        sum_max += w * t.max_heat_release_w_m2;
        sum_thresh += w * t.animal_heat_threshold_w_m2;
        sum_weight += w;
        out.herd_safe_lower_c = std::max(out.herd_safe_lower_c, t.lower_critical_c);
        out.herd_safe_upper_c = std::min(out.herd_safe_upper_c, t.upper_critical_c);
    }
    out.lower_critical_c = weighted_quantile(lcts, 0.90);
    out.upper_critical_c = weighted_quantile(ucts, 0.10);
    out.min_heat_release_w_m2 = sum_min / sum_weight;
    out.max_heat_release_w_m2 = sum_max / sum_weight;
    out.animal_heat_threshold_w_m2 = sum_thresh / sum_weight;
    if (ambient_temp_c < out.lower_critical_c) out.zone = ThermalZoneClass::BelowTNZ;
    else if (ambient_temp_c > out.upper_critical_c) out.zone = ThermalZoneClass::AboveTNZ;
    else out.zone = ThermalZoneClass::InTNZ;
    return out;
}


}  // namespace beefclimate
