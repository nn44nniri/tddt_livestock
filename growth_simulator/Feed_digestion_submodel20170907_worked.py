"""
#######################################################################################
#                                                                                     #
# LiGAPS-Beef (Livestock simulator for Generic analysis of Animal Production Systems) #
#                                                                                     #
# Feed intake and digestion sub-model                                                 #
#                                                                                     #
# Direct Python conversion from the provided R code                                   #
# Logic preserved as closely as possible                                              #
#                                                                                     #
#######################################################################################
"""

import math
import numpy as np

# Feed parameters related to heat generation, digestion (Chilibroste et al, 1997)
# and fill units (Jarrige, 1986)

# Abbreviations
# HIF = Heat Increment of feeding
# FU = Fill Units
# SNSC = Soluble, Non-Structural Carbohydrates (g kg-1 DM)
# INSC = Insoluble, Non-Structural Carbohydrates (g kg-1 DM)
# DNDF = Digestible Neutral Detergent Fibre (g kg-1 DM)
# SCP = Soluble Crude Protein (g kg-1 DM)
# DCP = Digestible Crude Protein (g kg-1 DM)
# kdINSC = digestion rate Insoluble, Non-Structural Carbohydrates (% hr-1)
# kdNDF = digestion rate Neutral Detergent Fibre (% hr-1)
# kdDCP = digestion rate Digestible Crude Protein (% hr-1)
# kdPass = standard passage rate in the rumen (% hr-1)
# UNDF = Undegradable Neutral Detergent Fibre (g kg-1 DM)
# pef = physical effectiveness factor for Neutral Detergent Fibre (-)
# CP = crude protein (g kg DM-1)
# GE = gross energy (MJ kg DM-1)

# R vectors translated directly to Python/NumPy arrays.
# R NA is represented as np.nan.

BARLEY = np.array([
    0.245, 0.573, 389, 214, 156.00, 34.50, 82.80, np.nan, 0.242, 0.145, 0.125,
    0.040, 21.00, np.nan, 0.34, 138, 18.4
], dtype=float)

CONCENTRATE = np.array([
    0.249, 0.619, 262, 175, 243.10, 72.80, 87.36, np.nan, 0.150, 0.060, 0.100,
    0.040, 42.90, np.nan, 0.69, 182, 18.5
], dtype=float)

HAY = np.array([
    0.318, 1.120, 100, 150, 345.80, 48.16, 74.30, np.nan, 0.300, 0.040, 0.085,
    0.035, 148.20, np.nan, 1.00, 172, 18.5
], dtype=float)

HAYPOOR = np.array([
    0.420, 1.370, 73, 73, 462.00, 20.30, 149.10, np.nan, 0.300, 0.040, 0.085,
    0.035, 198.00, np.nan, 1.00, 70, 18.2
], dtype=float)

GRASSSPRING = np.array([
    0.304, 0.960, 130, 30, 360.00, 66.25, 97.40, np.nan, 0.300, 0.040, 0.085,
    0.035, 120.00, np.nan, 0.40, 265, 18.6
], dtype=float)

GRASSSUMMER = np.array([
    0.356, 1.120, 100, 60, 376.00, 49.50, 76.50, np.nan, 0.300, 0.040, 0.085,
    0.035, 141.00, np.nan, 0.50, 180, 18.4
], dtype=float)

GRASSSUMMERDRY = np.array([
    0.447, 1.280, 50, 60, 409.50, 23.00, 69.00, np.nan, 0.300, 0.040, 0.085,
    0.035, 175.50, np.nan, 1.00, 115, 18.1
], dtype=float)

MAIZE = np.array([
    0.237, 0.438, 202, 532, 101.70, 20.10, 86.56, np.nan, 0.040, 0.051, 0.035,
    0.050, 11.30, np.nan, 0.40, 134, 17.0
], dtype=float)

MOLASSE = np.array([
    0.050, 0.200, 828, 0, 0, 3.8, 0.2, np.nan, 0, 0, 0.125,
    0.040, 0, np.nan, 0, 4, 17.0
], dtype=float)

SBM = np.array([
    0.242, 0.526, 107, 0, 138.60, 202.80, 243.40, np.nan, 0.242, 0.145, 0.125,
    0.040, 0.00, np.nan, 0.34, 507, 19.7
], dtype=float)

STRAWCER = np.array([
    0.557, 1.800, 14, 78, 401.00, 10.00, 5.00, np.nan, 0.300, 0.040, 0.085,
    0.035, 0.00, np.nan, 0.34, 40, 18.3
], dtype=float)

WHEAT = np.array([
    0.239, 0.475, 475, 212, 80.00, 39.90, 69.80, np.nan, 0.182, 0.150, 0.080,
    0.040, 34.20, np.nan, 0.00, 133, 18.2
], dtype=float)

MAIZESILAGE = np.array([
    0.290, 1.000, 100, 351, 239.00, 54.94, 23.00, np.nan, 0.250, 0.040, 0.040,
    0.030, 239.00, np.nan, 0.93, 82, 18.5
], dtype=float)

SUNFLOWERHULLS = np.array([0.000, 1.000], dtype=float)

# FEEDS <- rbind(BARLEY,CONCENTRATE, HAY, HAYPOOR, GRASSSPRING, GRASSSUMMER,
#                GRASSSUMMERDRY, MAIZE, MOLASSE, SBM, STRAWCER, WHEAT, MAIZESILAGE)
FEEDS = np.vstack([
    BARLEY,
    CONCENTRATE,
    HAY,
    HAYPOOR,
    GRASSSPRING,
    GRASSSUMMER,
    GRASSSUMMERDRY,
    MAIZE,
    MOLASSE,
    SBM,
    STRAWCER,
    WHEAT,
    MAIZESILAGE
])

# FEEDS <- rbind(FEEDS,FEEDS)
FEEDS = np.vstack([FEEDS, FEEDS])

# PASSREDFRAC <- c(rep(0.55,13), rep(1.00,13))
PASSREDFRAC = np.array([0.55] * 13 + [1.00] * 13, dtype=float)

# MECONTENTS <- c(NA)
MECONTENTS = [np.nan]

# ------------------------------------------------------------------------------
# Main simulation loop
# ------------------------------------------------------------------------------
for s in range(26):
    FEED1 = FEEDS[s, :]
    FEED2 = HAY
    FEED3 = GRASSSPRING
    FEED4 = GRASSSUMMERDRY

    # minimum feed quantities based on rumen digestive capacity,
    # feed quantity offered and feed fractions
    FEED1QNTY = 1.0
    FEED2QNTY = 0.0
    FEED3QNTY = 0.0
    FEED4QNTY = 0.0

    FEEDQNTY = FEED1QNTY + FEED2QNTY + FEED3QNTY + FEED4QNTY

    # Preserve original R logic exactly, including FEED4QNTY * FEED1[16]
    # R index [16] -> Python index [15]
    if FEEDQNTY == 0:
        CPAVG = 0.0
    else:
        CPAVG = (
            FEED1QNTY * FEED1[15]
            + FEED2QNTY * FEED2[15]
            + FEED3QNTY * FEED3[15]
            + FEED4QNTY * FEED1[15]
        ) / FEEDQNTY

    PASSRED = PASSREDFRAC[s]

    # CH digestion (INSC = insoluble, non-structural carbohydrates)
    # R index [4] -> Python [3]
    # R index [9] -> Python [8]
    # R index [12] -> Python [11]
    INSC = (
        FEED1QNTY * FEED1[3] * FEED1[8] / (FEED1[8] + FEED1[11] * PASSRED)
        + FEED2QNTY * FEED2[3] * FEED2[8] / (FEED2[8] + FEED2[11] * PASSRED)
        + FEED3QNTY * FEED3[3] * FEED3[8] / (FEED3[8] + FEED3[11] * PASSRED)
        + FEED4QNTY * FEED4[3] * FEED4[8] / (FEED4[8] + FEED4[11] * PASSRED)
    )

    INSCTOTAL = (
        FEED1QNTY * FEED1[3]
        + FEED2QNTY * FEED2[3]
        + FEED3QNTY * FEED3[3]
        + FEED4QNTY * FEED4[3]
    )

    INSCDIG = INSC / INSCTOTAL
    INSCINT = max(0.0, INSCTOTAL * 0.97 - INSC)
    INSCINTDIG = INSCINT / INSCTOTAL

    # NDF digestion
    # R index [5] -> Python [4]
    # R index [10] -> Python [9]
    NDF = (
        FEED1QNTY * FEED1[4] * FEED1[9] / (FEED1[9] + FEED1[11] * PASSRED)
        + FEED2QNTY * FEED2[4] * FEED2[9] / (FEED2[9] + FEED2[11] * PASSRED)
        + FEED3QNTY * FEED3[4] * FEED3[9] / (FEED3[9] + FEED3[11] * PASSRED)
        + FEED4QNTY * FEED4[4] * FEED4[9] / (FEED4[9] + FEED4[11] * PASSRED)
    )

    NDFTOTAL = (
        FEED1QNTY * FEED1[4]
        + FEED2QNTY * FEED2[4]
        + FEED3QNTY * FEED3[4]
        + FEED4QNTY * FEED4[4]
    )

    NDFDIG = NDF / NDFTOTAL

    NDFINT = (
        FEED1QNTY
        * FEED1[4]
        * (1 - FEED1[9] / (FEED1[9] + FEED1[11] * PASSRED))
        * (FEED1[9] * 0.9)
        / (FEED1[9] * 0.9 + 0.125)
        + FEED2QNTY
        * FEED2[4]
        * (1 - FEED2[9] / (FEED2[9] + FEED2[11] * PASSRED))
        * (FEED2[9] * 0.9)
        / (FEED2[9] * 0.9 + 0.125)
        + FEED3QNTY
        * FEED3[4]
        * (1 - FEED3[9] / (FEED3[9] + FEED3[11] * PASSRED))
        * (FEED3[9] * 0.9)
        / (FEED3[9] * 0.9 + 0.125)
        + FEED4QNTY
        * FEED4[4]
        * (1 - FEED4[9] / (FEED4[9] + FEED4[11] * PASSRED))
        * (FEED4[9] * 0.9)
        / (FEED4[9] * 0.9 + 0.125)
    )

    NDFINTDIG = NDFINT / NDFTOTAL
    NDFINTDIGTOT = NDFINT / (FEEDQNTY * 1000)

    # Protein digestion
    # R index [7] -> Python [6]
    # R index [11] -> Python [10]
    PICP = (
        FEED1QNTY * FEED1[6] * FEED1[10] / (FEED1[10] + FEED1[11] * PASSRED)
        + FEED2QNTY * FEED2[6] * FEED2[10] / (FEED2[10] + FEED2[11] * PASSRED)
        + FEED3QNTY * FEED3[6] * FEED3[10] / (FEED3[10] + FEED3[11] * PASSRED)
        + FEED4QNTY * FEED4[6] * FEED4[10] / (FEED4[10] + FEED4[11] * PASSRED)
    )

    PROTTOTAL = (
        FEED1QNTY * FEED1[15]
        + FEED2QNTY * FEED2[15]
        + FEED3QNTY * FEED3[15]
        + FEED4QNTY * FEED4[15]
    )

    PROTINT = (
        PROTTOTAL
        - (
            FEED1QNTY * FEED1[5]
            + FEED2QNTY * FEED2[5]
            + FEED3QNTY * FEED3[5]
            + FEED4QNTY * FEED4[5]
        )
        - PICP
    )

    PROTUPT = 0.9 * PROTTOTAL - 32 * FEEDQNTY
    PROTEXCR = PROTTOTAL - PROTUPT

    PROTDIGRU = (PROTTOTAL - PROTINT) / PROTTOTAL
    PROTDIGWT = PROTUPT / PROTTOTAL

    # Digestion and excretion
    # Preserve original order and logic
    if FEED1[15] > 500:
        PROTTOTAL = 0.0

    DIGFRAC = (
        FEED1QNTY * FEED1[2]
        + FEED2QNTY * (FEED2[2] + FEED2[5])
        + FEED3QNTY * (FEED3[2] + FEED3[5])
        + FEED4QNTY * (FEED4[2] + FEED4[5])
        + INSC
        + INSCINT
        + NDF
        + NDFINT
        + PROTUPT
        + PROTTOTAL
        * (121.7 - 12.01 * (FEED1[15] / 10) + 0.3235 * (FEED1[15] / 10) ** 2)
        / 100.0
    )

    CHEXCR = FEEDQNTY * 1000 - DIGFRAC - PROTEXCR
    EXCRFRAC = FEEDQNTY * 1000 - DIGFRAC

    GEEXCR = (
        (PROTEXCR * 23.8 + CHEXCR * 17.4) / (PROTEXCR + CHEXCR)
    )

    GEUPTAKE = (
        (PROTUPT * 23.8 + (DIGFRAC - PROTUPT) * 17.4) / DIGFRAC
    )

    if EXCRFRAC == 0:
        MEUPTAKE = 0.0
    else:
        MEUPTAKE = DIGFRAC / 1000.0 * GEUPTAKE * 0.82

    if EXCRFRAC == 0:
        Q = 0.0
    else:
        Q = DIGFRAC / (FEEDQNTY * 1000.0)

    MECONTENTS.append(MEUPTAKE)

# Print result similar to the final accumulated R vector
print("MECONTENTS:")
print(MECONTENTS)


########################## line:168 ##############################
import numpy as np
import pandas as pd
from scipy.stats import t
import statsmodels.api as sm

# ------------------------------------------------------------------
# Direct conversion of the R code block
# Assumes MECONTENTS already exists from the previous simulation code
# ------------------------------------------------------------------

# R:
# MEpred1 <- matrix(nrow=13, ncol=2, MECONTENTS[2:length(MECONTENTS)])
#
# In Python, MECONTENTS[1:] skips the initial NA/np.nan.
# R fills matrices column-wise by default, so order='F' is used.
MEpred1 = np.array(MECONTENTS[1:], dtype=float).reshape((13, 2), order="F")

MEpred1_df = pd.DataFrame(
    MEpred1,
    index=[
        "BARLEY", "CONCENTRATE", "HAY", "HAYPOOR", "GRASSSPRING",
        "GRASSSUMMER", "GRASSSUMMERDRY", "MAIZE", "MOLASSE",
        "SBM", "STRAWCER", "WHEAT", "MAIZESILAGE"
    ],
    columns=["low rumen fill", "high rumen fill"]
)

MEpred = MEpred1.mean(axis=1)

BARS = MEpred1[:, 0] - MEpred

# Maize silage is excluded from the analysis for the dataset of Kolver (2000),
# as this feed type is not part of the measure data.

MEMAFF = np.array([12.8, 13.8, 9.6, 8.1, 13.1, 10.7, 8.4, 13.8, 12.6, 13.4, 6.4, 13.6, 11.2], dtype=float)
MEKolver = np.array([13, 13.6, 9.7, 7.3, 11.75, 10, 8.75, 13.6, 12.0, np.nan, 6.4, 12.6, 10.3], dtype=float)

MINMAFF = np.array([12.1, np.nan, 8.7, 7.5, 12.9, 9.7, 7.0, 12.2, np.nan, 12.6, 3.4, 12.3, 10.2], dtype=float)
MAXMAFF = np.array([13.7, np.nan, 10.3, 8.7, 13.5, 12.3, 9.3, 16.4, np.nan, 14.3, 10.4, 14.7, 11.7], dtype=float)

DIG = pd.DataFrame({
    "MEpred": MEpred,
    "MEMAFF": MEMAFF,
    "MEKolver": MEKolver
})

DIG1 = pd.DataFrame({
    "rep_MEpred_2": np.tile(MEpred, 2),
    "combined_measured": np.concatenate([MEMAFF, MEKolver])
})

# ------------------------------------------------------------------
# Statistics for MAFF, 1986
# R: summary(lm(MEpred ~ MEMAFF, data = DIG))
# ------------------------------------------------------------------
X_maff = sm.add_constant(DIG["MEMAFF"])
model_maff = sm.OLS(DIG["MEpred"], X_maff, missing="drop").fit()

stats = model_maff.summary()

slope = model_maff.params["MEMAFF"]
se_slope = model_maff.bse["MEMAFF"]
t_value = (slope - 1) / se_slope
p_one_sided_maff = 1 - t.cdf(t_value, df=13 - 2)

RMSE = np.sqrt(np.sum((MEMAFF - MEpred) ** 2) / 13)
RMSEmean = np.mean(MEMAFF)
RMSErel = RMSE / RMSEmean

meanerror = np.sum(np.abs(MEMAFF - MEpred)) / 13
meanrel = meanerror / RMSEmean

# ------------------------------------------------------------------
# Statistics for Kolver, 2000
# R: summary(lm(MEKolver ~ MEpred, data = DIG))
# ------------------------------------------------------------------
X_kolver = sm.add_constant(DIG["MEpred"])
model_kolver = sm.OLS(DIG["MEKolver"], X_kolver, missing="drop").fit()

sm_kolver = model_kolver.summary()

slope = model_kolver.params["MEpred"]
se_slope = model_kolver.bse["MEpred"]
t_value = (slope - 1) / se_slope
p_one_sided_kolver = 1 - t.cdf(t_value, df=12 - 2)

# R:
# c(MEKolver[1:9],MEKolver[11:13])  -> Python [0:9] and [10:13]
MEKolver_sub = np.concatenate([MEKolver[0:9], MEKolver[10:13]])
MEpred_sub = np.concatenate([MEpred[0:9], MEpred[10:13]])

RMSE_kolver = np.sqrt(np.sum((MEKolver_sub - MEpred_sub) ** 2) / 12)
RMSEmean_kolver = np.mean(MEKolver_sub)
RMSErel_kolver = RMSE_kolver / RMSEmean_kolver

meanerror_kolver = np.sum(np.abs(MEKolver_sub - MEpred_sub)) / 12
meanrel_kolver = meanerror_kolver / RMSEmean_kolver

# ------------------------------------------------------------------
# Again statistics for both MAFF (1986) and Kolver (2000)
# ------------------------------------------------------------------
model_summaff = sm.OLS(DIG["MEMAFF"], sm.add_constant(DIG["MEpred"]), missing="drop").fit()

SUMMAFF = model_summaff.summary()
slope = model_summaff.params["MEpred"]
se_slope = model_summaff.bse["MEpred"]
t_value = (slope - 1) / se_slope
p_two_sided_maff = (1 - t.cdf(t_value, df=13 - 2)) * 2

model_sumkol = sm.OLS(DIG["MEKolver"], sm.add_constant(DIG["MEpred"]), missing="drop").fit()

SUMKOL = model_sumkol.summary()
slope = model_sumkol.params["MEpred"]
se_slope = model_sumkol.bse["MEpred"]
t_value = (slope - 1) / se_slope
p_two_sided_kolver = (1 - t.cdf(t_value, df=12 - 2)) * 2

# ------------------------------------------------------------------
# Root mean square error (RMSE) and mean absolute error (MAE)
# ------------------------------------------------------------------
PRED = np.concatenate([MEpred, MEpred])
PRED = np.concatenate([PRED[0:22], PRED[23:26]])

MEAS = np.concatenate([MEMAFF, MEKolver])
MEAS = np.concatenate([MEAS[0:22], MEAS[23:26]])

RMSE_all = np.sqrt(np.sum((PRED - MEAS) ** 2) / 25)
RMSEmean_all = np.mean(MEAS)
RMSErel_all = RMSE_all / RMSEmean_all

MAE = np.sum(np.abs(PRED - MEAS)) / 25
MAErel = meanerror / RMSEmean_all  # preserved exactly from the R code

# ------------------------------------------------------------------
# Optional prints to mirror the R workflow
# ------------------------------------------------------------------
print("MEpred1:")
print(MEpred1_df)
print()

print("MEpred:")
print(MEpred)
print()

print("BARS:")
print(BARS)
print()

print("DIG:")
print(DIG)
print()

print("DIG1:")
print(DIG1)
print()

print("Statistics for MAFF (MEpred ~ MEMAFF):")
print(stats)
print(f"slope = {slope}")
print(f"se.slope = {se_slope}")
print(f"t.value = {t_value}")
print(f"one-sided p-value = {p_one_sided_maff}")
print()

print("RMSE (MAFF) =", RMSE)
print("RMSEmean (MAFF) =", RMSEmean)
print("RMSErel (MAFF) =", RMSErel)
print("meanerror (MAFF) =", meanerror)
print("meanrel (MAFF) =", meanrel)
print()

print("Statistics for Kolver (MEKolver ~ MEpred):")
print(sm_kolver)
print(f"slope = {model_kolver.params['MEpred']}")
print(f"se.slope = {model_kolver.bse['MEpred']}")
print(f"one-sided p-value = {p_one_sided_kolver}")
print()

print("RMSE (Kolver) =", RMSE_kolver)
print("RMSEmean (Kolver) =", RMSEmean_kolver)
print("RMSErel (Kolver) =", RMSErel_kolver)
print("meanerror (Kolver) =", meanerror_kolver)
print("meanrel (Kolver) =", meanrel_kolver)
print()

print("Again statistics for both MAFF and Kolver:")
print(SUMMAFF)
print(f"two-sided p-value (MAFF) = {p_two_sided_maff}")
print()

print(SUMKOL)
print(f"two-sided p-value (Kolver) = {p_two_sided_kolver}")
print()

print("Combined errors:")
print("RMSE_all =", RMSE_all)
print("RMSEmean_all =", RMSEmean_all)
print("RMSErel_all =", RMSErel_all)
print("MAE =", MAE)
print("MAErel =", MAErel)
####################### line:250 #################################
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Create a plot with simulated versus measured ME contents
# Figure 4 in the main paper of Van der Linden et al.
# Direct Python conversion of the R plotting block
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=200)

one = np.array([0, 16], dtype=float)

# Equivalent to:
# plot(one~c(0,16), type = "l", lty="dashed", xlim = c(0,16), ylim = c(0,16), ...)
ax.plot([0, 16], one, linestyle="--")

ax.set_xlim(0, 16)
ax.set_ylim(0, 16)

# Labels approximating the R mathematical expressions
ax.set_xlabel(r"Simulated ME content (MJ kg$^{-1}$)")
ax.set_ylabel(r"Measured ME content (MJ kg$^{-1}$)")

# Similar to las=1 in R (horizontal tick labels)
ax.tick_params(axis='both', labelrotation=0)

# Match xaxs="i", yaxs="i" behavior closely
ax.margins(x=0, y=0)

# ------------------------------------------------------------------
# Regression lines
# R:
# lines(c(SUMMAFF$coefficients[1,1], SUMMAFF$coefficients[1,1]+ 16*SUMMAFF$coefficients[2,1])~c(0, 16), lty="solid")
# lines(c(SUMKOL$coefficients[1,1], SUMKOL$coefficients[1,1]+ 16*SUMKOL$coefficients[2,1])~c(0, 16), lty="dotted")
# ------------------------------------------------------------------

intercept_maff = model_summaff.params["const"]
slope_maff = model_summaff.params["MEpred"]

intercept_kol = model_sumkol.params["const"]
slope_kol = model_sumkol.params["MEpred"]

x_line = np.array([0, 16], dtype=float)

y_line_maff = np.array([
    intercept_maff,
    intercept_maff + 16 * slope_maff
], dtype=float)

y_line_kol = np.array([
    intercept_kol,
    intercept_kol + 16 * slope_kol
], dtype=float)

ax.plot(x_line, y_line_maff, linestyle="-")
ax.plot(x_line, y_line_kol, linestyle=":")

# ------------------------------------------------------------------
# Horizontal arrows:
# arrows(DIG1[,1]-BARS, DIG1[,2], DIG1[,1]+BARS, DIG1[,2], ...)
#
# DIG1 first column = repeated MEpred values
# DIG1 second column = combined measured values
# ------------------------------------------------------------------
for x_center, y_val, half_width in zip(DIG1.iloc[:, 0], DIG1.iloc[:, 1], np.tile(BARS, 2)):
    ax.annotate(
        "",
        xy=(x_center + half_width, y_val),
        xytext=(x_center - half_width, y_val),
        arrowprops=dict(arrowstyle="|-|", lw=1, shrinkA=0, shrinkB=0)
    )

# ------------------------------------------------------------------
# Vertical arrows:
# arrows(DIG$MEpred, MINMAFF, DIG$MEpred, MAXMAFF, ...)
# ------------------------------------------------------------------
for x_val, y_min, y_max in zip(DIG["MEpred"], MINMAFF, MAXMAFF):
    if not (np.isnan(y_min) or np.isnan(y_max)):
        ax.annotate(
            "",
            xy=(x_val, y_max),
            xytext=(x_val, y_min),
            arrowprops=dict(arrowstyle="|-|", lw=1, shrinkA=0, shrinkB=0)
        )

# ------------------------------------------------------------------
# Points
# R:
# points(MEMAFF~MEpred, pch = 19)
# points(MEKolver~MEpred, pch = 24, bg="white")
# ------------------------------------------------------------------

# Filled circles for MEMAFF
ax.scatter(MEpred, MEMAFF, marker="o", s=35)

# Open triangles for MEKolver
mask_kolver = ~np.isnan(MEKolver)
ax.scatter(
    MEpred[mask_kolver],
    MEKolver[mask_kolver],
    marker="^",
    s=55,
    facecolors="white",
    edgecolors="black"
)

# ------------------------------------------------------------------
# Save figure
# R:
# tiff("D:/Plot4.tiff", width = 5.0, height = 5.0, units = 'in', res = 200)
# ------------------------------------------------------------------
#plt.tight_layout()
#plt.savefig("Plot4.png", format="png", dpi=200)
#plt.show()
######################### line: 271 #######################################
# ------------------------------------------------------------------
# Legend
# R:
# legend("topleft",
#        legend=c("MAFF (1986)","Kolver (2000)", "x = y"),
#        col = c("black", "black", "black"),
#        pch=c(19, 24, NA),
#        pt.bg = c(NA,"white",NA),
#        lty= c("solid","dotted","dashed"),
#        cex=0.9,
#        bty = "n")
# ------------------------------------------------------------------
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0], [0],
        color="black",
        linestyle="-",
        marker="o",
        markersize=6,
        linewidth=1,
        label="MAFF (1986)"
    ),
    Line2D(
        [0], [0],
        color="black",
        linestyle=":",
        marker="^",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=7,
        linewidth=1,
        label="Kolver (2000)"
    ),
    Line2D(
        [0], [0],
        color="black",
        linestyle="--",
        linewidth=1,
        label="x = y"
    )
]

ax.legend(
    handles=legend_elements,
    loc="upper left",
    frameon=False,
    fontsize=9
)

# ------------------------------------------------------------------
# Text annotations
# R:
# text(9.5,2.0, bquote(MAFF~(1986)~":"~y~"="~.(round(SUMMAFF$coefficients[1,1], digits = 2))~+~.(round(SUMMAFF$coefficients[2,1], digits = 2))~"x;"~R^{2}~adj.~"= 0.90"), cex = 0.8)
#
# text(9.5,0.7, bquote(Kolver~(2000)~":"~y~"="~.(round(SUMKOL$coefficients[1,1], digits = 2))~+~.(round(SUMKOL$coefficients[2,1], digits = 2))~"x;"~R^{2}~adj.~"="~.(round(SUMKOL$adj.r.squared, digits = 2))), cex = 0.8)
# ------------------------------------------------------------------

intercept_maff_rounded = round(model_summaff.params["const"], 2)
slope_maff_rounded = round(model_summaff.params["MEpred"], 2)

intercept_kol_rounded = round(model_sumkol.params["const"], 2)
slope_kol_rounded = round(model_sumkol.params["MEpred"], 2)
adj_r2_kol_rounded = round(model_sumkol.rsquared_adj, 2)

# The R code hard-codes 0.90 for MAFF adjusted R², so we preserve that exactly.
text_maff = (
    f"MAFF (1986): y = {intercept_maff_rounded} + {slope_maff_rounded}x; "
    f"R\u00b2 adj. = 0.90"
)

text_kolver = (
    f"Kolver (2000): y = {intercept_kol_rounded} + {slope_kol_rounded}x; "
    f"R\u00b2 adj. = {adj_r2_kol_rounded}"
)

ax.text(9.5, 2.0, text_maff, fontsize=8)
ax.text(9.5, 0.7, text_kolver, fontsize=8)

# ------------------------------------------------------------------
# Equivalent of dev.off()
# In Python, saving and closing the figure is the equivalent.
# ------------------------------------------------------------------
plt.tight_layout()
plt.savefig("Plot4.png", format="png", dpi=200)
plt.close(fig)
########################### line:286 ######################################