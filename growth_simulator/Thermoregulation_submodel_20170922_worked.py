import numpy as np

#######################################################################################
#                                                                                     #
# LiGAPS-Beef (Livestock simulator for Generic analysis of Animal Production Systems) #
#                                                                                     #
# Sub-model thermoregulation 2017-09-22                                               #
#                                                                                     #
# Direct Python conversion of the provided R block                                    #
# Logic preserved as closely as possible                                              #
#                                                                                     #
#######################################################################################

######################################################
# Part I: sensitivity analysis on weather conditions #
######################################################

# Weather parameters included in the sensitivity analysis:
# 1. Wind speed (m s-1)
# 2. Relative humidity (%)
# 3. Solar radiation (kJ m-2 day-1)
# 4. Cloud cover (Octa)
# 5. Rainfall (mm)
# 6. Total body weight (kg)
# 7. Heat production (x maintenance heat production)
def r_seq(from_, to_, by_):
    """
    R-like seq(from=..., to=..., by=...) for this script.
    If by == 0 and from_ == to_, return a single-value vector.
    """
    if by_ == 0:
        return np.array([from_], dtype=float)
    return np.arange(from_, to_ + by_ / 2.0, by_, dtype=float)

def r_matrix_byrow(seq_vals, nrow, ncol):
    """
    Mimic R: matrix(nrow=..., ncol=..., data=rep(seq_vals, nrow), byrow=TRUE)
    """
    seq_vals = np.asarray(seq_vals, dtype=float)
    data = np.tile(seq_vals, nrow)
    total = nrow * ncol
    if data.size < total:
        reps = int(np.ceil(total / data.size))
        data = np.tile(data, reps)
    data = data[:total]
    return data.reshape((nrow, ncol), order="C")
# R:
# GRAPHNR <- NULL
# GRAPHNR[1] <- 1
#
# In Python, use a list and place the first value directly.
GRAPHNR = [1]

# Default values for each of the seven weather inputs (4th value is jmax)
DWIND_default = np.array([4, 4, 0, 2], dtype=float)         # default value wind speed: 4 ms-1
DRH_default   = np.array([50, 50, 0, 2], dtype=float)       # default value relative humidity: 50%
DSWR_default  = np.array([20000, 20000, 0, 2], dtype=float) # default value solar radiation: 20000 kJ m-2 day-1
DCC_default   = np.array([4, 4, 0, 2], dtype=float)         # default value cloud cover: 4 okta
DRAIN_default = np.array([0, 0, 0, 2], dtype=float)         # default value precipitation: 0 mm day-1
DTBW_default  = np.array([450, 450, 0, 2], dtype=float)     # default value total body weight (TBW): 450 kg
DHP_default   = np.array([0.36, 0.36, 0, 2], dtype=float)   # default value heat production

# R:
# DEFAULT <- rbind(DWIND, DRH, DSWR, DCC, DRAIN, DTBW, DHP)
DEFAULT = np.vstack([
    DWIND_default,
    DRH_default,
    DSWR_default,
    DCC_default,
    DRAIN_default,
    DTBW_default,
    DHP_default
])

# Values for sensitivity analysis (4th value is jmax)
DWIND = np.array([0.1, 8, 0.1, 80], dtype=float)       # range wind speed: 0.1-8 ms-1
DRH   = np.array([10, 100, 1, 91], dtype=float)        # range relative humidity: 10-100%
DSWR  = np.array([0, 30000, 300, 101], dtype=float)    # range solar radiation: 0-30000 kJ m-2 day-1
DCC   = np.array([0, 8, 0.1, 81], dtype=float)         # range cloud cover: 0-8 okta
DRAIN = np.array([0, 30, 0.3, 101], dtype=float)       # range precipitation: 0-30 mm day-1
DTBW  = np.array([50, 1300, 12.5, 101], dtype=float)   # range total body weight (TBW): 50-1300 kg
DHP   = np.array([0, 1, 0.01, 101], dtype=float)       # range heat production

# R:
# library(plot3D)
#
# In Python, no equivalent import is required yet for this exact block.
# If later plotting is needed, that can be translated separately.

BREED = 1     # Choose breed (1= Charolais; 2 = Boran; 3 = Brahman (3/4) x Shorthorn(1/4))
HOUSING = 1   # Stable == 0; outdoor conditions == 1, open feedlot = 2
imax = 81     # Number of steps (days) for model simulations

smax = 7      # Five weather variables, and TBW and heat production are included.

################ line:83  ####################################
######################################################
# Part II: sensitivity analysis on parameters        #
######################################################
#**************************************************
for s in range(1, smax + 1):  ### line 89-1082
   
#################### line:90  ##################################
    GRAPHNR.append(GRAPHNR[s - 1] + 1)
    print(GRAPHNR[s - 1])

    DEFAULT1 = DEFAULT.copy()

    if GRAPHNR[s - 1] == 1:
        DEFAULT1[0, :] = DWIND
    elif GRAPHNR[s - 1] == 2:
        DEFAULT1[1, :] = DRH
    elif GRAPHNR[s - 1] == 3:
        DEFAULT1[2, :] = DSWR
    elif GRAPHNR[s - 1] == 4:
        DEFAULT1[3, :] = DCC
    elif GRAPHNR[s - 1] == 5:
        DEFAULT1[4, :] = DRAIN
    elif GRAPHNR[s - 1] == 6:
        DEFAULT1[5, :] = DTBW
    elif GRAPHNR[s - 1] == 7:
        DEFAULT1[6, :] = DHP

    jmax = int(DEFAULT1[s - 1, 3])

    if GRAPHNR[s - 1] == 6:
        #TBWACT = np.arange(
        #    DEFAULT1[5, 0],
        #    DEFAULT1[5, 1] + DEFAULT1[5, 2] / 2.0,
        #    DEFAULT1[5, 2],
        #    dtype=float
        #)
        TBWACT = r_seq(DEFAULT1[5, 0], DEFAULT1[5, 1], DEFAULT1[5, 2])
    else:
        TBWACT = np.repeat(DEFAULT1[5, 0], jmax).astype(float)

    # TBWACT = np.repeat(450, jmax)

    if GRAPHNR[s - 1] == 7:
        #MAINTT = np.arange(
        #    DEFAULT1[6, 0],
        #    DEFAULT1[6, 1] + DEFAULT1[6, 2] / 2.0,
        #    DEFAULT1[6, 2],
        #    dtype=float
        #)
        MAINTT = r_seq(DEFAULT1[6, 0], DEFAULT1[6, 1], DEFAULT1[6, 2])
    else:
        MAINTT = np.repeat(DEFAULT1[6, 0], jmax).astype(float)

    # MAINTT = np.repeat(0.36, jmax)

    TIME = np.arange(1, imax + 1, dtype=int)

    ###########################################################################################
    #                                       Weather data                                      #
    ###########################################################################################

    # Wind speed
    ws_seq = r_seq(DEFAULT1[0, 0], DEFAULT1[0, 1], DEFAULT1[0, 2])
    WSSTABLE = r_matrix_byrow(ws_seq, imax, jmax)

    # Average ambient temperature
    tstable_data = np.tile(np.arange(-40, 41, 1, dtype=float), jmax)
    TSTABLE = np.array(tstable_data, dtype=float).reshape((imax, jmax), order="F")

    # Relative humidity
    rh_seq = r_seq(DEFAULT1[1, 0], DEFAULT1[1, 1], DEFAULT1[1, 2])
    RHSTABLE = r_matrix_byrow(rh_seq, imax, jmax)

    # Solar radiation
    swr_seq = r_seq(DEFAULT1[2, 0], DEFAULT1[2, 1], DEFAULT1[2, 2])
    SWRAD = r_matrix_byrow(swr_seq, imax, jmax)

    # Conversion factor solar radiation on soil to solar radiation on coat
    AHAdata = np.full((imax, jmax), 0.5, dtype=float)

    # Cloud cover
    octa_seq = r_seq(DEFAULT1[3, 0], DEFAULT1[3, 1], DEFAULT1[3, 2])
    OCTA = r_matrix_byrow(octa_seq, imax, jmax)

    # Rain
    rain_seq = r_seq(DEFAULT1[4, 0], DEFAULT1[4, 1], DEFAULT1[4, 2])
    RAIN = r_matrix_byrow(rain_seq, imax, jmax)

    MATRIX = np.repeat(0.0, imax)
    MATRIX1 = np.repeat(0.0, imax)
    MATRIXSK = np.repeat(0.0, imax)
    #################### line:210 #################################
    ###########################################################################################
    #                     Genetic parameters (related to BREED and GENDER)                    #
    ###########################################################################################

    # This genetic parameter section contains a list with parameters which are specific for breed and gender
    # No significant difference between males and females in the thermoregulation model (except for TBWACT)

    # Parameters for Charolais
    LIBRARY10 = np.array([
        0.6,
        0.012,
        1.00,
        64.1,
        3.08,
        1.73,
        1.000,
        35.3,
        1.00
    ], dtype=float)

    # Parameters for Borans
    LIBRARY20 = np.array([
        0.6,
        0.012,
        1.12,
        64.1,
        4.89,
        0.80,
        1.30,
        34.5,
        0.91
    ], dtype=float)

    # Parameters for 3/4 Brahman x 1/4 Shorthorns
    LIBRARY30 = np.array([
        0.56,
        0.012,
        1.09,
        64.1,
        4.44,
        1.03,
        1.000,
        34.7,
        0.93
    ], dtype=float)

    if BREED == 1:
        LIBRARY = LIBRARY10
    elif BREED == 2:
        LIBRARY = LIBRARY20
    elif BREED == 3:
        LIBRARY = LIBRARY30
    #################### line:275 ############################
    ###########################################################################################
    #                         Initial values thermoregulation sub-model                       #
    ###########################################################################################

    # Constants from 'LIBRARY'
    REFLC = LIBRARY[0]        # fraction light reflected from coat (-)
    LC = LIBRARY[1]           # coat length (m)
    AREAFACTOR = LIBRARY[2]   # surface area temperate Bos taurus (1.00) or Bos indicus/tropical Bos taurus (1.12)
    CBSMAX = LIBRARY[3]       # maximum body-skin conductivity (W m-2 K-1)
    RBCSf = LIBRARY[6]

    # General constants used in physics
    pi = 3.14159265
    P = 101325
    Rdair = 287.058
    Rwater = 461.495
    CtoK = 273.15
    Cp = 1.005
    L = 2260
    REFLEgrass = 0.10
    REFLEconcr = 0.50
    GRAV = 9.81
    SIGMA = 5.67037 * 10**(-8)
    GAMMA = 66
    EMISS = 0.98
    MuSt = 1.827 * 10**(-5)
    TR0 = 527
    CCONV1 = 120
    CCONV2 = 0.61
    kJdaytoW = 1000 / (3600 * 24)
    KtoR = 9 / 5
    Schmidt = 0.61

    # Cattle specific constants
    CoatConst = 1.90 * 10**(-5)
    ZC = 11000
    TbodyC = 39
    LASMIN = 10
    RESPINCR = 7.64
    RAINFRAC = 0.3

    # Variables
    # 1. Respiration
    #import numpy as np

    # 1. Respiration
    TBW = np.full(imax, np.nan)
    AREA = np.full(imax, np.nan)
    DIAMETER = np.full(imax, np.nan)
    LENGTH = np.full(imax, np.nan)
    brr = np.full(imax, np.nan)
    btv = np.full(imax, np.nan)
    brv = np.full(imax, np.nan)
    irv = np.full(imax, np.nan)
    TAVGC = np.full(imax, np.nan)
    TAVGK = np.full(imax, np.nan)
    VPSATAIR = np.full(imax, np.nan)
    VPAIRTOT = np.full(imax, np.nan)
    RHAIR = np.full(imax, np.nan)
    RHOVP = np.full(imax, np.nan)
    RHODAIR = np.full(imax, np.nan)
    RHOAIR = np.full(imax, np.nan)
    CHIAIR = np.full(imax, np.nan)
    VISCAIR = np.full(imax, np.nan)
    Texh = np.full(imax, np.nan)
    VPSATAIROUT = np.full(imax, np.nan)
    RHOVPOUT = np.full(imax, np.nan)
    RHODAIROUT = np.full(imax, np.nan)
    RHOAIROUT = np.full(imax, np.nan)
    CHIAIROUT = np.full(imax, np.nan)
    AIREXCH = np.full(imax, np.nan)
    LHEATRESP = np.full(imax, np.nan)
    CHEATRESP = np.full(imax, np.nan)
    TGRESP = np.full(imax, np.nan)
    TNRESP = np.full(imax, np.nan)
    TNRESPH = np.full(imax, np.nan)
    NERESP = np.full(imax, np.nan)
    NERESPWM = np.full(imax, np.nan)
    ENRESPC = np.full(imax, np.nan)
    MetheatSKIN = np.full(imax, np.nan)
    TskinC = np.full(imax, np.nan)
    TskinCH = np.full(imax, np.nan)
    CBSMIN = np.full(imax, np.nan)
    CONDBS = np.full(imax, np.nan)

    # 2. Sweating
    DLC = np.full(imax, np.nan)
    DIFFC = np.full(imax, np.nan)
    RV = np.full(imax, np.nan)
    VPSKINTOT = np.full(imax, np.nan)
    VPSKINHALF = np.full(imax, np.nan)
    LASMAXENV = np.full(imax, np.nan)
    LASMAXPHYS = np.full(imax, np.nan)
    LASMAXCORR = np.full(imax, np.nan)
    ACTSW = np.full(imax, np.nan)
    ACTSWH = np.full(imax, np.nan)
    CSC = np.full(imax, np.nan)
    MetheatCOAT = np.full(imax, np.nan)
    TcoatC = np.full(imax, np.nan)
    TcoatCH = np.full(imax, np.nan)
    TcoatK = np.full(imax, np.nan)
    QSC = np.full(imax, np.nan)

    # 3. LWR
    LWRSKY = np.full(imax, np.nan)
    LWRENV = np.full(imax, np.nan)
    LB = np.full(imax, np.nan)
    LWRCOAT = np.full(imax, np.nan)
    LWRCOATH = np.full(imax, np.nan)

    # 4. Convection
    TAVGR = np.full(imax, np.nan)
    Ea = np.full(imax, np.nan)
    Ec = np.full(imax, np.nan)
    GRASHOF = np.full(imax, np.nan)
    WINDSP = np.full(imax, np.nan)
    REYNOLDS = np.full(imax, np.nan)
    ReH = np.full(imax, np.nan)
    ReL = np.full(imax, np.nan)
    NUSSELTH = np.full(imax, np.nan)
    NUSSELTL = np.full(imax, np.nan)
    NUSSELT = np.full(imax, np.nan)
    NUSSELTM = np.full(imax, np.nan)
    ka = np.full(imax, np.nan)
    CONVCOAT = np.full(imax, np.nan)
    CONVCOATH = np.full(imax, np.nan)

    # 5. Solar radiation
    SAAC = np.full(imax, np.nan)
    SWRS = np.full(imax, np.nan)
    SWRC = np.full(imax, np.nan)
    ISWRC = np.full(imax, np.nan)
    SWR = np.full(imax, np.nan)
    REFLE = np.full(imax, np.nan)
    RAINEVAP = np.full(imax, np.nan)

    # Synthesis
    MetheatAIR = np.full(imax, np.nan)
    Metheatopt = np.full(imax, np.nan)
    METABFEED = np.full(imax, np.nan)
    CHECKHEAT1 = np.full(imax, "", dtype=object)
    METABFEEDCH = np.full(imax, np.nan)
    METABFEEDC = np.full(imax, np.nan)
    Metheatcold = np.full(imax, np.nan)
    CHECKHEAT2 = np.full(imax, "", dtype=object)

    # Per-j outputs
    METTBWACT = np.full(int(jmax), np.nan)
    MAINTME = np.full(int(jmax), np.nan)
    TOTNE = np.full(int(jmax), np.nan)
    WM2 = np.full(int(jmax), np.nan)

    # cbind-style accumulators
    MATRIX = np.zeros((imax, 1), dtype=float)
    MATRIX1 = np.zeros((imax, 1), dtype=float)
    MATRIXSK = np.zeros((imax, 1), dtype=float)
    ###################  line:424 ######################################
    ##################### line: 429 ####################################
    for j in range(1, int(jmax) + 1):
        for i in range(1, imax + 1):
            ###########################################################################################
            # 2.                                   Dynamic section                                    #
            #                                     (time and animals)                                  #
            ###########################################################################################

            ###########################################################################################
            # 2.1                             Thermoregulation submodel                               #
            ###########################################################################################

            # Aim: To calculate the maximum and minimum heat release (W m-2) of an animal with its environment

            # Five flows of energy between an animal and its environment
            #   1. Latent and convective heat release from respiration
            #   2. Latent heat release from the skin
            #   3. Long wave radiation balance of the coat
            #   4. Convective heat losses from the coat
            #   5. Solar radiation intercepted by the coat

            ###########################################################################################
            # 2.1.1                             Maximum heat release                                  #
            ###########################################################################################

            # Heat release mechanisms of cattle at maximum heat release
            TISSUEFRAC = 1.00
            SWEATING = 1.00
            PANTING = 0.25

            # Calculations related to weather conditions
            TAVGC[i - 1] = TSTABLE[i - 1, j - 1]
            TAVGK[i - 1] = CtoK + TAVGC[i - 1]
            VPSATAIR[i - 1] = 6.1078 * 10 ** ((7.5 * TAVGC[i - 1]) / (TAVGC[i - 1] + 237.3)) * 100
            VPAIRTOT[i - 1] = VPSATAIR[i - 1] * RHSTABLE[i - 1, j - 1] / 100
            RHAIR[i - 1] = VPAIRTOT[i - 1] / VPSATAIR[i - 1]
            RHOVP[i - 1] = VPAIRTOT[i - 1] / (Rwater * TAVGK[i - 1])
            RHODAIR[i - 1] = (P - VPAIRTOT[i - 1]) / (Rdair * TAVGK[i - 1])
            RHOAIR[i - 1] = RHOVP[i - 1] + RHODAIR[i - 1]
            CHIAIR[i - 1] = RHOVP[i - 1] * RHOAIR[i - 1]

            ###########################################################################################
            #                1. Latent and convective heat release from respiration                   #
            ###########################################################################################

            # Animal
            AREA[i - 1] = 0.14 * (TBWACT[j - 1]) ** 0.57 * AREAFACTOR
            DIAMETER[i - 1] = 0.06 * TBWACT[j - 1] ** 0.39
            LENGTH[i - 1] = (AREA[i - 1] - pi * DIAMETER[i - 1] ** 2 / 2) / (pi * DIAMETER[i - 1])

            # In the thermoneutral zone
            brr[i - 1] = 73.8 * TBWACT[j - 1] ** (-0.286)
            btv[i - 1] = 0.0117 * TBWACT[j - 1]
            brv[i - 1] = brr[i - 1] * btv[i - 1]
            irv[i - 1] = brv[i - 1] + PANTING * ((RESPINCR - 1) * brv[i - 1])

            Texh[i - 1] = 17 + 0.3 * TAVGC[i - 1] + np.exp(0.01611 * RHAIR[i - 1] + 0.0387 * TAVGC[i - 1])

            # Assumption: exhaled air is saturated with water
            VPSATAIROUT[i - 1] = 6.1078 * 10 ** ((7.5 * Texh[i - 1]) / (Texh[i - 1] + 237.3)) * 100
            RHOVPOUT[i - 1] = VPSATAIROUT[i - 1] / (Rwater * (Texh[i - 1] + CtoK))
            RHODAIROUT[i - 1] = (P - VPSATAIROUT[i - 1]) / (Rdair * (Texh[i - 1] + CtoK))
            RHOAIROUT[i - 1] = RHOVPOUT[i - 1] + RHODAIROUT[i - 1]
            CHIAIROUT[i - 1] = RHOVPOUT[i - 1] * RHOAIROUT[i - 1]

            RHAIR[i - 1] = VPAIRTOT[i - 1] / VPSATAIR[i - 1] * 100

            AIREXCH[i - 1] = (irv[i - 1] * 60 * 24 / 1000 * RHOAIR[i - 1]) / AREA[i - 1]

            LHEATRESP[i - 1] = AIREXCH[i - 1] * L * (CHIAIROUT[i - 1] - CHIAIR[i - 1]) * kJdaytoW
            CHEATRESP[i - 1] = AIREXCH[i - 1] * Cp * (Texh[i - 1] - TAVGC[i - 1]) * kJdaytoW

            TGRESP[i - 1] = LHEATRESP[i - 1] + CHEATRESP[i - 1]

            # Energy for respiration (panting)
            NERESPWM[i - 1] = 1.1 * (RESPINCR * brr[i - 1]) ** 2.78 * 10 ** (-5) * PANTING
            NERESP[i - 1] = NERESPWM[i - 1] / kJdaytoW

            TNRESP[i - 1] = TGRESP[i - 1] - NERESPWM[i - 1]
            TNRESPH[i - 1] = TNRESP[i - 1]

            ###########################################################################################
            #                                    1a. Skin temperature                                 #
            ###########################################################################################

            # Resistance body core and skin
            CBSMIN[i - 1] = RBCSf / (0.03 * TBWACT[j - 1] ** 0.33)
            CONDBS[i - 1] = CBSMIN[i - 1] + TISSUEFRAC * (CBSMAX - CBSMIN[i - 1])

            ###########################################################################################
            #                            2. Latent heat release from the skin                         #
            ###########################################################################################

            # Latent energy flow between skin and air
            DLC[i - 1] = (CoatConst * WSSTABLE[i - 1, j - 1]) / (
                (CoatConst * WSSTABLE[i - 1, j - 1]) / LC + 1 / (ZC * LC)
            )
            DIFFC[i - 1] = 0.187 * 10 ** (-9) * TAVGK[i - 1] ** 2.072

            ###########################################################################################
            #                                    2a. Coat temperature                                 #
            ###########################################################################################

            # Resistance and conductivity between skin and coat
            CSC[i - 1] = 1 / (ZC * (LC - DLC[i - 1]) * (0.078 / 100))
            CSC[i - 1] = CSC[i - 1] / (1 - min(RAINFRAC, RAIN[i - 1, j - 1] * RAINFRAC / 24))

            ###########################################################################################
            #                            3. Long wave radiation from the coat                         #
            ###########################################################################################

            # Incoming LWR from the sky
            LWRSKY[i - 1] = (
                (1 - OCTA[i - 1, j - 1] / 8)
                * (SIGMA * TAVGK[i - 1] ** 4)
                * (1 - 0.261 * np.exp(-0.000777 * (273 - TAVGK[i - 1]) ** 2))
                + (OCTA[i - 1, j - 1] / 8) * (SIGMA * TAVGK[i - 1] ** 4 - 9)
            )

            # Incoming LWR from soil surface
            LWRENV[i - 1] = SIGMA * TAVGK[i - 1] ** 4

            if HOUSING == 0:
                LWRSKY[i - 1] = LWRENV[i - 1]

            ###########################################################################################
            #                          4. Convective heat losses from the coat                        #
            ###########################################################################################

            # Calculation of the air viscosity
            TAVGR[i - 1] = TAVGK[i - 1] * KtoR
            VISCAIR[i - 1] = MuSt * (
                ((0.555 * TR0 + CCONV1) / (0.555 * TAVGR[i - 1] + CCONV1))
                * (TAVGR[i - 1] / TR0) ** (3 / 2)
            )

            # Calculation of the grashof number
            Ea[i - 1] = VPAIRTOT[i - 1] * 10

            # Calculation of the Reynolds number
            WINDSP[i - 1] = WSSTABLE[i - 1, j - 1]
            REYNOLDS[i - 1] = WINDSP[i - 1] * DIAMETER[i - 1] * RHOAIR[i - 1] / VISCAIR[i - 1]

            ReH[i - 1] = 16 * REYNOLDS[i - 1] ** 2
            ReL[i - 1] = 0.1 * REYNOLDS[i - 1] ** 2
            ka[i - 1] = (
                1.5207 * 10 ** (-11) * TAVGK[i - 1] ** 3
                - 4.8574 * 10 ** (-8) * TAVGK[i - 1] ** 2
                + 1.0184 * 10 ** (-4) * TAVGK[i - 1]
                - 0.00039333
            )

            ###########################################################################################
            #                        5. Solar radiation intercepted by the coat                       #
            ###########################################################################################

            # Direct solar radiation
            SAAC[i - 1] = AHAdata[i - 1, j - 1]
            SWRS[i - 1] = SWRAD[i - 1, j - 1] * 1000 / (3600 * 24)
            SWRC[i - 1] = SWRS[i - 1] * SAAC[i - 1] * (1 - REFLC)

            # Indirect solar radiation
            if HOUSING == 1:
                REFLE[i - 1] = REFLEgrass
            elif HOUSING == 2:
                REFLE[i - 1] = REFLEconcr
            else:
                REFLE[i - 1] = 0

            ISWRC[i - 1] = 0.5 * REFLE[i - 1] * SWRAD[i - 1, j - 1] * 1000 / (3600 * 24)

            # Total solar radiation
            SWR[i - 1] = SWRC[i - 1] + ISWRC[i - 1]

            # Heat loss by evaporation of rain
            RAINEVAP[i - 1] = (
                0.15
                * (LENGTH[i - 1] * DIAMETER[i - 1]) / AREA[i - 1]
                * min(24, RAIN[i - 1, j - 1])
                * L
                * kJdaytoW
            )

            METABFEED[i - 1] = 170  

    ###################### line:601  #####################################
            while True:
                ###########################################################################################
                #                                    1a. Skin temperature                                 #
                ###########################################################################################

                MetheatSKIN[i - 1] = METABFEED[i - 1] - TNRESP[i - 1]
                TskinC[i - 1] = TbodyC - MetheatSKIN[i - 1] / CONDBS[i - 1]

                METABFEEDCH[i - 1] = METABFEED[i - 1]

                ###########################################################################################
                #                            2. Latent heat release from the skin                         #
                ###########################################################################################

                LASMAXPHYS[i - 1] = (
                    LASMIN
                    + LIBRARY[4] * np.exp(LIBRARY[5] * (TskinC[i - 1] - LIBRARY[7])) * L / 3600
                )

                RV[i - 1] = (
                    (LC - DLC[i - 1])
                    / (
                        DIFFC[i - 1]
                        * (
                            1
                            + 1.54
                            * ((LC - DLC[i - 1]) / DIAMETER[i - 1])
                            * (TskinC[i - 1] - min(TAVGC[i - 1], TskinC[i - 1])) ** 0.7
                        )
                    )
                )

                VPSKINTOT[i - 1] = (
                    6.1078 * 10 ** ((7.5 * TskinC[i - 1]) / (TskinC[i - 1] + 237.3)) * 100
                )

                LASMAXENV[i - 1] = (
                    (RHOAIR[i - 1] * Cp * 1000 / GAMMA)
                    * (VPSKINTOT[i - 1] - VPAIRTOT[i - 1])
                    / RV[i - 1]
                )

                LASMAXCORR[i - 1] = min(LASMAXPHYS[i - 1], LASMAXENV[i - 1])
                ACTSW[i - 1] = LASMIN + SWEATING * (LASMAXCORR[i - 1] - LASMIN)

                ACTSWH[i - 1] = ACTSW[i - 1]

                ###########################################################################################
                #                                    2a. Coat temperature                                 #
                ###########################################################################################

                MetheatCOAT[i - 1] = MetheatSKIN[i - 1] - ACTSW[i - 1]

                TcoatC[i - 1] = TskinC[i - 1] - MetheatCOAT[i - 1] / CSC[i - 1]
                TcoatK[i - 1] = TcoatC[i - 1] + CtoK

                ###########################################################################################
                #                            3. Long wave radiation from the coat                         #
                ###########################################################################################

                QSC[i - 1] = CSC[i - 1] * (TskinC[i - 1] - TcoatC[i - 1])
                LB[i - 1] = EMISS * SIGMA * TcoatK[i - 1] ** 4

                # LWR balance (net energy loss is a negative value)
                LWRCOAT[i - 1] = (
                    (EMISS * ((LWRSKY[i - 1] + LWRENV[i - 1]) / 2) - LB[i - 1])
                    / (1 - min(RAINFRAC, RAIN[i - 1, j - 1] * RAINFRAC / 24))
                )
                LWRCOATH[i - 1] = LWRCOAT[i - 1]

                ###########################################################################################
                #                          4. Convective heat losses from the coat                        #
                ###########################################################################################

                Ec[i - 1] = (
                    (6.1078 * 10 ** ((7.5 * TcoatC[i - 1]) / (TcoatC[i - 1] + 237.3))) + Ea[i - 1]
                ) / 2

                GRASHOF[i - 1] = (
                    GRAV * DIAMETER[i - 1] ** 3 * P / 100 * (TcoatC[i - 1] - TAVGC[i - 1])
                    + Schmidt * (Ec[i - 1] * TcoatC[i - 1] - Ea[i - 1] * TAVGC[i - 1])
                ) / (273 * P / 100 * VISCAIR[i - 1] ** 2)

                if GRASHOF[i - 1] > ReH[i - 1]:
                    NUSSELT[i - 1] = 0.48 * GRASHOF[i - 1] ** 0.25
                elif GRASHOF[i - 1] < ReL[i - 1]:
                    NUSSELT[i - 1] = 0.0112 * REYNOLDS[i - 1] ** 0.875
                else:
                    NUSSELT[i - 1] = max(
                        0.48 * GRASHOF[i - 1] ** 0.25,
                        0.0112 * REYNOLDS[i - 1] ** 0.875
                    )

                # Energy flow between coat and air
                CONVCOAT[i - 1] = (
                    (ka[i - 1] * NUSSELT[i - 1]) / DIAMETER[i - 1]
                    * (TcoatC[i - 1] - TAVGC[i - 1])
                    / (1 - min(RAINFRAC, RAIN[i - 1, j - 1] * RAINFRAC / 24))
                )
                CONVCOATH[i - 1] = CONVCOAT[i - 1]

                ###########################################################################################
                #                                         Synthesis                                       #
                ###########################################################################################

                MetheatAIR[i - 1] = (
                    MetheatCOAT[i - 1] + SWR[i - 1] - RAINEVAP[i - 1] + LWRCOAT[i - 1] - CONVCOAT[i - 1]
                )

                if MetheatAIR[i - 1] > 1:
                    METABFEED[i - 1] = METABFEED[i - 1] - 0.01 * MetheatAIR[i - 1]
                if MetheatAIR[i - 1] < -1:
                    METABFEED[i - 1] = METABFEED[i - 1] - 0.01 * MetheatAIR[i - 1]

                Metheatopt[i - 1] = METABFEED[i - 1]

                if MetheatAIR[i - 1] < 1 and MetheatAIR[i - 1] > -1:
                    CHECKHEAT1[i - 1] = "CORRECT"
                else:
                    CHECKHEAT1[i - 1] = "FALSE"

                if CHECKHEAT1[i - 1] == "CORRECT":
                    break
    ####################### line: 702 ##################################
            ###########################################################################################
            # 2.1.2                             Minimum heat release                                  #
            ###########################################################################################

            # Heat release mechanisms of cattle at minimum heat release
            TISSUEFRAC = 0.0
            SWEATING = 0.0
            PANTING = 0.0

            ###########################################################################################
            #                1. Latent and convective heat release from respiration                   #
            ###########################################################################################

            irv[i - 1] = brv[i - 1] + PANTING * (6.64 * brv[i - 1])

            AIREXCH[i - 1] = (irv[i - 1] * 60 * 24 / 1000 * RHOAIR[i - 1]) / AREA[i - 1]

            LHEATRESP[i - 1] = AIREXCH[i - 1] * L * (CHIAIROUT[i - 1] - CHIAIR[i - 1]) * 1000 / (24 * 3600)
            CHEATRESP[i - 1] = AIREXCH[i - 1] * Cp * (Texh[i - 1] - TAVGC[i - 1]) * 1000 / (24 * 3600)

            TGRESP[i - 1] = LHEATRESP[i - 1] + CHEATRESP[i - 1]

            TNRESP[i - 1] = TGRESP[i - 1]

            ###########################################################################################
            #                                    1a. Skin temperature                                 #
            ###########################################################################################

            CONDBS[i - 1] = CBSMIN[i - 1]

            ###########################################################################################
            #                            2. Latent heat release from the skin                         #
            ###########################################################################################

            ACTSW[i - 1] = LASMIN

            METABFEEDC[i - 1] = 80
    ####################### line: 741 ##################################
            while True:
                ###########################################################################################
                #                                    1a. Skin temperature                                 #
                ###########################################################################################

                # Skin temperature
                MetheatSKIN[i - 1] = METABFEEDC[i - 1] - TNRESP[i - 1]
                TskinC[i - 1] = TbodyC - MetheatSKIN[i - 1] / CONDBS[i - 1]
                TskinCH[i - 1] = TskinC[i - 1]

                ###########################################################################################
                #                                    2a. Coat temperature                                 #
                ###########################################################################################

                # Coat temperature
                MetheatCOAT[i - 1] = MetheatSKIN[i - 1] - ACTSW[i - 1]

                TcoatC[i - 1] = TskinC[i - 1] - MetheatCOAT[i - 1] / CSC[i - 1]
                TcoatK[i - 1] = TcoatC[i - 1] + CtoK
                TcoatCH[i - 1] = TcoatC[i - 1]

                ###########################################################################################
                #                            3. Long wave radiation from the coat                         #
                ###########################################################################################

                QSC[i - 1] = CSC[i - 1] * (TskinC[i - 1] - TcoatC[i - 1])

                LB[i - 1] = SIGMA * TcoatK[i - 1] ** 4

                # LWR balance (net energy loss is a negative value)
                LWRCOAT[i - 1] = (
                    (EMISS * ((LWRSKY[i - 1] + LWRENV[i - 1]) / 2) - EMISS * LB[i - 1])
                    / (1 - min(RAINFRAC, RAIN[i - 1, j - 1] * RAINFRAC / 24))
                )

                ###########################################################################################
                #                          4. Convective heat losses from the coat                        #
                ###########################################################################################

                Ec[i - 1] = (
                    (6.1078 * 10 ** ((7.5 * TcoatC[i - 1]) / (TcoatC[i - 1] + 237.3))) + Ea[i - 1]
                ) / 2

                GRASHOF[i - 1] = (
                    GRAV * DIAMETER[i - 1] ** 3 * P / 100 * (TcoatC[i - 1] - TAVGC[i - 1])
                    + Schmidt * (Ec[i - 1] * TcoatC[i - 1] - Ea[i - 1] * TAVGC[i - 1])
                ) / (273 * P / 100 * VISCAIR[i - 1] ** 2)

                if GRASHOF[i - 1] > ReH[i - 1]:
                    NUSSELT[i - 1] = 0.48 * GRASHOF[i - 1] ** 0.25
                elif GRASHOF[i - 1] < ReL[i - 1]:
                    NUSSELT[i - 1] = 0.0112 * REYNOLDS[i - 1] ** 0.875
                else:
                    NUSSELT[i - 1] = max(
                        0.48 * GRASHOF[i - 1] ** 0.25,
                        0.0112 * REYNOLDS[i - 1] ** 0.875
                    )

                # Energy flow between coat and air
                CONVCOAT[i - 1] = (
                    (ka[i - 1] * NUSSELT[i - 1]) / DIAMETER[i - 1]
                    * (TcoatC[i - 1] - TAVGC[i - 1])
                    / (1 - min(RAINFRAC, RAIN[i - 1, j - 1] * RAINFRAC / 24))
                )

                ###########################################################################################
                #                                         Synthesis                                       #
                ###########################################################################################

                MetheatAIR[i - 1] = (
                    MetheatCOAT[i - 1] + SWR[i - 1] - RAINEVAP[i - 1] + LWRCOAT[i - 1] - CONVCOAT[i - 1]
                )

                if MetheatAIR[i - 1] > 1:
                    METABFEEDC[i - 1] = METABFEEDC[i - 1] - 0.01 * MetheatAIR[i - 1]
                if MetheatAIR[i - 1] < -1:
                    METABFEEDC[i - 1] = METABFEEDC[i - 1] - 0.01 * MetheatAIR[i - 1]

                Metheatcold[i - 1] = METABFEEDC[i - 1]

                if MetheatAIR[i - 1] < 1 and MetheatAIR[i - 1] > -1:
                    CHECKHEAT2[i - 1] = "CORRECT"
                else:
                    CHECKHEAT2[i - 1] = "FALSE"

                if CHECKHEAT2[i - 1] == "CORRECT":
                    break
     ####################### line: 830 ##################################
            # after finishing the for i loop body

        MATRIX = np.column_stack((MATRIX, Metheatopt))
        MATRIX1 = np.column_stack((MATRIX1, Metheatcold))
        MATRIXSK = np.column_stack((MATRIXSK, TskinC))

        METTBWACT[j - 1] = (TBWACT[j - 1] * 0.9) ** 0.75    # Rumen is 10% of TBW
        MAINTME[j - 1] = 311 * METTBWACT[j - 1]             # ME requirement for maintenance (kJ day-1)
        TOTNE[j - 1] = MAINTME[j - 1] + MAINTME[j - 1] * MAINTT[j - 1]
        HIF = 0.30
        WM2[j - 1] = TOTNE[j - 1] * (1 + HIF / (1 - HIF)) / AREA[i - 1] * 1000 / (3600 * 24)
        ######################## line:840  ################################
    MATRIX = MATRIX[:, 1:]
    MATRIX1 = MATRIX1[:, 1:]
    MATRIXSK = MATRIXSK[:, 1:]

    MATRIXWM2 = np.tile(WM2, (imax, 1))

    # Model output
    x = TSTABLE.flatten(order="F")[:imax]
    y = RHSTABLE[0, :jmax]
    z = MATRIX

    RASTER = np.full((MATRIX.shape[0], MATRIX.shape[1]), 1, dtype=float)

    RASTER[MATRIX < MATRIXWM2] = 2
    RASTER[MATRIX1 > MATRIXWM2] = 0

    if GRAPHNR[s - 1] == 1:
        RASTERWIND = RASTER
    elif GRAPHNR[s - 1] == 2:
        RASTERRH = RASTER
    elif GRAPHNR[s - 1] == 3:
        RASTERSWR = RASTER
    elif GRAPHNR[s - 1] == 4:
        RASTERCC = RASTER
    elif GRAPHNR[s - 1] == 5:
        RASTERRAIN = RASTER
    elif GRAPHNR[s - 1] == 6:
        RASTERTBW = RASTER
    elif GRAPHNR[s - 1] == 7:
        RASTERHP = RASTER
        ######################## line:888 ###########################################
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        HAS_MATPLOTLIB = True
    except ModuleNotFoundError:
        HAS_MATPLOTLIB = False
        print("matplotlib is not installed. Plot generation is skipped.")

    if HAS_MATPLOTLIB:
        cmap = ListedColormap(["blue", "green", "red"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        # GRAPHNR == 1 : Wind speed
        if GRAPHNR[s - 1] == 1:
            fig, ax = plt.subplots(figsize=(6, 5))

            y_vals = np.arange(DWIND[0], DWIND[1] + DWIND[2] / 2.0, DWIND[2], dtype=float)

            ax.imshow(
                RASTERWIND,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
                extent=[x.min(), x.max(), y_vals.min(), y_vals.max()]
            )

            ax.set_xlabel(r"temperature ($^\circ$Celsius)")
            ax.set_ylabel(r"wind speed (ms$^{-1}$)")

            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor="blue", edgecolor="black"),
                plt.Rectangle((0, 0), 1, 1, facecolor="green", edgecolor="black"),
                plt.Rectangle((0, 0), 1, 1, facecolor="red", edgecolor="black"),
            ]

            ax.legend(
                legend_handles,
                ["<TNZ", "TNZ", ">TNZ"],
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
                fontsize=10
            )

            plt.tight_layout()
            plt.show()

        # GRAPHNR == 2 : Relative humidity
        if GRAPHNR[s - 1] == 2:
            fig, ax = plt.subplots(figsize=(6, 5))

            y_vals = np.arange(DRH[0], DRH[1] + DRH[2] / 2.0, DRH[2], dtype=float)

            ax.imshow(
                RASTERRH,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
                extent=[x.min(), x.max(), y_vals.min(), y_vals.max()]
            )

            ax.set_xlabel(r"temperature ($^\circ$Celsius)")
            ax.set_ylabel("relative humidity (%)")

            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor="blue", edgecolor="black"),
                plt.Rectangle((0, 0), 1, 1, facecolor="green", edgecolor="black"),
                plt.Rectangle((0, 0), 1, 1, facecolor="red", edgecolor="black"),
            ]

            ax.legend(
                legend_handles,
                ["<TNZ", "TNZ", ">TNZ"],
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
                fontsize=10
            )

            plt.tight_layout()
            plt.show()

        # GRAPHNR == 3 : Solar radiation
        if GRAPHNR[s - 1] == 3:
            fig, ax = plt.subplots(figsize=(6, 5))

            y_vals = np.arange(
                DSWR[0] / 1000.0,
                DSWR[1] / 1000.0 + (DSWR[2] / 1000.0) / 2.0,
                DSWR[2] / 1000.0,
                dtype=float
            )

            ax.imshow(
                RASTERSWR,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
                extent=[x.min(), x.max(), y_vals.min(), y_vals.max()]
            )

            ax.set_xlabel(r"temperature ($^\circ$Celsius)")
            ax.set_ylabel(r"solar radiation (MJ m$^{-2}$ day$^{-1}$)")

            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor="blue", edgecolor="black"),
                plt.Rectangle((0, 0), 1, 1, facecolor="green", edgecolor="black"),
                plt.Rectangle((0, 0), 1, 1, facecolor="red", edgecolor="black"),
            ]

            ax.legend(
                legend_handles,
                ["<TNZ", "TNZ", ">TNZ"],
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
                fontsize=10
            )

            plt.tight_layout()
            plt.show()

        # GRAPHNR == 4 : Cloud cover
        if GRAPHNR[s - 1] == 4:
            fig, ax = plt.subplots(figsize=(6, 5))

            y_vals = np.arange(DCC[0], DCC[1] + DCC[2] / 2.0, DCC[2], dtype=float)

            ax.imshow(
                RASTERCC,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
                extent=[x.min(), x.max(), y_vals.min(), y_vals.max()]
            )

            ax.set_xlabel(r"temperature ($^\circ$Celsius)")
            ax.set_ylabel("cloud cover   Ω")

            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor="blue", edgecolor="black"),
                plt.Rectangle((0, 0), 1, 1, facecolor="green", edgecolor="black"),
                plt.Rectangle((0, 0), 1, 1, facecolor="red", edgecolor="black"),
            ]

            ax.legend(
                legend_handles,
                ["<TNZ", "TNZ", ">TNZ"],
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
                fontsize=10
            )

            plt.tight_layout()
            plt.show()
    ######################### line: 998 #####################
    if GRAPHNR[s - 1] == 5:
        fig, ax = plt.subplots(figsize=(6, 5))

        y_vals = np.arange(DRAIN[0], DRAIN[1] + DRAIN[2] / 2.0, DRAIN[2], dtype=float)

        ax.imshow(
            RASTERRAIN,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[x.min(), x.max(), y_vals.min(), y_vals.max()]
        )

        ax.set_xlabel(r"temperature ($^\circ$Celsius)")
        ax.set_ylabel(r"precipitation (mm day$^{-1}$)")

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="blue", edgecolor="black"),
            plt.Rectangle((0, 0), 1, 1, facecolor="green", edgecolor="black"),
            plt.Rectangle((0, 0), 1, 1, facecolor="red", edgecolor="black"),
        ]

        ax.legend(
            legend_handles,
            ["<TNZ", "TNZ", ">TNZ"],
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            fontsize=10
        )

        plt.tight_layout()
        plt.show()
    ########################### line:1024 ###########################

    if GRAPHNR[s - 1] == 6:
        fig, ax = plt.subplots(figsize=(6, 5))

        y_vals = np.arange(
            DTBW[0],
            DTBW[1] + DTBW[2] / 2.0,
            DTBW[2],
            dtype=float
        )

        ax.imshow(
            RASTERTBW,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[x.min(), x.max(), y_vals.min(), y_vals.max()]
        )

        ax.set_xlabel(r"temperature ($^\circ$Celsius)")
        ax.set_ylabel("total body weight (kg)")

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="blue", edgecolor="black"),
            plt.Rectangle((0, 0), 1, 1, facecolor="green", edgecolor="black"),
            plt.Rectangle((0, 0), 1, 1, facecolor="red", edgecolor="black"),
        ]

        ax.legend(
            legend_handles,
            ["<TNZ", "TNZ", ">TNZ"],
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            fontsize=10
        )

        plt.tight_layout()
        plt.show()

    if GRAPHNR[s - 1] == 7:
        fig, ax = plt.subplots(figsize=(6, 5))

        y_vals = np.arange(
            DHP[0] + 1,
            DHP[1] + 1 + DHP[2] / 2.0,
            DHP[2],
            dtype=float
        )

        ax.imshow(
            RASTERHP,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[x.min(), x.max(), y_vals.min(), y_vals.max()]
        )

        ax.set_xlabel(r"temperature ($^\circ$Celsius)")
        ax.set_ylabel("heat production (x maintenance)")

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="blue", edgecolor="black"),
            plt.Rectangle((0, 0), 1, 1, facecolor="green", edgecolor="black"),
            plt.Rectangle((0, 0), 1, 1, facecolor="red", edgecolor="black"),
        ]

        ax.legend(
            legend_handles,
            ["<TNZ", "TNZ", ">TNZ"],
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            fontsize=10
        )

        plt.tight_layout()
        plt.show()
#**************************************************
###################### line:1082 ###################################
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False
    print("matplotlib is not installed. Figure generation is skipped.")


def _plot_panel(ax, raster, x_vals, y_vals, xlabel, ylabel, colors):
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    ax.imshow(
        raster,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", labelrotation=0)


def _plot_figure2(output_path, colors):
    fig, axes = plt.subplots(4, 2, figsize=(4.2, 8.4), dpi=200)
    axes = axes.flatten()

    # Solar radiation
    y_swr = np.arange(
        DSWR[0] / 1000.0,
        DSWR[1] / 1000.0 + (DSWR[2] / 1000.0) / 2.0,
        DSWR[2] / 1000.0,
        dtype=float
    )
    _plot_panel(
        axes[0],
        RASTERSWR,
        x,
        y_swr,
        r"Temperature ($^\circ$Celsius)",
        r"Solar radiation (MJ m$^{-2}$ day$^{-1}$)",
        colors
    )

    # Relative humidity
    y_rh = np.arange(DRH[0], DRH[1] + DRH[2] / 2.0, DRH[2], dtype=float)
    _plot_panel(
        axes[1],
        RASTERRH,
        x,
        y_rh,
        r"Temperature ($^\circ$Celsius)",
        "Relative humidity (%)",
        colors
    )

    # Wind speed
    y_wind = np.arange(DWIND[0], DWIND[1] + DWIND[2] / 2.0, DWIND[2], dtype=float)
    _plot_panel(
        axes[2],
        RASTERWIND,
        x,
        y_wind,
        r"Temperature ($^\circ$Celsius)",
        r"Wind speed (ms$^{-1}$)",
        colors
    )

    # Rain
    y_rain = np.arange(DRAIN[0], DRAIN[1] + DRAIN[2] / 2.0, DRAIN[2], dtype=float)
    _plot_panel(
        axes[3],
        RASTERRAIN,
        x,
        y_rain,
        r"Temperature ($^\circ$Celsius)",
        r"Precipitation (mm day$^{-1}$)",
        colors
    )

    # Cloud cover
    y_cc = np.arange(DCC[0], DCC[1] + DCC[2] / 2.0, DCC[2], dtype=float)
    _plot_panel(
        axes[4],
        RASTERCC,
        x,
        y_cc,
        r"Temperature ($^\circ$Celsius)",
        "Cloud cover  Ω",
        colors
    )

    # Total body weight
    y_tbw = np.arange(DTBW[0], DTBW[1] + DTBW[2] / 2.0, DTBW[2], dtype=float)
    _plot_panel(
        axes[5],
        RASTERTBW,
        x,
        y_tbw,
        r"Temperature ($^\circ$Celsius)",
        "Total body weight (kg)",
        colors
    )

    # Heat production
    y_hp = np.arange(DHP[0] + 1, DHP[1] + 1 + DHP[2] / 2.0, DHP[2], dtype=float)
    _plot_panel(
        axes[6],
        RASTERHP,
        x,
        y_hp,
        r"Temperature ($^\circ$Celsius)",
        "Heat production (x maintenance)",
        colors
    )

    # Legend panel
    axes[7].axis("off")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[0], edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[1], edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[2], edgecolor="black"),
    ]
    axes[7].legend(
        legend_handles,
        ["< TNZ", "TNZ", "> TNZ"],
        loc="upper left",
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=200)
    plt.close(fig)


if HAS_MATPLOTLIB:
    # Reproduces Figure 2 of the main paper in colour
    _plot_figure2("P2Fig2col.png", ["blue", "green", "red"])

    # Reproduces Figure 2 of the main paper in grayscale
    _plot_figure2("P2Fig2.png", ["grey", "white", "black"])
else:
    print("P2Fig2col.tiff and P2Fig2.tiff were not created because matplotlib is unavailable.")



