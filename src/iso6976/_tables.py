"""ISO 6976:2016 reference tables and physical constants.

All tables and constants are taken verbatim from ISO 6976:2016.

Table numbering in this module follows the standard:

* ``Table 1`` — molar masses and atomic indices of the 60 components
  (main body, clause 12).
* ``Table 2`` — summation factors s_j at the four reference temperatures
  (main body, clause 12).
* ``Table 3`` — ideal-gas gross calorific values at the five combustion
  reference temperatures (main body, clause 12).
* ``Table A.1`` — molar gas constant R (Annex A).
* ``Table A.2`` — atomic weights and their uncertainties (Annex A).
* ``Table A.3`` — molar mass of dry air (Annex A).
* ``Table A.4`` — compression factor Z_air at the four volume reference
  temperatures (Annex A).
* ``Table A.5`` — standard enthalpy of vaporisation of water at the five
  combustion reference temperatures (Annex A).

Formula and equation numbers in comments refer to the published standard.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Reference pressure p_0 = 101.325 kPa. Used as the common reference for
# Table 2 (note at foot of table) and for Z_air in Table A.4.
P_REF: float = 101.325       # [kPa]

# 0 °C in kelvin — used to convert the volume-reference temperatures to
# absolute temperature in Eq. (8).
T_ZERO: float = 273.15       # [K]

# Molar gas constant R, from ISO 6976:2016 Table A.1 (CODATA recommendation).
R_MOL:   float = 8.3144621   # [J/(mol·K)]
U_R_MOL: float = 0.0000075   # standard uncertainty of R

# Molar mass of dry air and its standard uncertainty, ISO 6976:2016 Table A.3.
M_AIR:   float = 28.96546    # [kg/kmol]
U_M_AIR: float = 0.00017

N_COMPONENTS: int = 60

# ---------------------------------------------------------------------------
# ISO 6976:2016 Table 1 — molar masses and atomic indices
# Columns: M [kg/kmol], a_j, b_j, c_j, d_j, e_j (C, H, N, O, S atom counts),
# plus three extra columns nHe, nNe, nAr not present in ISO Table 1. These
# are a deliberate extension: Table 1 only carries atomic indices for C, H,
# N, O and S, so strictly applying Formula (25) to helium, neon and argon
# would yield u(M_i) = 0 for those components. By labelling each noble-gas
# component with a single "atom" of itself (nHe = 1 for helium, etc.) the
# atomic-weight uncertainties for He/Ne/Ar from Table A.2 propagate through
# Formula (25) in the usual way, giving a small but physically justified
# non-zero u(M_i). The effect on end results is negligible, but this keeps
# the uncertainty propagation internally consistent.
#
# Row order matches Table 1 (components 1–60, as numbered in the standard);
# the trailing comment on each row gives the component name.
# ---------------------------------------------------------------------------

TAB_M: np.ndarray = np.array([
    #     M          nC  nH  nN nO nS nHe nNe nAr
    [ 16.04246,  1,  4, 0, 0, 0, 0, 0, 0],  #  1 methane
    [ 30.06904,  2,  6, 0, 0, 0, 0, 0, 0],  #  2 ethane
    [ 44.09562,  3,  8, 0, 0, 0, 0, 0, 0],  #  3 propane
    [ 58.12220,  4, 10, 0, 0, 0, 0, 0, 0],  #  4 n-butane
    [ 58.12220,  4, 10, 0, 0, 0, 0, 0, 0],  #  5 isobutane
    [ 72.14878,  5, 12, 0, 0, 0, 0, 0, 0],  #  6 n-pentane
    [ 72.14878,  5, 12, 0, 0, 0, 0, 0, 0],  #  7 isopentane
    [ 72.14878,  5, 12, 0, 0, 0, 0, 0, 0],  #  8 neopentane
    [ 86.17536,  6, 14, 0, 0, 0, 0, 0, 0],  #  9 n-hexane
    [ 86.17536,  6, 14, 0, 0, 0, 0, 0, 0],  # 10 2-methylpentane
    [ 86.17536,  6, 14, 0, 0, 0, 0, 0, 0],  # 11 3-methylpentane
    [ 86.17536,  6, 14, 0, 0, 0, 0, 0, 0],  # 12 2,2-dimethylbutane
    [ 86.17536,  6, 14, 0, 0, 0, 0, 0, 0],  # 13 2,3-dimethylbutane
    [100.20194,  7, 16, 0, 0, 0, 0, 0, 0],  # 14 n-heptane
    [114.22852,  8, 18, 0, 0, 0, 0, 0, 0],  # 15 n-octane
    [128.25510,  9, 20, 0, 0, 0, 0, 0, 0],  # 16 n-nonane
    [142.28168, 10, 22, 0, 0, 0, 0, 0, 0],  # 17 n-decane
    [ 28.05316,  2,  4, 0, 0, 0, 0, 0, 0],  # 18 ethylene
    [ 42.07974,  3,  6, 0, 0, 0, 0, 0, 0],  # 19 propylene
    [ 56.10632,  4,  8, 0, 0, 0, 0, 0, 0],  # 20 1-butene
    [ 56.10632,  4,  8, 0, 0, 0, 0, 0, 0],  # 21 cis-2-butene
    [ 56.10632,  4,  8, 0, 0, 0, 0, 0, 0],  # 22 trans-2-butene
    [ 56.10632,  4,  8, 0, 0, 0, 0, 0, 0],  # 23 isobutylene
    [ 70.13290,  5, 10, 0, 0, 0, 0, 0, 0],  # 24 1-pentene
    [ 40.06386,  3,  4, 0, 0, 0, 0, 0, 0],  # 25 propadiene
    [ 54.09044,  4,  6, 0, 0, 0, 0, 0, 0],  # 26 1,2-butadiene
    [ 54.09044,  4,  6, 0, 0, 0, 0, 0, 0],  # 27 1,3-butadiene
    [ 26.03728,  2,  2, 0, 0, 0, 0, 0, 0],  # 28 acetylene
    [ 70.13290,  5, 10, 0, 0, 0, 0, 0, 0],  # 29 cyclopentane
    [ 84.15948,  6, 12, 0, 0, 0, 0, 0, 0],  # 30 methylcyclopentane
    [ 98.18606,  7, 14, 0, 0, 0, 0, 0, 0],  # 31 ethylcyclopentane
    [ 84.15948,  6, 12, 0, 0, 0, 0, 0, 0],  # 32 cyclohexane
    [ 98.18606,  7, 14, 0, 0, 0, 0, 0, 0],  # 33 methylcyclohexane
    [112.21264,  8, 16, 0, 0, 0, 0, 0, 0],  # 34 ethylcyclohexane
    [ 78.11184,  6,  6, 0, 0, 0, 0, 0, 0],  # 35 benzene
    [ 92.13842,  7,  8, 0, 0, 0, 0, 0, 0],  # 36 toluene
    [106.16500,  8, 10, 0, 0, 0, 0, 0, 0],  # 37 ethylbenzene
    [106.16500,  8, 10, 0, 0, 0, 0, 0, 0],  # 38 o-xylene
    [ 32.04186,  1,  4, 0, 1, 0, 0, 0, 0],  # 39 methanol
    [ 48.10746,  1,  4, 0, 0, 1, 0, 0, 0],  # 40 methanethiol
    [  2.01588,  0,  2, 0, 0, 0, 0, 0, 0],  # 41 hydrogen
    [ 18.01528,  0,  2, 0, 1, 0, 0, 0, 0],  # 42 water
    [ 34.08088,  0,  2, 0, 0, 1, 0, 0, 0],  # 43 hydrogen sulphide
    [ 17.03052,  0,  3, 1, 0, 0, 0, 0, 0],  # 44 ammonia
    [ 27.02534,  1,  1, 1, 0, 0, 0, 0, 0],  # 45 hydrogen cyanide
    [ 28.0101 ,  1,  0, 0, 1, 0, 0, 0, 0],  # 46 carbon monoxide
    [ 60.0751 ,  1,  0, 0, 1, 1, 0, 0, 0],  # 47 carbonyl sulphide
    [ 76.1407 ,  1,  0, 0, 0, 2, 0, 0, 0],  # 48 carbon disulphide
    [  4.002602, 0,  0, 0, 0, 0, 1, 0, 0],  # 49 helium
    [ 20.1797 ,  0,  0, 0, 0, 0, 0, 1, 0],  # 50 neon
    [ 39.948  ,  0,  0, 0, 0, 0, 0, 0, 1],  # 51 argon
    [ 28.0134 ,  0,  0, 2, 0, 0, 0, 0, 0],  # 52 nitrogen
    [ 31.9988 ,  0,  0, 0, 2, 0, 0, 0, 0],  # 53 oxygen
    [ 44.0095 ,  1,  0, 0, 2, 0, 0, 0, 0],  # 54 carbon dioxide
    [ 64.0638 ,  0,  0, 0, 2, 1, 0, 0, 0],  # 55 sulphur dioxide
    [156.30826, 11, 24, 0, 0, 0, 0, 0, 0],  # 56 n-undecane
    [170.33484, 12, 26, 0, 0, 0, 0, 0, 0],  # 57 n-dodecane
    [184.36142, 13, 28, 0, 0, 0, 0, 0, 0],  # 58 n-tridecane
    [198.38800, 14, 30, 0, 0, 0, 0, 0, 0],  # 59 n-tetradecane
    [212.41458, 15, 32, 0, 0, 0, 0, 0, 0],  # 60 n-pentadecane
], dtype=np.float64)

# Molar mass of each component [kg/kmol] — column 0 of TAB_M.
M_I: np.ndarray = TAB_M[:, 0]

# Atom-count column views (1-D arrays of length 60) — columns 1–8 of TAB_M.
# Used in Formula (25) to propagate the uncertainty of the molar mass from
# the tabulated atomic-weight uncertainties.
N_C:  np.ndarray = TAB_M[:, 1]         # carbon atom count
N_H:  np.ndarray = TAB_M[:, 2]         # hydrogen atom count
N_N:  np.ndarray = TAB_M[:, 3]         # nitrogen atom count
N_O:  np.ndarray = TAB_M[:, 4]         # oxygen atom count
N_S:  np.ndarray = TAB_M[:, 5]         # sulphur atom count
N_HE: np.ndarray = TAB_M[:, 6]         # helium  "atom count" (1 for He, else 0)
N_NE: np.ndarray = TAB_M[:, 7]         # neon    "atom count" (1 for Ne, else 0)
N_AR: np.ndarray = TAB_M[:, 8]         # argon   "atom count" (1 for Ar, else 0)

# ---------------------------------------------------------------------------
# ISO 6976:2016 Table 2 — summation factors s_j
# Columns: s at 0 °C, 15 °C, 15.55 °C, 20 °C, and u(s).
# Row order matches Table 2 (components 1–60, as numbered in the standard);
# row comments identify each component. Used for the compression factor Z,
# Eq. (1). All values refer to the reference pressure P_REF (101.325 kPa).
# ---------------------------------------------------------------------------

TAB_S: np.ndarray = np.array([
    #   0 °C      15 °C     15.55 °C  20 °C     u(s)
    [0.04886, 0.04452, 0.04437, 0.04317, 0.0005],  #  1 methane
    [0.0997 , 0.0919 , 0.0916 , 0.0895 , 0.0011],  #  2 ethane
    [0.1465 , 0.1344 , 0.1340 , 0.1308 , 0.0016],  #  3 propane
    [0.2022 , 0.1840 , 0.1834 , 0.1785 , 0.0039],  #  4 n-butane
    [0.1885 , 0.1722 , 0.1717 , 0.1673 , 0.0031],  #  5 isobutane
    [0.2586 , 0.2361 , 0.2354 , 0.2295 , 0.0107],  #  6 n-pentane
    [0.2458 , 0.2251 , 0.2244 , 0.2189 , 0.0088],  #  7 isopentane
    [0.2245 , 0.2040 , 0.2033 , 0.1979 , 0.0060],  #  8 neopentane
    [0.3319 , 0.3001 , 0.2990 , 0.2907 , 0.0271],  #  9 n-hexane
    [0.3114 , 0.2826 , 0.2816 , 0.2740 , 0.0221],  # 10 2-methylpentane
    [0.2997 , 0.2762 , 0.2754 , 0.2690 , 0.0234],  # 11 3-methylpentane
    [0.2530 , 0.2350 , 0.2344 , 0.2295 , 0.0173],  # 12 2,2-dimethylbutane
    [0.2836 , 0.2632 , 0.2625 , 0.2569 , 0.0207],  # 13 2,3-dimethylbutane
    [0.4076 , 0.3668 , 0.3654 , 0.3547 , 0.1001],  # 14 n-heptane
    [0.4845 , 0.4346 , 0.4329 , 0.4198 , 0.1002],  # 15 n-octane
    [0.5617 , 0.5030 , 0.5010 , 0.4856 , 0.1006],  # 16 n-nonane
    [0.6713 , 0.5991 , 0.5967 , 0.5778 , 0.1006],  # 17 n-decane
    [0.0868 , 0.0799 , 0.0797 , 0.0778 , 0.0010],  # 18 ethylene
    [0.1381 , 0.1267 , 0.1263 , 0.1232 , 0.0016],  # 19 propylene
    [0.1964 , 0.1776 , 0.1770 , 0.1721 , 0.0041],  # 20 1-butene
    [0.2075 , 0.1870 , 0.1863 , 0.1810 , 0.0045],  # 21 cis-2-butene
    [0.2072 , 0.1868 , 0.1862 , 0.1809 , 0.0043],  # 22 trans-2-butene
    [0.1966 , 0.1777 , 0.1770 , 0.1721 , 0.0037],  # 23 isobutylene
    [0.2622 , 0.2297 , 0.2287 , 0.2208 , 0.0102],  # 24 1-pentene
    [0.1417 , 0.1313 , 0.1310 , 0.1282 , 0.0025],  # 25 propadiene
    [0.2063 , 0.1862 , 0.1855 , 0.1803 , 0.0110],  # 26 1,2-butadiene
    [0.1993 , 0.1739 , 0.1731 , 0.1673 , 0.0038],  # 27 1,3-butadiene
    [0.0936 , 0.0836 , 0.0833 , 0.0808 , 0.0024],  # 28 acetylene
    [0.2409 , 0.2221 , 0.2215 , 0.2164 , 0.0137],  # 29 cyclopentane
    [0.2817 , 0.2612 , 0.2605 , 0.2548 , 0.0262],  # 30 methylcyclopentane
    [0.4227 , 0.3684 , 0.3666 , 0.3531 , 0.1006],  # 31 ethylcyclopentane
    [0.2939 , 0.2686 , 0.2677 , 0.2610 , 0.0325],  # 32 cyclohexane
    [0.3667 , 0.3317 , 0.3305 , 0.3213 , 0.0668],  # 33 methylcyclohexane
    [0.5275 , 0.4547 , 0.4524 , 0.4345 , 0.1006],  # 34 ethylcyclohexane
    [0.2752 , 0.2527 , 0.2520 , 0.2460 , 0.0274],  # 35 benzene
    [0.3726 , 0.3359 , 0.3347 , 0.3251 , 0.1002],  # 36 toluene
    [0.4129 , 0.3797 , 0.3785 , 0.3694 , 0.1002],  # 37 ethylbenzene
    [0.4852 , 0.4411 , 0.4396 , 0.4277 , 0.1004],  # 38 o-xylene
    [0.5806 , 0.4464 , 0.4423 , 0.4117 , 0.0233],  # 39 methanol
    [0.1909 , 0.1700 , 0.1693 , 0.1640 , 0.0117],  # 40 methanethiol
    [-0.01  , -0.01  , -0.01  , -0.01  , 0.0250],  # 41 hydrogen
    [0.3093 , 0.2562 , 0.2546 , 0.2419 , 0.0150],  # 42 water
    [0.1006 , 0.0923 , 0.0920 , 0.0898 , 0.0023],  # 43 hydrogen sulphide
    [0.1230 , 0.1100 , 0.1096 , 0.1062 , 0.0021],  # 44 ammonia
    [0.3175 , 0.2765 , 0.2751 , 0.2644 , 0.0076],  # 45 hydrogen cyanide
    [0.0258 , 0.0217 , 0.0215 , 0.0203 , 0.0010],  # 46 carbon monoxide
    [0.1211 , 0.1114 , 0.1110 , 0.1084 , 0.0054],  # 47 carbonyl sulphide
    [0.2182 , 0.1958 , 0.1951 , 0.1894 , 0.0098],  # 48 carbon disulphide
    [-0.01  , -0.01  , -0.01  , -0.01  , 0.0250],  # 49 helium
    [-0.01  , -0.01  , -0.01  , -0.01  , 0.0250],  # 50 neon
    [0.0307 , 0.0273 , 0.0272 , 0.0262 , 0.0010],  # 51 argon
    [0.0214 , 0.0170 , 0.0169 , 0.0156 , 0.0010],  # 52 nitrogen
    [0.0311 , 0.0276 , 0.0275 , 0.0265 , 0.0010],  # 53 oxygen
    [0.0821 , 0.0752 , 0.0749 , 0.0730 , 0.0020],  # 54 carbon dioxide
    [0.1579 , 0.1406 , 0.1400 , 0.1356 , 0.0035],  # 55 sulphur dioxide
    [0.7228 , 0.6402 , 0.6374 , 0.6159 , 0.1006],  # 56 n-undecane
    [0.8567 , 0.7615 , 0.7583 , 0.7335 , 0.1006],  # 57 n-dodecane
    [0.9129 , 0.8061 , 0.8026 , 0.7748 , 0.1006],  # 58 n-tridecane
    [1.0135 , 0.8940 , 0.8900 , 0.8589 , 0.1006],  # 59 n-tetradecane
    [1.1176 , 0.9849 , 0.9804 , 0.9459 , 0.1006],  # 60 n-pentadecane
], dtype=np.float64)

# Volume reference temperatures for Table 2, in °C.
T_S: np.ndarray = np.array([0.0, 15.0, 15.55, 20.0], dtype=np.float64)

# Compression factor Z_air of dry air at (T_S, P_REF), ISO 6976:2016 Table A.4,
# with its standard uncertainty u(Z_air) (identical for all four temperatures).
Z_AIR:   np.ndarray = np.array([0.999419, 0.999595, 0.999601, 0.999645], dtype=np.float64)
U_Z_AIR: np.ndarray = np.array([0.000015, 0.000015, 0.000015, 0.000015], dtype=np.float64)

# Named view of Table 2 last column: standard uncertainty u(s_j).
U_S_I: np.ndarray = TAB_S[:, 4]

# ---------------------------------------------------------------------------
# ISO 6976:2016 Table 3 — ideal-gas gross calorific values H_ch,j^o [kJ/mol]
# Columns: at 0 °C, 15 °C, 15.55 °C, 20 °C, 25 °C, and u(H_ch).
# Row order matches Table 3 (components 1–60, as numbered in the standard);
# row comments identify each component. The seven zero rows (49–55) are the
# inert / non-combustible components: He, Ne, Ar, N₂, O₂, CO₂ and SO₂. ISO
# Table 3 simply omits them; here they are kept as explicit zero rows so
# every array in the module is length 60 in the same component order.
# ---------------------------------------------------------------------------

TAB_HC: np.ndarray = np.array([
    #    0 °C      15 °C     15.55 °C  20 °C     25 °C     u(H_ch)
    [  892.92,  891.51,  891.46,  891.05,  890.58, 0.19 ],  #  1 methane
    [ 1564.35, 1562.14, 1562.06, 1561.42, 1560.69, 0.51 ],  #  2 ethane
    [ 2224.03, 2221.10, 2220.99, 2220.13, 2219.17, 0.51 ],  #  3 propane
    [ 2883.35, 2879.76, 2879.63, 2878.58, 2877.40, 0.72 ],  #  4 n-butane
    [ 2874.21, 2870.58, 2870.45, 2869.39, 2868.20, 0.72 ],  #  5 isobutane
    [ 3542.91, 3538.60, 3538.45, 3537.19, 3535.77, 0.23 ],  #  6 n-pentane
    [ 3536.01, 3531.68, 3531.52, 3530.25, 3528.83, 0.23 ],  #  7 isopentane
    [ 3521.75, 3517.44, 3517.28, 3516.02, 3514.61, 0.25 ],  #  8 neopentane
    [ 4203.24, 4198.24, 4198.06, 4196.60, 4194.95, 0.32 ],  #  9 n-hexane
    [ 4195.64, 4190.62, 4190.44, 4188.97, 4187.32, 0.53 ],  # 10 2-methylpentane
    [ 4198.27, 4193.22, 4193.04, 4191.56, 4189.90, 0.53 ],  # 11 3-methylpentane
    [ 4185.86, 4180.83, 4180.65, 4179.17, 4177.52, 0.48 ],  # 12 2,2-dimethylbutane
    [ 4193.68, 4188.61, 4188.43, 4186.94, 4185.28, 0.46 ],  # 13 2,3-dimethylbutane
    [ 4862.88, 4857.18, 4856.98, 4855.31, 4853.43, 0.67 ],  # 14 n-heptane
    [ 5522.41, 5516.01, 5515.78, 5513.90, 5511.80, 0.76 ],  # 15 n-octane
    [ 6182.92, 6175.82, 6175.56, 6173.48, 6171.15, 0.81 ],  # 16 n-nonane
    [ 6842.69, 6834.90, 6834.62, 6832.33, 6829.77, 0.87 ],  # 17 n-decane
    [ 1413.55, 1412.12, 1412.07, 1411.65, 1411.18, 0.21 ],  # 18 ethylene
    [ 2061.57, 2059.43, 2059.35, 2058.73, 2058.02, 0.34 ],  # 19 propylene
    [ 2721.57, 2718.71, 2718.60, 2717.76, 2716.82, 0.39 ],  # 20 1-butene
    [ 2714.88, 2711.94, 2711.83, 2710.97, 2710.00, 0.50 ],  # 21 cis-2-butene
    [ 2711.09, 2708.26, 2708.16, 2707.33, 2706.40, 0.47 ],  # 22 trans-2-butene
    [ 2704.88, 2702.06, 2701.96, 2701.13, 2700.20, 0.42 ],  # 23 isobutylene
    [ 3381.32, 3377.76, 3377.63, 3376.59, 3375.42, 0.73 ],  # 24 1-pentene
    [ 1945.26, 1943.97, 1943.92, 1943.54, 1943.11, 0.60 ],  # 25 propadiene
    [ 2597.15, 2595.12, 2595.05, 2594.46, 2593.79, 0.40 ],  # 26 1,2-butadiene
    [ 2544.14, 2542.11, 2542.03, 2541.44, 2540.77, 0.41 ],  # 27 1,3-butadiene
    [ 1301.86, 1301.37, 1301.35, 1301.21, 1301.05, 0.32 ],  # 28 acetylene
    [ 3326.14, 3322.19, 3322.05, 3320.89, 3319.59, 0.36 ],  # 29 cyclopentane
    [ 3977.05, 3972.46, 3972.29, 3970.95, 3969.44, 0.56 ],  # 30 methylcyclopentane
    [ 4637.20, 4631.93, 4631.74, 4630.20, 4628.47, 0.71 ],  # 31 ethylcyclopentane
    [ 3960.68, 3956.02, 3955.85, 3954.49, 3952.96, 0.32 ],  # 32 cyclohexane
    [ 4609.33, 4604.08, 4603.89, 4602.36, 4600.64, 0.71 ],  # 33 methylcyclohexane
    [ 5272.76, 5266.90, 5266.69, 5264.97, 5263.05, 0.95 ],  # 34 ethylcyclohexane
    [ 3305.12, 3302.90, 3302.81, 3302.16, 3301.43, 0.27 ],  # 35 benzene
    [ 3952.77, 3949.83, 3949.72, 3948.86, 3947.89, 0.51 ],  # 36 toluene
    [ 4613.16, 4609.54, 4609.40, 4608.34, 4607.15, 0.66 ],  # 37 ethylbenzene
    [ 4602.18, 4598.64, 4598.52, 4597.48, 4596.31, 0.76 ],  # 38 o-xylene
    [  766.60,  765.09,  765.03,  764.59,  764.09, 0.13 ],  # 39 methanol
    [ 1241.64, 1240.28, 1240.23, 1239.84, 1239.39, 0.32 ],  # 40 methanethiol
    [  286.64,  286.15,  286.13,  285.99,  285.83, 0.02 ],  # 41 hydrogen
    [   45.064,  44.431,  44.408,  44.222,  44.013, 0.004],  # 42 water
    [  562.93,  562.38,  562.36,  562.19,  562.01, 0.23 ],  # 43 hydrogen sulphide
    [  384.57,  383.51,  383.47,  383.16,  382.81, 0.18 ],  # 44 ammonia
    [  671.92,  671.67,  671.66,  671.58,  671.50, 1.26 ],  # 45 hydrogen cyanide
    [  282.80,  282.91,  282.91,  282.95,  282.98, 0.06 ],  # 46 carbon monoxide
    [  548.01,  548.14,  548.15,  548.19,  548.23, 0.24 ],  # 47 carbonyl sulphide
    [ 1104.05, 1104.32, 1104.33, 1104.40, 1104.49, 0.43 ],  # 48 carbon disulphide
    [    0.0 ,    0.0 ,    0.0 ,    0.0 ,    0.0 , 0.0  ],  # 49 helium         (inert)
    [    0.0 ,    0.0 ,    0.0 ,    0.0 ,    0.0 , 0.0  ],  # 50 neon           (inert)
    [    0.0 ,    0.0 ,    0.0 ,    0.0 ,    0.0 , 0.0  ],  # 51 argon          (inert)
    [    0.0 ,    0.0 ,    0.0 ,    0.0 ,    0.0 , 0.0  ],  # 52 nitrogen       (inert)
    [    0.0 ,    0.0 ,    0.0 ,    0.0 ,    0.0 , 0.0  ],  # 53 oxygen         (non-combustible)
    [    0.0 ,    0.0 ,    0.0 ,    0.0 ,    0.0 , 0.0  ],  # 54 carbon dioxide (non-combustible)
    [    0.0 ,    0.0 ,    0.0 ,    0.0 ,    0.0 , 0.0  ],  # 55 sulphur dioxide (non-combustible)
    [ 7502.22, 7493.73, 7493.42, 7490.93, 7488.14, 1.54 ],  # 56 n-undecane
    [ 8162.43, 8153.24, 8152.91, 8150.21, 8147.19, 1.13 ],  # 57 n-dodecane
    [ 8821.88, 8811.99, 8811.63, 8808.73, 8805.48, 1.21 ],  # 58 n-tridecane
    [ 9481.71, 9471.12, 9470.73, 9467.63, 9464.15, 1.32 ],  # 59 n-tetradecane
    [10141.65,10130.23,10129.82,10126.52,10122.82, 1.44 ],  # 60 n-pentadecane
], dtype=np.float64)

# Combustion reference temperatures for Table 3, in °C.
T_HC: np.ndarray = np.array([0.0, 15.0, 15.55, 20.0, 25.0], dtype=np.float64)

# Named view of Table 3 last column: standard uncertainty u(H_ch,j) [kJ/mol].
U_HC_I: np.ndarray = TAB_HC[:, 5]

# Standard enthalpy of vaporisation of water at the T_HC temperatures, and
# its standard uncertainty — ISO 6976:2016 Table A.5. Used for the net-CV
# correction in Eq. (3).
L_VAP:   np.ndarray = np.array([45.064, 44.431, 44.408, 44.222, 44.013], dtype=np.float64)
U_L_VAP: np.ndarray = np.array([ 0.004,  0.004,  0.004,  0.004,  0.004], dtype=np.float64)

# ---------------------------------------------------------------------------
# Derived: molar-mass uncertainties u(M_i) and correlations r(M_i,M_j)
# ISO 6976:2016 Eqs. (24)–(25), using the atomic-weight uncertainties from
# Table A.2. The ordered sequence below is the single source of truth for
# the (element, atom-count-column, u(A)) mapping used by both Eq. (24) and
# Eq. (25); reordering or extending it is the only place that needs to
# change if a new element is added.
# ---------------------------------------------------------------------------

_ATOMIC_CONTRIBUTIONS: tuple[tuple[str, np.ndarray, float], ...] = (
    ("C",  N_C,  0.0004),
    ("H",  N_H,  0.000035),
    ("N",  N_N,  0.0001),
    ("O",  N_O,  0.00015),
    ("S",  N_S,  0.0025),
    ("He", N_HE, 0.000001),
    ("Ne", N_NE, 0.0003),
    ("Ar", N_AR, 0.0005),
)


def _build_molar_mass_uncertainty() -> tuple[np.ndarray, np.ndarray]:
    """Return (u_M, r_M) — shapes (60,) and (60, 60)."""
    atoms = np.stack(
        [counts for _, counts, _ in _ATOMIC_CONTRIBUTIONS], axis=1,
    )  # (60, K)
    u_atoms = np.array(
        [u for _, _, u in _ATOMIC_CONTRIBUTIONS], dtype=np.float64,
    )  # (K,)

    # Eq. (25): u(M_i)^2 = sum_k n_ik^2 * u_ak^2
    u_M = np.sqrt((atoms ** 2) @ (u_atoms ** 2))

    # Eq. (24): cov(M_i, M_j) = sum_k n_ik * n_jk * u_ak^2
    cov = (atoms * u_atoms ** 2) @ atoms.T  # (60, 60)

    # r(M_i, M_j) = cov / (u_Mi * u_Mj); handle zero-uncertainty components.
    denom = np.outer(u_M, u_M)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_M = np.where(denom > 0.0, cov / denom, 0.0)

    return u_M, r_M


U_M: np.ndarray
R_M: np.ndarray
U_M, R_M = _build_molar_mass_uncertainty()


# ---------------------------------------------------------------------------
# Temperature-index lookup helpers
# ---------------------------------------------------------------------------

# Allowed combustion temperatures (Table 3) and volume temperatures (Table 2).
# Derived from T_HC / T_S so the two live in exactly one place. Callers are
# expected to pass an exact match; tolerance handling for 15.55 rounding
# noise lives in the public API layer.
ALLOWED_T_HC: tuple[float, ...] = tuple(T_HC.tolist())
ALLOWED_T_S:  tuple[float, ...] = tuple(T_S.tolist())

_HC_INDEX: dict[float, int] = {t: i for i, t in enumerate(ALLOWED_T_HC)}
_S_INDEX:  dict[float, int] = {t: i for i, t in enumerate(ALLOWED_T_S)}


def idx_hc(t: float) -> int:
    """Return column index into Table 3 for combustion temperature t [°C]."""
    return _HC_INDEX[t]


def idx_s(t: float) -> int:
    """Return column index into Table 2 / Z_air for volume temperature t [°C]."""
    return _S_INDEX[t]
