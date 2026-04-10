"""Core ISO 6976:2016 property and uncertainty calculations.

All equation numbers refer to ISO 6976:2016. "Annex B" refers to the
uncertainty-propagation annex of the same standard.
"""

from __future__ import annotations

import numpy as np

from ._tables import (
    L_VAP,
    M_AIR,
    M_I,
    N_H,
    P_REF,
    R_M,
    R_MOL,
    T_S,
    T_ZERO,
    TAB_HC,
    TAB_S,
    U_L_VAP,
    U_M,
    U_M_AIR,
    U_R_MOL,
    U_Z_AIR,
    Z_AIR,
)

# ---------------------------------------------------------------------------
# Core property calculations (Section 5)
# ---------------------------------------------------------------------------


def compression_factor(x: np.ndarray, p2: float, ks: int) -> float:
    """Eq. (1): compression factor Z."""
    s = TAB_S[:, ks]
    sum_xs = float(np.dot(x, s))
    return 1.0 - p2 / P_REF * sum_xs * sum_xs


def molar_gcv(x: np.ndarray, kh: int) -> float:
    """Eq. (2): ideal-gas gross calorific value H_ch^o [kJ/mol]."""
    return float(np.dot(x, TAB_HC[:, kh]))


def molar_ncv(x: np.ndarray, kh: int) -> float:
    """Eq. (3): ideal-gas net calorific value H_cn^o [kJ/mol]."""
    sum_nH = float(np.dot(x, N_H))
    return molar_gcv(x, kh) - sum_nH / 2.0 * L_VAP[kh]


def molar_mass(x: np.ndarray) -> float:
    """Eq. (5): molar mass M [kg/kmol]."""
    return float(np.dot(x, M_I))


def ideal_molar_volume(t2: float, p2: float) -> float:
    """Eq. (8): ideal molar volume V_o = R*T / p [m^3/mol]."""
    return R_MOL * (t2 + T_ZERO) / p2


# ---------------------------------------------------------------------------
# Uncertainty helpers
# ---------------------------------------------------------------------------


def _quad_form(c: np.ndarray, u: np.ndarray, r: np.ndarray) -> float:
    """Compute (c*u)^T r (c*u) — used in Annex B composition-driven terms."""
    w = c * u
    return float(w @ (r @ w))


def _sigma_Z(Z: float, p2: float) -> float:
    """Intermediate sqrt((1 - Z) * p2 / p_ref), Eq. (B.7)."""
    return float(np.sqrt((1.0 - Z) * p2 / P_REF))


def _table_Hc_term(x: np.ndarray) -> float:
    """Sum_i x_i^2 * u(H_ch,i)^2 — table-value uncertainty of GCV."""
    return float(np.sum((x * TAB_HC[:, 5]) ** 2))


def _table_s_term(x: np.ndarray) -> float:
    """Sum_i x_i^2 * u(s_i)^2 — table-value uncertainty of summing factor."""
    return float(np.sum((x * TAB_S[:, 4]) ** 2))


def _molar_mass_covariance(x: np.ndarray) -> float:
    """Sum_ij x_i * u(M_i) * r(M_i,M_j) * x_j * u(M_j), Eqs. (24)–(25)."""
    w = x * U_M
    return float(w @ (R_M @ w))


# ---------------------------------------------------------------------------
# Annex B: uncertainty of individual properties
# ---------------------------------------------------------------------------


def u_Hc_o_G(x: np.ndarray, u: np.ndarray, r: np.ndarray, kh: int) -> float:
    """Eq. (B.4): standard uncertainty of molar GCV, H_ch^o."""
    s1 = _quad_form(TAB_HC[:, kh], u, r)
    s2 = _table_Hc_term(x)
    return float(np.sqrt(s1 + s2))


def u_Hm_o_G(x: np.ndarray, u: np.ndarray, r: np.ndarray, kh: int) -> float:
    """Eq. (B.5): standard uncertainty of mass-basis GCV, H_cm^o."""
    Hg = molar_gcv(x, kh)
    M = molar_mass(x)
    Hm = Hg / M

    c = TAB_HC[:, kh] / Hg - M_I / M
    s1 = _quad_form(c, u, r)
    s2 = _table_Hc_term(x)
    s3 = _molar_mass_covariance(x)

    return float(np.sqrt(s1 + s2 / (Hg * Hg) + s3 / (M * M)) * Hm)


def u_Hv_o_G(
    x: np.ndarray,
    u: np.ndarray,
    r: np.ndarray,
    kh: int,
    ks: int,
    Z: float,
    p2: float,
) -> float:
    """Eq. (B.6): standard uncertainty of volumetric GCV, H_cv^o (ideal gas)."""
    Hg = molar_gcv(x, kh)
    Vox = ideal_molar_volume(T_S[ks], p2)
    Hv = Hg / Vox
    sg = _sigma_Z(Z, p2)

    c = TAB_HC[:, kh] / Hg + 2.0 * TAB_S[:, ks] * sg / Z
    s1 = _quad_form(c, u, r)
    s2 = _table_Hc_term(x)
    s3 = _table_s_term(x)

    tmp = (
        s1
        + s2 / (Hg * Hg)
        + 4.0 * sg * sg * s3 / (Z * Z)
        + (U_R_MOL / R_MOL) ** 2
    )
    return float(np.sqrt(tmp) * Hv)


def u_Hc_o_N(x: np.ndarray, u: np.ndarray, r: np.ndarray, kh: int) -> float:
    """Eq. (B.8): standard uncertainty of molar NCV, H_cn^o."""
    L = L_VAP[kh]
    uL = U_L_VAP[kh]

    c = TAB_HC[:, kh] - (L / 2.0) * N_H
    s1 = _quad_form(c, u, r)
    s2 = _table_Hc_term(x)
    sum_nH = float(np.dot(x, N_H))

    return float(np.sqrt(s1 + s2 + (sum_nH / 2.0) ** 2 * uL * uL))


def u_Hm_o_N(x: np.ndarray, u: np.ndarray, r: np.ndarray, kh: int) -> float:
    """Eq. (B.9): standard uncertainty of mass-basis NCV, H_cm^n."""
    Hn = molar_ncv(x, kh)
    M = molar_mass(x)
    Hmn = Hn / M
    L = L_VAP[kh]
    uL = U_L_VAP[kh]

    c = (TAB_HC[:, kh] - (L / 2.0) * N_H) / Hn - M_I / M
    s1 = _quad_form(c, u, r)
    s2 = _table_Hc_term(x)
    s3 = _molar_mass_covariance(x)
    sum_nH = float(np.dot(x, N_H))

    tmp = (
        s1
        + s2 / (Hn * Hn)
        + s3 / (M * M)
        + (sum_nH / (2.0 * Hn)) ** 2 * uL * uL
    )
    return float(np.sqrt(tmp) * Hmn)


def u_Hv_o_N(
    x: np.ndarray,
    u: np.ndarray,
    r: np.ndarray,
    kh: int,
    ks: int,
    Z: float,
    p2: float,
) -> float:
    """Eq. (B.10): standard uncertainty of volumetric NCV, H_cv^n (ideal gas)."""
    Hn = molar_ncv(x, kh)
    Vox = ideal_molar_volume(T_S[ks], p2)
    Hvn = Hn / Vox
    L = L_VAP[kh]
    uL = U_L_VAP[kh]
    sg = _sigma_Z(Z, p2)

    c = (TAB_HC[:, kh] - (L / 2.0) * N_H) / Hn + 2.0 * TAB_S[:, ks] * sg / Z
    s1 = _quad_form(c, u, r)
    s2 = _table_Hc_term(x)
    s3 = _table_s_term(x)
    sum_nH = float(np.dot(x, N_H))

    tmp = (
        s1
        + s2 / (Hn * Hn)
        + 4.0 * sg * sg * s3 / (Z * Z)
        + (U_R_MOL / R_MOL) ** 2
        + (sum_nH / (2.0 * Hn)) ** 2 * uL * uL
    )
    return float(np.sqrt(tmp) * Hvn)


def u_D(
    x: np.ndarray,
    u: np.ndarray,
    r: np.ndarray,
    ks: int,
    Z: float,
    D: float,
    p2: float,
) -> float:
    """Eq. (B.11): standard uncertainty of density rho (real gas)."""
    M = molar_mass(x)
    sg = _sigma_Z(Z, p2)

    c = M_I / M + 2.0 * TAB_S[:, ks] * sg / Z
    s1 = _quad_form(c, u, r)
    s2 = _molar_mass_covariance(x)
    s3 = _table_s_term(x)

    tmp = (
        s1
        + s2 / (M * M)
        + 4.0 * sg * sg * s3 / (Z * Z)
        + (U_R_MOL / R_MOL) ** 2
    )
    return float(np.sqrt(tmp) * D)


def u_G(
    x: np.ndarray,
    u: np.ndarray,
    r: np.ndarray,
    ks: int,
    Z: float,
    G: float,
    p2: float,
) -> float:
    """Eq. (B.12): standard uncertainty of relative density d (real gas)."""
    M = molar_mass(x)
    Z_air_k = Z_AIR[ks]
    uZa = U_Z_AIR[ks]
    sg = _sigma_Z(Z, p2)

    c = M_I / M + 2.0 * TAB_S[:, ks] * sg / Z
    s1 = _quad_form(c, u, r)
    s2 = _molar_mass_covariance(x)
    s3 = _table_s_term(x)

    tmp = (
        s1
        + s2 / (M * M)
        + 4.0 * sg * sg * s3 / (Z * Z)
        + (uZa / Z_air_k) ** 2
    )
    return float(np.sqrt(tmp) * G)


def u_W_G(
    x: np.ndarray,
    u: np.ndarray,
    r: np.ndarray,
    kh: int,
    ks: int,
    Z: float,
    W_G: float,
    p2: float,
) -> float:
    """Eq. (B.13): standard uncertainty of gross Wobbe index."""
    Hg = molar_gcv(x, kh)
    M = molar_mass(x)
    Z_air_k = Z_AIR[ks]
    uZa = U_Z_AIR[ks]
    sg = _sigma_Z(Z, p2)

    c = TAB_HC[:, kh] / Hg + TAB_S[:, ks] * sg / Z - M_I / (2.0 * M)
    s1 = _quad_form(c, u, r)
    s2 = _table_Hc_term(x)
    s3 = _table_s_term(x)
    s4 = _molar_mass_covariance(x)

    tmp = (
        s1
        + s2 / (Hg * Hg)
        + sg * sg * s3 / (Z * Z)
        + s4 / (4.0 * M * M)
        + (U_R_MOL / R_MOL) ** 2
        + (U_M_AIR / (2.0 * M_AIR)) ** 2
        + (uZa / (2.0 * Z_air_k)) ** 2
    )
    return float(np.sqrt(tmp) * W_G)


def u_W_N(
    x: np.ndarray,
    u: np.ndarray,
    r: np.ndarray,
    kh: int,
    ks: int,
    Z: float,
    W_N: float,
    p2: float,
) -> float:
    """Eq. (B.14): standard uncertainty of net Wobbe index."""
    Hn = molar_ncv(x, kh)
    M = molar_mass(x)
    L = L_VAP[kh]
    uL = U_L_VAP[kh]
    Z_air_k = Z_AIR[ks]
    uZa = U_Z_AIR[ks]
    sg = _sigma_Z(Z, p2)

    c = (TAB_HC[:, kh] - (L / 2.0) * N_H) / Hn + TAB_S[:, ks] * sg / Z - M_I / (2.0 * M)
    s1 = _quad_form(c, u, r)
    s2 = _table_Hc_term(x)
    s3 = _table_s_term(x)
    s4 = _molar_mass_covariance(x)
    sum_nH = float(np.dot(x, N_H))

    tmp = (
        s1
        + s2 / (Hn * Hn)
        + sg * sg * s3 / (Z * Z)
        + s4 / (4.0 * M * M)
        + (U_R_MOL / R_MOL) ** 2
        + (U_M_AIR / (2.0 * M_AIR)) ** 2
        + (uZa / (2.0 * Z_air_k)) ** 2
        + (sum_nH / (2.0 * Hn)) ** 2 * uL * uL
    )
    return float(np.sqrt(tmp) * W_N)
