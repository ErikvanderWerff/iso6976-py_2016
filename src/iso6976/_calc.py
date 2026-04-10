"""Core ISO 6976:2016 property and uncertainty calculations.

All equation numbers refer to ISO 6976:2016. "Annex B" refers to the
uncertainty-propagation annex of the same standard.

Design
------
A single :class:`Context` dataclass holds the composition vectors together
with every scalar and column slice that the uncertainty helpers need. It is
built once per ``calculate_properties`` call via :func:`build_context`; each
``u_*`` function then reads from it instead of recomputing the molar mass,
calorific values, molar volume, compression factor and related column
slices from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    U_HC_I,
    U_L_VAP,
    U_M,
    U_M_AIR,
    U_R_MOL,
    U_S_I,
    U_Z_AIR,
    Z_AIR,
)

# ---------------------------------------------------------------------------
# Shared computation context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """Precomputed scalars and column views used by every uncertainty helper.

    Attributes
    ----------
    x, u, r
        Composition vector, standard uncertainties and correlation matrix.
    kh, ks
        Column indices into Table A.4 and Table A.3 respectively.
    p2
        Reference pressure [kPa].
    M
        Molar mass M [kg/kmol], Eq. (5).
    Hcg, Hcn
        Molar gross / net calorific values [kJ/mol], Eqs. (2) and (3).
    Vo
        Ideal molar volume V_o [m^3/mol], Eq. (8).
    Z
        Compression factor, Eq. (1).
    sigma_Z
        Intermediate sqrt((1 - Z) * p2 / P_REF), Eq. (B.7).
    sum_nH
        Sum_i x_i * n_H,i — appears in every net-CV uncertainty term.
    hc_col, hn_col, s_col
        Column slices of the reference tables at the active temperatures.
        ``hn_col`` is the NCV coefficient vector ``hc_col - L/2 * N_H``.
    L, uL
        Enthalpy of vaporisation of water and its uncertainty [kJ/mol].
    """

    x: np.ndarray
    u: np.ndarray
    r: np.ndarray
    kh: int
    ks: int
    p2: float
    M: float
    Hcg: float
    Hcn: float
    Vo: float
    Z: float
    sigma_Z: float
    sum_nH: float
    hc_col: np.ndarray
    hn_col: np.ndarray
    s_col: np.ndarray
    L: float
    uL: float


def build_context(
    x: np.ndarray,
    u: np.ndarray,
    r: np.ndarray,
    kh: int,
    ks: int,
    p2: float,
) -> Context:
    """Precompute every scalar and column slice needed downstream."""
    hc_col = TAB_HC[:, kh]
    s_col = TAB_S[:, ks]
    L = float(L_VAP[kh])
    uL = float(U_L_VAP[kh])

    M = float(np.dot(x, M_I))
    sum_nH = float(np.dot(x, N_H))
    Hcg = float(np.dot(x, hc_col))
    Hcn = Hcg - sum_nH / 2.0 * L

    Vo = R_MOL * (T_S[ks] + T_ZERO) / p2

    sum_xs = float(np.dot(x, s_col))
    Z = 1.0 - p2 / P_REF * sum_xs * sum_xs
    sigma_Z = float(np.sqrt(max(0.0, (1.0 - Z) * p2 / P_REF)))

    hn_col = hc_col - (L / 2.0) * N_H

    return Context(
        x=x, u=u, r=r, kh=kh, ks=ks, p2=p2,
        M=M, Hcg=Hcg, Hcn=Hcn, Vo=Vo, Z=Z, sigma_Z=sigma_Z,
        sum_nH=sum_nH, hc_col=hc_col, hn_col=hn_col, s_col=s_col,
        L=L, uL=uL,
    )


# ---------------------------------------------------------------------------
# Internal quadratic-form helpers
# ---------------------------------------------------------------------------


def _quad_form(c: np.ndarray, u: np.ndarray, r: np.ndarray) -> float:
    """Compute (c*u)^T r (c*u) — the composition-driven part of Annex B."""
    w = c * u
    return float(w @ (r @ w))


def _table_Hc_term(x: np.ndarray) -> float:
    """Sum_i x_i^2 * u(H_ch,i)^2 — table-value uncertainty of GCV."""
    return float(np.sum((x * U_HC_I) ** 2))


def _table_s_term(x: np.ndarray) -> float:
    """Sum_i x_i^2 * u(s_i)^2 — table-value uncertainty of summing factor."""
    return float(np.sum((x * U_S_I) ** 2))


def _molar_mass_covariance(x: np.ndarray) -> float:
    """Sum_ij x_i u(M_i) r(M_i,M_j) x_j u(M_j), Eqs. (24)–(25)."""
    w = x * U_M
    return float(w @ (R_M @ w))


def _ncv_extra(ctx: Context, H: float) -> float:
    """Extra (sum_nH/(2H))^2 * u(L)^2 term shared by all NCV uncertainties."""
    return (ctx.sum_nH / (2.0 * H)) ** 2 * ctx.uL * ctx.uL


# ---------------------------------------------------------------------------
# Annex B: uncertainty of individual properties
# ---------------------------------------------------------------------------


def u_Hc_o(ctx: Context, *, gross: bool) -> float:
    """Eq. (B.4) / (B.8): standard uncertainty of molar GCV / NCV."""
    if gross:
        c = ctx.hc_col
        extra = 0.0
    else:
        c = ctx.hn_col
        extra = (ctx.sum_nH / 2.0) ** 2 * ctx.uL * ctx.uL

    s1 = _quad_form(c, ctx.u, ctx.r)
    s2 = _table_Hc_term(ctx.x)
    return float(np.sqrt(s1 + s2 + extra))


def u_Hm_o(ctx: Context, *, gross: bool) -> float:
    """Eq. (B.5) / (B.9): standard uncertainty of mass-basis GCV / NCV."""
    if gross:
        H, num, extra = ctx.Hcg, ctx.hc_col, 0.0
    else:
        H, num = ctx.Hcn, ctx.hn_col
        extra = _ncv_extra(ctx, H)

    Hm = H / ctx.M
    c = num / H - M_I / ctx.M

    tmp = (
        _quad_form(c, ctx.u, ctx.r)
        + _table_Hc_term(ctx.x) / (H * H)
        + _molar_mass_covariance(ctx.x) / (ctx.M * ctx.M)
        + extra
    )
    return float(np.sqrt(tmp) * Hm)


def u_Hv_o(ctx: Context, *, gross: bool) -> float:
    """Eq. (B.6) / (B.10): standard uncertainty of ideal-gas volumetric CV."""
    if gross:
        H, num, extra = ctx.Hcg, ctx.hc_col, 0.0
    else:
        H, num = ctx.Hcn, ctx.hn_col
        extra = _ncv_extra(ctx, H)

    Hv = H / ctx.Vo
    sg, Z = ctx.sigma_Z, ctx.Z
    c = num / H + 2.0 * ctx.s_col * sg / Z

    tmp = (
        _quad_form(c, ctx.u, ctx.r)
        + _table_Hc_term(ctx.x) / (H * H)
        + 4.0 * sg * sg * _table_s_term(ctx.x) / (Z * Z)
        + (U_R_MOL / R_MOL) ** 2
        + extra
    )
    return float(np.sqrt(tmp) * Hv)


def _u_density_like(ctx: Context, extra_term: float, scale: float) -> float:
    """Shared core of Eqs. (B.11) and (B.12) — density / relative density."""
    sg, Z = ctx.sigma_Z, ctx.Z
    c = M_I / ctx.M + 2.0 * ctx.s_col * sg / Z

    tmp = (
        _quad_form(c, ctx.u, ctx.r)
        + _molar_mass_covariance(ctx.x) / (ctx.M * ctx.M)
        + 4.0 * sg * sg * _table_s_term(ctx.x) / (Z * Z)
        + extra_term
    )
    return float(np.sqrt(tmp) * scale)


def u_D(ctx: Context, D: float) -> float:
    """Eq. (B.11): standard uncertainty of density rho (real gas)."""
    return _u_density_like(ctx, (U_R_MOL / R_MOL) ** 2, D)


def u_G(ctx: Context, G: float) -> float:
    """Eq. (B.12): standard uncertainty of relative density d (real gas)."""
    Z_air_k = Z_AIR[ctx.ks]
    uZa = U_Z_AIR[ctx.ks]
    return _u_density_like(ctx, (uZa / Z_air_k) ** 2, G)


def u_W(ctx: Context, W: float, *, gross: bool) -> float:
    """Eq. (B.13) / (B.14): standard uncertainty of gross / net Wobbe index."""
    Z_air_k = Z_AIR[ctx.ks]
    uZa = U_Z_AIR[ctx.ks]
    sg, Z = ctx.sigma_Z, ctx.Z

    if gross:
        H, num, extra = ctx.Hcg, ctx.hc_col, 0.0
    else:
        H, num = ctx.Hcn, ctx.hn_col
        extra = _ncv_extra(ctx, H)

    c = num / H + ctx.s_col * sg / Z - M_I / (2.0 * ctx.M)

    tmp = (
        _quad_form(c, ctx.u, ctx.r)
        + _table_Hc_term(ctx.x) / (H * H)
        + sg * sg * _table_s_term(ctx.x) / (Z * Z)
        + _molar_mass_covariance(ctx.x) / (4.0 * ctx.M * ctx.M)
        + (U_R_MOL / R_MOL) ** 2
        + (U_M_AIR / (2.0 * M_AIR)) ** 2
        + (uZa / (2.0 * Z_air_k)) ** 2
        + extra
    )
    return float(np.sqrt(tmp) * W)
