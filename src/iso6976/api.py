"""High-level entry point: :func:`calculate_properties`."""

from __future__ import annotations

import numpy as np

from . import _calc
from ._tables import (
    ALLOWED_T_HC,
    ALLOWED_T_S,
    M_AIR,
    N_COMPONENTS,
    P_REF,
    Z_AIR,
    idx_hc,
    idx_s,
)


def _normalize_temperature(
    t: float, allowed: tuple[float, ...], name: str
) -> float:
    """Snap 15.55 rounding noise, then check against the allowed values."""
    if 15.54 < t < 15.56:
        t = 15.55
    if t not in allowed:
        raise ValueError(f"{name} must be one of {allowed}")
    return t


def calculate_properties(
    composition: np.ndarray,
    uncertainty: np.ndarray,
    correlation: np.ndarray,
    *,
    combustion_temperature: float = 25.0,
    volume_temperature: float = 15.0,
    pressure: float = 101.325,
    coverage: float = 1.0,
) -> dict[str, float]:
    """Calculate natural gas properties per ISO 6976:2016.

    Parameters
    ----------
    composition
        Length-60 numeric array of mole fractions [mol/mol] in the
        component order of ISO 6976:2016 Table A.2. If values are given in
        mol %, divide by 100 before passing them here.
    uncertainty
        Length-60 numeric array of standard uncertainties of the mole
        fractions.
    correlation
        60x60 numeric matrix of correlation coefficients between component
        mole fractions. Pass ``numpy.eye(60)`` when correlations are
        unknown or assumed zero.
    combustion_temperature
        Combustion reference temperature in °C. Permitted: 0, 15, 15.55,
        20, 25. Default: 25.
    volume_temperature
        Volume reference temperature in °C. Permitted: 0, 15, 15.55, 20.
        Default: 15.
    pressure
        Reference pressure in kPa. Must be in [90, 110]. Default: 101.325.
    coverage
        Coverage factor *k*; uncertainties are multiplied by *k* before
        being returned. Default: 1 (standard uncertainty).

    Returns
    -------
    dict
        Named results (all numeric scalars). Keys follow the ISO 6976
        convention; see the module docstring of :mod:`iso6976` for the
        full list.
    """
    x = np.asarray(composition, dtype=np.float64)
    u = np.asarray(uncertainty, dtype=np.float64)
    r = np.asarray(correlation, dtype=np.float64)

    if x.shape != (N_COMPONENTS,):
        raise ValueError(f"composition must have length {N_COMPONENTS}")
    if u.shape != (N_COMPONENTS,):
        raise ValueError(f"uncertainty must have length {N_COMPONENTS}")
    if r.shape != (N_COMPONENTS, N_COMPONENTS):
        raise ValueError(
            f"correlation must be a {N_COMPONENTS}x{N_COMPONENTS} matrix"
        )
    if not 90.0 <= pressure <= 110.0:
        raise ValueError("pressure must be in the range 90–110 kPa")

    combustion_temperature = _normalize_temperature(
        combustion_temperature, ALLOWED_T_HC, "combustion_temperature",
    )
    volume_temperature = _normalize_temperature(
        volume_temperature, ALLOWED_T_S, "volume_temperature",
    )

    kh = idx_hc(combustion_temperature)
    ks = idx_s(volume_temperature)
    k = coverage

    ctx = _calc.build_context(x, u, r, kh, ks, pressure)
    if ctx.Z <= 0.9:
        raise ValueError("computed Z <= 0.9: outside application range")

    M, Hcg, Hcn, Vo, Z = ctx.M, ctx.Hcg, ctx.Hcn, ctx.Vo, ctx.Z

    # ---- deterministic values ----
    Hmg = Hcg / M
    Hmn = Hcn / M

    V = Z * Vo                # real molar volume, Eq. (11)
    Hvg_o = Hcg / Vo          # Eq. (7)  — ideal-gas vol. GCV
    Hvn_o = Hcn / Vo          # Eq. (9)  — ideal-gas vol. NCV
    Hvg = Hcg / V             # Eq. (10) — real-gas vol. GCV
    Hvn = Hcn / V             # Eq. (12) — real-gas vol. NCV

    G_o = M / M_AIR           # Eq. (13) — ideal relative density
    D_o = M / Vo              # Eq. (14) — ideal density

    Z_air_real = 1.0 - pressure / P_REF * (1.0 - Z_AIR[ks])   # Eq. (18)
    G = G_o * Z_air_real / Z                                  # Eq. (17)
    D = D_o / Z                                               # Eq. (19)

    sqrt_G_o = np.sqrt(G_o)
    sqrt_G = np.sqrt(G)
    Wg_o = Hvg_o / sqrt_G_o   # Eq. (15)
    Wn_o = Hvn_o / sqrt_G_o   # Eq. (16)
    Wg = Hvg / sqrt_G         # Eq. (20)
    Wn = Hvn / sqrt_G         # Eq. (21)

    # ---- uncertainties (all computed from the same Context) ----
    uHcg = _calc.u_Hc_o(ctx, gross=True)
    uHcn = _calc.u_Hc_o(ctx, gross=False)
    uHmg = _calc.u_Hm_o(ctx, gross=True)
    uHmn = _calc.u_Hm_o(ctx, gross=False)
    uHvg_o = _calc.u_Hv_o(ctx, gross=True)
    uHvn_o = _calc.u_Hv_o(ctx, gross=False)
    # Eq. (10)/(12): H_v = H_v^o / Z with Z treated as a constant factor, so
    # the same relative uncertainty applies to both.
    uHvg = uHvg_o / Z
    uHvn = uHvn_o / Z

    uD = _calc.u_D(ctx, D)
    uG = _calc.u_G(ctx, G)
    uWg = _calc.u_W(ctx, Wg, gross=True)
    uWn = _calc.u_W(ctx, Wn, gross=False)

    return {
        "M":      M,
        "Z":      Z,
        "G_o":    G_o,
        "D_o":    D_o,
        "G":      G,      "u_G":     k * uG,
        "D":      D,      "u_D":     k * uD,
        "Hcg":    Hcg,    "u_Hcg":   k * uHcg,
        "Hcn":    Hcn,    "u_Hcn":   k * uHcn,
        "Hmg":    Hmg,    "u_Hmg":   k * uHmg,
        "Hmn":    Hmn,    "u_Hmn":   k * uHmn,
        "Hvg_o":  Hvg_o,  "u_Hvg_o": k * uHvg_o,
        "Hvn_o":  Hvn_o,  "u_Hvn_o": k * uHvn_o,
        "Hvg":    Hvg,    "u_Hvg":   k * uHvg,
        "Hvn":    Hvn,    "u_Hvn":   k * uHvn,
        "Wg_o":   Wg_o,
        "Wn_o":   Wn_o,
        "Wg":     Wg,     "u_Wg":    k * uWg,
        "Wn":     Wn,     "u_Wn":    k * uWn,
    }
