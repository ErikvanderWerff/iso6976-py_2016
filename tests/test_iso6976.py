"""Validation tests for the Python port against ISO 6976:2016 Annex D.

Expected numerical values are taken from the upstream R package
``ISO6976.2016`` test suite, which is itself validated against the worked
examples in Annex D of the standard. See the comments on Example 2 and
Example 3_ex in the R tests for known-issue notes copied verbatim here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iso6976 import (
    calculate_properties,
    component_index,
    component_name,
    component_names,
)

DATA_DIR = Path(__file__).parent / "data"


def _load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.load(DATA_DIR / f"{name}.npz")
    return f["fractionArray"], f["uncertaintyArray"], f["correlationMatrix"]


# ---------------------------------------------------------------------------
# Annex D Example 1 — 5-component mixture, 15/15 °C
# ---------------------------------------------------------------------------


def test_example1_15_15():
    x, u, r = _load("example1")
    res = calculate_properties(
        x, u, r,
        combustion_temperature=15, volume_temperature=15,
    )

    assert res["M"]     == pytest.approx(17.3884301,   rel=1e-7)
    assert res["Z"]     == pytest.approx(0.99776224,   rel=1e-8)
    assert res["Hcg"]   == pytest.approx(906.1799588,  rel=1e-7)
    assert res["u_Hcg"] == pytest.approx(0.615609872,  rel=1e-9)
    assert res["Hmg"]   == pytest.approx(52.113961,    rel=1e-6)
    assert res["u_Hmg"] == pytest.approx(0.024301,     rel=1e-5)
    assert res["Hvg"]   == pytest.approx(38.410611,    rel=1e-6)
    assert res["u_Hvg"] == pytest.approx(0.026267,     rel=1e-5)


# ---------------------------------------------------------------------------
# Annex D Example 2 — 15.55/15.55 °C (mixture with water vapour).
# Known mismatch vs. standard for Z: -1.78e-5
# ---------------------------------------------------------------------------


def test_example2_15_55():
    x, u, r = _load("example2")
    res = calculate_properties(
        x, u, r,
        combustion_temperature=15.55, volume_temperature=15.55,
    )

    assert res["M"]     == pytest.approx(16.9891697,   rel=1e-7)
    assert res["Z"]     == pytest.approx(0.9975690,    rel=1e-7)  # -1.78e-5 vs standard
    assert res["Hcg"]   == pytest.approx(871.443916,   rel=1e-7)
    assert res["u_Hcg"] == pytest.approx(0.522493911,  rel=1e-8)
    assert res["Hmg"]   == pytest.approx(51.294085,    rel=1e-6)
    assert res["u_Hmg"] == pytest.approx(0.025938,     rel=1e-4)
    assert res["Hvg"]   == pytest.approx(36.874304,    rel=1e-4)
    assert res["u_Hvg"] == pytest.approx(0.022289,     rel=1e-4)


# ---------------------------------------------------------------------------
# Annex D Example 3 — identity correlation matrix, 15/15 °C
# ---------------------------------------------------------------------------


def test_example3_identity_15_15():
    x, u, r = _load("example3")
    res = calculate_properties(
        x, u, r,
        combustion_temperature=15, volume_temperature=15,
    )

    assert res["Hvg"]  == pytest.approx(39.73351, abs=1e-5)
    assert res["u_Hvg"] == pytest.approx(0.026917, abs=1e-6)
    assert res["Hvn"]  == pytest.approx(35.86811, abs=1e-5)
    assert res["u_Hvn"] == pytest.approx(0.024757, abs=1e-6)
    assert res["D"]    == pytest.approx(0.76462,  abs=1e-5)
    assert res["u_D"]  == pytest.approx(0.000586, abs=1e-6)
    assert res["G"]    == pytest.approx(0.62391,  abs=1e-5)
    assert res["u_G"]  == pytest.approx(0.000478, abs=1e-6)
    assert res["Wg"]   == pytest.approx(50.30318, abs=1e-5)
    assert res["u_Wg"] == pytest.approx(0.021588, abs=1e-6)
    assert res["Wn"]   == pytest.approx(45.40954, abs=1e-5)
    assert res["u_Wn"] == pytest.approx(0.020151, abs=1e-6)


def test_example3_identity_25_0():
    x, u, r = _load("example3")
    res = calculate_properties(
        x, u, r,
        combustion_temperature=25, volume_temperature=0,
    )

    assert res["Hvg"]  == pytest.approx(41.89360, abs=1e-5)
    assert res["u_Hvg"] == pytest.approx(0.028425, abs=1e-6)
    assert res["Hvn"]  == pytest.approx(37.85228, abs=1e-5)
    assert res["u_Hvn"] == pytest.approx(0.026164, abs=1e-6)
    assert res["D"]    == pytest.approx(0.80701,  abs=1e-5)
    assert res["u_D"]  == pytest.approx(0.000619, abs=1e-6)
    assert res["G"]    == pytest.approx(0.62411,  abs=1e-5)
    assert res["u_G"]  == pytest.approx(0.000479, abs=1e-6)
    assert res["Wg"]   == pytest.approx(53.02930, abs=1e-5)
    assert res["u_Wg"] == pytest.approx(0.022783, abs=1e-6)
    assert res["Wn"]   == pytest.approx(47.91376, abs=1e-5)
    assert res["u_Wn"] == pytest.approx(0.021278, abs=1e-6)


# ---------------------------------------------------------------------------
# Annex D Example 3 — full correlation matrix (example3_ex).
# Expected u values below are the standard uncertainties u(Y) taken verbatim
# from ISO 6976:2016 Annex D, clause D.4.3.2; they match the software exactly.
# ---------------------------------------------------------------------------


def test_example3_ex_full_15_15():
    x, u, r = _load("example3_ex")
    res = calculate_properties(
        x, u, r,
        combustion_temperature=15, volume_temperature=15,
    )

    assert res["Hvg"]  == pytest.approx(39.73351, abs=1e-5)
    assert res["u_Hvg"] == pytest.approx(0.016316, abs=1e-6)
    assert res["Hvn"]  == pytest.approx(35.86811, abs=1e-5)
    assert res["u_Hvn"] == pytest.approx(0.015305, abs=1e-6)
    assert res["D"]    == pytest.approx(0.76462,  abs=1e-5)
    assert res["u_D"]  == pytest.approx(0.000277, abs=1e-6)
    assert res["G"]    == pytest.approx(0.62391,  abs=1e-5)
    assert res["u_G"]  == pytest.approx(0.000226, abs=1e-6)
    assert res["Wg"]   == pytest.approx(50.30318, abs=1e-5)
    assert res["u_Wg"] == pytest.approx(0.019823, abs=1e-6)
    assert res["Wn"]   == pytest.approx(45.40954, abs=1e-5)
    assert res["u_Wn"] == pytest.approx(0.018498, abs=1e-6)


def test_example3_ex_full_25_0():
    x, u, r = _load("example3_ex")
    res = calculate_properties(
        x, u, r,
        combustion_temperature=25, volume_temperature=0,
    )

    assert res["Hvg"]  == pytest.approx(41.89360, abs=1e-5)
    assert res["u_Hvg"] == pytest.approx(0.017241, abs=1e-6)
    assert res["Hvn"]  == pytest.approx(37.85228, abs=1e-5)
    assert res["u_Hvn"] == pytest.approx(0.016181, abs=1e-6)
    assert res["D"]    == pytest.approx(0.80701,  abs=1e-5)
    assert res["u_D"]  == pytest.approx(0.000293, abs=1e-6)
    assert res["G"]    == pytest.approx(0.62411,  abs=1e-5)
    assert res["u_G"]  == pytest.approx(0.000227, abs=1e-6)
    assert res["Wg"]   == pytest.approx(53.02930, abs=1e-5)
    assert res["u_Wg"] == pytest.approx(0.020914, abs=1e-6)
    assert res["Wn"]   == pytest.approx(47.91376, abs=1e-5)
    assert res["u_Wn"] == pytest.approx(0.019528, abs=1e-6)


# ---------------------------------------------------------------------------
# Internal-consistency checks
# ---------------------------------------------------------------------------


def test_example3_ideal_gas_relationships():
    x, u, r = _load("example3")
    res = calculate_properties(
        x, u, r,
        combustion_temperature=15, volume_temperature=15,
    )

    M_air = 28.96546
    assert res["G_o"]   == pytest.approx(res["M"] / M_air,           abs=1e-9)
    assert res["D_o"]   == pytest.approx(res["D"]   * res["Z"],      abs=1e-6)
    assert res["Hvg_o"] == pytest.approx(res["Hvg"] * res["Z"],      abs=1e-6)
    assert res["Hvn_o"] == pytest.approx(res["Hvn"] * res["Z"],      abs=1e-6)
    assert res["Wg_o"]  == pytest.approx(res["Hvg_o"] / np.sqrt(res["G_o"]), abs=1e-6)
    assert res["Wn_o"]  == pytest.approx(res["Hvn_o"] / np.sqrt(res["G_o"]), abs=1e-6)


def test_example1_ideal_gas_uncertainty_identity():
    x, u, r = _load("example1")
    res = calculate_properties(
        x, u, r,
        combustion_temperature=15, volume_temperature=15,
    )

    assert res["u_Hvg_o"] == pytest.approx(res["u_Hvg"] * res["Z"], abs=1e-12)
    assert res["u_Hvn_o"] == pytest.approx(res["u_Hvn"] * res["Z"], abs=1e-12)
    assert res["u_Hvg_o"] > 0
    assert res["u_Hvn_o"] > 0


def test_example1_ncv_sanity():
    x, u, r = _load("example1")
    res = calculate_properties(
        x, u, r,
        combustion_temperature=15, volume_temperature=15,
    )

    for key in ("Hcn", "Hmn", "u_Hcn", "u_Hmn"):
        assert np.isfinite(res[key]) and res[key] > 0, key
    assert res["Hcn"] < res["Hcg"]
    assert res["Hmn"] < res["Hmg"]


# ---------------------------------------------------------------------------
# Component name helpers
# ---------------------------------------------------------------------------


def test_component_names_length():
    nm = component_names()
    assert isinstance(nm, tuple)
    assert len(nm) == 60


def test_component_index_boundary_components():
    # 0-based: methane=0, ethane=1, nitrogen=51, carbon dioxide=53, n-pentadecane=59
    assert component_index("methane")        == 0
    assert component_index("ethane")         == 1
    assert component_index("nitrogen")       == 51
    assert component_index("carbon dioxide") == 53
    assert component_index("n-pentadecane")  == 59


def test_component_name_boundary_indices():
    assert component_name(0)  == "methane"
    assert component_name(51) == "nitrogen"
    assert component_name(53) == "carbon dioxide"
    assert component_name(59) == "n-pentadecane"


def test_component_index_name_roundtrip():
    for i in (0, 9, 29, 53, 59):
        assert component_index(component_name(i)) == i


def test_component_index_errors_on_unknown_name():
    with pytest.raises(ValueError, match="Unknown component"):
        component_index("unobtainium")


def test_component_name_errors_on_out_of_range():
    with pytest.raises(IndexError):
        component_name(-1)
    with pytest.raises(IndexError):
        component_name(60)


# ---------------------------------------------------------------------------
# calculate_properties — input validation
# ---------------------------------------------------------------------------


def test_calc_rejects_wrong_length_composition():
    with pytest.raises(ValueError, match="length 60"):
        calculate_properties(np.zeros(59), np.zeros(60), np.eye(60))


def test_calc_rejects_wrong_length_uncertainty():
    with pytest.raises(ValueError, match="length 60"):
        calculate_properties(np.zeros(60), np.zeros(61), np.eye(60))


def test_calc_rejects_wrong_size_correlation():
    with pytest.raises(ValueError, match="60x60"):
        calculate_properties(np.zeros(60), np.zeros(60), np.zeros((59, 60)))


def test_calc_rejects_invalid_combustion_temperature():
    with pytest.raises(ValueError, match="combustion_temperature"):
        calculate_properties(
            np.zeros(60), np.zeros(60), np.eye(60),
            combustion_temperature=30,
        )


def test_calc_rejects_invalid_volume_temperature():
    with pytest.raises(ValueError, match="volume_temperature"):
        calculate_properties(
            np.zeros(60), np.zeros(60), np.eye(60),
            volume_temperature=10,
        )


def test_calc_rejects_out_of_range_pressure():
    for bad in (50.0, 200.0):
        with pytest.raises(ValueError, match="pressure"):
            calculate_properties(
                np.zeros(60), np.zeros(60), np.eye(60), pressure=bad,
            )


# ---------------------------------------------------------------------------
# Shared key lists used by the dict-based tests below
# ---------------------------------------------------------------------------


_U_KEYS = (
    "u_G", "u_D", "u_Hcg", "u_Hcn", "u_Hmg", "u_Hmn",
    "u_Hvg_o", "u_Hvn_o", "u_Hvg", "u_Hvn", "u_Wg", "u_Wn",
)
_DET_KEYS = (
    "M", "Z", "G_o", "D_o", "G", "D",
    "Hcg", "Hcn", "Hmg", "Hmn",
    "Hvg_o", "Hvn_o", "Hvg", "Hvn",
    "Wg_o", "Wn_o", "Wg", "Wn",
)


# ---------------------------------------------------------------------------
# Dict-based API
# ---------------------------------------------------------------------------


def test_dict_composition_matches_array_composition():
    x, u, r = _load("example1")
    comp_dict = {component_name(i): float(x[i]) for i in range(60) if x[i] != 0.0}
    unc_dict = {component_name(i): float(u[i]) for i in range(60) if u[i] != 0.0}

    res_array = calculate_properties(
        x, u, r,
        combustion_temperature=15, volume_temperature=15,
    )
    res_dict = calculate_properties(
        comp_dict, unc_dict, r,
        combustion_temperature=15, volume_temperature=15,
    )
    for key in _DET_KEYS + _U_KEYS:
        assert res_dict[key] == pytest.approx(res_array[key], abs=1e-12), key


def test_dict_composition_only_methane_nitrogen():
    res = calculate_properties(
        {"methane": 0.95, "nitrogen": 0.05},
        combustion_temperature=15, volume_temperature=15,
    )
    # Deterministic values must be finite and physically sane.
    assert res["M"] > 0
    assert 0.9 < res["Z"] < 1.0
    assert res["Hcg"] > 0
    assert res["Hcn"] > 0
    assert res["Hcn"] < res["Hcg"]
    # Without composition uncertainty, u_* is driven only by the residual
    # uncertainty of the tabulated physical constants — so still finite and
    # non-negative, but small.
    for key in _U_KEYS:
        assert np.isfinite(res[key]) and res[key] >= 0.0, key


def test_dict_uncertainty_default_matches_explicit_zero():
    """Omitting uncertainty must give the same result as passing zeros."""
    res_default = calculate_properties(
        {"methane": 0.9, "ethane": 0.1},
        combustion_temperature=15, volume_temperature=15,
    )
    res_zero = calculate_properties(
        {"methane": 0.9, "ethane": 0.1},
        uncertainty={},  # empty dict → all-zero
        combustion_temperature=15, volume_temperature=15,
    )
    res_zero_array = calculate_properties(
        {"methane": 0.9, "ethane": 0.1},
        uncertainty=np.zeros(60),
        combustion_temperature=15, volume_temperature=15,
    )
    for key in _DET_KEYS + _U_KEYS:
        assert res_default[key] == pytest.approx(res_zero[key], abs=1e-12), key
        assert res_default[key] == pytest.approx(res_zero_array[key], abs=1e-12), key


def test_dict_correlation_optional_defaults_to_identity():
    x, u, _ = _load("example1")
    comp_dict = {component_name(i): float(x[i]) for i in range(60) if x[i] != 0.0}
    unc_dict = {component_name(i): float(u[i]) for i in range(60) if u[i] != 0.0}

    res_default = calculate_properties(
        comp_dict, unc_dict,
        combustion_temperature=15, volume_temperature=15,
    )
    res_identity = calculate_properties(
        comp_dict, unc_dict, np.eye(60),
        combustion_temperature=15, volume_temperature=15,
    )
    for key in _DET_KEYS + _U_KEYS:
        assert res_default[key] == pytest.approx(res_identity[key], abs=1e-12), key


def test_dict_correlation_as_dict():
    comp = {"methane": 0.9, "ethane": 0.1}
    unc = {"methane": 0.001, "ethane": 0.001}

    res_uncorr = calculate_properties(
        comp, unc,
        combustion_temperature=15, volume_temperature=15,
    )
    res_corr = calculate_properties(
        comp, unc, {("methane", "ethane"): 0.5},
        combustion_temperature=15, volume_temperature=15,
    )
    # Deterministic values unchanged; uncertainties shift with correlation.
    for key in _DET_KEYS:
        assert res_corr[key] == pytest.approx(res_uncorr[key], abs=1e-12), key
    assert res_corr["u_Hcg"] != pytest.approx(res_uncorr["u_Hcg"], abs=1e-10)


def test_dict_composition_rejects_unknown_component():
    with pytest.raises(ValueError, match="Unknown component"):
        calculate_properties({"methane": 0.9, "unobtainium": 0.1})


def test_dict_uncertainty_rejects_unknown_component():
    with pytest.raises(ValueError, match="Unknown component"):
        calculate_properties(
            {"methane": 1.0},
            uncertainty={"unobtainium": 0.01},
        )


def test_dict_correlation_rejects_unknown_component():
    with pytest.raises(ValueError, match="Unknown component"):
        calculate_properties(
            {"methane": 0.9, "ethane": 0.1},
            correlation={("methane", "unobtainium"): 0.3},
        )


def test_dict_correlation_rejects_out_of_range_value():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        calculate_properties(
            {"methane": 0.9, "ethane": 0.1},
            correlation={("methane", "ethane"): 1.5},
        )
