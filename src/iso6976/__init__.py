"""iso6976 — Python port of the R package ``ISO6976.2016``.

Implements ISO 6976:2016 *Natural Gas — Calculation of calorific values,
density, relative density and Wobbe indices from composition*, including
the uncertainty propagation of its Annex B.

Basic usage::

    import numpy as np
    from iso6976 import calculate_properties, component_index

    x = np.zeros(60)
    x[component_index("methane")] = 1.0

    res = calculate_properties(
        x, np.zeros(60), np.eye(60),
        combustion_temperature=15, volume_temperature=15,
    )

The result dictionary contains (all numeric scalars):

================= ============================================ ==========
Key               Description                                  Unit
================= ============================================ ==========
``M``             Molar mass                                   kg/kmol
``Z``             Compression factor                           —
``G_o``           Ideal-gas relative density                   —
``D_o``           Ideal-gas density                            kg/m³
``G``, ``u_G``    Real-gas relative density + uncertainty      —
``D``, ``u_D``    Real-gas density + uncertainty               kg/m³
``Hcg``, ``u_Hcg``  Molar gross calorific value                kJ/mol
``Hcn``, ``u_Hcn``  Molar net calorific value                  kJ/mol
``Hmg``, ``u_Hmg``  Mass-basis gross calorific value           MJ/kg
``Hmn``, ``u_Hmn``  Mass-basis net calorific value             MJ/kg
``Hvg_o``, ``u_Hvg_o``  Ideal-gas vol. gross CV                MJ/m³
``Hvn_o``, ``u_Hvn_o``  Ideal-gas vol. net CV                  MJ/m³
``Hvg``, ``u_Hvg``  Real-gas vol. gross CV                     MJ/m³
``Hvn``, ``u_Hvn``  Real-gas vol. net CV                       MJ/m³
``Wg_o``          Ideal-gas gross Wobbe index                  MJ/m³
``Wn_o``          Ideal-gas net Wobbe index                    MJ/m³
``Wg``, ``u_Wg``  Real-gas gross Wobbe index + uncertainty     MJ/m³
``Wn``, ``u_Wn``  Real-gas net Wobbe index + uncertainty       MJ/m³
================= ============================================ ==========
"""

from __future__ import annotations

from .api import calculate_properties
from .components import (
    GasComponents,
    component_index,
    component_name,
    component_names,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "calculate_properties",
    "GasComponents",
    "component_index",
    "component_name",
    "component_names",
]
