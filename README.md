# iso6976

Python implementation of **ISO 6976:2016** — *Natural Gas — Calculation of calorific values, density, relative density and Wobbe indices from composition*.

Calculates gross/net calorific values (molar, mass and volumetric), density, relative density, and Wobbe indices from a natural gas composition, together with their standard uncertainties following Annex B of the standard.

## Status

Alpha — port in progress. Validated against the four Annex D worked examples
of ISO 6976:2016.

## Installation

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from iso6976 import calculate_properties, component_index

x = np.zeros(60)
x[component_index("methane")]        = 0.90
x[component_index("ethane")]         = 0.05
x[component_index("nitrogen")]       = 0.03
x[component_index("carbon dioxide")] = 0.02

u_x = np.zeros(60)
r_x = np.eye(60)

res = calculate_properties(
    x, u_x, r_x,
    combustion_temperature=15,
    volume_temperature=15,
)

print(f"M   = {res['M']:.4f} kg/kmol")
print(f"Hvg = {res['Hvg']:.4f} ± {res['u_Hvg']:.4f} MJ/m³")
print(f"Wg  = {res['Wg']:.4f} ± {res['u_Wg']:.4f} MJ/m³")
```

## Component ordering

All composition, uncertainty and correlation arrays are ordered according to ISO 6976:2016 Table A.2 (methane = 0, ethane = 1, …, n-pentadecane = 59). Use `component_index(name)` / `component_name(index)` for lookups.

## Credits

This package is a clean-room Python port of the R package
[`ISO6976.2016`](https://github.com/RuedigerForster/ISO_6976) by Rüdiger
Forster (GPL-3). The ISO tables, formulae and worked examples come from
ISO 6976:2016.

## License

GPL-3.0-or-later.
