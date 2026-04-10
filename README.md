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

Pass a dictionary with just the components you actually have — everything
else is assumed to be zero:

```python
from iso6976 import calculate_properties

res = calculate_properties(
    composition={
        "methane":        0.90,
        "ethane":         0.05,
        "nitrogen":       0.03,
        "carbon dioxide": 0.02,
    },
    combustion_temperature=15,
    volume_temperature=15,
)

print(f"M   = {res['M']:.4f} kg/kmol")
print(f"Hvg = {res['Hvg']:.4f} ± {res['u_Hvg']:.4f} MJ/m³")
print(f"Wg  = {res['Wg']:.4f} ± {res['u_Wg']:.4f} MJ/m³")
```

### With composition uncertainty

Pass a second dictionary with the standard uncertainties of the mole
fractions. Components not mentioned are assumed to have zero composition
uncertainty (the result still carries the residual uncertainty of the
tabulated physical constants from ISO 6976):

```python
res = calculate_properties(
    composition={"methane": 0.9, "ethane": 0.1},
    uncertainty={"methane": 0.001, "ethane": 0.001},
    combustion_temperature=15,
    volume_temperature=15,
)
```

### With correlations between components

Pass a dictionary keyed by `(name1, name2)` tuples. All unmentioned
off-diagonal entries are assumed zero (components uncorrelated):

```python
res = calculate_properties(
    composition={"methane": 0.9, "ethane": 0.1},
    uncertainty={"methane": 0.001, "ethane": 0.001},
    correlation={("methane", "ethane"): 0.3},
    combustion_temperature=15,
    volume_temperature=15,
)
```

### Array-based input (advanced)

For bulk workflows you can still pass length-60 numpy arrays (and a 60x60
correlation matrix) in the order of ISO 6976:2016 Table A.2. Use
`component_index(name)` / `component_name(index)` / `component_names()` to
navigate the ordering.

## Result keys

The returned dictionary contains numeric scalars under the following keys:

| Key                | Description                              | Unit    |
| ------------------ | ---------------------------------------- | ------- |
| `M`                | Molar mass                               | kg/kmol |
| `Z`                | Compression factor                       | —       |
| `G_o`              | Ideal-gas relative density               | —       |
| `D_o`              | Ideal-gas density                        | kg/m³   |
| `G`,    `u_G`      | Real-gas relative density                | —       |
| `D`,    `u_D`      | Real-gas density                         | kg/m³   |
| `Hcg`,  `u_Hcg`    | Molar gross calorific value              | kJ/mol  |
| `Hcn`,  `u_Hcn`    | Molar net calorific value                | kJ/mol  |
| `Hmg`,  `u_Hmg`    | Mass-basis gross calorific value         | MJ/kg   |
| `Hmn`,  `u_Hmn`    | Mass-basis net calorific value           | MJ/kg   |
| `Hvg_o`,`u_Hvg_o`  | Ideal-gas volumetric gross CV            | MJ/m³   |
| `Hvn_o`,`u_Hvn_o`  | Ideal-gas volumetric net CV              | MJ/m³   |
| `Hvg`,  `u_Hvg`    | Real-gas volumetric gross CV             | MJ/m³   |
| `Hvn`,  `u_Hvn`    | Real-gas volumetric net CV               | MJ/m³   |
| `Wg_o`             | Ideal-gas gross Wobbe index              | MJ/m³   |
| `Wn_o`             | Ideal-gas net Wobbe index                | MJ/m³   |
| `Wg`,   `u_Wg`     | Real-gas gross Wobbe index               | MJ/m³   |
| `Wn`,   `u_Wn`     | Real-gas net Wobbe index                 | MJ/m³   |

Uncertainties are standard uncertainties (k = 1) by default. Pass
`coverage=2` to `calculate_properties` for expanded (k = 2) uncertainties.

## Credits

This package is a clean-room Python port of the R package
[`ISO6976.2016`](https://github.com/RuedigerForster/ISO_6976) by Rüdiger
Forster (GPL-3). The ISO tables, formulae and worked examples come from
ISO 6976:2016.

## License

GPL-3.0-or-later.
