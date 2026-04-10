"""Gas component names and the :class:`GasComponents` helper class."""

from __future__ import annotations

import numpy as np

from ._tables import N_COMPONENTS

# ISO 6976:2016 Table A.2 component names (English).
# Order matches the rows of ``_tables.TAB_M``.
_COMPONENT_NAMES: tuple[str, ...] = (
    "methane",              #  1
    "ethane",               #  2
    "propane",              #  3
    "n-butane",             #  4
    "isobutane",            #  5
    "n-pentane",            #  6
    "isopentane",           #  7
    "neopentane",           #  8
    "n-hexane",             #  9
    "2-methylpentane",      # 10
    "3-methylpentane",      # 11
    "2,2-dimethylbutane",   # 12
    "2,3-dimethylbutane",   # 13
    "n-heptane",            # 14
    "n-octane",             # 15
    "n-nonane",             # 16
    "n-decane",             # 17
    "ethylene",             # 18
    "propylene",            # 19
    "1-butene",             # 20
    "cis-2-butene",         # 21
    "trans-2-butene",       # 22
    "isobutylene",          # 23
    "1-pentene",            # 24
    "propadiene",           # 25
    "1,2-butadiene",        # 26
    "1,3-butadiene",        # 27
    "acetylene",            # 28
    "cyclopentane",         # 29
    "methylcyclopentane",   # 30
    "ethylcyclopentane",    # 31
    "cyclohexane",          # 32
    "methylcyclohexane",    # 33
    "ethylcyclohexane",     # 34
    "benzene",              # 35
    "toluene",              # 36
    "ethylbenzene",         # 37
    "o-xylene",             # 38
    "methanol",             # 39
    "methanethiol",         # 40
    "hydrogen",             # 41
    "water",                # 42
    "hydrogen sulphide",    # 43
    "ammonia",              # 44
    "hydrogen cyanide",     # 45
    "carbon monoxide",      # 46
    "carbonyl sulphide",    # 47
    "carbon disulphide",    # 48
    "helium",               # 49
    "neon",                 # 50
    "argon",                # 51
    "nitrogen",             # 52
    "oxygen",               # 53
    "carbon dioxide",       # 54
    "sulphur dioxide",      # 55
    "n-undecane",           # 56
    "n-dodecane",           # 57
    "n-tridecane",          # 58
    "n-tetradecane",        # 59
    "n-pentadecane",        # 60
)

assert len(_COMPONENT_NAMES) == N_COMPONENTS

_NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(_COMPONENT_NAMES)}


def component_names() -> tuple[str, ...]:
    """Return the ordered tuple of all 60 ISO 6976:2016 component names."""
    return _COMPONENT_NAMES


def component_index(name: str) -> int:
    """Return the 0-based index of ``name`` in the composition vector.

    Note: R uses 1-based indexing; this Python port uses 0-based indexing
    throughout (``methane`` = 0, ``n-pentadecane`` = 59).
    """
    try:
        return _NAME_TO_INDEX[name]
    except KeyError as exc:
        raise ValueError(f"Unknown component: {name}") from exc


def component_name(index: int) -> str:
    """Return the English name of the component at the given 0-based index."""
    if not 0 <= index < N_COMPONENTS:
        raise IndexError(f"index must be between 0 and {N_COMPONENTS - 1}")
    return _COMPONENT_NAMES[index]


# ---------------------------------------------------------------------------
# GasComponents helper
# ---------------------------------------------------------------------------


class GasComponents:
    """Holder for mole fractions, standard uncertainties and correlations.

    All three quantities are required by :func:`iso6976.calculate_properties`.
    Components are identified either by 0-based integer index (0–59) or by
    English name (see :func:`component_names`).
    """

    def __init__(self) -> None:
        self.fractions: np.ndarray = np.zeros(N_COMPONENTS, dtype=np.float64)
        self.uncertainties: np.ndarray = np.zeros(N_COMPONENTS, dtype=np.float64)
        self.correlations: np.ndarray = np.eye(N_COMPONENTS, dtype=np.float64)

    # -- internal ---------------------------------------------------------

    @staticmethod
    def _resolve(key: int | str) -> int:
        if isinstance(key, (int, np.integer)):
            idx = int(key)
            if not 0 <= idx < N_COMPONENTS:
                raise IndexError(
                    f"index must be between 0 and {N_COMPONENTS - 1}"
                )
            return idx
        return component_index(key)

    # -- single-component getters/setters --------------------------------

    def get_fraction(self, key: int | str) -> float:
        return float(self.fractions[self._resolve(key)])

    def set_fraction(self, key: int | str, value: float) -> None:
        self.fractions[self._resolve(key)] = value

    def get_uncertainty(self, key: int | str) -> float:
        return float(self.uncertainties[self._resolve(key)])

    def set_uncertainty(self, key: int | str, value: float) -> None:
        self.uncertainties[self._resolve(key)] = value

    def get_correlation(self, key1: int | str, key2: int | str) -> float:
        return float(self.correlations[self._resolve(key1), self._resolve(key2)])

    def set_correlation(
        self, key1: int | str, key2: int | str, value: float
    ) -> None:
        i = self._resolve(key1)
        j = self._resolve(key2)
        self.correlations[i, j] = value
        self.correlations[j, i] = value

    # -- bulk setters -----------------------------------------------------

    def set_fraction_array(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (N_COMPONENTS,):
            raise ValueError(f"x must have length {N_COMPONENTS}")
        self.fractions = x.copy()

    def set_uncertainty_array(self, u: np.ndarray) -> None:
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (N_COMPONENTS,):
            raise ValueError(f"u must have length {N_COMPONENTS}")
        self.uncertainties = u.copy()

    def set_correlation_matrix(self, r: np.ndarray) -> None:
        r = np.asarray(r, dtype=np.float64)
        if r.shape != (N_COMPONENTS, N_COMPONENTS):
            raise ValueError(
                f"r must be a {N_COMPONENTS}x{N_COMPONENTS} matrix"
            )
        if np.any(r < -1.0) or np.any(r > 1.0):
            raise ValueError("All correlation coefficients must be in [-1, 1]")
        self.correlations = r.copy()
