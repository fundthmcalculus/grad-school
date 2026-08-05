"""TSPLIB instance loading, official distance rounding, and published optima.

Self-contained: reads the ``.tsp`` files and the ``solutions`` index that already
ship in ``ClusteringExperiments/tsplib/``, with no GPU or submodule dependency.
Only the coordinate instances are handled (EUC_2D and CEIL_2D) — those are the
ones whose distances are a plain rounding of the euclidean metric, so a solver
can work from coordinates alone and never materialise an O(n^2) matrix.

Usage:
    from tsplib import load, catalogue
    inst = load("pr1002")
    inst.coords   # (n, 2) float64
    inst.ceil     # True for CEIL_2D, False for EUC_2D nint
    inst.opt      # published optimal tour length (int) or None
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TSPLIB_DIR = Path(__file__).resolve().parent.parent / "ClusteringExperiments" / "tsplib"


@dataclass(frozen=True)
class Instance:
    """One TSPLIB coordinate instance plus its published optimum."""

    name: str
    coords: np.ndarray  # (n, 2) float64
    ewt: str  # "EUC_2D" | "CEIL_2D"
    opt: int | None  # published optimal tour length

    @property
    def n(self) -> int:
        return int(self.coords.shape[0])

    @property
    def ceil(self) -> bool:
        return self.ewt == "CEIL_2D"

    def gap(self, length: float) -> float:
        """Percent over the published optimum."""
        if not self.opt:
            return float("nan")
        return 100.0 * (length - self.opt) / self.opt


def _header(path: Path) -> tuple[int | None, str | None]:
    """(dimension, edge_weight_type) read from the header alone."""
    dim = ewt = None
    with open(path, errors="ignore") as fh:
        for line in fh:
            u = line.upper()
            if u.startswith("DIMENSION"):
                found = re.findall(r"\d+", line)
                if found:
                    dim = int(found[-1])
            elif u.startswith("EDGE_WEIGHT_TYPE"):
                ewt = line.split(":")[-1].strip()
            elif "SECTION" in u:
                break
    return dim, ewt


def parse_tsplib(name: str) -> tuple[np.ndarray, str]:
    """(coords (n,2) float64, edge_weight_type) for a coordinate instance."""
    path = TSPLIB_DIR / f"{name}.tsp"
    ewt = "EUC_2D"
    coords: list[tuple[float, float]] = []
    with open(path, errors="ignore") as fh:
        in_nodes = False
        for line in fh:
            u = line.strip().upper()
            if u.startswith("EDGE_WEIGHT_TYPE"):
                ewt = line.split(":")[-1].strip()
            elif u.startswith("NODE_COORD_SECTION"):
                in_nodes = True
            elif u.startswith("EOF"):
                break
            elif in_nodes:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    return np.ascontiguousarray(np.array(coords, dtype=np.float64)), ewt


_SOLUTIONS: dict[str, int] | None = None


def optimal_length(name: str) -> int | None:
    """Published optimal tour length, from the ``solutions`` index. None if the
    instance is not listed. Using the published value rather than running a
    reference solver keeps the comparison exact and instant."""
    global _SOLUTIONS
    if _SOLUTIONS is None:
        _SOLUTIONS = {}
        sol = TSPLIB_DIR / "solutions"
        if sol.exists():
            for line in open(sol, errors="ignore"):
                if ":" in line:
                    nm, val = line.split(":", 1)
                    m = re.search(r"\d+", val)  # first integer; ignore "(CEIL_2D)"
                    if m:
                        _SOLUTIONS[nm.strip()] = int(m.group())
    return _SOLUTIONS.get(name)


_CATALOGUE: dict[str, tuple[int, str]] | None = None


def catalogue(with_optimum: bool = True) -> dict[str, tuple[int, str]]:
    """{name: (dimension, ewt)} for every EUC_2D / CEIL_2D instance present.

    With ``with_optimum`` (the default) only instances that have a published
    optimum are returned — those are the ones we can score against.
    """
    global _CATALOGUE
    if _CATALOGUE is None:
        out: dict[str, tuple[int, str]] = {}
        for path in sorted(TSPLIB_DIR.glob("*.tsp")):
            dim, ewt = _header(path)
            if dim and ewt in ("EUC_2D", "CEIL_2D"):
                out[path.stem] = (dim, ewt)
        _CATALOGUE = out
    if with_optimum:
        return {k: v for k, v in _CATALOGUE.items() if optimal_length(k)}
    return dict(_CATALOGUE)


def load(name: str) -> Instance:
    coords, ewt = parse_tsplib(name)
    return Instance(name=name, coords=coords, ewt=ewt, opt=optimal_length(name))


def instances_in_range(lo: int, hi: int) -> list[str]:
    """Names of scorable coordinate instances with ``lo <= n <= hi``, by size."""
    cat = catalogue()
    names = [k for k, (d, _) in cat.items() if lo <= d <= hi]
    return sorted(names, key=lambda k: cat[k][0])


def reference_length(tour: np.ndarray, inst: Instance) -> float:
    """Official closed-tour length under TSPLIB rounding, computed in float64
    from the coordinates. Independent of the solver's own bookkeeping, so it is
    the number we report."""
    c = inst.coords
    a = tour
    b = np.roll(tour, -1)
    d = np.hypot(c[a, 0] - c[b, 0], c[a, 1] - c[b, 1])
    d = np.ceil(d) if inst.ceil else np.floor(d + 0.5)
    return float(d.sum())


def validate_tour(tour: np.ndarray, n: int) -> None:
    """Raise unless ``tour`` is a permutation of 0..n-1. Every reported tour is
    passed through this — a solver bug that drops or duplicates a city would
    otherwise look like a fantastic result."""
    if tour.shape[0] != n:
        raise ValueError(f"tour has {tour.shape[0]} cities, expected {n}")
    if not np.array_equal(np.sort(tour), np.arange(n)):
        raise ValueError("tour is not a permutation of the cities")


if __name__ == "__main__":
    cat = catalogue()
    print(f"{len(cat)} scorable coordinate instances in {TSPLIB_DIR}")
    for name, (dim, ewt) in sorted(cat.items(), key=lambda kv: kv[1][0]):
        print(f"  {name:>10s} {dim:7d}  {ewt:8s} opt={optimal_length(name)}")
