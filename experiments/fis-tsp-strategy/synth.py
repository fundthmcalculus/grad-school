"""Synthetic TSP instances, to break the sample-size limit on fitting.

Fitting has been limited to nine TSPLIB training instances with n >= 1000, against 170
parameters, and overfitting has been the dominant failure mode throughout — more search
reliably produced a worse answer. The obvious fix is more instances, and TSPLIB does not
have more: there are only sixteen at that size that are not in the test set.

What makes synthetic instances usable here is that **the objective needs no optimum.** The
reported metric is

    q = (this arm's tour length) / (the tour length the swept baseline reaches at the
                                    same cost on the same instance)

which is a ratio of two tours on one instance, so whatever reference would have cancelled
is not needed at all. Published optima are required only to state a *gap*, which is a
reporting convenience on the test set, not something fitting depends on. So the training
pool can be as large as patience allows.

The generators cover the structures that TSPLIB instances actually exhibit, because the
families behave differently and a rule base fitted on one transfers poorly to another:

* ``uniform`` — points uniform in a square. The easy case, and the one most synthetic
  benchmarks stop at.
* ``clustered`` — Gaussian blobs. Candidate lists inside a blob are useless (every
  neighbour is close), and the interesting edges are the few between blobs. This is what
  makes the fl* and vm* instances hard.
* ``grid`` — jittered lattice. Massively tie-heavy, which stresses candidate ordering and
  is where the rl* instances live.
* ``mixed`` — half a dense blob field, half sparse uniform, so a single instance contains
  both regimes and a rule base cannot succeed by assuming one global scale. This is the
  case an effort-allocation rule base ought to win on, and no TSPLIB instance isolates it.

Coordinates are scaled to a spread comparable to TSPLIB's so the fuzzy features — all
scale-free ratios — see the same numeric ranges they do on real instances.

Run:  python synth.py --list        # show the standard pool
"""

from __future__ import annotations

import argparse

import numpy as np

from tsplib import Instance

# The standard pools. Sizes stay in the n >= 1000 regime this engine is aimed at, and the
# seeds are fixed so that "the training set" is a reproducible object rather than whatever
# the RNG produced that afternoon. Train and validation draw on *disjoint seeds and
# different sizes*, so a validation instance is never a near-duplicate of a training one.
TRAIN_SPEC = [
    ("uniform", 1200, 11),
    ("uniform", 2600, 12),
    ("uniform", 5200, 13),
    ("clustered", 1400, 21),
    ("clustered", 3000, 22),
    ("clustered", 6000, 23),
    ("grid", 1600, 31),
    ("grid", 3400, 32),
    ("mixed", 1800, 41),
    ("mixed", 3800, 42),
    ("mixed", 7000, 43),
]
VALID_SPEC = [
    ("uniform", 1500, 111),
    ("uniform", 4000, 112),
    ("clustered", 2000, 121),
    ("clustered", 4600, 122),
    ("grid", 2200, 131),
    ("mixed", 2800, 141),
    ("mixed", 5600, 142),
]

SPREAD = 10000.0  # coordinate range, chosen to match TSPLIB's typical magnitudes


def _uniform(n, rng):
    return rng.uniform(0.0, SPREAD, size=(n, 2))


def _clustered(n, rng, n_clusters=None):
    """Gaussian blobs. Cluster count grows as sqrt(n) so blob population stays comparable
    across sizes rather than blobs becoming denser with n."""
    if n_clusters is None:
        n_clusters = max(4, int(np.sqrt(n) / 3))
    centres = rng.uniform(0.0, SPREAD, size=(n_clusters, 2))
    sigma = SPREAD / (6.0 * np.sqrt(n_clusters))
    which = rng.integers(0, n_clusters, size=n)
    return centres[which] + rng.normal(0.0, sigma, size=(n, 2))


def _grid(n, rng):
    """Jittered lattice. The jitter is a small fraction of the cell so that many pairwise
    distances remain exactly equal after TSPLIB's integer rounding — which is the property
    that stresses candidate-list tie handling."""
    side = int(np.ceil(np.sqrt(n)))
    step = SPREAD / side
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)[:n] * step
    return pts + rng.uniform(-0.05 * step, 0.05 * step, size=pts.shape)


def _mixed(n, rng):
    """Half dense blobs, half sparse uniform, in disjoint halves of the square.

    The point of this family is that no single global effort setting is right for both
    halves, so it separates a rule base that genuinely reads local structure from one that
    has merely learned a good constant.
    """
    n_dense = n // 2
    dense = _clustered(n_dense, rng, n_clusters=max(3, int(np.sqrt(n_dense) / 4)))
    dense[:, 0] = dense[:, 0] * 0.45  # squeeze into the left 45%
    sparse = rng.uniform(0.0, SPREAD, size=(n - n_dense, 2))
    sparse[:, 0] = SPREAD * 0.55 + sparse[:, 0] * 0.45  # right 45%
    return np.vstack([dense, sparse])


GENERATORS = {
    "uniform": _uniform,
    "clustered": _clustered,
    "grid": _grid,
    "mixed": _mixed,
}


def make(kind, n, seed):
    """One synthetic instance, as the same ``Instance`` the TSPLIB loader returns.

    ``opt`` is None: nothing here knows the optimal tour, and the frontier-relative metric
    does not need it. ``Instance.gap`` already returns NaN without an optimum, so anything
    that tries to state a gap on one of these produces a visible NaN rather than a silently
    wrong number.
    """
    if kind not in GENERATORS:
        raise ValueError(f"unknown family {kind!r}; have {sorted(GENERATORS)}")
    rng = np.random.default_rng(seed)
    coords = np.ascontiguousarray(GENERATORS[kind](int(n), rng), dtype=np.float64)
    return Instance(name=f"{kind}{n}s{seed}", coords=coords, ewt="EUC_2D", opt=None)


def pool(spec):
    return [make(kind, n, seed) for kind, n, seed in spec]


def train_pool():
    return pool(TRAIN_SPEC)


def valid_pool():
    return pool(VALID_SPEC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or True:
        for label, spec in (("train", TRAIN_SPEC), ("valid", VALID_SPEC)):
            print(f"{label}: {len(spec)} instances")
            for inst in pool(spec):
                lo = inst.coords.min(axis=0)
                hi = inst.coords.max(axis=0)
                print(
                    f"  {inst.name:>18s} n={inst.n:6d} "
                    f"extent {hi[0] - lo[0]:8.0f} x {hi[1] - lo[1]:8.0f}"
                )


if __name__ == "__main__":
    main()
