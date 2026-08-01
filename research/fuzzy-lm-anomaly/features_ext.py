"""Stage 12 -- an expanded family of derived variables.

Everything here is computed from activations already on disk. The residual
stream was stored at all 33 layers for three pooling sites, and so far only one
view of it has been used (distance and cosine to the truthful centroid). These
are the other views, including the aggregate functions that summarise a whole
depth profile into a few numbers.

Families (all fit-split-only where a reference is needed):

  centroid      per-layer distance + cosine to the truthful centroid   (66)
  delta         per-layer UPDATE: ||h[L+1]-h[L]||, cos(h[L+1], h[L])   (64)
  deltaref      the update compared to the mean truthful update        (64)
  geom          per-layer norm and consecutive norm ratio              (65)
  curve         curvature of the depth profile (2nd differences)       (31)
  agg           AGGREGATE functions over the depth profile: slope,
                argmax, area, monotonicity, early/late contrast        (~24)
  stats         the 19 output-distribution statistics                  (19)

`agg` is the interesting one conceptually: rather than 33 separate per-layer
features it asks "what shape does this trajectory have", which is closer to how
one would describe the behaviour linguistically -- and a fuzzy rule over 6
shape descriptors is far more readable than one over 66 per-layer numbers.
"""

import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze import SCALAR_COLS


def _unit(A, eps=1e-8):
    return A / (np.linalg.norm(A, axis=-1, keepdims=True) + eps)


def f_centroid(meta, H, fit):
    cols = {}
    for li in range(H.shape[1]):
        A = H[:, li, :].astype(np.float32)
        c = A[fit].mean(0, keepdims=True)
        cols[f"L{li:02d}_dist"] = np.linalg.norm(A - c, axis=1)
        cols[f"L{li:02d}_cos"] = (_unit(A) @ _unit(c).T).ravel()
    return pd.DataFrame(cols)


def f_delta(meta, H, fit):
    """The update each layer applies, independent of where the state sits."""
    cols = {}
    for li in range(H.shape[1] - 1):
        a = H[:, li, :].astype(np.float32)
        b = H[:, li + 1, :].astype(np.float32)
        d = b - a
        cols[f"D{li:02d}_norm"] = np.linalg.norm(d, axis=1)
        cols[f"D{li:02d}_cos"] = (_unit(a) * _unit(b)).sum(1)
    return pd.DataFrame(cols)


def f_deltaref(meta, H, fit):
    """How unusual is this layer's update, versus the mean truthful update."""
    cols = {}
    for li in range(H.shape[1] - 1):
        d = (H[:, li + 1, :].astype(np.float32) - H[:, li, :].astype(np.float32))
        r = d[fit].mean(0, keepdims=True)
        cols[f"R{li:02d}_dist"] = np.linalg.norm(d - r, axis=1)
        cols[f"R{li:02d}_cos"] = (_unit(d) @ _unit(r).T).ravel()
    return pd.DataFrame(cols)


def f_geom(meta, H, fit):
    cols = {}
    prev = None
    for li in range(H.shape[1]):
        n = np.linalg.norm(H[:, li, :].astype(np.float32), axis=1)
        cols[f"N{li:02d}_norm"] = n
        if prev is not None:
            cols[f"N{li:02d}_ratio"] = n / (prev + 1e-8)
        prev = n
    return pd.DataFrame(cols)


def f_curve(meta, H, fit):
    """Second difference of the per-layer distance profile."""
    d = np.stack([np.linalg.norm(H[:, li, :].astype(np.float32)
                                 - H[fit][:, li, :].astype(np.float32).mean(0),
                                 axis=1)
                  for li in range(H.shape[1])], axis=1)
    c = d[:, 2:] - 2 * d[:, 1:-1] + d[:, :-2]
    return pd.DataFrame(c, columns=[f"C{i:02d}_curv" for i in range(c.shape[1])])


def f_agg(meta, H, fit):
    """Aggregate SHAPE descriptors of the depth profile.

    Collapses 33 per-layer numbers into a handful of interpretable summaries --
    the kind of variable a fuzzy rule can talk about ("the distance profile
    rises late and peaks near the output").
    """
    L = H.shape[1]
    dist = np.stack([np.linalg.norm(H[:, li, :].astype(np.float32)
                                    - H[fit][:, li, :].astype(np.float32).mean(0),
                                    axis=1) for li in range(L)], axis=1)
    cos = np.stack([(_unit(H[:, li, :].astype(np.float32))
                     @ _unit(H[fit][:, li, :].astype(np.float32).mean(0)[None]).T
                     ).ravel() for li in range(L)], axis=1)
    nrm = np.stack([np.linalg.norm(H[:, li, :].astype(np.float32), axis=1)
                    for li in range(L)], axis=1)

    x = np.arange(L, dtype=np.float32)
    xc = x - x.mean()

    def slope(P):
        return (P * xc).sum(1) / (xc ** 2).sum()

    out = {}
    for nm, P in (("dist", dist), ("cos", cos), ("norm", nrm)):
        Pn = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-8)
        out[f"agg_{nm}_mean"] = P.mean(1)
        out[f"agg_{nm}_std"] = P.std(1)
        out[f"agg_{nm}_slope"] = slope(P)
        out[f"agg_{nm}_argmax"] = P.argmax(1).astype(np.float32) / L
        out[f"agg_{nm}_max"] = P.max(1)
        out[f"agg_{nm}_range"] = P.max(1) - P.min(1)
        out[f"agg_{nm}_early"] = P[:, :L // 3].mean(1)
        out[f"agg_{nm}_late"] = P[:, 2 * L // 3:].mean(1)
        out[f"agg_{nm}_contrast"] = out[f"agg_{nm}_late"] - out[f"agg_{nm}_early"]
        # monotonicity: fraction of consecutive steps that increase
        out[f"agg_{nm}_mono"] = (np.diff(P, axis=1) > 0).mean(1)
        # roughness: mean |2nd difference| of the standardised profile
        out[f"agg_{nm}_rough"] = np.abs(np.diff(Pn, 2, axis=1)).mean(1)
    return pd.DataFrame(out)


def f_stats(meta, H, fit):
    return meta[SCALAR_COLS].reset_index(drop=True).astype(float)


FAMILIES = {
    "centroid": f_centroid, "delta": f_delta, "deltaref": f_deltaref,
    "geom": f_geom, "curve": f_curve, "agg": f_agg, "stats": f_stats,
}


def build(meta, H, fit, families=None):
    """Return {family: DataFrame}, constant columns dropped on the fit split."""
    out = {}
    for name in (families or FAMILIES):
        F = FAMILIES[name](meta, H, fit)
        F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        v = F.iloc[fit].var()
        out[name] = F[list(v[v > 1e-8].index)]
    return out


if __name__ == "__main__":
    from analyze import DATA, make_splits
    from nopca import POOLING
    meta = pd.read_parquet(DATA / "capture_v2_meta.parquet")
    H = np.load(DATA / f"capture_v2_hidden_{POOLING}.npy")
    fit = np.flatnonzero((meta.family == "triviaqa")
                         & (meta.label == "correct"))[:2000]
    fam = build(meta, H, fit)
    print(f"{'family':<12} {'features':>9}")
    for k, v in fam.items():
        print(f"{k:<12} {v.shape[1]:>9}")
    print(f"{'TOTAL':<12} {sum(v.shape[1] for v in fam.values()):>9}")
