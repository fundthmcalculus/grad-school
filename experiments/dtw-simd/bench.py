"""Verify dtw_simd against aeon exactly, then benchmark both.

Run from experiments/dtw-simd:
    uv run --project ../../tribble-cluster --with aeon python bench.py
NOTE: absolute numbers here were measured WHILE a single-core DTW job was
running elsewhere on the box (the G2 harness); both sides see the same
contention, so the ratio stands.
"""

import time

import numpy as np
from aeon.datasets import load_classification
from aeon.distances import dtw_pairwise_distance

import dtw_simd

rng = np.random.default_rng(0)

# --- correctness: real ED slice + random long series ------------------------
X, _ = load_classification("ElectricDevices")
X = np.ascontiguousarray(X[:, 0, :].astype(np.float64))
sub = X[:200]
D_aeon = dtw_pairwise_distance(sub)
D_ours = dtw_simd.dtw_pairwise(sub)
diff = np.abs(D_aeon - D_ours)
print(
    f"correctness ED slice (200x{sub.shape[1]}): max|diff|={diff.max():.3e}  allclose={np.allclose(D_aeon, D_ours)}"
)

lng = np.ascontiguousarray(rng.normal(size=(40, 1024)))
Da = dtw_pairwise_distance(lng)
Do = dtw_simd.dtw_pairwise(lng)
print(
    f"correctness long (40x1024): max|diff|={np.abs(Da-Do).max():.3e}  allclose={np.allclose(Da, Do)}"
)


# --- throughput --------------------------------------------------------------
def cells(n, L):
    return (n * (n - 1) / 2) * L * L


for label, data in (
    ("ED-like 600x96", X[:600]),
    ("SLC-like 150x1024", np.ascontiguousarray(rng.normal(size=(150, 1024)))),
):
    n, L = data.shape
    t0 = time.time()
    dtw_pairwise_distance(data)
    t_a = time.time() - t0
    t0 = time.time()
    dtw_simd.dtw_pairwise(data)
    t_o = time.time() - t0
    ra, ro = cells(n, L) / t_a, cells(n, L) / t_o
    print(
        f"{label}: aeon {t_a:.1f}s ({ra:.2e} cells/s) | simd {t_o:.2f}s ({ro:.2e} cells/s) | speedup x{t_a/t_o:.1f}"
    )

# extrapolations at the simd rate measured on the long-series case
print("\nextrapolations at the SLC-like simd rate:")
t_slc_full = cells(9236, 1024) / ro / 3600
t_ed_full = cells(16637, 96) / ro / 60
print(
    f"  StarLightCurves FULL N=9236: {t_slc_full:.1f} h  (aeon measured-rate estimate was ~30 h)"
)
print(f"  ElectricDevices FULL N=16637: {t_ed_full:.0f} min")
