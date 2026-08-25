"""E7: real fault-mode clustering from N-CMAPSS degradation trajectories.

The gap NONMETRIC_FINDINGS' E5 left open: flight-class truth is a duration
bin, not a cluster. Fault modes ARE cluster-shaped truth -- each N-CMAPSS
dataset plants a distinct failure mode (DS01: HPT efficiency; DS04: fan;
DS07: LPT), so pooling units across datasets and clustering their degradation
trajectories asks a real question: can trajectory shape alone recover which
component is failing?

Pipeline (per unit, dev + test splits, 10 units per dataset, n = 30):
  1. Per-cycle medians of the 4 operating conditions (W) and 14 sensors (X_s).
  2. CONDITION CORRECTION -- the step this repo's CMAPSS work already
     established as decisive (cluster-tendency memo: RUL R^2 0.02 -> 0.81):
     ridge-fit each sensor on [1, W] over the unit's first 10 (healthy)
     cycles; the trajectory is the residual sequence. Without it, per-cycle
     sensor levels are flight-regime, not degradation (measured here too:
     raw-trajectory ARI ~= 0.03).
  3. Keep 4 mode-informative channels (T48, T50, P15, P21 -- turbine temps
     vs fan-path pressures), scale by pooled residual std, median-smooth
     (7 cycles), keep the final 40% of life (the degradation segment; the
     healthy prefix is noise that dilutes DTW).
  4. Multivariate DTW (Euclidean local cost) -> the battery.

Skips gracefully (JSON stub) when the gitignored .h5 files are absent.

`python run_fault_modes.py` writes outputs/fault_modes_results.json (before
the figure) + outputs/fig19_fault_modes.png.
"""

from __future__ import annotations

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score

import ivat_mf as im
import nonmetric_data as ND
import selection as S
from nerfcm import nerfcm

OUT = "./outputs"
SEEDS = [0, 1, 2, 3, 4]

DATASETS = {
    "N-CMAPSS_DS01-005": "HPT efficiency",
    "N-CMAPSS_DS04": "fan",
    "N-CMAPSS_DS07": "LPT",
}
DATA_DIR = "../NASA-CMAPSS"
CH_ALL = [
    "T24",
    "T30",
    "T48",
    "T50",
    "P15",
    "P2",
    "P21",
    "P24",
    "Ps30",
    "P40",
    "P50",
    "Nf",
    "Nc",
    "Wf",
]
CH_USE = [2, 3, 4, 6]  # T48, T50, P15, P21
N_HEALTHY = 10
SMOOTH_CYCLES = 7
TAIL_FRACTION = 0.4

results: dict = {}


def unit_residual_trajectories(path):
    """Per unit: condition-corrected residual trajectory, (n_cycles, 14)."""
    import h5py

    out = []
    with h5py.File(path, "r") as f:
        for split in ("dev", "test"):
            A = f[f"A_{split}"][:]
            W = f[f"W_{split}"][:]
            XS = f[f"X_s_{split}"][:]
            key = (A[:, 0] * 10000 + A[:, 1]).astype(np.int64)
            order = np.argsort(key, kind="stable")
            key_s, u_s = key[order], A[order, 0]
            W_s, XS_s = W[order], XS[order]
            _, starts = np.unique(key_s, return_index=True)
            ends = np.append(starts[1:], len(key_s))
            per_unit: dict = {}
            for s, e in zip(starts, ends):
                per_unit.setdefault(u_s[s], []).append(
                    (np.median(W_s[s:e], axis=0), np.median(XS_s[s:e], axis=0))
                )
            for _, rows in per_unit.items():
                Wm = np.array([r[0] for r in rows])
                Xm = np.array([r[1] for r in rows])
                nh = min(N_HEALTHY, len(rows) // 2)
                Phi = np.hstack([np.ones((len(rows), 1)), Wm])
                beta = np.linalg.solve(
                    Phi[:nh].T @ Phi[:nh] + 1e-3 * np.eye(Phi.shape[1]),
                    Phi[:nh].T @ Xm[:nh],
                )
                out.append(Xm - Phi @ beta)
    return out


def prepare_trajectories():
    """(trajectories, labels, mode_names) -- smoothed, scaled, tail-cropped."""
    from scipy.ndimage import median_filter

    raw, y, names = [], [], []
    for li, (name, mode) in enumerate(DATASETS.items()):
        ts = unit_residual_trajectories(os.path.join(DATA_DIR, f"{name}.h5"))
        raw += [t[:, CH_USE] for t in ts]
        y += [li] * len(ts)
        names.append(mode)
    y = np.asarray(y, dtype=int)
    scale = np.vstack(raw).std(axis=0)
    scale[scale == 0] = 1.0
    trajs = []
    for t in raw:
        t = median_filter(t / scale, size=(SMOOTH_CYCLES, 1), mode="nearest")
        cut = int((1.0 - TAIL_FRACTION) * len(t))
        trajs.append(t[cut:])
    return trajs, y, names, raw, scale


def run_fault_modes():
    if not all(os.path.exists(os.path.join(DATA_DIR, f"{n}.h5")) for n in DATASETS):
        results["fault_modes"] = {
            "skipped": "one or more N-CMAPSS .h5 files absent (gitignored data)"
        }
        return None

    trajs, y, mode_names, raw, scale = prepare_trajectories()
    D = ND.pairwise(trajs, ND.dtw_distance_multivariate)
    Dstar = im.minimax_transform_fast(D)

    tv = ND.triangle_violation_stats(D)

    def nerfcm_mean(Dx, yy, c):
        return float(
            np.mean(
                [
                    adjusted_rand_score(yy, np.argmax(nerfcm(Dx, c, seed=s)[0], 0))
                    for s in SEEDS
                ]
            )
        )

    m3 = nerfcm_mean(D, y, 3)
    m3s = nerfcm_mean(Dstar, y, 3)
    y2 = (y == 1).astype(int)  # fan vs turbine (HPT+LPT)
    m2 = nerfcm_mean(D, y2, 2)

    sel = S.select_coverage_cover(Dstar)
    n = D.shape[0]
    if sel:
        Db = np.zeros((len(sel), n))
        for k, b in enumerate(sel):
            mem = np.array(sorted(b["members"]), dtype=int)
            Db[k] = Dstar[:, mem].min(axis=1)
        cover = {
            "k": len(sel),
            "coverage": round(float(S.coverage_of(sel, n)), 3),
            "ari": round(float(adjusted_rand_score(y, np.argmin(Db, 0))), 3),
        }
    else:
        cover = {"k": 0, "coverage": 0.0, "ari": None}

    # Reference: endpoint fingerprint (mean residual over the last 10 cycles,
    # Euclidean) -- what a static feature vector achieves without the
    # trajectory. DTW beating this is the pro-trajectory finding.
    from scipy.spatial.distance import pdist, squareform

    F = np.array([(r / scale)[-10:].mean(axis=0) for r in raw])
    Dfp = squareform(pdist(F))
    m_fp = nerfcm_mean(Dfp, y, 3)

    table = {
        "setup": {
            "datasets": {k: v for k, v in DATASETS.items()},
            "n_units": int(n),
            "channels": [CH_ALL[i] for i in CH_USE],
            "condition_correction": f"per-unit ridge on [1, W], first {N_HEALTHY} cycles",
            "smoothing_cycles": SMOOTH_CYCLES,
            "tail_fraction": TAIL_FRACTION,
        },
        "ti_violation_pair_fraction": round(tv["pair_violation_fraction"], 4),
        "NERFCM_D_ari_3way": round(m3, 3),
        "NERFCM_Dstar_ari_3way": round(m3s, 3),
        "NERFCM_D_ari_fan_vs_turbine": round(m2, 3),
        "gap_cover": cover,
        "fingerprint_reference_ari_3way": round(m_fp, 3),
    }
    results["fault_modes"] = table
    return table, trajs, y, mode_names


def fig_fault_modes(trajs, y, mode_names):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    ch_labels = [CH_ALL[i] for i in CH_USE]
    colors = plt.cm.tab10(np.linspace(0, 1, len(CH_USE)))
    for mode, ax in enumerate(axes):
        members = [t for t, yy in zip(trajs, y) if yy == mode]
        # align on the tail: plot vs fraction of remaining life
        for ci in range(len(CH_USE)):
            for t in members:
                x = np.linspace(0, 1, len(t))
                ax.plot(x, t[:, ci], color=colors[ci], alpha=0.25, lw=0.8)
            mean_len = int(np.median([len(t) for t in members]))
            resampled = np.array(
                [
                    np.interp(
                        np.linspace(0, 1, mean_len), np.linspace(0, 1, len(t)), t[:, ci]
                    )
                    for t in members
                ]
            )
            ax.plot(
                np.linspace(0, 1, mean_len),
                resampled.mean(axis=0),
                color=colors[ci],
                lw=2.0,
                label=ch_labels[ci] if mode == 0 else None,
            )
        ax.set_title(f"{mode_names[mode]}", fontsize=10)
        ax.set_xlabel("degradation segment (fraction)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("condition-corrected residual (pooled-std units)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Fault modes leave distinct residual signatures (turbine temps vs fan-path pressures)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig19_fault_modes.png", dpi=96, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("E7: fault-mode DTW clustering (DS01 HPT / DS04 fan / DS07 LPT)...")
    out = run_fault_modes()
    with open(f"{OUT}/fault_modes_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Numeric results -> {OUT}/fault_modes_results.json")
    if out is None:
        print("skipped: data not present")
        return
    table, trajs, y, mode_names = out
    fig_fault_modes(trajs, y, mode_names)
    print("Figure -> fig19_fault_modes.png in", OUT)
    print("\nFAULT-MODE CLUSTERING (n=30 units, 3 modes):")
    print(f"  TI-violated pairs: {table['ti_violation_pair_fraction']}")
    print(
        f"  NERFCM(D) 3-way={table['NERFCM_D_ari_3way']} "
        f"NERFCM(D*) 3-way={table['NERFCM_Dstar_ari_3way']} "
        f"fan-vs-turbine={table['NERFCM_D_ari_fan_vs_turbine']}"
    )
    print(
        f"  gap-cover: {table['gap_cover']} | "
        f"fingerprint reference 3-way={table['fingerprint_reference_ari_3way']}"
    )


if __name__ == "__main__":
    main()
