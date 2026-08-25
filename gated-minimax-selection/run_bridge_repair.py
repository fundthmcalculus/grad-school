"""Does one-sided metric repair rescue the minimax pipeline from shortcuts?

Companion to run_nonmetric.py: that driver established (finding 2) that
shortcut-type corruption -- deflated entries acting as single-linkage bridges
-- collapses every D*-based method while stretch-type corruption is harmless.
This driver measures the defense metric_repair.reverse_ti_repair on the SAME
sweep cells, so the two result sets compose into one dose-response story.

`python run_bridge_repair.py` writes (JSON before figures, as always):

  - outputs/bridge_repair_results.json
  - outputs/fig16_repair_doseresponse.png
  - outputs/fig17_repair_noharm.png

Experiments:

R1 Dose-response -- the shortcut sweep (same axes as run_nonmetric E3) with
   the pipeline run on raw D vs repaired(D, q=0.5) vs repaired(D, q=0.75).
   The stretch sweep is included as a no-harm axis.

R2 No-harm battery -- fraction of entries lifted and ARI deltas on every
   non-Euclidean family from run_nonmetric E2 (all essentially untouched:
   metric families exactly, DTW/cosine because their violations are
   stretch-type), plus the real-DTW flight data when the .h5 is present.

R3 Multi-scale restoration -- the relational nested hierarchy at the
   corruption levels where select_multiscale was measured to break
   (fine band lost at rate 0.1 / strength 1.0; full collapse at 0.2), with
   and without repair.
"""

from __future__ import annotations

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score

import ivat_mf as im
import multiscale_persistence as MS
import nonmetric_data as ND
from metric_repair import reverse_ti_repair
from run_nonmetric import (
    NCMAPSS_DS01,
    SEEDS,
    DATASET_SEEDS,
    load_flight_traces,
    nerfcm_score,
    score_selection,
)
import selection as S

OUT = "./outputs"
QS = [0.5, 0.75]

results: dict = {}


def _cover(Dstar, y):
    sel = S.select_coverage_cover(Dstar)
    k, cov, ari = score_selection(Dstar, y, sel)
    return k, cov, ari


# ---------------------------------------------------------------------------
# R1: dose-response on the violation sweep
# ---------------------------------------------------------------------------

SWEEP_STRENGTHS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SWEEP_RATES = [0.0, 0.05, 0.1, 0.2, 0.4]
FIXED_RATE = 0.2
FIXED_STRENGTH = 1.0  # the collapse regime; run_nonmetric used 0.8 for by_rate


def _r1_cell(mode, rate, strength):
    """Mean over dataset replicates: cover + NERFCM(D*) for raw and repaired."""
    out = {"raw": {"cover": [], "nerfcm": []}}
    for q in QS:
        out[f"q{q}"] = {"cover": [], "nerfcm": []}
    for ds in DATASET_SEEDS:
        D0, y, _ = ND.euclidean_blobs(seed=206 + 100 * ds)
        Dv = ND.violate_pairs(D0, rate, strength, mode, seed=1 + ds)
        variants = {"raw": Dv}
        for q in QS:
            variants[f"q{q}"] = reverse_ti_repair(Dv, q)
        for name, Dx in variants.items():
            Dstar = im.minimax_transform_fast(Dx)
            _, _, ari_c = _cover(Dstar, y)
            m_ds, _, _ = nerfcm_score(Dstar, y, 3)
            out[name]["cover"].append(0.0 if ari_c is None else ari_c)
            out[name]["nerfcm"].append(m_ds)
    cell = {}
    for name, d in out.items():
        cell[name] = {
            "cover_ari": round(float(np.mean(d["cover"])), 4),
            "cover_ari_std": round(float(np.std(d["cover"])), 4),
            "nerfcm_dstar_ari": round(float(np.mean(d["nerfcm"])), 4),
            "nerfcm_dstar_ari_std": round(float(np.std(d["nerfcm"])), 4),
        }
    return cell


def run_dose_response():
    table = {"by_strength": {}, "by_rate": {}}
    for mode in ("shortcut", "stretch"):
        table["by_strength"][mode] = {
            str(s): _r1_cell(mode, FIXED_RATE, s) for s in SWEEP_STRENGTHS
        }
        table["by_rate"][mode] = {
            str(r): _r1_cell(mode, r, FIXED_STRENGTH) for r in SWEEP_RATES
        }
    table["params"] = {
        "strengths": SWEEP_STRENGTHS,
        "rates": SWEEP_RATES,
        "fixed_rate": FIXED_RATE,
        "fixed_strength": FIXED_STRENGTH,
        "quantiles": QS,
        "dataset_seeds": DATASET_SEEDS,
        "nerfcm_restarts": SEEDS,
    }
    results["dose_response"] = table
    return table


# ---------------------------------------------------------------------------
# R2: no-harm battery
# ---------------------------------------------------------------------------


def _noharm_row(D, y, k_true):
    row = {}
    Dstar = im.minimax_transform_fast(D)
    _, _, ari0 = _cover(Dstar, y)
    m0, _, _ = nerfcm_score(D, y, k_true)
    row["raw"] = {"cover_ari": ari0, "nerfcm_D_ari": round(m0, 3)}
    for q in QS:
        Dr = reverse_ti_repair(D, q)
        lifted = float(np.mean(Dr > D + 1e-12))
        Dstar_r = im.minimax_transform_fast(Dr)
        _, _, ari_r = _cover(Dstar_r, y)
        m_r, _, _ = nerfcm_score(Dr, y, k_true)
        row[f"q{q}"] = {
            "fraction_lifted": round(lifted, 4),
            "cover_ari": ari_r,
            "nerfcm_D_ari": round(m_r, 3),
        }
    return row


def run_noharm():
    import os

    table = {}
    for name, (fn, k_true) in ND.BATTERY.items():
        D, y = fn()
        table[name] = _noharm_row(D, y, k_true)
    # clean Euclidean base
    D0, y0, _ = ND.euclidean_blobs()
    table["blobs_clean"] = _noharm_row(D0, y0, 3)
    # real DTW flight data, if present
    if os.path.exists(NCMAPSS_DS01):
        traces, y, _ = load_flight_traces()
        D = ND.pairwise(traces, ND.dtw_distance)
        table["real_dtw_ncmapss"] = _noharm_row(D, y, 3)
    else:
        table["real_dtw_ncmapss"] = {"skipped": "h5 not present"}
    results["noharm"] = table
    return table


# ---------------------------------------------------------------------------
# R3: multi-scale restoration
# ---------------------------------------------------------------------------

R3_CELLS = [(0.05, 0.8), (0.1, 1.0), (0.2, 1.0)]


def _ms_per_level(Dstar, levels):
    msel = MS.select_multiscale(Dstar)
    labs = [MS.assign_band(b, Dstar) for b in msel.bands]
    per = {}
    for lname, y in levels.items():
        per[lname] = max(
            (round(float(adjusted_rand_score(y, l)), 3) for l in labs), default=None
        )
    return per, [len(b.blocks) for b in msel.bands]


def run_multiscale_restoration():
    D, yf, yc = ND.relational_nested_hierarchy()
    levels = {"fine6": yf, "coarse3": yc}
    table = {}
    for rate, strength in R3_CELLS:
        Dv = ND.violate_pairs(D, rate, strength, "shortcut", seed=3)
        entry = {}
        per, bands = _ms_per_level(im.minimax_transform_fast(Dv), levels)
        entry["raw"] = {"per_level": per, "bands": bands}
        for q in QS:
            Dr = reverse_ti_repair(Dv, q)
            per, bands = _ms_per_level(im.minimax_transform_fast(Dr), levels)
            entry[f"q{q}"] = {"per_level": per, "bands": bands}
        table[f"shortcut r={rate} s={strength}"] = entry
    results["multiscale_restoration"] = table
    return table


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def save_figure(fig, filename):
    fig.savefig(f"{OUT}/{filename}", dpi=96, bbox_inches="tight")
    plt.close(fig)


def fig_dose_response(table):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    panels = [
        (
            "by_strength",
            "shortcut",
            SWEEP_STRENGTHS,
            f"shortcut: strength (rate={FIXED_RATE})",
        ),
        (
            "by_rate",
            "shortcut",
            SWEEP_RATES,
            f"shortcut: rate (strength={FIXED_STRENGTH})",
        ),
        (
            "by_strength",
            "stretch",
            SWEEP_STRENGTHS,
            f"stretch: strength (rate={FIXED_RATE}) [no-harm axis]",
        ),
        (
            "by_rate",
            "stretch",
            SWEEP_RATES,
            f"stretch: rate (strength={FIXED_STRENGTH}) [no-harm axis]",
        ),
    ]
    series = [
        ("raw", "unrepaired", "coral"),
        ("q0.5", "repaired q=.5", "seagreen"),
        ("q0.75", "repaired q=.75", "steelblue"),
    ]
    for ax, (axis, mode, grid, title) in zip(axes.ravel(), panels):
        cells = table[axis][mode]
        for skey, label, color in series:
            vals = [cells[str(g)][skey]["cover_ari"] for g in grid]
            errs = [cells[str(g)][skey]["cover_ari_std"] for g in grid]
            ax.errorbar(
                grid,
                vals,
                yerr=errs,
                label=label,
                color=color,
                marker="o",
                ms=4,
                capsize=2,
            )
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=0.3)
        ax.set_ylabel("gap-cover ARI")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "One-sided metric repair restores the minimax pipeline under shortcuts",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "fig16_repair_doseresponse.png")


def fig_noharm(table):
    names = [n for n, e in table.items() if "skipped" not in e]
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    width = 0.38
    for i, q in enumerate(QS):
        lifted = [table[n][f"q{q}"]["fraction_lifted"] for n in names]
        ax1.bar(x + (i - 0.5) * width, lifted, width, label=f"q={q}")
    ax1.set_ylabel("fraction of entries lifted")
    ax1.set_title("How much the repair touches uncorrupted data")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax1.legend()

    for i, q in enumerate(QS):
        deltas = []
        for n in names:
            a0 = table[n]["raw"]["cover_ari"]
            a1 = table[n][f"q{q}"]["cover_ari"]
            deltas.append((0.0 if a1 is None else a1) - (0.0 if a0 is None else a0))
        ax2.bar(x + (i - 0.5) * width, deltas, width, label=f"q={q}")
    ax2.set_ylabel("gap-cover ARI delta (repaired - raw)")
    ax2.set_title("Effect on clustering quality")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.legend()
    fig.suptitle("No-harm profile of reverse-TI repair", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig17_repair_noharm.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    import os

    os.makedirs(OUT, exist_ok=True)
    print("R1: dose-response sweep (raw vs repaired)...")
    dose = run_dose_response()
    print("R2: no-harm battery...")
    noharm = run_noharm()
    print("R3: multi-scale restoration...")
    ms = run_multiscale_restoration()

    with open(f"{OUT}/bridge_repair_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Numeric results -> {OUT}/bridge_repair_results.json")

    fig_dose_response(dose)
    fig_noharm(noharm)
    print("Figures -> fig16, fig17 in", OUT)

    print("\nDOSE-RESPONSE (gap-cover ARI, shortcut by strength at rate 0.2):")
    for s in SWEEP_STRENGTHS:
        cell = dose["by_strength"]["shortcut"][str(s)]
        print(
            f"  s={s}: raw={cell['raw']['cover_ari']} "
            f"q.5={cell['q0.5']['cover_ari']} q.75={cell['q0.75']['cover_ari']}"
        )
    print("\nNO-HARM (fraction lifted / cover ARI delta at q=0.5):")
    for n, e in noharm.items():
        if "skipped" in e:
            print(f"  {n}: skipped")
            continue
        a0 = e["raw"]["cover_ari"]
        a1 = e["q0.5"]["cover_ari"]
        d = (0.0 if a1 is None else a1) - (0.0 if a0 is None else a0)
        print(f"  {n}: lifted={e['q0.5']['fraction_lifted']} dARI={d:+.3f}")
    print("\nMULTI-SCALE RESTORATION (best-band ARI fine/coarse):")
    for cell, e in ms.items():
        print(
            f"  {cell}: raw={e['raw']['per_level']} (bands={e['raw']['bands']}) "
            f"q.5={e['q0.5']['per_level']} (bands={e['q0.5']['bands']}) "
            f"q.75={e['q0.75']['per_level']} (bands={e['q0.75']['bands']})"
        )


if __name__ == "__main__":
    main()
