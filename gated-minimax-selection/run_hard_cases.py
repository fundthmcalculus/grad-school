"""The hard-case map: where does each aggregation strategy actually fail?

Third driver in the non-Euclidean series (after run_nonmetric.py and
run_bridge_repair.py). The violation sweep characterized CORRUPTION -- entries
inconsistent with the rest of the matrix, which the reverse-TI repair can
detect and fix. This driver maps the two GEOMETRIC failure regimes, where
every distance is a real distance (metric, r_hat = 0, repair correctly inert)
and the damage comes from actual points sitting between clusters:

H1 Hub dose-response -- `knn_graph_hubs(n_hubs)`: hub nodes wired across
   communities, the exaggerated form of kNN-graph hubness. Masked scoring
   (hubs labeled -1 and excluded from ARI, but the methods see the full
   matrix, matching run_all.py's noise convention).

H2 Tail dose-response -- `heavy_tailed_blobs(df)`: Student-t cluster noise
   throwing genuine members into the void as bridge points.

`python run_hard_cases.py` writes (JSON before figures):

  - outputs/hard_cases_results.json
  - outputs/fig18_hard_cases.png
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
import nonmetric_data as ND
import selection as S
from metric_repair import auto_repair
from nerfcm import nerfcm

OUT = "./outputs"
SEEDS = [0, 1, 2, 3, 4]  # NERFCM restarts, matching run_all.py
DATASET_SEEDS = [0, 1, 2]

HUB_GRID = [0, 1, 2, 3, 6]
DF_GRID = [1.2, 1.5, 2.0, 3.0, 5.0]

results: dict = {}


def _masked_ari(y, lab):
    m = y >= 0
    return float(adjusted_rand_score(y[m], lab[m])) if m.sum() else float("nan")


def _cell(D, y):
    """One dataset through the battery, with masked scoring and the repair
    inertness check."""
    Dstar = im.minimax_transform_fast(D)

    nerfcm_d, nerfcm_ds = [], []
    for s in SEEDS:
        U, _, _ = nerfcm(D, 3, seed=s)
        nerfcm_d.append(_masked_ari(y, np.argmax(U, axis=0)))
        Us, _, _ = nerfcm(Dstar, 3, seed=s)
        nerfcm_ds.append(_masked_ari(y, np.argmax(Us, axis=0)))

    sel = S.select_coverage_cover(Dstar)
    n = D.shape[0]
    if sel:
        Db = np.zeros((len(sel), n))
        for k, b in enumerate(sel):
            mem = np.array(sorted(b["members"]), dtype=int)
            Db[k] = Dstar[:, mem].min(axis=1)
        cover_ari = _masked_ari(y, np.argmin(Db, axis=0))
        cover_k, cover_cov = len(sel), float(S.coverage_of(sel, n))
    else:
        cover_ari, cover_k, cover_cov = None, 0, 0.0

    _, info = auto_repair(D)

    return {
        "NERFCM_D_ari": round(float(np.mean(nerfcm_d)), 3),
        "NERFCM_Dstar_ari": round(float(np.mean(nerfcm_ds)), 3),
        "cover_k": cover_k,
        "cover_coverage": round(cover_cov, 3),
        "cover_ari": None if cover_ari is None else round(cover_ari, 3),
        "repair_r_hat": round(info["r_hat"], 4),
        "repair_declined": info["declined"],
    }


def _averaged(cells):
    """Average the numeric fields of per-seed cells; keep abstention visible
    by averaging cover_ari with None treated as 0 and reporting abstain count."""
    out = {}
    for key in (
        "NERFCM_D_ari",
        "NERFCM_Dstar_ari",
        "cover_k",
        "cover_coverage",
        "repair_r_hat",
    ):
        out[key] = round(float(np.mean([c[key] for c in cells])), 3)
    aris = [0.0 if c["cover_ari"] is None else c["cover_ari"] for c in cells]
    out["cover_ari"] = round(float(np.mean(aris)), 3)
    out["cover_ari_std"] = round(float(np.std(aris)), 3)
    out["cover_abstained"] = sum(1 for c in cells if c["cover_ari"] is None)
    out["repair_ever_lifted"] = any(c["repair_r_hat"] > 0 for c in cells)
    return out


def run_hub_sweep():
    table = {}
    for n_hubs in HUB_GRID:
        cells = []
        for ds in DATASET_SEEDS:
            D, y = ND.knn_graph_hubs(n_hubs=n_hubs, seed=209 + 100 * ds)
            cells.append(_cell(D, y))
        table[str(n_hubs)] = _averaged(cells)
    results["hub_sweep"] = table
    return table


def run_tail_sweep():
    table = {}
    for df in DF_GRID:
        cells = []
        for ds in DATASET_SEEDS:
            D, y = ND.heavy_tailed_blobs(df=df, seed=210 + 100 * ds)
            cells.append(_cell(D, y))
        table[str(df)] = _averaged(cells)
    results["tail_sweep"] = table
    return table


def fig_hard_cases(hub, tail):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    series = [
        ("NERFCM_D_ari", "NERFCM(D)", "steelblue"),
        ("NERFCM_Dstar_ari", "NERFCM(D*)", "coral"),
        ("cover_ari", "gap-cover(D*)", "seagreen"),
    ]
    for key, label, color in series:
        ax1.plot(
            HUB_GRID,
            [hub[str(g)][key] for g in HUB_GRID],
            marker="o",
            ms=4,
            label=label,
            color=color,
        )
    ax1.set_xlabel("number of hub nodes")
    ax1.set_ylabel("ARI (hubs masked from scoring)")
    ax1.set_title("H1: kNN-graph hubs break BOTH aggregations")
    ax1.set_ylim(-0.05, 1.1)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    for key, label, color in series:
        ax2.plot(
            DF_GRID,
            [tail[str(g)][key] for g in DF_GRID],
            marker="o",
            ms=4,
            label=label,
            color=color,
        )
    ax2.set_xlabel("Student-t degrees of freedom (left = heavier tails)")
    ax2.set_ylabel("ARI")
    ax2.set_title("H2: heavy tails -- averaging degrades, minimax collapses/abstains")
    ax2.set_ylim(-0.05, 1.1)
    ax2.grid(alpha=0.3)
    fig.suptitle(
        "Geometric bridges (metric, repair inert): the not-repairable regime",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig18_hard_cases.png", dpi=96, bbox_inches="tight")
    plt.close(fig)


def main():
    import os

    os.makedirs(OUT, exist_ok=True)
    print("H1: hub dose-response...")
    hub = run_hub_sweep()
    print("H2: heavy-tail dose-response...")
    tail = run_tail_sweep()

    results["params"] = {
        "hub_grid": HUB_GRID,
        "df_grid": DF_GRID,
        "dataset_seeds": DATASET_SEEDS,
        "nerfcm_restarts": SEEDS,
        "scoring": "ARI masked to y >= 0 (run_all.py noise convention)",
    }
    with open(f"{OUT}/hard_cases_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Numeric results -> {OUT}/hard_cases_results.json")
    fig_hard_cases(hub, tail)
    print("Figure -> fig18_hard_cases.png in", OUT)

    print("\nH1 HUBS (masked ARI; repair r_hat should be 0 everywhere):")
    for g in HUB_GRID:
        e = hub[str(g)]
        print(
            f"  n_hubs={g}: NERFCM(D)={e['NERFCM_D_ari']} NERFCM(D*)={e['NERFCM_Dstar_ari']} "
            f"cover={e['cover_ari']} (k={e['cover_k']}, cov={e['cover_coverage']}, "
            f"abstained {e['cover_abstained']}/3) r_hat={e['repair_r_hat']}"
        )
    print("\nH2 HEAVY TAILS:")
    for g in DF_GRID:
        e = tail[str(g)]
        print(
            f"  df={g}: NERFCM(D)={e['NERFCM_D_ari']} NERFCM(D*)={e['NERFCM_Dstar_ari']} "
            f"cover={e['cover_ari']} (k={e['cover_k']}, cov={e['cover_coverage']}, "
            f"abstained {e['cover_abstained']}/3) r_hat={e['repair_r_hat']}"
        )


if __name__ == "__main__":
    main()
