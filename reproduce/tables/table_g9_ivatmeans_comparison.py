"""Goal G9 / checklist C11 -- IVATMeans against Fuzzy C-Means, k-means and E-means.

Chapter 3 §3.3.5 presents IVATMeans as a contribution, and nothing in the repository
had ever timed it or scored its partitions against FCM or k-means. This generator
does both, and adds E-means (Bessa et al., arXiv:2611.09501, shipped in faiss 1.16.0
as faiss.contrib.e_means) as a fourth arm.

Two halves, per the G9 decision rule (prose/07-goals-for-completion.md):

  1. Performance at scale. Wall clock over the N ladder on the shuttle set
     (N = 1,000 ... 58,000, k = 7, z-scored features), ten seeds. The seed draws the
     subsample and sets each arm's random state. IVATMeans holds an N x N float32
     matrix; any rung whose predicted peak (17 bytes × N², calibrated) exceeds 80% of physical RAM is marked
     INFEASIBLE and not run. It is never silently dropped.

  2. Quality of construction. ARI against ground truth on the six Table 3.5
     constructions (two moons, circles, aniso, varied_density, bridged, easy_blobs),
     generated from seeds 0..9 by ClusteringExperiments/adversarial_eval.py, with
     matched k. Bridge points (label -1) are excluded from ARI, as in Table 3.5.

Determinism asymmetry: IVATMeans has no random state in its construction. It is run
under all ten seeds on a fixed dataset, and the number of distinct labelings is
reported (1 = elementwise identical). FCM, k-means and E-means are reported as
spreads over their seeds.

Pre-registered outcomes, written before the run:

  P1  Performance: IVATMeans wall clock within 10x of FCM at every rung IVATMeans can
      run. Failure at a feasible rung fails P1.
  P2  Predicted loss: on two moons and circles, IVATMeans lands near the k-means row
      of Table 3.5 (about 0.27 and 0.00), not near exact VAT's 1.00. Mean ARI >= 0.99
      on either set REFUTES the Euclidean-prototype bound (G9 §2): the refutation
      condition is the predicted loss, and it is reported either way.
  P3  Blob set: IVATMeans mean ARI on easy_blobs within 0.05 of max(FCM, k-means).
  P4  Determinism: IVATMeans produces one distinct labelling across the ten seeds on
      every quality set.
  P5  Manipulation check: the seed varies the baseline arms. At least one baseline has
      more than one distinct labelling across seeds on at least one set, otherwise the
      seed sweep did not vary anything and P4 is vacuous.

Run from the repo root, in the tribble-cluster environment built outside the submodule:

    UV_PROJECT_ENVIRONMENT=$HOME/.cache/grad-school/venv-tribble-cluster \\
      uv run --project tribble-cluster --with faiss-cpu==1.16.0 \\
        --with scikit-learn --with scipy --with matplotlib --with pandas \\
        python reproduce/tables/table_g9_ivatmeans_comparison.py

Knobs:
    REPRO_SEEDS="0,1,2"          narrow the seed set (smoke runs only, NOT CITABLE)
    REPRO_G9_LADDER="1000,2000"  override the N ladder
    REPRO_G9_SKIP_LADDER=1       run the quality half only
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_TABLES = os.path.dirname(os.path.abspath(__file__))
_REPRO = os.path.dirname(_TABLES)
_ROOT = os.path.dirname(_REPRO)
sys.path.insert(0, _REPRO)
sys.path.insert(0, _TABLES)
sys.path.insert(0, os.path.join(_ROOT, "ClusteringExperiments"))
import common as C  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import faiss  # noqa: E402
from faiss.contrib.e_means import NumpyEmeans  # noqa: E402
from sklearn.cluster import kmeans_plusplus  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from tribbleclustering import FuzzyCMeans, IVATMeans, KMeans  # noqa: E402
import adversarial_eval as AE  # noqa: E402  (Table 3.5 constructions)

SEEDS = list(C.SEEDS)
LADDER = [
    int(v)
    for v in os.environ.get(
        "REPRO_G9_LADDER", "1000,2000,5000,16000,32000,58000"
    ).split(",")
]
SKIP_LADDER = os.environ.get("REPRO_G9_SKIP_LADDER") == "1"
LADDER_K = 7
MEM_BUDGET_FRAC = 0.8
# Measured peak RSS of IVATMeans.fit is 16-18 bytes per N^2 (N=8,000: 17.7; N=16,000: 16.4;
# N=32,000 was OOM-killed at 13.9 GB resident on this 15 GB host). Calibrated, not assumed.
MEM_BYTES_PER_N2 = 17
DATA_ROOT = os.environ.get("GRAD_SCHOOL_DATA", os.path.join(_ROOT, "data"))
REPORT = os.path.join(C.OUTPUT_DIR, "g9_comparison_report.md")


def _phys_ram_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


# --------------------------------------------------------------------------- #
# arms: each returns hard labels for (X, k, seed)
# --------------------------------------------------------------------------- #
def fit_ivat(X, k, s):
    return np.asarray(IVATMeans(n_clusters=k, random_state=s).fit(X).labels_)


def fit_ivat_rel(X, k, s):
    return np.asarray(
        IVATMeans(n_clusters=k, refine="relational", random_state=s).fit(X).labels_
    )


def fit_fcm(X, k, s):
    return np.asarray(FuzzyCMeans(k, random_state=s).fit(X).labels_)


def fit_km(X, k, s):
    return np.asarray(KMeans(k, random_state=s).fit(X).labels_)


def fit_em_default(X, k, s):
    X32 = np.ascontiguousarray(X, np.float32)
    _, lab = NumpyEmeans(X.shape[1], k, seed=s).train(
        X32
    )  # faiss default: 5 epochs, its own init
    return np.asarray(lab).ravel()


def fit_em_kpp25(X, k, s):
    X32 = np.ascontiguousarray(X, np.float32)
    c0, _ = kmeans_plusplus(X, k, random_state=s)
    _, lab = NumpyEmeans(X.shape[1], k, nepochs=25, seed=s).train(
        X32, init_centroids=np.ascontiguousarray(c0, np.float32)
    )
    return np.asarray(lab).ravel()


PERF_ARMS = [
    ("ivat", "IVATMeans", fit_ivat),
    ("fcm", "FCM (m=2)", fit_fcm),
    ("kmeans", "k-means (k-means++)", fit_km),
    ("em5", "E-means (5 ep, default init)", fit_em_default),
    ("em25", "E-means (25 ep, k-means++ init)", fit_em_kpp25),
]
QUALITY_ARMS = PERF_ARMS + [("ivat_rel", "IVATMeans (relational refine)", fit_ivat_rel)]


# --------------------------------------------------------------------------- #
# half 1: performance ladder on shuttle
# --------------------------------------------------------------------------- #
def shuttle_z() -> np.ndarray:
    df = pd.read_csv(os.path.join(DATA_ROOT, "shuttle.csv"))
    X = df.iloc[:, :-1].to_numpy(float)
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def run_ladder(X_all):
    ram = _phys_ram_bytes()
    budget = MEM_BUDGET_FRAC * ram
    rows, raw = [], {}
    # JIT warm-up so numba compile time is not billed to the first rung
    warm = X_all[:300]
    for key, _, fn in PERF_ARMS:
        fn(warm, LADDER_K, 0)
    for N in LADDER:
        raw[N] = {}
        ivat_bytes = N * N * MEM_BYTES_PER_N2
        ivat_ok = ivat_bytes <= budget
        for key, _, fn in PERF_ARMS:
            if key == "ivat" and not ivat_ok:
                raw[N][key] = "INFEASIBLE"
                print(
                    f"  N={N:6d} ivat INFEASIBLE: predicted peak {ivat_bytes/1e9:.1f} GB > budget {budget/1e9:.1f} GB",
                    flush=True,
                )
                continue
            times = []
            for s in SEEDS:
                idx = np.random.default_rng(s).choice(len(X_all), N, replace=False)
                Xs = X_all[idx]
                t0 = time.perf_counter()
                fn(Xs, LADDER_K, s)
                times.append(time.perf_counter() - t0)
            raw[N][key] = times
            print(f"  N={N:6d} {key:8s} mean={np.mean(times):8.3f}s", flush=True)
    return raw, ivat_ok_map(ram)


def ivat_ok_map(ram):
    budget = MEM_BUDGET_FRAC * ram
    return {N: (N * N * MEM_BYTES_PER_N2 <= budget) for N in LADDER}


def perf_table(raw):
    header = [
        "N",
        "IVATMeans (s)",
        "FCM (s)",
        "k-means (s)",
        "E-means 5ep (s)",
        "E-means 25ep kpp (s)",
        "IVATMeans / FCM",
    ]
    rows = []
    for N in LADDER:
        cells, means = [], {}
        for key, _, _ in PERF_ARMS:
            v = raw[N][key]
            if isinstance(v, str):
                cells.append(v)
                means[key] = None
            else:
                means[key] = float(np.mean(v))
                cells.append(f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}")
        ratio = (
            f"{means['ivat'] / means['fcm']:.2f}×"
            if means.get("ivat") and means.get("fcm")
            else "N/A (infeasible)"
        )
        rows.append([str(N)] + cells + [ratio])
    return header, rows


# --------------------------------------------------------------------------- #
# half 2: quality on the Table 3.5 constructions
# --------------------------------------------------------------------------- #
def run_quality():
    ari = {}  # (dataset, arm_key) -> list over seeds
    dist = {}  # (dataset, arm_key) -> number of distinct labelings across seeds
    scatter = {}  # dataset -> {arm_key: labels at seed 0}, and X, y
    for dname, gen, k in AE.DATASETS:
        X0, y0 = gen(seed=0)
        scatter[dname] = {"X": X0, "y": y0, "k": k}
        for key, _, fn in QUALITY_ARMS:
            vals, labs = [], []
            for s in SEEDS:
                X, y = gen(seed=s)
                lab = fn(X, k, s)
                mask = y >= 0
                vals.append(adjusted_rand_score(y[mask], lab[mask]))
                labs.append(lab)
                if s == 0:
                    scatter[dname][key] = lab
            ari[(dname, key)] = vals
            dist[(dname, key)] = len({np.asarray(l).tobytes() for l in labs})
            print(
                f"  {dname:15s} {key:9s} ARI={np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}  distinct={dist[(dname, key)]}",
                flush=True,
            )
        # determinism on a fixed dataset, IVATMeans across the seed set
        labs_fixed = [fit_ivat(X0, k, s) for s in SEEDS]
        dist[(dname, "ivat_fixed")] = len({l.tobytes() for l in labs_fixed})
    return ari, dist, scatter


def quality_table(ari, dist):
    arm_keys = [k for k, _, _ in QUALITY_ARMS]
    header = ["dataset", "k"] + [name for _, name, _ in QUALITY_ARMS]
    rows = []
    for dname, _, k in AE.DATASETS:
        row = [dname, str(k)]
        for key in arm_keys:
            v = ari[(dname, key)]
            row.append(f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}")
        rows.append(row)
    return header, rows


def determinism_table(dist):
    header = [
        "dataset",
        "IVATMeans distinct labelings (10 seeds, fixed data)",
        "FCM distinct",
        "k-means distinct",
        "E-means 5ep distinct",
        "E-means 25ep kpp distinct",
    ]
    rows = []
    for dname, _, _ in AE.DATASETS:
        rows.append(
            [
                dname,
                str(dist[(dname, "ivat_fixed")]),
                str(dist[(dname, "fcm")]),
                str(dist[(dname, "kmeans")]),
                str(dist[(dname, "em5")]),
                str(dist[(dname, "em25")]),
            ]
        )
    return header, rows


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_runtime(raw, feasible):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, name, _ in PERF_ARMS:
        xs, ys, es = [], [], []
        for N in LADDER:
            v = raw[N][key]
            if isinstance(v, str):
                continue
            xs.append(N)
            ys.append(np.mean(v))
            es.append(np.std(v, ddof=1))
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    infeas = [N for N in LADDER if not feasible[N]]
    for N in infeas:
        ax.axvline(N, color="grey", ls=":", lw=1)
    if infeas:
        ax.text(
            infeas[0] * 0.92,
            0.04,
            f"IVATMeans infeasible\nat N ≥ {infeas[0]:,}\n(predicted peak > 80% RAM)",
            color="grey",
            fontsize=8,
            ha="right",
            va="bottom",
            transform=ax.get_xaxis_transform(),
        )
    ax.set_xlabel("N (shuttle subsample)")
    ax.set_ylabel("wall clock, seconds (mean ± std, 10 seeds)")
    ax.set_title("G9 performance: CPU wall clock, k = 7")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    C.save_figure(fig, "fig_g9_runtime")
    plt.close(fig)


def fig_quality(ari):
    names = [n for _, n, _ in QUALITY_ARMS]
    keys = [k for k, _, _ in QUALITY_ARMS]
    dnames = [d for d, _, _ in AE.DATASETS]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    w = 0.8 / len(keys)
    for i, (key, name) in enumerate(zip(keys, names)):
        mu = [np.mean(ari[(d, key)]) for d in dnames]
        sd = [np.std(ari[(d, key)], ddof=1) for d in dnames]
        ax.bar(
            np.arange(len(dnames)) + (i - len(keys) / 2 + 0.5) * w,
            mu,
            w,
            yerr=sd,
            capsize=2,
            label=name,
        )
    ax.set_xticks(range(len(dnames)))
    ax.set_xticklabels(dnames, rotation=0)
    ax.set_ylabel("ARI (mean ± std, 10 seeds)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "G9 quality: ARI on the Table 3.5 constructions (bridge points excluded)"
    )
    ax.legend(
        fontsize=7,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    C.save_figure(fig, "fig_g9_quality")
    plt.close(fig)


def fig_partitions(scatter, ari):
    dnames = [d for d, _, _ in AE.DATASETS]
    keys = [k for k, _, _ in PERF_ARMS]
    names = {k: n for k, n, _ in PERF_ARMS}
    fig, axes = plt.subplots(
        len(dnames), len(keys), figsize=(2.6 * len(keys), 2.4 * len(dnames))
    )
    for r, d in enumerate(dnames):
        X, y = scatter[d]["X"], scatter[d]["y"]
        for c, key in enumerate(keys):
            ax = axes[r, c]
            lab = scatter[d][key]
            ax.scatter(X[:, 0], X[:, 1], c=lab, cmap="tab10", s=3, linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            m = np.asarray(ari[(d, key)])
            ax.set_title(f"ARI {m[0]:.2f} (seed 0)", fontsize=7)
            if r == 0:
                ax.set_xlabel(names[key], fontsize=7)
                ax.xaxis.set_label_position("top")
            if c == 0:
                ax.set_ylabel(d, fontsize=8)
    fig.suptitle("G9 partitions, seed 0: colour = predicted cluster", fontsize=10)
    fig.tight_layout()
    C.save_figure(fig, "fig_g9_partitions")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# verdicts and the report
# --------------------------------------------------------------------------- #
def verdicts(raw, ari, dist, feasible):
    out = {}
    # P1
    ratios = []
    for N in LADDER:
        if feasible[N]:
            ratios.append((N, np.mean(raw[N]["ivat"]) / np.mean(raw[N]["fcm"])))
    if ratios:
        worst = max(r for _, r in ratios)
        ok = worst <= 10
        out["P1"] = (
            f"{'PASS' if ok else 'FAIL'}: worst feasible IVATMeans/FCM ratio {worst:.2f}× "
            f"over N = {', '.join(str(n) for n, _ in ratios)}"
        )
    else:
        out["P1"] = "NOT EVALUATED: no feasible rung"
    # P2
    m_moons = np.mean(ari[("two_moons", "ivat")])
    m_circ = np.mean(ari[("circles", "ivat")])
    k_moons = np.mean(ari[("two_moons", "kmeans")])
    k_circ = np.mean(ari[("circles", "kmeans")])
    refuted = m_moons >= 0.99 or m_circ >= 0.99
    r_moons = np.mean(ari[("two_moons", "ivat_rel")])
    r_circ = np.mean(ari[("circles", "ivat_rel")])
    out["P2"] = (
        f"{'REFUTED' if refuted else 'NOT REFUTED'} for the default medoid refine: IVATMeans ARI two_moons "
        f"{m_moons:.3f}, circles {m_circ:.3f} (predicted near k-means row 0.27 / 0.00; our k-means arm "
        f"{k_moons:.3f} / {k_circ:.3f}). The relational refine reads {r_moons:.3f} / {r_circ:.3f}; "
        f"a relational ARI >= 0.99 is a separate finding about §5.2, reported here, not a P2 verdict"
    )
    # P3
    b_ivat = np.mean(ari[("easy_blobs", "ivat")])
    b_best = max(
        np.mean(ari[("easy_blobs", "fcm")]), np.mean(ari[("easy_blobs", "kmeans")])
    )
    gap = b_best - b_ivat
    out["P3"] = (
        f"{'PASS' if gap <= 0.05 else 'FAIL'}: easy_blobs IVATMeans {b_ivat:.3f} vs best baseline {b_best:.3f} "
        f"(gap {gap:+.3f}, threshold 0.05)"
    )
    # P4
    per = {d: dist[(d, "ivat_fixed")] for d, _, _ in AE.DATASETS}
    out["P4"] = (
        f"{'PASS' if all(v == 1 for v in per.values()) else 'FAIL'}: IVATMeans distinct labelings across "
        f"10 seeds per dataset = {per}"
    )
    # P5
    base = {
        (d, a): dist[(d, a)]
        for d, _, _ in AE.DATASETS
        for a in ("fcm", "kmeans", "em5", "em25")
    }
    varied = any(v > 1 for v in base.values())
    out["P5"] = (
        f"{'PASS' if varied else 'FAIL'}: baselines vary across seeds on "
        f"{sum(v > 1 for v in base.values())}/{len(base)} dataset×arm cells"
    )
    return out


def write_report(
    header_perf, rows_perf, header_q, rows_q, header_d, rows_d, v, feasible
):
    def md(h, rs):
        s = "| " + " | ".join(h) + " |\n|" + "---|" * len(h) + "\n"
        for r in rs:
            s += "| " + " | ".join(r) + " |\n"
        return s

    infeas = [str(N) for N in LADDER if not feasible[N]]
    text = f"""# G9 comparison report: IVATMeans vs FCM vs k-means vs E-means

Generated by `reproduce/tables/table_g9_ivatmeans_comparison.py`. Ten seeds (`common.SEEDS`).
Pre-registered predictions are in the generator's docstring and are evaluated below.

## Verdicts

| | Outcome |
|---|---|
| P1 performance | {v['P1']} |
| P2 predicted loss (two moons, circles) | {v['P2']} |
| P3 blob set | {v['P3']} |
| P4 determinism | {v['P4']} |
| P5 manipulation check | {v['P5']} |

## 1. Performance at scale (CPU wall clock, shuttle, k = 7)

{md(header_perf, rows_perf)}
Infeasible rungs for IVATMeans (predicted peak above 80% of RAM; 32,000 was OOM-killed in a dry run): {', '.join(infeas) if infeas else 'none'}.
Only CPU rows are reported. The GPU and whole-pipeline-on-device path (`on_device`) need a CUDA host, which this one is not.

## 2. Quality of construction (ARI, mean ± std over 10 seeds)

{md(header_q, rows_q)}
Bridge points (label -1) are excluded from ARI for `bridged`, as in Table 3.5.

## 3. Determinism asymmetry and manipulation check

{md(header_d, rows_d)}
IVATMeans should show 1 distinct labelling on both columns. Baselines should show more than 1 wherever their restarts actually move the partition.

## Figures

- `figures/fig_g9_runtime.{{png,eps}}`: wall clock vs N, log-log, with infeasible rungs marked.
- `figures/fig_g9_quality.{{png,eps}}`: ARI per construction and arm, with std error bars.
- `figures/fig_g9_partitions.{{png,eps}}`: predicted partitions, seed 0, per construction and arm.

## Scope and caveats

- CPU only. No GPU rows.
- The ladder uses one dataset (shuttle, z-scored). Its k = 7 is the class count, not an intrinsic one.
- The quality sets are synthetic, 1,500 points each, from `ClusteringExperiments/adversarial_eval.py`.
- E-means is a k-means replacement. It is not an FCM or IVATMeans replacement, so its arms are the k-means family.
- E-means is evaluated at its paper's default (5 epochs, its own init) and at 25 epochs with k-means++ init. The paper's headline claim is at 25 epochs against Lloyd.
- Versions: faiss {faiss.__version__}, tribble-cluster {tribble_version()}.
"""
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as fh:
        fh.write(text)
    return REPORT


def tribble_version():
    try:
        import importlib.metadata as md

        return md.version("tribble-clustering")
    except Exception:
        return "unknown (not installed as a distribution)"


def main():
    print("G9: IVATMeans vs FCM vs k-means vs E-means", flush=True)
    ram = _phys_ram_bytes()
    feasible = ivat_ok_map(ram)
    print(
        f"RAM {ram/1e9:.1f} GB, IVATMeans budget {MEM_BUDGET_FRAC*ram/1e9:.1f} GB",
        flush=True,
    )

    if SKIP_LADDER:
        raw = {}
        feasible = {N: False for N in LADDER}
        header_perf, rows_perf = ["N"], [["ladder skipped (REPRO_G9_SKIP_LADDER=1)"]]
    else:
        print("Half 1: performance ladder", flush=True)
        raw, _ = run_ladder(shuttle_z())
        header_perf, rows_perf = perf_table(raw)
        fig_runtime(raw, feasible)

    print("Half 2: quality on Table 3.5 constructions", flush=True)
    ari, dist, scatter = run_quality()
    header_q, rows_q = quality_table(ari, dist)
    header_d, rows_d = determinism_table(dist)
    fig_quality(ari)
    fig_partitions(scatter, ari)

    v = verdicts(raw, ari, dist, feasible)
    import json

    raw_path = os.path.join(C.OUTPUT_DIR, "g9_raw.json")
    with open(raw_path, "w") as fh:
        json.dump(
            {
                "ladder": {
                    str(N): {
                        k: (v_ if isinstance(v_, str) else [float(x) for x in v_])
                        for k, v_ in d.items()
                    }
                    for N, d in raw.items()
                },
                "quality_ari": {
                    f"{d}|{a}": [float(x) for x in vals] for (d, a), vals in ari.items()
                },
                "distinct_labelings": {
                    f"{d}|{a}": int(n) for (d, a), n in dist.items()
                },
            },
            fh,
            indent=1,
        )
    print("raw:", raw_path, flush=True)

    C.emit(
        "table_g9_performance",
        "G9 performance: CPU wall clock on the shuttle N ladder, k = 7 (mean ± std, 10 seeds)",
        header_perf,
        rows_perf,
        note="CPU only. Infeasible IVATMeans rungs are marked, not dropped.",
        seeds=SEEDS,
    )
    C.emit(
        "table_g9_quality",
        "G9 quality: ARI on the six Table 3.5 constructions (mean ± std, 10 seeds)",
        header_q,
        rows_q,
        note="Bridge points excluded from ARI. Determinism and manipulation check in table_g9_determinism.",
        seeds=SEEDS,
    )
    C.emit(
        "table_g9_determinism",
        "G9 determinism and manipulation check: distinct labelings across 10 seeds",
        header_d,
        rows_d,
        note="IVATMeans should read 1. Baselines should read >1 where their restarts move the partition.",
        seeds=SEEDS,
    )
    path = write_report(
        header_perf, rows_perf, header_q, rows_q, header_d, rows_d, v, feasible
    )
    print("\n".join(f"{k}: {val}" for k, val in v.items()), flush=True)
    print("report:", path, flush=True)


if __name__ == "__main__":
    main()
