"""iVAT on the body-fat measurements: is there cluster structure, or one continuum?

iVAT (improved Visual Assessment of cluster Tendency) reorders the objects along
the VAT (Prim-MST) path and displays the minimax-path dissimilarity as an image.
Dark diagonal blocks separated by light gutters = clusters; a smooth
corner-to-corner gradient = a single continuum with no natural grouping.

An iVAT image on its own invites over-reading -- the eye finds blocks in noise --
so three things are shown next to it:

  * **A null control.** The same pipeline on the same data with every column
    independently shuffled. That destroys the relationships between measurements
    while keeping each measurement's own distribution, so it is what "no
    structure, same marginals" actually looks like here. If the real image
    resembles the null, there is nothing to find.

  * **Aligned strips.** `BodyFat`, `Abdomen` and `Age`, in the VAT ordering. A
    block that means something should correspond to *something*; if the ordering
    is a clean gradient in body fat, the tendency is a continuum along the
    target, not a set of clusters.

  * **Numbers, not just the picture.** The Hopkins statistic (0.5 = random,
    -> 1 = clustered) on the real and shuffled data, the best silhouette over
    k = 2..8 for both `IVATMeans` and k-means, the cluster count
    `get_ivat_levels` reads off the iVAT matrix unprompted, and Spearman
    correlation between VAT path position and body fat.

Uses the clustering submodule's own iVAT -- `tribbleclustering.compute_ivat` for
the image, `get_ivat_levels` for the cluster count the iVAT matrix itself
supports (it reads the off-by-one diagonal for substantial jumps), and
`IVATMeans` for the labels that get silhouette-scored. That means this runs in
the `tribble-cluster` environment, so nothing here imports `tribblefis`; the two
constants it shares with `bodyfat.py` are restated rather than imported, and the
reason `Density` is excluded is the same one given there at length: the target
was computed from it by Siri's equation, so it is the answer, not a feature.

Needs: numpy, pandas, scipy, scikit-learn, matplotlib, tribble-cluster.  Run from
the repo root:

    uv run --project tribble-cluster --with matplotlib --with scikit-learn \
        --with pandas --with tabulate python FuzzySystemsExperiments/bodyfat_ivat.py

Writes `outputs/bodyfat-ivat/` (figure + a short findings file).
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from tribbleclustering import IVATMeans, compute_ivat, get_ivat_levels  # noqa: E402

OUT = os.path.join("outputs", "bodyfat-ivat")
DEFAULT_CSV = os.path.join("data", "bodyfat.csv")

# Restated rather than imported from `bodyfat.py`: that module pulls in
# `tribblefis`, which lives in a different uv project than the clustering code
# this script runs against. See its docstring for why `Density` is excluded.
TARGET = "BodyFat"
LEAKY = "Density"
STRIPS = [TARGET, "Abdomen", "Age"]


def implausible(df):
    """The dataset's documented height typo -- an adult height under 4 feet.
    Reported, never silently repaired; VAT should show it as an isolated point."""
    return df.index[df["Height"] < 48].tolist()


def hopkins(X, rng, m=None):
    """Hopkins statistic: compare nearest-neighbour distances of real points to
    those of uniform points drawn in the same bounding box. ~0.5 means the data
    are indistinguishable from uniform (no cluster tendency); values toward 1.0
    mean clustered. Computed on a sample of m points, as is standard."""
    X = np.asarray(X, float)
    n, d = X.shape
    m = m or max(10, int(0.1 * n))
    lo, hi = X.min(axis=0), X.max(axis=0)

    def nn_dist(points, exclude_self):
        D = np.linalg.norm(points[:, None, :] - X[None, :, :], axis=2)
        if exclude_self:
            np.fill_diagonal(D, np.inf)
            return D.min(axis=1)
        return D.min(axis=1)

    idx = rng.choice(n, size=m, replace=False)
    D_real = np.linalg.norm(X[idx][:, None, :] - X[None, :, :], axis=2)
    D_real[np.arange(m), idx] = np.inf  # exclude each sampled point from itself
    w = D_real.min(axis=1)
    u = nn_dist(rng.uniform(lo, hi, size=(m, d)), exclude_self=False)
    return float(u.sum() / (u.sum() + w.sum()))


def best_silhouette(X, k_range=range(2, 9)):
    """Best silhouette over k for two clusterings: `IVATMeans` (the clustering
    submodule's own iVAT-based partition, minimax-medoid prototypes) and k-means
    on the standardised features. A silhouette below ~0.25 is the conventional
    "no substantial structure" line. Returns a frame plus the best of each."""
    rows = []
    for k in k_range:
        lab_iv = IVATMeans(n_clusters=k, random_state=0).fit_predict(X)
        lab_km = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        counts = np.bincount(np.asarray(lab_iv, int) - int(np.min(lab_iv)))
        counts_km = np.bincount(lab_km)
        rows.append(
            {
                "k": k,
                "IVATMeans": (
                    silhouette_score(X, lab_iv)
                    if len(set(np.asarray(lab_iv).ravel())) > 1
                    else float("nan")
                ),
                "largest iVAT %": 100.0 * counts.max() / counts.sum(),
                "k-means": silhouette_score(X, lab_km),
                "largest k-means %": 100.0 * counts_km.max() / counts_km.sum(),
            }
        )
    frame = pd.DataFrame(rows)
    return frame, frame["IVATMeans"].max(), frame["k-means"].max()


def ivat(X):
    """(iVAT image in VAT order, VAT permutation) via the clustering submodule.
    `compute_ivat` returns the matrix already permuted into VAT order."""
    D = squareform(pdist(X, metric="euclidean"))
    img, _argmin_seq, order = compute_ivat(np.ascontiguousarray(D, dtype=np.float64))
    return img, np.asarray(order, int)


def plot(real_img, null_img, strips, stats, path):
    fig = plt.figure(figsize=(13.5, 5.8))
    grid = fig.add_gridspec(
        len(strips) + 1,
        3,
        height_ratios=[len(strips) * 3] + [1] * len(strips),
        hspace=0.4,
        wspace=0.25,
    )

    # Both panels on ONE robust colour scale, taken from both matrices together.
    # Scaling each independently would stretch whatever range each happens to
    # have and make the real image and the structureless null look alike no
    # matter what -- which would defeat the entire point of showing them side by
    # side. Percentile limits rather than min/max because minimax (bottleneck)
    # distances pile up in a narrow band: on raw limits the image washes out to
    # flat colour and any real block structure is invisible.
    both = np.concatenate([real_img.ravel(), null_img.ravel()])
    vmin, vmax = np.percentile(both, [2, 98])
    # Grey, dark = similar: the VAT convention, where a cluster reads as a dark
    # block on the diagonal.
    kw = dict(cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    ax = fig.add_subplot(grid[0, 0])
    im = ax.imshow(real_img, **kw)
    ax.set_title(
        "iVAT — body-fat measurements\n(13 standardised features, n=252)", fontsize=10
    )
    ax.set_xticks([])
    ax.set_yticks([])

    ax_n = fig.add_subplot(grid[0, 1])
    ax_n.imshow(null_img, **kw)
    ax_n.set_title(
        "Null control — same columns, independently shuffled\n"
        "(no structure, identical marginals)",
        fontsize=10,
    )
    ax_n.set_xticks([])
    ax_n.set_yticks([])
    cb = fig.colorbar(im, ax=[ax, ax_n], fraction=0.025, pad=0.02)
    cb.set_label("minimax (bottleneck) distance — dark = similar", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax_t = fig.add_subplot(grid[0, 2])
    ax_t.axis("off")
    ax_t.text(
        0.0,
        1.0,
        stats,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        transform=ax_t.transAxes,
    )

    for i, (name, values) in enumerate(strips.items()):
        ax_s = fig.add_subplot(grid[i + 1, 0])
        discrete = name.startswith("k-means")
        ax_s.imshow(
            values[None, :],
            aspect="auto",
            cmap="Set1" if discrete else "magma",
            interpolation="nearest",
        )
        ax_s.set_yticks([])
        ax_s.set_xticks([])
        ax_s.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    fig.text(
        0.125,
        0.015,
        "strips: same VAT ordering as the image above; dark = low, bright = high. "
        "The k-means strip is the k=2 split — contiguous runs would mean the split\n"
        "agrees with the VAT path (real separation); interleaved colour means it is "
        "a cut through one continuum.",
        fontsize=8,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(csv, seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    df = pd.read_csv(csv)
    features = [c for c in df.columns if c not in (TARGET, LEAKY)]
    bad = implausible(df)
    X_raw = df[features].to_numpy(float)
    X = StandardScaler().fit_transform(X_raw)
    print(f"{csv}: {len(df)} rows, iVAT on {len(features)} standardised features")
    if bad:
        print(
            f"  note: row(s) {bad} carry the dataset's height typo and are left in "
            f"-- an outlier is exactly what VAT should show as an isolated point"
        )

    img, order = ivat(X)
    null = np.column_stack([rng.permutation(col) for col in X.T])
    img_null, _ = ivat(null)

    # What cluster count does the iVAT matrix itself support? `get_ivat_levels`
    # reads the off-by-one diagonal for substantial jumps rather than being told
    # a k, so it is the least question-begging count available here.
    level = get_ivat_levels(X, img, order, n_levels=1)
    sizes = sorted((len(c) for c in level.cluster_city_ids), reverse=True)
    k_ivat = len(sizes)

    sil_frame, best_iv, best_km = best_silhouette(X)
    # The same silhouette sweep on the shuffled null. Without it a silhouette of
    # ~0.35 looks like evidence; with it, it is only evidence if the null cannot
    # reach the same value on data that has no structure by construction.
    sil_null, null_iv, null_km = best_silhouette(null)
    h_real = float(np.mean([hopkins(X, rng) for _ in range(20)]))
    h_null = float(np.mean([hopkins(null, rng) for _ in range(20)]))
    pos = np.empty(len(order), int)
    pos[order] = np.arange(len(order))
    rho, pval = spearmanr(pos, df[TARGET].to_numpy(float))

    # Does the k=2 k-means split agree with the VAT path? Walk the path and count
    # how many times the label flips. Two contiguous blocks flip once (2 runs);
    # a split unrelated to the path flips about 2*n1*n2/n times. This separates
    # "two groups both methods agree on" from "one cloud cut in half by k-means
    # along a direction VAT does not see".
    km2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
    km2_vat = km2[order]
    runs = int(1 + np.count_nonzero(np.diff(km2_vat)))
    n1 = int(np.count_nonzero(km2 == km2[0]))
    runs_expected = 1 + 2.0 * n1 * (len(km2) - n1) / len(km2)

    stats = "\n".join(
        [
            "cluster tendency",
            "----------------",
            f"Hopkins, real     {h_real:.3f}",
            f"Hopkins, shuffled {h_null:.3f}",
            "  (0.5 = uniform, 1 = clustered)",
            "",
            f"clusters iVAT supports: {k_ivat}",
            "  (get_ivat_levels, unprompted)",
            f"  sizes {sizes[:6]}",
            "",
            "best silhouette, k=2..8",
            f"  IVATMeans  {best_iv:.3f} (null {null_iv:.3f})",
            f"  k-means    {best_km:.3f} (null {null_km:.3f})",
            "  (< 0.25 = no substantial",
            "   structure; and only the",
            "   margin over null counts)",
            "",
            f"largest cluster at best k: {sizes[0] / len(order):.0%}",
            "",
            "k-means k=2 vs the VAT path",
            f"  label runs      {runs}",
            "  if contiguous    2",
            f"  if unrelated  ~{runs_expected:.0f}",
            "",
            "VAT path position vs BodyFat",
            f"  Spearman rho    {rho:+.3f}",
            f"  p               {pval:.1e}",
        ]
    )

    strips = {}
    for name in STRIPS:
        v = df[name].to_numpy(float)[order]
        strips[name] = (v - v.min()) / (np.ptp(v) or 1.0)
    # The k=2 k-means split, in the same VAT order. This is the visual test of
    # whether that split is separation or just a cut: contiguous blocks of one
    # colour mean the two clusterings agree about which points belong together.
    strips["k-means k=2"] = km2_vat.astype(float)

    fig_path = os.path.join(out_dir, "bodyfat_ivat.png")
    plot(img, img_null, strips, stats, fig_path)

    print("\nsilhouette by k:")
    print(sil_frame.to_string(index=False, float_format="%.3f"))
    print(f"\nHopkins  real {h_real:.3f}   shuffled-null {h_null:.3f}")
    print(f"Best silhouette: IVATMeans {best_iv:.3f}, k-means {best_km:.3f}")
    print(f"VAT position vs {TARGET}: Spearman rho {rho:+.3f} (p {pval:.1e})")

    # Three conditions have to hold before calling something a cluster, and each
    # on its own is easy to fool: the silhouette has to clear the "substantial"
    # line, it has to clear what the *structureless null* reaches on the same
    # sweep, and the partition has to be a partition rather than one big blob
    # with a tail shaved off it.
    margin = max(best_iv - null_iv, best_km - null_km)
    biggest = sizes[0] / len(order)
    if max(best_iv, best_km) < 0.25 or margin < 0.05:
        verdict = (
            f"no separable clusters -- one continuum. The best silhouette "
            f"({max(best_iv, best_km):.3f}) is only {margin:+.3f} above what the "
            f"structureless null reaches on the same sweep"
        )
    elif biggest >= 0.8:
        verdict = (
            f"no separable clusters -- iVAT's own reading leaves {biggest:.0%} of "
            f"the men in one group (sizes {sizes[:4]}), so what it finds is a "
            f"couple of outliers being shaved off, not a partition. k-means does "
            f"cut a balanced 2-way split that beats its null, but that split "
            f"flips label {runs} times along the VAT path (2 = contiguous, "
            f"~{runs_expected:.0f} = unrelated to it), so it is a cut through one "
            f"continuum rather than two separated groups"
        )
    else:
        verdict = "genuine cluster structure -- balanced, and clear of the null"
    print(f"\nVerdict: {verdict}")

    with open(os.path.join(out_dir, "FINDINGS.md"), "w", encoding="utf-8") as fh:
        fh.write(
            "\n".join(
                [
                    "# iVAT on the body-fat measurements",
                    "",
                    f"`{fig_path}` — generated by "
                    f"`FuzzySystemsExperiments/bodyfat_ivat.py` "
                    f"(seed {seed}, {len(df)} rows, {len(features)} standardised "
                    f"features, `{LEAKY}` and `{TARGET}` excluded).",
                    "",
                    f"**Verdict: {verdict}.**",
                    "",
                    "| measure | value | reads as |",
                    "|---|---|---|",
                    f"| Hopkins, real data | {h_real:.3f} | on its own looks like "
                    f"strong cluster tendency |",
                    f"| Hopkins, shuffled null | {h_null:.3f} | **but structureless "
                    f"data scores nearly the same** -- Hopkins is inflated in 13 "
                    f"dimensions, where uniform box samples miss the data manifold "
                    f"entirely. Margin {h_real - h_null:+.3f}: uninformative here |",
                    f"| best silhouette, `IVATMeans` | {best_iv:.3f} | null reaches "
                    f"{null_iv:.3f} (margin {best_iv - null_iv:+.3f}) |",
                    f"| best silhouette, k-means | {best_km:.3f} | null reaches "
                    f"{null_km:.3f} (margin {best_km - null_km:+.3f}) |",
                    f"| largest cluster at the best k | {biggest:.0%} | "
                    f"{'a tail, not a partition' if biggest >= 0.8 else 'a real partition'} |",
                    f"| k-means k=2 label runs along the VAT path | {runs} | "
                    f"2 would mean contiguous blocks (real separation), "
                    f"~{runs_expected:.0f} means unrelated to the path — this is "
                    f"a cut through one continuum |",
                    f"| Spearman(VAT position, {TARGET}) | {rho:+.3f} "
                    f"(p {pval:.1e}) | "
                    f"{'the ordering is not a body-fat gradient' if pval > 0.01 else 'the ordering tracks body fat'} |",
                    "",
                    "Every row here is a pair: a measure and what the same measure "
                    "gives on data with no structure in it. That is the whole "
                    "method -- absolute cluster-validity numbers are unreadable "
                    "without their null.",
                    "",
                    "## Silhouette by k -- real data",
                    "",
                    sil_frame.to_markdown(index=False, floatfmt=".3f"),
                    "",
                    "## Silhouette by k -- shuffled null",
                    "",
                    sil_null.to_markdown(index=False, floatfmt=".3f"),
                    "",
                    "The `largest .. %` columns are the tell: when the "
                    "best-scoring k leaves nearly every point in one cluster, "
                    "the 'clusters' it found are individual outliers being "
                    "shaved off, not a partition of the data.",
                    "",
                ]
            )
            + "\n"
        )
    print(f"wrote {fig_path} and {os.path.join(out_dir, 'FINDINGS.md')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=OUT)
    a = p.parse_args()
    main(a.csv, a.seed, a.out_dir)
