"""Animate the VAT -> iVAT pipeline used by tribble-cluster (mergeVAT).

Phase 1 (VAT):  Prim's MST from an endpoint of the max-distance pair. Each point
                added to the tree is the next row/column of the reordered matrix.
Phase 2 (iVAT): row r of the iVAT image is filled from its argmin parent j in
                columns [0, r):  D'[r, c] = max(D*[r, j], D'[j, c])  -- O(N^2).

The animation is built from its own step-by-step Prim and path-max recurrence;
`verify` asserts the end state equals tribbleclustering.compute_ivat, so the
frames cannot drift from the library.

    uv run --project tribble-cluster --with matplotlib --with pillow \
        python presentations/animations/vat_ivat_animation.py [--n 45] [--out PATH]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from tribbleclustering import compute_ivat

CMAP = "gray"
PALETTE = ["#1f77b4", "#d95f02", "#1b9e77"]


def make_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [4.0, 0.5], [1.8, 3.8]])
    labels = np.arange(n) % 3
    x = centers[labels] + rng.normal(scale=0.55, size=(n, 2))
    perm = rng.permutation(n)  # shuffle so the raw matrix shows no structure
    return x[perm], labels[perm]


def prim_order(d: np.ndarray) -> list[int]:
    """Dense Prim, started from an endpoint of the max-distance pair (as pvat.py)."""
    n = len(d)
    start = int(np.argmax(d)) // n
    order = [start]
    in_tree = np.zeros(n, bool)
    in_tree[start] = True
    key = d[start].copy()
    for _ in range(n - 1):
        k = np.where(in_tree, np.inf, key)
        u = int(np.argmin(k))
        order.append(u)
        in_tree[u] = True
        key = np.minimum(key, d[u])
    return order


def ivat_rows(ds: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Path-max recurrence, returning the image and each row's argmin parent."""
    n = len(ds)
    out = np.zeros_like(ds)
    parent = [0] * n
    for r in range(1, n):
        j = int(np.argmin(ds[r, :r]))
        parent[r] = j
        out[r, j] = out[j, r] = ds[r, j]
        for c in range(r):
            if c != j:
                out[r, c] = out[c, r] = max(ds[r, j], out[j, c])
    return out, parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "out" / "vat_ivat.gif"
    )
    a = ap.parse_args()

    x, lab = make_data(a.n, a.seed)
    n = a.n
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    order = prim_order(d)
    ds = d[np.ix_(order, order)]
    div, parent = ivat_rows(ds)

    # Verify against the library (the numbers the proposal cites come from it).
    lib_ivat, _, lib_p = compute_ivat(d.copy())
    assert np.array_equal(np.asarray(lib_p), np.asarray(order)), "VAT order differs"
    assert np.allclose(lib_ivat, div), "iVAT image differs"
    print("verified: animation order and iVAT image match compute_ivat")

    hold = 8
    frames = (
        [("raw", 0)] * hold
        + [("vat", k) for k in range(1, n + 1)]
        + [("vat", n)] * hold
        + [("ivat", r) for r in range(1, n)]
        + [("ivat", n - 1)] * (2 * hold)
    )

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.5), dpi=90)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.08, wspace=0.12)
    vmax = d.max()
    blank = np.full((n, n), np.nan)
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("#f2e9d8")

    def draw(i: int) -> None:
        phase, k = frames[i]
        for axis in ax:
            axis.clear()
        a0, a1, a2 = ax

        # --- scatter -------------------------------------------------------
        a0.set_title("data + Prim MST", fontsize=11)
        a0.set_aspect("equal")
        a0.set_xticks([])
        a0.set_yticks([])
        in_tree = order[:k] if phase != "raw" else []
        if phase == "ivat":
            in_tree = order
        pos = {p: i2 for i2, p in enumerate(order)}
        for i2 in range(1, len(in_tree)):
            j = parent_ord[i2]
            a0.plot(*x[[order[i2], order[j]]].T, color="0.35", lw=1.2, zorder=1)
        a0.scatter(*x.T, s=34, c="0.8", zorder=2)
        if len(in_tree):
            a0.scatter(
                *x[in_tree].T, s=34, c=[PALETTE[l] for l in lab[in_tree]], zorder=3
            )
        if phase == "vat" and k <= n:
            a0.scatter(
                *x[order[k - 1]][None].T,
                s=140,
                facecolor="none",
                edgecolor="crimson",
                lw=2,
                zorder=4,
            )
        if phase == "ivat":
            a0.scatter(
                *x[order[k]][None].T,
                s=140,
                facecolor="none",
                edgecolor="crimson",
                lw=2,
                zorder=4,
            )
            a0.scatter(
                *x[order[parent[k]]][None].T,
                s=140,
                facecolor="none",
                edgecolor="royalblue",
                lw=2,
                zorder=4,
            )

        # --- VAT image -----------------------------------------------------
        if phase == "raw":
            a1.set_title("dissimilarity D (input order)", fontsize=11)
            a1.imshow(d, cmap=cmap, vmin=0, vmax=vmax)
        else:
            m = int(k) if phase == "vat" else n
            shown = blank.copy()
            shown[:m, :m] = ds[:m, :m]
            a1.set_title(f"VAT: D reordered by Prim ({m}/{n})", fontsize=11)
            a1.imshow(shown, cmap=cmap, vmin=0, vmax=vmax)

        # --- iVAT image ----------------------------------------------------
        a2.set_title("iVAT: minimax path distance", fontsize=11)
        shown = blank.copy()
        if phase == "ivat":
            shown[: k + 1, : k + 1] = div[: k + 1, : k + 1]
            a2.imshow(shown, cmap=cmap, vmin=0, vmax=div.max())
            a2.axhline(k, color="crimson", lw=1.2)
            a2.plot([parent[k]], [k], "s", color="royalblue", ms=5)
            a2.set_xlabel(
                f"row {k}: parent = col {parent[k]}, then max(D*[r,j], D'[j,c])",
                fontsize=9,
            )
        else:
            a2.imshow(shown, cmap=cmap, vmin=0, vmax=vmax)
        for axis in (a1, a2):
            axis.set_xticks([])
            axis.set_yticks([])
        fig.suptitle(
            {
                "raw": "1. Input: no visible structure",
                "vat": "2. VAT: add the nearest point to the tree; it becomes the next row/column",
                "ivat": "3. iVAT: fill each row from its parent's row (O(N²) recurrence)",
            }[phase],
            fontsize=12,
        )

    # parent in *ordered* index space, used for MST edges in the scatter
    parent_ord = [0] + [int(np.argmin(ds[r, :r])) for r in range(1, n)]

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / a.fps)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(a.out, writer=PillowWriter(fps=a.fps))
    print(f"wrote {a.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
