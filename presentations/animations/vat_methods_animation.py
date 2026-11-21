"""Animate the efficient VAT ordering and memory methods of tribble-cluster.

Two GIFs, each checked against the library before it is written:

  ordering  Classical linear scan vs. priority queue (pvat.vat_prim_mst) vs.
            compact active set (pcvat.pyx kernel), on the same data. Each panel
            shows the cells it examines and its own data structure; the bottom
            panel is the running count of primitive operations. The three
            orderings are asserted identical to each other and to compute_vat.
  memory    Matrix-free reorder (one row computed on demand), the in-place
            cycle-walk permutation (checked against _permute_sym_inplace), and
            the k*s*N^2 footprint arithmetic of Table 3.3.

Op counts are what these implementations do on this small instance, not the
asymptotic claims: classical counts (placed x unplaced) pair comparisons per
round; heap counts the relax comparisons plus heap pushes/pops weighted by
log2(heap size); active set counts slot visits. The one-off O(N^2) global-max
scan that seeds all three is excluded.

    PYTHONPATH=tribble-cluster/src tribble-cluster/.venv/Scripts/python.exe \
        presentations/animations/vat_methods_animation.py [--which all]
"""

import argparse
import heapq
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from tribbleclustering import compute_ivat, compute_vat
from tribbleclustering.pvat import _permute_sym_inplace, _vat_prim_mst_seq_python

from vat_ivat_animation import ivat_rows, make_data, prim_order

OUT = Path(__file__).parent / "out"
CMAP = plt.get_cmap("gray").copy()
RED, BLUE, GREY = "#d62728", "#1f77b4", "#bbbbbb"


# --------------------------------------------------------------------------
# Simulations — one snapshot per round, mirroring the real implementations
# --------------------------------------------------------------------------
def sim_classical(d: np.ndarray, start: int):
    n = len(d)
    order, snaps, ops = [start], [{"u": start, "block": None, "ops": 0}], [0]
    while len(order) < n:
        placed = np.array(order)
        un = np.setdiff1d(np.arange(n), placed)
        sub = d[np.ix_(placed, un)]
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        u = int(un[j])
        snaps.append(
            {"u": u, "block": (placed.copy(), un.copy()), "cell": (int(placed[i]), u)}
        )
        ops.append(len(placed) * len(un))
        order.append(u)
    return order, snaps, np.cumsum(ops)


def sim_heap(d: np.ndarray, start: int):
    """Mirror of pvat.vat_prim_mst: push on every improved key, skip stale pops."""
    n = len(d)
    key = np.full(n, np.inf)
    in_mst = np.zeros(n, bool)
    src_key = d.max()
    pq = [(src_key, start)]
    key[start] = src_key
    order, snaps, ops = [], [], []
    while len(order) < n:
        cost, stale, pops = 0.0, 0, 0
        while True:
            cost += np.log2(len(pq) + 1)
            w, u = heapq.heappop(pq)
            pops += 1
            if in_mst[u]:
                stale += 1
                continue
            break
        in_mst[u] = True
        order.append(u)
        verts = np.arange(n)
        mask = (verts != u) & ~in_mst & (key > d[u])
        key[mask] = d[u][mask]
        for v in verts[mask]:
            cost += np.log2(len(pq) + 2)
            heapq.heappush(pq, (key[v], int(v)))
        cost += n  # the relax comparisons (full-width mask in the real code)
        snaps.append(
            {
                "u": u,
                "pushed": verts[mask].copy(),
                "stale": stale,
                "pq": sorted(pq),
                "in_mst": in_mst.copy(),
            }
        )
        ops.append(cost)
    return order, snaps, np.cumsum(ops)


def sim_active(d: np.ndarray, start: int):
    """Mirror of the pcvat.pyx compact-active-set kernel."""
    n = len(d)
    av = np.arange(n)
    ak = np.full(n, np.inf)
    ak[start] = d.max()
    bk, m = start, n
    order, snaps, ops = [], [], []
    for rnd in range(n):
        u = int(av[bk])
        order.append(u)
        m -= 1
        av[bk], ak[bk] = av[m], ak[m]
        changed = np.zeros(n, bool)
        best, nbk = np.inf, -1
        for i in range(m):
            dd = d[u, av[i]]
            if dd < ak[i]:
                ak[i] = dd
                changed[i] = True
            if ak[i] < best:
                best, nbk = ak[i], i
        bk = nbk
        snaps.append(
            {
                "u": u,
                "m": m,
                "av": av.copy(),
                "ak": ak.copy(),
                "changed": changed,
                "bk": bk,
            }
        )
        ops.append(m)
    return order, snaps, np.cumsum(ops)


# --------------------------------------------------------------------------
# GIF 1 — ordering race
# --------------------------------------------------------------------------
def overlay(n: int, cells: list[tuple[np.ndarray, np.ndarray, tuple, float]]):
    rgba = np.zeros((n, n, 4))
    for rows, cols, color, alpha in cells:
        rgba[np.ix_(rows, cols)] = (*color, alpha)
    return rgba


def build_ordering(n: int, seed: int, fps: int) -> None:
    x, _ = make_data(n, seed)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    start = int(np.argmax(d)) // n

    oc, sc, cc = sim_classical(d, start)
    oh, sh, ch = sim_heap(d, start)
    oa, sa, ca = sim_active(d, start)
    lib = compute_vat(d.copy())[1]
    assert oc == oh == oa == [int(v) for v in lib], "orderings differ"
    print(
        "verified: classical, heap and active-set orderings are identical to compute_vat"
    )

    order = oc
    ds_ = d[np.ix_(order, order)]
    div, par = ivat_rows(ds_)
    assert np.allclose(compute_ivat(d.copy())[0], div), "iVAT image differs"
    print("verified: iVAT recurrence image matches compute_ivat")
    n_ord = n + 6  # ordering frames, including a short hold
    nfr = n_ord + (n - 1) + 8
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    gs = GridSpec(
        3,
        3,
        height_ratios=[1.1, 1, 0.75],
        hspace=0.38,
        wspace=0.18,
        left=0.04,
        right=0.98,
        top=0.9,
        bottom=0.06,
    )
    axm = [fig.add_subplot(gs[0, c]) for c in range(3)]
    axs = [fig.add_subplot(gs[1, c]) for c in range(3)]
    axo = fig.add_subplot(gs[2, :2])
    axv = fig.add_subplot(gs[2, 2])
    ds = d[np.ix_(order, order)]
    home = [a.get_position().bounds for a in axm]
    cmap_bad = CMAP.copy()
    cmap_bad.set_bad("#f2e9d8")
    names = [
        "classical scan  O(N³)",
        "priority queue  O(N² log N)",
        "compact active set  O(N²)",
    ]

    def draw(f: int) -> None:
        r = min(f, n - 1)
        ivat_phase = f >= n_ord
        for a, pos0 in zip(axm, home):
            a.set_position(pos0)
        for a in axm + axs + [axo, axv]:
            a.clear()
        for k in range(3):
            axm[k].imshow(d, cmap=CMAP, vmin=0, vmax=d.max())
            axm[k].set_xticks([])
            axm[k].set_yticks([])
            axm[k].set_title(names[k], fontsize=12, fontweight="bold")

        u = order[r]
        # ---- classical: examined block = placed x unplaced -------------------
        s = sc[r]
        if s["block"] is not None:
            p, un = s["block"]
            axm[0].imshow(overlay(n, [(p, un, (0.84, 0.15, 0.16), 0.45)]))
            ci, cj = s["cell"]
            axm[0].add_patch(
                plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1, fill=False, ec="gold", lw=2.5)
            )
        placed = order[: r + 1]
        a = axs[0]
        a.set_xlim(0, n)
        a.set_ylim(0, 1)
        for v in range(n):
            a.add_patch(
                plt.Rectangle(
                    (v, 0.55), 0.9, 0.3, fc=(BLUE if v in placed else "white"), ec="0.4"
                )
            )
        a.text(0, 0.95, "tree membership (blue = placed)", fontsize=9)
        pc = len(placed) - 1
        a.text(
            0.5,
            0.3,
            f"recompute min over {pc} placed × {n - pc - 1} unplaced pairs\n"
            f"= {pc * (n - pc - 1)} comparisons this round, all discarded\n"
            "except the one winner",
            fontsize=10,
            va="center",
        )
        a.axis("off")

        # ---- heap ------------------------------------------------------------
        s = sh[r]
        axm[1].imshow(
            overlay(
                n,
                [
                    (np.array([u]), np.arange(n), (0.12, 0.47, 0.71), 0.25),
                    (np.array([u]), s["pushed"], (0.12, 0.47, 0.71), 0.9),
                ],
            )
            if len(s["pushed"])
            else overlay(n, [(np.array([u]), np.arange(n), (0.12, 0.47, 0.71), 0.25)])
        )
        a = axs[1]
        shown = s["pq"][:14]
        for i, (w, v) in enumerate(shown):
            stale = s["in_mst"][v]
            fresh = v in set(s["pushed"].tolist()) and not stale
            a.bar(
                i,
                w,
                color=GREY if stale else (BLUE if fresh else "#7fa8cf"),
                hatch="//" if stale else None,
                ec="0.5",
            )
            a.text(i, w, str(v), ha="center", va="bottom", fontsize=8)
        a.set_xticks([])
        a.set_title(
            f"heap: {len(s['pq'])} entries · +{len(s['pushed'])} pushed · {s['stale']} stale popped",
            fontsize=9,
        )
        a.set_ylim(0, d.max() * 1.1)
        a.text(
            0.99,
            0.95,
            "grey hatched = stale (vertex already in tree)\nblue = pushed this round",
            transform=a.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

        # ---- compact active set ---------------------------------------------
        s = sa[r]
        mrow = np.array([u])
        act = s["av"][: s["m"]]
        ov = [(mrow, act, (0.12, 0.47, 0.71), 0.25)]
        ch_cols = act[s["changed"][: s["m"]]]
        if len(ch_cols):
            ov.append((mrow, ch_cols, (0.12, 0.47, 0.71), 0.9))
        axm[2].imshow(overlay(n, ov))
        a = axs[2]
        for i in range(n):
            if i < s["m"]:
                col = RED if i == s["bk"] else (BLUE if s["changed"][i] else "#7fa8cf")
                a.bar(i, s["ak"][i], color=col, ec="0.5")
                a.text(
                    i, s["ak"][i], str(s["av"][i]), ha="center", va="bottom", fontsize=7
                )
            else:
                a.bar(i, d.max() * 0.04, color="white", ec="0.7", ls=":")
        a.set_xticks([])
        a.set_ylim(0, d.max() * 1.1)
        a.set_title(f"packed slots [0,{s['m']}): one fused pass", fontsize=9)
        a.text(
            0.99,
            0.95,
            "red = next minimum  ·  blue = key improved this round\n"
            "no heap, no stale entries, no log factor",
            transform=a.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

        # ---- ops ------------------------------------------------------------
        for cum, nm, col in (
            (cc, "classical", "#d95f02"),
            (ch, "heap", "#1f77b4"),
            (ca, "active set", "#1b9e77"),
        ):
            axo.plot(np.arange(n), cum, color=col, lw=1.2, alpha=0.35)
            axo.plot(np.arange(r + 1), cum[: r + 1], color=col, lw=2.5, label=nm)
            axo.plot([r], [cum[r]], "o", color=col)
        axo.set_xlim(0, n - 1)
        axo.set_xlabel("round (points placed)", fontsize=9)
        axo.set_ylabel("cumulative primitive ops", fontsize=9)
        axo.legend(loc="upper left", fontsize=9)
        axo.set_title(
            f"work so far: classical {int(cc[r])}  ·  heap {int(ch[r])}  ·  "
            f"active set {int(ca[r])}   (same ordering; N={n} is too small for the heap to win)",
            fontsize=10,
        )
        # ---- the reordered image (identical for all three methods) ---------
        shown = np.full((n, n), np.nan)
        shown[: r + 1, : r + 1] = ds[: r + 1, : r + 1]
        axv.imshow(shown, cmap=cmap_bad, vmin=0, vmax=d.max())
        axv.set_xticks([])
        axv.set_yticks([])
        axv.set_title(f"VAT image D[p,p]: {r + 1}/{n} placed", fontsize=10)
        fig.suptitle(
            f"VAT ordering: three ways to pick the next point   "
            f"(round {r + 1}/{n}, placed vertex {u})",
            fontsize=13,
        )

        # ---- phase 2: iVAT from the finished ordering: top row only --------
        # MST (left) -> VAT image D* (middle) -> iVAT image D' (right)
        if ivat_phase:
            q = min(f - n_ord + 1, n - 1)
            for a in axm + axs + [axo, axv]:
                a.clear()
            for a in axs + [axo, axv]:
                a.axis("off")
            for a, (x0, _, w, _) in zip(axm, home):
                a.set_position([x0, 0.08, w, 0.78])
            ax_t, ax_v, ax_i = axm
            # 1. the MST walk: row q is the q-th vertex Prim added
            ax_t.set_aspect("equal")
            ax_t.scatter(*x.T, s=40, c="0.8", zorder=1)
            for t in range(1, n):
                seg = x[[order[t], order[par[t]]]].T
                ax_t.plot(*seg, color=("0.25" if t < q else "0.8"), lw=1.6, zorder=2)
            ax_t.scatter(*x[order[: q + 1]].T, s=40, c=BLUE, zorder=3)
            ax_t.plot(*x[[order[q], order[par[q]]]].T, color=RED, lw=4, zorder=4)
            ax_t.scatter(
                *x[order[q]][None].T, s=200, fc="none", ec=RED, lw=2.5, zorder=5
            )
            ax_t.scatter(
                *x[order[par[q]]][None].T, s=200, fc="none", ec=BLUE, lw=2.5, zorder=5
            )
            ax_t.set_title(
                "MST, walked in Prim order\nred edge = row q to its parent",
                fontsize=12,
                fontweight="bold",
            )
            # 2. the VAT image
            ax_v.imshow(ds, cmap=CMAP, vmin=0, vmax=d.max())
            ax_v.axhline(q, color=RED, lw=1.5)
            ax_v.plot([par[q]], [q], "s", color=BLUE, ms=7)
            ax_v.set_title(
                f"VAT image D*\nrow {q}: nearest earlier column is {par[q]} (MST parent)",
                fontsize=12,
                fontweight="bold",
            )
            # 3. the iVAT image
            part = np.full((n, n), np.nan)
            part[: q + 1, : q + 1] = div[: q + 1, : q + 1]
            ax_i.imshow(part, cmap=cmap_bad, vmin=0, vmax=div.max())
            ax_i.axhline(q, color=RED, lw=1.5)
            ax_i.set_title(
                f"iVAT image D'\nrow {q}/{n - 1}: D'[r,c] = max(D*[r,j], D'[j,c])",
                fontsize=12,
                fontweight="bold",
            )
            for a in (ax_t, ax_v, ax_i):
                a.set_xticks([])
                a.set_yticks([])
            fig.suptitle(
                "iVAT: fill each row from its parent's row, O(N²) total "
                "(the ordering is final)",
                fontsize=13,
            )

    anim = FuncAnimation(fig, draw, frames=nfr, interval=1000 / fps)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "vat_ordering_methods.gif"
    anim.save(out, writer=PillowWriter(fps=fps))
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# GIF 2 — matrix-free, in-place permutation, memory arithmetic
# --------------------------------------------------------------------------
def cycle_list(p: np.ndarray) -> list[list[int]]:
    """Cycles of length > 1, each started at its smallest index (as the kernel does)."""
    n, seen, out = len(p), set(), []
    for s in range(n):
        if s in seen:
            continue
        c, i = [], s
        while i not in seen:
            seen.add(i)
            c.append(i)
            i = int(p[i])
        if len(c) > 1:
            out.append(c)
    return out


def walk_steps(d: np.ndarray, p: np.ndarray):
    """Replay _permute_sym_inplace on a real copy of D, one assignment per step.

    Phase "rows": M[i,:] <- M[p[i],:] around each cycle with one temp row.
    Phase "cols": the same permutation applied to columns. (The kernel does this
    as a per-row gather through an O(n) temp; the resulting matrix is identical,
    shown here as whole-column moves so the symmetry is visible.)
    Each step: dict(phase, M, cur, src, tmp, cyc)
    """
    M = d.copy()
    steps = [dict(phase="start", M=M.copy(), cur=None, src=None, tmp=None, cyc=-1)]
    cycles = cycle_list(p)
    for phase in ("rows", "cols"):
        for ci, c in enumerate(cycles):
            start = c[0]
            tmp = M[start, :].copy() if phase == "rows" else M[:, start].copy()
            steps.append(
                dict(phase=phase, M=M.copy(), cur=start, src=None, tmp=start, cyc=ci)
            )
            i = start
            while True:
                nxt = int(p[i])
                if nxt == start:
                    if phase == "rows":
                        M[i, :] = tmp
                    else:
                        M[:, i] = tmp
                    steps.append(
                        dict(
                            phase=phase, M=M.copy(), cur=i, src="tmp", tmp=None, cyc=ci
                        )
                    )
                    break
                if phase == "rows":
                    M[i, :] = M[nxt, :]
                else:
                    M[:, i] = M[:, nxt]
                steps.append(
                    dict(phase=phase, M=M.copy(), cur=i, src=nxt, tmp=start, cyc=ci)
                )
                i = nxt
        steps.append(
            dict(
                phase=phase + "_done", M=M.copy(), cur=None, src=None, tmp=None, cyc=-1
            )
        )
    return steps, M, cycles


def build_memory(n: int, seed: int, fps: int) -> None:
    x, _ = make_data(n, seed)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    order = prim_order(d)
    p = np.asarray(order)

    # ---- verification against the library ---------------------------------
    assert [
        int(v) for v in _vat_prim_mst_seq_python(x)
    ] == order, "matrix-free order differs"
    m = d.copy()
    _permute_sym_inplace(m, p.astype(np.int32))
    steps, final, cycles = walk_steps(d, p)
    assert np.array_equal(
        m, d[np.ix_(p, p)]
    ), "in-place permutation differs from P.D.P^T"
    assert np.array_equal(
        final, m
    ), "replayed cycle walk differs from _permute_sym_inplace"
    print(
        f"verified: matrix-free order and in-place walk ({len(cycles)} cycles) match the library"
    )

    hold = 4
    tl = (
        [("free", k) for k in range(0, n + 1)]
        + [("free", n)] * hold
        + [("perm", k) for k in range(len(steps))]
        + [("perm", len(steps) - 1)] * (2 * hold)
    )

    fig, axs = plt.subplots(
        3,
        1,
        figsize=(10.8, 19.2),
        dpi=100,  # portrait 1080x1920 (phone)
        gridspec_kw={
            "hspace": 0.2,
            "left": 0.05,
            "right": 0.95,
            "top": 0.94,
            "bottom": 0.04,
        },
    )
    aF, aC, aM = axs
    cm = plt.get_cmap("tab10")
    cyc_col = {ci: cm(ci % 10) for ci in range(len(cycles))}
    cyc_of = {v: ci for ci, c in enumerate(cycles) for v in c}
    ang = 2 * np.pi * np.arange(n) / n
    pos = np.c_[np.sin(ang), np.cos(ang)]

    def draw(f: int) -> None:
        ph, k = tl[f]
        for a in (aF, aC, aM):
            a.clear()
        kk = k if ph == "perm" else 0
        st = steps[kk]

        # ---- matrix-free -----------------------------------------------------
        aF.set_title(
            "matrix-free: compute row u on demand",
            fontsize=11,
            fontweight="bold" if ph == "free" else "normal",
        )
        placed = order[:k] if ph == "free" else order
        aF.set_aspect("equal")
        aF.scatter(*x.T, s=30, c="0.8")
        if len(placed):
            aF.scatter(*x[placed].T, s=30, c=BLUE)
        if ph == "free" and k >= 1:
            u = order[k - 1]
            aF.scatter(*x[u][None].T, s=150, fc="none", ec=RED, lw=2)
            for v in range(n):
                if v not in placed:
                    aF.plot(*x[[u, v]].T, color=RED, lw=0.5, alpha=0.5)
            aF.set_xlabel(
                f"step {k}: {n} distances from point {u} -> relax keys -> discard the row",
                fontsize=9,
            )
        else:
            aF.set_xlabel(
                "no NxN matrix is ever allocated: workspace = keys + one row",
                fontsize=9,
            )
        aF.set_xticks([])
        aF.set_yticks([])

        # ---- cycle diagram ---------------------------------------------------
        aC.set_title(
            "the permutation as cycles: walk each loop",
            fontsize=11,
            fontweight="bold" if ph == "perm" else "normal",
        )
        aC.set_xlim(-1.45, 1.45)
        aC.set_ylim(-1.45, 1.45)
        aC.set_aspect("equal")
        aC.axis("off")
        upto = kk + 1 if ph != "free" else 1
        done = {
            s2["cyc"]
            for s2 in steps[:upto]
            if s2["src"] == "tmp" and s2["phase"] == "rows"
        }
        for ci, c in enumerate(cycles):
            for i in c:  # arrow: row p[i] moves into slot i
                lw, al = (3.0, 0.95) if ci in done else (1.2, 0.35)
                aC.annotate(
                    "",
                    xy=pos[i],
                    xytext=pos[int(p[i])],
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=cyc_col[ci],
                        lw=lw,
                        alpha=al,
                        shrinkA=9,
                        shrinkB=9,
                        connectionstyle="arc3,rad=0.15",
                    ),
                )
        for v in range(n):
            col = cyc_col[cyc_of[v]] if v in cyc_of else "0.85"
            aC.scatter(*pos[v], s=330, c=[col], ec="0.3", zorder=3)
            aC.text(*pos[v], str(v), ha="center", va="center", fontsize=9, zorder=4)
        if ph == "perm" and st["cur"] is not None:
            aC.scatter(*pos[st["cur"]], s=700, fc="none", ec=RED, lw=3, zorder=5)
            if isinstance(st["src"], int):
                aC.annotate(
                    "",
                    xy=pos[st["cur"]],
                    xytext=pos[st["src"]],
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=RED,
                        lw=3.5,
                        shrinkA=9,
                        shrinkB=9,
                        connectionstyle="arc3,rad=0.15",
                    ),
                    zorder=6,
                )
        what = "row" if st["phase"] == "rows" else "column"
        if ph == "perm" and st["tmp"] is not None:
            aC.text(
                0,
                0,
                f"tmp <- {what} {st['tmp']}\n(the one spare {what})",
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(boxstyle="round", fc="#fff3cd", ec=cyc_col[st["cyc"]]),
            )
        elif ph == "perm" and st["src"] == "tmp":
            aC.text(
                0,
                0,
                f"slot <- tmp\ncycle closed",
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(boxstyle="round", fc="#d4edda", ec="0.4"),
            )
        else:
            aC.text(
                0,
                0,
                f"{len(cycles)} cycles\ngrey = fixed points",
                ha="center",
                va="center",
                fontsize=10,
                color="0.4",
            )

        # ---- live D matrix ---------------------------------------------------
        M = st["M"]
        aM.imshow(M, cmap=CMAP, vmin=0, vmax=d.max())
        aM.set_xticks([])
        aM.set_yticks([])
        sym = np.allclose(M, M.T)
        phase = st["phase"]
        label = {
            "start": "D in input order",
            "rows": "step 1: permute ROWS   M[i,:] <- M[p[i],:]",
            "rows_done": "rows done: D is no longer symmetric",
            "cols": "step 2: permute COLUMNS (same loops)",
            "cols_done": "P.D.P^T: symmetric again, the VAT image",
        }[phase]
        aM.set_title(
            label, fontsize=11, fontweight="bold" if ph == "perm" else "normal"
        )
        if ph == "perm" and st["cur"] is not None:
            cur, srcv = st["cur"], st["src"]
            if phase == "rows":
                aM.add_patch(
                    plt.Rectangle((-0.5, cur - 0.5), n, 1, fill=False, ec=RED, lw=3)
                )
                if isinstance(srcv, int):
                    aM.add_patch(
                        plt.Rectangle(
                            (-0.5, srcv - 0.5), n, 1, fill=False, ec="gold", lw=2
                        )
                    )
            else:
                aM.add_patch(
                    plt.Rectangle((cur - 0.5, -0.5), 1, n, fill=False, ec=RED, lw=3)
                )
                if isinstance(srcv, int):
                    aM.add_patch(
                        plt.Rectangle(
                            (srcv - 0.5, -0.5), 1, n, fill=False, ec="gold", lw=2
                        )
                    )
        aM.set_xlabel(
            f"symmetric: {sym}     red = slot being written, gold = where it copies from",
            fontsize=9,
        )

        fig.suptitle(
            {
                "free": "1. Matrix-free: never build D",
                "perm": "2. In-place: reorder D without a second matrix",
            }[ph],
            fontsize=13,
        )

    anim = FuncAnimation(fig, draw, frames=len(tl), interval=1000 / fps)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "vat_memory_methods.gif"
    anim.save(out, writer=PillowWriter(fps=fps))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["ordering", "memory", "all"], default="all")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--mem-seed", type=int, default=6, help="seed with several cycles")
    ap.add_argument("--fps", type=int, default=4)
    a = ap.parse_args()
    if a.which in ("ordering", "all"):
        build_ordering(24, a.seed, a.fps)
    if a.which in ("memory", "all"):
        build_memory(16, a.mem_seed, a.fps + 2)


if __name__ == "__main__":
    main()
