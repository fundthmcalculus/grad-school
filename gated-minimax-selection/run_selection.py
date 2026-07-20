"""
Compare block selectors. The metric that matters under TRIBBLE's conorm model:
COVERAGE (did we reach every genuine population?) and PURITY (are the selected
blocks clean cluster proxies, not slivers or merged chains?). Number of blocks
is an output for the coverage selector, so we report it.
"""

import numpy as np
import ivat_mf as im
import battery as B
import selection as S


DATASETS = [
    ("two_gaussians",     B.two_gaussians,     2),
    ("bridged_gaussians", B.bridged_gaussians, 2),
    ("concentric_rings",  B.concentric_rings,  2),
    ("varying_density",   B.varying_density,   3),
    ("uniform_noise",     B.uniform_noise,     2),
]


def main():
    print(f"{'dataset':<19}{'selector':<16}{'#blk':>5}{'cover':>7}{'purity':>8}"
          f"   sizes")
    print("-" * 74)
    for name, fn, c in DATASETS:
        X, y = fn()
        D = im.dissimilarity(X)
        Dstar = im.minimax_transform(D)
        n = Dstar.shape[0]

        selectors = [
            ("topc_disjoint",  lambda: S.select_topc_disjoint(Dstar, c)),
            ("relpersist",     lambda: S.select_relpersist(Dstar, c)),
            ("coverage_cover", lambda: S.select_coverage_cover(Dstar)),
        ]
        for sname, fnsel in selectors:
            sel = fnsel()
            cov = S.coverage_of(sel, n)
            pur = S.purity_vs_truth(sel, y)
            sizes = sorted([b['size'] for b in sel], reverse=True)
            sizes_str = str(sizes[:6]) + ("..." if len(sizes) > 6 else "")
            purs = "  n/a" if np.isnan(pur) else f"{pur:6.2f}"
            print(f"{name:<19}{sname:<16}{len(sel):>5}{cov:>7.2f}{purs:>8}   {sizes_str}")
        print()

    print("What good looks like under the conorm model:")
    print("  coverage ~1.0 on structured data (every population reached)")
    print("  purity ~1.0 (blocks are clean cluster proxies, not chains/slivers)")
    print("  varying_density: coverage_cover should reach cover~1.0 with clean")
    print("    blocks where relpersist/topc previously dropped a cluster.")
    print("  uniform_noise: LOW coverage or few eligible blocks = correctly")
    print("    declining to assert structure.")


if __name__ == "__main__":
    main()
