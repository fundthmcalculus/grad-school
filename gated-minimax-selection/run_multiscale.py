"""
Experiments for Option D: Multi-Scale Persistence Selection.

Produces four results, all printed and (where visual) saved to outputs/:

  1. hierarchy_recovery   -- the headline result. On nested/multi-level data,
     flat coverage_cover recovers ONE level; multi-scale recovers EVERY level.
  2. no_regression_flat   -- on the ordinary single-scale battery, multi-scale
     discovers exactly one band and reproduces coverage_cover (strict
     generalization, no cost on the easy case).
  3. flat_is_scale_invariant -- documents WHY we do not claim a flat-ARI win on
     single-level varying-density: flat coverage_cover is already scale-invariant
     across extreme density contrast when clusters are separable.
  4. figures              -- persistence diagram colored by discovered band, and
     the recovered hierarchy, for nested_gaussians.

Run: python run_multiscale.py
"""

import os
import numpy as np
from sklearn.metrics import adjusted_rand_score

import ivat_mf as im
import selection as S
import battery as B
import battery_hierarchical as BH
import multiscale_persistence as MS

OUT = os.path.join(os.path.dirname(__file__), 'outputs')


# ---------------------------------------------------------------------------
def _flat_assign(sel, Dstar):
    return MS.assign(sel, Dstar)


def _best_ari_per_level(assignment, level_labels):
    """ARI of a single hard assignment against each ground-truth level."""
    return [adjusted_rand_score(y, assignment) for y in level_labels]


# ---------------------------------------------------------------------------
def hierarchy_recovery():
    print("=" * 78)
    print("1. HIERARCHY RECOVERY  (flat forced to one level; multi-scale gets all)")
    print("=" * 78)

    results = {}
    for name, (gen, level_names) in BH.HIERARCHICAL.items():
        out = gen()
        X, levels = out[0], list(out[1:])
        Dstar = im.minimax_transform(im.dissimilarity(X))

        # --- flat baseline ---
        flat = S.select_coverage_cover(Dstar)
        flat_assign = _flat_assign(flat, Dstar)
        flat_ari = _best_ari_per_level(flat_assign, levels)

        # --- multi-scale ---
        msel = MS.select_multiscale(Dstar)
        # For each ground-truth level, the best-matching band's ARI.
        band_assigns = [MS.assign_band(b, Dstar) for b in msel.bands]
        per_level_best = []
        for y in levels:
            aris = [adjusted_rand_score(y, a) for a in band_assigns] or [np.nan]
            per_level_best.append(max(aris))

        print(f"\n{name}   (levels: {', '.join(level_names)})")
        print(f"  flat coverage_cover: k={len(flat)}  "
              f"ARI/level = {['%.3f' % v for v in flat_ari]}")
        print(f"  multi-scale: {msel.n_scales} bands, "
              f"granularities {msel.granularities()}")
        for b, a in zip(msel.bands, band_assigns):
            aris = [adjusted_rand_score(y, a) for y in levels]
            print(f"     band {b.band_id}: k={b.k:2d} births[{b.birth_lo:.2f},"
                  f"{b.birth_hi:.2f}] cov={b.coverage_fraction(msel.n):.2f}  "
                  f"ARI/level = {['%.3f' % v for v in aris]}")
        print(f"  --> best ARI per level:  flat={['%.3f' % v for v in flat_ari]}  "
              f"multi-scale={['%.3f' % v for v in per_level_best]}")

        results[name] = {
            'level_names': level_names,
            'flat_k': len(flat), 'flat_ari': flat_ari,
            'ms_granularities': msel.granularities(),
            'ms_best_ari': per_level_best,
        }

    # Summary: mean ARI across ALL levels (the multi-scale claim).
    print("\n" + "-" * 78)
    print("Mean ARI averaged over ALL ground-truth levels (higher = recovers "
          "the whole hierarchy):")
    print(f"  {'dataset':<24}{'flat':>10}{'multi-scale':>14}")
    for name, r in results.items():
        print(f"  {name:<24}{np.mean(r['flat_ari']):>10.3f}"
              f"{np.mean(r['ms_best_ari']):>14.3f}")
    return results


# ---------------------------------------------------------------------------
def no_regression_flat():
    print("\n" + "=" * 78)
    print("2. NO REGRESSION ON THE FLAT BATTERY  (single-scale data -> one band)")
    print("=" * 78)
    datasets = [('two_gaussians', B.two_gaussians, 2),
                ('bridged_gaussians', B.bridged_gaussians, 2),
                ('concentric_rings', B.concentric_rings, 2),
                ('varying_density', B.varying_density, 3),
                ('uniform_noise', B.uniform_noise, 0)]
    print(f"\n  {'dataset':<20}{'flat k':>8}{'flat ARI':>10}"
          f"{'#bands':>8}{'finest k':>10}{'finest ARI':>12}")
    for name, gen, _ in datasets:
        X, y = gen()
        Dstar = im.minimax_transform(im.dissimilarity(X))
        flat = S.select_coverage_cover(Dstar)
        fa = _flat_assign(flat, Dstar)
        f_ari = adjusted_rand_score(y[y >= 0], fa[y >= 0]) if (y >= 0).any() else float('nan')
        msel = MS.select_multiscale(Dstar)
        if msel.n_scales:
            fb = msel.finest()
            ba = MS.assign_band(fb, Dstar)
            b_ari = adjusted_rand_score(y[y >= 0], ba[y >= 0]) if (y >= 0).any() else float('nan')
            fk = fb.k
        else:
            b_ari, fk = float('nan'), 0
        print(f"  {name:<20}{len(flat):>8}{f_ari:>10.3f}"
              f"{msel.n_scales:>8}{fk:>10}{b_ari:>12.3f}")
    print("\n  Expectation: single-scale data yields ONE band whose selection "
          "matches\n  the flat baseline (uniform_noise correctly yields zero "
          "structure).")


# ---------------------------------------------------------------------------
def flat_is_scale_invariant():
    print("\n" + "=" * 78)
    print("3. WHY NO FLAT-ARI CLAIM ON SINGLE-LEVEL VARYING DENSITY")
    print("   (flat coverage_cover is already scale-invariant when separable)")
    print("=" * 78)

    def make(contrast, n=180, seed=104, sep=6.0):
        rng = np.random.default_rng(seed)
        base = 0.25
        sig = [base, base * contrast, base * contrast ** 2]
        xs = [0.0]
        for k in range(1, 3):
            xs.append(xs[-1] + sep * (sig[k - 1] + sig[k]) / 2)
        parts, ys = [], []
        for k, (x, s) in enumerate(zip(xs, sig)):
            parts.append(rng.normal([x, 0], s, (n // 3, 2)))
            ys += [k] * (n // 3)
        return np.vstack(parts), np.array(ys)

    print(f"\n  separation scaled with spread (clusters stay separable):")
    print(f"  {'contrast':>10}{'spread ratio':>16}{'flat k':>8}{'flat ARI':>10}")
    for contrast in [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        X, y = make(contrast)
        Dstar = im.minimax_transform(im.dissimilarity(X))
        flat = S.select_coverage_cover(Dstar)
        a = _flat_assign(flat, Dstar)
        ari = adjusted_rand_score(y, a)
        print(f"  {contrast:>10.1f}{'1:%.0f:%.0f' % (contrast, contrast**2):>16}"
              f"{len(flat):>8}{ari:>10.3f}")
    print("\n  Flat ARI is ~constant across a 30x spread ratio: the minimax "
          "transform +\n  MAD-gate already handle single-level scale gaps. The "
          "genuine gap flat\n  selection cannot cross is NESTED structure "
          "(result #1), not density contrast.")


# ---------------------------------------------------------------------------
def figures():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as e:      # pragma: no cover
        print(f"\n[figures skipped: {e}]")
        return

    X, y_fine, y_coarse = BH.nested_gaussians()
    Dstar = im.minimax_transform(im.dissimilarity(X))
    blocks, n = S._all_blocks(Dstar)
    sig = MS.significant_blocks(blocks, n)
    msel = MS.select_multiscale(Dstar)
    edges = msel.band_edges_log

    def band_of(birth):
        lb = np.log(birth + 1e-12)
        b = 0
        for e in edges:
            if lb >= e:
                b += 1
        return b

    fig, ax = plt.subplots(3, 1, figsize=(7, 15))

    # (a) data colored by fine truth
    ax[0].scatter(X[:, 0], X[:, 1], c=y_fine, cmap='tab10', s=18)
    ax[0].set_title('nested_gaussians\n(6 fine / 2 coarse clusters)')
    ax[0].set_aspect('equal', 'datalim')

    # (b) persistence diagram (birth vs persistence), colored by discovered band
    colors = plt.cm.viridis(np.linspace(0, 0.85, max(1, len(edges) + 1)))
    for b in blocks:
        if 3 <= b['size'] <= 0.6 * n:
            ax[1].scatter(b['birth'], b['persistence'], s=12,
                          color='0.8', zorder=1)
    for b in sig:
        bi = band_of(b['birth'])
        ax[1].scatter(b['birth'], b['persistence'], s=55,
                      color=colors[min(bi, len(colors) - 1)],
                      edgecolor='k', linewidth=0.4, zorder=3)
    for e in edges:
        ax[1].axvline(np.exp(e), ls='--', color='crimson', lw=1)
    ax[1].set_xscale('log')
    ax[1].set_xlabel('birth height (log)')
    ax[1].set_ylabel('persistence (death - birth)')
    ax[1].set_title('persistence diagram\n(significant blocks, colored by scale '
                    'band;\ndashed = discovered band edges)')

    # (c) recovered hierarchy: ARI matrix band x level
    levels = [y_fine, y_coarse]
    level_names = ['fine (6)', 'coarse (2)']
    M = np.array([[adjusted_rand_score(y, MS.assign_band(b, Dstar))
                   for y in levels] for b in msel.bands])
    im_ = ax[2].imshow(M, cmap='YlGn', vmin=0, vmax=1, aspect='auto')
    ax[2].set_xticks(range(len(level_names)))
    ax[2].set_xticklabels(level_names)
    ax[2].set_yticks(range(len(msel.bands)))
    ax[2].set_yticklabels([f'band {b.band_id}\n(k={b.k})' for b in msel.bands])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax[2].text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                       color='k' if M[i, j] < 0.6 else 'w')
    ax[2].set_title('ARI: discovered band vs ground-truth level\n'
                    '(diagonal ~1 = each level recovered by its own band)')
    fig.colorbar(im_, ax=ax[2], fraction=0.046)

    fig.tight_layout()
    path = os.path.join(OUT, 'fig8_multiscale_hierarchy.png')
    fig.savefig(path, dpi=130)
    print(f"\n[figure saved: {path}]")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    hierarchy_recovery()
    no_regression_flat()
    flat_is_scale_invariant()
    figures()
    print("\nDone.")
