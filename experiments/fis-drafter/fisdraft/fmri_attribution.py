"""Phase 2: per-layer anomaly attribution from the FIS 'none of the above' rule.

On the scalar AUROC the FIS ties Mahalanobis (Part 3), so the FIS has to earn
its place another way. Its mechanism gives one for free that Mahalanobis does
not: the anomaly score is a **sum of per-feature contributions**, so it
decomposes by layer into a signature that says *where* in the network a prompt
looks abnormal.

The rule's anomaly score is 1 - prod_j membership_j, and in log space

    -log(1 - anomaly) approx  sum_j 0.5 * z_j^2 ,   z_j = (x_j - mu_j)/sigma_j

so feature j contributes 0.5 z_j^2 and a layer's contribution is the sum over
its features. Grouping by layer turns a scalar detector into a per-prompt,
per-layer profile -- an attribution Mahalanobis' single quadratic form does not
hand you.

Two things this establishes, neither of which is an AUROC:

  1. the attribution is FAITHFUL -- the layers it weights are the layers whose
     own one-class detector actually discriminates (correlate per-layer
     attribution gap against per-layer AUROC).
  2. different attack styles have different signatures -- deepset's terse
     instruction-overrides vs jailbreak's long role-play prompts light up
     different depths.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def per_layer_attribution(act, fit_idx, per_layer_pca=8):
    """Return (attrib, layers): attrib[i, l] = layer l's anomaly contribution
    for prompt i under a FIS fitted on `fit_idx`. Whitened per layer so the
    per-feature independence the rule assumes actually holds."""
    n, Lp1, D = act.shape
    attrib = np.zeros((n, Lp1))
    for l in range(Lp1):
        A = act[:, l, :]
        k = min(per_layer_pca, D, len(fit_idx) - 1)
        p = PCA(n_components=k, whiten=True, random_state=0).fit(A[fit_idx])
        Z = p.transform(A)
        mu = Z[fit_idx].mean(0)
        sd = Z[fit_idx].std(0)
        sd = np.maximum(sd, 1e-3 * np.abs(mu).mean() + 1e-6)
        z = (Z - mu) / sd
        attrib[:, l] = 0.5 * (z**2).mean(1)  # mean so layers are comparable
    return attrib, np.arange(Lp1)


def run(rundir: Path, variant="mean", seed=0) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / f"act_{variant}.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    benign = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(benign)
    fit = benign[: int(0.6 * len(benign))]
    test_b = benign[int(0.6 * len(benign)) :]
    inj = np.where(y == 1)[0]

    attrib, layers = per_layer_attribution(act, fit)

    # per-layer discrimination (AUROC on that layer's attribution alone)
    yt = np.r_[np.zeros(len(test_b)), np.ones(len(inj))]
    per_layer_auc, gap = [], []
    for l in layers:
        s = np.r_[attrib[test_b, l], attrib[inj, l]]
        per_layer_auc.append(roc_auc_score(yt, s))
        gap.append(float(attrib[inj, l].mean() - attrib[test_b, l].mean()))

    # faithfulness: does the attribution gap track per-layer AUROC?
    faith = float(np.corrcoef(gap, per_layer_auc)[0, 1])

    # signature of the whole score, normalised so it sums to 1 per group
    def sig(idx):
        m = attrib[idx].mean(0)
        return (m / m.sum()).tolist()

    out = {
        "variant": variant,
        "n_layers": len(layers),
        "faithfulness_corr_gap_vs_auroc": round(faith, 3),
        "per_layer_auroc": [round(a, 3) for a in per_layer_auc],
        "per_layer_gap": [round(g, 3) for g in gap],
        "signature_benign": [round(x, 4) for x in sig(test_b)],
        "signature_injection": [round(x, 4) for x in sig(inj)],
        "argmax_layer_benign": int(np.argmax(sig(test_b))),
        "argmax_layer_injection": int(np.argmax(sig(inj))),
    }

    # per-prompt: fraction of injections whose most-anomalous layer is 'deep'
    deep = len(layers) // 2
    inj_argmax = attrib[inj].argmax(1)
    ben_argmax = attrib[test_b].argmax(1)
    out["frac_peak_deep_injection"] = round(float((inj_argmax >= deep).mean()), 3)
    out["frac_peak_deep_benign"] = round(float((ben_argmax >= deep).mean()), 3)
    return out


def compare(runs: dict[str, Path], variant="mean", seed=0):
    """Signature comparison across attack corpora / models."""
    print("\nper-layer injection-vs-benign attribution gap (normalised profile):")
    for name, rd in runs.items():
        r = run(rd, variant=variant, seed=seed)
        g = np.array(r["per_layer_gap"])
        gn = g / (np.abs(g).max() + 1e-9)
        prof = "".join(" .:-=+*#@"[min(8, int(max(0, v) * 8))] for v in gn)
        print(
            f"  {name:16s} faith={r['faithfulness_corr_gap_vs_auroc']:+.2f} "
            f"peak L{r['argmax_layer_injection']:2d} "
            f"deep%={r['frac_peak_deep_injection']:.2f}  |{prof}|"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/injection")
    ap.add_argument("--variant", default="mean")
    ap.add_argument(
        "--compare",
        nargs="*",
        default=None,
        help="name=path pairs for signature comparison",
    )
    a = ap.parse_args()
    if a.compare:
        runs = {kv.split("=")[0]: Path(kv.split("=")[1]) for kv in a.compare}
        compare(runs, variant=a.variant)
        return
    rundir = Path(a.run)
    r = run(rundir, variant=a.variant)
    (rundir / f"attribution_{a.variant}.json").write_text(json.dumps(r, indent=2))
    print(json.dumps({k: v for k, v in r.items() if not isinstance(v, list)}, indent=2))
    print("\nper-layer AUROC (attribution) and gap:")
    for l, (au, g) in enumerate(zip(r["per_layer_auroc"], r["per_layer_gap"])):
        if l % 3 == 0 or au > 0.7:
            print(f"  layer {l:2d}: auroc {au:.3f}  gap {g:+.3f}")


if __name__ == "__main__":
    main()
