"""Analyze the within-sequence exogenous shift: when, where, how graded, one direction?

Reads a run from fmri_stream.py and answers four questions the hypothesis raises.

1. WHERE  For each dose, the paired differential shift vs the benign_cont control
   (same preamble, same boundary structure -- so the boundary artefact cancels and
   only the payload *content* remains), per layer. -> the most responsive layer.

2. GRADED?  Is the shift ordered benign_cont < offtopic < bizarre < injection (a
   clean dose-response), or does it track *surprise* (bizarre high) rather than
   *adversarial intent* (injection)? This separates novelty from attack.

3. DETECT  AUROC of the within-sequence shift score (best layer, payload onset) for
   injection-vs-benign_cont and bizarre-vs-benign_cont. Because the baseline is the
   preceding text in the SAME sequence, this needs no cross-prompt calibration --
   the property Parts 16-17 showed the corpus-baseline monitor lacks.

4. ONE DIRECTION?  At the most responsive layer, is the payload-onset shift
   concentrated in a single direction (so it could be clamped)? PCA on the onset
   vectors; report top-1 variance explained and cosine agreement across probes,
   for injection vs bizarre. The top direction is saved for the clamp experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CURVES = ["rms_z", "max_z", "pca_residual"]


def onset_vectors(win, win_valid, layer):
    """(n, 2W, Lp1, D) raw window -> per-probe payload-onset shift vector at `layer`.

    Window index W is the boundary (payload first token); [0,W) is the pre-boundary
    baseline. Onset vector = payload-first-token activation minus pre-boundary mean.
    """
    n, TW, Lp1, D = win.shape
    W = TW // 2
    V = np.zeros((n, D), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        pre = win[i, :W, layer, :][win_valid[i, :W]]
        if len(pre) >= 2 and win_valid[i, W]:
            V[i] = win[i, W, layer, :] - pre.mean(0)
            ok[i] = True
    return V, ok


def run(rundir: Path) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    dev = np.load(rundir / "dev.npy")  # (n, T, Lp1, 3)
    bnd = np.load(rundir / "boundary.npy")
    sl = np.load(rundir / "seqlen.npy")
    meta = json.loads((rundir / "meta.json").read_text())
    mid = meta["config"]["model_id"]
    Lp1 = dev.shape[2]
    doses = ["benign_cont", "offtopic", "bizarre", "injection"]

    def onset_dev(i, c, layer, span=1):
        """mean of curve c at layer over the first `span` payload tokens."""
        b = int(bnd[i])
        s = int(sl[i])
        return float(np.nanmean(dev[i, b : min(s, b + span), layer, c]))

    bybase = {}
    for i in range(len(df)):
        bybase.setdefault(int(df.base[i]), {})[df.dose_name[i]] = i

    # ---- 1+2: paired differential shift per dose per layer per curve ----
    atlas = {cn: {ds: np.zeros(Lp1) for ds in doses} for cn in CURVES}
    for c, cn in enumerate(CURVES):
        for layer in range(Lp1):
            for ds in doses:
                dd = [
                    onset_dev(b[ds], c, layer) - onset_dev(b["benign_cont"], c, layer)
                    for b in bybase.values()
                    if ds in b and "benign_cont" in b
                ]
                atlas[cn][ds][layer] = float(np.nanmean(dd))

    # best layer per curve = argmax injection differential
    best_layer = {cn: int(np.nanargmax(atlas[cn]["injection"])) for cn in CURVES}

    # ---- 3: detection AUROC at the best layer, payload onset ----
    detect = {}
    for c, cn in enumerate(CURVES):
        L = best_layer[cn]
        s = {
            ds: np.array([onset_dev(b[ds], c, L) for b in bybase.values() if ds in b])
            for ds in doses
        }
        bc = s["benign_cont"]
        detect[cn] = {"best_layer": L}
        for ds in ["offtopic", "bizarre", "injection"]:
            y = np.r_[np.zeros(len(bc)), np.ones(len(s[ds]))]
            sc = np.r_[bc, s[ds]]
            detect[cn][f"auroc_{ds}_vs_benign"] = round(float(roc_auc_score(y, sc)), 3)

    # ---- 4: one direction? PCA on payload-onset vectors at a chosen layer ----
    win = np.load(rundir / "win.npy")
    win_valid = np.load(rundir / "win_valid.npy")
    # use the residual-curve best layer (novelty layer) for direction work
    dir_layer = best_layer["pca_residual"]
    direction = {}
    saved_dir = None
    for ds in ["bizarre", "injection"]:
        idx = [b[ds] for b in bybase.values() if ds in b]
        V, ok = onset_vectors(win, win_valid, dir_layer)
        Vd = V[[i for i in idx if ok[i]]]
        Vd = Vd - Vd.mean(0)
        # PCA via SVD
        U, S, Vt = np.linalg.svd(Vd, full_matrices=False)
        var = (S**2) / (S**2).sum()
        top = Vt[0]
        # cosine agreement: how aligned are individual onset vectors with top dir
        raw = V[[i for i in idx if ok[i]]]
        cos = (raw @ top) / (np.linalg.norm(raw, axis=1) * np.linalg.norm(top) + 1e-8)
        direction[ds] = {
            "layer": dir_layer,
            "n": int(len(Vd)),
            "top1_var": round(float(var[0]), 3),
            "top3_var": round(float(var[:3].sum()), 3),
            "mean_abs_cos": round(float(np.abs(cos).mean()), 3),
        }
        if ds == "injection":
            saved_dir = top.astype(np.float32)
    if saved_dir is not None:
        np.save(rundir / "exo_direction.npy", saved_dir)
        (rundir / "exo_direction_meta.json").write_text(
            json.dumps({"layer": dir_layer, "dose": "injection"}, indent=2)
        )

    return {
        "model": mid,
        "dataset": rundir.name,
        "best_layer": best_layer,
        "dose_response_at_best": {
            cn: {ds: round(float(atlas[cn][ds][best_layer[cn]]), 3) for ds in doses}
            for cn in CURVES
        },
        "detect": detect,
        "direction": direction,
        "atlas": {
            cn: {ds: atlas[cn][ds].round(3).tolist() for ds in doses} for cn in CURVES
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/stream_qwen3b")
    a = ap.parse_args()
    r = run(Path(a.run))
    (Path(a.run) / "stream_shift.json").write_text(json.dumps(r, indent=2))
    print(f"Within-sequence exogenous shift -- {r['model']} / {r['dataset']}\n")
    print("Dose-response (paired diff vs benign_cont, at each curve's best layer):")
    for cn in CURVES:
        L = r["best_layer"][cn]
        dd = r["dose_response_at_best"][cn]
        print(
            f"  {cn:<13} L{L:<3} " + "  ".join(f"{k}={v:+.3f}" for k, v in dd.items())
        )
    print("\nDetection AUROC vs benign_cont (within-sequence, no cross-prompt calib):")
    for cn in CURVES:
        d = r["detect"][cn]
        print(
            f"  {cn:<13} L{d['best_layer']:<3} "
            f"offtopic={d['auroc_offtopic_vs_benign']}  "
            f"bizarre={d['auroc_bizarre_vs_benign']}  "
            f"injection={d['auroc_injection_vs_benign']}"
        )
    print("\nOne responsive direction? (PCA on payload-onset vectors)")
    for ds, v in r["direction"].items():
        print(
            f"  {ds:<10} L{v['layer']} top1_var={v['top1_var']} "
            f"top3_var={v['top3_var']} mean|cos|={v['mean_abs_cos']}"
        )


if __name__ == "__main__":
    main()
