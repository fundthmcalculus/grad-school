"""Scale the causal clamp: WHERE does the exogenous-response subspace live, and is
its dimensionality universal across models?

Part 21 found, at one layer of one model, a specific ~8-dim subspace whose removal
causally suppresses the injection response. This scans that finding across depth
and across models.

Two cross-layer-comparable metrics (both need a fixed readout, since 'downstream
layers' changes meaning with the clamp layer):

  behav_KL      KL(final-token dist with clamp-at-L || without), injection prompts.
                The effect of clamping layer L on what the model actually outputs.
                behav_KL(clamp) - behav_KL(random-equal-rank) is the layer's causal
                centrality for the exogenous response.
  final_shift   the injection's within-sequence rms-z shift at the LAST layer
                (paired vs benign_cont), clamped-at-L vs unclamped. Fraction
                retained; a fixed readout comparable across all clamp layers.

Procedure per model: (1) layer sweep at a fixed generous rank to find the causally
central layer(s); (2) k-sweep {1,2,4,8,16,32} at the peak layer to read off the
effective dimensionality. The subspace at each layer is fit from the streaming
window (fmri_stream) in memory; a random equal-rank subspace is the control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .fmri_stream import build, Cfg


class Scanner:
    def __init__(self, rundir: Path):
        self.rundir = rundir
        meta = json.loads((rundir / "meta.json").read_text())
        self.cfg = Cfg(**meta["config"])
        self.win = np.load(rundir / "win.npy")  # (n, 2W, Lp1, D)
        self.wv = np.load(rundir / "win_valid.npy")
        self.W = self.win.shape[1] // 2
        df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
        self.bybase = {}
        for i in range(len(df)):
            self.bybase.setdefault(int(df.base[i]), {})[df.dose_name[i]] = i

        self.tok = AutoTokenizer.from_pretrained(self.cfg.model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.cfg.model_id, dtype=getattr(torch, self.cfg.dtype)
            )
            .to("cuda")
            .eval()
        )
        self.Lp1 = self.model.config.num_hidden_layers + 1
        self.D = self.model.config.hidden_size
        self.probes = build(self.cfg)
        self.pbase = {}
        for p in self.probes:
            self.pbase.setdefault(p["base"], {})[p["dose_name"]] = p

        self.state = {"B": None, "layer": None}
        self._handles = []

    def onset(self, i, layer):
        pre = self.win[i, : self.W, layer, :][self.wv[i, : self.W]]
        if len(pre) >= 2 and self.wv[i, self.W]:
            return self.win[i, self.W, layer, :] - pre.mean(0)
        return None

    def subspace(self, layer, k, fit_bases=None):
        V = []
        for base, b in self.bybase.items():
            if fit_bases is not None and base not in fit_bases:
                continue
            if "injection" in b and "benign_cont" in b:
                a, z = self.onset(b["injection"], layer), self.onset(
                    b["benign_cont"], layer
                )
                if a is not None and z is not None:
                    V.append(a - z)
        V = np.array(V, dtype=np.float64)
        V = V - V.mean(0)
        _, _, Vt = np.linalg.svd(V, full_matrices=False)
        return Vt[:k].astype(np.float32)

    def _hook(self, module, inp, out):
        if self.state["B"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        B = self.state["B"].to(h.dtype)
        h = h - (h @ B.t()) @ B
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def _install(self, layer):
        for hd in self._handles:
            hd.remove()
        self._handles = [
            self.model.model.layers[layer - 1].register_forward_hook(self._hook)
        ]

    def encode(self, p):
        pre = self.tok(p["preamble"] + "\n\n", add_special_tokens=True)["input_ids"]
        full = self.tok(p["preamble"] + "\n\n" + p["payload"], add_special_tokens=True)[
            "input_ids"
        ]
        return full, len(pre)

    @torch.no_grad()
    def forward(self, ids_list):
        L = max(len(x) for x in ids_list)
        ids = torch.full((len(ids_list), L), self.tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(ids_list), L), dtype=torch.long)
        for r, b in enumerate(ids_list):
            ids[r, : len(b)] = torch.tensor(b)
            attn[r, : len(b)] = 1
        out = self.model(
            input_ids=ids.cuda(), attention_mask=attn.cuda(), output_hidden_states=True
        )
        logit = torch.stack(
            [out.logits[r, len(ids_list[r]) - 1] for r in range(len(ids_list))]
        ).float()
        hlast = out.hidden_states[-1].float().cpu().numpy()
        return logit, hlast

    def final_shift(self, hlast, ids_list, bnds):
        v = []
        for r in range(len(ids_list)):
            s = len(ids_list[r])
            b = bnds[r]
            h = hlast[r, :s]
            base = h[:b]
            mu = base.mean(0)
            sd = base.std(0).clip(1e-4)
            z = (h - mu) / sd
            v.append(float(np.sqrt((z[b : b + 3] ** 2).mean())))
        return np.array(v)

    @torch.no_grad()
    def measure(self, layer, k, pairs, fit_bases, rng):
        """return dict with behav_KL and final_shift fraction, for clamp & random."""
        inj = [self.encode(a) for a, _ in pairs]
        ben = [self.encode(b) for _, b in pairs]
        inj_ids, inj_b = [x[0] for x in inj], [x[1] for x in inj]
        ben_ids, ben_b = [x[0] for x in ben], [x[1] for x in ben]
        self._install(layer)
        self.state["B"] = None
        logit0, hI0 = self.forward(inj_ids)
        _, hB0 = self.forward(ben_ids)
        base_fs = (
            self.final_shift(hI0, inj_ids, inj_b)
            - self.final_shift(hB0, ben_ids, ben_b)
        ).mean()
        res = {"baseline_final_shift": round(float(base_fs), 4)}
        for arm, B in (
            ("clamp", self.subspace(layer, k, fit_bases)),
            (
                "random",
                np.linalg.qr(rng.normal(size=(self.D, k)))[0].T.astype(np.float32),
            ),
        ):
            self.state["B"] = torch.tensor(B).cuda()
            logit, hI = self.forward(inj_ids)
            _, hB = self.forward(ben_ids)
            fs = (
                self.final_shift(hI, inj_ids, inj_b)
                - self.final_shift(hB, ben_ids, ben_b)
            ).mean()
            pc = torch.log_softmax(logit, -1)
            p0 = torch.log_softmax(logit0, -1)
            kl = float((pc.exp() * (pc - p0)).sum(-1).mean())
            self.state["B"] = None
            res[arm] = {
                "behav_KL": round(kl, 3),
                "final_shift_frac": round(float(fs / base_fs), 3) if base_fs else None,
            }
        return res

    def close(self):
        for hd in self._handles:
            hd.remove()


def run(rundir: Path, n_probe=40, layer_rank=16, k_grid=(1, 2, 4, 8, 16, 32)) -> dict:
    sc = Scanner(rundir)
    rng = np.random.default_rng(0)
    all_bases = sorted(sc.pbase)
    fit_bases = set(all_bases[::2])  # held-out: fit on even
    eval_bases = [b for b in all_bases if b % 2 == 1]
    pairs = [
        (sc.pbase[b]["injection"], sc.pbase[b]["benign_cont"])
        for b in eval_bases
        if "injection" in sc.pbase[b] and "benign_cont" in sc.pbase[b]
    ][:n_probe]

    # 1) layer sweep at fixed rank
    grid = sorted(set(int(x) for x in np.linspace(2, sc.Lp1 - 2, 12)))
    sweep = {}
    for L in grid:
        sweep[L] = sc.measure(L, layer_rank, pairs, fit_bases, rng)
        print(
            f"[scan {rundir.name}] L{L}: "
            f"clampKL={sweep[L]['clamp']['behav_KL']} "
            f"randKL={sweep[L]['random']['behav_KL']} "
            f"clampFS={sweep[L]['clamp']['final_shift_frac']}",
            flush=True,
        )
    # causal centrality = clamp behav_KL minus random behav_KL
    central = {
        L: sweep[L]["clamp"]["behav_KL"] - sweep[L]["random"]["behav_KL"] for L in grid
    }
    peak_layer = max(central, key=central.get)

    # 2) k-sweep at peak layer
    ksweep = {}
    for k in k_grid:
        ksweep[k] = sc.measure(peak_layer, k, pairs, fit_bases, rng)
        print(
            f"[kscan {rundir.name}] L{peak_layer} k{k}: "
            f"clampKL={ksweep[k]['clamp']['behav_KL']} "
            f"FSfrac={ksweep[k]['clamp']['final_shift_frac']}",
            flush=True,
        )
    sc.close()
    return {
        "model": sc.cfg.model_id,
        "n_layers": sc.Lp1 - 1,
        "hidden": sc.D,
        "dataset": rundir.name,
        "layer_rank": layer_rank,
        "n_probe": len(pairs),
        "peak_layer": peak_layer,
        "peak_layer_frac_depth": round(peak_layer / sc.Lp1, 2),
        "causal_centrality": {int(L): round(float(v), 3) for L, v in central.items()},
        "layer_sweep": {int(L): sweep[L] for L in grid},
        "k_sweep_at_peak": {int(k): ksweep[k] for k in k_grid},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/stream_qwen3b")
    ap.add_argument("--n-probe", type=int, default=40)
    a = ap.parse_args()
    r = run(Path(a.run), n_probe=a.n_probe)
    (Path(a.run) / "clamp_scan.json").write_text(json.dumps(r, indent=2))
    print(f"\n== {r['model']} ({r['n_layers']}L, d={r['hidden']}) / {r['dataset']} ==")
    print(
        f"peak causal layer = {r['peak_layer']} ({r['peak_layer_frac_depth']:.0%} depth)"
    )
    print("k-sweep at peak (clamp): k  behav_KL  final_shift_frac")
    for k, v in r["k_sweep_at_peak"].items():
        print(
            f"   {k:>3}   {v['clamp']['behav_KL']:>6.3f}   {v['clamp']['final_shift_frac']}"
        )


if __name__ == "__main__":
    main()
