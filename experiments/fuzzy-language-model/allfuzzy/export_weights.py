"""One-off export of the all-fuzzy checkpoint to a plain .npz for allfuzzy_lm.py (needs torch; the script itself needs only numpy and tokenizers).

    python allfuzzy/export_weights.py outputs/v52-e2e/e2e_tail_300M_s0.pt allfuzzy/e2e_tail_300M_s0.npz

Besides every tensor of the state dict it stores which vocabulary ids are 'head' tokens (crisp embedding rows) and which are 'tail' tokens (fuzzy
sets over 12 prototypes), recomputed from the token-frequency file exactly as flm/run_tiered.py does.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import torch

if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    ck = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    arrays = {k: v.numpy().astype(np.float64) for k, v in ck["state"].items()}
    nh = arrays["emb_h.weight"].shape[0]
    freq = np.load("outputs/freq_bpe3072.npz")["freq"]
    order = np.argsort(-freq, kind="stable")
    arrays["head_ids"], arrays["tail_ids"] = order[:nh], order[nh:]
    for k in ("d_model", "n_layers", "n_heads", "n_rules", "vocab_size", "max_len"):
        arrays["cfg_" + k] = np.array(cfg[k])
    np.savez_compressed(dst, **arrays)
    shutil.copyfile("../../data/tinystories/bpe3072.json", dst.parent / "bpe3072.json")
    print(
        f"wrote {dst} ({dst.stat().st_size / 1e3:.0f} KB), {sum(v.numel() for v in ck['state'].values())} parameters"
    )
