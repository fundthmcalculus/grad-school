"""One-off export of the all-fuzzy checkpoint to a plain .npz for allfuzzy_lm.py (needs torch; the script itself needs only numpy and tokenizers).

    python allfuzzy/export_weights.py outputs/v52-e2e/e2e_tail_300M_s0.pt allfuzzy/e2e_tail_300M_s0.npz
    python allfuzzy/export_weights.py outputs/v59-e2e300/e2e_all_tanh_300M_s1.pt allfuzzy/e2e_all_tanh_300M_s1.npz --head-tanh

A checkpoint whose head table is fuzzy too (``head_proto`` in the state) is exported with the head rows replaced by their effective embeddings, mu(emb_h) @ head_proto.T with
signed (tanh) memberships mu (``--head-tanh``; the state cannot say which membership function was used) or softmax: the tied output scores of a head token are h . (that row), exactly as for the
fuzzy tail table, so the numpy and browser models need no new code. The true parameter count is stored as ``cfg_params_total``.

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
    head_tanh = "--head-tanh" in sys.argv
    ck = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    arrays = {k: v.numpy().astype(np.float64) for k, v in ck["state"].items()}
    total = sum(v.numel() for v in ck["state"].values())
    if "head_proto" in arrays:
        eh = arrays["emb_h.weight"]
        mu = (
            np.tanh(eh)
            if head_tanh
            else np.exp(eh - eh.max(1, keepdims=True))
            / np.exp(eh - eh.max(1, keepdims=True)).sum(1, keepdims=True)
        )
        arrays["emb_h.weight"] = mu @ arrays.pop("head_proto").T
    assert not any(
        k in arrays for k in ("prior_alpha", "tok_bias", "tier_w", "head_pq", "mos_pi")
    ), "this export handles the signed / softmax head table only"
    arrays["cfg_params_total"] = np.array(total)
    arrays["cfg_tanh"] = np.array(
        int(head_tanh)
    )  # the tail memberships use the same function (flm/fuzzy_tokens.py)
    nh = arrays["emb_h.weight"].shape[0]
    freq = np.load("outputs/freq_bpe3072.npz")["freq"]
    order = np.argsort(-freq, kind="stable")
    arrays["head_ids"], arrays["tail_ids"] = order[:nh], order[nh:]
    for k in ("d_model", "n_layers", "n_heads", "n_rules", "vocab_size", "max_len"):
        arrays["cfg_" + k] = np.array(cfg[k])
    np.savez_compressed(dst, **arrays)
    shutil.copyfile("../../data/tinystories/bpe3072.json", dst.parent / "bpe3072.json")
    print(f"wrote {dst} ({dst.stat().st_size / 1e3:.0f} KB), {total} parameters")
