"""Score a checkpoint on another validation token file: ``python -m flm.eval_other ckpt.pt valid_tokens.bin lens.npy [chars]``.
Bits per character over non-overlapping ctx-length windows, same metric as flm.train.evaluate (lens = characters per token).
"""

import sys

import numpy as np
import torch

from flm import train
from flm.models import ModelConfig, TinyLM

ck = torch.load(sys.argv[1], map_location="cpu")
m = TinyLM(ModelConfig(**ck["config"]))
m.load_state_dict(ck["state"])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m.to(dev).eval()
val = np.fromfile(sys.argv[2], dtype=np.uint16)


class _T:
    lens = np.load(sys.argv[3])


train.TOK = _T()
n = int(float(sys.argv[4])) if len(sys.argv) > 4 else 1_000_000
print(round(train.evaluate(m, val, 512, n, device=dev), 4))
