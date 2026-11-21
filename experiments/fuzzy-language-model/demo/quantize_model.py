"""Make an int8 copy of an exported demo model (one scale per matrix row, vectors one scale per tensor): about a quarter of the size, +0.001 bits per character (`report/infer_precision.md`).

    python demo/quantize_model.py --web outputs/demo14m_site --id flm-14m-tinystories --npz outputs/exp14/ts/m14.npz --out outputs/demo14m_int8 [--tokenizer-json path] [--label "..."]

Reads ``<web>/models/<id>.{json,bin,rules.json}`` (float32) and writes ``<out>/models/<id>.{json,bin,rules.json,check.json}`` with ``dtype: "int8"``: the bin holds every tensor as int8 bytes followed by its float32 row scales (the manifest names both offsets), the two id lists stay float32.
The browser engine (`flm.js` readTensors) rebuilds float32 weights as q x scale.  ``check.json`` is made with the numpy model (`allfuzzy/allfuzzy_lm.py`) on exactly the weights the browser will rebuild, so ``node demo/web/test_parity.mjs`` checks the int8 file against its own reference, not against the float32 model.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "allfuzzy"))
from allfuzzy_lm import AllFuzzyLM  # noqa: E402

KEEP = ("head_ids", "tail_ids")


def quantise(a: np.ndarray):
    """(int8 array, float32 scales, dequantised float32 array): one scale per row of a matrix, one per tensor otherwise."""
    rows = a.shape[0] if a.ndim >= 2 else 1
    flat = a.reshape(rows, -1).astype(np.float32)
    scale = (np.abs(flat).max(1).clip(min=1e-12) / np.float32(127)).astype(np.float32)
    q = np.clip(np.rint(flat / scale[:, None]), -127, 127).astype(np.int8)
    deq = (q.astype(np.float32) * scale[:, None]).reshape(a.shape)
    return q.reshape(a.shape), scale, deq


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument(
        "--npz",
        required=True,
        help="the checkpoint's npz (allfuzzy/export_weights.py), for the reference logits",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-json", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    web, out = Path(a.web), Path(a.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    man = json.loads((web / "models" / f"{a.id}.json").read_text())
    raw = (web / "models" / f"{a.id}.bin").read_bytes()
    chunks, off, tensors, deq = [], 0, [], {}
    for t in man["tensors"]:
        n = int(np.prod(t["shape"]))
        arr = np.frombuffer(raw, dtype="<f4", count=n, offset=t["offset"]).reshape(
            t["shape"]
        )
        if t["name"] in KEEP:
            pad = (-off) % 4
            chunks.append(b"\0" * pad)
            off += pad
            tensors.append({"name": t["name"], "shape": t["shape"], "offset": off})
            chunks.append(arr.astype("<f4").tobytes())
            off += n * 4
            deq[t["name"]] = arr
            continue
        q, scale, d = quantise(arr)
        deq[t["name"]] = d
        qoff = off
        chunks.append(q.tobytes())
        off += n
        pad = (-off) % 4
        chunks.append(b"\0" * pad)
        off += pad
        chunks.append(scale.astype("<f4").tobytes())
        tensors.append(
            {
                "name": t["name"],
                "shape": t["shape"],
                "offset": qoff,
                "q": 1,
                "scale_offset": off,
            }
        )
        off += scale.size * 4
    (out / "models" / f"{a.id}.bin").write_bytes(b"".join(chunks))
    man8 = dict(man, dtype="int8", tensors=tensors)
    if a.label:
        man8["label"] = a.label
    (out / "models" / f"{a.id}.json").write_text(json.dumps(man8))
    shutil.copyfile(
        web / "models" / f"{a.id}.rules.json", out / "models" / f"{a.id}.rules.json"
    )
    # reference logits of the numpy model on the weights the browser rebuilds
    z = dict(np.load(a.npz))
    for k, v in deq.items():
        if k not in KEEP:  # the id lists stay the integers of the npz
            z[k] = v.astype(np.float64)
    tmp = out / "_int8_ref.npz"
    np.savez(tmp, **z)
    tok = (
        Path(a.tokenizer_json)
        if a.tokenizer_json
        else ROOT / "allfuzzy" / f"bpe{man['config']['vocab_size']}.json"
    )
    m = AllFuzzyLM(str(tmp), tokenizer=tok)
    ids = json.loads((web / "models" / f"{a.id}.check.json").read_text())["ids"]
    m.reset()
    last = None
    for i in ids:
        last, _ = m.step(int(i))
    (out / "models" / f"{a.id}.check.json").write_text(
        json.dumps({"ids": ids, "last_logits": [float(x) for x in last]})
    )
    tmp.unlink()
    size = (out / "models" / f"{a.id}.bin").stat().st_size
    print(f"{a.id}: {size / 1e6:.2f} MB int8 (float32: {len(raw) / 1e6:.2f} MB)")
