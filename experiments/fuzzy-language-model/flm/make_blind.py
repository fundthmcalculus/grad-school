"""Build a blind rating set from saved texts: ``python -m flm.make_blind report/coherenceNN label=path.json[:per_ckpt_prompt] ...`` (`PLAN_REPEAT.md`).

Each ``label=path`` takes the first ``per_ckpt_prompt`` texts (default 4) of every (checkpoint, prompt) pair of a ``texts_*.json`` file written by `flm.decode_sweep` (or ``flm.post_ul``), so a group of 2 checkpoints x 4 prompts x 4 = 32 texts per group.
Writes ``<out>_blind.json`` (id and text, shuffled, labels removed) and ``<out>_key.json`` (id -> group label, checkpoint and prompt), as `flm.coherence_set` does.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

if __name__ == "__main__":
    out = sys.argv[1]
    items = []
    for spec in sys.argv[2:]:
        label, rest = spec.split("=", 1)
        path, _, n = rest.partition(":")
        n = int(n or 4)
        taken = defaultdict(int)
        for r in json.loads(Path(path).read_text()):
            k = (r["ckpt"], r["prompt"])
            if taken[k] < n:
                taken[k] += 1
                items.append(
                    dict(
                        group=label, ckpt=r["ckpt"], prompt=r["prompt"], text=r["text"]
                    )
                )
    random.Random(20261022).shuffle(items)
    blind, key = [], {}
    for i, it in enumerate(items, 1):
        sid = f"S{i:03d}"
        blind.append({"id": sid, "text": it["text"]})
        key[sid] = {"group": it["group"], "ckpt": it["ckpt"], "prompt": it["prompt"]}
    Path(out + "_blind.json").write_text(json.dumps(blind, indent=1))
    Path(out + "_key.json").write_text(json.dumps(key, indent=1))
    print(
        len(items),
        "texts;",
        {
            g: sum(1 for v in key.values() if v["group"] == g)
            for g in {v["group"] for v in key.values()}
        },
    )
