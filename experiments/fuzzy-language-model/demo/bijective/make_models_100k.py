"""Export the three 100K models of ``outputs/bij-demo100`` into ``demo/bijective/models`` and write the index (the models the GitHub Pages demo shows).

    python demo/bijective/make_models_100k.py

The checkpoints come from ``outputs/bij10k/run_size_save.sh 100k <arm> 0 <lr> 30 outputs/bij-demo100`` (30M characters, seed 0): the routed-back model with the invertible gain (lr 0.01), the routed-back model
without it (lr 0.04) and the usual tied head (lr 0.01).  ``demo/bijective/models`` is git-ignored; its files go to gh-pages only.
"""

import json
import subprocess
import sys
from pathlib import Path

SRC, DST = Path("outputs/bij-demo100"), Path("demo/bijective/models")
ARMS = {  # arm: (feed-forward rules layer 1, layer 0, label)
    "ci": (
        156,
        109,
        "100K routed back through the exact inverse, with the invertible gain",
    ),
    "c": (156, 109, "100K routed back through the exact inverse (lossy norm)"),
    "tied": (186, 130, "100K tied head (no invertible layer)"),
}
idx = []
for arm, (r, r0, label) in ARMS.items():
    pt = next(SRC.glob(f"100k_{arm}_lr*_s0.pt"), None)
    if pt is None or not pt.with_suffix(".json").exists():
        print("skipping", arm, "(not trained yet)")
        continue
    val = json.loads(pt.with_suffix(".json").read_text())["val_bpc"]
    subprocess.run(
        [
            sys.executable,
            "demo/export_bijective.py",
            "--ckpt",
            str(pt),
            "--arm",
            arm,
            "--ffn-r",
            str(r),
            "--ffn-r0",
            str(r0),
            "--id",
            f"bij-100k-{arm}",
            "--label",
            f"{label}: {val:.3f} bits per character",
            "--val",
            str(val),
            "--out",
            str(DST),
        ],
        check=True,
    )
    idx.append(
        {"id": f"bij-100k-{arm}", "label": f"{label} ({val:.3f} bits per character)"}
    )
(DST / "index.json").write_text(json.dumps(idx, indent=1))
print("wrote", len(idx), "models to", DST)
