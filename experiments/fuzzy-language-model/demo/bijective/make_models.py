"""Export the four 10K models of ``outputs/bij-demo`` (trained with ``outputs/bij10k/run_arm.sh <arm> 0 <lr> 30 outputs/bij-demo --save-model``) into ``demo/bijective/models`` and write its index.

    python demo/bijective/make_models.py

Local only: ``demo/bijective/models`` is git-ignored and nothing here is published.
"""

import json
import subprocess
import sys
from pathlib import Path

SRC, DST = Path("outputs/bij-demo"), Path("demo/bijective/models")
ARMS = {
    "tied": (42, 29),
    "ident": (42, 29),
    "c": (32, 21),
    "z": (32, 21),
}  # feed-forward rule counts of each arm at 10K (run_arm.sh)
idx = []
for arm, (r, r0) in ARMS.items():
    pt = next(SRC.glob(f"{arm}_lr*_s0.pt"), None)
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
            f"bij-10k-{arm}",
            "--val",
            str(val),
            "--out",
            str(DST),
        ],
        check=True,
    )
    idx.append({"id": f"bij-10k-{arm}", "label": f"10K {arm}"})
(DST / "index.json").write_text(json.dumps(idx, indent=1))
print("wrote", len(idx), "models to", DST)
