"""Reference encodings for the browser tokenizer test: ``python demo/export_tok_check.py`` -> web/tokenizer.check.json."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flm.bpe import BPE  # noqa: E402

TEXTS = [
    "Once upon a time",
    "Lily and Ben went to the park.",
    "The little dog was sad because",
    'One day, a girl named Mia found a shiny red ball!\n\nShe said, "Wow, it\'s so big."',
    "  leading spaces,  double  spaces and a trailing newline\n",
    "Numbers 123 and 4567; symbols #$%&* -- ok?",
    "It's they're we've I'm you'll he'd",
    "Once upon a time\x1fThe end.\x1f",
]
if __name__ == "__main__":
    import os

    os.environ.setdefault("GRAD_SCHOOL_DATA", str(ROOT.parents[1] / "data"))
    bp = BPE(1024)
    out = [{"text": t, "ids": [int(i) for i in bp.encode(t)]} for t in TEXTS]
    for o in out:
        o["decoded"] = bp.decode(o["ids"])
    (ROOT / "demo" / "web" / "tokenizer.check.json").write_text(json.dumps(out))
    print(len(out), "reference encodings")
