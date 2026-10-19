"""Build ONE self-contained HTML file of the in-browser demo (opens from a file:// address, works offline).

    python demo/build_standalone.py [--all]  ->  demo/dist/index.html

The weights and rule views of the models are embedded as base64. By default only the sub-100K models go in (about 2.5 MB);
``--all`` adds the 1.32M model (over the cap, about 4.5 MB more). The committed copy (demo/dist/index.html, built with --all, 7.4 MB: the one approved exception to the 5 MB data limit) lets GitHub Pages serve it statically; rebuild it with --all and commit it when demo/web changes.
"""

import base64
import json
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
DIST = Path(__file__).resolve().parent / "dist"


def main():
    everything = "--all" in sys.argv
    index = json.loads((WEB / "models" / "index.json").read_text())
    if not everything:
        index = [m for m in index if m["params"] < 100_000]
    files = {"tokenizer.json": WEB / "tokenizer.json"}
    for m in index:
        for ext in ("json", "bin", "rules.json"):
            files[f"models/{m['id']}.{ext}"] = WEB / "models" / f"{m['id']}.{ext}"
    emb = {k: base64.b64encode(p.read_bytes()).decode() for k, p in files.items()}
    emb["models/index.json"] = base64.b64encode(json.dumps(index).encode()).decode()
    html = (WEB / "index.html").read_text(encoding="utf-8")
    js = (WEB / "flm.js").read_text(encoding="utf-8")
    html = html.replace(
        '<script src="flm.js"></script>', "<script>\n" + js + "\n</script>"
    )
    html = html.replace(
        "<!--EMBED-->", "<script>window.FLM_EMBED = " + json.dumps(emb) + ";</script>"
    )
    DIST.mkdir(exist_ok=True)
    out = DIST / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"{out}  {out.stat().st_size / 1e6:.2f} MB  ({len(index)} models)")


if __name__ == "__main__":
    main()
