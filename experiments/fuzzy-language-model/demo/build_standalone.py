"""Build ONE self-contained HTML file of the in-browser demo (opens from a file:// address, works offline).

    python demo/build_standalone.py [--all]  ->  demo/dist/index.html

The weights and rule views of the models are embedded as base64. By default only the sub-100K models go in (about 2.5 MB);
The demo carries only all-fuzzy models (both are under the cap, so ``--all`` changes nothing). The committed copy (demo/dist/index.html) lets GitHub Pages serve it statically; rebuild it and commit it when demo/web changes.
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
    remote = None
    if (
        "--remote-dir" in sys.argv
    ):  # models hosted beside the page, not embedded: their index entries (remote: true) and files in <dir>/models
        remote = Path(sys.argv[sys.argv.index("--remote-dir") + 1])
        index += json.loads((remote / "models" / "index.json").read_text())
    if not everything:
        index = [m for m in index if m["params"] < 100_000 or m.get("remote")]
    files = {"tokenizer.json": WEB / "tokenizer.json"}
    for tok in sorted({m["tok"] for m in index if m["tok"] != "bpe1024"}):
        name = f"tokenizer-{tok.removeprefix('bpe')}.json"
        files[name] = WEB / name
    for m in index:
        if m.get("remote"):
            continue
        for ext in ("json", "bin", "rules.json"):
            files[f"models/{m['id']}.{ext}"] = WEB / "models" / f"{m['id']}.{ext}"
    emb = {k: base64.b64encode(p.read_bytes()).decode() for k, p in files.items()}
    emb["models/index.json"] = base64.b64encode(json.dumps(index).encode()).decode()
    html = (WEB / "index.html").read_text(encoding="utf-8")
    js = (WEB / "flm.js").read_text(encoding="utf-8")
    sjs = (WEB / "surgery.js").read_text(encoding="utf-8")
    html = html.replace(
        '<script src="flm.js"></script>\n<script src="surgery.js"></script>',
        "<script>\n" + js + "\n</script>\n<script>\n" + sjs + "\n</script>",
    )
    html = html.replace(
        "<!--EMBED-->", "<script>window.FLM_EMBED = " + json.dumps(emb) + ";</script>"
    )
    if (
        "--out" in sys.argv
    ):  # write elsewhere (the gh-pages build with a hosted model keeps the committed dist unchanged)
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        DIST.mkdir(exist_ok=True)
        out = DIST / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"{out}  {out.stat().st_size / 1e6:.2f} MB  ({len(index)} models)")
    if remote:
        print(
            "hosted beside the page (publish next to index.html):",
            [
                str(p.relative_to(remote))
                for p in sorted((remote / "models").glob("*"))
                if p.name != "index.json" and not p.name.endswith(".check.json")
            ],
        )


if __name__ == "__main__":
    main()
