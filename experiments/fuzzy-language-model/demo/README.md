# FLM inference demo

## Browser-only version (no server, no Python at run time): `demo/web/`

The whole demo, including generation, runs in the browser tab. A JavaScript port of the model's step form (`web/flm.js`: byte-level BPE, causal convolution, fuzzy delta-rule mixer, first-order TSK feed-forward rules, tied embedding) reads the weights from `web/models/`. Same page as below: prompt, length / temperature / top-k / seed, per-piece alternatives, the rules behind each piece, structure card, rule timeline, rule browser and rule inspector.

```bash
cd experiments/fuzzy-language-model/demo/web && python -m http.server 8000     # then open http://localhost:8000
python demo/build_standalone.py [--all]      # or one self-contained file, demo/dist/index.html, that opens from a file:// address
```

* **GitHub Pages:** `demo/dist/index.html` is committed and self-contained (4.3 MB, six all-fuzzy models, no external requests), so publishing the `demo/dist` folder (or the repository with Pages pointed at it) serves the demo statically. The demo carries only all-fuzzy models: the non-fuzzy reference, tiered and 1.32M models were removed on 2026-10-21 (still in git history).
* **All-fuzzy model:** `flm-allfuzzy` (99,019 parameters, BPE3072, 300M characters) runs on its own engine (`AllFuzzyModel` in `web/flm.js`, a port of `allfuzzy/allfuzzy_lm.py`; parity with the numpy model 6e-7); `python demo/export_web_allfuzzy.py` exports it. The page draws it with every block component fuzzy.
* Models: `flm-allfuzzy-f7` and `flm-allfuzzy-f7-s1` (99,882 parameters; the shape-search winner: BPE6144, 640 crisp head tokens, tail rank 8, width 32; 300M characters, 0.9660 and 0.9686 bits per character; the default), `flm-allfuzzy-1b` and `flm-allfuzzy-1b-s1` (99,019 parameters, the earlier shape, 1B characters, 0.974 and 0.976), `flm-allfuzzy` (the earlier shape at 300M characters) and `flm-allfuzzy-embed` (both token tables fuzzy). The F7 models use `tokenizer-6144.json`; exports: `python allfuzzy/export_weights.py <ckpt> allfuzzy/<name>.npz --vocab 6144` then `python demo/export_web_allfuzzy.py --npz allfuzzy/<name>.npz --id <id> --checkpoint <ckpt> --vocab 6144 --label ...`. Generation costs 0.2-1 ms per word piece.
* `python demo/export_web.py` is the exporter for the removed non-fuzzy models (it rewrites `web/models/index.json` with them); do not run it for the demo as it stands. `python demo/export_web_allfuzzy.py` rebuilds the two all-fuzzy models from their `allfuzzy/*.npz` weights.
* Checks: `node demo/web/test_parity.mjs` compares the engine with the numpy all-fuzzy model's logits for the models in `web/models/index.json` and the tokenizer with the HF tokenizer on 8 strings (`python demo/export_tok_check.py` regenerates the references). The model checks are the same ones `allfuzzy/check_parity.py` runs.
* Limits: the seeded sampler is JavaScript's (it does not reproduce torch's random stream); prompt plus generated text is capped at the model's context (512 word pieces).

## What the page shows
* Pick a model, type a prompt, set length, temperature, top-k and seed, press Generate.
* **Architecture diagrams** (three, drawn from the selected model's real sizes): the whole model, one fuzzy memory head, one fuzzy feed-forward rule.
* Hover or click any word piece: the five most likely alternatives at that point, and the strongest fuzzy FFN rules that pushed that piece (gate, signed effect on the piece's score, and what the rule raises most overall). The rule view is the exact direct-path decomposition of `flm/rule_dist.py`.
* **Model structure card**, **rule timeline** (which rules fired for each generated piece, per layer), **most used rules** per layer and a **rule inspector** (membership function on each antecedent dimension, data distribution overall and where the rule fires, contexts where it fires most).
* Pieces that repeat a three-piece run already in the text get a dotted underline, and the line under the story gives the share. In the rated stories, repetition tracked lower coherence scores at r of about -0.3 (RESULTS.md, rule-view analysis).
* The rule view explains how a piece's score is assembled, not why the story has or lacks a plot; a rule's listed pushes are not necessarily the piece it helped produce (the output is a sum of many rules).

The former Python inference server (`demo/server.py`, torch on localhost) and its static sample output were removed: the browser page does everything they did, and `rule_views.py` keeps only the part the exporter needs.
