# FLM inference demo

## Browser-only version (no server, no Python at run time): `demo/web/`

The whole demo, including generation, runs in the browser tab. A JavaScript port of the model's step form (`web/flm.js`: byte-level BPE, causal convolution, fuzzy delta-rule mixer, first-order TSK feed-forward rules, tied embedding) reads the weights from `web/models/`. Same page as below: prompt, length / temperature / top-k / seed, per-piece alternatives, the rules behind each piece, structure card, rule timeline, rule browser and rule inspector.

```bash
cd experiments/fuzzy-language-model/demo/web && python -m http.server 8000     # then open http://localhost:8000
python demo/build_standalone.py [--all]      # or one self-contained file, demo/dist/index.html, that opens from a file:// address
```

* **GitHub Pages:** `demo/dist/index.html` is committed and self-contained (3.3 MB, both sub-100K models, no external requests), so publishing the `demo/dist` folder (or the repository with Pages pointed at it) serves the demo statically.
* Models: the 96K 3B-character reference (386 KB of weights), the 98K tiered-embedding model (the best sub-100K likelihood, 0.961 BPC at 1B characters, seed 1), and the 1.32M model (fp16, 2.6 MB; over the 100K cap, included for comparison). Generation costs 0.2-1 ms per word piece.
* `python demo/export_web.py` rebuilds `web/models/` from the checkpoints (run from the experiment directory). It densifies the tiered embedding (input row and tied output row of a token are both `model.embed(token)`; checked against the original model before writing) and precomputes the rule views (`rule_views.py`: membership histograms over 48 validation windows). For the 1.32M model only the most used rules carry data histograms; any other rule shows its membership functions from the weights.
* Checks: `node demo/web/test_parity.mjs` compares the engine with the torch logits (max |dlogit| 2e-5 for the 96K model, 1e-4 for the tiered one, 1e-2 for the fp16 1.32M one; argmax identical) and the tokenizer with the HF tokenizer on 8 strings (`python demo/export_tok_check.py` regenerates the references). Before the server was removed, a greedy 12-piece run also reproduced all 36 rule rows of its rule explanations (layer, rule, contribution, raised pieces).
* Limits: the seeded sampler is JavaScript's (it does not reproduce torch's random stream); prompt plus generated text is capped at the model's context (512 word pieces).

## What the page shows
* Pick a model, type a prompt, set length, temperature, top-k and seed, press Generate.
* **Architecture diagrams** (three, drawn from the selected model's real sizes): the whole model, one fuzzy memory head, one fuzzy feed-forward rule.
* Hover or click any word piece: the five most likely alternatives at that point, and the strongest fuzzy FFN rules that pushed that piece (gate, signed effect on the piece's score, and what the rule raises most overall). The rule view is the exact direct-path decomposition of `flm/rule_dist.py`.
* **Model structure card**, **rule timeline** (which rules fired for each generated piece, per layer), **most used rules** per layer and a **rule inspector** (membership function on each antecedent dimension, data distribution overall and where the rule fires, contexts where it fires most).
* Pieces that repeat a three-piece run already in the text get a dotted underline, and the line under the story gives the share. In the rated stories, repetition tracked lower coherence scores at r of about -0.3 (RESULTS.md, rule-view analysis).
* The rule view explains how a piece's score is assembled, not why the story has or lacks a plot; a rule's listed pushes are not necessarily the piece it helped produce (the output is a sum of many rules).

The former Python inference server (`demo/server.py`, torch on localhost) and its static sample output were removed: the browser page does everything they did, and `rule_views.py` keeps only the part the exporter needs.
