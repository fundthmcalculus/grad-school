# FLM inference demo

## Browser-only version (no server, no Python at run time): `demo/web/`

The whole demo, including generation, runs in the browser tab. A JavaScript port of the model's step form (`web/flm.js`: byte-level BPE, causal convolution, fuzzy delta-rule mixer, first-order TSK feed-forward rules, tied embedding) reads the weights from `web/models/`. Same page as below: prompt, length / temperature / top-k / seed, per-piece alternatives, the rules behind each piece, structure card, rule timeline, rule browser and rule inspector.

```bash
cd experiments/fuzzy-language-model/demo/web && python -m http.server 8000     # then open http://localhost:8000
python demo/build_standalone.py [--all]      # or one self-contained file, demo/dist/flm-demo.html, that opens from a file:// address
```

* Models: the 96K 3B-character reference (386 KB of weights), the 98K tiered-embedding model (the best sub-100K likelihood, 0.961 BPC at 1B characters, seed 1), and the 1.32M model (fp16, 2.6 MB; over the 100K cap, included for comparison). Generation costs 0.2-1 ms per word piece.
* `python demo/export_web.py` rebuilds `web/models/` from the checkpoints (run from the experiment directory). It densifies the tiered embedding (input row and tied output row of a token are both `model.embed(token)`; checked against the original model before writing) and precomputes the rule views of the Python demo (membership histograms over 48 validation windows) with the same functions. For the 1.32M model only the most used rules carry data histograms; any other rule shows its membership functions from the weights.
* Checks: `node demo/web/test_parity.mjs` compares the engine with the torch logits (max |dlogit| 2e-5 for the 96K model, 1e-4 for the tiered one, 1e-2 for the fp16 1.32M one; argmax identical) and the tokenizer with the HF tokenizer on 8 strings (`python demo/export_tok_check.py` regenerates the references). A greedy 12-piece run also reproduced all 36 rule rows of the Python demo (layer, rule, contribution, raised pieces).
* Differences from the Python demo: the seeded sampler is JavaScript's (the same seed gives different text than the Python demo); prompt plus generated text is capped at the model's context (512 word pieces) instead of sliding the window.

## Local-server version (Python, torch)
A small local web page for trying the fuzzy language models, served by Python and torch.

```bash
cd experiments/fuzzy-language-model
.venv/bin/python demo/server.py            # then open http://127.0.0.1:8765
```

* Pick a checkpoint (any BPE1024 `.pt` under `outputs/`; the committed 3B-character 96K model is first), type a prompt, set length, temperature, top-k and seed, press Generate.
* Hover or click any word piece: the five most likely alternatives at that point, and the strongest fuzzy FFN rules that pushed that piece (gate, signed effect on the piece's score, and what the rule raises most overall).
  The rule view uses the exact direct-path decomposition in `flm/rule_dist.py`; turn it off for speed, or it switches itself off for models whose FFN is not the first-order TSK form.
* **Model structure card:** layer count and what each layer contains (causal convolution, fuzzy recurrent mixer heads and memory rules, TSK feed-forward rules and antecedent size).
* **Rule timeline:** a heatmap of which feed-forward rules fired (gate sign and size) for each generated piece, per layer; click a rule label to inspect it, or a column to select that piece.
* **Most used rules** per layer, with the pieces each raises, and a **rule inspector**: the rule's membership function on each of its 16 antecedent dimensions (centre and width, the product of which is the firing), the data distribution along each dimension overall and where the rule fires, ordered by selectivity, plus the contexts where it fires most. Endpoints: `/structure`, `/rules`, `/rule`, `/mixrule` (memory-rule centres and widths).
* CPU only, localhost only (binds 127.0.0.1), standard library HTTP server plus torch. Generation is serialised by a lock; the first request for a model loads it.
* Pieces that repeat a three-piece run already in the text get a dotted underline, and the line under the story gives the share. In the rated stories, repetition (of rule sets and of token runs) tracked lower coherence scores at r of about -0.3 (RESULTS.md, rule-view analysis).
* The rule view explains how a piece's score is assembled, not why the story has or lacks a plot; a rule's listed pushes are not necessarily the piece it helped produce (the output is a sum of many rules).
