# FLM inference demo

A small local web page for trying the fuzzy language models.

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
