# Chapter 8 — Conclusion

**What is built:**

- **mergeVAT** (published, NAFIPS 2025/26) — exact VAT/iVAT from a few thousand points to 135,000+
- **MoG FIS construction** (paper in preparation) — interpretable classifiers/regressors in seconds, no stochastic search in the fit
- **One shared ridge-TSK solver** across flat and hierarchical models

**What is proposed:**

- Ch. 5 → Ch. 6 link: topological memberships consumed by the inference models *(the bridge, unfinished by design — and the most interesting remaining work)*
- G2 (real non-coordinate data), G9 (the estimator head-to-head), G4b (the exact/approximate rivals)

**Three honest limits:** the speedup claim rests on the slowest fuzzy baseline (14×–194× vs ANFIS / GA-FIS) · coordinate-free claim rested on synthetic data until the G2 measurement · interpretability payoff described, not quantified.

*Tally from the harness:* one goal refuted, one crossover retracted, one model found diverging on one split in ten, one headline speedup cut by an order of magnitude. Repetition is not the same thing as coverage.
