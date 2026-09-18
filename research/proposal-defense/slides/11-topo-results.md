# Chapter 5 — Preliminary Results

- **The transform works:** concentric rings ARI 0.02 → **1.00** on the minimax-transformed matrix (k = 2 discovered)
- **Multi-scale recovery:** three-level hierarchy, flat mean ARI 0.576 → per-level max **1.000** at each level — the partitions returned go from *one* to *three*, each correct at its own granularity (a structural difference, not an accuracy lift)
- **Scale (ten seeds):** `many_scale` recovers [8, 4, 2] at ARI 1.00 for every n from 100 → 5,000; partition-of-unity error at machine precision
- **Coordinate-free is now measured (G2):** on real DTW data, ECG5000 set-cover ARI **0.715** vs NERFCM-given-k 0.593 — while discovering k
- **Recorded failures:** the bridge (0.001, k = 3 vs 2); selection is at parity with HDBSCAN\*, not ahead — the selection machinery is *machinery*, the memberships are the contribution

![](fig/05-battery.png){width=92%}
