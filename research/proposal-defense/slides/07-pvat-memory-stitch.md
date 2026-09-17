# Chapter 3 — Memory, Stitching, and the Estimator

- **In-place permutation** — the reorder overwrites the original matrix (cycle-walking): 2–3 matrices → 1, lifting the size ceiling by $\sqrt{3}$ (and $\sqrt{2}$ more at float32), exact at no cost
- **Divide-and-conquer stitch** — blocks merged at bounded cost: principled stitch recovers ARI **1.00** where naive concatenation collapses to **0.47**
- **`IVATMeans`** — a clustering *estimator* read off the exact iVAT image: deterministic, initialization-free, membership + assignment from one fit (head-to-head vs FCM/k-means = Goal G9)
- Works on **any** dissimilarity matrix — non-metric (DTW) included; 3 real DTW sets reordered exactly (Table 3.7)

![](fig/03-memory-ceiling.png){width=48%}
