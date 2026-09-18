# Chapter 5 — Topological Membership Generation *(the bridge)*

Chapter 4 assumed coordinates and blobs. Real data often arrives as a **dissimilarity matrix only** (DTW, edit distance, kernels), with non-blob structure (rings, bridges).

**The contribution:** membership functions read off the *merge heights* of the minimax hierarchy — no coordinates, no Gaussian assumption.

- cluster count $k$ = **output**, discovered by a persistence-gated set-cover
- multi-scale: a stack of partitions, one per true scale — a flat method can only return one
- the bridge between the clustering half and the fuzzy-modeling half of the dissertation

![](fig/05-minimax-transform.png){width=90%}
