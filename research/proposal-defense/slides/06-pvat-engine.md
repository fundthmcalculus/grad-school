# Chapter 3 — mergeVAT: Exact VAT and iVAT at Scale

Classical VAT: $O(N^3)$, 2–3 matrices of memory. A two-stage reorder fixes it:

1. **Priority-queue reorder** → $O(N^2 \log N)$ (published, NAFIPS 2025/26)
2. **Active-set reorder** → $O(N^2)$ (unpublished; ~11× faster again)

Same ordering, **exact** (bit-identical to the serial reference at float64):

| | classical | mergeVAT |
|---|---|---|
| 4,096 points | 124 s | 0.23 s |
| 58,000 points (shuttle) | ~4 days | **~1 min** |

![](fig/03-three-arm-seconds.png){width=78%}
