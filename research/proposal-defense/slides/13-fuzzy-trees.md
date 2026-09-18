# Chapter 6 — Soft Fuzzy Trees

CART-style recursive partition, with two differences:

- **Soft splits** — a point flows down multiple paths with graded membership (weights form a partition of unity)
- **Ridge-TSK leaves** — each leaf is the shared closed-form model, not a constant

Reads as a short list of IF–THEN rules over *named* variables:

> Concrete splits on **cement**, then on **age at exactly 28** — the standard curing mark, recovered without being told.

The trade is explicit, and the accuracy half is not favorable: under one protocol the tree does not beat the flat model on Concrete — it buys an **explicit, readable decision path**, not accuracy.

![](fig/06-fuzzy-tree.png){width=66%}
