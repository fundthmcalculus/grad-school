# Chapter 4 — Fast FIS from Mixtures of Gaussians

Build the model **answer-first**: fit a per-feature Gaussian mixture *per output class* — memberships and rules come straight off the fit.

- **No grid** — rules are built per class, not per combination: a $K$-class problem → ~$K$ rules (never a product over inputs)
- **No GA, no gradient descent in the construction** — the consequent fit is closed-form (the ridge-TSK solver of Ch. 6)
- Bonus: explicit class rules → the complement is automatically a **"none of the above" rule** — open-set detection that can say *why* it fired

![](fig/04-mog-classification.png){width=58%}
