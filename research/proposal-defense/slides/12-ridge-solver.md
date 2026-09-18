# Chapter 6 — One Solver, Every Model

Key observation: a TSK output is **linear in the consequent coefficients** for fixed firing strengths.

→ Consequents need no search: they are the closed-form solution of a firing-weighted ridge least-squares.

One solver, reused everywhere: the flat FIS (Ch. 4), the leaves of a soft fuzzy tree, the mixture-of-experts extension.

![](fig/06-ridge-solver.png){width=62%}
