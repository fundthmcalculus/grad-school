# Chapter 2 — Background: Takagi–Sugeno–Kang Systems

A TSK rule: *IF* $x_1$ is $A_1$ *and* $x_2$ is $A_2$ *THEN* $y = f(x)$

- Consequent $f$ is a function (constant → polynomial), not a fuzzy set
- Output = weighted average of the rules: $\hat{y}=\dfrac{\sum_r w_r f_r(x)}{\sum_r w_r}$
- Two knobs: **antecedents** (membership functions) and **consequents**

![](fig/02-tsk-inference.png){width=46%}
