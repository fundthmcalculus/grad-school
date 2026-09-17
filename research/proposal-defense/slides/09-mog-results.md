# Chapter 4 — Results

| dataset | task | train time | accuracy |
|---|---|---:|---|
| PhiUSIIL (235,795 × 47) | 2-class phishing | **0.13 s** | 0.440 (leak-free — see note) |
| RT-IOT2022 (12-class) | scale target | **3.64 s** | 0.927 (RF ref. 0.998) |
| Concrete (1,030) | regression | seconds | $R^2$ = **0.861** (full 2nd order) |

Speed vs. the fuzzy baselines it displaces (ANFIS, GA-tuned FIS): **14× to 194×** — read the range, not the headline.

*Honest cell:* dropping `URLSimilarityIndex` (a target leak, AUC 0.996 on its own) moves PhiUSIIL below the majority baseline — what survives is the **rule count and training time**, the 194× speedup.

![](fig/04-speedup.png){width=82%}
