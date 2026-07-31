# Bibliography

The references for this proposal are maintained as a single BibTeX file, [`../references.bib`](../references.bib), and the final document's reference list is generated from it. This page is a reading guide to that file, grouped the way the chapters use it; it is not a second copy of the data.

A note on provenance and verification. The fuzzy-inference, fuzzy-tree, and mixture-of-experts entries were assembled and adversarially verified during the hierarchical-FIS novelty review. The VAT/clustering, persistence, optimization, and interpretability entries were added while drafting this proposal and then verified against Crossref, arXiv, and DBLP in a dedicated pass (2026-07-31). Every entry now carries a `[V]` (verified) or `[S]` (seminal, standard DOI) marker in the `.bib`; there are no unresolved entries. Notable resolutions from that pass:

- The two former placeholders are real papers: **eVAT** is Meng & Yuan, *"Parallel Edge-Based Visual Assessment of Cluster Tendency on GPU"* (Int. J. Data Science and Analytics, 2018); **Fast-VAT** is Avinash & Lachheb, arXiv:2507.15904 (2025).
- **Bonis–Oudot** (the nearest precedent for Chapter 5) was confirmed as Pattern Recognition Letters vol. 102, pp. 37–43, 2018.
- **AuToMATo**'s title and authors were corrected (Huber, Kališnik, Schnider; *"An Out-Of-The-Box…"*, TMLR 2025).
- **ConiVAT** remains an arXiv preprint only (arXiv:2008.09570) — no journal DOI exists, so none is asserted.

One proof-stage item remains: confirm the "Kališnik" accent survives the final BibTeX/LaTeX encoding.

## Reading guide by area

- **Fuzzy inference systems, trees, and mixtures of experts** — the TSK form and ANFIS (`takagi1985fuzzy`, `jang1993anfis`, `wu2020optimize`); rule generation from data (`sugeno1993qualitative`, `wang1992generating`, `chiu1994fuzzy`, `abe1995method`); genetic fuzzy systems (`cordon2001genetic`, `herrera2008genetic`, `alcala2007rule`); fuzzy and soft trees (`janikow1998fuzzy`, `yuan1995induction`, `suarez1999globally`, `olaru2003complete`, `medina2001backpropagation`, `fumanal2025fast`); hierarchical mixtures and TSK-fusion (`jordan1994hierarchical`, `wu2020functional`, `raju1991hierarchical`, `zhou2017deep`, `zhang2023tsk`); Ruspini partitions and interpretable design (`ruspini1969new`, `guillaume2004generating`, `deoliveira1999semantic`, `guillaume2006expert`, `guillaume2011learning`, `nanfack2022constraint`); cascades (`viola2001rapid`, `cavalin2019confusion`); universal approximation and the interpretability caveat (`wang1998universal`, `wang1999analysis`, `joo2002universal`, `magdalena2018do`, `higashi1983measures`).
- **VAT / iVAT / cluster tendency** — `bezdek2002vat`, `wang2010ivat`, `havens2012efficient`, `kumar2016clusivat`, `kumar2020vatsurvey`, `rathore2020conivat`, and the fast-VAT competitors `meng2018evat`, `avinash2025fastvat`.
- **MST, single-linkage, and Fuzzy C-Means** — `prim1957shortest`, `gower1969mst`, `zahn1971graph`, `dunn1973fuzzy`, `bezdek1981pattern`, `hathaway1994nerf`, `bien2011hierarchical`, `tibshirani2001gap`, `cate1977insitu`.
- **Persistence and topological data analysis** — `chazal2013persistence` (ToMATo), `bonis2018fuzzy` (the nearest precedent for Chapter 5), `automato2024`.
- **Optimization, TSP, and quality-diversity** — `lin1973effective`, `helsgaun2000effective`, `croes1958method`, `dorigo1996ant`, `kennedy1995particle`, `deb2002nsga2`, `mouret2015illuminating`, `vassiliades2018cvt`.
- **Interpretability / XAI** — `lundberg2017shap`.

## Not intellectual sources

The two reference dissertation proposals in `research/proposal-defense/` (Pickering 2024, Arnett 2018) informed the *document structure and format only*. They are not cited as sources for any method or framing in this work, which was developed independently.

---

*Source of truth: `../references.bib`. Citation shorthand in the prose chapters (e.g. `[Bezdek and Hathaway 2002]`) will be replaced by `\cite{}` keys against this file when the document is assembled in LaTeX.*
