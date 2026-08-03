# Bibliography

The references for this proposal are maintained as a single BibTeX file, [`../references.bib`](../references.bib), and the final document's reference list is generated from it. This page is a reading guide to that file, grouped the way the chapters use it; it is not a second copy of the data.

A note on provenance and verification. The fuzzy-inference, fuzzy-tree, and mixture-of-experts entries were assembled and adversarially verified during the hierarchical-FIS novelty review. The VAT/clustering, persistence, optimization, and interpretability entries were added while drafting this proposal and then verified against Crossref, arXiv, and DBLP in a dedicated pass (2026-07-31). Every entry now carries a `[V]` (verified) or `[S]` (seminal, standard DOI) marker in the `.bib`; there are no unresolved entries. Notable resolutions from that pass:

- The two former placeholders are real papers: **eVAT** is Meng & Yuan, *"Parallel Edge-Based Visual Assessment of Cluster Tendency on GPU"* (Int. J. Data Science and Analytics, 2018); **Fast-VAT** is Avinash & Lachheb, arXiv:2507.15904 (2025).
- **Bonis–Oudot** (the nearest precedent for Chapter 5) was confirmed as Pattern Recognition Letters vol. 102, pp. 37–43, 2018.
- **AuToMATo**'s title and authors were corrected (Huber, Kališnik, Schnider; *"An Out-Of-The-Box…"*, TMLR 2025).
- **ConiVAT** remains an arXiv preprint only (arXiv:2008.09570) — no journal DOI exists, so none is asserted.

One item remains open, and it is cosmetic:

- A proof-stage check: confirm the "Kališnik" accent survives the final BibTeX/LaTeX encoding.

The formerly broken citation is resolved. Chapters 2 and 3 once referred to the kd-tree memory methods only as "[*Information Sciences* 2024]" — a journal name and a year, with no author, title, or DOI. That entry is now `deshpande2024scalable`, verified, and cited by name in §2.2 and §3.2. It matters more than most, because it is the line achieving *sub-quadratic* memory for Euclidean data and is therefore load-bearing for Chapter 3's argument about which regime is unoccupied.

## Reading guide by area

- **Fuzzy inference systems, trees, and mixtures of experts** — the TSK form and ANFIS (`takagi1985fuzzy`, `jang1993anfis`, `wu2020optimize`); rule generation from data (`sugeno1993qualitative`, `wang1992generating`, `chiu1994fuzzy`, `abe1995method`, `jang1993functional`); genetic fuzzy systems (`cordon2001genetic`, `herrera2008genetic`, `alcala2007rule`); fuzzy and soft trees (`janikow1998fuzzy`, `yuan1995induction`, `suarez1999globally`, `olaru2003complete`, `medina2001backpropagation`, `fumanal2025fast`); hierarchical mixtures and TSK-fusion (`jordan1994hierarchical`, `wu2020functional`, `raju1991hierarchical`, `zhou2017deep`, `zhang2023tsk`); Ruspini partitions and interpretable design (`ruspini1969new`, `guillaume2004generating`, `deoliveira1999semantic`, `guillaume2006expert`, `guillaume2011learning`, `nanfack2022constraint`); cascades (`viola2001rapid`, `cavalin2019confusion`); universal approximation and the interpretability caveat (`wang1998universal`, `wang1999analysis`, `joo2002universal`, `magdalena2018do`, `higashi1983measures`).
- **VAT / iVAT / cluster tendency** — `bezdek2002vat`, `wang2010ivat`, `havens2012efficient`, `kumar2016clusivat`, `kumar2016incvat`, `kumar2020vatsurvey`, `rathore2020conivat`; the fast-VAT competitors `meng2018evat`, `avinash2025fastvat`, and `deshpande2024scalable` (the closest prior art, and the source of Chapter 3 §3.2's narrowed niche claim); and `parveen2013pvat`, the prior method of that name whose collision with mine is discussed in §3.3.1.
- **MST, single-linkage, and Fuzzy C-Means** — `prim1957shortest`, `gower1969mst`, `zahn1971graph`, `havens2009disguise` and `mullner2011modern` (together the two papers from which §3.3.1's space-bound correction can be assembled, which is why §9.3 scopes that note as an audit rather than a result), `dunn1973fuzzy`, `bezdek1981pattern`, `hathaway1994nerf`, `bien2011hierarchical`, `tibshirani2001gap`, `cate1977insitu`.
- **Persistence and topological data analysis** — `chazal2013persistence` (ToMATo), `bonis2018fuzzy` (the nearest precedent for Chapter 5), `automato2024`.
- **Optimization, TSP, and quality-diversity** — `lin1973effective`, `helsgaun2000effective`, `croes1958method`, `dorigo1996ant`, `kennedy1995particle`, `deb2002nsga2`, `mouret2015illuminating`, `vassiliades2018cvt`.
- **Interpretability / XAI** — `lundberg2017shap`.

## On the reference proposals

Two dissertation proposals from this department sit in `research/proposal-defense/` (Pickering 2024, Arnett 2018). They informed the *document structure and format* of this proposal. Neither is a source for any method or framing here; this work was developed independently, and in particular the interpretability position in §2.6 is my own and predates any awareness of Pickering's treatment.

One qualification, so the record is exact. Arnett (2018) *is* cited once, in §2.1, because the FIS constraint set I adopt — triangular membership functions, Ruspini partition, product t-norm, weighted-average defuzzification — appears there in the same combination. That citation marks a parallel in the same department, not a debt: the constraints themselves trace to Ruspini (1969) and de Oliveira (1999), which is what §2.1 cites for them. I would rather note the overlap than have a committee member who has read both documents notice it unremarked.

---

*Source of truth: `../references.bib`. Citation shorthand in the prose chapters (e.g. `[Bezdek and Hathaway 2002]`) will be replaced by `\cite{}` keys against this file when the document is assembled in LaTeX.*
