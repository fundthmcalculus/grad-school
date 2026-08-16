# Bibliography

The references live in a single BibTeX file, [`../references.bib`](../references.bib), from which the final reference list is generated. This page is a reading guide to that file, grouped the way the chapters use it, not a second copy of the data.

**Provenance.** The fuzzy-inference, fuzzy-tree, and mixture-of-experts entries were assembled and adversarially verified during the hierarchical-FIS novelty review. The VAT/clustering, persistence, optimization, and interpretability entries were added while drafting, then verified against Crossref, arXiv, and DBLP (2026-07-31). Notable resolutions:

- Both former placeholders are real papers: **eVAT** is Meng & Yuan, *"Parallel Edge-Based Visual Assessment of Cluster Tendency on GPU"* (Int. J. Data Science and Analytics, 2018), and **Fast-VAT** is Avinash & Lachheb, arXiv:2507.15904 (2025).
- **Bonis–Oudot** (the nearest precedent for Chapter 5) was confirmed as Pattern Recognition Letters vol. 102, pp. 37–43, 2018.
- **AuToMATo**'s title and authors were corrected (Huber, Kališnik, Schnider; *"An Out-Of-The-Box…"*, TMLR 2025).
- **ConiVAT** remains an arXiv preprint only (arXiv:2008.09570). No journal DOI exists, so none is asserted.

## The state of the file, counted

The file holds **70 entries: 45 `[V]`, 23 `[S]`, and 2 still `[?]`.** An earlier tally elsewhere in the repo read "47 `[V]` + 24 `[S]` across 70 entries, zero unresolved" — wrong on every figure and summing to 71 — and has since been corrected (`CHECKLIST.md` §F) to match this page, which is the source of truth for the count. The `.bib` header legend calls the `[?]` entries "resolved … and promoted to `[V]`", untrue of the two below and needing the same correction.

Two distinctions matter more than the arithmetic. **`[V]` is metadata-verified, not content-verified**: author list, title, venue, year and DOI resolve against a publisher index, and nothing is claimed about the paper having been read. And `[V]` can coexist with an unresolved field inside the entry; two do.

**Five entries are not clean, and four of them are load-bearing.**

| Entry                   | Marker | The gap                                                                                                                         | Stakes                                                                                                                                                           |
|-------------------------|--------|---------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mullner2011modern`     | `[?]`  | arXiv record (`arXiv:1109.2378`), `journal = {arXiv preprint}`, no volume or pages                                              | the more serious `[?]`: load-bearing for §3.3.1's space-bound correction, cited prominently and early per Chapter 9 §9.3, so an unresolved venue is not cosmetic |
| `parveen2013pvat`       | `[V]`  | author placeholder `{Parveen, [given name] and Sreevalsan-Nair, Jaya}`; note asks to verify that given name and the exact title | the `pVAT` name collision §3.3.1 turns on, which forced this work's rename to mergeVAT; it will be read closely, and it cannot yet be typeset                    |
| `deshpande2024scalable` | `[V]`  | note reads `FULL TEXT NOT YET READ` / `BLOCKING: obtain full text`                                                              | see below                                                                                                                                                        |
| `kumar2016incvat`       | `[V]`  | note asks to verify the title                                                                                                   | smallest of the five, the only genuine proof-stage item                                                                                                          |

**`deshpande2024scalable` in full.** Metadata confirmed (*Information Sciences* 664:120324, DOI 10.1016/j.ins.2024.120324); content not, which makes it the concrete case of `[V]` meaning metadata and nothing more. `CHECKLIST` **E8** lists the full-text read as blocking: if the paper already states the $O(N)$-workspace result for VAT itself and not only for MST-iVAT, the Chapter 9 §9.3 note has no contribution left. Chapter 3's unoccupied-regime argument turns on what this line of work achieves, so until the read is done the entry supports a citation, not a characterization, and this page states no result of the paper's as settled. It is also the formerly broken citation, now resolved as a *citation*: Chapters 2 and 3 once had only "[*Information Sciences* 2024]", a journal name and a year with no author, title, or DOI, for the kd-tree memory methods that §2.2 and §3.2 now cite by name.

**One further open item is substantive.** `CHECKLIST` **E7** records a **Zhang-2023 attribution error**: the HFIS reference material misattributes `zhang2023tsk` to "H. Wang et al." when the first author is Zhang. Chapter 6 §6.4 calls it "a small attribution fix in the references", which undersells it. The `.bib` entry carries the correct author list and an inline comment saying so, so the fix belongs in the prose and README that cite it, not in the file. But a misattributed survey citation is what a committee notices; it does not belong alongside an accent check. So: **two open items, one substantive (E7) and one cosmetic** (confirm the "Kališnik" accent survives the final BibTeX/LaTeX encoding), plus the five entry-level gaps above.

## Reading guide by area

- **Fuzzy inference systems, trees, and mixtures of experts.** The TSK form and ANFIS (`takagi1985fuzzy`, `jang1993anfis`, `wu2020optimize`); rule generation from data (`sugeno1993qualitative`, `wang1992generating`, `chiu1994fuzzy`, `abe1995method`, `jang1993functional`); genetic fuzzy systems (`cordon2001genetic`, `herrera2008genetic`, `alcala2007rule`); fuzzy and soft trees (`janikow1998fuzzy`, `yuan1995induction`, `suarez1999globally`, `olaru2003complete`, `medina2001backpropagation`, `fumanal2025fast`); hierarchical mixtures and TSK-fusion (`jordan1994hierarchical`, `wu2020functional`, `raju1991hierarchical`, `zhou2017deep`, `zhang2023tsk`); Ruspini partitions and interpretable design (`ruspini1969new`, `guillaume2004generating`, `deoliveira1999semantic`, `guillaume2006expert`, `guillaume2011learning`, `nanfack2022constraint`); cascades (`viola2001rapid`, `cavalin2019confusion`); universal approximation and the interpretability caveat (`wang1998universal`, `wang1999analysis`, `joo2002universal`, `magdalena2018do`, `higashi1983measures`).
- **VAT / iVAT / cluster tendency.** `bezdek2002vat`, `wang2010ivat`, `havens2012efficient`, `kumar2016clusivat`, `kumar2016incvat`, `kumar2020vatsurvey`, `rathore2020conivat`; the fast-VAT competitors `meng2018evat`, `avinash2025fastvat`, and `deshpande2024scalable` (the closest prior art, and the source of Chapter 3 §3.2's narrowed niche claim); and `parveen2013pvat`, the prior method of that name whose collision with mine is discussed in §3.3.1.
- **MST, single-linkage, and Fuzzy C-Means.** `prim1957shortest`, `gower1969mst`, `zahn1971graph`, `havens2009disguise` and `mullner2011modern` (together the two papers from which §3.3.1's space-bound correction can be assembled, which is why §9.3 scopes that note as an audit and not a result), `dunn1973fuzzy`, `bezdek1981pattern`, `hathaway1994nerf`, `bien2011hierarchical`, `tibshirani2001gap`, `cate1977insitu`.
- **Persistence and topological data analysis.** `chazal2013persistence` (ToMATo), `bonis2018fuzzy` (the nearest precedent for Chapter 5), `automato2024`.
- **Optimization, TSP, and quality-diversity.** `lin1973effective`, `helsgaun2000effective`, `croes1958method`, `dorigo1996ant`, `kennedy1995particle`, `deb2002nsga2`, `mouret2015illuminating`, `vassiliades2018cvt`.
- **Interpretability / XAI.** `lundberg2017shap`.

---

*Source of truth: `../references.bib`. Citation shorthand in the prose chapters (e.g. `[Bezdek and Hathaway 2002]`) will be replaced by `\cite{}` keys against this file when the document is assembled in LaTeX.*
