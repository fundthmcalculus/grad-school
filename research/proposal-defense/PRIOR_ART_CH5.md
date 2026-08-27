# Chapter 5 — Prior-Art Head-to-Head

**Date:** 2026-08-27
**Discharges:** the "prior-art head-to-head … plus a formal literature search to bound the novelty claim" owed in §5.5, and the overlap-management item in §5.6.
**Method:** four parallel reviews, one per claimed contribution, each required to verify citations against primary sources and to flag anything verified only second-hand. Verdicts below are deliberately unkind: a false "novel" is far more expensive than a false "known".

---

## Verdict summary

| # | Claim as written in §5.1 | Verdict |
|---|---|---|
| 1 | The minimax transform as the preprocessing step that makes relational fuzzy clustering succeed on non-convex data | **DEAD.** Published, by Bezdek. |
| 2 | A selection rule making the cluster count an *output*, by gating persistence and covering the data | **DEAD as framed**, and the "covering" half was factually wrong about our own code. A narrow empirical claim survives. |
| 3 | A multi-scale extension recovering a hierarchy of partitions and discovering how many scales there are | **DEAD as framed.** One mechanism + one setting survive. |
| 4 | Membership functions extracted natively from each piece of the hierarchy, as FIS antecedents | **NARROWED BUT ALIVE — the only surviving pillar.** |
| — | "The whole VAT family stops at assessment; nobody turns VAT structure into a model" | **FALSE as written.** A narrower version is true and defensible. |

**The chapter currently claims four contributions. After this review it has one, plus two methods-section results.** That is a smaller chapter, but a defensible one, and the surviving claim is the one §5.1 already calls "the piece I consider most novel."

---

## 1. Minimax transform for relational fuzzy clustering — DEAD

Every link is published, and the chain has been assembled twice:

- D\* is the **subdominant ultrametric** = single-linkage cophenetic = MST bottleneck: Johnson, *Psychometrika* 32(3):241–254, 1967; Jardine & Sibson, *Mathematical Taxonomy*, 1971; Leclerc, *Math. Sci. Hum.* 73:5–37, 1981. Axiomatic modern treatment: **Carlsson & Mémoli, *JMLR* 11:1425–1470, 2010.**
- Finite ultrametrics embed isometrically in Euclidean space: **Lemin, *Soviet Math. Dokl.* 32(3):740–744, 1985**; Timan & Vestfrid 1983; Fiedler, *ELA* 3:23–30, 1998 (sharp dimension); Faver et al., *Glasgow Math. J.* 56(3):519–535, 2014 (via strict *p*-negative type for all *p* ≥ 0).
- The composition — minimax D\* of an *arbitrary non-metric* D yields a PSD centred Gram matrix: **Chehreghani, AAAI-17, Theorem 3**; journal version *Machine Learning* 109:2063–2097, 2020.
- Using it in relational FCM *in place of* the beta-spread: **Khalilia, Bezdek, Popescu & Keller, *Pattern Recognition* 47(12):3920–3930, 2014** (iRFCM). Compares five Euclideanizations including beta-spread; finds the subdominant ultrametric best. **Co-authored by Bezdek**, who wrote the beta-spread paper.

**Do not claim.** Cite as a design justification: we use D\*, so the beta-spread of Hathaway & Bezdek (1994) is provably inert, following Khalilia et al.

Two corrections this produced:
- iRFCM does **not** replace D with the ultrametric. It uses it as the **spread matrix** δ in the Benasseni–Dosse–Joly generalized spread, `D̂ = D² + γ·δ`, `δ = u(D²)` (Benasseni, Dosse & Joly, *J. Classification* 24(1):33–51, 2007). Any description of iRFCM-SU as "replace D by its ultrametric" is wrong.
- **u(D²) = u(D)² exactly.** The bottleneck transform is a max–min composition and monotone non-decreasing maps commute with both max and min, so u(f(D)) = f(u(D)) entrywise. Verified numerically to 0.0 and bit-exact (a non-monotone f breaks it, as it must). Squaring before or after is therefore equivalent — but the *surrounding pipeline* is not invariant, because iRFCM's γ is fitted against D². This produced a live defect report against the library (clustering#89).

## 2. Persistence-gated set-cover selection — DEAD as framed

**The framing was wrong about our own code.** `select_coverage_cover` was documented as covering points "tolerating overlap", "overlap allowed", "not a partition". It cannot overlap. Dendrogram nodes form a **laminar** family, so greedy-by-uncovered-gain always takes a maximal eligible node, after which every eligible descendant has gain 0 and is dropped by the stopping rule. **Measured: 0 overlapping pairs across 14 datasets**, including a real 5,000-point ECG5000 DTW matrix at k=25. Corrected in code with a regression test (`test_selection_antichain.py`).

What it therefore computes is exactly a **local cut through the hierarchy**, which is the object already optimised by:

- **Campello, Moulavi, Zimek & Sander, "A framework for semi-supervised and unsupervised optimal extraction of clusters from hierarchies", *DAMI* 27(3):344–371, 2013** (FOSC) — "the optimal extraction of flat clusterings from local cuts through cluster hierarchies", solved globally, k emergent.
- **Campello, Moulavi & Sander, PAKDD 2013**; **Campello, Moulavi, Zimek & Sander, *ACM TKDD* 10(1):5, 2015** (HDBSCAN\*).
- Earlier: **Sander, Qin, Lu, Niu & Kovarsky, PAKDD 2003** (automatic significant-cluster extraction; also proves reachability plots and dendrograms carry the same information); **Stuetzle, *J. Classification* 20(1):25–47, 2003**; **Stuetzle & Nugent, *JCGS* 19(2):397–418, 2010**; **Gupta, Liu & Ghosh, *IEEE/ACM TCBB* 7(2):223–237, 2010** (Auto-HDS — greedy stability-ranked selection); **Langfelder, Zhang & Horvath, *Bioinformatics* 24(5):719–720, 2008** (Dynamic Tree Cut).

**And the density-free distinction is a parameter setting inside the pre-empting paper.** TKDD 2015 **Corollary 3.5**: at *mpts* ∈ {1,2}, mutual reachability equals the original distance and HDBSCAN\* *is* single-linkage on those distances. So our hierarchy is a one-parameter special case of published machinery.

Other ingredients, all published:
- Enumerating every internal node with (birth, death) has a name — the **mergegram**: **Elkin & Kurlin, MFCS 2020, LIPIcs 170:56** (Definition 3.4), which also proves it strictly dominates 0-D persistence and is stable. **Cite this: it gives us a stability theorem for free.**
- Statistical gating of a per-node persistence-like statistic, k an output, possibly zero: **Sun & Krasnitz, *BMC Genomics* 15:1000, 2014** (TBEST — persistence normalised by parent height, tested at *all* internal nodes); Munneke et al., *Genetics* 170(4), 2005; Suzuki & Shimodaira, *Bioinformatics* 22(12), 2006 (pvclust); Liu et al., *JASA* 103(483), 2008 (SigClust).
- Persistence-gap selection of k: **Chazal, Guibas, Oudot & Skraba, *JACM* 60(6):41, 2013** (ToMATo — "look for the largest drop in the sequence of prominences"); AuToMATo (Huber, Kališnik & Schnider, TMLR 2025); **Bois, Tervil & Oudre, *Front. Appl. Math. Stat.*, 2024, DOI 10.3389/fams.2024.1260828** — connectivity-flavoured (not density), auto-thresholded, k an output, explicit non-coverage. **Closest non-HDBSCAN analogue; must be engaged.**
- The MAD gate is the Iglewicz–Hoaglin modified Z-score dropped into ToMATo's slot. An engineering choice, not a contribution.

### What survives: a measured gate comparison

Run at the setting where HDBSCAN\* reduces to our hierarchy (`min_samples=1`, EOM extraction), on identical inputs:

| dataset | ours (ARI / k) | HDBSCAN EOM (ARI / k) |
|---|---|---|
| concentric_rings | **1.000** / 2 | 0.061 / 30 |
| bridged_gaussians | **0.982** / 3 | 0.452 / 14 |
| varying_density | **0.980** / 3 | 0.750 / 9 |
| edit_strings | **1.000** / 3 | 0.647 / 7 |
| graph_communities | **0.331** / 4 | 0.168 / 8 |
| two_gaussians, three_clusters_tree, chain_then_ring, dtw_traces, hamming_categorical | 1.000 | 1.000 |
| multi_scale_hierarchy | 0.551 / 3 | **1.000** / 6 |
| cosine_topics | 0.803 / 3 | **0.903** / 3 |

The lever is not "density-free" — it is that EOM's stability is a **size-weighted sum in λ = 1/ε units** (TKDD Eq. 3), while ours is **unweighted persistence in raw dissimilarity units**. EOM over-segments badly here (30 clusters on rings). Campello explicitly invites the substitution: "the excess of mass adopted in this article is by no means the only possible measure for cluster stability that can be used in our framework."

**Claimable:** an unweighted, absolute-scale, robust-outlier stability measure inside Campello's framework, measured against EOM on non-metric dissimilarities. A methods-section result, not a pillar.

## 3. Multi-scale band discovery — DEAD as framed

The structural antecedent is **VDBSCAN** (Liu, Zhou & Wu, ICSSSM 2007, DOI 10.1109/ICSSSM.2007.4280175): read several density scales off the knees of a sorted k-distance curve, then run the base clusterer once per scale. Option D is that recipe on a dendrogram's height axis. Same family: MDBSCAN, DMDBSCAN.

Also settled:
- Scale-count as an output: **Leung, Zhang & Xu, *IEEE TPAMI* 22(12):1396–1410, 2000** ("the critical scale is estimated by analyzing the distribution of cluster lifetime in the scale space" — lifetime-based scale selection, in 2000); **Peixoto, *Phys. Rev. X* 4:011047, 2014** (MDL level selection, no spurious modules on noise); Arenas et al., *NJP* 10:053039, 2008; Delvenne et al., *PNAS* 107(29), 2010; Traag et al., *Sci. Rep.* 3:2930, 2013; Jeub et al., *Sci. Rep.* 8:3259, 2018; Guigues et al., *IJCV* 68(3), 2006 (scale-sets).
- Gap-in-merge-heights → a cut: textbook (Manning, Raghavan & Schütze, *IR*, Ch. 17; the inconsistency coefficient in Jain & Dubes 1988, shipped in scipy/MATLAB).
- Birth height ≈ inverse local density: Hartigan, *JASA* 76(374), 1981; Stuetzle & Nugent 2010. (Note Hartigan also proved single linkage is *not* consistent for d > 1 — relevant to how hard we lean on the density reading.)
- Flat-can't-represent-nested: Kleinberg 2002 / Carlsson & Mémoli 2010; and it is HDBSCAN's own founding motivation ("any choice of cut line is … a single fixed density level").

**What survives:** (a) the specific mechanism — log-relative gaps on the birth heights of the *persistence-significant blocks only*, plus a containment-aware band-merge — no exact match found; (b) exact degeneration to the flat selector on single-scale data and zero bands on noise, a clean checkable property; (c) **the setting**: VDBSCAN/OPTICS/HDBSCAN all require a kNN density estimate in a metric space. On a non-metric D\* where no density estimator is available, birth height is the only density proxy there is. That is a methodological reason rather than a re-derivation, and it is the strongest angle.

**Required before claiming:** HDBSCAN `leaf` extraction and a `dbscan_clustering(eps)` sweep on the nested [8,4,2] synthetic. If leaf recovers 8 and an eps sweep recovers 4 and 2, a reviewer will ask why an eps sweep is not the whole contribution. Also position against OPTICS ξ-extraction ("what several DBSCANs would produce at varying density thresholds") and Rolle & Scoccola, *JMLR* 25(258), 2024 (multiparameter persistence — whose scale selection is *human-guided*, which is our opening).

## 4. Membership functions from merge heights — NARROWED, AND THE ONE SURVIVING PILLAR

Pre-empted components:
- **HDBSCAN soft clustering already derives membership from merge heights** — `hdbscan.all_points_membership_vectors`, combining a distance-to-exemplar term with a λ/merge-height term.
- **Bonis & Oudot, arXiv:1406.7130** ("A Fuzzy Clustering Algorithm for the Mode Seeking Framework"; **no evidence it was ever published in PRL — cite as preprint only**) is *explicitly persistence-based and does use birth/death heights*: "a fuzzy generalization of the ToMATo algorithm which relies on the concept of prominence". **§5.2's current distinction — "a deterministic ramp read off the merge heights, not a random-walk hitting probability" — concedes too little and misses the load-bearing difference.**
- **Harada & Nishino, "Multi-dimensional fuzzy set identification using persistent homology", IFSA-SCIS 2017, DOI 10.1109/IFSA-SCIS.2017.8023281** — persistence sets the threshold defining a fuzzy set's **support**. One citation in OpenAlex: easy to miss, easy for a committee member to find. **Full text unread — obtain before writing the novelty paragraph.**
- Coordinate-based MF generation from clusters is of course standard: Chiu (subtractive clustering, 1994); Yager & Filev (mountain method); Sugeno & Yasukawa, *IEEE T-FS* 1993; Jang (ANFIS, 1993).

### Why it nevertheless survives

**HDBSCAN's soft clustering cannot run on a dissimilarity matrix.** Verified directly against the library, which refuses with its own warning:

> `UserWarning: Cannot generate prediction data for non-vector space inputs -- access to the source data rather than mere distances is required!`
> `AttributeError: No prediction data was generated`

With coordinates it returns a (120, 3) membership matrix; with the *same data* as a precomputed distance matrix it fails. Our construction runs on the precomputed matrix and returns per-block ramps parameterised by each block's own (birth, death). That is the gap, stated by the pre-empting library itself.

And the VAT side of the house is crisp-only. Every VAT-derived partition found — CLODD (*IJIS* 24(5), 2009), DBE (*TKDE* 21(3), 2009), CCE (*Soft Computing* 13(12), 2009), aVAT (PAKDD 2010), SpecVAT-partitioning (*TKDE* 22(10), 2010), sVAT-SL (ISSNIP 2013), clusiVAT (IEEE Big Data 2013), FensiVAT (*TKDE* 31(4), 2019), ConiVAT (2020), kernel-iVAT+CER (*KAIS* 66(11), 2024), HaVAT (CODS-COMAD 2024) — is **crisp**. Bezdek's own framing of cluster analysis asks which objects belong to which cluster *"and to what degree"* (Havens & Bezdek, *TKDE* 24(5):813–822, 2012); no work in that family answers the second half. FIM-VAT (Achary, Kachroo & Rathore, IJCNN 2024) is the field reaching for interpretability and stopping at post-hoc feature importances — the best paper to position against.

### The claim, as it must now be written

> A fuzzy set whose support width **and** slope are both derived, per cluster and deterministically, from that cluster's own persistence *h_d − h_b*, computed from a dissimilarity matrix alone and consumed as an FIS antecedent.

Three-way distinction from Bonis & Oudot, replacing §5.2's single axis: (i) their core width is a **global** threshold τ/2, identical for every cluster, not each cluster's own lifetime; (ii) their graded part is a **stochastic** process with temperature β, not a deterministic slope; (iii) their memberships form a **partition of unity**, not independent fuzzy sets combinable by a t-conorm. **(i) is the load-bearing one and currently goes unmade.**

**The sentence "no prior work connects persistence to fuzzy membership" must not appear in the dissertation.**

## Also required: a framing correction

§5.2's "the whole VAT family stops at *assessment* … no one turns the VAT structure into a fuzzy model" must be split. The first clause is false (see the eleven papers above, most Bezdek-co-authored); the second is true as far as searching can establish. Suggested replacement:

> VAT operates on exactly a dissimilarity matrix and finds structure of any shape, built on single-linkage/MST connectivity rather than centroids. The family did not stop at assessment: CLODD extracts an aligned crisp partition from the VAT image; DBE, CCE, aVAT and HaVAT automate the cluster count; clusiVAT, FensiVAT and ConiVAT are end-to-end clustering algorithms on VAT orderings. **But every one outputs a crisp partition.** Bezdek's own framing asks which objects belong to which cluster *and to what degree* — and no work in this family answers the second half, let alone turns the answer into a fuzzy inference system.

Also worth stating, and not found published: **iVAT's minimax transform *is* the subdominant-ultrametric Euclideanization that Khalilia et al. independently found best of five for relational FCM.** Two Bezdek-co-authored lines converge on the same max–min operator without either citing the other's use of it.

---

## Verification caveats

Read before any of this enters a bibliography.

- **Primary sources read directly:** HDBSCAN\* TKDD 2015 (institutional-repository PDF; Corollary 3.5 and Eq. 3 quoted verbatim); ToMATo JACM PDF; mergegram arXiv:2007.11278; TBEST preprint; Havens et al. AMAI 2009; Havens & Bezdek iVAT TKDE 2012 preprint; ConiVAT arXiv; Iredale et al. FUZZ-IEEE 2017; the complete iRFCM codebase; the `hdbscan` package source and its runtime behaviour.
- **Metadata verified, full text paywalled:** Khalilia et al. 2014 (claims about its content come from the abstract plus the authors' own code — **get the PDF**); Hathaway & Bezdek 1994 and Hathaway, Davenport & Bezdek 1989 (the admissibility condition is corroborated by the iRFCM code, but the quotation is second-hand); CLODD, DBE, SpecVAT, clusiVAT, FensiVAT, kernel-iVAT, HaVAT, FIM-VAT; VDBSCAN (abstract elided by publisher); Guigues 2006; Leung 2000.
- **Second-hand only — verify before citing:** Lemin 1985 and Timan & Vestfrid 1983 (transcribed from Fiedler 1998 and Faver et al. 2014, consistent across both); Jain & Dubes 1988; Kleinberg 2002; Chaudhuri & Dasgupta 2010; Hopkins & Skellam 1954; Iglewicz & Hoaglin; Sledge et al. 2009 ("count-only" is inferred from title and grouping).
- **Unresolved, worth a look:** Carpio & Duro, "Hierarchical topological clustering", arXiv:2601.00892 — recent, on-topic, no trustworthy quotes obtained.
- **Highest-value remaining check:** an **IEEE Xplore full-text** search for "persistent homology" + "membership function". Harada & Nishino surfaced from exactly that corner (short IFSA/FUZZ-IEEE papers, thin abstract indexing), so one or two more may exist. Neither review could reach Xplore full text.
- Negative evidence on record: arXiv API `all:"persistent homology" AND all:"fuzzy membership"` → 0 (control query returns results, so the syntax works); OpenAlex `"membership function"` + `"persistent homology"` → 0; `"fuzzy inference system"` + `"topological data analysis"` → 0. Absence of a search hit is weak evidence.

## Code corrections this review forced

| finding | action |
|---|---|
| `select_coverage_cover` cannot overlap; docstring said it could | fixed + `test_selection_antichain.py` (14 tests) |
| `nerfcm.py` documents the beta-spread condition backwards | filed clustering#89 |
| `IVATMeans._fit_relational` feeds `u(D)` where the RFCM dual needs squared distances | filed clustering#89 |
| "canonical Euclideanizer" and "one-sided metric repair" claimed as findings in the notes | corrected with citation trails (PR #179) |
