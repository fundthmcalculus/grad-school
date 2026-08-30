# Clustering novelty & prior-art write-ups

The literature search, prior-art assessment, and novelty argument behind the
`tribble-clustering` work. Moved here from
[`fundthmcalculus/clustering`](https://github.com/fundthmcalculus/clustering)
in that repo's issue #97: these are thesis/dissertation material, and that
repo is meant to be a shippable package rather than a record of the research
around it. They join the experiment spikes in `../`, which moved for the same
reason in clustering PR #53.

| file | what it is |
|---|---|
| `novel-niche.md` | The niche argument — where the defensible contribution sits, three ranked instantiations, and the experiment that characterises the envelope. |
| `novelty-review.md` | Prior-art review of the VAT/iVAT implementation, `IVATMeans`, and FCM. |
| `performance-novelty.md` | The performance-integration framing: what the integrated pipeline is and which four literature lines it unifies. |
| `vat-tsp-prior-art.md` | Prior-art review of the VAT↔TSP thread (seriation↔TSP, warm start, MST-seeded ACO, cluster blocking). |
| `vat-tsp-session2-novelty.md` | Prior-art review of the second VAT↔TSP stream (GPU VAT, dual-VAT construction, intersection-driven uncrossing, variable-depth LK). |
| `bibliography.md` | Curated, DOI-verified bibliography backing all of the above. |
| `popmusic-spacefilling.md` | Plan for POPMUSIC + space-filling baselines and a scale benchmark, written for a follow-up run on unrestricted hardware. |

## Two conventions carried over from the old location

**Prior-art PDFs.** Three open-access PDFs are cited by `bibliography.md` and
`novelty-review.md`. They stay in the clustering repo under
[`docs/papers/`](https://github.com/fundthmcalculus/clustering/tree/main/docs/papers)
— they are ~6.8 MB and one of them is cited from a shipped library docstring
(`conivat.py`), so that repo needs them regardless. References here point at
them by URL rather than duplicating them.

**`docs/sources/`.** Several entries name a PDF under `docs/sources/`. That was
a *git-ignored* scratch cache in the clustering repo during the retrieval
sessions — those files were never committed to either repo. Treat such a
reference as "an open-access PDF exists at the cited DOI," not as a path.

## What stayed behind

The clustering repo keeps
[`docs/design-notes.md`](https://github.com/fundthmcalculus/clustering/blob/main/docs/design-notes.md),
extracted from `novel-niche.md` §1/§3/§6. It carries only the part the library
itself needs — the geometry argument behind `IVATMeans(refine=...)` and
`dissimilarity_power` — with the thesis framing removed, because those
docstrings cite it and a package's docs should not depend on a coursework repo.
