# IVATMeans and clustering for embeddings: discovery and comparison

Status: discovery pass on subsampled corpora, run on a 4-core, 15 GB host. Not a final result. Full-data runs belong on the larger machine.
Generated from `reproduce/experiments/ivat_embeddings/` (commits up to `run_fcm_embedding.py`). Raw results: `ivat_embed_raw.json`, `ivat_metric_raw.json`, `fcm_embed_raw.json`.

## 1. Setup

- **Corpora (labelled, text):** AG News (4 classes), Emotion (6), 20 Newsgroups (20). Each: 10,000 training documents as the base set, 500 held-out test documents as queries.
- **Embeddings:** all-MiniLM-L6-v2 (384-d, L2-normalised), and TF-IDF (min_df 2, stop words removed) → TruncatedSVD 128-d, L2-normalised.
- **Empty-document rule:** documents with no surviving TF-IDF terms were replaced from the same split (base: Emotion 1, 20 Newsgroups 296; queries: Emotion 1, 20 Newsgroups 16). Both embeddings use the same final documents.
- **Methods:** k-means (k-means++), E-means (5 epochs default init; 25 epochs k-means++ init), FCM (m = 2), IVATMeans (medoid refine). k = number of classes. Baselines over 10 seeds; IVATMeans run once (no random state).

## 2. Clustering quality (ARI against document labels)

| Corpus / embedding | k-means | E-means 5 ep | E-means 25 ep | FCM | IVATMeans |
|---|---|---|---|---|---|
| AG News / MiniLM | 0.578 | 0.608 | 0.628 | 0.380 | 0.037 |
| AG News / TF-IDF | 0.299 | 0.277 | 0.231 | 0.205 | 0.038 |
| Emotion / MiniLM | 0.044 | 0.045 | 0.046 | 0.028 | 0.018 |
| Emotion / TF-IDF | 0.005 | 0.002 | 0.003 | 0.007 | 0.004 |
| 20 Newsgroups / MiniLM | 0.368 | 0.352 | 0.369 | 0.066 | 0.055 |
| 20 Newsgroups / TF-IDF | 0.208 | 0.191 | 0.201 | 0.053 | 0.045 |

Emotion is at chance for every method in both embeddings, so its ARI does not separate the methods.

## 3. Cost

- On 20 Newsgroups / MiniLM (N = 10,000): k-means 6.0 s, E-means 5 ep 0.27 s, E-means 25 ep 1.3 s, FCM 0.35 s, IVATMeans 5.2 s.
- IVATMeans is the slowest method at N = 10,000 and cannot run at N ≥ 32,000 on this host (predicted peak 17 bytes × N² above 80% of RAM; 32,000 was OOM-killed in a dry run).

## 4. Similarity search (IVF-style, cells from each method)

Recall@10 against exact cosine search over the original embedding, at nprobe = 1 (share of base scanned in brackets):

| Corpus / embedding | k-means | E-means 5 ep | FCM | IVATMeans |
|---|---|---|---|---|
| AG News / MiniLM | 0.868 (27%) | 0.868 (26%) | 0.879 (43%) | 0.532 (33%) |
| AG News / TF-IDF | 0.793 (31%) | 0.815 (36%) | 0.703 (35%) | 0.547 (27%) |
| Emotion / MiniLM | 0.590 (17%) | 0.586 (17%) | 0.645 (36%) | 0.511 (32%) |
| Emotion / TF-IDF | 0.754 (30%) | 0.731 (26%) | 0.806 (51%) | 0.522 (17%) |
| 20 Newsgroups / MiniLM | 0.695 (6%) | 0.690 (6%) | 0.877 (48%) | 0.256 (6%) |
| 20 Newsgroups / TF-IDF | 0.649 (8%) | 0.663 (9%) | 0.757 (43%) | 0.348 (7%) |

Read recall at equal scan, not raw recall. FCM's higher recall comes from scanning 35–50% of the base set, against 6–30% for the others.

## 5. Distance metric (IVATMeans)

The `metric` argument does not change IVAT's ordering. The sweep patched the distance stage to test Euclidean, Manhattan, and cosine.

- Cosine equals Euclidean on all six combinations, as expected for unit-normalised embeddings.
- Manhattan is worse than Euclidean on five of six (ARI and recall). The exception is AG News / MiniLM (ARI 0.051 vs 0.037).
- No metric choice brings IVATMeans near the baselines.

## 6. FCM membership vectors as embeddings

Each document is represented by its k-dimensional fuzzy membership vector, and search runs on that vector. The sweep covered m ∈ {1.1, 1.5, 2.0, 3.0}.

- Recall@10 against the original space stays at 0.001–0.033 for every m and every combination. Membership vectors do not preserve the original neighbourhood structure, so they are unusable as a search embedding.
- Clustering from membership argmax matches the hard FCM result (ARI 0.33–0.39 on AG News / MiniLM, near chance on Emotion, 0.05–0.10 on 20 Newsgroups).
- Conclusion: FCM is useful for search only through its cells (section 4), at a higher scan cost.

## 6b. PCA before clustering (k-means, FCM, E-means, IVATMeans)

PCA fitted on the base set (centred, no whitening) to 32 and 64 dimensions, against the original dimension. Ten seeds for k-means, FCM and E-means; IVATMeans is one run per setting.

| Corpus / embedding | k-means (orig / 32 / 64) | E-means 5 ep (orig / 32 / 64) | FCM (orig / 32 / 64) | IVATMeans (orig / 32 / 64) |
|---|---|---|---|---|
| AG News / MiniLM | 0.578 / 0.606 / 0.572 | 0.608 / 0.625 / 0.624 | 0.380 / 0.329 / 0.340 | 0.037 / 0.060 / 0.095 |
| AG News / TF-IDF | 0.299 / 0.262 / 0.238 | 0.277 / 0.245 / 0.295 | 0.205 / 0.256 / 0.233 | 0.038 / 0.014 / 0.017 |
| Emotion / MiniLM | 0.044 / 0.047 / 0.046 | 0.045 / 0.045 / 0.044 | 0.028 / 0.027 / 0.029 | 0.018 / 0.011 / 0.008 |
| Emotion / TF-IDF | 0.005 / 0.001 / 0.003 | 0.002 / 0.004 / 0.002 | 0.007 / 0.008 / 0.007 | 0.004 / 0.004 / −0.003 |
| 20 Newsgroups / MiniLM | 0.368 / 0.358 / 0.368 | 0.352 / 0.357 / 0.374 | 0.066 / 0.061 / 0.061 | 0.055 / 0.121 / 0.081 |
| 20 Newsgroups / TF-IDF | 0.208 / 0.224 / 0.207 | 0.191 / 0.223 / 0.210 | 0.053 / 0.046 / 0.046 | 0.045 / 0.048 / 0.045 |

- **FCM stays collapsed at every dimension.** Its memberships stay near uniform, and its clusters stay empty or oversized.
- **PCA costs k-means and E-means little.** Both hold their ARI within the seed spread on MiniLM. On TF-IDF, k-means drops on AG News (0.299 to 0.238).
- **IVATMeans improves most under PCA on MiniLM,** but stays far below the baselines. The best case is 20 Newsgroups at 32 dimensions, 0.121 against k-means' 0.358. Its clusters are no longer collapsed (sizes 147 to 1,187), so the gap is in the partition itself, not an empty-cluster artefact.
- **IVATMeans is still near chance on Emotion,** and on TF-IDF it stays below 0.05 in every setting.

Tables: `reproduce/outputs/table_pca_before_clustering.md` and `reproduce/outputs/table_pca_ivat.md`. Raw: `pca_compare_raw.json` and `pca_ivat_raw.json`. Figure: `reproduce/outputs/figures/fig_pca_before_clustering.png`.

## 6c. Diagnostics: why the variations are so bad

Figures: `figures/fig_diag_distance_contrast`, `figures/fig_diag_fcm_membership`, `figures/fig_diag_ivat_profile`. Source: `run_diagnostics_16d.py`, raw in `diag16_raw.json`.

- **Distance contrast.** Relative contrast (dmax − dmin)/dmin, median over 2,000 sampled documents: MiniLM 384-d is 0.53–0.59 on all three corpora, so pairwise distances concentrate near the median. TF-IDF SVD-16 (direct) is 3.3–8.9. Contrast is necessary for neighbour search but does not by itself produce a better partition: Emotion has the highest contrast and still sits at chance for every method.
- **FCM collapse.** The largest-membership histogram sits on the uniform 1/k spike for all MiniLM and TF-IDF-128 settings, and for the PCA-16 TF-IDF embeddings on AG News and 20 Newsgroups. Only the 16-dim TF-IDF embeddings on Emotion spread out, and that does not improve ARI.
- **IVATMeans cuts land at the tail.** The k−1 largest link-distance jumps along the VAT order sit at positions 1,480–1,500 of a 1,500-document subsample, in every panel. The gap rule therefore trims outlying points rather than separating classes, which explains the lopsided partitions. The iVAT images do show diagonal blocks on the TF-IDF embedding, so the structure exists but the cut rule does not select it.

Direct 16-dim comparison (ARI; `table_direct16_comparison`):

| Corpus | Embedding | k-means | E-means 5 ep | FCM | IVATMeans |
|---|---|---|---|---|---|
| AG News | MiniLM PCA-16 | 0.607 | 0.621 | 0.333 | 0.087 |
| AG News | TF-IDF SVD-16 direct | 0.402 | 0.435 | 0.207 | 0.025 |
| 20 Newsgroups | MiniLM PCA-16 | 0.315 | 0.317 | 0.061 | 0.144 |
| 20 Newsgroups | TF-IDF SVD-16 direct | 0.225 | 0.221 | 0.048 | 0.107 |

MiniLM has no native 16-dim form, so its row is PCA-16. The TF-IDF direct row is the only true direct 16-dim embedding.

## 6d. Log transforms

Source: `run_log_transform.py`, table `table_log_transform`, raw `log_transform_raw.json`. Baselines use the same documents. Baseline values are repeated from sections 2 and 6b for comparison.

| Corpus | Case | k-means | E-means 5 ep | FCM | IVATMeans |
|---|---|---|---|---|---|
| AG News | MiniLM baseline | 0.578 | 0.608 | 0.380 | 0.037 |
| AG News | MiniLM, signed log | 0.577 | 0.604 | 0.379 | 0.037 |
| AG News | MiniLM, IVAT on log1p distances | — | — | — | 0.037 |
| AG News | TF-IDF SVD-16 baseline | 0.402 | 0.435 | 0.207 | 0.025 |
| AG News | TF-IDF SVD-16, log1p before SVD | 0.410 | 0.424 | 0.203 | 0.021 |
| AG News | TF-IDF SVD-16, IVAT on log1p distances | — | — | — | 0.025 |
| Emotion | MiniLM baseline | 0.044 | 0.045 | 0.028 | 0.018 |
| Emotion | MiniLM, signed log | 0.042 | 0.042 | 0.029 | 0.016 |
| Emotion | MiniLM, IVAT on log1p distances | — | — | — | 0.018 |
| Emotion | TF-IDF SVD-16 baseline | 0.003 | 0.003 | 0.002 | 0.002 |
| Emotion | TF-IDF SVD-16, log1p before SVD | 0.003 | 0.003 | 0.002 | 0.004 |
| Emotion | TF-IDF SVD-16, IVAT on log1p distances | — | — | — | 0.002 |
| 20 Newsgroups | MiniLM baseline | 0.368 | 0.352 | 0.066 | 0.055 |
| 20 Newsgroups | MiniLM, signed log | 0.363 | 0.357 | 0.066 | 0.068 |
| 20 Newsgroups | MiniLM, IVAT on log1p distances | — | — | — | 0.055 |
| 20 Newsgroups | TF-IDF SVD-16 baseline | 0.225 | 0.221 | 0.048 | 0.107 |
| 20 Newsgroups | TF-IDF SVD-16, log1p before SVD | 0.220 | 0.220 | 0.048 | 0.092 |
| 20 Newsgroups | TF-IDF SVD-16, IVAT on log1p distances | — | — | — | 0.107 |

- **No logarithm improves any method meaningfully.** The one apparent gain, IVATMeans on 20 Newsgroups with MiniLM under the signed log (0.055 to 0.068), comes from a single run and should be treated as noise.
- **A log on the IVAT distances changes nothing** on any corpus. The minimax ordering is invariant to monotone transforms, so the partitions are identical, and only the gap-based cut selection could move, and it didn't.
- **Log1p before the TF-IDF SVD** lowers IVATMeans on 20 Newsgroups (0.107 to 0.092) and leaves the other methods within seed spread.

## 7. Findings

1. On these corpora, IVATMeans is the weakest method on clustering quality, cost, and search recall. It did not improve on k-means or E-means under any metric.
2. E-means matches k-means on ARI and search recall, at a fraction of the time with 5 epochs. It is the practical choice for embedding clustering here.
3. FCM's cells give the best raw recall, but at roughly 2× the scan of k-means for the same recall.
4. Membership-vector embeddings are not a search representation.
5. Embedding geometry matters: the TF-IDF embedding has zero-vector documents, which produce distance ties. Removing them was required before any TF-IDF comparison was valid.
6. PCA to 32 or 64 dimensions does not rescue FCM. Its memberships stay uniform, and its clusters stay empty or oversized at every dimension.
7. PCA is close to free for k-means and E-means on the MiniLM embeddings, and E-means stays at or near the top of the three.
8. PCA helps IVATMeans most on MiniLM (20 Newsgroups 0.055 to 0.121 at 32 dimensions), but it still trails the baselines by a wide margin.

## 8. Caveats and open items

- Subsamples of 10,000 documents and 500 queries. Full-data results may differ.
- CPU only. No GPU or on-device runs.
- Timings are on a 4-core VM and are not device-independent.
- The compiled IVATMeans kernel crashes (double free) on NaN distance input. It should validate input. Report to the library owner.
- Next step on the larger machine: rerun `build_embeddings.py` with `REPRO_EMBED_NBASE` raised, then the three scripts in `reproduce/experiments/ivat_embeddings/`. Check the IVATMeans memory ceiling there before choosing N.
