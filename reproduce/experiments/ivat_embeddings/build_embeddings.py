"""Build the labelled text corpora and embeddings for the IVATMeans embedding study.

For each corpus: a base set of N_BASE labelled training documents, and N_QUERY held-out
documents from the test split, used as similarity-search queries. Each is embedded two
ways so the result does not depend on one embedding family:

  st     all-MiniLM-L6-v2 sentence embeddings (384-d), L2-normalised
  tfidf  TF-IDF (fit on the base set) -> TruncatedSVD(128), L2-normalised

Empty-document handling (2026-11-18 rebuild): a document whose TF-IDF vector has no
surviving terms (every token stop-worded or below min_df) is all zeros. Its cosine
distance is undefined, and the resulting ties make IVATMeans' partition depend on the
tie-breaking implementation (the compiled library and a scipy patch disagree). Such
documents are replaced by fresh documents from the same split, and the vectorizer is
refit until no base or query document is empty. The same final documents are used for
both embeddings, so the two are compared on identical data. The replaced count is printed.

Output: one .npz per corpus in the embedding cache (gitignored; ~15 MB each).
Cache directory: $REPRO_EMBED_CACHE (default ~/.cache/grad-school/embeddings).

Run (embedding env, outside the repo):
    UV_PROJECT_ENVIRONMENT=~/.cache/grad-school/venv-embed uv run --no-project --python 3.12 \\
      --with sentence-transformers --with datasets --with scikit-learn --with numpy \\
      python reproduce/experiments/ivat_embeddings/build_embeddings.py
"""

from __future__ import annotations

import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

CACHE = os.environ.get(
    "REPRO_EMBED_CACHE", os.path.expanduser("~/.cache/grad-school/embeddings")
)
N_BASE = int(os.environ.get("REPRO_EMBED_NBASE", "10000"))
N_QUERY = 500
SEED = 0
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TFIDF_DIM = 128

CORPORA = {
    "ag_news": "fancyzhx/ag_news",
    "emotion": "dair-ai/emotion",
    "newsgroups20": "SetFit/20_newsgroups",
}


def load_split(repo, split):
    from datasets import load_dataset

    d = load_dataset(repo, split=split)
    return list(d["text"]), np.asarray(d["label"])


def make_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        min_df=2, stop_words="english", max_features=50000, sublinear_tf=True
    )


def select_nonempty(rng, tr_text, te_text, ib, iq):
    """Replace empty-vector documents until the base and query TF-IDF rows are all non-zero."""
    replaced = {"base": 0, "query": 0}
    while True:
        vec = make_vectorizer()
        tf_b = vec.fit_transform([tr_text[i] for i in ib])
        tf_q = vec.transform([te_text[i] for i in iq])
        empty_b = np.flatnonzero(np.asarray(tf_b.getnnz(axis=1)) == 0)
        empty_q = np.flatnonzero(np.asarray(tf_q.getnnz(axis=1)) == 0)
        if empty_b.size == 0 and empty_q.size == 0:
            return vec, tf_b, tf_q, ib, iq, replaced
        if empty_b.size:
            pool = np.setdiff1d(np.arange(len(tr_text)), ib)
            new = rng.choice(pool, empty_b.size, replace=False)
            for j, v in zip(empty_b, new):
                ib[j] = int(v)
            replaced["base"] += int(empty_b.size)
        if empty_q.size:
            pool = np.setdiff1d(np.arange(len(te_text)), iq)
            new = rng.choice(pool, empty_q.size, replace=False)
            for j, v in zip(empty_q, new):
                iq[j] = int(v)
            replaced["query"] += int(empty_q.size)


def main():
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import TruncatedSVD

    os.makedirs(CACHE, exist_ok=True)
    model = SentenceTransformer(ST_MODEL)
    rng = np.random.default_rng(SEED)
    for name, repo in CORPORA.items():
        tr_text, tr_y = load_split(repo, "train")
        te_text, te_y = load_split(repo, "test")
        ib = [int(v) for v in rng.choice(len(tr_text), N_BASE, replace=False)]
        iq = [int(v) for v in rng.choice(len(te_text), N_QUERY, replace=False)]
        vec, tf_b, tf_q, ib, iq, replaced = select_nonempty(
            rng, tr_text, te_text, ib, iq
        )
        print(
            f"{name}: replaced empty-vector documents: base {replaced['base']}, query {replaced['query']}",
            flush=True,
        )

        base_t = [tr_text[i] for i in ib]
        query_t = [te_text[i] for i in iq]
        yb, yq = tr_y[ib], te_y[iq]

        st_b = model.encode(
            base_t, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        st_q = model.encode(
            query_t, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )

        svd = TruncatedSVD(TFIDF_DIM, random_state=SEED)
        zb = svd.fit_transform(tf_b)
        zq = svd.transform(tf_q)
        assert np.all(np.linalg.norm(zb, axis=1) > 1e-9) and np.all(
            np.linalg.norm(zq, axis=1) > 1e-9
        ), "zero row after SVD"
        zb /= np.linalg.norm(zb, axis=1, keepdims=True)
        zq /= np.linalg.norm(zq, axis=1, keepdims=True)

        path = os.path.join(CACHE, f"{name}.npz")
        np.savez(
            path,
            st_base=st_b.astype(np.float32),
            st_query=st_q.astype(np.float32),
            tfidf_base=zb.astype(np.float32),
            tfidf_query=zq.astype(np.float32),
            y_base=yb,
            y_query=yq,
            n_classes=np.int64(len(np.unique(tr_y))),
            replaced_base=np.int64(replaced["base"]),
            replaced_query=np.int64(replaced["query"]),
        )
        print(
            f"{name}: base {st_b.shape} query {st_q.shape} classes {len(np.unique(yb))} -> {path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
