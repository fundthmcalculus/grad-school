#!/usr/bin/env python3
"""Read `dataset_specs.yaml`, expand it into template fields, and re-derive it.

Two callers, two jobs:

  `load_specs()` / `template_values()` -- used by
  `research/proposal-defense/build_pdf.py` at assembly time to substitute
  `{{dataset.<key>.<field>}}` in the prose. Import-light on purpose: it needs
  PyYAML and nothing else, so a PDF build never pulls in pandas or sklearn.

  `--verify` -- re-derives every locally checkable row from the data files
  themselves and reports disagreements. This is what keeps the spec file
  honest; it needs pandas, and optionally sklearn.

      uv run --project tribble-fis --with pyyaml python reproduce/dataset_specs.py --verify
      uv run --project tribble-fis --with pyyaml python reproduce/dataset_specs.py --fields

`check_prose.py` answers "does the document quote the run the harness
produced" for `mean +/- std` pairs, and says outright that it does not check
bare numbers. Dataset dimensions ARE bare numbers, so they were unchecked by
anything. This file closes that hole from the other side: rather than checking
the prose's dimensions after the fact, it generates them, so they cannot be
wrong in one chapter and right in another.

FIELD LIST
----------
Declared in the YAML, per dataset: `name`, `task`, `rows`, `features`,
`classes`, plus optional `published_features`, `note`, and any extra scalar
(BETH's `rows_train` and friends). Derived here:

  rows          1,030          `rows` with thousands separators
  rows_approx   123K / 1.14M   short form; exact comma form below 10,000
  features      8
  classes       6
  shape         1,030 x 8      rows x features, with a real times sign
  shape_full    1,030 x 8, 6 classes
  name / task   verbatim from the YAML

A `{{dataset...}}` reference to a key or field that does not exist is a build
error, never a silent pass-through -- see `build_pdf.py`'s substitution step.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SPEC_FILE = os.path.join(HERE, "dataset_specs.yaml")

TIMES = "×"  # the times sign the prose already uses, not the letter x


# --------------------------------------------------------------------------- #
# loading and template expansion
# --------------------------------------------------------------------------- #
def load_specs(path=SPEC_FILE):
    """Parse the spec file. Raises if it is missing or malformed."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover -- environment problem
        raise RuntimeError(
            "PyYAML is required to read dataset_specs.yaml. Install it into the "
            "environment running the build (`uv pip install pyyaml`)."
        ) from exc

    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict) or "datasets" not in doc:
        raise ValueError(f"{path}: expected a top-level `datasets:` mapping")
    return doc["datasets"]


def _commas(n):
    return f"{int(n):,}"


def _approx(n):
    """Short human form: 1,030 / 58K / 123K / 1.14M.

    Below 10,000 the exact comma form is already short enough and rounding it
    would lose real precision, so it is returned unchanged.
    """
    n = int(n)
    if n < 10_000:
        return _commas(n)
    if n < 1_000_000:
        return f"{round(n / 1_000):,}K"
    return f"{n / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"


def template_values(specs=None):
    """Flatten the specs to {'concrete.shape': '1,030 x 8', ...}.

    Every value is a string, ready to drop into markdown.
    """
    specs = load_specs() if specs is None else specs
    out = {}
    for key, spec in specs.items():
        rows = spec.get("rows")
        feats = spec.get("features")
        classes = spec.get("classes")

        fields = {}
        # Pass through every declared scalar, so a dataset-specific field like
        # BETH's `rows_train` is usable without teaching this function about it.
        for k, v in spec.items():
            if isinstance(v, (str, int, float)):
                fields[k] = _commas(v) if isinstance(v, int) and v >= 10_000 else str(v)

        if rows is not None:
            fields["rows"] = _commas(rows)
            fields["rows_approx"] = _approx(rows)
        if rows is not None and feats is not None:
            fields["shape"] = f"{_commas(rows)} {TIMES} {feats}"
            fields["shape_full"] = fields["shape"] + (
                f", {classes} classes" if classes else ""
            )
        for name, value in fields.items():
            out[f"{key}.{name}"] = value
    return out


# --------------------------------------------------------------------------- #
# verification -- re-derive the numbers from the data
# --------------------------------------------------------------------------- #
def _derive_csv(verify):
    """(rows, features, classes) measured from one or more CSVs, or None.

    `features` counts columns that would reach a model: everything except the
    label, the columns named in `drop_columns`, and any non-numeric column.
    """
    import numpy as np
    import pandas as pd

    paths = verify.get("paths") or [verify["path"]]
    label = verify.get("label_column")
    drop = set(verify.get("drop_columns") or [])

    total_rows = 0
    feature_count = None
    labels = set()
    for rel in paths:
        full = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(full):
            return None
        df = pd.read_csv(full, low_memory=False)
        total_rows += len(df)
        X = df.drop(columns=[c for c in ({label} | drop) if c in df.columns])
        X = X.select_dtypes(include=[np.number])
        if feature_count is None:
            feature_count = X.shape[1]
        elif feature_count != X.shape[1]:
            raise ValueError(f"{rel}: split disagrees on feature count")
        if label in df.columns:
            labels.update(pd.Series(df[label]).unique().tolist())
    return total_rows, feature_count, len(labels) if labels else None


def _derive_sklearn(verify):
    from sklearn import datasets as skd

    import numpy as np

    loader = getattr(skd, verify["loader"])
    bunch = loader()
    X, y = bunch.data, bunch.target
    n_classes = len(np.unique(y)) if len(np.unique(y)) < 50 else None
    return X.shape[0], X.shape[1], n_classes


def verify(specs=None):
    """Re-derive every checkable row. Returns the number of disagreements."""
    specs = load_specs() if specs is None else specs
    bad = skipped = ok = 0

    for key, spec in specs.items():
        v = spec.get("verify") or {"method": "none"}
        method = v.get("method", "none")
        if method == "none":
            print(f"  --  {key}: not independently verifiable (declared)")
            skipped += 1
            continue
        try:
            derived = _derive_csv(v) if method == "csv" else _derive_sklearn(v)
        except Exception as exc:  # noqa: BLE001 -- report, keep checking
            print(f"  ??  {key}: could not derive ({exc.__class__.__name__}: {exc})")
            skipped += 1
            continue

        if derived is None:
            note = "optional, not present" if v.get("optional") else "DATA MISSING"
            print(f"  --  {key}: {note}")
            skipped += 1
            continue

        d_rows, d_feats, d_classes = derived
        problems = []
        if spec.get("rows") != d_rows:
            problems.append(f"rows: spec {spec.get('rows')} vs derived {d_rows}")
        if spec.get("features") != d_feats:
            problems.append(
                f"features: spec {spec.get('features')} vs derived {d_feats}"
            )
        if spec.get("classes") is not None and d_classes not in (None, spec["classes"]):
            problems.append(
                f"classes: spec {spec['classes']} vs derived {d_classes}"
            )

        if problems:
            bad += 1
            print(f"  XX  {key}: " + "; ".join(problems))
        else:
            ok += 1
            print(f"  ok  {key}: {d_rows:,} {TIMES} {d_feats}")

    print(f"\n{ok} verified, {bad} disagreeing, {skipped} unchecked")
    if bad:
        print(
            "\nA disagreement means the spec file is stale, not that the prose is. "
            "Fix dataset_specs.yaml, then rebuild the PDF so the prose follows."
        )
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--verify",
        action="store_true",
        help="re-derive every locally checkable spec from the data files",
    )
    ap.add_argument(
        "--fields",
        action="store_true",
        help="print every {{dataset.*}} template key and its expansion",
    )
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.fields:
        for k, v in sorted(template_values().items()):
            print(f"{{{{dataset.{k}}}}}\t{v}")
        return 0
    if args.verify:
        return 1 if verify() else 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
