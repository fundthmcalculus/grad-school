"""Datasets for the experiment.

Glass is read from the copy committed at the repository root, never from a
network fallback: substituting a differently-preprocessed copy of a dataset is
exactly the failure documented in `WORKINGDOC.md` §1, and it is silent. If the
file is missing this raises rather than fetching something that looks similar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import load_iris, load_wine

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Dataset:
    name: str
    x: NDArray[np.float64]
    y: NDArray[np.int64]  # already 0..C-1
    feature_names: list[str]
    class_names: list[str]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.x.shape[0]), int(self.x.shape[1]))


def _from_sklearn(loader, name: str) -> Dataset:
    bunch = loader()
    return Dataset(
        name=name,
        x=np.asarray(bunch.data, dtype=np.float64),
        y=np.asarray(bunch.target, dtype=np.int64),
        feature_names=[str(f) for f in bunch.feature_names],
        class_names=[str(t) for t in bunch.target_names],
    )


def load_glass() -> Dataset:
    path = os.path.join(REPO_ROOT, "glass.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"glass.csv not found at {path}; refusing to substitute a downloaded copy"
        )
    frame = pd.read_csv(path)
    x = frame.drop(columns=["Type"]).to_numpy(dtype=np.float64)
    raw = frame["Type"].to_numpy(dtype=np.int64)
    present = np.unique(raw)  # type 4 has no instances in this database
    remap = {int(t): i for i, t in enumerate(present)}
    y = np.array([remap[int(t)] for t in raw], dtype=np.int64)
    return Dataset(
        name="glass",
        x=x,
        y=y,
        feature_names=[str(c) for c in frame.columns if c != "Type"],
        class_names=[f"type{int(t)}" for t in present],
    )


LOADERS = {
    "iris": lambda: _from_sklearn(load_iris, "iris"),
    "wine": lambda: _from_sklearn(load_wine, "wine"),
    "glass": load_glass,
}


def load(name: str) -> Dataset:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(LOADERS)}")
    return LOADERS[name]()
