"""Shared dataset access for the ATLAS vision articles (module 1.3).

The image datasets are large enough that they do not belong in the repository
(MNIST alone is 64 MB unpacked), so they live in a user-level cache exactly
the way scikit-learn keeps covtype and the wine data for the earlier modules:

    ~/.atlas_vision_data/

Nothing here is committed; the generators fetch on first run and reuse
afterwards, so a fresh clone reproduces every number with one command.
"""
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".atlas_vision_data"


def _read_idx(path):
    """Minimal IDX reader: the format MNIST has shipped in since 1998."""
    with open(path, "rb") as fh:
        magic = int.from_bytes(fh.read(4), "big")
        n_dims = magic & 0xFF
        dims = [int.from_bytes(fh.read(4), "big") for _ in range(n_dims)]
        return np.frombuffer(fh.read(), dtype=np.uint8).reshape(dims)


def mnist():
    """(train_images, train_labels, test_images, test_labels), uint8.

    Images are (n, 28, 28) with 0 = background. Downloads through torchvision
    on first use, then reads the raw IDX files directly so nothing but numpy
    is needed afterwards.
    """
    raw = CACHE / "MNIST" / "raw"
    if not (raw / "train-images-idx3-ubyte").exists():
        from torchvision import datasets

        CACHE.mkdir(parents=True, exist_ok=True)
        datasets.MNIST(str(CACHE), train=True, download=True)
        datasets.MNIST(str(CACHE), train=False, download=True)
    return (
        _read_idx(raw / "train-images-idx3-ubyte"),
        _read_idx(raw / "train-labels-idx1-ubyte"),
        _read_idx(raw / "t10k-images-idx3-ubyte"),
        _read_idx(raw / "t10k-labels-idx1-ubyte"),
    )


def subset(X, y, n, seed=19):
    """A seeded, class-stratified slice, so every article's 'first n' means
    the same thing and no experiment gets an accidentally easier draw."""
    rs = np.random.RandomState(seed)
    idx = []
    per = n // len(np.unique(y))
    for c in np.unique(y):
        pool = np.where(y == c)[0]
        idx.append(rs.choice(pool, per, replace=False))
    idx = np.sort(np.concatenate(idx))
    return X[idx], y[idx]
