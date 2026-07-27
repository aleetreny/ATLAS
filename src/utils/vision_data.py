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


def fashion_mnist():
    """Zalando's drop-in replacement for MNIST: same 28x28 uint8 format, same
    ten classes, several times harder. The depth article needs a task where
    optimisation is the bottleneck, and handwritten digits are not one."""
    raw = CACHE / "FashionMNIST" / "raw"
    if not (raw / "train-images-idx3-ubyte").exists():
        from torchvision import datasets

        CACHE.mkdir(parents=True, exist_ok=True)
        datasets.FashionMNIST(str(CACHE), train=True, download=True)
        datasets.FashionMNIST(str(CACHE), train=False, download=True)
    return (
        _read_idx(raw / "train-images-idx3-ubyte"),
        _read_idx(raw / "train-labels-idx1-ubyte"),
        _read_idx(raw / "t10k-images-idx3-ubyte"),
        _read_idx(raw / "t10k-labels-idx1-ubyte"),
    )


FASHION_CLASSES = ["t-shirt", "trouser", "pullover", "dress", "coat",
                   "sandal", "shirt", "sneaker", "bag", "ankle boot"]


def cifar10():
    """(train_images, train_labels, test_images, test_labels), uint8, with
    images (n, 32, 32, 3). Read straight out of the python pickles Krizhevsky
    ships, so no torchvision call is needed once the tarball is cached."""
    import pickle
    import tarfile

    home = CACHE / "cifar-10-batches-py"
    if not home.exists():
        tar = CACHE / "cifar-10-python.tar.gz"
        if not tar.exists():
            raise FileNotFoundError(
                f"{tar} not found: the CIFAR-10 tarball downloads slowly here, "
                "so it is fetched once by hand rather than on demand")
        with tarfile.open(tar) as fh:
            fh.extractall(CACHE)

    def load(name):
        with open(home / name, "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        X = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return X, np.array(d[b"labels"], dtype=np.int64)

    parts = [load(f"data_batch_{i}") for i in range(1, 6)]
    Xtr = np.concatenate([p[0] for p in parts])
    ytr = np.concatenate([p[1] for p in parts])
    Xte, yte = load("test_batch")
    return Xtr, ytr, Xte, yte


CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


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
