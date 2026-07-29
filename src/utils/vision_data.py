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


# Los cuatro ficheros que componen cualquiera de los dos datasets, con el nombre
# que torchvision les deja ya descomprimidos.
_IDX_FILES = [
    "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
]

# De donde bajarlos sin torchvision. El formato IDX es el mismo fichero, asi
# que los bytes que salen de aqui son identicos a los que deja torchvision y
# ninguna cifra publicada se mueve por cambiar de camino.
_MIRRORS = {
    "MNIST": "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "FashionMNIST": ("https://raw.githubusercontent.com/zalandoresearch/"
                     "fashion-mnist/master/data/fashion/"),
}


def _ensure_idx(nombre):
    """Los IDX en `~/.atlas_vision_data/<nombre>/raw`, cueste lo que cueste.

    torchvision primero, porque es lo que uso la maquina de referencia y no hay
    razon para cambiarle el camino a nadie. Pero arrastra torch entero (unos 3
    GB de ruedas CUDA que aqui no sirven para nada), y una sesion que solo
    necesita los pixeles para medir sobre un modelo YA entrenado no deberia
    pagar eso: si torchvision no esta, se bajan los cuatro .gz y se
    descomprimen con la stdlib. Los bytes son los mismos.
    """
    raw = CACHE / nombre / "raw"
    if (raw / _IDX_FILES[0]).exists():
        return raw
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        from torchvision import datasets

        clase = getattr(datasets, nombre)
        clase(str(CACHE), train=True, download=True)
        clase(str(CACHE), train=False, download=True)
        return raw
    except ImportError:
        pass

    import gzip
    import urllib.request

    raw.mkdir(parents=True, exist_ok=True)
    for fichero in _IDX_FILES:
        destino = raw / fichero
        if destino.exists():
            continue
        url = _MIRRORS[nombre] + fichero + ".gz"
        with urllib.request.urlopen(url, timeout=120) as resp:
            crudo = gzip.decompress(resp.read())
        # A un temporal y luego rename, por lo mismo que los caches de los
        # generadores: un corte a mitad de escritura no puede dejar un fichero
        # a medias que la tirada siguiente lea como bueno.
        tmp = destino.with_suffix(".part")
        tmp.write_bytes(crudo)
        tmp.replace(destino)
    return raw


def mnist():
    """(train_images, train_labels, test_images, test_labels), uint8.

    Images are (n, 28, 28) with 0 = background. Downloads through torchvision
    on first use, then reads the raw IDX files directly so nothing but numpy
    is needed afterwards.
    """
    raw = _ensure_idx("MNIST")
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
    raw = _ensure_idx("FashionMNIST")
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
