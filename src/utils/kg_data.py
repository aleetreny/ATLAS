"""Los grafos de conocimiento del módulo 6.2.

WN18RR es la versión de WordNet que se usa para medir esto desde 2018: 40.943
palabras, once relaciones (hiperónimo, merónimo, "deriva de", "es parecido
a"...) y 93.003 tripletas, con las relaciones inversas quitadas, que es lo que
la "RR" del nombre significa y lo que impide que un modelo apruebe copiando la
respuesta del otro lado.

Y al lado, un grafo escrito aquí donde los cuatro patrones que la literatura
discute (simetría, inversión, composición y jerarquía uno a muchos) están
puestos a mano, uno por relación, para poder puntuar cada patrón por separado
en vez de puntuar una media sobre once relaciones que mezclan todos.
"""
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".atlas_vision_data" / "wn18rr"
BASE = ("https://raw.githubusercontent.com/DeepGraphLearning/"
        "KnowledgeGraphEmbedding/master/data/wn18rr/")
FILES = ["train.txt", "valid.txt", "test.txt", "entities.dict", "relations.dict"]


def _fetch(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        print(f"  downloading wn18rr/{name}")
        with urllib.request.urlopen(BASE + name, timeout=300) as r:
            path.write_bytes(r.read())
    return path


def wn18rr():
    """(train, valid, test) como arrays de enteros, más los diccionarios."""
    ent = {}
    for line in _fetch("entities.dict").read_text().splitlines():
        i, name = line.split("\t")
        ent[name] = int(i)
    rel = {}
    for line in _fetch("relations.dict").read_text().splitlines():
        i, name = line.split("\t")
        rel[name] = int(i)

    def load(name):
        rows = []
        for line in _fetch(name).read_text().splitlines():
            h, r, t = line.split("\t")
            rows.append((ent[h], rel[r], ent[t]))
        return np.array(rows, dtype=np.int64)

    return load("train.txt"), load("valid.txt"), load("test.txt"), ent, rel


def digest():
    h = hashlib.sha256()
    for name in ["train.txt", "valid.txt", "test.txt"]:
        h.update(_fetch(name).read_bytes())
    return h.hexdigest()[:16]


def word_of(name):
    """`__dog_NN_1` es ilegible en una etiqueta; `dog` no lo es."""
    core = name.strip("_")
    parts = core.split("_")
    while parts and (len(parts[-1]) <= 2 or parts[-1].isdigit()):
        parts.pop()
    return " ".join(parts).replace("_", " ") or core


def pattern_kg(n_ent=600, seed=19):
    """Un grafo con los cuatro patrones escritos, uno por relación.

    Nada de esto se descubre: se escribe, y por eso se puede puntuar por
    patrón. Las relaciones son

      * `mirror`   simétrica: si (a, mirror, b) entonces (b, mirror, a)
      * `parent` y `child`  inversas la una de la otra
      * `grandparent`  la composición de `parent` consigo misma
      * `kind_of`  jerárquica, uno a muchos: muchos hijos, un padre
    """
    rng = np.random.default_rng(seed)
    rels = {"mirror": 0, "parent": 1, "child": 2, "grandparent": 3, "kind_of": 4}
    triples = []

    # un bosque de padres, con el que salen las tres relaciones de familia
    parent = np.full(n_ent, -1)
    for i in range(1, n_ent):
        if rng.random() < 0.75:
            parent[i] = rng.integers(0, i)
    for c in range(n_ent):
        p = parent[c]
        if p >= 0:
            triples.append((c, rels["parent"], p))
            triples.append((p, rels["child"], c))
            g = parent[p]
            if g >= 0:
                triples.append((c, rels["grandparent"], g))

    # parejas simétricas
    perm = rng.permutation(n_ent)
    for a, b in zip(perm[::2], perm[1::2]):
        triples.append((int(a), rels["mirror"], int(b)))
        triples.append((int(b), rels["mirror"], int(a)))

    # una jerarquía de tipos: 12 clases, cada entidad cuelga de una
    n_types = 12
    types = rng.integers(0, n_types, n_ent)
    for e in range(n_ent):
        triples.append((int(e), rels["kind_of"], int(n_ent + types[e])))

    T = np.array(sorted(set(triples)), dtype=np.int64)
    order = rng.permutation(len(T))
    T = T[order]
    n_test = len(T) // 10
    return (T[2 * n_test:], T[:n_test], T[n_test:2 * n_test],
            {"entities": n_ent + n_types, "relations": len(rels), "names": rels,
             "triples": int(len(T))})
