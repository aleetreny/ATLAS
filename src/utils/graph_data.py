"""El grafo real del módulo 6.2, y los sintéticos con la verdad escrita.

Cora es el dataset canónico de redes en grafos: 2.708 artículos de aprendizaje
automático, 5.429 citas entre ellos, un vector de 1.433 palabras por artículo y
siete temas. Se descarga en el formato de Planetoid (los ficheros `ind.cora.*`
del repositorio de Yang, Cohen y Salakhutdinov), que es exactamente la partición
que usa todo el mundo, así que las cifras se pueden comparar con las publicadas
en vez de con nada.

Y al lado, un generador de grafos donde la respuesta se conoce porque la
escribe el propio fichero: un bloque estocástico con rasgos, con un mando de
homofilia. Eso es lo que permite medir **cuándo** una red en grafos ayuda, en
vez de medir que ayuda en el único grafo que todo el mundo prueba.
"""
import hashlib
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp

CACHE = Path.home() / ".atlas_vision_data" / "planetoid"
BASE = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/"
PARTS = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]

def _fetch(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        print(f"  downloading {name}")
        with urllib.request.urlopen(BASE + name, timeout=300) as r:
            path.write_bytes(r.read())
    return path


def cora():
    """(A, X, y, idx_train, idx_val, idx_test) en el reparto de Planetoid.

    A es la matriz de adyacencia simétrica sin bucles, X los rasgos binarios de
    palabras, y las siete clases. La partición es la estándar: 20 nodos
    etiquetados por clase, 500 de validación y 1.000 de test."""
    objs = {}
    for part in PARTS[:-1]:
        with open(_fetch(f"ind.cora.{part}"), "rb") as fh:
            objs[part] = pickle.load(fh, encoding="latin1")
    test_idx = np.loadtxt(_fetch("ind.cora.test.index"), dtype=np.int64)
    test_sorted = np.sort(test_idx)

    allx, tx = objs["allx"], objs["tx"]
    ally, ty = objs["ally"], objs["ty"]
    n = allx.shape[0] + tx.shape[0]
    X = sp.lil_matrix((n, allx.shape[1]), dtype=np.float32)
    X[: allx.shape[0]] = allx
    # Los nodos de test vienen BARAJADOS en el fichero: la fila j de `tx`
    # pertenece al nodo `test_idx[j]`, no al j-ésimo id ordenado. Asignarlos
    # ordenados no da un error, da otro grafo: la homofilia por arista cae de
    # 0,81 a 0,45 y todas las cifras del artículo con ella.
    X[test_idx] = tx
    Y = np.zeros((n, ally.shape[1]), dtype=np.float32)
    Y[: ally.shape[0]] = ally
    Y[test_idx] = ty
    y = Y.argmax(1)

    rows, cols = [], []
    for src, dsts in objs["graph"].items():
        for dst in dsts:
            rows.append(src)
            cols.append(dst)
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()

    idx_train = np.arange(objs["y"].shape[0])
    idx_val = np.arange(objs["y"].shape[0], objs["y"].shape[0] + 500)
    idx_test = test_sorted
    return sp.csr_matrix(X), y, A, idx_train, idx_val, idx_test


def digest_cora():
    h = hashlib.sha256()
    for part in ["x", "y", "graph"]:
        h.update(_fetch(f"ind.cora.{part}").read_bytes())
    return h.hexdigest()[:16]


def contextual_sbm(n=1200, classes=2, d=32, degree=10.0, homophily=0.9,
                   signal=1.0, seed=19):
    """Un grafo con la respuesta escrita antes de existir la primera arista.

    Cada nodo tiene una clase, un vector de rasgos que es la media de su clase
    más ruido, y aristas repartidas entre "de la misma clase" y "de otra" según
    el mando de homofilia. Con eso se puede preguntar lo que ningún grafo real
    contesta: a partir de qué homofilia una red en grafos deja de ayudar."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, n)
    centres = rng.normal(size=(classes, d)) * signal
    X = centres[y] + rng.normal(size=(n, d))

    # p_in y p_out fijan a la vez el grado medio y la homofilia
    m = degree * n / 2.0
    m_in = m * homophily
    m_out = m - m_in
    same = y[:, None] == y[None, :]
    np.fill_diagonal(same, False)
    n_same = same.sum() / 2.0
    n_diff = (~same).sum() / 2.0 - 0.0
    p_in = min(m_in / max(n_same, 1), 1.0)
    p_out = min(m_out / max(n_diff, 1), 1.0)

    iu = np.triu_indices(n, k=1)
    probs = np.where(y[iu[0]] == y[iu[1]], p_in, p_out)
    keep = rng.random(iu[0].size) < probs
    rows, cols = iu[0][keep], iu[1][keep]
    A = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    A = A + A.T
    return X.astype(np.float32), y, sp.csr_matrix(A), {"p_in": p_in, "p_out": p_out,
                                                       "degree": float(A.sum() / n)}
