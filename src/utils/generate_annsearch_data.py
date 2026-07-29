"""Las cifras del artículo de búsqueda aproximada de vecinos.

El artículo 29 ya buscó vecinos, y los buscó **exactos**: mirar todos los
puntos, ordenar, quedarse con los k primeros. Esta página empieza donde eso deja
de caber, y la primera cosa que hay que medir es dónde deja de caber, porque la
respuesta habitual ("cuando hay muchos vectores") no es un número.

Todo el coste de esta página se cuenta en **distancias evaluadas**, no en
segundos. Un reloj mide la contención de la máquina y no se reproduce en otra;
el número de distancias que un índice calcula para contestar es una propiedad
del índice y del dato. Los índices están escritos en `annkit.py` y llevan el
contador dentro, así que la cifra publicada es la que el índice pagó.

Lo que se mide:

  1. **El suelo del problema**: cuánto se parecen entre sí el vecino más cercano
     y el más lejano según sube la dimensión, sobre datos gaussianos y sobre los
     de verdad, con la dimensión intrínseca de los reales estimada. Si esa
     razón se va a uno, "aproximado" deja de significar algo.
  2. **El índice invertido**: partición por k-means (el algoritmo del artículo
     12) y sondeo de celdas, con recall contra coste.
  3. **El grafo navegable**, con sus dos piezas ablables: las capas de arriba y
     el heurístico de poda de aristas. La afirmación de que la jerarquía es lo
     que lo hace funcionar es comprobable, y aquí se comprueba.
  4. **La cuantización por producto**, que es el corazón de FAISS y el mismo
     objeto que el libro de códigos del artículo 37: error de la distancia
     estimada contra bytes por vector, y la asimétrica contra la simétrica.
  5. **Y lo que casi nunca se mide: si el recall importa.** Con las ocho
     categorías de Reuters como clave de respuestas, se puede puntuar la
     recuperación de verdad y ver cuánta calidad cuesta cada punto de recall
     perdido.

Ejecutar desde cualquier sitio: `python src/utils/generate_annsearch_data.py`.
"""
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annkit import HNSW, IVF, PQ, brute, kmeans, recall_at  # noqa: E402
from nlp_data import reuters, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ann-search" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "annsearch_cache")
EMBED = ROOT / "embeddings" / "data" / "embeddings.json"

SEED = 19
R8 = ("acq", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade")
DIMS = 64
N_INDEX = 6000
N_QUERY = 200
K = 10
M = 16
EF_C = 80
EFS = (10, 16, 24, 40, 64, 100, 160, 250)
NPROBES = (1, 2, 4, 8, 16, 32, 64)
NLIST = 64
SCALE_NS = (750, 1500, 3000, 6000)
ABLATE_N = 3000
PQ_MS = (2, 4, 8, 16, 32)
CURSE_DIMS = (2, 4, 8, 16, 32, 64, 128, 256)
DEMO_N = 250

TAG = (f"d{DIMS}n{N_INDEX}q{N_QUERY}k{K}m{M}e{EF_C}l{NLIST}"
       f"a{ABLATE_N}v{SEED}")


def cached(name, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.{TAG}.pkl"
    if path.is_file():
        with path.open("rb") as fh:
            return pickle.load(fh)
    val = fn()
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(val, fh)
    tmp.replace(path)
    return val


def r(v, d=6):
    if isinstance(v, dict):
        return {k: r(x, d) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [r(x, d) for x in v]
    if v is None or isinstance(v, (bool, str)):
        return v
    v = float(v)
    return None if not np.isfinite(v) else round(v, d)


# ------------------------------------------------------------------- los datos
def reuters_vectors():
    docs, cats, split = reuters()
    keep = [i for i, c in enumerate(cats) if len(c) == 1 and c[0] in R8]
    texts = [tokenize(docs[i]) for i in keep]
    lab = np.array([R8.index(cats[i][0]) for i in keep])
    vocab = Counter()
    for t in texts:
        vocab.update(set(t))
    words = sorted([w for w, c in vocab.items() if c >= 5])
    idx = {w: k for k, w in enumerate(words)}
    X = np.zeros((len(texts), len(words)), np.float32)
    for i, t in enumerate(texts):
        for w in t:
            j = idx.get(w)
            if j is not None:
                X[i, j] += 1.0
    df = (X > 0).sum(0)
    X = np.log1p(X) * np.log(len(texts) / np.maximum(df, 1)).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    Z = (U[:, :DIMS] * S[:DIMS]).astype(np.float64)
    return Z, lab, len(words), [" ".join(t[:12]) for t in texts]


def load_words():
    d = json.loads(EMBED.read_text("utf-8"))
    v = d["vectors"]
    W = np.array(v["models"]["skip-gram"]["vec"], float)
    return v["words"], W, d["meta"], v["tags"]


# ------------------------------------------------------------ 1. concentración
def two_nn_dimension(X, rng, sample=1500):
    """Estimador de Facco: la razón entre el segundo y el primer vecino.

    La dimensión intrínseca sale de una recta sin término independiente sobre
    log(mu) contra menos log(1 - F), y no depende de ninguna escala.
    """
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    S = X[idx]
    d = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    part = np.sort(d, axis=1)[:, :2]
    mu = part[:, 1] / np.maximum(part[:, 0], 1e-12)
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1)])
    F = np.arange(1, len(mu) + 1) / (len(mu) + 1)
    x = np.log(mu)
    yv = -np.log(1 - F)
    return float((x @ yv) / (x @ x))


def stage_concentration(Z):
    rng = np.random.default_rng(SEED)
    rows = []
    for d in CURSE_DIMS:
        X = rng.standard_normal((2000, d))
        Q = rng.standard_normal((100, d))
        dist = np.sqrt(((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        near = dist.min(1)
        far = dist.max(1)
        rows.append(dict(d=d, contrast=float(np.mean((far - near) / near)),
                         near=float(near.mean()), far=float(far.mean()),
                         ratio=float(np.mean(far / near))))
    Q = Z[rng.choice(len(Z), 100, replace=False)]
    dist = np.sqrt(((Q[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    dist[dist == 0] = np.inf
    near, far = dist.min(1), np.where(np.isfinite(dist), dist, -np.inf).max(1)
    real = dict(d=DIMS, contrast=float(np.mean((far - near) / near)),
                near=float(near.mean()), far=float(far.mean()),
                ratio=float(np.mean(far / near)),
                intrinsic=two_nn_dimension(Z, rng))
    words, W, wmeta, tags = load_words()
    Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    dw = np.sqrt(((Wn[:100, None, :] - Wn[None, :, :]) ** 2).sum(-1))
    dw[dw == 0] = np.inf
    nw, fw = dw.min(1), np.where(np.isfinite(dw), dw, -np.inf).max(1)
    wordrow = dict(d=int(W.shape[1]), n=int(len(words)),
                   contrast=float(np.mean((fw - nw) / nw)),
                   ratio=float(np.mean(fw / nw)),
                   intrinsic=two_nn_dimension(Wn, rng))
    return dict(gaussian=rows, reuters=real, words=wordrow,
                word_meta=dict(corpus=wmeta["corpus"], dim=wmeta["dim"],
                               shipped=wmeta["shipped"]))


# ------------------------------------------------------------------ 2 y 3 y 4
def stage_indices(Z, lab):
    rng = np.random.default_rng(SEED)
    qi = rng.choice(len(Z), N_QUERY, replace=False)
    mask = np.ones(len(Z), bool)
    mask[qi] = False
    X = Z[mask][:N_INDEX]
    lab_x = lab[mask][:N_INDEX]
    Q = Z[qi]
    lab_q = lab[qi]
    truth, cost = brute(X, Q, K)
    print(f"    verdad exacta: {cost} distancias para {len(Q)} consultas")

    ivf = IVF(X, NLIST, np.random.default_rng(SEED + 1))
    ivf_rows = []
    for npb in NPROBES:
        res = [ivf.search(q, K, npb) for q in Q]
        found = np.array([a for a, _ in res])
        ivf_rows.append(dict(nprobe=npb, recall=recall_at(found, truth),
                             cost=float(np.mean([c for _, c in res])),
                             quality=category_hits(found, lab_x, lab_q)))
        print(f"    ivf nprobe {npb}: recall {ivf_rows[-1]['recall']:.4f}")

    print("    construyendo el grafo")
    h = HNSW(X, M=M, ef_construction=EF_C, rng=np.random.default_rng(SEED + 2))
    hnsw_rows = []
    for ef in EFS:
        res = [h.search(q, K, ef) for q in Q]
        found = np.array([a for a, _ in res])
        hnsw_rows.append(dict(ef=ef, recall=recall_at(found, truth),
                              cost=float(np.mean([c for _, c in res])),
                              quality=category_hits(found, lab_x, lab_q)))
        print(f"    hnsw ef {ef}: recall {hnsw_rows[-1]['recall']:.4f} "
              f"coste {hnsw_rows[-1]['cost']:.0f}")
    deg = h.degrees(0)
    levels = np.bincount(h.node_level)

    print("    la cuantización por producto")
    pq_rows = []
    for m in PQ_MS:
        if DIMS % m:
            continue
        pq = PQ(X, m, 8, np.random.default_rng(SEED + 3))
        d_true = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        adc = np.stack([pq.adc(q) for q in Q])
        sdc = np.stack([pq.sdc(q) for q in Q])
        found = np.argsort(adc, 1)[:, :K]
        pq_rows.append(dict(m=m, bytes=pq.bytes_per_vector(),
                            adc_err=float(np.abs(adc - d_true).mean() / d_true.mean()),
                            sdc_err=float(np.abs(sdc - d_true).mean() / d_true.mean()),
                            recall=recall_at(found, truth),
                            quality=category_hits(found, lab_x, lab_q),
                            rec_err=float(np.linalg.norm(pq.reconstruct() - X)
                                          / np.linalg.norm(X))))
        print(f"    pq m={m}: {pq.bytes_per_vector():.0f} bytes, recall "
              f"{pq_rows[-1]['recall']:.4f}")
    exact_quality = category_hits(truth, lab_x, lab_q)
    return dict(ivf=ivf_rows, hnsw=hnsw_rows, pq=pq_rows,
                brute_cost=float(cost / len(Q)), n=int(len(X)),
                queries=int(len(Q)), k=K,
                degrees=dict(mean=float(deg.mean()), max=int(deg.max()),
                             min=int(deg.min()),
                             hist=np.bincount(deg, minlength=2 * M + 2).tolist()),
                levels=levels.tolist(),
                build_cost=int(h.build_cost),
                exact_quality=exact_quality,
                float_bytes=float(DIMS * 4))


def category_hits(found, lab_x, lab_q):
    """La medida que le importa a quien consulta: cuántos de los diez devueltos
    son de la categoría de la consulta."""
    return float(np.mean([(lab_x[row] == q).mean() for row, q in zip(found, lab_q)]))


def stage_ablation(Z):
    """Las capas y el heurístico, quitados uno a uno."""
    rng = np.random.default_rng(SEED)
    qi = rng.choice(len(Z), N_QUERY, replace=False)
    mask = np.ones(len(Z), bool)
    mask[qi] = False
    X = Z[mask][:ABLATE_N]
    Q = Z[qi]
    truth, _ = brute(X, Q, K)
    rows = []
    for tag, levels, heur in (("full", True, True), ("one layer", False, True),
                              ("nearest m edges", True, False),
                              ("neither", False, False)):
        h = HNSW(X, M=M, ef_construction=EF_C, levels=levels, heuristic=heur,
                 rng=np.random.default_rng(SEED + 2))
        for ef in (16, 64, 250):
            res = [h.search(q, K, ef) for q in Q]
            found = np.array([a for a, _ in res])
            rows.append(dict(arm=tag, ef=ef, recall=recall_at(found, truth),
                             cost=float(np.mean([c for _, c in res])),
                             build=int(h.build_cost),
                             layers=int(len(h.graph))))
        print(f"    ablación {tag} hecha")
    return dict(rows=rows, n=int(len(X)))


def stage_scale(Z):
    """Cómo crece el coste de una consulta con el tamaño del índice."""
    rng = np.random.default_rng(SEED)
    qi = rng.choice(len(Z), 100, replace=False)
    mask = np.ones(len(Z), bool)
    mask[qi] = False
    pool = Z[mask]
    Q = Z[qi]
    rows = []
    for n in SCALE_NS:
        X = pool[:n]
        truth, _ = brute(X, Q, K)
        h = HNSW(X, M=M, ef_construction=EF_C, rng=np.random.default_rng(SEED + 2))
        res = [h.search(q, K, 64) for q in Q]
        found = np.array([a for a, _ in res])
        rows.append(dict(n=n, recall=recall_at(found, truth),
                         cost=float(np.mean([c for _, c in res])),
                         brute=float(n), build=int(h.build_cost),
                         ratio=float(n / np.mean([c for _, c in res]))))
        print(f"    n = {n}: coste {rows[-1]['cost']:.0f} contra {n} de fuerza bruta")
    return rows


def stage_demo(Z, lab):
    """Un grafo pequeño en dos dimensiones, para poder verlo caminar."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Z), DEMO_N, replace=False)
    P = Z[idx][:, :2]
    P = (P - P.mean(0)) / P.std(0)
    h = HNSW(P, M=4, ef_construction=32, rng=np.random.default_rng(SEED + 5))
    edges = sorted({(min(a, b), max(a, b))
                    for a, nbs in h.graph[0].items() for b in nbs})
    walks = []
    for s in range(4):
        q = rng.standard_normal(2) * 1.1
        path, ep = [], [h.entry]
        for L in range(len(h.graph) - 1, 0, -1):
            if not h.graph[L]:
                continue
            nxt = h._search_layer(q, ep, 1, L)[0][1]
            path.append(dict(level=L, node=int(nxt)))
            ep = [nxt]
        found = h._search_layer(q, ep, 16, 0)
        truth = int(np.argmin(((P - q) ** 2).sum(1)))
        walks.append(dict(q=q.tolist(), path=path,
                          found=[int(i) for _, i in found[:5]], truth=truth,
                          hit=bool(found[0][1] == truth)))
    return dict(points=P.tolist(), labels=lab[idx].tolist(),
                edges=[[int(a), int(b)] for a, b in edges],
                levels=h.node_level.tolist(), entry=int(h.entry),
                upper=[sorted(int(k) for k in h.graph[L]) for L in range(len(h.graph))],
                walks=walks, m=4)


def main():
    print("  cargando Reuters")
    Z, lab, vocab, snippets = cached("vectors", reuters_vectors)
    print(f"  {len(Z)} noticias, {vocab} palabras, {DIMS} dimensiones")

    print("  1/5 la concentración de las distancias")
    conc = cached("concentration", lambda: stage_concentration(Z))
    print("  2/5 los tres índices")
    idxs = cached("indices", lambda: stage_indices(Z, lab))
    print("  3/5 las ablaciones del grafo")
    abl = cached("ablation", lambda: stage_ablation(Z))
    print("  4/5 el crecimiento con el tamaño")
    scale = cached("scale", lambda: stage_scale(Z))
    print("  5/5 el grafo de la página")
    demo = cached("demo", lambda: stage_demo(Z, lab))

    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(
        meta=dict(seed=SEED, dims=DIMS, docs=int(len(Z)), vocab=int(vocab),
                  index_n=idxs["n"], queries=idxs["queries"], k=K, M=M,
                  ef_construction=EF_C, nlist=NLIST, classes=list(R8),
                  note="cost is counted in distance evaluations, never in "
                       "seconds; the ground truth is exhaustive search"),
        concentration=r(conc),
        indices=r(idxs),
        ablation=r(abl),
        scale=r(scale),
        demo=dict(points=r(demo["points"], 4), labels=demo["labels"],
                  edges=demo["edges"], levels=demo["levels"],
                  entry=demo["entry"], upper=demo["upper"],
                  walks=r(demo["walks"], 4), m=demo["m"]),
    )
    path = OUT / "annsearch.json"
    path.write_text(json.dumps(out, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
