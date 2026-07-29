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
import base64
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annkit import HNSW, IVF, PQ, brute, kmeans, pad_to, recall_at  # noqa: E402
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

def pairdist(Q, X):
    """Distancias al cuadrado por la identidad de la norma, sin el tensor de en medio."""
    return np.maximum((X ** 2).sum(1)[None, :] - 2.0 * (Q @ X.T)
                      + (Q ** 2).sum(1)[:, None], 0.0)

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
    # Descomposición truncada y aleatorizada en vez de la completa: solo hacen
    # falta las primeras columnas, y una SVD completa de una matriz de miles por
    # miles materializa una U cuadrada de cientos de megas para tirar el 99% de
    # ella. Con semilla fija, así que es tan reproducible como la otra.
    U, S, _ = randomized_svd(X - X.mean(0), n_components=DIMS,
                             random_state=SEED)
    Z = (U[:, :DIMS] * S[:DIMS]).astype(np.float64)
    return Z, lab, len(words), [" ".join(t[:12]) for t in texts]


def decode_half(b64, n, d):
    """Los vectores del artículo 41 viajan en float16 y base64, no en JSON.

    Es el mismo contrato que lee `decodeHalf` de `assets/js/textkit.js`: little
    endian, por filas. Leerlos con `np.array(..., float)` falla con un
    ValueError sobre una cadena de base64, que es un error honesto y confuso.
    """
    W = np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float64)
    assert W.size == n * d, f"{W.size} medios contra {n} por {d} esperados"
    return W.reshape(n, d)


def load_words():
    d = json.loads(EMBED.read_text("utf-8"))
    v = d["vectors"]
    W = decode_half(v["models"]["skip-gram"]["vec"], len(v["words"]), v["dim"])
    return v["words"], W, d["meta"], v["tags"]


# ------------------------------------------------------------ 1. concentración
# Las dos medidas de esta sección son cocientes CON LA DISTANCIA AL VECINO MÁS
# CERCANO EN EL DENOMINADOR, así que dos documentos iguales las hacen explotar:
# Reuters archiva la misma noticia más de una vez (es justo lo que mide la
# sección de grupos del artículo de validación cruzada) y la primera versión de
# esta página publicó un contraste de 94.829.612 y una dimensión intrínseca de
# 0,28, las dos sin sentido y las dos con pinta de cifra. Poner a infinito solo
# las distancias EXACTAMENTE cero no basta: un casi duplicado deja 1e-8, que es
# finito y sobrevive al filtro.
#
# La regla, que es la que usa el propio artículo de Facco: se descartan los
# puntos cuyo vecino más cercano está a menos de DUP_EPS veces la distancia
# típica, se cuentan, y el recuento se publica. Un duplicado no es ruido que
# haya que esconder, es una propiedad del corpus que el lector quiere saber.
DUP_EPS = 1e-4


def drop_duplicates(near, scale):
    """Máscara de puntos cuyo vecino más cercano es otro documento de verdad."""
    return near > DUP_EPS * scale


def two_nn_dimension(X, rng, sample=1500):
    """Estimador de Facco: la razón entre el segundo y el primer vecino.

    La dimensión intrínseca sale de una recta sin término independiente sobre
    log(mu) contra menos log(1 - F), y no depende de ninguna escala. Devuelve
    también cuántos puntos se descartaron por duplicado.
    """
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    S = X[idx]
    d = np.sqrt(np.maximum(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1), 0.0))
    np.fill_diagonal(d, np.inf)
    part = np.sort(d, axis=1)[:, :2]
    scale = float(np.median(part[:, 0][np.isfinite(part[:, 0])]))
    keep = drop_duplicates(part[:, 0], scale)
    part = part[keep]
    mu = part[:, 1] / part[:, 0]
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1)])
    F = np.arange(1, len(mu) + 1) / (len(mu) + 1)
    x = np.log(mu)
    yv = -np.log(1 - F)
    return float((x @ yv) / (x @ x)), int((~keep).sum()), int(len(keep))


def stage_concentration(Z):
    rng = np.random.default_rng(SEED)
    rows = []
    for d in CURSE_DIMS:
        X = rng.standard_normal((2000, d))
        Q = rng.standard_normal((100, d))
        dist = np.sqrt(pairdist(Q, X))
        near = dist.min(1)
        far = dist.max(1)
        rows.append(dict(d=d, contrast=float(np.mean((far - near) / near)),
                         near=float(near.mean()), far=float(far.mean()),
                         ratio=float(np.mean(far / near))))
    def real_row(All, qi, d, extra=None):
        """Contraste sobre consultas cuyo vecino más cercano es otro documento.

        La consulta se excluye POR ÍNDICE y no por valor. La identidad de normas
        deja la distancia de un punto a sí mismo en unos 5e-9 y no en cero, así
        que un `dist[dist == 0] = inf` no la quita: la primera versión de esta
        función tomaba cada consulta como su propio vecino más cercano y
        publicaba un contraste de 94.829.612.
        """
        P = All[qi]
        dist = np.sqrt(np.maximum(
            (All ** 2).sum(1)[None, :] - 2.0 * (P @ All.T) + (P ** 2).sum(1)[:, None], 0.0))
        dist[np.arange(len(qi)), qi] = np.inf
        near = dist.min(1)
        far = np.where(np.isfinite(dist), dist, -np.inf).max(1)
        # la escala sale de la distancia LEJANA, que ningún duplicado contamina
        keep = drop_duplicates(near, float(np.median(far)))
        n, f = near[keep], far[keep]
        row = dict(d=d, contrast=float(np.mean((f - n) / n)),
                   near=float(n.mean()), far=float(f.mean()),
                   ratio=float(np.mean(f / n)),
                   queries=int(keep.sum()), duplicate_queries=int((~keep).sum()))
        row.update(extra or {})
        return row

    qi = rng.choice(len(Z), 100, replace=False)
    dim, dup, seen = two_nn_dimension(Z, rng)
    real = real_row(Z, qi, DIMS,
                    dict(intrinsic=dim, duplicate_points=dup, dim_sample=seen))
    words, W, wmeta, tags = load_words()
    Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    wdim, wdup, wseen = two_nn_dimension(Wn, rng)
    wordrow = real_row(Wn, np.arange(100), int(W.shape[1]),
                       dict(n=int(len(words)), intrinsic=wdim,
                            duplicate_points=wdup, dim_sample=wseen))
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
        found = pad_to([a for a, _ in res], K)
        ivf_rows.append(dict(nprobe=npb, recall=recall_at(found, truth),
                             cost=float(np.mean([c for _, c in res])),
                             quality=category_hits(found, lab_x, lab_q)))
        print(f"    ivf nprobe {npb}: recall {ivf_rows[-1]['recall']:.4f}")

    print("    construyendo el grafo")
    h = HNSW(X, M=M, ef_construction=EF_C, rng=np.random.default_rng(SEED + 2))
    hnsw_rows = []
    for ef in EFS:
        res = [h.search(q, K, ef) for q in Q]
        found = pad_to([a for a, _ in res], K)
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
        d_true = pairdist(Q, X)
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
    # `row` puede traer -1 de relleno cuando el índice devolvió menos de k, y
    # lab_x[-1] es la última etiqueta del conjunto, así que sumaría un acierto
    # fantasma. Se divide entre k y no entre los devueltos: un hueco no es un
    # acierto de categoría, es un resultado que no está.
    k = found.shape[1]
    return float(np.mean([(lab_x[row[row >= 0]] == q).sum() / k
                          for row, q in zip(found, lab_q)]))


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
            found = pad_to([a for a, _ in res], K)
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
        found = pad_to([a for a, _ in res], K)
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
