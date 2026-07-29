"""Las cifras del artículo de bases de datos vectoriales.

El artículo anterior mide un índice sobre un montón de vectores quietos. Una
base de datos no es eso: los vectores llegan y se van, traen columnas que no son
vectores, y la consulta casi nunca es "los diez más parecidos" a secas, sino
"los diez más parecidos **de entre los que cumplen esto**". Cada una de esas
tres diferencias rompe algo que el artículo anterior daba por hecho, y las tres
se pueden medir.

Los datos son los mismos: las noticias de Reuters con una sola etiqueta, sus
ocho categorías y su partición ModApte, que es una partición **por fecha** y por
tanto una columna de metadatos de verdad y no una inventada.

Lo que se mide:

  1. **Filtrar después contra filtrar antes**, con la selectividad barrida. El
     recall de filtrar después se hunde de una forma que se puede predecir, y la
     razón por la que filtrar dentro del grafo tampoco funciona es medible: el
     subgrafo inducido por el filtro **se parte en trozos**, y se cuentan.
  2. **Borrar y volver a insertar.** Un índice de grafo no borra, marca; se mide
     qué le pasa al recall según se acumulan marcas, y qué cuesta reconstruir.
  3. **La búsqueda híbrida.** BM25 sobre los mismos conteos del artículo 44
     contra los vectores densos, y las dos fusionadas por rango recíproco, con
     las categorías como clave de respuestas y consultas de dos longitudes,
     porque el resultado depende de eso y casi nadie lo dice.
  4. **La factura**, en bytes por vector y con la calidad al lado: float32,
     enteros de ocho bits y cuantización por producto.
  5. **Y la pregunta que ninguna comparación de índices contesta**: cuánto de la
     calidad final se pierde por cada punto de recall, que es lo que decide si
     merece la pena pagar por el recall que falta.

Ejecutar desde cualquier sitio: `python src/utils/generate_vectordb_data.py`.
"""
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annkit import HNSW, PQ, brute, recall_at  # noqa: E402
from nlp_data import reuters, stopwords, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "vector-db" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "vectordb_cache")

SEED = 19
R8 = ("acq", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade")
DIMS = 64
N_INDEX = 6000
N_QUERY = 200
K = 10
M = 16
EF_C = 80
EF = 64
SELECTIVITIES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)
CHURN = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)
QUERY_LENGTHS = (3, 10)
RRF_K = 60

TAG = f"d{DIMS}n{N_INDEX}q{N_QUERY}k{K}m{M}e{EF_C}f{EF}v{SEED}"


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


def corpus():
    docs, cats, split = reuters()
    keep = [i for i, c in enumerate(cats) if len(c) == 1 and c[0] in R8]
    texts = [tokenize(docs[i]) for i in keep]
    lab = np.array([R8.index(cats[i][0]) for i in keep])
    later = np.array([split[i] == "test" for i in keep])
    vocab = Counter()
    for t in texts:
        vocab.update(set(t))
    words = sorted([w for w, c in vocab.items() if c >= 5])
    idx = {w: k for k, w in enumerate(words)}
    C = np.zeros((len(texts), len(words)), np.float32)
    for i, t in enumerate(texts):
        for w in t:
            j = idx.get(w)
            if j is not None:
                C[i, j] += 1.0
    df = (C > 0).sum(0)
    T = np.log1p(C) * np.log(len(texts) / np.maximum(df, 1)).astype(np.float32)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    U, S, _ = np.linalg.svd(T - T.mean(0), full_matrices=False)
    Z = (U[:, :DIMS] * S[:DIMS]).astype(np.float64)
    return dict(Z=Z, counts=C, df=df, words=words, lab=lab, later=later,
                texts=texts)


# ------------------------------------------------------------------ 1. filtrar
def stage_filter(data):
    Z, lab = data["Z"], data["lab"]
    rng = np.random.default_rng(SEED)
    qi = rng.choice(len(Z), N_QUERY, replace=False)
    mask = np.ones(len(Z), bool)
    mask[qi] = False
    X = Z[mask][:N_INDEX]
    Q = Z[qi]
    h = HNSW(X, M=M, ef_construction=EF_C, rng=np.random.default_rng(SEED + 2))
    rows = []
    for sel in SELECTIVITIES:
        g = np.random.default_rng(SEED + int(sel * 1000))
        allowed = np.zeros(len(X), bool)
        allowed[g.choice(len(X), max(K, int(round(sel * len(X)))), replace=False)] = True
        sub = np.where(allowed)[0]
        # la verdad: los k más cercanos DENTRO del subconjunto
        truth, pre_cost = brute(X[sub], Q, min(K, len(sub)))
        truth = sub[truth]
        # filtrar después: pedir k al índice y tirar lo que no cumple
        post, post_cost, empty = [], [], 0
        for q in Q:
            got, c = h.search(q, K, EF)
            keep = [int(i) for i in got if allowed[i]]
            post.append(keep + [-1] * (K - len(keep)))
            post_cost.append(c)
            if not keep:
                empty += 1
        post = np.array(post)
        # filtrar después con sobremuestreo: pedir k/sel
        over, over_cost = [], []
        for q in Q:
            want = int(min(len(X), np.ceil(K / max(sel, 1e-9))))
            got, c = h.search(q, want, max(EF, want))
            keep = [int(i) for i in got if allowed[i]][:K]
            over.append(keep + [-1] * (K - len(keep)))
            over_cost.append(c)
        over = np.array(over)
        # el subgrafo inducido: en cuántos trozos se parte
        pieces, biggest = connected_pieces(h.graph[0], allowed)
        rows.append(dict(selectivity=sel, kept=int(allowed.sum()),
                         post_recall=recall_at(post, truth),
                         post_cost=float(np.mean(post_cost)),
                         post_empty=empty / len(Q),
                         over_recall=recall_at(over, truth),
                         over_cost=float(np.mean(over_cost)),
                         pre_cost=float(pre_cost / len(Q)),
                         pieces=pieces, biggest=biggest / max(1, int(allowed.sum()))))
        print(f"    selectividad {sel}: recall filtrando después "
              f"{rows[-1]['post_recall']:.4f}, trozos {pieces}")
    # el mismo barrido con un filtro que NO es aleatorio: la categoría
    cat_rows = []
    lab_x = lab[mask][:N_INDEX]
    for c, name in enumerate(R8):
        allowed = lab_x == c
        if allowed.sum() < K:
            continue
        sub = np.where(allowed)[0]
        truth, _ = brute(X[sub], Q, K)
        truth = sub[truth]
        post = []
        for q in Q:
            got, _ = h.search(q, K, EF)
            keep = [int(i) for i in got if allowed[i]]
            post.append(keep + [-1] * (K - len(keep)))
        pieces, biggest = connected_pieces(h.graph[0], allowed)
        cat_rows.append(dict(category=name, kept=int(allowed.sum()),
                             selectivity=float(allowed.mean()),
                             post_recall=recall_at(np.array(post), truth),
                             pieces=pieces,
                             biggest=biggest / int(allowed.sum())))
    return dict(rows=rows, categories=cat_rows, n=int(len(X)), ef=EF)


def connected_pieces(graph, allowed):
    """Cuántos trozos deja el filtro en el grafo, y cómo de grande es el mayor.

    Es la razón mecánica de que buscar dentro del grafo saltando los que no
    cumplen no funcione: la búsqueda es un paseo, y un paseo no cruza a otra
    componente.
    """
    seen = set()
    sizes = []
    nodes = [i for i in graph if allowed[i]]
    for s in nodes:
        if s in seen:
            continue
        stack, size = [s], 0
        seen.add(s)
        while stack:
            v = stack.pop()
            size += 1
            for w in graph.get(v, ()):
                if allowed[w] and w not in seen:
                    seen.add(w)
                    stack.append(w)
        sizes.append(size)
    return len(sizes), max(sizes) if sizes else 0


# -------------------------------------------------------------------- 2. borrar
def stage_churn(data):
    Z = data["Z"]
    rng = np.random.default_rng(SEED)
    qi = rng.choice(len(Z), N_QUERY, replace=False)
    mask = np.ones(len(Z), bool)
    mask[qi] = False
    X = Z[mask][:N_INDEX]
    Q = Z[qi]
    h = HNSW(X, M=M, ef_construction=EF_C, rng=np.random.default_rng(SEED + 2))
    rows = []
    for frac in CHURN:
        g = np.random.default_rng(SEED + int(frac * 100))
        dead = np.zeros(len(X), bool)
        dead[g.choice(len(X), int(round(frac * len(X))), replace=False)] = True
        alive = np.where(~dead)[0]
        truth, _ = brute(X[alive], Q, K)
        truth = alive[truth]
        tomb, cost = [], []
        for q in Q:
            got, c = h.search(q, K * 3, max(EF, K * 3))
            keep = [int(i) for i in got if not dead[i]][:K]
            tomb.append(keep + [-1] * (K - len(keep)))
            cost.append(c)
        rebuilt = None
        if frac in (0.3, 0.7):
            h2 = HNSW(X[alive], M=M, ef_construction=EF_C,
                      rng=np.random.default_rng(SEED + 2))
            got = np.array([h2.search(q, K, EF)[0] for q in Q])
            rebuilt = recall_at(alive[got], truth)
        rows.append(dict(deleted=frac, alive=int(len(alive)),
                         tombstone_recall=recall_at(np.array(tomb), truth),
                         cost=float(np.mean(cost)),
                         rebuilt_recall=rebuilt,
                         build_cost=int(h.build_cost)))
        print(f"    borrado {frac:.0%}: recall con marcas "
              f"{rows[-1]['tombstone_recall']:.4f}")
    return dict(rows=rows, n=int(len(X)))


# ------------------------------------------------------------------- 3. híbrida
def bm25(counts, df, n_docs, query_idx, k1=1.5, b=0.75):
    lens = counts.sum(1)
    avg = lens.mean()
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
    tf = counts[:, query_idx]
    denom = tf + k1 * (1 - b + b * lens[:, None] / avg)
    return (idf[query_idx] * tf * (k1 + 1) / np.maximum(denom, 1e-9)).sum(1)


def ndcg(order, lab_x, target, k=K):
    rel = (lab_x[order[:k]] == target).astype(float)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.sort(rel)[::-1]
    d = (ideal * disc).sum()
    return float((rel * disc).sum() / d) if d > 0 else 0.0


def stage_hybrid(data):
    Z, C, df, lab = data["Z"], data["counts"], data["df"], data["lab"]
    words = data["words"]
    stop = set(stopwords())
    rng = np.random.default_rng(SEED)
    qi = rng.choice(len(Z), N_QUERY, replace=False)
    mask = np.ones(len(Z), bool)
    mask[qi] = False
    keepx = np.where(mask)[0][:N_INDEX]
    X, Cx, lab_x = Z[keepx], C[keepx], lab[keepx]
    dfx = (Cx > 0).sum(0)
    rows = []
    for L in QUERY_LENGTHS:
        dense, sparse, fused, oracle = [], [], [], []
        for q in qi:
            toks = [w for w in data["texts"][q] if w in set(words) and w not in stop]
            if len(toks) < L:
                continue
            counts = Counter(toks)
            top = [w for w, _ in counts.most_common(L)]
            widx = np.array([words.index(w) for w in top])
            s = bm25(Cx, dfx, len(Cx), widx)
            dv = -((X - Z[q]) ** 2).sum(1)
            os_ = np.argsort(-s)
            od = np.argsort(-dv)
            rs = np.empty(len(X), int)
            rs[os_] = np.arange(len(X))
            rd = np.empty(len(X), int)
            rd[od] = np.arange(len(X))
            f = 1.0 / (RRF_K + rs + 1) + 1.0 / (RRF_K + rd + 1)
            of = np.argsort(-f)
            sparse.append(ndcg(os_, lab_x, lab[q]))
            dense.append(ndcg(od, lab_x, lab[q]))
            fused.append(ndcg(of, lab_x, lab[q]))
            oracle.append(max(sparse[-1], dense[-1]))
        rows.append(dict(length=L, queries=len(dense),
                         sparse=float(np.mean(sparse)), dense=float(np.mean(dense)),
                         fused=float(np.mean(fused)), oracle=float(np.mean(oracle)),
                         fused_beats_both=float(np.mean(
                             [f > max(a, b) for f, a, b in zip(fused, sparse, dense)])),
                         dense_beats_sparse=float(np.mean(
                             [d > s for d, s in zip(dense, sparse)]))))
        print(f"    consultas de {L} palabras: densa {rows[-1]['dense']:.4f}, "
              f"BM25 {rows[-1]['sparse']:.4f}, fusionadas {rows[-1]['fused']:.4f}")
    return dict(rows=rows, rrf_k=RRF_K, n=int(len(X)))


# ------------------------------------------------------------------ 4 y 5. bytes
def stage_bytes(data):
    Z, lab = data["Z"], data["lab"]
    rng = np.random.default_rng(SEED)
    qi = rng.choice(len(Z), N_QUERY, replace=False)
    mask = np.ones(len(Z), bool)
    mask[qi] = False
    X = Z[mask][:N_INDEX]
    lab_x = lab[mask][:N_INDEX]
    Q, lab_q = Z[qi], lab[qi]
    truth, _ = brute(X, Q, K)
    rows = []

    def quality(found):
        return float(np.mean([(lab_x[row] == q).mean()
                              for row, q in zip(found, lab_q)]))

    rows.append(dict(scheme="float32", bytes=float(DIMS * 4), recall=1.0,
                     quality=quality(truth)))
    lo, hi = X.min(0), X.max(0)
    q8 = np.round((X - lo) / np.maximum(hi - lo, 1e-12) * 255)
    rec8 = q8 / 255 * (hi - lo) + lo
    found = np.argsort(((Q[:, None, :] - rec8[None, :, :]) ** 2).sum(-1), 1)[:, :K]
    rows.append(dict(scheme="int8 per column", bytes=float(DIMS + 8),
                     recall=recall_at(found, truth), quality=quality(found)))
    for m in (4, 8, 16):
        pq = PQ(X, m, 8, np.random.default_rng(SEED + 3))
        adc = np.stack([pq.adc(q) for q in Q])
        found = np.argsort(adc, 1)[:, :K]
        rows.append(dict(scheme=f"product quantiser, {m} chunks",
                         bytes=pq.bytes_per_vector(),
                         recall=recall_at(found, truth), quality=quality(found)))
    # el rerank: pedir más con el índice barato y reordenar con el caro
    pq = PQ(X, 8, 8, np.random.default_rng(SEED + 3))
    adc = np.stack([pq.adc(q) for q in Q])
    rer = []
    for w in (10, 20, 50, 100, 200):
        cand = np.argsort(adc, 1)[:, :w]
        out = []
        for i, q in enumerate(Q):
            d = ((X[cand[i]] - q) ** 2).sum(1)
            out.append(cand[i][np.argsort(d)[:K]])
        out = np.array(out)
        rer.append(dict(width=w, recall=recall_at(out, truth), quality=quality(out),
                        exact_dists=float(w)))
    return dict(rows=rows, rerank=rer, n=int(len(X)), dims=DIMS)


def main():
    print("  cargando Reuters")
    data = cached("corpus", corpus)
    print(f"  {len(data['Z'])} noticias, {len(data['words'])} palabras")
    print("  1/4 filtrar")
    filt = cached("filter", lambda: stage_filter(data))
    print("  2/4 borrar")
    churn = cached("churn", lambda: stage_churn(data))
    print("  3/4 híbrida")
    hyb = cached("hybrid", lambda: stage_hybrid(data))
    print("  4/4 la factura")
    by = cached("bytes", lambda: stage_bytes(data))

    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(
        meta=dict(seed=SEED, dims=DIMS, docs=int(len(data["Z"])),
                  vocab=int(len(data["words"])), index_n=N_INDEX,
                  queries=N_QUERY, k=K, M=M, ef=EF, classes=list(R8),
                  later=int(data["later"].sum()),
                  note="the answer key is the Reuters category, and cost is "
                       "counted in distance evaluations"),
        filter=r(filt),
        churn=r(churn),
        hybrid=r(hyb),
        bytes=r(by),
    )
    path = OUT / "vectordb.json"
    path.write_text(json.dumps(out, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
