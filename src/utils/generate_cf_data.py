"""Las cifras del artículo de filtrado colaborativo (módulo 6.1).

El objeto de este módulo es nuevo en el atlas: una tabla de personas por cosas,
casi entera vacía, donde no hay rasgos de nada. `goodbooks-10k` da 5.976.479
valoraciones de 1 a 5 estrellas de 53.424 lectores sobre 10.000 libros **con
títulos**, que es lo que permite juzgar una recomendación mirándola en vez de
leyendo un índice.

Lo que se mide, y en qué orden:

  1. **El agujero**: densidad real de la matriz, y la cola larga contada por los
     dos lados (qué fracción del catálogo hace falta para la mitad de las
     valoraciones).
  2. **Predecir la nota**: media global, media del usuario, media del libro,
     el modelo de sesgos, y kNN por libro y por usuario, todos sobre la misma
     partición. Con barrido de k y de la constante de encogimiento.
  3. **Ordenar**, que es la tarea de verdad: aciertos en el top-k sobre los
     mismos usuarios, con los dos controles que casi nadie publica (lo más
     popular y la nota media más alta) y la cobertura del catálogo.
  4. **Qué significa "parecido"**: cuatro definiciones de similitud sobre los
     mismos pares, con la popularidad de los vecinos que devuelve cada una,
     que es el sesgo que el coseno crudo tiene y nadie mira.
  5. **Dónde falla**: por decil de actividad del usuario y por decil de
     popularidad del libro, más lo que el propio protocolo de evaluación se
     lleva puesto (los libros retenidos no son una muestra del catálogo).

Y exporta los estadísticos suficientes de 60 libros para que el navegador
calcule las cuatro similitudes en vivo, con el guardia que compara su coseno
contra el de aquí.

Ejecutar desde cualquier sitio: `python src/utils/generate_cf_data.py`.
"""
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recsys_data import books, digest, ratings, short_title  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "collaborative-filtering" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "cf_cache")

SEED = 19
HELD_PER_USER = 5          # valoraciones retenidas por usuario
N_EVAL_USERS = 4000        # usuarios sobre los que se mide el ranking
K_NEIGH = 30               # vecinos por defecto
SHRINK = 25.0              # constante de encogimiento por defecto
TOP_K = 10                 # tamaño de la lista recomendada
WIDGET_BOOKS = 60          # libros cuyos estadísticos suficientes viajan
BLOCK = 4000               # usuarios por bloque en los productos grandes
TAG = f"s{SEED}h{HELD_PER_USER}e{N_EVAL_USERS}k{K_NEIGH}"


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
    """Redondeo para el JSON, con los no finitos convertidos en None.

    `json.dumps` escribe `NaN` sin protestar y eso no es JSON: el navegador lo
    rechaza al parsear y la página se queda sin prosa y sin guardia."""
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, dict):
        return {k: r(x, d) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r(x, d) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return None if not np.isfinite(v) else round(v, d)


# --------------------------------------------------------------------------
# Álgebra compartida
# --------------------------------------------------------------------------
def norm_cols(M):
    nn = np.sqrt(np.asarray(M.multiply(M).sum(0)).ravel())
    return sp.csr_matrix(M @ sp.diags((1.0 / np.maximum(nn, 1e-8)).astype(np.float32)),
                         dtype=np.float32)


def dense_gram(M):
    """M normalizada por columnas, y el coseno entre todas ellas."""
    Mn = norm_cols(M)
    return np.asarray((Mn.T @ Mn).todense(), dtype=np.float32)


def topk_sparse(S, k):
    """Se queda con los k vecinos mayores de cada fila, como matriz dispersa."""
    n, m = S.shape
    k = min(k, m - 1)
    idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    vals = S[rows, idx.ravel()]
    keep = vals > 0
    return sp.csr_matrix((vals[keep], (rows[keep], idx.ravel()[keep])), shape=(n, m))


def topk_arrays(S, k):
    """Los k vecinos de cada fila como dos arrays (n x k): índices y pesos.

    La versión dispersa de arriba sirve para puntuar un catálogo entero de una
    vez; para puntuar celdas sueltas es la forma equivocada, porque el producto
    disperso materializa la matriz entera de usuarios por libros (2 GB) para
    leer cinco números de cada fila. Con los vecinos en dos arrays, cada celda
    es una recogida de k valores y la etapa pasa de veinte minutos a uno."""
    n, m = S.shape
    k = min(k, m - 1)
    idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
    vals = np.take_along_axis(S, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    return idx.astype(np.int32), np.where(vals > 0, vals, 0.0).astype(np.float32)


def rmse(pred, truth):
    return float(np.sqrt(np.mean((np.clip(pred, 1.0, 5.0) - truth) ** 2)))


def centred_matrix(train, mu, bu, bi):
    """Los residuos del modelo de sesgos sobre las celdas observadas."""
    coo = train.tocoo()
    data = coo.data.astype(np.float32) - (mu + bu[coo.row] + bi[coo.col]).astype(np.float32)
    return sp.csr_matrix((data, (coo.row, coo.col)), shape=train.shape)


# --------------------------------------------------------------------------
# S0. Datos y partición. Se retiran HELD_PER_USER valoraciones por usuario, y
# la huella del conjunto retenido se exporta para que el artículo siguiente
# pueda afirmar que corre sobre la misma partición en vez de declararlo.
# --------------------------------------------------------------------------
def stage_split():
    u, b, rat = ratings()
    n_u, n_b = int(u.max()) + 1, int(b.max()) + 1
    rng = np.random.default_rng(SEED)
    order = np.lexsort((rng.random(u.size), u))       # barajado dentro de cada usuario
    u_sorted = u[order]
    rank = np.empty(u.size, dtype=np.int64)
    starts = np.searchsorted(u_sorted, np.arange(n_u))
    rank[order] = np.arange(u.size) - starts[u_sorted]
    is_test = rank < HELD_PER_USER
    train = sp.csr_matrix((rat[~is_test].astype(np.float32), (u[~is_test], b[~is_test])),
                          shape=(n_u, n_b))
    test = (u[is_test].astype(np.int64), b[is_test].astype(np.int64),
            rat[is_test].astype(np.float32))
    fp = hashlib.sha256(np.flatnonzero(is_test).astype(np.int64).tobytes()).hexdigest()[:16]
    return train, test, fp, n_u, n_b


# --------------------------------------------------------------------------
# S1. La forma de la tabla y la cola larga.
# --------------------------------------------------------------------------
def stage_shape(train, n_u, n_b):
    per_book = np.asarray((train > 0).sum(0)).ravel()
    per_user = np.asarray((train > 0).sum(1)).ravel()
    order = np.argsort(-per_book)
    cum = np.cumsum(per_book[order]) / per_book.sum()
    counts = np.bincount(train.data.astype(np.int64), minlength=6)[1:]
    ranks = np.unique(np.round(np.geomspace(1, n_b, 70)).astype(int))
    return {
        "users": n_u, "books": n_b,
        "ratings": int(train.nnz + HELD_PER_USER * n_u),
        "train_ratings": int(train.nnz), "cells": int(n_u) * int(n_b),
        "density": float(train.nnz / (n_u * n_b)),
        "half_books": int(np.searchsorted(cum, 0.5) + 1),
        "ninety_books": int(np.searchsorted(cum, 0.9) + 1),
        "per_book_median": float(np.median(per_book)),
        "per_book_max": int(per_book.max()), "per_book_min": int(per_book.min()),
        "per_user_median": float(np.median(per_user)),
        "per_user_min": int(per_user.min()), "per_user_max": int(per_user.max()),
        "star_counts": [int(c) for c in counts],
        "mean_rating": float(train.data.mean()),
        "curve": [[int(k), int(per_book[order][k - 1]), float(cum[k - 1])] for k in ranks],
        "per_book": per_book, "per_user": per_user,
    }


# --------------------------------------------------------------------------
# S2. Predecir la nota.
# --------------------------------------------------------------------------
def fit_bias(train, lam_u=10.0, lam_i=10.0, rounds=10):
    """mu + b_u + b_i, alternando, con encogimiento en las dos direcciones."""
    mu = float(train.data.mean())
    n_u, n_b = train.shape
    coo = train.tocoo()
    ui, ii, val = coo.row, coo.col, coo.data.astype(np.float64)
    cnt_u = np.bincount(ui, minlength=n_u).astype(np.float64)
    cnt_i = np.bincount(ii, minlength=n_b).astype(np.float64)
    bu = np.zeros(n_u)
    bi = np.zeros(n_b)
    for _ in range(rounds):
        bu = np.bincount(ui, weights=val - mu - bi[ii], minlength=n_u) / (cnt_u + lam_u)
        bi = np.bincount(ii, weights=val - mu - bu[ui], minlength=n_b) / (cnt_i + lam_i)
    return mu, bu, bi


def predict_knn_cells(idx, wts, Rc, tu, tb):
    """kNN por libro sobre las celdas de test, por bloques de usuarios.

    Para cada celda (u, i) hacen falta las notas que u dio a los k vecinos de
    i, así que se densifica un bloque de usuarios y se recogen k valores por
    celda. Lo que NO se hace es el producto entero de matrices: la respuesta
    completa son 53.424 x 10.000 números por cada mitad del cociente y aquí se
    leen cinco por usuario."""
    n_u = Rc.shape[0]
    pattern = sp.csr_matrix((np.ones_like(Rc.data), Rc.indices, Rc.indptr), shape=Rc.shape)
    adj = np.zeros(tu.size, dtype=np.float64)
    cov = np.zeros(tu.size, dtype=bool)
    order = np.argsort(tu, kind="stable")
    for a in range(0, n_u, BLOCK):
        m = order[(tu[order] >= a) & (tu[order] < a + BLOCK)]
        if not m.size:
            continue
        Rd = np.asarray(Rc[a:a + BLOCK].todense(), dtype=np.float32)
        Bd = np.asarray(pattern[a:a + BLOCK].todense(), dtype=np.float32)
        rows = (tu[m] - a)[:, None]
        nb, wv = idx[tb[m]], wts[tb[m]]
        num = (wv * Rd[rows, nb]).sum(1)
        den = (wv * Bd[rows, nb]).sum(1)
        adj[m] = np.where(den > 1e-8, num / np.maximum(den, 1e-8), 0.0)
        cov[m] = den > 1e-8
        del Rd, Bd
    return adj, cov


def stage_rating(train, test, shape, co):
    tu, tb, tr = test
    mu, bu, bi = fit_bias(train)
    Rc = centred_matrix(train, mu, bu, bi)
    base = mu + bu[tu] + bi[tb]

    per_user_mean = np.asarray(train.sum(1)).ravel() / np.maximum(shape["per_user"], 1)
    per_book_sum = np.asarray(train.sum(0)).ravel()
    per_book_mean = np.where(shape["per_book"] > 0,
                             per_book_sum / np.maximum(shape["per_book"], 1), mu)
    rows = {
        "global": rmse(np.full(tr.size, mu), tr),
        "user_mean": rmse(per_user_mean[tu], tr),
        "book_mean": rmse(per_book_mean[tb], tr),
        "bias": rmse(base, tr),
    }

    Sfull = dense_gram(Rc)
    np.fill_diagonal(Sfull, 0.0)

    sweep_shrink = []
    for lam in [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0]:
        S = Sfull * (co / (co + lam)) if lam > 0 else Sfull
        adj, _ = predict_knn_cells(*topk_arrays(S, K_NEIGH), Rc, tu, tb)
        sweep_shrink.append({"shrink": lam, "rmse": rmse(base + adj, tr)})
        if lam != SHRINK:
            del S

    S = Sfull * (co / (co + SHRINK))
    sweep_k = []
    for k in [1, 2, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 4000]:
        adj, cov = predict_knn_cells(*topk_arrays(S, k), Rc, tu, tb)
        sweep_k.append({"k": k, "rmse": rmse(base + adj, tr)})
        if k == K_NEIGH:
            rows["item_knn"] = sweep_k[-1]["rmse"]
            covered = float(cov.mean())
    del Sfull

    iu = np.triu_indices(co.shape[0], k=1)
    coa = co[iu]
    pairs = {"pairs": int(coa.size), "under5": float(np.mean(coa < 5)),
             "zero": float(np.mean(coa == 0)), "median": float(np.median(coa))}
    del coa, iu
    return {"rmse": rows, "sweep_k": sweep_k, "sweep_shrink": sweep_shrink,
            "covered": covered, "pairs": pairs}, mu, bu, bi


# --------------------------------------------------------------------------
# S3. Lo mismo por el otro lado, y lo que cuesta.
# --------------------------------------------------------------------------
def stage_user_knn(train, test, mu, bu, bi, S):
    """Vecinos por usuario, y el mismo kNN por libro sobre LAS MISMAS celdas.

    Poner el 0,8713 de aquí al lado del 0,8284 de la tabla de arriba sería
    comparar dos protocolos: aquel se mide sobre las 267.120 celdas y este sobre
    las 20.000 de los usuarios muestreados. La comparación limpia es sobre las
    mismas celdas y con el mismo modelo de sesgos debajo."""
    tu, tb, tr = test
    Rc = centred_matrix(train, mu, bu, bi)
    nn = np.sqrt(np.asarray(Rc.multiply(Rc).sum(1)).ravel())
    Rn = sp.csr_matrix(sp.diags((1.0 / np.maximum(nn, 1e-8)).astype(np.float32)) @ Rc,
                       dtype=np.float32)
    B = sp.csr_matrix((np.ones_like(train.data), train.indices, train.indptr),
                      shape=train.shape)
    rng = np.random.default_rng(SEED + 1)
    users = np.sort(rng.choice(train.shape[0], size=N_EVAL_USERS, replace=False))
    sel = np.isin(tu, users)
    tu_s, tb_s, tr_s = tu[sel], tb[sel], tr[sel]
    pos = np.full(train.shape[0], -1, dtype=np.int64)
    pos[users] = np.arange(users.size)
    local = pos[tu_s]

    adj = np.zeros(tu_s.size)
    step = 400
    for a in range(0, users.size, step):
        blk = users[a:a + step]
        Sim = np.asarray((Rn[blk] @ Rn.T).todense(), dtype=np.float32)
        cnt = np.asarray((B[blk] @ B.T).todense(), dtype=np.float32)
        Sim *= cnt / (cnt + SHRINK)
        Sim[np.arange(blk.size), blk] = 0.0
        Sk = topk_sparse(Sim, K_NEIGH)
        num = np.asarray((Sk @ Rc).todense(), dtype=np.float32)
        den = np.asarray((abs(Sk) @ B).todense(), dtype=np.float32)
        m = (local >= a) & (local < a + blk.size)
        lr = local[m] - a
        d = den[lr, tb_s[m]]
        adj[m] = np.where(d > 1e-8, num[lr, tb_s[m]] / np.maximum(d, 1e-8), 0.0)
        del Sim, cnt, Sk, num, den
    base = mu + bu[tu_s] + bi[tb_s]
    adj_item, _ = predict_knn_cells(*topk_arrays(S, K_NEIGH), Rc, tu_s, tb_s)
    return {
        "rmse": rmse(base + adj, tr_s),
        "item_rmse_same_cells": rmse(base + adj_item, tr_s),
        "bias_rmse_same_users": rmse(base, tr_s),
        "n_test": int(tr_s.size), "n_users": int(users.size),
        "sims_user_full": int(train.shape[0]) ** 2,
        "sims_item": int(train.shape[1]) ** 2,
        "ratio": float(train.shape[0] ** 2 / train.shape[1] ** 2),
    }, users


# --------------------------------------------------------------------------
# S4. Ordenar. Aquí la nota no importa: importa si el libro retenido aparece
# en la lista de diez que el método pondría delante del lector.
# --------------------------------------------------------------------------
def rank_eval(scores, seen, held_list, pop, srt, ks=(1, 3, 5, 10, 20, 50)):
    scores = scores.astype(np.float32, copy=True)
    scores[seen.nonzero()] = -np.inf
    hits = {k: 0.0 for k in ks}
    ndcg, ranks, tops = 0.0, [], []
    for i, items in enumerate(held_list):
        row = scores[i]
        rk = (row > row[items][:, None]).sum(1)          # cuántos van por delante
        ranks.append(rk)
        for k in ks:
            hits[k] += float(np.mean(rk < k))
        ndcg += float(np.mean(np.where(rk < TOP_K, 1.0 / np.log2(rk + 2.0), 0.0)))
        top = np.argpartition(-row, TOP_K)[:TOP_K]
        tops.append(top[np.argsort(-row[top])])
    m = len(held_list)
    ranks = np.concatenate(ranks)
    tops = np.array(tops)
    return {"hits": {str(k): hits[k] / m for k in ks}, "ndcg": ndcg / m,
            "median_rank": float(np.median(ranks)) + 1.0,
            "pop_pct": float(np.mean(np.searchsorted(srt, pop[tops.ravel()]) / pop.size)),
            "coverage": int(np.unique(tops).size)}, tops


def stage_ranking(train, test, S, shape, users, mu, bu, bi):
    tu, tb, tr = test
    n_b = train.shape[1]
    keep = np.isin(tu, users) & (tr >= 4)          # solo lo que al lector le gustó
    held = {}
    for u, b in zip(tu[keep], tb[keep]):
        held.setdefault(int(u), []).append(int(b))
    eval_users = np.array(sorted(held))
    held_list = [np.array(held[u]) for u in eval_users]
    seen = train[eval_users]

    pop = shape["per_book"].astype(np.float64)
    srt = np.sort(pop)
    book_mean = np.where(pop > 0, np.asarray(train.sum(0)).ravel() / np.maximum(pop, 1), 0.0)
    rng = np.random.default_rng(SEED + 2)

    out, lists = {}, {}
    def run(name, scores):
        out[name], lists[name] = rank_eval(scores, seen, held_list, pop, srt)

    run("random", rng.random((eval_users.size, n_b)).astype(np.float32))
    run("popular", np.tile(pop.astype(np.float32), (eval_users.size, 1)))
    run("book_mean", np.tile(book_mean.astype(np.float32), (eval_users.size, 1)))
    Sk = topk_sparse(S, K_NEIGH)
    B = sp.csr_matrix((np.ones_like(seen.data), seen.indices, seen.indptr), shape=seen.shape)
    run("item_knn", np.asarray((B @ Sk.T).todense(), dtype=np.float32))
    Rc = centred_matrix(train, mu, bu, bi)[eval_users]
    run("item_knn_rating", np.asarray((Rc @ Sk.T).todense(), dtype=np.float32))

    all_held = np.concatenate(held_list)
    meta = {
        "held_pop_pct": float(np.mean(np.searchsorted(srt, pop[all_held]) / pop.size)),
        "eval_users": int(eval_users.size), "held_items": int(all_held.size),
        "catalogue": int(n_b),
        "top_overlap": float(np.mean(np.isin(lists["item_knn"], lists["popular"][0]))),
    }
    return out, meta, eval_users, held_list


# --------------------------------------------------------------------------
# S5. Qué significa parecido: cuatro similitudes sobre los mismos pares.
# --------------------------------------------------------------------------
def stage_flavours(train, Rc, shape, co):
    pop = shape["per_book"].astype(np.float64)
    srt = np.sort(pop)
    n_i = shape["per_book"].astype(np.float32)
    B = sp.csr_matrix((np.ones_like(train.data), train.indices, train.indptr),
                      shape=train.shape)
    stats = {}

    def score(name, S):
        np.fill_diagonal(S, 0.0)
        top = np.argpartition(-S, kth=8, axis=1)[:, :8]
        pct = (np.searchsorted(srt, pop[top.ravel()]) / pop.size).reshape(-1, 8)
        cop = co[np.repeat(np.arange(S.shape[0]), 8), top.ravel()]
        stats[name] = {"pop_pct": float(pct.mean()),
                       "co_median": float(np.median(cop)),
                       "co_under5": float(np.mean(cop < 5)),
                       "top_pop_pct": float(pct[np.argsort(-pop)[:100]].mean())}

    score("jaccard", co / np.maximum(n_i[:, None] + n_i[None, :] - co, 1e-8))
    score("cosine", dense_gram(B))
    Sc = dense_gram(Rc)
    score("centred", Sc.copy())
    score("shrunk", Sc * (co / (co + SHRINK)))
    del Sc
    return stats


# --------------------------------------------------------------------------
# S6. Los estadísticos suficientes que viajan al navegador.
# --------------------------------------------------------------------------
def stage_widget(train, Rc, shape, titles, authors):
    idx = np.sort(np.argsort(-shape["per_book"])[:WIDGET_BOOKS])
    B = sp.csr_matrix((np.ones_like(train.data), train.indices, train.indptr),
                      shape=train.shape)[:, idx]
    Rs, Cs = train[:, idx], Rc[:, idx]
    co = np.asarray((B.T @ B).todense(), dtype=np.float64)
    dot_raw = np.asarray((Rs.T @ Rs).todense(), dtype=np.float64)
    dot_c = np.asarray((Cs.T @ Cs).todense(), dtype=np.float64)
    n_i = np.asarray(B.sum(0)).ravel()
    norm_raw = np.sqrt(np.asarray(Rs.multiply(Rs).sum(0)).ravel())
    norm_c = np.sqrt(np.asarray(Cs.multiply(Cs).sum(0)).ravel())
    sums = np.asarray(Rs.sum(0)).ravel()
    pop = shape["per_book"].astype(np.float64)
    srt = np.sort(pop)
    iu = np.triu_indices(len(idx), k=1)
    items = [{
        "id": int(i), "title": short_title(titles[i]),
        "author": authors[i].split(",")[0],
        "n": int(n_i[j]), "rank": int((pop > pop[i]).sum()) + 1,
        "pct": float(np.searchsorted(srt, pop[i]) / pop.size),
        "mean": float(sums[j] / max(n_i[j], 1)),
        "norm_raw": float(norm_raw[j]), "norm_c": float(norm_c[j]),
    } for j, i in enumerate(idx)]
    # La referencia del guardia: las cuatro similitudes de 24 pares, calculadas
    # aquí desde las columnas y NO desde el triángulo empaquetado que viaja, que
    # es lo único que el navegador puede leer mal.
    rng = np.random.default_rng(SEED + 7)
    pairs = set()
    while len(pairs) < 24:
        a, b = rng.integers(0, len(idx), size=2)
        if a != b:
            pairs.add((int(min(a, b)), int(max(a, b))))
    check = []
    for a, b in sorted(pairs):
        ca = np.asarray(B[:, a].todense()).ravel()
        cb = np.asarray(B[:, b].todense()).ravel()
        ra = np.asarray(Rs[:, a].todense()).ravel()
        rb = np.asarray(Rs[:, b].todense()).ravel()
        ea = np.asarray(Cs[:, a].todense()).ravel()
        eb = np.asarray(Cs[:, b].todense()).ravel()
        both = float(ca @ cb)
        check.append({
            "i": a, "j": b,
            "count": both / float(np.sqrt(ca.sum() * cb.sum())),
            "jaccard": both / float(ca.sum() + cb.sum() - both),
            "rating": float(ra @ rb) / float(np.linalg.norm(ra) * np.linalg.norm(rb)),
            "centred": float(ea @ eb) / float(np.linalg.norm(ea) * np.linalg.norm(eb)),
        })
    return {"items": items, "co": [int(v) for v in co[iu]],
            "dot_raw": [float(v) for v in dot_raw[iu]],
            "dot_c": [float(v) for v in dot_c[iu]], "check": check}


# --------------------------------------------------------------------------
# S7. Dónde falla: por decil de actividad y por decil de popularidad.
# --------------------------------------------------------------------------
def stage_slices(train, test, S, shape, mu, bu, bi):
    tu, tb, tr = test
    per_user = shape["per_user"]
    pop = shape["per_book"].astype(np.float64)
    Rc = centred_matrix(train, mu, bu, bi)
    Sk = topk_sparse(S, K_NEIGH)
    adj, covered = predict_knn_cells(*topk_arrays(S, K_NEIGH), Rc, tu, tb)
    base = mu + bu[tu] + bi[tb]
    pred = base + adj

    out = {"user": [], "book": []}
    qu = np.quantile(per_user[tu], np.linspace(0, 1, 11))
    for d in range(10):
        m = (per_user[tu] >= qu[d]) & (per_user[tu] <= qu[d + 1])
        out["user"].append({"decile": d + 1, "n": int(m.sum()),
                            "activity": float(np.median(per_user[tu][m])),
                            "knn": rmse(pred[m], tr[m]), "bias": rmse(base[m], tr[m])})
    qb = np.quantile(pop[tb], np.linspace(0, 1, 11))
    for d in range(10):
        m = (pop[tb] >= qb[d]) & (pop[tb] <= qb[d + 1])
        out["book"].append({"decile": d + 1, "n": int(m.sum()),
                            "popularity": float(np.median(pop[tb][m])),
                            "knn": rmse(pred[m], tr[m]), "bias": rmse(base[m], tr[m]),
                            "covered": float(covered[m].mean())})
    reach = np.asarray((Sk > 0).sum(0)).ravel()
    out["unreachable"] = int((reach == 0).sum())
    out["thin"] = int((shape["per_book"] < 20).sum())
    out["knn_all"] = rmse(pred, tr)
    out["bias_all"] = rmse(base, tr)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("cargando y partiendo")
    train, test, fp, n_u, n_b = cached("split", stage_split)
    titles, authors, _ = books()
    shape = cached("shape", lambda: stage_shape(train, n_u, n_b))
    print(f"  {n_u} usuarios x {n_b} libros, {train.nnz} de entrenamiento, "
          f"densidad {shape['density']:.4%}")

    B = sp.csr_matrix((np.ones_like(train.data), train.indices, train.indptr),
                      shape=train.shape)
    co = np.asarray((B.T @ B).todense(), dtype=np.float32)
    del B

    print("prediciendo la nota")
    rating, mu, bu, bi = cached("rating", lambda: stage_rating(train, test, shape, co))
    print(f"  bias {rating['rmse']['bias']:.4f}  item-knn {rating['rmse']['item_knn']:.4f}")

    Rc = centred_matrix(train, mu, bu, bi)
    S = dense_gram(Rc)
    np.fill_diagonal(S, 0.0)
    S *= co / (co + SHRINK)

    print("por el otro lado, usuario contra usuario")
    userknn, users = cached("userknn", lambda: stage_user_knn(train, test, mu, bu, bi, S))
    print(f"  user-knn {userknn['rmse']:.4f} sobre {userknn['n_test']} celdas")

    print("ordenando")
    rank, rank_meta, eval_users, held_list = cached(
        "rank", lambda: stage_ranking(train, test, S, shape, users, mu, bu, bi))
    print(f"  popular @10 {rank['popular']['hits']['10']:.4f}  "
          f"item-knn @10 {rank['item_knn']['hits']['10']:.4f}")

    print("cuatro similitudes")
    flav = cached("flavours", lambda: stage_flavours(train, Rc, shape, co))

    print("estadísticos suficientes para el navegador")
    widget = cached("widget", lambda: stage_widget(train, Rc, shape, titles, authors))

    print("dónde falla")
    slices = cached("slices", lambda: stage_slices(train, test, S, shape, mu, bu, bi))

    data = {
        "meta": {
            "seed": SEED, "held_per_user": HELD_PER_USER, "k": K_NEIGH, "shrink": SHRINK,
            "top_k": TOP_K, "split_fingerprint": fp,
            "digest_ratings": digest("goodbooks_ratings.csv"),
            "digest_books": digest("goodbooks_books.csv"),
            "test_cells": int(HELD_PER_USER * n_u),
        },
        # La cuenta de cada libro, ordenada, va entera: son 10.000 enteros y con
        # ellos el navegador calcula la curva, el umbral y las dos fracciones de
        # forma exacta, en vez de interpolar sobre una curva submuestreada.
        "shape": {**{k: r(v) for k, v in shape.items()
                     if k not in ("per_book", "per_user", "curve")},
                  "counts_sorted": [int(v) for v in np.sort(shape["per_book"])[::-1]]},
        "rating": r(rating, 4),
        "user_knn": r(userknn, 4),
        "rank": r(rank, 4),
        "rank_meta": r(rank_meta, 4),
        "flavours": r(flav, 4),
        "widget": {"items": r(widget["items"], 4), "co": widget["co"],
                   "dot_raw": r(widget["dot_raw"], 3), "dot_c": r(widget["dot_c"], 3),
                   "check": r(widget["check"], 9)},
        "slices": r(slices, 4),
    }
    path = OUT / "cf.json"
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    print(f"escrito {path} ({path.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
