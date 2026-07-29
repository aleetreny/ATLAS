"""Las cifras del artículo de two-tower y DeepFM (módulo 6.1).

Los dos artículos anteriores solo tienen identificadores: un libro es su columna
y nada más. Aquí llegan los rasgos (autor, año, palabras del título) y con ellos
las dos preguntas que definen este escalón:

  1. **¿Qué compra un rasgo?** Un libro que nadie ha valorado no tiene columna,
     pero sí tiene autor. Se mide con el experimento que ni el filtrado
     colaborativo ni la factorización pueden hacer: 300 libros con **todas** sus
     valoraciones borradas del entrenamiento, puntuados con el mismo protocolo
     de ranking de los dos artículos anteriores.
  2. **¿Qué cuesta la forma del modelo?** Una torre por lado y un producto
     escalar al final es una restricción de **rango**, no una elección de
     arquitectura: la matriz de puntuaciones que un modelo así puede escribir
     tiene rango k como mucho, así que su mejor ajuste posible es el SVD
     truncado de la matriz verdadera. Eso es una cota exacta y aquí se calcula,
     se dibuja, y se compara con lo que el modelo entrenado alcanza.

Y en medio, lo que las máquinas de factorización resuelven: el modelo lineal no
puede decir "a este lector le gusta este autor" y el de segundo orden completo
tiene un peso por par de rasgos, la mayoría de los cuales no aparecen juntos
nunca. Se cuenta cuántos.

Ejecutar desde cualquier sitio: `python src/utils/generate_tower_data.py`.
"""
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recsys_data import books, digest, short_title  # noqa: E402
import generate_cf_data as CF  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "two-tower" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "tower_cache")

SEED = 19
K = 32                 # dimensión de las dos torres
EPOCHS = 6
BATCH = 1024
NEG = 256              # negativos compartidos por lote (softmax muestreado)
# La tasa de aprendizaje NO es un detalle aqui y por eso viaja en la clave del
# cache: con el paso escalado por 1/lote, 0,05 deja las torres literalmente en
# el azar sobre los libros frios y por debajo del control de popularidad sobre
# los calientes. Medido sobre 400.000 interacciones y dos pasadas: 0,0099 de
# aciertos en diez con 0,05 y 0,0552 con 1,0. Un experimento a ese primer nivel
# no mide lo que dice medir, mide que el modelo no ha aprendido.
LR = 1.0
COLD_BOOKS = 300       # libros con todas sus valoraciones borradas
TOP_TOKENS = 2000      # palabras de título que se quedan como rasgo
TOP_AUTHORS = 1500
WIDGET_BOOKS = 400
TAG = f"s{SEED}k{K}e{EPOCHS}n{NEG}c{COLD_BOOKS}lr{LR}"


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


r = CF.r
STOP = {"the", "a", "an", "of", "and", "in", "to", "for", "on", "at", "is", "it",
        "de", "la", "el", "los", "las", "der", "die", "das", "un", "una"}


# --------------------------------------------------------------------------
# Los rasgos de un libro, que es lo único que un libro frío tiene.
# --------------------------------------------------------------------------
def item_features(titles, authors, years, n_books):
    """Matriz dispersa (libros x rasgos): autor, década y palabras del título."""
    counts = {}
    toks = []
    for i in range(n_books):
        t = [w for w in re.findall(r"[a-z']+", titles[i].lower()) if w not in STOP and len(w) > 2]
        toks.append(t)
        for w in set(t):
            counts[w] = counts.get(w, 0) + 1
    vocab = {w: j for j, (w, _) in enumerate(
        sorted(counts.items(), key=lambda kv: -kv[1])[:TOP_TOKENS])}

    acount = {}
    first = []
    for i in range(n_books):
        a = authors[i].split(",")[0].strip()
        first.append(a)
        acount[a] = acount.get(a, 0) + 1
    avocab = {a: j for j, (a, _) in enumerate(
        sorted(acount.items(), key=lambda kv: -kv[1])[:TOP_AUTHORS])}

    rows, cols = [], []
    off_auth = len(vocab)
    off_dec = off_auth + len(avocab) + 1
    decades = sorted({(y // 10) * 10 for y in years if y is not None})
    dec_idx = {d: j for j, d in enumerate(decades)}
    for i in range(n_books):
        for w in set(toks[i]):
            if w in vocab:
                rows.append(i)
                cols.append(vocab[w])
        rows.append(i)
        cols.append(off_auth + avocab.get(first[i], len(avocab)))
        y = years[i]
        if y is not None and (y // 10) * 10 in dec_idx:
            rows.append(i)
            cols.append(off_dec + dec_idx[(y // 10) * 10])
    n_feat = off_dec + len(dec_idx)
    F = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(n_books, n_feat))
    meta = {"tokens": len(vocab), "authors": len(avocab), "decades": len(dec_idx),
            "features": int(n_feat),
            "per_book": float(np.asarray(F.sum(1)).mean()),
            "unknown_author_books": int(sum(1 for a in first if a not in avocab))}
    return F, meta, vocab, avocab


# --------------------------------------------------------------------------
# La torre. Un vector por lado y un producto escalar, entrenado con softmax
# muestreado, que es lo que se usa cuando el catálogo no cabe en la salida.
# --------------------------------------------------------------------------
class Tower:
    """item = (id) + (suma de sus rasgos), con las dos partes conmutables.

    `use_id=False` es el brazo que solo puede mirar el contenido, y es el que
    contesta la pregunta del arranque en frío. `id_drop` apaga el identificador
    en una fracción de los pasos, que es lo que obliga a la parte de contenido a
    valer para algo cuando el identificador ya lo explica todo."""

    def __init__(self, n_users, n_items, F, k, use_id=True, use_content=True,
                 id_drop=0.0, seed=SEED):
        rng = np.random.default_rng(seed)
        self.k, self.use_id, self.use_content, self.id_drop = k, use_id, use_content, id_drop
        self.U = rng.normal(0, 0.05, (n_users, k))
        self.V = rng.normal(0, 0.05, (n_items, k)) if use_id else None
        self.F = F.tocsr()
        self.W = rng.normal(0, 0.05, (F.shape[1], k)) if use_content else None
        self.bias = np.zeros(n_items)
        self.rng = rng

    def item_vec(self, idx, drop_id=False):
        out = np.zeros((len(idx), self.k))
        if self.use_id and not drop_id:
            out += self.V[idx]
        if self.use_content:
            out += self.F[idx] @ self.W
        return out

    def all_items(self, drop_id=False):
        out = np.zeros((self.F.shape[0], self.k))
        if self.use_id and not drop_id:
            out += self.V
        if self.use_content:
            out += self.F @ self.W
        return out

    def fit(self, users, items, n_items, epochs=EPOCHS, batch=BATCH, neg=NEG, lr=LR):
        n = users.size
        for ep in range(epochs):
            order = self.rng.permutation(n)
            for a in range(0, n, batch):
                sel = order[a:a + batch]
                u, i = users[sel], items[sel]
                cand = np.concatenate([i, self.rng.integers(0, n_items, neg)])
                drop = self.use_content and self.rng.random() < self.id_drop
                P = self.U[u]                                   # (B, k)
                Q = self.item_vec(cand, drop_id=drop)            # (B+neg, k)
                logits = P @ Q.T + self.bias[cand]
                logits -= logits.max(1, keepdims=True)
                pr = np.exp(logits)
                pr /= pr.sum(1, keepdims=True)
                g = -pr
                g[np.arange(len(u)), np.arange(len(u))] += 1.0   # el positivo es su columna
                gU = g @ Q
                gQ = g.T @ P
                gb = g.sum(0)
                scale = lr / len(u)
                np.add.at(self.U, u, scale * gU)
                if self.use_id and not drop:
                    np.add.at(self.V, cand, scale * gQ)
                if self.use_content:
                    self.W += scale * (self.F[cand].T @ gQ)
                np.add.at(self.bias, cand, scale * gb)
        return self


def rank_metrics(scores, seen, held_list, pop, srt):
    stats, tops = CF.rank_eval(scores.astype(np.float32), seen, held_list, pop, srt)
    return stats


# --------------------------------------------------------------------------
# S1. El techo exacto de un producto escalar.
# --------------------------------------------------------------------------
def stage_ceiling():
    """Un mundo pequeño donde la verdad se conoce y no es de rango bajo.

    La puntuación verdadera de (usuario, artículo) tiene una parte de gustos
    (que sí es de rango bajo) y una parte de **cruce**: un término que depende
    de los dos a la vez y que no se puede escribir como producto de un vector
    por lado. Con la verdad en la mano, el mejor ajuste posible de un modelo de
    dos torres de dimensión k es el SVD truncado de esa matriz, que es una
    cota, no un resultado de entrenamiento. Y un modelo que ve el par entero no
    la tiene."""
    rng = np.random.default_rng(SEED)
    n_u, n_i, d = 200, 300, 6
    A = rng.normal(size=(n_u, d))
    B = rng.normal(size=(n_i, d))
    taste = A @ B.T / np.sqrt(d)
    # El cruce tiene que ser algo que un producto escalar NO pueda escribir, y
    # elegirlo mal es fácil: un umbral sobre un rasgo de cada lado es un
    # producto exterior de dos indicadores, o sea rango 1, y con una dimensión
    # más el modelo lo representa exacto (la primera versión de esta sección
    # daba rango 7 y error cero en k = 7, que es lo contrario de lo que dice).
    # Lo que sí es de rango alto es una preferencia CON PICO: el lector tiene un
    # nivel favorito y le gusta lo que cae cerca, y "cerca" es un valor
    # absoluto, que no es bilineal en nada.
    cross = -2.0 * np.abs(A[:, 0:1] - B[:, 1:2].T)
    cross = cross - cross.mean()
    truth = taste + cross
    s = np.linalg.svd(truth, compute_uv=False)
    total = float((truth ** 2).sum())
    rows = []
    for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        keep = float((s[:k] ** 2).sum())
        # la cola de una suma de cuadrados puede salir negativa por redondeo
        # cuando ya no queda nada, y una raiz de eso es un NaN que viaja al
        # fichero y deja el widget sin curva
        rows.append({"k": k, "rmse": float(np.sqrt(max(total - keep, 0.0) / truth.size)),
                     "explained": min(keep / total, 1.0)})
    # y el mismo mundo sin el término de cruce, donde el techo es exacto en k=d
    s_pure = np.linalg.svd(taste, compute_uv=False)
    pure = [{"k": k, "rmse": float(np.sqrt(max(float((s_pure ** 2).sum())
                                               - float((s_pure[:k] ** 2).sum()), 0.0) / taste.size))}
            for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]]
    return {"rows": rows, "pure": pure, "n_users": n_u, "n_items": n_i, "d": d,
            "rank_needed": int(np.linalg.matrix_rank(truth)),
            "cross_share": float((cross ** 2).sum() / total),
            "singular": [float(v) for v in s[:32]]}


# --------------------------------------------------------------------------
# S2. Los pares de rasgos que nunca aparecen juntos, que es lo que la
# factorización de la interacción resuelve.
# --------------------------------------------------------------------------
def stage_pairs(F, train, users_sample):
    """Cuántos de los pares (rasgo de libro, rasgo de libro) se ven juntos."""
    B = sp.csr_matrix((np.ones_like(train.data), train.indices, train.indptr),
                      shape=train.shape)[users_sample]
    UF = (B @ F) > 0                       # rasgos que cada lector ha tocado
    UF = sp.csr_matrix(UF, dtype=np.float32)
    co = (UF.T @ UF).tocoo()
    d = F.shape[1]
    seen = int((co.data > 0).sum())
    total = d * (d + 1) // 2
    return {"features": int(d), "pairs": int(total), "seen": int(seen // 2),
            "share": float(seen / 2 / total),
            "params_full": int(total), "params_fm": int(d * K),
            "ratio": float(total / (d * K))}


# --------------------------------------------------------------------------
# S3. Los brazos sobre las valoraciones de verdad, con arranque en frío.
# --------------------------------------------------------------------------
def stage_arms(train, test, F, shape, users):
    tu, tb, tr = test
    n_u, n_i = train.shape
    rng = np.random.default_rng(SEED + 4)

    # los libros fríos: se eligen entre los que tienen bastantes valoraciones
    # retenidas para poder puntuarlos, y se borran ENTEROS del entrenamiento
    keep = np.isin(tu, users) & (tr >= 4)
    held_u, held_b = tu[keep], tb[keep]
    per_book_held = np.bincount(held_b, minlength=n_i)
    candidates = np.flatnonzero((per_book_held >= 3) & (shape["per_book"] >= 50))
    cold = np.sort(rng.choice(candidates, size=min(COLD_BOOKS, candidates.size), replace=False))
    cold_mask = np.zeros(n_i, dtype=bool)
    cold_mask[cold] = True

    warm_train = train.copy().tocoo()
    keep_cells = ~cold_mask[warm_train.col]
    train_cold = sp.csr_matrix((warm_train.data[keep_cells],
                                (warm_train.row[keep_cells], warm_train.col[keep_cells])),
                               shape=train.shape)
    lost = int((~keep_cells).sum())

    # las interacciones de entrenamiento: lo que cada lector valoró con 4 o 5
    coo = train_cold.tocoo()
    pos = coo.data >= 4
    tr_u, tr_i = coo.row[pos], coo.col[pos]

    held = {}
    for u, b in zip(held_u, held_b):
        held.setdefault(int(u), []).append(int(b))
    eval_users = np.array(sorted(held))
    held_list = [np.array(held[u]) for u in eval_users]
    seen = train_cold[eval_users]
    pop = shape["per_book"].astype(np.float64)
    srt = np.sort(pop)
    cold_held = [np.array([b for b in held[u] if cold_mask[b]]) for u in eval_users]
    warm_held = [np.array([b for b in held[u] if not cold_mask[b]]) for u in eval_users]
    ok = np.array([len(h) > 0 for h in cold_held])

    arms = {}
    for name, kw in [("ids", dict(use_id=True, use_content=False)),
                     ("content", dict(use_id=False, use_content=True)),
                     ("both", dict(use_id=True, use_content=True, id_drop=0.5))]:
        model = Tower(n_u, n_i, F, K, seed=SEED, **kw).fit(tr_u, tr_i, n_i)
        Q = model.all_items()
        scores = model.U[eval_users] @ Q.T + model.bias
        stats, _ = CF.rank_eval(scores.astype(np.float32), seen, held_list, pop, srt)
        # y sobre los libros fríos solos, restringiendo el catálogo a ellos para
        # que la pregunta sea "de estos 300, ¿cuáles son suyos?"
        cold_scores = scores[np.ix_(ok, cold)]
        cold_seen = sp.csr_matrix((ok.sum(), cold.size))
        remap = {int(b): j for j, b in enumerate(cold)}
        cold_lists = [np.array([remap[int(b)] for b in h]) for h, o in zip(cold_held, ok) if o]
        cstats, _ = CF.rank_eval(cold_scores.astype(np.float32), cold_seen, cold_lists,
                                 pop[cold], np.sort(pop[cold]), ks=(1, 3, 5, 10, 20, 50))
        arms[name] = {"warm": stats, "cold": cstats}
        print(f"    {name}: caliente @10 {stats['hits']['10']:.4f} "
              f"frío @10 {cstats['hits']['10']:.4f}")

    # el control del brazo frío: la popularidad dentro de los 300, que es lo que
    # queda cuando no hay ni identificador ni contenido
    pop_scores = np.tile(pop[cold].astype(np.float32), (int(ok.sum()), 1))
    remap = {int(b): j for j, b in enumerate(cold)}
    cold_lists = [np.array([remap[int(b)] for b in h]) for h, o in zip(cold_held, ok) if o]
    pstats, _ = CF.rank_eval(pop_scores, sp.csr_matrix((int(ok.sum()), cold.size)),
                             cold_lists, pop[cold], np.sort(pop[cold]))
    arms["popular"] = {"cold": pstats}

    return {"arms": arms, "cold_books": int(cold.size), "cold_cells": lost,
            "_eval_users": eval_users, "_cold_lists": cold_lists, "_ok": ok,
            "eval_users": int(eval_users.size),
            "cold_users": int(ok.sum()),
            "cold_items_held": int(sum(len(h) for h in cold_held)),
            "warm_items_held": int(sum(len(h) for h in warm_held)),
            "chance_at_10": float(10.0 / cold.size)}, cold


# --------------------------------------------------------------------------
# S3b. El mismo arranque en frío sin entrenar nada.
#
# Las tres torres comparten presupuesto y por eso son comparables entre ellas,
# pero su nivel absoluto depende del presupuesto, y una pregunta que solo se
# puede contestar con un modelo bien entrenado no es una buena pregunta. Esta
# regla no tiene parámetros: el perfil de un lector es la suma de los rasgos de
# lo que ha leído, ponderada por rareza, y un libro se puntúa por el coseno
# entre su vector de rasgos y ese perfil. Si los rasgos dicen algo sobre lo que
# alguien va a leer, esto lo dice sin entrenamiento y sin tasa de aprendizaje.
# --------------------------------------------------------------------------
def cold_eval_sets(test, users, cold, shape):
    """Los mismos conjuntos de evaluación en frío que arma `stage_arms`.

    Se calcula aparte y de forma determinista para que el brazo sin entrenar no
    dependa de lo que la etapa cara dejó en su caché, que es justo lo que hace
    que añadir una medida nueva no obligue a repetir tres entrenamientos."""
    tu, tb, tr = test
    keep = np.isin(tu, users) & (tr >= 4)
    cold_mask = np.zeros(shape["per_book"].size, dtype=bool)
    cold_mask[cold] = True
    held = {}
    for u, b in zip(tu[keep], tb[keep]):
        held.setdefault(int(u), []).append(int(b))
    eval_users = np.array(sorted(held))
    cold_held = [np.array([b for b in held[u] if cold_mask[b]]) for u in eval_users]
    ok = np.array([len(h) > 0 for h in cold_held])
    remap = {int(b): j for j, b in enumerate(cold)}
    cold_lists = [np.array([remap[int(b)] for b in h]) for h, o in zip(cold_held, ok) if o]
    return eval_users, cold_lists, ok


def stage_content_knn(train, F, shape, eval_users, cold, cold_lists, ok):
    df = np.asarray((F > 0).sum(0)).ravel()
    idf = np.log(F.shape[0] / np.maximum(df, 1)).astype(np.float32)
    Fw = F @ sp.diags(idf)
    norms = np.sqrt(np.asarray(Fw.multiply(Fw).sum(1)).ravel())
    Fn = sp.diags(1.0 / np.maximum(norms, 1e-9)) @ Fw

    B = sp.csr_matrix((np.ones_like(train.data), train.indices, train.indptr),
                      shape=train.shape)
    liked = sp.csr_matrix((np.where(train.data >= 4, 1.0, 0.0), train.indices, train.indptr),
                          shape=train.shape)
    liked.eliminate_zeros()
    prof = np.asarray((liked[eval_users][ok] @ Fn).todense(), dtype=np.float32)
    pn = np.sqrt((prof ** 2).sum(1, keepdims=True))
    prof /= np.maximum(pn, 1e-9)
    scores = prof @ np.asarray(Fn[cold].todense(), dtype=np.float32).T
    pop = shape["per_book"].astype(np.float64)
    stats, _ = CF.rank_eval(scores, sp.csr_matrix(scores.shape), cold_lists,
                            pop[cold], np.sort(pop[cold]))
    del B
    return {"cold": stats, "features": int(F.shape[1]),
            "profile_nonzero": float((prof != 0).sum(1).mean())}


# --------------------------------------------------------------------------
# S4. El coste, contado en multiplicaciones y sin relojes.
# --------------------------------------------------------------------------
def stage_cost(n_items, k, hidden=64, feats=8):
    """Puntuar un catálogo entero con las dos formas.

    Dos torres: una pasada por el lector y un producto escalar por artículo, y
    los vectores de artículo se calculan una vez al día. Una red que ve el par:
    una pasada por artículo, y no hay nada que precalcular porque la entrada
    depende de los dos."""
    rows = []
    for n in [100, 1000, 10000, 100000, 1000000]:
        tower = k * n                     # el producto escalar por artículo
        tower_user = k * hidden + hidden * k
        cross = n * (feats * hidden + hidden * hidden + hidden)
        rows.append({"n": n, "tower": int(tower + tower_user), "cross": int(cross),
                     "ratio": float(cross / (tower + tower_user))})
    return {"rows": rows, "k": k, "hidden": hidden, "feats": feats,
            "catalogue": int(n_items)}


# --------------------------------------------------------------------------
# S5. Lo que viaja al navegador: una torre pequeña que la página ejecuta.
# --------------------------------------------------------------------------
def stage_widget(train, F, shape, titles, authors, cold):
    order = np.argsort(-shape["per_book"])[:WIDGET_BOOKS]
    n_u, n_i = train.shape
    coo = train.tocoo()
    pos = coo.data >= 4
    model = Tower(n_u, n_i, F, K, use_id=True, use_content=True, id_drop=0.5,
                  seed=SEED + 2).fit(coo.row[pos], coo.col[pos], n_i)
    Q = model.all_items()[order]
    pop = shape["per_book"].astype(np.float64)
    items = [{"id": int(i), "title": short_title(titles[i], 38),
              "author": authors[i].split(",")[0],
              "rank": int((pop > pop[i]).sum()) + 1,
              "cold": bool(i in set(cold.tolist()))}
             for i in order]
    # el guardia: la puntuación de veinte perfiles, calculada aquí
    rng = np.random.default_rng(SEED + 9)
    checks = []
    for _ in range(12):
        picked = rng.choice(len(order), size=3, replace=False)
        vec = Q[picked].mean(0)
        sc = Q @ vec + model.bias[order]
        checks.append({"picked": [int(p) for p in picked],
                       "top": [int(t) for t in np.argsort(-sc)[:5]],
                       "best": float(sc.max())})
    return {"items": items, "vectors": [[float(v) for v in row] for row in Q],
            "bias": [float(v) for v in model.bias[order]], "check": checks}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("la partición de los dos artículos anteriores")
    train, test, fp, n_u, n_b = CF.cached("split", CF.stage_split)
    prev = json.loads((ROOT / "collaborative-filtering" / "data" / "cf.json").read_text())
    assert prev["meta"]["split_fingerprint"] == fp, "la partición no es la misma"
    shape = CF.cached("shape", lambda: CF.stage_shape(train, n_u, n_b))
    titles, authors, years = books()
    print(f"  huella {fp} confirmada")

    F, fmeta, vocab, avocab = cached("features", lambda: item_features(titles, authors, years, n_b))
    print(f"  {fmeta['features']} rasgos, {fmeta['per_book']:.1f} por libro")

    print("el techo exacto de un producto escalar")
    ceiling = cached("ceiling", stage_ceiling)

    rng = np.random.default_rng(SEED + 1)
    users = np.sort(rng.choice(n_u, size=CF.N_EVAL_USERS, replace=False))

    print("los pares de rasgos")
    pairs = cached("pairs", lambda: stage_pairs(F, train, users[:1000]))
    print(f"  {pairs['share']:.4%} de los pares se ven juntos alguna vez")

    print("los tres brazos, con 300 libros borrados")
    arms, cold = cached("arms", lambda: stage_arms(train, test, F, shape, users))

    print("y la misma pregunta sin entrenar nada")
    ev_u, cold_lists, ok = cold_eval_sets(test, users, cold, shape)
    cknn = cached("content_knn", lambda: stage_content_knn(
        train, F, shape, ev_u, cold, cold_lists, ok))
    print(f"  contenido sin entrenar, frío @10 {cknn['cold']['hits']['10']:.4f}")

    cost = stage_cost(n_b, K)
    widget = cached("widget", lambda: stage_widget(train, F, shape, titles, authors, cold))

    data = {
        "meta": {"seed": SEED, "k": K, "epochs": EPOCHS, "neg": NEG, "batch": BATCH,
                 "split_fingerprint": fp, "digest_ratings": digest("goodbooks_ratings.csv"),
                 "users": n_u, "books": n_b, "features": fmeta},
        "previous": {
            "item_knn_hits10": prev["rank"]["item_knn"]["hits"]["10"],
            "popular_hits10": prev["rank"]["popular"]["hits"]["10"],
            "eval_users": prev["rank_meta"]["eval_users"],
        },
        "ceiling": r(ceiling, 6),
        "pairs": r(pairs, 6),
        "arms": r({k: v for k, v in arms.items() if not k.startswith("_")}, 5),
        "content_knn": r(cknn, 5),
        "cost": r(cost, 4),
        "widget": {"items": widget["items"], "vectors": r(widget["vectors"], 5),
                   "bias": r(widget["bias"], 5), "check": r(widget["check"], 5)},
    }
    path = OUT / "tower.json"
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    print(f"escrito {path} ({path.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
