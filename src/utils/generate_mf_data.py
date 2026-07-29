"""Las cifras del artículo de factorización de matrices (módulo 6.1).

El artículo 25 (`directions/`) enseñó que PCA, ICA y NMF son **la misma
factorización** con tres restricciones distintas. Este añade la cuarta, que es
de otra especie: casi ninguna entrada de la matriz existe. Y esa diferencia se
mide en vez de contarse, sobre los mismos 2.000 dígitos de aquel artículo:
misma matriz, mismos rangos, y el único cambio es tapar el 90% de los píxeles.

Lo que se mide:

  1. **La identidad con el artículo 25**: el SVD truncado de la matriz centrada
     reproduce el error de reconstrucción que aquella página publicó para PCA,
     leyéndolo de su propio JSON. Sin eso, "es la misma factorización" es una
     intención.
  2. **El agujero cambia el problema**: sobre esos mismos dígitos con el 90% de
     los píxeles tapados, rellenar con cero, rellenar con la media y ajustar
     solo sobre lo observado, puntuados contra los píxeles tapados.
  3. **Sobre las valoraciones de verdad**, con la partición del artículo
     anterior afirmada por huella: sesgos, SVD sobre la matriz rellena de
     ceros, SVD sobre los residuos, ALS sobre lo observado, y descenso por
     gradiente estocástico sobre el mismo objetivo.
  4. **Rango y regularización**, barridos hasta donde la curva se da la vuelta.
  5. **La invariancia por rotación**, que es exacta y por lo tanto un guardia:
     cualquier transformación invertible de los factores deja las predicciones
     idénticas, así que "el factor 3 es la fantasía juvenil" no es una
     propiedad del modelo.
  6. **ALS implícito** sobre la misma partición, puntuado con el mismo protocolo
     de ranking del artículo anterior, que es la comparación que aquel artículo
     dejó pendiente.

Ejecutar desde cualquier sitio: `python src/utils/generate_mf_data.py`.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recsys_data import books, digest, short_title  # noqa: E402
from vision_data import mnist, subset  # noqa: E402
import generate_cf_data as CF  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "matrix-factorization" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "mf_cache")

SEED = 19
K = 32                     # factores por defecto
LAMBDA = 0.08              # regularización por defecto
EPOCHS = 10
RANKS = [1, 2, 4, 8, 16, 32, 64]
LAMBDAS = [0.02, 0.05, 0.08, 0.15, 0.3, 0.6]
DIGIT_RANKS = [2, 4, 8, 16, 25]
HIDE = 0.9                 # fracción de píxeles tapados en el experimento de dígitos
WIDGET_BOOKS = 300
TAG = f"s{SEED}k{K}e{EPOCHS}"


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
rmse = CF.rmse


# --------------------------------------------------------------------------
# ALS sobre las celdas observadas.
#
# El bucle por usuario es la escritura obvia y cuesta minutos por época. Lo que
# se hace aquí es formar las 528 entradas del triángulo superior de cada matriz
# de Gram con `add.reduceat` sobre las entradas ya ordenadas, en trozos de 16
# columnas para no materializar (5,7 millones x 528).
# --------------------------------------------------------------------------
def gram_and_rhs(F, idx_other, indptr, resid, k, chunk=None):
    """Para cada fila: sum(q q^T) sobre sus entradas, y sum(resid q).

    Escrito como un bucle con una multiplicacion de matrices pequena por fila,
    y no como una version "vectorizada" sobre las 5,7 millones de entradas a la
    vez. La segunda es la que uno escribe primero y es diez veces mas lenta:
    formar las 528 columnas del triangulo superior obliga a recorrer un array
    de 5,7 millones por 16 unas treinta veces por lado, o sea gigabytes de
    trafico de memoria, mientras que el bucle deja el trabajo dentro de BLAS
    con matrices de 107 por 32 que caben en cache. Una epoca pasa de cien
    segundos a unos diez."""
    n = len(indptr) - 1
    A = np.zeros((n, k, k))
    rhs = np.zeros((n, k))
    for row in range(n):
        a, b = indptr[row], indptr[row + 1]
        if b <= a:
            continue
        Qe = F[idx_other[a:b]]
        A[row] = Qe.T @ Qe
        rhs[row] = Qe.T @ resid[a:b]
    return A, rhs


def als_fit(train, resid_data, k, lam, epochs, seed=SEED, log=None):
    """ALS sobre los residuos del modelo de sesgos, solo en lo observado."""
    n_u, n_i = train.shape
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.05, (n_u, k))
    Q = rng.normal(0, 0.05, (n_i, k))
    csr = sp.csr_matrix((resid_data, train.indices, train.indptr), shape=train.shape)
    csc = csr.tocsc()
    eye = np.eye(k)
    cnt_u = np.diff(csr.indptr).astype(np.float64)
    cnt_i = np.diff(csc.indptr).astype(np.float64)
    hist = []
    rows_coo = csr.tocoo().row
    for ep in range(epochs):
        A, b = gram_and_rhs(Q, csr.indices, csr.indptr, csr.data.astype(np.float64), k)
        P = np.linalg.solve(A + lam * cnt_u[:, None, None] * eye, b[:, :, None])[:, :, 0]
        A, b = gram_and_rhs(P, csc.indices, csc.indptr, csc.data.astype(np.float64), k)
        Q = np.linalg.solve(A + lam * cnt_i[:, None, None] * eye, b[:, :, None])[:, :, 0]
        if log is not None:
            fit = np.einsum('ij,ij->i', P[rows_coo], Q[csr.indices])
            hist.append(float(np.sqrt(np.mean((csr.data - fit) ** 2))))
    return P, Q, hist


def randomised_svd(M, k, oversample=12, power=4, seed=SEED):
    """El SVD truncado por proyeccion aleatoria, con iteraciones de potencia.

    `svds` de ARPACK sobre 53.424 x 10.000 con 5,7 millones de entradas tarda
    mas que todo lo demas de esta pagina junta, porque los valores singulares de
    una matriz de valoraciones decaen despacio y el metodo necesita miles de
    productos. El rango aleatorio con cuatro iteraciones de potencia da el mismo
    subespacio en segundos, y para lo que la pagina afirma (que rellenar de
    ceros contesta otra pregunta) la diferencia entre los dos es varios ordenes
    de magnitud mas pequena que el efecto."""
    rng = np.random.default_rng(seed)
    Omega = rng.normal(size=(M.shape[1], k + oversample))
    Y = np.asarray(M @ Omega)
    for _ in range(power):
        Q, _ = np.linalg.qr(Y)
        Y = np.asarray(M @ np.asarray(M.T @ Q))
    Q, _ = np.linalg.qr(Y)
    B = np.asarray(Q.T @ M)
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


def predict_cells(P, Q, base, tu, tb):
    return base + np.einsum('ij,ij->i', P[tu], Q[tb])


# --------------------------------------------------------------------------
# S1. La identidad con el artículo 25, y lo que cambia al tapar entradas.
# --------------------------------------------------------------------------
def stage_digits():
    Xtr, ytr, _, _ = mnist()
    Xd, _ = subset(Xtr, ytr, 2000, seed=SEED)
    DIG = Xd.reshape(len(Xd), -1) / 255.0
    published = json.loads((ROOT / "directions" / "data" / "directions.json").read_text())
    ref = {row["rank"]: row["pca_unquantised"] for row in published["atoms"]["errors"]}

    # (a) la matriz entera: SVD truncado de la matriz centrada = PCA
    mean = DIG.mean(0)
    Xc = DIG - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    complete = []
    for k in DIGIT_RANKS:
        rec = mean + U[:, :k] * s[:k] @ Vt[:k]
        complete.append({"rank": k, "svd": float(np.mean((rec - DIG) ** 2)),
                         "published_pca": ref[k]})

    # (b) la misma matriz con el 90% de los píxeles tapados
    rng = np.random.default_rng(SEED + 3)
    mask = rng.random(DIG.shape) >= HIDE          # True = observado
    obs = mask.astype(np.float64)
    hidden = ~mask
    col_mean = (DIG * obs).sum(0) / np.maximum(obs.sum(0), 1)

    def score(rec):
        return float(np.sqrt(np.mean((rec[hidden] - DIG[hidden]) ** 2)))

    holed = []
    for k in DIGIT_RANKS:
        Z = DIG * obs                                     # ceros en lo tapado
        Uz, sz, Vtz = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        rec_zero = Z.mean(0) + Uz[:, :k] * sz[:k] @ Vtz[:k]

        M = np.where(mask, DIG, col_mean)                  # media de la columna
        Um, sm, Vtm = np.linalg.svd(M - M.mean(0), full_matrices=False)
        rec_mean = M.mean(0) + Um[:, :k] * sm[:k] @ Vtm[:k]

        # y el ajuste que solo mira lo observado, por mínimos cuadrados alternos
        F = M.copy()
        rec_em = None
        for _ in range(60):
            Ue, se, Vte = np.linalg.svd(F - F.mean(0), full_matrices=False)
            rec_em = F.mean(0) + Ue[:, :k] * se[:k] @ Vte[:k]
            F = np.where(mask, DIG, rec_em)
        holed.append({"rank": k, "zero": score(rec_zero), "mean": score(rec_mean),
                      "observed": score(rec_em)})
    baseline = float(np.sqrt(np.mean((np.broadcast_to(col_mean, DIG.shape)[hidden]
                                      - DIG[hidden]) ** 2)))
    return {"complete": complete, "holed": holed, "hidden_baseline": baseline,
            "n": int(DIG.shape[0]), "pixels": int(DIG.shape[1]), "hide": HIDE,
            "hidden_cells": int(hidden.sum())}


# --------------------------------------------------------------------------
# S2. Las valoraciones. La partición es la del artículo anterior, afirmada.
# --------------------------------------------------------------------------
def stage_ratings(train, test, mu, bu, bi):
    tu, tb, tr = test
    base = mu + bu[tu] + bi[tb]
    resid = CF.centred_matrix(train, mu, bu, bi).data.astype(np.float64)
    out = {"bias": rmse(base, tr)}

    # SVD sobre la matriz cruda con los huecos como ceros, que es lo que hace
    # cualquier rutina de álgebra lineal si se le da la matriz dispersa
    Uz, sz, Vtz = cached("svd_zero", lambda: randomised_svd(train.astype(np.float64), K))
    pred_zero = np.einsum('ij,ij->i', (Uz * sz)[tu], Vtz.T[tb])
    out["svd_zero"] = rmse(pred_zero, tr)

    # y sobre los residuos, que es la misma operación con el modelo de sesgos
    # ya restado: sigue tratando lo que falta como un cero, pero ahora el cero
    # significa "lo que el sesgo ya predice"
    Rc = CF.centred_matrix(train, mu, bu, bi).astype(np.float64)
    Ur, sr, Vtr = cached("svd_resid", lambda: randomised_svd(Rc, K))
    out["svd_resid"] = rmse(base + np.einsum('ij,ij->i', (Ur * sr)[tu], Vtr.T[tb]), tr)

    t0 = time.time()
    P, Q, hist = cached("als", lambda: als_fit(train, resid, K, LAMBDA, EPOCHS, log=True))
    als_seconds = time.time() - t0
    out["als"] = rmse(predict_cells(P, Q, base, tu, tb), tr)

    # el mismo objetivo por otra búsqueda
    t0 = time.time()
    Ps, Qs, sgd_hist = cached("sgd", lambda: sgd_fit(train, resid, K, LAMBDA, epochs=EPOCHS))
    sgd_seconds = time.time() - t0
    out["sgd"] = rmse(predict_cells(Ps, Qs, base, tu, tb), tr)

    return {"rmse": out, "als_train_curve": hist, "sgd_train_curve": sgd_hist,
            "seconds_ratio": sgd_seconds / max(als_seconds, 1e-9),
            "params": int(P.size + Q.size),
            "agree": float(np.corrcoef(predict_cells(P, Q, base, tu, tb),
                                       predict_cells(Ps, Qs, base, tu, tb))[0, 1])}, P, Q


def sgd_fit(train, resid_data, k, lam, epochs, lr=0.01, seed=SEED):
    """Funk SGD sobre el mismo objetivo, en pasadas barajadas.

    Escrito con numpy sobre lotes pequeños en vez de celda a celda: celda a
    celda en python son 5,7 millones de iteraciones por época y la comparación
    entre las dos búsquedas mediría el intérprete."""
    coo = train.tocoo()
    n_u, n_i = train.shape
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.05, (n_u, k))
    Q = rng.normal(0, 0.05, (n_i, k))
    rows, cols, vals = coo.row, coo.col, resid_data
    hist = []
    batch = 8192
    for ep in range(epochs):
        order = rng.permutation(vals.size)
        for a in range(0, vals.size, batch):
            sel = order[a:a + batch]
            u, i = rows[sel], cols[sel]
            pu, qi = P[u], Q[i]
            err = vals[sel] - np.einsum('ij,ij->i', pu, qi)
            gp = err[:, None] * qi - lam * pu
            gq = err[:, None] * pu - lam * qi
            # `np.add.at` es la escritura obvia y cuesta el articulo entero:
            # sobre lotes de 8.192 por 32 tarda unos cien milisegundos, o sea
            # veinte minutos por epoca. Un `bincount` por columna hace la misma
            # dispersion acumulada, duplicados incluidos, en microsegundos.
            for c in range(k):
                P[:, c] += lr * np.bincount(u, weights=gp[:, c], minlength=n_u)
                Q[:, c] += lr * np.bincount(i, weights=gq[:, c], minlength=n_i)
        fit = np.einsum('ij,ij->i', P[rows], Q[cols])
        hist.append(float(np.sqrt(np.mean((vals - fit) ** 2))))
    return P, Q, hist


# --------------------------------------------------------------------------
# S3. Rango y regularización.
# --------------------------------------------------------------------------
SWEEP_EPOCHS = 6           # los barridos son forma de curva, no cifra de titular


def stage_sweep_rank(train, test, mu, bu, bi):
    tu, tb, tr = test
    base = mu + bu[tu] + bi[tb]
    resid = CF.centred_matrix(train, mu, bu, bi).data.astype(np.float64)
    coo = train.tocoo()

    rank_rows = []
    for k in RANKS:
        P, Q, _ = als_fit(train, resid, k, LAMBDA, SWEEP_EPOCHS)
        fit = np.einsum('ij,ij->i', P[coo.row], Q[coo.col])
        rank_rows.append({
            "k": k, "test": rmse(predict_cells(P, Q, base, tu, tb), tr),
            "train": float(np.sqrt(np.mean((resid - fit) ** 2))),
            "params": int(P.size + Q.size)})
        print(f"    rango {k}: test {rank_rows[-1]['test']:.4f} "
              f"train {rank_rows[-1]['train']:.4f}")

    return rank_rows


def stage_sweep_lam(train, test, mu, bu, bi):
    """El barrido de regularizacion, que no puede empezar en cero.

    Con K factores, un lector con menos de K valoraciones da una matriz de Gram
    de rango deficiente, y sin penalizacion el sistema es literalmente singular:
    `np.linalg.solve` lanza. Eso no es un problema de implementacion que haya
    que esquivar con un pinv, es la razon por la que el termino existe, y el
    numero de lectores a los que les pasa se exporta para poder decirlo."""
    tu, tb, tr = test
    base = mu + bu[tu] + bi[tb]
    resid = CF.centred_matrix(train, mu, bu, bi).data.astype(np.float64)
    coo = train.tocoo()
    lam_rows = []
    for lam in LAMBDAS:
        P, Q, _ = als_fit(train, resid, K, lam, SWEEP_EPOCHS)
        fit = np.einsum('ij,ij->i', P[coo.row], Q[coo.col])
        lam_rows.append({
            "lam": lam, "test": rmse(predict_cells(P, Q, base, tu, tb), tr),
            "train": float(np.sqrt(np.mean((resid - fit) ** 2))),
            "norm": float(np.sqrt((P ** 2).sum(1)).mean())})
        print(f"    lambda {lam}: test {lam_rows[-1]['test']:.4f}")
    counts_u = np.diff(train.tocsr().indptr)
    counts_i = np.diff(train.tocsc().indptr)
    return {"rows": lam_rows,
            "singular_users": int((counts_u < K).sum()),
            "singular_items": int((counts_i < K).sum()),
            "users": int(train.shape[0]), "items": int(train.shape[1])}


# --------------------------------------------------------------------------
# S4. La invariancia por rotación, que es exacta.
# --------------------------------------------------------------------------
def stage_rotation(P, Q, shape_counts):
    rng = np.random.default_rng(SEED + 5)
    k = P.shape[1]
    A = rng.normal(size=(k, k))
    Ainv = np.linalg.inv(A)
    users = rng.choice(P.shape[0], 400, replace=False)
    items = rng.choice(Q.shape[0], 400, replace=False)
    before = P[users] @ Q[items].T
    after = (P[users] @ A) @ (Q[items] @ Ainv.T).T
    worst = float(np.max(np.abs(before - after)))
    scale = float(np.max(np.abs(before)))

    # y lo que sí cambia: qué libros encabezan cada factor
    def tops(F, j, n=5):
        pop = shape_counts >= 300           # solo libros con lectores de sobra
        v = np.where(pop, F[:, j], -np.inf)
        return [int(i) for i in np.argsort(-v)[:n]]

    Qr = Q @ Ainv.T
    same = 0
    for j in range(k):
        a, b = set(tops(Q, j)), set(tops(Qr, j))
        same += len(a & b)
    return {"worst_abs": worst, "scale": scale,
            "relative": worst / scale,
            "top_overlap": same / (5 * k),
            "cond": float(np.linalg.cond(A))}


# --------------------------------------------------------------------------
# S5. ALS implícito, y el ranking con el protocolo del artículo anterior.
# --------------------------------------------------------------------------
def implicit_als(train, k, lam, alpha, epochs, seed=SEED):
    """Hu, Koren y Volinsky: todo el cero cuenta, con menos confianza.

    El truco es el que hace viable mirar las 534 millones de celdas: la suma
    sobre TODOS los items se hace una vez (la matriz de Gram Y'Y) y por usuario
    solo se corrige con los que tocó, que son ciento y pico.

        A_u = Y'Y + alpha * sum_{i observado} y_i y_i' + lambda I
        b_u = (1 + alpha) * sum_{i observado} y_i
    """
    n_u, n_i = train.shape
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.05, (n_u, k))
    Q = rng.normal(0, 0.05, (n_i, k))
    B = sp.csr_matrix((np.ones_like(train.data), train.indices, train.indptr),
                      shape=train.shape)
    Bc = B.tocsc()
    eye = np.eye(k)
    ones_u = np.ones(B.indices.size)
    ones_i = np.ones(Bc.indices.size)
    for _ in range(epochs):
        G = Q.T @ Q
        A, rhs = gram_and_rhs(Q, B.indices, B.indptr, ones_u, k)
        P = np.linalg.solve(G[None] + alpha * A + lam * eye,
                            ((1.0 + alpha) * rhs)[:, :, None])[:, :, 0]
        G = P.T @ P
        A, rhs = gram_and_rhs(P, Bc.indices, Bc.indptr, ones_i, k)
        Q = np.linalg.solve(G[None] + alpha * A + lam * eye,
                            ((1.0 + alpha) * rhs)[:, :, None])[:, :, 0]
    return P, Q


def stage_implicit(train, test, shape, users):
    tu, tb, tr = test
    P, Q = implicit_als(train, K, 0.1, 40.0, 8)
    keep = np.isin(tu, users) & (tr >= 4)
    held = {}
    for u, b in zip(tu[keep], tb[keep]):
        held.setdefault(int(u), []).append(int(b))
    eval_users = np.array(sorted(held))
    held_list = [np.array(held[u]) for u in eval_users]
    seen = train[eval_users]
    pop = shape["per_book"].astype(np.float64)
    srt = np.sort(pop)
    scores = (P[eval_users] @ Q.T).astype(np.float32)
    stats, tops = CF.rank_eval(scores, seen, held_list, pop, srt)
    return {"rank": stats, "eval_users": int(eval_users.size),
            "held_items": int(sum(len(h) for h in held_list))}, P, Q


# --------------------------------------------------------------------------
# S6. Lo que viaja al navegador.
# --------------------------------------------------------------------------
def stage_widget(P, Q, Qi, shape, titles, authors, mu, bu, bi):
    order = np.argsort(-shape["per_book"])[:WIDGET_BOOKS]
    pop = shape["per_book"].astype(np.float64)
    srt = np.sort(pop)
    items = [{
        "id": int(i), "title": short_title(titles[i], 38),
        "author": authors[i].split(",")[0],
        "n": int(shape["per_book"][i]),
        "rank": int((pop > pop[i]).sum()) + 1,
        "bias": float(bi[i]),
    } for i in order]
    # ocho factores para el explorador, en el orden en que ALS los dejó
    Fexp = Q[order][:, :8]
    Fimp = Qi[order][:, :8]
    rng = np.random.default_rng(SEED + 11)
    pairs = [[int(a), int(b)] for a, b in rng.integers(0, len(order), size=(20, 2)) if a != b]
    check = [{"a": a, "b": b,
              "dot": float(Q[order[a]] @ Q[order[b]]),
              "cos": float(Q[order[a]] @ Q[order[b]]
                           / (np.linalg.norm(Q[order[a]]) * np.linalg.norm(Q[order[b]])))}
             for a, b in pairs]
    return {"items": items,
            "factors": [[float(v) for v in row] for row in Fexp],
            "implicit": [[float(v) for v in row] for row in Fimp],
            "full": [[float(v) for v in row] for row in Q[order]],
            "check": check}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("la partición del artículo anterior")
    train, test, fp, n_u, n_b = CF.cached("split", CF.stage_split)
    published = json.loads((ROOT / "collaborative-filtering" / "data" / "cf.json").read_text())
    assert published["meta"]["split_fingerprint"] == fp, (
        "la partición no es la del artículo anterior: "
        f"{fp} contra {published['meta']['split_fingerprint']}")
    print(f"  huella {fp} confirmada contra collaborative-filtering/data/cf.json")
    shape = CF.cached("shape", lambda: CF.stage_shape(train, n_u, n_b))
    titles, authors, _ = books()
    mu, bu, bi = CF.fit_bias(train)

    print("los dígitos del artículo 25, con y sin agujeros")
    digits = cached("digits", stage_digits)
    for row in digits["complete"]:
        print(f"    rango {row['rank']}: svd {row['svd']:.8f} "
              f"publicado {row['published_pca']:.8f}")

    print("las valoraciones")
    ratings_out, P, Q = cached("ratings", lambda: stage_ratings(train, test, mu, bu, bi))
    print(f"  {ratings_out['rmse']}")

    print("rango y regularización")
    sweep_rank = cached("sweep_rank", lambda: stage_sweep_rank(train, test, mu, bu, bi))
    sweep_lam = cached("sweep_lam", lambda: stage_sweep_lam(train, test, mu, bu, bi))
    sweeps = {"rank": sweep_rank, "lam": sweep_lam["rows"],
              "singular": {k: v for k, v in sweep_lam.items() if k != "rows"}}

    print("la rotación")
    rot = cached("rotation", lambda: stage_rotation(P, Q, shape["per_book"]))
    print(f"  la predicción se mueve {rot['worst_abs']:.3e} y los factores cambian de significado")

    print("ALS implícito y el ranking")
    rng = np.random.default_rng(SEED + 1)
    users = np.sort(rng.choice(n_u, size=CF.N_EVAL_USERS, replace=False))
    imp, Pi, Qi = cached("implicit", lambda: stage_implicit(train, test, shape, users))
    print(f"  implicit @10 {imp['rank']['hits']['10']:.4f} contra "
          f"{published['rank']['item_knn']['hits']['10']:.4f} de los vecinos")

    widget = cached("widget", lambda: stage_widget(P, Q, Qi, shape, titles, authors, mu, bu, bi))

    data = {
        "meta": {"seed": SEED, "k": K, "lam": LAMBDA, "epochs": EPOCHS,
                 "sweep_epochs": SWEEP_EPOCHS,
                 "split_fingerprint": fp, "digest_ratings": digest("goodbooks_ratings.csv"),
                 "users": n_u, "books": n_b, "test_cells": int(test[0].size),
                 "train_ratings": int(train.nnz),
                 "hide": HIDE, "digit_ranks": DIGIT_RANKS},
        "previous": {
            "item_knn_rmse": published["rating"]["rmse"]["item_knn"],
            "bias_rmse": published["rating"]["rmse"]["bias"],
            "item_knn_hits10": published["rank"]["item_knn"]["hits"]["10"],
            "popular_hits10": published["rank"]["popular"]["hits"]["10"],
            "item_knn_coverage": published["rank"]["item_knn"]["coverage"],
            "eval_users": published["rank_meta"]["eval_users"],
            "held_items": published["rank_meta"]["held_items"],
        },
        "digits": r(digits, 8),
        "ratings": r({k: v for k, v in ratings_out.items()}, 4),
        "sweeps": r(sweeps, 4),
        "rotation": r(rot, 10),
        "implicit": r(imp, 4),
        "widget": {"items": r(widget["items"], 4),
                   "factors": r(widget["factors"], 5),
                   "implicit": r(widget["implicit"], 5),
                   "full": r(widget["full"], 5),
                   "check": r(widget["check"], 8)},
    }
    path = OUT / "mf.json"
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    print(f"escrito {path} ({path.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
