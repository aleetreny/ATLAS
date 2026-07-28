"""Las cifras del articulo de TF-IDF y n-gramas.

Corpus: las 7.790 noticias de Reuters-21578 que llevan exactamente una
categoria y caen en las ocho mayores (el subconjunto que la literatura llama
R8), con la particion ModApte que trae el propio corpus. Se elige por tres
razones y las tres se usan: tiene etiquetas, tiene documentos de 2 a 963
palabras (y la normalizacion por longitud solo significa algo cuando la
longitud varia de verdad), y es un corpus de recuperacion, asi que la busqueda
por coseno de la pagina es la tarea para la que se construyo.

La pieza central no es el ranking de pesos, es el **factorial completo**: la
formula de TF-IDF son tres decisiones independientes (que se hace con la
frecuencia, si se pesa por rareza, si se normaliza la longitud) y aqui se miden
las doce combinaciones sobre la misma particion y con dos clasificadores
distintos, en vez de citar la formula entera como si fuera una sola cosa. Lo
que sale de ahi decide como se escribe el articulo.

Todo lo que la pagina dibuja sale de conteos enteros, asi que el navegador
rehace la ponderacion desde los mismos enteros y el guardia puede exigir
igualdad a 1e-12 en vez de "coincide dentro de lo razonable".

Ejecutar desde cualquier sitio: `python src/utils/generate_tfidf_data.py`.
"""
import base64
import json
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlp_data import reuters, stopwords, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tf-idf" / "data"
CACHE = Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data")) / "tfidf_cache"

SEED = 19
# Ocho categorias, las mayores de las que solo tienen una etiqueta. El reparto
# es brutalmente desigual (3.923 contra 144) y por eso el marcador lleva macro
# F1 al lado de la exactitud: una exactitud de 0,80 la saca contestar "earn" y
# "acq" a todo.
# Dos celdas del factorial (las que llevan idf y no normalizan) agotaban 20.000
# iteraciones del dual de liblinear, y son justo las dos que sostienen la
# afirmacion de que idf resta cuando no se normaliza: una celda que no converge
# mide el presupuesto de iteraciones y no su representacion. A 200.000 las doce
# convergen y las dos se mueven 0,0009 y 0,0000 en macro F1, asi que la
# conclusion aguanta, pero ahora esta medida en vez de supuesta.
MAX_ITER = 200000

R8 = ["earn", "acq", "crude", "trade", "money-fx", "interest", "money-supply", "ship"]


def cached(name, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.is_file():
        with path.open("rb") as fh:
            return pickle.load(fh)
    val = fn()
    # A un temporal y luego rename: si algo falla a mitad de `pickle.dump`, el
    # fichero definitivo no llega a existir. Sin esto, un fallo posterior deja
    # un pickle de cero bytes y la tirada siguiente muere al LEERLO, que es un
    # error mucho mas confuso que el original.
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(val, fh)
    tmp.replace(path)
    return val


def b64_u16(a):
    return base64.b64encode(np.asarray(a, dtype="<u2").tobytes()).decode()


def b64_u32(a):
    return base64.b64encode(np.asarray(a, dtype="<u4").tobytes()).decode()


def b64_u8(a):
    return base64.b64encode(np.asarray(a, dtype=np.uint8).tobytes()).decode()


# --------------------------------------------------------------------- corpus
def load_corpus():
    docs, cats, split = reuters()
    keep = [(d, c[0], s) for d, c, s in zip(docs, cats, split)
            if len(c) == 1 and c[0] in R8]
    texts = [d for d, _, _ in keep]
    labels = [c for _, c, _ in keep]
    parts = [s for _, _, s in keep]
    toks = [tokenize(t) for t in texts]
    return texts, labels, parts, toks


# ---------------------------------------------------------------------- zipf
def zipf_stats(toks):
    counts = Counter()
    for t in toks:
        counts.update(t)
    total = sum(counts.values())
    ordered = counts.most_common()
    freqs = np.array([c for _, c in ordered], dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)

    # La pendiente se ajusta en una banda de rangos declarada, no sobre la curva
    # entera: la cola de los que aparecen una vez es una escalera plana de miles
    # de puntos y arrastra cualquier ajuste hacia ella. La banda es 10 a 1.000,
    # que es donde Zipf es una afirmacion y no un artefacto del truncamiento.
    lo, hi = 10, 1000
    m = (ranks >= lo) & (ranks <= hi)
    slope, intercept = np.polyfit(np.log10(ranks[m]), np.log10(freqs[m]), 1)
    pred = 10 ** (intercept + slope * np.log10(ranks[m]))
    ss_res = float(np.sum((np.log10(freqs[m]) - np.log10(pred)) ** 2))
    ss_tot = float(np.sum((np.log10(freqs[m]) - np.log10(freqs[m]).mean()) ** 2))

    hapax = int(sum(1 for _, c in ordered if c == 1))
    cum = np.cumsum(freqs)
    shares = {k: float(cum[min(k, len(cum)) - 1] / total) for k in (10, 100, 1000)}
    # Cuantos tipos distintos hacen falta para cubrir la mitad de los tokens.
    half = int(np.searchsorted(cum, total / 2) + 1)
    return {
        "tokens": int(total),
        "types": len(ordered),
        "hapax": hapax,
        "hapax_frac": hapax / len(ordered),
        "slope": float(slope),
        "intercept": float(intercept),
        "fit_lo": lo, "fit_hi": hi,
        "fit_r2": float(1 - ss_res / ss_tot),
        "share_10": shares[10], "share_100": shares[100], "share_1000": shares[1000],
        "types_for_half": half,
        "top": [{"w": w, "c": int(c)} for w, c in ordered[:40]],
        # La curva viaja diezmada en escala logaritmica: 240 puntos dibujan la
        # misma recta que 23.415 y el JSON no engorda por un grafico.
        "curve": zipf_curve(ranks, freqs),
    }


def zipf_curve(ranks, freqs, n=240):
    idx = np.unique(np.round(np.logspace(0, math.log10(len(ranks)), n)).astype(int) - 1)
    idx = idx[(idx >= 0) & (idx < len(ranks))]
    return [[int(ranks[i]), int(freqs[i])] for i in idx]


def heaps(toks):
    """Tipos contra tokens segun se van acumulando documentos, y el exponente.

    El orden de acumulacion es el del corpus, no uno aleatorio: es el orden en
    que a un sistema real le llegan los documentos, y es reproducible sin
    barajar nada.
    """
    seen = set()
    pts = []
    total = 0
    for t in toks:
        total += len(t)
        seen.update(t)
        pts.append((total, len(seen)))
    take = np.unique(np.round(np.logspace(0, math.log10(len(pts)), 200)).astype(int) - 1)
    take = take[(take >= 0) & (take < len(pts))]
    curve = [[int(pts[i][0]), int(pts[i][1])] for i in take]
    xs = np.log10([p[0] for p in pts[20:]])
    ys = np.log10([p[1] for p in pts[20:]])
    beta, logk = np.polyfit(xs, ys, 1)
    return {"curve": curve, "beta": float(beta), "k": float(10 ** logk),
            "final_tokens": int(pts[-1][0]), "final_types": int(pts[-1][1])}


# ------------------------------------------------------------------- n-gramas
def ngram_inflation(toks, parts):
    """Lo que cuesta subir de n, y lo que de eso es utilizable.

    Tres cifras por orden, y la tercera es la que casi nunca se publica: que
    fraccion de los n-gramas del test no aparecio nunca en train. Un rasgo que
    el modelo no vio en entrenamiento no es un rasgo, es una columna de ceros.
    """
    out = []
    tr = [t for t, p in zip(toks, parts) if p == "train"]
    te = [t for t, p in zip(toks, parts) if p == "test"]
    for n in (1, 2, 3, 4):
        cnt = Counter()
        for t in toks:
            for i in range(len(t) - n + 1):
                cnt[tuple(t[i:i + n])] += 1
        train_set = set()
        for t in tr:
            for i in range(len(t) - n + 1):
                train_set.add(tuple(t[i:i + n]))
        te_tok = 0
        te_unseen = 0
        te_types = set()
        for t in te:
            for i in range(len(t) - n + 1):
                g = tuple(t[i:i + n])
                te_tok += 1
                te_types.add(g)
                if g not in train_set:
                    te_unseen += 1
        hap = sum(1 for c in cnt.values() if c == 1)
        out.append({
            "n": n,
            "types": len(cnt),
            "tokens": int(sum(cnt.values())),
            "hapax": hap,
            "hapax_frac": hap / len(cnt),
            "train_types": len(train_set),
            "test_unseen_token_frac": te_unseen / te_tok,
            "test_unseen_type_frac": len(te_types - train_set) / len(te_types),
        })
    return out


# ------------------------------------------------------------- la ponderacion
def build_vocab(toks, parts, min_df=2, drop_stop=False, stop=None):
    df = Counter()
    for t, p in zip(toks, parts):
        if p != "train":
            continue
        df.update(set(t))
    words = sorted(w for w, d in df.items()
                   if d >= min_df and not (drop_stop and w in stop))
    return {w: i for i, w in enumerate(words)}, words, df


def doc_matrix(toks, vocab):
    """CSR a mano: sin sklearn, para que la pagina pueda repetirlo igual."""
    indptr = [0]
    indices = []
    data = []
    for t in toks:
        c = Counter(vocab[w] for w in t if w in vocab)
        for j in sorted(c):
            indices.append(j)
            data.append(c[j])
        indptr.append(len(indices))
    return (np.array(data, dtype=np.float64), np.array(indices, dtype=np.int32),
            np.array(indptr, dtype=np.int64))


def weight(mat, n_docs, df_arr, tf_mode, use_idf, normalise):
    """La formula, escrita como las tres decisiones que es.

    idf es la suavizada, ln((1+N)/(1+df)) + 1, que es la de scikit-learn y la
    unica que no se va a infinito con un termino que solo esta en test.
    """
    data, indices, indptr = mat
    v = data.copy()
    if tf_mode == "binary":
        v[:] = 1.0
    elif tf_mode == "sublinear":
        v = 1.0 + np.log(v)
    elif tf_mode != "raw":
        raise ValueError(tf_mode)
    if use_idf:
        idf = np.log((1.0 + n_docs) / (1.0 + df_arr)) + 1.0
        v = v * idf[indices]
    if normalise:
        for i in range(len(indptr) - 1):
            a, b = indptr[i], indptr[i + 1]
            nrm = math.sqrt(float(np.dot(v[a:b], v[a:b])))
            if nrm > 0:
                v[a:b] /= nrm
    return v


def to_csr(v, mat, n_rows, n_cols):
    from scipy.sparse import csr_matrix
    _, indices, indptr = mat
    return csr_matrix((v, indices, indptr), shape=(n_rows, n_cols))


def score(Xtr, ytr, Xte, yte, clf_name):
    """Ajusta y puntua, y **grita si el solver no converge**.

    Un solver que agota sus iteraciones devuelve un modelo peor sin decir nada
    mas que un warning, y este experimento compara doce representaciones entre
    si: una celda que no converge no mide su representacion, mide el limite de
    iteraciones. Se cuenta cuantas celdas avisan y el numero se publica.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    if clf_name == "svm":
        clf = LinearSVC(C=1.0, random_state=SEED, dual=True, max_iter=MAX_ITER)
    else:
        clf = LogisticRegression(C=10.0, random_state=SEED, max_iter=5000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(Xtr, ytr)
        stalled = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    p = clf.predict(Xte)
    return (float(accuracy_score(yte, p)),
            float(f1_score(yte, p, average="macro")),
            bool(stalled))


def factorial(toks, labels, parts, stop):
    """Las doce celdas: tres formas de tf, idf si o no, normalizar si o no.

    Con dos clasificadores, que es el control de que el orden de las celdas es
    una propiedad de la representacion y no del modelo que la lee.
    """
    vocab, words, df = build_vocab(toks, parts, min_df=2)
    mat = doc_matrix(toks, vocab)
    df_arr = np.array([df[w] for w in words], dtype=float)
    y = np.array(labels)
    tr = np.array([p == "train" for p in parts])
    n_train = int(tr.sum())
    rows = []
    for tf_mode in ("binary", "raw", "sublinear"):
        for use_idf in (False, True):
            for normalise in (False, True):
                v = weight(mat, n_train, df_arr, tf_mode, use_idf, normalise)
                X = to_csr(v, mat, len(toks), len(words))
                cell = {"tf": tf_mode, "idf": use_idf, "norm": normalise}
                for clf in ("svm", "logreg"):
                    acc, f1, stalled = score(X[tr], y[tr], X[~tr], y[~tr], clf)
                    cell[f"{clf}_acc"] = acc
                    cell[f"{clf}_f1"] = f1
                    cell[f"{clf}_stalled"] = stalled
                rows.append(cell)
                print(f"    tf={tf_mode:<9} idf={int(use_idf)} norm={int(normalise)}  "
                      f"svm {cell['svm_acc']:.4f}/{cell['svm_f1']:.4f}  "
                      f"logreg {cell['logreg_acc']:.4f}/{cell['logreg_f1']:.4f}")
    return {"cells": rows, "vocab": len(words), "n_train": n_train,
            "n_test": int((~tr).sum())}


def noise_floor(toks, labels, parts, reps=400):
    """Cuanto se mueve macro F1 sin que cambie nada de la representacion.

    Hace falta antes de escribir una sola frase sobre el factorial, porque el
    resultado que se ve es que idf mueve el marcador menos que la normalizacion
    y a veces hacia abajo, y "menos" y "hacia abajo" solo significan algo contra
    un suelo. Dos suelos, que miden cosas distintas y no se pueden mezclar:

      muestreo  remuestreo con reemplazo de los 2.207 documentos de test sobre
                las MISMAS predicciones. Es lo que el conjunto de test puede
                resolver, y es el suelo que se aplica a comparar dos celdas
                puntuadas sobre el mismo test.
      solver    la misma celda ajustada con diez semillas distintas del SVM.
                El dual de liblinear recorre los ejemplos en orden aleatorio,
                asi que dos ajustes de la misma matriz no dan el mismo modelo.

    El suelo que se publica es el mayor de los dos, porque una diferencia entre
    celdas tiene que sobrevivir a los dos ruidos a la vez.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import LinearSVC
    from sklearn.metrics import f1_score, accuracy_score

    vocab, words, df = build_vocab(toks, parts, min_df=2)
    mat = doc_matrix(toks, vocab)
    df_arr = np.array([df[w] for w in words], dtype=float)
    y = np.array(labels)
    tr = np.array([p == "train" for p in parts])
    n_train = int(tr.sum())
    v = weight(mat, n_train, df_arr, "sublinear", True, True)
    X = to_csr(v, mat, len(toks), len(words))
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[~tr], y[~tr]

    clf = LinearSVC(C=1.0, random_state=SEED, dual=True, max_iter=MAX_ITER).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    rng = np.random.default_rng(SEED)
    n = len(yte)
    boot_f1, boot_acc = [], []
    for _ in range(reps):
        s = rng.integers(0, n, n)
        boot_f1.append(f1_score(yte[s], pred[s], average="macro"))
        boot_acc.append(accuracy_score(yte[s], pred[s]))

    seed_f1, seed_acc = [], []
    stalls = 0
    for s in range(10):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            c = LinearSVC(C=1.0, random_state=s, dual=True, max_iter=MAX_ITER).fit(Xtr, ytr)
            stalls += any(issubclass(w.category, ConvergenceWarning) for w in caught)
        p = c.predict(Xte)
        seed_f1.append(f1_score(yte, p, average="macro"))
        seed_acc.append(accuracy_score(yte, p))

    out = {
        "boot_reps": reps,
        "boot_f1_sd": float(np.std(boot_f1)),
        "boot_acc_sd": float(np.std(boot_acc)),
        "seed_reps": 10,
        "seed_f1_sd": float(np.std(seed_f1)),
        "seed_acc_sd": float(np.std(seed_acc)),
        "seed_f1_range": [float(np.min(seed_f1)), float(np.max(seed_f1))],
        "seed_stalls": int(stalls),
    }
    out["f1_floor"] = max(out["boot_f1_sd"], out["seed_f1_sd"])
    out["acc_floor"] = max(out["boot_acc_sd"], out["seed_acc_sd"])
    print(f"    remuestreo del test: sd de macro F1 {out['boot_f1_sd']:.4f}, "
          f"exactitud {out['boot_acc_sd']:.4f}")
    print(f"    diez semillas del SVM: sd de macro F1 {out['seed_f1_sd']:.4f}, "
          f"exactitud {out['seed_acc_sd']:.4f}")
    print(f"    suelo publicado: {out['f1_floor']:.4f} en macro F1")
    return out


def idf_mechanism(toks, labels, parts):
    """Por que idf no compra nada aqui, escrito como una prediccion falsable.

    idf es un reescalado por columna, y un modelo lineal ya tiene un peso por
    columna: sin regularizacion, multiplicar la columna j por c y dividir su
    peso por c da EXACTAMENTE la misma funcion, asi que idf no puede cambiar
    nada. Lo unico que rompe esa equivalencia es la penalizacion, que mide los
    pesos y por tanto no es invariante a la escala.

    La prediccion, entonces, es que el hueco de idf se va a cero segun se afloja
    la regularizacion, y no que sea pequeno porque si. Se barre C y se mira.
    """
    y = np.array(labels)
    tr = np.array([p == "train" for p in parts])
    vocab, words, df = build_vocab(toks, parts, min_df=2)
    mat = doc_matrix(toks, vocab)
    df_arr = np.array([df[w] for w in words], dtype=float)
    n_train = int(tr.sum())
    X = {u: to_csr(weight(mat, n_train, df_arr, "sublinear", u, True), mat,
                   len(toks), len(words)) for u in (False, True)}
    out = []
    for C in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        row = {"C": C}
        for u in (False, True):
            from sklearn.svm import LinearSVC
            from sklearn.metrics import accuracy_score, f1_score
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                clf = LinearSVC(C=C, random_state=SEED, dual=True,
                                max_iter=MAX_ITER).fit(X[u][tr], y[tr])
                st = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            p = clf.predict(X[u][~tr])
            key = "idf" if u else "plain"
            row[f"{key}_acc"] = float(accuracy_score(y[~tr], p))
            row[f"{key}_f1"] = float(f1_score(y[~tr], p, average="macro"))
            row[f"{key}_stalled"] = bool(st)
        row["gap_f1"] = row["idf_f1"] - row["plain_f1"]
        out.append(row)
        print(f"    C={C:<7} sin idf {row['plain_f1']:.4f}  con idf {row['idf_f1']:.4f}  "
              f"hueco {row['gap_f1']:+.4f}"
              + ("  (uno de los dos no convergio)"
                 if row["idf_stalled"] or row["plain_stalled"] else ""))
    return out


def retrieval_idf(toks, labels, parts):
    """El otro lado del mismo mango: recuperar no tiene pesos que aprender.

    En clasificacion hay un peso por columna que puede absorber a idf. En una
    busqueda por coseno no hay ninguno: la unica ponderacion que existe es la
    que traiga el vector. Si idf sirve para algo, tiene que verse aqui, y con
    su propio suelo de ruido al lado.
    """
    vocab, words, df = build_vocab(toks, parts, min_df=2)
    mat = doc_matrix(toks, vocab)
    df_arr = np.array([df[w] for w in words], dtype=float)
    n_train = sum(1 for p in parts if p == "train")
    te = np.array([p == "test" for p in parts])
    y = np.array(labels)[te]
    out = {}
    rng = np.random.default_rng(SEED)
    for tf_mode in ("binary", "raw", "sublinear"):
        for u in (False, True):
            X = to_csr(weight(mat, n_train, df_arr, tf_mode, u, True), mat,
                       len(toks), len(words))[te]
            C = (X @ X.T).toarray()
            np.fill_diagonal(C, -np.inf)
            order = np.argsort(-C, axis=1)
            hit1 = (y[order[:, 0]] == y)
            hit10 = (y[order[:, :10]] == y[:, None])
            # El suelo de esta cifra es el remuestreo de las consultas, que es
            # la unica fuente de variacion que hay: no se ajusta nada.
            boots = [hit10[rng.integers(0, len(y), len(y))].mean() for _ in range(200)]
            out[f"{tf_mode}_{int(u)}"] = {
                "p1": float(hit1.mean()),
                "p10": float(hit10.mean()),
                "p10_sd": float(np.std(boots)),
            }
    for tf_mode in ("binary", "raw", "sublinear"):
        a = out[f"{tf_mode}_0"]["p10"]
        b = out[f"{tf_mode}_1"]["p10"]
        print(f"    {tf_mode:<10} P@10 sin idf {a:.4f}  con idf {b:.4f}  "
              f"hueco {b - a:+.4f} (suelo {out[f'{tf_mode}_1']['p10_sd']:.4f})")
    return out


def term_value(toks, labels, parts, stop):
    """Lo que idf premia contra lo que de verdad distingue una categoria.

    idf es una funcion de una sola cosa: en cuantos documentos sale el termino.
    No mira las etiquetas, porque no puede: se calcula antes de que exista una
    tarea. La pregunta medible es si ese proxy ordena los terminos como los
    ordenaria alguien que si mira las etiquetas, y la respuesta se mide con la
    informacion mutua entre "sale el termino" y "la categoria es esta", sobre
    train y nada mas que train.

    Se publican las dos correlaciones (Pearson sobre los valores y Spearman
    sobre los rangos) porque miden cosas distintas y la segunda es la que
    importa: a nadie le interesa si idf y MI estan en la misma escala, le
    interesa si ponen los mismos terminos arriba.
    """
    from scipy.stats import spearmanr
    tr = [(t, l) for t, l, p in zip(toks, labels, parts) if p == "train"]
    n = len(tr)
    df = Counter()
    for t, _ in tr:
        df.update(set(t))
    cls = Counter(l for _, l in tr)
    joint = {c: Counter() for c in cls}
    for t, l in tr:
        joint[l].update(set(t))

    words = sorted(w for w, d in df.items() if d >= 2)
    idf, mi = [], []
    for w in words:
        d = df[w]
        idf.append(math.log((1 + n) / (1 + d)) + 1)
        # I(presencia del termino ; categoria), en nats, con los cuatro cuadrantes.
        acc = 0.0
        p_w = d / n
        for c, nc in cls.items():
            p_c = nc / n
            a = joint[c][w] / n
            for p_joint, p_a in ((a, p_w), (p_c - a, 1 - p_w)):
                if p_joint > 0:
                    acc += p_joint * math.log(p_joint / (p_a * p_c))
        mi.append(acc)
    idf = np.array(idf)
    mi = np.array(mi)
    rho = float(spearmanr(idf, mi).statistic)
    r = float(np.corrcoef(idf, mi)[0, 1])

    # "Los quince de mayor idf" es un empate enorme, no un ranking: idf es una
    # funcion de df sola, asi que TODOS los terminos que salen en 2 documentos
    # comparten el maximo. Cogidos por argsort salen los quince primeros por
    # orden alfabetico, que se lee como si empezar por "a" fuese informativo.
    # Se toma una muestra repartida por el grupo empatado y el recuento del
    # empate se publica, que es la cifra honesta.
    top_val = idf.max()
    tied = np.flatnonzero(idf >= top_val - 1e-12)
    top_idf = tied[np.linspace(0, len(tied) - 1, min(15, len(tied))).astype(int)]
    top_mi = np.argsort(-mi)[:15]
    # Cuantos de los 200 terminos con mas informacion estan en el 50% de mayor
    # idf: la version de la correlacion que se puede leer en una frase.
    cut = np.median(idf)
    best200 = np.argsort(-mi)[:200]
    stop_set = set(stop)
    return {
        "terms": len(words),
        "pearson": r,
        "spearman": rho,
        "top_idf": [{"w": words[i], "idf": float(idf[i]), "mi": float(mi[i]),
                     "df": int(df[words[i]])} for i in top_idf],
        "top_mi": [{"w": words[i], "idf": float(idf[i]), "mi": float(mi[i]),
                    "df": int(df[words[i]])} for i in top_mi],
        "informative_above_median_idf": int((idf[best200] > cut).sum()),
        "median_idf": float(cut),
        "top_mi_stopwords": int(sum(1 for i in best200 if words[i] in stop_set)),
        # El techo de la informacion mutua NO es empirico, es aritmetica: la
        # informacion que puede llevar la presencia de un termino esta acotada
        # por su propia entropia, H(p) con p = df/n, que vale cero en df = 1 y
        # es maxima en df = n/2. Es decir que un termino rarisimo no puede ser
        # informativo por mucho que correlacione, y idf ordena exactamente al
        # reves de esa cota. La correlacion negativa de arriba es en buena parte
        # esta cota, y decirlo es el articulo: idf no puede estar ordenando por
        # utilidad porque la utilidad tiene un techo que depende de la
        # frecuencia y idf es una funcion decreciente de la frecuencia.
        "ceiling": [{"df": int(d),
                     "idf": math.log((1 + n) / (1 + d)) + 1,
                     "cap": float(-(d / n) * math.log(d / n)
                                  - (1 - d / n) * math.log(1 - d / n))}
                    for d in sorted({max(1, min(n - 1, int(round(x))))
                                     for x in np.logspace(0, math.log10(n - 1), 70)})],
        "best_mi": float(mi.max()),
        "best_mi_df": int(df[words[int(mi.argmax())]]),
        "best_mi_word": words[int(mi.argmax())],
        "n_train": n,
        # La nube diezmada para dibujarla. Una muestra uniforme sola **recorta
        # la cima**: los terminos que llevan mas informacion son unos pocos
        # cientos entre diez mil, asi que caen fuera de la muestra y el dibujo
        # acaba diciendo que el maximo es 0,25 cuando el texto de al lado dice
        # 0,34. Se unen la muestra uniforme (que da la forma) y los 300 de mas
        # informacion (que dan el extremo), que es lo que el grafico afirma.
        "cloud": [list(v) for v in sorted({
            (round(float(idf[i]), 4), round(float(mi[i]), 6))
            for i in list(np.linspace(0, len(words) - 1, 1300).astype(int))
                     + list(np.argsort(-mi)[:300])
        })],
        "tied_at_max_idf": int(len(tied)),
        "max_idf_df": int(df[words[int(tied[0])]]),
    }


def stopword_arm(toks, labels, parts, stop):
    """Quitar la lista de vacias, con idf puesto y con idf quitado.

    La pregunta no es si ayuda, es si aporta algo que idf no estuviera haciendo
    ya, y eso solo se ve con los dos brazos.
    """
    y = np.array(labels)
    tr = np.array([p == "train" for p in parts])
    out = []
    for drop in (False, True):
        vocab, words, df = build_vocab(toks, parts, min_df=2, drop_stop=drop, stop=set(stop))
        mat = doc_matrix(toks, vocab)
        df_arr = np.array([df[w] for w in words], dtype=float)
        for use_idf in (False, True):
            v = weight(mat, int(tr.sum()), df_arr, "sublinear", use_idf, True)
            X = to_csr(v, mat, len(toks), len(words))
            acc, f1, stalled = score(X[tr], y[tr], X[~tr], y[~tr], "svm")
            out.append({"dropped": drop, "idf": use_idf, "vocab": len(words),
                        "acc": acc, "f1": f1, "stalled": stalled})
            print(f"    stopwords {'out' if drop else 'in '} idf={int(use_idf)}  "
                  f"{len(words):>6} features  {acc:.4f}/{f1:.4f}")
    return out


def ngram_arm(toks, labels, parts):
    """Que compra subir de n, medido con el coste al lado."""
    y = np.array(labels)
    tr = np.array([p == "train" for p in parts])
    n_train = int(tr.sum())
    out = []
    for nmax in (1, 2, 3):
        grams = []
        for t in toks:
            g = []
            for n in range(1, nmax + 1):
                for i in range(len(t) - n + 1):
                    g.append(" ".join(t[i:i + n]))
            grams.append(g)
        vocab, words, df = build_vocab(grams, parts, min_df=2)
        mat = doc_matrix(grams, vocab)
        df_arr = np.array([df[w] for w in words], dtype=float)
        v = weight(mat, n_train, df_arr, "sublinear", True, True)
        X = to_csr(v, mat, len(toks), len(words))
        acc, f1, stalled = score(X[tr], y[tr], X[~tr], y[~tr], "svm")
        out.append({"nmax": nmax, "features": len(words), "acc": acc, "f1": f1,
                    "nnz": int(mat[0].size), "stalled": stalled})
        print(f"    up to {nmax}-grams: {len(words):>7} features  {acc:.4f}/{f1:.4f}")
    return out


def mindf_sweep(toks, labels, parts):
    """Casi todo el vocabulario no hace nada, y la curva lo dice."""
    y = np.array(labels)
    tr = np.array([p == "train" for p in parts])
    out = []
    for min_df in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500):
        vocab, words, df = build_vocab(toks, parts, min_df=min_df)
        if not words:
            continue
        mat = doc_matrix(toks, vocab)
        df_arr = np.array([df[w] for w in words], dtype=float)
        v = weight(mat, int(tr.sum()), df_arr, "sublinear", True, True)
        X = to_csr(v, mat, len(toks), len(words))
        acc, f1, stalled = score(X[tr], y[tr], X[~tr], y[~tr], "svm")
        out.append({"min_df": min_df, "features": len(words), "acc": acc, "f1": f1,
                    "stalled": stalled})
        print(f"    min_df {min_df:>3}: {len(words):>6} features  {acc:.4f}/{f1:.4f}")
    return out


# ------------------------------------------------------------- longitud y coseno
def length_experiment(toks, labels, parts, vocab, words, df, n_train):
    """La normalizacion no es cosmetica: decide a que se parece un documento.

    Se mide el vecino mas cercano de cada documento de test bajo las dos
    geometrias sobre los MISMOS vectores sin normalizar (euclidea) y
    normalizados (coseno, que es lo mismo que euclidea sobre los normalizados),
    y se compara en dos ejes: si el vecino comparte categoria, y cuanto se
    parecen de longitud. La segunda es la que explica la primera.
    """
    mat = doc_matrix(toks, vocab)
    df_arr = np.array([df[w] for w in words], dtype=float)
    raw = weight(mat, n_train, df_arr, "sublinear", True, False)
    Xr = to_csr(raw, mat, len(toks), len(words))
    Xn = to_csr(weight(mat, n_train, df_arr, "sublinear", True, True), mat,
                len(toks), len(words))
    te = np.array([p == "test" for p in parts])
    y = np.array(labels)
    lens = np.array([len(t) for t in toks], dtype=float)

    Ar, An = Xr[te], Xn[te]
    yt, lt = y[te], lens[te]
    norms = np.sqrt(Ar.multiply(Ar).sum(axis=1)).A.ravel()

    # Euclidea sobre los sin normalizar contra coseno sobre los normalizados.
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, asi que las dos salen del mismo
    # producto y no hay dos implementaciones que puedan discrepar.
    G = (Ar @ Ar.T).toarray()
    d2 = norms[:, None] ** 2 + norms[None, :] ** 2 - 2 * G
    C = (An @ An.T).toarray()
    np.fill_diagonal(d2, np.inf)
    np.fill_diagonal(C, -np.inf)
    nn_e = d2.argmin(axis=1)
    nn_c = C.argmax(axis=1)
    res = {
        "n": int(te.sum()),
        "corr_len_norm": float(np.corrcoef(lt, norms)[0, 1]),
        "euclid_same_label": float((yt[nn_e] == yt).mean()),
        "cosine_same_label": float((yt[nn_c] == yt).mean()),
        "euclid_len_gap": float(np.abs(lt[nn_e] - lt).mean()),
        "cosine_len_gap": float(np.abs(lt[nn_c] - lt).mean()),
        "median_len": float(np.median(lt)),
        "len_min": int(lt.min()), "len_max": int(lt.max()),
    }
    # Precision media a 10, que es la cifra de recuperacion de verdad.
    for name, S in (("euclid", -d2), ("cosine", C)):
        top = np.argsort(-S, axis=1)[:, :10]
        hit = (yt[top] == yt[:, None])
        res[f"{name}_p10"] = float(hit.mean())
    return res


# ------------------------------------------------- lo que viaja al navegador
def ship(toks, labels, parts, texts):
    """El indice que la pagina busca en vivo, y sus referencias.

    Se manda el test entero porque es la particion sobre la que el articulo
    cita todas sus cifras de recuperacion, y mandar una muestra obligaria a
    decir en cada pie que la pagina y la tabla miran cosas distintas.

    Viajan CONTEOS ENTEROS, no pesos. El navegador rehace idf y las tres formas
    de tf desde los mismos enteros, asi que el guardia puede exigir igualdad a
    1e-12 y no "coincide dentro de lo razonable": no hay ningun redondeo entre
    las dos orillas del que dudar.
    """
    tr_toks = [t for t, p in zip(toks, parts) if p == "train"]
    df = Counter()
    for t in tr_toks:
        df.update(set(t))
    n_train = len(tr_toks)
    # El vocabulario que viaja se corta por df para que quepa; el corte se
    # publica y la tabla de min_df de arriba dice exactamente lo que cuesta.
    words = sorted(w for w, d in df.items() if d >= 4)
    vocab = {w: i for i, w in enumerate(words)}
    dfv = [df[w] for w in words]

    idx_te = [i for i, p in enumerate(parts) if p == "test"]
    flat_idx, flat_cnt, ptr = [], [], [0]
    for i in idx_te:
        c = Counter(vocab[w] for w in toks[i] if w in vocab)
        for j in sorted(c):
            flat_idx.append(j)
            flat_cnt.append(min(c[j], 255))
        ptr.append(len(flat_idx))
    over = sum(1 for i in idx_te for v in Counter(
        vocab[w] for w in toks[i] if w in vocab).values() if v > 255)
    assert over == 0, f"{over} conteos por encima de 255, uint8 no vale"

    titles = []
    for i in idx_te:
        first = texts[i].strip().splitlines()[0] if texts[i].strip() else ""
        titles.append(first[:64])
    return {
        "words": words,
        "df": b64_u32(dfv),
        "n_train": n_train,
        "min_df": 4,
        "idx": b64_u16(flat_idx),
        "cnt": b64_u8(flat_cnt),
        "ptr": b64_u32(ptr),
        "nnz": len(flat_idx),
        "docs": len(idx_te),
        "labels": [labels[i] for i in idx_te],
        "lens": b64_u16([len(toks[i]) for i in idx_te]),
        "titles": titles,
        "label_names": R8,
    }


def ship_refs(shipped, toks, labels, parts):
    """Lo que el navegador tiene que reproducir exactamente al cargar."""
    words = shipped["words"]
    vocab = {w: i for i, w in enumerate(words)}
    n_train = shipped["n_train"]
    idx_te = [i for i, p in enumerate(parts) if p == "test"]
    df_arr = np.array([0.0] * len(words))
    dfc = Counter()
    for t, p in zip(toks, parts):
        if p == "train":
            dfc.update(set(t))
    for w, j in vocab.items():
        df_arr[j] = dfc[w]

    sub = [toks[i] for i in idx_te]
    mat = doc_matrix(sub, vocab)
    out = {}
    for tf_mode in ("binary", "raw", "sublinear"):
        for use_idf in (False, True):
            v = weight(mat, n_train, df_arr, tf_mode, use_idf, True)
            X = to_csr(v, mat, len(sub), len(words))
            key = f"{tf_mode}_{int(use_idf)}"
            # Los diez terminos de mayor peso de tres documentos, y la suma de
            # cuadrados de cada uno: el primero comprueba el orden y el segundo
            # comprueba las magnitudes, que un orden solo no fija.
            probe = [0, len(sub) // 3, len(sub) - 1]
            out[key] = {
                "top": [top_terms(X, words, d, 10) for d in probe],
                "sumsq": [float(X[d].multiply(X[d]).sum()) for d in probe],
            }
    v = weight(mat, n_train, df_arr, "sublinear", True, True)
    X = to_csr(v, mat, len(sub), len(words))
    C = (X @ X.T).toarray()
    np.fill_diagonal(C, -np.inf)
    queries = [0, 40, 400, len(sub) // 2, len(sub) - 3]
    out["cosine_top"] = [{
        "q": int(q),
        "top": [[int(j), round(float(C[q, j]), 9)] for j in np.argsort(-C[q])[:8]],
    } for q in queries]
    out["probe_docs"] = [0, len(sub) // 3, len(sub) - 1]
    return out


def top_terms(X, words, d, k):
    """Los k terminos de mas peso, con el desempate ESCRITO.

    Con la ponderacion binaria todos los terminos de un documento valen lo
    mismo, asi que sin un desempate declarado los dos lados imprimen listas
    distintas de numeros identicos y el guardia dispara sin que haya nada roto.
    El desempate es el indice del vocabulario, que es alfabetico, y el navegador
    ordena por el mismo par.
    """
    row = X[d]
    order = sorted(range(row.data.size),
                   key=lambda o: (-row.data[o], int(row.indices[o])))[:k]
    return [[words[row.indices[o]], round(float(row.data[o]), 9)] for o in order]


def idf_examples(toks, parts, n_show=12):
    df = Counter()
    n_train = 0
    for t, p in zip(toks, parts):
        if p != "train":
            continue
        n_train += 1
        df.update(set(t))
    common = [w for w, _ in df.most_common(n_show)]
    rare = [w for w, d in sorted(df.items(), key=lambda kv: (kv[1], kv[0])) if d >= 4][:n_show]
    def row(w):
        d = df[w]
        return {"w": w, "df": d, "idf": math.log((1 + n_train) / (1 + d)) + 1}
    return {"n_train": n_train, "common": [row(w) for w in common],
            "rare": [row(w) for w in rare],
            "curve": [{"df": d, "idf": math.log((1 + n_train) / (1 + d)) + 1}
                      for d in sorted({max(1, int(round(x)))
                                       for x in np.logspace(0, math.log10(n_train), 60)})]}


def sklearn_reference(toks, labels, parts):
    """El segundo solver: la misma celda calculada por scikit-learn.

    No es decoracion. La formula de idf tiene media docena de convenios y esta
    pagina implementa uno; si el que implementa no es el que dice implementar,
    esta comprobacion es la unica que lo dice.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vocab, words, df = build_vocab(toks, parts, min_df=2)
    tr = [i for i, p in enumerate(parts) if p == "train"]
    docs = [" ".join(t) for t in toks]
    vec = TfidfVectorizer(vocabulary=vocab, sublinear_tf=True, smooth_idf=True,
                          norm="l2", token_pattern=r"\S+", lowercase=False)
    vec.fit([docs[i] for i in tr])
    S = vec.transform(docs)
    mat = doc_matrix(toks, vocab)
    df_arr = np.array([df[w] for w in words], dtype=float)
    M = to_csr(weight(mat, len(tr), df_arr, "sublinear", True, True), mat,
               len(toks), len(words))
    delta = abs(S - M)
    worst = float(delta.max()) if delta.nnz else 0.0
    return {"worst_abs_diff": worst, "nnz": int(S.nnz)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("cargando el corpus")
    texts, labels, parts, toks = cached("corpus_r8", load_corpus)
    stop = stopwords()
    print(f"  {len(texts)} documentos, {sum(1 for p in parts if p == 'train')} train, "
          f"{sum(len(t) for t in toks)} tokens")

    print("zipf y heaps")
    zi = cached("zipf", lambda: zipf_stats(toks))
    he = cached("heaps", lambda: heaps(toks))
    print(f"  {zi['types']} tipos, {zi['hapax']} de una sola aparicion "
          f"({zi['hapax_frac']:.1%}), pendiente {zi['slope']:.4f}")

    print("inflacion de n-gramas")
    ng = cached("ngram_inflation", lambda: ngram_inflation(toks, parts))
    for r in ng:
        print(f"  n={r['n']}: {r['types']:>9} tipos, {r['hapax_frac']:.1%} una vez, "
              f"{r['test_unseen_token_frac']:.1%} de los del test nunca vistos")

    print("el factorial de la formula")
    fac = cached("factorial", lambda: factorial(toks, labels, parts, stop))

    print("el suelo de ruido, antes de comparar nada")
    nf = cached("noise_floor", lambda: noise_floor(toks, labels, parts))

    print("por que idf no compra nada: barrido de regularizacion")
    im = cached("idf_mechanism", lambda: idf_mechanism(toks, labels, parts))

    print("y el lado donde si deberia comprar: recuperacion")
    ri = cached("retrieval_idf", lambda: retrieval_idf(toks, labels, parts))

    print("que premia idf contra que distingue de verdad")
    tv = cached("term_value", lambda: term_value(toks, labels, parts, stop))
    print(f"  spearman(idf, informacion mutua) = {tv['spearman']:.4f}; "
          f"de los 200 terminos mas informativos, {tv['informative_above_median_idf']} "
          f"estan por encima del idf mediano")

    print("las palabras vacias")
    sw = cached("stopword_arm", lambda: stopword_arm(toks, labels, parts, stop))

    print("subir de n")
    na = cached("ngram_arm", lambda: ngram_arm(toks, labels, parts))

    print("barrido de min_df")
    md = cached("mindf", lambda: mindf_sweep(toks, labels, parts))

    print("longitud, euclidea y coseno")
    vocab, words, df = build_vocab(toks, parts, min_df=2)
    n_train = sum(1 for p in parts if p == "train")
    ln = cached("length", lambda: length_experiment(toks, labels, parts, vocab, words,
                                                    df, n_train))
    print(f"  vecino euclideo comparte categoria {ln['euclid_same_label']:.4f}, "
          f"coseno {ln['cosine_same_label']:.4f}")

    print("referencia de scikit-learn")
    skl = cached("sklearn_ref", lambda: sklearn_reference(toks, labels, parts))
    print(f"  peor diferencia absoluta contra TfidfVectorizer: {skl['worst_abs_diff']:.3e}")

    print("empaquetando el indice del navegador")
    shipped = ship(toks, labels, parts, texts)
    refs = cached("ship_refs", lambda: ship_refs(shipped, toks, labels, parts))
    print(f"  {shipped['docs']} documentos, {len(shipped['words'])} terminos, "
          f"{shipped['nnz']} no nulos")

    data = {
        "meta": {
            "corpus": "Reuters-21578, ModApte split, single-label documents in the "
                      "eight largest categories",
            "docs": len(texts),
            "train": n_train,
            "test": len(texts) - n_train,
            "categories": R8,
            "category_counts": {c: labels.count(c) for c in R8},
            "tokenizer": "lowercase, runs of ascii letters, nothing else",
            "stopword_list": len(stop),
            "seed": SEED,
        },
        "zipf": zi,
        "heaps": he,
        "ngrams": ng,
        "idf": idf_examples(toks, parts),
        "factorial": fac,
        "noise": nf,
        "idf_mechanism": im,
        "retrieval": ri,
        "term_value": tv,
        "stopwords": sw,
        "ngram_arm": na,
        "mindf": md,
        "length": ln,
        "sklearn": skl,
        "search": shipped,
        "refs": refs,
    }
    path = OUT / "tfidf.json"
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
