"""Las cifras del artículo de AutoML.

El artículo anterior busca dos mandos de un modelo fijo. Este busca el modelo,
lo que va delante del modelo y lo que va detrás, que es lo que una herramienta
de AutoML vende. Y como en el anterior, la única forma de decir algo que no sea
publicidad es **enumerar el espacio entero** y quedarse con la respuesta.

El espacio son cuatro decisiones que se combinan: cómo se escalan las columnas,
si se reducen y cómo, si se pesan las clases, y qué modelo con qué ajuste. Salen
unas cuantas centenas de tuberías por tabla, y cada una se puntúa por validación
cruzada (lo que la búsqueda ve) y sobre un trozo retenido (lo que nadie ve).
Cuatro tablas distintas, tres para aprender el atajo y una para probarlo en frío.

Lo que se mide:

  1. **Cuánto compra buscar.** La tubería por defecto contra la mejor
     encontrada, sobre el retenido y con el suelo de ruido del propio retenido
     al lado, porque media docena de décimas sobre 400 filas no son nada.
  2. **Qué decisión manda.** Las cuatro en las mismas unidades: cuánto se mueve
     lo mejor alcanzable al fijar cada eje en su peor valor. Es la única forma
     de decir "el preprocesado importa más que el modelo" sin que sea una
     opinión.
  3. **La maldición del ganador, a escala de AutoML.** Con cientos de
     candidatos el mejor de validación está sesgado hacia arriba, y el sesgo se
     calcula exactamente porque las dos puntuaciones de cada candidato están.
     Con la validación cruzada anidada al lado, y lo que cuesta.
  4. **El atajo que hace de AutoML algo distinto de una búsqueda.** Una cartera
     de cuatro configuraciones elegida sobre las otras tablas y aplicada en
     frío a la cuarta: cuatro evaluaciones contra cuatro al azar contra la
     búsqueda entera.
  5. **Y lo que gana los concursos, que no es la búsqueda**: promediar las k
     mejores en vez de quedarse con la mejor. Se elige la mezcla con las
     probabilidades fuera de pliegue y se puntúa en el retenido, que es el
     único orden honesto.

Ejecutar desde cualquier sitio: `python src/utils/generate_automl_data.py`.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer,
                                   StandardScaler)
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlp_data import reuters, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "automl" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "automl_cache")

SEED = 19
FOLDS = 5
TEST_FRAC = 0.35
R8 = ("acq", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade")
R8_DOCS = 1500            # de las 7.674 de R8; el resto no cabe en el presupuesto
R8_DIMS = 50
DIGITS_N = 1200           # de los 1.797, por el mismo motivo
TREES = 80
BOOST_ITERS = 40
NESTED_OUTER = 3
ENSEMBLE_MAX = 25
PORTFOLIO = 4
BOOT = 400

# La huella de la configuración va en el nombre del caché, y no solo la de los
# tamaños: recortar el presupuesto a mitad de una tanda y dejar los nombres
# quietos publica una comparación entre configuraciones distintas sin un solo
# aviso, que es la trampa 59 de docs/verificacion.md.
TAG = (f"f{FOLDS}t{TEST_FRAC}s{SEED}r{R8_DOCS}d{R8_DIMS}g{DIGITS_N}"
       f"w{TREES}b{BOOST_ITERS}e{ENSEMBLE_MAX}p{PORTFOLIO}")


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
    """Redondeo para el JSON, con los no finitos convertidos en None."""
    if isinstance(v, dict):
        return {k: r(x, d) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [r(x, d) for x in v]
    if v is None or isinstance(v, (bool, str)):
        return v
    v = float(v)
    return None if not np.isfinite(v) else round(v, d)


# ------------------------------------------------------------------ las tablas
def reuters_lsa():
    """Las noticias de R8 como una tabla densa, por el camino del artículo 44.

    Conteos, ponderación por rareza, longitud normalizada y una descomposición
    en valores singulares. No es una decisión de esta página: es la tubería que
    el artículo de TF-IDF midió, congelada aquí para que la tabla exista.
    """
    docs, cats, split = reuters()
    keep = [i for i, c in enumerate(cats)
            if len(c) == 1 and c[0] in R8]
    rng = np.random.default_rng(SEED)
    keep = list(rng.permutation(keep)[:R8_DOCS])
    texts = [tokenize(docs[i]) for i in keep]
    y = np.array([R8.index(cats[i][0]) for i in keep])
    vocab = {}
    for t in texts:
        for w in set(t):
            vocab[w] = vocab.get(w, 0) + 1
    words = sorted([w for w, c in vocab.items() if c >= 5])
    idx = {w: k for k, w in enumerate(words)}
    X = np.zeros((len(texts), len(words)))
    for i, t in enumerate(texts):
        for w in t:
            j = idx.get(w)
            if j is not None:
                X[i, j] += 1.0
    df = (X > 0).sum(0)
    X = np.log1p(X) * np.log(len(texts) / np.maximum(df, 1))
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X /= np.maximum(n, 1e-9)
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return (U[:, :R8_DIMS] * S[:R8_DIMS]), y, len(words)


def tables():
    out = {}
    for name, load in (("wine", load_wine), ("cancer", load_breast_cancer),
                       ("digits", load_digits)):
        d = load()
        X, y = np.asarray(d.data, float), np.asarray(d.target, int)
        if name == "digits":
            keep = np.random.default_rng(SEED).permutation(len(y))[:DIGITS_N]
            X, y = X[np.sort(keep)], y[np.sort(keep)]
        out[name] = (X, y, None)
    Xr, yr, vocab = cached("reuters", reuters_lsa)
    out["reuters"] = (Xr, yr, vocab)
    return out


# --------------------------------------------------------------- el espacio
SCALERS = ("none", "standard", "minmax", "quantile")
REDUCERS = ("none", "select", "pca")
WEIGHTS = ("none", "balanced")
MODELS = (
    ("logistic-C0.1", dict(kind="logistic", C=0.1)),
    ("logistic-C1", dict(kind="logistic", C=1.0)),
    ("logistic-C10", dict(kind="logistic", C=10.0)),
    ("knn-1", dict(kind="knn", k=1)),
    ("knn-5", dict(kind="knn", k=5)),
    ("knn-15", dict(kind="knn", k=15)),
    ("svm-C1", dict(kind="svm", C=1.0)),
    ("svm-C10", dict(kind="svm", C=10.0)),
    ("svm-C100", dict(kind="svm", C=100.0)),
    ("forest-d3", dict(kind="forest", depth=3)),
    ("forest-d8", dict(kind="forest", depth=8)),
    ("forest-full", dict(kind="forest", depth=None)),
    ("boost-lr0.05", dict(kind="boost", lr=0.05)),
    ("boost-lr0.2", dict(kind="boost", lr=0.2)),
    ("boost-lr0.5", dict(kind="boost", lr=0.5)),
)
# La tubería que sale de la caja: nada delante y regresión logística con la C
# por defecto de scikit-learn. Es la referencia contra la que se lee todo.
DEFAULT = ("none", "none", "none", "logistic-C1")


def build(scaler, reducer, weight, model, n_features):
    steps = []
    if scaler == "standard":
        steps.append(("sc", StandardScaler()))
    elif scaler == "minmax":
        steps.append(("sc", MinMaxScaler()))
    elif scaler == "quantile":
        steps.append(("sc", QuantileTransformer(n_quantiles=100,
                                                output_distribution="normal",
                                                random_state=SEED)))
    k = max(2, n_features // 2)
    if reducer == "select":
        steps.append(("rd", SelectKBest(f_classif, k=k)))
    elif reducer == "pca":
        steps.append(("rd", PCA(n_components=min(k, n_features), random_state=SEED)))
    cw = "balanced" if weight == "balanced" else None
    spec = dict(MODELS)[model]
    kind = spec["kind"]
    if kind == "logistic":
        clf = LogisticRegression(C=spec["C"], max_iter=4000, class_weight=cw)
    elif kind == "knn":
        clf = KNeighborsClassifier(n_neighbors=spec["k"])
    elif kind == "svm":
        clf = SVC(C=spec["C"], gamma="scale", probability=False, class_weight=cw)
    elif kind == "forest":
        clf = RandomForestClassifier(n_estimators=TREES, max_depth=spec["depth"],
                                     random_state=SEED, class_weight=cw, n_jobs=1)
    else:
        clf = HistGradientBoostingClassifier(learning_rate=spec["lr"], max_iter=BOOST_ITERS,
                                             random_state=SEED,
                                             class_weight=cw)
    steps.append(("clf", clf))
    return Pipeline(steps)


def configs():
    out = []
    for s in SCALERS:
        for rd in REDUCERS:
            for w in WEIGHTS:
                for m, _ in MODELS:
                    out.append((s, rd, w, m))
    return out


def scores_matrix(name, X, y):
    """Para cada tubería: puntuación fuera de pliegue, del retenido, y su voto.

    Las probabilidades fuera de pliegue son lo único con lo que se puede elegir
    una mezcla sin mirar el retenido, así que se guardan aunque no se publiquen.
    La SVM no da probabilidades sin un ajuste extra por dentro, y aquí se
    quiere comparar tuberías y no calibraciones: el voto es la clase predicha
    en las dos, en forma de indicador.
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=SEED, stratify=y)
    classes = int(y.max()) + 1
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(Xtr, ytr))
    cfgs = configs()
    cv = np.zeros(len(cfgs))
    te = np.zeros(len(cfgs))
    oof = np.zeros((len(cfgs), len(ytr), classes), np.float32)
    tst = np.zeros((len(cfgs), len(yte), classes), np.float32)
    for ci, (s, rd, w, m) in enumerate(cfgs):
        acc = []
        for tr, va in folds:
            p = build(s, rd, w, m, X.shape[1])
            p.fit(Xtr[tr], ytr[tr])
            pred = p.predict(Xtr[va])
            acc.append(float(np.mean(pred == ytr[va])))
            oof[ci, va, pred] = 1.0
        cv[ci] = float(np.mean(acc))
        p = build(s, rd, w, m, X.shape[1])
        p.fit(Xtr, ytr)
        pred = p.predict(Xte)
        tst[ci, np.arange(len(yte)), pred] = 1.0
        te[ci] = float(np.mean(pred == yte))
        if (ci + 1) % 60 == 0:
            print(f"    {name}: {ci + 1}/{len(cfgs)} tuberías")
    return dict(cv=cv, test=te, oof=oof, tst=tst, ytr=ytr, yte=yte,
                n_train=len(ytr), n_test=len(yte), classes=classes,
                features=int(X.shape[1]))


# ------------------------------------------------------------------ 2. los ejes
def axis_moves(cfgs, cv):
    """Cuánto se mueve lo mejor alcanzable al fijar cada eje.

    La cantidad es la misma para las cuatro decisiones, que es todo el punto:
    "el preprocesado importa" y "el modelo importa" solo se pueden comparar si
    se miden con la misma regla.
    """
    axes = dict(scaler=0, reducer=1, weight=2, model=3)
    out = {}
    for name, k in axes.items():
        vals = sorted({c[k] for c in cfgs})
        best = {v: float(max(cv[i] for i, c in enumerate(cfgs) if c[k] == v))
                for v in vals}
        out[name] = dict(best=best, span=float(max(best.values()) - min(best.values())),
                         levels=len(vals),
                         worst_level=min(best, key=best.get),
                         best_level=max(best, key=best.get))
    return out


# ------------------------------------------------- 3. la maldición y el anidado
def curse(cv, te, trials=4000, budgets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 360)):
    rows = []
    for b in budgets:
        if b > len(cv):
            continue
        rng = np.random.default_rng(SEED + b)
        sel = rng.integers(0, len(cv), (trials, b))
        w = np.argmax(cv[sel], axis=1)
        take = sel[np.arange(trials), w]
        rows.append(dict(b=int(b), cv=float(cv[take].mean()),
                         test=float(te[take].mean()),
                         gap=float((cv[take] - te[take]).mean())))
    return rows


def nested(name, X, y, outer=NESTED_OUTER):
    """Validación cruzada anidada: el precio de una cifra honesta.

    La búsqueda entera se repite dentro de cada pliegue exterior, así que la
    cifra que sale no ha visto nunca los datos con los que se la puntúa. El
    coste es exactamente `outer` veces la búsqueda, y eso se cuenta en ajustes.
    """
    cfgs = configs()
    skf = StratifiedKFold(n_splits=outer, shuffle=True, random_state=SEED)
    accs, picks = [], []
    fits = 0
    for tr, te in skf.split(X, y):
        inner = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        sc = np.zeros(len(cfgs))
        for ci, (s, rd, w, m) in enumerate(cfgs):
            a = []
            for itr, iva in inner.split(X[tr], y[tr]):
                p = build(s, rd, w, m, X.shape[1])
                p.fit(X[tr][itr], y[tr][itr])
                a.append(float(np.mean(p.predict(X[tr][iva]) == y[tr][iva])))
                fits += 1
            sc[ci] = float(np.mean(a))
        best = int(np.argmax(sc))
        p = build(*cfgs[best], X.shape[1])
        p.fit(X[tr], y[tr])
        fits += 1
        accs.append(float(np.mean(p.predict(X[te]) == y[te])))
        picks.append(cfgs[best])
        print(f"    {name}: pliegue exterior {len(accs)}/{outer}")
    return dict(mean=float(np.mean(accs)), sd=float(np.std(accs)),
                folds=accs, picks=["|".join(p) for p in picks],
                fits=int(fits), outer=outer,
                same_pick=len({tuple(p) for p in picks}) == 1)


# ---------------------------------------------------------------- 4. la cartera
def portfolio(train_sets, cfgs, size=PORTFOLIO):
    """Cartera voraz: la lista corta que maximiza el mejor de la lista.

    No es un ranking. La segunda entrada se elige sabiendo cuál es la primera,
    porque lo que se quiere es cubrir tablas donde la primera falla, y eso es
    exactamente lo que hace un `argmax` sobre el máximo acumulado.
    """
    M = np.stack([s / s.max() for s in train_sets])   # tabla por fila, normalizada
    chosen, best = [], np.zeros(M.shape[0])
    for _ in range(size):
        gain = np.array([np.mean(np.maximum(best, M[:, c])) for c in range(M.shape[1])])
        gain[chosen] = -1
        k = int(np.argmax(gain))
        chosen.append(k)
        best = np.maximum(best, M[:, k])
    return [cfgs[c] for c in chosen], chosen


def portfolio_vs_random(cold, chosen, seeds=2000):
    """La cartera en frío contra el mismo número de sorteos al azar."""
    cv, te = cold["cv"], cold["test"]
    k = int(chosen[int(np.argmax(cv[chosen]))])
    rows = []
    rng = np.random.default_rng(SEED)
    rand = []
    for _ in range(seeds):
        pick = rng.choice(len(cv), len(chosen), replace=False)
        w = pick[int(np.argmax(cv[pick]))]
        rand.append(float(te[w]))
    rows.append(dict(arm="portfolio", evals=len(chosen), cv=float(cv[k]),
                     test=float(te[k])))
    rows.append(dict(arm="random", evals=len(chosen), cv=None,
                     test=float(np.mean(rand)), test_sd=float(np.std(rand)),
                     beats=float(np.mean(np.array(rand) >= te[k]))))
    full = int(np.argmax(cv))
    rows.append(dict(arm="full", evals=int(len(cv)), cv=float(cv[full]),
                     test=float(te[full])))
    return rows


# ------------------------------------------------------------- 5. la mezcla
def ensemble(sc, max_k=ENSEMBLE_MAX):
    """Selección de conjunto con reemplazo, al estilo de Caruana.

    Se parte del vacío y se añade, una a una, la tubería que más sube la
    puntuación **fuera de pliegue** del promedio actual, pudiendo repetir (que
    es como se le da más peso a una sin tener que ajustar pesos). El retenido no
    se toca hasta el final.
    """
    oof, tst, ytr, yte = sc["oof"], sc["tst"], sc["ytr"], sc["yte"]
    n_cfg = oof.shape[0]
    picks, cur_oof, cur_tst = [], np.zeros_like(oof[0]), np.zeros_like(tst[0])
    rows = []
    for step in range(max_k):
        best, best_acc = -1, -1.0
        for c in range(n_cfg):
            m = (cur_oof * len(picks) + oof[c]) / (len(picks) + 1)
            a = float(np.mean(np.argmax(m, 1) == ytr))
            if a > best_acc:
                best, best_acc = c, a
        cur_oof = (cur_oof * len(picks) + oof[best]) / (len(picks) + 1)
        cur_tst = (cur_tst * len(picks) + tst[best]) / (len(picks) + 1)
        picks.append(int(best))
        rows.append(dict(k=step + 1, oof=best_acc,
                         test=float(np.mean(np.argmax(cur_tst, 1) == yte)),
                         added=int(best), distinct=len(set(picks))))
    return dict(rows=rows, picks=picks)


def boot_sd(pred_ok, seeds=BOOT):
    rng = np.random.default_rng(SEED)
    n = len(pred_ok)
    return float(np.std([float(pred_ok[rng.integers(0, n, n)].mean())
                         for _ in range(seeds)]))


def main():
    T = tables()
    cfgs = configs()
    print(f"  {len(cfgs)} tuberías por tabla, {len(T)} tablas")
    S = {}
    for name, (X, y, _) in T.items():
        print(f"  midiendo {name} ({X.shape[0]} filas, {X.shape[1]} columnas)")
        S[name] = cached(f"scores-{name}", lambda X=X, y=y, name=name:
                         scores_matrix(name, X, y))

    cold = "reuters"
    warm = [n for n in T if n != cold]

    per = {}
    for name, sc in S.items():
        cv, te = sc["cv"], sc["test"]
        d = DEFAULT
        di = cfgs.index(d)
        bi = int(np.argmax(cv))
        yte = sc["yte"]
        okd = (np.argmax(sc["tst"][di], 1) == yte).astype(float)
        okb = (np.argmax(sc["tst"][bi], 1) == yte).astype(float)
        per[name] = dict(
            n_train=sc["n_train"], n_test=sc["n_test"], classes=sc["classes"],
            features=sc["features"],
            default=dict(config="|".join(d), cv=float(cv[di]), test=float(te[di]),
                         test_sd=boot_sd(okd)),
            best=dict(config="|".join(cfgs[bi]), cv=float(cv[bi]), test=float(te[bi]),
                      test_sd=boot_sd(okb)),
            best_test=dict(config="|".join(cfgs[int(np.argmax(te))]),
                           test=float(te.max())),
            span=dict(cv_lo=float(cv.min()), cv_hi=float(cv.max()),
                      test_lo=float(te.min()), test_hi=float(te.max())),
            axes=axis_moves(cfgs, cv),
            curse=curse(cv, te),
            cv=cv.tolist(), test=te.tolist(),
        )

    print("  la cartera, aprendida sobre las tres y probada en la cuarta")
    chosen_cfgs, chosen = portfolio([S[n]["cv"] for n in warm], cfgs)
    pf = portfolio_vs_random(S[cold], chosen)

    print("  las mezclas")
    ens = {name: cached(f"ensemble-{name}", lambda sc=S[name]: ensemble(sc))
           for name in S}

    print("  la validación cruzada anidada, sobre la tabla pequeña")
    nest = cached(f"nested-wine-o{NESTED_OUTER}",
                  lambda: nested("wine", *T["wine"][:2]))

    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(
        meta=dict(seed=SEED, folds=FOLDS, test_frac=TEST_FRAC,
                  configs=len(cfgs), scalers=list(SCALERS), reducers=list(REDUCERS),
                  weights=list(WEIGHTS), models=[m for m, _ in MODELS], trees=TREES,
                  boost_iters=BOOST_ITERS, digits_n=DIGITS_N,
                  default="|".join(DEFAULT), boot=BOOT,
                  cold=cold, warm=warm, r8_docs=R8_DOCS, r8_dims=R8_DIMS,
                  r8_vocab=int(T["reuters"][2]),
                  note="accuracy everywhere; the cross validated column is what "
                       "a search sees and the held out one is what nobody sees"),
        tables={k: r(v) for k, v in per.items()},
        configs=["|".join(c) for c in cfgs],
        portfolio=dict(size=PORTFOLIO, cold=cold, learnt_on=warm,
                       picks=["|".join(c) for c in chosen_cfgs],
                       rows=r(pf)),
        ensemble={k: dict(rows=r(v["rows"]), picks=v["picks"][:ENSEMBLE_MAX])
                  for k, v in ens.items()},
        nested=r(nest),
    )
    path = OUT / "automl.json"
    path.write_text(json.dumps(out, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
