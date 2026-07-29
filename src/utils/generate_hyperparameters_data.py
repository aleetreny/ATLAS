"""Las cifras del artículo de búsqueda de hiperparámetros.

Todos los artículos anteriores eligen hiperparámetros y ninguno enseña cómo. Se
dice "validación cruzada sobre una rejilla" y se pasa página. Este mide lo que
esa frase cuesta y lo que compra, y puede hacerlo por una razón que casi ningún
artículo sobre el tema tiene: **la superficie entera está calculada**.

Sobre los 1.797 dígitos de scikit-learn (los mismos que usa el artículo de
conformal prediction), una SVM con núcleo gaussiano tiene dos mandos, C y gamma.
Se enumeran 31 x 31 = 961 combinaciones y cada una se puntúa dos veces: por
validación cruzada sobre el trozo de entrenamiento (que es lo que un buscador
ve) y sobre el trozo retenido (que es lo que nadie ve). Con eso, cualquier
búsqueda se puntúa contra **la respuesta**, no contra otra búsqueda.

Lo que se mide:

  1. **La superficie, entera.** Con las dos puntuaciones por celda, y el suelo
     de ruido de cada una medido aparte: repetir la validación cruzada con otro
     sorteo de pliegues sobre la MISMA celda dice cuánto de una diferencia es
     el modelo y cuánto el sorteo.
  2. **Rejilla contra azar al mismo presupuesto.** La afirmación de Bergstra y
     Bengio no es que el azar sea mejor: es que una rejilla de b puntos solo
     prueba raíz de b valores distintos por eje, así que desperdicia el
     presupuesto cuando un eje no importa. Aquí eso se puede medir en vez de
     citarse, porque la importancia de cada eje sale de la superficie.
  3. **Y la falsación**: la misma comparación sobre una superficie donde los
     dos ejes sí importan (un bosque, profundidad contra número de rasgos), con
     la importancia de cada eje medida igual. Si el mecanismo es el que dice el
     párrafo, la ventaja del azar tiene que encogerse ahí.
  4. **Búsqueda bayesiana.** Un proceso gaussiano sobre la rejilla, con mejora
     esperada, escrito aquí (no hay librería de por medio) y puntuado contra el
     azar sobre las mismas semillas. Con el detalle que decide si sirve: el
     modelo sustituto tiene que saber que la observación trae ruido, porque la
     validación cruzada lo trae.
  5. **Detenerse pronto.** Halving supone que el orden de los candidatos a
     presupuesto pequeño predice el orden a presupuesto grande. Eso es una
     correlación y se mide, presupuesto a presupuesto, sobre los mismos
     candidatos.
  6. **La maldición del ganador.** El mejor de b puntuaciones de validación
     está sesgado hacia arriba, y con las dos puntuaciones de las 961 celdas el
     sesgo se calcula exactamente en función de b. Es el resultado que enlaza
     con el artículo de validación cruzada y con el de AutoML.

Ejecutar desde cualquier sitio: `python src/utils/generate_hyperparameters_data.py`.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "hyperparameters" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "hyperparameters_cache")

SEED = 19
GRID = 31                 # 31 x 31 = 961 celdas enumeradas
FOLDS = 5
REPEATS = 3               # repeticiones de la validación cruzada por celda
FLOOR_REPEATS = 12        # repeticiones extra, solo para medir el suelo de ruido
N_TRAIN = 1000            # de los 1.797 dígitos; el resto se retiene
SEARCH_SEEDS = 400
BUDGETS = (4, 9, 16, 25, 36, 49, 64)
GP_SEEDS = 120
GP_INIT = 5
GP_BUDGET = 30

# El bosque de la falsación: los dos ejes de esta superficie sí importan.
FOREST_DEPTHS = tuple(range(1, 16))
FOREST_FEATS = tuple(range(1, 16))

TAG = (f"g{GRID}f{FOLDS}r{REPEATS}n{N_TRAIN}s{SEED}"
       f"e{SEARCH_SEEDS}gp{GP_SEEDS}x{GP_BUDGET}")


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

    `json.dumps` escribe `NaN` sin protestar y eso no es JSON: el `fetch` del
    navegador revienta al parsearlo y la página se queda sin prosa y sin
    guardia, sin un solo error en el sitio donde está el fallo. El volcado del
    final lleva además `allow_nan=False`.
    """
    if isinstance(v, dict):
        return {k: r(x, d) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [r(x, d) for x in v]
    if v is None or isinstance(v, (bool, str)):
        return v
    v = float(v)
    return None if not np.isfinite(v) else round(v, d)


def data():
    d = load_digits()
    X, y = d.data / 16.0, d.target
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, train_size=N_TRAIN, random_state=SEED, stratify=y)
    return Xtr, ytr, Xte, yte


def cv_score(model_fn, X, y, folds, seed):
    """Exactitud media sobre una partición estratificada, con la semilla dada."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    acc = []
    for tr, va in skf.split(X, y):
        m = model_fn()
        m.fit(X[tr], y[tr])
        acc.append(float(m.score(X[va], y[va])))
    return float(np.mean(acc))


# ---------------------------------------------------------------- 1. superficie
def stage_surface(X, y, Xte, yte):
    """Las 961 celdas, cada una con su cifra de validación y su cifra retenida.

    La segunda es la que ningún buscador puede ver y la única contra la que
    tiene sentido puntuar a un buscador: comparar dos búsquedas por la
    puntuación de validación que ellas mismas maximizan es comparar quién
    explota mejor el ruido de los pliegues.
    """
    cs = np.linspace(-5, 10, GRID)          # log2 C
    gs = np.linspace(-13, 2, GRID)          # log2 gamma
    cv = np.zeros((GRID, GRID))
    te = np.zeros((GRID, GRID))
    for i, lc in enumerate(cs):
        for j, lg in enumerate(gs):
            mk = (lambda lc=lc, lg=lg: SVC(C=2.0 ** lc, gamma=2.0 ** lg))
            cv[i, j] = np.mean([cv_score(mk, X, y, FOLDS, SEED + 100 * k)
                                for k in range(REPEATS)])
            m = mk()
            m.fit(X, y)
            te[i, j] = float(m.score(Xte, yte))
        print(f"    fila {i + 1}/{GRID} de la superficie")
    return dict(c=cs.tolist(), g=gs.tolist(), cv=cv.tolist(), test=te.tolist())


def stage_surface2(X, y):
    """La MISMA rejilla, puntuada otra vez con otros pliegues.

    Hace falta para separar dos sesgos que se suman en la comparación obvia. El
    mejor de b candidatos está sesgado hacia arriba **respecto a su propia
    puntuación de validación**, y eso es la maldición del ganador; pero la
    validación cruzada entrena con menos filas que el ajuste final, y eso la
    sesga hacia abajo. Restar la puntuación retenida mezcla los dos. Restar una
    segunda validación cruzada independiente deja solo el primero.
    """
    cs = np.linspace(-5, 10, GRID)
    gs = np.linspace(-13, 2, GRID)
    cv = np.zeros((GRID, GRID))
    for i, lc in enumerate(cs):
        for j, lg in enumerate(gs):
            mk = (lambda lc=lc, lg=lg: SVC(C=2.0 ** lc, gamma=2.0 ** lg))
            cv[i, j] = np.mean([cv_score(mk, X, y, FOLDS, SEED + 9973 + 100 * k)
                                for k in range(REPEATS)])
        print(f"    fila {i + 1}/{GRID} de la segunda superficie")
    return cv.tolist()


def stage_floor(X, y, Xte, yte, surf):
    """Cuánto se mueve una puntuación sin que cambie el modelo.

    Dos suelos distintos y los dos hacen falta: el de la validación cruzada
    (sortear otros pliegues) y el del conjunto retenido (remuestrear el propio
    test), porque la página compara cifras de los dos tipos.
    """
    cv = np.array(surf["cv"])
    order = np.argsort(cv.ravel())
    # tres celdas repartidas por la superficie: la mejor, la mediana y una mala
    picks = [order[-1], order[len(order) // 2], order[len(order) // 6]]
    rows = []
    for flat in picks:
        i, j = divmod(int(flat), GRID)
        lc, lg = surf["c"][i], surf["g"][j]
        mk = (lambda: SVC(C=2.0 ** lc, gamma=2.0 ** lg))
        reps = [cv_score(mk, X, y, FOLDS, SEED + 7919 * k)
                for k in range(FLOOR_REPEATS)]
        m = mk()
        m.fit(X, y)
        pred = m.predict(Xte)
        rng = np.random.default_rng(SEED + flat)
        boot = [float(np.mean(pred[k] == yte[k]))
                for k in (rng.integers(0, len(yte), len(yte)) for _ in range(400))]
        rows.append(dict(c=lc, g=lg, cv=float(np.mean(reps)),
                         cv_sd=float(np.std(reps)),
                         cv_lo=float(min(reps)), cv_hi=float(max(reps)),
                         test=float(np.mean(pred == yte)),
                         test_sd=float(np.std(boot))))
    return dict(rows=rows, repeats=FLOOR_REPEATS,
                cv_sd=float(np.mean([x["cv_sd"] for x in rows])),
                test_sd=float(np.mean([x["test_sd"] for x in rows])))


# ------------------------------------------------- 2 y 3. rejilla contra azar
def axis_importance(surface):
    """Cuánto mueve cada eje la mejor puntuación alcanzable.

    Es la cantidad de la que habla el argumento de la rejilla: si al fijar un
    eje en su peor valor la mejor puntuación no se mueve, ese eje no importa y
    todas las columnas de la rejilla que se gastan en él son evaluaciones
    tiradas.
    """
    a = np.array(surface)
    best_by_row = a.max(axis=1)     # lo mejor alcanzable fijando el eje 0
    best_by_col = a.max(axis=0)
    return dict(axis0=float(best_by_row.max() - best_by_row.min()),
                axis1=float(best_by_col.max() - best_by_col.min()),
                best=float(a.max()))


def grid_points(n, size):
    """Los índices de una rejilla de n puntos repartida sobre `size` valores."""
    k = int(round(np.sqrt(n)))
    idx = np.unique(np.round(np.linspace(0, size - 1, k)).astype(int))
    return [(i, j) for i in idx for j in idx]


def search_compare(cv, test, seeds, budgets):
    """Rejilla contra azar, puntuando siempre sobre la cifra retenida."""
    G = cv.shape[0]
    rows = []
    for b in budgets:
        pts = grid_points(b, G)
        gi = np.array([p[0] for p in pts])
        gj = np.array([p[1] for p in pts])
        k = int(np.argmax(cv[gi, gj]))
        grid_cv, grid_test = float(cv[gi[k], gj[k]]), float(test[gi[k], gj[k]])
        rc, rt = [], []
        rng = np.random.default_rng(SEED + b)
        for _ in range(seeds):
            ii = rng.integers(0, G, len(pts))
            jj = rng.integers(0, G, len(pts))
            w = int(np.argmax(cv[ii, jj]))
            rc.append(float(cv[ii[w], jj[w]]))
            rt.append(float(test[ii[w], jj[w]]))
        rows.append(dict(budget=len(pts), asked=b,
                         grid_cv=grid_cv, grid_test=grid_test,
                         rand_cv=float(np.mean(rc)), rand_test=float(np.mean(rt)),
                         rand_test_sd=float(np.std(rt)),
                         rand_beats=float(np.mean(np.array(rt) > grid_test)),
                         distinct_grid=int(round(np.sqrt(len(pts)))),
                         distinct_rand=len(pts)))
    return rows


def flatten_axis(cv, test, axis):
    """La misma superficie con un eje muerto, hecha de sus propias cifras.

    Se sustituye cada celda por el máximo de su fila (o de su columna), así que
    el eje aplanado deja de mover nada y el otro conserva exactamente la forma
    medida. No es una simulación: son las mismas 961 puntuaciones reordenadas,
    y sirve para aislar el mecanismo del que habla el argumento de la rejilla
    sin cambiar de problema ni de datos.
    """
    if axis == 0:
        return (np.tile(cv.max(0), (cv.shape[0], 1)),
                np.tile(test[np.argmax(cv, 0), np.arange(cv.shape[1])],
                        (cv.shape[0], 1)))
    return (np.tile(cv.max(1)[:, None], (1, cv.shape[1])),
            np.tile(test[np.arange(cv.shape[0]), np.argmax(cv, 1)][:, None],
                    (1, cv.shape[1])))


def stage_forest(X, y, Xte, yte):
    """La superficie de la falsación: profundidad contra rasgos por corte."""
    cv = np.zeros((len(FOREST_DEPTHS), len(FOREST_FEATS)))
    te = np.zeros_like(cv)
    for i, d in enumerate(FOREST_DEPTHS):
        for j, f in enumerate(FOREST_FEATS):
            mk = (lambda d=d, f=f: RandomForestClassifier(
                n_estimators=60, max_depth=d, max_features=f,
                random_state=SEED, n_jobs=1))
            cv[i, j] = np.mean([cv_score(mk, X, y, FOLDS, SEED + 100 * k)
                                for k in range(REPEATS)])
            m = mk()
            m.fit(X, y)
            te[i, j] = float(m.score(Xte, yte))
        print(f"    fila {i + 1}/{len(FOREST_DEPTHS)} del bosque")
    return dict(depth=list(FOREST_DEPTHS), feats=list(FOREST_FEATS),
                cv=cv.tolist(), test=te.tolist())


# --------------------------------------------------------- 4. proceso gaussiano
def gp_search(cv, test, budget, init, seed, noise, length=3.0):
    """Mejora esperada sobre una rejilla, con el proceso gaussiano escrito aquí.

    La rejilla es finita, así que no hace falta optimizar la adquisición: se
    evalúa en las 961 celdas y se coge el máximo. `noise` es la varianza que el
    sustituto le supone a la observación, y es el mando que decide si el
    buscador persigue el ruido de los pliegues o lo ignora.
    """
    G = cv.shape[0]
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    Z = np.stack([ii.ravel(), jj.ravel()], 1).astype(float)
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    K = np.exp(-0.5 * d2 / length ** 2)
    rng = np.random.default_rng(seed)
    seen = list(rng.choice(len(Z), init, replace=False))
    flat_cv, flat_te = cv.ravel(), test.ravel()
    trace = [float(flat_cv[seen].max())]
    for _ in range(budget - init):
        s = np.array(seen)
        Kss = K[np.ix_(s, s)] + noise * np.eye(len(s))
        L = np.linalg.cholesky(Kss)
        yv = flat_cv[s]
        mu0 = float(yv.mean())
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yv - mu0))
        Ks = K[:, s]
        mu = mu0 + Ks @ alpha
        v = np.linalg.solve(L, Ks.T)
        var = np.clip(1.0 - (v ** 2).sum(0), 1e-12, None)
        sd = np.sqrt(var)
        best = float(yv.max())
        z = (mu - best) / sd
        # mejora esperada, con la normal escrita a mano para no arrastrar scipy
        cdf = 0.5 * (1.0 + np.vectorize(_erf)(z / np.sqrt(2.0)))
        pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        ei = (mu - best) * cdf + sd * pdf
        ei[s] = -1.0
        seen.append(int(np.argmax(ei)))
        trace.append(float(flat_cv[np.array(seen)].max()))
    s = np.array(seen)
    pick = int(s[np.argmax(flat_cv[s])])
    return dict(trace=trace, cv=float(flat_cv[pick]), test=float(flat_te[pick]))


def _erf(x):
    """Abramowitz y Stegun 7.1.26, que basta de sobra para una adquisición."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y


def _hits(traces, target):
    """Evaluaciones hasta alcanzar el objetivo, y cuántas tiradas no llegan.

    El valor final de una búsqueda es una medida ciega cuando casi cualquier
    tirada acaba cerca del óptimo: lo que separa a dos buscadores es en cuántas
    evaluaciones llegan, y eso hay que contarlo con las que no llegan aparte, o
    la media es una media sobre tiradas censuradas.
    """
    n, done = [], 0
    for tr in traces:
        w = np.where(np.array(tr) >= target)[0]
        if len(w):
            n.append(int(w[0]) + 1)
            done += 1
    return dict(median=float(np.median(n)) if n else None,
                mean=float(np.mean(n)) if n else None,
                reached=done / len(traces), runs=len(traces))


def stage_gp(cv, test, floor_sd):
    """El bayesiano contra el azar, y contra sí mismo con y sin ruido supuesto."""
    flat_cv, flat_te = cv.ravel(), test.ravel()
    target = float(flat_cv.max()) - floor_sd
    arms = {}
    for tag, noise in (("noisy", max(floor_sd ** 2, 1e-6)), ("noiseless", 1e-8)):
        traces, tests, cvs = [], [], []
        for s in range(GP_SEEDS):
            out = gp_search(cv, test, GP_BUDGET, GP_INIT, SEED + s, noise)
            traces.append(out["trace"])
            tests.append(out["test"])
            cvs.append(out["cv"])
        arms[tag] = dict(trace=np.mean(traces, 0).tolist(),
                         test=float(np.mean(tests)), test_sd=float(np.std(tests)),
                         cv=float(np.mean(cvs)), noise=float(noise),
                         hits=_hits(traces, target))
    rand_tr, rand_te, rand_cv = [], [], []
    for s in range(GP_SEEDS):
        rng = np.random.default_rng(SEED + 5000 + s)
        pick = rng.choice(len(flat_cv), GP_BUDGET, replace=False)
        run = np.maximum.accumulate(flat_cv[pick])
        rand_tr.append(run.tolist())
        w = pick[int(np.argmax(flat_cv[pick]))]
        rand_te.append(float(flat_te[w]))
        rand_cv.append(float(flat_cv[w]))
    arms["random"] = dict(trace=np.mean(rand_tr, 0).tolist(),
                          test=float(np.mean(rand_te)), test_sd=float(np.std(rand_te)),
                          cv=float(np.mean(rand_cv)), noise=None,
                          hits=_hits(rand_tr, target))
    # cuántas celdas de las 961 valen ya lo mismo que la mejor: si son muchas,
    # ningún buscador puede lucirse, y esa es la explicación y no una excusa
    near = float(np.mean(flat_cv >= target))
    return dict(arms=arms, budget=GP_BUDGET, init=GP_INIT, seeds=GP_SEEDS,
                target=target, near_optimal=near,
                expected_draws=float(1.0 / near) if near > 0 else None,
                best_cv=float(flat_cv.max()), best_test=float(flat_te.max()),
                oracle_test=float(flat_te[int(np.argmax(flat_cv))]))


# ------------------------------------------------------------- 5. detenerse pronto
def stage_halving(X, y, Xte, yte, surf):
    """El supuesto de halving, medido: ¿el orden a poco presupuesto es el orden?

    El presupuesto aquí es cuántas filas de entrenamiento ve el candidato, que
    es el que usa halving por defecto cuando el modelo no tiene iteraciones.
    """
    cv = np.array(surf["cv"])
    order = np.argsort(cv.ravel())[::-1]
    # 24 candidatos repartidos por la superficie: los mejores no bastan, porque
    # halving tiene que distinguir entre un candidato bueno y uno mediocre
    picks = list(order[:8]) + list(order[len(order) // 3:len(order) // 3 + 8]) \
        + list(order[len(order) // 2:len(order) // 2 + 8])
    cfgs = []
    for flat in picks:
        i, j = divmod(int(flat), GRID)
        cfgs.append((surf["c"][i], surf["g"][j]))
    fracs = (0.125, 0.25, 0.5, 1.0)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    scores = np.zeros((len(fracs), len(cfgs)))
    for fi, f in enumerate(fracs):
        n = int(len(y) * f)
        sub = idx[:n]
        for ci, (lc, lg) in enumerate(cfgs):
            mk = (lambda lc=lc, lg=lg: SVC(C=2.0 ** lc, gamma=2.0 ** lg))
            scores[fi, ci] = cv_score(mk, X[sub], y[sub], FOLDS, SEED)
    full = scores[-1]
    rows = []
    for fi, f in enumerate(fracs):
        rows.append(dict(frac=f, n=int(len(y) * f),
                         spearman=float(_spearman(scores[fi], full)),
                         top1_kept=bool(int(np.argmax(scores[fi])) == int(np.argmax(full))),
                         top_half_kept=float(_overlap(scores[fi], full, len(cfgs) // 2)),
                         mean=float(scores[fi].mean())))
    # el ahorro exacto de halving, contado en filas por candidato
    survivors, cost_halving = list(range(len(cfgs))), 0
    for fi, f in enumerate(fracs):
        cost_halving += len(survivors) * f
        if fi < len(fracs) - 1:
            keep = max(1, len(survivors) // 2)
            sc = scores[fi][survivors]
            survivors = [survivors[k] for k in np.argsort(sc)[::-1][:keep]]
    return dict(fracs=list(fracs), configs=len(cfgs), rows=rows,
                scores=scores.tolist(),
                halving_pick=int(survivors[0]),
                halving_is_best=bool(survivors[0] == int(np.argmax(full))),
                cost_halving=float(cost_halving), cost_full=float(len(cfgs)),
                saving=float(len(cfgs) / cost_halving))


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def _overlap(a, b, k):
    ta = set(np.argsort(a)[::-1][:k].tolist())
    tb = set(np.argsort(b)[::-1][:k].tolist())
    return len(ta & tb) / k


# ------------------------------------------------------ 6. la maldición del ganador
def stage_curse(cv, cv2, test, trials=4000):
    """El sesgo del máximo de b puntuaciones, calculado sobre las 961 celdas.

    No hay nada que entrenar: las tres puntuaciones de cada celda ya están, así
    que para cada b se sortean b celdas, se coge la mejor por la primera
    validación y se miran las otras dos. Contra la segunda validación sale la
    maldición del ganador sola; contra la retenida sale eso **más** el sesgo de
    entrenar con menos filas, que va en el otro sentido.
    """
    flat_cv, flat_te, flat_2 = cv.ravel(), test.ravel(), cv2.ravel()
    rows = []
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        rng = np.random.default_rng(SEED + b)
        sel = rng.integers(0, len(flat_cv), (trials, b))
        w = np.argmax(flat_cv[sel], axis=1)
        take = sel[np.arange(trials), w]
        rows.append(dict(b=b, cv=float(flat_cv[take].mean()),
                         cv2=float(flat_2[take].mean()),
                         test=float(flat_te[take].mean()),
                         selection=float((flat_cv[take] - flat_2[take]).mean()),
                         gap=float((flat_cv[take] - flat_te[take]).mean()),
                         gap_sd=float((flat_cv[take] - flat_te[take]).std())))
    honest = float((flat_cv - flat_te).mean())
    return dict(rows=rows, trials=trials, honest_gap=honest,
                honest_selection=float((flat_cv - flat_2).mean()),
                pair_sd=float((flat_cv - flat_2).std()),
                best_cv=float(flat_cv.max()), best_test=float(flat_te.max()),
                oracle=float(flat_te[int(np.argmax(flat_cv))]))


def main():
    Xtr, ytr, Xte, yte = data()
    print(f"  dígitos: {len(ytr)} de entrenamiento, {len(yte)} retenidos, "
          f"{Xtr.shape[1]} columnas")

    print("  1/6 la superficie entera")
    surf = cached("surface", lambda: stage_surface(Xtr, ytr, Xte, yte))
    # Se redondea AQUÍ, antes de medir nada sobre la superficie, y no al
    # exportar: el navegador solo va a tener cinco decimales, así que toda
    # cifra que la página vaya a recalcular (el mejor, la importancia de cada
    # eje, la meseta) tiene que salir de los mismos números que él tendrá. Con
    # el redondeo al final, el guardia del navegador dispara por 7e-6 sin que
    # haya ningún error, que es la forma más cara de tener un guardia.
    cv = np.round(np.array(surf["cv"]), 5)
    test = np.round(np.array(surf["test"]), 5)
    surf["cv"] = cv.tolist()
    surf["test"] = test.tolist()

    print("  2/6 el suelo de ruido de las dos puntuaciones")
    floor = cached("floor", lambda: stage_floor(Xtr, ytr, Xte, yte, surf))

    print("  3/6 el bosque de la falsación")
    forest = cached("forest", lambda: stage_forest(Xtr, ytr, Xte, yte))
    fcv = np.round(np.array(forest["cv"]), 5)
    ftest = np.round(np.array(forest["test"]), 5)
    forest["cv"] = fcv.tolist()
    forest["test"] = ftest.tolist()

    print("  4/6 el proceso gaussiano")
    gp = cached("gp2", lambda: stage_gp(cv, test, floor["cv_sd"]))

    print("  5/6 detenerse pronto")
    halving = cached("halving", lambda: stage_halving(Xtr, ytr, Xte, yte, surf))

    print("  6/6 la segunda superficie y la maldición del ganador")
    cv2 = np.round(np.array(cached("surface2", lambda: stage_surface2(Xtr, ytr))), 5)
    curse = stage_curse(cv, cv2, test)

    svm_rows = search_compare(cv, test, SEARCH_SEEDS, BUDGETS)
    forest_rows = search_compare(fcv, ftest, SEARCH_SEEDS,
                                 tuple(b for b in BUDGETS if b <= len(FOREST_DEPTHS) ** 2))
    flat_cv, flat_te = flatten_axis(cv, test, 0)
    flat_rows = search_compare(flat_cv, flat_te, SEARCH_SEEDS, BUDGETS)
    svm_imp = axis_importance(surf["cv"])
    forest_imp = axis_importance(forest["cv"])
    flat_imp = axis_importance(flat_cv.tolist())

    OUT.mkdir(parents=True, exist_ok=True)
    data_out = dict(
        meta=dict(seed=SEED, grid=GRID, folds=FOLDS, repeats=REPEATS,
                  n_train=int(len(ytr)), n_test=int(len(yte)),
                  features=int(Xtr.shape[1]), classes=int(len(set(ytr.tolist()))),
                  search_seeds=SEARCH_SEEDS,
                  dataset="the 1,797 handwritten digits that ship with "
                          "scikit-learn, the same set the conformal article uses",
                  note="every score on this page is an accuracy; the cross "
                       "validated one is what a search sees and the held out "
                       "one is what nobody sees"),
        surface=dict(c=r(surf["c"], 4), g=r(surf["g"], 4),
                     cv=[r(row, 5) for row in surf["cv"]],
                     test=[r(row, 5) for row in surf["test"]],
                     importance=r(svm_imp),
                     best=dict(cv=float(cv.max()), test=float(test.max()),
                               at_cv=[int(x) for x in np.unravel_index(int(np.argmax(cv)), cv.shape)],
                               at_test=[int(x) for x in np.unravel_index(int(np.argmax(test)), test.shape)])),
        floor=dict(repeats=floor["repeats"], cv_sd=r(floor["cv_sd"]),
                   test_sd=r(floor["test_sd"]),
                   rows=[{k: r(v) for k, v in row.items()} for row in floor["rows"]]),
        forest=dict(depth=forest["depth"], feats=forest["feats"],
                    cv=[r(row, 5) for row in forest["cv"]],
                    test=[r(row, 5) for row in forest["test"]],
                    importance=r(forest_imp),
                    trees=60),
        compare=dict(svm=[{k: r(v) for k, v in row.items()} for row in svm_rows],
                     forest=[{k: r(v) for k, v in row.items()} for row in forest_rows],
                     flattened=[{k: r(v) for k, v in row.items()} for row in flat_rows],
                     flat_importance=r(flat_imp),
                     flat_note="the same 961 measured scores with the C axis "
                               "replaced by its own column maximum, so that axis "
                               "moves nothing and the other keeps its shape"),
        gp=dict(budget=gp["budget"], init=gp["init"], seeds=gp["seeds"],
                best_cv=r(gp["best_cv"]), best_test=r(gp["best_test"]),
                oracle_test=r(gp["oracle_test"]),
                arms={k: {kk: (r(vv) if kk != "trace" else r(vv, 5))
                          for kk, vv in v.items()} for k, v in gp["arms"].items()}),
        halving=dict(fracs=r(halving["fracs"], 3), configs=halving["configs"],
                     rows=[{k: (v if isinstance(v, bool) else r(v))
                            for k, v in row.items()} for row in halving["rows"]],
                     scores=[r(row, 5) for row in halving["scores"]],
                     halving_pick=halving["halving_pick"],
                     halving_is_best=halving["halving_is_best"],
                     cost_halving=r(halving["cost_halving"]),
                     cost_full=r(halving["cost_full"]), saving=r(halving["saving"])),
        curse=dict(trials=curse["trials"], honest_gap=r(curse["honest_gap"]),
                   best_cv=r(curse["best_cv"]), best_test=r(curse["best_test"]),
                   oracle=r(curse["oracle"]),
                   rows=[{k: r(v) for k, v in row.items()} for row in curse["rows"]]),
    )
    path = OUT / "hyperparameters.json"
    path.write_text(json.dumps(data_out, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
