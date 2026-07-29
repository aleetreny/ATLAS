"""Las cifras del artículo de validación cruzada.

Cada artículo del atlas cita una cifra retenida y sigue. Esta página pregunta de
qué es estimación esa cifra, cuánto se mueve, y en qué condiciones deja de
estimar lo que dice. Se puede contestar de verdad porque en la primera sección
**hay una verdad**: los datos salen de un proceso escrito, así que el riesgo de
un modelo se calcula sobre un millón de filas frescas y la validación cruzada se
puntúa contra él en vez de contra otra validación cruzada.

Lo que se mide:

  1. **Sesgo y varianza de k**, contra el riesgo verdadero, con k = 2, 5, 10 y
     dejar uno fuera. Y las dos cosas por separado, porque van en direcciones
     distintas y "k = 10 es lo estándar" no dice cuál de las dos se está
     comprando.
  2. **La varianza de una validación cruzada no es la varianza de sus
     pliegues.** La dispersión entre pliegues dividida por raíz de k es lo que
     todo el mundo imprime, y aquí se puede comparar con la dispersión real del
     estimador entre conjuntos de datos distintos, que es lo que esa barra de
     error dice ser.
  3. **La fuga por preprocesado**, con datos donde no hay absolutamente nada que
     aprender: etiquetas al azar y cinco mil columnas de ruido. Elegir columnas
     mirando todas las filas y luego validar da una exactitud alta sobre nada.
  4. **La fuga por el tiempo**, sobre la misma serie de CO2 que publica el
     artículo 51 (mismo digest afirmado antes de correr), con el control que
     aísla el orden: barajar los mismos pares.
  5. **Y la fuga por solape**, que TimeSeriesSplit no arregla: con un rasgo de
     media móvil de anchura w, las filas de los dos lados del corte comparten
     observaciones. Se barre el margen de purga y se mide dónde deja de haber
     optimismo.
  6. **Los grupos que nadie declaró.** Reuters tiene noticias casi idénticas
     (la misma historia archivada dos veces). Se cuentan, se agrupan y se mide
     lo que cuesta partir por fila en vez de por grupo.
  7. **Los pliegues sin positivos**, contados y comparados con su fórmula
     hipergeométrica exacta.

Ejecutar desde cualquier sitio: `python src/utils/generate_crossval_data.py`.
"""
import json
import os
import pickle
import sys
from math import comb
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlp_data import reuters, tokenize  # noqa: E402
from timeseries_data import co2_monthly, digest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "crossval" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "crossval_cache")

SEED = 19
CO2_DIGEST = "22fe4dedba02bb9f"

TRUTH_N = 200             # filas por conjunto de entrenamiento
TRUTH_D = 8               # columnas del proceso escrito
TRUTH_SETS = 500          # conjuntos de datos independientes
TRUTH_POP = 200000        # filas frescas con las que se calcula el riesgo
KS = (2, 5, 10, 20, TRUTH_N)
REPEATS = (1, 2, 5, 10, 20)

NOISE_N = 100
NOISE_P = 5000
NOISE_KEEP = 20
NOISE_SEEDS = 200

LAGS = 12
MA_WIDTH = 12
EMBARGOES = (0, 1, 3, 6, 12, 18, 24)

R8 = ("acq", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade")
DUP_DIMS = 60
DUP_THRESH = 0.92
DUP_SEEDS = 40

TAG = (f"n{TRUTH_N}d{TRUTH_D}s{TRUTH_SETS}p{TRUTH_POP}"
       f"x{NOISE_P}k{NOISE_KEEP}e{NOISE_SEEDS}l{LAGS}m{MA_WIDTH}"
       f"t{DUP_THRESH}v{SEED}")


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


# ------------------------------------------------------- 1 y 2. contra la verdad
def truth_process(rng, n, beta):
    X = rng.standard_normal((n, len(beta)))
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    y = (rng.random(n) < p).astype(int)
    return X, y


def fit(X, y):
    return LogisticRegression(max_iter=1000, C=1.0).fit(X, y)


def kfold_scores(X, y, k, rng):
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    out = []
    for f in folds:
        tr = np.setdiff1d(idx, f)
        if len(np.unique(y[tr])) < 2:
            continue
        m = fit(X[tr], y[tr])
        out.append(float(np.mean(m.predict(X[f]) == y[f])))
    return np.array(out)


def stage_truth():
    """Sesgo y varianza de k contra el riesgo verdadero.

    Dos verdades hacen falta y son distintas: el riesgo del modelo ajustado a
    las n filas de ESTE conjunto (que es lo que el usuario tiene en la mano) y
    el riesgo esperado de ajustar a n filas cualesquiera (que es lo que la
    validación cruzada estima de verdad). La página publica las dos porque la
    diferencia entre ellas es media discusión del tema.
    """
    rng = np.random.default_rng(SEED)
    beta = rng.standard_normal(TRUTH_D) * 0.8
    Xp, yp = truth_process(np.random.default_rng(SEED + 1), TRUTH_POP, beta)
    bayes = float(np.mean((Xp @ beta > 0).astype(int) == yp))
    rows = {k: [] for k in KS}
    spread = {k: [] for k in KS}
    true_this = []
    for s in range(TRUTH_SETS):
        g = np.random.default_rng(SEED + 100 + s)
        X, y = truth_process(g, TRUTH_N, beta)
        m = fit(X, y)
        true_this.append(float(np.mean(m.predict(Xp) == yp)))
        for k in KS:
            sc = kfold_scores(X, y, k, np.random.default_rng(SEED + 5000 + s))
            rows[k].append(float(sc.mean()))
            spread[k].append(float(sc.std(ddof=1) / np.sqrt(len(sc))) if len(sc) > 1
                             else 0.0)
        if (s + 1) % 100 == 0:
            print(f"    {s + 1}/{TRUTH_SETS} conjuntos")
    exp_risk = float(np.mean(true_this))
    out = []
    for k in KS:
        v = np.array(rows[k])
        out.append(dict(k=int(k), mean=float(v.mean()),
                        sd_true=float(v.std(ddof=1)),
                        sd_claimed=float(np.mean(spread[k])),
                        bias=float(v.mean() - exp_risk),
                        n_train=int(TRUTH_N - TRUTH_N // k),
                        ratio=float(v.std(ddof=1) / max(np.mean(spread[k]), 1e-12))))
    # repetir la partición: qué parte del ruido se puede quitar y cuál no
    rep = []
    for R in REPEATS:
        vals = []
        for s in range(150):
            g = np.random.default_rng(SEED + 100 + s)
            X, y = truth_process(g, TRUTH_N, beta)
            e = [kfold_scores(X, y, 10, np.random.default_rng(SEED + 9000 + 31 * s + j)).mean()
                 for j in range(R)]
            vals.append(float(np.mean(e)))
        rep.append(dict(repeats=R, sd=float(np.std(vals, ddof=1))))
        print(f"    repeticiones {R}: sd {rep[-1]['sd']:.5f}")
    # la parte del ruido que es del propio conjunto de datos, aislada
    g = np.random.default_rng(SEED + 100)
    Xf, yf = truth_process(g, TRUTH_N, beta)
    within = [float(kfold_scores(Xf, yf, 10, np.random.default_rng(SEED + 777 + j)).mean())
              for j in range(200)]
    return dict(rows=out, repeats=rep, bayes=bayes, expected_risk=exp_risk,
                true_sd=float(np.std(true_this, ddof=1)),
                partition_sd=float(np.std(within, ddof=1)),
                sets=TRUTH_SETS, n=TRUTH_N, d=TRUTH_D, pop=TRUTH_POP)


# ------------------------------------------------------------ 3. fuga de columnas
def stage_noise():
    """Cinco mil columnas de ruido y etiquetas a cara o cruz."""
    inside, outside, full = [], [], []
    for s in range(NOISE_SEEDS):
        rng = np.random.default_rng(SEED + s)
        X = rng.standard_normal((NOISE_N, NOISE_P))
        y = (rng.random(NOISE_N) < 0.5).astype(int)
        c = np.abs(X[y == 1].mean(0) - X[y == 0].mean(0))
        keep = np.argsort(c)[::-1][:NOISE_KEEP]
        idx = rng.permutation(NOISE_N)
        folds = np.array_split(idx, 10)
        out_acc, in_acc = [], []
        for f in folds:
            tr = np.setdiff1d(idx, f)
            m = fit(X[tr][:, keep], y[tr])
            out_acc.append(float(np.mean(m.predict(X[f][:, keep]) == y[f])))
            ci = np.abs(X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0))
            k2 = np.argsort(ci)[::-1][:NOISE_KEEP]
            m2 = fit(X[tr][:, k2], y[tr])
            in_acc.append(float(np.mean(m2.predict(X[f][:, k2]) == y[f])))
        outside.append(float(np.mean(out_acc)))
        inside.append(float(np.mean(in_acc)))
        full.append(float(fit(X[:, keep], y).score(X[:, keep], y)))
    return dict(seeds=NOISE_SEEDS, n=NOISE_N, p=NOISE_P, keep=NOISE_KEEP,
                outside=float(np.mean(outside)), outside_sd=float(np.std(outside)),
                inside=float(np.mean(inside)), inside_sd=float(np.std(inside)),
                resub=float(np.mean(full)),
                outside_all=outside, inside_all=inside,
                chance=0.5)


# ---------------------------------------------------------------- 4 y 5. el tiempo
def lag_table(y, lags, ma=0):
    rows, target = [], []
    start = max(lags, ma)
    for t in range(start, len(y)):
        f = list(y[t - lags:t][::-1])
        if ma:
            f.append(float(np.mean(y[t - ma:t])))
        rows.append(f)
        target.append(y[t])
    return np.array(rows), np.array(target), start


def mae(m, X, y):
    return float(np.mean(np.abs(m.predict(X) - y)))


def stage_time():
    y, dates, meta = co2_monthly()
    got = digest(y)
    assert got == CO2_DIGEST, f"el CO2 cambió: {got} en vez de {CO2_DIGEST}"
    X, t, start = lag_table(y, LAGS)
    n = len(t)
    rng = np.random.default_rng(SEED)

    def random_kfold(X, t, k=5, seed=SEED):
        g = np.random.default_rng(seed)
        idx = g.permutation(len(t))
        folds = np.array_split(idx, k)
        errs = []
        for f in folds:
            tr = np.setdiff1d(idx, f)
            m = Ridge(alpha=1e-6).fit(X[tr], t[tr])
            errs.append(mae(m, X[f], t[f]))
        return float(np.mean(errs))

    def forward(X, t, k=5, embargo=0):
        cuts = np.linspace(len(t) // (k + 1), len(t), k + 1).astype(int)
        errs = []
        for i in range(k):
            lo, hi = cuts[i], cuts[i + 1]
            tr = np.arange(0, max(0, lo - embargo))
            if len(tr) < LAGS + 5:
                continue
            m = Ridge(alpha=1e-6).fit(X[tr], t[tr])
            errs.append(mae(m, X[lo:hi], t[lo:hi]))
        return float(np.mean(errs))

    # el control: los MISMOS pares, barajados, para aislar el orden
    perm = rng.permutation(n)
    shuffled = random_kfold(X[perm], t[perm])
    rows = [
        dict(arm="random k-fold", mae=random_kfold(X, t)),
        dict(arm="random k-fold, rows shuffled", mae=shuffled),
        dict(arm="forward chaining", mae=forward(X, t)),
        dict(arm="forward chaining, embargo 12", mae=forward(X, t, embargo=12)),
    ]
    # con un rasgo de media móvil, que hace que las filas compartan observaciones
    Xm, tm, _ = lag_table(y, LAGS, ma=MA_WIDTH)
    emb = []
    for e in EMBARGOES:
        emb.append(dict(embargo=e, mae=forward(Xm, tm, embargo=e)))
    emb_random = random_kfold(Xm, tm)
    # el horizonte: cuánto se degrada al predecir más lejos
    hz = []
    for h in (1, 3, 6, 12, 24):
        Xh, th = X[:-h] if h else X, t[h:] if h else t
        hz.append(dict(horizon=h, random=random_kfold(Xh, th),
                       forward=forward(Xh, th)))
    return dict(rows=rows, embargo=emb, embargo_random=emb_random,
                horizon=hz, n=int(n), lags=LAGS, ma=MA_WIDTH,
                series=meta, digest=got,
                scale=float(np.mean(np.abs(np.diff(y)))))


# --------------------------------------------------------------- 6. los grupos
def stage_groups():
    """Las noticias casi idénticas de Reuters, contadas y usadas como grupos."""
    docs, cats, split = reuters()
    keep = [i for i, c in enumerate(cats) if len(c) == 1 and c[0] in R8]
    texts = [tokenize(docs[i]) for i in keep]
    y = np.array([R8.index(cats[i][0]) for i in keep])
    vocab = {}
    for t in texts:
        for w in set(t):
            vocab[w] = vocab.get(w, 0) + 1
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
    Z = (U[:, :DUP_DIMS] * S[:DUP_DIMS]).astype(np.float64)
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

    # parejas casi idénticas, por bloques para no materializar la matriz entera
    parent = np.arange(len(Zn))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = 0
    B = 512
    for i in range(0, len(Zn), B):
        sim = Zn[i:i + B] @ Zn.T
        for a in range(sim.shape[0]):
            gi = i + a
            hits = np.where(sim[a] > DUP_THRESH)[0]
            for gj in hits:
                if gj <= gi:
                    continue
                pairs += 1
                ra, rb = find(gi), find(int(gj))
                if ra != rb:
                    parent[rb] = ra
    groups = np.array([find(i) for i in range(len(Zn))])
    _, gid = np.unique(groups, return_inverse=True)
    sizes = np.bincount(gid)

    def cv(by_group, seed):
        g = np.random.default_rng(seed)
        if by_group:
            order = g.permutation(sizes.shape[0])
            chunks = np.array_split(order, 5)
            folds = [np.where(np.isin(gid, c))[0] for c in chunks]
        else:
            order = g.permutation(len(y))
            folds = np.array_split(order, 5)
        accs = []
        for f in folds:
            tr = np.setdiff1d(np.arange(len(y)), f)
            m = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr])
            accs.append(float(np.mean(m.predict(Z[f]) == y[f])))
        return float(np.mean(accs))

    rowwise = [cv(False, SEED + s) for s in range(DUP_SEEDS // 8)]
    groupwise = [cv(True, SEED + s) for s in range(DUP_SEEDS // 8)]
    return dict(docs=int(len(y)), vocab=int(len(words)), dims=DUP_DIMS,
                threshold=DUP_THRESH, pairs=int(pairs),
                groups=int(sizes.shape[0]),
                in_groups=int(sizes[sizes > 1].sum()),
                multi_groups=int((sizes > 1).sum()),
                biggest=int(sizes.max()),
                by_row=float(np.mean(rowwise)), by_row_sd=float(np.std(rowwise)),
                by_group=float(np.mean(groupwise)),
                by_group_sd=float(np.std(groupwise)),
                seeds=DUP_SEEDS // 8)


# ------------------------------------------------------- 7. pliegues sin positivos
def empty_fold_prob(n, pos, k):
    """P(algún pliegue se queda sin un positivo), por inclusión y exclusión.

    Un pliegue tiene n/k filas y la probabilidad de que ninguno de los `pos`
    positivos caiga en él es un cociente de combinatorios. Con k pliegues los
    sucesos no son independientes, así que el primer término sobreestima y el
    segundo lo corrige; con estos tamaños los términos de tres pliegues ya son
    despreciables y se dice en la página.
    """
    m = n // k
    p1 = comb(n - m, pos) / comb(n, pos) if n - m >= pos else 0.0
    p2 = comb(n - 2 * m, pos) / comb(n, pos) if n - 2 * m >= pos else 0.0
    return min(1.0, k * p1 - comb(k, 2) * p2)


def stage_empty():
    rows = []
    for pos in (2, 3, 5, 8, 12, 20, 40):
        n, k = 500, 10
        exact = empty_fold_prob(n, pos, k)
        hit = 0
        trials = 20000
        rng = np.random.default_rng(SEED + pos)
        for _ in range(trials):
            lab = np.zeros(n, int)
            lab[rng.choice(n, pos, replace=False)] = 1
            folds = np.array_split(rng.permutation(n), k)
            if any(lab[f].sum() == 0 for f in folds):
                hit += 1
        rows.append(dict(positives=pos, n=n, k=k, exact=exact,
                         measured=hit / trials, trials=trials,
                         prevalence=pos / n))
    return rows


def main():
    print("  1/5 contra una verdad conocida")
    truth = cached("truth", stage_truth)
    print("  2/5 la fuga por elegir columnas fuera del pliegue")
    noise = cached("noise", stage_noise)
    print("  3/5 el tiempo")
    time_ = cached("time", stage_time)
    print("  4/5 los grupos que nadie declaró")
    groups = cached("groups", stage_groups)
    print("  5/5 los pliegues sin positivos")
    empty = cached("empty", stage_empty)

    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(
        meta=dict(seed=SEED, truth_sets=TRUTH_SETS, truth_n=TRUTH_N,
                  truth_d=TRUTH_D, pop=TRUTH_POP, ks=[int(k) for k in KS],
                  noise_p=NOISE_P, noise_n=NOISE_N, noise_keep=NOISE_KEEP,
                  co2_digest=CO2_DIGEST, lags=LAGS, ma=MA_WIDTH,
                  note="accuracy in the simulated sections, mean absolute error "
                       "in parts per million in the time section"),
        truth=r(truth),
        noise=r(noise),
        time=r(time_),
        groups=r(groups),
        empty=r(empty),
    )
    path = OUT / "crossval.json"
    path.write_text(json.dumps(out, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
