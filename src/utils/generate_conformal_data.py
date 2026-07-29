"""Las cifras del artículo de conformal prediction.

Casi todo el atlas mide **medias**: un error cuadrático, una exactitud, una
verosimilitud. Un error medio de 0,3 no dice qué hacer con la predicción de
mañana. Este artículo va de lo otro: un intervalo con una garantía escrita, sin
suponer nada del modelo ni de la distribución de los datos.

La garantía es rara entre las garantías del campo por dos motivos: vale para
**cualquier** modelo (incluido uno malo) y vale en **muestra finita** (no es un
resultado asintótico). A cambio pide una cosa sola, intercambiabilidad, y esa
cosa es exactamente lo que falla en la mitad de los problemas reales.

Lo que se mide:

  1. **La cobertura exacta, contra la fórmula.** No "aproximadamente 90%":
     ceil((n+1)(1-alpha))/(n+1) exactamente, y se comprueba a cuatro decimales
     sobre 20.000 tiradas. Como la garantía solo depende de rangos, las
     puntuaciones se sortean de una distribución continua conocida y la
     cobertura condicional de cada calibración se calcula **en forma cerrada**
     en vez de estimarse, así que la comparación no tiene ruido de muestreo.
  2. **Y la que casi nadie enseña: esa cobertura es una media sobre
     calibraciones.** Fijado el conjunto de calibración, la cobertura es una
     Beta(k, n+1-k), y con 100 puntos su desviación es del orden de tres
     puntos. La página dibuja la Beta exacta encima del histograma medido.
  3. **La validez es gratis y la anchura no.** Cuatro modelos sobre los mismos
     datos, del bueno al que contesta siempre lo mismo: la misma cobertura
     dentro del ruido, anchuras que se separan por un factor grande.
  4. **Marginal no es condicional.** Con dos grupos de dificultad distinta, el
     90% global esconde un grupo muy por encima y otro muy por debajo. Mondrian
     (calibrar por grupo) y CQR (calibrar sobre cuantiles) lo arreglan de dos
     maneras distintas, y las dos se miden.
  5. **La intercambiabilidad es la hipótesis, y se rompe sola.** Sobre la serie
     de CO2 que publica el artículo 51 (mismo digest afirmado antes de correr),
     partir por tiempo hunde la cobertura; barajar **los mismos pares** la
     devuelve. El control aísla el orden como causa, que es lo que convierte
     una anécdota en una medida.
  6. **Y en clasificación**, donde la predicción es un conjunto: dos
     puntuaciones clásicas, el tamaño medio del conjunto, los vacíos y la
     cobertura por clase, que es donde se ve otra vez la diferencia entre
     marginal y condicional.

Ejecutar desde cualquier sitio: `python src/utils/generate_conformal_data.py`.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timeseries_data import co2_monthly, digest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conformal" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "conformal_cache")

SEED = 19
ALPHA = 0.1
TRIALS = 20000
SPLITS = 200
N_DATA = 3000

# El digest del CO2 que publica el 51, afirmado antes de medir nada. Si otra
# versión de statsmodels cambiara un valor, esta línea lo dice en vez de
# publicar una comparación entre dos series distintas.
CO2_DIGEST = "22fe4dedba02bb9f"

TAG = f"a{ALPHA}t{TRIALS}s{SPLITS}n{N_DATA}d{SEED}"


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
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r(x, d) for x in np.asarray(v).tolist()]
    return round(float(v), d)


def k_of(n, alpha=ALPHA):
    """El rango que hay que coger: ceil((n+1)(1-alpha)).

    Es la única decisión del método y toda la garantía sale de ella. Se escribe
    aparte porque el navegador la reescribe igual y el guardia compara las dos.
    """
    return int(np.ceil((n + 1) * (1 - alpha)))


# =====================================================================
#  1. La cobertura exacta, sin ruido de muestreo
# =====================================================================
def stage_exact():
    """La garantía depende solo de rangos, así que se puede medir sin datos.

    Se sortean n puntuaciones de calibración de una exponencial (cualquier
    continua vale: eso ES el teorema), se coge la k-ésima, y la cobertura de
    ESA calibración es F(q) con F conocida. Nada se estima, así que lo que la
    tabla compara es la fórmula contra la definición y no contra una media
    ruidosa.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for n in [20, 50, 100, 200, 500, 2000]:
        k = k_of(n)
        s = rng.exponential(size=(TRIALS, n))
        s.sort(axis=1)
        q = s[:, k - 1]
        cov = 1.0 - np.exp(-q)              # F de la exponencial, exacta
        a, b = k, n + 1 - k
        rows.append(dict(
            n=n, k=k, target=1 - ALPHA,
            formula=k / (n + 1),
            mean=float(cov.mean()),
            upper=k / (n + 1) + 1.0 / (n + 1),
            sd=float(cov.std(ddof=1)),
            sd_closed=float(np.sqrt(a * b / ((n + 1) ** 2 * (n + 2)))),
            below=float((cov < 1 - ALPHA).mean()),
            below_closed=float(stats.beta.cdf(1 - ALPHA, a, b)),
            q05=float(np.quantile(cov, 0.05)), q95=float(np.quantile(cov, 0.95)),
            beta_a=a, beta_b=b,
        ))
    # El histograma que la página dibuja, con su Beta exacta encima.
    n = 100
    k = k_of(n)
    s = rng.exponential(size=(TRIALS, n))
    s.sort(axis=1)
    cov = 1.0 - np.exp(-s[:, k - 1])
    edges = np.linspace(0.70, 1.0, 61)
    hist, _ = np.histogram(cov, bins=edges, density=True)
    xs = np.linspace(0.70, 1.0, 121)
    return dict(rows=rows, trials=TRIALS, alpha=ALPHA,
                hist=dict(n=n, k=k, edges=r(edges, 5), density=r(hist, 5),
                          x=r(xs, 5), beta=r(stats.beta.pdf(xs, k, n + 1 - k), 5),
                          beta_a=k, beta_b=n + 1 - k))


# =====================================================================
#  2 y 3. Sobre datos: cualquier modelo, y marginal contra condicional
# =====================================================================
def make_data(n, rng):
    """Dos grupos con la misma señal y ruido muy distinto.

    El grupo 1 tiene cuatro veces la desviación del grupo 0, que es la
    situación en la que un intervalo de anchura fija tiene que estar mal en los
    dos sitios a la vez: ancho de más para el fácil y corto para el difícil.
    """
    g = (rng.random(n) < 0.5).astype(int)
    x = rng.uniform(-3, 3, size=(n, 3))
    sig = np.where(g == 1, 2.0, 0.5)
    y = (np.sin(1.5 * x[:, 0]) * 2.0 + 0.8 * x[:, 1] - 0.5 * x[:, 2] ** 2
         + sig * rng.standard_normal(n))
    return x, y, g, sig


class ConstantModel:
    def fit(self, X, y):
        self.m = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


class WrongModel:
    """Un modelo lineal sobre datos que no lo son, sin nada que lo salve."""

    def fit(self, X, y):
        self.f = LinearRegression().fit(X[:, :1], y)
        return self

    def predict(self, X):
        return self.f.predict(X[:, :1])


def split_conformal(res_cal, res_test, alpha=ALPHA):
    n = len(res_cal)
    k = k_of(n, alpha)
    q = np.sort(res_cal)[k - 1] if k <= n else np.inf
    return q, res_test <= q


def stage_models():
    """Cuatro modelos, la misma garantía, anchuras que no se parecen."""
    rng = np.random.default_rng(SEED + 1)
    builders = [
        ("forest", "random forest", lambda: RandomForestRegressor(
            n_estimators=120, min_samples_leaf=4, random_state=0, n_jobs=1)),
        ("linear", "linear on all three", lambda: LinearRegression()),
        ("wrong", "linear on one feature", WrongModel),
        ("constant", "the mean of the training set", ConstantModel),
    ]
    out = {k: dict(cov=[], width=[]) for k, _, _ in builders}
    for _ in range(SPLITS):
        X, y, g, sig = make_data(N_DATA, rng)
        idx = rng.permutation(N_DATA)
        tr, cal, te = idx[:1000], idx[1000:2000], idx[2000:]
        for key, _, build in builders:
            m = build().fit(X[tr], y[tr])
            rc = np.abs(y[cal] - m.predict(X[cal]))
            rt = np.abs(y[te] - m.predict(X[te]))
            q, hit = split_conformal(rc, rt)
            out[key]["cov"].append(float(hit.mean()))
            out[key]["width"].append(float(2 * q))
    rows = []
    for key, label, _ in builders:
        c = np.array(out[key]["cov"])
        w = np.array(out[key]["width"])
        rows.append(dict(key=key, label=label, coverage=float(c.mean()),
                         coverage_sd=float(c.std(ddof=1)), width=float(w.mean()),
                         width_sd=float(w.std(ddof=1))))
    return dict(rows=rows, splits=SPLITS, n=N_DATA, cal=1000)


def stage_conditional():
    """Tres formas de repartir el 90%, y qué grupo paga cada una."""
    rng = np.random.default_rng(SEED + 2)
    acc = {k: dict(g0=[], g1=[], all=[], w0=[], w1=[])
           for k in ["plain", "mondrian", "cqr"]}
    for _ in range(SPLITS):
        X, y, g, sig = make_data(N_DATA, rng)
        idx = rng.permutation(N_DATA)
        tr, cal, te = idx[:1000], idx[1000:2000], idx[2000:]
        Xtr, ytr = np.c_[X[tr], g[tr]], y[tr]
        m = RandomForestRegressor(n_estimators=120, min_samples_leaf=4,
                                  random_state=0, n_jobs=1).fit(Xtr, ytr)
        pc = m.predict(np.c_[X[cal], g[cal]])
        pt = m.predict(np.c_[X[te], g[te]])
        rc, rt = np.abs(y[cal] - pc), np.abs(y[te] - pt)

        q, hit = split_conformal(rc, rt)
        acc["plain"]["all"].append(float(hit.mean()))
        for gg in (0, 1):
            m_te = g[te] == gg
            acc["plain"][f"g{gg}"].append(float(hit[m_te].mean()))
            acc["plain"][f"w{gg}"].append(float(2 * q))

        # Mondrian: una calibración por grupo, que es literalmente el mismo
        # método aplicado dos veces.
        hits = np.zeros(len(te), dtype=bool)
        for gg in (0, 1):
            mc, mt = g[cal] == gg, g[te] == gg
            qg, h = split_conformal(rc[mc], rt[mt])
            hits[mt] = h
            acc["mondrian"][f"g{gg}"].append(float(h.mean()))
            acc["mondrian"][f"w{gg}"].append(float(2 * qg))
        acc["mondrian"]["all"].append(float(hits.mean()))

        # CQR: dos regresiones cuantílicas y una puntuación firmada, sin que
        # nadie le diga cuál es el grupo difícil.
        lo = QuantileRegressor(quantile=ALPHA / 2, alpha=0.0, solver="highs").fit(
            np.c_[X[tr], g[tr]], ytr)
        hi = QuantileRegressor(quantile=1 - ALPHA / 2, alpha=0.0, solver="highs").fit(
            np.c_[X[tr], g[tr]], ytr)
        lc, hc = lo.predict(np.c_[X[cal], g[cal]]), hi.predict(np.c_[X[cal], g[cal]])
        lt, ht = lo.predict(np.c_[X[te], g[te]]), hi.predict(np.c_[X[te], g[te]])
        sc = np.maximum(lc - y[cal], y[cal] - hc)
        st = np.maximum(lt - y[te], y[te] - ht)
        q2, hit2 = split_conformal(sc, st)
        acc["cqr"]["all"].append(float(hit2.mean()))
        for gg in (0, 1):
            m_te = g[te] == gg
            acc["cqr"][f"g{gg}"].append(float(hit2[m_te].mean()))
            acc["cqr"][f"w{gg}"].append(float(np.mean((ht - lt + 2 * q2)[m_te])))
    rows = []
    labels = dict(plain="one interval for everybody", mondrian="calibrated per group",
                  cqr="conformalised quantile regression")
    for k, v in acc.items():
        rows.append(dict(key=k, label=labels[k],
                         **{f"{kk}": float(np.mean(vv)) for kk, vv in v.items()},
                         **{f"{kk}_sd": float(np.std(vv, ddof=1)) for kk, vv in v.items()}))
    return dict(rows=rows, splits=SPLITS, sigma=[0.5, 2.0])


# =====================================================================
#  4. La hipótesis, rota sobre datos de verdad
# =====================================================================
def stage_exchange():
    """El CO2 del artículo 51, partido de dos maneras.

    Los rasgos son doce retardos, así que el modelo no ve el tiempo: si la
    cobertura se cae al partir por tiempo no es porque el modelo extrapole una
    fecha, es porque la distribución de los residuos de mañana no es la de
    ayer. El brazo barajado usa **los mismos pares** y el mismo modelo, así que
    la única diferencia entre los dos es el orden.
    """
    y, labels, meta = co2_monthly()
    y = np.round(y, 4)
    assert digest(y) == CO2_DIGEST, (
        f"la serie de CO2 no es la que publica el artículo 51: {digest(y)}")

    lag = 12
    X = np.stack([y[i:len(y) - lag + i] for i in range(lag)], axis=1)
    target = y[lag:]
    n = len(target)

    def run(order, tag):
        tr, cal, te = order[:n // 3], order[n // 3:2 * n // 3], order[2 * n // 3:]
        m = LinearRegression().fit(X[tr], target[tr])
        rc = np.abs(target[cal] - m.predict(X[cal]))
        rt = np.abs(target[te] - m.predict(X[te]))
        q, hit = split_conformal(rc, rt)
        return dict(kind=tag, coverage=float(hit.mean()), width=float(2 * q),
                    n_test=int(len(te)),
                    mean_abs_cal=float(rc.mean()), mean_abs_test=float(rt.mean()))

    chrono = run(np.arange(n), "in time order")
    rng = np.random.default_rng(SEED + 3)
    shuffled = [run(rng.permutation(n), "shuffled") for _ in range(40)]
    sh = dict(kind="shuffled",
              coverage=float(np.mean([s["coverage"] for s in shuffled])),
              coverage_sd=float(np.std([s["coverage"] for s in shuffled], ddof=1)),
              width=float(np.mean([s["width"] for s in shuffled])),
              n_test=shuffled[0]["n_test"],
              mean_abs_cal=float(np.mean([s["mean_abs_cal"] for s in shuffled])),
              mean_abs_test=float(np.mean([s["mean_abs_test"] for s in shuffled])))

    # Y la versión que la página dibuja: el intervalo mes a mes en el tramo de
    # test cronológico, con los fallos marcados.
    tr, cal, te = np.arange(n // 3), np.arange(n // 3, 2 * n // 3), np.arange(2 * n // 3, n)
    m = LinearRegression().fit(X[tr], target[tr])
    rc = np.abs(target[cal] - m.predict(X[cal]))
    q = np.sort(rc)[k_of(len(rc)) - 1]
    pred = m.predict(X[te])
    return dict(
        chrono=chrono, shuffled=sh, reps=40, lag=lag, n=n,
        digest=CO2_DIGEST, months=int(meta["months_interpolated"]),
        series=dict(labels=[labels[lag + i] for i in te.tolist()],
                    y=r(target[te], 4), pred=r(pred, 4), q=r(q, 4),
                    cal_labels=[labels[lag + i] for i in [cal[0], cal[-1]]],
                    train_labels=[labels[lag + i] for i in [tr[0], tr[-1]]]),
    )


# =====================================================================
#  5. Clasificación: la predicción es un conjunto
# =====================================================================
def stage_classification():
    """Dos puntuaciones sobre los mismos dígitos, y qué compra cada una.

    LAC coge la clase verdadera y se queda con 1 - p: da los conjuntos más
    pequeños que existen a esa cobertura, y por eso mismo deja clases enteras
    por debajo. APS suma probabilidades ordenadas hasta llegar a la verdadera,
    que cuesta conjuntos más grandes y reparte mejor la cobertura.
    """
    d = load_digits()
    X, y = d.data / 16.0, d.target
    rng = np.random.default_rng(SEED + 4)
    n = len(y)
    accs = {k: dict(cov=[], size=[], empty=[], worst=[], singleton=[])
            for k in ["lac", "aps", "aps_rand"]}
    top1 = []
    for _ in range(60):
        idx = rng.permutation(n)
        tr, cal, te = idx[:900], idx[900:1350], idx[1350:]
        m = RandomForestClassifier(n_estimators=200, random_state=0,
                                   n_jobs=1).fit(X[tr], y[tr])
        pc, pt = m.predict_proba(X[cal]), m.predict_proba(X[te])
        top1.append(float((pt.argmax(1) == y[te]).mean()))

        # LAC
        s_cal = 1.0 - pc[np.arange(len(cal)), y[cal]]
        q = np.sort(s_cal)[k_of(len(cal)) - 1]
        sets = (1.0 - pt) <= q
        hit = sets[np.arange(len(te)), y[te]]
        accs["lac"]["cov"].append(float(hit.mean()))
        accs["lac"]["size"].append(float(sets.sum(1).mean()))
        accs["lac"]["empty"].append(float((sets.sum(1) == 0).mean()))
        accs["lac"]["singleton"].append(float((sets.sum(1) == 1).mean()))
        accs["lac"]["worst"].append(float(min(
            hit[y[te] == c].mean() for c in range(10))))

        # APS
        def aps_scores(P, labels=None):
            order = np.argsort(-P, axis=1)
            sortedP = np.take_along_axis(P, order, axis=1)
            cum = np.cumsum(sortedP, axis=1)
            if labels is None:
                return order, cum
            rank = np.argmax(order == labels[:, None], axis=1)
            return cum[np.arange(len(P)), rank]

        s_cal = aps_scores(pc, y[cal])
        q2 = np.sort(s_cal)[k_of(len(cal)) - 1]
        order, cum = aps_scores(pt)
        keep = cum - np.take_along_axis(pt, order, axis=1) < q2
        sets2 = np.zeros_like(keep)
        np.put_along_axis(sets2, order, keep, axis=1)
        hit2 = sets2[np.arange(len(te)), y[te]]
        accs["aps"]["cov"].append(float(hit2.mean()))
        accs["aps"]["size"].append(float(sets2.sum(1).mean()))
        accs["aps"]["empty"].append(float((sets2.sum(1) == 0).mean()))
        accs["aps"]["singleton"].append(float((sets2.sum(1) == 1).mean()))
        accs["aps"]["worst"].append(float(min(
            hit2[y[te] == c].mean() for c in range(10))))

        # APS aleatorizado. El de arriba mete SIEMPRE la clase que cruza el
        # umbral, así que se pasa por arriba, y con un clasificador seguro se
        # pasa muchísimo: la masa acumulada hasta la clase verdadera vale casi
        # 1 en casi todas las filas de calibración, el cuantil se pega a 1 y
        # los conjuntos se llenan. La versión del paper se queda con una
        # fracción uniforme de la última clase, y eso es lo que devuelve la
        # cobertura exacta en vez de una cota.
        u_cal = rng.random(len(cal))
        order_c = np.argsort(-pc, axis=1)
        sortedc = np.take_along_axis(pc, order_c, axis=1)
        cumc = np.cumsum(sortedc, axis=1)
        rank_c = np.argmax(order_c == y[cal][:, None], axis=1)
        take = np.arange(len(cal))
        s_rand = (cumc[take, rank_c] - sortedc[take, rank_c]
                  + u_cal * sortedc[take, rank_c])
        q3 = np.sort(s_rand)[k_of(len(cal)) - 1]
        u_te = rng.random(len(te))
        order_t = np.argsort(-pt, axis=1)
        sortedt = np.take_along_axis(pt, order_t, axis=1)
        cumt = np.cumsum(sortedt, axis=1)
        keep3 = (cumt - sortedt + u_te[:, None] * sortedt) <= q3
        sets3 = np.zeros_like(keep3)
        np.put_along_axis(sets3, order_t, keep3, axis=1)
        hit3 = sets3[np.arange(len(te)), y[te]]
        accs["aps_rand"]["cov"].append(float(hit3.mean()))
        accs["aps_rand"]["size"].append(float(sets3.sum(1).mean()))
        accs["aps_rand"]["empty"].append(float((sets3.sum(1) == 0).mean()))
        accs["aps_rand"]["singleton"].append(float((sets3.sum(1) == 1).mean()))
        accs["aps_rand"]["worst"].append(float(min(
            hit3[y[te] == c].mean() for c in range(10))))

    rows = []
    labels = dict(lac="least ambiguous sets (1 - p)",
                  aps="adaptive sets, not randomised",
                  aps_rand="adaptive sets, randomised")
    for k, v in accs.items():
        rows.append(dict(key=k, label=labels[k],
                         **{kk: float(np.mean(vv)) for kk, vv in v.items()},
                         cov_sd=float(np.std(v["cov"], ddof=1))))
    # Una muestra para el widget: las probabilidades de veinte dígitos de test.
    idx = rng.permutation(n)
    tr, cal, te = idx[:900], idx[900:1350], idx[1350:]
    m = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1).fit(X[tr], y[tr])
    pc, pt = m.predict_proba(X[cal]), m.predict_proba(X[te])
    pick = np.argsort(-(pt.max(1)))[::len(te) // 20][:20]
    return dict(rows=rows, splits=60, top1=float(np.mean(top1)), classes=10,
                n=n, cal=450,
                demo=dict(p=r(pt[pick], 5), y=[int(v) for v in y[te][pick]],
                          images=r((X[te][pick] * 16).astype(int), 0),
                          cal_scores=r(np.sort(1.0 - pc[np.arange(len(cal)), y[cal]]), 5)))


def stage_cloud():
    """Un split para que el navegador recalcule el intervalo mientras el lector
    mueve alpha: viajan las puntuaciones de calibración y las de test, que es
    todo lo que el método necesita."""
    rng = np.random.default_rng(SEED + 5)
    X, y, g, sig = make_data(1600, rng)
    idx = rng.permutation(1600)
    tr, cal, te = idx[:600], idx[600:1100], idx[1100:]
    m = RandomForestRegressor(n_estimators=120, min_samples_leaf=4, random_state=0,
                              n_jobs=1).fit(X[tr], y[tr])
    rc = np.abs(y[cal] - m.predict(X[cal]))
    pt = m.predict(X[te])
    return dict(cal_scores=r(np.sort(rc), 5), n_cal=int(len(cal)),
                test=dict(x=r(X[te][:, 0], 4), y=r(y[te], 4), pred=r(pt, 4),
                          g=[int(v) for v in g[te]]),
                ref=[dict(alpha=a, k=k_of(len(cal), a),
                          q=r(float(np.sort(rc)[min(k_of(len(cal), a), len(rc)) - 1])),
                          coverage=r(float((np.abs(y[te] - pt)
                                            <= np.sort(rc)[min(k_of(len(cal), a),
                                                               len(rc)) - 1]).mean())))
                     for a in [0.01, 0.05, 0.1, 0.2, 0.3]])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exact = cached("exact", stage_exact)
    for row in exact["rows"]:
        print(f"  n={row['n']:>5}  k={row['k']:>5}  formula {row['formula']:.6f}  "
              f"measured {row['mean']:.6f}  sd {row['sd']:.4f} "
              f"(closed {row['sd_closed']:.4f})  below target {row['below']:.3f}")
    models = cached("models", stage_models)
    for row in models["rows"]:
        print(f"    {row['label']:<26} coverage {row['coverage']:.4f}  "
              f"width {row['width']:.3f}")
    cond = cached("conditional", stage_conditional)
    for row in cond["rows"]:
        print(f"    {row['label']:<34} all {row['all']:.4f}  easy {row['g0']:.4f}  "
              f"hard {row['g1']:.4f}  width {row['w0']:.2f}/{row['w1']:.2f}")
    ex = cached("exchange", stage_exchange)
    print(f"  CO2 in time order: coverage {ex['chrono']['coverage']:.4f}; "
          f"shuffled {ex['shuffled']['coverage']:.4f} "
          f"(sd {ex['shuffled']['coverage_sd']:.4f})")
    clf = cached("classification", stage_classification)
    for row in clf["rows"]:
        print(f"    {row['label']:<32} coverage {row['cov']:.4f}  size {row['size']:.3f}  "
              f"worst class {row['worst']:.4f}  empty {row['empty']:.4f}")
    cloud = stage_cloud()

    worst = max(abs(row["mean"] - row["formula"]) for row in exact["rows"])
    assert worst < 5e-4, f"la cobertura medida no reproduce la fórmula ({worst:.2e})"
    covs = [row["coverage"] for row in models["rows"]]
    assert max(covs) - min(covs) < 0.01, \
        "los cuatro modelos no tienen la misma cobertura, y esa es la tesis"
    widths = [row["width"] for row in models["rows"]]
    assert max(widths) / min(widths) > 1.5, \
        "las anchuras no se separan, así que la sección no enseña el precio"
    plain = [row for row in cond["rows"] if row["key"] == "plain"][0]
    assert plain["g0"] - plain["g1"] > 0.1, \
        "los dos grupos no se separan en cobertura y la sección no tiene sujeto"
    mond = [row for row in cond["rows"] if row["key"] == "mondrian"][0]
    assert abs(mond["g0"] - mond["g1"]) < 0.03, "Mondrian no está igualando los grupos"
    byc = {row["key"]: row for row in clf["rows"]}
    assert abs(byc["aps_rand"]["cov"] - (1 - ALPHA)) < 0.01, \
        "el APS aleatorizado no da la cobertura exacta"
    assert byc["aps"]["cov"] > byc["aps_rand"]["cov"] + 0.02, \
        "el APS sin aleatorizar no se pasa, y esa fila existe para enseñar que se pasa"
    assert byc["lac"]["size"] < byc["aps_rand"]["size"], \
        "LAC no da los conjuntos más pequeños, que es lo único que compra"
    assert ex["chrono"]["coverage"] < 1 - ALPHA - 0.05, \
        "partir por tiempo no rompe la cobertura, y esa es la sección"
    assert abs(ex["shuffled"]["coverage"] - (1 - ALPHA)) < 0.02, \
        "el control barajado tampoco cubre, así que la causa no es el orden"

    data = dict(
        meta=dict(seed=SEED, alpha=ALPHA, trials=exact["trials"], splits=SPLITS,
                  n=N_DATA, co2_digest=CO2_DIGEST,
                  note="split conformal; the score is the absolute residual "
                       "unless the row says otherwise"),
        exact=exact,
        models=models,
        conditional=dict(splits=cond["splits"], sigma=cond["sigma"],
                         rows=[{k: (v if isinstance(v, str) else r(v))
                                for k, v in row.items()} for row in cond["rows"]]),
        exchange=ex,
        classification=dict(splits=clf["splits"], top1=r(clf["top1"]),
                            classes=clf["classes"], n=clf["n"], cal=clf["cal"],
                            rows=[{k: (v if isinstance(v, str) else r(v))
                                   for k, v in row.items()} for row in clf["rows"]],
                            demo=clf["demo"]),
        cloud=cloud,
    )
    path = OUT / "conformal.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
