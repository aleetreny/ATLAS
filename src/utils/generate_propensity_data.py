"""Las cifras del artículo de propensity scores.

El artículo anterior deja una pregunta abierta: si ajustar por el confusor
arregla el sesgo, ¿por qué existe toda una literatura encima? Porque ajustar
por un vector de confusores no es ajustar por uno. Con cuatro covariables
continuas no hay dos personas con el mismo perfil, así que "comparar iguales
con iguales" deja de tener sujetos, y el teorema de Rosenbaum y Rubin dice algo
que no es evidente: basta con igualar **un solo número**, la probabilidad de
haber sido tratado.

Lo que se mide aquí, con el mundo escrito en forma cerrada para que el efecto
verdadero sea un parámetro y no una estimación:

  1. **La propiedad de balanceo, comprobada en vez de citada.** Dentro de
     estratos finos del score verdadero, la diferencia estandarizada de cada
     covariable entre tratados y controles se hunde. Y el control que hace que
     eso signifique algo: un score **equivocado** (calculado con dos de las
     cuatro covariables) balancea las dos que mira y deja las otras dos
     exactamente donde estaban.
  2. **El marcador de los siete estimadores**, con suelo de ruido. Cada uno se
     repite sobre 240 muestras independientes, así que la tabla publica sesgo
     y desviación, y "gana" quiere decir algo.
  3. **La tabla dos por dos de la doble robustez.** Modelo de resultado bien o
     mal, modelo de propensión bien o mal: cuatro celdas. Es la única forma de
     enseñar que "doblemente robusto" no quiere decir "más robusto", quiere
     decir "vale con que UNO de los dos esté bien", y la celda de abajo a la
     derecha existe.
  4. **El solapamiento**, barrido: la cuota del peso máximo, el tamaño efectivo
     de muestra de Kish y lo que le pasa a la varianza. Y lo que casi nunca se
     dice del recorte: recortar no arregla el estimador, **cambia la población
     sobre la que se estima**, y como aquí el efecto es heterogéneo se puede
     medir cuánto se mueve el estimando (no el error) al recortar.
  5. **Balancear no es identificar.** Se esconde una covariable, se consigue
     balance perfecto en las que quedan, y el sesgo sigue ahí entero. Ningún
     diagnóstico de balance puede verlo, y esa es la frase que hay que decir
     al lado de cualquier tabla de balance.

Ejecutar desde cualquier sitio: `python src/utils/generate_propensity_data.py`.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "propensity" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "propensity_cache")

SEED = 19

# El mundo. Cuatro covariables independientes N(0,1):
#   V0, V1 deciden el tratamiento Y el resultado    (confusores de verdad)
#   V2     decide solo el resultado                 (pronóstica, no confusora)
#   V3     decide solo el tratamiento               (instrumento observado)
#
#   e(V) = sigmoide(THETA . V)
#   X ~ Bernoulli(e(V))
#   Y = BASE + TAU(V)*X + f(V) + ruido
#
# Esa separación no es decorativa: las cuatro columnas se comportan distinto al
# meterlas en el score, y la tabla de la sección de solapamiento lo mide.
THETA = np.array([1.0, 0.8, 0.0, 1.1])
PROG = np.array([1.5, -1.0, 2.0, 0.0])
BASE = 5.0
NOISE = 1.0
N = 2000
REPS = 240

# El efecto es heterogéneo a propósito: TAU_0 más TAU_V . V. Con efecto
# constante la sección de recorte no tendría nada que medir, porque el
# estimando no dependería de a quién se recorta.
# Y el efecto depende de V0, que es TAMBIÉN una de las que deciden el
# tratamiento. Eso no es un adorno: es lo que hace que recortar por score
# cambie la población y con ella el número correcto. La primera versión hacía
# depender el efecto solo de V2, que no toca la asignación, y entonces recortar
# dejaba el estimando clavado (2,0013 contra 2,0053) y la sección no tenía nada
# que medir.
TAU_0 = 2.0
TAU_V = np.array([0.9, 0.0, 0.5, 0.0])

TAG = f"n{N}r{REPS}s{SEED}-v2"


def cached(name, fn, tag=True):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (f"{name}.{TAG}.pkl" if tag else f"{name}.pkl")
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


def tau(V):
    return TAU_0 + V @ TAU_V


def propensity(V, theta=THETA, scale=1.0):
    return 1.0 / (1.0 + np.exp(-scale * (V @ theta)))


def draw(n, rng, scale=1.0, prog=PROG):
    V = rng.standard_normal((n, 4))
    e = propensity(V, scale=scale)
    X = (rng.random(n) < e).astype(float)
    Y = BASE + tau(V) * X + V @ prog + NOISE * rng.standard_normal(n)
    return V, X, Y, e


# La verdad. El ATE es E[tau(V)] = TAU_0 porque V tiene media cero, pero se
# comprueba simulando el mundo intervenido en vez de darlo por bueno.
def true_ate(rng, n=4_000_000):
    V = rng.standard_normal((n, 4))
    return float(tau(V).mean())


# ------------------------------------------------------------ estimadores
def est_naive(V, X, Y, e_hat, **kw):
    return Y[X == 1].mean() - Y[X == 0].mean()


def est_regression(V, X, Y, e_hat, cols=None, **kw):
    """Regresión con las covariables dentro. El modelo de resultado correcto es
    lineal en V, así que este es el brazo 'modelo bien especificado'."""
    Z = V if cols is None else V[:, cols]
    return LinearRegression().fit(np.c_[X, Z], Y).coef_[0]


def est_ipw(V, X, Y, e_hat, **kw):
    """Horvitz-Thompson: cada unidad pesa lo que cuesta encontrarla."""
    w1 = X / e_hat
    w0 = (1 - X) / (1 - e_hat)
    return float((w1 * Y).sum() / w1.sum() - (w0 * Y).sum() / w0.sum())


def est_ipw_unstabilised(V, X, Y, e_hat, **kw):
    n = len(Y)
    return float((X * Y / e_hat).sum() / n - ((1 - X) * Y / (1 - e_hat)).sum() / n)


def est_strata(V, X, Y, e_hat, k=5, **kw):
    edges = np.quantile(e_hat, np.linspace(0, 1, k + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, e_hat, side="right") - 1, 0, k - 1)
    num, den = 0.0, 0
    for s in range(k):
        m = idx == s
        a, b = Y[m & (X == 1)], Y[m & (X == 0)]
        if a.size < 2 or b.size < 2:
            continue
        num += m.sum() * (a.mean() - b.mean())
        den += m.sum()
    return num / den if den else np.nan


def est_matching(V, X, Y, e_hat, **kw):
    """Emparejamiento 1 a 1 con reemplazo sobre el logit del score.

    Sobre el logit y no sobre el score: cerca de 0 y de 1 el score se comprime
    y dos unidades a 0,98 y 0,999 parecen vecinas cuando sus odds difieren por
    veinte. Es la recomendación de Rosenbaum y Rubin y se nota en el sesgo.
    """
    lg = np.log(np.clip(e_hat, 1e-9, 1 - 1e-9) / np.clip(1 - e_hat, 1e-9, 1 - 1e-9))
    t, c = np.where(X == 1)[0], np.where(X == 0)[0]
    # ATT emparejando tratados con controles y ATC al revés, promediados a ATE.
    d_t = np.abs(lg[t][:, None] - lg[c][None, :])
    att = (Y[t] - Y[c[d_t.argmin(1)]]).mean()
    atc = (Y[t[d_t.argmin(0)]] - Y[c]).mean()
    return (att * len(t) + atc * len(c)) / len(Y)


def est_aipw(V, X, Y, e_hat, cols=None, **kw):
    """La forma aumentada: modelo de resultado más corrección por propensión.

    Si el modelo de resultado acierta, el segundo término tiene media cero; si
    el de propensión acierta, el primero se corrige. De ahí el nombre, y de ahí
    que la tabla dos por dos tenga tres celdas buenas y no cuatro.
    """
    Z = V if cols is None else V[:, cols]
    m1 = LinearRegression().fit(Z[X == 1], Y[X == 1])
    m0 = LinearRegression().fit(Z[X == 0], Y[X == 0])
    mu1, mu0 = m1.predict(Z), m0.predict(Z)
    a = mu1 + X * (Y - mu1) / e_hat
    b = mu0 + (1 - X) * (Y - mu0) / (1 - e_hat)
    return float((a - b).mean())


ESTIMATORS = [
    ("naive", "difference in means", est_naive),
    ("regression", "outcome regression", est_regression),
    ("ipw", "IPW, stabilised", est_ipw),
    ("ipw_raw", "IPW, unstabilised", est_ipw_unstabilised),
    ("strata", "five strata of the score", est_strata),
    ("matching", "1-NN matching on the logit", est_matching),
    ("aipw", "doubly robust (AIPW)", est_aipw),
]


# ------------------------------------------------------------------ etapas
def smd(V, X, w=None):
    """Diferencia estandarizada de medias, columna a columna.

    Con la desviación agrupada del denominador, que es la convención que hace
    la cifra comparable entre covariables y entre estudios.
    """
    out = []
    for j in range(V.shape[1]):
        a, b = V[X == 1, j], V[X == 0, j]
        if w is None:
            ma, mb = a.mean(), b.mean()
        else:
            wa, wb = w[X == 1], w[X == 0]
            ma, mb = (wa * a).sum() / wa.sum(), (wb * b).sum() / wb.sum()
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        out.append(float((ma - mb) / sd) if sd > 0 else 0.0)
    return out


def stage_balance():
    """La propiedad de balanceo, con su control.

    Se usa el score VERDADERO, no uno estimado, porque lo que el teorema dice
    es sobre el verdadero y mezclarlo con el error de estimación haría el
    experimento ilegible. El score equivocado es el mismo cálculo con dos
    covariables, y es el control que convierte "balancea" en "balancea lo que
    mira".
    """
    rng = np.random.default_rng(SEED)
    n = 200_000
    V, X, Y, e = draw(n, rng)
    theta_bad = THETA.copy()
    theta_bad[1] = 0.0
    theta_bad[3] = 0.0
    e_bad = propensity(V, theta=theta_bad)

    def within(score, k=20):
        edges = np.quantile(score, np.linspace(0, 1, k + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        idx = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, k - 1)
        acc = np.zeros(V.shape[1])
        tot = 0
        for s in range(k):
            m = idx == s
            if (X[m] == 1).sum() < 20 or (X[m] == 0).sum() < 20:
                continue
            acc += m.sum() * np.abs(smd(V[m], X[m]))
            tot += m.sum()
        return (acc / tot).tolist()

    return dict(
        before=smd(V, X),
        after_true=within(e),
        after_wrong=within(e_bad),
        after_ipw=smd(V, X, w=np.where(X == 1, 1 / e, 1 / (1 - e))),
        strata=20, n=n,
        cols=["V0 (treatment and outcome)", "V1 (treatment and outcome)",
              "V2 (outcome only)", "V3 (treatment only)"],
    )


def stage_scoreboard():
    """El marcador, repetido REPS veces para tener suelo de ruido.

    El score se ESTIMA aquí con una logística bien especificada, que es el caso
    favorable y el que corresponde a esta sección: lo que se compara es el
    estimador, no la calidad del modelo de propensión, y ese es el sujeto de la
    sección siguiente.
    """
    rng = np.random.default_rng(SEED + 1)
    rows = {k: [] for k, _, _ in ESTIMATORS}
    for _ in range(REPS):
        V, X, Y, e = draw(N, rng)
        lr = LogisticRegression(max_iter=2000).fit(V, X)
        e_hat = np.clip(lr.predict_proba(V)[:, 1], 1e-6, 1 - 1e-6)
        for key, _, fn in ESTIMATORS:
            rows[key].append(fn(V, X, Y, e_hat))
    out = []
    for key, label, _ in ESTIMATORS:
        a = np.array(rows[key], dtype=float)
        out.append(dict(key=key, label=label, mean=float(a.mean()),
                        bias=float(a.mean() - TAU_0), sd=float(a.std(ddof=1)),
                        rmse=float(np.sqrt(((a - TAU_0) ** 2).mean())),
                        se_of_mean=float(a.std(ddof=1) / np.sqrt(REPS))))
    return dict(rows=out, reps=REPS, n=N)


def stage_double_robust():
    """La tabla dos por dos, que es la única razón de que AIPW exista.

    "Mal especificado" aquí quiere decir algo concreto y honesto: los dos
    modelos ven las covariables **al cuadrado menos uno** en vez de las
    covariables. Es una transformación que conserva la escala y la media pero
    rompe la linealidad, así que un modelo lineal sobre ellas está de verdad
    equivocado y no solo despeinado.
    """
    rng = np.random.default_rng(SEED + 2)
    cells = {}
    for ps_ok in (True, False):
        for om_ok in (True, False):
            ests = {"regression": [], "ipw": [], "aipw": []}
            for _ in range(REPS):
                V, X, Y, e = draw(N, rng)
                W = V if ps_ok else V ** 2 - 1.0
                Z = V if om_ok else V ** 2 - 1.0
                lr = LogisticRegression(max_iter=2000).fit(W, X)
                e_hat = np.clip(lr.predict_proba(W)[:, 1], 1e-6, 1 - 1e-6)
                ests["regression"].append(
                    LinearRegression().fit(np.c_[X, Z], Y).coef_[0])
                ests["ipw"].append(est_ipw(V, X, Y, e_hat))
                m1 = LinearRegression().fit(Z[X == 1], Y[X == 1]).predict(Z)
                m0 = LinearRegression().fit(Z[X == 0], Y[X == 0]).predict(Z)
                a = m1 + X * (Y - m1) / e_hat
                b = m0 + (1 - X) * (Y - m0) / (1 - e_hat)
                ests["aipw"].append(float((a - b).mean()))
            cells[f"{int(ps_ok)}{int(om_ok)}"] = {
                k: dict(bias=float(np.mean(v) - TAU_0),
                        sd=float(np.std(v, ddof=1)),
                        se=float(np.std(v, ddof=1) / np.sqrt(REPS)))
                for k, v in ests.items()}
    return dict(cells=cells, reps=REPS, n=N)


def stage_overlap():
    """Solapamiento: qué se rompe primero cuando los dos grupos se separan.

    El mando es una escala sobre el logit, así que el mismo mundo se vuelve más
    o menos separable sin cambiar qué covariable importa. Se mide la cuota del
    peso mayor, el tamaño efectivo de Kish, y la desviación del estimador entre
    repeticiones, que es lo que de verdad explota.
    """
    rng = np.random.default_rng(SEED + 3)
    rows = []
    for scale in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
        ests, shares, esss, mins = [], [], [], []
        for _ in range(60):
            V, X, Y, e = draw(N, rng, scale=scale)
            lr = LogisticRegression(max_iter=3000).fit(V, X)
            e_hat = np.clip(lr.predict_proba(V)[:, 1], 1e-6, 1 - 1e-6)
            w = np.where(X == 1, 1 / e_hat, 1 / (1 - e_hat))
            ests.append(est_ipw(V, X, Y, e_hat))
            shares.append(float(w.max() / w.sum()))
            esss.append(float(w.sum() ** 2 / (w ** 2).sum()))
            mins.append(float(min(e_hat.min(), 1 - e_hat.max())))
        a = np.array(ests)
        rows.append(dict(scale=scale, bias=float(a.mean() - TAU_0),
                         sd=float(a.std(ddof=1)),
                         max_share=float(np.mean(shares)),
                         ess=float(np.mean(esss)), ess_frac=float(np.mean(esss) / N,),
                         min_tail=float(np.mean(mins))))
    return rows


def stage_trimming():
    """Recortar cambia la pregunta, y aquí se puede medir cuánto.

    Con efecto heterogéneo, tirar las unidades de score extremo deja una
    población distinta, así que el número correcto **también** cambia. Se
    calcula el ATE verdadero dentro de cada población recortada (que se puede,
    porque tau(V) está escrito) y se compara con lo que el estimador devuelve:
    el error baja y el estimando se mueve, y las dos cosas hay que enseñarlas
    juntas o el recorte parece gratis.
    """
    rng = np.random.default_rng(SEED + 4)
    rows = []
    for lo in [0.0, 0.01, 0.05, 0.1, 0.2]:
        ests, truths, kept = [], [], []
        for _ in range(80):
            V, X, Y, e = draw(N, rng, scale=4.0)
            lr = LogisticRegression(max_iter=3000).fit(V, X)
            e_hat = np.clip(lr.predict_proba(V)[:, 1], 1e-6, 1 - 1e-6)
            m = (e_hat > lo) & (e_hat < 1 - lo)
            if m.sum() < 50 or (X[m] == 1).sum() < 10 or (X[m] == 0).sum() < 10:
                continue
            ests.append(est_ipw(V[m], X[m], Y[m], e_hat[m]))
            truths.append(float(tau(V[m]).mean()))
            kept.append(float(m.mean()))
        a, t = np.array(ests), np.array(truths)
        rows.append(dict(trim=lo, estimate=float(a.mean()), sd=float(a.std(ddof=1)),
                         truth_kept=float(t.mean()), truth_all=TAU_0,
                         kept=float(np.mean(kept)),
                         error_vs_kept=float(a.mean() - t.mean()),
                         error_vs_all=float(a.mean() - TAU_0)))
    return rows


def stage_hidden():
    """Balance perfecto, sesgo intacto.

    Se esconde V1 (confusor de verdad) del analista. El score estimado con las
    tres que quedan balancea esas tres, cualquier tabla de balance sale limpia,
    y el sesgo sigue ahí. Es la sección que dice para qué NO sirve un
    diagnóstico de balance.
    """
    rng = np.random.default_rng(SEED + 5)
    seen = [0, 2, 3]
    ests, smds_seen, smds_hidden = [], [], []
    for _ in range(REPS):
        V, X, Y, e = draw(N, rng)
        lr = LogisticRegression(max_iter=2000).fit(V[:, seen], X)
        e_hat = np.clip(lr.predict_proba(V[:, seen])[:, 1], 1e-6, 1 - 1e-6)
        w = np.where(X == 1, 1 / e_hat, 1 / (1 - e_hat))
        ests.append(est_ipw(V, X, Y, e_hat))
        s = smd(V, X, w=w)
        smds_seen.append([abs(s[j]) for j in seen])
        smds_hidden.append(abs(s[1]))
    a = np.array(ests)
    return dict(bias=float(a.mean() - TAU_0), sd=float(a.std(ddof=1)),
                se=float(a.std(ddof=1) / np.sqrt(REPS)),
                smd_seen=float(np.mean(smds_seen)),
                smd_hidden=float(np.mean(smds_hidden)),
                hidden="V1", reps=REPS)


def stage_cloud():
    """Una muestra para el navegador: la página vuelve a estimar sobre ella."""
    rng = np.random.default_rng(SEED + 6)
    V, X, Y, e = draw(600, rng)
    lr = LogisticRegression(max_iter=2000).fit(V, X)
    e_hat = np.clip(lr.predict_proba(V)[:, 1], 1e-6, 1 - 1e-6)
    return dict(
        V=r(V, 4), X=[int(v) for v in X], Y=r(Y, 4),
        e_true=r(e, 5), e_hat=r(e_hat, 5),
        coef=r(lr.coef_[0], 6), intercept=r(float(lr.intercept_[0]), 6),
        ref=dict(naive=r(est_naive(V, X, Y, e_hat)),
                 ipw=r(est_ipw(V, X, Y, e_hat)),
                 ipw_true=r(est_ipw(V, X, Y, e)),
                 strata=r(est_strata(V, X, Y, e_hat)),
                 matching=r(est_matching(V, X, Y, e_hat)),
                 aipw=r(est_aipw(V, X, Y, e_hat)),
                 regression=r(est_regression(V, X, Y, e_hat))),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    truth = cached("truth", lambda: true_ate(np.random.default_rng(SEED + 90)))
    balance = cached("balance", stage_balance)
    board = cached("board", stage_scoreboard)
    dr = cached("dr", stage_double_robust)
    overlap = cached("overlap", stage_overlap)
    trim = cached("trim", stage_trimming)
    hidden = cached("hidden", stage_hidden)
    cloud = stage_cloud()

    print(f"  true ATE {truth:.5f} (parameter {TAU_0})")
    print("  scoreboard:")
    for row in board["rows"]:
        print(f"    {row['label']:<28} bias {row['bias']:+.4f}  sd {row['sd']:.4f}")
    print(f"  balance: before {max(abs(v) for v in balance['before']):.3f}, "
          f"after {max(abs(v) for v in balance['after_true']):.3f}, "
          f"wrong score {max(abs(v) for v in balance['after_wrong']):.3f}")
    for k, v in dr["cells"].items():
        print(f"  DR cell {k}: reg {v['regression']['bias']:+.4f}  "
              f"ipw {v['ipw']['bias']:+.4f}  aipw {v['aipw']['bias']:+.4f}")
    print(f"  hidden covariate: bias {hidden['bias']:+.4f}, "
          f"smd seen {hidden['smd_seen']:.4f} hidden {hidden['smd_hidden']:.4f}")

    assert abs(truth - TAU_0) < 0.005, "el ATE simulado no es el parámetro"
    assert max(abs(v) for v in balance["after_true"]) < 0.1 * max(abs(v) for v in balance["before"]), \
        "el score verdadero no está balanceando, así que el teorema no se ve"
    assert abs(balance["after_wrong"][1]) > 0.3 * abs(balance["before"][1]), \
        "el score equivocado balancea lo que no mira, y entonces no es un control"
    aipw_bias = abs(dr["cells"]["10"]["aipw"]["bias"])
    ipw_bias = abs(dr["cells"]["10"]["ipw"]["bias"])
    assert aipw_bias < ipw_bias, "AIPW no gana en la celda donde la propensión es la buena"
    assert abs(hidden["bias"]) > 5 * hidden["se"], \
        "esconder un confusor no deja sesgo, y esa sección no tiene sujeto"

    data = dict(
        meta=dict(seed=SEED, n=N, reps=REPS, tau0=TAU_0, tau_v=r(TAU_V),
                  theta=r(THETA), prog=r(PROG), noise=NOISE, base=BASE,
                  truth=r(truth),
                  model="V ~ N(0,I4); e = sigmoid(theta.V); X ~ Bern(e); "
                        "Y = base + tau(V) X + prog.V + n"),
        balance=dict(before=r(balance["before"]), after_true=r(balance["after_true"]),
                     after_wrong=r(balance["after_wrong"]),
                     after_ipw=r(balance["after_ipw"]),
                     strata=balance["strata"], n=balance["n"], cols=balance["cols"]),
        scoreboard=dict(reps=board["reps"], n=board["n"],
                        rows=[{k: (v if isinstance(v, str) else r(v))
                               for k, v in row.items()} for row in board["rows"]]),
        double_robust=dict(reps=dr["reps"], n=dr["n"], cells={
            k: {m: {kk: r(vv) for kk, vv in c.items()} for m, c in v.items()}
            for k, v in dr["cells"].items()}),
        overlap=[{k: r(v) for k, v in row.items()} for row in overlap],
        trimming=[{k: r(v) for k, v in row.items()} for row in trim],
        hidden={k: (v if isinstance(v, str) else r(v)) for k, v in hidden.items()},
        cloud=cloud,
    )
    path = OUT / "propensity.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
