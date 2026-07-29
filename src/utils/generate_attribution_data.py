"""Las cifras del artículo de SHAP y LIME.

Una atribución dice cuánto de esta predicción es culpa de esta columna. Es la
salida más citada del campo y casi nunca se comprueba, porque parece que no hay
contra qué: no existe un dataset con la columna "atribución verdadera".

Existe si se fabrica. Dos veces, y de dos maneras distintas:

  1. **Un modelo aditivo tiene atribución conocida en forma cerrada.** Si el
     modelo es la suma de una función por columna, el valor de Shapley de la
     columna j es exactamente f_j(x_j) menos su media, por el axioma de
     dummy y el de simetría. Así que un modelo aditivo ajustado sobre datos de
     verdad es una **clave de respuestas** para el estimador, y la enumeración
     exacta tiene que reproducirla al último decimal.
  2. **Con rasgos gaussianos y un modelo lineal, las dos variantes de SHAP
     tienen forma cerrada las dos**, así que el desacuerdo entre la
     intervencional y la condicional deja de ser una discusión y es una resta.
     Ese desacuerdo es el que decide si una columna correlacionada con la que
     el modelo usa recibe crédito o no, y la mitad de las peleas sobre "SHAP
     dice que esta variable importa" son en realidad sobre cuál de las dos se
     ha calculado.

Sobre las mismas 178 botellas de vino de los artículos 1, 25, 26, 27 y 33
(afirmadas columna a columna antes de medir), se calcula:

  - **El valor de Shapley exacto enumerando los 1.024 subconjuntos**, sin
     muestrear nada, y la identidad de eficiencia comprobada a 1e-12 en cada
     instancia: las diez atribuciones suman exactamente la predicción menos la
     media.
  - **Cuánto cuesta la aproximación**: el error de KernelSHAP contra el exacto
     como función del número de coaliciones muestreadas.
  - **Qué hace una interacción**: con un término cruzado explícito, el reparto
     entre las dos columnas implicadas, que sale mitad y mitad y es una
     propiedad del axioma, no del ajuste.
  - **LIME**, que es otra cosa aunque se cite en la misma frase: su
     inestabilidad entre semillas, y el barrido del ancho del kernel, que
     **decide qué columna sale primera** y que casi nadie reporta.
  - **Y dos modelos igual de buenos con explicaciones distintas**, con el
     acierto de test dentro del ruido entre semillas y la correlación de rangos
     entre sus atribuciones medida.

Ejecutar desde cualquier sitio: `python src/utils/generate_attribution_data.py`.
"""
import json
import os
import pickle
from itertools import combinations
from math import comb, factorial
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "attribution" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "attribution_cache")

SEED = 19
P = 10               # columnas para la enumeración exacta: 2^10 = 1.024 coaliciones
N_BACK = 80          # filas de fondo para la esperanza intervencional
N_SHOW = 24          # instancias explicadas del todo

TAG = f"p{P}b{N_BACK}n{N_SHOW}s{SEED}"


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

    `json.dumps` escribe `NaN` sin protestar y eso **no es JSON**: el navegador
    lo rechaza al parsear, la promesa del fetch revienta y la página se queda
    sin prosa y sin guardia, sin un solo error visible en el sitio donde está
    el fallo. Pasó con la correlación del brazo constante, que no está definida
    porque su predicción no varía. Aquí se convierte en None y la prosa tiene
    su rama; y el volcado de abajo lleva `allow_nan=False` para que ningún otro
    campo pueda colarse.
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r(x, d) for x in np.asarray(v).tolist()]
    v = float(v)
    if not np.isfinite(v):
        return None
    return round(v, d)


# ------------------------------------------------------------------ datos
def wine_data():
    """Las mismas botellas de siempre, afirmadas y no declaradas.

    La afirmación es la del resto del atlas: forma, media y suma de la matriz
    cruda, que es lo que cambia si scikit-learn tocara el fichero.
    """
    w = load_wine()
    X, y = w.data, w.target.astype(float)
    assert X.shape == (178, 13), f"el vino no tiene la forma de siempre: {X.shape}"
    # La columna a predecir: el flavanoids, que es continua y la que el 27
    # midió como la más compartida entre factores. Así el problema es una
    # regresión de verdad sobre columnas de verdad, no una clase inventada.
    j_target = list(w.feature_names).index("flavanoids")
    keep = [j for j in range(13) if j != j_target]
    names = [w.feature_names[j] for j in keep]
    return X[:, keep], X[:, j_target], names, w


def top_columns(X, y, p=P):
    """Las p columnas que más mueven al objetivo, elegidas por una sola
    regresión sobre datos estandarizados. La elección se publica."""
    Xs = (X - X.mean(0)) / X.std(0)
    b = np.abs(LinearRegression().fit(Xs, y).coef_)
    return list(np.argsort(-b)[:p])


# --------------------------------------------------- Shapley exacto
def shapley_exact(value_fn, p):
    """El valor de Shapley por su definición, enumerando todo.

    phi_j = suma sobre S sin j de |S|!(p-|S|-1)!/p! * [v(S+j) - v(S)]. Con p =
    10 son 1.024 coaliciones y 10.240 diferencias: es caro y es exacto, que es
    justo lo que hace falta para poder juzgar a los que aproximan.
    """
    idx = {}
    subsets = []
    for size in range(p + 1):
        for S in combinations(range(p), size):
            idx[S] = len(subsets)
            subsets.append(S)
    v = value_fn(subsets)
    phi = np.zeros(p)
    for S in subsets:
        s = len(S)
        for j in range(p):
            if j in S:
                continue
            Sj = tuple(sorted(S + (j,)))
            w = factorial(s) * factorial(p - s - 1) / factorial(p)
            phi[j] += w * (v[idx[Sj]] - v[idx[S]])
    return phi, v, subsets, idx


def interventional_value(model, x, background):
    """v(S) = media sobre el fondo de f(x_S, fondo_resto).

    Es la variante que rompe la correlación entre columnas a propósito: mete
    valores de x en unas columnas y valores de otra fila en las demás, aunque
    esa combinación no exista en el mundo. Eso es lo que la hace medir el
    modelo y no los datos.
    """
    def fn(subsets):
        # Las 1.024 coaliciones se apilan en UNA sola llamada al modelo. La
        # version obvia llama a predict una vez por coalicion, y con un bosque
        # el coste por llamada domina al coste por fila: 8.192 llamadas de 80
        # filas tardan mas que una de 655.360. Es la misma cuenta que hizo el
        # articulo 49 con el softmax del vocabulario.
        nb = len(background)
        Z = np.repeat(background[None, :, :], len(subsets), axis=0)
        for i, S in enumerate(subsets):
            if S:
                Z[i][:, list(S)] = x[list(S)]
        preds = model.predict(Z.reshape(-1, background.shape[1]))
        return preds.reshape(len(subsets), nb).mean(axis=1)
    return fn


def kernel_shap(model, x, background, m, rng):
    """KernelSHAP: muestrear coaliciones y resolver la regresión ponderada.

    Los pesos son los del kernel de Shapley, y los dos extremos (el conjunto
    vacío y el completo) entran como restricciones exactas en vez de como
    filas, que es lo que hace la implementación de referencia y lo que evita
    que el peso infinito reviente el sistema.
    """
    p = len(x)
    base = model.predict(background).mean()
    full = model.predict(x[None, :]).mean()
    Zs, ws, blocks = [], [], []
    for _ in range(m):
        s = rng.integers(1, p)
        S = rng.choice(p, size=s, replace=False)
        z = np.zeros(p)
        z[S] = 1
        Z = background.copy()
        Z[:, S] = x[S]
        Zs.append(z)
        ws.append((p - 1) / (comb(p, s) * s * (p - s)))
        blocks.append(Z)
    # Igual que arriba: una llamada, no m.
    stacked = np.concatenate(blocks, axis=0)
    ys = model.predict(stacked).reshape(m, len(background)).mean(axis=1)
    Z = np.array(Zs)
    w = np.array(ws)
    yv = np.asarray(ys) - base
    # Restricción de eficiencia: se elimina la última variable sustituyéndola.
    A = Z[:, :-1] - Z[:, -1:]
    b = yv - Z[:, -1] * (full - base)
    sol = Ridge(alpha=1e-8, fit_intercept=False).fit(A, b, sample_weight=w).coef_
    phi = np.append(sol, (full - base) - sol.sum())
    return phi


class AdditiveModel:
    """Un modelo aditivo de verdad: una función por columna y nada más.

    Cada f_j es una regresión lineal por tramos sobre los cuantiles de esa
    columna, ajustada por backfitting. Importa que sea aditivo **exactamente**
    y no aproximadamente, porque de eso depende que la clave de respuestas lo
    sea: el Shapley de una suma de funciones de una variable es cada término
    centrado, y eso es un teorema, no una aproximación.
    """

    def __init__(self, knots=6):
        self.knots = knots

    def _basis(self, xj, j):
        k = self.edges[j]
        return np.stack([np.clip(xj - t, 0, None) for t in k] + [xj], axis=1)

    def fit(self, X, y):
        n, p = X.shape
        self.edges = [np.quantile(X[:, j], np.linspace(0.15, 0.85, self.knots))
                      for j in range(p)]
        self.coef = [np.zeros(self.knots + 1) for _ in range(p)]
        self.base = float(y.mean())
        self.centre = np.zeros(p)
        resid = y - self.base
        for _ in range(60):
            for j in range(p):
                B = self._basis(X[:, j], j)
                cur = B @ self.coef[j] - self.centre[j]
                target = resid + cur
                c = Ridge(alpha=1.0, fit_intercept=False).fit(B, target).coef_
                self.coef[j] = c
                # Centrado sobre la distribución de entrenamiento, que es lo
                # que hace que base + sum(contribuciones) sea una identidad.
                self.centre[j] = float((B @ c).mean())
                new = B @ c - self.centre[j]
                resid = target - new
        return self

    def parts(self, X):
        return np.stack([self._basis(X[:, j], j) @ self.coef[j] - self.centre[j]
                         for j in range(X.shape[1])], axis=1)

    def predict(self, X):
        return self.base + self.parts(X).sum(1)


def stage_answer_key(X, y, names):
    """La enumeración exacta contra la forma cerrada de un modelo aditivo."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    idx = rng.permutation(n)
    tr, te = idx[:120], idx[120:]
    m = AdditiveModel().fit(X[tr], y[tr])
    back = X[tr][:N_BACK]
    rows = []
    worst_key, worst_eff = 0.0, 0.0
    for i in te[:N_SHOW]:
        phi, v, subsets, _ = shapley_exact(interventional_value(m, X[i], back), P)
        # La clave: el propio término del modelo, centrado sobre el FONDO, que
        # es la distribución respecto a la que se está explicando.
        key = m.parts(X[i][None, :])[0] - m.parts(back).mean(0)
        worst_key = max(worst_key, float(np.abs(phi - key).max()))
        eff = float(abs(phi.sum() - (m.predict(X[i][None, :])[0] - m.predict(back).mean())))
        worst_eff = max(worst_eff, eff)
        rows.append(dict(i=int(i), phi=phi, key=key,
                         pred=float(m.predict(X[i][None, :])[0])))
    return dict(worst_key=worst_key, worst_eff=worst_eff, rows=rows,
                base=float(m.predict(back).mean()), n_show=len(rows),
                r2=float(1 - np.mean((y[te] - m.predict(X[te])) ** 2)
                         / np.var(y[te])))


def stage_interaction(X, y):
    """Una interacción explícita, y el reparto que el axioma obliga."""
    rng = np.random.default_rng(SEED + 1)
    Xs = (X - X.mean(0)) / X.std(0)

    class Cross:
        """f(x) = a*x0 + b*x1 + c*x0*x1, escrito y no ajustado."""

        def __init__(self, a, b, c):
            self.a, self.b, self.c = a, b, c

        def predict(self, Z):
            return self.a * Z[:, 0] + self.b * Z[:, 1] + self.c * Z[:, 0] * Z[:, 1]

    m = Cross(1.0, 0.5, 2.0)
    # El fondo se **centra**, y no es cosmética: con medias distintas de cero
    # el término cruzado no se reparte a medias, y el primer intento de esta
    # sección afirmaba que sí. La forma cerrada general es
    #   phi0 = a(x0-m0) + (c/2)[x0*m1 - m01 + x0*x1 - m0*x1]
    # y las dos mitades solo coinciden cuando m0 = m1 = 0. Como el fondo son
    # las primeras filas del vino, que están ordenadas por cultivar, sus medias
    # estaban lejos de cero y la desviación llegaba a 1,735.
    back = Xs[:N_BACK] - Xs[:N_BACK].mean(0)
    m0, m1 = float(back[:, 0].mean()), float(back[:, 1].mean())
    m01 = float((back[:, 0] * back[:, 1]).mean())
    out = []
    for i in range(6):
        z = Xs[i] - Xs[:N_BACK].mean(0)
        phi, _, _, _ = shapley_exact(interventional_value(m, z, back), P)
        share0 = m.c / 2 * (z[0] * m1 - m01 + z[0] * z[1] - m0 * z[1])
        share1 = m.c / 2 * (m0 * z[1] - m01 + z[0] * z[1] - z[0] * m1)
        out.append(dict(i=i, phi0=float(phi[0]), phi1=float(phi[1]),
                        share=float(share0), share1=float(share1),
                        surplus=float(m.c * (z[0] * z[1] - m01)),
                        main0=float(m.a * (z[0] - m0)),
                        main1=float(m.b * (z[1] - m1))))
    worst = max(abs(o["phi0"] - o["main0"] - o["share"]) for o in out)
    halves = max(abs(o["share"] - o["share1"]) for o in out)
    return dict(rows=out, worst=worst, halves=halves, a=m.a, b=m.b, c=m.c,
                m01=m01)


def stage_kernel_error(X, y):
    """Cuántas coaliciones hacen falta, medido contra el exacto."""
    rng = np.random.default_rng(SEED + 2)
    n = len(y)
    idx = rng.permutation(n)
    tr, te = idx[:120], idx[120:]
    m = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=0,
                              n_jobs=1).fit(X[tr], y[tr])
    back = X[tr][:N_BACK]
    exact = []
    for i in te[:8]:
        phi, _, _, _ = shapley_exact(interventional_value(m, X[i], back), P)
        exact.append(phi)
    exact = np.array(exact)
    rows = []
    for msamp in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        errs, tops = [], []
        for rep in range(5):
            for k, i in enumerate(te[:8]):
                ph = kernel_shap(m, X[i], back, msamp,
                                 np.random.default_rng(SEED + 900 + rep * 17 + k))
                errs.append(float(np.abs(ph - exact[k]).max()))
                tops.append(float(np.argmax(np.abs(ph)) == np.argmax(np.abs(exact[k]))))
        rows.append(dict(m=msamp, err=float(np.mean(errs)),
                         err_sd=float(np.std(errs, ddof=1)),
                         top1=float(np.mean(tops)),
                         relative=float(np.mean(errs) / np.abs(exact).max())))
    return dict(rows=rows, scale=float(np.abs(exact).max()), instances=8,
                exact=exact, model="random forest")


def stage_two_shaps(X, y, names, cols):
    """Intervencional contra condicional, las dos en forma cerrada.

    Se ajusta a los datos una gaussiana (media y covarianza de las columnas
    elegidas) y un modelo lineal. Con eso:

      intervencional: v(S) = b_S . x_S + b_R . mu_R
      condicional:    v(S) = b_S . x_S + b_R . E[X_R | X_S = x_S]

    y la segunda esperanza es la fórmula de la condicional gaussiana. Ninguna
    de las dos se muestrea, así que la diferencia entre las dos columnas de la
    tabla no tiene ruido: es lo que significan.
    """
    mu = X.mean(0)
    S_cov = np.cov(X.T)
    b = LinearRegression().fit(X, y).coef_
    rng = np.random.default_rng(SEED + 3)
    p = X.shape[1]

    def cond_value(x):
        def fn(subsets):
            out = np.empty(len(subsets))
            for i, S in enumerate(subsets):
                S = list(S)
                R = [j for j in range(p) if j not in S]
                if not R:
                    out[i] = b @ x
                    continue
                if not S:
                    out[i] = b @ mu
                    continue
                A = S_cov[np.ix_(R, S)] @ np.linalg.inv(S_cov[np.ix_(S, S)])
                er = mu[R] + A @ (x[S] - mu[S])
                out[i] = b[S] @ x[S] + b[R] @ er
            return out
        return fn

    def interv_value(x):
        def fn(subsets):
            out = np.empty(len(subsets))
            for i, S in enumerate(subsets):
                S = list(S)
                R = [j for j in range(p) if j not in S]
                out[i] = (b[S] @ x[S] if S else 0.0) + (b[R] @ mu[R] if R else 0.0)
            return out
        return fn

    rows = []
    for i in range(8):
        pi, _, _, _ = shapley_exact(interv_value(X[i]), p)
        pc, _, _, _ = shapley_exact(cond_value(X[i]), p)
        rows.append(dict(i=i, interventional=pi, conditional=pc))
    PI = np.array([row["interventional"] for row in rows])
    PC = np.array([row["conditional"] for row in rows])
    corr = np.corrcoef(np.abs(S_cov[np.triu_indices(p, 1)]),
                       np.abs(S_cov[np.triu_indices(p, 1)]))[0, 1]
    # La columna que el modelo NO usa pero que está correlacionada con una que
    # sí: se mide su atribución en las dos variantes.
    unused = int(np.argmin(np.abs(b) * X.std(0)))
    partner = int(np.argmax([abs(np.corrcoef(X[:, unused], X[:, j])[0, 1])
                             if j != unused else 0 for j in range(p)]))
    return dict(
        rows=[dict(i=row["i"], interventional=r(row["interventional"]),
                   conditional=r(row["conditional"])) for row in rows],
        max_gap=float(np.abs(PI - PC).max()),
        mean_gap=float(np.abs(PI - PC).mean()),
        unused=unused, unused_name=names[cols[unused]],
        partner=partner, partner_name=names[cols[partner]],
        unused_corr=float(np.corrcoef(X[:, unused], X[:, partner])[0, 1]),
        unused_interventional=float(np.abs(PI[:, unused]).mean()),
        unused_conditional=float(np.abs(PC[:, unused]).mean()),
        beta=r(b), sd=r(X.std(0)), corr=float(corr),
    )


# ---------------------------------------------------------------- LIME
def lime_explain(model, x, X, width, rng, n=800):
    """LIME tal cual: perturbar, pesar por cercanía y ajustar una recta.

    Las perturbaciones se sortean de la distribución de los datos por columna
    (que es lo que hace la implementación tabular de referencia), y el peso es
    un kernel exponencial sobre la distancia estandarizada.
    """
    sd = X.std(0)
    Z = x + rng.standard_normal((n, len(x))) * sd
    d = np.sqrt((((Z - x) / sd) ** 2).sum(1))
    w = np.exp(-(d ** 2) / (2 * width ** 2))
    yv = model.predict(Z)
    f = Ridge(alpha=1.0).fit((Z - x) / sd, yv, sample_weight=w)
    return f.coef_


def stage_lime(X, y):
    rng = np.random.default_rng(SEED + 4)
    n = len(y)
    idx = rng.permutation(n)
    tr, te = idx[:120], idx[120:]
    m = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=0,
                              n_jobs=1).fit(X[tr], y[tr])
    x = X[te[0]]

    # El gradiente local de verdad, por diferencias centradas sobre el modelo.
    h = 0.05 * X.std(0)
    grad = np.array([(m.predict((x + np.eye(len(x))[j] * h[j])[None, :])[0]
                      - m.predict((x - np.eye(len(x))[j] * h[j])[None, :])[0])
                     / (2 * h[j]) for j in range(len(x))])

    widths = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    rows = []
    for wd in widths:
        runs = np.array([lime_explain(m, x, X[tr], wd, np.random.default_rng(SEED + 50 + k))
                         for k in range(12)])
        tops = [int(np.argmax(np.abs(rr))) for rr in runs]
        pair = [spearmanr(runs[a], runs[b]).statistic
                for a in range(len(runs)) for b in range(a + 1, len(runs))]
        rows.append(dict(width=wd, top=int(np.bincount(tops).argmax()),
                         top_share=float(np.bincount(tops).max() / len(tops)),
                         distinct_tops=int(len(set(tops))),
                         stability=float(np.mean(pair)),
                         grad_corr=float(spearmanr(runs.mean(0), grad).statistic),
                         coef=r(runs.mean(0), 5)))
    tops = [row["top"] for row in rows]
    return dict(rows=rows, grad=r(grad, 5), flips=int(len(set(tops))),
                widths=widths, instance=int(te[0]))


def stage_two_models(X, y):
    """Dos modelos que aciertan igual y explican distinto."""
    rng = np.random.default_rng(SEED + 5)
    n = len(y)
    accs, corrs, tops = [], [], []
    for rep in range(8):
        idx = np.random.default_rng(SEED + 60 + rep).permutation(n)
        tr, te = idx[:120], idx[120:]
        back = X[tr][:N_BACK]
        a = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=0,
                                  n_jobs=1).fit(X[tr], y[tr])
        b = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=1,
                                  max_features=0.4, n_jobs=1).fit(X[tr], y[tr])
        ea = float(np.mean((y[te] - a.predict(X[te])) ** 2))
        eb = float(np.mean((y[te] - b.predict(X[te])) ** 2))
        pa, pb = [], []
        for i in te[:6]:
            pa.append(shapley_exact(interventional_value(a, X[i], back), P)[0])
            pb.append(shapley_exact(interventional_value(b, X[i], back), P)[0])
        pa, pb = np.array(pa), np.array(pb)
        accs.append((ea, eb))
        corrs.append(float(np.mean([spearmanr(u, v).statistic for u, v in zip(pa, pb)])))
        tops.append(float(np.mean([np.argmax(np.abs(u)) == np.argmax(np.abs(v))
                                   for u, v in zip(pa, pb)])))
    accs = np.array(accs)
    return dict(mse_a=float(accs[:, 0].mean()), mse_b=float(accs[:, 1].mean()),
                mse_gap=float(np.abs(accs[:, 0] - accs[:, 1]).mean()),
                mse_seed_spread=float(accs[:, 0].std(ddof=1)),
                rank_corr=float(np.mean(corrs)), rank_sd=float(np.std(corrs, ddof=1)),
                same_top=float(np.mean(tops)), reps=8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Xall, y, names, wine = wine_data()
    cols = top_columns(Xall, y)
    X = Xall[:, cols]
    used = [names[j] for j in cols]
    print(f"  wine: {Xall.shape[0]} rows, predicting flavanoids from {P} of 12 columns")
    print(f"    {', '.join(used)}")

    key = cached("key", lambda: stage_answer_key(X, y, used))
    print(f"  additive answer key: worst |exact - closed form| {key['worst_key']:.3e}, "
          f"worst efficiency gap {key['worst_eff']:.3e}, r2 {key['r2']:.4f}")

    inter = cached("interaction", lambda: stage_interaction(X, y))
    print(f"  interaction: worst |exact - closed form| {inter['worst']:.3e}, "
          f"worst gap between the two halves {inter['halves']:.3e}")

    ke = cached("kernel", lambda: stage_kernel_error(X, y))
    for row in ke["rows"]:
        print(f"    {row['m']:>5} coalitions  max error {row['err']:.4f}  "
              f"({100 * row['relative']:.2f}% of scale)  top-1 agrees {row['top1']:.3f}")

    two = cached("two_shaps", lambda: stage_two_shaps(X, y, names, cols))
    print(f"  interventional vs conditional: mean gap {two['mean_gap']:.4f}, "
          f"max {two['max_gap']:.4f}")
    print(f"    the column the model barely uses ({two['unused_name']}, corr "
          f"{two['unused_corr']:+.3f} with {two['partner_name']}): "
          f"{two['unused_interventional']:.4f} interventional, "
          f"{two['unused_conditional']:.4f} conditional")

    lime = cached("lime", lambda: stage_lime(X, y))
    for row in lime["rows"]:
        print(f"    width {row['width']:>4}: top column {used[row['top']]:<22} "
              f"({row['top_share']:.2f} of runs)  stability {row['stability']:.3f}  "
              f"vs local gradient {row['grad_corr']:+.3f}")

    models = cached("models", lambda: stage_two_models(X, y))
    print(f"  two forests: mse {models['mse_a']:.4f} and {models['mse_b']:.4f} "
          f"(seed spread {models['mse_seed_spread']:.4f}), attribution rank "
          f"correlation {models['rank_corr']:.3f}, same top feature "
          f"{models['same_top']:.3f}")

    assert key["worst_key"] < 1e-9, "la enumeración no reproduce la clave de respuestas"
    assert key["worst_eff"] < 1e-9, "la identidad de eficiencia no se cumple"
    assert inter["worst"] < 1e-9, "la forma cerrada de la interacción no cuadra"
    assert inter["halves"] < 1e-9, "la interacción no se reparte a medias"
    assert ke["rows"][0]["err"] > 5 * ke["rows"][-1]["err"], \
        "muestrear más no mejora, así que la curva no dice nada"
    assert lime["flips"] > 1, "el ancho del kernel no cambia la explicación"
    assert two["max_gap"] > 0.05, "las dos variantes de SHAP coinciden aquí"

    data = dict(
        meta=dict(seed=SEED, p=P, background=N_BACK, n_show=N_SHOW,
                  rows=int(Xall.shape[0]), columns=used, target="flavanoids",
                  all_names=names,
                  note="exact Shapley by enumerating all 1024 coalitions; the "
                       "value function is interventional unless a row says otherwise"),
        answer_key=dict(worst=r(key["worst_key"], 12), worst_eff=r(key["worst_eff"], 12),
                        base=r(key["base"]), r2=r(key["r2"]), n=key["n_show"],
                        rows=[dict(i=row["i"], phi=r(row["phi"]), key=r(row["key"]),
                                   pred=r(row["pred"])) for row in key["rows"][:12]]),
        interaction=dict(worst=r(inter["worst"], 12), halves=r(inter["halves"], 12),
                         a=inter["a"], b=inter["b"], c=inter["c"], m01=r(inter["m01"]),
                         rows=[{k: r(v) for k, v in row.items()} for row in inter["rows"]]),
        kernel=dict(scale=r(ke["scale"]), instances=ke["instances"], model=ke["model"],
                    rows=[{k: r(v) for k, v in row.items()} for row in ke["rows"]],
                    exact=r(ke["exact"])),
        two_shaps=two,
        lime=dict(instance=lime["instance"], flips=lime["flips"], widths=lime["widths"],
                  grad=lime["grad"],
                  rows=[{k: (v if isinstance(v, list) else r(v))
                         for k, v in row.items()} for row in lime["rows"]]),
        models={k: r(v) for k, v in models.items()},
    )
    path = OUT / "attribution.json"
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
