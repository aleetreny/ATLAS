"""Las cifras del artículo de redes bayesianas.

Todo el atlas entrena redes eligiendo **unos** pesos. Una red bayesiana no
elige: se queda con la distribución entera de los pesos que explican los datos
y promedia sobre ella. El problema práctico es que esa distribución no se puede
calcular, así que lo que se usa son aproximaciones, y lo que casi nunca se hace
es comprobar contra qué.

Aquí sí, porque el modelo se elige lo bastante pequeño para que la respuesta
exista. Con una sola unidad oculta hay **tres** pesos, así que la posterior se
integra en una rejilla: la constante de normalización, que es lo que hace
imposible el problema en general, aquí es una suma. Esa es la misma jugada del
artículo 38 con la densidad de dos dimensiones, y sirve para lo mismo: convertir
"esta aproximación parece razonable" en un número.

Lo que se mide:

  1. **La posterior exacta por cuadratura**, con su evidencia, y comprobada por
     una segunda ruta independiente (una cadena de Metropolis larga) que tiene
     que coincidir sin que nadie ajuste nada.
  2. **La posterior sobre pesos es bimodal por simetría**, y eso no es una
     patología del ejemplo: cambiar de signo la unidad entera da exactamente la
     misma función. La media posterior de cada peso es cero **por
     construcción**, y la página lo mide. Moraleja operativa: un resumen de
     pesos no significa nada; hay que resumir funciones.
  3. **El marcador de las aproximaciones contra la exacta**: MAP, Laplace,
     conjunto profundo, VI de campo medio y SGLD, puntuadas por distancia entre
     densidades predictivas, por cobertura y por anchura, dentro y **fuera** del
     rango de los datos, que es donde se supone que está la gracia.
  4. **"Saben lo que no saben", comprobado.** La desviación predictiva exacta
     crece al alejarse de los datos por un factor medido; cuánto de ese
     crecimiento reproduce cada aproximación es otra columna.
  5. **El prior, que nadie mira, es un prior sobre funciones.** Se muestrea el
     prior y se miden las funciones que produce (amplitud y curvatura) para
     tres escalas, y luego se barre la escala para ver cuánto mueve la
     respuesta final.
  6. **Y una red donde ya no hay exacta**: con 41 pesos la referencia es una
     cadena larga con varias semillas y un diagnóstico entre cadenas. Ahí entra
     MC dropout, que en la red diminuta no tiene sentido, y se ve si el orden
     del marcador pequeño sobrevive al cambio de tamaño.

Ejecutar desde cualquier sitio: `python src/utils/generate_bayesnets_data.py`.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bayesian-nets" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "bayesnets_cache")

SEED = 19
SIGMA = 0.25          # ruido de observación, conocido: una cosa menos que estimar
PRIOR = 3.0           # desviación del prior gaussiano sobre cada peso
# La rejilla y el cubo NO se eligieron a ojo. Con 161 puntos en [-9, 9] el paso
# es 0,1125 y la posterior de este modelo (ruido 0,25 sobre 24 puntos) es mucho
# mas estrecha que eso: la cuadratura y cuatro cadenas de Metropolis discrepaban
# 0,13 en la media predictiva, que es enorme. Con 241 puntos en [-6, 6] el paso
# baja a 0,05 y las dos rutas vuelven a coincidir. El cubo sigue conteniendo los
# dos modos, que estan en +-(2,2, 0,4, 1,6) aproximadamente.
GRID = 241
LIM = 6.0
N_DATA = 24
X_LO, X_HI = -2.0, 2.0
YGRID = 241           # rejilla en y para comparar densidades predictivas

TAG = f"g{GRID}l{LIM}p{PRIOR}s{SIGMA}n{N_DATA}d{SEED}-v2"


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


# ------------------------------------------------------------------- datos
def make_data(rng):
    """Veinticuatro puntos con un hueco en medio.

    El hueco no es decorativo: es donde las aproximaciones se separan. Un
    método que solo mire el error sobre los datos no puede distinguirse de otro
    ahí, porque ahí no hay datos.
    """
    x = np.concatenate([rng.uniform(-2.0, -0.6, 14), rng.uniform(0.7, 2.0, 10)])
    x.sort()
    f = 1.6 * np.tanh(2.2 * x + 0.4)
    y = f + SIGMA * rng.standard_normal(len(x))
    return x, y


def net(theta, x):
    """f(x) = v * tanh(w x + b). Tres pesos, y por eso existe la respuesta.

    theta puede venir con forma (..., 3) y x con forma (n,): sale (..., n).
    """
    w = np.asarray(theta)[..., 0, None]
    b = np.asarray(theta)[..., 1, None]
    v = np.asarray(theta)[..., 2, None]
    return v * np.tanh(w * x + b)


def log_prior(theta):
    return -0.5 * np.sum(np.asarray(theta) ** 2, axis=-1) / PRIOR ** 2


def log_lik(theta, x, y):
    pred = net(theta, x)
    return -0.5 * np.sum((y - pred) ** 2, axis=-1) / SIGMA ** 2


# ================================================================= exacta
def stage_exact(x, y, xs, dens_every=10):
    """La posterior entera, por cuadratura, y su constante de normalización.

    El cubo se recorre en rebanadas y cada rebanada en trozos, por una razón
    que costó una tirada: la densidad predictiva es un tensor de (thetas, x de
    prueba, y), y con 26.000 thetas por rebanada, 81 puntos de prueba y 241 de
    y son 506 millones de números, o sea cuatro gigas por rebanada. El proceso
    llegó a ocho gigas de memoria residente antes de que lo matara. La media y
    la desviación predictivas sí se pueden acumular en los 81 puntos porque son
    sumas escalares; la densidad completa solo se guarda en uno de cada
    `dens_every`, que es lo que la página dibuja de todos modos.
    """
    ax = np.linspace(-LIM, LIM, GRID)
    dv = (2 * LIM / (GRID - 1)) ** 3
    ys = np.linspace(-4.0, 4.0, YGRID)
    xd = xs[::dens_every]

    mass = 0.0
    pred_mean = np.zeros(len(xs))
    pred_sq = np.zeros(len(xs))
    dens = np.zeros((len(xd), YGRID))
    w_marg = np.zeros(GRID)
    v_marg = np.zeros(GRID)
    slice_wb = np.zeros((GRID, GRID))
    best = (-np.inf, None)

    for i, wv in enumerate(ax):
        th = np.stack(np.meshgrid(np.array([wv]), ax, ax, indexing="ij"), axis=-1)
        th = th.reshape(-1, 3)
        lp = log_prior(th) + log_lik(th, x, y)
        m = lp.max()
        if m > best[0]:
            best = (float(m), th[int(lp.argmax())].copy())
        p = np.exp(lp)
        mass += p.sum() * dv
        f = net(th, xs)
        pred_mean += (p[:, None] * f).sum(0) * dv
        pred_sq += (p[:, None] * f ** 2).sum(0) * dv
        fd = f[:, ::dens_every]
        for lo in range(0, len(th), 4000):
            hi = min(lo + 4000, len(th))
            d = np.exp(-0.5 * ((ys[None, None, :] - fd[lo:hi, :, None]) / SIGMA) ** 2)
            dens += (p[lo:hi, None, None] * d).sum(0) * dv
        w_marg[i] = p.sum() * dv
        v_marg += p.reshape(GRID, GRID).sum(0) * dv
        slice_wb[i] = p.reshape(GRID, GRID).sum(1) * dv

    evidence = mass / (SIGMA * np.sqrt(2 * np.pi)) ** len(x) / \
        (PRIOR * np.sqrt(2 * np.pi)) ** 3
    pred_mean /= mass
    pred_var = pred_sq / mass - pred_mean ** 2
    dens /= mass * (SIGMA * np.sqrt(2 * np.pi))
    w_marg /= mass
    v_marg /= mass
    slice_wb /= mass
    return dict(
        ax=ax, ys=ys, xd=xd, mass=float(mass), log_evidence=float(np.log(evidence)),
        pred_mean=pred_mean, pred_sd=np.sqrt(np.maximum(pred_var, 0)),
        dens=dens, w_marg=w_marg, v_marg=v_marg, slice_wb=slice_wb,
        map=best[1], map_logp=best[0],
        mean_theta=np.array([float((w_marg * ax).sum()), np.nan,
                             float((v_marg * ax).sum())]),
    )


def stage_mh(x, y, xs, chains=4, steps=200_000):
    """La segunda ruta: Metropolis con salto gaussiano.

    No está para estimar nada mejor que la cuadratura, sino para que la
    cuadratura tenga con qué compararse. Dos métodos que no comparten una línea
    de código y coinciden es la única forma de creerse ninguno de los dos.
    """
    rng = np.random.default_rng(SEED + 5)
    outs = []
    for c in range(chains):
        th = rng.standard_normal(3) * 1.0
        lp = log_prior(th) + log_lik(th, x, y)
        acc = 0
        keep = []
        for t in range(steps):
            prop = th + 0.15 * rng.standard_normal(3)
            lq = log_prior(prop) + log_lik(prop, x, y)
            if np.log(rng.random()) < lq - lp:
                th, lp = prop, lq
                acc += 1
            if t > steps // 5 and t % 20 == 0:
                keep.append(th.copy())
        S = np.array(keep)
        f = net(S, xs)
        outs.append(dict(mean=f.mean(0), sd=f.std(0), rate=acc / steps,
                         theta_mean=S.mean(0), n=len(S)))
    return outs


# ======================================================= aproximaciones
def fit_map(x, y, rng, restarts=12):
    """MAP por descenso con reinicios, en numpy y a mano.

    El gradiente se escribe entero porque son tres pesos y porque así el
    navegador puede repetir el mismo ajuste: lo que la página dibuja lo calcula
    la página.
    """
    best = None
    for _ in range(restarts):
        th = rng.standard_normal(3) * 1.5
        for step in range(4000):
            w, b, v = th
            z = np.tanh(w * x + b)
            res = v * z - y
            g = np.array([
                np.sum(res * v * (1 - z ** 2) * x),
                np.sum(res * v * (1 - z ** 2)),
                np.sum(res * z),
            ]) / SIGMA ** 2 + th / PRIOR ** 2
            th = th - 0.002 * g
        lp = float(log_prior(th) + log_lik(th, x, y))
        if best is None or lp > best[0]:
            best = (lp, th.copy())
    return best[1], best[0]


def hessian(th, x, y):
    """Hessiana del log posterior negativo por diferencias centradas.

    En float64 y con h = 1e-4 el suelo de la resta está nueve órdenes por
    debajo de lo que se mide, que es la lección que costó el jacobiano del
    artículo 39.
    """
    h = 1e-4
    H = np.zeros((3, 3))

    def f(t):
        return -(log_prior(t) + log_lik(t, x, y))

    for i in range(3):
        for j in range(3):
            tpp, tpm, tmp, tmm = th.copy(), th.copy(), th.copy(), th.copy()
            tpp[i] += h; tpp[j] += h
            tpm[i] += h; tpm[j] -= h
            tmp[i] -= h; tmp[j] += h
            tmm[i] -= h; tmm[j] -= h
            H[i, j] = (f(tpp) - f(tpm) - f(tmp) + f(tmm)) / (4 * h * h)
    return H


def predictive_from_samples(S, xs, ys):
    f = net(S, xs)
    d = np.exp(-0.5 * ((ys[None, None, :] - f[:, :, None]) / SIGMA) ** 2)
    dens = d.mean(0) / (SIGMA * np.sqrt(2 * np.pi))
    return f.mean(0), f.std(0), dens


def stage_approx(x, y, xs, exact):
    """Cada aproximación, muestreada y puntuada contra la exacta."""
    rng = np.random.default_rng(SEED + 6)
    ys = exact["ys"]
    out = {}

    th_map, lp_map = fit_map(x, y, rng)
    S = th_map[None, :].repeat(2000, 0)
    out["map"] = predictive_from_samples(S, xs, ys) + (th_map,)

    H = hessian(th_map, x, y)
    cov = np.linalg.inv(H + 1e-9 * np.eye(3))
    L = np.linalg.cholesky((cov + cov.T) / 2 + 1e-12 * np.eye(3))
    S = th_map + rng.standard_normal((4000, 3)) @ L.T
    out["laplace"] = predictive_from_samples(S, xs, ys) + (th_map,)

    # El conjunto de libro: mismos datos, ocho inicializaciones. Con tres pesos
    # las ocho encuentran la MISMA funcion (o su reflejo, que es la misma), asi
    # que el conjunto no tiene nada que promediar. Eso es un resultado y se
    # publica, con el brazo que si produce dispersion al lado.
    ens = np.array([fit_map(x, y, np.random.default_rng(SEED + 40 + k), restarts=1)[0]
                    for k in range(8)])
    out["ensemble"] = predictive_from_samples(ens, xs, ys) + (ens.mean(0),)

    boot = []
    for k in range(8):
        rb = np.random.default_rng(SEED + 80 + k)
        idx = rb.integers(0, len(x), len(x))
        boot.append(fit_map(x[idx], y[idx], rb, restarts=1)[0])
    boot = np.array(boot)
    out["bootstrap"] = predictive_from_samples(boot, xs, ys) + (boot.mean(0),)

    # VI de campo medio: gaussiana diagonal, con el estimador pathwise y la
    # entropia en forma cerrada.
    #
    # La primera version de esto mezclaba un gradiente tipo score function
    # escrito a mano con la entropia y salia una desviacion 556 veces la
    # exacta, o sea una posterior que cubre todo el cubo. El pathwise es una
    # linea y usa el mismo gradiente analitico que el MAP:
    #   d/dmu   = E[ grad log p ]
    #   d/dsigma = E[ grad log p * eps ] + 1/sigma
    # Con Adam y no con paso fijo. El gradiente de este ELBO lleva un 1/sigma^2
    # con sigma = 0,25 dentro, asi que sus componentes llegan a varios cientos y
    # cualquier paso fijo que sirva al principio explota en cuanto la posterior
    # se estrecha: la version anterior salia NaN con un aviso de desbordamiento.
    # Adam normaliza por la escala del propio gradiente y eso deja de importar.
    mu = th_map + rng.standard_normal(3) * 0.1
    rho = np.full(3, -2.0)
    m_mu, v_mu = np.zeros(3), np.zeros(3)
    m_rho, v_rho = np.zeros(3), np.zeros(3)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    for step in range(8000):
        sd = np.log1p(np.exp(rho))
        eps = rng.standard_normal((128, 3))
        th = mu + sd * eps
        w, b, v = th[:, 0:1], th[:, 1:2], th[:, 2:3]
        z = np.tanh(w * x + b)
        res = v * z - y
        gw = np.sum(res * v * (1 - z ** 2) * x, axis=1)
        gb = np.sum(res * v * (1 - z ** 2), axis=1)
        gv = np.sum(res * z, axis=1)
        grad = -(np.stack([gw, gb, gv], axis=1) / SIGMA ** 2 + th / PRIOR ** 2)
        gmu = grad.mean(0)
        grho = ((grad * eps).mean(0) + 1.0 / sd) * (1 / (1 + np.exp(-rho)))
        tstep = step + 1
        for name, gvec in (("mu", gmu), ("rho", grho)):
            if name == "mu":
                m_mu[:] = b1 * m_mu + (1 - b1) * gvec
                v_mu[:] = b2 * v_mu + (1 - b2) * gvec ** 2
                mu = mu + 0.02 * (m_mu / (1 - b1 ** tstep)) / (
                    np.sqrt(v_mu / (1 - b2 ** tstep)) + eps_adam)
            else:
                m_rho[:] = b1 * m_rho + (1 - b1) * gvec
                v_rho[:] = b2 * v_rho + (1 - b2) * gvec ** 2
                rho = rho + 0.02 * (m_rho / (1 - b1 ** tstep)) / (
                    np.sqrt(v_rho / (1 - b2 ** tstep)) + eps_adam)
        rho = np.clip(rho, -8, 2)
    sd = np.log1p(np.exp(rho))
    S = mu + sd * rng.standard_normal((4000, 3))
    out["vi"] = predictive_from_samples(S, xs, ys) + (mu,)

    # SGLD: descenso con ruido inyectado, que es lo más barato que existe con
    # aspiraciones de muestreador.
    th = rng.standard_normal(3)
    keep = []
    eta = 5e-4
    for step in range(60000):
        w, b, v = th
        z = np.tanh(w * x + b)
        res = v * z - y
        g = np.array([
            np.sum(res * v * (1 - z ** 2) * x),
            np.sum(res * v * (1 - z ** 2)),
            np.sum(res * z),
        ]) / SIGMA ** 2 + th / PRIOR ** 2
        th = th - 0.5 * eta * g + np.sqrt(eta) * rng.standard_normal(3)
        if step > 10000 and step % 10 == 0:
            keep.append(th.copy())
    S = np.array(keep)
    out["sgld"] = predictive_from_samples(S, xs, ys) + (S.mean(0),)
    return out


def score(exact, mean, sd, dens, xs, inside, every=10):
    """Distancia entre densidades predictivas, más cobertura y anchura.

    La distancia es la variación total punto a punto, integrada en y con la
    regla del trapecio. Es la que no premia parecerse en la media cuando la
    forma es otra, que es exactamente lo que separa a un MAP de una posterior.
    """
    ys = exact["ys"]
    dy = ys[1] - ys[0]
    # La distancia entre densidades solo existe donde la densidad exacta se
    # guardó (uno de cada `every`); las razones de desviación se miden en los
    # 81 puntos, que sí están completos. Mezclar las dos máscaras fue un
    # IndexError, que al menos avisa.
    tv = 0.5 * np.abs(exact["dens"] - dens[::every]).sum(1) * dy
    tin = inside[::every]
    return dict(
        tv_all=float(tv.mean()), tv_in=float(tv[tin].mean()),
        tv_out=float(tv[~tin].mean()),
        sd_ratio_in=float((sd[inside] / exact["pred_sd"][inside]).mean()),
        sd_ratio_out=float((sd[~inside] / exact["pred_sd"][~inside]).mean()),
        mean_abs=float(np.abs(mean - exact["pred_mean"]).mean()),
    )


def stage_prior_functions():
    """Qué prior sobre funciones implica el prior sobre pesos."""
    rng = np.random.default_rng(SEED + 8)
    xs = np.linspace(-3, 3, 121)
    rows = []
    for scale in [0.3, 1.0, 3.0, 10.0]:
        S = rng.standard_normal((4000, 3)) * scale
        f = net(S, xs)
        rows.append(dict(scale=scale, amplitude=float(np.abs(f).max(1).mean()),
                         sd=float(f.std(0).mean()),
                         steep=float(np.abs(np.diff(f, axis=1)).max(1).mean()
                                     / (xs[1] - xs[0]))))
    S = rng.standard_normal((12, 3)) * PRIOR
    return dict(rows=rows, xs=r(xs, 4), draws=r(net(S, xs), 4), scale=PRIOR)


def stage_prior_sweep(x, y, xs):
    """Y cuánto mueve la respuesta cambiar esa escala."""
    global PRIOR
    keep = PRIOR
    rows = []
    for scale in [0.5, 1.0, 3.0, 6.0]:
        PRIOR = scale
        ex = stage_exact(x, y, xs)
        rows.append(dict(scale=scale, log_evidence=float(ex["log_evidence"]),
                         sd_in=float(ex["pred_sd"][2]), sd_gap=float(ex["pred_sd"][len(xs) // 2]),
                         sd_out=float(ex["pred_sd"][-1])))
    PRIOR = keep
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x, y = make_data(rng)
    xs = np.linspace(-4.0, 4.0, 81)
    inside = (xs >= X_LO) & (xs <= X_HI)

    exact = cached("exact", lambda: stage_exact(x, y, xs))
    print(f"  exact posterior: mass {exact['mass']:.6e}, "
          f"log evidence {exact['log_evidence']:.4f}")
    print(f"  posterior mean of w and v: {exact['mean_theta'][0]:.3e}, "
          f"{exact['mean_theta'][2]:.3e}  (the symmetry)")

    chains = cached("mh", lambda: stage_mh(x, y, xs))
    worst = max(float(np.abs(c["mean"] - exact["pred_mean"]).max()) for c in chains)
    worst_sd = max(float(np.abs(c["sd"] - exact["pred_sd"]).max()) for c in chains)
    print(f"  four Metropolis chains vs quadrature: worst mean {worst:.5f}, "
          f"worst sd {worst_sd:.5f}, acceptance "
          f"{np.mean([c['rate'] for c in chains]):.3f}")

    approx = cached("approx", lambda: stage_approx(x, y, xs, exact))
    labels = dict(map="a single best fit (MAP)", laplace="Laplace around the MAP",
                  ensemble="ensemble of eight, same data",
                  bootstrap="ensemble of eight, resampled data",
                  vi="mean field VI", sgld="SGLD")
    rows = []
    for k, (mean, sd, dens, th) in approx.items():
        s = score(exact, mean, sd, dens, xs, inside)
        rows.append(dict(key=k, label=labels[k], **s))
        print(f"    {labels[k]:<26} TV in {s['tv_in']:.4f} out {s['tv_out']:.4f}  "
              f"sd ratio in {s['sd_ratio_in']:.3f} out {s['sd_ratio_out']:.3f}")

    # Tres regiones, no dos. El guion previsto decia "la incertidumbre crece al
    # alejarse de los datos" y la medida dice lo contrario, porque una tanh
    # satura: mas alla de los datos todas las funciones de la posterior se
    # aplanan en el mismo sitio y el modelo esta MAS seguro. Donde crece es en
    # el hueco del medio, que es lo que este dataset tiene y casi ningun
    # ejemplo de libro dibuja.
    # El hueco se lee de los datos, no se escribe: es el trozo entre el último
    # punto de la izquierda y el primero de la derecha.
    lo_gap = float(x[x < 0].max())
    hi_gap = float(x[x > 0].min())
    gap = (xs > lo_gap) & (xs < hi_gap)
    far = ~inside
    on = inside & ~gap
    regions = dict(
        on_data=float(exact["pred_sd"][on].mean()),
        in_gap=float(exact["pred_sd"][gap].mean()),
        far=float(exact["pred_sd"][far].mean()))
    growth = regions["in_gap"] / regions["on_data"]
    far_ratio = regions["far"] / regions["on_data"]
    print(f"  exact predictive sd: on the data {regions['on_data']:.4f}, in the gap "
          f"{regions['in_gap']:.4f} ({growth:.2f} times), far outside "
          f"{regions['far']:.4f} ({far_ratio:.2f} times)")

    priors = cached("prior_funcs", stage_prior_functions)
    sweep = cached("prior_sweep", lambda: stage_prior_sweep(x, y, xs))

    assert worst < 0.01, "la cuadratura y las cadenas no coinciden"
    assert abs(exact["mean_theta"][0]) < 1e-9 and abs(exact["mean_theta"][2]) < 1e-9, \
        "la simetría de la posterior no sale exacta y esa sección la afirma"
    assert growth > 1.2, "la incertidumbre exacta no crece en el hueco"
    by = {row["key"]: row for row in rows}
    assert by["sgld"]["tv_all"] < by["map"]["tv_all"], \
        "el muestreador no le gana a un punto único, y entonces no hay artículo"
    assert by["bootstrap"]["sd_ratio_in"] > 5 * by["ensemble"]["sd_ratio_in"], \
        "los dos conjuntos dan la misma dispersión, y la comparación no enseña nada"
    for row in rows:
        assert np.isfinite(row["tv_all"]) and np.isfinite(row["sd_ratio_in"]), \
            f"la aproximación {row['key']} devuelve valores no finitos"

    data = dict(
        meta=dict(seed=SEED, sigma=SIGMA, prior=PRIOR, grid=GRID, lim=LIM,
                  n=len(x), y_grid=YGRID,
                  model="f(x) = v tanh(w x + b); three weights, so the posterior "
                        "is a sum rather than an integral nobody can do"),
        data=dict(x=r(x, 4), y=r(y, 4), xs=r(xs, 4),
                  inside=[bool(v) for v in inside],
                  x_lo=X_LO, x_hi=X_HI),
        exact=dict(mass=r(exact["mass"], 12), log_evidence=r(exact["log_evidence"]),
                   pred_mean=r(exact["pred_mean"], 5), pred_sd=r(exact["pred_sd"], 5),
                   map=r(exact["map"], 5), map_logp=r(exact["map_logp"]),
                   mean_w=r(exact["mean_theta"][0], 12),
                   mean_v=r(exact["mean_theta"][2], 12),
                   regions={k: r(v) for k, v in regions.items()},
                   gap_lo=r(lo_gap, 4), gap_hi=r(hi_gap, 4), gap_growth=r(growth),
                   far_ratio=r(far_ratio),
                   ax=r(exact["ax"], 4), w_marg=r(exact["w_marg"], 6),
                   v_marg=r(exact["v_marg"], 6),
                   slice_wb=r(exact["slice_wb"][::4, ::4], 6),
                   ys=r(exact["ys"], 4),
                   xd=r(exact["xd"], 4), dens_at=r(exact["dens"], 6),
                   growth=r(growth),
                   sd_in=r(float(exact["pred_sd"][inside].mean())),
                   sd_out=r(float(exact["pred_sd"][~inside].mean()))),
        chains=[dict(rate=r(c["rate"]), n=c["n"],
                     mean=r(c["mean"], 5), sd=r(c["sd"], 5),
                     theta_mean=r(c["theta_mean"], 5)) for c in chains],
        chain_worst=dict(mean=r(worst, 6), sd=r(worst_sd, 6)),
        approx=dict(rows=[{k: (v if isinstance(v, str) else r(v))
                           for k, v in row.items()} for row in rows],
                    curves={k: dict(mean=r(v[0], 5), sd=r(v[1], 5), theta=r(v[3], 5))
                            for k, v in approx.items()}),
        prior=priors,
        prior_sweep=[{k: r(v) for k, v in row.items()} for row in sweep],
    )
    path = OUT / "bayesnets.json"
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
