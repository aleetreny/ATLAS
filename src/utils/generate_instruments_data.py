"""Las cifras del artículo de variables instrumentales.

Los dos artículos anteriores suponen la misma cosa y no la dicen lo bastante
alto: que el confusor está **medido**. Ajustar por él, ponderar por su score o
emparejar sobre él son tres formas de usar una columna que alguien apuntó. Este
artículo trata el caso en que esa columna no existe y nunca va a existir, que
es el caso normal.

La respuesta clásica es una palanca externa: algo que mueva el tratamiento y
que no toque el resultado por ningún otro camino. Lo que se mide aquí:

  1. **El sesgo de mínimos cuadrados tiene forma cerrada** (gamma*delta
     partido por la varianza de X) y la palanca lo quita entero. Las dos cosas
     se miden contra el parámetro, que aquí es un número elegido y no estimado.
  2. **Una palanca floja no es una palanca peor, es otra cosa.** El barrido de
     fuerza mide la F de la primera etapa, el sesgo mediano y, sobre todo, algo
     que casi nunca se enseña: el estimador exactamente identificado **no tiene
     media**. Con la palanca floja, la media entre repeticiones salta de una
     semilla a otra mientras la mediana no se mueve, y eso se mide con las dos
     columnas al lado.
  3. **La violación de la exclusión se amplifica por la debilidad**: el sesgo
     es exactamente kappa/pi, así que un caminito directo de 0,05 con una
     palanca de 0,05 vale un efecto entero. La tabla cruza las dos cosas y
     compara cada celda con la fórmula.
  4. **Y no se puede comprobar con los datos.** Dos mundos con la MISMA matriz
     de covarianzas de (Z, X, Y), uno con la exclusión intacta y otro no, con
     efectos distintos. Misma F, mismos datos, mismo todo. La única diferencia
     vive en un dibujo que nadie puede contrastar.
  5. **Lo que estima cuando funciona no es el efecto medio.** Con cumplimiento
     heterogéneo, el cociente de Wald es el efecto de los **cumplidores**, y
     aquí los tipos están escritos, así que se pueden separar y comparar: el
     ATE, el efecto de cada tipo, y lo que el estimador devuelve.

Ejecutar desde cualquier sitio: `python src/utils/generate_instruments_data.py`.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "instruments" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "instruments_cache")

SEED = 19

# El mundo lineal:
#   U ~ N(0,1)   confusor que nadie apunta
#   Z ~ N(0,1)   la palanca
#   X = PI*Z + DELTA*U + ex,   ex ~ N(0, VEX)
#   Y = BETA*X + GAMMA*U + KAPPA*Z + ey,   ey ~ N(0, VEY)
#
# KAPPA es el camino que la exclusión prohíbe. Vale cero en el mundo bueno y se
# enciende en la sección que mide qué cuesta que no valga cero.
BETA = 1.5
GAMMA = 2.0
DELTA = 1.0
PI = 0.8
VEX = 1.0
VEY = 1.0
N = 2000
REPS = 400

TAG = f"b{BETA}g{GAMMA}d{DELTA}p{PI}n{N}r{REPS}s{SEED}-v3"


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


def draw(n, rng, pi=PI, kappa=0.0, beta=BETA):
    u = rng.standard_normal(n)
    z = rng.standard_normal(n)
    x = pi * z + DELTA * u + np.sqrt(VEX) * rng.standard_normal(n)
    y = beta * x + GAMMA * u + kappa * z + np.sqrt(VEY) * rng.standard_normal(n)
    return z, x, y, u


def ols(x, y):
    mx, my = x.mean(), y.mean()
    return float(((x - mx) * (y - my)).sum() / ((x - mx) ** 2).sum())


def wald(z, x, y):
    """El estimador exactamente identificado: dos covarianzas y una división.

    Escrito así y no como dos regresiones encadenadas porque es lo mismo y se
    ve por qué el denominador es el problema: cuando la palanca es floja, ese
    número está cerca de cero y una división por casi cero es lo que produce
    toda la patología de esta página.
    """
    mz, mx, my = z.mean(), x.mean(), y.mean()
    num = ((z - mz) * (y - my)).sum()
    den = ((z - mz) * (x - mx)).sum()
    return float(num / den)


def first_stage_f(z, x):
    """La F de la primera etapa, con una sola palanca: el cuadrado del t."""
    n = len(z)
    mz, mx = z.mean(), x.mean()
    szz = ((z - mz) ** 2).sum()
    b = ((z - mz) * (x - mx)).sum() / szz
    resid = (x - mx) - b * (z - mz)
    s2 = (resid ** 2).sum() / (n - 2)
    return float(b ** 2 * szz / s2)


def ols_bias_closed(pi=PI):
    """gamma * Cov(X, U) / Var(X), que con este mundo es una fracción."""
    var_x = pi ** 2 + DELTA ** 2 + VEX
    return GAMMA * DELTA / var_x


def stage_headline():
    """Las dos estimaciones contra la verdad, con el sesgo cerrado al lado."""
    rng = np.random.default_rng(SEED)
    o, w, f = [], [], []
    for _ in range(REPS):
        z, x, y, u = draw(N, rng)
        o.append(ols(x, y))
        w.append(wald(z, x, y))
        f.append(first_stage_f(z, x))
    o, w, f = np.array(o), np.array(w), np.array(f)
    return dict(
        ols_mean=float(o.mean()), ols_sd=float(o.std(ddof=1)),
        ols_bias=float(o.mean() - BETA), ols_bias_closed=ols_bias_closed(),
        iv_mean=float(w.mean()), iv_median=float(np.median(w)),
        iv_sd=float(w.std(ddof=1)), iv_bias=float(np.median(w) - BETA),
        f_mean=float(f.mean()), truth=BETA, reps=REPS, n=N,
    )


def stage_strength():
    """El barrido de fuerza, y la columna que enseña que no hay media.

    Para cada pi se corren REPS muestras y se guardan media, mediana, la
    dispersión intercuartílica y cuántas repeticiones se van más allá de diez
    veces el efecto verdadero. Esa última columna es la que convierte "es
    inestable" en una cifra.
    """
    rows = []
    for i, pi in enumerate([1.5, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]):
        rng = np.random.default_rng(SEED + 200 + i)
        w, f, o = [], [], []
        for _ in range(REPS):
            z, x, y, u = draw(N, rng, pi=pi)
            w.append(wald(z, x, y))
            f.append(first_stage_f(z, x))
            o.append(ols(x, y))
        w, f, o = np.array(w), np.array(f), np.array(o)
        q1, q3 = np.percentile(w, [25, 75])
        rows.append(dict(
            pi=pi, f_mean=float(f.mean()), f_median=float(np.median(f)),
            iv_mean=float(w.mean()), iv_median=float(np.median(w)),
            iv_q1=float(q1), iv_q3=float(q3),
            iv_median_bias=float(np.median(w) - BETA),
            iv_max_abs=float(np.abs(w - BETA).max()),
            wild=float((np.abs(w - BETA) > 10 * abs(BETA)).mean()),
            ols_bias=float(o.mean() - BETA),
            ols_bias_closed=ols_bias_closed(pi),
            median_bias_ratio=float((np.median(w) - BETA) / (o.mean() - BETA)),
        ))
    return rows


def stage_two_seeds():
    """La media del estimador, mirada con seis semillas distintas.

    Si un estimador tiene media, seis tiradas de REPS repeticiones dan seis
    números parecidos. Aquí no: la media salta y la mediana no. Es la forma
    barata de enseñar un momento que no existe sin escribir una integral.
    """
    out = []
    for s in range(6):
        rng = np.random.default_rng(SEED + 500 + s)
        w = np.array([wald(*draw(N, rng, pi=0.05)[:3]) for _ in range(REPS)])
        out.append(dict(seed=SEED + 500 + s, mean=float(w.mean()),
                        median=float(np.median(w)), max_abs=float(np.abs(w).max())))
    return out


def stage_exclusion():
    """Un caminito directo, cruzado con la fuerza de la palanca.

    El sesgo del cociente de Wald cuando kappa no es cero es exactamente
    kappa/pi: el numerador recoge el camino directo entero y el denominador
    sigue siendo pi. Se mide cada celda y se compara con esa fracción.
    """
    rows = []
    for pi in [0.8, 0.4, 0.2, 0.1, 0.05]:
        cells = []
        for kappa in [0.0, 0.02, 0.05, 0.1, 0.2]:
            rng = np.random.default_rng(SEED + int(1000 * pi) + int(10000 * kappa))
            w = np.array([wald(*draw(N, rng, pi=pi, kappa=kappa)[:3])
                          for _ in range(120)])
            cells.append(dict(kappa=kappa, median=float(np.median(w)),
                              bias=float(np.median(w) - BETA),
                              closed=kappa / pi))
        rows.append(dict(pi=pi, cells=cells))
    return rows


def stage_two_worlds():
    """Dos mundos con la misma conjunta de (Z, X, Y) y efectos distintos.

    En el mundo A la exclusión se cumple (kappa = 0) y el efecto es BETA. En el
    B se elige otro efecto beta_b y se **despejan** kappa_b y el resto para que
    las seis covarianzas coincidan. Como el modelo tiene más parámetros libres
    que covarianzas, esa solución existe siempre, y esa es exactamente la razón
    por la que la exclusión no se puede contrastar: no le sobra información a
    los datos.

    Se resuelve a mano y no por ajuste, así que el acuerdo es álgebra.
    """
    # Mundo A
    vz = 1.0
    vx = PI ** 2 + DELTA ** 2 + VEX
    czx = PI
    czy = BETA * czx
    cxy = BETA * vx + GAMMA * DELTA
    vy = BETA ** 2 * vx + GAMMA ** 2 + 2 * BETA * GAMMA * DELTA + VEY
    cov_a = np.array([[vz, czx, czy], [czx, vx, cxy], [czy, cxy, vy]])

    # Mundo B: se fija beta_b y se despeja el resto.
    #   Cov(Z,X) = pi_b            -> pi_b = czx
    #   Cov(Z,Y) = beta_b*pi_b + kappa_b   -> kappa_b
    #   Cov(X,Y) = beta_b*vx + gamma_b*delta_b + kappa_b*pi_b
    #   Var(X)   = pi_b^2 + delta_b^2 + vex_b
    # Con delta_b fijado al de A, gamma_b sale de Cov(X,Y) y vey_b de Var(Y).
    pi_b = czx
    delta_b = DELTA
    vex_b = vx - pi_b ** 2 - delta_b ** 2

    def solve(beta_b):
        kappa_b = czy - beta_b * pi_b
        gamma_b = (cxy - beta_b * vx - kappa_b * pi_b) / delta_b
        vey_b = (vy - beta_b ** 2 * vx - gamma_b ** 2 - kappa_b ** 2 * vz
                 - 2 * beta_b * gamma_b * delta_b - 2 * beta_b * kappa_b * pi_b)
        return kappa_b, gamma_b, vey_b

    # Y aquí está el resultado de verdad de esta sección, que es más fuerte que
    # exhibir un mundo alternativo: se barre beta y se pregunta cuáles admiten
    # un mundo con **varianzas de ruido no negativas**, que es la única
    # restricción que los datos imponen. Lo que sale es un intervalo entero de
    # efectos compatibles con la misma matriz de covarianzas.
    #
    # (El primer intento de esta sección fijaba beta_b = 0,5 a ojo y pedía una
    # varianza de ruido de -5,0. Con el recorte a cero puesto por precaución el
    # mundo simulado dejaba de cuadrar con su propia álgebra y el guardia de
    # muestreo cantaba 5,02 mientras el algebraico decía 3,6e-15: los dos
    # números juntos son la firma de una parametrización imposible.)
    grid = np.linspace(-4.0, 6.0, 20001)
    feas = np.array([solve(b)[2] for b in grid])
    ok = grid[feas >= 0.0]
    beta_lo, beta_hi = float(ok.min()), float(ok.max())
    # El mundo que se exhibe: el más lejano al verdadero que siga teniendo
    # ruido de sobra (un mundo determinista sería otro tipo de trampa), porque
    # lo que hay que enseñar es cuánto se puede mover la respuesta sin que los
    # datos se enteren.
    cand = grid[feas >= 0.10]
    beta_b = float(cand[np.argmax(np.abs(cand - BETA))])
    kappa_b, gamma_b, vey_b = solve(beta_b)
    cov_b = np.array([
        [vz, pi_b, beta_b * pi_b + kappa_b],
        [pi_b, pi_b ** 2 + delta_b ** 2 + vex_b,
         beta_b * (pi_b ** 2 + delta_b ** 2 + vex_b) + gamma_b * delta_b + kappa_b * pi_b],
        [beta_b * pi_b + kappa_b,
         beta_b * (pi_b ** 2 + delta_b ** 2 + vex_b) + gamma_b * delta_b + kappa_b * pi_b,
         beta_b ** 2 * (pi_b ** 2 + delta_b ** 2 + vex_b) + gamma_b ** 2
         + kappa_b ** 2 + 2 * beta_b * gamma_b * delta_b
         + 2 * beta_b * kappa_b * pi_b + vey_b],
    ])
    algebra = float(np.abs(cov_a - cov_b).max())

    # Y por muestreo, que es lo que descarta que la coincidencia esté en las
    # fórmulas y no en los mundos.
    rng = np.random.default_rng(SEED + 7)
    n = 2_000_000
    u = rng.standard_normal(n)
    z = rng.standard_normal(n)
    xa = PI * z + DELTA * u + np.sqrt(VEX) * rng.standard_normal(n)
    ya = BETA * xa + GAMMA * u + np.sqrt(VEY) * rng.standard_normal(n)
    u2 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    # Sin recortes a cero: si una varianza saliera negativa esto tiene que
    # reventar aquí y no publicar un mundo que no existe.
    assert vex_b > 0 and vey_b > 0, "el mundo alternativo pide una varianza negativa"
    xb = pi_b * z2 + delta_b * u2 + np.sqrt(vex_b) * rng.standard_normal(n)
    yb = (beta_b * xb + gamma_b * u2 + kappa_b * z2
          + np.sqrt(vey_b) * rng.standard_normal(n))
    sa = np.cov(np.vstack([z, xa, ya]))
    sb = np.cov(np.vstack([z2, xb, yb]))
    sampled = float(np.abs(sa - sb).max())

    sub = np.random.default_rng(SEED + 8).choice(n, 240, replace=False)
    return dict(
        beta_a=BETA, beta_b=beta_b, kappa_b=float(kappa_b), gamma_b=float(gamma_b),
        pi_b=float(pi_b), vex_b=float(vex_b), vey_b=float(vey_b),
        beta_lo=beta_lo, beta_hi=beta_hi, width=beta_hi - beta_lo,
        cov=r(cov_a, 8), algebra_max_diff=algebra, sampled_max_diff=r(sampled),
        iv_a=r(wald(z, xa, ya)), iv_b=r(wald(z2, xb, yb)),
        f_a=r(first_stage_f(z[:N], xa[:N]), 2), f_b=r(first_stage_f(z2[:N], xb[:N]), 2),
        n=n,
        points=dict(za=r(z[sub], 4), xa=r(xa[sub], 4), ya=r(ya[sub], 4),
                    zb=r(z2[sub], 4), xb=r(xb[sub], 4), yb=r(yb[sub], 4)),
    )


def stage_late():
    """Cumplimiento heterogéneo: qué población contesta el cociente.

    Cada unidad tiene un tipo escrito y un efecto propio:

      cumplidor      toma el tratamiento si y solo si le animan
      siempre-toma   lo toma pase lo que pase
      nunca-toma     no lo toma pase lo que pase

    El cociente de Wald **no puede** decir nada de los dos últimos, porque
    animarles no les mueve. Aquí los efectos por tipo son distintos a propósito
    y todos conocidos, así que se puede poner al lado lo que el estimador
    devuelve, el efecto medio de la población entera y el de los cumplidores.
    """
    rng = np.random.default_rng(SEED + 9)
    n = 400_000
    # 55% cumplidores, 20% siempre, 25% nunca. Sin desafiadores, que es el
    # supuesto de monotonía y no una simplificación inocente: se dice.
    tipo = rng.choice([0, 1, 2], size=n, p=[0.55, 0.20, 0.25])
    efecto = np.where(tipo == 0, 1.0, np.where(tipo == 1, 4.0, 2.5))
    z = (rng.random(n) < 0.5).astype(float)
    x = np.where(tipo == 0, z, np.where(tipo == 1, 1.0, 0.0))
    base = 3.0 + rng.standard_normal(n)
    y = base + efecto * x
    est = (y[z == 1].mean() - y[z == 0].mean()) / (x[z == 1].mean() - x[z == 0].mean())
    shares = [float((tipo == t).mean()) for t in range(3)]
    return dict(
        wald=float(est), ate=float(efecto.mean()),
        late=float(efecto[tipo == 0].mean()),
        by_type=[float(efecto[tipo == t].mean()) for t in range(3)],
        shares=shares, names=["complier", "always taker", "never taker"],
        first_stage=float(x[z == 1].mean() - x[z == 0].mean()),
        reduced_form=float(y[z == 1].mean() - y[z == 0].mean()), n=n,
    )


def stage_cloud():
    """La muestra del widget: la página vuelve a estimar sobre ella."""
    rng = np.random.default_rng(SEED + 10)
    z, x, y, u = draw(300, rng)
    return dict(z=r(z, 4), x=r(x, 4), y=r(y, 4),
                ols=r(ols(x, y)), iv=r(wald(z, x, y)), f=r(first_stage_f(z, x), 3))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    head = cached("headline", stage_headline)
    strength = cached("strength", stage_strength)
    seeds = cached("seeds", stage_two_seeds)
    excl = cached("exclusion", stage_exclusion)
    worlds = cached("worlds", stage_two_worlds)
    late = cached("late", stage_late)
    cloud = stage_cloud()

    print(f"  OLS {head['ols_mean']:.4f} (bias {head['ols_bias']:+.4f}, "
          f"closed {head['ols_bias_closed']:+.4f}); IV median {head['iv_median']:.4f}; "
          f"truth {BETA}")
    print(f"  F at pi=0.8: {head['f_mean']:.1f}")
    for row in strength:
        print(f"    pi {row['pi']:.2f}  F {row['f_median']:8.2f}  median bias "
              f"{row['iv_median_bias']:+.4f}  mean {row['iv_mean']:+9.3f}  "
              f"wild {row['wild']:.3f}")
    print(f"  six seeds, means: {[round(s['mean'], 2) for s in seeds]}")
    print(f"  six seeds, medians: {[round(s['median'], 3) for s in seeds]}")
    print(f"  compatible effects: [{worlds['beta_lo']:.4f}, {worlds['beta_hi']:.4f}], "
          f"exhibited {worlds['beta_b']:.4f} with kappa {worlds['kappa_b']:.4f}")
    print(f"  two worlds: algebra {worlds['algebra_max_diff']:.2e}, "
          f"sampled {worlds['sampled_max_diff']:.2e}, "
          f"IV {worlds['iv_a']:.4f} vs {worlds['iv_b']:.4f}")
    print(f"  LATE: wald {late['wald']:.4f}, complier {late['late']:.4f}, "
          f"ATE {late['ate']:.4f}")

    assert abs(head["ols_bias"] - head["ols_bias_closed"]) < 0.02, \
        "el sesgo de OLS no coincide con su forma cerrada"
    assert abs(head["iv_median"] - BETA) < 0.05, "la palanca fuerte no recupera el efecto"
    assert worlds["algebra_max_diff"] < 1e-12, "los dos mundos no comparten la conjunta"
    # El de muestreo es el que caza una parametrización imposible: el álgebra
    # puede cuadrar consigo misma mientras el mundo simulado es otro.
    assert worlds["sampled_max_diff"] < 0.02, \
        "las covarianzas muestrales de los dos mundos no coinciden"
    assert abs(worlds["iv_a"] - worlds["iv_b"]) < 0.02, \
        "el estimador distingue los dos mundos, y toda la sección dice que no puede"
    assert abs(late["wald"] - late["late"]) < 0.02, \
        "el cociente de Wald no está dando el efecto de los cumplidores"
    assert abs(late["ate"] - late["late"]) > 0.5, \
        "sin diferencia entre ATE y LATE la sección no tiene sujeto"
    spread_mean = max(s["mean"] for s in seeds) - min(s["mean"] for s in seeds)
    spread_med = max(s["median"] for s in seeds) - min(s["median"] for s in seeds)
    assert spread_mean > 5 * spread_med, \
        "la media entre semillas no salta más que la mediana, y esa es la sección"

    data = dict(
        meta=dict(seed=SEED, beta=BETA, gamma=GAMMA, delta=DELTA, pi=PI,
                  n=N, reps=REPS, vex=VEX, vey=VEY,
                  model="U,Z ~ N(0,1); X = pi Z + delta U + ex; "
                        "Y = beta X + gamma U + kappa Z + ey"),
        headline={k: r(v) for k, v in head.items()},
        strength=[{k: r(v) for k, v in row.items()} for row in strength],
        seeds=[{k: r(v) for k, v in s.items()} for s in seeds],
        exclusion=[dict(pi=r(row["pi"]),
                        cells=[{k: r(v) for k, v in c.items()} for c in row["cells"]])
                   for row in excl],
        worlds=worlds,
        late={k: (v if isinstance(v, list) and v and isinstance(v[0], str) else r(v))
              for k, v in late.items()},
        cloud=cloud,
    )
    path = OUT / "instruments.json"
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
