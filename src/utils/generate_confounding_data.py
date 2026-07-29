"""Las cifras del artículo de confusores y la paradoja de Simpson.

Este es el primer artículo del atlas cuyo objeto no es una predicción sino una
**intervención**, y eso cambia qué cuenta como verdad de referencia. En todo lo
publicado hasta ahora la verdad era una etiqueta escrita en el dataset; aquí la
verdad es un número que **ninguna columna contiene**: cuánto cambiaría el
resultado si alguien moviera el tratamiento a mano. No hay forma de leerlo de
los datos observados, así que el mundo se escribe en forma cerrada (igual que
la escena del artículo 32 o los esprites del 42) y entonces el efecto causal es
un parámetro, no una estimación.

Con eso, cuatro cosas que se pueden medir en vez de contarse:

  1. **El sesgo del confusor tiene forma cerrada.** Con asignación probit y un
     confusor gaussiano, E[U | X=1] - E[U | X=0] vale 2a/sqrt(1+a^2) * sqrt(2/pi)
     por la razón de Mills, así que el hueco entre la diferencia de medias y el
     efecto verdadero se predice **antes** de mirar ninguna muestra. La página
     compara las dos cosas y publica el error estándar al lado, que es lo único
     que hace interpretable un acuerdo.
  2. **Ajustar por un confusor medido a trozos deja sesgo residual**, y la
     curva del sesgo contra el número de estratos se mide. Lo mismo con un
     confusor medido con ruido: la corrección se queda a medias en proporción
     al ruido. Las dos cosas se citan siempre y casi nunca se dibujan.
  3. **Un colisionador crea asociación donde no la hay.** El mismo motor, sin
     ningún confusor, con el ajuste equivocado: la diferencia de medias es
     insesgada y "controlar por todo lo que hay" la rompe. Es la medida que
     contradice el consejo por defecto de cualquier tutorial de regresión.
  4. **Dos mundos con los mismos datos.** Un confusor y un mediador generan la
     **misma** distribución conjunta (la misma matriz de covarianzas hasta el
     último bit, porque las dos parametrizaciones se resuelven a mano) y piden
     ajustes opuestos. Ahí es donde se ve que el dibujo del grafo no es una
     ilustración del método: es un dato de entrada que los datos no traen.

Y una medida sobre datos reales, porque la paradoja se presenta siempre con el
mismo ejemplo de libro: sobre las 178 botellas de vino que ya publican los
artículos 1, 25, 26, 27 y 33, se **buscan** todas las inversiones de signo que
existen (1.716 triples) y se cuentan, con un test de permutación al lado para
saber cuántas saldrían por azar. La paradoja no es una anécdota de un dataset
famoso: es una frecuencia.

Ejecutar desde cualquier sitio: `python src/utils/generate_confounding_data.py`.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "confounding" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "confounding_cache")

SEED = 19

# El mundo de la sección 1, escrito entero antes de que exista una muestra.
#
#     U ~ N(0, 1)                        el confusor: nadie lo apunta
#     X = 1[ ALPHA*U + e > 0 ],  e ~ N(0, 1)   quién recibe el tratamiento
#     Y = BASE + BETA*X + GAMMA*U + n,   n ~ N(0, NOISE^2)
#
# BETA es el efecto causal. No se estima: se elige. Todo lo que la página mide
# se lee contra él.
BETA = 2.0
GAMMA = 3.0
ALPHA = 1.2
BASE = 10.0
NOISE = 1.0
N_BIG = 4_000_000          # la muestra grande con la que se mide el sesgo
N_SHOW = 420               # la nube que viaja al navegador

# La sección del colisionador usa su propio mundo, sin confusor ninguno:
#     X ~ Bernoulli(1/2)
#     Y = BASE + BETA*X + n
#     C = C_X*X + C_Y*Y + m        un efecto común de los dos
C_X, C_Y, C_NOISE = 1.0, 1.0, 1.0

SPREAD = 0.999  # el tamaño de un tercil no importa aquí; ver bin_edges

TAG = f"b{BETA}g{GAMMA}a{ALPHA}n{N_BIG}s{SEED}"


def cached(name, fn, tag=True):
    """Cada experimento terminado, en disco y fuera del repo.

    Aquí ninguna etapa cuesta minutos, pero la regla del repo es la regla del
    repo y el coste de escribirla es una función de seis líneas. La huella de
    la configuración va pegada al nombre para que cambiar un parámetro invalide
    el caché solo, que es el fallo que costó una tirada entera en el 49.
    """
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
    """Redondeo para el JSON. Lo que viaja redondeado marca el suelo de
    cualquier comparación que lo use, así que se redondea una vez, aquí, y
    todas las referencias se calculan sobre el valor redondeado."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r(x, d) for x in np.asarray(v).tolist()]
    return round(float(v), d)


# --------------------------------------------------------------- el motor
def draw(n, rng, alpha=ALPHA, gamma=GAMMA, beta=BETA):
    """Una muestra del mundo observacional: nadie interviene en nada."""
    u = rng.standard_normal(n)
    e = rng.standard_normal(n)
    x = (alpha * u + e > 0).astype(float)
    y = BASE + beta * x + gamma * u + NOISE * rng.standard_normal(n)
    return u, x, y


def intervene(n, rng, value, alpha=ALPHA, gamma=GAMMA, beta=BETA):
    """El mismo mundo con la flecha hacia X **cortada** y X puesto a mano.

    Esto es literalmente do(X = value): se conserva todo el mecanismo salvo la
    ecuación de X, que se sustituye por una constante. El confusor sigue ahí y
    sigue empujando a Y; lo que desaparece es que empuje también a X.
    """
    u = rng.standard_normal(n)
    y = BASE + beta * value + gamma * u + NOISE * rng.standard_normal(n)
    return y


def mills_gap(alpha=ALPHA):
    """E[U | X=1] - E[U | X=0] en forma cerrada.

    S = alpha*U + e es gaussiana de varianza 1+alpha^2 y Cov(U, S) = alpha, así
    que U condicionado al signo de S es una normal truncada por la mitad:
    E[U | S>0] = Cov(U,S)/sd(S) * phi(0)/(1-Phi(0)) = alpha/sqrt(1+alpha^2) *
    sqrt(2/pi). Por simetría el otro lado es el mismo con signo cambiado, así
    que el hueco es el doble.
    """
    return 2.0 * alpha / np.sqrt(1.0 + alpha ** 2) * np.sqrt(2.0 / np.pi)


def stage_engine():
    """La sección 1: el sesgo medido contra el sesgo predicho."""
    rng = np.random.default_rng(SEED)
    u, x, y = draw(N_BIG, rng)

    y1, y0 = y[x == 1], y[x == 0]
    naive = y1.mean() - y0.mean()
    # El error estándar de ESA diferencia, que es lo que decide si un acuerdo
    # significa algo. Publicarlo al lado no es un adorno: sin él, "coinciden a
    # 0,0008" no se puede leer.
    se_naive = np.sqrt(y1.var(ddof=1) / y1.size + y0.var(ddof=1) / y0.size)

    # La verdad por dos caminos independientes. Uno es el parámetro; el otro es
    # simular el mundo intervenido, que es la definición. Que coincidan no
    # demuestra nada del mundo, demuestra que el simulador hace lo que dice.
    rng_i = np.random.default_rng(SEED + 1)
    ate_sim = intervene(N_BIG, rng_i, 1.0).mean() - intervene(N_BIG, rng_i, 0.0).mean()
    se_sim = np.sqrt(2.0 * (GAMMA ** 2 + NOISE ** 2) / N_BIG)

    gap_closed = mills_gap()
    gap_measured = u[x == 1].mean() - u[x == 0].mean()
    bias_closed = GAMMA * gap_closed
    bias_measured = naive - BETA

    # El ajuste por el confusor, que aquí se puede hacer porque el mundo es de
    # mentira y U está escrito. Regresión lineal con U dentro: el coeficiente
    # de X ES el efecto, porque la ecuación de Y es lineal en U.
    adj = LinearRegression().fit(np.c_[x, u], y).coef_[0]

    return dict(
        naive=naive, se_naive=se_naive, truth=BETA, ate_sim=ate_sim, se_sim=se_sim,
        gap_closed=gap_closed, gap_measured=gap_measured,
        bias_closed=bias_closed, bias_measured=bias_measured, adjusted=adj,
        p_treated=float((x == 1).mean()), n=N_BIG,
    )


def stage_alpha_sweep():
    """El sesgo contra la fuerza con la que el confusor decide el tratamiento.

    La curva cerrada es GAMMA * mills_gap(alpha) y no depende de nada más; la
    medida son muestras finitas. Se dibujan las dos juntas, que es la forma de
    que el lector vea que la primera no es un ajuste de la segunda.
    """
    rows = []
    for i, alpha in enumerate([0.0, 0.2, 0.4, 0.7, 1.0, 1.2, 1.6, 2.0, 3.0, 5.0]):
        rng = np.random.default_rng(SEED + 100 + i)
        n = 400_000
        u, x, y = draw(n, rng, alpha=alpha)
        y1, y0 = y[x == 1], y[x == 0]
        naive = y1.mean() - y0.mean()
        se = np.sqrt(y1.var(ddof=1) / y1.size + y0.var(ddof=1) / y0.size)
        rows.append(dict(alpha=alpha, naive=naive, se=se,
                         closed=BETA + GAMMA * mills_gap(alpha),
                         bias_closed=GAMMA * mills_gap(alpha)))
    return rows


def stage_strata():
    """Ajustar por un confusor **a trozos**: cuánto sesgo sobrevive.

    Estratificar por U en k grupos y promediar la diferencia dentro de cada
    grupo es el ajuste que se hace en la práctica cuando el confusor es
    continuo. Dentro de un estrato U todavía varía, así que dentro de un
    estrato sigue habiendo confusión: el sesgo no desaparece, se encoge, y la
    forma en que se encoge es lo que esta tabla mide.
    """
    rng = np.random.default_rng(SEED + 2)
    n = 2_000_000
    u, x, y = draw(n, rng)
    rows = []
    for k in [1, 2, 3, 5, 10, 20, 50, 100]:
        # Cortes por cuantiles de U, que es lo que hace cualquiera que
        # estratifique: mismos tamaños de estrato, no mismos anchos.
        edges = np.quantile(u, np.linspace(0, 1, k + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        idx = np.searchsorted(edges, u, side="right") - 1
        idx = np.clip(idx, 0, k - 1)
        num, den = 0.0, 0
        usable = 0
        for s in range(k):
            m = idx == s
            a, b = y[m & (x == 1)], y[m & (x == 0)]
            if a.size < 2 or b.size < 2:
                continue
            num += m.sum() * (a.mean() - b.mean())
            den += m.sum()
            usable += 1
        est = num / den
        rows.append(dict(k=k, estimate=est, bias=est - BETA, strata_used=usable))
    return rows


def stage_proxy():
    """Ajustar por un confusor medido con ruido.

    Lo que se apunta no es U sino W = U + ruido, y se ajusta por W. El sesgo
    que sobrevive es una función del ruido, y en el caso lineal gaussiano tiene
    forma cerrada: la regresión de Y sobre (X, W) recupera solo la parte de U
    que W explica, es decir 1/(1+s^2) de ella, así que sobra GAMMA * s^2/(1+s^2)
    veces el hueco de Mills condicionado. Aquí se mide y se pone al lado del
    caso perfecto y del caso de no ajustar nada, que son los dos extremos.
    """
    rng = np.random.default_rng(SEED + 3)
    n = 1_000_000
    u, x, y = draw(n, rng)
    rows = []
    for s in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        w = u + s * rng.standard_normal(n)
        est = LinearRegression().fit(np.c_[x, w], y).coef_[0]
        rows.append(dict(noise=s, estimate=est, bias=est - BETA,
                         reliability=1.0 / (1.0 + s ** 2)))
    return rows


def stage_collider():
    """El mundo sin confusor: ajustar es lo que rompe la respuesta.

    X se echa a suertes, así que la diferencia de medias es insesgada por
    construcción y no hay nada que arreglar. C es un efecto común de X y de Y,
    y meterlo en la regresión abre un camino entre los dos que no existía.
    """
    rng = np.random.default_rng(SEED + 4)
    n = 2_000_000
    x = (rng.random(n) < 0.5).astype(float)
    y = BASE + BETA * x + NOISE * rng.standard_normal(n)
    c = C_X * x + C_Y * y + C_NOISE * rng.standard_normal(n)

    y1, y0 = y[x == 1], y[x == 0]
    naive = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / y1.size + y0.var(ddof=1) / y0.size)
    with_c = LinearRegression().fit(np.c_[x, c], y).coef_[0]

    # Y el mismo experimento estratificando por C en terciles, que es la forma
    # no paramétrica del mismo error y descarta que lo anterior sea un artefacto
    # de la forma lineal.
    edges = np.quantile(c, [1 / 3, 2 / 3])
    idx = np.searchsorted(edges, c)
    strata = []
    for s in range(3):
        m = idx == s
        a, b = y[m & (x == 1)], y[m & (x == 0)]
        strata.append(dict(stratum=s, n=int(m.sum()), estimate=float(a.mean() - b.mean())))
    pooled = float(sum(s["n"] * s["estimate"] for s in strata) / sum(s["n"] for s in strata))

    # La nube que viaja, adelgazada: 300 puntos bastan para ver las dos rectas.
    sub = np.random.default_rng(SEED + 5).choice(n, 300, replace=False)
    return dict(naive=naive, se=se, truth=BETA, with_collider=with_c,
                strata=strata, stratified=pooled,
                points=dict(x=r(x[sub], 4), y=r(y[sub], 4), c=r(c[sub], 4),
                            edges=r(edges, 4)))


def stage_two_worlds():
    """Dos grafos que producen la MISMA distribución conjunta.

    Mundo A (confusor):   Z -> X,  Z -> Y,  X -> Y   con efecto b_a
    Mundo B (mediador):   X -> Z,  Z -> Y,  X -> Y   con efecto b_b

    Los dos son gaussianos, así que la conjunta es la matriz de covarianzas y
    nada más. Se elige el mundo A y se resuelven a mano los parámetros del
    mundo B que reproducen esa matriz **exactamente**, en vez de buscarlos por
    ajuste: así el acuerdo es álgebra y no un residuo pequeño.

    Con la conjunta idéntica, las dos preguntas tienen respuestas distintas:
    en A el efecto de X sobre Y es b_a y hay que ajustar por Z; en B es
    b_b + c_b*d_b (el camino directo más el que pasa por Z) y ajustar por Z
    **tapa** parte del efecto. La regresión con Z dentro devuelve el mismo
    número en los dos casos, y ese número es la respuesta correcta solo en uno.
    """
    # Mundo A: Z ~ N(0,1); X = az*Z + ex; Y = ba*X + gz*Z + ey
    az, ba, gz = 0.9, 2.0, 1.5
    vx_e, vy_e = 1.0, 1.0
    # Covarianzas de A, en forma cerrada
    vz = 1.0
    vx = az ** 2 * vz + vx_e
    czx = az * vz
    czy = ba * czx + gz * vz
    vy = ba ** 2 * vx + gz ** 2 * vz + 2 * ba * gz * czx + vy_e
    cxy = ba * vx + gz * czx
    cov_a = np.array([[vz, czx, czy], [czx, vx, cxy], [czy, cxy, vy]])

    # Mundo B: X ~ N(0, vx); Z = cb*X + ez; Y = bb*X + db*Z + ey2
    # Se imponen las tres covarianzas de A y se despeja.
    #   Cov(X,Z) = cb*vx            -> cb
    #   Var(Z)   = cb^2*vx + vez    -> vez
    #   Cov(X,Y) = bb*vx + db*cb*vx
    #   Cov(Z,Y) = bb*cb*vx + db*Var(Z)
    cb = czx / vx
    vez = vz - cb ** 2 * vx
    m = np.array([[vx, cb * vx], [cb * vx, vz]])
    bb, db = np.linalg.solve(m, np.array([cxy, czy]))
    vey = vy - (bb ** 2 * vx + db ** 2 * vz + 2 * bb * db * cb * vx)
    cov_b = np.array([
        [vz, cb * vx, bb * cb * vx + db * vz],
        [cb * vx, vx, bb * vx + db * cb * vx],
        [bb * cb * vx + db * vz, bb * vx + db * cb * vx,
         bb ** 2 * vx + db ** 2 * vz + 2 * bb * db * cb * vx + vey],
    ])
    same = float(np.abs(cov_a - cov_b).max())

    # Y una comprobación por muestreo, porque una identidad algebraica
    # comprobada sobre las fórmulas que la derivan es circular: se simulan los
    # dos mundos y se comparan las covarianzas muestrales.
    rng = np.random.default_rng(SEED + 6)
    n = 1_000_000
    z = rng.standard_normal(n)
    xa = az * z + np.sqrt(vx_e) * rng.standard_normal(n)
    ya = ba * xa + gz * z + np.sqrt(vy_e) * rng.standard_normal(n)
    xb = np.sqrt(vx) * rng.standard_normal(n)
    zb = cb * xb + np.sqrt(max(vez, 0.0)) * rng.standard_normal(n)
    yb = bb * xb + db * zb + np.sqrt(max(vey, 0.0)) * rng.standard_normal(n)
    sa = np.cov(np.vstack([z, xa, ya]))
    sb = np.cov(np.vstack([zb, xb, yb]))
    sampled = float(np.abs(sa - sb).max())

    # Las tres cifras que decide el grafo y no los datos.
    adj_a = LinearRegression().fit(np.c_[xa, z], ya).coef_[0]
    adj_b = LinearRegression().fit(np.c_[xb, zb], yb).coef_[0]
    raw_a = LinearRegression().fit(xa.reshape(-1, 1), ya).coef_[0]
    raw_b = LinearRegression().fit(xb.reshape(-1, 1), yb).coef_[0]

    sub = np.random.default_rng(SEED + 7).choice(n, 260, replace=False)
    return dict(
        world_a=dict(az=az, beta=ba, gz=gz, truth=ba, adjusted=adj_a, unadjusted=raw_a),
        world_b=dict(cb=float(cb), beta=float(bb), db=float(db),
                     truth=float(bb + db * cb), adjusted=adj_b, unadjusted=raw_b),
        cov=r(cov_a, 8), cov_max_diff=same, cov_sampled_diff=sampled, n=n,
        points=dict(za=r(z[sub], 4), xa=r(xa[sub], 4), ya=r(ya[sub], 4),
                    zb=r(zb[sub], 4), xb=r(xb[sub], 4), yb=r(yb[sub], 4)),
    )


# ----------------------------------------------------------- datos reales
def slope(x, y):
    """Pendiente de mínimos cuadrados de y sobre x, sin librería.

    Es una división de dos sumas y la escribe también el navegador; tenerla a
    mano evita que el guardia compare dos implementaciones distintas de lo
    mismo.
    """
    mx, my = x.mean(), y.mean()
    den = ((x - mx) ** 2).sum()
    return float(((x - mx) * (y - my)).sum() / den) if den > 0 else 0.0


def reversals(X, groups, n_groups, min_n=12):
    """Todos los triples cuya pendiente se da la vuelta al mirar por grupos.

    Un triple es (i, j): la pendiente de la columna j sobre la columna i. La
    inversión estricta pide que **todos** los grupos con datos suficientes
    tengan la pendiente del signo contrario al del conjunto entero, que es la
    versión de la paradoja que no se puede explicar con un grupo raro.
    """
    p = X.shape[1]
    found = []
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            s_all = slope(X[:, i], X[:, j])
            if abs(s_all) < 1e-12:
                continue
            within = []
            ok = True
            for g in range(n_groups):
                m = groups == g
                if m.sum() < min_n:
                    ok = False
                    break
                within.append(slope(X[m, i], X[m, j]))
            if not ok:
                continue
            if all(np.sign(w) == -np.sign(s_all) for w in within):
                found.append(dict(i=i, j=j, pooled=s_all, within=within,
                                  strength=min(abs(w) for w in within) * abs(s_all)))
    return found


def stage_wine():
    """La paradoja contada como frecuencia, sobre botellas de verdad.

    Las 178 botellas son las de los artículos 1, 25, 26, 27 y 33: mismas trece
    columnas y mismos tres cultivares. El cultivar es un confusor real y
    conocido (decide a la vez casi todas las químicas), así que no hay que
    fabricar nada para tener un estratificador honesto.

    Y el control que casi nadie pone: barajar el cultivar y volver a contar. Si
    salieran tantas inversiones con grupos al azar, el recuento no diría nada
    sobre el cultivar.
    """
    wine = load_wine()
    X, g = wine.data, wine.target
    # Las columnas se estandarizan para que las pendientes sean comparables
    # entre pares; el signo, que es lo único que la búsqueda mira, no cambia.
    Xs = (X - X.mean(0)) / X.std(0)

    found = reversals(Xs, g, 3)
    total = X.shape[1] * (X.shape[1] - 1)

    rng = np.random.default_rng(SEED + 8)
    null = []
    for _ in range(200):
        null.append(len(reversals(Xs, rng.permutation(g), 3)))
    null = np.array(null, dtype=float)

    # También con un tercer químico binado en terciles como estratificador, que
    # es el caso en el que nadie sabe si el grupo es un confusor o no.
    third = 0
    third_total = 0
    for k in range(X.shape[1]):
        edges = np.quantile(Xs[:, k], [1 / 3, 2 / 3])
        gk = np.searchsorted(edges, Xs[:, k])
        f = [t for t in reversals(np.delete(Xs, k, axis=1), gk, 3)]
        third += len(f)
        third_total += (X.shape[1] - 1) * (X.shape[1] - 2)

    best = max(found, key=lambda d: d["strength"]) if found else None
    detail = None
    if best:
        i, j = best["i"], best["j"]
        # Las pendientes que la página publica se miden sobre las coordenadas
        # que la página va a tener, no sobre las de dentro de este proceso. Con
        # cuatro decimales la diferencia es pequeña, pero es la diferencia
        # entre un guardia que puede exigir 1e-6 y uno que tiene que perdonar
        # 1e-3 sin saber por qué.
        xr = np.round(Xs[:, i], 4)
        yr = np.round(Xs[:, j], 4)
        detail = dict(
            i=i, j=j, xname=wine.feature_names[i], yname=wine.feature_names[j],
            pooled=slope(xr, yr),
            within=r([slope(xr[g == k], yr[g == k]) for k in range(3)], 6),
            x=r(xr, 4), y=r(yr, 4), g=[int(v) for v in g],
        )
    return dict(
        n=int(X.shape[0]), p=int(X.shape[1]), groups=3,
        pairs=total, reversals=len(found),
        rate=len(found) / total,
        null_mean=float(null.mean()), null_max=float(null.max()),
        null_zero=float((null == 0).mean()),
        third_pairs=third_total, third_reversals=third,
        best=detail,
        names=list(wine.feature_names),
        class_names=[str(c) for c in wine.target_names],
        digest_mean=r(float(X.mean()), 8), digest_sum=r(float(X.sum()), 6),
    )


def stage_cloud():
    """La nube del primer widget, con el mundo cerrado detrás.

    Viaja sin redondear a menos de cuatro decimales porque la página vuelve a
    ajustar sobre ella y el guardia compara los dos ajustes: lo que se compara
    es la aritmética, no los datos.
    """
    rng = np.random.default_rng(SEED + 9)
    u, x, y = draw(N_SHOW, rng)
    edges = np.quantile(u, [1 / 3, 2 / 3])
    return dict(u=r(u, 4), x=[int(v) for v in x], y=r(y, 4), edges=r(edges, 4),
                naive=r(float(y[x == 1].mean() - y[x == 0].mean()), 6),
                adjusted=r(float(LinearRegression().fit(np.c_[x, u], y).coef_[0]), 6))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    engine = cached("engine", stage_engine)
    print(f"  naive {engine['naive']:.4f} (se {engine['se_naive']:.4f}), "
          f"truth {engine['truth']}, adjusted {engine['adjusted']:.4f}")
    print(f"  bias closed {engine['bias_closed']:.4f} measured {engine['bias_measured']:.4f}")

    sweep = cached("alpha", stage_alpha_sweep)
    strata = cached("strata", stage_strata)
    proxy = cached("proxy", stage_proxy)
    collider = cached("collider", stage_collider)
    worlds = cached("worlds", stage_two_worlds)
    wine = cached("wine", stage_wine, tag=False)
    cloud = stage_cloud()

    print(f"  strata: k=1 bias {strata[0]['bias']:.4f}, "
          f"k={strata[-1]['k']} bias {strata[-1]['bias']:.4f}")
    print(f"  collider: naive {collider['naive']:.4f}, with C {collider['with_collider']:.4f}")
    print(f"  two worlds: max cov diff {worlds['cov_max_diff']:.2e}, "
          f"truths {worlds['world_a']['truth']:.4f} vs {worlds['world_b']['truth']:.4f}")
    print(f"  wine: {wine['reversals']} reversals of {wine['pairs']} pairs, "
          f"null mean {wine['null_mean']:.2f}")

    # Los asertos que hacen imposible publicar una versión rota de esto. No son
    # decoración: cada uno corresponde a una afirmación de la página.
    assert abs(engine["bias_measured"] - engine["bias_closed"]) < 4 * engine["se_naive"], \
        "el sesgo medido y el cerrado no coinciden dentro de cuatro errores estándar"
    assert abs(engine["adjusted"] - BETA) < 0.01, "ajustar por U no devuelve el efecto"
    assert abs(collider["naive"] - BETA) < 0.01, "sin confusor la ingenua tiene que acertar"
    assert abs(collider["with_collider"] - BETA) > 10 * collider["se"], \
        "el colisionador no está creando sesgo, así que la sección no tiene sujeto"
    assert worlds["cov_max_diff"] < 1e-12, "los dos mundos no comparten la conjunta"
    assert abs(strata[0]["bias"] - engine["bias_closed"]) < 0.02, \
        "un solo estrato tiene que ser exactamente no ajustar"

    data = dict(
        meta=dict(seed=SEED, beta=BETA, gamma=GAMMA, alpha=ALPHA, base=BASE,
                  noise=NOISE, n_big=N_BIG,
                  model="U ~ N(0,1); X = 1[alpha U + e > 0]; Y = base + beta X + gamma U + n",
                  collider="X ~ Bern(1/2); Y = base + beta X + n; C = cx X + cy Y + m",
                  cx=C_X, cy=C_Y),
        engine={k: r(v) for k, v in engine.items()},
        alpha_sweep=[{k: r(v) for k, v in row.items()} for row in sweep],
        strata=[{k: r(v) for k, v in row.items()} for row in strata],
        proxy=[{k: r(v) for k, v in row.items()} for row in proxy],
        collider=dict(
            naive=r(collider["naive"]), se=r(collider["se"]), truth=BETA,
            with_collider=r(collider["with_collider"]),
            stratified=r(collider["stratified"]),
            strata=[{k: r(v) for k, v in s.items()} for s in collider["strata"]],
            points=collider["points"],
        ),
        worlds=dict(
            world_a={k: r(v) for k, v in worlds["world_a"].items()},
            world_b={k: r(v) for k, v in worlds["world_b"].items()},
            cov=worlds["cov"], cov_max_diff=worlds["cov_max_diff"],
            cov_sampled_diff=r(worlds["cov_sampled_diff"]), n=worlds["n"],
            points=worlds["points"],
        ),
        wine=wine,
        cloud=cloud,
    )
    path = OUT / "confounding.json"
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
