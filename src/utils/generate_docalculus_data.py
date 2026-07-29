"""Las cifras del artículo de do-calculus y bosques causales.

Los tres artículos anteriores de la rama van pasando el problema de un sitio a
otro: ajusta por el confusor, resume el ajuste en un score, busca una palanca
cuando no hay confusor medido. Los tres deciden **a mano** por qué ajustar, y
esa decisión es la que este artículo mecaniza y luego mide.

La primera mitad no estima nada. Todas las variables son binarias y el grafo es
pequeño, así que la distribución intervenida se **calcula**: se tacha del
producto el factor de la variable intervenida y se suma sobre las demás. Con la
verdad en aritmética exacta se pueden hacer dos cosas que casi nunca se ven
juntas:

  1. **Enumerar todos los conjuntos de ajuste posibles** (los 2^k subconjuntos
     de las variables observadas) y comprobar uno a uno cuáles devuelven la
     respuesta exacta. El criterio de la puerta trasera, implementado como un
     algoritmo de separación en el grafo, tiene que predecir exactamente esa
     lista. Es la misma forma de argumentar que el retículo enumerado del 35:
     no se cita el criterio, se comprueba contra la aritmética.
  2. **Contar cuántas veces "ajusta por todo lo que tengas" da la respuesta
     equivocada**, sobre miles de grafos al azar. Es el consejo por defecto de
     cualquier tutorial de regresión y aquí es una frecuencia medida.

Y la puerta delantera, que es el resultado que a la gente le parece un truco:
un mundo donde el confusor no está medido, ningún conjunto de ajuste sirve, y
sin embargo el efecto se recupera **exacto** pasando por el mediador.

La segunda mitad estima, porque la pregunta cambia: no cuánto sube el resultado
en promedio, sino para quién sube. tau(V) está escrito, así que los cuatro
métodos (S, T, R y un bosque causal con estimación honesta) se puntúan contra
la verdad punto a punto en vez de compararse entre ellos.

Ejecutar desde cualquier sitio: `python src/utils/generate_docalculus_data.py`.
"""
import json
import os
import pickle
from itertools import combinations, product
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "do-calculus" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "docalculus_cache")

SEED = 19
N_RANDOM_DAGS = 3000
N_HET = 4000
HET_REPS = 8

TAG = f"d{N_RANDOM_DAGS}h{N_HET}r{HET_REPS}s{SEED}"


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


# =====================================================================
#  Primera mitad: un modelo pequeño, resuelto en aritmética exacta
# =====================================================================
class DAG:
    """Un modelo causal con todas las variables binarias.

    `parents[i]` es la lista de padres del nodo i y `cpt[i]` una tabla con
    2**len(parents) entradas: la probabilidad de que el nodo valga 1 para cada
    combinación de sus padres. Nada de esto se muestrea: con k nodos binarios
    hay 2^k mundos posibles y se suman todos.
    """

    def __init__(self, parents, cpt, names=None):
        self.parents = parents
        self.cpt = cpt
        self.k = len(parents)
        self.names = names or [f"V{i}" for i in range(self.k)]

    def worlds(self):
        return list(product([0, 1], repeat=self.k))

    def p_row(self, w, do=None):
        """Probabilidad de un mundo, con los factores intervenidos tachados.

        `do` es un diccionario nodo -> valor. La factorización truncada es
        literalmente esto: el mismo producto sin los factores de las variables
        que alguien ha fijado a mano. Si el mundo no es compatible con la
        intervención, vale cero.
        """
        do = do or {}
        p = 1.0
        for i in range(self.k):
            if i in do:
                if w[i] != do[i]:
                    return 0.0
                continue        # el factor se tacha: eso ES el do
            idx = 0
            for pa in self.parents[i]:
                idx = idx * 2 + w[pa]
            pi = self.cpt[i][idx]
            p *= pi if w[i] == 1 else 1 - pi
        return p

    def joint(self, do=None):
        return np.array([self.p_row(w, do) for w in self.worlds()])

    def p_do(self, x_node, x_val, y_node):
        """P(Y=1 | do(X=x)), exacta."""
        ws = self.worlds()
        j = self.joint({x_node: x_val})
        tot = j.sum()
        num = sum(p for w, p in zip(ws, j) if w[y_node] == 1)
        return num / tot if tot > 0 else np.nan

    def ate(self, x_node, y_node):
        return self.p_do(x_node, 1, y_node) - self.p_do(x_node, 0, y_node)

    def adjust(self, x_node, y_node, S):
        """La fórmula de ajuste sobre el conjunto S, en aritmética exacta.

        sum_s P(s) [ P(y|x=1,s) - P(y|x=0,s) ]. Se calcula de la conjunta
        observacional, sin muestrear, así que lo que salga distinto de la
        verdad no es error de estimación: es el estimando equivocado.
        """
        ws = self.worlds()
        j = self.joint()
        S = list(S)
        acc = 0.0
        for svals in product([0, 1], repeat=len(S)):
            mask = [all(w[s] == v for s, v in zip(S, svals)) for w in ws]
            ps = sum(p for p, m in zip(j, mask) if m)
            if ps <= 1e-15:
                continue
            out = []
            for xv in (0, 1):
                num = sum(p for w, p, m in zip(ws, j, mask)
                          if m and w[x_node] == xv and w[y_node] == 1)
                den = sum(p for w, p, m in zip(ws, j, mask) if m and w[x_node] == xv)
                if den <= 1e-15:
                    return np.nan          # sin positividad no hay fórmula
                out.append(num / den)
            acc += ps * (out[1] - out[0])
        return acc

    # ---- el criterio gráfico, implementado como algoritmo y no como frase
    def children(self, i):
        return [j for j in range(self.k) if i in self.parents[j]]

    def descendants(self, i):
        seen, stack = set(), [i]
        while stack:
            v = stack.pop()
            for c in self.children(v):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    def d_separated(self, a, b, Z):
        """Bayes-ball: ¿queda algún camino abierto entre a y b dado Z?

        Se recorre el grafo con el sentido de llegada en el estado, que es lo
        que distingue una cadena de un colisionador. Un colisionador abre
        cuando él o alguno de sus descendientes está condicionado, y esa es la
        única regla que hace falta escribir aparte.
        """
        Z = set(Z)
        cond_desc = set()
        for z in Z:
            cond_desc |= {z} | self.descendants(z)
        visited = set()
        stack = [(a, "up")]      # "up": venimos de un hijo; "down": de un padre
        while stack:
            node, direction = stack.pop()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))
            if node == b:
                return False
            if direction == "up":
                if node not in Z:
                    for p in self.parents[node]:
                        stack.append((p, "up"))
                    for c in self.children(node):
                        stack.append((c, "down"))
            else:
                if node not in Z:
                    for c in self.children(node):
                        stack.append((c, "down"))
                if node in cond_desc:
                    for p in self.parents[node]:
                        stack.append((p, "up"))
        return True

    def backdoor(self, x, y, Z):
        """El criterio, en sus dos mitades.

        (1) ningún elemento de Z es descendiente de X, y (2) Z bloquea todos
        los caminos que salen de X hacia atrás. La segunda se comprueba sobre
        el grafo con las flechas que SALEN de X borradas, que es la forma
        estándar de decir "solo los caminos por la puerta de atrás".
        """
        desc = self.descendants(x)
        if any(z in desc or z == x for z in Z):
            return False
        cut = DAG([[p for p in pa if not (i != x and p == x)] if True else pa
                   for i, pa in enumerate(self.parents)], self.cpt)
        cut.parents = [list(pa) for pa in self.parents]
        for i in range(self.k):
            if x in cut.parents[i]:
                cut.parents[i] = [p for p in cut.parents[i] if p != x]
        return cut.d_separated(x, y, Z)


def pick_demo_dag():
    """El grafo de ejemplo se coloca por búsqueda, no a ojo.

    La estructura está elegida por lo que tiene que enseñar; las tablas de
    probabilidad no, y con tablas al azar el primer intento salió con un efecto
    verdadero de 0,000351, o sea una demostración cuyo sujeto no se ve. Se
    barren semillas hasta encontrar unas tablas donde las cuatro cosas que la
    sección afirma sean visibles a la vez: efecto de tamaño decente, ajustar
    por todo mal, no ajustar por nada mal, y más de un conjunto correcto. Es la
    misma regla que colocó el punto que estrecha el margen del artículo 8.
    """
    for seed in range(5000):
        rng = np.random.default_rng(seed)
        g = demo_dag(rng)
        x, y = 2, 4
        truth = g.ate(x, y)
        if abs(truth) < 0.15:
            continue
        others = [i for i in range(g.k) if i not in (x, y)]
        full = g.adjust(x, y, others)
        empty = g.adjust(x, y, [])
        if not np.isfinite(full) or abs(full - truth) < 0.05:
            continue
        if abs(empty - truth) < 0.05:
            continue
        valid = [S for size in range(len(others) + 1)
                 for S in combinations(others, size) if g.backdoor(x, y, S)]
        if len(valid) < 2:
            continue
        if max(abs(g.adjust(x, y, S) - truth) for S in valid) > 1e-12:
            continue
        return g, seed
    raise RuntimeError("no hay tablas que enseñen las cuatro cosas a la vez")


def demo_dag(rng):
    """El modelo de la primera sección: seis variables binarias.

        A -> X,  A -> Y        confusor clásico
        B -> A,  B -> M        un abuelo por un lado
        X -> M -> Y            un mediador
        X -> Y                 el camino directo
        Y -> D                 un descendiente del resultado

    Está elegido para que las cuatro trampas quepan en un dibujo: ajustar por
    M tapa parte del efecto, ajustar por D abre lo que no había, A hace falta y
    B no, y aun así hay más de un conjunto correcto.
    """
    names = ["B", "A", "X", "M", "Y", "D"]
    parents = [[], [0], [1], [2, 0], [2, 3, 1], [4]]
    cpt = []
    for i, pa in enumerate(parents):
        n = 2 ** len(pa)
        # Tablas al azar pero lejos de 0 y de 1, para que la positividad se
        # cumpla con holgura y ninguna fórmula divida por casi nada.
        cpt.append(rng.uniform(0.18, 0.82, size=n))
    return DAG(parents, cpt, names)


def stage_exact():
    """Enumerar todos los conjuntos de ajuste y ver cuáles aciertan."""
    g, dag_seed = pick_demo_dag()
    x, y = 2, 4
    truth = g.ate(x, y)
    others = [i for i in range(g.k) if i not in (x, y)]
    rows = []
    for size in range(len(others) + 1):
        for S in combinations(others, size):
            est = g.adjust(x, y, S)
            ok = g.backdoor(x, y, S)
            rows.append(dict(S=list(S), names=[g.names[i] for i in S],
                             estimate=float(est),
                             error=float(abs(est - truth)) if np.isfinite(est) else None,
                             backdoor=bool(ok)))
    valid = [row for row in rows if row["backdoor"]]
    worst_valid = max((row["error"] for row in valid), default=0.0)
    # Los que NO cumplen el criterio y aun así aciertan: existen, y decirlo es
    # más honesto que fingir que el criterio es necesario además de suficiente.
    lucky = [row for row in rows if not row["backdoor"] and row["error"] is not None
             and row["error"] < 1e-12]
    full = [row for row in rows if len(row["S"]) == len(others)][0]
    empty = [row for row in rows if not row["S"]][0]
    return dict(
        truth=float(truth), rows=rows, n_sets=len(rows), n_valid=len(valid),
        worst_valid_error=float(worst_valid), n_lucky=len(lucky),
        full=full, empty=empty,
        names=g.names, parents=[list(p) for p in g.parents], dag_seed=dag_seed,
        cpt=[list(map(float, c)) for c in g.cpt], x=x, y=y,
        p_do1=float(g.p_do(x, 1, y)), p_do0=float(g.p_do(x, 0, y)),
        p_obs1=float(sum(p for w, p in zip(g.worlds(), g.joint())
                         if w[x] == 1 and w[y] == 1)
                     / sum(p for w, p in zip(g.worlds(), g.joint()) if w[x] == 1)),
        p_obs0=float(sum(p for w, p in zip(g.worlds(), g.joint())
                         if w[x] == 0 and w[y] == 1)
                     / sum(p for w, p in zip(g.worlds(), g.joint()) if w[x] == 0)),
    )


def random_dag(rng, k=6, p_edge=0.45):
    parents = [[] for _ in range(k)]
    for j in range(k):
        for i in range(j):
            if rng.random() < p_edge:
                parents[j].append(i)
    cpt = [rng.uniform(0.15, 0.85, size=2 ** len(pa)) for pa in parents]
    return DAG(parents, cpt)


def stage_random_dags():
    """Cuántas veces "ajusta por todo" contesta mal, contado.

    Se sortean grafos, se elige X e Y con al menos un camino dirigido de uno a
    otro, y se compara la fórmula de ajuste con TODAS las demás variables
    observadas contra la verdad calculada. Se cuentan tres cosas: cuántas veces
    el conjunto completo es válido, cuántas veces existe algún conjunto válido
    (que es otra pregunta), y cuánto se equivoca cuando se equivoca.
    """
    rng = np.random.default_rng(SEED + 1)
    n_ok, n_any, errors, n_used = 0, 0, [], 0
    empty_ok = 0
    while n_used < N_RANDOM_DAGS:
        g = random_dag(rng)
        pairs = [(a, b) for a in range(g.k) for b in range(g.k)
                 if a != b and b in g.descendants(a)]
        if not pairs:
            continue
        x, y = pairs[rng.integers(len(pairs))]
        others = [i for i in range(g.k) if i not in (x, y)]
        truth = g.ate(x, y)
        if not np.isfinite(truth):
            continue
        n_used += 1
        full = g.adjust(x, y, others)
        if np.isfinite(full) and abs(full - truth) < 1e-10:
            n_ok += 1
        elif np.isfinite(full):
            errors.append(abs(full - truth))
        if g.backdoor(x, y, []):
            empty_ok += 1
        if any(g.backdoor(x, y, S) for size in range(len(others) + 1)
               for S in combinations(others, size)):
            n_any += 1
    errors = np.array(errors) if errors else np.array([0.0])
    return dict(n=n_used, full_valid=n_ok, full_rate=n_ok / n_used,
                any_valid=n_any, any_rate=n_any / n_used,
                empty_valid=empty_ok, empty_rate=empty_ok / n_used,
                err_median=float(np.median(errors)), err_max=float(errors.max()),
                err_over_05=float((errors > 0.05).mean()))


def stage_frontdoor():
    """El confusor no está medido y aun así el efecto sale exacto.

        U -> X,  U -> Y,  X -> M -> Y

    U no se observa. Ningún conjunto de las variables observadas cumple la
    puerta trasera (se comprueba enumerando los cuatro). La fórmula de la
    puerta delantera pasa por M en dos pasos y devuelve la verdad, que aquí se
    calcula tachando el factor de X en el modelo COMPLETO, U incluido.
    """
    rng = np.random.default_rng(SEED + 2)
    names = ["U", "X", "M", "Y"]
    parents = [[], [0], [1], [2, 0]]
    cpt = [rng.uniform(0.2, 0.8, size=2 ** len(pa)) for pa in parents]
    g = DAG(parents, cpt, names)
    u, x, m, y = 0, 1, 2, 3
    truth = g.ate(x, y)

    ws, j = g.worlds(), g.joint()

    def p(cond):
        return sum(pp for w, pp in zip(ws, j) if cond(w))

    def pc(a, b):
        den = p(b)
        return p(lambda w: a(w) and b(w)) / den if den > 1e-15 else np.nan

    # P(m|x) y P(y|x',m), las dos piezas observables.
    front = 0.0
    for mv in (0, 1):
        pm1 = pc(lambda w: w[m] == mv, lambda w: w[x] == 1)
        pm0 = pc(lambda w: w[m] == mv, lambda w: w[x] == 0)
        inner = 0.0
        for xv in (0, 1):
            px = p(lambda w, xv=xv: w[x] == xv)
            pyxm = pc(lambda w, xv=xv, mv=mv: w[y] == 1,
                      lambda w, xv=xv, mv=mv: w[x] == xv and w[m] == mv)
            inner += pyxm * px
        front += (pm1 - pm0) * inner

    obs = [i for i in (m,)]     # lo observable aparte de X e Y
    back = []
    for size in range(len(obs) + 1):
        for S in combinations(obs, size):
            back.append(dict(S=[names[i] for i in S], estimate=float(g.adjust(x, y, S)),
                             backdoor=bool(g.backdoor(x, y, S))))
    naive = pc(lambda w: w[y] == 1, lambda w: w[x] == 1) - \
        pc(lambda w: w[y] == 1, lambda w: w[x] == 0)
    return dict(truth=float(truth), frontdoor=float(front),
                error=float(abs(front - truth)), naive=float(naive),
                backdoor_sets=back, names=names,
                parents=[list(pa) for pa in parents],
                cpt=[list(map(float, c)) for c in cpt])


def stage_rules():
    """Las tres reglas, cada una con un caso donde vale y otro donde no.

    Una regla del do-calculus es una igualdad entre dos distribuciones
    intervenidas que se sostiene cuando una separación se cumple en un grafo
    recortado. Aquí las dos partes se calculan por separado (las dos por
    factorización truncada) y se resta. Lo que la tabla enseña es que la
    igualdad no es aproximada: cuando la condición se cumple, el residuo es del
    orden del épsilon de la máquina, y cuando no, es un número que se ve.
    """
    rng = np.random.default_rng(SEED + 3)
    out = []

    # Regla 2 (la que todo el mundo usa sin nombrarla): si no hay puerta
    # trasera entre X e Y, intervenir en X es lo mismo que condicionar.
    #   caso bueno: X sin padres (aleatorizado)
    #   caso malo:  X con un confusor
    good = DAG([[], [0], [0, 1]], [rng.uniform(.2, .8, 1), rng.uniform(.2, .8, 2),
                                   rng.uniform(.2, .8, 4)], ["Z", "X", "Y"])
    bad = DAG([[], [0], [0, 1]], [rng.uniform(.2, .8, 1), rng.uniform(.2, .8, 2),
                                  rng.uniform(.2, .8, 4)], ["U", "X", "Y"])
    for tag, g, cond in (("randomised", good, True), ("confounded", bad, False)):
        ws, j = g.worlds(), g.joint()
        obs1 = (sum(p for w, p in zip(ws, j) if w[1] == 1 and w[2] == 1)
                / sum(p for w, p in zip(ws, j) if w[1] == 1))
        obs0 = (sum(p for w, p in zip(ws, j) if w[1] == 0 and w[2] == 1)
                / sum(p for w, p in zip(ws, j) if w[1] == 0))
        do = g.ate(1, 2)
        # En "randomised" el nodo 0 no apunta a X: se quita esa flecha.
        if cond:
            g2 = DAG([[], [], [0, 1]], g.cpt, g.names)
            ws2, j2 = g2.worlds(), g2.joint()
            obs1 = (sum(p for w, p in zip(ws2, j2) if w[1] == 1 and w[2] == 1)
                    / sum(p for w, p in zip(ws2, j2) if w[1] == 1))
            obs0 = (sum(p for w, p in zip(ws2, j2) if w[1] == 0 and w[2] == 1)
                    / sum(p for w, p in zip(ws2, j2) if w[1] == 0))
            do = g2.ate(1, 2)
        out.append(dict(rule=2, case=tag, holds=cond, observational=float(obs1 - obs0),
                        interventional=float(do), residual=float(abs(obs1 - obs0 - do))))

    # Regla 3: intervenir en una variable que no tiene ningún camino causal
    # hacia Y no cambia nada.
    g3 = demo_dag(np.random.default_rng(SEED + 4))
    base = g3.ate(2, 4)
    ws, j = g3.worlds(), g3.joint()
    p_y = sum(p for w, p in zip(ws, j) if w[4] == 1)
    p_y_do_d = g3.p_do(5, 1, 4)      # D es un descendiente de Y: no la toca
    p_y_do_b = g3.p_do(0, 1, 4)      # B sí tiene camino hasta Y
    out.append(dict(rule=3, case="no causal path (D)", holds=True,
                    observational=float(p_y), interventional=float(p_y_do_d),
                    residual=float(abs(p_y - p_y_do_d))))
    out.append(dict(rule=3, case="causal path (B)", holds=False,
                    observational=float(p_y), interventional=float(p_y_do_b),
                    residual=float(abs(p_y - p_y_do_b))))
    return dict(rows=out, base=float(base))


# =====================================================================
#  Segunda mitad: para quién funciona, con tau(V) escrito
# =====================================================================
def het_world(n, rng, prognostic=1.0):
    """El mundo heterogéneo.

    tau(V) = 0.5 + 2 * 1[V0 > 0] * sigmoide(2 V1): un salto y una rampa, que es
    la forma que separa un método que puede partir el espacio de uno que no.
    El término pronóstico es lo que hace difícil el problema: mueve mucho a Y y
    no tiene nada que ver con el efecto, así que un método que modele Y de
    frente gasta toda su capacidad ahí.
    """
    V = rng.standard_normal((n, 5))
    tau = 0.5 + 2.0 * (V[:, 0] > 0) / (1.0 + np.exp(-2.0 * V[:, 1]))
    e = 1.0 / (1.0 + np.exp(-(0.8 * V[:, 2] + 0.6 * V[:, 0])))
    X = (rng.random(n) < e).astype(float)
    prog = prognostic * (2.0 * V[:, 2] + 1.5 * np.sin(2.0 * V[:, 3]) + V[:, 4])
    Y = 1.0 + prog + tau * X + rng.standard_normal(n)
    return V, X, Y, tau, e


class CausalTree:
    """Un árbol causal con estimación honesta (Athey e Imbens, 2016).

    Dos diferencias con un árbol de regresión, y las dos importan:

      1. **El criterio parte por heterogeneidad del efecto**, no por error de
         predicción: se maximiza n_L*tau_L^2 + n_R*tau_R^2, donde cada tau es
         una diferencia de medias dentro de la hoja. Un árbol de regresión
         normal partiría por lo que mueve a Y, que aquí es el término
         pronóstico y no tiene nada que ver con el efecto.
      2. **La hoja se estima con datos que no eligieron el corte.** La muestra
         se parte en dos: una mitad decide dónde cortar y la otra rellena los
         valores. Sin eso, el efecto de una hoja es el máximo de un montón de
         diferencias ruidosas y sale sesgado hacia fuera, que es justo lo que
         la tabla de honestidad mide.
    """

    def __init__(self, max_depth=4, min_leaf=40, honest=True, rng=None,
                 centered=False):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.honest = honest
        self.centered = centered
        self.rng = rng or np.random.default_rng(0)

    def _tau(self, X, Y, idx):
        """El efecto de una hoja, de una de las dos maneras.

        Sin centrar es la diferencia de medias dentro de la hoja, que es lo que
        dice el árbol causal original y **hereda toda la confusión** que quede
        dentro de la hoja: si el tratamiento se reparte según una covariable
        por la que el árbol no ha partido, la diferencia de medias de esa hoja
        está sesgada exactamente igual que la diferencia de medias global del
        primer artículo de la rama.

        Centrado (que es lo que hace un bosque causal moderno) es el
        coeficiente de la regresión de los residuos: se le quita antes a Y su
        parte predecible y a X la suya, y dentro de la hoja se estima
        sum(rx*ry)/sum(rx^2). Medirlo en las dos versiones es lo que convierte
        "el bosque va mal" en "sin centrar, un bosque causal es un estimador
        confundido".
        """
        if self.centered:
            rx, ry = X[idx], Y[idx]
            den = float((rx * rx).sum())
            if den < 1e-8 or idx.size < 4:
                return None
            return float((rx * ry).sum() / den)
        a, b = Y[idx][X[idx] == 1], Y[idx][X[idx] == 0]
        if a.size < 2 or b.size < 2:
            return None
        return a.mean() - b.mean()

    def fit(self, V, X, Y):
        n = len(Y)
        perm = self.rng.permutation(n)
        if self.honest:
            half = n // 2
            split_idx, est_idx = perm[:half], perm[half:]
        else:
            split_idx = est_idx = perm
        self.root = self._grow(V, X, Y, split_idx, est_idx, 0)
        return self

    def _grow(self, V, X, Y, sidx, eidx, depth):
        tau = self._tau(X, Y, eidx)
        node = dict(leaf=True, tau=tau if tau is not None else 0.0, n=len(eidx))
        if depth >= self.max_depth or len(sidx) < 4 * self.min_leaf:
            return node
        best = None
        for j in range(V.shape[1]):
            vals = np.quantile(V[sidx, j], np.linspace(0.1, 0.9, 9))
            for t in np.unique(vals):
                ls = sidx[V[sidx, j] <= t]
                rs = sidx[V[sidx, j] > t]
                if len(ls) < self.min_leaf or len(rs) < self.min_leaf:
                    continue
                tl, tr = self._tau(X, Y, ls), self._tau(X, Y, rs)
                if tl is None or tr is None:
                    continue
                score = len(ls) * tl ** 2 + len(rs) * tr ** 2
                if best is None or score > best[0]:
                    best = (score, j, t, ls, rs)
        if best is None:
            return node
        _, j, t, ls, rs = best
        le, re = eidx[V[eidx, j] <= t], eidx[V[eidx, j] > t]
        if len(le) < self.min_leaf // 2 or len(re) < self.min_leaf // 2:
            return node
        return dict(leaf=False, j=j, t=float(t),
                    left=self._grow(V, X, Y, ls, le, depth + 1),
                    right=self._grow(V, X, Y, rs, re, depth + 1))

    def predict(self, V):
        out = np.empty(len(V))
        for i, v in enumerate(V):
            node = self.root
            while not node["leaf"]:
                node = node["left"] if v[node["j"]] <= node["t"] else node["right"]
            out[i] = node["tau"]
        return out


def cross_fitted(V, X, Y, folds=4, seed=0):
    """m(V) = E[Y|V] y e(V) = E[X|V], ajustadas fuera de muestra.

    Fuera de muestra y no dentro: un bosque que ha visto la fila que va a
    residualizar se lleva parte de la señal con el ruido, y el residuo que
    queda ya no es el que la teoría nombra.
    """
    rng = np.random.default_rng(seed)
    n = len(Y)
    order = rng.permutation(n)
    my, mx = np.empty(n), np.empty(n)
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        my[te] = RandomForestRegressor(n_estimators=150, min_samples_leaf=8,
                                       random_state=f, n_jobs=1).fit(
            V[tr], Y[tr]).predict(V[te])
        mx[te] = RandomForestRegressor(n_estimators=150, min_samples_leaf=8,
                                       random_state=100 + f, n_jobs=1).fit(
            V[tr], X[tr]).predict(V[te])
    return my, mx


class CausalForest:
    def __init__(self, n_trees=200, honest=True, seed=0, centered=False, **kw):
        self.trees = []
        self.n_trees = n_trees
        self.honest = honest
        self.seed = seed
        self.centered = centered
        self.kw = kw

    def fit(self, V, X, Y):
        rng = np.random.default_rng(self.seed)
        n = len(Y)
        if self.centered:
            my, mx = cross_fitted(V, X, Y, seed=self.seed)
            A, B = X - mx, Y - my
        else:
            A, B = X, Y
        for b in range(self.n_trees):
            idx = rng.choice(n, size=int(0.6 * n), replace=False)
            t = CausalTree(honest=self.honest, centered=self.centered,
                           rng=np.random.default_rng(self.seed + b), **self.kw)
            t.fit(V[idx], A[idx], B[idx])
            self.trees.append(t)
        return self

    def predict(self, V):
        return np.mean([t.predict(V) for t in self.trees], axis=0)


def s_learner(V, X, Y, Vt):
    f = RandomForestRegressor(n_estimators=200, min_samples_leaf=8, random_state=0,
                              n_jobs=1).fit(np.c_[V, X], Y)
    one = np.c_[Vt, np.ones(len(Vt))]
    zero = np.c_[Vt, np.zeros(len(Vt))]
    return f.predict(one) - f.predict(zero)


def t_learner(V, X, Y, Vt):
    f1 = RandomForestRegressor(n_estimators=200, min_samples_leaf=8, random_state=0,
                               n_jobs=1).fit(V[X == 1], Y[X == 1])
    f0 = RandomForestRegressor(n_estimators=200, min_samples_leaf=8, random_state=1,
                               n_jobs=1).fit(V[X == 0], Y[X == 0])
    return f1.predict(Vt) - f0.predict(Vt)


def r_learner(V, X, Y, Vt, folds=4, seed=0):
    """Residualizar los dos lados y regresar uno sobre otro, con cross-fitting.

    Es la forma de Robinson y la que usa por dentro el bosque causal moderno:
    quitarle a Y su parte predecible y a X la suya deja una regresión cuyo
    coeficiente local es el efecto, y el cross-fitting evita que el modelo que
    quita la parte predecible se quede también con parte de la señal.
    """
    rng = np.random.default_rng(seed)
    n = len(Y)
    order = rng.permutation(n)
    my = np.empty(n)
    mx = np.empty(n)
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        my[te] = RandomForestRegressor(n_estimators=150, min_samples_leaf=8,
                                       random_state=f, n_jobs=1).fit(
            V[tr], Y[tr]).predict(V[te])
        mx[te] = RandomForestRegressor(n_estimators=150, min_samples_leaf=8,
                                       random_state=100 + f, n_jobs=1).fit(
            V[tr], X[tr]).predict(V[te])
    ry, rx = Y - my, X - mx
    # El paso final: un bosque que predice el pseudo-efecto ry/rx con peso rx^2.
    w = rx ** 2
    keep = w > 1e-8
    pseudo = np.zeros(n)
    pseudo[keep] = ry[keep] / rx[keep]
    f = RandomForestRegressor(n_estimators=250, min_samples_leaf=12, random_state=7,
                              n_jobs=1).fit(V[keep], pseudo[keep], sample_weight=w[keep])
    return f.predict(Vt)


def stage_heterogeneity():
    """Los cuatro métodos puntuados contra tau(V), con suelo de semillas."""
    rows = {k: [] for k in ["s", "t", "r", "cf_centered", "cf", "cf_adaptive",
                            "constant"]}
    corr = {k: [] for k in rows}
    for rep in range(HET_REPS):
        rng = np.random.default_rng(SEED + 300 + rep)
        V, X, Y, tau, e = het_world(N_HET, rng)
        Vt, Xt, Yt, taut, et = het_world(2000, rng)
        preds = dict(
            s=s_learner(V, X, Y, Vt),
            t=t_learner(V, X, Y, Vt),
            r=r_learner(V, X, Y, Vt, seed=rep),
            cf_centered=CausalForest(n_trees=120, honest=True, centered=True,
                                     seed=rep).fit(V, X, Y).predict(Vt),
            cf=CausalForest(n_trees=120, honest=True, seed=rep).fit(V, X, Y).predict(Vt),
            cf_adaptive=CausalForest(n_trees=120, honest=False, seed=rep)
            .fit(V, X, Y).predict(Vt),
        )
        # El brazo que hay que batir: contestar el mismo número a todo el mundo.
        preds["constant"] = np.full(len(Vt), float(tau.mean()))
        for k, p in preds.items():
            rows[k].append(float(np.mean((p - taut) ** 2)))
            corr[k].append(float(np.corrcoef(p, taut)[0, 1])
                           if np.std(p) > 1e-12 else np.nan)
    out = []
    labels = dict(s="S-learner", t="T-learner", r="R-learner",
                  cf_centered="causal forest, honest and centred",
                  cf="causal forest, honest, not centred",
                  cf_adaptive="causal forest, adaptive, not centred",
                  constant="one number for everybody")
    for k, v in rows.items():
        a = np.array(v)
        # El brazo constante no tiene correlación: su predicción no varía, así
        # que el denominador es cero. Viaja como None y la prosa tiene su rama,
        # en vez de imprimir NaN en una tabla.
        c = np.array(corr[k], dtype=float)
        has_corr = bool(np.isfinite(c).all())
        out.append(dict(key=k, label=labels[k], mse=float(a.mean()),
                        sd=float(a.std(ddof=1)),
                        corr=float(c.mean()) if has_corr else None,
                        corr_sd=float(c.std(ddof=1)) if has_corr else None))
    return dict(rows=out, reps=HET_REPS, n=N_HET)


def stage_prognostic_sweep():
    """El fallo conocido del S-learner, medido: cuando el término pronóstico
    crece, el efecto se le encoge hacia cero porque el bosque gasta todos sus
    cortes en lo que mueve a Y. El T-learner y el bosque causal no dependen de
    eso de la misma manera, y la curva lo enseña."""
    rows = []
    for prog in [0.0, 0.5, 1.0, 2.0, 4.0]:
        got = {k: [] for k in ["s", "t", "cf"]}
        for rep in range(3):
            rng = np.random.default_rng(SEED + 700 + rep)
            V, X, Y, tau, e = het_world(N_HET, rng, prognostic=prog)
            Vt, _, _, taut, _ = het_world(1500, rng, prognostic=prog)
            got["s"].append(float(np.mean((s_learner(V, X, Y, Vt) - taut) ** 2)))
            got["t"].append(float(np.mean((t_learner(V, X, Y, Vt) - taut) ** 2)))
            got["cf"].append(float(np.mean(
                (CausalForest(n_trees=100, honest=True, centered=True, seed=rep)
                 .fit(V, X, Y).predict(Vt) - taut) ** 2)))
        rows.append(dict(prognostic=prog, **{k: float(np.mean(v)) for k, v in got.items()},
                         **{f"{k}_sd": float(np.std(v, ddof=1)) for k, v in got.items()}))
    return rows


def stage_het_cloud():
    """Lo que viaja: una muestra con tau verdadero y las predicciones."""
    rng = np.random.default_rng(SEED + 900)
    V, X, Y, tau, e = het_world(N_HET, rng)
    Vt, Xt, Yt, taut, et = het_world(600, rng)
    cf = CausalForest(n_trees=150, honest=True, centered=True, seed=0).fit(V, X, Y)
    my, mx = cross_fitted(V, X, Y, seed=0)
    tree = CausalTree(max_depth=3, honest=True, centered=True,
                      rng=np.random.default_rng(3)).fit(V, X - mx, Y - my)
    preds = dict(cf=cf.predict(Vt), s=s_learner(V, X, Y, Vt), t=t_learner(V, X, Y, Vt))

    def node(nd, names):
        if nd["leaf"]:
            return dict(leaf=True, tau=r(nd["tau"]), n=nd["n"])
        return dict(leaf=False, j=nd["j"], name=names[nd["j"]], t=r(nd["t"], 4),
                    left=node(nd["left"], names), right=node(nd["right"], names))

    names = [f"V{i}" for i in range(5)]
    return dict(
        V=r(Vt[:, :2], 4), tau=r(taut), x=[int(v) for v in Xt], y=r(Yt, 4),
        pred={k: r(v) for k, v in preds.items()},
        tree=node(tree.root, names), names=names,
        mse={k: r(float(np.mean((v - taut) ** 2))) for k, v in preds.items()},
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exact = cached("exact", stage_exact)
    print(f"  demo DAG: truth {exact['truth']:.6f}, {exact['n_sets']} adjustment sets, "
          f"{exact['n_valid']} valid, worst valid error {exact['worst_valid_error']:.2e}")
    print(f"    adjust for everything: {exact['full']['estimate']:.6f} "
          f"(error {exact['full']['error']:.4f}, backdoor {exact['full']['backdoor']})")
    print(f"    adjust for nothing:    {exact['empty']['estimate']:.6f} "
          f"(error {exact['empty']['error']:.4f})")

    randoms = cached("random", stage_random_dags)
    print(f"  {randoms['n']} random DAGs: adjust-for-everything valid "
          f"{randoms['full_rate']:.3f}, some valid set exists {randoms['any_rate']:.3f}, "
          f"median error when wrong {randoms['err_median']:.4f}")

    front = cached("frontdoor", stage_frontdoor)
    print(f"  front door: truth {front['truth']:.6f}, formula {front['frontdoor']:.6f}, "
          f"error {front['error']:.2e}, naive {front['naive']:.6f}")

    rules = cached("rules", stage_rules)
    for row in rules["rows"]:
        print(f"    rule {row['rule']} {row['case']:<22} residual {row['residual']:.2e}")

    het = cached("het", stage_heterogeneity)
    for row in het["rows"]:
        corr = "n/a (constant)" if row["corr"] is None else f"{row['corr']:.3f}"
        print(f"    {row['label']:<38} mse {row['mse']:.4f} (sd {row['sd']:.4f})  "
              f"corr {corr}")
    sweep = cached("prognostic", stage_prognostic_sweep)
    cloud = cached("cloud", stage_het_cloud)

    assert exact["worst_valid_error"] < 1e-12, \
        "un conjunto que cumple la puerta trasera no reproduce la verdad"
    assert exact["full"]["error"] > 1e-6, \
        "en el grafo de ejemplo ajustar por todo acierta, y entonces no enseña nada"
    assert front["error"] < 1e-12, "la puerta delantera no cuadra con la verdad"
    assert all(row["residual"] < 1e-12 for row in rules["rows"] if row["holds"]), \
        "una regla con su condición cumplida no da una identidad"
    assert all(row["residual"] > 1e-6 for row in rules["rows"] if not row["holds"]), \
        "una regla sin su condición da lo mismo, así que la condición no se ve"
    best = min(het["rows"], key=lambda d: d["mse"])
    assert best["key"] != "constant", "ningún método bate a contestar la media"
    by = {row["key"]: row for row in het["rows"]}
    assert by["cf_centered"]["mse"] < by["cf"]["mse"], \
        "centrar no mejora al bosque causal, y la sección dice que sí"
    assert abs(exact["truth"]) > 0.15, "el efecto del grafo de ejemplo no se ve"

    data = dict(
        meta=dict(seed=SEED, n_random_dags=randoms["n"], n_het=N_HET,
                  het_reps=HET_REPS),
        exact=exact, random=randoms, frontdoor=front, rules=rules,
        heterogeneity=dict(reps=het["reps"], n=het["n"],
                           rows=[{k: (v if isinstance(v, str) else r(v))
                                  for k, v in row.items()} for row in het["rows"]]),
        prognostic=[{k: r(v) for k, v in row.items()} for row in sweep],
        cloud=cloud,
    )
    path = OUT / "docalculus.json"
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
