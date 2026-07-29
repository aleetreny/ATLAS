"""Todo lo que cita el artículo 5.1c (recocido simulado y enjambre), medido.

Dos métodos que se presentan siempre con su metáfora (un metal que se enfría,
un banco de peces) y que tienen debajo dos afirmaciones matemáticas concretas.
Las dos se comprueban aquí en vez de citarse.

 1. Un paseo de Metropolis a temperatura fija **muestrea la distribución de
    Boltzmann**. Sobre un paisaje de 64 estados donde esa distribución se puede
    escribir exactamente, se mide la distancia entre lo que visita la cadena y
    lo que dice la fórmula, y la tasa de aceptación medida contra la predicha.
 2. Los horarios de enfriamiento, al mismo presupuesto, sobre las MISMAS 40
    ciudades del artículo anterior (se afirma contra su fichero, no se declara),
    incluido el horario logarítmico, que es el único con demostración de
    convergencia y el peor de todos a cualquier presupuesto real.
 3. El enjambre contra dos controles que no son enjambres, sobre cuatro
    funciones de prueba, y el barrido de sus dos coeficientes contra la región
    de estabilidad que la teoría predice: fuera de ella las partículas divergen,
    y eso se cuenta.
 4. Una tirada de referencia con generador congruencial escrito a mano, que el
    navegador reproduce paso a paso.

Tarda unos cuatro minutos en cuatro núcleos. Etapas cacheadas en
~/.atlas_vision_data/anneal_cache/.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "annealing-swarm" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "anneal_cache"
CACHE.mkdir(parents=True, exist_ok=True)

SEED = 60
SEEDS = 30


def cached(name, fn):
    f = CACHE / f"{name}.json"
    if f.is_file():
        return json.loads(f.read_text())
    t0 = time.time()
    out = fn()
    f.write_text(json.dumps(out))
    print(f"  [{name}] {time.time() - t0:.1f}s")
    return out


def r(x, k=4):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, k) for v in np.asarray(x, dtype=float).tolist()]
    return round(float(x), k)


# ---------------------------------------------------------------------------
# 1. El paisaje discreto donde la distribución de Boltzmann se puede escribir
# ---------------------------------------------------------------------------
STATES = 64


def ring_energy():
    """Un paisaje de 64 estados en un anillo, con dos valles de profundidad
    distinta y una barrera entre ellos. Todo lo que hace falta es que se pueda
    sumar exactamente sobre los 64, que es lo que convierte la afirmación de
    Boltzmann en una comparación y no en una intuición."""
    i = np.arange(STATES)
    x = 2 * np.pi * i / STATES
    e = 1.8 * np.cos(x) + 0.9 * np.cos(3 * x + 0.7) + 0.4 * np.cos(5 * x)
    return e - e.min()


ENERGY = ring_energy()


def boltzmann(T):
    w = np.exp(-ENERGY / T)
    return w / w.sum()


def metropolis_chain(T, steps, rng):
    """Propone un paso a un vecino del anillo y lo acepta con la regla de
    Metropolis. Devuelve el histograma de visitas y las cuentas de aceptación."""
    counts = np.zeros(STATES)
    i = int(rng.integers(STATES))
    up_prop, up_acc = 0, 0
    for _ in range(steps):
        j = (i + (1 if rng.random() < 0.5 else -1)) % STATES
        d = ENERGY[j] - ENERGY[i]
        if d <= 0:
            i = j
        else:
            up_prop += 1
            if rng.random() < math.exp(-d / T):
                i = j
                up_acc += 1
        counts[i] += 1
    return counts / counts.sum(), up_prop, up_acc


def stage_boltzmann():
    rows = []
    for T in (0.2, 0.5, 1.0, 2.0, 5.0):
        rng = np.random.default_rng(SEED)
        steps = 400000
        visits, up_prop, up_acc = metropolis_chain(T, steps, rng)
        exact = boltzmann(T)
        tv = 0.5 * float(np.abs(visits - exact).sum())
        # La aceptación que la regla predice. La media simple sobre todas las
        # aristas de subida es la cifra equivocada y se nota (0.383 contra 0.468
        # medido a T = 0.2): la cadena no visita todos los estados por igual, así
        # que cada arista pesa lo que pesa su estado de salida en la propia
        # distribución de Boltzmann. Con ese peso, la predicción es la
        # probabilidad condicional de aceptar dado que se ha propuesto subir.
        num, den = 0.0, 0.0
        for i in range(STATES):
            for j in ((i + 1) % STATES, (i - 1) % STATES):
                d = ENERGY[j] - ENERGY[i]
                if d > 0:
                    num += exact[i] * 0.5 * math.exp(-d / T)
                    den += exact[i] * 0.5
        rows.append({
            "T": T,
            "tv_distance": r(tv, 6),
            "accept_measured": r(up_acc / max(up_prop, 1), 6),
            "accept_predicted": r(num / den, 6),
            "visits": r(visits, 6),
            "exact": r(exact, 6),
            "steps": steps,
            "in_lowest": r(float(exact[ENERGY < np.quantile(ENERGY, 0.1)].sum()), 4),
        })
    # Lo que queda de distancia, ¿es sesgo o es muestreo? Se distingue mirando
    # cómo cae con la longitud de la cadena: el error de muestreo va como la
    # raíz de N y un sesgo no se mueve. Se mide y se ajusta el exponente.
    exact1 = boltzmann(1.0)
    lengths, dists = [], []
    for steps in (25000, 50000, 100000, 200000, 400000, 800000):
        rep = []
        for s in range(4):
            rng = np.random.default_rng(SEED + 100 + s)
            v, _, _ = metropolis_chain(1.0, steps, rng)
            rep.append(0.5 * float(np.abs(v - exact1).sum()))
        lengths.append(steps)
        dists.append(float(np.mean(rep)))
    slope = float(np.polyfit(np.log(lengths), np.log(dists), 1)[0])
    return {"rows": rows, "energy": r(ENERGY, 5), "states": STATES,
            "convergence": {"steps": lengths, "distance": r(dists, 6),
                            "slope": r(slope, 4), "predicted_slope": -0.5},
            "noise_floor": r(dists[-1], 6)}


# ---------------------------------------------------------------------------
# 2. Los horarios, sobre las ciudades del artículo anterior
# ---------------------------------------------------------------------------
def load_cities():
    f = ROOT / "genetic-algorithms" / "data" / "ga.json"
    d = json.loads(f.read_text())
    pts = np.array(d["tsp"]["points"])
    return pts, d["tsp"]


def tour_length(order, D):
    return float(D[order, np.roll(order, -1)].sum())


def sa_tsp(D, rng, budget, schedule, T0, T1):
    """Recocido con el movimiento estándar de 2-opt: invertir un tramo.

    El coste de un movimiento se calcula de las cuatro aristas que cambian, no
    recorriendo el tour: eso hace que una evaluación sea una evaluación y que la
    comparación con el artículo anterior sea justa.
    """
    n = D.shape[0]
    order = rng.permutation(n)
    length = tour_length(order, D)
    best = length
    best_order = order.copy()
    accepted, uphill_prop, uphill_acc = 0, 0, 0
    trace = []
    for k in range(budget):
        frac = k / budget
        if schedule == "geometric_fast":
            T = T0 * (T1 / T0) ** frac
        elif schedule == "geometric_slow":
            T = T0 * (T1 / T0) ** min(1.0, frac * 4)
        elif schedule == "linear":
            T = T0 + (T1 - T0) * frac
        elif schedule == "hot":
            T = T0
        elif schedule == "cold":
            T = T1
        elif schedule == "log":
            # el horario con demostración: T = c / log(2 + k)
            T = T0 * math.log(2) / math.log(2 + k)
        else:
            raise ValueError(schedule)
        i = int(rng.integers(0, n - 2))
        j = int(rng.integers(i + 2, n))
        if i == 0 and j == n - 1:
            continue                       # invertir el tour entero no cambia nada
        a, b = order[i], order[(i + 1) % n]
        c, d = order[j % n], order[(j + 1) % n]
        delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
        take = delta < 0
        if not take:
            uphill_prop += 1
            if T > 1e-12 and rng.random() < math.exp(-delta / T):
                take = True
                uphill_acc += 1
        if take:
            order[i + 1:j + 1] = order[i + 1:j + 1][::-1]
            length += delta
            accepted += 1
            if length < best:
                best = length
                best_order = order.copy()
        if k % (budget // 200) == 0:
            trace.append([k, r(length, 4), r(T, 5)])
    return {"best": best, "order": best_order, "accepted": accepted,
            "uphill_share": uphill_acc / max(uphill_prop, 1), "trace": trace}


SCHEDULES = {
    "geometric_fast": "geometric, cools over the whole run",
    "geometric_slow": "geometric, cold after a quarter",
    "linear": "linear",
    "hot": "no cooling at all, stays hot",
    "cold": "no heat at all, downhill only",
    "log": "the schedule with the proof",
}


def stage_schedules():
    pts, ga = load_cities()
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    n = len(pts)
    budget = ga["budget"]
    # una temperatura inicial calibrada: la desviación de los cambios de coste
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(4000):
        order = rng.permutation(n)
        i = int(rng.integers(0, n - 3))
        j = int(rng.integers(i + 2, n))
        a, b = order[i], order[i + 1]
        c, d = order[j], order[(j + 1) % n]
        deltas.append(abs(D[a, c] + D[b, d] - D[a, b] - D[c, d]))
    T0 = float(np.mean(deltas))
    T1 = T0 / 1000
    rows = []
    traces = {}
    tours = {}
    for key in SCHEDULES:
        lens = []
        for s in range(12):
            rng = np.random.default_rng(SEED * 3 + s)
            out = sa_tsp(D, rng, budget, key, T0, T1)
            lens.append(out["best"])
            if s == 0:
                traces[key] = out["trace"]
            if out["best"] <= min(lens):
                tours[key] = out["order"].tolist()
        rows.append({"schedule": key, "label": SCHEDULES[key],
                     "mean": r(float(np.mean(lens)), 4),
                     "best": r(float(min(lens)), 4),
                     "spread": r(float(np.std(lens)), 4),
                     "uphill_share": r(out["uphill_share"], 4)})
    return {
        "rows": rows, "traces": traces, "tours": tours,
        "T0": r(T0, 5), "T1": r(T1, 6), "budget": budget, "seeds": 12,
        "cities": r(pts, 4), "n": n,
        "reference": {"two_opt": ga["two_opt"]["mean"], "ga": ga["ga"]["mean"],
                      "random": ga["random"]["mean"],
                      "two_opt_best": ga["two_opt"]["best"]},
        "same_instance": True,
    }


# ---------------------------------------------------------------------------
# 3. El enjambre
# ---------------------------------------------------------------------------
def f_sphere(X):
    return (X ** 2).sum(axis=-1)


def f_rastrigin(X):
    return 10 * X.shape[-1] + (X ** 2 - 10 * np.cos(2 * np.pi * X)).sum(axis=-1)


def f_rosenbrock(X):
    return (100 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (1 - X[..., :-1]) ** 2).sum(axis=-1)


def f_ackley(X):
    d = X.shape[-1]
    return (-20 * np.exp(-0.2 * np.sqrt((X ** 2).sum(axis=-1) / d))
            - np.exp(np.cos(2 * np.pi * X).sum(axis=-1) / d) + 20 + np.e)


BENCH = {
    "sphere": (f_sphere, 5.12, 0.0, "one bowl"),
    "rastrigin": (f_rastrigin, 5.12, 0.0, "a bowl with a lattice of dents"),
    "rosenbrock": (f_rosenbrock, 2.048, 0.0, "a banana shaped valley"),
    "ackley": (f_ackley, 32.0, 0.0, "a needle in a nearly flat plain"),
}


def pso(fn, bound, rng, dim, budget, n_part=30, w=0.729, c1=1.494, c2=1.494):
    X = rng.uniform(-bound, bound, size=(n_part, dim))
    V = rng.uniform(-bound, bound, size=(n_part, dim)) * 0.1
    F = fn(X)
    evals = n_part
    pbest, pf = X.copy(), F.copy()
    g = int(pf.argmin())
    gbest, gf = pbest[g].copy(), float(pf[g])
    spread = [float(np.linalg.norm(X - X.mean(axis=0), axis=1).mean())]
    while evals < budget:
        r1 = rng.random((n_part, dim))
        r2 = rng.random((n_part, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V
        F = fn(X)
        evals += n_part
        better = F < pf
        pbest[better] = X[better]
        pf[better] = F[better]
        if pf.min() < gf:
            g = int(pf.argmin())
            gbest, gf = pbest[g].copy(), float(pf[g])
        spread.append(float(np.linalg.norm(X - X.mean(axis=0), axis=1).mean()))
    return gf, gbest, spread


def restart_local(fn, bound, rng, dim, budget):
    """Control: descenso por coordenadas con reinicios, que no sabe nada de
    enjambres y sí de que la función es continua."""
    best, evals = np.inf, 0
    while evals < budget:
        x = rng.uniform(-bound, bound, size=dim)
        fx = float(fn(x))
        evals += 1
        step = bound / 4
        while step > 1e-6 and evals < budget:
            improved = False
            for i in range(dim):
                for s in (+1, -1):
                    y = x.copy()
                    y[i] += s * step
                    fy = float(fn(y))
                    evals += 1
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                    if evals >= budget:
                        break
                if evals >= budget:
                    break
            if not improved:
                step /= 2
        best = min(best, fx)
    return best


def random_search(fn, bound, rng, dim, budget):
    best = np.inf
    left = budget
    while left > 0:
        chunk = min(5000, left)
        X = rng.uniform(-bound, bound, size=(chunk, dim))
        best = min(best, float(fn(X).min()))
        left -= chunk
    return best


def stage_swarm():
    dim, budget = 10, 30000
    out = {}
    for name, (fn, bound, opt, label) in BENCH.items():
        rows = []
        for arm in ("pso", "local", "random"):
            vals = []
            for s in range(SEEDS):
                rng = np.random.default_rng(SEED * 11 + s)
                if arm == "pso":
                    v, _, _ = pso(fn, bound, rng, dim, budget)
                elif arm == "local":
                    v = restart_local(fn, bound, rng, dim, budget)
                else:
                    v = random_search(fn, bound, rng, dim, budget)
                vals.append(v)
            rows.append({"arm": arm, "median": r(float(np.median(vals)), 6),
                         "best": r(float(np.min(vals)), 6),
                         "spread": r(float(np.std(vals)), 6),
                         "hit": int(sum(1 for v in vals if v < 1e-6))})
        out[name] = {"label": label, "rows": rows, "optimum": opt, "bound": bound}
    return {"bench": out, "dim": dim, "budget": budget, "seeds": SEEDS,
            "particles": 30}


def stage_stability():
    """La región de estabilidad: la teoría dice que las partículas divergen si
    c1 + c2 pasa de 24(1 - w^2) / (7 - 5w). Se barre y se cuenta.

    Con un matiz que hay que medir en vez de esquivar: ese criterio se deriva
    con los dos atractores QUIETOS, y en un enjambre de verdad el mejor personal
    y el mejor global se mueven con la partícula, lo que la frena. Así que lo
    que se mide no es solo si algo se desborda, sino cuánto se estira la nube,
    que es la cantidad de la que habla el criterio.
    """
    dim, budget = 10, 6000
    fn, bound = f_sphere, 5.12
    rows = []
    for w in (0.2, 0.4, 0.6, 0.729, 0.9):
        limit = 24 * (1 - w ** 2) / (7 - 5 * w)
        for c in (0.5, 1.0, 1.494, 2.0, 2.5, 3.0, 4.0):
            diverged, vals, ratios = 0, [], []
            for s in range(10):
                rng = np.random.default_rng(SEED * 13 + s)
                v, gbest, spread = pso(fn, bound, rng, dim, budget, w=w, c1=c, c2=c)
                if not np.isfinite(v) or spread[-1] > 1e6:
                    diverged += 1
                ratios.append(spread[-1] / max(spread[0], 1e-12))
                vals.append(v if np.isfinite(v) else 1e30)
            rows.append({"w": w, "c": c, "sum_c": r(2 * c, 3), "limit": r(limit, 4),
                         "predicted_stable": bool(2 * c < limit),
                         "diverged": diverged, "of": 10,
                         "spread_ratio": r(float(np.median(ratios)), 10),
                         "median": r(float(np.median(vals)), 6)})
    agree = sum(1 for row in rows
                if (row["diverged"] == 0) == row["predicted_stable"])
    stable = [row for row in rows if row["predicted_stable"]]
    unstable = [row for row in rows if not row["predicted_stable"]]
    return {"rows": rows, "agree": agree, "of": len(rows), "dim": dim,
            "budget": budget,
            "stable_median": r(float(np.median([row["median"] for row in stable])), 6),
            "unstable_median": r(float(np.median([row["median"] for row in unstable])), 6),
            "stable_spread": r(float(np.median([row["spread_ratio"] for row in stable])), 10),
            "unstable_spread": r(float(np.median([row["spread_ratio"] for row in unstable])), 4),
            "any_overflow": bool(sum(row["diverged"] for row in rows) > 0)}


def stage_swarm_shape():
    """Una tirada de enjambre en dos dimensiones, para dibujarla: posiciones de
    cada partícula en cada paso, y cómo se encoge la nube."""
    rng = np.random.default_rng(SEED + 5)
    fn, bound = f_rastrigin, 5.12
    n_part, steps = 20, 60
    X = rng.uniform(-bound, bound, size=(n_part, 2))
    V = rng.uniform(-bound, bound, size=(n_part, 2)) * 0.1
    F = fn(X)
    pbest, pf = X.copy(), F.copy()
    g = int(pf.argmin())
    gbest, gf = pbest[g].copy(), float(pf[g])
    frames, spreads = [], []
    for _ in range(steps):
        frames.append(r(X, 3))
        spreads.append(r(float(np.linalg.norm(X - X.mean(axis=0), axis=1).mean()), 4))
        r1, r2 = rng.random((n_part, 2)), rng.random((n_part, 2))
        V = 0.729 * V + 1.494 * r1 * (pbest - X) + 1.494 * r2 * (gbest - X)
        X = X + V
        F = fn(X)
        better = F < pf
        pbest[better] = X[better]
        pf[better] = F[better]
        if pf.min() < gf:
            g = int(pf.argmin())
            gbest, gf = pbest[g].copy(), float(pf[g])
    return {"frames": frames, "spreads": spreads, "best": r(gbest, 4),
            "best_value": r(gf, 6), "bound": bound, "particles": n_part,
            "steps": steps}


# ---------------------------------------------------------------------------
# 4. La tirada de referencia, con el azar escrito a mano
# ---------------------------------------------------------------------------
class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next_u32(self):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s

    def random(self):
        return self.next_u32() / 4294967296.0

    def randint(self, n):
        return self.next_u32() % n


REF = {"seed": 20260729, "steps": 4000, "T0": 1.5, "T1": 0.01}


def ref_chain(cfg):
    """El paseo de Metropolis sobre el anillo, escrito para reproducirse en el
    navegador: la energía viaja redondeada a 5 decimales y los dos lados leen
    exactamente ese array, así que la comparación es exacta."""
    rng = LCG(cfg["seed"])
    e = [round(float(v), 5) for v in ENERGY]
    i = rng.randint(STATES)
    path, accepted, uphill_acc, uphill_prop = [i], 0, 0, 0
    for k in range(cfg["steps"]):
        T = cfg["T0"] * (cfg["T1"] / cfg["T0"]) ** (k / cfg["steps"])
        j = (i + (1 if rng.random() < 0.5 else STATES - 1)) % STATES
        d = e[j] - e[i]
        take = d <= 0
        if not take:
            uphill_prop += 1
            if rng.random() < math.exp(-d / T):
                take = True
                uphill_acc += 1
        if take:
            i = j
            accepted += 1
        path.append(i)
    return {"config": cfg, "path": path[::10], "path_stride": 10,
            "final": i, "accepted": accepted,
            "uphill_accepted": uphill_acc, "uphill_proposed": uphill_prop,
            "energy": [round(float(v), 5) for v in ENERGY],
            "visits_tail": int(np.argmin(ENERGY))}


def main():
    t0 = time.time()
    print("1/5 Boltzmann")
    bol = cached("boltzmann-v2", stage_boltzmann)
    print("2/5 horarios sobre las ciudades del 5.1b")
    sch = cached("schedules", stage_schedules)
    print("3/5 el enjambre contra dos controles")
    sw = cached("swarm", stage_swarm)
    print("4/5 la región de estabilidad")
    st = cached("stability-v3", stage_stability)
    print("5/5 forma del enjambre y tirada de referencia")
    shape = cached("shape", stage_swarm_shape)
    ref = cached("reference", lambda: ref_chain(REF))

    data = {
        "meta": {"seed": SEED, "seeds": SEEDS, "states": STATES,
                 "note": "counts and distances, never a clock"},
        "boltzmann": bol,
        "schedules": sch,
        "swarm": sw,
        "stability": st,
        "shape": shape,
        "reference": ref,
    }
    f = OUT / "anneal.json"
    f.write_text(json.dumps(data, indent=1))
    print(f"escrito {f} ({f.stat().st_size / 1024:.1f} kB) en {time.time() - t0:.1f}s")

    print("\ncomprobaciones:")
    for row in bol["rows"]:
        print(f"  T={row['T']}: distancia {row['tv_distance']} (suelo "
              f"{bol['noise_floor']}), aceptación {row['accept_measured']} contra "
              f"{row['accept_predicted']}")
    for row in sch["rows"]:
        print(f"  {row['schedule']}: {row['mean']} (2-opt {sch['reference']['two_opt']}, "
              f"GA {sch['reference']['ga']})")
    print(f"  estabilidad: {st['agree']} de {st['of']} celdas de acuerdo con la teoría")
    for name, d in sw["bench"].items():
        print(f"  {name}: " + ", ".join(f"{row['arm']} {row['median']}" for row in d["rows"]))


if __name__ == "__main__":
    main()
