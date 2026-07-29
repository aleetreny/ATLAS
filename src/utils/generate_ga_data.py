"""Todo lo que cita el artículo 5.1b (algoritmos genéticos), medido.

La pregunta del artículo es la única que importa de este método y casi nunca se
contesta con una medida: **qué compra el cruce**. Se contesta poniendo al lado,
al mismo presupuesto de evaluaciones, la misma búsqueda sin cruce y una que no
es un algoritmo genético en absoluto.

 1. Cuatro paisajes con óptimo conocido por construcción, no estimado: contar
    unos, bloques todo o nada, trampas que engañan al gradiente local, y esas
    mismas trampas con los bits barajados, que es el experimento entero.
 2. Cinco buscadores al mismo presupuesto: cruce de un punto, cruce uniforme,
    solo mutación, escalada con reinicios y búsqueda al azar. Treinta semillas
    cada uno, tasa de éxito y evaluaciones medianas.
 3. La hipótesis de los bloques constructores, que es falsable: si el cruce de
    un punto gana porque conserva trozos contiguos, barajar las posiciones de
    los bits tiene que quitarle la ventaja sin tocar el paisaje. Se barajan.
 4. La presión de selección medida como tiempo de dominación, con el torneo de
    1 a 8, y contra la predicción logarítmica.
 5. El tamaño de población, barrido a presupuesto fijo, que es donde vive el
    compromiso de verdad.
 6. Un viajante de comercio, donde el algoritmo genético se mide contra 2-opt
    con reinicios al mismo número de evaluaciones.
 7. Una tirada de referencia con un generador congruencial escrito a mano, que
    el navegador reproduce paso a paso: todo son enteros, así que la
    comparación no tiene tolerancia.

Tarda unos tres minutos en cuatro núcleos. Etapas cacheadas en
~/.atlas_vision_data/ga_cache/. Ninguna cifra publicada es un reloj de pared.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "genetic-algorithms" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "ga_cache"
CACHE.mkdir(parents=True, exist_ok=True)

SEED = 59
SEEDS = 30
BUDGET = 60000
POP = 400          # calibrado abajo: es lo que hace falta para que las trampas se resuelvan


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
# Los paisajes. Cada uno tiene su óptimo escrito, no buscado.
# ---------------------------------------------------------------------------
BITS = 40
BLOCK = 4
BLOCKS = BITS // BLOCK


def f_onemax(X):
    return X.sum(axis=-1)


def f_royal(X):
    B = X.reshape(*X.shape[:-1], BLOCKS, BLOCK)
    return (B.all(axis=-1) * BLOCK).sum(axis=-1)


def _trap_block(u):
    """Trampa de k bits: el gradiente local apunta a los ceros, el óptimo son
    los unos. Vale k con el bloque lleno y k-1-u si no."""
    return np.where(u == BLOCK, BLOCK, BLOCK - 1 - u)


def f_trap(X):
    B = X.reshape(*X.shape[:-1], BLOCKS, BLOCK)
    return _trap_block(B.sum(axis=-1)).sum(axis=-1)


# El barajado de posiciones: el MISMO paisaje, con los bits de cada bloque
# repartidos por el cromosoma. Nada de la función cambia; solo dónde vive cada
# bit, que es exactamente lo que la hipótesis de los bloques dice que importa.
SHUFFLE = np.random.default_rng(SEED).permutation(BITS)


def f_trap_loose(X):
    return f_trap(X[..., SHUFFLE])


LANDSCAPES = {
    "onemax": (f_onemax, BITS, "count the ones"),
    "royal": (f_royal, BITS, "blocks, all or nothing"),
    "trap_tight": (f_trap, BLOCKS * BLOCK, "traps, blocks side by side"),
    "trap_loose": (f_trap_loose, BLOCKS * BLOCK, "traps, bits shuffled"),
}


# ---------------------------------------------------------------------------
# Los buscadores. Todos cuentan evaluaciones y todos paran al llegar al óptimo.
# ---------------------------------------------------------------------------
def ga(fn, optimum, rng, pop=POP, cross="one_point", pmut=None, tour=3,
       budget=BUDGET, n=BITS):
    pmut = 1.0 / n if pmut is None else pmut
    P = rng.integers(0, 2, size=(pop, n))
    fit = fn(P)
    evals = pop
    best = int(fit.max())
    if best >= optimum:
        return True, evals, best
    while evals < budget:
        # torneo
        idx = rng.integers(0, pop, size=(pop, tour))
        winners = idx[np.arange(pop), fit[idx].argmax(axis=1)]
        Q = P[winners].copy()
        if cross == "one_point":
            for i in range(0, pop - 1, 2):
                c = rng.integers(1, n)
                Q[i, c:], Q[i + 1, c:] = Q[i + 1, c:].copy(), Q[i, c:].copy()
        elif cross == "uniform":
            for i in range(0, pop - 1, 2):
                m = rng.random(n) < 0.5
                Q[i, m], Q[i + 1, m] = Q[i + 1, m].copy(), Q[i, m].copy()
        flip = rng.random((pop, n)) < pmut
        Q = np.where(flip, 1 - Q, Q)
        qfit = fn(Q)
        evals += pop
        # elitismo de uno: el mejor de antes sobrevive
        worst = int(qfit.argmin())
        elite = int(fit.argmax())
        Q[worst] = P[elite]
        qfit[worst] = fit[elite]
        P, fit = Q, qfit
        if int(fit.max()) > best:
            best = int(fit.max())
        if best >= optimum:
            return True, evals, best
    return False, evals, best


def hill_climber(fn, optimum, rng, budget=BUDGET, n=BITS):
    """Escalada con reinicios: sube por el primer bit que mejore, y cuando no
    queda ninguno vuelve a empezar desde otro sitio."""
    evals, best = 0, -1
    while evals < budget:
        x = rng.integers(0, 2, size=n)
        fx = int(fn(x))
        evals += 1
        best = max(best, fx)
        if best >= optimum:
            return True, evals, best
        improved = True
        while improved and evals < budget:
            improved = False
            for i in rng.permutation(n):
                x[i] ^= 1
                fy = int(fn(x))
                evals += 1
                if fy > fx:
                    fx = fy
                    improved = True
                    best = max(best, fx)
                    if best >= optimum:
                        return True, evals, best
                else:
                    x[i] ^= 1
                if evals >= budget:
                    break
    return False, evals, best


def random_search(fn, optimum, rng, budget=BUDGET, n=BITS):
    evals, best = 0, -1
    while evals < budget:
        chunk = min(2000, budget - evals)
        X = rng.integers(0, 2, size=(chunk, n))
        f = fn(X)
        evals += chunk
        if int(f.max()) > best:
            best = int(f.max())
        if best >= optimum:
            return True, evals, best
    return False, evals, best


ARMS = {
    "one_point": lambda fn, opt, rng: ga(fn, opt, rng, cross="one_point"),
    "uniform": lambda fn, opt, rng: ga(fn, opt, rng, cross="uniform"),
    "mutation_only": lambda fn, opt, rng: ga(fn, opt, rng, cross="none"),
    "hill_climber": lambda fn, opt, rng: hill_climber(fn, opt, rng),
    "random": lambda fn, opt, rng: random_search(fn, opt, rng),
}
ARM_LABELS = {
    "one_point": "one point crossover",
    "uniform": "uniform crossover",
    "mutation_only": "mutation only",
    "hill_climber": "hill climber, restarts",
    "random": "random guessing",
}


def stage_arms():
    out = {}
    for land, (fn, n, label) in LANDSCAPES.items():
        opt = BITS if land in ("onemax", "royal") else BLOCKS * BLOCK
        rows = []
        for arm, run in ARMS.items():
            wins, evals, bests, per_seed = 0, [], [], []
            for s in range(SEEDS):
                rng = np.random.default_rng(SEED * 1000 + s)
                ok, ev, best = run(fn, opt, rng)
                wins += int(ok)
                bests.append(best)
                per_seed.append(bool(ok))
                if ok:
                    evals.append(ev)
            rows.append({
                "arm": arm, "label": ARM_LABELS[arm],
                "success": wins, "of": SEEDS, "per_seed": per_seed,
                "median_evals": int(np.median(evals)) if evals else None,
                "mean_best": r(float(np.mean(bests)), 3),
                "best_possible": opt,
            })
        out[land] = {"label": label, "optimum": opt, "rows": rows}
    return out


def stage_takeover():
    """Presión de selección: cuánto tarda el mejor en llenar la población.

    Sin mutación y sin cruce, así que lo único que actúa es la selección. Es la
    cantidad que la teoría predice y casi nadie mide, y tiene dos formas de no
    terminar que hay que contar por separado en vez de promediar: que el mejor
    se pierda (con torneo de uno pasa siempre) y que el reloj se acabe.
    """
    rows = []
    pop, n, cap, runs = 100, BITS, 300, 20
    for tour in (1, 2, 3, 4, 6, 8):
        gens, lost, censored, curve = [], 0, 0, None
        for s in range(runs):
            rng = np.random.default_rng(SEED * 77 + s)
            P = rng.integers(0, 2, size=(pop, n))
            fit = f_onemax(P)
            top = int(fit.max())
            share = [float((fit == top).mean())]
            g = 0
            while share[-1] < 1.0 and share[-1] > 0.0 and g < cap:
                idx = rng.integers(0, pop, size=(pop, tour))
                winners = idx[np.arange(pop), fit[idx].argmax(axis=1)]
                P = P[winners]
                fit = fit[winners]
                share.append(float((fit == top).mean()))
                g += 1
            if share[-1] >= 1.0:
                gens.append(g)
            elif share[-1] <= 0.0:
                lost += 1
            else:
                censored += 1
            if s == 0:
                curve = share[:40]
        rows.append({
            "tournament": tour,
            "took_over": len(gens),
            "of": runs,
            "median": int(np.median(gens)) if gens else None,
            "spread": r(float(np.std(gens)), 2) if gens else None,
            "best_lost": lost,
            "ran_out": censored,
            "curve": r(curve, 4),
            # la predicción de libro: el tiempo de dominación crece como
            # log(pop) / log(torneo), que con torneo de uno no está definida
            "predicted": (r(np.log(pop) / np.log(tour), 2) if tour > 1 else None),
        })
    return {"rows": rows, "pop": pop, "cap": cap, "runs": runs,
            "note": "no mutation and no crossover, so selection is the only force"}


def stage_population():
    """El tamaño de población, a presupuesto de evaluaciones FIJO.

    Es el compromiso que de verdad hay que elegir, y sale medido en vez de
    citado: población pequeña son muchas generaciones con poca diversidad,
    población grande son pocas generaciones.
    """
    rows = []
    fn, opt = f_trap, BLOCKS * BLOCK
    for pop in (25, 50, 100, 200, 400, 800, 1600):
        wins, evals = 0, []
        for s in range(SEEDS):
            rng = np.random.default_rng(SEED * 31 + s)
            ok, ev, _ = ga(fn, opt, rng, pop=pop, cross="one_point")
            wins += int(ok)
            if ok:
                evals.append(ev)
        rows.append({"pop": pop, "success": wins, "of": SEEDS,
                     "median_evals": int(np.median(evals)) if evals else None})
    best = max(rows, key=lambda d: (d["success"], -(d["median_evals"] or 1e9)))
    return {"rows": rows, "best_pop": best["pop"], "budget": BUDGET,
            "landscape": "trap_tight"}


# ---------------------------------------------------------------------------
# El viajante: el sitio donde este método se compara con algo que no es él
# ---------------------------------------------------------------------------
def tour_length(order, D):
    return float(D[order, np.roll(order, -1)].sum())


def two_opt(D, rng, budget, order=None):
    """2-opt con reinicios, contando cada longitud calculada como evaluación."""
    n = D.shape[0]
    best_order, best_len, evals = None, np.inf, 0
    while evals < budget:
        order = rng.permutation(n) if order is None else order
        length = tour_length(order, D)
        evals += 1
        improved = True
        while improved and evals < budget:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b = order[i], order[i + 1]
                    c, d = order[j], order[(j + 1) % n]
                    delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                    evals += 1
                    if delta < -1e-12:
                        order[i + 1:j + 1] = order[i + 1:j + 1][::-1]
                        length += delta
                        improved = True
                    if evals >= budget:
                        break
                if evals >= budget:
                    break
        if length < best_len:
            best_len, best_order = length, order.copy()
        order = None
    return best_order, best_len, evals


def order_crossover(p1, p2, rng):
    """El cruce clásico para permutaciones: un trozo del primero, el resto en
    el orden del segundo."""
    n = len(p1)
    a, b = sorted(rng.integers(0, n, size=2))
    child = -np.ones(n, dtype=int)
    child[a:b] = p1[a:b]
    fill = [c for c in p2 if c not in set(p1[a:b])]
    k = 0
    for i in range(n):
        if child[i] < 0:
            child[i] = fill[k]
            k += 1
    return child


def ga_tsp(D, rng, budget, pop=60, tour=3, pmut=0.25):
    n = D.shape[0]
    P = np.array([rng.permutation(n) for _ in range(pop)])
    fit = np.array([tour_length(p, D) for p in P])
    evals = pop
    trace = [(evals, float(fit.min()))]
    while evals < budget:
        idx = rng.integers(0, pop, size=(pop, tour))
        winners = idx[np.arange(pop), fit[idx].argmin(axis=1)]
        Q = []
        for i in range(pop):
            child = order_crossover(P[winners[i]], P[winners[(i + 1) % pop]], rng)
            if rng.random() < pmut:
                a, b = sorted(rng.integers(0, n, size=2))
                child[a:b] = child[a:b][::-1]
            Q.append(child)
        Q = np.array(Q)
        qfit = np.array([tour_length(q, D) for q in Q])
        evals += pop
        Q[int(qfit.argmax())] = P[int(fit.argmin())]
        qfit[int(qfit.argmax())] = fit.min()
        P, fit = Q, qfit
        trace.append((evals, float(fit.min())))
    return P[int(fit.argmin())], float(fit.min()), trace


def stage_tsp():
    rng0 = np.random.default_rng(SEED + 9)
    n = 40
    pts = rng0.uniform(0, 1, size=(n, 2))
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    budget = 200000
    ga_lens, opt_lens, rnd_lens = [], [], []
    ga_tourbest, opt_tourbest = None, None
    trace_ga = None
    for s in range(12):
        rng = np.random.default_rng(SEED * 5 + s)
        order, length, trace = ga_tsp(D, rng, budget)
        ga_lens.append(length)
        if length <= min(ga_lens):
            ga_tourbest = order.tolist()
            trace_ga = trace
        rng = np.random.default_rng(SEED * 5 + s)
        order2, len2, _ = two_opt(D, rng, budget)
        opt_lens.append(len2)
        if len2 <= min(opt_lens):
            opt_tourbest = order2.tolist()
        rng = np.random.default_rng(SEED * 5 + s)
        bestr = min(tour_length(rng.permutation(n), D) for _ in range(2000))
        rnd_lens.append(bestr)
    return {
        "n": n,
        "points": r(pts, 4),
        "budget": budget,
        "seeds": 12,
        "ga": {"mean": r(float(np.mean(ga_lens)), 4), "best": r(float(min(ga_lens)), 4),
               "spread": r(float(np.std(ga_lens)), 4), "tour": ga_tourbest},
        "two_opt": {"mean": r(float(np.mean(opt_lens)), 4), "best": r(float(min(opt_lens)), 4),
                    "spread": r(float(np.std(opt_lens)), 4), "tour": opt_tourbest},
        "random": {"mean": r(float(np.mean(rnd_lens)), 4), "best": r(float(min(rnd_lens)), 4)},
        "trace": [[int(e), r(v, 4)] for e, v in trace_ga[::20]],
        "gap": r(float(np.mean(ga_lens)) / float(np.mean(opt_lens)) - 1, 4),
    }


# ---------------------------------------------------------------------------
# La tirada de referencia, con el azar escrito a mano para que el navegador la
# repita bit a bit. Todo son enteros: la comparación no necesita tolerancia.
# ---------------------------------------------------------------------------
class LCG:
    """s = (s*1664525 + 1013904223) mod 2^32, escrito igual en js/gakit.js."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next_u32(self):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s

    def random(self):
        return self.next_u32() / 4294967296.0

    def randint(self, n):
        return self.next_u32() % n


REF = {"pop": 100, "bits": BITS, "tour": 3, "pmut": 1.0 / BITS, "gens": 100,
       "cross": "one_point", "seed": 12345}


def ref_run(cfg):
    """La misma función que corre el navegador, escrita aquí una vez.

    El orden en que se piden los números al generador es parte del contrato: si
    los dos lados no lo respetan, las dos tiradas se separan y ninguna de las
    dos está mal.
    """
    rng = LCG(cfg["seed"])
    pop, n = cfg["pop"], cfg["bits"]
    P = [[rng.randint(2) for _ in range(n)] for _ in range(pop)]
    fit = [int(f_trap(np.array(p))) for p in P]
    history = [{"gen": 0, "best": max(fit), "mean": r(sum(fit) / pop, 4),
                "unique": len({tuple(p) for p in P})}]
    for _ in range(cfg["gens"]):
        winners = []
        for _ in range(pop):
            bi, bf = None, -1
            for _ in range(cfg["tour"]):
                k = rng.randint(pop)
                if fit[k] > bf:
                    bf, bi = fit[k], k
            winners.append(bi)
        Q = [P[w][:] for w in winners]
        if cfg["cross"] == "one_point":
            for i in range(0, pop - 1, 2):
                c = 1 + rng.randint(n - 1)
                Q[i][c:], Q[i + 1][c:] = Q[i + 1][c:], Q[i][c:]
        elif cfg["cross"] == "uniform":
            for i in range(0, pop - 1, 2):
                for j in range(n):
                    if rng.random() < 0.5:
                        Q[i][j], Q[i + 1][j] = Q[i + 1][j], Q[i][j]
        for i in range(pop):
            for j in range(n):
                if rng.random() < cfg["pmut"]:
                    Q[i][j] ^= 1
        qfit = [int(f_trap(np.array(q))) for q in Q]
        w = qfit.index(min(qfit))
        e = fit.index(max(fit))
        Q[w], qfit[w] = P[e][:], fit[e]
        P, fit = Q, qfit
        history.append({"gen": len(history), "best": max(fit),
                        "mean": r(sum(fit) / pop, 4),
                        "unique": len({tuple(p) for p in P})})
    return {"config": cfg, "history": history,
            "final_best": max(fit),
            "final_population": P[:12],
            "optimum": BLOCKS * BLOCK}


def stage_reference():
    return ref_run(REF)


def stage_landscape_probe():
    """Un puñado de cromosomas con su valor en los cuatro paisajes, para que el
    navegador pueda comprobar que evalúa lo mismo."""
    rng = np.random.default_rng(SEED + 3)
    X = rng.integers(0, 2, size=(24, BITS))
    X[0] = 1
    X[1] = 0
    return {
        "bits": BITS, "block": BLOCK, "blocks": BLOCKS,
        "shuffle": [int(i) for i in SHUFFLE],
        "genomes": [[int(b) for b in row] for row in X],
        "values": {k: [int(v) for v in fn(X)] for k, (fn, _, _) in LANDSCAPES.items()},
        "optima": {k: (BITS if k in ("onemax", "royal") else BLOCKS * BLOCK)
                   for k in LANDSCAPES},
        "deceptive_attractor": int(f_trap(np.zeros(BITS, dtype=int))),
    }


def main():
    t0 = time.time()
    print("1/6 los paisajes")
    probe = cached("probe-v2", stage_landscape_probe)
    print("2/6 cinco buscadores en cuatro paisajes")
    arms = cached("arms-v3", stage_arms)
    print("3/6 presión de selección")
    take = cached("takeover-v3", stage_takeover)
    print("4/6 tamaño de población")
    popu = cached("population-v2", stage_population)
    print("5/6 el viajante")
    tsp = cached("tsp", stage_tsp)
    print("6/6 la tirada de referencia")
    ref = cached("reference-v2", stage_reference)

    data = {
        "meta": {
            "seed": SEED, "seeds": SEEDS, "budget": BUDGET, "bits": BITS,
            "block": BLOCK, "blocks": BLOCKS,
            "pop": POP,
            "note": ("counts of evaluations, not seconds: the same numbers on "
                     "any machine"),
        },
        "probe": probe,
        "arms": arms,
        "takeover": take,
        "population": popu,
        "tsp": tsp,
        "reference": ref,
    }
    f = OUT / "ga.json"
    f.write_text(json.dumps(data, indent=1))
    print(f"escrito {f} ({f.stat().st_size / 1024:.1f} kB) en {time.time() - t0:.1f}s")

    print("\ncomprobaciones:")
    for land, d in arms.items():
        line = ", ".join(f"{row['arm']} {row['success']}/{row['of']}" for row in d["rows"])
        print(f"  {land}: {line}")
    print(f"  viajante: GA {tsp['ga']['mean']} contra 2-opt {tsp['two_opt']['mean']} "
          f"(hueco {tsp['gap']})")
    print(f"  referencia: mejor {ref['final_best']} de {ref['optimum']} "
          f"en {len(ref['history']) - 1} generaciones")


if __name__ == "__main__":
    main()
