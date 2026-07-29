"""Todo lo que cita el artículo 5.1a (programación lineal y MILP), medido.

La pregunta del artículo es dónde está el óptimo y cuánto cuesta encontrarlo, y
las dos tienen respuesta exacta, así que aquí no hay nada que estimar: se
enumera, se resuelve dos veces y se comparan las dos respuestas.

 1. El politopo del problema de producción, con TODAS sus esquinas enumeradas a
    fuerza bruta (cada par de restricciones, resuelto y comprobado contra el
    resto) y el óptimo de HiGHS encima. La afirmación "el óptimo está en una
    esquina" se comprueba sobre mil instancias al azar, no se cita.
 2. El dial del objetivo: 720 direcciones, y qué esquina gana en cada una. Los
    puntos de corte del dial se predicen desde las normales de las restricciones
    y se comparan con los medidos.
 3. El símplex, escrito aquí, con su camino sobre ese politopo; y el cubo de
    Klee y Minty, donde la regla de Dantzig visita 2^n - 1 esquinas. Eso se mide
    en pivotes, que es una cifra de la máquina de nadie.
 4. Los precios sombra: el valor dual que devuelve el solver contra volver a
    resolver con el lado derecho movido, y hasta dónde vale esa predicción,
    buscando el punto de ruptura por bisección.
 5. Los enteros: relajación contra óptimo entero sobre 300 mochilas, ramificación
    y acotación contada en nodos (con la cota y sin ella), la matriz totalmente
    unimodular del problema de asignación cuya relajación sale entera sola, y el
    ciclo impar donde la relajación vale exactamente la mitad de más.

Tarda menos de dos minutos en cuatro núcleos. Las etapas se cachean en
~/.atlas_vision_data/lp_cache/ y ninguna cifra publicada es un reloj de pared.
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "linear-programming" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "lp_cache"
CACHE.mkdir(parents=True, exist_ok=True)

SEED = 58
TOL = 1e-9


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
# El problema de producción que recorre el artículo entero.
#
# Dos productos, cuatro recursos. Los números están elegidos para que el óptimo
# NO caiga en un eje ni en una esquina entera, que es lo que hace falta para que
# la segunda mitad del artículo (los enteros) tenga algo que contar.
# ---------------------------------------------------------------------------
PROFIT = [5.0, 4.0]
ROWS = [
    ([6.0, 4.0], 24.0, "lathe hours"),
    ([1.0, 2.0], 6.0, "press hours"),
    ([-1.0, 1.0], 1.0, "mix rule"),
    ([0.0, 1.0], 2.0, "steel, tonnes"),
]
A = np.array([row[0] for row in ROWS])
B = np.array([row[1] for row in ROWS])
C = np.array(PROFIT)


def all_constraints():
    """Las cuatro del problema más las dos de no negatividad, como -x <= 0."""
    A2 = np.vstack([A, [[-1.0, 0.0], [0.0, -1.0]]])
    b2 = np.concatenate([B, [0.0, 0.0]])
    names = [row[2] for row in ROWS] + ["x >= 0", "y >= 0"]
    return A2, b2, names


def enumerate_vertices(A2, b2, tol=1e-9):
    """Cada par de restricciones, resuelto, y comprobado contra todas.

    Devuelve (vertices, pares, singulares, factibles). Un vértice es un punto
    donde n restricciones se cortan y ninguna se viola: eso es la definición, y
    enumerarla es lo que convierte "el óptimo está en una esquina" en algo que
    se puede comprobar en vez de citar.
    """
    verts, pairs, singular = [], 0, 0
    for i, j in combinations(range(len(b2)), 2):
        pairs += 1
        M = A2[[i, j]]
        if abs(np.linalg.det(M)) < 1e-12:
            singular += 1
            continue
        p = np.linalg.solve(M, b2[[i, j]])
        if np.all(A2 @ p <= b2 + tol):
            verts.append((p, (i, j)))
    uniq = []
    for p, act in verts:
        for q, _ in uniq:
            if np.allclose(p, q, atol=1e-9):
                break
        else:
            uniq.append((p, act))
    return uniq, pairs, singular, len(verts)


def count_vertices_nd(A2, b2, n, tol=1e-9):
    """Lo mismo que arriba en n dimensiones: cada subconjunto de n filas.

    En dos dimensiones son pares y se puede dibujar; en cuatro son C(m, 4) y ya
    no, pero la cuenta es la misma y es la que hay que poner al lado del número
    de pivotes para que "el símplex no las visita todas" signifique algo.
    """
    seen = []
    for idx in combinations(range(len(b2)), n):
        M = A2[list(idx)]
        if abs(np.linalg.det(M)) < 1e-10:
            continue
        p = np.linalg.solve(M, b2[list(idx)])
        if np.all(A2 @ p <= b2 + tol):
            for q in seen:
                if np.allclose(p, q, atol=1e-9):
                    break
            else:
                seen.append(p)
    return len(seen)


def solve_lp(c, A2, b2):
    res = linprog(-np.asarray(c), A_ub=A2, b_ub=b2, bounds=(None, None),
                  method="highs")
    return res


# ---------------------------------------------------------------------------
# Símplex propio: el mismo que corre el navegador, escrito una vez aquí para
# poder afirmar que las dos implementaciones coinciden.
# ---------------------------------------------------------------------------
def simplex(c, A_ub, b_ub, rule="dantzig", max_iter=100000):
    """max c'x, A x <= b, x >= 0, con b >= 0 (la base de holguras es factible).

    Devuelve (x, valor, pivotes, camino). El camino es la lista de vértices
    visitados, que es lo que dibuja el artículo.
    """
    m, n = A_ub.shape
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = A_ub
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = b_ub
    T[-1, :n] = -np.asarray(c, dtype=float)
    basis = list(range(n, n + m))
    path, pivots = [], 0

    def point():
        x = np.zeros(n + m)
        for i, bi in enumerate(basis):
            x[bi] = T[i, -1]
        return x[:n].copy()

    path.append(point())
    while True:
        red = T[-1, :n + m]
        cands = np.where(red < -1e-9)[0]
        if cands.size == 0:
            break
        if rule == "bland":
            col = int(cands[0])
        else:
            col = int(np.argmin(red))
        ratios = np.where(T[:m, col] > 1e-9, T[:m, -1] / np.where(T[:m, col] > 1e-9, T[:m, col], 1),
                          np.inf)
        if not np.isfinite(ratios).any():
            return None, np.inf, pivots, path
        row = int(np.argmin(ratios))
        T[row] /= T[row, col]
        for k in range(m + 1):
            if k != row and abs(T[k, col]) > 1e-12:
                T[k] -= T[k, col] * T[row]
        basis[row] = col
        pivots += 1
        path.append(point())
        if pivots > max_iter:
            break
    return point(), float(T[-1, -1]), pivots, path


def klee_minty(n):
    """El cubo deformado donde la regla del mejor coeficiente visita 2^n - 1."""
    c = np.array([10.0 ** (n - j) for j in range(1, n + 1)])
    A_ub = np.zeros((n, n))
    b_ub = np.zeros(n)
    for i in range(1, n + 1):
        for j in range(1, i):
            A_ub[i - 1, j - 1] = 2 * 10.0 ** (i - j)
        A_ub[i - 1, i - 1] = 1.0
        b_ub[i - 1] = 100.0 ** (i - 1)
    return c, A_ub, b_ub


# ---------------------------------------------------------------------------
# Etapa 1: el politopo, sus esquinas, y el óptimo
# ---------------------------------------------------------------------------
def stage_polytope():
    A2, b2, names = all_constraints()
    verts, pairs, singular, feasible_hits = enumerate_vertices(A2, b2)
    vals = [float(C @ p) for p, _ in verts]
    order = np.argsort([np.arctan2(p[1] - np.mean([q[1] for q, _ in verts]),
                                   p[0] - np.mean([q[0] for q, _ in verts]))
                        for p, _ in verts])
    ring = [{"x": r(verts[i][0][0]), "y": r(verts[i][0][1]),
             "value": r(vals[i]), "active": list(verts[i][1])} for i in order]
    res = solve_lp(C, A2, b2)
    best = int(np.argmax([v["value"] for v in ring]))
    hit = bool(abs(-res.fun - ring[best]["value"]) < 1e-7
               and np.allclose([ring[best]["x"], ring[best]["y"]], res.x, atol=1e-7))
    return {
        "rows": [{"a": r(row[0]), "b": r(row[1]), "name": row[2]} for row in ROWS],
        "profit": r(PROFIT),
        "vertices": ring,
        "pairs": pairs,
        "singular": singular,
        "feasible_hits": feasible_hits,
        "n_vertices": len(ring),
        "best_index": best,
        "solver_x": r(res.x, 6),
        "solver_value": r(-res.fun, 6),
        "solver_iterations": int(res.nit),
        "solver_matches_corner": hit,
    }


# ---------------------------------------------------------------------------
# Etapa 2: ¿siempre en una esquina? Mil politopos al azar.
# ---------------------------------------------------------------------------
def stage_corner_claim():
    rng = np.random.default_rng(SEED)
    n_inst, worst, ties, unbounded = 1000, 0.0, 0, 0
    faces = []
    for _ in range(n_inst):
        m = int(rng.integers(3, 8))
        Ai = rng.uniform(0.2, 3.0, size=(m, 2))
        bi = rng.uniform(1.0, 8.0, size=m)
        ci = rng.uniform(-1.0, 3.0, size=2)
        A2 = np.vstack([Ai, [[-1.0, 0.0], [0.0, -1.0]]])
        b2 = np.concatenate([bi, [0.0, 0.0]])
        verts, *_ = enumerate_vertices(A2, b2)
        res = solve_lp(ci, A2, b2)
        if not res.success:
            unbounded += 1
            continue
        best = max(float(ci @ p) for p, _ in verts)
        worst = max(worst, abs(best - (-res.fun)))
        near = [p for p, _ in verts if abs(float(ci @ p) - best) < 1e-9]
        if len(near) > 1:
            ties += 1
        faces.append(len(verts))
    return {
        "instances": n_inst,
        "unbounded": unbounded,
        "worst_gap": r(worst, 12),
        "ties": ties,
        "mean_vertices": r(float(np.mean(faces)), 2),
        "max_vertices": int(max(faces)),
    }


# ---------------------------------------------------------------------------
# Etapa 3: el dial del objetivo. 720 direcciones, y qué esquina gana.
# ---------------------------------------------------------------------------
def stage_dial():
    A2, b2, _ = all_constraints()
    verts, *_ = enumerate_vertices(A2, b2)
    P = np.array([p for p, _ in verts])
    steps = 720

    def winner(deg):
        th = np.radians(deg)
        return int(np.argmax(P @ np.array([np.cos(th), np.sin(th)])))

    owner, switches = [], []
    prev = None
    for k in range(steps):
        deg = 360 * k / steps
        j = winner(deg)
        owner.append(j)
        if prev is not None and j != prev:
            # La rejilla solo dice entre qué medio grado cambia el dueño; el
            # ángulo exacto se saca por bisección, que es lo que permite
            # comparar contra la normal predicha y no contra la resolución.
            lo, hi = deg - 360 / steps, deg
            for _ in range(60):
                mid = (lo + hi) / 2
                if winner(mid) == prev:
                    lo = mid
                else:
                    hi = mid
            switches.append(r((lo + hi) / 2, 9))
        prev = j
    # Los cortes predichos: una dirección cambia de dueño cuando es
    # perpendicular a una arista del politopo, o sea paralela a su normal.
    hull_order = np.argsort(np.arctan2(P[:, 1] - P[:, 1].mean(), P[:, 0] - P[:, 0].mean()))
    predicted = []
    ring = P[hull_order]
    for i in range(len(ring)):
        e = ring[(i + 1) % len(ring)] - ring[i]
        nrm = np.array([e[1], -e[0]])
        predicted.append(float(np.degrees(np.arctan2(nrm[1], nrm[0])) % 360))
    predicted = sorted(predicted)
    err = []
    for s in switches:
        err.append(min(abs(s - p) for p in predicted))
    # Justo en un corte, dos esquinas empatan: es el único sitio donde un
    # problema lineal tiene más de una respuesta, y se puede enseñar.
    th = np.radians(switches[0])
    scores = P @ np.array([np.cos(th), np.sin(th)])
    top = np.sort(scores)[::-1]
    return {
        "steps": steps,
        "distinct_owners": len(set(owner)),
        "owner": [int(o) for o in owner[::4]],          # una de cada cuatro, para dibujar
        "owner_stride": 4,
        "switch_angles": switches,
        "predicted_angles": r(predicted, 6),
        "worst_switch_error_deg": r(max(err) if err else 0.0, 9),
        "resolution_deg": r(360 / steps, 3),
        "tie_angle": r(switches[0], 6),
        "tie_gap": r(abs(top[0] - top[1]), 12),
    }


# ---------------------------------------------------------------------------
# Etapa 4: el símplex y el camino, más Klee y Minty
# ---------------------------------------------------------------------------
def stage_simplex():
    x, val, piv, path = simplex(C, A, B)
    res = solve_lp(C, *all_constraints()[:2])
    A2, b2, _ = all_constraints()
    verts, *_ = enumerate_vertices(A2, b2)

    # Cuatro dimensiones y diez restricciones: bastantes esquinas para que la
    # comparación signifique algo, y bastantes pocas para enumerarlas todas.
    rng = np.random.default_rng(SEED + 1)
    dims, m_rows, n_inst = 4, 10, 200
    piv_list, vert_list = [], []
    for _ in range(n_inst):
        Ai = rng.uniform(0.2, 3.0, size=(m_rows, dims))
        bi = rng.uniform(1.0, 8.0, size=m_rows)
        ci = rng.uniform(0.5, 3.0, size=dims)
        A2r = np.vstack([Ai, -np.eye(dims)])
        b2r = np.concatenate([bi, np.zeros(dims)])
        vert_list.append(count_vertices_nd(A2r, b2r, dims))
        _, _, p, _ = simplex(ci, Ai, bi)
        piv_list.append(p)

    km = []
    for n in range(2, 9):
        c, Ak, bk = klee_minty(n)
        _, vd, pd, _ = simplex(c, Ak, bk, rule="dantzig")
        _, vb, pb, _ = simplex(c, Ak, bk, rule="bland")
        # Con el presolve puesto, HiGHS resuelve este cubo sin pivotar ninguna
        # vez (informa nit = 0), que es un resultado sobre el presolve y no
        # sobre el símplex. Se mide con el presolve apagado, que es lo que hace
        # comparable la columna con las otras dos.
        rk = linprog(-c, A_ub=Ak, b_ub=bk, bounds=(0, None), method="highs",
                     options={"presolve": False})
        rk_pre = linprog(-c, A_ub=Ak, b_ub=bk, bounds=(0, None), method="highs")
        km.append({
            "n": n, "vertices": 2 ** n,
            "dantzig": pd, "bland": pb,
            "highs": int(rk.nit),
            "highs_presolve": int(rk_pre.nit),
            "predicted": 2 ** n - 1,
            "value": r(vd, 3),
            "highs_value": r(-rk.fun, 3),
            "agree": bool(abs(vd + rk.fun) < 1e-6 * max(1.0, abs(vd))),
        })
    return {
        "path": [{"x": r(p[0]), "y": r(p[1]), "value": r(float(C @ p))} for p in path],
        "pivots": piv,
        "value": r(val),
        "value_highs": r(-res.fun),
        "agrees_with_highs": bool(abs(val - (-res.fun)) < 1e-9),
        "n_vertices": len(verts),
        "random": {
            "instances": n_inst,
            "dims": dims,
            "constraints": m_rows,
            "mean_pivots": r(float(np.mean(piv_list)), 2),
            "max_pivots": int(max(piv_list)),
            "mean_vertices": r(float(np.mean(vert_list)), 2),
            "max_vertices": int(max(vert_list)),
            "ratio": r(float(np.mean(vert_list)) / float(np.mean(piv_list)), 2),
        },
        "klee_minty": km,
    }


# ---------------------------------------------------------------------------
# Etapa 5: precios sombra, y hasta dónde valen
# ---------------------------------------------------------------------------
def stage_duals():
    A2, b2, names = all_constraints()
    res = linprog(-C, A_ub=A, b_ub=B, bounds=(0, None), method="highs")
    marg = -np.asarray(res.ineqlin.marginals)     # >= 0 para un máximo
    base = -res.fun
    rows = []
    for i, (a, b, name) in enumerate(ROWS):
        bb = B.copy()
        h = 1e-6
        bb[i] = B[i] + h
        up = -linprog(-C, A_ub=A, b_ub=bb, bounds=(0, None), method="highs").fun
        bb[i] = B[i] - h
        dn = -linprog(-C, A_ub=A, b_ub=bb, bounds=(0, None), method="highs").fun
        fd = (up - dn) / (2 * h)
        # hasta dónde vale: el mayor delta cuyo valor sigue siendo lineal
        lo, hi = 0.0, 64.0
        for _ in range(60):
            mid = (lo + hi) / 2
            bb = B.copy()
            bb[i] = B[i] + mid
            v = -linprog(-C, A_ub=A, b_ub=bb, bounds=(0, None), method="highs").fun
            if abs(v - (base + marg[i] * mid)) < 1e-7:
                lo = mid
            else:
                hi = mid
        beyond = lo * 1.5 + 1.0
        bb = B.copy()
        bb[i] = B[i] + beyond
        v_far = -linprog(-C, A_ub=A, b_ub=bb, bounds=(0, None), method="highs").fun
        slack = float(B[i] - A[i] @ res.x)
        rows.append({
            "name": name,
            "b": r(b),
            "shadow": r(marg[i], 6),
            "finite_difference": r(fd, 6),
            "gap": r(abs(marg[i] - fd), 9),
            "slack": r(slack, 6),
            "valid_up_to": r(lo, 3),
            "predicted_at_beyond": r(base + marg[i] * beyond, 4),
            "actual_at_beyond": r(v_far, 4),
            "beyond": r(beyond, 3),
        })
    total = float(sum(row["shadow"] * b for row, b in zip(rows, B)))
    return {
        "rows": rows,
        "objective": r(base, 6),
        "dual_objective": r(total, 6),
        "duality_gap": r(abs(total - base), 9),
        "worst_shadow_gap": r(max(row["gap"] for row in rows), 9),
    }


# ---------------------------------------------------------------------------
# Etapa 6: los enteros
# ---------------------------------------------------------------------------
def knapsack_instance(rng, n):
    w = rng.integers(5, 60, size=n).astype(float)
    v = (w * rng.uniform(0.6, 1.6, size=n)).round(1)
    cap = float(np.floor(w.sum() * 0.5))
    return w, v, cap


def density_order(w, v):
    """Por densidad, y a igualdad por índice.

    `np.argsort` no es estable por defecto, así que dos artículos con la misma
    densidad salían en un orden que el navegador no podía reproducir: el mismo
    fallo de desempates que ya costó un artículo en este repo. Aquí el criterio
    se escribe entero y se escribe igual en `js/lpkit.js`.
    """
    return sorted(range(len(w)), key=lambda i: (-(v[i] / w[i]), i))


def knapsack_lp(w, v, cap):
    """La relajación tiene forma cerrada: ordenar por densidad y llenar."""
    idx = density_order(w, v)
    total, left, x = 0.0, cap, np.zeros(len(w))
    for i in idx:
        take = min(1.0, left / w[i])
        x[i] = take
        total += take * v[i]
        left -= take * w[i]
        if left <= 1e-12:
            break
    return total, x


def knapsack_exact(w, v, cap):
    """Programación dinámica sobre pesos enteros: la respuesta, sin discusión."""
    cap_i = int(cap)
    dp = np.zeros(cap_i + 1)
    for wi, vi in zip(w.astype(int), v):
        dp[wi:] = np.maximum(dp[wi:], dp[:cap_i + 1 - wi] + vi)
    return float(dp[cap_i])


def branch_and_bound(w, v, cap, use_bound=True, node_cap=4_000_000):
    """Ramificación y acotación sobre la mochila, contando nodos.

    Con `use_bound` la cota es la relajación lineal del subproblema, que es la
    misma forma cerrada de arriba. Sin ella solo se poda por capacidad, que es
    exactamente el experimento: cuánto compra la cota.
    """
    order = density_order(w, v)
    w, v = w[order], v[order]
    n = len(w)
    best = 0.0
    nodes = 0

    def bound(i, room, taken):
        total, left = taken, room
        while i < n and left > 1e-12:
            take = min(1.0, left / w[i])
            total += take * v[i]
            left -= take * w[i]
            i += 1
        return total

    stack = [(0, cap, 0.0)]
    while stack:
        i, room, taken = stack.pop()
        nodes += 1
        if nodes > node_cap:
            return best, nodes, False
        if taken > best:
            best = taken
        if i == n:
            continue
        if use_bound and bound(i, room, taken) <= best + 1e-9:
            continue
        if w[i] <= room:
            stack.append((i + 1, room - w[i], taken + v[i]))
        stack.append((i + 1, room, taken))
    return best, nodes, True


def stage_integers():
    rng = np.random.default_rng(SEED + 2)
    gaps, rounded_loss, rounded_optimal, n_inst = [], [], 0, 300
    for _ in range(n_inst):
        w, v, cap = knapsack_instance(rng, 30)
        lp, x = knapsack_lp(w, v, cap)
        exact = knapsack_exact(w, v, cap)
        gaps.append((lp - exact) / exact)
        down = float(v[(x > 1 - 1e-9)].sum())
        rounded_loss.append((exact - down) / exact)
        if abs(down - exact) < 1e-9:
            rounded_optimal += 1

    # nodos: con cota y sin ella, contra la enumeración completa
    nodes = []
    rng2 = np.random.default_rng(SEED + 3)
    for n in (10, 14, 18, 22, 26, 30):
        with_b, without_b, ok = [], [], True
        for _ in range(12):
            w, v, cap = knapsack_instance(rng2, n)
            _, nb, _ = branch_and_bound(w, v, cap, use_bound=True)
            _, nn, done = branch_and_bound(w, v, cap, use_bound=False,
                                           node_cap=3_000_000)
            with_b.append(nb)
            without_b.append(nn)
            ok = ok and done
        nodes.append({
            "n": n,
            "brute_force": 2 ** n,
            "with_bound": int(np.median(with_b)),
            "without_bound": int(np.median(without_b)) if ok else None,
            "saving": r(np.median(without_b) / np.median(with_b), 1) if ok else None,
            "instances": 12,
        })

    # totalmente unimodular: asignación
    worst_frac_assign, integral_assign, n_assign = 0.0, 0, 400
    rng3 = np.random.default_rng(SEED + 4)
    k = 6
    Aeq = np.zeros((2 * k, k * k))
    for i in range(k):
        for j in range(k):
            Aeq[i, i * k + j] = 1.0
            Aeq[k + j, i * k + j] = 1.0
    beq = np.ones(2 * k)
    for _ in range(n_assign):
        cost = rng3.uniform(1, 20, size=k * k)
        res = linprog(cost, A_eq=Aeq, b_eq=beq, bounds=(0, 1), method="highs")
        d = float(np.abs(res.x - np.round(res.x)).max())
        worst_frac_assign = max(worst_frac_assign, d)
        if d < 1e-9:
            integral_assign += 1

    # y la mochila, por contraste: cuántas relajaciones salen enteras
    integral_knap, n_knap = 0, 300
    rng4 = np.random.default_rng(SEED + 5)
    for _ in range(n_knap):
        w, v, cap = knapsack_instance(rng4, 30)
        _, x = knapsack_lp(w, v, cap)
        if np.abs(x - np.round(x)).max() < 1e-9:
            integral_knap += 1

    # el ciclo impar: la relajación vale un medio en cada vértice
    cycles = []
    for k5 in (3, 5, 7, 9):
        n = k5
        Ac = np.zeros((n, n))
        for i in range(n):
            Ac[i, i] = -1.0
            Ac[i, (i + 1) % n] = -1.0
        res = linprog(np.ones(n), A_ub=Ac, b_ub=-np.ones(n), bounds=(0, 1),
                      method="highs")
        ip = milp(np.ones(n), constraints=LinearConstraint(-Ac, 1, np.inf),
                  integrality=np.ones(n), bounds=Bounds(0, 1))
        cycles.append({
            "n": n, "lp": r(res.fun, 4), "ip": r(ip.fun, 4),
            "ratio": r(ip.fun / res.fun, 4),
            "all_half": bool(np.abs(res.x - 0.5).max() < 1e-9),
        })
    return {
        "knapsack": {
            "instances": n_inst, "n_items": 30,
            "mean_gap": r(float(np.mean(gaps)), 5),
            "max_gap": r(float(np.max(gaps)), 5),
            "mean_rounded_loss": r(float(np.mean(rounded_loss)), 5),
            "max_rounded_loss": r(float(np.max(rounded_loss)), 5),
            "rounded_optimal": rounded_optimal,
            "integral_relaxations": integral_knap,
            "integral_of": n_knap,
        },
        "nodes": nodes,
        "assignment": {
            "instances": n_assign, "size": k,
            "integral": integral_assign,
            "worst_distance_to_integer": r(worst_frac_assign, 12),
        },
        "cycles": cycles,
    }


# ---------------------------------------------------------------------------
# Etapa 7: una mochila concreta para el widget, con su árbol
# ---------------------------------------------------------------------------
def stage_demo_knapsack():
    rng = np.random.default_rng(SEED + 6)
    n = 12
    w = rng.integers(4, 22, size=n).astype(float)
    v = (w * rng.uniform(0.7, 1.5, size=n)).round(0)
    cap = float(np.floor(w.sum() * 0.45))
    lp, x = knapsack_lp(w, v, cap)
    exact = knapsack_exact(w, v, cap)
    best, nodes, _ = branch_and_bound(w, v, cap, use_bound=True)
    _, nodes_nb, _ = branch_and_bound(w, v, cap, use_bound=False)
    order = density_order(w, v)
    ip = milp(-v, constraints=LinearConstraint(w.reshape(1, -1), -np.inf, cap),
              integrality=np.ones(n), bounds=Bounds(0, 1))
    return {
        "n": n,
        "weights": [int(t) for t in w],
        "values": [int(t) for t in v],
        "capacity": r(cap),
        "order": [int(t) for t in order],
        "lp_bound": r(lp, 3),
        "lp_x": r(x, 6),
        "optimum": r(exact, 3),
        "milp_optimum": r(-ip.fun, 3),
        "milp_nodes": int(ip.mip_node_count),
        "bb_nodes": nodes,
        "bb_nodes_no_bound": nodes_nb,
        "brute_force": 2 ** n,
        "gap": r((lp - exact) / exact, 5),
        "agrees": bool(abs(exact - best) < 1e-9 and abs(exact + ip.fun) < 1e-6),
    }


def main():
    t0 = time.time()
    print("1/7 politopo")
    poly = cached("polytope", stage_polytope)
    print("2/7 la afirmación de la esquina")
    corner = cached("corner", stage_corner_claim)
    print("3/7 el dial")
    dial = cached("dial-v2", stage_dial)
    print("4/7 símplex y Klee-Minty")
    spx = cached("simplex-v2", stage_simplex)
    print("5/7 precios sombra")
    duals = cached("duals", stage_duals)
    print("6/7 enteros")
    ints = cached("integers-v3", stage_integers)
    print("7/7 la mochila del widget")
    demo = cached("demo_knapsack-v2", stage_demo_knapsack)

    data = {
        "meta": {
            "seed": SEED,
            "solver": "HiGHS",
            "scipy": scipy.__version__,
            "note": ("every count on this page is a count, not a clock: pivots, "
                     "nodes and vertices are the same on any machine"),
        },
        "polytope": poly,
        "corner": corner,
        "dial": dial,
        "simplex": spx,
        "duals": duals,
        "integers": ints,
        "demo": demo,
    }
    f = OUT / "lp.json"
    f.write_text(json.dumps(data, indent=1))
    print(f"escrito {f} ({f.stat().st_size / 1024:.1f} kB) en {time.time() - t0:.1f}s")

    print("\ncomprobaciones:")
    print(f"  óptimo en una esquina: peor hueco {corner['worst_gap']} "
          f"sobre {corner['instances']} politopos")
    print(f"  símplex propio contra HiGHS: {spx['agrees_with_highs']}")
    print(f"  Klee-Minty: " + ", ".join(f"n={k['n']}:{k['dantzig']}/{k['predicted']}"
                                        for k in spx["klee_minty"]))
    print(f"  precios sombra, peor desacuerdo: {duals['worst_shadow_gap']}")
    print(f"  asignación entera sola: {ints['assignment']['integral']}"
          f"/{ints['assignment']['instances']}")


if __name__ == "__main__":
    main()
