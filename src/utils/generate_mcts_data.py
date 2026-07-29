"""Todo lo que cita el artículo 5.3d (MCTS y modelos aprendidos), medido.

Un juego pequeño **resuelto entero** por minimax con memoria, para poder
puntuar la búsqueda contra la respuesta en vez de contra otra búsqueda, y un
modelo aprendido de un mundo diminuto, para medir a qué velocidad se estropea
un plan hecho dentro de la cabeza.

 1. Conecta cuatro en un tablero de 4 por 5, resuelto: el valor exacto de cada
    posición alcanzable y qué jugadas son óptimas en cada una.
 2. La búsqueda por árbol contra esa respuesta: qué fracción de posiciones
    resuelve bien, barriendo el número de simulaciones.
 3. La constante de exploración, barrida, y lo que pasa en los dos extremos.
 4. Las tiradas al azar contra una tirada con una pizca de conocimiento.
 5. Contra minimax con la misma cuenta de nodos.
 6. Un modelo aprendido de un mundo pequeño, y el error de un plan hecho dentro
    de él, medido paso a paso contra lo que pasa de verdad.

Tarda unos seis minutos en cuatro núcleos. Etapas cacheadas en
~/.atlas_vision_data/mcts_cache/.
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "mcts" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "mcts_cache"
CACHE.mkdir(parents=True, exist_ok=True)
sys.setrecursionlimit(10000)

SEED = 66
ROWS, COLS, CONNECT = 4, 5, 4


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
# El juego. Un estado es una tupla de columnas, cada una con las fichas de abajo
# arriba: inmutable, hashable, y suficiente para memorizar el árbol entero.
# ---------------------------------------------------------------------------
EMPTY = tuple(() for _ in range(COLS))


def moves(state):
    return [c for c in range(COLS) if len(state[c]) < ROWS]


def play(state, col, who):
    lst = list(state)
    lst[col] = lst[col] + (who,)
    return tuple(lst)


def _lines():
    """Las 17 lineas de cuatro casillas que existen en un tablero de 4 por 5,
    en indices planos. Enumerarlas una vez y comprobarlas es bastante mas rapido
    que barrer el tablero entero en cuatro direcciones desde cada casilla, y esta
    funcion se llama millones de veces."""
    out = []
    for r_ in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = []
                rr, cc = r_, c
                for _ in range(CONNECT):
                    if not (0 <= rr < ROWS and 0 <= cc < COLS):
                        break
                    cells.append(rr * COLS + cc)
                    rr += dr
                    cc += dc
                if len(cells) == CONNECT:
                    out.append(tuple(cells))
    return out


LINES = _lines()


def winner(state):
    """Quién ha hecho línea, si alguien."""
    g = [0] * (ROWS * COLS)
    for c, colvals in enumerate(state):
        base = c
        for r_, v in enumerate(colvals):
            g[r_ * COLS + base] = v
    for a, b, c2, d in LINES:
        v = g[a]
        if v and v == g[b] and v == g[c2] and v == g[d]:
            return v
    return 0


def terminal(state):
    w = winner(state)
    if w:
        return w
    if not moves(state):
        return 0.5                      # tablas
    return None


SOLVED = {}


def solve(state, who):
    """Negamax con memoria. Devuelve el valor para el que mueve: 1 gana, -1
    pierde, 0 tablas, y el número de jugadas hasta el final se ignora a
    propósito, porque la pregunta del artículo es qué jugada es óptima."""
    key = (state, who)
    if key in SOLVED:
        return SOLVED[key]
    t = terminal(state)
    if t is not None:
        val = 0.0 if t == 0.5 else (1.0 if t == who else -1.0)
        SOLVED[key] = (val, [])
        return SOLVED[key]
    best, bestmoves = -2.0, []
    for c in moves(state):
        v, _ = solve(play(state, c, who), 3 - who)
        v = -v
        if v > best + 1e-9:
            best, bestmoves = v, [c]
        elif abs(v - best) < 1e-9:
            bestmoves.append(c)
    SOLVED[key] = (best, bestmoves)
    return SOLVED[key]


# ---------------------------------------------------------------------------
# La búsqueda por árbol
# ---------------------------------------------------------------------------
def rollout_random(state, who, rng):
    while True:
        t = terminal(state)
        if t is not None:
            return 0.0 if t == 0.5 else (1.0 if t == who else -1.0)
        ms = moves(state)
        state = play(state, ms[rng.randrange(len(ms))], who)
        who = 3 - who


def rollout_smart(state, who, rng):
    """Una pizca de conocimiento: si hay una jugada que gana ya, se juega; si el
    rival la tiene, se tapa. Nada más."""
    while True:
        t = terminal(state)
        if t is not None:
            return 0.0 if t == 0.5 else (1.0 if t == who else -1.0)
        ms = moves(state)
        pick = None
        for c in ms:
            if winner(play(state, c, who)) == who:
                pick = c
                break
        if pick is None:
            for c in ms:
                if winner(play(state, c, 3 - who)) == 3 - who:
                    pick = c
                    break
        if pick is None:
            pick = ms[rng.randrange(len(ms))]
        state = play(state, pick, who)
        who = 3 - who


def mcts(state, who, sims, rng, c_uct=1.4, smart=False):
    """Cuatro pasos por simulación: bajar por el árbol con la fórmula, añadir un
    hijo, jugar al azar hasta el final, y subir el resultado.

    Un convenio y nada más: `W[nodo]` guarda el valor **desde el punto de vista
    de quien mueve en ese nodo**, así que al subir por el camino el signo se da
    la vuelta en cada paso y al leer un hijo desde su padre también. Escrito de
    cualquier otra forma la búsqueda juega contra sí misma sin avisar.
    """
    root = (state, who)
    N = {root: 0}
    W = {root: 0.0}
    children = {}
    rollout = rollout_smart if smart else rollout_random
    for _ in range(sims):
        path = [root]
        node = root
        while True:
            s, w = node
            t = terminal(s)
            if t is not None:
                value = 0.0 if t == 0.5 else (1.0 if t == w else -1.0)
                break
            if node not in children:
                children[node] = [(play(s, cc, w), 3 - w) for cc in moves(s)]
                for ch in children[node]:
                    N.setdefault(ch, 0)
                    W.setdefault(ch, 0.0)
            unseen = [ch for ch in children[node] if N[ch] == 0]
            if unseen:
                node = unseen[rng.randrange(len(unseen))]
                path.append(node)
                value = rollout(node[0], node[1], rng)
                break
            total = sum(N[ch] for ch in children[node])
            best, bestch = -1e18, None
            for ch in children[node]:
                # el valor guardado es para el que mueve EN el hijo, así que se
                # invierte para leerlo desde el padre
                q = -W[ch] / N[ch]
                u = q + c_uct * math.sqrt(math.log(total + 1) / N[ch])
                if u > best:
                    best, bestch = u, ch
            node = bestch
            path.append(node)
        # el valor está en el punto de vista del último nodo del camino
        for j in range(len(path)):
            N[path[j]] += 1
            W[path[j]] += value if (len(path) - 1 - j) % 2 == 0 else -value
    kids = children.get(root, [])
    counts = {}
    for cc, ch in zip(moves(state), kids):
        counts[cc] = N[ch]
    if not counts:
        return None, {}
    pick = max(counts, key=lambda k: counts[k])
    return pick, counts


def minimax_limited(state, who, depth, nodes):
    """Minimax cortado a una profundidad, con un contador de nodos, para poder
    compararlo con la búsqueda por árbol al mismo presupuesto."""
    t = terminal(state)
    nodes[0] += 1
    if t is not None:
        return (0.0 if t == 0.5 else (1.0 if t == who else -1.0)), None
    if depth == 0:
        return 0.0, None
    best, bestmove = -2.0, None
    for c in moves(state):
        v, _ = minimax_limited(play(state, c, who), 3 - who, depth - 1, nodes)
        v = -v
        if v > best:
            best, bestmove = v, c
    return best, bestmove


def sample_positions(n, rng, min_plies=4, max_plies=10):
    """Posiciones alcanzables y no terminales, para preguntarles a las dos
    búsquedas. Se sortean jugando al azar y se filtran las que ya están
    decididas."""
    out = []
    seen = set()
    while len(out) < n:
        state, who = EMPTY, 1
        plies = rng.randrange(min_plies, max_plies + 1)
        ok = True
        for _ in range(plies):
            if terminal(state) is not None:
                ok = False
                break
            ms = moves(state)
            state = play(state, ms[rng.randrange(len(ms))], who)
            who = 3 - who
        if not ok or terminal(state) is not None:
            continue
        key = (state, who)
        if key in seen:
            continue
        seen.add(key)
        out.append((state, who))
    return out


def stage_solve():
    t0 = time.time()
    val, best = solve(EMPTY, 1)
    return {"value": r(val, 3), "best_moves": best, "positions": len(SOLVED),
            "rows": ROWS, "cols": COLS, "connect": CONNECT,
            "seconds_note": "the count of positions is the machine independent number",
            "solved_in": r(time.time() - t0, 1)}


def stage_accuracy():
    rng = random.Random(SEED)
    positions = sample_positions(150, rng)
    truth = {}
    for state, who in positions:
        truth[(state, who)] = solve(state, who)
    rows = []
    for sims in (8, 16, 32, 64, 128, 256, 512, 1024, 2048):
        for smart in (False, True):
            good = 0
            rng2 = random.Random(SEED + 1)
            for state, who in positions:
                pick, _ = mcts(state, who, sims, rng2, smart=smart)
                if pick in truth[(state, who)][1]:
                    good += 1
            rows.append({"sims": sims, "smart": smart, "correct": good,
                         "of": len(positions),
                         "share": r(good / len(positions), 4)})
    return {"rows": rows, "positions": len(positions)}


def stage_constant():
    rng = random.Random(SEED + 2)
    positions = sample_positions(120, rng)
    truth = {p: solve(*p) for p in positions}
    rows = []
    for c_uct in (0.0, 0.25, 0.5, 1.0, 1.4, 2.0, 4.0, 8.0):
        good = 0
        rng2 = random.Random(SEED + 3)
        for state, who in positions:
            pick, _ = mcts(state, who, 256, rng2, c_uct=c_uct)
            if pick in truth[(state, who)][1]:
                good += 1
        rows.append({"c": c_uct, "correct": good, "of": len(positions),
                     "share": r(good / len(positions), 4)})
    best = max(rows, key=lambda d: d["correct"])
    return {"rows": rows, "best_c": best["c"], "sims": 256,
            "textbook_c": 1.4, "positions": len(positions)}


def stage_noise():
    """El suelo de ruido, que es lo que decide si las otras comparaciones dicen
    algo. La misma configuración repetida con semillas de búsqueda distintas se
    mueve sola; cualquier diferencia entre dos brazos por debajo de eso no es una
    diferencia. Se usan **las mismas 120 posiciones** que el barrido de la
    constante, para que su suelo sirva también allí.
    """
    rng = random.Random(SEED + 2)
    positions = sample_positions(120, rng)
    truth = {p: solve(*p) for p in positions}
    out = []
    for sims in (64, 256, 1024):
        for smart in (False, True):
            scores = []
            for s in range(6):
                good = 0
                rng2 = random.Random(SEED + 100 + s)
                for state, who in positions:
                    pick, _ = mcts(state, who, sims, rng2, smart=smart)
                    if pick in truth[(state, who)][1]:
                        good += 1
                scores.append(good)
            out.append({"sims": sims, "smart": smart, "scores": scores,
                        "mean": r(sum(scores) / len(scores) / len(positions), 4),
                        "spread": r(float(np.std([g / len(positions) for g in scores])), 4),
                        "seeds": len(scores)})
    return {"rows": out, "positions": len(positions), "note":
            "same positions as the constant sweep, so its floor is this one"}


def stage_versus_minimax():
    rng = random.Random(SEED + 4)
    positions = sample_positions(100, rng)
    truth = {p: solve(*p) for p in positions}
    rows = []
    for depth in (1, 2, 3, 4, 5):
        good, nodes_total = 0, 0
        for state, who in positions:
            nodes = [0]
            _, mv = minimax_limited(state, who, depth, nodes)
            nodes_total += nodes[0]
            if mv in truth[(state, who)][1]:
                good += 1
        rows.append({"depth": depth, "correct": good, "of": len(positions),
                     "nodes": int(nodes_total / len(positions)),
                     "share": r(good / len(positions), 4)})
    # y la búsqueda por árbol al mismo número de nodos
    tree = []
    for row in rows:
        sims = max(8, row["nodes"])
        good = 0
        rng2 = random.Random(SEED + 5)
        for state, who in positions:
            pick, _ = mcts(state, who, sims, rng2)
            if pick in truth[(state, who)][1]:
                good += 1
        tree.append({"sims": sims, "correct": good, "of": len(positions),
                     "share": r(good / len(positions), 4)})
    return {"minimax": rows, "tree": tree, "positions": len(positions)}


def stage_tree():
    """Un árbol concreto para dibujarlo: la posición, lo que el árbol visita, y
    lo que dice la respuesta exacta."""
    rng = random.Random(SEED + 6)
    positions = sample_positions(40, rng)
    # se elige una posición donde la respuesta exacta distinga jugadas, o el
    # dibujo no enseña nada
    for state, who in positions:
        val, best = solve(state, who)
        if 0 < len(best) < len(moves(state)):
            break
    frames = []
    for sims in (16, 64, 256, 1024):
        rng2 = random.Random(SEED + 7)
        pick, counts = mcts(state, who, sims, rng2)
        exact = {}
        for c in moves(state):
            v, _ = solve(play(state, c, who), 3 - who)
            exact[c] = -v
        frames.append({"sims": sims, "counts": {str(k): v for k, v in counts.items()},
                       "pick": pick, "correct": pick in best,
                       "exact": {str(k): r(v, 3) for k, v in exact.items()}})
    grid = [[0] * COLS for _ in range(ROWS)]
    for c, colvals in enumerate(state):
        for r_, v in enumerate(colvals):
            grid[ROWS - 1 - r_][c] = v
    return {"grid": grid, "who": who, "value": r(val, 3), "best": best,
            "frames": frames, "moves": moves(state)}


# ---------------------------------------------------------------------------
# La otra mitad: planear dentro de un modelo aprendido
# ---------------------------------------------------------------------------
def stage_model():
    """Un mundo de una dimensión con física escrita, un modelo lineal aprendido
    de él, y el error de imaginar k pasos, medido contra lo que pasa de verdad.

    El mundo: un carro con posición y velocidad, empujado por la acción, con
    rozamiento. Es lineal a propósito, para que el modelo pueda ser exacto en el
    límite y lo que se mida sea el efecto de los datos y no el de la forma.
    """
    rng = np.random.default_rng(SEED + 8)
    A = np.array([[1.0, 0.1], [0.0, 0.95]])
    B = np.array([[0.0], [0.1]])
    noise = 0.01

    def rollout(n, policy_noise=1.0):
        x = np.zeros(2)
        X, U, Y = [], [], []
        for _ in range(n):
            u = rng.normal(0, policy_noise, 1)
            nxt = A @ x + (B @ u).ravel() + rng.normal(0, noise, 2)
            X.append(x)
            U.append(u)
            Y.append(nxt)
            x = nxt
            if np.abs(x).max() > 10:
                x = np.zeros(2)
        return np.array(X), np.array(U), np.array(Y)

    rows = []
    for n in (50, 200, 1000, 5000):
        X, U, Y = rollout(n)
        Z = np.hstack([X, U])
        Wm, *_ = np.linalg.lstsq(Z, Y, rcond=None)
        # Z es [posicion, velocidad, empuje] y Y el estado siguiente, asi que la
        # matriz verdadera es A y B apiladas y traspuestas, no concatenadas
        truth = np.vstack([A.T, B.T])
        # error de un paso, y error de imaginar k pasos
        Xt, Ut, Yt = rollout(400)
        one = float(np.abs(np.hstack([Xt, Ut]) @ Wm - Yt).mean())
        horizon = []
        for k in (1, 2, 5, 10, 20, 40):
            errs = []
            for _ in range(200):
                x = rng.normal(0, 1, 2)
                xi = x.copy()
                us = rng.normal(0, 1, (k, 1))
                for t in range(k):
                    x = A @ x + (B @ us[t]).ravel()
                    xi = np.hstack([xi, us[t]]) @ Wm
                errs.append(float(np.linalg.norm(x - xi)))
            horizon.append({"k": k, "error": r(float(np.mean(errs)), 5)})
        rows.append({"samples": n, "one_step": r(one, 6),
                     "weight_error": r(float(np.abs(Wm - truth).max()), 6),
                     "horizon": horizon})
    # el suelo del error de un paso medido contra lo que pasó de verdad: el
    # mundo mete ruido gaussiano en cada paso, así que ni el modelo exacto puede
    # bajar de la media del valor absoluto de ese ruido
    floor = float(noise * math.sqrt(2 / math.pi))
    return {"rows": rows, "noise": noise, "one_step_floor": r(floor, 6),
            "note": "a linear world and a linear model, so what is measured is data, not shape"}


def main():
    t0 = time.time()
    print("1/7 resolver el juego entero")
    sol = cached("solve", stage_solve)
    print("2/7 acierto contra simulaciones")
    acc = cached("accuracy", stage_accuracy)
    print("3/7 la constante")
    con = cached("constant", stage_constant)
    print("4/7 el suelo de ruido")
    noise = cached("noise", stage_noise)
    print("5/7 contra minimax")
    vs = cached("versus", stage_versus_minimax)
    print("6/7 un árbol para dibujar")
    tree = cached("tree", stage_tree)
    print("7/7 planear dentro de un modelo")
    mod = cached("model", stage_model)

    data = {"meta": {"seed": SEED, "rows": ROWS, "cols": COLS, "connect": CONNECT,
                     "note": "positions and node counts, never a clock"},
            "solve": sol, "accuracy": acc, "constant": con, "noise": noise, "versus": vs,
            "tree": tree, "model": mod}
    f = OUT / "mcts.json"
    f.write_text(json.dumps(data, indent=1))
    print(f"escrito {f} ({f.stat().st_size / 1024:.1f} kB) en {time.time() - t0:.1f}s")

    print("\ncomprobaciones:")
    print(f"  juego resuelto: valor {sol['value']} para quien empieza, "
          f"{sol['positions']} posiciones en memoria")
    for row in acc["rows"]:
        if row["sims"] in (8, 128, 2048):
            print(f"  {row['sims']} simulaciones, listo={row['smart']}: {row['share']}")
    print(f"  mejor constante: {con['best_c']} (el libro dice {con['textbook_c']})")
    for row in noise["rows"]:
        print(f"  suelo {row['sims']} sims listo={row['smart']}: media {row['mean']}, "
              f"dispersion {row['spread']} sobre {row['seeds']} semillas")
    for row, tr in zip(vs["minimax"], vs["tree"]):
        print(f"  profundidad {row['depth']}: minimax {row['share']} con {row['nodes']} nodos, "
              f"árbol {tr['share']} con {tr['sims']}")
    for row in mod["rows"]:
        print(f"  {row['samples']} muestras: un paso {row['one_step']}, "
              f"40 pasos {row['horizon'][-1]['error']}")


if __name__ == "__main__":
    main()
