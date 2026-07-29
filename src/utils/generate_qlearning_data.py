"""Todo lo que cita el artículo 5.3a (Q-learning y SARSA), medido.

Aquí el mundo recuerda: una acción no solo paga, también mueve, y lo que se
puede hacer después depende de dónde te ha dejado. El mundo es el del acantilado
de Sutton y Barto, elegido porque es lo bastante pequeño para **resolverlo
exactamente** con iteración de valor, así que todo lo que aprenden los dos
métodos se puede puntuar contra la respuesta y no contra otra estimación.

 1. La respuesta exacta por iteración de valor, con su criterio de parada.
 2. SARSA y Q-learning, 50 semillas, con las dos cosas que hay que separar y
    casi nunca se separan: lo que cobran mientras aprenden y lo que vale la
    política que dejan.
 3. El camino que aprende cada uno, dibujado, y por qué son distintos.
 4. El barrido de epsilon, que es donde la diferencia entre los dos aparece y
    desaparece: con exploración cero los dos aprenden lo mismo.
 5. El barrido de alfa.
 6. El sesgo del máximo, en el problema de dos estados donde se puede calcular a
    mano lo que debería salir, y el doble estimador que lo quita.

Tarda unos tres minutos en cuatro núcleos. Etapas cacheadas en
~/.atlas_vision_data/qlearn_cache/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "q-learning" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "qlearn_cache"
CACHE.mkdir(parents=True, exist_ok=True)

SEED = 63
SEEDS = 50
EPISODES = 500
ROWS, COLS = 4, 12
START = (3, 0)
GOAL = (3, 11)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]      # arriba, abajo, izquierda, derecha
ACTION_NAMES = ["up", "down", "left", "right"]


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


def is_cliff(rc):
    return rc[0] == ROWS - 1 and 0 < rc[1] < COLS - 1


def step(rc, a):
    """El mundo entero, en cinco líneas. Determinista y escrito, no aprendido."""
    dr, dc = ACTIONS[a]
    nr = min(max(rc[0] + dr, 0), ROWS - 1)
    nc = min(max(rc[1] + dc, 0), COLS - 1)
    nxt = (nr, nc)
    if is_cliff(nxt):
        return START, -100.0, False
    return nxt, -1.0, nxt == GOAL


def sid(rc):
    return rc[0] * COLS + rc[1]


N_STATES = ROWS * COLS
N_ACTIONS = len(ACTIONS)


def value_iteration(tol=1e-12, max_sweeps=100000):
    """La respuesta. Sin descuento: el episodio termina, así que no hace falta.

    El criterio de parada no se elige por costumbre: se itera hasta que el mayor
    cambio de un barrido cae por debajo de la precisión de un doble, y se guarda
    cuántos barridos hicieron falta.
    """
    Q = np.zeros((N_STATES, N_ACTIONS))
    sweeps = 0
    for _ in range(max_sweeps):
        delta = 0.0
        for s in range(N_STATES):
            rc = (s // COLS, s % COLS)
            if rc == GOAL:
                continue
            for a in range(N_ACTIONS):
                nxt, rew, done = step(rc, a)
                target = rew + (0.0 if done else Q[sid(nxt)].max())
                delta = max(delta, abs(target - Q[s, a]))
                Q[s, a] = target
        sweeps += 1
        if delta < tol:
            break
    return Q, sweeps, delta


def greedy_path(Q, N=None, limit=200):
    """El camino que sigue una tabla si se deja de explorar.

    Con un detalle que no es cosmético. Todas las recompensas son negativas y la
    tabla empieza en cero, así que una casilla que nunca se probó vale MÁS que
    cualquier cosa aprendida, y una política avariciosa leída sin cuidado se va
    derecha a lo que no ha visto nunca. Una política solo se puede leer donde
    hay datos: si se pasan los conteos, solo se eligen acciones probadas.
    """
    rc = START
    path, total = [rc], 0.0
    for _ in range(limit):
        q = Q[sid(rc)].copy()
        if N is not None and N[sid(rc)].max() > 0:
            q = np.where(N[sid(rc)] > 0, q, -np.inf)
        a = int(np.argmax(q))
        rc, rew, done = step(rc, a)
        total += rew
        path.append(rc)
        if done:
            return path, total, True
    return path, total, False


def learn(kind, seed, episodes=EPISODES, eps=0.1, alpha=0.5, double=False):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    Q2 = np.zeros((N_STATES, N_ACTIONS))
    N = np.zeros((N_STATES, N_ACTIONS))
    returns = np.zeros(episodes)
    for ep in range(episodes):
        rc = START
        total = 0.0
        pick = (lambda s: int(rng.integers(N_ACTIONS)) if rng.random() < eps
                else int(np.argmax((Q + Q2)[s] if double else Q[s])))
        a = pick(sid(rc))
        for _ in range(400):
            nxt, rew, done = step(rc, a)
            total += rew
            N[sid(rc), a] += 1
            a2 = pick(sid(nxt))
            if double:
                if rng.random() < 0.5:
                    best = int(np.argmax(Q[sid(nxt)]))
                    target = rew + (0.0 if done else Q2[sid(nxt), best])
                    Q[sid(rc), a] += alpha * (target - Q[sid(rc), a])
                else:
                    best = int(np.argmax(Q2[sid(nxt)]))
                    target = rew + (0.0 if done else Q[sid(nxt), best])
                    Q2[sid(rc), a] += alpha * (target - Q2[sid(rc), a])
            elif kind == "sarsa":
                target = rew + (0.0 if done else Q[sid(nxt), a2])
                Q[sid(rc), a] += alpha * (target - Q[sid(rc), a])
            else:
                target = rew + (0.0 if done else Q[sid(nxt)].max())
                Q[sid(rc), a] += alpha * (target - Q[sid(rc), a])
            rc, a = nxt, a2
            if done:
                break
        returns[ep] = total
    return (Q + Q2 if double else Q), returns, N


def stage_truth():
    Q, sweeps, delta = value_iteration()
    path, total, ok = greedy_path(Q)
    V = Q.max(axis=1)
    return {"Q": r(Q, 4), "V": r(V, 4), "sweeps": sweeps, "delta": float(delta),
            "path": [list(p) for p in path], "return": r(total, 2), "reaches_goal": ok,
            "rows": ROWS, "cols": COLS, "start": list(START), "goal": list(GOAL),
            "cliff": [[ROWS - 1, c] for c in range(1, COLS - 1)],
            "actions": ACTION_NAMES}


def stage_learners():
    truth = value_iteration()[0]
    out = {}
    for kind in ("sarsa", "qlearning"):
        curves, greedy_returns, gaps, paths = [], [], [], []
        reached = 0
        for s in range(SEEDS):
            Q, ret, N = learn(kind, SEED * 100 + s)
            curves.append(ret)
            path, total, ok = greedy_path(Q, N)
            reached += int(ok)
            # Solo las que llegan entran en la media. SARSA aprende el valor de
            # SU política exploradora, no el óptimo, así que leerle una política
            # avariciosa es pedirle algo que no promete: en las que no llegan, la
            # lectura da vueltas. Promediar eso con un castigo inventado
            # convertiría un hecho en un número sin sentido.
            if ok:
                greedy_returns.append(total)
            visited = sorted({sid(tuple(p)) for p in path})
            gaps.append(float(np.abs(Q[visited] - truth[visited]).max()))
            paths.append([list(p) for p in path])
        curves = np.array(curves)
        # el camino que se publica es el más frecuente entre las semillas, no el
        # de la primera: una semilla suelta puede aprender un rodeo raro
        keys = ["|".join(f"{p[0]},{p[1]}" for p in path) for path in paths]
        common = max(set(keys), key=keys.count)
        idx = keys.index(common)
        out[kind] = {
            "online": r(curves.mean(axis=0), 3),
            "online_last100": r(float(curves[:, -100:].mean()), 3),
            "greedy_return": r(float(np.mean(greedy_returns)), 3),
            "greedy_best": r(float(np.max(greedy_returns)), 3),
            "greedy_worst": r(float(np.min(greedy_returns)), 3),
            "path": paths[idx],
            "path_share": r(keys.count(common) / len(keys), 3),
            "path_return": r(greedy_path(*learn(kind, SEED * 100 + idx)[::2])[1], 2),
            "q_gap": r(float(np.mean(gaps)), 4),
            "reaches_goal": reached,
            "seeds": SEEDS,
        }
    return out


def stage_epsilon():
    truth = value_iteration()[0]
    rows = []
    for eps in (0.5, 0.2, 0.1, 0.05, 0.01, 0.0):
        row = {"eps": eps}
        for kind in ("sarsa", "qlearning"):
            online, greedy_ret, optimal, reached = [], [], 0, 0
            for s in range(20):
                Q, ret, N = learn(kind, SEED * 7 + s, eps=eps)
                online.append(float(ret[-100:].mean()))
                path, total, ok = greedy_path(Q, N)
                reached += int(ok)
                if ok:
                    greedy_ret.append(total)
                if ok and abs(total - (-13.0)) < 1e-9:
                    optimal += 1
            row[kind] = {"online": r(float(np.mean(online)), 3),
                         "greedy": r(float(np.mean(greedy_ret)), 3) if greedy_ret else None,
                         "reaches_goal": reached,
                         "optimal_paths": optimal, "of": 20}
        rows.append(row)
    return {"rows": rows, "optimal_return": r(greedy_path(truth)[1], 2)}


def stage_alpha():
    rows = []
    for alpha in (0.05, 0.1, 0.25, 0.5, 0.8, 1.0):
        row = {"alpha": alpha}
        for kind in ("sarsa", "qlearning"):
            online, greedy_ret = [], []
            for s in range(20):
                Q, ret, N = learn(kind, SEED * 11 + s, alpha=alpha)
                online.append(float(ret[-100:].mean()))
                path, total, ok = greedy_path(Q, N)
                if ok:
                    greedy_ret.append(total)
            row[kind] = {"online": r(float(np.mean(online)), 3),
                         "greedy": r(float(np.mean(greedy_ret)), 3) if greedy_ret else None,
                         "reaches_goal": len(greedy_ret)}
        rows.append(row)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# El sesgo del máximo, donde la respuesta se puede escribir
# ---------------------------------------------------------------------------
def bias_env(rng, n_actions=10, mu=-0.1, sigma=1.0):
    """Dos estados. Desde A, derecha termina con 0; izquierda va a B, y desde B
    cualquiera de sus muchas acciones termina con una recompensa de media -0,1.
    Así que izquierda es peor, y el máximo sobre muchas estimaciones ruidosas
    parece bueno."""
    return n_actions, mu, sigma


def run_bias(double, seeds=1000, episodes=300, eps=0.1, alpha=0.1,
             n_actions=10, mu=-0.1, sigma=1.0):
    rng = np.random.default_rng(SEED + 5)
    left_share = np.zeros(episodes)
    qa_left = np.zeros(episodes)
    for s in range(seeds):
        QA = np.zeros(2)                     # A: izquierda, derecha
        QB = np.zeros(n_actions)
        QA2 = np.zeros(2)
        QB2 = np.zeros(n_actions)
        for ep in range(episodes):
            if rng.random() < eps:
                a = int(rng.integers(2))
            else:
                q = (QA + QA2) if double else QA
                a = int(np.argmax(q)) if q[0] != q[1] else int(rng.integers(2))
            left_share[ep] += a == 0
            qa_left[ep] += (QA[0] + QA2[0]) if double else QA[0]
            if a == 1:
                target = 0.0
                if double and rng.random() < 0.5:
                    QA2[1] += alpha * (target - QA2[1])
                elif double:
                    QA[1] += alpha * (target - QA[1])
                else:
                    QA[1] += alpha * (target - QA[1])
                continue
            # izquierda: a B, y desde B una acción cualquiera termina
            if double:
                if rng.random() < 0.5:
                    best = int(np.argmax(QB))
                    QA[0] += alpha * (QB2[best] - QA[0])
                else:
                    best = int(np.argmax(QB2))
                    QA2[0] += alpha * (QB[best] - QA2[0])
            else:
                QA[0] += alpha * (QB.max() - QA[0])
            b = int(rng.integers(n_actions)) if rng.random() < eps else (
                int(np.argmax((QB + QB2) if double else QB)))
            reward = rng.normal(mu, sigma)
            if double:
                if rng.random() < 0.5:
                    QB[b] += alpha * (reward - QB[b])
                else:
                    QB2[b] += alpha * (reward - QB2[b])
            else:
                QB[b] += alpha * (reward - QB[b])
    return left_share / seeds, qa_left / seeds


def stage_bias():
    single, q_single = run_bias(False)
    double, q_double = run_bias(True)
    n_actions, mu, sigma = 10, -0.1, 1.0
    # lo que el máximo de n estimaciones de media mu aporta de más, en el límite
    # de estimaciones independientes con la varianza que tienen al principio
    return {
        "single": r(single[::5], 4), "double": r(double[::5], 4), "stride": 5,
        "q_single": r(q_single[::5], 4), "q_double": r(q_double[::5], 4),
        "episodes": len(single), "seeds": 1000, "eps": 0.1,
        "n_actions": n_actions, "mu": mu, "sigma": sigma,
        "truth_left": r(mu, 3), "truth_right": 0.0,
        "final_single": r(float(single[-20:].mean()), 4),
        "final_double": r(float(double[-20:].mean()), 4),
        "greedy_share": r(1 - 0.1 / 2, 3),
        "chance": 0.5,
    }


def main():
    t0 = time.time()
    print("1/5 la respuesta exacta")
    truth = cached("truth-v2", stage_truth)
    print("2/5 los dos que aprenden")
    lear = cached("learners-v3", stage_learners)
    print("3/5 barrido de epsilon")
    eps = cached("epsilon-v3", stage_epsilon)
    print("4/5 barrido de alfa")
    alp = cached("alpha-v3", stage_alpha)
    print("5/5 el sesgo del máximo")
    bias = cached("bias", stage_bias)

    data = {"meta": {"seed": SEED, "seeds": SEEDS, "episodes": EPISODES,
                     "eps": 0.1, "alpha": 0.5,
                     "note": "returns, not clocks: the same numbers anywhere"},
            "truth": truth, "learners": lear, "epsilon": eps, "alpha": alp,
            "bias": bias}
    f = OUT / "qlearn.json"
    f.write_text(json.dumps(data, indent=1))
    print(f"escrito {f} ({f.stat().st_size / 1024:.1f} kB) en {time.time() - t0:.1f}s")

    print("\ncomprobaciones:")
    print(f"  iteración de valor: {truth['sweeps']} barridos, delta {truth['delta']:.2e}, "
          f"camino óptimo {truth['return']}")
    for kind, d in lear.items():
        print(f"  {kind}: mientras aprende {d['online_last100']}, su política "
              f"{d['greedy_return']} (llega en {d['reaches_goal']}/{d['seeds']}), "
              f"camino más común en {d['path_share']} de las semillas")
    for row in eps["rows"]:
        print(f"  eps={row['eps']}: sarsa {row['sarsa']['greedy']} (llega "
              f"{row['sarsa']['reaches_goal']}/20, óptimos {row['sarsa']['optimal_paths']}), "
              f"qlearning {row['qlearning']['greedy']} (llega "
              f"{row['qlearning']['reaches_goal']}/20, óptimos "
              f"{row['qlearning']['optimal_paths']})")
    print(f"  sesgo: una tabla {bias['final_single']} de izquierdas, dos tablas "
          f"{bias['final_double']}, azar {bias['chance']}")


if __name__ == "__main__":
    main()
