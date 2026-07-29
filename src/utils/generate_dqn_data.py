"""Todo lo que cita el artículo 5.3b (DQN), medido.

El artículo anterior guardaba un número por casilla y podía comprobarse contra
la respuesta exacta. Aquí la tabla se sustituye por una función de dos entradas
que tiene que **adivinar** los valores que nunca vio, sobre el mismo acantilado,
así que la respuesta exacta sigue estando y se puede medir exactamente lo que
cuesta el cambio.

 1. Tres representaciones sobre el mismo mundo: la tabla del artículo anterior,
    una función lineal de las dos coordenadas, y una red pequeña. Distancia a la
    respuesta exacta y valor de la política que dejan.
 2. Las dos muletas de DQN, ablacionadas en cuadro (repetición sí o no, red
    objetivo sí o no), cinco semillas cada celda, con suelo de ruido.
 3. La sobreestimación: el valor que la red se cree contra lo que su política
    cobra de verdad, y el doble estimador que lo corrige.
 4. Los pesos de la red entrenada, para que el navegador la ejecute y se pueda
    comprobar que da los mismos valores.

Tarda unos ocho minutos en cuatro núcleos. Etapas cacheadas en
~/.atlas_vision_data/dqn_cache/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "dqn" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "dqn_cache"
CACHE.mkdir(parents=True, exist_ok=True)

SEED = 64
SEEDS = 5
EPISODES = 2500
TABLE_EPISODES = 400
GAMMA = 0.95
SCALE = 100.0
ROWS, COLS = 4, 12
START = (3, 0)
GOAL = (3, 11)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_STATES, N_ACTIONS = ROWS * COLS, 4
HIDDEN = 64


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
    dr, dc = ACTIONS[a]
    nr = min(max(rc[0] + dr, 0), ROWS - 1)
    nc = min(max(rc[1] + dc, 0), COLS - 1)
    nxt = (nr, nc)
    if is_cliff(nxt):
        return START, -100.0, False
    return nxt, -1.0, nxt == GOAL


def sid(rc):
    return rc[0] * COLS + rc[1]


def obs(rc):
    """Lo único que ve la red: dos números. No hay una entrada por casilla, así
    que dos casillas parecidas tienen entradas parecidas y lo que aprenda de una
    se le pega a la otra, que es a la vez toda la ventaja y todo el peligro."""
    return np.array([rc[1] / (COLS - 1), rc[0] / (ROWS - 1)])


ALL_OBS = np.array([obs((s // COLS, s % COLS)) for s in range(N_STATES)])


def value_iteration(tol=1e-12, gamma=None):
    """La respuesta exacta, con el mismo descuento que usan los que aprenden.

    El artículo anterior no descontaba y no le hacía falta: una tabla con
    episodios que terminan converge sin descuento. Una función que arranca de
    sus propias predicciones no tiene esa garantía, así que aquí hay descuento, y
    la respuesta contra la que se puntúa lleva el mismo.
    """
    gamma = GAMMA if gamma is None else gamma
    Q = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(20000):
        delta = 0.0
        for s in range(N_STATES):
            rc = (s // COLS, s % COLS)
            if rc == GOAL:
                continue
            for a in range(N_ACTIONS):
                nxt, rew, done = step(rc, a)
                target = rew + (0.0 if done else gamma * Q[sid(nxt)].max())
                delta = max(delta, abs(target - Q[s, a]))
                Q[s, a] = target
        if delta < tol:
            break
    return Q


TRUTH = value_iteration()


def greedy_path(qfun, limit=200):
    """Devuelve el camino, lo que cobra sin descontar (que es lo legible) y lo
    que cobra descontado (que es lo que la red dice estimar).

    Los dos hacen falta y no se pueden mezclar: el valor que la red guarda es la
    suma descontada, y ponerlo al lado de la suma sin descontar compara dos
    cantidades distintas, que en este mundo se llevan 3,3 de diferencia sin que
    haya ningún sesgo de por medio.
    """
    rc = START
    total, disc, path = 0.0, 0.0, [rc]
    for k in range(limit):
        a = int(np.argmax(qfun(rc)))
        rc, rew, done = step(rc, a)
        total += rew
        disc += (GAMMA ** k) * rew
        path.append(rc)
        if done:
            return path, total, True, disc
    return path, total, False, disc


# ---------------------------------------------------------------------------
# Una red de dos capas, escrita a mano para poder exportarla y volver a
# ejecutarla en el navegador sin depender de nada.
# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, rng, n_in=2, hidden=HIDDEN, n_out=N_ACTIONS):
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t = 0

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, X):
        H = np.maximum(X @ self.W1 + self.b1, 0)
        return H @ self.W2 + self.b2, H

    def __call__(self, rc):
        return self.forward(obs(rc)[None, :])[0][0]

    def fit(self, X, A, Y, lr=1e-3):
        """Un paso de Adam sobre el error cuadrático de las acciones tomadas."""
        Q, H = self.forward(X)
        pred = Q[np.arange(len(A)), A]
        dQ = np.zeros_like(Q)
        dQ[np.arange(len(A)), A] = 2 * (pred - Y) / len(A)
        gW2 = H.T @ dQ
        gb2 = dQ.sum(axis=0)
        dH = (dQ @ self.W2.T) * (H > 0)
        gW1 = X.T @ dH
        gb1 = dH.sum(axis=0)
        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
        self.t += 1
        for k, p in self.params().items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return float(((pred - Y) ** 2).mean())

    def export(self):
        return {k: r(v, 6) for k, v in self.params().items()}


def train_dqn(seed, replay=True, target=True, double=False, episodes=EPISODES,
              eps0=0.3, eps1=0.05, lr=1e-3, batch=32, target_every=200, buffer=10000):
    rng = np.random.default_rng(seed)
    net = MLP(rng)
    tgt = MLP(rng)
    for k, v in net.params().items():
        tgt.params()[k][...] = v
    mem = []
    returns, qmax = [], []
    steps = 0
    for ep in range(episodes):
        # la exploración baja durante la tirada: con una tabla daba igual, con
        # una función no, porque cada paso raro también reescribe los vecinos
        eps = eps0 + (eps1 - eps0) * min(1.0, ep / (0.6 * episodes))
        rc = START
        total = 0.0
        for _ in range(120):
            q = net(rc)
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(np.argmax(q))
            nxt, rew, done = step(rc, a)
            total += rew
            # la recompensa entra escalada: sin eso el objetivo del primer paso
            # vale -100 y la red se va a pique antes de aprender nada
            mem.append((obs(rc), a, rew / SCALE, obs(nxt), done))
            if len(mem) > buffer:
                mem.pop(0)
            # con repetición se muestrea del recuerdo; sin ella, el lote es lo
            # que acaba de pasar, que es la diferencia entera
            if replay and len(mem) >= batch:
                idx = rng.integers(0, len(mem), batch)
                sample = [mem[i] for i in idx]
            else:
                sample = mem[-1:]
            X = np.array([s[0] for s in sample])
            A = np.array([s[1] for s in sample])
            R = np.array([s[2] for s in sample])
            X2 = np.array([s[3] for s in sample])
            D = np.array([s[4] for s in sample], dtype=float)
            qnext_online = net.forward(X2)[0]
            qnext_target = (tgt if target else net).forward(X2)[0]
            if double:
                pick = qnext_online.argmax(axis=1)
                boot = qnext_target[np.arange(len(A)), pick]
            else:
                boot = qnext_target.max(axis=1)
            Y = R + (1 - D) * GAMMA * boot
            net.fit(X, A, Y, lr=lr)
            steps += 1
            if target and steps % target_every == 0:
                for k, v in net.params().items():
                    tgt.params()[k][...] = v
            rc = nxt
            if done:
                break
        returns.append(total)
        qmax.append(float(np.abs(net.forward(ALL_OBS)[0]).max()))
    path, total, ok, disc = greedy_path(net)
    Qnet = net.forward(ALL_OBS)[0] * SCALE
    return {"net": net, "returns": np.array(returns), "qmax": np.array(qmax),
            "path": [list(p) for p in path], "return": total, "reached": ok,
            "discounted": disc,
            # la distancia se mide donde la política pisa: los tramos que nadie
            # visita nunca (el acantilado entra teletransportando, así que no se
            # está nunca en él) no tienen dato y la red extrapola lo que quiere
            "gap": float(np.abs(Qnet[[sid(tuple(p)) for p in path]]
                                - TRUTH[[sid(tuple(p)) for p in path]]).max()),
            "gap_all": float(np.abs(Qnet - TRUTH).max()),
            "Q": Qnet}


def train_linear(seed, episodes=EPISODES, eps=0.1, lr=0.005):
    """El control barato: los mismos dos números, pero la salida es una recta.

    Sirve para separar dos cosas que se confunden siempre: que hay que
    generalizar, y que hay que generalizar con una red. Y trae su propio
    resultado: con una función que arranca de sus propias predicciones, los
    pesos pueden irse a infinito, así que la divergencia se detecta y se cuenta
    en vez de dejarla convertirse en NaN.
    """
    rng = np.random.default_rng(seed)
    Wl = np.zeros((3, N_ACTIONS))           # todo en unidades escaladas
    feat = lambda rc: np.concatenate([obs(rc), [1.0]])
    returns, diverged = [], False
    for ep in range(episodes):
        rc = START
        total = 0.0
        for _ in range(120):
            x = feat(rc)
            q = x @ Wl
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(np.argmax(q))
            nxt, rew, done = step(rc, a)
            total += rew
            target = rew / SCALE + (0.0 if done else GAMMA * float((feat(nxt) @ Wl).max()))
            Wl[:, a] += lr * (target - q[a]) * x
            rc = nxt
            if done:
                break
        returns.append(total)
        if not np.isfinite(Wl).all() or np.abs(Wl).max() > 1e6:
            diverged = True
            returns.extend([returns[-1]] * (episodes - len(returns)))
            break
    qfun = lambda rc: feat(rc) @ Wl
    path, total, ok, disc = (([START], 0.0, False, 0.0) if diverged
                             else greedy_path(qfun))
    Q = np.array([feat((s // COLS, s % COLS)) @ Wl for s in range(N_STATES)]) * SCALE
    vis = sorted({sid(tuple(p)) for p in path})
    return {"returns": np.array(returns), "path": [list(p) for p in path],
            "return": total, "reached": ok, "discounted": disc, "W": Wl,
            "diverged": diverged,
            "gap": (float("inf") if diverged
                    else float(np.abs(Q[vis] - TRUTH[vis]).max())), "Q": Q}


def train_table(seed, episodes=TABLE_EPISODES, eps=0.1, alpha=0.5):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    N = np.zeros((N_STATES, N_ACTIONS))
    returns = []
    for ep in range(episodes):
        rc = START
        total = 0.0
        for _ in range(120):
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(np.argmax(Q[sid(rc)]))
            nxt, rew, done = step(rc, a)
            total += rew
            N[sid(rc), a] += 1
            Q[sid(rc), a] += alpha * (rew + (0.0 if done else GAMMA * Q[sid(nxt)].max())
                                      - Q[sid(rc), a])
            rc = nxt
            if done:
                break
        returns.append(total)
    qfun = lambda rc: np.where(N[sid(rc)] > 0, Q[sid(rc)], -np.inf) if N[sid(rc)].max() > 0 else Q[sid(rc)]
    path, total, ok, disc = greedy_path(qfun)
    vis = sorted({sid(tuple(p)) for p in path})
    return {"returns": np.array(returns), "path": [list(p) for p in path],
            "return": total, "reached": ok, "discounted": disc,
            "gap": float(np.abs(Q[vis] - TRUTH[vis]).max()), "Q": Q,
            "episodes": episodes}


def stage_representations():
    out = {}
    for name, fn in (("table", train_table), ("linear", train_linear),
                     ("network", lambda s: train_dqn(s))):
        rets, gaps, finals, reached, paths, blew = [], [], [], 0, [], 0
        for s in range(SEEDS):
            res = fn(SEED * 10 + s)
            blew += int(res.get("diverged", False))
            rets.append(res["returns"])
            gaps.append(res["gap"])
            finals.append(res["return"] if res["reached"] else None)
            reached += int(res["reached"])
            paths.append(res["path"])
        good = [f for f in finals if f is not None]
        keys = ["|".join(f"{p[0]},{p[1]}" for p in path) for path in paths]
        common = max(set(keys), key=keys.count)
        arr = np.array(rets)
        stride = max(1, arr.shape[1] // 200)
        out[name] = {
            "curve": r(arr.mean(axis=0)[::stride], 3), "stride": stride,
            "episodes": arr.shape[1],
            "last50": r(float(arr[:, -50:].mean()), 3),
            "gap": (None if not np.isfinite(gaps).all()
                    else r(float(np.mean(gaps)), 3)),
            "diverged": blew,
            "greedy": r(float(np.mean(good)), 3) if good else None,
            "reaches_goal": reached, "seeds": SEEDS,
            "path": paths[keys.index(common)],
            "path_share": r(keys.count(common) / len(keys), 3),
        }
    return out


def stage_ablation():
    rows = []
    for replay in (True, False):
        for target in (True, False):
            finals, gaps, qmaxes, reached = [], [], [], 0
            for s in range(SEEDS):
                res = train_dqn(SEED * 20 + s, replay=replay, target=target)
                finals.append(res["return"] if res["reached"] else None)
                gaps.append(res["gap"])
                qmaxes.append(float(res["qmax"][-1]))
                reached += int(res["reached"])
            good = [f for f in finals if f is not None]
            rows.append({"replay": replay, "target": target,
                         "greedy": r(float(np.mean(good)), 3) if good else None,
                         "reaches_goal": reached, "of": SEEDS,
                         "gap": r(float(np.mean(gaps)), 3),
                         "gap_spread": r(float(np.std(gaps)), 3),
                         "qmax": r(float(np.mean(qmaxes)), 3)})
    # suelo de ruido: la misma celda con semillas distintas
    base = [row for row in rows if row["replay"] and row["target"]][0]
    return {"rows": rows, "seeds": SEEDS, "episodes": EPISODES,
            "noise_floor": base["gap_spread"]}


def stage_overestimate():
    out = []
    for double in (False, True):
        believed, actual, gaps = [], [], []
        for s in range(SEEDS):
            res = train_dqn(SEED * 30 + s, double=double)
            q0 = res["Q"][sid(START)].max()
            believed.append(float(q0))
            # lo que su propia política cobra, DESCONTADO, que es la misma
            # cantidad que la red dice estimar
            actual.append(res["discounted"] if res["reached"] else None)
            gaps.append(res["gap"])
        good = [a for a in actual if a is not None]
        out.append({"double": double,
                    "believes": r(float(np.mean(believed)), 3),
                    "gets": r(float(np.mean(good)), 3) if good else None,
                    "bias": r(float(np.mean(believed)) - (float(np.mean(good)) if good else 0), 3),
                    "gap": r(float(np.mean(gaps)), 3),
                    "reached": len(good), "of": SEEDS})
    undiscounted = greedy_path(lambda rc: TRUTH[sid(rc)])[1]
    return {"rows": out, "truth_at_start": r(float(TRUTH[sid(START)].max()), 3),
            "truth_undiscounted": r(undiscounted, 3), "gamma": GAMMA}


def stage_weights():
    """Una red concreta, exportada, para que el navegador la ejecute."""
    res = train_dqn(SEED * 40 + 1)
    net = res["net"]
    Q = net.forward(ALL_OBS)[0] * SCALE
    return {"weights": net.export(), "hidden": HIDDEN, "scale": SCALE,
            "Q": r(Q, 4), "path": res["path"], "return": res["return"],
            "gap": r(res["gap_all"], 4), "gap_on_path": r(res["gap"], 4),
            "obs": r(ALL_OBS, 6), "seed": SEED * 40 + 1}


def main():
    t0 = time.time()
    print("1/4 tres representaciones")
    rep = cached("representations-v3", stage_representations)
    print("2/4 las dos muletas, en cuadro")
    abl = cached("ablation-v2", stage_ablation)
    print("3/4 sobreestimación")
    over = cached("overestimate-v3", stage_overestimate)
    print("4/4 pesos para el navegador")
    wts = cached("weights-v2", stage_weights)

    data = {"meta": {"seed": SEED, "seeds": SEEDS, "episodes": EPISODES,
                     "rows": ROWS, "cols": COLS, "start": list(START),
                     "goal": list(GOAL), "hidden": HIDDEN,
                     "cliff": [[ROWS - 1, c] for c in range(1, COLS - 1)],
                     "gamma": GAMMA, "scale": SCALE,
                     "table_episodes": TABLE_EPISODES,
                     "truth_return": r(greedy_path(lambda rc: TRUTH[sid(rc)])[1], 2),
                     "truth_discounted": r(greedy_path(lambda rc: TRUTH[sid(rc)])[3], 3),
                     "truth_Q": r(TRUTH, 4),
                     "note": "returns and distances, never a clock"},
            "representations": rep, "ablation": abl, "overestimate": over,
            "weights": wts}
    f = OUT / "dqn.json"
    f.write_text(json.dumps(data, indent=1))
    print(f"escrito {f} ({f.stat().st_size / 1024:.1f} kB) en {time.time() - t0:.1f}s")

    print("\ncomprobaciones:")
    for name, d in rep.items():
        print(f"  {name}: política {d['greedy']} (llega {d['reaches_goal']}/{d['seeds']}, "
              f"revienta {d['diverged']}), distancia a la respuesta {d['gap']}")
    for row in abl["rows"]:
        print(f"  repetición={row['replay']} objetivo={row['target']}: "
              f"{row['greedy']} (llega {row['reaches_goal']}/{row['of']}), "
              f"distancia {row['gap']}, mayor valor {row['qmax']}")
    for row in over["rows"]:
        print(f"  doble={row['double']}: se cree {row['believes']}, cobra {row['gets']}, "
              f"sesgo {row['bias']}")


if __name__ == "__main__":
    main()
