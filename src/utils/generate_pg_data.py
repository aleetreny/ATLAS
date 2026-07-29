"""Todo lo que cita el artículo 5.3c (REINFORCE, A2C, PPO, SAC), medido.

Los métodos de este artículo no aprenden un valor y luego actúan: mueven la
política directamente, en la dirección de un gradiente que hay que estimar con
muestras. Así que la pregunta que decide todo es cuánto ruido tiene esa
estimación, y aquí se puede contestar sin discusión, porque el mundo es lo
bastante pequeño para calcular el gradiente **exacto** y comparar contra él.

 1. El gradiente exacto de la política, resuelto con álgebra lineal: la
    distribución de visitas y los valores salen de invertir una matriz, no de
    muestrear.
 2. El estimador de REINFORCE contra ese gradiente: error relativo y coseno,
    barriendo el número de episodios, con línea base y sin ella. La reducción de
    varianza que compra la línea base es un número, no un adjetivo.
 3. Por qué la línea base no sesga nada, comprobado en vez de citado.
 4. El recorte de PPO: cuántas muestras recorta de verdad, cuánto se mueve la
    política en cada actualización, y qué pasa sin él.
 5. El premio a la entropía, barrido: lo que compra y lo que cuesta.

Tarda unos cuatro minutos en cuatro núcleos. Etapas cacheadas en
~/.atlas_vision_data/pg_cache/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "policy-gradient" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "pg_cache"
CACHE.mkdir(parents=True, exist_ok=True)

SEED = 65
SIZE = 5
GAMMA = 0.9
N_STATES = SIZE * SIZE
N_ACTIONS = 4
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
START = 0                       # esquina de arriba a la izquierda
GOAL = N_STATES - 1             # esquina de abajo a la derecha
TRAP = SIZE * 2 + 2             # el centro, que paga mal
STEP_COST = -0.05
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0


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


def move(s, a):
    row, col = divmod(s, SIZE)
    dr, dc = ACTIONS[a]
    nr = min(max(row + dr, 0), SIZE - 1)
    nc = min(max(col + dc, 0), SIZE - 1)
    return nr * SIZE + nc


TERMINAL = {GOAL}
P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
R = np.zeros((N_STATES, N_ACTIONS))
for s in range(N_STATES):
    for a in range(N_ACTIONS):
        if s in TERMINAL:
            P[s, a, s] = 1.0
            continue
        nxt = move(s, a)
        P[s, a, nxt] = 1.0
        R[s, a] = STEP_COST + (GOAL_REWARD if nxt == GOAL else 0.0) + (
            TRAP_REWARD if nxt == TRAP else 0.0)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def policy(theta):
    return softmax(theta.reshape(N_STATES, N_ACTIONS))


def exact_values(theta):
    """Q y V exactos de ESTA política, resolviendo el sistema lineal.

    Nada de muestrear: la ecuación de Bellman para una política fija es lineal,
    así que se invierte y se acabó. Es lo que convierte el resto del artículo en
    una comparación en vez de en dos estimaciones discutiendo.
    """
    pi = policy(theta)
    Ppi = np.einsum("sa,san->sn", pi, P)
    Rpi = (pi * R).sum(axis=1)
    A = np.eye(N_STATES) - GAMMA * Ppi
    A[GOAL] = 0.0
    A[GOAL, GOAL] = 1.0
    b = Rpi.copy()
    b[GOAL] = 0.0
    V = np.linalg.solve(A, b)
    Q = R + GAMMA * P @ V
    return V, Q, pi


def exact_gradient(theta):
    """El gradiente exacto de J respecto de theta, por el teorema del gradiente
    de la política, con la distribución de visitas descontada resuelta también
    en forma cerrada."""
    V, Q, pi = exact_values(theta)
    Ppi = np.einsum("sa,san->sn", pi, P)
    mu = np.zeros(N_STATES)
    mu[START] = 1.0
    d = np.linalg.solve(np.eye(N_STATES) - GAMMA * Ppi.T, mu)
    d[GOAL] = 0.0
    grad = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        if s in TERMINAL:
            continue
        for a in range(N_ACTIONS):
            # d log pi(a|s) / d theta[s,b] = 1[a=b] - pi(b|s)
            grad[s] += d[s] * pi[s, a] * Q[s, a] * (np.eye(N_ACTIONS)[a] - pi[s])
    return grad.ravel(), V, Q, pi


def rollout(theta, rng, max_steps=60):
    pi = policy(theta)
    s = START
    traj = []
    for _ in range(max_steps):
        a = int(rng.choice(N_ACTIONS, p=pi[s]))
        nxt = move(s, a)
        rew = R[s, a]
        traj.append((s, a, rew))
        s = nxt
        if s == GOAL:
            break
    return traj


def reinforce_gradient(theta, rng, episodes, baseline=None):
    """El estimador. `baseline` resta un número que no depende de la acción, que
    es la única condición para que no sesgue nada."""
    pi = policy(theta)
    grad = np.zeros((N_STATES, N_ACTIONS))
    total = 0.0
    for _ in range(episodes):
        traj = rollout(theta, rng)
        G = 0.0
        returns = []
        for _, _, rew in reversed(traj):
            G = rew + GAMMA * G
            returns.append(G)
        returns.reverse()
        total += returns[0]
        for t, (s, a, _) in enumerate(traj):
            adv = returns[t] - (0.0 if baseline is None else baseline[s])
            grad[s] += (GAMMA ** t) * adv * (np.eye(N_ACTIONS)[a] - pi[s])
    return grad.ravel() / episodes, total / episodes


def stage_gradient():
    rng0 = np.random.default_rng(SEED)
    theta = rng0.normal(0, 0.5, N_STATES * N_ACTIONS)
    exact, V, Q, pi = exact_gradient(theta)
    rows = []
    for episodes in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        for kind, base in (("plain", None), ("baseline", V)):
            errs, coss, norms = [], [], []
            rng = np.random.default_rng(SEED + 1)
            for _ in range(400):
                g, _ = reinforce_gradient(theta, rng, episodes, baseline=base)
                errs.append(float(np.linalg.norm(g - exact)))
                coss.append(float(g @ exact / (np.linalg.norm(g) * np.linalg.norm(exact) + 1e-12)))
                norms.append(float(np.linalg.norm(g)))
            rows.append({"episodes": episodes, "kind": kind,
                         "error": r(float(np.mean(errs)), 5),
                         "cosine": r(float(np.mean(coss)), 5),
                         "norm": r(float(np.mean(norms)), 5),
                         "repeats": 400})
    # el sesgo: el promedio de muchas estimaciones tiene que caer sobre el exacto
    bias = {}
    for kind, base in (("plain", None), ("baseline", V)):
        rng = np.random.default_rng(SEED + 2)
        acc = np.zeros_like(exact)
        n = 4000
        for _ in range(n):
            g, _ = reinforce_gradient(theta, rng, 1, baseline=base)
            acc += g
        acc /= n
        bias[kind] = {"distance": r(float(np.linalg.norm(acc - exact)), 5),
                      "cosine": r(float(acc @ exact / (np.linalg.norm(acc)
                                                       * np.linalg.norm(exact))), 5),
                      "samples": n}
    return {"rows": rows, "exact_norm": r(float(np.linalg.norm(exact)), 5),
            "bias": bias, "theta": r(theta, 5), "V": r(V, 5),
            "policy": r(pi, 5), "J": r(float(V[START]), 5)}


def train_pg(seed, episodes=600, batch=8, lr=0.5, baseline=True, clip=None,
             entropy=0.0, epochs=1, record_every=10):
    """Un entrenamiento entero. `clip` activa el recorte de PPO sobre varias
    pasadas del mismo lote, que es la única razón por la que el recorte existe:
    sin repasar el lote, la razón vale uno y no hay nada que recortar."""
    rng = np.random.default_rng(seed)
    theta = np.zeros(N_STATES * N_ACTIONS)
    curve, clipped, kls = [], [], []
    for it in range(episodes // batch):
        V, _, _ = exact_values(theta)
        old = policy(theta).copy()
        trajs = [rollout(theta, rng) for _ in range(batch)]
        rets = []
        data = []
        for traj in trajs:
            G = 0.0
            returns = []
            for _, _, rew in reversed(traj):
                G = rew + GAMMA * G
                returns.append(G)
            returns.reverse()
            rets.append(returns[0])
            for t, (s, a, _) in enumerate(traj):
                data.append((s, a, returns[t] - (V[s] if baseline else 0.0), GAMMA ** t))
        for _ in range(epochs):
            pi = policy(theta)
            grad = np.zeros((N_STATES, N_ACTIONS))
            nclip = 0
            for s, a, adv, disc in data:
                ratio = pi[s, a] / max(old[s, a], 1e-12)
                if clip is not None:
                    if (adv > 0 and ratio > 1 + clip) or (adv < 0 and ratio < 1 - clip):
                        nclip += 1
                        continue
                    scale = ratio
                else:
                    scale = 1.0
                grad[s] += disc * adv * scale * (np.eye(N_ACTIONS)[a] - pi[s])
            if entropy > 0:
                for s in range(N_STATES):
                    logp = np.log(np.maximum(pi[s], 1e-12))
                    grad[s] += entropy * (-(pi[s] * (logp + 1)) + pi[s] * (pi[s] @ (logp + 1)))
            theta = theta + lr * grad.ravel() / len(trajs)
        new = policy(theta)
        kl = float((old * (np.log(np.maximum(old, 1e-12))
                           - np.log(np.maximum(new, 1e-12)))).sum(axis=1).mean())
        clipped.append(nclip / max(len(data), 1))
        kls.append(kl)
        if it % record_every == 0:
            V2, _, _ = exact_values(theta)
            curve.append(float(V2[START]))
    V, _, pi = exact_values(theta)
    ent = float(-(pi * np.log(np.maximum(pi, 1e-12))).sum(axis=1).mean())
    return {"J": float(V[START]), "curve": curve, "clipped": float(np.mean(clipped)),
            "kl": float(np.mean(kls)), "kl_max": float(np.max(kls)),
            "entropy": ent, "record_every": record_every, "batch": batch}


def stage_training():
    out = {}
    for name, kw in (
        ("reinforce", {"baseline": False, "epochs": 1}),
        ("baseline", {"baseline": True, "epochs": 1}),
        ("repeat", {"baseline": True, "epochs": 8}),
        ("ppo", {"baseline": True, "epochs": 8, "clip": 0.2}),
    ):
        runs = [train_pg(SEED * 3 + s, **kw) for s in range(10)]
        curves = np.array([run["curve"] for run in runs])
        out[name] = {
            "curve": r(curves.mean(axis=0), 5),
            "J": r(float(np.mean([run["J"] for run in runs])), 5),
            "J_spread": r(float(np.std([run["J"] for run in runs])), 5),
            "clipped": r(float(np.mean([run["clipped"] for run in runs])), 5),
            "kl": r(float(np.mean([run["kl"] for run in runs])), 6),
            "kl_max": r(float(np.max([run["kl_max"] for run in runs])), 6),
            "entropy": r(float(np.mean([run["entropy"] for run in runs])), 5),
            "record_every": runs[0]["record_every"], "batch": runs[0]["batch"],
            "seeds": 10,
        }
    return out


def stage_entropy():
    rows = []
    for beta in (0.0, 0.001, 0.01, 0.05, 0.1, 0.3):
        runs = [train_pg(SEED * 5 + s, entropy=beta, epochs=8, clip=0.2) for s in range(8)]
        rows.append({"beta": beta,
                     "J": r(float(np.mean([run["J"] for run in runs])), 5),
                     "spread": r(float(np.std([run["J"] for run in runs])), 5),
                     "entropy": r(float(np.mean([run["entropy"] for run in runs])), 5),
                     "seeds": 8})
    return {"rows": rows, "max_entropy": r(float(np.log(N_ACTIONS)), 5)}


def stage_optimal():
    """El techo: la mejor política determinista, por iteración de valor."""
    Q = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(20000):
        delta = 0.0
        for s in range(N_STATES):
            if s in TERMINAL:
                continue
            for a in range(N_ACTIONS):
                target = R[s, a] + GAMMA * Q[move(s, a)].max()
                delta = max(delta, abs(target - Q[s, a]))
                Q[s, a] = target
        if delta < 1e-12:
            break
    return {"J": r(float(Q[START].max()), 5), "Q": r(Q, 4),
            "V": r(Q.max(axis=1), 4), "size": SIZE, "gamma": GAMMA,
            "start": START, "goal": GOAL, "trap": TRAP,
            "step_cost": STEP_COST, "goal_reward": GOAL_REWARD,
            "trap_reward": TRAP_REWARD}


def main():
    t0 = time.time()
    print("1/4 el techo")
    opt = cached("optimal", stage_optimal)
    print("2/4 el gradiente exacto y su estimador")
    grad = cached("gradient", stage_gradient)
    print("3/4 cuatro entrenamientos")
    tr = cached("training", stage_training)
    print("4/4 el premio a la entropía")
    ent = cached("entropy", stage_entropy)

    data = {"meta": {"seed": SEED, "size": SIZE, "gamma": GAMMA,
                     "states": N_STATES, "actions": N_ACTIONS,
                     "note": "gradients and returns, never a clock"},
            "optimal": opt, "gradient": grad, "training": tr, "entropy": ent}
    f = OUT / "pg.json"
    f.write_text(json.dumps(data, indent=1))
    print(f"escrito {f} ({f.stat().st_size / 1024:.1f} kB) en {time.time() - t0:.1f}s")

    print("\ncomprobaciones:")
    print(f"  el mejor posible: {opt['J']}")
    print(f"  gradiente exacto, norma {grad['exact_norm']}")
    for kind, d in grad["bias"].items():
        print(f"    {kind}: media de {d['samples']} estimaciones a {d['distance']} "
              f"del exacto, coseno {d['cosine']}")
    for eps in (1, 16, 256):
        for kind in ("plain", "baseline"):
            row = [x for x in grad["rows"] if x["episodes"] == eps and x["kind"] == kind][0]
            print(f"    {eps} episodios {kind}: error {row['error']}, coseno {row['cosine']}")
    for name, d in tr.items():
        print(f"  {name}: J {d['J']} +- {d['J_spread']}, recorta {d['clipped']}, "
              f"KL medio {d['kl']} (máx {d['kl_max']})")
    for row in ent["rows"]:
        print(f"  entropía {row['beta']}: J {row['J']}, entropía final {row['entropy']}")


if __name__ == "__main__":
    main()
