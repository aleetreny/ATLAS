"""Todo lo que cita el artículo 5.2a (multi-armed bandits), medido.

El primer artículo del atlas donde una evaluación **cuesta**. Hasta aquí probar
una respuesta era gratis y su puntuación era exacta; aquí cada prueba se paga y
lo que devuelve es una muestra, así que la pregunta deja de ser dónde está el
óptimo y pasa a ser cuánto estás dispuesto a pagar por averiguarlo.

 1. Lo que pierde la avaricia pura, contado: sobre 4.000 tiradas, cuántas veces
    se queda enganchada al brazo equivocado y cuánto cuesta eso.
 2. El barrido de epsilon, con su mínimo medido y con la consecuencia que la
    teoría predice: a epsilon fijo el arrepentimiento crece en línea recta, y la
    pendiente sale de una cuenta de una línea.
 3. Explorar y comprometerse, con el número de rondas barrido y comparado con el
    que dice la fórmula.
 4. Epsilon decreciente, que sí da arrepentimiento logarítmico, comprobado
    ajustando la pendiente contra el logaritmo del horizonte.
 5. La cota de Lai y Robbins, calculada exactamente para esta instancia y
    dibujada al lado de lo medido.
 6. El barrido del hueco entre brazos, donde el problema más difícil no es el
    del hueco más pequeño.
 7. Una tirada de referencia con generador congruencial escrito a mano, que el
    navegador reproduce tirón a tirón.

Tarda unos dos minutos en cuatro núcleos. Etapas cacheadas en
~/.atlas_vision_data/bandit_cache/.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "bandits" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "bandit_cache"
CACHE.mkdir(parents=True, exist_ok=True)

SEED = 61
RUNS = 4000
HORIZON = 2000
# Cinco máquinas de Bernoulli. Los valores están escritos, no estimados, que es
# lo que permite calcular el arrepentimiento exacto en vez de aproximarlo.
ARMS = np.array([0.50, 0.55, 0.45, 0.60, 0.40])
BEST = int(ARMS.argmax())
GAPS = ARMS.max() - ARMS


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


def kl(p, q):
    """La divergencia entre dos monedas, que es lo que la cota usa de verdad."""
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def lai_robbins(T, arms=ARMS):
    """La cota de Lai y Robbins para esta instancia, exacta.

    No es una constante universal: sale de los brazos concretos, y por eso se
    puede dibujar al lado de lo medido en vez de citarse.
    """
    best = arms.max()
    total = 0.0
    for a in arms:
        if a < best:
            total += (best - a) / kl(a, best) * math.log(T)
    return total


# ---------------------------------------------------------------------------
# Los buscadores. Todos vectorizados sobre las tiradas: una fila por tirada.
# ---------------------------------------------------------------------------
def run_policy(policy, runs=RUNS, horizon=HORIZON, seed=SEED, **kw):
    """Devuelve el arrepentimiento acumulado medio y cuántas tiradas acabaron
    tirando del brazo equivocado."""
    rng = np.random.default_rng(seed)
    k = len(ARMS)
    counts = np.zeros((runs, k))
    sums = np.zeros((runs, k))
    regret = np.zeros((runs, horizon))
    total = np.zeros(runs)
    idx = np.arange(runs)
    for t in range(horizon):
        a = policy(t, counts, sums, rng, **kw)
        reward = (rng.random(runs) < ARMS[a]).astype(float)
        counts[idx, a] += 1
        sums[idx, a] += reward
        total += GAPS[a]
        regret[:, t] = total
    means = np.divide(sums, np.maximum(counts, 1))
    final_choice = means.argmax(axis=1)
    return {
        "regret_mean": regret.mean(axis=0),
        "regret_final": float(regret[:, -1].mean()),
        "regret_std": float(regret[:, -1].std()),
        "wrong_arm": int((final_choice != BEST).sum()),
        "share_best": float((counts[:, BEST] / horizon).mean()),
        "runs": runs,
    }


def greedy(t, counts, sums, rng):
    """Puro avaricioso: una tirada a cada brazo y luego siempre el mejor."""
    k = counts.shape[1]
    if t < k:
        return np.full(counts.shape[0], t)
    means = np.divide(sums, np.maximum(counts, 1))
    return means.argmax(axis=1)


def eps_greedy(t, counts, sums, rng, eps=0.1):
    k = counts.shape[1]
    if t < k:
        return np.full(counts.shape[0], t)
    means = np.divide(sums, np.maximum(counts, 1))
    a = means.argmax(axis=1)
    explore = rng.random(counts.shape[0]) < eps
    a = np.where(explore, rng.integers(0, k, size=counts.shape[0]), a)
    return a


def eps_decay(t, counts, sums, rng, c=1.0):
    k = counts.shape[1]
    if t < k:
        return np.full(counts.shape[0], t)
    eps = min(1.0, c * k / (t + 1))
    return eps_greedy(t, counts, sums, rng, eps=eps)


def etc(t, counts, sums, rng, m=20):
    """Explorar y comprometerse: m tiradas a cada brazo y luego el mejor."""
    k = counts.shape[1]
    if t < m * k:
        return np.full(counts.shape[0], t % k)
    means = np.divide(sums, np.maximum(counts, 1))
    return means.argmax(axis=1)


def stage_policies():
    out = {}
    for name, fn, kw in (
        ("greedy", greedy, {}),
        ("eps_10", eps_greedy, {"eps": 0.1}),
        ("eps_01", eps_greedy, {"eps": 0.01}),
        ("eps_decay", eps_decay, {"c": 1.0}),
        ("etc", etc, {"m": 30}),
    ):
        res = run_policy(fn, **kw)
        out[name] = {
            "regret": r(res["regret_mean"][::20], 4),
            "stride": 20,
            "final": r(res["regret_final"], 4),
            "std": r(res["regret_std"], 4),
            "wrong_arm": res["wrong_arm"],
            "share_best": r(res["share_best"], 4),
        }
    return out


def stage_epsilon_sweep():
    rows = []
    for eps in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0):
        res = run_policy(eps_greedy, eps=eps, runs=1500)
        # la predicción de una línea: a epsilon fijo, a la larga el
        # arrepentimiento por paso es epsilon por el hueco medio de un brazo
        # elegido al azar, más lo que cueste equivocarse de brazo
        slope = eps * GAPS.mean()
        rows.append({"eps": eps, "regret": r(res["regret_final"], 4),
                     "std": r(res["regret_std"], 4),
                     "wrong_arm": res["wrong_arm"], "of": 1500,
                     "predicted_slope": r(slope, 6),
                     "measured_slope": r((res["regret_mean"][-1]
                                          - res["regret_mean"][HORIZON // 2])
                                         / (HORIZON - HORIZON // 2), 6)})
    best = min(rows, key=lambda d: d["regret"])
    return {"rows": rows, "best_eps": best["eps"], "horizon": HORIZON}


def two_arm_run(policy, delta, runs=1500, horizon=HORIZON, seed=SEED, **kw):
    """Lo mismo que arriba pero sobre dos brazos, que es donde las fórmulas de
    libro están escritas: la de explorar y comprometerse es para dos."""
    global ARMS, BEST, GAPS
    saved = ARMS.copy()
    ARMS = np.array([0.5, 0.5 + delta])
    BEST = int(ARMS.argmax())
    GAPS = ARMS.max() - ARMS
    res = run_policy(policy, runs=runs, horizon=horizon, seed=seed, **kw)
    ARMS = saved
    BEST = int(ARMS.argmax())
    GAPS = ARMS.max() - ARMS
    return res


def etc_formula(delta, T):
    """m* = (4/delta^2) ln(T delta^2 / 4), que es lo que sale de minimizar la
    COTA del arrepentimiento, no el arrepentimiento."""
    inner = T * delta ** 2 / 4
    if inner <= 1:
        return 1
    return max(1, int(round(4 / delta ** 2 * math.log(inner))))


def stage_etc_sweep():
    """El número de rondas de exploración, barrido, contra el que dice la
    fórmula. Sobre dos brazos, que es para lo que la fórmula está escrita."""
    out = []
    grid = (1, 2, 5, 10, 20, 40, 80, 160, 320, 640)
    for delta in (0.05, 0.1, 0.2):
        rows = []
        # la m que dice la fórmula entra en la rejilla, o su casilla sale vacía
        # y la comparación que el artículo hace no se puede hacer
        ms = sorted(set(list(grid) + [etc_formula(delta, HORIZON)]))
        for m in ms:
            if 2 * m > HORIZON:
                continue
            res = two_arm_run(etc, delta, m=m)
            rows.append({"m": m, "regret": r(res["regret_final"], 4),
                         "std": r(res["regret_std"], 4),
                         "wrong_arm": res["wrong_arm"], "of": 1500})
        best = min(rows, key=lambda d: d["regret"])
        pred = etc_formula(delta, HORIZON)
        out.append({"delta": delta, "rows": rows, "best_m": best["m"],
                    "best_regret": best["regret"], "predicted_m": pred,
                    "regret_at_predicted": next(
                        (row["regret"] for row in rows if row["m"] == pred), None),
                    "ratio": r(pred / best["m"], 2)})
    return {"sweeps": out, "horizon": HORIZON, "grid": list(grid)}


def stage_horizon():
    """¿Crece el arrepentimiento como el logaritmo o como una recta? Se mide en
    cinco horizontes y se ajusta contra las dos formas."""
    rows = []
    for T in (250, 500, 1000, 2000, 4000, 8000):
        row = {"T": T, "bound": r(lai_robbins(T), 4)}
        for name, fn, kw in (("eps_10", eps_greedy, {"eps": 0.1}),
                             ("eps_decay", eps_decay, {"c": 1.0}),
                             ("greedy", greedy, {})):
            res = run_policy(fn, runs=1200, horizon=T, **kw)
            row[name] = r(res["regret_final"], 4)
        rows.append(row)
    fits = {}
    for name in ("eps_10", "eps_decay", "greedy"):
        y = np.array([row[name] for row in rows])
        T = np.array([row["T"] for row in rows], dtype=float)
        lin = float(np.polyfit(T, y, 1)[0])
        log = float(np.polyfit(np.log(T), y, 1)[0])
        r2 = lambda pred: 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        fits[name] = {
            "linear_slope": r(lin, 6),
            "linear_r2": r(r2(np.polyval(np.polyfit(T, y, 1), T)), 5),
            "log_slope": r(log, 4),
            "log_r2": r(r2(np.polyval(np.polyfit(np.log(T), y, 1), np.log(T))), 5),
        }
    return {"rows": rows, "fits": fits}


def stage_gap_sweep():
    """El hueco entre los dos brazos, barrido, para dos políticas distintas.

    Con epsilon fijo el impuesto de explorar se paga entero pase lo que pase, así
    que el arrepentimiento crece con el hueco y no hay nada más que ver. Con una
    política que deja de explorar cuando ya sabe, el problema fácil por arriba
    (hueco grande, se nota enseguida) y el fácil por abajo (hueco cero, da igual
    equivocarse) dejan el caso difícil en medio, que es lo que hay que enseñar.
    """
    deltas = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
    rows = []
    for delta in deltas:
        eps_res = two_arm_run(eps_greedy, delta, eps=0.1)
        m = etc_formula(delta, HORIZON)
        m = min(m, HORIZON // 2)
        etc_res = two_arm_run(etc, delta, m=m)
        rows.append({"delta": delta,
                     "eps": r(eps_res["regret_final"], 4),
                     "etc": r(etc_res["regret_final"], 4),
                     "etc_m": m,
                     "eps_wrong": eps_res["wrong_arm"],
                     "etc_wrong": etc_res["wrong_arm"], "of": 1500})
    worst_etc = max(rows, key=lambda d: d["etc"])
    worst_eps = max(rows, key=lambda d: d["eps"])
    return {"rows": rows, "worst_delta_etc": worst_etc["delta"],
            "worst_delta_eps": worst_eps["delta"],
            "predicted_delta": r(1 / math.sqrt(HORIZON), 4), "horizon": HORIZON}


def stage_decay_constant():
    """Epsilon decreciente solo da arrepentimiento logarítmico si la constante
    es lo bastante grande, y lo bastante grande depende del hueco, que es
    justamente lo que no se sabe. Se barre la constante y se ajusta la forma."""
    rows = []
    for c in (0.2, 1.0, 5.0, 25.0, 125.0):
        ys, Ts = [], (250, 500, 1000, 2000, 4000)
        for T in Ts:
            res = run_policy(eps_decay, runs=1000, horizon=T, c=c)
            ys.append(res["regret_final"])
        y = np.array(ys)
        Tf = np.array(Ts, dtype=float)
        r2 = lambda pred: 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        lin = np.polyfit(Tf, y, 1)
        log = np.polyfit(np.log(Tf), y, 1)
        rows.append({"c": c, "regret": r(ys, 4), "T": list(Ts),
                     "linear_r2": r(r2(np.polyval(lin, Tf)), 5),
                     "log_r2": r(r2(np.polyval(log, np.log(Tf))), 5),
                     "log_slope": r(float(log[0]), 4)})
    # la constante que la teoría pide para este hueco
    d = float(GAPS[GAPS > 0].min())
    # El resultado de libro es eps_t = min(1, c* K / (d^2 t)) con c* > 5. Aquí
    # el mando es c en eps_t = min(1, c K / t), así que la constante que la
    # teoría pide para este hueco es 5 / d^2, no 5.
    return {"rows": rows, "required_c": r(5 / d ** 2, 1),
            "smallest_gap": r(d, 4), "arms": len(ARMS)}


# ---------------------------------------------------------------------------
# La tirada de referencia, con el azar escrito a mano
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


REF = {"seed": 987654321, "pulls": 400, "eps": 0.1}


def ref_run(cfg):
    rng = LCG(cfg["seed"])
    k = len(ARMS)
    counts = [0] * k
    sums = [0.0] * k
    history = []
    regret = 0.0
    for t in range(cfg["pulls"]):
        if t < k:
            a = t
        elif rng.random() < cfg["eps"]:
            a = rng.randint(k)
        else:
            means = [sums[i] / counts[i] if counts[i] else 0.0 for i in range(k)]
            a = max(range(k), key=lambda i: (means[i], -i))
        reward = 1.0 if rng.random() < ARMS[a] else 0.0
        counts[a] += 1
        sums[a] += reward
        regret += float(GAPS[a])
        history.append({"t": t, "arm": a, "reward": reward, "regret": r(regret, 4)})
    return {"config": cfg, "arms": r(ARMS, 4), "history": history,
            "counts": counts, "regret": r(regret, 4),
            "means": r([sums[i] / counts[i] if counts[i] else 0.0 for i in range(k)], 6)}


def main():
    t0 = time.time()
    print("1/6 las cinco políticas")
    pol = cached("policies", stage_policies)
    print("2/6 barrido de epsilon")
    eps = cached("epsilon", stage_epsilon_sweep)
    print("3/6 explorar y comprometerse")
    e = cached("etc-v3", stage_etc_sweep)
    print("4/6 horizontes")
    hor = cached("horizon", stage_horizon)
    print("5/7 el hueco")
    gap = cached("gaps-v2", stage_gap_sweep)
    print("6/7 la constante del decaimiento")
    dec = cached("decay-v2", stage_decay_constant)
    print("7/7 la tirada de referencia")
    ref = cached("reference", lambda: ref_run(REF))

    data = {
        "meta": {"seed": SEED, "runs": RUNS, "horizon": HORIZON,
                 "arms": r(ARMS, 4), "best": BEST, "gaps": r(GAPS, 4),
                 "bound": r(lai_robbins(HORIZON), 4),
                 "note": "regret in units of expected reward, never a clock"},
        "policies": pol,
        "epsilon": eps,
        "etc": e,
        "horizon": hor,
        "gaps": gap,
        "decay": dec,
        "reference": ref,
    }
    f = OUT / "bandits.json"
    f.write_text(json.dumps(data, indent=1))
    print(f"escrito {f} ({f.stat().st_size / 1024:.1f} kB) en {time.time() - t0:.1f}s")

    print("\ncomprobaciones:")
    for name, d in pol.items():
        print(f"  {name}: arrepentimiento {d['final']} +- {d['std']}, "
              f"brazo equivocado en {d['wrong_arm']} de {RUNS}")
    print(f"  cota de Lai y Robbins a T={HORIZON}: {lai_robbins(HORIZON):.4f}")
    print(f"  mejor epsilon medido: {eps['best_eps']}")
    for s in e["sweeps"]:
        print(f"  ETC hueco {s['delta']}: mejor m {s['best_m']} ({s['best_regret']}), "
              f"la fórmula dice {s['predicted_m']} ({s['regret_at_predicted']})")
    for row in dec["rows"]:
        print(f"  decaimiento c={row['c']}: r2 recta {row['linear_r2']}, "
              f"r2 log {row['log_r2']}")
    for name, d in hor["fits"].items():
        print(f"  {name}: r2 recta {d['linear_r2']}, r2 logaritmo {d['log_r2']}")
    print(f"  hueco más caro: epsilon en {gap['worst_delta_eps']}, ETC en "
          f"{gap['worst_delta_etc']}, la teoría dice {gap['predicted_delta']}")


if __name__ == "__main__":
    main()
