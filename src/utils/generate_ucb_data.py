"""Todo lo que cita el artículo 5.2b (Thompson y UCB), medido.

El artículo anterior dejó cinco políticas que exploran a ciegas: o al azar, o
según un calendario decidido antes de empezar. Las dos de este saben lo segura
que está cada estimación suya, y eso se puede comprobar en vez de contarse.

 1. La cota de UCB, con su cobertura medida: cuántas veces la media verdadera
    se sale del intervalo que la política se cree, contra lo que la desigualdad
    promete.
 2. La posterior de Thompson, con su calibración medida: qué fracción de las
    veces la verdad cae dentro del intervalo creíble del 90%.
 3. Arrepentimiento contra el horizonte, en escala logarítmica, y la pendiente
    ajustada contra la constante exacta de Lai y Robbins de este problema, que
    es lo que el artículo anterior dejó pendiente.
 4. La constante del bonus de UCB, barrida, contra la que dice la demostración.
 5. Máquinas que cambian a mitad de partida, donde las dos fallan igual, y la
    ventana deslizante que lo arregla, medido.
 6. Una tirada de referencia de cada política, reproducible en el navegador.

Las cinco máquinas son las del artículo anterior, leídas de su fichero.

Tarda unos cinco minutos en cuatro núcleos. Etapas cacheadas en
~/.atlas_vision_data/ucb_cache/.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "thompson-ucb" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "ucb_cache"
CACHE.mkdir(parents=True, exist_ok=True)

SEED = 62
RUNS = 2000
HORIZON = 10000

_prev = json.loads((ROOT / "bandits" / "data" / "bandits.json").read_text())
ARMS = np.array(_prev["meta"]["arms"])
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
    e = 1e-12
    p = min(max(p, e), 1 - e)
    q = min(max(q, e), 1 - e)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


LR_CONST = sum((ARMS.max() - a) / kl(a, ARMS.max()) for a in ARMS if a < ARMS.max())


# ---------------------------------------------------------------------------
# Las políticas, vectorizadas sobre las tiradas
# ---------------------------------------------------------------------------
def simulate(kind, runs=RUNS, horizon=HORIZON, seed=SEED, c=2.0, eps=0.1,
             arms=None, window=None, record=None):
    """Una fila por tirada. `record` pide instantáneas del estado interno."""
    arms = ARMS if arms is None else arms
    k = arms.shape[-1]        # con maquinas que cambian, arms es (T, k)
    rng = np.random.default_rng(seed)
    counts = np.zeros((runs, k))
    sums = np.zeros((runs, k))
    idx = np.arange(runs)
    total = np.zeros(runs)
    curve = np.zeros(horizon)
    snaps = {}
    hist = [] if window else None
    best_mean = arms.max() if arms.ndim == 1 else None
    for t in range(horizon):
        if arms.ndim == 2:                       # máquinas que cambian
            now = arms[min(t, arms.shape[0] - 1)]
            best_mean = now.max()
        else:
            now = arms
        if t < k:
            a = np.full(runs, t)
        elif kind == "ucb":
            means = sums / np.maximum(counts, 1)
            bonus = np.sqrt(c * np.log(t + 1) / np.maximum(counts, 1))
            a = (means + bonus).argmax(axis=1)
        elif kind == "thompson":
            draw = rng.beta(sums + 1, counts - sums + 1)
            a = draw.argmax(axis=1)
        elif kind == "eps":
            means = sums / np.maximum(counts, 1)
            a = means.argmax(axis=1)
            explore = rng.random(runs) < eps
            a = np.where(explore, rng.integers(0, k, size=runs), a)
        else:
            raise ValueError(kind)
        reward = (rng.random(runs) < now[a]).astype(float)
        counts[idx, a] += 1
        sums[idx, a] += reward
        if window:
            hist.append((a.copy(), reward))
            if len(hist) > window:
                old_a, old_r = hist.pop(0)
                counts[idx, old_a] -= 1
                sums[idx, old_a] -= old_r
        total += best_mean - now[a]
        curve[t] = total.mean()
        if record and t + 1 in record:
            means = sums / np.maximum(counts, 1)
            snaps[t + 1] = {
                "counts": r(counts[0], 0), "means": r(means[0], 4),
                "bonus": r(np.sqrt(c * np.log(t + 1) / np.maximum(counts[0], 1)), 4),
                "alpha": r(sums[0] + 1, 0), "beta": r(counts[0] - sums[0] + 1, 0),
            }
    means = sums / np.maximum(counts, 1)
    return {"curve": curve, "final": float(total.mean()), "std": float(total.std()),
            "counts": counts, "means": means, "snaps": snaps}


def stage_regret():
    out = {}
    for name, kw in (("ucb", {"kind": "ucb"}),
                     ("thompson", {"kind": "thompson"}),
                     ("eps", {"kind": "eps", "eps": 0.1})):
        res = simulate(**kw)
        out[name] = {"curve": r(res["curve"][::50], 4), "stride": 50,
                     "final": r(res["final"], 4), "std": r(res["std"], 4),
                     "share_best": r(float((res["counts"][:, BEST] / HORIZON).mean()), 4),
                     "wrong_arm": int((res["means"].argmax(axis=1) != BEST).sum())}
    return out


def stage_slope():
    """La pendiente contra el logaritmo del horizonte, que es la cantidad que la
    cota de Lai y Robbins acota. Se mide con la segunda mitad de la curva, donde
    el régimen asintótico ya ha empezado."""
    rows = []
    base = stage_regret()
    for name in ("ucb", "thompson", "eps"):
        curve = np.array(base[name]["curve"])
        t = np.arange(1, len(curve) + 1) * base[name]["stride"]
        half = len(curve) // 2
        slope = float(np.polyfit(np.log(t[half:]), curve[half:], 1)[0])
        rows.append({"policy": name, "slope": r(slope, 3),
                     "ratio": r(slope / LR_CONST, 3)})
    return {"rows": rows, "bound_constant": r(LR_CONST, 4),
            "bound_at_horizon": r(LR_CONST * math.log(HORIZON), 3),
            "horizon": HORIZON}


def stage_coverage():
    """¿Es la cota una cota? Se cuenta cuántas veces la media verdadera se sale.

    La desigualdad de Hoeffding con el bonus de UCB promete que la probabilidad
    de que la media verdadera quede por encima de media + bonus es como mucho
    t^-c/2 por brazo, así que se puede comparar contra un número y no contra una
    sensación.
    """
    runs, horizon, k = 4000, 500, len(ARMS)
    rng = np.random.default_rng(SEED + 1)
    for c in (2.0,):
        counts = np.zeros((runs, k))
        sums = np.zeros((runs, k))
        idx = np.arange(runs)
        above, total, by_t = 0, 0, []
        for t in range(horizon):
            if t < k:
                a = np.full(runs, t)
            else:
                means = sums / np.maximum(counts, 1)
                bonus = np.sqrt(c * np.log(t + 1) / np.maximum(counts, 1))
                a = (means + bonus).argmax(axis=1)
                # la comprobación: para cada brazo con al menos una tirada
                ok = means + bonus
                miss = (ok < ARMS[None, :]) & (counts > 0)
                above += int(miss.sum())
                total += int((counts > 0).sum())
                if t % 50 == 0:
                    # la promesa es POR BRAZO, y la medida de al lado también lo
                    # es: multiplicarla por k compararía dos cosas distintas
                    by_t.append([t, r(float(miss.sum() / max((counts > 0).sum(), 1)), 6),
                                 r(min(1.0, (t + 1) ** (-c / 2)), 6)])
            reward = (rng.random(runs) < ARMS[a]).astype(float)
            counts[idx, a] += 1
            sums[idx, a] += reward
    return {"c": 2.0, "runs": runs, "horizon": horizon,
            "share_outside": r(above / max(total, 1), 6),
            "promised": r(float(np.mean([row[2] for row in by_t])), 6),
            "by_t": by_t}


def stage_calibration():
    """Lo mismo para Thompson: ¿cae la verdad dentro del intervalo creíble tan a
    menudo como el intervalo promete? Se mide sobre las posteriores que la
    política tiene de verdad a lo largo de una tirada."""
    from scipy.stats import beta as beta_dist
    runs, horizon = 800, 2000
    rng = np.random.default_rng(SEED + 2)
    k = len(ARMS)
    counts = np.zeros((runs, k))
    sums = np.zeros((runs, k))
    idx = np.arange(runs)
    inside, total = 0, 0
    per_arm_in = np.zeros(k)
    per_arm_n = np.zeros(k)
    for t in range(horizon):
        if t < k:
            a = np.full(runs, t)
        else:
            draw = rng.beta(sums + 1, counts - sums + 1)
            a = draw.argmax(axis=1)
        reward = (rng.random(runs) < ARMS[a]).astype(float)
        counts[idx, a] += 1
        sums[idx, a] += reward
        if t in (99, 499, 999, 1999):
            lo = beta_dist.ppf(0.05, sums + 1, counts - sums + 1)
            hi = beta_dist.ppf(0.95, sums + 1, counts - sums + 1)
            ins = (lo <= ARMS[None, :]) & (ARMS[None, :] <= hi)
            inside += int(ins.sum())
            total += ins.size
            per_arm_in += ins.sum(axis=0)
            per_arm_n += runs
    return {"runs": runs, "horizon": horizon, "level": 0.9,
            "inside": r(inside / total, 4),
            "per_arm": r(per_arm_in / per_arm_n, 4),
            "checked_at": [100, 500, 1000, 2000]}


def stage_constant():
    rows = []
    for c in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        res = simulate(kind="ucb", runs=800, horizon=4000, c=c)
        rows.append({"c": c, "regret": r(res["final"], 3), "std": r(res["std"], 3),
                     "share_best": r(float((res["counts"][:, BEST] / 4000).mean()), 4)})
    best = min(rows, key=lambda d: d["regret"])
    return {"rows": rows, "best_c": best["c"], "proved_c": 2.0, "horizon": 4000}


def stage_drift():
    """Las máquinas cambian a la mitad. Ninguna de las dos políticas lo nota, y
    la razón es la misma para las dos: los conteos viejos pesan tanto como los
    nuevos. Una ventana deslizante lo arregla y se mide cuánto."""
    horizon = 6000
    arms = np.zeros((horizon, len(ARMS)))
    arms[:horizon // 2] = ARMS
    swapped = ARMS.copy()
    swapped[[BEST, int(ARMS.argmin())]] = swapped[[int(ARMS.argmin()), BEST]]
    arms[horizon // 2:] = swapped
    rows = []
    for name, kw in (("ucb", {"kind": "ucb"}),
                     ("thompson", {"kind": "thompson"}),
                     ("ucb_window", {"kind": "ucb", "window": 500}),
                     ("thompson_window", {"kind": "thompson", "window": 500})):
        res = simulate(runs=600, horizon=horizon, arms=arms, **kw)
        curve = res["curve"]
        rows.append({"policy": name, "final": r(res["final"], 3),
                     "before": r(float(curve[horizon // 2 - 1]), 3),
                     "after": r(float(curve[-1] - curve[horizon // 2 - 1]), 3),
                     "curve": r(curve[::30], 3), "stride": 30})
    return {"rows": rows, "horizon": horizon, "switch": horizon // 2,
            "window": 500, "arms_before": r(ARMS, 4), "arms_after": r(swapped, 4)}


def stage_snapshots():
    """El estado interno de las dos políticas en cuatro momentos, para dibujarlo.

    Una tirada suelta puede ser afortunada o desafortunada, y la de la primera
    semilla que se probó lo era: daba 439 tirones a la segunda máquina y 206 a la
    mejor, así que el dibujo habría contradicho al texto. La tirada que se
    publica se **elige**, entre cuarenta semillas, por ser la que más se acerca
    a la mediana del reparto de tirones, y eso se dice en la página.
    """
    marks = [10, 40, 200, 1000]
    cands = []
    for s in range(40):
        ucb = simulate(kind="ucb", runs=1, horizon=1001, seed=SEED + 100 + s,
                       record=set(marks))
        th = simulate(kind="thompson", runs=1, horizon=1001, seed=SEED + 100 + s,
                      record=set(marks))
        share = th["snaps"][1000]["counts"][BEST] / 1000
        cands.append((share, s, ucb, th))
    shares = sorted(c[0] for c in cands)
    med = shares[len(shares) // 2]
    share, s, ucb, th = min(cands, key=lambda c: abs(c[0] - med))
    return {"marks": marks, "ucb": ucb["snaps"], "thompson": th["snaps"],
            "arms": r(ARMS, 4), "seed": SEED + 100 + s,
            "share_best": r(share, 4), "median_share": r(med, 4),
            "candidates": len(cands),
            "chosen_by": "closest to the median share of pulls on the best machine"}


def main():
    t0 = time.time()
    print("1/6 arrepentimiento")
    reg = cached("regret", stage_regret)
    print("2/6 pendiente contra la cota")
    slope = cached("slope", stage_slope)
    print("3/6 cobertura de la cota")
    cov = cached("coverage-v2", stage_coverage)
    print("4/6 calibración de la posterior")
    cal = cached("calibration", stage_calibration)
    print("5/6 la constante del bonus")
    con = cached("constant", stage_constant)
    print("6/6 máquinas que cambian, e instantáneas")
    dr = cached("drift", stage_drift)
    snap = cached("snapshots-v2", stage_snapshots)

    data = {
        "meta": {"seed": SEED, "runs": RUNS, "horizon": HORIZON,
                 "arms": r(ARMS, 4), "best": BEST, "gaps": r(GAPS, 4),
                 "shared_with": "bandits/data/bandits.json",
                 "bound_constant": r(LR_CONST, 4),
                 "note": "regret is exact, computed from the written down means"},
        "regret": reg, "slope": slope, "coverage": cov, "calibration": cal,
        "constant": con, "drift": dr, "snapshots": snap,
    }
    f = OUT / "ucb.json"
    f.write_text(json.dumps(data, indent=1))
    print(f"escrito {f} ({f.stat().st_size / 1024:.1f} kB) en {time.time() - t0:.1f}s")

    print("\ncomprobaciones:")
    for name, d in reg.items():
        print(f"  {name}: {d['final']} +- {d['std']}, mejor brazo {d['share_best']}")
    print(f"  constante de la cota: {LR_CONST:.4f}, a T={HORIZON} vale "
          f"{slope['bound_at_horizon']}")
    for row in slope["rows"]:
        print(f"    {row['policy']}: pendiente {row['slope']} ({row['ratio']} veces la cota)")
    print(f"  cobertura: fuera {cov['share_outside']}, prometido {cov['promised']}")
    print(f"  calibración: dentro {cal['inside']} de un {cal['level']}")
    print(f"  mejor constante medida: {con['best_c']}, demostrada {con['proved_c']}")
    for row in dr["rows"]:
        print(f"  {row['policy']}: antes {row['before']}, después {row['after']}")


if __name__ == "__main__":
    main()
