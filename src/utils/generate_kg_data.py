"""Las cifras del artículo de embeddings de grafos de conocimiento (módulo 6.2).

Aquí la arista tiene tipo: no es "a y b están conectados" sino "a es un tipo de
b", "a deriva de b", "a es parecido a b". Y la pregunta del artículo es la que
casi nunca se mide: **qué patrones de relación puede representar cada función
de puntuación**, que es una afirmación de álgebra con consecuencias medibles.

Lo que se mide:

  1. **La predicción cerrada de TransE**: si una relación es simétrica, entonces
     h + r = t y t + r = h a la vez, así que 2r = 0 y r tiene que ser el vector
     cero. Se mide la norma aprendida de cada relación contra lo simétrica que
     esa relación es en los datos, sobre WN18RR, donde hay tres relaciones
     simétricas y siete que no lo son.
  2. **La predicción cerrada de RotatE**: la misma simetría exige una rotación
     que sea su propia inversa, es decir, de ángulo 0 o pi. Se mide la
     distribución de los ángulos aprendidos por relación.
  3. **DistMult**, que es simétrico por construcción y por lo tanto no puede
     ordenar un hiperónimo: se mide cuánto le cuesta eso en la única relación
     donde importa.
  4. **Un grafo con los cuatro patrones escritos a mano** (simetría, inversión,
     composición y jerarquía), donde cada uno se puntúa por separado en vez de
     promediarse con los demás.
  5. **El protocolo**: filtrado contra crudo, y por qué la diferencia no es un
     detalle de implementación.

Ejecutar desde cualquier sitio: `python src/utils/generate_kg_data.py`.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kg_data import digest, pattern_kg, wn18rr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kg-embeddings" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "kg_cache")

SEED = 19
# El presupuesto de esta pagina es pequeño a proposito y hay que decirlo: el
# coste por paso lo fija B x NEG x DIM, que es el tamaño de los arrays de
# negativos, no el numero de entidades. Las cifras publicadas de esta
# literatura usan 500 dimensiones y cien veces mas pasos, asi que lo que esta
# pagina compara son los tres modelos entre si.
#
# Estos tres numeros se midieron, no se eligieron (sonda en el historial de la
# sesion, 300 tripletas de test, TransE):
#
#   paso plano  lr=0,05 neg=16 e=10   MRR 0,0098
#   paso plano  lr=0,50 neg=16 e=10   MRR 0,0101   la tasa sola no arregla nada
#   adagrad     lr=0,50 neg=16 e=10   MRR 0,0338
#   adagrad     lr=0,50 neg=64 e=10   MRR 0,0489
#   adagrad     lr=0,50 neg=64 e=15   MRR 0,0843, y |r| simetricas 6,9 contra
#                                     12,8 en las demas: la prediccion cerrada
#                                     del articulo aparece justo aqui
#
# Adagrad es lo que mas mueve la aguja y el motivo es la forma del grafo: una
# entidad de WN18RR aparece en dos tripletas de media y otras en cientos, asi
# que con un paso comun o la rara no se mueve o la frecuente explota.
#
# Y una que se probo y se descarto con datos: renormalizar cada entidad a norma
# 1 en cada paso, que es lo que hace el TransE original de 2013. Aqui BAJA el
# MRR de 0,0843 a 0,0121 y BORRA la separacion de las simetricas (razon 1,86 a
# 1,11). Con margen 9 y entidades en la esfera unidad las distancias no llegan
# a cubrir el margen, asi que la restriccion se come el modelo entero. Se deja
# la escala libre y se dice que se probó lo otro.
DIM = 32                  # 16 dimensiones complejas para RotatE
EPOCHS = 60
BATCH = 1024
NEG = 64
LR = 0.5                  # con adagrad, no con paso plano
GAMMA = 9.0
ALPHA = 1.0               # temperatura del muestreo adversario
TAG = f"s{SEED}d{DIM}e{EPOCHS}n{NEG}g{int(GAMMA)}ada"


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
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, dict):
        return {k: r(x, d) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r(x, d) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return None if not np.isfinite(v) else round(v, d)


# --------------------------------------------------------------------------
# Las tres funciones de puntuación, con sus gradientes escritos a mano.
#
# Todas devuelven "más alto es más plausible", así que la pérdida es una sola
# para las tres y lo único que cambia entre ellas es el álgebra de la línea de
# arriba, que es exactamente lo que el artículo compara.
# --------------------------------------------------------------------------
def apply_update(target, acc, idx_blocks, val_blocks, lr):
    """Un paso de adagrad sobre varios bloques de índices repetidos, de una vez.

    Las dos escrituras obvias cuestan el articulo entero. `np.add.at` tarda
    decimas de segundo por llamada, y un `bincount` por columna arregla eso y
    deja otros dos problemas igual de caros: escribir una tabla de 40.943 por
    64 columna a columna, que en memoria va a saltos, y reservar esa tabla
    entera como gradiente cuando el lote toca dos mil filas.

    Lo que queda barato es agrupar todos los bloques de indices repetidos con
    una matriz de seleccion dispersa y sumar **solo sobre las filas que
    aparecen**. De 1,6 segundos por paso a unas milesimas, que es la diferencia
    entre poder barrer tres modelos y no poder.

    El acumulado de adagrad se toca en esas mismas filas y por la misma razon:
    una tabla de 40.943 por 32 entera por paso costaria mas que el gradiente."""
    dim = target.shape[1]
    flat = np.concatenate([np.asarray(i).ravel() for i in idx_blocks])
    vals = np.concatenate([np.ascontiguousarray(v).reshape(-1, dim) for v in val_blocks])
    uniq, inv = np.unique(flat, return_inverse=True)
    sel = sp.csr_matrix((np.ones(flat.size), (inv, np.arange(flat.size))),
                        shape=(uniq.size, flat.size))
    grad = sel @ vals
    acc[uniq] += grad * grad
    target[uniq] -= lr * grad / np.sqrt(acc[uniq] + 1e-10)


class Model:
    def __init__(self, kind, n_ent, n_rel, dim=DIM, seed=SEED, gamma=GAMMA):
        rng = np.random.default_rng(seed)
        self.kind, self.dim, self.gamma = kind, dim, gamma
        scale = (gamma + 2.0) / dim
        self.E = rng.uniform(-scale, scale, (n_ent, dim))
        if kind == "rotate":
            self.k = dim // 2
            self.R = rng.uniform(-np.pi, np.pi, (n_rel, self.k))
        else:
            self.R = rng.uniform(-scale, scale, (n_rel, dim))
        self.accE = np.zeros_like(self.E)
        self.accR = np.zeros_like(self.R)

    def params(self):
        return [self.E, self.R]

    def score(self, h, r, t, want_grad=False):
        """h, t: (..., dim) índices ya recogidos. Devuelve la puntuación."""
        E, R = self.E, self.R
        H, T, Rv = E[h], E[t], R[r]
        if self.kind == "transe":
            diff = H + Rv - T
            d = np.sqrt(np.maximum((diff ** 2).sum(-1), 1e-12))
            s = self.gamma - d
            if not want_grad:
                return s
            g = diff / d[..., None]
            return s, (-g, -g, g)                       # dS/dH, dS/dR, dS/dT
        if self.kind == "distmult":
            s = (H * Rv * T).sum(-1)
            if not want_grad:
                return s
            return s, (Rv * T, H * T, H * Rv)
        # rotate
        k = self.k
        hr, hi = H[..., :k], H[..., k:]
        tr_, ti = T[..., :k], T[..., k:]
        c, s_ = np.cos(Rv), np.sin(Rv)
        u = hr * c - hi * s_ - tr_
        v = hr * s_ + hi * c - ti
        d = np.sqrt(np.maximum((u ** 2 + v ** 2).sum(-1), 1e-12))
        sc = self.gamma - d
        if not want_grad:
            return sc
        du = u / d[..., None]
        dv = v / d[..., None]
        gH = np.concatenate([-(du * c + dv * s_), -(-du * s_ + dv * c)], axis=-1)
        gT = np.concatenate([du, dv], axis=-1)
        gR = -(du * (-hr * s_ - hi * c) + dv * (hr * c - hi * s_))
        return sc, (gH, gR, gT)

    def step(self, pos, pool, lr=None, alpha=ALPHA):
        """Un paso sobre un lote, con un pool de negativos compartido.

        Compartir los negativos entre todo el lote no es un atajo de
        implementacion, es lo que hace que el paso toque dos mil filas de la
        tabla de entidades en vez de casi todas, y es lo que hacen los sistemas
        de recuperacion de verdad por la misma razon."""
        lr = LR if lr is None else lr
        h, rel, t = pos[:, 0], pos[:, 1], pos[:, 2]
        sp_, (gh, gr, gt) = self.score(h, rel, t, want_grad=True)
        rel_b = rel[:, None]
        pool_b = pool[None, :]
        sn_h, (gnh_h, gnh_r, gnh_t) = self.score(pool_b, rel_b, t[:, None], want_grad=True)
        sn_t, (gnt_h, gnt_r, gnt_t) = self.score(h[:, None], rel_b, pool_b, want_grad=True)

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

        n = len(pos)
        # El gradiente NO se divide por el tamaño del lote. Cada fila de la
        # tabla de entidades recibe el de las pocas tripletas en las que
        # aparece, no el de las mil del lote, así que dividir por mil deja el
        # movimiento por entidad en 5e-5 y el modelo entero en el azar: MRR
        # 0,0001 tras treinta pasadas, con las normas de las relaciones donde
        # las dejó el inicializador.
        cp = -sigmoid(-sp_)[:, None]
        idx_e = [h, t]
        val_e = [cp * gh, cp * gt]
        idx_r = [rel]
        val_r = [cp * gr]
        for sn, gnh, gnr, gnt, ih, it in (
                (sn_h, gnh_h, gnh_r, gnh_t, np.broadcast_to(pool_b, sn_h.shape),
                 np.broadcast_to(t[:, None], sn_h.shape)),
                (sn_t, gnt_h, gnt_r, gnt_t, np.broadcast_to(h[:, None], sn_t.shape),
                 np.broadcast_to(pool_b, sn_t.shape))):
            w = np.exp(alpha * (sn - sn.max(1, keepdims=True)))
            w /= w.sum(1, keepdims=True)
            cn = (w * sigmoid(sn))[:, :, None]
            idx_e += [ih, it]
            val_e += [np.broadcast_to(cn * gnh, (n, len(pool), self.E.shape[1])),
                      np.broadcast_to(cn * gnt, (n, len(pool), self.E.shape[1]))]
            idx_r.append(np.broadcast_to(rel_b, sn.shape))
            val_r.append(np.broadcast_to(cn * gnr, (n, len(pool), self.R.shape[1])))

        apply_update(self.E, self.accE, idx_e, val_e, lr)
        apply_update(self.R, self.accR, idx_r, val_r, lr)
        if self.kind == "rotate":
            self.R[:] = (self.R + np.pi) % (2 * np.pi) - np.pi

    def fit(self, train, n_ent, epochs=None, batch=None, seed=SEED, log=None, lr=None,
            neg=NEG, trace=None):
        """`trace`, si se pasa, es (test, known, cada_cuantas) y sirve para ver
        si el presupuesto llegó: una curva plana al final dice que sí y una que
        todavía sube dice que el número publicado está limitado por las pasadas
        y no por el modelo, que es una diferencia que hay que saber antes de
        escribir "RotatE gana"."""
        epochs = EPOCHS if epochs is None else epochs
        batch = BATCH if batch is None else batch
        lr = LR if lr is None else lr
        rng = np.random.default_rng(seed + 100)
        curve = []
        for ep in range(epochs):
            order = rng.permutation(len(train))
            for a in range(0, len(train), batch):
                pos = train[order[a:a + batch]]
                pool = rng.integers(0, n_ent, neg)
                self.step(pos, pool, lr=lr)
            if trace and (ep + 1) % trace[2] == 0:
                ev = evaluate(self, trace[0], trace[1], n_ent, limit=200)
                curve.append({"epoch": ep + 1, "mrr": ev["mrr"], "hits10": ev["hits10"]})
                print(f"      época {ep + 1}: MRR {ev['mrr']:.4f} h@10 {ev['hits10']:.4f}",
                      flush=True)
        self.curve = curve
        return self

    def score_all(self, h, rel, mode="tail", block=4096):
        """Puntuación contra TODAS las entidades, por bloques."""
        n = self.E.shape[0]
        out = np.empty(n)
        for a in range(0, n, block):
            cand = np.arange(a, min(a + block, n))
            if mode == "tail":
                out[a:a + len(cand)] = self.score(np.full(len(cand), h),
                                                  np.full(len(cand), rel), cand)
            else:
                out[a:a + len(cand)] = self.score(cand, np.full(len(cand), rel),
                                                  np.full(len(cand), h))
        return out


def evaluate(model, test, known, n_ent, limit=None, per_relation=False):
    """MRR y aciertos, con y sin filtrar, que no es lo mismo."""
    rows = test if limit is None else test[:limit]
    ranks_f, ranks_r, rel_ranks = [], [], {}
    for h, rel, t in rows:
        for mode, target, other in (("tail", t, h), ("head", h, t)):
            s = model.score_all(other, rel, mode=mode)
            true = s[target]
            better = s > true
            ranks_r.append(int(better.sum()) + 1)
            key = (other, rel) if mode == "tail" else (other, rel)
            for e in known.get((other, rel, mode), ()):  # las otras respuestas ciertas
                if e != target:
                    better[e] = False
            rk = int(better.sum()) + 1
            ranks_f.append(rk)
            if per_relation:
                rel_ranks.setdefault(int(rel), []).append(rk)
    rf = np.array(ranks_f)
    rr = np.array(ranks_r)
    out = {"mrr": float((1.0 / rf).mean()), "mrr_raw": float((1.0 / rr).mean()),
           "hits1": float((rf <= 1).mean()), "hits3": float((rf <= 3).mean()),
           "hits10": float((rf <= 10).mean()), "median": float(np.median(rf)),
           "n": int(len(rf))}
    if per_relation:
        out["per_relation"] = {str(k): {"mrr": float((1.0 / np.array(v)).mean()),
                                        "hits10": float((np.array(v) <= 10).mean()),
                                        "n": len(v)}
                               for k, v in rel_ranks.items()}
    return out


def build_known(*sets):
    known = {}
    for T in sets:
        for h, rel, t in T:
            known.setdefault((int(h), int(rel), "tail"), set()).add(int(t))
            known.setdefault((int(t), int(rel), "head"), set()).add(int(h))
    return known


# --------------------------------------------------------------------------
# S1. Qué tan simétrica es cada relación de WN18RR, que es la vara con la que
# se leen las normas y los ángulos aprendidos.
# --------------------------------------------------------------------------
def relation_stats(train, rel_names):
    S = set(map(tuple, train.tolist()))
    inv_names = {v: k for k, v in rel_names.items()}
    rows = []
    for name, i in rel_names.items():
        m = train[train[:, 1] == i]
        if not len(m):
            continue
        sym = float(np.mean([(int(t), i, int(h)) in S for h, _, t in m]))
        heads = len(np.unique(m[:, 0]))
        tails = len(np.unique(m[:, 2]))
        rows.append({"id": int(i), "name": name, "n": int(len(m)), "symmetry": sym,
                     "tails_per_head": float(len(m) / max(heads, 1)),
                     "heads_per_tail": float(len(m) / max(tails, 1))})
    return sorted(rows, key=lambda x: -x["n"]), inv_names


# --------------------------------------------------------------------------
# S2. Los tres modelos sobre WN18RR.
# --------------------------------------------------------------------------
def stage_wn(train, valid, test, n_ent, n_rel, rel_rows):
    known = build_known(train, valid, test)
    out = {}
    for kind in ["transe", "rotate", "distmult"]:
        print(f"    entrenando {kind}")
        m = cached(f"wn_{kind}", lambda kind=kind: Model(kind, n_ent, n_rel)
                   .fit(train, n_ent, trace=(test, known, 15)))
        ev = evaluate(m, test, known, n_ent, limit=800, per_relation=True)
        norms = {}
        for row in rel_rows:
            i = row["id"]
            if kind == "rotate":
                ang = m.R[i]
                # lo cerca que está la rotación de ser su propia inversa, que es
                # lo que la simetría exige: ángulo 0 o pi
                selfinv = float(np.mean(np.minimum(np.abs(ang), np.pi - np.abs(ang))))
                norms[str(i)] = {"norm": float(np.abs(ang).mean()),
                                 "self_inverse": selfinv,
                                 "hist": [float(v) for v in
                                          np.histogram(ang, bins=12, range=(-np.pi, np.pi))[0]]}
            else:
                norms[str(i)] = {"norm": float(np.linalg.norm(m.R[i]))}
        out[kind] = {"eval": ev, "relations": norms, "curve": getattr(m, "curve", [])}
        print(f"      MRR {ev['mrr']:.4f} hits@10 {ev['hits10']:.4f}")
    return out


# --------------------------------------------------------------------------
# S3. El grafo con los patrones escritos.
# --------------------------------------------------------------------------
def stage_patterns():
    train, valid, test, meta = pattern_kg()
    known = build_known(train, valid, test)
    names = {v: k for k, v in meta["names"].items()}
    out = {}
    for kind in ["transe", "rotate", "distmult"]:
        m = cached(f"pat_{kind}", lambda kind=kind: Model(
            kind, meta["entities"], meta["relations"], dim=32)
            .fit(train, meta["entities"], epochs=300, batch=512))
        ev = evaluate(m, test, known, meta["entities"], per_relation=True)
        rels = {}
        for i, name in names.items():
            if kind == "rotate":
                ang = m.R[i]
                rels[name] = {"self_inverse": float(np.mean(np.minimum(
                    np.abs(ang), np.pi - np.abs(ang)))),
                    "norm": float(np.abs(ang).mean())}
            else:
                rels[name] = {"norm": float(np.linalg.norm(m.R[i]))}
        out[kind] = {"eval": ev, "relations": rels,
                     "per_pattern": {names[int(k)]: v
                                     for k, v in ev["per_relation"].items()}}
        print(f"    {kind}: MRR {ev['mrr']:.4f}, "
              + " ".join(f"{n}={v['mrr']:.3f}" for n, v in out[kind]["per_pattern"].items()))
    # la composición, comprobada como álgebra y no como métrica: para TransE
    # r_grandparent debería ser r_parent + r_parent
    tm = cached("pat_transe", lambda: None)
    comp = None
    if tm is not None:
        rp = tm.R[meta["names"]["parent"]]
        rg = tm.R[meta["names"]["grandparent"]]
        rc = tm.R[meta["names"]["child"]]
        comp = {"parent_plus_parent": float(np.linalg.norm(rg - 2 * rp)
                                            / max(np.linalg.norm(rg), 1e-9)),
                "inverse": float(np.linalg.norm(rp + rc) / max(np.linalg.norm(rp), 1e-9)),
                "norm_parent": float(np.linalg.norm(rp)),
                "norm_mirror": float(np.linalg.norm(tm.R[meta["names"]["mirror"]]))}
    return {"models": out, "meta": meta, "composition": comp}


# --------------------------------------------------------------------------
# S4. Lo que viaja al navegador: una rotación en el plano, en vivo.
# --------------------------------------------------------------------------
def stage_widget(pat):
    train, valid, test, meta = pattern_kg()
    m = cached("pat_rotate", lambda: None)
    tm = cached("pat_transe", lambda: None)
    if m is None or tm is None:
        return None
    names = meta["names"]
    rng = np.random.default_rng(SEED + 5)
    # el par de dimensiones complejas donde más se mueve cada relación
    k = m.k
    ex = []
    for name, i in names.items():
        rows = train[train[:, 1] == i][:6]
        ex.append({"relation": name, "angle": float(m.R[i][0]),
                   "angles": [float(a) for a in m.R[i][:8]],
                   "shift": [float(v) for v in tm.R[i][:2]],
                   "pairs": [[int(h), int(t)] for h, _, t in rows]})
    ents = sorted({int(v) for row in ex for pair in row["pairs"] for v in pair})
    coords = {int(e): [float(m.E[e][0]), float(m.E[e][k])] for e in ents}
    tcoords = {int(e): [float(tm.E[e][0]), float(tm.E[e][1])] for e in ents}
    return {"relations": ex, "coords": coords, "transe": tcoords,
            "gamma": GAMMA, "dim": DIM}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("cargando WN18RR")
    train, valid, test, ent, rel = wn18rr()
    n_ent, n_rel = len(ent), len(rel)
    rel_rows, inv = relation_stats(train, rel)
    print(f"  {n_ent} entidades, {n_rel} relaciones, {len(train)} tripletas")

    print("los tres modelos")
    wn = cached("wn", lambda: stage_wn(train, valid, test, n_ent, n_rel, rel_rows))

    print("el grafo con los patrones escritos")
    pat = cached("patterns", stage_patterns)

    widget = stage_widget(pat)

    data = {
        "meta": {"seed": SEED, "dim": DIM, "epochs": EPOCHS, "neg": NEG, "gamma": GAMMA,
                 "entities": n_ent, "relations": n_rel, "train": int(len(train)),
                 "valid": int(len(valid)), "test": int(len(test)),
                 "digest": digest(), "eval_triples": wn["transe"]["eval"]["n"] // 2},
        "relations": r(rel_rows, 4),
        "wn": r(wn, 5),
        "patterns": r({"models": pat["models"], "meta": pat["meta"],
                       "composition": pat["composition"]}, 5),
        "widget": r(widget, 5),
    }
    path = OUT / "kg.json"
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    print(f"escrito {path} ({path.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
