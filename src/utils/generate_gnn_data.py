"""Las cifras del artículo de redes en grafos (módulo 6.2).

Todo lo que este atlas ha clasificado hasta ahora venía en filas independientes.
Aquí una fila tiene vecinos, y la pregunta del módulo es qué añade eso. Se mide
sobre Cora (2.708 artículos, 5.278 citas, 1.433 palabras, siete temas, la
partición estándar de Planetoid) y sobre grafos escritos aquí, donde la
respuesta se conoce antes de existir la primera arista.

Lo que se mide:

  1. **Los dos controles que casi nadie corre juntos**: los rasgos sin el grafo
     (un perceptrón) y el grafo sin los rasgos (propagación de etiquetas). En
     Cora, el segundo es sorprendentemente fuerte, y sin él la mejora de una red
     en grafos se lee como si viniera de la arquitectura.
  2. **El mando de homofilia**, sobre grafos sintéticos con las clases escritas:
     a partir de qué punto promediar a los vecinos deja de ayudar y empieza a
     estorbar. Con el cruce medido, no citado.
  3. **La profundidad**: exactitud y colapso de las representaciones capa a
     capa, con la energía de Dirichlet como medida del segundo, que es lo que
     "over-smoothing" quiere decir cuando se mide.
  4. **La atención**: si GAT aprende a bajarle el peso a las aristas que unen
     clases distintas, contra la verdad conocida y contra el 1/grado que un GCN
     usa sin aprender nada.
  5. **GraphSAGE y la inducción**: nodos que no existían durante el
     entrenamiento, y el muestreo de vecinos con su varianza.

Ejecutar desde cualquier sitio: `python src/utils/generate_gnn_data.py`.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_data import contextual_sbm, cora, digest_cora  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "gnn" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "gnn_cache")

SEED = 19
HIDDEN = 64
EPOCHS = 200
LR = 0.01
WD = 5e-4
SEEDS = 3                  # semillas por brazo, que es lo que da la dispersion publicada
TAG = f"s{SEED}h{HIDDEN}e{EPOCHS}n{SEEDS}"


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
# Los operadores del grafo. Los tres que aparecen en la página, escritos una
# vez: nada de esto es aprendido y elegir uno u otro mueve la cifra.
# --------------------------------------------------------------------------
def normalise(A, mode="sym", self_loops=True):
    n = A.shape[0]
    M = A + sp.eye(n, format="csr", dtype=np.float32) if self_loops else A.tocsr()
    deg = np.asarray(M.sum(1)).ravel()
    if mode == "none":
        return M
    if mode == "row":
        return sp.diags(1.0 / np.maximum(deg, 1e-9)) @ M
    d = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    return sp.diags(d) @ M @ sp.diags(d)


def adam(params, grads, state, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        if i not in state:
            state[i] = [np.zeros_like(p), np.zeros_like(p), 0]
        m, v, t = state[i]
        t += 1
        m[:] = b1 * m + (1 - b1) * g
        v[:] = b2 * v + (1 - b2) * (g * g)
        p -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        state[i][2] = t


def softmax_ce(logits, y, idx):
    z = logits[idx]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    loss = -np.log(np.maximum(p[np.arange(len(idx)), y[idx]], 1e-12)).mean()
    g = p
    g[np.arange(len(idx)), y[idx]] -= 1.0
    g /= len(idx)
    full = np.zeros_like(logits)
    full[idx] = g
    return loss, full


def accuracy(logits, y, idx):
    return float((logits[idx].argmax(1) == y[idx]).mean())


# --------------------------------------------------------------------------
# Los modelos. Dos capas, gradientes a mano, y la única diferencia entre ellos
# es qué hace la capa con los vecinos.
# --------------------------------------------------------------------------
class Net:
    """MLP, GCN y SAGE en una sola clase: el operador es un argumento.

    `op` es None para el perceptrón (cada fila sola), la matriz normalizada
    para un GCN, y para SAGE hay dos matrices de pesos porque el nodo y sus
    vecinos entran por caminos distintos, que es la diferencia entera."""

    def __init__(self, d_in, d_hid, d_out, op=None, sage=False, layers=2, seed=SEED):
        rng = np.random.default_rng(seed)
        self.op, self.sage, self.layers = op, sage, layers
        dims = [d_in] + [d_hid] * (layers - 1) + [d_out]
        self.W = [rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
                  for i in range(layers)]
        self.Ws = ([rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
                    for i in range(layers)] if sage else None)
        self.state = {}

    def params(self):
        return self.W + (self.Ws if self.sage else [])

    def forward(self, X):
        H = X
        self.cache = []
        for li in range(self.layers):
            pre_self = H @ self.W[li]
            if self.op is None:
                Z = pre_self
                agg = None
            elif self.sage:
                agg = self.op @ H
                Z = pre_self + agg @ self.Ws[li]
            else:
                agg = self.op @ pre_self
                Z = agg
            self.cache.append((H, agg))
            H = np.maximum(Z, 0) if li < self.layers - 1 else Z
            if li < self.layers - 1:
                self.mask = H > 0
                self.cache[-1] = (self.cache[-1][0], self.cache[-1][1], H > 0)
        return H

    def backward(self, dZ):
        gW = [None] * self.layers
        gWs = [None] * self.layers if self.sage else None
        for li in range(self.layers - 1, -1, -1):
            item = self.cache[li]
            H, agg = item[0], item[1]
            if li < self.layers - 1:
                dZ = dZ * item[2]
            if self.op is None:
                gW[li] = (H.T @ dZ) if not sp.issparse(H) else np.asarray((H.T @ dZ))
                dH = dZ @ self.W[li].T
            elif self.sage:
                gW[li] = np.asarray(H.T @ dZ)
                gWs[li] = np.asarray(agg.T @ dZ)
                dH = dZ @ self.W[li].T + self.op.T @ (dZ @ self.Ws[li].T)
            else:
                dpre = self.op.T @ dZ
                gW[li] = np.asarray(H.T @ dpre)
                dH = dpre @ self.W[li].T
            dZ = dH
        return gW + (gWs if self.sage else [])

    def fit(self, X, y, idx_train, idx_val, epochs=EPOCHS, lr=LR, wd=WD):
        best, best_w, patience = -1, None, 0
        for ep in range(epochs):
            logits = self.forward(X)
            loss, dlog = softmax_ce(logits, y, idx_train)
            grads = self.backward(dlog)
            ps = self.params()
            for g, p in zip(grads, ps):
                g += wd * p
            adam(ps, grads, self.state, lr=lr)
            if ep % 5 == 0 or ep == epochs - 1:
                acc = accuracy(self.forward(X), y, idx_val)
                if acc > best:
                    best, best_w = acc, [p.copy() for p in ps]
                    patience = 0
                else:
                    patience += 1
                    if patience > 12:
                        break
        if best_w is not None:
            for p, b in zip(self.params(), best_w):
                p[:] = b
        return self


class GAT:
    """Una capa de atención escrita a mano, para poder mirarle los pesos.

    Es la versión de una cabeza del paper: un vector de atención por lado, una
    LeakyReLU, y softmax sobre los vecinos de cada nodo. Lo que la página mide
    no es su exactitud, que apenas se mueve, sino si el peso que le da a una
    arista entre clases distintas es menor que el que un GCN le da sin aprender
    nada."""

    def __init__(self, d_in, d_hid, d_out, edges, n, seed=SEED, slope=0.2):
        rng = np.random.default_rng(seed)
        self.src, self.dst = edges
        self.n, self.slope = n, slope
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), (d_in, d_hid))
        self.a1s = rng.normal(0, 0.1, d_hid)
        self.a1d = rng.normal(0, 0.1, d_hid)
        self.W2 = rng.normal(0, np.sqrt(2.0 / d_hid), (d_hid, d_out))
        self.a2s = rng.normal(0, 0.1, d_out)
        self.a2d = rng.normal(0, 0.1, d_out)
        self.state = {}

    def params(self):
        return [self.W1, self.a1s, self.a1d, self.W2, self.a2s, self.a2d]

    def _attend(self, Z, a_s, a_d):
        s = Z @ a_s
        d = Z @ a_d
        raw = s[self.dst] + d[self.src]                 # mensaje de src hacia dst
        e = np.where(raw > 0, raw, self.slope * raw)
        # softmax por vecindario, estabilizado restando el máximo de cada uno:
        # los nodos sin aristas entrantes se quedarían en -inf y su exponencial
        # en NaN, así que el máximo arranca en cero y no en -inf.
        mx = np.zeros(self.n)
        np.maximum.at(mx, self.dst, e)
        ex = np.exp(e - mx[self.dst])
        den = np.bincount(self.dst, weights=ex, minlength=self.n)
        alpha = ex / np.maximum(den[self.dst], 1e-12)
        out = np.zeros((self.n, Z.shape[1]))
        for c in range(Z.shape[1]):
            out[:, c] = np.bincount(self.dst, weights=alpha * Z[self.src, c], minlength=self.n)
        return out, alpha, raw, s, d

    def forward(self, X):
        Z1 = np.asarray(X @ self.W1)
        H1, a1, raw1, s1, d1 = self._attend(Z1, self.a1s, self.a1d)
        A1 = np.maximum(H1, 0)
        Z2 = A1 @ self.W2
        H2, a2, raw2, s2, d2 = self._attend(Z2, self.a2s, self.a2d)
        self.cache = (X, Z1, H1, a1, raw1, A1, Z2, a2, raw2)
        self.alpha = (a1, a2)
        return H2

    def _attend_backward(self, dOut, Z, alpha, raw, a_s, a_d):
        src, dst = self.src, self.dst
        dalpha = np.einsum('ij,ij->i', dOut[dst], Z[src])
        dZ = np.zeros_like(Z)
        for c in range(Z.shape[1]):
            dZ[:, c] += np.bincount(src, weights=alpha * dOut[dst, c], minlength=self.n)
        dot = np.bincount(dst, weights=alpha * dalpha, minlength=self.n)
        de = alpha * (dalpha - dot[dst])
        dl = np.where(raw > 0, de, self.slope * de)
        ds = np.bincount(dst, weights=dl, minlength=self.n)
        dd = np.bincount(src, weights=dl, minlength=self.n)
        g_as = Z.T @ ds
        g_ad = Z.T @ dd
        dZ += np.outer(ds, a_s) + np.outer(dd, a_d)
        return dZ, g_as, g_ad

    def backward(self, dOut):
        X, Z1, H1, a1, raw1, A1, Z2, a2, raw2 = self.cache
        dZ2, g_a2s, g_a2d = self._attend_backward(dOut, Z2, a2, raw2, self.a2s, self.a2d)
        gW2 = A1.T @ dZ2
        dA1 = dZ2 @ self.W2.T
        dH1 = dA1 * (H1 > 0)
        dZ1, g_a1s, g_a1d = self._attend_backward(dH1, Z1, a1, raw1, self.a1s, self.a1d)
        gW1 = np.asarray(X.T @ dZ1)
        return [gW1, g_a1s, g_a1d, gW2, g_a2s, g_a2d]

    def fit(self, X, y, idx_train, idx_val, epochs=EPOCHS, lr=LR, wd=WD):
        best, best_w, patience = -1, None, 0
        for ep in range(epochs):
            logits = self.forward(X)
            loss, dlog = softmax_ce(logits, y, idx_train)
            grads = self.backward(dlog)
            for g, p in zip(grads, self.params()):
                g += wd * p
            adam(self.params(), grads, self.state, lr=lr)
            if ep % 5 == 0 or ep == epochs - 1:
                acc = accuracy(self.forward(X), y, idx_val)
                if acc > best:
                    best, best_w, patience = acc, [p.copy() for p in self.params()], 0
                else:
                    patience += 1
                    if patience > 12:
                        break
        if best_w is not None:
            for p, b in zip(self.params(), best_w):
                p[:] = b
        self.forward(X)
        return self


def label_propagation(A, y, idx_train, n_classes, alpha=0.9, steps=50):
    """El grafo sin los rasgos, que es el control que casi nadie publica."""
    n = A.shape[0]
    Y = np.zeros((n, n_classes))
    Y[idx_train, y[idx_train]] = 1.0
    P = normalise(A, "row", self_loops=False)
    F = Y.copy()
    for _ in range(steps):
        F = alpha * (P @ F) + (1 - alpha) * Y
    return F


# --------------------------------------------------------------------------
# S1. Cora: los dos controles y los tres modelos.
# --------------------------------------------------------------------------
def dense(X):
    return np.asarray(X.todense()) if sp.issparse(X) else X


def stage_cora(X, y, A, tr, va, te):
    n_cls = int(y.max()) + 1
    Xd = dense(X)
    Asym = normalise(A, "sym")
    Arow = normalise(A, "row")
    edges = A.tocoo()
    src = np.concatenate([edges.row, np.arange(A.shape[0])])
    dst = np.concatenate([edges.col, np.arange(A.shape[0])])

    out = {}
    def run(name, maker):
        accs = []
        for s in range(SEEDS):
            m = maker(SEED + s).fit(Xd, y, tr, va)
            accs.append(accuracy(m.forward(Xd), y, te))
        out[name] = {"acc": float(np.mean(accs)), "sd": float(np.std(accs)),
                     "runs": [float(a) for a in accs]}
        print(f"    {name}: {out[name]['acc']:.4f} +- {out[name]['sd']:.4f}")
        return m

    run("mlp", lambda s: Net(Xd.shape[1], HIDDEN, n_cls, op=None, seed=s))
    run("gcn", lambda s: Net(Xd.shape[1], HIDDEN, n_cls, op=Asym, seed=s))
    run("sage", lambda s: Net(Xd.shape[1], HIDDEN, n_cls, op=Arow, sage=True, seed=s))
    gat = run("gat", lambda s: GAT(Xd.shape[1], HIDDEN, n_cls, (src, dst), A.shape[0], seed=s))

    F = label_propagation(A, y, tr, n_cls)
    out["label_prop"] = {"acc": float((F[te].argmax(1) == y[te]).mean()), "sd": 0.0, "runs": []}
    # y el control de nivel cero: la clase mayoritaria del conjunto etiquetado
    maj = np.bincount(y[tr], minlength=n_cls).argmax()
    out["majority"] = {"acc": float((y[te] == maj).mean()), "sd": 0.0, "runs": []}

    # el operador importa: el mismo GCN con tres normalizaciones
    ops = {"sym": Asym, "row": Arow, "none": normalise(A, "none")}
    norms = {}
    for key, op in ops.items():
        accs = [accuracy(Net(Xd.shape[1], HIDDEN, n_cls, op=op, seed=SEED + s)
                         .fit(Xd, y, tr, va).forward(Xd), y, te) for s in range(3)]
        norms[key] = {"acc": float(np.mean(accs)), "sd": float(np.std(accs))}
    out["_norms"] = norms

    # la atención, contra la verdad conocida
    a2 = gat.alpha[1]
    same = y[src] == y[dst]
    self_edge = src == dst
    deg = np.asarray(A.sum(1)).ravel() + 1
    uniform = 1.0 / deg[dst]
    att = {
        "same": float(a2[same & ~self_edge].mean()),
        "diff": float(a2[(~same) & ~self_edge].mean()),
        "self": float(a2[self_edge].mean()),
        "uniform_same": float(uniform[same & ~self_edge].mean()),
        "uniform_diff": float(uniform[(~same) & ~self_edge].mean()),
        "ratio": float(a2[same & ~self_edge].mean() / max(a2[(~same) & ~self_edge].mean(), 1e-9)),
        "uniform_ratio": float(uniform[same & ~self_edge].mean()
                               / max(uniform[(~same) & ~self_edge].mean(), 1e-9)),
        "edges": int(src.size), "cross_edges": int(((~same) & ~self_edge).sum()),
        "corr": float(np.corrcoef(a2[~self_edge], same[~self_edge].astype(float))[0, 1]),
    }
    out["_attention"] = att
    return out


# --------------------------------------------------------------------------
# S2. El mando de homofilia, sobre grafos con la respuesta escrita.
# --------------------------------------------------------------------------
def stage_homophily():
    rows = []
    for h in [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0]:
        accs = {"mlp": [], "gcn": [], "sage": []}
        real_h = []
        for s in range(SEEDS):
            # Dos parámetros de este generador deciden si el barrido mide algo:
            # con cuatro clases en vez de dos, "mis vecinos son de otra clase"
            # deja de ser tan informativo como "son de la mía", que es lo que
            # hace que la heterofilia duela de verdad; y con la señal de los
            # rasgos bajada, el perceptrón deja de saturar en 0,998 y hay sitio
            # para que el grafo aporte. La primera versión tenía dos clases y
            # señal 1,0, y las tres curvas salían planas y pegadas al techo.
            X, y, A, meta = contextual_sbm(n=1200, classes=4, d=16, signal=0.45,
                                           homophily=h, seed=SEED + s)
            co = A.tocoo()
            real_h.append(float((y[co.row] == y[co.col]).mean()))
            n = len(y)
            idx = np.random.default_rng(SEED + s).permutation(n)
            tr, va, te = idx[:100], idx[100:300], idx[300:]
            n_cls = int(y.max()) + 1
            accs["mlp"].append(accuracy(
                Net(X.shape[1], 32, n_cls, op=None, seed=SEED + s).fit(X, y, tr, va, epochs=120)
                .forward(X), y, te))
            Asym = normalise(A, "sym")
            accs["gcn"].append(accuracy(
                Net(X.shape[1], 32, n_cls, op=Asym, seed=SEED + s).fit(X, y, tr, va, epochs=120)
                .forward(X), y, te))
            accs["sage"].append(accuracy(
                Net(X.shape[1], 32, n_cls, op=normalise(A, "row"), sage=True, seed=SEED + s)
                .fit(X, y, tr, va, epochs=120).forward(X), y, te))
        rows.append({"homophily": h, "measured": float(np.mean(real_h)),
                     **{k: float(np.mean(v)) for k, v in accs.items()},
                     **{f"{k}_sd": float(np.std(v)) for k, v in accs.items()}})
        print(f"    homofilia {h}: mlp {rows[-1]['mlp']:.3f} gcn {rows[-1]['gcn']:.3f} "
              f"sage {rows[-1]['sage']:.3f}")
    return rows


# --------------------------------------------------------------------------
# S3. Profundidad y colapso.
# --------------------------------------------------------------------------
def dirichlet(H, A):
    """Energía de Dirichlet normalizada: cuánto difieren dos vecinos.

    Es la medida de lo que "over-smoothing" quiere decir: si tiende a cero,
    todos los nodos han acabado con la misma representación y ninguna capa
    posterior puede distinguirlos."""
    co = A.tocoo()
    d = np.asarray(A.sum(1)).ravel() + 1.0
    Hn = H / np.sqrt(d)[:, None]
    diff = Hn[co.row] - Hn[co.col]
    return float((diff ** 2).sum() / max((H ** 2).sum(), 1e-12))


def stage_depth(X, y, A, tr, va, te):
    Xd = dense(X)
    Asym = normalise(A, "sym")
    n_cls = int(y.max()) + 1
    rows = []
    for L in [1, 2, 3, 4, 6, 8]:
        accs, energies = [], []
        for s in range(3):
            m = Net(Xd.shape[1], HIDDEN, n_cls, op=Asym, layers=L, seed=SEED + s)
            m.fit(Xd, y, tr, va)
            logits = m.forward(Xd)
            accs.append(accuracy(logits, y, te))
            energies.append(dirichlet(logits, A))
        rows.append({"layers": L, "acc": float(np.mean(accs)), "sd": float(np.std(accs)),
                     "energy": float(np.mean(energies))})
        print(f"    {L} capas: {rows[-1]['acc']:.4f} energía {rows[-1]['energy']:.5f}")
    # y el mismo suavizado sin entrenar nada: A^L X, que es lo que la capa hace
    smooth = []
    H = np.asarray(Xd, dtype=np.float64)
    for L in range(0, 13):
        smooth.append({"steps": L, "energy": dirichlet(H, A)})
        H = Asym @ H
    return {"trained": rows, "smoothing": smooth}


# --------------------------------------------------------------------------
# S4. GraphSAGE: nodos nuevos y muestreo de vecinos.
# --------------------------------------------------------------------------
def stage_inductive(X, y, A, tr, va, te):
    """El experimento que separa a SAGE del resto: quitar nodos del grafo.

    Transductivo quiere decir que los nodos de test estaban ahí durante el
    entrenamiento, aunque sin etiqueta. Aquí se borran enteros, aristas
    incluidas, y se devuelven al final."""
    Xd = dense(X)
    n_cls = int(y.max()) + 1
    rng = np.random.default_rng(SEED)
    hidden_nodes = rng.choice(te, size=len(te) // 2, replace=False)
    keep = np.ones(A.shape[0], dtype=bool)
    keep[hidden_nodes] = False
    Acut = A.copy().tolil()
    Acut[hidden_nodes, :] = 0
    Acut[:, hidden_nodes] = 0
    Acut = sp.csr_matrix(Acut)

    out = {}
    for name, op_cut, op_full, sage in [
        ("gcn", normalise(Acut, "sym"), normalise(A, "sym"), False),
        ("sage", normalise(Acut, "row"), normalise(A, "row"), True)]:
        accs_t, accs_i = [], []
        for s in range(3):
            m = Net(Xd.shape[1], HIDDEN, n_cls, op=op_cut, sage=sage, seed=SEED + s)
            m.fit(Xd, y, tr, va)
            m.op = op_full                                  # el grafo entero al evaluar
            logits = m.forward(Xd)
            accs_i.append(accuracy(logits, y, hidden_nodes))
            seen = np.setdiff1d(te, hidden_nodes)
            accs_t.append(accuracy(logits, y, seen))
        out[name] = {"unseen": float(np.mean(accs_i)), "seen": float(np.mean(accs_t)),
                     "unseen_sd": float(np.std(accs_i))}
        print(f"    {name}: vistos {out[name]['seen']:.4f} nuevos {out[name]['unseen']:.4f}")

    # el muestreo de vecinos: cuánto se pierde y cuánto baila
    Arow = normalise(A, "row")
    m = Net(Xd.shape[1], HIDDEN, n_cls, op=Arow, sage=True, seed=SEED).fit(Xd, y, tr, va)
    sample_rows = []
    for k in [1, 2, 3, 5, 10, 25]:
        accs = []
        for s in range(5):
            rs = np.random.default_rng(1000 + s)
            co = A.tocoo()
            order = rs.permutation(co.data.size)
            row, col = co.row[order], co.col[order]
            counts = {}
            keep_e = []
            for e, (a, b) in enumerate(zip(row, col)):
                c = counts.get(a, 0)
                if c < k:
                    counts[a] = c + 1
                    keep_e.append(e)
            As = sp.csr_matrix((np.ones(len(keep_e), dtype=np.float32),
                                (row[keep_e], col[keep_e])), shape=A.shape)
            m.op = normalise(As, "row")
            accs.append(accuracy(m.forward(Xd), y, te))
        sample_rows.append({"k": k, "acc": float(np.mean(accs)), "sd": float(np.std(accs))})
        print(f"    muestreo {k}: {sample_rows[-1]['acc']:.4f} +- {sample_rows[-1]['sd']:.4f}")
    m.op = Arow
    full = accuracy(m.forward(Xd), y, te)
    return {"inductive": out, "sampling": sample_rows, "full": float(full),
            "hidden_nodes": int(len(hidden_nodes))}


# --------------------------------------------------------------------------
# S5. Lo que viaja al navegador: un grafo pequeño donde la página propaga.
# --------------------------------------------------------------------------
def stage_widget(X, y, A, tr, te):
    """Un trozo de Cora con sus vecindarios, para propagar en la página."""
    rng = np.random.default_rng(SEED + 3)
    # un nodo semilla bien conectado y sus dos anillos de vecinos
    deg = np.asarray(A.sum(1)).ravel()
    seed_node = int(np.argsort(-deg)[7])
    ring1 = A[seed_node].indices
    nodes = [seed_node] + list(ring1)
    for v in ring1[:14]:
        nodes += list(A[v].indices)
    nodes = list(dict.fromkeys(nodes))[:70]
    nodes = np.array(sorted(nodes))
    sub = A[np.ix_(nodes, nodes)].tocoo()
    pos_rng = np.random.default_rng(SEED)
    # una disposición de resortes barata, calculada aquí para que la página no
    # tenga que traer una librería de layout
    P = pos_rng.normal(size=(len(nodes), 2))
    Adj = np.asarray(A[np.ix_(nodes, nodes)].todense())
    for _ in range(400):
        diff = P[:, None, :] - P[None, :, :]
        d2 = (diff ** 2).sum(-1) + 1e-6
        rep = (diff / d2[:, :, None]).sum(1) * 0.02
        att = (Adj[:, :, None] * diff).sum(1) * 0.01
        P += rep - att - 0.001 * P
    P -= P.mean(0)
    P /= np.abs(P).max()
    X_sub = dense(X[nodes])
    return {"nodes": [{"id": int(v), "y": int(y[v]),
                       "labelled": bool(v in set(tr.tolist())),
                       "x": float(P[i, 0]), "y2": float(P[i, 1]),
                       "deg": int(deg[v])}
                      for i, v in enumerate(nodes)],
            "edges": [[int(a), int(b)] for a, b in zip(sub.row, sub.col) if a < b],
            "classes": int(y.max()) + 1,
            "feature_sum": [float(v) for v in X_sub.sum(1)]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("cargando Cora")
    X, y, A, tr, va, te = cora()
    co = A.tocoo()
    homophily = float((y[co.row] == y[co.col]).mean())
    print(f"  {A.shape[0]} nodos, {int(A.sum() // 2)} aristas, homofilia {homophily:.4f}")

    print("los dos controles y los tres modelos")
    cora_out = cached("cora", lambda: stage_cora(X, y, A, tr, va, te))

    print("el mando de homofilia")
    homo = cached("homophily", stage_homophily)

    print("profundidad")
    depth = cached("depth", lambda: stage_depth(X, y, A, tr, va, te))

    print("nodos nuevos y muestreo")
    ind = cached("inductive", lambda: stage_inductive(X, y, A, tr, va, te))

    widget = cached("widget", lambda: stage_widget(X, y, A, tr, te))

    deg = np.asarray(A.sum(1)).ravel()
    data = {
        "meta": {"seed": SEED, "hidden": HIDDEN, "epochs": EPOCHS, "seeds": SEEDS,
                 "nodes": int(A.shape[0]), "edges": int(A.sum() // 2),
                 "features": int(X.shape[1]), "classes": int(y.max()) + 1,
                 "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                 "homophily": homophily, "digest": digest_cora(),
                 "degree_mean": float(deg.mean()), "degree_max": int(deg.max()),
                 "isolated": int((deg == 0).sum()),
                 "feature_density": float(X.nnz / (X.shape[0] * X.shape[1]))},
        "cora": r({k: v for k, v in cora_out.items() if not k.startswith("_")}, 4),
        "norms": r(cora_out["_norms"], 4),
        "attention": r(cora_out["_attention"], 5),
        "homophily": r(homo, 4),
        "depth": r(depth, 6),
        "inductive": r(ind, 4),
        "widget": widget,
    }
    path = OUT / "gnn.json"
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    print(f"escrito {path} ({path.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
