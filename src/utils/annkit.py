"""Índices de vecinos aproximados, escritos aquí, para el módulo 8.3.

Los dos artículos de búsqueda vectorial necesitan los mismos tres índices y la
misma contabilidad, así que viven juntos. Y la contabilidad es la mitad del
asunto: **lo que se cuenta son distancias calculadas**, no segundos. Un reloj de
pared mide la contención de la máquina (trampa 28 de `verificacion.md`) y además
no se puede reproducir en otro sitio; el número de distancias que un índice
evalúa para contestar una consulta es una propiedad del índice y del dato, y es
la que decide el coste en cualquier máquina.

Lo que hay:

  * `brute`, que además es la verdad contra la que se puntúa todo.
  * `IVF`, la partición por k-means con sondeo de las celdas más cercanas, que
     es el índice invertido de FAISS.
  * `HNSW`, el grafo navegable por capas, con la búsqueda voraz y el heurístico
     de poda de aristas del artículo original, y con la jerarquía **ablable**
     para poder medir qué compra.
  * `PQ`, la cuantización por producto: el espacio se parte en trozos, cada
     trozo tiene su propio libro de códigos, y una distancia se estima sumando
     las de los trozos.
"""
import heapq

import numpy as np


def brute(X, Q, k, block=64):
    """La respuesta, y el coste exacto de conseguirla.

    Por bloques de consultas y con la identidad de la norma: la forma directa
    (`(Q[:, None] - X[None]) ** 2`) materializa un tensor de consultas por
    vectores por dimensiones, que con doscientas consultas sobre seis mil
    vectores de sesenta y cuatro columnas son seiscientos megas para tirarlos
    en la línea siguiente.
    """
    xn = (X ** 2).sum(1)[None, :]
    out = np.zeros((len(Q), k), int)
    for i in range(0, len(Q), block):
        q = Q[i:i + block]
        d = xn - 2.0 * (q @ X.T) + (q ** 2).sum(1)[:, None]
        out[i:i + block] = np.argsort(d, axis=1)[:, :k]
    return out, len(X) * len(Q)


def pad_to(res, k):
    """Rellena listas cortas con -1 hasta k, para poder apilarlas.

    Un índice invertido que sondea pocas celdas puede devolver MENOS de k
    vecinos, porque en las celdas que miró no hay más. Es comportamiento real y
    no un fallo, así que no se descarta la consulta ni se rellena con vecinos
    inventados: se rellena con un índice imposible, que ningún conjunto de
    verdad contiene y que por tanto cuenta como no encontrado en el recall.
    """
    out = np.full((len(res), k), -1, dtype=int)
    for i, a in enumerate(res):
        a = np.asarray(a, dtype=int)[:k]
        out[i, :len(a)] = a
    return out


def recall_at(found, truth):
    """Fracción de los vecinos verdaderos que el índice devolvió."""
    k = truth.shape[1]
    return float(np.mean([len(set(a[:k].tolist()) & set(b.tolist())) / k
                          for a, b in zip(found, truth)]))


class Counter:
    """Un contador de distancias que se pasa a los índices.

    Existe para que el coste que la página publica sea el que el índice pagó de
    verdad y no una estimación escrita a mano al lado del bucle.
    """

    def __init__(self):
        self.n = 0

    def dist(self, X, i, q):
        self.n += len(i) if hasattr(i, "__len__") else 1
        return ((X[i] - q) ** 2).sum(-1)


# ------------------------------------------------------------------- k-means
def kmeans(X, k, rng, iters=25):
    """k-means++ sencillo; el artículo 12 explica lo que aquí solo se usa."""
    n = len(X)
    c = [int(rng.integers(0, n))]
    d2 = ((X - X[c[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        p = d2 / max(d2.sum(), 1e-12)
        c.append(int(rng.choice(n, p=p)))
        d2 = np.minimum(d2, ((X - X[c[-1]]) ** 2).sum(1))
    C = X[c].copy()
    # La distancia se calcula por la identidad de la norma en vez de restando
    # tensores: con 256 centros y ocho mil filas, el temporal (n, k, d) son
    # cientos de megas y el producto escalar son unos pocos.
    xn = (X ** 2).sum(1)[:, None]

    def assign(C):
        return np.argmin(xn - 2.0 * (X @ C.T) + (C ** 2).sum(1)[None, :], 1)

    for _ in range(iters):
        a = assign(C)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    return C, assign(C)


class IVF:
    def __init__(self, X, nlist, rng, iters=25):
        self.X = X
        self.C, self.assign = kmeans(X, nlist, rng, iters)
        self.lists = [np.where(self.assign == j)[0] for j in range(nlist)]
        self.nlist = nlist

    def search(self, q, k, nprobe):
        cost = self.nlist
        dc = ((self.C - q) ** 2).sum(1)
        cells = np.argsort(dc)[:nprobe]
        cand = np.concatenate([self.lists[c] for c in cells]) if len(cells) else \
            np.array([], int)
        if len(cand) == 0:
            return np.array([], int), cost
        d = ((self.X[cand] - q) ** 2).sum(1)
        cost += len(cand)
        return cand[np.argsort(d)[:k]], cost


# --------------------------------------------------------------------- HNSW
class HNSW:
    """Grafo navegable de mundo pequeño por capas.

    Se escribe entero porque el artículo mide sus piezas por separado: sin las
    capas de arriba es un NSW, y esa ablación es una de las medidas de la
    página. `heuristic` activa la poda de aristas del artículo original, que es
    lo que hace que un nodo conserve vecinos en direcciones distintas en vez de
    quedarse con los k más cercanos, todos del mismo lado.
    """

    def __init__(self, X, M=16, ef_construction=100, levels=True, rng=None,
                 heuristic=True):
        self.X = X
        self.M = M
        self.M0 = 2 * M
        self.ef_c = ef_construction
        self.levels = levels
        self.heuristic = heuristic
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.mL = 1.0 / np.log(M) if M > 1 else 1.0
        self.graph = []           # una lista de dict por capa
        self.node_level = np.zeros(len(X), int)
        self.entry = None
        self.build_cost = 0
        self._build()

    def _d(self, i, q):
        self.build_cost += 1
        return float(((self.X[i] - q) ** 2).sum())

    def _layer(self, lvl):
        while len(self.graph) <= lvl:
            self.graph.append({})
        return self.graph[lvl]

    def _search_layer(self, q, entry, ef, lvl, counter=None):
        g = self.graph[lvl]
        visited = set(entry)
        cand, res = [], []
        for e in entry:
            d = self._d(e, q) if counter is None else \
                float(counter.dist(self.X, np.array([e]), q)[0])
            heapq.heappush(cand, (d, e))
            heapq.heappush(res, (-d, e))
        while cand:
            d, c = heapq.heappop(cand)
            if -res[0][0] < d and len(res) >= ef:
                break
            for nb in g.get(c, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                dn = self._d(nb, q) if counter is None else \
                    float(counter.dist(self.X, np.array([nb]), q)[0])
                if len(res) < ef or dn < -res[0][0]:
                    heapq.heappush(cand, (dn, nb))
                    heapq.heappush(res, (-dn, nb))
                    if len(res) > ef:
                        heapq.heappop(res)
        return sorted([(-d, i) for d, i in res])

    def _select(self, q, cands, m):
        """Los m vecinos que se conservan, con o sin heurístico."""
        cands = sorted(cands)
        if not self.heuristic:
            return [i for _, i in cands[:m]]
        keep = []
        for d, i in cands:
            if len(keep) >= m:
                break
            ok = True
            for _, j in keep:
                if float(((self.X[i] - self.X[j]) ** 2).sum()) < d:
                    ok = False
                    break
            if ok:
                keep.append((d, i))
        if len(keep) < m:
            have = {i for _, i in keep}
            for d, i in cands:
                if len(keep) >= m:
                    break
                if i not in have:
                    keep.append((d, i))
        return [i for _, i in keep]

    def _build(self):
        order = np.arange(len(self.X))
        for n, i in enumerate(order):
            lvl = (int(np.floor(-np.log(max(self.rng.random(), 1e-12)) * self.mL))
                   if self.levels else 0)
            self.node_level[i] = lvl
            self._layer(lvl)
            if self.entry is None:
                for L in range(lvl + 1):
                    self._layer(L).setdefault(i, [])
                self.entry = i
                continue
            ep = [self.entry]
            top = len(self.graph) - 1
            for L in range(top, lvl, -1):
                if not self.graph[L]:
                    continue
                ep = [self._search_layer(self.X[i], ep, 1, L)[0][1]]
            for L in range(min(lvl, len(self.graph) - 1), -1, -1):
                g = self._layer(L)
                g.setdefault(i, [])
                found = self._search_layer(self.X[i], ep, self.ef_c, L)
                m = self.M0 if L == 0 else self.M
                nbrs = self._select(self.X[i], found, m)
                g[i] = list(nbrs)
                for nb in nbrs:
                    g.setdefault(nb, [])
                    if i not in g[nb]:
                        g[nb].append(i)
                    if len(g[nb]) > m:
                        d = [(float(((self.X[x] - self.X[nb]) ** 2).sum()), x)
                             for x in g[nb]]
                        g[nb] = self._select(self.X[nb], d, m)
                ep = [x for _, x in found[:1]] or ep
            if lvl > len(self.graph) - 1 or self.node_level[i] > self.node_level[self.entry]:
                self.entry = i

    def search(self, q, k, ef):
        c = Counter()
        ep = [self.entry]
        for L in range(len(self.graph) - 1, 0, -1):
            if not self.graph[L]:
                continue
            ep = [self._search_layer(q, ep, 1, L, counter=c)[0][1]]
        found = self._search_layer(q, ep, max(ef, k), 0, counter=c)
        return np.array([i for _, i in found[:k]]), c.n

    def degrees(self, layer=0):
        return np.array([len(v) for v in self.graph[layer].values()])


# ----------------------------------------------------------------------- PQ
class PQ:
    """Cuantización por producto: el espacio partido en trozos con su libro."""

    def __init__(self, X, m, bits, rng, iters=20):
        self.m = m
        self.k = 2 ** bits
        self.bits = bits
        self.d = X.shape[1]
        assert self.d % m == 0, "los trozos tienen que repartirse enteros"
        self.sub = self.d // m
        self.books = []
        self.codes = np.zeros((len(X), m), np.int32)
        for j in range(m):
            sl = slice(j * self.sub, (j + 1) * self.sub)
            C, a = kmeans(X[:, sl], self.k, rng, iters)
            self.books.append(C)
            self.codes[:, j] = a

    def reconstruct(self, i=None):
        codes = self.codes if i is None else self.codes[i]
        out = np.zeros((len(codes), self.d))
        for j in range(self.m):
            out[:, j * self.sub:(j + 1) * self.sub] = self.books[j][codes[:, j]]
        return out

    def table(self, q):
        """Distancias asimétricas: la consulta se queda entera."""
        T = np.zeros((self.m, self.k))
        for j in range(self.m):
            sl = slice(j * self.sub, (j + 1) * self.sub)
            T[j] = ((self.books[j] - q[sl]) ** 2).sum(1)
        return T

    def adc(self, q):
        T = self.table(q)
        return T[np.arange(self.m)[:, None], self.codes.T].sum(0)

    def sdc(self, q):
        """Distancias simétricas: la consulta también se cuantiza."""
        code = np.array([int(np.argmin(((self.books[j] -
                                         q[j * self.sub:(j + 1) * self.sub]) ** 2).sum(1)))
                         for j in range(self.m)])
        out = np.zeros(len(self.codes))
        for j in range(self.m):
            d = ((self.books[j] - self.books[j][code[j]]) ** 2).sum(1)
            out += d[self.codes[:, j]]
        return out

    def bytes_per_vector(self):
        return self.m * self.bits / 8.0
