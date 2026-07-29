"""Una red pequeña en numpy, para los artículos del módulo 8 que la necesitan.

Existe por una razón concreta: los dos artículos de ajuste fino (transferencia y
destilación por un lado, LoRA y QLoRA por el otro) tienen que **entrenar** algo,
y lo que miden no es la arquitectura sino lo que se le hace a una matriz de
pesos. Un perceptrón multicapa tiene matrices de pesos igual que un
transformador, así que la pregunta se puede contestar sin arrastrar torch a un
sitio donde el objeto de estudio son unas cuantas capas lineales.

Lo que hay aquí y por qué:

  * `Net`, un MLP con activación ReLU, softmax al final y Adam, que acepta
    objetivos **blandos** (una distribución por fila) además de etiquetas duras.
    Sin eso no hay destilación que medir.
  * Un parámetro de temperatura en la pérdida, con el factor T² que la iguala
    en escala a la pérdida dura, porque los gradientes de una entropía cruzada
    con logits divididos por T son T veces menores.
  * `lora_wrap`, que congela una matriz y le pone al lado el producto de dos
    matrices finas, que es literalmente toda la idea de LoRA.
  * `quantize_nf4` y compañía, que cuantizan una matriz a 4 bits por tres
    caminos distintos (uniforme, coma flotante y los cuantiles de una normal) y
    devuelven además cuántos bits por parámetro cuesta de verdad cada uno con
    sus escalas.

Todo determinista: el generador de números aleatorios se pasa siempre, y no hay
ninguna llamada a `np.random` global.
"""
import numpy as np


# ------------------------------------------------------------------ la red
def he_init(rng, n_in, n_out):
    return rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)


class Net:
    """MLP denso, con Adam y objetivos duros o blandos.

    `shape` es la lista de anchuras, entrada incluida y salida incluida:
    `[784, 128, 10]` es una capa oculta.
    """

    def __init__(self, shape, rng, scale=None):
        self.shape = list(shape)
        self.W = [he_init(rng, a, b) if scale is None else
                  rng.standard_normal((a, b)) * scale
                  for a, b in zip(shape, shape[1:])]
        self.b = [np.zeros(b) for b in shape[1:]]
        self._reset_adam()

    def _reset_adam(self):
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(x) for x in self.b]
        self.vb = [np.zeros_like(x) for x in self.b]
        self.t = 0

    def copy(self):
        out = Net.__new__(Net)
        out.shape = list(self.shape)
        out.W = [w.copy() for w in self.W]
        out.b = [x.copy() for x in self.b]
        out._reset_adam()
        return out

    def n_params(self):
        return int(sum(w.size for w in self.W) + sum(x.size for x in self.b))

    def forward(self, X, keep=False):
        acts = [X]
        h = X
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = h @ w + b
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
            if keep:
                acts.append(h)
        return (h, acts) if keep else h

    def features(self, X, layer=-1):
        """La representación oculta, que es lo que se transfiere."""
        h = X
        n = len(self.W) - 1 if layer < 0 else layer
        for i in range(n):
            h = np.maximum(h @ self.W[i] + self.b[i], 0.0)
        return h

    def probs(self, X, T=1.0):
        z = self.forward(X) / T
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def accuracy(self, X, y, batch=4096):
        ok = 0
        for i in range(0, len(X), batch):
            ok += int((np.argmax(self.forward(X[i:i + batch]), 1) == y[i:i + batch]).sum())
        return ok / len(X)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def onehot(y, k):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def train(net, X, Y, rng, epochs=20, batch=128, lr=1e-3, wd=0.0,
          trainable=None, trainable_bias=None, T=1.0, soft_weight=1.0,
          Yhard=None, log=None):
    """Adam sobre entropía cruzada, con objetivos blandos y temperatura.

    `Y` es una matriz de probabilidades por fila (para etiquetas duras, su
    indicador). Con `T > 1` el objetivo blando se lee a esa temperatura y el
    gradiente se multiplica por T², que es lo que deja la escala comparable a
    la de la pérdida dura; `soft_weight` mezcla las dos cuando `Yhard` viene.

    `trainable` es una lista de índices de capa: lo que no esté ahí se congela,
    que es como se entrena una cabeza sola o un adaptador. `trainable_bias` va
    aparte porque hay un método entero (BitFit) que consiste exactamente en
    mover los sesgos y nada más, y sin separarlos no se puede medir.
    """
    n = len(X)
    idx = np.arange(n)
    live = list(range(len(net.W))) if trainable is None else list(trainable)
    live_b = list(live) if trainable_bias is None else list(trainable_bias)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for ep in range(epochs):
        rng.shuffle(idx)
        tot = 0.0
        for s in range(0, n, batch):
            sel = idx[s:s + batch]
            xb, yb = X[sel], Y[sel]
            z, acts = net.forward(xb, keep=True)
            p = softmax(z / T)
            g = (p - yb) / len(sel) * (T * T) * soft_weight
            if Yhard is not None and soft_weight < 1.0:
                ph = softmax(z)
                g = g + (ph - Yhard[sel]) / len(sel) * (1.0 - soft_weight)
            tot += float(-(yb * np.log(np.clip(p, 1e-12, None))).sum() / len(sel))
            net.t += 1
            for i in range(len(net.W) - 1, -1, -1):
                gW = acts[i].T @ g
                gb = g.sum(0)
                if i > 0:
                    g = (g @ net.W[i].T) * (acts[i] > 0)
                c1 = 1 - b1 ** net.t
                c2 = 1 - b2 ** net.t
                if i in live:
                    if wd:
                        gW = gW + wd * net.W[i]
                    net.mW[i] = b1 * net.mW[i] + (1 - b1) * gW
                    net.vW[i] = b2 * net.vW[i] + (1 - b2) * gW * gW
                    net.W[i] -= lr * (net.mW[i] / c1) / (np.sqrt(net.vW[i] / c2) + eps)
                if i in live_b:
                    net.mb[i] = b1 * net.mb[i] + (1 - b1) * gb
                    net.vb[i] = b2 * net.vb[i] + (1 - b2) * gb * gb
                    net.b[i] -= lr * (net.mb[i] / c1) / (np.sqrt(net.vb[i] / c2) + eps)
        if log is not None:
            log.append(tot / max(1, n // batch))
    return net


# ------------------------------------------------------------------- LoRA
class LoraNet:
    """Una red con la matriz de una capa congelada y un añadido de rango r.

    El peso efectivo de la capa adaptada es `W0 + (alpha/r) B A`, con `A`
    inicializada al azar y `B` a cero, que es lo que hace que al empezar la red
    sea exactamente la de partida. Las dos matrices finas son lo único que se
    entrena, junto con lo que diga `extra`.
    """

    def __init__(self, base, layers, rank, rng, alpha=None, extra=()):
        self.base = base
        self.layers = list(layers)
        self.rank = rank
        self.alpha = rank if alpha is None else alpha
        self.extra = list(extra)
        self.A, self.B = {}, {}
        for i in self.layers:
            n_in, n_out = base.W[i].shape
            self.A[i] = rng.standard_normal((n_in, rank)) / np.sqrt(n_in)
            self.B[i] = np.zeros((rank, n_out))
        self._reset_adam()

    def _reset_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in
                  list(self.A.items()) + [(("b", i), self.B[i]) for i in self.layers]}
        self.v = {k: np.zeros_like(v) for k, v in self.m.items()}
        self.mW = {i: np.zeros_like(self.base.W[i]) for i in self.extra}
        self.vW = {i: np.zeros_like(self.base.W[i]) for i in self.extra}
        self.mb = {i: np.zeros_like(self.base.b[i]) for i in self.extra}
        self.vb = {i: np.zeros_like(self.base.b[i]) for i in self.extra}
        self.t = 0

    def n_trainable(self):
        n = sum(self.A[i].size + self.B[i].size for i in self.layers)
        n += sum(self.base.W[i].size + self.base.b[i].size for i in self.extra)
        return int(n)

    def eff(self, i):
        if i in self.layers:
            return self.base.W[i] + (self.alpha / self.rank) * (self.A[i] @ self.B[i])
        return self.base.W[i]

    def forward(self, X, keep=False):
        acts = [X]
        h = X
        for i in range(len(self.base.W)):
            z = h @ self.eff(i) + self.base.b[i]
            h = np.maximum(z, 0.0) if i < len(self.base.W) - 1 else z
            if keep:
                acts.append(h)
        return (h, acts) if keep else h

    def accuracy(self, X, y, batch=4096):
        ok = 0
        for i in range(0, len(X), batch):
            ok += int((np.argmax(self.forward(X[i:i + batch]), 1) == y[i:i + batch]).sum())
        return ok / len(X)

    def merged(self):
        """La red equivalente con el adaptador ya sumado, que es lo que se despliega."""
        out = self.base.copy()
        for i in self.layers:
            out.W[i] = self.eff(i)
        return out


def train_lora(net, X, Y, rng, epochs=20, batch=128, lr=1e-3):
    n = len(X)
    idx = np.arange(n)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, batch):
            sel = idx[s:s + batch]
            z, acts = net.forward(X[sel], keep=True)
            g = (softmax(z) - Y[sel]) / len(sel)
            net.t += 1
            c1 = 1 - b1 ** net.t
            c2 = 1 - b2 ** net.t
            for i in range(len(net.base.W) - 1, -1, -1):
                gW = acts[i].T @ g
                gb = g.sum(0)
                gnext = (g @ net.eff(i).T) * (acts[i] > 0) if i > 0 else None
                if i in net.layers:
                    s_ = net.alpha / net.rank
                    gA = s_ * (gW @ net.B[i].T)
                    gB = s_ * (net.A[i].T @ gW)
                    for key, grad, par in ((i, gA, net.A[i]), (("b", i), gB, net.B[i])):
                        net.m[key] = b1 * net.m[key] + (1 - b1) * grad
                        net.v[key] = b2 * net.v[key] + (1 - b2) * grad * grad
                        par -= lr * (net.m[key] / c1) / (np.sqrt(net.v[key] / c2) + eps)
                if i in net.extra:
                    net.mW[i] = b1 * net.mW[i] + (1 - b1) * gW
                    net.vW[i] = b2 * net.vW[i] + (1 - b2) * gW * gW
                    net.mb[i] = b1 * net.mb[i] + (1 - b1) * gb
                    net.vb[i] = b2 * net.vb[i] + (1 - b2) * gb * gb
                    net.base.W[i] -= lr * (net.mW[i] / c1) / (np.sqrt(net.vW[i] / c2) + eps)
                    net.base.b[i] -= lr * (net.mb[i] / c1) / (np.sqrt(net.vb[i] / c2) + eps)
                g = gnext
    return net


# ------------------------------------------------------------- cuantización
# Los dieciséis niveles de NF4, que son los cuantiles de una normal estándar
# construidos como en el artículo de QLoRA: ocho negativos, ocho positivos, el
# cero exacto dentro, y todo reescalado para que el extremo caiga en -1 y 1.
def _nf4_levels():
    from math import erf, sqrt

    def ppf(p):
        lo, hi = -8.0, 8.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1 + erf(mid / sqrt(2))) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # El reparto del artículo: ocho niveles positivos, siete negativos y el
    # cero una sola vez, que son dieciséis y no diecisiete. Contarlo mal
    # regalaría un nivel de más justo en la comparación contra los otros dos
    # formatos, que es lo único que esta función existe para medir.
    offset = 0.9677083
    pos = [ppf(p) for p in np.linspace(offset, 0.5, 9)[:-1]]
    neg = [-ppf(p) for p in np.linspace(offset, 0.5, 8)[:-1]]
    lv = np.array(sorted(pos + [0.0] + neg))
    return lv / lv.max()


NF4 = _nf4_levels()
INT4 = np.linspace(-1.0, 1.0, 16)
# Coma flotante de 4 bits, E2M1 con signo, normalizada al mayor nivel
_FP4 = sorted({0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0})
FP4 = np.array(sorted({s * v for v in _FP4 for s in (-1, 1)})) / 6.0


def quantize(W, levels, block=64):
    """Cuantiza por bloques con escala de valor absoluto máximo.

    Devuelve la matriz reconstruida, los índices y las escalas. Es la receta
    de bitsandbytes: cada bloque de `block` pesos consecutivos se divide por su
    mayor valor absoluto, se redondea al nivel más cercano y se vuelve a
    multiplicar.
    """
    flat = W.ravel().astype(np.float64)
    pad = (-len(flat)) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    blocks = flat.reshape(-1, block)
    scale = np.abs(blocks).max(1, keepdims=True)
    scale[scale == 0] = 1.0
    norm = blocks / scale
    idx = np.abs(norm[..., None] - levels[None, None, :]).argmin(-1)
    rec = (levels[idx] * scale).ravel()[:W.size].reshape(W.shape)
    return rec, idx, scale.ravel()


def bits_per_param(n, block=64, double=False, scale_block=256):
    """Los bits por peso de verdad, escalas incluidas.

    Cuatro bits son los del índice; cada bloque arrastra además una escala en
    coma flotante de 32 bits, que sobre bloques de 64 son 0,5 bits más por peso.
    La doble cuantización guarda esas escalas a 8 bits con una segunda escala
    por grupo, y ese ahorro también se cuenta aquí en vez de citarse.
    """
    n_blocks = int(np.ceil(n / block))
    bits = 4 * n
    if double:
        n_meta = int(np.ceil(n_blocks / scale_block))
        bits += 8 * n_blocks + 32 * n_meta
    else:
        bits += 32 * n_blocks
    return bits / n
