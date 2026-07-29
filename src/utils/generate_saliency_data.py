"""Las cifras del artículo de Grad-CAM y visualización de atención.

Este artículo no entrena nada. Los dos modelos que explica ya están publicados
en el atlas y ya viajan al navegador:

  * el **detector del artículo 23** (42.814 pesos en float16), que corre aquí
    en numpy desde el mismo fichero que lee la página de aquel artículo, y se
    comprueba contra las cajas que aquel artículo publicó;
  * el **codificador del artículo 48** (2.064.204 parámetros), reconstruido
    igual y comprobado contra su exactitud publicada.

Y usa una tercera cosa publicada que es la que hace todo lo demás posible: las
**máscaras exactas del artículo 24**. Aquellas escenas se componen pegando
prendas sobre un lienzo y anotando qué objeto gana cada píxel, así que existe
la máscara verdadera de cada objeto, píxel a píxel. Casi ningún artículo de
mapas de saliencia tiene eso: se mira el mapa, se dice que se parece al objeto
y se pasa a la siguiente figura. Aquí se puede **puntuar**.

Lo que se mide:

  1. **Seis mapas puntuados contra la máscara verdadera** con tres métricas
     (el juego del puntero, la energía dentro de la máscara y el solape del
     percentil superior), y dos de los seis son controles que no explican
     nada: ruido suave, y **la propia imagen de entrada**. El segundo es el
     control que hay que mirar dos veces, porque en un dataset donde el objeto
     es lo que brilla, la entrada sola puntúa como un método.
  2. **Grad-CAM sobre una cabeza 1x1 es exactamente CAM**, y eso no se cuenta,
     se comprueba: los pesos que Grad-CAM promedia son los de la cabeza, y las
     dos rutas coinciden hasta el redondeo.
  3. **La escalera de profundidad**: el mismo mapa calculado sobre los cuatro
     mapas de rasgos (64, 32, 16 y 8 píxeles de lado) para separar resolución
     de semántica, que es el compromiso del que Grad-CAM vive.
  4. **La prueba de cordura de Adebayo**, que es la que hunde a la mitad de los
     métodos publicados y no cuesta entrenar nada: se aleatorizan los pesos por
     capas, de arriba abajo, y se mide cuánto cambia el mapa. Un mapa que no se
     entera de que el modelo ya no es el modelo no está explicando al modelo.
  5. **Y en atención**: el artículo 48 midió por ablación que de sus ocho
     cabezas solo una importa. Aquí se pregunta si eso se puede **leer** en los
     mapas, correlacionando cada estadístico del mapa con la importancia
     medida; y se busca, para una frase dada, otra distribución de atención muy
     distinta que deje la salida casi igual, que es la versión medible de "la
     atención no es una explicación".

Ejecutar desde cualquier sitio: `python src/utils/generate_saliency_data.py`.
"""
import base64
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from skimage.transform import resize as skresize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_data import fashion_mnist  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "saliency" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "saliency_cache")

# --------------------------------------------------------------------------
# Las constantes del constructor de escenas son las del artículo 23, y no se
# tocan: cambiar cualquiera cambia las imágenes y el guardia lo dice.
# --------------------------------------------------------------------------
SEED = 19
CANVAS = 64
INK = 32
MIN_SIDE, MAX_SIDE = 16, 41
N_TRAIN = 6000
N_VAL = 1500
N_TEST_MAX = 2500
MAX_OBJ = 3
DET_SOURCE = [1, 3, 0, 8, 7]
HELD_SOURCE = [2, 4, 6, 5, 9]
N_CLS = 5
DIST_ID = 1000

N_IMAGES = 300          # escenas puntuadas
N_OCCLUSION = 150       # las que además pagan el barrido de oclusión
TAG = f"n{N_IMAGES}o{N_OCCLUSION}s{SEED}"

# Los nombres que se leen en la página. Fuera del caché a propósito: el TAG no
# los cubre, así que si vivieran dentro, corregir una palabra obligaría a
# repetir la oclusión o dejaría el JSON diciendo lo viejo para siempre.
LABELS = dict(gradient="plain gradient", grad_input="gradient times input",
              gradcam="Grad-CAM on the last block",
              occlusion="occlusion, 8 by 8 patches",
              input="the input image itself", random="smooth noise")


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
    """Redondeo para el JSON, con los no finitos convertidos en None.

    `json.dumps` escribe `NaN` sin protestar y eso **no es JSON**: el navegador
    lo rechaza al parsear, la promesa del fetch revienta y la página se queda
    sin prosa y sin guardia, sin un solo error visible en el sitio donde está
    el fallo. Pasó con la correlación del brazo constante, que no está definida
    porque su predicción no varía. Aquí se convierte en None y la prosa tiene
    su rama; y el volcado de abajo lleva `allow_nan=False` para que ningún otro
    campo pueda colarse.
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r(x, d) for x in np.asarray(v).tolist()]
    v = float(v)
    if not np.isfinite(v):
        return None
    return round(v, d)


# =====================================================================
#  Las escenas del 23, con las máscaras del 24
# =====================================================================
def iou_xyxy(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def resize_tile(img_u8, s):
    rr = skresize(img_u8.astype(np.float64) / 255.0, (s, s), order=1, anti_aliasing=True)
    return np.clip(np.round(rr * 255.0), 0, 255).astype(np.uint8)


def ink_box(tile):
    m = tile >= INK
    if not m.any():
        return None
    rr = np.where(m.any(1))[0]
    c = np.where(m.any(0))[0]
    return (int(c[0]), int(rr[0]), int(c[-1]) + 1, int(rr[-1]) + 1)


def sprite_for(img_u8, target_long):
    native = ink_box(img_u8)
    if native is None:
        return None, None
    long_native = max(native[2] - native[0], native[3] - native[1])
    s = int(round(target_long * img_u8.shape[0] / max(long_native, 1)))
    s = int(np.clip(s, 8, CANVAS))
    tile = resize_tile(img_u8, s)
    box = ink_box(tile)
    if box is None or box[2] <= box[0] or box[3] <= box[1]:
        return None, None
    return tile, box


def class_pools(labels, source_ids, lo=0.0, hi=1.0):
    pools = []
    for c in source_ids:
        idx = np.where(labels == c)[0]
        pools.append(idx[int(len(idx) * lo):int(len(idx) * hi)])
    return pools


def synth_split(n, images, pools, dpools, base, cap, crowd_frac, crowd_cap):
    """El constructor de escenas de los artículos 23 y 24, copiado.

    Copiado y no importado, porque aquellos ficheros importan torch en la
    cabecera y esta sesión no lo necesita para nada. Lo que importa es que **no
    se mueve ni una llamada al generador de números aleatorios**: la secuencia
    es la misma, así que las imágenes son las mismas, y eso no se declara, se
    comprueba contra las cifras que aquellos artículos publicaron.
    """
    canvases = np.zeros((n, CANVAS, CANVAS), np.uint8)
    owners = np.full((n, CANVAS, CANVAS), -1, np.int16)
    rows, vis = [], []
    for i in range(n):
        rng = np.random.default_rng(SEED * 100003 + base + i)
        is_crowd = int(rng.random() < crowd_frac)
        cap_i = crowd_cap if is_crowd else cap
        cv = rng.integers(0, 21, (CANVAS, CANVAS)).astype(np.uint8)
        n_obj = (int(rng.integers(2, MAX_OBJ + 1)) if is_crowd
                 else int(rng.integers(1, MAX_OBJ + 1)))
        placed = []
        for _ in range(n_obj):
            c = int(rng.integers(0, N_CLS))
            tile, bx = sprite_for(images[int(rng.choice(pools[c]))],
                                  int(rng.integers(MIN_SIDE, MAX_SIDE)))
            if bx is None:
                continue
            s = tile.shape[0]
            for _ in range(40):
                ox = int(rng.integers(0, CANVAS - s + 1))
                oy = int(rng.integers(0, CANVAS - s + 1))
                box = (ox + bx[0], oy + bx[1], ox + bx[2], oy + bx[3])
                if all(iou_xyxy(box, p[1]) <= cap_i for p in placed):
                    placed.append((c, box, tile, ox, oy))
                    break
        dist = []
        for _ in range(int(rng.integers(0, 4))):
            tile, bx = sprite_for(images[int(rng.choice(dpools[int(rng.integers(0, 5))]))],
                                  int(rng.integers(MIN_SIDE, MAX_SIDE)))
            if bx is None:
                continue
            s = tile.shape[0]
            for _ in range(40):
                ox = int(rng.integers(0, CANVAS - s + 1))
                oy = int(rng.integers(0, CANVAS - s + 1))
                box = (ox + bx[0], oy + bx[1], ox + bx[2], oy + bx[3])
                if (all(iou_xyxy(box, p[1]) <= 0.1 for p in placed)
                        and all(iou_xyxy(box, d[1]) <= cap_i for d in dist)):
                    dist.append((-1, box, tile, ox, oy))
                    break
        own = np.full((CANVAS, CANVAS), -1, np.int16)
        for k, (_c, _b, tile, ox, oy) in enumerate(placed + dist):
            s = tile.shape[0]
            sub = cv[oy:oy + s, ox:ox + s]
            take = (tile >= INK) & (tile > sub)
            osub = own[oy:oy + s, ox:ox + s]
            osub[take] = k if k < len(placed) else DIST_ID + (k - len(placed))
            cv[oy:oy + s, ox:ox + s] = np.maximum(sub, tile)
        canvases[i] = cv
        owners[i] = own
        for k, (c, box, tile, ox, oy) in enumerate(placed):
            s = tile.shape[0]
            m = tile >= INK
            sub = cv[oy:oy + s, ox:ox + s]
            vis.append(float((sub[m] == tile[m]).mean()) if m.any() else 0.0)
            rows.append([i, c, box[0], box[1], box[2], box[3], int(m.sum()),
                         int((own == k).sum())])
    return dict(canvas=canvases, owner=owners,
                boxes=np.array(rows, np.float32).reshape(-1, 8),
                vis=np.array(vis, np.float32))


def build_scenes():
    Ftr_x, Ftr_y, Fte_x, Fte_y = fashion_mnist()
    te_pool = class_pools(Fte_y, DET_SOURCE, 0.5, 1.0)
    te_dpool = class_pools(Fte_y, HELD_SOURCE, 0.5, 1.0)
    return synth_split(N_TEST_MAX, Fte_x, te_pool, te_dpool, 2000000, 0.35, 0.20, 0.60)


# =====================================================================
#  El detector del 23, en numpy, desde su propio fichero
# =====================================================================
class Detector:
    def __init__(self, path):
        d = json.loads(Path(path).read_text())
        raw = np.frombuffer(base64.b64decode(d["weights_b64"]), dtype="<f2")
        self.raw = raw.astype(np.float64)
        self.spec = d["layers"]
        self.arch = d["arch"]
        self.live = d["live"]
        self.W, self.B = [], []
        for L in self.spec:
            w = self.raw[L["w_offset"]:L["w_offset"] + L["w_size"]].reshape(L["shape"])
            b = self.raw[L["b_offset"]:L["b_offset"] + L["b_size"]]
            self.W.append(w.copy())
            self.B.append(b.copy())
        self.pool_after = set(self.arch["pool_after"])
        self.anchors = np.array(self.arch["anchors"], dtype=np.float64)
        self.stride = self.arch["stride"]
        self.classes = self.arch["classes"]

    # ---- convolución por im2col, escrita a mano para poder derivarla
    @staticmethod
    def im2col(x, k, pad):
        C, H, Wd = x.shape
        xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
        cols = np.empty((C * k * k, H * Wd))
        idx = 0
        for c in range(C):
            for i in range(k):
                for j in range(k):
                    cols[idx] = xp[c, i:i + H, j:j + Wd].reshape(-1)
                    idx += 1
        return cols

    @staticmethod
    def col2im(cols, shape, k, pad):
        C, H, Wd = shape
        xp = np.zeros((C, H + 2 * pad, Wd + 2 * pad))
        idx = 0
        for c in range(C):
            for i in range(k):
                for j in range(k):
                    xp[c, i:i + H, j:j + Wd] += cols[idx].reshape(H, Wd)
                    idx += 1
        return xp[:, pad:pad + H, pad:pad + Wd] if pad else xp

    def forward(self, img, keep=False, weights=None):
        """img: (64, 64) uint8. Devuelve la salida cruda (30, 8, 8)."""
        Ws = self.W if weights is None else weights
        x = img.astype(np.float64)[None, :, :] / 255.0
        acts, pools = [], []
        for li, L in enumerate(self.spec):
            k = L["shape"][2]
            pad = 1 if k == 3 else 0
            cols = self.im2col(x, k, pad)
            out = (Ws[li].reshape(L["shape"][0], -1) @ cols + self.B[li][:, None])
            out = out.reshape(L["shape"][0], x.shape[1], x.shape[2])
            pre = out
            if li < len(self.spec) - 1:
                out = np.maximum(out, 0.0)
            if li in self.pool_after:
                C, H, Wd = out.shape
                v = out.reshape(C, H // 2, 2, Wd // 2, 2)
                flat = v.transpose(0, 1, 3, 2, 4).reshape(C, H // 2, Wd // 2, 4)
                arg = flat.argmax(-1)
                out = flat.max(-1)
                pools.append((li, arg, (C, H, Wd)))
            if keep:
                acts.append(dict(layer=li, x_in_shape=x.shape, pre=pre, out=out))
            x = out
        return (x, acts, pools) if keep else x

    def grad_to(self, img, cell, anchor, cls, target_layer):
        """Gradiente de la puntuación respecto a la salida de `target_layer`.

        La puntuación que se explica es la suma de dos logits, el de objeto y
        el de la clase, en una celda y un ancla concretas. Se dice cuál porque
        Grad-CAM no tiene una sola definición: explicar la probabilidad después
        del softmax, el logit, o la puntuación final da tres mapas distintos y
        casi ningún paper escribe cuál usó.
        """
        raw, acts, pools = self.forward(img, keep=True)
        C = self.arch["n_classes"]
        row, col = cell
        base = anchor * (5 + C)
        g_out = np.zeros_like(raw)
        g_out[base + 4, row, col] = 1.0
        g_out[base + 5 + cls, row, col] = 1.0

        g = g_out
        for li in range(len(self.spec) - 1, target_layer, -1):
            L = self.spec[li]
            k = L["shape"][2]
            pad = 1 if k == 3 else 0
            # deshacer el pool si esta capa lo tenía
            for (pl, arg, shape) in pools:
                if pl == li:
                    C_, H, Wd = shape
                    up = np.zeros((C_, H // 2, Wd // 2, 4))
                    ii, jj, kk = np.meshgrid(np.arange(C_), np.arange(H // 2),
                                             np.arange(Wd // 2), indexing="ij")
                    up[ii, jj, kk, arg] = g
                    g = (up.reshape(C_, H // 2, Wd // 2, 2, 2)
                         .transpose(0, 1, 3, 2, 4).reshape(C_, H, Wd))
            if li < len(self.spec) - 1:
                g = g * (acts[li]["pre"] > 0)
            cols = (self.W[li].reshape(L["shape"][0], -1).T
                    @ g.reshape(L["shape"][0], -1))
            g = self.col2im(cols, acts[li]["x_in_shape"], k, pad)
        # y el pool de la propia capa objetivo, si lo tiene
        for (pl, arg, shape) in pools:
            if pl == target_layer:
                C_, H, Wd = shape
                up = np.zeros((C_, H // 2, Wd // 2, 4))
                ii, jj, kk = np.meshgrid(np.arange(C_), np.arange(H // 2),
                                         np.arange(Wd // 2), indexing="ij")
                up[ii, jj, kk, arg] = g
                g = (up.reshape(C_, H // 2, Wd // 2, 2, 2)
                     .transpose(0, 1, 3, 2, 4).reshape(C_, H, Wd))
        return g, acts, raw

    def grad_input(self, img, cell, anchor, cls):
        g, _, raw = self.grad_to(img, cell, anchor, cls, -1)
        return g[0], raw

    def cam(self, img, cell, anchor, cls, layer=3):
        """Grad-CAM sobre el mapa de rasgos de `layer`."""
        g, acts, raw = self.grad_to(img, cell, anchor, cls, layer)
        A = acts[layer]["out"]
        alpha = g.mean(axis=(1, 2))
        m = np.maximum((alpha[:, None, None] * A).sum(0), 0.0)
        return m, alpha, A, raw

    def decode(self, raw):
        """Las cajas, con la fórmula que el propio fichero declara."""
        C = self.arch["n_classes"]
        G = self.arch["grid"]
        out = []
        for a in range(len(self.anchors)):
            b = a * (5 + C)
            for rr in range(G):
                for cc in range(G):
                    t = raw[b:b + 4, rr, cc]
                    obj = 1 / (1 + np.exp(-raw[b + 4, rr, cc]))
                    lo = raw[b + 5:b + 5 + C, rr, cc]
                    p = np.exp(lo - lo.max())
                    p /= p.sum()
                    x = (cc + 1 / (1 + np.exp(-t[0]))) * self.stride
                    y = (rr + 1 / (1 + np.exp(-t[1]))) * self.stride
                    w = self.anchors[a, 0] * np.exp(np.clip(t[2], -4, 4))
                    h = self.anchors[a, 1] * np.exp(np.clip(t[3], -4, 4))
                    out.append(dict(a=a, r=rr, c=cc, score=float(obj * p.max()),
                                    cls=int(p.argmax()),
                                    box=[float(np.clip(v, 0, CANVAS)) for v in
                                         (x - w / 2, y - h / 2, x + w / 2, y + h / 2)]))
        return out


# =====================================================================
#  Los mapas
# =====================================================================
def smooth_random(rng, size=CANVAS, scale=8):
    """Ruido suave: el control que cualquier método tiene que batir."""
    small = rng.random((size // scale, size // scale))
    return skresize(small, (size, size), order=1)


def upsample(m, size=CANVAS):
    return skresize(m, (size, size), order=1)


def saliency_maps(det, img, cell, anchor, cls, rng, do_occlusion=True):
    maps = {}
    g_in, raw = det.grad_input(img, cell, anchor, cls)
    maps["gradient"] = np.abs(g_in)
    maps["grad_input"] = np.abs(g_in * (img.astype(np.float64) / 255.0))
    cam, alpha, A, _ = det.cam(img, cell, anchor, cls, layer=3)
    maps["gradcam"] = upsample(cam)
    maps["input"] = img.astype(np.float64) / 255.0
    maps["random"] = smooth_random(rng)
    if do_occlusion:
        base = float(raw[anchor * (5 + det.arch["n_classes"]) + 4, cell[0], cell[1]])
        occ = np.zeros((CANVAS // 8, CANVAS // 8))
        for i in range(CANVAS // 8):
            for j in range(CANVAS // 8):
                patched = img.copy()
                patched[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = 10
                rr = det.forward(patched)
                occ[i, j] = base - float(
                    rr[anchor * (5 + det.arch["n_classes"]) + 4, cell[0], cell[1]])
        maps["occlusion"] = upsample(np.maximum(occ, 0))
    return maps, cam, alpha, A


def score_map(m, mask):
    """Tres métricas contra la máscara exacta.

    - el juego del puntero: si el máximo cae dentro
    - energía: qué fracción de la masa del mapa cae dentro
    - solape: se cogen tantos píxeles como tiene la máscara, los de mayor
      valor, y se mide el solape. Coger exactamente ese número es lo que hace
      la cifra comparable entre mapas con escalas distintas.
    """
    m = np.asarray(m, dtype=float)
    if m.max() > m.min():
        m = (m - m.min()) / (m.max() - m.min())
    k = int(mask.sum())
    flat = m.ravel()
    top = np.argpartition(-flat, k)[:k]
    sel = np.zeros(m.size, dtype=bool)
    sel[top] = True
    sel = sel.reshape(m.shape)
    inter = float((sel & mask).sum())
    return dict(
        pointing=float(mask.ravel()[int(np.argmax(flat))]),
        energy=float(m[mask].sum() / max(m.sum(), 1e-12)),
        overlap=inter / max(k, 1),
        iou=inter / max(float((sel | mask).sum()), 1),
    )


def pick_targets(scenes, det, n):
    """Para cada escena, el objeto mayor y la celda donde el detector lo ve."""
    per_img = {}
    for row in scenes["boxes"]:
        i = int(row[0])
        per_img.setdefault(i, []).append(row)
    out = []
    for i in sorted(per_img)[:n * 3]:
        rows = per_img[i]
        best = max(rows, key=lambda rr: rr[7])       # más píxeles visibles
        if best[7] < 60:
            continue
        k = [id(x) for x in rows].index(id(best))
        mask = scenes["owner"][i] == k
        if mask.sum() < 60:
            continue
        cx = (best[2] + best[4]) / 2
        cy = (best[3] + best[5]) / 2
        cell = (int(min(7, cy // det.stride)), int(min(7, cx // det.stride)))
        raw = det.forward(scenes["canvas"][i])
        C = det.arch["n_classes"]
        best_a, best_s = 0, -1e9
        for a in range(len(det.anchors)):
            s = raw[a * (5 + C) + 4, cell[0], cell[1]]
            if s > best_s:
                best_a, best_s = a, s
        out.append(dict(i=i, cell=cell, anchor=best_a, cls=int(best[1]),
                        mask=mask, box=[float(v) for v in best[2:6]],
                        visible=int(best[7])))
        if len(out) >= n:
            break
    return out


def stage_scores(det, scenes, targets):
    methods = ["gradient", "grad_input", "gradcam", "occlusion", "input", "random"]
    acc = {m: {k: [] for k in ["pointing", "energy", "overlap", "iou"]} for m in methods}
    for t in targets:
        rng = np.random.default_rng(SEED + 1000 + t["i"])
        img = scenes["canvas"][t["i"]]
        do_occ = len([1 for x in acc["occlusion"]["pointing"]]) < N_OCCLUSION
        maps, *_ = saliency_maps(det, img, t["cell"], t["anchor"], t["cls"], rng,
                                 do_occlusion=do_occ)
        for m in methods:
            if m not in maps:
                continue
            s = score_map(maps[m], t["mask"])
            for k, v in s.items():
                acc[m][k].append(v)
    rows = []
    for m in methods:
        rows.append(dict(key=m, label=LABELS[m], n=len(acc[m]["pointing"]),
                         **{k: float(np.mean(v)) for k, v in acc[m].items()},
                         **{f"{k}_se": float(np.std(v, ddof=1) / np.sqrt(len(v)))
                            for k, v in acc[m].items()}))
    return dict(rows=rows, n=len(targets), occlusion_n=N_OCCLUSION)


def stage_depth(det, scenes, targets):
    """El mismo mapa en las cuatro profundidades: resolución contra semántica."""
    rows = []
    for layer, side in [(0, 32), (1, 16), (2, 8), (3, 8)]:
        acc = {k: [] for k in ["pointing", "energy", "overlap"]}
        for t in targets[:150]:
            img = scenes["canvas"][t["i"]]
            cam, _, A, _ = det.cam(img, t["cell"], t["anchor"], t["cls"], layer=layer)
            s = score_map(upsample(cam), t["mask"])
            for k in acc:
                acc[k].append(s[k])
        rows.append(dict(layer=layer, side=int(side), channels=det.spec[layer]["shape"][0],
                         **{k: float(np.mean(v)) for k, v in acc.items()}))
    return rows


def stage_cam_identity(det, scenes, targets):
    """Grad-CAM sobre una cabeza 1x1 ES CAM, comprobado y no contado.

    Los pesos que Grad-CAM promedia son el gradiente de la puntuación respecto
    al mapa de rasgos, y con una cabeza de 1x1 ese gradiente solo es no nulo en
    una celda y vale exactamente la fila de la cabeza. Promediarlo sobre las
    8x8 celdas lo divide por 64, así que la CAM de libro y la de Grad-CAM son
    proporcionales con constante conocida. Se comprueba la proporción.
    """
    worst, ratios = 0.0, []
    C = det.arch["n_classes"]
    for t in targets[:40]:
        img = scenes["canvas"][t["i"]]
        cam, alpha, A, _ = det.cam(img, t["cell"], t["anchor"], t["cls"], layer=3)
        b = t["anchor"] * (5 + C)
        w = det.W[4][b + 4, :, 0, 0] + det.W[4][b + 5 + t["cls"], :, 0, 0]
        direct = np.maximum((w[:, None, None] * A).sum(0), 0.0)
        g = direct / 64.0
        d = float(np.abs(g - cam).max() / max(cam.max(), 1e-12))
        worst = max(worst, d)
        ratios.append(float(np.abs(alpha * 64 - w).max()))
    return dict(worst_relative=worst, worst_alpha=float(max(ratios)), n=40,
                factor=64)


def stage_sanity(det, scenes, targets):
    """La prueba de Adebayo: aleatorizar de arriba abajo y ver quién se entera."""
    rng = np.random.default_rng(SEED + 7)
    methods = ["gradient", "grad_input", "gradcam", "input"]
    rows = []
    base_maps = {}
    for t in targets[:60]:
        img = scenes["canvas"][t["i"]]
        maps, *_ = saliency_maps(det, img, t["cell"], t["anchor"], t["cls"],
                                 np.random.default_rng(1), do_occlusion=False)
        base_maps[t["i"]] = {m: maps[m] for m in methods}
    for cut in [4, 3, 2, 1, 0]:
        Ws = [w.copy() for w in det.W]
        for li in range(cut, len(Ws)):
            sd = float(det.W[li].std())
            Ws[li] = rng.standard_normal(Ws[li].shape) * sd
        old = det.W
        det.W = Ws
        acc = {m: [] for m in methods}
        for t in targets[:60]:
            img = scenes["canvas"][t["i"]]
            maps, *_ = saliency_maps(det, img, t["cell"], t["anchor"], t["cls"],
                                     np.random.default_rng(1), do_occlusion=False)
            for m in methods:
                a = base_maps[t["i"]][m].ravel()
                b = maps[m].ravel()
                if np.std(b) < 1e-12 or np.std(a) < 1e-12:
                    acc[m].append(1.0 if np.allclose(a, b) else 0.0)
                else:
                    acc[m].append(float(spearmanr(a, b).statistic))
        det.W = old
        rows.append(dict(randomised_from=cut,
                         **{m: float(np.mean(v)) for m, v in acc.items()}))
    return rows


# =====================================================================
#  La atención del 48
# =====================================================================
class Encoder:
    """El codificador del artículo 48, reconstruido en numpy desde su blob."""

    def __init__(self, path):
        d = json.loads(Path(path).read_text())
        self.data = d
        m = d["model"]
        raw = np.frombuffer(base64.b64decode(m["blob"]), dtype="<f2").astype(np.float64)
        self.p = {k: raw[v["offset"]:v["offset"] + int(np.prod(v["shape"]))]
                  .reshape(v["shape"]).copy() for k, v in m["spec"].items()}
        self.d_model = m["d_model"]
        self.heads = m["heads"]
        self.layers = m["layers"]
        self.dk = self.d_model // self.heads
        self.words = m["words"]
        self.tags = m["tags"]
        self.sentences = m["sentences"]

    @staticmethod
    def ln(x, w, b):
        mu = x.mean(-1, keepdims=True)
        v = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(v + 1e-5) * w + b

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def forward(self, ids, drop=None, override=None):
        """drop: (capa, cabeza) a poner a cero, como la ablación del 48.
        override: {(capa, cabeza): matriz} para sustituir la atención entera."""
        n = len(ids)
        h = self.p["emb.weight"][ids] + self.p["pos.weight"][:n]
        attn = {}
        for L in range(self.layers):
            pre = f"blocks.{L}."
            x = self.ln(h, self.p[pre + "n1.weight"], self.p[pre + "n1.bias"])
            q = x @ self.p[pre + "att.q.weight"].T + self.p[pre + "att.q.bias"]
            k = x @ self.p[pre + "att.k.weight"].T + self.p[pre + "att.k.bias"]
            v = x @ self.p[pre + "att.v.weight"].T + self.p[pre + "att.v.bias"]
            out = np.zeros_like(x)
            for hd in range(self.heads):
                sl = slice(hd * self.dk, (hd + 1) * self.dk)
                s = q[:, sl] @ k[:, sl].T / np.sqrt(self.dk)
                a = np.exp(s - s.max(-1, keepdims=True))
                a /= a.sum(-1, keepdims=True)
                if override and (L, hd) in override:
                    a = override[(L, hd)]
                attn[(L, hd)] = a
                if drop and (L, hd) == drop:
                    continue
                out[:, sl] = a @ v[:, sl]
            h = h + out @ self.p[pre + "att.o.weight"].T + self.p[pre + "att.o.bias"]
            x2 = self.ln(h, self.p[pre + "n2.weight"], self.p[pre + "n2.bias"])
            ff = self.gelu(x2 @ self.p[pre + "ff.0.weight"].T + self.p[pre + "ff.0.bias"])
            h = h + ff @ self.p[pre + "ff.2.weight"].T + self.p[pre + "ff.2.bias"]
        z = self.ln(h, self.p["norm.weight"], self.p["norm.bias"])
        return z @ self.p["head.weight"].T + self.p["head.bias"], attn


def stage_attention():
    enc = Encoder(ROOT / "attention" / "data" / "attention.json")
    d = enc.data
    sents = enc.sentences

    # Las etiquetas de oro vienen como NOMBRES de categoría, no como índices, y
    # comparar un entero con una cadena en numpy no falla: devuelve False en
    # todas las posiciones, o sea exactitud cero, que es exactamente el aspecto
    # de un modelo mal reconstruido. Se traducen por la misma lista de
    # etiquetas que el fichero publica.
    tag_index = {name: k for k, name in enumerate(enc.tags)}
    gold_of = lambda s: np.array([tag_index[g] for g in s["gold"]])

    # Primero: el modelo reconstruido tiene que acertar lo que el 48 dice que
    # acierta en SUS frases de ejemplo, o no es su modelo.
    ok, tot = 0, 0
    for s in sents:
        logits, _ = enc.forward(s["ids"])
        ok += int((logits.argmax(1) == gold_of(s)).sum())
        tot += len(s["gold"])
    demo_acc = ok / tot

    # La importancia por ablación, en dos resoluciones muy distintas.
    #
    # El 48 publica la caída de exactitud (`drop`, positiva) medida sobre sus
    # 5.734 frases de test. Aquí sólo hay las ocho frases de ejemplo que el
    # fichero publica, 95 etiquetas, así que la ablación medida en esta página
    # sólo puede moverse en múltiplos de 1/95 = 0,0105 y las caídas publicadas
    # valen entre 0,0005 y 0,031: por debajo de la resolución.
    #
    # Esto empezó siendo un guardia ("mi ablación tiene que coincidir con la
    # suya") y disparó en cinco cabezas de ocho. Tenía razón: la de 95 fichas
    # no mide lo mismo. La comparación se queda, pero como resultado, porque la
    # correlación con una y con otra sale del derecho y del revés.
    published = {(row["layer"], row["head"]): row["drop"]
                 for row in d["ablation"]["rows"]}
    mine = {}
    for L in range(enc.layers):
        for hd in range(enc.heads):
            good = 0
            for s in sents:
                logits, _ = enc.forward(s["ids"], drop=(L, hd))
                good += int((logits.argmax(1) == gold_of(s)).sum())
            mine[(L, hd)] = good / tot - demo_acc

    # Y los estadísticos del mapa que la gente mira: entropía, cuánto se mira a
    # sí mismo cada token, y cuánto pesa el vecino inmediato.
    rows = []
    for (L, hd), delta in mine.items():
        ents, selfs, neigh, maxw = [], [], [], []
        for s in sents:
            _, attn = enc.forward(s["ids"])
            a = attn[(L, hd)]
            ents.append(float(-(a * np.log(a + 1e-12)).sum(1).mean()))
            selfs.append(float(np.diag(a).mean()))
            n = len(s["ids"])
            neigh.append(float(np.mean([a[i, max(0, i - 1)] for i in range(n)])))
            maxw.append(float(a.max(1).mean()))
        rows.append(dict(layer=L, head=hd, ablation=float(delta),
                         entropy=float(np.mean(ents)), self=float(np.mean(selfs)),
                         previous=float(np.mean(neigh)), peak=float(np.mean(maxw)),
                         published=(published.get((L, hd)) if published else None)))
    # `corrs` es contra la medición publicada, que es la buena. `corrs_small`
    # es contra la de ocho frases, y está para enseñar hasta dónde se llega
    # midiendo la importancia con lo que cabe en un fichero de ejemplo.
    stats = ["entropy", "self", "previous", "peak"]
    imp = np.array([published[(row["layer"], row["head"])] for row in rows])
    small = np.array([-row["ablation"] for row in rows])
    corrs = {k: float(spearmanr([row[k] for row in rows], imp).statistic) for k in stats}
    corrs_small = {k: float(spearmanr([row[k] for row in rows], small).statistic)
                   for k in stats}
    assert len(set(np.round(small, 6))) < len(rows), \
        "la ablación de ocho frases distingue las ocho cabezas, y entonces no hay empates"

    # "La atención no es una explicación", en su forma medible: buscar otra
    # distribución muy distinta que deje la salida casi igual.
    rng = np.random.default_rng(SEED + 11)
    s = sents[0]
    logits, attn = enc.forward(s["ids"])
    pred = logits.argmax(1)
    best = None
    for _ in range(4000):
        L, hd = 0, int(rng.integers(enc.heads))
        a = attn[(L, hd)].copy()
        n = a.shape[0]
        # una permutación aleatoria de cada fila: misma masa, otro reparto
        for i in range(n):
            a[i] = a[i][rng.permutation(n)]
        lo, _ = enc.forward(s["ids"], override={(L, hd): a})
        same = float((lo.argmax(1) == pred).mean())
        tv = float(0.5 * np.abs(a - attn[(L, hd)]).sum(1).mean())
        if same == 1.0 and (best is None or tv > best["tv"]):
            best = dict(layer=L, head=hd, tv=tv, same=same,
                        logit_shift=float(np.abs(lo - logits).max()))
    return dict(
        demo_acc=demo_acc, tokens=tot, sentences=len(sents),
        published_acc=d["acc"], rows=rows, corrs=corrs, corrs_small=corrs_small,
        published_n=d["meta"]["test_sents"], published_base=d["ablation"]["base"],
        top_head=max(rows, key=lambda rr: published[(rr["layer"], rr["head"])]),
        top_peak=max(rows, key=lambda rr: rr["peak"]),
        adversarial=best,
        example=dict(words=[enc.words[i] for i in sents[0]["ids"]],
                     gold=list(sents[0]["gold"]),
                     attn={f"{L}-{hd}": r(enc.forward(sents[0]["ids"])[1][(L, hd)], 5)
                           for L in range(enc.layers) for hd in range(enc.heads)}),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    det = Detector(ROOT / "detection" / "data" / "detector.json")
    scenes = cached("scenes", build_scenes)

    # ---- la afirmación de reutilización, comprobada contra lo publicado
    pc = det.live["pixel_check"]
    idx = det.live["rows"][pc["index"]]["index"] if isinstance(
        det.live["rows"][pc["index"]], dict) else pc["index"]
    got = scenes["canvas"][idx][pc["row"]].tolist()
    same_row = got == pc["values"]
    print(f"  scenes rebuilt: {len(scenes['canvas'])} canvases, "
          f"{len(scenes['boxes'])} boxes")
    print(f"  row {pc['row']} of canvas {idx} matches the detector export: {same_row}")
    assert same_row, ("las escenas reconstruidas no son las del artículo 23: "
                      f"{got[:8]} contra {pc['values'][:8]}")

    # ---- y el forward en numpy contra las cajas que aquel artículo publicó
    row0 = det.live["rows"][0]
    raw = det.forward(scenes["canvas"][row0["index"]])
    dec = det.decode(raw)
    dec.sort(key=lambda dd: -dd["score"])
    pub = row0["decoded"]
    worst_box = 0.0
    if pub and isinstance(pub[0], (list, tuple)):
        for got_d, want in zip(dec[:len(pub)], pub):
            worst_box = max(worst_box, max(abs(a - b) for a, b in
                                           zip(got_d["box"], list(want)[:4])))
    print(f"  numpy forward vs the published decode: worst corner {worst_box:.4f} px")

    targets = cached("targets", lambda: pick_targets(scenes, det, N_IMAGES))
    print(f"  {len(targets)} scenes with a target object of at least 60 visible pixels")

    scores = cached("scores", lambda: stage_scores(det, scenes, targets))
    for row in scores["rows"]:
        print(f"    {row['label']:<28} pointing {row['pointing']:.3f}  "
              f"energy {row['energy']:.3f}  overlap {row['overlap']:.3f}")
    depth = cached("depth", lambda: stage_depth(det, scenes, targets))
    for row in depth:
        print(f"    layer {row['layer']} ({row['side']}x{row['side']}, "
              f"{row['channels']} ch): pointing {row['pointing']:.3f}  "
              f"overlap {row['overlap']:.3f}")
    ident = cached("identity", lambda: stage_cam_identity(det, scenes, targets))
    print(f"  Grad-CAM equals CAM to {ident['worst_relative']:.2e} relative")
    sanity = cached("sanity", lambda: stage_sanity(det, scenes, targets))
    for row in sanity:
        print(f"    randomised from layer {row['randomised_from']}: "
              + "  ".join(f"{k} {row[k]:+.3f}" for k in
                          ["gradient", "grad_input", "gradcam", "input"]))
    att = cached("attention", stage_attention)
    print(f"  encoder rebuilt: {att['demo_acc']:.4f} on the demo sentences "
          f"({att['tokens']} tokens); published overall {att['published_acc']:.4f}")
    print(f"  map statistic vs the drop published over {att['published_n']} test sentences: " +
          ", ".join(f"{k} {v:+.3f}" for k, v in att["corrs"].items()))
    print(f"  the same against the ablation these {att['sentences']} sentences can resolve: " +
          ", ".join(f"{k} {v:+.3f}" for k, v in att["corrs_small"].items()))
    if att["adversarial"]:
        print(f"  a head's attention can be moved {att['adversarial']['tv']:.3f} "
              f"in total variation with every tag unchanged")

    by = {row["key"]: row for row in scores["rows"]}
    assert by["gradcam"]["pointing"] > by["random"]["pointing"], \
        "Grad-CAM no bate al ruido, y entonces algo está roto"
    assert ident["worst_relative"] < 1e-6, "Grad-CAM y CAM no coinciden"
    assert att["demo_acc"] > 0.8, "el codificador reconstruido no etiqueta"

    data = dict(
        meta=dict(seed=SEED, canvas=CANVAS, n_images=len(targets),
                  occlusion_n=N_OCCLUSION,
                  detector=dict(weights=int(det.raw.size), classes=det.classes,
                                stride=det.stride, grid=det.arch["grid"]),
                  reuse=dict(row=pc["row"], canvas=int(idx), matches=bool(same_row),
                             worst_box=r(worst_box, 5),
                             boxes=int(len(scenes["boxes"])),
                             canvases=int(len(scenes["canvas"]))),
                  score="the objectness logit plus the class logit, at one cell "
                        "and one anchor"),
        # Las etiquetas se vuelven a poner aquí y no se leen del caché: viven
        # en LABELS, y una corrección de redacción no puede obligar a repetir
        # la pasada de oclusión, que es la cara.
        scores=dict(n=scores["n"], occlusion_n=scores["occlusion_n"],
                    rows=[{**{k: (v if isinstance(v, (str, int)) else r(v))
                              for k, v in row.items()},
                           "label": LABELS[row["key"]]} for row in scores["rows"]]),
        depth=[{k: r(v) for k, v in row.items()} for row in depth],
        identity={k: r(v) for k, v in ident.items()},
        sanity=[{k: r(v) for k, v in row.items()} for row in sanity],
        attention=dict(
            demo_acc=r(att["demo_acc"]), tokens=att["tokens"],
            sentences=att["sentences"], published_acc=r(att["published_acc"]),
            published_n=int(att["published_n"]), published_base=r(att["published_base"]),
            top_head=dict(layer=att["top_head"]["layer"], head=att["top_head"]["head"]),
            top_peak=dict(layer=att["top_peak"]["layer"], head=att["top_peak"]["head"]),
            rows=[{k: (v if isinstance(v, int) or v is None else r(v))
                   for k, v in row.items()} for row in att["rows"]],
            corrs={k: r(v) for k, v in att["corrs"].items()},
            corrs_small={k: r(v) for k, v in att["corrs_small"].items()},
            adversarial=({k: r(v) for k, v in att["adversarial"].items()}
                         if att["adversarial"] else None),
            example=att["example"]),
    )
    path = OUT / "saliency.json"
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
