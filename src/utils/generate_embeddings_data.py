"""Las cifras del articulo de vectores de palabras.

Corpus: el Brown, 1.161.192 palabras en 57.340 frases, y se elige por una razon
que ningun otro corpus de este modulo da: **viene etiquetado con categoria
gramatical**. Eso convierte en medible la afirmacion mas repetida y menos
comprobada sobre embeddings, que la ventana decide que significa "parecido". Con
etiquetas se puede preguntar que fraccion de los diez vecinos de una palabra
comparten su categoria, y ver esa fraccion caer segun la ventana se ensancha, en
vez de mirar dos listas de palabras y opinar.

Tres modelos sobre el mismo corpus, el mismo vocabulario y el mismo presupuesto:
skip-gram con muestreo negativo, GloVe sobre la matriz de coocurrencia, y
FastText, que es el primero mas n-gramas de caracteres. Comparten todo lo que se
puede compartir para que lo que los separa sea lo unico que los separa.

Y la afirmacion que se puede comprobar exactamente, que es la mejor de las tres
cosas que este articulo mide: Levy y Goldberg demostraron que skip-gram con
muestreo negativo esta factorizando implicitamente la matriz de PMI desplazada.
Eso no es una intuicion, es una igualdad, y aqui se mide como tal.

Ejecutar desde cualquier sitio: `python src/utils/generate_embeddings_data.py`.
"""
import base64
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlp_data import brown  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "embeddings" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "embeddings_cache")

SEED = 19
# 64 y no 128 por una razon que no es de modelado: los tres modelos viajan
# enteros al navegador para que el lector pueda pedir los vecinos de cualquier
# palabra en vez de una lista precocinada, y 900 palabras por 64 dimensiones por
# tres modelos en float16 son unos 345 kB, que es lo que este sitio se permite.
# Con 1,16 millones de tokens, 64 dimensiones tampoco se quedan cortas.
DIM = 64
MIN_COUNT = 12
NEG = 5
EPOCHS = 5
SUBSAMPLE = 1e-4


def cached(name, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.is_file():
        with path.open("rb") as fh:
            return pickle.load(fh)
    val = fn()
    with path.open("wb") as fh:
        pickle.dump(val, fh)
    return val


def b64_f16(a):
    return base64.b64encode(np.asarray(a, dtype="<f2").tobytes()).decode()


def b64_u16(a):
    return base64.b64encode(np.asarray(a, dtype="<u2").tobytes()).decode()


def threads():
    import torch
    n = min(4, os.cpu_count() or 4)
    torch.set_num_threads(n)
    return n


# --------------------------------------------------------------------- corpus
def load():
    sents = brown(universal=True)
    words = [[w.lower() for w, _ in s] for s in sents]
    tags = [[t for _, t in s] for s in sents]
    return words, tags


def vocabulary(words, min_count=MIN_COUNT):
    c = Counter(w for s in words for w in s)
    keep = sorted((w for w, n in c.items() if n >= min_count), key=lambda w: (-c[w], w))
    return keep, {w: i for i, w in enumerate(keep)}, c


def dominant_tags(words, tags, vocab):
    """La categoria gramatical mayoritaria de cada palabra, y como de mayoritaria.

    Una palabra no tiene una categoria, tiene una distribucion ("run" es verbo y
    nombre), asi que se guardan las dos cosas: la moda y que fraccion de sus
    apariciones cae en ella. La segunda es la que dice cuanto vale la primera
    como clave de respuestas, y sin ella la medida de la ventana estaria
    puntuando contra una etiqueta inventada.
    """
    counts = defaultdict(Counter)
    for s, ts in zip(words, tags):
        for w, t in zip(s, ts):
            if w in vocab:
                counts[w][t] += 1
    out = {}
    for w, c in counts.items():
        tot = sum(c.values())
        tag, n = c.most_common(1)[0]
        out[w] = (tag, n / tot)
    return out


# ------------------------------------------------------------ skip-gram (SGNS)
def make_pairs(words, vocab, counts, window, seed=SEED, subsample=SUBSAMPLE):
    """Los pares (centro, contexto), con las dos cosas que el paper hace y nadie cuenta.

    Una: el submuestreo de palabras frecuentes, que tira apariciones de "the"
    con probabilidad creciente en su frecuencia. Dos: la ventana **aleatoria**,
    que para cada aparicion sortea un ancho entre 1 y `window`, lo que en la
    practica pesa a los vecinos cercanos mas que a los lejanos sin escribir
    ningun peso. Sin las dos, esto no es skip-gram, es otra cosa que se le
    parece.
    """
    rng = np.random.default_rng(seed)
    total = sum(counts[w] for s in words for w in s if w in vocab)
    keep_p = {}
    for w in vocab:
        f = counts[w] / total
        keep_p[w] = min(1.0, (math.sqrt(f / subsample) + 1) * (subsample / f))
    centers, contexts = [], []
    for s in words:
        ids = [vocab[w] for w in s if w in vocab and rng.random() < keep_p[w]]
        n = len(ids)
        if n < 2:
            continue
        widths = rng.integers(1, window + 1, size=n)
        for i in range(n):
            b = int(widths[i])
            lo = max(0, i - b)
            hi = min(n, i + b + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                centers.append(ids[i])
                contexts.append(ids[j])
    return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)


def train_sgns(words, vocab, counts, window, dim=DIM, epochs=EPOCHS, neg=NEG,
               seed=SEED, subword=None, label="sgns"):
    """Skip-gram con muestreo negativo, en torch, con dos matrices separadas.

    La matriz de salida (la de "contexto") es tan parte del modelo como la de
    entrada, y casi todo el mundo la tira. Aqui se conserva, porque la
    comprobacion central del articulo (que el producto de las dos aproxima la
    PMI desplazada) es una afirmacion sobre las dos y no se puede hacer con una.
    """
    import torch
    threads()
    torch.manual_seed(seed)
    V = len(vocab)
    centers, contexts = make_pairs(words, vocab, counts, window, seed=seed)
    print(f"    {label}: {len(centers):,} pares, ventana {window}")

    # Distribucion del ruido: la frecuencia elevada a 3/4, que es el unico
    # hiperparametro del paper que no tiene justificacion teorica y aun asi
    # funciona mejor que la frecuencia cruda o que la uniforme.
    freq = np.array([counts[w] for w in vocab], dtype=np.float64)
    noise = torch.tensor(freq ** 0.75 / (freq ** 0.75).sum(), dtype=torch.float)

    n_in = subword["n_rows"] if subword else V
    W = torch.nn.Parameter(torch.empty(n_in, dim).uniform_(-0.5 / dim, 0.5 / dim))
    C = torch.nn.Parameter(torch.zeros(V, dim))
    opt = torch.optim.Adam([W, C], lr=0.0025)

    if subword:
        subword = dict(subword)
        subword["S"] = subword_matrix(subword, V)

    B = 8192
    order = np.arange(len(centers))
    rng = np.random.default_rng(seed)
    losses = []
    for ep in range(epochs):
        rng.shuffle(order)
        tot = 0.0
        nb = 0
        for s in range(0, len(order), B):
            b = order[s:s + B]
            ci = torch.tensor(centers[b])
            oi = torch.tensor(contexts[b])
            ni = torch.multinomial(noise, len(b) * neg, replacement=True).view(len(b), neg)
            if subword:
                v = subword_vectors(W, subword, ci)
            else:
                v = W[ci]
            pos = (v * C[oi]).sum(-1)
            negs = torch.bmm(C[ni], v.unsqueeze(-1)).squeeze(-1)
            loss = (torch.nn.functional.softplus(-pos).mean()
                    + torch.nn.functional.softplus(negs).sum(-1).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        losses.append(tot / max(1, nb))
        print(f"      epoca {ep + 1}: perdida {losses[-1]:.4f}")
    with torch.no_grad():
        if subword:
            emb = subword_vectors(W, subword,
                                  torch.arange(V)).numpy().astype(np.float64)  # noqa: E501
        else:
            emb = W.numpy().astype(np.float64).copy()
        ctx = C.numpy().astype(np.float64).copy()
    return {"emb": emb, "ctx": ctx, "losses": losses, "window": window,
            "pairs": int(len(centers)),
            "raw": W.detach().numpy().astype(np.float64).copy() if subword else None}


def subword_matrix(sub, V):
    """La media sobre n-gramas escrita como una matriz dispersa, de una vez.

    La version obvia (juntar las filas de cada palabra en un tensor rellenado a
    la longitud maxima y promediar) construye por lote un tensor de
    8.192 x 52 x 64 y tarda horas: la palabra mas larga decide el relleno de
    todas, y la mayoria son cortas. Escrito como una matriz dispersa S de
    V x n_rows con 1/k en las filas de cada palabra, el vector de una palabra es
    una fila de S por W, la de un lote es un producto disperso-denso, y torch lo
    resuelve sin materializar nada. Es exactamente la misma media.
    """
    import torch
    rows = sub["rows"]
    offs = sub["offsets"]
    idx_i, idx_j, vals = [], [], []
    for i in range(V):
        a, b = offs[i], offs[i + 1]
        k = b - a
        if k == 0:
            continue
        for j in rows[a:b]:
            idx_i.append(i)
            idx_j.append(j)
            vals.append(1.0 / k)
    S = torch.sparse_coo_tensor(
        torch.tensor([idx_i, idx_j], dtype=torch.long),
        torch.tensor(vals, dtype=torch.float),
        (V, sub["n_rows"])).coalesce()
    return S


def subword_vectors(W, sub, idx):
    """Los vectores de un lote de palabras: filas de la matriz dispersa por W."""
    import torch
    S = sub["S"]
    # Se seleccionan las filas del lote como una matriz dispersa propia y se
    # multiplica una vez, en vez de reunir indices por palabra.
    sel = torch.index_select(S, 0, idx)
    return torch.sparse.mm(sel, W)


def build_subwords(vocab_list, nmin=3, nmax=5, buckets=60000):
    """Los n-gramas de caracteres de FastText, con su hash y sus delimitadores.

    Los signos "<" y ">" alrededor de la palabra no son decoracion: son lo que
    distingue el n-grama "<un" de un "un" en mitad de una palabra, y por tanto
    lo que permite que un prefijo signifique algo.
    """
    rows = []
    offsets = [0]
    V = len(vocab_list)
    for i, w in enumerate(vocab_list):
        rows.append(i)                       # la palabra entera tiene fila propia
        s = f"<{w}>"
        for n in range(nmin, nmax + 1):
            for j in range(len(s) - n + 1):
                g = s[j:j + n]
                rows.append(V + (hash_ft(g) % buckets))
        offsets.append(len(rows))
    return {"rows": rows, "offsets": offsets, "n_rows": V + buckets,
            "nmin": nmin, "nmax": nmax, "buckets": buckets}


def hash_ft(s):
    """El hash FNV-1a de 32 bits que usa fastText, escrito para reproducirlo."""
    h = 2166136261
    for ch in s.encode("utf-8"):
        h = (h ^ ch) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# ------------------------------------------------------------------- GloVe
def cooccurrence(words, vocab, window):
    """La matriz de coocurrencia con pesos 1/d, que es la que GloVe factoriza."""
    co = defaultdict(float)
    for s in words:
        ids = [vocab[w] for w in s if w in vocab]
        n = len(ids)
        for i in range(n):
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if i == j:
                    continue
                co[(ids[i], ids[j])] += 1.0 / abs(i - j)
    keys = np.array(list(co.keys()), dtype=np.int64)
    vals = np.array(list(co.values()), dtype=np.float64)
    return keys[:, 0], keys[:, 1], vals


def train_glove(rowi, colj, vals, V, dim=DIM, epochs=40, seed=SEED,
                xmax=100.0, alpha=0.75):
    """GloVe: minimos cuadrados ponderados sobre el logaritmo de la coocurrencia.

    La diferencia de fondo con skip-gram no es el optimizador, es que GloVe
    mira la matriz entera de una vez y skip-gram la recorre par a par. La
    funcion de peso es la que evita que las diez celdas mas grandes decidan el
    ajuste entero, y su forma esta en el paper.
    """
    import torch
    threads()
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.empty(V, dim).uniform_(-0.5 / dim, 0.5 / dim))
    C = torch.nn.Parameter(torch.empty(V, dim).uniform_(-0.5 / dim, 0.5 / dim))
    bw = torch.nn.Parameter(torch.zeros(V))
    bc = torch.nn.Parameter(torch.zeros(V))
    opt = torch.optim.Adagrad([W, C, bw, bc], lr=0.05)
    ri = torch.tensor(rowi)
    ci = torch.tensor(colj)
    x = torch.tensor(vals, dtype=torch.float)
    fw = torch.clamp(x / xmax, max=1.0) ** alpha
    lx = torch.log(x)
    n = len(vals)
    B = 200000
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    losses = []
    for ep in range(epochs):
        rng.shuffle(order)
        tot = 0.0
        for s in range(0, n, B):
            b = torch.tensor(order[s:s + B])
            pred = (W[ri[b]] * C[ci[b]]).sum(-1) + bw[ri[b]] + bc[ci[b]]
            loss = (fw[b] * (pred - lx[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        losses.append(tot / n)
        if (ep + 1) % 10 == 0:
            print(f"      epoca {ep + 1}: perdida {losses[-1]:.5f}")
    with torch.no_grad():
        # GloVe suma las dos matrices, que es lo que el paper recomienda y lo
        # que hace que sus vectores no sean comparables uno a uno con los de
        # skip-gram sin decirlo.
        emb = (W + C).numpy().astype(np.float64)
    return {"emb": emb, "losses": losses, "window": None}


# ------------------------------------------------- la identidad de Levy y Goldberg
def pmi_identity(model, rowi, colj, vals, counts, vocab_list, shift_k=NEG,
                 min_count=50):
    """Skip-gram con muestreo negativo factoriza la PMI desplazada. Medido.

    No es una intuicion ni una analogia: Levy y Goldberg demuestran que el
    optimo de la funcion objetivo de SGNS pone el producto de los dos vectores
    de una pareja en PMI(w, c) - log(k). Aqui se calcula la PMI de la matriz de
    coocurrencia, se calcula el producto que el modelo aprendio, y se comparan.

    Se restringe a pares con al menos `min_count` coocurrencias porque la PMI de
    una celda vista tres veces es ruido con una formula alrededor, y meterla
    mide el ruido y no la identidad. El corte se publica.
    """
    N = vals.sum()
    row_tot = np.zeros(len(vocab_list))
    col_tot = np.zeros(len(vocab_list))
    np.add.at(row_tot, rowi, vals)
    np.add.at(col_tot, colj, vals)
    m = vals >= min_count
    pmi = np.log((vals[m] * N) / (row_tot[rowi[m]] * col_tot[colj[m]]))
    target = pmi - math.log(shift_k)
    got = np.einsum("ij,ij->i", model["emb"][rowi[m]], model["ctx"][colj[m]])
    r = float(np.corrcoef(target, got)[0, 1])
    # La pendiente de la regresion es la cifra que dice si la relacion es la
    # identidad o solo una relacion: una correlacion alta con pendiente 0,3 no
    # es "esta factorizando PMI", es "esta ordenando como PMI".
    slope, intercept = np.polyfit(target, got, 1)
    return {"pairs": int(m.sum()), "min_count": min_count, "shift": shift_k,
            "pearson": r, "slope": float(slope), "intercept": float(intercept),
            "rmse": float(np.sqrt(np.mean((target - got) ** 2))),
            "cloud": [[round(float(a), 4), round(float(b), 4)]
                      for a, b in zip(target[::max(1, len(target) // 1200)],
                                      got[::max(1, len(target) // 1200)])]}


# ---------------------------------------------------- lo que decide la ventana
def neighbour_pos(emb, vocab_list, tagmap, topn=10, min_purity=0.8, probes=600):
    """Que fraccion de los vecinos de una palabra comparten su categoria.

    Solo se preguntan palabras cuya categoria mayoritaria lo sea al menos en un
    `min_purity` de sus apariciones: para "run", que es verbo la mitad de las
    veces y nombre la otra mitad, la pregunta no tiene respuesta y meterla
    ensucia la medida con ambiguedad lexica en vez de medir la ventana.
    """
    X = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    ok = [i for i, w in enumerate(vocab_list)
          if w in tagmap and tagmap[w][1] >= min_purity]
    ok = ok[:probes]
    S = X[ok] @ X.T
    for r, i in enumerate(ok):
        S[r, i] = -np.inf
    top = np.argsort(-S, axis=1)[:, :topn]
    hits = 0
    total = 0
    for r, i in enumerate(ok):
        want = tagmap[vocab_list[i]][0]
        for j in top[r]:
            w = vocab_list[j]
            if w in tagmap:
                hits += (tagmap[w][0] == want)
                total += 1
    return {"probes": len(ok), "topn": topn, "min_purity": min_purity,
            "same_pos": hits / max(1, total), "compared": total}


def stem_share(emb, vocab_list, topn=10, probes=500, pref=4):
    """Que fraccion de los vecinos comparten los primeros caracteres.

    Es el proxy de morfologia de este articulo, y es deliberadamente tonto: sin
    un lematizador (que seria una dependencia mas y otra decision que explicar),
    compartir cuatro caracteres iniciales es lo que separa "running" de "runs"
    sin separar "bond" de "wheat". Se dice que es un proxy en la pagina.
    """
    X = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    idx = [i for i, w in enumerate(vocab_list) if len(w) >= pref + 2][:probes]
    S = X[idx] @ X.T
    for r, i in enumerate(idx):
        S[r, i] = -np.inf
    top = np.argsort(-S, axis=1)[:, :topn]
    hits = 0
    total = 0
    for r, i in enumerate(idx):
        p = vocab_list[i][:pref]
        for j in top[r]:
            hits += vocab_list[j].startswith(p)
            total += 1
    return {"probes": len(idx), "share": hits / max(1, total), "prefix": pref}


def analogies(emb, vocab_list, vocab, sets):
    """Las analogias, con el detalle que decide el resultado y casi nadie dice.

    "a es a b como c es a ?" se resuelve buscando el vector mas cercano a
    b - a + c, y **excluyendo a, b y c de la busqueda**. Sin esa exclusion la
    respuesta suele ser c, porque c es lo mas parecido a algo que es casi c, y
    la precision publicada sube sola. Aqui se mide con y sin, porque la
    diferencia entre las dos es mas instructiva que cualquiera de ellas.
    """
    X = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    out = []
    for name, quads in sets.items():
        usable = [q for q in quads if all(w in vocab for w in q)]
        hit = 0
        hit_lazy = 0
        for a, b, c, d in usable:
            v = X[vocab[b]] - X[vocab[a]] + X[vocab[c]]
            v = v / max(np.linalg.norm(v), 1e-12)
            s = X @ v
            lazy = int(np.argmax(s))
            for w in (a, b, c):
                s[vocab[w]] = -np.inf
            hit += (vocab_list[int(np.argmax(s))] == d)
            hit_lazy += (vocab_list[lazy] == d)
        out.append({"set": name, "asked": len(usable), "of": len(quads),
                    "accuracy": hit / max(1, len(usable)),
                    "accuracy_without_exclusion": hit_lazy / max(1, len(usable))})
        print(f"    {name:<22} {len(usable):>3} of {len(quads):>3} askable, "
              f"{hit / max(1, len(usable)):.4f} correct "
              f"({hit_lazy / max(1, len(usable)):.4f} without the exclusion)")
    return out


ANALOGY_SETS = {
    "singular to plural": [("year", "years", "day", "days"), ("man", "men", "child", "children"),
                           ("car", "cars", "book", "books"), ("hand", "hands", "eye", "eyes"),
                           ("word", "words", "school", "schools"),
                           ("problem", "problems", "question", "questions")],
    "present to past": [("is", "was", "are", "were"), ("has", "had", "have", "had"),
                        ("say", "said", "come", "came"), ("take", "took", "give", "gave"),
                        ("go", "went", "see", "saw"), ("make", "made", "find", "found")],
    "man to woman": [("he", "she", "him", "her"), ("man", "woman", "boy", "girl"),
                     ("his", "her", "man", "woman"), ("father", "mother", "son", "daughter"),
                     ("men", "women", "boys", "girls")],
    "comparative": [("good", "better", "great", "greater"), ("long", "longer", "high", "higher"),
                    ("small", "smaller", "large", "larger"), ("old", "older", "young", "younger")],
}


def oov_experiment(model_ft, sub, vocab, vocab_list, counts, words, emb_sg,
                   min_count=MIN_COUNT):
    """Lo unico que FastText puede hacer y skip-gram no: contestar sobre lo que no vio.

    Se toman palabras que quedaron JUSTO por debajo del corte de vocabulario,
    asi que ninguno de los dos modelos tiene una fila para ellas. FastText puede
    componer una desde sus n-gramas de caracteres; skip-gram no tiene nada, y
    eso no es un mal resultado, es la ausencia de un resultado, que es
    precisamente el punto.

    Para puntuar hace falta una verdad: se estima el vector "correcto" de cada
    palabra fuera de vocabulario como la media de los vectores de sus vecinos
    reales en el corpus, que es la definicion distribucional del significado y
    no depende de ninguno de los dos modelos.
    """
    import torch
    ctx = defaultdict(list)
    for s in words:
        for i, w in enumerate(s):
            if w in vocab:
                continue
            for j in range(max(0, i - 5), min(len(s), i + 6)):
                if j != i and s[j] in vocab:
                    ctx[w].append(vocab[s[j]])
    cand = [w for w, c in ctx.items()
            if len(c) >= 30 and 3 <= counts[w] < min_count and w.isalpha() and len(w) >= 5]
    cand = sorted(cand)[:150]
    if not cand:
        return None
    # El espacio NO es isotropo: los vectores entrenados comparten una direccion
    # dominante (el efecto de frecuencia), asi que el coseno entre dos palabras
    # al azar ya vale 0,36 y comparar contra el no mide nada. Se centra, que es
    # la correccion estandar, y **se mide el coseno centrado a los dos lados**.
    raw = emb_sg
    centre = raw.mean(axis=0, keepdims=True)
    cen = raw - centre
    X = cen / np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-12)
    W = torch.tensor(model_ft["raw"], dtype=torch.float)

    def compose(w):
        g = []
        s = f"<{w}>"
        for n in range(sub["nmin"], sub["nmax"] + 1):
            for j in range(len(s) - n + 1):
                g.append(len(vocab_list) + (hash_ft(s[j:j + n]) % sub["buckets"]))
        if not g:
            return None
        v = W[torch.tensor(g)].mean(0).numpy() - centre[0]
        return v / max(np.linalg.norm(v), 1e-12)

    truths, made, kept = [], [], []
    for w in cand:
        t = X[ctx[w]].mean(axis=0)
        nt = np.linalg.norm(t)
        if nt < 1e-9:
            continue
        v = compose(w)
        if v is None:
            continue
        truths.append(t / nt)
        made.append(v)
        kept.append(w)
    if len(kept) < 20:
        return None
    truths = np.array(truths)
    made = np.array(made)

    matched = np.einsum("ij,ij->i", made, truths)
    # El control que importa: el MISMO vector compuesto contra la verdad de OTRA
    # palabra. Empareja las dos distribuciones y deja variar solo el
    # emparejamiento, que es lo unico que la afirmacion es sobre. Un control de
    # "dos palabras al azar" mide la anisotropia del espacio y no esto.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(kept))
    perm = np.array([p if p != i else (p + 1) % len(kept) for i, p in enumerate(perm)])
    mismatched = np.einsum("ij,ij->i", made, truths[perm])
    # Y el techo: cuanto se parecen dos estimaciones de contexto entre si, que
    # es lo mejor a lo que la composicion podria aspirar.
    return {"words": len(kept),
            "mean_cosine": float(matched.mean()),
            "median_cosine": float(np.median(matched)),
            "mismatched_mean": float(mismatched.mean()),
            "mismatched_sd": float(mismatched.std()),
            "lift": float(matched.mean() - mismatched.mean()),
            "wins": float((matched > mismatched).mean()),
            "centred": True,
            "examples": kept[:12],
            "note": "skip-gram has no row for any of these"}


def ship_vectors(models, vocab_list, counts, tagmap, keep=900):
    """Lo que viaja: los `keep` vectores mas frecuentes de cada modelo, en float16.

    Todo lo que la pagina dibuja se mide sobre estos, no sobre los originales,
    porque es lo que el navegador va a tener. La proyeccion a dos dimensiones se
    calcula aqui y viaja hecha; los vecinos se calculan en el navegador desde
    los vectores, que es lo que hace que el lector pueda preguntar por cualquier
    palabra en vez de por una lista precocinada.
    """
    idx = list(range(min(keep, len(vocab_list))))
    out = {"words": [vocab_list[i] for i in idx],
           "counts": [int(counts[vocab_list[i]]) for i in idx],
           "tags": [tagmap.get(vocab_list[i], ("X", 0.0))[0] for i in idx],
           "tag_purity": [round(float(tagmap.get(vocab_list[i], ("X", 0.0))[1]), 3)
                          for i in idx],
           "dim": DIM, "models": {}}
    for name, m in models.items():
        E = m["emb"][idx]
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        # Se redondea ANTES de medir nada, porque el redondeo es parte del
        # experimento y no un detalle de la exportacion.
        E16 = E.astype(np.float16).astype(np.float64)
        E16 = E16 / np.maximum(np.linalg.norm(E16, axis=1, keepdims=True), 1e-12)
        out["models"][name] = {
            "vec": b64_f16(E.astype(np.float16).ravel()),
            "proj": project(E16),
        }
    return out, idx


def project(E, seed=SEED):
    """Dos dimensiones por PCA sobre los vectores ya cuantizados.

    PCA y no t-SNE a proposito: el articulo de vecinos de la seccion 2 ya midio
    que t-SNE no conserva ni tamanos ni huecos, asi que un mapa de t-SNE aqui
    seria bonito y no se podria leer. Con PCA, la fraccion de varianza que las
    dos direcciones capturan se puede publicar al lado del dibujo, que es la
    forma honesta de decir cuanto se esta perdiendo.
    """
    C = E - E.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    P = C @ Vt[:2].T
    var = float((S[:2] ** 2).sum() / (S ** 2).sum())
    return {"xy": [[round(float(a), 4), round(float(b), 4)] for a, b in P],
            "explained": var}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("cargando el Brown")
    words, tags = cached("brown", load)
    vocab_list, vocab, counts = vocabulary(words)
    tagmap = dominant_tags(words, tags, vocab)
    n_tok = sum(len(s) for s in words)
    print(f"  {len(words)} frases, {n_tok} tokens, {len(vocab_list)} palabras "
          f"con al menos {MIN_COUNT} apariciones")

    print("skip-gram con muestreo negativo, tres ventanas")
    sg = {}
    for win in (2, 5, 10):
        sg[win] = cached(f"sgns_w{win}",
                         lambda w=win: train_sgns(words, vocab, counts, w,
                                                  label=f"sgns w{w}"))

    print("coocurrencia y GloVe")
    rowi, colj, vals = cached("cooc", lambda: cooccurrence(words, vocab, 5))
    print(f"  {len(vals)} celdas no nulas")
    gl = cached("glove", lambda: train_glove(rowi, colj, vals, len(vocab_list)))

    print("FastText")
    sub = cached("subwords", lambda: build_subwords(vocab_list))
    ft = cached("fasttext",
                lambda: train_sgns(words, vocab, counts, 5, subword=sub, label="fasttext"))

    print("la identidad de Levy y Goldberg")
    ident = cached("pmi", lambda: pmi_identity(sg[5], rowi, colj, vals, counts, vocab_list))
    print(f"    {ident['pairs']} pares: r = {ident['pearson']:.4f}, "
          f"pendiente {ident['slope']:.4f}, rmse {ident['rmse']:.4f}")

    print("lo que decide la ventana")
    win_rows = []
    for win, m in sg.items():
        p = neighbour_pos(m["emb"], vocab_list, tagmap)
        st = stem_share(m["emb"], vocab_list)
        win_rows.append({"window": win, **p, "stem_share": st["share"],
                         "pairs": m["pairs"]})
        print(f"    ventana {win:>2}: {p['same_pos']:.4f} de los vecinos comparten "
              f"categoria, {st['share']:.4f} comparten prefijo")

    print("los tres modelos, mismos vecinos")
    models = {"skip-gram": sg[5], "glove": gl, "fasttext": ft}
    model_rows = []
    for name, m in models.items():
        p = neighbour_pos(m["emb"], vocab_list, tagmap)
        st = stem_share(m["emb"], vocab_list)
        an = analogies(m["emb"], vocab_list, vocab, ANALOGY_SETS)
        tot = sum(a["asked"] for a in an)
        acc = sum(a["accuracy"] * a["asked"] for a in an) / max(1, tot)
        lazy = sum(a["accuracy_without_exclusion"] * a["asked"] for a in an) / max(1, tot)
        model_rows.append({"model": name, "same_pos": p["same_pos"],
                           "stem_share": st["share"], "analogy": acc,
                           "analogy_lazy": lazy, "analogy_asked": tot,
                           "sets": an})
        print(f"    {name:<10} categoria {p['same_pos']:.4f}  prefijo {st['share']:.4f}  "
              f"analogias {acc:.4f}")

    print("palabras que ninguno vio")
    oov = cached("oov", lambda: oov_experiment(ft, sub, vocab, vocab_list, counts,
                                               words, sg[5]["emb"]))
    if oov:
        print(f"    {oov['words']} palabras: coseno medio {oov['mean_cosine']:.4f} contra "
              f"{oov['mismatched_mean']:.4f} del MISMO vector contra otra palabra "
              f"(diferencia {oov['lift']:+.4f}, acierta el emparejamiento "
              f"{oov['wins']:.1%} de las veces)")
    else:
        print("    no hay bastantes palabras utilizables; el articulo lo dira en vez de "
              "medir sobre un punado")

    print("empaquetando")
    shipped, idx = ship_vectors(models, vocab_list, counts, tagmap)

    data = {
        "meta": {
            "corpus": "the Brown corpus, part-of-speech tagged",
            "sentences": len(words), "tokens": n_tok,
            "vocab": len(vocab_list), "min_count": MIN_COUNT,
            "dim": DIM, "epochs": EPOCHS, "negatives": NEG,
            "subsample": SUBSAMPLE, "seed": SEED,
            "tagset": sorted({t for s in tags for t in s}),
            "shipped": len(idx),
            "cooc_cells": int(len(vals)),
            "subword_buckets": sub["buckets"],
            "subword_range": [sub["nmin"], sub["nmax"]],
        },
        "pmi": ident,
        "windows": win_rows,
        "models": model_rows,
        "oov": oov,
        "vectors": shipped,
        "losses": {name: [float(x) for x in m.get("losses", [])]
                   for name, m in models.items()},
    }
    path = OUT / "embeddings.json"
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
