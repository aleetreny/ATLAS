"""Las cifras del articulo de modelos de secuencia (HMM, RNN, LSTM, GRU).

Corpus: el Brown otra vez, y por la misma razon que en el articulo de vectores,
solo que aqui la razon es todavia mas directa: las etiquetas gramaticales no son
una clave de respuestas auxiliar, **son la tarea**. Etiquetar cada palabra con su
categoria es el problema de secuencia mas viejo que hay, tiene una respuesta
conocida token a token, y se puede resolver con un modelo de 1966 y con uno de
2014 sobre exactamente los mismos datos.

Cuatro modelos y una sola pregunta: cuanto de lo que vino antes hace falta
recordar, y cuanto puede cada uno recordar de verdad. El HMM contesta lo primero
por construccion (recuerda una etiqueta) y las tres redes contestan lo segundo
por medicion.

Dos cosas se miden aqui que casi nunca se publican juntas:

  1. **El gradiente contra la distancia**, que es la afirmacion central sobre
     por que existen LSTM y GRU. Se mide la norma de la derivada del estado en
     el paso T respecto al estado en el paso t, para todas las distancias, en
     las tres arquitecturas, con los mismos datos y el mismo entrenamiento.
  2. **Donde se rompe cada una**, en una tarea de copia con la distancia como
     unico parametro. Un modelo que aprende a distancia 5 y falla a distancia
     60 no es "peor", es otra cosa, y la cifra que lo dice es la distancia de
     ruptura y no la exactitud media.

Ejecutar desde cualquier sitio: `python src/utils/generate_sequences_data.py`.
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
OUT = ROOT / "sequences" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "sequences_cache")

SEED = 19
HIDDEN = 96
EMB = 64
EPOCHS = 4
MIN_COUNT = 3


def cached(name, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.is_file():
        with path.open("rb") as fh:
            return pickle.load(fh)
    val = fn()
    # A un temporal y luego rename: si algo falla a mitad de `pickle.dump`, el
    # fichero definitivo no llega a existir. Sin esto, un fallo posterior deja
    # un pickle de cero bytes y la tirada siguiente muere al LEERLO, que es un
    # error mucho mas confuso que el original.
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(val, fh)
    tmp.replace(path)
    return val


def b64_f16(a):
    return base64.b64encode(np.asarray(a, dtype="<f2").tobytes()).decode()


def threads():
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 4))


# --------------------------------------------------------------------- corpus
def load_split():
    """Frases del Brown, partidas por FRASE y no por token.

    Partir por token dejaria la mitad de una frase en entrenamiento y la otra
    mitad en test, y como el modelo mira el contexto eso es una fuga: el test
    dejaria de medir lo que se le pide medir. Se parte por frase, con una
    semilla, y el reparto se publica.
    """
    sents = brown(universal=True)
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(sents))
    rng.shuffle(idx)
    cut = int(0.9 * len(sents))
    tr = [sents[i] for i in idx[:cut]]
    te = [sents[i] for i in idx[cut:]]
    return tr, te


def vocabularies(train, min_count=MIN_COUNT):
    wc = Counter(w.lower() for s in train for w, _ in s)
    words = ["<unk>"] + sorted(w for w, n in wc.items() if n >= min_count)
    tags = sorted({t for s in train for _, t in s})
    return words, {w: i for i, w in enumerate(words)}, tags, {t: i for i, t in enumerate(tags)}


# ----------------------------------------------------------------------- HMM
def train_hmm(train, words, wi, tags, ti, alpha=0.1):
    """Un HMM bigrama contado, no ajustado: transiciones, emisiones e inicios.

    Con las etiquetas observadas no hace falta EM para nada, y usarlo seria
    fingir una dificultad que este problema no tiene. Lo unico que hay que
    decidir es el suavizado, y se elige uno solo (Laplace con alpha) que se
    aplica igual a las tres tablas para que ninguna comparacion posterior
    dependa de haber afinado una de ellas.
    """
    T = len(tags)
    V = len(words)
    trans = np.full((T, T), alpha)
    emit = np.full((T, V), alpha)
    start = np.full(T, alpha)
    for s in train:
        prev = None
        for w, t in s:
            j = ti[t]
            emit[j, wi.get(w.lower(), 0)] += 1
            if prev is None:
                start[j] += 1
            else:
                trans[prev, j] += 1
            prev = j
    log_trans = np.log(trans / trans.sum(axis=1, keepdims=True))
    log_emit = np.log(emit / emit.sum(axis=1, keepdims=True))
    log_start = np.log(start / start.sum())
    return {"log_trans": log_trans, "log_emit": log_emit, "log_start": log_start,
            "alpha": alpha}


def viterbi(hmm, ids):
    """El algoritmo de Viterbi, en logaritmos y con los punteros guardados.

    En logaritmos y no en probabilidades porque el producto de doscientas
    probabilidades es cero en coma flotante mucho antes de que la frase se
    acabe: no es una optimizacion, es la diferencia entre funcionar y no.
    """
    A = hmm["log_trans"]
    B = hmm["log_emit"]
    pi = hmm["log_start"]
    T = A.shape[0]
    n = len(ids)
    delta = np.zeros((n, T))
    back = np.zeros((n, T), dtype=np.int32)
    delta[0] = pi + B[:, ids[0]]
    for i in range(1, n):
        m = delta[i - 1][:, None] + A
        back[i] = m.argmax(axis=0)
        delta[i] = m.max(axis=0) + B[:, ids[i]]
    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(delta[-1].argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path, delta, back


def score_hmm(hmm, test, wi, ti, tags, train):
    """Exactitud del HMM y de la referencia que hay que batir.

    La referencia no es el azar, es **la etiqueta mas frecuente de cada
    palabra**, que no mira el contexto en absoluto. Es la unica comparacion que
    dice si el modelo de secuencia esta aportando secuencia o solo un
    diccionario, y en esta tarea el diccionario llega muy lejos.
    """
    most = defaultdict(Counter)
    for s in train:
        for w, t in s:
            most[w.lower()][t] += 1
    fallback = Counter(t for s in train for _, t in s).most_common(1)[0][0]
    best = {w: c.most_common(1)[0][0] for w, c in most.items()}
    ambiguous = {w for w, c in most.items() if len(c) > 1}

    tot = hit_h = hit_b = 0
    amb_tot = amb_h = amb_b = 0
    unk_tot = unk_h = unk_b = 0
    conf = np.zeros((len(tags), len(tags)), dtype=np.int64)
    for s in test:
        ids = [wi.get(w.lower(), 0) for w, _ in s]
        path, _, _ = viterbi(hmm, ids)
        for k, (w, t) in enumerate(s):
            lw = w.lower()
            gold = ti[t]
            ph = int(path[k])
            pb = ti[best.get(lw, fallback)]
            conf[gold, ph] += 1
            tot += 1
            hit_h += (ph == gold)
            hit_b += (pb == gold)
            if lw in ambiguous:
                amb_tot += 1
                amb_h += (ph == gold)
                amb_b += (pb == gold)
            if lw not in best:
                unk_tot += 1
                unk_h += (ph == gold)
                unk_b += (pb == gold)
    return {
        "tokens": tot,
        "hmm": hit_h / tot, "baseline": hit_b / tot,
        "ambiguous_tokens": amb_tot,
        "ambiguous_share": amb_tot / tot,
        "hmm_ambiguous": amb_h / max(1, amb_tot),
        "baseline_ambiguous": amb_b / max(1, amb_tot),
        "unknown_tokens": unk_tot,
        "hmm_unknown": unk_h / max(1, unk_tot),
        "baseline_unknown": unk_b / max(1, unk_tot),
        "confusion": conf.tolist(),
        "tags": tags,
        "ambiguous_types": len(ambiguous),
        "types": len(best),
    }


# ------------------------------------------------------------------- las redes
def make_batches(sents, wi, ti, batch=64, seed=SEED):
    import torch
    order = sorted(range(len(sents)), key=lambda i: len(sents[i]))
    out = []
    for s in range(0, len(order), batch):
        chunk = [sents[i] for i in order[s:s + batch]]
        n = max(len(c) for c in chunk)
        x = torch.zeros(len(chunk), n, dtype=torch.long)
        y = torch.full((len(chunk), n), -100, dtype=torch.long)
        for r, c in enumerate(chunk):
            for k, (w, t) in enumerate(c):
                x[r, k] = wi.get(w.lower(), 0)
                y[r, k] = ti[t]
        out.append((x, y))
    rng = np.random.default_rng(seed)
    rng.shuffle(out)
    return out


class Tagger:
    pass


def build_tagger(kind, V, T, hidden=HIDDEN, emb=EMB, seed=SEED):
    """Las tres celdas con el MISMO presupuesto, que es lo que hace comparable la tabla.

    Una LSTM tiene cuatro puertas y una GRU tres y una RNN una, asi que a igual
    anchura oculta la LSTM tiene casi cuatro veces mas parametros y cualquier
    comparacion mide el tamano. Se ajusta la anchura de cada una para que las
    tres caigan dentro del 2% del mismo recuento, y el recuento se publica.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    gates = {"rnn": 1, "gru": 3, "lstm": 4}[kind]
    # params de la recurrencia ~ gates * (h*h + h*emb + 2h); se resuelve para h
    target = 4 * (HIDDEN * HIDDEN + HIDDEN * EMB + 2 * HIDDEN)
    h = HIDDEN
    while gates * (h * h + h * EMB + 2 * h) < target:
        h += 1
    while gates * (h * h + h * EMB + 2 * h) > target and h > 8:
        h -= 1
    cell = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
    m = nn.Module()
    m.emb = nn.Embedding(V, emb, padding_idx=0)
    m.rnn = cell(emb, h, batch_first=True, bidirectional=False)
    m.out = nn.Linear(h, T)
    m.hidden = h
    m.kind = kind
    return m


def forward_tagger(m, x):
    e = m.emb(x)
    o, _ = m.rnn(e)
    return m.out(o), o


def train_tagger(kind, train, test, wi, ti, epochs=EPOCHS, seed=SEED):
    import torch
    import torch.nn as nn
    threads()
    torch.manual_seed(seed)
    V, T = len(wi), len(ti)
    m = build_tagger(kind, V, T, seed=seed)
    rec = sum(p.numel() for p in m.rnn.parameters())
    tot = sum(p.numel() for p in m.parameters())
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    tb = make_batches(train, wi, ti, seed=seed)
    eb = make_batches(test, wi, ti, seed=seed)
    hist = []
    for ep in range(epochs):
        m.train()
        s = 0.0
        for x, y in tb:
            logits, _ = forward_tagger(m, x)
            loss = lossf(logits.reshape(-1, T), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            s += float(loss)
        acc = eval_tagger(m, eb, T)
        hist.append({"epoch": ep + 1, "loss": s / len(tb), "acc": acc})
        print(f"      epoca {ep + 1}: perdida {s / len(tb):.4f}, exactitud {acc:.4f}")
    return {"model": m, "hist": hist, "hidden": m.hidden,
            "recurrent_params": rec, "params": tot}


def eval_tagger(m, batches, T):
    import torch
    m.eval()
    hit = tot = 0
    with torch.no_grad():
        for x, y in batches:
            logits, _ = forward_tagger(m, x)
            p = logits.argmax(-1)
            mask = y >= 0
            hit += int((p[mask] == y[mask]).sum())
            tot += int(mask.sum())
    return hit / max(1, tot)


# -------------------------------------------------- el gradiente y la distancia
def gradient_flow(models, test, wi, ti, length=60, samples=24, seed=SEED):
    """Cuanto mueve todavia la entrada del paso t al estado del paso T.

    La cantidad medida se nombra con precision porque hay dos parecidas y no son
    la misma: aqui es la norma de la derivada del estado FINAL respecto a la
    ENTRADA del paso t, no respecto al estado oculto de ese paso. Es la que
    responde a la pregunta que le importa a un lector ("una palabra de hace
    treinta pasos, sigue influyendo en lo que la red cree ahora"), es la que el
    entrenamiento propaga de verdad, y no depende de como cada celda decida
    representar su estado, que es lo que haria incomparables a las tres.


    Esta es la afirmacion que justifica que existan LSTM y GRU, y casi siempre
    se cuenta con un dibujo en vez de con una medida. Se mide asi: se toma una
    frase de `length` pasos, se hace el forward guardando la secuencia de
    estados, y para cada distancia d se pide a autograd la norma de la
    derivada del estado final respecto al estado en el paso final menos d.

    Dos cuidados que cambian el resultado si se saltan.

    Uno: se mide sobre las tres redes **ya entrenadas**, no sobre pesos
    iniciales. Con pesos iniciales el resultado es una propiedad de la
    inicializacion y no de la arquitectura, que es lo que se quiere saber.

    Dos: la sonda no puede dejar el modelo distinto de como estaba. Aqui no hay
    normalizacion por lotes de por medio, asi que no hay estadisticas moviles
    que contaminar, pero se pone `eval()` y se restaura igualmente, y el
    generador comprueba que la exactitud antes y despues de sondear es la
    misma hasta el ultimo bit.
    """
    import torch
    rng = np.random.default_rng(seed)
    long_sents = [s for s in test if len(s) >= length]
    if not long_sents:
        long_sents = sorted(test, key=len, reverse=True)[:samples]
    pick = [long_sents[i] for i in rng.choice(len(long_sents),
                                              size=min(samples, len(long_sents)),
                                              replace=False)]
    out = {}
    for name, bundle in models.items():
        m = bundle["model"]
        was_training = m.training
        m.eval()
        before = None
        curves = []
        for s in pick:
            ids = torch.tensor([[wi.get(w.lower(), 0) for w, _ in s[:length]]])
            e = m.emb(ids).detach().clone().requires_grad_(True)
            o, _ = m.rnn(e)
            n = o.shape[1]
            # UNA sola pasada hacia atras. La derivada del estado final respecto
            # a TODAS las entradas sale de una llamada; pedirla dentro del bucle
            # de distancias recalcula lo mismo n veces y no cambia el resultado,
            # solo el tiempo.
            g = torch.autograd.grad(o[0, n - 1].sum(), e)[0][0]
            curves.append([float(g[n - 1 - d].norm()) for d in range(1, n)])
        L = min(len(c) for c in curves)
        arr = np.array([c[:L] for c in curves])
        med = np.median(arr, axis=0)
        # Se normaliza por la distancia 1, porque lo que se compara entre
        # arquitecturas es la FORMA de la caida y no su escala: tres modelos con
        # normas de partida distintas darian tres curvas separadas por una
        # constante y la comparacion mediria esa constante.
        rel = med / max(med[0], 1e-30)
        half = next((i + 1 for i, v in enumerate(rel) if v < 0.5), None)
        hundredth = next((i + 1 for i, v in enumerate(rel) if v < 0.01), None)
        out[name] = {
            "distances": list(range(1, L + 1)),
            "median": [float(v) for v in med],
            "relative": [float(v) for v in rel],
            "half_life": half, "hundredth_at": hundredth,
            "at_10": float(rel[9]) if L > 9 else None,
            "at_30": float(rel[29]) if L > 29 else None,
            "sentences": len(pick),
        }
        if was_training:
            m.train()
        print(f"    {name:<5} el gradiente cae a la mitad en "
              f"{half if half else '>' + str(L)} pasos y a la centesima en "
              f"{hundredth if hundredth else '>' + str(L)}")
    return out


# ------------------------------------------------------------- la tarea de copia
def copy_task(distance, n=3000, vocab=12, seed=SEED):
    """Recordar un simbolo a traves de `distance` distractores.

    La tarea mas honesta que existe para medir memoria, porque no hay nada mas
    que memoria en ella: el primer simbolo es la respuesta, todo lo que viene
    despues es ruido sacado de un alfabeto distinto, y el modelo tiene que
    decir el primero al final. Un modelo sin memoria acierta 1 de `vocab`.
    """
    rng = np.random.default_rng(seed + distance)
    x = np.zeros((n, distance + 2), dtype=np.int64)
    y = rng.integers(1, vocab + 1, size=n)
    x[:, 0] = y
    x[:, 1:-1] = rng.integers(vocab + 1, vocab + 6, size=(n, distance))
    x[:, -1] = vocab + 6                      # la senal de "contesta ahora"
    return x, y - 1


def copy_curve(kind, distances, epochs=30, hidden=48, seed=SEED):
    """Donde se rompe cada arquitectura, con el mismo presupuesto en las tres."""
    import torch
    import torch.nn as nn
    threads()
    rows = []
    for d in distances:
        torch.manual_seed(seed)
        x, y = copy_task(d, seed=seed)
        xt = torch.tensor(x)
        yt = torch.tensor(y)
        ntr = int(0.8 * len(x))
        V = int(x.max()) + 1
        K = int(y.max()) + 1
        gates = {"rnn": 1, "gru": 3, "lstm": 4}[kind]
        h = hidden
        target = 4 * (hidden * hidden + hidden * 32)
        while gates * (h * h + h * 32) < target:
            h += 1
        while gates * (h * h + h * 32) > target and h > 6:
            h -= 1
        emb = nn.Embedding(V, 32)
        cell = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind](32, h, batch_first=True)
        head = nn.Linear(h, K)
        params = list(emb.parameters()) + list(cell.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=3e-3)
        lossf = nn.CrossEntropyLoss()
        B = 256
        for ep in range(epochs):
            perm = torch.randperm(ntr)
            for s in range(0, ntr, B):
                b = perm[s:s + B]
                o, _ = cell(emb(xt[b]))
                loss = lossf(head(o[:, -1]), yt[b])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
        with torch.no_grad():
            o, _ = cell(emb(xt[ntr:]))
            acc = float((head(o[:, -1]).argmax(-1) == yt[ntr:]).float().mean())
        rows.append({"distance": d, "acc": acc, "hidden": h, "chance": 1.0 / K})
        print(f"      {kind} a distancia {d:>3}: {acc:.4f}")
    return rows


def ship_hmm(hmm, words, wi, tags, examples):
    """El HMM al navegador, en float16, para que Viterbi corra alli.

    Las tres tablas son log-probabilidades, asi que se envian tal cual y el
    navegador no tiene que normalizar nada. De la matriz de emision viajan
    **solo las columnas que las frases de ejemplo usan**, remapeadas, que es lo
    unico que la pagina evalua.

    El primer intento recortaba a las 6.000 primeras columnas, y eso estaba mal
    de una forma que solo se ve en pantalla: el vocabulario esta ordenado
    ALFABETICAMENTE, asi que las 6.000 primeras son las palabras que empiezan
    por a, b y c. Todo lo demas caia en <unk>, el navegador etiquetaba media
    pagina como X y el guardia lo canto. Un recorte por indice solo es inocente
    si el indice significa algo.
    """
    need = sorted({0} | {wi.get(w.lower(), 0) for ex in examples for w in ex["words"]})
    remap = {j: i for i, j in enumerate(need)}
    return {
        "tags": tags,
        "words": [words[j] for j in need],
        "log_start": [float(v) for v in hmm["log_start"]],
        "log_trans": b64_f16(np.asarray(hmm["log_trans"]).ravel()),
        "log_emit": b64_f16(np.asarray(hmm["log_emit"])[:, need].ravel()),
        "n_tags": len(tags),
        "n_words": len(need),
        "sentences": [{"words": ex["words"], "gold": ex["gold"],
                       "ids": [remap[wi.get(w.lower(), 0)] for w in ex["words"]]}
                      for ex in examples],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("cargando el Brown y partiendo por frase")
    train, test = cached("split", load_split)
    words, wi, tags, ti = vocabularies(train)
    print(f"  {len(train)} frases de entrenamiento, {len(test)} de test, "
          f"{sum(len(s) for s in train)} y {sum(len(s) for s in test)} tokens, "
          f"{len(words)} palabras, {len(tags)} etiquetas")

    print("el HMM, contado")
    hmm = cached("hmm", lambda: train_hmm(train, words, wi, tags, ti))
    hs = cached("hmm_score", lambda: score_hmm(hmm, test, wi, ti, tags, train))
    print(f"  HMM {hs['hmm']:.4f} contra {hs['baseline']:.4f} de la etiqueta mas frecuente")
    print(f"  en las {hs['ambiguous_share']:.1%} de palabras ambiguas: "
          f"{hs['hmm_ambiguous']:.4f} contra {hs['baseline_ambiguous']:.4f}")

    print("las tres redes, mismo presupuesto")
    nets = {}
    for kind in ("rnn", "gru", "lstm"):
        print(f"    {kind}")
        nets[kind] = cached(f"net_{kind}",
                            lambda k=kind: train_tagger(k, train, test, wi, ti))

    print("el gradiente contra la distancia, entrenadas")
    flow = cached("flow", lambda: gradient_flow(nets, test, wi, ti))

    # Las tres curvas entrenadas salen casi identicas, y eso no es un fallo de
    # la sonda: entrenadas en una tarea local aprenden a ser locales, asi que lo
    # que se mide es lo aprendido y no la arquitectura. El teorema del gradiente
    # que se desvanece habla de la INICIALIZACION, asi que se mide tambien ahi,
    # con los mismos pesos iniciales que tuvieron estas mismas redes.
    print("y el mismo gradiente ANTES de entrenar, que es de lo que habla el teorema")
    fresh = {k: {"model": build_tagger(k, len(wi), len(ti))}
             for k in ("rnn", "gru", "lstm")}
    flow0 = cached("flow_untrained", lambda: gradient_flow(fresh, test, wi, ti))

    print("la tarea de copia")
    dists = [5, 10, 20, 40, 80]
    copy = {}
    for kind in ("rnn", "gru", "lstm"):
        copy[kind] = cached(f"copy_{kind}", lambda k=kind: copy_curve(k, dists))

    print("frases de ejemplo para el trellis")
    examples = pick_examples(test, wi, ti, hmm, tags)

    data = {
        "meta": {
            "corpus": "the Brown corpus, tagged with the twelve universal tags",
            "train_sents": len(train), "test_sents": len(test),
            "train_tokens": sum(len(s) for s in train),
            "test_tokens": sum(len(s) for s in test),
            "vocab": len(words), "min_count": MIN_COUNT,
            "tags": tags, "seed": SEED,
            "hidden": HIDDEN, "emb": EMB, "epochs": EPOCHS,
        },
        "hmm": {k: v for k, v in hs.items()},
        "hmm_alpha": hmm["alpha"],
        "nets": {k: {"hist": v["hist"], "hidden": v["hidden"],
                     "recurrent_params": v["recurrent_params"], "params": v["params"],
                     "acc": v["hist"][-1]["acc"]} for k, v in nets.items()},
        "flow": flow,
        "flow_untrained": flow0,
        "copy": copy,
        "copy_distances": dists,
        "model": ship_hmm(hmm, words, wi, tags, examples),
        "examples": examples,
    }
    path = OUT / "sequences.json"
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


def pick_examples(test, wi, ti, hmm, tags, n=6):
    """Frases donde el contexto DECIDE, que es lo que el trellis existe para ensenar.

    Se buscan frases de longitud razonable que contengan al menos una palabra
    ambigua y en las que Viterbi acierte donde la etiqueta mas frecuente falla.
    Una frase donde los dos aciertan no ensena nada, y elegirla a ojo seria
    elegir la que ilustra la conclusion.
    """
    out = []
    for s in test:
        if not (6 <= len(s) <= 14):
            continue
        ids = [wi.get(w.lower(), 0) for w, _ in s]
        path, _, _ = viterbi(hmm, ids)
        gold = [ti[t] for _, t in s]
        if list(path) == gold:
            out.append({
                "words": [w for w, _ in s],
                "gold": [t for _, t in s],
                "ids": ids,
            })
        if len(out) >= n:
            break
    return out


if __name__ == "__main__":
    main()
