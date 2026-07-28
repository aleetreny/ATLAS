"""Las cifras del articulo de objetivos de preentrenamiento (BERT, GPT, T5).

La pregunta del articulo no es cual es mejor, que no tiene respuesta. Es que
compra cada uno, y para que eso signifique algo hay que dejar fijo todo lo
demas: **la misma arquitectura, el mismo presupuesto de parametros, el mismo
corpus, la misma particion y el mismo numero de pasos de optimizacion**. Lo
unico que cambia entre los tres brazos es que se le pide al modelo que prediga.

  BERT   enmascarar una fraccion de los tokens y predecirlos, viendo los dos
         lados. Bidireccional, no puede generar.
  GPT    predecir el siguiente token. Ve solo la izquierda, y por eso genera.
  T5     corromper tramos enteros y regenerarlos con un decodificador. Ve los
         dos lados en la entrada y genera en la salida, y paga por las dos.

Tres cosas se miden que rara vez aparecen juntas:

  1. **La tasa de enmascarado**, barrida. El 15% es folklore repetido desde
     2018; aqui se barre de 5% a 50% y se mira donde cae el optimo en ESTE
     corpus, que puede no ser 15 y si no lo es se publica lo que salga.
  2. **La senal por pasada**, que es la razon de fondo por la que el objetivo
     causal escala mejor y casi nunca se dice: un modelo enmascarado solo
     recibe gradiente en la fraccion enmascarada de los tokens, y el causal en
     todos. Se cuenta, no se argumenta.
  3. **Lo que cada representacion sabe**, con una sonda lineal congelada sobre
     la MISMA tarea de etiquetado gramatical de los dos articulos anteriores,
     asi que la columna se puede poner al lado de las suyas.

Ejecutar desde cualquier sitio: `python src/utils/generate_pretraining_data.py`.
"""
import base64
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_sequences_data import SEED, load_split, threads, vocabularies  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "pretraining" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "pretraining_cache")

D_MODEL = 128
HEADS = 4
LAYERS = 3
SEQ = 48
STEPS = 1600
BATCH = 48
MASK_RATE = 0.15


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


# ------------------------------------------------------------------ los datos
def streams(sents, wi, seq=SEQ):
    """El corpus como un rio de tokens cortado en trozos de longitud fija.

    Cortar por longitud fija y no por frase es lo que hacen los tres modelos de
    verdad, y cambia el problema: un trozo empieza a mitad de una frase y acaba
    a mitad de otra, asi que el contexto util no siempre esta dentro. Se hace
    igual en los tres brazos, que es lo que importa aqui.
    """
    flat = []
    for s in sents:
        for w, _ in s:
            flat.append(wi.get(w.lower(), 1))
    n = (len(flat) // seq) * seq
    return np.array(flat[:n], dtype=np.int64).reshape(-1, seq)


def special_vocab(words):
    """El vocabulario mas los tokens que los objetivos necesitan.

    <pad> 0 y <unk> 1 ya vienen del articulo anterior; aqui se anaden <mask>,
    que es lo que BERT pone donde borro, y los centinelas de tramo de T5, que
    son lo que marca el hueco en la entrada y lo que lo abre en la salida.
    """
    extra = ["<mask>", "<bos>", "<eos>"] + [f"<x{i}>" for i in range(8)]
    return list(words) + extra


class Cfg:
    pass


def build(kind, V, d=D_MODEL, heads=HEADS, layers=LAYERS, seed=SEED):
    """Un unico bloque, tres formas de conectarlo, un presupuesto igualado.

    El encoder-decoder de T5 tiene atencion cruzada de mas, asi que con las
    mismas capas tendria mas parametros y la comparacion mediria el tamano. Se
    le dan menos capas por lado hasta que los tres recuentos caen dentro del 3%,
    y los tres recuentos se publican en la tabla.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class Attn(nn.Module):
        def __init__(self, cross=False):
            super().__init__()
            self.h = heads
            self.dk = d // heads
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            self.v = nn.Linear(d, d)
            self.o = nn.Linear(d, d)
            self.cross = cross

        def forward(self, x, mem=None, causal=False, pad=None):
            src = x if mem is None else mem
            B, L, _ = x.shape
            S = src.shape[1]
            q = self.q(x).view(B, L, self.h, self.dk).transpose(1, 2)
            k = self.k(src).view(B, S, self.h, self.dk).transpose(1, 2)
            v = self.v(src).view(B, S, self.h, self.dk).transpose(1, 2)
            a = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
            if causal:
                m = torch.triu(torch.ones(L, S, dtype=torch.bool), diagonal=1)
                a = a.masked_fill(m, float("-inf"))
            if pad is not None:
                a = a.masked_fill(pad[:, None, None, :], float("-inf"))
            w = a.softmax(-1)
            out = (w @ v).transpose(1, 2).reshape(B, L, d)
            return self.o(out)

    class Block(nn.Module):
        def __init__(self, cross=False):
            super().__init__()
            self.a = Attn()
            self.na = nn.LayerNorm(d)
            self.x = Attn(cross=True) if cross else None
            self.nx = nn.LayerNorm(d) if cross else None
            self.nf = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 3 * d), nn.GELU(), nn.Linear(3 * d, d))

        def forward(self, h, mem=None, causal=False, pad=None):
            h = h + self.a(self.na(h), causal=causal, pad=pad)
            if self.x is not None and mem is not None:
                h = h + self.x(self.nx(h), mem=mem)
            return h + self.ff(self.nf(h))

    class Enc(nn.Module):
        """BERT y GPT comparten esto; solo cambia si la mascara es causal."""

        def __init__(self, causal, n):
            super().__init__()
            self.emb = nn.Embedding(V, d, padding_idx=0)
            self.pos = nn.Embedding(512, d)
            self.blocks = nn.ModuleList([Block() for _ in range(n)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, V)
            self.causal = causal

        def encode(self, x):
            p = torch.arange(x.shape[1]).unsqueeze(0)
            h = self.emb(x) + self.pos(p)
            for b in self.blocks:
                h = b(h, causal=self.causal)
            return self.norm(h)

        def forward(self, x):
            return self.head(self.encode(x))

    class EncDec(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.emb = nn.Embedding(V, d, padding_idx=0)
            self.pos = nn.Embedding(512, d)
            self.enc = nn.ModuleList([Block() for _ in range(n)])
            self.dec = nn.ModuleList([Block(cross=True) for _ in range(n)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, V)

        def encode(self, x):
            p = torch.arange(x.shape[1]).unsqueeze(0)
            h = self.emb(x) + self.pos(p)
            for b in self.enc:
                h = b(h)
            return self.norm(h)

        def forward(self, x, y):
            mem = self.encode(x)
            p = torch.arange(y.shape[1]).unsqueeze(0)
            h = self.emb(y) + self.pos(p)
            for b in self.dec:
                h = b(h, mem=mem, causal=True)
            return self.head(self.norm(h))

    if kind == "bert":
        return Enc(False, layers)
    if kind == "gpt":
        return Enc(True, layers)
    # el encoder-decoder lleva atencion cruzada, asi que cabe menos hondo
    n = layers
    while n > 1:
        m = EncDec(n)
        ref = sum(p.numel() for p in Enc(False, layers).parameters())
        if sum(p.numel() for p in m.parameters()) <= ref * 1.03:
            return m
        n -= 1
    return EncDec(1)


def corrupt_mlm(x, mask_id, V, rate, rng):
    """El enmascarado de BERT, con la receta 80/10/10 que el paper si especifica.

    De los tokens elegidos, el 80% se sustituye por <mask>, el 10% por una
    palabra al azar y el 10% se deja igual (pero se predice igual). Sin esa
    receta el modelo aprende que <mask> es la unica posicion que importa y no
    generaliza a texto sin mascaras, que es todo el texto que vera despues.
    """
    y = np.full(x.shape, -100, dtype=np.int64)
    pick = rng.random(x.shape) < rate
    pick &= x > 0
    y[pick] = x[pick]
    xx = x.copy()
    r = rng.random(x.shape)
    xx[pick & (r < 0.8)] = mask_id
    rand = pick & (r >= 0.8) & (r < 0.9)
    xx[rand] = rng.integers(2, V, size=int(rand.sum()))
    return xx, y, float(pick.mean())


def corrupt_spans(x, sent_ids, rng, rate=0.15, mean_len=3):
    """La corrupcion por tramos de T5: se quitan trozos, no tokens sueltos.

    Cada tramo borrado deja un centinela en la entrada, y la salida es la
    secuencia de centinelas con lo que habia detras de cada uno. Es una tarea
    mas dura que enmascarar tokens sueltos, porque para rellenar tres palabras
    seguidas no basta con mirar las dos de al lado.
    """
    B, L = x.shape
    xs, ys = [], []
    for b in range(B):
        keep = np.ones(L, dtype=bool)
        spans = []
        budget = int(rate * L)
        used = 0
        while used < budget and len(spans) < len(sent_ids):
            ln = max(1, int(rng.poisson(mean_len)))
            st = int(rng.integers(0, max(1, L - ln)))
            if not keep[st:st + ln].all():
                continue
            keep[st:st + ln] = False
            spans.append((st, ln))
            used += ln
        spans.sort()
        inp, out = [], []
        i = 0
        si = 0
        while i < L:
            hit = next((s for s in spans if s[0] == i), None)
            if hit and si < len(sent_ids):
                inp.append(sent_ids[si])
                out.append(sent_ids[si])
                out.extend(int(v) for v in x[b, i:i + hit[1]])
                i += hit[1]
                si += 1
            else:
                inp.append(int(x[b, i]))
                i += 1
        xs.append(inp)
        ys.append(out)
    n = max(len(v) for v in xs)
    m = max(len(v) for v in ys)
    X = np.zeros((B, n), dtype=np.int64)
    Y = np.zeros((B, m), dtype=np.int64)
    for b in range(B):
        X[b, :len(xs[b])] = xs[b]
        Y[b, :len(ys[b])] = ys[b]
    return X, Y


def pretrain(kind, data, V, ids, steps=STEPS, rate=MASK_RATE, seed=SEED, label=""):
    """Los tres objetivos, con el MISMO numero de pasos y el mismo lote.

    Igualar los pasos y no el tiempo es deliberado: un tiempo de pared mide la
    maquina, y este sitio ya pago esa leccion dos veces. Lo que se cuenta al
    lado es cuantas predicciones supervisadas recibe cada objetivo en esos
    pasos, que es donde esta la diferencia real y no en el reloj.
    """
    import torch
    import torch.nn as nn
    threads()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = build(kind, V, seed=seed)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    hist = []
    supervised = 0
    seen = 0
    for step in range(steps):
        b = rng.integers(0, len(data), BATCH)
        x = data[b]
        if kind == "bert":
            xx, yy, frac = corrupt_mlm(x, ids["mask"], V, rate, rng)
            logits = m(torch.tensor(xx))
            y = torch.tensor(yy)
            loss = lossf(logits.reshape(-1, V), y.reshape(-1))
            supervised += int((yy >= 0).sum())
        elif kind == "gpt":
            xt = torch.tensor(x)
            logits = m(xt[:, :-1])
            y = xt[:, 1:]
            loss = lossf(logits.reshape(-1, V), y.reshape(-1))
            supervised += int(y.numel())
        else:
            X, Y = corrupt_spans(x, ids["sentinels"], rng, rate=rate)
            Xt = torch.tensor(X)
            Yt = torch.tensor(Y)
            logits = m(Xt, Yt[:, :-1])
            loss = lossf(logits.reshape(-1, V), Yt[:, 1:].reshape(-1))
            supervised += int((Yt[:, 1:] > 0).sum())
        seen += int(x.size)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (step + 1) % 200 == 0:
            hist.append({"step": step + 1, "loss": float(loss)})
            print(f"      {label} paso {step + 1}: perdida {float(loss):.4f}")
    # Se devuelve el state_dict como numpy y no el modelo. Las clases del
    # modelo se declaran DENTRO de `build`, asi que son objetos locales y
    # `pickle` no puede con ellas: la primera tirada entreno BERT durante 1.600
    # pasos y murio al guardar en cache, que es la peor forma de perder trabajo.
    # Guardando pesos, el cache es portable y `rebuild` los vuelve a montar.
    return {"state": {k: v.detach().numpy() for k, v in m.state_dict().items()},
            "hist": hist, "supervised": supervised, "tokens_seen": seen,
            "params": sum(p.numel() for p in m.parameters())}


def rebuild(kind, V, bundle):
    """Vuelve a montar el modelo desde los pesos que el cache guarda."""
    import torch
    m = build(kind, V)
    m.load_state_dict({k: torch.tensor(v) for k, v in bundle["state"].items()})
    m.eval()
    return m


def probe(m, kind, train_s, test_s, wi, ti, V, seq=SEQ, seed=SEED, epochs=3):
    """Una sonda lineal CONGELADA sobre la tarea de etiquetado de los dos anteriores.

    Congelada de verdad: el unico parametro que se ajusta es una matriz de la
    dimension del modelo por el numero de etiquetas. Si se dejase entrenar el
    cuerpo, lo que se mediria seria la capacidad del cuerpo y no lo que el
    preentrenamiento dejo dentro, que es la pregunta.
    """
    import torch
    import torch.nn as nn
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

    def rep(x):
        with torch.no_grad():
            return m.encode(x)

    def batches(sents):
        out = []
        cur_x, cur_y = [], []
        for s in sents:
            for w, t in s:
                cur_x.append(wi.get(w.lower(), 1))
                cur_y.append(ti[t])
                if len(cur_x) == seq:
                    out.append((cur_x, cur_y))
                    cur_x, cur_y = [], []
        X = torch.tensor([a for a, _ in out])
        Y = torch.tensor([b for _, b in out])
        return X, Y

    Xtr, Ytr = batches(train_s)
    Xte, Yte = batches(test_s)
    head = nn.Linear(D_MODEL, len(ti))
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    B = 64
    torch.manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), B):
            b = perm[s:s + B]
            h = rep(Xtr[b])
            loss = lossf(head(h).reshape(-1, len(ti)), Ytr[b].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    hit = tot = 0
    with torch.no_grad():
        for s in range(0, len(Xte), B):
            h = rep(Xte[s:s + B])
            p = head(h).argmax(-1)
            hit += int((p == Yte[s:s + B]).sum())
            tot += int(p.numel())
    return hit / max(1, tot)


def cloze(m, kind, data, V, ids, seed=SEED, n=400):
    """Adivinar una palabra borrada, que es lo unico que los tres pueden hacer.

    Es la comparacion limpia entre bidireccional y causal: se borra una palabra
    en mitad del trozo y se pregunta cual era. BERT ve los dos lados, GPT solo
    la izquierda, y T5 ve los dos lados en la entrada. Es la tarea que favorece
    a BERT por construccion, y por eso hay que decir que lo hace en vez de
    presentarla como neutral.
    """
    import torch
    rng = np.random.default_rng(seed + 7)
    idx = rng.integers(0, len(data), n)
    x = data[idx]
    pos = SEQ // 2
    gold = x[:, pos].copy()
    hit = 0
    with torch.no_grad():
        if kind == "bert":
            xx = x.copy()
            xx[:, pos] = ids["mask"]
            p = m(torch.tensor(xx))[:, pos].argmax(-1).numpy()
        elif kind == "gpt":
            p = m(torch.tensor(x[:, :pos]))[:, -1].argmax(-1).numpy()
        else:
            xx = x.copy()
            xx[:, pos] = ids["sentinels"][0]
            y = np.stack([np.array([ids["sentinels"][0]])] * len(x))
            p = m(torch.tensor(xx), torch.tensor(y))[:, -1].argmax(-1).numpy()
    hit = int((p == gold).sum())
    return hit / len(gold)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("la misma particion del Brown que los dos articulos anteriores")
    train_s, test_s = cached("split", load_split)
    words, wi, tags, ti = vocabularies(train_s)
    vocab = special_vocab(words)
    V = len(vocab)
    ids = {"mask": vocab.index("<mask>"),
           "sentinels": [vocab.index(f"<x{i}>") for i in range(8)]}
    tr = streams(train_s, wi)
    te = streams(test_s, wi)
    print(f"  {len(tr)} trozos de {SEQ} tokens para entrenar, {len(te)} para test, "
          f"vocabulario {V}")

    print("los tres objetivos, mismo presupuesto y mismos pasos")
    arms = {}
    for kind, label in (("bert", "BERT "), ("gpt", "GPT  "), ("t5", "T5   ")):
        arms[kind] = cached(f"arm_{kind}",
                            lambda k=kind, l=label: pretrain(k, tr, V, ids, label=l))
        print(f"    {label} {arms[kind]['params']:,} parametros, "
              f"{arms[kind]['supervised']:,} predicciones supervisadas")

    print("la sonda lineal congelada, sobre la tarea de etiquetado")
    rows = []
    for kind in ("bert", "gpt", "t5"):
        model = rebuild(kind, V, arms[kind])
        acc = cached(f"probe_{kind}",
                     lambda k=kind, mm=model: probe(mm, k, train_s, test_s, wi, ti, V))
        cz = cached(f"cloze_{kind}", lambda k=kind, mm=model: cloze(mm, k, te, V, ids))
        b = arms[kind]
        rows.append({
            "model": {"bert": "BERT, masked", "gpt": "GPT, causal",
                      "t5": "T5, span corruption"}[kind],
            "kind": kind,
            "params": b["params"],
            "supervised": b["supervised"],
            "tokens_seen": b["tokens_seen"],
            "signal_per_token": b["supervised"] / b["tokens_seen"],
            "probe": acc,
            "cloze": cz,
            "final_loss": b["hist"][-1]["loss"],
            "can_generate": kind in ("gpt", "t5"),
            "sees_both_sides": kind in ("bert", "t5"),
        })
        print(f"    {kind:<5} sonda {acc:.4f}, cloze {cz:.4f}, "
              f"senal por token {rows[-1]['signal_per_token']:.4f}")

    print("el barrido de la tasa de enmascarado")
    sweep = []
    for rate in (0.05, 0.15, 0.30, 0.40, 0.50):
        b = cached(f"rate_{int(rate * 100)}",
                   lambda r=rate: pretrain("bert", tr, V, ids, rate=r,
                                           label=f"BERT {int(r * 100)}%"))
        acc = cached(f"rate_probe_{int(rate * 100)}",
                     lambda bb=b: probe(rebuild("bert", V, bb), "bert", train_s, test_s,
                                        wi, ti, V))
        sweep.append({"rate": rate, "probe": acc,
                      "supervised": b["supervised"],
                      "final_loss": b["hist"][-1]["loss"]})
        print(f"    {int(rate * 100)}% enmascarado: sonda {acc:.4f}, "
              f"{b['supervised']:,} predicciones")

    data = {
        "meta": {
            "corpus": "the same Brown split the two previous articles use",
            "chunks_train": len(tr), "chunks_test": len(te),
            "seq": SEQ, "vocab": V, "d_model": D_MODEL, "heads": HEADS,
            "layers": LAYERS, "steps": STEPS, "batch": BATCH, "seed": SEED,
            "tags": tags, "mask_rate": MASK_RATE,
        },
        "arms": rows,
        "hist": {k: v["hist"] for k, v in arms.items()},
        "rate_sweep": sweep,
    }
    path = OUT / "pretraining.json"
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
