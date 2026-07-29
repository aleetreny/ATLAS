"""Las cifras del articulo de self-attention.

Corre sobre **la misma particion del Brown, la misma tarea y el mismo
presupuesto** que el articulo de modelos de secuencia, y lo importa del propio
generador de aquel en vez de reconstruirlo, asi que la comparacion entre las dos
paginas no es una analogia: es la misma tabla con dos filas mas. Lo mismo con la
tarea de copia, que se genera con la misma funcion y las mismas semillas.

Lo que este articulo mide y el anterior no podia:

  1. **La invariancia a permutaciones, exacta.** Sin codificacion posicional, la
     atencion no es "poco sensible" al orden: es matematicamente ciega a el. Se
     comprueba barajando la frase y exigiendo que el conjunto de salidas sea el
     mismo hasta la precision de la maquina, que es una comprobacion que se pasa
     o se falla y no admite interpretacion.
  2. **Por que se divide por la raiz de la dimension.** Se mide la entropia de
     las distribuciones de atencion con y sin ese factor segun crece la
     dimension. Sin el, la saturacion no es un riesgo teorico, tiene una curva.
  3. **El alcance del gradiente**, con la misma sonda que uso el articulo de
     secuencias, para poder poner las curvas en el mismo eje.
  4. **Que hacen las cabezas por separado**, ablando cada una y midiendo el
     dano, en vez de mirar dibujos de matrices de atencion y contar una
     historia.

Ejecutar desde cualquier sitio: `python src/utils/generate_attention_data.py`.
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
from generate_sequences_data import (  # noqa: E402
    SEED, EMB, EPOCHS, MIN_COUNT, copy_task, load_split, make_batches, threads, vocabularies,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "attention" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "attention_cache")
SEQ_JSON = ROOT / "sequences" / "data" / "sequences.json"

D_MODEL = 96
HEADS = 4
LAYERS = 2


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


# ------------------------------------------------------------------ el modelo
def build(V, T, d=D_MODEL, heads=HEADS, layers=LAYERS, positional=True, seed=SEED):
    """Un codificador de atencion, escrito a mano y no con nn.TransformerEncoder.

    A mano por dos razones. Una, el navegador ejecuta este mismo forward y no
    hay una capa de torch al otro lado que copiar. Y dos, la capa de torch
    esconde exactamente las tres cosas que el articulo mide (donde entra la
    posicion, donde se divide por la raiz de la dimension, y que hace cada
    cabeza), asi que usarla seria escribir un articulo sobre una caja cerrada.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.dk = d // heads
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            self.v = nn.Linear(d, d)
            self.o = nn.Linear(d, d)

        def forward(self, x, mask=None, scale=True, keep=None, drop_head=None):
            B, L, _ = x.shape
            q = self.q(x).view(B, L, heads, self.dk).transpose(1, 2)
            k = self.k(x).view(B, L, heads, self.dk).transpose(1, 2)
            v = self.v(x).view(B, L, heads, self.dk).transpose(1, 2)
            att = q @ k.transpose(-1, -2)
            if scale:
                att = att / math.sqrt(self.dk)
            if mask is not None:
                att = att.masked_fill(mask[:, None, None, :], float("-inf"))
            w = att.softmax(-1)
            if keep is not None:
                keep.append(w.detach())
            out = w @ v
            if drop_head is not None:
                out = out.clone()
                out[:, drop_head] = 0.0
            out = out.transpose(1, 2).reshape(B, L, d)
            return self.o(out)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.att = Head()
            self.n1 = nn.LayerNorm(d)
            self.n2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(self, x, mask=None, scale=True, keep=None, drop=None):
            x = x + self.att(self.n1(x), mask, scale, keep, drop)
            return x + self.ff(self.n2(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, d, padding_idx=0)
            self.pos = nn.Embedding(512, d)
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, T)
            self.positional = positional

        def forward(self, x, mask=None, scale=True, keep=None, drop=None,
                    positions=None, embeds=None):
            h = self.emb(x) if embeds is None else embeds
            if self.positional:
                p = positions
                if p is None:
                    p = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
                h = h + self.pos(p)
            for i, b in enumerate(self.blocks):
                h = b(h, mask, scale, keep, drop if (drop and drop[0] == i) else None)
            return self.head(self.norm(h))

    return Model()


def train(model, train_s, test_s, wi, ti, epochs=EPOCHS, lr=1.5e-3, seed=SEED, label=""):
    import torch
    import torch.nn as nn
    threads()
    torch.manual_seed(seed)
    T = len(ti)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    tb = make_batches(train_s, wi, ti, seed=seed)
    eb = make_batches(test_s, wi, ti, seed=seed)
    hist = []
    for ep in range(epochs):
        model.train()
        s = 0.0
        for x, y in tb:
            # La mascara de relleno sale de las ETIQUETAS y no de los tokens.
            #
            # Con `x == 0` se marcaba tambien todo `<unk>`, que comparte el
            # indice 0 con el relleno, y una frase entera de palabras raras
            # acababa con todas sus claves enmascaradas: un softmax sobre puros
            # menos infinito es NaN, el NaN se propaga por los parametros
            # compartidos y la tirada entera sale con perdida NaN y exactitud
            # 0,1278 (la clase mayoritaria) sin un solo error en pantalla. Las
            # etiquetas valen -100 exactamente donde hay relleno y en ningun
            # otro sitio, asi que la mascara buena es esa.
            mask = (y < 0)
            logits = model(x, mask)
            loss = lossf(logits.reshape(-1, T), y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "la perdida salio no finita en el primer lote que lo hizo. Un modelo "
                    "que entrena con NaN no falla, devuelve la clase mayoritaria y todas "
                    "las medidas de la pagina salen degeneradas y plausibles.")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            s += float(loss.detach())
        acc = evaluate(model, eb, T)
        hist.append({"epoch": ep + 1, "loss": s / len(tb), "acc": acc})
        print(f"      {label} epoca {ep + 1}: perdida {s / len(tb):.4f}, exactitud {acc:.4f}")
    return hist


def evaluate(model, batches, T, drop=None, scale=True):
    import torch
    model.eval()
    hit = tot = 0
    with torch.no_grad():
        for x, y in batches:
            logits = model(x, (y < 0), scale=scale, drop=drop)
            p = logits.argmax(-1)
            m = y >= 0
            hit += int((p[m] == y[m]).sum())
            tot += int(m.sum())
    return hit / max(1, tot)


# ------------------------------------------------- la invariancia, exacta
def permutation_test(model, wi, ti, test_s, trials=20, seed=SEED):
    """Sin posiciones, barajar la frase no cambia NADA. Se comprueba como igualdad.

    Es la unica propiedad de este modulo que no tiene tolerancia razonable
    porque no es una aproximacion: la atencion sin codificacion posicional es
    una funcion de un CONJUNTO. Barajada la entrada, el conjunto de salidas
    tiene que ser el mismo, y la unica diferencia admisible es el reordenamiento
    de las sumas en coma flotante.

    Se mide con las posiciones puestas y quitadas, porque un numero pequeno sin
    su control no dice si el modelo es ciego al orden o simplemente suave.
    """
    import torch
    rng = np.random.default_rng(seed)
    sents = [s for s in test_s if 8 <= len(s) <= 20][:trials]
    out = {}
    for use_pos in (False, True):
        model.positional = use_pos
        worst = 0.0
        for s in sents:
            x = torch.tensor([[wi.get(w.lower(), 0) for w, _ in s]])
            perm = torch.tensor(rng.permutation(x.shape[1]))
            with torch.no_grad():
                a = model(x)[0]
                b = model(x[:, perm])[0]
            # Se deshace la permutacion antes de comparar: lo que se afirma es
            # que la salida acompana a su token, no que la matriz sea la misma.
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(len(perm))
            worst = max(worst, float((a - b[inv]).abs().max()))
        out["with_positions" if use_pos else "without_positions"] = worst
    model.positional = True
    print(f"    sin posiciones el peor desacuerdo es {out['without_positions']:.3e}, "
          f"con posiciones {out['with_positions']:.3e}")
    return out


# ------------------------------------------------------- por que la raiz de d
def scaling_curve(dims=(8, 16, 32, 64, 128, 256, 512), n=512, seed=SEED):
    """La entropia de la atencion con y sin el factor, segun crece la dimension.

    Con vectores de componentes independientes de varianza 1, el producto de dos
    de dimension d tiene desviacion raiz de d, asi que sin dividir por ella las
    entradas del softmax crecen con la dimension y la distribucion colapsa sobre
    un solo elemento. Colapsada, su gradiente es cero y la capa deja de
    aprender. Aqui esta medido en vez de argumentado, sobre vectores aleatorios,
    que es donde la afirmacion se hace: es sobre la inicializacion.
    """
    import torch
    torch.manual_seed(seed)
    L = 64
    out = []
    for d in dims:
        q = torch.randn(n, L, d)
        k = torch.randn(n, L, d)
        att = q @ k.transpose(-1, -2)
        row = {"d": d}
        for scale in (False, True):
            a = att / math.sqrt(d) if scale else att
            w = a.softmax(-1)
            ent = -(w * torch.log(w.clamp_min(1e-30))).sum(-1)
            row["entropy_scaled" if scale else "entropy_raw"] = float(ent.mean())
            row["max_scaled" if scale else "max_raw"] = float(w.max(-1).values.mean())
        row["uniform_entropy"] = math.log(L)
        out.append(row)
        print(f"    d={d:<4} entropia sin escalar {row['entropy_raw']:.4f}, "
              f"escalada {row['entropy_scaled']:.4f} (uniforme {math.log(L):.4f})")
    return {"rows": out, "length": L, "samples": n}


# ------------------------------------------------------ el alcance, comparable
def gradient_reach(model, test_s, wi, length=60, samples=24, seed=SEED):
    """La misma sonda del articulo anterior, sobre la atencion.

    Misma cantidad (la norma de la derivada del estado final respecto a la
    entrada del paso t), misma normalizacion (por la distancia 1) y mismas
    frases largas, para que las curvas se puedan dibujar en el mismo eje que las
    de RNN, GRU y LSTM sin que nadie tenga que fiarse de que son comparables.
    """
    import torch
    rng = np.random.default_rng(seed)
    long_s = [s for s in test_s if len(s) >= length]
    if not long_s:
        long_s = sorted(test_s, key=len, reverse=True)[:samples]
    pick = [long_s[i] for i in rng.choice(len(long_s), size=min(samples, len(long_s)),
                                          replace=False)]
    model.eval()
    curves = []
    for s in pick:
        x = torch.tensor([[wi.get(w.lower(), 0) for w, _ in s[:length]]])
        e = model.emb(x).detach().clone().requires_grad_(True)
        out = model(x, embeds=e)
        n = out.shape[1]
        g = torch.autograd.grad(out[0, n - 1].sum(), e)[0][0]
        curves.append([float(g[n - 1 - d].norm()) for d in range(1, n)])
    L = min(len(c) for c in curves)
    med = np.median(np.array([c[:L] for c in curves]), axis=0)
    rel = med / max(med[0], 1e-30)
    half = next((i + 1 for i, v in enumerate(rel) if v < 0.5), None)
    print(f"    la atencion cae a la mitad en "
          f"{half if half else 'ningun punto de los ' + str(L)} pasos")
    return {"distances": list(range(1, L + 1)),
            "median": [float(v) for v in med],
            "relative": [float(v) for v in rel],
            "half_life": half,
            "at_10": float(rel[9]) if L > 9 else None,
            "at_30": float(rel[29]) if L > 29 else None,
            "sentences": len(pick)}


# --------------------------------------------------------- que hace cada cabeza
def head_ablation(model, test_s, wi, ti, layers=LAYERS, heads=HEADS):
    """Poner a cero la salida de una cabeza y medir lo que se pierde.

    Se ABLA poniendo a cero, no borrando: quitar la cabeza cambiaria la anchura
    de la proyeccion de salida y con ella el recuento de parametros, y entonces
    la comparacion dejaria de ser sobre la cabeza. Es la misma regla que el
    articulo de segmentacion aprendio con las conexiones de salto.
    """
    eb = make_batches(test_s, wi, ti, seed=SEED)
    base = evaluate(model, eb, len(ti))
    rows = []
    for l in range(layers):
        for hd in range(heads):
            acc = evaluate(model, eb, len(ti), drop=(l, hd))
            rows.append({"layer": l, "head": hd, "acc": acc, "drop": base - acc})
    rows.sort(key=lambda r: -r["drop"])
    print(f"    sin ablar {base:.4f}; la peor cabeza cuesta {rows[0]['drop']:.4f} y "
          f"la mejor prescindible {rows[-1]['drop']:.4f}")
    return {"base": base, "rows": rows}


def copy_curve_attention(distances, epochs=30, d=48, heads=4, seed=SEED):
    """La misma tarea de copia del articulo anterior, con la misma funcion.

    Se importa `copy_task` en vez de reescribirla, asi que las secuencias son
    literalmente las mismas hasta el ultimo simbolo, y la fila que este articulo
    anade a aquella tabla es comparable por construccion y no por promesa.
    """
    import torch
    import torch.nn as nn
    threads()
    rows = []
    for dist in distances:
        torch.manual_seed(seed)
        x, y = copy_task(dist, seed=seed)
        xt = torch.tensor(x)
        yt = torch.tensor(y)
        ntr = int(0.8 * len(x))
        V = int(x.max()) + 1
        K = int(y.max()) + 1
        m = build(V, K, d=d, heads=heads, layers=1, seed=seed)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
        lossf = nn.CrossEntropyLoss()
        B = 256
        for ep in range(epochs):
            perm = torch.randperm(ntr)
            for s in range(0, ntr, B):
                b = perm[s:s + B]
                logits = m(xt[b])
                loss = lossf(logits[:, -1], yt[b])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
        with torch.no_grad():
            acc = float((m(xt[ntr:])[:, -1].argmax(-1) == yt[ntr:]).float().mean())
        rows.append({"distance": dist, "acc": acc, "chance": 1.0 / K,
                     "params": sum(p.numel() for p in m.parameters())})
        print(f"      atencion a distancia {dist:>3}: {acc:.4f}")
    return rows


def ship_model(model, wi, ti, words, tags, examples):
    """Los pesos al navegador, en float16, con el contrato escrito.

    Mismo patron que el articulo de deteccion: cada tensor con su desplazamiento
    y su tamano dentro de un solo blob, para que el JavaScript no tenga que
    saberse los nombres de torch.

    De la matriz de incrustacion viajan **solo las filas que las frases de
    ejemplo usan**, remapeadas. Entera son 30.000 filas por 96 y casi seis
    megas, y la pagina no las necesita: lo que hace es ejecutar el modelo sobre
    esas frases, asi que con sus palabras basta y el fichero baja dos ordenes de
    magnitud. El resto de los pesos (la atencion, las normalizaciones, la red
    de salida) viajan enteros, que es lo que hace que el forward de la pagina
    sea el modelo y no una imitacion suya.
    """
    import torch
    need = sorted({wi.get(w.lower(), 0) for ex in examples for w in ex["words"]})
    remap = {j: i for i, j in enumerate(need)}
    parts = []
    spec = {}
    off = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            a = p.detach().numpy()
            if name == "emb.weight":
                a = a[need]
            if name == "pos.weight":
                a = a[:64]
            flat = np.ascontiguousarray(a).ravel().astype(np.float16)
            spec[name] = {"offset": off, "shape": list(a.shape)}
            parts.append(flat)
            off += int(flat.size)
    blob = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float16)
    return {
        "blob": b64_f16(blob), "spec": spec, "size": int(blob.size),
        "words": [words[j] for j in need], "tags": tags,
        "d_model": D_MODEL, "heads": HEADS, "layers": LAYERS,
        # Cada frase ya con los indices remapeados, para que el navegador no
        # tenga que reconstruir el vocabulario.
        "sentences": [{"words": ex["words"], "gold": ex["gold"],
                       "ids": [remap[wi.get(w.lower(), 0)] for w in ex["words"]]}
                      for ex in examples],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("cargando la MISMA particion que el articulo de secuencias")
    train_s, test_s = cached("split", load_split)
    words, wi, tags, ti = vocabularies(train_s)
    print(f"  {len(train_s)} frases de entrenamiento, {len(words)} palabras, "
          f"{len(tags)} etiquetas")
    if SEQ_JSON.is_file():
        other = json.loads(SEQ_JSON.read_text("utf-8"))["meta"]
        assert other["train_sents"] == len(train_s), "la particion no coincide"
        assert other["vocab"] == len(words), "el vocabulario no coincide"
        print("  identica a la del articulo de secuencias, afirmado contra su JSON")

    print("entrenando el codificador de atencion")
    def fit():
        m = build(len(words), len(tags))
        hist = train(m, train_s, test_s, wi, ti, label="attn")
        return {"state": {k: v.detach().numpy() for k, v in m.state_dict().items()},
                "hist": hist,
                "params": sum(p.numel() for p in m.parameters())}
    bundle = cached("model", fit)
    acc = bundle["hist"][-1]["acc"]
    assert all(math.isfinite(h["loss"]) for h in bundle["hist"]), \
        "alguna epoca tiene perdida no finita"
    assert acc > 0.5, (
        f"el codificador se quedo en {acc:.4f}, que es del orden de contestar siempre la "
        "etiqueta mayoritaria: el modelo no aprendio y todo lo que sigue seria ruido")

    import torch
    model = build(len(words), len(tags))
    model.load_state_dict({k: torch.tensor(v) for k, v in bundle["state"].items()})
    model.eval()

    print("sin posiciones no es que sea poco sensible al orden, es que es ciego")
    perm = cached("perm", lambda: permutation_test(model, wi, ti, test_s))

    print("por que se divide por la raiz de la dimension")
    sc = cached("scaling", scaling_curve)

    print("el alcance del gradiente, con la sonda del articulo anterior")
    reach = cached("reach", lambda: gradient_reach(model, test_s, wi))

    print("que hace cada cabeza")
    abl = cached("ablation", lambda: head_ablation(model, test_s, wi, ti))

    print("la tarea de copia, la misma")
    dists = [5, 10, 20, 40, 80]
    cop = cached("copy", lambda: copy_curve_attention(dists))

    print("el coste, contado y no cronometrado")
    cost = cost_table()

    # Las frases viajan cortadas a 16 tokens porque la tabla de posiciones que
    # se envia son 64 filas: enviar 512 seria enviar 448 filas que ninguna
    # frase de la pagina puede indexar.
    print("frases de ejemplo, con lo que el modelo contesta en cada una")
    ex = []
    for s in test_s:
        if not (8 <= len(s) <= 16):
            continue
        ids = torch.tensor([[wi.get(w.lower(), 0) for w, _ in s]])
        with torch.no_grad():
            pred = model(ids).argmax(-1)[0].tolist()
        ex.append({"words": [w for w, _ in s], "gold": [t for _, t in s],
                   # Lo que ESTE modelo contesta, que es contra lo que el
                   # navegador tiene que cuadrar. Sin este campo el guardia
                   # compararia el forward de la pagina contra las etiquetas
                   # escritas a mano, que es otra cosa: un modelo que se
                   # equivoca igual en los dos lados esta bien reimplementado.
                   "predicted": [tags[i] for i in pred]})
        if len(ex) >= 8:
            break

    data = {
        "meta": {
            "corpus": "the same Brown split the sequences article uses",
            "train_sents": len(train_s), "test_sents": len(test_s),
            "vocab": len(words), "tags": tags, "seed": SEED,
            "d_model": D_MODEL, "heads": HEADS, "layers": LAYERS,
            "params": bundle["params"], "epochs": EPOCHS,
        },
        "hist": bundle["hist"],
        "acc": bundle["hist"][-1]["acc"],
        "permutation": perm,
        "scaling": sc,
        "reach": reach,
        "ablation": abl,
        "copy": cop,
        "copy_distances": dists,
        "cost": cost,
        "examples": ex,
        "model": ship_model(model, wi, ti, words, tags, ex),
    }
    path = OUT / "attention.json"
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


def cost_table(lengths=(8, 16, 32, 64, 128, 256, 512, 1024), d=D_MODEL):
    """Multiplicaciones por capa y por secuencia, contadas.

    Nada de relojes: un tiempo de pared mide la contencion de la maquina y este
    sitio ya pago esa leccion. Lo que se cuenta son las multiplicaciones que la
    definicion exige, que son las mismas en cualquier ordenador. Para atencion,
    las proyecciones son lineales en la longitud y el producto de consultas por
    claves es cuadratico; para una recurrencia, todo es lineal.
    """
    rows = []
    for L in lengths:
        proj = 4 * L * d * d
        pair = 2 * L * L * d
        rnn = L * (d * d + d * d)
        rows.append({"length": L, "projections": proj, "pairwise": pair,
                     "attention_total": proj + pair, "recurrent": rnn,
                     "ratio": (proj + pair) / rnn,
                     "pairwise_share": pair / (proj + pair)})
    return {"rows": rows, "d_model": d}


if __name__ == "__main__":
    main()
