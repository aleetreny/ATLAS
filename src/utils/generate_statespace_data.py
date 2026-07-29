"""Las cifras del articulo de modelos de espacio de estados (SSM, S4, Mamba).

El articulo cierra el modulo volviendo al sitio donde empezo la parte
secuencial: una recurrencia. La diferencia con la del articulo de RNN es que
esta es **lineal**, y de ahi salen las tres cosas que se miden aqui.

  1. **Que es dos cosas a la vez.** Una recurrencia lineal se puede desplegar en
     una convolucion con un nucleo que se calcula en forma cerrada, asi que el
     mismo modelo se entrena en paralelo como una convolucion y se ejecuta paso
     a paso como una recurrencia. Eso no es una analogia: son dos calculos que
     tienen que dar el mismo numero, y aqui se comprueba que lo dan, a 1e-13,
     en las dos orillas.
  2. **Que la inicializacion es el modelo.** Con una matriz de transicion
     aleatoria, una recurrencia lineal olvida en unos pocos pasos o explota. Se
     mide el alcance del nucleo con inicializacion aleatoria y con una diagonal
     estable del tipo que usan S4 y Mamba, sobre el mismo entrenamiento.
  3. **Que la selectividad compra memoria, no capacidad.** Un SSM lineal e
     invariante en el tiempo no puede decidir que recordar en funcion de lo que
     esta leyendo, porque sus dinamicas no dependen de la entrada. Mamba las
     hace depender. La tarea de copia SELECTIVA separa las dos cosas, y la de
     copia normal, que es la de los dos articulos anteriores, no.

Todo lo que se compara con articulos anteriores usa sus mismas funciones
importadas, no una reimplementacion: la tarea de copia es literalmente la misma
funcion con la misma semilla.

Ejecutar desde cualquier sitio: `python src/utils/generate_statespace_data.py`.
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
from generate_sequences_data import SEED, copy_task, threads  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "state-space" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "statespace_cache")


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


# --------------------------------------------------- la identidad de las dos formas
def ssm_recurrent(a, b, c, x):
    """h[t] = a h[t-1] + b x[t],  y[t] = c h[t]. Paso a paso, como en inferencia."""
    n = len(a)
    h = np.zeros(n)
    out = np.zeros(len(x))
    for t in range(len(x)):
        h = a * h + b * x[t]
        out[t] = float(c @ h)
    return out


def ssm_kernel(a, b, c, L):
    """El nucleo de la convolucion equivalente: k[t] = c a^t b.

    Sale de desplegar la recurrencia y agrupar: y = k * x. Es la forma en que
    estos modelos se entrenan, porque una convolucion se paraleliza sobre la
    longitud y una recurrencia no.
    """
    powers = np.stack([a ** t for t in range(L)])
    return (powers * (c * b)).sum(axis=1)


def ssm_convolution(k, x):
    out = np.zeros(len(x))
    for t in range(len(x)):
        m = min(t + 1, len(k))
        out[t] = float(k[:m] @ x[t::-1][:m])
    return out


def duality_check(n=16, L=256, trials=12, seed=SEED):
    """Las dos formas, sobre los mismos parametros, comparadas como una igualdad.

    No hay tolerancia que negociar aqui: son dos maneras de escribir la misma
    suma, asi que la unica diferencia admisible es el orden de las operaciones
    en coma flotante. Se prueba con dinamicas lentas y rapidas, porque el error
    de la forma convolucional crece con el numero de terminos que importan.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    rows = []
    for decay in (0.5, 0.9, 0.99, 0.999):
        errs = []
        for _ in range(trials):
            a = decay ** (1 + rng.random(n))
            b = rng.normal(size=n) / math.sqrt(n)
            c = rng.normal(size=n) / math.sqrt(n)
            x = rng.normal(size=L)
            yr = ssm_recurrent(a, b, c, x)
            yc = ssm_convolution(ssm_kernel(a, b, c, L), x)
            scale = max(np.abs(yr).max(), 1e-12)
            errs.append(float(np.abs(yr - yc).max() / scale))
        rows.append({"decay": decay, "worst_relative": max(errs),
                     "median_relative": float(np.median(errs))})
        worst = max(worst, max(errs))
        print(f"    decaimiento {decay}: peor error relativo {max(errs):.3e}")
    return {"rows": rows, "worst": worst, "state": n, "length": L, "trials": trials}


def kernel_reach(n=32, L=256, seed=SEED):
    """Cuanto alcanza el nucleo segun como se inicialice la matriz de transicion.

    Tres inicializaciones sobre la misma forma: gaussiana sin mas, uniforme en
    (0,1) (que es lo que sale de una sigmoide ingenua), y la diagonal estable
    con decaimientos repartidos en escala logaritmica que es la idea de S4 y de
    Mamba. Se mide donde el nucleo cae por debajo de una centesima de su valor
    inicial, que es el equivalente en este modelo de la vida media que el
    articulo de RNN midio sobre el gradiente.
    """
    rng = np.random.default_rng(seed)
    out = []
    inits = {
        "gaussian": lambda: rng.normal(size=n),
        "uniform": lambda: rng.random(n),
        "log-spaced decays": lambda: np.exp(-np.exp(np.linspace(math.log(1e-3),
                                                               math.log(1e-1), n))),
    }
    for name, f in inits.items():
        a = f()
        b = rng.normal(size=n) / math.sqrt(n)
        c = rng.normal(size=n) / math.sqrt(n)
        k = np.abs(ssm_kernel(a, b, c, L))
        k0 = max(k[0], 1e-30)
        rel = k / k0
        blows = bool(np.abs(a).max() > 1.0)
        hund = next((i for i, v in enumerate(rel) if v < 0.01), None)
        # Un nucleo que no cae puede ser bueno o catastrofico y son cosas
        # opuestas, asi que no se pueden imprimir con la misma frase. Con algun
        # |a| > 1 el nucleo no alcanza lejos, DIVERGE, y el maximo lo dice.
        grew = float(rel.max())
        out.append({
            "init": name,
            "hundredth_at": None if blows else hund,
            "max_abs_a": float(np.abs(a).max()),
            "unstable": blows,
            "growth": grew,
            "kernel": [float(v) for v in rel[:128]],
        })
        if blows:
            print(f"    {name:<18} DIVERGE: algun |a| vale {np.abs(a).max():.3f} y el nucleo "
                  f"llega a {grew:.3e} veces su valor inicial")
        else:
            print(f"    {name:<18} cae a la centesima en "
                  f"{hund if hund is not None else '>' + str(L)} pasos")
    return out


# ------------------------------------------------------------ la copia selectiva
def selective_copy_task(length, n_items=4, vocab=10, n=3000, seed=SEED):
    """Copiar SOLO los tokens marcados, que es lo que un SSM invariante no puede.

    En la copia normal, la posicion de lo que hay que recordar es siempre la
    misma, asi que un modelo con dinamicas fijas puede aprender "guarda el
    primero". Aqui los tokens a recordar estan repartidos al azar por la
    secuencia y solo se distinguen porque llevan una marca, asi que decidir si
    guardar depende de lo que se esta leyendo. Esa es exactamente la diferencia
    entre un sistema invariante en el tiempo y uno selectivo, y es la razon de
    que la tarea exista.
    """
    rng = np.random.default_rng(seed + length)
    x = np.zeros((n, length + n_items + 1), dtype=np.int64)
    y = np.zeros((n, n_items), dtype=np.int64)
    for i in range(n):
        seq = rng.integers(1, vocab + 1, size=length)
        pos = np.sort(rng.choice(length, size=n_items, replace=False))
        marks = np.zeros(length, dtype=np.int64)
        marks[pos] = 1
        # el token marcado se codifica desplazado, que es la "marca"
        row = seq + marks * vocab
        x[i, :length] = row
        x[i, length] = 2 * vocab + 1
        y[i] = seq[pos] - 1
    return x, y


def train_ssm(kind, x, y, epochs=40, d=64, state=32, seed=SEED, lr=3e-3):
    """Un SSM diagonal, invariante o selectivo, con el mismo presupuesto.

    El invariante tiene un decaimiento por canal y ya esta. El selectivo hace
    que el paso de discretizacion dependa de la entrada, que es la unica idea
    de Mamba que cambia lo que el modelo PUEDE hacer (las otras dos cambian lo
    rapido que va). Se implementa el escaneo explicitamente, sin trucos, porque
    la pagina va a ejecutar el mismo escaneo y tiene que poder comparar.
    """
    import torch
    import torch.nn as nn
    threads()
    torch.manual_seed(seed)
    V = int(x.max()) + 1
    K = int(y.max()) + 1
    n_out = y.shape[1]
    xt = torch.tensor(x)
    yt = torch.tensor(y)
    ntr = int(0.8 * len(x))

    class SSM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, d)
            # log del decaimiento, repartido en escala logaritmica: es la
            # inicializacion que el articulo mide como la que alcanza lejos.
            self.log_dt = nn.Parameter(
                torch.linspace(math.log(1e-3), math.log(1e-1), state).repeat(d, 1))
            self.B = nn.Parameter(torch.randn(d, state) / math.sqrt(state))
            self.C = nn.Parameter(torch.randn(d, state) / math.sqrt(state))
            self.D = nn.Parameter(torch.ones(d))
            self.selective = (kind == "selective")
            if self.selective:
                self.to_dt = nn.Linear(d, state)
                self.to_B = nn.Linear(d, state)
                self.to_C = nn.Linear(d, state)
            self.head = nn.Linear(d, K * n_out)

        def forward(self, ids):
            u = self.emb(ids)
            B_, L, _ = u.shape
            h = torch.zeros(B_, d, state)
            outs = []
            base_dt = torch.exp(self.log_dt)
            for t in range(L):
                ut = u[:, t]
                if self.selective:
                    dt = torch.nn.functional.softplus(self.to_dt(ut))[:, None, :] * base_dt
                    Bt = self.B[None] + self.to_B(ut)[:, None, :]
                    Ct = self.C[None] + self.to_C(ut)[:, None, :]
                else:
                    dt = base_dt[None]
                    Bt = self.B[None]
                    Ct = self.C[None]
                a = torch.exp(-dt)
                h = a * h + Bt * ut[:, :, None]
                outs.append((Ct * h).sum(-1) + self.D * ut)
            o = torch.stack(outs, 1)
            return self.head(o[:, -1]).view(B_, n_out, K)

    m = SSM()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Bsz = 128
    for ep in range(epochs):
        perm = torch.randperm(ntr)
        for s in range(0, ntr, Bsz):
            b = perm[s:s + Bsz]
            logits = m(xt[b])
            loss = lossf(logits.reshape(-1, K), yt[b].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    with torch.no_grad():
        p = m(xt[ntr:]).argmax(-1)
        acc = float((p == yt[ntr:]).float().mean())
        exact = float((p == yt[ntr:]).all(-1).float().mean())
    return {"acc": acc, "exact": exact, "chance": 1.0 / K,
            "params": sum(q.numel() for q in m.parameters())}


def copy_curve_ssm(distances, epochs=30, seed=SEED):
    """La copia normal de los dos articulos anteriores, con la misma funcion."""
    import torch
    import torch.nn as nn
    threads()
    rows = []
    for dist in distances:
        x, y = copy_task(dist, seed=seed)
        y2 = y.reshape(-1, 1)
        r = train_ssm("invariant", x, y2, epochs=epochs, seed=seed)
        rows.append({"distance": dist, **r})
        print(f"      SSM a distancia {dist:>3}: {r['acc']:.4f}")
    return rows


def cost_table(lengths=(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096), d=96, state=32):
    """Multiplicaciones por secuencia, contadas, para las tres familias.

    La comparacion que todo el mundo cita ("atencion es cuadratica") es cierta y
    esconde la constante: para longitudes cortas la parte cuadratica ni siquiera
    domina, y el cruce esta en un sitio concreto que se puede calcular. Aqui se
    calcula.
    """
    rows = []
    for L in lengths:
        proj = 4 * L * d * d
        pair = 2 * L * L * d
        rnn = L * 2 * d * d
        ssm = L * (2 * d * state + d * d)
        rows.append({"length": L, "projections": proj, "pairwise": pair,
                     "attention": proj + pair, "recurrent": rnn, "ssm": ssm,
                     "attention_over_ssm": (proj + pair) / ssm,
                     "pairwise_share": pair / (proj + pair)})
    # La cifra que de verdad significa algo NO es "a partir de que longitud la
    # atencion cuesta el doble", porque eso lo decide la constante de las
    # proyecciones y sale ridiculamente pronto. La que significa algo es a
    # partir de donde el termino CUADRATICO supera al lineal de la propia
    # atencion, que es donde "cuadratico" empieza a describir la factura: sale
    # de 2 L^2 d > 4 L d^2, es decir L > 2d, y no depende de nada mas.
    quad = next((r["length"] for r in rows if r["pairwise"] > r["projections"]), None)
    return {"rows": rows, "d_model": d, "state": state,
            "quadratic_takes_over_at": quad, "predicted": 2 * d}


def ship(seed=SEED, n=24, L=160):
    """Los parametros de un SSM concreto, para que la pagina ejecute las dos formas.

    Se envian a, b y c en float16 y el navegador calcula el nucleo y la
    recurrencia por su cuenta, que es lo que le permite comprobar la identidad
    delante del lector en vez de mostrarle el resultado de que alguien la
    comprobara.
    """
    rng = np.random.default_rng(seed)
    a = 0.97 ** (1 + rng.random(n))
    b = rng.normal(size=n) / math.sqrt(n)
    c = rng.normal(size=n) / math.sqrt(n)
    x = rng.normal(size=L)
    return {
        "n": n, "L": L,
        "a": b64_f16(a), "b": b64_f16(b), "c": b64_f16(c), "x": b64_f16(x),
        "kernel_ref": [float(v) for v in ssm_kernel(a.astype(np.float16).astype(float),
                                                    b.astype(np.float16).astype(float),
                                                    c.astype(np.float16).astype(float), 64)],
        "y_ref": [float(v) for v in ssm_recurrent(a.astype(np.float16).astype(float),
                                                  b.astype(np.float16).astype(float),
                                                  c.astype(np.float16).astype(float),
                                                  x.astype(np.float16).astype(float))],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("la recurrencia y la convolucion son el mismo calculo")
    dual = cached("duality", duality_check)

    print("el alcance del nucleo segun la inicializacion")
    reach = cached("reach", kernel_reach)

    print("la copia normal, con la funcion de los articulos anteriores")
    dists = [5, 10, 20, 40, 80]
    cop = cached("copy", lambda: copy_curve_ssm(dists))

    print("la copia SELECTIVA, que es donde invariante y selectivo se separan")
    # Dos longitudes y no tres, y el motivo se publica en la pagina en vez de
    # dejar la tabla corta sin explicacion. El escaneo esta escrito paso a paso
    # a proposito (la pagina ejecuta el mismo), asi que el grafo de autograd
    # tiene un nodo por paso y por operacion: a longitud 80 un solo modelo pasa
    # de la hora en esta maquina, y hacen falta dos por fila. Con 20 y 40 la
    # comparacion ya esta hecha a presupuesto identico y la direccion se lee,
    # que es lo que la tabla tiene que sostener.
    SEL_LENGTHS = (20, 40)
    sel = []
    for L in SEL_LENGTHS:
        x, y = selective_copy_task(L)
        row = {"length": L}
        for kind in ("invariant", "selective"):
            r = cached(f"sel_{kind}_{L}", lambda k=kind, xx=x, yy=y: train_ssm(k, xx, yy))
            row[kind] = r
            print(f"    longitud {L:>3} {kind:<10} {r['acc']:.4f} por token, "
                  f"{r['exact']:.4f} secuencias enteras")
        sel.append(row)

    print("el coste, contado")
    cost = cost_table()

    data = {
        "meta": {
            "seed": SEED,
            "state": 32,
            "note": "the copy task is imported from the sequences article, same seeds",
        },
        "duality": dual,
        "reach": reach,
        "copy": cop,
        "copy_distances": dists,
        "selective": sel,
        "cost": cost,
        "demo": ship(),
    }
    path = OUT / "statespace.json"
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
