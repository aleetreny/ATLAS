"""Las cifras del artículo de LoRA y QLoRA.

LoRA descansa sobre una afirmación empírica que casi siempre se cita y casi
nunca se dibuja: **lo que el ajuste fino le hace a una matriz de pesos tiene
rango bajo**. Eso no es una propiedad de los transformadores, es una propiedad
de una diferencia de matrices, así que se puede medir en cualquier sitio donde
haya una matriz que se ajusta, y aquí se mide sobre las mismas redes y las
mismas dos tablas que usa el artículo de destilación: un perceptrón entrenado
en MNIST y ajustado a Fashion-MNIST.

Y se mide **con su control**, que es lo que decide si la afirmación dice algo:
una matriz aleatoria del mismo tamaño y la misma norma tiene un espectro
conocido, así que "el 90% de la energía en r direcciones" solo significa algo
comparado con lo que sale de no haber aprendido nada.

Lo que se mide:

  1. **El espectro de la diferencia**, contra el de los pesos de partida y
     contra el de una matriz aleatoria de la misma norma. Y el rango efectivo
     de los tres, que es una cifra y no una impresión.
  2. **El techo de un adaptador de rango r**: truncar la diferencia real a
     rango r por descomposición en valores singulares y aplicarla. Ningún LoRA
     puede hacerlo mejor, porque eso es la mejor aproximación de rango r que
     existe. Con eso al lado, lo que LoRA alcanza deja de ser una cifra suelta.
  3. **LoRA contra las alternativas a igual número de parámetros entrenables**:
     la cabeza sola, los sesgos solos (BitFit) y el ajuste fino entero.
  4. **alpha**, que la gente ajusta como si fuera capacidad y es un factor de
     tasa de aprendizaje. Se barre con la tasa fija y con la tasa compensada; si
     la explicación es la correcta, la segunda curva es plana.
  5. **La cuantización a cuatro bits, con su supuesto medido.** NF4 se justifica
     porque los pesos son aproximadamente normales, y eso es comprobable capa
     por capa. Los tres formatos sobre los pesos de verdad, el tamaño de bloque
     barrido, la doble cuantización contada en bits, y al final lo único que
     importa: entrenar el adaptador encima de la base cuantizada.
  6. **La factura de memoria**, en bytes exactos y derivada del recuento de
     parámetros de estas redes, no de una tabla de un artículo.

Ejecutar desde cualquier sitio: `python src/utils/generate_lora_data.py`.
"""
import os

# Un lote de 32 filas contra una matriz de 784 por 256 es una multiplicación
# diminuta, y repartirla entre cuatro hilos cuesta más que hacerla: medido en
# esta máquina, cinco épocas tardan 1,7 s con un hilo y 15,2 s con cuatro, un
# factor de 9 en la dirección contraria a la que uno espera. Se fija antes de
# importar numpy, que es cuando el BLAS lee estas variables, y el número va en
# el TAG porque el reparto de hilos cambia el orden de las reducciones en coma
# flotante y por tanto los últimos decimales.
THREADS = 1
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tinynet import (FP4, INT4, NF4, LoraNet, Net, bits_per_param,  # noqa: E402
                     onehot, quantize, train, train_lora)
from vision_data import fashion_mnist, mnist  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "lora" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "lora_cache")

SEED = 19
HID = (256, 128)
SRC_N = 12000
TGT_N = 5000
TEST_N = 5000
SRC_EPOCHS = 12
FT_EPOCHS = 25
RANKS = (1, 2, 4, 8, 16, 32)
ALPHAS = (2, 4, 8, 16, 32)
ALPHA_RANK = 8
SEEDS = 2
DATA_SWEEP = (250, 1000, 5000, 12000)
BLOCKS = (16, 32, 64, 128, 256, 1024)

TAG = (f"h{'-'.join(map(str, HID))}s{SRC_N}t{TGT_N}e{SRC_EPOCHS}f{FT_EPOCHS}"
       f"r{max(RANKS)}n{SEEDS}v{SEED}th{THREADS}")


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
    if isinstance(v, dict):
        return {k: r(x, d) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [r(x, d) for x in v]
    if v is None or isinstance(v, (bool, str)):
        return v
    v = float(v)
    return None if not np.isfinite(v) else round(v, d)


def flat(X):
    return X.reshape(len(X), -1).astype(np.float64) / 255.0


def stratified(y, n, rng):
    classes = np.unique(y)
    per = n // len(classes)
    take = []
    for c in classes:
        idx = np.where(y == c)[0]
        take.extend(rng.choice(idx, min(per, len(idx)), replace=False).tolist())
    return np.array(sorted(take))


def load():
    a, b = mnist(), fashion_mnist()
    return (flat(a[0]), a[1]), (flat(b[0]), b[1], flat(b[2])[:TEST_N], b[3][:TEST_N])


def stage_base(src):
    X, y = src
    rng = np.random.default_rng(SEED)
    idx = stratified(y, SRC_N, rng)
    net = Net([X.shape[1], *HID, 10], np.random.default_rng(SEED))
    train(net, X[idx], onehot(y[idx], 10), np.random.default_rng(SEED + 1),
          epochs=SRC_EPOCHS, batch=128, lr=1e-3)
    return net


def full_finetune(base, Xg, yg, idx, seed, epochs=FT_EPOCHS):
    net = base.copy()
    net.W[-1] = np.zeros_like(net.W[-1])
    net.b[-1] = np.zeros_like(net.b[-1])
    train(net, Xg[idx], onehot(yg[idx], 10), np.random.default_rng(seed),
          epochs=epochs, batch=32, lr=3e-4)
    return net


# --------------------------------------------------------------- 1. el espectro
def eff_rank(s):
    """Rango efectivo: la exponencial de la entropía del espectro normalizado.

    Un número que se puede comparar entre matrices de tamaños distintos, y que
    no depende de dónde ponga uno el corte del 90%.
    """
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def spectrum(M):
    s = np.linalg.svd(M, compute_uv=False)
    e = (s ** 2).cumsum() / (s ** 2).sum()
    return dict(s=s.tolist(), energy=e.tolist(), eff=eff_rank(s),
                r90=int(np.searchsorted(e, 0.90) + 1),
                r50=int(np.searchsorted(e, 0.50) + 1),
                shape=list(M.shape), fro=float(np.linalg.norm(M)))


def stage_spectra(base, tgt):
    Xg, yg, Xt, yt = tgt
    idx = stratified(yg, TGT_N, np.random.default_rng(SEED))
    tuned = full_finetune(base, Xg, yg, idx, SEED)
    rng = np.random.default_rng(SEED + 3)
    layers = []
    for i in range(len(base.W) - 1):     # la cabeza se reinicia, no se ajusta
        D = tuned.W[i] - base.W[i]
        R = rng.standard_normal(D.shape)
        R *= np.linalg.norm(D) / np.linalg.norm(R)
        layers.append(dict(layer=i, delta=spectrum(D), base=spectrum(base.W[i]),
                           random=spectrum(R),
                           rel=float(np.linalg.norm(D) / np.linalg.norm(base.W[i]))))
    return dict(layers=layers, tuned_acc=tuned.accuracy(Xt, yt),
                base_acc=base.accuracy(Xt, yt), tuned=tuned, idx=idx)


def stage_data_sweep(base, tgt):
    """¿Sube el rango del cambio con la cantidad de datos de ajuste?"""
    Xg, yg, Xt, yt = tgt
    rows = []
    for n in DATA_SWEEP:
        idx = stratified(yg, n, np.random.default_rng(SEED + n))
        t = full_finetune(base, Xg, yg, idx, SEED)
        D = t.W[0] - base.W[0]
        sp = spectrum(D)
        rows.append(dict(n=n, r90=sp["r90"], eff=sp["eff"], fro=sp["fro"],
                         acc=t.accuracy(Xt, yt)))
        print(f"    ajuste con {n} filas: r90 = {sp['r90']}, rango efectivo "
              f"{sp['eff']:.1f}")
    return rows


# --------------------------------------------------- 2 y 3. el techo y los brazos
def truncate_apply(base, tuned, tgt, layer=0):
    """El mejor adaptador posible de cada rango: la propia diferencia truncada."""
    Xt, yt = tgt[2], tgt[3]
    D = tuned.W[layer] - base.W[layer]
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    rows = []
    for k in (0,) + RANKS:
        net = tuned.copy()          # el resto de capas, ya ajustado
        net.W[layer] = base.W[layer] + (U[:, :k] * S[:k]) @ Vt[:k]
        rows.append(dict(rank=k, acc=net.accuracy(Xt, yt),
                         kept=float((S[:k] ** 2).sum() / (S ** 2).sum())))
    return rows


def stage_arms(base, tgt):
    Xg, yg, Xt, yt = tgt
    idx = stratified(yg, TGT_N, np.random.default_rng(SEED))
    Y = onehot(yg[idx], 10)
    rows = []

    def head_reset(net):
        net.W[-1] = np.zeros_like(net.W[-1])
        net.b[-1] = np.zeros_like(net.b[-1])

    for rank in RANKS:
        accs, n_tr = [], None
        for s in range(SEEDS):
            b = base.copy()
            head_reset(b)
            lo = LoraNet(b, [0, 1], rank, np.random.default_rng(SEED + s),
                         extra=[len(b.W) - 1])
            train_lora(lo, Xg[idx], Y, np.random.default_rng(SEED + s),
                       epochs=FT_EPOCHS, batch=32, lr=3e-4)
            accs.append(lo.accuracy(Xt, yt))
            n_tr = lo.n_trainable()
        rows.append(dict(arm=f"lora r={rank}", rank=rank, trainable=n_tr,
                         mean=float(np.mean(accs)), sd=float(np.std(accs))))
        print(f"    lora rango {rank}: {np.mean(accs):.4f} con {n_tr} parámetros")

    def plain(tag, **kw):
        accs = []
        for s in range(SEEDS):
            net = base.copy()
            head_reset(net)
            train(net, Xg[idx], Y, np.random.default_rng(SEED + s),
                  epochs=FT_EPOCHS, batch=32, lr=3e-4, **kw)
            accs.append(net.accuracy(Xt, yt))
        return accs

    last = len(base.W) - 1
    accs = plain("head", trainable=[last], trainable_bias=[last])
    n_head = int(base.W[last].size + base.b[last].size)
    rows.append(dict(arm="head only", rank=None, trainable=n_head,
                     mean=float(np.mean(accs)), sd=float(np.std(accs))))

    accs = plain("bitfit", trainable=[last],
                 trainable_bias=list(range(len(base.W))))
    n_bit = int(sum(b.size for b in base.b) + base.W[last].size)
    rows.append(dict(arm="biases plus head", rank=None, trainable=n_bit,
                     mean=float(np.mean(accs)), sd=float(np.std(accs))))

    accs = plain("full")
    rows.append(dict(arm="everything", rank=None, trainable=int(base.n_params()),
                     mean=float(np.mean(accs)), sd=float(np.std(accs))))
    for row in rows[-3:]:
        print(f"    {row['arm']}: {row['mean']:.4f} con {row['trainable']} parámetros")
    return rows


def stage_where(base, tgt, rank=8):
    """A qué capa ponerle el adaptador, a igual rango."""
    Xg, yg, Xt, yt = tgt
    idx = stratified(yg, TGT_N, np.random.default_rng(SEED))
    Y = onehot(yg[idx], 10)
    rows = []
    for tag, layers in (("first", [0]), ("second", [1]), ("both", [0, 1])):
        accs, n_tr = [], None
        for s in range(SEEDS):
            b = base.copy()
            b.W[-1] = np.zeros_like(b.W[-1])
            b.b[-1] = np.zeros_like(b.b[-1])
            lo = LoraNet(b, layers, rank, np.random.default_rng(SEED + s),
                         extra=[len(b.W) - 1])
            train_lora(lo, Xg[idx], Y, np.random.default_rng(SEED + s),
                       epochs=FT_EPOCHS, batch=32, lr=3e-4)
            accs.append(lo.accuracy(Xt, yt))
            n_tr = lo.n_trainable()
        rows.append(dict(arm=tag, layers=layers, trainable=n_tr,
                         mean=float(np.mean(accs)), sd=float(np.std(accs))))
    return dict(rank=rank, rows=rows)


def stage_alpha(base, tgt):
    """alpha con la tasa fija y con la tasa compensada."""
    Xg, yg, Xt, yt = tgt
    idx = stratified(yg, TGT_N, np.random.default_rng(SEED))
    Y = onehot(yg[idx], 10)
    rows = []
    for a in ALPHAS:
        for tag, lr in (("fixed lr", 3e-4), ("compensated lr", 3e-4 * ALPHA_RANK / a)):
            accs = []
            for s in range(SEEDS):
                b = base.copy()
                b.W[-1] = np.zeros_like(b.W[-1])
                b.b[-1] = np.zeros_like(b.b[-1])
                lo = LoraNet(b, [0, 1], ALPHA_RANK, np.random.default_rng(SEED + s),
                             alpha=a, extra=[len(b.W) - 1])
                train_lora(lo, Xg[idx], Y, np.random.default_rng(SEED + s),
                           epochs=FT_EPOCHS, batch=32, lr=lr)
                accs.append(lo.accuracy(Xt, yt))
            rows.append(dict(alpha=a, arm=tag, mean=float(np.mean(accs)),
                             sd=float(np.std(accs))))
        print(f"    alpha {a} hecho")
    return dict(rank=ALPHA_RANK, rows=rows)


# ------------------------------------------------------------------- 5. QLoRA
def normality(w):
    """Cuánto se parece a una normal la distribución de una capa.

    Dos cifras baratas y honestas: la curtosis (3 en una normal) y el peor
    hueco entre la función de distribución empírica y la de la normal con la
    misma media y desviación, que es el estadístico de Kolmogorov y Smirnov
    calculado sin librería.
    """
    from math import erf, sqrt
    x = np.sort(w.ravel())
    m, sd = x.mean(), x.std()
    z = (x - m) / sd
    cdf = np.array([0.5 * (1 + erf(v / sqrt(2))) for v in z])
    emp = (np.arange(1, len(x) + 1)) / len(x)
    ks = float(np.max(np.abs(cdf - emp)))
    kurt = float(((z ** 4).mean()))
    return dict(kurtosis=kurt, ks=ks, sd=float(sd), mean=float(m))


def stage_quant(base, tgt):
    Xg, yg, Xt, yt = tgt
    rows, blocks = [], []
    for i, W in enumerate(base.W):
        rec = dict(layer=i, shape=list(W.shape), n=int(W.size), **normality(W))
        for tag, lv in (("nf4", NF4), ("int4", INT4), ("fp4", FP4)):
            q, _, _ = quantize(W, lv, block=64)
            rec[tag] = float(np.linalg.norm(q - W) / np.linalg.norm(W))
            rec[tag + "_levels"] = int(len(lv))
        rows.append(rec)
    W = base.W[0]
    for b in BLOCKS:
        q, _, _ = quantize(W, NF4, block=b)
        blocks.append(dict(block=b,
                           err=float(np.linalg.norm(q - W) / np.linalg.norm(W)),
                           bits=bits_per_param(W.size, block=b),
                           bits_double=bits_per_param(W.size, block=b, double=True)))
    # la red entera cuantizada, y lo que le pasa a su exactitud
    qnet = base.copy()
    for i in range(len(qnet.W)):
        qnet.W[i], _, _ = quantize(base.W[i], NF4, block=64)
    return dict(rows=rows, blocks=blocks,
                base_acc=base.accuracy(Xt, yt), quant_acc=qnet.accuracy(Xt, yt),
                qnet=qnet)


def stage_qlora(base, qnet, tgt):
    Xg, yg, Xt, yt = tgt
    idx = stratified(yg, TGT_N, np.random.default_rng(SEED))
    Y = onehot(yg[idx], 10)
    rows = []
    for tag, b0 in (("float base", base), ("nf4 base", qnet)):
        for rank in (4, 8, 16):
            accs = []
            for s in range(SEEDS):
                b = b0.copy()
                b.W[-1] = np.zeros_like(b.W[-1])
                b.b[-1] = np.zeros_like(b.b[-1])
                lo = LoraNet(b, [0, 1], rank, np.random.default_rng(SEED + s),
                             extra=[len(b.W) - 1])
                train_lora(lo, Xg[idx], Y, np.random.default_rng(SEED + s),
                           epochs=FT_EPOCHS, batch=32, lr=3e-4)
                accs.append(lo.accuracy(Xt, yt))
            rows.append(dict(arm=tag, rank=rank, mean=float(np.mean(accs)),
                             sd=float(np.std(accs))))
            print(f"    {tag} rango {rank}: {np.mean(accs):.4f}")
    return rows


def memory(base, rank):
    """Los bytes de cada receta, contados sobre estas redes.

    Con Adam en float32, entrenar un peso cuesta cuatro copias suyas: el peso,
    su gradiente y los dos momentos. Eso es lo que hace que congelar la base no
    ahorre un cuarto sino casi todo.
    """
    n = base.n_params()
    n_lora = sum(base.W[i].shape[0] * rank + rank * base.W[i].shape[1]
                 for i in (0, 1)) + base.W[-1].size + base.b[-1].size
    out = []
    out.append(dict(arm="full fine tune", weights=4 * n, grads=4 * n,
                    optimiser=8 * n, trainable=int(n)))
    out.append(dict(arm="lora", weights=4 * n, grads=4 * n_lora,
                    optimiser=8 * n_lora, trainable=int(n_lora)))
    out.append(dict(arm="qlora", weights=int(np.ceil(bits_per_param(n, double=True)
                                                     * n / 8)),
                    grads=4 * n_lora, optimiser=8 * n_lora, trainable=int(n_lora)))
    for row in out:
        row["total"] = int(row["weights"] + row["grads"] + row["optimiser"])
    return dict(rows=out, params=int(n), rank=rank)


def main():
    src, tgt = load()
    print(f"  origen {src[0].shape}, destino {tgt[0].shape}, test {tgt[2].shape}")

    print("  1/7 la red de partida")
    base = cached("base", lambda: stage_base(src))

    print("  2/7 los espectros")
    sp = cached("spectra", lambda: stage_spectra(base, tgt))

    print("  3/7 el rango contra cuántos datos de ajuste")
    sweep = cached("data-sweep", lambda: stage_data_sweep(base, tgt))

    print("  4/7 el techo de cada rango")
    ceiling = cached("ceiling", lambda: truncate_apply(base, sp["tuned"], tgt))

    print("  5/7 los brazos a igual presupuesto")
    arms = cached("arms", lambda: stage_arms(base, tgt))
    where = cached("where", lambda: stage_where(base, tgt))
    alpha = cached("alpha", lambda: stage_alpha(base, tgt))

    print("  6/7 la cuantización")
    quant = cached("quant", lambda: stage_quant(base, tgt))

    print("  7/7 el adaptador sobre la base cuantizada")
    qlora = cached("qlora", lambda: stage_qlora(base, quant["qnet"], tgt))

    OUT.mkdir(parents=True, exist_ok=True)
    layers = []
    for L in sp["layers"]:
        layers.append(dict(
            layer=L["layer"], shape=L["delta"]["shape"], rel=L["rel"],
            delta=dict(eff=L["delta"]["eff"], r90=L["delta"]["r90"],
                       r50=L["delta"]["r50"], fro=L["delta"]["fro"],
                       energy=L["delta"]["energy"][:64], s=L["delta"]["s"][:64]),
            base=dict(eff=L["base"]["eff"], r90=L["base"]["r90"],
                      energy=L["base"]["energy"][:64], s=L["base"]["s"][:64]),
            random=dict(eff=L["random"]["eff"], r90=L["random"]["r90"],
                        energy=L["random"]["energy"][:64], s=L["random"]["s"][:64]),
        ))
    out = dict(
        meta=dict(seed=SEED, hidden=list(HID), source_n=SRC_N, target_n=TGT_N,
                  test_n=TEST_N, epochs=FT_EPOCHS, seeds=SEEDS,
                  ranks=list(RANKS), alphas=list(ALPHAS), alpha_rank=ALPHA_RANK,
                  params=int(base.n_params()),
                  note="a plain multilayer perceptron pretrained on MNIST and "
                       "adapted to Fashion-MNIST; every number is accuracy on "
                       "the target test split"),
        spectra=dict(layers=r(layers), base_acc=r(sp["base_acc"]),
                     tuned_acc=r(sp["tuned_acc"])),
        data_sweep=r(sweep),
        ceiling=r(ceiling),
        arms=r(arms),
        where=r(where),
        alpha=r(alpha),
        quant=dict(rows=r(quant["rows"]), blocks=r(quant["blocks"]),
                   base_acc=r(quant["base_acc"]), quant_acc=r(quant["quant_acc"]),
                   nf4=r(NF4.tolist()), int4=r(INT4.tolist()), fp4=r(FP4.tolist())),
        qlora=r(qlora),
        memory=r(memory(base, 8)),
    )
    path = OUT / "lora.json"
    path.write_text(json.dumps(out, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
